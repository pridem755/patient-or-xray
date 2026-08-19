from __future__ import annotations

import io
import zipfile

import pytest

pytest.importorskip("torch", reason="training tests need torch; CI runs the rest")
pytest.importorskip("torchvision")

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pxr.data.cache import build_cache
from pxr.model.train import (
    CachedImageDataset,
    TrainConfig,
    TrainError,
    build_model,
    macro_auroc,
    masked_bce_loss,
    positive_weights,
    predict,
    train_fold,
)

LABELS = ["Cardiomegaly", "Edema", "Pneumothorax"]


@pytest.fixture
def cache(tmp_path):
    """A small cache of synthetic radiographs."""
    rng = np.random.default_rng(0)
    names = [f"img{i}.png" for i in range(60)]
    with zipfile.ZipFile(tmp_path / "a.zip", "w") as z:
        for name in names:
            buffer = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (224, 224), dtype=np.uint8),
                            mode="L").save(buffer, format="PNG")
            z.writestr(name, buffer.getvalue())
    return build_cache(tmp_path / "a.zip", [n[:-4] for n in names], tmp_path / "cache")


@pytest.fixture
def frame():
    rng = np.random.default_rng(1)
    data = {
        "patient_id": [f"p{i}" for i in range(60)],
        "image_id": [f"img{i}" for i in range(60)],
    }
    for label in LABELS:
        values = rng.binomial(1, 0.35, 60).astype(float)
        values[rng.random(60) < 0.15] = np.nan      # unrecorded labels
        data[label] = values
    return pd.DataFrame(data)


@pytest.fixture
def config():
    return TrainConfig(labels=LABELS, batch_size=8, max_epochs=2, patience=1,
                       num_workers=0, mixed_precision=False)


class TestConfig:
    def test_horizontal_flip_is_refused(self):
        """Flipping reverses laterality - the study's exposure is acquisition geometry."""
        with pytest.raises(TrainError, match="laterality"):
            TrainConfig(labels=LABELS, horizontal_flip=True)

    def test_empty_label_set_refused(self):
        with pytest.raises(TrainError, match="no labels"):
            TrainConfig(labels=[])


class TestMaskedLoss:
    def test_missing_labels_contribute_no_gradient(self):
        """Treating 'not mentioned' as negative is the assumption the cohort refused."""
        import torch

        logits = torch.zeros(4, 3, requires_grad=True)
        targets = torch.full((4, 3), float("nan"))
        targets[0, 0] = 1.0
        loss = masked_bce_loss(logits, targets)
        loss.backward()
        assert logits.grad[0, 0] != 0
        assert torch.allclose(logits.grad[1:], torch.zeros(3, 3))

    def test_loss_averages_over_observed_entries_only(self):
        import torch

        logits = torch.zeros(2, 2)
        dense = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        sparse = torch.tensor([[1.0, float("nan")], [float("nan"), float("nan")]])
        assert abs(float(masked_bce_loss(logits, dense))
                   - float(masked_bce_loss(logits, sparse))) < 1e-6

    def test_all_missing_batch_returns_zero(self):
        import torch

        logits = torch.zeros(3, 3, requires_grad=True)
        loss = masked_bce_loss(logits, torch.full((3, 3), float("nan")))
        assert float(loss) == 0.0

    def test_positive_weight_raises_the_cost_of_missed_positives(self):
        import torch

        logits = torch.full((8, 1), -3.0)         
        targets = torch.ones(8, 1)                  
        plain = float(masked_bce_loss(logits, targets))
        weighted = float(masked_bce_loss(logits, targets, torch.tensor([5.0])))
        assert weighted > plain


class TestPositiveWeights:
    def test_rare_labels_get_larger_weights(self):
        labels = np.zeros((1000, 2))
        labels[:500, 0] = 1        # 50% prevalence
        labels[:10, 1] = 1         # 1% prevalence
        weights = positive_weights(labels)
        assert weights[1] > weights[0]

    def test_weights_are_capped(self):
        """An unbounded weight on a 0.2% label swamps every other label's gradient."""
        labels = np.zeros((10000, 1))
        labels[:5, 0] = 1
        assert positive_weights(labels, cap=20.0)[0] == 20.0

    def test_missing_entries_are_excluded_from_the_counts(self):
        labels = np.full((100, 1), np.nan)
        labels[:20, 0] = 1
        labels[20:40, 0] = 0
        assert positive_weights(labels)[0] == pytest.approx(1.0)

    def test_label_without_positives_keeps_weight_one(self):
        assert positive_weights(np.zeros((50, 1)))[0] == 1.0


class TestMacroAuroc:
    def test_perfect_ranking_scores_one(self):
        truth = np.array([[0.0], [0.0], [1.0], [1.0]])
        score = np.array([[0.1], [0.2], [0.8], [0.9]])
        macro, _ = macro_auroc(truth, score)
        assert macro == pytest.approx(1.0)

    def test_single_class_labels_are_skipped_not_scored_as_zero(self):
        """A fold missing a rare label's positives should be visible, not averaged in."""
        truth = np.array([[0.0, 1.0], [1.0, 1.0]])
        score = np.array([[0.2, 0.5], [0.8, 0.6]])
        macro, per_label = macro_auroc(truth, score)
        assert np.isnan(per_label[1])
        assert macro == pytest.approx(1.0)

    def test_missing_targets_are_excluded(self):
        truth = np.array([[0.0], [1.0], [np.nan]])
        score = np.array([[0.1], [0.9], [0.5]])
        macro, _ = macro_auroc(truth, score)
        assert macro == pytest.approx(1.0)


class TestDataset:
    def test_returns_three_channel_normalised_images(self, frame, cache, config):
        dataset = CachedImageDataset(frame, cache, LABELS, train=False, config=config)
        image, targets = dataset[0]
        assert image.shape == (3, 224, 224)
        assert targets.shape == (3,)

    def test_evaluation_is_deterministic(self, frame, cache, config):
        """Validation and test metrics must not move because of augmentation."""
        import torch

        dataset = CachedImageDataset(frame, cache, LABELS, train=False, config=config)
        assert torch.equal(dataset[0][0], dataset[0][0])

    def test_training_augmentation_perturbs_the_image(self, frame, cache, config):
        import torch

        dataset = CachedImageDataset(frame, cache, LABELS, train=True, config=config)
        torch.manual_seed(0)
        first = dataset[0][0]
        torch.manual_seed(1)
        assert not torch.equal(first, dataset[0][0])

    def test_missing_labels_survive_as_nan(self, frame, cache, config):
        import torch

        frame = frame.copy()
        frame.loc[0, "Cardiomegaly"] = np.nan
        dataset = CachedImageDataset(frame, cache, LABELS, train=False, config=config)
        assert torch.isnan(dataset[0][1][0])


class TestModel:
    def test_head_matches_the_label_count(self):
        import torch

        model = build_model("densenet121", 8, pretrained=False)
        assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 8)

    def test_unsupported_architecture_raises(self):
        with pytest.raises(TrainError, match="unsupported architecture"):
            build_model("inception_v99", 3)


class TestTrainFold:
    def test_overlapping_patients_raise(self, frame, cache, config):
        """Validation chooses the stopping point; sharing patients would leak it."""
        with pytest.raises(TrainError, match="both train and validation"):
            train_fold(frame, frame, cache, config, site="test", fold=0)

    def test_training_runs_and_records_history(self, frame, cache, config, monkeypatch):
        monkeypatch.setattr("pxr.model.train.build_model",
                            lambda a, n, pretrained=True: build_model(a, n, pretrained=False))
        model, history = train_fold(frame.iloc[:40], frame.iloc[40:], cache, config,
                                    site="test", fold=0, progress=lambda _: None)
        assert len(history.epochs) >= 1
        assert history.best_epoch >= 0
        assert not np.isnan(history.best_val_macro_auroc)

    def test_checkpoint_and_history_are_written(self, frame, cache, config, tmp_path,
                                                monkeypatch):
        monkeypatch.setattr("pxr.model.train.build_model",
                            lambda a, n, pretrained=True: build_model(a, n, pretrained=False))
        train_fold(frame.iloc[:40], frame.iloc[40:], cache, config, site="test", fold=2,
                   checkpoint_dir=tmp_path / "ckpt", progress=lambda _: None)
        assert (tmp_path / "ckpt" / "test_fold2.pt").exists()
        assert (tmp_path / "ckpt" / "test_fold2_history.csv").exists()
        assert (tmp_path / "ckpt" / "test_fold2_config.json").exists()

    def test_returns_the_best_epoch_not_the_last(self, frame, cache, config, monkeypatch):
        monkeypatch.setattr("pxr.model.train.build_model",
                            lambda a, n, pretrained=True: build_model(a, n, pretrained=False))
        _, history = train_fold(frame.iloc[:40], frame.iloc[40:], cache, config,
                                site="test", fold=0, progress=lambda _: None)
        scores = [e.val_macro_auroc for e in history.epochs]
        assert history.best_val_macro_auroc == pytest.approx(max(scores))


class TestPredict:
    def test_one_row_per_patient_with_a_column_per_label(self, frame, cache, config):
        model = build_model("densenet121", len(LABELS), pretrained=False)
        out = predict(model, frame.iloc[:16], cache, config)
        assert len(out) == 16
        assert list(out.columns) == ["patient_id", "image_id", *LABELS]

    def test_scores_are_probabilities(self, frame, cache, config):
        model = build_model("densenet121", len(LABELS), pretrained=False)
        out = predict(model, frame.iloc[:16], cache, config)
        assert ((out[LABELS] >= 0) & (out[LABELS] <= 1)).all().all()

    def test_prediction_is_deterministic(self, frame, cache, config):
        model = build_model("densenet121", len(LABELS), pretrained=False)
        a = predict(model, frame.iloc[:8], cache, config)
        b = predict(model, frame.iloc[:8], cache, config)
        pd.testing.assert_frame_equal(a, b)

    def test_identifiers_align_with_the_input_order(self, frame, cache, config):
        model = build_model("densenet121", len(LABELS), pretrained=False)
        subset = frame.iloc[[5, 1, 9]]
        out = predict(model, subset, cache, config)
        assert list(out["patient_id"]) == list(subset["patient_id"])