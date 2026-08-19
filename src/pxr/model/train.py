from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "TrainError",
    "TrainConfig",
    "EpochRecord",
    "TrainingHistory",
    "CachedImageDataset",
    "build_model",
    "masked_bce_loss",
    "positive_weights",
    "macro_auroc",
    "train_fold",
    "predict",
]


class TrainError(RuntimeError):
    """Raised when training cannot proceed or its inputs are inconsistent."""


@dataclass
class TrainConfig:
    """Everything that defines a training run.

    Defaults follow the conventional multi-label chest-radiography recipe. Every
    value is read from ``study_config.yaml`` in practice, so a change is recorded in
    the config hash rather than living only in a notebook cell.
    """

    labels: list[str]
    architecture: str = "densenet121"
    image_size: int = 224
    batch_size: int = 32
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-5
    max_epochs: int = 30
    patience: int = 5
    warmup_epochs: int = 1
    mixed_precision: bool = True
    positive_weighting: bool = True
    max_positive_weight: float = 20.0
    rotation_degrees: float = 5.0
    translate_fraction: float = 0.05
    scale_range: tuple[float, float] = (0.95, 1.05)
    horizontal_flip: bool = False       
    num_workers: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        if self.horizontal_flip:
            raise TrainError(
                "horizontal_flip reverses laterality; the study's exposure is "
                "acquisition geometry, so this augmentation is not permitted"
            )
        if not self.labels:
            raise TrainError("no labels to train on")


@dataclass
class EpochRecord:
    """One epoch's metrics."""

    epoch: int
    train_loss: float
    val_loss: float
    val_macro_auroc: float
    lr: float
    seconds: float


@dataclass
class TrainingHistory:
    """The record of a fold's training run."""

    site: str
    fold: int
    epochs: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    best_val_macro_auroc: float = float("nan")
    stopped_early: bool = False

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(e) for e in self.epochs])

    def summary(self) -> str:
        return (
            f"{self.site} fold {self.fold}: best macro-AUROC "
            f"{self.best_val_macro_auroc:.4f} at epoch {self.best_epoch}"
            f"{' (early stop)' if self.stopped_early else ''}"
        )


# --------------------------------------------------------------------------- #
# Metrics and loss
# --------------------------------------------------------------------------- #


def positive_weights(
    labels: np.ndarray, *, cap: float = 20.0
) -> np.ndarray:
    """Per-label ``negatives / positives``, capped.

    Computed from the training fold alone. The cap matters: a label at 0.2%
    prevalence would otherwise earn a weight near 500, and the resulting gradients
    swamp every other label in a multi-label head.

    Missing entries are excluded from both counts, so a label's weight reflects the
    images where it was actually recorded.
    """
    weights = np.ones(labels.shape[1], dtype=np.float32)
    for j in range(labels.shape[1]):
        column = labels[:, j]
        observed = column[~np.isnan(column)]
        positives = float((observed == 1).sum())
        negatives = float((observed == 0).sum())
        if positives > 0:
            weights[j] = float(np.clip(negatives / positives, 1.0, cap))
    return weights


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray]:
    """Macro-averaged AUROC, skipping labels without both classes present.

    Returns
    -------
    (macro, per_label)
        ``per_label`` holds ``nan`` where a label could not be scored, so a fold
        missing a rare label's positives is visible rather than silently averaged
        away.
    """
    from sklearn.metrics import roc_auc_score

    per_label = np.full(y_true.shape[1], np.nan, dtype=float)
    for j in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, j])
        truth = y_true[mask, j]
        if len(np.unique(truth)) < 2:
            continue
        per_label[j] = roc_auc_score(truth, y_score[mask, j])
    return float(np.nanmean(per_label)) if np.any(~np.isnan(per_label)) else float("nan"), per_label


def masked_bce_loss(logits, targets, pos_weight=None):
    """Binary cross-entropy that ignores missing labels.

    A target of ``nan`` contributes no gradient for that image and label. The loss is
    averaged over observed entries only, so a batch full of unrecorded labels does
    not dilute the signal from those that were recorded.
    """
    import torch
    import torch.nn.functional as F

    observed = ~torch.isnan(targets)
    if not observed.any():
        return logits.sum() * 0.0     # keeps the graph intact for an empty batch

    safe = torch.nan_to_num(targets, nan=0.0)
    per_element = F.binary_cross_entropy_with_logits(
        logits, safe, reduction="none",
        pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
    )
    return (per_element * observed).sum() / observed.sum()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


class CachedImageDataset:
    """Serves images from the memmap cache with their labels.

    The cache stores raw ``uint8`` pixels, so normalisation and the
    grayscale-to-three-channel expansion ImageNet weights expect happen here - where
    they are visible beside the model and can change without rebuilding the cache.

    Augmentation applies to training only; validation and test see the image as
    stored, so their metrics are not perturbed by randomness.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        cache_index,
        labels: list[str],
        *,
        train: bool,
        config: TrainConfig,
    ):
        self.frame = frame.reset_index(drop=True)
        self.cache = cache_index
        self.labels = labels
        self.train = train
        self.config = config
        self._transform = self._build_transform()

    def _build_transform(self) -> Callable | None:
        if not self.train:
            return None
        from torchvision import transforms

        return transforms.RandomAffine(
            degrees=self.config.rotation_degrees,
            translate=(self.config.translate_fraction, self.config.translate_fraction),
            scale=self.config.scale_range,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        import torch

        from pxr.data.cache import read_images

        row = self.frame.iloc[idx]
        pixels = read_images(self.cache, [row["image_id"]])[0]

        # uint8 [0,255] -> float [0,1], then ImageNet statistics on three channels.
        image = torch.from_numpy(pixels).float().div_(255.0).unsqueeze(0)
        if self._transform is not None:
            image = self._transform(image)
        image = image.repeat(3, 1, 1)
        image = (image - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        targets = torch.tensor(
            [float(row[label]) if pd.notna(row[label]) else float("nan")
             for label in self.labels],
            dtype=torch.float32,
        )
        return image, targets


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_model(architecture: str, n_labels: int, *, pretrained: bool = True):
    """A pretrained backbone with a fresh multi-label head.

    ImageNet initialisation is standard for chest radiography and materially better
    than training from scratch at this cohort size; the three-channel expansion in
    the dataset exists to satisfy it.
    """
    import torch.nn as nn
    from torchvision import models

    if architecture == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, n_labels)
        return model
    if architecture == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_labels)
        return model
    raise TrainError(f"unsupported architecture: {architecture!r}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _loader(dataset, config: TrainConfig, *, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def train_fold(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    cache_index,
    config: TrainConfig,
    *,
    site: str,
    fold: int,
    checkpoint_dir: str | Path | None = None,
    device: str | None = None,
    progress: Callable[[str], None] = print,
) -> tuple[object, TrainingHistory]:
    """Train one fold and return the best model by validation macro-AUROC.

    The learning rate follows a linear warmup into a cosine decay - the conventional
    schedule, worth a point or two of AUROC over a flat rate and not tuned to these
    data. Training stops when validation macro-AUROC has not improved for
    ``patience`` epochs, and the returned model is the best epoch's, not the last.

    Raises
    ------
    TrainError
        If the frames overlap. Validation must be disjoint from training, or the
        stopping point and the operating threshold are chosen on data the model has
        already seen.
    """
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    overlap = set(train_frame["patient_id"]) & set(val_frame["patient_id"])
    if overlap:
        raise TrainError(
            f"{len(overlap)} patient(s) appear in both train and validation; the "
            "stopping point would be chosen on data the model trained on"
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed + fold)
    np.random.seed(config.seed + fold)

    train_set = CachedImageDataset(train_frame, cache_index, config.labels,
                                   train=True, config=config)
    val_set = CachedImageDataset(val_frame, cache_index, config.labels,
                                 train=False, config=config)
    train_loader = _loader(train_set, config, shuffle=True)
    val_loader = _loader(val_set, config, shuffle=False)

    model = build_model(config.architecture, len(config.labels)).to(device)
    optimiser = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    pos_weight = None
    if config.positive_weighting:
        weights = positive_weights(
            train_frame[config.labels].to_numpy(dtype=float), cap=config.max_positive_weight
        )
        pos_weight = torch.tensor(weights)
        progress(f"  positive weights: {dict(zip(config.labels, weights.round(1), strict=True))}")

    steps = max(1, len(train_loader))
    total = config.max_epochs * steps
    warmup = config.warmup_epochs * steps

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progressed = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, progressed)))

    scheduler = LambdaLR(optimiser, schedule)
    use_amp = config.mixed_precision and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = TrainingHistory(site=site, fold=fold)
    best_state, best_score, since_improvement = None, -np.inf, 0

    for epoch in range(config.max_epochs):
        started = time.time()
        model.train()
        running, seen = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = masked_bce_loss(model(images), targets, pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            running += float(loss) * len(images)
            seen += len(images)

        model.eval()
        val_running, val_seen = 0.0, 0
        scores, truths = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    loss = masked_bce_loss(logits, targets, pos_weight)
                val_running += float(loss) * len(images)
                val_seen += len(images)
                scores.append(torch.sigmoid(logits.float()).cpu().numpy())
                truths.append(targets.cpu().numpy())

        macro, _ = macro_auroc(np.concatenate(truths), np.concatenate(scores))
        record = EpochRecord(
            epoch=epoch,
            train_loss=running / max(1, seen),
            val_loss=val_running / max(1, val_seen),
            val_macro_auroc=macro,
            lr=optimiser.param_groups[0]["lr"],
            seconds=time.time() - started,
        )
        history.epochs.append(record)
        progress(
            f"  epoch {epoch:>2}  train {record.train_loss:.4f}  val {record.val_loss:.4f}  "
            f"macro-AUROC {macro:.4f}  ({record.seconds:.0f}s)"
        )

        if macro > best_score:
            best_score, history.best_epoch, since_improvement = macro, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since_improvement += 1
            if since_improvement >= config.patience:
                history.stopped_early = True
                progress(f"  no improvement for {config.patience} epochs; stopping")
                break

    history.best_val_macro_auroc = best_score
    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_dir is not None:
        directory = Path(checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, directory / f"{site}_fold{fold}.pt")
        history.to_frame().to_csv(directory / f"{site}_fold{fold}_history.csv", index=False)
        (directory / f"{site}_fold{fold}_config.json").write_text(
            json.dumps({k: str(v) for k, v in vars(config).items()}, indent=2)
        )

    return model, history


def predict(
    model,
    frame: pd.DataFrame,
    cache_index,
    config: TrainConfig,
    *,
    device: str | None = None,
) -> pd.DataFrame:
    """Score a set of patients with a trained model.

    Returns
    -------
    DataFrame
        ``patient_id``, ``image_id``, and one probability column per label. No
        augmentation is applied, so a patient's score is deterministic.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dataset = CachedImageDataset(frame, cache_index, config.labels, train=False,
                                 config=config)
    loader = _loader(dataset, config, shuffle=False)

    scores = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=config.mixed_precision and device == "cuda"):
                logits = model(images)
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())

    out = pd.DataFrame(np.concatenate(scores), columns=config.labels)
    out.insert(0, "image_id", frame["image_id"].to_numpy())
    out.insert(0, "patient_id", frame["patient_id"].to_numpy())
    return out


def training_config_from(cfg, labels: Iterable[str] | None = None) -> TrainConfig:
    """Build a :class:`TrainConfig` from the study configuration."""
    model_cfg = cfg.model
    augment = model_cfg.get("augmentation", {})
    return TrainConfig(
        labels=list(labels) if labels is not None else list(cfg.analysis_labels),
        architecture=model_cfg["architecture"],
        image_size=int(model_cfg["image_size"]),
        batch_size=int(model_cfg["batch_size"]),
        lr=float(model_cfg["lr"]),
        weight_decay=float(model_cfg["weight_decay"]),
        max_epochs=int(model_cfg["max_epochs"]),
        patience=int(model_cfg["early_stopping"]["patience"]),
        warmup_epochs=int(model_cfg.get("warmup_epochs", 1)),
        mixed_precision=bool(model_cfg.get("mixed_precision", True)),
        positive_weighting=bool(model_cfg.get("positive_weighting", True)),
        max_positive_weight=float(model_cfg.get("max_positive_weight", 20.0)),
        rotation_degrees=float(augment.get("rotation_degrees", 5)),
        translate_fraction=float(augment.get("translate_fraction", 0.05)),
        scale_range=tuple(augment.get("scale_range", (0.95, 1.05))),
        horizontal_flip=bool(augment.get("horizontal_flip", False)),
        seed=int(cfg.splits["seed"]),
    )