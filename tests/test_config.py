from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from pxr.config import (
    Config,
    ConfigError,
    compute_config_hash,
    compute_scoped_hash,
    freeze_config,
    load_config,
)

REAL = "config/study_config.yaml"


@pytest.fixture
def raw() -> dict:
    with open(REAL, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write(target, data: dict):
    """Write a config; ``target`` may be a directory or a file path."""
    target = Path(target)
    if target.suffix != ".yaml":
        target.mkdir(parents=True, exist_ok=True)
        target = target / "study_config.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    return target


class TestRealConfig:
    def test_real_config_loads_and_validates(self):
        cfg = load_config(REAL)
        assert isinstance(cfg, Config)
        assert len(cfg.config_hash) == 12

    def test_eight_analysis_labels(self):
        cfg = load_config(REAL)
        assert len(cfg.analysis_labels) == 8
        assert "No Finding" in cfg.analysis_labels

    def test_no_finding_is_secondary_lane_not_primary_family(self):
        cfg = load_config(REAL)
        assert cfg.secondary_lane == ["No Finding"]
        assert "No Finding" not in cfg.primary_family
        assert len(cfg.primary_family) == 7

    def test_observation_schema_is_broader_than_analysis(self):
        cfg = load_config(REAL)
        for site in cfg.site_names:
            schema = cfg.observation_schema(site)
            assert len(schema) > len(cfg.analysis_labels)

    def test_nih_harmonisation_maps_effusion(self):
        cfg = load_config(REAL)
        assert cfg.harmonisation("nih")["Effusion"] == "Pleural Effusion"
        assert cfg.harmonisation("mimic-cxr") == {}

    def test_age_bins_align_with_cohort_age_min(self):
        cfg = load_config(REAL)
        assert cfg.age_bin_edges[0] == cfg.cohort["age_min"]
        assert len(cfg.age_bin_labels) == len(cfg.age_bin_edges) - 1

    def test_strata_are_sex_and_age_bin(self):
        assert load_config(REAL).strata == ("sex", "age_bin")


class TestHash:
    def test_hash_is_deterministic(self, raw):
        assert compute_config_hash(raw) == compute_config_hash(raw)

    def test_value_change_changes_hash(self, raw):
        other = copy.deepcopy(raw)
        other["model"]["lr"] = 5.0e-4
        assert compute_config_hash(other) != compute_config_hash(raw)

    def test_key_order_does_not_change_hash(self, raw):
        reordered = dict(reversed(list(raw.items())))
        assert compute_config_hash(reordered) == compute_config_hash(raw)

    def test_comments_and_formatting_do_not_change_hash(self, tmp_path, raw):
        """Reformatting the YAML must not orphan artifacts; changing a value must."""
        a = load_config(write(tmp_path, raw))
        commented = (tmp_path / "study_config.yaml").read_text() + "\n# a trailing comment\n"
        (tmp_path / "study_config.yaml").write_text(commented, encoding="utf-8")
        b = load_config(tmp_path / "study_config.yaml")
        assert a.config_hash == b.config_hash

    def test_hash_is_lowercase_hex_12(self, raw):
        h = compute_config_hash(raw)
        assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)


class TestValidation:
    def test_missing_section_rejected(self, tmp_path, raw):
        raw.pop("splits")
        with pytest.raises(ConfigError, match="missing config section"):
            load_config(write(tmp_path, raw))

    def test_duplicate_analysis_label_rejected(self, tmp_path, raw):
        raw["labels"]["analysis"].append("Edema")
        with pytest.raises(ConfigError, match="duplicates"):
            load_config(write(tmp_path, raw))

    def test_label_unreachable_from_a_site_rejected(self, tmp_path, raw):
        """A label no site schema can supply is a silent-empty-column trap."""
        raw["labels"]["analysis"].append("Fracture")  # absent from nih14
        with pytest.raises(ConfigError, match="cannot supply analysis label"):
            load_config(write(tmp_path, raw))

    def test_removing_nih_harmonisation_rejected(self, tmp_path, raw):
        raw["labels"]["harmonisation"]["nih14"] = {}
        with pytest.raises(ConfigError, match="Pleural Effusion"):
            load_config(write(tmp_path, raw))

    def test_secondary_lane_outside_analysis_rejected(self, tmp_path, raw):
        raw["labels"]["secondary_lane"] = ["Hernia"]
        with pytest.raises(ConfigError, match="secondary_lane"):
            load_config(write(tmp_path, raw))

    def test_age_bin_label_count_mismatch_rejected(self, tmp_path, raw):
        raw["demographics"]["age_bins"]["labels"] = ["18-39", "40+"]
        with pytest.raises(ConfigError, match="age_bins"):
            load_config(write(tmp_path, raw))

    def test_non_monotonic_age_edges_rejected(self, tmp_path, raw):
        raw["demographics"]["age_bins"]["edges"] = [18, 60, 40, 80, 121]
        with pytest.raises(ConfigError, match="strictly increasing"):
            load_config(write(tmp_path, raw))

    def test_age_edges_must_start_at_cohort_age_min(self, tmp_path, raw):
        raw["demographics"]["age_bins"]["edges"] = [21, 40, 60, 80, 121]
        with pytest.raises(ConfigError, match="cohort.age_min"):
            load_config(write(tmp_path, raw))

    def test_single_fold_rejected(self, tmp_path, raw):
        raw["splits"]["n_folds"] = 1
        with pytest.raises(ConfigError, match="n_folds"):
            load_config(write(tmp_path, raw))

    def test_reference_mix_not_summing_to_one_rejected(self, tmp_path, raw):
        raw["analysis"]["standardization"]["reference_mix"] = {"AP": 0.5, "PA": 0.4}
        with pytest.raises(ConfigError, match="reference_mix"):
            load_config(write(tmp_path, raw))

    def test_unknown_primary_source_rejected(self, tmp_path, raw):
        raw["model"]["primary_source"] = "padchest"
        with pytest.raises(ConfigError, match="primary_source"):
            load_config(write(tmp_path, raw))

    def test_unsupported_stratum_rejected(self, tmp_path, raw):
        raw["demographics"]["strata"] = ["sex", "race"]
        with pytest.raises(ConfigError, match="unsupported stratum"):
            load_config(write(tmp_path, raw))

    def test_early_stopping_on_training_metric_rejected(self, tmp_path, raw):
        raw["model"]["early_stopping"]["monitor"] = "train_loss"
        with pytest.raises(ConfigError, match="validation metric"):
            load_config(write(tmp_path, raw))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")


class TestPathsAndNaming:
    def test_drive_path_resolves_under_root(self):
        cfg = load_config(REAL)
        assert str(cfg.drive_path("cohorts")).endswith("patient-or-xray/data/cohorts")

    def test_unknown_path_key_raises(self):
        with pytest.raises(ConfigError, match="unknown path key"):
            load_config(REAL).drive_path("nowhere")

    def test_artifact_name_is_hash_stamped(self):
        """Scores depend on the model, so they carry the model-scoped stamp."""
        cfg = load_config(REAL)
        name = cfg.artifact_name("scores", source="mimic-cxr", site="nih", seed=1)
        assert name == f"scores_mimic-cxr_nih_seed1_{cfg.model_hash}.parquet"

    def test_artifact_name_minimal_form(self):
        """Cohorts carry the cohort-scoped stamp, so tuning the model leaves them valid."""
        cfg = load_config(REAL)
        assert cfg.artifact_name("cohort", site="nih") == f"cohort_nih_{cfg.cohort_hash}.parquet"


class TestFreeze:
    def test_freeze_writes_stamped_copy(self, tmp_path):
        cfg = load_config(REAL)
        out = freeze_config(cfg, frozen_dir=tmp_path)
        assert out.exists()
        assert cfg.config_hash in out.name
        assert yaml.safe_load(out.read_text()) == cfg.data


class TestNewValidations:
    """Rules added with the power gate."""

    def test_age_threshold_outside_eligible_range_rejected(self, tmp_path, raw):
        """A threshold at the boundary leaves one side of the contrast empty."""
        raw["demographics"]["primary_age_threshold"] = 15   # below cohort.age_min
        with pytest.raises(ConfigError, match="primary_age_threshold"):
            load_config(write(tmp_path, raw))

    def test_age_threshold_inside_range_accepted(self, tmp_path, raw):
        raw["demographics"]["primary_age_threshold"] = 70
        assert load_config(write(tmp_path, raw)).primary_age_threshold == 70

    def test_unknown_inferential_site_rejected(self, tmp_path, raw):
        """A typo here would silently gate tiers on nothing."""
        raw["analysis"]["inferential_sites"] = ["mimic-cxr", "padchest"]
        raw["model"]["training_sites"] = ["mimic-cxr", "padchest"]
        with pytest.raises(ConfigError, match="unconfigured site"):
            load_config(write(tmp_path, raw))

    def test_inferential_site_without_its_own_model_rejected(self, tmp_path, raw):
        """A site cannot carry within-site claims without training its own model."""
        raw["model"]["training_sites"] = ["mimic-cxr"]
        with pytest.raises(ConfigError, match="not in model.training_sites"):
            load_config(write(tmp_path, raw))

    def test_training_sites_must_be_configured(self, tmp_path, raw):
        raw["model"]["training_sites"] = ["padchest"]
        with pytest.raises(ConfigError, match="unconfigured site"):
            load_config(write(tmp_path, raw))

    def test_inferential_sites_defaults_to_all_sites(self, tmp_path, raw):
        raw["analysis"].pop("inferential_sites", None)
        cfg = load_config(write(tmp_path, raw))
        assert cfg.inferential_sites == cfg.site_names


class TestScopedHashes:
    """An artifact must depend only on the settings it actually reads.

    A single hash over the whole file made a learning-rate change orphan cohorts that
    were byte-identical to what they had been, forcing rebuilds that changed nothing.
    """

    def test_model_change_leaves_cohorts_and_folds_valid(self, tmp_path, raw):
        before = load_config(write(tmp_path / "a", raw))
        raw["model"]["lr"] = 3.0e-4
        after = load_config(write(tmp_path / "b", raw))
        assert after.cohort_hash == before.cohort_hash
        assert after.split_hash == before.split_hash
        assert after.model_hash != before.model_hash

    def test_split_change_leaves_cohorts_valid(self, tmp_path, raw):
        before = load_config(write(tmp_path / "a", raw))
        raw["splits"]["n_folds"] = 10
        after = load_config(write(tmp_path / "b", raw))
        assert after.cohort_hash == before.cohort_hash
        assert after.split_hash != before.split_hash

    def test_split_change_invalidates_models(self, tmp_path, raw):
        """Models trained on different folds are different models."""
        before = load_config(write(tmp_path / "a", raw))
        raw["splits"]["seed"] = 99
        after = load_config(write(tmp_path / "b", raw))
        assert after.model_hash != before.model_hash

    def test_cohort_change_invalidates_everything_downstream(self, tmp_path, raw):
        before = load_config(write(tmp_path / "a", raw))
        raw["labels"]["analysis"] = raw["labels"]["analysis"][:-1]
        raw["labels"]["secondary_lane"] = []
        after = load_config(write(tmp_path / "b", raw))
        assert after.cohort_hash != before.cohort_hash
        assert after.split_hash != before.split_hash
        assert after.model_hash != before.model_hash

    def test_analysis_change_leaves_models_valid(self, tmp_path, raw):
        """Re-analysing existing predictions must not require retraining."""
        before = load_config(write(tmp_path / "a", raw))
        raw["analysis"]["bootstrap"]["replicates"] = 5000
        after = load_config(write(tmp_path / "b", raw))
        assert after.model_hash == before.model_hash
        assert after.analysis_hash != before.analysis_hash

    def test_artifact_names_carry_the_right_scope(self, tmp_path, raw):
        cfg = load_config(write(tmp_path / "a", raw))
        assert cfg.cohort_hash in cfg.artifact_name("cohort", site="nih")
        assert cfg.split_hash in cfg.artifact_name("folds")
        assert cfg.model_hash in cfg.artifact_name("oof", site="nih")

    def test_explicit_scope_overrides_the_mapping(self, tmp_path, raw):
        cfg = load_config(write(tmp_path / "a", raw))
        assert cfg.model_hash in cfg.artifact_name("anything", scope="model")

    def test_unknown_scope_rejected(self, tmp_path, raw):
        cfg = load_config(write(tmp_path / "a", raw))
        with pytest.raises(ConfigError, match="unknown hash scope"):
            compute_scoped_hash(cfg.data, "invented")

    def test_comments_do_not_change_any_hash(self, tmp_path, raw):
        """Hashes are over the parse, so re-commenting the YAML is free."""
        a = load_config(write(tmp_path / "a", raw))
        b = load_config(write(tmp_path / "b", raw))
        assert (a.cohort_hash, a.split_hash, a.model_hash) == (
            b.cohort_hash, b.split_hash, b.model_hash)

    def test_config_hash_still_covers_the_whole_file(self, tmp_path, raw):
        """The preregistration cites the whole configuration, not a scope."""
        before = load_config(write(tmp_path / "a", raw))
        raw["analysis"]["bootstrap"]["replicates"] = 5000
        after = load_config(write(tmp_path / "b", raw))
        assert after.config_hash != before.config_hash