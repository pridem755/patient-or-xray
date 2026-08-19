from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "compute_config_hash",
    "compute_scoped_hash",
    "HASH_SCOPES",
    "freeze_config",
]

HASH_LENGTH = 12


class ConfigError(ValueError):
    """Raised when the configuration is internally inconsistent."""


def _canonical_json(data: Any) -> str:
    """Deterministic serialisation: sorted keys, no incidental whitespace."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)

HASH_SCOPES: dict[str, tuple[str, ...]] = {
    "cohort": ("sites", "labels", "cohort", "demographics"),
    "split": ("sites", "labels", "cohort", "demographics", "splits"),
    "model": ("sites", "labels", "cohort", "demographics", "splits", "model"),
    "analysis": ("sites", "labels", "cohort", "demographics", "splits", "model",
                 "analysis"),
}


def compute_scoped_hash(
    data: dict, scope: str, length: int = HASH_LENGTH
) -> str:
    """Hash the config sections a given kind of artifact depends on.

    Examples
    --------
    >>> a = {"cohort": {"age_min": 18}, "model": {"lr": 1e-4}, "sites": {}, "labels": {},
    ...      "demographics": {}, "splits": {}, "analysis": {}}
    >>> b = {**a, "model": {"lr": 3e-4}}
    >>> compute_scoped_hash(a, "cohort") == compute_scoped_hash(b, "cohort")
    True
    >>> compute_scoped_hash(a, "model") == compute_scoped_hash(b, "model")
    False
    """
    if scope not in HASH_SCOPES:
        raise ConfigError(f"unknown hash scope {scope!r}; expected {sorted(HASH_SCOPES)}")
    scoped = {section: data.get(section) for section in HASH_SCOPES[scope]}
    digest = hashlib.sha256(_canonical_json(scoped).encode("utf-8")).hexdigest()
    return digest[:length]


def compute_config_hash(data: dict, length: int = HASH_LENGTH) -> str:
    """First ``length`` hex characters of the SHA-256 of the canonical config.

    Computed over the parsed structure so that comments and formatting do not
    affect it, while any change to a value does.

    Examples
    --------
    >>> compute_config_hash({"a": 1}) == compute_config_hash({"a": 1})
    True
    >>> compute_config_hash({"a": 1}) == compute_config_hash({"a": 2})
    False
    """
    digest = hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()
    return digest[:length]


@dataclass(frozen=True)
class Config:
    """Parsed study configuration with convenience accessors.

    Attributes
    ----------
    data
        The full parsed YAML.
    path
        Where it was loaded from.
    config_hash
        Stamp applied to every artifact produced under this configuration.
    """

    data: dict
    path: Path
    config_hash: str

    # -- section accessors ------------------------------------------------- #

    @property
    def cohort_hash(self) -> str:
        """Stamp for cohort tables - unchanged by model or analysis settings."""
        return compute_scoped_hash(self.data, "cohort")

    @property
    def split_hash(self) -> str:
        """Stamp for fold assignments - unchanged by model settings."""
        return compute_scoped_hash(self.data, "split")

    @property
    def model_hash(self) -> str:
        """Stamp for checkpoints and predictions - changes when training changes."""
        return compute_scoped_hash(self.data, "model")

    @property
    def analysis_hash(self) -> str:
        """Stamp for analysis outputs."""
        return compute_scoped_hash(self.data, "analysis")

    def hash_for(self, artifact: str) -> str:
        """The stamp an artifact kind should carry."""
        return {
            "cohort": self.cohort_hash, "flow": self.cohort_hash,
            "audit": self.cohort_hash,
            "folds": self.split_hash, "splits": self.split_hash,
            "model": self.model_hash, "scores": self.model_hash,
            "oof": self.model_hash,
        }.get(artifact, self.analysis_hash)

    @property
    def meta(self) -> dict:
        return self.data["meta"]

    @property
    def paths(self) -> dict:
        return self.data["paths"]

    @property
    def sites(self) -> dict:
        return self.data["sites"]

    @property
    def cohort(self) -> dict:
        return self.data["cohort"]

    @property
    def demographics(self) -> dict:
        return self.data["demographics"]

    @property
    def splits(self) -> dict:
        return self.data["splits"]

    @property
    def model(self) -> dict:
        return self.data["model"]

    @property
    def analysis(self) -> dict:
        return self.data["analysis"]

    # -- derived values ---------------------------------------------------- #

    @property
    def analysis_labels(self) -> list[str]:
        """The labels the model predicts and the study analyses."""
        return list(self.data["labels"]["analysis"])

    @property
    def secondary_lane(self) -> list[str]:
        """Labels excluded from the primary corrected family (inverted semantics)."""
        return list(self.data["labels"].get("secondary_lane", []))

    @property
    def primary_family(self) -> list[str]:
        """Analysis labels eligible for the primary family, before the power gate."""
        secondary = set(self.secondary_lane)
        return [lab for lab in self.analysis_labels if lab not in secondary]

    def observation_schema(self, site: str) -> list[str]:
        """The full source label set for ``site``.

        Passed to the No Finding integrity guard so a contradiction cannot hide in
        a label that was not selected for modelling.
        """
        schema_name = self.sites[site]["label_schema"]
        return list(self.data["labels"]["observation_schema"][schema_name])

    def harmonisation(self, site: str) -> dict[str, str]:
        """Source-column -> analysis-label renames for ``site``."""
        schema_name = self.sites[site]["label_schema"]
        return dict(self.data["labels"]["harmonisation"].get(schema_name) or {})

    @property
    def age_bin_edges(self) -> list[int]:
        return list(self.demographics["age_bins"]["edges"])

    @property
    def age_bin_labels(self) -> list[str]:
        return list(self.demographics["age_bins"]["labels"])

    @property
    def training_sites(self) -> list[str]:
        """Sites that train their own model, one per institution."""
        return list(self.model.get("training_sites") or [self.model["primary_source"]])

    @property
    def n_folds(self) -> int:
        return int(self.splits["n_folds"])

    @property
    def val_fraction(self) -> float:
        """Share of the development set held out for validation within each fold."""
        return float(self.splits.get("val_fraction", 0.15))

    @property
    def val_stratify_by(self) -> list[str]:
        """Variables balanced within the inner validation split (coarser than folds)."""
        declared = self.splits.get("val_stratify_by")
        if declared:
            return list(declared)
        return [c for c in self.stratify_by if c in ("view", "sex", "age_bin")]

    @property
    def min_val_stratum(self) -> int:
        """Smallest stratum that contributes validation patients."""
        return int(self.splits.get("min_val_stratum", 7))

    @property
    def stratify_by(self) -> list[str]:
        """Variables held even across folds - a pre-specified scientific choice."""
        return list(self.splits["stratify_by"])

    @property
    def inferential_sites(self) -> list[str]:
        """Sites whose cells decide label tiers; others are reported descriptively."""
        return list(self.analysis.get("inferential_sites") or self.site_names)

    @property
    def primary_age_threshold(self) -> int:
        """Fixed clinical cut-point for the inferential age contrast."""
        return int(self.demographics.get("primary_age_threshold", 65))

    @property
    def strata(self) -> tuple[str, ...]:
        """Demographic columns crossed with view when enumerating inferential cells."""
        return tuple(self.demographics["strata"])

    @property
    def min_positives_warning(self) -> int:
        return int(self.analysis["power"]["min_positives_warning"])

    @property
    def site_names(self) -> list[str]:
        return list(self.sites)

    def drive_path(self, key: str) -> Path:
        """Absolute Drive path for a configured directory key, e.g. ``"cohorts"``."""
        if key not in self.paths:
            raise ConfigError(f"unknown path key: {key!r}")
        root = Path(self.paths["drive_root"])
        value = Path(self.paths[key])
        return value if value.is_absolute() else root / value

    def artifact_name(
        self,
        artifact: str,
        *,
        site: str | None = None,
        source: str | None = None,
        seed: int | None = None,
        ext: str = "parquet",
        scope: str | None = None,
    ) -> str:
        """Hash-stamped filename: ``{artifact}_{source}_{site}_seed{k}_{hash}.{ext}``.

        The stamp is scoped to what the artifact depends on, so a cohort keeps its
        name when a training hyperparameter changes. ``scope`` overrides the mapping
        in :meth:`hash_for` when an artifact does not follow it.
        """
        parts = [artifact]
        if source:
            parts.append(source)
        if site:
            parts.append(site)
        if seed is not None:
            parts.append(f"seed{seed}")
        parts.append(
            compute_scoped_hash(self.data, scope) if scope else self.hash_for(artifact)
        )
        return "_".join(parts) + f".{ext}"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

_REQUIRED_SECTIONS = (
    "meta",
    "paths",
    "sites",
    "labels",
    "cohort",
    "demographics",
    "splits",
    "model",
    "analysis",
)


def _validate(data: dict) -> None:
    """Check internal consistency; raise :class:`ConfigError` on the first problem."""
    missing = [s for s in _REQUIRED_SECTIONS if s not in data]
    if missing:
        raise ConfigError(f"missing config section(s): {missing}")

    labels = data["labels"]
    analysis = labels.get("analysis") or []
    if not analysis:
        raise ConfigError("labels.analysis is empty")
    if len(set(analysis)) != len(analysis):
        raise ConfigError("labels.analysis contains duplicates")

    # Every analysis label must be reachable from every site's observation schema,
    # either natively or through that schema's harmonisation map.
    schemas = labels.get("observation_schema") or {}
    harmon = labels.get("harmonisation") or {}
    for site, spec in data["sites"].items():
        schema_name = spec.get("label_schema")
        if schema_name not in schemas:
            raise ConfigError(f"site {site!r} references unknown label_schema {schema_name!r}")
        source_labels = set(schemas[schema_name])
        renames = harmon.get(schema_name) or {}
        reachable = {renames.get(lab, lab) for lab in source_labels}
        unreachable = [lab for lab in analysis if lab not in reachable]
        if unreachable:
            raise ConfigError(
                f"site {site!r} (schema {schema_name!r}) cannot supply analysis label(s) "
                f"{unreachable}; add a harmonisation entry or drop the label"
            )

    secondary = set(labels.get("secondary_lane") or [])
    stray = secondary - set(analysis)
    if stray:
        raise ConfigError(f"labels.secondary_lane contains non-analysis labels: {sorted(stray)}")

    bins = data["demographics"]["age_bins"]
    edges, bin_labels = bins["edges"], bins["labels"]
    if len(edges) != len(bin_labels) + 1:
        raise ConfigError(
            f"age_bins: {len(edges)} edges require {len(edges) - 1} labels, "
            f"found {len(bin_labels)}"
        )
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ConfigError(f"age_bins.edges must be strictly increasing: {edges}")
    age_min = data["cohort"].get("age_min")
    age_max = data["cohort"].get("age_max")
    if age_min is not None and edges[0] != age_min:
        raise ConfigError(
            f"age_bins.edges starts at {edges[0]} but cohort.age_min is {age_min}"
        )
    if age_max is not None and edges[-1] <= age_max:
        raise ConfigError(
            f"age_bins.edges ends at {edges[-1]}, which does not cover cohort.age_max "
            f"({age_max}); the last edge must be strictly greater"
        )

    threshold = data["demographics"].get("primary_age_threshold")
    if threshold is not None and not (age_min or 0) < threshold < (age_max or 200):
        raise ConfigError(
            f"demographics.primary_age_threshold ({threshold}) must lie strictly inside "
            f"the eligible age range [{age_min}, {age_max}], or one side of the "
            "contrast would be empty"
        )

    for col in data["demographics"].get("strata", []):
        if col not in ("sex", "age_bin"):
            raise ConfigError(f"unsupported stratum {col!r} (demographic scope is age/sex/site)")

    n_folds = data["splits"].get("n_folds")
    if n_folds is None or int(n_folds) < 2:
        raise ConfigError(f"splits.n_folds must be at least 2, got {n_folds}")

    val_fraction = data["splits"].get("val_fraction", 0.15)
    if not 0 < float(val_fraction) < 1:
        raise ConfigError(f"splits.val_fraction must lie in (0, 1), got {val_fraction}")
    strat = data["splits"].get("stratify_by") or []
    if not strat:
        raise ConfigError(
            "splits.stratify_by must name at least one column; which variables are "
            "balanced across folds is a pre-specified choice"
        )

    training = data["model"].get("training_sites") or []
    if not training:
        raise ConfigError("model.training_sites must name at least one site")
    unknown_training = [s for s in training if s not in data["sites"]]
    if unknown_training:
        raise ConfigError(f"model.training_sites names unconfigured site(s): {unknown_training}")
    inferential_set = set(data["analysis"].get("inferential_sites") or [])
    not_trained = sorted(inferential_set - set(training))
    if not_trained:
        raise ConfigError(
            f"inferential site(s) {not_trained} are not in model.training_sites; a site "
            "cannot carry within-site inferential claims without its own model"
        )

    mix = data["analysis"]["standardization"].get("reference_mix") or {}
    if mix and abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ConfigError(f"standardization.reference_mix must sum to 1.0, got {sum(mix.values())}")

    target = data["analysis"]["threshold_rule"].get("fixed_sensitivity_target")
    if target is not None and not 0 < float(target) < 1:
        raise ConfigError(f"fixed_sensitivity_target must be in (0, 1), got {target}")

    inferential = data["analysis"].get("inferential_sites") or []
    unknown_sites = [s for s in inferential if s not in data["sites"]]
    if unknown_sites:
        raise ConfigError(
            f"analysis.inferential_sites names unconfigured site(s): {unknown_sites}"
        )

    source = data["model"].get("primary_source")
    if source not in data["sites"]:
        raise ConfigError(f"model.primary_source {source!r} is not a configured site")

    if data["model"]["early_stopping"]["monitor"].startswith("val_") is False:
        raise ConfigError("early stopping must monitor a validation metric")


def load_config(path: str | Path = "config/study_config.yaml") -> Config:
    """Load, validate, and hash the study configuration.

    Parameters
    ----------
    path
        Path to ``study_config.yaml``.

    Returns
    -------
    Config

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ConfigError
        If the configuration is internally inconsistent.

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg.config_hash
    'a1b2c3d4e5f6'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config must parse to a mapping, got {type(data).__name__}")
    _validate(data)
    return Config(data=data, path=path, config_hash=compute_config_hash(data))


def freeze_config(cfg: Config, frozen_dir: str | Path = "config/frozen") -> Path:
    """Write a timestamped, hash-stamped copy of the config.

    Called by notebook 04 immediately before the ``v1.0-prereg`` tag, so the exact
    configuration under which the study was pre-specified is recoverable.

    Returns
    -------
    Path
        Location of the frozen copy.
    """
    frozen_dir = Path(frozen_dir)
    frozen_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = frozen_dir / f"study_config_{stamp}_{cfg.config_hash}.yaml"
    shutil.copy2(cfg.path, target)
    return target