from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

__all__ = [
    "Severity",
    "Violation",
    "ValidationReport",
    "ContractError",
    "CohortShape",
    "COHORT_REQUIRED_COLUMNS",
    "VALID_VIEWS",
    "VALID_SEX",
    "CONFIG_HASH_PATTERN",
    "validate_primary_cohort",
    "validate_repeated_image_cohort",
    "cohort_integrity_stats",
    "inferential_cell_table",
]


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class CohortShape(str, Enum):
    """Which uniqueness invariant the cohort is expected to satisfy."""

    ONE_PER_PATIENT = "one_per_patient"
    REPEATED_IMAGES = "repeated_images"


@dataclass(frozen=True)
class Violation:
    """A single contract breach."""

    check: str
    severity: Severity
    message: str
    n_affected: int = 0
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - formatting only
        head = f"[{self.severity.value}] {self.check}: {self.message}"
        if self.n_affected:
            head += f" (n={self.n_affected:,})"
        if self.examples:
            head += f" e.g. {list(self.examples)[:3]}"
        return head


@dataclass
class ValidationReport:
    """Outcome of validating one artifact."""

    artifact: str
    n_rows: int
    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """True when no ERROR-severity violations were found."""
        return not self.errors

    def raise_if_failed(self) -> ValidationReport:
        """Raise :class:`ContractError` if any ERROR violations exist; else return self."""
        if not self.ok:
            raise ContractError(self)
        return self

    def summary(self, max_warnings: int | None = 20) -> str:
        lines = [
            f"Contract report - {self.artifact} ({self.n_rows:,} rows): "
            f"{'PASS' if self.ok else 'FAIL'} "
            f"[{len(self.errors)} error(s), {len(self.warnings)} warning(s)]"
        ]
        lines += [f"  {v}" for v in self.errors]
        warns = self.warnings
        shown = warns if max_warnings is None else warns[:max_warnings]
        lines += [f"  {v}" for v in shown]
        if max_warnings is not None and len(warns) > max_warnings:
            lines.append(f"  ... {len(warns) - max_warnings} further warning(s) suppressed")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Violations as a table, for the reproducibility appendix."""
        return pd.DataFrame(
            [
                {
                    "check": v.check,
                    "severity": v.severity.value,
                    "message": v.message,
                    "n_affected": v.n_affected,
                    "examples": ", ".join(v.examples),
                }
                for v in self.violations
            ]
        )

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return self.summary()


class ContractError(AssertionError):
    """Raised when an artifact violates its contract."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.summary())


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Columns every cohort table must carry, with the dtype family each must satisfy.
COHORT_REQUIRED_COLUMNS: dict[str, str] = {
    "patient_id": "id",
    "study_id": "id",
    "image_id": "id",
    "site": "string",
    "view": "string",
    "sex": "string",
    "age": "numeric",
    "config_hash": "string",
}

VALID_VIEWS: frozenset[str] = frozenset({"AP", "PA"})
VALID_SEX: frozenset[str] = frozenset({"Male", "Female"})

#: config_hash is the first 12 hex characters of the SHA-256 of study_config.yaml.
CONFIG_HASH_PATTERN = re.compile(r"^[0-9a-f]{12}$")

_NO_FINDING = "No Finding"


def _safe_examples(values, limit: int = 3) -> tuple[str, ...]:
    out: list[str] = []
    try:
        for v in values:
            out.append(repr(v))
            if len(out) >= limit:
                break
    except Exception:  # pragma: no cover - defensive
        return ("<unrenderable>",)
    return tuple(out)


def _is_blank(series: pd.Series) -> pd.Series:
    """True where a value is an empty or whitespace-only string."""
    as_str = series.astype("string")
    return as_str.notna() & (as_str.str.strip().str.len() == 0)


def _dtype_ok(series: pd.Series, kind: str) -> bool:
    if kind == "id":
        # Site-native identifier types are permitted; semantics are checked separately.
        return True
    if kind == "numeric":
        return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    if kind == "string":
        return (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
    raise ValueError(f"unknown dtype kind: {kind}")  # pragma: no cover


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_frame_shape(df: pd.DataFrame) -> list[Violation]:
    out: list[Violation] = []
    if len(df) == 0:
        out.append(
            Violation(
                "frame.empty",
                Severity.ERROR,
                "cohort has zero rows - a failed join or over-aggressive filter cannot be "
                "allowed to pass as a valid cohort",
            )
        )
    dupes = df.columns[df.columns.duplicated()].unique().tolist()
    if dupes:
        out.append(
            Violation(
                "frame.duplicate_columns",
                Severity.ERROR,
                f"duplicate column names make column access ambiguous: {dupes}",
                n_affected=len(dupes),
            )
        )
    return out


def check_columns(df: pd.DataFrame, analysis_labels: list[str]) -> list[Violation]:
    out: list[Violation] = []
    missing = [c for c in COHORT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        out.append(Violation("columns.missing", Severity.ERROR, f"absent columns: {missing}"))
    missing_labels = [c for c in analysis_labels if c not in df.columns]
    if missing_labels:
        out.append(
            Violation(
                "columns.missing_labels",
                Severity.ERROR,
                f"absent label columns: {missing_labels}",
            )
        )
    for col, kind in COHORT_REQUIRED_COLUMNS.items():
        if col not in df.columns:
            continue
        series = df[col]
        if not _dtype_ok(series, kind):
            msg = f"column '{col}' expected {kind}, found {series.dtype}"
            if kind == "numeric":
                msg += " (booleans are not valid numerics here)"
            out.append(Violation("columns.dtype", Severity.ERROR, msg))
        n_null = int(series.isna().sum())
        if n_null:
            out.append(
                Violation(
                    "columns.null",
                    Severity.ERROR,
                    f"column '{col}' contains nulls",
                    n_affected=n_null,
                )
            )
    return out


def check_identifiers(df: pd.DataFrame, shape: CohortShape) -> list[Violation]:
    """Identifiers are unique, non-blank, and relationally coherent."""
    out: list[Violation] = []
    unique_cols = (
        ("patient_id", "study_id", "image_id")
        if shape is CohortShape.ONE_PER_PATIENT
        else ("image_id",)
    )

    for col in unique_cols:
        if col not in df.columns:
            continue
        dup = df[col][df[col].duplicated(keep=False)]
        if len(dup):
            out.append(
                Violation(
                    f"identifiers.duplicate_{col}",
                    Severity.ERROR,
                    f"'{col}' must be unique under shape={shape.value}",
                    n_affected=int(dup.nunique()),
                    examples=_safe_examples(pd.unique(dup)),
                )
            )

    # Blank identifiers are non-null but meaningless.
    for col in ("patient_id", "study_id", "image_id", "site", "config_hash"):
        if col not in df.columns:
            continue
        n_blank = int(_is_blank(df[col]).sum())
        if n_blank:
            out.append(
                Violation(
                    "identifiers.blank",
                    Severity.ERROR,
                    f"column '{col}' contains empty or whitespace-only values",
                    n_affected=n_blank,
                )
            )

    # Relational coherence: a study must belong to exactly one patient.
    if {"patient_id", "study_id"} <= set(df.columns):
        per_study = df.groupby("study_id", observed=True)["patient_id"].nunique()
        offenders = per_study[per_study > 1]
        if len(offenders):
            out.append(
                Violation(
                    "identifiers.study_spans_patients",
                    Severity.ERROR,
                    "a study_id maps to more than one patient_id, indicating a join defect",
                    n_affected=len(offenders),
                    examples=_safe_examples(offenders.index),
                )
            )
    return out


def check_provenance(
    df: pd.DataFrame,
    expected_site: str | None,
    expected_config_hash: str | None,
) -> list[Violation]:
    """Provenance columns are constant and match the caller's expectations."""
    out: list[Violation] = []
    for col, expected in (("site", expected_site), ("config_hash", expected_config_hash)):
        if col not in df.columns:
            continue
        observed = pd.unique(df[col].dropna())
        if len(observed) > 1:
            out.append(
                Violation(
                    f"provenance.{col}_not_constant",
                    Severity.ERROR,
                    f"'{col}' must be constant within a cohort table",
                    n_affected=len(observed),
                    examples=_safe_examples(observed),
                )
            )
            continue
        if len(observed) == 0:
            continue  # nullness is reported by check_columns
        value = observed[0]
        if expected is not None and str(value) != str(expected):
            out.append(
                Violation(
                    f"provenance.{col}_mismatch",
                    Severity.ERROR,
                    f"'{col}' is {value!r} but the caller expected {expected!r}",
                    examples=_safe_examples([value]),
                )
            )
    if "config_hash" in df.columns:
        observed = pd.unique(df["config_hash"].dropna())
        if len(observed) == 1 and not CONFIG_HASH_PATTERN.match(str(observed[0])):
            out.append(
                Violation(
                    "provenance.config_hash_format",
                    Severity.ERROR,
                    f"config_hash {observed[0]!r} is not 12 lowercase hex characters",
                )
            )
    return out


def check_view(df: pd.DataFrame) -> list[Violation]:
    """The exposure variable is binary and free of laterals (non-null values only)."""
    if "view" not in df.columns:
        return []
    present = df["view"].dropna()
    bad = present[~present.isin(VALID_VIEWS)]
    if len(bad):
        return [
            Violation(
                "view.invalid",
                Severity.ERROR,
                f"view must be one of {sorted(VALID_VIEWS)}",
                n_affected=len(bad),
                examples=_safe_examples(pd.unique(bad)),
            )
        ]
    return []


def check_demographics(
    df: pd.DataFrame, age_min: float = 18.0, age_max: float = 120.0
) -> list[Violation]:
    """Sex is in the analysed categories; age is numeric and plausible."""
    out: list[Violation] = []
    if "sex" in df.columns:
        present = df["sex"].dropna()
        bad = present[~present.isin(VALID_SEX)]
        if len(bad):
            out.append(
                Violation(
                    "demographics.sex",
                    Severity.ERROR,
                    f"sex must be one of {sorted(VALID_SEX)}",
                    n_affected=len(bad),
                    examples=_safe_examples(pd.unique(bad)),
                )
            )
    if "age" in df.columns and pd.api.types.is_numeric_dtype(df["age"]):
        if pd.api.types.is_bool_dtype(df["age"]):
            out.append(
                Violation("demographics.age_bool", Severity.ERROR, "age must not be boolean")
            )
        else:
            present = df["age"].dropna()
            bad_age = present[(present < age_min) | (present > age_max)]
            if len(bad_age):
                out.append(
                    Violation(
                        "demographics.age_range",
                        Severity.ERROR,
                        f"age outside [{age_min}, {age_max}]",
                        n_affected=len(bad_age),
                        examples=_safe_examples(pd.unique(bad_age)),
                    )
                )
    return out


def check_label_values(df: pd.DataFrame, labels: list[str]) -> list[Violation]:
    """Labels are binary or missing (0, 1, or NaN)."""
    out: list[Violation] = []
    for lab in labels:
        if lab not in df.columns:
            continue
        series = df[lab]
        allowed = series.isna() | series.isin([0, 1])
        bad = series[~allowed]
        if len(bad):
            out.append(
                Violation(
                    "labels.values",
                    Severity.ERROR,
                    f"label '{lab}' must be 0/1/NaN",
                    n_affected=len(bad),
                    examples=_safe_examples(pd.unique(bad)),
                )
            )
    return out


def check_no_finding_consistency(
    df: pd.DataFrame,
    observation_labels: list[str],
    analysis_labels: list[str] | None = None,
) -> list[Violation]:
    out: list[Violation] = []
    if _NO_FINDING not in df.columns:
        return out

    disease = [lab for lab in observation_labels if lab != _NO_FINDING and lab in df.columns]
    if not disease:
        return out

    if analysis_labels is not None and set(observation_labels) == set(analysis_labels):
        out.append(
            Violation(
                "labels.observation_schema_not_supplied",
                Severity.WARNING,
                "the No Finding guard is running over the analysis labels only; supply the "
                "full source observation schema so contradictions cannot hide in unmodelled "
                "labels",
                n_affected=len(disease),
            )
        )

    contradictory = (df[_NO_FINDING] == 1) & (df[disease] == 1).any(axis=1)
    n = int(contradictory.sum())
    if n:
        ex = (
            _safe_examples(df.loc[contradictory, "image_id"])
            if "image_id" in df.columns
            else ()
        )
        out.append(
            Violation(
                "labels.no_finding_contradiction",
                Severity.ERROR,
                "No Finding=1 co-occurs with a positive disease label; study-level label "
                "integrity is violated. Possible causes include patient-level aggregation, "
                "label mapping errors, or a duplicated join",
                n_affected=n,
                examples=ex,
            )
        )
    return out


def check_inferential_cells(
    df: pd.DataFrame,
    analysis_labels: list[str],
    min_positives: int = 25,
    age_bin_col: str = "age_bin",
    strata: tuple[str, ...] = ("sex",),
) -> list[Violation]:
    out: list[Violation] = []
    if "view" not in df.columns:
        return out

    if age_bin_col not in df.columns:
        out.append(
            Violation(
                "power.age_bin_missing",
                Severity.WARNING,
                f"'{age_bin_col}' absent, so age-stratified cells were not enumerated; the "
                "age dimension of the power gate is unchecked",
            )
        )

    strat_cols = [c for c in (*strata, age_bin_col) if c in df.columns]
    if not strat_cols:
        return out

    for lab in analysis_labels:
        if lab not in df.columns:
            continue
        for strat in strat_cols:
            for (view, level), series in df.groupby(["view", strat], observed=True)[lab]:
                n_pos = int((series == 1).sum())
                if n_pos < min_positives:
                    out.append(
                        Violation(
                            "power.thin_cell",
                            Severity.WARNING,
                            f"'{lab}' x view={view} x {strat}={level} has {n_pos} positives "
                            f"(< {min_positives}); underpowered for a within-view gap",
                            n_affected=n_pos,
                        )
                    )
    return out


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def _validate(
    df: pd.DataFrame,
    *,
    shape: CohortShape,
    analysis_labels: list[str],
    observation_labels: list[str] | None,
    artifact: str,
    expected_site: str | None,
    expected_config_hash: str | None,
    age_min: float,
    age_max: float,
    min_positives: int,
    age_bin_col: str,
    strata: tuple[str, ...],
    strict: bool,
) -> ValidationReport:
    obs = list(observation_labels) if observation_labels is not None else list(analysis_labels)

    violations = check_frame_shape(df)
    # Duplicate column names break every column-dependent check; stop after shape.
    if not any(v.check == "frame.duplicate_columns" for v in violations):
        violations += check_columns(df, analysis_labels)
        violations += check_identifiers(df, shape)
        violations += check_provenance(df, expected_site, expected_config_hash)
        violations += check_view(df)
        violations += check_demographics(df, age_min=age_min, age_max=age_max)
        violations += check_label_values(df, obs)
        violations += check_no_finding_consistency(
            df, observation_labels=obs, analysis_labels=analysis_labels
        )
        if len(df):
            violations += check_inferential_cells(
                df,
                analysis_labels,
                min_positives=min_positives,
                age_bin_col=age_bin_col,
                strata=strata,
            )

    report = ValidationReport(artifact=artifact, n_rows=len(df), violations=violations)
    return report.raise_if_failed() if strict else report


def validate_primary_cohort(
    df: pd.DataFrame,
    analysis_labels: list[str],
    *,
    observation_labels: list[str] | None = None,
    artifact: str = "cohort",
    expected_site: str | None = None,
    expected_config_hash: str | None = None,
    age_min: float = 18.0,
    age_max: float = 120.0,
    min_positives: int = 25,
    age_bin_col: str = "age_bin",
    strata: tuple[str, ...] = ("sex",),
    strict: bool = True,
) -> ValidationReport:
    return _validate(
        df,
        shape=CohortShape.ONE_PER_PATIENT,
        analysis_labels=analysis_labels,
        observation_labels=observation_labels,
        artifact=artifact,
        expected_site=expected_site,
        expected_config_hash=expected_config_hash,
        age_min=age_min,
        age_max=age_max,
        min_positives=min_positives,
        age_bin_col=age_bin_col,
        strata=strata,
        strict=strict,
    )


def validate_repeated_image_cohort(
    df: pd.DataFrame,
    analysis_labels: list[str],
    *,
    observation_labels: list[str] | None = None,
    artifact: str = "cohort_repeated",
    expected_site: str | None = None,
    expected_config_hash: str | None = None,
    age_min: float = 18.0,
    age_max: float = 120.0,
    min_positives: int = 25,
    age_bin_col: str = "age_bin",
    strata: tuple[str, ...] = ("sex",),
    strict: bool = True,
) -> ValidationReport:
    """Validate a repeated-image cohort: multiple images per patient are expected."""
    return _validate(
        df,
        shape=CohortShape.REPEATED_IMAGES,
        analysis_labels=analysis_labels,
        observation_labels=observation_labels,
        artifact=artifact,
        expected_site=expected_site,
        expected_config_hash=expected_config_hash,
        age_min=age_min,
        age_max=age_max,
        min_positives=min_positives,
        age_bin_col=age_bin_col,
        strata=strata,
        strict=strict,
    )


# --------------------------------------------------------------------------- #
# Reporting tables
# --------------------------------------------------------------------------- #


def cohort_integrity_stats(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Shallow label x view counts for each notebook's integrity cell."""
    rows: list[dict] = []
    for lab in labels:
        if lab not in df.columns:
            continue
        for view in sorted(VALID_VIEWS):
            sub = df[df["view"] == view] if "view" in df.columns else df.iloc[0:0]
            rows.append(
                {
                    "label": lab,
                    "view": view,
                    "n_rows": len(sub),
                    "n_positive": int((sub[lab] == 1).sum()),
                    "n_negative": int((sub[lab] == 0).sum()),
                    "n_missing": int(sub[lab].isna().sum()),
                }
            )
    return pd.DataFrame(rows)


def inferential_cell_table(
    df: pd.DataFrame,
    labels: list[str],
    *,
    age_bin_col: str = "age_bin",
    strata: tuple[str, ...] = ("sex",),
) -> pd.DataFrame:
    strat_cols = [c for c in (*strata, age_bin_col) if c in df.columns]
    site = df["site"].iloc[0] if "site" in df.columns and len(df) else None
    rows: list[dict] = []
    for lab in labels:
        if lab not in df.columns:
            continue
        for strat in strat_cols:
            for (view, level), series in df.groupby(["view", strat], observed=True)[lab]:
                rows.append(
                    {
                        "site": site,
                        "label": lab,
                        "view": view,
                        "stratum": strat,
                        "level": level,
                        "n_rows": len(series),
                        "n_positive": int((series == 1).sum()),
                        "n_negative": int((series == 0).sum()),
                        "n_missing": int(series.isna().sum()),
                    }
                )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["label", "stratum", "level", "view"]).reset_index(drop=True)
    return out