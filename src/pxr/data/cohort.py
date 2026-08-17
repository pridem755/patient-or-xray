#-------------------------- Cohort Construction ----------------------------------

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pxr.config import Config

__all__ = [
    "CohortBuildError",
    "CohortBuildResult",
    "CohortFlow",
    "apply_uncertain_policy",
    "build_cohort",
    "canonical_image_key",
    "load_site_metadata",
    "verify_manifest",
]

_NO_FINDING = "No Finding"

#: Projection codes that denote a lateral film .
LATERAL_CODES: frozenset[str] = frozenset({"LATERAL", "LL", "RL"})

#: Projections analysed. Anything else frontal is counted as non-standard.
FRONTAL_CODES: frozenset[str] = frozenset({"AP", "PA"})


class CohortBuildError(RuntimeError):
    """Raised when the images and metadata cannot be assembled into a cohort."""


@dataclass
class CohortFlow:
    """Image counts at each construction step, for the Methods flow diagram."""

    site: str
    steps: list[dict] = field(default_factory=list)

    def record(
        self, step: str, n_images: int, n_patients: int, note: str = "", kind: str = "population"
    ) -> None:
        self.steps.append(
            {
                "site": self.site,
                "step": step,
                "kind": kind,
                "n_images": n_images,
                "n_patients": n_patients,
                "note": note,
            }
        )

    def record_frame(self, step: str, df: pd.DataFrame, note: str = "") -> None:
        self.record(
            step,
            len(df),
            int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0,
            note,
        )

    def to_frame(self) -> pd.DataFrame:
        """Flow as a table; ``n_dropped`` is computed over population steps only."""
        out = pd.DataFrame(self.steps)
        if out.empty:
            return out
        population = out["kind"] == "population"
        out["n_dropped"] = pd.NA
        out.loc[population, "n_dropped"] = (
            -out.loc[population, "n_images"].diff().fillna(0)
        ).astype("Int64")
        return out

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return self.to_frame().to_string(index=False)


@dataclass
class CohortBuildResult:
    cohort: pd.DataFrame
    flow: CohortFlow
    audit: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    def exclusion_summary(self) -> pd.DataFrame:
        """Counts per exclusion outcome, optionally read alongside the audit."""
        if "outcome" not in self.audit.columns:  # pragma: no cover - defensive
            return pd.DataFrame()
        return (
            self.audit.groupby("outcome", observed=True)
            .size()
            .rename("n_images")
            .reset_index()
            .sort_values("n_images", ascending=False, ignore_index=True)
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def canonical_image_key(
    values: pd.Series, depth: int = 1, pattern: str | None = None
) -> pd.Series:
    """Reduce site-specific image identifiers to one comparable form.
    """
    if depth < 1:
        raise CohortBuildError(f"image key depth must be >= 1, got {depth}")
    text = values.astype("string").str.replace(r"\\", "/", regex=True)

    if pattern is not None:
        extracted = text.str.extract(pattern)
        if extracted.shape[1] == 0:  # pragma: no cover - defensive
            raise CohortBuildError(f"image_key_pattern {pattern!r} has no capture groups")
        joined = extracted.iloc[:, 0].astype("string")
        for col in extracted.columns[1:]:
            joined = joined + "_" + extracted[col].astype("string")
        return joined.str.lower()

    tail = text.str.rsplit("/", n=depth).str[-depth:].str.join("/")
    return tail.str.replace(r"\.[A-Za-z0-9]+$", "", regex=True).str.lower()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise CohortBuildError(f"required metadata file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], site: str, what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CohortBuildError(
            f"{site}: {what} is missing expected column(s) {missing}. Present: "
            f"{sorted(df.columns)[:12]}..."
        )


def _require_schema(df: pd.DataFrame, cfg: Config, site: str) -> list[str]:
    expected = cfg.observation_schema(site)
    missing = [lab for lab in expected if lab not in df.columns]
    if missing:
        raise CohortBuildError(
            f"{site}: configured observation labels absent from the metadata: {missing}. "
            "Either the wrong file was loaded or labels.observation_schema is stale."
        )
    return expected


# --------------------------------------------------------------------------- #
# Site loaders -> common intermediate schema
#   patient_id, study_id, image_id, view_raw, is_frontal, sex, age, <labels>
# --------------------------------------------------------------------------- #


def _load_nih(cfg: Config, metadata_dir: Path, flow: CohortFlow) -> pd.DataFrame:
    """NIH ChestX-ray14: one file; findings expand from a pipe-delimited field."""
    spec = cfg.sites["nih"]
    raw = _read_csv(metadata_dir / spec["metadata_files"]["records"])
    _require_columns(
        raw,
        ["Image Index", "Finding Labels", "Patient ID", "Patient Age", "Patient Gender",
         "View Position"],
        "nih",
        "Data_Entry file",
    )
    flow.record("metadata_rows", len(raw), int(raw["Patient ID"].nunique()),
                spec["metadata_files"]["records"])

    schema = cfg.observation_schema("nih")
    findings = raw["Finding Labels"].fillna("").str.split("|")

    observed = {t for tokens in findings for t in tokens if t}
    unknown = sorted(observed - set(schema))
    if unknown:
        warnings.warn(
            f"nih: finding tokens present in the source but absent from the configured "
            f"observation schema (possible schema drift): {unknown}",
            stacklevel=2,
        )

    labels = pd.DataFrame(
        {lab: findings.apply(lambda fs, lab=lab: float(lab in fs)) for lab in schema},
        index=raw.index,
    )

    out = pd.DataFrame(
        {
            "patient_id": raw["Patient ID"].astype(str),
            "study_id": raw["Image Index"].astype(str),
            "image_id": raw["Image Index"].astype(str),
            "view_raw": raw["View Position"].astype("string").str.strip().str.upper(),
            # The NIH release contains frontal radiographs only.
            "is_frontal": True,
            "sex": raw["Patient Gender"].map({"M": "Male", "F": "Female"}).astype("string"),
            "age": pd.to_numeric(raw["Patient Age"], errors="coerce"),
        }
    )
    return pd.concat([out, labels], axis=1)


def _load_chexpert(cfg: Config, metadata_dir: Path, flow: CohortFlow) -> pd.DataFrame:
    spec = cfg.sites["chexpert"]
    wanted = [("train", spec["metadata_files"]["records"])]
    extra = spec["metadata_files"].get("records_extra")
    if extra:
        wanted.append(("valid", extra))

    frames = []
    for partition, name in wanted:
        frame = _read_csv(metadata_dir / name)  # configured files are required
        frame["source_partition"] = partition
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    _require_columns(raw, ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"], "chexpert",
                     "label file")
    _require_schema(raw, cfg, "chexpert")
    flow.record("metadata_rows", len(raw), 0, " + ".join(n for _, n in wanted))

    ids = raw["Path"].str.extract(r"(patient\d+)/(study\d+)/")
    if ids[0].isna().any():
        raise CohortBuildError(
            f"chexpert: {int(ids[0].isna().sum())} Path values do not match the expected "
            "'patientNNNNN/studyN/' structure, so identifiers cannot be derived"
        )

    out = pd.DataFrame(
        {
            "patient_id": ids[0].astype(str),
            "study_id": (ids[0] + "/" + ids[1]).astype(str),
            "image_id": raw["Path"].astype(str),
            "view_raw": raw["AP/PA"].astype("string").str.strip().str.upper(),
            "is_frontal": raw["Frontal/Lateral"].astype("string").str.strip().eq("Frontal"),
            "sex": raw["Sex"].astype("string").str.strip(),
            "age": pd.to_numeric(raw["Age"], errors="coerce"),
            "source_partition": raw["source_partition"].astype("string"),
        }
    )
    return pd.concat([out, raw[cfg.observation_schema("chexpert")].astype(float)], axis=1)


def _load_mimic(cfg: Config, metadata_dir: Path, flow: CohortFlow) -> pd.DataFrame:
    files = cfg.sites["mimic-cxr"]["metadata_files"]
    records = _read_csv(metadata_dir / files["records"])
    labels = _read_csv(metadata_dir / files["labels"])
    _require_columns(records, ["dicom_id", "subject_id", "study_id", "ViewPosition"],
                     "mimic-cxr", "records file")
    _require_columns(labels, ["subject_id", "study_id"], "mimic-cxr", "label file")
    _require_schema(labels, cfg, "mimic-cxr")

    patients_name = files.get("patients")
    if not patients_name:
        raise CohortBuildError(
            "mimic-cxr requires MIMIC-IV patients.csv for age and sex (the CXR release "
            "carries neither); set sites['mimic-cxr'].metadata_files.patients"
        )
    patients = _read_csv(metadata_dir / patients_name)
    _require_columns(patients, ["subject_id", "gender", "anchor_age", "anchor_year"],
                     "mimic-cxr", "MIMIC-IV patients file")

    flow.record("metadata_rows", len(records), int(records["subject_id"].nunique()),
                files["records"])

    probed = records.merge(
        labels, on=["subject_id", "study_id"], how="left", validate="m:1", indicator=True
    )
    unmatched = probed[probed["_merge"] == "left_only"]
    note = f"{len(unmatched)} image rows had no study-level label"
    if len(unmatched):
        note += f"; unmatched by projection: {unmatched['ViewPosition'].value_counts().to_dict()}"
    merged = probed[probed["_merge"] == "both"].drop(columns="_merge")
    flow.record("labels_joined", len(merged), int(merged["subject_id"].nunique()), note)

    probed = merged.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id", how="left", validate="m:1", indicator=True,
    )
    n_no_demo = int((probed["_merge"] == "left_only").sum())
    merged = probed[probed["_merge"] == "both"].drop(columns="_merge")
    flow.record(
        "demographics_joined", len(merged), int(merged["subject_id"].nunique()),
        f"{n_no_demo} image rows had no MIMIC-IV demographic record",
    )

    if "StudyDate" in merged.columns:
        study_year = pd.to_datetime(
            merged["StudyDate"].astype(str), format="%Y%m%d", errors="coerce"
        ).dt.year
        age = merged["anchor_age"] + (study_year - merged["anchor_year"])
    else:
        warnings.warn(
            "mimic-cxr: StudyDate absent, so age is the MIMIC-IV anchor age rather than "
            "age at the time of the study",
            stacklevel=2,
        )
        age = merged["anchor_age"]

    out = pd.DataFrame(
        {
            "patient_id": merged["subject_id"].astype(str),
            "study_id": merged["subject_id"].astype(str) + "/" + merged["study_id"].astype(str),
            "image_id": merged["dicom_id"].astype(str),
            "view_raw": merged["ViewPosition"].astype("string").str.strip().str.upper(),
            "is_frontal": ~merged["ViewPosition"]
            .astype("string")
            .str.strip()
            .str.upper()
            .isin(LATERAL_CODES),
            "sex": merged["gender"].map({"M": "Male", "F": "Female"}).astype("string"),
            "age": pd.to_numeric(age, errors="coerce"),
        }
    )
    return pd.concat([out, merged[cfg.observation_schema("mimic-cxr")].astype(float)], axis=1)


_LOADERS = {"nih": _load_nih, "chexpert": _load_chexpert, "mimic-cxr": _load_mimic}


def load_site_metadata(
    site: str, cfg: Config, metadata_dir: str | Path, flow: CohortFlow | None = None
) -> pd.DataFrame:
    """Load one site's metadata into the common intermediate schema."""
    if site not in _LOADERS:
        raise CohortBuildError(f"no loader for site {site!r}; known: {sorted(_LOADERS)}")
    return _LOADERS[site](cfg, Path(metadata_dir), flow or CohortFlow(site=site))


# --------------------------------------------------------------------------- #
# Shared pipeline
# --------------------------------------------------------------------------- #


def apply_uncertain_policy(df: pd.DataFrame, labels: Iterable[str], policy: str) -> pd.DataFrame:
    """Resolve CheXpert-style uncertain labels (-1).

    ``nan``     -1 becomes NaN; the image leaves that label's denominator (primary).
    ``u_zeros`` -1 becomes 0; the pre-registered sensitivity variant.
    """
    if policy not in ("nan", "u_zeros"):
        raise CohortBuildError(f"unknown uncertain_policy: {policy!r}")
    out = df.copy()
    replacement = float("nan") if policy == "nan" else 0.0
    for lab in labels:
        if lab in out.columns:
            out[lab] = out[lab].where(out[lab] != -1.0, replacement)
    return out


def _harmonise(df: pd.DataFrame, cfg: Config, site: str) -> pd.DataFrame:
    """Rename source label columns to analysis names, one-to-one only."""
    renames = cfg.harmonisation(site)
    if not renames:
        return df
    targets = list(renames.values())
    if len(targets) != len(set(targets)):
        raise CohortBuildError(
            f"{site}: harmonisation maps several source labels onto the same target "
            f"({renames}); a many-to-one mapping needs an explicit aggregation rule"
        )
    clashes = [new for old, new in renames.items() if new in df.columns and old in df.columns]
    if clashes:
        raise CohortBuildError(
            f"{site}: harmonisation would overwrite existing column(s) {clashes}"
        )
    return df.rename(columns=renames)


def _classify(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Assign each image exactly one outcome, so exclusion reasons never conflate."""
    view = df["view_raw"]
    frontal = df["is_frontal"].fillna(True).astype(bool)
    has_demo = df["sex"].isin(cfg.demographics["sex_categories"]) & df["age"].notna()
    rules = cfg.cohort
    if rules.get("adults_only"):
        has_demo &= df["age"].between(rules["age_min"], rules["age_max"])

    outcome = pd.Series("retained", index=df.index, dtype="object")
    outcome[~frontal] = "lateral"
    outcome[frontal & view.isna()] = "view_missing"
    outcome[frontal & view.notna() & ~view.isin(FRONTAL_CODES)] = "view_nonstandard"
    outcome[(outcome == "retained") & ~has_demo] = "demographics_missing"
    return outcome


def _assign_age_bin(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Bin age, requiring every eligible age to land in exactly one bin."""
    out = df.copy()
    out["age_bin"] = pd.cut(
        out["age"], bins=cfg.age_bin_edges, labels=cfg.age_bin_labels,
        right=False, include_lowest=True,
    ).astype("string")
    unbinned = out["age"].notna() & out["age_bin"].isna()
    if unbinned.any():
        raise CohortBuildError(
            f"{int(unbinned.sum())} rows have a valid age but no age_bin "
            f"(ages {sorted(out.loc[unbinned, 'age'].unique())[:5]}); "
            f"demographics.age_bins edges {cfg.age_bin_edges} do not cover the eligible range"
        )
    return out


def verify_manifest(
    site: str,
    cfg: Config,
    metadata_dir: str | Path,
    image_manifest: Iterable[str],
) -> pd.DataFrame:
    names = pd.Series(sorted(set(image_manifest)), dtype="string")
    if names.empty:
        raise CohortBuildError(f"{site}: image_manifest is empty")

    spec = cfg.sites[site]
    depth = int(spec.get("image_key_depth", 1))
    pattern = spec.get("image_key_pattern")

    held_keys = canonical_image_key(names, depth, pattern)
    n_unparsed = int(held_keys.isna().sum())
    held = set(held_keys.dropna())

    meta = load_site_metadata(site, cfg, metadata_dir, CohortFlow(site=site))
    meta_keys = canonical_image_key(meta["image_id"], depth, pattern)
    available = set(meta_keys.dropna())

    matched = held & available
    unmatched_examples = sorted(held - available)[:3]

    checks = [
        ("held_images", len(names), True, ""),
        (
            "manifest_parsed",
            f"{1 - n_unparsed / len(names):.1%}",
            n_unparsed == 0,
            f"{n_unparsed:,} names did not match image_key_pattern" if n_unparsed else "",
        ),
        (
            "keys_unique",
            len(held),
            len(held) == len(names) - n_unparsed,
            "distinct images collapse onto one key - the key is too coarse"
            if len(held) != len(names) - n_unparsed
            else "",
        ),
        ("metadata_rows", len(meta), len(meta) > 0, ""),
        (
            "held_matched",
            f"{len(matched) / len(held):.1%}" if held else "0.0%",
            bool(held) and len(matched) / len(held) >= 0.95,
            f"unmatched e.g. {unmatched_examples}" if unmatched_examples else "",
        ),
        (
            "metadata_covered",
            f"{len(matched) / len(available):.1%}" if available else "0.0%",
            True,
            "share of metadata rows that are held (low is expected after selection)",
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "value", "ok", "note"])


def build_cohort(
    site: str,
    cfg: Config,
    metadata_dir: str | Path,
    *,
    image_manifest: Iterable[str] | None = None,
    uncertain_policy: str | None = None,
    min_manifest_match_rate: float = 0.95,
) -> CohortBuildResult:
    policy = uncertain_policy or cfg.data["labels"]["uncertain_policy"]["primary"]
    flow = CohortFlow(site=site)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = load_site_metadata(site, cfg, metadata_dir, flow)
    build_warnings = [str(w.message) for w in caught]
    for message in build_warnings:
        warnings.warn(message, stacklevel=2)

    # ---- restrict to the images actually held -------------------------------- #
    if image_manifest is None:
        message = (
            f"{site}: no image_manifest supplied, so the cohort is defined by metadata "
            "rather than by the images on disk"
        )
        build_warnings.append(message)
        warnings.warn(message, stacklevel=2)
    else:
        manifest = pd.Series(sorted(set(image_manifest)), dtype="string")
        if manifest.empty:
            raise CohortBuildError(f"{site}: image_manifest is empty")
        depth = int(cfg.sites[site].get("image_key_depth", 1))
        pattern = cfg.sites[site].get("image_key_pattern")
        wanted_keys = canonical_image_key(manifest, depth, pattern)
        if pattern is not None and wanted_keys.isna().any():
            n_bad = int(wanted_keys.isna().sum())
            raise CohortBuildError(
                f"{site}: {n_bad:,} of {len(manifest):,} held image names do not match "
                f"image_key_pattern {pattern!r} (e.g. "
                f"{manifest[wanted_keys.isna()].head(2).tolist()}); the pattern and the "
                "image naming disagree"
            )
        wanted = set(wanted_keys.dropna())
        keys = canonical_image_key(df["image_id"], depth, pattern)
        df = df[keys.isin(wanted)]
        matched = set(canonical_image_key(df["image_id"], depth, pattern).dropna())
        rate = len(matched) / len(wanted)
        missing = len(wanted) - len(matched)
        flow.record_frame(
            "manifest_matched", df,
            f"{rate:.1%} of {len(wanted):,} held images matched a usable metadata row; "
            f"{missing:,} unmatched (includes any lost to the joins above)",
        )
        if rate < min_manifest_match_rate:
            raise CohortBuildError(
                f"{site}: only {rate:.1%} of the {len(wanted):,} held images matched a metadata "
                f"row (threshold {min_manifest_match_rate:.0%}). This usually means the manifest "
                "uses a different identifier representation than the metadata, not that "
                "metadata are missing."
            )

    df = _harmonise(df, cfg, site)
    observation = [cfg.harmonisation(site).get(lab, lab) for lab in cfg.observation_schema(site)]
    df = apply_uncertain_policy(df, observation, policy)

    # ---- classify every image, then split ------------------------------------ #
    df = df.reset_index(drop=True)
    df["outcome"] = _classify(df, cfg)

    audit_cols = ["patient_id", "study_id", "image_id", "view_raw", "is_frontal",
                  "sex", "age", "outcome"]
    if "source_partition" in df.columns:
        audit_cols.append("source_partition")
    audit = df[audit_cols].copy()
    audit.insert(3, "site", pd.array([site] * len(audit), dtype="string"))

    counts = df["outcome"].value_counts()
    for outcome in ("lateral", "view_missing", "view_nonstandard", "demographics_missing"):
        if counts.get(outcome, 0):
            flow.record(
                f"excluded_{outcome}",
                int(counts[outcome]),
                0,
                {
                    "lateral": "lateral films - excluded by study design",
                    "view_missing": "frontal but no recorded projection - data quality",
                    "view_nonstandard": "projection code outside {AP, PA}",
                    "demographics_missing": "no usable age or sex",
                }[outcome],
                kind="exclusion",
            )

    cohort = df[df["outcome"] == "retained"].drop(columns=["outcome", "is_frontal"]).copy()
    cohort = cohort.rename(columns={"view_raw": "view"})
    flow.record_frame("retained", cohort, "frontal AP/PA with usable demographics")

    # ---- integrity: one image per patient (selection already done upstream) --- #
    dup = cohort["patient_id"][cohort["patient_id"].duplicated(keep=False)]
    if len(dup):
        raise CohortBuildError(
            f"{site}: {dup.nunique():,} patients contribute more than one retained image "
            f"(e.g. {sorted(dup.unique())[:3]}). The held images are expected to be one per "
            "patient; this build does not select between them, so investigate the image set."
        )

    cohort = _assign_age_bin(cohort, cfg)

    keep = ["patient_id", "study_id", "image_id", "view", "sex", "age", "age_bin"]
    label_cols = [lab for lab in observation if lab in cohort.columns]
    extra = ["source_partition"] if "source_partition" in cohort.columns else []
    cohort = cohort[keep + extra + label_cols].copy()
    cohort.insert(3, "site", pd.array([site] * len(cohort), dtype="string"))
    cohort["config_hash"] = pd.array([cfg.config_hash] * len(cohort), dtype="string")

    for col in ("patient_id", "study_id", "image_id", "view", "sex"):
        cohort[col] = cohort[col].astype("string")
    cohort["age"] = cohort["age"].astype(float)

    flow.record_frame("final", cohort, f"{len(label_cols)} label columns retained")
    return CohortBuildResult(
        cohort=cohort.reset_index(drop=True),
        flow=flow,
        audit=audit.reset_index(drop=True),
        warnings=build_warnings,
    )