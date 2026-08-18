from __future__ import annotations

import pandas as pd
import pytest

from pxr.data.contracts import (
    ContractError,
    Severity,
    cohort_integrity_stats,
    inferential_cell_table,
    validate_primary_cohort,
    validate_repeated_image_cohort,
)

ANALYSIS = ["Cardiomegaly", "Pleural Effusion", "No Finding"]
OBSERVED = ["Cardiomegaly", "Pleural Effusion", "Atelectasis", "No Finding"]
HASH = "abc123def456"


def make_cohort(n: int = 6) -> pd.DataFrame:
    """A minimal cohort satisfying every ERROR-level contract."""
    return pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(n)],
            "study_id": [f"s{i}" for i in range(n)],
            "image_id": [f"img{i}" for i in range(n)],
            "site": pd.array(["nih"] * n, dtype="string"),
            "view": pd.array(["AP", "PA"] * (n // 2), dtype="string"),
            "sex": pd.array(["Male", "Female"] * (n // 2), dtype="string"),
            "age": [45.0 + i for i in range(n)],
            "age_bin": pd.array(["40-59"] * n, dtype="string"),
            "config_hash": pd.array([HASH] * n, dtype="string"),
            "Cardiomegaly": [1.0, 0.0, 1.0, 0.0, None, 0.0],
            "Pleural Effusion": [0.0, 1.0, 0.0, 1.0, 0.0, None],
            "Atelectasis": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "No Finding": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        }
    )


def validate(df, **kw):
    kw.setdefault("observation_labels", OBSERVED)
    kw.setdefault("expected_site", "nih")
    kw.setdefault("expected_config_hash", HASH)
    kw.setdefault("strict", False)
    return validate_primary_cohort(df, ANALYSIS, **kw)


def codes(report) -> set[str]:
    return {v.check for v in report.violations}


class TestValidCohort:
    def test_clean_cohort_passes(self):
        assert validate(make_cohort()).ok

    def test_strict_returns_report_when_clean(self):
        r = validate_primary_cohort(
            make_cohort(), ANALYSIS, observation_labels=OBSERVED,
            expected_site="nih", expected_config_hash=HASH, strict=True,
        )
        assert r.n_rows == 6

    def test_report_renders_as_table(self):
        assert isinstance(validate(make_cohort()).to_frame(), pd.DataFrame)


class TestFrameShape:
    def test_empty_cohort_rejected(self):
        """A failed join must never masquerade as a valid cohort."""
        r = validate(make_cohort().iloc[0:0])
        assert "frame.empty" in codes(r)
        assert not r.ok

    def test_duplicate_columns_rejected(self):
        df = make_cohort()
        df = pd.concat([df, df[["sex"]]], axis=1)
        r = validate(df)
        assert "frame.duplicate_columns" in codes(r)
        assert not r.ok


class TestProvenance:
    def test_wrong_but_constant_site_rejected(self):
        df = make_cohort()
        df["site"] = pd.array(["mimic"] * len(df), dtype="string")
        r = validate(df)
        assert "provenance.site_mismatch" in codes(r)

    def test_stale_config_hash_rejected(self):
        df = make_cohort()
        df["config_hash"] = pd.array(["0123456789ab"] * len(df), dtype="string")
        r = validate(df)
        assert "provenance.config_hash_mismatch" in codes(r)

    def test_mixed_site_rejected(self):
        df = make_cohort()
        df.loc[0, "site"] = "chexpert"
        r = validate(df)
        assert "provenance.site_not_constant" in codes(r)

    def test_malformed_hash_format_rejected(self):
        df = make_cohort()
        df["config_hash"] = pd.array(["NOT-A-HASH"] * len(df), dtype="string")
        r = validate(df, expected_config_hash=None)
        assert "provenance.config_hash_format" in codes(r)


class TestIdentifiers:
    def test_duplicate_patient_rejected(self):
        df = make_cohort()
        df.loc[1, "patient_id"] = df.loc[0, "patient_id"]
        assert "identifiers.duplicate_patient_id" in codes(validate(df))

    def test_duplicate_study_rejected(self):
        df = make_cohort()
        df.loc[1, "study_id"] = df.loc[0, "study_id"]
        r = validate(df)
        assert "identifiers.duplicate_study_id" in codes(r)
        assert "identifiers.study_spans_patients" in codes(r)

    def test_blank_identifier_rejected(self):
        df = make_cohort()
        df.loc[0, "patient_id"] = "   "
        assert "identifiers.blank" in codes(validate(df))

    def test_empty_string_identifier_rejected(self):
        df = make_cohort()
        df.loc[0, "image_id"] = ""
        assert "identifiers.blank" in codes(validate(df))


class TestExposureAndDemographics:
    def test_lateral_view_rejected(self):
        df = make_cohort()
        df.loc[0, "view"] = "LATERAL"
        assert "view.invalid" in codes(validate(df))

    def test_unexpected_sex_rejected(self):
        df = make_cohort()
        df.loc[0, "sex"] = "Unknown"
        assert "demographics.sex" in codes(validate(df))

    def test_implausible_age_rejected(self):
        df = make_cohort()
        df.loc[0, "age"] = 7.0
        assert "demographics.age_range" in codes(validate(df))

    def test_boolean_age_rejected(self):
        df = make_cohort()
        df["age"] = [True] * len(df)
        assert "columns.dtype" in codes(validate(df))

    def test_null_view_reported_once(self):
        """Missingness is owned by the column contract, not double-reported."""
        df = make_cohort()
        df.loc[0, "view"] = None
        r = validate(df)
        assert "columns.null" in codes(r)
        assert "view.invalid" not in codes(r)


class TestLabelIntegrity:
    def test_uncertain_minus_one_rejected(self):
        df = make_cohort()
        df.loc[0, "Cardiomegaly"] = -1.0
        assert "labels.values" in codes(validate(df))

    def test_mixed_type_labels_do_not_crash_validator(self):
        """The validator must never fail while describing a failure."""
        df = make_cohort()
        df["Cardiomegaly"] = df["Cardiomegaly"].astype(object)
        df.loc[0, "Cardiomegaly"] = "bad"
        df.loc[1, "Cardiomegaly"] = 2
        r = validate(df)  # must not raise TypeError
        v = next(x for x in r.violations if x.check == "labels.values")
        assert v.n_affected == 2

    def test_no_finding_contradiction_rejected(self):
        """The exact defect that invalidated the previous cohort build."""
        df = make_cohort()
        df.loc[4, "Cardiomegaly"] = 1.0  # row 4 already has No Finding = 1
        v = next(
            x for x in validate(df).violations
            if x.check == "labels.no_finding_contradiction"
        )
        assert v.severity is Severity.ERROR and v.n_affected == 1

    def test_contradiction_found_in_unmodelled_label(self):
        """A contradiction must not hide in a label excluded from the analysis set."""
        df = make_cohort()
        df.loc[4, "Atelectasis"] = 1.0  # Atelectasis is observed but not analysed
        assert "labels.no_finding_contradiction" in codes(validate(df))

    def test_missing_observation_schema_warns(self):
        df = make_cohort()
        r = validate_primary_cohort(
            df, ANALYSIS, expected_site="nih", expected_config_hash=HASH, strict=False
        )
        assert "labels.observation_schema_not_supplied" in codes(r)
        assert r.ok  # warning only

    def test_cohort_without_no_finding_is_fine(self):
        labels = ["Cardiomegaly", "Pleural Effusion"]
        df = make_cohort().drop(columns=["No Finding"])
        r = validate_primary_cohort(
            df, labels, observation_labels=labels + ["Atelectasis"],
            expected_site="nih", expected_config_hash=HASH, strict=False,
        )
        assert "labels.no_finding_contradiction" not in codes(r)


class TestInferentialCells:
    def test_thin_demographic_cell_warns(self):
        """label x view x sex is the estimable cell, not label x view."""
        r = validate(make_cohort(), min_positives=25)
        assert r.ok
        thin = [v for v in r.warnings if v.check == "power.thin_cell"]
        assert any("sex=" in v.message for v in thin)
        assert any("age_bin=" in v.message for v in thin)

    def test_missing_age_bin_warns(self):
        df = make_cohort().drop(columns=["age_bin"])
        r = validate(df)
        assert "power.age_bin_missing" in codes(r)
        assert r.ok

    def test_no_warning_when_threshold_disabled(self):
        r = validate(make_cohort(), min_positives=0)
        assert not any(v.check == "power.thin_cell" for v in r.violations)


class TestCohortShapes:
    def test_repeated_patients_rejected_by_primary(self):
        df = make_cohort()
        df.loc[1, "patient_id"] = df.loc[0, "patient_id"]
        assert not validate(df).ok

    def test_repeated_patients_allowed_by_sensitivity_contract(self):
        df = make_cohort()
        df.loc[1, "patient_id"] = df.loc[0, "patient_id"]
        df.loc[1, "study_id"] = df.loc[0, "study_id"]
        r = validate_repeated_image_cohort(
            df, ANALYSIS, observation_labels=OBSERVED,
            expected_site="nih", expected_config_hash=HASH, strict=False,
        )
        assert r.ok

    def test_duplicate_images_rejected_by_both(self):
        df = make_cohort()
        df.loc[1, "image_id"] = df.loc[0, "image_id"]
        r = validate_repeated_image_cohort(
            df, ANALYSIS, observation_labels=OBSERVED,
            expected_site="nih", expected_config_hash=HASH, strict=False,
        )
        assert "identifiers.duplicate_image_id" in codes(r)


class TestStrictMode:
    def test_strict_raises_on_error(self):
        df = make_cohort()
        df.loc[0, "view"] = "LATERAL"
        with pytest.raises(ContractError) as exc:
            validate_primary_cohort(
                df, ANALYSIS, observation_labels=OBSERVED,
                expected_site="nih", expected_config_hash=HASH, strict=True,
            )
        assert "view.invalid" in str(exc.value)

    def test_strict_tolerates_warnings(self):
        validate_primary_cohort(
            make_cohort(), ANALYSIS, observation_labels=OBSERVED,
            expected_site="nih", expected_config_hash=HASH, strict=True,
        )


class TestReportingTables:
    def test_integrity_counts_match_construction(self):
        stats = cohort_integrity_stats(make_cohort(), ANALYSIS)
        row = stats[(stats.label == "Cardiomegaly") & (stats.view == "AP")].iloc[0]
        assert (row.n_positive, row.n_missing, row.n_rows) == (2, 1, 3)

    def test_inferential_table_has_demographic_rows(self):
        tab = inferential_cell_table(make_cohort(), ANALYSIS)
        assert set(tab.stratum.unique()) == {"sex", "age_bin"}
        assert (tab.site == "nih").all()
        sub = tab[(tab.label == "Cardiomegaly") & (tab.stratum == "sex")]
        assert sub.n_positive.sum() == 2


class TestNoFindingDerivation:
    """No Finding is derived, so it must agree with the diseases it derives from."""

    def test_consistent_derivation_passes(self):
        df = make_cohort()
        # rows 0 and 2 are Cardiomegaly-positive, 1 and 3 Effusion-positive
        df["No Finding"] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        assert validate(df).ok

    def test_no_finding_with_positive_disease_rejected(self):
        df = make_cohort()
        df["No Finding"] = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]   # row 0 has Cardiomegaly=1
        assert "labels.no_finding_derivation" in codes(validate(df))

    def test_disease_free_row_marked_not_no_finding_rejected(self):
        df = make_cohort()
        df["No Finding"] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]   # row 4 has no positive
        assert "labels.no_finding_derivation" in codes(validate(df))

    def test_missing_no_finding_column_is_not_an_error(self):
        labels = ["Cardiomegaly", "Pleural Effusion"]
        df = make_cohort().drop(columns=["No Finding"])
        report = validate_primary_cohort(
            df, labels, observation_labels=labels + ["Atelectasis"],
            expected_site="nih", expected_config_hash=HASH, strict=False,
        )
        assert "labels.no_finding_derivation" not in {v.check for v in report.violations}