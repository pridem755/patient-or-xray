from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxr.data.splits import (
    DEFAULT_MIN_VAL_STRATUM,
    EXTERNAL_FOLD,
    SplitError,
    assemble_out_of_fold,
    assign_folds,
    cell_balance_report,
    fold_balance_report,
    fold_membership,
    plan_validation_allocation,
    validate_folds,
)

STRATA = ["view", "sex", "Cardiomegaly"]


def make_cohort(site="mimic-cxr", n=2000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 95, n).astype(float)
    return pd.DataFrame({
        "patient_id": [f"{site}-{i}" for i in range(n)],
        "site": site,
        "view": rng.choice(["AP", "PA"], n, p=[0.42, 0.58]),
        "sex": rng.choice(["Female", "Male"], n),
        "age": age,
        "age_bin": pd.cut(age, [18, 40, 60, 80, 121],
                          labels=["18-39", "40-59", "60-79", "80+"],
                          right=False).astype(str),
        "Cardiomegaly": rng.binomial(1, 0.13, n).astype(float),
        "Atelectasis": rng.binomial(1, 0.13, n).astype(float),
    })


def folded(**kw):
    return assign_folds(make_cohort(**kw), is_training_site=True, stratify_by=STRATA)


class TestAssignment:
    def test_every_patient_gets_exactly_one_fold(self):
        out = folded()
        assert out["fold"].notna().all()
        assert out.groupby("patient_id")["fold"].nunique().max() == 1

    def test_folds_are_near_equal_in_size(self):
        sizes = folded(n=5000)["fold"].value_counts()
        assert (sizes.max() - sizes.min()) / sizes.mean() < 0.02

    def test_assignment_is_deterministic(self):
        a = assign_folds(make_cohort(), is_training_site=True, stratify_by=STRATA, seed=7)
        b = assign_folds(make_cohort(), is_training_site=True, stratify_by=STRATA, seed=7)
        pd.testing.assert_series_equal(a["fold"], b["fold"])

    def test_remainder_patients_are_not_biased_to_low_folds(self):
        """Dealing always from fold 0 would skew composition while sizes still looked even."""
        counts = np.zeros(5)
        for seed in range(40):
            cohort = make_cohort(n=137, seed=seed)      # deliberately not divisible by 5
            out = assign_folds(cohort, is_training_site=True, stratify_by=STRATA, seed=seed)
            sizes = out["fold"].value_counts().reindex(range(5), fill_value=0).to_numpy()
            counts += sizes == sizes.max()               # which folds took the remainder
        share = counts / counts.sum()
        assert share.max() - share.min() < 0.20, f"remainder skew: {share}"

    def test_non_training_site_is_wholly_external(self):
        out = assign_folds(make_cohort("nih"), is_training_site=False, stratify_by=STRATA)
        assert set(out["fold"]) == {EXTERNAL_FOLD}

    def test_stratified_variables_stay_even(self):
        cohort = make_cohort(n=10000)
        out = assign_folds(cohort, is_training_site=True, stratify_by=STRATA)
        for col, value in [("view", "AP"), ("sex", "Female"), ("Cardiomegaly", 1)]:
            overall = (cohort[col] == value).mean()
            for _, block in out.groupby("fold"):
                assert abs((block[col] == value).mean() - overall) < 0.02

    def test_prevalence_is_never_engineered(self):
        cohort = make_cohort(n=5000)
        out = assign_folds(cohort, is_training_site=True, stratify_by=STRATA)
        assert (out["Cardiomegaly"] == 1).mean() == (cohort["Cardiomegaly"] == 1).mean()
        assert len(out) == len(cohort)


class TestNestedMembership:
    def test_development_set_is_eighty_percent(self):
        cohort = folded(n=5000)
        role = fold_membership(0, cohort, stratify_by=STRATA)
        assert abs((role != "test").mean() - 0.80) < 0.02

    def test_training_share_exceeds_a_whole_validation_fold(self):
        """Spending a fold on validation would leave 60%; nesting leaves ~68%."""
        cohort = folded(n=5000)
        role = fold_membership(0, cohort, stratify_by=STRATA, val_fraction=0.15)
        assert 0.64 < (role == "train").mean() < 0.72

    def test_validation_never_touches_the_test_fold(self):
        """Otherwise the operating threshold is tuned on the patients it is judged against."""
        cohort = folded(n=4000)
        for k in range(5):
            role = fold_membership(k, cohort, stratify_by=STRATA)
            val_folds = set(cohort.loc[role == "val", "fold"])
            assert k not in val_folds

    def test_roles_partition_the_cohort(self):
        cohort = folded(n=3000)
        role = fold_membership(2, cohort, stratify_by=STRATA)
        assert set(role) == {"train", "val", "test"}
        assert len(role) == len(cohort)

    def test_every_fold_tests_once_across_the_loop(self):
        cohort = folded(n=3000)
        tested = pd.Series(0, index=cohort.index)
        for k in range(5):
            tested += (fold_membership(k, cohort, stratify_by=STRATA) == "test").astype(int)
        assert (tested == 1).all()

    def test_validation_is_deterministic(self):
        cohort = folded(n=2000)
        a = fold_membership(1, cohort, stratify_by=STRATA, seed=5)
        b = fold_membership(1, cohort, stratify_by=STRATA, seed=5)
        pd.testing.assert_series_equal(a, b)

    def test_validation_share_is_configurable(self):
        cohort = folded(n=5000)
        small = fold_membership(0, cohort, stratify_by=STRATA, val_fraction=0.10)
        large = fold_membership(0, cohort, stratify_by=STRATA, val_fraction=0.30)
        assert (large == "val").sum() > (small == "val").sum()

    def test_out_of_range_fold_raises(self):
        with pytest.raises(SplitError, match="outside range"):
            fold_membership(9, folded(), stratify_by=STRATA)

    def test_unassigned_cohort_raises(self):
        with pytest.raises(SplitError, match="assign_folds first"):
            fold_membership(0, make_cohort(), stratify_by=STRATA)

    def test_invalid_val_fraction_raises(self):
        with pytest.raises(SplitError, match="val_fraction"):
            fold_membership(0, folded(), stratify_by=STRATA, val_fraction=1.5)


class TestGuards:
    def test_empty_cohort_rejected(self):
        with pytest.raises(SplitError, match="empty"):
            assign_folds(make_cohort().iloc[0:0], is_training_site=True, stratify_by=STRATA)

    def test_duplicate_patients_rejected(self):
        cohort = pd.concat([make_cohort(n=20)] * 2, ignore_index=True)
        with pytest.raises(SplitError, match="not unique"):
            assign_folds(cohort, is_training_site=True, stratify_by=STRATA)

    def test_empty_stratification_rejected(self):
        """Which variables are balanced is a pre-specified choice, not a default."""
        with pytest.raises(SplitError, match="pre-specified"):
            assign_folds(make_cohort(), is_training_site=True, stratify_by=[])

    def test_missing_stratify_column_rejected(self):
        with pytest.raises(SplitError, match="absent column"):
            assign_folds(make_cohort(), is_training_site=True, stratify_by=["race"])

    def test_single_fold_rejected(self):
        with pytest.raises(SplitError, match="at least 2"):
            assign_folds(make_cohort(), is_training_site=True, stratify_by=STRATA, n_folds=1)


class TestValidation:
    def _folds(self):
        return {
            "mimic-cxr": folded(site="mimic-cxr"),
            "chexpert": folded(site="chexpert", seed=1),
            "nih": assign_folds(make_cohort("nih"), is_training_site=False,
                                stratify_by=STRATA),
        }

    def test_valid_folds_pass(self):
        report = validate_folds(self._folds(), training_sites=["mimic-cxr", "chexpert"])
        assert report.ok.all(), report.to_string(index=False)

    def test_patient_in_two_folds_raises(self):
        folds = self._folds()
        leaked = folds["mimic-cxr"].copy()
        dup = leaked.iloc[[0]].copy()
        dup["fold"] = (dup["fold"] + 1) % 5
        folds["mimic-cxr"] = pd.concat([leaked, dup], ignore_index=True)
        with pytest.raises(SplitError, match="more than one fold"):
            validate_folds(folds, training_sites=["mimic-cxr", "chexpert"])

    def test_external_site_with_a_real_fold_is_flagged(self):
        folds = self._folds()
        folds["nih"].loc[0, "fold"] = 2
        report = validate_folds(folds, training_sites=["mimic-cxr", "chexpert"])
        row = report[(report.site == "nih") & (report.check == "fold_membership")].iloc[0]
        assert not row.ok and "wholly external" in row.note


class TestBalanceReports:
    def test_fold_report_marks_controlled_variables(self):
        report = fold_balance_report(folded(n=6000), ["Cardiomegaly", "Atelectasis"],
                                     stratify_by=STRATA)
        assert len(report) == 5
        assert "Cardiomegaly*" in report.columns    # starred: stratified on
        assert "Atelectasis" in report.columns       # unstarred: verified only
        assert any(c.startswith("age ") for c in report.columns)

    def test_uncontrolled_label_stays_close_across_folds(self):
        report = fold_balance_report(folded(n=20000), ["Atelectasis"], stratify_by=STRATA)
        assert report["Atelectasis"].max() - report["Atelectasis"].min() < 0.02

    def test_cell_report_counts_positives_per_fold(self):
        """Fold sizes say little; positives per demographic x view cell say everything."""
        report = cell_balance_report(folded(n=8000), ["Cardiomegaly"], strata=("sex",))
        assert {"label", "view", "stratum", "level", "min", "max", "spread"} <= set(report.columns)
        assert (report["min"] > 0).all()

    def test_cell_report_sorts_thinnest_first(self):
        report = cell_balance_report(folded(n=8000), ["Cardiomegaly", "Atelectasis"])
        assert report["min"].is_monotonic_increasing

    def test_cell_report_requires_a_view_column(self):
        with pytest.raises(SplitError, match="view"):
            cell_balance_report(folded().drop(columns=["view"]), ["Cardiomegaly"])


class TestLeakageGuarantees:
    """The three properties the whole out-of-fold design rests on.

    Each is stated directly rather than inferred from summary statistics, because a
    violation of any one would invalidate every downstream number while leaving the
    fold sizes looking perfectly reasonable.
    """

    def test_train_val_test_are_mutually_exclusive(self):
        cohort = folded(n=4000)
        for k in range(5):
            role = fold_membership(k, cohort, stratify_by=STRATA)
            train = set(cohort.loc[role == "train", "patient_id"])
            val = set(cohort.loc[role == "val", "patient_id"])
            test = set(cohort.loc[role == "test", "patient_id"])
            assert not train & val, f"fold {k}: patient in train and val"
            assert not train & test, f"fold {k}: patient in train and test"
            assert not val & test, f"fold {k}: patient in val and test"
            assert len(train | val | test) == len(cohort), f"fold {k}: patient unassigned"

    def test_every_patient_belongs_to_exactly_one_outer_fold(self):
        cohort = folded(n=4000)
        per_patient = cohort.groupby("patient_id")["fold"].nunique()
        assert (per_patient == 1).all()
        assert len(per_patient) == len(cohort)

    def test_every_patient_is_a_test_patient_exactly_once(self):
        """This is what makes the pooled table a genuine out-of-fold prediction set."""
        cohort = folded(n=4000)
        times_tested = pd.Series(0, index=cohort.index)
        for k in range(5):
            times_tested += (fold_membership(k, cohort, stratify_by=STRATA) == "test").astype(int)
        assert (times_tested == 1).all(), (
            f"{int((times_tested != 1).sum())} patient(s) tested a number of times other than one"
        )

    def test_no_patient_ever_trains_on_the_fold_that_tests_them(self):
        cohort = folded(n=3000)
        for k in range(5):
            role = fold_membership(k, cohort, stratify_by=STRATA)
            tested_here = cohort.loc[role == "test", "patient_id"]
            trained_here = set(cohort.loc[role.isin(["train", "val"]), "patient_id"])
            assert not set(tested_here) & trained_here


class TestValidationAllocation:
    """Small strata are handled by an explicit rule, not by rounding."""

    def test_tiny_strata_are_skipped_explicitly(self):
        development = pd.DataFrame({
            "view": ["AP"] * 3 + ["PA"] * 100,
            "sex": ["Female"] * 103,
        })
        plan = plan_validation_allocation(
            development, stratify_by=["view", "sex"], val_fraction=0.15
        )
        tiny = plan[plan["n_patients"] == 3].iloc[0]
        assert tiny["skipped"] and tiny["n_val"] == 0

    def test_skipping_is_visible_in_the_plan(self):
        """A silent skip is the failure mode; the plan must record it."""
        development = pd.DataFrame({"view": ["AP"] * 2 + ["PA"] * 50,
                                    "sex": ["Male"] * 52})
        plan = plan_validation_allocation(
            development, stratify_by=["view", "sex"], val_fraction=0.15
        )
        assert plan["skipped"].any()
        assert {"stratum", "n_patients", "exact", "n_val", "skipped"} <= set(plan.columns)

    def test_total_matches_the_target_rather_than_drifting_low(self):
        """Independent rounding down in each stratum would lose validation patients."""
        cohort = folded(n=5000)
        development = cohort[cohort["fold"] != 0]
        plan = plan_validation_allocation(
            development, stratify_by=STRATA, val_fraction=0.15
        )
        target = round(len(development) * 0.15)
        assert abs(int(plan["n_val"].sum()) - target) <= 1

    def test_min_stratum_size_is_configurable(self):
        development = pd.DataFrame({"view": ["AP"] * 8 + ["PA"] * 80,
                                    "sex": ["Female"] * 88})
        strict = plan_validation_allocation(
            development, stratify_by=["view", "sex"], val_fraction=0.15,
            min_stratum_size=20,
        )
        lax = plan_validation_allocation(
            development, stratify_by=["view", "sex"], val_fraction=0.15,
            min_stratum_size=5,
        )
        assert strict["skipped"].sum() > lax["skipped"].sum()

    def test_default_threshold_is_stated_not_implicit(self):
        assert DEFAULT_MIN_VAL_STRATUM >= 2


class TestOutOfFoldAssembly:
    """The pooled table is what every downstream number is computed from."""

    def _setup(self, n=1000):
        cohort = folded(n=n)
        preds = {
            k: pd.DataFrame({
                "patient_id": cohort.loc[cohort.fold == k, "patient_id"].values,
                "score": 0.5,
            })
            for k in range(5)
        }
        return cohort, preds

    def test_pooled_table_covers_every_patient_exactly_once(self):
        cohort, preds = self._setup()
        pooled = assemble_out_of_fold(preds, cohort)
        assert len(pooled) == len(cohort)
        assert not pooled["patient_id"].duplicated().any()
        assert set(pooled["patient_id"]) == set(cohort["patient_id"])

    def test_each_prediction_records_its_fold(self):
        cohort, preds = self._setup()
        pooled = assemble_out_of_fold(preds, cohort)
        assigned = cohort.set_index("patient_id")["fold"]
        assert (pooled["patient_id"].map(assigned) == pooled["fold"]).all()

    def test_duplicate_prediction_raises(self):
        cohort, preds = self._setup()
        preds[1] = pd.concat([preds[1], preds[0].iloc[[0]]], ignore_index=True)
        with pytest.raises(SplitError, match="more than once"):
            assemble_out_of_fold(preds, cohort)

    def test_missing_patient_raises(self):
        """A silently excluded patient would bias every estimate that follows."""
        cohort, preds = self._setup()
        preds[2] = preds[2].iloc[1:]
        with pytest.raises(SplitError, match="no prediction"):
            assemble_out_of_fold(preds, cohort)

    def test_prediction_from_the_wrong_fold_raises(self):
        cohort, preds = self._setup()
        swapped = preds[0].copy()
        preds[0] = preds[1].copy()
        preds[1] = swapped
        with pytest.raises(SplitError, match="did not hold that patient out"):
            assemble_out_of_fold(preds, cohort)

    def test_missing_fold_raises(self):
        cohort, preds = self._setup()
        del preds[3]
        with pytest.raises(SplitError, match="no predictions for fold"):
            assemble_out_of_fold(preds, cohort)