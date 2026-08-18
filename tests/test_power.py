from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxr.stats.power import (
    DEFAULT_CONTRASTS,
    PowerError,
    add_age_group,
    apply_coarsening_ladder,
    assign_tiers,
    cell_positive_counts,
    closed_form_power,
    evaluate_cells,
    minimum_detectable_effect,
    simulate_power,
)

LABELS = ["Cardiomegaly", "Pneumonia", "No Finding"]


def make_cohort(site="mimic-cxr", n=400, seed=0):
    """A synthetic cohort with a common label and a rare one."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "site": site,
        "view": rng.choice(["AP", "PA"], n),
        "sex": rng.choice(["Female", "Male"], n),
        "age": rng.integers(20, 90, n).astype(float),
        "age_bin": rng.choice(["18-39", "40-59", "60-79", "80+"], n),
        "Cardiomegaly": rng.binomial(1, 0.5, n).astype(float),   # plentiful
        "Pneumonia": rng.binomial(1, 0.01, n).astype(float),     # rare
        "No Finding": rng.binomial(1, 0.5, n).astype(float),
    })


class TestSimulatePower:
    def test_agrees_with_closed_form_in_the_large_sample_regime(self):
        """Where the normal approximation is sound, simulation must match it."""
        rng = np.random.default_rng(0)
        sim = simulate_power(3000, 3000, 0.30, 0.36, replicates=6000, rng=rng)
        exact = closed_form_power(3000, 3000, 0.30, 0.36)
        assert abs(sim - exact) < 0.03

    def test_power_increases_with_effect_size(self):
        rng = np.random.default_rng(1)
        small = simulate_power(500, 500, 0.30, 0.32, rng=rng)
        large = simulate_power(500, 500, 0.30, 0.45, rng=rng)
        assert large > small

    def test_power_increases_with_sample_size(self):
        rng = np.random.default_rng(2)
        few = simulate_power(50, 50, 0.30, 0.40, rng=rng)
        many = simulate_power(5000, 5000, 0.30, 0.40, rng=rng)
        assert many > few

    def test_null_effect_rejects_at_about_alpha(self):
        """With no true difference, rejection should sit near the nominal level."""
        rng = np.random.default_rng(3)
        power = simulate_power(2000, 2000, 0.30, 0.30, alpha=0.05,
                               replicates=8000, rng=rng)
        assert 0.02 < power < 0.09

    def test_empty_group_has_no_power(self):
        assert simulate_power(0, 100, 0.3, 0.5) == 0.0

    def test_deterministic_under_a_seed(self):
        a = simulate_power(200, 200, 0.3, 0.4, rng=np.random.default_rng(7))
        b = simulate_power(200, 200, 0.3, 0.4, rng=np.random.default_rng(7))
        assert a == b

    def test_invalid_rate_rejected(self):
        with pytest.raises(PowerError, match="rates must lie"):
            simulate_power(10, 10, 0.3, 1.5)


class TestMinimumDetectableEffect:
    def test_more_positives_detect_smaller_gaps(self):
        rng = np.random.default_rng(0)
        big = minimum_detectable_effect(5000, 5000, replicates=800, rng=rng)
        small = minimum_detectable_effect(150, 150, replicates=800, rng=rng)
        assert big < small

    def test_matches_closed_form_mde_when_large(self):
        """Cross-check against the analytic MDE in the regime it is valid."""
        rng = np.random.default_rng(1)
        mde = minimum_detectable_effect(4000, 4000, baseline=0.30,
                                        replicates=1500, rng=rng)
        analytic = next(
            d / 1000 for d in range(1, 500)
            if closed_form_power(4000, 4000, 0.30, 0.30 + d / 1000) >= 0.80
        )
        assert abs(mde - analytic) < 0.02

    def test_tiny_cells_are_undetectable(self):
        assert np.isnan(minimum_detectable_effect(1, 1))
        assert np.isnan(minimum_detectable_effect(0, 500))

    def test_result_lies_in_the_feasible_range(self):
        rng = np.random.default_rng(2)
        mde = minimum_detectable_effect(300, 300, baseline=0.30, rng=rng)
        assert 0 < mde < 0.71

    def test_invalid_baseline_rejected(self):
        with pytest.raises(PowerError, match="baseline"):
            minimum_detectable_effect(100, 100, baseline=0.0)


class TestCellCounts:
    def test_counts_positives_not_patients(self):
        cohort = make_cohort(n=400)
        counts = cell_positive_counts(cohort, LABELS)
        row = counts[(counts.label == "Cardiomegaly") & (counts.view == "AP")
                     & (counts.stratum == "sex")].iloc[0]
        expected = int(((cohort.Cardiomegaly == 1) & (cohort.view == "AP")
                        & (cohort.sex == "Female")).sum())
        assert row.n_a == expected

    def test_one_row_per_label_view_stratum(self):
        counts = cell_positive_counts(make_cohort(), LABELS)
        assert len(counts) == len(LABELS) * 2 * len(DEFAULT_CONTRASTS)

    def test_age_contrast_is_the_clinical_threshold_not_the_extreme_bands(self):
        """Extremes maximise any monotone effect while sitting in the thinnest cells."""
        counts = cell_positive_counts(make_cohort(), LABELS)
        age = counts[counts.stratum == "age_group"].iloc[0]
        assert (age.level_a, age.level_b) == ("<65", ">=65")
        assert "age_bin" not in set(counts.stratum)   # bands stay descriptive

    def test_missing_view_column_raises(self):
        with pytest.raises(PowerError, match="view"):
            cell_positive_counts(pd.DataFrame({"sex": ["Male"]}), LABELS)


class TestEvaluateCells:
    def test_rare_labels_are_not_powered(self):
        """A 10-point gap needs cells in the low thousands, not the low hundreds."""
        counts = cell_positive_counts(make_cohort(n=12000), LABELS)
        table = evaluate_cells(counts, replicates=400, seed=0)
        rare = table[table.label == "Pneumonia"]
        common = table[table.label == "Cardiomegaly"]
        assert not rare.powered.any()
        assert common.powered.any()

    def test_reproducible_across_runs(self):
        counts = cell_positive_counts(make_cohort(), LABELS)
        a = evaluate_cells(counts, replicates=300, seed=11)
        b = evaluate_cells(counts, replicates=300, seed=11)
        pd.testing.assert_frame_equal(a, b)

    def test_meaningful_effect_threshold_governs_the_verdict(self):
        counts = cell_positive_counts(make_cohort(n=600), LABELS)
        strict = evaluate_cells(counts, replicates=400, seed=0, meaningful_effect=0.02)
        lax = evaluate_cells(counts, replicates=400, seed=0, meaningful_effect=0.50)
        assert lax.powered.sum() >= strict.powered.sum()


class TestCoarseningLadder:
    LADDER = ["demote_label_to_exploratory", "report_descriptively_only"]

    def test_unknown_rung_rejected(self):
        with pytest.raises(PowerError, match="unknown coarsening"):
            apply_coarsening_ladder({"s": make_cohort()}, LABELS, ["invent_something"])

    def test_data_derived_age_pooling_is_rejected(self):
        """A median taken from the cohorts would make the contrast data-dependent."""
        with pytest.raises(PowerError, match="Age pooling is no longer a rung"):
            apply_coarsening_ladder({"s": make_cohort()}, LABELS,
                                    ["pool_age_bins_to_median_split"])

    def test_ladder_returns_a_table_and_a_rung_per_label(self):
        cohorts = {"mimic-cxr": make_cohort("mimic-cxr", 800, seed=1)}
        table, applied = apply_coarsening_ladder(
            cohorts, LABELS, self.LADDER, replicates=300, seed=0
        )
        assert set(applied) == set(LABELS)
        assert set(table.columns) >= {"site", "label", "view", "mde", "powered"}

    def test_hopeless_label_is_demoted(self):
        cohorts = {"mimic-cxr": make_cohort("mimic-cxr", 400, seed=2)}
        _, applied = apply_coarsening_ladder(
            cohorts, LABELS, self.LADDER, replicates=300, seed=0
        )
        assert applied["Pneumonia"] in {
            "demote_label_to_exploratory", "report_descriptively_only"
        }


class TestTierAssignment:
    def _table(self, powered_by_label):
        rows = []
        for label, flags in powered_by_label.items():
            for site, ok in flags.items():
                rows.append({"site": site, "label": label, "view": "AP",
                             "stratum": "sex", "mde": 0.05 if ok else 0.4,
                             "powered": ok})
        return pd.DataFrame(rows)

    def test_all_cells_powered_is_primary(self):
        table = self._table({"Cardiomegaly": {"a": True, "b": True}})
        tiers = assign_tiers(table, ["Cardiomegaly"])
        assert tiers.iloc[0].tier == "primary"

    def test_partial_power_is_exploratory(self):
        table = self._table({"Edema": {"a": True, "b": False}})
        tiers = assign_tiers(table, ["Edema"])
        assert tiers.iloc[0].tier == "exploratory"
        assert "1/2 sites" in tiers.iloc[0].reason

    def test_no_power_anywhere_is_descriptive(self):
        table = self._table({"Pneumonia": {"a": False, "b": False}})
        tiers = assign_tiers(table, ["Pneumonia"])
        assert tiers.iloc[0].tier == "descriptive"

    def test_secondary_lane_never_reaches_primary(self):
        """No Finding has inverted error semantics; it cannot join the family."""
        table = self._table({"No Finding": {"a": True, "b": True}})
        tiers = assign_tiers(table, ["No Finding"], secondary_lane=["No Finding"])
        assert tiers.iloc[0].tier == "exploratory"
        assert "inverted" in tiers.iloc[0].reason

    def test_absent_label_is_descriptive_not_an_error(self):
        tiers = assign_tiers(pd.DataFrame(columns=["site", "label", "powered", "mde"]),
                             ["Ghost"])
        assert tiers.iloc[0].tier == "descriptive"

    def test_tiers_are_ordered_primary_first(self):
        table = pd.concat([
            self._table({"Pneumonia": {"a": False}}),
            self._table({"Cardiomegaly": {"a": True}}),
        ])
        tiers = assign_tiers(table, ["Pneumonia", "Cardiomegaly"])
        assert list(tiers.tier) == ["primary", "descriptive"]


class TestInferentialSites:
    """A descriptive site must not demote a label powered where it counts."""

    def _table(self, flags):
        return pd.DataFrame([
            {"site": site, "label": label, "view": "AP", "stratum": "sex",
             "mde": 0.05 if ok else 0.4, "powered": ok}
            for label, sites in flags.items() for site, ok in sites.items()
        ])

    def test_descriptive_site_does_not_gate(self):
        table = self._table({"Cardiomegaly": {"mimic-cxr": True, "chexpert": True,
                                              "nih": False}})
        without = assign_tiers(table, ["Cardiomegaly"])
        with_gate = assign_tiers(table, ["Cardiomegaly"],
                                 inferential_sites=["mimic-cxr", "chexpert"])
        assert without.iloc[0].tier == "exploratory"      # NIH drags it down
        assert with_gate.iloc[0].tier == "primary"        # gated on the two that count

    def test_excluded_site_cells_are_still_reported(self):
        table = self._table({"Cardiomegaly": {"mimic-cxr": True, "chexpert": True,
                                              "nih": False}})
        tiers = assign_tiers(table, ["Cardiomegaly"],
                             inferential_sites=["mimic-cxr", "chexpert"])
        assert tiers.iloc[0].n_cells == 3          # all cells counted
        assert tiers.iloc[0].n_gating_cells == 2   # only two decided the tier

    def test_failure_at_an_inferential_site_still_demotes(self):
        table = self._table({"Edema": {"mimic-cxr": False, "chexpert": True,
                                       "nih": True}})
        tiers = assign_tiers(table, ["Edema"], inferential_sites=["mimic-cxr", "chexpert"])
        assert tiers.iloc[0].tier == "exploratory"

    def test_unknown_inferential_site_raises(self):
        table = self._table({"Edema": {"mimic-cxr": True}})
        with pytest.raises(PowerError, match="no cells from inferential_sites"):
            assign_tiers(table, ["Edema"], inferential_sites=["padchest"])


class TestAgeGroup:
    """The inferential age split is a fixed clinical value, not a data quantile."""

    def test_threshold_split_is_correct(self):
        df = pd.DataFrame({"age": [30.0, 64.9, 65.0, 90.0]})
        groups = list(add_age_group(df)["age_group"])
        assert groups == ["<65", "<65", ">=65", ">=65"]

    def test_threshold_does_not_depend_on_the_data(self):
        """Two cohorts with very different age distributions split identically."""
        young = add_age_group(pd.DataFrame({"age": [20.0, 30.0, 40.0]}))
        old = add_age_group(pd.DataFrame({"age": [70.0, 80.0, 90.0]}))
        assert set(young["age_group"]) == {"<65"}
        assert set(old["age_group"]) == {">=65"}

    def test_missing_age_column_raises(self):
        with pytest.raises(PowerError, match="age"):
            add_age_group(pd.DataFrame({"sex": ["Male"]}))

    def test_derived_automatically_when_absent(self):
        cohort = make_cohort().drop(columns=["age_bin"])
        counts = cell_positive_counts(cohort, LABELS)
        assert "age_group" in set(counts.stratum)