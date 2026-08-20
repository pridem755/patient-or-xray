from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxr.stats.fairness import (
    FairnessError,
    analyse_label,
    benjamini_hochberg,
    delta_gap,
    false_negative_rate,
    group_gap,
    holm_correct,
    interaction_model,
    positivity_report,
    standardise_rate,
    within_view_gaps,
)

LABEL = "Cardiomegaly"


def synth(
    n=8000,
    *,
    ap_share_young=0.25,
    ap_share_old=0.70,
    fnr_ap_young=0.25, fnr_pa_young=0.15,
    fnr_ap_old=0.25, fnr_pa_old=0.15,
    prevalence=0.30,
    seed=0,
):
    """A cohort whose error rates are set by view, and optionally by age.

    With the four ``fnr_*`` values equal across age, any raw gap can only come from
    the differing AP shares - so standardisation must remove it entirely. Setting
    them unequal plants a genuine age effect that standardisation must preserve.
    """
    rng = np.random.default_rng(seed)
    old = rng.binomial(1, 0.5, n).astype(bool)
    ap_share = np.where(old, ap_share_old, ap_share_young)
    is_ap = rng.random(n) < ap_share

    truth = rng.binomial(1, prevalence, n).astype(float)
    fnr = np.where(
        old,
        np.where(is_ap, fnr_ap_old, fnr_pa_old),
        np.where(is_ap, fnr_ap_young, fnr_pa_young),
    )
    missed = rng.random(n) < fnr
    predicted = np.where((truth == 1) & missed, 0.0, truth)

    return pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "view": np.where(is_ap, "AP", "PA"),
        "age_group": np.where(old, ">=65", "<65"),
        "sex": rng.choice(["Female", "Male"], n),
        f"{LABEL}_true": truth,
        f"{LABEL}_predicted": predicted,
    })


AGE = ("<65", ">=65")


class TestRates:
    def test_rate_counts_only_positives(self):
        """The denominator is patients who have the disease - it is a *missed* rate."""
        frame = pd.DataFrame({
            f"{LABEL}_true": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            f"{LABEL}_predicted": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        })
        assert false_negative_rate(frame, LABEL) == pytest.approx(0.5)

    def test_no_positives_gives_nan(self):
        frame = pd.DataFrame({f"{LABEL}_true": [0.0, 0.0],
                              f"{LABEL}_predicted": [0.0, 0.0]})
        assert np.isnan(false_negative_rate(frame, LABEL))

    def test_missing_columns_raise(self):
        with pytest.raises(FairnessError, match="lacks"):
            false_negative_rate(pd.DataFrame({"x": [1]}), LABEL)


class TestGaps:
    def test_gap_sign_follows_the_declared_contrast(self):
        """gap = rate_b - rate_a, so positive means group b is missed more often."""
        frame = synth(fnr_ap_old=0.40, fnr_pa_old=0.40,
                      fnr_ap_young=0.10, fnr_pa_young=0.10, n=6000)
        estimate = group_gap(frame, LABEL, "age_group", AGE)
        assert estimate.gap > 0
        assert estimate.rate_b > estimate.rate_a

    def test_gap_records_positive_counts_not_row_counts(self):
        frame = synth(n=4000, prevalence=0.25)
        estimate = group_gap(frame, LABEL, "age_group", AGE)
        expected = int(((frame.age_group == "<65") & (frame[f"{LABEL}_true"] == 1)).sum())
        assert estimate.n_a == expected

    def test_within_view_gaps_cover_each_view(self):
        gaps = within_view_gaps(synth(), LABEL, "age_group", AGE)
        assert {g.view for g in gaps} == {"AP", "PA"}

    def test_within_view_gap_vanishes_when_the_effect_is_pure_acquisition(self):
        """Same rates by view for both ages: within a view there is nothing to find."""
        frame = synth(n=20000)
        for estimate in within_view_gaps(frame, LABEL, "age_group", AGE):
            assert abs(estimate.gap) < 0.05

    def test_missing_stratum_raises(self):
        with pytest.raises(FairnessError, match="stratum"):
            group_gap(synth(), LABEL, "race", ("a", "b"))


class TestStandardisation:
    def test_standardised_rate_reweights_without_dropping_anyone(self):
        frame = synth(n=20000)
        old = frame[frame.age_group == ">=65"]
        expected = 0.5 * false_negative_rate(old[old.view == "AP"], LABEL) + \
                   0.5 * false_negative_rate(old[old.view == "PA"], LABEL)
        assert standardise_rate(old, LABEL) == pytest.approx(expected)

    def test_empty_view_gives_nan_rather_than_a_partial_rate(self):
        frame = synth(n=2000)
        only_pa = frame[frame.view == "PA"]
        assert np.isnan(standardise_rate(only_pa, LABEL))

    def test_reference_mixture_is_used(self):
        frame = synth(n=20000)
        old = frame[frame.age_group == ">=65"]
        all_ap = standardise_rate(old, LABEL, reference={"AP": 1.0, "PA": 0.0})
        assert all_ap == pytest.approx(false_negative_rate(old[old.view == "AP"], LABEL))


class TestDeltaGap:
    """The three pre-specified outcomes, each produced by construction."""

    def test_pure_acquisition_effect_gives_attenuation(self):
        """Rates identical by age within view; the raw gap is entirely the AP mixture."""
        frame = synth(n=30000, ap_share_young=0.20, ap_share_old=0.75)
        result = delta_gap(frame, LABEL, "age_group", AGE, replicates=400)
        assert result.gap_raw > 0.02
        assert abs(result.gap_standardised) < 0.02
        assert result.delta > 0
        assert result.verdict == "attenuation"

    def test_genuine_demographic_effect_survives_standardisation(self):
        """A real age effect, with no difference in acquisition mixture to remove."""
        frame = synth(n=30000, ap_share_young=0.45, ap_share_old=0.45,
                      fnr_ap_old=0.40, fnr_pa_old=0.30,
                      fnr_ap_young=0.20, fnr_pa_young=0.10)
        result = delta_gap(frame, LABEL, "age_group", AGE, replicates=400)
        assert result.gap_standardised > 0.05
        assert abs(result.delta) < 0.03
        assert result.verdict in {"independence", "inconclusive"}

    def test_masking_is_detected_when_the_mixture_hides_a_disparity(self):
        """Older patients are missed more, but receive the view the model handles better."""
        frame = synth(n=30000, ap_share_young=0.80, ap_share_old=0.20,
                      fnr_ap_old=0.45, fnr_pa_old=0.35,
                      fnr_ap_young=0.30, fnr_pa_young=0.20)
        result = delta_gap(frame, LABEL, "age_group", AGE, replicates=400)
        assert result.gap_standardised > result.gap_raw
        assert result.delta < 0

    def test_delta_interval_is_narrower_than_differencing_two_intervals(self):
        """The paired bootstrap keeps the correlation; differencing would inflate it."""
        frame = synth(n=15000, ap_share_young=0.20, ap_share_old=0.75)
        result = delta_gap(frame, LABEL, "age_group", AGE, replicates=600)
        paired_width = result.delta_ci_high - result.delta_ci_low
        naive_width = ((result.raw_ci[1] - result.raw_ci[0])
                       + (result.standardised_ci[1] - result.standardised_ci[0]))
        assert paired_width < naive_width

    def test_result_is_reproducible(self):
        frame = synth(n=8000)
        a = delta_gap(frame, LABEL, "age_group", AGE, replicates=300, seed=7)
        b = delta_gap(frame, LABEL, "age_group", AGE, replicates=300, seed=7)
        assert a.delta_ci_low == b.delta_ci_low


class TestVerdict:
    def _result(self, low, high):
        from pxr.stats.fairness import StandardisedGap
        return StandardisedGap(label=LABEL, site="s", stratum="age_group",
                               gap_raw=0.05, gap_standardised=0.02, delta=0.03,
                               delta_ci_low=low, delta_ci_high=high)

    def test_interval_above_zero_is_attenuation(self):
        assert self._result(0.01, 0.05).verdict == "attenuation"

    def test_interval_below_zero_is_masking(self):
        assert self._result(-0.05, -0.01).verdict == "masking"

    def test_tight_interval_around_zero_is_independence(self):
        assert self._result(-0.01, 0.01).verdict == "independence"

    def test_wide_interval_around_zero_is_inconclusive_not_independence(self):
        """A null result and an uninformative one are different findings."""
        assert self._result(-0.15, 0.15).verdict == "inconclusive"


class TestPositivity:
    def test_balanced_groups_pass(self):
        ok, _, _ = positivity_report(synth(n=20000, ap_share_young=0.45,
                                           ap_share_old=0.55), LABEL, "age_group", AGE)
        assert ok

    def test_a_group_missing_a_view_fails(self):
        """Power does not catch this: the cell can be large and still empty in a view."""
        frame = synth(n=20000)
        frame = frame[~((frame.age_group == ">=65") & (frame.view == "PA"))]
        ok, note, _ = positivity_report(frame, LABEL, "age_group", AGE)
        assert not ok and ">=65/PA" in note

    def test_distant_reweighting_is_flagged_even_with_ample_positives(self):
        """A 90% PA group reweighted to 50% extrapolates, however many patients it has."""
        frame = synth(n=40000, ap_share_young=0.05, ap_share_old=0.08)
        ok, note, detail = positivity_report(frame, LABEL, "age_group", AGE,
                                             max_reweight_distance=0.30)
        assert not ok
        assert "reweighted to" in note
        assert (detail.n_positive > 100).any()      # not a sparsity problem

    def test_detail_reports_observed_shares(self):
        _, _, detail = positivity_report(synth(n=20000), LABEL, "age_group", AGE)
        assert {"level", "view", "n_positive", "observed_share",
                "reference_weight", "reweight_distance"} <= set(detail.columns)
        assert len(detail) == 4      # two groups x two views


class TestInteraction:
    def test_interaction_detected_when_the_effect_depends_on_view(self):
        frame = synth(n=30000, ap_share_young=0.45, ap_share_old=0.45,
                      fnr_ap_old=0.50, fnr_pa_old=0.15,
                      fnr_ap_young=0.20, fnr_pa_young=0.15)
        summary = interaction_model(frame, LABEL, "age_group")
        row = summary[summary.is_interaction].iloc[0]
        assert row.p_value < 0.05

    def test_no_interaction_when_the_effect_is_uniform_across_views(self):
        frame = synth(n=30000, ap_share_young=0.45, ap_share_old=0.45,
                      fnr_ap_old=0.30, fnr_pa_old=0.30,
                      fnr_ap_young=0.20, fnr_pa_young=0.20)
        summary = interaction_model(frame, LABEL, "age_group")
        assert summary[summary.is_interaction].iloc[0].p_value > 0.05

    def test_adjustment_terms_are_included(self):
        """Sex must be adjustable in the age model, or age may carry a sex effect."""
        summary = interaction_model(synth(n=10000), LABEL, "age_group",
                                    adjust_for=["sex"])
        assert any("sex" in t for t in summary.term)

    def test_too_few_positives_raises(self):
        with pytest.raises(FairnessError, match="too few"):
            interaction_model(synth(n=60, prevalence=0.1), LABEL, "age_group")


class TestMultiplicity:
    def test_holm_is_more_powerful_than_bonferroni(self):
        p = [0.01, 0.02, 0.03]
        holm = holm_correct(p)
        assert (holm.p_adjusted <= np.array(p) * 3).all()

    def test_holm_controls_the_smallest_at_full_family_size(self):
        holm = holm_correct([0.01, 0.6, 0.7])
        assert holm.p_adjusted[0] == pytest.approx(0.03)

    def test_holm_adjusted_values_are_monotone(self):
        holm = holm_correct([0.001, 0.01, 0.04, 0.2])
        assert list(holm.p_adjusted) == sorted(holm.p_adjusted)

    def test_bh_rejects_more_than_holm_at_the_same_alpha(self):
        p = [0.01, 0.02, 0.03, 0.04, 0.05]
        assert benjamini_hochberg(p).reject.sum() >= holm_correct(p).reject.sum()

    def test_both_leave_a_single_test_uncorrected(self):
        assert holm_correct([0.03]).p_adjusted[0] == pytest.approx(0.03)
        assert benjamini_hochberg([0.03]).p_adjusted[0] == pytest.approx(0.03)


class TestAnalyseLabel:
    def test_returns_every_component(self):
        result = analyse_label(synth(n=12000), LABEL, "age_group", AGE,
                               site="test", adjust_for=["sex"], replicates=300)
        assert result.raw is not None
        assert len(result.within_view) == 2
        assert result.standardised is not None
        assert result.interaction is not None

    def test_a_failed_interaction_does_not_lose_the_gaps(self):
        """One component failing must not cost the others."""
        frame = synth(n=200, prevalence=0.05)
        result = analyse_label(frame, LABEL, "age_group", AGE, replicates=200)
        assert result.raw is not None
        assert any("interaction" in note for note in result.notes)

    def test_positivity_problems_are_recorded_as_notes(self):
        frame = synth(n=12000)
        frame = frame[~((frame.age_group == ">=65") & (frame.view == "PA"))]
        result = analyse_label(frame, LABEL, "age_group", AGE, replicates=200)
        assert any("positivity" in note for note in result.notes)

    def test_intervals_are_attached_to_every_gap(self):
        result = analyse_label(synth(n=12000), LABEL, "age_group", AGE, replicates=300)
        assert not np.isnan(result.raw.ci_low)
        for estimate in result.within_view:
            assert not np.isnan(estimate.ci_low)


class TestEquivalence:
    """Independence is a claim, not the absence of a detected effect."""

    def _result(self, low, high, margin=0.02):
        from pxr.stats.fairness import StandardisedGap
        return StandardisedGap(label=LABEL, site="s", stratum="age_group",
                               gap_raw=0.05, gap_standardised=0.02, delta=0.03,
                               delta_ci_low=low, delta_ci_high=high,
                               equivalence_margin=margin)

    def test_interval_inside_the_margin_is_independence(self):
        assert self._result(-0.015, 0.018).verdict == "independence"

    def test_interval_covering_zero_but_exceeding_the_margin_is_inconclusive(self):
        """Not detecting a shift is compatible with a substantial one going unfound."""
        assert self._result(-0.06, 0.04).verdict == "inconclusive"

    def test_a_narrow_interval_away_from_zero_is_not_independence(self):
        """The old width rule would have called this independence; it is attenuation."""
        assert self._result(0.035, 0.045).verdict == "attenuation"

    def test_narrow_but_offset_interval_touching_zero_is_inconclusive(self):
        assert self._result(-0.001, 0.045).verdict == "inconclusive"

    def test_margin_is_configurable(self):
        assert self._result(-0.04, 0.04, margin=0.02).verdict == "inconclusive"
        assert self._result(-0.04, 0.04, margin=0.05).verdict == "independence"


class TestBootstrapAccounting:
    def test_usable_replicates_are_recorded(self):
        result = delta_gap(synth(n=8000), LABEL, "age_group", AGE, replicates=300)
        assert result.attempted_replicates == 300
        assert 0 < result.usable_replicates <= 300

    def test_discarded_replicates_are_visible(self):
        """A high discard rate means the interval rests on the cells that survived."""
        frame = synth(n=900, ap_share_old=0.97, prevalence=0.08)
        result = delta_gap(frame, LABEL, "age_group", AGE, replicates=200)
        assert result.usable_replicates <= result.attempted_replicates


class TestInteractionDiagnostics:
    def test_separation_in_a_cell_raises(self):
        """A cell with no errors sends the coefficient to infinity."""
        frame = synth(n=20000, ap_share_young=0.45, ap_share_old=0.45)
        mask = (frame.age_group == ">=65") & (frame.view == "AP")
        frame.loc[mask, f"{LABEL}_predicted"] = frame.loc[mask, f"{LABEL}_true"]
        with pytest.raises(FairnessError, match="separation"):
            interaction_model(frame, LABEL, "age_group")

    def test_thin_cells_raise_even_with_a_large_overall_sample(self):
        """Thousands of positives overall says nothing about events per cell."""
        frame = synth(n=30000, ap_share_young=0.02, ap_share_old=0.02)
        with pytest.raises(FairnessError, match="events|separation"):
            interaction_model(frame, LABEL, "age_group", min_events_per_cell=50)

    def test_healthy_cells_fit_normally(self):
        summary = interaction_model(synth(n=20000, ap_share_young=0.45,
                                          ap_share_old=0.55), LABEL, "age_group")
        assert summary.is_interaction.any()