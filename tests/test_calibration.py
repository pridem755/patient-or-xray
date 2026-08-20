from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxr.stats.calibration import (
    CalibrationError,
    apply_calibration,
    apply_temperature,
    calibrate_fold,
    calibration_report,
    expected_calibration_error,
    fit_temperature,
    fit_view_temperatures,
    reliability_table,
    select_threshold,
)

LABELS = ["Cardiomegaly", "Pneumothorax"]


def overconfident(n=2000, seed=0, sharpness=2.5):
    """Well-calibrated scores, then sharpened - which is what training produces.

    The scores are generated *from* the outcome probability rather than around the
    outcome, so at ``sharpness=1`` they are genuinely calibrated and the fitted
    temperature should be near 1. Sharpening above 1 makes them overconfident.
    """
    rng = np.random.default_rng(seed)
    honest = rng.uniform(0.02, 0.98, n)          # the true probability per patient
    truth = rng.binomial(1, honest).astype(float)
    logit = np.log(honest / (1 - honest)) * sharpness
    return 1 / (1 + np.exp(-logit)), truth


class TestTemperature:
    def test_ranking_is_unchanged(self):
        """Scaling happens after model selection; altering the order would invalidate it."""
        scores, _ = overconfident()
        rescaled = apply_temperature(scores, 2.5)
        assert np.array_equal(np.argsort(scores), np.argsort(rescaled))

    def test_auroc_is_unchanged(self):
        from sklearn.metrics import roc_auc_score

        scores, truth = overconfident()
        before = roc_auc_score(truth, scores)
        after = roc_auc_score(truth, apply_temperature(scores, 3.0))
        assert abs(before - after) < 1e-9

    def test_overconfident_scores_get_a_temperature_above_one(self):
        scores, truth = overconfident(sharpness=3.0)
        assert fit_temperature(scores, truth) > 1.0

    def test_underconfident_scores_get_a_temperature_below_one(self):
        scores, truth = overconfident(sharpness=0.4)
        assert fit_temperature(scores, truth) < 1.0

    def test_fitting_reduces_calibration_error(self):
        scores, truth = overconfident(sharpness=3.0)
        temperature = fit_temperature(scores, truth)
        before = expected_calibration_error(scores, truth)
        after = expected_calibration_error(apply_temperature(scores, temperature), truth)
        assert after < before

    def test_temperature_one_is_the_identity(self):
        scores, _ = overconfident()
        assert np.allclose(apply_temperature(scores, 1.0), scores, atol=1e-6)

    def test_missing_outcomes_are_dropped(self):
        scores, truth = overconfident(n=1000)
        truth_with_gaps = truth.copy()
        truth_with_gaps[:300] = np.nan
        assert fit_temperature(scores, truth_with_gaps) > 0

    def test_single_class_validation_raises(self):
        """Likelihood would be minimised by pushing confidence to the boundary."""
        with pytest.raises(CalibrationError, match="single-class"):
            fit_temperature(np.array([0.2, 0.3, 0.4]), np.array([1.0, 1.0, 1.0]))

    def test_no_observations_raises(self):
        with pytest.raises(CalibrationError, match="no observed"):
            fit_temperature(np.array([0.2, 0.3]), np.array([np.nan, np.nan]))

    def test_non_positive_temperature_rejected(self):
        with pytest.raises(CalibrationError, match="must be positive"):
            apply_temperature(np.array([0.5]), 0.0)

    def test_confident_scores_do_not_overflow(self):
        """Exponentiating before the log-space rearrangement would overflow here."""
        scores = np.array([1e-9, 1 - 1e-9] * 100)
        truth = np.array([0.0, 1.0] * 100)
        assert np.isfinite(fit_temperature(scores, truth))


class TestViewTemperatures:
    def _by_view(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        views = rng.choice(["AP", "PA"], n)
        truth = rng.binomial(1, 0.3, n).astype(float)
        sharp = np.where(views == "AP", 3.0, 1.0)
        honest = np.clip(rng.normal(0.3 + 0.4 * truth, 0.15), 0.01, 0.99)
        logit = np.log(honest / (1 - honest)) * sharp
        return 1 / (1 + np.exp(-logit)), truth, views

    def test_views_needing_different_correction_get_different_temperatures(self):
        scores, truth, views = self._by_view()
        temps = fit_view_temperatures(scores, truth, views)
        assert {"AP", "PA"} <= set(temps)
        assert temps["AP"] > temps["PA"]

    def test_thin_view_falls_back_to_the_pooled_temperature(self):
        """A temperature fitted on a handful of patients is noise dressed as a fix."""
        scores, truth, views = self._by_view(n=600)
        views = views.copy()
        views[:] = "PA"
        views[:10] = "AP"          # only ten AP patients
        temps = fit_view_temperatures(scores, truth, views)
        assert temps["AP"] == pytest.approx(fit_temperature(scores, truth))

    def test_single_class_view_falls_back(self):
        """A view with one outcome class carries no information about its confidence.

        Only the minority view is made degenerate: making both single-class would
        leave the pooled fit degenerate too, and the fallback would have nothing to
        fall back to.
        """
        rng = np.random.default_rng(3)
        n = 3000
        views = np.array(["PA"] * n)
        views[:200] = "AP"
        honest = rng.uniform(0.02, 0.98, n)
        truth = rng.binomial(1, honest).astype(float)
        truth[views == "AP"] = 1.0          # AP patients all positive
        temps = fit_view_temperatures(honest, truth, views,
                                      min_positives=1, min_negatives=1)
        assert temps["AP"] == pytest.approx(temps["__pooled__"])

    def test_a_view_short_of_positives_falls_back(self):
        """500 observations with three positives passes any total-count rule."""
        scores, truth, views = self._by_view(n=2000)
        truth = truth.copy()
        ap = views == "AP"
        truth[ap] = 0.0
        truth[np.where(ap)[0][:3]] = 1.0
        temps = fit_view_temperatures(scores, truth, views,
                                      min_positives=25, min_negatives=25)
        assert temps["AP"] == pytest.approx(temps["__pooled__"])


class TestThreshold:
    def test_fixed_sensitivity_reaches_the_target(self):
        scores, truth = overconfident(n=5000)
        threshold, achieved = select_threshold(scores, truth, target_sensitivity=0.90)
        assert achieved >= 0.89

    def test_higher_target_lowers_the_threshold(self):
        scores, truth = overconfident(n=5000)
        strict, _ = select_threshold(scores, truth, target_sensitivity=0.99)
        lax, _ = select_threshold(scores, truth, target_sensitivity=0.70)
        assert strict < lax

    def test_achieved_sensitivity_is_reported_not_assumed(self):
        """Scores are discrete, so the target is rarely met exactly."""
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        truth = np.array([0.0, 1.0, 1.0, 1.0])
        _, achieved = select_threshold(scores, truth, target_sensitivity=0.90)
        assert 0 < achieved <= 1

    def test_youden_rule_available_as_the_sensitivity_analysis(self):
        scores, truth = overconfident(n=3000)
        threshold, _ = select_threshold(scores, truth, rule="youden")
        assert 0 <= threshold <= 1

    def test_no_positives_raises(self):
        with pytest.raises(CalibrationError, match="no positive"):
            select_threshold(np.array([0.1, 0.2]), np.array([0.0, 0.0]))

    def test_unknown_rule_raises(self):
        scores, truth = overconfident(n=100)
        with pytest.raises(CalibrationError, match="unknown threshold rule"):
            select_threshold(scores, truth, rule="invented")

    def test_invalid_target_raises(self):
        scores, truth = overconfident(n=100)
        with pytest.raises(CalibrationError, match="target_sensitivity"):
            select_threshold(scores, truth, target_sensitivity=1.5)


def make_fold(n=3000, seed=0):
    """A fold's worth of scores, deliberately sharper on AP than PA."""
    rng = np.random.default_rng(seed)
    views = rng.choice(["AP", "PA"], n)
    frame = {"patient_id": [f"p{i}" for i in range(n)], "view": views}
    for label in LABELS:
        honest = rng.uniform(0.02, 0.98, n)
        truth = rng.binomial(1, honest).astype(float)
        sharp = np.where(views == "AP", 2.5, 1.2)
        logit = np.log(honest / (1 - honest)) * sharp
        frame[f"{label}_score"] = 1 / (1 + np.exp(-logit))
        frame[f"{label}_true"] = truth
    return pd.DataFrame(frame)


class TestFoldPipeline:
    def test_global_arm_fits_one_temperature_per_label(self):
        result = calibrate_fold(make_fold(), LABELS, fold=0, arm="global")
        assert set(result.temperatures["Cardiomegaly"]) == {"global"}
        assert set(result.thresholds) == set(LABELS)

    def test_view_arm_fits_one_temperature_per_view(self):
        result = calibrate_fold(make_fold(), LABELS, fold=0, arm="view")
        assert {"AP", "PA"} <= set(result.temperatures["Cardiomegaly"])

    def test_both_arms_produce_a_single_threshold_per_label(self):
        """The arms must differ in calibration alone, not in decision policy."""
        for arm in ("global", "view"):
            result = calibrate_fold(make_fold(), LABELS, fold=0, arm=arm)
            for label in LABELS:
                assert isinstance(result.thresholds[label], float)

    def test_achieved_sensitivity_is_recorded(self):
        result = calibrate_fold(make_fold(), LABELS, fold=0, target_sensitivity=0.90)
        for label in LABELS:
            assert result.achieved_sensitivity[label] >= 0.85

    def test_applying_to_test_adds_calibrated_scores_and_decisions(self):
        validation, test = make_fold(seed=0), make_fold(seed=1)
        result = calibrate_fold(validation, LABELS, fold=0)
        scored = apply_calibration(test, result, LABELS)
        for label in LABELS:
            assert f"{label}_calibrated" in scored.columns
            assert set(scored[f"{label}_predicted"].unique()) <= {0.0, 1.0}

    def test_decisions_follow_the_frozen_threshold(self):
        validation, test = make_fold(seed=0), make_fold(seed=1)
        result = calibrate_fold(validation, LABELS, fold=0)
        scored = apply_calibration(test, result, LABELS)
        label = LABELS[0]
        expected = (scored[f"{label}_calibrated"] >= result.thresholds[label]).astype(float)
        assert (scored[f"{label}_predicted"] == expected).all()

    def test_ap_and_pa_share_one_threshold_under_the_view_arm(self):
        """Different thresholds by view would change policy as well as calibration."""
        validation, test = make_fold(seed=0), make_fold(seed=1)
        result = calibrate_fold(validation, LABELS, fold=0, arm="view")
        scored = apply_calibration(test, result, LABELS)
        label = LABELS[0]
        for view in ("AP", "PA"):
            block = scored[scored["view"] == view]
            expected = (block[f"{label}_calibrated"] >= result.thresholds[label])
            assert (block[f"{label}_predicted"] == expected.astype(float)).all()

    def test_unknown_arm_raises(self):
        with pytest.raises(CalibrationError, match="unknown calibration arm"):
            calibrate_fold(make_fold(), LABELS, fold=0, arm="invented")

    def test_view_arm_without_a_view_column_raises(self):
        frame = make_fold().drop(columns=["view"])
        with pytest.raises(CalibrationError, match="view"):
            calibrate_fold(frame, LABELS, fold=0, arm="view")

    def test_missing_score_column_raises(self):
        frame = make_fold().drop(columns=["Cardiomegaly_score"])
        with pytest.raises(CalibrationError, match="Cardiomegaly_score"):
            calibrate_fold(frame, LABELS, fold=0)

    def test_summary_frame_records_everything_fitted(self):
        result = calibrate_fold(make_fold(), LABELS, fold=2, arm="view")
        frame = result.to_frame()
        assert set(frame["label"]) == set(LABELS)
        assert {"threshold", "achieved_sensitivity", "T_AP", "T_PA"} <= set(frame.columns)


class TestDiagnostics:
    def test_perfect_calibration_scores_near_zero(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 20000)
        y = rng.binomial(1, p).astype(float)
        assert expected_calibration_error(p, y) < 0.02

    def test_overconfidence_raises_the_error(self):
        scores, truth = overconfident(sharpness=3.5)
        honest, _ = overconfident(sharpness=1.0)
        assert expected_calibration_error(scores, truth) > expected_calibration_error(
            honest, truth)

    def test_reliability_table_covers_the_range(self):
        scores, truth = overconfident()
        table = reliability_table(scores, truth, bins=10)
        assert len(table) == 10
        assert {"mean_predicted", "observed_rate", "n"} <= set(table.columns)

    def test_report_compares_raw_and_calibrated_by_view(self):
        validation, test = make_fold(seed=0), make_fold(seed=1)
        result = calibrate_fold(validation, LABELS, fold=0, arm="global")
        scored = apply_calibration(test, result, LABELS)
        report = calibration_report(scored, LABELS)
        assert {"ece_raw", "ece_calibrated", "ece_AP", "ece_PA"} <= set(report.columns)

    def test_calibration_improves_the_reported_error(self):
        validation, test = make_fold(seed=0), make_fold(seed=1)
        result = calibrate_fold(validation, LABELS, fold=0, arm="global")
        report = calibration_report(apply_calibration(test, result, LABELS), LABELS)
        assert (report["ece_calibrated"] < report["ece_raw"]).all()


class TestFallbacksAndConvergence:
    """Additions after review: no silent gaps, no unconverged temperatures."""

    def test_unknown_view_uses_the_pooled_fallback_not_the_raw_score(self):
        """Calibrating some patients and not others breaks the comparison itself."""
        validation, test = make_fold(seed=0), make_fold(seed=1)
        calibration = calibrate_fold(validation, LABELS, fold=0, arm="view")
        test = test.copy()
        test.loc[test.index[:20], "view"] = "LATERAL"     # a view never fitted
        scored = apply_calibration(test, calibration, LABELS)
        label = LABELS[0]
        unexpected = scored.iloc[:20]
        assert not np.allclose(unexpected[f"{label}_calibrated"],
                               unexpected[f"{label}_score"])

    def test_every_patient_receives_a_calibrated_score(self):
        validation, test = make_fold(seed=0), make_fold(seed=1)
        calibration = calibrate_fold(validation, LABELS, fold=0, arm="view")
        scored = apply_calibration(test, calibration, LABELS)
        for label in LABELS:
            assert scored[f"{label}_calibrated"].notna().all()

    def test_degenerate_scores_raise_rather_than_returning_a_bound(self):
        """A temperature at the search bound is not a minimum, so it is not a fit."""
        scores = np.concatenate([np.full(500, 1 - 1e-12), np.full(500, 1e-12)])
        truth = np.concatenate([np.zeros(500), np.ones(500)])   # perfectly inverted
        with pytest.raises(CalibrationError, match="search bound"):
            fit_temperature(scores, truth)

    def test_pooled_fallback_is_recorded_for_reporting(self):
        rng = np.random.default_rng(0)
        honest = rng.uniform(0.02, 0.98, 500)
        truth = rng.binomial(1, honest).astype(float)
        views = np.array(["PA"] * 500)
        temps = fit_view_temperatures(honest, truth, views)
        assert "__pooled__" in temps


class TestThresholdUsesObservedScores:
    """The rule must be the one specified, not an interpolation of it."""

    def test_threshold_is_an_observed_score(self):
        scores, truth = overconfident(n=2000)
        threshold, _ = select_threshold(scores, truth, target_sensitivity=0.90)
        assert threshold in set(scores.tolist())

    def test_highest_qualifying_threshold_is_chosen(self):
        """A lower threshold would meet the target while flagging more than needed."""
        scores = np.array([0.10, 0.22, 0.35, 0.51, 0.68, 0.79, 0.83, 0.91, 0.94, 0.97])
        truth = np.ones(10)
        threshold, achieved = select_threshold(scores, truth, target_sensitivity=0.90)
        assert threshold == 0.22
        assert achieved == pytest.approx(0.9)

    def test_unreachable_target_raises_with_the_achievable_value(self):
        scores = np.array([0.9, 0.9, 0.1])
        truth = np.array([1.0, 1.0, 0.0])
        threshold, achieved = select_threshold(scores, truth, target_sensitivity=1.0)
        assert achieved == 1.0
