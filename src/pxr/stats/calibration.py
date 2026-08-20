from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "CalibrationError",
    "FoldCalibration",
    "fit_temperature",
    "apply_temperature",
    "fit_view_temperatures",
    "select_threshold",
    "calibrate_fold",
    "brier_score",
    "calibration_slope_intercept",
    "expected_calibration_error",
    "negative_log_likelihood",
    "reliability_table",
    "calibration_report",
]


class CalibrationError(ValueError):
    """Raised when calibration or thresholding cannot be performed."""


@dataclass
class FoldCalibration:
    """Everything fitted on one fold's validation data, then frozen.

    Attributes
    ----------
    temperatures
        Per label: one temperature under the global arm, or one per view under the
        view-conditional arm.
    thresholds
        Per label, one operating threshold applied to every patient in the fold
        regardless of view.
    achieved_sensitivity
        Sensitivity actually reached on validation at that threshold. It will not sit
        exactly on the target because scores are discrete; recording it keeps the gap
        visible rather than assumed away.
    """

    fold: int
    arm: str
    temperatures: dict[str, dict[str, float]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    achieved_sensitivity: dict[str, float] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for label, threshold in self.thresholds.items():
            temps = self.temperatures.get(label, {})
            rows.append({
                "fold": self.fold, "arm": self.arm, "label": label,
                "threshold": threshold,
                "achieved_sensitivity": self.achieved_sensitivity.get(label, float("nan")),
                **{f"T_{view}": value for view, value in temps.items()
                   if view != "__pooled__"},
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #


def _to_logit(probability: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(probability, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logit))


def fit_temperature(
    scores: np.ndarray,
    truth: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.01, 100.0),
    tolerance: float = 1e-4,
) -> float:
    """Fit the single temperature minimising negative log-likelihood.

    Temperature scaling divides the logit by ``T``. Above 1 it softens confidence,
    below 1 it sharpens; ``T = 1`` leaves the scores untouched. Because the transform
    is monotone, the ranking - and therefore AUROC - is unchanged, which is what makes
    it safe to apply after the model is selected.

    The objective is convex in ``log T``, so a bounded scalar minimisation is
    sufficient and deterministic; no gradient loop is needed.

    Parameters
    ----------
    scores, truth
        Validation probabilities and their binary outcomes. Entries where ``truth``
        is missing are dropped - an unrecorded label cannot inform calibration.

    Raises
    ------
    CalibrationError
        If no usable observations remain, or the outcomes are single-class. A
        temperature fitted on one class is meaningless: likelihood is then minimised
        by pushing confidence to the boundary.
    """
    from scipy.optimize import minimize_scalar

    observed = ~np.isnan(truth) & ~np.isnan(scores)
    y, p = truth[observed], scores[observed]
    if len(y) == 0:
        raise CalibrationError("no observed validation entries to calibrate on")
    if len(np.unique(y)) < 2:
        raise CalibrationError(
            "validation outcomes are single-class; a temperature fitted here would "
            "be driven to the boundary rather than reflecting confidence"
        )

    logits = _to_logit(p)

    def negative_log_likelihood(log_t: float) -> float:
        scaled = logits / np.exp(log_t)
        return float(np.mean(np.logaddexp(0, scaled) - y * scaled))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(np.log(bounds[0]), np.log(bounds[1])),
        method="bounded",
        options={"xatol": tolerance},
    )
    if not result.success:
        raise CalibrationError(
            f"temperature optimisation did not converge: {result.message}. Accepting "
            "the value would mean calibrating with a number the optimiser did not "
            "stand behind."
        )
    temperature = float(np.exp(result.x))
    at_lower = temperature <= bounds[0] * 1.001
    at_upper = temperature >= bounds[1] * 0.999
    if at_lower or at_upper:
        raise CalibrationError(
            f"temperature reached the search bound ({temperature:.4f}); the optimiser "
            "did not find a minimum, which usually means near-degenerate validation "
            "outcomes rather than a calibration that scaling can fix"
        )
    return temperature


def apply_temperature(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities by ``temperature``, leaving their order intact."""
    if temperature <= 0:
        raise CalibrationError(f"temperature must be positive, got {temperature}")
    return _sigmoid(_to_logit(scores) / temperature)


def _apply_by_view(
    scores: np.ndarray, views: np.ndarray, temperatures: dict[str, float]
) -> np.ndarray:
    """Apply per-view temperatures, never leaving a patient uncalibrated.

    A view absent from the fitted set - unexpected, or missing - falls back to the
    pooled temperature rather than passing the raw score through. Calibrating some
    patients while silently leaving others raw would make the two groups
    incomparable in exactly the dimension under study.
    """
    pooled = temperatures.get("__pooled__")
    out = np.empty(len(scores), dtype=float)
    for i, (score, view) in enumerate(zip(scores, views, strict=True)):
        temperature = temperatures.get(str(view))
        if temperature is None:
            if pooled is None:
                raise CalibrationError(
                    f"no temperature for view {str(view)!r} and no pooled fallback"
                )
            temperature = pooled
        out[i] = apply_temperature(np.array([score]), temperature)[0]
    return out


def fit_view_temperatures(
    scores: np.ndarray,
    truth: np.ndarray,
    views: np.ndarray,
    *,
    min_positives: int = 25,
    min_negatives: int = 25,
) -> dict[str, float]:
    """Fit one temperature per acquisition view.

    A view falls back to the pooled temperature unless it holds at least
    ``min_positives`` positive and ``min_negatives`` negative validation cases.
    Counting the classes separately matters: a view with 500 observations but three
    positives satisfies any total-count rule while yielding a temperature driven by
    those three patients, and applying it to that view's test set would inject noise
    dressed as a correction.

    The pooled temperature is retained under ``"__pooled__"`` so every fallback is
    recorded and can be reported rather than left silent.
    """
    pooled = fit_temperature(scores, truth)
    temperatures: dict[str, float] = {"__pooled__": pooled}
    for view in np.unique(views[~pd.isna(views)]):
        mask = (views == view) & ~np.isnan(truth) & ~np.isnan(scores)
        positives = int((truth[mask] == 1).sum())
        negatives = int((truth[mask] == 0).sum())
        if positives < min_positives or negatives < min_negatives:
            temperatures[str(view)] = pooled
            continue
        temperatures[str(view)] = fit_temperature(scores[mask], truth[mask])
    return temperatures


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


def select_threshold(
    scores: np.ndarray,
    truth: np.ndarray,
    *,
    target_sensitivity: float = 0.90,
    rule: str = "fixed_sensitivity",
) -> tuple[float, float]:
    """Choose the operating threshold on validation data.

    ``fixed_sensitivity``
        The highest threshold that still reaches ``target_sensitivity``. Taking the
        highest keeps specificity as good as the sensitivity constraint allows -
        a lower threshold would meet the target while flagging more patients than
        necessary.
    ``youden``
        Maximises sensitivity + specificity - 1. The pre-specified sensitivity
        analysis.

    Returns
    -------
    (threshold, achieved_sensitivity)
        The achieved value is returned rather than assumed: scores are discrete, so
        the target is rarely met exactly, and the shortfall belongs in the record.
    """
    observed = ~np.isnan(truth) & ~np.isnan(scores)
    y, p = truth[observed], scores[observed]
    if len(y) == 0:
        raise CalibrationError("no observed validation entries to threshold on")
    positives = p[y == 1]
    if len(positives) == 0:
        raise CalibrationError(
            "no positive validation cases; a sensitivity target cannot be met"
        )

    if rule == "fixed_sensitivity":
        if not 0 < target_sensitivity <= 1:
            raise CalibrationError(
                f"target_sensitivity must lie in (0, 1], got {target_sensitivity}"
            )
        candidates = np.unique(p)
        sensitivities = (positives[:, None] >= candidates[None, :]).mean(axis=0)
        meeting = candidates[sensitivities >= target_sensitivity]
        if len(meeting) == 0:
            raise CalibrationError(
                f"no threshold reaches sensitivity {target_sensitivity:.2f}; the "
                f"highest achievable is {sensitivities.max():.3f}"
            )
        threshold = float(meeting.max())
        achieved = float((positives >= threshold).mean())
        return threshold, achieved

    if rule == "youden":
        from sklearn.metrics import roc_curve

        false_positive, true_positive, cuts = roc_curve(y, p)
        best = int(np.argmax(true_positive - false_positive))
        threshold = float(cuts[best])
        return threshold, float((positives >= threshold).mean())

    raise CalibrationError(f"unknown threshold rule: {rule!r}")


# --------------------------------------------------------------------------- #
# Per-fold pipeline
# --------------------------------------------------------------------------- #


def calibrate_fold(
    validation: pd.DataFrame,
    labels: list[str],
    *,
    fold: int,
    arm: str = "global",
    target_sensitivity: float = 0.90,
    threshold_rule: str = "fixed_sensitivity",
    view_col: str = "view",
    min_positives_per_view: int = 25,
    min_negatives_per_view: int = 25,
) -> FoldCalibration:
    """Fit temperatures and thresholds on one fold's validation predictions.

    Parameters
    ----------
    validation
        That fold's validation patients, with a ``{label}_score`` and ``{label}_true``
        column per label, plus ``view``.
    arm
        ``"global"`` fits one temperature per label; ``"view"`` fits one per view.
        The threshold is then selected from the *combined* calibrated validation
        population either way, so the arms differ only in calibration and not in the
        decision policy applied to AP and PA patients.

    Returns
    -------
    FoldCalibration
        Frozen quantities, to be applied to that fold's test patients and nowhere
        else.
    """
    if arm not in ("global", "view"):
        raise CalibrationError(f"unknown calibration arm: {arm!r}")

    result = FoldCalibration(fold=fold, arm=arm)
    views = validation[view_col].to_numpy() if view_col in validation.columns else None
    if arm == "view" and views is None:
        raise CalibrationError(f"view-conditional arm needs a '{view_col}' column")

    for label in labels:
        score_col, truth_col = f"{label}_score", f"{label}_true"
        if score_col not in validation.columns or truth_col not in validation.columns:
            raise CalibrationError(f"validation frame lacks {score_col} or {truth_col}")

        scores = validation[score_col].to_numpy(dtype=float)
        truth = validation[truth_col].to_numpy(dtype=float)

        if arm == "global":
            temperature = fit_temperature(scores, truth)
            result.temperatures[label] = {"global": temperature}
            calibrated = apply_temperature(scores, temperature)
        else:
            per_view = fit_view_temperatures(
                scores, truth, views,
                min_positives=min_positives_per_view,
                min_negatives=min_negatives_per_view,
            )
            result.temperatures[label] = per_view
            calibrated = _apply_by_view(scores, views, per_view)

        threshold, achieved = select_threshold(
            calibrated, truth, target_sensitivity=target_sensitivity, rule=threshold_rule
        )
        result.thresholds[label] = threshold
        result.achieved_sensitivity[label] = achieved

    return result


def apply_calibration(
    test: pd.DataFrame,
    calibration: FoldCalibration,
    labels: list[str],
    *,
    view_col: str = "view",
) -> pd.DataFrame:
    """Apply a fold's frozen calibration and thresholds to its test patients.

    Returns
    -------
    DataFrame
        ``test`` with ``{label}_calibrated`` probabilities and ``{label}_predicted``
        binary decisions added.
    """
    out = test.copy()
    views = out[view_col].to_numpy() if view_col in out.columns else None

    for label in labels:
        score_col = f"{label}_score"
        if score_col not in out.columns:
            raise CalibrationError(f"test frame lacks {score_col}")
        scores = out[score_col].to_numpy(dtype=float)
        temps = calibration.temperatures[label]

        if "global" in temps:
            calibrated = apply_temperature(scores, temps["global"])
        else:
            if views is None:
                raise CalibrationError(
                    f"view-conditional calibration needs a '{view_col}' column"
                )
            calibrated = _apply_by_view(scores, views, temps)

        out[f"{label}_calibrated"] = calibrated
        out[f"{label}_predicted"] = (calibrated >= calibration.thresholds[label]).astype(float)
    return out


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def expected_calibration_error(
    scores: np.ndarray, truth: np.ndarray, *, bins: int = 10
) -> float:
    """Mean gap between predicted probability and observed rate, weighted by bin size.

    Zero means the probabilities can be read as rates: among patients predicted at
    0.3, three in ten are positive.
    """
    observed = ~np.isnan(truth) & ~np.isnan(scores)
    y, p = truth[observed], scores[observed]
    if len(y) == 0:
        return float("nan")

    edges = np.linspace(0, 1, bins + 1)
    total, error = len(y), 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (p > low) & (p <= high) if low > 0 else (p >= low) & (p <= high)
        if not mask.any():
            continue
        error += mask.sum() / total * abs(y[mask].mean() - p[mask].mean())
    return float(error)


def brier_score(scores: np.ndarray, truth: np.ndarray) -> float:
    """Mean squared error of the probabilities.

    A proper scoring rule, so it rewards both discrimination and calibration, and
    unlike ECE it needs no binning - which is why the two are reported together.
    """
    observed = ~np.isnan(truth) & ~np.isnan(scores)
    if not observed.any():
        return float("nan")
    return float(np.mean((scores[observed] - truth[observed]) ** 2))


def negative_log_likelihood(scores: np.ndarray, truth: np.ndarray) -> float:
    """Mean NLL - the quantity temperature scaling minimises, reported for the record."""
    observed = ~np.isnan(truth) & ~np.isnan(scores)
    if not observed.any():
        return float("nan")
    y, p = truth[observed], np.clip(scores[observed], 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_slope_intercept(
    scores: np.ndarray, truth: np.ndarray
) -> tuple[float, float]:
    """Slope and intercept of the outcome regressed on the logit of the score.

    Perfect calibration gives slope 1 and intercept 0. A slope below 1 is the
    signature of overconfidence - the scores spread wider than the outcomes justify -
    and unlike ECE this says *how* the calibration is wrong rather than only how much.
    """
    from sklearn.linear_model import LogisticRegression

    observed = ~np.isnan(truth) & ~np.isnan(scores)
    y, p = truth[observed], scores[observed]
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logits = _to_logit(p).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logits, y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def reliability_table(
    scores: np.ndarray, truth: np.ndarray, *, bins: int = 10
) -> pd.DataFrame:
    """Predicted against observed rate per bin - the reliability diagram as a table."""
    observed = ~np.isnan(truth) & ~np.isnan(scores)
    y, p = truth[observed], scores[observed]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (p > low) & (p <= high) if low > 0 else (p >= low) & (p <= high)
        rows.append({
            "bin_low": round(low, 3), "bin_high": round(high, 3),
            "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()) if mask.any() else float("nan"),
            "observed_rate": float(y[mask].mean()) if mask.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def calibration_report(
    frame: pd.DataFrame,
    labels: list[str],
    *,
    view_col: str = "view",
    bins: int = 10,
) -> pd.DataFrame:
    """Calibration error before and after, overall and within each view.

    Four measures rather than one: ECE depends on the binning, Brier is a proper
    scoring rule needing none, NLL is what the temperature minimises, and the
    slope/intercept say *how* the calibration is wrong - a slope below 1 is the
    signature of overconfidence - rather than only how much.

    The by-view columns are the informative ones: if calibration error differs between
    AP and PA after global scaling, the model's confidence depends on acquisition.
    That is a property of the model. Whether it produces demographic disparity is a
    separate question, answered by the primary analysis.
    """
    rows = []
    for label in labels:
        raw = frame.get(f"{label}_score")
        calibrated = frame.get(f"{label}_calibrated")
        truth = frame.get(f"{label}_true")
        if raw is None or truth is None:
            continue

        raw_values, truth_values = raw.to_numpy(float), truth.to_numpy(float)
        slope_raw, intercept_raw = calibration_slope_intercept(raw_values, truth_values)
        row = {
            "label": label,
            "ece_raw": expected_calibration_error(raw_values, truth_values, bins=bins),
            "brier_raw": brier_score(raw_values, truth_values),
            "nll_raw": negative_log_likelihood(raw_values, truth_values),
            "slope_raw": slope_raw,
            "intercept_raw": intercept_raw,
        }
        if calibrated is not None:
            values = calibrated.to_numpy(float)
            slope, intercept = calibration_slope_intercept(values, truth_values)
            row.update({
                "ece_calibrated": expected_calibration_error(values, truth_values, bins=bins),
                "brier_calibrated": brier_score(values, truth_values),
                "nll_calibrated": negative_log_likelihood(values, truth_values),
                "slope_calibrated": slope,
                "intercept_calibrated": intercept,
            })

        if view_col in frame.columns:
            for view in sorted(frame[view_col].dropna().unique()):
                mask = (frame[view_col] == view).to_numpy()
                source = calibrated if calibrated is not None else raw
                row[f"ece_{view}"] = expected_calibration_error(
                    source.to_numpy(float)[mask], truth.to_numpy(float)[mask], bins=bins)
        rows.append(row)
    return pd.DataFrame(rows)
