from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "FairnessError",
    "GapEstimate",
    "StandardisedGap",
    "DEFAULT_REFERENCE_MIX",
    "EQUIVALENCE_MARGIN",
    "MIN_POSITIVES_PER_CELL",
    "MAX_REWEIGHT_DISTANCE",
    "false_negative_rate",
    "group_gap",
    "within_view_gaps",
    "standardise_rate",
    "standardised_gap",
    "delta_gap",
    "positivity_report",
    "interaction_model",
    "holm_correct",
    "benjamini_hochberg",
    "analyse_label",
]

EQUIVALENCE_MARGIN = 0.02

MIN_POSITIVES_PER_CELL = 10

MAX_REWEIGHT_DISTANCE = 0.30

DEFAULT_REFERENCE_MIX: dict[str, float] = {"AP": 0.5, "PA": 0.5}


class FairnessError(ValueError):
    """Raised when a fairness estimate cannot be computed as specified."""


@dataclass
class GapEstimate:
    """A difference in false-negative rate between two groups."""

    label: str
    site: str
    stratum: str
    level_a: str
    level_b: str
    rate_a: float
    rate_b: float
    gap: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    n_a: int = 0
    n_b: int = 0
    view: str = "both"

    @property
    def significant(self) -> bool:
        """True when the interval excludes zero."""
        if np.isnan(self.ci_low) or np.isnan(self.ci_high):
            return False
        return not (self.ci_low <= 0 <= self.ci_high)

    def as_row(self) -> dict:
        return {**vars(self), "significant": self.significant}


@dataclass
class StandardisedGap:
    """Raw and acquisition-standardised gaps, and the difference between them."""

    label: str
    site: str
    stratum: str
    gap_raw: float
    gap_standardised: float
    delta: float
    delta_ci_low: float = float("nan")
    delta_ci_high: float = float("nan")
    raw_ci: tuple[float, float] = (float("nan"), float("nan"))
    standardised_ci: tuple[float, float] = (float("nan"), float("nan"))
    positivity_ok: bool = True
    positivity_note: str = ""
    n_a: int = 0
    n_b: int = 0

    equivalence_margin: float = EQUIVALENCE_MARGIN
    usable_replicates: int = 0
    attempted_replicates: int = 0

    @property
    def verdict(self) -> str:
        """Which pre-specified outcome the interval supports.

        Independence requires the whole interval to lie within the equivalence
        margin, not merely to cover zero. An interval covering zero establishes only
        that a shift was not detected - which is compatible with a substantial one
        the study lacked power to find. Judging independence by interval *width*
        would make matters worse still: a narrow interval centred at +0.04 is a
        precisely estimated attenuation, not an absence of effect.
        """
        if np.isnan(self.delta_ci_low):
            return "not estimated"
        if self.delta_ci_low > 0:
            return "attenuation"
        if self.delta_ci_high < 0:
            return "masking"
        inside = (self.delta_ci_low >= -self.equivalence_margin
                  and self.delta_ci_high <= self.equivalence_margin)
        return "independence" if inside else "inconclusive"

    def as_row(self) -> dict:
        return {
            "label": self.label, "site": self.site, "stratum": self.stratum,
            "gap_raw": self.gap_raw, "raw_ci_low": self.raw_ci[0],
            "raw_ci_high": self.raw_ci[1],
            "gap_standardised": self.gap_standardised,
            "std_ci_low": self.standardised_ci[0],
            "std_ci_high": self.standardised_ci[1],
            "delta": self.delta, "delta_ci_low": self.delta_ci_low,
            "delta_ci_high": self.delta_ci_high, "verdict": self.verdict,
            "positivity_ok": self.positivity_ok, "positivity_note": self.positivity_note,
            "equivalence_margin": self.equivalence_margin,
            "usable_replicates": self.usable_replicates,
            "attempted_replicates": self.attempted_replicates,
            "n_a": self.n_a, "n_b": self.n_b,
        }


# --------------------------------------------------------------------------- #
# Vectorised core
# --------------------------------------------------------------------------- #


def _as_arrays(
    frame: pd.DataFrame, label: str, stratum: str, levels: tuple[str, str],
    view_col: str = "view",
) -> dict:
    """Reduce a frame to the four arrays the bootstrap needs.

    The bootstrap recomputes several statistics thousands of times, and doing that on
    a DataFrame spends most of its time in pandas indexing rather than arithmetic.
    Converting once to boolean and integer arrays lets each replicate be a handful of
    numpy operations on an index vector.

    Only disease-positive patients are kept: every rate here has positives as its
    denominator, so the rest never enter a calculation.
    """
    truth, predicted = f"{label}_true", f"{label}_predicted"
    if truth not in frame.columns or predicted not in frame.columns:
        raise FairnessError(f"frame lacks {truth} or {predicted}")
    if stratum not in frame.columns:
        raise FairnessError(f"frame lacks the stratum column {stratum!r}")

    positives = frame[frame[truth] == 1]
    return {
        "missed": (positives[predicted] == 0).to_numpy(),
        "is_a": (positives[stratum] == levels[0]).to_numpy(),
        "is_b": (positives[stratum] == levels[1]).to_numpy(),
        "view": positives[view_col].to_numpy(),
        "n": len(positives),
    }


def _rate(missed: np.ndarray, mask: np.ndarray) -> float:
    """False-negative rate within a mask, or ``nan`` where the mask is empty."""
    total = mask.sum()
    return float(missed[mask].sum() / total) if total else float("nan")


def _gaps_from_arrays(
    arrays: dict, order: np.ndarray | None, reference: dict[str, float],
) -> tuple[float, float]:
    """Raw and standardised gaps for one (possibly resampled) ordering.

    Both are computed from the same rows, which is what makes the paired bootstrap
    paired: their correlation is carried into the interval on their difference.
    """
    missed = arrays["missed"] if order is None else arrays["missed"][order]
    is_a = arrays["is_a"] if order is None else arrays["is_a"][order]
    is_b = arrays["is_b"] if order is None else arrays["is_b"][order]
    view = arrays["view"] if order is None else arrays["view"][order]

    raw = _rate(missed, is_b) - _rate(missed, is_a)

    standardised = 0.0
    for group, sign in ((is_b, 1.0), (is_a, -1.0)):
        total = 0.0
        for name, weight in reference.items():
            rate = _rate(missed, group & (view == name))
            if np.isnan(rate):
                return raw, float("nan")
            total += weight * rate
        standardised += sign * total
    return raw, standardised


# --------------------------------------------------------------------------- #
# Rates and gaps
# --------------------------------------------------------------------------- #


def false_negative_rate(frame: pd.DataFrame, label: str) -> float:
    """Share of truly positive patients the model called negative.

    Computed among positives only - the denominator is patients who have the
    disease, which is what makes this a *missed diagnosis* rate rather than an
    error rate.
    """
    truth, predicted = f"{label}_true", f"{label}_predicted"
    if truth not in frame.columns or predicted not in frame.columns:
        raise FairnessError(f"frame lacks {truth} or {predicted}")
    positives = frame[frame[truth] == 1]
    if len(positives) == 0:
        return float("nan")
    return float((positives[predicted] == 0).mean())


def group_gap(
    frame: pd.DataFrame,
    label: str,
    stratum: str,
    levels: tuple[str, str],
    *,
    site: str = "",
    view: str = "both",
) -> GapEstimate:
    """False-negative rate difference between two demographic groups.

    The gap is ``rate_b − rate_a``, so a positive value means group ``b`` is missed
    more often. Levels are ordered by the study's contrast definition, not by the
    data, so the sign means the same thing at every site.
    """
    if stratum not in frame.columns:
        raise FairnessError(f"frame lacks the stratum column {stratum!r}")
    a, b = levels
    block_a = frame[frame[stratum] == a]
    block_b = frame[frame[stratum] == b]
    rate_a, rate_b = false_negative_rate(block_a, label), false_negative_rate(block_b, label)
    truth = f"{label}_true"
    return GapEstimate(
        label=label, site=site, stratum=stratum, level_a=a, level_b=b,
        rate_a=rate_a, rate_b=rate_b, gap=rate_b - rate_a,
        n_a=int((block_a[truth] == 1).sum()), n_b=int((block_b[truth] == 1).sum()),
        view=view,
    )


def within_view_gaps(
    frame: pd.DataFrame,
    label: str,
    stratum: str,
    levels: tuple[str, str],
    *,
    site: str = "",
    view_col: str = "view",
) -> list[GapEstimate]:
    """The same gap computed inside each acquisition view.

    Every patient in one of these comparisons received the same kind of radiograph,
    so a gap surviving here is not explained by *AP/PA composition*.

    That is a narrower claim than "not explained by acquisition". Projection is one
    observable feature of how an image was obtained; portable versus fixed equipment,
    patient positioning, exposure settings, supine versus erect, device presence and
    the clinical acuity that drives all of them are unmeasured here and remain
    plausible pathways. A within-view gap rules out the composition of AP and PA
    films, not acquisition as a whole.
    """
    return [
        group_gap(frame[frame[view_col] == view], label, stratum, levels,
                  site=site, view=str(view))
        for view in sorted(frame[view_col].dropna().unique())
    ]


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #


def standardise_rate(
    frame: pd.DataFrame,
    label: str,
    *,
    reference: dict[str, float] | None = None,
    view_col: str = "view",
) -> float:
    """False-negative rate under a common acquisition mixture.

    Direct standardisation: the within-view rates are kept as observed and reweighted
    to the reference mixture. Nothing is dropped and no patient is duplicated - only
    the weight each view carries changes.

    Returns ``nan`` if any referenced view has no positive patients, since a rate
    cannot be assigned to a view the group never received.
    """
    reference = reference or DEFAULT_REFERENCE_MIX
    total = 0.0
    for view, weight in reference.items():
        block = frame[frame[view_col] == view]
        rate = false_negative_rate(block, label)
        if np.isnan(rate):
            return float("nan")
        total += weight * rate
    return total


def positivity_report(
    frame: pd.DataFrame,
    label: str,
    stratum: str,
    levels: tuple[str, str],
    *,
    reference: dict[str, float] | None = None,
    view_col: str = "view",
    min_positives: int = MIN_POSITIVES_PER_CELL,
    max_reweight_distance: float = MAX_REWEIGHT_DISTANCE,
) -> tuple[bool, str, pd.DataFrame]:
    """Can both groups be reweighted to the reference mixture without extrapolating?

    Two distinct problems, both invisible to a power calculation.

    **Sparse cells.** Where a group has few positive patients in a view, that view's
    rate is estimated from almost nothing and then given substantial weight in the
    standardised figure.

    **Distant reweighting.** A group that is 90% PA can hold ample positives in both
    views and still be reweighted from 0.10 to 0.50 AP - a fivefold change in the
    weight carried by its smallest stratum. The standardised rate is then dominated
    by the group's least representative patients, and the estimate is closer to an
    extrapolation than a reweighting. ``max_reweight_distance`` bounds how far the
    observed mixture may sit from the reference.

    Returns
    -------
    (ok, note, detail)
        ``detail`` carries per-group cell counts and observed proportions, so the
        notebook can report the actual mixture rather than only a verdict.
    """
    reference = reference or DEFAULT_REFERENCE_MIX
    truth = f"{label}_true"
    rows, problems = [], []

    for level in levels:
        block = frame[frame[stratum] == level]
        positives = block[block[truth] == 1]
        total = len(positives)
        for view, weight in reference.items():
            n = int((positives[view_col] == view).sum())
            observed = n / total if total else float("nan")
            distance = abs(observed - weight) if total else float("nan")
            rows.append({
                "level": level, "view": view, "n_positive": n,
                "observed_share": round(observed, 4) if total else float("nan"),
                "reference_weight": weight,
                "reweight_distance": round(distance, 4) if total else float("nan"),
            })
            if n < min_positives:
                problems.append(f"{level}/{view}: {n} positives")
            elif not np.isnan(distance) and distance > max_reweight_distance:
                problems.append(
                    f"{level}/{view}: observed {observed:.2f} reweighted to {weight:.2f}"
                )

    return (not problems), "; ".join(problems), pd.DataFrame(rows)


def standardised_gap(
    frame: pd.DataFrame,
    label: str,
    stratum: str,
    levels: tuple[str, str],
    *,
    reference: dict[str, float] | None = None,
    view_col: str = "view",
) -> float:
    """Gap between groups when both carry the reference acquisition mixture."""
    a, b = levels
    rate_a = standardise_rate(frame[frame[stratum] == a], label,
                              reference=reference, view_col=view_col)
    rate_b = standardise_rate(frame[frame[stratum] == b], label,
                              reference=reference, view_col=view_col)
    return rate_b - rate_a


def delta_gap(
    frame: pd.DataFrame,
    label: str,
    stratum: str,
    levels: tuple[str, str],
    *,
    site: str = "",
    reference: dict[str, float] | None = None,
    view_col: str = "view",
    replicates: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    min_positives_for_positivity: int = MIN_POSITIVES_PER_CELL,
    equivalence_margin: float = EQUIVALENCE_MARGIN,
) -> StandardisedGap:
    """Raw gap, standardised gap, and their difference, with a paired bootstrap.

    Each replicate resamples patients *once* and recomputes both gaps from the same
    resample, so the correlation between them is preserved. Differencing two
    separately-estimated intervals would treat them as independent and inflate the
    uncertainty on ΔGap, which is the quantity the study's conclusion rests on.

    Resampling is at the patient level, matching the unit of analysis - one image per
    patient - so no clustering correction is needed.
    """
    raw = standardised = None
    raw = group_gap(frame, label, stratum, levels, site=site).gap
    standardised = standardised_gap(frame, label, stratum, levels,
                                    reference=reference, view_col=view_col)

    ok, note, _ = positivity_report(frame, label, stratum, levels, reference=reference,
                                    view_col=view_col,
                                    min_positives=min_positives_for_positivity)

    arrays = _as_arrays(frame, label, stratum, levels, view_col=view_col)
    reference = reference or DEFAULT_REFERENCE_MIX
    rng = np.random.default_rng(seed)
    n_positive = arrays["n"]

    raw_draws, std_draws, delta_draws = [], [], []
    discarded = 0
    for _ in range(replicates):
        order = rng.integers(0, n_positive, n_positive)
        r, st = _gaps_from_arrays(arrays, order, reference)
        if np.isnan(r) or np.isnan(st):
            discarded += 1
            continue
        raw_draws.append(r)
        std_draws.append(st)
        delta_draws.append(r - st)

    def interval(draws: list[float]) -> tuple[float, float]:
        if len(draws) < 100:
            return float("nan"), float("nan")
        low, high = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
        return float(np.percentile(draws, low)), float(np.percentile(draws, high))

    delta_low, delta_high = interval(delta_draws)
    truth = f"{label}_true"
    return StandardisedGap(
        label=label, site=site, stratum=stratum,
        gap_raw=raw, gap_standardised=standardised, delta=raw - standardised,
        delta_ci_low=delta_low, delta_ci_high=delta_high,
        raw_ci=interval(raw_draws), standardised_ci=interval(std_draws),
        positivity_ok=ok, positivity_note=note,
        equivalence_margin=equivalence_margin,
        usable_replicates=len(delta_draws), attempted_replicates=replicates,
        n_a=int(((frame[stratum] == levels[0]) & (frame[truth] == 1)).sum()),
        n_b=int(((frame[stratum] == levels[1]) & (frame[truth] == 1)).sum()),
    )


def bootstrap_gap_ci(
    frame: pd.DataFrame,
    label: str,
    stratum: str,
    levels: tuple[str, str],
    *,
    replicates: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
    view_col: str = "view",
) -> tuple[float, float]:
    """Patient bootstrap interval for a single gap.

    Resamples an index over disease-positive patients rather than the frame itself:
    the rate has positives as its denominator, so no other row can affect it, and
    working on arrays keeps each replicate to a few numpy operations.
    """
    arrays = _as_arrays(frame, label, stratum, levels, view_col=view_col)
    rng = np.random.default_rng(seed)
    n_positive = arrays["n"]
    missed, is_a, is_b = arrays["missed"], arrays["is_a"], arrays["is_b"]

    draws = []
    for _ in range(replicates):
        order = rng.integers(0, n_positive, n_positive)
        gap = _rate(missed[order], is_b[order]) - _rate(missed[order], is_a[order])
        if not np.isnan(gap):
            draws.append(gap)
    if len(draws) < 100:
        return float("nan"), float("nan")
    low, high = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return float(np.percentile(draws, low)), float(np.percentile(draws, high))


# --------------------------------------------------------------------------- #
# Interaction
# --------------------------------------------------------------------------- #


def interaction_model(
    frame: pd.DataFrame,
    label: str,
    stratum: str,
    *,
    adjust_for: list[str] | None = None,
    view_col: str = "view",
    site: str = "",
    min_events_per_cell: int = 10,
) -> pd.DataFrame:
    """Does the demographic effect on missed diagnosis depend on acquisition?

    Fitted among disease-positive patients only, since a false negative is undefined
    elsewhere::

        error ~ demographic + <other demographics> + view + demographic:view

    The other demographic terms matter for the reason the stratified gaps cannot
    address: if older patients skew female and the model performs differently by sex,
    part of an apparent age effect is a sex effect. Adjusting estimates the
    demographic association holding the measured competitors constant.

    The interaction coefficient is the test that acquisition *modifies* the
    disparity, rather than the two merely co-occurring.

    Returns
    -------
    DataFrame
        Coefficients, odds ratios, intervals and p-values, with the interaction row
        marked.
    """
    import statsmodels.formula.api as smf

    truth, predicted = f"{label}_true", f"{label}_predicted"
    positives = frame[frame[truth] == 1].copy()
    if len(positives) < 30:
        raise FairnessError(
            f"{label} at {site or 'this site'}: {len(positives)} positive patients is "
            "too few to fit an interaction model"
        )

    positives["error"] = (positives[predicted] == 0).astype(int)
    if positives["error"].nunique() < 2:
        raise FairnessError(f"{label}: the model made no errors, or only errors")

    cells = positives.groupby([stratum, view_col])["error"].agg(["size", "sum"])
    cells["non_events"] = cells["size"] - cells["sum"]
    empty = cells[(cells["sum"] == 0) | (cells["non_events"] == 0)]
    if len(empty):
        raise FairnessError(
            f"{label} at {site or 'this site'}: separation in "
            f"{len(empty)} of {len(cells)} demographic x view cell(s) "
            f"({empty.index.tolist()[:3]}); the interaction is not identified"
        )
    thin = cells[cells[["sum", "non_events"]].min(axis=1) < min_events_per_cell]
    if len(thin):
        raise FairnessError(
            f"{label} at {site or 'this site'}: fewer than {min_events_per_cell} "
            f"events or non-events in {len(thin)} cell(s) "
            f"({thin.index.tolist()[:3]}); the odds ratio would be unstable"
        )

    adjust_for = [c for c in (adjust_for or []) if c in positives.columns and c != stratum]
    terms = [f"C({stratum})", f"C({view_col})", f"C({stratum}):C({view_col})"]
    terms += [f"C({c})" if positives[c].dtype == object else c for c in adjust_for]
    formula = "error ~ " + " + ".join(terms)

    model = smf.logit(formula, data=positives).fit(disp=0)
    if not getattr(model.mle_retvals, "get", lambda *_: True)("converged", True):
        raise FairnessError(
            f"{label} at {site or 'this site'}: the interaction model did not "
            "converge; its coefficients are not interpretable"
        )
    summary = pd.DataFrame({
        "term": model.params.index,
        "coefficient": model.params.to_numpy(),
        "odds_ratio": np.exp(model.params.to_numpy()),
        "ci_low": np.exp(model.conf_int()[0].to_numpy()),
        "ci_high": np.exp(model.conf_int()[1].to_numpy()),
        "p_value": model.pvalues.to_numpy(),
    })
    summary.insert(0, "label", label)
    summary.insert(1, "site", site)
    summary["is_interaction"] = summary["term"].str.contains(":")
    return summary


# --------------------------------------------------------------------------- #
# Multiplicity
# --------------------------------------------------------------------------- #


def holm_correct(p_values: list[float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm-Bonferroni: strong family-wise error control for the confirmatory family.

    Chosen over plain Bonferroni because it is uniformly more powerful at the same
    guarantee, and over BH because the primary family makes confirmatory claims where
    a single false positive is the error to avoid.
    """
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    m = len(values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return pd.DataFrame({
        "p_value": values, "p_adjusted": adjusted, "reject": adjusted < alpha,
    })


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg: false-discovery control for the exploratory family.

    Appropriate where the aim is to surface candidates for later confirmation rather
    than to make claims, so tolerating a controlled share of false discoveries buys
    worthwhile power.
    """
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    m = len(values)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, m / (rank + 1) * values[idx])
        adjusted[idx] = min(1.0, running)
    return pd.DataFrame({
        "p_value": values, "p_adjusted": adjusted, "reject": adjusted < alpha,
    })


# --------------------------------------------------------------------------- #
# One label, one site, one contrast
# --------------------------------------------------------------------------- #


@dataclass
class LabelAnalysis:
    """Every estimate for one label, site, and demographic contrast."""

    label: str
    site: str
    stratum: str
    raw: GapEstimate
    within_view: list[GapEstimate] = field(default_factory=list)
    standardised: StandardisedGap | None = None
    interaction: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)


def analyse_label(
    frame: pd.DataFrame,
    label: str,
    stratum: str,
    levels: tuple[str, str],
    *,
    site: str = "",
    adjust_for: list[str] | None = None,
    reference: dict[str, float] | None = None,
    replicates: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> LabelAnalysis:
    """Run the full analysis for one label, site, and contrast.

    Raw gap, within-view gaps, the adjusted interaction model, and the standardised
    gap with ΔGap. Failures in any single component are recorded as notes rather than
    aborting the rest: an interaction model that will not converge should not cost
    you the gap estimates.
    """
    notes: list[str] = []

    raw = group_gap(frame, label, stratum, levels, site=site)
    raw.ci_low, raw.ci_high = bootstrap_gap_ci(
        frame, label, stratum, levels, replicates=replicates, ci=ci, seed=seed)

    views = []
    for estimate in within_view_gaps(frame, label, stratum, levels, site=site):
        block = frame[frame["view"] == estimate.view]
        estimate.ci_low, estimate.ci_high = bootstrap_gap_ci(
            block, label, stratum, levels, replicates=replicates, ci=ci, seed=seed)
        views.append(estimate)

    standardised = delta_gap(frame, label, stratum, levels, site=site,
                             reference=reference, replicates=replicates, ci=ci, seed=seed)

    interaction = None
    try:
        interaction = interaction_model(frame, label, stratum,
                                        adjust_for=adjust_for, site=site)
    except Exception as exc:
        notes.append(f"interaction model not fitted: {exc}")

    if not standardised.positivity_ok:
        notes.append(f"positivity: {standardised.positivity_note}")

    return LabelAnalysis(label=label, site=site, stratum=stratum, raw=raw,
                         within_view=views, standardised=standardised,
                         interaction=interaction, notes=notes)