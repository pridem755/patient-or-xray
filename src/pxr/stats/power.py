from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "PowerError",
    "CellPower",
    "DEFAULT_AGE_THRESHOLD",
    "DEFAULT_CONTRASTS",
    "add_age_group",
    "simulate_power",
    "closed_form_power",
    "minimum_detectable_effect",
    "cell_positive_counts",
    "evaluate_cells",
    "apply_coarsening_ladder",
    "assign_tiers",
]


class PowerError(ValueError):
    """Raised when a power analysis is misspecified."""

DEFAULT_AGE_THRESHOLD: int = 65

DEFAULT_CONTRASTS: dict[str, tuple[str, str]] = {
    "sex": ("Female", "Male"),
    "age_group": ("<65", ">=65"),
}


def add_age_group(
    df: pd.DataFrame, threshold: int = DEFAULT_AGE_THRESHOLD, column: str = "age_group"
) -> pd.DataFrame:
    """Derive the binary age contrast from continuous age.

    The threshold is a fixed clinical value declared in the configuration, never a
    quantile of the observed data: a median split would make the contrast depend on
    the cohorts themselves, which is exactly the data-dependent choice a
    preregistration is meant to exclude.
    """
    if "age" not in df.columns:
        raise PowerError("cannot derive an age group without an 'age' column")
    out = df.copy()
    out[column] = np.where(out["age"] < threshold, "<65", ">=65")
    if threshold != DEFAULT_AGE_THRESHOLD:  # keep labels honest if the cut moves
        out[column] = np.where(out["age"] < threshold, f"<{threshold}", f">={threshold}")
    return out


@dataclass(frozen=True)
class CellPower:
    """Power verdict for one inferential cell."""

    site: str
    label: str
    view: str
    stratum: str
    level_a: str
    level_b: str
    n_a: int
    n_b: int
    mde: float
    powered: bool


# --------------------------------------------------------------------------- #
# Core power calculations
# --------------------------------------------------------------------------- #


def closed_form_power(
    n_a: int, n_b: int, p_a: float, p_b: float, alpha: float = 0.05
) -> float:
    """Normal-approximation power for a two-proportion difference.

    Provided for comparison and testing. :func:`simulate_power` is the primary
    routine because this approximation degrades in small cells, which is precisely
    where the gate has to make its decisions.
    """
    if min(n_a, n_b) < 1:
        return 0.0
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    if se == 0:
        return 0.0 if p_a == p_b else 1.0
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    delta = abs(p_a - p_b)
    return float(
        stats.norm.cdf(delta / se - z_alpha) + stats.norm.cdf(-delta / se - z_alpha)
    )


def simulate_power(
    n_a: int,
    n_b: int,
    p_a: float,
    p_b: float,
    *,
    alpha: float = 0.05,
    replicates: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Probability of detecting the difference ``p_a - p_b`` at these cell sizes.

    Simulates the sampling process directly: draw the number of missed positives in
    each group from its binomial, apply the two-proportion z-test with unpooled
    standard error, and count rejections. Cells where either group has no positives
    have zero power by construction.

    Parameters
    ----------
    n_a, n_b
        Positive patients in each group - the denominators of the FNRs compared.
    p_a, p_b
        True false-negative rates under the alternative.
    alpha
        Two-sided significance level.
    replicates
        Simulation draws. 2,000 gives a power estimate accurate to about +-0.01.
    rng
        Seeded generator; required for reproducible tier assignment.

    Returns
    -------
    float
        Estimated power in [0, 1].

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> simulate_power(2000, 2000, 0.30, 0.35, rng=rng) > 0.8
    True
    >>> simulate_power(10, 10, 0.30, 0.35, rng=rng) < 0.2
    True
    """
    if min(n_a, n_b) < 1:
        return 0.0
    if not (0 <= p_a <= 1 and 0 <= p_b <= 1):
        raise PowerError(f"rates must lie in [0, 1], got p_a={p_a}, p_b={p_b}")
    if replicates < 1:
        raise PowerError(f"replicates must be positive, got {replicates}")
    rng = rng or np.random.default_rng()

    x_a = rng.binomial(n_a, p_a, size=replicates)
    x_b = rng.binomial(n_b, p_b, size=replicates)
    hat_a, hat_b = x_a / n_a, x_b / n_b

    # Unpooled standard error matches the interval-based inference used downstream.
    se = np.sqrt(hat_a * (1 - hat_a) / n_a + hat_b * (1 - hat_b) / n_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (hat_a - hat_b) / se, 0.0)
    critical = stats.norm.ppf(1 - alpha / 2)
    return float(np.mean(np.abs(z) > critical))


def minimum_detectable_effect(
    n_a: int,
    n_b: int,
    *,
    baseline: float = 0.30,
    alpha: float = 0.05,
    target_power: float = 0.80,
    replicates: int = 2000,
    rng: np.random.Generator | None = None,
    tolerance: float = 0.002,
    max_iterations: int = 25,
) -> float:
    """Smallest FNR difference detectable at ``target_power``.

    Power increases monotonically with the true difference, so the effect is found
    by bisection on that difference. The upper group's rate is held at ``baseline``
    and the lower group's is moved away from it.

    Returns
    -------
    float
        The minimum detectable difference expressed as a proportion (0.05 = 5
        points), or ``nan`` when no difference within the feasible range reaches the
        target - meaning the cell cannot support the comparison at all.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> big = minimum_detectable_effect(5000, 5000, rng=rng)
    >>> small = minimum_detectable_effect(200, 200, rng=rng)
    >>> big < small          # more positives detect smaller gaps
    True
    """
    if min(n_a, n_b) < 2:
        return float("nan")
    if not 0 < baseline < 1:
        raise PowerError(f"baseline must lie in (0, 1), got {baseline}")
    rng = rng or np.random.default_rng()

    # The alternative moves toward whichever boundary allows the larger excursion.
    span = max(baseline, 1 - baseline)
    direction = 1.0 if (1 - baseline) >= baseline else -1.0

    def power_at(delta: float) -> float:
        other = baseline + direction * delta
        return simulate_power(
            n_a, n_b, baseline, other, alpha=alpha, replicates=replicates, rng=rng
        )

    if power_at(span * 0.999) < target_power:
        return float("nan")  # not detectable even at the extreme

    low, high = 0.0, span * 0.999
    for _ in range(max_iterations):
        if high - low < tolerance:
            break
        mid = (low + high) / 2
        if power_at(mid) >= target_power:
            high = mid
        else:
            low = mid
    return float(high)


# --------------------------------------------------------------------------- #
# Cell enumeration and evaluation
# --------------------------------------------------------------------------- #


def cell_positive_counts(
    cohort: pd.DataFrame,
    labels: list[str],
    *,
    site: str | None = None,
    contrasts: dict[str, tuple[str, str]] | None = None,
    age_threshold: int | None = None,
) -> pd.DataFrame:
    """Positive-patient counts for each inferential cell and its contrast.

    One row per site x label x view x stratum, carrying the positive count for each
    side of the pre-specified contrast. These counts - not cohort size - determine
    what the comparison can detect.
    """
    contrasts = contrasts or DEFAULT_CONTRASTS
    if "view" not in cohort.columns:
        raise PowerError("cohort has no 'view' column")
    if "age_group" in contrasts and "age_group" not in cohort.columns:
        cohort = add_age_group(cohort, age_threshold or DEFAULT_AGE_THRESHOLD)
    site = site or (cohort["site"].iloc[0] if "site" in cohort.columns and len(cohort) else None)

    rows: list[dict] = []
    for label in labels:
        if label not in cohort.columns:
            continue
        positives = cohort[cohort[label] == 1]
        for view in sorted(cohort["view"].dropna().unique()):
            in_view = positives[positives["view"] == view]
            for stratum, (level_a, level_b) in contrasts.items():
                if stratum not in cohort.columns:
                    continue
                rows.append(
                    {
                        "site": site,
                        "label": label,
                        "view": view,
                        "stratum": stratum,
                        "level_a": level_a,
                        "level_b": level_b,
                        "n_a": int((in_view[stratum] == level_a).sum()),
                        "n_b": int((in_view[stratum] == level_b).sum()),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_cells(
    counts: pd.DataFrame,
    *,
    baseline: float = 0.30,
    alpha: float = 0.05,
    target_power: float = 0.80,
    replicates: int = 2000,
    meaningful_effect: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Attach a minimum detectable effect and a verdict to every cell.

    Parameters
    ----------
    counts
        Output of :func:`cell_positive_counts`.
    meaningful_effect
        The gap the study considers clinically meaningful. A cell is *powered* when
        its minimum detectable effect is no larger than this. Ten percentage points
        is the pre-specified value: smaller gaps are hard to call clinically
        consequential, larger ones would let very thin cells pass.
    seed
        Fixes the simulation so the tier assignment is reproducible - the gate must
        return the same answer on every run.

    Returns
    -------
    DataFrame
        ``counts`` plus ``mde`` and ``powered``.
    """
    rng = np.random.default_rng(seed)
    out = counts.copy()
    mdes = [
        minimum_detectable_effect(
            int(row.n_a), int(row.n_b),
            baseline=baseline, alpha=alpha, target_power=target_power,
            replicates=replicates, rng=rng,
        )
        for row in counts.itertuples()
    ]
    out["mde"] = mdes
    out["powered"] = out["mde"].notna() & (out["mde"] <= meaningful_effect)
    return out


# --------------------------------------------------------------------------- #
# Coarsening ladder and tier assignment
# --------------------------------------------------------------------------- #


def apply_coarsening_ladder(
    cohorts: dict[str, pd.DataFrame],
    labels: list[str],
    ladder: list[str],
    *,
    age_threshold: int | None = None,
    inferential_sites: list[str] | None = None,
    **evaluate_kwargs,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Work down the pre-specified ladder until each label's cells are powered.

    The ladder has two rungs:

    ``demote_label_to_exploratory``
        Keep the label but move it out of the primary corrected family.
    ``report_descriptively_only``
        Report counts and rates without inferential claims.

    ``inferential_sites`` restricts which sites can trigger a rung, for the same
    reason it restricts :func:`assign_tiers`: a site retained for description must
    not demote a label that is powered where the inferential claims are made.

    An earlier design pooled the age bands at the *observed pooled median* when a
    label was underpowered. That rung has been removed on two grounds: the primary
    age contrast is now a fixed clinical threshold, so there is no coarser age split
    to retreat to; and a median taken from the cohorts would make the contrast
    depend on the data, which is precisely the data-dependent choice preregistration
    exists to prevent.

    Returns
    -------
    (table, applied)
        The evaluated cell table and which rung each label ended on.
    """
    known = {"demote_label_to_exploratory", "report_descriptively_only"}
    unknown = [rung for rung in ladder if rung not in known]
    if unknown:
        raise PowerError(
            f"unknown coarsening rung(s): {unknown}; expected {sorted(known)}. "
            "Age pooling is no longer a rung: the primary age contrast is already a "
            "fixed clinical threshold, so there is no coarser age split to fall back "
            "on, and a data-derived one would break preregistration."
        )

    counts = pd.concat(
        [cell_positive_counts(df, labels, site=site, age_threshold=age_threshold)
         for site, df in cohorts.items()],
        ignore_index=True,
    )
    table = evaluate_cells(counts, **evaluate_kwargs)
    applied = {label: "none" for label in labels}

    
    gating = table
    if inferential_sites is not None:
        gating = table[table["site"].isin(inferential_sites)]
        if gating.empty:
            raise PowerError(
                f"no cells from inferential_sites={inferential_sites}; "
                f"available sites: {sorted(table['site'].unique())}"
            )

    failing = set(gating.loc[~gating["powered"], "label"])
    if not failing:
        return table, applied

    for label in failing:
        applied[label] = (
            "demote_label_to_exploratory"
            if "demote_label_to_exploratory" in ladder
            else "report_descriptively_only"
        )
    return table, applied


def assign_tiers(
    table: pd.DataFrame,
    labels: list[str],
    *,
    secondary_lane: list[str] | None = None,
    inferential_sites: list[str] | None = None,
    require_all_sites: bool = True,
) -> pd.DataFrame:
    """Assign each label to primary, exploratory, or descriptive.

    ``primary``
        Every cell powered (or, with ``require_all_sites=False``, powered at a
        majority of sites). Enters the Holm-corrected family.
    ``exploratory``
        Powered somewhere but not everywhere. Benjamini-Hochberg corrected, and
        reported as exploratory.
    ``descriptive``
        No powered cells. Counts and rates only, no inferential claim.

    Labels in ``secondary_lane`` (No Finding) are never primary regardless of power:
    their error semantics invert, so they cannot sit in a corrected family beside
    disease labels.

    ``inferential_sites`` restricts the tier decision to the sites that carry the
    study's inferential claims. A site retained for external validation and for the
    acquisition-coupling contrast - rather than for its own subgroup comparisons -
    must not be able to demote a label that is well powered everywhere else. Cells
    from excluded sites are still evaluated and reported; they simply do not gate.
    Which sites are inferential is a pre-specified design choice, declared before
    results are seen, not a reaction to the tier table.

    Returns
    -------
    DataFrame
        One row per label with cell counts, the worst and median detectable effect,
        and the assigned tier.
    """
    secondary = set(secondary_lane or [])
    gating = table
    if inferential_sites is not None:
        gating = table[table["site"].isin(inferential_sites)]
        if gating.empty:
            raise PowerError(
                f"no cells from inferential_sites={inferential_sites}; "
                f"available sites: {sorted(table['site'].unique())}"
            )

    rows: list[dict] = []
    for label in labels:
        cells = gating[gating["label"] == label]
        all_cells = table[table["label"] == label]
        if cells.empty:
            rows.append({"label": label, "n_cells": 0, "n_gating_cells": 0, "n_powered": 0,
                         "sites_powered": 0, "worst_mde": float("nan"),
                         "median_mde": float("nan"), "tier": "descriptive",
                         "reason": "no cells"})
            continue

        powered = cells["powered"]
        sites = cells["site"].nunique()
        sites_ok = cells.groupby("site")["powered"].all().sum()

        if powered.all():
            tier, reason = "primary", "all cells powered"
        elif powered.any():
            tier = "primary" if (not require_all_sites and sites_ok > sites / 2) else "exploratory"
            reason = f"powered at {int(sites_ok)}/{sites} sites"
        else:
            tier, reason = "descriptive", "no powered cells"

        if label in secondary and tier == "primary":
            tier, reason = "exploratory", "secondary lane: inverted error semantics"

        rows.append({
            "label": label,
            "n_cells": len(all_cells),
            "n_gating_cells": len(cells),
            "n_powered": int(powered.sum()),
            "sites_powered": int(sites_ok),
            "worst_mde": float(cells["mde"].max(skipna=True)),
            "median_mde": float(cells["mde"].median(skipna=True)),
            "tier": tier,
            "reason": reason,
        })
    order = {"primary": 0, "exploratory": 1, "descriptive": 2}
    return (
        pd.DataFrame(rows)
        .assign(_o=lambda d: d["tier"].map(order))
        .sort_values(["_o", "label"])
        .drop(columns="_o")
        .reset_index(drop=True)
    )