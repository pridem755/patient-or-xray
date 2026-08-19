from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "SplitError",
    "DEFAULT_N_FOLDS",
    "DEFAULT_VAL_FRACTION",
    "DEFAULT_MIN_VAL_STRATUM",
    "plan_validation_allocation",
    "EXTERNAL_FOLD",
    "assign_folds",
    "fold_membership",
    "validate_folds",
    "fold_balance_report",
    "cell_balance_report",
    "assemble_out_of_fold",
]

DEFAULT_N_FOLDS = 5

DEFAULT_VAL_FRACTION = 0.15

#: Fold label for a site that is never trained on.
EXTERNAL_FOLD = -1


class SplitError(ValueError):
    """Raised when folds cannot be assigned or fail their invariants."""


def assign_folds(
    cohort: pd.DataFrame,
    *,
    is_training_site: bool,
    stratify_by: list[str],
    n_folds: int = DEFAULT_N_FOLDS,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each patient to a cross-validation fold.

    Parameters
    ----------
    cohort
        One site's cohort table.
    is_training_site
        When False every patient is labelled :data:`EXTERNAL_FOLD` - the site is
        evaluated but never trained on.
    stratify_by
        Columns held even across folds. Required rather than defaulted: which
        variables are balanced is a scientific choice that belongs in
        ``study_config.yaml`` and in the preregistration, not in a function
        signature.
    n_folds
        Number of folds; each is the test fold exactly once.
    seed
        Fixes the assignment, so a rerun cannot quietly change which patients were
        evaluated by which model.

    Returns
    -------
    DataFrame
        ``cohort`` with a ``fold`` column added.

    Notes
    -----
    Within each stratum the patients are shuffled and dealt round-robin, but the
    starting fold is itself randomised. Dealing always from fold 0 would give every
    stratum's remainder patients to the lowest-numbered folds; with several strata
    that skew accumulates, and it would survive a check on fold *sizes* while still
    biasing fold *composition*.

    Examples
    --------
    >>> mimic = assign_folds(mimic, is_training_site=True,
    ...                      stratify_by=["view", "sex", "Cardiomegaly"])
    >>> sorted(mimic["fold"].unique())
    [0, 1, 2, 3, 4]
    """
    if "patient_id" not in cohort.columns:
        raise SplitError("cohort has no 'patient_id' column")
    if cohort.empty:
        raise SplitError("cannot assign folds to an empty cohort")

    out = cohort.copy()
    if not is_training_site:
        out["fold"] = EXTERNAL_FOLD
        return out

    if n_folds < 2:
        raise SplitError(f"n_folds must be at least 2, got {n_folds}")
    if out["patient_id"].duplicated().any():
        raise SplitError(
            "patient_id is not unique; assign folds on a one-image-per-patient cohort "
            "or group rows by patient first"
        )
    if not stratify_by:
        raise SplitError(
            "stratify_by must name at least one column; it is a pre-specified "
            "scientific choice, not an implementation default"
        )
    missing = [c for c in stratify_by if c not in out.columns]
    if missing:
        raise SplitError(f"cannot stratify on absent column(s): {missing}")

    rng = np.random.default_rng(seed)
    folds = pd.Series(index=out.index, dtype="int64")
    for _, block in out.groupby(stratify_by, observed=True, dropna=False):
        order = rng.permutation(len(block))
        offset = int(rng.integers(n_folds))   # see Notes: prevents remainder skew
        folds.loc[block.index[order]] = (np.arange(len(block)) + offset) % n_folds

    out["fold"] = folds.astype(int)
    return out


DEFAULT_MIN_VAL_STRATUM = 7


def plan_validation_allocation(
    development: pd.DataFrame,
    *,
    stratify_by: list[str],
    val_fraction: float,
    min_stratum_size: int = DEFAULT_MIN_VAL_STRATUM,
) -> pd.DataFrame:
    """Decide how many validation patients each stratum contributes.

    The total is enforced, not approximated. ``round(val_fraction * n)`` patients are
    allocated or the call fails: an allocator that quietly returns fewer would leave
    the operating threshold fitted on a fraction of the intended data, and nothing
    downstream would reveal it.

    Two rules govern the distribution.

    A stratum with fewer than ``min_stratum_size`` patients contributes none - taking
    one from a stratum of three would hand it a 33% validation share, and rounding
    should not settle that invisibly.

    Among eligible strata the floor is allocated first, then the remainder is
    distributed by largest fractional part. Where that still falls short - because
    ineligible strata contribute nothing - the shortfall is distributed over eligible
    strata in size order, largest first, so the additional draws land where they cost
    proportionally least.

    Raises
    ------
    SplitError
        If the eligible strata cannot supply the target between them. Lowering
        ``min_stratum_size`` or coarsening ``stratify_by`` is then a deliberate
        choice, made visibly, rather than an outcome absorbed in silence.

    Returns
    -------
    DataFrame
        One row per stratum: size, exact target, allocation, and whether it was
        skipped - so a notebook can report what was excluded.
    """
    present = [c for c in stratify_by if c in development.columns]
    groups = (
        list(development.groupby(present, observed=True, dropna=False))
        if present
        else [((), development)]
    )

    rows = []
    for key, block in groups:
        exact = len(block) * val_fraction
        eligible = len(block) >= min_stratum_size
        rows.append({
            "stratum": key if isinstance(key, tuple) else (key,),
            "n_patients": len(block),
            "exact": exact,
            "floor": int(np.floor(exact)) if eligible else 0,
            "remainder": (exact - np.floor(exact)) if eligible else -1.0,
            "eligible": eligible,
        })
    plan = pd.DataFrame(rows)
    plan["n_val"] = plan["floor"]

    target = int(round(len(development) * val_fraction))
    eligible_idx = plan.index[plan["eligible"]]
    capacity = int(plan.loc[eligible_idx, "n_patients"].sum())
    if capacity < target:
        raise SplitError(
            f"eligible strata hold {capacity:,} patients but {target:,} validation "
            f"patients are required ({val_fraction:.0%} of {len(development):,}). "
            f"{int((~plan['eligible']).sum())} of {len(plan)} strata were excluded as "
            f"smaller than {min_stratum_size}. Coarsen stratify_by or lower "
            "min_stratum_size - deliberately, not by accident."
        )

    shortfall = target - int(plan["n_val"].sum())
    if shortfall > 0:
        order = plan.loc[eligible_idx].sort_values("remainder", ascending=False).index
        for idx in order[:shortfall]:
            plan.loc[idx, "n_val"] += 1
        shortfall = target - int(plan["n_val"].sum())

    while shortfall > 0:
        order = plan.loc[eligible_idx].sort_values("n_patients", ascending=False).index
        placed = 0
        for idx in order:
            if shortfall == 0:
                break
            if plan.loc[idx, "n_val"] < plan.loc[idx, "n_patients"]:
                plan.loc[idx, "n_val"] += 1
                shortfall -= 1
                placed += 1
        if placed == 0:  
            raise SplitError(
                f"could not allocate {shortfall:,} further validation patients; "
                "eligible strata are exhausted"
            )

    allocated = int(plan["n_val"].sum())
    if allocated != target:  
        raise SplitError(f"allocated {allocated:,} validation patients, expected {target:,}")

    plan["skipped"] = ~plan["eligible"]
    return plan.drop(columns=["floor", "remainder"])


def fold_membership(
    fold: int,
    cohort: pd.DataFrame,
    *,
    stratify_by: list[str],
    val_stratify_by: list[str] | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    min_val_stratum: int = DEFAULT_MIN_VAL_STRATUM,
    seed: int = 42,
) -> pd.Series:
    """Label each patient ``train``, ``val``, or ``test`` for one outer fold.

    The named fold is the test set. The remaining folds form the development set,
    from which a stratified ``val_fraction`` is drawn for early stopping and
    threshold selection; everything else trains.

    ``val_stratify_by`` defaults to ``stratify_by`` minus its pathology terms, and
    is deliberately coarser. The outer folds carry the fairness estimates, so they
    balance acquisition, demographics, and case mix alike. Validation only selects a
    stopping epoch and an operating threshold; it needs acquisition and demographic
    balance but not pathology balance, and a finer partition would fragment it into
    strata too small to allocate - the failure this coarsening avoids.

    Returns
    -------
    Series
        Aligned to ``cohort.index``, with values ``train``, ``val``, ``test``.

    Examples
    --------
    >>> role = fold_membership(0, mimic, stratify_by=["view", "sex"])
    >>> role.value_counts(normalize=True).round(2).to_dict()
    {'train': 0.68, 'test': 0.2, 'val': 0.12}
    """
    if not 0 <= fold < n_folds:
        raise SplitError(f"fold {fold} outside range 0..{n_folds - 1}")
    if "fold" not in cohort.columns:
        raise SplitError("cohort has no 'fold' column; call assign_folds first")
    if not 0 < val_fraction < 1:
        raise SplitError(f"val_fraction must lie in (0, 1), got {val_fraction}")

    role = pd.Series("train", index=cohort.index, dtype="object")
    is_test = cohort["fold"] == fold
    role[is_test] = "test"

    development = cohort.loc[~is_test]
    if development.empty:  # pragma: no cover - defensive
        raise SplitError(f"fold {fold} leaves no development data")

    if val_stratify_by is None:
        val_stratify_by = [c for c in stratify_by if c in ("view", "sex", "age_bin")]

    plan = plan_validation_allocation(
        development, stratify_by=val_stratify_by, val_fraction=val_fraction,
        min_stratum_size=min_val_stratum,
    )

    allocation = dict(zip(plan["stratum"], plan["n_val"], strict=True))

    rng = np.random.default_rng(seed + fold)
    present = [c for c in val_stratify_by if c in development.columns]
    groups = (
        list(development.groupby(present, observed=True, dropna=False))
        if present
        else [((), development)]
    )
    for key, block in groups:
        key = key if isinstance(key, tuple) else (key,)
        n_val = int(allocation.get(key, 0))
        if n_val == 0:
            continue
        chosen = rng.permutation(len(block))[:n_val]
        role[block.index[chosen]] = "val"

    return role


def validate_folds(
    folds: dict[str, pd.DataFrame],
    *,
    training_sites: list[str],
    n_folds: int = DEFAULT_N_FOLDS,
) -> pd.DataFrame:
    """Check the invariants that make out-of-fold evaluation trustworthy.

    Raises
    ------
    SplitError
        If a patient carries more than one fold. That is the failure that silently
        invalidates every downstream number, so it stops the pipeline rather than
        being reported alongside cosmetic issues.
    """
    rows: list[dict] = []
    for site, df in folds.items():
        if "fold" not in df.columns:
            raise SplitError(f"{site}: no 'fold' column")

        multi = df.groupby("patient_id")["fold"].nunique().pipe(lambda s: s[s > 1])
        if len(multi):
            raise SplitError(
                f"{site}: {len(multi)} patient(s) carry more than one fold "
                f"(e.g. {list(multi.index[:3])}); out-of-fold evaluation would be leaked"
            )

        present = sorted(df["fold"].unique())
        if site in training_sites:
            expected = list(range(n_folds))
            ok = present == expected
            note = "" if ok else f"expected folds {expected}, found {present}"
            sizes = df["fold"].value_counts()
            spread = (sizes.max() - sizes.min()) / sizes.mean() if len(sizes) else 0.0
            rows.append({"site": site, "check": "fold_sizes_even",
                         "value": f"{spread:.1%} spread", "ok": spread < 0.05,
                         "note": "" if spread < 0.05 else "folds differ in size by >5%"})
        else:
            ok = present == [EXTERNAL_FOLD]
            note = "" if ok else f"non-training site must be wholly external, found {present}"

        rows.append({"site": site, "check": "fold_membership",
                     "value": str(present), "ok": ok, "note": note})
        rows.append({"site": site, "check": "one_fold_per_patient",
                     "value": f"{len(df):,} patients", "ok": True, "note": ""})

    return pd.DataFrame(rows)


def fold_balance_report(
    cohort: pd.DataFrame, labels: list[str], *, stratify_by: list[str]
) -> pd.DataFrame:
    """Compare fold composition on controlled and uncontrolled variables alike.

    Stratified variables should match almost exactly (marked ``*``). Age and the
    labels that were not stratification anchors were *not* controlled, so this is
    where they are checked rather than assumed.
    """
    rows: list[dict] = []
    for fold, block in cohort.groupby("fold", observed=True):
        row = {"fold": fold, "n_patients": len(block)}
        if "view" in block.columns:
            row["AP" + ("*" if "view" in stratify_by else "")] = round(
                float((block["view"] == "AP").mean()), 4
            )
        if "sex" in block.columns:
            row["female" + ("*" if "sex" in stratify_by else "")] = round(
                float((block["sex"] == "Female").mean()), 4
            )
        if "age_bin" in block.columns:
            for band in sorted(block["age_bin"].dropna().unique()):
                row[f"age {band}"] = round(float((block["age_bin"] == band).mean()), 4)
        elif "age" in block.columns:
            row["age_median"] = round(float(block["age"].median()), 1)
        for label in labels:
            if label in block.columns:
                key = label + ("*" if label in stratify_by else "")
                row[key] = round(float((block[label] == 1).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def cell_balance_report(
    cohort: pd.DataFrame,
    labels: list[str],
    *,
    strata: tuple[str, ...] = ("sex",),
    view_col: str = "view",
) -> pd.DataFrame:
    """Disease-positive counts per fold within demographic x view cells.

    This is the check that matters. Even fold *sizes* say little about whether the
    analysis is supportable; what governs every fairness estimate is the number of
    positive patients in each label x view x demographic cell, and this reports that
    per fold. A cell that is thin in one fold and rich in another would make the
    pooled estimate depend on which fold a patient happened to land in.
    """
    if view_col not in cohort.columns:
        raise SplitError(f"cohort has no '{view_col}' column")
    rows: list[dict] = []
    for label in labels:
        if label not in cohort.columns:
            continue
        positives = cohort[cohort[label] == 1]
        for stratum in strata:
            if stratum not in cohort.columns:
                continue
            grouped = positives.groupby(["fold", view_col, stratum], observed=True).size()
            for (fold, view, level), n in grouped.items():
                rows.append({"label": label, "view": view, "stratum": stratum,
                             "level": level, "fold": fold, "n_positive": int(n)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    wide = out.pivot_table(
        index=["label", "view", "stratum", "level"], columns="fold",
        values="n_positive", fill_value=0,
    )
    wide["min"] = wide.min(axis=1)
    wide["max"] = wide.max(axis=1)
    wide["spread"] = (
        (wide["max"] - wide["min"]) / wide.drop(columns=["min", "max"]).mean(axis=1)
    ).round(3)
    return wide.reset_index().sort_values("min").reset_index(drop=True)


def assemble_out_of_fold(
    fold_predictions: dict[int, pd.DataFrame],
    cohort: pd.DataFrame,
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    id_col: str = "patient_id",
) -> pd.DataFrame:
    """Concatenate the five test-prediction sets into one out-of-fold table.

    Every patient contributes exactly one prediction, produced by the model that did
    not train on them. That property is what lets the fairness analysis use the whole
    cohort's positives, so it is verified here rather than assumed - a duplicated or
    missing patient would quietly bias every estimate that follows.

    Parameters
    ----------
    fold_predictions
        Test-set predictions keyed by outer fold.
    cohort
        The fold-assigned cohort, used to check coverage and to confirm each
        prediction came from the fold that held that patient out.

    Returns
    -------
    DataFrame
        One row per patient, with a ``fold`` column recording which model predicted
        them. This is the table every subsequent analysis reads: AUROC, AUPRC,
        FNR/FPR, demographic gaps, view-stratified gaps, standardised disparities,
        interactions, and the patient bootstrap.

    Raises
    ------
    SplitError
        If a patient appears twice, is missing, or was predicted by a model that
        trained on them.
    """
    missing_folds = sorted(set(range(n_folds)) - set(fold_predictions))
    if missing_folds:
        raise SplitError(f"no predictions for fold(s) {missing_folds}")

    frames = []
    for fold, preds in fold_predictions.items():
        if id_col not in preds.columns:
            raise SplitError(f"fold {fold} predictions have no '{id_col}' column")
        frames.append(preds.assign(fold=fold))
    pooled = pd.concat(frames, ignore_index=True)

    duplicated = pooled[id_col][pooled[id_col].duplicated()]
    if len(duplicated):
        raise SplitError(
            f"{duplicated.nunique()} patient(s) predicted more than once "
            f"(e.g. {list(duplicated.unique()[:3])}); the pooled table is not out-of-fold"
        )

    expected = set(cohort.loc[cohort["fold"] != EXTERNAL_FOLD, id_col])
    absent = expected - set(pooled[id_col])
    if absent:
        raise SplitError(
            f"{len(absent)} patient(s) have no prediction (e.g. {sorted(absent)[:3]}); "
            "the fairness analysis would silently exclude them"
        )

    # Each prediction must come from the fold that held that patient out.
    assigned = cohort.set_index(id_col)["fold"]
    mismatched = pooled[pooled[id_col].map(assigned) != pooled["fold"]]
    if len(mismatched):
        raise SplitError(
            f"{len(mismatched)} prediction(s) came from a fold that did not hold that "
            "patient out; these are not out-of-fold predictions"
        )

    return pooled