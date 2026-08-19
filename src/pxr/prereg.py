from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from pxr.config import Config

__all__ = ["render_analysis_plan", "PreregError"]


class PreregError(ValueError):
    """Raised when the plan cannot be rendered from the artifacts supplied."""


def _table(df: pd.DataFrame, columns: list[str]) -> str:
    """Markdown table from the named columns."""
    present = [c for c in columns if c in df.columns]
    if not present or df.empty:
        return "_(none)_"
    head = "| " + " | ".join(present) + " |"
    rule = "|" + "|".join("---" for _ in present) + "|"
    body = [
        "| " + " | ".join(str(row[c]) for c in present) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([head, rule, *body])


def render_analysis_plan(
    cfg: Config,
    tiers: pd.DataFrame,
    coupling: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    *,
    sensitivity: pd.DataFrame | None = None,
    frozen_config_path: str | None = None,
) -> str:
    """Render the preregistration as markdown.

    Parameters
    ----------
    cfg
        The configuration being frozen.
    tiers
        Output of :func:`pxr.stats.power.assign_tiers`.
    coupling
        The acquisition-coupling model output from notebook 03.
    cohort_summary
        Per-site cohort sizes and composition.
    sensitivity
        Tier stability across assumed baselines, if computed.
    frozen_config_path
        Where the timestamped config copy was written.

    Raises
    ------
    PreregError
        If the tier table lacks the columns the plan must state.
    """
    required = {"label", "tier"}
    if not required <= set(tiers.columns):
        raise PreregError(f"tiers table must contain {sorted(required)}")

    primary = tiers.loc[tiers.tier == "primary", "label"].tolist()
    exploratory = tiers.loc[tiers.tier == "exploratory", "label"].tolist()
    descriptive = tiers.loc[tiers.tier == "descriptive", "label"].tolist()
    if not primary:
        raise PreregError(
            "no label reached the primary tier; the gate must be resolved before "
            "the plan is frozen"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    power_cfg = cfg.analysis["power"]
    thresh = cfg.analysis["threshold_rule"]
    std = cfg.analysis["standardization"]
    boot = cfg.analysis["bootstrap"]

    sens_section = ""
    if sensitivity is not None and not sensitivity.empty:
        stable = sensitivity["stable"].all() if "stable" in sensitivity.columns else None
        cols = [c for c in sensitivity.columns if c != "stable"]
        sens_table = sensitivity.reset_index().rename(columns={"index": "label"})
        sens_section = f"""
### 4.4 Sensitivity of the tier assignment

The baseline false-negative rate is an assumption, so the tier assignment is reported
across plausible values. Labels whose tier moves are identified here rather than
settled by the single assumed value.

{_table(sens_table, ["label", *cols, "stable"])}

All primary labels stable across the grid: **{stable}**.
"""

    return f"""# Preregistered Analysis Plan

**Study.** {cfg.meta['title']}

**Frozen.** {stamp}
**Configuration hash.** `{cfg.config_hash}`
{f"**Frozen configuration.** `{frozen_config_path}`" if frozen_config_path else ""}

This plan is generated from the frozen configuration and the power gate's saved
output; no value in it is transcribed by hand. It is committed and tagged before any
model is trained, and before any predicted score exists.

---

## 1. Question

Do radiograph acquisition pathways (AP versus PA projection) amplify, attenuate, or
operate independently of demographic performance disparities in chest X-ray
classifiers, and does that relationship hold across institutions?

All three outcomes are pre-specified and reportable. The study is not predicated on
finding amplification:

- **Attenuation** - the demographic gap shrinks once acquisition is standardised, so
  part of the apparent disparity travels through the acquisition pathway.
- **Amplification** - the gap is larger within an acquisition stratum than overall,
  so averaging across views conceals it.
- **Independence** - standardising acquisition leaves the gap unchanged, so the two
  are separable bias channels and acquisition auditing must be *added* to demographic
  auditing rather than substituted for it.

## 2. Data

{_table(cohort_summary, list(cohort_summary.columns))}

Cohorts are built from held images: one radiograph per patient, frontal AP or PA
only, with labels drawn from the study that produced each image. Full construction
and exclusion counts are in the cohort flow tables stamped `{cfg.config_hash}`.

**Institutions training their own model.** {', '.join(cfg.training_sites)}
**Sites reported descriptively.** {', '.join(cfg.analysis.get('descriptive_sites') or ['(none)'])}

### 2.1 Design

**Primary experiment: within-site replication.** Each institution trains and
evaluates its own model, and the full analysis - coupling, raw gaps, within-view
gaps, interaction, standardisation - is repeated independently at each. The claim is
whether the acquisition-fairness relationship *replicates* across two independent
institutions.

Evaluating one institution's model at another would confound acquisition with
everything else that differs between sites - scanners, case mix, prevalence,
labelling practice - making it impossible to attribute a change to AP/PA. That
question is asked separately.

**Secondary experiment: cross-site transport.** Each model is then applied to the
other institution, asking whether the relationship survives institutional shift.
This uses the same checkpoints and adds no training.

A descriptive site is retained for the acquisition-coupling contrast and external
calibration; its per-cell positive counts do not support subgroup inference, so it
does not decide which labels are confirmatory.

## 3. Acquisition coupling

Estimated on the cohorts themselves, so it is available before the freeze.

{_table(coupling.round(4), ["site", "term", "odds_ratio", "ci_low", "ci_high", "p_value"])}

## 4. Labels and tiers

### 4.1 Assignment

{_table(tiers.round(4), ["label", "tier", "n_powered", "n_gating_cells", "worst_mde", "reason"])}

- **Primary ({len(primary)}).** {', '.join(primary)} - Holm-corrected family.
- **Exploratory ({len(exploratory)}).** {', '.join(exploratory) or '(none)'} -
  Benjamini-Hochberg, reported as exploratory.
- **Descriptive ({len(descriptive)}).** {', '.join(descriptive) or '(none)'} -
  counts and rates only.

### 4.2 How tiers were decided

A cell is *powered* when it can detect a gap of at least
{power_cfg['meaningful_effect']:.0%} at {power_cfg['target_power']:.0%} power and
alpha {power_cfg['alpha']}. Power was estimated by simulating the comparison
{power_cfg['simulation_replicates']:,} times at the observed cell sizes, assuming a
baseline false-negative rate of {power_cfg['assumed_baseline_fnr']:.0%}.

The baseline is an assumption, not an estimate: estimating it would require the
trained model, which does not exist at gate time.

### 4.3 Contrasts

- **Sex.** Female versus Male.
- **Age.** Under {cfg.primary_age_threshold} versus {cfg.primary_age_threshold} and
  over - a fixed clinical cut-point, not a quantile of these cohorts. The four age
  bands are used for description only. Contrasting the extreme bands was rejected:
  extremes maximise any monotone difference while sitting in the thinnest cells, and
  answer a narrower question than whether performance varies systematically with age.

Underpowered labels are handled by the pre-specified ladder
({' -> '.join(power_cfg['coarsening_ladder'])}). Age pooling is deliberately not a
rung: the contrast is already a fixed clinical threshold, and a data-derived split
would make it depend on the cohorts.
{sens_section}
## 5. Model and evaluation

- **Architecture.** {cfg.model['architecture']}, {cfg.model['image_size']}px,
  seeds {cfg.model['seeds']}.
- **Training source.** {cfg.model['primary_source']}; all other sites held out entirely.
- **Model selection.** Early stopping on `{cfg.model['early_stopping']['monitor']}` -
  never on a fairness metric, which would contaminate the question being asked.
- **Evaluation protocol.** {cfg.n_folds}-fold cross-validation, assigned per patient
  with seed {cfg.splits['seed']} and stratified on {', '.join(cfg.stratify_by)}.
  Each outer fold reserves 20% for test; the remaining 80% is the development set,
  from which {cfg.val_fraction:.0%} is drawn - stratified, and strictly within the
  development folds - for early stopping, calibration, and threshold selection. The
  rest trains. Every fairness estimate is pooled over out-of-fold predictions, so it
  uses the full cohort's positives while no prediction comes from a model that
  trained on that patient.

  The validation split is stratified on {', '.join(cfg.val_stratify_by)} - coarser
  than the folds themselves. The folds carry the fairness estimates and so balance
  acquisition, demographics and case mix alike; validation only selects a stopping
  epoch and an operating threshold, which needs acquisition and demographic balance
  but not pathology balance, and a finer partition fragments into strata too small to
  allocate.

  Strata smaller than {cfg.min_val_stratum} patients contribute no validation
  patients, and the count of such strata is reported: at {cfg.val_fraction:.0%} a
  stratum of three rounds to zero, and rounding should not decide that silently. The
  allocator then enforces the exact total - {cfg.val_fraction:.0%} of the development
  set - distributing any shortfall over the eligible strata, and raising rather than
  returning a smaller set if they cannot supply it.

  After the {cfg.n_folds} models are trained, their test predictions are concatenated
  into a single out-of-fold table in which every patient appears exactly once,
  predicted by the model that did not train on them. Coverage, uniqueness, and
  fold-provenance are verified rather than assumed. All reported quantities - AUROC,
  AUPRC, FNR and FPR, demographic gaps, view-stratified gaps, acquisition-standardised
  disparities, interactions, and bootstrap intervals - are computed from that table.

  Validation is drawn inside the development set rather than taking a whole fold:
  spending a fold would leave 60% for training instead of roughly 68%, and validation
  needs far fewer patients than training does. It never overlaps the test fold, or the
  operating threshold would be tuned on the patients it is later judged against.

  A single held-out test set was rejected on power grounds, not convenience: holding
  out 15% leaves roughly a seventh of the positives in each demographic x view cell,
  which turns a six-point detectable gap into a fourteen-point one and would have
  left the primary family underpowered for the analysis it was selected for.

  Prevalence is never engineered. Folds are balanced on case mix, but the positive
  rate is left exactly as observed - oversampling would make the false-negative rates
  uninterpretable.

## 6. Endpoints and inference

- **Primary endpoint.** {cfg.analysis['primary_endpoint']} - the difference in
  false-negative rate between demographic groups, within an acquisition stratum.
- **Secondary.** {', '.join(cfg.analysis['secondary_endpoints'])}.
- **Operating threshold.** {thresh['primary']} at
  {thresh['fixed_sensitivity_target']:.0%} sensitivity, chosen on source validation
  data and then frozen; {thresh['sensitivity_rule']} reported as a sensitivity
  analysis.
- **Standardisation.** Acquisition-standardised gaps use a fixed
  {std['reference_mix']} reference mixture, applied to {', '.join(std['applies_to'])}
  only. Rate metrics decompose linearly over strata; AUROC and AUPRC do not, so they
  are reported within stratum and never standardised.
- **Uncertainty.** {boot['replicates']:,} {boot['method']} replicates,
  {boot['ci']:.0%} intervals.
- **Multiplicity.** {cfg.analysis['multiple_comparisons']['primary_family']} within
  the primary family; {cfg.analysis['multiple_comparisons']['exploratory_family']}
  for exploratory labels.

## 7. Pre-specified sensitivity analyses

- Uncertain labels treated as negative (`u_zeros`) rather than missing.
- Youden threshold in place of fixed sensitivity.
- Standardisation against the pooled source-validation view mixture.
- Global versus view-conditional calibration.

## 8. What this freeze does and does not guarantee

It fixes the analysis choices listed above: the label tiers, the contrasts, the
endpoints, the thresholds, and the correction procedures. Any departure from them
will be reported as a departure.

It does not make the study assumption-free. The baseline false-negative rate is
assumed; labels are derived from radiology reports and inherit their omissions;
"No Finding" means no pathology was reported, not a radiograph verified normal; and
the two inferential sites use different labellers, which is a source of cross-site
difference that acquisition standardisation does not address. These are stated in the
manuscript's limitations rather than resolved here.
"""