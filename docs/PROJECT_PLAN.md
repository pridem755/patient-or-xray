# Project Implementation Plan
## "Is It the Patient or the X-Ray? Disentangling Demographic and Acquisition-Driven Bias in Chest X-Ray AI"

**Repo name:** `patient-or-xray`
**Target:** ML4H 2026 (Sept 10 AoE) → fallback CHIL 2027 · arXiv preprint on submission
**Compute:** Google Colab (GPU for stages 5–7) · **Storage:** Google Drive + GitHub (code only)
**Demographic scope (locked):** age, sex, site. Race pre-declared as future work.
**Rebuild rationale:** prior cohort parquets carried patient-level label aggregation (No Finding
contradictions in 50% of CheXpert NF rows); all cohorts are rebuilt from raw metadata with
image-study-level labels enforced by construction and verified by automated checks.

---

## 1. Data handling: from zipped Drive archives to analysis-ready caches

Governing rule: **Drive is for storage, Colab local disk is for work.** Never extract a large
archive into Drive (slow, quota-hostile, corrupts easily). Every session: copy the needed zip
from Drive to `/content/` local disk, extract locally, compute, and write back only compact
derived artifacts (parquets, image caches, checkpoints).

Pipeline for images, run **once**:

1. `Drive: data/raw_zips/{site}.zip` → copy to Colab local disk.
2. Extract locally; verify against a recorded file manifest (count + sizes; MD5 for metadata files).
3. Decode each image → grayscale → resize 224×224 (aspect-preserving pad) → uint8.
4. Pack into **per-site NPY memmap shards + an index parquet** keyed by `image_id`
   (155k images × 224² uint8 ≈ 8 GB total — comfortably fits Drive).
5. Write shards to `Drive: data/image_cache/{site}/`. After this, **the zips are never opened again**;
   training epochs read the memmap cache (fast, deterministic, no per-epoch decode).

Metadata files needed alongside the zips (place in `Drive: data/metadata/{site}/`):

- **NIH:** `Data_Entry_2017.csv` (labels, View Position, age, sex).
- **CheXpert:** `train.csv` + `valid.csv` (labels, Frontal/Lateral, AP/PA, age, sex).
- **MIMIC-CXR-JPG:** `mimic-cxr-2.0.0-metadata.csv` (ViewPosition), `mimic-cxr-2.0.0-chexpert.csv`
  (study-level labels), `patients.csv` from MIMIC-IV hosp module ONLY IF race is later revived — not required now.

**Label-integrity rule (the fix for the old bug), enforced in code:** a row's labels must come from
the *same study as the selected image*. Join order is `image → its study → that study's labels`,
never `patient → pooled labels`. One-image-per-patient selection happens AFTER the join
(rule: earliest study with a valid frontal + non-missing view; deterministic tiebreak by image_id).
An automated contradiction check (`No Finding=1 ∧ any disease=1 ⇒ hard failure`) runs in CI and
at the end of the cohort notebook.

Uncertain labels (CheXpert/MIMIC `-1`): mapped to **NaN** (excluded from that label's denominator),
recorded in config, with U-zeros as a pre-registered sensitivity variant.

---

## 2. Repository structure

```
patient-or-xray/
├── README.md                  # paper abstract, pipeline diagram, how-to-reproduce
├── LICENSE                    # MIT for code; data NOT redistributed
├── environment.yml            # pinned deps; + pip freeze snapshot per run
├── config/
│   ├── study_config.yaml      # ALL constants: paths, labels, bins, seeds, thresholds rule,
│   │                          #   standardization reference, tier assignments
│   └── frozen/                # timestamped copies at each freeze (config_hash-stamped)
├── prereg/
│   └── analysis_plan.md       # locked analysis plan; git-tagged BEFORE test unblinding
├── src/pxr/                   # installable package — all logic lives here, not in notebooks
│   ├── data/    (extract.py, cohort.py, cache.py, splits.py, contracts.py)
│   ├── model/   (densenet.py, train.py, infer.py, calibrate.py)
│   ├── stats/   (metrics.py, standardize.py, bootstrap.py, power.py, interaction.py)
│   └── viz/     (figures.py, tables.py)
├── tests/                     # pytest: stats functions vs synthetic ground truth,
│                              #   schema contracts, label-integrity checks
├── notebooks/                 # thin orchestrators only (see §4)
└── .github/workflows/ci.yml   # ruff + pytest on push
```

GitHub holds code, config, prereg, tests, notebook sources — **never data, images, or PHI-adjacent
metadata**. `.gitignore` blocks `*.parquet`, `*.npy`, `*.csv` outside `tests/fixtures/`.

## 3. Drive layout (project root: `MyDrive/patient-or-xray/`)

```
data/raw_zips/          # your existing zips (read-only; never modified)
data/metadata/{site}/   # the CSVs above
data/cohorts/           # cohort_{site}_{config_hash}.parquet  + cohort_flow.csv
data/image_cache/{site}/# shards_XXX.npy + index.parquet
data/splits/            # splits_{config_hash}.parquet (patient_id → train/val/test)
models/{source}/seed{k}/  # checkpoints + train_log.csv + env_freeze.txt
outputs/scores/         # scores_{source}_{site}_seed{k}_{hash}.parquet  ← the crown jewels
outputs/calibration/    # temperature params, thresholds (json)
outputs/analysis/       # bootstrap results, gap tables (parquet/csv)
outputs/figures/  outputs/tables/   # camera-ready pdf/svg + csv/tex
```

Naming convention everywhere: `{artifact}_{source?}_{site?}_{seed?}_{config_hash}.{ext}`.
The `config_hash` (first 12 hex of the SHA of `study_config.yaml`) stamps every artifact —
you already used this pattern; it stays.

---

## 4. Notebooks: twelve, numbered, each idempotent

Each notebook: reads only canonical upstream artifacts, imports logic from `src/pxr`, writes
hash-stamped outputs, and ends with an **integrity cell** (row counts, positive counts, schema
check, contradiction check) whose printed values go in the paper's appendix. "Restart and run
all" must pass on every notebook before its outputs are considered real.

| # | Notebook | Purpose / key outputs | Runtime |
|---|----------|----------------------|---------|
| 00 | `00_setup_and_config.ipynb` | Mount Drive, install `pxr`, validate config, print config_hash | CPU, min |
| 01 | `01_extract_and_inventory.ipynb` | Zip → local extract, manifest + checksums, `inventory.csv` | CPU, ~1h once |
| 02 | `02_build_cohorts.ipynb` | Image-study-level labels, filters, one-per-patient rule, `cohort_{site}.parquet`, cohort flow table | CPU |
| 03 | `03_audit_and_power.ipynb` | Phase 0–1: coupling tables P(AP\|group), missingness, power simulation, **tier assignment table** | CPU |
| 04 | `04_freeze_prereg.ipynb` | Patient-level splits (fixed seed), frozen config copy, export `analysis_plan.md` → **git tag `v1.0-prereg`** | CPU |
| 05 | `05_cache_images.ipynb` | Decode/resize/pack memmap shards per site | CPU/GPU, ~2–3h once |
| 06 | `06_train_baseline.ipynb` | DenseNet-121 ERM on MIMIC train, 3 seeds, early stop on val macro-AUROC | GPU, ~3–5h/seed |
| 07 | `07_inference_all_sites.ipynb` | Frozen checkpoints → per-image scores for MIMIC test + all of CheXpert + all of NIH | GPU, ~1h |
| 08 | `08_calibrate_and_threshold.ipynb` | Global vs view-conditional temperature scaling (source val only); thresholds: 90%-sens primary, Youden sensitivity; freeze json | CPU |
| 09 | `09_primary_analysis.ipynb` | Raw gaps, within-view gaps, interaction models (site+pathology terms), standardized gaps ΔGap & R with patient bootstrap (1–2k reps), Holm/BH | CPU, ~1h |
| 10 | `10_mechanism_analyses.ipynb` | Disease-free true-negative score shifts by view×group; view-classifier metadata sensitivity; U-zeros sensitivity | CPU/GPU |
| 11 | `11_external_transport.ipynb` | Frozen model+calibration+threshold → CheXpert & NIH; fairness+utility co-endpoints; calibration-transport comparison | CPU |
| 12 | `12_figures_and_tables.ipynb` | Camera-ready figures (coupling, interaction, ΔGap waterfall, true-negative shift), LaTeX tables, results workbook | CPU |

Optional `13_robustness_vit.ipynb` (single ViT-B/16 run) only if the main result exists with ≥1 week
to deadline. It is explicitly out of the critical path.

**Execution order & gates:** 00→05 are sequential; **the prereg tag at 04 is a hard gate** — no
notebook ≥06 runs before it exists. 06–07 produce the score parquets; everything after 07 is pure
re-analysis of those parquets and can be iterated cheaply. Test-set scores are written by 07 but
**not opened** by any analysis notebook until 08's calibration (val-only) is frozen.

---

## 5. Programming style (the rigor contract)

1. **Notebooks orchestrate; `src/pxr` computes.** No function definitions in notebooks beyond
   trivial glue. Every statistical procedure (standardization, bootstrap, interaction model,
   power sim) is a pure, typed, docstringed function with a unit test against synthetic data
   where the true answer is known (e.g., standardization recovers a hand-computed ΔGap).
2. **One config to rule them all.** No magic numbers in code or notebooks; every constant —
   label list, age bins, seed list, bootstrap reps, threshold rule, reference view mix (50/50) —
   lives in `study_config.yaml`. Changing it changes the hash, which orphans stale artifacts.
3. **Determinism:** fixed seeds per stage; `torch.use_deterministic_algorithms(True)`,
   pinned `environment.yml`, `pip freeze` snapshot saved beside every checkpoint.
4. **Data contracts at boundaries:** each stage validates its input parquet's schema, dtypes,
   and invariants (unique patient_id, view ∈ {AP,PA}, label ∈ {0,1,NaN}, NF-contradiction = 0)
   and fails loudly. Contracts live in `src/pxr/data/contracts.py` and run in pytest too.
5. **Git discipline:** feature branches small; tags at milestones —
   `v0.9-cohorts`, `v1.0-prereg` (before any test-set analysis), `v1.1-scores`,
   `v2.0-results-frozen`, `v2.1-camera-ready`. The prereg tag is the paper's honesty anchor.
6. **Style tooling:** `ruff` (lint+format), type hints throughout, CI runs lint+tests on push.
7. **Every figure/table regenerable by one notebook run** from stored parquets — no hand edits.

## 6. Output inventory (what the paper consumes)

- Cohort flow diagram numbers (02) → Methods figure 1.
- Coupling tables + differential-missingness test (03) → Results §1 + appendix.
- Power/tier table (03) → Methods; justifies primary family (Cardiomegaly, Pleural Effusion,
  Atelectasis, Edema) vs exploratory (Pneumothorax, Pneumonia, Consolidation) vs secondary (No Finding).
- Frozen prereg plan + tag timestamp (04) → cited in Methods.
- Per-image score parquets (07) → the reusable core; every downstream analysis reads these.
- Gap tables with 95% CIs: raw / within-view / standardized, per site pair (09) → main results tables.
- Interaction model coefficients with cluster-robust CIs (09) → Results §2.
- True-negative shift figure (10) → the mechanism figure.
- Transport table: fairness + utility co-endpoints, global vs view-conditional calibration,
  frozen thresholds, all external sites (11) → Results §4.
- Reproducibility appendix: config_hash, env freeze, integrity-cell printouts, runtime log.

## 7. Compute & session plan

Colab Pro recommended (A100/L4 when available; T4 works). GPU needed only for 05–07 (+13):
budget ≈ 1 session for caching, 3 sessions for training seeds, 1 for inference — everything
else is CPU and fast. Long-running cells checkpoint to Drive every N steps so a session drop
costs minutes, not hours. Score parquets make the analysis loop (09–12) independent of GPUs
entirely — that is what protects the Sept 10 deadline.

## 8. Immediate next actions

1. Create GitHub repo `patient-or-xray`, push skeleton (this plan → `docs/`, config stub, src layout).
2. Notebook 01–02: rebuild cohorts with image-study labels; run the contradiction check to zero.
3. Notebook 03: regenerate audit + power on the corrected cohorts; confirm tier table.
4. Notebook 04: freeze splits + prereg, tag `v1.0-prereg`.
5. Then, and only then, GPU work begins.
