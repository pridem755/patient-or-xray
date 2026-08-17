# patient-or-xray

Code for **"Is It the Patient or the X-Ray? Disentangling Demographic and
Acquisition-Driven Bias in Chest X-Ray AI Across Three Institutions."**

Status: **scaffolding** — module implementations are added incrementally.

## What this is

A three-institution study (MIMIC-CXR, CheXpert, NIH ChestX-ray14) of whether
radiograph acquisition pathway (AP vs PA view) amplifies, attenuates, or operates
independently of demographic performance disparities in chest X-ray classifiers.

Demographic scope: age, sex, site. (Race is pre-declared future work.)

## Layout

| Path | Contents |
|------|----------|
| `config/` | `study_config.yaml` — every constant; `frozen/` holds hash-stamped copies |
| `prereg/` | locked analysis plan, git-tagged before any test-set analysis |
| `src/pxr/` | all logic: `data/`, `model/`, `stats/`, `viz/` |
| `notebooks/` | thin orchestrators (00–12); import from `pxr`, no logic |
| `tests/` | pytest — statistical functions vs synthetic ground truth, schema contracts |
| `docs/` | project implementation plan |

Data never lives in this repo. Raw archives, cohorts, image caches, model
checkpoints, and score files live in Google Drive (see the implementation plan).

## Use in Colab

```python
!git clone https://github.com/<user>/patient-or-xray.git /content/pxr-repo
%pip install -q -e /content/pxr-repo
%load_ext autoreload
%autoreload 2
```

## Reproducibility

Every artifact is stamped with a `config_hash` (first 12 hex of the SHA-256 of
`config/study_config.yaml`). Milestone tags: `v0.9-cohorts`, `v1.0-prereg`,
`v1.1-scores`, `v2.0-results-frozen`, `v2.1-camera-ready`.
