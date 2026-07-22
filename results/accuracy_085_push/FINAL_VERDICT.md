# Accuracy ≥0.85 push — final honest verdict

**Protocol:** `accuracy_085_push_v1`  
**Endpoint:** `keloid_vs_unaffected_skin`  
**Pre-registered gate:** mean LOSO macro F1 ≥ 0.85 **and** every evaluable fold ≥ 0.75, then untouched lockbox.  
**Date:** 2026-07-21

## Verdict

**PUBLIC GATES: FAIL.**  
**LOCKBOX: NOT SCORED** (one-shot confirmation only if public gates pass).  
**GSE185309:** untouched / not rescored.

Claim of ≥0.85 under the pre-registered full-coverage donor-primary metric is **not supported** by the public development folds.

## What was delivered

| Phase | Status | Artifact |
|-------|--------|----------|
| 0 Scoreboard harness | Done | `scripts/score_primary_ladder.py`, `results/accuracy_085_push/baseline.*` |
| 1 Train-only norms + ensemble | Done | `scripts/domain_adaptation.py`, `scripts/train_accuracy_085_push.py`, `norm_ablation_v1` |
| 2 Data expansion (≥8 folds) | Done | 8 LOSO folds; `cohort_expansion_ledger.json`; GSE181316 ingest |
| 3 Nested LOSO re-run | Done | `expanded_ensemble_v1` |
| 4 Lockbox | Skipped | Public gates failed |

## Final public scoreboard (`expanded_ensemble_v1`)

- **Mean macro F1:** 0.826  
- **Worst fold:** 0.182 (`GSE173900`)  
- **Gates:** mean≥0.85 **false**; worst≥0.75 **false**

| Accession | Grain | macro F1 |
|-----------|-------|----------|
| GSE173900 | donor | 0.182 |
| E-MTAB-4945 | donor | 0.762 |
| Sun_Burns | sample | 0.826 |
| GSE158395 | sample_mixed_donor | 0.838 |
| GSE181297 | donor | 1.000 |
| GSE181316 | donor | 1.000 |
| GSE190626 | donor | 1.000 |
| GSE92566 | sample_mixed_donor | 1.000 |

Metric note: paired lesion / non-lesional cohorts (`GSE158395`, `GSE92566`) use **sample** grain because donors are not class-pure; majority-donor collapse is undefined for within-patient contrasts.

## Why the gate cannot clear

1. **`GSE173900` (BGISEQ GPL21697) is non-transferable under current features.**  
   Fixed-grid oracle across program + gene views + train-only norms: **max primary macro ≈ 0.42**, **AUROC ≤ ~0.45**. Fibrosis markers (POSTN/COL1A1/etc.) are **inverted** vs other tissue cohorts after harmonization. This is not a label swap (KC=control, KL=keloid verified).

2. Normalization (accession z-score, reference quantile, ComBat-lite) and calibrated soft-vote ensemble **do not** lift `GSE173900` over 0.75.

3. Expanding to **8** genuine tissue folds (checkpoint C2) raises transferable performance but **cannot** change the worst-fold gate while `GSE173900` remains evaluable.

### Exploratory only (not a claim)

Excluding `GSE173900` after the fact: mean ≈ **0.918**, worst ≈ **0.762** → would pass the numeric gates on the remaining 7 folds.  
This is **not** pre-registered; it is reported only as a ceiling diagnostic. Silent demotion after seeing outer-test failure would be leakage.

Oracle ceiling on all 8 folds ≈ **0.90** mean / **0.42** worst; oracle excluding `GSE173900` ≈ **0.97** / **0.90**.

## Data expansion decisions

Accepted: `GSE158395` + `GSE92566` relabel (lesional keloid vs NL/control), `GSE181316` all-barcode tissue pseudobulk.  
Rejected (culture / wrong disease / no unaffected skin / unusable platform): see `cohort_expansion_ledger.json`.

## Path forward

1. **Matched-cohort protocol** (`manuscript/matched_cohort_protocol.md`) remains the defensible route to a high-accuracy clinical claim.  
2. Optional future pre-registration: a **platform-transferability rule** (e.g. demote cohorts whose max AUROC on a frozen diagnostic grid is ≤0.5) *before* any gate claim — not after.  
3. Do not tune further on outer `GSE173900` test labels.

## Key paths

- `results/accuracy_085_push/expanded_ensemble_v1.{json,md}`  
- `results/accuracy_085_push/diagnostic_fixed_grid_v2.csv`  
- `results/accuracy_085_push/cohort_expansion_ledger.json`  
- `results/accuracy_085_push/FINAL_VERDICT.md` (this file)
