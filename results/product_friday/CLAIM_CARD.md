# SpheroScar Friday product — claim card

- Built: Stage-B cascade export under `results/product_friday/`
- Source: breakthrough v2 **Stage B** (donor selection + calibration), refreshed 2026-07-20 with GSE178562
- Primary nested macro F1: **0.713**
- Primary worst fold: **0.430**
- Claim gates (full≥0.85 or selective≥0.90@≥60%, worst≥0.75): **FAIL**
- Decision: **EXECUTE_MATCHED_COHORT_PROTOCOL**

## What this product is
Molecular triage CLI: primary keloid-vs-unaffected-skin + scar specialists + fibroblast route,
with selective abstention. Honest research prototype — not a clinical diagnostic.

## Nested endpoint ceilings (Stage B)
| Endpoint | Macro F1 | Worst fold | Feature view |
|----------|----------|------------|--------------|
| keloid_vs_unaffected_skin | 0.713 | 0.430 | composition_only |
| fibroblast_keloid_binary | 0.630 | 0.333 | fibrosis_only |
| keloid_vs_pathologic_scar | 0.424 | 0.229 | fused_multiview |
| keloid_vs_normal_scar | 0.329 | 0.000 | fused_multiview |

## External one-shots
| Lockbox | Role | Macro F1 | Note |
|---------|------|----------|------|
| GSE185309 | v1 tissue lockbox | 0.526 (immutable) | Do not rescore |
| GSE218007 | v2 fibroblast lockbox | 0.326 (n=29, 9 donors) | Culture domain; AUROC inverted — not a tissue claim |

## What this product is not
- Not a validated ≥0.85 accuracy claim
- Does not rescore v1 lockbox GSE185309
- Does not mix culture/stiffness into tissue primary
- GSE218007 failure underscores matched-cohort requirement

## Run
```bash
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"
PY=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
$PY scripts/score_cascade_product.py \
  --features-parquet data/processed/training/features/composition_only.parquet \
  --feature-set composition_only \
  --endpoint keloid_vs_unaffected_skin \
  --out-csv results/product_friday/demo_primary_score.csv
```

## Endpoints shipped
- `keloid_vs_unaffected_skin`: nested macro F1=0.713, n_train=83
- `keloid_vs_normal_scar`: nested macro F1=0.329, n_train=282
- `keloid_vs_pathologic_scar`: nested macro F1=0.424, n_train=173
- `fibroblast_keloid_binary`: nested macro F1=0.630, n_train=184
