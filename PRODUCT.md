# SpheroScar Friday product

Ship target: **2026-07-24**. Research prototype for molecular triage of keloid-related expression profiles.

## Honest status
- Breakthrough v2 Stage B primary nested macro F1 ≈ **0.71**, worst fold ≈ **0.43**
- Claim gates (**FAIL**): full≥0.85 or selective≥0.90@≥60% coverage; worst≥0.75
- Go/no-go: **EXECUTE_MATCHED_COHORT_PROTOCOL**
- v2 fibroblast lockbox GSE218007 one-shot macro F1 ≈ **0.33** (domain fail; documented)

## What ships
| Artifact | Path |
|----------|------|
| Claim card | `results/product_friday/CLAIM_CARD.md` |
| Joblib cascade | `results/product_friday/models/*.joblib` |
| Scorer CLI | `scripts/score_cascade_product.py` |
| Export CLI | `scripts/export_product_cascade.py` |
| v2 lockbox score | `results/breakthrough_sprint_v2/lockbox_gse218007/` |
| Matched-cohort ops | `manuscript/matched_cohort/` |

## Quick start
```bash
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"
PY=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python

$PY scripts/export_product_cascade.py

$PY scripts/score_cascade_product.py \
  --features-parquet data/processed/training/features/composition_only.parquet \
  --feature-set composition_only \
  --endpoint keloid_vs_unaffected_skin \
  --out-csv results/product_friday/demo_score.csv
```

## Constraints
- Do **not** rescore GSE185309
- GSE218007 is fibroblast lockbox only (not tissue primary); already scored once
- Do not mix culture/stiffness into `keloid_vs_unaffected_skin`
- High-accuracy claims require matched-cohort lockbox success
