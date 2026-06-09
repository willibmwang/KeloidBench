# SpheroScar

Keloid spheroid modeling using **tabular data only** (no encoder-decoder LM).

## Quick start

```bash
# Collect local seed files and source manifests
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/01_download_sources.py

# Build Dataset A (Choi source Excel + Dirand seed labels)
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/02_extract_choi_dirand.py

# Build Dataset B (Bodenmiller morphology metadata + SpheroScan manifest)
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/03_build_morphology_table.py

# Build Dataset C (GEO keloid activity module scores)
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/04_build_keloid_signature.py

# Harmonize A/B/C and evaluate
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/05_harmonize_features.py
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/06_train_tabpfn.py
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/06_train_tabpfn.py \
  --data "data/processed/dataset_a_c_augmented.parquet" \
  --out-dir "results/c_augmented"
```

## Data

| File | Description |
|------|-------------|
| `supplementary_choi.xlsx` | Choi et al. 2024 Supplementary Data 1 |
| `data/processed/choi_*.parquet` | Long-format tables per figure |
| `data/processed/dataset_a_keloid_spheroid.parquet` | Unified modeling table |
| `data/processed/dataset_b_morphology.parquet` | Bodenmiller spheroid morphology metadata |
| `data/processed/dataset_c_keloid_signature.parquet` | GEO keloid marker module scores |
| `data/processed/dataset_a_c_augmented.parquet` | Dataset A with C-derived signature context |

## Targets

- `drug_response_class` — sensitive vs resistant (from 3D relative volume)
- `spheroid_state` — dominant morphology state (day 4)
- `fibrotic_state` — planned with Dirand data

## Config

See [`configs/mvp.yaml`](configs/mvp.yaml) and [`PLAN.md`](PLAN.md).

TabPFN is installed from the local clone at `../0shot_Tabular Inference/TabPFN/src`.
Local model-weight download requires `TABPFN_TOKEN`; without it, `06_train_tabpfn.py` uses the logistic regression fallback and records that in the output.
