# KeloidBench

Encoder-decoder modeling for keloid fibrotic biology from gene-expression data, with spheroid assays used as phenotype grounding and downstream validation.

[Open the KeloidBench website](https://willibmwang.github.io/KeloidBench/)

## Interactive demonstration

The five-screen research demo covers expression quality control, program-score
features, live prediction with confidence-based abstention, evidence-grounded
narratives, and study-held-out cohort exploration.

```bash
KB_PY=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
"$KB_PY" -m pip install -r requirements.txt
"$KB_PY" demo/launch_demo.py
```

See [`demo/README.md`](demo/README.md) for input formats, artifact boundaries,
and optional grounded language-model configuration.

The browser-native demonstration is published from the `gh-pages` branch. It
runs entirely as a GitHub website and does not require Codespaces or a Python
server.

## Quick start

```bash
# Collect local seed files and source manifests
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/sourcing.py

# Build the unified microarray disease-state pretraining corpus
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/preprocess_microarray.py

# Build spheroid seed/validation data (Choi source Excel + Dirand seed labels)
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/data_extraction.py --dataset dataset_a

# Build optional public morphology context
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/build_morphology_table.py

# Build current public keloid signature baseline
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/build_keloid_signature.py

# Harmonize current spheroid/signature tables
/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python scripts/harmonize_features.py
```

## Data

| File | Description |
|------|-------------|
| `supplementary_choi.xlsx` | Choi et al. 2024 Supplementary Data 1 |
| `data/processed/choi_*.parquet` | Long-format tables per figure |
| `data/processed/microarray/microarray_expression_wide.parquet` | Unified sample x gene microarray table |
| `data/processed/microarray/microarray_X.npy` | Encoder-ready dense expression matrix |
| `data/processed/microarray/microarray_encoder_decoder_samples.jsonl` | Prompt/response records for lightweight encoder-decoder training |
| `data/processed/microarray/signatures/dataset_c_keloid_signature.parquet` | Legacy GEO keloid marker module baseline |
| `data/processed/spheroid/dataset_a_keloid_spheroid.parquet` | Choi/Dirand spheroid phenotype grounding table |
| `data/processed/spheroid/dataset_b_morphology.parquet` | Optional public spheroid morphology context |

## New Direction

The project is pivoting from a small tabular MVP to a trainable expression-modeling project. Primary training data will come from larger microarray, bulk RNA-seq, scRNA-seq, and spatial keloid datasets. The intended model family is an encoder-decoder architecture adapted from `../Encoder_Decoder_LLM`, where a continuous expression encoder produces learned prefix tokens for a Qwen decoder.

Core tasks include:

- `keloid_vs_normal`
- `lesional_status`
- `scar_differential`
- `fibrotic_activity`
- fibroblast/endothelial cell-state programs
- spheroid-state and drug-response validation using Choi/Dirand

See [`PLAN.md`](PLAN.md) for the updated project plan.
