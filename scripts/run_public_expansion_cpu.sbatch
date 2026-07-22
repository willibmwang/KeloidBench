#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-public-exp
#SBATCH --partition=week
#SBATCH --account=dijk
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/public_expansion_loso

PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== SpheroScar public-cohort F1 expansion ladder ==="
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "start: $(date -Is)"

echo "--- unit tests ---"
"${PYTHON}" -m unittest tests.test_recovery_correctness -v

echo "--- ingest public keloid cohorts (development + lockbox) ---"
"${PYTHON}" scripts/preprocess_public_keloid.py

echo "--- refresh scRNA donor_tissue grain ---"
"${PYTHON}" scripts/preprocess_scrna_spatial.py

echo "--- rebuild training corpus ---"
"${PYTHON}" scripts/build_training_corpus.py

echo "--- corpus/metadata audit (no predictions) ---"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import pandas as pd
manifest = pd.read_parquet("data/processed/training/profile_manifest.parquet")
summary = json.loads(Path("data/processed/training/training_corpus_summary.json").read_text())
coverage = pd.read_csv("data/processed/training/accession_module_coverage.csv")
audit = {
    "n_profiles": int(len(manifest)),
    "keloid_binary": summary.get("keloid_binary"),
    "keloid_binary_all_profiles": summary.get("keloid_binary_all_profiles"),
    "clean_keloid_binary": summary.get("clean_keloid_binary"),
    "fibroblast_keloid_binary": summary.get("fibroblast_keloid_binary"),
    "n_external_fibrosis_excluded": summary.get("n_external_fibrosis_excluded"),
    "n_lockbox_profiles": summary.get("n_lockbox_profiles"),
    "feature_sets": summary.get("feature_sets"),
    "coverage_failures": summary.get("coverage_failures"),
    "public_accessions": sorted(manifest.loc[manifest.processed_modality.eq("public_keloid"), "accession"].unique().tolist()),
}
Path("results/public_expansion_loso/corpus_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
assert not summary.get("coverage_failures"), summary.get("coverage_failures")
assert int(summary.get("n_external_fibrosis_excluded", 0)) >= 0
PY

echo "--- nested LOSO: current10 + expanded + lockbox modes ---"
"${PYTHON}" scripts/train_nested_loso.py \
  --endpoint all \
  --eval-mode all_modes \
  --out-dir results/public_expansion_loso

echo "--- freeze one nested config from current10 and score lockbox once ---"
"${PYTHON}" scripts/evaluate_lockbox_frozen.py \
  --results-dir results/public_expansion_loso \
  --training-dir data/processed/training

echo "--- update manuscript ---"
"${PYTHON}" scripts/update_manuscript_public_expansion.py

echo "done: $(date -Is)"
