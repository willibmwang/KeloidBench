#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-recovery
#SBATCH --partition=gpu
#SBATCH --account=dijk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/recovery_loso results/publication

PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== SpheroScar F1 recovery ladder ==="
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "start: $(date -Is)"

echo "--- unit tests ---"
"${PYTHON}" -m unittest tests.test_recovery_correctness -v

echo "--- preprocess microarray (GPL6244 symbol fix) ---"
"${PYTHON}" scripts/preprocess_microarray.py --skip-platform-download

echo "--- preprocess bulk RNA-seq (Sun/Burns Ensembl map) ---"
"${PYTHON}" scripts/preprocess_bulk_rnaseq.py

echo "--- rebuild training corpus with coverage gates ---"
"${PYTHON}" scripts/build_training_corpus.py

echo "--- corrected baselines (raw programs) ---"
"${PYTHON}" scripts/train_baselines.py \
  --no-wandb \
  --tasks keloid_binary \
  --split-pattern 'leave_accession_out_*' \
  --feature-sets modules_only profibrotic_module_only robust_programs_only modules_rank_only published_markers_only \
  --normalization-mode none \
  --out-dir results/recovery_loso/baselines_raw

echo "--- nested LOSO ladder (broad + clean endpoints) ---"
"${PYTHON}" scripts/train_nested_loso.py \
  --endpoint both \
  --out-dir results/recovery_loso

echo "--- fixed ensemble on modules_only (no double-rank) ---"
"${PYTHON}" scripts/train_ensemble_loso.py \
  --normalization-mode none \
  --feature-sets modules_only profibrotic_module_only robust_programs_only \
  --out-dir results/recovery_loso/ensemble

echo "--- write recovery report ---"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("results/recovery_loso")
summary = pd.read_csv(root / "recovery_loso_summary.csv")
baseline = pd.read_csv(root / "baselines_raw/baseline_summary.csv")
coverage = pd.read_csv("data/processed/training/accession_module_coverage.csv")
corpus = json.loads(Path("data/processed/training/training_corpus_summary.json").read_text())

best_broad = summary[summary.endpoint.eq("keloid_binary")].sort_values("weighted_f1_mean", ascending=False).head(5)
best_clean = summary[summary.endpoint.eq("clean_keloid_binary")].sort_values("weighted_f1_mean", ascending=False).head(5)
best_base = baseline.sort_values("weighted_f1_mean", ascending=False).head(5)

report = {
    "corpus": {
        "n_profiles": corpus.get("n_profiles"),
        "keloid_binary": corpus.get("keloid_binary"),
        "clean_keloid_binary": corpus.get("clean_keloid_binary"),
        "coverage_failures": corpus.get("coverage_failures", []),
    },
    "coverage": coverage.to_dict(orient="records"),
    "best_baselines_raw": best_base.to_dict(orient="records"),
    "best_recovery_broad": best_broad.to_dict(orient="records"),
    "best_recovery_clean": best_clean.to_dict(orient="records"),
}
(root / "recovery_final_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY

echo "--- update manuscript ---"
"${PYTHON}" scripts/update_manuscript_recovery.py

echo "done: $(date -Is)"
