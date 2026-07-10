#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-push08
#SBATCH --partition=gpu
#SBATCH --account=dijk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/baselines_push08 results/ensemble_loso results/publication/figures

PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

FEATURE_SETS=(
  modules_only
  modules_rank_only
  modules_rank_plus_expanded
  profibrotic_module_only
  expanded_profibrotic_only
  published_markers_only
  low_i2_core_only
  shared_genes
  shared_genes_plus_modules
)

echo "=== SpheroScar push-0.8 keloid_binary LOSO pipeline ==="
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "start: $(date -Is)"

echo "--- rebuild training corpus ---"
"${PYTHON}" scripts/build_training_corpus.py

if [[ ! -f results/baselines_push08/raw/baseline_results.jsonl ]]; then
  echo "--- baseline LOSO (no normalization) ---"
  "${PYTHON}" scripts/train_baselines.py \
    --no-wandb \
    --tasks keloid_binary \
    --split-pattern 'leave_accession_out_*' \
    --feature-sets "${FEATURE_SETS[@]}" \
    --normalization-mode none \
    --out-dir results/baselines_push08/raw
else
  echo "--- skipping raw baselines (already complete) ---"
fi

echo "--- baseline LOSO + accession z-score ---"
"${PYTHON}" scripts/train_baselines.py \
  --no-wandb \
  --tasks keloid_binary \
  --split-pattern 'leave_accession_out_*' \
  --feature-sets modules_rank_only modules_rank_plus_expanded expanded_profibrotic_only published_markers_only shared_genes \
  --normalization-mode accession_zscore \
  --out-dir results/baselines_push08/zscore

echo "--- baseline LOSO + quantile rank ---"
"${PYTHON}" scripts/train_baselines.py \
  --no-wandb \
  --tasks keloid_binary \
  --split-pattern 'leave_accession_out_*' \
  --feature-sets modules_rank_only modules_rank_plus_expanded expanded_profibrotic_only published_markers_only shared_genes \
  --normalization-mode quantile_rank \
  --out-dir results/baselines_push08/quantile

echo "--- merge baseline outputs ---"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("results/baselines_push08")
parts = []
for sub in ["raw", "zscore", "quantile"]:
    path = root / sub / "baseline_results.jsonl"
    if path.exists():
        parts.append(pd.read_json(path, lines=True))
if parts:
    merged = pd.concat(parts, ignore_index=True)
    out = root / "baseline_results.jsonl"
    merged.to_json(out, orient="records", lines=True)
    merged.to_csv(root / "baseline_results.csv", index=False)
    print(f"merged {len(merged)} rows -> {out}")
PY

echo "--- calibrated ensemble LOSO ---"
"${PYTHON}" scripts/train_ensemble_loso.py \
  --normalization-mode quantile_rank \
  --out-dir results/ensemble_loso

echo "--- rigor stats + figures ---"
"${PYTHON}" scripts/rigor_stats.py --baseline-dir results/baselines_push08
"${PYTHON}" scripts/generate_publication_figures.py

echo "--- adapter v3 LOSO (first-token hybrid, secondary) ---"
bash scripts/run_loso_adapter_v3.sh

echo "done: $(date -Is)"
