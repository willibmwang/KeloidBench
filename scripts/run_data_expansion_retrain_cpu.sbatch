#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-data-expand
#SBATCH --partition=week
#SBATCH --account=dijk
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/breakthrough_sprint/figures results/accuracy_ladder

PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== SpheroScar data expansion retrain ==="
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "start: $(date -Is)"

echo "--- unit tests ---"
"${PYTHON}" -m unittest tests.test_recovery_correctness -v

echo "--- rebuild training corpus ---"
"${PYTHON}" scripts/build_training_corpus.py

echo "--- score accuracy-ladder lockbox GSE212954 once (fixed mapping) ---"
"${PYTHON}" scripts/evaluate_lockbox_frozen.py \
  --results-dir results/accuracy_ladder \
  --accessions GSE212954

echo "--- retrain breakthrough nested cascade (lockbox already scored; do not re-score) ---"
"${PYTHON}" scripts/train_breakthrough_cascade.py --out-dir results/breakthrough_sprint

echo "--- figures + manuscript ---"
"${PYTHON}" scripts/generate_breakthrough_figures.py
"${PYTHON}" scripts/update_manuscript_breakthrough.py || true

echo "--- eligibility snapshot ---"
"${PYTHON}" - <<'PY'
import pandas as pd
from pathlib import Path
elig = pd.read_csv("data/processed/training/endpoint_eligibility_matrix.csv")
keep = elig[elig["accession"].isin(["GSE210434","GSE303591","GSE282479","GSE232079","GSE212954","GSE245660","GSE188952"])]
print(keep.to_string(index=False))
report = Path("results/breakthrough_sprint/breakthrough_report.json")
if report.exists():
    import json
    r=json.loads(report.read_text())
    print("primary_macro", r.get("primary_nested_macro_f1") or r.get("gates") or list(r.keys())[:20])
PY

echo "done: $(date -Is)"
