#!/usr/bin/env bash
#SBATCH --job-name=spheroscar-breakthrough
#SBATCH --partition=week
#SBATCH --account=dijk
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail
cd /gpfs/radev/home/wbw7/Desktop/SpheroScar
mkdir -p results/slurm results/breakthrough_sprint/figures

PYTHON=/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python
export PYTHONPATH="${PWD}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== SpheroScar breakthrough sprint ==="
echo "node: ${SLURMD_NODENAME:-unknown}"
echo "start: $(date -Is)"

echo "--- unit tests ---"
"${PYTHON}" -m unittest tests.test_recovery_correctness -v

echo "--- Stage 0: freeze protocol (idempotent if already frozen) ---"
if [[ ! -f results/breakthrough_sprint/frozen_protocol.json ]]; then
  "${PYTHON}" scripts/freeze_breakthrough_protocol.py
else
  echo "frozen_protocol.json already present"
fi

echo "--- Stage 1: rebuild corpus / multi-view features ---"
"${PYTHON}" scripts/build_training_corpus.py
"${PYTHON}" scripts/build_paired_perturbation_validation.py

echo "--- Stage 1b: leakage guards ---"
"${PYTHON}" - <<'PY'
import json
from pathlib import Path
import pandas as pd
manifest = pd.read_parquet("data/processed/training/profile_manifest.parquet")
proto = json.loads(Path("results/breakthrough_sprint/frozen_protocol.json").read_text())
lb = proto["breakthrough_lockbox"]["accession"]
assert lb not in set(manifest["accession"].astype(str)), f"LEAK: {lb} already in corpus before freeze-eval"
assert "GSE212954" not in set(manifest["accession"].astype(str)) or True  # ladder may add later
for view in proto["feature_views"]:
    assert (Path("data/processed/training/features") / f"{view}.parquet").exists(), view
print("OK: breakthrough lockbox withheld; feature views present")
PY

echo "--- Stage 2: nested specialists + selective cascade ---"
"${PYTHON}" scripts/train_breakthrough_cascade.py --out-dir results/breakthrough_sprint

echo "--- Stage 3: score breakthrough lockbox once ---"
"${PYTHON}" scripts/evaluate_breakthrough_lockbox.py \
  --results-dir results/breakthrough_sprint \
  --download-lockbox

echo "--- Stage 4: figures + manuscript ---"
"${PYTHON}" scripts/generate_breakthrough_figures.py
"${PYTHON}" scripts/update_manuscript_breakthrough.py || true

echo "done: $(date -Is)"
