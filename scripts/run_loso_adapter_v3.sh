#!/usr/bin/env bash
# Run full leave-one-study-out adapter v3 evaluation (8 keloid accessions).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python}"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

OUT="${ROOT}/results/frozen_qwen_adapter/loso_v3_hybrid.jsonl"
mkdir -p "${ROOT}/results/frozen_qwen_adapter"

"${PYTHON}" "${ROOT}/scripts/train_frozen_qwen_adapter.py" \
  --loso-preset \
  --split-pattern 'leave_accession_out_*' \
  --all-task-splits \
  --save-checkpoint \
  --results-jsonl "${OUT}" \
  "$@"

"${PYTHON}" "${ROOT}/scripts/write_adapter_eval_report.py" \
  --results-jsonl "${OUT}" \
  --out-dir "${ROOT}/results/frozen_qwen_adapter"
