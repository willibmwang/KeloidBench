#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/radev/home/wbw7/.conda/envs/encdec_llm/bin/python}"

if [[ -z "${TABPFN_TOKEN:-}" ]]; then
  printf "Enter TABPFN_TOKEN: " >&2
  read -r -s TABPFN_TOKEN
  printf "\n" >&2
  export TABPFN_TOKEN
fi

cd "$PROJECT_ROOT"

"$PYTHON_BIN" scripts/06_train_tabpfn.py

mkdir -p results/c_augmented
"$PYTHON_BIN" scripts/06_train_tabpfn.py \
  --data "data/processed/dataset_a_c_augmented.parquet" \
  --out-dir "results/c_augmented"
