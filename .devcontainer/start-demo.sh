#!/usr/bin/env bash
set -euo pipefail

if curl --fail --silent http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then
  exit 0
fi

nohup python -m streamlit run demo/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  > /tmp/keloidbench-streamlit.log 2>&1 &
