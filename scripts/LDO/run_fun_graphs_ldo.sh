#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=scripts/run_fun_graphs.py

$PYTHON "$SCRIPT" \
  --base-dir netlists/LDO \
  --start 2300 \
  --end 2350 \
  --prompt-filename graph_query_prompt.txt \
  --out-filename fun_graph.json
