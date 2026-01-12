#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=scripts/transform_fun_graph.py
BASE_DIR=netlists/diff_amps

run_one () {
  local i="$1"
  local CIRCUIT_DIR="${BASE_DIR}/${i}"
  local IN_JSON="${CIRCUIT_DIR}/fun_graph.json"
  local OUT_JSON="${CIRCUIT_DIR}/fun_updated.json"

  if [[ -f "$IN_JSON" ]]; then
    echo "Transforming ${IN_JSON} -> ${OUT_JSON}"
    $PYTHON "$SCRIPT" --in "$IN_JSON" --out "$OUT_JSON"
  else
    echo "Skipping ${i}: missing ${IN_JSON}"
  fi
}

# 69–97
for i in {69..97}; do run_one "$i"; done

# 614–621
for i in {614..621}; do run_one "$i"; done

# 921–932
for i in {921..932}; do run_one "$i"; done
