#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=scripts/combine_graphs.py
BASE_DIR=netlists/diff_amps

run_one () {
  local i="$1"
  local CIRCUIT_DIR="${BASE_DIR}/${i}"

  local STR_JSON="${CIRCUIT_DIR}/str_graph.json"
  local FUN_UPDATED_JSON="${CIRCUIT_DIR}/fun_updated.json"
  local FUN_JSON="${CIRCUIT_DIR}/fun_graph.json"
  local OUT_JSON="${CIRCUIT_DIR}/comb_graph.json"

  if [[ ! -f "$STR_JSON" ]]; then
    echo "Skipping ${i}: missing ${STR_JSON}"
    return 0
  fi

  local FUN_IN=""
  if [[ -f "$FUN_UPDATED_JSON" ]]; then
    FUN_IN="$FUN_UPDATED_JSON"
  elif [[ -f "$FUN_JSON" ]]; then
    FUN_IN="$FUN_JSON"
  else
    echo "Skipping ${i}: missing functional graph (need fun_updated.json or fun_graph.json)"
    return 0
  fi

  echo "Combining ${i}:"
  echo "  str: $STR_JSON"
  echo "  fun: $FUN_IN"
  echo "  out: $OUT_JSON"

  $PYTHON "$SCRIPT" \
    --str_graph "$STR_JSON" \
    --fun_graph "$FUN_IN" \
    --out "$OUT_JSON"
}

# 69–97
for i in {69..97}; do run_one "$i"; done

# 614–621
for i in {614..621}; do run_one "$i"; done

# 921–932
for i in {921..932}; do run_one "$i"; done
