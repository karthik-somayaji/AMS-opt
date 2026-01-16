#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=scripts/combine_graphs.py
BASE_DIR=netlists/LDO

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Base directory not found: $BASE_DIR" >&2
  exit 1
fi

shopt -s nullglob
for CIRCUIT_DIR in "$BASE_DIR"/*/; do
  CIRCUIT_DIR="${CIRCUIT_DIR%/}"

  STR_JSON="${CIRCUIT_DIR}/str_graph.json"
  FUN_UPDATED_JSON="${CIRCUIT_DIR}/fun_updated.json"
  FUN_JSON="${CIRCUIT_DIR}/fun_graph.json"
  OUT_JSON="${CIRCUIT_DIR}/comb_graph.json"

  if [[ ! -f "$STR_JSON" ]]; then
    echo "Skipping $(basename "$CIRCUIT_DIR"): missing ${STR_JSON}"
    continue
  fi

  FUN_IN=""
  if [[ -f "$FUN_UPDATED_JSON" ]]; then
    FUN_IN="$FUN_UPDATED_JSON"
  elif [[ -f "$FUN_JSON" ]]; then
    FUN_IN="$FUN_JSON"
  else
    echo "Skipping $(basename "$CIRCUIT_DIR"): missing functional graph"
    continue
  fi

  echo "Combining $(basename "$CIRCUIT_DIR"):"
  echo "  str: $STR_JSON"
  echo "  fun: $FUN_IN"
  echo "  out: $OUT_JSON"

  $PYTHON "$SCRIPT" \
    --str_graph "$STR_JSON" \
    --fun_graph "$FUN_IN" \
    --out "$OUT_JSON"
done
shopt -u nullglob
