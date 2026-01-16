#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=scripts/transform_fun_graph.py
BASE_DIR=netlists/LDO

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Base directory not found: $BASE_DIR" >&2
  exit 1
fi

shopt -s nullglob
for CIRCUIT_DIR in "$BASE_DIR"/*/; do
  CIRCUIT_DIR="${CIRCUIT_DIR%/}"
  IN_JSON="${CIRCUIT_DIR}/fun_graph.json"
  OUT_JSON="${CIRCUIT_DIR}/fun_updated.json"

  if [[ -f "$IN_JSON" ]]; then
    echo "Transforming ${IN_JSON} -> ${OUT_JSON}"
    $PYTHON "$SCRIPT" --in "$IN_JSON" --out "$OUT_JSON"
  else
    echo "Skipping $(basename "$CIRCUIT_DIR"): missing ${IN_JSON}"
  fi
done
shopt -u nullglob
