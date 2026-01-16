#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=scripts/generate_prune_prompt.py
BASE_DIR=netlists/LDO

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Base directory not found: $BASE_DIR" >&2
  exit 1
fi

shopt -s nullglob
for CIRCUIT_DIR in "$BASE_DIR"/*/; do
  if [[ -d "$CIRCUIT_DIR" ]]; then
    CIRCUIT_DIR="${CIRCUIT_DIR%/}"
    echo "Generating prune prompt for: $CIRCUIT_DIR"
    $PYTHON "$SCRIPT" --circuit "$CIRCUIT_DIR" --out "$CIRCUIT_DIR/prune_prompt.txt"
  fi
done
shopt -u nullglob
