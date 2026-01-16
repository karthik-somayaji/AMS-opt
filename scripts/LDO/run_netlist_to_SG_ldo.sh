#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=get_netlist_to_SG.py
BASE_DIR=netlists/LDO

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Base directory not found: $BASE_DIR" >&2
  exit 1
fi

shopt -s nullglob
for CIRCUIT_DIR in "$BASE_DIR"/*/; do
  CIRCUIT_DIR="${CIRCUIT_DIR%/}"
  netlist_files=("$CIRCUIT_DIR"/*.cir "$CIRCUIT_DIR"/*.sp "$CIRCUIT_DIR"/*.net)
  if (( ${#netlist_files[@]} == 0 )); then
    echo "Skipping $(basename "$CIRCUIT_DIR"): no netlist found"
    continue
  fi

  NETLIST_PATH="${netlist_files[0]}"
  OUTPUT_JSON="${CIRCUIT_DIR}/str_graph.json"
  echo "Processing ${NETLIST_PATH}"
  $PYTHON "$SCRIPT" --netlist-path "$NETLIST_PATH" --output-jsonl "$OUTPUT_JSON"
done
shopt -u nullglob
