#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="netlists/LDO"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Base directory not found: $BASE_DIR" >&2
  exit 1
fi

shopt -s nullglob
for dir in "$BASE_DIR"/*/; do
  dir="${dir%/}"
  id="$(basename "$dir")"

  for f in \
    "${dir}/Cadence${id}.png" \
    "${dir}/Port${id}.txt" \
    "${dir}/Pagenumber${id}.txt"
  do
    if [[ -f "$f" ]]; then
      echo "Deleting $f"
      rm -f "$f"
    else
      echo "Not found (skip): $f"
    fi
  done
done
shopt -u nullglob
