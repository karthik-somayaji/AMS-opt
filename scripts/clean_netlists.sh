#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="netlists/diff_amps"

delete_for () {
  local i="$1"
  local dir="${BASE_DIR}/${i}"

  [[ -d "$dir" ]] || { echo "Skipping ${i}: directory not found"; return 0; }

  for f in \
    "${dir}/Cadence${i}.png" \
    "${dir}/Port${i}.txt" \
    "${dir}/Pagenumber${i}.txt"
  do
    if [[ -f "$f" ]]; then
      echo "Deleting $f"
      rm -f "$f"
    else
      echo "Not found (skip): $f"
    fi
  done
}

# 69–97
for i in {69..97}; do delete_for "$i"; done

# 614–621
for i in {614..621}; do delete_for "$i"; done

# 921–932
for i in {921..932}; do delete_for "$i"; done
