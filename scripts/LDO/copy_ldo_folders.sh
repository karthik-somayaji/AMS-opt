#!/usr/bin/env bash

set -euo pipefail

SRC_BASE="/home/lucasjia/data/AnalogGenie/Dataset"
DEST_BASE="/home/lucasjia/projects/AMS-opt/netlists/LDO"

mkdir -p "$DEST_BASE"

for id in {2300..2350}; do
  src_dir="${SRC_BASE}/${id}"
  dest_dir="${DEST_BASE}/${id}"

  if [[ -d "$src_dir" ]]; then
    echo "Copying ${src_dir} -> ${dest_dir}"
    cp -a "$src_dir" "$dest_dir"
  else
    echo "Skipping ${id}: ${src_dir} not found"
  fi
done
