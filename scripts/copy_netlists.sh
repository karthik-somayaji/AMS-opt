#!/usr/bin/env bash

SRC_BASE="data/AnalogGenie/Dataset"
DST_BASE="projects/AMS-opt/netlists/diff_amps"

copy_range () {
    for i in "$@"; do
        SRC_DIR="${SRC_BASE}/${i}"
        DST_DIR="${DST_BASE}/${i}"

        if [[ -d "$SRC_DIR" ]]; then
            if [[ -d "$DST_DIR" ]]; then
                echo "Skipping ${i}: destination already exists"
            else
                echo "Copying ${i}"
                cp -r "$SRC_DIR" "$DST_BASE/"
            fi
        else
            echo "Skipping ${i}: source does not exist"
        fi
    done
}

# 69–97
copy_range $(seq 69 97)

# 614–621
copy_range $(seq 614 621)

# 921–932
copy_range $(seq 921 932)
