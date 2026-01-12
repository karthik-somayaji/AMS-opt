#!/usr/bin/env bash

PYTHON=python
SCRIPT=scripts/generate_fun_graph_prompt.py
BASE_DIR=netlists/diff_amps

# 69–97
for i in {69..97}; do
    CIRCUIT_DIR="${BASE_DIR}/${i}"

    if [[ -d "$CIRCUIT_DIR" ]]; then
        echo "Processing circuit directory: $CIRCUIT_DIR"
        $PYTHON "$SCRIPT" --circuit "$CIRCUIT_DIR"
    else
        echo "Skipping ${i}: directory not found"
    fi
done

# 614–621
for i in {614..621}; do
    CIRCUIT_DIR="${BASE_DIR}/${i}"

    if [[ -d "$CIRCUIT_DIR" ]]; then
        echo "Processing circuit directory: $CIRCUIT_DIR"
        $PYTHON "$SCRIPT" --circuit "$CIRCUIT_DIR"
    else
        echo "Skipping ${i}: directory not found"
    fi
done

# 921–932
for i in {921..932}; do
    CIRCUIT_DIR="${BASE_DIR}/${i}"

    if [[ -d "$CIRCUIT_DIR" ]]; then
        echo "Processing circuit directory: $CIRCUIT_DIR"
        $PYTHON "$SCRIPT" --circuit "$CIRCUIT_DIR"
    else
        echo "Skipping ${i}: directory not found"
    fi
done
