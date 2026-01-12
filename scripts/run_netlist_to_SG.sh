#!/usr/bin/env bash

PYTHON=python
SCRIPT=get_netlist_to_SG.py
BASE_DIR=netlists/diff_amps

for i in {69..97}; do
    NETLIST_PATH="${BASE_DIR}/${i}/${i}.cir"
    OUTPUT_JSON="${BASE_DIR}/${i}/str_graph.json"

    if [[ -f "$NETLIST_PATH" ]]; then
        echo "Processing ${NETLIST_PATH}"
        $PYTHON "$SCRIPT" \
            --netlist-path "$NETLIST_PATH" \
            --output-jsonl "$OUTPUT_JSON"
    else
        echo "Skipping ${i}: ${NETLIST_PATH} not found"
    fi
done

for i in {614..621}; do
    NETLIST_PATH="${BASE_DIR}/${i}/${i}.cir"
    OUTPUT_JSON="${BASE_DIR}/${i}/str_graph.json"

    if [[ -f "$NETLIST_PATH" ]]; then
        echo "Processing ${NETLIST_PATH}"
        $PYTHON "$SCRIPT" \
            --netlist-path "$NETLIST_PATH" \
            --output-jsonl "$OUTPUT_JSON"
    else
        echo "Skipping ${i}: ${NETLIST_PATH} not found"
    fi
done

for i in {921..932}; do
    NETLIST_PATH="${BASE_DIR}/${i}/${i}.cir"
    OUTPUT_JSON="${BASE_DIR}/${i}/str_graph.json"

    if [[ -f "$NETLIST_PATH" ]]; then
        echo "Processing ${NETLIST_PATH}"
        $PYTHON "$SCRIPT" \
            --netlist-path "$NETLIST_PATH" \
            --output-jsonl "$OUTPUT_JSON"
    else
        echo "Skipping ${i}: ${NETLIST_PATH} not found"
    fi
done