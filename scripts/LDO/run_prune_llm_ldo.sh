#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
SCRIPT=scripts/run_prune_llm.py

$PYTHON "$SCRIPT" \
  --base-dir netlists/LDO \
  --prompt-filename prune_prompt.txt
