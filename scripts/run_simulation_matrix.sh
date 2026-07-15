#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m beliefkv.cli matrix \
  examples/subagent_ablation_matrix.json \
  --output-root experiments/results "$@"
