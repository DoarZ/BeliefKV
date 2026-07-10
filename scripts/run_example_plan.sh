#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m beliefkv.cli plan examples/simple_snapshot.json
