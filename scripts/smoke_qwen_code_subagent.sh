#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/experiments/raw/qwen-code-smoke/${RUN_ID}}"
OUTPUT_PATH="${OUTPUT_DIR}/explicit_explore.json"

mkdir -p "${OUTPUT_DIR}"

"${ROOT_DIR}/scripts/run_qwen_code_local.sh" \
  -p 'Use the agent tool with subagent_type Explore to inspect beliefkv/control/controller.py. Ask it for the first class and a one-sentence description. Do not modify files.' \
  --output-format json \
  --max-wall-time 5m \
  --max-session-turns 12 \
  --max-tool-calls 16 \
  | tee "${OUTPUT_PATH}"

jq -e '
  map(select(.type == "result" and .subtype == "success"))
  | last
  | (.stats.tools.byName.agent.success >= 1
     and .stats.tools.byName.read_file.success >= 1)
' "${OUTPUT_PATH}" >/dev/null

printf 'Qwen Code parent/subagent smoke passed: %s\n' "${OUTPUT_PATH}"
