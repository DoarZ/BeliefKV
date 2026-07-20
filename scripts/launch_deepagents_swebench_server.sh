#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s SERVER_DIR\n' "$0" >&2
  exit 2
fi

SERVER_DIR="$(realpath "$1")"
CONFIG_PATH="${SERVER_DIR}/beliefkv_config.json"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  printf 'Missing server config: %s\n' "${CONFIG_PATH}" >&2
  exit 2
fi

mkdir -p "${SERVER_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.952}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-163840}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"

exec "$(dirname "$0")/launch_qwen3_coder_qwencode_smoke.sh" \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-write-policy write_back \
  --enable-beliefkv \
  --beliefkv-config "${CONFIG_PATH}" \
  >"${SERVER_DIR}/server.log" 2>&1
