#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s SERVER_DIR [SGLANG_ARGS...]\n' "$0" >&2
  exit 2
fi

SERVER_DIR="$(realpath "$1")"
shift
CONFIG_PATH="${SERVER_DIR}/beliefkv_config.json"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  printf 'Missing server config: %s\n' "${CONFIG_PATH}" >&2
  exit 2
fi

mkdir -p "${SERVER_DIR}"
printf '%s\n' "$$" >"${SERVER_DIR}/server.pid"
server_start_ticks="$(awk '{print $22}' "/proc/$$/stat")"
printf '{"linux_start_time_ticks":%s,"pid":%s,"schema_version":1}\n' \
  "${server_start_ticks}" "$$" >"${SERVER_DIR}/server.pid.json"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PORT="${PORT:-18000}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.952}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-163840}"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
export HICACHE_SIZE_GB="${HICACHE_SIZE_GB:-96}"
export SLEEP_ON_IDLE="${SLEEP_ON_IDLE:-1}"

sleep_args=()
if [[ "${SLEEP_ON_IDLE}" == "1" ]]; then
  # The scheduler event loop busy-waits thousands of times per second when
  # idle, pinning one core at ~100% CPU. sleep_on_idle blocks the loop on a
  # zmq poller until a request/event arrives. Set SLEEP_ON_IDLE=0 to disable.
  sleep_args+=(--sleep-on-idle)
fi

exec "$(dirname "$0")/launch_qwen3_coder_qwencode_smoke.sh" \
  --enable-hierarchical-cache \
  --hicache-size "${HICACHE_SIZE_GB}" \
  --hicache-write-policy write_back \
  --enable-beliefkv \
  --beliefkv-config "${CONFIG_PATH}" \
  "${sleep_args[@]}" \
  "$@" \
  >"${SERVER_DIR}/server.log" 2>&1
