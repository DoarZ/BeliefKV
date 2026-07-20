#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_PAUSE_FILE="${BELIEFKV_EXPERIMENT_PAUSE_FILE:-/tmp/beliefkv-experiments.paused}"
if [[ -e "${EXPERIMENT_PAUSE_FILE}" ]]; then
  printf 'BeliefKV experiments are paused: %s\n' "${EXPERIMENT_PAUSE_FILE}" >&2
  exit 75
fi

MODEL_PATH="${MODEL_PATH:-/opt/downloaded_models/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-Coder-30B-A3B-Instruct-FP8}"
CONDA_ENV="${CONDA_ENV:-beliefkv}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS-163840}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  printf 'Model not found: %s\n' "${MODEL_PATH}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

server_args=(
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size 1
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --tool-call-parser qwen3_coder
  --enable-cache-report
  --log-level info
)

if [[ "${ENABLE_METRICS}" == "1" ]]; then
  server_args+=(--enable-metrics)
fi

if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
  server_args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
fi
if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
  server_args+=(--disable-cuda-graph)
else
  server_args+=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}")
fi

exec conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m sglang.launch_server \
  "${server_args[@]}" \
  "$@"
