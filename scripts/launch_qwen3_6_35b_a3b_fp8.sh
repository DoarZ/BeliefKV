#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/opt/downloaded_models/Qwen/Qwen3.6-35B-A3B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B-FP8}"
CONDA_ENV="${CONDA_ENV:-beliefkv}"
CONDA_BIN="${CONDA_BIN:-/home/longhao/miniconda3/bin/conda}"
EXPECTED_SGLANG_VERSION="${EXPECTED_SGLANG_VERSION:-0.5.10.post1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.952}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
CUDA_GRAPH_BATCH_SIZES="${CUDA_GRAPH_BATCH_SIZES:-1 2 4 8}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"

if [[ ! -x "${CONDA_BIN}" ]]; then
  printf 'Conda executable not found: %s\n' "${CONDA_BIN}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  printf 'Model not found: %s\n' "${MODEL_PATH}" >&2
  exit 2
fi

actual_sglang_version="$(
  "${CONDA_BIN}" run -n "${CONDA_ENV}" python -c \
    'import importlib.metadata as m; print(m.version("sglang"))'
)"
if [[ "${actual_sglang_version}" != "${EXPECTED_SGLANG_VERSION}" ]]; then
  printf 'Expected SGLang %s, found %s in conda env %s\n' \
    "${EXPECTED_SGLANG_VERSION}" "${actual_sglang_version}" "${CONDA_ENV}" >&2
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
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --disable-piecewise-cuda-graph
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
  read -r -a cuda_graph_batch_sizes <<<"${CUDA_GRAPH_BATCH_SIZES}"
  server_args+=(
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
    --cuda-graph-bs "${cuda_graph_batch_sizes[@]}"
  )
fi

exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" \
  python -m sglang.launch_server \
  "${server_args[@]}" \
  "$@"
