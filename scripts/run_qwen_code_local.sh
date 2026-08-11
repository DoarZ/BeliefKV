#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_CODE_ROOT="${QWEN_CODE_ROOT:-${HOME}/.local/qwen-code}"
NODE_ROOT="${NODE_ROOT:-/opt/node-v22.20.0-linux-x64}"
QWEN_BIN="${QWEN_BIN:-${QWEN_CODE_ROOT}/bin/qwen}"
QWEN_HOME="${QWEN_HOME:-${ROOT_DIR}/experiments/raw/qwen-code/home}"
QWEN_RUNTIME_DIR="${QWEN_RUNTIME_DIR:-${ROOT_DIR}/experiments/raw/qwen-code/runtime}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:18000/v1}"
OPENAI_LOG_DIR="${OPENAI_LOG_DIR:-${ROOT_DIR}/experiments/raw/qwen-code/openai-logs}"
SETTINGS_TEMPLATE="${QWEN_SETTINGS_TEMPLATE:-${ROOT_DIR}/configs/qwen_code/qwen3_coder_30b_a3b_fp8_matrix.json}"
SANDBOX_PROXY_URL=""
if [[ -n "${QWEN_SANDBOX_PROXY_COMMAND:-}" ]]; then
  SANDBOX_PROXY_URL="${QWEN_SANDBOX_PROXY_URL:-http://qwen-code-sandbox-proxy:8877}"
fi

for executable in "${QWEN_BIN}" "${NODE_ROOT}/bin/node"; do
  if [[ ! -x "${executable}" ]]; then
    printf 'Required executable not found: %s\n' "${executable}" >&2
    exit 2
  fi
done
if ! command -v jq >/dev/null 2>&1; then
  printf 'jq is required to materialize Qwen Code settings.\n' >&2
  exit 2
fi

mkdir -p "${QWEN_HOME}" "${QWEN_RUNTIME_DIR}" "${OPENAI_LOG_DIR}"
settings_tmp="${QWEN_HOME}/settings.json.tmp.$$"
trap 'rm -f "${settings_tmp}"' EXIT
jq \
  --arg base_url "${OPENAI_BASE_URL}" \
  --arg log_dir "${OPENAI_LOG_DIR}" \
  --arg proxy_url "${SANDBOX_PROXY_URL}" \
  '.modelProviders.openai[0].baseUrl = $base_url
   | .model.openAILoggingDir = $log_dir
   | if $proxy_url == "" then del(.proxy) else .proxy = $proxy_url end' \
  "${SETTINGS_TEMPLATE}" > "${settings_tmp}"
chmod 600 "${settings_tmp}"
mv "${settings_tmp}" "${QWEN_HOME}/settings.json"
trap - EXIT

export PATH="${NODE_ROOT}/bin:${QWEN_CODE_ROOT}/bin:${PATH}"
export QWEN_HOME
export QWEN_RUNTIME_DIR
export OPENAI_API_KEY="${OPENAI_API_KEY:-not-needed}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
unset NODE_TLS_REJECT_UNAUTHORIZED
unset DEBUG DEBUG_PORT

if jq -e '.tools.sandbox == true or (.tools.sandbox | type == "string")' \
  "${QWEN_HOME}/settings.json" >/dev/null; then
  approval_mode="$(jq -r '.tools.approvalMode // "default"' "${QWEN_HOME}/settings.json")"
  if [[ "${approval_mode}" == "yolo" \
    && "${QWEN_WORKSPACE_DISPOSABLE:-0}" != "1" \
    && ! -f "$(pwd -P)/.beliefkv-disposable-workspace" ]]; then
    printf '%s\n' \
      'Refusing sandboxed yolo mode in a non-disposable workspace.' \
      'Use a per-run clone and set QWEN_WORKSPACE_DISPOSABLE=1.' >&2
    exit 2
  fi
  sandbox_tmp="${QWEN_RUNTIME_DIR}/tmp"
  sandbox_home="${QWEN_RUNTIME_DIR}/home"
  mkdir -p "${sandbox_tmp}" "${sandbox_home}"
  export TMPDIR="${sandbox_tmp}"
  export HOME="${sandbox_home}"
  export SANDBOX_FLAGS="${SANDBOX_FLAGS:---cap-drop=ALL --cap-add=CHOWN --cap-add=SETGID --cap-add=SETUID --security-opt=no-new-privileges --pids-limit=1024 --memory=16g --cpus=16}"
  if [[ -n "${QWEN_SANDBOX_NETWORK:-}" ]]; then
    export SANDBOX_FLAGS="${SANDBOX_FLAGS} --network=${QWEN_SANDBOX_NETWORK}"
  fi
  export SANDBOX_SET_UID_GID="${SANDBOX_SET_UID_GID:-true}"
  case "${OPENAI_LOG_DIR}" in
    "${QWEN_HOME}"/*|"${QWEN_RUNTIME_DIR}"/*) ;;
    *)
      log_mount="${OPENAI_LOG_DIR}:${OPENAI_LOG_DIR}:rw"
      export SANDBOX_MOUNTS="${SANDBOX_MOUNTS:+${SANDBOX_MOUNTS},}${log_mount}"
      ;;
  esac
fi

qwen_args=()
if [[ -n "${SANDBOX_PROXY_URL}" ]]; then
  qwen_args+=(--proxy "${SANDBOX_PROXY_URL}")
fi

exec "${QWEN_BIN}" "${qwen_args[@]}" "$@"
