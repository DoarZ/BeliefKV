#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  printf 'Usage: %s RUN_DIR GPU PORT FRAGMENTATION ARM REPETITION GATE_ID\n' "$0" >&2
  exit 2
fi

run_dir="$(realpath -m "$1")"
gpu="$2"
port="$3"
fragmentation="$4"
arm="$5"
repetition="$6"
gate_id="$7"

case "${fragmentation}" in
  low) chunked_prefill_size=4096 ;;
  high) chunked_prefill_size=256 ;;
  *) printf 'Unknown fragmentation: %s\n' "${fragmentation}" >&2; exit 2 ;;
esac
case "${arm}" in
  treatment|control) ;;
  *) printf 'Unknown arm: %s\n' "${arm}" >&2; exit 2 ;;
esac

# A free-looking GPU can still belong to a restarting external job. Require a
# short stable window before creating an immutable arm directory.
for _ in $(seq 1 5); do
  free_mib="$(
    nvidia-smi --id="${gpu}" --query-gpu=memory.free \
      --format=csv,noheader,nounits | tr -d ' '
  )"
  if [[ ! "${free_mib}" =~ ^[0-9]+$ || "${free_mib}" -lt 47000 ]]; then
    printf 'GPU %s has only %s MiB free; arm not created\n' \
      "${gpu}" "${free_mib}" >&2
    exit 75
  fi
  sleep 2
done

if [[ -e "${run_dir}" ]]; then
  printf 'Refusing existing arm directory: %s\n' "${run_dir}" >&2
  exit 73
fi
mkdir -p "${run_dir}"

config_args=(
  --server-dir "${run_dir}/server"
  --queue-service-observer
  --request-queue-timeout-seconds 1800
)
if [[ "${arm}" == treatment ]]; then
  config_args+=(
    --enable-online-joint
    --enable-observed-admission
    --enable-running-retraction
    --enable-restore-micro-gate
    --restore-micro-gate-id "${gate_id}"
    --joint-workflow-active-window 3
  )
else
  config_args+=(--disable-reactive-transfer --disable-policy-shadow)
fi

conda run --no-capture-output -n beliefkv \
  python scripts/prepare_deepagents_server_config.py "${config_args[@]}"

launcher_pid=""
stopped=0
cleanup() {
  if [[ "${stopped}" == 0 && -f "${run_dir}/server/scheduler.pid.json" ]]; then
    scripts/stop_deepagents_swebench_server.sh "${run_dir}/server" || true
  elif [[ -n "${launcher_pid}" ]] && kill -0 "${launcher_pid}" 2>/dev/null; then
    kill -TERM -- "-${launcher_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${gpu}" \
PORT="${port}" \
MEM_FRACTION_STATIC=0.952 \
MAX_TOTAL_TOKENS=163840 \
CONTEXT_LENGTH=65536 \
MAX_RUNNING_REQUESTS=2 \
HICACHE_SIZE_GB=96 \
CHUNKED_PREFILL_SIZE="${chunked_prefill_size}" \
setsid scripts/launch_deepagents_swebench_server.sh "${run_dir}/server" &
launcher_pid=$!

ready=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${launcher_pid}" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  tail -40 "${run_dir}/server/server.log" >&2 || true
  exit 70
fi

workload_args=(
  --output-dir "${run_dir}/workload"
  --runtime-audit "${run_dir}/server/runtime_audit.jsonl"
  --base-url "http://127.0.0.1:${port}/v1"
  --victim-prompt-words 27043
  --anchor-prompt-words 8000
  --replacement-prompt-words 32000
  --anchor-output-tokens 4096
  --replacement-output-tokens 128
  --request-timeout-seconds 900
)
if [[ "${arm}" == treatment ]]; then
  workload_args+=(
    --gate-id "${gate_id}"
    --victim-output-tokens 1024
    --replacement-submit-delay-seconds 5
  )
else
  workload_args+=(--no-d2h-control --victim-output-tokens 684)
fi
conda run --no-capture-output -n beliefkv \
  python scripts/run_restore_micro_gate.py "${workload_args[@]}"

scripts/stop_deepagents_swebench_server.sh "${run_dir}/server"
stopped=1
wait "${launcher_pid}" || true

if [[ "${arm}" == treatment ]]; then
  conda run --no-capture-output -n beliefkv \
    python scripts/verify_restore_micro_gate.py \
    --runtime-audit "${run_dir}/server/runtime_audit.jsonl" \
    --runtime-summary "${run_dir}/server/latest_runtime_summary.json" \
    --gate-id "${gate_id}" \
    --output "${run_dir}/restore_micro_gate_validation.json"
fi

python - "${run_dir}" "${gpu}" "${fragmentation}" "${arm}" "${repetition}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "arm_manifest.json"
path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "gpu_id": sys.argv[2],
            "bytes_class": "large",
            "fragmentation_class": sys.argv[3],
            "arm": sys.argv[4],
            "repetition": int(sys.argv[5]),
            "completed": True,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
trap - EXIT
