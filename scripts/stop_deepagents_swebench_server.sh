#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: %s SERVER_DIR [SERVER_PID]\n' "$0" >&2
  exit 2
fi

server_dir="$(realpath "$1")"
identity_path="${server_dir}/scheduler.pid.json"
ack_path="${server_dir}/shutdown_ack.json"
summary_path="${server_dir}/latest_runtime_summary.json"
server_pid="${2:-}"
server_pgid=""

process_running() {
  local pid="$1"
  [[ -r "/proc/${pid}/stat" ]] || return 1
  [[ "$(awk '{print $3}' "/proc/${pid}/stat")" != "Z" ]]
}

process_group_running() {
  local pgid="$1"
  pgrep -g "${pgid}" >/dev/null 2>&1
}

if [[ ! -f "${identity_path}" ]]; then
  printf 'Missing scheduler identity: %s\n' "${identity_path}" >&2
  exit 66
fi

scheduler_pid="$(jq -er '.pid' "${identity_path}")"
expected_start="$(jq -er '.linux_start_time_ticks' "${identity_path}")"
proc_stat="/proc/${scheduler_pid}/stat"
if [[ ! -r "${proc_stat}" ]]; then
  printf 'Scheduler PID %s is not running\n' "${scheduler_pid}" >&2
  exit 69
fi
actual_start="$(awk '{print $22}' "${proc_stat}")"
process_name="$(tr -d '\n' <"/proc/${scheduler_pid}/comm")"
if [[ "${actual_start}" != "${expected_start}" ]]; then
  printf 'Refusing reused PID %s: start time mismatch\n' "${scheduler_pid}" >&2
  exit 70
fi
if [[ "${process_name}" != sglang::schedul* ]]; then
  printf 'Refusing non-scheduler PID %s (%s)\n' \
    "${scheduler_pid}" "${process_name}" >&2
  exit 70
fi

rm -f "${ack_path}"
kill -TERM "${scheduler_pid}"

deadline=$((SECONDS + 60))
while [[ ! -f "${ack_path}" && ${SECONDS} -lt ${deadline} ]]; do
  sleep 0.1
done
if [[ ! -f "${ack_path}" ]]; then
  printf 'Timed out waiting for BeliefKV shutdown ACK\n' >&2
  exit 124
fi

jq -e --argjson pid "${scheduler_pid}" --argjson start "${expected_start}" '
  .pid == $pid and
  .linux_start_time_ticks == $start and
  .shutdown_state == "acknowledged"
' "${ack_path}" >/dev/null
jq -e '
  .final == true and
  .shutdown_state == "acknowledged" and
  .correctness_gates.shutdown_summary_complete == true
' "${summary_path}" >/dev/null

if [[ -z "${server_pid}" && -f "${server_dir}/server.pid.json" ]]; then
  server_pid="$(jq -er '.pid' "${server_dir}/server.pid.json")"
fi
if [[ -n "${server_pid}" && -d "/proc/${server_pid}" ]]; then
  if [[ -f "${server_dir}/server.pid.json" ]]; then
    expected_server_start="$(
      jq -er '.linux_start_time_ticks' "${server_dir}/server.pid.json"
    )"
    actual_server_start="$(awk '{print $22}' "/proc/${server_pid}/stat")"
    if [[ "${actual_server_start}" != "${expected_server_start}" ]]; then
      printf 'Refusing reused server PID %s: start time mismatch\n' \
        "${server_pid}" >&2
      exit 70
    fi
  fi
  server_pgid="$(ps -o pgid= -p "${server_pid}" | tr -d ' ')"
  if [[ -z "${server_pgid}" || "${server_pgid}" == "$(ps -o pgid= -p $$ | tr -d ' ')" ]]; then
    printf 'Refusing unsafe server process group for PID %s\n' "${server_pid}" >&2
    exit 70
  fi
  kill -TERM -- "-${server_pgid}"
fi

# The BeliefKV SIGTERM handler acknowledges after transactions and audit state
# are durable. Some SGLang versions return to the scheduler loop after that
# first signal, so ACK alone is not proof that CUDA/Host allocations are gone.
if process_running "${scheduler_pid}"; then
  actual_start="$(awk '{print $22}' "${proc_stat}")"
  if [[ "${actual_start}" != "${expected_start}" ]]; then
    printf 'Refusing reused scheduler PID %s after shutdown ACK\n' \
      "${scheduler_pid}" >&2
    exit 70
  fi
  kill -TERM "${scheduler_pid}"
fi

exit_deadline=$((SECONDS + 30))
while process_running "${scheduler_pid}" && [[ ${SECONDS} -lt ${exit_deadline} ]]; do
  sleep 0.1
done
if process_running "${scheduler_pid}"; then
  actual_start="$(awk '{print $22}' "${proc_stat}")"
  if [[ "${actual_start}" != "${expected_start}" ]]; then
    printf 'Refusing reused scheduler PID %s during exit verification\n' \
      "${scheduler_pid}" >&2
    exit 70
  fi
  printf 'Scheduler PID %s acknowledged shutdown but did not exit\n' \
    "${scheduler_pid}" >&2
  exit 124
fi

if [[ -n "${server_pgid}" ]]; then
  group_deadline=$((SECONDS + 30))
  while process_group_running "${server_pgid}" && [[ ${SECONDS} -lt ${group_deadline} ]]; do
    sleep 0.1
  done
  if process_group_running "${server_pgid}"; then
    printf 'SGLang process group %s did not exit after shutdown ACK\n' \
      "${server_pgid}" >&2
    exit 124
  fi
fi

printf 'BeliefKV scheduler shutdown acknowledged: %s\n' "${ack_path}"
