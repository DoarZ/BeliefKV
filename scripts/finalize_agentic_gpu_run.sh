#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 64
fi

run_dir=$1
manifest="$run_dir/workloads/manifest.json"
direct_summary="$run_dir/workloads/summary.json"
summary="$run_dir/server/latest_runtime_summary.json"
audit="$run_dir/server/runtime_audit.jsonl"
config="$run_dir/server/beliefkv_config.json"
validation="$run_dir/transfer_validation.json"
timeline="$run_dir/kv_transfer_timeline.html"

for required in "$manifest" "$summary" "$audit" "$config"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 66
  fi
done

# Visualization is a publication artifact. Failed/censored runs retain their
# raw telemetry, but must not produce a timeline that looks like a valid run.
if [[ -f "$direct_summary" ]]; then
  jq -e '
    .workflow_count > 0 and
    .system_jct_eligible_workflows == .workflow_count and
    all(.workflows[]; .system_jct_eligible == true)
  ' "$direct_summary" >/dev/null
else
  jq -e '
    .experiment_valid == true and
    (.results | length) > 0 and
    all(.results[]; .clean_jct_eligible == true)
  ' "$manifest" >/dev/null
fi

jq -e '
  .correctness_gates.all_online_actions_have_source_joint_plan_id == true and
  .correctness_gates.no_pending_transactions == true and
  .correctness_gates.shutdown_summary_complete == true
' "$summary" >/dev/null

python -m beliefkv.cli validate-transfer-telemetry \
  "$audit" \
  --config "$config" \
  --output "$validation"

jq -e '
  .command_integrity.passes == true and
  .resource_consistency.hbm_mirror_is_allocator_subset == true and
  .resource_consistency.host_residency_matches_page_index == true
' "$validation" >/dev/null

python -m beliefkv.cli render-transfer-timeline \
  "$audit" \
  "$timeline"

echo "$timeline"
