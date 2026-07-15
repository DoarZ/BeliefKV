# ClawTrace Dataset Analysis

Date: 2026-07-13

## Download Location

The ClawTrace repository was downloaded from the official GitHub repository:

```text
https://github.com/epsilla-cloud/clawtrace.git
```

Local path:

```text
/home/longhao/datasets/clawtrace
```

Local checkout:

```text
commit: f472dc4
size: 23M
```

## What Was Checked

The analysis checked whether the public ClawTrace repository contains a released
trace dataset with explicit subagent lifecycle labels that BeliefKV can use
directly.

Checked file patterns:

```text
trace.jsonl
tracecard.yaml
*.jsonl
runs/
data/
dataset/
datasets/
```

Result:

```text
No released trace.jsonl or tracecard.yaml data files were found in the checkout.
No benchmark runs directory with captured traces was included.
```

The repository contains:

- instrumentation plugin code;
- event schema;
- SQL transforms;
- UI code;
- experiment scripts;
- task manifests;
- skills used by their experiments.

It does not include the generated run artifacts.

## Subagent Support in ClawTrace

ClawTrace explicitly supports subagent events in its instrumentation layer.

The plugin maps OpenClaw hooks to ingest events:

```text
subagent_spawning -> subagent_spawn
subagent_ended    -> subagent_join
```

Relevant files:

```text
/home/longhao/datasets/clawtrace/plugins/clawtrace/README.md
/home/longhao/datasets/clawtrace/plugins/clawtrace/src/types.ts
/home/longhao/datasets/clawtrace/plugins/clawtrace/src/tracker.ts
```

The ingest event schema includes:

```text
subagent_spawn
subagent_join
```

The tracker emits payload fields such as:

```text
runId
requesterSessionKey
childSessionKey
subagentId
label
mode
threadRequested
requester
targetSessionKey
targetKind
reason
endedAt
outcome
error
```

The SQL layer also maps spans with parent-child relationships into a subagent
actor type:

```text
actor_type = subagent
```

Relevant files:

```text
/home/longhao/datasets/clawtrace/sql/databricks/silver_tables/20_materialize_puppygraph_tables.sql
```

## TraceCard Subagent Field

The TraceCard compiler contains a `sub_agents` field:

```text
sub_agents: list[dict]
```

Relevant file:

```text
/home/longhao/datasets/clawtrace/costcraft/costcraft/tracecard.py
```

However, the current TraceCard builder detects subagents heuristically from
tool names:

```python
if tool_name in {"task", "agent", "subagent", "spawn_agent"}:
    sub_agent_spawns.append(...)
```

It then estimates whether child output was used in the final answer with a
Jaccard-overlap heuristic:

```text
output_used_in_final_heuristic
_note: heuristic, not authoritative
```

This is useful for cost/skill analysis, but it is not authoritative lifecycle
ground truth for serving-level KV scheduling.

## Reproduction Bundle

The paper experiment README states that scripts produce artifacts under:

```text
<benchmark>/runs/<task_id>/<condition>_<seed>/
  trace.jsonl
  tracecard.yaml
  grading.json
  run_meta.json
```

But those run outputs are not included in the repository. The checkout only
contains the scripts and manifests required to reproduce them.

## Conclusion

ClawTrace provides a useful subagent instrumentation schema, but the public
repository does not currently provide a ready-to-use dataset with explicit
subagent lifecycle annotations.

For BeliefKV, this means:

```text
Do not make explicit subagent lifecycle metadata the main experimental
dependency at this stage.
```

The interface-based subagent design should be treated as a future extension,
not the current core path.

## Impact on BeliefKV Direction

The project should temporarily avoid a design that requires agent frameworks to
send explicit fields such as:

```text
subagent_id
parent_agent_id
subagent_spawn
subagent_join
parent_join_start
parent_join_end
```

Instead, the current core should focus on metadata-free or weak-metadata
serving signals that are available in real traces:

```text
request timing
tool call timing
tool name/type
tool latency
tool result size
prompt growth
prefix reuse
LLM prefill/decode phase
workflow/session id
```

The stronger near-term direction is:

```text
Action-unlock-aware serving and KV management without requiring explicit
subagent lifecycle labels.
```

Subagent-aware lifecycle management remains a good future direction, but should
not be the primary experimental claim until a real labeled dataset or a low-cost
instrumented benchmark is available.

## Recommended Next Step

Use TraceLab as the primary real-trace motivation and evaluate generic
agent-native scheduling signals:

```text
time-to-tool-call
time-to-action
tool-wait overlap
observation size
prompt growth after tool result
workflow critical-path delay
```

If subagent-like events are present only as opaque tools, treat them as
long-latency tool/action boundaries rather than explicit child-agent lifecycles.
