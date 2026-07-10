# Runtime Patches

Runtime patches live here once the policy is connected to SGLang 0.5.2rc1.

Patch groups should be kept small:

- `metadata.patch`: request and workflow metadata plumbing;
- `snapshot.patch`: scheduler and cache snapshot export;
- `actions.patch`: keep, offload, prefetch, and recompute action execution;
- `metrics.patch`: runtime counters and trace output.

Each patch should have a matching unit or replay test in this repository.
