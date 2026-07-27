from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from beliefkv.policy.reference import (
    EvaluationMode,
    PolicyInput,
    ReactivePolicy,
    ReferencePolicyAdapter,
)


TRACE_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 1
VALID_TRACE_SENSITIVITIES = frozenset(
    {"schedule_invariant", "timing_sensitive", "semantic_race_sensitive"}
)
BASELINE_ID = "B0"


@dataclass(frozen=True)
class ReplaySnapshot:
    sequence: int
    trace_id: str
    trace_sensitivity: str
    policy_input: PolicyInput

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("replay sequence must be non-negative")
        if not self.trace_id:
            raise ValueError("trace_id must be non-empty")
        if self.trace_sensitivity not in VALID_TRACE_SENSITIVITIES:
            raise ValueError(
                f"unsupported trace sensitivity {self.trace_sensitivity!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "sequence": self.sequence,
            "trace_id": self.trace_id,
            "trace_sensitivity": self.trace_sensitivity,
            "policy_input": self.policy_input.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "ReplaySnapshot":
        version = int(raw.get("schema_version", -1))
        if version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported replay trace schema {version}")
        policy_input = raw.get("policy_input")
        if not isinstance(policy_input, Mapping):
            raise TypeError("policy_input must be a mapping")
        return cls(
            sequence=int(raw["sequence"]),
            trace_id=str(raw["trace_id"]),
            trace_sensitivity=str(raw["trace_sensitivity"]),
            policy_input=PolicyInput.from_dict(policy_input),
        )


@dataclass(frozen=True)
class PolicyReplayResult:
    output_dir: Path
    manifest: Mapping[str, object]
    summary: Mapping[str, object]


class PolicyReplayRunner:
    """Replay the B0 reactive baseline over immutable common snapshots.

    This runner deliberately does not report counterfactual JCT. A frozen
    physical trace is sufficient for decision comparison, not for replaying the
    queue, service, and allocation changes caused by a different scheduler.
    """

    def run(
        self,
        trace_path: Path,
        output_dir: Path,
        *,
        run_id: str,
    ) -> PolicyReplayResult:
        source = trace_path.expanduser().resolve()
        destination = output_dir.expanduser().resolve()
        snapshots = load_replay_trace(source)
        self._validate_snapshots(snapshots)
        if not run_id or any(character in run_id for character in "/\\\0"):
            raise ValueError("run_id must be a non-empty path-safe name")
        if destination.exists():
            raise FileExistsError(f"replay output already exists: {destination}")
        staging = destination.with_name(f".{destination.name}.incomplete")
        if staging.exists():
            raise FileExistsError(f"incomplete replay output exists: {staging}")
        staging.mkdir(parents=True)

        policy = ReactivePolicy()
        try:
            policy_summary = self._run_policy(
                policy,
                snapshots,
                staging / f"{BASELINE_ID}_{policy.name}.jsonl",
            )
            sensitivities = sorted(
                {item.trace_sensitivity for item in snapshots}
            )
            manifest = {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "trace_path": str(source),
                "trace_sha256": _sha256(source),
                "trace_id": snapshots[0].trace_id,
                "trace_sensitivities": sensitivities,
                "snapshot_count": len(snapshots),
                "policies": [
                    {
                        "baseline_id": BASELINE_ID,
                        "policy_name": policy.name,
                        "metadata_mode": "online",
                        "evaluation_mode": EvaluationMode.REPLAY.value,
                        "fidelity": "beliefkv_internal_baseline",
                    }
                ],
                "counterfactual_claim": "forbidden_without_whatif_resimulation",
            }
            summary = {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "run_id": run_id,
                "trace_id": snapshots[0].trace_id,
                "snapshot_count": len(snapshots),
                "counterfactual_validity": {
                    sensitivity: _counterfactual_validity(sensitivity)
                    for sensitivity in sensitivities
                },
                "policies": {BASELINE_ID: policy_summary},
            }
            _write_json(staging / "manifest.json", manifest)
            _write_json(staging / "summary.json", summary)
            staging.replace(destination)
        except BaseException:
            _remove_empty_staging(staging)
            raise
        return PolicyReplayResult(destination, manifest, summary)

    def _run_policy(
        self,
        policy: ReactivePolicy,
        snapshots: Sequence[ReplaySnapshot],
        output_path: Path,
    ) -> dict[str, object]:
        adapter = ReferencePolicyAdapter(
            policy,
            evaluation_mode=EvaluationMode.REPLAY,
        )
        action_counts: Counter[str] = Counter()
        unsupported_counts: Counter[str] = Counter()
        assumption_counts: Counter[str] = Counter()
        decision_ids: list[str] = []
        core_fingerprints: list[str] = []
        with output_path.open("x", encoding="utf-8", buffering=1) as stream:
            for snapshot in snapshots:
                decision = adapter.evaluate(snapshot.policy_input)
                core_fingerprint = _core_fingerprint(snapshot.policy_input)
                record = {
                    "schema_version": REPLAY_SCHEMA_VERSION,
                    "sequence": snapshot.sequence,
                    "trace_id": snapshot.trace_id,
                    "trace_sensitivity": snapshot.trace_sensitivity,
                    "counterfactual_validity": _counterfactual_validity(
                        snapshot.trace_sensitivity
                    ),
                    "baseline_id": BASELINE_ID,
                    "policy_name": policy.name,
                    "snapshot_id": snapshot.policy_input.snapshot_id,
                    "core_fingerprint": core_fingerprint,
                    "decision": decision.to_dict(),
                }
                stream.write(
                    json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
                )
                decision_ids.append(decision.output.decision_id)
                core_fingerprints.append(core_fingerprint)
                action_counts.update(
                    f"admission:{item.action.value}"
                    for item in decision.output.admissions
                )
                action_counts.update(
                    f"residency:{item.action.value}"
                    for item in decision.output.residency
                )
                unsupported_counts.update(
                    f"{item.kind.value}:{item.name}"
                    for item in decision.output.unsupported
                )
                assumption_counts.update(decision.output.metadata_assumptions)

        return {
            "policy_name": policy.name,
            "metadata_mode": "online",
            "decision_count": len(decision_ids),
            "unique_decision_count": len(set(decision_ids)),
            "core_snapshot_count": len(set(core_fingerprints)),
            "action_counts": dict(sorted(action_counts.items())),
            "unsupported_counts": dict(sorted(unsupported_counts.items())),
            "metadata_assumption_counts": dict(sorted(assumption_counts.items())),
            "shadow_only": True,
            "jct_reported": False,
        }

    @staticmethod
    def _validate_snapshots(snapshots: Sequence[ReplaySnapshot]) -> None:
        if not snapshots:
            raise ValueError("replay trace must contain at least one snapshot")
        trace_ids = {item.trace_id for item in snapshots}
        if len(trace_ids) != 1:
            raise ValueError("a replay file cannot mix trace IDs")
        sequences = [item.sequence for item in snapshots]
        if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
            raise ValueError("replay snapshot sequences must be strictly increasing")
        timestamps = [item.policy_input.resources.ts_ms for item in snapshots]
        if timestamps != sorted(timestamps):
            raise ValueError("replay resource timestamps must be monotonic")
        snapshot_ids = [item.policy_input.snapshot_id for item in snapshots]
        if len(set(snapshot_ids)) != len(snapshot_ids):
            raise ValueError("replay snapshot IDs must be unique")


def load_replay_trace(path: Path) -> tuple[ReplaySnapshot, ...]:
    snapshots: list[ReplaySnapshot] = []
    source = path.expanduser().resolve()
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, mode="rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise TypeError(f"replay line {line_number} must be an object")
            try:
                snapshots.append(ReplaySnapshot.from_dict(raw))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid replay snapshot at line {line_number}: {error}"
                ) from error
    return tuple(snapshots)


def write_replay_trace(path: Path, snapshots: Iterable[ReplaySnapshot]) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"replay trace already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    opener = gzip.open if destination.suffix == ".gz" else open
    with opener(temporary, mode="xt", encoding="utf-8") as stream:
        for snapshot in snapshots:
            stream.write(
                json.dumps(snapshot.to_dict(), sort_keys=True, allow_nan=False)
                + "\n"
            )
    temporary.replace(destination)


def _core_fingerprint(policy_input: PolicyInput) -> str:
    payload = {
        "runtime_graph": policy_input.runtime_graph.to_dict(),
        "runnable_frontier": [
            item.to_dict() for item in policy_input.runnable_frontier
        ],
        "physical_kv": policy_input.physical_kv.to_dict(),
        "resources": policy_input.resources.to_dict(),
        "identity_mappings": [
            item.to_dict() for item in policy_input.identity_mappings
        ],
        "capabilities": policy_input.capabilities.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _counterfactual_validity(sensitivity: str) -> str:
    if sensitivity == "schedule_invariant":
        return "decision_exact_physical_resimulation_required_for_jct"
    if sensitivity == "timing_sensitive":
        return "decision_only_timing_must_be_resimulated"
    return "optimistic_decision_bound_semantic_path_may_change"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_empty_staging(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        child.unlink(missing_ok=True)
    path.rmdir()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the B0 reactive baseline on PolicyInput snapshots."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = PolicyReplayRunner().run(
        args.trace,
        args.output,
        run_id=args.run_id,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
