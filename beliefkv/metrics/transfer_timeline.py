from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable

from beliefkv.metrics.summary import percentile


@dataclass(frozen=True)
class TimelineTransfer:
    command_id: str
    direction: str
    submit_ts_ms: float
    start_ts_ms: float | None
    complete_ts_ms: float
    actual_bytes: int
    closure_bytes: int
    status: str
    context_id: str | None
    command_kind: str
    reason: str
    measurement: str


@dataclass(frozen=True)
class TimelineResourcePoint:
    ts_ms: float
    hbm_used_bytes: int | None
    hbm_capacity_bytes: int | None
    host_used_bytes: int | None
    host_capacity_bytes: int | None
    source: str
    untracked_allocator_delta_bytes: int | None = None
    engine_locked_gpu_bytes: int | None = None
    engine_lock_ref_gpu_bytes: int | None = None
    engine_lock_full_attribution_coverage: float | None = None
    locked_but_not_served_gpu_bytes_100ms: int | None = None
    locked_but_not_served_gpu_bytes_500ms: int | None = None
    closure_blocked_gpu_bytes: int | None = None
    migratable_gpu_bytes: int | None = None
    dual_resident_gpu_bytes: int | None = None


@dataclass(frozen=True)
class TransferTimeline:
    run_id: str
    source_path: str
    start_ts_ms: float
    end_ts_ms: float
    transfers: tuple[TimelineTransfer, ...]
    resources: tuple[TimelineResourcePoint, ...]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_path": self.source_path,
            "start_ts_ms": self.start_ts_ms,
            "end_ts_ms": self.end_ts_ms,
            "summary": self.summary,
            "transfers": [asdict(item) for item in self.transfers],
            "resources": [asdict(item) for item in self.resources],
        }


def load_transfer_timeline(
    audit_path: Path,
    *,
    run_id: str | None = None,
    metrics_path: Path | None = None,
    kv_bytes_per_token: int | None = None,
    hbm_capacity_bytes: int | None = None,
) -> TransferTimeline:
    records = _read_jsonl(audit_path)
    selected_run_id = _select_run_id(records, run_id)
    selected = [
        record
        for record in records
        if selected_run_id == "unscoped" or record.get("run_id") == selected_run_id
    ]
    transfers = _extract_telemetry(selected)
    if not transfers:
        transfers = _extract_legacy_transfers(selected)
    resources = _extract_resources(selected)
    if metrics_path is not None:
        resources.extend(
            _extract_sglang_metrics(
                _read_jsonl(metrics_path),
                kv_bytes_per_token=kv_bytes_per_token,
                hbm_capacity_bytes=hbm_capacity_bytes,
            )
        )
    resources.sort(key=lambda item: item.ts_ms)
    transfers.sort(key=lambda item: (item.submit_ts_ms, item.command_id))
    timestamps = [
        value
        for transfer in transfers
        for value in (transfer.submit_ts_ms, transfer.complete_ts_ms)
    ] + [item.ts_ms for item in resources]
    if not timestamps:
        raise ValueError("no transfer or resource timeline records were found")
    start_ts_ms = min(timestamps)
    end_ts_ms = max(timestamps)
    summary = _summarize(transfers, resources, start_ts_ms, end_ts_ms)
    return TransferTimeline(
        run_id=selected_run_id,
        source_path=str(audit_path.resolve()),
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
        transfers=tuple(transfers),
        resources=tuple(resources),
        summary=summary,
    )


def render_transfer_timeline(
    timeline: TransferTimeline,
    output_path: Path,
    *,
    title: str = "BeliefKV KV Transfer Timeline",
) -> tuple[Path, Path]:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_path.with_suffix(".json")
    _atomic_write(
        data_path,
        json.dumps(timeline.to_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    _atomic_write(output_path, _render_html(timeline, title=title))
    return output_path, data_path


def summarize_transfer_timeline(
    transfers: Iterable[TimelineTransfer],
    resources: Iterable[TimelineResourcePoint],
    *,
    start_ts_ms: float,
    end_ts_ms: float,
) -> dict[str, Any]:
    """Build the renderer's complete summary for synthetic or replay timelines."""
    return _summarize(
        list(transfers),
        list(resources),
        start_ts_ms,
        end_ts_ms,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _select_run_id(records: Iterable[dict[str, Any]], requested: str | None) -> str:
    run_ids = [str(item["run_id"]) for item in records if item.get("run_id")]
    available = tuple(dict.fromkeys(run_ids))
    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"run_id {requested!r} is not present; available={list(available)!r}"
            )
        return requested
    if len(available) > 1:
        raise ValueError(
            "audit contains multiple run_ids; select one explicitly: "
            f"{list(available)!r}"
        )
    return available[0] if available else "unscoped"


def _extract_telemetry(records: Iterable[dict[str, Any]]) -> list[TimelineTransfer]:
    result: list[TimelineTransfer] = []
    for record in records:
        if record.get("event") != "transfer_telemetry":
            continue
        result.append(
            TimelineTransfer(
                command_id=str(record["command_id"]),
                direction=str(record["direction"]),
                submit_ts_ms=float(record["submit_ts_ms"]),
                start_ts_ms=(
                    float(record["start_ts_ms"])
                    if record.get("start_ts_ms") is not None
                    else None
                ),
                complete_ts_ms=float(record["complete_ts_ms"]),
                actual_bytes=int(record.get("actual_bytes", 0)),
                closure_bytes=int(record.get("closure_bytes", 0)),
                status=str(record.get("status", "unknown")),
                context_id=(
                    str(record["context_id"])
                    if record.get("context_id") is not None
                    else None
                ),
                command_kind=str(record.get("command_kind", "")),
                reason=str(record.get("reason", "")),
                measurement=str(
                    record.get("telemetry_origin", "backend_telemetry")
                ),
            )
        )
    return result


def _extract_legacy_transfers(
    records: Iterable[dict[str, Any]],
) -> list[TimelineTransfer]:
    dispatched: dict[str, dict[str, Any]] = {}
    result: list[TimelineTransfer] = []
    for record in records:
        event = record.get("event")
        command_id = str(record.get("command_id", ""))
        if event == "transfer_dispatched" and command_id:
            dispatched[command_id] = record
        elif event == "transfer_acknowledged" and command_id in dispatched:
            start = dispatched.pop(command_id)
            actions = start.get("action_counts", {})
            d2h = int(actions.get("start_d2h", 0)) if isinstance(actions, dict) else 0
            h2d = int(actions.get("start_h2d", 0)) if isinstance(actions, dict) else 0
            if d2h and h2d:
                direction = "mixed"
            elif d2h:
                direction = "d2h"
            elif h2d:
                direction = "h2d"
            else:
                direction = "reclaim"
            result.append(
                TimelineTransfer(
                    command_id=command_id,
                    direction=direction,
                    submit_ts_ms=float(start["ts_ms"]),
                    start_ts_ms=None,
                    complete_ts_ms=float(record["ts_ms"]),
                    actual_bytes=int(record.get("actual_bytes", 0)),
                    closure_bytes=int(start.get("selected_bytes", 0)),
                    status=str(record.get("status", "unknown")),
                    context_id=(
                        str(start["context_id"])
                        if start.get("context_id") is not None
                        else None
                    ),
                    command_kind=str(start.get("kind", "")),
                    reason=str(record.get("reason", "")),
                    measurement="legacy_dispatch_ack_aggregate",
                )
            )
    return result


def _extract_resources(
    records: Iterable[dict[str, Any]],
) -> list[TimelineResourcePoint]:
    result: list[TimelineResourcePoint] = []
    for record in records:
        if record.get("event") != "resource_snapshot":
            continue
        hbm_used = _optional_int(record.get("hbm_used_bytes"))
        indexed_gpu = _optional_int(record.get("page_index_gpu_bytes"))
        untracked_delta = _optional_int(
            record.get("untracked_allocator_delta_bytes")
        )
        if untracked_delta is None and hbm_used is not None and indexed_gpu is not None:
            untracked_delta = max(0, hbm_used - indexed_gpu)
        result.append(
            TimelineResourcePoint(
                ts_ms=float(record["ts_ms"]),
                hbm_used_bytes=hbm_used,
                hbm_capacity_bytes=_optional_int(record.get("hbm_capacity_bytes")),
                host_used_bytes=_optional_int(record.get("host_used_bytes")),
                host_capacity_bytes=_optional_int(record.get("host_capacity_bytes")),
                source="runtime_resource_snapshot",
                untracked_allocator_delta_bytes=untracked_delta,
                engine_locked_gpu_bytes=_optional_int(
                    record.get("engine_locked_gpu_bytes")
                ),
                engine_lock_ref_gpu_bytes=_optional_int(
                    record.get("engine_lock_ref_gpu_bytes")
                ),
                engine_lock_full_attribution_coverage=_optional_float(
                    record.get("engine_lock_full_attribution_coverage")
                ),
                locked_but_not_served_gpu_bytes_100ms=_optional_int(
                    record.get("locked_but_not_served_gpu_bytes_100ms")
                ),
                locked_but_not_served_gpu_bytes_500ms=_optional_int(
                    record.get("locked_but_not_served_gpu_bytes_500ms")
                ),
                closure_blocked_gpu_bytes=_optional_int(
                    record.get("closure_blocked_gpu_bytes")
                ),
                migratable_gpu_bytes=_optional_int(
                    record.get("migratable_gpu_bytes")
                ),
                dual_resident_gpu_bytes=_optional_int(
                    record.get("dual_resident_gpu_bytes")
                ),
            )
        )
    return result


def _extract_sglang_metrics(
    records: Iterable[dict[str, Any]],
    *,
    kv_bytes_per_token: int | None,
    hbm_capacity_bytes: int | None,
) -> list[TimelineResourcePoint]:
    if kv_bytes_per_token is None or kv_bytes_per_token <= 0:
        raise ValueError("kv_bytes_per_token is required for SGLang metrics")
    result: list[TimelineResourcePoint] = []
    for record in records:
        if "monotonic_ts_ms" not in record or "num_used_tokens" not in record:
            continue
        used_bytes = int(float(record["num_used_tokens"]) * kv_bytes_per_token)
        capacity = hbm_capacity_bytes
        pressure = float(record.get("resident_pressure", 0.0) or 0.0)
        if capacity is None and pressure > 0:
            capacity = int(used_bytes / pressure)
        result.append(
            TimelineResourcePoint(
                ts_ms=float(record["monotonic_ts_ms"]),
                hbm_used_bytes=used_bytes,
                hbm_capacity_bytes=capacity,
                host_used_bytes=None,
                host_capacity_bytes=None,
                source="sglang_metrics_derived_hbm",
            )
        )
    return result


def _summarize(
    transfers: list[TimelineTransfer],
    resources: list[TimelineResourcePoint],
    start_ts_ms: float,
    end_ts_ms: float,
) -> dict[str, Any]:
    by_direction: dict[str, dict[str, int]] = {}
    physical_durations = []
    for transfer in transfers:
        entry = by_direction.setdefault(
            transfer.direction,
            {
                "telemetry_count": 0,
                "physical_count": 0,
                "no_dma_count": 0,
                "actual_bytes": 0,
                "physical_closure_bytes": 0,
                "attempted_closure_bytes": 0,
            },
        )
        entry["telemetry_count"] += 1
        entry["actual_bytes"] += transfer.actual_bytes
        entry["attempted_closure_bytes"] += transfer.closure_bytes
        if _is_physical_transfer(transfer):
            entry["physical_count"] += 1
            entry["physical_closure_bytes"] += transfer.closure_bytes
            physical_durations.append(
                max(0.0, transfer.complete_ts_ms - transfer.submit_ts_ms)
            )
        else:
            entry["no_dma_count"] += 1
    runtime_hbm_values = [
        item.hbm_used_bytes
        for item in resources
        if item.source == "runtime_resource_snapshot"
        and item.hbm_used_bytes is not None
    ]
    all_hbm_values = [
        item.hbm_used_bytes for item in resources if item.hbm_used_bytes is not None
    ]
    diagnostic_fields = {
        "untracked_allocator_delta": "untracked_allocator_delta_bytes",
        "engine_locked_gpu": "engine_locked_gpu_bytes",
        "engine_lock_ref_gpu": "engine_lock_ref_gpu_bytes",
        "locked_but_not_served_gpu_100ms": (
            "locked_but_not_served_gpu_bytes_100ms"
        ),
        "locked_but_not_served_gpu_500ms": (
            "locked_but_not_served_gpu_bytes_500ms"
        ),
        "closure_blocked_gpu": "closure_blocked_gpu_bytes",
        "migratable_gpu": "migratable_gpu_bytes",
        "dual_resident_gpu": "dual_resident_gpu_bytes",
    }
    host_values = [
        item.host_used_bytes for item in resources if item.host_used_bytes is not None
    ]
    measurement_modes = sorted({transfer.measurement for transfer in transfers})
    resource_sources = sorted({resource.source for resource in resources})
    physical_transfers = [item for item in transfers if _is_physical_transfer(item)]
    no_dma_records = [item for item in transfers if not _is_physical_transfer(item)]
    lock_service_samples = [
        item for item in resources if item.engine_lock_ref_gpu_bytes is not None
    ]
    full_attribution_coverage = [
        item.engine_lock_full_attribution_coverage
        for item in lock_service_samples
        if item.engine_lock_full_attribution_coverage is not None
    ]

    def stale_ratios(field_name: str) -> list[float]:
        return [
            float(stale_bytes) / item.engine_lock_ref_gpu_bytes
            for item in lock_service_samples
            if item.engine_lock_ref_gpu_bytes
            and (stale_bytes := getattr(item, field_name)) is not None
        ]

    stale_ratio_100ms = stale_ratios("locked_but_not_served_gpu_bytes_100ms")
    stale_ratio_500ms = stale_ratios("locked_but_not_served_gpu_bytes_500ms")
    return {
        "duration_ms": max(0.0, end_ts_ms - start_ts_ms),
        # transfer_count remains as a compatibility alias, but now has the only
        # defensible meaning for a migration plot: operations with observed bytes.
        "transfer_count": len(physical_transfers),
        "physical_transfer_count": len(physical_transfers),
        "telemetry_record_count": len(transfers),
        "no_dma_record_count": len(no_dma_records),
        "no_dma_rejected_count": sum(
            item.status != "completed" for item in no_dma_records
        ),
        "resource_sample_count": len(resources),
        "lock_service_sample_count": len(lock_service_samples),
        "directions": by_direction,
        "callback_duration_p50_ms": percentile(physical_durations, 50),
        "callback_duration_p90_ms": percentile(physical_durations, 90),
        "partial_or_failed_count": sum(
            transfer.status != "completed" for transfer in physical_transfers
        ),
        "peak_hbm_used_bytes": max(
            runtime_hbm_values or all_hbm_values, default=None
        ),
        **{
            f"peak_{name}_bytes": max(
                (
                    value
                    for item in resources
                    if (value := getattr(item, field_name)) is not None
                ),
                default=None,
            )
            for name, field_name in diagnostic_fields.items()
        },
        "peak_host_used_bytes": max(host_values, default=None),
        "engine_lock_full_attribution_coverage_p50": percentile(
            full_attribution_coverage, 50
        ),
        "engine_lock_full_attribution_coverage_p90": percentile(
            full_attribution_coverage, 90
        ),
        "locked_but_not_served_lower_bound_ratio_100ms_p50": percentile(
            stale_ratio_100ms, 50
        ),
        "locked_but_not_served_lower_bound_ratio_100ms_p90": percentile(
            stale_ratio_100ms, 90
        ),
        "locked_but_not_served_lower_bound_ratio_500ms_p50": percentile(
            stale_ratio_500ms, 50
        ),
        "locked_but_not_served_lower_bound_ratio_500ms_p90": percentile(
            stale_ratio_500ms, 90
        ),
        "host_telemetry_available": bool(host_values),
        "measurement_modes": measurement_modes,
        "resource_sources": resource_sources,
    }


def _render_html(timeline: TransferTimeline, *, title: str) -> str:
    svg = _render_svg(timeline)
    summary = timeline.summary
    rows = "\n".join(_transfer_row(item, timeline.start_ts_ms) for item in timeline.transfers)
    direction_summary = " ".join(
        f"{escape(direction.upper())}: {values['physical_count']} physical / "
        f"{values['telemetry_count']} records / "
        f"{_format_bytes(values['actual_bytes'])}"
        for direction, values in sorted(summary["directions"].items())
    ) or "No transfer operations"
    host_state = (
        _format_bytes(summary["peak_host_used_bytes"])
        if summary["host_telemetry_available"]
        else "unavailable in source trace"
    )
    untracked_state = _format_bytes(
        summary["peak_untracked_allocator_delta_bytes"]
    )
    measurement_modes = summary["measurement_modes"]
    legacy_measurement = "legacy_dispatch_ack_aggregate" in measurement_modes
    measurement_note = (
        "Legacy source: transfer bars span dispatch-to-ACK and are not exact DMA "
        "intervals. Directional bytes may include aggregate residency actions."
        if legacy_measurement
        else "Backend telemetry: D2H/H2D bars show only operations with observed "
        "physical bytes. Zero-byte rejects are ticks on the No-DMA lane and are "
        "not counted as migrations."
    )
    resource_sources = summary["resource_sources"]
    resource_notes = []
    if "runtime_resource_snapshot" in resource_sources:
        resource_notes.append(
            "Allocator HBM/Host occupancy: runtime_resource_snapshot"
        )
        if any(
            item.untracked_allocator_delta_bytes is not None
            for item in timeline.resources
            if item.source == "runtime_resource_snapshot"
        ):
            resource_notes.append(
                "untracked allocator delta: max(allocator HBM - indexed GPU KV, 0); "
                "this is not classified as protected KV"
            )
        if any(
            item.locked_but_not_served_gpu_bytes_100ms is not None
            for item in timeline.resources
        ):
            resource_notes.append(
                "locked-but-not-served uses completed GPU batches as service "
                "evidence and exact running-request Radix lock paths; values are "
                "a conservative physical-byte lower bound"
            )
    if "sglang_metrics_derived_hbm" in resource_sources:
        resource_notes.append(
            "sampled HBM occupancy: SGLang num_used_tokens; sampled metrics can "
            "miss short-lived changes"
        )
    resource_note = "; ".join(resource_notes) or "No occupancy source"
    legend_items = []
    if "runtime_resource_snapshot" in resource_sources:
        legend_items.append(
            '<span class="key" style="--key:var(--hbm)">Allocator HBM occupancy</span>'
        )
    diagnostic_legends = (
        (
            "untracked_allocator_delta_bytes",
            "untracked",
            "Untracked allocator delta",
        ),
        ("engine_locked_gpu_bytes", "locked", "Engine-locked KV"),
        (
            "locked_but_not_served_gpu_bytes_100ms",
            "stale100",
            "Locked, no completed service >100 ms",
        ),
        (
            "locked_but_not_served_gpu_bytes_500ms",
            "stale500",
            "Locked, no completed service >500 ms",
        ),
        ("closure_blocked_gpu_bytes", "closure", "Closure-blocked KV"),
        ("migratable_gpu_bytes", "migratable", "Migratable KV"),
        ("dual_resident_gpu_bytes", "dual", "Dual-resident KV"),
    )
    for field_name, color_name, label in diagnostic_legends:
        if not any(
            getattr(item, field_name) is not None for item in timeline.resources
        ):
            continue
        legend_items.append(
            f'<span class="key key-dashed" style="--key:var(--{color_name})">{label}</span>'
        )
    if summary["host_telemetry_available"]:
        legend_items.append(
            '<span class="key" style="--key:var(--host)">Host KV occupancy</span>'
        )
    legend_items.extend(
        [
            '<span class="key" style="--key:var(--d2h)">D2H physical DMA</span>',
            '<span class="key" style="--key:var(--h2d)">H2D physical DMA</span>',
            '<span class="key" style="--key:var(--bad)">No-DMA reject</span>',
        ]
    )
    legend = "".join(legend_items)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#18212a; --muted:#66717d; --line:#d8dee5; --panel:#f6f8fa; --hbm:#147d73; --untracked:#7656a3; --locked:#a23d45; --stale100:#d15f4b; --stale500:#6f1d2a; --closure:#c97718; --migratable:#2d6f3e; --dual:#3968a8; --host:#59636e; --d2h:#16877d; --h2d:#c97718; --bad:#a23d45; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:#fff; font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }}
header {{ padding:24px 28px 18px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 6px; font-size:24px; font-weight:650; letter-spacing:0; }}
.meta {{ color:var(--muted); overflow-wrap:anywhere; }}
.notice {{ margin:18px 0 0; padding:10px 12px; border-left:3px solid #a76518; background:#fff8ec; color:#5d451f; }}
main {{ padding:22px 28px 36px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
.stat {{ padding:14px 12px; min-height:74px; border-right:1px solid var(--line); }}
.stat:last-child {{ border-right:0; }}
.label {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
.value {{ margin-top:5px; font-size:18px; font-variant-numeric:tabular-nums; }}
.chart-wrap {{ margin-top:22px; overflow-x:auto; border-bottom:1px solid var(--line); }}
svg {{ display:block; width:100%; min-width:980px; height:auto; background:#fff; }}
.legend {{ display:flex; flex-wrap:wrap; gap:18px; padding:10px 0 18px; color:var(--muted); }}
.key::before {{ content:""; display:inline-block; width:16px; height:3px; margin:0 7px 3px 0; background:var(--key); }}
.key-dashed::before {{ height:0; background:transparent; border-top:2px dashed var(--key); }}
h2 {{ margin:26px 0 10px; font-size:17px; letter-spacing:0; }}
.table-wrap {{ overflow:auto; max-height:520px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; }}
th {{ position:sticky; top:0; z-index:1; padding:8px 10px; text-align:left; background:var(--panel); border-bottom:1px solid var(--line); white-space:nowrap; }}
td {{ padding:7px 10px; border-bottom:1px solid #edf0f3; white-space:nowrap; }}
td.context {{ max-width:320px; overflow:hidden; text-overflow:ellipsis; }}
.status-bad {{ color:var(--bad); font-weight:600; }}
@media (max-width:720px) {{ header,main {{ padding-left:16px; padding-right:16px; }} .stat {{ border-bottom:1px solid var(--line); }} }}
</style>
</head>
<body>
<header><h1>{escape(title)}</h1><div class="meta">Run {escape(timeline.run_id)} | {escape(timeline.source_path)}</div></header>
<main>
<section class="stats">
<div class="stat"><div class="label">Elapsed</div><div class="value">{_format_duration(summary['duration_ms'])}</div></div>
<div class="stat"><div class="label">Physical DMA</div><div class="value">{summary['physical_transfer_count']}</div></div>
<div class="stat"><div class="label">Telemetry records</div><div class="value">{summary['telemetry_record_count']}</div></div>
<div class="stat"><div class="label">No-DMA rejects</div><div class="value">{summary['no_dma_rejected_count']}</div></div>
<div class="stat"><div class="label">Peak allocator HBM</div><div class="value">{_format_bytes(summary['peak_hbm_used_bytes'])}</div></div>
<div class="stat"><div class="label">Peak untracked delta</div><div class="value">{untracked_state}</div></div>
<div class="stat"><div class="label">Peak Host KV</div><div class="value">{host_state}</div></div>
<div class="stat"><div class="label">P90 callback</div><div class="value">{summary['callback_duration_p90_ms']:.2f} ms</div></div>
</section>
<div class="notice">{escape(measurement_note)}<br>{escape(resource_note)}</div>
<div class="chart-wrap">{svg}</div>
<div class="legend">{legend}</div>
<div class="meta">{direction_summary}</div>
<h2>All backend observations</h2>
<div class="table-wrap"><table><thead><tr><th>Start</th><th>Duration</th><th>Direction</th><th>Actual</th><th>Closure</th><th>Status</th><th>Measurement</th><th>Kind</th><th>Context</th><th>Command</th></tr></thead><tbody>{rows}</tbody></table></div>
</main>
</body>
</html>
"""


def _render_svg(timeline: TransferTimeline) -> str:
    width, height = 1500, 610
    left, right = 82, 24
    plot_top, plot_bottom = 42, 312
    lane_y = {"d2h": 372, "h2d": 416, "mixed": 460, "reclaim": 504}
    no_dma_y = 548
    plot_width = width - left - right
    duration = max(1.0, timeline.end_ts_ms - timeline.start_ts_ms)

    def x(ts_ms: float) -> float:
        return left + (ts_ms - timeline.start_ts_ms) / duration * plot_width

    hbm_capacity = max(
        (point.hbm_capacity_bytes or 0 for point in timeline.resources), default=0
    )
    host_capacity = max(
        (point.host_capacity_bytes or 0 for point in timeline.resources), default=0
    )
    if not hbm_capacity:
        hbm_capacity = max(
            (point.hbm_used_bytes or 0 for point in timeline.resources), default=1
        )
    if not host_capacity:
        host_capacity = max(
            (point.host_used_bytes or 0 for point in timeline.resources), default=1
        )

    def y(value: int, capacity: int) -> float:
        ratio = min(1.0, max(0.0, value / max(1, capacity)))
        return plot_bottom - ratio * (plot_bottom - plot_top)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="HBM and Host KV transfer timeline">',
        '<rect x="0" y="0" width="1500" height="610" fill="#ffffff"/>',
    ]
    for index in range(6):
        ratio = index / 5
        y_pos = plot_bottom - ratio * (plot_bottom - plot_top)
        parts.append(
            f'<line x1="{left}" y1="{y_pos:.2f}" x2="{width-right}" y2="{y_pos:.2f}" stroke="#e5e9ed" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{y_pos+4:.2f}" text-anchor="end" fill="#66717d" font-size="11">{ratio*100:.0f}%</text>'
        )
    for index in range(9):
        ratio = index / 8
        x_pos = left + ratio * plot_width
        ts_ms = ratio * duration
        parts.append(
            f'<line x1="{x_pos:.2f}" y1="{plot_top}" x2="{x_pos:.2f}" y2="566" stroke="#eef1f4" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_pos:.2f}" y="594" text-anchor="middle" fill="#66717d" font-size="11">{escape(_format_duration(ts_ms))}</text>'
        )
    parts.append(
        f'<text x="{left}" y="24" fill="#18212a" font-size="13" font-weight="600">KV tier occupancy (% of each tier capacity)</text>'
    )
    runtime_hbm_points = [
        (point.ts_ms, point.hbm_used_bytes)
        for point in timeline.resources
        if point.source == "runtime_resource_snapshot"
        and point.hbm_used_bytes is not None
    ]
    sampled_hbm_points = [
        (point.ts_ms, point.hbm_used_bytes)
        for point in timeline.resources
        if point.source == "sglang_metrics_derived_hbm"
        and point.hbm_used_bytes is not None
    ]
    hbm_points = runtime_hbm_points or sampled_hbm_points
    host_points = [
        (point.ts_ms, point.host_used_bytes)
        for point in timeline.resources
        if point.host_used_bytes is not None
    ]
    if hbm_points:
        parts.append(
            f'<path class="allocator-hbm-series" d="{_step_path(hbm_points, x, lambda value: y(value, hbm_capacity))}" fill="none" stroke="#147d73" stroke-width="2.5"/>'
        )
    diagnostic_series = (
        (
            "untracked-allocator-series",
            "untracked_allocator_delta_bytes",
            "#7656a3",
            "7 5",
        ),
        ("engine-locked-series", "engine_locked_gpu_bytes", "#a23d45", "3 4"),
        (
            "locked-not-served-100ms-series",
            "locked_but_not_served_gpu_bytes_100ms",
            "#d15f4b",
            "5 3",
        ),
        (
            "locked-not-served-500ms-series",
            "locked_but_not_served_gpu_bytes_500ms",
            "#6f1d2a",
            "10 3",
        ),
        (
            "closure-blocked-series",
            "closure_blocked_gpu_bytes",
            "#c97718",
            "8 4",
        ),
        ("migratable-series", "migratable_gpu_bytes", "#2d6f3e", "12 4"),
        ("dual-resident-series", "dual_resident_gpu_bytes", "#3968a8", "2 4"),
    )
    for class_name, field_name, color, dash in diagnostic_series:
        points = [
            (point.ts_ms, getattr(point, field_name))
            for point in timeline.resources
            if getattr(point, field_name) is not None
        ]
        if points:
            parts.append(
                f'<path class="{class_name}" d="{_step_path(points, x, lambda value: y(value, hbm_capacity))}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'
            )
    if host_points:
        parts.append(
            f'<path d="{_step_path(host_points, x, lambda value: y(value, host_capacity))}" fill="none" stroke="#3968a8" stroke-width="2.5"/>'
        )
    else:
        parts.append(
            f'<text x="{left+12}" y="{plot_top+24}" fill="#66717d" font-size="12">Host occupancy unavailable in this source trace</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="348" x2="{width-right}" y2="348" stroke="#cfd6dd" stroke-width="1"/>'
    )
    labels = {"d2h": "D2H", "h2d": "H2D", "mixed": "Mixed", "reclaim": "Reclaim"}
    colors = {"d2h": "#16877d", "h2d": "#c97718", "mixed": "#7a5aa6", "reclaim": "#7b858f"}
    physical_transfers = [
        item for item in timeline.transfers if _is_physical_transfer(item)
    ]
    no_dma_records = [
        item for item in timeline.transfers if not _is_physical_transfer(item)
    ]
    present = {item.direction for item in physical_transfers}
    for direction, y_pos in lane_y.items():
        if direction not in present:
            continue
        parts.append(
            f'<text x="{left-12}" y="{y_pos+5}" text-anchor="end" fill="#66717d" font-size="11">{labels[direction]}</text>'
        )
    for transfer in physical_transfers:
        y_pos = lane_y.get(transfer.direction, lane_y["mixed"])
        x_start = x(
            transfer.start_ts_ms
            if transfer.start_ts_ms is not None
            else transfer.submit_ts_ms
        )
        x_end = x(transfer.complete_ts_ms)
        bar_width = max(1.5, x_end - x_start)
        color = colors.get(transfer.direction, colors["mixed"])
        opacity = "0.86" if transfer.status == "completed" else "0.42"
        tooltip = escape(
            f"{transfer.command_id} | {transfer.direction.upper()} | "
            f"{_format_bytes(transfer.actual_bytes)} actual / "
            f"{_format_bytes(transfer.closure_bytes)} closure | "
            f"{transfer.complete_ts_ms-transfer.submit_ts_ms:.3f} ms | "
            f"{transfer.status} | {transfer.context_id or 'unscoped'}"
        )
        parts.append(
            f'<rect class="physical-transfer" x="{x_start:.2f}" y="{y_pos-8}" width="{bar_width:.2f}" height="16" fill="{color}" opacity="{opacity}" rx="2"><title>{tooltip}</title></rect>'
        )
    if no_dma_records:
        parts.append(
            f'<text x="{left-12}" y="{no_dma_y+5}" text-anchor="end" fill="#66717d" font-size="11">No DMA</text>'
        )
        for transfer in no_dma_records:
            x_pos = x(transfer.submit_ts_ms)
            tooltip = escape(
                f"{transfer.command_id} | no physical bytes | "
                f"{transfer.direction.upper()} | {transfer.status} | "
                f"{transfer.reason or 'no reason'}"
            )
            parts.append(
                f'<line class="no-dma-attempt" x1="{x_pos:.2f}" y1="{no_dma_y-7}" x2="{x_pos:.2f}" y2="{no_dma_y+7}" stroke="#a23d45" stroke-width="1" opacity="0.28"><title>{tooltip}</title></line>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _step_path(
    points: list[tuple[float, int | None]],
    x_scale: Any,
    y_scale: Any,
) -> str:
    clean = [(ts, int(value)) for ts, value in points if value is not None]
    if not clean:
        return ""
    pieces = [f"M {x_scale(clean[0][0]):.2f} {y_scale(clean[0][1]):.2f}"]
    for ts, value in clean[1:]:
        pieces.append(f"H {x_scale(ts):.2f} V {y_scale(value):.2f}")
    return " ".join(pieces)


def _transfer_row(transfer: TimelineTransfer, base_ts_ms: float) -> str:
    status_class = "" if transfer.status == "completed" else ' class="status-bad"'
    context = transfer.context_id or "-"
    return (
        "<tr>"
        f"<td>{_format_duration(transfer.submit_ts_ms-base_ts_ms)}</td>"
        f"<td>{transfer.complete_ts_ms-transfer.submit_ts_ms:.3f} ms</td>"
        f"<td>{escape(transfer.direction.upper())}</td>"
        f"<td>{_format_bytes(transfer.actual_bytes)}</td>"
        f"<td>{_format_bytes(transfer.closure_bytes)}</td>"
        f"<td{status_class}>{escape(transfer.status)}</td>"
        f"<td>{escape(transfer.measurement)}</td>"
        f"<td>{escape(transfer.command_kind or '-')}</td>"
        f'<td class="context" title="{escape(context)}">{escape(context)}</td>'
        f"<td>{escape(transfer.command_id)}</td>"
        "</tr>"
    )


def _is_physical_transfer(transfer: TimelineTransfer) -> bool:
    """Return whether the backend observed any bytes moved for this operation."""

    return (
        transfer.actual_bytes > 0
        and transfer.direction in {"d2h", "h2d", "mixed"}
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            break
        amount /= 1024
    return f"{amount:.2f} {unit}"


def _format_duration(value_ms: float) -> str:
    if value_ms < 1000:
        return f"{value_ms:.0f} ms"
    if value_ms < 60_000:
        return f"{value_ms/1000:.1f} s"
    return f"{value_ms/60_000:.1f} min"


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
