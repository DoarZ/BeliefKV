from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from beliefkv.core.config import BeliefKVConfig
from beliefkv.core.events import RuntimeEvent
from beliefkv.experiments.matrix import ExperimentMatrix, ExperimentMatrixRunner
from beliefkv.metrics.artifacts import ExperimentArtifactWriter
from beliefkv.predictor.training import extract_training_corpus, train_predictor
from beliefkv.runtime.sglang_adapter import (
    BASE_SGLANG_GIT_COMMIT,
    BASE_SGLANG_VERSION,
    SGLangSourceContract,
    required_hooks,
)
from beliefkv.simulator.page_simulator import PageLevelSimulator
from beliefkv.simulator.schema import SimulationScenario
from beliefkv.traces.normalizer import ClawTraceNormalizer, load_jsonl_records


def cmd_hooks(_: argparse.Namespace) -> None:
    payload = {
        "base_sglang_version": BASE_SGLANG_VERSION,
        "base_sglang_git_commit": BASE_SGLANG_GIT_COMMIT,
        "required_hooks": [asdict(hook) for hook in required_hooks()],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_check_sglang(args: argparse.Namespace) -> None:
    report = SGLangSourceContract().check(Path(args.source).resolve())
    payload = {
        "expected_version": BASE_SGLANG_VERSION,
        "expected_commit": BASE_SGLANG_GIT_COMMIT,
        "compatible": report.compatible,
        "failures": [asdict(item) for item in report.failures],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report.compatible:
        raise SystemExit(1)


def cmd_simulate(args: argparse.Namespace) -> None:
    scenario_path = Path(args.scenario).resolve()
    raw = json.loads(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("scenario file must contain a JSON object")
    config_raw = raw.get("config", {})
    if args.config:
        override = json.loads(Path(args.config).read_text(encoding="utf-8"))
        config_raw = {**config_raw, **override}
    config = BeliefKVConfig.from_mapping(config_raw)
    scenario = SimulationScenario.from_dict(raw)
    result = PageLevelSimulator(config).run(scenario)
    if args.no_write:
        print(json.dumps(result.summary, indent=2, sort_keys=True, allow_nan=False))
        return
    run_id = args.run_id or _default_run_id(scenario.name)
    repository_root = Path(__file__).resolve().parents[1]
    writer = ExperimentArtifactWriter(
        repository_root, Path(args.output_root).resolve()
    )
    run_dir = writer.write(
        run_id=run_id,
        workload=scenario.name,
        seed=scenario.seed,
        config=config,
        result=result,
        metadata={"scenario_path": str(scenario_path), **dict(scenario.metadata)},
    )
    print(
        json.dumps(
            {"run_dir": str(run_dir), "summary": result.summary},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_normalize_clawtrace(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    records = load_jsonl_records(input_path.read_text(encoding="utf-8").splitlines())
    normalized = ClawTraceNormalizer().normalize(
        records, close_workflows=not args.no_close_workflows
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for event in normalized.events:
            payload = asdict(event)
            payload["kind"] = event.kind.value
            payload["confidence"] = event.confidence.value
            for key in ("relation_type", "context_mode", "execution_mode"):
                value = getattr(event, key)
                payload[key] = value.value if value is not None else None
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output_path)
    print(json.dumps(asdict(normalized.report), indent=2, sort_keys=True))


def cmd_matrix(args: argparse.Namespace) -> None:
    matrix_path = Path(args.matrix).resolve()
    matrix = ExperimentMatrix.from_path(matrix_path)
    repository_root = Path(__file__).resolve().parents[1]
    result = ExperimentMatrixRunner(
        repository_root,
        Path(args.output_root).resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
    ).run(matrix, run_id=args.run_id)
    print(
        json.dumps(
            {
                "matrix_dir": str(result.matrix_dir),
                "run_count": len(result.run_rows),
                "summary": result.aggregate,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


def cmd_train_predictor(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    content = input_path.read_bytes()
    records = load_jsonl_records(content.decode("utf-8").splitlines())
    events = [RuntimeEvent.from_dict(record) for record in records]
    corpus = extract_training_corpus(events)
    predictor = train_predictor(
        corpus,
        max_context_order=args.max_context_order,
        min_context_count=args.min_context_count,
        min_family_samples=args.min_family_samples,
        min_backend_samples=args.min_backend_samples,
    )
    output_path = Path(args.output).resolve()
    predictor.save(
        output_path,
        metadata={
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(input_path),
            "source_sha256": hashlib.sha256(content).hexdigest(),
            "training_summary": corpus.summary.to_dict(),
        },
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "training_summary": corpus.summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _default_run_id(workload: str) -> str:
    safe_workload = re.sub(r"[^A-Za-z0-9_.-]+", "-", workload).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_workload or 'run'}-{timestamp}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="beliefkv")
    sub = parser.add_subparsers(required=True)

    hooks = sub.add_parser("hooks", help="print runtime hook points")
    hooks.set_defaults(func=cmd_hooks)

    check = sub.add_parser(
        "check-sglang", help="validate an SGLang source tree against the pinned contract"
    )
    check.add_argument("source")
    check.set_defaults(func=cmd_check_sglang)

    simulate = sub.add_parser(
        "simulate", help="run a deterministic page-level BeliefKV scenario"
    )
    simulate.add_argument("scenario")
    simulate.add_argument("--config", help="JSON config overrides")
    simulate.add_argument("--run-id")
    simulate.add_argument(
        "--output-root", default="experiments/results", help="artifact root"
    )
    simulate.add_argument("--no-write", action="store_true")
    simulate.set_defaults(func=cmd_simulate)

    normalize = sub.add_parser(
        "normalize-clawtrace", help="normalize ClawTrace JSONL into RCCG events"
    )
    normalize.add_argument("input")
    normalize.add_argument("output")
    normalize.add_argument("--no-close-workflows", action="store_true")
    normalize.set_defaults(func=cmd_normalize_clawtrace)

    matrix = sub.add_parser(
        "matrix", help="run a reproducible simulation ablation matrix"
    )
    matrix.add_argument("matrix")
    matrix.add_argument("--run-id")
    matrix.add_argument(
        "--output-root", default="experiments/results", help="artifact root"
    )
    matrix.add_argument("--bootstrap-resamples", type=int, default=2000)
    matrix.set_defaults(func=cmd_matrix)

    train = sub.add_parser(
        "train-predictor", help="fit a portable predictor artifact from RCCG JSONL"
    )
    train.add_argument("input", help="normalized RuntimeEvent JSONL")
    train.add_argument("output", help="output predictor JSON artifact")
    train.add_argument("--max-context-order", type=int, default=4)
    train.add_argument("--min-context-count", type=int, default=3)
    train.add_argument("--min-family-samples", type=int, default=8)
    train.add_argument("--min-backend-samples", type=int, default=5)
    train.set_defaults(func=cmd_train_predictor)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
