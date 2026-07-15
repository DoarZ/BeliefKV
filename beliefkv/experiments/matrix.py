from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from beliefkv.core.config import BeliefKVConfig
from beliefkv.metrics.artifacts import ExperimentArtifactWriter
from beliefkv.metrics.summary import bootstrap_mean_ci, mean, percentile
from beliefkv.simulator.page_simulator import PageLevelSimulator
from beliefkv.simulator.schema import SimulationScenario


def _path_name(value: str, field_name: str) -> str:
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(
            f"{field_name} must contain only letters, digits, '.', '_' or '-'"
        )
    return value


@dataclass(frozen=True)
class MatrixVariant:
    name: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _path_name(self.name, "variant name")
        # Validate keys and values before an expensive matrix starts.
        BeliefKVConfig.from_mapping(self.config)


@dataclass(frozen=True)
class MatrixScenario:
    path: Path
    label: str | None = None


@dataclass(frozen=True)
class ExperimentMatrix:
    name: str
    scenarios: tuple[MatrixScenario, ...]
    variants: tuple[MatrixVariant, ...]
    base_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _path_name(self.name, "matrix name")
        if not self.scenarios:
            raise ValueError("matrix requires at least one scenario")
        if not self.variants:
            raise ValueError("matrix requires at least one variant")
        names = [item.name for item in self.variants]
        if len(names) != len(set(names)):
            raise ValueError("matrix variant names must be unique")
        BeliefKVConfig.from_mapping(self.base_config)

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], *, base_directory: Path
    ) -> "ExperimentMatrix":
        scenarios: list[MatrixScenario] = []
        for item in raw.get("scenarios", []):
            if isinstance(item, str):
                relative_path, label = item, None
            elif isinstance(item, Mapping):
                relative_path = str(item["path"])
                label = str(item["label"]) if item.get("label") else None
            else:
                raise ValueError("matrix scenarios must be paths or objects")
            path = Path(relative_path)
            if not path.is_absolute():
                path = base_directory / path
            scenarios.append(MatrixScenario(path.resolve(), label))
        variants = tuple(
            MatrixVariant(str(item["name"]), dict(item.get("config", {})))
            for item in raw.get("variants", [])
        )
        return cls(
            name=str(raw.get("name", "matrix")),
            scenarios=tuple(scenarios),
            variants=variants,
            base_config=dict(raw.get("base_config", {})),
            metadata=dict(raw.get("metadata", {})),
        )

    @classmethod
    def from_path(cls, path: Path) -> "ExperimentMatrix":
        resolved = path.resolve()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("matrix file must contain a JSON object")
        return cls.from_dict(raw, base_directory=resolved.parent)


@dataclass(frozen=True)
class MatrixRunResult:
    matrix_dir: Path
    run_rows: tuple[dict[str, Any], ...]
    aggregate: dict[str, Any]


class ExperimentMatrixRunner:
    """Execute policy ablations with immutable per-run artifacts."""

    def __init__(
        self,
        repository_root: Path,
        output_root: Path,
        *,
        bootstrap_resamples: int = 2000,
    ) -> None:
        if bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        self.repository_root = repository_root.resolve()
        self.output_root = output_root.resolve()
        self.bootstrap_resamples = bootstrap_resamples

    def run(
        self, matrix: ExperimentMatrix, *, run_id: str | None = None
    ) -> MatrixRunResult:
        matrix_run_id = _path_name(
            run_id or self._default_run_id(matrix.name), "matrix run id"
        )
        prepared = self._preflight(matrix)
        self.output_root.mkdir(parents=True, exist_ok=True)
        matrix_dir = self.output_root / matrix_run_id
        staging_dir = self.output_root / f".{matrix_run_id}.incomplete"
        if matrix_dir.exists():
            raise FileExistsError(f"matrix directory already exists: {matrix_dir}")
        if staging_dir.exists():
            raise FileExistsError(
                f"incomplete matrix directory already exists: {staging_dir}"
            )
        staging_dir.mkdir(parents=True)
        writer = ExperimentArtifactWriter(self.repository_root, staging_dir / "runs")
        rows: list[dict[str, Any]] = []
        samples_by_variant: dict[str, dict[str, list[float]]] = {}
        bytes_by_variant: dict[str, dict[str, int]] = {}

        for scenario_spec, scenario, scenario_label, variant_configs in prepared:
            for variant, config in variant_configs:
                result = PageLevelSimulator(config).run(scenario)
                artifact_id = f"{scenario_label}--{variant.name}--s{scenario.seed}"
                writer.write(
                    run_id=artifact_id,
                    workload=scenario.name,
                    seed=scenario.seed,
                    config=config,
                    result=result,
                    metadata={
                        "matrix_name": matrix.name,
                        "matrix_run_id": matrix_run_id,
                        "variant": variant.name,
                        "scenario_path": str(scenario_spec.path),
                        **dict(matrix.metadata),
                        **dict(scenario.metadata),
                    },
                )
                row = self._run_row(
                    matrix_run_id,
                    scenario_label,
                    variant.name,
                    scenario.seed,
                    matrix_dir / "runs" / artifact_id,
                    config,
                    result.summary,
                )
                rows.append(row)
                samples = samples_by_variant.setdefault(
                    variant.name, {"completion": [], "admission": []}
                )
                samples["completion"].extend(
                    float(value)
                    for value in result.summary["workflow_completion_ms"].values()
                )
                samples["admission"].extend(
                    float(value)
                    for value in result.summary["admission_stall_ms"].values()
                )
                byte_totals = bytes_by_variant.setdefault(
                    variant.name, {"prepared": 0, "useful": 0, "wasted": 0}
                )
                byte_totals["prepared"] += int(
                    result.summary["shadow_prepared_bytes"]
                )
                byte_totals["useful"] += int(result.summary["useful_shadow_bytes"])
                byte_totals["wasted"] += int(result.summary["wasted_shadow_bytes"])

        aggregate = self._aggregate(samples_by_variant, bytes_by_variant)
        spec_payload = self._matrix_payload(matrix)
        self._atomic_json(
            staging_dir / "matrix_manifest.json",
            {
                "matrix_run_id": matrix_run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "matrix": spec_payload,
                "matrix_sha256": hashlib.sha256(
                    json.dumps(spec_payload, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "bootstrap_resamples": self.bootstrap_resamples,
                "run_count": len(rows),
            },
        )
        self._atomic_json(staging_dir / "summary.json", aggregate)
        self._write_csv(staging_dir / "runs.csv", rows)
        staging_dir.replace(matrix_dir)
        return MatrixRunResult(matrix_dir, tuple(rows), aggregate)

    def _preflight(
        self, matrix: ExperimentMatrix
    ) -> list[
        tuple[
            MatrixScenario,
            SimulationScenario,
            str,
            list[tuple[MatrixVariant, BeliefKVConfig]],
        ]
    ]:
        prepared = []
        labels: set[str] = set()
        for scenario_spec in matrix.scenarios:
            raw = self._load_scenario(scenario_spec.path)
            scenario = SimulationScenario.from_dict(raw)
            scenario_label = _path_name(
                scenario_spec.label or scenario.name, "scenario label"
            )
            if scenario_label in labels:
                raise ValueError(f"duplicate matrix scenario label: {scenario_label}")
            labels.add(scenario_label)
            scenario_config = dict(raw.get("config", {}))
            variant_configs = [
                (
                    variant,
                    BeliefKVConfig.from_mapping(
                        {
                            **scenario_config,
                            **dict(matrix.base_config),
                            **dict(variant.config),
                        }
                    ),
                )
                for variant in matrix.variants
            ]
            prepared.append(
                (scenario_spec, scenario, scenario_label, variant_configs)
            )
        return prepared

    @staticmethod
    def _load_scenario(path: Path) -> dict[str, Any]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"scenario must contain a JSON object: {path}")
        return raw

    @staticmethod
    def _run_row(
        matrix_run_id: str,
        scenario: str,
        variant: str,
        seed: int,
        run_dir: Path,
        config: BeliefKVConfig,
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        completion = [float(item) for item in summary["workflow_completion_ms"].values()]
        stalls = [float(item) for item in summary["admission_stall_ms"].values()]
        prepared = int(summary["shadow_prepared_bytes"])
        useful = int(summary["useful_shadow_bytes"])
        return {
            "matrix_run_id": matrix_run_id,
            "scenario": scenario,
            "variant": variant,
            "seed": seed,
            "run_dir": str(run_dir),
            "workflow_count": len(completion),
            "workflow_completion_mean_ms": mean(completion),
            "workflow_completion_p95_ms": percentile(completion, 95),
            "admission_stall_mean_ms": mean(stalls),
            "admission_stall_p95_ms": percentile(stalls, 95),
            "shadow_prepared_bytes": prepared,
            "useful_shadow_bytes": useful,
            "wasted_shadow_bytes": int(summary["wasted_shadow_bytes"]),
            "useful_shadow_fraction": useful / prepared if prepared else 0.0,
            "urgent_d2h_bytes": int(summary["urgent_d2h_bytes"]),
            "urgent_h2d_bytes": int(summary["urgent_h2d_bytes"]),
            "peak_gpu_bytes": int(summary["peak_gpu_bytes"]),
            "peak_cpu_bytes": int(summary["peak_cpu_bytes"]),
            "planner_ticks": int(summary["planner_ticks"]),
            "predictor_enabled": config.predictor_enabled,
            "shadow_enabled": config.shadow_enabled,
        }

    def _aggregate(
        self,
        samples_by_variant: Mapping[str, Mapping[str, list[float]]],
        bytes_by_variant: Mapping[str, Mapping[str, int]],
    ) -> dict[str, Any]:
        variants: dict[str, Any] = {}
        for index, variant in enumerate(sorted(samples_by_variant)):
            samples = samples_by_variant[variant]
            completion = samples["completion"]
            admission = samples["admission"]
            ci = bootstrap_mean_ci(
                completion,
                resamples=self.bootstrap_resamples,
                seed=index,
            )
            byte_totals = bytes_by_variant[variant]
            prepared = byte_totals["prepared"]
            variants[variant] = {
                "workflow_samples": len(completion),
                "workflow_completion_mean_ms": ci.estimate,
                "workflow_completion_mean_ci95_ms": [ci.lower, ci.upper],
                "workflow_completion_p50_ms": percentile(completion, 50),
                "workflow_completion_p95_ms": percentile(completion, 95),
                "workflow_completion_p99_ms": percentile(completion, 99),
                "admission_samples": len(admission),
                "admission_stall_mean_ms": mean(admission),
                "admission_stall_p95_ms": percentile(admission, 95),
                "shadow_prepared_bytes": prepared,
                "useful_shadow_bytes": byte_totals["useful"],
                "wasted_shadow_bytes": byte_totals["wasted"],
                "useful_shadow_fraction": (
                    byte_totals["useful"] / prepared if prepared else 0.0
                ),
            }
        return {"variants": variants}

    @staticmethod
    def _matrix_payload(matrix: ExperimentMatrix) -> dict[str, Any]:
        return {
            "name": matrix.name,
            "scenarios": [
                {"path": str(item.path), "label": item.label}
                for item in matrix.scenarios
            ],
            "variants": [
                {"name": item.name, "config": dict(item.config)}
                for item in matrix.variants
            ],
            "base_config": dict(matrix.base_config),
            "metadata": dict(matrix.metadata),
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        fieldnames = list(rows[0]) if rows else []
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(rows)
        temporary.replace(path)

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _default_run_id(matrix_name: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{matrix_name}-{timestamp}"
