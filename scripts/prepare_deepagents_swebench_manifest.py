#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


DEFAULT_INSTANCE_IDS = (
    "sympy__sympy-12489",
    "sympy__sympy-13852",
    "sympy__sympy-13878",
    "sympy__sympy-14248",
    "sympy__sympy-16597",
    "sympy__sympy-17630",
    "sympy__sympy-18199",
    "sympy__sympy-13877",
    "sympy__sympy-17318",
    "sympy__sympy-16792",
    "sympy__sympy-19040",
    "sympy__sympy-15599",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze real local SWE-bench SymPy workloads for Deep Agents."
    )
    parser.add_argument("--arrow", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--instance", action="append", default=[])
    args = parser.parse_args()

    arrow_path = args.arrow.expanduser().resolve()
    source_repo = args.source_repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output manifest already exists: {output}")
    table = ipc.open_stream(pa.memory_map(str(arrow_path), "r")).read_all()
    indexed = {str(row["instance_id"]): row for row in table.to_pylist()}
    instance_ids = tuple(args.instance) or DEFAULT_INSTANCE_IDS
    unknown = set(instance_ids) - set(indexed)
    if unknown:
        raise ValueError(f"instances absent from Arrow dataset: {sorted(unknown)}")

    workloads = []
    for instance_id in instance_ids:
        row = indexed[instance_id]
        if row["repo"] != "sympy/sympy":
            raise ValueError(f"non-SymPy instance selected: {instance_id}")
        base_commit = str(row["base_commit"])
        git_output(source_repo, "cat-file", "-e", f"{base_commit}^{{commit}}")
        workloads.append(
            {
                "instance_id": instance_id,
                "repo": str(row["repo"]),
                "base_commit": base_commit,
                "difficulty": str(row.get("difficulty") or "unknown"),
                "problem_statement": str(row["problem_statement"]),
            }
        )

    payload = {
        "schema_version": 1,
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "dataset_revision": args.dataset_revision,
        "split": "test",
        "arrow_path": str(arrow_path),
        "arrow_sha256": sha256(arrow_path),
        "source_repo": str(source_repo),
        "source_repo_head": git_output(source_repo, "rev-parse", "HEAD"),
        "gold_fields_exposed": False,
        "selection_policy": (
            "all seven SymPy instances labeled 1-4 hours or >4 hours, plus the "
            "five longest problem statements labeled 15 min - 1 hour whose base "
            "commits exist in the pinned local source repository"
        ),
        "workloads": workloads,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "workload_count": len(workloads),
                "manifest_sha256": sha256(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
