#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARROW = (
    REPOSITORY_ROOT
    / "workloads/raw/swebench_verified-91aa3ed/test/data-00000-of-00001.arrow"
)
DEFAULT_INSTANCE_IDS = (
    "sympy__sympy-13877",
    "sympy__sympy-17630",
    "sympy__sympy-13878",
    "sympy__sympy-17318",
)


def run_git(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze real SWE-bench inputs and create isolated Codex worktrees."
    )
    parser.add_argument("--arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    args = parser.parse_args()

    from datasets import Dataset

    arrow = args.arrow.resolve()
    source_repo = args.source_repo.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    if not (source_repo / ".git").exists():
        raise FileNotFoundError(f"source repository is missing .git: {source_repo}")
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    selected_ids = tuple(args.instance_ids or DEFAULT_INSTANCE_IDS)
    dataset = Dataset.from_file(str(arrow))
    by_id = {row["instance_id"]: row for row in dataset}
    records = []
    for instance_id in selected_ids:
        row = by_id.get(instance_id)
        if row is None:
            raise KeyError(f"SWE-bench instance not found: {instance_id}")
        if row["repo"] != "sympy/sympy":
            raise ValueError(f"expected a SymPy instance: {instance_id}")
        commit = str(row["base_commit"])
        run_git(source_repo, "cat-file", "-e", f"{commit}^{{commit}}")
        worktree = output_root / instance_id
        if worktree.exists():
            actual = run_git(worktree, "rev-parse", "HEAD")
            if actual != commit:
                raise RuntimeError(
                    f"existing worktree {worktree} is at {actual}, expected {commit}"
                )
        else:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_repo),
                    "worktree",
                    "add",
                    "--detach",
                    str(worktree),
                    commit,
                ],
                check=True,
            )
        records.append(
            {
                "instance_id": instance_id,
                "repo": row["repo"],
                "base_commit": commit,
                "problem_statement": row["problem_statement"],
                "difficulty": row.get("difficulty"),
                "worktree": str(worktree),
            }
        )

    payload = {
        "schema_version": 1,
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "dataset_revision": "91aa3ed51b709be6457e12d00300a6a596d4c6a3",
        "split": "test",
        "arrow_path": str(arrow),
        "arrow_sha256": hashlib.sha256(arrow.read_bytes()).hexdigest(),
        "gold_fields_exposed": False,
        "source_repo": str(source_repo),
        "source_repo_head": run_git(source_repo, "rev-parse", "HEAD"),
        "workloads": records,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
