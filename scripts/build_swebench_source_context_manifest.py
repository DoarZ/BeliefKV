#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", re.MULTILINE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SWE-bench characterization manifest with bounded pre-fix "
            "repository observations selected by an explicit path oracle."
        )
    )
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--context-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-context-chars", type=int, default=32_768)
    return parser.parse_args()


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _validate_repo_path(raw: object) -> str:
    path = PurePosixPath(str(raw))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe repository path: {raw}")
    return str(path)


def _first_changed_line(repo: Path, reference_commit: str, path: str) -> int | None:
    diff = _git(
        repo,
        ["diff", "--unified=0", f"{reference_commit}^1", reference_commit, "--", path],
    )
    match = _HUNK_RE.search(diff)
    return int(match.group(1)) if match is not None else None


def _bounded_excerpt(text: str, *, center_line: int | None, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    if center_line is None:
        split = max(1, budget // 2)
        return text[:split] + "\n# ... source omitted ...\n" + text[-split:]
    lines = text.splitlines(keepends=True)
    line_index = min(max(0, center_line - 1), max(0, len(lines) - 1))
    center = sum(len(item) for item in lines[:line_index])
    start = max(0, center - budget // 2)
    end = min(len(text), start + budget)
    start = max(0, end - budget)
    prefix = "# ... earlier source omitted ...\n" if start else ""
    suffix = "\n# ... later source omitted ..." if end < len(text) else ""
    available = max(0, budget - len(prefix) - len(suffix))
    return prefix + text[start : start + available] + suffix


def _build_context(
    repo: Path,
    *,
    base_commit: str,
    reference_commit: str,
    paths: Sequence[str],
    max_chars: int,
) -> str:
    if not paths:
        raise ValueError("source context path list must be non-empty")
    sections = []
    per_path = max(1, (max_chars - 256 * len(paths)) // len(paths))
    for path in paths:
        source = _git(repo, ["show", f"{base_commit}:{path}"])
        center_line = _first_changed_line(repo, reference_commit, path)
        excerpt = _bounded_excerpt(source, center_line=center_line, budget=per_path)
        sections.append(
            f"\n### {path} (base={base_commit}, localized_line={center_line})\n"
            f"```python\n{excerpt}\n```\n"
        )
    return "".join(sections)[:max_chars]


def main() -> int:
    args = _parse_args()
    if args.max_context_chars <= 0:
        raise ValueError("max context characters must be positive")
    workload_path = args.workload_manifest.expanduser().resolve()
    spec_path = args.context_spec.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    workload = _read_object(workload_path)
    spec = _read_object(spec_path)
    raw_workloads = workload.get("workloads")
    raw_specs = spec.get("workloads")
    if not isinstance(raw_workloads, list) or not isinstance(raw_specs, list):
        raise TypeError("workload and context spec must contain workload lists")
    repo = Path(str(workload["source_repo"])).expanduser().resolve()
    if not (repo / ".git").is_dir():
        raise FileNotFoundError(f"source repository is absent: {repo}")
    specs = {
        str(item["instance_id"]): item
        for item in raw_specs
        if isinstance(item, Mapping)
    }
    output_workloads = []
    context_records = []
    for raw in raw_workloads:
        if not isinstance(raw, Mapping):
            raise TypeError("workload entry must be an object")
        instance_id = str(raw["instance_id"])
        entry = specs.get(instance_id)
        if entry is None:
            raise ValueError(f"missing source context spec for {instance_id}")
        paths = tuple(_validate_repo_path(item) for item in entry.get("paths", ()))
        reference_commit = str(entry["reference_commit"])
        context = _build_context(
            repo,
            base_commit=str(raw["base_commit"]),
            reference_commit=reference_commit,
            paths=paths,
            max_chars=args.max_context_chars,
        )
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        enriched = dict(raw)
        enriched["problem_statement"] = (
            str(raw["problem_statement"])
            + "\n\nRepository observations (read-only pre-fix source; path selection "
            "uses benchmark oracle metadata for systems characterization only):\n"
            + context
        )
        enriched["source_context"] = {
            "selection": "gold_change_paths",
            "content_revision": str(raw["base_commit"]),
            "reference_commit_for_localization_only": reference_commit,
            "paths": list(paths),
            "context_chars": len(context),
            "context_sha256": context_hash,
        }
        output_workloads.append(enriched)
        context_records.append(enriched["source_context"])
    if set(specs) != {str(item["instance_id"]) for item in raw_workloads}:
        raise ValueError("context spec and workload instance sets differ")
    output = dict(workload)
    output["derived_from"] = {
        "workload_manifest": str(workload_path),
        "workload_manifest_sha256": _sha256(workload_path),
        "context_spec": str(spec_path),
        "context_spec_sha256": _sha256(spec_path),
        "max_context_chars": args.max_context_chars,
        "selection": str(spec.get("selection", "unspecified")),
        "evaluation_scope": "systems_characterization_not_swebench_correctness",
    }
    output["workloads"] = output_workloads
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "workload_count": len(output_workloads),
                "context_chars": [item["context_chars"] for item in context_records],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
