#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from beliefkv.runtime.sglang_adapter import (
    BASE_SGLANG_GIT_COMMIT,
    BASE_SGLANG_VERSION,
    SGLangSourceContract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="SGLang repository root")
    args = parser.parse_args()
    report = SGLangSourceContract().check(args.source.resolve())
    print(
        json.dumps(
            {
                "expected_version": BASE_SGLANG_VERSION,
                "expected_commit": BASE_SGLANG_GIT_COMMIT,
                "compatible": report.compatible,
                "failures": [asdict(item) for item in report.failures],
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if report.compatible else 1)


if __name__ == "__main__":
    main()
