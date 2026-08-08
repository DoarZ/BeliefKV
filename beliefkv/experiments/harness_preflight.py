from __future__ import annotations


REQUESTS_SANDBOX_PREFLIGHT = r"""
set -eu
python -c 'import pytest_httpbin'
probe="tests/beliefkv_httpbin_preflight_$$.py"
trace="/tmp/beliefkv-pytest-trace-$$.txt"
fixtures="/tmp/beliefkv-pytest-fixtures-$$.txt"
trap 'rm -f "$probe" "$trace" "$fixtures"' EXIT
printf '%s\n' 'def test_httpbin_plugin_is_live(httpbin):' \
  '    assert callable(httpbin)' \
  '    assert httpbin("get").startswith("http")' > "$probe"
if ! python -m pytest --trace-config --collect-only -q "$probe" > "$trace" 2>&1; then
  cat "$trace"
  exit 1
fi
if ! grep -Eqi 'pytest[-_.]?httpbin' "$trace"; then
  cat "$trace"
  exit 1
fi
if ! python -m pytest --fixtures -q "$probe" > "$fixtures" 2>&1; then
  cat "$fixtures"
  exit 1
fi
if ! grep -Eq '^httpbin[[:space:]]' "$fixtures"; then
  cat "$fixtures"
  exit 1
fi
python -m pytest -q "$probe"
""".strip()


_PREFLIGHT_COMMANDS = {
    "psf_requests_pytest_httpbin_v1": REQUESTS_SANDBOX_PREFLIGHT,
}


def preflight_command_for_policy(policy: str | None) -> str | None:
    """Resolve an explicitly declared, versioned harness preflight policy."""

    if policy is None:
        return None
    try:
        return _PREFLIGHT_COMMANDS[policy]
    except KeyError as error:
        raise ValueError(f"unknown harness preflight policy: {policy}") from error
