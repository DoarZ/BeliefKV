from __future__ import annotations


def require_id(value: str | None, field_name: str) -> str:
    """Validate an externally supplied runtime identifier."""

    if value is None or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
