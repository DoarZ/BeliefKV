from __future__ import annotations

import pytest

from beliefkv.experiments.arrival_schedule import (
    build_workflow_arrivals,
    group_arrivals,
)


def test_batched_arrivals_form_reproducible_waves() -> None:
    arrivals = build_workflow_arrivals(
        7,
        mode="batched",
        batch_size=3,
        batch_interval_seconds=12.5,
    )

    assert [item.scheduled_offset_seconds for item in arrivals] == [
        0.0,
        0.0,
        0.0,
        12.5,
        12.5,
        12.5,
        25.0,
    ]
    assert [len(batch) for batch in group_arrivals(arrivals)] == [3, 3, 1]


def test_simultaneous_arrivals_ignore_batch_interval() -> None:
    arrivals = build_workflow_arrivals(
        4,
        mode="simultaneous",
        batch_size=1,
        batch_interval_seconds=30,
    )

    assert {item.batch_index for item in arrivals} == {0}
    assert {item.scheduled_offset_seconds for item in arrivals} == {0.0}


@pytest.mark.parametrize(
    ("workflow_count", "mode", "batch_size", "interval"),
    (
        (0, "batched", 1, 1.0),
        (1, "invalid", 1, 1.0),
        (1, "batched", 0, 1.0),
        (1, "batched", 1, -1.0),
    ),
)
def test_invalid_arrival_policy_is_rejected(
    workflow_count: int,
    mode: str,
    batch_size: int,
    interval: float,
) -> None:
    with pytest.raises(ValueError):
        build_workflow_arrivals(
            workflow_count,
            mode=mode,
            batch_size=batch_size,
            batch_interval_seconds=interval,
        )
