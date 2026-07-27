from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from beliefkv.runtime.protocol import CommandQueueClass, ControlCommand


@dataclass(order=True)
class _QueueEntry:
    deadline_ms: float
    negative_priority: float
    sequence: int
    command: ControlCommand = field(compare=False)


class TransferCommandQueue:
    """Strict-priority urgent/shadow queue with deterministic tie breaking."""

    def __init__(self) -> None:
        self._urgent: list[_QueueEntry] = []
        self._shadow: list[_QueueEntry] = []
        self._sequence = 0
        self._active_ids: set[str] = set()
        self._cancelled_ids: set[str] = set()

    def put(self, command: ControlCommand) -> None:
        if command.command_id in self._active_ids:
            raise ValueError(f"duplicate command id: {command.command_id}")
        entry = _QueueEntry(
            deadline_ms=command.deadline_ms,
            negative_priority=-command.priority,
            sequence=self._sequence,
            command=command,
        )
        self._sequence += 1
        queue = (
            self._urgent
            if command.queue_class == CommandQueueClass.URGENT
            else self._shadow
        )
        heapq.heappush(queue, entry)
        self._active_ids.add(command.command_id)

    def pop(self, *, allow_shadow: bool = True) -> ControlCommand | None:
        command = self._pop_valid(self._urgent)
        if command is not None:
            return command
        if allow_shadow:
            return self._pop_valid(self._shadow)
        return None

    def _pop_valid(self, queue: list[_QueueEntry]) -> ControlCommand | None:
        while queue:
            command = heapq.heappop(queue).command
            self._active_ids.discard(command.command_id)
            if command.command_id in self._cancelled_ids:
                self._cancelled_ids.discard(command.command_id)
                continue
            return command
        return None

    def cancel(self, command_id: str) -> bool:
        if command_id not in self._active_ids:
            return False
        self._cancelled_ids.add(command_id)
        self._active_ids.discard(command_id)
        return True

    @property
    def urgent_count(self) -> int:
        return sum(
            entry.command.command_id not in self._cancelled_ids
            for entry in self._urgent
        )

    @property
    def shadow_count(self) -> int:
        return sum(
            entry.command.command_id not in self._cancelled_ids
            for entry in self._shadow
        )

    def pending_commands(self) -> tuple[ControlCommand, ...]:
        """Return a deterministic, non-destructive queue snapshot."""

        entries = [
            *(
                (0, item)
                for item in self._urgent
                if item.command.command_id not in self._cancelled_ids
            ),
            *(
                (1, item)
                for item in self._shadow
                if item.command.command_id not in self._cancelled_ids
            ),
        ]
        return tuple(
            item.command
            for _, item in sorted(
                entries,
                key=lambda value: (
                    value[0],
                    value[1].deadline_ms,
                    value[1].negative_priority,
                    value[1].sequence,
                ),
            )
        )

    def __len__(self) -> int:
        return self.urgent_count + self.shadow_count
