"""Sensor protocol definitions."""

from __future__ import annotations

from typing import Protocol, Sequence, Any, runtime_checkable


@runtime_checkable
class SensorProtocol(Protocol):
    """Interface for plume sensing components.

    Sensors provide detection or measurement capabilities for agents based on
    the underlying plume model state.
    """

    def detect(self, plume_state: Any, positions: Sequence[Sequence[float]]) -> Sequence[bool]:
        """Return detection events for agents at ``positions``."""

    def measure(self, plume_state: Any, positions: Sequence[Sequence[float]]) -> Sequence[float]:
        """Return scalar measurements for agents at ``positions``."""


__all__ = ["SensorProtocol"]
