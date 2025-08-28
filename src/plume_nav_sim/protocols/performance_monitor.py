"""Performance monitoring protocol definitions."""

from __future__ import annotations

from typing import Protocol, Dict, runtime_checkable


@runtime_checkable
class PerformanceMonitorProtocol(Protocol):
    """Interface for collecting simulation performance metrics."""

    def record_step(self, duration_ms: float, label: str | None = None) -> None:
        """Record metrics for a single simulation step."""

    def export(self) -> Dict[str, float]:
        """Return accumulated performance statistics."""


__all__ = ["PerformanceMonitorProtocol"]
