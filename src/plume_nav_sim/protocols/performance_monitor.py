"""Performance monitoring protocol definitions."""

from __future__ import annotations

from typing import Protocol, Dict, Any, runtime_checkable


@runtime_checkable
class PerformanceMonitorProtocol(Protocol):
    """Interface for collecting simulation performance metrics."""

    def record_step(self, data: Dict[str, Any]) -> None:
        """Record metrics for a single simulation step."""

    def record_episode(self, data: Dict[str, Any]) -> None:
        """Record aggregated metrics for an episode."""

    def get_metrics(self) -> Dict[str, float]:
        """Return accumulated performance statistics."""


__all__ = ["PerformanceMonitorProtocol"]
