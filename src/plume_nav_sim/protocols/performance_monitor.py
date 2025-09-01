"""Performance monitoring protocol definitions."""

from __future__ import annotations

from typing import Protocol, Dict, Any, runtime_checkable


@runtime_checkable
class PerformanceMonitorProtocol(Protocol):
    """Interface for collecting simulation performance metrics."""

    def record_step_time(self, seconds: float) -> None:
        """Record metrics for a single simulation step in seconds."""

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregated performance statistics."""

    # Backward compatibility shims
    def record_step(self, duration_ms: float, label: str | None = None) -> None:
        """Record metrics for a single simulation step in milliseconds."""
        self.record_step_time(duration_ms / 1000.0)

    def export(self) -> Dict[str, Any]:
        """Return aggregated performance statistics."""
        return self.get_summary()


__all__ = ["PerformanceMonitorProtocol"]
