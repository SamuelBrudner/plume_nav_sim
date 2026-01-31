"""Minimal protocol for concentration fields used by environments and policies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ConcentrationField(Protocol):
    """Duck-typed interface for plume concentration sampling."""

    def sample(self, x: float, y: float, t: float | None = None) -> float:
        """Return concentration at (x, y[, t])."""
        ...

    def reset(self, seed: int | None = None) -> None:
        """Reset any internal state for a new episode."""
        ...
