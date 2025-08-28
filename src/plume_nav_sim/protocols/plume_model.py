"""Plume model protocol definitions."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class PlumeModelProtocol(Protocol):
    """Interface for concentration field models."""

    def concentration_at(self, positions: Sequence[Sequence[float]]) -> Sequence[float]:
        """Return plume concentration at ``positions``."""

    def step(self, dt: float) -> None:
        """Advance the plume model by ``dt`` seconds."""

    def reset(self) -> None:
        """Reset the plume model state."""


__all__ = ["PlumeModelProtocol"]
