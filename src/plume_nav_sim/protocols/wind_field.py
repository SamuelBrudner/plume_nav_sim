"""Wind field protocol definitions."""

from __future__ import annotations

from typing import Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class WindFieldProtocol(Protocol):
    """Interface for wind field models affecting plume transport."""

    def velocity_at(self, positions: Sequence[Sequence[float]]) -> Sequence[Tuple[float, float]]:
        """Return wind velocity vectors at ``positions``."""

    def step(self, dt: float) -> None:
        """Advance the wind field by ``dt`` seconds."""

    def reset(self) -> None:
        """Reset the wind field state."""


__all__ = ["WindFieldProtocol"]
