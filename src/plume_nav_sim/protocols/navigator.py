"""Navigation protocol definitions."""

from __future__ import annotations

from typing import Protocol, Dict, Any, runtime_checkable
import numpy as np


@runtime_checkable
class NavigatorProtocol(Protocol):
    """Structural interface for navigation controllers.

    Implementations manage agent motion and internal state updates during a
    simulation. Concrete navigators should provide deterministic behavior
    and expose their state for inspection.
    """

    def reset(self) -> None:
        """Reset the navigator to its initial state."""

    def step(self, action: np.ndarray) -> None:
        """Advance the navigator state using the provided action array."""

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the navigator's current state."""


__all__ = ["NavigatorProtocol"]
