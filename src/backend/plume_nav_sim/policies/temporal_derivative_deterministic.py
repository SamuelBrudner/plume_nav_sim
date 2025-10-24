from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from ..actions.oriented_grid import OrientedGridActions
from ..interfaces import Policy


@dataclass
class TemporalDerivativeDeterministicPolicy(Policy):
    """Deterministic temporal-gradient policy for oriented control.

    - Maintains 1-back concentration measured only after FORWARD steps
    - Surges FORWARD on non-decreasing concentration (dC >= threshold)
    - Otherwise casts by turning; enforces a FORWARD probe right after any TURN
    - Casting alternates deterministically RIGHT/LEFT/RIGHT/...
    """

    threshold: float = 1e-6
    cast_right_first: bool = True

    def __post_init__(self) -> None:
        self._actions = OrientedGridActions()
        self._prev_moving: Optional[float] = None
        self._last_action: Optional[int] = None
        self._cast_right_next: bool = self.cast_right_first

    @property
    def action_space(self) -> gym.Space:
        return self._actions.action_space

    def reset(self, *, seed: int | None = None) -> None:
        self._prev_moving = None
        self._last_action = None
        self._cast_right_next = self.cast_right_first

    def select_action(self, observation: np.ndarray, *, explore: bool = False) -> int:
        c = float(observation[0])

        # First step: move forward to obtain moving sample
        if self._prev_moving is None:
            self._prev_moving = c
            self._last_action = 0  # FORWARD
            return 0

        # After a TURN, force a FORWARD probe to avoid spinning on zero dC
        if self._last_action in (1, 2):
            self._last_action = 0
            return 0

        # Greedy decision based on temporal derivative
        dc = c - self._prev_moving
        if dc >= self.threshold:
            action = 0  # FORWARD
            self._prev_moving = c
        else:
            # Deterministic alternating cast direction
            action = 2 if self._cast_right_next else 1  # RIGHT then LEFT alternating
            self._cast_right_next = not self._cast_right_next

        self._last_action = action
        return action
