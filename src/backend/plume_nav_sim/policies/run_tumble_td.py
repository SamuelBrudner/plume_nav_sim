from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from ..actions.oriented_grid import OrientedGridActions
from ..interfaces import Policy


@dataclass
class RunTumbleTemporalDerivativePolicy(Policy):
    """Temporal-derivative run-and-tumble policy over oriented actions.

    Semantics:
    - Always compute dc = c_now - last_c identically every step.
    - If a tumble is in progress, output the next queued action (one or more turns
      followed by a forward probe) until the queue is empty.
    - Otherwise:
        * If dc >= threshold: RUN → FORWARD
        * If dc < threshold: TUMBLE → choose a new heading at random by scheduling
          1–3 quarter-turns in a random direction, then a FORWARD probe.

    Notes:
    - This policy uses only the 1-D concentration observation; it does not require
      current orientation from the environment. Randomly sampling 1–3 turns in either
      direction approximates choosing a new absolute heading uniformly.
    """

    threshold: float = 1e-6
    eps_seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._actions = OrientedGridActions()
        self._rng = np.random.default_rng(self.eps_seed)
        self._last_c: Optional[float] = None
        self._pending: list[int] = (
            []
        )  # queued oriented actions (1=LEFT, 2=RIGHT, 0=FWD)

    # Policy protocol -----------------------------------------------------
    @property
    def action_space(self) -> gym.Space:
        return self._actions.action_space

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_c = None
        self._pending.clear()

    def select_action(
        self, observation: np.ndarray, *, explore: bool = False
    ) -> int:  # noqa: ARG002
        c = float(observation[0])

        # Initialize reference and start by moving forward
        if self._last_c is None:
            self._last_c = c
            return 0

        # If a tumble is in progress, continue it
        if self._pending:
            a = int(self._pending.pop(0))
            self._last_c = c
            return a

        # Compute dc identically every step
        dc = c - self._last_c

        if dc >= self.threshold:
            a = 0  # RUN -> FORWARD
        else:
            # TUMBLE: pick a random new heading via k quarter-turns and then forward.
            k = int(self._rng.integers(1, 4))  # 1..3 turns
            turn = 1 if self._rng.random() < 0.5 else 2  # LEFT or RIGHT
            self._pending = [turn] * k + [0]
            a = int(self._pending.pop(0))

        self._last_c = c
        return a
