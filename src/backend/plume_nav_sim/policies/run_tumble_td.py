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
    - One-step tumble: no action queue.
    - Decision:
        * If dc >= threshold: RUN → FORWARD
        * If dc < threshold: TUMBLE → single random TURN (LEFT or RIGHT). The
          next step will naturally be a FORWARD probe in consumer logic.

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

    # Policy protocol -----------------------------------------------------
    @property
    def action_space(self) -> gym.Space:
        return self._actions.action_space

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_c = None

    def select_action(
        self, observation: np.ndarray, *, explore: bool = False
    ) -> int:  # noqa: ARG002
        c = float(observation[0])

        # Initialize reference and start by moving forward
        if self._last_c is None:
            self._last_c = c
            return 0

        # Compute dc identically every step
        dc = c - self._last_c

        if dc >= self.threshold:
            a = 0  # RUN -> FORWARD
        else:
            # TUMBLE: one-step random turn; encoded as action=1 for run-tumble actions
            a = 1

        self._last_c = c
        return a
