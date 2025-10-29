from __future__ import annotations

"""
Stateless run–tumble policy that expects `[c_now, c_prev, dc]` observations.

This demonstrates how customizing the observation space can simplify the policy
to be memoryless. Pair with DeltaCObservationWrapper.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class DeltaBasedRunTumblePolicy:
    threshold: float = 1e-6
    eps: float = 0.0
    eps_seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.eps_seed)
        self._space = gym.spaces.Discrete(2)

    @property
    def action_space(self) -> gym.Space:
        return self._space

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def select_action(
        self, observation: np.ndarray, *, explore: bool = False
    ) -> int:  # noqa: ARG002
        # Accept either [c_now, c_prev, dc] or n-back sequence (oldest->newest)
        if observation.shape[0] >= 3:
            # If dc is provided explicitly, trust it
            dc = float(observation[2])
        elif observation.shape[0] >= 2:
            # Compute dc from last two entries (oldest->newest)
            dc = float(observation[-1]) - float(observation[-2])
        else:
            dc = 0.0
        a = 0 if dc >= self.threshold else 1
        if self.eps > 0.0 and self._rng.random() < self.eps:
            a = int(self._rng.integers(0, 2))
        return a
