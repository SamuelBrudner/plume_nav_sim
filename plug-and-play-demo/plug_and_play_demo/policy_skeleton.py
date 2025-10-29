from __future__ import annotations

"""
Hello Policy: minimal skeleton matching the plume_nav_sim Policy protocol.

Copy this into your project and fill in select_action() logic.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class MyPolicy:
    # Add your tunable parameters here
    seed: int | None = None

    def __post_init__(self) -> None:
        # Optional internal RNG/state
        self._rng = np.random.default_rng(self.seed)
        self._space = gym.spaces.Discrete(2)  # Replace with your action space

    @property
    def action_space(self) -> gym.Space:
        # Must return the same object instance each call
        return self._space

    def reset(self, *, seed: int | None = None) -> None:
        # Optional determinism hook; runner/compose call this when seeding
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def select_action(
        self, observation: np.ndarray, *, explore: bool = False
    ) -> int:  # noqa: ARG002
        # Replace with your logic. This fallback samples a random action.
        return int(self._space.sample())
