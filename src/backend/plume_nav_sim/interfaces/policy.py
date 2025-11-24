"""
Policy Protocol Definition.

Lightweight abstraction for action selection given observations. This enables
plugging in rule-based controllers and RL policies interchangeably.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import gymnasium as gym

from plume_nav_sim.core.types import ActionType, ObservationType


@runtime_checkable
class Policy(Protocol):
    """Protocol defining a stochastic or deterministic policy.

    Minimal interface:
      - action_space: Gymnasium space for actions
      - reset(seed): optional seeding hook
      - select_action(observation, explore): accepts ObservationModel outputs
        (np.ndarray/dict/tuple) and returns an action compatible with action_space
    """

    @property
    def action_space(self) -> gym.Space:
        pass

    def reset(self, *, seed: int | None = None) -> None:
        pass

    def select_action(
        self, observation: ObservationType, *, explore: bool = True
    ) -> ActionType:
        pass
