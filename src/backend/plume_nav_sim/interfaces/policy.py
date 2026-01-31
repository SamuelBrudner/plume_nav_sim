from __future__ import annotations

from typing import Protocol, runtime_checkable

import gymnasium as gym

from plume_nav_sim.core.types import ActionType, ObservationType


@runtime_checkable
class Policy(Protocol):
    @property
    def action_space(self) -> gym.Space:
        pass

    def reset(self, *, seed: int | None = None) -> None:
        pass

    def select_action(
        self, observation: ObservationType, *, explore: bool = True
    ) -> ActionType:
        pass
