"""Starter template for ActionProcessor implementations."""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np

from plume_nav_sim.core.geometry import GridSize
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.interfaces.action import ActionProcessor, ActionType


class ExampleActionProcessor(ActionProcessor):
    """Minimal action processor stub you can copy and customize."""

    def __init__(self) -> None:
        self._action_space = gym.spaces.Discrete(4)

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def process_action(
        self,
        action: ActionType,
        current_state: AgentState,
        grid_size: GridSize,
    ) -> AgentState:
        # TODO: map `action` to a new AgentState using `current_state` and `grid_size`.
        _ = (action, grid_size)
        return current_state

    def validate_action(self, action: ActionType) -> bool:
        # TODO: replace with your own action validation logic.
        return isinstance(action, (int, np.integer)) and 0 <= int(action) < 4

    def get_metadata(self) -> Dict[str, Any]:
        return {"name": "example_action_processor"}
