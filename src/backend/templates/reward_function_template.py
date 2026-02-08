"""Starter template for RewardFunction implementations."""

from __future__ import annotations

from typing import Any, Dict

from plume_nav_sim.core.state import AgentState
from plume_nav_sim.interfaces.action import ActionType
from plume_nav_sim.interfaces.reward import RewardFunction
from plume_nav_sim.plume.protocol import ConcentrationField


class ExampleRewardFunction(RewardFunction):
    """Minimal reward function stub you can copy and customize."""

    def compute_reward(
        self,
        prev_state: AgentState,
        action: ActionType,
        next_state: AgentState,
        plume_field: ConcentrationField,
    ) -> float:
        # TODO: compute reward from transition and plume signal.
        _ = (prev_state, action, next_state, plume_field)
        return 0.0

    def get_metadata(self) -> Dict[str, Any]:
        return {"name": "example_reward_function"}
