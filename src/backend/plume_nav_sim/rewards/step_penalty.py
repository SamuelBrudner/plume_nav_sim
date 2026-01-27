from typing import Any, Dict

import numpy as np

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]

from plume_nav_sim.core.geometry import Coordinates
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.interfaces.action import ActionType
from plume_nav_sim.plume.protocol import ConcentrationField


class StepPenaltyReward:
    def __init__(
        self,
        goal_position: Coordinates,
        goal_radius: float = 1.0,
        goal_reward: float = 1.0,
        step_penalty: float = 0.01,
    ):
        if goal_radius <= 0.0:
            raise ValueError(f"goal_radius must be positive, got {goal_radius}")
        if step_penalty < 0.0:
            raise ValueError(f"step_penalty must be non-negative, got {step_penalty}")

        self.goal_position = goal_position
        self.goal_radius = goal_radius
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty

    def compute_reward(
        self,
        prev_state: AgentState,
        action: ActionType,
        next_state: AgentState,
        plume_field: ConcentrationField | NDArray[np.floating],
    ) -> float:
        # Compute Euclidean distance from agent to goal
        dx = next_state.position.x - self.goal_position.x
        dy = next_state.position.y - self.goal_position.y
        distance = np.sqrt(dx**2 + dy**2)

        # Check if agent reached goal
        return self.goal_reward if distance <= self.goal_radius else -self.step_penalty

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": "step_penalty_reward",
            "goal_position": {
                "x": self.goal_position.x,
                "y": self.goal_position.y,
            },
            "goal_radius": float(self.goal_radius),
            "goal_reward": float(self.goal_reward),
            "step_penalty": float(self.step_penalty),
        }
