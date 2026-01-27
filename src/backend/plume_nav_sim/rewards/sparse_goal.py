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


class SparseGoalReward:
    def __init__(
        self,
        goal_position: Coordinates,
        goal_radius: float = 1.0,
    ):
        if goal_radius <= 0.0:
            raise ValueError(f"goal_radius must be positive, got {goal_radius}")

        self.goal_position = goal_position
        self.goal_radius = goal_radius

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

        # Return 1.0 if within goal radius, 0.0 otherwise
        if distance <= self.goal_radius:
            return 1.0
        else:
            return 0.0

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": "sparse_goal_reward",
            "goal_position": {
                "x": self.goal_position.x,
                "y": self.goal_position.y,
            },
            "goal_radius": float(self.goal_radius),
            "reward_range": [0.0, 1.0],
            "description": "Binary reward: 1.0 at goal, 0.0 elsewhere",
        }
