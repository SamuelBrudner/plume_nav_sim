"""
Sparse goal reward function - binary reward at goal.

Contract: reward_function_interface.md
"""

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
    """Sparse binary reward function for goal-reaching tasks.

    Satisfies RewardFunction protocol via duck typing.

    Reward Structure:
        - 1.0 if agent reaches goal (within goal_radius)
        - 0.0 otherwise

    Properties:
        - Deterministic: Same state → same reward
        - Pure: No side effects
        - Finite: Always returns 0.0 or 1.0
        - Memoryless: Only depends on current state, not history

    Contract: reward_function_interface.md - RewardFunction protocol

    Example:
        >>> reward_fn = SparseGoalReward(
        ...     goal_position=Coordinates(15, 15),
        ...     goal_radius=1.0
        ... )
        >>> # At goal
        >>> reward = reward_fn.compute_reward(prev, action, at_goal_state, field)
        >>> assert reward == 1.0
        >>> # Away from goal
        >>> reward = reward_fn.compute_reward(prev, action, away_state, field)
        >>> assert reward == 0.0
    """

    def __init__(
        self,
        goal_position: Coordinates,
        goal_radius: float = 1.0,
    ):
        """Initialize SparseGoalReward.

        Args:
            goal_position: Target position (x, y) in grid coordinates
            goal_radius: Distance threshold for goal success (grid cells)

        Raises:
            ValueError: If goal_radius is not positive
        """
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
        """Compute sparse goal reward.

        Returns 1.0 if agent is within goal_radius of goal_position,
        otherwise returns 0.0.

        Args:
            prev_state: AgentState before action (unused, for protocol compliance)
            action: Action taken (unused, for protocol compliance)
            next_state: AgentState after action (determines reward)
            plume_field: Concentration field (supports ConcentrationField or raw ndarray)

        Returns:
            1.0 if at goal, 0.0 otherwise

        Contract: reward_function_interface.md - compute_reward()

        Postconditions:
            C1: result is finite float (0.0 or 1.0)
            C2: result is deterministic
            C3: result is numeric type

        Properties:
            1. Determinism: Same next_state → same reward
            2. Purity: No side effects
            3. Finiteness: Always 0.0 or 1.0
        """
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
        """Return reward function metadata.

        Returns:
            Dictionary with reward configuration and type

        Contract: reward_function_interface.md - get_metadata()

        Postconditions:
            C1: Returns dictionary with 'type' key
            C2: All values are JSON-serializable
        """
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
