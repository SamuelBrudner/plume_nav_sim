"""
Dense navigation reward function - distance-based continuous reward.

Contract: reward_function_interface.md
"""

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from plume_nav_sim.core.geometry import Coordinates
from plume_nav_sim.core.state import AgentState


class DenseNavigationReward:
    """Dense distance-based reward function for navigation tasks.

    Satisfies RewardFunction protocol via duck typing.

    Reward Structure:
        - 1.0 at goal position (distance = 0)
        - Linear decay with distance
        - 0.0 at max_distance or beyond

    Properties:
        - Deterministic: Same position → same reward
        - Pure: No side effects
        - Finite: Always returns value in [0.0, 1.0]
        - Continuous: Smooth gradient for learning
        - Monotonic: Closer positions → higher rewards

    Contract: reward_function_interface.md - RewardFunction protocol

    Example:
        >>> reward_fn = DenseNavigationReward(
        ...     goal_position=Coordinates(15, 15),
        ...     max_distance=10.0
        ... )
        >>> # At goal
        >>> reward = reward_fn.compute_reward(prev, action, at_goal_state, field)
        >>> assert reward == 1.0
        >>> # At half distance
        >>> reward = reward_fn.compute_reward(prev, action, halfway_state, field)
        >>> assert reward == 0.5
        >>> # Beyond max_distance
        >>> reward = reward_fn.compute_reward(prev, action, far_state, field)
        >>> assert reward == 0.0
    """

    def __init__(
        self,
        goal_position: Coordinates,
        max_distance: float = 10.0,
    ):
        """Initialize DenseNavigationReward.

        Args:
            goal_position: Target position (x, y) in grid coordinates
            max_distance: Maximum distance for reward scaling

        Raises:
            ValueError: If max_distance is not positive
        """
        if max_distance <= 0.0:
            raise ValueError(f"max_distance must be positive, got {max_distance}")

        self.goal_position = goal_position
        self.max_distance = max_distance

    def compute_reward(
        self,
        prev_state: AgentState,
        action: int,
        next_state: AgentState,
        plume_field: NDArray[np.floating],
    ) -> float:
        """Compute dense navigation reward.

        Returns a continuous reward based on distance to goal:
        - 1.0 at goal position
        - Linearly decreases with distance
        - 0.0 at max_distance or beyond

        Formula: reward = max(0, 1 - distance/max_distance)

        Args:
            prev_state: AgentState before action (unused)
            action: Action taken (unused)
            next_state: AgentState after action (determines reward)
            plume_field: Concentration field (unused)

        Returns:
            Reward in [0.0, 1.0] based on distance to goal

        Properties:
            1. Determinism: Same position → same reward
            2. Purity: No side effects
            3. Finiteness: Always returns value in [0.0, 1.0]
            4. Continuous: Small position changes → small reward changes
            5. Monotonic: Closer to goal → higher reward
        """
        # Compute Euclidean distance from agent to goal
        dx = next_state.position.x - self.goal_position.x
        dy = next_state.position.y - self.goal_position.y
        distance = np.sqrt(dx**2 + dy**2)

        # Linear decay from 1.0 at goal to 0.0 at max_distance
        # Clamp to [0, 1] to handle distances beyond max_distance
        if distance >= self.max_distance:
            return 0.0
        else:
            return 1.0 - (distance / self.max_distance)

    def get_metadata(self) -> Dict[str, Any]:
        """Return reward function metadata.

        Returns:
            Dictionary with reward configuration
        """
        return {
            "type": "dense_navigation_reward",
            "goal_position": {
                "x": self.goal_position.x,
                "y": self.goal_position.y,
            },
            "max_distance": float(self.max_distance),
        }
