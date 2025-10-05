"""
Step penalty reward function - goal bonus with time penalty.

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


class StepPenaltyReward:
    """Goal reward with per-step penalty for time efficiency.

    Satisfies RewardFunction protocol via duck typing.

    Reward Structure:
        - goal_reward (e.g., 10.0) when agent reaches goal
        - -step_penalty (e.g., -0.01) for each step taken
        - Encourages finding goal quickly

    Properties:
        - Deterministic: Same state → same reward
        - Pure: No side effects
        - Finite: Always returns finite value
        - Time-aware: Penalizes long episodes

    Contract: reward_function_interface.md - RewardFunction protocol

    Example:
        >>> reward_fn = StepPenaltyReward(
        ...     goal_position=Coordinates(15, 15),
        ...     goal_radius=5.0,
        ...     goal_reward=10.0,
        ...     step_penalty=0.01
        ... )
        >>> # At goal
        >>> reward = reward_fn.compute_reward(prev, action, at_goal_state, field)
        >>> assert reward == 10.0
        >>> # Searching
        >>> reward = reward_fn.compute_reward(prev, action, search_state, field)
        >>> assert reward == -0.01
    """

    def __init__(
        self,
        goal_position: Coordinates,
        goal_radius: float = 1.0,
        goal_reward: float = 1.0,
        step_penalty: float = 0.01,
    ):
        """Initialize StepPenaltyReward.

        Args:
            goal_position: Target position (x, y) in grid coordinates
            goal_radius: Distance threshold for goal success (grid cells)
            goal_reward: Reward value when goal is reached
            step_penalty: Penalty per step (positive value, will be negated)

        Raises:
            ValueError: If goal_radius or step_penalty is not positive
        """
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
        action: int,
        next_state: AgentState,
        plume_field: NDArray[np.floating],
    ) -> float:
        """Compute step penalty reward.

        Returns goal_reward if agent reaches goal, otherwise returns
        negative step_penalty to encourage efficiency.

        Args:
            prev_state: AgentState before action (unused, for protocol compliance)
            action: Action taken (unused, for protocol compliance)
            next_state: AgentState after action (determines reward)
            plume_field: Concentration field (unused, for protocol compliance)

        Returns:
            goal_reward if at goal, -step_penalty otherwise

        Contract: reward_function_interface.md - compute_reward()

        Postconditions:
            C1: result is finite float
            C2: result is deterministic
            C3: result is numeric type

        Properties:
            1. Determinism: Same next_state → same reward
            2. Purity: No side effects
            3. Finiteness: Always finite (goal_reward or -step_penalty)
        """
        # Compute Euclidean distance from agent to goal
        dx = next_state.position.x - self.goal_position.x
        dy = next_state.position.y - self.goal_position.y
        distance = np.sqrt(dx**2 + dy**2)

        # Check if agent reached goal
        return self.goal_reward if distance <= self.goal_radius else -self.step_penalty

    def get_metadata(self) -> Dict[str, Any]:
        """Return reward function metadata.

        Returns:
            Dictionary with reward configuration

        Contract: reward_function_interface.md - get_metadata()
        """
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
