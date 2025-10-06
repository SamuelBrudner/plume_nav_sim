"""
Tests for SparseGoalReward implementation.

Contract: reward_function_interface.md

Inherits 13 universal tests from TestRewardFunctionInterface.
Adds implementation-specific tests for binary goal reward behavior.
"""

import numpy as np
import pytest

from plume_nav_sim.core.geometry import Coordinates
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.rewards import SparseGoalReward
from tests.contracts.test_reward_function_interface import TestRewardFunctionInterface


class TestSparseGoalReward(TestRewardFunctionInterface):
    __test__ = True
    """Concrete tests for SparseGoalReward.

    Inherits all universal tests from TestRewardFunctionInterface.
    Adds implementation-specific tests for sparse goal behavior.
    """

    # ==============================================================================
    # Fixture Override
    # ==============================================================================

    @pytest.fixture
    def reward_function(self):
        """Provide SparseGoalReward for testing.

        Uses default goal position (0, 0) and goal radius (1.0).

        Returns:
            SparseGoalReward instance with default config
        """
        return SparseGoalReward(
            goal_position=Coordinates(0, 0),
            goal_radius=1.0,
        )

    # ==============================================================================
    # Implementation-Specific Tests
    # ==============================================================================

    def test_returns_one_at_goal(self):
        """SparseGoalReward returns 1.0 when agent reaches goal."""
        goal_pos = Coordinates(10, 10)
        reward_fn = SparseGoalReward(goal_position=goal_pos, goal_radius=0.5)

        # Agent at goal
        prev_state = AgentState(position=Coordinates(5, 5))
        next_state = AgentState(position=goal_pos)
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        reward = reward_fn.compute_reward(prev_state, action, next_state, plume_field)

        assert reward == 1.0, f"Expected reward 1.0 at goal, got {reward}"

    def test_returns_zero_away_from_goal(self):
        """SparseGoalReward returns 0.0 when agent is not at goal."""
        goal_pos = Coordinates(10, 10)
        reward_fn = SparseGoalReward(goal_position=goal_pos, goal_radius=0.5)

        # Agent far from goal
        prev_state = AgentState(position=Coordinates(5, 5))
        next_state = AgentState(position=Coordinates(5, 6))
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        reward = reward_fn.compute_reward(prev_state, action, next_state, plume_field)

        assert reward == 0.0, f"Expected reward 0.0 away from goal, got {reward}"

    def test_goal_radius_works(self):
        """SparseGoalReward considers positions within goal_radius as goal."""
        goal_pos = Coordinates(10, 10)
        reward_fn = SparseGoalReward(goal_position=goal_pos, goal_radius=2.0)

        # Agent within radius (distance = sqrt(2) ≈ 1.41 < 2.0)
        prev_state = AgentState(position=Coordinates(5, 5))
        next_state = AgentState(position=Coordinates(11, 11))
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        reward = reward_fn.compute_reward(prev_state, action, next_state, plume_field)

        assert reward == 1.0, f"Expected reward 1.0 within radius, got {reward}"

    def test_just_outside_radius_returns_zero(self):
        """SparseGoalReward returns 0.0 just outside goal radius."""
        goal_pos = Coordinates(10, 10)
        reward_fn = SparseGoalReward(goal_position=goal_pos, goal_radius=1.0)

        # Agent just outside radius (distance = 2.0 > 1.0)
        prev_state = AgentState(position=Coordinates(5, 5))
        next_state = AgentState(position=Coordinates(12, 10))
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        reward = reward_fn.compute_reward(prev_state, action, next_state, plume_field)

        assert reward == 0.0, f"Expected reward 0.0 outside radius, got {reward}"

    def test_only_depends_on_next_state(self):
        """SparseGoalReward only depends on next_state, not prev_state."""
        goal_pos = Coordinates(10, 10)
        reward_fn = SparseGoalReward(goal_position=goal_pos, goal_radius=0.5)

        # Same next_state, different prev_state
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)
        next_state = AgentState(position=goal_pos)

        reward1 = reward_fn.compute_reward(
            AgentState(position=Coordinates(5, 5)), action, next_state, plume_field
        )
        reward2 = reward_fn.compute_reward(
            AgentState(position=Coordinates(20, 20)), action, next_state, plume_field
        )

        assert reward1 == reward2 == 1.0, "Reward should only depend on next_state"

    def test_metadata_structure(self, reward_function):
        """SparseGoalReward metadata has expected structure."""
        metadata = reward_function.get_metadata()

        assert metadata["type"] == "sparse_goal_reward"
        assert "goal_position" in metadata
        assert "goal_radius" in metadata
        assert isinstance(metadata["goal_position"], dict)
        assert "x" in metadata["goal_position"]
        assert "y" in metadata["goal_position"]
        assert isinstance(metadata["goal_radius"], float)

    def test_different_goal_positions(self):
        """SparseGoalReward works with different goal positions."""
        # Test multiple goal positions
        goals = [
            Coordinates(0, 0),
            Coordinates(15, 15),
            Coordinates(30, 5),
        ]

        for goal in goals:
            reward_fn = SparseGoalReward(goal_position=goal, goal_radius=0.5)

            prev_state = AgentState(position=Coordinates(10, 10))
            next_state = AgentState(position=goal)
            action = 0
            plume_field = np.zeros((32, 32), dtype=np.float32)

            reward = reward_fn.compute_reward(
                prev_state, action, next_state, plume_field
            )

            assert reward == 1.0, f"Expected reward 1.0 at goal {goal}, got {reward}"
