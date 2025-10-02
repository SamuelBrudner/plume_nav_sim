"""
Tests for DenseNavigationReward implementation.

Contract: reward_function_interface.md

Inherits 13 universal tests from TestRewardFunctionInterface.
Adds implementation-specific tests for dense distance-based reward behavior.
"""

import numpy as np
import pytest

from plume_nav_sim.core.geometry import Coordinates
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.rewards import DenseNavigationReward
from tests.contracts.test_reward_function_interface import TestRewardFunctionInterface


class TestDenseNavigationReward(TestRewardFunctionInterface):
    """Concrete tests for DenseNavigationReward.

    Inherits all universal tests from TestRewardFunctionInterface.
    Adds implementation-specific tests for dense distance-based shaping.
    """

    # ==============================================================================
    # Fixture Override
    # ==============================================================================

    @pytest.fixture
    def reward_function(self):
        """Provide DenseNavigationReward for testing.

        Uses goal at (0, 0) and max distance of 10.0.

        Returns:
            DenseNavigationReward instance with default config
        """
        return DenseNavigationReward(
            goal_position=Coordinates(0, 0),
            max_distance=10.0,
        )

    # ==============================================================================
    # Implementation-Specific Tests
    # ==============================================================================

    def test_returns_one_at_goal(self):
        """DenseNavigationReward returns 1.0 when agent is at goal."""
        goal_pos = Coordinates(10, 10)
        reward_fn = DenseNavigationReward(goal_position=goal_pos, max_distance=10.0)

        # Agent at goal
        prev_state = AgentState(position=Coordinates(5, 5))
        next_state = AgentState(position=goal_pos)
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        reward = reward_fn.compute_reward(prev_state, action, next_state, plume_field)

        assert reward == 1.0, f"Expected reward 1.0 at goal, got {reward}"

    def test_returns_zero_at_max_distance(self):
        """DenseNavigationReward returns 0.0 at max_distance."""
        goal_pos = Coordinates(10, 10)
        max_dist = 10.0
        reward_fn = DenseNavigationReward(goal_position=goal_pos, max_distance=max_dist)

        # Agent exactly at max_distance (10 units away)
        prev_state = AgentState(position=Coordinates(5, 5))
        next_state = AgentState(position=Coordinates(20, 10))  # 10 units away
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        reward = reward_fn.compute_reward(prev_state, action, next_state, plume_field)

        assert reward == 0.0, f"Expected reward 0.0 at max_distance, got {reward}"

    def test_reward_decreases_with_distance(self):
        """DenseNavigationReward decreases monotonically with distance."""
        goal_pos = Coordinates(10, 10)
        reward_fn = DenseNavigationReward(goal_position=goal_pos, max_distance=10.0)

        prev_state = AgentState(position=Coordinates(5, 5))
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        # Test positions at increasing distances
        distances = [0.0, 2.0, 5.0, 8.0, 10.0]
        rewards = []

        for dist in distances:
            # Position at distance 'dist' from goal (along x-axis for simplicity)
            next_state = AgentState(position=Coordinates(10 + int(dist), 10))
            reward = reward_fn.compute_reward(
                prev_state, action, next_state, plume_field
            )
            rewards.append(reward)

        # Rewards should be monotonically decreasing
        for i in range(len(rewards) - 1):
            assert (
                rewards[i] >= rewards[i + 1]
            ), f"Reward should decrease with distance: {rewards}"

    def test_reward_is_continuous(self):
        """DenseNavigationReward provides smooth continuous rewards."""
        goal_pos = Coordinates(10, 10)
        reward_fn = DenseNavigationReward(goal_position=goal_pos, max_distance=10.0)

        prev_state = AgentState(position=Coordinates(5, 5))
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        # Test nearby positions have similar rewards
        pos1 = AgentState(position=Coordinates(15, 10))  # distance = 5
        pos2 = AgentState(position=Coordinates(15, 11))  # distance ≈ 5.1

        reward1 = reward_fn.compute_reward(prev_state, action, pos1, plume_field)
        reward2 = reward_fn.compute_reward(prev_state, action, pos2, plume_field)

        # Small change in distance should produce small change in reward
        assert abs(reward1 - reward2) < 0.1, "Reward should be continuous"

    def test_reward_range_is_zero_to_one(self):
        """DenseNavigationReward always returns values in [0, 1]."""
        goal_pos = Coordinates(10, 10)
        reward_fn = DenseNavigationReward(goal_position=goal_pos, max_distance=10.0)

        prev_state = AgentState(position=Coordinates(5, 5))
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        # Test various positions
        test_positions = [
            Coordinates(10, 10),  # at goal
            Coordinates(11, 10),  # close
            Coordinates(15, 10),  # mid
            Coordinates(20, 10),  # at max
            Coordinates(25, 10),  # beyond max
        ]

        for pos in test_positions:
            next_state = AgentState(position=pos)
            reward = reward_fn.compute_reward(
                prev_state, action, next_state, plume_field
            )

            assert 0.0 <= reward <= 1.0, f"Reward {reward} at {pos} is outside [0, 1]"

    def test_beyond_max_distance_returns_zero(self):
        """DenseNavigationReward returns 0.0 beyond max_distance."""
        goal_pos = Coordinates(10, 10)
        max_dist = 10.0
        reward_fn = DenseNavigationReward(goal_position=goal_pos, max_distance=max_dist)

        prev_state = AgentState(position=Coordinates(5, 5))
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        # Agent well beyond max_distance
        next_state = AgentState(position=Coordinates(30, 10))  # 20 units away
        reward = reward_fn.compute_reward(prev_state, action, next_state, plume_field)

        assert reward == 0.0, f"Expected reward 0.0 beyond max_distance, got {reward}"

    def test_only_depends_on_next_state(self):
        """DenseNavigationReward only depends on next_state position."""
        goal_pos = Coordinates(10, 10)
        reward_fn = DenseNavigationReward(goal_position=goal_pos, max_distance=10.0)

        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)
        next_state = AgentState(position=Coordinates(15, 10))

        # Same next_state, different prev_state
        reward1 = reward_fn.compute_reward(
            AgentState(position=Coordinates(5, 5)), action, next_state, plume_field
        )
        reward2 = reward_fn.compute_reward(
            AgentState(position=Coordinates(20, 20)), action, next_state, plume_field
        )

        assert reward1 == reward2, "Reward should only depend on next_state"

    def test_metadata_structure(self, reward_function):
        """DenseNavigationReward metadata has expected structure."""
        metadata = reward_function.get_metadata()

        assert metadata["type"] == "dense_navigation_reward"
        assert "goal_position" in metadata
        assert "max_distance" in metadata
        assert isinstance(metadata["goal_position"], dict)
        assert "x" in metadata["goal_position"]
        assert "y" in metadata["goal_position"]
        assert isinstance(metadata["max_distance"], float)

    def test_different_max_distances(self):
        """DenseNavigationReward works with different max_distance values."""
        goal_pos = Coordinates(10, 10)

        # Test with different max distances
        max_distances = [5.0, 10.0, 20.0]

        for max_dist in max_distances:
            reward_fn = DenseNavigationReward(
                goal_position=goal_pos, max_distance=max_dist
            )

            prev_state = AgentState(position=Coordinates(5, 5))
            action = 0
            plume_field = np.zeros((32, 32), dtype=np.float32)

            # Test at goal
            next_state = AgentState(position=goal_pos)
            reward = reward_fn.compute_reward(
                prev_state, action, next_state, plume_field
            )
            assert (
                reward == 1.0
            ), f"Expected reward 1.0 at goal with max_dist={max_dist}"

            # Test at half max_distance
            half_dist_pos = Coordinates(10 + int(max_dist / 2), 10)
            next_state = AgentState(position=half_dist_pos)
            reward = reward_fn.compute_reward(
                prev_state, action, next_state, plume_field
            )
            assert (
                0.0 < reward < 1.0
            ), f"Expected reward in (0, 1) at half distance with max_dist={max_dist}"

    def test_validates_max_distance_positive(self):
        """DenseNavigationReward rejects non-positive max_distance."""
        goal_pos = Coordinates(10, 10)

        with pytest.raises(ValueError, match="max_distance must be positive"):
            DenseNavigationReward(goal_position=goal_pos, max_distance=0.0)

        with pytest.raises(ValueError, match="max_distance must be positive"):
            DenseNavigationReward(goal_position=goal_pos, max_distance=-5.0)
