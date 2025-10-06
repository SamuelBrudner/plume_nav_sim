"""
Unit tests for StepPenaltyReward.

Contract: src/backend/contracts/reward_function_interface.md

Tests both universal properties (via inheritance) and implementation-specific
behavior of StepPenaltyReward.
"""

import numpy as np
import pytest

from plume_nav_sim.core.geometry import Coordinates
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.interfaces import RewardFunction
from plume_nav_sim.rewards import StepPenaltyReward
from tests.contracts.test_reward_function_interface import TestRewardFunctionInterface


class TestStepPenaltyReward(TestRewardFunctionInterface):
    """Test suite for StepPenaltyReward.

    Inherits universal property tests from TestRewardFunctionInterface.
    Adds implementation-specific tests.
    """

    # ==============================================================================
    # Fixtures
    # ==============================================================================

    @pytest.fixture
    def reward_function(self) -> RewardFunction:
        """Provide StepPenaltyReward instance for testing."""
        return StepPenaltyReward(
            goal_position=Coordinates(64, 64),
            goal_radius=1.0,
            goal_reward=1.0,
            step_penalty=0.01,
        )

    # ==============================================================================
    # Implementation-Specific Tests
    # ==============================================================================

    def test_penalty_when_away_from_goal(self, reward_function):
        """Reward should be negative (penalty) when away from goal."""
        prev_state = AgentState(position=Coordinates(10, 10), orientation=0.0)
        next_state = AgentState(position=Coordinates(11, 10), orientation=0.0)
        plume_field = np.zeros((128, 128), dtype=np.float32)
        reward = reward_function.compute_reward(prev_state, 0, next_state, plume_field)
        assert reward <= 0.0

    def test_goal_reward_when_at_goal(self, reward_function):
        """Reward should equal goal_reward when at or within goal radius."""
        goal = Coordinates(64, 64)
        prev_state = AgentState(position=goal, orientation=0.0)
        next_state = AgentState(position=goal, orientation=0.0)
        plume_field = np.zeros((128, 128), dtype=np.float32)
        reward = reward_function.compute_reward(prev_state, 0, next_state, plume_field)
        assert reward == pytest.approx(1.0)
