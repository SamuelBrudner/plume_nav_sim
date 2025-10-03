"""
Universal test suite for RewardFunction protocol implementations.

Contract: src/backend/contracts/reward_function_interface.md

All reward function implementations MUST pass these tests.
Concrete test classes should inherit from TestRewardFunctionInterface
and provide a reward_function fixture.

Usage:
    class TestSparseGoalReward(TestRewardFunctionInterface):
        @pytest.fixture
        def reward_function(self):
            return SparseGoalReward(goal_radius=1.0, source_location=Coordinates(64, 64))
"""

import copy

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.interfaces import RewardFunction
from tests.strategies import (
    agent_state_strategy,
    concentration_field_strategy,
    discrete_action_strategy,
    grid_size_strategy,
)


class TestRewardFunctionInterface:
    """Universal test suite for RewardFunction implementations.

    Contract: reward_function_interface.md

    All implementations must pass these tests to be considered valid.
    Concrete test classes should inherit this and provide reward_function fixture.
    """

    __test__ = False

    # ==============================================================================
    # Fixtures (Override in concrete test classes)
    # ==============================================================================

    @pytest.fixture
    def reward_function(self) -> RewardFunction:
        """Override this fixture to provide the reward function to test.

        Returns:
            RewardFunction implementation to test

        Raises:
            NotImplementedError: If not overridden in subclass
        """
        raise NotImplementedError(
            "Concrete test classes must override reward_function fixture"
        )

    @pytest.fixture
    def grid_size(self) -> GridSize:
        """Default grid size for tests.

        Returns:
            GridSize(128, 128)
        """
        return GridSize(width=128, height=128)

    # ==============================================================================
    # Property 1: Determinism (UNIVERSAL)
    # ==============================================================================

    @given(
        prev_state=agent_state_strategy(),
        action=discrete_action_strategy(n_actions=4),
        next_state=agent_state_strategy(),
    )
    @settings(
        deadline=None,
        max_examples=50,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.differing_executors,
        ],
    )
    def test_determinism(self, reward_function, prev_state, action, next_state):
        """Property: Same inputs always produce same reward.

        Contract: reward_function_interface.md - Property 1: Determinism

        ∀ (s, a, s'): compute_reward(s, a, s') = compute_reward(s, a, s')
        """
        # Create a simple concentration field
        grid = GridSize(width=128, height=128)
        plume_field = np.zeros((grid.height, grid.width), dtype=np.float32)

        # Call twice with identical inputs
        reward1 = reward_function.compute_reward(
            prev_state, action, next_state, plume_field
        )
        reward2 = reward_function.compute_reward(
            prev_state, action, next_state, plume_field
        )

        # Must be identical
        assert reward1 == reward2, "Reward function is not deterministic"

        # Also check type consistency
        assert type(reward1) == type(reward2)

    # ==============================================================================
    # Property 2: Purity (UNIVERSAL)
    # ==============================================================================

    def test_purity_no_state_mutation(self, reward_function):
        """Property: Reward computation does not mutate inputs.

        Contract: reward_function_interface.md - Property 2: Purity

        No modification of prev_state, action, next_state, or plume_field.
        """
        from plume_nav_sim.core.state import AgentState

        # Create test inputs
        prev_state = AgentState(position=Coordinates(10, 10), orientation=0.0)
        next_state = AgentState(position=Coordinates(11, 10), orientation=0.0)
        action = 0
        plume_field = np.array([[0.5, 0.3], [0.2, 0.1]], dtype=np.float32)

        # Deep copy inputs
        prev_copy = copy.deepcopy(prev_state)
        next_copy = copy.deepcopy(next_state)
        field_copy = plume_field.copy()

        # Compute reward
        _ = reward_function.compute_reward(prev_state, action, next_state, plume_field)

        # Verify no mutations
        assert prev_state.position == prev_copy.position
        assert prev_state.orientation == prev_copy.orientation
        assert prev_state.step_count == prev_copy.step_count

        assert next_state.position == next_copy.position
        assert next_state.orientation == next_copy.orientation

        assert np.array_equal(plume_field, field_copy), "plume_field was mutated"

    # ==============================================================================
    # Property 3: Finiteness (UNIVERSAL)
    # ==============================================================================

    @given(
        prev_state=agent_state_strategy(),
        action=discrete_action_strategy(n_actions=4),
        next_state=agent_state_strategy(),
        grid=grid_size_strategy(
            min_width=16, max_width=64, min_height=16, max_height=64
        ),
    )
    @settings(
        deadline=None,
        max_examples=50,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.differing_executors,
        ],
    )
    def test_finiteness(self, reward_function, prev_state, action, next_state, grid):
        """Property: Reward is always finite (not NaN, not inf).

        Contract: reward_function_interface.md - Property 3: Finiteness

        ∀ inputs: isfinite(compute_reward(inputs))
        """
        # Create concentration field
        plume_field = np.random.rand(grid.height, grid.width).astype(np.float32)

        reward = reward_function.compute_reward(
            prev_state, action, next_state, plume_field
        )

        assert np.isfinite(reward), f"Reward is not finite: {reward}"
        assert not np.isnan(reward), f"Reward is NaN: {reward}"
        assert not np.isinf(reward), f"Reward is infinite: {reward}"

    # ==============================================================================
    # Return Type Validation
    # ==============================================================================

    def test_returns_numeric_type(self, reward_function):
        """Test: Reward is numeric (float, int, or numpy numeric).

        Contract: reward_function_interface.md - Postcondition C3
        """
        from plume_nav_sim.core.state import AgentState

        prev_state = AgentState(position=Coordinates(10, 10))
        next_state = AgentState(position=Coordinates(11, 10))
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        reward = reward_function.compute_reward(
            prev_state, action, next_state, plume_field
        )

        # Must be numeric type
        assert isinstance(
            reward, (int, float, np.integer, np.floating)
        ), f"Reward has invalid type: {type(reward)}"

    def test_returns_scalar_not_array(self, reward_function):
        """Test: Reward is scalar, not array.

        Contract: reward_function_interface.md - returns float
        """
        from plume_nav_sim.core.state import AgentState

        prev_state = AgentState(position=Coordinates(10, 10))
        next_state = AgentState(position=Coordinates(11, 10))
        action = 0
        plume_field = np.zeros((32, 32), dtype=np.float32)

        reward = reward_function.compute_reward(
            prev_state, action, next_state, plume_field
        )

        # Must be scalar
        assert (
            np.ndim(reward) == 0
        ), f"Reward is not scalar, has shape: {np.shape(reward)}"

    # ==============================================================================
    # Metadata Tests
    # ==============================================================================

    def test_has_get_metadata_method(self, reward_function):
        """Test: Reward function has get_metadata() method.

        Contract: reward_function_interface.md - RewardFunction protocol
        """
        assert hasattr(reward_function, "get_metadata")
        assert callable(reward_function.get_metadata)

    def test_metadata_has_required_keys(self, reward_function):
        """Test: Metadata contains at least 'type' key.

        Contract: reward_function_interface.md - get_metadata() postcondition
        """
        metadata = reward_function.get_metadata()

        assert isinstance(metadata, dict), "Metadata must be a dictionary"
        assert "type" in metadata, "Metadata must contain 'type' key"
        assert isinstance(metadata["type"], str), "'type' must be a string"

    def test_metadata_is_json_serializable(self, reward_function):
        """Test: Metadata can be JSON serialized.

        Contract: reward_function_interface.md - get_metadata() postcondition C2
        """
        import json

        metadata = reward_function.get_metadata()

        try:
            json.dumps(metadata)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Metadata is not JSON-serializable: {e}")

    # ==============================================================================
    # Protocol Conformance
    # ==============================================================================

    def test_conforms_to_reward_function_protocol(self, reward_function):
        """Test: Implementation satisfies RewardFunction protocol.

        Contract: reward_function_interface.md - RewardFunction protocol
        """
        assert isinstance(
            reward_function, RewardFunction
        ), f"{type(reward_function).__name__} does not satisfy RewardFunction protocol"

    def test_has_compute_reward_method(self, reward_function):
        """Test: Has compute_reward() method with correct signature."""
        assert hasattr(reward_function, "compute_reward")
        assert callable(reward_function.compute_reward)

        # Check method signature (basic check)
        import inspect

        sig = inspect.signature(reward_function.compute_reward)
        params = list(sig.parameters.keys())

        # Should have 4 parameters: prev_state, action, next_state, plume_field
        assert (
            len(params) == 4
        ), f"compute_reward should have 4 parameters, has {len(params)}"


# ==============================================================================
# Helper Functions for Concrete Tests
# ==============================================================================


def create_simple_test_scenario():
    """Helper to create a simple test scenario for manual tests.

    Returns:
        Tuple of (prev_state, action, next_state, plume_field)
    """
    from plume_nav_sim.core.state import AgentState

    prev_state = AgentState(position=Coordinates(10, 10), orientation=0.0)
    next_state = AgentState(position=Coordinates(11, 10), orientation=0.0)
    action = 0
    plume_field = np.zeros((32, 32), dtype=np.float32)

    return prev_state, action, next_state, plume_field
