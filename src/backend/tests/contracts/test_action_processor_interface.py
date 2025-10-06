"""
Universal test suite for ActionProcessor protocol implementations.

Contract: src/backend/contracts/action_processor_interface.md

All action processor implementations MUST pass these tests.
Concrete test classes should inherit from TestActionProcessorInterface
and provide an action_processor fixture.

Usage:
    class TestDiscreteGridActions(TestActionProcessorInterface):
        @pytest.fixture
        def action_processor(self):
            return DiscreteGridActions(step_size=1)
"""

import copy

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import gymnasium as gym
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.interfaces import ActionProcessor
from tests.strategies import (
    agent_state_strategy,
    discrete_action_strategy,
    grid_size_strategy,
    valid_position_for_grid_strategy,
)


class TestActionProcessorInterface:
    """Universal test suite for ActionProcessor implementations.

    Contract: action_processor_interface.md

    All implementations must pass these tests to be considered valid.
    Concrete test classes should inherit this and provide action_processor fixture.
    """

    # Prevent pytest from collecting this base class directly. Concrete subclasses
    # should inherit from this and provide the fixtures.
    __test__ = False

    # ==============================================================================
    # Fixtures (Override in concrete test classes)
    # ==============================================================================

    @pytest.fixture
    def action_processor(self) -> ActionProcessor:
        """Override this fixture to provide the action processor to test.

        Returns:
            ActionProcessor implementation to test

        Raises:
            NotImplementedError: If not overridden in subclass
        """
        raise NotImplementedError(
            "Concrete test classes must override action_processor fixture"
        )

    @pytest.fixture
    def grid_size(self) -> GridSize:
        """Default grid size for tests.

        Returns:
            GridSize(128, 128)
        """
        return GridSize(width=128, height=128)

    # ==============================================================================
    # Property 1: Boundary Safety (UNIVERSAL)
    # ==============================================================================

    @given(
        grid=grid_size_strategy(
            min_width=10, max_width=64, min_height=10, max_height=64
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
    def test_boundary_safety(self, action_processor, grid):
        """Property: Result position always within bounds.

        Contract: action_processor_interface.md - Property 1: Boundary Safety

        ∀ action, state, grid:
          grid.contains(state.position) ⇒ grid.contains(process_action(...).position)
        """
        # Generate valid position within grid
        position = Coordinates(
            x=np.random.randint(0, grid.width),
            y=np.random.randint(0, grid.height),
        )
        current_state = AgentState(position=position, orientation=0.0)

        # Sample a valid action from the action space
        action = action_processor.action_space.sample()

        # Process action
        new_state = action_processor.process_action(action, current_state, grid)

        # Result must be within bounds
        assert (
            0 <= new_state.position.x < grid.width
        ), f"x={new_state.position.x} out of bounds [0, {grid.width})"
        assert (
            0 <= new_state.position.y < grid.height
        ), f"y={new_state.position.y} out of bounds [0, {grid.height})"
        assert grid.contains(
            new_state.position
        ), f"Position {new_state.position} not in grid {grid}"

    @given(
        grid=grid_size_strategy(
            min_width=10, max_width=32, min_height=10, max_height=32
        ),
    )
    @settings(
        deadline=None,
        max_examples=30,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.differing_executors,
        ],
    )
    def test_corner_positions_stay_in_bounds(self, action_processor, grid):
        """Property: Actions from corners stay in bounds.

        Contract: action_processor_interface.md - Boundary Safety

        Tests all four corners to ensure boundary clamping works.
        """
        corners = [
            Coordinates(0, 0),  # Top-left
            Coordinates(grid.width - 1, 0),  # Top-right
            Coordinates(0, grid.height - 1),  # Bottom-left
            Coordinates(grid.width - 1, grid.height - 1),  # Bottom-right
        ]

        for corner in corners:
            current_state = AgentState(position=corner, orientation=0.0)

            # Try all valid actions
            for _ in range(10):  # Sample multiple actions
                action = action_processor.action_space.sample()

                new_state = action_processor.process_action(action, current_state, grid)

                assert grid.contains(new_state.position), (
                    f"Action {action} from corner {corner} produced out-of-bounds "
                    f"position {new_state.position} in grid {grid}"
                )

    # ==============================================================================
    # Property 2: Determinism (UNIVERSAL)
    # ==============================================================================

    @given(
        grid=grid_size_strategy(
            min_width=16, max_width=64, min_height=16, max_height=64
        ),
    )
    @settings(
        deadline=None,
        max_examples=30,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.differing_executors,
        ],
    )
    def test_determinism(self, action_processor, grid):
        """Property: Same inputs produce same output.

        Contract: action_processor_interface.md - Property 2: Determinism

        ∀ action, state, grid: process_action(action, state, grid) is deterministic
        """
        # Create test state
        position = Coordinates(x=grid.width // 2, y=grid.height // 2)
        current_state = AgentState(position=position, orientation=45.0)

        # Sample action
        action = action_processor.action_space.sample()

        # Process twice
        result1 = action_processor.process_action(action, current_state, grid)
        result2 = action_processor.process_action(action, current_state, grid)

        # Must be identical
        assert result1.position == result2.position, "Position is not deterministic"
        assert (
            result1.orientation == result2.orientation
        ), "Orientation is not deterministic"

    # ==============================================================================
    # Property 3: Purity (UNIVERSAL)
    # ==============================================================================

    def test_purity_no_state_mutation(self, action_processor, grid_size):
        """Property: process_action does not mutate current_state.

        Contract: action_processor_interface.md - Property 3: Purity

        No modification of action, current_state, or grid_size.
        """
        # Create test state
        current_state = AgentState(
            position=Coordinates(10, 10),
            orientation=90.0,
            step_count=5,
        )

        # Deep copy for comparison
        state_copy = copy.deepcopy(current_state)

        # Sample and process action
        action = action_processor.action_space.sample()
        new_state = action_processor.process_action(action, current_state, grid_size)

        # Verify no mutations to current_state
        assert current_state.position == state_copy.position, "Position was mutated"
        assert (
            current_state.orientation == state_copy.orientation
        ), "Orientation was mutated"
        assert (
            current_state.step_count == state_copy.step_count
        ), "Step count was mutated"

        # new_state must be a different instance
        assert (
            new_state is not current_state
        ), "Returned same instance instead of new one"

    def test_returns_new_instance(self, action_processor, grid_size):
        """Test: process_action returns new AgentState, not mutated one.

        Contract: action_processor_interface.md - Postcondition C4
        """
        current_state = AgentState(position=Coordinates(10, 10))
        action = action_processor.action_space.sample()

        new_state = action_processor.process_action(action, current_state, grid_size)

        assert new_state is not current_state, "Must return new AgentState instance"
        assert isinstance(new_state, AgentState), "Must return AgentState instance"

    # ==============================================================================
    # Action Space Tests
    # ==============================================================================

    def test_has_action_space_property(self, action_processor):
        """Test: action_processor has action_space property.

        Contract: action_processor_interface.md - ActionProcessor protocol
        """
        assert hasattr(action_processor, "action_space")
        assert isinstance(action_processor.action_space, gym.Space)

    def test_action_space_is_immutable(self, action_processor):
        """Test: action_space returns same instance.

        Contract: action_processor_interface.md - Postcondition C2
        """
        space1 = action_processor.action_space
        space2 = action_processor.action_space

        assert space1 is space2, "action_space should return same instance"

    def test_action_space_is_valid_gym_space(self, action_processor):
        """Test: action_space is a valid Gymnasium Space."""
        space = action_processor.action_space

        assert isinstance(
            space, gym.Space
        ), f"action_space must be gym.Space, got {type(space)}"

        # Should be able to sample from it
        try:
            sample = space.sample()
            assert space.contains(sample), "Space sample not contained in space"
        except Exception as e:
            pytest.fail(f"action_space.sample() failed: {e}")

    # ==============================================================================
    # Process Action Tests
    # ==============================================================================

    def test_has_process_action_method(self, action_processor):
        """Test: action_processor has process_action() method.

        Contract: action_processor_interface.md - ActionProcessor protocol
        """
        assert hasattr(action_processor, "process_action")
        assert callable(action_processor.process_action)

    def test_process_action_returns_agent_state(self, action_processor, grid_size):
        """Test: process_action returns AgentState.

        Contract: action_processor_interface.md - Postcondition C1
        """
        current_state = AgentState(position=Coordinates(10, 10))
        action = action_processor.action_space.sample()

        result = action_processor.process_action(action, current_state, grid_size)

        assert isinstance(
            result, AgentState
        ), f"process_action must return AgentState, got {type(result)}"

    def test_process_action_accepts_valid_actions(self, action_processor, grid_size):
        """Test: process_action accepts all valid actions from action_space."""
        current_state = AgentState(position=Coordinates(10, 10))

        # Sample multiple actions and ensure they all work
        for _ in range(10):
            action = action_processor.action_space.sample()

            try:
                result = action_processor.process_action(
                    action, current_state, grid_size
                )
                assert result is not None
            except Exception as e:
                pytest.fail(
                    f"process_action raised error for valid action {action}: {e}"
                )

    # ==============================================================================
    # Validate Action Tests
    # ==============================================================================

    def test_has_validate_action_method(self, action_processor):
        """Test: action_processor has validate_action() method.

        Contract: action_processor_interface.md - ActionProcessor protocol
        """
        assert hasattr(action_processor, "validate_action")
        assert callable(action_processor.validate_action)

    def test_validate_action_returns_bool(self, action_processor):
        """Test: validate_action returns boolean."""
        action = action_processor.action_space.sample()
        result = action_processor.validate_action(action)

        assert isinstance(
            result, (bool, np.bool_)
        ), f"validate_action must return bool, got {type(result)}"

    def test_validate_action_accepts_valid_actions(self, action_processor):
        """Test: validate_action returns True for valid actions."""
        # Sample multiple actions from action space
        for _ in range(10):
            action = action_processor.action_space.sample()
            assert action_processor.validate_action(action), (
                f"validate_action returned False for valid action {action} "
                f"from action_space"
            )

    # ==============================================================================
    # Metadata Tests
    # ==============================================================================

    def test_has_get_metadata_method(self, action_processor):
        """Test: action_processor has get_metadata() method.

        Contract: action_processor_interface.md - ActionProcessor protocol
        """
        assert hasattr(action_processor, "get_metadata")
        assert callable(action_processor.get_metadata)

    def test_metadata_has_required_keys(self, action_processor):
        """Test: Metadata contains required keys.

        Contract: action_processor_interface.md - get_metadata() return value
        """
        metadata = action_processor.get_metadata()

        assert isinstance(metadata, dict), "Metadata must be a dictionary"
        assert "type" in metadata, "Metadata must contain 'type' key"
        assert isinstance(metadata["type"], str), "'type' must be a string"

    # ==============================================================================
    # Protocol Conformance
    # ==============================================================================

    def test_conforms_to_action_processor_protocol(self, action_processor):
        """Test: Implementation satisfies ActionProcessor protocol.

        Contract: action_processor_interface.md - ActionProcessor protocol
        """
        assert isinstance(
            action_processor, ActionProcessor
        ), f"{type(action_processor).__name__} does not satisfy ActionProcessor protocol"


# ==============================================================================
# Helper Functions for Concrete Tests
# ==============================================================================


def create_simple_test_state(
    grid_size: GridSize | None = None,
) -> tuple[AgentState, GridSize]:
    """Helper to create a simple test state for manual tests.

    Args:
        grid_size: Optional grid size (default: 32x32)

    Returns:
        Tuple of (agent_state, grid_size)
    """
    if grid_size is None:
        grid_size = GridSize(width=32, height=32)

    agent_state = AgentState(
        position=Coordinates(x=grid_size.width // 2, y=grid_size.height // 2),
        orientation=0.0,
    )

    return agent_state, grid_size
