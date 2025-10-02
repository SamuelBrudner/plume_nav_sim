"""
Unit tests for DiscreteGridActions.

Contract: src/backend/contracts/action_processor_interface.md

Tests both universal properties (via inheritance) and implementation-specific behavior.
"""

import pytest

from plume_nav_sim.actions import DiscreteGridActions
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.interfaces import ActionProcessor
from tests.contracts.test_action_processor_interface import TestActionProcessorInterface


class TestDiscreteGridActions(TestActionProcessorInterface):
    """Test suite for DiscreteGridActions.

    Inherits universal property tests from TestActionProcessorInterface.
    Adds implementation-specific tests for discrete grid movement.
    """

    # ==============================================================================
    # Fixtures
    # ==============================================================================

    @pytest.fixture
    def action_processor(self) -> ActionProcessor:
        """Provide DiscreteGridActions instance for testing.

        Returns:
            DiscreteGridActions with default configuration
        """
        return DiscreteGridActions(step_size=1)

    @pytest.fixture
    def grid_size(self) -> GridSize:
        """Default grid size for tests.

        Returns:
            GridSize(128, 128)
        """
        return GridSize(width=128, height=128)

    # ==============================================================================
    # Implementation-Specific Tests
    # ==============================================================================

    def test_action_space_is_discrete_4(self, action_processor):
        """Test: Action space is Discrete(4)."""
        import gymnasium as gym

        assert isinstance(action_processor.action_space, gym.spaces.Discrete)
        assert action_processor.action_space.n == 4

    def test_up_action_increases_y(self, action_processor, grid_size):
        """Test: UP action (0) increases y coordinate."""
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)

        new_state = action_processor.process_action(0, state, grid_size)

        assert new_state.position.x == 10  # x unchanged
        assert new_state.position.y == 11  # y increased

    def test_right_action_increases_x(self, action_processor, grid_size):
        """Test: RIGHT action (1) increases x coordinate."""
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)

        new_state = action_processor.process_action(1, state, grid_size)

        assert new_state.position.x == 11  # x increased
        assert new_state.position.y == 10  # y unchanged

    def test_down_action_decreases_y(self, action_processor, grid_size):
        """Test: DOWN action (2) decreases y coordinate."""
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)

        new_state = action_processor.process_action(2, state, grid_size)

        assert new_state.position.x == 10  # x unchanged
        assert new_state.position.y == 9  # y decreased

    def test_left_action_decreases_x(self, action_processor, grid_size):
        """Test: LEFT action (3) decreases x coordinate."""
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)

        new_state = action_processor.process_action(3, state, grid_size)

        assert new_state.position.x == 9  # x decreased
        assert new_state.position.y == 10  # y unchanged

    def test_orientation_unchanged_by_movement(self, action_processor, grid_size):
        """Test: Orientation is preserved during movement."""
        state = AgentState(position=Coordinates(10, 10), orientation=45.0)

        for action in [0, 1, 2, 3]:
            new_state = action_processor.process_action(action, state, grid_size)
            assert new_state.orientation == 45.0, f"Action {action} changed orientation"

    def test_clamping_at_top_edge(self, action_processor, grid_size):
        """Test: UP action at top edge stays in bounds."""
        state = AgentState(
            position=Coordinates(10, grid_size.height - 1), orientation=0.0
        )

        new_state = action_processor.process_action(0, state, grid_size)

        assert new_state.position.y == grid_size.height - 1  # Clamped
        assert grid_size.contains(new_state.position)

    def test_clamping_at_right_edge(self, action_processor, grid_size):
        """Test: RIGHT action at right edge stays in bounds."""
        state = AgentState(
            position=Coordinates(grid_size.width - 1, 10), orientation=0.0
        )

        new_state = action_processor.process_action(1, state, grid_size)

        assert new_state.position.x == grid_size.width - 1  # Clamped
        assert grid_size.contains(new_state.position)

    def test_clamping_at_bottom_edge(self, action_processor, grid_size):
        """Test: DOWN action at bottom edge stays in bounds."""
        state = AgentState(position=Coordinates(10, 0), orientation=0.0)

        new_state = action_processor.process_action(2, state, grid_size)

        assert new_state.position.y == 0  # Clamped
        assert grid_size.contains(new_state.position)

    def test_clamping_at_left_edge(self, action_processor, grid_size):
        """Test: LEFT action at left edge stays in bounds."""
        state = AgentState(position=Coordinates(0, 10), orientation=0.0)

        new_state = action_processor.process_action(3, state, grid_size)

        assert new_state.position.x == 0  # Clamped
        assert grid_size.contains(new_state.position)

    def test_metadata_structure(self, action_processor):
        """Test: Metadata contains required fields."""
        metadata = action_processor.get_metadata()

        assert metadata["type"] == "discrete_grid"
        assert "parameters" in metadata
        assert metadata["parameters"]["n_actions"] == 4
        assert metadata["parameters"]["step_size"] == 1
        assert metadata["orientation_dependent"] is False

    def test_validate_action_accepts_valid_actions(self, action_processor):
        """Test: validate_action returns True for valid actions."""
        for action in [0, 1, 2, 3]:
            assert action_processor.validate_action(action) is True

    def test_validate_action_rejects_invalid_actions(self, action_processor):
        """Test: validate_action returns False for invalid actions."""
        for action in [-1, 4, 5, 100]:
            assert action_processor.validate_action(action) is False

    def test_custom_step_size(self):
        """Test: Custom step_size works correctly."""
        action_proc = DiscreteGridActions(step_size=2)
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)
        grid = GridSize(128, 128)

        # UP with step_size=2
        new_state = action_proc.process_action(0, state, grid)
        assert new_state.position.y == 12  # Moved 2 cells

        # RIGHT with step_size=2
        new_state = action_proc.process_action(1, state, grid)
        assert new_state.position.x == 12  # Moved 2 cells

    def test_protocol_conformance(self, action_processor):
        """Test: DiscreteGridActions satisfies ActionProcessor protocol."""
        assert isinstance(action_processor, ActionProcessor)

    def test_state_fields_preserved(self, action_processor, grid_size):
        """Test: Non-position/orientation fields are preserved."""
        state = AgentState(
            position=Coordinates(10, 10),
            orientation=45.0,
            step_count=5,
            total_reward=1.5,
            goal_reached=True,
        )

        new_state = action_processor.process_action(1, state, grid_size)

        assert new_state.step_count == 5
        assert new_state.total_reward == 1.5
        assert new_state.goal_reached is True
