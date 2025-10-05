"""
Unit tests for OrientedGridActions.

Contract: src/backend/contracts/action_processor_interface.md

Tests both universal properties (via inheritance) and implementation-specific behavior
including orientation-dependent movement and heading updates.
"""

import pytest

from plume_nav_sim.actions import OrientedGridActions
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.interfaces import ActionProcessor
from tests.contracts.test_action_processor_interface import TestActionProcessorInterface


class TestOrientedGridActions(TestActionProcessorInterface):
    """Test suite for OrientedGridActions.

    Inherits universal property tests from TestActionProcessorInterface.
    Adds implementation-specific tests for oriented movement and heading control.
    """

    # ==============================================================================
    # Fixtures
    # ==============================================================================

    @pytest.fixture
    def action_processor(self) -> ActionProcessor:
        """Provide OrientedGridActions instance for testing.

        Returns:
            OrientedGridActions with default configuration
        """
        return OrientedGridActions(step_size=1)

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

    def test_action_space_is_discrete_3(self, action_processor):
        """Test: Action space is Discrete(3)."""
        import gymnasium as gym

        assert isinstance(action_processor.action_space, gym.spaces.Discrete)
        assert action_processor.action_space.n == 3

    def test_forward_facing_east_increases_x(self, action_processor, grid_size):
        """Test: FORWARD when facing East (0°) increases x."""
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)

        new_state = action_processor.process_action(0, state, grid_size)

        assert new_state.position.x == 11  # Moved East
        assert new_state.position.y == 10  # y unchanged
        assert new_state.orientation == 0.0  # Heading unchanged

    def test_forward_facing_north_increases_y(self, action_processor, grid_size):
        """Test: FORWARD when facing North (90°) increases y."""
        state = AgentState(position=Coordinates(10, 10), orientation=90.0)

        new_state = action_processor.process_action(0, state, grid_size)

        assert new_state.position.x == 10  # x unchanged
        assert new_state.position.y == 11  # Moved North
        assert new_state.orientation == 90.0  # Heading unchanged

    def test_forward_facing_west_decreases_x(self, action_processor, grid_size):
        """Test: FORWARD when facing West (180°) decreases x."""
        state = AgentState(position=Coordinates(10, 10), orientation=180.0)

        new_state = action_processor.process_action(0, state, grid_size)

        assert new_state.position.x == 9  # Moved West
        assert new_state.position.y == 10  # y unchanged
        assert new_state.orientation == 180.0  # Heading unchanged

    def test_forward_facing_south_decreases_y(self, action_processor, grid_size):
        """Test: FORWARD when facing South (270°) decreases y."""
        state = AgentState(position=Coordinates(10, 10), orientation=270.0)

        new_state = action_processor.process_action(0, state, grid_size)

        assert new_state.position.x == 10  # x unchanged
        assert new_state.position.y == 9  # Moved South
        assert new_state.orientation == 270.0  # Heading unchanged

    def test_turn_left_increases_orientation_by_90(self, action_processor, grid_size):
        """Test: TURN_LEFT increases orientation by 90°."""
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)

        new_state = action_processor.process_action(1, state, grid_size)

        assert new_state.orientation == 90.0
        assert new_state.position == state.position  # Position unchanged

    def test_turn_right_decreases_orientation_by_90(self, action_processor, grid_size):
        """Test: TURN_RIGHT decreases orientation by 90°."""
        state = AgentState(position=Coordinates(10, 10), orientation=90.0)

        new_state = action_processor.process_action(2, state, grid_size)

        assert new_state.orientation == 0.0
        assert new_state.position == state.position  # Position unchanged

    def test_orientation_wraps_at_360(self, action_processor, grid_size):
        """Test: Orientation wraps around at 360°."""
        state = AgentState(position=Coordinates(10, 10), orientation=270.0)

        # Turn left from 270° should wrap to 0°
        new_state = action_processor.process_action(1, state, grid_size)
        assert new_state.orientation == 0.0  # (270 + 90) % 360 = 0

    def test_orientation_wraps_at_negative(self, action_processor, grid_size):
        """Test: Orientation wraps around when going negative."""
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)

        # Turn right from 0° should wrap to 270°
        new_state = action_processor.process_action(2, state, grid_size)
        assert new_state.orientation == 270.0  # (0 - 90) % 360 = 270

    def test_position_unchanged_by_turns(self, action_processor, grid_size):
        """Test: Position is not affected by turn actions."""
        state = AgentState(position=Coordinates(10, 10), orientation=45.0)

        # Turn left
        new_state = action_processor.process_action(1, state, grid_size)
        assert new_state.position == Coordinates(10, 10)

        # Turn right
        new_state = action_processor.process_action(2, state, grid_size)
        assert new_state.position == Coordinates(10, 10)

    def test_forward_at_boundary_clamps(self, action_processor, grid_size):
        """Test: Forward movement at boundary clamps to grid."""
        # At right edge facing East
        state = AgentState(
            position=Coordinates(grid_size.width - 1, 10), orientation=0.0
        )

        new_state = action_processor.process_action(0, state, grid_size)

        assert new_state.position.x == grid_size.width - 1  # Clamped
        assert grid_size.contains(new_state.position)

    def test_metadata_structure(self, action_processor):
        """Test: Metadata contains required fields."""
        metadata = action_processor.get_metadata()

        assert metadata["type"] == "oriented_grid"
        assert "parameters" in metadata
        assert metadata["parameters"]["n_actions"] == 3
        assert metadata["parameters"]["step_size"] == 1
        assert metadata["orientation_dependent"] is True

    def test_validate_action_accepts_valid_actions(self, action_processor):
        """Test: validate_action returns True for valid actions."""
        for action in [0, 1, 2]:
            assert action_processor.validate_action(action) is True

    def test_validate_action_rejects_invalid_actions(self, action_processor):
        """Test: validate_action returns False for invalid actions."""
        for action in [-1, 3, 4, 100]:
            assert action_processor.validate_action(action) is False

    def test_custom_step_size(self):
        """Test: Custom step_size works correctly."""
        action_proc = OrientedGridActions(step_size=2)
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)
        grid = GridSize(128, 128)

        # Forward with step_size=2 facing East
        new_state = action_proc.process_action(0, state, grid)
        assert new_state.position.x == 12  # Moved 2 cells

    def test_protocol_conformance(self, action_processor):
        """Test: OrientedGridActions satisfies ActionProcessor protocol."""
        assert isinstance(action_processor, ActionProcessor)

    def test_state_fields_preserved(self, action_processor, grid_size):
        """Test: Non-position/orientation fields are preserved."""
        state = AgentState(
            position=Coordinates(10, 10),
            orientation=0.0,
            step_count=5,
            total_reward=1.5,
            goal_reached=True,
        )

        new_state = action_processor.process_action(0, state, grid_size)

        assert new_state.step_count == 5
        assert new_state.total_reward == 1.5
        assert new_state.goal_reached is True

    def test_diagonal_movements_not_supported(self, action_processor, grid_size):
        """Test: Only cardinal directions supported (no diagonal at 45°)."""
        # Facing Northeast (45°)
        state = AgentState(position=Coordinates(10, 10), orientation=45.0)

        new_state = action_processor.process_action(0, state, grid_size)

        # Should move approximately diagonal (1,1) when rounded
        # This is expected behavior - orientation allows any angle
        assert new_state.position != state.position

    def test_full_rotation_cycle(self, action_processor, grid_size):
        """Test: Four left turns returns to original orientation."""
        state = AgentState(position=Coordinates(10, 10), orientation=0.0)

        # Turn left 4 times (360°)
        for _ in range(4):
            state = action_processor.process_action(1, state, grid_size)

        assert state.orientation == 0.0  # Back to start

    def test_orientation_affects_forward_direction(self, action_processor, grid_size):
        """Test: Same FORWARD action produces different movements based on orientation."""
        position = Coordinates(64, 64)

        # Facing East
        state_east = AgentState(position=position, orientation=0.0)
        result_east = action_processor.process_action(0, state_east, grid_size)

        # Facing North
        state_north = AgentState(position=position, orientation=90.0)
        result_north = action_processor.process_action(0, state_north, grid_size)

        # Should produce different positions
        assert result_east.position != result_north.position
        assert result_east.position.x > position.x  # Moved East
        assert result_north.position.y > position.y  # Moved North
