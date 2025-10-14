"""Property-Based Tests using Hypothesis.

Tests that use random inputs to verify properties hold for ALL valid inputs.
More powerful than example-based tests - finds edge cases automatically.

Reference: ../../CONTRACTS.md and ../../SEMANTIC_MODEL.md
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from plume_nav_sim.core.geometry import (
    Coordinates,
    GridSize,
    calculate_euclidean_distance,
)
from plume_nav_sim.utils.exceptions import ValidationError
from plume_nav_sim.utils.seeding import create_seeded_rng, validate_seed


class TestSeedValidationProperties:
    """Property-based tests for seed validation."""

    @given(st.integers(min_value=0, max_value=2**31 - 1))
    def test_valid_seeds_validate_to_themselves(self, seed):
        """Property: Any valid seed validates to itself (identity)."""
        is_valid, validated, error = validate_seed(seed)

        assert is_valid is True
        assert validated == seed
        assert error == ""

    @given(st.integers(max_value=-1))
    def test_negative_seeds_always_invalid(self, seed):
        """Property: All negative seeds are invalid."""
        is_valid, validated, error = validate_seed(seed)

        assert is_valid is False
        assert validated is None
        assert "non-negative" in error.lower() or "negative" in error.lower()

    @given(st.integers(min_value=2**32))
    def test_too_large_seeds_invalid(self, seed):
        """Property: Seeds beyond 32-bit range are invalid."""
        is_valid, validated, error = validate_seed(seed)

        assert is_valid is False
        assert validated is None

    @given(st.integers(min_value=0, max_value=2**31 - 1))
    def test_same_seed_produces_identical_rngs(self, seed):
        """Property: Same seed always produces identical RNG sequences."""
        rng1, seed1 = create_seeded_rng(seed)
        rng2, seed2 = create_seeded_rng(seed)

        # Seeds should match
        assert seed1 == seed2 == seed

        # Sequences must be identical
        seq1 = rng1.random(50)
        seq2 = rng2.random(50)

        assert np.array_equal(seq1, seq2)


class TestCoordinateProperties:
    """Property-based tests for coordinate operations."""

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
    )
    def test_coordinates_accept_non_negative_integers(self, x, y):
        """Property: Coordinates accept non-negative integer values."""
        # Should not raise
        coord = Coordinates(x=x, y=y)
        assert coord.x == x
        assert coord.y == y

    @given(
        st.integers(min_value=-100, max_value=-1),
        st.integers(min_value=0, max_value=100),
    )
    def test_coordinates_accept_negative_x(self, x, y):
        """Property: Coordinates accept negative x values per contract.

        Contract: core_types.md - Coordinates allow all integers
        Use case: Off-grid positions for boundary logic
        """
        coord = Coordinates(x=x, y=y)
        assert coord.x == x
        assert coord.y == y

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=-100, max_value=-1),
    )
    def test_coordinates_accept_negative_y(self, x, y):
        """Property: Coordinates accept negative y values per contract.

        Contract: core_types.md - Coordinates allow all integers
        Use case: Off-grid positions for boundary logic
        """
        coord = Coordinates(x=x, y=y)
        assert coord.x == x
        assert coord.y == y

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
    )
    def test_is_within_bounds_correct(self, x, y, width, height):
        """Property: is_within_bounds matches manual check."""
        coord = Coordinates(x=x, y=y)
        grid = GridSize(width=width, height=height)

        expected = (0 <= x < width) and (0 <= y < height)
        actual = coord.is_within_bounds(grid)

        assert actual == expected


class TestDistanceProperties:
    """Property-based tests for distance calculations."""

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
    )
    def test_distance_always_non_negative(self, x1, y1, x2, y2):
        """Property: Distance is always >= 0."""
        coord1 = Coordinates(x=x1, y=y1)
        coord2 = Coordinates(x=x2, y=y2)

        dist = calculate_euclidean_distance(coord1, coord2)

        assert dist >= 0.0

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
    )
    def test_distance_symmetric(self, x1, y1, x2, y2):
        """Property: distance(A, B) = distance(B, A)."""
        coord1 = Coordinates(x=x1, y=y1)
        coord2 = Coordinates(x=x2, y=y2)

        dist_ab = calculate_euclidean_distance(coord1, coord2)
        dist_ba = calculate_euclidean_distance(coord2, coord1)

        assert np.isclose(dist_ab, dist_ba, rtol=1e-9)

    @given(
        st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100)
    )
    def test_distance_to_self_is_zero(self, x, y):
        """Property: distance(A, A) = 0."""
        coord = Coordinates(x=x, y=y)

        dist = calculate_euclidean_distance(coord, coord)

        assert np.isclose(dist, 0.0, atol=1e-9)

    @given(
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50),
        st.integers(min_value=0, max_value=50),
    )
    def test_triangle_inequality(self, x1, y1, x2, y2, x3, y3):
        """Property: distance(A,C) <= distance(A,B) + distance(B,C)."""
        coord_a = Coordinates(x=x1, y=y1)
        coord_b = Coordinates(x=x2, y=y2)
        coord_c = Coordinates(x=x3, y=y3)

        dist_ac = calculate_euclidean_distance(coord_a, coord_c)
        dist_ab = calculate_euclidean_distance(coord_a, coord_b)
        dist_bc = calculate_euclidean_distance(coord_b, coord_c)

        # Triangle inequality with small epsilon for float errors
        assert dist_ac <= dist_ab + dist_bc + 1e-9


class TestGridSizeProperties:
    """Property-based tests for GridSize."""

    @given(
        st.integers(min_value=1, max_value=1024),
        st.integers(min_value=1, max_value=1024),
    )
    def test_grid_size_accepts_positive_dimensions(self, width, height):
        """Property: GridSize accepts any positive dimensions <= 1024."""
        grid = GridSize(width=width, height=height)

        assert grid.width == width
        assert grid.height == height

    @given(
        st.integers(min_value=-100, max_value=0),
        st.integers(min_value=1, max_value=100),
    )
    def test_grid_size_rejects_non_positive_width(self, width, height):
        """Property: GridSize rejects width <= 0."""
        with pytest.raises((ValidationError, ValueError)):
            GridSize(width=width, height=height)

    @given(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=-100, max_value=0),
    )
    def test_grid_size_rejects_non_positive_height(self, width, height):
        """Property: GridSize rejects height <= 0."""
        with pytest.raises((ValidationError, ValueError)):
            GridSize(width=width, height=height)

    @given(
        st.integers(min_value=1, max_value=1024),
        st.integers(min_value=1, max_value=1024),
    )
    def test_total_cells_equals_width_times_height(self, width, height):
        """Property: total_cells() = width * height."""
        grid = GridSize(width=width, height=height)

        assert grid.total_cells() == width * height

    @given(
        st.integers(min_value=1, max_value=1024),
        st.integers(min_value=1, max_value=1024),
    )
    def test_center_is_within_bounds(self, width, height):
        """Property: center() is always within grid bounds."""
        grid = GridSize(width=width, height=height)
        center = grid.center()

        assert 0 <= center.x < width
        assert 0 <= center.y < height

    @given(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
    )
    def test_to_tuple_round_trip(self, width, height):
        """Property: GridSize -> tuple -> comparison works."""
        grid = GridSize(width=width, height=height)
        tup = grid.to_tuple()

        assert tup == (width, height)
        assert isinstance(tup, tuple)
        assert len(tup) == 2


class TestRewardCalculationProperties:
    """Property-based tests for reward calculation."""

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
        st.floats(min_value=0.0, max_value=50.0),
    )
    def test_goal_reached_consistent_with_distance(self, x1, y1, x2, y2, goal_radius):
        """Property: goal_reached flag consistent with distance <= goal_radius."""
        from plume_nav_sim.core.reward_calculator import (
            RewardCalculator,
            RewardCalculatorConfig,
        )

        config = RewardCalculatorConfig(
            goal_radius=goal_radius,
            reward_goal_reached=1.0,
            reward_default=0.0,
        )
        calculator = RewardCalculator(config=config)

        agent_pos = Coordinates(x=x1, y=y1)
        goal_pos = Coordinates(x=x2, y=y2)

        result = calculator.calculate_reward(
            agent_position=agent_pos,
            source_location=goal_pos,
        )

        distance = calculate_euclidean_distance(agent_pos, goal_pos)
        expected_goal_reached = distance <= goal_radius

        assert result.goal_reached == expected_goal_reached

    @given(
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=0.0, max_value=10.0),
    )
    def test_reward_values_within_bounds(self, reward_goal, reward_default):
        """Property: Calculated rewards are always within [min, max] of config."""
        from plume_nav_sim.core.reward_calculator import (
            RewardCalculator,
            RewardCalculatorConfig,
        )

        # Skip if rewards are too close (validation requires difference for learning signal)
        if abs(reward_goal - reward_default) < 1e-6:
            return

        # Skip non-finite values
        if not (np.isfinite(reward_goal) and np.isfinite(reward_default)):
            return

        config = RewardCalculatorConfig(
            goal_radius=5.0,
            reward_goal_reached=reward_goal,
            reward_default=reward_default,
        )
        calculator = RewardCalculator(config=config)

        # Test various positions
        for x1, y1, x2, y2 in [(0, 0, 0, 0), (0, 0, 10, 10), (5, 5, 5, 5)]:
            agent_pos = Coordinates(x=x1, y=y1)
            goal_pos = Coordinates(x=x2, y=y2)

            result = calculator.calculate_reward(
                agent_position=agent_pos,
                source_location=goal_pos,
            )

            # Reward must be one of the configured values
            min_reward = min(reward_goal, reward_default)
            max_reward = max(reward_goal, reward_default)

            assert min_reward <= result.reward <= max_reward


class TestActionSpaceProperties:
    """Property-based tests for action space."""

    @given(st.integers(min_value=0, max_value=3))
    def test_all_actions_have_movement_vectors(self, action):
        """Property: Every valid action has a movement vector."""
        from plume_nav_sim.core.constants import MOVEMENT_VECTORS

        assert action in MOVEMENT_VECTORS
        vector = MOVEMENT_VECTORS[action]

        assert isinstance(vector, tuple)
        assert len(vector) == 2

    @given(st.integers(min_value=0, max_value=3))
    def test_movement_vectors_have_unit_length(self, action):
        """Property: All movement vectors have Manhattan distance = 1."""
        from plume_nav_sim.core.constants import MOVEMENT_VECTORS

        dx, dy = MOVEMENT_VECTORS[action]
        manhattan_distance = abs(dx) + abs(dy)

        assert manhattan_distance == 1


class TestBoundaryEnforcementProperties:
    """Property-based tests for boundary enforcement."""

    @given(
        st.integers(min_value=10, max_value=100),
        st.integers(min_value=10, max_value=100),
        st.integers(min_value=0, max_value=199),
        st.integers(min_value=0, max_value=199),
        st.integers(min_value=0, max_value=3),  # action
    )
    def test_enforce_movement_bounds_returns_valid_position(
        self, width, height, x, y, action
    ):
        """Property: enforce_movement_bounds always returns in-bounds position."""
        from plume_nav_sim.core.boundary_enforcer import BoundaryEnforcer

        # Clamp initial position to bounds
        x = min(x, width - 1)
        y = min(y, height - 1)

        grid = GridSize(width=width, height=height)
        enforcer = BoundaryEnforcer(grid_size=grid)

        coord = Coordinates(x=x, y=y)
        result = enforcer.enforce_movement_bounds(coord, action)

        # Result position must be within bounds
        assert 0 <= result.final_position.x < width
        assert 0 <= result.final_position.y < height

    @given(
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=10, max_value=50),
        st.integers(min_value=0, max_value=49),
        st.integers(min_value=0, max_value=49),
    )
    def test_validate_position_accepts_valid_positions(self, width, height, x, y):
        """Property: validate_position accepts positions within bounds."""
        from plume_nav_sim.core.boundary_enforcer import BoundaryEnforcer

        grid = GridSize(width=width, height=height)
        enforcer = BoundaryEnforcer(grid_size=grid)

        # Clamp to bounds
        x = min(x, width - 1)
        y = min(y, height - 1)

        coord = Coordinates(x=x, y=y)
        # Should not raise when position is valid and raise_on_invalid=True
        is_valid = enforcer.validate_position(coord, raise_on_invalid=False)

        # Position within bounds should be valid
        assert is_valid is True


class TestConfigurationProperties:
    """Property-based tests for configuration validation."""

    @given(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
    )
    def test_grid_size_config_accepts_valid_tuples(self, width, height):
        """Property: EnvironmentConfig accepts valid grid_size tuples."""
        from plume_nav_sim.core.types import EnvironmentConfig

        config = EnvironmentConfig(
            grid_size=(width, height),
            source_location=(width // 2, height // 2),
        )

        # Config converts tuple to GridSize
        assert config.grid_size.width == width
        assert config.grid_size.height == height

    @given(
        st.floats(min_value=0.0, max_value=100.0),
        st.floats(min_value=0.0, max_value=10.0),
        st.floats(min_value=-1.0, max_value=1.0),
    )
    def test_reward_config_accepts_finite_values(
        self, goal_radius, reward_goal, reward_default
    ):
        """Property: RewardCalculatorConfig accepts finite numeric values."""
        from plume_nav_sim.core.reward_calculator import RewardCalculatorConfig

        # Skip NaN and infinite values
        if not (
            np.isfinite(goal_radius)
            and np.isfinite(reward_goal)
            and np.isfinite(reward_default)
        ):
            return

        if goal_radius < 0:
            return  # Invalid

        config = RewardCalculatorConfig(
            goal_radius=goal_radius,
            reward_goal_reached=reward_goal,
            reward_default=reward_default,
        )

        assert config.goal_radius == goal_radius
        assert config.reward_goal_reached == reward_goal
        assert config.reward_default == reward_default


class TestMathematicalProperties:
    """General mathematical properties that must hold."""

    @given(
        st.integers(min_value=0, max_value=100),
        st.integers(min_value=0, max_value=100),
    )
    def test_coordinate_hash_consistent(self, x, y):
        """Property: Same coordinates produce same hash."""
        coord1 = Coordinates(x=x, y=y)
        coord2 = Coordinates(x=x, y=y)

        assert hash(coord1) == hash(coord2)

    @given(
        st.integers(min_value=1, max_value=100),
        st.integers(min_value=1, max_value=100),
    )
    def test_grid_size_hash_consistent(self, width, height):
        """Property: Same grid sizes produce same hash."""
        grid1 = GridSize(width=width, height=height)
        grid2 = GridSize(width=width, height=height)

        assert hash(grid1) == hash(grid2)

    @given(
        st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=1, max_size=100)
    )
    def test_reward_accumulation_associative(self, rewards):
        """Property: Sum of rewards is associative (order doesn't matter for sum)."""
        # Filter out non-finite values
        finite_rewards = [r for r in rewards if np.isfinite(r)]
        if not finite_rewards:
            return

        # Sum in different orders
        total1 = sum(finite_rewards)
        total2 = sum(reversed(finite_rewards))

        assert np.isclose(total1, total2, rtol=1e-9)
