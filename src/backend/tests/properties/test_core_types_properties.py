"""
Property Tests: Core Data Types

Uses Hypothesis to verify mathematical properties of Coordinates, GridSize, and AgentState.
Tests properties defined in contracts/core_types.md

Reference: CONTRACTS.md v1.1.0, TEST_TAXONOMY.md
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from plume_nav_sim.core.types import Coordinates, GridSize
from plume_nav_sim.utils.exceptions import ValidationError

# ============================================================================
# Hypothesis Strategies
# ============================================================================

# Note: Current implementation rejects negative coordinates
# We'll test both positive and (when fixed) negative coordinates
positive_coordinates = st.builds(
    Coordinates, x=st.integers(0, 1000), y=st.integers(0, 1000)
)

# When coordinate validation is fixed, use this:
# all_coordinates = st.builds(
#     Coordinates,
#     x=st.integers(-1000, 1000),
#     y=st.integers(-1000, 1000)
# )

valid_grid_sizes = st.builds(
    GridSize, width=st.integers(1, 100), height=st.integers(1, 100)
)


# ============================================================================
# Coordinates: Distance Metric Properties
# ============================================================================


class TestCoordinatesDistanceMetric:
    """Test distance metric mathematical properties.

    Contract: core_types.md - Coordinates distance_to()

    Distance must satisfy 4 properties:
    1. Non-negativity: d(a,b) >= 0
    2. Identity: d(a,a) = 0
    3. Symmetry: d(a,b) = d(b,a)
    4. Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    """

    @given(c1=positive_coordinates, c2=positive_coordinates)
    @settings(max_examples=200)
    def test_distance_non_negative(self, c1, c2):
        """Property 1: Distance is always non-negative.

        Contract: core_types.md
        ∀ a, b: distance(a, b) >= 0
        """
        distance = c1.distance_to(c2)
        assert distance >= 0, f"Distance must be non-negative, got {distance}"

    @given(coords=positive_coordinates)
    @settings(max_examples=100)
    def test_distance_identity(self, coords):
        """Property 2: Distance from point to itself is zero.

        Contract: core_types.md
        ∀ a: distance(a, a) = 0
        """
        distance = coords.distance_to(coords)
        assert distance == 0.0, f"Distance to self must be 0, got {distance}"

    @given(c1=positive_coordinates, c2=positive_coordinates)
    @settings(max_examples=200)
    def test_distance_symmetry(self, c1, c2):
        """Property 3: Distance is symmetric.

        Contract: core_types.md
        ∀ a, b: distance(a, b) = distance(b, a)

        CRITICAL for reward function symmetry.
        """
        d_ab = c1.distance_to(c2)
        d_ba = c2.distance_to(c1)

        assert (
            d_ab == d_ba
        ), f"Distance not symmetric: d({c1},{c2})={d_ab}, d({c2},{c1})={d_ba}"

    @given(c1=positive_coordinates, c2=positive_coordinates, c3=positive_coordinates)
    @settings(max_examples=100)
    def test_distance_triangle_inequality(self, c1, c2, c3):
        """Property 4: Triangle inequality holds.

        Contract: core_types.md
        ∀ a, b, c: distance(a, c) <= distance(a, b) + distance(b, c)

        The direct path is never longer than an indirect path.
        """
        d_ac = c1.distance_to(c3)
        d_ab = c1.distance_to(c2)
        d_bc = c2.distance_to(c3)

        # Allow small floating point tolerance
        assert (
            d_ac <= d_ab + d_bc + 1e-10
        ), f"Triangle inequality violated: d(a,c)={d_ac} > d(a,b)+d(b,c)={d_ab+d_bc}"

    def test_distance_euclidean_formula(self):
        """Distance follows Euclidean formula.

        Contract: core_types.md
        distance(p1, p2) = √((x2-x1)² + (y2-y1)²)
        """
        c1 = Coordinates(0, 0)
        c2 = Coordinates(3, 4)

        # 3² + 4² = 9 + 16 = 25, √25 = 5
        distance = c1.distance_to(c2)

        assert (
            abs(distance - 5.0) < 1e-10
        ), f"Euclidean distance should be 5.0, got {distance}"


# ============================================================================
# Coordinates: Immutability
# ============================================================================


class TestCoordinatesImmutability:
    """Test that Coordinates are immutable (frozen).

    Contract: core_types.md - Invariant I3
    """

    def test_coordinates_frozen(self):
        """Cannot modify x or y after creation.

        Contract: core_types.md
        Coordinates are immutable (frozen dataclass).
        """
        coords = Coordinates(5, 10)

        with pytest.raises(AttributeError):
            coords.x = 20  # type: ignore

        with pytest.raises(AttributeError):
            coords.y = 30  # type: ignore

    @given(x=st.integers(0, 100), y=st.integers(0, 100))
    @settings(max_examples=50)
    def test_coordinates_equality_structural(self, x, y):
        """Coordinates with same x,y are equal.

        Contract: core_types.md
        (x1, y1) = (x2, y2) ⟺ x1=x2 ∧ y1=y2
        """
        c1 = Coordinates(x, y)
        c2 = Coordinates(x, y)

        assert c1 == c2, "Coordinates with same x,y should be equal"

    @given(coords=positive_coordinates)
    @settings(max_examples=100)
    def test_coordinates_hashable(self, coords):
        """Coordinates can be used in sets/dicts.

        Contract: core_types.md - Invariant I4
        """
        # Should be hashable
        hash_value = hash(coords)
        assert isinstance(hash_value, int)

        # Can use in set
        coord_set = {coords}
        assert coords in coord_set

        # Can use as dict key
        coord_dict = {coords: "value"}
        assert coord_dict[coords] == "value"

    @given(x=st.integers(0, 100), y=st.integers(0, 100))
    @settings(max_examples=50)
    def test_equal_coordinates_same_hash(self, x, y):
        """Equal coordinates have same hash.

        Contract: core_types.md
        c1 = c2 ⇒ hash(c1) = hash(c2)
        """
        c1 = Coordinates(x, y)
        c2 = Coordinates(x, y)

        assert hash(c1) == hash(c2), "Equal coordinates must have same hash"


# ============================================================================
# GridSize: Validation
# ============================================================================


class TestGridSizeValidation:
    """Test GridSize validation invariants.

    Contract: core_types.md
    Invariants I1-I4: width, height must be positive and bounded.
    """

    def test_grid_size_requires_positive_dimensions(self):
        """Cannot create GridSize with zero or negative dimensions.

        Contract: core_types.md - Invariants I1, I2
        width > 0, height > 0
        """
        # Zero dimensions
        with pytest.raises((ValidationError, ValueError)):
            GridSize(width=0, height=10)

        with pytest.raises((ValidationError, ValueError)):
            GridSize(width=10, height=0)

        # Negative dimensions
        with pytest.raises((ValidationError, ValueError)):
            GridSize(width=-5, height=10)

        with pytest.raises((ValidationError, ValueError)):
            GridSize(width=10, height=-5)

    @given(width=st.integers(1, 100), height=st.integers(1, 100))
    @settings(max_examples=100)
    def test_grid_size_accepts_valid_dimensions(self, width, height):
        """Valid positive dimensions are accepted.

        Contract: core_types.md
        """
        grid = GridSize(width=width, height=height)

        assert grid.width == width
        assert grid.height == height

    def test_grid_size_maximum_enforced(self):
        """Dimensions cannot exceed MAX_GRID_DIMENSION.

        Contract: core_types.md - Invariants I3, I4
        Implementation enforces: width, height <= 1024
        """
        MAX_DIM = 1024  # Actual implementation limit

        # At maximum should work
        grid = GridSize(width=MAX_DIM, height=MAX_DIM)
        assert grid.width == MAX_DIM

        # Above maximum should fail
        with pytest.raises((ValidationError, ValueError)):
            GridSize(width=MAX_DIM + 1, height=10)


# ============================================================================
# GridSize: Operations
# ============================================================================


class TestGridSizeOperations:
    """Test GridSize operation contracts."""

    @given(grid=valid_grid_sizes)
    @settings(max_examples=100)
    def test_total_cells_equals_product(self, grid):
        """total_cells() = width × height

        Contract: core_types.md
        """
        expected = grid.width * grid.height
        actual = grid.total_cells()

        assert actual == expected, f"total_cells should be {expected}, got {actual}"

    @given(grid=valid_grid_sizes)
    @settings(max_examples=50)
    def test_total_cells_positive(self, grid):
        """total_cells() > 0 always.

        Contract: core_types.md - Postcondition C2
        """
        assert grid.total_cells() > 0, "Grid must have positive cell count"

    @given(grid=valid_grid_sizes, x=st.integers(-10, 110), y=st.integers(-10, 110))
    @settings(max_examples=200)
    def test_contains_definition(self, grid, x, y):
        """contains(coord) ⟺ 0 <= x < width ∧ 0 <= y < height

        Contract: core_types.md
        """
        try:
            coord = Coordinates(x, y)
        except ValidationError:
            # If coordinates are invalid (negative), skip
            assume(x >= 0 and y >= 0)
            coord = Coordinates(x, y)

        expected = (0 <= x < grid.width) and (0 <= y < grid.height)
        actual = grid.contains(coord)

        assert (
            actual == expected
        ), f"contains({x},{y}) in {grid.width}×{grid.height} should be {expected}, got {actual}"

    def test_contains_boundary_conditions(self):
        """Test exact boundaries of contains().

        Contract: core_types.md - Properties
        """
        grid = GridSize(width=10, height=10)

        # Corners inside
        assert grid.contains(Coordinates(0, 0)), "Origin should be inside"
        assert grid.contains(Coordinates(9, 9)), "Max corner inside"

        # Just outside
        assert not grid.contains(Coordinates(10, 5)), "x=width is outside"
        assert not grid.contains(Coordinates(5, 10)), "y=height is outside"

    @given(grid=valid_grid_sizes)
    @settings(max_examples=50)
    def test_center_within_grid(self, grid):
        """center() returns coordinate within grid.

        Contract: core_types.md - Postcondition C3
        """
        center = grid.center()

        assert grid.contains(center), f"Center {center} should be within grid {grid}"

    def test_center_formula(self):
        """center() uses integer division.

        Contract: core_types.md
        center = (width // 2, height // 2)
        """
        grid = GridSize(width=32, height=32)
        center = grid.center()

        assert center == Coordinates(
            16, 16
        ), f"Center of 32×32 should be (16,16), got {center}"

        # Odd dimensions
        grid2 = GridSize(width=11, height=11)
        center2 = grid2.center()

        assert center2 == Coordinates(
            5, 5
        ), f"Center of 11×11 should be (5,5), got {center2}"


# ============================================================================
# GridSize: Immutability
# ============================================================================


class TestGridSizeImmutability:
    """Test that GridSize is immutable (frozen).

    Contract: core_types.md - Invariant I5
    """

    def test_grid_size_frozen(self):
        """Cannot modify width or height after creation."""
        grid = GridSize(width=32, height=32)

        with pytest.raises(AttributeError):
            grid.width = 64  # type: ignore

        with pytest.raises(AttributeError):
            grid.height = 64  # type: ignore


# ============================================================================
# AgentState: Monotonicity Properties
# ============================================================================


class TestAgentStateMonotonicity:
    """Test AgentState progression properties.

    Step count remains monotonic, total reward tracks cumulative sum
    even when per-step rewards are negative.
    """

    def test_step_count_monotonic(self):
        """step_count never decreases.

        Contract: core_types.md - Invariant I5
        step_count' >= step_count
        """
        from plume_nav_sim.core.state_manager import AgentState

        state = AgentState(position=Coordinates(0, 0))

        prev_count = state.step_count

        for _ in range(10):
            state.increment_step()
            current_count = state.step_count

            assert (
                current_count > prev_count
            ), f"Step count decreased: {prev_count} -> {current_count}"
            assert (
                current_count == prev_count + 1
            ), f"Step count should increase by 1, got {current_count - prev_count}"

            prev_count = current_count

    def test_total_reward_tracks_cumulative_sum(self):
        """Total reward equals the cumulative sum of all rewards, including negatives."""
        from plume_nav_sim.core.state_manager import AgentState

        state = AgentState(position=Coordinates(0, 0))

        rewards = [0.0, 0.5, -0.25, -1.0, 2.0]
        expected_total = 0.0

        for reward in rewards:
            state.add_reward(reward)
            expected_total += reward
            assert state.total_reward == pytest.approx(
                expected_total
            ), f"Expected {expected_total}, got {state.total_reward}"

    def test_negative_reward_allowed(self):
        """Negative rewards are accumulated without validation errors."""
        from plume_nav_sim.core.state_manager import AgentState

        state = AgentState(position=Coordinates(0, 0))

        state.add_reward(-1.5)
        assert state.total_reward == pytest.approx(-1.5)


# ============================================================================
# AgentState: Idempotency
# ============================================================================


class TestAgentStateIdempotency:
    """Test AgentState idempotent operations.

    Contract: core_types.md - Invariant I7
    """

    def test_mark_goal_reached_idempotent(self):
        """Calling mark_goal_reached() multiple times is safe.

        Contract: core_types.md
        goal_reached is idempotent (once True, stays True).
        """
        from plume_nav_sim.core.state_manager import AgentState

        state = AgentState(position=Coordinates(0, 0))

        assert state.goal_reached == False, "Initially not reached"

        state.mark_goal_reached()
        assert state.goal_reached == True, "After mark, is reached"

        state.mark_goal_reached()
        assert state.goal_reached == True, "Still reached after second mark"

        state.mark_goal_reached()
        assert state.goal_reached == True, "Still reached after third mark"

    def test_cannot_unreach_goal(self):
        """Once goal reached, cannot be unset.

        Contract: core_types.md
        ¬∃ state: goal_reached = True ∧ goal_reached' = False
        """
        from plume_nav_sim.core.state_manager import AgentState

        state = AgentState(position=Coordinates(0, 0))
        state.mark_goal_reached()

        # No method should exist to set goal_reached = False
        # (except reset, which is for new episodes)
        assert state.goal_reached == True

        # Even adding reward shouldn't change goal status
        state.add_reward(1.0)
        assert state.goal_reached == True


# ============================================================================
# AgentState: Validation
# ============================================================================


class TestAgentStateValidation:
    """Test AgentState precondition validation."""

    def test_initial_step_count_non_negative(self):
        """Initial step_count must be non-negative.

        Contract: core_types.md - Invariant I2
        """
        from plume_nav_sim.core.state_manager import AgentState

        # Valid
        state = AgentState(position=Coordinates(0, 0), step_count=0)
        assert state.step_count == 0

        state2 = AgentState(position=Coordinates(0, 0), step_count=5)
        assert state2.step_count == 5

        # Invalid
        with pytest.raises((ValidationError, ValueError)):
            AgentState(position=Coordinates(0, 0), step_count=-1)

    def test_initial_total_reward_allows_negative(self):
        """Initial total_reward can be negative for debt-based reward schemes."""
        from plume_nav_sim.core.state_manager import AgentState

        # Valid
        state = AgentState(position=Coordinates(0, 0), total_reward=0.0)
        assert state.total_reward == 0.0

        state2 = AgentState(position=Coordinates(0, 0), total_reward=5.0)
        assert state2.total_reward == 5.0

        state3 = AgentState(position=Coordinates(0, 0), total_reward=-2.5)
        assert state3.total_reward == -2.5

        # Invalid input types still rejected
        with pytest.raises((ValidationError, ValueError)):
            AgentState(position=Coordinates(0, 0), total_reward="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
