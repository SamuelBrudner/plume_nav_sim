"""Semantic Invariant Tests (Component API).

Tests that enforce core semantic guarantees documented in SEMANTIC_MODEL.md,
targeting the modern component-based API (GridSize, AgentState, SparseGoalReward,
EnvironmentConfig).  Tests that required the removed PlumeSearchEnv have been
dropped; equivalent integration-level coverage lives in the runner and
environment-API test suites.

Reference: ../../SEMANTIC_MODEL.md Section "Semantic Invariants (Testable)"
"""

import numpy as np
import pytest

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.envs.config_types import EnvironmentConfig


class TestBoundaryEnforcement:
    """Invariant 1 (unit-level): GridSize.contains rejects out-of-bounds."""

    def test_boundary_enforcement_prevents_out_of_bounds(self):
        """GridSize.contains must reject out-of-bounds positions."""
        grid_size = GridSize(width=10, height=10)

        invalid_positions = [
            (10, 5),  # x at boundary
            (5, 10),  # y at boundary
            (15, 15),  # both beyond
        ]

        for x, y in invalid_positions:
            pos = Coordinates(x=x, y=y)
            assert not grid_size.contains(pos), f"Position ({x}, {y}) should be invalid"


class TestStepCountConsistency:
    """Invariant 2 (unit-level): AgentState.step_count tracks increments."""

    def test_agent_state_step_count_consistent(self):
        """AgentState.step_count must match actual steps taken."""
        from plume_nav_sim.core.types import AgentState

        state = AgentState(position=Coordinates(10, 10))

        for step in range(10):
            assert state.step_count == step
            state.increment_step()


class TestRewardAccumulationInvariant:
    """Invariant 3 (unit-level): AgentState.total_reward accumulates exactly."""

    def test_agent_state_reward_accumulation(self):
        """AgentState.total_reward must accumulate correctly."""
        from plume_nav_sim.core.types import AgentState

        state = AgentState(position=Coordinates(10, 10))

        rewards = [0.1, 0.2, 0.0, 0.5, 1.0]
        expected_total = 0.0

        for step_reward in rewards:
            state.add_reward(step_reward)
            expected_total += step_reward

        # INVARIANT: Accumulation is exact
        assert np.isclose(state.total_reward, expected_total, rtol=1e-9)


class TestGoalDetectionConsistency:
    """Invariant 5: Goal detection must be consistent with distance."""

    def test_goal_signals_consistent_when_reached(self):
        """When goal reached, all signals must agree."""
        from plume_nav_sim.core.geometry import calculate_euclidean_distance
        from plume_nav_sim.core.types import AgentState
        from plume_nav_sim.rewards.sparse_goal import SparseGoalReward

        goal_radius = 5.0
        goal_pos = Coordinates(12, 12)
        reward_fn = SparseGoalReward(goal_position=goal_pos, goal_radius=goal_radius)

        agent_pos = Coordinates(10, 10)  # distance ~ 2.83 < 5.0

        prev_state = AgentState(position=Coordinates(9, 9))
        next_state = AgentState(position=agent_pos)

        reward = reward_fn.compute_reward(
            prev_state=prev_state,
            action=0,
            next_state=next_state,
            plume_field=np.zeros((1,)),
        )

        dist = calculate_euclidean_distance(agent_pos, goal_pos)
        # INVARIANT: reward and distance must agree
        assert dist <= goal_radius
        assert reward == 1.0, "Reward should be goal_reached value"

    def test_goal_signals_consistent_when_not_reached(self):
        """When goal not reached, signals must agree."""
        from plume_nav_sim.core.geometry import calculate_euclidean_distance
        from plume_nav_sim.core.types import AgentState
        from plume_nav_sim.rewards.sparse_goal import SparseGoalReward

        goal_radius = 2.0
        goal_pos = Coordinates(20, 20)
        reward_fn = SparseGoalReward(goal_position=goal_pos, goal_radius=goal_radius)

        agent_pos = Coordinates(0, 0)  # distance ~ 28.28 > 2.0

        prev_state = AgentState(position=Coordinates(1, 1))
        next_state = AgentState(position=agent_pos)

        reward = reward_fn.compute_reward(
            prev_state=prev_state,
            action=0,
            next_state=next_state,
            plume_field=np.zeros((1,)),
        )

        dist = calculate_euclidean_distance(agent_pos, goal_pos)
        # INVARIANT: All signals must indicate goal NOT reached
        assert dist > goal_radius
        assert reward == 0.0, "Reward should be default"


class TestConfigImmutability:
    """Invariant 7: Configs must not mutate after creation."""

    def test_environment_config_is_frozen(self):
        """EnvironmentConfig must be immutable."""
        config = EnvironmentConfig()

        with pytest.raises((AttributeError, Exception)):
            config.max_steps = 9999  # type: ignore

    def test_config_values_dont_change_during_use(self):
        """Config values must remain constant throughout usage."""
        config = EnvironmentConfig()

        initial_max_steps = config.max_steps
        initial_goal_radius = config.goal_radius

        for _ in range(100):
            assert config.max_steps == initial_max_steps
            assert config.goal_radius == initial_goal_radius


class TestComponentIsolation:
    """Invariant: Components don't share mutable state."""

    def test_agent_state_instances_independent(self):
        """Multiple AgentState instances must not interfere."""
        from plume_nav_sim.core.types import AgentState

        state1 = AgentState(position=Coordinates(5, 5))
        state2 = AgentState(position=Coordinates(10, 10))

        # Update state1
        state1.increment_step()
        state1.add_reward(0.5)

        # state2 should be unaffected
        assert state1.step_count == 1
        assert state2.step_count == 0
        assert state1.total_reward == 0.5
        assert state2.total_reward == 0.0


class TestMathematicalConsistency:
    """Test mathematical relationships that must hold."""

    def test_distance_calculation_symmetric(self):
        """distance(A, B) must equal distance(B, A)."""
        from plume_nav_sim.core.geometry import calculate_euclidean_distance

        coord_a = Coordinates(5, 10)
        coord_b = Coordinates(15, 20)

        dist_ab = calculate_euclidean_distance(coord_a, coord_b)
        dist_ba = calculate_euclidean_distance(coord_b, coord_a)

        assert np.isclose(dist_ab, dist_ba, rtol=1e-9)

    def test_distance_triangle_inequality(self):
        """distance(A, C) <= distance(A, B) + distance(B, C)."""
        from plume_nav_sim.core.geometry import calculate_euclidean_distance

        coord_a = Coordinates(0, 0)
        coord_b = Coordinates(5, 5)
        coord_c = Coordinates(10, 0)

        dist_ac = calculate_euclidean_distance(coord_a, coord_c)
        dist_ab = calculate_euclidean_distance(coord_a, coord_b)
        dist_bc = calculate_euclidean_distance(coord_b, coord_c)

        assert dist_ac <= dist_ab + dist_bc + 1e-9

    def test_distance_non_negative(self):
        """All distances must be non-negative."""
        from plume_nav_sim.core.geometry import calculate_euclidean_distance

        rng = np.random.default_rng(42)

        for _ in range(100):
            x1, y1 = rng.integers(0, 100, size=2)
            x2, y2 = rng.integers(0, 100, size=2)

            coord_a = Coordinates(int(x1), int(y1))
            coord_b = Coordinates(int(x2), int(y2))

            dist = calculate_euclidean_distance(coord_a, coord_b)

            assert dist >= 0.0
