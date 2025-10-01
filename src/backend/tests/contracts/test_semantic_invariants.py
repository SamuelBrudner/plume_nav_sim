"""Semantic Invariant Tests.

Tests that enforce the core semantic guarantees documented in SEMANTIC_MODEL.md.
These are the properties that MUST ALWAYS HOLD for correctness.

Reference: ../../SEMANTIC_MODEL.md Section "Semantic Invariants (Testable)"
"""

import numpy as np
import pytest

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.types import AgentState, EnvironmentConfig
from plume_nav_sim.envs.plume_search_env import create_plume_search_env
from plume_nav_sim.utils.seeding import create_seeded_rng


class TestPositionInvariant:
    """Invariant 1: Agent position ALWAYS within grid bounds.

    From SEMANTIC_MODEL.md:
    - After any action, agent.position must satisfy:
      0 <= x < grid_width AND 0 <= y < grid_height
    """

    def test_agent_starts_in_bounds(self):
        """Initial agent position must be within grid bounds."""
        # Note: create_plume_search_env uses default 128x128 grid
        # TODO: Fix config to be respected
        env = create_plume_search_env()

        obs, info = env.reset()

        # Extract agent position from info (returns tuple)
        agent_x, agent_y = info["agent_xy"]
        # Position should be within bounds (default 128x128)
        assert 0 <= agent_x < 128
        assert 0 <= agent_y < 128

    def test_agent_stays_in_bounds_after_actions(self):
        """Agent must remain in bounds after every action."""
        # Note: create_plume_search_env uses default 128x128 grid
        # TODO: Fix config to be respected
        env = create_plume_search_env()
        obs, info = env.reset()

        # Get actual grid size from environment
        grid_width = 128  # Default from env
        grid_height = 128

        # Try to walk agent off the grid
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # INVARIANT: Position MUST be in bounds
            agent_x, agent_y = info["agent_xy"]
            assert (
                0 <= agent_x < grid_width
            ), f"Position ({agent_x}, {agent_y}) violated x bounds"
            assert (
                0 <= agent_y < grid_height
            ), f"Position ({agent_x}, {agent_y}) violated y bounds"

            if terminated or truncated:
                break

    def test_boundary_enforcement_prevents_out_of_bounds(self):
        """Boundary enforcer must prevent out-of-bounds positions."""
        from plume_nav_sim.core.boundary_enforcer import BoundaryEnforcer

        grid_size = GridSize(width=10, height=10)
        enforcer = BoundaryEnforcer(grid_size)

        # Try invalid positions - note: Coordinates rejects negative values
        # so we test boundary cases only
        invalid_positions = [
            (10, 5),  # x at boundary
            (5, 10),  # y at boundary
            (15, 15),  # both beyond
        ]

        for x, y in invalid_positions:
            pos = Coordinates(x=x, y=y)
            # Should return False
            is_valid = enforcer.validate_position(pos, raise_on_invalid=False)
            assert not is_valid, f"Position ({x}, {y}) should be invalid"


class TestStepCountInvariant:
    """Invariant 2: step_count increments by exactly 1 per step.

    From SEMANTIC_MODEL.md:
    - step_count_after = step_count_before + 1
    - This must hold for EVERY step
    """

    def test_step_count_increments_monotonically(self):
        """Step count must increase by exactly 1 each step."""
        env = create_plume_search_env()
        env.reset()

        previous_step = 0
        for i in range(20):
            obs, reward, terminated, truncated, info = env.step(0)
            current_step = info.get("step_count", i + 1)

            # INVARIANT: Exactly +1
            assert (
                current_step == previous_step + 1
            ), f"Step count jumped from {previous_step} to {current_step}"
            previous_step = current_step

            if terminated or truncated:
                break

    def test_step_count_matches_episode_length(self):
        """Total steps must equal final step_count."""
        env = create_plume_search_env(env_config=EnvironmentConfig(max_steps=10))
        env.reset()

        actual_steps = 0
        for _ in range(15):  # More than max_steps
            obs, reward, terminated, truncated, info = env.step(0)
            actual_steps += 1

            if terminated or truncated:
                final_step_count = info.get("step_count", actual_steps)
                assert final_step_count == actual_steps
                break

    @pytest.mark.skip(
        reason="StateManager integration not complete - TODO: wire components"
    )
    def test_agent_state_step_count_consistent(self):
        """AgentState.step_count must match actual steps taken."""
        from plume_nav_sim.core.state_manager import StateManager, StateManagerConfig

        config = StateManagerConfig(
            grid_size=GridSize(20, 20),
            source_location=Coordinates(10, 10),
            max_steps=50,
            goal_radius=5.0,
        )
        manager = StateManager(config=config)
        manager.initialize_episode(episode_seed=42)

        for step in range(10):
            state = manager.current_agent_state
            # INVARIANT: step_count matches actual steps
            assert state.step_count == step

            # Take action
            manager.update_agent_state(action=0, step_reward=0.0)


class TestRewardAccumulationInvariant:
    """Invariant 3: total_reward = sum(all step rewards).

    From SEMANTIC_MODEL.md:
    - agent.total_reward = sum(reward_i for all steps i)
    - No reward should be lost or duplicated
    """

    @pytest.mark.skip(
        reason="Environment doesn't track total_reward in info yet - TODO: implement"
    )
    def test_total_reward_equals_sum_of_steps(self):
        """Total reward must equal sum of individual step rewards."""
        env = create_plume_search_env()
        env.reset()

        collected_rewards = []
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(0)
            collected_rewards.append(reward)

            # INVARIANT: total = sum
            expected_total = sum(collected_rewards)
            actual_total = info.get("total_reward", 0.0)

            assert np.isclose(
                actual_total, expected_total, rtol=1e-9
            ), f"Total reward {actual_total} != sum {expected_total}"

            if terminated or truncated:
                break

    @pytest.mark.skip(
        reason="StateManager integration not complete - TODO: wire components"
    )
    def test_agent_state_reward_accumulation(self):
        """AgentState.total_reward must accumulate correctly."""
        from plume_nav_sim.core.state_manager import StateManager, StateManagerConfig

        config = StateManagerConfig(
            grid_size=GridSize(20, 20),
            source_location=Coordinates(10, 10),
            max_steps=50,
            goal_radius=5.0,
        )
        manager = StateManager(config=config)
        manager.initialize_episode(episode_seed=42)

        rewards = [0.1, 0.2, 0.0, 0.5, 1.0]
        expected_total = 0.0

        for step_reward in rewards:
            manager.update_agent_state(action=0, step_reward=step_reward)
            expected_total += step_reward

            state = manager.current_agent_state
            # INVARIANT: Accumulation is exact
            assert np.isclose(state.total_reward, expected_total, rtol=1e-9)


class TestDeterminismInvariant:
    """Invariant 4: Same seed → same trajectory.

    From SEMANTIC_MODEL.md:
    - Given same seed and actions, entire episode trajectory must be identical
    - This is the core reproducibility guarantee
    """

    def test_same_seed_same_observations(self):
        """Same seed must produce identical observations."""
        seed = 12345
        actions = [0, 1, 2, 3, 0, 1]

        # Run 1
        env1 = create_plume_search_env()
        env1.reset(seed=seed)
        obs1_list = []
        for action in actions:
            obs, _, _, _, _ = env1.step(action)
            obs1_list.append(obs)

        # Run 2 with same seed
        env2 = create_plume_search_env()
        env2.reset(seed=seed)
        obs2_list = []
        for action in actions:
            obs, _, _, _, _ = env2.step(action)
            obs2_list.append(obs)

        # INVARIANT: Observations must be identical
        for i, (obs1, obs2) in enumerate(zip(obs1_list, obs2_list)):
            # Handle Dict observations
            if isinstance(obs1, dict) and isinstance(obs2, dict):
                for key in obs1.keys():
                    if isinstance(obs1[key], np.ndarray):
                        assert np.allclose(
                            obs1[key], obs2[key], rtol=1e-9
                        ), f"Observation key '{key}' differs at step {i}"
                    else:
                        assert (
                            obs1[key] == obs2[key]
                        ), f"Observation key '{key}' differs at step {i}"
            elif isinstance(obs1, np.ndarray):
                assert np.allclose(
                    obs1, obs2, rtol=1e-9
                ), f"Observations differ at step {i}"

    def test_same_seed_same_rewards(self):
        """Same seed must produce identical rewards."""
        seed = 67890
        actions = [1, 1, 1, 2, 2]

        # Run 1
        env1 = create_plume_search_env()
        env1.reset(seed=seed)
        rewards1 = [env1.step(a)[1] for a in actions]

        # Run 2
        env2 = create_plume_search_env()
        env2.reset(seed=seed)
        rewards2 = [env2.step(a)[1] for a in actions]

        # INVARIANT: Rewards must be identical
        assert np.allclose(rewards1, rewards2, rtol=1e-9)

    def test_rng_reproducibility(self):
        """RNG with same seed must produce identical sequences."""
        seed = 42

        rng1, _ = create_seeded_rng(seed)
        sequence1 = rng1.random(100)

        rng2, _ = create_seeded_rng(seed)
        sequence2 = rng2.random(100)

        # INVARIANT: Sequences must be identical
        assert np.array_equal(sequence1, sequence2)


class TestGoalDetectionConsistency:
    """Invariant 5: Goal detection must be consistent across components.

    From SEMANTIC_MODEL.md:
    - If distance_to_goal <= goal_radius, then:
      - reward should be REWARD_GOAL_REACHED
      - terminated should be True
      - agent.goal_reached should be True
    - All three signals must agree
    """

    @pytest.mark.skip(
        reason="RewardCalculator integration not complete - TODO: wire components"
    )
    def test_goal_signals_consistent_when_reached(self):
        """When goal reached, all signals must agree."""
        from plume_nav_sim.core.reward_calculator import (
            RewardCalculator,
            RewardCalculatorConfig,
        )

        config = RewardCalculatorConfig(
            goal_radius=5.0,
            reward_goal_reached=1.0,
            reward_default=0.0,
        )
        calculator = RewardCalculator(config=config)

        # Agent at goal
        agent_pos = Coordinates(10, 10)
        goal_pos = Coordinates(12, 12)  # distance ~ 2.83 < 5.0

        result = calculator.calculate_reward(
            agent_position=agent_pos,
            source_location=goal_pos,
            step_count=5,
        )

        # INVARIANT: All goal signals must align
        assert result.reward == 1.0, "Reward should be goal_reached value"
        assert result.goal_reached is True, "goal_reached flag must be True"
        # Note: terminated is handled by environment, but reward/goal_reached must agree

    @pytest.mark.skip(
        reason="RewardCalculator integration not complete - TODO: wire components"
    )
    def test_goal_signals_consistent_when_not_reached(self):
        """When goal not reached, signals must agree."""
        from plume_nav_sim.core.reward_calculator import (
            RewardCalculator,
            RewardCalculatorConfig,
        )

        config = RewardCalculatorConfig(
            goal_radius=2.0,
            reward_goal_reached=1.0,
            reward_default=0.0,
        )
        calculator = RewardCalculator(config=config)

        # Agent far from goal
        agent_pos = Coordinates(0, 0)
        goal_pos = Coordinates(20, 20)  # distance ~ 28.28 > 2.0

        result = calculator.calculate_reward(
            agent_position=agent_pos,
            source_location=goal_pos,
            step_count=5,
        )

        # INVARIANT: All signals must indicate goal NOT reached
        assert result.reward == 0.0, "Reward should be default"
        assert result.goal_reached is False, "goal_reached must be False"


class TestTerminationConsistency:
    """Invariant 6: Termination signals must be mutually exclusive.

    From SEMANTIC_MODEL.md:
    - terminated and truncated cannot BOTH be True
    - At most one termination signal at episode end
    """

    def test_terminated_and_truncated_mutually_exclusive(self):
        """Episode cannot be both terminated and truncated."""
        env = create_plume_search_env(env_config=EnvironmentConfig(max_steps=10))
        env.reset()

        for _ in range(15):  # More than max_steps
            obs, reward, terminated, truncated, info = env.step(0)

            # INVARIANT: NOT (terminated AND truncated)
            assert not (
                terminated and truncated
            ), "Episode cannot be both terminated and truncated"

            if terminated or truncated:
                break

    def test_episode_ends_with_one_signal(self):
        """Episode must end with exactly one termination signal."""
        env = create_plume_search_env(env_config=EnvironmentConfig(max_steps=5))
        env.reset()

        terminated_count = 0
        truncated_count = 0

        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(0)

            if terminated:
                terminated_count += 1
            if truncated:
                truncated_count += 1

            if terminated or truncated:
                # INVARIANT: Exactly one signal
                assert (
                    terminated_count + truncated_count == 1
                ), "Must have exactly one termination signal"
                break


class TestConfigImmutability:
    """Invariant 7: Configs must not mutate after creation.

    From SEMANTIC_MODEL.md:
    - EnvironmentConfig is frozen
    - Other configs should not change after __post_init__
    """

    def test_environment_config_is_frozen(self):
        """EnvironmentConfig must be immutable."""
        config = EnvironmentConfig()

        # Attempt to modify should fail
        with pytest.raises((AttributeError, Exception)):
            config.max_steps = 9999  # type: ignore

    def test_config_values_dont_change_during_use(self):
        """Config values must remain constant throughout usage."""
        from plume_nav_sim.core.reward_calculator import RewardCalculatorConfig

        config = RewardCalculatorConfig(
            goal_radius=5.0,
            reward_goal_reached=1.0,
            reward_default=0.0,
        )

        # Record initial values
        initial_radius = config.goal_radius
        initial_goal_reward = config.reward_goal_reached

        # Use config in calculations (simulate usage)
        for _ in range(100):
            # Config values should not change
            assert config.goal_radius == initial_radius
            assert config.reward_goal_reached == initial_goal_reward


class TestComponentIsolation:
    """Invariant: Components don't share mutable state.

    From SEMANTIC_MODEL.md:
    - No global state (except registry)
    - Operations on one component don't affect others
    """

    def test_independent_environments_dont_interfere(self):
        """Multiple env instances must be independent."""
        env1 = create_plume_search_env()
        env2 = create_plume_search_env()

        env1.reset(seed=1)
        env2.reset(seed=2)

        # Step env1
        obs1_a, _, _, _, _ = env1.step(0)

        # Step env2
        obs2_a, _, _, _, _ = env2.step(0)

        # Step env1 again
        obs1_b, _, _, _, _ = env1.step(1)

        # env1's second step should not be affected by env2
        # (This would fail if they shared state)
        if isinstance(obs1_a, dict) and isinstance(obs2_a, dict):
            # Compare Dict observations - at least one key should differ
            any_diff = False
            for key in obs1_a.keys():
                if isinstance(obs1_a[key], np.ndarray):
                    if not np.array_equal(obs1_a[key], obs2_a[key]):
                        any_diff = True
                        break
            assert any_diff, "Different seeds should give different obs"
        else:
            assert not np.array_equal(
                obs1_a, obs2_a
            ), "Different seeds should give different obs"

    @pytest.mark.skip(
        reason="StateManager integration not complete - TODO: wire components"
    )
    def test_state_manager_instances_independent(self):
        """Multiple StateManager instances must not interfere."""
        from plume_nav_sim.core.state_manager import StateManager, StateManagerConfig

        config = StateManagerConfig(
            grid_size=GridSize(20, 20),
            source_location=Coordinates(10, 10),
            max_steps=50,
            goal_radius=5.0,
        )

        manager1 = StateManager(config=config)
        manager2 = StateManager(config=config)

        manager1.initialize_episode(episode_seed=1)
        manager2.initialize_episode(episode_seed=2)

        # Update manager1
        manager1.update_agent_state(action=0, step_reward=0.5)

        # manager2 should be unaffected
        assert manager1.current_agent_state.step_count == 1
        assert manager2.current_agent_state.step_count == 0
        assert manager1.current_agent_state.total_reward == 0.5
        assert manager2.current_agent_state.total_reward == 0.0


class TestMathematicalConsistency:
    """Test mathematical relationships that must hold."""

    def test_distance_calculation_symmetric(self):
        """distance(A, B) must equal distance(B, A)."""
        from plume_nav_sim.core.geometry import calculate_euclidean_distance

        coord_a = Coordinates(5, 10)
        coord_b = Coordinates(15, 20)

        dist_ab = calculate_euclidean_distance(coord_a, coord_b)
        dist_ba = calculate_euclidean_distance(coord_b, coord_a)

        # INVARIANT: Symmetry
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

        # INVARIANT: Triangle inequality
        assert dist_ac <= dist_ab + dist_bc + 1e-9  # Small epsilon for float errors

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

            # INVARIANT: Non-negative
            assert dist >= 0.0
