"""
Integration tests for environment factory function.

Tests that the factory correctly assembles environments from components
with various configuration options.
"""

import pytest

from plume_nav_sim.envs.factory import create_component_environment


class TestComponentEnvironmentFactory:
    """Tests for create_component_environment factory."""

    def test_default_configuration(self):
        """Test: Factory creates environment with defaults."""
        env = create_component_environment()

        assert env is not None
        assert env.grid_size.width == 128
        assert env.grid_size.height == 128
        assert env.goal_location.x == 64
        assert env.goal_location.y == 64
        assert env.max_steps == 1000

        # Should be able to reset and step
        obs, info = env.reset()
        assert obs is not None

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None

        env.close()

    def test_discrete_actions(self):
        """Test: Factory creates discrete action environment."""
        env = create_component_environment(action_type="discrete")

        # Discrete actions have 4 actions (UP, RIGHT, DOWN, LEFT)
        assert env.action_space.n == 4
        env.close()

    def test_oriented_actions(self):
        """Test: Factory creates oriented action environment."""
        env = create_component_environment(action_type="oriented")

        # Oriented actions have 3 actions (FORWARD, TURN_LEFT, TURN_RIGHT)
        assert env.action_space.n == 3
        env.close()

    def test_run_tumble_actions(self):
        """Test: Factory creates run/tumble action environment."""
        env = create_component_environment(action_type="run_tumble")

        # Run/tumble actions have 2 actions (RUN, TUMBLE)
        assert env.action_space.n == 2
        obs, _ = env.reset()
        assert obs is not None
        env.close()

    def test_concentration_observation(self):
        """Test: Factory creates concentration sensor."""
        env = create_component_environment(observation_type="concentration")

        obs, _ = env.reset()
        # ConcentrationSensor returns (1,) array
        assert obs.shape == (1,)
        env.close()

    def test_antennae_observation(self):
        """Test: Factory creates antennae array sensor."""
        env = create_component_environment(observation_type="antennae")

        obs, _ = env.reset()
        # AntennaeArraySensor with 2 sensors returns (2,) array
        assert obs.shape == (2,)
        env.close()

    def test_sparse_reward(self):
        """Test: Factory creates sparse reward function."""
        env = create_component_environment(reward_type="sparse")

        obs, _ = env.reset()
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)

        # Sparse reward is 0.0 or 1.0
        assert reward in [0.0, 1.0]
        env.close()

    def test_step_penalty_reward(self):
        """Test: Factory creates step penalty reward function."""
        env = create_component_environment(reward_type="step_penalty")

        obs, _ = env.reset()
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)

        # Step penalty reward should be negative (penalty) or positive (goal)
        assert isinstance(reward, (int, float))
        env.close()

    def test_custom_grid_size(self):
        """Test: Factory accepts custom grid size."""
        env = create_component_environment(grid_size=(64, 64))

        assert env.grid_size.width == 64
        assert env.grid_size.height == 64
        env.close()

    def test_custom_goal_location(self):
        """Test: Factory accepts custom goal location."""
        env = create_component_environment(goal_location=(50, 50))

        assert env.goal_location.x == 50
        assert env.goal_location.y == 50
        env.close()

    def test_custom_start_location(self):
        """Test: Factory accepts custom start location."""
        env = create_component_environment(start_location=(10, 10))

        assert env.start_location.x == 10
        assert env.start_location.y == 10

        obs, info = env.reset()
        # Agent should start at specified location
        assert info["agent_position"] == (10, 10)
        env.close()

    def test_custom_max_steps(self):
        """Test: Factory accepts custom max_steps."""
        env = create_component_environment(max_steps=500)

        assert env.max_steps == 500
        env.close()

    def test_custom_goal_radius(self):
        """Test: Factory accepts custom goal_radius."""
        env = create_component_environment(goal_radius=10.0)

        assert env.goal_radius == 10.0
        env.close()

    def test_invalid_action_type_raises_error(self):
        """Test: Factory raises error for invalid action_type."""
        with pytest.raises(
            ValueError,
            match=(
                "Invalid action_type: invalid\\. Must be 'discrete', 'oriented', "
                "or 'run_tumble'."
            ),
        ):
            create_component_environment(action_type="invalid")

    def test_invalid_observation_type_raises_error(self):
        """Test: Factory raises error for invalid observation_type."""
        with pytest.raises(ValueError, match="Invalid observation_type"):
            create_component_environment(observation_type="invalid")

    def test_invalid_reward_type_raises_error(self):
        """Test: Factory raises error for invalid reward_type."""
        with pytest.raises(ValueError, match="Invalid reward_type"):
            create_component_environment(reward_type="dense")  # 'dense' not supported

    def test_full_episode_execution(self):
        """Test: Factory environment can execute full episode."""
        env = create_component_environment(
            grid_size=(32, 32),
            goal_location=(16, 16),
            max_steps=50,
            action_type="discrete",
            reward_type="sparse",
        )

        obs, info = env.reset(seed=42)

        done = False
        step_count = 0
        while not done and step_count < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        # Should eventually terminate or truncate
        assert done
        env.close()

    def test_multiple_environments(self):
        """Test: Can create multiple environments simultaneously."""
        env1 = create_component_environment(grid_size=(32, 32))
        env2 = create_component_environment(grid_size=(64, 64))

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        # Should be independent
        assert obs1 is not None
        assert obs2 is not None

        env1.close()
        env2.close()
