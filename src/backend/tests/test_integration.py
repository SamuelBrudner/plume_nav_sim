"""Integration smoke tests for the lightweight plume navigation environment."""

from __future__ import annotations

import numpy as np

from plume_nav_sim.envs.plume_search_env import create_plume_search_env


class TestEnvironmentIntegration:
    """Pytest-style class wrapper exposing integration smoke tests."""

    def test_complete_episode_workflow(self) -> None:
        test_complete_episode_workflow()

    def test_cross_component_seeding(self) -> None:
        test_cross_component_seeding()

    def test_system_level_performance(self) -> None:
        test_system_level_performance()


def test_complete_episode_workflow() -> None:
    """Run a short deterministic episode to validate Gymnasium-style behaviour."""

    env = create_plume_search_env(
        grid_size=(16, 16), source_location=(8, 8), max_steps=5, goal_radius=0
    )
    try:
        observation, info = env.reset(seed=21)
        assert observation.shape == (
            3,
        ), "Reset should return compact observation vector"
        assert info["agent_xy"]

        total_steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(total_steps % 4)
            total_steps += 1
            assert observation.shape == (3,)
            assert isinstance(reward, float)

        assert total_steps <= 5, "Episode should respect the configured max_steps"
    finally:
        env.close()


def test_cross_component_seeding() -> None:
    """Ensure seeding produces identical trajectories across environment instances."""

    env_a = create_plume_search_env(grid_size=(12, 12))
    env_b = create_plume_search_env(grid_size=(12, 12))
    try:
        obs_a, info_a = env_a.reset(seed=99)
        obs_b, info_b = env_b.reset(seed=99)

        assert np.allclose(obs_a, obs_b)
        assert info_a["agent_xy"] == info_b["agent_xy"]

        trajectory_a = [env_a.step(0) for _ in range(3)]
        trajectory_b = [env_b.step(0) for _ in range(3)]

        for step_a, step_b in zip(trajectory_a, trajectory_b):
            np.testing.assert_allclose(step_a[0], step_b[0])
            assert step_a[1:] == step_b[1:]
    finally:
        env_a.close()
        env_b.close()


def test_system_level_performance() -> None:
    """Exercise rendering pathways to ensure they remain callable in integration flows."""

    env = create_plume_search_env(grid_size=(8, 8))
    try:
        env.reset(seed=7)
        rgb = env.render(mode="rgb_array")
        assert rgb.shape == (8, 8, 3)
        assert rgb.dtype == np.uint8

        try:
            env.render(mode="human")
        except Exception:
            # If the interactive backend is unavailable, ensure the fallback path still returns an array.
            fallback = env.render(mode="rgb_array")
            assert fallback is not None
    finally:
        env.close()
