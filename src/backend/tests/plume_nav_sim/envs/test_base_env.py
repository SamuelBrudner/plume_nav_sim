"""
Contract tests for the canonical PlumeEnv.

These tests verify:
1. Gymnasium API compliance (reset, step, render, close)
2. Basic state handling (reset required before step)
3. Input validation for actions
"""

import gymnasium as gym
import numpy as np
import pytest

from plume_nav_sim.envs.plume_env import PlumeEnv
from plume_nav_sim.utils.exceptions import StateError, ValidationError

# Test configuration
DEFAULT_TEST_GRID_SIZE = (32, 32)
TEST_SOURCE_LOCATION = (10, 10)


def create_test_environment(*, render_mode: str | None = None) -> PlumeEnv:
    """Create a small PlumeEnv for contract tests."""
    return PlumeEnv(
        grid_size=DEFAULT_TEST_GRID_SIZE,
        source_location=TEST_SOURCE_LOCATION,
        max_steps=25,
        render_mode=render_mode,
    )


class TestPlumeEnvContract:
    """Test Gymnasium API contract compliance."""

    def test_env_can_be_instantiated(self):
        env = create_test_environment()
        try:
            assert isinstance(env, PlumeEnv)
            assert isinstance(env, gym.Env)
        finally:
            env.close()

    def test_spaces_are_initialized(self):
        env = create_test_environment()
        try:
            assert hasattr(env, "action_space")
            assert isinstance(env.action_space, gym.spaces.Space)
            assert hasattr(env, "observation_space")
            assert isinstance(env.observation_space, gym.spaces.Space)
        finally:
            env.close()

    def test_reset_returns_observation_and_info(self):
        env = create_test_environment()
        try:
            obs, info = env.reset(seed=123)
            assert isinstance(obs, np.ndarray)
            assert isinstance(info, dict)
            assert "agent_position" in info
            assert "goal_location" in info
        finally:
            env.close()

    def test_step_returns_five_tuple(self):
        env = create_test_environment()
        try:
            env.reset()
            obs, reward, terminated, truncated, info = env.step(0)
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            assert "step_count" in info
        finally:
            env.close()

    def test_step_requires_reset_first(self):
        env = create_test_environment()
        try:
            with pytest.raises(StateError):
                env.step(0)
        finally:
            env.close()

    def test_invalid_actions_raise_error(self):
        env = create_test_environment()
        try:
            env.reset()
            for invalid_action in [-1, 4, 99, "invalid", None]:
                with pytest.raises(ValidationError):
                    env.step(invalid_action)
        finally:
            env.close()

    def test_render_method_exists(self):
        env = create_test_environment(render_mode="rgb_array")
        try:
            env.reset()
            result = env.render()
            assert result is None or isinstance(result, np.ndarray)
        finally:
            env.close()

    def test_close_is_idempotent(self):
        env = create_test_environment()
        env.reset()
        env.close()
        env.close()

    def test_metadata_dictionary_exists(self):
        env = create_test_environment()
        try:
            assert hasattr(env, "metadata")
            assert isinstance(env.metadata, dict)
            assert "render_modes" in env.metadata
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
