"""
Contract tests for BaseEnvironment abstract class.

These tests verify:
1. Gymnasium API compliance (reset, step, render, close)
2. Abstract method enforcement
3. Basic configuration validation

Implementation details (private attributes, error wrapping, performance metrics)
are not tested here - those are covered by concrete environment tests.
"""

import gymnasium as gym
import numpy as np
import pytest

from plume_nav_sim.core.types import (
    EnvironmentConfig,
    create_environment_config,
    create_grid_size,
)
from plume_nav_sim.envs.base_env import BaseEnvironment
from plume_nav_sim.utils.exceptions import ValidationError

# Test configuration
DEFAULT_TEST_GRID_SIZE = (32, 32)
TEST_SOURCE_LOCATION = (10, 10)  # Well within 32x32 grid (valid range 0-31)


def create_test_config() -> EnvironmentConfig:
    """Create a valid test configuration."""
    # Pass tuples directly - EnvironmentConfig will convert them
    return create_environment_config(
        grid_size=DEFAULT_TEST_GRID_SIZE,
        source_location=TEST_SOURCE_LOCATION,
        max_steps=100,
        goal_radius=0.0,
        plume_params={"sigma": 12.0},  # Will use source_location from config
    )


class MinimalConcreteEnvironment(BaseEnvironment):
    """Minimal concrete implementation for testing BaseEnvironment contract."""

    def _reset_environment_state(self) -> None:
        """Reset internal state."""
        pass

    def _process_action(self, action: int) -> None:
        """Process action - no-op for testing."""
        pass

    def _update_environment_state(self) -> None:
        """Update state - no-op for testing."""
        pass

    def _calculate_reward(self) -> float:
        """Return zero reward."""
        return 0.0

    def _check_terminated(self) -> bool:
        """Never terminate."""
        return False

    def _check_truncated(self) -> bool:
        """Never truncate."""
        return False

    def _get_observation(self) -> np.ndarray:
        """Return dummy observation."""
        return np.zeros((2, 2), dtype=np.float32)

    def _create_render_context(self) -> dict:
        """Return empty render context."""
        return {}

    def _create_renderer(self, render_mode: str) -> object:
        """Return None renderer."""
        return None

    def _seed_components(self, seed: int) -> None:
        """Seed components - no-op."""
        pass

    def _cleanup_components(self) -> None:
        """Cleanup - no-op."""
        pass

    def _validate_component_states(self, strict_validation: bool = False) -> bool:
        """Always valid."""
        return True


class TestBaseEnvironmentContract:
    """Test Gymnasium API contract compliance."""

    def test_abstract_class_cannot_be_instantiated(self):
        """BaseEnvironment is abstract and cannot be instantiated directly."""
        config = create_test_config()

        with pytest.raises(TypeError):
            BaseEnvironment(config)

    def test_concrete_implementation_can_be_instantiated(self):
        """Concrete implementations with all abstract methods can be instantiated."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        assert env is not None
        assert isinstance(env, BaseEnvironment)
        assert isinstance(env, gym.Env)

        env.close()

    def test_gymnasium_spaces_are_initialized(self):
        """Environment has valid action_space and observation_space."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        try:
            # Action space should be Discrete(4) for cardinal directions
            assert hasattr(env, "action_space")
            assert isinstance(env.action_space, gym.spaces.Discrete)
            assert env.action_space.n == 4

            # Observation space should be defined
            assert hasattr(env, "observation_space")
            assert isinstance(env.observation_space, gym.spaces.Box)
        finally:
            env.close()

    def test_reset_returns_observation_and_info(self):
        """reset() returns (observation, info) tuple per Gymnasium API."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        try:
            result = env.reset()

            # Should return 2-tuple
            assert isinstance(result, tuple)
            assert len(result) == 2

            obs, info = result

            # Observation should be ndarray
            assert isinstance(obs, np.ndarray)

            # Info should be dict
            assert isinstance(info, dict)
            assert "step_count" in info
        finally:
            env.close()

    def test_step_returns_five_tuple(self):
        """step() returns (obs, reward, terminated, truncated, info) per Gymnasium API."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        try:
            env.reset()
            result = env.step(0)

            # Should return 5-tuple
            assert isinstance(result, tuple)
            assert len(result) == 5

            obs, reward, terminated, truncated, info = result

            # Validate types
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
        finally:
            env.close()

    def test_step_requires_reset_first(self):
        """step() raises error if called before reset()."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        try:
            # Should raise error when stepping without reset
            with pytest.raises(Exception):  # StateError or similar
                env.step(0)
        finally:
            env.close()

    def test_valid_actions_are_accepted(self):
        """Valid actions (0-3) are accepted without error."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        try:
            env.reset()

            # All cardinal directions should work
            for action in [0, 1, 2, 3]:
                result = env.step(action)
                assert len(result) == 5
        finally:
            env.close()

    def test_invalid_actions_raise_error(self):
        """Invalid actions raise appropriate errors."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        try:
            env.reset()

            # Out of bounds actions should raise error
            invalid_actions = [-1, 4, 10, "invalid", None]
            for invalid_action in invalid_actions:
                with pytest.raises(
                    Exception
                ):  # ValidationError, StateError, or similar
                    env.step(invalid_action)
        finally:
            env.close()

    def test_render_method_exists(self):
        """render() method exists and can be called."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        try:
            env.reset()

            # render() should not crash (may return None or array)
            result = env.render()
            # Result type depends on render_mode, just verify it doesn't crash
            assert result is None or isinstance(result, np.ndarray)
        finally:
            env.close()

    def test_close_method_exists(self):
        """close() method exists and can be called."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        env.reset()
        env.close()

        # Should not crash when called multiple times
        env.close()

    def test_metadata_dictionary_exists(self):
        """Environment has metadata dict with render_modes."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        try:
            assert hasattr(env, "metadata")
            assert isinstance(env.metadata, dict)
            assert "render_modes" in env.metadata
        finally:
            env.close()


class TestAbstractMethodEnforcement:
    """Test that abstract methods must be implemented."""

    def test_missing_abstract_method_prevents_instantiation(self):
        """Class missing abstract methods cannot be instantiated."""

        # Create incomplete implementation missing _calculate_reward
        class IncompleteEnvironment(BaseEnvironment):
            def _reset_environment_state(self) -> None:
                pass

            def _process_action(self, action: int) -> None:
                pass

            def _update_environment_state(self) -> None:
                pass

            # Missing: _calculate_reward

            def _check_terminated(self) -> bool:
                return False

            def _check_truncated(self) -> bool:
                return False

            def _get_observation(self) -> np.ndarray:
                return np.zeros((2, 2), dtype=np.float32)

            def _create_render_context(self) -> dict:
                return {}

            def _create_renderer(self, render_mode: str) -> object:
                return None

            def _seed_components(self, seed: int) -> None:
                pass

            def _cleanup_components(self) -> None:
                pass

            def _validate_component_states(
                self, strict_validation: bool = False
            ) -> bool:
                return True

        config = create_test_config()

        # Should raise TypeError for missing abstract method
        with pytest.raises(TypeError):
            IncompleteEnvironment(config)


class TestConfigurationValidation:
    """Test basic configuration validation."""

    def test_valid_configuration_accepted(self):
        """Valid configuration is accepted."""
        config = create_test_config()
        env = MinimalConcreteEnvironment(config)

        assert env is not None
        env.close()

    def test_invalid_grid_size_rejected(self):
        """Invalid grid size raises error."""
        with pytest.raises((ValidationError, ValueError)):
            create_grid_size((0, 0))

    def test_negative_max_steps_rejected(self):
        """Negative max_steps raises error."""
        with pytest.raises((ValidationError, ValueError)):
            create_environment_config(
                grid_size=DEFAULT_TEST_GRID_SIZE,
                source_location=TEST_SOURCE_LOCATION,
                max_steps=-1,
                goal_radius=0.0,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
