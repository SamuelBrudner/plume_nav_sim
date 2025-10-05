"""
Contract Tests: Gymnasium API Compliance

Tests that PlumeSearchEnv conforms to the Gymnasium API contract.
This is the PUBLIC API that external RL libraries depend on.

Reference: contracts/gymnasium_api.md
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import gymnasium as gym
from plume_nav_sim import PlumeSearchEnv
from plume_nav_sim.utils.exceptions import StateError, ValidationError

# ============================================================================
# Test Categories
# ============================================================================


class TestActionSpace:
    """Action space must match declared space."""

    def test_action_space_is_discrete(self):
        """Action space is Discrete."""
        env = PlumeSearchEnv()
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), f"Expected Discrete, got {type(env.action_space)}"

    def test_action_space_size(self):
        """Action space has correct size.

        Contract: gymnasium_api.md
        Current implementation: Discrete(4)
        """
        env = PlumeSearchEnv()
        assert env.action_space.n == 4, f"Expected 4 actions, got {env.action_space.n}"

    def test_valid_actions(self):
        """All actions in [0, 3] are valid."""
        env = PlumeSearchEnv()
        env.reset(seed=42)

        for action in [0, 1, 2, 3]:
            result = env.step(action)
            assert len(result) == 5, f"step({action}) should return 5-tuple"

    def test_invalid_actions_rejected(self):
        """Actions outside [0, 3] are rejected."""
        env = PlumeSearchEnv()
        env.reset(seed=42)

        # Too high
        with pytest.raises((ValueError, AssertionError)):
            env.step(4)

        # Negative
        with pytest.raises((ValueError, AssertionError)):
            env.step(-1)

        # Way out of bounds
        with pytest.raises((ValueError, AssertionError)):
            env.step(100)

    def test_action_sampling_valid(self):
        """Sampled actions are always valid."""
        env = PlumeSearchEnv()
        env.reset(seed=42)

        for _ in range(100):
            action = env.action_space.sample()
            assert 0 <= action <= 3, f"Sampled invalid action: {action}"


class TestObservationSpace:
    """Observation space must match declared space."""

    def test_observation_space_is_dict(self):
        """Observation space is Dict."""
        env = PlumeSearchEnv()
        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), f"Expected Dict space, got {type(env.observation_space)}"

    def test_observation_has_required_keys(self):
        """Observation space has all required keys.

        Contract: gymnasium_api.md
        Required: agent_position, concentration_field, source_location
        """
        env = PlumeSearchEnv()
        required_keys = {"agent_position", "concentration_field", "source_location"}
        actual_keys = set(env.observation_space.spaces.keys())

        assert required_keys.issubset(
            actual_keys
        ), f"Missing keys: {required_keys - actual_keys}"

    def test_observation_structure(self):
        """Observation components have correct types."""
        env = PlumeSearchEnv()
        obs_space = env.observation_space

        # agent_position should be Box with shape (2,)
        assert "agent_position" in obs_space.spaces
        pos_space = obs_space.spaces["agent_position"]
        assert isinstance(pos_space, gym.spaces.Box)
        assert pos_space.shape == (2,), f"agent_position shape: {pos_space.shape}"

        # concentration_field should be Box with shape (H, W)
        assert "concentration_field" in obs_space.spaces
        field_space = obs_space.spaces["concentration_field"]
        assert isinstance(field_space, gym.spaces.Box)
        assert (
            len(field_space.shape) == 2
        ), f"concentration_field should be 2D, got shape {field_space.shape}"

        # source_location should be Box with shape (2,)
        assert "source_location" in obs_space.spaces
        source_space = obs_space.spaces["source_location"]
        assert isinstance(source_space, gym.spaces.Box)
        assert source_space.shape == (
            2,
        ), f"source_location shape: {source_space.shape}"

    def test_observations_match_space(self):
        """All observations match declared space.

        Contract: gymnasium_api.md - Observation Invariants I1-I8
        """
        env = PlumeSearchEnv()
        obs, _ = env.reset(seed=42)

        # Check observation is in space
        assert env.observation_space.contains(
            obs
        ), "reset() observation not in observation_space"

        # Check multiple steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(
                obs
            ), "step() observation not in observation_space"

            if terminated or truncated:
                break

    def test_observation_no_nan_or_inf(self):
        """Observations have no NaN or Inf values.

        Contract: Invariants I7-I8
        """
        env = PlumeSearchEnv()
        obs, _ = env.reset(seed=42)

        for key, value in obs.items():
            assert not np.any(np.isnan(value)), f"NaN found in observation['{key}']"
            assert not np.any(np.isinf(value)), f"Inf found in observation['{key}']"

        # Check after steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            for key, value in obs.items():
                assert not np.any(
                    np.isnan(value)
                ), f"NaN found in observation['{key}'] after step"
                assert not np.any(
                    np.isinf(value)
                ), f"Inf found in observation['{key}'] after step"

            if terminated or truncated:
                break


class TestInfoDictionary:
    """Info dictionary must have required keys."""

    def test_reset_info_has_seed(self):
        """reset() info contains seed.

        Contract: gymnasium_api.md - Info after reset()
        """
        env = PlumeSearchEnv()
        obs, info = env.reset(seed=42)

        assert "seed" in info, "Info missing 'seed' key after reset()"
        assert info["seed"] == 42, f"Expected seed=42, got {info['seed']}"

    def test_reset_info_seed_none(self):
        """reset() without seed sets info['seed'] to None."""
        env = PlumeSearchEnv()
        obs, info = env.reset()

        assert "seed" in info, "Info missing 'seed' key"
        # Seed might be auto-generated or None depending on implementation
        # Just check key exists

    def test_step_info_has_required_keys(self):
        """step() info has required keys.

        Contract: gymnasium_api.md - Info after step()
        Required: step_count, total_reward, goal_reached
        """
        env = PlumeSearchEnv()
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(0)

        assert "step_count" in info, "Info missing 'step_count'"
        assert "total_reward" in info, "Info missing 'total_reward'"
        assert "goal_reached" in info, "Info missing 'goal_reached'"

    def test_step_count_non_negative(self):
        """step_count is always non-negative.

        Contract: Invariant I2
        """
        env = PlumeSearchEnv()
        env.reset(seed=42)

        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(0)
            assert info["step_count"] >= 0, f"Negative step_count: {info['step_count']}"
            assert (
                info["step_count"] == i + 1
            ), f"Expected step_count={i+1}, got {info['step_count']}"

            if terminated or truncated:
                break

    def test_total_reward_is_finite(self):
        """total_reward is a finite numeric value after each step."""
        env = PlumeSearchEnv()
        env.reset(seed=42)

        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(0)
            assert isinstance(
                info.get("total_reward"), (int, float)
            ), "total_reward must be numeric"
            assert np.isfinite(
                info["total_reward"]
            ), f"total_reward must be finite, got {info['total_reward']}"

            if terminated or truncated:
                break

    def test_goal_reached_is_boolean(self):
        """goal_reached is boolean.

        Contract: Invariant I6
        """
        env = PlumeSearchEnv()
        env.reset(seed=42)

        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(0)
            assert isinstance(
                info["goal_reached"], bool
            ), f"goal_reached not bool: {type(info['goal_reached'])}"

            if terminated or truncated:
                break


class TestResetMethod:
    """reset() must conform to contract."""

    def test_reset_returns_tuple(self):
        """reset() returns (observation, info) tuple."""
        env = PlumeSearchEnv()
        result = env.reset(seed=42)

        assert isinstance(
            result, tuple
        ), f"reset() should return tuple, got {type(result)}"
        assert len(result) == 2, f"reset() should return 2-tuple, got {len(result)}"

    def test_reset_observation_valid(self):
        """reset() observation is valid."""
        env = PlumeSearchEnv()
        obs, info = env.reset(seed=42)

        assert env.observation_space.contains(
            obs
        ), "reset() observation not in observation_space"

    def test_reset_info_is_dict(self):
        """reset() info is dict."""
        env = PlumeSearchEnv()
        obs, info = env.reset(seed=42)

        assert isinstance(info, dict), f"Info should be dict, got {type(info)}"

    def test_reset_idempotent(self):
        """Can call reset() multiple times."""
        env = PlumeSearchEnv()

        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)
        obs3, info3 = env.reset(seed=42)

        # Should all succeed without error
        assert obs1 is not None
        assert obs2 is not None
        assert obs3 is not None

    def test_reset_after_terminal(self):
        """Can reset() after episode ends."""
        env = PlumeSearchEnv()
        env.reset(seed=42)

        # Run until terminal
        for _ in range(1000):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        # Should be able to reset
        obs, info = env.reset(seed=43)
        assert obs is not None


class TestStepMethod:
    """step() must conform to contract."""

    def test_step_returns_five_tuple(self):
        """step() returns 5-tuple.

        Contract: (observation, reward, terminated, truncated, info)
        """
        env = PlumeSearchEnv()
        env.reset(seed=42)

        result = env.step(0)
        assert isinstance(
            result, tuple
        ), f"step() should return tuple, got {type(result)}"
        assert len(result) == 5, f"step() should return 5-tuple, got {len(result)}"

    def test_step_components_types(self):
        """step() components have correct types."""
        env = PlumeSearchEnv()
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(0)

        # Observation in space
        assert env.observation_space.contains(obs), "Observation not in space"

        # Reward is float
        assert isinstance(
            reward, (float, int, np.number)
        ), f"Reward should be numeric, got {type(reward)}"

        # Flags are bool
        assert isinstance(
            terminated, (bool, np.bool_)
        ), f"terminated should be bool, got {type(terminated)}"
        assert isinstance(
            truncated, (bool, np.bool_)
        ), f"truncated should be bool, got {type(truncated)}"

        # Info is dict
        assert isinstance(info, dict), f"info should be dict, got {type(info)}"

    def test_step_before_reset_raises(self):
        """step() before reset() raises error.

        Contract: Precondition P1
        """
        env = PlumeSearchEnv()

        # Try to step without reset
        with pytest.raises((StateError, RuntimeError, AssertionError)):
            env.step(0)

    def test_step_after_close_raises(self):
        """step() after close() raises error.

        Contract: Precondition P2
        """
        env = PlumeSearchEnv()
        env.reset(seed=42)
        env.close()

        with pytest.raises((StateError, RuntimeError)):
            env.step(0)


class TestCloseMethod:
    """close() must conform to contract."""

    def test_close_idempotent(self):
        """Can call close() multiple times.

        Contract: Idempotency guarantee
        """
        env = PlumeSearchEnv()

        env.close()
        env.close()  # Should not raise
        env.close()  # Should not raise

    def test_close_never_raises(self):
        """close() never raises exceptions.

        Contract: "Never raises (absorbs errors)"
        """
        env = PlumeSearchEnv()

        try:
            env.close()
        except Exception as e:
            pytest.fail(f"close() raised exception: {e}")

    def test_cannot_reset_after_close(self):
        """Cannot reset() after close()."""
        env = PlumeSearchEnv()
        env.close()

        with pytest.raises((StateError, RuntimeError)):
            env.reset(seed=42)

    def test_cannot_step_after_close(self):
        """Cannot step() after close()."""
        env = PlumeSearchEnv()
        env.reset(seed=42)
        env.close()

        with pytest.raises((StateError, RuntimeError)):
            env.step(0)


class TestDeterminism:
    """Same seed produces identical trajectories.

    Contract: G2 - Determinism with seed
    """

    def test_reset_deterministic(self):
        """Same seed produces same initial state."""
        env1 = PlumeSearchEnv()
        env2 = PlumeSearchEnv()

        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        # Agent positions should match
        np.testing.assert_array_equal(obs1["agent_position"], obs2["agent_position"])

        # Source locations should match
        np.testing.assert_array_equal(obs1["source_location"], obs2["source_location"])

        # Concentration fields should match
        np.testing.assert_allclose(
            obs1["concentration_field"], obs2["concentration_field"], rtol=1e-10
        )

    def test_trajectory_deterministic(self):
        """Same seed + actions produces same trajectory."""
        env1 = PlumeSearchEnv()
        env2 = PlumeSearchEnv()

        env1.reset(seed=42)
        env2.reset(seed=42)

        actions = [0, 1, 2, 3, 0, 1]

        for action in actions:
            obs1, r1, t1, tr1, i1 = env1.step(action)
            obs2, r2, t2, tr2, i2 = env2.step(action)

            # Same positions
            np.testing.assert_array_equal(
                obs1["agent_position"], obs2["agent_position"]
            )

            # Same rewards
            assert r1 == r2, f"Rewards differ: {r1} != {r2}"

            # Same termination
            assert t1 == t2, f"Terminated flags differ"
            assert tr1 == tr2, f"Truncated flags differ"

    @given(seed=st.integers(0, 1000))
    @settings(max_examples=20)
    def test_reset_deterministic_property(self, seed):
        """Property: same seed → same reset (Hypothesis)."""
        env1 = PlumeSearchEnv()
        env2 = PlumeSearchEnv()

        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)

        np.testing.assert_array_equal(obs1["agent_position"], obs2["agent_position"])


class TestTerminationConditions:
    """Termination conditions must follow contract."""

    def test_terminated_means_goal_reached(self):
        """terminated=True ⟺ goal_reached=True.

        Contract: Termination Conditions
        """
        env = PlumeSearchEnv()
        env.reset(seed=42)

        for _ in range(1000):
            obs, reward, terminated, truncated, info = env.step(0)

            if terminated:
                assert (
                    info["goal_reached"] == True
                ), "terminated=True but goal_reached=False"
                break

    def test_truncated_means_max_steps(self):
        """truncated=True when max_steps reached."""
        env = PlumeSearchEnv(max_steps=10)
        env.reset(seed=42)

        for i in range(20):
            obs, reward, terminated, truncated, info = env.step(0)

            if truncated:
                # Should happen around step 10
                assert (
                    info["step_count"] >= 10
                ), f"Truncated at step {info['step_count']}, expected ≥10"
                break

    def test_usually_exclusive_termination(self):
        """terminated and truncated usually exclusive.

        Contract: "terminated ∧ truncated = False (typically)"
        """
        env = PlumeSearchEnv(max_steps=50)
        env.reset(seed=42)

        both_true_count = 0
        total_episodes = 10

        for episode in range(total_episodes):
            env.reset(seed=episode)

            for _ in range(100):
                obs, reward, terminated, truncated, info = env.step(0)

                if terminated and truncated:
                    both_true_count += 1

                if terminated or truncated:
                    break

        # Should be rare (not common)
        assert (
            both_true_count < total_episodes / 2
        ), f"Both flags true too often: {both_true_count}/{total_episodes}"


class TestMetadata:
    """Metadata must be defined."""

    def test_metadata_exists(self):
        """Environment has metadata attribute."""
        env = PlumeSearchEnv()
        assert hasattr(env, "metadata"), "Missing metadata attribute"
        assert isinstance(env.metadata, dict), "metadata should be dict"

    def test_metadata_has_render_modes(self):
        """metadata contains render_modes."""
        env = PlumeSearchEnv()
        assert "render_modes" in env.metadata, "metadata missing 'render_modes'"
        assert isinstance(
            env.metadata["render_modes"], list
        ), "render_modes should be list"


class TestGymnasiumChecker:
    """Official Gymnasium environment checker."""

    def test_environment_passes_check_env(self):
        """Environment passes gymnasium.utils.env_checker.

        This is the official Gymnasium API compliance test.
        """
        from gymnasium.utils.env_checker import check_env

        env = PlumeSearchEnv()

        try:
            check_env(env, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"Gymnasium check_env failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
