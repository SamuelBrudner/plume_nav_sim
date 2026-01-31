"""Tests for PlumeEnv behavior.

Focus:
- rgb_array rendering returns an ndarray regardless of configured render_mode.
- seeded resets produce reproducible trajectories across instances.
- constructor arguments are normalized onto public attributes, and reward
  semantics expose immediate reward while tracking cumulative reward in info.
"""

import numpy as np

from plume_nav_sim.envs.plume_env import PlumeEnv


def assert_rendering_output_valid(env: PlumeEnv, mode: str = "rgb_array") -> None:
    """Minimal rendering validation for rgb_array output."""
    frame = env.render(mode=mode)
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[-1] == 3
    assert frame.dtype == np.uint8


def test_rgb_array_fallback_returns_ndarray_small_grid():
    """rgb_array render returns ndarray when render_mode is unset."""
    env = PlumeEnv(grid_size=(8, 8))  # default render_mode=None
    try:
        # Utility asserts ndarray with 3 channels and dtype uint8
        assert_rendering_output_valid(env, mode="rgb_array")
    finally:
        env.close()


def test_rgb_array_fallback_with_human_mode_configured():
    """Even if configured for human mode, rgb_array requests return ndarray."""
    env = PlumeEnv(grid_size=(16, 16), render_mode="human")
    try:
        frame = env.render(mode="rgb_array")
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[-1] == 3
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_wrapper_attribute_reflection_and_normalization():
    """Ensure attributes mirror normalized constructor arguments."""
    # goal_radius=0 should normalize to a small positive epsilon
    env = PlumeEnv(
        grid_size=(20, 30),
        source_location=(5, 10),
        max_steps=50,
        goal_radius=0.0,
    )
    try:
        assert env.grid_size == (20, 30)
        assert env.source_location == (5, 10)
        assert env.max_steps == 50
        assert isinstance(env.goal_radius, float)
        assert env.goal_radius > 0.0
    finally:
        env.close()


def test_seeded_reproducibility_same_trajectory():
    """Two instances with same config and seed follow identical trajectories."""
    cfg = dict(grid_size=(32, 32), source_location=(16, 16), max_steps=25)
    env1 = PlumeEnv(**cfg)
    env2 = PlumeEnv(**cfg)

    try:
        obs1, info1 = env1.reset(seed=1234)
        obs2, info2 = env2.reset(seed=1234)

        # Observations are Box(1,) in the wrapper; compare elementwise
        np.testing.assert_array_equal(obs1, obs2)
        assert info1.get("seed") == info2.get("seed") == 1234

        actions = [0, 1, 2, 3, 0, 1]
        cum1 = 0.0
        cum2 = 0.0
        for a in actions:
            o1, r1, t1, tr1, i1 = env1.step(a)
            o2, r2, t2, tr2, i2 = env2.step(a)

            np.testing.assert_array_equal(o1, o2)
            assert r1 == r2
            assert t1 == t2
            assert tr1 == tr2

            cum1 += float(r1)
            cum2 += float(r2)
            assert i1.get("total_reward") == cum1
            assert i2.get("total_reward") == cum2

            if t1 or tr1:
                break
    finally:
        env1.close()
        env2.close()


def test_step_returns_immediate_reward_and_tracks_cumulative_in_info():
    env = PlumeEnv(grid_size=(16, 16), source_location=(8, 8), max_steps=10)
    try:
        _, info = env.reset(seed=7)
        assert info.get("total_reward") == 0.0

        obs, r1, term, trunc, info1 = env.step(0)
        assert isinstance(r1, (float, int))
        assert info1.get("total_reward") == float(r1)

        _, r2, _, _, info2 = env.step(1)
        assert info2.get("total_reward") == float(r1 + r2)
    finally:
        env.close()
