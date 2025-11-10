"""Tests for PlumeSearchEnv wrapper behavior.

Focus: ensure rgb_array render fallback always returns an ndarray even when
the underlying core environment does not produce a frame (returns None).
"""

import numpy as np

from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv

from . import assert_rendering_output_valid


def test_rgb_array_fallback_returns_ndarray_small_grid():
    """When core render returns None, wrapper returns zero ndarray (H,W,3)."""
    env = PlumeSearchEnv(grid_size=(8, 8))  # default render_mode=None
    try:
        # Utility asserts ndarray with 3 channels and dtype uint8
        assert_rendering_output_valid(env, mode="rgb_array")
    finally:
        env.close()


def test_rgb_array_fallback_with_human_mode_configured():
    """Even if configured for human mode, rgb_array requests return ndarray fallback."""
    env = PlumeSearchEnv(grid_size=(16, 16), render_mode="human")
    try:
        frame = env.render(mode="rgb_array")
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[-1] == 3
        assert frame.dtype == np.uint8
    finally:
        env.close()
