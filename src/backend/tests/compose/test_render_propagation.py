from __future__ import annotations

import numpy as np
import pytest

from plume_nav_sim.compose import SimulationSpec, prepare
from plume_nav_sim.envs.plume_search_env import unwrap_to_plume_env


@pytest.mark.parametrize("grid_size", [(8, 8), (128, 128)])
@pytest.mark.parametrize("render_flag", [True, False])
def test_simulation_spec_render_propagates_and_rgb_array_shapes(grid_size, render_flag):
    sim = SimulationSpec(
        grid_size=grid_size,
        max_steps=10,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        render=render_flag,
        seed=123,
    )

    env, _ = prepare(sim)

    # Unwrap to core ComponentBasedEnvironment and verify render_mode propagation
    plume_env = unwrap_to_plume_env(env)
    inner = plume_env._core_env  # ComponentBasedEnvironment

    if render_flag:
        assert (
            getattr(inner, "render_mode", None) == "rgb_array"
        ), "render=True should set inner.render_mode to 'rgb_array'"
    else:
        assert (
            getattr(inner, "render_mode", None) is None
        ), "render=False should leave inner.render_mode as None"

    # RGB render should return an ndarray with expected shape/dtype
    frame = env.render("rgb_array")
    assert isinstance(frame, np.ndarray), "env.render('rgb_array') must return ndarray"
    h, w = grid_size[1], grid_size[0]
    assert frame.shape == (
        h,
        w,
        3,
    ), f"frame shape must be (H,W,3) matching grid {grid_size}, got {frame.shape}"
    assert frame.dtype == np.uint8, "frame dtype must be uint8"

    env.close()
