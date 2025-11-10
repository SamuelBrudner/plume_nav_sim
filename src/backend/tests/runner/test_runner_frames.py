from __future__ import annotations

from typing import Iterator, List

import numpy as np

import plume_nav_sim as pns
from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy


def _make_env(rgb: bool = False):
    return pns.make_env(
        grid_size=(16, 16),
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        max_steps=100,
        render_mode=("rgb_array" if rgb else None),
    )


class _RenderFallbackWrapper:
    """Wrapper whose render() returns None unless explicitly asked for rgb_array.

    This simulates environments that require explicit mode to return a frame.
    """

    def __init__(self, env):
        self._env = env

    # Delegate core Gymnasium API
    def reset(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self._env.reset(*args, **kwargs)

    def step(self, action):  # noqa: ANN001
        return self._env.step(action)

    @property
    def action_space(self):  # noqa: D401
        return self._env.action_space

    @property
    def observation_space(self):  # noqa: D401
        return getattr(self._env, "observation_space", None)

    def render(self, mode: str = "human"):
        if mode == "rgb_array":
            return self._env.render(mode="rgb_array")
        return None


def test_stream_render_fallback_attaches_frame_via_rgb_array():
    """Runner should attach frames via rgb_array when render() returns None."""
    from plume_nav_sim.runner import runner as r

    base = _make_env(rgb=True)
    env = _RenderFallbackWrapper(base)
    policy = TemporalDerivativeDeterministicPolicy()

    it: Iterator = r.stream(env, policy, seed=123, render=True)
    first = next(it)

    frame = getattr(first, "frame", None)
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_run_episode_collects_frames_with_render_true():
    """on_step should see a frame ndarray for each executed step when supported."""
    from plume_nav_sim.runner import runner as r

    env = _make_env(rgb=True)
    policy = TemporalDerivativeDeterministicPolicy()

    frames: List[np.ndarray] = []

    def on_step(ev):
        if ev.frame is not None:
            frames.append(ev.frame)

    res = r.run_episode(
        env,
        policy,
        max_steps=30,
        seed=7,
        on_step=on_step,
        render=True,
    )

    assert res.steps > 0
    assert len(frames) == res.steps
    assert all(
        isinstance(fr, np.ndarray) and fr.ndim == 3 and fr.shape[2] == 3
        for fr in frames
    )
    assert all(fr.dtype == np.uint8 for fr in frames)


def test_run_episode_no_frames_with_render_false():
    """Negative control: render=False yields no frames in on_step events."""
    from plume_nav_sim.runner import runner as r

    env = _make_env(rgb=True)
    policy = TemporalDerivativeDeterministicPolicy()

    seen_frames: List[np.ndarray] = []

    def on_step(ev):
        if ev.frame is not None:
            seen_frames.append(ev.frame)

    res = r.run_episode(
        env, policy, max_steps=10, seed=11, on_step=on_step, render=False
    )

    assert res.steps > 0
    assert seen_frames == []
