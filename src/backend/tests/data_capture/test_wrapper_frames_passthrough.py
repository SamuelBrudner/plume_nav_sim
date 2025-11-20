from __future__ import annotations

from pathlib import Path

import numpy as np

import plume_nav_sim as pns
from plume_nav_sim.core.types import EnvironmentConfig
from plume_nav_sim.data_capture.recorder import RunRecorder
from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper
from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy
from plume_nav_sim.runner import runner as r


def _create_wrapped_env(tmp_path: Path):
    env = pns.make_env(
        grid_size=(16, 16),
        max_steps=50,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        render_mode="rgb_array",
    )
    cfg = EnvironmentConfig(grid_size=(16, 16), source_location=(8, 8))
    rec = RunRecorder(tmp_path, experiment="frames")
    wrapped = DataCaptureWrapper(env, rec, cfg)
    return wrapped, rec


def test_render_passthrough_returns_frame(tmp_path: Path):
    """DataCaptureWrapper must not suppress env.render frames (passthrough)."""
    env, rec = _create_wrapped_env(tmp_path)
    try:
        obs, info = env.reset(seed=42)
        frame = env.render("rgb_array")
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
    finally:
        # Ensure files closed cleanly even if assertions fail
        rec.finalize(export_parquet=False)
        env.close()


def test_runner_attaches_frames_with_wrapper(tmp_path: Path):
    """Runner frame attachment works when env is wrapped by DataCaptureWrapper."""
    env, rec = _create_wrapped_env(tmp_path)
    try:
        policy = TemporalDerivativeDeterministicPolicy()
        it = r.stream(env, policy, seed=7, render=True)
        first = next(it)
        frame = getattr(first, "frame", None)
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
    finally:
        rec.finalize(export_parquet=False)
        env.close()
