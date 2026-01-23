import time

import plume_nav_sim as pns
from plume_nav_sim.data_capture.recorder import RunRecorder
from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper
from plume_nav_sim.envs.config_types import EnvironmentConfig


def _run_steps(env, steps: int) -> float:
    t0 = time.perf_counter()
    obs, info = env.reset(seed=42)
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset(seed=43)
    return (time.perf_counter() - t0) * 1000.0


def test_wrapper_overhead_threshold(tmp_path):
    steps = 200
    env0 = pns.make_env(
        grid_size=(16, 16), action_type="oriented", observation_type="concentration"
    )
    try:
        base_ms = _run_steps(env0, steps)
    finally:
        env0.close()

    env1 = pns.make_env(
        grid_size=(16, 16), action_type="oriented", observation_type="concentration"
    )
    try:
        rec = RunRecorder(tmp_path, experiment="perf")
        cfg = EnvironmentConfig(grid_size=(16, 16), source_location=(8, 8))
        wrapped = DataCaptureWrapper(env1, rec, cfg)
        cap_ms = _run_steps(wrapped, steps)
        rec.finalize(export_parquet=False)
    finally:
        env1.close()

    # Allow generous margin in CI, but guard egregious overhead
    ratio = cap_ms / max(1e-6, base_ms)
    assert ratio < 5.0, f"Data capture overhead too high: {ratio:.2f}x"
