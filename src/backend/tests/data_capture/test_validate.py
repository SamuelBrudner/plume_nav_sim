import gzip
import json
from pathlib import Path

import pytest

import plume_nav_sim as pns
from plume_nav_sim.core.types import EnvironmentConfig
from plume_nav_sim.data_capture.recorder import RunRecorder
from plume_nav_sim.data_capture.validate import validate_run_artifacts
from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper


@pytest.mark.filterwarnings("ignore:.*experimental.*")
def test_pandera_validate_run_artifacts(tmp_path: Path):
    env = pns.make_env(
        grid_size=(8, 8),
        max_steps=5,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
    )
    try:
        cfg = EnvironmentConfig(grid_size=(8, 8), source_location=(4, 4))
        rec = RunRecorder(tmp_path, experiment="valexp")
        wrapped = DataCaptureWrapper(env, rec, cfg)

        obs, info = wrapped.reset(seed=123)
        for _ in range(5):
            obs, reward, term, trunc, info = wrapped.step(env.action_space.sample())
            if term or trunc:
                break
        rec.finalize(export_parquet=False)

        run_dir = next(tmp_path.joinpath("valexp").glob("run-*"))
        report = validate_run_artifacts(run_dir)
        assert report["steps"]["ok"] is True
        assert report["episodes"]["ok"] is True
    finally:
        env.close()


def test_runmeta_includes_system_info(tmp_path: Path):
    # Create recorder directly and write run meta through wrapper init
    env = pns.make_env()
    try:
        cfg = EnvironmentConfig()
        rec = RunRecorder(tmp_path, experiment="sysinfo")
        _ = DataCaptureWrapper(env, rec, cfg)
        run_dir = next(tmp_path.joinpath("sysinfo").glob("run-*"))
        with open(run_dir / "run.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        assert "system" in meta
        assert isinstance(meta["system"].get("hostname"), (str, type(None)))
        assert isinstance(meta["system"].get("python_version"), (str, type(None)))
    finally:
        env.close()
