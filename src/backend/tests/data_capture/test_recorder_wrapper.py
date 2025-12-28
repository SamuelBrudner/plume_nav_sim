import gzip
import json
from pathlib import Path

import plume_nav_sim as pns
from plume_nav_sim.core.types import EnvironmentConfig
from plume_nav_sim.data_capture.recorder import RunRecorder
from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper


def test_wrapper_records_steps_and_episode(tmp_path: Path):
    env = pns.make_env(
        grid_size=(8, 8),
        max_steps=5,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
    )
    try:
        cfg = EnvironmentConfig(grid_size=(8, 8), source_location=(4, 4))
        rec = RunRecorder(tmp_path, experiment="testexp")
        wrapped = DataCaptureWrapper(env, rec, cfg)

        obs, info = wrapped.reset(seed=123)
        steps = 0
        while steps < 5:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped.step(action)
            steps += 1
            if terminated or truncated:
                break

        rec.finalize(export_parquet=False)

        # Verify files exist and contain records
        run_dir = next(tmp_path.joinpath("testexp").glob("run-*"))
        steps_file = run_dir / "steps.jsonl.gz"
        episodes_file = run_dir / "episodes.jsonl.gz"
        assert steps_file.exists()
        assert episodes_file.exists()

        with gzip.open(steps_file, "rt", encoding="utf-8") as fh:
            step_lines = [json.loads(line) for line in fh if line.strip()]
        assert len(step_lines) >= 1
        # Optional: last record terminated or truncated
        assert isinstance(step_lines[-1]["terminated"], bool)

        with gzip.open(episodes_file, "rt", encoding="utf-8") as fh:
            ep_lines = [json.loads(line) for line in fh if line.strip()]
        assert len(ep_lines) >= 1
        assert "total_steps" in ep_lines[-1]
    finally:
        env.close()
