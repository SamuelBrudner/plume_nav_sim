from __future__ import annotations

import json
from pathlib import Path

from plume_nav_sim.cli import capture
from plume_nav_sim.data_capture import RunRecorder
from plume_nav_sim.data_capture.wrapper import DataCaptureWrapper
from plume_nav_sim import make_env
from plume_nav_sim.envs.config_types import EnvironmentConfig


def _latest_run_dir(output_root: Path, experiment: str) -> Path:
    exp_dir = output_root / experiment
    run_dirs = sorted(path for path in exp_dir.iterdir() if path.is_dir())
    assert run_dirs, f"no run directories under {exp_dir}"
    return run_dirs[-1]


def test_environment_config_to_dict_is_json_safe() -> None:
    cfg = EnvironmentConfig(grid_size=(8, 8), source_location=(0, 0))

    payload = cfg.to_dict()

    assert payload["plume_params"]["source_location"] == (0, 0)
    json.dumps(payload)


def test_data_capture_wrapper_handles_partial_meta_overrides(tmp_path: Path) -> None:
    env = make_env(
        grid_size=(8, 8),
        goal_location=(4, 4),
        max_steps=1,
    )
    rec = RunRecorder(tmp_path, experiment="wrapper")
    cfg = EnvironmentConfig(grid_size=(8, 8), source_location=(4, 4), max_steps=1)

    wrapped = DataCaptureWrapper(
        env,
        rec,
        cfg,
        meta_overrides={"config_hash": "abc123"},
    )
    try:
        wrapped.reset(seed=0)
        wrapped.step(0)
        rec.finalize()
    finally:
        wrapped.close()

    run_dir = _latest_run_dir(tmp_path, "wrapper")
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert meta["config_hash"] == "abc123"
    assert meta["env_config"]["plume_params"]["source_location"] == [4, 4]
    assert meta["extra"]["simulation_metadata"]["software_name"] == "plume-nav-sim"


def test_capture_main_smoke_supports_parquet(tmp_path: Path) -> None:
    output_root = tmp_path / "results"

    rc = capture.main(
        [
            "--output",
            str(output_root),
            "--experiment",
            "smoke",
            "--episodes",
            "1",
            "--grid",
            "8x8",
            "--max-steps",
            "5",
            "--parquet",
        ]
    )

    assert rc == 0

    run_dir = _latest_run_dir(output_root, "smoke")
    assert (run_dir / "run.json").exists()
    assert (run_dir / "steps.jsonl.gz").exists()
    assert (run_dir / "episodes.jsonl.gz").exists()

    try:
        import pyarrow  # noqa: F401
    except Exception:
        assert not (run_dir / "steps.parquet").exists()
        assert not (run_dir / "episodes.parquet").exists()
    else:
        assert (run_dir / "steps.parquet").exists()
        assert (run_dir / "episodes.parquet").exists()
