from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from plume_nav_sim.data_capture.loader import ReplayLoadError, load_replay_artifacts
from plume_nav_sim.data_capture.schemas import SCHEMA_VERSION

from .test_helpers import run_small_capture


def _has_pyarrow() -> bool:
    try:
        import importlib.util

        return bool(
            importlib.util.find_spec("pyarrow")
            and importlib.util.find_spec("pyarrow.parquet")
        )
    except Exception:
        return False


def test_loads_jsonl_with_schema_validation(tmp_path: Path) -> None:
    run_dir = run_small_capture(
        tmp_path / "results",
        experiment="loader_jsonl",
        episodes=1,
        max_steps=5,
        export_parquet=False,
    )

    artifacts = load_replay_artifacts(run_dir, prefer_parquet=False)

    assert artifacts.source_format == "jsonl"
    assert artifacts.run_dir == run_dir
    assert artifacts.run_meta.run_id == run_dir.name
    assert len(artifacts.steps) > 0
    assert len(artifacts.episodes) > 0
    assert all(step.schema_version == SCHEMA_VERSION for step in artifacts.steps)
    assert all(ep.schema_version == SCHEMA_VERSION for ep in artifacts.episodes)


def test_loads_parquet_when_available(tmp_path: Path) -> None:
    if not _has_pyarrow():
        pytest.skip("pyarrow not available; Parquet loader path not exercised")

    run_dir = run_small_capture(
        tmp_path / "results",
        experiment="loader_parquet",
        episodes=1,
        max_steps=5,
        export_parquet=True,
    )

    steps_parquet = run_dir / "steps.parquet"
    episodes_parquet = run_dir / "episodes.parquet"
    if not (steps_parquet.exists() and episodes_parquet.exists()):
        pytest.skip("Parquet artifacts were not produced for this run")

    artifacts = load_replay_artifacts(run_dir, prefer_parquet=True)

    assert artifacts.source_format == "parquet"
    assert len(artifacts.steps) > 0
    assert len(artifacts.episodes) > 0
    assert artifacts.steps[0].schema_version == SCHEMA_VERSION
    assert artifacts.episodes[0].schema_version == SCHEMA_VERSION


def test_schema_version_mismatch_is_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad" / "run-000001"
    run_dir.mkdir(parents=True)

    bad_meta = {
        "schema_version": "0.9.0",
        "run_id": "run-000001",
        "experiment": "bad",
        "package_version": None,
        "git_sha": None,
        "start_time": "2025-01-01T00:00:00Z",
        "env_config": {
            "grid_size": [4, 4],
            "source_location": [0, 0],
            "max_steps": 5,
            "goal_radius": 1.0,
            "enable_rendering": True,
            "plume_params": {"source_location": [0, 0], "sigma": 1.0},
        },
        "system": {},
    }
    with open(run_dir / "run.json", "w", encoding="utf-8") as fh:
        json.dump(bad_meta, fh)

    step_obj = {
        "schema_version": SCHEMA_VERSION,
        "ts": 0.0,
        "run_id": "run-000001",
        "episode_id": "ep-000001",
        "step": 1,
        "action": 0,
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "agent_position": {"x": 0, "y": 0},
        "distance_to_goal": 0.0,
    }
    with gzip.open(run_dir / "steps.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(step_obj) + "\n")

    episode_obj = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-000001",
        "episode_id": "ep-000001",
        "terminated": False,
        "truncated": False,
        "total_steps": 1,
        "total_reward": 0.0,
        "final_position": {"x": 0, "y": 0},
        "final_distance_to_goal": 0.0,
    }
    with gzip.open(run_dir / "episodes.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(episode_obj) + "\n")

    with pytest.raises(ReplayLoadError) as excinfo:
        load_replay_artifacts(run_dir)

    msg = str(excinfo.value)
    assert "schema_version" in msg
    assert "run.json" in msg
