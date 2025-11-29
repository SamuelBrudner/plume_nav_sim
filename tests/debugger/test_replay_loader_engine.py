from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from plume_nav_sim.data_capture import (
    ReplayEngine,
    ReplayLoadError,
    load_replay_artifacts,
)

FIXTURE_RUN = Path(__file__).resolve().parent.parent / "data" / "replay_fixture"


def _has_pyarrow() -> bool:
    try:
        import importlib.util

        return bool(
            importlib.util.find_spec("pyarrow")
            and importlib.util.find_spec("pyarrow.parquet")
        )
    except Exception:
        return False


def _read_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_fixture_loads_jsonl_and_parquet() -> None:
    artifacts = load_replay_artifacts(FIXTURE_RUN, prefer_parquet=False)

    assert artifacts.source_format == "jsonl"
    assert artifacts.run_meta.run_id.startswith("run-")
    assert len(artifacts.steps) == 6
    assert len(artifacts.episodes) == 2
    assert artifacts.run_meta.env_config.get("max_steps") == 3

    artifacts_parquet = load_replay_artifacts(FIXTURE_RUN, prefer_parquet=True)
    if _has_pyarrow():
        assert artifacts_parquet.source_format == "parquet"
    assert len(artifacts_parquet.steps) == 6
    assert len(artifacts_parquet.episodes) == 2


def test_loader_supports_multipart_segments(tmp_path: Path) -> None:
    run_dir = tmp_path / "replay-multipart"
    shutil.copytree(FIXTURE_RUN, run_dir)

    step_lines = _read_jsonl(run_dir / "steps.jsonl.gz")
    episode_lines = _read_jsonl(run_dir / "episodes.jsonl.gz")

    (run_dir / "steps.jsonl.gz").unlink()
    (run_dir / "episodes.jsonl.gz").unlink()

    for idx, chunk in enumerate((step_lines[:3], step_lines[3:])):
        with gzip.open(
            run_dir / f"steps.part{idx:04d}.jsonl.gz", "wt", encoding="utf-8"
        ) as fh:
            for rec in chunk:
                fh.write(json.dumps(rec) + "\n")

    for idx, chunk in enumerate((episode_lines[:1], episode_lines[1:])):
        with gzip.open(
            run_dir / f"episodes.part{idx:04d}.jsonl.gz", "wt", encoding="utf-8"
        ) as fh:
            for rec in chunk:
                fh.write(json.dumps(rec) + "\n")

    artifacts = load_replay_artifacts(run_dir, prefer_parquet=False)

    assert artifacts.source_format == "jsonl"
    assert len(artifacts.steps) == len(step_lines)
    assert len(artifacts.episodes) == len(episode_lines)
    assert artifacts.steps[0].step == 1
    assert artifacts.steps[-1].step == 3


def test_loader_rejects_schema_version_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "bad-schema"
    shutil.copytree(FIXTURE_RUN, run_dir)

    meta_path = run_dir / "run.json"
    run_meta = json.loads(meta_path.read_text())
    run_meta["schema_version"] = "0.9.9"
    meta_path.write_text(json.dumps(run_meta))

    with pytest.raises(ReplayLoadError) as excinfo:
        load_replay_artifacts(run_dir, prefer_parquet=False)

    assert "schema_version" in str(excinfo.value)


@pytest.mark.headless
def test_replay_engine_matches_recorded_events() -> None:
    artifacts = load_replay_artifacts(FIXTURE_RUN, prefer_parquet=False)

    engine = ReplayEngine(artifacts)
    events = list(engine.iter_events(validate=True))

    assert len(events) == len(artifacts.steps)
    for ev, rec in zip(events, artifacts.steps):
        assert ev.t == rec.step - 1
        assert int(ev.action) == int(rec.action)
        assert ev.terminated == rec.terminated
        assert ev.truncated == rec.truncated
        pos = ev.info.get("agent_position") or ev.info.get("agent_xy")
        assert pos is not None
        assert (int(pos[0]), int(pos[1])) == (
            rec.agent_position.x,
            rec.agent_position.y,
        )
        np.testing.assert_allclose(ev.reward, rec.reward)

    assert {ep.total_steps for ep in artifacts.episodes} == {3}
    assert any(ev.truncated for ev in events)


@pytest.mark.headless
def test_replay_engine_emits_frames_when_rendering() -> None:
    artifacts = load_replay_artifacts(FIXTURE_RUN, prefer_parquet=False)

    events = list(ReplayEngine(artifacts).iter_events(render=True, validate=True))

    assert events, "Replay should emit events"
    frames = [ev.frame for ev in events if ev.frame is not None]
    assert frames, "Frames should be captured when render=True"
    first_frame = frames[0]
    assert isinstance(first_frame, np.ndarray)
    assert first_frame.ndim == 3
    assert first_frame.shape[-1] in (3, 4)
