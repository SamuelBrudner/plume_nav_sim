from __future__ import annotations

import numpy as np

from plume_nav_sim.data_capture import ReplayEngine, load_replay_artifacts
from plume_nav_sim.data_capture.loader import ReplayArtifacts

from .test_helpers import run_small_capture


def test_replay_matches_recorded_steps(tmp_path) -> None:
    max_steps = 4
    run_dir = run_small_capture(
        tmp_path / "results",
        experiment="replay_match",
        episodes=2,
        max_steps=max_steps,
        export_parquet=False,
    )
    artifacts = load_replay_artifacts(run_dir, prefer_parquet=False)

    assert artifacts.run_meta.env_config.get("max_steps") == max_steps

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


def test_replay_seek_start_step(tmp_path) -> None:
    run_dir = run_small_capture(
        tmp_path / "results",
        experiment="replay_seek",
        episodes=2,
        max_steps=3,
        export_parquet=False,
    )
    artifacts = load_replay_artifacts(run_dir, prefer_parquet=False)

    start = 2 if len(artifacts.steps) > 2 else 0
    engine = ReplayEngine(artifacts)
    events = list(engine.iter_events(start_step=start))

    assert len(events) == len(artifacts.steps) - start
    first_rec = artifacts.steps[start]
    first_ev = events[0]
    assert first_ev.t == first_rec.step - 1
    assert int(first_ev.action) == int(first_rec.action)


def test_replay_emits_frames_when_rendering(tmp_path) -> None:
    run_dir = run_small_capture(
        tmp_path / "results",
        experiment="replay_frames",
        episodes=1,
        max_steps=3,
        export_parquet=False,
    )
    artifacts = load_replay_artifacts(run_dir, prefer_parquet=False)

    engine = ReplayEngine(artifacts)
    events = list(engine.iter_events(render=True))

    assert events, "Replay should emit events"
    assert any(ev.frame is not None for ev in events)

    first_frame = next(ev.frame for ev in events if ev.frame is not None)
    assert isinstance(first_frame, np.ndarray)
    assert first_frame.ndim == 3


def test_replay_infers_missing_max_steps(tmp_path) -> None:
    run_dir = run_small_capture(
        tmp_path / "results",
        experiment="replay_infer_limit",
        episodes=2,
        max_steps=3,
        export_parquet=False,
    )
    artifacts = load_replay_artifacts(run_dir, prefer_parquet=False)

    legacy_env_cfg = dict(artifacts.run_meta.env_config or {})
    legacy_env_cfg.pop("max_steps", None)
    legacy_meta = artifacts.run_meta.model_copy(update={"env_config": legacy_env_cfg})
    legacy_artifacts = ReplayArtifacts(
        run_dir=artifacts.run_dir,
        run_meta=legacy_meta,
        steps=artifacts.steps,
        episodes=artifacts.episodes,
        source_format=artifacts.source_format,
    )

    assert legacy_artifacts.run_meta.env_config.get("max_steps") is None

    engine = ReplayEngine(legacy_artifacts)
    events = list(engine.iter_events(validate=True))

    assert len(events) == len(legacy_artifacts.steps)
