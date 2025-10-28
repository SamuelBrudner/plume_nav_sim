import time
from datetime import datetime, timezone

import pytest

from plume_nav_sim.data_capture.schemas import (
    SCHEMA_VERSION,
    EpisodeRecord,
    Position,
    RunMeta,
    StepRecord,
)


def test_runmeta_validates_schema_version():
    meta = RunMeta(
        run_id="run-1",
        experiment="exp",
        package_version="0.0.1",
        git_sha=None,
        start_time=datetime.now(timezone.utc),
        env_config={"grid_size": (8, 8)},
    )
    assert meta.schema_version == SCHEMA_VERSION

    with pytest.raises(Exception):
        RunMeta(
            schema_version="0.9.9",
            run_id="run-1",
            start_time=datetime.now(timezone.utc),
            env_config={"grid_size": (8, 8)},
        )


def test_steprecord_required_fields_and_validation():
    now = time.time()
    rec = StepRecord(
        ts=now,
        run_id="run-x",
        episode_id="ep-000001",
        step=1,
        action=0,
        reward=0.0,
        terminated=False,
        truncated=False,
        agent_position=Position(x=1, y=2),
        distance_to_goal=1.23,
        observation_summary=[0.5],
        seed=42,
    )
    assert rec.schema_version == SCHEMA_VERSION
    assert rec.step == 1

    with pytest.raises(Exception):
        StepRecord(
            ts=now,
            run_id="run-x",
            episode_id="ep",
            step=0,  # must be positive
            action=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            agent_position=Position(x=0, y=0),
            distance_to_goal=-1.0,  # invalid
        )


def test_episoderecord_fields():
    ep = EpisodeRecord(
        run_id="run-x",
        episode_id="ep-000001",
        terminated=True,
        truncated=False,
        total_steps=10,
        total_reward=1.5,
        final_position=Position(x=3, y=4),
        final_distance_to_goal=0.0,
        duration_ms=100.0,
        avg_step_time_ms=10.0,
    )
    assert ep.terminated is True
    assert ep.total_steps == 10
