from __future__ import annotations

import sys
from pathlib import Path

import pytest

import plume_nav_sim as pns
from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy

# Expose demo package policy
_demo_path = Path(__file__).resolve().parents[4] / "plug-and-play-demo"
if _demo_path.is_dir():
    sys.path.append(str(_demo_path))

plug_demo = pytest.importorskip("plug_and_play_demo")
DeltaBasedRunTumblePolicy = plug_demo.DeltaBasedRunTumblePolicy
from plume_nav_sim.runner.runner import Runner


def _env(action_type: str):
    return pns.make_env(
        grid_size=(8, 8),
        source_location=(4, 4),
        max_steps=10,
        action_type=action_type,
        observation_type="concentration",
        reward_type="step_penalty",
        render_mode=None,
    )


def test_runner_instantiation_validates_superset_pair():
    # Oriented policy (n=3) on run_tumble env (n=2) should raise at construction
    env = _env("run_tumble")
    try:
        pol = TemporalDerivativeDeterministicPolicy()
        with pytest.raises(ValueError):
            _ = Runner(env, pol)
    finally:
        env.close()


def test_runner_instantiation_allows_subset_pair_and_streams():
    # Run/Tumble policy (n=2) on oriented env (n=3) should construct and stream
    env = _env("oriented")
    try:
        pol = DeltaBasedRunTumblePolicy()
        rr = Runner(env, pol)
        events = list(rr.stream(seed=123, render=False))
        assert len(events) >= 1
        for ev in events:
            assert env.action_space.contains(int(ev.action))
    finally:
        env.close()
