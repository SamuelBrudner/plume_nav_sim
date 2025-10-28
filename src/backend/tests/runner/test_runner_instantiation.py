from __future__ import annotations

import pytest

import plume_nav_sim as pns
from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy
from plume_nav_sim.policies.run_tumble_td import RunTumbleTemporalDerivativePolicy
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
        pol = RunTumbleTemporalDerivativePolicy()
        rr = Runner(env, pol)
        events = list(rr.stream(seed=123, render=False))
        assert len(events) >= 1
        for ev in events:
            assert env.action_space.contains(int(ev.action))
    finally:
        env.close()
