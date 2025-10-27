from __future__ import annotations

import pytest

import plume_nav_sim as pns
from plume_nav_sim.policies import (
    TemporalDerivativeDeterministicPolicy,
    TemporalDerivativePolicy,
)
from plume_nav_sim.policies.run_tumble_td import RunTumbleTemporalDerivativePolicy
from plume_nav_sim.runner import runner as r


def _make_env(action_type: str):
    return pns.make_env(
        grid_size=(8, 8),
        source_location=(4, 4),
        max_steps=5,
        action_type=action_type,
        observation_type="concentration",
        reward_type="step_penalty",
        render_mode=None,
    )


@pytest.mark.parametrize(
    "policy_ctor",
    [
        lambda: TemporalDerivativeDeterministicPolicy(),
        lambda: TemporalDerivativePolicy(),
        lambda: TemporalDerivativePolicy(uniform_random_on_non_increase=True),
    ],
)
def test_oriented_policies_match_oriented_env(policy_ctor):
    env = _make_env("oriented")
    try:
        pol = policy_ctor()
        assert getattr(env.action_space, "n", None) == getattr(
            pol.action_space, "n", None
        )
        # Should be able to stream at least one step without error
        events = list(r.stream(env, pol, seed=123, render=False))
        assert len(events) >= 1
        for ev in events:
            assert env.action_space.contains(int(ev.action))
    finally:
        env.close()


@pytest.mark.parametrize(
    "policy_ctor",
    [
        lambda: RunTumbleTemporalDerivativePolicy(),
        lambda: RunTumbleTemporalDerivativePolicy(eps=0.1, eps_seed=7),
    ],
)
def test_run_tumble_policies_match_run_tumble_env(policy_ctor):
    env = _make_env("run_tumble")
    try:
        pol = policy_ctor()
        assert getattr(env.action_space, "n", None) == getattr(
            pol.action_space, "n", None
        )
        events = list(r.stream(env, pol, seed=123, render=False))
        assert len(events) >= 1
        for ev in events:
            assert env.action_space.contains(int(ev.action))
    finally:
        env.close()


@pytest.mark.parametrize(
    "env_type, policy_ctor",
    [
        ("run_tumble", lambda: TemporalDerivativeDeterministicPolicy()),
        ("run_tumble", lambda: TemporalDerivativePolicy()),
        ("oriented", lambda: RunTumbleTemporalDerivativePolicy()),
    ],
)
def test_mismatched_policy_env_pairs_raise(env_type, policy_ctor):
    env = _make_env(env_type)
    try:
        pol = policy_ctor()
        with pytest.raises(ValueError):
            _ = next(r.stream(env, pol, seed=1, render=False))
    finally:
        env.close()
