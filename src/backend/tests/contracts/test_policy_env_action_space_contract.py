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


def _random_action_steps_succeed(env, pol, n: int = 16):
    """Sample random actions from the policy's action_space and step env.

    Asserts that all sampled actions are accepted by the env's action_space and
    that env.step(action) executes without raising.
    """
    import numpy as np

    obs, _ = env.reset(seed=123)
    rng = np.random.default_rng(0)
    for _ in range(n):
        a = int(pol.action_space.sample())
        assert env.action_space.contains(a)
        obs, reward, term, trunc, info = env.step(a)  # noqa: F841
        if term or trunc:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**32 - 1)))


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
        # Subset relationship at runtime: env must handle any policy action
        assert getattr(env.action_space, "n", 0) >= getattr(pol.action_space, "n", 0)
        # Should be able to stream at least one step without error
        events = list(r.stream(env, pol, seed=123, render=False))
        assert len(events) >= 1
        for ev in events:
            assert env.action_space.contains(int(ev.action))
        # Also verify random samples from policy.action_space can step the env
        _random_action_steps_succeed(env, pol)
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
        assert getattr(env.action_space, "n", 0) >= getattr(pol.action_space, "n", 0)
        events = list(r.stream(env, pol, seed=123, render=False))
        assert len(events) >= 1
        for ev in events:
            assert env.action_space.contains(int(ev.action))
        _random_action_steps_succeed(env, pol)
    finally:
        env.close()


def test_superset_pair_raises_and_subset_pair_streams():
    # Superset: oriented (n=3) on run_tumble (n=2) should raise
    env_rt = _make_env("run_tumble")
    try:
        pol_oriented = TemporalDerivativeDeterministicPolicy()
        with pytest.raises(ValueError):
            _ = next(r.stream(env_rt, pol_oriented, seed=1, render=False))
    finally:
        env_rt.close()

    # Subset: run_tumble (n=2) on oriented (n=3) should stream
    env_or = _make_env("oriented")
    try:
        pol_rt = RunTumbleTemporalDerivativePolicy()
        events = list(r.stream(env_or, pol_rt, seed=1, render=False))
        assert len(events) >= 1
        for ev in events:
            assert env_or.action_space.contains(int(ev.action))
    finally:
        env_or.close()
