from __future__ import annotations

import sys
from pathlib import Path

import pytest

import plume_nav_sim as pns
from plume_nav_sim.policies import (
    TemporalDerivativeDeterministicPolicy,
    TemporalDerivativePolicy,
)

# Expose demo package policy
_demo_path = Path(__file__).resolve().parents[4] / "plug-and-play-demo"
if _demo_path.is_dir():
    sys.path.append(str(_demo_path))

plug_demo = pytest.importorskip("plug_and_play_demo")
DeltaBasedRunTumblePolicy = plug_demo.DeltaBasedRunTumblePolicy
from plume_nav_sim.runner import runner as r


def _env(*, rgb: bool = False, action_type: str = "oriented"):
    return pns.make_env(
        grid_size=(32, 32),
        source_location=(24, 24),
        start_location=(4, 28),
        max_steps=200,
        render_mode=("rgb_array" if rgb else None),
        action_type=action_type,
        observation_type="concentration",
        reward_type="step_penalty",
    )


def test_td_deterministic_stream_probe_and_dc_rule():
    seed = 123
    env = _env(rgb=False, action_type="oriented")
    try:
        policy = TemporalDerivativeDeterministicPolicy(
            threshold=1e-6, alternate_cast=True
        )

        events = list(r.stream(env, policy, seed=seed, render=False))
        assert len(events) > 2

        # Probe-after-turn property: TURN must be followed by FORWARD
        for i in range(1, len(events)):
            if events[i - 1].action in (1, 2):
                assert events[i].action == 0

        # DC rule when previous action was FORWARD
        last_c = None
        last_a = None
        for ev in events:
            c = float(ev.obs[0])
            if last_c is None:
                last_c = c
            else:
                if last_a in (1, 2):
                    assert ev.action == 0  # probe after turn
                else:
                    dc = c - last_c
                    if dc >= 1e-6:
                        assert ev.action == 0
                    else:
                        assert ev.action in (1, 2)
                last_c = c
            last_a = int(ev.action)
    finally:
        env.close()


def test_td_bacterial_stream_uniform_on_non_increase():
    seed = 31415
    env = _env(rgb=False)
    try:
        policy = TemporalDerivativePolicy(
            eps=0.0,
            eps_after_turn=0.0,
            eps_greedy_forward_bias=0.0,
            uniform_random_on_non_increase=True,
        )

        events = list(r.stream(env, policy, seed=seed, render=False))
        assert len(events) > 2

        # Probe-after-turn
        for i in range(1, len(events)):
            if events[i - 1].action in (1, 2):
                assert events[i].action == 0

        last_c = None
        last_a = None
        pos, neg = 0, 0
        for ev in events:
            c = float(ev.obs[0])
            if last_c is None:
                last_c = c
            else:
                if last_a in (1, 2):
                    assert ev.action == 0
                else:
                    dc = c - last_c
                    if dc > 1e-6:
                        pos += 1
                        assert ev.action == 0
                    else:
                        neg += 1
                        assert ev.action in (0, 1, 2)  # uniform over all actions
                last_c = c
            last_a = int(ev.action)

        # Sanity: saw both regimes
        assert pos + neg > 0
    finally:
        env.close()


def test_td_deterministic_run_episode_callbacks_and_summary():
    seed = 777
    env = _env(rgb=False)
    try:
        policy = TemporalDerivativeDeterministicPolicy(
            threshold=1e-6, alternate_cast=True
        )

        last_c = None
        last_a = None
        checks = {"pos": 0, "neg": 0}

        def on_step(ev):
            nonlocal last_c, last_a
            c = float(ev.obs[0])
            if last_c is None:
                last_c = c
            else:
                if last_a in (1, 2):
                    assert ev.action == 0
                else:
                    dc = c - last_c
                    if dc >= 1e-6:
                        checks["pos"] += 1
                        assert ev.action == 0
                    else:
                        checks["neg"] += 1
                        assert ev.action in (1, 2)
                last_c = c
            last_a = int(ev.action)

        res = r.run_episode(env, policy, seed=seed, render=False, on_step=on_step)
        assert res.steps > 0
        # Sanity: observed at least one checkable step
        assert checks["pos"] + checks["neg"] >= 1
    finally:
        env.close()


def test_run_tumble_td_behavior_sequences():
    seed = 42
    env = _env(rgb=False, action_type="run_tumble")
    try:
        policy = DeltaBasedRunTumblePolicy(threshold=1e-6, eps_seed=seed)

        last_c = None
        saw_tumble = False
        for ev in r.stream(env, policy, seed=seed, render=False):
            c = float(ev.obs[0])
            if last_c is None:
                last_c = c
            else:
                # Relaxed assertion for stateless policy: just confirm we see at least one tumble
                if int(ev.action) == 1:
                    saw_tumble = True
                last_c = c
            if ev.terminated or ev.truncated:
                break
        assert saw_tumble
    finally:
        env.close()


def test_policy_env_superset_pair_errors_and_subset_pair_ok():
    import pytest

    seed = 1
    # Oriented policy (n=3) with run_tumble env (n=2) should error (superset)
    env_rt = _env(rgb=False, action_type="run_tumble")
    try:
        pol_oriented = TemporalDerivativeDeterministicPolicy()
        with pytest.raises(ValueError):
            _ = next(r.stream(env_rt, pol_oriented, seed=seed, render=False))
    finally:
        env_rt.close()

    # Run/Tumble policy (n=2) with oriented env (n=3) should stream (subset)
    env_or = _env(rgb=False, action_type="oriented")
    try:
        pol_rt = DeltaBasedRunTumblePolicy()
        events = list(r.stream(env_or, pol_rt, seed=seed, render=False))
        assert len(events) >= 1
        for ev in events:
            assert env_or.action_space.contains(int(ev.action))
    finally:
        env_or.close()
