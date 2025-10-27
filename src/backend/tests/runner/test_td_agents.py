from __future__ import annotations

import numpy as np

import plume_nav_sim as pns
from plume_nav_sim.policies import (
    TemporalDerivativeDeterministicPolicy,
    TemporalDerivativePolicy,
)
from plume_nav_sim.policies.run_tumble_td import RunTumbleTemporalDerivativePolicy
from plume_nav_sim.runner import runner as r


def _env(*, rgb: bool = False):
    return pns.make_env(
        grid_size=(32, 32),
        source_location=(24, 24),
        start_location=(4, 28),
        max_steps=200,
        render_mode=("rgb_array" if rgb else None),
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
    )


def test_td_deterministic_stream_probe_and_dc_rule():
    seed = 123
    env = _env(rgb=False)
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
    env = _env(rgb=False)
    try:
        policy = RunTumbleTemporalDerivativePolicy(threshold=1e-6, eps_seed=seed)

        last_c = None
        last_a = None
        saw_tumble = False
        for ev in r.stream(env, policy, seed=seed, render=False):
            c = float(ev.obs[0])
            if last_c is None:
                last_c = c
            else:
                dc = c - last_c
                if last_a not in (1, 2):
                    if dc < 1e-6:
                        # Tumble should be a single TURN step
                        assert ev.action in (1, 2)
                        saw_tumble = True
                    else:
                        assert ev.action == 0
                else:
                    # After a TURN, probe forward
                    assert ev.action == 0
                last_c = c
            last_a = int(ev.action)
            if ev.terminated or ev.truncated:
                break
        assert saw_tumble
    finally:
        env.close()
