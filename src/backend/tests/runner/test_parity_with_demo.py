from __future__ import annotations

"""Parity tests: runner.stream should match the demo's manual stepping.

These tests compare action sequences between a manual loop (policy.select_action → env.step)
and the UI-agnostic runner.stream() with the same seed and deterministic policy.
"""

import plume_nav_sim as pns
from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy
from plume_nav_sim.runner import runner as r


def _create_env(*, rgb: bool = False):
    return pns.make_env(
        grid_size=(64, 64),
        source_location=(48, 48),
        start_location=(16, 16),
        max_steps=300,
        render_mode=("rgb_array" if rgb else None),
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
    )


def _manual_actions(env, policy, seed: int, *, render: bool = False) -> list[int]:
    obs, _ = env.reset(seed=seed)
    policy.reset(seed=seed)
    actions: list[int] = []
    while True:
        a = int(policy.select_action(obs, explore=False))
        actions.append(a)
        obs, reward, term, trunc, info = env.step(a)
        if render:
            # Mirror runner's render call to guard against potential side effects
            env.render("rgb_array")
        if term or trunc:
            break
    return actions


def test_runner_parity_deterministic_no_render():
    seed = 123
    env_m = _create_env(rgb=False)
    env_r = _create_env(rgb=False)
    try:
        p_m = TemporalDerivativeDeterministicPolicy(threshold=1e-6, alternate_cast=True)
        p_r = TemporalDerivativeDeterministicPolicy(threshold=1e-6, alternate_cast=True)

        manual = _manual_actions(env_m, p_m, seed=seed, render=False)
        runner = [
            int(ev.action) for ev in r.stream(env_r, p_r, seed=seed, render=False)
        ]

        assert manual == runner
    finally:
        env_m.close()
        env_r.close()


def test_runner_parity_deterministic_with_render():
    seed = 123
    env_m = _create_env(rgb=True)
    env_r = _create_env(rgb=True)
    try:
        p_m = TemporalDerivativeDeterministicPolicy(threshold=1e-6, alternate_cast=True)
        p_r = TemporalDerivativeDeterministicPolicy(threshold=1e-6, alternate_cast=True)

        # Manual loop renders each step to mirror runner(render=True) behavior
        manual = _manual_actions(env_m, p_m, seed=seed, render=True)
        runner = [int(ev.action) for ev in r.stream(env_r, p_r, seed=seed, render=True)]

        assert manual == runner
    finally:
        env_m.close()
        env_r.close()
