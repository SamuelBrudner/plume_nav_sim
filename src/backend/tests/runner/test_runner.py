"""
TDD: Backend runner tests (fail first).

Contract for UI-agnostic runner utilities that orchestrate plume_nav_sim environments
with per-step streaming and episode-level summaries.

Target API (to be implemented under plume_nav_sim/runner/runner.py):
  - run_episode(env, policy, *, max_steps=None, seed=None, on_step=None, on_episode_end=None, render=False) -> EpisodeResult
  - stream(env, policy, *, seed=None, render=False, on_step=None) -> Iterator[StepEvent]

Where StepEvent contains: t, obs, action, reward, terminated, truncated, info, (optional) frame
and EpisodeResult contains: seed, steps, total_reward, terminated, truncated, metrics
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import pytest

import plume_nav_sim as pns
from plume_nav_sim.policies import TemporalDerivativeDeterministicPolicy

# Utilities -----------------------------------------------------------------


def create_test_env(*, rgb: bool = False):
    return pns.make_env(
        grid_size=(16, 16),
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        max_steps=100,
        render_mode=("rgb_array" if rgb else None),
    )


# Tests ---------------------------------------------------------------------


def test_run_episode_basic_contract():
    """run_episode returns coherent EpisodeResult and respects seed + max_steps."""
    from plume_nav_sim.runner import runner as r

    env = create_test_env(rgb=False)
    policy = TemporalDerivativeDeterministicPolicy()

    seed = 123
    result = r.run_episode(env, policy, max_steps=50, seed=seed)

    # Basic result fields
    assert isinstance(result.steps, int)
    assert 0 < result.steps <= 50
    assert isinstance(result.total_reward, (int, float))
    assert isinstance(result.terminated, bool)
    assert isinstance(result.truncated, bool)

    # Determinism: same seed, same result
    env2 = create_test_env(rgb=False)
    policy2 = TemporalDerivativeDeterministicPolicy()
    result2 = r.run_episode(env2, policy2, max_steps=50, seed=seed)

    assert result2.steps == result.steps
    assert result2.terminated == result.terminated
    assert result2.truncated == result.truncated
    assert np.isclose(result2.total_reward, result.total_reward)


def test_stream_yields_events_and_stops():
    """stream yields StepEvent sequence, then stops at termination or truncation."""
    from plume_nav_sim.runner import runner as r

    env = create_test_env(rgb=False)
    policy = TemporalDerivativeDeterministicPolicy()

    events = list(r.stream(env, policy, seed=42, render=False))

    assert len(events) > 0

    # Monotonic t and final stop flag
    ts = [ev.t for ev in events]
    assert ts == list(range(len(events))), "t should start at 0 and increment by 1"
    assert events[-1].terminated or events[-1].truncated

    # Expected fields on an event
    ev0 = events[0]
    assert isinstance(ev0.obs, np.ndarray)
    assert isinstance(ev0.action, (int, np.integer))
    assert isinstance(ev0.reward, (int, float))
    assert isinstance(ev0.terminated, bool)
    assert isinstance(ev0.truncated, bool)
    assert isinstance(ev0.info, dict)
    assert getattr(ev0, "frame", None) is None  # render=False


def test_stream_with_render_attaches_frame():
    """When render=True and env supports rgb_array, StepEvent carries frame ndarray."""
    from plume_nav_sim.runner import runner as r

    env = create_test_env(rgb=True)
    policy = TemporalDerivativeDeterministicPolicy()

    it: Iterator = r.stream(env, policy, seed=7, render=True)
    first = next(it)

    frame = getattr(first, "frame", None)
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8


def test_callbacks_invoked_in_order():
    """on_step fires per step; on_episode_end fires exactly once after steps."""
    from plume_nav_sim.runner import runner as r

    env = create_test_env(rgb=False)
    policy = TemporalDerivativeDeterministicPolicy()

    step_count = 0
    seen_flags: list[tuple[bool, bool]] = []
    episode_ended = {"called": False}

    def on_step(ev):
        nonlocal step_count
        step_count += 1
        seen_flags.append((ev.terminated, ev.truncated))

    def on_episode_end(res):
        episode_ended["called"] = True

    res = r.run_episode(
        env,
        policy,
        max_steps=40,
        seed=101,
        on_step=on_step,
        on_episode_end=on_episode_end,
        render=False,
    )

    assert episode_ended["called"] is True
    assert step_count == res.steps
    # Last step should align with res termination flags (if any steps occurred)
    if seen_flags:
        last_term, last_trunc = seen_flags[-1]
        assert last_term == res.terminated or last_trunc == res.truncated


def test_deterministic_sequence_across_instances():
    """Same seed + deterministic policy yields identical action/reward sequences."""
    from plume_nav_sim.runner import runner as r

    seed = 31415
    env1 = create_test_env(rgb=False)
    env2 = create_test_env(rgb=False)
    p1 = TemporalDerivativeDeterministicPolicy()
    p2 = TemporalDerivativeDeterministicPolicy()

    seq1 = [(ev.action, ev.reward) for ev in r.stream(env1, p1, seed=seed)]
    seq2 = [(ev.action, ev.reward) for ev in r.stream(env2, p2, seed=seed)]

    assert seq1 == seq2
