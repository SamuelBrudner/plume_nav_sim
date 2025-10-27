from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Protocol

import numpy as np


class _PolicyLike(Protocol):  # minimal protocol to support tests
    def reset(self, *, seed: int | None = None) -> None:
        pass

    def select_action(self, observation: np.ndarray, *, explore: bool = True) -> int:
        pass


@dataclass
class StepEvent:
    """Per-step event emitted by the runner stream.

    Attributes:
        t: Zero-based step index within the episode
        obs: Observation before taking the action (policy input)
        action: Action applied to the environment
        reward: Scalar reward returned by the step
        terminated: True if episode terminated
        truncated: True if episode truncated (time or other limit)
        info: Info dict returned by env.step
        frame: Optional RGB ndarray when render=True and env supports rgb_array
    """

    t: int
    obs: np.ndarray
    action: int | np.integer | np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    frame: Optional[np.ndarray] = None


@dataclass
class EpisodeResult:
    """Summary result of a completed episode run.

    Attributes:
        seed: Seed used to reset env/policy for this run
        steps: Number of steps executed
        total_reward: Sum of rewards across steps
        terminated: True if episode terminated
        truncated: True if episode truncated (time or other limit)
        metrics: Optional metrics dictionary (reserved for future use)
    """

    seed: Optional[int]
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    metrics: dict[str, Any]


def _maybe_policy_reset(policy: Any, *, seed: Optional[int]) -> None:
    try:
        policy.reset(seed=seed)  # type: ignore[attr-defined]
    except Exception:
        # If policy has no reset or incompatible signature, ignore
        pass


def _select_action(policy: Any, observation: np.ndarray) -> Any:
    # Prefer select_action if available (Policy protocol)
    if hasattr(policy, "select_action"):
        try:
            return policy.select_action(observation, explore=False)  # type: ignore[attr-defined]
        except TypeError:
            return policy.select_action(observation)  # type: ignore[misc]
    # Fallback to callable policy
    if callable(policy):
        return policy(observation)
    raise TypeError("Policy must implement select_action() or be callable")


def _ensure_action_space_compat(env: Any, policy: Any) -> None:
    """Validate that policy and env action spaces are compatible.

    Currently supports Discrete spaces; raises ValueError on mismatch.
    """
    try:
        import gymnasium as gym  # type: ignore
        from gymnasium.spaces import Discrete  # type: ignore
    except Exception:  # pragma: no cover
        return  # If gym not available, skip compatibility check

    env_space = getattr(env, "action_space", None)
    pol_space = getattr(policy, "action_space", None)
    if env_space is None or pol_space is None:
        return
    # Only enforce for Discrete spaces where sizes must match
    if isinstance(env_space, Discrete) and isinstance(pol_space, Discrete):
        if int(env_space.n) != int(pol_space.n):
            raise ValueError(
                f"Policy/env action_space mismatch: policy Discrete({int(pol_space.n)}) "
                f"vs env Discrete({int(env_space.n)}). Choose a compatible policy/env pair."
            )


def run_episode(
    env: Any,
    policy: Any,
    *,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    on_step: Optional[Callable[[StepEvent], None]] = None,
    on_episode_end: Optional[Callable[[EpisodeResult], None]] = None,
    render: bool = False,
) -> EpisodeResult:
    """Run a single episode and return summary result.

    Deterministic when given the same (seed, env, policy) triplet.
    """
    # Reset env and policy deterministically when seed provided
    if seed is not None:
        obs, _ = env.reset(seed=seed)
        _maybe_policy_reset(policy, seed=seed)
    else:
        obs, _ = env.reset()
        _maybe_policy_reset(policy, seed=None)
    _ensure_action_space_compat(env, policy)

    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    while True:
        if max_steps is not None and steps >= max_steps:
            # Respect soft cap through truncation semantics if env didn't enforce
            truncated = True
            break

        action = _select_action(policy, obs)
        next_obs, reward, term, trunc, info = env.step(action)
        frame = env.render("rgb_array") if render else None

        ev = StepEvent(
            t=steps,
            obs=obs,
            action=action,
            reward=float(reward),
            terminated=bool(term),
            truncated=bool(trunc),
            info=dict(info) if isinstance(info, dict) else {},
            frame=frame if (render and isinstance(frame, np.ndarray)) else None,
        )

        if on_step is not None:
            on_step(ev)

        steps += 1
        total_reward += float(reward)
        obs = next_obs
        terminated = bool(term)
        truncated = bool(trunc)

        if terminated or truncated:
            break

    result = EpisodeResult(
        seed=seed,
        steps=steps,
        total_reward=float(total_reward),
        terminated=terminated,
        truncated=truncated,
        metrics={},
    )

    if on_episode_end is not None:
        on_episode_end(result)

    return result


def stream(
    env: Any,
    policy: Any,
    *,
    seed: Optional[int] = None,
    render: bool = False,
    on_step: Optional[Callable[[StepEvent], None]] = None,
) -> Iterator[StepEvent]:
    """Yield one StepEvent per env.step() until termination.

    If seed is provided, resets env and policy deterministically.
    """
    if seed is not None:
        obs, _ = env.reset(seed=seed)
        _maybe_policy_reset(policy, seed=seed)
    else:
        obs, _ = env.reset()
        _maybe_policy_reset(policy, seed=None)
    _ensure_action_space_compat(env, policy)

    t = 0
    while True:
        action = _select_action(policy, obs)
        next_obs, reward, term, trunc, info = env.step(action)
        frame = env.render("rgb_array") if render else None

        ev = StepEvent(
            t=t,
            obs=obs,
            action=action,
            reward=float(reward),
            terminated=bool(term),
            truncated=bool(trunc),
            info=dict(info) if isinstance(info, dict) else {},
            frame=frame if (render and isinstance(frame, np.ndarray)) else None,
        )

        if on_step is not None:
            on_step(ev)

        yield ev

        if ev.terminated or ev.truncated:
            break

        obs = next_obs
        t += 1
