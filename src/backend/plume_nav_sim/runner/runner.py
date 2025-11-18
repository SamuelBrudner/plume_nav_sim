from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Protocol

import numpy as np

from plume_nav_sim.utils.spaces import is_space_subset

logger = logging.getLogger(__name__)


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
    env_space = getattr(env, "action_space", None)
    pol_space = getattr(policy, "action_space", None)
    if env_space is None or pol_space is None:
        return
    if not is_space_subset(pol_space, env_space):
        raise ValueError(
            "Policy action space must be a subset of the environment's action space"
        )


def _render_with_fallback(env: Any) -> tuple[Optional[np.ndarray], bool, bool]:
    """Attempt to render a frame, trying modern and legacy signatures.

    Returns (frame, fallback_success, fallback_failure) where fallback flags
    indicate whether the legacy path was attempted and succeeded/failed.
    """
    frame = None
    fallback_success = False
    fallback_failure = False
    try:
        # Prefer modern Gymnasium: render() uses configured render_mode
        frame = env.render()
    except TypeError:
        try:
            frame = env.render(mode="rgb_array")
            fallback_success = isinstance(frame, np.ndarray)
            if not fallback_success:
                fallback_failure = True
        except Exception:
            frame = None
            fallback_failure = True
    else:
        # If no frame produced, try explicit rgb_array
        if not isinstance(frame, np.ndarray):
            try:
                frame = env.render(mode="rgb_array")
                fallback_success = isinstance(frame, np.ndarray)
                if not fallback_success:
                    fallback_failure = True
            except Exception:
                frame = None
                fallback_failure = True

    return (
        frame if isinstance(frame, np.ndarray) else None,
        fallback_success,
        fallback_failure,
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
    frames_captured = 0
    fallback_successes = 0
    fallback_failures = 0
    warned_missing = False

    while True:
        if max_steps is not None and steps >= max_steps:
            # Respect soft cap through truncation semantics if env didn't enforce
            truncated = True
            break

        action = _select_action(policy, obs)
        next_obs, reward, term, trunc, info = env.step(action)
        frame = None
        if render:
            frame, succ, fail = _render_with_fallback(env)
            if succ:
                fallback_successes += 1
            if fail:
                fallback_failures += 1

            if frame is not None:
                frames_captured += 1
                try:
                    logger.debug(
                        "Captured frame at step %d: shape=%s dtype=%s",
                        steps,
                        getattr(frame, "shape", None),
                        getattr(frame, "dtype", None),
                    )
                except Exception:
                    logger.debug("Captured frame at step %d", steps)
            elif (
                fallback_failures > 0 or fallback_successes > 0
            ) and not warned_missing:
                logger.warning(
                    "No frame after fallback at step %d (successes=%d failures=%d)",
                    steps,
                    fallback_successes,
                    fallback_failures,
                )
                warned_missing = True

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

    logger.info(
        "Episode finished: steps=%d frames_captured=%d fallback_successes=%d fallback_failures=%d",
        steps,
        frames_captured,
        fallback_successes,
        fallback_failures,
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
    warned_missing = False
    fallback_successes = 0
    fallback_failures = 0
    while True:
        action = _select_action(policy, obs)
        next_obs, reward, term, trunc, info = env.step(action)
        frame = None
        if render:
            frame, succ, fail = _render_with_fallback(env)
            if succ:
                fallback_successes += 1
            if fail:
                fallback_failures += 1

            if frame is not None:
                try:
                    logger.debug(
                        "Captured frame at step %d: shape=%s dtype=%s",
                        t,
                        getattr(frame, "shape", None),
                        getattr(frame, "dtype", None),
                    )
                except Exception:
                    logger.debug("Captured frame at step %d", t)
            elif (
                fallback_failures > 0 or fallback_successes > 0
            ) and not warned_missing:
                logger.warning(
                    "No frame after fallback at step %d (successes=%d failures=%d)",
                    t,
                    fallback_successes,
                    fallback_failures,
                )
                warned_missing = True

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


class Runner:
    """Thin OO wrapper over runner functions with upfront validation.

    Validates policy/env action-space subset on construction and exposes
    `stream` and `run_episode` bound to the provided env and policy.
    """

    def __init__(self, env: Any, policy: Any) -> None:
        _ensure_action_space_compat(env, policy)
        self._env = env
        self._policy = policy

    @staticmethod
    def validate(env: Any, policy: Any) -> None:
        """Validate that policy.action_space ⊆ env.action_space."""
        _ensure_action_space_compat(env, policy)

    def run_episode(
        self,
        *,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        on_step: Optional[Callable[[StepEvent], None]] = None,
        on_episode_end: Optional[Callable[[EpisodeResult], None]] = None,
        render: bool = False,
    ) -> EpisodeResult:
        return run_episode(
            self._env,
            self._policy,
            max_steps=max_steps,
            seed=seed,
            on_step=on_step,
            on_episode_end=on_episode_end,
            render=render,
        )

    def stream(
        self,
        *,
        seed: Optional[int] = None,
        render: bool = False,
        on_step: Optional[Callable[[StepEvent], None]] = None,
    ) -> Iterator[StepEvent]:
        return stream(
            self._env, self._policy, seed=seed, render=render, on_step=on_step
        )
