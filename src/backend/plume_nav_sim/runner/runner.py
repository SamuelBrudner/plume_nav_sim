from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Protocol

import numpy as np

from plume_nav_sim._compat import is_space_subset
from plume_nav_sim.core.types import ActionType, ObservationType

logger = logging.getLogger(__name__)

PolicyObsAdapter = Callable[[ObservationType], Any]


class _PolicyLike(Protocol):  # minimal protocol to support tests
    def reset(self, *, seed: int | None = None) -> None:
        pass

    def select_action(
        self, observation: ObservationType, *, explore: bool = True
    ) -> ActionType:
        pass


@dataclass
class StepEvent:
    t: int
    obs: ObservationType
    action: ActionType
    reward: float
    terminated: bool
    truncated: bool
    info: dict
    frame: Optional[np.ndarray] = None


@dataclass
class EpisodeResult:
    seed: Optional[int]
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    metrics: dict[str, Any]


@dataclass
class _RenderContext:
    """Internal bookkeeping for render-related state and counters."""

    enabled: bool
    warned_missing: bool = False
    fallback_successes: int = 0
    fallback_failures: int = 0
    frames_captured: int = 0


def _reset_env_and_policy(
    env: Any, policy: Any, *, seed: Optional[int]
) -> ObservationType:
    if seed is not None:
        obs, _ = env.reset(seed=seed)
        _maybe_policy_reset(policy, seed=seed)
    else:
        obs, _ = env.reset()
        _maybe_policy_reset(policy, seed=None)
    return obs


def _maybe_policy_reset(policy: Any, *, seed: Optional[int]) -> None:
    try:
        policy.reset(seed=seed)  # type: ignore[attr-defined]
    except Exception:
        # If policy has no reset or incompatible signature, ignore
        pass


def _select_action(
    policy: Any, observation: ObservationType, *, adapter: Optional[PolicyObsAdapter]
) -> Any:
    policy_input = adapter(observation) if adapter is not None else observation

    # Prefer select_action if available (Policy protocol)
    if hasattr(policy, "select_action"):
        try:
            return policy.select_action(policy_input, explore=False)  # type: ignore[attr-defined]
        except TypeError:
            return policy.select_action(policy_input)  # type: ignore[misc]
    # Fallback to callable policy
    if callable(policy):
        return policy(policy_input)
    raise TypeError("Policy must implement select_action() or be callable")


def _ensure_action_space_compat(env: Any, policy: Any) -> None:
    env_space = getattr(env, "action_space", None)
    pol_space = getattr(policy, "action_space", None)
    if env_space is None or pol_space is None:
        return
    if not is_space_subset(pol_space, env_space):
        raise ValueError(
            "Policy action space must be a subset of the environment's action space"
        )


def _render_with_fallback(env: Any) -> tuple[Optional[np.ndarray], bool, bool]:
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


def _maybe_render_frame(
    env: Any, *, t: int, ctx: _RenderContext
) -> Optional[np.ndarray]:
    if not ctx.enabled:
        return None

    frame, succ, fail = _render_with_fallback(env)
    if succ:
        ctx.fallback_successes += 1
    if fail:
        ctx.fallback_failures += 1

    if frame is not None:
        ctx.frames_captured += 1
        try:
            logger.debug(
                "Captured frame at step %d: shape=%s dtype=%s",
                t,
                getattr(frame, "shape", None),
                getattr(frame, "dtype", None),
            )
        except Exception:
            logger.debug("Captured frame at step %d", t)
        return frame

    if (
        ctx.fallback_failures > 0 or ctx.fallback_successes > 0
    ) and not ctx.warned_missing:
        logger.warning(
            "No frame after fallback at step %d (successes=%d failures=%d)",
            t,
            ctx.fallback_successes,
            ctx.fallback_failures,
        )
        ctx.warned_missing = True
    return None


def _build_event(
    *,
    t: int,
    obs: ObservationType,
    action: ActionType,
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: Any,
    frame: Optional[np.ndarray],
    render_enabled: bool,
) -> StepEvent:
    return StepEvent(
        t=t,
        obs=obs,
        action=action,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=dict(info) if isinstance(info, dict) else {},
        frame=frame if (render_enabled and isinstance(frame, np.ndarray)) else None,
    )


def _step_once(
    env: Any,
    policy: Any,
    *,
    obs: ObservationType,
    t: int,
    render_ctx: _RenderContext,
    policy_obs_adapter: Optional[PolicyObsAdapter] = None,
) -> tuple[StepEvent, ObservationType, bool]:
    action = _select_action(policy, obs, adapter=policy_obs_adapter)
    next_obs, reward, term, trunc, info = env.step(action)
    frame = _maybe_render_frame(env, t=t, ctx=render_ctx)

    ev = _build_event(
        t=t,
        obs=obs,
        action=action,
        reward=reward,
        terminated=term,
        truncated=trunc,
        info=info,
        frame=frame,
        render_enabled=render_ctx.enabled,
    )

    done = ev.terminated or ev.truncated
    return ev, next_obs, done


def run_episode(
    env: Any,
    policy: Any,
    *,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    on_step: Optional[Callable[[StepEvent], None]] = None,
    on_episode_end: Optional[Callable[[EpisodeResult], None]] = None,
    render: bool = False,
    policy_obs_adapter: Optional[PolicyObsAdapter] = None,
) -> EpisodeResult:
    # Reset env and policy deterministically when seed provided
    obs = _reset_env_and_policy(env, policy, seed=seed)
    _ensure_action_space_compat(env, policy)

    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    render_ctx = _RenderContext(enabled=bool(render))

    while True:
        if max_steps is not None and steps >= max_steps:
            truncated = True
            break

        ev, next_obs, done = _step_once(
            env,
            policy,
            obs=obs,
            t=steps,
            render_ctx=render_ctx,
            policy_obs_adapter=policy_obs_adapter,
        )

        if on_step is not None:
            on_step(ev)

        steps += 1
        total_reward += float(ev.reward)
        obs = next_obs
        terminated = ev.terminated
        truncated = ev.truncated

        if done:
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
        render_ctx.frames_captured,
        render_ctx.fallback_successes,
        render_ctx.fallback_failures,
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
    policy_obs_adapter: Optional[PolicyObsAdapter] = None,
) -> Iterator[StepEvent]:
    obs = _reset_env_and_policy(env, policy, seed=seed)
    _ensure_action_space_compat(env, policy)

    t = 0
    render_ctx = _RenderContext(enabled=bool(render))
    while True:
        ev, next_obs, done = _step_once(
            env,
            policy,
            obs=obs,
            t=t,
            render_ctx=render_ctx,
            policy_obs_adapter=policy_obs_adapter,
        )

        if on_step is not None:
            on_step(ev)

        yield ev

        if done:
            break

        obs = next_obs
        t += 1


class Runner:
    def __init__(
        self,
        env: Any,
        policy: Any,
        *,
        policy_obs_adapter: Optional[PolicyObsAdapter] = None,
    ) -> None:
        _ensure_action_space_compat(env, policy)
        self._env = env
        self._policy = policy
        self._policy_obs_adapter = policy_obs_adapter

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
        policy_obs_adapter: Optional[PolicyObsAdapter] = None,
    ) -> EpisodeResult:
        adapter = (
            policy_obs_adapter
            if policy_obs_adapter is not None
            else self._policy_obs_adapter
        )
        return run_episode(
            self._env,
            self._policy,
            max_steps=max_steps,
            seed=seed,
            on_step=on_step,
            on_episode_end=on_episode_end,
            render=render,
            policy_obs_adapter=adapter,
        )

    def stream(
        self,
        *,
        seed: Optional[int] = None,
        render: bool = False,
        on_step: Optional[Callable[[StepEvent], None]] = None,
        policy_obs_adapter: Optional[PolicyObsAdapter] = None,
    ) -> Iterator[StepEvent]:
        adapter = (
            policy_obs_adapter
            if policy_obs_adapter is not None
            else self._policy_obs_adapter
        )
        return stream(
            self._env,
            self._policy,
            seed=seed,
            render=render,
            on_step=on_step,
            policy_obs_adapter=adapter,
        )
