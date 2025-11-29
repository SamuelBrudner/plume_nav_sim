from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import plume_nav_sim as pns
from plume_nav_sim.runner import runner as _runner

from .loader import ReplayArtifacts
from .schemas import EpisodeRecord, RunMeta, StepRecord

ReplayEnvFactory = Callable[[RunMeta, bool], Any]


class ReplayConsistencyError(RuntimeError):
    """Raised when replayed events diverge from recorded artifacts."""


@dataclass(frozen=True)
class _EpisodeSlice:
    episode_id: str
    steps: Sequence[StepRecord]
    index: int
    record: Optional[EpisodeRecord]


def _flatten_env_config(cfg: dict | None) -> dict[str, Any]:
    """Flatten nested env config payloads (e.g., Hydra env group)."""
    base = dict(cfg or {})
    nested = base.get("env")
    if isinstance(nested, dict):
        merged = {**base, **nested}
        merged.pop("env", None)
        return merged
    return base


def _normalize_env_kwargs(
    meta: RunMeta,
    *,
    render: bool,
    extra_env_kwargs: dict[str, Any],
    recorded_max_steps: int | None,
) -> dict[str, Any]:
    cfg = _flatten_env_config(meta.env_config)
    kwargs: dict[str, Any] = {}

    grid = cfg.get("grid_size")
    if grid:
        kwargs["grid_size"] = (int(grid[0]), int(grid[1]))

    src = cfg.get("source_location")
    if src:
        kwargs["source_location"] = (int(src[0]), int(src[1]))

    max_steps = cfg.get("max_steps")
    if max_steps is None:
        max_steps = recorded_max_steps
    if max_steps is not None:
        kwargs["max_steps"] = int(max_steps)

    if cfg.get("goal_radius") is not None:
        kwargs["goal_radius"] = float(cfg["goal_radius"])

    plume_params = cfg.get("plume_params")
    if plume_params:
        params = dict(plume_params)
        src_loc = params.get("source_location")
        if src_loc:
            params["source_location"] = (int(src_loc[0]), int(src_loc[1]))
        kwargs["plume_params"] = params

    enable_rendering = bool(cfg.get("enable_rendering", True))
    if render and enable_rendering:
        kwargs["render_mode"] = "rgb_array"

    defaults = {
        "action_type": "oriented",
        "observation_type": "concentration",
        "reward_type": "step_penalty",
    }
    merged = {**defaults, **kwargs, **extra_env_kwargs}
    return merged


def _default_env_factory(
    meta: RunMeta,
    *,
    render: bool,
    extra_env_kwargs: dict[str, Any],
    recorded_max_steps: int | None,
) -> Any:
    kwargs = _normalize_env_kwargs(
        meta,
        render=render,
        extra_env_kwargs=extra_env_kwargs,
        recorded_max_steps=recorded_max_steps,
    )
    return pns.make_env(**kwargs)


def _group_steps(
    steps: Sequence[StepRecord], episodes: Iterable[EpisodeRecord]
) -> list[_EpisodeSlice]:
    ordered_records = list(episodes)
    slices: list[_EpisodeSlice] = []
    current: list[StepRecord] = []
    current_id: Optional[str] = None

    def _flush() -> None:
        """Close out the current slice, aligning by episode order when IDs repeat."""
        if not current:
            return
        record = (
            ordered_records[len(slices)] if len(slices) < len(ordered_records) else None
        )
        slices.append(
            _EpisodeSlice(
                episode_id=current_id or "unknown",
                steps=tuple(current),
                index=len(slices),
                record=record,
            )
        )
        current.clear()

    for rec in steps:
        if current and (rec.episode_id != current_id or rec.step <= current[-1].step):
            _flush()
            current_id = None

        if current_id is None:
            current_id = rec.episode_id
        current.append(rec)

    _flush()

    return slices


def _resolve_seed(
    meta: RunMeta, episode_index: int, steps: Sequence[StepRecord]
) -> int | None:
    if meta.episode_seeds and episode_index < len(meta.episode_seeds):
        return int(meta.episode_seeds[episode_index])
    if meta.base_seed is not None:
        return int(meta.base_seed + episode_index)
    if steps and steps[0].seed is not None:
        return int(steps[0].seed)
    return None


def _infer_recorded_max_steps(
    meta: RunMeta, episodes: Sequence[_EpisodeSlice]
) -> int | None:
    """Infer max_steps from recorded config or consistent truncation markers."""
    cfg = _flatten_env_config(meta.env_config)
    raw_max = cfg.get("max_steps")
    if raw_max is not None:
        try:
            return int(raw_max)
        except Exception:
            pass

    limits: set[int] = set()
    for ep in episodes:
        if ep.record and ep.record.truncated and ep.record.total_steps:
            limits.add(int(ep.record.total_steps))
        else:
            truncated_steps = [rec.step for rec in ep.steps if rec.truncated]
            if truncated_steps:
                limits.add(int(truncated_steps[-1]))

    if len(limits) == 1:
        return limits.pop()
    return None


class ReplayEngine:
    """Deterministically replay captured runs as StepEvent streams."""

    def __init__(
        self,
        artifacts: ReplayArtifacts,
        *,
        env_factory: ReplayEnvFactory | None = None,
        env_kwargs: Optional[dict[str, Any]] = None,
        reward_tolerance: float = 1e-6,
    ) -> None:
        self._artifacts = artifacts
        self._env_factory = env_factory
        self._env_kwargs = env_kwargs or {}
        self._reward_tol = float(reward_tolerance)
        self._episodes = _group_steps(artifacts.steps, artifacts.episodes)
        self._total_steps = sum(len(ep.steps) for ep in self._episodes)
        self._recorded_max_steps = _infer_recorded_max_steps(
            artifacts.run_meta, self._episodes
        )

    def iter_events(
        self,
        *,
        render: bool = False,
        validate: bool = False,
        start_step: int = 0,
    ) -> Iterator[_runner.StepEvent]:
        """Yield StepEvents by replaying recorded actions through the environment.

        Args:
            render: Capture frames when the environment supports rgb_array rendering.
            validate: Raise ReplayConsistencyError on reward/position mismatches.
            start_step: Zero-based global step offset for seeking within the run.
        """
        if start_step < 0:
            raise ValueError("start_step must be non-negative")
        if start_step >= self._total_steps:
            return  # type: ignore[return-value]

        factory = self._env_factory
        if factory is None:
            factory = lambda meta, render_flag: _default_env_factory(  # noqa: E731
                meta,
                render=render_flag,
                extra_env_kwargs=self._env_kwargs,
                recorded_max_steps=self._recorded_max_steps,
            )

        env = factory(self._artifacts.run_meta, bool(render))
        cfg = _flatten_env_config(self._artifacts.run_meta.env_config)
        render_allowed = bool(render and cfg.get("enable_rendering", True))
        render_ctx = _runner._RenderContext(enabled=render_allowed)  # type: ignore[attr-defined]

        global_index = 0
        try:
            for ep in self._episodes:
                steps = ep.steps
                if global_index + len(steps) <= start_step:
                    global_index += len(steps)
                    continue

                seed = _resolve_seed(self._artifacts.run_meta, ep.index, steps)
                obs, info = env.reset(seed=seed)
                if validate and seed is not None:
                    recorded_seed = info.get("seed")
                    if recorded_seed is not None and int(recorded_seed) != int(seed):
                        raise ReplayConsistencyError(
                            f"Seed mismatch at episode {ep.episode_id}: "
                            f"env reported {recorded_seed}, expected {seed}"
                        )

                offset = max(0, start_step - global_index)
                episode_total = 0.0
                last_event: Optional[_runner.StepEvent] = None

                for idx, step_rec in enumerate(steps):
                    should_emit = idx >= offset
                    ev, obs = self._replay_step(
                        env=env,
                        step_rec=step_rec,
                        current_obs=obs,
                        render_ctx=render_ctx,
                        capture_frame=render_allowed and should_emit,
                    )
                    episode_total += ev.reward
                    last_event = ev
                    if validate:
                        self._validate_step(step_rec, ev)
                    if should_emit:
                        yield ev
                    if ev.terminated or ev.truncated:
                        if validate and idx + 1 < len(steps):
                            raise ReplayConsistencyError(
                                f"Run terminated at step {step_rec.step} in "
                                f"episode {ep.episode_id} with "
                                f"{len(steps) - idx - 1} recorded steps remaining"
                            )
                        break

                if validate and ep.record is not None and last_event is not None:
                    self._validate_episode(ep, episode_total, last_event)

                global_index += len(steps)
        finally:
            with contextlib.suppress(Exception):
                env.close()

    def _replay_step(
        self,
        *,
        env: Any,
        step_rec: StepRecord,
        current_obs: Any,
        render_ctx: _runner._RenderContext,  # type: ignore[attr-defined]
        capture_frame: bool,
    ) -> tuple[_runner.StepEvent, Any]:
        next_obs, reward, terminated, truncated, info = env.step(step_rec.action)
        frame = (
            _runner._maybe_render_frame(env, t=step_rec.step - 1, ctx=render_ctx)
            if capture_frame
            else None
        )
        ev = _runner._build_event(
            t=step_rec.step - 1,
            obs=current_obs,
            action=step_rec.action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            frame=frame,
            render_enabled=capture_frame,
        )
        return ev, next_obs

    def _validate_step(self, step_rec: StepRecord, ev: _runner.StepEvent) -> None:
        if ev.t != step_rec.step - 1:
            raise ReplayConsistencyError(
                f"Step index mismatch for episode {step_rec.episode_id}: "
                f"expected t={step_rec.step - 1}, got {ev.t}"
            )

        if abs(ev.reward - step_rec.reward) > self._reward_tol:
            raise ReplayConsistencyError(
                f"Reward mismatch at step {step_rec.step} "
                f"(episode {step_rec.episode_id}): "
                f"event={ev.reward}, recorded={step_rec.reward}"
            )

        if ev.terminated != step_rec.terminated or ev.truncated != step_rec.truncated:
            raise ReplayConsistencyError(
                f"Done flag mismatch at step {step_rec.step} "
                f"(episode {step_rec.episode_id}): "
                f"event term/trunc=({ev.terminated},{ev.truncated}) "
                f"recorded=({step_rec.terminated},{step_rec.truncated})"
            )

        pos = ev.info.get("agent_position") or ev.info.get("agent_xy")
        if pos is not None:
            try:
                px, py = int(pos[0]), int(pos[1])
            except Exception:
                px = py = None  # type: ignore[assignment]
            if (
                px is not None
                and py is not None
                and (px != step_rec.agent_position.x or py != step_rec.agent_position.y)
            ):
                raise ReplayConsistencyError(
                    f"Agent position mismatch at step {step_rec.step} "
                    f"(episode {step_rec.episode_id}): "
                    f"event=({px},{py}), recorded="
                    f"({step_rec.agent_position.x},{step_rec.agent_position.y})"
                )

    def _validate_episode(
        self,
        ep: _EpisodeSlice,
        total_reward: float,
        last_event: _runner.StepEvent,
    ) -> None:
        rec = ep.record
        if rec is None:
            return

        if rec.total_steps != len(ep.steps):
            raise ReplayConsistencyError(
                f"Episode {rec.episode_id} step count mismatch: "
                f"events={len(ep.steps)}, recorded={rec.total_steps}"
            )

        if abs(total_reward - rec.total_reward) > self._reward_tol:
            raise ReplayConsistencyError(
                f"Episode {rec.episode_id} total reward mismatch: "
                f"events={total_reward}, recorded={rec.total_reward}"
            )

        if (
            last_event.terminated != rec.terminated
            or last_event.truncated != rec.truncated
        ):
            raise ReplayConsistencyError(
                f"Episode {rec.episode_id} done flag mismatch: "
                f"event=({last_event.terminated},{last_event.truncated}), "
                f"recorded=({rec.terminated},{rec.truncated})"
            )

        pos = last_event.info.get("agent_position") or last_event.info.get("agent_xy")
        if pos is not None:
            try:
                px, py = int(pos[0]), int(pos[1])
            except Exception:
                px = py = None  # type: ignore[assignment]
            if (
                px is not None
                and py is not None
                and (px != rec.final_position.x or py != rec.final_position.y)
            ):
                raise ReplayConsistencyError(
                    f"Episode {rec.episode_id} final position mismatch: "
                    f"event=({px},{py}), recorded="
                    f"({rec.final_position.x},{rec.final_position.y})"
                )


__all__ = [
    "ReplayConsistencyError",
    "ReplayEngine",
    "ReplayEnvFactory",
]
