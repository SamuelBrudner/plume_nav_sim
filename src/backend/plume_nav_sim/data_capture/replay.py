from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np

from .loader import ReplayArtifacts, ReplayLoadError, iter_episode_steps, load_replay_artifacts
from .schemas import EpisodeRecord, RunMeta, StepRecord

ReplayEnvFactory = Callable[[RunMeta, bool], Any]


class ReplayConsistencyError(RuntimeError):
    """Raised when replayed events diverge from recorded artifacts."""

    def __init__(self, message: str, *, diff: ReplayValidationDiff | None = None):
        super().__init__(message)
        self.diff = diff


@dataclass(frozen=True)
class ReplayStepEvent:
    t: int
    obs: object
    action: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    frame: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ReplayFieldMismatch:
    field: str
    expected: object
    actual: object

    def to_dict(self) -> dict[str, object]:
        return {"field": self.field, "expected": self.expected, "actual": self.actual}


@dataclass(frozen=True)
class ReplayValidationDiff:
    run_dir: str
    global_step_index: int
    episode_id: str
    episode_step: int
    action: int
    mismatches: Sequence[ReplayFieldMismatch]
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "run_dir": self.run_dir,
            "global_step_index": int(self.global_step_index),
            "episode_id": str(self.episode_id),
            "episode_step": int(self.episode_step),
            "action": int(self.action),
            "mismatches": [m.to_dict() for m in self.mismatches],
        }
        if self.note:
            out["note"] = str(self.note)
        return out


def _coerce_grid_size(meta: RunMeta) -> tuple[int, int]:
    cfg = meta.env_config if isinstance(meta.env_config, dict) else {}
    grid = cfg.get("grid_size")
    if isinstance(grid, (tuple, list)) and len(grid) == 2:
        try:
            return int(grid[0]), int(grid[1])
        except Exception:
            return (64, 64)
    return (64, 64)


def _coerce_goal_xy(meta: RunMeta) -> tuple[int, int] | None:
    cfg = meta.env_config if isinstance(meta.env_config, dict) else {}
    goal = cfg.get("source_location")
    if isinstance(goal, (tuple, list)) and len(goal) == 2:
        try:
            return int(goal[0]), int(goal[1])
        except Exception:
            return None
    return None


def _render_frame(
    *,
    grid_size: tuple[int, int],
    agent_xy: tuple[int, int] | None,
    goal_xy: tuple[int, int] | None,
) -> np.ndarray:
    w, h = int(grid_size[0]), int(grid_size[1])
    w = max(2, min(w, 512))
    h = max(2, min(h, 512))
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    if goal_xy is not None:
        gx, gy = goal_xy
        if 0 <= gx < w and 0 <= gy < h:
            frame[gy, gx] = np.array([255, 215, 0], dtype=np.uint8)

    if agent_xy is not None:
        ax, ay = agent_xy
        if 0 <= ax < w and 0 <= ay < h:
            frame[ay, ax] = np.array([0, 255, 255], dtype=np.uint8)

    return frame


def _safe_close(env: object) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _summarize_observation(obs: object) -> Optional[list[float]]:
    """Match DataCaptureWrapper observation_summary behavior."""
    if hasattr(obs, "__len__"):
        try:
            if len(obs) > 0:  # type: ignore[arg-type]
                return [float(obs[0])]  # type: ignore[index]
        except Exception:
            return None
    return None


def _extract_agent_xy(info: object) -> tuple[int, int] | None:
    if not isinstance(info, dict):
        return None
    xy = info.get("agent_position") or info.get("agent_xy")
    if isinstance(xy, (tuple, list)) and len(xy) == 2:
        try:
            return int(xy[0]), int(xy[1])
        except Exception:
            return None
    if isinstance(xy, dict):
        try:
            return int(xy["x"]), int(xy["y"])
        except Exception:
            return None
    if hasattr(xy, "x") and hasattr(xy, "y"):
        try:
            return int(getattr(xy, "x")), int(getattr(xy, "y"))
        except Exception:
            return None
    return None


def _isclose(a: float, b: float, *, tol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=float(tol))


class ReplayEngine:
    """Load a recorded run and yield replay events.

    This is a lightweight implementation intended for debugger playback.
    """

    def __init__(
        self,
        artifacts: ReplayArtifacts,
        *,
        reward_tolerance: float = 1e-6,
    ) -> None:
        self._artifacts = artifacts
        self._reward_tol = reward_tolerance
        self._episodes = iter_episode_steps(artifacts.steps)
        self._grid_size = _coerce_grid_size(artifacts.run_meta)
        self._goal_xy = _coerce_goal_xy(artifacts.run_meta)

    @property
    def run_meta(self) -> RunMeta:
        return self._artifacts.run_meta

    @property
    def episodes(self) -> list[list[StepRecord]]:
        return list(self._episodes)

    def total_steps(self) -> int:
        return len(self._artifacts.steps)

    def total_episodes(self) -> int:
        return len(self._episodes)

    def validate(
        self,
        *,
        env_factory: ReplayEnvFactory,
        render: bool = False,
        distance_tolerance: float | None = None,
        obs_tolerance: float | None = None,
    ) -> None:
        """Validate recorded artifacts by re-stepping the environment.

        This is a best-effort debugger utility: callers must provide an env_factory
        that reconstructs the original environment deterministically.

        Raises ReplayConsistencyError on the first detected divergence.
        """

        steps = self._artifacts.steps
        if not steps:
            return

        dist_tol = float(self._reward_tol if distance_tolerance is None else distance_tolerance)
        obs_tol = float(self._reward_tol if obs_tolerance is None else obs_tolerance)

        env: object | None = None
        current_episode_id: str | None = None

        try:
            for global_idx, rec in enumerate(steps):
                if current_episode_id != rec.episode_id:
                    # Episode boundary: create a fresh env and reset with the recorded seed (if present).
                    if env is not None:
                        _safe_close(env)
                        env = None
                    current_episode_id = rec.episode_id

                    try:
                        env = env_factory(self._artifacts.run_meta, bool(render))
                    except Exception as exc:
                        diff = ReplayValidationDiff(
                            run_dir=str(self._artifacts.run_dir),
                            global_step_index=int(global_idx),
                            episode_id=str(rec.episode_id),
                            episode_step=int(rec.step),
                            action=int(rec.action),
                            mismatches=(
                                ReplayFieldMismatch(
                                    field="exception",
                                    expected="env_factory() to succeed",
                                    actual=f"{type(exc).__name__}: {exc}",
                                ),
                            ),
                            note="stage=env_factory",
                        )
                        raise ReplayConsistencyError(
                            f"Replay validation failed at step {global_idx}: env_factory error",
                            diff=diff,
                        ) from exc

                    seed = rec.seed
                    if seed is None:
                        seed = getattr(self._artifacts.run_meta, "base_seed", None)
                    try:
                        reset = getattr(env, "reset", None)
                        if not callable(reset):
                            raise TypeError("env has no reset(seed=...) method")
                        reset(seed=seed)
                    except Exception as exc:
                        diff = ReplayValidationDiff(
                            run_dir=str(self._artifacts.run_dir),
                            global_step_index=int(global_idx),
                            episode_id=str(rec.episode_id),
                            episode_step=int(rec.step),
                            action=int(rec.action),
                            mismatches=(
                                ReplayFieldMismatch(
                                    field="exception",
                                    expected="env.reset(seed=...) to succeed",
                                    actual=f"{type(exc).__name__}: {exc}",
                                ),
                            ),
                            note=f"stage=env_reset seed={seed}",
                        )
                        raise ReplayConsistencyError(
                            f"Replay validation failed at step {global_idx}: env.reset error",
                            diff=diff,
                        ) from exc

                if env is None:
                    continue  # defensive: should not happen

                try:
                    step = getattr(env, "step", None)
                    if not callable(step):
                        raise TypeError("env has no step(action) method")
                    obs, reward, terminated, truncated, info = step(int(rec.action))
                except Exception as exc:
                    diff = ReplayValidationDiff(
                        run_dir=str(self._artifacts.run_dir),
                        global_step_index=int(global_idx),
                        episode_id=str(rec.episode_id),
                        episode_step=int(rec.step),
                        action=int(rec.action),
                        mismatches=(
                            ReplayFieldMismatch(
                                field="exception",
                                expected="env.step(action) to succeed",
                                actual=f"{type(exc).__name__}: {exc}",
                            ),
                        ),
                        note="stage=env_step",
                    )
                    raise ReplayConsistencyError(
                        f"Replay validation failed at step {global_idx}: env.step error",
                        diff=diff,
                    ) from exc

                mismatches: list[ReplayFieldMismatch] = []

                # Reward
                try:
                    if not _isclose(float(rec.reward), float(reward), tol=float(self._reward_tol)):
                        mismatches.append(
                            ReplayFieldMismatch("reward", float(rec.reward), float(reward))
                        )
                except Exception:
                    mismatches.append(
                        ReplayFieldMismatch("reward", float(rec.reward), reward)
                    )

                # Termination flags
                if bool(rec.terminated) != bool(terminated):
                    mismatches.append(
                        ReplayFieldMismatch("terminated", bool(rec.terminated), bool(terminated))
                    )
                if bool(rec.truncated) != bool(truncated):
                    mismatches.append(
                        ReplayFieldMismatch("truncated", bool(rec.truncated), bool(truncated))
                    )

                # Info-derived metrics
                agent_xy = _extract_agent_xy(info)
                exp_xy = (int(rec.agent_position.x), int(rec.agent_position.y))
                if agent_xy is None or (int(agent_xy[0]), int(agent_xy[1])) != exp_xy:
                    mismatches.append(ReplayFieldMismatch("agent_xy", exp_xy, agent_xy))

                dist_val = None
                if isinstance(info, dict):
                    dist_val = info.get("distance_to_goal")
                if dist_val is None:
                    mismatches.append(
                        ReplayFieldMismatch("distance_to_goal", float(rec.distance_to_goal), None)
                    )
                else:
                    try:
                        if not _isclose(float(rec.distance_to_goal), float(dist_val), tol=dist_tol):
                            mismatches.append(
                                ReplayFieldMismatch(
                                    "distance_to_goal",
                                    float(rec.distance_to_goal),
                                    float(dist_val),
                                )
                            )
                    except Exception:
                        mismatches.append(
                            ReplayFieldMismatch(
                                "distance_to_goal",
                                float(rec.distance_to_goal),
                                dist_val,
                            )
                        )

                # Step count (if available)
                if isinstance(info, dict) and "step_count" in info:
                    try:
                        if int(info.get("step_count")) != int(rec.step):
                            mismatches.append(
                                ReplayFieldMismatch("step_count", int(rec.step), info.get("step_count"))
                            )
                    except Exception:
                        mismatches.append(
                            ReplayFieldMismatch("step_count", int(rec.step), info.get("step_count"))
                        )

                # Seed (best-effort; only compare when both sides present)
                if isinstance(info, dict) and "seed" in info and rec.seed is not None:
                    try:
                        if int(info.get("seed")) != int(rec.seed):
                            mismatches.append(
                                ReplayFieldMismatch("seed", int(rec.seed), info.get("seed"))
                            )
                    except Exception:
                        mismatches.append(ReplayFieldMismatch("seed", rec.seed, info.get("seed")))

                # Observation summary (matches DataCaptureWrapper behavior)
                exp_obs_summary = rec.observation_summary
                act_obs_summary = _summarize_observation(obs)
                if exp_obs_summary is not None:
                    if act_obs_summary is None:
                        mismatches.append(
                            ReplayFieldMismatch("observation_summary", exp_obs_summary, act_obs_summary)
                        )
                    else:
                        if len(exp_obs_summary) != len(act_obs_summary):
                            mismatches.append(
                                ReplayFieldMismatch("observation_summary", exp_obs_summary, act_obs_summary)
                            )
                        else:
                            for i, (a, b) in enumerate(zip(exp_obs_summary, act_obs_summary)):
                                try:
                                    if not _isclose(float(a), float(b), tol=obs_tol):
                                        mismatches.append(
                                            ReplayFieldMismatch(
                                                f"observation_summary[{i}]",
                                                float(a),
                                                float(b),
                                            )
                                        )
                                except Exception:
                                    mismatches.append(
                                        ReplayFieldMismatch(
                                            f"observation_summary[{i}]",
                                            a,
                                            b,
                                        )
                                    )

                if mismatches:
                    diff = ReplayValidationDiff(
                        run_dir=str(self._artifacts.run_dir),
                        global_step_index=int(global_idx),
                        episode_id=str(rec.episode_id),
                        episode_step=int(rec.step),
                        action=int(rec.action),
                        mismatches=tuple(mismatches),
                    )
                    raise ReplayConsistencyError(
                        f"Replay divergence at step {global_idx} (episode {rec.episode_id}, t={rec.step})",
                        diff=diff,
                    )
        finally:
            if env is not None:
                _safe_close(env)

    def iter_events(
        self,
        *,
        start_index: int = 0,
        render: bool = False,
    ) -> Iterator[ReplayStepEvent]:
        steps = self._artifacts.steps
        idx0 = max(0, min(start_index, max(0, len(steps) - 1))) if steps else 0
        for idx in range(idx0, len(steps)):
            rec = steps[idx]
            agent_xy = (rec.agent_position.x, rec.agent_position.y)
            info = {
                "seed": rec.seed,
                "agent_xy": agent_xy,
                "agent_position": agent_xy,
                "distance_to_goal": rec.distance_to_goal,
                "goal_location": self._goal_xy,
                "source_xy": self._goal_xy,
                "total_reward": math.nan,
            }
            frame = (
                _render_frame(
                    grid_size=self._grid_size,
                    agent_xy=agent_xy,
                    goal_xy=self._goal_xy,
                )
                if render
                else None
            )
            yield ReplayStepEvent(
                t=rec.step,
                obs=rec.observation_summary,
                action=rec.action,
                reward=float(rec.reward),
                terminated=rec.terminated,
                truncated=rec.truncated,
                info=info,
                frame=frame,
            )


def load_replay_engine(run_dir: str) -> ReplayEngine:
    artifacts = load_replay_artifacts(run_dir)
    return ReplayEngine(artifacts)


__all__ = [
    "ReplayArtifacts",
    "ReplayLoadError",
    "ReplayConsistencyError",
    "ReplayEnvFactory",
    "ReplayEngine",
    "ReplayFieldMismatch",
    "ReplayStepEvent",
    "ReplayValidationDiff",
    "load_replay_artifacts",
    "load_replay_engine",
]
