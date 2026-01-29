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
    "ReplayStepEvent",
    "load_replay_artifacts",
    "load_replay_engine",
]
