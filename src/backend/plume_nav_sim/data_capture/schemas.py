from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class Position:
    x: int
    y: int


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    experiment: str
    start_time: str
    env_config: dict
    base_seed: Optional[int] = None
    package_version: Optional[str] = None
    git_sha: Optional[str] = None
    system: Optional[dict] = None
    extra: Optional[dict] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StepRecord:
    ts: float
    run_id: str
    episode_id: str
    step: int
    action: int
    reward: float
    terminated: bool
    truncated: bool
    agent_position: Position
    distance_to_goal: float
    observation_summary: Optional[list[float]] = None
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeRecord:
    run_id: str
    episode_id: str
    terminated: bool
    truncated: bool
    total_steps: int
    total_reward: float
    final_position: Position
    final_distance_to_goal: float
    duration_ms: Optional[float] = None
    avg_step_time_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)
