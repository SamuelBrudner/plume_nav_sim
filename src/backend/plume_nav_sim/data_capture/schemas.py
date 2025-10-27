from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"


class RunMeta(BaseModel):
    """Metadata describing a run and its environment/configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str
    experiment: Optional[str] = None
    package_version: Optional[str] = None
    git_sha: Optional[str] = None
    start_time: datetime

    # Serialized EnvironmentConfig
    env_config: dict

    # Seeding information
    base_seed: Optional[int] = None
    episode_seeds: Optional[List[int]] = None

    # System metadata for provenance
    class SystemInfo(BaseModel):
        """Host and runtime information captured for provenance."""

        model_config = ConfigDict(extra="forbid", frozen=True)
        hostname: Optional[str] = None
        platform: Optional[str] = None
        python_version: Optional[str] = None
        pid: Optional[int] = None
        user: Optional[str] = None

    system: SystemInfo = Field(default_factory=SystemInfo)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {v}")
        return v


class Position(BaseModel):
    """2D integer grid coordinate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    x: int
    y: int


class StepRecord(BaseModel):
    """Per-step event record for analysis and auditing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION)
    ts: float  # seconds since epoch
    run_id: str
    episode_id: str
    step: PositiveInt
    action: int
    reward: float
    terminated: bool
    truncated: bool
    agent_position: Position
    distance_to_goal: float = Field(ge=0.0)
    # Keep observation summary compact (scalar or short vector)
    observation_summary: Optional[List[float]] = None
    seed: Optional[int] = None

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {v}")
        return v


class EpisodeRecord(BaseModel):
    """Summary of an episode with totals and final state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION)
    run_id: str
    episode_id: str
    terminated: bool
    truncated: bool
    total_steps: int = Field(ge=0)
    total_reward: float
    final_position: Position
    final_distance_to_goal: Optional[float] = Field(default=None, ge=0.0)
    duration_ms: Optional[float] = Field(default=None, ge=0.0)
    avg_step_time_ms: Optional[float] = Field(default=None, ge=0.0)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version: {v}")
        return v
