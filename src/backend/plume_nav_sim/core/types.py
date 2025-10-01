"""Canonical core type definitions and factories for plume_nav_sim."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLUME_SIGMA,
    DEFAULT_SOURCE_LOCATION,
    MOVEMENT_VECTORS,
)
from .enums import Action, RenderMode
from .geometry import Coordinates, GridSize, calculate_euclidean_distance
from .models import PlumeModel
from .snapshots import StateSnapshot
from .state import AgentState, EpisodeState


def _validation_error_class():
    """Return the canonical ValidationError class on demand."""

    from ..utils.exceptions import (
        ValidationError as _ValidationError,  # Local import avoids circular dependency
    )

    return _ValidationError


def _raise_validation_error(message: str, **kwargs: Any) -> None:
    """Raise the shared ValidationError with deferred import semantics."""

    raise _validation_error_class()(message, **kwargs)


CoordinateType = Union[Coordinates, Tuple[int, int], Sequence[int]]
GridDimensions = Union[GridSize, Tuple[int, int], Sequence[int]]
MovementVector = Tuple[int, int]
ActionType = Union[Action, int]
ObservationType = np.ndarray
RewardType = float
InfoType = Dict[str, Any]

RGBArray = NDArray[np.uint8]

PlumeParameters = PlumeModel


@dataclass
class PerformanceMetrics:
    """Minimal performance metrics container shared across components.

    Provides step timing capture plus a simple record_timing/get_performance_summary
    interface used by higher-level modules.
    """

    step_durations_ms: list[float] = field(default_factory=list)
    total_steps: int = 0
    other_timings_ms: Dict[str, list[float]] = field(default_factory=dict)

    def record_step(self, duration_ms: float) -> None:
        """Record a single step duration in milliseconds."""
        self.step_durations_ms.append(duration_ms)
        self.total_steps += 1

    def record_timing(self, name: str, value_ms: float) -> None:
        """Generic timing recorder; maps the "episode_step" series to record_step."""
        if name == "episode_step":
            self.record_step(value_ms)
            return
        self.other_timings_ms.setdefault(name, []).append(value_ms)

    def average_step_time_ms(self) -> float:
        """Return the rolling average step duration."""
        if not self.step_durations_ms:
            return 0.0
        return sum(self.step_durations_ms) / len(self.step_durations_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the metrics for logging or debugging."""
        return {
            "total_steps": self.total_steps,
            "average_step_time_ms": self.average_step_time_ms(),
            "step_durations_ms": list(self.step_durations_ms),
            "other_timings_ms": {k: list(v) for k, v in self.other_timings_ms.items()},
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return a lightweight summary compatible with EpisodeManager consumers."""
        total_ms = sum(self.step_durations_ms)
        return {
            "total_step_time_ms": total_ms,
            "average_step_time_ms": self.average_step_time_ms(),
            "timings": {
                "episode_step": list(self.step_durations_ms),
                **{k: list(v) for k, v in self.other_timings_ms.items()},
            },
        }


@dataclass(frozen=True)
class EnvironmentConfig:
    """Validated environment configuration shared by state and episode managers."""

    grid_size: GridDimensions = DEFAULT_GRID_SIZE
    source_location: CoordinateType = DEFAULT_SOURCE_LOCATION
    max_steps: int = DEFAULT_MAX_STEPS
    goal_radius: float = DEFAULT_GOAL_RADIUS
    plume_params: Union[PlumeParameters, Mapping[str, Any], None] = None
    enable_rendering: bool = True

    def __post_init__(self) -> None:
        grid = create_grid_size(self.grid_size)
        object.__setattr__(self, "grid_size", grid)

        source = create_coordinates(self.source_location)
        object.__setattr__(self, "source_location", source)

        plume = self._normalize_plume_params(self.plume_params)
        object.__setattr__(self, "plume_params", plume)

        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            _raise_validation_error("max_steps must be a positive integer")

        if not isinstance(self.goal_radius, (int, float)) or self.goal_radius < 0:
            _raise_validation_error("goal_radius must be a non-negative number")

        if not isinstance(self.enable_rendering, bool):
            _raise_validation_error("enable_rendering must be a boolean flag")

        if not self.source_location.is_within_bounds(self.grid_size):
            _raise_validation_error(
                "source_location must be within the provided grid_size bounds"
            )

        if plume.grid_compatibility and plume.grid_compatibility != self.grid_size:
            _raise_validation_error(
                "plume_params.grid_compatibility must match the environment grid_size"
            )

    def _normalize_plume_params(
        self, plume_params: Union[PlumeParameters, Mapping[str, Any], None]
    ) -> PlumeParameters:
        if plume_params is None:
            return PlumeParameters(
                source_location=self.source_location,
                sigma=DEFAULT_PLUME_SIGMA,
                grid_compatibility=self.grid_size,
            )

        if isinstance(plume_params, PlumeParameters):
            return plume_params

        if isinstance(plume_params, Mapping):
            params = dict(plume_params)
            source = create_coordinates(
                params.get("source_location", self.source_location)
            )
            sigma = params.get("sigma", DEFAULT_PLUME_SIGMA)
            compatibility = params.get("grid_compatibility", self.grid_size)
            grid = create_grid_size(compatibility)
            return PlumeParameters(
                source_location=source,
                sigma=float(sigma),
                grid_compatibility=grid,
            )

        _raise_validation_error(
            "plume_params must be a PlumeParameters instance or mapping"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration for downstream consumers."""
        return {
            "grid_size": self.grid_size.to_tuple(),
            "source_location": self.source_location.to_tuple(),
            "max_steps": self.max_steps,
            "goal_radius": self.goal_radius,
            "enable_rendering": self.enable_rendering,
            "plume_params": {
                "source_location": self.plume_params.source_location.to_tuple(),
                "sigma": self.plume_params.sigma,
            },
        }

    def clone_with_overrides(self, **overrides: Any) -> "EnvironmentConfig":
        """Return a new configuration with the provided overrides applied."""
        data: Dict[str, Any] = {
            "grid_size": self.grid_size,
            "source_location": self.source_location,
            "max_steps": self.max_steps,
            "goal_radius": self.goal_radius,
            "plume_params": self.plume_params,
            "enable_rendering": self.enable_rendering,
        }
        data.update(overrides)
        return EnvironmentConfig(**data)

    def validate(self) -> bool:
        """Explicit validation hook used by higher-level utilities."""
        return True


def _coerce_pair(value: Sequence[int], *, name: str) -> Tuple[int, int]:
    if len(value) != 2:
        _raise_validation_error(f"{name} must contain exactly two values")
    try:
        first, second = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise _validation_error_class()(f"{name} values must be integers") from exc
    return first, second


def create_coordinates(value: CoordinateType) -> Coordinates:
    """Create a Coordinates object from tuple or Coordinates instance.

    Args:
        value: Either a Coordinates instance or a (x, y) tuple

    Returns:
        Coordinates instance

    Raises:
        ValidationError: If value is not valid
    """
    if isinstance(value, Coordinates):
        return value
    if isinstance(value, Sequence):
        x, y = _coerce_pair(value, name="Coordinates")
        return Coordinates(x=x, y=y)
    _raise_validation_error(
        "Coordinates must be a Coordinates instance or length-2 sequence"
    )


def create_grid_size(value: GridDimensions) -> GridSize:
    """Create a GridSize object from diverse inputs."""
    if isinstance(value, GridSize):
        return value
    if isinstance(value, Sequence):
        width, height = _coerce_pair(value, name="GridSize")
        return GridSize(width=width, height=height)
    _raise_validation_error("GridSize must be a GridSize instance or length-2 sequence")


def create_agent_state(
    position: Union[AgentState, CoordinateType],
    *,
    orientation: Optional[float] = None,
    step_count: Optional[int] = None,
    total_reward: Optional[float] = None,
    goal_reached: Optional[bool] = None,
) -> AgentState:
    """Create or clone an AgentState with optional overrides.

    Args:
        position: Either AgentState to clone or coordinates for new state
        orientation: Heading in degrees (auto-normalized to [0, 360))
        step_count: Override step count
        total_reward: Override total reward
        goal_reached: Override goal reached flag

    Returns:
        AgentState with specified values
    """
    if isinstance(position, AgentState):
        base = AgentState(
            position=position.position.clone(),
            orientation=position.orientation,
            step_count=position.step_count,
            total_reward=position.total_reward,
            movement_history=list(position.movement_history),
            goal_reached=position.goal_reached,
            performance_metrics=position.performance_metrics.copy(),
        )
    else:
        base = AgentState(
            position=create_coordinates(position),
            orientation=orientation if orientation is not None else 0.0,
        )

    # Apply overrides
    if orientation is not None and isinstance(position, AgentState):
        base.orientation = float(orientation) % 360.0
    if step_count is not None:
        if step_count < 0:
            _raise_validation_error("step_count override must be non-negative")
        base.step_count = step_count
    if total_reward is not None:
        base.total_reward = float(total_reward)
    if goal_reached is not None:
        base.goal_reached = bool(goal_reached)
    return base


def create_episode_state(
    agent_state: Union[AgentState, CoordinateType],
    *,
    terminated: bool = False,
    truncated: bool = False,
    episode_id: Optional[str] = None,
) -> EpisodeState:
    """Factory for EpisodeState with sensible defaults."""
    state = EpisodeState(
        agent_state=create_agent_state(agent_state),
        terminated=terminated,
        truncated=truncated,
    )
    if episode_id is not None:
        state.episode_id = episode_id
    return state


def create_environment_config(
    config: Union[EnvironmentConfig, Mapping[str, Any], None] = None,
    **overrides: Any,
) -> EnvironmentConfig:
    """Factory helper that accepts existing configs, mappings, or keyword overrides."""
    if isinstance(config, EnvironmentConfig) and not overrides:
        return config

    data: Dict[str, Any] = {}
    if isinstance(config, EnvironmentConfig):
        data = {
            "grid_size": config.grid_size,
            "source_location": config.source_location,
            "max_steps": config.max_steps,
            "goal_radius": config.goal_radius,
            "plume_params": config.plume_params,
            "enable_rendering": config.enable_rendering,
        }
    elif isinstance(config, Mapping):
        data = dict(config)
    elif config is not None:
        _raise_validation_error("config must be EnvironmentConfig, mapping, or None")

    data.update(overrides)
    return EnvironmentConfig(**data)


def create_step_info(
    agent_state: AgentState,
    additional_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the info dictionary returned from environment step calls."""
    if not isinstance(agent_state, AgentState):
        _raise_validation_error("agent_state must be an AgentState instance")

    info: Dict[str, Any] = {
        "agent_position": agent_state.position.to_tuple(),
        "step_count": agent_state.step_count,
        "total_reward": agent_state.total_reward,
        "goal_reached": agent_state.goal_reached,
        "timestamp": time.time(),
    }
    if additional_info:
        info.update(additional_info)
    return info


def validate_action(action: ActionType) -> Action:
    """Normalize and validate an action value, returning the Action enum."""
    if isinstance(action, Action):
        return action
    if isinstance(action, int) and action in MOVEMENT_VECTORS:
        return Action(action)
    _raise_validation_error(
        "Action must be an Action enum or integer in the action space"
    )


def get_movement_vector(action: ActionType) -> MovementVector:
    """Return the movement vector associated with an action."""
    validated = validate_action(action)
    return validated.to_vector()


def __getattr__(name: str) -> Any:
    if name == "ValidationError":
        cls = _validation_error_class()
        globals()["ValidationError"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Action",
    "ActionType",
    "AgentState",
    "CoordinateType",
    "Coordinates",
    "EnvironmentConfig",
    "EpisodeState",
    "GridDimensions",
    "GridSize",
    "InfoType",
    "MovementVector",
    "ObservationType",
    "PerformanceMetrics",
    "PlumeParameters",
    "RGBArray",
    "RenderMode",
    "RewardType",
    "StateSnapshot",
    "calculate_euclidean_distance",
    "create_agent_state",
    "create_coordinates",
    "create_environment_config",
    "create_episode_state",
    "create_grid_size",
    "create_step_info",
    "get_movement_vector",
    "validate_action",
]
