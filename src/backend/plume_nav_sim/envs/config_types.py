"""Lightweight environment configuration types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..core.types import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLUME_SIGMA,
    DEFAULT_SOURCE_LOCATION,
    Coordinates,
    CoordinateType,
    GridDimensions,
    GridSize,
    create_coordinates,
    create_grid_size,
)
from ..utils.exceptions import ValidationError


@dataclass(frozen=True)
class EnvironmentConfig:
    grid_size: GridDimensions = DEFAULT_GRID_SIZE
    source_location: CoordinateType = DEFAULT_SOURCE_LOCATION
    max_steps: int = DEFAULT_MAX_STEPS
    goal_radius: float = DEFAULT_GOAL_RADIUS
    plume_params: Mapping[str, Any] | None = None
    enable_rendering: bool = True

    def __post_init__(self) -> None:
        try:
            grid = create_grid_size(self.grid_size)
        except Exception as exc:
            raise ValidationError(
                "grid_size must be GridSize or length-2 sequence",
                parameter_name="grid_size",
                parameter_value=str(self.grid_size),
            ) from exc
        try:
            source = create_coordinates(self.source_location)
        except Exception as exc:
            raise ValidationError(
                "source_location must be Coordinates or length-2 sequence",
                parameter_name="source_location",
                parameter_value=str(self.source_location),
            ) from exc
        object.__setattr__(self, "grid_size", grid)
        object.__setattr__(self, "source_location", source)
        try:
            plume_params = dict(self.plume_params) if self.plume_params else {}
        except Exception as exc:
            raise ValidationError(
                "plume_params must be mapping-like",
                parameter_name="plume_params",
                parameter_value=str(self.plume_params),
            ) from exc
        plume_params.setdefault("source_location", source)
        plume_params.setdefault("sigma", float(DEFAULT_PLUME_SIGMA))
        object.__setattr__(self, "plume_params", plume_params)
        self.validate()

    def validate(self, strict_mode: bool = False, **_: Any) -> bool:
        grid = self.grid_size
        if grid.width <= 0 or grid.height <= 0:
            raise ValidationError(
                "grid_size must be positive",
                parameter_name="grid_size",
                parameter_value=grid.to_tuple(),
            )
        if not grid.contains(self.source_location):
            raise ValidationError(
                "source_location must be within grid bounds",
                parameter_name="source_location",
                parameter_value=self.source_location.to_tuple(),
            )
        if not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValidationError(
                "max_steps must be a positive integer",
                parameter_name="max_steps",
                parameter_value=str(self.max_steps),
            )
        if not isinstance(self.goal_radius, (int, float)) or self.goal_radius < 0:
            raise ValidationError(
                "goal_radius must be non-negative",
                parameter_name="goal_radius",
                parameter_value=str(self.goal_radius),
            )
        if self.plume_params:
            sigma = self.plume_params.get("sigma")
            if sigma is not None and (
                not isinstance(sigma, (int, float)) or sigma <= 0
            ):
                raise ValidationError(
                    "plume_params.sigma must be positive",
                    parameter_name="plume_params.sigma",
                    parameter_value=str(sigma),
                )
        if strict_mode and self.max_steps > 100000:
            raise ValidationError(
                "max_steps too large for strict validation",
                parameter_name="max_steps",
                parameter_value=str(self.max_steps),
            )
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_size": self.grid_size.to_tuple(),
            "source_location": self.source_location.to_tuple(),
            "max_steps": self.max_steps,
            "goal_radius": self.goal_radius,
            "enable_rendering": self.enable_rendering,
            "plume_params": dict(self.plume_params) if self.plume_params else {},
        }


def create_environment_config(
    config: EnvironmentConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> EnvironmentConfig:
    if isinstance(config, EnvironmentConfig) and not overrides:
        return config
    data: dict[str, Any] = {}
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
        raise TypeError("config must be EnvironmentConfig, mapping, or None")
    data.update(overrides)
    return EnvironmentConfig(**data)


__all__ = ["EnvironmentConfig", "create_environment_config"]
