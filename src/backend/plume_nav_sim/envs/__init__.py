"""Environment package public exports and the remaining thin constructor shim."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym

from .._compat import ValidationError, validate_coordinates, validate_grid_size
from ..constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_SOURCE_LOCATION,
)
from .plume_env import PlumeEnv, create_plume_env

SUPPORTED_ENVIRONMENTS = ("plume_env", "plume", "PlumeEnv")


def create_environment(
    env_type: Optional[str] = None,
    grid_size: Optional[tuple[int, int]] = None,
    source_location: Optional[tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    goal_radius: Optional[float] = None,
    render_mode: Optional[str] = None,
    env_options: Optional[dict[str, Any]] = None,
) -> gym.Env:
    """Compatibility constructor kept as a thin adapter over the active factories."""

    effective_env_type = env_type or "plume_env"
    if effective_env_type not in SUPPORTED_ENVIRONMENTS:
        raise ValidationError(
            f"Unsupported environment type: {effective_env_type}",
            parameter_name="env_type",
            parameter_value=effective_env_type,
            expected_format=f"One of: {SUPPORTED_ENVIRONMENTS}",
        )

    effective_grid = validate_grid_size(grid_size or DEFAULT_GRID_SIZE).to_tuple()
    effective_source = validate_coordinates(
        source_location or DEFAULT_SOURCE_LOCATION,
        effective_grid,
    ).to_tuple()

    effective_max_steps = DEFAULT_MAX_STEPS if max_steps is None else max_steps
    if not isinstance(effective_max_steps, int) or effective_max_steps <= 0:
        raise ValidationError(
            "max_steps must be a positive integer",
            parameter_name="max_steps",
            parameter_value=effective_max_steps,
        )

    effective_goal_radius = (
        DEFAULT_GOAL_RADIUS if goal_radius is None else float(goal_radius)
    )
    if effective_goal_radius < 0:
        raise ValidationError(
            "goal_radius must be non-negative",
            parameter_name="goal_radius",
            parameter_value=effective_goal_radius,
        )

    effective_render_mode = "rgb_array" if render_mode is None else render_mode

    plume_env_kwargs = dict(env_options or {})
    plume_env_kwargs.update(
        grid_size=effective_grid,
        source_location=effective_source,
        max_steps=effective_max_steps,
        goal_radius=effective_goal_radius,
        render_mode=effective_render_mode,
    )
    return create_plume_env(**plume_env_kwargs)


__all__ = [
    "PlumeEnv",
    "create_plume_env",
    "create_environment",
]
