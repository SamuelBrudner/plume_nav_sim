"""Compatibility shims for slimmed plume_nav_sim utilities."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import gymnasium.utils.seeding as gym_seeding
except Exception:  # pragma: no cover
    gym_seeding = None  # type: ignore[assignment]

import gymnasium.spaces as spaces

from .constants import (
    ACTION_SPACE_SIZE,
    SEED_MAX_VALUE,
    SEED_MIN_VALUE,
    SUPPORTED_RENDER_MODES,
)
from .core.types import (
    Action,
    Coordinates,
    GridSize,
    create_coordinates,
    create_grid_size,
)


class ValidationError(ValueError):
    def __init__(
        self,
        message: str,
        parameter_name: str | None = None,
        parameter_value: Any = None,
        expected_format: str | None = None,
        parameter_constraints: dict | None = None,
        context: dict | None = None,
        invalid_value: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        if parameter_value is None and invalid_value is not None:
            parameter_value = invalid_value
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.invalid_value = parameter_value
        self.expected_format = expected_format
        self.parameter_constraints = parameter_constraints
        self.context = context
        if kwargs:
            self.extra = kwargs


class ConfigError(ValueError):
    def __init__(
        self,
        message: str,
        config_parameter: str | None = None,
        parameter_value: Any = None,
        valid_options: Any = None,
        invalid_value: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        if parameter_value is None and invalid_value is not None:
            parameter_value = invalid_value
        self.config_parameter = config_parameter
        self.parameter_value = parameter_value
        self.invalid_value = parameter_value
        self.valid_options = valid_options
        if kwargs:
            self.extra = kwargs


class _KwargError(RuntimeError):
    def __init__(self, message: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(message)
        self.message = message
        self.__dict__.update(kwargs)


class ComponentError(_KwargError):
    pass


class StateError(_KwargError):
    pass


class RenderingError(_KwargError):
    pass


class ResourceError(_KwargError):
    pass


ConfigurationError = ConfigError


def validate_seed_value(
    seed: Any,
    *,
    allow_none: bool = True,
    strict_type_checking: bool = True,
    min_value: int = SEED_MIN_VALUE,
    max_value: int = SEED_MAX_VALUE,
) -> Optional[int]:
    if seed is None:
        if allow_none:
            return None
        raise ValidationError("Seed cannot be None", parameter_name="seed")

    if strict_type_checking and not isinstance(seed, (int, np.integer)):
        raise ValidationError(
            "Seed must be an integer",
            parameter_name="seed",
            parameter_value=seed,
        )
    try:
        seed_int = int(seed)
    except Exception as exc:
        raise ValidationError(
            "Seed must be convertible to int",
            parameter_name="seed",
            parameter_value=seed,
        ) from exc

    if seed_int < min_value or seed_int > max_value:
        raise ValidationError(
            "Seed out of range",
            parameter_name="seed",
            parameter_value=seed_int,
            expected_format=f"[{min_value}, {max_value}]",
        )
    return seed_int


def create_seeded_rng(seed: Optional[int] = None) -> tuple[np.random.Generator, int]:
    if gym_seeding is None:  # pragma: no cover - optional dependency guard
        raise ImportError("gymnasium is required for seeding utilities")
    rng, used = gym_seeding.np_random(seed)
    used_seed = int(used) if used is not None else int(rng.integers(0, 2**32 - 1))
    return rng, used_seed


class SeedManager:
    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        default_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if seed is None and default_seed is not None:
            seed = default_seed
        self.rng, self.seed_value = create_seeded_rng(seed)
        self.operation_count = 0
        for key, value in kwargs.items():
            setattr(self, key, value)

    def seed(self, seed: Optional[int]) -> None:
        self.rng, self.seed_value = create_seeded_rng(seed)
        self.operation_count += 1

    def generate_random_position(
        self, grid_size: GridSize, *, exclude_position: Coordinates | None = None
    ) -> Coordinates:
        grid = create_grid_size(grid_size)
        for _ in range(100):
            x = int(self.rng.integers(0, grid.width))
            y = int(self.rng.integers(0, grid.height))
            coord = Coordinates(x=x, y=y)
            if exclude_position is not None and coord == exclude_position:
                continue
            return coord
        raise RuntimeError("Failed to sample a valid position")


def validate_grid_size(grid_size: GridSize | tuple[int, int]) -> GridSize:
    grid = create_grid_size(grid_size)
    if grid.width <= 0 or grid.height <= 0:
        raise ValidationError(
            "grid_size must have positive dimensions",
            parameter_name="grid_size",
            parameter_value=grid.to_tuple(),
        )
    return grid


def validate_coordinates(
    coordinates: Coordinates | tuple[int, int],
    grid_bounds: GridSize | tuple[int, int] | None = None,
) -> Coordinates:
    coords = create_coordinates(coordinates)
    if coords.x < 0 or coords.y < 0:
        raise ValidationError(
            "coordinates must be non-negative",
            parameter_name="coordinates",
            parameter_value=coords.to_tuple(),
        )
    if grid_bounds is not None:
        grid = create_grid_size(grid_bounds)
        if not grid.contains(coords):
            raise ValidationError(
                "coordinates outside grid bounds",
                parameter_name="coordinates",
                parameter_value=coords.to_tuple(),
                expected_format=grid.to_tuple(),
            )
    return coords


def validate_action_parameter(action: Any, *, allow_enum_types: bool = True) -> int:
    if allow_enum_types and isinstance(action, Action):
        return int(action)
    if isinstance(action, (int, np.integer)):
        action_id = int(action)
        if 0 <= action_id < ACTION_SPACE_SIZE:
            return action_id
    raise ValidationError(
        "Invalid action",
        parameter_name="action",
        parameter_value=action,
        expected_format=f"int in [0, {ACTION_SPACE_SIZE - 1}]",
    )


def validate_action_input(action: Any) -> int:
    return validate_action_parameter(action, allow_enum_types=True)


def validate_render_mode(mode: Any) -> str:
    if isinstance(mode, str):
        mode_value = mode
    else:
        mode_value = getattr(mode, "value", None) or str(mode)
    if mode_value not in SUPPORTED_RENDER_MODES:
        raise ValidationError(
            "Unsupported render mode",
            parameter_name="render_mode",
            parameter_value=mode_value,
            expected_format=str(SUPPORTED_RENDER_MODES),
        )
    return mode_value


def _is_discrete_subset(policy_space: spaces.Discrete, env_space: spaces.Space) -> bool:
    if not isinstance(env_space, spaces.Discrete):
        return False
    return int(policy_space.n) <= int(env_space.n)


def _is_multidiscrete_subset(
    policy_space: spaces.MultiDiscrete, env_space: spaces.Space
) -> bool:
    if not isinstance(env_space, spaces.MultiDiscrete):
        return False
    return np.all(policy_space.nvec <= env_space.nvec)


def _is_multibinary_subset(
    policy_space: spaces.MultiBinary, env_space: spaces.Space
) -> bool:
    if not isinstance(env_space, spaces.MultiBinary):
        return False
    return int(policy_space.n) <= int(env_space.n)


def _is_box_subset(policy_space: spaces.Box, env_space: spaces.Space) -> bool:
    if not isinstance(env_space, spaces.Box):
        return False
    return (
        tuple(policy_space.shape) == tuple(env_space.shape)
        and np.all(policy_space.low >= env_space.low)
        and np.all(policy_space.high <= env_space.high)
    )


def _is_tuple_subset(policy_space: spaces.Tuple, env_space: spaces.Space) -> bool:
    if not isinstance(env_space, spaces.Tuple):
        return False
    return len(policy_space.spaces) == len(env_space.spaces) and all(
        is_space_subset(ps, es) for ps, es in zip(policy_space.spaces, env_space.spaces)
    )


def _is_dict_subset(policy_space: spaces.Dict, env_space: spaces.Space) -> bool:
    if not isinstance(env_space, spaces.Dict):
        return False
    if not set(policy_space.spaces.keys()).issubset(env_space.spaces.keys()):
        return False
    return all(
        is_space_subset(policy_space.spaces[k], env_space.spaces[k])
        for k in policy_space.spaces.keys()
    )


_SPACE_SUBSET_CHECKERS: tuple[tuple[type[spaces.Space], Any], ...] = (
    (spaces.Discrete, _is_discrete_subset),
    (spaces.MultiDiscrete, _is_multidiscrete_subset),
    (spaces.MultiBinary, _is_multibinary_subset),
    (spaces.Box, _is_box_subset),
    (spaces.Tuple, _is_tuple_subset),
    (spaces.Dict, _is_dict_subset),
)


def is_space_subset(policy_space: spaces.Space, env_space: spaces.Space) -> bool:
    if policy_space is env_space:
        return True
    for space_type, checker in _SPACE_SUBSET_CHECKERS:
        if isinstance(policy_space, space_type):
            return checker(policy_space, env_space)
    return False


__all__ = [
    "ValidationError",
    "ConfigError",
    "ConfigurationError",
    "ComponentError",
    "StateError",
    "RenderingError",
    "ResourceError",
    "validate_seed_value",
    "create_seeded_rng",
    "SeedManager",
    "validate_grid_size",
    "validate_coordinates",
    "validate_action_input",
    "validate_action_parameter",
    "validate_render_mode",
    "is_space_subset",
]
