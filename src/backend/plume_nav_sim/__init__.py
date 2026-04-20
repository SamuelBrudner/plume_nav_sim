"""Public package initializer exposing canonical constants, types, and utilities."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Optional

from .core import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLUME_SIGMA,
    DEFAULT_SOURCE_LOCATION,
    ENVIRONMENT_ID,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    Action,
    ActionType,
    AgentState,
    Coordinates,
    CoordinateType,
    GridDimensions,
    GridSize,
    InfoType,
    MovementVector,
    ObservationType,
    RenderMode,
    calculate_euclidean_distance,
    create_coordinates,
    create_grid_size,
    get_movement_vector,
    validate_action,
)
from .envs.config_types import EnvironmentConfig, create_environment_config
from ._compat import ValidationError

# Optional environment exports (import may fail if optional deps are missing)
# Predeclare symbols with precise optional types to satisfy mypy strict rules.
PlumeEnv: Optional[object]
create_plume_env: Optional[object]
create_component_environment: Optional[object]
_ENV_IMPORT_ERROR: Optional[Exception] = None
_COMPATIBILITY_COMPONENT_KEYS = frozenset(
    {
        "action_type",
        "observation_type",
        "reward_type",
        "plume_sigma",
        "step_size",
        "enable_wind",
        "wind_direction_deg",
        "wind_speed",
        "wind_vector",
        "wind_noise_std",
    }
)
_COMPATIBILITY_COMPONENT_PREFIXES = ("movie_",)
_STABLE_ONLY_ENV_KEYS = frozenset(
    {
        "plume_type",
        "sensor_model",
        "action_model",
        "reward_fn",
        "video_path",
        "video_data",
        "video_fps",
        "video_pixel_to_grid",
        "video_origin",
        "video_extent",
        "video_step_policy",
        "video_config",
    }
)

try:
    from .envs import PlumeEnv as _PlumeEnv
    from .envs import create_component_environment as _create_component_environment
    from .envs import create_plume_env as _create_plume_env

    PlumeEnv = _PlumeEnv
    create_plume_env = _create_plume_env
    create_component_environment = _create_component_environment
except Exception as env_import_error:  # pragma: no cover - optional dependency guard
    PlumeEnv = None
    create_plume_env = None
    create_component_environment = None
    _ENV_IMPORT_ERROR = env_import_error


def initialize_package(  # noqa: C901
    *,
    configure_logging: bool = True,
    auto_register_environment: bool = True,
    validate_constants: bool = False,
) -> Dict[str, object]:
    status: Dict[str, object] = {
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "environment_id": ENVIRONMENT_ID,
        "initialized": True,
        "logging_configured": False,
        "environment_registered": False,
        "constants_validated": False,
        "errors": [],
    }

    if configure_logging:
        try:
            from plume_nav_sim.logging import configure_development_logging

            configure_development_logging()
            status["logging_configured"] = True
        except Exception as exc:  # pragma: no cover - defensive guard
            _record_init_error(status, "configure_logging", exc)
    if auto_register_environment:
        try:
            from plume_nav_sim.registration import ensure_registered

            ensure_registered()
            status["environment_registered"] = True
        except Exception as exc:  # pragma: no cover - defensive guard
            _record_init_error(status, "register_environment", exc)
    if validate_constants:
        try:
            from plume_nav_sim.core.constants import validate_constant_consistency

            is_valid, report = validate_constant_consistency(strict_mode=False)
            status["constants_validated"] = bool(is_valid)
            status["constant_validation_report"] = report
        except Exception as exc:  # pragma: no cover - defensive guard
            _record_init_error(status, "validate_constants", exc)
    return status


def _record_init_error(status, step, exc):
    # errors is a List[dict[str, str]] but stored in a heterogeneous mapping
    errors = status.get("errors")
    if isinstance(errors, list):
        errors.append({"step": step, "error": str(exc)})
    status["initialized"] = False


def get_package_info(
    *,
    include_environment_info: bool = True,
    include_defaults: bool = True,
    include_performance_targets: bool = True,
    include_registration_status: bool = False,
) -> Dict[str, object]:
    """Return package metadata for tooling."""

    info: Dict[str, object] = {
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "environment_id": ENVIRONMENT_ID,
    }

    if include_defaults:
        info["default_configuration"] = {
            "grid_size": DEFAULT_GRID_SIZE,
            "source_location": DEFAULT_SOURCE_LOCATION,
            "max_steps": DEFAULT_MAX_STEPS,
            "goal_radius": DEFAULT_GOAL_RADIUS,
            "plume_sigma": DEFAULT_PLUME_SIGMA,
        }

    if include_performance_targets:
        info["performance_targets"] = {
            "step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
            "rgb_render_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
        }

    if include_environment_info:

        environment_details: Dict[str, object] = {"available": []}

        if PlumeEnv is not None:
            environment_details["available"].append("PlumeEnv")
            environment_details["factory"] = "create_plume_env"
            environment_details["module"] = "plume_nav_sim.envs.plume_env"
        if create_component_environment is not None:
            environment_details["available"].append("ComponentBasedEnvironment")
        if not environment_details["available"]:
            environment_details["error"] = str(_ENV_IMPORT_ERROR)

        info["environment"] = environment_details

    if include_registration_status:
        try:
            from plume_nav_sim.registration import get_registration_status

            info["registration"] = get_registration_status()
        except Exception as exc:  # pragma: no cover - defensive guard
            info["registration_error"] = str(exc)

    return info


def _compatibility_component_keys(kwargs: Mapping[str, object]) -> set[str]:
    keys = {key for key in kwargs if key in _COMPATIBILITY_COMPONENT_KEYS}
    keys.update(
        key
        for key in kwargs
        if any(key.startswith(prefix) for prefix in _COMPATIBILITY_COMPONENT_PREFIXES)
    )
    return keys


def _stable_only_keys(kwargs: Mapping[str, object]) -> set[str]:
    keys = {key for key in kwargs if key in _STABLE_ONLY_ENV_KEYS}
    plume_value = kwargs.get("plume")
    if "plume" in kwargs and not isinstance(plume_value, str):
        keys.add("plume")
    return keys


def _classify_make_env_kwargs(kwargs: Mapping[str, object]) -> str:
    compatibility_keys = _compatibility_component_keys(kwargs)
    stable_keys = _stable_only_keys(kwargs)
    if compatibility_keys and stable_keys:
        raise ValidationError(
            "Cannot mix deprecated component kwargs with stable PlumeEnv kwargs",
            parameter_name="kwargs",
            parameter_value=sorted(compatibility_keys | stable_keys),
            context={
                "deprecated_component_kwargs": sorted(compatibility_keys),
                "stable_plume_env_kwargs": sorted(stable_keys),
            },
        )
    return "component" if compatibility_keys else "stable"


def _normalize_component_make_env_kwargs(kwargs: Mapping[str, object]) -> dict[str, Any]:
    component_kwargs: dict[str, Any] = dict(kwargs)
    source_location = component_kwargs.pop("source_location", None)
    if source_location is not None and "goal_location" not in component_kwargs:
        component_kwargs["goal_location"] = source_location
    plume_params = component_kwargs.pop("plume_params", None)
    if plume_params is not None and "plume_sigma" not in component_kwargs:
        if isinstance(plume_params, Mapping):
            sigma_value = plume_params.get("sigma")
            if sigma_value is not None:
                component_kwargs["plume_sigma"] = float(sigma_value)
    plume_value = component_kwargs.get("plume")
    if isinstance(plume_value, str):
        normalized_plume = plume_value.strip().lower()
        if normalized_plume not in {"static", "movie"}:
            raise ValidationError(
                "plume string must be 'static' or 'movie'",
                parameter_name="plume",
                parameter_value=plume_value,
                expected_format="static|movie",
            )
        component_kwargs["plume"] = normalized_plume
    return component_kwargs


def _normalize_stable_make_env_kwargs(kwargs: Mapping[str, object]) -> dict[str, Any]:
    stable_kwargs: dict[str, Any] = dict(kwargs)
    plume_value = stable_kwargs.get("plume")
    stable_video_keys = {key for key in stable_kwargs if key.startswith("video_")}

    if "plume_type" in stable_kwargs and isinstance(plume_value, str):
        raise ValidationError(
            "plume and plume_type must not both be provided as string selectors",
            parameter_name="plume",
            parameter_value=plume_value,
        )

    if not isinstance(plume_value, str):
        return stable_kwargs

    normalized_plume = plume_value.strip().lower()
    stable_kwargs.pop("plume", None)

    if normalized_plume == "static":
        if stable_video_keys:
            raise ValidationError(
                "video_* kwargs require plume='movie' or plume_type='video'",
                parameter_name="plume",
                parameter_value=plume_value,
                expected_format="movie|video",
            )
        stable_kwargs.setdefault("plume_type", "gaussian")
        return stable_kwargs

    if normalized_plume == "movie":
        warnings.warn(
            "make_env(plume='movie') is deprecated; use plume_type='video' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        stable_kwargs["plume_type"] = "video"
        return stable_kwargs

    raise ValidationError(
        "plume string must be 'static' or 'movie'",
        parameter_name="plume",
        parameter_value=plume_value,
        expected_format="static|movie|ConcentrationField",
    )


def make_env(**kwargs):
    route = _classify_make_env_kwargs(kwargs)
    if route == "component":
        if create_component_environment is None:
            raise RuntimeError(
                "Component environment factory not available to handle compatibility kwargs."
            )
        component_kwargs = _normalize_component_make_env_kwargs(kwargs)

        warnings.warn(
            "make_env received compatibility kwargs; routing to component environment factory.",
            DeprecationWarning,
            stacklevel=2,
        )
        return create_component_environment(**component_kwargs)

    if create_plume_env is None:
        raise RuntimeError(
            "PlumeEnv not available. Ensure gymnasium and dependencies are installed."
        )
    return create_plume_env(**_normalize_stable_make_env_kwargs(kwargs))


def get_conf_dir():
    try:
        from pathlib import Path

        # src/backend/plume_nav_sim/__init__.py -> conf at src/backend/conf
        return Path(__file__).resolve().parents[1] / "conf"
    except Exception:  # pragma: no cover - defensive fallback
        return None  # type: ignore[return-value]


__all__ = [
    # Recommended entry point
    "make_env",
    # Metadata
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "ENVIRONMENT_ID",
    # Default constants
    "DEFAULT_GRID_SIZE",
    "DEFAULT_SOURCE_LOCATION",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_GOAL_RADIUS",
    "DEFAULT_PLUME_SIGMA",
    "PERFORMANCE_TARGET_STEP_LATENCY_MS",
    "PERFORMANCE_TARGET_RGB_RENDER_MS",
    # Core types
    "Action",
    "RenderMode",
    "Coordinates",
    "GridSize",
    "AgentState",
    "EnvironmentConfig",
    # Type aliases
    "ActionType",
    "CoordinateType",
    "GridDimensions",
    "MovementVector",
    "ObservationType",
    "InfoType",
    # Utility functions
    "calculate_euclidean_distance",
    "create_coordinates",
    "create_grid_size",
    "create_environment_config",
    "validate_action",
    "get_movement_vector",
    # Environments
    "PlumeEnv",
    "create_plume_env",
    "create_component_environment",
    "initialize_package",
    "get_package_info",
    "get_conf_dir",
]

__version__ = PACKAGE_VERSION
