"""Public package initializer exposing canonical constants, types, and utilities."""

from __future__ import annotations

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
_ENV_IMPORT_ERROR: Optional[Exception] = None
_DIRECT_INJECTION_KEYS = frozenset(
    {"sensor_model", "action_model", "reward_fn", "video_config"}
)
_REMOVED_MAKE_ENV_ALIASES = {
    "plume_type": "Use plume='static' or plume='movie'.",
    "video_path": "Use movie_path instead.",
    "video_fps": "Use movie_fps instead.",
    "video_pixel_to_grid": "Use movie_pixel_to_grid instead.",
    "video_origin": "Use movie_origin instead.",
    "video_extent": "Use movie_extent instead.",
    "video_step_policy": "Use movie_step_policy instead.",
    "video_data": "Use movie_data instead.",
}
try:
    from .envs import PlumeEnv as _PlumeEnv
    from .envs import create_plume_env as _create_plume_env

    PlumeEnv = _PlumeEnv
    create_plume_env = _create_plume_env
except ImportError as env_import_error:  # pragma: no cover - optional dependency guard
    PlumeEnv = None
    create_plume_env = None
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


def _normalize_make_env_kwargs(kwargs: Mapping[str, object]) -> dict[str, Any]:
    selector_kwargs: dict[str, Any] = dict(kwargs)

    removed_aliases = sorted(
        key for key in selector_kwargs if key in _REMOVED_MAKE_ENV_ALIASES
    )
    if removed_aliases:
        alias = removed_aliases[0]
        raise ValidationError(
            f"{alias} is no longer supported by make_env(). {_REMOVED_MAKE_ENV_ALIASES[alias]}",
            parameter_name=alias,
            parameter_value=selector_kwargs.get(alias),
        )

    forbidden_keys = sorted(key for key in selector_kwargs if key in _DIRECT_INJECTION_KEYS)
    plume_value = selector_kwargs.get("plume")
    if "plume" in selector_kwargs and plume_value is not None and not isinstance(
        plume_value, str
    ):
        forbidden_keys.append("plume")
    if forbidden_keys:
        raise ValidationError(
            "make_env() accepts selector kwargs only. Use PlumeEnv(...) for direct component injection.",
            parameter_name="kwargs",
            parameter_value=forbidden_keys,
        )

    source_location = selector_kwargs.pop("source_location", None)
    if source_location is not None and "goal_location" not in selector_kwargs:
        selector_kwargs["goal_location"] = source_location

    plume_params = selector_kwargs.pop("plume_params", None)
    if plume_params is not None:
        if not isinstance(plume_params, Mapping):
            raise ValidationError(
                "plume_params must be mapping-like",
                parameter_name="plume_params",
                parameter_value=plume_params,
            )
        if "plume_sigma" not in selector_kwargs and "sigma" in plume_params:
            selector_kwargs["plume_sigma"] = float(plume_params["sigma"])

    if isinstance(plume_value, str):
        normalized_plume = plume_value.strip().lower()
        if normalized_plume not in {"static", "movie"}:
            raise ValidationError(
                "plume must be 'static' or 'movie'",
                parameter_name="plume",
                parameter_value=plume_value,
                expected_format="static|movie",
            )
        selector_kwargs["plume"] = normalized_plume

    return selector_kwargs


def make_env(**kwargs):
    if create_plume_env is None:
        raise RuntimeError(
            "PlumeEnv not available. Ensure gymnasium and dependencies are installed."
        )
    return create_plume_env(**_normalize_make_env_kwargs(kwargs))


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
    "initialize_package",
    "get_package_info",
    "get_conf_dir",
]

__version__ = PACKAGE_VERSION
