"""Public package initializer exposing canonical constants, types, and utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, cast

from typing_extensions import NotRequired, TypedDict

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
    EnvironmentConfig,
    EpisodeState,
    GridDimensions,
    GridSize,
    InfoType,
    MovementVector,
    ObservationType,
    PerformanceMetrics,
    PlumeParameters,
    RenderMode,
    RewardType,
    StateSnapshot,
    calculate_euclidean_distance,
    create_agent_state,
    create_coordinates,
    create_environment_config,
    create_episode_state,
    create_grid_size,
    create_step_info,
    get_movement_vector,
    validate_action,
)

# Optional environment exports (import may fail if optional deps are missing)
# Predeclare symbols with precise optional types to satisfy mypy strict rules.
PlumeSearchEnv: Optional[object]
create_plume_search_env: Optional[object]
_ENV_IMPORT_ERROR: Optional[Exception] = None

try:
    from .envs import PlumeSearchEnv as _PlumeSearchEnv
    from .envs import create_plume_search_env as _create_plume_search_env

    PlumeSearchEnv = _PlumeSearchEnv
    create_plume_search_env = _create_plume_search_env
except Exception as env_import_error:  # pragma: no cover - optional dependency guard
    PlumeSearchEnv = None
    create_plume_search_env = None
    _ENV_IMPORT_ERROR = env_import_error


def initialize_package(
    *,
    configure_logging: bool = True,
    auto_register_environment: bool = True,
    validate_constants: bool = False,
) -> Dict[str, object]:
    """Initialize core plume_nav_sim subsystems and return status information.

    This compatibility helper keeps the legacy package-level bootstrap routine
    available while delegating to the modern module structure.  It conditionally
    configures logging, registers the Gymnasium environment, and can validate the
    exported constants without raising hard failures when optional dependencies
    are absent.

    Args:
        configure_logging: Configure the development logging profile using the
            :mod:`plume_nav_sim.utils.logging` helpers.
        auto_register_environment: Register the default environment with
            Gymnasium if the dependency is available.
        validate_constants: Run the lightweight constant consistency validation
            utility from :mod:`plume_nav_sim.core.constants`.

    Returns:
        A dictionary containing initialization metadata, success flags, and any
        captured errors for optional steps.
    """

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
            from plume_nav_sim.utils.logging import configure_logging_for_development

            configure_logging_for_development()
            status["logging_configured"] = True
        except Exception as exc:  # pragma: no cover - defensive guard
            # errors is a List[dict[str, str]] but stored in a heterogeneous mapping
            errors = status.get("errors")
            if isinstance(errors, list):
                errors.append({"step": "configure_logging", "error": str(exc)})
            status["initialized"] = False

    if auto_register_environment:
        try:
            from plume_nav_sim.registration import ensure_registered

            ensure_registered()
            status["environment_registered"] = True
        except Exception as exc:  # pragma: no cover - defensive guard
            errors = status.get("errors")
            if isinstance(errors, list):
                errors.append({"step": "register_environment", "error": str(exc)})
            status["initialized"] = False

    if validate_constants:
        try:
            from plume_nav_sim.core.constants import validate_constant_consistency

            is_valid, report = validate_constant_consistency(strict_mode=False)
            status["constants_validated"] = bool(is_valid)
            status["constant_validation_report"] = report
        except Exception as exc:  # pragma: no cover - defensive guard
            errors = status.get("errors")
            if isinstance(errors, list):
                errors.append({"step": "validate_constants", "error": str(exc)})
            status["initialized"] = False

    return status


def get_package_info(
    *,
    include_environment_info: bool = True,
    include_defaults: bool = True,
    include_performance_targets: bool = True,
    include_registration_status: bool = False,
) -> Dict[str, object]:
    """Return high-level package metadata for tooling and legacy scripts."""

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

        class EnvironmentDetails(TypedDict, total=False):
            available: List[str]
            factory: NotRequired[str]
            module: NotRequired[str]
            error: NotRequired[str]

        environment_details: EnvironmentDetails = {
            "available": [],
        }

        if PlumeSearchEnv is not None:
            environment_details["available"].append("PlumeSearchEnv")
            environment_details["factory"] = "create_plume_search_env"
            environment_details["module"] = "plume_nav_sim.envs.plume_search_env"
        else:
            environment_details["error"] = str(_ENV_IMPORT_ERROR)

        info["environment"] = environment_details

    if include_registration_status:
        try:
            from plume_nav_sim.registration import get_registration_status

            info["registration"] = get_registration_status()
        except Exception as exc:  # pragma: no cover - defensive guard
            info["registration_error"] = str(exc)

    return info


__all__ = [
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "ENVIRONMENT_ID",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_SOURCE_LOCATION",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_GOAL_RADIUS",
    "DEFAULT_PLUME_SIGMA",
    "PERFORMANCE_TARGET_STEP_LATENCY_MS",
    "PERFORMANCE_TARGET_RGB_RENDER_MS",
    "Action",
    "RenderMode",
    "Coordinates",
    "GridSize",
    "AgentState",
    "EpisodeState",
    "PlumeParameters",
    "EnvironmentConfig",
    "StateSnapshot",
    "PerformanceMetrics",
    "ActionType",
    "CoordinateType",
    "GridDimensions",
    "MovementVector",
    "ObservationType",
    "RewardType",
    "InfoType",
    "calculate_euclidean_distance",
    "create_coordinates",
    "create_grid_size",
    "create_agent_state",
    "create_episode_state",
    "create_environment_config",
    "create_step_info",
    "validate_action",
    "get_movement_vector",
    "PlumeSearchEnv",
    "create_plume_search_env",
    "initialize_package",
    "get_package_info",
]

__version__ = PACKAGE_VERSION
