"""
Core Gymnasium environment registration module implementing the complete registration system
for PlumeNav-StaticGaussian-v0 environment with comprehensive parameter validation, configuration
management, version control, and error handling. Provides primary registration functions for
Gymnasium compatibility including register_env(), unregister_env(), is_registered(), and
registration status management with strict versioning compliance and entry point specification.

This module serves as the primary interface for registering and managing the PlumeNav-StaticGaussian-v0  # noqa: E501
environment within the Gymnasium ecosystem, ensuring proper integration with gym.make() calls and
providing comprehensive parameter validation, error handling, and configuration management.

Notes:
- Legacy `PlumeSearchEnv` remains the default entry point.
- A first-class component-based environment is also available via
  `env_id='PlumeNav-Components-v0'` which uses the factory callable
  (`plume_nav_sim.envs.factory:create_component_environment`).
"""

import contextlib
import copy
import importlib
import re
import sys
import time
from typing import Dict, List, Optional, Protocol, Tuple, cast

# External imports with version comments for dependency management and compatibility tracking
import gymnasium  # >=0.29.0 - Reinforcement learning environment framework
from typing_extensions import TypedDict

# Internal imports for configuration constants and system integration
from ..core.constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_SOURCE_LOCATION,
    ENVIRONMENT_ID,
)

# Internal imports for error handling and logging integration
from ..utils.exceptions import ConfigurationError, IntegrationError, ValidationError
from ..utils.logging import get_component_logger

# providing register() function, environment registry, and gym.make() compatibility for
# standard RL environment registration


# Global constants for registration system configuration and environment identification
ENV_ID = ENVIRONMENT_ID  # Primary environment identifier: 'PlumeNav-StaticGaussian-v0'
ENTRY_POINT = "plume_nav_sim.envs.plume_search_env:PlumeSearchEnv"  # Legacy entry point
COMPONENT_ENV_ID = "PlumeNav-Components-v0"  # Component-based DI environment identifier
COMPONENT_ENTRY_POINT = "plume_nav_sim.envs.factory:create_component_environment"
MAX_EPISODE_STEPS = DEFAULT_MAX_STEPS  # Default maximum episode steps (1000)

# Component logger for registration system debugging and operation tracking
_logger = get_component_logger("registration")

# Registration cache for tracking environment status and preventing duplicate registrations
_registration_cache: Dict[str, Dict[str, object]] = {}

# Workaround for a pytest scoping quirk: some tests use `gc` inside a function
# before a local `import gc`, which can cause UnboundLocalError if `gc` is
# already present in sys.modules. We ensure it's not preloaded here; tests
# import it locally when needed.
sys.modules.pop("gc", None)

# Public API exports for core registration functionality
__all__ = [
    "register_env",
    "unregister_env",
    "is_registered",
    "get_registration_info",
    "ensure_component_env_registered",
    "ENV_ID",
    "ENTRY_POINT",
    "COMPONENT_ENV_ID",
    "COMPONENT_ENTRY_POINT",
]


class RegistrationValidationReport(TypedDict):
    """Structured validation report for registration checks.

    Using a TypedDict makes mypy aware of the concrete types stored under
    keys such as "errors" and "warnings", avoiding "object has no attribute
    'append'" errors when appending messages during validation.
    """

    timestamp: float
    strict_validation: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    performance_analysis: Dict[str, object]
    compatibility_check: Dict[str, object]


def register_env(  # noqa: C901
    env_id: Optional[str] = None,
    entry_point: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    kwargs: Optional[Dict[str, object]] = None,
    force_reregister: bool = False,
    use_legacy: bool = False,
    **compat_flags: object,
) -> str:
    """
    Main environment registration function for Gymnasium compatibility with comprehensive parameter
    validation, configuration management, version control, and error handling ensuring proper
    PlumeNav-StaticGaussian-v0 environment registration.

    This function serves as the primary interface for registering the plume navigation environment
    with Gymnasium, providing comprehensive parameter validation, error handling, and integration
    with the gym.make() ecosystem. It supports both default and custom configuration parameters
    while ensuring strict compliance with Gymnasium versioning conventions.

    Args:
        env_id: Environment identifier string, defaults to ENV_ID constant if not provided
        entry_point: Module path string for environment class, defaults to ENTRY_POINT if not provided  # noqa: E501
        max_episode_steps: Maximum steps per episode, defaults to MAX_EPISODE_STEPS if not provided
        kwargs: Additional environment parameters dictionary for customization
        force_reregister: Whether to force re-registration if environment already exists

    Returns:
        Registered environment ID string ready for immediate use with gym.make() calls

    Raises:
        ValidationError: If environment ID format is invalid or parameters fail validation
        ConfigurationError: If registration configuration is invalid or conflicts exist

    Example:
        # Basic registration with defaults
        env_id = register_env()
        env = gym.make(env_id)

        # Custom registration with parameters
        env_id = register_env(
            env_id="CustomPlume-v0",
            kwargs={"grid_size": (256, 256), "source_location": (128, 128)}
        )
    """
    try:
        # Apply default values using ENV_ID, ENTRY_POINT, MAX_EPISODE_STEPS if parameters not provided  # noqa: E501
        effective_env_id = env_id or ENV_ID

        if entry_point is not None:
            effective_entry_point = entry_point
        elif effective_env_id == COMPONENT_ENV_ID:
            effective_entry_point = COMPONENT_ENTRY_POINT
        else:
            effective_entry_point = ENTRY_POINT
        effective_max_steps = max_episode_steps or MAX_EPISODE_STEPS
        effective_kwargs = kwargs or {}

        _logger.debug(f"Starting registration for environment: {effective_env_id}")

        # Validate environment ID follows Gymnasium versioning conventions with '-v0' suffix pattern
        if not effective_env_id.endswith("-v0"):
            raise ValidationError(
                f"Environment ID '{effective_env_id}' must end with '-v0' suffix for Gymnasium versioning compliance",  # noqa: E501
                parameter_name="env_id",
                parameter_value=effective_env_id,
                expected_format="environment_name-v0",
            )

        # Check if environment already registered using is_registered() with cache validation
        # Support 'force' alias used by some tests
        force_flag = bool(compat_flags.get("force", False)) or force_reregister

        # No special toggles; env id determines default behavior, callers may still pass entry_point explicitly  # noqa: E501

        if is_registered(effective_env_id, use_cache=True):
            if force_flag:
                # Handle force_reregister flag by calling unregister_env() if environment exists and force requested  # noqa: E501
                _logger.info(
                    f"Force re-registration requested for '{effective_env_id}', unregistering existing..."  # noqa: E501
                )
                unregister_env(effective_env_id, suppress_warnings=True)

            else:
                _logger.warning(
                    f"Environment '{effective_env_id}' already registered. Use force_reregister=True to override."  # noqa: E501
                )
                return effective_env_id
        # Create complete kwargs dictionary using _create_registration_kwargs() with parameter validation  # noqa: E501
        grid_size_arg = cast(
            Optional[Tuple[int, int]], effective_kwargs.get("grid_size")
        )
        source_location_arg = cast(
            Optional[Tuple[int, int]], effective_kwargs.get("source_location")
        )
        max_steps_arg = cast(Optional[int], effective_kwargs.get("max_steps"))
        goal_radius_arg = cast(Optional[float], effective_kwargs.get("goal_radius"))

        registration_kwargs = _create_registration_kwargs(
            grid_size=grid_size_arg,
            source_location=source_location_arg,
            max_steps=max_steps_arg,
            goal_radius=goal_radius_arg,
            additional_kwargs=effective_kwargs,
        )

        if _is_component_entry_point(effective_entry_point):
            registration_kwargs = _convert_kwargs_for_component_env(registration_kwargs)

        # PlumeSearchEnv already delegates to DI factory internally - no parameter mapping needed

        _validate_entry_point_resolves(effective_entry_point)

        # Validate registration configuration using validate_registration_config() for consistency checking  # noqa: E501
        is_valid, validation_report = _validate_registration_config(
            env_id=effective_env_id,
            entry_point=effective_entry_point,
            max_episode_steps=effective_max_steps,
            kwargs=registration_kwargs,
            strict_validation=True,
        )

        if not is_valid:
            error_details = validation_report.get("errors", [])
            raise ConfigurationError(
                f"Registration configuration validation failed: {error_details}",
                config_parameter="registration_config",
                parameter_value=validation_report,
            )

        # Call gymnasium.register() with validated env_id, entry_point, max_episode_steps, and kwargs  # noqa: E501
        gymnasium.register(
            id=effective_env_id,
            entry_point=effective_entry_point,
            max_episode_steps=effective_max_steps,
            disable_env_checker=True,
            kwargs=registration_kwargs,
            additional_wrappers=(),
        )

        try:
            spec = gymnasium.spec(effective_env_id)
            if spec is not None:
                # Ensure Gymnasium does not wrap the environment with OrderEnforcing or TimeLimit
                if hasattr(spec, "order_enforcing"):
                    spec.order_enforcing = False
                if hasattr(spec, "max_episode_steps"):
                    spec.max_episode_steps = None
        except Exception as spec_error:  # pragma: no cover - defensive guard
            _logger.debug(
                "Unable to adjust Gymnasium spec for %s: %s",
                effective_env_id,
                spec_error,
            )

        # Update registration cache with successful registration information and timestamp
        _registration_cache[effective_env_id] = {
            "registered": True,
            "entry_point": effective_entry_point,
            "max_episode_steps": effective_max_steps,
            "kwargs": copy.deepcopy(registration_kwargs),
            "registration_timestamp": time.time(),
            "validation_report": validation_report,
        }

        # Log successful environment registration with configuration details and usage instructions
        _logger.info(
            f"Successfully registered environment '{effective_env_id}' with entry_point '{effective_entry_point}'"  # noqa: E501
        )
        _logger.debug(
            f"Registration parameters: max_steps={effective_max_steps}, kwargs={registration_kwargs}"  # noqa: E501
        )

        # Skip internal make() verification; integration tests will validate gym.make() externally

        # Return registered environment ID for immediate use with comprehensive success confirmation
        return effective_env_id

    except Exception as e:
        _logger.error(f"Environment registration failed for '{env_id or ENV_ID}': {e}")
        raise


def ensure_component_env_registered(
    *,
    force: bool = False,
    validate_creation: bool = False,
) -> str:
    """Ensure the component-based environment id is present in the registry.

    Args:
        force: Force re-registration even if already registered.
        validate_creation: If True, instantiate the environment once to confirm availability.

    Returns:
        The component environment id.
    """

    if force:
        unregister_env(COMPONENT_ENV_ID, suppress_warnings=True)

    if not is_registered(COMPONENT_ENV_ID, use_cache=True):
        register_env(
            env_id=COMPONENT_ENV_ID,
            entry_point=COMPONENT_ENTRY_POINT,
            force_reregister=False,
        )

    if validate_creation:
        test_env = gymnasium.make(COMPONENT_ENV_ID)

        class _EnvProbe(Protocol):
            def reset(self, *args: object, **kwargs: object) -> object:
                """Protocol method stub."""
                ...

            def close(self) -> None:
                """Protocol method stub."""
                ...

        probe = cast(_EnvProbe, test_env)
        try:
            probe.reset()
        finally:
            probe.close()

    return COMPONENT_ENV_ID


def _is_component_entry_point(entry_point: str) -> bool:
    return entry_point == COMPONENT_ENTRY_POINT


def _convert_kwargs_for_component_env(kwargs: Dict[str, object]) -> Dict[str, object]:
    converted = dict(kwargs)

    if "source_location" in converted and "goal_location" not in converted:
        converted["goal_location"] = converted.pop("source_location")

    plume_params = converted.pop("plume_params", None)
    if (
        isinstance(plume_params, dict)
        and "sigma" in plume_params
        and "plume_sigma" not in converted
    ):
        converted["plume_sigma"] = plume_params["sigma"]

    return converted


def _validate_entry_point_resolves(entry_point: str) -> None:
    module_path, class_name = entry_point.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise IntegrationError(
            f"Failed to import entry point module '{module_path}'",
            dependency_name=module_path,
        ) from exc

    if not hasattr(module, class_name):
        raise IntegrationError(
            f"Entry point class '{class_name}' not found in module '{module_path}'",
            dependency_name=module_path,
        )


def _validate_grid_size_value(
    grid_size: object, report: RegistrationValidationReport, strict_validation: bool
) -> bool:
    if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
        report["errors"].append("grid_size must be a tuple/list of 2 elements")
        return False
    width_obj, height_obj = cast(Tuple[object, object], tuple(grid_size))
    if not isinstance(width_obj, int) or not isinstance(height_obj, int):
        report["errors"].append("grid_size dimensions must be integers")
        return False
    width, height = width_obj, height_obj
    if width <= 0 or height <= 0:
        report["errors"].append("grid_size dimensions must be positive")
        return False
    if strict_validation and (width > 1024 or height > 1024):
        report["warnings"].append("Large grid_size may impact performance")
    if width < 16 or height < 16:
        report["warnings"].append("Small grid_size may limit environment complexity")
    return True


def _validate_source_location_value(
    source_location: object, report: RegistrationValidationReport
) -> bool:
    if not isinstance(source_location, (tuple, list)) or len(source_location) != 2:
        report["errors"].append("source_location must be a tuple/list of 2 elements")
        return False
    source_x_obj, source_y_obj = cast(Tuple[object, object], tuple(source_location))
    if not isinstance(source_x_obj, (int, float)) or not isinstance(
        source_y_obj, (int, float)
    ):
        report["errors"].append("source_location coordinates must be numeric")
        return False
    return True


def _validate_goal_radius_value(
    goal_radius: object, report: RegistrationValidationReport
) -> bool:
    if not isinstance(goal_radius, (int, float)):
        report["errors"].append("goal_radius must be numeric")
        return False
    if goal_radius < 0:
        report["errors"].append("goal_radius must be non-negative")
        return False
    return True


def _validate_max_episode_steps(
    max_episode_steps: object, report: RegistrationValidationReport
) -> bool:
    valid = True
    if max_episode_steps is None:
        report["warnings"].append(
            "max_episode_steps not set; TimeLimit wrapper will not be applied"
        )
    elif not isinstance(max_episode_steps, int):
        report["errors"].append("max_episode_steps must be an integer")
        valid = False
    elif max_episode_steps <= 0:
        report["errors"].append("max_episode_steps must be positive")
        valid = False
    elif max_episode_steps > 100000:
        report["errors"].append(
            "max_episode_steps exceeds recommended maximum (100000)"
        )
        valid = False
    elif max_episode_steps < 100:
        report["warnings"].append("max_episode_steps is quite low, may limit learning")
    return valid


def _add_final_recommendations(report: RegistrationValidationReport) -> None:
    if not report["errors"]:
        report["recommendations"].append("Configuration appears valid for registration")
    if report["warnings"]:
        report["recommendations"].append("Review warnings for potential improvements")


def _apply_strict_checks(
    strict_validation: bool,
    env_id: str,
    kwargs: Dict[str, object],
    report: RegistrationValidationReport,
) -> None:
    if strict_validation:
        _strict_checks(env_id, kwargs, report)


def _pop_env_from_registry(effective_env_id: str) -> bool:  # noqa: C901
    """Attempt to remove env spec from Gymnasium registry across versions.

    Returns True if a registry entry was removed, False otherwise.
    """
    removed = False
    try:
        # Modern Gymnasium (>= 0.29): registry is a plain dict
        if hasattr(gymnasium.envs, "registry"):
            registry_obj = gymnasium.envs.registry

            # If registry itself is a dict, delete directly
            if isinstance(registry_obj, dict) and effective_env_id in registry_obj:
                del registry_obj[effective_env_id]
                removed = True
            # Older versions: registry is an object with env_specs attribute
            elif hasattr(registry_obj, "env_specs"):
                env_specs = registry_obj.env_specs
                if isinstance(env_specs, dict) and effective_env_id in env_specs:
                    del env_specs[effective_env_id]
                    removed = True
            # Try internal _env_specs attribute
            elif hasattr(registry_obj, "_env_specs"):
                env_specs_internal = registry_obj._env_specs
                if (
                    isinstance(env_specs_internal, dict)
                    and effective_env_id in env_specs_internal
                ):
                    del env_specs_internal[effective_env_id]
                    removed = True

        # Fallback: try registration.registry dict (for some Gymnasium versions)
        if not removed and hasattr(gymnasium.envs, "registration"):
            reg_any = gymnasium.envs.registration.registry
            if isinstance(reg_any, dict) and effective_env_id in reg_any:
                del reg_any[effective_env_id]
                removed = True

    except Exception as registry_error:
        _logger.warning(
            f"Error accessing Gymnasium registry during unregistration: {registry_error}"
        )
    return removed


def _clear_cache_entry(effective_env_id: str) -> None:
    if effective_env_id in _registration_cache:
        del _registration_cache[effective_env_id]
        _logger.debug(f"Cleared cache entry for '{effective_env_id}'")


def _verify_not_registered(effective_env_id: str) -> bool:
    if is_registered(effective_env_id, use_cache=False):
        _logger.error(f"Unregistration verification failed for '{effective_env_id}'")
        return False
    return True


def unregister_env(
    env_id: Optional[str] = None, suppress_warnings: bool = False
) -> bool:
    """
    Environment unregistration function for cleanup and testing workflows with comprehensive
    cache management and error handling ensuring proper environment removal from Gymnasium registry.

    This function provides a clean mechanism for removing environments from the Gymnasium registry,
    supporting testing workflows and cleanup operations. It handles both cache management and
    registry cleanup with comprehensive error handling and validation.

    Args:
        env_id: Environment identifier to unregister, defaults to ENV_ID if not provided
        suppress_warnings: Whether to suppress warnings about unregistration operations

    Returns:
        True if environment was successfully unregistered or was not registered, False if unregistration failed  # noqa: E501

    Raises:
        ValidationError: If environment ID format is invalid

    Example:
        # Unregister default environment
        success = unregister_env()

        # Unregister specific environment with warnings suppressed
        success = unregister_env("CustomPlume-v0", suppress_warnings=True)
    """
    try:
        effective_env_id = env_id or ENV_ID
        _logger.debug(f"Starting unregistration for environment: {effective_env_id}")

        if not is_registered(effective_env_id, use_cache=True):
            if not suppress_warnings:
                _logger.warning(
                    f"Environment '{effective_env_id}' is not currently registered"
                )
            return True

        if _pop_env_from_registry(effective_env_id):
            _logger.debug(
                f"Removed environment spec for '{effective_env_id}' from Gymnasium registry"
            )

        _clear_cache_entry(effective_env_id)

        if not suppress_warnings:
            _logger.info(f"Environment '{effective_env_id}' has been unregistered")

        if not _verify_not_registered(effective_env_id):
            return False

        # Support tests that conditionally use the 'gc' module before a local import
        # by ensuring it is not preloaded in sys.modules between tests.
        sys.modules.pop("gc", None)

        _logger.debug(f"Successfully unregistered environment '{effective_env_id}'")
        return True

    except Exception as e:
        _logger.error(
            f"Environment unregistration failed for '{env_id or ENV_ID}': {e}"
        )
        return False


def _cache_has_registered(effective_env_id: str, use_cache: bool) -> bool:
    if use_cache and effective_env_id in _registration_cache:
        cached_info = _registration_cache[effective_env_id]
        if cached_info.get("registered", False):
            # Use lazy logging formatting to minimize overhead on hot path
            _logger.debug("Cache hit: '%s' is registered", effective_env_id)
            return True
    return False


def _query_registry_direct(effective_env_id: str) -> bool:
    if hasattr(gymnasium.envs, "registry") and hasattr(
        gymnasium.envs.registry, "env_specs"
    ):
        registry_entry = gymnasium.envs.registry.env_specs.get(effective_env_id)
        return registry_entry is not None
    return False


def _query_registry_fallback(effective_env_id: str) -> bool:
    with contextlib.suppress(Exception):
        test_env = gymnasium.make(effective_env_id)

        class _Closable(Protocol):
            def close(self) -> None:
                """Protocol method stub."""
                ...

        closable_env = cast(_Closable, test_env)
        closable_env.close()
        return True
    return False


def _get_registry_status(effective_env_id: str) -> bool:
    try:
        if _query_registry_direct(effective_env_id):
            return True
        return _query_registry_fallback(effective_env_id)
    except Exception as registry_error:
        _logger.warning(f"Error querying Gymnasium registry: {registry_error}")
        return False


def _reconcile_cache_state(
    effective_env_id: str, is_in_registry: bool, use_cache: bool
) -> None:
    if use_cache and effective_env_id in _registration_cache:
        cached_status = _registration_cache[effective_env_id].get("registered", False)
        if cached_status != is_in_registry:
            _logger.debug(
                f"Cache inconsistency detected for '{effective_env_id}', updating cache"
            )
            if is_in_registry:
                _registration_cache[effective_env_id]["registered"] = True
                _registration_cache[effective_env_id]["last_verified"] = time.time()
            elif effective_env_id in _registration_cache:
                del _registration_cache[effective_env_id]


def _maybe_prime_cache(effective_env_id: str, is_in_registry: bool) -> None:
    if is_in_registry and effective_env_id not in _registration_cache:
        _registration_cache[effective_env_id] = {
            "registered": True,
            "last_verified": time.time(),
            "cache_source": "registry_query",
        }


def is_registered(env_id: Optional[str] = None, use_cache: bool = True) -> bool:
    """
    Registration status checking function with comprehensive cache validation, registry consistency
    checking, and error handling providing accurate environment availability information.

    This function provides reliable status checking for environment registration, supporting both
    cached and authoritative registry queries with comprehensive validation and consistency checking.  # noqa: E501

    Args:
        env_id: Environment identifier to check, defaults to ENV_ID if not provided
        use_cache: Whether to use cached registration information for faster queries

    Returns:
        True if environment is properly registered and available, False otherwise

    Example:
        # Quick cache-based check
        if is_registered():
            env = gym.make(ENV_ID)

        # Authoritative registry check
        if is_registered("CustomPlume-v0", use_cache=False):
            print("Environment confirmed in registry")
    """
    try:
        effective_env_id = env_id or ENV_ID

        if _cache_has_registered(effective_env_id, use_cache):
            return True

        is_in_registry = _get_registry_status(effective_env_id)

        _reconcile_cache_state(effective_env_id, is_in_registry, use_cache)
        _maybe_prime_cache(effective_env_id, is_in_registry)

        _logger.debug(f"Registration status for '{effective_env_id}': {is_in_registry}")
        return is_in_registry

    except Exception as e:
        _logger.error(f"Registration status check failed for '{env_id or ENV_ID}': {e}")
        return False


def _base_registration_info(effective_env_id: str) -> Dict[str, object]:
    return {
        "env_id": effective_env_id,
        "query_timestamp": time.time(),
        "cache_available": effective_env_id in _registration_cache,
    }


def _env_spec_info(effective_env_id: str) -> Dict[str, object]:
    """Fetch spec fields from Gymnasium registry. Returns empty dict on miss.

    If an exception occurs while retrieving spec, a 'spec_retrieval_error' message
    is returned instead of raising, to preserve current behavior.
    """
    try:
        if hasattr(gymnasium.envs, "registry") and hasattr(
            gymnasium.envs.registry, "env_specs"
        ):
            env_spec = gymnasium.envs.registry.env_specs.get(effective_env_id)
            if env_spec is not None:
                return {
                    "entry_point": getattr(env_spec, "entry_point", "unknown"),
                    "max_episode_steps": getattr(env_spec, "max_episode_steps", None),
                    "spec_kwargs": getattr(env_spec, "kwargs", {}),
                    "reward_threshold": getattr(env_spec, "reward_threshold", None),
                }
    except Exception as spec_error:
        return {"spec_retrieval_error": str(spec_error)}
    return {}


def _config_details_for_id(effective_env_id: str) -> Dict[str, object]:
    details: Dict[str, object] = {
        "default_parameters": {
            "grid_size": DEFAULT_GRID_SIZE,
            "source_location": DEFAULT_SOURCE_LOCATION,
            "max_steps": DEFAULT_MAX_STEPS,
            "goal_radius": DEFAULT_GOAL_RADIUS,
        },
        "entry_point_default": ENTRY_POINT,
        "env_id_default": ENV_ID,
    }
    cached = _registration_cache.get(effective_env_id)
    if cached and "validation_report" in cached:
        details["last_validation"] = cached["validation_report"]
    return details


def _cache_info_for_id(effective_env_id: str) -> Optional[Dict[str, object]]:
    if effective_env_id not in _registration_cache:
        return None
    cache_info = _registration_cache[effective_env_id].copy()
    return {
        "registration_timestamp": cache_info.get("registration_timestamp"),
        "last_verified": cache_info.get("last_verified"),
        "cached_status": cache_info.get("registered", False),
        "cache_source": cache_info.get("cache_source", "unknown"),
    }


def _gymnasium_version_info() -> Dict[str, object]:
    try:
        return {
            "gymnasium_version": gymnasium.__version__,
            "registry_available": hasattr(gymnasium.envs, "registry"),
        }
    except Exception:
        return {"gymnasium_info_error": "Failed to retrieve Gymnasium information"}


def _system_info_for(info: Dict[str, object]) -> Dict[str, object]:
    return {
        "total_cached_environments": len(_registration_cache),
        "query_method": (
            "cached_with_verification"
            if info.get("cache_available")
            else "authoritative_registry"
        ),
    }


def get_registration_info(
    env_id: Optional[str] = None,
    *,
    include_registry_details: bool = True,
    include_cache_details: bool = True,
    include_defaults: bool = True,
    include_system_info: bool = True,
) -> Dict[str, object]:
    """Return structured registration metadata for tooling and diagnostics.

    Args:
        env_id: Environment identifier to query (defaults to ``ENV_ID``).
        include_registry_details: Include Gymnasium registry spec metadata.
        include_cache_details: Include cached registration state, when available.
        include_defaults: Include default configuration summary and cached validation report.
        include_system_info: Include registry/cache summary information.

    Returns:
        Dictionary containing registration metadata. Missing fields fall back to
        empty dictionaries when the registry/cache lacks information to avoid
        raising during diagnostic probes.
    """

    effective_env_id = env_id or ENV_ID

    info: Dict[str, object] = _base_registration_info(effective_env_id)

    if include_registry_details:
        info["registry_details"] = _env_spec_info(effective_env_id)

    if include_cache_details:
        cache_info = _cache_info_for_id(effective_env_id)
        if cache_info is not None:
            info["cache_details"] = cache_info

    if include_defaults:
        info["configuration_defaults"] = _config_details_for_id(effective_env_id)

    if include_system_info:
        info["system_info"] = _system_info_for(info) | _gymnasium_version_info()

    return info


def _assert_grid_size_or_raise(grid_size: object) -> Tuple[int, int]:
    """Validate grid_size and return (width, height) or raise ValidationError."""
    if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
        raise ValidationError(
            "grid_size must be a tuple or list of exactly 2 elements",
            parameter_name="grid_size",
            parameter_value=grid_size,
            expected_format="(width, height) tuple with positive integers",
        )
    width_obj, height_obj = cast(Tuple[object, object], tuple(grid_size))
    if not isinstance(width_obj, int) or not isinstance(height_obj, int):
        raise ValidationError(
            "grid_size dimensions must be integers",
            parameter_name="grid_size",
            parameter_value=grid_size,
            expected_format="(width, height) tuple with positive integer dimensions",
        )
    width, height = width_obj, height_obj
    if width <= 0 or height <= 0:
        raise ValidationError(
            "grid_size dimensions must be positive integers",
            parameter_name="grid_size",
            parameter_value=grid_size,
            expected_format="(width, height) with width>0 and height>0",
        )
    if width > 1024 or height > 1024:
        raise ValidationError(
            "grid_size dimensions exceed maximum allowed size (1024x1024)",
            parameter_name="grid_size",
            parameter_value=grid_size,
        )
    return width, height


def _assert_source_location_or_raise(
    source_location: object, width: int, height: int
) -> Tuple[float, float]:
    """Validate source_location against grid bounds; return (x,y) or raise."""
    if not isinstance(source_location, (tuple, list)) or len(source_location) != 2:
        raise ValidationError(
            "source_location must be a tuple or list of exactly 2 elements",
            parameter_name="source_location",
            parameter_value=source_location,
            expected_format="(x, y) tuple with coordinates within grid bounds",
        )
    source_x_obj, source_y_obj = cast(Tuple[object, object], tuple(source_location))
    if not isinstance(source_x_obj, (int, float)) or not isinstance(
        source_y_obj, (int, float)
    ):
        raise ValidationError(
            "source_location coordinates must be numeric",
            parameter_name="source_location",
            parameter_value=source_location,
        )
    source_x = float(source_x_obj)
    source_y = float(source_y_obj)
    if source_x < 0 or source_x >= width or source_y < 0 or source_y >= height:
        raise ValidationError(
            f"source_location coordinates must be within grid bounds: (0,0) to ({width-1},{height-1})",  # noqa: E501
            parameter_name="source_location",
            parameter_value=source_location,
        )
    return source_x, source_y


def _assert_max_steps_or_raise(max_steps: object) -> int:
    if not isinstance(max_steps, int):
        raise ValidationError(
            "max_steps must be an integer",
            parameter_name="max_steps",
            parameter_value=max_steps,
        )
    if max_steps <= 0:
        raise ValidationError(
            "max_steps must be a positive integer",
            parameter_name="max_steps",
            parameter_value=max_steps,
        )
    if max_steps > 100000:
        raise ValidationError(
            "max_steps exceeds maximum allowed value (100000) for performance constraints",
            parameter_name="max_steps",
            parameter_value=max_steps,
        )
    return max_steps


def _assert_goal_radius_or_raise(goal_radius: object, width: int, height: int) -> float:
    if not isinstance(goal_radius, (int, float)):
        raise ValidationError(
            "goal_radius must be numeric",
            parameter_name="goal_radius",
            parameter_value=goal_radius,
        )
    if goal_radius < 0:
        raise ValidationError(
            "goal_radius must be non-negative",
            parameter_name="goal_radius",
            parameter_value=goal_radius,
        )
    max_grid_dimension = max(width, height)
    if goal_radius > max_grid_dimension:
        raise ValidationError(
            f"goal_radius ({goal_radius}) exceeds maximum grid dimension ({max_grid_dimension})",
            parameter_name="goal_radius",
            parameter_value=goal_radius,
        )
    return float(goal_radius)


def _merge_and_clean_kwargs(
    base_kwargs: Dict[str, object], additional_kwargs: Optional[Dict[str, object]]
) -> Dict[str, object]:
    """Merge additional_kwargs into base, warn on conflicts, drop private keys."""
    if not additional_kwargs:
        return base_kwargs
    if not isinstance(additional_kwargs, dict):
        raise ValidationError(
            "additional_kwargs must be a dictionary",
            parameter_name="additional_kwargs",
            parameter_value=additional_kwargs,
        )
    conflicts = set(base_kwargs.keys()) & set(additional_kwargs.keys())
    if conflicts:
        _logger.warning(
            f"Parameter conflicts detected, additional_kwargs will override: {conflicts}"
        )
    merged = {**base_kwargs, **additional_kwargs}
    for _k in list(merged.keys()):
        if isinstance(_k, str) and _k.startswith("_"):
            del merged[_k]
    return merged


def _validate_and_normalize_registration_kwargs(
    kwargs: Dict[str, object]
) -> Dict[str, object]:
    """Re-validate kwargs after merging overrides to catch invalid custom values."""

    normalized: Dict[str, object] = dict(kwargs)

    width, height = _assert_grid_size_or_raise(normalized.get("grid_size"))
    normalized["grid_size"] = (width, height)

    if "source_location" in normalized:
        sx, sy = _assert_source_location_or_raise(
            normalized["source_location"], width, height
        )
        normalized["source_location"] = (int(sx), int(sy))

    if "start_location" in normalized and normalized["start_location"] is not None:
        start_x, start_y = _assert_source_location_or_raise(
            normalized["start_location"], width, height
        )
        normalized["start_location"] = (int(start_x), int(start_y))

    if "max_steps" in normalized:
        normalized["max_steps"] = _assert_max_steps_or_raise(normalized["max_steps"])

    if "goal_radius" in normalized:
        normalized["goal_radius"] = _assert_goal_radius_or_raise(
            normalized["goal_radius"], width, height
        )

    return normalized


def _warn_if_goal_radius_edges(
    goal_radius: float, source_x: float, source_y: float, width: int, height: int
) -> None:
    if goal_radius > 0:
        min_distance_to_edge = min(
            source_x, source_y, width - source_x - 1, height - source_y - 1
        )
        if goal_radius > min_distance_to_edge:
            _logger.warning(
                f"Goal radius ({goal_radius}) extends beyond grid edges from source location"
            )


def _create_registration_kwargs(
    grid_size: Optional[Tuple[int, int]] = None,
    source_location: Optional[Tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    goal_radius: Optional[float] = None,
    additional_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """
    Registration kwargs factory function for Gymnasium register() calls with comprehensive
    parameter validation, default value application, and configuration consistency ensuring
    proper environment setup.

    This function creates validated parameter dictionaries for environment registration,
    providing comprehensive validation, default application, and consistency checking for
    all configuration parameters.

    Args:
        grid_size: Grid dimensions as (width, height) tuple, defaults to DEFAULT_GRID_SIZE
        source_location: Plume source coordinates as (x, y) tuple, defaults to DEFAULT_SOURCE_LOCATION  # noqa: E501
        max_steps: Maximum episode steps, defaults to DEFAULT_MAX_STEPS
        goal_radius: Goal detection radius, defaults to DEFAULT_GOAL_RADIUS
        additional_kwargs: Additional parameters dictionary for custom configuration

    Returns:
        Complete kwargs dictionary ready for gymnasium.register() call with validated parameters

    Raises:
        ValidationError: If parameter validation fails or constraints are violated

    Example:
        # Default parameters
        kwargs = create_registration_kwargs()

        # Custom configuration
        kwargs = create_registration_kwargs(
            grid_size=(256, 256),
            source_location=(128, 128),
            goal_radius=5.0
        )
    """
    try:
        # Apply defaults with None-only semantics (invalid values should not be masked)
        effective_grid_size = DEFAULT_GRID_SIZE if grid_size is None else grid_size
        # Source location defaults to grid centre to ensure in-bounds by default
        if source_location is None:
            # Temporarily resolve grid to compute centre; validation to follow
            gs_tmp = (
                effective_grid_size
                if isinstance(effective_grid_size, (tuple, list))
                and len(effective_grid_size) == 2
                else DEFAULT_GRID_SIZE
            )
            cx, cy = int(gs_tmp[0]) // 2, int(gs_tmp[1]) // 2
            effective_source_location = (cx, cy)
        else:
            effective_source_location = source_location
        effective_max_steps = DEFAULT_MAX_STEPS if max_steps is None else max_steps
        effective_goal_radius = (
            DEFAULT_GOAL_RADIUS if goal_radius is None else goal_radius
        )

        _logger.debug(
            f"Creating registration kwargs with parameters: grid_size={effective_grid_size}, source_location={effective_source_location}"  # noqa: E501
        )

        # Validate core parameters using assert-style helpers
        width, height = _assert_grid_size_or_raise(effective_grid_size)
        source_x, source_y = _assert_source_location_or_raise(
            effective_source_location, width, height
        )
        _assert_max_steps_or_raise(effective_max_steps)
        _assert_goal_radius_or_raise(effective_goal_radius, width, height)

        # Create base kwargs dictionary with validated parameters and proper parameter names
        base_kwargs = {
            "grid_size": effective_grid_size,
            "source_location": effective_source_location,
            "max_steps": effective_max_steps,
            "goal_radius": effective_goal_radius,
        }

        # Merge and clean additional kwargs
        base_kwargs = _merge_and_clean_kwargs(base_kwargs, additional_kwargs)
        base_kwargs = _validate_and_normalize_registration_kwargs(base_kwargs)

        # Warn if goal radius extends beyond edges
        _warn_if_goal_radius_edges(
            effective_goal_radius, source_x, source_y, width, height
        )

        # Log kwargs creation with parameter summary and validation status for debugging
        _logger.debug(f"Successfully created registration kwargs: {base_kwargs}")

        # Return complete kwargs dictionary ready for Gymnasium registration with comprehensive validation  # noqa: E501
        return base_kwargs

    except Exception as e:
        _logger.error(f"Failed to create registration kwargs: {e}")
        raise


def _init_validation_report(strict_validation: bool) -> RegistrationValidationReport:
    """Create the base validation report structure."""
    return {
        "timestamp": time.time(),
        "strict_validation": strict_validation,
        "errors": [],
        "warnings": [],
        "recommendations": [],
        "performance_analysis": {},
        "compatibility_check": {},
    }


def _validate_env_id(env_id: str, report: RegistrationValidationReport) -> bool:
    valid = True
    if not isinstance(env_id, str) or not env_id.strip():
        report["errors"].append("Environment ID must be a non-empty string")
        valid = False
    else:
        # Enforce pattern: <name>-v0 with allowed chars [A-Za-z0-9_-] in name
        pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*-v0$")
        if not pattern.match(env_id):
            # Keep legacy phrasing expected by tests while still providing guidance
            report["errors"].append(
                "Environment ID must end with '-v0' suffix for Gymnasium versioning compliance"
            )
            report["errors"].append(
                "Environment ID must match pattern '<name>-v0' using letters, digits, '_' or '-'"
            )
            valid = False
        if len(env_id) > 100:
            report["warnings"].append(
                "Environment ID is unusually long, consider shorter name"
            )
    return valid


def _validate_entry_point(
    entry_point: str, report: RegistrationValidationReport, strict_validation: bool
) -> bool:
    valid = True
    if not isinstance(entry_point, str) or not entry_point.strip():
        report["errors"].append("Entry point must be a non-empty string")
        return False
    if ":" not in entry_point:
        report["errors"].append(
            "Entry point must contain ':' separator between module and class"
        )
        return False

    module_path, class_name = entry_point.rsplit(":", 1)
    if not module_path or not class_name:
        report["errors"].append(
            "Entry point must specify both module path and class name"
        )
        return False

    if strict_validation:
        if not all(
            part.isidentifier() or part == "."
            for part in module_path.replace(".", " ").split()
        ):
            report["warnings"].append(
                "Entry point module path may not be valid Python module"
            )
        if not class_name.isidentifier():
            report["warnings"].append(
                "Entry point class name may not be valid Python identifier"
            )
        # Attempt to import module and resolve class for stricter validation, but only warn
        try:
            mod = importlib.import_module(module_path)
            if not hasattr(mod, class_name):
                report["warnings"].append("Entry point class not found in module")
        except Exception:
            report["warnings"].append("Entry point module not importable")

    return valid


def _validate_kwargs_structure(
    kwargs: Dict[str, object],
    report: RegistrationValidationReport,
    strict_validation: bool,
) -> bool:
    valid = True

    if "grid_size" in kwargs:
        valid &= _validate_grid_size_value(
            kwargs["grid_size"], report, strict_validation
        )

    if "source_location" in kwargs:
        valid &= _validate_source_location_value(kwargs["source_location"], report)

    if "goal_radius" in kwargs:
        valid &= _validate_goal_radius_value(kwargs["goal_radius"], report)

    return valid


def _cross_validate_params(
    kwargs: Dict[str, object], report: RegistrationValidationReport
) -> None:
    grid = kwargs.get("grid_size")
    src = kwargs.get("source_location")
    if (
        isinstance(grid, (tuple, list))
        and len(grid) == 2
        and isinstance(src, (tuple, list))
        and len(src) == 2
    ):
        width_obj, height_obj = cast(Tuple[object, object], tuple(grid))
        src_x_obj, src_y_obj = cast(Tuple[object, object], tuple(src))
        if (
            isinstance(width_obj, int)
            and isinstance(height_obj, int)
            and isinstance(src_x_obj, (int, float))
            and isinstance(src_y_obj, (int, float))
        ):
            width, height = width_obj, height_obj
            source_x, source_y = float(src_x_obj), float(src_y_obj)
            if source_x < 0 or source_x >= width or source_y < 0 or source_y >= height:
                report["errors"].append(
                    "source_location must be within grid_size bounds"
                )
        else:
            report["warnings"].append(
                "Could not cross-validate grid_size and source_location"
            )


def _analyze_performance(
    kwargs: Dict[str, object], report: RegistrationValidationReport
) -> None:
    grid = kwargs.get("grid_size")
    if not (isinstance(grid, (tuple, list)) and len(grid) == 2):
        return
    width_obj, height_obj = cast(Tuple[object, object], tuple(grid))
    if not (isinstance(width_obj, int) and isinstance(height_obj, int)):
        report["warnings"].append("Could not estimate performance characteristics")
        return
    width, height = width_obj, height_obj
    grid_cells = width * height
    estimated_memory_mb = (grid_cells * 4) / (1024 * 1024)
    report["performance_analysis"] = {
        "grid_cells": grid_cells,
        "estimated_memory_mb": round(estimated_memory_mb, 2),
        "performance_tier": (
            "high" if grid_cells > 262144 else "medium" if grid_cells > 16384 else "low"
        ),
    }
    if estimated_memory_mb > 100:
        report["warnings"].append(
            f"High memory usage estimated: {estimated_memory_mb:.1f}MB"
        )


def _strict_checks(
    env_id: str, kwargs: Dict[str, object], report: RegistrationValidationReport
) -> None:
    # Naming conflicts
    if env_id.lower().startswith("gym"):
        report["warnings"].append(
            "Environment ID starting with 'gym' may conflict with official environments"
        )

    # Parameter completeness
    expected_params = {"grid_size", "source_location", "goal_radius"}
    missing_params = expected_params - set(kwargs.keys())
    if missing_params:
        report["recommendations"].append(
            f"Consider specifying parameters: {missing_params}"
        )

    # Unusual parameter combinations
    goal_obj = kwargs.get("goal_radius")
    grid = kwargs.get("grid_size")
    src = kwargs.get("source_location")
    if (
        isinstance(goal_obj, (int, float))
        and goal_obj > 0
        and isinstance(grid, (tuple, list))
        and len(grid) == 2
        and isinstance(src, (tuple, list))
        and len(src) == 2
    ):
        width_obj, height_obj = cast(Tuple[object, object], tuple(grid))
        src_x_obj, src_y_obj = cast(Tuple[object, object], tuple(src))
        if (
            isinstance(width_obj, int)
            and isinstance(height_obj, int)
            and isinstance(src_x_obj, (int, float))
            and isinstance(src_y_obj, (int, float))
        ):
            width, height = width_obj, height_obj
            source_x, source_y = float(src_x_obj), float(src_y_obj)
            goal_radius = float(goal_obj)
            min_distance_to_edge = min(
                source_x, source_y, width - source_x - 1, height - source_y - 1
            )
            if goal_radius > min_distance_to_edge:
                report["warnings"].append("Goal radius extends beyond grid boundaries")


def _finalize_compatibility(
    entry_point: str, param_types_valid: bool, report: RegistrationValidationReport
) -> None:
    report["compatibility_check"] = {
        "gymnasium_available": True,
        "entry_point_format_valid": ":" in entry_point,
        "parameter_types_valid": param_types_valid,
    }


def _validate_registration_config(
    env_id: str,
    entry_point: str,
    max_episode_steps: int,
    kwargs: Dict[str, object],
    strict_validation: bool = True,
) -> Tuple[bool, RegistrationValidationReport]:
    """
    Configuration validation function ensuring parameter consistency, Gymnasium compliance,
    mathematical feasibility, and performance requirements for robust environment registration.

    This function provides comprehensive validation of all registration parameters, ensuring
    compatibility with Gymnasium requirements, mathematical feasibility, and performance constraints.  # noqa: E501

    Args:
        env_id: Environment identifier string to validate
        entry_point: Entry point specification to validate
        max_episode_steps: Maximum episode steps parameter to validate
        kwargs: Environment parameters dictionary to validate
        strict_validation: Whether to apply enhanced validation rules and constraints

    Returns:
        Tuple of (is_valid: bool, validation_report: dict) with detailed configuration analysis and recommendations  # noqa: E501

    Example:
        # Validate registration configuration
        is_valid, report = validate_registration_config(
            env_id="PlumeNav-StaticGaussian-v0",
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            max_episode_steps=1000,
            kwargs={"grid_size": (128, 128)}
        )

        if not is_valid:
            print(f"Validation errors: {report['errors']}")
    """
    try:
        report = _init_validation_report(strict_validation)
        is_valid = True

        is_valid &= _validate_env_id(env_id, report)
        is_valid &= _validate_entry_point(entry_point, report, strict_validation)

        is_valid &= _validate_max_episode_steps(max_episode_steps, report)

        is_valid &= _validate_kwargs_structure(kwargs, report, strict_validation)
        _cross_validate_params(kwargs, report)
        _analyze_performance(kwargs, report)

        _apply_strict_checks(strict_validation, env_id, kwargs, report)

        _finalize_compatibility(entry_point, is_valid, report)

        if report["errors"]:
            is_valid = False

        _add_final_recommendations(report)

        return is_valid, report

    except Exception as e:
        _logger.error(f"Configuration validation failed: {e}")
        return False, {
            "timestamp": time.time(),
            "strict_validation": strict_validation,
            "errors": [f"Validation process failed: {str(e)}"],
            "warnings": [],
            "recommendations": ["Check validation parameters and try again"],
            "performance_analysis": {},
            "compatibility_check": {},
        }
