"""
Core Gymnasium environment registration module implementing the complete registration system
for PlumeNav-StaticGaussian-v0 environment with comprehensive parameter validation, configuration
management, version control, and error handling. Provides primary registration functions for
Gymnasium compatibility including register_env(), unregister_env(), is_registered(), and
registration status management with strict versioning compliance and entry point specification.

This module serves as the primary interface for registering and managing the PlumeNav-StaticGaussian-v0
environment within the Gymnasium ecosystem, ensuring proper integration with gym.make() calls and
providing comprehensive parameter validation, error handling, and configuration management.

Notes:
- Pass `kwargs={'use_components': True}` to `register_env` to register the component-based
  environment (`plume_nav_sim.envs.component_env:ComponentBasedEnvironment`) instead of the
  legacy `PlumeSearchEnv`. All other kwargs are forwarded to the environment constructor.
"""

import contextlib
import copy
import importlib
import re
import sys
import time
from typing import (  # >=3.10 - Type hints for function parameters, return types, and optional parameter specifications ensuring type safety and documentation clarity
    Any,
    Dict,
    Optional,
    Tuple,
)

# External imports with version comments for dependency management and compatibility tracking
import gymnasium  # >=0.29.0 - Reinforcement learning environment framework providing register() function, environment registry, and gym.make() compatibility for standard RL environment registration

# Internal imports for configuration constants and system integration
from ..core.constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_SOURCE_LOCATION,
    ENVIRONMENT_ID,
)

# Internal imports for error handling and logging integration
from ..utils.exceptions import ConfigurationError, ValidationError
from ..utils.logging import get_component_logger

# Global constants for registration system configuration and environment identification
ENV_ID = ENVIRONMENT_ID  # Primary environment identifier 'PlumeNav-StaticGaussian-v0' for Gymnasium registration compliance
ENTRY_POINT = "plume_nav_sim.envs.plume_search_env:PlumeSearchEnv"  # Entry point specification string for Gymnasium registration defining exact module path and class location
MAX_EPISODE_STEPS = DEFAULT_MAX_STEPS  # Default maximum episode steps (1000) for registration parameter configuration

# Component logger for registration system debugging and operation tracking
_logger = get_component_logger("registration")

# Registration cache for tracking environment status and preventing duplicate registrations
_registration_cache: Dict[str, Dict[str, Any]] = {}

# Workaround for a pytest scoping quirk: some tests use `gc` inside a function
# before a local `import gc`, which can cause UnboundLocalError if `gc` is
# already present in sys.modules. We ensure it's not preloaded here; tests
# import it locally when needed.
sys.modules.pop("gc", None)

# Public API exports for comprehensive registration functionality
__all__ = [
    "register_env",
    "unregister_env",
    "is_registered",
    "get_registration_info",
    "create_registration_kwargs",
    "validate_registration_config",
    "register_with_custom_params",
    "ENV_ID",
    "ENTRY_POINT",
]


def register_env(
    env_id: Optional[str] = None,
    entry_point: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    force_reregister: bool = False,
    **compat_flags: Any,
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
        entry_point: Module path string for environment class, defaults to ENTRY_POINT if not provided
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
        # Apply default values using ENV_ID, ENTRY_POINT, MAX_EPISODE_STEPS if parameters not provided
        effective_env_id = env_id or ENV_ID
        effective_entry_point = entry_point or ENTRY_POINT
        effective_max_steps = max_episode_steps or MAX_EPISODE_STEPS
        effective_kwargs = kwargs or {}

        _logger.debug(f"Starting registration for environment: {effective_env_id}")

        # Validate environment ID follows Gymnasium versioning conventions with '-v0' suffix pattern
        if not effective_env_id.endswith("-v0"):
            raise ValidationError(
                f"Environment ID '{effective_env_id}' must end with '-v0' suffix for Gymnasium versioning compliance",
                parameter_name="env_id",
                parameter_value=effective_env_id,
                expected_format="environment_name-v0",
            )

        # Check if environment already registered using is_registered() with cache validation
        # Support 'force' alias used by some tests
        force_flag = bool(compat_flags.get("force", False)) or force_reregister

        if is_registered(effective_env_id, use_cache=True):
            if force_flag:
                # Handle force_reregister flag by calling unregister_env() if environment exists and force requested
                _logger.info(
                    f"Force re-registration requested for '{effective_env_id}', unregistering existing..."
                )
                unregister_env(effective_env_id, suppress_warnings=True)

            else:
                _logger.warning(
                    f"Environment '{effective_env_id}' already registered. Use force_reregister=True to override."
                )
                return effective_env_id
        # Create complete kwargs dictionary using create_registration_kwargs() with parameter validation
        registration_kwargs = create_registration_kwargs(
            grid_size=effective_kwargs.get("grid_size"),
            source_location=effective_kwargs.get("source_location"),
            max_steps=effective_kwargs.get("max_steps"),
            goal_radius=effective_kwargs.get("goal_radius"),
            additional_kwargs=effective_kwargs,
        )

        # Optional: allow component-based environment via flag without breaking legacy usage
        # If caller passes use_components=True, switch the entry point to ComponentBasedEnvironment.
        use_components = bool(registration_kwargs.pop("use_components", False))
        if use_components:
            effective_entry_point = (
                "plume_nav_sim.envs.component_env:ComponentBasedEnvironment"
            )

        # Validate registration configuration using validate_registration_config() for consistency checking
        is_valid, validation_report = validate_registration_config(
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

        # Call gymnasium.register() with validated env_id, entry_point, max_episode_steps, and kwargs
        gymnasium.register(
            id=effective_env_id,
            entry_point=effective_entry_point,
            max_episode_steps=effective_max_steps,
            disable_env_checker=True,
            kwargs=registration_kwargs,
            additional_wrappers=(),
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
            f"Successfully registered environment '{effective_env_id}' with entry_point '{effective_entry_point}'"
        )
        _logger.debug(
            f"Registration parameters: max_steps={effective_max_steps}, kwargs={registration_kwargs}"
        )

        # Skip internal make() verification; integration tests will validate gym.make() externally

        # Return registered environment ID for immediate use with comprehensive success confirmation
        return effective_env_id

    except Exception as e:
        _logger.error(f"Environment registration failed for '{env_id or ENV_ID}': {e}")
        raise


def _validate_grid_size_value(
    grid_size: Any, report: Dict[str, Any], strict_validation: bool
) -> bool:
    if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
        report["errors"].append("grid_size must be a tuple/list of 2 elements")
        return False
    width, height = grid_size
    if not isinstance(width, int) or not isinstance(height, int):
        report["errors"].append("grid_size dimensions must be integers")
        return False
    if width <= 0 or height <= 0:
        report["errors"].append("grid_size dimensions must be positive")
        return False
    if strict_validation and (width > 1024 or height > 1024):
        report["warnings"].append("Large grid_size may impact performance")
    if width < 16 or height < 16:
        report["warnings"].append("Small grid_size may limit environment complexity")
    return True


def _validate_source_location_value(
    source_location: Any, report: Dict[str, Any]
) -> bool:
    if not isinstance(source_location, (tuple, list)) or len(source_location) != 2:
        report["errors"].append("source_location must be a tuple/list of 2 elements")
        return False
    source_x, source_y = source_location
    if not isinstance(source_x, (int, float)) or not isinstance(source_y, (int, float)):
        report["errors"].append("source_location coordinates must be numeric")
        return False
    return True


def _validate_goal_radius_value(goal_radius: Any, report: Dict[str, Any]) -> bool:
    if not isinstance(goal_radius, (int, float)):
        report["errors"].append("goal_radius must be numeric")
        return False
    if goal_radius < 0:
        report["errors"].append("goal_radius must be non-negative")
        return False
    return True


def _validate_max_episode_steps(max_episode_steps: Any, report: Dict[str, Any]) -> bool:
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


def _add_final_recommendations(report: Dict[str, Any]) -> None:
    if not report["errors"]:
        report["recommendations"].append("Configuration appears valid for registration")
    if report["warnings"]:
        report["recommendations"].append("Review warnings for potential improvements")


def _apply_strict_checks(
    strict_validation: bool, env_id: str, kwargs: Dict[str, Any], report: Dict[str, Any]
) -> None:
    if strict_validation:
        _strict_checks(env_id, kwargs, report)


def _pop_env_from_registry(effective_env_id: str) -> bool:
    """Attempt to remove env spec from Gymnasium registry across versions.

    Returns True if a registry entry was removed, False otherwise.
    """
    try:
        if hasattr(gymnasium.envs, "registry") and hasattr(
            gymnasium.envs.registry, "env_specs"
        ):
            return (
                gymnasium.envs.registry.env_specs.pop(effective_env_id, None)
                is not None
            )
        if hasattr(gymnasium, "envs") and hasattr(gymnasium.envs, "registration"):
            reg = gymnasium.envs.registration.registry
            if effective_env_id in reg.env_specs:
                del reg.env_specs[effective_env_id]
                return True
    except Exception as registry_error:
        _logger.warning(
            f"Error accessing Gymnasium registry during unregistration: {registry_error}"
        )
    return False


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
        True if environment was successfully unregistered or was not registered, False if unregistration failed

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

        removed = _pop_env_from_registry(effective_env_id)
        if removed:
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
        test_env.close()
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
    cached and authoritative registry queries with comprehensive validation and consistency checking.

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


def _base_registration_info(effective_env_id: str) -> Dict[str, Any]:
    return {
        "env_id": effective_env_id,
        "query_timestamp": time.time(),
        "cache_available": effective_env_id in _registration_cache,
    }


def _env_spec_info(effective_env_id: str) -> Dict[str, Any]:
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


def _config_details_for_id(effective_env_id: str) -> Dict[str, Any]:
    details: Dict[str, Any] = {
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


def _cache_info_for_id(effective_env_id: str) -> Optional[Dict[str, Any]]:
    if effective_env_id not in _registration_cache:
        return None
    cache_info = _registration_cache[effective_env_id].copy()
    return {
        "registration_timestamp": cache_info.get("registration_timestamp"),
        "last_verified": cache_info.get("last_verified"),
        "cached_status": cache_info.get("registered", False),
        "cache_source": cache_info.get("cache_source", "unknown"),
    }


def _gymnasium_version_info() -> Dict[str, Any]:
    try:
        return {
            "gymnasium_version": gymnasium.__version__,
            "registry_available": hasattr(gymnasium.envs, "registry"),
        }
    except Exception:
        return {"gymnasium_info_error": "Failed to retrieve Gymnasium information"}


def _system_info_for(info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_cached_environments": len(_registration_cache),
        "query_method": (
            "cached_with_verification"
            if info.get("cache_available")
            else "authoritative_registry"
        ),
    }


def get_registration_info(
    env_id: Optional[str] = None, include_config_details: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive registration information retrieval function providing detailed environment
    metadata, configuration parameters, registration status, and debugging information for
    monitoring and troubleshooting.

    This function provides complete registration information for debugging, monitoring, and
    administrative purposes, including detailed configuration analysis and system status.

    Args:
        env_id: Environment identifier to get information for, defaults to ENV_ID if not provided
        include_config_details: Whether to include detailed configuration parameter breakdown

    Returns:
        Complete registration information dictionary including status, configuration, metadata, and debugging details

    Example:
        # Basic registration info
        info = get_registration_info()
        print(f"Status: {info['registered']}")

        # Detailed configuration analysis
        detailed_info = get_registration_info(include_config_details=True)
        print(f"Config: {detailed_info['config_details']}")
    """
    try:
        effective_env_id = env_id or ENV_ID

        info = _base_registration_info(effective_env_id)

        is_currently_registered = is_registered(effective_env_id, use_cache=False)
        info["registered"] = is_currently_registered

        if is_currently_registered:
            info |= _env_spec_info(effective_env_id)

        if include_config_details:
            info["config_details"] = _config_details_for_id(effective_env_id)

        cache_info = _cache_info_for_id(effective_env_id)
        if cache_info is not None:
            info["cache_info"] = cache_info

        info |= _gymnasium_version_info()
        info["system_info"] = _system_info_for(info)

        _logger.debug(
            f"Retrieved registration info for '{effective_env_id}', detailed={include_config_details}"
        )
        return info

    except Exception as e:
        _logger.error(
            f"Failed to retrieve registration info for '{env_id or ENV_ID}': {e}"
        )
        return {
            "env_id": env_id or ENV_ID,
            "error": str(e),
            "query_timestamp": time.time(),
            "registered": False,
        }


def _assert_grid_size_or_raise(grid_size: Any) -> Tuple[int, int]:
    """Validate grid_size and return (width, height) or raise ValidationError."""
    if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
        raise ValidationError(
            "grid_size must be a tuple or list of exactly 2 elements",
            parameter_name="grid_size",
            parameter_value=grid_size,
            expected_format="(width, height) tuple with positive integers",
        )
    width, height = grid_size
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValidationError(
            "grid_size dimensions must be integers",
            parameter_name="grid_size",
            parameter_value=grid_size,
            expected_format="(width, height) tuple with positive integer dimensions",
        )
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
    source_location: Any, width: int, height: int
) -> Tuple[float, float]:
    """Validate source_location against grid bounds; return (x,y) or raise."""
    if not isinstance(source_location, (tuple, list)) or len(source_location) != 2:
        raise ValidationError(
            "source_location must be a tuple or list of exactly 2 elements",
            parameter_name="source_location",
            parameter_value=source_location,
            expected_format="(x, y) tuple with coordinates within grid bounds",
        )
    source_x, source_y = source_location
    if not isinstance(source_x, (int, float)) or not isinstance(source_y, (int, float)):
        raise ValidationError(
            "source_location coordinates must be numeric",
            parameter_name="source_location",
            parameter_value=source_location,
        )
    if source_x < 0 or source_x >= width or source_y < 0 or source_y >= height:
        raise ValidationError(
            f"source_location coordinates must be within grid bounds: (0,0) to ({width-1},{height-1})",
            parameter_name="source_location",
            parameter_value=source_location,
        )
    return float(source_x), float(source_y)


def _assert_max_steps_or_raise(max_steps: Any) -> int:
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


def _assert_goal_radius_or_raise(goal_radius: Any, width: int, height: int) -> float:
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
    base_kwargs: Dict[str, Any], additional_kwargs: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
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


def create_registration_kwargs(
    grid_size: Optional[Tuple[int, int]] = None,
    source_location: Optional[Tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    goal_radius: Optional[float] = None,
    additional_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Registration kwargs factory function for Gymnasium register() calls with comprehensive
    parameter validation, default value application, and configuration consistency ensuring
    proper environment setup.

    This function creates validated parameter dictionaries for environment registration,
    providing comprehensive validation, default application, and consistency checking for
    all configuration parameters.

    Args:
        grid_size: Grid dimensions as (width, height) tuple, defaults to DEFAULT_GRID_SIZE
        source_location: Plume source coordinates as (x, y) tuple, defaults to DEFAULT_SOURCE_LOCATION
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
            f"Creating registration kwargs with parameters: grid_size={effective_grid_size}, source_location={effective_source_location}"
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

        # Warn if goal radius extends beyond edges
        _warn_if_goal_radius_edges(
            effective_goal_radius, source_x, source_y, width, height
        )

        # Log kwargs creation with parameter summary and validation status for debugging
        _logger.debug(f"Successfully created registration kwargs: {base_kwargs}")

        # Return complete kwargs dictionary ready for Gymnasium registration with comprehensive validation
        return base_kwargs

    except Exception as e:
        _logger.error(f"Failed to create registration kwargs: {e}")
        raise


def _init_validation_report(strict_validation: bool) -> Dict[str, Any]:
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


def _validate_env_id(env_id: str, report: Dict[str, Any]) -> bool:
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
    entry_point: str, report: Dict[str, Any], strict_validation: bool
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
    kwargs: Dict[str, Any], report: Dict[str, Any], strict_validation: bool
) -> bool:
    valid = True
    if not isinstance(kwargs, dict):
        report["errors"].append("kwargs must be a dictionary")
        return False

    if "grid_size" in kwargs:
        valid &= _validate_grid_size_value(
            kwargs["grid_size"], report, strict_validation
        )

    if "source_location" in kwargs:
        valid &= _validate_source_location_value(kwargs["source_location"], report)

    if "goal_radius" in kwargs:
        valid &= _validate_goal_radius_value(kwargs["goal_radius"], report)

    return valid


def _cross_validate_params(kwargs: Dict[str, Any], report: Dict[str, Any]) -> None:
    if "grid_size" in kwargs and "source_location" in kwargs:
        try:
            width, height = kwargs["grid_size"]
            source_x, source_y = kwargs["source_location"]
            if source_x < 0 or source_x >= width or source_y < 0 or source_y >= height:
                report["errors"].append(
                    "source_location must be within grid_size bounds"
                )
        except (ValueError, TypeError):
            report["warnings"].append(
                "Could not cross-validate grid_size and source_location"
            )


def _analyze_performance(kwargs: Dict[str, Any], report: Dict[str, Any]) -> None:
    if "grid_size" not in kwargs:
        return
    try:
        width, height = kwargs["grid_size"]
        grid_cells = width * height
        estimated_memory_mb = (grid_cells * 4) / (1024 * 1024)
        report["performance_analysis"] = {
            "grid_cells": grid_cells,
            "estimated_memory_mb": round(estimated_memory_mb, 2),
            "performance_tier": (
                "high"
                if grid_cells > 262144
                else "medium" if grid_cells > 16384 else "low"
            ),
        }
        if estimated_memory_mb > 100:
            report["warnings"].append(
                f"High memory usage estimated: {estimated_memory_mb:.1f}MB"
            )
    except Exception:
        report["warnings"].append("Could not estimate performance characteristics")


def _strict_checks(env_id: str, kwargs: Dict[str, Any], report: Dict[str, Any]) -> None:
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
    if (
        "goal_radius" in kwargs
        and kwargs["goal_radius"] > 0
        and ("source_location" in kwargs and "grid_size" in kwargs)
    ):
        with contextlib.suppress(Exception):
            width, height = kwargs["grid_size"]
            source_x, source_y = kwargs["source_location"]
            goal_radius = kwargs["goal_radius"]

            min_distance_to_edge = min(
                source_x, source_y, width - source_x - 1, height - source_y - 1
            )
            if goal_radius > min_distance_to_edge:
                report["warnings"].append("Goal radius extends beyond grid boundaries")


def _finalize_compatibility(
    entry_point: str, param_types_valid: bool, report: Dict[str, Any]
) -> None:
    report["compatibility_check"] = {
        "gymnasium_available": True,
        "entry_point_format_valid": ":" in entry_point,
        "parameter_types_valid": param_types_valid,
    }


def validate_registration_config(
    env_id: str,
    entry_point: str,
    max_episode_steps: int,
    kwargs: Dict[str, Any],
    strict_validation: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Configuration validation function ensuring parameter consistency, Gymnasium compliance,
    mathematical feasibility, and performance requirements for robust environment registration.

    This function provides comprehensive validation of all registration parameters, ensuring
    compatibility with Gymnasium requirements, mathematical feasibility, and performance constraints.

    Args:
        env_id: Environment identifier string to validate
        entry_point: Entry point specification to validate
        max_episode_steps: Maximum episode steps parameter to validate
        kwargs: Environment parameters dictionary to validate
        strict_validation: Whether to apply enhanced validation rules and constraints

    Returns:
        Tuple of (is_valid: bool, validation_report: dict) with detailed configuration analysis and recommendations

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
            "errors": [f"Validation process failed: {str(e)}"],
            "warnings": [],
            "recommendations": ["Check validation parameters and try again"],
        }


def register_with_custom_params(
    grid_size: Optional[Tuple[int, int]] = None,
    source_location: Optional[Tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    goal_radius: Optional[float] = None,
    custom_env_id: Optional[str] = None,
    force_reregister: bool = False,
) -> str:
    """
    Convenience registration function with custom parameter overrides providing streamlined
    environment registration with validation, error handling, and immediate availability for
    specialized research configurations.

    This function provides a convenient interface for registering environments with custom
    parameters, handling all validation and configuration automatically while supporting
    specialized research requirements.

    Args:
        grid_size: Custom grid dimensions, defaults to DEFAULT_GRID_SIZE if not provided
        source_location: Custom source location, defaults to DEFAULT_SOURCE_LOCATION if not provided
        max_steps: Custom maximum steps, defaults to DEFAULT_MAX_STEPS if not provided
        goal_radius: Custom goal radius, defaults to DEFAULT_GOAL_RADIUS if not provided
        custom_env_id: Custom environment identifier, defaults to ENV_ID if not provided
        force_reregister: Whether to force re-registration if environment already exists

    Returns:
        Registered environment ID ready for immediate use with gym.make() calls

    Raises:
        ValidationError: If custom parameters fail validation
        ConfigurationError: If registration fails due to configuration issues

    Example:
        # Register with larger grid
        env_id = register_with_custom_params(
            grid_size=(256, 256),
            source_location=(128, 128)
        )
        env = gym.make(env_id)

        # Register completely custom environment
        env_id = register_with_custom_params(
            grid_size=(64, 64),
            source_location=(32, 32),
            goal_radius=3.0,
            custom_env_id="SmallPlume-v0"
        )
    """
    try:
        # Generate custom environment ID if custom_env_id provided with version suffix validation
        if custom_env_id:
            if not custom_env_id.endswith("-v0"):
                custom_env_id = f"{custom_env_id}-v0"
                _logger.info(
                    f"Added version suffix to custom environment ID: {custom_env_id}"
                )
            effective_env_id = custom_env_id
        else:
            effective_env_id = ENV_ID

        _logger.info(
            f"Registering environment with custom parameters: {effective_env_id}"
        )

        # Create complete kwargs using create_registration_kwargs() with provided parameters
        registration_kwargs = create_registration_kwargs(
            grid_size=grid_size,
            source_location=source_location,
            max_steps=max_steps,
            goal_radius=goal_radius,
        )

        # Apply custom environment ID or use default ENV_ID for registration
        # Call register_env() with custom parameters and force_reregister flag
        registered_env_id = register_env(
            env_id=effective_env_id,
            entry_point=ENTRY_POINT,
            max_episode_steps=max_steps or MAX_EPISODE_STEPS,
            kwargs=registration_kwargs,
            force_reregister=force_reregister,
        )

        # Validate successful registration with immediate gym.make() test
        try:
            test_env = gymnasium.make(registered_env_id)
            test_env.close()
            _logger.debug(f"Custom registration verified for '{registered_env_id}'")
        except Exception as test_error:
            _logger.error(f"Custom registration verification failed: {test_error}")
            raise ConfigurationError(
                f"Custom registration verification failed: {test_error}",
                config_parameter="custom_registration",
            ) from test_error

        # Log custom registration with parameter overrides and configuration summary
        param_summary = {
            "grid_size": grid_size or DEFAULT_GRID_SIZE,
            "source_location": source_location or DEFAULT_SOURCE_LOCATION,
            "max_steps": max_steps or DEFAULT_MAX_STEPS,
            "goal_radius": (
                goal_radius if goal_radius is not None else DEFAULT_GOAL_RADIUS
            ),
        }
        _logger.info(
            f"Successfully registered custom environment '{registered_env_id}' with parameters: {param_summary}"
        )

        # Return registered environment ID for immediate use with success confirmation
        return registered_env_id

    except Exception as e:
        _logger.error(f"Custom registration failed: {e}")
        raise
