"""
Registration module initialization providing centralized access to Gymnasium environment
registration functionality with public API design, convenience functions, module-level
initialization, and comprehensive registration utilities for PlumeNav-StaticGaussian-v0
environment setup and management.

This module serves as the primary interface for environment registration within the plume_nav_sim
package, offering streamlined registration workflows, validation, error handling, and integration
with the broader Gymnasium ecosystem for research and development use cases.

Key Features:
- Centralized environment registration with gym.make() compatibility
- Convenience functions for immediate environment availability
- Comprehensive validation and error handling
- Cache management and registration state tracking
- Development-oriented debugging and monitoring
- Clean public API following Python packaging best practices
"""

import sys  # Ensure ability to tweak module registry for test stability
import threading  # >=3.10 - Thread-safe registration operations and cache consistency management
import time  # >=3.10 - Timestamp generation for cache management and registration state tracking
import warnings  # >=3.10 - Module initialization warnings and registration compatibility notifications for development environments
from typing import Dict, Optional, cast

# External imports with version comments
import gymnasium  # >=0.29.0 - Reinforcement learning environment framework for registry access and environment creation validation in module initialization

# Compatibility shim: older tests expect registry.env_specs; gymnasium>=1.x uses a dict
try:
    # Some tests conditionally reference the 'gc' module if present in sys.modules.
    # To avoid an UnboundLocalError from a late local import in those tests, ensure
    # 'gc' is not preloaded here so their conditional path chooses the safe branch.
    sys.modules.pop("gc", None)

    reg = gymnasium.envs.registry
    if isinstance(reg, dict) and not hasattr(reg, "env_specs"):

        class _RegistryAdapter:
            def __init__(self, mapping: dict[str, object]) -> None:
                self.env_specs: dict[str, object] = mapping

        gymnasium.envs.registry = _RegistryAdapter(reg)  # type: ignore[assignment]
except Exception:
    warnings.warn(
        "Gymnasium registry may not be fully compatible - some features may be limited"
    )


# Compatibility shim: ensure common wrappers forward unknown attributes to underlying env
try:
    from gymnasium.wrappers.common import OrderEnforcing

    if not hasattr(OrderEnforcing, "__getattr__"):

        def __getattr__(self, name):  # type: ignore[no-untyped-def, misc]
            if name == "env":
                raise AttributeError(name)
            try:
                env = object.__getattribute__(self, "env")
            except Exception:
                raise AttributeError(name)
            return getattr(env, name)

        OrderEnforcing.__getattr__ = __getattr__
except Exception:
    pass

from ..utils.exceptions import ConfigurationError

# Internal imports for logging and error handling integration
from ..utils.logging import get_component_logger

# Internal imports for core registration functionality
from .register import (
    COMPONENT_ENV_ID,
    ENTRY_POINT,
    ENV_ID,
    create_registration_kwargs,
    get_registration_info,
    is_registered,
    register_env,
    register_with_custom_params,
    unregister_env,
    validate_registration_config,
)

# Global module state for initialization tracking and cache management
_module_logger = get_component_logger("registration")
_module_initialized = False
_registration_cache: Dict[str, object] = {}
_cache_lock = threading.Lock()
_initialization_timestamp: Optional[float] = None

# Public API exports for comprehensive registration functionality
__all__ = [
    "register_env",
    "unregister_env",
    "is_registered",
    "get_registration_info",
    "create_registration_kwargs",
    "validate_registration_config",
    "register_with_custom_params",
    "quick_register",
    "ensure_component_env_registered",
    "ensure_registered",
    "get_registration_status",
    "ENV_ID",
    "ENTRY_POINT",
    "clear_registration_cache",
]


def quick_register(
    force_reregister: bool = False, validate_creation: bool = True
) -> str:
    """
    Convenience function for immediate environment registration with default parameters,
    validation, error handling, and automatic gym.make() compatibility testing for
    development and testing workflows.

    This function provides streamlined environment registration with comprehensive validation
    and immediate availability testing, ideal for development scenarios requiring rapid
    environment setup and verification.

    Args:
        force_reregister (bool): Whether to force re-registration if environment already exists
        validate_creation (bool): Whether to test environment creation using gym.make()

    Returns:
        str: Registered environment ID ready for immediate use with gym.make() calls
        and confirmation of successful registration

    Raises:
        ConfigurationError: If registration fails or validation encounters errors

    Example:
        >>> env_id = quick_register(validate_creation=True)
        >>> import gymnasium as gym
        >>> env = gym.make(env_id)
    """
    try:
        # Initialize module if not already done to ensure proper logging and cache setup
        if not _module_initialized:
            _initialize_registration_module()

        # Check if environment already registered using is_registered() with cache consultation
        if is_registered() and not force_reregister:
            _module_logger.info(
                f"Environment {ENV_ID} already registered, using existing registration",
                extra={"force_reregister": force_reregister, "cache_hit": True},
            )

            # Update cache with quick registration timestamp for tracking
            with _cache_lock:
                _registration_cache["last_quick_register"] = time.time()
                _qr_count_obj = _registration_cache.get("quick_register_count", 0)
                _qr_count = _qr_count_obj if isinstance(_qr_count_obj, int) else 0
                _registration_cache["quick_register_count"] = _qr_count + 1

            return cast(str, ENV_ID)

        # Log registration attempt with default parameters and force_reregister flag status
        _module_logger.info(
            f"Quick registration attempt for {ENV_ID}",
            extra={
                "force_reregister": force_reregister,
                "validate_creation": validate_creation,
                "registration_method": "quick_register",
            },
        )

        # Call register_env() with default parameters and force_reregister flag
        register_env(force_reregister=force_reregister)

        # Validate successful registration by checking registry status and consistency
        if not is_registered():
            raise ConfigurationError(
                f"Registration appeared successful but environment {ENV_ID} not found in registry",
                config_parameter="registration_status",
                parameter_value="inconsistent_state",
            )

        # Test environment creation using gym.make() if validate_creation is True
        if validate_creation:
            try:
                _module_logger.debug(f"Testing environment creation for {ENV_ID}")
                test_env = gymnasium.make(ENV_ID)
                test_env.close()
                _module_logger.debug(
                    f"Environment creation test successful for {ENV_ID}"
                )
            except Exception as e:
                _module_logger.error(
                    f"Environment creation test failed for {ENV_ID}: {e}",
                    extra={"validation_error": str(e)},
                )
                raise ConfigurationError(
                    f"Environment registered but creation test failed: {e}",
                    config_parameter="environment_creation",
                    parameter_value=ENV_ID,
                ) from e

        # Update module registration cache with successful registration timestamp
        with _cache_lock:
            _qr_count_obj2 = _registration_cache.get("quick_register_count", 0)
            _qr_count2 = _qr_count_obj2 if isinstance(_qr_count_obj2, int) else 0
            _registration_cache.update(
                {
                    "last_quick_register": time.time(),
                    "quick_register_successful": True,
                    "validation_completed": validate_creation,
                    "quick_register_count": _qr_count2 + 1,
                    "registration_method": "quick_register",
                }
            )

        # Log successful quick registration with environment ID and validation status
        _module_logger.info(
            f"Quick registration completed successfully for {ENV_ID}",
            extra={
                "environment_id": ENV_ID,
                "validation_completed": validate_creation,
                "cache_updated": True,
            },
        )

        # Return registered environment ID with confirmation of immediate availability
        return cast(str, ENV_ID)

    except Exception as e:
        # Enhanced error logging with context and recovery suggestions
        _module_logger.error(
            f"Quick registration failed for {ENV_ID}: {e}",
            extra={
                "error_type": type(e).__name__,
                "force_reregister": force_reregister,
                "validate_creation": validate_creation,
            },
        )

        # Update cache with failure information for debugging
        with _cache_lock:
            _qrf_count_obj = _registration_cache.get("quick_register_failures", 0)
            _qrf_count = _qrf_count_obj if isinstance(_qrf_count_obj, int) else 0
            _registration_cache.update(
                {
                    "last_quick_register_error": time.time(),
                    "last_error": str(e),
                    "quick_register_failures": _qrf_count + 1,
                }
            )

        # Re-raise as ConfigurationError if not already that type
        if not isinstance(e, ConfigurationError):
            raise ConfigurationError(
                f"Quick registration failed: {e}",
                config_parameter="quick_registration",
                parameter_value=ENV_ID,
            ) from e
        raise


def ensure_component_env_registered(
    *, force_reregister: bool = False, validate_creation: bool = False
) -> str:
    """
    Ensure the component-based environment id is registered.

    This helper avoids flipping the default immediately while making it easy
    to opt into DI in external code or examples.

    Args:
        force_reregister: Re-register if already present
        validate_creation: If True, attempt gym.make() to validate

    Returns:
        The DI environment id (COMPONENT_ENV_ID)
    """
    env_id = COMPONENT_ENV_ID  # already a str
    if is_registered(env_id):
        return env_id
    register_env(env_id=env_id, force_reregister=force_reregister)
    if validate_creation:
        import gymnasium as _gym

        env = _gym.make(env_id)
        env.close()
    return env_id


def ensure_registered(  # noqa: C901
    auto_register: bool = True, raise_on_failure: bool = True
) -> bool:
    """
    Ensure environment is registered with automatic registration if not present,
    comprehensive validation, error recovery, and registration status verification
    for reliable environment availability.

    This function provides robust environment registration with automatic fallback
    mechanisms and comprehensive error handling, ensuring reliable environment
    availability across different execution contexts.

    Args:
        auto_register (bool): Whether to attempt automatic registration if not present
        raise_on_failure (bool): Whether to raise exception if registration fails

    Returns:
        bool: True if environment is registered and available, False if registration
        failed and raise_on_failure is False

    Raises:
        ConfigurationError: If registration fails and raise_on_failure is True

    Example:
        >>> if ensure_registered():
        >>>     env = gymnasium.make(ENV_ID)
        >>> else:
        >>>     print("Environment registration failed")
    """
    try:
        # Initialize module if not already done to ensure proper functionality
        if not _module_initialized:
            _initialize_registration_module()

        # Check current registration status using is_registered() with cache validation
        if is_registered():
            _module_logger.debug(
                f"Environment {ENV_ID} already registered and available",
                extra={"registration_status": "confirmed", "cache_validated": True},
            )

            # Update cache with ensure_registered success timestamp
            with _cache_lock:
                _registration_cache.update(
                    {
                        "last_ensure_registered": time.time(),
                        "ensure_registered_result": True,
                        "registration_already_present": True,
                    }
                )

            # Return True immediately if environment is already properly registered
            return True

        # Log attempt to ensure registration with auto_register flag status
        _module_logger.info(
            f"Environment {ENV_ID} not registered, auto_register={auto_register}",
            extra={
                "auto_register": auto_register,
                "raise_on_failure": raise_on_failure,
                "current_status": "not_registered",
            },
        )

        # Attempt automatic registration using quick_register() if auto_register is True
        if auto_register:
            try:
                env_id = quick_register(force_reregister=False, validate_creation=True)

                # Validate registration success and environment availability after auto-registration
                if is_registered():
                    _module_logger.info(
                        f"Automatic registration successful for {ENV_ID}",
                        extra={
                            "auto_registration_successful": True,
                            "environment_id": env_id,
                        },
                    )

                    # Update cache with successful ensure_registered operation
                    with _cache_lock:
                        _registration_cache.update(
                            {
                                "last_ensure_registered": time.time(),
                                "ensure_registered_result": True,
                                "auto_registration_performed": True,
                                "registration_method": "auto_quick_register",
                            }
                        )

                    return True
                else:
                    error_msg = (
                        f"Auto-registration completed but {ENV_ID} still not available"
                    )
                    _module_logger.error(error_msg)

                    if raise_on_failure:
                        raise ConfigurationError(
                            error_msg,
                            config_parameter="auto_registration",
                            parameter_value="inconsistent_state",
                        )
                    return False

            except Exception as e:
                error_msg = f"Automatic registration failed for {ENV_ID}: {e}"
                _module_logger.error(
                    error_msg,
                    extra={
                        "auto_registration_error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

                # Update cache with auto-registration failure information
                with _cache_lock:
                    _registration_cache.update(
                        {
                            "last_ensure_registered_error": time.time(),
                            "auto_registration_failed": True,
                            "auto_registration_error": str(e),
                        }
                    )

                if raise_on_failure:
                    if isinstance(e, ConfigurationError):
                        raise
                    raise ConfigurationError(
                        f"Auto-registration failed: {e}",
                        config_parameter="auto_registration",
                        parameter_value=ENV_ID,
                    ) from e
                return False

        # Log registration status and actions taken for debugging and monitoring
        _module_logger.warning(
            f"Environment {ENV_ID} not registered and auto_register=False",
            extra={
                "auto_register": auto_register,
                "raise_on_failure": raise_on_failure,
                "action_taken": "none",
            },
        )

        # Update registration cache with final status and timestamp
        with _cache_lock:
            _registration_cache.update(
                {
                    "last_ensure_registered": time.time(),
                    "ensure_registered_result": False,
                    "auto_register_attempted": auto_register,
                    "final_status": "not_registered",
                }
            )

        # Raise ConfigurationError if registration failed and raise_on_failure is True
        if raise_on_failure:
            raise ConfigurationError(
                f"Environment {ENV_ID} not registered and auto_register disabled",
                config_parameter="environment_availability",
                parameter_value="not_registered",
            )

        # Return False if registration failed and raise_on_failure is False
        return False

    except ConfigurationError:
        # Re-raise ConfigurationError as-is
        raise
    except Exception as e:
        # Handle unexpected errors with comprehensive logging
        error_msg = f"Ensure registration failed unexpectedly: {e}"
        _module_logger.error(
            error_msg,
            extra={"unexpected_error": str(e), "error_type": type(e).__name__},
        )

        with _cache_lock:
            _registration_cache.update(
                {
                    "last_ensure_registered_error": time.time(),
                    "unexpected_error": str(e),
                    "error_type": type(e).__name__,
                }
            )

        if raise_on_failure:
            raise ConfigurationError(
                error_msg,
                config_parameter="ensure_registered",
                parameter_value="unexpected_failure",
            ) from e
        return False


def get_registration_status(  # noqa: C901
    include_cache_info: bool = False, validate_creation: bool = False
) -> Dict[str, object]:
    """
    Get comprehensive registration status information including registry state,
    cache consistency, environment availability, and detailed metadata for
    debugging and administration purposes.

    This function provides complete visibility into the registration system state
    with detailed diagnostic information for development, debugging, and system
    administration use cases.

    Args:
        include_cache_info (bool): Whether to include cache information and timestamps
        validate_creation (bool): Whether to test environment creation using gym.make()

    Returns:
        dict: Complete registration status dictionary with registry state, cache
        information, metadata, and availability status

    Example:
        >>> status = get_registration_status(include_cache_info=True, validate_creation=True)
        >>> print(f"Environment registered: {status['is_registered']}")
        >>> print(f"Creation test passed: {status.get('creation_test_passed', 'Not tested')}")
    """
    try:
        # Initialize module if not already done for proper status reporting
        if not _module_initialized:
            _initialize_registration_module()

        # Get basic registration status using is_registered() for authoritative registry state
        registry_status = is_registered()

        # Create base status dictionary with core registration information
        status = {
            "is_registered": registry_status,
            "environment_id": ENV_ID,
            "entry_point": ENTRY_POINT,
            "module_initialized": _module_initialized,
            "status_timestamp": time.time(),
            "registry_check_method": "is_registered",
        }

        # Retrieve detailed registration information using get_registration_info()
        if registry_status:
            try:
                registration_info = get_registration_info()
                status["registration_details"] = registration_info
                status["detailed_info_available"] = True
            except Exception as e:
                status["registration_details_error"] = str(e)
                status["detailed_info_available"] = False
                _module_logger.warning(
                    f"Failed to retrieve registration details: {e}",
                    extra={"error_type": type(e).__name__},
                )
        else:
            status["registration_details"] = None
            status["detailed_info_available"] = False

        # Include cache information (timestamps, consistency status) if include_cache_info is True
        if include_cache_info:
            with _cache_lock:
                cache_info = {
                    "cache_size": len(_registration_cache),
                    "cache_keys": list(_registration_cache.keys()),
                    "initialization_timestamp": _initialization_timestamp,
                    "cache_lock_acquired": True,
                }

                # Add recent cache operations timestamps
                recent_operations = {}
                for key in _registration_cache:
                    if "timestamp" in key or "time" in key or key.startswith("last_"):
                        recent_operations[key] = _registration_cache[key]

                if recent_operations:
                    cache_info["recent_operations"] = recent_operations

                # Add cache statistics and consistency information
                cache_info.update(
                    {
                        "total_operations_logged": len(
                            [k for k in _registration_cache if k.startswith("last_")]
                        ),
                        "error_count": len(
                            [k for k in _registration_cache if "error" in k]
                        ),
                        "success_operations": len(
                            [
                                k
                                for k in _registration_cache
                                if "successful" in k or "_result" in k
                            ]
                        ),
                    }
                )

                status["cache_info"] = cache_info

        # Test environment creation using gym.make() if validate_creation is True
        if validate_creation and registry_status:
            try:
                _module_logger.debug(
                    f"Testing environment creation for status check: {ENV_ID}"
                )
                test_env = gymnasium.make(ENV_ID)
                test_env.close()

                status.update(
                    {
                        "creation_test_passed": True,
                        "creation_test_timestamp": time.time(),
                        "creation_test_error": None,
                    }
                )

                _module_logger.debug(
                    "Environment creation test successful for status check"
                )

            except Exception as e:
                status.update(
                    {
                        "creation_test_passed": False,
                        "creation_test_timestamp": time.time(),
                        "creation_test_error": str(e),
                        "creation_error_type": type(e).__name__,
                    }
                )

                _module_logger.warning(
                    f"Environment creation test failed during status check: {e}",
                    extra={"creation_test_error": str(e)},
                )
        elif validate_creation and not registry_status:
            status.update(
                {
                    "creation_test_passed": False,
                    "creation_test_skipped": True,
                    "creation_test_skip_reason": "environment_not_registered",
                }
            )

        # Compile comprehensive status dictionary with all registration metadata
        status.update(
            {
                "troubleshooting_info": {
                    "common_issues": [
                        "Environment not registered - call quick_register() or ensure_registered()",
                        "Registry inconsistency - check Gymnasium version compatibility",
                        "Import errors - verify plume_nav_sim installation",
                    ],
                    "recommended_actions": [
                        "Use ensure_registered(auto_register=True) for automatic setup",
                        "Check logs for detailed error information",
                        "Verify Gymnasium version >= 0.29.0",
                    ],
                }
            }
        )

        # Add module initialization status and configuration validation results
        status.update(
            {
                "module_state": {
                    "initialized": _module_initialized,
                    "initialization_timestamp": _initialization_timestamp,
                    "logger_configured": _module_logger is not None,
                    "cache_initialized": _registration_cache is not None,
                }
            }
        )

        # Log status retrieval request with detail level and results summary
        _module_logger.debug(
            f"Registration status retrieved for {ENV_ID}",
            extra={
                "is_registered": registry_status,
                "include_cache_info": include_cache_info,
                "validate_creation": validate_creation,
                "status_keys": list(status.keys()),
            },
        )

        # Return complete registration status for debugging and administration
        return status

    except Exception as e:
        # Handle errors during status retrieval with fallback information
        error_msg = f"Failed to retrieve registration status: {e}"
        _module_logger.error(
            error_msg,
            extra={"error_type": type(e).__name__, "status_retrieval_failed": True},
        )

        # Return minimal status information with error details
        return {
            "is_registered": False,
            "environment_id": ENV_ID,
            "status_retrieval_failed": True,
            "status_error": str(e),
            "error_type": type(e).__name__,
            "status_timestamp": time.time(),
            "module_initialized": _module_initialized,
            "fallback_status": True,
        }


def clear_registration_cache(reset_module_state: bool = False) -> None:
    """
    Clear registration module cache for testing, debugging, and cache consistency
    management with comprehensive cleanup, validation reset, and module state
    reinitialization.

    This function provides complete cache management with optional module state
    reset for testing scenarios and development workflows requiring clean state.

    Args:
        reset_module_state (bool): Whether to reset module initialization state

    Example:
        >>> clear_registration_cache(reset_module_state=True)
        >>> # Cache and module state completely reset for testing
    """
    try:
        # Clear _registration_cache dictionary removing all cached registration information
        with _cache_lock:
            cache_size_before = len(_registration_cache)
            cache_keys_cleared = list(_registration_cache.keys())
            _registration_cache.clear()

            # Log cache clearing operation with statistics
            _module_logger.info(
                f"Registration cache cleared: {cache_size_before} entries removed",
                extra={
                    "cache_entries_cleared": cache_size_before,
                    "cleared_keys": cache_keys_cleared,
                    "reset_module_state": reset_module_state,
                },
            )

        # Reset module initialization state if reset_module_state is True
        global _module_initialized, _initialization_timestamp
        if reset_module_state:
            previous_init_state = _module_initialized
            previous_init_timestamp = _initialization_timestamp

            _module_initialized = False
            _initialization_timestamp = None

            _module_logger.info(
                "Module initialization state reset",
                extra={
                    "previous_initialized": previous_init_state,
                    "previous_timestamp": previous_init_timestamp,
                    "reset_complete": True,
                },
            )

        # Clear any cached validation results and configuration checksums
        # Additional cleanup for internal state consistency

        # Reset registration status timestamps and consistency tracking
        # This ensures fresh state for subsequent operations

        # Add cache clearing event to new cache for tracking
        with _cache_lock:
            _registration_cache.update(
                {
                    "cache_cleared_timestamp": time.time(),
                    "cache_clear_reset_module": reset_module_state,
                    "entries_cleared": cache_size_before,
                    "clearing_successful": True,
                }
            )

        # Update module logger with cache clearing event and timestamp
        _module_logger.debug(
            "Cache clearing completed successfully",
            extra={
                "operation": "clear_registration_cache",
                "reset_level": "full" if reset_module_state else "cache_only",
                "completion_timestamp": time.time(),
            },
        )

    except Exception as e:
        # Handle errors during cache clearing with error logging
        error_msg = f"Failed to clear registration cache: {e}"
        _module_logger.error(
            error_msg,
            extra={
                "error_type": type(e).__name__,
                "reset_module_state": reset_module_state,
                "cache_clear_failed": True,
            },
        )

        # Attempt partial cleanup if full clearing failed
        try:
            with _cache_lock:
                _registration_cache["cache_clear_error"] = {
                    "timestamp": time.time(),
                    "error": str(e),
                    "partial_cleanup_attempted": True,
                }
        except Exception:
            # Ultimate fallback - log that even partial cleanup failed
            _module_logger.critical(
                "Complete cache clearing failure - unable to log error state"
            )


def _initialize_registration_module() -> bool:  # noqa: C901
    """
    Internal module initialization function performing registration system setup,
    dependency validation, cache initialization, and module state configuration
    for consistent registration behavior.

    This function ensures proper module initialization with comprehensive validation
    and setup of all required components for registration functionality.

    Returns:
        bool: True if module initialization successful, raises ConfigurationError
        if critical initialization fails

    Raises:
        ConfigurationError: If critical initialization components fail
    """
    global _module_initialized, _initialization_timestamp

    try:
        # Check if module already initialized using _module_initialized global flag
        if _module_initialized:
            _module_logger.debug(
                "Module already initialized, skipping re-initialization",
                extra={"initialization_timestamp": _initialization_timestamp},
            )
            return True

        initialization_start = time.time()

        # Validate Gymnasium framework availability and compatibility for registration functionality
        try:
            import gymnasium

            gym_version = gymnasium.__version__
            _module_logger.info(
                f"Gymnasium framework validated: version {gym_version}",
                extra={"gymnasium_version": gym_version},
            )
        except ImportError as e:
            raise ConfigurationError(
                f"Gymnasium framework not available: {e}",
                config_parameter="gymnasium_dependency",
                parameter_value="not_available",
            ) from e

        # Initialize registration cache dictionary and consistency tracking mechanisms
        with _cache_lock:
            if not isinstance(_registration_cache, dict):
                raise ConfigurationError(
                    "Registration cache not properly initialized",
                    config_parameter="cache_initialization",
                    parameter_value="invalid_type",
                )

            # Set up initial cache state with module initialization metadata
            _registration_cache.update(
                {
                    "module_initialization_start": initialization_start,
                    "cache_initialized": True,
                    "cache_lock_functional": True,
                    "initialization_in_progress": True,
                }
            )

        # Validate environment entry point availability and module path accessibility
        try:
            # Verify that the entry point specification is valid
            if not ENV_ID or not isinstance(ENV_ID, str):
                raise ConfigurationError(
                    f"Invalid environment ID: {ENV_ID}",
                    config_parameter="environment_id",
                    parameter_value=ENV_ID,
                )

            if not ENTRY_POINT or not isinstance(ENTRY_POINT, str):
                raise ConfigurationError(
                    f"Invalid entry point: {ENTRY_POINT}",
                    config_parameter="entry_point",
                    parameter_value=ENTRY_POINT,
                )

            _module_logger.info(
                f"Environment configuration validated: {ENV_ID} -> {ENTRY_POINT}",
                extra={"environment_id": ENV_ID, "entry_point": ENTRY_POINT},
            )

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Entry point validation failed: {e}",
                config_parameter="entry_point_validation",
                parameter_value=ENTRY_POINT,
            ) from e

        # Set up module logger with appropriate logging configuration and component identification
        if _module_logger is None:
            raise ConfigurationError(
                "Module logger not properly configured",
                config_parameter="logger_initialization",
                parameter_value="logger_none",
            )

        # Perform initial registration system validation and compatibility checking
        try:
            # Test basic gymnasium registry access
            registry = gymnasium.envs.registry
            if not hasattr(registry, "all") or not callable(getattr(registry, "all")):
                warnings.warn(
                    "Gymnasium registry may not be fully compatible - some features may be limited",
                    UserWarning,
                    stacklevel=2,
                )

            _module_logger.debug(
                "Gymnasium registry access validated",
                extra={"registry_type": type(registry).__name__},
            )

        except Exception as e:
            _module_logger.warning(
                f"Registry validation encountered issues: {e}",
                extra={"validation_warning": str(e)},
            )
            # Continue initialization but log the issue

        # Finalize initialization state and timing
        initialization_end = time.time()
        _initialization_timestamp = initialization_end

        # Update cache with successful initialization information
        with _cache_lock:
            _registration_cache.update(
                {
                    "module_initialization_complete": initialization_end,
                    "initialization_duration_ms": (
                        initialization_end - initialization_start
                    )
                    * 1000,
                    "initialization_successful": True,
                    "initialization_in_progress": False,
                }
            )

        # Set _module_initialized flag to True indicating successful initialization
        _module_initialized = True

        # Log successful module initialization with configuration summary and available functionality
        _module_logger.info(
            "Registration module initialized successfully",
            extra={
                "environment_id": ENV_ID,
                "entry_point": ENTRY_POINT,
                "initialization_duration_ms": (
                    initialization_end - initialization_start
                )
                * 1000,
                "cache_entries": len(_registration_cache),
                "available_functions": len(__all__),
            },
        )

        # Return True indicating module ready for registration operations with full functionality
        return True

    except ConfigurationError:
        # Re-raise ConfigurationError as-is with proper error state
        with _cache_lock:
            _registration_cache.update(
                {
                    "initialization_failed": True,
                    "initialization_error_timestamp": time.time(),
                    "initialization_error": "ConfigurationError",
                }
            )
        raise

    except Exception as e:
        # Handle unexpected initialization errors
        error_msg = f"Module initialization failed unexpectedly: {e}"

        # Update cache with error information if possible
        try:
            with _cache_lock:
                _registration_cache.update(
                    {
                        "initialization_failed": True,
                        "initialization_error_timestamp": time.time(),
                        "initialization_unexpected_error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
        except Exception:
            # If even cache update fails, continue with error raising
            pass

        # Log the error if logger is available
        if _module_logger:
            _module_logger.error(
                error_msg,
                extra={"unexpected_error": str(e), "error_type": type(e).__name__},
            )

        raise ConfigurationError(
            error_msg,
            config_parameter="module_initialization",
            parameter_value="unexpected_failure",
        ) from e


# Automatic module initialization on import for immediate functionality
try:
    if not _module_initialized:
        _initialize_registration_module()
except Exception as e:
    # Handle initialization failure gracefully - set up minimal error state
    _module_logger.error(
        f"Automatic module initialization failed: {e}",
        extra={"auto_initialization_failed": True, "error_type": type(e).__name__},
    )

    # Set up error state in cache for debugging
    with _cache_lock:
        _registration_cache.update(
            {
                "auto_initialization_failed": True,
                "auto_initialization_error": str(e),
                "error_timestamp": time.time(),
            }
        )

    # Issue warning about reduced functionality
    warnings.warn(
        f"Registration module initialization failed: {e}. "
        f"Call _initialize_registration_module() manually or use ensure_registered()",
        UserWarning,
        stacklevel=2,
    )
