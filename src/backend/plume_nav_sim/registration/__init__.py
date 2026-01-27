import sys
import threading
import time
import warnings
from typing import Dict, Mapping, Optional

import gymnasium

try:
    sys.modules.pop("gc", None)

    envs_mod = gymnasium.envs
    reg = getattr(envs_mod, "registry", None)
    if isinstance(reg, dict) and not hasattr(reg, "env_specs"):

        class _RegistryAdapter:
            def __init__(self, mapping: Mapping[str, object]) -> None:
                self.env_specs: Mapping[str, object] = mapping

        setattr(envs_mod, "registry", _RegistryAdapter(reg))
except Exception:
    warnings.warn(
        "Gymnasium registry may not be fully compatible - some features may be limited",
        stacklevel=2,
    )


try:
    from gymnasium.wrappers.common import OrderEnforcing

    def _order_enforcing_getattr(self: object, name: str) -> object:
        if name == "env":
            raise AttributeError(name)
        env = object.__getattribute__(self, "env")
        return getattr(env, name)

    setattr(OrderEnforcing, "__getattr__", _order_enforcing_getattr)
except Exception:
    pass

try:
    import gymnasium.error as _gym_error_module
except Exception:

    class _GymnasiumErrorNamespace:
        class Error(Exception):
            pass

        class UnregisteredEnv(Error):
            pass

        class RegistrationError(Error):
            pass

        class ResetNeeded(Error):
            pass

    setattr(gymnasium, "error", _GymnasiumErrorNamespace())
else:
    setattr(gymnasium, "error", _gym_error_module)

from ..utils.exceptions import ConfigurationError

from ..utils.logging import get_component_logger

from .register import ENTRY_POINT, ENV_ID, is_registered, register_env, unregister_env

_module_logger = get_component_logger("registration")
_module_initialized = False
_registration_cache: Dict[str, object] = {}
_cache_lock = threading.Lock()
_initialization_timestamp: Optional[float] = None

__all__ = [
    "register_env",
    "unregister_env",
    "is_registered",
    "ensure_registered",
    "ENV_ID",
    "ENTRY_POINT",
]


def ensure_registered(  # noqa: C901
    auto_register: bool = True, raise_on_failure: bool = True
) -> bool:
    try:
        if not _module_initialized:
            _initialize_registration_module()

        if is_registered():
            _module_logger.debug(
                f"Environment {ENV_ID} already registered and available",
                extra={"registration_status": "confirmed", "cache_validated": True},
            )

            with _cache_lock:
                _registration_cache.update(
                    {
                        "last_ensure_registered": time.time(),
                        "ensure_registered_result": True,
                        "registration_already_present": True,
                    }
                )

            return True

        _module_logger.info(
            f"Environment {ENV_ID} not registered, auto_register={auto_register}",
            extra={
                "auto_register": auto_register,
                "raise_on_failure": raise_on_failure,
                "current_status": "not_registered",
            },
        )

        if auto_register:
            try:
                register_env(force_reregister=False)
                env_id = ENV_ID

                if is_registered():
                    _module_logger.info(
                        f"Automatic registration successful for {ENV_ID}",
                        extra={
                            "auto_registration_successful": True,
                            "environment_id": env_id,
                        },
                    )

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

        _module_logger.warning(
            f"Environment {ENV_ID} not registered and auto_register=False",
            extra={
                "auto_register": auto_register,
                "raise_on_failure": raise_on_failure,
                "action_taken": "none",
            },
        )

        with _cache_lock:
            _registration_cache.update(
                {
                    "last_ensure_registered": time.time(),
                    "ensure_registered_result": False,
                    "auto_register_attempted": auto_register,
                    "final_status": "not_registered",
                }
            )

        if raise_on_failure:
            raise ConfigurationError(
                f"Environment {ENV_ID} not registered and auto_register disabled",
                config_parameter="environment_availability",
                parameter_value="not_registered",
            )

        return False

    except ConfigurationError:
        raise
    except Exception as e:
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


def _initialize_registration_module() -> bool:  # noqa: C901
    global _module_initialized, _initialization_timestamp

    try:
        if _module_initialized:
            _module_logger.debug(
                "Module already initialized, skipping re-initialization",
                extra={"initialization_timestamp": _initialization_timestamp},
            )
            return True

        initialization_start = time.time()

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

        with _cache_lock:
            if not isinstance(_registration_cache, dict):
                raise ConfigurationError(
                    "Registration cache not properly initialized",
                    config_parameter="cache_initialization",
                    parameter_value="invalid_type",
                )

            _registration_cache.update(
                {
                    "module_initialization_start": initialization_start,
                    "cache_initialized": True,
                    "cache_lock_functional": True,
                    "initialization_in_progress": True,
                }
            )

        try:
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

        if _module_logger is None:
            raise ConfigurationError(
                "Module logger not properly configured",
                config_parameter="logger_initialization",
                parameter_value="logger_none",
            )

        try:
            envs_mod2 = gymnasium.envs
            registry = getattr(envs_mod2, "registry", None)

            is_dict_registry = isinstance(registry, dict)
            has_env_specs_attr = hasattr(registry, "env_specs")

            if registry is None:
                warnings.warn(
                    "Gymnasium registry may not be fully compatible - some features may be limited",
                    UserWarning,
                    stacklevel=2,
                )
            elif not (is_dict_registry or has_env_specs_attr):
                warnings.warn(
                    "Gymnasium registry may not be fully compatible - some features may be limited",
                    UserWarning,
                    stacklevel=2,
                )

            _module_logger.debug(
                "Gymnasium registry access validated",
                extra={
                    "registry_type": (
                        type(registry).__name__ if registry is not None else "None"
                    ),
                    "dict_registry": is_dict_registry,
                    "has_env_specs": has_env_specs_attr,
                },
            )

        except Exception as e:
            _module_logger.warning(
                f"Registry validation encountered issues: {e}",
                extra={"validation_warning": str(e)},
            )

        initialization_end = time.time()
        _initialization_timestamp = initialization_end

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

        _module_initialized = True

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

        return True

    except ConfigurationError:
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
        error_msg = f"Module initialization failed unexpectedly: {e}"

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
            pass

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


try:
    if not _module_initialized:
        _initialize_registration_module()
except Exception as e:
    _module_logger.error(
        f"Automatic module initialization failed: {e}",
        extra={"auto_initialization_failed": True, "error_type": type(e).__name__},
    )

    with _cache_lock:
        _registration_cache.update(
            {
                "auto_initialization_failed": True,
                "auto_initialization_error": str(e),
                "error_timestamp": time.time(),
            }
        )

    warnings.warn(
        f"Registration module initialization failed: {e}. "
        f"Call _initialize_registration_module() manually or use ensure_registered()",
        UserWarning,
        stacklevel=2,
    )
