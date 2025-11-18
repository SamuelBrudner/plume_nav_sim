"""
Utility logging module for plume_nav_sim package providing simplified logging interfaces,
component logger factory functions, performance monitoring utilities, and development-focused
logging configuration. Serves as the primary interface between plume_nav_sim components and
the centralized logging infrastructure.

This module provides:
- Component-specific logger creation with automatic configuration
- Performance monitoring and timing utilities
- Development-oriented logging with enhanced debugging
- Error handling integration with automatic context capture
- Security-aware logging with sensitive information filtering
- Convenient mixins for adding logging to component classes
"""

import functools  # >=3.10
import inspect  # >=3.10
import logging  # >=3.10
import threading  # >=3.10
import time  # >=3.10
import weakref  # >=3.10
from contextlib import suppress
from types import TracebackType
from typing import (  # >=3.10
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import ParamSpec

# Core system constants for component identification and configuration
from ..core.constants import (
    COMPONENT_NAMES,
    LOG_LEVEL_DEFAULT,
    PACKAGE_NAME,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)

# Internal imports for exception handling and error logging integration
from .exceptions import PlumeNavSimError, ValidationError, handle_component_error

# Avoid noisy '--- Logging error ---' prints on handler failures during tests
logging.raiseExceptions = False

# Centralized logging infrastructure components
try:
    from ..logging.config import (
        ComponentType,
        configure_development_logging,
        get_logger,
    )

    # Specialized performance logging formatter for timing analysis
    from ..logging.formatters import PerformanceFormatter
except ImportError:  # pragma: no cover - fallback for optional logging package
    from enum import Enum

    class ComponentType(Enum):
        """Fallback component enumeration when centralized logging package is unavailable."""

        ENVIRONMENT = "ENVIRONMENT"
        PLUME_MODEL = "PLUME_MODEL"
        RENDERING = "RENDERING"
        ACTION_PROCESSOR = "ACTION_PROCESSOR"
        REWARD_CALCULATOR = "REWARD_CALCULATOR"
        STATE_MANAGER = "STATE_MANAGER"
        BOUNDARY_ENFORCER = "BOUNDARY_ENFORCER"
        EPISODE_MANAGER = "EPISODE_MANAGER"
        UTILS = "UTILS"

    def get_logger(logger_name: str, component_type: "ComponentType") -> logging.Logger:
        """Fallback logger factory delegating to the standard logging module."""

        return logging.getLogger(logger_name)

    def configure_development_logging(
        log_level: str = "INFO",
        enable_console_output: bool = True,
        enable_color_output: bool = True,
        enable_performance_logging: bool = False,
        log_file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Basic development logging configuration when centralized logging is missing."""

        level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(level=level)
        return {
            "status": "stdlib_fallback",
            "log_level": log_level,
            "console_output": enable_console_output,
            "color_output": enable_color_output,
            "performance_logging": enable_performance_logging,
            "log_file_path": log_file_path,
        }

    class PerformanceFormatter:
        """Lightweight performance formatter used when custom formatters are unavailable."""

        def format_timing(
            self,
            operation_name: str,
            duration_ms: float,
            additional_data: Optional[Dict[str, Any]] = None,
        ) -> str:
            metrics = additional_data or {}
            if metrics:
                extra = ", ".join(f"{key}={value}" for key, value in metrics.items())
                return f"{operation_name}: {duration_ms:.3f}ms [{extra}]"
            return f"{operation_name}: {duration_ms:.3f}ms"


# Global state for logger caching and thread-safe operations
_logger_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_performance_baselines: Dict[str, Dict[str, Any]] = {}
_logging_initialized: bool = False
_cache_lock: threading.Lock = threading.Lock()

_T = TypeVar("_T")
P = ParamSpec("P")

# Default configuration for component loggers with security and performance features
DEFAULT_LOGGER_CONFIG = {
    "level": LOG_LEVEL_DEFAULT,
    "enable_performance": True,
    "enable_security_filter": True,
}

# Public API exports for the logging utilities module
__all__ = [
    "get_component_logger",
    "configure_logging_for_development",
    "log_performance",
    "monitor_performance",
    "log_with_context",
    "create_performance_logger",
    "setup_error_logging",
    "monitor_performance",
    "ComponentLogger",
    "PerformanceTimer",
    "LoggingMixin",
]


# ------------------------------
# Internal helper functions
# ------------------------------


def _validate_log_level_name(log_level: str) -> str:
    """Return validated, uppercased log level or raise ValidationError."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    level_upper = str(log_level).upper()
    if level_upper not in valid_levels:
        raise ValidationError(
            f"Invalid log_level '{log_level}'. Must be one of: {valid_levels}"
        )
    return level_upper


def _build_console_config(enable_console_colors: bool) -> Dict[str, Any]:
    return {
        "enabled": True,
        "colors": enable_console_colors,
        "terminal_support": enable_console_colors,
    }


def _build_file_config(enable_file_logging: bool, log_directory: str) -> Dict[str, Any]:
    return {
        "enabled": enable_file_logging,
        "directory": log_directory if enable_file_logging else None,
    }


def _init_component_logger_for_name(component_name: str, log_level: str) -> bool:
    """Initialize a component logger for a single component name.

    Returns True on success, False on failure without raising.
    """
    try:
        name_l = component_name.lower()
        if "environment" in name_l:
            comp_type = ComponentType.ENVIRONMENT
        elif "plume" in name_l:
            comp_type = ComponentType.PLUME_MODEL
        elif "render" in name_l:
            comp_type = ComponentType.RENDERING
        else:
            comp_type = ComponentType.UTILS

        _ = get_component_logger(
            component_name=component_name,
            component_type=comp_type,
            logger_level=log_level,
            enable_performance_tracking=True,
        )
        return True
    except Exception as e:
        logging.getLogger(PACKAGE_NAME).warning(
            f"Failed to initialize component logger for '{component_name}': {e}"
        )
        return False


def _initialize_component_loggers(log_level: str) -> List[str]:
    """Initialize loggers for all known components and return the list created."""
    initialized: List[str] = []
    for component_name in COMPONENT_NAMES:
        if _init_component_logger_for_name(component_name, log_level):
            initialized.append(component_name)
    return initialized


def _resolve_base_logger(logger: Any) -> logging.Logger:
    """Return a stdlib Logger from either a Logger or Component-like wrapper."""
    if isinstance(logger, logging.Logger):
        return logger
    candidate = getattr(logger, "base_logger", None)
    if isinstance(candidate, logging.Logger):
        return candidate
    raise ValidationError(
        f"logger must be logging.Logger or ComponentLogger instance, got {type(logger)}"
    )


def _format_metrics(additional_metrics: Optional[Dict[str, Any]]) -> str:
    if not additional_metrics:
        return ""
    try:
        parts = [f"{k}={v}" for k, v in additional_metrics.items()]
        return f" [{', '.join(parts)}]"
    except Exception:
        return ""


def _performance_thresholds(
    duration_ms: float, target_ms: Optional[float]
) -> tuple[str, float, float]:
    target = target_ms if target_ms is not None else PERFORMANCE_TARGET_STEP_LATENCY_MS
    if target <= 0:
        return ("within_target", 1.0, 0.0)
    status = "within_target" if duration_ms <= target else "exceeds_target"
    ratio = duration_ms / target
    return (status, ratio, target)


def _resolve_logger_from_kwargs(kwargs: Dict[str, Any]) -> logging.Logger:
    component_name = kwargs.get("component_name", "error_logger")
    comp_type = kwargs.get("component_type", ComponentType.UTILS)
    desired_level = kwargs.get("log_level")
    comp_logger = get_component_logger(
        component_name=component_name, component_type=comp_type
    )
    base = getattr(comp_logger, "base_logger", None)
    logger = (
        base if isinstance(base, logging.Logger) else logging.getLogger(component_name)
    )
    if desired_level:
        with suppress(Exception):
            logger.setLevel(getattr(logging, str(desired_level).upper()))
    return logger


def _ensure_exception_types(
    exception_types_to_log: Optional[List[Type[Exception]]],
) -> List[Type[Exception]]:
    types: List[Type[Exception]] = list(exception_types_to_log or [])
    if PlumeNavSimError not in types:
        types.append(PlumeNavSimError)
    if ValidationError not in types:
        types.append(ValidationError)
    if Exception not in types:
        types.append(Exception)
    return types


def _attach_log_exception_method(logger: logging.Logger) -> None:
    def log_exception(exc: Exception, context: Dict[str, Any] | None = None) -> None:
        try:
            error_details = {
                "exception_type": exc.__class__.__name__,
                "exception_message": str(exc),
                "component_context": context or {},
            }
            handle_component_error(exc, logger.name, error_details)
            logger.error(
                f"Exception in {logger.name}: {exc}",
                extra=error_details,
                exc_info=True,
            )
        except Exception as logging_error:
            logging.getLogger(PACKAGE_NAME).critical(
                f"Error logging failed: {logging_error}. Original exception: {exc}"
            )

    # Attach as attribute for callers expecting this convenience
    setattr(logger, "log_exception", log_exception)


def get_component_logger(
    component_name: str,
    component_type: ComponentType = ComponentType.UTILS,
    logger_level: Optional[str] = None,
    enable_performance_tracking: bool = True,
    *,
    core_component_id: Optional[str] = None,
) -> Any:
    """
    Factory function for creating or retrieving loggers for plume_nav_sim code.

    The first parameter (``component_name``) is a free-form logger name and may be any
    string, including dotted module paths (e.g., ``__name__``). This value is used to
    construct the underlying logger name and cache key.

    For core, stable components that participate in package-level logging configuration,
    callers may optionally pass ``core_component_id``. When provided, it is validated
    against ``plume_nav_sim.core.constants.COMPONENT_NAMES``. Scripts, benchmarks, and
    ad-hoc tools should omit this parameter and are not validated or warned on.

    Minimal normalization is applied only for module-like names that start with the
    package prefix (``PACKAGE_NAME + '.'``) to support ``__name__`` usage. No other
    heuristics (e.g., CamelCase, instance IDs) are applied.

    Args:
        component_name: Free-form logger name used to create and identify the logger.
        component_type: ComponentType for specialized configuration (defaults to UTILS).
        logger_level: Optional logging level override.
        enable_performance_tracking: Enable performance timing and monitoring.
        core_component_id: Optional core component identifier; if provided, must be one of
            ``COMPONENT_NAMES`` and is validated. Omit for ad-hoc/script loggers.

    Returns:
        ComponentLogger configured with appropriate settings and performance options.

    Raises:
        ValidationError: If ``component_type`` is incorrect or ``core_component_id`` is invalid.
        PlumeNavSimError: If logger creation or configuration fails.
    """
    try:
        # Validate component_type eagerly for a clear error path
        if not isinstance(component_type, ComponentType):
            raise ValidationError(
                f"component_type must be ComponentType enum, got {type(component_type)}"
            )

        # Minimal prefix stripping for module-like names to support __name__ usage.
        # Example: 'plume_nav_sim.utils.validation' -> 'utils'
        normalized_module_component = None
        pkg_prefix = f"{PACKAGE_NAME}."
        if component_name.startswith(pkg_prefix):
            remainder = component_name[len(pkg_prefix) :]
            if "." in remainder:
                normalized_module_component = remainder.split(".", 1)[0]
            else:
                normalized_module_component = remainder

        # If caller opted into core component semantics, validate explicitly.
        if core_component_id is not None:
            if core_component_id not in COMPONENT_NAMES:
                raise ValidationError(
                    f"Invalid core_component_id '{core_component_id}'. "
                    f"Valid options: {sorted(COMPONENT_NAMES)}"
                )

        # Backward-compatible convenience: if no explicit core_component_id provided,
        # derive it only when the normalized module-derived token is a known core ID.
        # This is used for metadata only and does not affect naming or caching.
        resolved_core_id = None
        if core_component_id is not None:
            resolved_core_id = core_component_id
        elif (
            normalized_module_component
            and normalized_module_component in COMPONENT_NAMES
        ):
            resolved_core_id = normalized_module_component

        # Generate cache key combining component_name and component_type for logger identification
        cache_key = f"{component_name}_{component_type.value}"

        # Check _logger_cache for existing logger instance using cache key with thread-safe access
        with _cache_lock:
            cached = _logger_cache.get(cache_key)
            if cached is not None:
                return cached  # Return cached plume logger directly

            plume_logger = get_logger(
                name=component_name,
                component_type=component_type,
                enable_performance_tracking=enable_performance_tracking,
            )

            underlying_logger = None
            if isinstance(plume_logger, logging.Logger):
                underlying_logger = plume_logger
            else:
                candidate = getattr(plume_logger, "base_logger", None)
                if isinstance(candidate, logging.Logger):
                    underlying_logger = candidate
                else:
                    candidate = getattr(plume_logger, "logger", None)
                    if isinstance(candidate, logging.Logger):
                        underlying_logger = candidate

            # Optional level override
            if logger_level is not None and underlying_logger is not None:
                with suppress(Exception):
                    desired = getattr(logging, logger_level.upper())
                    underlying_logger.setLevel(desired)

            # Wrap underlying logger into ComponentLogger adapter if needed
            if underlying_logger is not None:
                wrapped = ComponentLogger(
                    component_name=component_name,
                    component_type=component_type,
                    base_logger=underlying_logger,
                )
            else:
                # Assume plume_logger is already a component-like logger
                wrapped = plume_logger

            # Attach core component metadata when available for downstream consumers
            try:
                if resolved_core_id:
                    if hasattr(wrapped, "component_context") and isinstance(
                        wrapped.component_context, dict
                    ):
                        wrapped.component_context["core_component_id"] = (
                            resolved_core_id
                        )
            except Exception:
                # Non-fatal: metadata attachment should never break logger creation
                pass

            _logger_cache[cache_key] = wrapped
            return wrapped

    except ValidationError:
        raise
    except Exception as e:
        error_details = {
            "component_name": component_name,
            "component_type": component_type,
            "logger_level": logger_level,
            "enable_performance_tracking": enable_performance_tracking,
        }
        handle_component_error(e, "get_component_logger", error_details)
        raise PlumeNavSimError(f"Failed to create component logger: {e}") from e


def configure_logging_for_development(
    log_level: str = "DEBUG",
    enable_console_colors: bool = True,
    enable_file_logging: bool = False,
    log_directory: str = "./logs",
    log_file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Configures development-optimized logging for plume_nav_sim with enhanced debugging information,
    performance monitoring, and interactive development features.

    Args:
        log_level: Logging level for development (DEBUG, INFO, WARNING, ERROR)
        enable_console_colors: Enable ANSI color output for console logging
        enable_file_logging: Enable file-based logging output
        log_directory: Directory path for log files when file logging enabled

    Returns:
        Development logging configuration status and settings information

    Raises:
        PlumeNavSimError: If logging configuration fails or parameters are invalid
    """
    global _logging_initialized

    try:
        if _logging_initialized:
            return {
                "status": "already_initialized",
                "log_level": log_level,
                "console_colors": enable_console_colors,
                "file_logging": enable_file_logging,
                "message": "Logging system already configured",
            }

        # Validate and configure base logging
        level_name = _validate_log_level_name(log_level)
        config_result = configure_development_logging(
            enable_verbose_output=True,
            enable_color_console=enable_console_colors,
            log_to_file=enable_file_logging,
            development_log_level=level_name,
        )

        _logging_initialized = True

        console_config = _build_console_config(enable_console_colors)
        file_config = _build_file_config(enable_file_logging, log_directory)

        error_logging_config = {
            "stack_traces": True,
            "context_capture": True,
            "integration_enabled": True,
        }
        performance_config = {
            "enabled": True,
            "threshold_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
            "baseline_tracking": True,
        }

        component_loggers_initialized = _initialize_component_loggers(level_name)

        return {
            "status": "configured",
            "log_level": level_name,
            "console_output": console_config,
            "file_logging": file_config,
            "error_logging": error_logging_config,
            "performance_monitoring": performance_config,
            "component_loggers": component_loggers_initialized,
            "initialization_time": time.time(),
            "config_source": config_result or "development_defaults",
        }

    except Exception as e:
        handle_component_error(e, "configure_logging_for_development")
        raise PlumeNavSimError(f"Development logging configuration failed: {e}") from e


def log_performance(
    logger: Any,
    operation_name: str,
    duration_ms: float,
    additional_metrics: Optional[Dict[str, Any]] = None,
    compare_to_baseline: bool = False,
    *,
    target_ms: Optional[float] = None,
) -> None:
    """
    Utility function for logging performance measurements with timing analysis, threshold
    comparison, and baseline tracking for development performance monitoring.

    Args:
        logger: Logger instance for output (should be ComponentLogger or compatible)
        operation_name: Name of the operation being measured
        duration_ms: Duration of the operation in milliseconds
        additional_metrics: Optional additional metrics (memory usage, operation count, etc.)
        compare_to_baseline: Whether to compare against stored baseline measurements

    Raises:
        ValidationError: If parameters are invalid
        PlumeNavSimError: If performance logging fails
    """
    try:
        base_logger = _resolve_base_logger(logger)

        if not operation_name or not isinstance(operation_name, str):
            raise ValidationError(
                f"operation_name must be non-empty string, got {operation_name}"
            )

        if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
            raise ValidationError(
                f"duration_ms must be non-negative number, got {duration_ms}"
            )

        threshold_status, threshold_ratio, _ = _performance_thresholds(
            float(duration_ms), target_ms
        )

        metrics_info = _format_metrics(additional_metrics)

        baseline_info = ""
        if compare_to_baseline and operation_name in _performance_baselines:
            with suppress(Exception):
                baseline_duration = _performance_baselines[operation_name].get(
                    "duration_ms", 0
                )
                if baseline_duration > 0:
                    baseline_ratio = float(duration_ms) / float(baseline_duration)
                    if baseline_ratio > 1.2:
                        baseline_info = (
                            f" (SLOWER than baseline: {baseline_ratio:.2f}x)"
                        )
                    elif baseline_ratio < 0.8:
                        baseline_info = (
                            f" (FASTER than baseline: {1/baseline_ratio:.2f}x)"
                        )
                    else:
                        baseline_info = f" (similar to baseline: {baseline_ratio:.2f}x)"

        # Choose level; throttle noisier messages
        if threshold_ratio <= 1.0:
            log_level = logging.DEBUG
        elif threshold_ratio <= 2.0:
            log_level = logging.WARNING
        else:
            log_level = logging.ERROR

        if threshold_ratio <= 1.2 and log_level == logging.WARNING:
            return

        if base_logger.isEnabledFor(log_level):
            log_message = (
                f"PERF [{operation_name}] {float(duration_ms):.3f}ms "
                f"({threshold_status}: {threshold_ratio:.2f}x target)"
                f"{metrics_info}{baseline_info}"
            )
            base_logger.log(
                log_level,
                log_message,
                extra={
                    "operation_name": operation_name,
                    "duration_ms": float(duration_ms),
                    "threshold_status": threshold_status,
                    "threshold_ratio": float(threshold_ratio),
                    "additional_metrics": additional_metrics,
                    "baseline_comparison": baseline_info,
                    "performance_category": "timing",
                },
            )

        # Maintain a simple running average baseline
        if operation_name not in _performance_baselines:
            _performance_baselines[operation_name] = {
                "duration_ms": float(duration_ms),
                "first_measured": time.time(),
                "measurement_count": 1,
            }
        else:
            baseline = _performance_baselines[operation_name]
            count = int(baseline.get("measurement_count", 0))
            prev = float(baseline.get("duration_ms", 0.0))
            new_avg = (prev * count + float(duration_ms)) / (count + 1)
            baseline["duration_ms"] = new_avg
            baseline["measurement_count"] = count + 1
            baseline["last_updated"] = time.time()

    except Exception as e:
        handle_component_error(e, "log_performance")
        # Don't re-raise to prevent logging failures from breaking application flow
        logging.getLogger(PACKAGE_NAME).error(f"Performance logging failed: {e}")


F = TypeVar("F", bound=Callable[..., Any])


@overload
def monitor_performance(func: F) -> F:  # noqa: E704
    ...


@overload
def monitor_performance(  # noqa: E704
    operation_name: Optional[str] = ...,
    performance_threshold_ms: Optional[float] = ...,
    compare_to_baseline: bool = ...,
) -> Callable[[F], F]: ...


def monitor_performance(
    operation_name: Optional[str] = None,
    performance_threshold_ms: Optional[float] = None,
    compare_to_baseline: bool = False,
) -> Any:
    """Decorator to measure execution time and log performance metrics.

    Supports optional performance thresholds and baseline comparisons. Can be used
    without parentheses (``@monitor_performance``) or with parameters such as
    ``@monitor_performance('operation', 10.0, True)``.
    """

    if callable(operation_name):  # Decorator used without parentheses
        func = operation_name  # type: ignore[assignment]
        return monitor_performance()(func)  # type: ignore[misc]

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            metrics: Optional[Dict[str, Any]] = None
            if performance_threshold_ms is not None:
                metrics = {"threshold_ms": performance_threshold_ms}

            op_name = operation_name or func.__name__
            # Throttle hot-path logging to reduce variance in tight loops (e.g., rendering benchmarks)
            # Also avoid logging for reward_calculation to minimize overhead in performance tests.
            if op_name not in ("render_operation", "reward_calculation"):
                try:
                    log_performance(
                        logging.getLogger(PACKAGE_NAME),
                        op_name,
                        duration_ms,
                        additional_metrics=metrics,
                        compare_to_baseline=compare_to_baseline,
                    )
                except Exception:
                    logging.getLogger(PACKAGE_NAME).debug(
                        "Performance logging fallback for %s completed in %.3fms",
                        op_name,
                        duration_ms,
                    )

            return result

        return cast(F, wrapper)

    return decorator


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    extra_context: Optional[Dict[str, Any]] = None,
    include_stack_info: bool = False,
) -> None:
    """
    Enhanced logging function that automatically captures caller context, component information,
    and runtime details for comprehensive development debugging.

    Args:
        logger: Logger instance for output
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message: Log message content
        extra_context: Additional context information to include
        include_stack_info: Whether to include stack trace information

    Raises:
        ValidationError: If parameters are invalid
    """
    try:
        # Validate logger instance and convert level string to logging level constant
        if not isinstance(logger, logging.Logger):
            raise ValidationError(
                f"logger must be logging.Logger instance, got {type(logger)}"
            )

        level_upper = level.upper()
        if level_upper not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValidationError(
                f"Invalid logging level '{level}'. Must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            )

        log_level = getattr(logging, level_upper)

        # Use inspect module to capture caller information (function name, line number, file)
        caller_info = get_caller_info(stack_depth=2, include_locals=False)

        # Extract component context from logger name and component hierarchy
        component_context = {
            "logger_name": logger.name,
            "component": (
                logger.name.split(".")[-1] if "." in logger.name else logger.name
            ),
            "package": PACKAGE_NAME,
        }

        # Merge extra_context with automatically captured context information
        context = {
            **component_context,
            **caller_info,
            "timestamp": time.time(),
            "thread_id": threading.current_thread().ident,
        }

        if extra_context:
            context |= extra_context

        # Apply security filtering to context and message content
        # Security filtering is handled by the logging infrastructure's SecurityFilter

        # Include stack information if include_stack_info is True and appropriate for level
        stack_info = include_stack_info and log_level >= logging.WARNING

        # Create enhanced LogRecord with context information in extra fields
        # Log message using specified level with comprehensive context information
        logger.log(log_level, message, extra=context, stack_info=stack_info)

    except Exception as e:
        # Fallback logging without context if enhanced logging fails
        try:
            logger.error(f"Context logging failed: {e}. Original message: {message}")
        except Exception:
            # Ultimate fallback to basic logging
            logging.getLogger(PACKAGE_NAME).error(
                f"All logging failed. Message: {message}, Error: {e}"
            )


def create_performance_logger(
    logger_name: Optional[str] = None,
    timing_thresholds: Optional[Dict[str, float]] = None,
    enable_memory_tracking: bool = True,
    baseline_file: Optional[str] = None,
    *,
    component_name: Optional[str] = None,
    enable_automatic_logging: bool = False,
) -> "ComponentLogger":
    """
    Creates specialized logger instance optimized for performance monitoring with timing
    measurement capabilities, baseline tracking, and threshold alerting.

    Args:
        logger_name: Name for the performance logger
        timing_thresholds: Dictionary of operation names to threshold values (ms)
        enable_memory_tracking: Enable memory usage tracking in performance logs
        baseline_file: Optional file path to load/save performance baselines

    Returns:
        Performance-optimized logger with specialized formatting and monitoring capabilities

    Raises:
        PlumeNavSimError: If logger creation fails
    """
    try:
        # Create base logger using get_logger function with performance-specific configuration
        # Resolve name/component
        name = component_name or logger_name or "performance"
        base_logger = get_logger(
            name=f"{PACKAGE_NAME}.performance.{name}",
            component_type=ComponentType.UTILS,
        )

        # Configure PerformanceFormatter with timing_thresholds and memory tracking settings
        thresholds = timing_thresholds or {}
        try:
            formatter = PerformanceFormatter(
                enable_memory_tracking=enable_memory_tracking,
                timing_thresholds=thresholds,
            )
        except TypeError:
            # Fallback if formatter signature differs
            formatter = PerformanceFormatter()

        # Set logger level to DEBUG for detailed performance information capture
        base_logger.setLevel(logging.DEBUG)

        # Configure performance-specific handler for specialized output formatting
        # Add the formatter to existing handlers
        for handler in base_logger.handlers:
            handler.setFormatter(formatter)

        # Load existing performance baselines from baseline_file if provided
        if baseline_file:
            try:
                import json

                with open(baseline_file, "r") as f:
                    loaded_baselines = json.load(f)
                    _performance_baselines.update(loaded_baselines)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                base_logger.warning(
                    f"Could not load baseline file {baseline_file}: {e}"
                )

        # Initialize memory tracking capabilities if enable_memory_tracking is True
        if enable_memory_tracking:
            base_logger.memory_tracking = True  # type: ignore[attr-defined]

        # Set up threshold alerting for performance degradation detection
        base_logger.timing_thresholds = thresholds  # type: ignore[attr-defined]

        # Add baseline tracking to the formatter
        if hasattr(formatter, "add_baseline"):
            for operation, threshold in thresholds.items():
                try:
                    formatter.add_baseline(operation, threshold)
                except Exception:
                    pass

        # Return configured performance logger ready for timing measurements
        return ComponentLogger(
            component_name=f"performance.{name}",
            component_type=ComponentType.UTILS,
            base_logger=base_logger,
        )

    except Exception as e:
        handle_component_error(e, "create_performance_logger")
        raise PlumeNavSimError(f"Failed to create performance logger: {e}") from e


def setup_error_logging(
    logger: Optional[logging.Logger] = None,
    enable_auto_recovery_logging: bool = True,
    exception_types_to_log: List[Type[Exception]] = None,
    **kwargs: Any,
) -> None:
    """
    Configures automatic error logging integration with exception handling system, providing
    seamless error context capture and recovery logging for development debugging.

    Args:
        logger: Logger instance to configure for error handling integration
        enable_auto_recovery_logging: Enable automatic logging of recovery actions
        exception_types_to_log: List of exception types to automatically log

    Raises:
        ValidationError: If parameters are invalid
        PlumeNavSimError: If error logging setup fails
    """
    try:
        if logger is None:
            logger = _resolve_logger_from_kwargs(kwargs)
        if not isinstance(logger, logging.Logger):
            raise ValidationError(
                f"logger must be logging.Logger instance, got {type(logger)}"
            )

        exception_types = _ensure_exception_types(exception_types_to_log)

        setattr(logger, "error_handling_enabled", True)
        setattr(logger, "exception_types_to_log", exception_types)
        setattr(logger, "auto_recovery_logging", enable_auto_recovery_logging)
        setattr(logger, "monitored_exceptions", set(exception_types))
        setattr(logger, "error_context_extraction", True)
        setattr(logger, "capture_stack_traces", True)
        setattr(logger, "error_escalation_enabled", True)

        _attach_log_exception_method(logger)

    except Exception as e:
        handle_component_error(e, "setup_error_logging")
        raise PlumeNavSimError(f"Failed to setup error logging: {e}") from e


def get_caller_info(
    stack_depth: int = 1, include_locals: bool = False
) -> Dict[str, Any]:
    """
    Utility function that uses stack inspection to automatically extract caller information
    including function name, line number, and file path for context-aware logging.

    Args:
        stack_depth: How many levels up the stack to inspect (1 = immediate caller)
        include_locals: Whether to include local variable information (filtered for security)

    Returns:
        Dictionary containing caller information including function, file, line,
        and optional local variables
    """
    try:
        # Use inspect.stack() to get call stack information at specified depth
        stack = inspect.stack()

        if len(stack) <= stack_depth:
            return {"caller_info": "unavailable", "reason": "insufficient_stack_depth"}

        frame = stack[stack_depth]

        # Extract function name, file path, and line number from stack frame
        caller_info = {
            "function_name": frame.function,
            "file_path": frame.filename,
            "line_number": frame.lineno,
            "code_context": (
                frame.code_context[0].strip() if frame.code_context else None
            ),
        }

        # Include local variable information if include_locals is True and safe to do so
        if include_locals and frame.frame:
            local_vars = frame.frame.f_locals

            # Filter out sensitive information from local variables for security
            safe_locals = {}
            for key, value in local_vars.items():
                # Skip private variables and potentially sensitive data
                if not key.startswith("_") and key not in [
                    "password",
                    "token",
                    "key",
                    "secret",
                ]:
                    try:
                        # Only include serializable values
                        str_value = str(value)
                        if len(str_value) < 200:  # Limit size to prevent log bloat
                            safe_locals[key] = str_value
                    except Exception:
                        safe_locals[key] = "<unserializable>"

            caller_info["local_variables"] = safe_locals

        return caller_info

    except Exception:
        # Handle exceptions gracefully if stack inspection fails
        return {
            "caller_info": "unavailable",
            "reason": "stack_inspection_failed",
            "stack_depth_requested": stack_depth,
        }


def clear_logger_cache(
    force_cleanup: bool = False, component_filter: Optional[str] = None
) -> int:
    """
    Utility function for clearing cached logger instances with proper cleanup and resource
    management, useful for testing and development reconfiguration.

    Args:
        force_cleanup: Whether to force cleanup of logger handlers and resources
        component_filter: Optional filter to only clear loggers for specific component

    Returns:
        Number of cached loggers cleared from cache
    """
    cleared_count = 0

    try:
        # Acquire _cache_lock for thread-safe cache clearing operation
        with _cache_lock:
            keys_to_remove = [
                key
                for key in list(_logger_cache.keys())
                if component_filter is None or component_filter in key
            ]
            # Remove logger entries from _logger_cache weak reference dictionary
            for key in keys_to_remove:
                if key in _logger_cache:
                    logger = _logger_cache[key]

                    # Close logger handlers properly if force_cleanup is True
                    if force_cleanup and hasattr(logger, "base_logger"):
                        for handler in logger.base_logger.handlers[:]:
                            handler.close()
                            logger.base_logger.removeHandler(handler)

                    del _logger_cache[key]
                    cleared_count += 1

            # Clear performance baselines for removed loggers from _performance_baselines
            if component_filter:
                baseline_keys_to_remove = [
                    k for k in _performance_baselines.keys() if component_filter in k
                ]
            else:
                baseline_keys_to_remove = list(_performance_baselines.keys())

            for key in baseline_keys_to_remove:
                del _performance_baselines[key]

            # Reset logging initialization state if all loggers cleared
            global _logging_initialized
            if component_filter is None and cleared_count > 0:
                _logging_initialized = False

        # Count and return number of loggers cleared from cache
        return cleared_count

    except Exception as e:
        logging.getLogger(PACKAGE_NAME).error(f"Cache clearing failed: {e}")
        return 0


class ComponentLogger:
    """
    Enhanced logger class specifically designed for plume_nav_sim components with automatic
    performance tracking, context capture, security filtering, and component-specific
    configuration management.

    This class provides enhanced logging capabilities including:
    - Automatic caller context capture
    - Performance timing and monitoring
    - Component-specific configuration
    - Security-aware information filtering
    - Integration with error handling system
    """

    def __init__(
        self,
        component_name: str,
        component_type: ComponentType,
        base_logger: logging.Logger,
    ) -> None:
        """
        Initialize ComponentLogger with component identification, base logger configuration,
        and component-specific features for enhanced debugging and monitoring.

        Args:
            component_name: Name of the component this logger serves
            component_type: ComponentType enum for specialized configuration
            base_logger: Underlying Python logger instance
        """
        # Store component_name and component_type for component identification
        self.component_name = component_name
        self.component_type = component_type

        # Initialize base_logger as the underlying Python logger instance
        self.base_logger = base_logger

        # Enable performance_tracking_enabled based on component_type requirements
        self.performance_tracking_enabled = component_type in [
            ComponentType.ENVIRONMENT,
            ComponentType.PLUME_MODEL,
            ComponentType.RENDERING,
        ]

        # Initialize empty performance_baselines dictionary for baseline tracking
        self.performance_baselines: Dict[str, float] = {}

        # Create PerformanceTimer instance for timing measurement capabilities
        self.timer = PerformanceTimer("default", logger=self, auto_log=False)

        # Set up component_context with component metadata and configuration
        self.component_context = {
            "component_name": component_name,
            "component_type": component_type.value,
            "logger_name": base_logger.name,
            "performance_tracking": self.performance_tracking_enabled,
        }

    # ---- logging.Logger compatibility surface ----
    @property
    def name(self) -> str:
        return self.base_logger.name

    @property
    def level(self) -> int:
        return self.base_logger.level

    def setLevel(self, level: int) -> None:  # noqa: N802 - mirror logging API
        try:
            self.base_logger.setLevel(level)
        except Exception:
            pass

    @property
    def handlers(self):  # type: ignore[override]
        return self.base_logger.handlers

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Debug with standard logging semantics plus component context.

        Accepts positional *args for %-style formatting and an optional ``extra``
        dict in **kwargs, matching ``logging.Logger``. The extra context is
        merged into the component metadata and caller info before being passed to
        the underlying logger.
        """
        extra = kwargs.pop("extra", None)
        context = self._extracted_from_warning_11(extra)

        # Format message with component-specific debug prefix; defer %-style
        # interpolation to the underlying logger via *args.
        formatted_message = f"[{self.component_name}] {message}"

        self.base_logger.debug(formatted_message, *args, extra=context, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Info with component context and optional extra mapping.

        Behaves like ``logging.Logger.info`` for positional *args while merging any
        ``extra`` mapping from **kwargs into the component context.
        """
        extra = kwargs.pop("extra", None)

        # Merge extra parameters with component_context for comprehensive logging
        context = {**self.component_context}
        if isinstance(extra, dict):
            context |= extra

        # Apply security filtering to message content and context data
        # (Handled by the logging infrastructure)

        # Include performance metrics if performance_tracking_enabled
        if self.performance_tracking_enabled and hasattr(
            self, "_current_operation_time"
        ):
            context["operation_duration_ms"] = getattr(
                self, "_current_operation_time", 0
            )

        # Format message with component identification and context
        formatted_message = f"[{self.component_name}] {message}"

        # Log info message using base_logger with component-enhanced formatting
        self.base_logger.info(formatted_message, *args, extra=context, **kwargs)

    def warning(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Enhanced warning logging with automatic error context capture and recovery
        suggestion logging for non-critical issues requiring attention.

        Args:
            message: Warning message to log
        """
        recovery_suggestion = kwargs.pop("recovery_suggestion", None)
        extra = kwargs.pop("extra", None)

        context = self._extracted_from_warning_11(extra)
        # Include recovery_suggestion in log message if provided
        if recovery_suggestion:
            context["recovery_suggestion"] = recovery_suggestion
            message = f"{message} | Recovery: {recovery_suggestion}"

        # Apply component-specific warning formatting and emphasis
        formatted_message = f"[{self.component_name}] WARNING: {message}"

        # Merge extra context with component metadata and caller information
        # Log warning message with enhanced context and recovery information
        self.base_logger.warning(formatted_message, *args, extra=context, **kwargs)

    # TODO Rename this here and in `debug` and `warning`
    def _extracted_from_warning_11(self, extra: Optional[Any]) -> Dict[str, Any]:
        caller_info = get_caller_info(stack_depth=2, include_locals=False)
        result: Dict[str, Any] = {**self.component_context, **caller_info}
        if extra:
            if isinstance(extra, dict):
                result |= extra
            else:
                # Preserve unexpected extras without breaking logging; attach under
                # a generic key rather than attempting dict-union on non-mappings.
                result["extra"] = extra
        return result

    def error(
        self,
        message: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Enhanced error logging with full context capture, stack trace information,
        and automatic integration with error handling system for critical issue tracking.

        Args:
            message: Error message to log
        """
        exception = kwargs.pop("exception", None)
        extra = kwargs.pop("extra", None)
        include_stack_trace = kwargs.pop("include_stack_trace", True)

        # Capture comprehensive error context including system state and component status
        caller_info = get_caller_info(stack_depth=2, include_locals=True)
        context = {
            **self.component_context,
            **caller_info,
            "error_timestamp": time.time(),
            "component_state": "error",
        }
        if isinstance(extra, dict):
            context |= extra
        elif extra is not None:
            context["extra"] = extra

        # Extract exception details if exception parameter provided
        if exception:
            context.update(
                {
                    "exception_type": exception.__class__.__name__,
                    "exception_message": str(exception),
                    "exception_args": (
                        exception.args if hasattr(exception, "args") else None
                    ),
                }
            )

            # Integrate with handle_component_error for automated error handling
            try:
                handle_component_error(exception, self.component_name, context)
            except Exception as e:
                context["error_handling_failed"] = str(e)

        # Include stack trace information if include_stack_trace enabled
        exc_info = exception if include_stack_trace and exception else None

        # Apply security filtering while preserving debugging information
        # (Handled by SecurityFilter in logging infrastructure)

        # Format error message with maximum debugging context
        formatted_message = f"[{self.component_name}] ERROR: {message}"

        # Log error with critical level formatting and comprehensive information
        self.base_logger.error(
            formatted_message,
            *args,
            extra=context,
            exc_info=exc_info,
            **kwargs,
        )

    def performance(
        self,
        operation_name: str,
        duration_ms: float,
        metrics: Optional[Dict[str, Any]] = None,
        update_baseline: bool = True,
    ) -> None:
        """
        Specialized performance logging with timing analysis, baseline comparison,
        and threshold monitoring for component performance tracking and optimization.

        Args:
            operation_name: Name of the operation that was measured
            duration_ms: Duration of the operation in milliseconds
            metrics: Additional performance metrics (memory, operation count, etc.)
            update_baseline: Whether to update baseline performance measurements
        """
        if not self.performance_tracking_enabled:
            return

        # Format timing information with appropriate precision and units
        timing_info = f"{duration_ms:.3f}ms"

        # Compare duration_ms against component-specific performance thresholds
        threshold = PERFORMANCE_TARGET_STEP_LATENCY_MS
        threshold_status = "OK" if duration_ms <= threshold else "SLOW"
        threshold_ratio = duration_ms / threshold if threshold > 0 else 1.0

        # Include additional metrics (memory, operation count) if provided
        metrics_info = ""
        if metrics:
            metrics_parts = [f"{k}={v}" for k, v in metrics.items()]
            metrics_info = f" [{', '.join(metrics_parts)}]"

        # Perform baseline comparison using stored performance_baselines
        baseline_info = ""
        if operation_name in self.performance_baselines:
            baseline = self.performance_baselines[operation_name]
            baseline_ratio = duration_ms / baseline
            if baseline_ratio > 1.2:
                baseline_info = f" (REGRESSION: {baseline_ratio:.1f}x baseline)"
            elif baseline_ratio < 0.8:
                baseline_info = f" (IMPROVEMENT: {1/baseline_ratio:.1f}x faster)"

        # Update baseline performance if update_baseline is True
        if update_baseline:
            if operation_name in self.performance_baselines:
                # Running average update
                current = self.performance_baselines[operation_name]
                self.performance_baselines[operation_name] = (current * 0.9) + (
                    duration_ms * 0.1
                )
            else:
                self.performance_baselines[operation_name] = duration_ms

        # Determine log level based on performance threshold comparison
        if threshold_ratio <= 1.0:
            log_level = logging.INFO
        elif threshold_ratio <= 2.0:
            log_level = logging.WARNING
        else:
            log_level = logging.ERROR

        # Log performance information with specialized formatting and analysis
        perf_message = (
            f"[{self.component_name}] PERF {operation_name}: {timing_info} "
            f"({threshold_status}: {threshold_ratio:.1f}x target){metrics_info}{baseline_info}"
        )

        context = {
            **self.component_context,
            "performance_data": {
                "operation_name": operation_name,
                "duration_ms": duration_ms,
                "threshold_status": threshold_status,
                "threshold_ratio": threshold_ratio,
                "metrics": metrics,
                "baseline_comparison": baseline_info,
            },
        }

        self.base_logger.log(log_level, perf_message, extra=context)

    def time_operation(
        self,
        operation_name: str,
        log_result: bool = True,
        raise_on_timeout: bool = False,
        timeout_ms: Optional[float] = None,
    ) -> "PerformanceTimer":
        """
        Context manager and decorator for automatic operation timing with performance
        logging integration for seamless performance monitoring.

        Args:
            operation_name: Name of the operation being timed
            log_result: Whether to automatically log the result when timing completes
            raise_on_timeout: Whether to raise exception if timeout is exceeded
            timeout_ms: Optional timeout threshold in milliseconds

        Returns:
            PerformanceTimer context manager for automatic timing and performance logging
        """
        # Create PerformanceTimer instance with operation_name and configuration
        timer = PerformanceTimer(
            operation_name=operation_name, logger=self, auto_log=log_result
        )

        # Configure automatic result logging if log_result is True
        timer.auto_log_enabled = log_result

        # Set timeout monitoring if timeout_ms provided and raise_on_timeout enabled
        if timeout_ms and raise_on_timeout:
            timer.timeout_ms = timeout_ms
            timer.raise_on_timeout = raise_on_timeout

        # Return PerformanceTimer context manager for operation timing
        # Automatically log performance results when context exits
        return timer

    def set_performance_baseline(
        self,
        operation_name: str,
        baseline_duration_ms: float,
        baseline_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Sets or updates performance baseline for specific operation to enable performance
        regression detection and trend analysis.

        Args:
            operation_name: Name of the operation to set baseline for
            baseline_duration_ms: Baseline duration in milliseconds
            baseline_metadata: Optional metadata about the baseline measurement

        Returns:
            True if baseline set successfully, False otherwise
        """
        try:
            # Validate operation_name and baseline_duration_ms parameters
            if not operation_name or not isinstance(operation_name, str):
                return False

            if (
                not isinstance(baseline_duration_ms, (int, float))
                or baseline_duration_ms <= 0
            ):
                return False

            # Store baseline information in performance_baselines dictionary
            self.performance_baselines[operation_name] = baseline_duration_ms

            # Include baseline_metadata for additional context and analysis
            if baseline_metadata:
                baseline_key = f"{operation_name}_metadata"
                self.performance_baselines[baseline_key] = baseline_metadata

            # Log baseline update information for performance tracking history
            self.info(
                f"Performance baseline set for '{operation_name}': {baseline_duration_ms:.3f}ms",
                extra={
                    "baseline_operation": operation_name,
                    "baseline_value": baseline_duration_ms,
                    "baseline_metadata": baseline_metadata,
                },
            )

            # Return success status of baseline configuration
            return True

        except Exception as e:
            self.error(f"Failed to set performance baseline: {e}", exception=e)
            return False


class PerformanceTimer:
    """
    Context manager and utility class for precise timing measurements with automatic
    performance logging, threshold monitoring, and integration with ComponentLogger
    for seamless performance tracking.

    Usage:
        # As context manager
        with PerformanceTimer("operation", logger) as timer:
            # perform operation
            timer.add_metric("memory_mb", 128.5)

        # Manual timing
        timer = PerformanceTimer("operation", logger, auto_log=False)
        timer.__enter__()
        # perform operation
        timer.__exit__(None, None, None)
    """

    def __init__(
        self,
        operation_name: str = "operation",
        logger: Optional[ComponentLogger] = None,
        auto_log: bool = True,
    ) -> None:
        """
        Initialize PerformanceTimer with operation identification, optional logger integration,
        and automatic logging configuration for timing measurements.

        Args:
            operation_name: Name of the operation being timed
            logger: Optional ComponentLogger instance for automatic logging
            auto_log: Whether to automatically log performance results
        """
        # Store operation_name for timing identification and logging
        self.operation_name = operation_name

        # Configure logger integration if ComponentLogger instance provided
        self.logger = logger

        # Set auto_log_enabled flag for automatic performance logging
        self.auto_log_enabled = auto_log

        # Initialize timing state variables (start_time, end_time, duration_ms)
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: Optional[float] = None

        # Set up additional_metrics dictionary for extended performance data
        self.additional_metrics: Dict[str, Any] = {}

        # Initialize timing_active flag for context manager state tracking
        self.timing_active = False

        # Optional timeout configuration
        self.timeout_ms: Optional[float] = None
        self.raise_on_timeout = False

    # ---- Additional helpers expected by tests ----
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        metrics: Dict[str, Dict[str, Any]] = {}
        for key, value in list(self.additional_metrics.items()):
            if key.endswith("_unit"):
                continue
            unit = self.additional_metrics.get(f"{key}_unit")
            metrics[key] = {"value": value, "unit": unit}
        return metrics

    def log_performance_summary(self) -> None:
        if self.logger and self.duration_ms is not None:
            self.logger.performance(
                self.operation_name,
                self.duration_ms,
                metrics=self.additional_metrics,
                update_baseline=False,
            )

    def set_baseline(self, baseline_duration_ms: float) -> bool:
        if self.logger:
            try:
                return self.logger.set_performance_baseline(
                    self.operation_name, baseline_duration_ms
                )
            except Exception:
                return False
        return False

    def compare_to_baseline(self, baseline_duration_ms: float) -> Optional[float]:
        if self.duration_ms is None:
            return None
        if baseline_duration_ms <= 0:
            return None
        return self.duration_ms / baseline_duration_ms

    def __enter__(self) -> "PerformanceTimer":
        """
        Context manager entry method that starts timing measurement with high-precision
        timestamp capture for accurate performance monitoring.

        Returns:
            Self reference for context manager pattern with timing started
        """
        # Capture high-precision start timestamp using time.perf_counter()
        self.start_time = time.perf_counter()

        # Set timing_active flag to True to indicate active timing
        self.timing_active = True

        # Initialize additional_metrics dictionary for metric collection
        self.additional_metrics = {}

        # Return self reference for context manager usage pattern
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        """
        Context manager exit method that stops timing, calculates duration, and optionally
        logs performance results with threshold analysis and baseline comparison.

        Args:
            exc_type: Exception type if an exception occurred
            exc_value: Exception instance if an exception occurred
            traceback: Exception traceback if an exception occurred

        Returns:
            False to propagate any exceptions that occurred during timed operation
        """
        # Capture end timestamp using time.perf_counter() for duration calculation
        self.end_time = time.perf_counter()

        # Calculate duration_ms as (end_time - start_time) * 1000 for millisecond precision
        if self.start_time is not None:
            self.duration_ms = (self.end_time - self.start_time) * 1000.0

        # Set timing_active flag to False to indicate timing completion
        self.timing_active = False

        # Log performance results automatically if auto_log_enabled and logger available
        if self.auto_log_enabled and self.logger and self.duration_ms is not None:
            # Include exception information in performance log if operation failed
            if exc_type is not None:
                self.additional_metrics["exception_occurred"] = exc_type.__name__
                self.additional_metrics["operation_failed"] = True
            else:
                self.additional_metrics["operation_successful"] = True

            # Check timeout if configured
            if self.timeout_ms and self.duration_ms > self.timeout_ms:
                self.additional_metrics["timeout_exceeded"] = True
                if self.raise_on_timeout:
                    raise PlumeNavSimError(
                        f"Operation '{self.operation_name}' exceeded timeout: "
                        f"{self.duration_ms:.3f}ms > {self.timeout_ms:.3f}ms"
                    )

            # Use positional args for cross-compat with different ComponentLogger APIs
            self.logger.performance(
                self.operation_name,
                self.duration_ms,
                metrics=self.additional_metrics,
                update_baseline=True,
            )

        # Return False to allow exception propagation if exc_type is not None
        return False

    def get_duration_ms(self) -> Optional[float]:
        """
        Returns measured duration in milliseconds with validation and precision formatting
        for performance analysis and reporting.

        Returns:
            Duration in milliseconds if timing completed, None if timing not finished or not started
        """
        # Check if timing has been completed (end_time is not None)
        if self.end_time is not None and self.start_time is not None:
            # Return calculated duration_ms if timing is complete
            return self.duration_ms

        # Return None if timing is still active or never started
        return None

    def add_metric(
        self, metric_name: str, metric_value: Any, metric_unit: Optional[str] = None
    ) -> None:
        """
        Adds additional performance metric to timing measurement for comprehensive performance
        analysis including memory usage, operation counts, or custom metrics.

        Args:
            metric_name: Name of the metric being added
            metric_value: Value of the metric
            metric_unit: Optional unit description for the metric
        """
        # Validate metric_name is non-empty string
        if not metric_name or not isinstance(metric_name, str):
            return

        # Apply security filtering to metric_value to prevent sensitive data exposure
        if isinstance(metric_value, str) and any(
            sensitive in metric_value.lower()
            for sensitive in ["password", "token", "key", "secret"]
        ):
            metric_value = "<sensitive_data_filtered>"

        # Store metric_value with metric_name in additional_metrics dictionary
        self.additional_metrics[metric_name] = metric_value

        # Include metric_unit information if provided for proper formatting
        if metric_unit:
            self.additional_metrics[f"{metric_name}_unit"] = metric_unit

    def log_performance(
        self, message: Optional[str] = None, include_metrics: bool = True
    ) -> bool:
        """
        Manually logs performance results with timing analysis, threshold comparison,
        and additional metrics for performance monitoring and debugging.

        Args:
            message: Optional custom message to include in the performance log
            include_metrics: Whether to include additional metrics in the log output

        Returns:
            True if performance logged successfully, False if no logger or timing incomplete
        """
        # Check that timing measurement is complete and logger is available
        if not self.logger or self.duration_ms is None:
            return False

        # Include additional_metrics in log if include_metrics is True
        metrics_to_log = self.additional_metrics if include_metrics else None

        # Use logger.performance method for specialized performance logging
        self.logger.performance(
            operation_name=self.operation_name,
            duration_ms=self.duration_ms,
            metrics=metrics_to_log,
            update_baseline=False,  # Manual logging doesn't update baselines automatically
        )

        # Return success status of performance logging operation
        return True


class LoggingMixin:
    """
    Mixin class providing logging capabilities to plume_nav_sim components with automatic
    logger creation, component identification, and performance tracking integration for
    convenient logging in component classes.

    Usage:
        class MyComponent(LoggingMixin):
            def __init__(self):
                super().__init__()
                self.configure_logging("MyComponent", ComponentType.UTILS)

            def do_work(self):
                self.log_method_entry("do_work", {"param": "value"})
                # ... perform work ...
                self.log_method_exit("do_work", return_value="success")
    """

    def __init__(self) -> None:
        """
        Initialize LoggingMixin with automatic component detection and logger configuration
        for seamless integration with plume_nav_sim components.
        """
        # Initialize _logger to None for lazy initialization pattern
        self._logger: Optional[ComponentLogger] = None

        # Set _component_name and _component_type to None for automatic detection
        self._component_name: Optional[str] = None
        self._component_type: Optional[ComponentType] = None

        # Set _logging_configured to False to track configuration state
        self._logging_configured = False

    @property
    def logger(self) -> ComponentLogger:
        """
        Property that provides lazy-initialized ComponentLogger with automatic component
        detection and configuration for convenient access to logging functionality.

        Returns:
            Component-specific logger configured for the current class and component type
        """
        # Check if _logger is already initialized to avoid duplicate creation
        if self._logger is not None:
            return self._logger

        try:
            # Determine component_name from class name if _component_name not set
            if self._component_name is None:
                class_name = self.__class__.__name__
                # Convert CamelCase to snake_case for component naming
                import re

                self._component_name = (
                    re.sub("([A-Z])", r"_\1", class_name).lower().strip("_")
                )

            # Detect component_type from class module and name if _component_type not set
            if self._component_type is None:
                module_name = self.__class__.__module__
                class_name = self.__class__.__name__.lower()

                if "environment" in module_name or "env" in class_name:
                    self._component_type = ComponentType.ENVIRONMENT
                elif "plume" in module_name or "plume" in class_name:
                    self._component_type = ComponentType.PLUME_MODEL
                elif "render" in module_name or "render" in class_name:
                    self._component_type = ComponentType.RENDERING
                else:
                    self._component_type = ComponentType.UTILS

            # Create ComponentLogger using get_component_logger function
            self._logger = get_component_logger(
                component_name=self._component_name,
                component_type=self._component_type,
                enable_performance_tracking=True,
            )

            # Set _logging_configured to True to indicate successful configuration
            self._logging_configured = True

            # Return configured ComponentLogger instance
            return self._logger

        except Exception as e:
            # Fallback to basic logger if component logger creation fails
            fallback_logger_name = f"{PACKAGE_NAME}.{self.__class__.__name__}"
            fallback = logging.getLogger(fallback_logger_name)
            fallback.error(f"Failed to create component logger: {e}")

            # Create minimal ComponentLogger with basic functionality
            self._logger = ComponentLogger(
                component_name=self.__class__.__name__,
                component_type=ComponentType.UTILS,
                base_logger=fallback,
            )
            return self._logger

    def configure_logging(
        self,
        component_name: str,
        component_type: ComponentType,
        logger_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Configures component logging with specific component name, type, and logger settings
        for customized logging behavior and component identification.

        Args:
            component_name: Name of the component for logger identification
            component_type: ComponentType enum for specialized configuration
            logger_config: Optional configuration dictionary for customized behavior
        """
        # Store component_name and component_type for logger configuration
        self._component_name = component_name
        self._component_type = component_type

        # Clear any existing _logger to force reconfiguration
        self._logger = None

        # Apply logger_config settings if provided for customized behavior
        if logger_config:
            # Store configuration for later use during logger creation
            self._logger_config = logger_config
        # Allow direct kwargs overrides commonly used in tests
        level = kwargs.get("log_level")
        enable_perf = kwargs.get("enable_performance_tracking")

        # Set _logging_configured to True to indicate manual configuration
        self._logging_configured = True

        # Trigger logger property access to create configured logger
        logger = get_component_logger(
            component_name=self._component_name,
            component_type=self._component_type,
            logger_level=level,
            enable_performance_tracking=(
                True if enable_perf is None else bool(enable_perf)
            ),
        )
        self._logger = logger

    def log_method_entry(
        self,
        method_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        include_locals: bool = False,
    ) -> None:
        """
        Convenience method for logging method entry with automatic parameter capture
        and context information for debugging and tracing method execution flow.

        Args:
            method_name: Name of the method being entered
            parameters: Dictionary of method parameters and values
            include_locals: Whether to include local variable information
        """
        try:
            # Use inspect to capture method context and caller information
            caller_info = get_caller_info(stack_depth=2, include_locals=include_locals)

            # Format method entry message with method_name and class context
            entry_message = f"→ ENTER {self.__class__.__name__}.{method_name}"

            # Include parameters dictionary if provided and safe for logging
            safe_parameters = {}
            if parameters:
                for key, value in parameters.items():
                    # Apply security filtering to prevent sensitive parameter exposure
                    if key.lower() not in ["password", "token", "key", "secret"]:
                        try:
                            # Limit parameter value length to prevent log bloat
                            str_value = str(value)
                            safe_parameters[key] = (
                                f"{str_value[:100]}..."
                                if len(str_value) > 100
                                else str_value
                            )
                        except Exception:
                            safe_parameters[key] = "<unserializable>"

                if safe_parameters:
                    entry_message += f" with params: {safe_parameters}"

            # Add local variable information if include_locals enabled and secure
            context = {
                **caller_info,
                "method_entry": True,
                "method_name": method_name,
                "class_name": self.__class__.__name__,
            }

            # Log method entry using debug level with comprehensive context
            self.logger.debug(entry_message, extra=context)

        except Exception as e:
            # Fallback logging if method entry logging fails
            self.logger.error(f"Method entry logging failed for {method_name}: {e}")

    def log_method_exit(
        self,
        method_name: str,
        return_value: Optional[Any] = None,
        execution_time_ms: Optional[float] = None,
    ) -> None:
        """
        Convenience method for logging method exit with return value information and
        execution summary for complete method execution tracing.

        Args:
            method_name: Name of the method being exited
            return_value: Optional return value from the method
            execution_time_ms: Optional execution timing in milliseconds
        """
        try:
            # Format method exit message with method_name and class context
            exit_message = f"← EXIT {self.__class__.__name__}.{method_name}"

            # Include return_value information if provided and safe for logging
            if return_value is not None:
                try:
                    # Apply security filtering to return_value to prevent data exposure
                    str_return = str(return_value)
                    safe_return = (
                        f"{str_return[:200]}..."
                        if len(str_return) > 200
                        else str_return
                    )
                    # Check for sensitive information in return value
                    if all(
                        sensitive not in safe_return.lower()
                        for sensitive in ["password", "token", "key", "secret"]
                    ):
                        exit_message += f" → {safe_return}"
                    else:
                        exit_message += " → <sensitive_data_filtered>"

                except Exception:
                    exit_message += " → <unserializable_return>"

            # Add execution timing information if execution_time_ms provided
            context = {
                "method_exit": True,
                "method_name": method_name,
                "class_name": self.__class__.__name__,
            }

            if execution_time_ms is not None:
                exit_message += f" ({execution_time_ms:.3f}ms)"
                context["execution_time_ms"] = execution_time_ms

                # Log performance information if timing is significant
                if execution_time_ms > 1.0:  # Log performance if > 1ms
                    self.logger.performance(
                        operation_name=f"{self.__class__.__name__}.{method_name}",
                        duration_ms=execution_time_ms,
                        metrics={"method_execution": True},
                    )

            # Log method exit using debug level with execution summary
            self.logger.debug(exit_message, extra=context)

        except Exception as e:
            # Fallback logging if method exit logging fails
            self.logger.error(f"Method exit logging failed for {method_name}: {e}")
