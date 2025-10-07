"""
Package initialization file for plume_nav_sim logging infrastructure providing
centralized access to logging configuration, formatters, handlers, and loggers.
Exposes comprehensive logging capabilities including security-aware formatting,
performance monitoring, component-specific loggers, and development-focused
debugging tools for the proof-of-life reinforcement learning environment implementation.

This module serves as the unified entry point for the entire logging system, providing
factory functions, configuration management, and resource cleanup capabilities for
robust production-ready logging throughout the plume_nav_sim system.
"""

import atexit  # >=3.10 - Cleanup registration for graceful shutdown
import datetime  # >=3.10 - Timestamp management for package lifecycle tracking

# Standard library imports with version comments
import logging  # >=3.10 - Base logging functionality and system integration
import threading  # >=3.10 - Thread synchronization for concurrent operations
from typing import (  # >=3.10 - Type hints for package interface functions and logger factory methods
    Any,
    Dict,
    Optional,
    Union,
)

# Internal imports - System constants for performance and configuration
from ..core.constants import COMPONENT_NAMES as CORE_COMPONENT_NAMES
from ..core.constants import LOG_LEVEL_DEFAULT as CORE_LOG_LEVEL_DEFAULT
from ..core.constants import (
    PERFORMANCE_TARGET_STEP_LATENCY_MS as CORE_PERFORMANCE_TARGET,
)

# Internal imports - Configuration and enumeration infrastructure
from .config import (
    COMPONENT_NAMES,
    DEFAULT_LOGGING_CONFIG,
    LOG_LEVEL_DEFAULT,
    LOGGER_NAME_PREFIX,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
)
from .config import (
    ComponentLogger as ConfigComponentLogger,  # Core configuration classes and enums; Configuration and setup functions; Logger factory and management functions; Security and filtering infrastructure; Factory functions and utilities; Validation and configuration management; Configuration constants and defaults
)
from .config import (
    ComponentType,
    LoggerFactory,
    LoggingConfig,
    LogLevel,
    SensitiveInfoFilter,
    configure_development_logging,
    configure_logging,
    create_component_logger,
)
from .config import get_logger as config_get_logger
from .config import (
    get_logging_status,
    reset_logging_config,
    setup_performance_logging,
    validate_logging_config,
)

# Internal imports - Formatting infrastructure with security filtering
from .formatters import (  # Core formatter classes; Security filtering and sanitization; Color support and scheme management; Utility functions for formatter configuration; Formatting constants and color specifications
    CONSOLE_COLOR_CODES,
    DEFAULT_LOG_FORMAT,
    PERFORMANCE_LOG_FORMAT,
    SENSITIVE_REGEX_PATTERNS,
    ColorScheme,
    ConsoleFormatter,
    LogFormatter,
    PerformanceFormatter,
    SecurityFilter,
    detect_color_support,
    sanitize_message,
)

# Internal imports - Logger classes and management system (no __all__ in loggers.py)
from .loggers import (  # Core logger classes for component and performance monitoring; Primary logger factory functions; System configuration and lifecycle management; Handler creation factory functions; Cleanup and resource management functions
    ComponentLogger,
    LoggerManager,
    PerformanceLogger,
    configure_logging_system,
    create_console_handler,
    create_file_handler,
    create_performance_handler,
    ensure_logging_initialized,
    get_component_logger,
    get_logger,
    get_logging_statistics,
    get_performance_logger,
    register_cleanup_handlers,
    shutdown_logging_system,
)

# Package version and identification constants
PACKAGE_VERSION = "0.0.1"
DEFAULT_LOG_LEVEL = LogLevel.INFO
DEVELOPMENT_LOG_LEVEL = LogLevel.DEBUG

# Package initialization state management with thread safety
_package_initialized = False
_initialization_lock = threading.Lock()
_package_init_time: Optional[datetime.datetime] = None

# Default logging configuration cache and registry management
_default_logging_config: Optional[LoggingConfig] = None
_component_loggers_registry: Dict[str, ComponentLogger] = {}
_performance_loggers_registry: Dict[str, PerformanceLogger] = {}

# Package statistics and monitoring for health tracking
_package_stats = {
    "initialization_count": 0,
    "component_loggers_created": 0,
    "performance_loggers_created": 0,
    "configuration_resets": 0,
    "cleanup_operations": 0,
}


# Internal helper functions to reduce complexity of public APIs
def _flush_component_handlers() -> int:
    flushed_handlers = 0
    for logger in _component_loggers_registry.values():
        if hasattr(logger, "base_logger"):
            for handler in logger.base_logger.handlers:
                try:
                    handler.flush()
                    flushed_handlers += 1
                except Exception as e:
                    logging.warning(f"Error flushing handler: {e}")
    return flushed_handlers


def _close_root_handlers() -> int:
    closed_handlers = 0
    for handler in logging.root.handlers[:]:
        try:
            handler.close()
            logging.root.removeHandler(handler)
            closed_handlers += 1
        except Exception as e:
            logging.warning(f"Error closing handler: {e}")
    return closed_handlers


def _collect_performance_reports() -> Dict[str, Any]:
    performance_reports: Dict[str, Any] = {}
    for name, perf_logger in _performance_loggers_registry.items():
        try:
            performance_reports[name] = perf_logger.get_performance_report(
                "summary", False
            )
        except Exception as e:
            logging.warning(f"Error getting performance report for {name}: {e}")
    return performance_reports


def _build_logger_breakdown() -> Dict[str, Any]:
    component_types: Dict[str, int] = {}
    for key in _component_loggers_registry.keys():
        comp_type = key.split(":")[0] if ":" in key else "unknown"
        component_types[comp_type] = component_types.get(comp_type, 0) + 1
    return {
        "by_component_type": component_types,
        "performance_operations": list(_performance_loggers_registry.keys()),
    }


def _build_configuration_summary() -> Dict[str, Any]:
    return {
        "log_level": _default_logging_config.log_level.name,
        "console_output": getattr(
            _default_logging_config, "enable_console_output", True
        ),
        "file_output": getattr(_default_logging_config, "enable_file_output", False),
        "color_output": getattr(_default_logging_config, "enable_color_output", False),
        "security_filtering": getattr(
            _default_logging_config, "security_filtering_enabled", True
        ),
    }


def _collect_logger_details() -> Dict[str, Any]:
    details: Dict[str, Any] = {}
    for key, logger in _component_loggers_registry.items():
        try:
            details[key] = {
                "component_name": getattr(logger, "component_name", "unknown"),
                "log_level": (
                    logger.configured_log_level.name
                    if hasattr(logger, "configured_log_level")
                    else "unknown"
                ),
                "creation_time": (
                    logger.creation_time.isoformat()
                    if hasattr(logger, "creation_time")
                    else "unknown"
                ),
                "message_count": getattr(logger, "message_count", 0),
                "performance_tracking": getattr(
                    logger, "performance_tracking_enabled", False
                ),
            }
        except Exception as e:
            details[key] = {"error": str(e)}
    return details


def _collect_performance_statistics() -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for name, perf_logger in _performance_loggers_registry.items():
        try:
            stats[name] = {
                "measurement_count": getattr(perf_logger, "measurement_count", 0),
                "average_timing_ms": getattr(perf_logger, "average_timing", 0.0),
                "threshold_ms": getattr(perf_logger, "timing_threshold_ms", 0.0),
                "memory_tracking": getattr(
                    perf_logger, "memory_tracking_enabled", False
                ),
            }
        except Exception as e:
            stats[name] = {"error": str(e)}
    return stats


def _fetch_system_health(
    include_performance_stats: bool, include_logger_details: bool
) -> Dict[str, Any]:
    try:
        system_stats = get_logging_statistics(
            include_performance_data=include_performance_stats,
            include_registry_details=include_logger_details,
        )
        return {
            "logging_system_active": system_stats.get("system_status", {}).get(
                "logging_initialized", False
            ),
            "resource_utilization": system_stats.get("resource_utilization", {}),
            "health_status": "healthy" if _package_initialized else "uninitialized",
        }
    except Exception as e:
        return {"error": str(e)}


def _create_base_config(use_development_config: bool) -> LoggingConfig:
    if use_development_config:
        return LoggingConfig(
            log_level=DEVELOPMENT_LOG_LEVEL,
            enable_console_output=True,
            enable_file_output=False,
            enable_color_output=True,
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        return LoggingConfig(
            log_level=DEFAULT_LOG_LEVEL,
            enable_console_output=True,
            enable_file_output=True,
            enable_color_output=False,
            log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def _determine_component_type(component_name: str) -> ComponentType:
    component_type = ComponentType.ENVIRONMENT
    for comp_type in ComponentType:
        if comp_type.name.lower() in component_name.lower():
            component_type = comp_type
            break
    return component_type


def _register_component_loggers() -> None:
    for component_name in CORE_COMPONENT_NAMES:
        try:
            component_type = _determine_component_type(component_name)
            logger = get_component_logger(component_type, component_name)
            registry_key = f"{component_type.name}:{component_name}"
            _component_loggers_registry[registry_key] = logger
            _package_stats["component_loggers_created"] += 1
        except Exception as e:
            logging.warning(
                f"Failed to create component logger for {component_name}: {e}"
            )


def _register_performance_loggers() -> None:
    for operation_name in ["step_execution", "rendering", "plume_calculation"]:
        try:
            perf_logger = get_performance_logger(
                operation_name=operation_name,
                timing_threshold_ms=CORE_PERFORMANCE_TARGET,
                enable_memory_tracking=True,
            )
            _performance_loggers_registry[operation_name] = perf_logger
            _package_stats["performance_loggers_created"] += 1
        except Exception as e:
            logging.warning(
                f"Failed to create performance logger for {operation_name}: {e}"
            )


def init_logging_package(
    use_development_config: bool = True,
    enable_performance_monitoring: bool = True,
    custom_config_overrides: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Initializes the plume_nav_sim logging package with default configuration,
    component loggers, and performance monitoring setup for system-wide
    logging infrastructure.

    Args:
        use_development_config: Enable development-friendly logging configuration
        enable_performance_monitoring: Enable performance tracking and monitoring
        custom_config_overrides: Dictionary of custom configuration overrides

    Returns:
        bool: True if package initialized successfully, False if initialization failed
    """
    global _package_initialized, _default_logging_config, _package_init_time

    with _initialization_lock:
        try:
            # Check if logging package already initialized to avoid duplicate initialization
            if _package_initialized:
                _package_stats["initialization_count"] += 1
                return True

            # Create default LoggingConfig with development or production settings based on parameters
            base_config = _create_base_config(use_development_config)

            # Apply custom configuration overrides if provided in custom_config_overrides
            if custom_config_overrides:
                for key, value in custom_config_overrides.items():
                    if hasattr(base_config, key):
                        setattr(base_config, key, value)

            # Initialize logging system using configure_logging_system with validated configuration
            success = configure_logging_system(
                config=base_config, force_reconfiguration=True, validate_config=True
            )

            if not success:
                return False

            # Set up component loggers registry for centralized logger management
            _register_component_loggers()

            # Initialize performance monitoring infrastructure if enable_performance_monitoring enabled
            if enable_performance_monitoring:
                _register_performance_loggers()

            # Configure security filtering and sensitive information protection
            # This is handled automatically by the formatters with SecurityFilter

            # Set package initialization flag and store default configuration
            _package_initialized = True
            _default_logging_config = base_config
            _package_init_time = datetime.datetime.now()
            _package_stats["initialization_count"] += 1

            # Register cleanup handlers for graceful shutdown
            register_cleanup_handlers()

            # Return initialization success status with error handling
            return True

        except Exception as e:
            logging.error(f"Failed to initialize logging package: {e}", exc_info=True)
            return False


def get_default_config(
    development_mode: bool = True,
    enable_console_colors: bool = True,
    enable_file_logging: bool = False,
) -> LoggingConfig:
    """
    Returns default logging configuration for plume_nav_sim with development-friendly
    settings, security filtering, and performance monitoring configuration.

    Args:
        development_mode: Enable development-friendly configuration settings
        enable_console_colors: Enable color support in console output
        enable_file_logging: Enable file-based logging with rotation

    Returns:
        LoggingConfig: Default logging configuration with specified settings and security filtering
    """
    # Create LoggingConfig with development or production log levels based on development_mode
    base_log_level = DEVELOPMENT_LOG_LEVEL if development_mode else DEFAULT_LOG_LEVEL

    # Configure console logging with color support if enable_console_colors and terminal supports colors
    console_colors_enabled = enable_console_colors and detect_color_support()

    # Create comprehensive logging configuration
    config = LoggingConfig(
        log_level=base_log_level,
        enable_console_output=True,
        enable_file_output=enable_file_logging,
        enable_color_output=console_colors_enabled,
        log_format=(
            DEFAULT_LOG_FORMAT
            if not development_mode
            else "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"
        ),
        component_log_levels={},
        security_filtering_enabled=True,
        performance_monitoring_enabled=development_mode,
    )

    # Set up file logging with rotation and compression if enable_file_logging enabled
    if enable_file_logging:
        config.file_handler_config = {
            "filename": f"{LOGGER_NAME_PREFIX}.log",
            "max_bytes": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5,
            "encoding": "utf-8",
        }

    # Configure component-specific log levels based on ComponentType defaults
    for component_type in ComponentType:
        if hasattr(component_type, "get_default_log_level"):
            config.component_log_levels[component_type] = (
                component_type.get_default_log_level()
            )

    # Add security filtering patterns for sensitive information protection
    config.security_patterns = SENSITIVE_REGEX_PATTERNS.copy()

    # Set up performance logging configuration with appropriate thresholds
    if development_mode:
        config.performance_config = {
            "step_threshold_ms": CORE_PERFORMANCE_TARGET,
            "render_threshold_ms": 50.0,
            "memory_tracking_enabled": True,
            "baseline_measurements": 10,
        }

    # Configure formatters for console, file, and performance logging
    config.formatter_configs = {
        "console": ConsoleFormatter if console_colors_enabled else LogFormatter,
        "file": LogFormatter,
        "performance": PerformanceFormatter,
    }

    # Return comprehensive default configuration ready for system use
    return config


def setup_quick_logging(
    log_level: LogLevel = LogLevel.INFO,
    enable_console: bool = True,
    enable_colors: bool = True,
) -> ComponentLogger:
    """
    Quick setup function for basic logging configuration with minimal parameters,
    providing convenient initialization for development and testing scenarios.

    Args:
        log_level: Logging level for quick setup configuration
        enable_console: Enable console output for quick logging setup
        enable_colors: Enable color support in console output

    Returns:
        ComponentLogger: Configured default logger ready for immediate use
    """
    # Initialize logging package if not already initialized using init_logging_package
    if not _package_initialized:
        init_success = init_logging_package(
            use_development_config=True,
            enable_performance_monitoring=False,
            custom_config_overrides={
                "log_level": log_level,
                "enable_console_output": enable_console,
                "enable_color_output": enable_colors and detect_color_support(),
            },
        )

        if not init_success:
            # Fallback to basic logging if initialization fails
            logging.basicConfig(
                level=log_level.get_numeric_level(),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

    # Create console handler with color support if enable_colors and terminal supports it
    if enable_console:
        formatter_class = (
            ConsoleFormatter
            if (enable_colors and detect_color_support())
            else LogFormatter
        )
        console_formatter = formatter_class()

        # Configure basic formatter with specified log level and development-friendly format
        console_handler = create_console_handler(console_formatter)
        console_handler.setLevel(log_level.get_numeric_level())

    # Create default component logger with quick configuration settings
    default_logger = get_logger(
        name="quick_setup",
        component_type=ComponentType.UTILS,
        log_level=log_level,
        enable_performance_tracking=False,
    )

    # Apply security filtering for sensitive information protection
    # This is automatically handled by the formatters

    # Return configured logger ready for immediate use in development or testing
    return default_logger


def cleanup_logging_package(
    shutdown_timeout: float = 10.0, force_cleanup: bool = False
) -> Dict[str, Any]:
    """
    Cleans up logging package resources including handlers, loggers, and registry
    management for graceful shutdown and resource release.

    Args:
        shutdown_timeout: Maximum time in seconds to wait for cleanup completion
        force_cleanup: Force immediate cleanup without graceful shutdown

    Returns:
        Dict[str, Any]: Dictionary containing cleanup results including resources freed and handlers closed
    """
    global _package_initialized, _default_logging_config

    cleanup_start_time = datetime.datetime.now()

    try:
        with _initialization_lock:
            # Flush all active loggers and handlers to ensure data integrity
            flushed_handlers = _flush_component_handlers()

            # Close file handlers and release file system resources
            closed_handlers = _close_root_handlers()

            # Shutdown performance monitoring and export remaining performance data
            performance_reports = _collect_performance_reports()

            # Clear component and performance logger registries
            initial_component_count = len(_component_loggers_registry)
            initial_performance_count = len(_performance_loggers_registry)

            _component_loggers_registry.clear()
            _performance_loggers_registry.clear()

            # Release handler resources and cleanup temporary files
            # This is handled by the individual handler close() methods

            # Reset package initialization flags and configuration
            _package_initialized = False
            _default_logging_config = None

            # Shutdown the logging system infrastructure
            system_shutdown_results = shutdown_logging_system(
                shutdown_timeout=shutdown_timeout / 2, force_cleanup=force_cleanup
            )

            # Update cleanup statistics
            _package_stats["cleanup_operations"] += 1

            # Generate cleanup report with resources freed and status information
            cleanup_duration = (
                datetime.datetime.now() - cleanup_start_time
            ).total_seconds()

            cleanup_results = {
                "cleanup_completed": True,
                "cleanup_duration_seconds": cleanup_duration,
                "resources_freed": {
                    "component_loggers_cleared": initial_component_count,
                    "performance_loggers_cleared": initial_performance_count,
                    "handlers_flushed": flushed_handlers,
                    "handlers_closed": closed_handlers,
                },
                "performance_reports": performance_reports,
                "system_shutdown": system_shutdown_results,
                "package_statistics": _package_stats.copy(),
                "cleanup_timestamp": cleanup_start_time.isoformat(),
            }

            # Return comprehensive cleanup results dictionary
            return cleanup_results

    except Exception as e:
        cleanup_duration = (
            datetime.datetime.now() - cleanup_start_time
        ).total_seconds()
        return {
            "cleanup_completed": False,
            "cleanup_error": str(e),
            "cleanup_duration_seconds": cleanup_duration,
            "partial_cleanup": True,
            "cleanup_timestamp": cleanup_start_time.isoformat(),
        }


def get_package_info(
    include_logger_details: bool = False, include_performance_stats: bool = False
) -> Dict[str, Any]:
    """
    Returns comprehensive package information including version, configuration status,
    active loggers, and system health for monitoring and debugging.

    Args:
        include_logger_details: Include detailed information about active loggers
        include_performance_stats: Include performance statistics from performance loggers

    Returns:
        Dict[str, Any]: Dictionary containing package information, configuration status, and system statistics
    """
    # Globals not required here since we only read module-level state

    # Collect package version and initialization status information
    package_info = {
        "package_version": PACKAGE_VERSION,
        "logger_name_prefix": LOGGER_NAME_PREFIX,
        "initialization_status": {
            "initialized": _package_initialized,
            "init_time": _package_init_time.isoformat() if _package_init_time else None,
            "uptime_seconds": (
                (datetime.datetime.now() - _package_init_time).total_seconds()
                if _package_init_time
                else 0
            ),
            "has_default_config": _default_logging_config is not None,
        },
        "registry_status": {
            "component_loggers_count": len(_component_loggers_registry),
            "performance_loggers_count": len(_performance_loggers_registry),
            "total_active_loggers": len(_component_loggers_registry)
            + len(_performance_loggers_registry),
        },
        "package_statistics": _package_stats.copy(),
    }

    # Include active logger counts and registry status
    if _component_loggers_registry:
        package_info["logger_breakdown"] = _build_logger_breakdown()

    # Add configuration details and handler status information
    if _default_logging_config:
        package_info["configuration_summary"] = _build_configuration_summary()

    # Include logger details with component types and levels if requested
    if include_logger_details:
        package_info["logger_details"] = _collect_logger_details()

    # Add performance statistics from performance loggers if requested
    if include_performance_stats:
        package_info["performance_statistics"] = _collect_performance_statistics()

    # Include system health indicators and resource utilization
    package_info["system_health"] = _fetch_system_health(
        include_performance_stats, include_logger_details
    )

    # Format comprehensive package information report
    package_info["report_generated"] = datetime.datetime.now().isoformat()

    # Return detailed package information dictionary
    return package_info


# Comprehensive exports for external access to logging infrastructure
__all__ = [
    # Enums and configuration classes
    "LogLevel",
    "ComponentType",
    "LoggingConfig",
    # Formatter classes and security infrastructure
    "LogFormatter",
    "ConsoleFormatter",
    "PerformanceFormatter",
    "SecurityFilter",
    # Handler creation functions (from loggers.py since handlers.py doesn't exist)
    "ConsoleHandler",
    "FileHandler",
    "PerformanceHandler",
    # Logger classes for component and performance monitoring
    "ComponentLogger",
    "PerformanceLogger",
    # Primary logger factory functions
    "get_logger",
    "get_component_logger",
    "get_performance_logger",
    # Configuration and system management functions
    "configure_logging",
    "configure_development_logging",
    "setup_performance_logging",
    "configure_logging_system",
    # Handler factory functions
    "create_console_handler",
    "create_file_handler",
    "create_performance_handler",
    # Package-level initialization and management functions
    "init_logging_package",
    "get_default_config",
    "setup_quick_logging",
    "cleanup_logging_package",
    "get_package_info",
]
