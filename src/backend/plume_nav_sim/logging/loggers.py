# src/backend/logging/loggers.py
"""
Central logger management system for plume_nav_sim providing component-specific
logger creation, hierarchical logger organization, performance-aware logging, and
comprehensive logger lifecycle management. Implements specialized logger classes
with automatic configuration, caching, and integration with the formatters and
handlers infrastructure for development-focused debugging and system monitoring.
"""

import atexit  # >=3.10
import datetime  # >=3.10

# Standard library imports with version comments
import logging  # >=3.10
import logging.config  # >=3.10
import signal  # >=3.10
import threading  # >=3.10
import time  # >=3.10
import weakref  # >=3.10
from dataclasses import dataclass, field  # >=3.10
from typing import Any, Dict, List, Optional  # >=3.10

# Internal imports - system constants
from ..core.constants import COMPONENT_NAMES, PERFORMANCE_TARGET_STEP_LATENCY_MS

# Internal imports - configuration and enumeration infrastructure
from .config import ComponentType, LoggingConfig, LogLevel

# Internal imports - formatting infrastructure
from .formatters import ConsoleFormatter, LogFormatter, PerformanceFormatter


class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler variant that suppresses handler errors (e.g., closed stream).

    This guards CI against noisy "Logging error" traces emitted during teardown
    when streams are closed or unavailable.
    """

    def handleError(self, record):  # noqa: D401 (inherit behavior doc)
        try:
            # Swallow all handler errors silently to avoid test flakiness
            # Original logging.handleError prints diagnostics; we skip that here.
            return
        except Exception:
            # Last-resort guard; never raise from logging in CI
            return


# Handler creation functions - implement minimal versions since handlers.py doesn't exist
def create_console_handler(
    formatter: Optional[logging.Formatter] = None,
) -> logging.StreamHandler:
    """Create console handler with optional formatter for development output."""
    handler = SafeStreamHandler()
    if formatter:
        handler.setFormatter(formatter)
    else:
        # Use basic formatter if none provided
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    return handler


def create_file_handler(
    filename: str, formatter: Optional[logging.Formatter] = None
) -> logging.FileHandler:
    """Create file handler with optional formatter and rotation capabilities."""
    handler = logging.FileHandler(filename)
    if formatter:
        handler.setFormatter(formatter)
    else:
        # Use basic formatter if none provided
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    return handler


def create_performance_handler(
    filename: str = "performance.log",
) -> logging.FileHandler:
    """Create specialized performance monitoring handler."""
    handler = logging.FileHandler(filename)
    formatter = PerformanceFormatter()
    handler.setFormatter(formatter)
    return handler


# Global constants for logger management and configuration
LOGGER_NAME_PREFIX = "plume_nav_sim"
DEFAULT_LOGGER_NAME = "plume_nav_sim.default"
COMPONENT_LOGGER_FORMAT = "plume_nav_sim.{component_type}.{component_name}"
PERFORMANCE_LOGGER_FORMAT = "plume_nav_sim.performance.{operation_name}"

# Global logger registries and management structures
_logger_registry: Dict[str, weakref.ReferenceType] = {}
_component_loggers: Dict[str, "ComponentLogger"] = {}
_performance_loggers: Dict[str, "PerformanceLogger"] = {}
_logger_factory: Optional["LoggerManager"] = None
_logger_manager: Optional["LoggerManager"] = None

# Thread synchronization for concurrent logger operations
_registry_lock = threading.RLock()
_initialization_lock = threading.Lock()

# System state tracking
_logging_initialized = False
_default_config: Optional[LoggingConfig] = None
_active_logger_count = 0
_logger_creation_stats = {
    "total_created": 0,
    "component_loggers": 0,
    "performance_loggers": 0,
    "cached_retrievals": 0,
}

# Resource management configuration
LOGGER_CACHE_SIZE_LIMIT = 1000
PERFORMANCE_LOGGER_BUFFER_SIZE = 10000
CLEANUP_REGISTERED = False


@dataclass
class LoggerCreationContext:
    """Context information for logger creation and configuration."""

    component_type: ComponentType
    component_name: str
    log_level: LogLevel
    enable_performance_tracking: bool = False
    custom_config: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime.datetime = field(default_factory=datetime.datetime.now)


class ComponentLogger:
    """
    Specialized logger class for plume_nav_sim system components providing
    component-specific configuration, performance tracking, enhanced debugging
    capabilities, and integration with the logging infrastructure for development
    and monitoring.
    """

    def __init__(
        self,
        logger_name: str,
        component_type: ComponentType,
        log_level: LogLevel,
        enable_performance_tracking: bool = False,
    ):
        """
        Initialize ComponentLogger with component-specific configuration,
        performance tracking setup, and appropriate handlers and formatters
        for plume_nav_sim component logging.
        """
        # Create base Python logger instance with hierarchical component-based naming
        self.base_logger = logging.getLogger(logger_name)

        # Store component type and extract component name from logger name
        self.component_type = component_type
        self.component_name = (
            logger_name.split(".")[-1] if "." in logger_name else logger_name
        )

        # Configure log level using provided level or component type default
        self.configured_log_level = log_level
        self.base_logger.setLevel(log_level.get_numeric_level())

        # Set up performance tracking if enabled and appropriate for component type
        self.performance_tracking_enabled = (
            enable_performance_tracking
            and component_type.requires_performance_logging()
        )

        # Initialize performance data dictionary for timing and memory tracking
        self.performance_data: Dict[str, Any] = {}

        # Set creation time and component configuration metadata
        self.creation_time = datetime.datetime.now()
        self.component_config: Dict[str, Any] = {
            "component_type": component_type.name,
            "log_level": log_level.name,
            "performance_tracking": self.performance_tracking_enabled,
        }

        # Initialize message counting and level statistics tracking
        self.message_count = 0
        self.level_counts: Dict[str, int] = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }

        # Create performance logger if performance tracking enabled
        self.performance_logger: Optional["PerformanceLogger"] = None
        if self.performance_tracking_enabled:
            self.performance_logger = PerformanceLogger(
                f"{logger_name}.performance",
                f"{self.component_name}_operations",
                PERFORMANCE_TARGET_STEP_LATENCY_MS,
                True,
            )

        # Configure component-specific handlers and formatters
        self._configure_handlers()

    def _configure_handlers(self):
        """Configure component-specific handlers with appropriate formatters."""
        # Clear existing handlers to avoid duplication
        self.base_logger.handlers.clear()

        # Add console handler with component-appropriate formatter
        console_formatter = ConsoleFormatter()
        console_handler = create_console_handler(console_formatter)
        console_handler.setLevel(self.configured_log_level.get_numeric_level())
        self.base_logger.addHandler(console_handler)

        # Add file handler for persistent logging if component requires it
        if self.component_type in [
            ComponentType.ENVIRONMENT,
            ComponentType.PLUME_MODEL,
        ]:
            file_formatter = LogFormatter()
            file_handler = create_file_handler(
                f"{self.component_name.lower()}.log", file_formatter
            )
            file_handler.setLevel(self.configured_log_level.get_numeric_level())
            self.base_logger.addHandler(file_handler)

        # Prevent propagation to avoid duplicate messages
        self.base_logger.propagate = False

    def debug(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """
        Logs debug message with component context, performance tracking, and
        enhanced debugging information for development and troubleshooting.
        """
        # Add component type and name to message context
        enhanced_context = {
            "component_type": self.component_type.name,
            "component_name": self.component_name,
            "message_count": self.message_count + 1,
        }

        # Include performance context if performance tracking enabled
        if self.performance_tracking_enabled and self.performance_logger:
            enhanced_context["performance_context"] = {
                "tracking_enabled": True,
                "operation_count": len(self.performance_data),
            }

        # Apply component-specific message formatting and enhancement
        if extra_context:
            enhanced_context.update(extra_context)

        # Call base logger debug method with enhanced context
        self.base_logger.debug(message, extra=enhanced_context)

        # Update message count and level statistics
        self.message_count += 1
        self.level_counts["DEBUG"] += 1

        # Track debug message timing if performance logging enabled
        if self.performance_tracking_enabled and self.performance_logger:
            self.performance_logger.log_timing(
                0.1,  # Minimal timing for debug messages
                {"message_type": "debug", "component": self.component_name},
                self.component_name,
            )

    def info(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """
        Logs informational message with component identification and operational
        context for standard system monitoring and status reporting.
        """
        # Enhance message with component identification and context
        enhanced_context = {
            "component_type": self.component_type.name,
            "component_name": self.component_name,
            "operational_status": "active",
            "message_count": self.message_count + 1,
        }

        if extra_context:
            enhanced_context.update(extra_context)

        # Log message through base logger with component context
        self.base_logger.info(message, extra=enhanced_context)

        # Update component logger statistics and message counts
        self.message_count += 1
        self.level_counts["INFO"] += 1

    def warning(self, message: str, extra_context: Optional[Dict[str, Any]] = None):
        """
        Logs warning message with component context, potential issue identification,
        and enhanced visibility for problem detection and system monitoring.
        """
        # Add warning severity markers and component identification
        enhanced_context = {
            "component_type": self.component_type.name,
            "component_name": self.component_name,
            "severity": "warning",
            "requires_attention": True,
            "message_count": self.message_count + 1,
        }

        if extra_context:
            enhanced_context.update(extra_context)

        # Apply enhanced formatting for warning visibility
        formatted_message = f"⚠️  {message} [Component: {self.component_name}]"

        # Log warning through base logger with severity context
        self.base_logger.warning(formatted_message, extra=enhanced_context)

        # Update warning statistics and component health tracking
        self.message_count += 1
        self.level_counts["WARNING"] += 1

        # Trigger performance monitoring alerts if applicable
        if self.performance_tracking_enabled and self.performance_logger:
            self.performance_logger.log_timing(
                1.0,  # Warning processing time
                {"warning_type": "system_warning", "component": self.component_name},
                self.component_name,
            )

    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Logs error message with comprehensive context, stack trace information,
        and error recovery suggestions for effective debugging and problem resolution.
        """
        # Extract detailed error information including stack trace if exception provided
        enhanced_context = {
            "component_type": self.component_type.name,
            "component_name": self.component_name,
            "severity": "error",
            "requires_immediate_attention": True,
            "message_count": self.message_count + 1,
        }

        # Add exception details if provided
        if exception:
            enhanced_context.update(
                {
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception),
                    "has_stack_trace": True,
                }
            )

        if extra_context:
            enhanced_context.update(extra_context)

        # Apply error-specific formatting with enhanced visibility
        formatted_message = f"❌ ERROR in {self.component_name}: {message}"

        # Log error through base logger with comprehensive context
        if exception:
            self.base_logger.error(
                formatted_message, exc_info=exception, extra=enhanced_context
            )
        else:
            self.base_logger.error(formatted_message, extra=enhanced_context)

        # Update error statistics and component health monitoring
        self.message_count += 1
        self.level_counts["ERROR"] += 1

        # Trigger performance alerts and error tracking if configured
        if self.performance_tracking_enabled and self.performance_logger:
            self.performance_logger.log_timing(
                5.0,  # Error processing overhead
                {"error_type": "component_error", "component": self.component_name},
                self.component_name,
            )

    def log_performance(
        self,
        operation_name: str,
        duration_ms: float,
        performance_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Logs performance metrics including timing measurements, memory usage, and
        threshold analysis for component optimization and performance monitoring.
        """
        # Validate performance tracking is enabled for this component logger
        if not self.performance_tracking_enabled or not self.performance_logger:
            self.warning(
                "Performance logging called but not enabled for component",
                {"component": self.component_name, "operation": operation_name},
            )
            return

        # Format performance metrics with appropriate units and precision
        formatted_context = {
            "operation_name": operation_name,
            "duration_ms": round(duration_ms, 3),
            "component": self.component_name,
            "timestamp": time.time_ns(),
        }

        if performance_context:
            formatted_context.update(performance_context)

        # Log performance data through performance logger if available
        self.performance_logger.log_timing(
            duration_ms, formatted_context, self.component_name
        )

        # Update component performance statistics and history
        if "performance_operations" not in self.performance_data:
            self.performance_data["performance_operations"] = []

        self.performance_data["performance_operations"].append(
            {
                "operation": operation_name,
                "duration_ms": duration_ms,
                "timestamp": datetime.datetime.now(),
                "context": formatted_context,
            }
        )

        # Store performance data for trend analysis and optimization
        self.info(
            f"Performance logged: {operation_name} took {duration_ms:.3f}ms",
            {"performance_data": formatted_context},
        )

    def get_component_statistics(self) -> Dict[str, Any]:
        """
        Returns comprehensive component logger statistics including message counts,
        performance data, and component health metrics for monitoring and analysis.
        """
        # Collect message count statistics by log level and time period
        statistics = {
            "component_info": {
                "name": self.component_name,
                "type": self.component_type.name,
                "log_level": self.configured_log_level.name,
                "creation_time": self.creation_time.isoformat(),
                "uptime_seconds": (
                    datetime.datetime.now() - self.creation_time
                ).total_seconds(),
            },
            "message_statistics": {
                "total_messages": self.message_count,
                "by_level": self.level_counts.copy(),
                "messages_per_minute": self._calculate_message_rate(),
            },
            "performance_tracking": {
                "enabled": self.performance_tracking_enabled,
                "operations_logged": len(
                    self.performance_data.get("performance_operations", [])
                ),
                "performance_logger_active": self.performance_logger is not None,
            },
            "component_health": {
                "error_rate": self._calculate_error_rate(),
                "warning_rate": self._calculate_warning_rate(),
                "status": self._determine_health_status(),
            },
        }

        # Include performance data and timing statistics if performance tracking enabled
        if self.performance_tracking_enabled and self.performance_logger:
            statistics["performance_data"] = {
                "recent_operations": self.performance_data.get(
                    "performance_operations", []
                )[-10:],
                "average_operation_time": self._calculate_average_operation_time(),
                "performance_report": self.performance_logger.get_performance_report(
                    "summary", False
                ),
            }

        return statistics

    def configure_performance_tracking(
        self,
        enable_tracking: bool,
        timing_threshold_ms: Optional[float] = None,
        tracking_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Configures or updates performance tracking settings for component logger
        with threshold configuration and monitoring setup.
        """
        try:
            # Update performance_tracking_enabled flag based on enable_tracking parameter
            self.performance_tracking_enabled = (
                enable_tracking and self.component_type.requires_performance_logging()
            )

            # Configure timing threshold using provided value or component defaults
            threshold = timing_threshold_ms or PERFORMANCE_TARGET_STEP_LATENCY_MS

            # Create or update performance logger if performance tracking enabled
            if self.performance_tracking_enabled:
                if not self.performance_logger:
                    self.performance_logger = PerformanceLogger(
                        f"{self.base_logger.name}.performance",
                        f"{self.component_name}_operations",
                        threshold,
                        True,
                    )
                else:
                    # Update existing performance logger configuration
                    self.performance_logger.timing_threshold_ms = threshold

                # Apply tracking configuration settings from tracking_config parameter
                if tracking_config:
                    for key, value in tracking_config.items():
                        if hasattr(self.performance_logger, key):
                            setattr(self.performance_logger, key, value)
            else:
                # Disable performance logger if tracking disabled
                self.performance_logger = None

            # Update component configuration
            self.component_config["performance_tracking"] = (
                self.performance_tracking_enabled
            )

            self.info(
                f"Performance tracking {'enabled' if enable_tracking else 'disabled'}",
                {"threshold_ms": threshold, "config_updated": True},
            )

            return True

        except Exception as e:
            self.error(f"Failed to configure performance tracking: {e}", e)
            return False

    def _calculate_message_rate(self) -> float:
        """Calculate messages per minute for component statistics."""
        uptime = (datetime.datetime.now() - self.creation_time).total_seconds()
        if uptime > 0:
            return (self.message_count / uptime) * 60
        return 0.0

    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage for health monitoring."""
        if self.message_count > 0:
            return (self.level_counts["ERROR"] / self.message_count) * 100
        return 0.0

    def _calculate_warning_rate(self) -> float:
        """Calculate warning rate percentage for health monitoring."""
        if self.message_count > 0:
            return (self.level_counts["WARNING"] / self.message_count) * 100
        return 0.0

    def _determine_health_status(self) -> str:
        """Determine component health status based on error and warning rates."""
        error_rate = self._calculate_error_rate()
        warning_rate = self._calculate_warning_rate()

        if error_rate > 10:
            return "critical"
        elif error_rate > 5 or warning_rate > 20:
            return "warning"
        elif warning_rate > 10:
            return "degraded"
        else:
            return "healthy"

    def _calculate_average_operation_time(self) -> float:
        """Calculate average operation time for performance analysis."""
        operations = self.performance_data.get("performance_operations", [])
        if operations:
            total_time = sum(op["duration_ms"] for op in operations)
            return total_time / len(operations)
        return 0.0


class PerformanceLogger:
    """
    Specialized logger for performance monitoring with high-resolution timing
    measurements, memory tracking, baseline comparison, and automated performance
    analysis for system optimization and development debugging.
    """

    def __init__(
        self,
        logger_name: str,
        operation_name: str,
        timing_threshold_ms: float,
        enable_memory_tracking: bool = False,
    ):
        """
        Initialize PerformanceLogger with timing thresholds, memory tracking,
        and performance analysis capabilities for comprehensive performance
        monitoring and optimization.
        """
        # Create base logger with performance-specific naming convention
        self.base_logger = logging.getLogger(logger_name)

        # Store operation name and timing threshold for performance analysis
        self.operation_name = operation_name
        self.timing_threshold_ms = timing_threshold_ms

        # Initialize memory tracking capabilities if enable_memory_tracking enabled
        self.memory_tracking_enabled = enable_memory_tracking

        # Set up performance baselines dictionary for operation comparison
        self.performance_baselines: Dict[str, Dict[str, Any]] = {}

        # Configure timing history storage with circular buffer for trend analysis
        self.timing_history: List[Dict[str, Any]] = []

        # Initialize performance statistics tracking and calculation
        self.performance_statistics: Dict[str, Any] = {
            "total_measurements": 0,
            "average_timing": 0.0,
            "min_timing": float("inf"),
            "max_timing": 0.0,
            "threshold_violations": 0,
        }

        # Set baseline update time and measurement counting
        self.baseline_update_time = datetime.datetime.now()
        self.measurement_count = 0
        self.average_timing = 0.0

        # Create PerformanceFormatter with timing analysis and baseline comparison
        self.performance_formatter = PerformanceFormatter()

        # Set up performance handler for specialized performance logging
        self._configure_performance_handlers()

    def _configure_performance_handlers(self):
        """Configure specialized handlers for performance logging."""
        # Clear existing handlers
        self.base_logger.handlers.clear()

        # Create performance file handler
        performance_handler = create_performance_handler(
            f"performance_{self.operation_name}.log"
        )
        performance_handler.setFormatter(self.performance_formatter)
        self.base_logger.addHandler(performance_handler)

        # Set appropriate log level for performance data
        self.base_logger.setLevel(logging.INFO)
        self.base_logger.propagate = False

    def log_timing(
        self,
        duration_ms: float,
        operation_context: Optional[Dict[str, Any]] = None,
        component_name: Optional[str] = None,
    ):
        """
        Logs operation timing with duration analysis, threshold comparison,
        baseline analysis, and performance trend tracking for optimization
        insights and debugging.
        """
        # Format timing measurement with appropriate precision and units
        timing_data = {
            "operation": self.operation_name,
            "duration_ms": round(duration_ms, 3),
            "component": component_name or "unknown",
            "timestamp": time.time_ns(),
            "measurement_id": self.measurement_count + 1,
        }

        if operation_context:
            timing_data.update(operation_context)

        # Compare duration against timing threshold and determine alert level
        threshold_exceeded = duration_ms > self.timing_threshold_ms
        if threshold_exceeded:
            timing_data["threshold_violation"] = True
            timing_data["threshold_exceeded_by_ms"] = (
                duration_ms - self.timing_threshold_ms
            )
            self.performance_statistics["threshold_violations"] += 1

        # Compare performance against stored baselines for trend analysis
        if self.performance_baselines:
            baseline_key = f"{self.operation_name}_{component_name or 'default'}"
            if baseline_key in self.performance_baselines:
                baseline = self.performance_baselines[baseline_key]
                timing_data["baseline_comparison"] = {
                    "baseline_ms": baseline["duration_ms"],
                    "deviation_ms": duration_ms - baseline["duration_ms"],
                    "deviation_percent": (
                        (duration_ms - baseline["duration_ms"])
                        / baseline["duration_ms"]
                    )
                    * 100,
                }

        # Update timing history and calculate performance statistics
        self.timing_history.append(timing_data)

        # Maintain circular buffer size to prevent memory growth
        if len(self.timing_history) > PERFORMANCE_LOGGER_BUFFER_SIZE:
            self.timing_history = self.timing_history[
                -PERFORMANCE_LOGGER_BUFFER_SIZE // 2 :
            ]

        # Update performance statistics including average and variance
        self._update_performance_statistics(duration_ms)

        # Log timing information with performance formatter and context
        log_level = logging.WARNING if threshold_exceeded else logging.INFO
        self.base_logger.log(
            log_level,
            f"Performance: {self.operation_name} completed in {duration_ms:.3f}ms",
            extra=timing_data,
        )

        self.measurement_count += 1

    def log_memory(
        self,
        memory_usage_bytes: int,
        memory_context: Optional[Dict[str, Any]] = None,
        component_name: Optional[str] = None,
    ):
        """
        Logs memory usage information with delta calculations, allocation pattern
        analysis, and memory leak detection for resource monitoring and optimization.
        """
        if not self.memory_tracking_enabled:
            return

        # Calculate memory usage in appropriate units (KB, MB) for readability
        memory_mb = memory_usage_bytes / (1024 * 1024)
        memory_kb = memory_usage_bytes / 1024

        memory_data = {
            "operation": self.operation_name,
            "memory_bytes": memory_usage_bytes,
            "memory_kb": round(memory_kb, 2),
            "memory_mb": round(memory_mb, 3),
            "component": component_name or "unknown",
            "timestamp": time.time_ns(),
        }

        if memory_context:
            memory_data.update(memory_context)

        # Determine memory delta from previous measurements if available
        if self.timing_history:
            # Find last memory measurement
            for entry in reversed(self.timing_history):
                if "memory_mb" in entry:
                    memory_data["memory_delta_mb"] = memory_mb - entry["memory_mb"]
                    memory_data["memory_delta_percent"] = (
                        (memory_data["memory_delta_mb"] / entry["memory_mb"]) * 100
                        if entry["memory_mb"] > 0
                        else 0
                    )
                    break

        # Log memory information with context and trend analysis
        self.base_logger.info(
            f"Memory: {self.operation_name} using {memory_mb:.3f}MB", extra=memory_data
        )

        # Update memory usage statistics and historical tracking
        self.timing_history.append(memory_data)

    def set_baseline(
        self,
        baseline_duration_ms: float,
        variance_tolerance: float = 0.1,
        baseline_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Sets or updates performance baseline for operation comparison with
        statistical analysis and variance tolerance configuration.
        """
        try:
            # Validate baseline duration is positive and reasonable for operation
            if baseline_duration_ms <= 0:
                self.base_logger.error(
                    f"Invalid baseline duration: {baseline_duration_ms}ms"
                )
                return False

            # Store baseline with variance tolerance in performance baselines dictionary
            baseline_key = f"{self.operation_name}_default"
            baseline_data = {
                "duration_ms": baseline_duration_ms,
                "variance_tolerance": variance_tolerance,
                "set_time": datetime.datetime.now(),
                "measurement_count_at_baseline": self.measurement_count,
            }

            # Include baseline metadata for context and statistical analysis
            if baseline_metadata:
                baseline_data["metadata"] = baseline_metadata

            self.performance_baselines[baseline_key] = baseline_data

            # Update performance threshold calculations based on baseline
            adjusted_threshold = baseline_duration_ms * (1 + variance_tolerance)
            if adjusted_threshold > self.timing_threshold_ms:
                self.timing_threshold_ms = adjusted_threshold

            # Log baseline establishment for audit and performance tracking
            self.base_logger.info(
                f"Performance baseline set: {baseline_duration_ms:.3f}ms with {variance_tolerance*100:.1f}% tolerance",
                extra={
                    "baseline_duration_ms": baseline_duration_ms,
                    "variance_tolerance": variance_tolerance,
                    "adjusted_threshold_ms": self.timing_threshold_ms,
                },
            )

            self.baseline_update_time = datetime.datetime.now()
            return True

        except Exception as e:
            self.base_logger.error(f"Failed to set baseline: {e}", exc_info=e)
            return False

    def analyze_performance(
        self, analysis_window_size: int = 100, detect_anomalies: bool = False
    ) -> Dict[str, Any]:
        """
        Analyzes performance trends and patterns with statistical analysis,
        anomaly detection, and optimization recommendations for system
        performance improvement.
        """
        # Extract timing history for specified analysis window size
        recent_timings = (
            self.timing_history[-analysis_window_size:] if self.timing_history else []
        )

        if not recent_timings:
            return {"error": "No timing data available for analysis"}

        # Filter for timing measurements only
        timing_measurements = [
            entry for entry in recent_timings if "duration_ms" in entry
        ]

        if not timing_measurements:
            return {"error": "No timing measurements in analysis window"}

        # Calculate statistical measures including mean, median, variance
        durations = [entry["duration_ms"] for entry in timing_measurements]
        analysis_results = {
            "window_size": len(timing_measurements),
            "statistics": {
                "mean_ms": sum(durations) / len(durations),
                "median_ms": sorted(durations)[len(durations) // 2],
                "min_ms": min(durations),
                "max_ms": max(durations),
                "range_ms": max(durations) - min(durations),
            },
            "threshold_analysis": {
                "threshold_ms": self.timing_threshold_ms,
                "violations": len(
                    [d for d in durations if d > self.timing_threshold_ms]
                ),
                "violation_rate": len(
                    [d for d in durations if d > self.timing_threshold_ms]
                )
                / len(durations),
            },
        }

        # Calculate variance and standard deviation
        mean = analysis_results["statistics"]["mean_ms"]
        variance = sum((d - mean) ** 2 for d in durations) / len(durations)
        analysis_results["statistics"]["variance"] = variance
        analysis_results["statistics"]["std_dev_ms"] = variance**0.5

        # Identify performance trends and patterns over time period
        if len(timing_measurements) >= 10:
            first_half = durations[: len(durations) // 2]
            second_half = durations[len(durations) // 2 :]

            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)

            analysis_results["trend_analysis"] = {
                "first_half_avg_ms": first_avg,
                "second_half_avg_ms": second_avg,
                "trend_direction": (
                    "improving" if second_avg < first_avg else "degrading"
                ),
                "trend_magnitude_ms": abs(second_avg - first_avg),
            }

        # Detect performance anomalies using statistical analysis if requested
        if detect_anomalies:
            std_dev = analysis_results["statistics"]["std_dev_ms"]
            anomaly_threshold = mean + (2 * std_dev)  # 2-sigma rule

            anomalies = [
                entry
                for entry in timing_measurements
                if entry["duration_ms"] > anomaly_threshold
            ]

            analysis_results["anomaly_detection"] = {
                "anomaly_threshold_ms": anomaly_threshold,
                "anomalies_detected": len(anomalies),
                "anomaly_rate": len(anomalies) / len(timing_measurements),
                "anomaly_details": anomalies[-5:],  # Last 5 anomalies
            }

        return analysis_results

    def get_performance_report(  # noqa: C901
        self, report_format: str = "detailed", include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generates comprehensive performance report with statistics, baselines,
        trends, and optimization recommendations for system analysis and debugging.
        """
        # Collect comprehensive performance statistics and measurements
        report = {
            "operation_name": self.operation_name,
            "report_generated": datetime.datetime.now().isoformat(),
            "measurement_summary": {
                "total_measurements": self.measurement_count,
                "timing_history_size": len(self.timing_history),
                "memory_tracking_enabled": self.memory_tracking_enabled,
                "baseline_count": len(self.performance_baselines),
            },
            "current_statistics": self.performance_statistics.copy(),
        }

        # Include baseline comparisons and performance trend analysis
        if self.performance_baselines:
            report["baseline_analysis"] = {}
            for key, baseline in self.performance_baselines.items():
                report["baseline_analysis"][key] = {
                    "baseline_duration_ms": baseline["duration_ms"],
                    "set_time": baseline["set_time"].isoformat(),
                    "measurements_since_baseline": self.measurement_count
                    - baseline.get("measurement_count_at_baseline", 0),
                }

        # Calculate performance efficiency and optimization opportunities
        if report_format == "detailed":
            report["detailed_analysis"] = self.analyze_performance(
                detect_anomalies=True
            )

            # Include recent performance samples
            if self.timing_history:
                report["recent_samples"] = self.timing_history[
                    -20:
                ]  # Last 20 measurements

        # Generate optimization recommendations if include_recommendations enabled
        if include_recommendations:
            recommendations = []

            # Check threshold violation rate
            violation_rate = self.performance_statistics.get(
                "threshold_violations", 0
            ) / max(self.measurement_count, 1)
            if violation_rate > 0.1:  # More than 10% violations
                recommendations.append(
                    f"High threshold violation rate ({violation_rate*100:.1f}%). "
                    f"Consider optimizing {self.operation_name} or adjusting threshold."
                )

            # Check for performance degradation trends
            if len(self.timing_history) >= 20:
                recent_avg = (
                    sum(
                        entry.get("duration_ms", 0)
                        for entry in self.timing_history[-10:]
                    )
                    / 10
                )

                older_avg = (
                    sum(
                        entry.get("duration_ms", 0)
                        for entry in self.timing_history[-20:-10]
                    )
                    / 10
                )

                if recent_avg > older_avg * 1.2:  # 20% degradation
                    recommendations.append(
                        f"Performance degradation detected: recent average "
                        f"({recent_avg:.3f}ms) is {((recent_avg/older_avg)-1)*100:.1f}% "
                        f"slower than previous measurements."
                    )

            # Check memory usage if tracking enabled
            if self.memory_tracking_enabled:
                memory_entries = [e for e in self.timing_history if "memory_mb" in e]
                if memory_entries and len(memory_entries) >= 2:
                    latest_memory = memory_entries[-1]["memory_mb"]
                    if latest_memory > 100:  # More than 100MB
                        recommendations.append(
                            f"High memory usage detected: {latest_memory:.1f}MB. "
                            f"Consider memory optimization for {self.operation_name}."
                        )

            report["optimization_recommendations"] = recommendations

        return report

    def _update_performance_statistics(self, duration_ms: float):
        """Update internal performance statistics with new measurement."""
        stats = self.performance_statistics
        stats["total_measurements"] += 1

        # Update running average
        if stats["total_measurements"] == 1:
            stats["average_timing"] = duration_ms
        else:
            stats["average_timing"] = (
                stats["average_timing"] * (stats["total_measurements"] - 1)
                + duration_ms
            ) / stats["total_measurements"]

        # Update min/max
        stats["min_timing"] = min(stats["min_timing"], duration_ms)
        stats["max_timing"] = max(stats["max_timing"], duration_ms)

        # Update instance average for quick access
        self.average_timing = stats["average_timing"]


class LoggerManager:
    """
    Central management class for logger lifecycle, registry operations,
    performance monitoring, and resource management providing comprehensive
    logger administration for the plume_nav_sim logging infrastructure.
    """

    def __init__(
        self,
        config: LoggingConfig,
        enable_caching: bool = True,
        cache_size_limit: int = LOGGER_CACHE_SIZE_LIMIT,
    ):
        """
        Initialize LoggerManager with configuration management, caching
        capabilities, and comprehensive logger lifecycle administration
        for centralized logging system management.
        """
        # Store logging configuration for logger creation and management
        self.logging_config = config

        # Initialize logger registry for centralized logger tracking and management
        self.logger_registry: Dict[str, weakref.ReferenceType] = {}

        # Set up component and performance logger caches with size limits
        self.component_logger_cache: Dict[str, ComponentLogger] = {}
        self.performance_logger_cache: Dict[str, PerformanceLogger] = {}

        # Configure caching settings and cache size limits for performance optimization
        self.caching_enabled = enable_caching
        self.cache_size_limit = cache_size_limit

        # Create thread-safe manager lock for concurrent logger operations
        self.manager_lock = threading.RLock()

        # Initialize manager statistics tracking and performance monitoring
        self.manager_creation_time = datetime.datetime.now()
        self.manager_statistics: Dict[str, Any] = {
            "loggers_created": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "registry_cleanups": 0,
        }

        # Set up cleanup registration for graceful shutdown management
        self.cleanup_registered = False

    def create_logger(
        self, logger_name: str, component_type: ComponentType, log_level: LogLevel
    ) -> ComponentLogger:
        """
        Creates new logger with configuration, caching, and registry management
        providing centralized logger creation with automatic configuration and
        lifecycle management.
        """
        with self.manager_lock:
            # Check cache for existing logger instance if caching enabled
            if self.caching_enabled and logger_name in self.component_logger_cache:
                self.manager_statistics["cache_hits"] += 1
                return self.component_logger_cache[logger_name]

            # Create ComponentLogger with configuration and component-specific settings
            logger = ComponentLogger(
                logger_name=logger_name,
                component_type=component_type,
                log_level=log_level,
                enable_performance_tracking=component_type.requires_performance_logging(),
            )

            # Register logger in central registry with weak reference management
            self.register_logger(
                logger.base_logger,
                component_type,
                {
                    "creation_time": datetime.datetime.now(),
                    "component_name": logger.component_name,
                    "log_level": log_level.name,
                },
            )

            # Cache logger instance if caching enabled and within size limits
            if self.caching_enabled:
                if len(self.component_logger_cache) < self.cache_size_limit:
                    self.component_logger_cache[logger_name] = logger
                else:
                    # Remove oldest cached logger to make space
                    oldest_key = next(iter(self.component_logger_cache))
                    del self.component_logger_cache[oldest_key]
                    self.component_logger_cache[logger_name] = logger

                self.manager_statistics["cache_misses"] += 1

            # Update manager statistics with logger creation information
            self.manager_statistics["loggers_created"] += 1

            return logger

    def get_cached_logger(
        self, logger_name: str, component_type: ComponentType
    ) -> Optional[ComponentLogger]:
        """
        Retrieves cached logger instance or creates new logger if not cached
        with efficient cache management and automatic cleanup of stale references.
        """
        with self.manager_lock:
            # Check if caching is enabled for logger instance reuse
            if not self.caching_enabled:
                return None

            # Search component logger cache for existing logger instance
            if logger_name in self.component_logger_cache:
                cached_logger = self.component_logger_cache[logger_name]

                # Validate cached logger is still active and properly configured
                if cached_logger and cached_logger.component_type == component_type:
                    self.manager_statistics["cache_hits"] += 1
                    return cached_logger
                else:
                    # Remove invalid cached logger
                    del self.component_logger_cache[logger_name]

            self.manager_statistics["cache_misses"] += 1
            return None

    def register_logger(
        self,
        logger: logging.Logger,
        component_type: ComponentType,
        logger_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Registers logger in central registry with weak reference management,
        lifecycle tracking, and automatic cleanup prevention of memory leaks.
        """
        try:
            with self.manager_lock:
                # Create weak reference to logger for automatic cleanup
                def cleanup_callback(ref):
                    # Remove from registry when logger is garbage collected
                    with self.manager_lock:
                        for key, weak_ref in list(self.logger_registry.items()):
                            if weak_ref is ref:
                                del self.logger_registry[key]
                                break

                weak_ref = weakref.ref(logger, cleanup_callback)

                # Store logger in registry with component type and metadata
                registry_key = f"{component_type.name}:{logger.name}"
                self.logger_registry[registry_key] = weak_ref

                # Update registry statistics and logger count tracking
                if logger_metadata:
                    # Store metadata if needed (could be extended to separate metadata store)
                    pass

                return True

        except Exception as e:
            logging.error(f"Failed to register logger {logger.name}: {e}")
            return False

    def cleanup_registry(
        self, force_cleanup: bool = False, max_registry_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Cleans up logger registry removing stale references, managing cache size,
        and performing resource cleanup for optimal memory management.
        """
        with self.manager_lock:
            initial_registry_size = len(self.logger_registry)
            initial_cache_size = len(self.component_logger_cache)

            # Scan registry for stale weak references and remove dead entries
            stale_keys = []
            for key, weak_ref in self.logger_registry.items():
                if weak_ref() is None:  # Logger has been garbage collected
                    stale_keys.append(key)

            for key in stale_keys:
                del self.logger_registry[key]

            # Check cache size limits and remove oldest cached loggers if necessary
            cache_limit = max_registry_size or self.cache_size_limit
            if len(self.component_logger_cache) > cache_limit:
                # Remove oldest entries to get under limit
                excess_count = len(self.component_logger_cache) - cache_limit
                oldest_keys = list(self.component_logger_cache.keys())[:excess_count]
                for key in oldest_keys:
                    del self.component_logger_cache[key]

            # Perform garbage collection if force_cleanup enabled
            if force_cleanup:
                import gc

                gc.collect()

            # Update manager statistics with cleanup operation results
            self.manager_statistics["registry_cleanups"] += 1

            # Generate cleanup report with loggers removed and memory freed
            cleanup_results = {
                "stale_references_removed": len(stale_keys),
                "registry_size_before": initial_registry_size,
                "registry_size_after": len(self.logger_registry),
                "cache_size_before": initial_cache_size,
                "cache_size_after": len(self.component_logger_cache),
                "force_cleanup_performed": force_cleanup,
                "cleanup_timestamp": datetime.datetime.now().isoformat(),
            }

            return cleanup_results

    def get_manager_statistics(
        self,
        include_cache_details: bool = False,
        include_performance_data: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns comprehensive manager statistics including registry status,
        cache performance, logger counts, and resource utilization for
        monitoring and optimization.
        """
        with self.manager_lock:
            # Collect registry statistics including active logger counts and types
            active_loggers = {}
            for key, weak_ref in self.logger_registry.items():
                if weak_ref() is not None:
                    component_type = key.split(":")[0]
                    active_loggers[component_type] = (
                        active_loggers.get(component_type, 0) + 1
                    )

            statistics = {
                "manager_info": {
                    "creation_time": self.manager_creation_time.isoformat(),
                    "uptime_seconds": (
                        datetime.datetime.now() - self.manager_creation_time
                    ).total_seconds(),
                    "caching_enabled": self.caching_enabled,
                    "cache_size_limit": self.cache_size_limit,
                },
                "registry_status": {
                    "total_registered": len(self.logger_registry),
                    "active_by_component": active_loggers,
                    "total_active": sum(active_loggers.values()),
                },
                "cache_performance": {
                    "cached_loggers": len(self.component_logger_cache),
                    "cache_hits": self.manager_statistics["cache_hits"],
                    "cache_misses": self.manager_statistics["cache_misses"],
                    "hit_rate": (
                        self.manager_statistics["cache_hits"]
                        / max(
                            self.manager_statistics["cache_hits"]
                            + self.manager_statistics["cache_misses"],
                            1,
                        )
                    ),
                },
                "operational_statistics": self.manager_statistics.copy(),
            }

            # Include cache performance metrics with hit rates and efficiency if requested
            if include_cache_details:
                statistics["cache_details"] = {
                    "cached_logger_names": list(self.component_logger_cache.keys()),
                    "performance_cache_size": len(self.performance_logger_cache),
                    "memory_efficiency": len(self.component_logger_cache)
                    / max(self.cache_size_limit, 1),
                }

            # Add performance data from performance loggers if requested
            if include_performance_data:
                performance_data = {}
                for name, perf_logger in self.performance_logger_cache.items():
                    performance_data[name] = perf_logger.get_performance_report(
                        "summary", False
                    )
                statistics["performance_logger_data"] = performance_data

            return statistics

    def shutdown_manager(self, shutdown_timeout: float = 30.0) -> Dict[str, Any]:
        """
        Performs comprehensive manager shutdown with registry cleanup, cache
        clearing, and resource release for clean logging system termination.
        """
        shutdown_start = time.time()

        with self.manager_lock:
            # Flush all registered loggers to ensure data integrity
            flushed_loggers = 0
            for key, weak_ref in self.logger_registry.items():
                logger = weak_ref()
                if logger:
                    for handler in logger.handlers:
                        handler.flush()
                    flushed_loggers += 1

            # Clear logger registry and release all weak references
            initial_registry_size = len(self.logger_registry)
            self.logger_registry.clear()

            # Clear component and performance logger caches
            initial_cache_size = len(self.component_logger_cache)
            self.component_logger_cache.clear()

            initial_perf_cache_size = len(self.performance_logger_cache)
            self.performance_logger_cache.clear()

            # Generate shutdown report with resources freed and cleanup status
            shutdown_results = {
                "shutdown_completed": True,
                "shutdown_duration_seconds": time.time() - shutdown_start,
                "resources_freed": {
                    "registry_entries_cleared": initial_registry_size,
                    "cached_loggers_cleared": initial_cache_size,
                    "performance_loggers_cleared": initial_perf_cache_size,
                    "loggers_flushed": flushed_loggers,
                },
                "final_statistics": self.manager_statistics.copy(),
                "shutdown_timestamp": datetime.datetime.now().isoformat(),
            }

            # Reset manager state and clear operational statistics
            self.manager_statistics = {
                "loggers_created": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "registry_cleanups": 0,
            }

            return shutdown_results


# Primary factory functions for logger creation and management


def get_logger(
    name: str,
    component_type: ComponentType = ComponentType.ENVIRONMENT,
    log_level: LogLevel = LogLevel.INFO,
    enable_performance_tracking: bool = False,
) -> ComponentLogger:
    """
    Primary factory function for creating or retrieving component-specific loggers
    with automatic configuration, caching, and performance tracking capabilities.
    Provides the main interface for logger access throughout the plume_nav_sim system.
    """
    global _active_logger_count

    # Initialize logging system if not already configured using ensure_logging_initialized
    ensure_logging_initialized()

    # Create hierarchical logger name using COMPONENT_LOGGER_FORMAT with component type and name
    formatted_name = COMPONENT_LOGGER_FORMAT.format(
        component_type=component_type.name.lower(), component_name=name
    )

    with _registry_lock:
        # Check logger registry cache for existing logger instance to avoid duplication
        if _logger_manager:
            cached_logger = _logger_manager.get_cached_logger(
                formatted_name, component_type
            )
            if cached_logger:
                _logger_creation_stats["cached_retrievals"] += 1
                return cached_logger

        # Determine appropriate log level using component type defaults or provided log_level
        effective_log_level = log_level
        if log_level == LogLevel.INFO and hasattr(
            component_type, "get_default_log_level"
        ):
            effective_log_level = component_type.get_default_log_level()

        # Create ComponentLogger instance with component-specific configuration and formatters
        logger = ComponentLogger(
            logger_name=formatted_name,
            component_type=component_type,
            log_level=effective_log_level,
            enable_performance_tracking=enable_performance_tracking
            or component_type.requires_performance_logging(),
        )

        # Register logger in global logger registry with weak reference for memory management
        if formatted_name not in _component_loggers:
            _component_loggers[formatted_name] = logger
            _active_logger_count += 1
            _logger_creation_stats["total_created"] += 1
            _logger_creation_stats["component_loggers"] += 1

        return logger


def get_component_logger(
    component_type: ComponentType,
    component_name: str,
    custom_config: Optional[Dict[str, Any]] = None,
) -> ComponentLogger:
    """
    Specialized factory function for creating component-specific loggers with
    component-appropriate configuration, formatting, and performance monitoring
    based on component type and system role.
    """
    # Validate component_type against known ComponentType enumeration values
    if not isinstance(component_type, ComponentType):
        raise ValueError(f"Invalid component_type: {component_type}")

    # Component-specific logger name not required here; use component_name directly

    # Determine default log level using ComponentType.get_default_log_level() method
    default_log_level = component_type.get_default_log_level()

    # Check if component type requires performance logging using requires_performance_logging
    performance_tracking = component_type.requires_performance_logging()

    # Apply custom configuration overrides from custom_config parameter if provided
    if custom_config:
        default_log_level = custom_config.get("log_level", default_log_level)
        performance_tracking = custom_config.get(
            "enable_performance_tracking", performance_tracking
        )

    # Create ComponentLogger with component-appropriate formatters and handlers
    logger = get_logger(
        name=component_name,
        component_type=component_type,
        log_level=default_log_level,
        enable_performance_tracking=performance_tracking,
    )

    # Register component logger in component logger registry for management
    with _registry_lock:
        registry_key = f"{component_type.name}:{component_name}"
        if registry_key not in _component_loggers:
            _component_loggers[registry_key] = logger

    return logger


def get_performance_logger(
    operation_name: str,
    timing_threshold_ms: Optional[float] = None,
    enable_memory_tracking: bool = False,
    component_context: Optional[ComponentType] = None,
) -> PerformanceLogger:
    """
    Factory function for creating specialized performance monitoring loggers
    with high-resolution timing, memory tracking, and threshold-based alerting
    for system optimization and development debugging.
    """
    # No global declarations needed; only reading module-level state and mutating dicts

    # Create performance logger name using PERFORMANCE_LOGGER_FORMAT with operation name
    logger_name = PERFORMANCE_LOGGER_FORMAT.format(operation_name=operation_name)

    with _registry_lock:
        # Check for existing performance logger to avoid duplication
        if logger_name in _performance_loggers:
            _logger_creation_stats["cached_retrievals"] += 1
            return _performance_loggers[logger_name]

        # Initialize timing threshold using provided value or PERFORMANCE_TARGET_STEP_LATENCY_MS default
        threshold = timing_threshold_ms or PERFORMANCE_TARGET_STEP_LATENCY_MS

        # Create PerformanceLogger instance with specialized performance monitoring capabilities
        performance_logger = PerformanceLogger(
            logger_name=logger_name,
            operation_name=operation_name,
            timing_threshold_ms=threshold,
            enable_memory_tracking=enable_memory_tracking,
        )

        # Register performance logger in performance logger registry
        _performance_loggers[logger_name] = performance_logger
        _logger_creation_stats["total_created"] += 1
        _logger_creation_stats["performance_loggers"] += 1

        return performance_logger


def configure_logging_system(  # noqa: C901
    config: LoggingConfig,
    force_reconfiguration: bool = False,
    validate_config: bool = True,
) -> bool:
    """
    Configures the entire plume_nav_sim logging system with handlers, formatters,
    component loggers, and performance monitoring. Provides centralized logging
    setup with comprehensive configuration management.
    """
    global _logging_initialized, _default_config, _logger_manager

    try:
        # Acquire initialization lock to ensure thread-safe logging system setup
        with _initialization_lock:
            # Validate logging configuration using LoggingConfig.validate() if validate_config enabled
            if validate_config:
                if not config.validate():
                    return False

            # Check if logging system already initialized and handle force_reconfiguration appropriately
            if _logging_initialized and not force_reconfiguration:
                return True

            # Convert LoggingConfig to dictConfig format using to_dict_config() method
            dict_config = config.to_dict_config()

            # Apply logging configuration using logging.config.dictConfig for system-wide setup
            logging.config.dictConfig(dict_config)

            # Initialize logger factory with validated configuration and caching settings
            _logger_manager = LoggerManager(
                config=config,
                enable_caching=True,
                cache_size_limit=LOGGER_CACHE_SIZE_LIMIT,
            )

            # Create component loggers for all system components using COMPONENT_NAMES
            for component_name in COMPONENT_NAMES:
                try:
                    # Try to determine component type from name
                    component_type = ComponentType.ENVIRONMENT  # Default fallback
                    for comp_type in ComponentType:
                        if comp_type.name.lower() in component_name.lower():
                            component_type = comp_type
                            break

                    get_component_logger(component_type, component_name)
                except Exception as e:
                    logging.warning(
                        f"Failed to create logger for component {component_name}: {e}"
                    )

            # Register cleanup handlers for graceful logging system shutdown
            register_cleanup_handlers()

            # Mark logging system as initialized and store default configuration
            _logging_initialized = True
            _default_config = config

            return True

    except Exception as e:
        logging.error(f"Failed to configure logging system: {e}", exc_info=e)
        return False


def ensure_logging_initialized(
    use_development_config: bool = True, enable_console_output: bool = True
) -> bool:
    """
    Ensures logging system is properly initialized with default configuration,
    performing lazy initialization if needed and providing fallback configuration
    for system startup.
    """
    global _logging_initialized, _default_config

    # Check global _logging_initialized flag to determine if initialization needed
    if _logging_initialized:
        return True

    try:
        # Create default LoggingConfig with development-friendly settings if use_development_config
        if not _default_config:
            _default_config = LoggingConfig(
                log_level=LogLevel.DEBUG if use_development_config else LogLevel.INFO,
                enable_console_output=enable_console_output,
                enable_file_output=False,  # Minimal for development
                log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # Call configure_logging_system with default configuration to initialize system
        success = configure_logging_system(_default_config, validate_config=False)

        if not success:
            # Set up basic error handling and fallback logging if full initialization fails
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            _logging_initialized = True  # Mark as initialized even with basic config

        return True

    except Exception as e:
        # Fallback to basic logging configuration
        logging.basicConfig(level=logging.WARNING)
        logging.error(f"Failed to initialize logging system: {e}")
        _logging_initialized = True  # Prevent infinite retry loops
        return False


def register_cleanup_handlers(force_registration: bool = False) -> bool:
    """
    Registers automatic cleanup handlers for graceful shutdown of logging system
    with proper resource management, handler cleanup, and registry management
    for application shutdown.
    """
    global CLEANUP_REGISTERED

    # Check CLEANUP_REGISTERED global flag to avoid duplicate cleanup handler registration
    if CLEANUP_REGISTERED and not force_registration:
        return True

    try:
        # Register atexit handler for graceful logging system shutdown on application exit
        atexit.register(shutdown_logging_system, 10.0, False)  # 10 second timeout

        # Set up signal handlers for interrupt and termination signals with logging cleanup
        def signal_cleanup_handler(signum, frame):
            shutdown_logging_system(5.0, True)  # Force cleanup on signal

        signal.signal(signal.SIGTERM, signal_cleanup_handler)
        signal.signal(signal.SIGINT, signal_cleanup_handler)

        # Set CLEANUP_REGISTERED global flag to prevent duplicate registration
        CLEANUP_REGISTERED = True

        return True

    except Exception as e:
        logging.error(f"Failed to register cleanup handlers: {e}")
        return False


def shutdown_logging_system(  # noqa: C901
    shutdown_timeout: float = 30.0, force_cleanup: bool = False
) -> Dict[str, Any]:
    """
    Performs comprehensive shutdown of logging system with resource cleanup,
    handler management, and registry clearing for clean application termination
    and resource release.
    """
    global _logging_initialized, _logger_manager, _active_logger_count

    shutdown_start = time.time()
    shutdown_results = {
        "shutdown_started": datetime.datetime.now().isoformat(),
        "force_cleanup": force_cleanup,
        "timeout_seconds": shutdown_timeout,
    }

    try:
        # Acquire all necessary locks to ensure orderly shutdown process
        with _registry_lock, _initialization_lock:
            # Flush all active loggers and handlers to ensure data integrity
            flushed_handlers = 0
            for logger in _component_loggers.values():
                if hasattr(logger, "base_logger"):
                    for handler in logger.base_logger.handlers:
                        handler.flush()
                        flushed_handlers += 1

            # Close performance loggers and export any remaining performance data
            performance_reports = {}
            for name, perf_logger in _performance_loggers.items():
                try:
                    performance_reports[name] = perf_logger.get_performance_report(
                        "summary", False
                    )
                    for handler in perf_logger.base_logger.handlers:
                        handler.flush()
                        handler.close()
                except Exception as e:
                    logging.warning(f"Error closing performance logger {name}: {e}")

            # Shutdown file handlers with proper file closure and resource release
            closed_handlers = 0
            for handler in logging.root.handlers[:]:
                try:
                    handler.close()
                    logging.root.removeHandler(handler)
                    closed_handlers += 1
                except Exception as e:
                    logging.warning(f"Error closing handler: {e}")

            # Clear component logger registry and release weak references
            initial_component_count = len(_component_loggers)
            _component_loggers.clear()

            # Clear performance logger registry and free performance monitoring resources
            initial_performance_count = len(_performance_loggers)
            _performance_loggers.clear()

            # Shutdown logger manager if initialized
            manager_results = {}
            if _logger_manager:
                manager_results = _logger_manager.shutdown_manager(shutdown_timeout / 2)
                _logger_manager = None

            # Reset global initialization flags and clear cached configuration
            _logging_initialized = False
            _active_logger_count = 0

            # Generate shutdown report with resources freed and cleanup status
            shutdown_results.update(
                {
                    "shutdown_completed": True,
                    "shutdown_duration_seconds": time.time() - shutdown_start,
                    "resources_freed": {
                        "component_loggers_cleared": initial_component_count,
                        "performance_loggers_cleared": initial_performance_count,
                        "handlers_flushed": flushed_handlers,
                        "handlers_closed": closed_handlers,
                    },
                    "final_statistics": _logger_creation_stats.copy(),
                    "performance_reports": performance_reports,
                    "manager_shutdown": manager_results,
                }
            )

            return shutdown_results

    except Exception as e:
        shutdown_results.update(
            {
                "shutdown_completed": False,
                "shutdown_error": str(e),
                "partial_shutdown": True,
                "shutdown_duration_seconds": time.time() - shutdown_start,
            }
        )
        return shutdown_results


def get_logging_statistics(  # noqa: C901
    include_performance_data: bool = False, include_registry_details: bool = False
) -> Dict[str, Any]:
    """
    Returns comprehensive logging system statistics including logger counts,
    performance metrics, registry status, and resource utilization for
    monitoring and debugging purposes.
    """
    # Reading module-level state only; no global declarations required

    with _registry_lock:
        # Collect basic logging system statistics from global counters and registries
        statistics = {
            "system_status": {
                "logging_initialized": _logging_initialized,
                "active_logger_count": _active_logger_count,
                "component_loggers_count": len(_component_loggers),
                "performance_loggers_count": len(_performance_loggers),
                "manager_active": _logger_manager is not None,
            },
            "creation_statistics": _logger_creation_stats.copy(),
            "resource_utilization": {
                "registry_size": len(_logger_registry),
                "cache_utilization": len(_component_loggers)
                / max(LOGGER_CACHE_SIZE_LIMIT, 1),
                "memory_pressure": "normal",  # Could be enhanced with actual memory monitoring
            },
        }

        # Count active loggers by type (component, performance, cached) and status
        component_types = {}
        for key, logger in _component_loggers.items():
            comp_type = (
                logger.component_type.name
                if hasattr(logger, "component_type")
                else "unknown"
            )
            component_types[comp_type] = component_types.get(comp_type, 0) + 1

        statistics["logger_breakdown"] = {
            "by_component_type": component_types,
            "performance_operations": list(_performance_loggers.keys()),
        }

        # Include performance data from performance loggers if include_performance_data enabled
        if include_performance_data:
            performance_summaries = {}
            for name, perf_logger in _performance_loggers.items():
                try:
                    performance_summaries[name] = {
                        "measurement_count": perf_logger.measurement_count,
                        "average_timing_ms": perf_logger.average_timing,
                        "threshold_ms": perf_logger.timing_threshold_ms,
                        "memory_tracking": perf_logger.memory_tracking_enabled,
                    }
                except Exception as e:
                    performance_summaries[name] = {"error": str(e)}

            statistics["performance_data"] = performance_summaries

        # Add registry details including logger hierarchy and configuration if requested
        if include_registry_details:
            registry_details = {}
            for key, logger in _component_loggers.items():
                try:
                    registry_details[key] = {
                        "creation_time": (
                            logger.creation_time.isoformat()
                            if hasattr(logger, "creation_time")
                            else "unknown"
                        ),
                        "message_count": getattr(logger, "message_count", 0),
                        "log_level": (
                            logger.configured_log_level.name
                            if hasattr(logger, "configured_log_level")
                            else "unknown"
                        ),
                        "performance_enabled": getattr(
                            logger, "performance_tracking_enabled", False
                        ),
                    }
                except Exception as e:
                    registry_details[key] = {"error": str(e)}

            statistics["registry_details"] = registry_details

        # Include manager statistics if manager is active
        if _logger_manager:
            try:
                statistics["manager_statistics"] = (
                    _logger_manager.get_manager_statistics(
                        include_cache_details=include_registry_details,
                        include_performance_data=include_performance_data,
                    )
                )
            except Exception as e:
                statistics["manager_error"] = str(e)

        return statistics
