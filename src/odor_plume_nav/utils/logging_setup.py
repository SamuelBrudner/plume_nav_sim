"""
Enhanced Logging Configuration Module for Odor Plume Navigation.

This module provides a comprehensive, configuration-driven logging setup across the application,
using Loguru for advanced structured logging capabilities with Hydra and Pydantic integration.
Supports environment-specific configurations, performance monitoring, and automatic correlation 
tracking for research and production environments.

Key Features:
- Configuration-driven initialization with Hydra and Pydantic integration
- Environment-specific logging configurations (development, testing, production)
- Performance monitoring and diagnostic logging for real-time simulation monitoring
- Enhanced module logger creation with automatic context binding
- Correlation ID generation for experiment traceability (request_id/episode_id support)
- Multiple format patterns for different use cases
- Automatic performance threshold monitoring and alerting
- Environment step() latency monitoring with ≤10ms threshold warnings
- Structured JSON logging with distributed tracing support
- Legacy gym API deprecation detection and warnings
- Comprehensive performance timing integration (frame rate, memory, database)

Enhanced Performance Monitoring:
- Environment step() latency tracking with automatic WARN logging when ≤10ms threshold exceeded
- Frame rate measurement with automatic warnings below 30 FPS target
- Memory usage delta tracking with warnings for significant increases
- Database operation timing with latency violation detection
- Structured JSON output with correlation IDs for distributed analysis

Legacy API Deprecation Support:
- Automatic detection of legacy gym imports vs gymnasium
- Structured DeprecationWarning messages via logger.warning
- Migration guidance with specific code examples and resources

Example Usage:
    >>> # Enhanced correlation context with distributed tracing
    >>> with correlation_context("rl_training", episode_id="ep_001") as ctx:
    ...     logger.info("Starting RL episode")
    ...     with create_step_timer() as step_metrics:
    ...         obs, reward, done, info = env.step(action)
    ...     # Automatic warning if step() > 10ms
    
    >>> # Legacy API monitoring
    >>> monitor_environment_creation("CartPole-v1", "gym.make")  # Triggers deprecation warning
    
    >>> # Comprehensive performance monitoring
    >>> logger = get_enhanced_logger(__name__)
    >>> with logger.performance_timer("training_batch") as metrics:
    ...     model.train(batch_data)
    >>> logger.log_step_latency_violation(0.015)  # >10ms step warning
"""

import sys
import os
import json
import time
import uuid
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Literal, ContextManager
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, asdict
import platform
import psutil

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing_extensions import Self

# Import configuration models for integration
try:
    from odor_plume_nav.config.models import SimulationConfig
except ImportError:
    # Fallback for cases where config models aren't available yet
    SimulationConfig = None


# Preserved existing format constants for backward compatibility
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

MODULE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<blue>module={extra[module]}</blue> - "
    "<level>{message}</level>"
)

# Enhanced format patterns for different environments and use cases
ENHANCED_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>correlation_id={extra[correlation_id]}</magenta> | "
    "<blue>module={extra[module]}</blue> - "
    "<level>{message}</level>"
)

HYDRA_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan> | "
    "<yellow>config_hash={extra[config_hash]}</yellow> | "
    "<magenta>correlation_id={extra[correlation_id]}</magenta> - "
    "<level>{message}</level>"
)

CLI_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

MINIMAL_FORMAT = "<level>{level: <8}</level> | <level>{message}</level>"

PRODUCTION_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "correlation_id={extra[correlation_id]} | "
    "module={extra[module]} | "
    "{message}"
)

# JSON format for structured logging in production environments
JSON_FORMAT = "{time} | {level} | {name} | {message} | {extra}"

# Log levels with enhanced metadata for performance correlation
LOG_LEVELS = {
    "TRACE": {"color": "<cyan>", "value": 5},
    "DEBUG": {"color": "<blue>", "value": 10},
    "INFO": {"color": "<green>", "value": 20},
    "SUCCESS": {"color": "<green>", "value": 25},
    "WARNING": {"color": "<yellow>", "value": 30},
    "ERROR": {"color": "<red>", "value": 40},
    "CRITICAL": {"color": "<red>", "value": 50},
}

# Performance monitoring thresholds (in seconds unless noted)
PERFORMANCE_THRESHOLDS = {
    "cli_init": 2.0,
    "config_validation": 0.5,
    "db_connection": 0.5,
    "simulation_fps_min": 30.0,  # FPS, not seconds
    "video_frame_processing": 0.033,  # 33ms per frame
    "db_operation": 0.1,  # 100ms for typical operations
    "environment_step": 0.010,  # 10ms per step - critical RL performance requirement
    "frame_rate_measurement": 0.033,  # 33ms target frame rate
    "memory_usage_delta": 0.050,  # 50ms for memory measurement operations
}

# Environment-specific logging defaults
ENVIRONMENT_DEFAULTS = {
    "development": {
        "level": "DEBUG",
        "enable_performance": True,
        "format": "enhanced",
        "console_enabled": True,
        "file_enabled": True,
        "correlation_enabled": True,
        "memory_tracking": True,
    },
    "testing": {
        "level": "INFO",
        "enable_performance": False,
        "format": "minimal",
        "console_enabled": True,
        "file_enabled": False,
        "correlation_enabled": False,
        "memory_tracking": False,
    },
    "production": {
        "level": "INFO",
        "enable_performance": True,
        "format": "production",
        "console_enabled": True,
        "file_enabled": True,
        "correlation_enabled": True,
        "memory_tracking": True,
    },
    "batch": {
        "level": "WARNING",
        "enable_performance": False,
        "format": "json",
        "console_enabled": False,
        "file_enabled": True,
        "correlation_enabled": True,
        "memory_tracking": False,
    },
}


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking structure for logging correlation."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_delta: Optional[float] = None
    thread_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def complete(self) -> Self:
        """Mark performance measurement as complete and calculate metrics."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.memory_before is not None:
            self.memory_after = self._get_memory_usage()
            self.memory_delta = self.memory_after - self.memory_before
        return self

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except (psutil.Error, ImportError):
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert performance metrics to dictionary for logging."""
        return asdict(self)

    def is_slow(self, threshold: Optional[float] = None) -> bool:
        """Check if operation exceeded performance threshold."""
        if self.duration is None:
            return False
        
        # Use provided threshold or lookup from PERFORMANCE_THRESHOLDS
        if threshold is None:
            threshold = PERFORMANCE_THRESHOLDS.get(self.operation_name, 1.0)
        
        return self.duration > threshold


class LoggingConfig(BaseModel):
    """
    Enhanced Pydantic configuration model for logging setup with Hydra integration.
    
    Provides comprehensive logging configuration supporting environment-specific settings,
    performance monitoring, and structured correlation tracking for research and production
    environments.
    """
    
    # Environment and level configuration
    environment: Literal["development", "testing", "production", "batch"] = Field(
        default="development",
        description="Deployment environment determining default logging behavior"
    )
    level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Base logging level. Supports ${oc.env:LOG_LEVEL,INFO} interpolation"
    )
    
    # Format configuration with environment variable support
    format: Literal["default", "module", "enhanced", "hydra", "cli", "minimal", "production", "json"] = Field(
        default="enhanced",
        description="Log message format pattern. Supports ${oc.env:LOG_FORMAT,enhanced}"
    )
    
    # Output configuration
    console_enabled: bool = Field(
        default=True,
        description="Enable console output. Supports ${oc.env:LOG_CONSOLE,true}"
    )
    file_enabled: bool = Field(
        default=True,
        description="Enable file logging. Supports ${oc.env:LOG_FILE,true}"
    )
    file_path: Optional[Union[str, Path]] = Field(
        default=None,
        description="Log file path. Supports ${oc.env:LOG_PATH} interpolation"
    )
    
    # File rotation and retention
    rotation: str = Field(
        default="10 MB",
        description="Log file rotation trigger. Supports ${oc.env:LOG_ROTATION,10 MB}"
    )
    retention: str = Field(
        default="1 week",
        description="Log file retention period. Supports ${oc.env:LOG_RETENTION,1 week}"
    )
    
    # Performance monitoring configuration
    enable_performance: bool = Field(
        default=False,
        description="Enable performance monitoring. Supports ${oc.env:ENABLE_PERF_LOGGING,false}"
    )
    performance_threshold: float = Field(
        default=1.0,
        description="Slow operation threshold in seconds. Supports ${oc.env:PERF_THRESHOLD,1.0}"
    )
    
    # Correlation and tracing
    correlation_enabled: bool = Field(
        default=True,
        description="Enable correlation ID tracking. Supports ${oc.env:LOG_CORRELATION,true}"
    )
    memory_tracking: bool = Field(
        default=False,
        description="Enable memory usage tracking. Supports ${oc.env:LOG_MEMORY,false}"
    )
    
    # Advanced features
    backtrace: bool = Field(
        default=True,
        description="Include backtrace in error logs"
    )
    diagnose: bool = Field(
        default=True,
        description="Enable enhanced exception diagnosis"
    )
    enqueue: bool = Field(
        default=True,
        description="Enqueue log messages for better multiprocessing support"
    )
    
    # Context binding defaults
    default_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default context fields to include in all log messages"
    )
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        """Validate and normalize log file path."""
        if v is None:
            return None
        
        if isinstance(v, str):
            # Handle Hydra environment variable interpolation
            if v.startswith('${oc.env:'):
                return v
            path = Path(v)
        else:
            path = v
        
        # Ensure directory exists
        if not str(path).startswith('${'):  # Skip for interpolated paths
            path.parent.mkdir(parents=True, exist_ok=True)
        
        return str(path)
    
    @field_validator('environment')
    @classmethod
    def apply_environment_defaults(cls, v, info):
        """Apply environment-specific defaults when environment is specified."""
        # Note: This validator runs before other fields are set, so we can't modify
        # other fields here. Environment defaults are applied in setup_logger instead.
        return v
    
    @field_validator('level')
    @classmethod 
    def validate_log_level(cls, v):
        """Validate log level is supported."""
        if v not in LOG_LEVELS:
            raise ValueError(f"Invalid log level '{v}'. Must be one of: {list(LOG_LEVELS.keys())}")
        return v
    
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "environment": "development",
                    "level": "DEBUG",
                    "format": "enhanced",
                    "enable_performance": True,
                    "correlation_enabled": True
                },
                {
                    "environment": "production",
                    "level": "INFO", 
                    "format": "production",
                    "file_path": "${oc.env:LOG_PATH,./logs/app.log}",
                    "enable_performance": True
                }
            ]
        }
    )
    
    def get_format_pattern(self) -> str:
        """Get the actual format string for the specified format type."""
        format_patterns = {
            "default": DEFAULT_FORMAT,
            "module": MODULE_FORMAT,
            "enhanced": ENHANCED_FORMAT,
            "hydra": HYDRA_FORMAT,
            "cli": CLI_FORMAT,
            "minimal": MINIMAL_FORMAT,
            "production": PRODUCTION_FORMAT,
            "json": JSON_FORMAT,
        }
        return format_patterns.get(self.format, ENHANCED_FORMAT)
    
    def apply_environment_defaults(self) -> "LoggingConfig":
        """Apply environment-specific defaults to unset fields."""
        env_defaults = ENVIRONMENT_DEFAULTS.get(self.environment, {})
        
        # Create a new config with environment defaults applied
        config_dict = self.model_dump()
        for key, default_value in env_defaults.items():
            # Only apply default if the field wasn't explicitly set
            if key in config_dict and config_dict[key] == getattr(LoggingConfig.model_fields[key], 'default', None):
                config_dict[key] = default_value
        
        return LoggingConfig(**config_dict)


# Thread-local storage for correlation context
_context_storage = threading.local()


class CorrelationContext:
    """
    Thread-local correlation context manager for experiment traceability.
    
    Maintains correlation IDs and context metadata across function calls within
    the same thread, enabling comprehensive experiment tracking and debugging.
    Enhanced with request_id/episode_id support for distributed tracing.
    """
    
    def __init__(self, request_id: Optional[str] = None, episode_id: Optional[str] = None):
        self.correlation_id = str(uuid.uuid4())
        self.request_id = request_id or str(uuid.uuid4())
        self.episode_id = episode_id
        self.experiment_metadata = {}
        self.performance_stack = []
        self.start_time = time.time()
        self.step_count = 0  # Track environment steps for performance monitoring
    
    def bind_context(self, **kwargs) -> Dict[str, Any]:
        """Get context dictionary for binding to loggers with enhanced distributed tracing."""
        context = {
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "thread_id": threading.current_thread().ident,
            "process_id": os.getpid(),
            "step_count": self.step_count,
            **self.experiment_metadata,
            **kwargs
        }
        
        # Add episode_id if available (for RL environments)
        if self.episode_id is not None:
            context["episode_id"] = self.episode_id
            
        return context
    
    def add_metadata(self, **metadata):
        """Add metadata to correlation context."""
        self.experiment_metadata.update(metadata)
    
    def push_performance(self, operation: str, **metadata) -> PerformanceMetrics:
        """Start tracking performance for an operation."""
        metrics = PerformanceMetrics(
            operation_name=operation,
            start_time=time.time(),
            correlation_id=self.correlation_id,
            thread_id=str(threading.current_thread().ident),
            metadata=metadata
        )
        
        # Track memory if enabled
        try:
            metrics.memory_before = metrics._get_memory_usage()
        except Exception:
            pass
        
        self.performance_stack.append(metrics)
        return metrics
    
    def pop_performance(self) -> Optional[PerformanceMetrics]:
        """Complete performance tracking for the most recent operation."""
        if not self.performance_stack:
            return None
        
        metrics = self.performance_stack.pop()
        return metrics.complete()
    
    def increment_step(self):
        """Increment step counter for environment step tracking."""
        self.step_count += 1
    
    def set_episode_id(self, episode_id: str):
        """Set episode ID for RL environment tracking."""
        self.episode_id = episode_id
    
    def reset_episode(self, new_episode_id: Optional[str] = None):
        """Reset episode tracking with optional new episode ID."""
        self.episode_id = new_episode_id or str(uuid.uuid4())
        self.step_count = 0


def get_correlation_context() -> CorrelationContext:
    """Get or create correlation context for current thread."""
    if not hasattr(_context_storage, 'context'):
        _context_storage.context = CorrelationContext()
    return _context_storage.context


def set_correlation_context(context: CorrelationContext):
    """Set correlation context for current thread."""
    _context_storage.context = context


def create_step_timer() -> ContextManager[PerformanceMetrics]:
    """
    Create a context manager for timing environment step() operations.
    
    Automatically logs WARN messages when step() exceeds the 10ms threshold
    as required by Section 5.4 performance requirements.
    
    Returns:
        Context manager that tracks step timing and issues warnings
        
    Example:
        >>> with create_step_timer() as metrics:
        ...     obs, reward, done, info = env.step(action)
        >>> # Automatic warning if step took >10ms
    """
    return step_performance_timer()


@contextmanager
def step_performance_timer() -> ContextManager[PerformanceMetrics]:
    """
    Context manager for environment step() performance monitoring.
    
    Tracks step latency and automatically logs structured warnings when
    the ≤10ms threshold is exceeded per Section 5.4.5.1 requirements.
    """
    context = get_correlation_context()
    metrics = context.push_performance("environment_step")
    
    try:
        yield metrics
    finally:
        completed_metrics = context.pop_performance()
        context.increment_step()
        
        if completed_metrics and completed_metrics.is_slow(PERFORMANCE_THRESHOLDS["environment_step"]):
            bound_logger = logger.bind(**context.bind_context())
            bound_logger.warning(
                f"Environment step() latency exceeded threshold: {completed_metrics.duration:.3f}s > {PERFORMANCE_THRESHOLDS['environment_step']:.3f}s",
                extra={
                    "metric_type": "step_latency_violation",
                    "operation": "environment_step",
                    "actual_latency_ms": completed_metrics.duration * 1000,
                    "threshold_latency_ms": PERFORMANCE_THRESHOLDS["environment_step"] * 1000,
                    "performance_metrics": completed_metrics.to_dict(),
                    "step_count": context.step_count,
                    "overage_percent": ((completed_metrics.duration - PERFORMANCE_THRESHOLDS["environment_step"]) / PERFORMANCE_THRESHOLDS["environment_step"]) * 100
                }
            )


@contextmanager
def correlation_context(
    operation_name: str = "operation",
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    **metadata
) -> ContextManager[CorrelationContext]:
    """
    Context manager for correlation tracking with automatic cleanup and distributed tracing.
    
    Enhanced with request_id/episode_id support for distributed tracing per Section 5.4.2.1.
    
    Args:
        operation_name: Name of the operation for logging and performance tracking
        correlation_id: Optional explicit correlation ID (generates new if None)
        request_id: Optional request ID for distributed tracing
        episode_id: Optional episode ID for RL environment tracking
        **metadata: Additional metadata to bind to the correlation context
        
    Yields:
        CorrelationContext: Context object for the operation
        
    Example:
        >>> with correlation_context("simulation_execution", episode_id="ep_001", agent_count=5) as ctx:
        ...     logger.info("Starting simulation")
        ...     # All logs within this context will include correlation_id, request_id, episode_id and metadata
    """
    # Create new context or get existing
    if correlation_id or request_id or episode_id:
        context = CorrelationContext(request_id=request_id, episode_id=episode_id)
        if correlation_id:
            context.correlation_id = correlation_id
    else:
        context = get_correlation_context()
        # Update episode_id if provided
        if episode_id:
            context.set_episode_id(episode_id)
    
    # Add operation metadata
    context.add_metadata(**metadata)
    
    # Set context for thread
    old_context = getattr(_context_storage, 'context', None)
    set_correlation_context(context)
    
    # Start performance tracking
    perf_metrics = context.push_performance(operation_name, **metadata)
    
    try:
        yield context
    finally:
        # Complete performance tracking
        completed_metrics = context.pop_performance()
        if completed_metrics and completed_metrics.is_slow():
            bound_logger = logger.bind(**context.bind_context())
            bound_logger.warning(
                f"Slow operation detected: {operation_name}",
                extra={
                    "performance_metrics": completed_metrics.to_dict(),
                    "metric_type": "slow_operation"
                }
            )


@contextmanager
def frame_rate_timer() -> ContextManager[PerformanceMetrics]:
    """
    Context manager for frame rate measurement timing.
    
    Tracks frame processing performance and logs warnings when
    frame rate falls below 30 FPS target.
    """
    context = get_correlation_context()
    metrics = context.push_performance("frame_rate_measurement")
    
    try:
        yield metrics
    finally:
        completed_metrics = context.pop_performance()
        
        if completed_metrics and completed_metrics.duration:
            fps = 1.0 / completed_metrics.duration
            target_fps = PERFORMANCE_THRESHOLDS["simulation_fps_min"]
            
            if fps < target_fps:
                bound_logger = logger.bind(**context.bind_context())
                bound_logger.warning(
                    f"Frame rate below target: {fps:.1f} FPS < {target_fps:.1f} FPS",
                    extra={
                        "metric_type": "frame_rate_violation",
                        "operation": "frame_rate_measurement",
                        "actual_fps": fps,
                        "target_fps": target_fps,
                        "frame_time_ms": completed_metrics.duration * 1000,
                        "performance_metrics": completed_metrics.to_dict()
                    }
                )


@contextmanager  
def memory_usage_timer(operation_name: str = "memory_operation") -> ContextManager[PerformanceMetrics]:
    """
    Context manager for memory usage delta tracking.
    
    Monitors memory usage changes during operations and logs
    warnings for excessive memory growth.
    
    Args:
        operation_name: Name of the operation being monitored
    """
    context = get_correlation_context()
    metrics = context.push_performance(operation_name)
    
    try:
        yield metrics
    finally:
        completed_metrics = context.pop_performance()
        
        if completed_metrics and completed_metrics.memory_delta and completed_metrics.memory_delta > 0:
            # Log warning for significant memory increases (>100MB)
            if completed_metrics.memory_delta > 100.0:
                bound_logger = logger.bind(**context.bind_context())
                bound_logger.warning(
                    f"Significant memory usage increase: +{completed_metrics.memory_delta:.1f}MB during {operation_name}",
                    extra={
                        "metric_type": "memory_usage_warning",
                        "operation": operation_name,
                        "memory_delta_mb": completed_metrics.memory_delta,
                        "memory_before_mb": completed_metrics.memory_before,
                        "memory_after_mb": completed_metrics.memory_after,
                        "performance_metrics": completed_metrics.to_dict()
                    }
                )


@contextmanager
def database_operation_timer(operation_name: str = "db_operation") -> ContextManager[PerformanceMetrics]:
    """
    Context manager for database operation performance monitoring.
    
    Tracks database operation duration and logs warnings when
    operations exceed the 100ms threshold.
    
    Args:
        operation_name: Name of the database operation
    """
    context = get_correlation_context()
    metrics = context.push_performance(operation_name)
    
    try:
        yield metrics
    finally:
        completed_metrics = context.pop_performance()
        
        if completed_metrics and completed_metrics.is_slow(PERFORMANCE_THRESHOLDS["db_operation"]):
            bound_logger = logger.bind(**context.bind_context())
            bound_logger.warning(
                f"Slow database operation: {operation_name} took {completed_metrics.duration:.3f}s",
                extra={
                    "metric_type": "db_latency_violation", 
                    "operation": operation_name,
                    "actual_latency_ms": completed_metrics.duration * 1000,
                    "threshold_latency_ms": PERFORMANCE_THRESHOLDS["db_operation"] * 1000,
                    "performance_metrics": completed_metrics.to_dict()
                }
            )


def detect_legacy_gym_import() -> bool:
    """
    Detect if legacy gym package is being used instead of gymnasium.
    
    Inspects the call stack to determine if legacy gym.make() or
    similar legacy API calls are being used.
    
    Returns:
        True if legacy gym usage is detected
    """
    import inspect
    
    # Check if 'gym' module (not 'gymnasium') is in the current stack
    for frame_info in inspect.stack():
        frame = frame_info.frame
        frame_globals = frame.f_globals
        
        # Check if 'gym' module is imported (but not gymnasium)
        if ('gym' in frame_globals and 
            hasattr(frame_globals.get('gym'), 'make') and
            'gymnasium' not in str(frame_globals.get('gym', ''))):
            return True
            
        # Check for direct gym imports in the module
        module_name = frame_globals.get('__name__', '')
        if module_name and 'gym' in module_name and 'gymnasium' not in module_name:
            return True
    
    return False


def log_legacy_api_deprecation(
    operation: str,
    legacy_call: str,
    recommended_call: str,
    migration_guide: Optional[str] = None
):
    """
    Log structured deprecation warning for legacy gym API usage.
    
    Per Section 5.4.3.2, triggers DeprecationWarning messages via
    logger.warning with structured guidance for migration to Gymnasium API.
    
    Args:
        operation: Name of the deprecated operation
        legacy_call: The legacy API call being used
        recommended_call: The recommended new API call
        migration_guide: Optional URL or text with migration instructions
    """
    import warnings
    
    context = get_correlation_context()
    bound_logger = logger.bind(**context.bind_context())
    
    # Create structured deprecation message
    message = (
        f"Legacy gym API usage detected: {legacy_call}. "
        f"Please migrate to: {recommended_call}"
    )
    
    if migration_guide:
        message += f". Migration guide: {migration_guide}"
    
    # Issue Python warning
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    
    # Log structured warning via Loguru
    bound_logger.warning(
        f"Legacy API deprecation: {operation}",
        extra={
            "metric_type": "legacy_api_deprecation",
            "operation": operation,
            "legacy_call": legacy_call,
            "recommended_call": recommended_call,
            "migration_guide": migration_guide or "https://gymnasium.farama.org/introduction/migration_guide/",
            "deprecation_category": "gym_to_gymnasium_migration"
        }
    )


def monitor_environment_creation(env_id: str, make_function: str = "gym.make"):
    """
    Monitor environment creation for legacy API usage.
    
    Automatically detects and logs deprecation warnings when legacy
    gym.make() is used instead of gymnasium.make().
    
    Args:
        env_id: Environment identifier being created
        make_function: The make function being used
    """
    if detect_legacy_gym_import() or make_function == "gym.make":
        log_legacy_api_deprecation(
            operation="environment_creation",
            legacy_call=f"gym.make('{env_id}')",
            recommended_call=f"gymnasium.make('{env_id}')",
            migration_guide="Replace 'import gym' with 'import gymnasium as gym' for drop-in compatibility"
        )
        
        # Restore previous context
        if old_context:
            set_correlation_context(old_context)
        elif hasattr(_context_storage, 'context'):
            delattr(_context_storage, 'context')


class EnhancedLogger:
    """
    Enhanced logger wrapper providing automatic context binding and performance tracking.
    
    Wraps Loguru logger with automatic correlation context binding, performance measurement
    capabilities, and structured metadata management for comprehensive observability.
    """
    
    def __init__(self, name: str, config: Optional[LoggingConfig] = None):
        self.name = name
        self.config = config or LoggingConfig()
        self._base_context = {"module": name}
    
    def _get_bound_logger(self, **extra_context):
        """Get logger bound with correlation context and module information."""
        context = get_correlation_context()
        bound_context = {
            **self._base_context,
            **context.bind_context(),
            **extra_context
        }
        return logger.bind(**bound_context)
    
    def _log_with_context(self, level: str, message: str, **kwargs):
        """Log message with automatic context binding."""
        extra = kwargs.pop('extra', {})
        bound_logger = self._get_bound_logger(**extra)
        getattr(bound_logger, level.lower())(message, **kwargs)
    
    def trace(self, message: str, **kwargs):
        """Log trace message with context."""
        self._log_with_context("TRACE", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log_with_context("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log_with_context("INFO", message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message with context."""
        self._log_with_context("SUCCESS", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log_with_context("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log_with_context("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._log_with_context("CRITICAL", message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with full traceback and context."""
        extra = kwargs.pop('extra', {})
        bound_logger = self._get_bound_logger(**extra)
        bound_logger.exception(message, **kwargs)
    
    @contextmanager
    def performance_timer(
        self, 
        operation: str, 
        threshold: Optional[float] = None,
        log_start: bool = True,
        log_completion: bool = True,
        **metadata
    ) -> ContextManager[PerformanceMetrics]:
        """
        Context manager for performance timing with automatic logging.
        
        Args:
            operation: Name of the operation being timed
            threshold: Custom threshold for slow operation detection
            log_start: Whether to log operation start
            log_completion: Whether to log operation completion
            **metadata: Additional metadata for the operation
            
        Yields:
            PerformanceMetrics: Metrics object for the operation
            
        Example:
            >>> logger = get_enhanced_logger(__name__)
            >>> with logger.performance_timer("database_query", table="experiments") as metrics:
            ...     result = execute_query()
            >>> # Automatic logging of performance metrics
        """
        context = get_correlation_context()
        metrics = context.push_performance(operation, **metadata)
        
        if log_start:
            self.debug(f"Starting operation: {operation}", extra={
                "metric_type": "operation_start",
                "operation": operation,
                **metadata
            })
        
        try:
            yield metrics
        finally:
            completed_metrics = context.pop_performance()
            
            if log_completion and completed_metrics:
                # Determine log level based on performance
                if completed_metrics.is_slow(threshold):
                    log_level = "warning"
                    message = f"Slow operation completed: {operation}"
                else:
                    log_level = "debug"
                    message = f"Operation completed: {operation}"
                
                getattr(self, log_level)(message, extra={
                    "metric_type": "operation_complete",
                    "performance_metrics": completed_metrics.to_dict(),
                    **metadata
                })
    
    def bind_experiment_metadata(self, **metadata):
        """Bind experiment metadata to correlation context."""
        context = get_correlation_context()
        context.add_metadata(**metadata)
    
    def log_performance_metrics(self, metrics: Dict[str, Any], metric_type: str = "performance"):
        """Log structured performance metrics."""
        self.info(f"Performance metrics: {metric_type}", extra={
            "metric_type": metric_type,
            "metrics": metrics
        })
    
    def log_threshold_violation(self, operation: str, actual: float, threshold: float, unit: str = "seconds"):
        """Log performance threshold violation."""
        self.warning(f"Performance threshold exceeded for {operation}", extra={
            "metric_type": "threshold_violation",
            "operation": operation,
            "actual_value": actual,
            "threshold_value": threshold,
            "unit": unit,
            "overage_percent": ((actual - threshold) / threshold) * 100
        })
    
    def log_system_health(self, component: str, status: str, **details):
        """Log system health status."""
        log_level = "info" if status.lower() == "healthy" else "warning"
        getattr(self, log_level)(f"{component} health check: {status}", extra={
            "metric_type": "health_check",
            "component": component,
            "status": status,
            **details
        })
    
    def log_step_latency_violation(self, actual_latency: float, threshold_latency: float = 0.010):
        """
        Log environment step() latency violation per Section 5.4 requirements.
        
        Args:
            actual_latency: Measured step latency in seconds
            threshold_latency: Threshold latency in seconds (default 10ms)
        """
        self.warning(
            f"Environment step() latency exceeded: {actual_latency*1000:.1f}ms > {threshold_latency*1000:.1f}ms",
            extra={
                "metric_type": "step_latency_violation",
                "actual_latency_ms": actual_latency * 1000,
                "threshold_latency_ms": threshold_latency * 1000,
                "overage_percent": ((actual_latency - threshold_latency) / threshold_latency) * 100
            }
        )
    
    def log_frame_rate_measurement(self, fps: float, target_fps: float = 30.0):
        """
        Log frame rate measurements with warnings for low performance.
        
        Args:
            fps: Measured frames per second
            target_fps: Target FPS threshold
        """
        if fps < target_fps:
            self.warning(
                f"Frame rate below target: {fps:.1f} FPS < {target_fps:.1f} FPS",
                extra={
                    "metric_type": "frame_rate_violation",
                    "actual_fps": fps,
                    "target_fps": target_fps,
                    "performance_degradation_percent": ((target_fps - fps) / target_fps) * 100
                }
            )
        else:
            self.debug(
                f"Frame rate measurement: {fps:.1f} FPS",
                extra={
                    "metric_type": "frame_rate_measurement",
                    "actual_fps": fps,
                    "target_fps": target_fps
                }
            )
    
    def log_memory_usage_delta(self, memory_delta: float, operation: str = "unknown"):
        """
        Log memory usage changes during operations.
        
        Args:
            memory_delta: Memory change in MB
            operation: Operation name causing memory change
        """
        if memory_delta > 100.0:  # Warning for >100MB increases
            self.warning(
                f"Significant memory increase: +{memory_delta:.1f}MB during {operation}",
                extra={
                    "metric_type": "memory_usage_warning",
                    "memory_delta_mb": memory_delta,
                    "operation": operation
                }
            )
        else:
            self.debug(
                f"Memory usage delta: {memory_delta:+.1f}MB for {operation}",
                extra={
                    "metric_type": "memory_usage_measurement",
                    "memory_delta_mb": memory_delta,
                    "operation": operation
                }
            )
    
    def log_database_operation_latency(self, operation: str, latency: float, threshold: float = 0.1):
        """
        Log database operation performance with warnings for slow operations.
        
        Args:
            operation: Database operation name
            latency: Operation latency in seconds
            threshold: Warning threshold in seconds
        """
        if latency > threshold:
            self.warning(
                f"Slow database operation: {operation} took {latency:.3f}s",
                extra={
                    "metric_type": "db_latency_violation",
                    "operation": operation,
                    "actual_latency_ms": latency * 1000,
                    "threshold_latency_ms": threshold * 1000
                }
            )
        else:
            self.debug(
                f"Database operation: {operation} completed in {latency:.3f}s",
                extra={
                    "metric_type": "db_operation_measurement",
                    "operation": operation,
                    "latency_ms": latency * 1000
                }
            )
    
    def log_legacy_api_usage(self, operation: str, legacy_call: str, recommended_call: str):
        """
        Log legacy API usage deprecation warnings per Section 5.4.3.2.
        
        Args:
            operation: Operation name
            legacy_call: Legacy API call being used
            recommended_call: Recommended modern API call
        """
        log_legacy_api_deprecation(
            operation=operation,
            legacy_call=legacy_call, 
            recommended_call=recommended_call
        )


def setup_logger(
    config: Optional[Union[LoggingConfig, Dict[str, Any]]] = None,
    sink: Union[str, Path, None] = None,
    level: Optional[str] = None,
    format: Optional[str] = None,
    rotation: Optional[str] = None,
    retention: Optional[str] = None,
    enqueue: Optional[bool] = None,
    backtrace: Optional[bool] = None,
    diagnose: Optional[bool] = None,
    environment: Optional[str] = None,
    **kwargs
) -> LoggingConfig:
    """
    Enhanced logger configuration with Hydra and Pydantic integration.
    
    Configures the global Loguru logger with comprehensive settings including
    environment-specific defaults, performance monitoring, correlation tracking,
    and structured output formatting.
    
    Args:
        config: LoggingConfig object or dictionary with configuration settings
        sink: Output path for log file, or None for console only (backward compatibility)
        level: Minimum log level to display (backward compatibility)
        format: Log message format (backward compatibility)
        rotation: When to rotate log files (backward compatibility)
        retention: How long to keep log files (backward compatibility)
        enqueue: Whether to enqueue log messages (backward compatibility)
        backtrace: Whether to include a backtrace for exceptions (backward compatibility)
        diagnose: Whether to diagnose exceptions (backward compatibility)
        environment: Environment type for applying defaults
        **kwargs: Additional configuration parameters
        
    Returns:
        LoggingConfig: The resolved configuration object
        
    Example:
        >>> # Using configuration object
        >>> config = LoggingConfig(environment="production", level="INFO")
        >>> setup_logger(config)
        
        >>> # Using backward-compatible parameters
        >>> setup_logger(level="DEBUG", sink="./logs/debug.log")
        
        >>> # Using environment-based defaults
        >>> setup_logger(environment="development")
    """
    # Handle different input types and backward compatibility
    if config is None:
        # Create config from individual parameters and environment defaults
        config_dict = {}
        
        # Apply environment defaults first
        if environment:
            config_dict["environment"] = environment
            env_defaults = ENVIRONMENT_DEFAULTS.get(environment, {})
            config_dict.update(env_defaults)
        
        # Override with explicit parameters (backward compatibility)
        if level is not None:
            config_dict["level"] = level
        if sink is not None:
            config_dict["file_path"] = sink
            config_dict["file_enabled"] = True
        if format is not None:
            # Map old format names to new format types
            format_mapping = {
                DEFAULT_FORMAT: "default",
                MODULE_FORMAT: "module",
            }
            config_dict["format"] = format_mapping.get(format, "custom")
        if rotation is not None:
            config_dict["rotation"] = rotation
        if retention is not None:
            config_dict["retention"] = retention
        if enqueue is not None:
            config_dict["enqueue"] = enqueue
        if backtrace is not None:
            config_dict["backtrace"] = backtrace
        if diagnose is not None:
            config_dict["diagnose"] = diagnose
        
        # Add any additional kwargs
        config_dict.update(kwargs)
        
        config = LoggingConfig(**config_dict)
    
    elif isinstance(config, dict):
        config = LoggingConfig(**config)
    
    # Apply environment defaults if not already applied
    if hasattr(config, 'apply_environment_defaults'):
        config = config.apply_environment_defaults()
    
    # Remove existing handlers
    logger.remove()
    
    # Get format pattern
    format_pattern = config.get_format_pattern()
    
    # Prepare default context for all logs with enhanced correlation tracking
    default_context = {
        "correlation_id": "none",
        "request_id": "none",
        "module": "system",
        "config_hash": "unknown",
        "step_count": 0,
        **config.default_context
    }
    
    # Configure console logging
    if config.console_enabled:
        console_format = format_pattern
        
        # Use simpler format for CLI environment
        if config.format == "cli":
            console_format = CLI_FORMAT
        elif config.format == "minimal":
            console_format = MINIMAL_FORMAT
        
        logger.add(
            sys.stderr,
            format=console_format,
            level=config.level,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
            enqueue=config.enqueue,
            filter=_create_context_filter(default_context)
        )
    
    # Configure file logging
    if config.file_enabled and config.file_path:
        # Ensure log directory exists
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_format = format_pattern
        if config.format == "json":
            file_format = _create_json_formatter()
        
        logger.add(
            str(log_path),
            format=file_format,
            level=config.level,
            rotation=config.rotation,
            retention=config.retention,
            enqueue=config.enqueue,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
            filter=_create_context_filter(default_context),
            serialize=(config.format == "json")
        )
    
    # Configure performance monitoring if enabled
    if config.enable_performance:
        _setup_performance_monitoring(config)
    
    # Log configuration completion
    startup_logger = logger.bind(**default_context)
    startup_logger.info(
        "Enhanced logging system initialized",
        extra={
            "metric_type": "system_startup",
            "environment": config.environment,
            "log_level": config.level,
            "format": config.format,
            "performance_monitoring": config.enable_performance,
            "correlation_tracking": config.correlation_enabled,
        }
    )
    
    return config


def _create_context_filter(default_context: Dict[str, Any]):
    """Create a filter function that ensures required context fields are present."""
    def context_filter(record):
        # Ensure all required context fields are present
        for key, default_value in default_context.items():
            if key not in record["extra"]:
                record["extra"][key] = default_value
        return True
    return context_filter


def _create_json_formatter():
    """Create a JSON formatter function for structured logging with enhanced correlation tracking."""
    def json_formatter(record):
        # Extract relevant fields for JSON output with enhanced correlation support
        json_record = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "correlation_id": record["extra"].get("correlation_id", "none"),
            "request_id": record["extra"].get("request_id", "none"),
            "module": record["extra"].get("module", "unknown"),
            "thread_id": record["extra"].get("thread_id"),
            "process_id": record["extra"].get("process_id"),
            "step_count": record["extra"].get("step_count", 0),
        }
        
        # Add episode_id for RL environment tracking
        if "episode_id" in record["extra"]:
            json_record["episode_id"] = record["extra"]["episode_id"]
        
        # Add performance metrics if present
        if "performance_metrics" in record["extra"]:
            json_record["performance"] = record["extra"]["performance_metrics"]
        
        # Add metric type for structured analysis
        if "metric_type" in record["extra"]:
            json_record["metric_type"] = record["extra"]["metric_type"]
        
        # Add performance timing fields for automatic capture
        performance_fields = [
            "actual_latency_ms", "threshold_latency_ms", "actual_fps", "target_fps",
            "memory_delta_mb", "memory_before_mb", "memory_after_mb",
            "overage_percent", "performance_degradation_percent"
        ]
        
        for field in performance_fields:
            if field in record["extra"]:
                json_record[field] = record["extra"][field]
        
        # Add any additional extra fields (excluding internal fields)
        for key, value in record["extra"].items():
            if key not in json_record and not key.startswith("_"):
                json_record[key] = value
        
        return json.dumps(json_record, default=str)
    
    return json_formatter


def _setup_performance_monitoring(config: LoggingConfig):
    """Setup performance monitoring and threshold checking."""
    # Log system information for performance baseline
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total": _get_total_memory(),
    }
    
    perf_logger = logger.bind(correlation_id="system_init", module="performance")
    perf_logger.info(
        "Performance monitoring enabled",
        extra={
            "metric_type": "system_baseline",
            "system_info": system_info,
            "thresholds": PERFORMANCE_THRESHOLDS,
        }
    )


def _get_total_memory() -> Optional[float]:
    """Get total system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)  # Convert to GB
    except ImportError:
        return None


def get_module_logger(name: str, config: Optional[LoggingConfig] = None) -> EnhancedLogger:
    """
    Get an enhanced logger for a specific module with automatic context binding.
    
    Args:
        name: Module name (typically __name__)
        config: Optional logging configuration
        
    Returns:
        EnhancedLogger: Enhanced logger instance with automatic context binding
        
    Example:
        >>> logger = get_module_logger(__name__)
        >>> logger.info("Module initialized")
        >>> 
        >>> # With performance timing
        >>> with logger.performance_timer("database_operation") as metrics:
        ...     result = database.query()
    """
    return EnhancedLogger(name, config)


def get_enhanced_logger(name: str, config: Optional[LoggingConfig] = None) -> EnhancedLogger:
    """
    Alias for get_module_logger for backward compatibility and clarity.
    
    Args:
        name: Module name (typically __name__)
        config: Optional logging configuration
        
    Returns:
        EnhancedLogger: Enhanced logger instance
    """
    return get_module_logger(name, config)


def get_logger(name: str) -> EnhancedLogger:
    """
    Simple factory function for getting enhanced loggers.
    
    This function provides a clean interface similar to the standard logging
    library while returning enhanced loggers with correlation support.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        EnhancedLogger: Enhanced logger instance
        
    Example:
        >>> from odor_plume_nav.utils.logging_setup import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return EnhancedLogger(name)


def create_configuration_from_hydra(hydra_config: Optional[Any] = None) -> LoggingConfig:
    """
    Create LoggingConfig from Hydra configuration with environment variable resolution.
    
    This function integrates with Hydra's configuration system to create logging
    configurations that support environment variable interpolation and hierarchical
    configuration composition.
    
    Args:
        hydra_config: Hydra configuration object (DictConfig)
        
    Returns:
        LoggingConfig: Resolved logging configuration
        
    Example:
        >>> # In a Hydra app
        >>> @hydra.main(config_path="conf", config_name="config")
        >>> def my_app(cfg: DictConfig) -> None:
        ...     log_config = create_configuration_from_hydra(cfg.logging)
        ...     setup_logger(log_config)
    """
    if hydra_config is None:
        return LoggingConfig()
    
    # Convert Hydra config to dict, handling environment variable interpolation
    try:
        from omegaconf import OmegaConf
        
        # Resolve environment variables and convert to dict
        resolved_config = OmegaConf.to_container(hydra_config, resolve=True)
        return LoggingConfig(**resolved_config)
    
    except ImportError:
        # Fallback if OmegaConf not available
        if hasattr(hydra_config, '_content'):
            return LoggingConfig(**hydra_config._content)
        else:
            return LoggingConfig(**dict(hydra_config))


def setup_performance_logging(
    enable: bool = True,
    threshold: float = 1.0,
    memory_tracking: bool = False
):
    """
    Configure performance logging with specific settings.
    
    This function provides a convenient way to enable performance monitoring
    with custom thresholds and memory tracking capabilities.
    
    Args:
        enable: Whether to enable performance logging
        threshold: Default threshold for slow operation detection (seconds)
        memory_tracking: Whether to track memory usage
        
    Example:
        >>> setup_performance_logging(enable=True, threshold=0.5, memory_tracking=True)
        >>> logger = get_logger(__name__)
        >>> with logger.performance_timer("slow_operation") as metrics:
        ...     time.sleep(1)  # Will trigger slow operation warning
    """
    config = LoggingConfig(
        enable_performance=enable,
        performance_threshold=threshold,
        memory_tracking=memory_tracking
    )
    setup_logger(config)


# Maintain backward compatibility with original function signature
def get_module_logger_legacy(name: str) -> logger:
    """
    Legacy function for backward compatibility.
    
    Returns the original Loguru logger bound with module context.
    This function is preserved for backward compatibility but the enhanced
    version (get_module_logger) is recommended for new code.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Loguru logger instance bound with module context
    """
    return logger.bind(module=name)


# Initialize with default console logging for immediate availability
_default_config = LoggingConfig(
    environment="development",
    console_enabled=True,
    file_enabled=False,
    format="default"
)

# Setup basic logging on module import
try:
    setup_logger(_default_config)
except Exception:
    # Fallback to basic setup if enhanced setup fails
    logger.remove()
    logger.add(sys.stderr, format=DEFAULT_FORMAT, level="INFO")


# Export the configuration creation function for Hydra integration
def register_logging_config_schema():
    """
    Register LoggingConfig with Hydra ConfigStore for structured configuration.
    
    This function enables automatic schema discovery and validation within Hydra's
    configuration composition system.
    """
    try:
        from hydra.core.config_store import ConfigStore
        
        cs = ConfigStore.instance()
        cs.store(
            group="logging",
            name="enhanced",
            node=LoggingConfig,
            package="logging"
        )
        
        logger.info("Successfully registered LoggingConfig schema with Hydra ConfigStore")
        
    except ImportError:
        logger.warning("Hydra not available, skipping ConfigStore registration")
    except Exception as e:
        logger.error(f"Failed to register logging configuration schema: {e}")


# Enhanced exports for comprehensive functionality
__all__ = [
    # Configuration classes
    "LoggingConfig",
    "PerformanceMetrics",
    
    # Enhanced logger classes
    "EnhancedLogger",
    "CorrelationContext",
    
    # Main setup functions
    "setup_logger",
    "get_module_logger",
    "get_enhanced_logger", 
    "get_logger",
    
    # Context managers and utilities
    "correlation_context",
    "get_correlation_context",
    "set_correlation_context",
    
    # Performance monitoring context managers
    "create_step_timer",
    "step_performance_timer",
    "frame_rate_timer",
    "memory_usage_timer",
    "database_operation_timer",
    
    # Legacy API detection and deprecation
    "detect_legacy_gym_import",
    "log_legacy_api_deprecation",
    "monitor_environment_creation",
    
    # Hydra integration
    "create_configuration_from_hydra",
    "register_logging_config_schema",
    
    # Specialized setup functions
    "setup_performance_logging",
    
    # Backward compatibility
    "get_module_logger_legacy",
    
    # Constants (preserved for backward compatibility)
    "DEFAULT_FORMAT",
    "MODULE_FORMAT",
    "ENHANCED_FORMAT",
    "HYDRA_FORMAT",
    "CLI_FORMAT",
    "MINIMAL_FORMAT",
    "PRODUCTION_FORMAT",
    "JSON_FORMAT",
    "LOG_LEVELS",
    "PERFORMANCE_THRESHOLDS",
    "ENVIRONMENT_DEFAULTS",
]


# Remove duplicate function definitions - using enhanced versions above
