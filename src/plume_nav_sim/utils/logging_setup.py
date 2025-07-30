"""
Enhanced Logging Configuration Module for Plume Navigation Simulation with Gymnasium Migration Support.

This module provides a comprehensive, configuration-driven logging setup across the plume_nav_sim 
application, using Loguru for advanced structured logging capabilities with Hydra and Pydantic 
integration. Supports environment-specific configurations, performance monitoring, frame cache 
observability, CLI debug logging integration, collaborative debugging, and legacy Gym API 
deprecation management for the migration to Gymnasium 0.29.x.

Key Features:
- Configuration-driven initialization with Hydra and Pydantic integration
- Environment-specific logging configurations (development, testing, production)
- YAML configuration loading with dual sink architecture (JSON + console)
- Frame cache statistics integration and memory pressure monitoring (≤2 GiB limit)
- Performance monitoring and diagnostic logging for real-time simulation monitoring
- Enhanced module logger creation with automatic context binding and correlation IDs
- Environment step() latency monitoring with ≤10ms threshold warnings per Section 5.4
- Structured JSON logging with distributed tracing support for operational monitoring
- Legacy Gym API deprecation detection and structured warning system for migration guidance
- Comprehensive performance timing integration (frame rate, memory, database, cache)
- PSUtil-based memory pressure monitoring for intelligent cache management

Enhanced CLI Debug Integration (Section 7.6.4.1):
- Debug command correlation tracking with sequential command numbering
- Collaborative debugging session management with host/participant modes
- Debug-specific performance thresholds for CLI operations and viewer interactions
- Debug session state tracking for interactive debugging workflows
- Performance monitoring for debug viewer launch, frame navigation, and state inspection
- Automatic performance violation detection for debug operations

Enhanced Performance Monitoring:
- Environment step() latency tracking with automatic WARN logging when >10ms threshold exceeded
- Frame rate measurement with automatic warnings below 30 FPS target for real-time performance
- Memory usage delta tracking with warnings for significant increases (>100MB)
- Database operation timing with latency violation detection (>100ms)
- Frame cache hit rate monitoring (>90% target) and memory pressure tracking
- Cache memory consumption tracking (≤2 GiB default) via PSUtil integration with LRU eviction
- Structured JSON output with correlation IDs for distributed analysis and experiment tracking

Frame Cache Integration:
- Automatic inclusion of cache statistics (hits, misses, evictions) in JSON log records
- Cache memory pressure monitoring with ResourceError category logging for operational alerts
- Cache performance metrics embedded in info["perf_stats"] for RL training integration
- Thread-safe cache operation tracking in correlation context for multi-agent scenarios
- Configurable cache modes (none, lru, all) with performance monitoring and Hydra integration

Legacy API Deprecation Support:
- Automatic detection of legacy gym imports vs gymnasium for migration guidance
- Structured DeprecationWarning messages via logger.warning with migration examples
- Migration guidance with specific code examples and documentation links for smooth transition

Example Usage:
    >>> # Enhanced correlation context with cache monitoring
    >>> with correlation_context("rl_training", episode_id="ep_001") as ctx:
    ...     logger.info("Starting RL episode")
    ...     with create_step_timer() as step_metrics:
    ...         obs, reward, terminated, truncated, info = env.step(action)
    ...     # Automatic warning if step() > 10ms, includes cache stats
    
    >>> # CLI debug command performance monitoring
    >>> with debug_command_timer("debug_viewer_launch", backend="pyside6") as metrics:
    ...     launch_debug_viewer(results, backend="pyside6")
    ...     # Automatic warning if launch > 3s threshold
    
    >>> # Debug session context with collaborative debugging
    >>> debug_ctx = create_debug_session_context("analyze_run1", results_path="results/run1")
    >>> debug_ctx.enable_collaborative_debugging("localhost", 8502, mode="host")
    >>> with debug_ctx:
    ...     log_debug_command_correlation("frame_navigation", {"frame": 245})
    ...     log_debug_session_event("breakpoint_hit", {"condition": "odor_reading > 0.8"})
    
    >>> # YAML configuration with dual sinks
    >>> setup_logger(logging_config_path="./logging.yaml")
    
    >>> # Cache statistics integration
    >>> update_cache_metrics(cache_hit_count=150, cache_miss_count=25)
    >>> logger.info("Cache operation completed")  # Includes cache stats in JSON
    
    >>> # Memory pressure monitoring
    >>> log_cache_memory_pressure_violation(1800.0, 2048.0)  # >90% usage warning
    
    >>> # Legacy API deprecation detection
    >>> monitor_environment_creation("PlumeNavSim-v0", "gym.make")  # Triggers warning
"""

import sys
import os
import json
import time
import uuid
import threading
import yaml
import platform
import warnings
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Literal, ContextManager
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, asdict
import inspect

from loguru import logger
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing_extensions import Self

# Import configuration models for integration (graceful fallback)
try:
    from plume_nav_sim.config.models import SimulationConfig
except ImportError:
    # Fallback for cases where config models aren't available yet
    SimulationConfig = None


@dataclass
class FrameCacheConfig:
    """
    Configuration for frame caching system supporting dual-mode caching with memory management.
    
    Defines parameters for LRU and full-preload cache modes with memory management,
    performance monitoring, and integration with structured logging for comprehensive
    frame cache observability in reinforcement learning environments.
    
    Attributes:
        mode: Cache operation mode (none, lru, all)
        memory_limit_gb: Maximum memory usage in GiB (default: 2.0)
        memory_pressure_threshold: Threshold ratio for memory pressure warnings (default: 0.9)
        max_entries: Maximum number of cache entries (auto-calculated if None)
        preload_enabled: Enable preloading for 'all' mode
        enable_statistics: Enable cache statistics tracking for monitoring
        log_cache_events: Log cache operations for debugging and monitoring
    """
    
    # Cache mode configuration
    mode: Literal["none", "lru", "all"] = "none"
    
    # Memory management parameters with 2 GiB default per Section 5.4.4
    memory_limit_gb: float = 2.0  # 2 GiB default memory limit
    memory_pressure_threshold: float = 0.9  # 90% threshold for memory pressure warnings
    
    # Cache sizing parameters
    max_entries: Optional[int] = None  # Maximum number of cache entries (auto-calculated if None)
    preload_enabled: bool = False  # Enable preloading for 'all' mode
    
    # Performance monitoring and observability
    enable_statistics: bool = True  # Enable cache statistics tracking
    log_cache_events: bool = True  # Log cache operations for monitoring
    
    def __post_init__(self):
        """Post-initialization validation for cache configuration."""
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
        if not (0.0 <= self.memory_pressure_threshold <= 1.0):
            raise ValueError("memory_pressure_threshold must be between 0.0 and 1.0")


# Enhanced format constants for comprehensive logging patterns
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

# Enhanced format patterns for correlation tracking and experiment traceability
ENHANCED_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>correlation_id={extra[correlation_id]}</magenta> | "
    "<yellow>request_id={extra[request_id]}</yellow> | "
    "<blue>module={extra[module]}</blue> - "
    "<level>{message}</level>"
)

HYDRA_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan> | "
    "<yellow>config_hash={extra[config_hash]}</yellow> | "
    "<magenta>correlation_id={extra[correlation_id]}</magenta> | "
    "<yellow>request_id={extra[request_id]}</yellow> - "
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

# JSON format for structured logging in production environments with machine parsing
# Use _create_json_serializer() for JSON formatting in setup_logger
JSON_FORMAT = "{time} | {level} | {name} | {message}"  # Safe fallback format without {extra}

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

# Performance monitoring thresholds (in seconds unless noted) per Section 5.4.4
PERFORMANCE_THRESHOLDS = {
    "cli_init": 2.0,
    "config_validation": 0.5,
    "db_connection": 0.5,
    "simulation_fps_min": 30.0,  # FPS, not seconds - real-time performance target
    "video_frame_processing": 0.033,  # 33ms per frame for 30 FPS
    "db_operation": 0.1,  # 100ms for typical database operations
    "environment_step": 0.010,  # 10ms per step - critical RL performance requirement per Section 5.4.4
    "frame_rate_measurement": 0.033,  # 33ms target frame rate for real-time performance
    "memory_usage_delta": 0.050,  # 50ms for memory measurement operations
    # Enhanced CLI debug operation thresholds per Section 7.6.4.1
    "debug_viewer_launch": 3.0,  # 3s for debug viewer initialization
    "debug_session_init": 1.0,  # 1s for debug session creation
    "debug_frame_navigation": 0.100,  # 100ms for frame-to-frame navigation
    "debug_state_inspection": 0.050,  # 50ms for state inspector operations
    "debug_breakpoint_eval": 0.025,  # 25ms for breakpoint condition evaluation
    "debug_export_operation": 2.0,  # 2s for debug frame/session export
    "debug_session_sharing": 1.5,  # 1.5s for collaborative session setup
    "debug_performance_analysis": 5.0,  # 5s for automated performance analysis
}

# Environment-specific logging defaults with operational configurations
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
    """
    Performance metrics tracking structure for logging correlation with frame cache monitoring.
    
    Captures comprehensive performance data including timing, memory usage, and cache statistics
    for integration with structured logging and operational monitoring systems.
    
    Attributes:
        operation_name: Name of the operation being measured
        start_time: Operation start timestamp
        end_time: Operation completion timestamp
        duration: Calculated operation duration in seconds
        memory_before: Memory usage before operation (MB)
        memory_after: Memory usage after operation (MB)
        memory_delta: Memory usage change (MB)
        thread_id: Thread identifier for concurrent operations
        correlation_id: Correlation ID for distributed tracing
        metadata: Additional operation metadata
        cache_hit_count: Total cache hits for this operation
        cache_miss_count: Total cache misses for this operation
        cache_evictions: Number of cache evictions during operation
        cache_hit_rate: Calculated cache hit rate (0.0-1.0)
        cache_memory_usage_mb: Current cache memory usage in MB
        cache_memory_limit_mb: Cache memory limit in MB
    """
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
    
    # Frame cache performance fields for monitoring integration
    cache_hit_count: Optional[int] = None
    cache_miss_count: Optional[int] = None
    cache_evictions: Optional[int] = None
    cache_hit_rate: Optional[float] = None
    cache_memory_usage_mb: Optional[float] = None
    cache_memory_limit_mb: Optional[float] = None

    def complete(self) -> Self:
        """Mark performance measurement as complete and calculate metrics."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        if self.memory_before is not None:
            self.memory_after = self._get_memory_usage()
            self.memory_delta = self.memory_after - self.memory_before
        return self

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB using PSUtil."""
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
    performance monitoring, frame cache observability, and structured correlation tracking 
    for research and production environments in the Gymnasium migration context.
    
    Attributes:
        environment: Deployment environment determining default logging behavior
        level: Base logging level with Hydra environment variable interpolation support
        format: Log message format pattern with environment variable support
        console_enabled: Enable console output with environment variable override
        file_enabled: Enable file logging with environment variable override
        file_path: Log file path supporting Hydra interpolation
        rotation: Log file rotation trigger with environment variable support
        retention: Log file retention period with environment variable support
        enable_performance: Enable performance monitoring for step timing
        performance_threshold: Slow operation threshold with environment variable support
        correlation_enabled: Enable correlation ID tracking for distributed tracing
        memory_tracking: Enable memory usage tracking for cache monitoring
        backtrace: Include backtrace in error logs for debugging
        diagnose: Enable enhanced exception diagnosis
        enqueue: Enqueue log messages for better multiprocessing support
        default_context: Default context fields to include in all log messages
    """
    
    # Environment and level configuration with Hydra interpolation support
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
    
    # Output configuration with environment variable overrides
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
    
    # File rotation and retention with environment variable support
    rotation: str = Field(
        default="10 MB",
        description="Log file rotation trigger. Supports ${oc.env:LOG_ROTATION,10 MB}"
    )
    retention: str = Field(
        default="1 week",
        description="Log file retention period. Supports ${oc.env:LOG_RETENTION,1 week}"
    )
    
    # Performance monitoring configuration for step timing and cache monitoring
    enable_performance: bool = Field(
        default=False,
        description="Enable performance monitoring. Supports ${oc.env:ENABLE_PERF_LOGGING,false}"
    )
    performance_threshold: float = Field(
        default=1.0,
        description="Slow operation threshold in seconds. Supports ${oc.env:PERF_THRESHOLD,1.0}"
    )
    
    # Correlation and tracing for distributed monitoring
    correlation_enabled: bool = Field(
        default=True,
        description="Enable correlation ID tracking. Supports ${oc.env:LOG_CORRELATION,true}"
    )
    memory_tracking: bool = Field(
        default=False,
        description="Enable memory usage tracking. Supports ${oc.env:LOG_MEMORY,false}"
    )
    
    # Advanced Loguru features
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
    
    # Context binding defaults for consistent logging
    default_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default context fields to include in all log messages"
    )
    
    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        """Validate and normalize log file path with Hydra interpolation support."""
        if v is None:
            return None
        
        if isinstance(v, str):
            # Handle Hydra environment variable interpolation
            if v.startswith('${oc.env:'):
                return v
            path = Path(v)
        else:
            path = v
        
        # Ensure directory exists (skip for interpolated paths)
        if not str(path).startswith('${'):
            path.parent.mkdir(parents=True, exist_ok=True)
        
        return str(path)
    
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
            "json": ENHANCED_FORMAT,  # Use enhanced format as fallback for JSON (handlers will override)
        }
        return format_patterns.get(self.format, ENHANCED_FORMAT)
    
    def apply_environment_defaults(self) -> "LoggingConfig":
        """Apply environment-specific defaults to unset fields."""
        env_defaults = ENVIRONMENT_DEFAULTS.get(self.environment, {})
        
        # Create a new config with environment defaults applied
        config_dict = self.model_dump()
        for key, default_value in env_defaults.items():
            # Only apply default if the field wasn't explicitly set
            if key in config_dict and hasattr(LoggingConfig.model_fields.get(key), 'default'):
                if config_dict[key] == LoggingConfig.model_fields[key].default:
                    config_dict[key] = default_value
        
        return LoggingConfig(**config_dict)


# Thread-local storage for correlation context with multi-agent support
_context_storage = threading.local()


class CorrelationContext:
    """
    Thread-local correlation context manager for experiment traceability and distributed monitoring.
    
    Maintains correlation IDs and context metadata across function calls within the same thread,
    enabling comprehensive experiment tracking, debugging, and operational monitoring. Enhanced 
    with request_id/episode_id support for distributed tracing, frame cache integration, and
    debug session tracking for collaborative debugging per Section 7.6.4.1.
    
    Attributes:
        correlation_id: Unique correlation identifier for request tracing
        request_id: Request identifier for distributed tracing
        episode_id: Episode identifier for RL environment tracking
        experiment_metadata: Additional experiment context data
        performance_stack: Stack of active performance measurements
        start_time: Context creation timestamp
        step_count: Environment step counter for performance monitoring
        debug_session_id: Debug session identifier for collaborative debugging
        debug_command_sequence: Sequence counter for debug command correlation
        debug_viewer_state: Current debug viewer state for session tracking
        collaborative_session_info: Information for shared debug sessions
    """
    
    def __init__(self, request_id: Optional[str] = None, episode_id: Optional[str] = None, 
                 debug_session_id: Optional[str] = None):
        self.correlation_id = str(uuid.uuid4())
        self.request_id = request_id or str(uuid.uuid4())
        self.episode_id = episode_id
        self.experiment_metadata = {}
        self.performance_stack = []
        self.start_time = time.time()
        self.step_count = 0  # Track environment steps for performance monitoring
        
        # Enhanced debug session tracking per Section 7.6.4.1
        self.debug_session_id = debug_session_id or str(uuid.uuid4())
        self.debug_command_sequence = 0  # Sequential counter for debug command correlation
        self.debug_viewer_state = {}  # Current debug viewer state for session tracking
        self.collaborative_session_info = {
            "session_host": None,
            "session_port": None,
            "shared_session_active": False,
            "collaboration_mode": None,  # "host", "participant", or None
            "participant_count": 0
        }
    
    def bind_context(self, **kwargs) -> Dict[str, Any]:
        """
        Get context dictionary for binding to loggers with enhanced distributed tracing, cache statistics,
        and debug session tracking per Section 7.6.4.1.
        
        Args:
            **kwargs: Additional context parameters to include
            
        Returns:
            Dict containing complete context for logger binding including cache statistics and debug session info
        """
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
        
        # Add debug session tracking information per Section 7.6.4.1
        context.update({
            "debug_session_id": self.debug_session_id,
            "debug_command_sequence": self.debug_command_sequence,
            "collaborative_session_active": self.collaborative_session_info["shared_session_active"],
            "collaboration_mode": self.collaborative_session_info["collaboration_mode"]
        })
        
        # Add collaborative session info when active
        if self.collaborative_session_info["shared_session_active"]:
            context.update({
                "session_host": self.collaborative_session_info["session_host"],
                "session_port": self.collaborative_session_info["session_port"],
                "participant_count": self.collaborative_session_info["participant_count"]
            })
        
        # Add cache statistics if available from current performance stack
        if self.performance_stack:
            latest_metrics = self.performance_stack[-1]
            if latest_metrics.cache_hit_count is not None:
                context.update({
                    "cache_hit_count": latest_metrics.cache_hit_count,
                    "cache_miss_count": latest_metrics.cache_miss_count,
                    "cache_evictions": latest_metrics.cache_evictions,
                    "cache_hit_rate": latest_metrics.cache_hit_rate,
                    "cache_memory_usage_mb": latest_metrics.cache_memory_usage_mb,
                    "cache_memory_limit_mb": latest_metrics.cache_memory_limit_mb
                })
            
        return context
    
    def add_metadata(self, **metadata):
        """Add metadata to correlation context for experiment tracking."""
        self.experiment_metadata.update(metadata)
    
    def push_performance(self, operation: str, **metadata) -> PerformanceMetrics:
        """Start tracking performance for an operation with memory monitoring."""
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
    
    def increment_debug_command(self, command_name: str, command_args: Optional[Dict[str, Any]] = None) -> int:
        """
        Increment debug command sequence counter for command correlation tracking.
        
        This method enables CLI debug command correlation per Section 7.6.4.1 by
        providing sequential tracking of debug operations within a session.
        
        Args:
            command_name: Name of the debug command being executed
            command_args: Optional arguments passed to the debug command
            
        Returns:
            Current debug command sequence number
        """
        self.debug_command_sequence += 1
        
        # Log debug command correlation for session tracking
        if hasattr(self, '_logger_context'):
            logger.bind(**self.bind_context()).debug(
                f"Debug command correlation: {command_name}",
                extra={
                    "metric_type": "debug_command_correlation",
                    "debug_command": command_name,
                    "command_sequence": self.debug_command_sequence,
                    "command_args": command_args or {},
                    "debug_session_id": self.debug_session_id
                }
            )
        
        return self.debug_command_sequence
    
    def update_debug_viewer_state(self, **state_updates):
        """
        Update debug viewer state for session tracking and collaborative debugging.
        
        Args:
            **state_updates: Key-value pairs representing debug viewer state changes
        """
        self.debug_viewer_state.update(state_updates)
        
        # Add timestamp for state change tracking
        self.debug_viewer_state["last_updated"] = time.time()
        
        # Log viewer state update for collaborative session synchronization
        if self.collaborative_session_info["shared_session_active"]:
            logger.bind(**self.bind_context()).debug(
                "Debug viewer state updated in collaborative session",
                extra={
                    "metric_type": "debug_viewer_state_update",
                    "state_updates": state_updates,
                    "collaboration_mode": self.collaborative_session_info["collaboration_mode"],
                    "participant_count": self.collaborative_session_info["participant_count"]
                }
            )
    
    def enable_collaborative_debugging(self, host: str, port: int, mode: str = "host"):
        """
        Enable collaborative debugging session sharing per Section 7.6.4.1.
        
        Args:
            host: Host address for collaborative session
            port: Port number for collaborative session
            mode: Collaboration mode ("host" or "participant")
        """
        self.collaborative_session_info.update({
            "session_host": host,
            "session_port": port,
            "shared_session_active": True,
            "collaboration_mode": mode
        })
        
        logger.bind(**self.bind_context()).info(
            f"Collaborative debugging session enabled as {mode}",
            extra={
                "metric_type": "collaborative_session_enabled",
                "session_host": host,
                "session_port": port,
                "collaboration_mode": mode,
                "debug_session_id": self.debug_session_id
            }
        )
    
    def disable_collaborative_debugging(self):
        """Disable collaborative debugging session."""
        old_mode = self.collaborative_session_info["collaboration_mode"]
        
        self.collaborative_session_info.update({
            "session_host": None,
            "session_port": None,
            "shared_session_active": False,
            "collaboration_mode": None,
            "participant_count": 0
        })
        
        logger.bind(**self.bind_context()).info(
            f"Collaborative debugging session disabled (was {old_mode})",
            extra={
                "metric_type": "collaborative_session_disabled",
                "previous_mode": old_mode,
                "debug_session_id": self.debug_session_id
            }
        )
    
    def update_participant_count(self, count: int):
        """Update participant count for collaborative debugging session."""
        self.collaborative_session_info["participant_count"] = count
        
        if self.collaborative_session_info["shared_session_active"]:
            logger.bind(**self.bind_context()).debug(
                f"Collaborative session participant count updated: {count}",
                extra={
                    "metric_type": "participant_count_update",
                    "participant_count": count,
                    "debug_session_id": self.debug_session_id
                }
            )


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
    as required by Section 5.4.4 performance requirements for real-time operation.
    
    Returns:
        Context manager that tracks step timing and issues warnings
        
    Example:
        >>> with create_step_timer() as metrics:
        ...     obs, reward, terminated, truncated, info = env.step(action)
        >>> # Automatic warning if step took >10ms
    """
    return step_performance_timer()


@contextmanager
def step_performance_timer() -> ContextManager[PerformanceMetrics]:
    """
    Context manager for environment step() performance monitoring.
    
    Tracks step latency and automatically logs structured warnings when
    the ≤10ms threshold is exceeded per Section 5.4.4 requirements for
    real-time simulation performance.
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
    
    Enhanced with request_id/episode_id support for distributed tracing and experiment
    reproducibility in the Gymnasium migration context.
    
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
        
        # Restore previous context
        if old_context:
            set_correlation_context(old_context)
        elif hasattr(_context_storage, 'context'):
            delattr(_context_storage, 'context')


@contextmanager
def frame_rate_timer() -> ContextManager[PerformanceMetrics]:
    """
    Context manager for frame rate measurement timing.
    
    Tracks frame processing performance and logs warnings when
    frame rate falls below 30 FPS target for real-time operation.
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
def debug_command_timer(command_name: str, **metadata) -> ContextManager[PerformanceMetrics]:
    """
    Context manager for CLI debug command performance monitoring per Section 7.6.4.1.
    
    Tracks debug command execution time and logs performance violations when
    commands exceed their defined thresholds for responsive debugging experience.
    
    Args:
        command_name: Name of the debug command being executed
        **metadata: Additional metadata to include in performance tracking
        
    Returns:
        Context manager that tracks debug command timing and issues warnings
        
    Example:
        >>> with debug_command_timer("debug_viewer_launch", backend="pyside6") as metrics:
        ...     launch_debug_viewer(results, backend="pyside6")
        >>> # Automatic warning if command took >3s threshold
    """
    context = get_correlation_context()
    
    # Increment debug command sequence for correlation tracking
    command_sequence = context.increment_debug_command(command_name, metadata)
    
    # Start performance tracking with debug-specific operation name
    metrics = context.push_performance(f"debug_{command_name}", **metadata)
    
    try:
        yield metrics
    finally:
        completed_metrics = context.pop_performance()
        
        if completed_metrics and completed_metrics.is_slow():
            threshold = PERFORMANCE_THRESHOLDS.get(f"debug_{command_name}", 
                                                 PERFORMANCE_THRESHOLDS.get(command_name, 1.0))
            
            bound_logger = logger.bind(**context.bind_context())
            bound_logger.warning(
                f"Debug command exceeded threshold: {command_name} took {completed_metrics.duration:.3f}s > {threshold:.3f}s",
                extra={
                    "metric_type": "debug_performance_violation",
                    "debug_command": command_name,
                    "command_sequence": command_sequence,
                    "actual_duration_ms": completed_metrics.duration * 1000,
                    "threshold_duration_ms": threshold * 1000,
                    "performance_metrics": completed_metrics.to_dict(),
                    "overage_percent": ((completed_metrics.duration - threshold) / threshold) * 100
                }
            )


@contextmanager
def debug_session_timer(session_operation: str, **metadata) -> ContextManager[PerformanceMetrics]:
    """
    Context manager for debug session operation performance monitoring.
    
    Tracks debug session-specific operations like initialization, state updates,
    and collaborative session management with appropriate performance thresholds.
    
    Args:
        session_operation: Type of debug session operation being performed
        **metadata: Additional metadata for the session operation
        
    Returns:
        Context manager that tracks session operation timing
    """
    context = get_correlation_context()
    metrics = context.push_performance(f"debug_session_{session_operation}", **metadata)
    
    try:
        yield metrics
    finally:
        completed_metrics = context.pop_performance()
        
        # Check against debug session thresholds
        threshold_key = f"debug_{session_operation}"
        threshold = PERFORMANCE_THRESHOLDS.get(threshold_key, 1.0)
        
        if completed_metrics and completed_metrics.duration > threshold:
            bound_logger = logger.bind(**context.bind_context())
            bound_logger.warning(
                f"Debug session operation exceeded threshold: {session_operation}",
                extra={
                    "metric_type": "debug_session_performance_violation",
                    "session_operation": session_operation,
                    "actual_duration_ms": completed_metrics.duration * 1000,
                    "threshold_duration_ms": threshold * 1000,
                    "performance_metrics": completed_metrics.to_dict(),
                    "collaborative_session_active": context.collaborative_session_info["shared_session_active"]
                }
            )


def detect_legacy_gym_import() -> bool:
    """
    Detect if legacy gym package is being used instead of gymnasium.
    
    Inspects the call stack to determine if legacy gym.make() or
    similar legacy API calls are being used, supporting the migration
    from OpenAI Gym 0.26 to Gymnasium 0.29.x.
    
    Returns:
        True if legacy gym usage is detected
    """
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
    logger.warning with structured guidance for migration to Gymnasium API
    in the context of the plume_nav_sim migration from Gym 0.26 to Gymnasium 0.29.x.
    
    Args:
        operation: Name of the deprecated operation
        legacy_call: The legacy API call being used
        recommended_call: The recommended new API call
        migration_guide: Optional URL or text with migration instructions
    """
    context = get_correlation_context()
    bound_logger = logger.bind(**context.bind_context())
    
    # Create structured deprecation message
    message = (
        f"Legacy gym API usage detected: {legacy_call}. "
        f"Please migrate to: {recommended_call}"
    )
    
    if migration_guide:
        message += f". Migration guide: {migration_guide}"
    
    # Issue Python warning for compatibility
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
    gym.make() is used instead of gymnasium.make() for plume_nav_sim environments.
    
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


class EnhancedLogger:
    """
    Enhanced logger wrapper providing automatic context binding and performance tracking.
    
    Wraps Loguru logger with automatic correlation context binding, performance measurement
    capabilities, and structured metadata management for comprehensive observability in
    the plume_nav_sim environment with frame cache and Gymnasium migration support.
    
    Attributes:
        name: Logger name (typically module name)
        config: Logging configuration object
        _base_context: Base context fields for all log messages
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
    def performance_timer(self, operation: str, threshold: Optional[float] = None):
        """Context manager for performance timing with automatic logging."""
        start_time = time.time()
        metrics = PerformanceMetrics(operation_name=operation, start_time=start_time)
        
        self.info(f"Starting operation: {operation}")
        
        try:
            yield metrics
        finally:
            end_time = time.time()
            metrics.end_time = end_time
            metrics = metrics.complete()
            
            if threshold and metrics.is_slow(threshold):
                self.warning(f"Slow operation completed: {operation}", 
                           performance_metrics=metrics.to_dict(),
                           operation_complete=True)
            else:
                self.info(f"Operation completed: {operation}", 
                         performance_metrics=metrics.to_dict(),
                         operation_complete=True)
    
    def log_legacy_api_usage(self, operation: str, legacy_call: str, recommended_call: str):
        """Log usage of legacy API with deprecation warning."""
        message = f"Legacy API usage detected in {operation}: {legacy_call} → {recommended_call}"
        
        warnings.warn(
            f"DEPRECATION WARNING: {message}. Please migrate to the recommended call.",
            DeprecationWarning,
            stacklevel=2
        )
        
        self.warning(f"Legacy API usage: {operation}", 
                    legacy_call=legacy_call,
                    recommended_call=recommended_call,
                    deprecation_warning=True)
    
    def log_step_latency_violation(self, actual_time: float, threshold: float):
        """Log environment step latency violations."""
        self.warning(f"Step latency violation: {actual_time:.3f}s > {threshold:.3f}s threshold",
                    actual_latency=actual_time,
                    threshold=threshold,
                    violation_type="step_latency")
    
    def log_frame_rate_measurement(self, actual_fps: float, target_fps: float):
        """Log frame rate measurements."""
        if actual_fps < target_fps:
            self.warning(f"Frame rate below target: {actual_fps:.1f} FPS < {target_fps:.1f} FPS target",
                        actual_fps=actual_fps,
                        target_fps=target_fps,
                        performance_warning=True)
        else:
            self.info(f"Frame rate nominal: {actual_fps:.1f} FPS",
                     actual_fps=actual_fps,
                     target_fps=target_fps)
    
    def log_memory_usage_delta(self, delta_mb: float, operation: str, threshold: float = 100.0):
        """Log memory usage deltas with threshold warnings."""
        if delta_mb > threshold:
            self.warning(f"High memory usage increase: {delta_mb:.1f}MB in {operation}",
                        memory_delta_mb=delta_mb,
                        operation=operation,
                        threshold=threshold,
                        memory_warning=True)
        else:
            self.debug(f"Memory usage delta: {delta_mb:.1f}MB in {operation}",
                      memory_delta_mb=delta_mb,
                      operation=operation)
    
    def log_database_operation_latency(self, operation: str, duration: float, threshold: float):
        """Log database operation latency with threshold monitoring."""
        if duration > threshold:
            self.warning(f"Slow database operation: {operation} took {duration:.3f}s",
                        operation=operation,
                        duration_s=duration,
                        threshold_s=threshold,
                        db_latency_violation=True)
        else:
            self.debug(f"Database operation completed: {operation} in {duration:.3f}s",
                      operation=operation,
                      duration_s=duration)


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
    logging_config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> LoggingConfig:
    """
    Enhanced logger configuration with Hydra and Pydantic integration.
    
    Configures the global Loguru logger with comprehensive settings including
    environment-specific defaults, performance monitoring, correlation tracking,
    structured output formatting, and dual sink architecture supporting JSON and
    console outputs via logging.yaml configuration files.
    
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
        logging_config_path: Path to logging.yaml configuration file for dual sink architecture
        **kwargs: Additional configuration parameters
        
    Returns:
        LoggingConfig: The resolved configuration object
        
    Example:
        >>> # Using logging.yaml configuration
        >>> setup_logger(logging_config_path="./logging.yaml")
        
        >>> # Using configuration object
        >>> config = LoggingConfig(environment="production", level="INFO")
        >>> setup_logger(config)
        
        >>> # Using backward-compatible parameters
        >>> setup_logger(level="DEBUG", sink="./logs/debug.log")
        
        >>> # Using environment-based defaults
        >>> setup_logger(environment="development")
    """
    # Load configuration from logging.yaml if provided
    yaml_config = None
    if logging_config_path is not None:
        yaml_config = _load_logging_yaml(logging_config_path)
    
    # Handle different input types and backward compatibility
    if config is None:
        # Create config from individual parameters and environment defaults
        config_dict = {}
        
        # Apply YAML configuration first if available
        if yaml_config:
            config_dict.update(yaml_config)
        
        # Apply environment defaults second
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
            # Map old format names to new format types and validate format parameter
            format_mapping = {
                DEFAULT_FORMAT: "default",
                MODULE_FORMAT: "module",
            }
            
            # Valid format literals from LoggingConfig
            valid_formats = {"default", "module", "enhanced", "hydra", "cli", "minimal", "production", "json"}
            
            # Use format_mapping for constants, direct value for valid strings, otherwise default to enhanced
            if format in format_mapping:
                config_dict["format"] = format_mapping[format]
            elif format in valid_formats:
                config_dict["format"] = format
            else:
                config_dict["format"] = "enhanced"
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
        if config.format == "json":
            # Use custom JSON formatter for structured console logging
            logger.add(
                sys.stderr,
                format=_create_json_serializer(),
                level=config.level,
                backtrace=config.backtrace,
                diagnose=config.diagnose,
                enqueue=config.enqueue,
                filter=_create_context_filter(default_context)
            )
        else:
            # Use standard text format for console
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
        
        if config.format == "json":
            # Use custom JSON formatter for structured logging
            logger.add(
                str(log_path),
                format=_create_json_serializer(),
                level=config.level,
                rotation=config.rotation,
                retention=config.retention,
                enqueue=config.enqueue,
                backtrace=config.backtrace,
                diagnose=config.diagnose,
                filter=_create_context_filter(default_context)
            )
        else:
            # Use standard text format
            logger.add(
                str(log_path),
                format=format_pattern,
                level=config.level,
                rotation=config.rotation,
                retention=config.retention,
                enqueue=config.enqueue,
                backtrace=config.backtrace,
                diagnose=config.diagnose,
                filter=_create_context_filter(default_context)
            )
    
    # Configure dual sink architecture from YAML if available
    logger.debug(f"YAML config available: {yaml_config is not None}")
    if yaml_config and "sinks" in yaml_config:
        logger.debug(f"Found YAML sinks configuration: {list(yaml_config['sinks'].keys())}")
        _setup_yaml_sinks(yaml_config["sinks"], default_context)
    else:
        logger.debug("No YAML sinks configuration found, using default setup only")
    
    # Configure performance monitoring if enabled
    if config.enable_performance:
        _setup_performance_monitoring(config)
    
    # Log configuration completion
    startup_logger = logger.bind(**default_context)
    startup_logger.info(
        "Enhanced logging system initialized for plume_nav_sim",
        extra={
            "metric_type": "system_startup",
            "environment": config.environment,
            "log_level": config.level,
            "format": config.format,
            "performance_monitoring": config.enable_performance,
            "correlation_tracking": config.correlation_enabled,
            "gymnasium_migration_support": True,
        }
    )
    
    return config


def _create_context_filter(default_context: Dict[str, Any]):
    """Create a filter function that ensures required context fields are present and merges correlation context."""
    def context_filter(record):
        # Merge current correlation context if available
        current_context = getattr(_context_storage, 'context', None)
        if current_context:
            correlation_data = current_context.bind_context()
            for key, value in correlation_data.items():
                if key not in record["extra"]:
                    record["extra"][key] = value
        
        # Ensure all required context fields are present
        for key, default_value in default_context.items():
            if key not in record["extra"]:
                record["extra"][key] = default_value
        return True
    return context_filter


def _create_json_serializer():
    """Create a JSON serializer function for structured logging with cache statistics, enhanced correlation tracking, and debug session support."""
    def json_serializer(record):
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
            "thread_id": record["thread"].id,
            "process_id": record["process"].id,
            "step_count": record["extra"].get("step_count", 0),
        }
        
        # Add episode_id for RL environment tracking
        if "episode_id" in record["extra"]:
            json_record["episode_id"] = record["extra"]["episode_id"]
        
        # Add debug session tracking fields per Section 7.6.4.1
        debug_fields = [
            "debug_session_id", "debug_command_sequence", "collaborative_session_active",
            "collaboration_mode", "session_host", "session_port", "participant_count"
        ]
        
        debug_info = {}
        for field in debug_fields:
            if field in record["extra"]:
                debug_info[field] = record["extra"][field]
        
        if debug_info:
            json_record["debug_session"] = debug_info
        
        # Add performance metrics if present
        if "performance_metrics" in record["extra"]:
            json_record["performance"] = record["extra"]["performance_metrics"]
        
        # Add metric type for structured analysis
        if "metric_type" in record["extra"]:
            json_record["metric_type"] = record["extra"]["metric_type"]
        
        # Add cache statistics for frame cache monitoring
        cache_fields = [
            "cache_hit_count", "cache_miss_count", "cache_evictions",
            "cache_hit_rate", "cache_memory_usage_mb", "cache_memory_limit_mb"
        ]
        
        cache_stats = {}
        for field in cache_fields:
            if field in record["extra"]:
                cache_stats[field] = record["extra"][field]
        
        if cache_stats:
            json_record["cache_stats"] = cache_stats
        
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
            if key not in json_record and not key.startswith("_") and key not in cache_fields:
                json_record[key] = value
        
        # Generate JSON string 
        json_str = json.dumps(json_record, default=str)
        
        # Escape any characters that Loguru might interpret as markup or format strings
        # Replace angle brackets and curly braces
        escaped_json = (json_str
                       .replace("<", "&lt;")
                       .replace(">", "&gt;")
                       .replace("{", "{{")
                       .replace("}", "}}"))
        
        return escaped_json + "\n"
    
    return json_serializer


def _resolve_environment_variables(value: Any) -> Any:
    """
    Recursively resolve environment variables in YAML configuration values.
    
    Supports syntax: ${VAR_NAME:default_value}
    
    Args:
        value: Configuration value that may contain environment variables
        
    Returns:
        Value with environment variables resolved
    """
    if isinstance(value, str):
        # Handle ${VAR:default} syntax
        import re
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name, default_value = var_expr, ''
            return os.getenv(var_name, default_value)
        
        # Replace all ${VAR:default} patterns
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, value)
    
    elif isinstance(value, dict):
        return {k: _resolve_environment_variables(v) for k, v in value.items()}
    
    elif isinstance(value, list):
        return [_resolve_environment_variables(item) for item in value]
    
    else:
        return value


def _load_logging_yaml(config_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load logging configuration from YAML file with validation and error handling.
    
    Args:
        config_path: Path to logging.yaml configuration file
        
    Returns:
        Dictionary containing logging configuration or None if loading fails
        
    Example logging.yaml structure:
        sinks:
          console:
            level: INFO
            format: enhanced
          json_file:
            level: DEBUG
            format: json
            file_path: "./logs/structured.log"
            rotation: "10 MB"
            retention: "1 week"
    """
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Logging configuration file not found: {config_path}")
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # Validate required structure
        if not isinstance(yaml_config, dict):
            logger.error(f"Invalid YAML configuration structure in {config_path}")
            return None
        
        # Resolve environment variables
        yaml_config = _resolve_environment_variables(yaml_config)
        
        logger.info(f"Successfully loaded logging configuration from {config_path}")
        return yaml_config
        
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML logging configuration: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load logging configuration from {config_path}: {e}")
        return None


def _setup_yaml_sinks(sinks_config: Dict[str, Any], default_context: Dict[str, Any]):
    """
    Setup dual sink architecture from YAML configuration for JSON and console outputs.
    
    Args:
        sinks_config: Sink configuration dictionary from YAML
        default_context: Default context fields for filtering
    """
    logger.debug(f"=== _setup_yaml_sinks called with {len(sinks_config)} sinks: {list(sinks_config.keys())}")
    try:
        for sink_name, sink_config in sinks_config.items():
            logger.debug(f"Processing sink: {sink_name}")
            if not isinstance(sink_config, dict):
                logger.warning(f"Invalid sink configuration for {sink_name}")
                continue
            
            # Skip if sink is disabled
            if not sink_config.get("enabled", True):
                logger.debug(f"Skipping disabled sink {sink_name}")
                continue
            
            # Extract sink parameters with defaults
            level = sink_config.get("level", "INFO")
            format_type = sink_config.get("format", "default")
            
            # Determine sink target
            if sink_name == "console" or sink_config.get("target") == "console":
                sink_target = sys.stderr
            elif "sink" in sink_config:  # Handle 'sink' field from YAML
                sink_target = sink_config["sink"]
                if sink_target != "sys.stderr":
                    # It's a file path, ensure directory exists
                    Path(sink_target).parent.mkdir(parents=True, exist_ok=True)
                else:
                    sink_target = sys.stderr
            elif "file_path" in sink_config:
                sink_target = sink_config["file_path"]
                # Ensure directory exists
                Path(sink_target).parent.mkdir(parents=True, exist_ok=True)
            else:
                logger.warning(f"No valid target specified for sink {sink_name}")
                continue
            
            # Determine format - always use proper formatter functions, never raw format strings
            # Enhanced detection for JSON format with explicit protection against raw format strings
            
            # Handle YAML literal block scalars that may contain JSON format strings
            actual_format_content = format_type
            if isinstance(format_type, str):
                # Remove YAML literal block scalar indicators and get actual content
                cleaned_format = format_type.strip()
                if cleaned_format.startswith('|') or cleaned_format.startswith('>'):
                    # Extract content after YAML scalar indicator
                    lines = cleaned_format.split('\n')[1:]  # Skip first line with indicator
                    actual_format_content = '\n'.join(line.strip() for line in lines if line.strip())
            
            is_json_format = (
                format_type == "json" or 
                sink_name == "json" or 
                (isinstance(actual_format_content, str) and actual_format_content.strip().startswith("{"))
            )
            
            if is_json_format:
                # Use custom JSON formatter for structured logging
                json_formatter = _create_json_serializer()
                format_func = json_formatter
                serialize = False  # Don't use serialize when using custom format function
                logger.debug(f"Using JSON formatter for sink {sink_name} (detected: {format_type[:50] if isinstance(format_type, str) else format_type}...)")
                logger.debug(f"JSON formatter type: {type(json_formatter)}")
            else:
                format_patterns = {
                    "default": DEFAULT_FORMAT,
                    "enhanced": ENHANCED_FORMAT,
                    "minimal": MINIMAL_FORMAT,
                    "production": PRODUCTION_FORMAT,
                }
                format_func = format_patterns.get(format_type, ENHANCED_FORMAT)
                serialize = False
                logger.debug(f"Using format pattern {format_type} for sink {sink_name}")
            
            # Prepare sink configuration - rotation/retention only for file sinks
            sink_kwargs = {
                "format": format_func,
                "level": level,
                "enqueue": sink_config.get("enqueue", True),
                "backtrace": sink_config.get("backtrace", True),
                "diagnose": sink_config.get("diagnose", True),
                "filter": _create_context_filter(default_context),
            }
            
            # Only add rotation/retention for file sinks, not console sinks
            if sink_target != sys.stderr and sink_target != sys.stdout:
                sink_kwargs["rotation"] = sink_config.get("rotation", "10 MB")
                sink_kwargs["retention"] = sink_config.get("retention", "1 week")
                if "compression" in sink_config:
                    sink_kwargs["compression"] = sink_config["compression"]
            
            # Add sink with configuration
            logger.debug(f"About to add sink '{sink_name}' with target: {sink_target}, serialize: {serialize}")
            logger.debug(f"Sink kwargs: {sink_kwargs}")
            
            try:
                handler_id = logger.add(sink_target, **sink_kwargs)
                logger.debug(f"Successfully added {sink_name} sink with handler ID {handler_id} and level {level}")
            except Exception as e:
                logger.error(f"Failed to add sink '{sink_name}': {e}")
                logger.exception("Sink addition error details:")
            
    except Exception as e:
        logger.error(f"Failed to setup YAML sinks: {e}")


def _setup_performance_monitoring(config: LoggingConfig):
    """Setup performance monitoring and threshold checking with cache memory pressure tracking."""
    # Log system information for performance baseline
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total": _get_total_memory(),
    }
    
    perf_logger = logger.bind(correlation_id="system_init", module="performance")
    perf_logger.info(
        "Performance monitoring enabled with cache memory tracking for plume_nav_sim",
        extra={
            "metric_type": "system_baseline",
            "system_info": system_info,
            "thresholds": PERFORMANCE_THRESHOLDS,
            "cache_memory_monitoring": True,
            "gymnasium_migration": True,
        }
    )
    
    # Perform initial memory pressure check
    _monitor_cache_memory_pressure()


def _get_total_memory() -> Optional[float]:
    """Get total system memory in GB."""
    try:
        return psutil.virtual_memory().total / (1024**3)  # Convert to GB
    except ImportError:
        return None


def _monitor_cache_memory_pressure() -> bool:
    """
    Monitor cache memory pressure using psutil and log warnings when approaching limits.
    
    This function implements ResourceError category logging when memory usage approaches
    the 2 GiB default limit, enabling proactive cache management and system stability.
    
    Returns:
        True if memory pressure detected (>90% usage)
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        memory_limit_mb = 2048  # 2 GiB default limit per Section 5.4.4
        
        memory_usage_ratio = memory_usage_mb / memory_limit_mb
        
        if memory_usage_ratio > 0.9:  # 90% threshold
            logger.bind(
                metric_type="memory_pressure_warning",
                memory_usage_mb=memory_usage_mb,
                memory_limit_mb=memory_limit_mb,
                memory_usage_ratio=memory_usage_ratio,
                resource_category="cache_memory"
            ).warning(
                f"Cache memory pressure detected: {memory_usage_mb:.1f}MB / {memory_limit_mb}MB ({memory_usage_ratio:.1%})"
            )
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to monitor memory pressure: {e}")
        return False


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
        >>> with correlation_context("database_operation") as ctx:
        ...     logger.info("Database operation started")
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
        >>> from plume_nav_sim.utils.logging_setup import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return EnhancedLogger(name)


def update_cache_metrics(
    context: Optional[CorrelationContext] = None,
    cache_hit_count: Optional[int] = None,
    cache_miss_count: Optional[int] = None,
    cache_evictions: Optional[int] = None,
    cache_memory_usage_mb: Optional[float] = None,
    cache_memory_limit_mb: Optional[float] = None
) -> None:
    """
    Update cache metrics in the current correlation context for logging integration.
    
    This function enables cache implementations to report statistics that will be
    automatically included in JSON log records and correlation context for
    frame cache monitoring and memory pressure tracking.
    
    Args:
        context: Optional correlation context (uses current thread context if None)
        cache_hit_count: Total number of cache hits
        cache_miss_count: Total number of cache misses  
        cache_evictions: Total number of cache evictions
        cache_memory_usage_mb: Current cache memory usage in MB
        cache_memory_limit_mb: Cache memory limit in MB
        
    Example:
        >>> # Update cache statistics for logging
        >>> update_cache_metrics(
        ...     cache_hit_count=150,
        ...     cache_miss_count=25,
        ...     cache_memory_usage_mb=512.5
        ... )
        >>> logger.info("Cache operation completed")  # Will include cache stats
    """
    if context is None:
        context = get_correlation_context()
    
    if context.performance_stack:
        current_metrics = context.performance_stack[-1]
        
        if cache_hit_count is not None:
            current_metrics.cache_hit_count = cache_hit_count
        if cache_miss_count is not None:
            current_metrics.cache_miss_count = cache_miss_count
        if cache_evictions is not None:
            current_metrics.cache_evictions = cache_evictions
        if cache_memory_usage_mb is not None:
            current_metrics.cache_memory_usage_mb = cache_memory_usage_mb
        if cache_memory_limit_mb is not None:
            current_metrics.cache_memory_limit_mb = cache_memory_limit_mb
        
        # Calculate hit rate if both hits and misses are available
        if (current_metrics.cache_hit_count is not None and 
            current_metrics.cache_miss_count is not None):
            total_requests = current_metrics.cache_hit_count + current_metrics.cache_miss_count
            if total_requests > 0:
                current_metrics.cache_hit_rate = current_metrics.cache_hit_count / total_requests


def log_cache_memory_pressure_violation(
    current_usage_mb: float,
    limit_mb: float,
    threshold_ratio: float = 0.9
) -> None:
    """
    Log cache memory pressure violation with ResourceError category.
    
    This function implements the specification requirement for ResourceError category
    logging when cache memory usage approaches the 2 GiB limit per Section 5.4.4.
    
    Args:
        current_usage_mb: Current memory usage in MB
        limit_mb: Memory limit in MB
        threshold_ratio: Threshold ratio for pressure warnings (default 0.9 = 90%)
    """
    usage_ratio = current_usage_mb / limit_mb
    
    if usage_ratio >= threshold_ratio:
        context = get_correlation_context()
        bound_logger = logger.bind(**context.bind_context())
        
        bound_logger.warning(
            f"Cache memory pressure violation: {current_usage_mb:.1f}MB / {limit_mb:.1f}MB ({usage_ratio:.1%})",
            extra={
                "metric_type": "memory_pressure_violation",
                "resource_category": "cache_memory",
                "current_usage_mb": current_usage_mb,
                "limit_mb": limit_mb,
                "usage_ratio": usage_ratio,
                "threshold_ratio": threshold_ratio,
                "requires_cache_clear": usage_ratio >= 0.95  # Suggest cache.clear() at 95%
            }
        )


def log_debug_command_correlation(
    command_name: str,
    command_args: Optional[Dict[str, Any]] = None,
    result_status: str = "success",
    execution_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log debug command execution with correlation tracking per Section 7.6.4.1.
    
    This function provides comprehensive logging for CLI debug commands with correlation
    IDs, command sequencing, and execution context for debugging session tracking
    and collaborative debugging support.
    
    Args:
        command_name: Name of the debug command executed
        command_args: Arguments passed to the debug command
        result_status: Execution result status ("success", "warning", "error")
        execution_context: Additional execution context (e.g., file paths, frame ranges)
        
    Example:
        >>> log_debug_command_correlation(
        ...     "debug_viewer_launch",
        ...     command_args={"backend": "pyside6", "results_path": "results/run1"},
        ...     result_status="success",
        ...     execution_context={"frame_count": 1500, "duration_seconds": 45.2}
        ... )
    """
    context = get_correlation_context()
    
    # Increment debug command sequence for correlation
    command_sequence = context.increment_debug_command(command_name, command_args)
    
    # Determine log level based on result status
    log_level = {
        "success": "info",
        "warning": "warning", 
        "error": "error"
    }.get(result_status, "info")
    
    bound_logger = logger.bind(**context.bind_context())
    log_method = getattr(bound_logger, log_level)
    
    log_method(
        f"Debug command executed: {command_name}",
        extra={
            "metric_type": "debug_command_execution",
            "debug_command": command_name,
            "command_sequence": command_sequence,
            "command_args": command_args or {},
            "result_status": result_status,
            "execution_context": execution_context or {},
            "debug_session_id": context.debug_session_id,
            "collaborative_session_active": context.collaborative_session_info["shared_session_active"]
        }
    )


def create_debug_session_context(
    session_name: str,
    results_path: Optional[str] = None,
    collaborative_mode: Optional[str] = None,
    session_config: Optional[Dict[str, Any]] = None
) -> CorrelationContext:
    """
    Create a new correlation context specifically for debug session tracking.
    
    This function creates an enhanced correlation context with debug session
    capabilities for collaborative debugging and session state management
    per Section 7.6.4.1.
    
    Args:
        session_name: Human-readable name for the debug session
        results_path: Path to simulation results being debugged
        collaborative_mode: Collaboration mode ("host", "participant", or None)
        session_config: Debug session configuration parameters
        
    Returns:
        CorrelationContext: Enhanced context with debug session capabilities
        
    Example:
        >>> debug_context = create_debug_session_context(
        ...     "analyze_run1", 
        ...     results_path="results/run1",
        ...     collaborative_mode="host",
        ...     session_config={"backend": "pyside6", "auto_analyze": True}
        ... )
        >>> with debug_context:
        ...     # All operations within this context will be correlated
        ...     launch_debug_viewer()
    """
    # Create new correlation context with debug session ID
    debug_session_id = f"debug_{session_name}_{int(time.time())}"
    context = CorrelationContext(debug_session_id=debug_session_id)
    
    # Add debug session metadata
    context.add_metadata(
        debug_session_name=session_name,
        results_path=results_path,
        session_config=session_config or {},
        session_created_at=time.time()
    )
    
    # Configure collaboration if specified
    if collaborative_mode and collaborative_mode in ["host", "participant"]:
        # Note: Actual host/port would be configured when enable_collaborative_debugging is called
        context.collaborative_session_info["collaboration_mode"] = collaborative_mode
    
    # Log debug session creation
    bound_logger = logger.bind(**context.bind_context())
    bound_logger.info(
        f"Debug session created: {session_name}",
        extra={
            "metric_type": "debug_session_created",
            "session_name": session_name,
            "debug_session_id": debug_session_id,
            "results_path": results_path,
            "collaborative_mode": collaborative_mode,
            "session_config": session_config or {}
        }
    )
    
    return context


def log_debug_session_event(
    event_type: str,
    event_data: Dict[str, Any],
    session_context: Optional[CorrelationContext] = None
) -> None:
    """
    Log debug session events for collaborative debugging and session tracking.
    
    This function provides structured logging for debug session events such as
    breakpoint hits, frame navigation, state inspection, and collaboration events
    per Section 7.6.4.1.
    
    Args:
        event_type: Type of debug session event
        event_data: Event-specific data and parameters
        session_context: Optional debug session context (uses current if None)
        
    Example:
        >>> log_debug_session_event(
        ...     "breakpoint_hit",
        ...     {
        ...         "condition": "odor_reading > 0.8",
        ...         "frame_number": 245,
        ...         "agent_id": "agent_0",
        ...         "odor_value": 0.92
        ...     }
        ... )
    """
    context = session_context or get_correlation_context()
    bound_logger = logger.bind(**context.bind_context())
    
    # Increment debug command sequence for event correlation
    event_sequence = context.increment_debug_command(f"session_event_{event_type}", event_data)
    
    bound_logger.info(
        f"Debug session event: {event_type}",
        extra={
            "metric_type": "debug_session_event",
            "event_type": event_type,
            "event_sequence": event_sequence,
            "event_data": event_data,
            "debug_session_id": context.debug_session_id,
            "collaborative_session_active": context.collaborative_session_info["shared_session_active"],
            "timestamp": time.time()
        }
    )


def register_logging_config_schema():
    """
    Register LoggingConfig with Hydra ConfigStore for structured configuration.
    
    This function enables automatic schema discovery and validation within Hydra's
    configuration composition system for the plume_nav_sim project.
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
        
        # Register FrameCacheConfig for cache configuration support
        cs.store(
            group="cache",
            name="frame_cache",
            node=FrameCacheConfig,
            package="cache"
        )
        
        logger.info("Successfully registered LoggingConfig and FrameCacheConfig schemas with Hydra ConfigStore")
        
    except ImportError:
        logger.warning("Hydra not available, skipping ConfigStore registration")
    except Exception as e:
        logger.error(f"Failed to register logging configuration schema: {e}")


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


@contextmanager
def memory_usage_timer(operation: str, threshold_mb: float = 100.0):
    """Context manager for memory usage monitoring with automatic warnings."""
    try:
        import psutil
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        start_memory = 0
    
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        try:
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - start_memory
            duration = end_time - start_time
            
            context = get_correlation_context()
            bound_logger = logger.bind(**context.bind_context())
            
            if memory_delta > threshold_mb:
                bound_logger.warning(
                    f"High memory usage in {operation}: {memory_delta:.1f}MB increase",
                    memory_delta_mb=memory_delta,
                    operation=operation,
                    duration_s=duration,
                    threshold_mb=threshold_mb,
                    memory_warning=True
                )
            else:
                bound_logger.debug(
                    f"Memory usage in {operation}: {memory_delta:.1f}MB",
                    memory_delta_mb=memory_delta,
                    operation=operation,
                    duration_s=duration
                )
        except Exception as e:
            logger.debug(f"Failed to monitor memory for {operation}: {e}")


@contextmanager
def database_operation_timer(operation: str, threshold_s: float = 0.1):
    """Context manager for database operation timing with automatic warnings."""
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        context = get_correlation_context()
        bound_logger = logger.bind(**context.bind_context())
        
        if duration > threshold_s:
            bound_logger.warning(
                f"Slow database operation: {operation} took {duration:.3f}s",
                operation=operation,
                duration_s=duration,
                threshold_s=threshold_s,
                db_latency_violation=True
            )
        else:
            bound_logger.debug(
                f"Database operation: {operation} completed in {duration:.3f}s",
                operation=operation,
                duration_s=duration
            )


def create_configuration_from_hydra(hydra_config):
    """Create LoggingConfig from Hydra configuration object."""
    try:
        # Extract logging configuration from Hydra config
        if hasattr(hydra_config, 'logging'):
            logging_cfg = hydra_config.logging
        else:
            # Fallback to defaults if no logging config
            return LoggingConfig()
        
        # Convert Hydra config to LoggingConfig
        config_dict = {}
        
        # Map common fields
        if hasattr(logging_cfg, 'level'):
            config_dict['level'] = logging_cfg.level
        if hasattr(logging_cfg, 'format'):
            config_dict['format'] = logging_cfg.format
        if hasattr(logging_cfg, 'file_path'):
            config_dict['file_path'] = logging_cfg.file_path
        if hasattr(logging_cfg, 'enable_performance'):
            config_dict['enable_performance'] = logging_cfg.enable_performance
        if hasattr(logging_cfg, 'memory_tracking'):
            config_dict['memory_tracking'] = logging_cfg.memory_tracking
        
        return LoggingConfig(**config_dict)
        
    except Exception as e:
        logger.warning(f"Failed to create LoggingConfig from Hydra config: {e}")
        return LoggingConfig()


# Enhanced exports for comprehensive functionality
__all__ = [
    # Configuration classes
    "LoggingConfig",
    "PerformanceMetrics",
    "FrameCacheConfig",
    
    # Enhanced logger classes
    "EnhancedLogger",
    "CorrelationContext",
    
    # Main setup functions
    "setup_logger",
    "get_module_logger",
    "get_enhanced_logger", 
    "get_logger",
    "create_configuration_from_hydra",
    
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
    
    # Enhanced CLI debug performance monitoring per Section 7.6.4.1
    "debug_command_timer",
    "debug_session_timer",
    
    # Cache monitoring and integration functions
    "update_cache_metrics",
    "log_cache_memory_pressure_violation",
    
    # Debug session management and correlation tracking per Section 7.6.4.1
    "log_debug_command_correlation",
    "create_debug_session_context",
    "log_debug_session_event",
    
    # Legacy API detection and deprecation
    "detect_legacy_gym_import",
    "log_legacy_api_deprecation",
    "monitor_environment_creation",
    
    # YAML configuration support
    "_load_logging_yaml",
    "_setup_yaml_sinks",
    "_monitor_cache_memory_pressure",
    
    # Hydra integration
    "register_logging_config_schema",
    
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