"""
Central logging configuration module for plume_nav_sim providing system-wide logging setup,
component-specific logger management, performance monitoring configuration, and development-focused
debugging capabilities.

This module implements standardized logging architecture with security filtering, formatter
configuration, and hierarchical logger organization following Python logging best practices
for reinforcement learning environment development and research workflows.
"""

# External imports with version comments
import logging  # >=3.10 - Standard Python logging module for logger creation and configuration
import logging.config  # >=3.10 - Advanced logging configuration management and dictConfig setup
import os  # >=3.10 - Environment variable access for development/production configuration
import sys  # >=3.10 - System parameters for stdout/stderr configuration and platform detection
from pathlib import Path  # >=3.10 - Path handling for log directory creation and cross-platform compatibility
import threading  # >=3.10 - Thread synchronization for thread-safe logger configuration
from typing import Dict, List, Optional, Union, Tuple, Any  # >=3.10 - Type hints for configuration parameters
from enum import Enum  # >=3.10 - Enumeration definitions for log levels and component types
from dataclasses import dataclass, field  # >=3.10 - Data classes for configuration structures
import warnings  # >=3.10 - Warning management for logging configuration issues
import re  # >=3.10 - Regular expressions for sensitive information filtering
import time  # >=3.10 - Timing measurements for performance logging
import gc  # >=3.10 - Garbage collection for memory management
from functools import wraps  # >=3.10 - Decorator utilities for performance monitoring
import json  # >=3.10 - JSON serialization for configuration data

# Internal imports
from ..plume_nav_sim.core.constants import (
    PERFORMANCE_TARGET_STEP_LATENCY_MS,  # Performance target for timing threshold configuration
    PACKAGE_NAME  # System package identifier for logger hierarchy
)

# Global logging configuration constants
LOGGER_NAME_PREFIX = 'plume_nav_sim'
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEVELOPMENT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
PERFORMANCE_LOG_FORMAT = '%(asctime)s - PERF - %(name)s - %(message)s [%(duration_ms).3fms]'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE_NAME_FORMAT = 'plume_nav_sim_%Y%m%d.log'
PERFORMANCE_LOG_NAME = 'plume_nav_sim_performance.log'

# Default logging levels
DEFAULT_LOG_LEVEL = logging.INFO
DEVELOPMENT_LOG_LEVEL = logging.DEBUG
PERFORMANCE_LOG_LEVEL = logging.DEBUG

# Global state management
_logging_initialized = False
_logger_cache = {}
_logger_manager = None
_config_lock = threading.Lock()

# Security and filtering patterns
SENSITIVE_PATTERNS = ['password', 'token', 'api_key', 'secret', 'auth', 'credential']

# File management constants
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
DEFAULT_LOG_DIR = Path('./logs')

# Component names derived from system architecture
COMPONENT_NAMES = [
    'ENVIRONMENT',      # PlumeSearchEnv and environment management
    'PLUME_MODEL',      # Static Gaussian plume and concentration field
    'RENDERING',        # RGB array and matplotlib visualization
    'ACTION_PROCESSOR', # Agent action processing and validation
    'REWARD_CALCULATOR', # Goal detection and reward computation
    'STATE_MANAGER',    # Episode and agent state coordination
    'BOUNDARY_ENFORCER', # Grid boundary validation and constraints
    'EPISODE_MANAGER',  # Episode lifecycle and termination management
    'UTILS'            # Utility components including seeding and validation
]

# LOG_LEVEL_DEFAULT derived from environment configuration
LOG_LEVEL_DEFAULT = 'INFO'


class LogLevel(Enum):
    """
    Enumeration defining supported logging levels with usage guidelines and component-specific
    recommendations for plume_nav_sim logging hierarchy management.
    """
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'
    
    def get_numeric_level(self) -> int:
        """
        Returns numeric logging level value compatible with Python logging module level hierarchy.
        
        Returns:
            int: Numeric logging level value for use with logging module configuration
        """
        level_mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        return level_mapping[self]
    
    @classmethod
    def from_string(cls, level_string: str) -> 'LogLevel':
        """
        Creates LogLevel instance from string representation with case-insensitive parsing.
        
        Args:
            level_string: String representation of log level
            
        Returns:
            LogLevel: LogLevel instance corresponding to string representation
            
        Raises:
            ValueError: If level_string does not match any LogLevel value
        """
        try:
            return cls(level_string.upper())
        except ValueError:
            valid_levels = [level.value for level in cls]
            raise ValueError(f"Invalid log level '{level_string}'. Valid levels: {valid_levels}")


class ComponentType(Enum):
    """
    Enumeration defining plume_nav_sim system components for specialized logging configuration,
    hierarchy management, and component-specific log level settings.
    """
    ENVIRONMENT = 'ENVIRONMENT'
    PLUME_MODEL = 'PLUME_MODEL'
    RENDERING = 'RENDERING'
    ACTION_PROCESSOR = 'ACTION_PROCESSOR'
    REWARD_CALCULATOR = 'REWARD_CALCULATOR'
    STATE_MANAGER = 'STATE_MANAGER'
    BOUNDARY_ENFORCER = 'BOUNDARY_ENFORCER'
    EPISODE_MANAGER = 'EPISODE_MANAGER'
    UTILS = 'UTILS'
    
    def get_default_log_level(self) -> LogLevel:
        """
        Returns appropriate default logging level for component type based on criticality.
        
        Returns:
            LogLevel: Default logging level appropriate for component type
        """
        critical_components = {
            ComponentType.ENVIRONMENT,
            ComponentType.PLUME_MODEL,
            ComponentType.ACTION_PROCESSOR,
            ComponentType.REWARD_CALCULATOR,
            ComponentType.STATE_MANAGER
        }
        
        if self in critical_components:
            return LogLevel.DEBUG  # Detailed debugging for critical components
        elif self in {ComponentType.RENDERING, ComponentType.UTILS}:
            return LogLevel.INFO  # Informational for utility components
        else:
            return LogLevel.WARNING  # Warnings for validation components
    
    def requires_performance_logging(self) -> bool:
        """
        Determines if component type requires specialized performance logging and timing measurements.
        
        Returns:
            bool: True if component requires performance logging, False otherwise
        """
        performance_critical = {
            ComponentType.ENVIRONMENT,      # Step latency critical
            ComponentType.PLUME_MODEL,      # Concentration sampling performance
            ComponentType.RENDERING,        # Visualization performance targets
            ComponentType.ACTION_PROCESSOR, # Processing performance
            ComponentType.STATE_MANAGER     # State update performance
        }
        return self in performance_critical


@dataclass
class LoggingConfig:
    """
    Data class containing comprehensive logging configuration parameters with validation,
    serialization, and dictConfig conversion capabilities for flexible logging setup management.
    """
    log_level: LogLevel = LogLevel.INFO
    console_logging_enabled: bool = True
    file_logging_enabled: bool = True
    performance_logging_enabled: bool = False
    log_directory_path: Path = field(default_factory=lambda: DEFAULT_LOG_DIR)
    log_format_string: str = DEFAULT_LOG_FORMAT
    date_format_string: str = DEFAULT_DATE_FORMAT
    sensitive_information_patterns: List[str] = field(default_factory=lambda: SENSITIVE_PATTERNS.copy())
    component_specific_levels: Dict[ComponentType, LogLevel] = field(default_factory=dict)
    color_console_output: bool = True
    max_log_file_size: int = MAX_LOG_FILE_SIZE
    log_backup_count: int = LOG_BACKUP_COUNT
    
    def __post_init__(self):
        """Initialize component-specific levels with defaults if not provided."""
        if not self.component_specific_levels:
            self.component_specific_levels = {
                component: component.get_default_log_level()
                for component in ComponentType
            }
    
    def validate(self, check_permissions: bool = True) -> Tuple[bool, List[str]]:
        """
        Validates logging configuration for parameter correctness and resource availability.
        
        Args:
            check_permissions: Whether to check file system permissions
            
        Returns:
            Tuple[bool, List[str]]: Tuple of (is_valid, validation_errors) with detailed results
        """
        validation_errors = []
        
        # Validate log level
        if not isinstance(self.log_level, LogLevel):
            validation_errors.append("log_level must be LogLevel enumeration instance")
        
        # Validate log directory
        if self.file_logging_enabled:
            try:
                self.log_directory_path.mkdir(parents=True, exist_ok=True)
                if check_permissions:
                    # Test write permissions
                    test_file = self.log_directory_path / '.test_write'
                    test_file.touch()
                    test_file.unlink()
            except (OSError, PermissionError) as e:
                validation_errors.append(f"Cannot create or write to log directory: {e}")
        
        # Validate log format string
        try:
            test_record = logging.LogRecord(
                name='test', level=logging.INFO, pathname='', lineno=0,
                msg='test', args=(), exc_info=None
            )
            logging.Formatter(self.log_format_string).format(test_record)
        except (ValueError, TypeError) as e:
            validation_errors.append(f"Invalid log format string: {e}")
        
        # Validate sensitive patterns
        for pattern in self.sensitive_information_patterns:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                validation_errors.append(f"Invalid regex pattern '{pattern}': {e}")
        
        # Validate component levels
        for component, level in self.component_specific_levels.items():
            if not isinstance(component, ComponentType):
                validation_errors.append(f"Invalid component type: {component}")
            if not isinstance(level, LogLevel):
                validation_errors.append(f"Invalid log level for {component}: {level}")
        
        # Validate file size and backup parameters
        if self.max_log_file_size <= 0:
            validation_errors.append("max_log_file_size must be positive")
        if self.log_backup_count < 0:
            validation_errors.append("log_backup_count must be non-negative")
        
        return len(validation_errors) == 0, validation_errors
    
    def to_dict_config(self) -> Dict[str, Any]:
        """
        Converts LoggingConfig to Python logging dictConfig format.
        
        Returns:
            Dict[str, Any]: Dictionary configuration suitable for logging.config.dictConfig
        """
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': self.log_format_string,
                    'datefmt': self.date_format_string
                },
                'development': {
                    'format': DEVELOPMENT_LOG_FORMAT,
                    'datefmt': self.date_format_string
                },
                'performance': {
                    'format': PERFORMANCE_LOG_FORMAT,
                    'datefmt': self.date_format_string
                }
            },
            'handlers': {},
            'loggers': {},
            'root': {
                'level': self.log_level.get_numeric_level(),
                'handlers': []
            }
        }
        
        # Console handler
        if self.console_logging_enabled:
            config['handlers']['console'] = {
                'class': 'logging.StreamHandler',
                'level': self.log_level.get_numeric_level(),
                'formatter': 'development' if self.log_level == LogLevel.DEBUG else 'standard',
                'stream': 'ext://sys.stdout'
            }
            config['root']['handlers'].append('console')
        
        # File handler
        if self.file_logging_enabled:
            log_file = self.log_directory_path / time.strftime(LOG_FILE_NAME_FORMAT)
            config['handlers']['file'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': logging.DEBUG,
                'formatter': 'development',
                'filename': str(log_file),
                'maxBytes': self.max_log_file_size,
                'backupCount': self.log_backup_count,
                'encoding': 'utf-8'
            }
            config['root']['handlers'].append('file')
        
        # Performance handler
        if self.performance_logging_enabled:
            perf_file = self.log_directory_path / PERFORMANCE_LOG_NAME
            config['handlers']['performance'] = {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': logging.DEBUG,
                'formatter': 'performance',
                'filename': str(perf_file),
                'maxBytes': self.max_log_file_size,
                'backupCount': self.log_backup_count,
                'encoding': 'utf-8'
            }
        
        # Component-specific loggers
        for component, level in self.component_specific_levels.items():
            logger_name = f"{LOGGER_NAME_PREFIX}.{component.value.lower()}"
            config['loggers'][logger_name] = {
                'level': level.get_numeric_level(),
                'propagate': True
            }
        
        return config
    
    def update_component_level(self, component_type: ComponentType, new_level: LogLevel) -> bool:
        """
        Updates logging level for specific component type.
        
        Args:
            component_type: Component to update
            new_level: New logging level
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if not isinstance(component_type, ComponentType):
                return False
            if not isinstance(new_level, LogLevel):
                return False
                
            self.component_specific_levels[component_type] = new_level
            return True
        except Exception:
            return False
    
    def add_sensitive_pattern(self, pattern: str, description: str = "") -> bool:
        """
        Adds regex pattern for sensitive information detection and filtering.
        
        Args:
            pattern: Regular expression pattern
            description: Optional description for documentation
            
        Returns:
            bool: True if pattern added successfully, False if invalid regex
        """
        try:
            re.compile(pattern, re.IGNORECASE)
            if pattern not in self.sensitive_information_patterns:
                self.sensitive_information_patterns.append(pattern)
            return True
        except re.error:
            return False


class SensitiveInfoFilter(logging.Filter):
    """Logging filter to remove sensitive information from log messages."""
    
    def __init__(self, sensitive_patterns: List[str]):
        """
        Initialize filter with sensitive information patterns.
        
        Args:
            sensitive_patterns: List of regex patterns to filter
        """
        super().__init__()
        self.compiled_patterns = []
        for pattern in sensitive_patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                # Skip invalid patterns
                continue
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record to remove sensitive information.
        
        Args:
            record: Log record to filter
            
        Returns:
            bool: True to keep record, False to filter out
        """
        try:
            # Check message content
            if hasattr(record, 'msg') and record.msg:
                message = str(record.msg)
                for pattern in self.compiled_patterns:
                    if pattern.search(message):
                        record.msg = pattern.sub('[FILTERED]', message)
            
            # Check arguments
            if hasattr(record, 'args') and record.args:
                filtered_args = []
                for arg in record.args:
                    arg_str = str(arg)
                    for pattern in self.compiled_patterns:
                        if pattern.search(arg_str):
                            arg_str = pattern.sub('[FILTERED]', arg_str)
                    filtered_args.append(arg_str)
                record.args = tuple(filtered_args)
                
            return True
        except Exception:
            # If filtering fails, still allow the record through
            return True


class ComponentLogger:
    """
    Specialized logger wrapper for plume_nav_sim components with performance tracking
    and component-specific configuration.
    """
    
    def __init__(self, component_type: ComponentType, logger: logging.Logger, 
                 enable_performance: bool = False):
        """
        Initialize component logger with specialized capabilities.
        
        Args:
            component_type: Type of component this logger serves
            logger: Underlying Python logger instance
            enable_performance: Whether to enable performance tracking
        """
        self.component_type = component_type
        self.logger = logger
        self.enable_performance = enable_performance
        self.performance_data = {}
        
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with component context."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message with component context."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with component context."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message with component context."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message with component context."""
        self.logger.critical(msg, *args, **kwargs)
    
    def performance(self, operation: str, duration_ms: float, **context):
        """
        Log performance measurement with operation context.
        
        Args:
            operation: Name of operation being measured
            duration_ms: Execution time in milliseconds
            **context: Additional context information
        """
        if self.enable_performance:
            perf_msg = f"{self.component_type.value} - {operation}"
            extra = {'duration_ms': duration_ms, **context}
            
            # Store performance data
            if operation not in self.performance_data:
                self.performance_data[operation] = []
            self.performance_data[operation].append(duration_ms)
            
            # Log with performance format
            perf_logger = logging.getLogger(f"{self.logger.name}.performance")
            perf_logger.debug(perf_msg, extra=extra)
    
    def time_operation(self, operation_name: str):
        """
        Decorator for timing operations automatically.
        
        Args:
            operation_name: Name of operation to time
            
        Returns:
            Decorator function for timing operations
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self.performance(operation_name, duration_ms, status='success')
                    return result
                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self.performance(operation_name, duration_ms, status='error', error=str(e))
                    raise
            return wrapper
        return decorator


class LoggerFactory:
    """
    Factory class for creating and caching component-specific loggers with automatic
    configuration, performance tracking, and centralized logger management.
    """
    
    def __init__(self, config: LoggingConfig, enable_caching: bool = True):
        """
        Initialize LoggerFactory with configuration and caching settings.
        
        Args:
            config: Logging configuration for factory operations
            enable_caching: Whether to enable logger instance caching
        """
        self.logging_config = config
        self.logger_cache = {}
        self.caching_enabled = enable_caching
        self.factory_lock = threading.Lock()
        self.performance_loggers = {}
        self.component_configurations = {}
        self.creation_count = 0
        self.cache_hits = 0
        
        # Configure sensitive information filtering
        self.sensitive_filter = SensitiveInfoFilter(config.sensitive_information_patterns)
    
    def create_logger(self, name: str, component_type: ComponentType, 
                     log_level: Optional[LogLevel] = None) -> ComponentLogger:
        """
        Creates component-specific logger with automatic configuration and caching.
        
        Args:
            name: Logger name identifier
            component_type: Type of component for specialized configuration
            log_level: Optional override for component log level
            
        Returns:
            ComponentLogger: Configured component logger instance
        """
        with self.factory_lock:
            # Check cache first
            cache_key = f"{name}:{component_type.value}"
            if self.caching_enabled and cache_key in self.logger_cache:
                self.cache_hits += 1
                return self.logger_cache[cache_key]
            
            # Create hierarchical logger name
            full_name = f"{LOGGER_NAME_PREFIX}.{component_type.value.lower()}.{name}"
            
            # Determine log level
            effective_level = log_level or self.logging_config.component_specific_levels.get(
                component_type, component_type.get_default_log_level()
            )
            
            # Create Python logger
            python_logger = logging.getLogger(full_name)
            python_logger.setLevel(effective_level.get_numeric_level())
            
            # Add sensitive information filter
            python_logger.addFilter(self.sensitive_filter)
            
            # Create component logger wrapper
            enable_performance = (
                self.logging_config.performance_logging_enabled and 
                component_type.requires_performance_logging()
            )
            
            component_logger = ComponentLogger(
                component_type=component_type,
                logger=python_logger,
                enable_performance=enable_performance
            )
            
            # Cache if enabled
            if self.caching_enabled:
                self.logger_cache[cache_key] = component_logger
            
            # Track creation
            self.creation_count += 1
            self.component_configurations[cache_key] = {
                'component_type': component_type,
                'log_level': effective_level,
                'performance_enabled': enable_performance,
                'created_at': time.time()
            }
            
            return component_logger
    
    def create_performance_logger(self, operation_name: str, timing_threshold_ms: float, 
                                enable_memory_tracking: bool = False) -> ComponentLogger:
        """
        Creates specialized performance logger with timing and memory tracking.
        
        Args:
            operation_name: Name of operation being monitored
            timing_threshold_ms: Threshold for performance warnings
            enable_memory_tracking: Whether to enable memory usage tracking
            
        Returns:
            ComponentLogger: Specialized performance logger
        """
        logger_name = f"performance.{operation_name}"
        perf_logger = self.create_logger(
            name=logger_name,
            component_type=ComponentType.UTILS,
            log_level=LogLevel.DEBUG
        )
        
        # Store in performance loggers registry
        self.performance_loggers[operation_name] = {
            'logger': perf_logger,
            'threshold_ms': timing_threshold_ms,
            'memory_tracking': enable_memory_tracking,
            'operation_count': 0,
            'total_time_ms': 0.0,
            'max_time_ms': 0.0,
            'min_time_ms': float('inf')
        }
        
        return perf_logger
    
    def get_cached_logger(self, logger_name: str, component_type: ComponentType) -> Optional[ComponentLogger]:
        """
        Retrieves cached logger instance or returns None if not cached.
        
        Args:
            logger_name: Name of logger to retrieve
            component_type: Component type for cache key
            
        Returns:
            Optional[ComponentLogger]: Cached logger or None
        """
        if not self.caching_enabled:
            return None
            
        cache_key = f"{logger_name}:{component_type.value}"
        return self.logger_cache.get(cache_key)
    
    def clear_cache(self, close_loggers: bool = False) -> int:
        """
        Clears logger cache and optionally closes cached loggers.
        
        Args:
            close_loggers: Whether to close logger handlers
            
        Returns:
            int: Number of cached loggers cleared
        """
        with self.factory_lock:
            cleared_count = len(self.logger_cache)
            
            if close_loggers:
                for component_logger in self.logger_cache.values():
                    # Close handlers if any
                    for handler in component_logger.logger.handlers[:]:
                        handler.close()
                        component_logger.logger.removeHandler(handler)
            
            self.logger_cache.clear()
            self.performance_loggers.clear()
            self.component_configurations.clear()
            
            return cleared_count
    
    def get_factory_statistics(self) -> Dict[str, Any]:
        """
        Returns comprehensive factory statistics and performance metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing factory statistics
        """
        with self.factory_lock:
            cache_hit_rate = (self.cache_hits / max(1, self.creation_count + self.cache_hits)) * 100
            
            return {
                'total_loggers_created': self.creation_count,
                'cache_hits': self.cache_hits,
                'cache_hit_rate_percent': cache_hit_rate,
                'cached_loggers_count': len(self.logger_cache),
                'performance_loggers_count': len(self.performance_loggers),
                'component_configurations': len(self.component_configurations),
                'caching_enabled': self.caching_enabled,
                'memory_usage_estimate_kb': (
                    len(self.logger_cache) * 1024 +  # Rough estimate
                    len(self.performance_loggers) * 512 +
                    len(self.component_configurations) * 256
                ) / 1024
            }


# Global factory instance
_global_factory: Optional[LoggerFactory] = None


def configure_logging(log_level: Union[str, LogLevel] = LogLevel.INFO,
                     enable_console_logging: bool = True,
                     enable_file_logging: bool = True,
                     enable_performance_logging: bool = False,
                     log_directory: Optional[str] = None,
                     override_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Primary function for configuring the plume_nav_sim logging system with handlers,
    formatters, and component-specific settings.
    
    Args:
        log_level: Base logging level for the system
        enable_console_logging: Whether to enable console output
        enable_file_logging: Whether to enable log file output
        enable_performance_logging: Whether to enable performance monitoring
        log_directory: Optional directory for log files
        override_config: Optional configuration overrides
        
    Returns:
        bool: True if logging configuration successful, False otherwise
    """
    global _logging_initialized, _global_factory
    
    with _config_lock:
        try:
            # Prevent duplicate configuration
            if _logging_initialized:
                logging.getLogger(LOGGER_NAME_PREFIX).warning(
                    "Logging already initialized, skipping duplicate configuration"
                )
                return True
            
            # Normalize log level
            if isinstance(log_level, str):
                log_level = LogLevel.from_string(log_level)
            
            # Create log directory if needed
            log_dir = Path(log_directory) if log_directory else DEFAULT_LOG_DIR
            if enable_file_logging:
                log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create logging configuration
            config = LoggingConfig(
                log_level=log_level,
                console_logging_enabled=enable_console_logging,
                file_logging_enabled=enable_file_logging,
                performance_logging_enabled=enable_performance_logging,
                log_directory_path=log_dir
            )
            
            # Apply configuration overrides
            if override_config:
                for key, value in override_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Validate configuration
            is_valid, errors = config.validate()
            if not is_valid:
                print(f"Logging configuration validation failed: {errors}", file=sys.stderr)
                return False
            
            # Apply logging configuration
            dict_config = config.to_dict_config()
            logging.config.dictConfig(dict_config)
            
            # Create global factory
            _global_factory = LoggerFactory(config, enable_caching=True)
            
            # Mark as initialized
            _logging_initialized = True
            
            # Log successful initialization
            root_logger = logging.getLogger(LOGGER_NAME_PREFIX)
            root_logger.info(f"Logging system initialized successfully with level {log_level.value}")
            
            return True
            
        except Exception as e:
            print(f"Failed to configure logging: {e}", file=sys.stderr)
            return False


def get_logger(name: str, component_type: ComponentType = ComponentType.UTILS,
               enable_performance_tracking: bool = False,
               logger_config: Optional[Dict[str, Any]] = None) -> ComponentLogger:
    """
    Factory function for creating or retrieving component-specific loggers with
    automatic configuration and performance tracking capabilities.
    
    Args:
        name: Logger name identifier
        component_type: Component type for specialized configuration
        enable_performance_tracking: Whether to enable performance tracking
        logger_config: Optional logger-specific configuration
        
    Returns:
        ComponentLogger: Configured component logger instance
    """
    global _global_factory
    
    # Initialize logging if not already done
    if not _logging_initialized:
        configure_logging()
    
    # Use global factory or create temporary one
    if _global_factory is None:
        _global_factory = LoggerFactory(LoggingConfig())
    
    # Check cache first
    cached_logger = _global_factory.get_cached_logger(name, component_type)
    if cached_logger is not None:
        return cached_logger
    
    # Create new logger
    return _global_factory.create_logger(name, component_type)


def configure_development_logging(enable_verbose_output: bool = True,
                                enable_color_console: bool = True,
                                log_to_file: bool = True,
                                development_log_level: str = 'DEBUG') -> Dict[str, Any]:
    """
    Configures enhanced logging for development environment with detailed debugging
    information, performance monitoring, and enhanced error context.
    
    Args:
        enable_verbose_output: Whether to enable detailed debug output
        enable_color_console: Whether to enable colored console output
        log_to_file: Whether to enable file logging for development
        development_log_level: Log level for development mode
        
    Returns:
        Dict[str, Any]: Dictionary containing development logging configuration status
    """
    try:
        # Configure enhanced development logging
        log_level = LogLevel.from_string(development_log_level)
        
        success = configure_logging(
            log_level=log_level,
            enable_console_logging=True,
            enable_file_logging=log_to_file,
            enable_performance_logging=True,
            override_config={
                'log_format_string': DEVELOPMENT_LOG_FORMAT,
                'color_console_output': enable_color_console
            }
        )
        
        if success:
            dev_logger = get_logger('development', ComponentType.UTILS)
            dev_logger.info("Development logging configured successfully")
            
            return {
                'status': 'success',
                'log_level': development_log_level,
                'verbose_output': enable_verbose_output,
                'color_console': enable_color_console,
                'file_logging': log_to_file,
                'performance_logging': True
            }
        else:
            return {
                'status': 'failed',
                'error': 'Failed to configure development logging'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


def setup_performance_logging(timing_threshold_ms: float = PERFORMANCE_TARGET_STEP_LATENCY_MS,
                            enable_memory_tracking: bool = False,
                            monitored_operations: Optional[List[str]] = None,
                            performance_log_file: Optional[str] = None) -> ComponentLogger:
    """
    Configures specialized performance logging for timing measurements, memory tracking,
    and operation profiling with threshold-based monitoring.
    
    Args:
        timing_threshold_ms: Threshold for performance warning alerts
        enable_memory_tracking: Whether to enable memory usage tracking
        monitored_operations: List of operations to monitor automatically
        performance_log_file: Optional specific log file for performance data
        
    Returns:
        ComponentLogger: Configured performance logger instance
    """
    global _global_factory
    
    # Ensure logging system is initialized
    if not _logging_initialized:
        configure_logging(enable_performance_logging=True)
    
    if _global_factory is None:
        _global_factory = LoggerFactory(LoggingConfig(performance_logging_enabled=True))
    
    # Create performance logger
    perf_logger = _global_factory.create_performance_logger(
        operation_name='system_performance',
        timing_threshold_ms=timing_threshold_ms,
        enable_memory_tracking=enable_memory_tracking
    )
    
    # Set up monitored operations
    if monitored_operations:
        for operation in monitored_operations:
            _global_factory.create_performance_logger(
                operation_name=operation,
                timing_threshold_ms=timing_threshold_ms,
                enable_memory_tracking=enable_memory_tracking
            )
    
    perf_logger.info(f"Performance logging initialized with {timing_threshold_ms}ms threshold")
    
    return perf_logger


def create_component_logger(component_type: ComponentType,
                          component_name: str,
                          log_level: Optional[LogLevel] = None,
                          enable_timing_logs: bool = False,
                          additional_config: Optional[Dict[str, Any]] = None) -> ComponentLogger:
    """
    Creates specialized logger for specific plume_nav_sim component with component-appropriate
    configuration, performance tracking, and hierarchical naming.
    
    Args:
        component_type: Type of component for specialized configuration
        component_name: Specific name identifier for the component instance
        log_level: Optional override for component log level
        enable_timing_logs: Whether to enable timing measurements
        additional_config: Optional additional configuration parameters
        
    Returns:
        ComponentLogger: Component-specific logger configured for the component type
    """
    # Validate component type
    if not isinstance(component_type, ComponentType):
        raise ValueError(f"Invalid component_type. Must be ComponentType enum value.")
    
    # Ensure logging is initialized
    if not _logging_initialized:
        configure_logging()
    
    # Get effective log level
    effective_level = log_level or component_type.get_default_log_level()
    
    # Create logger through factory
    logger = get_logger(
        name=component_name,
        component_type=component_type,
        enable_performance_tracking=enable_timing_logs
    )
    
    # Apply additional configuration if provided
    if additional_config:
        # Handle any additional configuration parameters
        pass
    
    logger.debug(f"Component logger created for {component_type.value}:{component_name}")
    
    return logger


def validate_logging_config(config_dict: Dict[str, Any],
                          strict_validation: bool = False,
                          check_permissions: bool = True) -> Tuple[bool, List[str]]:
    """
    Validates logging configuration parameters for correctness, compatibility,
    and resource requirements.
    
    Args:
        config_dict: Dictionary containing logging configuration parameters
        strict_validation: Whether to apply strict validation rules
        check_permissions: Whether to check file system permissions
        
    Returns:
        Tuple[bool, List[str]]: Tuple of (is_valid, validation_errors) with detailed results
    """
    validation_errors = []
    
    try:
        # Create LoggingConfig from dictionary
        config = LoggingConfig(**config_dict)
        
        # Validate using built-in validation
        is_valid, config_errors = config.validate(check_permissions)
        validation_errors.extend(config_errors)
        
        # Additional strict validation
        if strict_validation:
            # Check for optimal performance settings
            if config.max_log_file_size > 100 * 1024 * 1024:  # 100MB
                validation_errors.append("Large log file size may impact performance")
            
            if config.log_backup_count > 10:
                validation_errors.append("High backup count may consume excessive disk space")
            
            # Validate component configuration completeness
            expected_components = set(ComponentType)
            configured_components = set(config.component_specific_levels.keys())
            missing_components = expected_components - configured_components
            
            if missing_components:
                validation_errors.append(
                    f"Missing component configurations: {[c.value for c in missing_components]}"
                )
        
        return len(validation_errors) == 0, validation_errors
        
    except Exception as e:
        validation_errors.append(f"Configuration validation failed: {e}")
        return False, validation_errors


def reset_logging_config(preserve_log_files: bool = True,
                        clear_logger_cache: bool = True,
                        shutdown_timeout: float = 5.0) -> bool:
    """
    Resets logging configuration to default state with proper cleanup of handlers,
    formatters, and cached loggers.
    
    Args:
        preserve_log_files: Whether to keep existing log files
        clear_logger_cache: Whether to clear the logger cache
        shutdown_timeout: Maximum time to wait for shutdown
        
    Returns:
        bool: True if logging reset completed successfully, False otherwise
    """
    global _logging_initialized, _global_factory, _logger_cache
    
    with _config_lock:
        try:
            # Flush all pending log records
            logging.shutdown()
            
            # Clear global factory cache if requested
            if clear_logger_cache and _global_factory:
                _global_factory.clear_cache(close_loggers=True)
            
            # Reset global state
            _logging_initialized = False
            _global_factory = None
            _logger_cache.clear()
            
            # Remove all handlers from plume_nav_sim loggers
            plume_logger = logging.getLogger(LOGGER_NAME_PREFIX)
            for handler in plume_logger.handlers[:]:
                handler.close()
                plume_logger.removeHandler(handler)
            
            # Reset to default configuration
            logging.basicConfig(level=logging.WARNING, force=True)
            
            # Clean up log files if requested
            if not preserve_log_files:
                try:
                    for log_file in DEFAULT_LOG_DIR.glob("plume_nav_sim_*.log*"):
                        log_file.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
            
            return True
            
        except Exception as e:
            print(f"Failed to reset logging configuration: {e}", file=sys.stderr)
            return False


def get_logging_status(include_performance_stats: bool = False,
                      include_handler_details: bool = False,
                      include_logger_hierarchy: bool = False) -> Dict[str, Any]:
    """
    Returns comprehensive logging system status including configuration details,
    active loggers, handler status, and performance metrics.
    
    Args:
        include_performance_stats: Whether to include performance statistics
        include_handler_details: Whether to include detailed handler information
        include_logger_hierarchy: Whether to include logger hierarchy mapping
        
    Returns:
        Dict[str, Any]: Dictionary containing comprehensive logging system status
    """
    global _global_factory, _logging_initialized
    
    status = {
        'system_initialized': _logging_initialized,
        'timestamp': time.time(),
        'configuration': {},
        'factory_stats': {},
        'active_loggers': {},
        'handlers': {},
        'performance_data': {},
        'memory_usage': {}
    }
    
    # Basic system status
    status['configuration'] = {
        'logger_prefix': LOGGER_NAME_PREFIX,
        'default_log_level': DEFAULT_LOG_LEVEL,
        'supported_components': [c.value for c in ComponentType]
    }
    
    # Factory statistics
    if _global_factory:
        status['factory_stats'] = _global_factory.get_factory_statistics()
    
    # Performance statistics
    if include_performance_stats and _global_factory:
        perf_data = {}
        for op_name, data in _global_factory.performance_loggers.items():
            if 'logger' in data and hasattr(data['logger'], 'performance_data'):
                perf_data[op_name] = {
                    'operation_count': len(data['logger'].performance_data.get(op_name, [])),
                    'threshold_ms': data.get('threshold_ms', 0),
                    'memory_tracking': data.get('memory_tracking', False)
                }
        status['performance_data'] = perf_data
    
    # Handler details
    if include_handler_details:
        plume_logger = logging.getLogger(LOGGER_NAME_PREFIX)
        status['handlers'] = {
            'root_handlers': len(logging.root.handlers),
            'plume_handlers': len(plume_logger.handlers),
            'handler_types': [type(h).__name__ for h in plume_logger.handlers]
        }
    
    # Logger hierarchy
    if include_logger_hierarchy:
        hierarchy = {}
        for name, logger in logging.Logger.manager.loggerDict.items():
            if name.startswith(LOGGER_NAME_PREFIX):
                if hasattr(logger, 'level'):
                    hierarchy[name] = {
                        'level': logger.level,
                        'effective_level': logger.getEffectiveLevel(),
                        'propagate': getattr(logger, 'propagate', True)
                    }
        status['logger_hierarchy'] = hierarchy
    
    # Memory usage estimation
    status['memory_usage'] = {
        'cache_entries': len(_logger_cache),
        'estimated_kb': len(_logger_cache) * 2  # Rough estimate
    }
    
    return status


# Create default logging configuration instance
DEFAULT_LOGGING_CONFIG = LoggingConfig()

# Module-level exports
__all__ = [
    # Enums
    'LogLevel',
    'ComponentType',
    
    # Classes
    'LoggingConfig',
    'LoggerFactory',
    'ComponentLogger',
    'SensitiveInfoFilter',
    
    # Configuration functions
    'configure_logging',
    'get_logger',
    'configure_development_logging',
    'setup_performance_logging',
    'create_component_logger',
    'validate_logging_config',
    'reset_logging_config',
    'get_logging_status',
    
    # Constants
    'DEFAULT_LOGGING_CONFIG',
    'LOGGER_NAME_PREFIX',
    'COMPONENT_NAMES',
    'LOG_LEVEL_DEFAULT',
    'PERFORMANCE_TARGET_STEP_LATENCY_MS'
]