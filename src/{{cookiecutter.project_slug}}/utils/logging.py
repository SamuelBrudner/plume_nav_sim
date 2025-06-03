"""
Enhanced logging configuration with Hydra integration and experiment tracking.

This module provides comprehensive logging setup using Loguru with integrated Hydra
configuration support, structured output formats, and experiment tracking capabilities.
It establishes centralized logging configuration with configurable sinks, formatting
patterns, and correlation IDs for comprehensive traceability across experiments and
CLI operations.

Features:
- Centralized logging configuration via enhanced setup per Section 5.4.1 monitoring approach
- Hydra job tracking integration for enhanced experiment observability per Feature F-006
- CLI command execution metrics including parameter validation time per Section 5.4.1
- Reproducibility context binding with seed manager integration per Feature F-014
- Configuration composition tracking for hierarchical config validation per Section 5.4.2
- Structured output formats compatible with both development and production environments
- Environment variable interpolation logging for secure credential tracking
- Runtime override documentation for parameter sweep analysis support
"""

import os
import sys
import time
import hashlib
import platform
import threading
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, ContextManager, Set
from contextlib import contextmanager
from datetime import datetime, timezone

from loguru import logger
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import JobReturn, JobStatus
from omegaconf import DictConfig, OmegaConf, ListConfig
import numpy as np

from ..config.schemas import BaseModel, Field


class LoggingConfig(BaseModel):
    """Configuration schema for enhanced logging system."""
    
    # Basic logging configuration
    level: str = Field(
        default="INFO",
        description="Minimum log level to display (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    console_format: str = Field(
        default="enhanced",
        description="Console output format (simple, enhanced, structured, json)"
    )
    file_format: str = Field(
        default="structured",
        description="File output format (simple, enhanced, structured, json)"
    )
    
    # File logging configuration
    enable_file_logging: bool = Field(
        default=True,
        description="Enable file-based logging output"
    )
    log_directory: Optional[str] = Field(
        default=None,
        description="Directory for log files. If None, uses Hydra output directory"
    )
    log_filename: str = Field(
        default="experiment.log",
        description="Filename for log output"
    )
    rotation: str = Field(
        default="100 MB",
        description="Log file rotation trigger (size or time-based)"
    )
    retention: str = Field(
        default="30 days",
        description="Log file retention period"
    )
    compression: str = Field(
        default="gz",
        description="Compression format for rotated logs (gz, bz2, xz)"
    )
    
    # Enhanced features
    enable_hydra_integration: bool = Field(
        default=True,
        description="Enable Hydra job tracking and configuration logging"
    )
    enable_seed_context: bool = Field(
        default=True,
        description="Enable automatic seed manager context binding"
    )
    enable_cli_metrics: bool = Field(
        default=True,
        description="Enable CLI command execution metrics tracking"
    )
    enable_config_tracking: bool = Field(
        default=True,
        description="Enable configuration composition tracking"
    )
    enable_correlation_ids: bool = Field(
        default=True,
        description="Enable correlation ID generation for request tracing"
    )
    
    # Performance and debugging
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance metrics collection and logging"
    )
    performance_threshold_ms: float = Field(
        default=100.0,
        description="Performance threshold in milliseconds for warning logs"
    )
    enable_exception_diagnostics: bool = Field(
        default=True,
        description="Enable enhanced exception diagnostics with backtraces"
    )
    enable_environment_logging: bool = Field(
        default=True,
        description="Enable environment variable interpolation logging"
    )
    
    # Development and debugging
    enable_module_filtering: bool = Field(
        default=False,
        description="Enable module-specific log level filtering"
    )
    filtered_modules: List[str] = Field(
        default_factory=list,
        description="List of module names to apply specific filtering"
    )
    debug_hydra_composition: bool = Field(
        default=False,
        description="Enable detailed Hydra configuration composition debugging"
    )
    
    class Config:
        extra = "forbid"  # Strict validation for configuration integrity


class CorrelationContext:
    """Thread-local correlation context for request tracing."""
    
    _local = threading.local()
    
    @classmethod
    def get_correlation_id(cls) -> str:
        """Get current correlation ID or generate new one."""
        if not hasattr(cls._local, 'correlation_id'):
            cls._local.correlation_id = str(uuid.uuid4())[:8]
        return cls._local.correlation_id
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID for current thread."""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def clear_correlation_id(cls) -> None:
        """Clear correlation ID for current thread."""
        if hasattr(cls._local, 'correlation_id'):
            delattr(cls._local, 'correlation_id')
    
    @classmethod
    @contextmanager
    def correlation_scope(cls, correlation_id: Optional[str] = None) -> ContextManager[str]:
        """Context manager for correlation ID scope."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
        
        # Save previous correlation ID
        previous_id = getattr(cls._local, 'correlation_id', None)
        
        try:
            cls.set_correlation_id(correlation_id)
            yield correlation_id
        finally:
            # Restore previous correlation ID
            if previous_id is not None:
                cls.set_correlation_id(previous_id)
            else:
                cls.clear_correlation_id()


class HydraJobTracker:
    """Hydra job tracking and configuration composition monitoring."""
    
    def __init__(self):
        self._job_start_time: Optional[float] = None
        self._job_name: Optional[str] = None
        self._config_checksum: Optional[str] = None
        self._composition_info: Dict[str, Any] = {}
        self._override_info: Dict[str, Any] = {}
        self._environment_vars: Set[str] = set()
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize Hydra job tracking."""
        job_info = {}
        
        try:
            # Check if Hydra is initialized
            if GlobalHydra().is_initialized():
                hydra_cfg = GlobalHydra.instance().hydra_cfg
                hydra_runtime = GlobalHydra.instance().cfg
                
                # Extract job information
                if hydra_cfg:
                    self._job_name = hydra_cfg.job.name
                    job_info['job_name'] = self._job_name
                    job_info['hydra_version'] = hydra_cfg.hydra.version
                    job_info['runtime_choices'] = dict(hydra_cfg.runtime.choices) if hasattr(hydra_cfg.runtime, 'choices') else {}
                
                # Extract configuration checksum
                if hydra_runtime:
                    self._config_checksum = self._generate_config_checksum(hydra_runtime)
                    job_info['config_checksum'] = self._config_checksum
                
                # Extract composition information
                self._composition_info = self._extract_composition_info(hydra_cfg)
                job_info['composition_info'] = self._composition_info
                
                # Extract override information
                self._override_info = self._extract_override_info(hydra_cfg)
                job_info['override_info'] = self._override_info
                
                # Track environment variable usage
                self._environment_vars = self._extract_environment_variables(hydra_runtime)
                job_info['environment_variables'] = list(self._environment_vars)
                
                # Initialize job start time
                self._job_start_time = time.perf_counter()
                job_info['job_start_time'] = datetime.now(timezone.utc).isoformat()
            
            else:
                job_info['status'] = 'hydra_not_initialized'
        
        except Exception as e:
            job_info['initialization_error'] = str(e)
            logger.warning(f"Failed to initialize Hydra job tracking: {e}")
        
        return job_info
    
    def _generate_config_checksum(self, config: DictConfig) -> str:
        """Generate checksum for configuration reproducibility."""
        try:
            # Convert to container and sort for deterministic hashing
            config_dict = OmegaConf.to_container(config, resolve=True)
            config_str = str(sorted(config_dict.items()) if isinstance(config_dict, dict) else config_dict)
            return hashlib.md5(config_str.encode()).hexdigest()[:12]
        except Exception as e:
            logger.debug(f"Failed to generate config checksum: {e}")
            return "unknown"
    
    def _extract_composition_info(self, hydra_cfg: DictConfig) -> Dict[str, Any]:
        """Extract configuration composition information."""
        composition_info = {}
        
        try:
            # Extract config search path
            if hasattr(hydra_cfg.hydra, 'searchpath'):
                composition_info['search_paths'] = list(hydra_cfg.hydra.searchpath)
            
            # Extract composed configuration sources
            if hasattr(hydra_cfg.hydra, 'compose'):
                composition_info['compose_info'] = OmegaConf.to_container(hydra_cfg.hydra.compose)
            
            # Extract config groups
            if hasattr(hydra_cfg, 'defaults'):
                composition_info['config_groups'] = OmegaConf.to_container(hydra_cfg.defaults)
            
        except Exception as e:
            logger.debug(f"Failed to extract composition info: {e}")
        
        return composition_info
    
    def _extract_override_info(self, hydra_cfg: DictConfig) -> Dict[str, Any]:
        """Extract configuration override information."""
        override_info = {}
        
        try:
            # Extract command-line overrides
            if hasattr(hydra_cfg.hydra, 'overrides'):
                overrides = hydra_cfg.hydra.overrides
                if hasattr(overrides, 'task'):
                    override_info['task_overrides'] = list(overrides.task)
                if hasattr(overrides, 'hydra'):
                    override_info['hydra_overrides'] = list(overrides.hydra)
            
            # Extract runtime choices
            if hasattr(hydra_cfg, 'runtime') and hasattr(hydra_cfg.runtime, 'choices'):
                override_info['runtime_choices'] = dict(hydra_cfg.runtime.choices)
            
        except Exception as e:
            logger.debug(f"Failed to extract override info: {e}")
        
        return override_info
    
    def _extract_environment_variables(self, config: DictConfig) -> Set[str]:
        """Extract environment variables used in configuration."""
        env_vars = set()
        
        try:
            # Convert to string and search for environment variable patterns
            config_str = OmegaConf.to_yaml(config)
            
            # Look for ${oc.env:VAR_NAME} patterns
            import re
            env_pattern = r'\$\{oc\.env:([^}]+)\}'
            matches = re.findall(env_pattern, config_str)
            env_vars.update(matches)
            
            # Look for ${ENV_VAR} patterns
            simple_env_pattern = r'\$\{([A-Z_][A-Z0-9_]*)\}'
            simple_matches = re.findall(simple_env_pattern, config_str)
            env_vars.update(simple_matches)
            
        except Exception as e:
            logger.debug(f"Failed to extract environment variables: {e}")
        
        return env_vars
    
    def get_job_metrics(self) -> Dict[str, Any]:
        """Get current job execution metrics."""
        metrics = {
            'job_name': self._job_name,
            'config_checksum': self._config_checksum,
        }
        
        if self._job_start_time is not None:
            metrics['job_duration_seconds'] = time.perf_counter() - self._job_start_time
        
        return metrics
    
    def log_configuration_change(self, change_type: str, details: Dict[str, Any]) -> None:
        """Log configuration changes during runtime."""
        logger.info(
            f"Configuration change detected: {change_type}",
            extra={
                'event_type': 'config_change',
                'change_type': change_type,
                'change_details': details,
                'job_name': self._job_name,
                'config_checksum': self._config_checksum,
            }
        )


class CLIMetricsTracker:
    """CLI command execution metrics and performance tracking."""
    
    def __init__(self):
        self._command_start_time: Optional[float] = None
        self._command_name: Optional[str] = None
        self._parameter_validation_time: Optional[float] = None
        self._initialization_time: Optional[float] = None
        
    @contextmanager
    def track_command(self, command_name: str) -> ContextManager[Dict[str, Any]]:
        """Track CLI command execution metrics."""
        self._command_name = command_name
        self._command_start_time = time.perf_counter()
        
        metrics = {
            'command_name': command_name,
            'start_time': datetime.now(timezone.utc).isoformat(),
        }
        
        try:
            yield metrics
        finally:
            if self._command_start_time is not None:
                total_time = (time.perf_counter() - self._command_start_time) * 1000
                metrics['total_execution_time_ms'] = total_time
                
                logger.info(
                    f"CLI command completed: {command_name}",
                    extra={
                        'event_type': 'cli_command_complete',
                        'command_metrics': metrics,
                    }
                )
    
    @contextmanager
    def track_parameter_validation(self) -> ContextManager[Dict[str, Any]]:
        """Track parameter validation timing."""
        start_time = time.perf_counter()
        validation_metrics = {}
        
        try:
            yield validation_metrics
        finally:
            validation_time = (time.perf_counter() - start_time) * 1000
            self._parameter_validation_time = validation_time
            validation_metrics['validation_time_ms'] = validation_time
            
            if validation_time > 100:  # Performance threshold
                logger.warning(
                    f"Parameter validation exceeded performance threshold: {validation_time:.2f}ms",
                    extra={
                        'event_type': 'performance_warning',
                        'validation_metrics': validation_metrics,
                    }
                )
            else:
                logger.debug(
                    f"Parameter validation completed: {validation_time:.2f}ms",
                    extra={
                        'event_type': 'parameter_validation',
                        'validation_metrics': validation_metrics,
                    }
                )


class EnhancedLoggingManager:
    """
    Enhanced logging management system with Hydra integration.
    
    Provides centralized logging configuration, experiment tracking, and
    comprehensive observability for odor plume navigation experiments.
    """
    
    _instance: Optional['EnhancedLoggingManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'EnhancedLoggingManager':
        """Singleton implementation ensuring single logging manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logging manager (only once due to singleton pattern)."""
        if not self._initialized:
            self._config: Optional[LoggingConfig] = None
            self._hydra_tracker = HydraJobTracker()
            self._cli_tracker = CLIMetricsTracker()
            self._log_context: Dict[str, Any] = {}
            self._formatters: Dict[str, str] = {}
            self._setup_formatters()
            EnhancedLoggingManager._initialized = True
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance for testing purposes."""
        cls._instance = None
        cls._initialized = False
    
    def initialize(
        self,
        config: Optional[Union[LoggingConfig, DictConfig, Dict[str, Any]]] = None
    ) -> None:
        """
        Initialize enhanced logging system.
        
        Args:
            config: Logging configuration (LoggingConfig, DictConfig, or dict).
                   If None, attempts to load from Hydra global config.
        """
        start_time = time.perf_counter()
        
        try:
            # Load configuration
            self._config = self._load_config(config)
            
            # Remove existing loggers
            logger.remove()
            
            # Initialize Hydra tracking if enabled
            if self._config.enable_hydra_integration:
                hydra_info = self._hydra_tracker.initialize()
                self._log_context.update(hydra_info)
            
            # Setup logging context
            self._setup_logging_context()
            
            # Configure console logging
            self._setup_console_logging()
            
            # Configure file logging if enabled
            if self._config.enable_file_logging:
                self._setup_file_logging()
            
            # Setup performance monitoring if enabled
            if self._config.enable_performance_monitoring:
                self._setup_performance_monitoring()
            
            # Setup correlation ID handling if enabled
            if self._config.enable_correlation_ids:
                self._setup_correlation_context()
            
            # Setup seed manager integration if enabled
            if self._config.enable_seed_context:
                self._setup_seed_context()
            
            # Validate initialization performance
            initialization_time = (time.perf_counter() - start_time) * 1000
            if initialization_time > 100:  # Performance threshold
                logger.warning(
                    f"Logging initialization exceeded performance threshold: {initialization_time:.2f}ms"
                )
            
            logger.info(
                "Enhanced logging system initialized successfully",
                extra={
                    'event_type': 'logging_initialized',
                    'initialization_time_ms': f"{initialization_time:.2f}",
                    'config_summary': self._get_config_summary(),
                }
            )
            
        except Exception as e:
            # Fallback to basic logging if initialization fails
            self._setup_fallback_logging()
            logger.error(f"Enhanced logging initialization failed, using fallback: {e}")
            raise RuntimeError(f"Logging initialization failed: {e}") from e
    
    def _load_config(self, config: Optional[Union[LoggingConfig, DictConfig, Dict[str, Any]]]) -> LoggingConfig:
        """Load and validate logging configuration."""
        if config is None:
            # Attempt to load from Hydra global config
            config = self._load_from_hydra()
        
        if isinstance(config, LoggingConfig):
            return config
        elif isinstance(config, (DictConfig, dict)):
            return LoggingConfig(**dict(config))
        else:
            logger.warning("No valid logging configuration found, using defaults")
            return LoggingConfig()
    
    def _load_from_hydra(self) -> Dict[str, Any]:
        """Load logging configuration from Hydra global config."""
        try:
            if GlobalHydra().is_initialized():
                hydra_cfg = GlobalHydra.instance().cfg
                if hydra_cfg and "logging" in hydra_cfg:
                    return OmegaConf.to_container(hydra_cfg.logging, resolve=True)
        except Exception as e:
            logger.debug(f"Could not load logging config from Hydra: {e}")
        
        return {}
    
    def _setup_formatters(self) -> None:
        """Setup logging format templates."""
        # Simple format for basic console output
        self._formatters['simple'] = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )
        
        # Enhanced format with module and function context
        self._formatters['enhanced'] = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<blue>corr={extra[correlation_id]}</blue> - "
            "<level>{message}</level>"
        )
        
        # Structured format for detailed logging
        self._formatters['structured'] = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<blue>corr={extra[correlation_id]}</blue> | "
            "<blue>job={extra[job_name]}</blue> | "
            "<blue>seed={extra[seed]}</blue> - "
            "<level>{message}</level>"
        )
        
        # JSON format for machine-readable logs
        self._formatters['json'] = (
            '{"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSSZ}", '
            '"level": "{level}", '
            '"module": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"correlation_id": "{extra[correlation_id]}", '
            '"job_name": "{extra[job_name]}", '
            '"seed": "{extra[seed]}", '
            '"message": "{message}", '
            '"extra": {extra}}'
        )
    
    def _setup_logging_context(self) -> None:
        """Setup global logging context with default values."""
        # Default context values
        default_context = {
            'correlation_id': 'unknown',
            'job_name': 'unknown',
            'seed': 'unknown',
            'run_id': 'unknown',
            'environment_hash': 'unknown',
        }
        
        # Update with any existing context
        default_context.update(self._log_context)
        self._log_context = default_context
        
        # Configure Loguru patcher for automatic context injection
        def add_context(record):
            """Add global context to log record."""
            if 'extra' not in record:
                record['extra'] = {}
            
            # Add correlation ID
            if self._config.enable_correlation_ids:
                record['extra']['correlation_id'] = CorrelationContext.get_correlation_id()
            
            # Add global context
            for key, value in self._log_context.items():
                if key not in record['extra']:
                    record['extra'][key] = value
            
            return record
        
        logger.configure(patcher=add_context)
    
    def _setup_console_logging(self) -> None:
        """Setup console logging with appropriate format."""
        console_format = self._formatters.get(self._config.console_format, self._formatters['enhanced'])
        
        logger.add(
            sys.stderr,
            format=console_format,
            level=self._config.level,
            backtrace=self._config.enable_exception_diagnostics,
            diagnose=self._config.enable_exception_diagnostics,
            enqueue=True,  # Thread-safe logging
        )
    
    def _setup_file_logging(self) -> None:
        """Setup file logging with rotation and retention."""
        # Determine log directory
        log_dir = self._get_log_directory()
        log_file = Path(log_dir) / self._config.log_filename
        
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Select file format
        file_format = self._formatters.get(self._config.file_format, self._formatters['structured'])
        
        logger.add(
            str(log_file),
            format=file_format,
            level=self._config.level,
            rotation=self._config.rotation,
            retention=self._config.retention,
            compression=self._config.compression,
            backtrace=self._config.enable_exception_diagnostics,
            diagnose=self._config.enable_exception_diagnostics,
            enqueue=True,
        )
        
        logger.debug(f"File logging configured: {log_file}")
    
    def _get_log_directory(self) -> str:
        """Get log directory from configuration or Hydra output directory."""
        if self._config.log_directory:
            return self._config.log_directory
        
        # Try to use Hydra output directory
        try:
            if GlobalHydra().is_initialized():
                hydra_cfg = HydraConfig.get()
                if hydra_cfg.runtime.output_dir:
                    return hydra_cfg.runtime.output_dir
        except Exception as e:
            logger.debug(f"Could not get Hydra output directory: {e}")
        
        # Fallback to current directory
        return "logs"
    
    def _setup_performance_monitoring(self) -> None:
        """Setup performance monitoring and thresholds."""
        # Add performance monitoring filter
        def performance_filter(record):
            """Filter and enhance performance-related logs."""
            if 'event_type' in record.get('extra', {}):
                event_type = record['extra']['event_type']
                
                # Check performance thresholds
                if event_type in ['parameter_validation', 'config_load', 'initialization']:
                    duration_key = None
                    for key in record['extra'].keys():
                        if 'time_ms' in key or 'duration_ms' in key:
                            duration_key = key
                            break
                    
                    if duration_key and record['extra'][duration_key] > self._config.performance_threshold_ms:
                        record['level'] = 'WARNING'
                        record['message'] = f"PERFORMANCE: {record['message']}"
            
            return record
        
        logger.configure(patcher=performance_filter)
    
    def _setup_correlation_context(self) -> None:
        """Setup correlation ID context management."""
        # Correlation context is handled in the patcher function
        # This method can be extended for additional correlation features
        logger.debug("Correlation ID context management enabled")
    
    def _setup_seed_context(self) -> None:
        """Setup seed manager context integration."""
        try:
            # Import seed manager to get current context
            from .seed_manager import get_seed_manager
            
            seed_manager = get_seed_manager()
            if seed_manager.current_seed is not None:
                self._log_context.update({
                    'seed': seed_manager.current_seed,
                    'run_id': seed_manager.run_id or 'unknown',
                    'environment_hash': seed_manager.environment_hash or 'unknown',
                })
                
                logger.debug(
                    "Seed manager context integrated",
                    extra={
                        'seed_context': {
                            'seed': seed_manager.current_seed,
                            'run_id': seed_manager.run_id,
                            'environment_hash': seed_manager.environment_hash,
                        }
                    }
                )
        except Exception as e:
            logger.debug(f"Could not integrate seed manager context: {e}")
    
    def _setup_fallback_logging(self) -> None:
        """Setup minimal fallback logging if initialization fails."""
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
            level="INFO",
        )
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging."""
        return {
            'level': self._config.level,
            'console_format': self._config.console_format,
            'file_logging_enabled': self._config.enable_file_logging,
            'hydra_integration_enabled': self._config.enable_hydra_integration,
            'seed_context_enabled': self._config.enable_seed_context,
            'cli_metrics_enabled': self._config.enable_cli_metrics,
            'correlation_ids_enabled': self._config.enable_correlation_ids,
            'performance_monitoring_enabled': self._config.enable_performance_monitoring,
        }
    
    def update_context(self, context: Dict[str, Any]) -> None:
        """Update global logging context."""
        self._log_context.update(context)
        logger.debug(f"Logging context updated: {list(context.keys())}")
    
    def get_cli_tracker(self) -> CLIMetricsTracker:
        """Get CLI metrics tracker instance."""
        return self._cli_tracker
    
    def get_hydra_tracker(self) -> HydraJobTracker:
        """Get Hydra job tracker instance."""
        return self._hydra_tracker
    
    def log_environment_variables(self, var_names: List[str]) -> None:
        """Log environment variable usage for security tracking."""
        if self._config.enable_environment_logging:
            env_info = {}
            for var_name in var_names:
                env_value = os.getenv(var_name)
                if env_value:
                    # Mask sensitive values
                    if any(sensitive in var_name.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                        env_info[var_name] = f"***{env_value[-4:]}" if len(env_value) > 4 else "***"
                    else:
                        env_info[var_name] = env_value
                else:
                    env_info[var_name] = None
            
            logger.info(
                "Environment variables accessed",
                extra={
                    'event_type': 'environment_access',
                    'environment_variables': env_info,
                }
            )
    
    def log_configuration_composition(self, composition_details: Dict[str, Any]) -> None:
        """Log configuration composition for hierarchical tracking."""
        if self._config.enable_config_tracking:
            logger.info(
                "Configuration composition tracked",
                extra={
                    'event_type': 'config_composition',
                    'composition_details': composition_details,
                }
            )
    
    def create_correlation_scope(self, correlation_id: Optional[str] = None) -> ContextManager[str]:
        """Create correlation ID scope for request tracing."""
        return CorrelationContext.correlation_scope(correlation_id)
    
    def create_cli_command_scope(self, command_name: str) -> ContextManager[Dict[str, Any]]:
        """Create CLI command tracking scope."""
        return self._cli_tracker.track_command(command_name)
    
    def create_parameter_validation_scope(self) -> ContextManager[Dict[str, Any]]:
        """Create parameter validation tracking scope."""
        return self._cli_tracker.track_parameter_validation()


# Global logging manager instance
_global_logging_manager: Optional[EnhancedLoggingManager] = None


def get_logging_manager() -> EnhancedLoggingManager:
    """Get the global enhanced logging manager instance."""
    global _global_logging_manager
    if _global_logging_manager is None:
        _global_logging_manager = EnhancedLoggingManager()
    return _global_logging_manager


def setup_enhanced_logging(
    config: Optional[Union[LoggingConfig, DictConfig, Dict[str, Any]]] = None
) -> None:
    """
    Setup enhanced logging system with Hydra integration.
    
    Convenience function for initializing the enhanced logging system.
    
    Args:
        config: Logging configuration. If None, uses defaults or loads from Hydra.
    
    Example:
        # Basic setup
        setup_enhanced_logging()
        
        # With custom configuration
        config = LoggingConfig(level="DEBUG", enable_cli_metrics=True)
        setup_enhanced_logging(config)
        
        # From dictionary
        setup_enhanced_logging({"level": "DEBUG", "enable_file_logging": False})
    """
    logging_manager = get_logging_manager()
    logging_manager.initialize(config)


def get_module_logger(name: str) -> logger:
    """
    Get a logger for a specific module with enhanced context.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Loguru logger instance with module context binding
    
    Example:
        logger = get_module_logger(__name__)
        logger.info("Module-specific log message")
    """
    return logger.bind(module=name)


def create_correlation_scope(correlation_id: Optional[str] = None) -> ContextManager[str]:
    """
    Create correlation ID scope for request tracing.
    
    Args:
        correlation_id: Optional correlation ID. If None, generates new one.
    
    Returns:
        Context manager yielding correlation ID
    
    Example:
        with create_correlation_scope() as corr_id:
            logger.info("Operation with correlation tracking")
    """
    logging_manager = get_logging_manager()
    return logging_manager.create_correlation_scope(correlation_id)


def create_cli_command_scope(command_name: str) -> ContextManager[Dict[str, Any]]:
    """
    Create CLI command tracking scope.
    
    Args:
        command_name: Name of the CLI command being executed
    
    Returns:
        Context manager yielding command metrics dictionary
    
    Example:
        with create_cli_command_scope("simulate") as metrics:
            # Command execution
            metrics['parameter_count'] = 5
    """
    logging_manager = get_logging_manager()
    return logging_manager.create_cli_command_scope(command_name)


def create_parameter_validation_scope() -> ContextManager[Dict[str, Any]]:
    """
    Create parameter validation tracking scope.
    
    Returns:
        Context manager yielding validation metrics dictionary
    
    Example:
        with create_parameter_validation_scope() as validation_metrics:
            # Parameter validation logic
            validation_metrics['parameters_validated'] = 10
    """
    logging_manager = get_logging_manager()
    return logging_manager.create_parameter_validation_scope()


def log_environment_variables(var_names: List[str]) -> None:
    """
    Log environment variable usage for security tracking.
    
    Args:
        var_names: List of environment variable names to log
    
    Example:
        log_environment_variables(['DATABASE_URL', 'API_KEY'])
    """
    logging_manager = get_logging_manager()
    logging_manager.log_environment_variables(var_names)


def log_configuration_composition(composition_details: Dict[str, Any]) -> None:
    """
    Log configuration composition for hierarchical tracking.
    
    Args:
        composition_details: Details about configuration composition
    
    Example:
        log_configuration_composition({
            'base_config': 'base.yaml',
            'overrides': ['experiment=test'],
            'resolved_keys': ['model.learning_rate', 'data.batch_size']
        })
    """
    logging_manager = get_logging_manager()
    logging_manager.log_configuration_composition(composition_details)


# Register configuration schema with Hydra
cs = ConfigStore.instance()
cs.store(name="logging_config", node=LoggingConfig)


# Default setup for basic console logging (fallback)
try:
    setup_enhanced_logging()
except Exception:
    # If enhanced setup fails, fall back to basic logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
        level="INFO",
    )


__all__ = [
    "LoggingConfig",
    "EnhancedLoggingManager",
    "CorrelationContext",
    "HydraJobTracker",
    "CLIMetricsTracker",
    "get_logging_manager",
    "setup_enhanced_logging", 
    "get_module_logger",
    "create_correlation_scope",
    "create_cli_command_scope",
    "create_parameter_validation_scope",
    "log_environment_variables",
    "log_configuration_composition",
]