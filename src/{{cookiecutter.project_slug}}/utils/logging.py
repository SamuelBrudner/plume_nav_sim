"""
Enhanced logging configuration system with Hydra integration and experiment tracking.

This module provides comprehensive logging setup for the odor plume navigation system,
integrating Loguru with Hydra configuration management, seed manager context binding,
and structured experiment tracking. It establishes centralized logging infrastructure
with configurable sinks, correlation IDs, and comprehensive traceability across 
experiments and CLI operations.

The logging system supports:
- Hydra job tracking with configuration checksum validation
- Automatic seed manager context injection for reproducibility
- CLI command execution metrics and parameter validation timing
- Configuration composition tracking with hierarchical source identification
- Structured output formats compatible with development and production environments
- Environment variable interpolation logging for secure credential management
- Runtime override documentation for parameter sweep analysis

Examples:
    Basic setup:
        >>> from {{cookiecutter.project_slug}}.utils.logging import setup_enhanced_logging
        >>> setup_enhanced_logging()
        >>> logger.info("System initialized")
        
    With Hydra configuration:
        >>> from {{cookiecutter.project_slug}}.utils.logging import configure_from_hydra
        >>> configure_from_hydra(cfg)
        >>> logger.info("Hydra-configured logging active")
        
    CLI command tracking:
        >>> from {{cookiecutter.project_slug}}.utils.logging import track_cli_command
        >>> with track_cli_command("simulate", {"agents": 5}) as tracker:
        ...     # Command execution with automatic metrics
        ...     result = run_simulation()
        
    Experiment context:
        >>> from {{cookiecutter.project_slug}}.utils.logging import bind_experiment_context
        >>> context = bind_experiment_context(experiment_id="exp_001")
        >>> logger.bind(**context).info("Experiment started")
"""

import os
import sys
import time
import uuid
import hashlib
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional, Union, List, Generator, Tuple
from dataclasses import dataclass, field

import loguru
from loguru import logger

# Hydra imports with graceful fallback
try:
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf, ListConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    HydraConfig = None
    DictConfig = dict
    OmegaConf = None
    ListConfig = list

# Import seed manager with fallback
try:
    from .seed_manager import SeedManager, get_global_seed_manager
    SEED_MANAGER_AVAILABLE = True
except ImportError:
    SEED_MANAGER_AVAILABLE = False
    SeedManager = None
    get_global_seed_manager = lambda: None

# Import configuration schemas with fallback
try:
    from ..config.schemas import NavigatorConfig, VideoPlumeConfig
    CONFIG_SCHEMAS_AVAILABLE = True
except ImportError:
    CONFIG_SCHEMAS_AVAILABLE = False
    NavigatorConfig = dict
    VideoPlumeConfig = dict


# Enhanced log format patterns with correlation and experiment tracking
ENHANCED_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<blue>correlation_id={extra[correlation_id]}</blue> | "
    "<magenta>experiment_id={extra[experiment_id]}</magenta> - "
    "<level>{message}</level>"
)

# Hydra-aware format with job and configuration tracking
HYDRA_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<blue>correlation_id={extra[correlation_id]}</blue> | "
    "<magenta>experiment_id={extra[experiment_id]}</magenta> | "
    "<yellow>hydra_job={extra[hydra_job_name]}</yellow> | "
    "<white>config_checksum={extra[config_checksum]}</white> - "
    "<level>{message}</level>"
)

# CLI command format with execution metrics
CLI_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<blue>correlation_id={extra[correlation_id]}</blue> | "
    "<magenta>experiment_id={extra[experiment_id]}</magenta> | "
    "<yellow>cli_command={extra[cli_command]}</yellow> | "
    "<red>execution_time_ms={extra[execution_time_ms]}</red> - "
    "<level>{message}</level>"
)

# Minimal format for development environments
MINIMAL_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan> - "
    "<level>{message}</level>"
)

# Production format with structured JSON output
PRODUCTION_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSSZ} | "
    "{level} | "
    "{name}:{function}:{line} | "
    "correlation_id={extra[correlation_id]} | "
    "experiment_id={extra[experiment_id]} | "
    "seed_value={extra[seed_value]} | "
    "{message}"
)


@dataclass
class LoggingConfig:
    """
    Comprehensive logging configuration schema for enhanced logging setup.
    
    Supports multiple output sinks, format patterns, and integration configurations
    for Hydra job tracking, seed manager context binding, and CLI metrics.
    
    Attributes:
        level: Minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format pattern (enhanced, hydra, cli, minimal, production)
        console_enabled: Enable console output sink
        file_enabled: Enable file output sink
        file_path: Path for file output (supports environment variable interpolation)
        rotation: Log file rotation policy (e.g., "10 MB", "1 day")
        retention: Log file retention policy (e.g., "1 week", "30 days")
        compression: Compression format for rotated files (gz, bz2, xz)
        enqueue: Enable message queuing for better multiprocessing
        backtrace: Include backtrace in exception logs
        diagnose: Enable detailed exception diagnosis
        correlation_id_enabled: Enable automatic correlation ID generation
        experiment_tracking_enabled: Enable experiment context tracking
        hydra_integration_enabled: Enable Hydra job tracking integration
        seed_context_enabled: Enable automatic seed manager context binding
        cli_metrics_enabled: Enable CLI command execution metrics
        environment_logging_enabled: Enable environment variable logging
        performance_monitoring_enabled: Enable performance metric collection
    """
    
    # Basic logging configuration
    level: str = field(default="INFO")
    format: str = field(default="enhanced")
    
    # Output sink configuration
    console_enabled: bool = field(default=True)
    file_enabled: bool = field(default=False)
    file_path: Optional[str] = field(default=None)
    
    # File rotation and retention
    rotation: Optional[str] = field(default="10 MB")
    retention: Optional[str] = field(default="1 week")
    compression: Optional[str] = field(default=None)
    
    # Loguru configuration
    enqueue: bool = field(default=True)
    backtrace: bool = field(default=True)
    diagnose: bool = field(default=True)
    
    # Enhanced tracking features
    correlation_id_enabled: bool = field(default=True)
    experiment_tracking_enabled: bool = field(default=True)
    hydra_integration_enabled: bool = field(default=True)
    seed_context_enabled: bool = field(default=True)
    cli_metrics_enabled: bool = field(default=True)
    environment_logging_enabled: bool = field(default=True)
    performance_monitoring_enabled: bool = field(default=True)
    
    def get_format_string(self) -> str:
        """Get the appropriate format string based on configuration."""
        format_map = {
            'enhanced': ENHANCED_FORMAT,
            'hydra': HYDRA_FORMAT,
            'cli': CLI_FORMAT,
            'minimal': MINIMAL_FORMAT,
            'production': PRODUCTION_FORMAT
        }
        return format_map.get(self.format, ENHANCED_FORMAT)
    
    def resolve_file_path(self) -> Optional[Path]:
        """Resolve file path with environment variable interpolation."""
        if not self.file_path:
            return None
        
        # Handle environment variable interpolation
        resolved_path = os.path.expandvars(self.file_path)
        resolved_path = os.path.expanduser(resolved_path)
        return Path(resolved_path)


@dataclass
class ExperimentContext:
    """
    Comprehensive experiment context for logging integration.
    
    Captures all relevant experiment metadata including Hydra configuration,
    seed manager state, CLI command context, and system information for
    complete experiment traceability and reproducibility.
    
    Attributes:
        experiment_id: Unique experiment identifier
        correlation_id: Request/session correlation identifier
        hydra_job_name: Hydra job name from configuration
        hydra_run_id: Hydra run identifier
        config_checksum: Configuration checksum for validation
        seed_value: Random seed value for reproducibility
        cli_command: CLI command being executed
        cli_parameters: CLI command parameters and overrides
        start_time: Experiment start timestamp
        system_info: System and environment information
        config_composition: Configuration composition hierarchy
        environment_variables: Relevant environment variables
        performance_metrics: Performance and timing metrics
    """
    
    experiment_id: str = field(default_factory=lambda: f"exp_{int(time.time() * 1000000)}")
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Hydra integration
    hydra_job_name: Optional[str] = field(default=None)
    hydra_run_id: Optional[str] = field(default=None)
    config_checksum: Optional[str] = field(default=None)
    
    # Seed manager integration
    seed_value: Optional[int] = field(default=None)
    seed_manager_context: Optional[Dict[str, Any]] = field(default=None)
    
    # CLI command integration
    cli_command: Optional[str] = field(default=None)
    cli_parameters: Optional[Dict[str, Any]] = field(default=None)
    
    # Timing and performance
    start_time: float = field(default_factory=time.time)
    execution_time_ms: Optional[float] = field(default=None)
    
    # System information
    system_info: Dict[str, Any] = field(default_factory=dict)
    config_composition: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_logger_context(self) -> Dict[str, Any]:
        """Convert experiment context to logger binding context."""
        context = {
            'experiment_id': self.experiment_id,
            'correlation_id': self.correlation_id,
            'seed_value': self.seed_value or 'N/A',
            'hydra_job_name': self.hydra_job_name or 'N/A',
            'config_checksum': self.config_checksum or 'N/A',
            'cli_command': self.cli_command or 'N/A',
            'execution_time_ms': self.execution_time_ms or 0.0
        }
        
        # Add CLI parameters if available
        if self.cli_parameters:
            context['cli_parameters'] = str(self.cli_parameters)
        
        # Add performance metrics
        if self.performance_metrics:
            context.update(self.performance_metrics)
        
        return context
    
    def update_from_hydra(self, cfg: Optional[DictConfig] = None) -> None:
        """Update context from Hydra configuration."""
        if not HYDRA_AVAILABLE:
            return
        
        try:
            hydra_config = HydraConfig.get()
            self.hydra_job_name = hydra_config.job.name
            self.hydra_run_id = hydra_config.runtime.output_dir.split('/')[-1]
            
            # Generate configuration checksum
            if cfg is not None:
                config_str = OmegaConf.to_yaml(cfg)
                self.config_checksum = hashlib.md5(config_str.encode()).hexdigest()[:8]
                
                # Extract configuration composition
                self.config_composition = self._extract_config_composition(cfg)
                
        except Exception as e:
            logger.debug(f"Failed to update context from Hydra: {e}")
    
    def update_from_seed_manager(self) -> None:
        """Update context from seed manager."""
        if not SEED_MANAGER_AVAILABLE:
            return
        
        try:
            seed_manager = get_global_seed_manager()
            if seed_manager:
                self.seed_value = seed_manager.seed
                self.seed_manager_context = seed_manager.bind_to_logger()
                
        except Exception as e:
            logger.debug(f"Failed to update context from seed manager: {e}")
    
    def update_system_info(self) -> None:
        """Update system and environment information."""
        import platform
        
        self.system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'hostname': platform.node()
        }
        
        # Capture relevant environment variables
        env_vars = [
            'PYTHONPATH', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'HYDRA_FULL_ERROR', 'LOGURU_LEVEL'
        ]
        
        self.environment_variables = {
            key: os.environ.get(key, 'N/A') for key in env_vars
        }
    
    def _extract_config_composition(self, cfg: DictConfig) -> List[str]:
        """Extract configuration composition hierarchy."""
        composition = []
        
        try:
            # Extract primary config sources
            if hasattr(cfg, '_metadata') and cfg._metadata:
                metadata = cfg._metadata
                if hasattr(metadata, 'ref_type'):
                    composition.append(f"ref_type: {metadata.ref_type}")
                if hasattr(metadata, 'object_type'):
                    composition.append(f"object_type: {metadata.object_type}")
            
            # Extract config groups
            if hasattr(cfg, 'hydra') and cfg.hydra:
                hydra_cfg = cfg.hydra
                if hasattr(hydra_cfg, 'runtime') and hydra_cfg.runtime:
                    if hasattr(hydra_cfg.runtime, 'choices'):
                        choices = hydra_cfg.runtime.choices
                        for key, value in choices.items():
                            composition.append(f"{key}: {value}")
            
        except Exception as e:
            logger.debug(f"Failed to extract config composition: {e}")
            composition.append("composition_extraction_failed")
        
        return composition


class EnhancedLogger:
    """
    Enhanced logger implementation with comprehensive experiment tracking.
    
    Provides centralized logging configuration with Hydra integration, seed manager
    context binding, CLI command tracking, and structured experiment metadata
    for complete traceability and reproducibility across research experiments.
    
    Features:
    - Automatic correlation ID generation for request tracing
    - Hydra job name and configuration checksum integration
    - Seed manager context binding for reproducibility
    - CLI command execution metrics and parameter tracking
    - Configuration composition tracking with hierarchical source identification
    - Environment variable interpolation logging for secure credential management
    - Performance monitoring with timing metrics
    - Structured output formats for development and production environments
    
    Examples:
        Basic usage:
            >>> enhanced_logger = EnhancedLogger()
            >>> enhanced_logger.setup()
            >>> logger.info("System initialized")
            
        With configuration:
            >>> config = LoggingConfig(level="DEBUG", file_enabled=True)
            >>> enhanced_logger = EnhancedLogger(config)
            >>> enhanced_logger.setup()
            
        Context binding:
            >>> context = enhanced_logger.create_experiment_context()
            >>> logger.bind(**context.to_logger_context()).info("Experiment started")
    """
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        Initialize enhanced logger with configuration.
        
        Args:
            config: Logging configuration (uses defaults if not provided)
        """
        self.config = config or LoggingConfig()
        self.experiment_context: Optional[ExperimentContext] = None
        self._setup_complete = False
        self._sink_ids: List[int] = []
        self._lock = threading.Lock()
        
        # Performance tracking
        self._setup_start_time: Optional[float] = None
        self._setup_duration_ms: Optional[float] = None
    
    def setup(self, 
              cfg: Optional[DictConfig] = None,
              experiment_id: Optional[str] = None) -> 'EnhancedLogger':
        """
        Setup enhanced logging system with comprehensive integration.
        
        Configures Loguru with enhanced features including Hydra integration,
        seed manager context binding, and experiment tracking. Ensures all
        logging configuration meets performance requirements (<100ms setup).
        
        Args:
            cfg: Optional Hydra configuration for integration
            experiment_id: Optional experiment identifier
            
        Returns:
            EnhancedLogger: Configured logger instance
            
        Raises:
            RuntimeError: If setup exceeds performance requirements
        """
        self._setup_start_time = time.perf_counter()
        
        with self._lock:
            if self._setup_complete:
                return self
            
            try:
                # Remove default logger to start fresh
                logger.remove()
                
                # Create experiment context
                self.experiment_context = self.create_experiment_context(
                    cfg=cfg,
                    experiment_id=experiment_id
                )
                
                # Setup console sink
                if self.config.console_enabled:
                    self._setup_console_sink()
                
                # Setup file sink
                if self.config.file_enabled:
                    self._setup_file_sink()
                
                # Configure global context
                self._configure_global_context()
                
                # Log setup completion
                self._setup_duration_ms = (time.perf_counter() - self._setup_start_time) * 1000
                self._setup_complete = True
                
                # Validate performance requirement
                if self._setup_duration_ms > 100:
                    warning_msg = (
                        f"Logging setup took {self._setup_duration_ms:.2f}ms, "
                        f"exceeding 100ms performance requirement"
                    )
                    logger.warning(warning_msg)
                
                # Log successful setup
                logger.bind(**self.experiment_context.to_logger_context()).info(
                    f"Enhanced logging configured successfully "
                    f"(setup_time={self._setup_duration_ms:.2f}ms, "
                    f"experiment_id={self.experiment_context.experiment_id})"
                )
                
                return self
                
            except Exception as e:
                error_msg = f"Failed to setup enhanced logging: {str(e)}"
                # Use basic logger since enhanced setup failed
                loguru.logger.error(error_msg)
                raise RuntimeError(error_msg) from e
    
    def _setup_console_sink(self) -> None:
        """Setup console output sink with enhanced formatting."""
        format_string = self.config.get_format_string()
        
        sink_id = logger.add(
            sys.stderr,
            format=format_string,
            level=self.config.level,
            backtrace=self.config.backtrace,
            diagnose=self.config.diagnose,
            enqueue=self.config.enqueue,
            colorize=True
        )
        
        self._sink_ids.append(sink_id)
    
    def _setup_file_sink(self) -> None:
        """Setup file output sink with rotation and compression."""
        file_path = self.config.resolve_file_path()
        if not file_path:
            return
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        format_string = self.config.get_format_string()
        
        sink_id = logger.add(
            str(file_path),
            format=format_string,
            level=self.config.level,
            rotation=self.config.rotation,
            retention=self.config.retention,
            compression=self.config.compression,
            backtrace=self.config.backtrace,
            diagnose=self.config.diagnose,
            enqueue=self.config.enqueue,
            encoding="utf-8"
        )
        
        self._sink_ids.append(sink_id)
    
    def _configure_global_context(self) -> None:
        """Configure global logger context with experiment metadata."""
        if not self.experiment_context:
            return
        
        # Configure default extra values for all log records
        global_context = self.experiment_context.to_logger_context()
        
        # Add environment and system information if enabled
        if self.config.environment_logging_enabled:
            global_context.update({
                'platform': self.experiment_context.system_info.get('platform', 'unknown'),
                'python_version': self.experiment_context.system_info.get('python_version', 'unknown')
            })
        
        # Configure logger with global context
        logger.configure(extra=global_context)
    
    def create_experiment_context(self, 
                                  cfg: Optional[DictConfig] = None,
                                  experiment_id: Optional[str] = None) -> ExperimentContext:
        """
        Create comprehensive experiment context for tracking.
        
        Generates experiment context with Hydra integration, seed manager
        binding, and system information for complete experiment traceability.
        
        Args:
            cfg: Optional Hydra configuration
            experiment_id: Optional experiment identifier
            
        Returns:
            ExperimentContext: Complete experiment context
        """
        context = ExperimentContext(experiment_id=experiment_id or f"exp_{int(time.time() * 1000000)}")
        
        # Update from various sources
        if self.config.hydra_integration_enabled:
            context.update_from_hydra(cfg)
        
        if self.config.seed_context_enabled:
            context.update_from_seed_manager()
        
        if self.config.performance_monitoring_enabled:
            context.update_system_info()
        
        return context
    
    def bind_experiment_context(self, **additional_context) -> Dict[str, Any]:
        """
        Create logger context binding for experiment tracking.
        
        Args:
            **additional_context: Additional context parameters
            
        Returns:
            Dict[str, Any]: Complete logger binding context
        """
        if not self.experiment_context:
            self.experiment_context = self.create_experiment_context()
        
        context = self.experiment_context.to_logger_context()
        context.update(additional_context)
        
        return context
    
    def track_configuration_composition(self, cfg: DictConfig) -> None:
        """
        Track configuration composition for hierarchical validation.
        
        Logs configuration composition hierarchy, override sources, and
        environment variable interpolation for comprehensive config tracking.
        
        Args:
            cfg: Hydra configuration to track
        """
        if not self.config.hydra_integration_enabled or not self.experiment_context:
            return
        
        try:
            # Update experiment context with configuration
            self.experiment_context.update_from_hydra(cfg)
            
            # Log configuration composition
            logger.bind(**self.experiment_context.to_logger_context()).info(
                f"Configuration composition tracked: {self.experiment_context.config_composition}"
            )
            
            # Log environment variable interpolation if enabled
            if self.config.environment_logging_enabled:
                env_vars = self.experiment_context.environment_variables
                interpolated_vars = {k: v for k, v in env_vars.items() if v != 'N/A'}
                
                if interpolated_vars:
                    logger.bind(**self.experiment_context.to_logger_context()).info(
                        f"Environment variables tracked: {interpolated_vars}"
                    )
            
        except Exception as e:
            logger.warning(f"Failed to track configuration composition: {e}")
    
    def track_performance_metric(self, metric_name: str, value: float, unit: str = "ms") -> None:
        """
        Track performance metric with experiment context.
        
        Args:
            metric_name: Name of the performance metric
            value: Metric value
            unit: Metric unit (default: ms)
        """
        if not self.config.performance_monitoring_enabled or not self.experiment_context:
            return
        
        metric_key = f"{metric_name}_{unit}"
        self.experiment_context.performance_metrics[metric_key] = value
        
        logger.bind(**self.experiment_context.to_logger_context()).info(
            f"Performance metric tracked: {metric_name}={value}{unit}"
        )
    
    def cleanup(self) -> None:
        """Cleanup logging configuration and remove sinks."""
        with self._lock:
            for sink_id in self._sink_ids:
                try:
                    logger.remove(sink_id)
                except ValueError:
                    # Sink already removed
                    pass
            
            self._sink_ids.clear()
            self._setup_complete = False
    
    def get_setup_metrics(self) -> Dict[str, Any]:
        """Get logging setup performance metrics."""
        return {
            'setup_duration_ms': self._setup_duration_ms,
            'sinks_configured': len(self._sink_ids),
            'experiment_id': self.experiment_context.experiment_id if self.experiment_context else None,
            'config_checksum': self.experiment_context.config_checksum if self.experiment_context else None
        }


# Global enhanced logger instance
_global_enhanced_logger: Optional[EnhancedLogger] = None
_global_logger_lock = threading.Lock()


def setup_enhanced_logging(
    config: Optional[LoggingConfig] = None,
    cfg: Optional[DictConfig] = None,
    experiment_id: Optional[str] = None
) -> EnhancedLogger:
    """
    Setup global enhanced logging system with comprehensive integration.
    
    Provides centralized logging configuration with Hydra integration, seed manager
    context binding, and experiment tracking. This is the primary entry point for
    configuring enhanced logging throughout the application.
    
    Args:
        config: Logging configuration (uses defaults if not provided)
        cfg: Optional Hydra configuration for integration
        experiment_id: Optional experiment identifier
        
    Returns:
        EnhancedLogger: Configured enhanced logger instance
        
    Examples:
        Basic setup:
            >>> setup_enhanced_logging()
            
        With custom configuration:
            >>> config = LoggingConfig(level="DEBUG", file_enabled=True)
            >>> setup_enhanced_logging(config)
            
        With Hydra integration:
            >>> setup_enhanced_logging(cfg=hydra_config, experiment_id="exp_001")
    """
    global _global_enhanced_logger
    
    with _global_logger_lock:
        if _global_enhanced_logger is not None:
            _global_enhanced_logger.cleanup()
        
        _global_enhanced_logger = EnhancedLogger(config)
        _global_enhanced_logger.setup(cfg=cfg, experiment_id=experiment_id)
        
        return _global_enhanced_logger


def configure_from_hydra(cfg: DictConfig, experiment_id: Optional[str] = None) -> bool:
    """
    Configure enhanced logging from Hydra configuration.
    
    Convenience function for integrating enhanced logging with Hydra-based
    configuration systems. Automatically extracts logging configuration and
    initializes enhanced logging with comprehensive tracking.
    
    Args:
        cfg: Hydra configuration object
        experiment_id: Optional experiment identifier
        
    Returns:
        bool: True if configuration was successful
        
    Examples:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     configure_from_hydra(cfg)
    """
    if not HYDRA_AVAILABLE:
        logger.warning("Hydra not available, using default logging configuration")
        setup_enhanced_logging(experiment_id=experiment_id)
        return False
    
    try:
        # Extract logging configuration from Hydra config
        logging_config = LoggingConfig()
        
        # Update configuration from Hydra if available
        if hasattr(cfg, 'logging'):
            logging_cfg = cfg.logging
            
            # Update basic configuration
            if hasattr(logging_cfg, 'level'):
                logging_config.level = logging_cfg.level
            if hasattr(logging_cfg, 'format'):
                logging_config.format = logging_cfg.format
            if hasattr(logging_cfg, 'file_enabled'):
                logging_config.file_enabled = logging_cfg.file_enabled
            if hasattr(logging_cfg, 'file_path'):
                logging_config.file_path = logging_cfg.file_path
        
        # Setup enhanced logging with Hydra integration
        enhanced_logger = setup_enhanced_logging(
            config=logging_config,
            cfg=cfg,
            experiment_id=experiment_id
        )
        
        # Track configuration composition
        enhanced_logger.track_configuration_composition(cfg)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure logging from Hydra: {e}")
        # Fallback to default configuration
        setup_enhanced_logging(experiment_id=experiment_id)
        return False


def bind_experiment_context(**additional_context) -> Dict[str, Any]:
    """
    Create logger context binding for experiment tracking.
    
    Generates complete experiment context for logger binding with automatic
    integration of seed manager, Hydra configuration, and system information.
    
    Args:
        **additional_context: Additional context parameters
        
    Returns:
        Dict[str, Any]: Complete logger binding context
        
    Examples:
        >>> context = bind_experiment_context(step=1, agent_count=5)
        >>> logger.bind(**context).info("Simulation step completed")
    """
    global _global_enhanced_logger
    
    if _global_enhanced_logger is None:
        setup_enhanced_logging()
    
    return _global_enhanced_logger.bind_experiment_context(**additional_context)


@contextmanager
def track_cli_command(
    command_name: str,
    parameters: Optional[Dict[str, Any]] = None,
    track_performance: bool = True
) -> Generator['CLICommandTracker', None, None]:
    """
    Context manager for CLI command execution tracking.
    
    Provides comprehensive CLI command tracking with execution metrics,
    parameter validation timing, and performance monitoring for enhanced
    observability of command-line operations.
    
    Args:
        command_name: Name of the CLI command being executed
        parameters: Command parameters and overrides
        track_performance: Enable performance metric collection
        
    Yields:
        CLICommandTracker: Command tracker instance
        
    Examples:
        >>> with track_cli_command("simulate", {"agents": 5}) as tracker:
        ...     result = run_simulation()
        ...     tracker.log_metric("agents_processed", 5)
    """
    tracker = CLICommandTracker(command_name, parameters, track_performance)
    
    try:
        yield tracker
    finally:
        tracker.complete()


class CLICommandTracker:
    """
    CLI command execution tracker with performance monitoring.
    
    Tracks CLI command execution metrics including parameter validation time,
    execution duration, and performance indicators for comprehensive command
    observability and optimization analysis.
    
    Attributes:
        command_name: Name of the CLI command
        parameters: Command parameters and overrides
        start_time: Command start timestamp
        metrics: Performance metrics collection
        context: Logger context for command tracking
    """
    
    def __init__(self, 
                 command_name: str,
                 parameters: Optional[Dict[str, Any]] = None,
                 track_performance: bool = True):
        """
        Initialize CLI command tracker.
        
        Args:
            command_name: Name of the CLI command
            parameters: Command parameters and overrides
            track_performance: Enable performance tracking
        """
        self.command_name = command_name
        self.parameters = parameters or {}
        self.track_performance = track_performance
        self.start_time = time.perf_counter()
        self.metrics: Dict[str, float] = {}
        
        # Create command-specific context
        self.context = bind_experiment_context(
            cli_command=command_name,
            cli_parameters=self.parameters
        )
        
        # Log command start
        logger.bind(**self.context).info(
            f"CLI command started: {command_name} with parameters {self.parameters}"
        )
    
    def log_metric(self, metric_name: str, value: float, unit: str = "ms") -> None:
        """
        Log performance metric for the command.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Metric unit
        """
        if self.track_performance:
            self.metrics[f"{metric_name}_{unit}"] = value
            
            logger.bind(**self.context).info(
                f"CLI metric: {metric_name}={value}{unit}"
            )
    
    def log_validation_time(self, validation_time_ms: float) -> None:
        """Log parameter validation time."""
        self.log_metric("parameter_validation_time", validation_time_ms, "ms")
    
    def complete(self) -> None:
        """Complete command tracking and log final metrics."""
        end_time = time.perf_counter()
        execution_time_ms = (end_time - self.start_time) * 1000
        
        # Update context with execution time
        self.context['execution_time_ms'] = execution_time_ms
        
        # Log command completion
        logger.bind(**self.context).info(
            f"CLI command completed: {self.command_name} "
            f"(execution_time={execution_time_ms:.2f}ms, "
            f"metrics={self.metrics})"
        )


def get_module_logger(name: str, **extra_context) -> loguru.Logger:
    """
    Get module-specific logger with enhanced context binding.
    
    Creates a module-specific logger with automatic experiment context binding
    and optional additional context for component identification and tracing.
    
    Args:
        name: Module name (typically __name__)
        **extra_context: Additional context for the module logger
        
    Returns:
        loguru.Logger: Configured logger instance with module context
        
    Examples:
        >>> module_logger = get_module_logger(__name__)
        >>> module_logger.info("Module operation completed")
        
        >>> component_logger = get_module_logger(__name__, component="navigator")
        >>> component_logger.debug("Navigation step executed")
    """
    # Ensure enhanced logging is setup
    global _global_enhanced_logger
    if _global_enhanced_logger is None:
        setup_enhanced_logging()
    
    # Bind module context
    context = bind_experiment_context(module=name, **extra_context)
    return logger.bind(**context)


def log_configuration_override(override_key: str, 
                               old_value: Any, 
                               new_value: Any,
                               source: str = "runtime") -> None:
    """
    Log configuration override for parameter sweep analysis.
    
    Provides comprehensive logging of runtime configuration overrides with
    source identification and value comparison for parameter sweep analysis
    and configuration composition tracking.
    
    Args:
        override_key: Configuration key being overridden
        old_value: Original configuration value
        new_value: New configuration value
        source: Source of the override (runtime, hydra, cli, etc.)
        
    Examples:
        >>> log_configuration_override("navigator.max_speed", 1.0, 2.0, "cli")
        >>> log_configuration_override("seed", 42, 123, "hydra")
    """
    context = bind_experiment_context(
        override_key=override_key,
        override_source=source
    )
    
    logger.bind(**context).info(
        f"Configuration override: {override_key} changed from {old_value} to {new_value} "
        f"(source: {source})"
    )


def get_logging_metrics() -> Dict[str, Any]:
    """
    Get comprehensive logging system metrics.
    
    Returns:
        Dict[str, Any]: Complete logging metrics and status information
    """
    global _global_enhanced_logger
    
    if _global_enhanced_logger is None:
        return {"status": "not_initialized"}
    
    metrics = _global_enhanced_logger.get_setup_metrics()
    metrics.update({
        "status": "initialized",
        "config": {
            "level": _global_enhanced_logger.config.level,
            "format": _global_enhanced_logger.config.format,
            "console_enabled": _global_enhanced_logger.config.console_enabled,
            "file_enabled": _global_enhanced_logger.config.file_enabled,
            "hydra_integration_enabled": _global_enhanced_logger.config.hydra_integration_enabled,
            "seed_context_enabled": _global_enhanced_logger.config.seed_context_enabled,
            "cli_metrics_enabled": _global_enhanced_logger.config.cli_metrics_enabled
        }
    })
    
    return metrics


# Export public API
__all__ = [
    'EnhancedLogger',
    'LoggingConfig',
    'ExperimentContext',
    'CLICommandTracker',
    'setup_enhanced_logging',
    'configure_from_hydra',
    'bind_experiment_context',
    'track_cli_command',
    'get_module_logger',
    'log_configuration_override',
    'get_logging_metrics',
    'ENHANCED_FORMAT',
    'HYDRA_FORMAT',
    'CLI_FORMAT',
    'MINIMAL_FORMAT',
    'PRODUCTION_FORMAT'
]