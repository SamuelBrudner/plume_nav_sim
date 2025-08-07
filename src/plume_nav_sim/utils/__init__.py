"""
Centralized Utilities Package for Plume Navigation Simulation

This module serves as the unified entry point for all utility functions in the plume navigation
system, providing comprehensive access to frame caching, I/O operations, logging configuration,
visualization components, seed management, and navigator utilities under a single namespace.

The module supports multiple import patterns for different frameworks and use cases:
- Kedro projects: Direct imports for pipeline integration
- RL frameworks: Seed management and reproducibility functions
- ML/neural network analyses: Comprehensive utility access
- CLI operations: Complete functionality suite
- Performance-critical applications: Frame caching with memory monitoring

Key Features:
- **Frame Caching**: High-performance caching with configurable modes (none, lru, all)
- **I/O Operations**: Atomic file handling for YAML, JSON, and NumPy data formats
- **Seed Management**: Global and scoped seed control with thread-local isolation
- **Visualization**: Real-time animations and publication-quality static plots
- **Logging Infrastructure**: Structured logging setup with loguru integration
- **Navigator Utilities**: Factory functions for navigator creation and management
- **Graceful Degradation**: Availability flags and fallback implementations

All utility functions integrate with the Hydra configuration system and maintain backward
compatibility with existing interfaces while providing enhanced functionality through
the new project structure supporting the Gymnasium 0.29.x migration.

Examples:
    Basic utility imports:
        >>> from plume_nav_sim.utils import set_global_seed, setup_logger
        >>> setup_logger()
        >>> set_global_seed(42)
        
    Frame cache configuration:
        >>> from plume_nav_sim.utils import FrameCache, CacheMode
        >>> if CACHE_AVAILABLE:
        ...     cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=2048)
        
    Visualization and analysis:
        >>> from plume_nav_sim.utils import visualize_trajectory, SimulationVisualization
        >>> if VISUALIZATION_AVAILABLE:
        ...     viz = SimulationVisualization.from_config(cfg.visualization)
        ...     visualize_trajectory(positions, orientations, output_path="trajectory.png")
        
    Comprehensive ML/RL workflow:
        >>> from plume_nav_sim.utils import (
        ...     set_global_seed, get_enhanced_logger, FrameCache,
        ...     SimulationVisualization, seed_context
        ... )
        >>> setup_logger()
        >>> logger = get_enhanced_logger(__name__)
        >>> with seed_context(42):
        ...     # Reproducible operations with frame caching
        ...     cache = FrameCache(mode="lru", memory_limit_mb=1024)
        ...     result = run_experiment()
        
    Hydra configuration integration:
        >>> from plume_nav_sim.utils import configure_from_hydra
        >>> configure_from_hydra(cfg)
"""

# Core imports with error handling for optional dependencies
import warnings
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Generator
from pathlib import Path

# Frame cache imports - critical for performance per Section 0.2.3
try:
    from plume_nav_sim.utils.frame_cache import (
        FrameCache,
        CacheMode,
        CacheStatistics,
        create_lru_cache,
        create_preload_cache,
        create_no_cache
    )
    CACHE_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Frame cache functionality not available: {e}")
    CACHE_AVAILABLE = False
    # Provide fallback classes
    class FrameCache:
        """Fallback frame cache class."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Frame cache functionality not available")
    
    class CacheMode:
        """Fallback cache mode enum."""
        NONE = "none"
        LRU = "lru"
        ALL = "all"
    
    class CacheStatistics:
        """Fallback cache statistics class."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Frame cache functionality not available")

# Enhanced logging system imports
try:
    from plume_nav_sim.utils.logging_setup import (
        setup_logger,
        get_enhanced_logger,
        get_module_logger,
        get_logger,
        correlation_context,
        get_correlation_context,
        set_correlation_context,
        create_step_timer,
        step_performance_timer,
        frame_rate_timer,
        update_cache_metrics,
        log_cache_memory_pressure_violation,
        detect_legacy_gym_import,
        log_legacy_api_deprecation,
        monitor_environment_creation,
        register_logging_config_schema,
        # Configuration classes
        LoggingConfig,
        PerformanceMetrics,
        FrameCacheConfig,
        EnhancedLogger,
        CorrelationContext,
        # Format constants
        DEFAULT_FORMAT,
        MODULE_FORMAT,
        ENHANCED_FORMAT,
        HYDRA_FORMAT,
        CLI_FORMAT,
        MINIMAL_FORMAT,
        PRODUCTION_FORMAT,
        JSON_FORMAT,
        LOG_LEVELS,
        PERFORMANCE_THRESHOLDS,
        ENVIRONMENT_DEFAULTS
    )
    LOGGING_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Enhanced logging functionality not available: {e}")
    LOGGING_AVAILABLE = False
    # Provide fallback functions
    def setup_logger(*args, **kwargs):
        """Fallback logging setup."""
        warnings.warn("Enhanced logging not available, using basic logging")
        import logging
        logging.basicConfig(level=logging.INFO)
    
    def get_enhanced_logger(name: str, **kwargs):
        """Fallback enhanced logger."""
        import logging
        return logging.getLogger(name)
    
    def get_module_logger(name: str, **kwargs):
        """Fallback module logger."""
        import logging
        return logging.getLogger(name)
    
    def get_logger(name: str, **kwargs):
        """Fallback logger."""
        import logging
        return logging.getLogger(name)

# Visualization system imports
try:
    from plume_nav_sim.utils.visualization import (
        SimulationVisualization,
        visualize_trajectory,
        create_realtime_visualizer,
        create_static_plotter,
        VisualizationConfig
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Visualization functionality not available: {e}")
    VISUALIZATION_AVAILABLE = False
    # Provide fallback classes
    class SimulationVisualization:
        """Fallback visualization class."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Visualization functionality not available")
    
    def visualize_trajectory(*args, **kwargs):
        """Fallback trajectory visualization."""
        raise ImportError("Visualization functionality not available")
    
    def create_realtime_visualizer(*args, **kwargs):
        """Fallback realtime visualizer."""
        raise ImportError("Visualization functionality not available")
    
    def create_static_plotter(*args, **kwargs):
        """Fallback static plotter."""
        raise ImportError("Visualization functionality not available")

# Seed management system imports
try:
    from plume_nav_sim.utils.seed_manager import (
        set_global_seed,
        get_seed_manager as get_global_seed_manager,
        get_random_state,
        restore_random_state,
        capture_random_state,
        reset_random_state,
        scoped_seed,
        get_seed_context,
        seed_sensitive_operation,
        SeedConfig,
        setup_global_seed,
        create_seed_config_from_hydra,
        RandomState,
        SeedContext,
        validate_determinism,
        is_seeded,
        get_last_seed,
        generate_experiment_seed,
        register_seed_config_schema,
        get_reproducibility_report,
        SEED_PERFORMANCE_THRESHOLDS,
        DETERMINISM_TEST_ITERATIONS,
        CROSS_PLATFORM_SEED_MAX
    )
    SEED_MANAGER_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Seed management functionality not available: {e}")
    SEED_MANAGER_AVAILABLE = False
    # Provide fallback functions
    def set_global_seed(seed: int, **kwargs):
        """Fallback seed management."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        warnings.warn("Using basic seed management, enhanced features not available")
    
    def get_global_seed_manager(*args, **kwargs):
        """Fallback seed manager."""
        return None

# I/O Utilities
try:
    from plume_nav_sim.utils.io import (
        load_yaml,
        save_yaml,
        load_json,
        save_json,
        load_numpy,
        save_numpy,
        IOError,
        YAMLError,
        JSONError,
        NumpyError
    )
    IO_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"I/O utilities not available: {e}")
    IO_AVAILABLE = False
    # Provide basic fallback implementations
    def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Fallback YAML loader."""
        import yaml
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Fallback YAML saver."""
        import yaml
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, indent=2)

# Navigator utilities
try:
    from plume_nav_sim.utils.navigator_utils import (
        # Core navigator creation functions
        create_navigator_from_config,
        create_reproducible_navigator,
        create_navigator_from_params,
        create_navigator_factory,
        # Configuration and validation utilities
        NavigatorCreationResult,
        validate_navigator_configuration,
        normalize_array_parameter,
        # Sensor layout and management utilities
        PREDEFINED_SENSOR_LAYOUTS,
        get_predefined_sensor_layout,
        define_sensor_offsets,
        rotate_offset,
        calculate_sensor_positions,
        compute_sensor_positions,
        sample_odor_at_sensors,
        # State management utilities
        SingleAgentParams,
        MultiAgentParams,
        reset_navigator_state,
        reset_navigator_state_with_params,
        update_positions_and_orientations,
        read_odor_values,
        # Performance monitoring utilities
        navigator_performance_context,
        # Introspection and analysis utilities
        get_navigator_capabilities,
        create_navigator_comparison_report,
        # Property access utilities
        get_property_name,
        get_property_value,
        # CLI and database integration
        create_navigator_for_cli,
        create_navigator_with_database_logging
    )
    NAVIGATOR_UTILS_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Navigator utilities not available: {e}")
    NAVIGATOR_UTILS_AVAILABLE = False
    # Provide basic fallback implementations
    def normalize_array_parameter(param: Any, expected_shape: Optional[Tuple[int, ...]] = None) -> Any:
        """Fallback array parameter normalization."""
        import numpy as np
        if not isinstance(param, np.ndarray):
            param = np.asarray(param)
        if expected_shape and param.shape != expected_shape:
            try:
                param = np.broadcast_to(param, expected_shape)
            except ValueError:
                param = np.resize(param, expected_shape)
        return param
    
    def create_navigator_from_params(**kwargs) -> Any:
        """Fallback navigator creation."""
        warnings.warn("Navigator creation not available. Use plume_nav_sim.api.navigation module.")
        raise ImportError("Navigator creation not available")


# Convenience function for seed management (backward compatibility)
def get_random_state():
    """
    Capture current global random state for checkpointing.
    
    Convenience wrapper around the global seed manager's capture_state method.
    Returns the current random state that can be restored later for reproducibility.
    
    Returns:
        RandomState: Current random state snapshot, or None if no global manager exists
        
    Examples:
        >>> set_global_seed(42)
        >>> state = get_random_state()
        >>> # ... perform random operations ...
        >>> restore_random_state(state)  # Restore to captured state
    """
    if SEED_MANAGER_AVAILABLE:
        manager = get_global_seed_manager()
        return manager.capture_state() if manager else None
    else:
        warnings.warn("Enhanced seed management not available")
        return None


def restore_random_state(state):
    """
    Restore global random state from a previous snapshot.
    
    Convenience wrapper around the global seed manager's restore_state method.
    Restores random number generators to the exact state captured in the snapshot.
    
    Args:
        state (RandomState): Random state snapshot to restore
        
    Returns:
        bool: True if restoration was successful, False if no global manager exists
        
    Examples:
        >>> state = get_random_state()
        >>> # ... perform random operations ...
        >>> success = restore_random_state(state)
    """
    if SEED_MANAGER_AVAILABLE:
        manager = get_global_seed_manager()
        return manager.restore_state(state) if manager and state else False
    else:
        warnings.warn("Enhanced seed management not available")
        return False


# Convenience function for unified Hydra configuration
def configure_from_hydra(cfg: Any, components: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Configure multiple utility components from Hydra configuration.
    
    Provides unified configuration interface for all utility components,
    automatically detecting and configuring available subsystems.
    
    Args:
        cfg: Hydra configuration object
        components: Optional list of specific components to configure
                   ('logging', 'seed_manager', 'frame_cache'). If None, configures all available.
        
    Returns:
        Dictionary mapping component names to configuration success status
        
    Examples:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        >>> results = configure_from_hydra(cfg)
        >>> print(f"Logging configured: {results['logging']}")
        >>> print(f"Seed manager configured: {results['seed_manager']}")
        >>> print(f"Frame cache configured: {results['frame_cache']}")
    """
    results = {}
    available_components = []
    
    # Determine which components to configure
    if components is None:
        if LOGGING_AVAILABLE:
            available_components.append('logging')
        if SEED_MANAGER_AVAILABLE:
            available_components.append('seed_manager')
        if CACHE_AVAILABLE:
            available_components.append('frame_cache')
    else:
        available_components = components
    
    # Configure logging
    if 'logging' in available_components and LOGGING_AVAILABLE:
        try:
            if hasattr(cfg, 'logging'):
                from plume_nav_sim.utils.logging_setup import configure_from_hydra as configure_logging_from_hydra
                results['logging'] = configure_logging_from_hydra(cfg)
            else:
                results['logging'] = False
        except Exception as e:
            warnings.warn(f"Failed to configure logging from Hydra: {e}")
            results['logging'] = False
    else:
        results['logging'] = False
    
    # Configure seed manager
    if 'seed_manager' in available_components and SEED_MANAGER_AVAILABLE:
        try:
            if hasattr(cfg, 'seed'):
                from plume_nav_sim.utils.seed_manager import create_seed_config_from_hydra, setup_global_seed
                seed_config = create_seed_config_from_hydra(cfg.seed)
                setup_global_seed(seed_config)
                results['seed_manager'] = True
            else:
                results['seed_manager'] = False
        except Exception as e:
            warnings.warn(f"Failed to configure seed manager from Hydra: {e}")
            results['seed_manager'] = False
    else:
        results['seed_manager'] = False
    
    # Configure frame cache
    if 'frame_cache' in available_components and CACHE_AVAILABLE:
        try:
            if hasattr(cfg, 'frame_cache'):
                # Frame cache configuration is typically handled by the environment
                # that uses it, so we just validate the configuration exists
                results['frame_cache'] = True
            else:
                results['frame_cache'] = False
        except Exception as e:
            warnings.warn(f"Failed to configure frame cache from Hydra: {e}")
            results['frame_cache'] = False
    else:
        results['frame_cache'] = False
    
    return results


def initialize_reproducibility(seed: int, experiment_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize complete reproducibility environment with comprehensive setup.
    
    Convenience function that sets up seed management, logging, and returns
    complete reproducibility context for experiment documentation.
    
    Args:
        seed: Random seed for reproducibility
        experiment_id: Optional experiment identifier
        
    Returns:
        Dictionary containing complete reproducibility information
        
    Examples:
        >>> repro_info = initialize_reproducibility(42, "exp_001")
        >>> logger = get_enhanced_logger(__name__)
        >>> logger.info(f"Experiment started with seed {repro_info['seed']}")
    """
    # Initialize seed management
    if SEED_MANAGER_AVAILABLE:
        try:
            seed_result = set_global_seed(seed)
            if hasattr(seed_result, 'get_reproducibility_info'):
                repro_info = seed_result.get_reproducibility_info()
            else:
                repro_info = {
                    'seed_value': seed,
                    'experiment_id': experiment_id,
                    'timestamp': __import__('time').time(),
                    'seed_manager_available': True
                }
        except Exception as e:
            warnings.warn(f"Failed to initialize seed management: {e}")
            repro_info = {
                'seed_value': seed,
                'experiment_id': experiment_id,
                'timestamp': __import__('time').time(),
                'seed_manager_available': False,
                'error': str(e)
            }
    else:
        # Fallback to basic seed setting
        set_global_seed(seed)
        repro_info = {
            'seed_value': seed,
            'experiment_id': experiment_id,
            'timestamp': __import__('time').time(),
            'seed_manager_available': False
        }
    
    # Set up enhanced logging if available
    if LOGGING_AVAILABLE:
        try:
            setup_logger()
            repro_info['logging_configured'] = True
        except Exception as e:
            warnings.warn(f"Failed to setup logging: {e}")
            repro_info['logging_configured'] = False
    else:
        repro_info['logging_configured'] = False
    
    return repro_info


# Convenience function for seed context (backward compatibility)
def seed_context(seed: int, operation_name: str = "operation") -> Any:
    """
    Context manager for temporary seed changes with automatic state restoration.
    
    Args:
        seed: Temporary seed value
        operation_name: Name of the operation for tracking
        
    Returns:
        Context manager for scoped seed operations
    """
    if SEED_MANAGER_AVAILABLE:
        return scoped_seed(seed, operation_name)
    else:
        # Fallback context manager
        import contextlib
        import random
        import numpy as np
        
        @contextlib.contextmanager
        def fallback_seed_context():
            old_random_state = random.getstate()
            old_numpy_state = np.random.get_state()
            try:
                random.seed(seed)
                np.random.seed(seed)
                warnings.warn("Using basic seed context, enhanced features not available")
                yield
            finally:
                random.setstate(old_random_state)
                np.random.set_state(old_numpy_state)
        
        return fallback_seed_context()


# Expose availability flags for conditional imports
__availability__ = {
    'cache': CACHE_AVAILABLE,
    'logging': LOGGING_AVAILABLE,
    'visualization': VISUALIZATION_AVAILABLE,
    'seed_manager': SEED_MANAGER_AVAILABLE,
    'io': IO_AVAILABLE,
    'navigator_utils': NAVIGATOR_UTILS_AVAILABLE
}


# Public API - All utility functions accessible through unified imports
__all__ = [
    # Frame cache functionality (prominently featured per Section 0.2.3)
    'FrameCache',
    'CacheMode', 
    'CacheStatistics',
    'create_lru_cache',
    'create_preload_cache',
    'create_no_cache',
    
    # I/O utilities
    'load_yaml',
    'save_yaml', 
    'load_json',
    'save_json',
    'load_numpy',
    'save_numpy',
    'IOError',
    'YAMLError',
    'JSONError',
    'NumpyError',
    
    # Navigator utilities
    'create_navigator_from_config',
    'create_reproducible_navigator',
    'create_navigator_from_params',
    'create_navigator_factory',
    'NavigatorCreationResult',
    'validate_navigator_configuration',
    'normalize_array_parameter',
    'PREDEFINED_SENSOR_LAYOUTS',
    'get_predefined_sensor_layout',
    'define_sensor_offsets',
    'rotate_offset',
    'calculate_sensor_positions',
    'compute_sensor_positions',
    'sample_odor_at_sensors',
    'SingleAgentParams',
    'MultiAgentParams',
    'reset_navigator_state',
    'reset_navigator_state_with_params',
    'update_positions_and_orientations',
    'read_odor_values',
    'navigator_performance_context',
    'get_navigator_capabilities',
    'create_navigator_comparison_report',
    'get_property_name',
    'get_property_value',
    'create_navigator_for_cli',
    'create_navigator_with_database_logging',
    
    # Logging functions
    'setup_logger',
    'get_enhanced_logger',
    'get_module_logger',
    'get_logger',
    'correlation_context',
    'get_correlation_context',
    'set_correlation_context',
    'create_step_timer',
    'step_performance_timer', 
    'frame_rate_timer',
    'update_cache_metrics',
    'log_cache_memory_pressure_violation',
    'detect_legacy_gym_import',
    'log_legacy_api_deprecation',
    'monitor_environment_creation',
    'register_logging_config_schema',
    'LoggingConfig',
    'PerformanceMetrics',
    'FrameCacheConfig',
    'EnhancedLogger',
    'CorrelationContext',
    
    # Visualization functions
    'SimulationVisualization',
    'visualize_trajectory',
    'create_realtime_visualizer',
    'create_static_plotter',
    'VisualizationConfig',
    
    # Seed management functions
    'set_global_seed',
    'get_global_seed_manager',
    'get_random_state',
    'restore_random_state',
    'capture_random_state',
    'reset_random_state',
    'scoped_seed',
    'get_seed_context',
    'seed_sensitive_operation',
    'SeedConfig',
    'setup_global_seed',
    'create_seed_config_from_hydra',
    'RandomState',
    'SeedContext',
    'validate_determinism',
    'is_seeded',
    'get_last_seed',
    'generate_experiment_seed',
    'register_seed_config_schema',
    'get_reproducibility_report',
    'SEED_PERFORMANCE_THRESHOLDS',
    'DETERMINISM_TEST_ITERATIONS',
    'CROSS_PLATFORM_SEED_MAX',
    
    # Unified configuration and initialization
    'configure_from_hydra',
    'initialize_reproducibility',
    'seed_context',
    
    # Logging format constants
    'DEFAULT_FORMAT',
    'MODULE_FORMAT',
    'ENHANCED_FORMAT',
    'HYDRA_FORMAT',
    'CLI_FORMAT',
    'MINIMAL_FORMAT',
    'PRODUCTION_FORMAT',
    'JSON_FORMAT',
    'LOG_LEVELS',
    'PERFORMANCE_THRESHOLDS',
    'ENVIRONMENT_DEFAULTS',
    
    # Availability information
    '__availability__'
]


# Module-level initialization for optimal import experience
def _initialize_module():
    """
    Perform module-level initialization for optimal user experience.
    
    This function sets up basic logging and seed management if they're
    available, providing immediate functionality for common use cases.
    
    Note: Logger initialization is conditional to avoid interfering with
    test scenarios or explicit configurations.
    """
    # Set up basic logging if enhanced logging is available AND no handlers are configured
    if LOGGING_AVAILABLE:
        try:
            # Only configure logging if Loguru has no handlers (default state)
            # This prevents interference with tests or explicit configurations
            from loguru import logger
            if len(logger._core.handlers) == 0:
                # Configure basic enhanced logging with sane defaults
                setup_logger()
        except Exception:
            # Silently fall back to standard logging or no initialization
            pass
    
    # Initialize global seed manager with default if available
    if SEED_MANAGER_AVAILABLE:
        try:
            # Only initialize if no global manager exists
            if get_global_seed_manager() is None:
                # Use a deterministic default seed for reproducibility
                default_seed = 42
                set_global_seed(default_seed)
        except Exception:
            # Silently continue if initialization fails
            pass


# Execute module initialization
_initialize_module()