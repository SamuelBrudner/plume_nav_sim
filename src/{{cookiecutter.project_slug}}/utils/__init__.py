"""
Utility Functions Public API Aggregator

This module serves as the unified entry point for all utility functions in the odor plume
navigation system, providing comprehensive access to I/O operations, logging configuration,
visualization components, seed management, and navigator utilities under a single namespace.

The module supports multiple import patterns for different frameworks and use cases:
- Kedro projects: Direct imports for pipeline integration
- RL frameworks: Seed management and reproducibility functions
- ML/neural network analyses: Comprehensive utility access
- CLI operations: Complete functionality suite

All utility functions integrate with the Hydra configuration system and maintain backward
compatibility with existing interfaces while providing enhanced functionality through
the new project structure.

Examples:
    Basic utility imports:
        >>> from {{cookiecutter.project_slug}}.utils import set_global_seed, setup_enhanced_logging
        >>> setup_enhanced_logging()
        >>> set_global_seed(42)
        
    Visualization and analysis:
        >>> from {{cookiecutter.project_slug}}.utils import visualize_trajectory, SimulationVisualization
        >>> viz = SimulationVisualization.from_config(cfg.visualization)
        >>> visualize_trajectory(positions, orientations, output_path="trajectory.png")
        
    Comprehensive ML/RL workflow:
        >>> from {{cookiecutter.project_slug}}.utils import (
        ...     set_global_seed, setup_enhanced_logging, get_module_logger,
        ...     SimulationVisualization, seed_context
        ... )
        >>> setup_enhanced_logging()
        >>> logger = get_module_logger(__name__)
        >>> with seed_context(42):
        ...     # Reproducible operations
        ...     result = run_experiment()
        
    Hydra configuration integration:
        >>> from {{cookiecutter.project_slug}}.utils import configure_logging_from_hydra, configure_seed_from_hydra
        >>> configure_logging_from_hydra(cfg)
        >>> configure_seed_from_hydra(cfg)
"""

# Core imports with error handling for optional dependencies
import warnings
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Generator
from pathlib import Path

# Enhanced logging system imports
try:
    from .logging import (
        setup_enhanced_logging,
        configure_from_hydra as configure_logging_from_hydra,
        bind_experiment_context,
        track_cli_command,
        get_module_logger,
        log_configuration_override,
        get_logging_metrics,
        EnhancedLogger,
        LoggingConfig,
        ExperimentContext,
        CLICommandTracker,
        ENHANCED_FORMAT,
        HYDRA_FORMAT,
        CLI_FORMAT,
        MINIMAL_FORMAT,
        PRODUCTION_FORMAT
    )
    LOGGING_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Enhanced logging functionality not available: {e}")
    LOGGING_AVAILABLE = False
    # Provide fallback functions
    def setup_enhanced_logging(*args, **kwargs):
        """Fallback logging setup."""
        warnings.warn("Enhanced logging not available, using basic logging")
        import logging
        logging.basicConfig(level=logging.INFO)
    
    def get_module_logger(name: str, **kwargs):
        """Fallback module logger."""
        import logging
        return logging.getLogger(name)

# Visualization system imports
try:
    from .visualization import (
        SimulationVisualization,
        visualize_trajectory,
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

# Seed management system imports
try:
    from .seed_manager import (
        set_global_seed,
        get_global_seed_manager,
        configure_from_hydra as configure_seed_from_hydra,
        seed_context,
        get_reproducibility_report,
        SeedManager,
        RandomState
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


# I/O Utilities - Placeholder implementations for future functionality
# These functions are mentioned in the key changes but not yet implemented
# in the dependency files. Providing placeholder implementations for API completeness.

def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file with error handling and validation.
    
    Args:
        filepath: Path to YAML file
        
    Returns:
        Dictionary containing YAML data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML is invalid
        
    Note:
        This is a placeholder implementation. Enhanced I/O utilities
        will be implemented in future versions.
    """
    import yaml
    from pathlib import Path
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"YAML file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {filepath}: {e}")


def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save dictionary to YAML file with formatting and validation.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        
    Raises:
        IOError: If file cannot be written
        
    Note:
        This is a placeholder implementation. Enhanced I/O utilities
        will be implemented in future versions.
    """
    import yaml
    from pathlib import Path
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save YAML to {filepath}: {e}")


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file with error handling and validation.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary containing JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
        
    Note:
        This is a placeholder implementation. Enhanced I/O utilities
        will be implemented in future versions.
    """
    import json
    from pathlib import Path
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {e}")


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file with formatting.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
        indent: JSON indentation level
        
    Raises:
        IOError: If file cannot be written
        
    Note:
        This is a placeholder implementation. Enhanced I/O utilities
        will be implemented in future versions.
    """
    import json
    from pathlib import Path
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to save JSON to {filepath}: {e}")


def load_numpy(filepath: Union[str, Path]) -> Any:
    """
    Load NumPy array from file with format detection.
    
    Args:
        filepath: Path to NumPy file (.npy, .npz, .txt)
        
    Returns:
        NumPy array or dictionary of arrays
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported
        
    Note:
        This is a placeholder implementation. Enhanced I/O utilities
        will be implemented in future versions.
    """
    import numpy as np
    from pathlib import Path
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"NumPy file not found: {filepath}")
    
    try:
        if filepath.suffix == '.npy':
            return np.load(filepath)
        elif filepath.suffix == '.npz':
            return np.load(filepath)
        elif filepath.suffix in ['.txt', '.csv']:
            return np.loadtxt(filepath)
        else:
            raise ValueError(f"Unsupported NumPy file format: {filepath.suffix}")
    except Exception as e:
        raise ValueError(f"Failed to load NumPy file {filepath}: {e}")


def save_numpy(data: Any, filepath: Union[str, Path], compressed: bool = False) -> None:
    """
    Save NumPy array to file with format selection.
    
    Args:
        data: NumPy array or dictionary of arrays
        filepath: Output file path
        compressed: Use compressed format (.npz)
        
    Raises:
        IOError: If file cannot be written
        
    Note:
        This is a placeholder implementation. Enhanced I/O utilities
        will be implemented in future versions.
    """
    import numpy as np
    from pathlib import Path
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if compressed or filepath.suffix == '.npz':
            if isinstance(data, dict):
                np.savez_compressed(filepath, **data)
            else:
                np.savez_compressed(filepath, data=data)
        elif filepath.suffix == '.npy':
            np.save(filepath, data)
        elif filepath.suffix in ['.txt', '.csv']:
            np.savetxt(filepath, data)
        else:
            # Default to .npy format
            filepath = filepath.with_suffix('.npy')
            np.save(filepath, data)
    except Exception as e:
        raise IOError(f"Failed to save NumPy data to {filepath}: {e}")


# Navigator utilities - Placeholder implementations for backward compatibility
# These functions maintain compatibility with existing navigator interfaces

def normalize_array_parameter(param: Any, expected_shape: Optional[Tuple[int, ...]] = None) -> Any:
    """
    Normalize array parameter to expected shape and type.
    
    Args:
        param: Input parameter (scalar, list, or array)
        expected_shape: Expected output shape
        
    Returns:
        Normalized NumPy array
        
    Note:
        This is a placeholder implementation maintaining backward compatibility.
        Enhanced navigator utilities will be implemented as needed.
    """
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
    """
    Create navigator instance from parameter dictionary.
    
    Args:
        **kwargs: Navigator configuration parameters
        
    Returns:
        Navigator instance
        
    Note:
        This is a placeholder implementation maintaining backward compatibility.
        Use the new API through {{cookiecutter.project_slug}}.api.navigation module.
    """
    warnings.warn(
        "create_navigator_from_params is deprecated. "
        "Use create_navigator from {{cookiecutter.project_slug}}.api.navigation instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Placeholder implementation - redirect to new API when available
    try:
        from ..api.navigation import create_navigator
        return create_navigator(**kwargs)
    except ImportError:
        raise ImportError(
            "Navigator creation not available. "
            "Ensure {{cookiecutter.project_slug}}.api.navigation is properly installed."
        )


def calculate_sensor_positions(position: Tuple[float, float], 
                               orientation: float,
                               sensor_config: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Calculate sensor positions relative to agent position and orientation.
    
    Args:
        position: Agent position as (x, y)
        orientation: Agent orientation in degrees
        sensor_config: Sensor configuration parameters
        
    Returns:
        List of sensor positions as (x, y) tuples
        
    Note:
        This is a placeholder implementation maintaining backward compatibility.
        Enhanced sensor calculations will be integrated with the core module.
    """
    import math
    import numpy as np
    
    x, y = position
    orientation_rad = math.radians(orientation)
    
    # Default sensor configuration if not provided
    num_sensors = sensor_config.get('num_sensors', 2)
    sensor_distance = sensor_config.get('distance', 1.0)
    sensor_angle_spread = sensor_config.get('angle_spread', 45.0)
    
    sensor_positions = []
    
    if num_sensors == 1:
        # Single sensor at agent position
        sensor_positions.append((x, y))
    else:
        # Multiple sensors arranged around agent
        angle_step = math.radians(sensor_angle_spread) / (num_sensors - 1)
        start_angle = orientation_rad - math.radians(sensor_angle_spread) / 2
        
        for i in range(num_sensors):
            sensor_angle = start_angle + i * angle_step
            sensor_x = x + sensor_distance * math.cos(sensor_angle)
            sensor_y = y + sensor_distance * math.sin(sensor_angle)
            sensor_positions.append((sensor_x, sensor_y))
    
    return sensor_positions


def sample_odor_at_sensors(sensor_positions: List[Tuple[float, float]],
                          plume_frame: Any,
                          interpolation_method: str = 'bilinear') -> List[float]:
    """
    Sample odor concentration at sensor positions from plume frame.
    
    Args:
        sensor_positions: List of sensor positions as (x, y) tuples
        plume_frame: 2D array representing odor concentration
        interpolation_method: Interpolation method ('bilinear', 'nearest')
        
    Returns:
        List of odor concentration values at sensor positions
        
    Note:
        This is a placeholder implementation maintaining backward compatibility.
        Enhanced sensor sampling will be integrated with the data module.
    """
    import numpy as np
    from scipy.interpolate import griddata
    
    if not isinstance(plume_frame, np.ndarray):
        plume_frame = np.asarray(plume_frame)
    
    if plume_frame.ndim != 2:
        raise ValueError(f"Plume frame must be 2D array, got {plume_frame.ndim}D")
    
    height, width = plume_frame.shape
    odor_values = []
    
    for x, y in sensor_positions:
        # Clamp coordinates to frame boundaries
        x_idx = max(0, min(width - 1, int(x)))
        y_idx = max(0, min(height - 1, int(y)))
        
        if interpolation_method == 'nearest':
            # Nearest neighbor sampling
            odor_value = plume_frame[y_idx, x_idx]
        else:
            # Bilinear interpolation
            x_frac = x - int(x)
            y_frac = y - int(y)
            
            # Get neighboring pixels
            x0, x1 = int(x), min(int(x) + 1, width - 1)
            y0, y1 = int(y), min(int(y) + 1, height - 1)
            
            # Clamp to valid indices
            x0 = max(0, min(x0, width - 1))
            x1 = max(0, min(x1, width - 1))
            y0 = max(0, min(y0, height - 1))
            y1 = max(0, min(y1, height - 1))
            
            # Bilinear interpolation
            top = plume_frame[y0, x0] * (1 - x_frac) + plume_frame[y0, x1] * x_frac
            bottom = plume_frame[y1, x0] * (1 - x_frac) + plume_frame[y1, x1] * x_frac
            odor_value = top * (1 - y_frac) + bottom * y_frac
        
        odor_values.append(float(odor_value))
    
    return odor_values


# Convenience function for unified Hydra configuration
def configure_from_hydra(cfg: Any, components: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Configure multiple utility components from Hydra configuration.
    
    Provides unified configuration interface for all utility components,
    automatically detecting and configuring available subsystems.
    
    Args:
        cfg: Hydra configuration object
        components: Optional list of specific components to configure
                   ('logging', 'seed_manager'). If None, configures all available.
        
    Returns:
        Dictionary mapping component names to configuration success status
        
    Examples:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        >>> results = configure_from_hydra(cfg)
        >>> print(f"Logging configured: {results['logging']}")
        >>> print(f"Seed manager configured: {results['seed_manager']}")
    """
    results = {}
    available_components = []
    
    # Determine which components to configure
    if components is None:
        if LOGGING_AVAILABLE:
            available_components.append('logging')
        if SEED_MANAGER_AVAILABLE:
            available_components.append('seed_manager')
    else:
        available_components = components
    
    # Configure logging
    if 'logging' in available_components and LOGGING_AVAILABLE:
        try:
            results['logging'] = configure_logging_from_hydra(cfg)
        except Exception as e:
            warnings.warn(f"Failed to configure logging from Hydra: {e}")
            results['logging'] = False
    else:
        results['logging'] = False
    
    # Configure seed manager
    if 'seed_manager' in available_components and SEED_MANAGER_AVAILABLE:
        try:
            results['seed_manager'] = configure_seed_from_hydra(cfg)
        except Exception as e:
            warnings.warn(f"Failed to configure seed manager from Hydra: {e}")
            results['seed_manager'] = False
    else:
        results['seed_manager'] = False
    
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
        >>> logger = get_module_logger(__name__)
        >>> logger.info(f"Experiment started with seed {repro_info['seed']}")
    """
    # Initialize seed management
    if SEED_MANAGER_AVAILABLE:
        seed_manager = set_global_seed(seed, experiment_id=experiment_id)
        repro_info = seed_manager.get_reproducibility_info()
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
        setup_enhanced_logging(experiment_id=experiment_id)
        repro_info['logging_configured'] = True
    else:
        repro_info['logging_configured'] = False
    
    return repro_info


# Expose availability flags for conditional imports
__availability__ = {
    'logging': LOGGING_AVAILABLE,
    'visualization': VISUALIZATION_AVAILABLE,
    'seed_manager': SEED_MANAGER_AVAILABLE
}


# Public API - All utility functions accessible through unified imports
__all__ = [
    # I/O utilities
    'load_yaml',
    'save_yaml', 
    'load_json',
    'save_json',
    'load_numpy',
    'save_numpy',
    
    # Navigator utilities (backward compatibility)
    'normalize_array_parameter',
    'create_navigator_from_params',
    'calculate_sensor_positions',
    'sample_odor_at_sensors',
    
    # Logging functions
    'setup_enhanced_logging',
    'configure_logging_from_hydra',
    'bind_experiment_context',
    'track_cli_command',
    'get_module_logger',
    'log_configuration_override',
    'get_logging_metrics',
    'EnhancedLogger',
    'LoggingConfig',
    'ExperimentContext',
    'CLICommandTracker',
    
    # Visualization functions
    'SimulationVisualization',
    'visualize_trajectory',
    'VisualizationConfig',
    
    # Seed management functions
    'set_global_seed',
    'get_global_seed_manager',
    'configure_seed_from_hydra',
    'seed_context',
    'get_reproducibility_report',
    'SeedManager',
    'RandomState',
    
    # Unified configuration and initialization
    'configure_from_hydra',
    'initialize_reproducibility',
    
    # Logging format constants
    'ENHANCED_FORMAT',
    'HYDRA_FORMAT',
    'CLI_FORMAT',
    'MINIMAL_FORMAT',
    'PRODUCTION_FORMAT',
    
    # Availability information
    '__availability__'
]


# Module-level initialization for optimal import experience
def _initialize_module():
    """
    Perform module-level initialization for optimal user experience.
    
    This function sets up basic logging and seed management if they're
    available, providing immediate functionality for common use cases.
    """
    # Set up basic logging if enhanced logging is available
    if LOGGING_AVAILABLE:
        try:
            # Configure basic enhanced logging with sane defaults
            setup_enhanced_logging()
        except Exception:
            # Silently fall back to standard logging
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