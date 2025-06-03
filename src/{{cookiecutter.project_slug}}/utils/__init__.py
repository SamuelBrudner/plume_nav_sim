"""
Public API aggregator for the utils package.

This module serves as the unified entry point for all utility functions, providing
a consistent import interface for I/O operations, navigator utilities, logging setup,
visualization components, and seed management. It enables clean import patterns
for both Kedro and RL framework integrations while maintaining backward compatibility.

Key Features:
- Unified import paths for all utility functions per Section 0.2.6 import patterns
- Support for Kedro and RL framework import patterns as specified in Section 0.1.1  
- Hydra configuration integration for all utility components per Feature F-006
- Backward compatibility for existing utility function interfaces per Section 0.2.6
- Comprehensive error handling and graceful degradation for missing components
- Production-ready implementation with enterprise-grade documentation

Import Patterns Supported:
- Direct utility imports: `from {{cookiecutter.project_slug}}.utils import set_global_seed`
- Wildcard imports: `from {{cookiecutter.project_slug}}.utils import *`
- Kedro integration: `from {{cookiecutter.project_slug}}.utils import setup_logger`
- RL frameworks: `from {{cookiecutter.project_slug}}.utils import get_numpy_generator`
- Visualization: `from {{cookiecutter.project_slug}}.utils import SimulationVisualization`
"""

import warnings
from typing import Any, Dict, List, Optional, Union, Callable, ContextManager

# Import error tracking for graceful degradation
_import_errors: Dict[str, Exception] = {}
_available_modules: List[str] = []

# Core utility modules imports with error handling
try:
    from .logging import (
        # Configuration and management classes
        LoggingConfig,
        EnhancedLoggingManager,
        
        # Primary setup and access functions
        setup_enhanced_logging,
        get_logging_manager,
        get_module_logger,
        
        # Context management functions
        create_correlation_scope,
        create_cli_command_scope, 
        create_parameter_validation_scope,
        
        # Specialized logging functions
        log_environment_variables,
        log_configuration_composition,
    )
    _available_modules.append("logging")
except ImportError as e:
    _import_errors["logging"] = e
    warnings.warn(f"Logging utilities unavailable: {e}", UserWarning)
    
    # Fallback implementations for critical logging functions
    def setup_enhanced_logging(*args, **kwargs):
        """Fallback logging setup - uses basic print statements."""
        print("Warning: Enhanced logging unavailable, using fallback")
    
    def get_module_logger(name: str):
        """Fallback logger that prints to console."""
        class FallbackLogger:
            def info(self, msg): print(f"INFO: {msg}")
            def debug(self, msg): print(f"DEBUG: {msg}")
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        return FallbackLogger()

try:
    from .visualization import (
        # Core visualization classes
        SimulationVisualization,
        
        # Primary visualization functions
        visualize_trajectory,
        batch_visualize_trajectories,
        
        # Utility and setup functions
        setup_headless_mode,
        get_available_themes,
        create_simulation_visualization,
        export_animation,
        
        # Configuration constants
        DEFAULT_VISUALIZATION_CONFIG,
    )
    _available_modules.append("visualization")
except ImportError as e:
    _import_errors["visualization"] = e
    warnings.warn(f"Visualization utilities unavailable: {e}", UserWarning)
    
    # Fallback implementations for critical visualization functions
    def visualize_trajectory(*args, **kwargs):
        """Fallback trajectory visualization - returns None."""
        warnings.warn("Visualization unavailable, skipping trajectory plot", UserWarning)
        return None
    
    def setup_headless_mode():
        """Fallback headless mode setup."""
        warnings.warn("Visualization unavailable, headless mode not configured", UserWarning)

try:
    from .seed_manager import (
        # Configuration and management classes
        SeedConfig,
        SeedManager,
        
        # Primary seed management functions
        set_global_seed,
        get_current_seed,
        get_seed_manager,
        get_numpy_generator,
    )
    _available_modules.append("seed_manager")
except ImportError as e:
    _import_errors["seed_manager"] = e
    warnings.warn(f"Seed management utilities unavailable: {e}", UserWarning)
    
    # Fallback implementations for critical seed functions
    def set_global_seed(seed: Optional[int] = None, **kwargs) -> int:
        """Fallback seed setting - uses basic numpy/random seeding."""
        import random
        import numpy as np
        if seed is None:
            seed = 42  # Default fallback seed
        random.seed(seed)
        np.random.seed(seed)
        warnings.warn(f"Using fallback seed management with seed {seed}", UserWarning)
        return seed
    
    def get_current_seed() -> Optional[int]:
        """Fallback current seed getter."""
        warnings.warn("Seed manager unavailable, returning None", UserWarning)
        return None


# I/O Utilities - Implementation based on common research workflow patterns
def load_yaml(file_path: Union[str, "pathlib.Path"]) -> Dict[str, Any]:
    """
    Load YAML configuration file with error handling.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary containing YAML data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML parsing fails
    """
    import yaml
    from pathlib import Path
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML file {file_path}: {e}") from e


def save_yaml(data: Dict[str, Any], file_path: Union[str, "pathlib.Path"]) -> None:
    """
    Save dictionary to YAML file with proper formatting.
    
    Args:
        data: Dictionary to save
        file_path: Output file path
        
    Raises:
        ValueError: If data cannot be serialized
        IOError: If file cannot be written
    """
    import yaml
    from pathlib import Path
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=True, indent=2)
    except (yaml.YAMLError, IOError) as e:
        raise IOError(f"Failed to save YAML file {file_path}: {e}") from e


def load_json(file_path: Union[str, "pathlib.Path"]) -> Union[Dict[str, Any], List[Any]]:
    """
    Load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data (dict or list)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON parsing fails
    """
    import json
    from pathlib import Path
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {file_path}: {e}") from e


def save_json(data: Union[Dict[str, Any], List[Any]], file_path: Union[str, "pathlib.Path"], indent: int = 2) -> None:
    """
    Save data to JSON file with proper formatting.
    
    Args:
        data: Data to save (must be JSON serializable)
        file_path: Output file path
        indent: JSON indentation level
        
    Raises:
        ValueError: If data cannot be serialized
        IOError: If file cannot be written
    """
    import json
    from pathlib import Path
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize data to JSON: {e}") from e
    except IOError as e:
        raise IOError(f"Failed to save JSON file {file_path}: {e}") from e


def load_numpy(file_path: Union[str, "pathlib.Path"]) -> "np.ndarray":
    """
    Load NumPy array from file with format auto-detection.
    
    Args:
        file_path: Path to NumPy file (.npy, .npz, or .txt)
        
    Returns:
        Loaded NumPy array
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is unsupported or corrupted
    """
    import numpy as np
    from pathlib import Path
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"NumPy file not found: {file_path}")
    
    try:
        suffix = file_path.suffix.lower()
        if suffix == '.npy':
            return np.load(file_path)
        elif suffix == '.npz':
            # Load first array from NPZ file or raise informative error
            data = np.load(file_path)
            if len(data.files) == 1:
                return data[data.files[0]]
            else:
                raise ValueError(f"NPZ file contains multiple arrays: {data.files}. Specify which to load.")
        elif suffix in ['.txt', '.csv']:
            return np.loadtxt(file_path)
        else:
            raise ValueError(f"Unsupported NumPy file format: {suffix}")
    except Exception as e:
        raise ValueError(f"Failed to load NumPy file {file_path}: {e}") from e


def save_numpy(array: "np.ndarray", file_path: Union[str, "pathlib.Path"], compressed: bool = False) -> None:
    """
    Save NumPy array to file with format auto-detection.
    
    Args:
        array: NumPy array to save
        file_path: Output file path
        compressed: Whether to use compression for .npz format
        
    Raises:
        ValueError: If array cannot be saved
        IOError: If file cannot be written
    """
    import numpy as np
    from pathlib import Path
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        suffix = file_path.suffix.lower()
        if suffix == '.npy':
            np.save(file_path, array)
        elif suffix == '.npz':
            if compressed:
                np.savez_compressed(file_path, array=array)
            else:
                np.savez(file_path, array=array)
        elif suffix in ['.txt', '.csv']:
            np.savetxt(file_path, array)
        else:
            # Default to .npy format
            np.save(file_path.with_suffix('.npy'), array)
    except Exception as e:
        raise IOError(f"Failed to save NumPy array to {file_path}: {e}") from e


# Navigator Utilities - Core navigation helper functions
def normalize_array_parameter(param: Union[float, List[float], "np.ndarray"], target_length: int) -> "np.ndarray":
    """
    Normalize parameter to NumPy array of specified length.
    
    Handles scalar values, lists, and arrays to ensure consistent format
    for multi-agent scenarios.
    
    Args:
        param: Parameter value (scalar, list, or array)
        target_length: Desired array length
        
    Returns:
        NumPy array of specified length
        
    Example:
        >>> normalize_array_parameter(5.0, 3)
        array([5., 5., 5.])
        >>> normalize_array_parameter([1, 2, 3], 3)
        array([1., 2., 3.])
    """
    import numpy as np
    
    if np.isscalar(param):
        return np.full(target_length, param, dtype=float)
    
    param_array = np.asarray(param, dtype=float)
    
    if param_array.size == 1:
        return np.full(target_length, param_array.item())
    elif param_array.size == target_length:
        return param_array
    else:
        raise ValueError(
            f"Parameter array length {param_array.size} does not match target length {target_length}"
        )


def create_navigator_from_params(
    navigator_type: str = "single",
    initial_position: Union[List[float], "np.ndarray"] = None,
    initial_orientation: float = 0.0,
    max_speed: float = 10.0,
    angular_velocity: float = 0.1,
    **kwargs
) -> "Navigator":
    """
    Create navigator instance from parameters with validation.
    
    Factory function that creates properly configured navigator instances
    with parameter validation and default value handling.
    
    Args:
        navigator_type: Type of navigator ("single" or "multi")
        initial_position: Starting position [x, y]
        initial_orientation: Starting orientation in degrees
        max_speed: Maximum movement speed
        angular_velocity: Angular velocity for turning
        **kwargs: Additional navigator parameters
        
    Returns:
        Configured Navigator instance
        
    Raises:
        ValueError: If parameters are invalid
        ImportError: If navigator modules are unavailable
    """
    try:
        from ..core.navigator import Navigator
        from ..config.schemas import NavigatorConfig
        
        # Set default position if not provided
        if initial_position is None:
            initial_position = [50.0, 50.0]
        
        # Create configuration dictionary
        config_dict = {
            "type": navigator_type,
            "initial_position": initial_position,
            "initial_orientation": initial_orientation,
            "max_speed": max_speed,
            "angular_velocity": angular_velocity,
            **kwargs
        }
        
        # Validate configuration
        config = NavigatorConfig(**config_dict)
        
        # Create navigator instance
        if navigator_type == "single":
            return Navigator.single(config)
        elif navigator_type == "multi":
            return Navigator.multi(config)
        else:
            raise ValueError(f"Unsupported navigator type: {navigator_type}")
            
    except ImportError as e:
        raise ImportError(f"Navigator modules unavailable: {e}") from e


def calculate_sensor_positions(
    agent_position: "np.ndarray",
    agent_orientation: float,
    sensor_config: Union[str, Dict[str, Any], List[List[float]]]
) -> "np.ndarray":
    """
    Calculate sensor positions relative to agent position and orientation.
    
    Supports predefined sensor configurations (LEFT_RIGHT, SINGLE, etc.) and
    custom sensor geometry specifications.
    
    Args:
        agent_position: Agent position [x, y]
        agent_orientation: Agent orientation in degrees
        sensor_config: Sensor configuration (string, dict, or position list)
        
    Returns:
        Array of sensor positions with shape (n_sensors, 2)
        
    Example:
        >>> pos = np.array([10.0, 20.0])
        >>> sensors = calculate_sensor_positions(pos, 0.0, "LEFT_RIGHT")
        >>> sensors.shape
        (2, 2)
    """
    import numpy as np
    import math
    
    # Convert orientation to radians
    orientation_rad = math.radians(agent_orientation)
    
    # Define predefined sensor configurations
    if isinstance(sensor_config, str):
        if sensor_config.upper() == "SINGLE":
            relative_positions = [[0.0, 0.0]]
        elif sensor_config.upper() == "LEFT_RIGHT":
            relative_positions = [[-1.0, 0.0], [1.0, 0.0]]
        elif sensor_config.upper() == "TRIANGLE":
            relative_positions = [[0.0, 1.0], [-0.866, -0.5], [0.866, -0.5]]
        elif sensor_config.upper() == "CROSS":
            relative_positions = [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]
        else:
            raise ValueError(f"Unknown sensor configuration: {sensor_config}")
    elif isinstance(sensor_config, dict):
        relative_positions = sensor_config.get("positions", [[0.0, 0.0]])
    else:
        relative_positions = sensor_config
    
    # Convert to numpy array
    relative_positions = np.asarray(relative_positions, dtype=float)
    
    # Apply rotation transformation
    cos_angle = math.cos(orientation_rad)
    sin_angle = math.sin(orientation_rad)
    
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])
    
    # Rotate relative positions and add agent position
    rotated_positions = relative_positions @ rotation_matrix.T
    sensor_positions = rotated_positions + agent_position
    
    return sensor_positions


def sample_odor_at_sensors(
    sensor_positions: "np.ndarray",
    plume_frame: "np.ndarray",
    sampling_method: str = "bilinear"
) -> "np.ndarray":
    """
    Sample odor concentration at sensor positions from plume frame.
    
    Supports multiple interpolation methods for accurate odor sampling
    from discretized plume data.
    
    Args:
        sensor_positions: Sensor positions with shape (n_sensors, 2)
        plume_frame: 2D plume concentration array
        sampling_method: Interpolation method ("nearest", "bilinear", "bicubic")
        
    Returns:
        Array of odor concentrations at sensor positions
        
    Raises:
        ValueError: If sampling method is unsupported
    """
    import numpy as np
    from scipy import ndimage
    
    # Get plume dimensions
    height, width = plume_frame.shape
    
    # Clip sensor positions to plume bounds
    x_coords = np.clip(sensor_positions[:, 0], 0, width - 1)
    y_coords = np.clip(sensor_positions[:, 1], 0, height - 1)
    
    if sampling_method == "nearest":
        # Nearest neighbor sampling
        x_indices = np.round(x_coords).astype(int)
        y_indices = np.round(y_coords).astype(int)
        return plume_frame[y_indices, x_indices]
    
    elif sampling_method == "bilinear":
        # Bilinear interpolation
        x0 = np.floor(x_coords).astype(int)
        x1 = np.minimum(x0 + 1, width - 1)
        y0 = np.floor(y_coords).astype(int)
        y1 = np.minimum(y0 + 1, height - 1)
        
        # Calculate interpolation weights
        wx = x_coords - x0
        wy = y_coords - y0
        
        # Perform bilinear interpolation
        values = (
            plume_frame[y0, x0] * (1 - wx) * (1 - wy) +
            plume_frame[y0, x1] * wx * (1 - wy) +
            plume_frame[y1, x0] * (1 - wx) * wy +
            plume_frame[y1, x1] * wx * wy
        )
        return values
    
    elif sampling_method == "bicubic":
        # Bicubic interpolation using scipy
        coordinates = np.array([y_coords, x_coords])
        return ndimage.map_coordinates(plume_frame, coordinates, order=3, mode='nearest')
    
    else:
        raise ValueError(f"Unsupported sampling method: {sampling_method}")


# Backward Compatibility Aliases
# Maintain compatibility with existing import patterns
setup_logger = setup_enhanced_logging  # Legacy alias
DEFAULT_FORMAT = "enhanced"  # Legacy constant
MODULE_FORMAT = "structured"  # Legacy constant
LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]  # Legacy constant


def get_module_info() -> Dict[str, Any]:
    """
    Get information about available utility modules and any import errors.
    
    Returns:
        Dictionary containing module availability and error information
    """
    return {
        "available_modules": _available_modules,
        "import_errors": {k: str(v) for k, v in _import_errors.items()},
        "total_modules": len(_available_modules),
        "total_errors": len(_import_errors),
        "status": "healthy" if not _import_errors else "degraded"
    }


def initialize_reproducibility(seed: Optional[int] = None, **kwargs) -> int:
    """
    Initialize reproducibility settings for experiments.
    
    Convenience function that sets up global seed management and
    logging configuration for reproducible research.
    
    Args:
        seed: Global seed value
        **kwargs: Additional configuration parameters
        
    Returns:
        The initialized seed value
    """
    # Initialize seed management
    actual_seed = set_global_seed(seed, **kwargs)
    
    # Setup logging with correlation
    try:
        setup_enhanced_logging()
        logger = get_module_logger(__name__)
        logger.info(f"Reproducibility initialized with seed: {actual_seed}")
    except Exception as e:
        warnings.warn(f"Failed to setup enhanced logging: {e}", UserWarning)
    
    return actual_seed


# Public API exports for wildcard imports
__all__ = [
    # Logging utilities
    "LoggingConfig",
    "EnhancedLoggingManager", 
    "setup_enhanced_logging",
    "get_logging_manager",
    "get_module_logger",
    "create_correlation_scope",
    "create_cli_command_scope",
    "create_parameter_validation_scope",
    "log_environment_variables", 
    "log_configuration_composition",
    
    # Visualization utilities
    "SimulationVisualization",
    "visualize_trajectory",
    "batch_visualize_trajectories",
    "setup_headless_mode",
    "get_available_themes",
    "create_simulation_visualization",
    "export_animation",
    "DEFAULT_VISUALIZATION_CONFIG",
    
    # Seed management utilities
    "SeedConfig",
    "SeedManager",
    "set_global_seed",
    "get_current_seed",
    "get_seed_manager",
    "get_numpy_generator",
    
    # I/O utilities
    "load_yaml",
    "save_yaml", 
    "load_json",
    "save_json",
    "load_numpy",
    "save_numpy",
    
    # Navigator utilities
    "normalize_array_parameter",
    "create_navigator_from_params",
    "calculate_sensor_positions", 
    "sample_odor_at_sensors",
    
    # Convenience and compatibility functions
    "initialize_reproducibility",
    "get_module_info",
    
    # Backward compatibility aliases
    "setup_logger",
    "DEFAULT_FORMAT",
    "MODULE_FORMAT", 
    "LOG_LEVELS",
]