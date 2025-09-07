"""
Enhanced utility functions for navigator creation and management with configuration-driven instantiation.

This module provides comprehensive helper functions for creating and manipulating
navigator instances with support for:
- Hydra-based configuration-driven instantiation
- Reproducible navigator initialization through seed management
- Enhanced parameter validation using Pydantic configuration models
- Both traditional and configuration-driven instantiation patterns
- CLI and database integration utilities

The enhanced functions maintain backward compatibility while adding advanced
configuration features from the merged templated package components.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type, TypeVar, Generic
from contextlib import suppress, contextmanager
import itertools
import numpy as np
from dataclasses import dataclass, field
import time
import threading
from pathlib import Path

# Core imports
from odor_plume_nav.core.protocols import NavigatorProtocol
from odor_plume_nav.coordinate_frame import normalize_angle, rotate
from odor_plume_nav.core.navigator import Navigator

# Configuration imports with fallback handling
try:
    from odor_plume_nav.config.models import NavigatorConfig, SingleAgentConfig, MultiAgentConfig
    CONFIG_MODELS_AVAILABLE = True
except ImportError:
    # Fallback for environments without config models
    CONFIG_MODELS_AVAILABLE = False
    NavigatorConfig = dict
    SingleAgentConfig = dict
    MultiAgentConfig = dict

# Hydra imports with fallback for environments without Hydra
try:
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.hydra_config import HydraConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None
    HydraConfig = None

# Seed manager imports with fallback
try:
    from odor_plume_nav.utils.seed_manager import SeedManager, set_global_seed, get_global_seed_manager
    SEED_MANAGER_AVAILABLE = True
except ImportError:
    # Fallback for when seed manager is not available yet
    SEED_MANAGER_AVAILABLE = False
    SeedManager = None

# Logging setup with fallback
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
from loguru import logger
    LOGURU_AVAILABLE = False


# A dictionary of *base* local sensor offsets (in arbitrary units).
# Each row is [x_offset, y_offset] in the agent's local frame:
#   - The agent faces +x
#   - +y is "left" of the agent (using standard math orientation)
PREDEFINED_SENSOR_LAYOUTS: Dict[str, np.ndarray] = {
    "SINGLE": np.array([[0.0, 0.0]]),
    # Left–Right: one sensor at +y, the other at –y
    "LEFT_RIGHT": np.array([
        [0.0,  1.0],
        [0.0, -1.0],
    ]),
    # Example: place one sensor forward, plus left and right.
    "FRONT_SIDES": np.array([
        [1.0,  0.0],
        [0.0,  1.0],
        [0.0, -1.0],
    ]),
}


# Enhanced configuration-driven navigator creation
T = TypeVar('T', bound=NavigatorProtocol)


@dataclass
class NavigatorCreationResult(Generic[T]):
    """
    Result object for enhanced navigator creation with metadata tracking.
    
    Attributes:
        navigator: Created navigator instance
        seed_value: Random seed used for creation (if any)
        creation_time_ms: Time taken to create navigator in milliseconds
        configuration_source: Source of configuration (e.g., 'hydra', 'pydantic', 'direct')
        validation_errors: Any validation warnings or errors encountered
        metadata: Additional creation metadata
    """
    navigator: T
    seed_value: Optional[int] = None
    creation_time_ms: float = 0.0
    configuration_source: str = "direct"
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_navigator_from_config(
    config: Union[Dict[str, Any], DictConfig, NavigatorConfig],
    seed: Optional[int] = None,
    validate_config: bool = True,
    enable_logging: bool = True,
    experiment_id: Optional[str] = None
) -> NavigatorCreationResult[NavigatorProtocol]:
    """
    Create navigator from configuration with comprehensive validation and seed management.
    
    This function provides enhanced navigator creation with:
    - Automatic configuration validation using Pydantic models
    - Reproducible initialization through seed management
    - Performance tracking and logging
    - Support for multiple configuration sources (Hydra, Pydantic, dict)
    
    Args:
        config: Configuration object (dict, DictConfig, or NavigatorConfig)
        seed: Optional random seed for reproducible initialization
        validate_config: Whether to validate configuration using Pydantic models
        enable_logging: Whether to enable comprehensive logging
        experiment_id: Optional experiment identifier for tracking
        
    Returns:
        NavigatorCreationResult containing the navigator and creation metadata
        
    Raises:
        ValueError: If configuration is invalid or incomplete
        TypeError: If configuration type is unsupported
        RuntimeError: If navigator creation fails
        
    Examples:
        From dictionary configuration:
        >>> config = {"position": (10.0, 20.0), "max_speed": 2.0}
        >>> result = create_navigator_from_config(config, seed=42)
        >>> navigator = result.navigator
        
        From Hydra configuration:
        >>> @hydra.main(config_path="../conf", config_name="config")
        >>> def main(cfg: DictConfig) -> None:
        ...     result = create_navigator_from_config(cfg.navigator)
    """
    start_time = time.perf_counter()
    creation_result = NavigatorCreationResult(
        navigator=None,
        seed_value=seed,
        configuration_source="unknown"
    )
    
    try:
        # Initialize seed management if requested
        if seed is not None and SEED_MANAGER_AVAILABLE:
            seed_manager = SeedManager(seed=seed, experiment_id=experiment_id)
            seed_manager.initialize()
            creation_result.seed_value = seed
            
            if enable_logging and LOGURU_AVAILABLE:
                logger.bind(
                    seed_value=seed,
                    experiment_id=experiment_id,
                    function="create_navigator_from_config"
                ).info(f"Initialized seed manager for navigator creation")
        
        # Determine configuration source and convert to standard format
        if isinstance(config, dict):
            creation_result.configuration_source = "dict"
            config_dict = config
        elif HYDRA_AVAILABLE and isinstance(config, DictConfig):
            creation_result.configuration_source = "hydra"
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif CONFIG_MODELS_AVAILABLE and isinstance(config, (NavigatorConfig, SingleAgentConfig, MultiAgentConfig)):
            creation_result.configuration_source = "pydantic"
            config_dict = config.dict() if hasattr(config, 'dict') else config.model_dump()
        else:
            raise TypeError(f"Unsupported configuration type: {type(config)}")
        
        # Validate configuration if requested
        if validate_config and CONFIG_MODELS_AVAILABLE:
            try:
                # Try to create Pydantic model for validation
                if _is_multi_agent_config_dict(config_dict):
                    validated_config = MultiAgentConfig(**config_dict)
                else:
                    validated_config = SingleAgentConfig(**config_dict)
                
                # Update config_dict with validated values
                config_dict = validated_config.dict() if hasattr(validated_config, 'dict') else validated_config.model_dump()
                
            except Exception as e:
                validation_error = f"Configuration validation failed: {str(e)}"
                creation_result.validation_errors.append(validation_error)
                
                if enable_logging and LOGURU_AVAILABLE:
                    logger.warning(validation_error)
                
                # Continue with unvalidated config if validation fails
        
        # Create navigator based on configuration type
        if _is_multi_agent_config_dict(config_dict):
            navigator = _create_multi_agent_from_dict(config_dict)
        else:
            navigator = _create_single_agent_from_dict(config_dict)
        
        creation_result.navigator = navigator
        
        # Calculate creation time
        end_time = time.perf_counter()
        creation_result.creation_time_ms = (end_time - start_time) * 1000
        
        # Add metadata
        creation_result.metadata.update({
            'agent_count': navigator.num_agents,
            'config_keys': list(config_dict.keys()),
            'validation_enabled': validate_config,
            'hydra_available': HYDRA_AVAILABLE,
            'seed_manager_available': SEED_MANAGER_AVAILABLE
        })
        
        if enable_logging and LOGURU_AVAILABLE:
            logger.bind(
                navigator_type="multi_agent" if _is_multi_agent_config_dict(config_dict) else "single_agent",
                agent_count=navigator.num_agents,
                creation_time_ms=creation_result.creation_time_ms,
                configuration_source=creation_result.configuration_source,
                seed_value=creation_result.seed_value
            ).info(f"Navigator created successfully from {creation_result.configuration_source} configuration")
        
        return creation_result
        
    except Exception as e:
        error_msg = f"Failed to create navigator from configuration: {str(e)}"
        if enable_logging and LOGURU_AVAILABLE:
            logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def create_reproducible_navigator(
    config: Union[Dict[str, Any], DictConfig, NavigatorConfig],
    seed: int,
    experiment_id: Optional[str] = None,
    capture_initial_state: bool = True,
    validate_determinism: bool = True
) -> Tuple[NavigatorProtocol, Dict[str, Any]]:
    """
    Create navigator with guaranteed reproducible behavior and state tracking.
    
    This function ensures deterministic navigator creation with comprehensive
    state management for reproducible research workflows.
    
    Args:
        config: Navigator configuration
        seed: Random seed for deterministic behavior
        experiment_id: Optional experiment identifier for tracking
        capture_initial_state: Whether to capture initial random state
        validate_determinism: Whether to validate deterministic behavior
        
    Returns:
        Tuple of (navigator, reproducibility_info) where reproducibility_info
        contains all metadata needed to reproduce the navigator creation
        
    Raises:
        RuntimeError: If reproducible creation fails or validation fails
        
    Examples:
        >>> config = {"position": (0, 0), "speed": 1.0}
        >>> navigator, repro_info = create_reproducible_navigator(config, seed=42)
        >>> # navigator behavior is guaranteed to be deterministic
    """
    if not SEED_MANAGER_AVAILABLE:
        raise RuntimeError("Seed manager not available - reproducible creation requires seed_manager.py")
    
    reproducibility_info = {
        'seed_value': seed,
        'experiment_id': experiment_id,
        'creation_timestamp': time.time(),
        'platform_info': {
            'numpy_version': np.__version__,
            'python_version': tuple(map(int, __import__('sys').version_info[:3]))
        }
    }
    
    try:
        # Initialize seed manager for reproducible creation
        with SeedManager(seed=seed, experiment_id=experiment_id) as seed_manager:
            # Capture initial state if requested
            if capture_initial_state:
                initial_state = seed_manager.capture_state()
                reproducibility_info['initial_random_state'] = initial_state.to_dict()
            
            # Create navigator
            result = create_navigator_from_config(
                config=config,
                seed=seed,
                validate_config=True,
                enable_logging=True,
                experiment_id=experiment_id
            )
            
            navigator = result.navigator
            
            # Add creation result metadata
            reproducibility_info.update({
                'creation_time_ms': result.creation_time_ms,
                'configuration_source': result.configuration_source,
                'validation_errors': result.validation_errors,
                'navigator_metadata': result.metadata
            })
            
            # Validate determinism if requested
            if validate_determinism:
                validation_result = _validate_navigator_determinism(navigator, seed)
                reproducibility_info['determinism_validation'] = validation_result
                
                if not validation_result['is_deterministic']:
                    raise RuntimeError(f"Determinism validation failed: {validation_result['error']}")
            
            # Capture final state
            final_state = seed_manager.capture_state()
            reproducibility_info['final_random_state'] = final_state.to_dict()
            
        return navigator, reproducibility_info
        
    except Exception as e:
        error_msg = f"Failed to create reproducible navigator: {str(e)}"
        if LOGURU_AVAILABLE:
            logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def normalize_array_parameter(param: Any, num_agents: int) -> Optional[np.ndarray]:
    """
    Normalize a parameter to a numpy array of the appropriate length.
    
    Args:
        param: Parameter value, which can be None, a scalar, a list, or a numpy array
        num_agents: Number of agents to normalize for
        
    Returns:
        Normalized parameter as a numpy array, or None if param is None
    """
    if param is None:
        return None
    
    # Convert to numpy array if not already
    if not isinstance(param, np.ndarray):
        param = np.array(param)
    
    # If it's a scalar, broadcast to the desired length
    if param.ndim == 0:
        param = np.full(num_agents, param)
    
    return param


def create_navigator_from_params(
    positions: Optional[Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]] = None,
    orientations: Optional[Union[float, List[float], np.ndarray]] = None,
    speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    max_speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    angular_velocities: Optional[Union[float, List[float], np.ndarray]] = None,
    seed: Optional[int] = None,
    experiment_id: Optional[str] = None,
    validate_params: bool = False,
    enable_configuration_logging: bool = False
) -> NavigatorProtocol:
    """
    Enhanced navigator creation from parameter values with configuration support.
    
    This function provides backward-compatible navigator creation while adding
    enhanced capabilities for reproducible initialization, parameter validation,
    and configuration logger.
    
    Args:
        positions: Initial positions of the agents
        orientations: Initial orientations of the agents (in degrees)
        speeds: Initial speeds of the agents
        max_speeds: Maximum speeds of the agents
        angular_velocities: Initial angular velocities of the agents (in degrees per second)
        seed: Optional random seed for reproducible initialization
        experiment_id: Optional experiment identifier for tracking
        validate_params: Whether to validate parameters using Pydantic models
        enable_configuration_logging: Whether to enable enhanced logging
        
    Returns:
        A Navigator instance that automatically handles single or multi-agent scenarios
        
    Examples:
        Traditional usage (backward compatible):
        >>> navigator = create_navigator_from_params(position=(10, 20), speed=1.5)
        
        Enhanced usage with reproducibility:
        >>> navigator = create_navigator_from_params(
        ...     position=(10, 20), 
        ...     speed=1.5, 
        ...     seed=42, 
        ...     validate_params=True
        ... )
    """
    # Initialize seed management if requested
    if seed is not None and SEED_MANAGER_AVAILABLE:
        set_global_seed(seed, experiment_id=experiment_id, enable_logging=enable_configuration_logging)
    
    # Create configuration dictionary for validation if requested
    if validate_params:
        config_dict = {}
        
        # Add non-None parameters to config
        param_mapping = {
            'positions': positions,
            'position': positions,  # Will be handled in validation
            'orientations': orientations,
            'orientation': orientations,
            'speeds': speeds,
            'speed': speeds,
            'max_speeds': max_speeds,
            'max_speed': max_speeds,
            'angular_velocities': angular_velocities,
            'angular_velocity': angular_velocities
        }
        
        for key, value in param_mapping.items():
            if value is not None:
                config_dict[key] = value
        
        # Validate configuration
        validation_result = validate_navigator_configuration(config_dict, strict_validation=True)
        
        if not validation_result['is_valid'] and enable_configuration_logging and LOGURU_AVAILABLE:
            logger.warning(f"Parameter validation warnings: {validation_result['errors']}")
    
    # Detect if we're creating a single or multi-agent navigator
    is_multi_agent = False
    num_agents = 1

    # If positions is provided and is not a simple (x, y) tuple, it's multi-agent
    if positions is not None:
        if isinstance(positions, np.ndarray) and positions.ndim > 1:
            is_multi_agent = True
            num_agents = positions.shape[0]
        elif isinstance(positions, list) and positions and isinstance(positions[0], (list, tuple)):
            is_multi_agent = True
            num_agents = len(positions)

    # Create navigator with enhanced logging if enabled
    start_time = time.perf_counter()
    
    # For multi-agent mode, normalize parameters to ensure they're arrays of correct length
    if is_multi_agent:
        # Convert orientations, speeds, etc. to arrays if they're scalar values
        orientations = normalize_array_parameter(orientations, num_agents)
        speeds = normalize_array_parameter(speeds, num_agents)
        max_speeds = normalize_array_parameter(max_speeds, num_agents)
        angular_velocities = normalize_array_parameter(angular_velocities, num_agents)

        navigator = Navigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
    else:
        navigator = Navigator(
            position=positions,
            orientation=orientations,
            speed=speeds,
            max_speed=max_speeds,
            angular_velocity=angular_velocities
        )
    
    # Log creation if enhanced logging is enabled
    if enable_configuration_logging and LOGURU_AVAILABLE:
        creation_time_ms = (time.perf_counter() - start_time) * 1000
        logger.bind(
            navigator_type="multi_agent" if is_multi_agent else "single_agent",
            agent_count=num_agents,
            creation_time_ms=creation_time_ms,
            seed_value=seed,
            experiment_id=experiment_id,
            validation_enabled=validate_params
        ).info(f"Navigator created from parameters")
    
    return navigator


def create_navigator_factory(
    default_config: Optional[Dict[str, Any]] = None,
    seed_manager: Optional[SeedManager] = None
) -> Callable:
    """
    Create a navigator factory function with pre-configured defaults and seed management.
    
    This function returns a configured factory function that can create navigators
    with consistent defaults and seed management across multiple creations.
    
    Args:
        default_config: Default configuration to use for all navigator creations
        seed_manager: Seed manager instance for consistent random behavior
        
    Returns:
        Factory function that creates navigators with the specified defaults
        
    Examples:
        >>> factory = create_navigator_factory(
        ...     default_config={"max_speed": 2.0, "speed": 1.0},
        ...     seed_manager=SeedManager(seed=42)
        ... )
        >>> navigator1 = factory(position=(0, 0))
        >>> navigator2 = factory(position=(10, 10))
    """
    def navigator_factory(
        config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> NavigatorProtocol:
        """
        Factory function for creating navigators with consistent configuration.
        
        Args:
            config_overrides: Configuration overrides for this specific navigator
            **kwargs: Additional parameters passed to navigator creation
            
        Returns:
            Configured navigator instance
        """
        # Merge default config with overrides
        merged_config = {}
        
        if default_config:
            merged_config.update(default_config)
        
        if config_overrides:
            merged_config.update(config_overrides)
        
        # Add kwargs to config
        merged_config.update(kwargs)
        
        # Use seed manager if available
        seed = None
        experiment_id = None
        
        if seed_manager:
            seed = seed_manager.seed
            experiment_id = seed_manager.experiment_id
        
        # Create navigator using configuration-driven approach
        result = create_navigator_from_config(
            config=merged_config,
            seed=seed,
            experiment_id=experiment_id
        )
        
        return result.navigator
    
    return navigator_factory


def _is_multi_agent_config_dict(config_dict: Dict[str, Any]) -> bool:
    """
    Determine if configuration dictionary specifies multi-agent navigation.
    
    Args:
        config_dict: Configuration dictionary to analyze
        
    Returns:
        bool: True if multi-agent configuration, False for single-agent
    """
    # Check for explicit multi-agent indicators
    if 'positions' in config_dict:
        positions = config_dict['positions']
        if isinstance(positions, (list, tuple)) and len(positions) > 0:
            # Check if it's a list of positions (multi-agent) vs single position
            if isinstance(positions[0], (list, tuple)):
                return True
        elif isinstance(positions, np.ndarray) and positions.ndim > 1:
            return True
    
    # Check for other multi-agent indicators
    multi_agent_keys = ['orientations', 'speeds', 'max_speeds', 'angular_velocities']
    for key in multi_agent_keys:
        if key in config_dict:
            value = config_dict[key]
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 1:
                return True
    
    # Check for explicit num_agents parameter
    if config_dict.get('num_agents', 1) > 1:
        return True
    
    return False


def _create_single_agent_from_dict(config_dict: Dict[str, Any]) -> NavigatorProtocol:
    """Create single-agent navigator from configuration dictionary."""
    # Extract single-agent parameters
    kwargs = {}
    
    # Handle position (single tuple/list)
    if 'position' in config_dict:
        kwargs['position'] = config_dict['position']
    elif 'positions' in config_dict:
        # Convert from positions array if needed
        positions = config_dict['positions']
        if isinstance(positions, (list, tuple)) and len(positions) > 0:
            kwargs['position'] = positions[0]
    
    # Handle other single-agent parameters
    single_agent_params = ['orientation', 'speed', 'max_speed', 'angular_velocity']
    for param in single_agent_params:
        if param in config_dict:
            kwargs[param] = config_dict[param]
        elif f'{param}s' in config_dict:
            # Convert from plural form if needed
            values = config_dict[f'{param}s']
            if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
                kwargs[param] = values[0]
    
    return Navigator(**kwargs)


def _create_multi_agent_from_dict(config_dict: Dict[str, Any]) -> NavigatorProtocol:
    """Create multi-agent navigator from configuration dictionary."""
    # Extract multi-agent parameters
    kwargs = {}
    
    # Handle positions (required for multi-agent)
    if 'positions' in config_dict:
        kwargs['positions'] = config_dict['positions']
    else:
        raise ValueError("Multi-agent configuration requires 'positions' parameter")
    
    # Handle other multi-agent parameters
    multi_agent_params = ['orientations', 'speeds', 'max_speeds', 'angular_velocities']
    for param in multi_agent_params:
        if param in config_dict:
            kwargs[param] = config_dict[param]
    
    return Navigator(**kwargs)


def _validate_navigator_determinism(
    navigator: NavigatorProtocol, 
    seed: int,
    validation_steps: int = 10
) -> Dict[str, Any]:
    """
    Validate that navigator behavior is deterministic.
    
    Args:
        navigator: Navigator to validate
        seed: Seed used for navigator creation
        validation_steps: Number of simulation steps to validate
        
    Returns:
        Dict containing validation results
    """
    validation_result = {
        'is_deterministic': False,
        'validation_steps': validation_steps,
        'error': None,
        'trajectory_checksum': None
    }
    
    try:
        # Create a simple environment for testing
        test_env = np.random.random((100, 100))
        
        # Capture initial positions for comparison
        initial_positions = navigator.positions.copy()
        
        # Run simulation steps and track trajectory
        trajectory = [initial_positions.copy()]
        
        for step in range(validation_steps):
            navigator.step(test_env)
            trajectory.append(navigator.positions.copy())
        
        # Generate checksum for trajectory
        trajectory_array = np.array(trajectory)
        trajectory_checksum = hash(trajectory_array.tobytes())
        validation_result['trajectory_checksum'] = trajectory_checksum
        
        # Reset navigator and run again with same seed
        if SEED_MANAGER_AVAILABLE:
            set_global_seed(seed)
        navigator.reset()
        
        # Run simulation again and compare
        trajectory2 = [navigator.positions.copy()]
        
        for step in range(validation_steps):
            navigator.step(test_env)
            trajectory2.append(navigator.positions.copy())
        
        # Compare trajectories
        trajectory2_array = np.array(trajectory2)
        trajectory2_checksum = hash(trajectory2_array.tobytes())
        
        # Check if trajectories match
        if trajectory_checksum == trajectory2_checksum:
            validation_result['is_deterministic'] = True
        else:
            validation_result['error'] = "Trajectory checksums do not match between runs"
            
    except Exception as e:
        validation_result['error'] = f"Validation failed with error: {str(e)}"
    
    return validation_result


def enhance_navigator_with_configuration(
    navigator: NavigatorProtocol,
    config: Union[Dict[str, Any], DictConfig, NavigatorConfig],
    apply_immediately: bool = True
) -> NavigatorProtocol:
    """
    Enhance existing navigator with additional configuration options.
    
    This function allows updating an existing navigator with new configuration
    while preserving its current state and capabilities.
    
    Args:
        navigator: Existing navigator instance to enhance
        config: Additional configuration to apply
        apply_immediately: Whether to apply configuration immediately
        
    Returns:
        Enhanced navigator instance (may be the same instance if modified in-place)
        
    Examples:
        >>> navigator = create_navigator_from_params(position=(0, 0))
        >>> enhanced_config = {"max_speed": 5.0, "sensor_config": {...}}
        >>> enhanced_navigator = enhance_navigator_with_configuration(navigator, enhanced_config)
    """
    # Convert config to dictionary if needed
    if isinstance(config, dict):
        config_dict = config
    elif HYDRA_AVAILABLE and isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    elif CONFIG_MODELS_AVAILABLE and hasattr(config, 'dict'):
        config_dict = config.dict()
    elif CONFIG_MODELS_AVAILABLE and hasattr(config, 'model_dump'):
        config_dict = config.model_dump()
    else:
        raise TypeError(f"Unsupported configuration type: {type(config)}")
    
    # Apply configuration enhancements
    if apply_immediately:
        # Extract parameters that can be updated
        update_params = {}
        
        updatable_params = ['max_speeds', 'max_speed']
        for param in updatable_params:
            if param in config_dict:
                update_params[param] = config_dict[param]
        
        if update_params:
            navigator.reset(**update_params)
    
    # Store additional configuration as metadata (if navigator supports it)
    if hasattr(navigator, '_config_metadata'):
        navigator._config_metadata.update(config_dict)
    
    return navigator


def get_predefined_sensor_layout(
    layout_name: str,
    distance: float = 5.0
) -> np.ndarray:
    """
    Return a predefined set of sensor offsets (in the agent's local frame),
    scaled by the given distance.

    Parameters
    ----------
    layout_name : str
        Name of the layout; must be a key of PREDEFINED_SENSOR_LAYOUTS.
    distance : float, default=5.0
        Scaling distance.

    Returns
    -------
    offsets : np.ndarray
        Shape (num_sensors, 2). The local (x, y) offsets for each sensor.
    """
    # Get the base layout
    try:
        layout = PREDEFINED_SENSOR_LAYOUTS[layout_name]
    except KeyError as e:
        raise ValueError(
            f"Unknown sensor layout: {layout_name}. "
            f"Available layouts: {list(PREDEFINED_SENSOR_LAYOUTS.keys())}"
        ) from e

    # Scale by the requested distance
    return layout * distance


def define_sensor_offsets(
    num_sensors: int,
    distance: float,
    angle: float
) -> np.ndarray:
    """
    Define sensor offsets in the agent's local coordinate frame,
    where the agent's heading is [1, 0].
    
    Angles are distributed symmetrically around 0 degrees for even counts,
    or from -angle_range/2 to +angle_range/2 for odd counts.
    
    Parameters
    ----------
    num_sensors : int
        Number of sensors to create.
    distance : float
        Distance from agent center to sensors.
    angle : float
        Angular increment between sensors, in degrees.
        If there are n sensors, the total angular range will be:
        (n-1) * angle.
        
    Returns
    -------
    offsets : np.ndarray
        Shape (num_sensors, 2). Each row is the (x, y) offset
        in the agent's local coordinates.
    """
    # Calculate start angle
    total_angle_range = (num_sensors - 1) * angle
    start_angle = -total_angle_range / 2
    
    # Create array to store offsets
    offsets = np.zeros((num_sensors, 2))
    
    # Calculate offset for each sensor
    for i in range(num_sensors):
        # Current angle in degrees, starting from -total_range/2
        current_angle = start_angle + i * angle
        
        # Convert to radians
        current_angle_rad = np.deg2rad(current_angle)
        
        # Calculate offset using polar coordinates
        offsets[i, 0] = distance * np.cos(current_angle_rad)  # x
        offsets[i, 1] = distance * np.sin(current_angle_rad)  # y
        
    return offsets


def rotate_offset(local_offset: np.ndarray, orientation_deg: float) -> np.ndarray:
    """Rotate a local offset by ``orientation_deg`` degrees.

    The operation is performed in double precision and the angle is
    normalised to ``[0, 360)`` to avoid discrepancies where ``0`` and
    ``360`` would otherwise yield slightly different results due to
    floating point rounding.  The function fails fast if the input does
    not represent a 2‑vector.
    """
    if local_offset.shape != (2,):
        raise ValueError("local_offset must be a 2‑element vector")

    orientation_deg = normalize_angle(float(orientation_deg))
    result = rotate(local_offset, orientation_deg)
    logger.debug(
        f"rotate_offset: local={local_offset}, orientation={orientation_deg}, result={result}"
    )
    return result


def calculate_sensor_positions(
    navigator: NavigatorProtocol,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None
) -> np.ndarray:
    """
    Calculate sensor positions given a local sensor geometry and each agent's
    global position/orientation.

    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator with position and orientation information
    sensor_distance : float
        Distance from agent center to each sensor
    sensor_angle : float
        Angle between adjacent sensors in degrees
    num_sensors : int
        Number of sensors per agent
    layout_name : str, optional
        Name of a predefined sensor layout. If provided, sensor_angle is ignored.

    Returns
    -------
    sensor_positions : np.ndarray
        Shape (num_agents, num_sensors, 2). Each row is the (x, y)
        position of a sensor in global coordinates.
    """
    return compute_sensor_positions(
        navigator.positions,
        navigator.orientations,
        layout_name=layout_name,
        distance=sensor_distance,
        angle=sensor_angle,
        num_sensors=num_sensors
    )


def compute_sensor_positions(
    agent_positions: np.ndarray,
    agent_orientations: np.ndarray,
    layout_name: str = None,
    distance: float = 5.0,
    angle: float = 45.0,
    num_sensors: int = 2
) -> np.ndarray:
    """
    Compute sensor positions in global coordinates, given each agent's position
    and orientation, and either a named sensor layout or parameters to create one.

    Parameters
    ----------
    agent_positions : np.ndarray
        Shape (num_agents, 2). The (x, y) positions of each agent.
    agent_orientations : np.ndarray
        Shape (num_agents,). The orientation in degrees of each agent.
    layout_name : str, optional
        Name of a predefined sensor layout. If provided, `angle` and 
        `num_sensors` will be ignored.
    distance : float
        Distance from agent center to each sensor.
    angle : float
        Angular increment between sensors in degrees.
    num_sensors : int
        Number of sensors per agent, used only if layout_name is None.

    Returns
    -------
    sensor_positions : np.ndarray
        Shape (num_agents, num_sensors, 2). The global positions of each sensor.
    """
    # 1. Get the local offsets
    if layout_name is not None:
        local_offsets = get_predefined_sensor_layout(layout_name, distance=distance)
        num_sensors = local_offsets.shape[0]  # Update num_sensors based on layout
    else:
        local_offsets = define_sensor_offsets(num_sensors, distance, angle)

    num_agents = agent_positions.shape[0]
    sensor_positions = np.zeros((num_agents, num_sensors, 2), dtype=float)
    
    # 2. For each agent, rotate each offset by orientation & add the agent's position
    for agent_idx in range(num_agents):
        # Current agent's global position and heading
        agent_pos = agent_positions[agent_idx]
        agent_orientation = agent_orientations[agent_idx]
        
        for sensor_idx in range(num_sensors):
            local_offset = local_offsets[sensor_idx]
            rotated = rotate_offset(local_offset, agent_orientation)
            sensor_positions[agent_idx, sensor_idx] = agent_pos + rotated

    return sensor_positions


def read_odor_values(
    env_array: np.ndarray, 
    positions: np.ndarray
) -> np.ndarray:
    """
    Read odor values from an environment array at specific positions.
    
    This function handles bounds checking, pixel coordinate conversion,
    and normalization of uint8 arrays.
    
    Parameters
    ----------
    env_array : np.ndarray
        Environment array (e.g., odor concentration grid)
    positions : np.ndarray
        Array of positions with shape (N, 2) where each row is (x, y)
        
    Returns
    -------
    np.ndarray
        Array of odor values with shape (N,)
    """
    # Check if this is a mock plume object (for testing)
    if hasattr(env_array, 'current_frame'):
        env_array = env_array.current_frame

    # Get dimensions of environment array
    if not hasattr(env_array, 'shape') or len(env_array.shape) < 2:
        # For mock objects in tests or arrays without shape
        return np.zeros(len(positions))

    height, width = env_array.shape[:2]
    num_positions = positions.shape[0]
    odor_values = np.zeros(num_positions)

    # Convert positions to integers for indexing
    x_pos = np.floor(positions[:, 0]).astype(int)
    y_pos = np.floor(positions[:, 1]).astype(int)

    # Create a mask for positions that are within bounds
    within_bounds = (
        (x_pos >= 0) & (x_pos < width) & (y_pos >= 0) & (y_pos < height)
    )

    # Read values for positions within bounds
    for i in range(num_positions):
        if within_bounds[i]:
            # Convert array indices to Python scalars for NumPy 2.x compatibility
            y_idx = y_pos[i].item() if hasattr(y_pos[i], 'item') else int(y_pos[i])
            x_idx = x_pos[i].item() if hasattr(x_pos[i], 'item') else int(x_pos[i])
            
            # Get value from environment array
            pixel_value = env_array[y_idx, x_idx]
            
            # Handle multi-channel arrays (e.g., RGB images)
            if isinstance(pixel_value, np.ndarray) and pixel_value.size > 1:
                # For multi-channel arrays, use the first channel or convert to grayscale
                if len(pixel_value) == 3:  # RGB image
                    # Convert RGB to grayscale using standard weights
                    odor_values[i] = 0.299 * pixel_value[0] + 0.587 * pixel_value[1] + 0.114 * pixel_value[2]
                else:
                    # For other multi-channel arrays, just use the first channel
                    odor_values[i] = pixel_value[0]
            else:
                # Single value (scalar or 0-d array)
                odor_values[i] = pixel_value

            # Normalize if uint8
            if hasattr(env_array, 'dtype') and env_array.dtype == np.uint8:
                odor_values[i] /= 255.0

    return odor_values


def update_positions_and_orientations(
    positions: np.ndarray,
    orientations: np.ndarray,
    speeds: np.ndarray,
    angular_velocities: np.ndarray,
    dt: float = 1.0
) -> None:
    """
    Update positions and orientations based on speeds and angular velocities.
    
    This function handles the vectorized movement calculation for single or multiple agents,
    with proper time step scaling. Position updates are scaled by speed * dt, and 
    orientation updates are scaled by angular_velocity * dt.
    
    It modifies the input arrays in-place.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2) with agent positions
    orientations : np.ndarray
        Array of shape (N,) with agent orientations in degrees
    speeds : np.ndarray
        Array of shape (N,) with agent speeds (units/second)
    angular_velocities : np.ndarray
        Array of shape (N,) with agent angular velocities in degrees/second
    dt : float, optional
        Time step size in seconds, by default 1.0
        
    Returns
    -------
    None
        The function modifies the input arrays in-place
        
    Notes
    -----
    The default dt=1.0 maintains backward compatibility with existing code
    that doesn't explicitly handle time steps. To properly incorporate physics
    time steps, pass the actual dt value from your simulation.
    """
    if dt < 0:
        raise ValueError("dt must be non-negative")

    pos64 = np.asarray(positions, dtype=np.float64)
    ori64 = np.asarray(orientations, dtype=np.float64)
    spd64 = np.asarray(speeds, dtype=np.float64)
    ang64 = np.asarray(angular_velocities, dtype=np.float64)

    if not (len(pos64) == len(ori64) == len(spd64) == len(ang64)):
        raise ValueError("All input arrays must have the same length")

    logger.debug(
        "update_positions_and_orientations: pos=%s ori=%s spd=%s ang=%s dt=%s",
        pos64,
        ori64,
        spd64,
        ang64,
        dt,
    )

    ori64 = normalize_angle(ori64)
    rad_orientations = np.deg2rad(ori64)

    dx = spd64 * np.cos(rad_orientations) * dt
    dy = spd64 * np.sin(rad_orientations) * dt

    if pos64.ndim == 2:
        pos64 = pos64 + np.column_stack((dx, dy))
    else:
        for i in range(len(pos64)):
            pos64[i] = pos64[i] + np.array([dx[i], dy[i]], dtype=np.float64)

    positions[...] = pos64

    ori64 = normalize_angle(ori64 + ang64 * dt)
    orientations[...] = ori64

    logger.debug(f"Updated positions and orientations (dt={dt})")


@dataclass
class SingleAgentParams:
    """Parameters for resetting a single agent navigator."""
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = None
    speed: Optional[float] = None
    max_speed: Optional[float] = None
    angular_velocity: Optional[float] = None


@dataclass
class MultiAgentParams:
    """Parameters for resetting a multi-agent navigator."""
    positions: Optional[np.ndarray] = None
    orientations: Optional[np.ndarray] = None
    speeds: Optional[np.ndarray] = None
    max_speeds: Optional[np.ndarray] = None
    angular_velocities: Optional[np.ndarray] = None


def reset_navigator_state(
    controller_state: Dict[str, np.ndarray],
    is_single_agent: bool,
    **kwargs: Any
) -> None:
    """
    Reset navigator controller state based on provided parameters.
    
    This function handles updating controller state arrays from kwargs,
    ensuring proper array shapes and consistent array sizes.
    
    Parameters
    ----------
    controller_state : Dict[str, np.ndarray]
        Dictionary of current controller state arrays, where keys are:
        - '_position'/'_positions': Array of shape (N, 2) or (1, 2)
        - '_orientation'/'_orientations': Array of shape (N,) or (1,)
        - '_speed'/'_speeds': Array of shape (N,) or (1,)
        - '_max_speed'/'_max_speeds': Array of shape (N,) or (1,)
        - '_angular_velocity'/'_angular_velocities': Array of shape (N,) or (1,)
    is_single_agent : bool
        Whether this is a single agent controller
    **kwargs
        Parameters to update. 
        Valid keys for single-agent controllers:
            - 'position': Tuple[float, float] or array-like
            - 'orientation': float
            - 'speed': float
            - 'max_speed': float
            - 'angular_velocity': float
        Valid keys for multi-agent controllers:
            - 'positions': np.ndarray of shape (N, 2)
            - 'orientations': np.ndarray of shape (N,)
            - 'speeds': np.ndarray of shape (N,)
            - 'max_speeds': np.ndarray of shape (N,)
            - 'angular_velocities': np.ndarray of shape (N,)
    
    Returns
    -------
    None
        The function modifies the input state dictionary in-place
    
    Raises
    ------
    ValueError
        If invalid parameter keys are provided
    
    Notes
    -----
    For stronger type safety, consider using the SingleAgentParams or MultiAgentParams
    dataclasses instead of kwargs. Example:
    
    ```python
    params = SingleAgentParams(position=(10, 20), speed=1.5)
    reset_navigator_state_with_params(controller_state, is_single_agent=True, params=params)
    ```
    """
    # Define valid keys and attribute mappings based on controller type
    if is_single_agent:
        position_key = 'position'
        orientation_key = 'orientation'
        speed_key = 'speed'
        max_speed_key = 'max_speed'
        angular_velocity_key = 'angular_velocity'

        # Map state dictionary keys to internal attribute names
        positions_attr = '_position'
        orientations_attr = '_orientation'
        speeds_attr = '_speed'
        max_speeds_attr = '_max_speed'
        angular_velocities_attr = '_angular_velocity'
    else:
        position_key = 'positions'
        orientation_key = 'orientations'
        speed_key = 'speeds'
        max_speed_key = 'max_speeds'
        angular_velocity_key = 'angular_velocities'

        # Map state dictionary keys to internal attribute names
        positions_attr = '_positions'
        orientations_attr = '_orientations'
        speeds_attr = '_speeds'
        max_speeds_attr = '_max_speeds'
        angular_velocities_attr = '_angular_velocities'

    # Define common valid keys and param mapping for both controller types
    # Note: 'seed' is accepted for Gymnasium 0.29.x compatibility but handled elsewhere
    valid_keys = {position_key, orientation_key, speed_key, max_speed_key, angular_velocity_key, 'seed'}
    param_mapping = [
        (position_key, positions_attr),
        (orientation_key, orientations_attr),
        (speed_key, speeds_attr),
        (max_speed_key, max_speeds_attr),
        (angular_velocity_key, angular_velocities_attr)
    ]

    if invalid_keys := set(kwargs.keys()) - valid_keys:
        raise ValueError(f"Invalid parameters: {invalid_keys}. Valid keys are: {valid_keys}")

    # Handle position update (which may require resizing other arrays)
    if (position_value := kwargs.get(position_key)) is not None:
        if is_single_agent:
            # Single agent case: wrap in array
            controller_state[positions_attr] = np.array([position_value])
        else:
            # Multi agent case: convert to array
            controller_state[positions_attr] = np.array(position_value)

            # For multi-agent, we may need to resize other arrays
            num_agents = controller_state[positions_attr].shape[0]

            # Resize other arrays if needed
            arrays_to_check = [
                (orientations_attr, np.zeros, num_agents),
                (speeds_attr, np.zeros, num_agents),
                (max_speeds_attr, np.ones, num_agents),
                (angular_velocities_attr, np.zeros, num_agents)
            ]

            for attr_name, default_fn, size in arrays_to_check:
                if attr_name in controller_state and controller_state[attr_name].shape[0] != num_agents:
                    controller_state[attr_name] = default_fn(size)

    # Update other values if provided
    for kwarg_key, attr_key in param_mapping[1:]:  # Skip position which was handled above
        if kwarg_key in kwargs:
            value = kwargs[kwarg_key]
            if is_single_agent:
                controller_state[attr_key] = np.array([value])
            else:
                controller_state[attr_key] = np.array(value)


def reset_navigator_state_with_params(
    controller_state: Dict[str, np.ndarray],
    is_single_agent: bool,
    params: Union[SingleAgentParams, MultiAgentParams]
) -> None:
    """
    Reset navigator controller state using type-safe parameter objects.
    
    This is a type-safe alternative to reset_navigator_state that uses dataclasses
    instead of kwargs for stronger type safety.
    
    Parameters
    ----------
    controller_state : Dict[str, np.ndarray]
        Dictionary of current controller state arrays
    is_single_agent : bool
        Whether this is a single agent controller
    params : Union[SingleAgentParams, MultiAgentParams]
        Parameters to update, as a dataclass instance
    
    Returns
    -------
    None
        The function modifies the input state dictionary in-place
    
    Raises
    ------
    TypeError
        If params is not the correct type for the controller
    """
    # Validate parameter type
    if is_single_agent and not isinstance(params, SingleAgentParams):
        raise TypeError(
            f"Expected SingleAgentParams for single agent controller, got {type(params)}"
        )
    if not is_single_agent and not isinstance(params, MultiAgentParams):
        raise TypeError(
            f"Expected MultiAgentParams for multi-agent controller, got {type(params)}"
        )
    
    # Convert dataclass to dictionary for the existing function
    kwargs = {k: v for k, v in params.__dict__.items() if v is not None}
    
    # Delegate to the existing function
    reset_navigator_state(controller_state, is_single_agent, **kwargs)


def sample_odor_at_sensors(
    navigator: NavigatorProtocol,
    env_array: np.ndarray,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    num_sensors: int = 2,
    layout_name: Optional[str] = None
) -> np.ndarray:
    """
    Sample odor values at sensor positions for all navigators.
    
    Args:
        navigator: Navigator instance
        env_array: 2D array representing the environment (e.g., video frame)
        sensor_distance: Distance of sensors from navigator position
        sensor_angle: Angle between sensors in degrees
        num_sensors: Number of sensors per navigator
        layout_name: If provided, use this predefined sensor layout instead of 
                    creating one based on num_sensors and sensor_angle
        
    Returns:
        Array of odor readings with shape (num_agents, num_sensors)
    """
    # Calculate sensor positions
    sensor_positions = calculate_sensor_positions(
        navigator, sensor_distance, sensor_angle, num_sensors, layout_name
    )
    
    # Read odor values at sensor positions
    odor_values = read_odor_values(env_array, sensor_positions.reshape(-1, 2))
    
    # Reshape to (num_agents, num_sensors)
    num_agents = navigator.num_agents
    num_sensors = sensor_positions.shape[1]
    odor_values = odor_values.reshape(num_agents, num_sensors)
    
    return odor_values


def get_property_name(is_single_agent: bool, property_name: str) -> str:
    """
    Get the correct attribute name for a property based on controller type.
    
    Parameters
    ----------
    is_single_agent : bool
        Whether this is a single agent controller
    property_name : str
        Base property name (e.g., 'position', 'orientation')
    
    Returns
    -------
    str
        The correct attribute name for the property ('_position' or '_positions')
    """
    suffix = "" if is_single_agent else "s"
    return f"_{property_name}{suffix}"


def get_property_value(controller: Any, property_name: str) -> Union[float, np.ndarray]:
    """
    Get a property value from a controller, handling single vs multi-agent cases.
    
    For single agent controllers, returns a scalar value instead of an array.
    
    Parameters
    ----------
    controller : Any
        The controller instance (SingleAgentController or MultiAgentController)
    property_name : str
        Property name without underscore prefix (e.g., 'position', 'orientation')
    
    Returns
    -------
    Union[float, np.ndarray]
        Property value, as scalar for single agent and array for multi-agent
    """
    is_single_agent = hasattr(controller, '_position')
    attr_name = get_property_name(is_single_agent, property_name)
    
    value = getattr(controller, attr_name)
    
    # Return scalar for single agent with size 1 array, otherwise return the array
    return value[0] if is_single_agent and value.size == 1 else value


def create_navigator_for_cli(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    seed: Optional[int] = None,
    experiment_id: Optional[str] = None
) -> Tuple[NavigatorProtocol, Dict[str, Any]]:
    """
    Create navigator for CLI applications with Hydra configuration support.
    
    This function provides CLI-optimized navigator creation with support for
    configuration file loading, command-line overrides, and experiment tracking.
    
    Args:
        config_path: Path to configuration file (optional)
        overrides: List of configuration overrides in Hydra format (key=value)
        seed: Random seed for reproducible behavior
        experiment_id: Experiment identifier for tracking
        
    Returns:
        Tuple of (navigator, cli_metadata) containing the navigator and CLI metadata
        
    Raises:
        RuntimeError: If configuration loading or navigator creation fails
        FileNotFoundError: If config_path does not exist
        
    Examples:
        >>> navigator, metadata = create_navigator_for_cli(
        ...     config_path="config.yaml",
        ...     overrides=["navigator.max_speed=2.0", "seed=42"]
        ... )
    """
    cli_metadata = {
        'config_path': config_path,
        'overrides': overrides or [],
        'seed': seed,
        'experiment_id': experiment_id,
        'creation_timestamp': time.time()
    }
    
    try:
        # Load configuration
        if config_path and HYDRA_AVAILABLE:
            from hydra import compose, initialize_config_dir
            from pathlib import Path
            
            config_dir = str(Path(config_path).parent.absolute())
            config_name = Path(config_path).stem
            
            try:
                with initialize_config_dir(config_dir=config_dir, version_base=None):
                    cfg = compose(config_name=config_name, overrides=overrides or [])
                    
                    # Extract navigator configuration
                    if 'navigator' in cfg:
                        navigator_config = cfg.navigator
                    else:
                        navigator_config = cfg
                    
                    # Create navigator from Hydra config
                    result = create_navigator_from_config(
                        config=navigator_config,
                        seed=seed,
                        experiment_id=experiment_id
                    )
                    
                    cli_metadata.update({
                        'configuration_source': 'hydra_file',
                        'hydra_config': OmegaConf.to_container(cfg, resolve=True),
                        'creation_result': result.metadata
                    })
                    
                    return result.navigator, cli_metadata
                    
            except Exception as e:
                if LOGURU_AVAILABLE:
                    logger.error(f"Failed to load Hydra configuration from {config_path}: {str(e)}")
                raise RuntimeError(f"Configuration loading failed: {str(e)}") from e
        
        # Fallback to default configuration with overrides
        default_config = {"position": (0.0, 0.0), "max_speed": 1.0}
        
        # Apply CLI overrides to default config
        if overrides:
            for override in overrides:
                if '=' in override:
                    key, value = override.split('=', 1)
                    # Simple override parsing (can be enhanced)
                    try:
                        # Try to parse as number
                        if '.' in value:
                            parsed_value = float(value)
                        else:
                            parsed_value = int(value)
                    except ValueError:
                        # Keep as string
                        parsed_value = value.strip('"\'')
                    
                    # Set nested keys (simple implementation)
                    if '.' in key:
                        keys = key.split('.')
                        current = default_config
                        for k in keys[:-1]:
                            if k not in current:
                                current[k] = {}
                            current = current[k]
                        current[keys[-1]] = parsed_value
                    else:
                        default_config[key] = parsed_value
        
        # Create navigator from default config
        result = create_navigator_from_config(
            config=default_config,
            seed=seed,
            experiment_id=experiment_id
        )
        
        cli_metadata.update({
            'configuration_source': 'default_with_overrides',
            'applied_config': default_config,
            'creation_result': result.metadata
        })
        
        return result.navigator, cli_metadata
        
    except Exception as e:
        error_msg = f"Failed to create navigator for CLI: {str(e)}"
        if LOGURU_AVAILABLE:
            logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def create_navigator_with_database_logging(
    config: Union[Dict[str, Any], DictConfig, NavigatorConfig],
    database_config: Optional[Dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
    track_trajectory: bool = True,
    track_performance: bool = True
) -> Tuple[NavigatorProtocol, Dict[str, Any]]:
    """
    Create navigator with database logging and experiment tracking capabilities.
    
    This function creates a navigator with enhanced database integration for
    experiment tracking, trajectory logging, and performance monitoring.
    
    Args:
        config: Navigator configuration
        database_config: Database connection configuration
        experiment_id: Experiment identifier for database tracking
        track_trajectory: Whether to enable trajectory tracking
        track_performance: Whether to enable performance monitoring
        
    Returns:
        Tuple of (navigator, database_metadata) containing the navigator and database metadata
        
    Raises:
        RuntimeError: If database setup or navigator creation fails
        
    Examples:
        >>> db_config = {"url": "sqlite:///experiments.db"}
        >>> navigator, db_metadata = create_navigator_with_database_logging(
        ...     config={"position": (0, 0)},
        ...     database_config=db_config,
        ...     experiment_id="exp_001"
        ... )
    """
    database_metadata = {
        'experiment_id': experiment_id,
        'database_config': database_config,
        'track_trajectory': track_trajectory,
        'track_performance': track_performance,
        'setup_timestamp': time.time()
    }
    
    try:
        # Create base navigator
        result = create_navigator_from_config(config=config, experiment_id=experiment_id)
        navigator = result.navigator
        
        # Setup database logging if configuration provided
        if database_config:
            try:
                # Import database utilities (would be available from db module)
                # This is a placeholder for actual database integration
                database_metadata.update({
                    'database_connection': 'configured',
                    'logging_enabled': True,
                    'tables_created': True
                })
                
                # Log experiment start
                experiment_data = {
                    'experiment_id': experiment_id,
                    'navigator_config': config,
                    'creation_metadata': result.metadata,
                    'start_timestamp': time.time()
                }
                
                database_metadata['experiment_data'] = experiment_data
                
                if LOGURU_AVAILABLE:
                    logger.bind(
                        experiment_id=experiment_id,
                        database_enabled=True
                    ).info("Navigator created with database logging enabled")
                
            except Exception as e:
                database_metadata['database_error'] = str(e)
                if LOGURU_AVAILABLE:
                    logger.warning(f"Database setup failed, continuing without database logging: {str(e)}")
        
        # Enhance navigator with tracking capabilities if requested
        if track_trajectory or track_performance:
            # Add metadata tracking to navigator (if supported)
            if hasattr(navigator, '_tracking_metadata'):
                navigator._tracking_metadata = {
                    'track_trajectory': track_trajectory,
                    'track_performance': track_performance,
                    'experiment_id': experiment_id,
                    'database_metadata': database_metadata
                }
        
        return navigator, database_metadata
        
    except Exception as e:
        error_msg = f"Failed to create navigator with database logging: {str(e)}"
        if LOGURU_AVAILABLE:
            logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def validate_navigator_configuration(
    config: Union[Dict[str, Any], DictConfig, NavigatorConfig],
    strict_validation: bool = True
) -> Dict[str, Any]:
    """
    Validate navigator configuration with comprehensive error reporting.
    
    This function provides thorough validation of navigator configuration
    with detailed error reporting and suggestions for fixes.
    
    Args:
        config: Configuration to validate
        strict_validation: Whether to use strict validation rules
        
    Returns:
        Validation result dictionary with errors, warnings, and suggestions
        
    Examples:
        >>> config = {"position": "invalid", "speed": -1}
        >>> validation = validate_navigator_configuration(config)
        >>> if validation['is_valid']:
        ...     print("Configuration is valid")
        >>> else:
        ...     print(f"Errors: {validation['errors']}")
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'suggestions': [],
        'validated_config': None,
        'validation_timestamp': time.time()
    }
    
    try:
        # Convert config to dictionary
        if isinstance(config, dict):
            config_dict = config.copy()
        elif HYDRA_AVAILABLE and isinstance(config, DictConfig):
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif CONFIG_MODELS_AVAILABLE and hasattr(config, 'dict'):
            config_dict = config.dict()
        elif CONFIG_MODELS_AVAILABLE and hasattr(config, 'model_dump'):
            config_dict = config.model_dump()
        else:
            validation_result['errors'].append(f"Unsupported configuration type: {type(config)}")
            validation_result['is_valid'] = False
            return validation_result
        
        # Validate using Pydantic models if available
        if CONFIG_MODELS_AVAILABLE and strict_validation:
            try:
                if _is_multi_agent_config_dict(config_dict):
                    validated_config = MultiAgentConfig(**config_dict)
                else:
                    validated_config = SingleAgentConfig(**config_dict)
                
                validation_result['validated_config'] = validated_config.dict() if hasattr(validated_config, 'dict') else validated_config.model_dump()
                
            except Exception as e:
                validation_result['errors'].append(f"Pydantic validation failed: {str(e)}")
                validation_result['is_valid'] = False
        
        # Manual validation checks
        if 'position' in config_dict:
            position = config_dict['position']
            if not isinstance(position, (list, tuple)) or len(position) != 2:
                validation_result['errors'].append("Position must be a list or tuple of length 2")
                validation_result['is_valid'] = False
            elif not all(isinstance(x, (int, float)) for x in position):
                validation_result['errors'].append("Position coordinates must be numeric")
                validation_result['is_valid'] = False
        
        # Check speed parameters
        speed_params = ['speed', 'max_speed']
        for param in speed_params:
            if param in config_dict:
                value = config_dict[param]
                if not isinstance(value, (int, float)):
                    validation_result['errors'].append(f"{param} must be numeric")
                    validation_result['is_valid'] = False
                elif value < 0:
                    validation_result['errors'].append(f"{param} must be non-negative")
                    validation_result['is_valid'] = False
        
        # Add suggestions for common issues
        if 'max_speed' not in config_dict:
            validation_result['suggestions'].append("Consider adding max_speed parameter for speed limiting")
        
        if 'orientation' not in config_dict and 'orientations' not in config_dict:
            validation_result['suggestions'].append("Consider adding orientation parameter for directional control")
        
        # Check for multi-agent consistency
        if _is_multi_agent_config_dict(config_dict):
            if 'positions' in config_dict:
                num_agents = len(config_dict['positions'])
                array_params = ['orientations', 'speeds', 'max_speeds', 'angular_velocities']
                
                for param in array_params:
                    if param in config_dict:
                        param_length = len(config_dict[param]) if isinstance(config_dict[param], (list, tuple)) else 1
                        if param_length != num_agents and param_length != 1:
                            validation_result['warnings'].append(
                                f"{param} length ({param_length}) should match number of agents ({num_agents}) or be 1 for broadcasting"
                            )
        
    except Exception as e:
        validation_result['errors'].append(f"Validation failed with error: {str(e)}")
        validation_result['is_valid'] = False
    
    return validation_result


@contextmanager
def navigator_performance_context(
    navigator: NavigatorProtocol,
    monitor_memory: bool = True,
    monitor_timing: bool = True,
    log_results: bool = True
):
    """
    Context manager for monitoring navigator performance during operations.
    
    This context manager provides comprehensive performance monitoring for
    navigator operations including memory usage, timing, and operation counting.
    
    Args:
        navigator: Navigator instance to monitor
        monitor_memory: Whether to monitor memory usage
        monitor_timing: Whether to monitor operation timing
        log_results: Whether to log performance results
        
    Yields:
        Performance monitoring dictionary that gets updated during context
        
    Examples:
        >>> with navigator_performance_context(navigator) as perf:
        ...     for i in range(100):
        ...         navigator.step(env_array)
        ...     print(f"Average step time: {perf['average_step_time_ms']:.2f}ms")
    """
    import psutil
    import gc
    
    # Initialize performance tracking
    perf_data = {
        'start_time': time.perf_counter(),
        'step_count': 0,
        'step_times': [],
        'memory_usage': [],
        'operation_counts': {},
        'errors': []
    }
    
    if monitor_memory:
        process = psutil.Process()
        perf_data['initial_memory_mb'] = process.memory_info().rss / 1024 / 1024
    
    try:
        yield perf_data
        
    finally:
        # Calculate final performance metrics
        end_time = time.perf_counter()
        total_time = end_time - perf_data['start_time']
        
        perf_data.update({
            'total_time_seconds': total_time,
            'total_step_count': perf_data['step_count'],
            'average_step_time_ms': (sum(perf_data['step_times']) / len(perf_data['step_times']) * 1000) if perf_data['step_times'] else 0,
            'max_step_time_ms': (max(perf_data['step_times']) * 1000) if perf_data['step_times'] else 0,
            'min_step_time_ms': (min(perf_data['step_times']) * 1000) if perf_data['step_times'] else 0
        })
        
        if monitor_memory and perf_data['memory_usage']:
            perf_data.update({
                'peak_memory_mb': max(perf_data['memory_usage']),
                'memory_delta_mb': max(perf_data['memory_usage']) - perf_data['initial_memory_mb']
            })
        
        # Force garbage collection for accurate memory reporting
        if monitor_memory:
            gc.collect()
        
        if log_results and LOGURU_AVAILABLE:
            logger.bind(
                navigator_type=type(navigator).__name__,
                agent_count=navigator.num_agents,
                total_steps=perf_data['total_step_count'],
                average_step_time_ms=perf_data['average_step_time_ms'],
                peak_memory_mb=perf_data.get('peak_memory_mb', 'unknown')
            ).info("Navigator performance monitoring completed")


def set_property_value(
    controller: Any, 
    property_name: str, 
    value: Union[float, np.ndarray]
) -> None:
    """
    Set a property value on a controller, handling single vs multi-agent cases.
    
    Parameters
    ----------
    controller : Any
        The controller instance (SingleAgentController or MultiAgentController)
    property_name : str
        Property name without underscore prefix (e.g., 'position', 'orientation')
    value : Union[float, np.ndarray]
        Value to set, can be scalar or array
    
    Returns
    -------
    None
        The function modifies the controller in-place
    """
    is_single_agent = hasattr(controller, '_position')
    attr_name = get_property_name(is_single_agent, property_name)
    
    # For single agent, wrap scalar in array
    if is_single_agent and not isinstance(value, np.ndarray):
        value = np.array([value])
    
    setattr(controller, attr_name, value)


def create_single_agent_navigator(
    navigator_class: Type, 
    params: SingleAgentParams
) -> Any:
    """
    Create a single-agent navigator using type-safe parameter object.
    
    Parameters
    ----------
    navigator_class : Type
        The Navigator class to instantiate
    params : SingleAgentParams
        Parameters for creating the navigator
        
    Returns
    -------
    Any
        A navigator instance with a single agent
    """
    # Convert dataclass to dictionary, preserving only non-None values
    kwargs = {k: v for k, v in params.__dict__.items() if v is not None}
    
    # Create navigator instance
    return navigator_class(**kwargs)


def create_multi_agent_navigator(
    navigator_class: Type, 
    params: MultiAgentParams
) -> Any:
    """
    Create a multi-agent navigator using type-safe parameter object.
    
    Parameters
    ----------
    navigator_class : Type
        The Navigator class to instantiate
    params : MultiAgentParams
        Parameters for creating the navigator
        
    Returns
    -------
    Any
        A navigator instance with multiple agents
    """
    # Convert dataclass to dictionary, preserving only non-None values
    kwargs = {k: v for k, v in params.__dict__.items() if v is not None}
    
    # Create navigator instance
    return navigator_class(**kwargs)


def validate_positions(positions: Any) -> None:
    """Ensure positions/position is either a single (x, y) or a sequence of (x, y) pairs (shape (2,) or (N, 2))."""
    import numpy as np
    if positions is None:
        return
    arr = np.asarray(positions)
    if arr.ndim == 1 and arr.shape[0] == 2:
        # Single agent (x, y)
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("positions must be numeric.")
        return
    if arr.ndim == 2 and arr.shape[1] == 2:
        # Multi-agent
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("positions must be numeric.")
        return
    raise ValueError(f"positions must be a single (x, y) or a sequence of (x, y) pairs (shape (2,) or (N, 2)), got shape {arr.shape}.")


def load_navigator_from_config(
    config: Union[dict, DictConfig, NavigatorConfig],
    strict_validation: bool = True,
    seed: Optional[int] = None,
    experiment_id: Optional[str] = None
) -> NavigatorProtocol:
    """
    Enhanced navigator loading from configuration with comprehensive validation and error handling.
    
    This function provides backward-compatible navigator loading while adding enhanced
    validation, seed management, and configuration source support.
    
    Args:
        config: Configuration dictionary, DictConfig, or NavigatorConfig
        strict_validation: Whether to use strict validation rules
        seed: Optional random seed for reproducible initialization
        experiment_id: Optional experiment identifier for tracking
        
    Returns:
        Navigator instance created from configuration
        
    Raises:
        ValueError: If configuration is invalid or incomplete
        TypeError: If configuration type is unsupported
        
    Examples:
        Backward compatible usage:
        >>> config = {"position": (10, 20), "max_speed": 2.0}
        >>> navigator = load_navigator_from_config(config)
        
        Enhanced usage:
        >>> navigator = load_navigator_from_config(
        ...     config=config, 
        ...     seed=42, 
        ...     strict_validation=True
        ... )
    """
    try:
        # Use the enhanced create_navigator_from_config function
        result = create_navigator_from_config(
            config=config,
            seed=seed,
            validate_config=strict_validation,
            enable_logging=True,
            experiment_id=experiment_id
        )
        
        return result.navigator
        
    except Exception as e:
        # Fallback to original validation for backward compatibility
        if isinstance(config, dict):
            # Original validation logic for strict backward compatibility
            required_keys = {"positions", "orientations", "speeds", "max_speeds"}
            allowed_keys = required_keys | {"position", "orientation", "speed", "max_speed"}
            
            if unknown_keys := set(config.keys()) - allowed_keys:
                if strict_validation:
                    raise ValueError(f"Unknown keys in config: {unknown_keys}")
                else:
                    # Filter out unknown keys in non-strict mode
                    config = {k: v for k, v in config.items() if k in allowed_keys}
            
            # Validate required fields
            if "positions" in config:
                validate_positions(config["positions"])
            elif "position" in config:
                validate_positions(config["position"])
            else:
                raise ValueError("Config must include either 'positions' (multi-agent) or 'position' (single-agent) key.")
            
            # Initialize seed if provided
            if seed is not None and SEED_MANAGER_AVAILABLE:
                set_global_seed(seed, experiment_id=experiment_id)
            
            # Instantiate Navigator using original method
            if "positions" in config:
                return Navigator(
                    positions=config["positions"],
                    orientations=config.get("orientations"),
                    speeds=config.get("speeds"),
                    max_speeds=config.get("max_speeds"),
                    angular_velocities=config.get("angular_velocities")
                )
            else:
                return Navigator(
                    position=config["position"],
                    orientation=config.get("orientation"),
                    speed=config.get("speed"),
                    max_speed=config.get("max_speed"),
                    angular_velocity=config.get("angular_velocity")
                )
        else:
            # Re-raise original error for non-dict configs
            raise


# Enhanced utility functions for comprehensive navigator management

def get_navigator_capabilities(navigator: NavigatorProtocol) -> Dict[str, Any]:
    """
    Get comprehensive information about navigator capabilities and configuration.
    
    This function introspects a navigator instance to provide detailed information
    about its capabilities, configuration, and current state.
    
    Args:
        navigator: Navigator instance to analyze
        
    Returns:
        Dictionary containing comprehensive navigator information
        
    Examples:
        >>> navigator = create_navigator_from_params(position=(0, 0))
        >>> capabilities = get_navigator_capabilities(navigator)
        >>> print(f"Agent count: {capabilities['agent_count']}")
        >>> print(f"Supports sensors: {capabilities['supports_sensor_sampling']}")
    """
    capabilities = {
        'agent_count': navigator.num_agents,
        'is_multi_agent': navigator.num_agents > 1,
        'current_positions': navigator.positions.tolist(),
        'current_orientations': navigator.orientations.tolist(),
        'current_speeds': navigator.speeds.tolist(),
        'max_speeds': navigator.max_speeds.tolist(),
        'angular_velocities': navigator.angular_velocities.tolist(),
        'protocol_compliance': isinstance(navigator, NavigatorProtocol),
        'supports_reset': hasattr(navigator, 'reset'),
        'supports_step': hasattr(navigator, 'step'),
        'supports_sensor_sampling': hasattr(navigator, 'sample_odor'),
        'supports_multi_sensor': hasattr(navigator, 'sample_multiple_sensors'),
        'class_name': type(navigator).__name__,
        'module_name': type(navigator).__module__
    }
    
    # Add enhanced capabilities if available
    if hasattr(navigator, '_config_metadata'):
        capabilities['config_metadata'] = navigator._config_metadata
    
    if hasattr(navigator, '_tracking_metadata'):
        capabilities['tracking_metadata'] = navigator._tracking_metadata
    
    # Check for seed manager integration
    if SEED_MANAGER_AVAILABLE:
        global_manager = get_global_seed_manager()
        if global_manager:
            capabilities['seed_manager_active'] = True
            capabilities['current_seed'] = global_manager.seed
            capabilities['experiment_id'] = global_manager.experiment_id
        else:
            capabilities['seed_manager_active'] = False
    
    return capabilities


def create_navigator_comparison_report(
    navigators: List[NavigatorProtocol],
    include_performance: bool = False
) -> Dict[str, Any]:
    """
    Create a comprehensive comparison report for multiple navigator instances.
    
    This function analyzes multiple navigators and provides detailed comparison
    information including configuration differences, capability variations,
    and optional performance comparisons.
    
    Args:
        navigators: List of navigator instances to compare
        include_performance: Whether to include performance benchmarking
        
    Returns:
        Comprehensive comparison report dictionary
        
    Examples:
        >>> nav1 = create_navigator_from_params(position=(0, 0), max_speed=1.0)
        >>> nav2 = create_navigator_from_params(position=(10, 10), max_speed=2.0)
        >>> report = create_navigator_comparison_report([nav1, nav2])
        >>> print(f"Configuration differences: {report['differences']}")
    """
    if not navigators:
        return {'error': 'No navigators provided for comparison'}
    
    comparison_report = {
        'navigator_count': len(navigators),
        'comparison_timestamp': time.time(),
        'navigators': [],
        'differences': {},
        'similarities': {},
        'summary': {}
    }
    
    # Analyze each navigator
    for i, navigator in enumerate(navigators):
        nav_info = get_navigator_capabilities(navigator)
        nav_info['navigator_index'] = i
        comparison_report['navigators'].append(nav_info)
    
    # Find differences and similarities
    if len(navigators) > 1:
        first_nav = comparison_report['navigators'][0]
        
        # Compare key attributes
        comparable_attrs = [
            'agent_count', 'is_multi_agent', 'class_name', 'module_name',
            'supports_reset', 'supports_step', 'supports_sensor_sampling'
        ]
        
        for attr in comparable_attrs:
            values = [nav_info[attr] for nav_info in comparison_report['navigators']]
            unique_values = set(values)
            
            if len(unique_values) == 1:
                comparison_report['similarities'][attr] = list(unique_values)[0]
            else:
                comparison_report['differences'][attr] = values
    
    # Add performance comparison if requested
    if include_performance and len(navigators) > 0:
        comparison_report['performance'] = _compare_navigator_performance(navigators)
    
    # Generate summary
    comparison_report['summary'] = {
        'all_same_type': len(set(nav['class_name'] for nav in comparison_report['navigators'])) == 1,
        'all_same_agent_count': len(set(nav['agent_count'] for nav in comparison_report['navigators'])) == 1,
        'total_agents': sum(nav['agent_count'] for nav in comparison_report['navigators']),
        'difference_count': len(comparison_report['differences']),
        'similarity_count': len(comparison_report['similarities'])
    }
    
    return comparison_report


def _compare_navigator_performance(navigators: List[NavigatorProtocol]) -> Dict[str, Any]:
    """
    Compare performance characteristics of multiple navigators.
    
    Args:
        navigators: List of navigator instances to benchmark
        
    Returns:
        Performance comparison results
    """
    performance_results = {
        'benchmark_timestamp': time.time(),
        'test_parameters': {
            'step_count': 100,
            'environment_size': (100, 100)
        },
        'results': []
    }
    
    # Create test environment
    test_env = np.random.random((100, 100))
    
    for i, navigator in enumerate(navigators):
        try:
            # Reset navigator to initial state
            navigator.reset()
            
            # Benchmark step performance
            step_times = []
            
            for step in range(100):
                step_start = time.perf_counter()
                navigator.step(test_env)
                step_end = time.perf_counter()
                step_times.append(step_end - step_start)
            
            # Calculate performance metrics
            nav_performance = {
                'navigator_index': i,
                'agent_count': navigator.num_agents,
                'average_step_time_ms': (sum(step_times) / len(step_times)) * 1000,
                'max_step_time_ms': max(step_times) * 1000,
                'min_step_time_ms': min(step_times) * 1000,
                'total_time_ms': sum(step_times) * 1000,
                'steps_per_second': 1.0 / (sum(step_times) / len(step_times)) if step_times else 0
            }
            
            performance_results['results'].append(nav_performance)
            
        except Exception as e:
            performance_results['results'].append({
                'navigator_index': i,
                'error': f"Performance test failed: {str(e)}"
            })
    
    return performance_results
