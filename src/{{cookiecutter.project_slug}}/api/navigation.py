"""
Public API module for odor plume navigation with Hydra configuration support.

This module provides a consolidated, production-ready API for creating navigation controllers,
video plume environments, and executing simulations with comprehensive Hydra-based 
configuration management, error handling, and validation.

The API is designed to support:
- Kedro pipeline integration through factory method patterns
- Reinforcement learning frameworks via NumPy array interfaces  
- Machine learning analysis tools through standardized tensor interfaces
- Command-line interface integration with Hydra parameter overrides
- Reproducible research through seed management integration

Example Usage:
    # Basic usage with direct parameters
    navigator = create_navigator(position=(10.0, 20.0), max_speed=5.0)
    video_plume = create_video_plume("path/to/video.mp4", flip=True)
    results = run_plume_simulation(navigator, video_plume, num_steps=1000)
    
    # Using Hydra configuration objects (Kedro integration)
    from hydra import compose, initialize
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
        navigator = create_navigator(cfg=cfg.navigator)
        video_plume = create_video_plume(cfg=cfg.video_plume)
        results = run_plume_simulation(navigator, video_plume, cfg=cfg.simulation)
        
    # Command-line parameter overrides
    navigator = create_navigator(cfg=cfg.navigator, max_speed=12.0)  # Override config
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import pathlib
import warnings
import numpy as np
from contextlib import suppress
from dataclasses import asdict

try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    # Fallback for environments without Hydra
    DictConfig = dict
    OmegaConf = None
    HYDRA_AVAILABLE = False
    warnings.warn(
        "Hydra/OmegaConf not available. Some advanced configuration features may be limited.",
        ImportWarning
    )

# Import core navigation components
from {{cookiecutter.project_slug}}.core.navigator import Navigator, NavigatorProtocol
from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig, 
    VideoPlumeConfig, 
    SimulationConfig,
    SingleAgentConfig,
    MultiAgentConfig
)
from {{cookiecutter.project_slug}}.utils.seed_manager import set_global_seed, get_current_seed


class ConfigurationError(Exception):
    """Raised when configuration validation fails or required parameters are missing."""
    pass


class SimulationError(Exception):
    """Raised when simulation execution encounters an error."""
    pass


def _validate_and_merge_config(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    config_schema: type = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Validate and merge Hydra configuration with direct parameters.
    
    This function provides robust configuration handling supporting both Hydra
    DictConfig objects and traditional dictionaries, with comprehensive validation
    and parameter override capabilities.
    
    Args:
        cfg: Hydra DictConfig object or dictionary containing configuration parameters
        config_schema: Pydantic model class for validation (optional)
        **kwargs: Direct parameter overrides that take precedence over config
        
    Returns:
        Dict[str, Any]: Validated and merged configuration parameters
        
    Raises:
        ConfigurationError: If validation fails or required parameters are missing
    """
    # Initialize merged configuration
    merged_config = {}
    
    # Process Hydra configuration if provided
    if cfg is not None:
        try:
            if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
                # Convert DictConfig to dictionary, resolving interpolations
                merged_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            elif isinstance(cfg, dict):
                merged_config = cfg.copy()
            else:
                raise ConfigurationError(f"Unsupported configuration type: {type(cfg)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to process configuration: {e}") from e
    
    # Override with direct parameters (non-None values take precedence)
    for key, value in kwargs.items():
        if value is not None:
            merged_config[key] = value
    
    # Validate configuration using Pydantic schema if provided
    if config_schema and merged_config:
        try:
            validated = config_schema.model_validate(merged_config)
            merged_config = validated.model_dump(exclude_none=True)
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e
    
    return merged_config


def _normalize_positions(
    positions: Optional[Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]] = None,
    position: Optional[Union[Tuple[float, float], List[float], np.ndarray]] = None
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Normalize position parameters into consistent format.
    
    Args:
        positions: Multi-agent positions or None
        position: Single-agent position or None  
        
    Returns:
        Tuple of (normalized_positions, is_multi_agent)
        
    Raises:
        ConfigurationError: If both position and positions are provided or invalid format
    """
    if position is not None and positions is not None:
        raise ConfigurationError(
            "Cannot specify both 'position' (single-agent) and 'positions' (multi-agent). "
            "Please provide only one."
        )
    
    if positions is not None:
        # Multi-agent case
        try:
            positions_array = np.asarray(positions, dtype=float)
            if positions_array.ndim == 1 and len(positions_array) == 2:
                # Single position provided as positions
                return positions_array.reshape(1, 2), False
            elif positions_array.ndim == 2 and positions_array.shape[1] == 2:
                # Multiple positions
                return positions_array, positions_array.shape[0] > 1
            else:
                raise ValueError("Invalid positions shape")
        except Exception as e:
            raise ConfigurationError(f"Invalid positions format: {e}") from e
    
    elif position is not None:
        # Single-agent case  
        try:
            position_array = np.asarray(position, dtype=float)
            if position_array.shape != (2,):
                raise ValueError("Position must be a 2D coordinate")
            return position_array.reshape(1, 2), False
        except Exception as e:
            raise ConfigurationError(f"Invalid position format: {e}") from e
    
    return None, False


def create_navigator(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    positions: Optional[Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray]] = None,
    orientations: Optional[Union[float, List[float], np.ndarray]] = None,
    speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    max_speeds: Optional[Union[float, List[float], np.ndarray]] = None,
    angular_velocities: Optional[Union[float, List[float], np.ndarray]] = None,
    position: Optional[Union[Tuple[float, float], List[float], np.ndarray]] = None,
    orientation: Optional[float] = None,
    speed: Optional[float] = None,
    max_speed: Optional[float] = None,
    angular_velocity: Optional[float] = None,
    seed: Optional[int] = None,
    **kwargs: Any
) -> Navigator:
    """
    Create a Navigator instance with Hydra configuration support and comprehensive validation.
    
    This function provides a unified interface for creating both single-agent and multi-agent
    navigators, supporting Hydra DictConfig objects, direct parameters, and automatic
    type detection. It includes comprehensive error handling and parameter validation.
    
    Args:
        cfg: Hydra DictConfig or dictionary containing navigator configuration
        positions: Multi-agent initial positions as array-like (N, 2)
        orientations: Multi-agent initial orientations in degrees
        speeds: Multi-agent initial speeds
        max_speeds: Multi-agent maximum speeds
        angular_velocities: Multi-agent angular velocities in degrees per second
        position: Single-agent initial position as (x, y) tuple
        orientation: Single-agent initial orientation in degrees
        speed: Single-agent initial speed
        max_speed: Single-agent maximum speed  
        angular_velocity: Single-agent angular velocity in degrees per second
        seed: Random seed for reproducibility (integrates with seed_manager)
        **kwargs: Additional parameters passed to Navigator constructor
        
    Returns:
        Navigator: Configured navigator instance (single or multi-agent)
        
    Raises:
        ConfigurationError: If parameters are invalid or conflicting
        
    Examples:
        # Single-agent with direct parameters
        navigator = create_navigator(position=(10.0, 20.0), max_speed=5.0)
        
        # Multi-agent with positions array
        positions = np.array([[0, 0], [10, 10], [20, 20]])
        navigator = create_navigator(positions=positions, max_speeds=[5, 6, 7])
        
        # Using Hydra configuration
        navigator = create_navigator(cfg=hydra_config.navigator)
        
        # Configuration with overrides
        navigator = create_navigator(cfg=config, max_speed=12.0, seed=42)
    """
    # Set random seed if provided
    if seed is not None:
        set_global_seed(seed)
    
    # Merge and validate configuration
    merged_config = _validate_and_merge_config(
        cfg=cfg,
        config_schema=NavigatorConfig,
        positions=positions,
        orientations=orientations,
        speeds=speeds,
        max_speeds=max_speeds,
        angular_velocities=angular_velocities,
        position=position,
        orientation=orientation,
        speed=speed,
        max_speed=max_speed,
        angular_velocity=angular_velocity,
        **kwargs
    )
    
    # Normalize position parameters
    final_positions, is_multi_agent = _normalize_positions(
        positions=merged_config.get('positions'),
        position=merged_config.get('position')
    )
    
    try:
        if is_multi_agent or (final_positions is not None and final_positions.shape[0] > 1):
            # Multi-agent navigator
            nav_params = {
                'positions': final_positions,
                'orientations': merged_config.get('orientations'),
                'speeds': merged_config.get('speeds'),
                'max_speeds': merged_config.get('max_speeds'),
                'angular_velocities': merged_config.get('angular_velocities')
            }
            # Filter None values
            nav_params = {k: v for k, v in nav_params.items() if v is not None}
            return Navigator.multi(**nav_params)
        
        else:
            # Single-agent navigator
            nav_params = {
                'position': final_positions[0] if final_positions is not None else merged_config.get('position'),
                'orientation': merged_config.get('orientation', 0.0),
                'speed': merged_config.get('speed', 0.0),
                'max_speed': merged_config.get('max_speed', 1.0),
                'angular_velocity': merged_config.get('angular_velocity', 0.0)
            }
            # Filter None values
            nav_params = {k: v for k, v in nav_params.items() if v is not None}
            return Navigator.single(**nav_params)
            
    except Exception as e:
        raise ConfigurationError(f"Failed to create navigator: {e}") from e


def create_video_plume(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    video_path: Optional[Union[str, pathlib.Path]] = None,
    flip: Optional[bool] = None,
    kernel_size: Optional[int] = None,
    kernel_sigma: Optional[float] = None,
    **kwargs: Any
) -> VideoPlume:
    """
    Create a VideoPlume instance with Hydra configuration support and validation.
    
    This function provides a robust interface for creating video-based odor plume
    environments with comprehensive parameter validation, file existence checking,
    and Hydra configuration integration.
    
    Args:
        cfg: Hydra DictConfig or dictionary containing video plume configuration
        video_path: Path to the video file (MP4/AVI formats supported)
        flip: Whether to flip the video horizontally
        kernel_size: Size of Gaussian smoothing kernel (must be odd and positive)
        kernel_sigma: Standard deviation for Gaussian smoothing
        **kwargs: Additional parameters passed to VideoPlume constructor
        
    Returns:
        VideoPlume: Configured video plume environment instance
        
    Raises:
        ConfigurationError: If parameters are invalid
        FileNotFoundError: If video file does not exist
        
    Examples:
        # Basic usage with file path
        plume = create_video_plume("data/plume_video.mp4", flip=True)
        
        # Using Hydra configuration
        plume = create_video_plume(cfg=hydra_config.video_plume)
        
        # Configuration with overrides
        plume = create_video_plume(cfg=config, kernel_size=5, kernel_sigma=2.0)
    """
    # Merge and validate configuration
    merged_config = _validate_and_merge_config(
        cfg=cfg,
        config_schema=VideoPlumeConfig,
        video_path=video_path,
        flip=flip,
        kernel_size=kernel_size,
        kernel_sigma=kernel_sigma,
        **kwargs
    )
    
    # Validate required video_path
    if 'video_path' not in merged_config or merged_config['video_path'] is None:
        raise ConfigurationError("video_path is required")
    
    # Validate file existence
    video_path_obj = pathlib.Path(merged_config['video_path'])
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path_obj}")
    
    # Validate optional parameters
    if 'flip' in merged_config and not isinstance(merged_config['flip'], bool):
        raise ConfigurationError("flip must be a boolean")
    
    if 'kernel_size' in merged_config:
        ks = merged_config['kernel_size'] 
        if ks is not None and (not isinstance(ks, int) or ks <= 0):
            raise ConfigurationError("kernel_size must be a positive integer")
    
    if 'kernel_sigma' in merged_config:
        sigma = merged_config['kernel_sigma']
        if sigma is not None and (not isinstance(sigma, (float, int)) or sigma <= 0):
            raise ConfigurationError("kernel_sigma must be a positive number")
    
    try:
        # Create VideoPlume instance with validated parameters
        return VideoPlume(
            video_path=video_path_obj,
            flip=merged_config.get('flip', False),
            kernel_size=merged_config.get('kernel_size', 0),
            kernel_sigma=merged_config.get('kernel_sigma', 1.0)
        )
    except Exception as e:
        raise ConfigurationError(f"Failed to create VideoPlume: {e}") from e


def run_plume_simulation(
    navigator: Navigator,
    video_plume: VideoPlume,
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    num_steps: Optional[int] = None,
    dt: Optional[float] = None,
    step_size: Optional[float] = None,  # Backward compatibility
    sensor_distance: Optional[float] = None,
    sensor_angle: Optional[float] = None,
    record_trajectory: bool = True,
    seed: Optional[int] = None,
    **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute a complete odor plume navigation simulation with Hydra configuration support.
    
    This function orchestrates the simulation execution, providing frame-by-frame
    navigation updates, sensor sampling, and comprehensive trajectory recording.
    It supports both single-agent and multi-agent simulations with vectorized
    operations for optimal performance.
    
    Args:
        navigator: Navigator instance (single or multi-agent)
        video_plume: VideoPlume environment instance
        cfg: Hydra DictConfig or dictionary containing simulation configuration
        num_steps: Number of simulation steps to execute
        dt: Simulation time step size in seconds (preferred over step_size)
        step_size: Legacy parameter for time step (use dt instead)
        sensor_distance: Distance of sensors from agent center
        sensor_angle: Angular separation between sensors in degrees
        record_trajectory: Whether to record full trajectory data
        seed: Random seed for reproducibility
        **kwargs: Additional simulation parameters
        
    Returns:
        Tuple of (positions_history, orientations_history, odor_readings):
            - positions_history: Agent positions, shape (n_agents, n_steps + 1, 2)
            - orientations_history: Agent orientations, shape (n_agents, n_steps + 1)
            - odor_readings: Sensor readings, shape (n_agents, n_steps + 1)
            
    Raises:
        SimulationError: If simulation execution fails
        ConfigurationError: If parameters are invalid
        
    Examples:
        # Basic simulation
        positions, orientations, readings = run_plume_simulation(
            navigator, video_plume, num_steps=1000, dt=0.1
        )
        
        # Using Hydra configuration
        results = run_plume_simulation(
            navigator, video_plume, cfg=hydra_config.simulation
        )
        
        # With parameter overrides
        results = run_plume_simulation(
            navigator, video_plume, cfg=config, num_steps=500, seed=42
        )
    """
    # Set random seed if provided
    if seed is not None:
        set_global_seed(seed)
    
    # Merge and validate configuration
    merged_config = _validate_and_merge_config(
        cfg=cfg,
        config_schema=SimulationConfig,
        num_steps=num_steps,
        dt=dt,
        step_size=step_size,
        sensor_distance=sensor_distance,
        sensor_angle=sensor_angle,
        record_trajectory=record_trajectory,
        **kwargs
    )
    
    # Handle backward compatibility for step_size -> dt
    if 'step_size' in merged_config and 'dt' not in merged_config:
        merged_config['dt'] = merged_config.pop('step_size')
        warnings.warn(
            "Parameter 'step_size' is deprecated, use 'dt' instead",
            DeprecationWarning,
            stacklevel=2
        )
    
    # Validate required parameters
    if not hasattr(navigator, 'positions'):
        raise SimulationError("Navigator must have 'positions' attribute")
    
    if not hasattr(video_plume, 'get_frame'):
        raise SimulationError("VideoPlume must have 'get_frame' method")
    
    # Set default values for required parameters
    final_num_steps = merged_config.get('num_steps', 1000)
    final_dt = merged_config.get('dt', 1.0)
    final_sensor_distance = merged_config.get('sensor_distance', 5.0)
    final_sensor_angle = merged_config.get('sensor_angle', 45.0)
    
    # Validate parameter types and ranges
    if not isinstance(final_num_steps, int) or final_num_steps <= 0:
        raise ConfigurationError("num_steps must be a positive integer")
    
    if not isinstance(final_dt, (float, int)) or final_dt <= 0:
        raise ConfigurationError("dt must be a positive number")
    
    if final_sensor_distance < 0:
        raise ConfigurationError("sensor_distance must be non-negative")
    
    # Get simulation dimensions
    try:
        num_agents = navigator.num_agents
        if hasattr(video_plume, 'frame_count'):
            max_frames = video_plume.frame_count
        else:
            max_frames = final_num_steps + 1
            
        # Limit simulation steps to available frames
        effective_steps = min(final_num_steps, max_frames - 1)
        
        if effective_steps != final_num_steps:
            warnings.warn(
                f"Requested {final_num_steps} steps, but only {effective_steps} "
                f"frames available. Simulation will run for {effective_steps} steps.",
                UserWarning
            )
        
    except Exception as e:
        raise SimulationError(f"Failed to determine simulation dimensions: {e}") from e
    
    # Initialize trajectory recording arrays
    if record_trajectory:
        positions_history = np.zeros((num_agents, effective_steps + 1, 2), dtype=np.float64)
        orientations_history = np.zeros((num_agents, effective_steps + 1), dtype=np.float64)
        odor_readings = np.zeros((num_agents, effective_steps + 1), dtype=np.float64)
        
        # Record initial state
        positions_history[:, 0] = navigator.positions
        orientations_history[:, 0] = navigator.orientations
    else:
        # Minimal recording for return values
        positions_history = np.zeros((num_agents, 2, 2), dtype=np.float64)
        orientations_history = np.zeros((num_agents, 2), dtype=np.float64)
        odor_readings = np.zeros((num_agents, 2), dtype=np.float64)
        
        positions_history[:, 0] = navigator.positions
        orientations_history[:, 0] = navigator.orientations
    
    # Get initial odor readings
    try:
        initial_frame = video_plume.get_frame(0)
        if initial_frame is not None:
            initial_readings = navigator.sample_odor(initial_frame)
            if np.isscalar(initial_readings):
                odor_readings[:, 0] = initial_readings
            else:
                odor_readings[:, 0] = initial_readings
    except Exception as e:
        warnings.warn(f"Failed to get initial odor readings: {e}", UserWarning)
        odor_readings[:, 0] = 0.0
    
    # Execute simulation loop
    try:
        for step in range(effective_steps):
            # Get current frame (with bounds checking)
            frame_idx = min(step + 1, max_frames - 1)
            current_frame = video_plume.get_frame(frame_idx)
            
            if current_frame is None:
                warnings.warn(
                    f"Failed to get frame {frame_idx}, ending simulation early",
                    UserWarning
                )
                break
            
            # Update navigator state
            navigator.step(current_frame, dt=final_dt)
            
            # Record trajectory data
            if record_trajectory:
                positions_history[:, step + 1] = navigator.positions
                orientations_history[:, step + 1] = navigator.orientations
            
            # Sample odor at current position
            try:
                current_readings = navigator.sample_odor(current_frame)
                if record_trajectory:
                    if np.isscalar(current_readings):
                        odor_readings[:, step + 1] = current_readings
                    else:
                        odor_readings[:, step + 1] = current_readings
                else:
                    # Store only final readings for non-trajectory mode
                    if np.isscalar(current_readings):
                        odor_readings[:, 1] = current_readings
                    else:
                        odor_readings[:, 1] = current_readings
                        
            except Exception as e:
                warnings.warn(f"Failed to sample odor at step {step}: {e}", UserWarning)
                if record_trajectory:
                    odor_readings[:, step + 1] = 0.0
                else:
                    odor_readings[:, 1] = 0.0
    
    except Exception as e:
        raise SimulationError(f"Simulation failed at step {step}: {e}") from e
    
    # Handle non-trajectory mode by recording final state
    if not record_trajectory:
        positions_history[:, 1] = navigator.positions
        orientations_history[:, 1] = navigator.orientations
    
    # Validate output shapes
    expected_time_steps = effective_steps + 1 if record_trajectory else 2
    if (positions_history.shape != (num_agents, expected_time_steps, 2) or
        orientations_history.shape != (num_agents, expected_time_steps) or
        odor_readings.shape != (num_agents, expected_time_steps)):
        
        raise SimulationError(
            f"Output shape mismatch. Expected ({num_agents}, {expected_time_steps}, 2), "
            f"({num_agents}, {expected_time_steps}), ({num_agents}, {expected_time_steps}). "
            f"Got {positions_history.shape}, {orientations_history.shape}, {odor_readings.shape}"
        )
    
    return positions_history, orientations_history, odor_readings


# Factory methods for backward compatibility and convenience
def create_navigator_from_config(
    cfg: Union[DictConfig, Dict[str, Any], str, pathlib.Path],
    **kwargs: Any
) -> Navigator:
    """
    Create Navigator from Hydra configuration with override support.
    
    This is a convenience method that delegates to create_navigator() for
    backward compatibility with existing code that expects a config-first approach.
    
    Args:
        cfg: Hydra DictConfig, dictionary, or path to configuration file
        **kwargs: Parameter overrides
        
    Returns:
        Navigator: Configured navigator instance
        
    Examples:
        # Using DictConfig
        navigator = create_navigator_from_config(hydra_config.navigator)
        
        # With overrides
        navigator = create_navigator_from_config(config, max_speed=10.0)
    """
    if isinstance(cfg, (str, pathlib.Path)):
        raise ConfigurationError(
            "File path configuration loading is deprecated. "
            "Use Hydra compose API or pass DictConfig/dict objects directly."
        )
    
    return create_navigator(cfg=cfg, **kwargs)


def create_video_plume_from_config(
    cfg: Union[DictConfig, Dict[str, Any], str, pathlib.Path],
    video_path: Optional[Union[str, pathlib.Path]] = None,
    **kwargs: Any
) -> VideoPlume:
    """
    Create VideoPlume from Hydra configuration with override support.
    
    This is a convenience method that delegates to create_video_plume() for
    backward compatibility with existing code that expects a config-first approach.
    
    Args:
        cfg: Hydra DictConfig, dictionary, or path to configuration file
        video_path: Video file path (required if not in config)
        **kwargs: Parameter overrides
        
    Returns:
        VideoPlume: Configured video plume instance
        
    Examples:
        # Using DictConfig
        plume = create_video_plume_from_config(hydra_config.video_plume)
        
        # With video path and overrides
        plume = create_video_plume_from_config(config, video_path="video.mp4", flip=True)
    """
    if isinstance(cfg, (str, pathlib.Path)):
        raise ConfigurationError(
            "File path configuration loading is deprecated. "
            "Use Hydra compose API or pass DictConfig/dict objects directly."
        )
    
    return create_video_plume(cfg=cfg, video_path=video_path, **kwargs)


# Re-export key types for convenience
__all__ = [
    # Main API functions
    'create_navigator',
    'create_video_plume', 
    'run_plume_simulation',
    
    # Factory methods (backward compatibility)
    'create_navigator_from_config',
    'create_video_plume_from_config',
    
    # Core types for type hints
    'Navigator',
    'NavigatorProtocol', 
    'VideoPlume',
    
    # Configuration schemas
    'NavigatorConfig',
    'VideoPlumeConfig',
    'SimulationConfig',
    
    # Exceptions
    'ConfigurationError',
    'SimulationError',
]