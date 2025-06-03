"""
Consolidated public API module for odor plume navigation.

This module provides a unified, research-oriented API integrating navigation orchestration,
factory methods, and simulation execution with comprehensive Hydra-based configuration
support. Combines functionality from legacy interfaces/api.py, services/simulation_runner.py,
services/factories.py, and services/video_plume_factory.py into a standardized API surface.

The API supports multiple research frameworks:
- Kedro pipeline integration through factory method patterns
- RL framework compatibility via NumPy array interfaces and protocol-based definitions
- ML analysis tools through standardized data exchange and configuration management
- Interactive research environments through comprehensive parameter validation

Key Features:
    - Hydra DictConfig integration for hierarchical configuration management
    - Factory pattern implementation supporting both direct parameters and structured configs
    - Enhanced error handling with structured logging for research reproducibility
    - Protocol-based interfaces ensuring extensibility and algorithm compatibility
    - Automatic seed management integration for deterministic experiment execution
    - Performance-optimized initialization meeting <2s requirement for complex configurations

Example Usage:
    Kedro pipeline integration:
        >>> from hydra import compose, initialize
        >>> from {{cookiecutter.project_slug}}.api.navigation import create_navigator, run_plume_simulation
        >>> 
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = create_navigator(cfg.navigator)
        ...     plume = create_video_plume(cfg.video_plume)
        ...     results = run_plume_simulation(navigator, plume, cfg.simulation)

    RL framework integration:
        >>> from {{cookiecutter.project_slug}}.api.navigation import create_navigator
        >>> from {{cookiecutter.project_slug}}.utils.seed_manager import set_global_seed
        >>> 
        >>> set_global_seed(42)  # Reproducible RL experiments
        >>> navigator = create_navigator(position=(10, 10), max_speed=5.0)
        >>> # Use navigator.step(env_array) in RL loop

    Direct parameter usage:
        >>> navigator = create_navigator(
        ...     position=(50.0, 50.0),
        ...     orientation=45.0,
        ...     max_speed=10.0
        ... )
        >>> plume = create_video_plume(
        ...     video_path="data/plume_video.mp4",
        ...     flip=True,
        ...     kernel_size=5
        ... )
"""

import pathlib
from typing import List, Optional, Tuple, Union, Any, Dict
from contextlib import suppress
import numpy as np
from loguru import logger

# Hydra imports for configuration management
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    logger.warning("Hydra not available. Falling back to dictionary configuration.")

# Import dependencies from new module structure
from ..core.navigator import NavigatorProtocol
from ..core.controllers import SingleAgentController, MultiAgentController
from ..data.video_plume import VideoPlume
from ..config.schemas import (
    NavigatorConfig, 
    SingleAgentConfig, 
    MultiAgentConfig, 
    VideoPlumeConfig,
    SimulationConfig
)
from ..utils.seed_manager import get_seed_manager, SeedManager
from ..utils.visualization import visualize_simulation_results, visualize_trajectory


def create_navigator(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    position: Optional[Union[Tuple[float, float], List[float], np.ndarray]] = None,
    positions: Optional[Union[List[Tuple[float, float]], np.ndarray]] = None,
    orientation: Optional[float] = None,
    orientations: Optional[Union[List[float], np.ndarray]] = None,
    speed: Optional[float] = None,
    speeds: Optional[Union[List[float], np.ndarray]] = None,
    max_speed: Optional[float] = None,
    max_speeds: Optional[Union[List[float], np.ndarray]] = None,
    angular_velocity: Optional[float] = None,
    angular_velocities: Optional[Union[List[float], np.ndarray]] = None,
    **kwargs: Any
) -> NavigatorProtocol:
    """
    Create a Navigator instance with Hydra configuration support and enhanced validation.

    This function provides a unified interface for creating both single-agent and multi-agent
    navigators using either direct parameter specification or Hydra DictConfig objects.
    Supports automatic detection of single vs multi-agent scenarios based on parameter shapes.

    Parameters
    ----------
    cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Hydra configuration object or dictionary containing navigator parameters.
        Takes precedence over individual parameters if provided, by default None
    position : Optional[Union[Tuple[float, float], List[float], np.ndarray]], optional
        Initial position for single-agent navigator [x, y], by default None
    positions : Optional[Union[List[Tuple[float, float]], np.ndarray]], optional
        Initial positions for multi-agent navigator [[x1, y1], [x2, y2], ...], by default None
    orientation : Optional[float], optional
        Initial orientation in degrees for single agent, by default None
    orientations : Optional[Union[List[float], np.ndarray]], optional
        Initial orientations in degrees for multiple agents, by default None
    speed : Optional[float], optional
        Initial speed for single agent, by default None
    speeds : Optional[Union[List[float], np.ndarray]], optional
        Initial speeds for multiple agents, by default None
    max_speed : Optional[float], optional
        Maximum speed for single agent, by default None
    max_speeds : Optional[Union[List[float], np.ndarray]], optional
        Maximum speeds for multiple agents, by default None
    angular_velocity : Optional[float], optional
        Angular velocity for single agent in degrees/second, by default None
    angular_velocities : Optional[Union[List[float], np.ndarray]], optional
        Angular velocities for multiple agents in degrees/second, by default None
    **kwargs : Any
        Additional parameters for navigator configuration

    Returns
    -------
    NavigatorProtocol
        Configured navigator instance (SingleAgentController or MultiAgentController)

    Raises
    ------
    ValueError
        If both single and multi-agent parameters are specified
        If configuration validation fails
        If required parameters are missing
    TypeError
        If parameter types are invalid
    RuntimeError
        If navigator initialization exceeds performance requirements

    Examples
    --------
    Create single agent navigator with Hydra config:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = create_navigator(cfg.navigator)

    Create multi-agent navigator with direct parameters:
        >>> navigator = create_navigator(
        ...     positions=[(10, 20), (30, 40)],
        ...     orientations=[0, 90],
        ...     max_speeds=[5.0, 8.0]
        ... )

    Create single agent with mixed configuration:
        >>> cfg = DictConfig({"max_speed": 10.0, "angular_velocity": 0.1})
        >>> navigator = create_navigator(cfg, position=(50, 50), orientation=45)
    
    Notes
    -----
    Configuration precedence order:
    1. Direct parameters (position, orientation, etc.)
    2. Hydra DictConfig object values
    3. Default values from configuration schemas
    
    The function automatically detects single vs multi-agent scenarios:
    - Uses positions/orientations (plural) for multi-agent
    - Uses position/orientation (singular) for single-agent
    - Raises error if both are specified
    """
    # Initialize logger with function context
    func_logger = logger.bind(
        module=__name__,
        function="create_navigator",
        cfg_provided=cfg is not None,
        direct_params_provided=any([
            position is not None, positions is not None,
            orientation is not None, orientations is not None
        ])
    )

    try:
        # Validate parameter consistency
        if position is not None and positions is not None:
            raise ValueError(
                "Cannot specify both 'position' (single-agent) and 'positions' (multi-agent). "
                "Please provide only one."
            )

        if orientation is not None and orientations is not None:
            raise ValueError(
                "Cannot specify both 'orientation' (single-agent) and 'orientations' (multi-agent). "
                "Please provide only one."
            )

        # Process configuration object
        config_params = {}
        if cfg is not None:
            try:
                if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
                    config_params = OmegaConf.to_container(cfg, resolve=True)
                elif isinstance(cfg, dict):
                    config_params = cfg.copy()
                else:
                    raise TypeError(f"Configuration must be DictConfig or dict, got {type(cfg)}")
            except Exception as e:
                func_logger.error(f"Failed to process configuration: {e}")
                raise ValueError(f"Invalid configuration object: {e}") from e

        # Determine navigator type and merge parameters with precedence
        # Direct parameters override configuration values
        is_multi_agent = (
            positions is not None or 
            orientations is not None or
            speeds is not None or
            max_speeds is not None or
            angular_velocities is not None or
            "positions" in config_params
        )

        if is_multi_agent:
            # Multi-agent navigator
            merged_params = {
                "positions": positions or config_params.get("positions"),
                "orientations": orientations or config_params.get("orientations"),
                "speeds": speeds or config_params.get("speeds"),
                "max_speeds": max_speeds or config_params.get("max_speeds"),
                "angular_velocities": angular_velocities or config_params.get("angular_velocities"),
            }
            
            # Add any additional config parameters
            merged_params.update({k: v for k, v in config_params.items() 
                                if k not in merged_params})
            merged_params.update(kwargs)

            # Validate configuration using Pydantic schema
            try:
                validated_config = MultiAgentConfig(**merged_params)
                func_logger.info(
                    "Multi-agent navigator configuration validated",
                    extra={
                        "num_agents": len(validated_config.positions) if validated_config.positions else None,
                        "config_source": "hydra" if cfg is not None else "direct"
                    }
                )
            except Exception as e:
                func_logger.error(f"Multi-agent configuration validation failed: {e}")
                raise ValueError(f"Invalid multi-agent configuration: {e}") from e

            # Create multi-agent controller
            navigator = MultiAgentController(
                positions=validated_config.positions,
                orientations=validated_config.orientations,
                speeds=validated_config.speeds,
                max_speeds=validated_config.max_speeds,
                angular_velocities=validated_config.angular_velocities,
                config=validated_config,
                seed_manager=get_seed_manager()
            )

        else:
            # Single-agent navigator
            merged_params = {
                "position": position or config_params.get("position"),
                "orientation": orientation or config_params.get("orientation", 0.0),
                "speed": speed or config_params.get("speed", 0.0),
                "max_speed": max_speed or config_params.get("max_speed", 1.0),
                "angular_velocity": angular_velocity or config_params.get("angular_velocity", 0.0),
            }
            
            # Add any additional config parameters
            merged_params.update({k: v for k, v in config_params.items() 
                                if k not in merged_params})
            merged_params.update(kwargs)

            # Validate configuration using Pydantic schema
            try:
                validated_config = SingleAgentConfig(**merged_params)
                func_logger.info(
                    "Single-agent navigator configuration validated",
                    extra={
                        "position": validated_config.position,
                        "orientation": validated_config.orientation,
                        "max_speed": validated_config.max_speed,
                        "config_source": "hydra" if cfg is not None else "direct"
                    }
                )
            except Exception as e:
                func_logger.error(f"Single-agent configuration validation failed: {e}")
                raise ValueError(f"Invalid single-agent configuration: {e}") from e

            # Create single-agent controller
            navigator = SingleAgentController(
                position=validated_config.position,
                orientation=validated_config.orientation,
                speed=validated_config.speed,
                max_speed=validated_config.max_speed,
                angular_velocity=validated_config.angular_velocity,
                config=validated_config,
                seed_manager=get_seed_manager()
            )

        func_logger.info(
            f"Navigator created successfully",
            extra={
                "navigator_type": "multi-agent" if is_multi_agent else "single-agent",
                "num_agents": navigator.num_agents
            }
        )
        
        return navigator

    except Exception as e:
        func_logger.error(f"Navigator creation failed: {e}")
        raise RuntimeError(f"Failed to create navigator: {e}") from e


def create_video_plume(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    video_path: Optional[Union[str, pathlib.Path]] = None,
    flip: Optional[bool] = None,
    grayscale: Optional[bool] = None,
    kernel_size: Optional[int] = None,
    kernel_sigma: Optional[float] = None,
    threshold: Optional[float] = None,
    normalize: Optional[bool] = None,
    **kwargs: Any
) -> VideoPlume:
    """
    Create a VideoPlume instance with Hydra configuration support and enhanced validation.

    This function provides a unified interface for creating video-based odor plume environments
    using either direct parameter specification or Hydra DictConfig objects. Supports
    comprehensive video preprocessing options and automatic parameter validation.

    Parameters
    ----------
    cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Hydra configuration object or dictionary containing video plume parameters.
        Takes precedence over individual parameters if provided, by default None
    video_path : Optional[Union[str, pathlib.Path]], optional
        Path to the video file (MP4/AVI formats supported), by default None
    flip : Optional[bool], optional
        Whether to flip frames horizontally, by default None
    grayscale : Optional[bool], optional
        Whether to convert frames to grayscale, by default None
    kernel_size : Optional[int], optional
        Size of Gaussian kernel for smoothing (must be odd and positive), by default None
    kernel_sigma : Optional[float], optional
        Standard deviation for Gaussian kernel, by default None
    threshold : Optional[float], optional
        Threshold value for binary detection, by default None
    normalize : Optional[bool], optional
        Whether to normalize frame values to [0, 1] range, by default None
    **kwargs : Any
        Additional parameters for VideoPlume configuration

    Returns
    -------
    VideoPlume
        Configured VideoPlume instance ready for simulation use

    Raises
    ------
    ValueError
        If configuration validation fails
        If video_path is not provided or invalid
        If preprocessing parameters are invalid
    FileNotFoundError
        If the specified video file does not exist
    TypeError
        If parameter types are invalid
    RuntimeError
        If VideoPlume initialization exceeds performance requirements

    Examples
    --------
    Create with Hydra configuration:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     plume = create_video_plume(cfg.video_plume)

    Create with direct parameters:
        >>> plume = create_video_plume(
        ...     video_path="data/plume_video.mp4",
        ...     flip=True,
        ...     kernel_size=5,
        ...     kernel_sigma=1.0
        ... )

    Create with mixed configuration:
        >>> cfg = DictConfig({"flip": True, "grayscale": True})
        >>> plume = create_video_plume(
        ...     cfg,
        ...     video_path="data/experiment_plume.mp4",
        ...     kernel_size=3
        ... )

    Notes
    -----
    Configuration precedence order:
    1. Direct parameters (video_path, flip, etc.)
    2. Hydra DictConfig object values
    3. Default values from VideoPlumeConfig schema

    Supported video formats:
    - MP4 (recommended for best compatibility)
    - AVI with standard codecs
    - Automatic frame count and metadata extraction
    """
    # Initialize logger with function context
    func_logger = logger.bind(
        module=__name__,
        function="create_video_plume",
        cfg_provided=cfg is not None,
        video_path_provided=video_path is not None
    )

    try:
        # Process configuration object
        config_params = {}
        if cfg is not None:
            try:
                if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
                    config_params = OmegaConf.to_container(cfg, resolve=True)
                elif isinstance(cfg, dict):
                    config_params = cfg.copy()
                else:
                    raise TypeError(f"Configuration must be DictConfig or dict, got {type(cfg)}")
            except Exception as e:
                func_logger.error(f"Failed to process configuration: {e}")
                raise ValueError(f"Invalid configuration object: {e}") from e

        # Merge parameters with precedence (direct params override config)
        merged_params = {
            "video_path": video_path or config_params.get("video_path"),
            "flip": flip if flip is not None else config_params.get("flip"),
            "grayscale": grayscale if grayscale is not None else config_params.get("grayscale"),
            "kernel_size": kernel_size if kernel_size is not None else config_params.get("kernel_size"),
            "kernel_sigma": kernel_sigma if kernel_sigma is not None else config_params.get("kernel_sigma"),
            "threshold": threshold if threshold is not None else config_params.get("threshold"),
            "normalize": normalize if normalize is not None else config_params.get("normalize"),
        }

        # Add any additional config parameters
        merged_params.update({k: v for k, v in config_params.items() 
                            if k not in merged_params})
        merged_params.update(kwargs)

        # Remove None values for cleaner validation
        merged_params = {k: v for k, v in merged_params.items() if v is not None}

        # Validate required video_path parameter
        if "video_path" not in merged_params or merged_params["video_path"] is None:
            raise ValueError("video_path is required for VideoPlume creation")

        # Validate configuration using Pydantic schema
        try:
            validated_config = VideoPlumeConfig(**merged_params)
            func_logger.info(
                "VideoPlume configuration validated",
                extra={
                    "video_path": str(validated_config.video_path),
                    "flip": validated_config.flip,
                    "grayscale": validated_config.grayscale,
                    "preprocessing_enabled": any([
                        validated_config.kernel_size and validated_config.kernel_size > 0,
                        validated_config.flip,
                        validated_config.threshold is not None
                    ]),
                    "config_source": "hydra" if cfg is not None else "direct"
                }
            )
        except Exception as e:
            func_logger.error(f"VideoPlume configuration validation failed: {e}")
            raise ValueError(f"Invalid VideoPlume configuration: {e}") from e

        # Validate video file existence
        video_path_obj = pathlib.Path(validated_config.video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file does not exist: {video_path_obj}")

        if not video_path_obj.is_file():
            raise ValueError(f"Video path is not a file: {video_path_obj}")

        # Create VideoPlume instance using factory method
        plume = VideoPlume.from_config(validated_config)

        func_logger.info(
            "VideoPlume created successfully",
            extra={
                "video_path": str(video_path_obj),
                "frame_count": plume.frame_count,
                "width": plume.width,
                "height": plume.height,
                "fps": plume.fps,
                "duration": plume.duration
            }
        )

        return plume

    except Exception as e:
        func_logger.error(f"VideoPlume creation failed: {e}")
        raise RuntimeError(f"Failed to create VideoPlume: {e}") from e


def run_plume_simulation(
    navigator: NavigatorProtocol,
    video_plume: VideoPlume,
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    num_steps: Optional[int] = None,
    dt: Optional[float] = None,
    step_size: Optional[float] = None,  # Backward compatibility
    sensor_distance: Optional[float] = None,
    sensor_angle: Optional[float] = None,
    record_trajectories: bool = True,
    **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute a complete plume navigation simulation with Hydra configuration support.

    This function orchestrates frame-by-frame agent navigation through video-based odor
    plume environments with comprehensive data collection and performance monitoring.
    Supports both single-agent and multi-agent scenarios with automatic trajectory recording.

    Parameters
    ----------
    navigator : NavigatorProtocol
        Navigator instance (SingleAgentController or MultiAgentController)
    video_plume : VideoPlume
        VideoPlume environment instance providing odor concentration data
    cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Hydra configuration object or dictionary containing simulation parameters, by default None
    num_steps : Optional[int], optional
        Number of simulation steps to execute, by default None
    dt : Optional[float], optional
        Simulation timestep in seconds, by default None
    step_size : Optional[float], optional
        Legacy parameter for backward compatibility (converted to dt), by default None
    sensor_distance : Optional[float], optional
        Distance for sensor sampling, by default None
    sensor_angle : Optional[float], optional
        Angle for sensor sampling, by default None
    record_trajectories : bool, optional
        Whether to record position and orientation trajectories, by default True
    **kwargs : Any
        Additional simulation parameters

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        positions_history : np.ndarray
            Agent positions with shape (num_agents, num_steps + 1, 2)
        orientations_history : np.ndarray
            Agent orientations with shape (num_agents, num_steps + 1)
        odor_readings : np.ndarray
            Sensor readings with shape (num_agents, num_steps + 1)

    Raises
    ------
    ValueError
        If required parameters are missing or invalid
        If navigator or video_plume are None
        If configuration validation fails
    TypeError
        If navigator doesn't implement NavigatorProtocol
        If video_plume is not a VideoPlume instance
    RuntimeError
        If simulation execution fails or exceeds performance requirements

    Examples
    --------
    Run simulation with Hydra configuration:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = create_navigator(cfg.navigator)
        ...     plume = create_video_plume(cfg.video_plume)
        ...     positions, orientations, readings = run_plume_simulation(
        ...         navigator, plume, cfg.simulation
        ...     )

    Run simulation with direct parameters:
        >>> positions, orientations, readings = run_plume_simulation(
        ...     navigator=navigator,
        ...     video_plume=plume,
        ...     num_steps=1000,
        ...     dt=0.1
        ... )

    Use results for analysis:
        >>> print(f"Simulation completed: {positions.shape[1]} steps, {positions.shape[0]} agents")
        >>> final_positions = positions[:, -1, :]
        >>> trajectory_lengths = np.sum(np.diff(positions, axis=1)**2, axis=(1,2))**0.5

    Notes
    -----
    Performance characteristics:
    - Optimized for 30+ FPS execution with real-time visualization
    - Memory-efficient trajectory recording with configurable storage
    - Automatic frame synchronization between navigator and video plume
    - Progress logging for long-running simulations

    Configuration precedence order:
    1. Direct parameters (num_steps, dt, etc.)
    2. Hydra DictConfig object values
    3. Default values from SimulationConfig schema

    Backward compatibility:
    - step_size parameter is converted to dt for legacy compatibility
    - Original parameter names are preserved where possible
    """
    # Initialize logger with simulation context
    sim_logger = logger.bind(
        module=__name__,
        function="run_plume_simulation",
        navigator_type=type(navigator).__name__,
        num_agents=navigator.num_agents,
        video_frames=video_plume.frame_count,
        cfg_provided=cfg is not None
    )

    try:
        # Validate required inputs
        if navigator is None:
            raise ValueError("navigator parameter is required")
        if video_plume is None:
            raise ValueError("video_plume parameter is required")

        # Type validation
        if not hasattr(navigator, 'positions') or not hasattr(navigator, 'step'):
            raise TypeError("navigator must implement NavigatorProtocol interface")
        
        if not hasattr(video_plume, 'get_frame') or not hasattr(video_plume, 'frame_count'):
            raise TypeError("video_plume must be a VideoPlume instance")

        # Process configuration object
        config_params = {}
        if cfg is not None:
            try:
                if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
                    config_params = OmegaConf.to_container(cfg, resolve=True)
                elif isinstance(cfg, dict):
                    config_params = cfg.copy()
                else:
                    raise TypeError(f"Configuration must be DictConfig or dict, got {type(cfg)}")
            except Exception as e:
                sim_logger.error(f"Failed to process configuration: {e}")
                raise ValueError(f"Invalid configuration object: {e}") from e

        # Merge parameters with precedence and handle backward compatibility
        merged_params = {
            "num_steps": num_steps or config_params.get("num_steps"),
            "dt": dt or step_size or config_params.get("dt") or config_params.get("step_size"),
            "sensor_distance": sensor_distance or config_params.get("sensor_distance", 5.0),
            "sensor_angle": sensor_angle or config_params.get("sensor_angle", 45.0),
            "record_trajectories": record_trajectories and config_params.get("record_trajectories", True),
        }

        # Add any additional config parameters
        merged_params.update({k: v for k, v in config_params.items() 
                            if k not in merged_params})
        merged_params.update(kwargs)

        # Remove None values for validation
        merged_params = {k: v for k, v in merged_params.items() if v is not None}

        # Validate configuration using Pydantic schema
        try:
            validated_config = SimulationConfig(**merged_params)
            sim_logger.info(
                "Simulation configuration validated",
                extra={
                    "num_steps": validated_config.num_steps,
                    "dt": validated_config.dt,
                    "sensor_distance": validated_config.sensor_distance,
                    "sensor_angle": validated_config.sensor_angle,
                    "record_trajectories": validated_config.record_trajectories,
                    "config_source": "hydra" if cfg is not None else "direct"
                }
            )
        except Exception as e:
            sim_logger.error(f"Simulation configuration validation failed: {e}")
            raise ValueError(f"Invalid simulation configuration: {e}") from e

        # Initialize seed manager for reproducible execution
        seed_manager = get_seed_manager()
        if seed_manager.current_seed is not None:
            sim_logger.info(f"Simulation running with seed: {seed_manager.current_seed}")

        # Get simulation parameters
        num_steps = validated_config.num_steps
        dt = validated_config.dt
        num_agents = navigator.num_agents

        # Initialize trajectory storage if recording enabled
        if validated_config.record_trajectories:
            positions_history = np.zeros((num_agents, num_steps + 1, 2))
            orientations_history = np.zeros((num_agents, num_steps + 1))
            odor_readings = np.zeros((num_agents, num_steps + 1))
            
            # Store initial state
            positions_history[:, 0] = navigator.positions
            orientations_history[:, 0] = navigator.orientations
            
            # Get initial odor readings
            current_frame = video_plume.get_frame(0)
            if hasattr(navigator, 'sample_odor'):
                initial_readings = navigator.sample_odor(current_frame)
                if isinstance(initial_readings, (int, float)):
                    odor_readings[:, 0] = initial_readings
                else:
                    odor_readings[:, 0] = initial_readings
            else:
                odor_readings[:, 0] = 0.0
        else:
            # Minimal storage for return compatibility
            positions_history = np.zeros((num_agents, 2, 2))
            orientations_history = np.zeros((num_agents, 2))
            odor_readings = np.zeros((num_agents, 2))

        sim_logger.info(
            "Starting simulation execution",
            extra={
                "total_steps": num_steps,
                "estimated_duration": num_steps * dt,
                "memory_usage": f"{positions_history.nbytes + orientations_history.nbytes + odor_readings.nbytes:.1f} bytes"
            }
        )

        # Execute simulation loop
        for step in range(num_steps):
            try:
                # Get current frame with bounds checking
                frame_idx = min(step + 1, video_plume.frame_count - 1)
                current_frame = video_plume.get_frame(frame_idx)
                
                # Update navigator state
                navigator.step(current_frame)
                
                # Record trajectory data if enabled
                if validated_config.record_trajectories:
                    positions_history[:, step + 1] = navigator.positions
                    orientations_history[:, step + 1] = navigator.orientations
                    
                    # Sample odor at current position
                    if hasattr(navigator, 'sample_odor'):
                        readings = navigator.sample_odor(current_frame)
                        if isinstance(readings, (int, float)):
                            odor_readings[:, step + 1] = readings
                        else:
                            odor_readings[:, step + 1] = readings
                    else:
                        odor_readings[:, step + 1] = 0.0

                # Progress logging for long simulations
                if num_steps > 100 and (step + 1) % (num_steps // 10) == 0:
                    progress = (step + 1) / num_steps * 100
                    sim_logger.debug(f"Simulation progress: {progress:.1f}% ({step + 1}/{num_steps} steps)")

            except Exception as e:
                sim_logger.error(f"Simulation failed at step {step}: {e}")
                raise RuntimeError(f"Simulation execution failed at step {step}: {e}") from e

        # Handle non-recording case by storing final state
        if not validated_config.record_trajectories:
            positions_history[:, 0] = navigator.positions
            orientations_history[:, 0] = navigator.orientations
            if hasattr(navigator, 'sample_odor'):
                final_frame = video_plume.get_frame(video_plume.frame_count - 1)
                readings = navigator.sample_odor(final_frame)
                if isinstance(readings, (int, float)):
                    odor_readings[:, 0] = readings
                else:
                    odor_readings[:, 0] = readings

        sim_logger.info(
            "Simulation completed successfully",
            extra={
                "steps_executed": num_steps,
                "final_positions": positions_history[:, -1, :].tolist() if validated_config.record_trajectories else positions_history[:, 0, :].tolist(),
                "trajectory_recorded": validated_config.record_trajectories,
                "data_shape": {
                    "positions": positions_history.shape,
                    "orientations": orientations_history.shape,
                    "readings": odor_readings.shape
                }
            }
        )

        return positions_history, orientations_history, odor_readings

    except Exception as e:
        sim_logger.error(f"Simulation execution failed: {e}")
        raise RuntimeError(f"Failed to execute simulation: {e}") from e


def visualize_plume_simulation(
    positions: np.ndarray,
    orientations: np.ndarray,
    odor_readings: Optional[np.ndarray] = None,
    plume_frames: Optional[np.ndarray] = None,
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    output_path: Optional[Union[str, pathlib.Path]] = None,
    show_plot: bool = True,
    close_plot: Optional[bool] = None,
    animation: bool = False,
    **kwargs: Any
) -> "matplotlib.figure.Figure":
    """
    Visualize simulation results with comprehensive formatting and export options.

    This function provides publication-quality visualization of agent trajectories
    and environmental data with support for both static plots and animated sequences.
    Integrates with Hydra configuration for consistent visualization parameters.

    Parameters
    ----------
    positions : np.ndarray
        Agent positions with shape (num_agents, num_steps, 2)
    orientations : np.ndarray
        Agent orientations with shape (num_agents, num_steps)
    odor_readings : Optional[np.ndarray], optional
        Sensor readings with shape (num_agents, num_steps), by default None
    plume_frames : Optional[np.ndarray], optional
        Video frames with shape (num_steps, height, width, channels), by default None
    cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Hydra configuration for visualization parameters, by default None
    output_path : Optional[Union[str, pathlib.Path]], optional
        Path to save visualization output, by default None
    show_plot : bool, optional
        Whether to display the plot interactively, by default True
    close_plot : Optional[bool], optional
        Whether to close plot after saving, by default None
    animation : bool, optional
        Whether to create animated visualization, by default False
    **kwargs : Any
        Additional visualization parameters

    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib figure object

    Examples
    --------
    Basic trajectory visualization:
        >>> fig = visualize_plume_simulation(positions, orientations)

    Publication-quality plot with configuration:
        >>> fig = visualize_plume_simulation(
        ...     positions, orientations, odor_readings,
        ...     cfg=viz_config,
        ...     output_path="results/trajectory.png",
        ...     show_plot=False
        ... )

    Animated visualization:
        >>> fig = visualize_plume_simulation(
        ...     positions, orientations, 
        ...     plume_frames=frames,
        ...     animation=True,
        ...     output_path="results/animation.mp4"
        ... )
    """
    # Initialize logger
    viz_logger = logger.bind(
        module=__name__,
        function="visualize_plume_simulation",
        num_agents=positions.shape[0],
        num_steps=positions.shape[1],
        animation=animation
    )

    try:
        # Process configuration
        config_params = {}
        if cfg is not None:
            if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
                config_params = OmegaConf.to_container(cfg, resolve=True)
            elif isinstance(cfg, dict):
                config_params = cfg.copy()

        # Merge visualization parameters
        viz_params = {
            "output_path": output_path,
            "show_plot": show_plot,
            "close_plot": close_plot,
            **config_params,
            **kwargs
        }

        # Select appropriate visualization function
        if animation:
            return visualize_simulation_results(
                positions=positions,
                orientations=orientations,
                odor_readings=odor_readings,
                plume_frames=plume_frames,
                **viz_params
            )
        else:
            return visualize_trajectory(
                positions=positions,
                orientations=orientations,
                plume_frames=plume_frames,
                **viz_params
            )

    except Exception as e:
        viz_logger.error(f"Visualization failed: {e}")
        raise RuntimeError(f"Failed to create visualization: {e}") from e


# Legacy compatibility aliases
create_navigator_from_config = create_navigator
create_video_plume_from_config = create_video_plume
run_simulation = run_plume_simulation
visualize_simulation_results = visualize_plume_simulation

# Export public API
__all__ = [
    "create_navigator",
    "create_video_plume", 
    "run_plume_simulation",
    "visualize_plume_simulation",
    # Legacy aliases
    "create_navigator_from_config",
    "create_video_plume_from_config", 
    "run_simulation",
    "visualize_simulation_results"
]