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
    - Enhanced factory methods for PlumeNavSim-v0 environment registration
    - Structured JSON logging with correlation IDs (request_id/episode_id) for distributed tracing
    - Compatibility layer support for seamless legacy gym and modern Gymnasium API operation
    - Performance timing integration with ≤10ms threshold monitoring

Example Usage:
    Kedro pipeline integration:
        >>> from hydra import compose, initialize
        >>> from odor_plume_nav.api.navigation import create_navigator, run_plume_simulation
        >>> 
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     navigator = create_navigator(cfg.navigator)
        ...     plume = create_video_plume(cfg.video_plume)
        ...     results = run_plume_simulation(navigator, plume, cfg.simulation)

    RL framework integration:
        >>> from odor_plume_nav.api.navigation import create_navigator
        >>> from odor_plume_nav.utils.seed_utils import set_global_seed
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

    Reinforcement learning training workflow:
        >>> from odor_plume_nav.api.navigation import create_gymnasium_environment
        >>> from stable_baselines3 import PPO
        >>> 
        >>> # Create RL-ready environment with PlumeNavSim-v0 support
        >>> env = create_gymnasium_environment(
        ...     environment_id="PlumeNavSim-v0",  # New Gymnasium 0.29.x compliant environment
        ...     video_path="data/plume_experiment.mp4",
        ...     initial_position=(50, 50),
        ...     max_speed=2.0,
        ...     render_mode="human"
        ... )
        >>> 
        >>> # Train policy with stable-baselines3
        >>> model = PPO("MultiInputPolicy", env, verbose=1)
        >>> model.learn(total_timesteps=50000)

    Legacy migration to RL:
        >>> # Start with traditional components
        >>> navigator = create_navigator(position=(50, 50), max_speed=2.0)
        >>> plume = create_video_plume(video_path="data/plume_video.mp4")
        >>> 
        >>> # Migrate to RL environment
        >>> env = from_legacy(navigator, plume, render_mode="human")
        >>> 
        >>> # Now use with RL algorithms
        >>> from stable_baselines3 import SAC
        >>> model = SAC("MultiInputPolicy", env)
"""

import pathlib
import time
import uuid
from typing import List, Optional, Tuple, Union, Any, Dict
from contextlib import suppress, contextmanager
import numpy as np

# Enhanced logging and correlation tracking
try:
    from ..utils.logging_setup import (
        get_logger, correlation_context, PerformanceMetrics, get_correlation_context
    )
    logger = get_logger(__name__)
    ENHANCED_LOGGING_AVAILABLE = True
except ImportError:
    from loguru import logger
    ENHANCED_LOGGING_AVAILABLE = False

# Hydra imports for configuration management
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    logger.warning("Hydra not available. Falling back to dictionary configuration.")

# Import dependencies from unified module structure
from ..core.protocols import NavigatorProtocol
from ..core.controllers import SingleAgentController, MultiAgentController
from ..environments.video_plume import VideoPlume
from ..config.schemas import (
    NavigatorConfig, 
    SingleAgentConfig, 
    MultiAgentConfig, 
    VideoPlumeConfig,
    SimulationConfig
)

# Seed management utilities with enhanced integration
try:
    from ..utils.seed_utils import (
        set_global_seed, get_seed_context, setup_reproducible_environment,
        get_gymnasium_seed_parameter
    )
    SEED_UTILS_AVAILABLE = True
except ImportError:
    # Fallback imports for backward compatibility
    try:
        from ..utils.seed_manager import get_seed_manager, SeedManager
        SEED_UTILS_AVAILABLE = False
    except ImportError:
        SEED_UTILS_AVAILABLE = False

from ..utils.visualization import visualize_simulation_results, visualize_trajectory

# Gymnasium environment imports for RL integration
try:
    from ..environments.gymnasium_env import GymnasiumEnv, create_gymnasium_environment as _create_gymnasium_env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    GymnasiumEnv = None
    _create_gymnasium_env = None

# Compatibility layer for dual API support
try:
    from ..environments.compat import (
        detect_api_version, create_compatibility_wrapper, 
        CompatibilityMode, PerformanceViolationError
    )
    COMPATIBILITY_LAYER_AVAILABLE = True
except ImportError:
    COMPATIBILITY_LAYER_AVAILABLE = False

# Performance monitoring constants
STEP_PERFORMANCE_THRESHOLD = 0.010  # 10ms step() threshold per requirements


@contextmanager
def performance_monitor(operation_name: str, threshold: Optional[float] = None):
    """
    Context manager for performance monitoring with automatic threshold warnings.
    
    Args:
        operation_name: Name of the operation being monitored
        threshold: Performance threshold in seconds (default: based on operation type)
        
    Yields:
        PerformanceMetrics object for the operation
    """
    if ENHANCED_LOGGING_AVAILABLE:
        with correlation_context(operation_name) as ctx:
            start_time = time.perf_counter()
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                correlation_id=ctx.correlation_id if hasattr(ctx, 'correlation_id') else str(uuid.uuid4())
            )
            
            try:
                yield metrics
            finally:
                duration = time.perf_counter() - start_time
                metrics.duration = duration
                
                # Apply threshold checking
                if threshold is None:
                    if "step" in operation_name.lower():
                        threshold = STEP_PERFORMANCE_THRESHOLD
                    else:
                        threshold = 1.0
                
                if duration > threshold:
                    logger.warning(
                        f"Slow operation detected: {operation_name}",
                        extra={
                            "operation": operation_name,
                            "duration": duration,
                            "threshold": threshold,
                            "metric_type": "performance_violation",
                            "correlation_id": metrics.correlation_id
                        }
                    )
                else:
                    logger.debug(
                        f"Operation completed: {operation_name}",
                        extra={
                            "operation": operation_name,
                            "duration": duration,
                            "correlation_id": metrics.correlation_id
                        }
                    )
    else:
        # Fallback for basic performance monitoring
        start_time = time.perf_counter()
        metrics = type('PerformanceMetrics', (), {
            'operation_name': operation_name,
            'start_time': start_time,
            'duration': None
        })()
        
        try:
            yield metrics
        finally:
            duration = time.perf_counter() - start_time
            metrics.duration = duration
            
            if threshold and duration > threshold:
                logger.warning(f"Slow operation: {operation_name} took {duration:.3f}s")


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
    Create a Navigator instance with enhanced seed management and structured logging.

    This function provides a unified interface for creating both single-agent and multi-agent
    navigators using either direct parameter specification or Hydra DictConfig objects.
    Supports automatic detection of single vs multi-agent scenarios based on parameter shapes
    with enhanced performance monitoring and correlation tracking.

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
    
    Enhanced features:
    - Automatic seed management integration for reproducible experiments
    - Performance monitoring with ≤2s initialization requirement
    - Structured JSON logging with correlation ID tracking
    - Request ID and episode ID support for distributed tracing
    """
    # Generate correlation context for operation tracking
    correlation_id = str(uuid.uuid4())
    request_id = kwargs.get('request_id', str(uuid.uuid4()))
    
    if ENHANCED_LOGGING_AVAILABLE:
        context_data = {
            "operation": "create_navigator",
            "correlation_id": correlation_id,
            "request_id": request_id,
            "cfg_provided": cfg is not None,
            "direct_params_provided": any([
                position is not None, positions is not None,
                orientation is not None, orientations is not None
            ])
        }
        
        with correlation_context("create_navigator", **context_data):
            func_logger = logger.bind(**context_data)
    else:
        func_logger = logger.bind(
            module=__name__,
            function="create_navigator",
            correlation_id=correlation_id,
            cfg_provided=cfg is not None,
            direct_params_provided=any([
                position is not None, positions is not None,
                orientation is not None, orientations is not None
            ])
        )

    # Performance monitoring for navigator creation
    with performance_monitor("navigator_creation", threshold=2.0) as perf_metrics:
        try:
            # Initialize seed management if available
            if SEED_UTILS_AVAILABLE:
                seed_context = get_seed_context()
                if seed_context.global_seed is not None:
                    func_logger.info(
                        f"Creating navigator with active seed: {seed_context.global_seed}",
                        extra={"seed": seed_context.global_seed, "correlation_id": correlation_id}
                    )

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
                            "config_source": "hydra" if cfg is not None else "direct",
                            "correlation_id": correlation_id
                        }
                    )
                except Exception as e:
                    func_logger.error(f"Multi-agent configuration validation failed: {e}")
                    raise ValueError(f"Invalid multi-agent configuration: {e}") from e

                # Create multi-agent controller with enhanced seed management
                if SEED_UTILS_AVAILABLE:
                    navigator = MultiAgentController(
                        positions=validated_config.positions,
                        orientations=validated_config.orientations,
                        speeds=validated_config.speeds,
                        max_speeds=validated_config.max_speeds,
                        angular_velocities=validated_config.angular_velocities,
                        config=validated_config,
                        seed_manager=get_seed_context()
                    )
                else:
                    navigator = MultiAgentController(
                        positions=validated_config.positions,
                        orientations=validated_config.orientations,
                        speeds=validated_config.speeds,
                        max_speeds=validated_config.max_speeds,
                        angular_velocities=validated_config.angular_velocities,
                        config=validated_config
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
                            "config_source": "hydra" if cfg is not None else "direct",
                            "correlation_id": correlation_id
                        }
                    )
                except Exception as e:
                    func_logger.error(f"Single-agent configuration validation failed: {e}")
                    raise ValueError(f"Invalid single-agent configuration: {e}") from e

                # Create single-agent controller with enhanced seed management
                if SEED_UTILS_AVAILABLE:
                    navigator = SingleAgentController(
                        position=validated_config.position,
                        orientation=validated_config.orientation,
                        speed=validated_config.speed,
                        max_speed=validated_config.max_speed,
                        angular_velocity=validated_config.angular_velocity,
                        config=validated_config,
                        seed_manager=get_seed_context()
                    )
                else:
                    navigator = SingleAgentController(
                        position=validated_config.position,
                        orientation=validated_config.orientation,
                        speed=validated_config.speed,
                        max_speed=validated_config.max_speed,
                        angular_velocity=validated_config.angular_velocity,
                        config=validated_config
                    )

            func_logger.info(
                f"Navigator created successfully",
                extra={
                    "navigator_type": "multi-agent" if is_multi_agent else "single-agent",
                    "num_agents": navigator.num_agents,
                    "correlation_id": correlation_id,
                    "performance_duration": perf_metrics.duration if hasattr(perf_metrics, 'duration') else None
                }
            )
            
            return navigator

        except Exception as e:
            func_logger.error(f"Navigator creation failed: {e}", extra={"correlation_id": correlation_id})
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
    Create a VideoPlume instance with enhanced performance monitoring and structured logging.

    This function provides a unified interface for creating video-based odor plume environments
    using either direct parameter specification or Hydra DictConfig objects. Supports
    comprehensive video preprocessing options and automatic parameter validation with
    performance monitoring and correlation tracking.

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
    
    Enhanced features:
    - Performance monitoring with <2s initialization requirement
    - Structured JSON logging with correlation tracking
    - Automatic seed management integration for deterministic processing
    """
    # Generate correlation context for operation tracking
    correlation_id = str(uuid.uuid4())
    request_id = kwargs.get('request_id', str(uuid.uuid4()))
    
    if ENHANCED_LOGGING_AVAILABLE:
        context_data = {
            "operation": "create_video_plume",
            "correlation_id": correlation_id,
            "request_id": request_id,
            "cfg_provided": cfg is not None,
            "video_path_provided": video_path is not None
        }
        
        with correlation_context("create_video_plume", **context_data):
            func_logger = logger.bind(**context_data)
    else:
        func_logger = logger.bind(
            module=__name__,
            function="create_video_plume",
            correlation_id=correlation_id,
            cfg_provided=cfg is not None,
            video_path_provided=video_path is not None
        )

    # Performance monitoring for video plume creation
    with performance_monitor("video_plume_creation", threshold=2.0) as perf_metrics:
        try:
            # Initialize seed management if available
            if SEED_UTILS_AVAILABLE:
                seed_context = get_seed_context()
                if seed_context.global_seed is not None:
                    func_logger.info(
                        f"Creating video plume with active seed: {seed_context.global_seed}",
                        extra={"seed": seed_context.global_seed, "correlation_id": correlation_id}
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
                        "config_source": "hydra" if cfg is not None else "direct",
                        "correlation_id": correlation_id
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
                    "duration": plume.duration,
                    "correlation_id": correlation_id,
                    "performance_duration": perf_metrics.duration if hasattr(perf_metrics, 'duration') else None
                }
            )

            return plume

        except Exception as e:
            func_logger.error(f"VideoPlume creation failed: {e}", extra={"correlation_id": correlation_id})
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
    Execute a complete plume navigation simulation with enhanced performance monitoring.

    This function orchestrates frame-by-frame agent navigation through video-based odor
    plume environments with comprehensive data collection and performance monitoring.
    Supports both single-agent and multi-agent scenarios with automatic trajectory recording
    and ≤10ms step() performance monitoring per requirements.

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
    - Step() performance monitoring with ≤10ms threshold warnings per requirements

    Configuration precedence order:
    1. Direct parameters (num_steps, dt, etc.)
    2. Hydra DictConfig object values
    3. Default values from SimulationConfig schema

    Backward compatibility:
    - step_size parameter is converted to dt for legacy compatibility
    - Original parameter names are preserved where possible
    
    Enhanced features:
    - Structured JSON logging with correlation ID tracking
    - Automatic performance threshold monitoring and alerting
    - Episode ID support for RL environment integration
    """
    # Generate correlation context for operation tracking
    correlation_id = str(uuid.uuid4())
    episode_id = kwargs.get('episode_id', f"episode_{int(time.time())}")
    request_id = kwargs.get('request_id', str(uuid.uuid4()))
    
    if ENHANCED_LOGGING_AVAILABLE:
        context_data = {
            "operation": "run_plume_simulation",
            "correlation_id": correlation_id,
            "episode_id": episode_id,
            "request_id": request_id,
            "navigator_type": type(navigator).__name__,
            "num_agents": navigator.num_agents,
            "video_frames": video_plume.frame_count,
            "cfg_provided": cfg is not None
        }
        
        with correlation_context("run_plume_simulation", **context_data):
            sim_logger = logger.bind(**context_data)
    else:
        sim_logger = logger.bind(
            module=__name__,
            function="run_plume_simulation",
            correlation_id=correlation_id,
            navigator_type=type(navigator).__name__,
            num_agents=navigator.num_agents,
            video_frames=video_plume.frame_count,
            cfg_provided=cfg is not None
        )

    # Performance monitoring for entire simulation
    with performance_monitor("simulation_execution") as perf_metrics:
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

            # Initialize seed management if available
            if SEED_UTILS_AVAILABLE:
                seed_context = get_seed_context()
                if seed_context.global_seed is not None:
                    sim_logger.info(
                        f"Running simulation with active seed: {seed_context.global_seed}",
                        extra={"seed": seed_context.global_seed, "correlation_id": correlation_id}
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
                        "num_steps": validated_config.max_steps,
                        "dt": validated_config.step_size,
                        "sensor_distance": merged_params.get("sensor_distance", 5.0),
                        "sensor_angle": merged_params.get("sensor_angle", 45.0),
                        "record_trajectories": merged_params.get("record_trajectories", True),
                        "config_source": "hydra" if cfg is not None else "direct",
                        "correlation_id": correlation_id
                    }
                )
            except Exception as e:
                sim_logger.error(f"Simulation configuration validation failed: {e}")
                raise ValueError(f"Invalid simulation configuration: {e}") from e

            # Get simulation parameters
            num_steps = merged_params.get("num_steps", validated_config.max_steps)
            dt = merged_params.get("dt", validated_config.step_size)
            num_agents = navigator.num_agents

            # Initialize trajectory storage if recording enabled
            record_trajectories = merged_params.get("record_trajectories", True)
            if record_trajectories:
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
                    "memory_usage": f"{positions_history.nbytes + orientations_history.nbytes + odor_readings.nbytes:.1f} bytes",
                    "correlation_id": correlation_id
                }
            )

            # Execute simulation loop with enhanced step performance monitoring
            step_violations = 0
            total_step_time = 0.0
            
            for step in range(num_steps):
                # Performance monitoring for individual step
                step_start_time = time.perf_counter()
                
                try:
                    # Get current frame with bounds checking
                    frame_idx = min(step + 1, video_plume.frame_count - 1)
                    current_frame = video_plume.get_frame(frame_idx)
                    
                    # Update navigator state
                    navigator.step(current_frame)
                    
                    # Measure step performance
                    step_duration = time.perf_counter() - step_start_time
                    total_step_time += step_duration
                    
                    # Check step performance threshold (≤10ms requirement)
                    if step_duration > STEP_PERFORMANCE_THRESHOLD:
                        step_violations += 1
                        sim_logger.warning(
                            f"Step performance violation detected",
                            extra={
                                "step": step,
                                "duration": step_duration,
                                "threshold": STEP_PERFORMANCE_THRESHOLD,
                                "metric_type": "step_performance_violation",
                                "correlation_id": correlation_id
                            }
                        )
                    
                    # Record trajectory data if enabled
                    if record_trajectories:
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
                        avg_step_time = total_step_time / (step + 1)
                        sim_logger.debug(
                            f"Simulation progress: {progress:.1f}% ({step + 1}/{num_steps} steps)",
                            extra={
                                "avg_step_time": avg_step_time,
                                "violations": step_violations,
                                "correlation_id": correlation_id
                            }
                        )

                except Exception as e:
                    sim_logger.error(f"Simulation failed at step {step}: {e}")
                    raise RuntimeError(f"Simulation execution failed at step {step}: {e}") from e

            # Handle non-recording case by storing final state
            if not record_trajectories:
                positions_history[:, 0] = navigator.positions
                orientations_history[:, 0] = navigator.orientations
                if hasattr(navigator, 'sample_odor'):
                    final_frame = video_plume.get_frame(video_plume.frame_count - 1)
                    readings = navigator.sample_odor(final_frame)
                    if isinstance(readings, (int, float)):
                        odor_readings[:, 0] = readings
                    else:
                        odor_readings[:, 0] = readings

            # Calculate performance metrics
            avg_step_time = total_step_time / num_steps if num_steps > 0 else 0.0
            violation_rate = step_violations / num_steps if num_steps > 0 else 0.0

            sim_logger.info(
                "Simulation completed successfully",
                extra={
                    "steps_executed": num_steps,
                    "avg_step_time": avg_step_time,
                    "step_violations": step_violations,
                    "violation_rate": violation_rate,
                    "final_positions": positions_history[:, -1, :].tolist() if record_trajectories else positions_history[:, 0, :].tolist(),
                    "trajectory_recorded": record_trajectories,
                    "data_shape": {
                        "positions": positions_history.shape,
                        "orientations": orientations_history.shape,
                        "readings": odor_readings.shape
                    },
                    "correlation_id": correlation_id,
                    "performance_duration": perf_metrics.duration if hasattr(perf_metrics, 'duration') else None
                }
            )

            return positions_history, orientations_history, odor_readings

        except Exception as e:
            sim_logger.error(f"Simulation execution failed: {e}", extra={"correlation_id": correlation_id})
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
    Visualize simulation results with enhanced logging and performance monitoring.

    This function provides publication-quality visualization of agent trajectories
    and environmental data with support for both static plots and animated sequences.
    Integrates with Hydra configuration for consistent visualization parameters and
    enhanced correlation tracking.

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
    # Generate correlation context for operation tracking
    correlation_id = str(uuid.uuid4())
    request_id = kwargs.get('request_id', str(uuid.uuid4()))
    
    if ENHANCED_LOGGING_AVAILABLE:
        context_data = {
            "operation": "visualize_plume_simulation",
            "correlation_id": correlation_id,
            "request_id": request_id,
            "num_agents": positions.shape[0],
            "num_steps": positions.shape[1],
            "animation": animation
        }
        
        with correlation_context("visualize_plume_simulation", **context_data):
            viz_logger = logger.bind(**context_data)
    else:
        viz_logger = logger.bind(
            module=__name__,
            function="visualize_plume_simulation",
            correlation_id=correlation_id,
            num_agents=positions.shape[0],
            num_steps=positions.shape[1],
            animation=animation
        )

    # Performance monitoring for visualization
    with performance_monitor("visualization_creation") as perf_metrics:
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

            viz_logger.info(
                "Starting visualization creation",
                extra={
                    "animation": animation,
                    "output_path": str(output_path) if output_path else None,
                    "correlation_id": correlation_id
                }
            )

            # Select appropriate visualization function
            if animation:
                result = visualize_simulation_results(
                    positions=positions,
                    orientations=orientations,
                    odor_readings=odor_readings,
                    plume_frames=plume_frames,
                    **viz_params
                )
            else:
                result = visualize_trajectory(
                    positions=positions,
                    orientations=orientations,
                    plume_frames=plume_frames,
                    **viz_params
                )

            viz_logger.info(
                "Visualization completed successfully",
                extra={
                    "output_path": str(output_path) if output_path else None,
                    "correlation_id": correlation_id,
                    "performance_duration": perf_metrics.duration if hasattr(perf_metrics, 'duration') else None
                }
            )

            return result

        except Exception as e:
            viz_logger.error(f"Visualization failed: {e}", extra={"correlation_id": correlation_id})
            raise RuntimeError(f"Failed to create visualization: {e}") from e


def create_gymnasium_environment(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    environment_id: str = "PlumeNavSim-v0",  # Default to new Gymnasium 0.29.x compliant environment
    video_path: Optional[Union[str, pathlib.Path]] = None,
    initial_position: Optional[Tuple[float, float]] = None,
    initial_orientation: float = 0.0,
    max_speed: float = 2.0,
    max_angular_velocity: float = 90.0,
    include_multi_sensor: bool = False,
    num_sensors: int = 2,
    sensor_distance: float = 5.0,
    sensor_layout: str = "bilateral",
    reward_config: Optional[Dict[str, float]] = None,
    max_episode_steps: int = 1000,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    performance_monitoring: bool = True,
    **kwargs: Any
) -> "GymnasiumEnv":
    """
    Create enhanced Gymnasium-compliant environment with PlumeNavSim-v0 support and dual API compatibility.

    This factory function serves as the primary entry point for creating RL-ready environments
    that integrate seamlessly with stable-baselines3 and other modern reinforcement learning
    frameworks. Enhanced with PlumeNavSim-v0 environment registration, compatibility layer
    integration, performance monitoring, and structured logging with correlation tracking.

    The environment wraps the existing plume navigation simulation infrastructure within
    the standard Gymnasium API, enabling researchers to leverage existing RL algorithms
    while maintaining full compatibility with the navigation core and supporting both
    legacy gym and modern Gymnasium APIs automatically.

    Parameters
    ----------
    cfg : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Hydra configuration object or dictionary containing environment parameters.
        Takes precedence over individual parameters if provided, by default None
    environment_id : str, optional
        Environment identifier supporting both legacy and modern formats:
        - "PlumeNavSim-v0": New Gymnasium 0.29.x compliant environment (default)
        - "OdorPlumeNavigation-v1": Legacy gym compatibility environment
        by default "PlumeNavSim-v0"
    video_path : Optional[Union[str, pathlib.Path]], optional
        Path to video file containing odor plume data, by default None
    initial_position : Optional[Tuple[float, float]], optional
        Starting (x, y) position for agent (default: video center), by default None
    initial_orientation : float, optional
        Starting orientation in degrees, by default 0.0
    max_speed : float, optional
        Maximum agent speed in units per time step, by default 2.0
    max_angular_velocity : float, optional
        Maximum angular velocity in degrees/sec, by default 90.0
    include_multi_sensor : bool, optional
        Whether to include multi-sensor observations, by default False
    num_sensors : int, optional
        Number of additional sensors for multi-sensor mode, by default 2
    sensor_distance : float, optional
        Distance from agent center to sensors, by default 5.0
    sensor_layout : str, optional
        Sensor arrangement ("bilateral", "triangular", "custom"), by default "bilateral"
    reward_config : Optional[Dict[str, float]], optional
        Dictionary of reward function weights, by default None
    max_episode_steps : int, optional
        Maximum steps per episode, by default 1000
    render_mode : Optional[str], optional
        Rendering mode ("human", "rgb_array", "headless"), by default None
    seed : Optional[int], optional
        Random seed for reproducible experiments, by default None
    performance_monitoring : bool, optional
        Enable performance tracking with ≤10ms step threshold monitoring, by default True
    **kwargs : Any
        Additional configuration parameters

    Returns
    -------
    GymnasiumEnv
        Configured Gymnasium environment instance ready for RL training with dual API compatibility

    Raises
    ------
    ImportError
        If Gymnasium dependencies are not available
    ValueError
        If configuration parameters are invalid or incomplete
    FileNotFoundError
        If video file does not exist
    RuntimeError
        If environment initialization fails

    Examples
    --------
    Create environment with new PlumeNavSim-v0 ID (default):
        >>> env = create_gymnasium_environment(
        ...     video_path="data/plume_experiment.mp4",
        ...     initial_position=(320, 240),
        ...     max_speed=2.5,
        ...     include_multi_sensor=True,
        ...     render_mode="human"
        ... )

    Create environment with Hydra configuration:
        >>> from hydra import compose, initialize
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="rl_config")
        ...     env = create_gymnasium_environment(cfg.environment)

    Create environment with legacy compatibility:
        >>> env = create_gymnasium_environment(
        ...     environment_id="OdorPlumeNavigation-v1",  # Legacy ID for backward compatibility
        ...     video_path="data/experiment.mp4"
        ... )

    Integration with stable-baselines3:
        >>> from stable_baselines3 import PPO
        >>> env = create_gymnasium_environment(cfg.environment)
        >>> model = PPO("MultiInputPolicy", env, verbose=1)
        >>> model.learn(total_timesteps=100000)

    Vectorized training:
        >>> from stable_baselines3.common.vec_env import DummyVecEnv
        >>> def make_env():
        ...     return create_gymnasium_environment(env_config)
        >>> vec_env = DummyVecEnv([make_env for _ in range(4)])
        >>> model = PPO("MultiInputPolicy", vec_env)

    Notes
    -----
    Configuration precedence order:
    1. Direct parameters (video_path, max_speed, etc.)
    2. Hydra DictConfig object values
    3. Default values from GymnasiumEnv

    Environment compatibility:
    - Full Gymnasium API compliance verified by env_checker
    - Compatible with stable-baselines3 algorithms (PPO, SAC, TD3)
    - Supports vectorized environments for parallel training
    - Integration with Hydra configuration management
    - Automatic API detection and compatibility layer integration

    Performance characteristics:
    - Initialization time: <2s for typical configurations
    - Step execution: ≤10ms threshold monitoring with automatic warnings
    - Memory usage: Scales linearly with episode length
    
    Enhanced features:
    - PlumeNavSim-v0 environment ID support for Gymnasium 0.29.x compliance
    - Automatic compatibility layer integration for dual gym/Gymnasium API support
    - Performance monitoring with threshold-based warnings
    - Structured JSON logging with correlation IDs for distributed tracing
    - Automatic seed management integration for reproducible experiments
    """
    # Generate correlation context for operation tracking
    correlation_id = str(uuid.uuid4())
    request_id = kwargs.get('request_id', str(uuid.uuid4()))
    
    # Check Gymnasium availability
    if not GYMNASIUM_AVAILABLE:
        raise ImportError(
            "Gymnasium environment support is not available. "
            "Install with: pip install 'odor_plume_nav[rl]' to enable RL functionality."
        )

    if ENHANCED_LOGGING_AVAILABLE:
        context_data = {
            "operation": "create_gymnasium_environment",
            "correlation_id": correlation_id,
            "request_id": request_id,
            "environment_id": environment_id,
            "cfg_provided": cfg is not None,
            "video_path_provided": video_path is not None
        }
        
        with correlation_context("create_gymnasium_environment", **context_data):
            func_logger = logger.bind(**context_data)
    else:
        func_logger = logger.bind(
            module=__name__,
            function="create_gymnasium_environment",
            correlation_id=correlation_id,
            environment_id=environment_id,
            cfg_provided=cfg is not None,
            video_path_provided=video_path is not None
        )

    # Performance monitoring for environment creation
    with performance_monitor("gymnasium_environment_creation", threshold=2.0) as perf_metrics:
        try:
            func_logger.info(
                f"Creating Gymnasium environment for RL training",
                extra={
                    "environment_id": environment_id,
                    "performance_monitoring": performance_monitoring,
                    "correlation_id": correlation_id
                }
            )

            # Initialize seed management if available and seed provided
            if SEED_UTILS_AVAILABLE and seed is not None:
                func_logger.info(f"Setting up reproducible environment with seed: {seed}")
                set_global_seed(seed)
                seed_params = get_gymnasium_seed_parameter(seed)
                kwargs.update(seed_params)

            # Process configuration object if provided
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
                "environment_id": environment_id,
                "video_path": video_path or config_params.get("video_path"),
                "initial_position": initial_position or config_params.get("initial_position"),
                "initial_orientation": initial_orientation if initial_orientation != 0.0 else config_params.get("initial_orientation", 0.0),
                "max_speed": max_speed if max_speed != 2.0 else config_params.get("max_speed", 2.0),
                "max_angular_velocity": max_angular_velocity if max_angular_velocity != 90.0 else config_params.get("max_angular_velocity", 90.0),
                "include_multi_sensor": include_multi_sensor if include_multi_sensor else config_params.get("include_multi_sensor", False),
                "num_sensors": num_sensors if num_sensors != 2 else config_params.get("num_sensors", 2),
                "sensor_distance": sensor_distance if sensor_distance != 5.0 else config_params.get("sensor_distance", 5.0),
                "sensor_layout": sensor_layout if sensor_layout != "bilateral" else config_params.get("sensor_layout", "bilateral"),
                "reward_config": reward_config or config_params.get("reward_config"),
                "max_episode_steps": max_episode_steps if max_episode_steps != 1000 else config_params.get("max_episode_steps", 1000),
                "render_mode": render_mode or config_params.get("render_mode"),
                "seed": seed or config_params.get("seed"),
                "performance_monitoring": performance_monitoring if performance_monitoring else config_params.get("performance_monitoring", True),
            }

            # Add any additional config parameters
            merged_params.update({k: v for k, v in config_params.items() 
                                if k not in merged_params})
            merged_params.update(kwargs)

            # Remove None values to use GymnasiumEnv defaults
            constructor_params = {k: v for k, v in merged_params.items() if v is not None}

            # Validate required video_path parameter
            if "video_path" not in constructor_params or constructor_params["video_path"] is None:
                raise ValueError("video_path is required for Gymnasium environment creation")

            # Detect API compatibility requirements
            api_compatibility_mode = None
            if COMPATIBILITY_LAYER_AVAILABLE:
                try:
                    # Detect whether caller expects legacy gym or modern Gymnasium API
                    api_detection = detect_api_version()
                    
                    # Determine compatibility mode based on environment_id and detection
                    if environment_id == "PlumeNavSim-v0" or api_detection.is_legacy == False:
                        api_compatibility_mode = CompatibilityMode(
                            use_legacy_api=False,
                            detection_result=api_detection,
                            performance_monitoring=performance_monitoring,
                            created_at=time.time(),
                            correlation_id=correlation_id
                        )
                    else:
                        api_compatibility_mode = CompatibilityMode(
                            use_legacy_api=True,
                            detection_result=api_detection,
                            performance_monitoring=performance_monitoring,
                            created_at=time.time(),
                            correlation_id=correlation_id
                        )
                    
                    func_logger.info(
                        f"API compatibility mode determined",
                        extra={
                            "use_legacy_api": api_compatibility_mode.use_legacy_api,
                            "confidence": api_detection.confidence,
                            "detection_method": api_detection.detection_method,
                            "correlation_id": correlation_id
                        }
                    )
                    
                    # Add compatibility mode to constructor params
                    constructor_params["_compatibility_mode"] = api_compatibility_mode
                    
                except Exception as e:
                    func_logger.warning(
                        f"API compatibility detection failed, using defaults: {e}",
                        extra={"correlation_id": correlation_id}
                    )

            # Create GymnasiumEnv instance
            env = GymnasiumEnv(**constructor_params)

            # Apply compatibility wrapper if needed
            if COMPATIBILITY_LAYER_AVAILABLE and api_compatibility_mode:
                env = create_compatibility_wrapper(env, api_compatibility_mode)
                func_logger.info(
                    f"Applied compatibility wrapper",
                    extra={
                        "wrapper_type": "legacy" if api_compatibility_mode.use_legacy_api else "modern",
                        "correlation_id": correlation_id
                    }
                )

            func_logger.info(
                "Gymnasium environment created successfully",
                extra={
                    "environment_id": environment_id,
                    "video_path": str(constructor_params["video_path"]),
                    "action_space": str(env.action_space),
                    "observation_space_keys": list(env.observation_space.spaces.keys()) if hasattr(env.observation_space, 'spaces') else "simple",
                    "max_episode_steps": env.max_episode_steps if hasattr(env, 'max_episode_steps') else constructor_params.get("max_episode_steps"),
                    "config_source": "hydra" if cfg is not None else "direct",
                    "api_compatibility": api_compatibility_mode.use_legacy_api if api_compatibility_mode else None,
                    "performance_monitoring": performance_monitoring,
                    "correlation_id": correlation_id,
                    "performance_duration": perf_metrics.duration if hasattr(perf_metrics, 'duration') else None
                }
            )

            return env

        except Exception as e:
            func_logger.error(f"Gymnasium environment creation failed: {e}", extra={"correlation_id": correlation_id})
            raise RuntimeError(f"Failed to create Gymnasium environment: {e}") from e


def from_legacy(
    navigator: NavigatorProtocol,
    video_plume: VideoPlume,
    simulation_config: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    reward_config: Optional[Dict[str, float]] = None,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    **env_kwargs: Any
) -> "GymnasiumEnv":
    """
    Create a Gymnasium environment from existing legacy simulation components with enhanced integration.

    This migration function provides backward compatibility for users transitioning from
    the traditional simulation API to the Gymnasium RL interface. It takes existing
    navigator and video plume instances and wraps them in a Gymnasium-compliant environment
    with enhanced performance monitoring, compatibility layer integration, and structured logging.

    The function serves as a bridge between legacy simulation workflows and modern RL
    training, enabling researchers to leverage their existing configurations while
    gaining access to the standardized RL ecosystem with automatic API compatibility detection.

    Parameters
    ----------
    navigator : NavigatorProtocol
        Existing navigator instance (SingleAgentController or MultiAgentController)
    video_plume : VideoPlume
        Configured VideoPlume environment instance
    simulation_config : Optional[Union[DictConfig, Dict[str, Any]]], optional
        Existing simulation configuration to extract parameters from, by default None
    reward_config : Optional[Dict[str, float]], optional
        Custom reward function weights for RL training, by default None
    max_episode_steps : Optional[int], optional
        Maximum steps per episode (default: derived from simulation config), by default None
    render_mode : Optional[str], optional
        Rendering mode for the Gymnasium environment, by default None
    **env_kwargs : Any
        Additional environment configuration parameters

    Returns
    -------
    GymnasiumEnv
        Gymnasium environment configured with legacy components and compatibility layer

    Raises
    ------
    ImportError
        If Gymnasium dependencies are not available
    TypeError
        If navigator or video_plume don't meet required protocols
    ValueError
        If component configurations are incompatible
    RuntimeError
        If environment creation fails

    Examples
    --------
    Migrate from traditional simulation workflow:
        >>> # Traditional workflow
        >>> navigator = create_navigator(position=(100, 100), max_speed=2.0)
        >>> plume = create_video_plume(video_path="experiment.mp4")
        >>> 
        >>> # Migrate to RL environment
        >>> env = from_legacy(navigator, plume, render_mode="human")
        >>> 
        >>> # Now use with stable-baselines3
        >>> from stable_baselines3 import PPO
        >>> model = PPO("MultiInputPolicy", env)

    Preserve existing configuration:
        >>> # With existing simulation config
        >>> sim_config = {"max_steps": 2000, "dt": 0.1}
        >>> env = from_legacy(
        ...     navigator, plume, 
        ...     simulation_config=sim_config,
        ...     reward_config={"odor_concentration": 2.0}
        ... )

    Multi-agent migration:
        >>> # Multi-agent navigator
        >>> multi_navigator = create_navigator(
        ...     positions=[(50, 50), (150, 150)],
        ...     max_speeds=[2.0, 3.0]
        ... )
        >>> env = from_legacy(multi_navigator, plume)
        >>> # Note: Results in vectorized single-agent envs for RL compatibility

    Custom reward configuration:
        >>> custom_rewards = {
        ...     "odor_concentration": 1.5,
        ...     "distance_penalty": -0.02,
        ...     "exploration_bonus": 0.15
        ... }
        >>> env = from_legacy(navigator, plume, reward_config=custom_rewards)

    Notes
    -----
    Migration considerations:
    - Single-agent navigators create single Gymnasium environments
    - Multi-agent navigators are converted to vectorized single-agent environments
    - Existing VideoPlume preprocessing is preserved
    - Navigator max_speed and position constraints are maintained
    - Simulation timestep (dt) is normalized to 1.0 for RL compatibility

    Configuration extraction:
    - Navigator position and orientation become initial_position/initial_orientation
    - Navigator max_speed becomes environment max_speed constraint
    - VideoPlume video_path and preprocessing settings are preserved
    - Simulation max_steps maps to max_episode_steps

    Performance optimization:
    - Environment initialization reuses existing component configurations
    - No additional video file loading or navigator re-initialization
    - Minimal overhead for component wrapping
    - Automatic compatibility layer integration based on usage context
    
    Enhanced features:
    - Automatic API compatibility detection and wrapper application
    - Performance monitoring with ≤10ms step threshold
    - Structured JSON logging with correlation tracking
    - Seed management integration for reproducible experiments
    """
    # Generate correlation context for operation tracking
    correlation_id = str(uuid.uuid4())
    request_id = env_kwargs.get('request_id', str(uuid.uuid4()))
    
    # Check Gymnasium availability
    if not GYMNASIUM_AVAILABLE:
        raise ImportError(
            "Gymnasium environment support is not available. "
            "Install with: pip install 'odor_plume_nav[rl]' to enable RL functionality."
        )

    if ENHANCED_LOGGING_AVAILABLE:
        context_data = {
            "operation": "from_legacy",
            "correlation_id": correlation_id,
            "request_id": request_id,
            "navigator_type": type(navigator).__name__,
            "num_agents": navigator.num_agents,
            "video_path": str(video_plume.video_path) if hasattr(video_plume, 'video_path') else "unknown"
        }
        
        with correlation_context("from_legacy", **context_data):
            func_logger = logger.bind(**context_data)
    else:
        func_logger = logger.bind(
            module=__name__,
            function="from_legacy",
            correlation_id=correlation_id,
            navigator_type=type(navigator).__name__,
            num_agents=navigator.num_agents,
            video_path=str(video_plume.video_path) if hasattr(video_plume, 'video_path') else "unknown"
        )

    # Performance monitoring for legacy migration
    with performance_monitor("legacy_migration", threshold=2.0) as perf_metrics:
        try:
            func_logger.info(
                "Migrating legacy simulation components to Gymnasium environment",
                extra={"correlation_id": correlation_id}
            )

            # Validate input components
            if not hasattr(navigator, 'positions') or not hasattr(navigator, 'step'):
                raise TypeError("navigator must implement NavigatorProtocol interface")
            
            if not hasattr(video_plume, 'get_frame') or not hasattr(video_plume, 'video_path'):
                raise TypeError("video_plume must be a VideoPlume instance")

            # Extract configuration from simulation_config if provided
            config_params = {}
            if simulation_config is not None:
                try:
                    if HYDRA_AVAILABLE and isinstance(simulation_config, DictConfig):
                        config_params = OmegaConf.to_container(simulation_config, resolve=True)
                    elif isinstance(simulation_config, dict):
                        config_params = simulation_config.copy()
                    else:
                        raise TypeError(f"simulation_config must be DictConfig or dict, got {type(simulation_config)}")
                except Exception as e:
                    func_logger.error(f"Failed to process simulation configuration: {e}")
                    raise ValueError(f"Invalid simulation configuration: {e}") from e

            # Extract navigator configuration
            # For single agents, use first agent's parameters
            if navigator.num_agents == 1:
                initial_position = tuple(navigator.positions[0])
                initial_orientation = float(navigator.orientations[0])
                max_speed = float(navigator.max_speeds[0])
            else:
                # For multi-agent, use first agent as template and warn
                func_logger.warning(
                    f"Multi-agent navigator with {navigator.num_agents} agents detected. "
                    "Creating single-agent Gymnasium environment using first agent's parameters. "
                    "Consider using vectorized environments for true multi-agent RL training.",
                    extra={"correlation_id": correlation_id}
                )
                initial_position = tuple(navigator.positions[0])
                initial_orientation = float(navigator.orientations[0])
                max_speed = float(navigator.max_speeds[0])

            # Extract video plume configuration
            video_path = video_plume.video_path

            # Determine episode length from simulation config or use default
            if max_episode_steps is None:
                max_episode_steps = (
                    config_params.get("max_steps", 
                    config_params.get("num_steps", 1000))
                )

            # Build environment parameters
            env_params = {
                "environment_id": "PlumeNavSim-v0",  # Use new Gymnasium compliant environment ID
                "video_path": video_path,
                "initial_position": initial_position,
                "initial_orientation": initial_orientation,
                "max_speed": max_speed,
                "max_angular_velocity": 90.0,  # Default, can be overridden
                "reward_config": reward_config,
                "max_episode_steps": max_episode_steps,
                "render_mode": render_mode,
                "performance_monitoring": True
            }

            # Apply any additional overrides
            env_params.update(env_kwargs)

            # Create Gymnasium environment using the enhanced factory function
            env = create_gymnasium_environment(**env_params)

            func_logger.info(
                "Legacy migration completed successfully",
                extra={
                    "source_navigator": type(navigator).__name__,
                    "source_agents": navigator.num_agents,
                    "video_path": str(video_path),
                    "environment_config": {
                        "initial_position": initial_position,
                        "max_speed": max_speed,
                        "max_episode_steps": max_episode_steps
                    },
                    "correlation_id": correlation_id,
                    "performance_duration": perf_metrics.duration if hasattr(perf_metrics, 'duration') else None
                }
            )

            return env

        except Exception as e:
            func_logger.error(f"Legacy migration failed: {e}", extra={"correlation_id": correlation_id})
            raise RuntimeError(f"Failed to migrate legacy components to Gymnasium environment: {e}") from e


# Legacy compatibility aliases
create_navigator_from_config = create_navigator
create_video_plume_from_config = create_video_plume
run_simulation = run_plume_simulation
visualize_simulation_results = visualize_plume_simulation

# Gymnasium environment aliases for compatibility
create_rl_environment = create_gymnasium_environment
migrate_to_rl = from_legacy

# Export public API
__all__ = [
    "create_navigator",
    "create_video_plume", 
    "run_plume_simulation",
    "visualize_plume_simulation",
    # Gymnasium environment functions
    "create_gymnasium_environment",
    "from_legacy",
    # Legacy aliases
    "create_navigator_from_config",
    "create_video_plume_from_config", 
    "run_simulation",
    "visualize_simulation_results",
    # Gymnasium environment aliases
    "create_rl_environment",
    "migrate_to_rl"
]