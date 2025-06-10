"""
Main Gymnasium environment wrapper for odor plume navigation simulation.

This module provides the primary GymnasiumEnv class that implements the gymnasium.Env 
interface to expose the existing plume_nav_sim simulation framework via standardized 
reset/step/render/close methods. This class serves as the critical bridge between the 
existing NavigatorProtocol, VideoPlume components and the Gymnasium RL ecosystem, 
enabling seamless integration with stable-baselines3 algorithms and the broader RL 
research community.

The implementation maintains full compatibility with existing simulation infrastructure 
while providing the standardized RL interface required for modern reinforcement learning 
workflows. It supports both single-agent and multi-agent configurations, configurable 
reward functions, and performance-optimized execution targeting ≥30 FPS simulation rates.

Key Features:
- Full Gymnasium API compliance with env_checker validation
- Integration with existing NavigatorProtocol and VideoPlume components  
- Configurable action and observation spaces supporting multi-modal sensing
- Domain-specific reward functions for olfactory navigation research
- Vectorized environment support for parallel training workflows
- Comprehensive seed management for reproducible experiments
- Real-time rendering and headless export capabilities

Technical Architecture:
- Wraps existing simulation components without modification
- Maps Gymnasium actions to NavigatorProtocol control inputs
- Converts navigator state to standardized observation dictionaries
- Implements configurable reward computation with domain expertise
- Maintains performance requirements through optimized execution paths
"""

from __future__ import annotations
import time
import warnings
from typing import Dict, Tuple, Optional, Any, Union, List, SupportsFloat
from pathlib import Path
import numpy as np

# Gymnasium and RL framework imports
try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Dict as DictSpace
    from gymnasium.error import DependencyNotInstalled
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    # Create mock classes to prevent import errors
    class gym:
        class Env:
            def __init__(self): pass
    Box = DictSpace = None

# Core plume navigation imports
from odor_plume_nav.core.protocols import NavigatorProtocol, NavigatorFactory
from odor_plume_nav.environments.video_plume import VideoPlume
from odor_plume_nav.environments.spaces import (
    ActionSpace, ObservationSpace, SpaceFactory,
    ActionType, ObservationType
)
from odor_plume_nav.utils.seed_manager import (
    set_global_seed, get_seed_context, scoped_seed, SeedConfig
)

# Hydra configuration integration
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict

# Enhanced logging and performance monitoring
try:
    from odor_plume_nav.utils.logging_setup import get_enhanced_logger, PerformanceMetrics
    logger = get_enhanced_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Optional imports for advanced features
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Type aliases for enhanced readability
ConfigType = Union[DictConfig, Dict[str, Any]]
InfoType = Dict[str, Any]
RenderModeType = Optional[str]
SeedType = Optional[int]


class GymnasiumEnv(gym.Env):
    """
    Gymnasium-compliant environment wrapper for odor plume navigation simulation.
    
    This class implements the standard Gymnasium interface while integrating seamlessly 
    with the existing plume navigation simulation infrastructure. It exposes the 
    NavigatorProtocol and VideoPlume components through standardized RL methods,
    enabling direct compatibility with stable-baselines3 and other modern RL frameworks.
    
    The environment supports configurable action/observation spaces, domain-specific 
    reward functions, and performance-optimized execution for research workflows.
    Multi-agent support is provided through vectorized environment patterns.
    
    Key Design Principles:
    - Zero-modification integration with existing simulation components
    - Performance-first implementation targeting ≥30 FPS execution
    - Configurable spaces supporting both single and multi-sensor configurations  
    - Domain-expert reward functions for olfactory navigation research
    - Comprehensive seed management for reproducible experiments
    - Flexible rendering supporting both real-time and headless modes
    
    Examples:
        Basic environment creation:
        >>> env = GymnasiumEnv(
        ...     video_path="plume_movie.mp4",
        ...     initial_position=(320, 240),
        ...     max_speed=2.0
        ... )
        >>> obs, info = env.reset()
        >>> action = env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = env.step(action)
        
        Configuration-driven setup:
        >>> config = {
        ...     "video_path": "experiments/plume_data.mp4",
        ...     "navigator": {"max_speed": 3.0, "position": [100, 200]},
        ...     "spaces": {"include_multi_sensor": True, "num_sensors": 3},
        ...     "reward": {"odor_weight": 1.0, "distance_weight": -0.1}
        ... }
        >>> env = GymnasiumEnv.from_config(config)
        
        Integration with stable-baselines3:
        >>> from stable_baselines3 import PPO
        >>> env = GymnasiumEnv.from_config(training_config)
        >>> model = PPO("MultiInputPolicy", env, verbose=1)
        >>> model.learn(total_timesteps=100000)
    """
    
    # Gymnasium metadata for environment registration
    metadata = {
        "render_modes": ["human", "rgb_array", "headless"],
        "render_fps": 30,
        "spec_id": "OdorPlumeNavigation-v1",
        "max_episode_steps": 1000,
        "reward_threshold": 100.0,
        "nondeterministic": False,
        "author": "Blitzy Platform",
        "environment_type": "continuous_control",
        "action_type": "continuous",
        "observation_type": "multi_modal_dict"
    }
    
    def __init__(
        self,
        video_path: Union[str, Path],
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
        **kwargs
    ):
        """
        Initialize Gymnasium environment wrapper with comprehensive configuration.
        
        Args:
            video_path: Path to video file containing odor plume data
            initial_position: Starting (x, y) position for agent (default: video center)
            initial_orientation: Starting orientation in degrees (default: 0.0)
            max_speed: Maximum agent speed in units per time step (default: 2.0)
            max_angular_velocity: Maximum angular velocity in degrees/sec (default: 90.0)
            include_multi_sensor: Whether to include multi-sensor observations (default: False)
            num_sensors: Number of additional sensors for multi-sensor mode (default: 2)
            sensor_distance: Distance from agent center to sensors (default: 5.0)
            sensor_layout: Sensor arrangement ("bilateral", "triangular", "custom") (default: "bilateral")
            reward_config: Dictionary of reward function weights (default: standard weights)
            max_episode_steps: Maximum steps per episode (default: 1000)
            render_mode: Rendering mode ("human", "rgb_array", "headless") (default: None)
            seed: Random seed for reproducible experiments (default: None)
            performance_monitoring: Enable performance tracking (default: True)
            **kwargs: Additional configuration parameters
            
        Raises:
            ImportError: If gymnasium is not available
            ValueError: If configuration parameters are invalid
            FileNotFoundError: If video file does not exist
            RuntimeError: If environment initialization fails
            
        Note:
            The environment automatically configures action and observation spaces based
            on the provided parameters. Video dimensions are extracted automatically
            from the provided video file for space configuration.
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for GymnasiumEnv. "
                "Install with: pip install 'odor_plume_nav[rl]'"
            )
        
        super().__init__()
        
        # Store configuration parameters
        self.video_path = Path(video_path)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.performance_monitoring = performance_monitoring
        
        # Validate video file existence
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Initialize performance tracking
        self._step_count = 0
        self._episode_count = 0
        self._total_reward = 0.0
        self._start_time = time.time()
        self._step_times: List[float] = []
        
        logger.info(f"Initializing GymnasiumEnv with video: {self.video_path}")
        
        try:
            # Initialize video plume environment
            self._init_video_plume()
            
            # Configure reward function parameters
            self._init_reward_config(reward_config)
            
            # Initialize navigator with specified parameters
            self._init_navigator(
                initial_position, initial_orientation, 
                max_speed, max_angular_velocity
            )
            
            # Configure action and observation spaces
            self._init_spaces(
                include_multi_sensor, num_sensors, 
                sensor_distance, sensor_layout
            )
            
            # Set up rendering system
            self._init_rendering()
            
            # Apply seed if provided
            if seed is not None:
                self.seed(seed)
            
            # Initialize episode state
            self._reset_episode_state()
            
            logger.info(
                f"GymnasiumEnv initialized successfully",
                extra={
                    "video_dims": f"{self.video_width}x{self.video_height}",
                    "action_space": str(self.action_space),
                    "obs_space_keys": list(self.observation_space.spaces.keys()),
                    "max_episode_steps": self.max_episode_steps
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize GymnasiumEnv: {e}")
            raise RuntimeError(f"Environment initialization failed: {e}") from e
    
    def _init_video_plume(self) -> None:
        """Initialize VideoPlume environment processor."""
        try:
            self.video_plume = VideoPlume(str(self.video_path))
            
            # Extract video metadata for space configuration
            metadata = self.video_plume.get_metadata()
            self.video_width = metadata['width']
            self.video_height = metadata['height']
            self.video_fps = metadata['fps']
            self.video_frame_count = metadata['frame_count']
            
            logger.debug(
                f"VideoPlume initialized: {self.video_width}x{self.video_height}, "
                f"{self.video_frame_count} frames at {self.video_fps:.1f} fps"
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VideoPlume: {e}") from e
    
    def _init_reward_config(self, reward_config: Optional[Dict[str, float]]) -> None:
        """Initialize reward function configuration with domain-specific defaults."""
        # Default reward weights based on olfactory navigation research
        self.reward_weights = {
            "odor_concentration": 1.0,      # Primary reward for finding odor
            "distance_penalty": -0.01,      # Small penalty for distance from source
            "control_effort": -0.005,       # Penalty for excessive control actions
            "boundary_penalty": -1.0,       # Penalty for hitting boundaries
            "time_penalty": -0.001,         # Small time penalty to encourage efficiency
            "exploration_bonus": 0.1,       # Bonus for exploring new areas
            "gradient_following": 0.5       # Bonus for following odor gradients
        }
        
        # Override with user-provided weights
        if reward_config:
            self.reward_weights.update(reward_config)
        
        # Initialize reward computation state
        self._previous_odor = 0.0
        self._previous_position = None
        self._visited_positions: List[Tuple[float, float]] = []
        self._exploration_grid = np.zeros((50, 50))  # Coarse grid for exploration tracking
        
        logger.debug(f"Reward configuration: {self.reward_weights}")
    
    def _init_navigator(
        self, 
        initial_position: Optional[Tuple[float, float]],
        initial_orientation: float,
        max_speed: float,
        max_angular_velocity: float
    ) -> None:
        """Initialize navigator with specified parameters."""
        # Use video center as default position if not specified
        if initial_position is None:
            initial_position = (self.video_width / 2, self.video_height / 2)
        
        # Validate position bounds
        x, y = initial_position
        if not (0 <= x <= self.video_width and 0 <= y <= self.video_height):
            logger.warning(
                f"Initial position {initial_position} outside video bounds "
                f"({self.video_width}x{self.video_height}), clipping to bounds"
            )
            x = np.clip(x, 0, self.video_width)
            y = np.clip(y, 0, self.video_height)
            initial_position = (x, y)
        
        # Store initial configuration for reset operations
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.max_speed = max_speed
        self.max_angular_velocity = max_angular_velocity
        
        # Create navigator using factory method
        try:
            self.navigator = NavigatorFactory.single_agent(
                position=initial_position,
                orientation=initial_orientation,
                speed=0.0,  # Start stationary
                max_speed=max_speed,
                angular_velocity=0.0  # Start with no rotation
            )
            
            logger.debug(
                f"Navigator initialized at {initial_position} with "
                f"max_speed={max_speed}, max_angular_velocity={max_angular_velocity}"
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize navigator: {e}") from e
    
    def _init_spaces(
        self,
        include_multi_sensor: bool,
        num_sensors: int,
        sensor_distance: float,
        sensor_layout: str
    ) -> None:
        """Initialize action and observation spaces using space definitions."""
        try:
            # Create action space for continuous control
            self.action_space = ActionSpace.create(
                max_speed=self.max_speed,
                max_angular_velocity=self.max_angular_velocity,
                min_speed=0.0,
                dtype=np.float32
            )
            
            # Create observation space with multi-sensor support
            self.observation_space = ObservationSpace.create(
                env_width=self.video_width,
                env_height=self.video_height,
                num_sensors=num_sensors,
                include_multi_sensor=include_multi_sensor,
                dtype=np.float32
            )
            
            # Store sensor configuration for observation generation
            self.include_multi_sensor = include_multi_sensor
            self.num_sensors = num_sensors
            self.sensor_distance = sensor_distance
            self.sensor_layout = sensor_layout
            
            logger.debug(
                f"Spaces initialized - Action: {self.action_space}, "
                f"Observation keys: {list(self.observation_space.spaces.keys())}"
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize spaces: {e}") from e
    
    def _init_rendering(self) -> None:
        """Initialize rendering system for visualization."""
        self.render_initialized = False
        self.fig = None
        self.ax = None
        self.agent_plot = None
        self.trajectory_plot = None
        self.render_trajectory: List[Tuple[float, float]] = []
        
        if self.render_mode == "human" and not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "Matplotlib not available, falling back to headless rendering"
            )
            self.render_mode = "headless"
    
    def _reset_episode_state(self) -> None:
        """Reset episode-specific state variables."""
        self._step_count = 0
        self._total_reward = 0.0
        self._previous_odor = 0.0
        self._previous_position = None
        self._visited_positions.clear()
        self._exploration_grid.fill(0)
        self.render_trajectory.clear()
        
        # Reset navigator to initial state
        self.navigator.reset(
            position=self.initial_position,
            orientation=self.initial_orientation,
            speed=0.0,
            angular_velocity=0.0
        )
    
    @classmethod
    def from_config(
        cls, 
        config: ConfigType,
        **override_kwargs
    ) -> 'GymnasiumEnv':
        """
        Create GymnasiumEnv from Hydra configuration with parameter validation.
        
        Args:
            config: Configuration dictionary or DictConfig containing environment parameters
            **override_kwargs: Additional parameters to override configuration values
            
        Returns:
            Configured GymnasiumEnv instance
            
        Raises:
            ValueError: If configuration is invalid or incomplete
            TypeError: If configuration type is unsupported
            
        Examples:
            >>> config = {
            ...     "video_path": "data/plume.mp4",
            ...     "navigator": {"max_speed": 2.5, "position": [100, 200]},
            ...     "spaces": {"include_multi_sensor": True},
            ...     "reward": {"odor_concentration": 2.0}
            ... }
            >>> env = GymnasiumEnv.from_config(config)
            
            >>> # With Hydra DictConfig
            >>> @hydra.main(config_path="conf", config_name="env_config")
            >>> def main(cfg: DictConfig) -> None:
            ...     env = GymnasiumEnv.from_config(cfg.environment)
        """
        logger.info("Creating GymnasiumEnv from configuration")
        
        # Handle different configuration types
        if isinstance(config, dict):
            config_dict = config.copy()
        elif HYDRA_AVAILABLE and hasattr(config, 'to_container'):
            # Handle Hydra DictConfig
            config_dict = OmegaConf.to_container(config, resolve=True)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")
        
        # Apply override parameters
        config_dict.update(override_kwargs)
        
        # Extract required video path
        if "video_path" not in config_dict:
            raise ValueError("Configuration must include 'video_path'")
        
        # Extract navigator configuration
        nav_config = config_dict.get("navigator", {})
        initial_position = nav_config.get("position", nav_config.get("initial_position"))
        if isinstance(initial_position, list):
            initial_position = tuple(initial_position)
        
        # Extract space configuration  
        space_config = config_dict.get("spaces", {})
        
        # Extract reward configuration
        reward_config = config_dict.get("reward", {})
        
        # Build constructor arguments
        constructor_args = {
            "video_path": config_dict["video_path"],
            "initial_position": initial_position,
            "initial_orientation": nav_config.get("orientation", 0.0),
            "max_speed": nav_config.get("max_speed", 2.0),
            "max_angular_velocity": nav_config.get("max_angular_velocity", 90.0),
            "include_multi_sensor": space_config.get("include_multi_sensor", False),
            "num_sensors": space_config.get("num_sensors", 2),
            "sensor_distance": space_config.get("sensor_distance", 5.0),
            "sensor_layout": space_config.get("sensor_layout", "bilateral"),
            "reward_config": reward_config,
            "max_episode_steps": config_dict.get("max_episode_steps", 1000),
            "render_mode": config_dict.get("render_mode"),
            "seed": config_dict.get("seed"),
            "performance_monitoring": config_dict.get("performance_monitoring", True)
        }
        
        # Remove None values to use defaults
        constructor_args = {k: v for k, v in constructor_args.items() if v is not None}
        
        try:
            env = cls(**constructor_args)
            
            logger.info(
                "GymnasiumEnv created successfully from configuration",
                extra={
                    "video_path": constructor_args["video_path"],
                    "navigator_config": nav_config,
                    "space_config": space_config,
                    "reward_config": reward_config
                }
            )
            
            return env
            
        except Exception as e:
            logger.error(f"Failed to create GymnasiumEnv from config: {e}")
            raise
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObservationType, InfoType]:
        """
        Reset environment to initial state with optional parameter overrides.
        
        Resets the navigator to initial position and orientation, clears episode
        state, and returns the initial observation. Supports optional parameter
        overrides for varying episode initial conditions.
        
        Args:
            seed: Random seed for episode reproducibility (optional)
            options: Dictionary of reset options (optional). Supported keys:
                - position: Override initial position as (x, y) tuple
                - orientation: Override initial orientation in degrees
                - frame_index: Start from specific video frame
                
        Returns:
            Tuple containing:
                - observation: Initial observation dictionary
                - info: Episode metadata and diagnostics
                
        Raises:
            ValueError: If override options are invalid
            RuntimeError: If reset operation fails
            
        Note:
            Reset operations are performance-critical and target <10ms completion
            time for real-time training workflows.
        """
        if self.performance_monitoring:
            reset_start = time.time()
        
        logger.debug(f"Resetting environment (episode {self._episode_count + 1})")
        
        # Handle seeding if provided
        if seed is not None:
            self.seed(seed)
        
        # Process reset options
        reset_position = self.initial_position
        reset_orientation = self.initial_orientation
        start_frame = 0
        
        if options:
            if "position" in options:
                pos = options["position"]
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    reset_position = tuple(pos)
                else:
                    raise ValueError(f"Invalid position format: {pos}")
                    
            if "orientation" in options:
                reset_orientation = float(options["orientation"])
                
            if "frame_index" in options:
                start_frame = int(options["frame_index"])
                if not (0 <= start_frame < self.video_frame_count):
                    raise ValueError(f"Frame index {start_frame} out of range [0, {self.video_frame_count})")
        
        try:
            # Reset episode state
            self._reset_episode_state()
            
            # Apply position override if provided
            if reset_position != self.initial_position:
                self.navigator.reset(
                    position=reset_position,
                    orientation=reset_orientation,
                    speed=0.0,
                    angular_velocity=0.0
                )
            
            # Set current video frame
            self.current_frame_index = start_frame
            
            # Generate initial observation
            observation = self._get_observation()
            
            # Prepare info dictionary
            info = {
                "episode": self._episode_count + 1,
                "step": 0,
                "agent_position": list(self.navigator.positions[0]),
                "agent_orientation": float(self.navigator.orientations[0]),
                "current_frame": self.current_frame_index,
                "video_metadata": self.video_plume.get_metadata(),
                "seed": getattr(self, "_last_seed", None),
                "reset_options": options or {}
            }
            
            # Update episode counter
            self._episode_count += 1
            
            if self.performance_monitoring:
                reset_time = time.time() - reset_start
                info["reset_time"] = reset_time
                
                if reset_time > 0.01:  # 10ms target
                    logger.warning(f"Reset took {reset_time:.3f}s, exceeding 10ms target")
            
            logger.debug(f"Environment reset completed in episode {self._episode_count}")
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            raise RuntimeError(f"Environment reset failed: {e}") from e
    
    def step(
        self, 
        action: ActionType
    ) -> Tuple[ObservationType, SupportsFloat, bool, bool, InfoType]:
        """
        Execute one environment step with the given action.
        
        Applies the provided action to the navigator, advances the simulation by one
        time step, computes the reward based on domain-specific criteria, and
        determines episode termination conditions.
        
        Args:
            action: Action array with shape (2,) containing [speed, angular_velocity]
                   Values are automatically clipped to valid action space bounds
                   
        Returns:
            Tuple containing:
                - observation: Next state observation dictionary
                - reward: Scalar reward for the transition  
                - terminated: Whether episode ended due to success/failure
                - truncated: Whether episode ended due to time/step limits
                - info: Step metadata and diagnostics
                
        Raises:
            ValueError: If action format is invalid
            RuntimeError: If step execution fails
            
        Note:
            Step execution is performance-critical and targets <1ms completion
            time for ≥30 FPS simulation performance.
        """
        if self.performance_monitoring:
            step_start = time.time()
        
        # Validate and clip action
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")
        
        action = ActionSpace.clip_action(action, self.action_space)
        speed, angular_velocity = action
        
        # Store previous state for reward computation
        prev_position = self.navigator.positions[0].copy()
        prev_orientation = self.navigator.orientations[0]
        prev_odor = self._previous_odor
        
        try:
            # Apply action to navigator
            self.navigator.speeds[0] = speed
            self.navigator.angular_velocities[0] = angular_velocity
            
            # Get current environment frame
            current_frame = self.video_plume.get_frame(self.current_frame_index)
            if current_frame is None:
                # Handle end of video by cycling or terminating
                self.current_frame_index = 0
                current_frame = self.video_plume.get_frame(0)
            
            # Execute navigator step
            self.navigator.step(current_frame, dt=1.0)
            
            # Advance frame index for next step
            self.current_frame_index = (self.current_frame_index + 1) % self.video_frame_count
            
            # Get new state observation
            observation = self._get_observation()
            
            # Compute reward based on transition
            reward = self._compute_reward(
                action, prev_position, prev_orientation, prev_odor, observation
            )
            
            # Update step counter and tracking
            self._step_count += 1
            self._total_reward += reward
            
            # Check termination conditions
            terminated, truncated = self._check_termination(observation)
            
            # Update exploration tracking
            self._update_exploration_tracking(self.navigator.positions[0])
            
            # Prepare detailed info dictionary
            info = self._prepare_step_info(
                action, reward, prev_position, prev_orientation, 
                step_start if self.performance_monitoring else None
            )
            
            if self.performance_monitoring:
                step_time = time.time() - step_start
                self._step_times.append(step_time)
                info["step_time"] = step_time
                
                # Warn if step time exceeds performance target
                if step_time > 0.001:  # 1ms target for ≥30 FPS
                    logger.warning(f"Step took {step_time:.3f}s, exceeding 1ms target")
            
            return observation, float(reward), terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            raise RuntimeError(f"Environment step failed: {e}") from e
    
    def _get_observation(self) -> ObservationType:
        """
        Generate current observation dictionary from navigator and environment state.
        
        Returns:
            Observation dictionary containing:
                - odor_concentration: Current odor level at agent position
                - agent_position: Agent coordinates [x, y]
                - agent_orientation: Agent heading in degrees
                - multi_sensor_readings: Optional multi-sensor data (if enabled)
        """
        # Get current environment frame
        current_frame = self.video_plume.get_frame(self.current_frame_index)
        
        # Sample odor concentration at agent position
        odor_concentration = self.navigator.sample_odor(current_frame)
        self._previous_odor = float(odor_concentration)
        
        # Get agent state
        agent_position = self.navigator.positions[0].astype(np.float32)
        agent_orientation = np.float32(self.navigator.orientations[0])
        
        # Build core observation
        observation = {
            "odor_concentration": np.float32(odor_concentration),
            "agent_position": agent_position,
            "agent_orientation": agent_orientation
        }
        
        # Add multi-sensor readings if enabled
        if self.include_multi_sensor:
            # Configure sensor layout
            sensor_kwargs = {
                "sensor_distance": self.sensor_distance,
                "num_sensors": self.num_sensors
            }
            
            if self.sensor_layout == "bilateral":
                sensor_kwargs["layout_name"] = "LEFT_RIGHT"
            elif self.sensor_layout == "triangular":
                sensor_kwargs["layout_name"] = "TRIANGLE"
            elif self.sensor_layout == "forward_back":
                sensor_kwargs["layout_name"] = "FORWARD_BACK"
            else:
                # Custom layout with angular spacing
                sensor_kwargs["sensor_angle"] = 360.0 / self.num_sensors
            
            # Sample multi-sensor readings
            multi_sensor_readings = self.navigator.sample_multiple_sensors(
                current_frame, **sensor_kwargs
            )
            
            # Ensure correct shape for single agent
            if multi_sensor_readings.ndim == 2:
                multi_sensor_readings = multi_sensor_readings[0]
            
            observation["multi_sensor_readings"] = multi_sensor_readings.astype(np.float32)
        
        return observation
    
    def _compute_reward(
        self,
        action: np.ndarray,
        prev_position: np.ndarray, 
        prev_orientation: float,
        prev_odor: float,
        observation: ObservationType
    ) -> float:
        """
        Compute domain-specific reward based on olfactory navigation criteria.
        
        The reward function implements research-validated incentives for chemotaxis
        behavior including odor following, efficient movement, and exploration.
        
        Args:
            action: Applied action [speed, angular_velocity]
            prev_position: Agent position before action
            prev_orientation: Agent orientation before action  
            prev_odor: Odor concentration before action
            observation: Current observation after action
            
        Returns:
            Scalar reward value for the transition
        """
        current_position = observation["agent_position"]
        current_odor = float(observation["odor_concentration"])
        speed, angular_velocity = action
        
        reward = 0.0
        
        # Primary reward: Odor concentration at current position
        odor_reward = current_odor * self.reward_weights["odor_concentration"]
        reward += odor_reward
        
        # Gradient following reward: Positive for increasing odor concentration
        odor_change = current_odor - prev_odor
        gradient_reward = odor_change * self.reward_weights["gradient_following"]
        reward += gradient_reward
        
        # Distance penalty: Encourage staying in high-odor regions
        # Note: This is a simplified penalty; in practice, the true source location
        # would be used if known, or estimated through gradient ascent
        if current_odor > 0:
            # Assume source is at highest odor location visited so far
            if hasattr(self, "_best_odor_position"):
                if current_odor > self._best_odor_value:
                    self._best_odor_position = current_position.copy()
                    self._best_odor_value = current_odor
            else:
                self._best_odor_position = current_position.copy()
                self._best_odor_value = current_odor
            
            if hasattr(self, "_best_odor_position"):
                distance_to_best = np.linalg.norm(current_position - self._best_odor_position)
                distance_penalty = distance_to_best * self.reward_weights["distance_penalty"]
                reward += distance_penalty
        
        # Control effort penalty: Discourage excessive actions
        control_effort = (speed / self.max_speed) ** 2 + (angular_velocity / self.max_angular_velocity) ** 2
        effort_penalty = control_effort * self.reward_weights["control_effort"]
        reward += effort_penalty
        
        # Boundary penalty: Strong negative reward for hitting environment edges
        x, y = current_position
        boundary_margin = 10.0  # Pixels from edge
        if (x < boundary_margin or x > self.video_width - boundary_margin or
            y < boundary_margin or y > self.video_height - boundary_margin):
            boundary_penalty = self.reward_weights["boundary_penalty"]
            reward += boundary_penalty
        
        # Time penalty: Small negative reward to encourage efficiency
        time_penalty = self.reward_weights["time_penalty"]
        reward += time_penalty
        
        # Exploration bonus: Reward for visiting new areas
        exploration_bonus = self._get_exploration_bonus(current_position)
        reward += exploration_bonus
        
        return reward
    
    def _get_exploration_bonus(self, position: np.ndarray) -> float:
        """
        Calculate exploration bonus based on position novelty.
        
        Args:
            position: Current agent position [x, y]
            
        Returns:
            Exploration bonus value
        """
        # Map position to exploration grid
        grid_x = int(position[0] / self.video_width * self._exploration_grid.shape[1])
        grid_y = int(position[1] / self.video_height * self._exploration_grid.shape[0])
        
        # Clamp to grid bounds
        grid_x = np.clip(grid_x, 0, self._exploration_grid.shape[1] - 1)
        grid_y = np.clip(grid_y, 0, self._exploration_grid.shape[0] - 1)
        
        # Check if this cell has been visited
        if self._exploration_grid[grid_y, grid_x] == 0:
            self._exploration_grid[grid_y, grid_x] = 1
            return self.reward_weights["exploration_bonus"]
        
        return 0.0
    
    def _check_termination(self, observation: ObservationType) -> Tuple[bool, bool]:
        """
        Check episode termination conditions.
        
        Args:
            observation: Current observation dictionary
            
        Returns:
            Tuple of (terminated, truncated) booleans
        """
        terminated = False
        truncated = False
        
        # Check step limit (truncation)
        if self._step_count >= self.max_episode_steps:
            truncated = True
        
        # Check boundary termination
        x, y = observation["agent_position"]
        if x < 0 or x > self.video_width or y < 0 or y > self.video_height:
            terminated = True
        
        # Check success condition: High odor concentration for sustained period
        current_odor = float(observation["odor_concentration"])
        if current_odor > 0.8:  # High odor threshold
            if not hasattr(self, "_high_odor_steps"):
                self._high_odor_steps = 0
            self._high_odor_steps += 1
            
            # Success if high odor maintained for 10 steps
            if self._high_odor_steps >= 10:
                terminated = True
        else:
            self._high_odor_steps = 0
        
        return terminated, truncated
    
    def _update_exploration_tracking(self, position: np.ndarray) -> None:
        """Update exploration tracking with current position."""
        self._visited_positions.append(tuple(position))
        self._previous_position = position.copy()
    
    def _prepare_step_info(
        self,
        action: np.ndarray,
        reward: float,
        prev_position: np.ndarray,
        prev_orientation: float,
        step_start_time: Optional[float]
    ) -> InfoType:
        """
        Prepare comprehensive step information dictionary.
        
        Args:
            action: Applied action
            reward: Computed reward
            prev_position: Previous agent position
            prev_orientation: Previous agent orientation
            step_start_time: Step timing start (if performance monitoring enabled)
            
        Returns:
            Info dictionary with step metadata
        """
        current_position = self.navigator.positions[0]
        current_orientation = self.navigator.orientations[0]
        
        info = {
            "step": self._step_count,
            "episode": self._episode_count,
            "total_reward": self._total_reward,
            "action": action.tolist(),
            "reward": reward,
            "agent_position": current_position.tolist(),
            "agent_orientation": float(current_orientation),
            "current_frame": self.current_frame_index,
            "odor_concentration": self._previous_odor,
            "movement_distance": float(np.linalg.norm(current_position - prev_position)),
            "orientation_change": float(abs(current_orientation - prev_orientation)),
            "exploration_cells_visited": int(np.sum(self._exploration_grid))
        }
        
        # Add performance metrics if monitoring enabled
        if self.performance_monitoring and step_start_time is not None:
            step_time = time.time() - step_start_time
            info["performance"] = {
                "step_time": step_time,
                "avg_step_time": np.mean(self._step_times[-100:]) if self._step_times else step_time,
                "fps_estimate": 1.0 / step_time if step_time > 0 else float('inf')
            }
        
        return info
    
    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Render environment state for visualization.
        
        Args:
            mode: Rendering mode override. Options:
                - "human": Display interactive window
                - "rgb_array": Return RGB array
                - "headless": No visual output
                - None: Use environment default
                
        Returns:
            RGB array if mode="rgb_array", otherwise None
            
        Raises:
            ValueError: If render mode is invalid
            RuntimeError: If rendering fails
        """
        render_mode = mode or self.render_mode
        
        if render_mode is None:
            return None
        
        if render_mode not in ["human", "rgb_array", "headless"]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        
        if render_mode == "headless":
            return None
        
        if not MATPLOTLIB_AVAILABLE and render_mode != "headless":
            logger.warning("Matplotlib not available, skipping render")
            return None
        
        try:
            # Initialize rendering if not done
            if not self.render_initialized:
                self._init_render_display()
            
            # Get current frame and agent state
            current_frame = self.video_plume.get_frame(self.current_frame_index)
            agent_pos = self.navigator.positions[0]
            agent_orientation = self.navigator.orientations[0]
            
            # Update trajectory
            self.render_trajectory.append(tuple(agent_pos))
            
            # Clear and redraw
            self.ax.clear()
            
            # Display environment frame
            self.ax.imshow(current_frame, cmap='gray', origin='upper')
            
            # Draw agent trajectory
            if len(self.render_trajectory) > 1:
                traj_x, traj_y = zip(*self.render_trajectory)
                self.ax.plot(traj_x, traj_y, 'b-', alpha=0.7, linewidth=2, label='Trajectory')
            
            # Draw agent as oriented arrow
            arrow_length = 15
            dx = arrow_length * np.cos(np.radians(agent_orientation))
            dy = arrow_length * np.sin(np.radians(agent_orientation))
            
            self.ax.arrow(
                agent_pos[0], agent_pos[1], dx, dy,
                head_width=8, head_length=6, fc='red', ec='red', 
                linewidth=2, label='Agent'
            )
            
            # Add info text
            info_text = (
                f"Step: {self._step_count}\n"
                f"Reward: {self._total_reward:.2f}\n"
                f"Odor: {self._previous_odor:.3f}\n"
                f"Position: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})"
            )
            
            self.ax.text(
                0.02, 0.98, info_text, transform=self.ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10
            )
            
            self.ax.set_xlim(0, self.video_width)
            self.ax.set_ylim(self.video_height, 0)  # Invert y-axis for image coordinates
            self.ax.set_title(f"Odor Plume Navigation - Episode {self._episode_count}")
            self.ax.legend(loc='upper right')
            
            if render_mode == "human":
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                return None
            elif render_mode == "rgb_array":
                # Convert matplotlib figure to RGB array
                self.fig.canvas.draw()
                buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                return buf
            
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            if render_mode == "rgb_array":
                # Return black frame on failure
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return None
    
    def _init_render_display(self) -> None:
        """Initialize matplotlib display for rendering."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.ion()  # Interactive mode
        self.render_initialized = True
    
    def close(self) -> None:
        """
        Clean up environment resources.
        
        Closes video files, clears rendering resources, and performs cleanup
        operations to prevent resource leaks.
        """
        logger.debug("Closing GymnasiumEnv")
        
        try:
            # Close video plume
            if hasattr(self, 'video_plume'):
                self.video_plume.close()
            
            # Close rendering resources
            if self.render_initialized and MATPLOTLIB_AVAILABLE:
                plt.close(self.fig)
                self.render_initialized = False
            
            # Log performance summary if monitoring enabled
            if self.performance_monitoring and self._step_times:
                avg_step_time = np.mean(self._step_times)
                avg_fps = 1.0 / avg_step_time if avg_step_time > 0 else float('inf')
                
                logger.info(
                    f"Environment closed - Performance summary:",
                    extra={
                        "episodes": self._episode_count,
                        "total_steps": len(self._step_times),
                        "avg_step_time": avg_step_time,
                        "avg_fps": avg_fps,
                        "total_runtime": time.time() - self._start_time
                    }
                )
            
        except Exception as e:
            logger.error(f"Error during environment cleanup: {e}")
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set random seed for reproducible episodes.
        
        Args:
            seed: Random seed value (default: None for random seeding)
            
        Returns:
            List containing the used seed value
            
        Raises:
            ValueError: If seed is outside valid range
        """
        if seed is not None:
            if not (0 <= seed <= 2**32 - 1):
                raise ValueError(f"Seed must be between 0 and 2^32-1, got {seed}")
        
        # Use seed manager for comprehensive seeding
        try:
            if seed is not None:
                set_global_seed(seed)
                self._last_seed = seed
                logger.debug(f"Environment seed set to {seed}")
            else:
                # Generate random seed
                import random
                seed = random.randint(0, 2**32 - 1)
                set_global_seed(seed)
                self._last_seed = seed
                logger.debug(f"Environment auto-seeded with {seed}")
            
            return [seed]
            
        except Exception as e:
            logger.error(f"Failed to set seed: {e}")
            # Fallback to basic NumPy seeding
            if seed is not None:
                np.random.seed(seed)
                self._last_seed = seed
            return [seed] if seed is not None else [0]
    
    def get_attr(self, attr_name: str) -> Any:
        """
        Get environment attribute for vectorized environment compatibility.
        
        Args:
            attr_name: Name of attribute to retrieve
            
        Returns:
            Attribute value
            
        Raises:
            AttributeError: If attribute does not exist
        """
        return getattr(self, attr_name)
    
    def set_attr(self, attr_name: str, value: Any) -> None:
        """
        Set environment attribute for vectorized environment compatibility.
        
        Args:
            attr_name: Name of attribute to set
            value: New attribute value
            
        Raises:
            AttributeError: If attribute cannot be set
        """
        setattr(self, attr_name, value)
    
    def env_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call environment method for vectorized environment compatibility.
        
        Args:
            method_name: Name of method to call
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Method return value
            
        Raises:
            AttributeError: If method does not exist
        """
        method = getattr(self, method_name)
        return method(*args, **kwargs)
    
    def __str__(self) -> str:
        """String representation of environment."""
        return (
            f"GymnasiumEnv(video_path={self.video_path.name}, "
            f"dimensions={self.video_width}x{self.video_height}, "
            f"max_speed={self.max_speed}, "
            f"max_episode_steps={self.max_episode_steps})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation of environment."""
        return (
            f"GymnasiumEnv(\n"
            f"  video_path={self.video_path},\n"
            f"  dimensions=({self.video_width}, {self.video_height}),\n"
            f"  action_space={self.action_space},\n"
            f"  observation_space={self.observation_space},\n"
            f"  max_episode_steps={self.max_episode_steps},\n"
            f"  render_mode={self.render_mode}\n"
            f")"
        )


# Convenience functions for environment creation and validation

def create_gymnasium_environment(config: ConfigType, **kwargs) -> GymnasiumEnv:
    """
    Factory function to create GymnasiumEnv from configuration.
    
    This is the primary entry point for environment creation, supporting both
    programmatic instantiation and Hydra configuration-driven workflows.
    
    Args:
        config: Configuration dictionary or DictConfig
        **kwargs: Additional parameters to override configuration
        
    Returns:
        Configured GymnasiumEnv instance
        
    Examples:
        >>> config = {"video_path": "data/plume.mp4", "max_speed": 2.0}
        >>> env = create_gymnasium_environment(config)
        >>> 
        >>> # With overrides
        >>> env = create_gymnasium_environment(config, render_mode="human")
    """
    return GymnasiumEnv.from_config(config, **kwargs)


def validate_gymnasium_environment(env: GymnasiumEnv) -> Dict[str, Any]:
    """
    Validate environment compliance with Gymnasium API.
    
    Uses gymnasium.utils.env_checker to verify complete API compliance
    and returns validation results for debugging and testing.
    
    Args:
        env: GymnasiumEnv instance to validate
        
    Returns:
        Dictionary containing validation results and diagnostic information
        
    Raises:
        ImportError: If gymnasium env_checker is not available
        
    Examples:
        >>> env = GymnasiumEnv(video_path="test.mp4")
        >>> results = validate_gymnasium_environment(env)
        >>> assert results["is_valid"], f"Validation failed: {results['errors']}"
    """
    try:
        from gymnasium.utils.env_checker import check_env
        
        logger.info("Validating Gymnasium environment compliance")
        
        validation_results = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "validation_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Run comprehensive API validation
            check_env(env, warn=True, skip_render_check=False)
            validation_results["is_valid"] = True
            logger.info("Environment validation passed")
            
        except Exception as e:
            validation_results["errors"].append(str(e))
            logger.error(f"Environment validation failed: {e}")
        
        validation_results["validation_time"] = time.time() - start_time
        
        return validation_results
        
    except ImportError:
        raise ImportError(
            "gymnasium.utils.env_checker is required for validation. "
            "Install with: pip install 'gymnasium[other]'"
        )


# Public API exports
__all__ = [
    "GymnasiumEnv",
    "create_gymnasium_environment", 
    "validate_gymnasium_environment"
]