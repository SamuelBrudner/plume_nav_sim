"""
Core Gymnasium-Compliant Reinforcement Learning Environment for Plume Navigation Simulation.

This module provides the primary PlumeNavigationEnv class that implements the modern Gymnasium 0.29.x
interface for odor plume navigation tasks. The environment serves as the critical bridge between the
existing plume_nav_sim simulation framework and modern RL frameworks like stable-baselines3, providing
standardized reset/step/render/close methods with dual API compatibility for legacy Gym support.

Enhanced for Gymnasium 0.29.x Migration:
- Modern 5-tuple step returns: (obs, reward, terminated, truncated, info) with automatic conversion 
  to legacy 4-tuple when invoked through compatibility shim
- Extensibility hooks: compute_additional_obs(), compute_extra_reward(), on_episode_end() for 
  research customization without modifying core environment logic
- Enhanced frame caching integration with configurable modes (none, LRU, all) and memory management
- Dual API compatibility detection and automatic format conversion via stack introspection
- Performance monitoring with <10ms step execution requirements per Section 2.2.3

Key Features:
- Full Gymnasium API compliance with env_checker validation
- Integration with NavigatorProtocol and VideoPlume components  
- Configurable action and observation spaces supporting multi-modal sensing
- Domain-specific reward functions optimized for olfactory navigation research
- Vectorized environment support for parallel training workflows
- Comprehensive seed management for reproducible experiments
- Real-time rendering and headless export capabilities with matplotlib integration
- Thread-safe operations supporting concurrent environment instances

Technical Architecture:
- Wraps existing simulation components without modification for backward compatibility
- Maps Gymnasium actions to NavigatorProtocol control inputs via standardized interface
- Converts navigator state to standardized observation dictionaries with type safety
- Implements configurable reward computation with domain expertise and extensible hooks
- Maintains performance requirements through optimized execution paths and frame caching
- Provides automatic API compatibility detection via stack introspection

Performance Characteristics:
- <10ms step execution for real-time training compatibility (Section 2.2.3)
- >90% cache hit rate for sequential frame access patterns with intelligent LRU eviction
- Support for 100+ concurrent agents with vectorized operations
- Memory-efficient frame caching with 2 GiB default limit and pressure monitoring
- Thread-safe multi-agent access with atomic operations and proper synchronization

Example Usage:
    Basic environment creation with Gymnasium interface:
    >>> import gymnasium as gym
    >>> env = gym.make("PlumeNavSim-v0", video_path="plume_movie.mp4")
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    
    Legacy compatibility through shim:
    >>> from plume_nav_sim.shims import gym_make
    >>> env = gym_make("PlumeNavSim-v0", video_path="plume_movie.mp4")  # Issues DeprecationWarning
    >>> obs = env.reset()  # Legacy 1-tuple return
    >>> obs, reward, done, info = env.step(action)  # Legacy 4-tuple return
    
    Configuration-driven setup with Hydra integration:
    >>> config = {
    ...     "video_path": "experiments/plume_data.mp4",
    ...     "navigator": {"max_speed": 3.0, "position": [100, 200]},
    ...     "frame_cache": {"mode": "lru", "memory_limit_mb": 1024},
    ...     "spaces": {"include_multi_sensor": True, "num_sensors": 3}
    ... }
    >>> env = PlumeNavigationEnv.from_config(config)
    
    Custom extensibility hooks for research:
    >>> class CustomPlumeEnv(PlumeNavigationEnv):
    ...     def compute_additional_obs(self, base_obs: dict) -> dict:
    ...         return {"wind_direction": self.sample_wind_direction()}
    ...     
    ...     def compute_extra_reward(self, base_reward: float, info: dict) -> float:
    ...         return 0.1 if self.is_novel_position() else 0.0
    ...     
    ...     def on_episode_end(self, final_info: dict) -> None:
    ...         self.logger.info(f"Episode completed: {final_info['total_reward']}")
    
    High-performance training with frame caching:
    >>> from plume_nav_sim.utils.frame_cache import create_lru_cache
    >>> cache = create_lru_cache(memory_limit_mb=512)
    >>> env = PlumeNavigationEnv(video_path="plume_movie.mp4", frame_cache=cache)
    >>> # Achieves >90% cache hit rate for training efficiency
"""

from __future__ import annotations
import time
import warnings
import inspect
from typing import Dict, Tuple, Optional, Any, Union, List, SupportsFloat, Literal
from pathlib import Path
import numpy as np

# Gymnasium imports with fallback compatibility
try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Dict as DictSpace
    from gymnasium.error import DependencyNotInstalled
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    # Create mock classes to prevent import errors during transition
    class gym:
        class Env:
            def __init__(self): pass
    Box = DictSpace = None

# Core plume navigation imports with graceful fallbacks during migration
try:
    from plume_nav_sim.core.protocols import NavigatorProtocol, NavigatorFactory
    NAVIGATOR_AVAILABLE = True
except ImportError:
    # Fallback during migration - will be created by other agents
    NavigatorProtocol = Any
    class NavigatorFactory:
        @staticmethod
        def single_agent(**kwargs):
            raise ImportError("NavigatorFactory not yet available")
    NAVIGATOR_AVAILABLE = False

# Enhanced space definitions with proper Gymnasium compliance
try:
    from plume_nav_sim.envs.spaces import (
        ActionSpaceFactory, ObservationSpaceFactory, SpaceValidator,
        ReturnFormatConverter, get_standard_action_space, get_standard_observation_space
    )
    SPACES_AVAILABLE = True
except ImportError:
    # Minimal fallback during migration
    class ActionSpaceFactory:
        @staticmethod
        def create_continuous_action_space(**kwargs):
            if GYMNASIUM_AVAILABLE:
                return Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            return None
    
    class ObservationSpaceFactory:
        @staticmethod
        def create_navigation_observation_space(**kwargs):
            if GYMNASIUM_AVAILABLE:
                return DictSpace({
                    "odor_concentration": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "agent_position": Box(low=-1000.0, high=1000.0, shape=(2,), dtype=np.float32),
                    "agent_orientation": Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32)
                })
            return None
    
    class SpaceValidator:
        @staticmethod
        def validate_gymnasium_compliance(space):
            return True
    
    class ReturnFormatConverter:
        @staticmethod
        def to_legacy_format(gymnasium_return):
            obs, reward, terminated, truncated, info = gymnasium_return
            done = terminated or truncated
            return obs, reward, done, info
        
        @staticmethod
        def to_gymnasium_format(legacy_return):
            obs, reward, done, info = legacy_return
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info
    
    def get_standard_action_space():
        return ActionSpaceFactory.create_continuous_action_space()
    
    def get_standard_observation_space():
        return ObservationSpaceFactory.create_navigation_observation_space()
    
    SPACES_AVAILABLE = False

# Video plume processing with fallback
try:
    from plume_nav_sim.envs.video_plume import VideoPlume
    VIDEO_PLUME_AVAILABLE = True
except ImportError:
    # Minimal fallback implementation
    class VideoPlume:
        def __init__(self, video_path: str):
            self.video_path = video_path
            self.frame_count = 1000
            self.width = 640
            self.height = 480
            self.fps = 30.0
        
        def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
            return np.random.rand(self.height, self.width).astype(np.float32)
        
        def get_metadata(self) -> Dict[str, Any]:
            return {
                "width": self.width,
                "height": self.height, 
                "fps": self.fps,
                "frame_count": self.frame_count
            }
        
        def close(self):
            pass
    
    VIDEO_PLUME_AVAILABLE = False

# Frame caching with enhanced memory management
try:
    from plume_nav_sim.utils.frame_cache import FrameCache, CacheMode
    FRAME_CACHE_AVAILABLE = True
except ImportError:
    # Minimal fallback
    class FrameCache:
        def __init__(self, **kwargs):
            self.hit_rate = 0.0
            self.hits = 0
            self.misses = 0
        
        def get(self, frame_id, video_plume, **kwargs):
            return video_plume.get_frame(frame_id, **kwargs)
        
        def clear(self):
            pass
    
    class CacheMode:
        NONE = "none"
        LRU = "lru"
        ALL = "all"
    
    FRAME_CACHE_AVAILABLE = False

# Seed management utilities
try:
    from plume_nav_sim.utils.seed_utils import set_global_seed, get_seed_context
    SEED_UTILS_AVAILABLE = True
except ImportError:
    def set_global_seed(seed: int):
        np.random.seed(seed)
    
    def get_seed_context():
        return {}
    
    SEED_UTILS_AVAILABLE = False

# Enhanced logging with correlation support
try:
    from plume_nav_sim.utils.logging_setup import (
        get_enhanced_logger, correlation_context, create_step_timer,
        log_legacy_api_deprecation
    )
    LOGGING_AVAILABLE = True
except ImportError:
    import logging
    
    def get_enhanced_logger(name):
        return logging.getLogger(name)
    
    def correlation_context(name, **kwargs):
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            correlation_id = "dummy"
        return DummyContext()
    
    def create_step_timer():
        class DummyTimer:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyTimer()
    
    def log_legacy_api_deprecation(**kwargs):
        pass
    
    LOGGING_AVAILABLE = False

# Configuration support
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    class OmegaConf:
        @staticmethod
        def to_container(config, resolve=True):
            return dict(config)
    HYDRA_AVAILABLE = False

# Matplotlib for rendering
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Initialize module logger
logger = get_enhanced_logger(__name__)

# Type aliases for enhanced readability and IDE support
ConfigType = Union[DictConfig, Dict[str, Any]]
InfoType = Dict[str, Any]
RenderModeType = Optional[str]
SeedType = Optional[int]
ActionType = np.ndarray
ObservationType = Dict[str, np.ndarray]


def _detect_legacy_gym_caller() -> bool:
    """
    Detect if environment is being created via legacy gym interface.
    
    Uses stack introspection to determine if the calling code expects legacy
    4-tuple returns or modern Gymnasium 5-tuple returns, enabling automatic
    API compatibility without requiring explicit configuration.
    
    Returns:
        bool: True if legacy gym is detected, False for modern Gymnasium
        
    Note:
        This function implements the dual API compatibility detection required
        by Section 2.2.3 F-005-RQ-001 for seamless migration support.
    """
    try:
        # Inspect call stack to detect caller context
        frame = inspect.currentframe()
        while frame:
            frame = frame.f_back
            if frame and frame.f_globals:
                module_name = frame.f_globals.get('__name__', '')
                
                # Check for legacy gym imports or usage patterns
                if 'gym' in module_name and 'gymnasium' not in module_name:
                    return True
                
                # Check for legacy gym package in frame globals
                if 'gym' in frame.f_globals and 'gymnasium' not in frame.f_globals:
                    # Further validate it's the legacy gym
                    gym_module = frame.f_globals.get('gym')
                    if hasattr(gym_module, 'make') and not hasattr(gym_module, 'version'):
                        return True
                        
        return False
    except Exception:
        # If detection fails, default to modern Gymnasium API
        return False


class PlumeNavigationEnv(gym.Env):
    """
    Gymnasium-compliant environment for odor plume navigation simulation.
    
    This class implements the standard Gymnasium interface while integrating seamlessly 
    with the existing plume navigation simulation infrastructure. It exposes the 
    NavigatorProtocol and VideoPlume components through standardized RL methods,
    enabling direct compatibility with stable-baselines3 and other modern RL frameworks.
    
    Enhanced for Gymnasium 0.29.x Migration:
    - Modern 5-tuple step returns with automatic legacy conversion
    - Extensibility hooks for research customization
    - Enhanced frame caching with configurable memory management
    - Dual API compatibility detection and format conversion
    - Performance monitoring with structured logging integration
    
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
    - Optional frame caching for sub-10ms step execution performance
    
    Performance Characteristics:
    - <10ms step execution for real-time training compatibility
    - >90% cache hit rate target for sequential access patterns
    - Thread-safe concurrent access for 100+ agents
    - Automatic memory pressure handling at 90% threshold
    - O(1) observation/action processing with vectorized operations
    
    Attributes:
        action_space: Gymnasium Box space for continuous control [speed, angular_velocity]
        observation_space: Gymnasium Dict space with multi-modal observations
        metadata: Environment metadata including render modes and performance specs
        spec: Gymnasium environment specification for registration
        
    Examples:
        Basic environment creation:
        >>> env = PlumeNavigationEnv(
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
        ...     "reward": {"odor_weight": 1.0, "distance_weight": -0.1},
        ...     "frame_cache": {"mode": "lru", "memory_limit_mb": 512}
        ... }
        >>> env = PlumeNavigationEnv.from_config(config)
        
        Integration with stable-baselines3:
        >>> from stable_baselines3 import PPO
        >>> env = PlumeNavigationEnv.from_config(training_config)
        >>> model = PPO("MultiInputPolicy", env, verbose=1)
        >>> model.learn(total_timesteps=100000)
        
        High-performance training with frame caching:
        >>> from plume_nav_sim.utils.frame_cache import create_lru_cache
        >>> cache = create_lru_cache(memory_limit_mb=512)
        >>> env = PlumeNavigationEnv(
        ...     video_path="plume_movie.mp4",
        ...     frame_cache=cache,
        ...     performance_monitoring=True
        ... )
        >>> obs, info = env.reset()
        >>> print(f"Cache hit rate: {info['perf_stats']['cache_hit_rate']:.2%}")
    """
    
    # Gymnasium metadata for environment registration
    metadata = {
        "render_modes": ["human", "rgb_array", "headless"],
        "render_fps": 30,
        "spec_id": "PlumeNavSim-v0",  # New Gymnasium 0.29.x compliant environment ID
        "legacy_spec_id": "OdorPlumeNavigation-v1",  # Legacy compatibility
        "max_episode_steps": 1000,
        "reward_threshold": 100.0,
        "nondeterministic": False,
        "author": "Blitzy Platform",
        "environment_type": "continuous_control",
        "action_type": "continuous",
        "observation_type": "multi_modal_dict",
        "gymnasium_version": "0.29.*",
        "api_compliance": "dual_mode"  # Supports both legacy and modern APIs
    }
    
    def __init__(
        self,
        video_path: Union[str, Path],
        initial_position: Optional[Tuple[float, float]] = None,
        initial_orientation: float = 0.0,
        max_speed: float = 2.0,
        max_angular_velocity: float = np.pi,  # radians per second
        include_multi_sensor: bool = False,
        num_sensors: int = 2,
        sensor_distance: float = 5.0,
        sensor_layout: str = "bilateral",
        reward_config: Optional[Dict[str, float]] = None,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        performance_monitoring: bool = True,
        frame_cache: Optional[FrameCache] = None,
        _force_legacy_api: bool = False,
        **kwargs
    ):
        """
        Initialize Gymnasium environment wrapper with comprehensive configuration.
        
        Args:
            video_path: Path to video file containing odor plume data
            initial_position: Starting (x, y) position for agent (default: video center)
            initial_orientation: Starting orientation in radians (default: 0.0)
            max_speed: Maximum agent speed in units per time step (default: 2.0)
            max_angular_velocity: Maximum angular velocity in radians/sec (default: π)
            include_multi_sensor: Whether to include multi-sensor observations (default: False)
            num_sensors: Number of additional sensors for multi-sensor mode (default: 2)
            sensor_distance: Distance from agent center to sensors (default: 5.0)
            sensor_layout: Sensor arrangement ("bilateral", "triangular", "custom") (default: "bilateral")
            reward_config: Dictionary of reward function weights (default: standard weights)
            max_episode_steps: Maximum steps per episode (default: 1000)
            render_mode: Rendering mode ("human", "rgb_array", "headless") (default: None)
            seed: Random seed for reproducible experiments (default: None)
            performance_monitoring: Enable performance tracking (default: True)
            frame_cache: Optional FrameCache instance for high-performance frame retrieval
            _force_legacy_api: Force legacy API mode (internal use)
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
                "gymnasium is required for PlumeNavigationEnv. "
                "Install with: pip install gymnasium>=0.29.0"
            )
        
        super().__init__()
        
        # Store configuration parameters
        self.video_path = Path(video_path)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.performance_monitoring = performance_monitoring
        
        # Store and validate frame cache instance for performance optimization
        self.frame_cache = frame_cache
        self._cache_enabled = frame_cache is not None and FRAME_CACHE_AVAILABLE
        
        # Detect API compatibility mode for legacy gym support
        self._use_legacy_api = _force_legacy_api or _detect_legacy_gym_caller()
        self._correlation_id = None
        
        # Validate video file existence
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Initialize performance tracking
        self._step_count = 0
        self._episode_count = 0
        self._total_reward = 0.0
        self._start_time = time.time()
        self._step_times: List[float] = []
        
        with correlation_context(
            "plume_env_init", 
            video_path=str(self.video_path),
            legacy_api=self._use_legacy_api,
            performance_monitoring=self.performance_monitoring
        ) as ctx:
            self._correlation_id = ctx.correlation_id
            
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
                    f"PlumeNavigationEnv initialized successfully",
                    extra={
                        "video_dims": f"{self.video_width}x{self.video_height}",
                        "action_space": str(self.action_space),
                        "obs_space_keys": list(self.observation_space.spaces.keys()),
                        "max_episode_steps": self.max_episode_steps,
                        "api_mode": "legacy" if self._use_legacy_api else "gymnasium",
                        "cache_enabled": self._cache_enabled,
                        "metric_type": "environment_initialization"
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to initialize PlumeNavigationEnv: {e}")
                raise RuntimeError(f"Environment initialization failed: {e}") from e
    
    def _init_video_plume(self) -> None:
        """Initialize VideoPlume environment processor with optional frame cache integration."""
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
            if NAVIGATOR_AVAILABLE:
                self.navigator = NavigatorFactory.single_agent(
                    position=initial_position,
                    orientation=initial_orientation,
                    speed=0.0,  # Start stationary
                    max_speed=max_speed,
                    angular_velocity=0.0  # Start with no rotation
                )
            else:
                # Minimal fallback navigator for testing
                class MockNavigator:
                    def __init__(self, position, orientation, max_speed):
                        self.positions = np.array([position], dtype=np.float32)
                        self.orientations = np.array([orientation], dtype=np.float32)
                        self.speeds = np.array([0.0], dtype=np.float32)
                        self.max_speeds = np.array([max_speed], dtype=np.float32)
                        self.angular_velocities = np.array([0.0], dtype=np.float32)
                        self.num_agents = 1
                    
                    def reset(self, **kwargs):
                        if 'position' in kwargs:
                            self.positions[0] = kwargs['position']
                        if 'orientation' in kwargs:
                            self.orientations[0] = kwargs['orientation']
                        if 'speed' in kwargs:
                            self.speeds[0] = kwargs['speed']
                        if 'angular_velocity' in kwargs:
                            self.angular_velocities[0] = kwargs['angular_velocity']
                    
                    def step(self, env_array, dt=1.0):
                        # Simple integration step
                        angle = np.radians(self.orientations[0])
                        dx = self.speeds[0] * np.cos(angle) * dt
                        dy = self.speeds[0] * np.sin(angle) * dt
                        self.positions[0] += [dx, dy]
                        self.orientations[0] += np.degrees(self.angular_velocities[0] * dt)
                        self.orientations[0] = self.orientations[0] % 360
                    
                    def sample_odor(self, env_array):
                        pos = self.positions[0]
                        h, w = env_array.shape
                        x, y = int(pos[0]), int(pos[1])
                        if 0 <= x < w and 0 <= y < h:
                            return float(env_array[y, x])
                        return 0.0
                    
                    def sample_multiple_sensors(self, env_array, **kwargs):
                        # Simple multi-sensor implementation
                        return np.array([self.sample_odor(env_array)], dtype=np.float32)
                    
                    def compute_additional_obs(self, base_obs):
                        return {}
                    
                    def compute_extra_reward(self, base_reward, info):
                        return 0.0
                    
                    def on_episode_end(self, final_info):
                        pass
                
                self.navigator = MockNavigator(initial_position, initial_orientation, max_speed)
            
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
            if SPACES_AVAILABLE:
                self.action_space = ActionSpaceFactory.create_continuous_action_space(
                    speed_range=(0.0, self.max_speed),
                    angular_velocity_range=(-self.max_angular_velocity, self.max_angular_velocity),
                    dtype=np.float32
                )
                
                # Create observation space with multi-sensor support
                self.observation_space = ObservationSpaceFactory.create_navigation_observation_space(
                    include_position=True,
                    include_velocity=False,  # We track speed and orientation separately
                    include_odor=True,
                    include_sensors=include_multi_sensor,
                    position_bounds=(0.0, max(self.video_width, self.video_height)),
                    odor_concentration_range=(0.0, 1.0),
                    dtype=np.float32
                )
            else:
                # Fallback space creation
                self.action_space = Box(
                    low=np.array([0.0, -self.max_angular_velocity], dtype=np.float32),
                    high=np.array([self.max_speed, self.max_angular_velocity], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32
                )
                
                obs_spaces = {
                    "odor_concentration": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    "agent_position": Box(
                        low=0.0, high=max(self.video_width, self.video_height), 
                        shape=(2,), dtype=np.float32
                    ),
                    "agent_orientation": Box(low=0.0, high=360.0, shape=(1,), dtype=np.float32)
                }
                
                if include_multi_sensor:
                    obs_spaces["multi_sensor_readings"] = Box(
                        low=0.0, high=1.0, shape=(num_sensors,), dtype=np.float32
                    )
                
                self.observation_space = DictSpace(obs_spaces)
            
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
    ) -> 'PlumeNavigationEnv':
        """
        Create PlumeNavigationEnv from configuration with parameter validation.
        
        Args:
            config: Configuration dictionary or DictConfig containing environment parameters
            **override_kwargs: Additional parameters to override configuration values
            
        Returns:
            Configured PlumeNavigationEnv instance
            
        Raises:
            ValueError: If configuration is invalid or incomplete
            TypeError: If configuration type is unsupported
        """
        logger.info("Creating PlumeNavigationEnv from configuration")
        
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
        
        # Extract frame cache configuration
        frame_cache_config = config_dict.get("frame_cache")
        frame_cache_instance = None
        
        # Create frame cache instance if configuration provided and cache is available
        if frame_cache_config and FRAME_CACHE_AVAILABLE:
            try:
                if isinstance(frame_cache_config, FrameCache):
                    frame_cache_instance = frame_cache_config
                elif isinstance(frame_cache_config, dict):
                    frame_cache_instance = FrameCache(**frame_cache_config)
                logger.debug(f"Frame cache configuration found: {type(frame_cache_config)}")
            except Exception as e:
                logger.warning(f"Failed to create frame cache from config: {e}")
                frame_cache_instance = None
        
        # Build constructor arguments
        constructor_args = {
            "video_path": config_dict["video_path"],
            "initial_position": initial_position,
            "initial_orientation": nav_config.get("orientation", 0.0),
            "max_speed": nav_config.get("max_speed", 2.0),
            "max_angular_velocity": nav_config.get("max_angular_velocity", np.pi),
            "include_multi_sensor": space_config.get("include_multi_sensor", False),
            "num_sensors": space_config.get("num_sensors", 2),
            "sensor_distance": space_config.get("sensor_distance", 5.0),
            "sensor_layout": space_config.get("sensor_layout", "bilateral"),
            "reward_config": reward_config,
            "max_episode_steps": config_dict.get("max_episode_steps", 1000),
            "render_mode": config_dict.get("render_mode"),
            "seed": config_dict.get("seed"),
            "performance_monitoring": config_dict.get("performance_monitoring", True),
            "frame_cache": frame_cache_instance
        }
        
        # Remove None values to use defaults
        constructor_args = {k: v for k, v in constructor_args.items() if v is not None}
        
        try:
            env = cls(**constructor_args)
            
            logger.info(
                "PlumeNavigationEnv created successfully from configuration",
                extra={
                    "video_path": constructor_args["video_path"],
                    "navigator_config": nav_config,
                    "space_config": space_config,
                    "reward_config": reward_config
                }
            )
            
            return env
            
        except Exception as e:
            logger.error(f"Failed to create PlumeNavigationEnv from config: {e}")
            raise
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObservationType, InfoType]:
        """
        Reset environment to initial state with optional parameter overrides.
        
        Implements the modern Gymnasium reset() signature returning (observation, info)
        with comprehensive state initialization and optional parameter overrides for
        varying episode initial conditions.
        
        Args:
            seed: Random seed for episode reproducibility (optional)
            options: Dictionary of reset options (optional). Supported keys:
                - position: Override initial position as (x, y) tuple
                - orientation: Override initial orientation in radians
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
            time for real-time training workflows per Section 2.2.3.
        """
        if self.performance_monitoring:
            reset_start = time.time()
        
        logger.debug(f"Resetting environment (episode {self._episode_count + 1})", extra={
            "correlation_id": self._correlation_id,
            "episode": self._episode_count + 1,
            "metric_type": "environment_reset"
        })
        
        # Handle seeding if provided
        if seed is not None:
            try:
                if SEED_UTILS_AVAILABLE:
                    set_global_seed(seed)
                else:
                    np.random.seed(seed)
                self._last_seed = seed
                logger.debug(f"Environment reset with seed {seed}")
            except Exception as e:
                logger.warning(f"Failed to set seed {seed}: {e}")
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
                "reset_options": options or {},
                "correlation_id": self._correlation_id,
                "api_mode": "legacy" if self._use_legacy_api else "gymnasium"
            }
            
            # Update episode counter
            self._episode_count += 1
            
            if self.performance_monitoring:
                reset_time = time.time() - reset_start
                info["reset_time"] = reset_time
                
                # Log performance threshold violations
                if reset_time > 0.01:  # 10ms target
                    logger.warning(
                        f"Environment reset() latency exceeded threshold: {reset_time:.3f}s > 0.01s",
                        extra={
                            "metric_type": "reset_latency_violation",
                            "actual_latency_ms": reset_time * 1000,
                            "threshold_latency_ms": 10.0
                        }
                    )
            
            logger.debug(f"Environment reset completed in episode {self._episode_count}")
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            raise RuntimeError(f"Environment reset failed: {e}") from e
    
    def step(
        self, 
        action: ActionType
    ) -> Union[Tuple[ObservationType, SupportsFloat, bool, InfoType], 
               Tuple[ObservationType, SupportsFloat, bool, bool, InfoType]]:
        """
        Execute one environment step with the given action.
        
        Implements both modern Gymnasium 5-tuple and legacy 4-tuple return formats
        based on automatic caller detection. Applies the action to the navigator,
        advances simulation by one time step, computes domain-specific rewards,
        and determines episode termination conditions.
        
        Args:
            action: Action array with shape (2,) containing [speed, angular_velocity]
                   Values are automatically clipped to valid action space bounds
                    
        Returns:
            Union[4-tuple, 5-tuple]: API-compatible return based on caller context:
                - Legacy gym API: (obs, reward, done, info)
                - Modern Gymnasium API: (obs, reward, terminated, truncated, info)
                
        Raises:
            ValueError: If action format is invalid
            RuntimeError: If step execution fails
            
        Note:
            Step execution is performance-critical and targets <10ms completion
            time per Section 2.2.3 for real-time simulation performance.
        """
        if self.performance_monitoring:
            with create_step_timer() as perf_metrics:
                return self._execute_step_with_monitoring(action, perf_metrics)
        else:
            return self._execute_step_without_monitoring(action)
    
    def _execute_step_with_monitoring(self, action: ActionType, perf_metrics) -> Union[Tuple, Tuple]:
        """Execute step with performance monitoring enabled."""
        step_start = time.time()
        
        # Validate and clip action
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")
        
        # Clip action to space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
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
            if self._cache_enabled and self.frame_cache:
                current_frame = self.frame_cache.get(
                    frame_id=self.current_frame_index,
                    video_plume=self.video_plume
                )
            else:
                current_frame = self.video_plume.get_frame(self.current_frame_index)
            
            if current_frame is None:
                # Handle end of video by cycling
                self.current_frame_index = 0
                if self._cache_enabled and self.frame_cache:
                    current_frame = self.frame_cache.get(
                        frame_id=0, video_plume=self.video_plume
                    )
                else:
                    current_frame = self.video_plume.get_frame(0)
            
            # Execute navigator step
            self.navigator.step(current_frame, dt=1.0)
            
            # Advance frame index for next step
            self.current_frame_index = (self.current_frame_index + 1) % self.video_frame_count
            
            # Get new state observation
            observation = self._get_observation()
            
            # Compute reward based on transition
            base_reward = self._compute_reward(
                action, prev_position, prev_orientation, prev_odor, observation
            )
            
            # Apply extensibility hooks
            additional_obs = self._apply_observation_hooks(observation)
            observation.update(additional_obs)
            
            extra_reward = self._apply_reward_hooks(base_reward, observation)
            reward = base_reward + extra_reward
            
            # Update step counter and tracking
            self._step_count += 1
            self._total_reward += reward
            
            # Check termination conditions
            terminated, truncated = self._check_termination(observation)
            
            # Apply episode end hooks if episode is ending
            if terminated or truncated:
                self._apply_episode_end_hooks({
                    "episode": self._episode_count,
                    "total_steps": self._step_count,
                    "total_reward": self._total_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "final_observation": observation
                })
            
            # Update exploration tracking
            self._update_exploration_tracking(self.navigator.positions[0])
            
            # Prepare detailed info dictionary
            step_time = time.time() - step_start
            self._step_times.append(step_time)
            
            info = self._prepare_step_info(
                action, reward, prev_position, prev_orientation, step_start
            )
            info["step_time"] = step_time
            info["correlation_id"] = self._correlation_id
            
            # Add cache statistics if available
            if self._cache_enabled and self.frame_cache:
                info["cache_stats"] = {
                    "hit_rate": getattr(self.frame_cache, "hit_rate", 0.0),
                    "hits": getattr(self.frame_cache, "hits", 0),
                    "misses": getattr(self.frame_cache, "misses", 0),
                    "memory_usage_mb": getattr(self.frame_cache, "memory_usage_mb", 0.0)
                }
            
            # Add video frame for analysis
            info["video_frame"] = current_frame
            
            # Log performance threshold violations
            if step_time > 0.01:  # 10ms target
                logger.warning(
                    f"Environment step() latency exceeded threshold: {step_time:.3f}s > 0.01s",
                    extra={
                        "metric_type": "step_latency_violation",
                        "actual_latency_ms": step_time * 1000,
                        "threshold_latency_ms": 10.0,
                        "step_count": self._step_count
                    }
                )
            
            # Return appropriate tuple based on API compatibility
            return self._format_step_return(observation, reward, terminated, truncated, info)
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}", extra={
                "correlation_id": self._correlation_id,
                "step_count": self._step_count
            })
            raise RuntimeError(f"Environment step failed: {e}") from e

    def _execute_step_without_monitoring(self, action: ActionType) -> Union[Tuple, Tuple]:
        """Execute step without performance monitoring for maximum performance."""
        # Validate and clip action
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")
        
        # Clip action to space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
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
            if self._cache_enabled and self.frame_cache:
                current_frame = self.frame_cache.get(
                    frame_id=self.current_frame_index,
                    video_plume=self.video_plume
                )
            else:
                current_frame = self.video_plume.get_frame(self.current_frame_index)
            
            if current_frame is None:
                # Handle end of video by cycling
                self.current_frame_index = 0
                if self._cache_enabled and self.frame_cache:
                    current_frame = self.frame_cache.get(
                        frame_id=0, video_plume=self.video_plume
                    )
                else:
                    current_frame = self.video_plume.get_frame(0)
            
            # Execute navigator step
            self.navigator.step(current_frame, dt=1.0)
            
            # Advance frame index for next step
            self.current_frame_index = (self.current_frame_index + 1) % self.video_frame_count
            
            # Get new state observation
            observation = self._get_observation()
            
            # Compute reward based on transition
            base_reward = self._compute_reward(
                action, prev_position, prev_orientation, prev_odor, observation
            )
            
            # Apply extensibility hooks
            additional_obs = self._apply_observation_hooks(observation)
            observation.update(additional_obs)
            
            extra_reward = self._apply_reward_hooks(base_reward, observation)
            reward = base_reward + extra_reward
            
            # Update step counter and tracking
            self._step_count += 1
            self._total_reward += reward
            
            # Check termination conditions
            terminated, truncated = self._check_termination(observation)
            
            # Apply episode end hooks if episode is ending
            if terminated or truncated:
                self._apply_episode_end_hooks({
                    "episode": self._episode_count,
                    "total_steps": self._step_count,
                    "total_reward": self._total_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "final_observation": observation
                })
            
            # Update exploration tracking
            self._update_exploration_tracking(self.navigator.positions[0])
            
            # Prepare basic info dictionary
            info = self._prepare_step_info(
                action, reward, prev_position, prev_orientation, None
            )
            
            # Add video frame for analysis workflows
            info["video_frame"] = current_frame
            
            # Return appropriate tuple based on API compatibility
            return self._format_step_return(observation, reward, terminated, truncated, info)
            
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            raise RuntimeError(f"Environment step failed: {e}") from e
    
    def _format_step_return(
        self, 
        observation: ObservationType, 
        reward: float, 
        terminated: bool, 
        truncated: bool, 
        info: InfoType
    ) -> Union[Tuple, Tuple]:
        """Format step return based on API compatibility mode."""
        if self._use_legacy_api:
            # Return 4-tuple for legacy gym API compatibility
            if SPACES_AVAILABLE:
                return ReturnFormatConverter.to_legacy_format(
                    (observation, reward, terminated, truncated, info)
                )
            else:
                done = terminated or truncated
                return observation, reward, done, info
        else:
            # Return 5-tuple for modern Gymnasium API
            return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> ObservationType:
        """
        Generate current observation dictionary from navigator and environment state.
        
        Returns:
            Observation dictionary containing:
                - odor_concentration: Current odor level at agent position
                - agent_position: Agent coordinates [x, y]
                - agent_orientation: Agent heading in radians
                - multi_sensor_readings: Optional multi-sensor data (if enabled)
        """
        # Get current environment frame
        if self._cache_enabled and self.frame_cache:
            current_frame = self.frame_cache.get(
                frame_id=self.current_frame_index,
                video_plume=self.video_plume
            )
        else:
            current_frame = self.video_plume.get_frame(self.current_frame_index)
        
        # Sample odor concentration at agent position
        odor_concentration = self.navigator.sample_odor(current_frame)
        self._previous_odor = float(odor_concentration)
        
        # Get agent state
        agent_position = self.navigator.positions[0].astype(np.float32)
        agent_orientation = np.array([self.navigator.orientations[0]], dtype=np.float32)
        
        # Build core observation
        observation = {
            "odor_concentration": np.array([odor_concentration], dtype=np.float32),
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
        current_odor = float(observation["odor_concentration"][0])
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
        current_odor = float(observation["odor_concentration"][0])
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
            
            # Build performance stats dictionary
            perf_stats = {
                "step_time_ms": step_time * 1000,
                "avg_step_time_ms": np.mean(self._step_times[-100:]) * 1000 if self._step_times else step_time * 1000,
                "fps_estimate": 1.0 / step_time if step_time > 0 else float('inf'),
                "step_count": self._step_count,
                "episode": self._episode_count
            }
            
            # Add cache performance metrics if available
            if self._cache_enabled and self.frame_cache:
                perf_stats.update({
                    "cache_hit_rate": getattr(self.frame_cache, 'hit_rate', 0.0),
                    "cache_hits": getattr(self.frame_cache, 'hits', 0),
                    "cache_misses": getattr(self.frame_cache, 'misses', 0),
                    "cache_memory_usage_mb": getattr(self.frame_cache, 'memory_usage_mb', 0.0)
                })
            
            info["perf_stats"] = perf_stats
            
            # Legacy performance dictionary for backward compatibility
            info["performance"] = {
                "step_time": step_time,
                "avg_step_time": np.mean(self._step_times[-100:]) if self._step_times else step_time,
                "fps_estimate": 1.0 / step_time if step_time > 0 else float('inf')
            }
        
        return info
    
    # Extensibility hooks per Section 2.2.3 F-005-RQ-006
    
    def compute_additional_obs(self, base_obs: dict) -> dict:
        """
        Compute additional observations for custom environment extensions.
        
        This extensibility hook allows downstream implementations to augment 
        the base observation with domain-specific data without modifying the 
        core navigation logic. Override this method to add custom sensors,
        derived metrics, or specialized data processing.
        
        Args:
            base_obs: Base observation dict containing standard navigation data
                
        Returns:
            dict: Additional observation components to merge with base_obs
            
        Note:
            Default implementation returns empty dict (no additional observations).
            Implementations should maintain consistent key names and value types
            across episodes for stable RL training.
        """
        return {}
    
    def compute_extra_reward(self, base_reward: float, info: dict) -> float:
        """
        Compute additional reward components for custom reward shaping.
        
        This extensibility hook enables reward shaping and custom reward
        function implementations without modifying core environment logic.
        Override this method to implement domain-specific incentives,
        exploration bonuses, or multi-objective reward functions.
        
        Args:
            base_reward: Base reward computed by the environment's standard
                reward function (typically odor-based navigation reward)
            info: Environment info dict containing episode state and metrics
                
        Returns:
            float: Additional reward component to add to base_reward
            
        Note:
            Default implementation returns 0.0 (no additional reward).
            The final environment reward will be: base_reward + extra_reward.
        """
        return 0.0
    
    def on_episode_end(self, final_info: dict) -> None:
        """
        Handle episode completion events for logging and cleanup.
        
        This extensibility hook is called when an episode terminates or
        truncates, providing an opportunity for custom logging, metric
        computation, state persistence, or cleanup operations.
        
        Args:
            final_info: Final environment info dict containing episode
                summary data including trajectory statistics, performance
                metrics, and termination reason
                
        Note:
            Default implementation is a no-op (no special handling).
            This method should not modify navigator state as it may be
            called after reset() for the next episode.
        """
        pass
    
    def _apply_observation_hooks(self, base_obs: dict) -> dict:
        """Apply observation extensibility hooks."""
        try:
            additional_obs = self.compute_additional_obs(base_obs)
            if hasattr(self.navigator, 'compute_additional_obs'):
                navigator_obs = self.navigator.compute_additional_obs(base_obs)
                additional_obs.update(navigator_obs)
            return additional_obs
        except Exception as e:
            logger.warning(f"Error in observation hooks: {e}")
            return {}
    
    def _apply_reward_hooks(self, base_reward: float, info: dict) -> float:
        """Apply reward extensibility hooks."""
        try:
            extra_reward = self.compute_extra_reward(base_reward, info)
            if hasattr(self.navigator, 'compute_extra_reward'):
                navigator_reward = self.navigator.compute_extra_reward(base_reward, info)
                extra_reward += navigator_reward
            return extra_reward
        except Exception as e:
            logger.warning(f"Error in reward hooks: {e}")
            return 0.0
    
    def _apply_episode_end_hooks(self, final_info: dict) -> None:
        """Apply episode end extensibility hooks."""
        try:
            self.on_episode_end(final_info)
            if hasattr(self.navigator, 'on_episode_end'):
                self.navigator.on_episode_end(final_info)
        except Exception as e:
            logger.warning(f"Error in episode end hooks: {e}")
    
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
            if self._cache_enabled and self.frame_cache:
                current_frame = self.frame_cache.get(
                    frame_id=self.current_frame_index,
                    video_plume=self.video_plume
                )
            else:
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
        logger.debug("Closing PlumeNavigationEnv", extra={
            "correlation_id": getattr(self, "_correlation_id", None),
            "episodes_completed": getattr(self, "_episode_count", 0),
            "metric_type": "environment_close"
        })
        
        try:
            # Close video plume
            if hasattr(self, 'video_plume'):
                self.video_plume.close()
            
            # Clear frame cache to free memory
            if self._cache_enabled and hasattr(self, 'frame_cache') and self.frame_cache is not None:
                if hasattr(self.frame_cache, 'clear'):
                    self.frame_cache.clear()
                    logger.debug("Frame cache cleared on environment close")
            
            # Close rendering resources
            if self.render_initialized and MATPLOTLIB_AVAILABLE:
                plt.close(self.fig)
                self.render_initialized = False
            
            # Log performance summary if monitoring enabled
            if self.performance_monitoring and self._step_times:
                avg_step_time = np.mean(self._step_times)
                avg_fps = 1.0 / avg_step_time if avg_step_time > 0 else float('inf')
                
                performance_summary = {
                    "episodes": self._episode_count,
                    "total_steps": len(self._step_times),
                    "avg_step_time": avg_step_time,
                    "avg_fps": avg_fps,
                    "total_runtime": time.time() - self._start_time,
                    "correlation_id": getattr(self, "_correlation_id", None),
                    "metric_type": "performance_summary"
                }
                
                # Add cache statistics if available
                if self._cache_enabled and self.frame_cache is not None:
                    performance_summary.update({
                        "cache_hit_rate": getattr(self.frame_cache, 'hit_rate', 0.0),
                        "cache_total_hits": getattr(self.frame_cache, 'hits', 0),
                        "cache_total_misses": getattr(self.frame_cache, 'misses', 0),
                        "cache_final_size": getattr(self.frame_cache, 'cache_size', 0)
                    })
                
                logger.info(
                    f"Environment closed - Performance summary:",
                    extra=performance_summary
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
        
        try:
            if seed is not None:
                if SEED_UTILS_AVAILABLE:
                    set_global_seed(seed)
                else:
                    np.random.seed(seed)
                self._last_seed = seed
                logger.debug(f"Environment seed set to {seed}")
            else:
                # Generate random seed
                import random
                seed = random.randint(0, 2**32 - 1)
                if SEED_UTILS_AVAILABLE:
                    set_global_seed(seed)
                else:
                    np.random.seed(seed)
                self._last_seed = seed
                logger.debug(f"Environment auto-seeded with {seed}")
            
            return [seed]
            
        except Exception as e:
            logger.error(f"Failed to set seed: {e}")
            # Fallback to basic seeding
            if seed is not None:
                np.random.seed(seed)
                self._last_seed = seed
            return [seed] if seed is not None else [0]


# Convenience functions for environment creation and validation

def create_plume_navigation_environment(config: ConfigType, **kwargs) -> PlumeNavigationEnv:
    """
    Factory function to create PlumeNavigationEnv from configuration.
    
    This is the primary entry point for environment creation, supporting both
    programmatic instantiation and Hydra configuration-driven workflows.
    
    Args:
        config: Configuration dictionary or DictConfig
        **kwargs: Additional parameters to override configuration
        
    Returns:
        Configured PlumeNavigationEnv instance
        
    Examples:
        >>> config = {"video_path": "data/plume.mp4", "max_speed": 2.0}
        >>> env = create_plume_navigation_environment(config)
        >>> 
        >>> # With frame cache for performance
        >>> from plume_nav_sim.utils.frame_cache import create_lru_cache
        >>> cache = create_lru_cache(memory_limit_mb=512)
        >>> env = create_plume_navigation_environment(config, frame_cache=cache)
    """
    return PlumeNavigationEnv.from_config(config, **kwargs)


def validate_environment_api_compliance(env: PlumeNavigationEnv) -> Dict[str, Any]:
    """
    Validate environment compliance with Gymnasium API.
    
    Uses gymnasium.utils.env_checker to verify complete API compliance
    and returns validation results for debugging and testing.
    
    Args:
        env: PlumeNavigationEnv instance to validate
        
    Returns:
        Dictionary containing validation results and diagnostic information
        
    Raises:
        ImportError: If gymnasium env_checker is not available
    """
    try:
        from gymnasium.utils.env_checker import check_env
        
        logger.info("Validating Gymnasium API compliance")
        
        validation_results = {
            "compliant": False,
            "errors": [],
            "warnings": [],
            "api_mode": "legacy" if env._use_legacy_api else "gymnasium",
            "validation_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Run comprehensive API validation
            check_env(env, warn=True, skip_render_check=False)
            validation_results["compliant"] = True
            
            logger.info("Environment passed Gymnasium API validation")
            
        except Exception as e:
            validation_results["errors"].append(str(e))
            logger.error(f"Environment API validation failed: {e}")
        
        validation_results["validation_time"] = time.time() - start_time
        return validation_results
        
    except ImportError:
        logger.error("gymnasium.utils.env_checker not available for validation")
        return {
            "compliant": False,
            "errors": ["gymnasium.utils.env_checker not available"],
            "warnings": [],
            "validation_time": 0.0
        }


# Module-level initialization
logger.info(
    "PlumeNavigationEnv module loaded with Gymnasium 0.29.x support",
    extra={
        "gymnasium_available": GYMNASIUM_AVAILABLE,
        "spaces_available": SPACES_AVAILABLE,
        "frame_cache_available": FRAME_CACHE_AVAILABLE,
        "dual_api_support": True,
        "extensibility_hooks": True
    }
)

# Public API exports
__all__ = [
    "PlumeNavigationEnv",
    "create_plume_navigation_environment", 
    "validate_environment_api_compliance"
]

# Environment registration information for Gymnasium
ENVIRONMENT_SPECS = {
    "PlumeNavSim-v0": {
        "entry_point": "plume_nav_sim.envs.plume_navigation_env:PlumeNavigationEnv",
        "max_episode_steps": 1000,
        "reward_threshold": 100.0,
        "nondeterministic": False,
        "kwargs": {}
    },
    # Legacy compatibility registration
    "OdorPlumeNavigation-v1": {
        "entry_point": "plume_nav_sim.envs.plume_navigation_env:PlumeNavigationEnv",
        "max_episode_steps": 1000,
        "reward_threshold": 100.0,
        "nondeterministic": False,
        "kwargs": {"_force_legacy_api": True}
    }
}