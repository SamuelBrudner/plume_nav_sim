"""
Gymnasium-compliant action and observation space definitions for odor plume navigation.

This module provides standardized space definitions essential for RL training workflows
and ensures compatibility with the broader Gymnasium ecosystem. It defines continuous
action spaces for agent control and multi-modal observation spaces for environmental
perception, supporting both single-sensor and multi-sensor configurations.

The space definitions integrate seamlessly with the existing NavigatorProtocol and
VideoPlume components while maintaining full compatibility with stable-baselines3
and other modern reinforcement learning frameworks.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np

try:
    import gymnasium as gym
    from gymnasium.spaces import Box, Dict as DictSpace
    GYMNASIUM_AVAILABLE = True
except ImportError:
    # Fallback for environments without Gymnasium
    GYMNASIUM_AVAILABLE = False
    # Create mock classes to prevent import errors
    class Box:
        def __init__(self, *args, **kwargs):
            pass
    
    class DictSpace:
        def __init__(self, *args, **kwargs):
            pass
    
    gym = None

# Type aliases for enhanced clarity and IDE support
ActionType = np.ndarray
"""Type alias for action arrays with shape (2,) containing [speed, angular_velocity]."""

ObservationType = Dict[str, np.ndarray]
"""Type alias for observation dictionaries containing multi-modal sensor data."""

BoundsType = Union[Tuple[float, float], np.ndarray]
"""Type alias for bounds specification - supports (low, high) tuples or numpy arrays."""


class ActionSpace:
    """
    Defines Gymnasium-compliant action space for continuous agent control.
    
    The action space represents continuous control actions using gymnasium.spaces.Box
    with two components: linear speed and angular velocity. This provides direct
    compatibility with NavigatorProtocol control interfaces while ensuring
    standardized RL algorithm integration.
    
    Action Components:
        - speed: Linear velocity in units per time step [0.0, max_speed]
        - angular_velocity: Rotational velocity in degrees per second [-max_angular_vel, +max_angular_vel]
    
    Design Principles:
        - Continuous control for smooth navigation policies
        - Direct mapping to NavigatorProtocol speed/angular_velocity properties
        - Configurable bounds supporting different environment scales
        - Compatible with all major RL algorithms (PPO, SAC, TD3)
    
    Examples:
        Basic action space creation:
        >>> action_space = ActionSpace.create()
        >>> action = action_space.sample()  # Random valid action
        >>> print(action)  # [speed, angular_velocity]
        
        Custom bounds configuration:
        >>> action_space = ActionSpace.create(
        ...     max_speed=2.5,
        ...     max_angular_velocity=180.0
        ... )
        
        Action validation:
        >>> action = np.array([1.5, 45.0])
        >>> is_valid = action_space.contains(action)
    """
    
    @staticmethod
    def create(
        max_speed: float = 2.0,
        max_angular_velocity: float = 90.0,
        min_speed: float = 0.0,
        dtype: np.dtype = np.float32
    ) -> Box:
        """
        Create Gymnasium Box action space for continuous control.
        
        Args:
            max_speed: Maximum allowed linear speed (default: 2.0)
            max_angular_velocity: Maximum angular velocity magnitude in degrees/sec (default: 90.0)
            min_speed: Minimum allowed linear speed (default: 0.0, non-negative)
            dtype: NumPy data type for action values (default: np.float32)
            
        Returns:
            gymnasium.spaces.Box: Action space with shape (2,) for [speed, angular_velocity]
            
        Notes:
            - Speed is constrained to [min_speed, max_speed] for forward-only movement
            - Angular velocity is symmetric around zero: [-max_angular_velocity, +max_angular_velocity]
            - Values are automatically clipped during environment step() execution
            - Compatible with continuous control algorithms (PPO, SAC, TD3)
            
        Raises:
            ValueError: If max_speed <= min_speed or max_angular_velocity <= 0
            ImportError: If gymnasium is not available
            
        Examples:
            Standard configuration:
            >>> action_space = ActionSpace.create()
            >>> print(action_space.shape)  # (2,)
            >>> print(action_space.low)    # [0.0, -90.0]
            >>> print(action_space.high)   # [2.0, 90.0]
            
            High-speed configuration:
            >>> action_space = ActionSpace.create(
            ...     max_speed=5.0,
            ...     max_angular_velocity=180.0
            ... )
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for action space creation. "
                "Install with: pip install 'odor_plume_nav[rl]'"
            )
        
        if max_speed <= min_speed:
            raise ValueError(f"max_speed ({max_speed}) must be greater than min_speed ({min_speed})")
        
        if max_angular_velocity <= 0:
            raise ValueError(f"max_angular_velocity ({max_angular_velocity}) must be positive")
        
        # Define action bounds: [speed, angular_velocity]
        low = np.array([min_speed, -max_angular_velocity], dtype=dtype)
        high = np.array([max_speed, max_angular_velocity], dtype=dtype)
        
        return Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=dtype,
            seed=None
        )
    
    @staticmethod
    def validate_action(action: ActionType, action_space: Box) -> bool:
        """
        Validate action against action space constraints.
        
        Args:
            action: Action array with shape (2,) containing [speed, angular_velocity]
            action_space: Gymnasium Box action space for validation
            
        Returns:
            bool: True if action is valid, False otherwise
            
        Examples:
            >>> action_space = ActionSpace.create()
            >>> valid_action = np.array([1.0, 30.0])
            >>> invalid_action = np.array([3.0, 200.0])  # Exceeds bounds
            >>> ActionSpace.validate_action(valid_action, action_space)    # True
            >>> ActionSpace.validate_action(invalid_action, action_space)  # False
        """
        return action_space.contains(action)
    
    @staticmethod
    def clip_action(action: ActionType, action_space: Box) -> ActionType:
        """
        Clip action to valid action space bounds.
        
        Args:
            action: Action array that may exceed bounds
            action_space: Gymnasium Box action space defining valid bounds
            
        Returns:
            np.ndarray: Clipped action within valid bounds
            
        Examples:
            >>> action_space = ActionSpace.create(max_speed=2.0, max_angular_velocity=90.0)
            >>> excessive_action = np.array([5.0, 150.0])
            >>> clipped_action = ActionSpace.clip_action(excessive_action, action_space)
            >>> print(clipped_action)  # [2.0, 90.0]
        """
        return np.clip(action, action_space.low, action_space.high)


class ObservationSpace:
    """
    Defines Gymnasium-compliant observation space for multi-modal environmental perception.
    
    The observation space represents sensor data using gymnasium.spaces.Dict with multiple
    components for comprehensive environmental awareness. This supports both basic single-sensor
    configurations and advanced multi-sensor arrays for sophisticated navigation strategies.
    
    Observation Components:
        - odor_concentration: Scalar odor intensity at agent position [0.0, 1.0]
        - agent_position: Agent coordinates as [x, y] in environment frame
        - agent_orientation: Agent heading in degrees [0.0, 360.0)
        - multi_sensor_readings: Optional array of sensor values for bilateral/triangular sensing
    
    Design Principles:
        - Multi-modal observations for rich environmental perception
        - Configurable sensor layouts (single, bilateral, triangular, custom)
        - Direct integration with NavigatorProtocol state and VideoPlume data
        - Extensible design supporting additional sensor modalities
    
    Examples:
        Basic observation space:
        >>> obs_space = ObservationSpace.create(env_width=640, env_height=480)
        >>> obs = obs_space.sample()
        >>> print(obs.keys())  # ['odor_concentration', 'agent_position', 'agent_orientation']
        
        Multi-sensor configuration:
        >>> obs_space = ObservationSpace.create(
        ...     env_width=640,
        ...     env_height=480,
        ...     num_sensors=3,
        ...     include_multi_sensor=True
        ... )
        >>> obs = obs_space.sample()
        >>> print(obs['multi_sensor_readings'].shape)  # (3,)
    """
    
    @staticmethod
    def create(
        env_width: float,
        env_height: float,
        num_sensors: int = 2,
        include_multi_sensor: bool = False,
        position_bounds: Optional[BoundsType] = None,
        dtype: np.dtype = np.float32
    ) -> DictSpace:
        """
        Create Gymnasium Dict observation space for multi-modal sensing.
        
        Args:
            env_width: Environment width in pixels or units
            env_height: Environment height in pixels or units
            num_sensors: Number of additional sensors for multi-sensor readings (default: 2)
            include_multi_sensor: Whether to include multi-sensor readings in observations (default: False)
            position_bounds: Optional custom position bounds as ((x_min, x_max), (y_min, y_max))
            dtype: NumPy data type for observation values (default: np.float32)
            
        Returns:
            gymnasium.spaces.Dict: Observation space containing all sensor modalities
            
        Notes:
            - Odor concentration normalized to [0.0, 1.0] representing intensity
            - Agent position bounded by environment dimensions
            - Agent orientation in degrees [0.0, 360.0) following navigation conventions
            - Multi-sensor readings optional for advanced sensing strategies
            
        Raises:
            ValueError: If environment dimensions are invalid or num_sensors <= 0
            ImportError: If gymnasium is not available
            
        Examples:
            Standard video environment:
            >>> obs_space = ObservationSpace.create(
            ...     env_width=640,
            ...     env_height=480
            ... )
            
            Multi-sensor bilateral configuration:
            >>> obs_space = ObservationSpace.create(
            ...     env_width=800,
            ...     env_height=600,
            ...     num_sensors=2,
            ...     include_multi_sensor=True
            ... )
            
            Custom position bounds:
            >>> obs_space = ObservationSpace.create(
            ...     env_width=1000,
            ...     env_height=800,
            ...     position_bounds=((10, 990), (10, 790))  # 10-pixel border
            ... )
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gymnasium is required for observation space creation. "
                "Install with: pip install 'odor_plume_nav[rl]'"
            )
        
        if env_width <= 0 or env_height <= 0:
            raise ValueError(f"Environment dimensions must be positive: width={env_width}, height={env_height}")
        
        if num_sensors <= 0:
            raise ValueError(f"num_sensors must be positive: {num_sensors}")
        
        # Define position bounds
        if position_bounds is not None:
            (x_min, x_max), (y_min, y_max) = position_bounds
            if x_min >= x_max or y_min >= y_max:
                raise ValueError(f"Invalid position bounds: {position_bounds}")
        else:
            x_min, x_max = 0.0, float(env_width)
            y_min, y_max = 0.0, float(env_height)
        
        # Core observation components
        spaces = {
            # Odor concentration from VideoPlume sensor sampling
            'odor_concentration': Box(
                low=0.0,
                high=1.0,
                shape=(),
                dtype=dtype,
                seed=None
            ),
            
            # Agent position in environment coordinates
            'agent_position': Box(
                low=np.array([x_min, y_min], dtype=dtype),
                high=np.array([x_max, y_max], dtype=dtype),
                shape=(2,),
                dtype=dtype,
                seed=None
            ),
            
            # Agent orientation in degrees [0, 360)
            'agent_orientation': Box(
                low=0.0,
                high=360.0,
                shape=(),
                dtype=dtype,
                seed=None
            )
        }
        
        # Optional multi-sensor readings for advanced sensing strategies
        if include_multi_sensor:
            spaces['multi_sensor_readings'] = Box(
                low=0.0,
                high=1.0,
                shape=(num_sensors,),
                dtype=dtype,
                seed=None
            )
        
        return DictSpace(spaces, seed=None)
    
    @staticmethod
    def create_single_sensor(
        env_width: float,
        env_height: float,
        dtype: np.dtype = np.float32
    ) -> DictSpace:
        """
        Create observation space for single-sensor configuration.
        
        Convenience method for basic single-sensor setups without multi-sensor arrays.
        Equivalent to create() with include_multi_sensor=False.
        
        Args:
            env_width: Environment width in pixels or units
            env_height: Environment height in pixels or units
            dtype: NumPy data type for observation values (default: np.float32)
            
        Returns:
            gymnasium.spaces.Dict: Basic observation space with core components only
            
        Examples:
            >>> obs_space = ObservationSpace.create_single_sensor(640, 480)
            >>> obs = obs_space.sample()
            >>> assert 'multi_sensor_readings' not in obs
        """
        return ObservationSpace.create(
            env_width=env_width,
            env_height=env_height,
            include_multi_sensor=False,
            dtype=dtype
        )
    
    @staticmethod
    def create_bilateral_sensor(
        env_width: float,
        env_height: float,
        dtype: np.dtype = np.float32
    ) -> DictSpace:
        """
        Create observation space for bilateral sensor configuration.
        
        Convenience method for two-sensor configurations mimicking biological
        antennae or bilateral sensor arrangements common in chemotaxis research.
        
        Args:
            env_width: Environment width in pixels or units
            env_height: Environment height in pixels or units
            dtype: NumPy data type for observation values (default: np.float32)
            
        Returns:
            gymnasium.spaces.Dict: Observation space with bilateral sensors
            
        Examples:
            >>> obs_space = ObservationSpace.create_bilateral_sensor(640, 480)
            >>> obs = obs_space.sample()
            >>> assert obs['multi_sensor_readings'].shape == (2,)
        """
        return ObservationSpace.create(
            env_width=env_width,
            env_height=env_height,
            num_sensors=2,
            include_multi_sensor=True,
            dtype=dtype
        )
    
    @staticmethod
    def create_triangular_sensor(
        env_width: float,
        env_height: float,
        dtype: np.dtype = np.float32
    ) -> DictSpace:
        """
        Create observation space for triangular sensor configuration.
        
        Convenience method for three-sensor triangular arrangements providing
        enhanced directional sensing capabilities for advanced navigation policies.
        
        Args:
            env_width: Environment width in pixels or units
            env_height: Environment height in pixels or units
            dtype: NumPy data type for observation values (default: np.float32)
            
        Returns:
            gymnasium.spaces.Dict: Observation space with triangular sensors
            
        Examples:
            >>> obs_space = ObservationSpace.create_triangular_sensor(640, 480)
            >>> obs = obs_space.sample()
            >>> assert obs['multi_sensor_readings'].shape == (3,)
        """
        return ObservationSpace.create(
            env_width=env_width,
            env_height=env_height,
            num_sensors=3,
            include_multi_sensor=True,
            dtype=dtype
        )
    
    @staticmethod
    def validate_observation(observation: ObservationType, observation_space: DictSpace) -> bool:
        """
        Validate observation against observation space constraints.
        
        Args:
            observation: Observation dictionary containing sensor data
            observation_space: Gymnasium Dict observation space for validation
            
        Returns:
            bool: True if observation is valid, False otherwise
            
        Examples:
            >>> obs_space = ObservationSpace.create(640, 480)
            >>> valid_obs = {
            ...     'odor_concentration': np.array(0.5),
            ...     'agent_position': np.array([320.0, 240.0]),
            ...     'agent_orientation': np.array(45.0)
            ... }
            >>> ObservationSpace.validate_observation(valid_obs, obs_space)  # True
        """
        return observation_space.contains(observation)
    
    @staticmethod
    def normalize_observation(
        observation: ObservationType,
        observation_space: DictSpace
    ) -> ObservationType:
        """
        Normalize observation components to standard ranges.
        
        Applies normalization to observation components for improved RL training
        stability. Position values are normalized to [0, 1] based on environment
        bounds, while other components maintain their natural scales.
        
        Args:
            observation: Raw observation dictionary
            observation_space: Gymnasium Dict observation space defining bounds
            
        Returns:
            Dict[str, np.ndarray]: Normalized observation dictionary
            
        Examples:
            >>> obs_space = ObservationSpace.create(640, 480)
            >>> raw_obs = {
            ...     'odor_concentration': np.array(0.8),
            ...     'agent_position': np.array([320.0, 240.0]),
            ...     'agent_orientation': np.array(90.0)
            ... }
            >>> norm_obs = ObservationSpace.normalize_observation(raw_obs, obs_space)
            >>> print(norm_obs['agent_position'])  # [0.5, 0.5] (center of environment)
        """
        normalized = {}
        
        for key, value in observation.items():
            if key == 'agent_position':
                # Normalize position to [0, 1] based on environment bounds
                space = observation_space.spaces[key]
                position_range = space.high - space.low
                normalized[key] = (value - space.low) / position_range
            elif key == 'agent_orientation':
                # Normalize orientation to [0, 1] based on [0, 360) range
                normalized[key] = value / 360.0
            else:
                # Keep other components in their natural ranges
                normalized[key] = value.copy()
        
        return normalized


class SpaceFactory:
    """
    Factory class for creating standardized action and observation spaces.
    
    Provides a unified interface for space creation with common configurations
    and validation. Supports both programmatic creation and configuration-driven
    instantiation for research workflow integration.
    
    Examples:
        Standard environment setup:
        >>> spaces = SpaceFactory.create_standard_spaces(
        ...     env_width=640,
        ...     env_height=480,
        ...     max_speed=2.0
        ... )
        >>> action_space, obs_space = spaces
        
        Multi-sensor research configuration:
        >>> spaces = SpaceFactory.create_research_spaces(
        ...     env_width=800,
        ...     env_height=600,
        ...     sensor_config='triangular'
        ... )
    """
    
    @staticmethod
    def create_standard_spaces(
        env_width: float,
        env_height: float,
        max_speed: float = 2.0,
        max_angular_velocity: float = 90.0,
        include_multi_sensor: bool = False,
        num_sensors: int = 2
    ) -> Tuple[Box, DictSpace]:
        """
        Create standard action and observation spaces for typical experiments.
        
        Args:
            env_width: Environment width in pixels or units
            env_height: Environment height in pixels or units
            max_speed: Maximum agent speed (default: 2.0)
            max_angular_velocity: Maximum angular velocity in degrees/sec (default: 90.0)
            include_multi_sensor: Whether to include multi-sensor observations (default: False)
            num_sensors: Number of additional sensors if multi-sensor enabled (default: 2)
            
        Returns:
            Tuple[Box, DictSpace]: (action_space, observation_space)
            
        Examples:
            >>> action_space, obs_space = SpaceFactory.create_standard_spaces(640, 480)
            >>> print(action_space.shape)  # (2,)
            >>> print(list(obs_space.spaces.keys()))  # Core observation components
        """
        action_space = ActionSpace.create(
            max_speed=max_speed,
            max_angular_velocity=max_angular_velocity
        )
        
        observation_space = ObservationSpace.create(
            env_width=env_width,
            env_height=env_height,
            num_sensors=num_sensors,
            include_multi_sensor=include_multi_sensor
        )
        
        return action_space, observation_space
    
    @staticmethod
    def create_research_spaces(
        env_width: float,
        env_height: float,
        sensor_config: str = 'single',
        speed_config: str = 'standard'
    ) -> Tuple[Box, DictSpace]:
        """
        Create research-oriented spaces with predefined configurations.
        
        Args:
            env_width: Environment width in pixels or units
            env_height: Environment height in pixels or units
            sensor_config: Sensor configuration ('single', 'bilateral', 'triangular', 'custom')
            speed_config: Speed configuration ('slow', 'standard', 'fast')
            
        Returns:
            Tuple[Box, DictSpace]: (action_space, observation_space)
            
        Raises:
            ValueError: If configuration strings are invalid
            
        Examples:
            >>> # Bilateral sensing with fast movement
            >>> action_space, obs_space = SpaceFactory.create_research_spaces(
            ...     env_width=800,
            ...     env_height=600,
            ...     sensor_config='bilateral',
            ...     speed_config='fast'
            ... )
        """
        # Speed configuration mapping
        speed_configs = {
            'slow': {'max_speed': 1.0, 'max_angular_velocity': 45.0},
            'standard': {'max_speed': 2.0, 'max_angular_velocity': 90.0},
            'fast': {'max_speed': 4.0, 'max_angular_velocity': 180.0}
        }
        
        if speed_config not in speed_configs:
            raise ValueError(f"Invalid speed_config: {speed_config}. "
                           f"Valid options: {list(speed_configs.keys())}")
        
        speed_params = speed_configs[speed_config]
        action_space = ActionSpace.create(**speed_params)
        
        # Sensor configuration mapping
        if sensor_config == 'single':
            observation_space = ObservationSpace.create_single_sensor(env_width, env_height)
        elif sensor_config == 'bilateral':
            observation_space = ObservationSpace.create_bilateral_sensor(env_width, env_height)
        elif sensor_config == 'triangular':
            observation_space = ObservationSpace.create_triangular_sensor(env_width, env_height)
        else:
            raise ValueError(f"Invalid sensor_config: {sensor_config}. "
                           f"Valid options: 'single', 'bilateral', 'triangular'")
        
        return action_space, observation_space
    
    @staticmethod
    def validate_spaces(action_space: Box, observation_space: DictSpace) -> bool:
        """
        Validate that action and observation spaces are properly configured.
        
        Args:
            action_space: Gymnasium Box action space
            observation_space: Gymnasium Dict observation space
            
        Returns:
            bool: True if spaces are valid, False otherwise
            
        Examples:
            >>> action_space, obs_space = SpaceFactory.create_standard_spaces(640, 480)
            >>> is_valid = SpaceFactory.validate_spaces(action_space, obs_space)
            >>> assert is_valid
        """
        try:
            # Validate action space structure
            if not isinstance(action_space, Box):
                return False
            if action_space.shape != (2,):
                return False
            if not np.all(action_space.low <= action_space.high):
                return False
            
            # Validate observation space structure
            if not isinstance(observation_space, DictSpace):
                return False
            
            required_keys = {'odor_concentration', 'agent_position', 'agent_orientation'}
            if not required_keys.issubset(observation_space.spaces.keys()):
                return False
            
            # Validate individual observation components
            odor_space = observation_space.spaces['odor_concentration']
            if not isinstance(odor_space, Box) or odor_space.shape != ():
                return False
            
            position_space = observation_space.spaces['agent_position']
            if not isinstance(position_space, Box) or position_space.shape != (2,):
                return False
            
            orientation_space = observation_space.spaces['agent_orientation']
            if not isinstance(orientation_space, Box) or orientation_space.shape != ():
                return False
            
            return True
            
        except Exception:
            return False


# Public API exports
__all__ = [
    "ActionSpace",
    "ObservationSpace", 
    "SpaceFactory",
    "ActionType",
    "ObservationType",
    "BoundsType",
    "Box",
    "DictSpace",
    "GYMNASIUM_AVAILABLE"
]