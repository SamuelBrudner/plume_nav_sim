"""
Gymnasium-Compliant Space Factory and Validation Module for Plume Navigation Simulation.

This module provides comprehensive space creation, validation, and compatibility utilities
for the plume_nav_sim environment, supporting the migration from OpenAI Gym 0.26 to 
Gymnasium 0.29.x while maintaining backward compatibility through intelligent conversion
utilities and type-safe space definitions.

Key Features:
- Enhanced space factory methods with validated bounds and dtypes for Gymnasium 0.29.x compliance
- Type-safe space creation utilities ensuring consistent observation and action space definitions
- Comprehensive space validation methods for Gymnasium compliance checking
- Step-return format validation and conversion utilities supporting dual API compatibility
- Integration with structured logging system for comprehensive observability and debugging

Space Factory Components:
- ActionSpaceFactory: Creates standardized action spaces (Box, Discrete) with domain constraints
- ObservationSpaceFactory: Constructs multi-modal observation spaces with proper type validation
- SpaceValidator: Comprehensive validation utilities for Gymnasium API compliance
- ReturnFormatConverter: Handles conversion between 4-tuple (legacy) and 5-tuple (modern) formats

Gymnasium Compliance Features:
- Full compatibility with gymnasium.utils.env_checker validation
- Proper dtype specification and bounds validation for all spaces
- Support for Dict, Box, and Discrete space types with nested validation
- Thread-safe space operations supporting concurrent environment instances

Performance Optimizations:
- Zero-copy space validation where possible
- Cached space instances for repeated environment creation
- Minimal overhead conversion utilities maintaining <10ms step execution requirements
- Efficient space sampling and validation operations

Example Usage:
    >>> # Create action space for continuous control
    >>> action_space = ActionSpaceFactory.create_continuous_action_space()
    >>> print(action_space)  # Box(2,) with proper bounds
    
    >>> # Create multi-modal observation space
    >>> obs_space = ObservationSpaceFactory.create_navigation_observation_space()
    >>> print(obs_space)  # Dict space with odor, position, velocity components
    
    >>> # Validate space for Gymnasium compliance
    >>> is_valid = SpaceValidator.validate_gymnasium_compliance(action_space)
    >>> print(is_valid)  # True if compliant
    
    >>> # Convert between return formats
    >>> legacy_return = (obs, reward, done, info)
    >>> modern_return = ReturnFormatConverter.to_gymnasium_format(legacy_return)
    >>> print(modern_return)  # (obs, reward, terminated, truncated, info)
    
    >>> # Type-safe space creation with validation
    >>> validated_space = SpaceFactory.create_validated_box_space(
    ...     low=-1.0, high=1.0, shape=(2,), dtype=np.float32
    ... )
"""

import warnings
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional, List, Literal, Type, cast
from pathlib import Path
import threading
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

try:
    import gymnasium as gym
    from gymnasium.spaces import Space, Box, Discrete, Dict as DictSpace, MultiBinary, MultiDiscrete
    GYMNASIUM_AVAILABLE = True
except ImportError:
    # Fallback for environments where gymnasium isn't available yet
    try:
        import gym
        from gym.spaces import Space, Box, Discrete, Dict as DictSpace, MultiBinary, MultiDiscrete
        GYMNASIUM_AVAILABLE = False
        warnings.warn(
            "Gymnasium not available, falling back to legacy gym. "
            "Please install gymnasium>=0.29.0 for full compatibility.",
            UserWarning,
            stacklevel=2
        )
    except ImportError:
        raise ImportError(
            "Neither gymnasium nor gym are available. "
            "Please install gymnasium>=0.29.0 for full functionality."
        )

# Import logging utilities for comprehensive observability
from plume_nav_sim.utils.logging_setup import (
    get_logger, 
    correlation_context,
    log_legacy_api_deprecation
)

# Import sensor protocol for dynamic space construction
try:
    from ..core.protocols import SensorProtocol
    SENSOR_PROTOCOL_AVAILABLE = True
except ImportError:
    # Handle case where protocols don't exist yet
    SensorProtocol = object
    SENSOR_PROTOCOL_AVAILABLE = False

# Import sensor implementations for type checking
try:
    from plume_nav_sim.core.sensors.binary_sensor import BinarySensor
    from plume_nav_sim.core.sensors.concentration_sensor import ConcentrationSensor
    from plume_nav_sim.core.sensors.gradient_sensor import GradientSensor
    SENSOR_IMPLEMENTATIONS_AVAILABLE = True
except ImportError:
    # Handle case where sensor implementations don't exist yet
    BinarySensor = None
    ConcentrationSensor = None
    GradientSensor = None
    SENSOR_IMPLEMENTATIONS_AVAILABLE = False

# Module logger for structured logging integration
logger = get_logger(__name__)


@dataclass
class SpaceDefinition:
    """
    Configuration dataclass defining space parameters with validation and metadata.
    
    Provides comprehensive space configuration supporting both Box and Discrete space
    types with domain-specific validation, performance optimization hints, and 
    integration metadata for the plume navigation simulation environment.
    
    Attributes:
        space_type: Type of gymnasium space to create
        shape: Shape tuple for Box spaces or None for Discrete
        low: Lower bounds for Box spaces (scalar or array-like)
        high: Upper bounds for Box spaces (scalar or array-like)
        n: Number of discrete actions for Discrete spaces
        dtype: NumPy dtype for space elements
        name: Human-readable name for the space
        description: Detailed description of space semantics
        validation_enabled: Enable runtime validation (affects performance)
        bounds_checking: Enable bounds validation for Box spaces
        metadata: Additional space metadata for observability
    """
    
    # Core space parameters
    space_type: Literal["Box", "Discrete", "Dict", "MultiBinary", "MultiDiscrete"]
    shape: Optional[Tuple[int, ...]] = None
    low: Optional[Union[float, np.ndarray]] = None
    high: Optional[Union[float, np.ndarray]] = None
    n: Optional[int] = None  # For Discrete spaces
    dtype: np.dtype = field(default_factory=lambda: np.float32)
    
    # Metadata and validation
    name: str = "unnamed_space"
    description: str = "Space definition for plume navigation environment"
    validation_enabled: bool = True
    bounds_checking: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation for space definition consistency."""
        if self.space_type == "Box":
            if self.shape is None:
                raise ValueError("Box spaces require shape specification")
            if self.low is None or self.high is None:
                raise ValueError("Box spaces require low and high bounds")
        elif self.space_type == "Discrete":
            if self.n is None:
                raise ValueError("Discrete spaces require n specification")
            if self.n <= 0:
                raise ValueError("Discrete space n must be positive")
        
        # Validate dtype compatibility
        if not isinstance(self.dtype, np.dtype):
            try:
                self.dtype = np.dtype(self.dtype)
            except TypeError as e:
                raise ValueError(f"Invalid dtype specification: {e}")


@dataclass
class SensorSpaceDefinition:
    """
    Configuration for sensor-specific observation space components.
    
    Defines how different sensor types contribute to the overall observation space,
    enabling dynamic space construction based on active sensor configurations.
    
    Attributes:
        sensor_type: Type of sensor (binary, concentration, gradient)
        sensor_id: Unique identifier for the sensor instance
        output_shape: Expected shape of sensor output
        value_range: Range of sensor output values
        dtype: Data type for sensor outputs
        metadata: Additional sensor-specific metadata
    """
    sensor_type: str
    sensor_id: str
    output_shape: Tuple[int, ...]
    value_range: Tuple[float, float]
    dtype: np.dtype = np.float32
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WindDataConfig:
    """
    Configuration for wind data integration in observation spaces.
    
    Defines how wind field information is incorporated into observations
    when environmental dynamics are enabled.
    
    Attributes:
        enabled: Whether wind data should be included in observations
        velocity_components: Number of wind velocity components (2D or 3D)
        velocity_range: Range of wind velocity values
        include_direction: Include wind direction as separate observation
        include_magnitude: Include wind magnitude as separate observation
        temporal_history: Number of previous wind states to include
        coordinate_system: Wind coordinate system ('cartesian' or 'polar')
    """
    enabled: bool = False
    velocity_components: int = 2
    velocity_range: Tuple[float, float] = (-10.0, 10.0)
    include_direction: bool = False
    include_magnitude: bool = False
    temporal_history: int = 0
    coordinate_system: str = "cartesian"
    
    def __post_init__(self):
        """Validate wind data configuration."""
        if self.velocity_components not in [2, 3]:
            raise ValueError("velocity_components must be 2 or 3")
        if self.coordinate_system not in ["cartesian", "polar"]:
            raise ValueError("coordinate_system must be 'cartesian' or 'polar'")
        if self.temporal_history < 0:
            raise ValueError("temporal_history must be non-negative")


class SpaceValidator:
    """
    Comprehensive space validation utilities for Gymnasium compliance checking.
    
    Provides static methods for validating space definitions, checking Gymnasium
    API compliance, and ensuring type safety across the plume navigation simulation
    environment. Supports both runtime validation and development-time checking.
    """
    
    @staticmethod
    def validate_gymnasium_compliance(space: Space) -> bool:
        """
        Validate space for full Gymnasium 0.29.x API compliance.
        
        Performs comprehensive validation including type checking, bounds validation,
        dtype consistency, and API method availability. Designed to catch common
        compatibility issues during the migration from OpenAI Gym to Gymnasium.
        
        Args:
            space: Gymnasium space instance to validate
            
        Returns:
            True if space is fully Gymnasium compliant
            
        Raises:
            ValueError: If space fails critical compliance checks
            
        Example:
            >>> action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
            >>> is_valid = SpaceValidator.validate_gymnasium_compliance(action_space)
            >>> print(is_valid)  # True for valid spaces
        """
        try:
            with correlation_context("space_validation", space_type=type(space).__name__) as ctx:
                logger.debug(f"Validating Gymnasium compliance for {type(space).__name__}")
                
                # Check basic Space interface
                if not isinstance(space, Space):
                    logger.error(f"Object is not a valid Space instance: {type(space)}")
                    return False
                
                # Validate required methods exist
                required_methods = ['sample', 'contains', 'seed']
                for method in required_methods:
                    if not hasattr(space, method):
                        logger.error(f"Space missing required method: {method}")
                        return False
                
                # Validate dtype attribute exists and is valid
                if hasattr(space, 'dtype'):
                    if not isinstance(space.dtype, np.dtype):
                        logger.error(f"Space dtype is not a valid numpy dtype: {space.dtype}")
                        return False
                
                # Type-specific validation
                if isinstance(space, Box):
                    if not SpaceValidator._validate_box_space(space):
                        return False
                elif isinstance(space, Discrete):
                    if not SpaceValidator._validate_discrete_space(space):
                        return False
                elif isinstance(space, DictSpace):
                    if not SpaceValidator._validate_dict_space(space):
                        return False
                
                # Test space operations
                try:
                    sample = space.sample()
                    if not space.contains(sample):
                        logger.error("Space sample failed contains check")
                        return False
                except Exception as e:
                    logger.error(f"Space operation failed: {e}")
                    return False
                
                logger.info(f"Space validation successful: {type(space).__name__}")
                return True
                
        except Exception as e:
            logger.error(f"Space validation failed with exception: {e}")
            return False
    
    @staticmethod
    def _validate_box_space(space: Box) -> bool:
        """Validate Box space specific requirements."""
        # Check shape consistency
        if space.shape != space.low.shape or space.shape != space.high.shape:
            logger.error("Box space shape inconsistency between bounds")
            return False
        
        # Check bounds validity
        if np.any(space.low >= space.high):
            logger.error("Box space low bounds must be less than high bounds")
            return False
        
        # Check dtype consistency
        if space.low.dtype != space.high.dtype:
            logger.error("Box space low and high bounds must have same dtype")
            return False
        
        return True
    
    @staticmethod
    def _validate_discrete_space(space: Discrete) -> bool:
        """Validate Discrete space specific requirements."""
        if not isinstance(space.n, int) or space.n <= 0:
            logger.error(f"Discrete space n must be positive integer, got: {space.n}")
            return False
        
        return True
    
    @staticmethod
    def _validate_dict_space(space: DictSpace) -> bool:
        """Validate Dict space specific requirements."""
        if not isinstance(space.spaces, dict):
            logger.error("Dict space must contain spaces dictionary")
            return False
        
        # Recursively validate nested spaces
        for key, subspace in space.spaces.items():
            if not SpaceValidator.validate_gymnasium_compliance(subspace):
                logger.error(f"Dict space subspace '{key}' failed validation")
                return False
        
        return True
    
    @staticmethod
    def validate_step_return_format(
        step_return: Tuple, 
        expected_format: Literal["legacy", "gymnasium"] = "gymnasium"
    ) -> bool:
        """
        Validate step return format for API compliance.
        
        Supports validation of both legacy 4-tuple (obs, reward, done, info) and
        modern Gymnasium 5-tuple (obs, reward, terminated, truncated, info) formats.
        
        Args:
            step_return: Tuple returned from environment step() method
            expected_format: Expected format ('legacy' or 'gymnasium')
            
        Returns:
            True if format matches expectations
        """
        try:
            if expected_format == "legacy":
                if len(step_return) != 4:
                    logger.error(f"Legacy format requires 4 elements, got {len(step_return)}")
                    return False
                obs, reward, done, info = step_return
                if not isinstance(done, bool):
                    logger.error("Legacy format 'done' must be boolean")
                    return False
            elif expected_format == "gymnasium":
                if len(step_return) != 5:
                    logger.error(f"Gymnasium format requires 5 elements, got {len(step_return)}")
                    return False
                obs, reward, terminated, truncated, info = step_return
                if not isinstance(terminated, bool) or not isinstance(truncated, bool):
                    logger.error("Gymnasium format 'terminated' and 'truncated' must be boolean")
                    return False
            else:
                raise ValueError(f"Unknown format: {expected_format}")
            
            # Common validations
            if not isinstance(info, dict):
                logger.error("Info must be a dictionary")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Step return validation failed: {e}")
            return False
    
    @staticmethod
    def validate_sensor_aware_space(
        observation_space: DictSpace,
        sensor_configs: List[Dict[str, Any]],
        wind_enabled: bool = False
    ) -> bool:
        """
        Validate sensor-aware observation space for configuration compliance.
        
        Performs comprehensive validation of observation space structure to ensure
        it matches the expected sensor configuration and wind data settings.
        
        Args:
            observation_space: Observation space to validate
            sensor_configs: List of sensor configuration dicts
            wind_enabled: Whether wind data should be present
            
        Returns:
            bool: True if space matches sensor configuration
            
        Example:
            Validate sensor space:
                >>> space = create_sensor_observation_space(sensors)
                >>> configs = [{"type": "binary"}, {"type": "concentration"}]
                >>> is_valid = SpaceValidator.validate_sensor_aware_space(space, configs)
        """
        try:
            with correlation_context("validate_sensor_aware_space") as ctx:
                logger.debug("Validating sensor-aware observation space")
                
                if not isinstance(observation_space, DictSpace):
                    logger.error("Expected DictSpace for sensor-aware observations")
                    return False
                
                space_keys = set(observation_space.spaces.keys())
                
                # Validate sensor components
                for i, sensor_config in enumerate(sensor_configs):
                    sensor_type = sensor_config.get('type', 'unknown')
                    sensor_prefix = f"sensor_{i}_{sensor_type}"
                    
                    sensor_keys = [key for key in space_keys if sensor_prefix in key.lower()]
                    if not sensor_keys:
                        logger.error(f"No space components found for sensor {i} (type: {sensor_type})")
                        return False
                
                # Validate wind components
                wind_keys = [key for key in space_keys if key.startswith("wind_")]
                if wind_enabled and not wind_keys:
                    logger.error("Wind data enabled but no wind components found")
                    return False
                elif not wind_enabled and wind_keys:
                    logger.warning(f"Wind components found but wind not enabled: {wind_keys}")
                
                # Validate space component types
                for key, space in observation_space.spaces.items():
                    if not SpaceValidator.validate_gymnasium_compliance(space):
                        logger.error(f"Space component '{key}' failed Gymnasium compliance")
                        return False
                
                logger.info("Sensor-aware space validation successful")
                return True
                
        except Exception as e:
            logger.error(f"Sensor-aware space validation failed: {e}")
            return False
    
    @staticmethod
    def validate_multi_modal_observation_data(
        observation: Dict[str, Any],
        observation_space: DictSpace
    ) -> bool:
        """
        Validate multi-modal observation data against observation space.
        
        Ensures observation data conforms to the expected space structure
        and value ranges for all components.
        
        Args:
            observation: Observation data to validate
            observation_space: Expected observation space
            
        Returns:
            bool: True if observation data is valid
            
        Example:
            Validate observation data:
                >>> obs = {"sensor_0_binary": True, "agent_position": [0.0, 0.0]}
                >>> space = create_observation_space()
                >>> is_valid = SpaceValidator.validate_multi_modal_observation_data(obs, space)
        """
        try:
            with correlation_context("validate_multi_modal_observation") as ctx:
                logger.debug("Validating multi-modal observation data")
                
                if not isinstance(observation, dict):
                    logger.error("Observation must be a dictionary")
                    return False
                
                if not isinstance(observation_space, DictSpace):
                    logger.error("Expected DictSpace for validation")
                    return False
                
                # Check all required keys are present
                space_keys = set(observation_space.spaces.keys())
                obs_keys = set(observation.keys())
                
                missing_keys = space_keys - obs_keys
                if missing_keys:
                    logger.error(f"Missing observation keys: {missing_keys}")
                    return False
                
                extra_keys = obs_keys - space_keys
                if extra_keys:
                    logger.warning(f"Extra observation keys (ignored): {extra_keys}")
                
                # Validate each component
                for key in space_keys:
                    if key in observation:
                        component_space = observation_space.spaces[key]
                        component_value = observation[key]
                        
                        # Convert to numpy array if needed
                        if not isinstance(component_value, np.ndarray):
                            component_value = np.array(component_value)
                        
                        # Check if value is within space
                        if not component_space.contains(component_value):
                            logger.error(f"Observation component '{key}' out of bounds: {component_value}")
                            return False
                
                logger.debug("Multi-modal observation data validation successful")
                return True
                
        except Exception as e:
            logger.error(f"Multi-modal observation validation failed: {e}")
            return False


class ReturnFormatConverter:
    """
    Utilities for converting between legacy and modern step/reset return formats.
    
    Handles intelligent conversion between OpenAI Gym legacy 4-tuple returns
    (obs, reward, done, info) and modern Gymnasium 5-tuple returns 
    (obs, reward, terminated, truncated, info) while preserving semantics
    and maintaining performance requirements.
    """
    
    @staticmethod
    def to_gymnasium_format(
        legacy_return: Tuple[Any, float, bool, Dict[str, Any]]
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Convert legacy 4-tuple step return to Gymnasium 5-tuple format.
        
        Splits the legacy 'done' flag into 'terminated' and 'truncated' flags
        based on episode completion reason. Uses info dict metadata when
        available to determine the appropriate flag assignment.
        
        Args:
            legacy_return: (observation, reward, done, info) tuple
            
        Returns:
            (observation, reward, terminated, truncated, info) tuple
            
        Example:
            >>> legacy = (obs, 1.0, True, {"reason": "goal_reached"})
            >>> modern = ReturnFormatConverter.to_gymnasium_format(legacy)
            >>> print(modern)  # (obs, 1.0, True, False, {"reason": "goal_reached"})
        """
        obs, reward, done, info = legacy_return
        
        # Determine termination vs truncation based on info
        terminated = False
        truncated = False
        
        if done:
            # Check for explicit termination/truncation indicators in info
            if "terminated" in info:
                terminated = info["terminated"]
                truncated = not terminated
            elif "truncated" in info:
                truncated = info["truncated"]
                terminated = not truncated
            elif "reason" in info:
                reason = info["reason"].lower()
                if reason in ["goal_reached", "collision", "boundary_hit", "terminal_state"]:
                    terminated = True
                elif reason in ["time_limit", "max_steps", "timeout", "truncated"]:
                    truncated = True
                else:
                    # Default to terminated for unknown reasons
                    terminated = True
            else:
                # Default to terminated when no additional information available
                terminated = True
        
        # Ensure exactly one flag is set when done=True
        if done and not (terminated or truncated):
            terminated = True
        
        # Add format metadata to info
        info_copy = info.copy()
        info_copy["_format_converted"] = True
        info_copy["_original_done"] = done
        
        return obs, reward, terminated, truncated, info_copy
    
    @staticmethod
    def to_legacy_format(
        gymnasium_return: Tuple[Any, float, bool, bool, Dict[str, Any]]
    ) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Convert Gymnasium 5-tuple step return to legacy 4-tuple format.
        
        Combines 'terminated' and 'truncated' flags into a single 'done' flag
        using logical OR operation while preserving the original flag values
        in the info dictionary for potential downstream use.
        
        Args:
            gymnasium_return: (obs, reward, terminated, truncated, info) tuple
            
        Returns:
            (observation, reward, done, info) tuple
            
        Example:
            >>> modern = (obs, 1.0, True, False, {"reason": "goal_reached"})
            >>> legacy = ReturnFormatConverter.to_legacy_format(modern)
            >>> print(legacy)  # (obs, 1.0, True, {"reason": "goal_reached", ...})
        """
        obs, reward, terminated, truncated, info = gymnasium_return
        
        # Combine flags using logical OR
        done = terminated or truncated
        
        # Preserve original flags in info for potential downstream use
        info_copy = info.copy()
        info_copy["_terminated"] = terminated
        info_copy["_truncated"] = truncated
        info_copy["_format_converted"] = True
        
        # Add termination reason if not already present
        if done and "reason" not in info_copy:
            if terminated:
                info_copy["reason"] = "terminated"
            elif truncated:
                info_copy["reason"] = "truncated"
        
        return obs, reward, done, info_copy
    
    @staticmethod
    def detect_return_format(step_return: Tuple) -> Literal["legacy", "gymnasium", "unknown"]:
        """
        Automatically detect the format of a step return tuple.
        
        Analyzes tuple length and element types to determine whether the return
        follows legacy 4-tuple or modern Gymnasium 5-tuple format conventions.
        
        Args:
            step_return: Tuple returned from environment step() method
            
        Returns:
            Format type: 'legacy', 'gymnasium', or 'unknown'
        """
        if len(step_return) == 4:
            # Check if third element is boolean (done flag)
            if isinstance(step_return[2], bool):
                return "legacy"
        elif len(step_return) == 5:
            # Check if third and fourth elements are boolean (terminated, truncated)
            if isinstance(step_return[2], bool) and isinstance(step_return[3], bool):
                return "gymnasium"
        
        return "unknown"


class ActionSpaceFactory:
    """
    Factory for creating standardized action spaces with domain-specific constraints.
    
    Provides pre-configured action space definitions optimized for plume navigation
    tasks including continuous control, discrete actions, and hybrid action spaces
    with proper bounds validation and performance optimization.
    """
    
    # Default action space parameters for plume navigation
    DEFAULT_SPEED_RANGE = (0.0, 2.0)  # Units per second
    DEFAULT_ANGULAR_VELOCITY_RANGE = (-np.pi, np.pi)  # Radians per second
    DEFAULT_DISCRETE_ACTIONS = 5  # Stop, Forward, Left, Right, Backward
    
    @staticmethod
    def create_continuous_action_space(
        speed_range: Tuple[float, float] = DEFAULT_SPEED_RANGE,
        angular_velocity_range: Tuple[float, float] = DEFAULT_ANGULAR_VELOCITY_RANGE,
        dtype: np.dtype = np.float32
    ) -> Box:
        """
        Create continuous action space for speed and angular velocity control.
        
        Defines a 2D Box space where actions represent [speed, angular_velocity]
        with validated bounds appropriate for realistic agent movement in the
        plume navigation environment.
        
        Args:
            speed_range: (min_speed, max_speed) bounds in units/second
            angular_velocity_range: (min_angular, max_angular) bounds in rad/second
            dtype: NumPy dtype for action values
            
        Returns:
            Box space with shape (2,) for continuous control
            
        Example:
            >>> action_space = ActionSpaceFactory.create_continuous_action_space()
            >>> sample_action = action_space.sample()
            >>> print(sample_action)  # [speed, angular_velocity]
        """
        with correlation_context("create_continuous_action_space") as ctx:
            logger.debug(f"Creating continuous action space with speed_range={speed_range}, "
                        f"angular_velocity_range={angular_velocity_range}")
            
            # Validate input ranges
            if speed_range[0] < 0 or speed_range[1] <= speed_range[0]:
                raise ValueError("Speed range must be non-negative with max > min")
            
            if angular_velocity_range[1] <= angular_velocity_range[0]:
                raise ValueError("Angular velocity range must have max > min")
            
            # Create bounds arrays
            low = np.array([speed_range[0], angular_velocity_range[0]], dtype=dtype)
            high = np.array([speed_range[1], angular_velocity_range[1]], dtype=dtype)
            
            # Create space with metadata
            action_space = Box(
                low=low, 
                high=high, 
                shape=(2,), 
                dtype=dtype
            )
            
            # Add domain-specific metadata
            action_space.metadata = {
                "action_names": ["speed", "angular_velocity"],
                "action_units": ["units/second", "radians/second"],
                "space_type": "continuous_navigation",
                "created_by": "ActionSpaceFactory"
            }
            
            logger.info(f"Created continuous action space: {action_space}")
            return action_space
    
    @staticmethod
    def create_discrete_action_space(
        n_actions: int = DEFAULT_DISCRETE_ACTIONS,
        action_names: Optional[List[str]] = None
    ) -> Discrete:
        """
        Create discrete action space for grid-world style navigation.
        
        Defines discrete actions for simplified navigation including movement
        directions and stopping. Commonly used for debugging and baseline
        algorithms in the plume navigation environment.
        
        Args:
            n_actions: Number of discrete actions available
            action_names: Optional list of action names for metadata
            
        Returns:
            Discrete space with specified number of actions
            
        Example:
            >>> action_space = ActionSpaceFactory.create_discrete_action_space()
            >>> print(action_space.n)  # 5 (default actions)
        """
        with correlation_context("create_discrete_action_space") as ctx:
            logger.debug(f"Creating discrete action space with {n_actions} actions")
            
            if n_actions <= 0:
                raise ValueError("Number of actions must be positive")
            
            # Default action names for common configurations
            if action_names is None:
                if n_actions == 5:
                    action_names = ["stop", "forward", "left", "right", "backward"]
                elif n_actions == 4:
                    action_names = ["forward", "left", "right", "backward"]
                elif n_actions == 3:
                    action_names = ["left", "forward", "right"]
                else:
                    action_names = [f"action_{i}" for i in range(n_actions)]
            
            # Create discrete space
            action_space = Discrete(n_actions)
            
            # Add metadata
            action_space.metadata = {
                "action_names": action_names[:n_actions],
                "space_type": "discrete_navigation",
                "created_by": "ActionSpaceFactory"
            }
            
            logger.info(f"Created discrete action space: {action_space}")
            return action_space
    
    @staticmethod
    def create_validated_action_space(
        space_def: SpaceDefinition
    ) -> Union[Box, Discrete]:
        """
        Create action space from validated definition with comprehensive error checking.
        
        Provides type-safe action space creation with comprehensive validation
        and error reporting. Integrates with the logging system for debugging
        and operational monitoring.
        
        Args:
            space_def: Space definition with validation parameters
            
        Returns:
            Validated action space instance
            
        Raises:
            ValueError: If space definition is invalid or inconsistent
        """
        with correlation_context("create_validated_action_space", 
                                space_type=space_def.space_type) as ctx:
            logger.debug(f"Creating validated action space: {space_def.name}")
            
            if space_def.space_type == "Box":
                if space_def.shape is None or len(space_def.shape) == 0:
                    raise ValueError("Box action space requires non-empty shape")
                
                action_space = Box(
                    low=space_def.low,
                    high=space_def.high,
                    shape=space_def.shape,
                    dtype=space_def.dtype
                )
                
            elif space_def.space_type == "Discrete":
                if space_def.n is None or space_def.n <= 0:
                    raise ValueError("Discrete action space requires positive n")
                
                action_space = Discrete(space_def.n)
                
            else:
                raise ValueError(f"Unsupported action space type: {space_def.space_type}")
            
            # Add definition metadata
            action_space.metadata = {
                **space_def.metadata,
                "definition_name": space_def.name,
                "definition_description": space_def.description,
                "validation_enabled": space_def.validation_enabled,
                "created_by": "ActionSpaceFactory"
            }
            
            # Validate if requested
            if space_def.validation_enabled:
                if not SpaceValidator.validate_gymnasium_compliance(action_space):
                    raise ValueError(f"Created action space failed validation: {space_def.name}")
            
            logger.info(f"Created validated action space: {space_def.name}")
            return action_space


class SensorAwareSpaceFactory:
    """
    Advanced factory for creating dynamic observation spaces based on sensor configurations.
    
    This factory enables observation space construction that adapts to the specific
    sensor suite and environmental dynamics configuration, providing flexible
    multi-modal observation spaces with optimal type safety and performance.
    
    Key Features:
    - Dynamic space construction based on active SensorProtocol list
    - Wind data integration when environmental dynamics are enabled
    - Sensor-aware validation ensuring space-data compatibility
    - Gymnasium 0.29.x compliance with comprehensive type validation
    - Performance optimization for multi-agent scenarios
    
    Examples:
        Create space from sensor list:
            >>> sensors = [BinarySensor(), ConcentrationSensor()]
            >>> space = SensorAwareSpaceFactory.create_sensor_observation_space(sensors)
            
        Include wind data:
            >>> wind_config = WindDataConfig(enabled=True, velocity_components=2)
            >>> space = SensorAwareSpaceFactory.create_sensor_observation_space(
            ...     sensors, wind_config=wind_config
            ... )
    """
    
    @staticmethod
    def create_sensor_observation_space(
        sensors: List[Any],
        wind_config: Optional[WindDataConfig] = None,
        include_agent_state: bool = True,
        agent_state_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        dtype: np.dtype = np.float32
    ) -> DictSpace:
        """
        Create observation space dynamically based on active sensor configuration.
        
        Constructs a Dict observation space with components for each active sensor,
        optional wind data, and agent state information. The space automatically
        adapts to the sensor suite configuration without requiring code changes.
        
        Args:
            sensors: List of active sensor instances implementing SensorProtocol
            wind_config: Configuration for wind data integration (optional)
            include_agent_state: Include agent position/velocity in observations
            agent_state_bounds: Custom bounds for agent state components
            dtype: NumPy dtype for observation values
            
        Returns:
            DictSpace: Multi-modal observation space with sensor-specific components
            
        Raises:
            ValueError: If sensor list is empty or contains invalid sensors
            TypeError: If sensors don't implement SensorProtocol interface
            
        Examples:
            Multi-sensor configuration:
                >>> sensors = [
                ...     BinarySensor(threshold=0.1),
                ...     ConcentrationSensor(dynamic_range=(0, 1)),
                ...     GradientSensor()
                ... ]
                >>> space = SensorAwareSpaceFactory.create_sensor_observation_space(sensors)
                
            With wind data:
                >>> wind_config = WindDataConfig(
                ...     enabled=True, 
                ...     velocity_components=2,
                ...     include_direction=True
                ... )
                >>> space = SensorAwareSpaceFactory.create_sensor_observation_space(
                ...     sensors, wind_config=wind_config
                ... )
        """
        with correlation_context("create_sensor_observation_space") as ctx:
            logger.debug(f"Creating sensor-aware observation space for {len(sensors)} sensors")
            
            if not sensors:
                raise ValueError("At least one sensor must be provided")
            
            # Validate sensors implement SensorProtocol
            if SENSOR_PROTOCOL_AVAILABLE:
                for i, sensor in enumerate(sensors):
                    if not hasattr(sensor, 'detect') or not callable(getattr(sensor, 'detect')):
                        raise TypeError(f"Sensor {i} does not implement SensorProtocol.detect method")
            
            spaces_dict = {}
            
            # Add agent state components
            if include_agent_state:
                state_bounds = agent_state_bounds or {}
                
                # Agent position
                pos_bounds = state_bounds.get('position', (-100.0, 100.0))
                spaces_dict["agent_position"] = Box(
                    low=pos_bounds[0], high=pos_bounds[1], shape=(2,), dtype=dtype
                )
                
                # Agent velocity
                vel_bounds = state_bounds.get('velocity', (-10.0, 10.0))
                spaces_dict["agent_velocity"] = Box(
                    low=vel_bounds[0], high=vel_bounds[1], shape=(2,), dtype=dtype
                )
                
                # Agent orientation
                orient_bounds = state_bounds.get('orientation', (0.0, 360.0))
                spaces_dict["agent_orientation"] = Box(
                    low=orient_bounds[0], high=orient_bounds[1], shape=(1,), dtype=dtype
                )
            
            # Add sensor-specific components
            for i, sensor in enumerate(sensors):
                sensor_spaces = SensorAwareSpaceFactory._create_sensor_component_space(
                    sensor, i, dtype
                )
                spaces_dict.update(sensor_spaces)
            
            # Add wind data components if enabled
            if wind_config and wind_config.enabled:
                wind_spaces = SensorAwareSpaceFactory._create_wind_data_space(wind_config, dtype)
                spaces_dict.update(wind_spaces)
            
            # Create Dict space
            observation_space = DictSpace(spaces_dict)
            
            # Add comprehensive metadata
            observation_space.metadata = {
                "sensor_count": len(sensors),
                "sensor_types": [type(sensor).__name__ for sensor in sensors],
                "wind_enabled": wind_config.enabled if wind_config else False,
                "agent_state_included": include_agent_state,
                "space_type": "sensor_aware_navigation",
                "created_by": "SensorAwareSpaceFactory",
                "components": list(spaces_dict.keys())
            }
            
            logger.info(f"Created sensor-aware observation space with {len(spaces_dict)} components")
            return observation_space
    
    @staticmethod
    def _create_sensor_component_space(
        sensor: Any, 
        sensor_index: int, 
        dtype: np.dtype
    ) -> Dict[str, Box]:
        """
        Create observation space components for a specific sensor.
        
        Args:
            sensor: Sensor instance to create space for
            sensor_index: Index of sensor in sensor list
            dtype: NumPy dtype for observation values
            
        Returns:
            Dict[str, Box]: Space components for this sensor
        """
        spaces = {}
        sensor_name = type(sensor).__name__.lower()
        
        if SENSOR_IMPLEMENTATIONS_AVAILABLE:
            if isinstance(sensor, type(BinarySensor)) if BinarySensor else False:
                # Binary sensor returns boolean detection
                spaces[f"sensor_{sensor_index}_{sensor_name}_detection"] = Box(
                    low=0, high=1, shape=(1,), dtype=bool
                )
                
            elif isinstance(sensor, type(ConcentrationSensor)) if ConcentrationSensor else False:
                # Concentration sensor returns float concentration value
                # Try to get dynamic range from sensor config
                try:
                    config = getattr(sensor, 'config', None)
                    if config and hasattr(config, 'dynamic_range'):
                        low, high = config.dynamic_range
                    else:
                        low, high = 0.0, 1.0
                except:
                    low, high = 0.0, 1.0
                    
                spaces[f"sensor_{sensor_index}_{sensor_name}_concentration"] = Box(
                    low=low, high=high, shape=(1,), dtype=dtype
                )
                
            elif isinstance(sensor, type(GradientSensor)) if GradientSensor else False:
                # Gradient sensor returns 2D gradient vector
                spaces[f"sensor_{sensor_index}_{sensor_name}_gradient"] = Box(
                    low=-10.0, high=10.0, shape=(2,), dtype=dtype
                )
                # Also include gradient magnitude and direction
                spaces[f"sensor_{sensor_index}_{sensor_name}_magnitude"] = Box(
                    low=0.0, high=10.0, shape=(1,), dtype=dtype
                )
                spaces[f"sensor_{sensor_index}_{sensor_name}_direction"] = Box(
                    low=0.0, high=360.0, shape=(1,), dtype=dtype
                )
                
            else:
                # Generic sensor - assume float output
                spaces[f"sensor_{sensor_index}_{sensor_name}_output"] = Box(
                    low=0.0, high=1.0, shape=(1,), dtype=dtype
                )
        else:
            # Fallback when sensor implementations not available
            spaces[f"sensor_{sensor_index}_{sensor_name}_output"] = Box(
                low=0.0, high=1.0, shape=(1,), dtype=dtype
            )
        
        return spaces
    
    @staticmethod
    def _create_wind_data_space(
        wind_config: WindDataConfig, 
        dtype: np.dtype
    ) -> Dict[str, Box]:
        """
        Create observation space components for wind data.
        
        Args:
            wind_config: Wind data configuration
            dtype: NumPy dtype for observation values
            
        Returns:
            Dict[str, Box]: Wind data space components
        """
        spaces = {}
        
        if wind_config.coordinate_system == "cartesian":
            # Wind velocity components (vx, vy) or (vx, vy, vz)
            spaces["wind_velocity"] = Box(
                low=wind_config.velocity_range[0],
                high=wind_config.velocity_range[1],
                shape=(wind_config.velocity_components,),
                dtype=dtype
            )
        else:  # polar
            # Wind magnitude and direction
            spaces["wind_magnitude"] = Box(
                low=0.0, high=wind_config.velocity_range[1], shape=(1,), dtype=dtype
            )
            spaces["wind_direction"] = Box(
                low=0.0, high=360.0, shape=(1,), dtype=dtype
            )
        
        # Optional additional wind components
        if wind_config.include_direction and wind_config.coordinate_system == "cartesian":
            spaces["wind_direction"] = Box(
                low=0.0, high=360.0, shape=(1,), dtype=dtype
            )
        
        if wind_config.include_magnitude and wind_config.coordinate_system == "cartesian":
            spaces["wind_magnitude"] = Box(
                low=0.0, high=wind_config.velocity_range[1], shape=(1,), dtype=dtype
            )
        
        # Temporal history if configured
        if wind_config.temporal_history > 0:
            history_shape = (wind_config.temporal_history, wind_config.velocity_components)
            spaces["wind_velocity_history"] = Box(
                low=wind_config.velocity_range[0],
                high=wind_config.velocity_range[1],
                shape=history_shape,
                dtype=dtype
            )
        
        return spaces
    
    @staticmethod
    def validate_sensor_space_compatibility(
        observation_space: DictSpace,
        sensors: List[Any],
        wind_config: Optional[WindDataConfig] = None
    ) -> bool:
        """
        Validate that observation space matches sensor configuration.
        
        Ensures that the observation space contains appropriate components
        for each configured sensor and wind data configuration.
        
        Args:
            observation_space: Observation space to validate
            sensors: List of sensor instances
            wind_config: Wind data configuration (optional)
            
        Returns:
            bool: True if space is compatible with sensor configuration
            
        Raises:
            ValueError: If critical compatibility issues are found
        """
        with correlation_context("validate_sensor_space_compatibility") as ctx:
            logger.debug("Validating sensor-space compatibility")
            
            if not isinstance(observation_space, DictSpace):
                raise ValueError("Expected DictSpace for sensor-aware observations")
            
            space_keys = set(observation_space.spaces.keys())
            expected_keys = set()
            
            # Check agent state components
            if any(key.startswith("agent_") for key in space_keys):
                expected_keys.update(["agent_position", "agent_velocity", "agent_orientation"])
            
            # Check sensor components
            for i, sensor in enumerate(sensors):
                sensor_name = type(sensor).__name__.lower()
                sensor_keys = [key for key in space_keys if f"sensor_{i}_{sensor_name}" in key]
                
                if not sensor_keys:
                    logger.error(f"No space components found for sensor {i} ({sensor_name})")
                    return False
                    
                expected_keys.update(sensor_keys)
            
            # Check wind components
            if wind_config and wind_config.enabled:
                wind_keys = [key for key in space_keys if key.startswith("wind_")]
                if not wind_keys:
                    logger.error("Wind data enabled but no wind components in space")
                    return False
                expected_keys.update(wind_keys)
            
            # Validate all expected keys are present
            missing_keys = expected_keys - space_keys
            if missing_keys:
                logger.error(f"Missing space components: {missing_keys}")
                return False
            
            logger.info("Sensor-space compatibility validation successful")
            return True


class ObservationSpaceFactory:
    """
    Factory for creating multi-modal observation spaces for plume navigation.
    
    Constructs standardized observation spaces combining odor concentration readings,
    agent state information, and environmental context with proper type validation
    and Gymnasium compliance for RL training workflows.
    
    Note: This class provides backward compatibility. For new implementations,
    consider using SensorAwareSpaceFactory for more flexible sensor-based spaces.
    """
    
    @staticmethod
    def create_navigation_observation_space(
        include_position: bool = True,
        include_velocity: bool = True,
        include_odor: bool = True,
        include_sensors: bool = False,
        position_bounds: Tuple[float, float] = (-10.0, 10.0),
        velocity_bounds: Tuple[float, float] = (-5.0, 5.0),
        odor_concentration_range: Tuple[float, float] = (0.0, 1.0),
        dtype: np.dtype = np.float32
    ) -> DictSpace:
        """
        Create comprehensive multi-modal observation space for navigation tasks.
        
        Constructs a Dictionary space containing multiple observation components
        relevant to odor plume navigation including agent state, environmental
        readings, and optional sensor data with validated bounds and types.
        
        Args:
            include_position: Include agent position (x, y) in observations
            include_velocity: Include agent velocity (vx, vy) in observations  
            include_odor: Include odor concentration reading
            include_sensors: Include additional sensor data
            position_bounds: (min, max) bounds for position coordinates
            velocity_bounds: (min, max) bounds for velocity components
            odor_concentration_range: (min, max) range for odor readings
            dtype: NumPy dtype for observation values
            
        Returns:
            Dict space with selected observation components
            
        Example:
            >>> obs_space = ObservationSpaceFactory.create_navigation_observation_space()
            >>> sample_obs = obs_space.sample()
            >>> print(sample_obs.keys())  # ['position', 'velocity', 'odor_concentration']
        """
        with correlation_context("create_navigation_observation_space") as ctx:
            logger.debug(f"Creating navigation observation space with components: "
                        f"position={include_position}, velocity={include_velocity}, "
                        f"odor={include_odor}, sensors={include_sensors}")
            
            spaces_dict = {}
            
            # Position component (x, y coordinates)
            if include_position:
                spaces_dict["position"] = Box(
                    low=position_bounds[0],
                    high=position_bounds[1],
                    shape=(2,),
                    dtype=dtype
                )
            
            # Velocity component (vx, vy)
            if include_velocity:
                spaces_dict["velocity"] = Box(
                    low=velocity_bounds[0],
                    high=velocity_bounds[1],
                    shape=(2,),
                    dtype=dtype
                )
            
            # Odor concentration scalar
            if include_odor:
                spaces_dict["odor_concentration"] = Box(
                    low=odor_concentration_range[0],
                    high=odor_concentration_range[1],
                    shape=(1,),
                    dtype=dtype
                )
            
            # Additional sensor data (configurable size)
            if include_sensors:
                spaces_dict["sensors"] = Box(
                    low=0.0,
                    high=1.0,
                    shape=(4,),  # Example: 4 additional sensors
                    dtype=dtype
                )
            
            if not spaces_dict:
                raise ValueError("At least one observation component must be included")
            
            # Create Dict space
            observation_space = DictSpace(spaces_dict)
            
            # Add comprehensive metadata
            observation_space.metadata = {
                "components": list(spaces_dict.keys()),
                "component_descriptions": {
                    "position": "Agent (x, y) coordinates",
                    "velocity": "Agent velocity (vx, vy)",
                    "odor_concentration": "Scalar odor concentration reading",
                    "sensors": "Additional sensor readings"
                },
                "space_type": "multi_modal_navigation",
                "created_by": "ObservationSpaceFactory"
            }
            
            logger.info(f"Created navigation observation space with {len(spaces_dict)} components")
            return observation_space
    
    @staticmethod
    def create_odor_field_observation_space(
        field_size: Tuple[int, int] = (64, 64),
        concentration_range: Tuple[float, float] = (0.0, 1.0),
        include_metadata: bool = True,
        dtype: np.dtype = np.float32
    ) -> DictSpace:
        """
        Create observation space for full odor field visualization.
        
        Constructs observation space including complete odor concentration field
        data for visualization and analysis workflows. Supports both local and
        global field observations with configurable resolution.
        
        Args:
            field_size: (height, width) of odor field grid
            concentration_range: (min, max) odor concentration values
            include_metadata: Include field metadata in observations
            dtype: NumPy dtype for field values
            
        Returns:
            Dict space with odor field and optional metadata
        """
        with correlation_context("create_odor_field_observation_space") as ctx:
            logger.debug(f"Creating odor field observation space: {field_size}")
            
            spaces_dict = {}
            
            # Main odor field
            spaces_dict["odor_field"] = Box(
                low=concentration_range[0],
                high=concentration_range[1],
                shape=field_size,
                dtype=dtype
            )
            
            # Optional metadata
            if include_metadata:
                spaces_dict["field_metadata"] = Box(
                    low=0.0,
                    high=1.0,
                    shape=(4,),  # Example: timestamp, max_concentration, mean, std
                    dtype=dtype
                )
            
            observation_space = DictSpace(spaces_dict)
            
            # Add metadata
            observation_space.metadata = {
                "field_size": field_size,
                "concentration_range": concentration_range,
                "space_type": "odor_field_visualization",
                "created_by": "ObservationSpaceFactory"
            }
            
            logger.info(f"Created odor field observation space: {field_size}")
            return observation_space
    
    @staticmethod
    def create_dynamic_sensor_observation_space(
        sensors: List[Any],
        wind_config: Optional[WindDataConfig] = None,
        include_position: bool = True,
        include_velocity: bool = True,
        include_orientation: bool = True,
        position_bounds: Tuple[float, float] = (-10.0, 10.0),
        velocity_bounds: Tuple[float, float] = (-5.0, 5.0),
        dtype: np.dtype = np.float32
    ) -> DictSpace:
        """
        Create observation space dynamically based on sensor configuration.
        
        Extended version of create_navigation_observation_space that supports
        dynamic sensor configuration and wind data integration. This method
        bridges backward compatibility with new sensor-aware functionality.
        
        Args:
            sensors: List of sensor instances implementing SensorProtocol
            wind_config: Configuration for wind data integration
            include_position: Include agent position in observations
            include_velocity: Include agent velocity in observations
            include_orientation: Include agent orientation in observations
            position_bounds: Bounds for position components
            velocity_bounds: Bounds for velocity components
            dtype: NumPy dtype for observation values
            
        Returns:
            DictSpace: Multi-modal observation space with sensor components
            
        Examples:
            Basic sensor-aware space:
                >>> sensors = [BinarySensor(), ConcentrationSensor()]
                >>> space = ObservationSpaceFactory.create_dynamic_sensor_observation_space(sensors)
                
            With wind data:
                >>> wind_config = WindDataConfig(enabled=True)
                >>> space = ObservationSpaceFactory.create_dynamic_sensor_observation_space(
                ...     sensors, wind_config=wind_config
                ... )
        """
        # Delegate to SensorAwareSpaceFactory for implementation
        agent_state_bounds = {}
        if include_position:
            agent_state_bounds['position'] = position_bounds
        if include_velocity:
            agent_state_bounds['velocity'] = velocity_bounds
        if include_orientation:
            agent_state_bounds['orientation'] = (0.0, 360.0)
        
        include_agent_state = include_position or include_velocity or include_orientation
        
        return SensorAwareSpaceFactory.create_sensor_observation_space(
            sensors=sensors,
            wind_config=wind_config,
            include_agent_state=include_agent_state,
            agent_state_bounds=agent_state_bounds,
            dtype=dtype
        )
    
    @staticmethod
    def create_wind_integrated_observation_space(
        base_observation_components: Dict[str, Any],
        wind_config: WindDataConfig,
        dtype: np.dtype = np.float32
    ) -> DictSpace:
        """
        Integrate wind data into existing observation space components.
        
        Takes an existing set of observation components and adds wind data
        according to the specified configuration. Useful for upgrading
        existing observation spaces with environmental dynamics.
        
        Args:
            base_observation_components: Dict of existing observation components
            wind_config: Wind data configuration
            dtype: NumPy dtype for wind data
            
        Returns:
            DictSpace: Enhanced observation space with wind components
            
        Example:
            Add wind to existing space:
                >>> base_components = {
                ...     "odor_concentration": Box(low=0, high=1, shape=(1,)),
                ...     "position": Box(low=-10, high=10, shape=(2,))
                ... }
                >>> wind_config = WindDataConfig(enabled=True, velocity_components=2)
                >>> space = ObservationSpaceFactory.create_wind_integrated_observation_space(
                ...     base_components, wind_config
                ... )
        """
        with correlation_context("create_wind_integrated_observation_space") as ctx:
            logger.debug("Creating wind-integrated observation space")
            
            if not wind_config.enabled:
                logger.warning("Wind config indicates disabled - returning base components only")
                return DictSpace(base_observation_components)
            
            # Copy base components
            enhanced_components = base_observation_components.copy()
            
            # Add wind components
            wind_spaces = SensorAwareSpaceFactory._create_wind_data_space(wind_config, dtype)
            enhanced_components.update(wind_spaces)
            
            # Create enhanced space
            observation_space = DictSpace(enhanced_components)
            
            # Add metadata
            observation_space.metadata = {
                "base_components": list(base_observation_components.keys()),
                "wind_components": list(wind_spaces.keys()),
                "wind_enabled": True,
                "wind_coordinate_system": wind_config.coordinate_system,
                "space_type": "wind_integrated_navigation",
                "created_by": "ObservationSpaceFactory"
            }
            
            logger.info(f"Created wind-integrated space with {len(enhanced_components)} total components")
            return observation_space
    
    @staticmethod
    def validate_observation_space_sensor_compatibility(
        observation_space: DictSpace,
        sensor_list: List[Any],
        wind_enabled: bool = False
    ) -> Dict[str, Any]:
        """
        Validate observation space compatibility with sensor configuration.
        
        Performs comprehensive validation of observation space structure
        against the provided sensor configuration, returning detailed
        compatibility analysis and recommendations.
        
        Args:
            observation_space: Observation space to validate
            sensor_list: List of sensor instances
            wind_enabled: Whether wind data should be present
            
        Returns:
            Dict[str, Any]: Validation results with compatibility status and details
            
        Example:
            Validate space compatibility:
                >>> sensors = [BinarySensor(), ConcentrationSensor()]
                >>> space = create_some_observation_space()
                >>> results = ObservationSpaceFactory.validate_observation_space_sensor_compatibility(
                ...     space, sensors, wind_enabled=True
                ... )
                >>> if results['compatible']:
                ...     print("Space is compatible")
                ... else:
                ...     print(f"Issues: {results['issues']}")
        """
        with correlation_context("validate_observation_space_sensor_compatibility") as ctx:
            logger.debug("Validating observation space sensor compatibility")
            
            validation_results = {
                "compatible": True,
                "issues": [],
                "warnings": [],
                "recommendations": [],
                "space_analysis": {},
                "sensor_analysis": {}
            }
            
            if not isinstance(observation_space, DictSpace):
                validation_results["compatible"] = False
                validation_results["issues"].append("Observation space must be DictSpace for sensor compatibility")
                return validation_results
            
            space_keys = set(observation_space.spaces.keys())
            validation_results["space_analysis"]["component_count"] = len(space_keys)
            validation_results["space_analysis"]["component_names"] = list(space_keys)
            
            # Analyze sensor requirements
            expected_sensor_components = []
            for i, sensor in enumerate(sensor_list):
                sensor_name = type(sensor).__name__.lower()
                sensor_components = [key for key in space_keys if f"sensor_{i}_{sensor_name}" in key]
                
                if not sensor_components:
                    validation_results["issues"].append(
                        f"No components found for sensor {i} ({sensor_name})"
                    )
                    validation_results["compatible"] = False
                else:
                    expected_sensor_components.extend(sensor_components)
            
            validation_results["sensor_analysis"]["expected_components"] = expected_sensor_components
            validation_results["sensor_analysis"]["sensor_count"] = len(sensor_list)
            validation_results["sensor_analysis"]["sensor_types"] = [type(s).__name__ for s in sensor_list]
            
            # Check wind components
            wind_components = [key for key in space_keys if key.startswith("wind_")]
            if wind_enabled and not wind_components:
                validation_results["issues"].append("Wind data enabled but no wind components found in space")
                validation_results["compatible"] = False
            elif not wind_enabled and wind_components:
                validation_results["warnings"].append(
                    f"Wind components present but wind not enabled: {wind_components}"
                )
            
            validation_results["space_analysis"]["wind_components"] = wind_components
            validation_results["space_analysis"]["wind_component_count"] = len(wind_components)
            
            # Check for extra components
            known_prefixes = ["agent_", "sensor_", "wind_", "odor_"]
            unknown_components = [
                key for key in space_keys 
                if not any(key.startswith(prefix) for prefix in known_prefixes)
            ]
            if unknown_components:
                validation_results["warnings"].append(f"Unknown components found: {unknown_components}")
            
            # Generate recommendations
            if not validation_results["compatible"]:
                validation_results["recommendations"].append(
                    "Use SensorAwareSpaceFactory.create_sensor_observation_space() for automatic sensor-space compatibility"
                )
            
            if wind_enabled and not wind_components:
                validation_results["recommendations"].append(
                    "Add wind_config parameter to space creation for wind data integration"
                )
            
            logger.info(f"Validation complete: compatible={validation_results['compatible']}, "
                       f"issues={len(validation_results['issues'])}, warnings={len(validation_results['warnings'])}")
            
            return validation_results


class SpaceFactory:
    """
    Unified factory interface for creating type-safe spaces with comprehensive validation.
    
    Provides a single entry point for creating validated spaces with consistent
    error handling, logging integration, and performance optimization. Supports
    all standard Gymnasium space types with domain-specific defaults.
    """
    
    @staticmethod
    def create_validated_box_space(
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        name: str = "box_space",
        validate: bool = True
    ) -> Box:
        """
        Create validated Box space with comprehensive error checking.
        
        Provides type-safe Box space creation with bounds validation, shape
        consistency checking, and optional Gymnasium compliance validation.
        Integrates with logging system for debugging and error reporting.
        
        Args:
            low: Lower bounds (scalar or array-like)
            high: Upper bounds (scalar or array-like)
            shape: Space shape tuple
            dtype: NumPy dtype for space elements
            name: Human-readable space name for logging
            validate: Enable comprehensive validation
            
        Returns:
            Validated Box space instance
            
        Raises:
            ValueError: If space parameters are invalid
            
        Example:
            >>> space = SpaceFactory.create_validated_box_space(
            ...     low=-1.0, high=1.0, shape=(2,), name="action_space"
            ... )
            >>> print(space)  # Box(2,) with proper validation
        """
        with correlation_context("create_validated_box_space", space_name=name) as ctx:
            logger.debug(f"Creating validated Box space: {name}")
            
            try:
                # Convert bounds to numpy arrays if needed
                if not isinstance(low, np.ndarray):
                    low = np.full(shape, low, dtype=dtype)
                if not isinstance(high, np.ndarray):
                    high = np.full(shape, high, dtype=dtype)
                
                # Validate shapes
                if low.shape != shape or high.shape != shape:
                    raise ValueError(f"Bounds shape mismatch: low={low.shape}, "
                                   f"high={high.shape}, expected={shape}")
                
                # Validate bounds relationship
                if np.any(low >= high):
                    raise ValueError("Low bounds must be strictly less than high bounds")
                
                # Create Box space
                space = Box(low=low, high=high, shape=shape, dtype=dtype)
                
                # Add metadata
                space.metadata = {
                    "name": name,
                    "created_by": "SpaceFactory",
                    "validation_enabled": validate
                }
                
                # Validate if requested
                if validate:
                    if not SpaceValidator.validate_gymnasium_compliance(space):
                        raise ValueError(f"Box space failed Gymnasium compliance: {name}")
                
                logger.info(f"Created validated Box space: {name} with shape {shape}")
                return space
                
            except Exception as e:
                logger.error(f"Failed to create Box space '{name}': {e}")
                raise
    
    @staticmethod
    def create_validated_discrete_space(
        n: int,
        name: str = "discrete_space",
        validate: bool = True
    ) -> Discrete:
        """
        Create validated Discrete space with error checking.
        
        Args:
            n: Number of discrete actions
            name: Human-readable space name
            validate: Enable validation
            
        Returns:
            Validated Discrete space instance
        """
        with correlation_context("create_validated_discrete_space", space_name=name) as ctx:
            logger.debug(f"Creating validated Discrete space: {name} with n={n}")
            
            try:
                if not isinstance(n, int) or n <= 0:
                    raise ValueError(f"Discrete space n must be positive integer, got {n}")
                
                space = Discrete(n)
                
                # Add metadata
                space.metadata = {
                    "name": name,
                    "created_by": "SpaceFactory",
                    "validation_enabled": validate
                }
                
                # Validate if requested
                if validate:
                    if not SpaceValidator.validate_gymnasium_compliance(space):
                        raise ValueError(f"Discrete space failed validation: {name}")
                
                logger.info(f"Created validated Discrete space: {name} with n={n}")
                return space
                
            except Exception as e:
                logger.error(f"Failed to create Discrete space '{name}': {e}")
                raise


# Thread-local storage for space caching to improve performance
_space_cache = threading.local()


def get_cached_space(space_key: str, factory_func, *args, **kwargs) -> Space:
    """
    Get or create cached space instance for improved performance.
    
    Maintains thread-local cache of commonly used spaces to avoid repeated
    creation overhead in multi-environment scenarios. Automatically handles
    cache invalidation and memory management.
    
    Args:
        space_key: Unique identifier for the space
        factory_func: Function to create the space if not cached
        *args: Arguments for factory function
        **kwargs: Keyword arguments for factory function
        
    Returns:
        Cached or newly created space instance
    """
    if not hasattr(_space_cache, 'spaces'):
        _space_cache.spaces = {}
    
    if space_key not in _space_cache.spaces:
        logger.debug(f"Creating new cached space: {space_key}")
        _space_cache.spaces[space_key] = factory_func(*args, **kwargs)
    else:
        logger.debug(f"Using cached space: {space_key}")
    
    return _space_cache.spaces[space_key]


def clear_space_cache():
    """Clear thread-local space cache to free memory."""
    if hasattr(_space_cache, 'spaces'):
        logger.debug(f"Clearing space cache with {len(_space_cache.spaces)} entries")
        _space_cache.spaces.clear()


# Pre-configured standard spaces for common use cases
def get_standard_action_space() -> Box:
    """Get standard continuous action space for plume navigation."""
    return get_cached_space(
        "standard_action_space",
        ActionSpaceFactory.create_continuous_action_space
    )


def get_standard_observation_space() -> DictSpace:
    """Get standard multi-modal observation space for plume navigation."""
    return get_cached_space(
        "standard_observation_space", 
        ObservationSpaceFactory.create_navigation_observation_space
    )


def get_sensor_aware_observation_space(
    sensors: List[Any],
    wind_config: Optional[WindDataConfig] = None,
    cache_key: Optional[str] = None
) -> DictSpace:
    """
    Get cached sensor-aware observation space for improved performance.
    
    Creates and caches observation spaces based on sensor configuration
    to avoid repeated space construction overhead in multi-environment scenarios.
    
    Args:
        sensors: List of sensor instances
        wind_config: Wind data configuration (optional)
        cache_key: Custom cache key (auto-generated if None)
        
    Returns:
        DictSpace: Cached sensor-aware observation space
        
    Example:
        Get cached sensor space:
            >>> sensors = [BinarySensor(), ConcentrationSensor()]
            >>> space = get_sensor_aware_observation_space(sensors)
            >>> # Subsequent calls return cached instance
    """
    if cache_key is None:
        # Generate cache key from sensor configuration
        sensor_types = [type(sensor).__name__ for sensor in sensors]
        wind_suffix = f"_wind_{wind_config.coordinate_system}" if wind_config and wind_config.enabled else ""
        cache_key = f"sensor_space_{'_'.join(sensor_types)}{wind_suffix}"
    
    return get_cached_space(
        cache_key,
        SensorAwareSpaceFactory.create_sensor_observation_space,
        sensors,
        wind_config
    )


def validate_sensor_observation_compatibility(
    observation: Dict[str, Any],
    sensors: List[Any],
    wind_config: Optional[WindDataConfig] = None
) -> bool:
    """
    Validate that observation data matches sensor configuration.
    
    Checks that the provided observation dict contains appropriate
    data for each configured sensor and wind configuration.
    
    Args:
        observation: Observation data to validate
        sensors: List of sensor instances
        wind_config: Wind data configuration (optional)
        
    Returns:
        bool: True if observation is compatible with sensor configuration
        
    Raises:
        ValueError: If critical compatibility issues are found
        
    Example:
        Validate observation:
            >>> obs = {"sensor_0_binarysensor_detection": True, "agent_position": [0, 0]}
            >>> sensors = [BinarySensor()]
            >>> is_valid = validate_sensor_observation_compatibility(obs, sensors)
    """
    with correlation_context("validate_sensor_observation_compatibility") as ctx:
        logger.debug("Validating sensor observation compatibility")
        
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a dictionary")
        
        obs_keys = set(observation.keys())
        
        # Check sensor components
        for i, sensor in enumerate(sensors):
            sensor_name = type(sensor).__name__.lower()
            sensor_keys = [key for key in obs_keys if f"sensor_{i}_{sensor_name}" in key]
            
            if not sensor_keys:
                logger.error(f"No observation data found for sensor {i} ({sensor_name})")
                return False
        
        # Check wind components if enabled
        if wind_config and wind_config.enabled:
            wind_keys = [key for key in obs_keys if key.startswith("wind_")]
            if not wind_keys:
                logger.error("Wind data enabled but no wind observations found")
                return False
        
        logger.debug("Sensor observation compatibility validation successful")
        return True


# Backward compatibility functions for legacy code
def create_gym_spaces() -> Tuple[Box, DictSpace]:
    """
    Create standard gym-compatible spaces (deprecated).
    
    This function is deprecated and will be removed in v1.0.
    Use get_standard_action_space() and get_standard_observation_space() instead.
    """
    warnings.warn(
        "create_gym_spaces() is deprecated. Use get_standard_action_space() and "
        "get_standard_observation_space() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    log_legacy_api_deprecation(
        operation="space_creation",
        legacy_call="create_gym_spaces()",
        recommended_call="get_standard_action_space(), get_standard_observation_space()",
        migration_guide="Replace with individual space factory methods for better control"
    )
    
    return get_standard_action_space(), get_standard_observation_space()


# Module exports for clean API
__all__ = [
    # Core classes
    "SpaceDefinition",
    "SensorSpaceDefinition",
    "WindDataConfig",
    "SpaceValidator", 
    "ReturnFormatConverter",
    "ActionSpaceFactory",
    "ObservationSpaceFactory",
    "SensorAwareSpaceFactory",
    "SpaceFactory",
    
    # Utility functions
    "get_cached_space",
    "clear_space_cache",
    "get_standard_action_space",
    "get_standard_observation_space",
    "get_sensor_aware_observation_space",
    "validate_sensor_observation_compatibility",
    
    # Legacy compatibility (deprecated)
    "create_gym_spaces",
]