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


class ObservationSpaceFactory:
    """
    Factory for creating multi-modal observation spaces for plume navigation.
    
    Constructs standardized observation spaces combining odor concentration readings,
    agent state information, and environmental context with proper type validation
    and Gymnasium compliance for RL training workflows.
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
    "SpaceValidator", 
    "ReturnFormatConverter",
    "ActionSpaceFactory",
    "ObservationSpaceFactory",
    "SpaceFactory",
    
    # Utility functions
    "get_cached_space",
    "clear_space_cache",
    "get_standard_action_space",
    "get_standard_observation_space",
    
    # Legacy compatibility (deprecated)
    "create_gym_spaces",
]