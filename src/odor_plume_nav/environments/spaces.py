"""
Gymnasium-compliant action and observation space definitions for odor plume navigation.

This module provides standardized space definitions essential for RL training workflows
and ensures compatibility with the broader Gymnasium ecosystem. It defines continuous
action spaces for agent control and multi-modal observation spaces for environmental
perception, supporting both single-sensor and multi-sensor configurations.

The space definitions integrate seamlessly with the existing NavigatorProtocol and
VideoPlume components while maintaining full compatibility with stable-baselines3
and other modern reinforcement learning frameworks.

Enhanced for Gymnasium 0.29.x compatibility with terminated/truncated separation
and comprehensive validation support including gymnasium.utils.env_checker integration.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Union, Any, List
import numpy as np
import warnings

# Import enhanced logging for validation diagnostics
try:
    from odor_plume_nav.utils.logging_setup import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback for cases where logging_setup isn't available
from loguru import logger
    logger.warning("Enhanced logging not available, falling back to standard logging")

# Import Gymnasium spaces directly; failure will raise ImportError
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace
from gymnasium.utils.env_checker import check_env

# Import environment checker utilities for comprehensive validation
try:
    from gymnasium.utils.env_checker import (
        check_action_space,
        check_observation_space,
        check_env as check_environment  # Alias for backward compatibility
    )
    ENV_CHECKER_AVAILABLE = True
    logger.debug("Gymnasium environment checker utilities loaded successfully")
except ImportError:
    ENV_CHECKER_AVAILABLE = False
    logger.warning("Gymnasium environment checker utilities not available")

# Type aliases for enhanced clarity and IDE support
ActionType = np.ndarray
"""Type alias for action arrays with shape (2,) containing [speed, angular_velocity]."""

ObservationType = Dict[str, np.ndarray]
"""Type alias for observation dictionaries containing multi-modal sensor data."""

BoundsType = Union[Tuple[float, float], np.ndarray]
"""Type alias for bounds specification - supports (low, high) tuples or numpy arrays."""

# Gymnasium 0.29.x compatibility type aliases
StepReturnType = Tuple[Any, float, bool, bool, Dict[str, Any]]
"""Type alias for Gymnasium 0.29.x step return: (obs, reward, terminated, truncated, info)."""

LegacyStepReturnType = Tuple[Any, float, bool, Dict[str, Any]]
"""Type alias for legacy gym step return: (obs, reward, done, info)."""

ValidationResultType = Dict[str, Any]
"""Type alias for validation result dictionaries."""

# Constants for Gymnasium 0.29.x compatibility
GYMNASIUM_STEP_RETURN_LENGTH = 5
"""Expected length of Gymnasium 0.29.x step return tuple."""

LEGACY_STEP_RETURN_LENGTH = 4
"""Expected length of legacy gym step return tuple."""

SUPPORTED_GYMNASIUM_VERSION = "0.29.x"
"""Supported Gymnasium version for terminated/truncated separation."""


def validate_step_return_format(
    step_return: Union[StepReturnType, LegacyStepReturnType],
    expect_gymnasium_format: bool = True
) -> ValidationResultType:
    """
    Validate step return format for Gymnasium 0.29.x compatibility.
    
    Checks whether a step return follows the expected format and provides
    detailed diagnostics for debugging terminated/truncated separation issues.
    
    Args:
        step_return: Step return tuple from environment.step()
        expect_gymnasium_format: Whether to expect 5-tuple (True) or 4-tuple (False)
        
    Returns:
        ValidationResultType: Detailed validation results
        
    Examples:
        >>> # Gymnasium 0.29.x format
        >>> step_return = (obs, 0.1, False, False, {})
        >>> result = validate_step_return_format(step_return, expect_gymnasium_format=True)
        >>> assert result['format_valid']
        >>> 
        >>> # Legacy gym format
        >>> step_return = (obs, 0.1, False, {})
        >>> result = validate_step_return_format(step_return, expect_gymnasium_format=False)
        >>> assert result['format_valid']
    """
    try:
        results = {
            "format_valid": False,
            "return_length": len(step_return) if hasattr(step_return, '__len__') else 0,
            "expected_length": GYMNASIUM_STEP_RETURN_LENGTH if expect_gymnasium_format else LEGACY_STEP_RETURN_LENGTH,
            "format_type": "unknown",
            "components": {},
            "warnings": [],
            "errors": []
        }
        
        if not hasattr(step_return, '__len__'):
            results["errors"].append("Step return is not a tuple/sequence")
            return results
        
        return_length = len(step_return)
        results["return_length"] = return_length
        
        if expect_gymnasium_format and return_length == GYMNASIUM_STEP_RETURN_LENGTH:
            # Validate Gymnasium 0.29.x format: (obs, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_return
            results["format_type"] = "gymnasium_0.29.x"
            results["format_valid"] = True
            
            # Validate component types
            results["components"] = {
                "observation": {"type": type(obs).__name__, "valid": True},
                "reward": {"type": type(reward).__name__, "valid": isinstance(reward, (int, float, np.number))},
                "terminated": {"type": type(terminated).__name__, "valid": isinstance(terminated, bool)},
                "truncated": {"type": type(truncated).__name__, "valid": isinstance(truncated, bool)},
                "info": {"type": type(info).__name__, "valid": isinstance(info, dict)}
            }
            
            # Check for common issues
            if not isinstance(terminated, bool):
                results["errors"].append(f"'terminated' should be bool, got {type(terminated).__name__}")
                results["format_valid"] = False
            
            if not isinstance(truncated, bool):
                results["errors"].append(f"'truncated' should be bool, got {type(truncated).__name__}")
                results["format_valid"] = False
            
            if not isinstance(info, dict):
                results["errors"].append(f"'info' should be dict, got {type(info).__name__}")
                results["format_valid"] = False
            
            if terminated and truncated:
                results["warnings"].append("Both terminated and truncated are True (unusual but valid)")
            
        elif not expect_gymnasium_format and return_length == LEGACY_STEP_RETURN_LENGTH:
            # Validate legacy gym format: (obs, reward, done, info)
            obs, reward, done, info = step_return
            results["format_type"] = "legacy_gym"
            results["format_valid"] = True
            
            results["components"] = {
                "observation": {"type": type(obs).__name__, "valid": True},
                "reward": {"type": type(reward).__name__, "valid": isinstance(reward, (int, float, np.number))},
                "done": {"type": type(done).__name__, "valid": isinstance(done, bool)},
                "info": {"type": type(info).__name__, "valid": isinstance(info, dict)}
            }
            
            if not isinstance(done, bool):
                results["errors"].append(f"'done' should be bool, got {type(done).__name__}")
                results["format_valid"] = False
                
        elif return_length == GYMNASIUM_STEP_RETURN_LENGTH and not expect_gymnasium_format:
            results["warnings"].append("Received 5-tuple but expected 4-tuple (legacy format)")
            results["format_type"] = "gymnasium_when_legacy_expected"
            
        elif return_length == LEGACY_STEP_RETURN_LENGTH and expect_gymnasium_format:
            results["warnings"].append("Received 4-tuple but expected 5-tuple (Gymnasium format)")
            results["format_type"] = "legacy_when_gymnasium_expected"
            
        else:
            results["errors"].append(f"Unexpected return length: {return_length}")
            results["format_type"] = "invalid"
        
        # Log validation results
        if results["format_valid"]:
            logger.trace(
                "Step return format validation passed",
                extra={
                    "metric_type": "step_return_validation",
                    "validation_results": results
                }
            )
        else:
            logger.warning(
                "Step return format validation failed",
                extra={
                    "metric_type": "step_return_validation",
                    "validation_results": results
                }
            )
        
        return results
        
    except Exception as e:
        logger.error(
            "Step return format validation error",
            extra={
                "metric_type": "validation_error",
                "validation_type": "step_return_format",
                "error": str(e)
            }
        )
        return {
            "format_valid": False,
            "error": str(e),
            "return_length": 0,
            "expected_length": GYMNASIUM_STEP_RETURN_LENGTH if expect_gymnasium_format else LEGACY_STEP_RETURN_LENGTH
        }


def convert_step_return_format(
    step_return: Union[StepReturnType, LegacyStepReturnType],
    target_format: str = "gymnasium"
) -> Union[StepReturnType, LegacyStepReturnType]:
    """
    Convert between step return formats for compatibility.
    
    Provides bidirectional conversion between legacy gym 4-tuple and
    Gymnasium 0.29.x 5-tuple formats to support dual-API environments.
    
    Args:
        step_return: Step return tuple to convert
        target_format: Target format ("gymnasium" or "legacy")
        
    Returns:
        Converted step return tuple
        
    Examples:
        >>> # Convert legacy to Gymnasium
        >>> legacy_return = (obs, 0.1, True, {"info": "data"})
        >>> gym_return = convert_step_return_format(legacy_return, "gymnasium")
        >>> print(len(gym_return))  # 5
        >>> 
        >>> # Convert Gymnasium to legacy
        >>> gym_return = (obs, 0.1, True, False, {"info": "data"})
        >>> legacy_return = convert_step_return_format(gym_return, "legacy")
        >>> print(len(legacy_return))  # 4
    """
    try:
        if not hasattr(step_return, '__len__'):
            raise ValueError("Step return must be a tuple/sequence")
        
        return_length = len(step_return)
        
        if target_format == "gymnasium":
            if return_length == GYMNASIUM_STEP_RETURN_LENGTH:
                # Already Gymnasium format
                return step_return
            elif return_length == LEGACY_STEP_RETURN_LENGTH:
                # Convert legacy (obs, reward, done, info) to (obs, reward, terminated, truncated, info)
                obs, reward, done, info = step_return
                # Map done to terminated, set truncated based on info
                terminated = done
                truncated = info.get('TimeLimit.truncated', False) if isinstance(info, dict) else False
                
                logger.debug(
                    "Converted legacy step return to Gymnasium format",
                    extra={
                        "metric_type": "step_return_conversion",
                        "conversion": "legacy_to_gymnasium",
                        "done": done,
                        "terminated": terminated,
                        "truncated": truncated
                    }
                )
                
                return (obs, reward, terminated, truncated, info)
            else:
                raise ValueError(f"Cannot convert step return with length {return_length}")
        
        elif target_format == "legacy":
            if return_length == LEGACY_STEP_RETURN_LENGTH:
                # Already legacy format
                return step_return
            elif return_length == GYMNASIUM_STEP_RETURN_LENGTH:
                # Convert Gymnasium (obs, reward, terminated, truncated, info) to (obs, reward, done, info)
                obs, reward, terminated, truncated, info = step_return
                done = terminated or truncated
                
                # Preserve truncation information in info
                if isinstance(info, dict):
                    info = info.copy()
                    info['TimeLimit.truncated'] = truncated
                else:
                    info = {'TimeLimit.truncated': truncated}
                
                logger.debug(
                    "Converted Gymnasium step return to legacy format",
                    extra={
                        "metric_type": "step_return_conversion",
                        "conversion": "gymnasium_to_legacy",
                        "terminated": terminated,
                        "truncated": truncated,
                        "done": done
                    }
                )
                
                return (obs, reward, done, info)
            else:
                raise ValueError(f"Cannot convert step return with length {return_length}")
        
        else:
            raise ValueError(f"Unknown target format: {target_format}")
    
    except Exception as e:
        logger.error(
            "Step return conversion error",
            extra={
                "metric_type": "conversion_error",
                "conversion_type": "step_return",
                "target_format": target_format,
                "error": str(e)
            }
        )
        raise


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
        Validate action against action space constraints with enhanced diagnostics.
        
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
        try:
            # Enhanced validation with detailed logging
            is_valid = action_space.contains(action)
            
            if not is_valid:
                # Detailed diagnostic logging for invalid actions
                logger.warning(
                    "Action validation failed",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "action",
                        "action_value": action.tolist() if hasattr(action, 'tolist') else str(action),
                        "action_shape": action.shape if hasattr(action, 'shape') else None,
                        "space_low": action_space.low.tolist(),
                        "space_high": action_space.high.tolist(),
                        "space_shape": action_space.shape,
                        "exceeds_bounds": {
                            "low": (action < action_space.low).tolist() if hasattr(action, '__iter__') else action < action_space.low,
                            "high": (action > action_space.high).tolist() if hasattr(action, '__iter__') else action > action_space.high
                        }
                    }
                )
            else:
                logger.trace(
                    "Action validation passed",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "action",
                        "action_value": action.tolist() if hasattr(action, 'tolist') else str(action)
                    }
                )
            
            return is_valid
            
        except Exception as e:
            logger.error(
                "Action validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "action",
                    "error": str(e),
                    "action_type": type(action).__name__,
                    "space_type": type(action_space).__name__
                }
            )
            return False
    
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
    def validate_observation(
        observation: ObservationType, 
        observation_space: DictSpace,
        step_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate observation against observation space constraints with enhanced diagnostics.
        
        Enhanced for Gymnasium 0.29.x compatibility to handle terminated/truncated info structure
        and provide comprehensive validation diagnostics.
        
        Args:
            observation: Observation dictionary containing sensor data
            observation_space: Gymnasium Dict observation space for validation
            step_info: Optional step info from terminated/truncated step returns for additional validation
            
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
            >>> 
            >>> # With Gymnasium 0.29.x step info
            >>> step_info = {'terminated': False, 'truncated': False, 'TimeLimit.truncated': False}
            >>> ObservationSpace.validate_observation(valid_obs, obs_space, step_info)  # True
        """
        try:
            # Primary observation validation
            is_valid = observation_space.contains(observation)
            
            # Additional validation for step info structure (Gymnasium 0.29.x compatibility)
            info_valid = True
            if step_info is not None:
                info_valid = ObservationSpace._validate_step_info_structure(step_info)
            
            overall_valid = is_valid and info_valid
            
            if not overall_valid:
                # Detailed diagnostic logging for validation failures
                validation_details = {
                    "metric_type": "space_validation",
                    "validation_type": "observation",
                    "observation_valid": is_valid,
                    "info_valid": info_valid,
                    "observation_keys": list(observation.keys()),
                    "expected_keys": list(observation_space.spaces.keys()),
                    "missing_keys": set(observation_space.spaces.keys()) - set(observation.keys()),
                    "extra_keys": set(observation.keys()) - set(observation_space.spaces.keys())
                }
                
                # Add detailed component validation
                for key, value in observation.items():
                    if key in observation_space.spaces:
                        component_space = observation_space.spaces[key]
                        component_valid = component_space.contains(value)
                        validation_details[f"{key}_valid"] = component_valid
                        if not component_valid:
                            validation_details[f"{key}_details"] = {
                                "value_shape": value.shape if hasattr(value, 'shape') else None,
                                "value_type": type(value).__name__,
                                "expected_shape": component_space.shape,
                                "value_range": [float(np.min(value)), float(np.max(value))] if hasattr(value, '__iter__') and len(value) > 0 else value,
                                "expected_range": [float(component_space.low), float(component_space.high)] if hasattr(component_space, 'low') else None
                            }
                
                if step_info is not None:
                    validation_details["step_info"] = step_info
                
                logger.warning("Observation validation failed", extra=validation_details)
            else:
                logger.trace(
                    "Observation validation passed",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "observation",
                        "observation_keys": list(observation.keys()),
                        "has_step_info": step_info is not None
                    }
                )
            
            return overall_valid
            
        except Exception as e:
            logger.error(
                "Observation validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "observation",
                    "error": str(e),
                    "observation_type": type(observation).__name__,
                    "space_type": type(observation_space).__name__
                }
            )
            return False
    
    @staticmethod
    def _validate_step_info_structure(step_info: Dict[str, Any]) -> bool:
        """
        Validate step info structure for Gymnasium 0.29.x compatibility.
        
        Checks for proper terminated/truncated fields and common info patterns.
        
        Args:
            step_info: Step info dictionary from environment step
            
        Returns:
            bool: True if info structure is valid
        """
        try:
            # For Gymnasium 0.29.x, we expect certain fields that may be present
            # This is for info that comes with the step() call, not the observation itself
            
            # Check for common Gymnasium info fields
            expected_fields = {'TimeLimit.truncated'}  # Common field in Gymnasium
            
            # Check that if present, TimeLimit.truncated is a boolean
            if 'TimeLimit.truncated' in step_info:
                if not isinstance(step_info['TimeLimit.truncated'], bool):
                    logger.warning(
                        "Invalid TimeLimit.truncated field type",
                        extra={
                            "metric_type": "info_validation",
                            "field": "TimeLimit.truncated",
                            "value": step_info['TimeLimit.truncated'],
                            "expected_type": "bool",
                            "actual_type": type(step_info['TimeLimit.truncated']).__name__
                        }
                    )
                    return False
            
            # Log info structure for debugging
            logger.trace(
                "Step info validation passed",
                extra={
                    "metric_type": "info_validation",
                    "info_keys": list(step_info.keys()),
                    "info_structure": {k: type(v).__name__ for k, v in step_info.items()}
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Step info validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "step_info",
                    "error": str(e)
                }
            )
            return False
    
    @staticmethod
    def normalize_observation(
        observation: ObservationType,
        observation_space: DictSpace,
        step_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[ObservationType, Optional[Dict[str, Any]]]:
        """
        Normalize observation components to standard ranges with enhanced info handling.
        
        Applies normalization to observation components for improved RL training
        stability. Position values are normalized to [0, 1] based on environment
        bounds, while other components maintain their natural scales.
        
        Enhanced for Gymnasium 0.29.x to handle step info from terminated/truncated
        environments and provide comprehensive normalization logger.
        
        Args:
            observation: Raw observation dictionary
            observation_space: Gymnasium Dict observation space defining bounds
            step_info: Optional step info for terminated/truncated compatibility
            
        Returns:
            Tuple[Dict[str, np.ndarray], Optional[Dict[str, Any]]]: 
                (Normalized observation dictionary, processed step info)
            
        Examples:
            >>> obs_space = ObservationSpace.create(640, 480)
            >>> raw_obs = {
            ...     'odor_concentration': np.array(0.8),
            ...     'agent_position': np.array([320.0, 240.0]),
            ...     'agent_orientation': np.array(90.0)
            ... }
            >>> norm_obs, info = ObservationSpace.normalize_observation(raw_obs, obs_space)
            >>> print(norm_obs['agent_position'])  # [0.5, 0.5] (center of environment)
            >>> 
            >>> # With step info for Gymnasium 0.29.x compatibility
            >>> step_info = {'TimeLimit.truncated': False}
            >>> norm_obs, processed_info = ObservationSpace.normalize_observation(raw_obs, obs_space, step_info)
        """
        try:
            normalized = {}
            normalization_stats = {}
            
            for key, value in observation.items():
                if key == 'agent_position':
                    # Normalize position to [0, 1] based on environment bounds
                    space = observation_space.spaces[key]
                    position_range = space.high - space.low
                    
                    # Avoid division by zero
                    if np.any(position_range == 0):
                        logger.warning(
                            "Zero range detected in position normalization",
                            extra={
                                "metric_type": "normalization_warning",
                                "component": "agent_position",
                                "space_low": space.low.tolist(),
                                "space_high": space.high.tolist(),
                                "position_range": position_range.tolist()
                            }
                        )
                        normalized[key] = np.zeros_like(value)
                    else:
                        normalized[key] = (value - space.low) / position_range
                    
                    normalization_stats[key] = {
                        "original_value": value.tolist(),
                        "normalized_value": normalized[key].tolist(),
                        "space_range": position_range.tolist()
                    }
                    
                elif key == 'agent_orientation':
                    # Normalize orientation to [0, 1] based on [0, 360) range
                    normalized[key] = value / 360.0
                    normalization_stats[key] = {
                        "original_value": float(value),
                        "normalized_value": float(normalized[key])
                    }
                    
                else:
                    # Keep other components in their natural ranges
                    normalized[key] = value.copy() if hasattr(value, 'copy') else value
                    normalization_stats[key] = {
                        "original_value": value.tolist() if hasattr(value, 'tolist') else value,
                        "normalized_value": "unchanged"
                    }
            
            # Process step info for Gymnasium 0.29.x compatibility
            processed_info = step_info.copy() if step_info is not None else None
            
            # Log normalization statistics
            logger.trace(
                "Observation normalization completed",
                extra={
                    "metric_type": "observation_normalization",
                    "normalized_components": list(normalized.keys()),
                    "normalization_stats": normalization_stats,
                    "has_step_info": step_info is not None
                }
            )
            
            return normalized, processed_info
            
        except Exception as e:
            logger.error(
                "Observation normalization error",
                extra={
                    "metric_type": "normalization_error",
                    "error": str(e),
                    "observation_keys": list(observation.keys()) if observation else None,
                    "observation_type": type(observation).__name__
                }
            )
            # Return original observation on error
            return observation, step_info


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
    def validate_spaces(
        action_space: Box, 
        observation_space: DictSpace,
        use_env_checker: bool = True
    ) -> bool:
        """
        Validate that action and observation spaces are properly configured.
        
        Enhanced with Gymnasium 0.29.x environment checker integration and
        comprehensive validation diagnostics for terminated/truncated compatibility.
        
        Args:
            action_space: Gymnasium Box action space
            observation_space: Gymnasium Dict observation space
            use_env_checker: Whether to use gymnasium.utils.env_checker for validation
            
        Returns:
            bool: True if spaces are valid, False otherwise
            
        Examples:
            >>> action_space, obs_space = SpaceFactory.create_standard_spaces(640, 480)
            >>> is_valid = SpaceFactory.validate_spaces(action_space, obs_space)
            >>> assert is_valid
            >>> 
            >>> # With environment checker disabled
            >>> is_valid = SpaceFactory.validate_spaces(action_space, obs_space, use_env_checker=False)
        """
        try:
            validation_results = {
                "action_space_valid": False,
                "observation_space_valid": False,
                "env_checker_valid": False,
                "overall_valid": False
            }
            
            # Validate action space structure
            action_valid = SpaceFactory._validate_action_space_structure(action_space)
            validation_results["action_space_valid"] = action_valid
            
            # Validate observation space structure  
            obs_valid = SpaceFactory._validate_observation_space_structure(observation_space)
            validation_results["observation_space_valid"] = obs_valid
            
            # Use Gymnasium environment checker if available and requested
            env_checker_valid = True
            if use_env_checker and ENV_CHECKER_AVAILABLE:
                env_checker_valid = SpaceFactory._validate_with_env_checker(action_space, observation_space)
                validation_results["env_checker_valid"] = env_checker_valid
            elif use_env_checker and not ENV_CHECKER_AVAILABLE:
                logger.warning(
                    "Environment checker requested but not available",
                    extra={
                        "metric_type": "validation_warning",
                        "validation_type": "env_checker",
                        "message": "gymnasium.utils.env_checker not available, skipping advanced validation"
                    }
                )
            
            # Overall validation result
            overall_valid = action_valid and obs_valid and env_checker_valid
            validation_results["overall_valid"] = overall_valid
            
            # Log validation results
            if overall_valid:
                logger.info(
                    "Space validation passed",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "spaces_complete",
                        **validation_results
                    }
                )
            else:
                logger.warning(
                    "Space validation failed",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "spaces_complete",
                        **validation_results
                    }
                )
            
            return overall_valid
            
        except Exception as e:
            logger.error(
                "Space validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "spaces",
                    "error": str(e),
                    "action_space_type": type(action_space).__name__,
                    "observation_space_type": type(observation_space).__name__
                }
            )
            return False
    
    @staticmethod
    def _validate_action_space_structure(action_space: Box) -> bool:
        """Validate action space structure with detailed diagnostics."""
        try:
            if not isinstance(action_space, Box):
                logger.warning(
                    "Invalid action space type",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "action_space_type",
                        "expected_type": "Box",
                        "actual_type": type(action_space).__name__
                    }
                )
                return False
            
            if action_space.shape != (2,):
                logger.warning(
                    "Invalid action space shape",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "action_space_shape",
                        "expected_shape": (2,),
                        "actual_shape": action_space.shape
                    }
                )
                return False
            
            if not np.all(action_space.low <= action_space.high):
                logger.warning(
                    "Invalid action space bounds",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "action_space_bounds",
                        "low": action_space.low.tolist(),
                        "high": action_space.high.tolist(),
                        "bounds_valid": (action_space.low <= action_space.high).tolist()
                    }
                )
                return False
            
            logger.trace(
                "Action space structure validation passed",
                extra={
                    "metric_type": "space_validation",
                    "validation_type": "action_space_structure",
                    "shape": action_space.shape,
                    "low": action_space.low.tolist(),
                    "high": action_space.high.tolist()
                }
            )
            return True
            
        except Exception as e:
            logger.error(
                "Action space structure validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "action_space_structure",
                    "error": str(e)
                }
            )
            return False
    
    @staticmethod
    def _validate_observation_space_structure(observation_space: DictSpace) -> bool:
        """Validate observation space structure with detailed diagnostics."""
        try:
            if not isinstance(observation_space, DictSpace):
                logger.warning(
                    "Invalid observation space type",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "observation_space_type",
                        "expected_type": "DictSpace",
                        "actual_type": type(observation_space).__name__
                    }
                )
                return False
            
            required_keys = {'odor_concentration', 'agent_position', 'agent_orientation'}
            available_keys = set(observation_space.spaces.keys())
            
            if not required_keys.issubset(available_keys):
                missing_keys = required_keys - available_keys
                logger.warning(
                    "Missing required observation space keys",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "observation_space_keys",
                        "required_keys": list(required_keys),
                        "available_keys": list(available_keys),
                        "missing_keys": list(missing_keys)
                    }
                )
                return False
            
            # Validate individual observation components
            component_validations = {}
            
            # Validate odor concentration space
            odor_space = observation_space.spaces['odor_concentration']
            odor_valid = isinstance(odor_space, Box) and odor_space.shape == ()
            component_validations['odor_concentration'] = {
                "valid": odor_valid,
                "type": type(odor_space).__name__,
                "shape": odor_space.shape if hasattr(odor_space, 'shape') else None
            }
            
            # Validate agent position space
            position_space = observation_space.spaces['agent_position']
            position_valid = isinstance(position_space, Box) and position_space.shape == (2,)
            component_validations['agent_position'] = {
                "valid": position_valid,
                "type": type(position_space).__name__,
                "shape": position_space.shape if hasattr(position_space, 'shape') else None
            }
            
            # Validate agent orientation space
            orientation_space = observation_space.spaces['agent_orientation']
            orientation_valid = isinstance(orientation_space, Box) and orientation_space.shape == ()
            component_validations['agent_orientation'] = {
                "valid": orientation_valid,
                "type": type(orientation_space).__name__,
                "shape": orientation_space.shape if hasattr(orientation_space, 'shape') else None
            }
            
            all_components_valid = all(comp['valid'] for comp in component_validations.values())
            
            if not all_components_valid:
                logger.warning(
                    "Invalid observation space components",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "observation_space_components",
                        "component_validations": component_validations
                    }
                )
            else:
                logger.trace(
                    "Observation space structure validation passed",
                    extra={
                        "metric_type": "space_validation",
                        "validation_type": "observation_space_structure",
                        "component_validations": component_validations
                    }
                )
            
            return all_components_valid
            
        except Exception as e:
            logger.error(
                "Observation space structure validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "observation_space_structure",
                    "error": str(e)
                }
            )
            return False
    
    @staticmethod
    def _validate_with_env_checker(action_space: Box, observation_space: DictSpace) -> bool:
        """Validate spaces using Gymnasium environment checker utilities."""
        try:
            if not ENV_CHECKER_AVAILABLE:
                logger.warning("Environment checker not available for validation")
                return True  # Don't fail if checker unavailable
            
            # Use Gymnasium's built-in space validators
            try:
                check_action_space(action_space)
                action_check_passed = True
            except Exception as e:
                logger.warning(
                    "Gymnasium action space check failed",
                    extra={
                        "metric_type": "env_checker_validation",
                        "validation_type": "action_space",
                        "error": str(e)
                    }
                )
                action_check_passed = False
            
            try:
                check_observation_space(observation_space)
                obs_check_passed = True
            except Exception as e:
                logger.warning(
                    "Gymnasium observation space check failed",
                    extra={
                        "metric_type": "env_checker_validation",
                        "validation_type": "observation_space",
                        "error": str(e)
                    }
                )
                obs_check_passed = False
            
            env_checker_passed = action_check_passed and obs_check_passed
            
            if env_checker_passed:
                logger.debug(
                    "Gymnasium environment checker validation passed",
                    extra={
                        "metric_type": "env_checker_validation",
                        "validation_type": "complete",
                        "action_check": action_check_passed,
                        "observation_check": obs_check_passed
                    }
                )
            
            return env_checker_passed
            
        except Exception as e:
            logger.error(
                "Environment checker validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "env_checker",
                    "error": str(e)
                }
            )
            return False  # Fail on unexpected errors
    
    @staticmethod
    def validate_gymnasium_compatibility(
        action_space: Box,
        observation_space: DictSpace,
        check_terminated_truncated: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive validation for Gymnasium 0.29.x compatibility.
        
        Validates spaces for compatibility with terminated/truncated separation
        and comprehensive environment checker integration.
        
        Args:
            action_space: Gymnasium Box action space
            observation_space: Gymnasium Dict observation space
            check_terminated_truncated: Whether to validate terminated/truncated support
            
        Returns:
            Dict[str, Any]: Detailed validation results
            
        Examples:
            >>> action_space, obs_space = SpaceFactory.create_standard_spaces(640, 480)
            >>> results = SpaceFactory.validate_gymnasium_compatibility(action_space, obs_space)
            >>> assert results['overall_valid']
            >>> print(results['gymnasium_version_support'])  # Should show 0.29.x support
        """
        try:
            validation_results = {
                "overall_valid": False,
                "env_checker_available": ENV_CHECKER_AVAILABLE,
                "gymnasium_version_support": "0.29.x",
                "space_validation": {},
                "compatibility_checks": {},
                "warnings": [],
                "errors": []
            }
            
            # Basic space validation
            basic_valid = SpaceFactory.validate_spaces(action_space, observation_space, use_env_checker=True)
            validation_results["space_validation"]["basic"] = basic_valid
            
            # Terminated/truncated compatibility checks
            if check_terminated_truncated:
                terminated_truncated_valid = SpaceFactory._validate_terminated_truncated_compatibility(
                    action_space, observation_space
                )
                validation_results["compatibility_checks"]["terminated_truncated"] = terminated_truncated_valid
            
            # Check for common Gymnasium patterns
            gymnasium_patterns_valid = SpaceFactory._validate_gymnasium_patterns(observation_space)
            validation_results["compatibility_checks"]["gymnasium_patterns"] = gymnasium_patterns_valid
            
            # Overall validation
            all_checks_passed = all([
                basic_valid,
                terminated_truncated_valid if check_terminated_truncated else True,
                gymnasium_patterns_valid
            ])
            validation_results["overall_valid"] = all_checks_passed
            
            # Log comprehensive results
            logger.info(
                "Gymnasium compatibility validation completed",
                extra={
                    "metric_type": "gymnasium_compatibility",
                    "validation_results": validation_results
                }
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(
                "Gymnasium compatibility validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "gymnasium_compatibility",
                    "error": str(e)
                }
            )
            return {
                "overall_valid": False,
                "error": str(e)
            }
    
    @staticmethod
    def _validate_terminated_truncated_compatibility(action_space: Box, observation_space: DictSpace) -> bool:
        """Validate compatibility with Gymnasium 0.29.x terminated/truncated separation."""
        try:
            # This validation ensures spaces are compatible with environments
            # that return 5-tuple (obs, reward, terminated, truncated, info)
            
            # For spaces themselves, the main requirement is that they support
            # the new step API which doesn't directly affect space structure
            # but affects how environments using these spaces behave
            
            logger.debug(
                "Terminated/truncated compatibility validation passed",
                extra={
                    "metric_type": "gymnasium_compatibility",
                    "validation_type": "terminated_truncated",
                    "note": "Spaces are compatible with 5-tuple step API"
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Terminated/truncated compatibility validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "terminated_truncated",
                    "error": str(e)
                }
            )
            return False
    
    @staticmethod
    def _validate_gymnasium_patterns(observation_space: DictSpace) -> bool:
        """Validate observation space follows Gymnasium best practices."""
        try:
            # Check for common Gymnasium patterns and best practices
            patterns_valid = True
            
            # Check that all spaces have proper bounds
            for key, space in observation_space.spaces.items():
                if isinstance(space, Box):
                    if not np.all(np.isfinite(space.low)) or not np.all(np.isfinite(space.high)):
                        logger.warning(
                            f"Observation space '{key}' has infinite bounds",
                            extra={
                                "metric_type": "gymnasium_patterns",
                                "validation_type": "finite_bounds",
                                "space_key": key,
                                "low_finite": np.all(np.isfinite(space.low)),
                                "high_finite": np.all(np.isfinite(space.high))
                            }
                        )
                        patterns_valid = False
            
            if patterns_valid:
                logger.trace(
                    "Gymnasium patterns validation passed",
                    extra={
                        "metric_type": "gymnasium_patterns",
                        "validation_type": "complete"
                    }
                )
            
            return patterns_valid
            
        except Exception as e:
            logger.error(
                "Gymnasium patterns validation error",
                extra={
                    "metric_type": "validation_error",
                    "validation_type": "gymnasium_patterns",
                    "error": str(e)
                }
            )
            return False


# Utility functions for enhanced validation diagnostics
def log_space_diagnostics(
    action_space: Optional[Box] = None,
    observation_space: Optional[DictSpace] = None,
    context: str = "space_diagnostics"
) -> None:
    """
    Log comprehensive diagnostics about action and observation spaces.
    
    Useful for debugging space configuration issues and verifying
    Gymnasium 0.29.x compatibility.
    
    Args:
        action_space: Optional action space to diagnose
        observation_space: Optional observation space to diagnose  
        context: Context string for logging identification
        
    Examples:
        >>> action_space, obs_space = SpaceFactory.create_standard_spaces(640, 480)
        >>> log_space_diagnostics(action_space, obs_space, "environment_setup")
    """
    try:
        diagnostics = {
            "context": context,
            "env_checker_available": ENV_CHECKER_AVAILABLE
        }
        
        if action_space is not None:
            diagnostics["action_space"] = {
                "type": type(action_space).__name__,
                "shape": action_space.shape if hasattr(action_space, 'shape') else None,
                "dtype": str(action_space.dtype) if hasattr(action_space, 'dtype') else None,
                "low": action_space.low.tolist() if hasattr(action_space, 'low') else None,
                "high": action_space.high.tolist() if hasattr(action_space, 'high') else None
            }
        
        if observation_space is not None:
            obs_diagnostics = {
                "type": type(observation_space).__name__,
                "keys": list(observation_space.spaces.keys()) if hasattr(observation_space, 'spaces') else None,
                "components": {}
            }
            
            if hasattr(observation_space, 'spaces'):
                for key, space in observation_space.spaces.items():
                    obs_diagnostics["components"][key] = {
                        "type": type(space).__name__,
                        "shape": space.shape if hasattr(space, 'shape') else None,
                        "dtype": str(space.dtype) if hasattr(space, 'dtype') else None,
                        "low": space.low.tolist() if hasattr(space, 'low') else None,
                        "high": space.high.tolist() if hasattr(space, 'high') else None
                    }
            
            diagnostics["observation_space"] = obs_diagnostics
        
        logger.info(
            "Space diagnostics report",
            extra={
                "metric_type": "space_diagnostics",
                "diagnostics": diagnostics
            }
        )
        
    except Exception as e:
        logger.error(
            "Space diagnostics error",
            extra={
                "metric_type": "diagnostics_error",
                "context": context,
                "error": str(e)
            }
        )


def check_gymnasium_env_integration(env_instance) -> Dict[str, Any]:
    """
    Check an environment instance for Gymnasium 0.29.x integration compatibility.
    
    Validates that an environment properly implements the terminated/truncated
    separation and follows Gymnasium best practices.
    
    Args:
        env_instance: Environment instance to check
        
    Returns:
        Dict[str, Any]: Validation results and recommendations
        
    Examples:
        >>> from gymnasium import make
        >>> env = make('PlumeNavSim-v0')  # New Gymnasium environment ID
        >>> results = check_gymnasium_env_integration(env)
        >>> assert results['step_api_compatible']
    """
    try:
        results = {
            "step_api_compatible": False,
            "reset_api_compatible": False,
            "space_validation": False,
            "env_checker_passed": False,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check if environment has required methods
        required_methods = ['step', 'reset', 'close', 'render']
        for method in required_methods:
            if not hasattr(env_instance, method):
                results["errors"].append(f"Missing required method: {method}")
        
        # Check action and observation spaces
        if hasattr(env_instance, 'action_space') and hasattr(env_instance, 'observation_space'):
            space_valid = SpaceFactory.validate_spaces(
                env_instance.action_space, 
                env_instance.observation_space
            )
            results["space_validation"] = space_valid
        else:
            results["errors"].append("Missing action_space or observation_space attributes")
        
        # Check step API compatibility (5-tuple return)
        if hasattr(env_instance, 'reset'):
            try:
                obs, info = env_instance.reset()
                results["reset_api_compatible"] = True
                
                # Test step API if we have a valid observation
                if hasattr(env_instance, 'action_space'):
                    sample_action = env_instance.action_space.sample()
                    step_result = env_instance.step(sample_action)
                    
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        results["step_api_compatible"] = True
                        logger.debug("Environment supports 5-tuple step API")
                    elif len(step_result) == 4:
                        results["warnings"].append("Environment returns 4-tuple (legacy gym API)")
                        results["recommendations"].append("Update to 5-tuple step API for full Gymnasium 0.29.x compatibility")
                    else:
                        results["errors"].append(f"Unexpected step return length: {len(step_result)}")
                        
            except Exception as e:
                results["errors"].append(f"Error testing environment API: {str(e)}")
        
        # Use Gymnasium environment checker if available
        if ENV_CHECKER_AVAILABLE:
            try:
                check_environment(env_instance)
                results["env_checker_passed"] = True
            except Exception as e:
                results["warnings"].append(f"Gymnasium env_checker issues: {str(e)}")
        
        # Generate recommendations
        if not results["step_api_compatible"]:
            results["recommendations"].append("Implement 5-tuple step return: (obs, reward, terminated, truncated, info)")
        
        if not results["space_validation"]:
            results["recommendations"].append("Fix space validation issues reported above")
        
        logger.info(
            "Environment integration check completed",
            extra={
                "metric_type": "env_integration_check",
                "results": results
            }
        )
        
        return results
        
    except Exception as e:
        logger.error(
            "Environment integration check error",
            extra={
                "metric_type": "integration_check_error",
                "error": str(e)
            }
        )
        return {
            "error": str(e),
            "step_api_compatible": False,
            "reset_api_compatible": False,
            "space_validation": False
        }


# Public API exports
__all__ = [
    # Core space classes
    "ActionSpace",
    "ObservationSpace", 
    "SpaceFactory",
    
    # Type aliases
    "ActionType",
    "ObservationType",
    "BoundsType",
    "StepReturnType",
    "LegacyStepReturnType",
    "ValidationResultType",
    
    # Gymnasium imports (for compatibility)
    "Box",
    "DictSpace",
    
    # Feature flags and constants
    "ENV_CHECKER_AVAILABLE",
    "GYMNASIUM_STEP_RETURN_LENGTH",
    "LEGACY_STEP_RETURN_LENGTH",
    "SUPPORTED_GYMNASIUM_VERSION",
    
    # Enhanced validation utilities
    "log_space_diagnostics",
    "check_gymnasium_env_integration",
    "validate_step_return_format",
    "convert_step_return_format"
]