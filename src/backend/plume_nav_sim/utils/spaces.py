"""
Gymnasium spaces utility module for plume_nav_sim package providing action and observation
space creation, validation, and compatibility utilities with comprehensive type checking,
performance optimization, and Gymnasium API compliance for reinforcement learning
environment integration.

This module implements factory functions for creating Gymnasium-compatible action and
observation spaces, validation utilities for runtime parameter checking, configuration
classes for customizable space creation, and performance-optimized operations with
caching and monitoring capabilities.
"""

import functools  # standard library - Caching decorators for performance optimization of space creation and validation operations
import time  # standard library - Performance timing for validation and creation operations
import warnings  # standard library - Space compatibility warnings and deprecation notifications for API changes
from dataclasses import (  # >=3.10 - Dataclass decorator for structured configuration management
    dataclass,
)
from typing import (  # >=3.10 - Type hints for space factory functions, validation methods, and parameter specifications
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# External imports with version comments
import gymnasium.spaces  # >=0.29.0 - Core Gymnasium space classes including Discrete and Box for action and observation space creation with standard RL API compliance
import numpy as np  # >=2.1.0 - Array operations, dtype specifications, and mathematical operations for space bounds and observation validation

from ..core.constants import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SPACE_SIZE,
    ACTION_UP,
    CONCENTRATION_RANGE,
    OBSERVATION_DTYPE,
)

# Internal imports from core module
from ..core.enums import Action
from ..core.types import ActionType, ObservationType

# Internal imports from utils module
from .exceptions import ValidationError
from .logging import ComponentType, get_component_logger

# Global constants for caching and performance
SPACE_VALIDATION_CACHE_SIZE = 1000
SPACE_CREATION_CACHE_SIZE = 100
DEFAULT_OBSERVATION_SHAPE = (1,)
SPACE_VALIDATION_TIMEOUT_MS = 10.0

# Module exports
__all__ = [
    "create_action_space",
    "create_observation_space",
    "validate_action",
    "validate_observation",
    "validate_action_space",
    "validate_observation_space",
    "SpaceConfig",
    "SpaceValidator",
    "get_space_info",
    "sample_valid_action",
    "sample_valid_observation",
    "check_space_compatibility",
    "optimize_space_operations",
    "is_space_subset",
]

# Initialize component logger for space operations and validation monitoring
logger = get_component_logger("validation", ComponentType.UTILS)


def create_action_space(
    num_actions: Optional[int] = None,
    validate_actions: bool = True,
    space_config: Optional[Dict] = None,
) -> gymnasium.spaces.Discrete:
    """
    Creates Gymnasium Discrete action space for cardinal direction navigation with validation,
    caching, and performance optimization for environment initialization.

    Args:
        num_actions: Number of discrete actions (defaults to ACTION_SPACE_SIZE)
        validate_actions: Whether to validate action space properties
        space_config: Custom space configuration dictionary

    Returns:
        Configured Discrete action space for 4-direction agent movement with validation and metadata

    Raises:
        ValidationError: If num_actions is invalid or space creation fails
        TypeError: If parameters are of incorrect type
    """
    if space_config:
        return _create_action_space_impl(num_actions, validate_actions, space_config)

    return _create_action_space_cached(num_actions, validate_actions)


def is_space_subset(
    policy_space: gymnasium.spaces.Space, env_space: gymnasium.spaces.Space
) -> bool:
    """Check if policy_space is a subset of env_space (structural, no sampling).

    Supports common Gymnasium spaces:
    - Discrete: policy.n <= env.n
    - MultiDiscrete: elementwise nvec <= nvec
    - MultiBinary: policy.n <= env.n
    - Box: same shape and bounds within env [low, high]
    - Tuple: same length and pairwise subset
    - Dict: policy keys subset of env keys and pairwise subset per key
    """
    # Trivial identity
    if policy_space is env_space:
        return True

    # Discrete
    if isinstance(policy_space, gymnasium.spaces.Discrete) and isinstance(
        env_space, gymnasium.spaces.Discrete
    ):
        return int(policy_space.n) <= int(env_space.n)

    # MultiDiscrete
    if isinstance(policy_space, gymnasium.spaces.MultiDiscrete) and isinstance(
        env_space, gymnasium.spaces.MultiDiscrete
    ):
        p = np.asarray(policy_space.nvec, dtype=np.int64)
        e = np.asarray(env_space.nvec, dtype=np.int64)
        return p.shape == e.shape and np.all(p <= e)

    # MultiBinary
    if isinstance(policy_space, gymnasium.spaces.MultiBinary) and isinstance(
        env_space, gymnasium.spaces.MultiBinary
    ):
        return int(policy_space.n) <= int(env_space.n)

    # Box (require identical shapes and tighter/equal bounds)
    if isinstance(policy_space, gymnasium.spaces.Box) and isinstance(
        env_space, gymnasium.spaces.Box
    ):
        if tuple(policy_space.shape) != tuple(env_space.shape):
            return False
        p_low = np.asarray(policy_space.low)
        p_high = np.asarray(policy_space.high)
        e_low = np.asarray(env_space.low)
        e_high = np.asarray(env_space.high)
        return bool(np.all(p_low >= e_low) and np.all(p_high <= e_high))

    # Tuple
    if isinstance(policy_space, gymnasium.spaces.Tuple) and isinstance(
        env_space, gymnasium.spaces.Tuple
    ):
        if len(policy_space.spaces) != len(env_space.spaces):
            return False
        return all(
            is_space_subset(ps, es)
            for ps, es in zip(policy_space.spaces, env_space.spaces)
        )

    # Dict
    if isinstance(policy_space, gymnasium.spaces.Dict) and isinstance(
        env_space, gymnasium.spaces.Dict
    ):
        p_keys = set(policy_space.spaces.keys())
        e_keys = set(env_space.spaces.keys())
        if not p_keys.issubset(e_keys):
            return False
        return all(
            is_space_subset(policy_space.spaces[k], env_space.spaces[k]) for k in p_keys
        )

    # Unknown space types: be conservative
    return False


@functools.lru_cache(maxsize=SPACE_CREATION_CACHE_SIZE)
def _create_action_space_cached(
    num_actions: Optional[int], validate_actions: bool
) -> gymnasium.spaces.Discrete:
    return _create_action_space_impl(num_actions, validate_actions, None)


def _create_action_space_impl(  # noqa: C901
    num_actions: Optional[int], validate_actions: bool, space_config: Optional[Dict]
) -> gymnasium.spaces.Discrete:
    start_time = time.time()
    final_num_actions = num_actions if num_actions is not None else ACTION_SPACE_SIZE
    config = dict(space_config or {})

    try:
        if not isinstance(final_num_actions, int) or final_num_actions <= 0:
            raise ValidationError(
                f"Invalid num_actions: {final_num_actions}. Must be positive integer.",
                context={
                    "expected_type": "int",
                    "expected_range": "> 0",
                    "received": final_num_actions,
                },
            )

        if final_num_actions != ACTION_SPACE_SIZE:
            warnings.warn(
                f"Creating action space with {final_num_actions} actions, but standard size is {ACTION_SPACE_SIZE}",
                UserWarning,
                stacklevel=2,
            )

        action_space = gymnasium.spaces.Discrete(final_num_actions)

        action_metadata = {
            "action_names": ["up", "right", "down", "left"][:final_num_actions],
            "movement_vectors": [(0, -1), (1, 0), (0, 1), (-1, 0)][:final_num_actions],
            "space_type": "discrete",
            "creation_time": time.time(),
        }

        if hasattr(action_space, "_metadata"):
            action_space._metadata.update(action_metadata)
        else:
            action_space._metadata = action_metadata

        if validate_actions:
            validation_result = validate_action_space(action_space, check_metadata=True)
            if not validation_result:
                raise ValidationError(
                    "Action space validation failed after creation",
                    context={
                        "num_actions": final_num_actions,
                        "space_type": type(action_space).__name__,
                    },
                )

        if config:
            seed_value = config.get("seed")
            if seed_value is not None:
                action_space.seed(seed_value)

            custom_metadata = config.get("custom_metadata")
            if custom_metadata:
                action_metadata.update(custom_metadata)

        elapsed_time = (time.time() - start_time) * 1000
        logger.info(
            f"Created action space with {final_num_actions} actions",
            extra={
                "duration_ms": elapsed_time,
                "space_size": final_num_actions,
                "validation_enabled": validate_actions,
                "custom_config": bool(config),
            },
        )

        return action_space

    except Exception as e:
        logger.error(
            f"Failed to create action space: {e}",
            extra={
                "num_actions": final_num_actions,
                "validate_actions": validate_actions,
                "error_type": type(e).__name__,
            },
        )
        raise


def create_observation_space(
    observation_shape: Optional[Tuple] = None,
    concentration_bounds: Optional[Tuple] = None,
    observation_dtype: Optional[np.dtype] = None,
    validate_bounds: bool = True,
    space_config: Optional[Dict] = None,
) -> gymnasium.spaces.Box:
    """
    Creates Gymnasium Box observation space for concentration values with proper bounds,
    dtype, and shape configuration for plume navigation observations.

    Args:
        observation_shape: Shape of observation array (defaults to DEFAULT_OBSERVATION_SHAPE)
        concentration_bounds: Low and high bounds for concentration values
        observation_dtype: NumPy dtype for observations
        validate_bounds: Whether to validate observation space properties
        space_config: Custom space configuration dictionary

    Returns:
        Configured Box observation space for concentration values with proper bounds and dtype specifications

    Raises:
        ValidationError: If bounds or shape are invalid
        TypeError: If dtype is not supported
    """
    if space_config:
        return _create_observation_space_impl(
            observation_shape,
            concentration_bounds,
            observation_dtype,
            validate_bounds,
            space_config,
        )

    return _create_observation_space_cached(
        observation_shape,
        concentration_bounds,
        observation_dtype,
        validate_bounds,
    )


@functools.lru_cache(maxsize=SPACE_CREATION_CACHE_SIZE)
def _create_observation_space_cached(
    observation_shape: Optional[Tuple],
    concentration_bounds: Optional[Tuple],
    observation_dtype: Optional[np.dtype],
    validate_bounds: bool,
) -> gymnasium.spaces.Box:
    return _create_observation_space_impl(
        observation_shape,
        concentration_bounds,
        observation_dtype,
        validate_bounds,
        None,
    )


def _create_observation_space_impl(  # noqa: C901
    observation_shape: Optional[Tuple],
    concentration_bounds: Optional[Tuple],
    observation_dtype: Optional[np.dtype],
    validate_bounds: bool,
    space_config: Optional[Dict],
) -> gymnasium.spaces.Box:
    start_time = time.time()
    final_shape = (
        observation_shape
        if observation_shape is not None
        else DEFAULT_OBSERVATION_SHAPE
    )
    bounds = (
        concentration_bounds
        if concentration_bounds is not None
        else CONCENTRATION_RANGE
    )
    dtype = observation_dtype if observation_dtype is not None else OBSERVATION_DTYPE
    config = dict(space_config or {})

    try:
        if not isinstance(final_shape, tuple) or len(final_shape) == 0:
            raise ValidationError(
                f"Invalid observation_shape: {final_shape}. Must be non-empty tuple.",
                context={"expected_type": "tuple", "received": final_shape},
            )

        if not all(isinstance(dim, int) and dim > 0 for dim in final_shape):
            raise ValidationError(
                f"All shape dimensions must be positive integers: {final_shape}",
                context={"shape": final_shape},
            )

        if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValidationError(
                f"Invalid concentration_bounds: {bounds}. Must be (low, high) with low < high.",
                context={"bounds": bounds},
            )

        try:
            np.dtype(dtype)
        except TypeError as e:
            raise ValidationError(
                f"Invalid observation_dtype: {dtype}",
                context={"dtype": dtype, "error": str(e)},
            )

        low_bound = np.full(final_shape, bounds[0], dtype=dtype)
        high_bound = np.full(final_shape, bounds[1], dtype=dtype)

        observation_space = gymnasium.spaces.Box(
            low=low_bound, high=high_bound, shape=final_shape, dtype=dtype
        )

        observation_metadata = {
            "concentration_bounds": bounds,
            "observation_dtype": dtype,
            "observation_shape": final_shape,
            "space_type": "box",
            "creation_time": time.time(),
        }

        if hasattr(observation_space, "_metadata"):
            observation_space._metadata.update(observation_metadata)
        else:
            observation_space._metadata = observation_metadata

        if validate_bounds:
            validation_result = validate_observation_space(
                observation_space, check_bounds=True, check_dtype=True
            )
            if not validation_result:
                raise ValidationError(
                    "Observation space validation failed after creation",
                    context={"bounds": bounds, "dtype": str(dtype)},
                )

        if config:
            seed_value = config.get("seed")
            if seed_value is not None:
                observation_space.seed(seed_value)

            custom_metadata = config.get("custom_metadata")
            if custom_metadata:
                observation_metadata.update(custom_metadata)

        elapsed_time = (time.time() - start_time) * 1000
        logger.info(
            f"Created observation space with shape {final_shape}",
            extra={
                "duration_ms": elapsed_time,
                "shape": final_shape,
                "bounds": bounds,
                "dtype": str(dtype),
                "validation_enabled": validate_bounds,
                "custom_config": bool(config),
            },
        )

        return observation_space

    except Exception as e:
        logger.error(
            f"Failed to create observation space: {e}",
            extra={
                "observation_shape": final_shape,
                "concentration_bounds": bounds,
                "observation_dtype": str(dtype),
                "validate_bounds": validate_bounds,
                "error_type": type(e).__name__,
            },
        )
        raise


def validate_action(  # noqa: C901
    action: ActionType,
    action_space: Optional[gymnasium.spaces.Discrete] = None,
    strict_mode: bool = False,
    log_validation: bool = False,
) -> int:
    """
    Validates action parameter for Discrete action space compliance with comprehensive type
    checking, bounds validation, and performance monitoring for runtime use.

    Args:
        action: Action value to validate (int, Action enum, or ActionType)
        action_space: Optional action space to validate against
        strict_mode: Enable strict validation rules
        log_validation: Whether to log validation activity

    Returns:
        Validated action integer in range [0, 3] ready for environment step processing

    Raises:
        ValidationError: If action is invalid or out of bounds
        TypeError: If action type is not supported
    """
    start_time = time.time() if log_validation else 0

    try:
        # Check if action is None and handle with appropriate error message
        if action is None:
            raise ValidationError(
                "Action type NoneType is invalid; expected integer or Action enum",
                context={
                    "expected_type": "int or Action enum",
                    "received": None,
                    "received_type": "NoneType",
                },
            )

        # Validate action is numeric type (int, numpy.integer, or Action enum) with type conversion
        validated_action = None

        if isinstance(action, Action):
            # Convert Action enum to integer value if action is Action type
            validated_action = action.value
        elif isinstance(action, (int, np.integer)):
            validated_action = int(action)
        elif isinstance(action, (float, np.floating)):
            if not float(action).is_integer():
                raise ValidationError(
                    f"Action value {action} must be an integer",
                    context={
                        "action_value": action,
                        "action_type": type(action).__name__,
                    },
                )
            validated_action = int(action)
        else:
            # Attempt conversion to int
            try:
                validated_action = int(action)
            except (ValueError, TypeError) as e:
                raise ValidationError(
                    f"Action must be convertible to integer: {action}",
                    context={
                        "action_type": type(action).__name__,
                        "action_value": action,
                        "conversion_error": str(e),
                    },
                )

        # Determine allowable action range either from constants or provided action space
        lower_bound = ACTION_UP
        upper_bound = ACTION_LEFT

        if action_space is not None:
            if not isinstance(action_space, gymnasium.spaces.Discrete):
                raise ValidationError(
                    f"Expected Discrete action space, got {type(action_space)}",
                    context={"space_type": type(action_space).__name__},
                )

            lower_bound = 0
            upper_bound = action_space.n - 1

        # Validate action is within valid range for the configured space
        if not (lower_bound <= validated_action <= upper_bound):
            raise ValidationError(
                f"Action {validated_action} out of valid range [{lower_bound}, {upper_bound}]",
                context={
                    "action": validated_action,
                    "valid_range": (lower_bound, upper_bound),
                    "action_meanings": {
                        ACTION_UP: "UP",
                        ACTION_RIGHT: "RIGHT",
                        ACTION_DOWN: "DOWN",
                        ACTION_LEFT: "LEFT",
                    },
                },
            )

        # Check action against provided action_space using space.contains() if available
        if action_space is not None:
            if not action_space.contains(validated_action):
                raise ValidationError(
                    f"Action {validated_action} not contained in action space",
                    context={
                        "action": validated_action,
                        "action_space_n": action_space.n,
                        "valid_range": f"[0, {action_space.n - 1}]",
                    },
                )

        # Apply strict validation rules including additional type checking if strict_mode enabled
        if strict_mode:
            # Additional validation for strict mode
            if validated_action not in [
                ACTION_UP,
                ACTION_RIGHT,
                ACTION_DOWN,
                ACTION_LEFT,
            ]:
                raise ValidationError(
                    f"Strict mode: action {validated_action} not in standard cardinal directions",
                    context={
                        "standard_actions": [
                            ACTION_UP,
                            ACTION_RIGHT,
                            ACTION_DOWN,
                            ACTION_LEFT,
                        ]
                    },
                )

        # Log validation activity with action value and timing if log_validation enabled
        if log_validation:
            elapsed_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Validated action {validated_action}",
                extra={
                    "duration_ms": elapsed_time,
                    "original_action": action,
                    "validated_action": validated_action,
                    "strict_mode": strict_mode,
                },
            )

        return validated_action

    except ValidationError:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during action validation: {e}",
            extra={"action": action, "error_type": type(e).__name__},
        )
        raise ValidationError(
            f"Action validation failed with unexpected error: {e}",
            context={"original_error": str(e), "action": action},
        )


def validate_observation(  # noqa: C901
    observation: ObservationType,
    observation_space: Optional[gymnasium.spaces.Box] = None,
    check_range: bool = True,
    strict_mode: bool = False,
    log_validation: bool = False,
) -> np.ndarray:
    """
    Validates observation parameter for Box observation space compliance with shape, dtype,
    and range checking for concentration value observations.

    Args:
        observation: Observation array to validate
        observation_space: Optional observation space to validate against
        check_range: Whether to check concentration range
        strict_mode: Enable strict validation rules
        log_validation: Whether to log validation activity

    Returns:
        Validated observation array with proper shape, dtype, and concentration range compliance

    Raises:
        ValidationError: If observation is invalid
        TypeError: If observation is not array-like
    """
    start_time = time.time() if log_validation else 0

    try:
        # Check if observation is None and handle with appropriate error message
        if observation is None:
            raise ValidationError(
                "Observation cannot be None",
                context={"expected_type": "numpy.ndarray", "received": None},
            )

        # Validate observation is numpy.ndarray with proper array properties
        if not isinstance(observation, np.ndarray):
            try:
                observation = np.asarray(observation)
            except Exception as e:
                raise ValidationError(
                    f"Cannot convert observation to numpy array: {e}",
                    context={
                        "observation_type": type(observation).__name__,
                        "conversion_error": str(e),
                    },
                )

        # Check observation shape matches expected DEFAULT_OBSERVATION_SHAPE (1,)
        expected_shape = DEFAULT_OBSERVATION_SHAPE
        if observation.shape != expected_shape:
            raise ValidationError(
                f"Observation shape {observation.shape} does not match expected {expected_shape}",
                context={
                    "observation_shape": observation.shape,
                    "expected_shape": expected_shape,
                    "observation_size": observation.size,
                },
            )

        # Validate observation dtype matches OBSERVATION_DTYPE (float32) for consistency
        if observation.dtype != OBSERVATION_DTYPE:
            warnings.warn(
                f"Observation dtype {observation.dtype} differs from expected {OBSERVATION_DTYPE}",
                UserWarning,
                stacklevel=2,
            )

            # Convert to expected dtype
            try:
                observation = observation.astype(OBSERVATION_DTYPE)
            except Exception as e:
                raise ValidationError(
                    f"Cannot convert observation to dtype {OBSERVATION_DTYPE}: {e}",
                    context={
                        "current_dtype": str(observation.dtype),
                        "expected_dtype": str(OBSERVATION_DTYPE),
                        "conversion_error": str(e),
                    },
                )

        # Check concentration values are within CONCENTRATION_RANGE [0.0, 1.0] if check_range enabled
        if check_range:
            min_val, max_val = CONCENTRATION_RANGE
            obs_min, obs_max = observation.min(), observation.max()

            if obs_min < min_val or obs_max > max_val:
                raise ValidationError(
                    f"Observation values [{obs_min:.3f}, {obs_max:.3f}] outside valid range {CONCENTRATION_RANGE}",
                    context={
                        "observation_range": (float(obs_min), float(obs_max)),
                        "valid_range": CONCENTRATION_RANGE,
                        "out_of_bounds_count": np.sum(
                            (observation < min_val) | (observation > max_val)
                        ),
                    },
                )

        # Validate observation against provided observation_space using space.contains() if available
        if observation_space is not None:
            if not isinstance(observation_space, gymnasium.spaces.Box):
                raise ValidationError(
                    f"Expected Box observation space, got {type(observation_space)}",
                    context={"space_type": type(observation_space).__name__},
                )

            if not observation_space.contains(observation):
                raise ValidationError(
                    "Observation not contained in observation space",
                    context={
                        "observation_shape": observation.shape,
                        "observation_dtype": str(observation.dtype),
                        "space_shape": observation_space.shape,
                        "space_dtype": str(observation_space.dtype),
                    },
                )

        # Apply strict validation rules including additional array property checking if strict_mode enabled
        if strict_mode:
            # Additional validation for strict mode
            if not observation.flags.c_contiguous:
                observation = np.ascontiguousarray(observation)

            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                raise ValidationError(
                    "Strict mode: observation contains NaN or infinite values",
                    context={
                        "has_nan": bool(np.any(np.isnan(observation))),
                        "has_inf": bool(np.any(np.isinf(observation))),
                        "nan_count": int(np.sum(np.isnan(observation))),
                        "inf_count": int(np.sum(np.isinf(observation))),
                    },
                )

        # Log validation activity with observation statistics and timing if log_validation enabled
        if log_validation:
            elapsed_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Validated observation with shape {observation.shape}",
                extra={
                    "duration_ms": elapsed_time,
                    "observation_shape": observation.shape,
                    "observation_dtype": str(observation.dtype),
                    "value_range": (float(observation.min()), float(observation.max())),
                    "check_range": check_range,
                    "strict_mode": strict_mode,
                },
            )

        return observation

    except ValidationError:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during observation validation: {e}",
            extra={
                "observation_type": (
                    type(observation).__name__ if observation is not None else None
                ),
                "error_type": type(e).__name__,
            },
        )
        raise ValidationError(
            f"Observation validation failed with unexpected error: {e}",
            context={"original_error": str(e)},
        )


def validate_action_space(  # noqa: C901
    action_space: gymnasium.spaces.Discrete,
    check_metadata: bool = False,
    strict_validation: bool = False,
) -> bool:
    """
    Validates Gymnasium Discrete action space configuration ensuring proper size, metadata,
    and compatibility with environment requirements.

    Args:
        action_space: Discrete action space to validate
        check_metadata: Whether to validate metadata
        strict_validation: Enable strict validation rules

    Returns:
        True if action space is valid and compatible, raises ValidationError if invalid

    Raises:
        ValidationError: If action space is invalid or incompatible
        TypeError: If action_space is not Discrete
    """
    try:
        # Validate action_space is instance of gymnasium.spaces.Discrete
        if not isinstance(action_space, gymnasium.spaces.Discrete):
            raise ValidationError(
                f"Expected gymnasium.spaces.Discrete, got {type(action_space)}",
                context={"space_type": type(action_space).__name__},
            )

        # Check action space n property equals ACTION_SPACE_SIZE (4) for cardinal directions
        if action_space.n != ACTION_SPACE_SIZE:
            if strict_validation:
                raise ValidationError(
                    f"Action space size {action_space.n} does not match expected {ACTION_SPACE_SIZE}",
                    context={
                        "space_size": action_space.n,
                        "expected_size": ACTION_SPACE_SIZE,
                    },
                )
            else:
                warnings.warn(
                    f"Action space size {action_space.n} differs from standard {ACTION_SPACE_SIZE}",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate action space sample() method produces valid actions in expected range
        try:
            for _ in range(10):  # Test multiple samples
                sample = action_space.sample()
                if not isinstance(sample, (int, np.integer)):
                    raise ValidationError(
                        f"Action space sample() returned non-integer: {sample}",
                        context={
                            "sample_type": type(sample).__name__,
                            "sample_value": sample,
                        },
                    )

                if sample < 0 or sample >= action_space.n:
                    raise ValidationError(
                        f"Action space sample {sample} out of valid range [0, {action_space.n-1}]",
                        context={
                            "sample": sample,
                            "valid_range": (0, action_space.n - 1),
                        },
                    )
        except Exception as e:
            raise ValidationError(
                f"Action space sample() method failed: {e}",
                context={"sample_error": str(e)},
            )

        # Check action space contains() method works correctly for all valid actions
        try:
            for action_val in range(action_space.n):
                if not action_space.contains(action_val):
                    raise ValidationError(
                        f"Action space contains() returned False for valid action {action_val}",
                        context={"action": action_val, "space_size": action_space.n},
                    )

            # Test invalid actions
            invalid_actions = [-1, action_space.n, action_space.n + 1]
            for invalid_action in invalid_actions:
                if action_space.contains(invalid_action):
                    raise ValidationError(
                        f"Action space contains() returned True for invalid action {invalid_action}",
                        context={
                            "invalid_action": invalid_action,
                            "space_size": action_space.n,
                        },
                    )
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Action space contains() method failed: {e}",
                context={"contains_error": str(e)},
            )

        # Validate action space metadata and documentation if check_metadata enabled
        if check_metadata:
            if hasattr(action_space, "_metadata"):
                metadata = action_space._metadata
                required_keys = ["action_names", "movement_vectors", "space_type"]
                missing_keys = [key for key in required_keys if key not in metadata]

                if missing_keys:
                    if strict_validation:
                        raise ValidationError(
                            f"Action space metadata missing required keys: {missing_keys}",
                            context={
                                "metadata": metadata,
                                "missing_keys": missing_keys,
                            },
                        )
                    else:
                        warnings.warn(
                            f"Action space metadata missing optional keys: {missing_keys}",
                            UserWarning,
                            stacklevel=2,
                        )

        # Apply strict validation including performance testing if strict_validation enabled
        if strict_validation:
            # Performance test for sample() method
            start_time = time.time()
            for _ in range(1000):
                action_space.sample()
            sample_time = (time.time() - start_time) * 1000

            if sample_time > 10.0:  # 10ms threshold for 1000 samples
                warnings.warn(
                    f"Action space sampling performance slower than expected: {sample_time:.3f}ms for 1000 samples",
                    UserWarning,
                    stacklevel=2,
                )

        # Test action space compatibility with Action enum values
        for action_enum in Action:
            action_value = action_enum.value
            if action_value < action_space.n:
                if not action_space.contains(action_value):
                    raise ValidationError(
                        f"Action space incompatible with Action.{action_enum.name} ({action_value})",
                        context={
                            "action_enum": action_enum.name,
                            "action_value": action_value,
                        },
                    )

        logger.debug(
            "Action space validation passed",
            extra={
                "space_size": action_space.n,
                "check_metadata": check_metadata,
                "strict_validation": strict_validation,
            },
        )

        return True

    except ValidationError:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during action space validation: {e}",
            extra={
                "space_type": type(action_space).__name__ if action_space else None,
                "error_type": type(e).__name__,
            },
        )
        raise ValidationError(
            f"Action space validation failed with unexpected error: {e}",
            context={"original_error": str(e)},
        )


def validate_observation_space(  # noqa: C901
    observation_space: gymnasium.spaces.Box,
    check_bounds: bool = True,
    check_dtype: bool = True,
    strict_validation: bool = False,
) -> bool:
    """
    Validates Gymnasium Box observation space configuration ensuring proper bounds, shape,
    dtype, and compatibility with concentration observations.

    Args:
        observation_space: Box observation space to validate
        check_bounds: Whether to check concentration bounds
        check_dtype: Whether to validate dtype
        strict_validation: Enable strict validation rules

    Returns:
        True if observation space is valid and compatible, raises ValidationError if invalid

    Raises:
        ValidationError: If observation space is invalid or incompatible
        TypeError: If observation_space is not Box
    """
    try:
        # Validate observation_space is instance of gymnasium.spaces.Box
        if not isinstance(observation_space, gymnasium.spaces.Box):
            raise ValidationError(
                f"Expected gymnasium.spaces.Box, got {type(observation_space)}",
                context={"space_type": type(observation_space).__name__},
            )

        # Check observation space shape matches DEFAULT_OBSERVATION_SHAPE (1,)
        expected_shape = DEFAULT_OBSERVATION_SHAPE
        if observation_space.shape != expected_shape:
            if strict_validation:
                raise ValidationError(
                    f"Observation space shape {observation_space.shape} does not match expected {expected_shape}",
                    context={
                        "space_shape": observation_space.shape,
                        "expected_shape": expected_shape,
                    },
                )
            else:
                warnings.warn(
                    f"Observation space shape {observation_space.shape} differs from standard {expected_shape}",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate observation space bounds match CONCENTRATION_RANGE [0.0, 1.0] if check_bounds enabled
        if check_bounds:
            expected_low, expected_high = CONCENTRATION_RANGE

            # Check low bounds
            if not np.allclose(observation_space.low, expected_low, rtol=1e-6):
                if strict_validation:
                    raise ValidationError(
                        f"Observation space low bounds {observation_space.low} do not match expected {expected_low}",
                        context={
                            "space_low": observation_space.low.tolist(),
                            "expected_low": expected_low,
                        },
                    )
                else:
                    warnings.warn(
                        "Observation space low bounds differ from standard concentration range",
                        UserWarning,
                        stacklevel=2,
                    )

            # Check high bounds
            if not np.allclose(observation_space.high, expected_high, rtol=1e-6):
                if strict_validation:
                    raise ValidationError(
                        f"Observation space high bounds {observation_space.high} do not match expected {expected_high}",
                        context={
                            "space_high": observation_space.high.tolist(),
                            "expected_high": expected_high,
                        },
                    )
                else:
                    warnings.warn(
                        "Observation space high bounds differ from standard concentration range",
                        UserWarning,
                        stacklevel=2,
                    )

        # Check observation space dtype matches OBSERVATION_DTYPE (float32) if check_dtype enabled
        if check_dtype:
            expected_dtype = OBSERVATION_DTYPE
            if observation_space.dtype != expected_dtype:
                if strict_validation:
                    raise ValidationError(
                        f"Observation space dtype {observation_space.dtype} does not match expected {expected_dtype}",
                        context={
                            "space_dtype": str(observation_space.dtype),
                            "expected_dtype": str(expected_dtype),
                        },
                    )
                else:
                    warnings.warn(
                        f"Observation space dtype {observation_space.dtype} differs from expected {expected_dtype}",
                        UserWarning,
                        stacklevel=2,
                    )

        # Validate observation space sample() method produces valid observations in expected range
        try:
            for _ in range(10):  # Test multiple samples
                sample = observation_space.sample()

                if not isinstance(sample, np.ndarray):
                    raise ValidationError(
                        f"Observation space sample() returned non-array: {type(sample)}",
                        context={"sample_type": type(sample).__name__},
                    )

                if sample.shape != observation_space.shape:
                    raise ValidationError(
                        f"Sample shape {sample.shape} does not match space shape {observation_space.shape}",
                        context={
                            "sample_shape": sample.shape,
                            "space_shape": observation_space.shape,
                        },
                    )

                if check_bounds:
                    if np.any(sample < observation_space.low) or np.any(
                        sample > observation_space.high
                    ):
                        raise ValidationError(
                            "Sample values outside space bounds",
                            context={
                                "sample_range": (
                                    float(sample.min()),
                                    float(sample.max()),
                                ),
                                "space_bounds": (
                                    float(observation_space.low.min()),
                                    float(observation_space.high.max()),
                                ),
                            },
                        )
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Observation space sample() method failed: {e}",
                context={"sample_error": str(e)},
            )

        # Check observation space contains() method works correctly for valid observations
        try:
            # Test with valid observation
            valid_obs = np.full(
                observation_space.shape, 0.5, dtype=observation_space.dtype
            )
            if not observation_space.contains(valid_obs):
                raise ValidationError(
                    "Observation space contains() returned False for valid observation",
                    context={"test_observation": valid_obs.tolist()},
                )

            # Test with invalid observation (out of bounds)
            if check_bounds:
                invalid_obs = np.full(
                    observation_space.shape, 2.0, dtype=observation_space.dtype
                )  # Outside [0, 1]
                if observation_space.contains(invalid_obs):
                    raise ValidationError(
                        "Observation space contains() returned True for invalid observation",
                        context={"invalid_observation": invalid_obs.tolist()},
                    )
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Observation space contains() method failed: {e}",
                context={"contains_error": str(e)},
            )

        # Apply strict validation including mathematical property testing if strict_validation enabled
        if strict_validation:
            # Performance test for sample() method
            start_time = time.time()
            for _ in range(1000):
                observation_space.sample()
            sample_time = (time.time() - start_time) * 1000

            if sample_time > 50.0:  # 50ms threshold for 1000 samples
                warnings.warn(
                    f"Observation space sampling performance slower than expected: {sample_time:.3f}ms for 1000 samples",
                    UserWarning,
                    stacklevel=2,
                )

            # Test mathematical properties
            if observation_space.is_bounded() != "both":
                warnings.warn(
                    f"Observation space is not bounded on both sides: {observation_space.is_bounded()}",
                    UserWarning,
                    stacklevel=2,
                )

        # Test observation space compatibility with plume concentration values
        try:
            concentration_values = [0.0, 0.25, 0.5, 0.75, 1.0]
            for conc_val in concentration_values:
                test_obs = np.array([conc_val], dtype=observation_space.dtype)
                if not observation_space.contains(test_obs):
                    if strict_validation:
                        raise ValidationError(
                            f"Observation space incompatible with concentration value {conc_val}",
                            context={"concentration_value": conc_val},
                        )
                    else:
                        warnings.warn(
                            f"Observation space may not be compatible with concentration value {conc_val}",
                            UserWarning,
                            stacklevel=2,
                        )
        except ValidationError:
            raise
        except Exception as e:
            warnings.warn(
                f"Could not test concentration compatibility: {e}",
                UserWarning,
                stacklevel=2,
            )

        logger.debug(
            "Observation space validation passed",
            extra={
                "space_shape": observation_space.shape,
                "space_dtype": str(observation_space.dtype),
                "check_bounds": check_bounds,
                "check_dtype": check_dtype,
                "strict_validation": strict_validation,
            },
        )

        return True

    except ValidationError:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during observation space validation: {e}",
            extra={
                "space_type": (
                    type(observation_space).__name__ if observation_space else None
                ),
                "error_type": type(e).__name__,
            },
        )
        raise ValidationError(
            f"Observation space validation failed with unexpected error: {e}",
            context={"original_error": str(e)},
        )


def get_space_info(  # noqa: C901
    space: Union[gymnasium.spaces.Space],
    include_samples: bool = False,
    include_metadata: bool = False,
    include_performance_info: bool = False,
) -> Dict[str, Any]:
    """
    Returns comprehensive information about Gymnasium space configuration including properties,
    bounds, metadata, and compatibility details for debugging and analysis.

    Args:
        space: Gymnasium space to analyze
        include_samples: Whether to include sample values
        include_metadata: Whether to include metadata
        include_performance_info: Whether to include performance timing

    Returns:
        Dictionary containing complete space information with properties, bounds, samples, and metadata
    """
    try:
        info = {"space_type": type(space).__name__, "module": type(space).__module__}

        # Extract basic space information including type, shape, and bounds
        if isinstance(space, gymnasium.spaces.Discrete):
            info.update(
                {
                    "n": space.n,
                    "dtype": str(space.dtype) if hasattr(space, "dtype") else "int64",
                    "valid_actions": list(range(space.n)),
                }
            )

        elif isinstance(space, gymnasium.spaces.Box):
            info.update(
                {
                    "shape": space.shape,
                    "dtype": str(space.dtype),
                    "low": space.low.tolist(),
                    "high": space.high.tolist(),
                    "bounded_below": space.bounded_below.tolist(),
                    "bounded_above": space.bounded_above.tolist(),
                }
            )

        else:
            info["description"] = str(space)

        # Include space-specific properties (n for Discrete, low/high for Box)
        if hasattr(space, "seed"):
            info["supports_seeding"] = True

        # Generate sample values using space.sample() if include_samples is True
        if include_samples:
            try:
                samples = []
                for _ in range(5):
                    sample = space.sample()
                    if isinstance(sample, np.ndarray):
                        samples.append(sample.tolist())
                    else:
                        samples.append(sample)
                info["sample_values"] = samples
            except Exception as e:
                info["sample_error"] = str(e)

        # Include space metadata and documentation if include_metadata is True
        if include_metadata:
            if hasattr(space, "_metadata"):
                info["metadata"] = space._metadata
            else:
                info["metadata"] = None

            # Add additional metadata for known space types
            if (
                isinstance(space, gymnasium.spaces.Discrete)
                and space.n == ACTION_SPACE_SIZE
            ):
                info["plume_nav_compatible"] = True
                info["action_meanings"] = {
                    ACTION_UP: "UP",
                    ACTION_RIGHT: "RIGHT",
                    ACTION_DOWN: "DOWN",
                    ACTION_LEFT: "LEFT",
                }
            elif (
                isinstance(space, gymnasium.spaces.Box)
                and space.shape == DEFAULT_OBSERVATION_SHAPE
            ):
                info["plume_nav_compatible"] = True
                info["concentration_space"] = True

        # Add performance information including validation timing if include_performance_info is True
        if include_performance_info:
            perf_info = {}

            # Test sample() performance
            try:
                start_time = time.time()
                for _ in range(100):
                    space.sample()
                sample_time = (time.time() - start_time) * 1000
                perf_info["sample_100_ms"] = round(sample_time, 3)
            except Exception as e:
                perf_info["sample_error"] = str(e)

            # Test contains() performance
            if isinstance(space, gymnasium.spaces.Discrete):
                try:
                    start_time = time.time()
                    for i in range(100):
                        space.contains(i % space.n)
                    contains_time = (time.time() - start_time) * 1000
                    perf_info["contains_100_ms"] = round(contains_time, 3)
                except Exception as e:
                    perf_info["contains_error"] = str(e)

            elif isinstance(space, gymnasium.spaces.Box):
                try:
                    test_val = np.zeros(space.shape, dtype=space.dtype)
                    start_time = time.time()
                    for _ in range(100):
                        space.contains(test_val)
                    contains_time = (time.time() - start_time) * 1000
                    perf_info["contains_100_ms"] = round(contains_time, 3)
                except Exception as e:
                    perf_info["contains_error"] = str(e)

            info["performance"] = perf_info

        # Test space methods (contains, sample) and include results in analysis
        method_info = {}

        try:
            # Test if sample method works
            sample = space.sample()
            method_info["sample_works"] = True
            method_info["sample_type"] = type(sample).__name__

            # Test if contains method works with the sample
            contains_result = space.contains(sample)
            method_info["contains_works"] = True
            method_info["sample_contained"] = contains_result

        except Exception as e:
            method_info["method_error"] = str(e)

        info["methods"] = method_info

        return info

    except Exception as e:
        logger.error(f"Failed to get space info: {e}")
        return {"error": str(e), "space_type": type(space).__name__ if space else None}


def sample_valid_action(
    action_space: Optional[gymnasium.spaces.Discrete] = None,
    excluded_actions: Optional[List[int]] = None,
    validate_sample: bool = True,
) -> int:
    """
    Generates valid action sample from action space with validation and optional constraints
    for testing and random policy implementations.

    Args:
        action_space: Action space to sample from (creates default if None)
        excluded_actions: List of actions to exclude from sampling
        validate_sample: Whether to validate sampled action

    Returns:
        Valid action integer sampled from action space with constraints applied

    Raises:
        ValidationError: If sampling fails or produces invalid action
    """
    try:
        # Create default action space if action_space is None using create_action_space()
        if action_space is None:
            action_space = create_action_space()

        if not isinstance(action_space, gymnasium.spaces.Discrete):
            raise ValidationError(
                f"Expected Discrete action space, got {type(action_space)}",
                context={"space_type": type(action_space).__name__},
            )

        # Generate action sample using action_space.sample() method
        max_attempts = 100
        excluded_set = set(excluded_actions) if excluded_actions else set()

        for _attempt in range(max_attempts):
            action_sample = action_space.sample()

            # Check sampled action against excluded_actions list and resample if necessary
            if action_sample not in excluded_set:
                # Validate sampled action using validate_action() if validate_sample is True
                if validate_sample:
                    validated_action = validate_action(action_sample, action_space)
                    return validated_action
                else:
                    return int(action_sample)

        # If we couldn't find a valid action after max_attempts
        raise ValidationError(
            f"Could not sample valid action after {max_attempts} attempts",
            context={
                "excluded_actions": excluded_actions,
                "action_space_size": action_space.n,
                "available_actions": [
                    i for i in range(action_space.n) if i not in excluded_set
                ],
            },
        )

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Failed to sample valid action: {e}")
        raise ValidationError(
            f"Action sampling failed: {e}", context={"original_error": str(e)}
        )


def sample_valid_observation(
    observation_space: Optional[gymnasium.spaces.Box] = None,
    concentration_range: Optional[Tuple[float, float]] = None,
    validate_sample: bool = True,
) -> np.ndarray:
    """
    Generates valid observation sample from observation space with validation and optional
    concentration constraints for testing and simulation.

    Args:
        observation_space: Observation space to sample from (creates default if None)
        concentration_range: Optional range to constrain concentration values
        validate_sample: Whether to validate sampled observation

    Returns:
        Valid observation array sampled from observation space with concentration constraints

    Raises:
        ValidationError: If sampling fails or produces invalid observation
    """
    try:
        # Create default observation space if observation_space is None using create_observation_space()
        if observation_space is None:
            observation_space = create_observation_space()

        if not isinstance(observation_space, gymnasium.spaces.Box):
            raise ValidationError(
                f"Expected Box observation space, got {type(observation_space)}",
                context={"space_type": type(observation_space).__name__},
            )

        # Generate observation sample using observation_space.sample() method
        observation_sample = observation_space.sample()

        # Apply concentration_range constraints if provided by clipping sample values
        if concentration_range is not None:
            if (
                not isinstance(concentration_range, tuple)
                or len(concentration_range) != 2
            ):
                raise ValidationError(
                    f"Invalid concentration_range: {concentration_range}. Must be tuple of (low, high).",
                    context={"concentration_range": concentration_range},
                )

            range_low, range_high = concentration_range
            if range_low >= range_high:
                raise ValidationError(
                    f"Invalid concentration range: low ({range_low}) must be less than high ({range_high})",
                    context={"concentration_range": concentration_range},
                )

            # Clip values to specified range
            observation_sample = np.clip(observation_sample, range_low, range_high)

        # Validate sampled observation using validate_observation() if validate_sample is True
        if validate_sample:
            validated_observation = validate_observation(
                observation_sample,
                observation_space,
                check_range=concentration_range
                is None,  # Skip range check if we manually constrained
            )
            return validated_observation

        return observation_sample

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Failed to sample valid observation: {e}")
        raise ValidationError(
            f"Observation sampling failed: {e}", context={"original_error": str(e)}
        )


def check_space_compatibility(  # noqa: C901
    action_space: gymnasium.spaces.Discrete,
    observation_space: gymnasium.spaces.Box,
    strict_checking: bool = False,
    compatibility_requirements: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Checks compatibility between action and observation spaces ensuring proper configuration
    for reinforcement learning environment integration.

    Args:
        action_space: Discrete action space to check
        observation_space: Box observation space to check
        strict_checking: Enable strict compatibility rules
        compatibility_requirements: Custom compatibility requirements

    Returns:
        Compatibility analysis report with status, issues, and recommendations
    """
    try:
        compatibility_report = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "space_info": {
                "action_space": get_space_info(action_space),
                "observation_space": get_space_info(observation_space),
            },
        }

        # Validate action_space and observation_space are properly configured spaces
        try:
            validate_action_space(action_space, strict_validation=strict_checking)
        except ValidationError as e:
            compatibility_report["compatible"] = False
            compatibility_report["issues"].append(
                f"Action space validation failed: {e}"
            )

        try:
            validate_observation_space(
                observation_space, strict_validation=strict_checking
            )
        except ValidationError as e:
            compatibility_report["compatible"] = False
            compatibility_report["issues"].append(
                f"Observation space validation failed: {e}"
            )

        # Check action space size matches expected ACTION_SPACE_SIZE for navigation
        if action_space.n != ACTION_SPACE_SIZE:
            if strict_checking:
                compatibility_report["compatible"] = False
                compatibility_report["issues"].append(
                    f"Action space size {action_space.n} does not match expected {ACTION_SPACE_SIZE}"
                )
            else:
                compatibility_report["warnings"].append(
                    f"Action space size {action_space.n} differs from standard {ACTION_SPACE_SIZE}"
                )

        # Validate observation space shape and bounds are suitable for concentration observations
        if observation_space.shape != DEFAULT_OBSERVATION_SHAPE:
            if strict_checking:
                compatibility_report["compatible"] = False
                compatibility_report["issues"].append(
                    f"Observation space shape {observation_space.shape} does not match expected {DEFAULT_OBSERVATION_SHAPE}"
                )
            else:
                compatibility_report["warnings"].append(
                    f"Observation space shape differs from standard {DEFAULT_OBSERVATION_SHAPE}"
                )

        # Check dtype compatibility and memory usage estimates for performance
        if observation_space.dtype != OBSERVATION_DTYPE:
            compatibility_report["warnings"].append(
                f"Observation space dtype {observation_space.dtype} differs from recommended {OBSERVATION_DTYPE}"
            )

            # Estimate memory usage difference
            current_size = np.dtype(observation_space.dtype).itemsize
            recommended_size = np.dtype(OBSERVATION_DTYPE).itemsize
            memory_ratio = current_size / recommended_size

            if memory_ratio > 1.5:
                compatibility_report["recommendations"].append(
                    f"Consider using {OBSERVATION_DTYPE} dtype for better memory efficiency"
                )

        # Apply strict compatibility rules if strict_checking enabled
        if strict_checking:
            # Check bounds exactly match concentration range
            expected_low, expected_high = CONCENTRATION_RANGE
            if not (
                np.allclose(observation_space.low, expected_low)
                and np.allclose(observation_space.high, expected_high)
            ):
                compatibility_report["compatible"] = False
                compatibility_report["issues"].append(
                    f"Observation space bounds do not exactly match concentration range {CONCENTRATION_RANGE}"
                )

        # Test space interaction patterns and sampling performance
        try:
            # Test action sampling and validation
            test_action = action_space.sample()
            validate_action(test_action, action_space)

            # Test observation sampling and validation
            test_observation = observation_space.sample()
            validate_observation(test_observation, observation_space)

            compatibility_report["interaction_test"] = "passed"

        except Exception as e:
            compatibility_report["compatible"] = False
            compatibility_report["issues"].append(f"Space interaction test failed: {e}")

        # Check compatibility against custom requirements if compatibility_requirements provided
        if compatibility_requirements:
            for req_name, req_value in compatibility_requirements.items():
                if req_name == "max_action_space_size":
                    if action_space.n > req_value:
                        compatibility_report["compatible"] = False
                        compatibility_report["issues"].append(
                            f"Action space size {action_space.n} exceeds requirement {req_value}"
                        )

                elif req_name == "required_observation_dtype":
                    if str(observation_space.dtype) != str(req_value):
                        compatibility_report["compatible"] = False
                        compatibility_report["issues"].append(
                            f"Observation dtype {observation_space.dtype} does not match required {req_value}"
                        )

                elif req_name == "max_observation_dimensions":
                    if len(observation_space.shape) > req_value:
                        compatibility_report["compatible"] = False
                        compatibility_report["issues"].append(
                            f"Observation dimensions {len(observation_space.shape)} exceed requirement {req_value}"
                        )

        # Generate compatibility report with findings and optimization recommendations
        if compatibility_report["compatible"]:
            compatibility_report["recommendations"].append(
                "Spaces are compatible for RL environment use"
            )
        else:
            compatibility_report["recommendations"].append(
                "Fix identified issues before using spaces in environment"
            )

        if compatibility_report["warnings"]:
            compatibility_report["recommendations"].append(
                "Consider addressing warnings for optimal performance"
            )

        # Performance recommendations
        if observation_space.dtype != OBSERVATION_DTYPE:
            compatibility_report["recommendations"].append(
                f"Use {OBSERVATION_DTYPE} dtype for optimal performance"
            )

        if action_space.n != ACTION_SPACE_SIZE:
            compatibility_report["recommendations"].append(
                f"Standard navigation uses {ACTION_SPACE_SIZE} actions for cardinal directions"
            )

        logger.debug(
            "Space compatibility check completed",
            extra={
                "compatible": compatibility_report["compatible"],
                "issues_count": len(compatibility_report["issues"]),
                "warnings_count": len(compatibility_report["warnings"]),
                "strict_checking": strict_checking,
            },
        )

        return compatibility_report

    except Exception as e:
        logger.error(f"Space compatibility check failed: {e}")
        return {
            "compatible": False,
            "error": str(e),
            "issues": [f"Compatibility check failed with error: {e}"],
            "warnings": [],
            "recommendations": ["Fix compatibility check error before proceeding"],
        }


def optimize_space_operations(  # noqa: C901
    space_config: Dict,
    enable_caching: bool = True,
    profile_operations: bool = False,
    optimization_settings: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Optimizes space creation, validation, and sampling operations for improved performance
    with caching, profiling, and configuration tuning.

    Args:
        space_config: Configuration dictionary for spaces
        enable_caching: Whether to enable operation caching
        profile_operations: Whether to profile operations for bottlenecks
        optimization_settings: Custom optimization settings

    Returns:
        Optimization results with performance improvements and configuration recommendations
    """
    try:
        optimization_results = {
            "optimization_applied": [],
            "performance_improvements": {},
            "recommendations": [],
            "configuration_changes": {},
        }

        # Analyze current space operation performance with timing measurements
        if profile_operations:
            performance_baseline = {}

            # Profile action space creation
            start_time = time.time()
            for _ in range(10):
                create_action_space()
            action_creation_time = (time.time() - start_time) * 1000 / 10
            performance_baseline["action_creation_ms"] = action_creation_time

            # Profile observation space creation
            start_time = time.time()
            for _ in range(10):
                create_observation_space()
            obs_creation_time = (time.time() - start_time) * 1000 / 10
            performance_baseline["observation_creation_ms"] = obs_creation_time

            optimization_results["performance_baseline"] = performance_baseline

        # Enable caching for space creation and validation if enable_caching is True
        if enable_caching:
            # Caching is already enabled via @functools.lru_cache decorators
            optimization_results["optimization_applied"].append("LRU caching enabled")
            optimization_results["configuration_changes"]["cache_size"] = {
                "creation_cache": SPACE_CREATION_CACHE_SIZE,
                "validation_cache": SPACE_VALIDATION_CACHE_SIZE,
            }

        # Profile space operations and identify performance bottlenecks if profile_operations enabled
        if profile_operations:
            bottlenecks = []

            # Check if space creation is slow
            if "action_creation_ms" in optimization_results.get(
                "performance_baseline", {}
            ):
                if (
                    optimization_results["performance_baseline"]["action_creation_ms"]
                    > 10.0
                ):
                    bottlenecks.append("Action space creation is slow (>10ms)")

            if "observation_creation_ms" in optimization_results.get(
                "performance_baseline", {}
            ):
                if (
                    optimization_results["performance_baseline"][
                        "observation_creation_ms"
                    ]
                    > 10.0
                ):
                    bottlenecks.append("Observation space creation is slow (>10ms)")

            optimization_results["bottlenecks"] = bottlenecks

        # Apply optimization settings from optimization_settings if provided
        if optimization_settings:
            for setting_name, setting_value in optimization_settings.items():
                if setting_name == "disable_validation":
                    if setting_value:
                        optimization_results["optimization_applied"].append(
                            "Validation disabled for performance"
                        )
                        optimization_results["recommendations"].append(
                            "Re-enable validation in production environments"
                        )

                elif setting_name == "fast_dtype_conversion":
                    if setting_value:
                        optimization_results["optimization_applied"].append(
                            "Fast dtype conversion enabled"
                        )

                elif setting_name == "memory_optimization":
                    if setting_value:
                        optimization_results["optimization_applied"].append(
                            "Memory optimization enabled"
                        )
                        optimization_results["recommendations"].append(
                            "Monitor memory usage with optimization"
                        )

        # Tune cache sizes and validation parameters for optimal performance
        cache_recommendations = []

        if enable_caching:
            # Recommend cache size adjustments based on usage patterns
            cache_recommendations.append(
                f"Current creation cache size: {SPACE_CREATION_CACHE_SIZE} (consider increasing for heavy usage)"
            )
            cache_recommendations.append(
                f"Current validation cache size: {SPACE_VALIDATION_CACHE_SIZE} (adjust based on validation frequency)"
            )

        optimization_results["cache_recommendations"] = cache_recommendations

        # Test optimized configuration and measure performance improvements
        if profile_operations and enable_caching:
            # Measure performance after optimization
            optimized_performance = {}

            # Test cached space creation (should be faster on subsequent calls)
            start_time = time.time()
            for _ in range(100):
                create_action_space()  # Should hit cache after first call
            cached_action_time = (time.time() - start_time) * 1000 / 100
            optimized_performance["cached_action_creation_ms"] = cached_action_time

            start_time = time.time()
            for _ in range(100):
                create_observation_space()  # Should hit cache after first call
            cached_obs_time = (time.time() - start_time) * 1000 / 100
            optimized_performance["cached_observation_creation_ms"] = cached_obs_time

            # Calculate improvement ratios
            if "performance_baseline" in optimization_results:
                baseline = optimization_results["performance_baseline"]
                if (
                    "action_creation_ms" in baseline
                    and baseline["action_creation_ms"] > 0
                ):
                    action_improvement = (
                        baseline["action_creation_ms"] / cached_action_time
                    )
                    optimized_performance["action_speed_improvement"] = (
                        f"{action_improvement:.2f}x"
                    )

                if (
                    "observation_creation_ms" in baseline
                    and baseline["observation_creation_ms"] > 0
                ):
                    obs_improvement = (
                        baseline["observation_creation_ms"] / cached_obs_time
                    )
                    optimized_performance["observation_speed_improvement"] = (
                        f"{obs_improvement:.2f}x"
                    )

            optimization_results["optimized_performance"] = optimized_performance

        # Generate optimization report with before/after performance metrics
        general_recommendations = [
            "Use caching for repeated space creation operations",
            "Profile space operations in production to identify bottlenecks",
            "Consider disabling validation in performance-critical paths (with caution)",
            "Monitor memory usage when using large cache sizes",
        ]

        if space_config.get("frequent_creation"):
            general_recommendations.append("Pre-create and reuse spaces when possible")

        if space_config.get("memory_constrained"):
            general_recommendations.append(
                "Reduce cache sizes in memory-constrained environments"
            )

        optimization_results["general_recommendations"] = general_recommendations

        logger.info(
            "Space operations optimization completed",
            extra={
                "optimizations_applied": len(
                    optimization_results["optimization_applied"]
                ),
                "caching_enabled": enable_caching,
                "profiling_enabled": profile_operations,
            },
        )

        return optimization_results

    except Exception as e:
        logger.error(f"Space optimization failed: {e}")
        return {
            "error": str(e),
            "optimization_applied": [],
            "recommendations": ["Fix optimization error before proceeding"],
            "performance_improvements": {},
        }


@dataclass
class SpaceConfig:
    """
    Configuration data class for Gymnasium space creation with validation, metadata, and
    optimization parameters for action and observation space customization.
    """

    action_space_size: int = ACTION_SPACE_SIZE
    observation_shape: Tuple = DEFAULT_OBSERVATION_SHAPE
    concentration_bounds: Tuple[float, float] = CONCENTRATION_RANGE
    observation_dtype: np.dtype = OBSERVATION_DTYPE
    enable_validation: bool = True
    enable_caching: bool = True
    metadata: Dict[str, Any] = None
    optimization_settings: Dict[str, Any] = None

    def __post_init__(self):
        """
        Initialize space configuration with action space size, observation parameters, and
        validation settings.
        """
        # Store action_space_size with validation against ACTION_SPACE_SIZE constant
        if not isinstance(self.action_space_size, int) or self.action_space_size <= 0:
            raise ValidationError(
                f"Invalid action_space_size: {self.action_space_size}. Must be positive integer.",
                context={"received": self.action_space_size, "expected_type": "int"},
            )

        # Store observation_shape with dimension validation and compatibility checking
        if (
            not isinstance(self.observation_shape, tuple)
            or len(self.observation_shape) == 0
        ):
            raise ValidationError(
                f"Invalid observation_shape: {self.observation_shape}. Must be non-empty tuple.",
                context={"received": self.observation_shape, "expected_type": "tuple"},
            )

        if any(
            not isinstance(dimension, int) or dimension <= 0
            for dimension in self.observation_shape
        ):
            raise ValidationError(
                f"Observation shape dimensions must be positive integers: {self.observation_shape}",
                context={"received": self.observation_shape},
            )

        # Store concentration_bounds with mathematical consistency validation
        if (
            not isinstance(self.concentration_bounds, tuple)
            or len(self.concentration_bounds) != 2
            or self.concentration_bounds[0] >= self.concentration_bounds[1]
        ):
            raise ValidationError(
                f"Invalid concentration_bounds: {self.concentration_bounds}. Must be (low, high) with low < high.",
                context={"received": self.concentration_bounds},
            )

        # Store observation_dtype with performance and compatibility considerations
        if self.observation_dtype is None:
            self.observation_dtype = OBSERVATION_DTYPE
        else:
            try:
                np.dtype(self.observation_dtype)
            except TypeError as exc:
                raise ValidationError(
                    f"Invalid observation_dtype: {self.observation_dtype}",
                    context={"received": self.observation_dtype, "error": str(exc)},
                )

        # Initialize empty metadata dictionary for space documentation
        if self.metadata is None:
            self.metadata = {}

        # Initialize default optimization_settings for performance tuning
        if self.optimization_settings is None:
            self.optimization_settings = {
                "enable_fast_validation": True,
                "cache_validation_results": True,
                "memory_efficient_creation": False,
            }

    def validate_config(self, strict_validation: bool = False) -> bool:  # noqa: C901
        """
        Validates space configuration parameters ensuring mathematical consistency, performance
        feasibility, and Gymnasium compatibility.

        Args:
            strict_validation: Enable strict validation rules

        Returns:
            True if configuration is valid, raises ValidationError if invalid
        """
        try:
            # Validate action_space_size is positive integer matching expected cardinal directions
            if self.action_space_size != ACTION_SPACE_SIZE and strict_validation:
                raise ValidationError(
                    f"Strict mode: action_space_size {self.action_space_size} must equal {ACTION_SPACE_SIZE}",
                    context={"action_space_size": self.action_space_size},
                )

            # Check observation_shape has proper dimensions for concentration observations
            if not all(
                isinstance(dim, int) and dim > 0 for dim in self.observation_shape
            ):
                raise ValidationError(
                    f"All observation shape dimensions must be positive integers: {self.observation_shape}",
                    context={"observation_shape": self.observation_shape},
                )

            # Validate concentration_bounds are mathematically consistent [min, max] with min < max
            low_bound, high_bound = self.concentration_bounds
            if not isinstance(low_bound, (int, float)) or not isinstance(
                high_bound, (int, float)
            ):
                raise ValidationError(
                    f"Concentration bounds must be numeric: {self.concentration_bounds}",
                    context={"bounds": self.concentration_bounds},
                )

            # Check observation_dtype is supported NumPy dtype for performance optimization
            try:
                np.dtype(self.observation_dtype)
            except TypeError as e:
                raise ValidationError(
                    f"Invalid observation_dtype: {self.observation_dtype}",
                    context={"dtype": self.observation_dtype, "error": str(e)},
                )

            # Apply strict validation rules including memory usage estimation if strict_validation enabled
            if strict_validation:
                # Estimate memory usage
                element_size = np.dtype(self.observation_dtype).itemsize
                total_elements = np.prod(self.observation_shape)
                memory_per_observation = element_size * total_elements

                if memory_per_observation > 1024:  # 1KB per observation
                    warnings.warn(
                        f"Large memory usage per observation: {memory_per_observation} bytes",
                        UserWarning,
                        stacklevel=2,
                    )

            return True

        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(
                f"Configuration validation failed: {e}", context={"error": str(e)}
            )

    def create_action_space(self) -> gymnasium.spaces.Discrete:
        """
        Creates Gymnasium Discrete action space using configuration parameters with validation and optimization.

        Returns:
            Configured Discrete action space based on configuration settings
        """
        return create_action_space(
            num_actions=self.action_space_size,
            validate_actions=self.enable_validation,
            space_config={
                "metadata": self.metadata,
                "optimization_settings": self.optimization_settings,
            },
        )

    def create_observation_space(self) -> gymnasium.spaces.Box:
        """
        Creates Gymnasium Box observation space using configuration parameters with bounds and dtype validation.

        Returns:
            Configured Box observation space based on configuration settings
        """
        return create_observation_space(
            observation_shape=self.observation_shape,
            concentration_bounds=self.concentration_bounds,
            observation_dtype=self.observation_dtype,
            validate_bounds=self.enable_validation,
            space_config={
                "metadata": self.metadata,
                "optimization_settings": self.optimization_settings,
            },
        )

    def estimate_performance(
        self, include_memory_usage: bool = True, include_timing_estimates: bool = True
    ) -> Dict[str, Any]:
        """
        Estimates performance characteristics for space operations based on configuration parameters.

        Args:
            include_memory_usage: Whether to include memory usage analysis
            include_timing_estimates: Whether to include timing estimates

        Returns:
            Performance estimation report with memory usage, timing, and optimization recommendations
        """
        performance_report = {}

        # Estimate memory usage for observation space based on shape and dtype
        if include_memory_usage:
            element_size = np.dtype(self.observation_dtype).itemsize
            total_elements = np.prod(self.observation_shape)
            memory_per_observation = element_size * total_elements

            performance_report["memory_usage"] = {
                "bytes_per_observation": memory_per_observation,
                "kb_per_observation": memory_per_observation / 1024,
                "element_size_bytes": element_size,
                "total_elements": total_elements,
            }

        # Calculate expected validation timing based on configuration complexity
        if include_timing_estimates:
            # Base timing estimates (rough approximations)
            base_action_creation = 0.1  # ms
            base_observation_creation = 0.5  # ms

            # Adjust for validation
            if self.enable_validation:
                base_action_creation *= 1.5
                base_observation_creation *= 2.0

            # Adjust for shape complexity
            shape_complexity = np.prod(self.observation_shape)
            if shape_complexity > 1:
                base_observation_creation *= 1 + np.log10(shape_complexity)

            performance_report["timing_estimates"] = {
                "action_space_creation_ms": base_action_creation,
                "observation_space_creation_ms": base_observation_creation,
                "total_space_creation_ms": base_action_creation
                + base_observation_creation,
            }

        # Generate performance optimization recommendations based on configuration
        recommendations = []

        if self.enable_validation:
            recommendations.append(
                "Validation enabled - consider disabling in performance-critical paths"
            )

        if self.enable_caching:
            recommendations.append(
                "Caching enabled - will improve performance for repeated operations"
            )

        if include_memory_usage and "memory_usage" in performance_report:
            memory_mb = performance_report["memory_usage"]["kb_per_observation"] / 1024
            if memory_mb > 1.0:
                recommendations.append(
                    f"High memory usage ({memory_mb:.2f} MB/obs) - consider smaller dtype"
                )

        performance_report["recommendations"] = recommendations

        return performance_report

    def to_dict(
        self, include_metadata: bool = True, include_optimization_settings: bool = True
    ) -> Dict[str, Any]:
        """
        Converts space configuration to dictionary for serialization, logging, and external analysis.

        Args:
            include_metadata: Whether to include metadata information
            include_optimization_settings: Whether to include optimization settings

        Returns:
            Dictionary representation of space configuration with optional metadata and settings
        """
        config_dict = {
            "action_space_size": self.action_space_size,
            "observation_shape": self.observation_shape,
            "concentration_bounds": self.concentration_bounds,
            "observation_dtype": str(self.observation_dtype),
            "enable_validation": self.enable_validation,
            "enable_caching": self.enable_caching,
        }

        # Include metadata information if include_metadata is True
        if include_metadata and self.metadata:
            config_dict["metadata"] = self.metadata

        # Include optimization_settings if include_optimization_settings is True
        if include_optimization_settings and self.optimization_settings:
            config_dict["optimization_settings"] = self.optimization_settings

        return config_dict


class SpaceValidator:
    """
    Comprehensive validation utility class for Gymnasium spaces with caching, performance
    monitoring, and detailed error reporting for development and production use.
    """

    def __init__(
        self,
        enable_caching: bool = True,
        strict_mode: bool = False,
        validation_config: Optional[Dict] = None,
    ):
        """
        Initialize space validator with caching, strict mode, and custom validation configuration.

        Args:
            enable_caching: Enable validation result caching for performance
            strict_mode: Enable strict validation rules
            validation_config: Custom validation configuration
        """
        # Store enable_caching flag for performance optimization of repeated validations
        self.enable_caching = enable_caching

        # Set strict_mode for enhanced validation rigor and comprehensive checking
        self.strict_mode = strict_mode

        # Initialize validation_config with default settings merged with custom config
        default_config = {
            "timeout_ms": SPACE_VALIDATION_TIMEOUT_MS,
            "max_samples_test": 10,
            "performance_thresholds": {"sample_ms": 1.0, "contains_ms": 0.1},
        }

        if validation_config:
            default_config.update(validation_config)
        self.validation_config = default_config

        # Create validation_cache dictionary for caching validation results
        self.validation_cache = {} if enable_caching else None

        # Initialize component logger using get_component_logger for validation operations
        self.logger = get_component_logger(f"{__name__}.SpaceValidator")

        # Initialize validation_stats dictionary for operation tracking and analysis
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Initialize performance_metrics dictionary for timing and performance monitoring
        self.performance_metrics = {
            "total_validation_time_ms": 0.0,
            "average_validation_time_ms": 0.0,
            "max_validation_time_ms": 0.0,
            "min_validation_time_ms": float("inf"),
        }

    def validate_space(  # noqa: C901
        self,
        space: gymnasium.spaces.Space,
        space_type: str,
        validation_requirements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generic space validation method with caching, comprehensive checking, and detailed error reporting.

        Args:
            space: Gymnasium space to validate
            space_type: Type of space ('action' or 'observation')
            validation_requirements: Validation requirements dictionary

        Returns:
            Validation result with status, errors, warnings, and recommendations
        """
        start_time = time.time()

        try:
            # Generate cache key from space properties and validation requirements
            cache_key = None
            if self.enable_caching:
                cache_key = f"{space_type}:{type(space).__name__}:{str(space)}:{str(validation_requirements)}"

                # Check validation cache if enable_caching is True and return cached result
                if cache_key in self.validation_cache:
                    cached_result = self.validation_cache[cache_key]
                    self.validation_stats["cache_hits"] += 1
                    self.validation_stats["total_validations"] += 1

                    if cached_result.get("valid", False):
                        self.validation_stats["successful_validations"] += 1
                    else:
                        self.validation_stats["failed_validations"] += 1

                    elapsed_cached = (time.time() - start_time) * 1000
                    self._update_performance_metrics(elapsed_cached)

                    cached_copy = dict(cached_result)
                    cached_copy["validation_time_ms"] = elapsed_cached
                    return cached_copy

                self.validation_stats["cache_misses"] += 1

            # Initialize validation result structure
            validation_result = {
                "valid": True,
                "errors": [],
                "warnings": [],
                "recommendations": [],
                "space_info": get_space_info(space, include_samples=True),
                "validation_time_ms": 0.0,
            }

            # Perform basic space validation including type checking and property validation
            if not hasattr(space, "sample") or not callable(space.sample):
                validation_result["valid"] = False
                validation_result["errors"].append(
                    "Space missing required sample() method"
                )

            if not hasattr(space, "contains") or not callable(space.contains):
                validation_result["valid"] = False
                validation_result["errors"].append(
                    "Space missing required contains() method"
                )

            # Apply space-type specific validation based on space_type parameter
            if space_type == "action":
                if isinstance(space, gymnasium.spaces.Discrete):
                    try:
                        validate_action_space(space, strict_validation=self.strict_mode)
                    except ValidationError as e:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"Action space validation failed: {e}"
                        )
                else:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Expected Discrete action space, got {type(space)}"
                    )

            elif space_type == "observation":
                if isinstance(space, gymnasium.spaces.Box):
                    try:
                        validate_observation_space(
                            space, strict_validation=self.strict_mode
                        )
                    except ValidationError as e:
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"Observation space validation failed: {e}"
                        )
                else:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Expected Box observation space, got {type(space)}"
                    )

            # Check space against validation_requirements if provided
            for req_name, req_value in validation_requirements.items():
                if req_name == "required_methods":
                    for method_name in req_value:
                        if not hasattr(space, method_name):
                            validation_result["warnings"].append(
                                f"Space missing recommended method: {method_name}"
                            )

                elif req_name == "performance_requirements":
                    # Test performance requirements
                    self._test_space_performance(space, req_value, validation_result)

            # Apply strict validation rules including performance testing if strict_mode enabled
            if self.strict_mode:
                # Additional strict validation
                if hasattr(space, "seed"):
                    try:
                        space.seed(42)
                        validation_result["recommendations"].append(
                            "Space seeding tested successfully"
                        )
                    except Exception as e:
                        validation_result["warnings"].append(
                            f"Space seeding failed: {e}"
                        )

            # Calculate validation timing
            elapsed_time = (time.time() - start_time) * 1000
            validation_result["validation_time_ms"] = elapsed_time

            # Generate comprehensive validation result with status, messages, and context
            if validation_result["valid"]:
                validation_result["recommendations"].append(
                    "Space passed all validation checks"
                )
            else:
                validation_result["recommendations"].append(
                    "Fix validation errors before using space"
                )

            # Cache validation result if enable_caching is True for performance optimization
            if self.enable_caching and cache_key:
                self.validation_cache[cache_key] = validation_result

            # Update validation statistics and performance metrics
            self.validation_stats["total_validations"] += 1
            if validation_result["valid"]:
                self.validation_stats["successful_validations"] += 1
            else:
                self.validation_stats["failed_validations"] += 1

            self._update_performance_metrics(elapsed_time)

            return validation_result

        except Exception as e:
            self.logger.error(f"Space validation failed with exception: {e}")
            return {
                "valid": False,
                "errors": [f"Validation exception: {e}"],
                "warnings": [],
                "recommendations": ["Fix validation exception"],
                "validation_time_ms": (time.time() - start_time) * 1000,
            }

    def validate_action_space(  # noqa: C901
        self,
        action_space: gymnasium.spaces.Discrete,
        check_navigation_compatibility: bool = True,
    ) -> Dict[str, Any]:
        """
        Specialized validation method for Discrete action spaces with cardinal direction checking
        and navigation compatibility.

        Args:
            action_space: Discrete action space to validate
            check_navigation_compatibility: Whether to check navigation compatibility

        Returns:
            Action space validation result with navigation-specific analysis
        """
        validation_requirements = {
            "space_size": ACTION_SPACE_SIZE,
            "navigation_compatible": check_navigation_compatibility,
        }

        result = self.validate_space(action_space, "action", validation_requirements)

        # Additional action space specific validation
        if check_navigation_compatibility and result["valid"]:
            if action_space.n == ACTION_SPACE_SIZE:
                result["recommendations"].append(
                    "Action space compatible with cardinal direction navigation"
                )
            else:
                result["warnings"].append(
                    f"Action space size {action_space.n} may not be optimal for navigation"
                )

        return result

    def validate_observation_space(  # noqa: C901
        self,
        observation_space: gymnasium.spaces.Box,
        check_concentration_compatibility: bool = True,
    ) -> Dict[str, Any]:
        """
        Specialized validation method for Box observation spaces with concentration bounds and dtype checking.

        Args:
            observation_space: Box observation space to validate
            check_concentration_compatibility: Whether to check concentration compatibility

        Returns:
            Observation space validation result with concentration-specific analysis
        """
        validation_requirements = {
            "expected_shape": DEFAULT_OBSERVATION_SHAPE,
            "expected_bounds": CONCENTRATION_RANGE,
            "expected_dtype": OBSERVATION_DTYPE,
            "concentration_compatible": check_concentration_compatibility,
        }

        result = self.validate_space(
            observation_space, "observation", validation_requirements
        )

        # Additional observation space specific validation
        if check_concentration_compatibility and result["valid"]:
            bounds_match = np.allclose(
                observation_space.low, CONCENTRATION_RANGE[0]
            ) and np.allclose(observation_space.high, CONCENTRATION_RANGE[1])

            if bounds_match:
                result["recommendations"].append(
                    "Observation space compatible with concentration values"
                )
            else:
                result["warnings"].append(
                    "Observation space bounds may not match concentration range"
                )

        return result

    def clear_cache(self) -> Dict[str, Any]:
        """
        Clears validation cache and resets performance statistics for memory management and fresh validation state.

        Returns:
            Cache clearing report with statistics and memory freed
        """
        cache_report = {
            "cache_entries_cleared": 0,
            "memory_freed_estimate_mb": 0.0,
            "stats_reset": False,
        }

        if self.validation_cache:
            # Count current cache entries and calculate memory usage before clearing
            cache_report["cache_entries_cleared"] = len(self.validation_cache)

            # Rough estimate of memory usage (this is approximate)
            cache_report["memory_freed_estimate_mb"] = (
                cache_report["cache_entries_cleared"] * 0.001
            )  # 1KB per entry estimate

            # Clear validation_cache dictionary and reset cache statistics
            self.validation_cache.clear()

            # Reset cache statistics
            self.validation_stats["cache_hits"] = 0
            self.validation_stats["cache_misses"] = 0
            cache_report["stats_reset"] = True

            self.logger.info(
                f"Cleared validation cache: {cache_report['cache_entries_cleared']} entries"
            )

        return cache_report

    def get_validation_stats(
        self,
        include_cache_stats: bool = True,
        include_performance_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns comprehensive validation statistics including cache performance, error rates, and timing analysis.

        Args:
            include_cache_stats: Whether to include cache statistics
            include_performance_analysis: Whether to include performance metrics

        Returns:
            Validation statistics with performance metrics and analysis
        """
        stats = {
            "validation_counts": {
                "total": self.validation_stats["total_validations"],
                "successful": self.validation_stats["successful_validations"],
                "failed": self.validation_stats["failed_validations"],
            }
        }

        # Calculate success rate
        if self.validation_stats["total_validations"] > 0:
            success_rate = (
                self.validation_stats["successful_validations"]
                / self.validation_stats["total_validations"]
            )
            stats["success_rate"] = round(success_rate * 100, 2)
        else:
            stats["success_rate"] = 0.0

        # Calculate cache hit/miss ratios if include_cache_stats is True
        if include_cache_stats and self.enable_caching:
            total_cache_requests = (
                self.validation_stats["cache_hits"]
                + self.validation_stats["cache_misses"]
            )

            cache_stats = {
                "cache_hits": self.validation_stats["cache_hits"],
                "cache_misses": self.validation_stats["cache_misses"],
                "total_requests": total_cache_requests,
            }

            if total_cache_requests > 0:
                cache_stats["hit_rate"] = round(
                    self.validation_stats["cache_hits"] / total_cache_requests * 100, 2
                )
            else:
                cache_stats["hit_rate"] = 0.0

            stats["cache_performance"] = cache_stats

        # Include performance analysis with timing metrics if include_performance_analysis is True
        if include_performance_analysis:
            perf_stats = dict(self.performance_metrics)

            # Calculate additional metrics
            if self.validation_stats["total_validations"] > 0:
                perf_stats["average_validation_time_ms"] = round(
                    perf_stats["total_validation_time_ms"]
                    / self.validation_stats["total_validations"],
                    3,
                )

            stats["performance_metrics"] = perf_stats

        return stats

    def _test_space_performance(
        self,
        space: gymnasium.spaces.Space,
        performance_requirements: Dict,
        validation_result: Dict,
    ):
        """
        Tests space performance against requirements.

        Args:
            space: Space to test
            performance_requirements: Performance requirement thresholds
            validation_result: Result dictionary to update
        """
        try:
            # Test sampling performance
            if "max_sample_time_ms" in performance_requirements:
                start_time = time.time()
                for _ in range(100):
                    space.sample()
                sample_time = (time.time() - start_time) * 1000

                if sample_time > performance_requirements["max_sample_time_ms"]:
                    validation_result["warnings"].append(
                        f"Space sampling slower than required: {sample_time:.2f}ms > {performance_requirements['max_sample_time_ms']}ms"
                    )

            # Test contains performance
            if "max_contains_time_ms" in performance_requirements:
                test_sample = space.sample()
                start_time = time.time()
                for _ in range(100):
                    space.contains(test_sample)
                contains_time = (time.time() - start_time) * 1000

                if contains_time > performance_requirements["max_contains_time_ms"]:
                    validation_result["warnings"].append(
                        f"Space contains() slower than required: {contains_time:.2f}ms > {performance_requirements['max_contains_time_ms']}ms"
                    )

        except Exception as e:
            validation_result["warnings"].append(f"Performance testing failed: {e}")

    def _update_performance_metrics(self, validation_time_ms: float):
        """
        Updates performance metrics with latest validation timing.

        Args:
            validation_time_ms: Validation time in milliseconds
        """
        self.performance_metrics["total_validation_time_ms"] += validation_time_ms
        self.performance_metrics["max_validation_time_ms"] = max(
            self.performance_metrics["max_validation_time_ms"], validation_time_ms
        )
        self.performance_metrics["min_validation_time_ms"] = min(
            self.performance_metrics["min_validation_time_ms"], validation_time_ms
        )
