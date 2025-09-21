"""
Core constants module for plume_nav_sim package.

This module provides centralized definition of system-wide constants including environment
defaults, mathematical parameters, performance targets, action mappings, rendering
specifications, data types, validation limits, and configuration values used throughout
the reinforcement learning environment implementation.

All constants are organized into logical groups with comprehensive factory functions and
consistency validation to prevent magic number anti-patterns and ensure parameter
relationships remain mathematically sound.
"""

from typing import (  # >=3.10 - Type hints for factory functions and validation
    Any,
    Dict,
    Tuple,
)

# External imports with version comments
import numpy as np  # >=2.1.0 - Data type constants and mathematical precision specifications

# Component Names for Logging and Identification
COMPONENT_NAMES = [
    "environment",
    "plume_model",
    "agent_state",
    "episode_manager",
    "state_manager",
    "action_processor",
    "reward_calculator",
    "boundary_enforcer",
    "renderer",
    "validation",
    "seeding",
    "logging",
    "exceptions",
]

# Logging Defaults
LOG_LEVEL_DEFAULT = "INFO"

# Package identification constants
PACKAGE_NAME = "plume_nav_sim"
PACKAGE_VERSION = "0.0.1"
ENVIRONMENT_ID = "PlumeNav-StaticGaussian-v0"

# Environment default configuration constants
DEFAULT_GRID_SIZE = (128, 128)
MIN_GRID_SIZE = (4, 4)
MAX_GRID_SIZE = (512, 512)
DEFAULT_SOURCE_LOCATION = (64, 64)
DEFAULT_PLUME_SIGMA = 12.0
DEFAULT_GOAL_RADIUS = 0
DEFAULT_MAX_STEPS = 1000
MIN_GRID_SIZE = (16, 16)
MAX_GRID_SIZE = (1024, 1024)

# Discrete action space constants for cardinal direction navigation
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_SPACE_SIZE = 4

# Movement vector mappings for position calculations
MOVEMENT_VECTORS = {
    0: (0, 1),  # UP: positive y-direction
    1: (1, 0),  # RIGHT: positive x-direction
    2: (0, -1),  # DOWN: negative y-direction
    3: (-1, 0),  # LEFT: negative x-direction
}

# Reward system constants for goal-based termination
REWARD_GOAL_REACHED = 1.0
REWARD_DEFAULT = 0.0

# Mathematical precision constants for plume model calculations
CONCENTRATION_RANGE = (0.0, 1.0)
GAUSSIAN_PRECISION = 1e-6
DISTANCE_PRECISION = 1e-6

# NumPy data type specifications for consistent array operations
FIELD_DTYPE = np.float32
OBSERVATION_DTYPE = np.float32
RGB_DTYPE = np.uint8
COORDINATE_DTYPE = np.int32

# Rendering constants for dual-mode visualization
AGENT_MARKER_COLOR = [255, 0, 0]  # Red RGB values for agent visualization
SOURCE_MARKER_COLOR = [255, 255, 255]  # White RGB values for source visualization
AGENT_MARKER_SIZE = (3, 3)  # Agent marker dimensions in pixels
SOURCE_MARKER_SIZE = (5, 5)  # Source marker dimensions in pixels
PIXEL_VALUE_MIN = 0
PIXEL_VALUE_MAX = 255
SUPPORTED_RENDER_MODES = ["rgb_array", "human"]
MATPLOTLIB_DEFAULT_FIGSIZE = (8, 8)
BACKEND_PRIORITY_LIST = ["TkAgg", "Qt5Agg", "Agg"]

COMPONENT_NAMES = ("core", "env", "plume", "render")
LOG_LEVEL_DEFAULT = "INFO"

# Performance target constants for system optimization
PERFORMANCE_TRACKING_ENABLED = True
PERFORMANCE_TARGET_STEP_LATENCY_MS = 1.0
PERFORMANCE_TARGET_RGB_RENDER_MS = 5.0
PERFORMANCE_TARGET_HUMAN_RENDER_MS = 33.0
PERFORMANCE_TARGET_EPISODE_RESET_MS = 10.0
PERFORMANCE_TARGET_PLUME_GENERATION_MS = 10.0
BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS = 0.1
PERFORMANCE_TRACKING_ENABLED = True

# Memory management constants for resource constraints
MEMORY_LIMIT_TOTAL_MB = 50
MEMORY_LIMIT_PLUME_FIELD_MB = 40
BOUNDARY_VALIDATION_CACHE_SIZE = 500

# Compact bundle of values that tests reference for configuration defaults.
TESTING_CONSTANTS = {
    "default_seed": 42,
    "small_grid_size": (32, 32),
    "performance_targets": {
        "step_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
        "render_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
    },
}

# Plume model validation constants
MIN_PLUME_SIGMA = 0.1
MAX_PLUME_SIGMA = 100.0
STATIC_GAUSSIAN_MODEL_TYPE = "static_gaussian"

# Seeding and Reproducibility Constants
SEED_MIN_VALUE = 0
SEED_MAX_VALUE = 2**31 - 1
VALID_SEED_TYPES = (int, np.integer)

# Comprehensive error message templates for validation
VALIDATION_ERROR_MESSAGES = {
    "invalid_action": "Action must be integer in range [0, 3] for cardinal directions",
    "invalid_coordinates": "Coordinates must be non-negative integers within grid bounds",
    "invalid_grid_size": "Grid size must be tuple of positive integers with reasonable dimensions",
    "invalid_sigma": f"Sigma must be positive float within range [{MIN_PLUME_SIGMA}, {MAX_PLUME_SIGMA}]",
    "invalid_seed": "Seed must be integer or None for random number generator",
    "memory_limit_exceeded": f"Configuration exceeds memory limit of {MEMORY_LIMIT_TOTAL_MB} MB",
    "performance_target_missed": "Operation exceeded performance target threshold",
    "invalid_render_mode": f"Render mode must be one of: {SUPPORTED_RENDER_MODES}",
}


def get_default_environment_constants() -> Dict[str, Any]:
    """
    Returns dictionary of default environment configuration constants.

    Includes grid dimensions, source location, episode parameters, and Gymnasium API
    settings for standard environment setup and registration.

    Returns:
        Dict[str, Any]: Dictionary containing environment constants with keys:
                       grid_size, source_location, goal_radius, max_steps, action_space_size
    """
    return {
        "grid_size": DEFAULT_GRID_SIZE,
        "source_location": DEFAULT_SOURCE_LOCATION,
        "goal_radius": DEFAULT_GOAL_RADIUS,
        "max_steps": DEFAULT_MAX_STEPS,
        "action_space_size": ACTION_SPACE_SIZE,
        "package_name": PACKAGE_NAME,
        "environment_id": ENVIRONMENT_ID,
        "package_version": PACKAGE_VERSION,
    }


def get_plume_model_constants() -> Dict[str, Any]:
    """
    Returns dictionary of plume model mathematical constants.

    Includes Gaussian parameters, concentration ranges, precision values, and validation
    limits for the static Gaussian plume model implementation.

    Returns:
        Dict[str, Any]: Dictionary containing plume model constants with mathematical
                       parameters and validation limits
    """
    return {
        "default_sigma": DEFAULT_PLUME_SIGMA,
        "sigma_range": (MIN_PLUME_SIGMA, MAX_PLUME_SIGMA),
        "concentration_range": CONCENTRATION_RANGE,
        "gaussian_precision": GAUSSIAN_PRECISION,
        "model_type": STATIC_GAUSSIAN_MODEL_TYPE,
        "field_dtype": FIELD_DTYPE,
        "mathematical_formula": "C(x,y) = exp(-((x-sx)² + (y-sy)²) / (2*σ²))",
        "normalization_range": "[0.0, 1.0] with peak = 1.0 at source location",
    }


def get_action_space_constants() -> Dict[str, Any]:
    """
    Returns dictionary of action space constants.

    Includes discrete action mappings, movement vectors, and space configuration
    parameters for cardinal direction navigation.

    Returns:
        Dict[str, Any]: Dictionary containing action space constants with action
                       mappings and movement vectors
    """
    return {
        "space_size": ACTION_SPACE_SIZE,
        "action_up": ACTION_UP,
        "action_right": ACTION_RIGHT,
        "action_down": ACTION_DOWN,
        "action_left": ACTION_LEFT,
        "movement_vectors": MOVEMENT_VECTORS,
        "action_names": {
            ACTION_UP: "UP",
            ACTION_RIGHT: "RIGHT",
            ACTION_DOWN: "DOWN",
            ACTION_LEFT: "LEFT",
        },
        "coordinate_deltas": MOVEMENT_VECTORS,
        "discrete_space_type": "Discrete(4)",
    }


def get_rendering_constants() -> Dict[str, Any]:
    """
    Returns dictionary of rendering constants.

    Includes supported modes, color specifications, marker configurations, and
    performance targets for dual-mode visualization system.

    Returns:
        Dict[str, Any]: Dictionary containing rendering constants with color schemes,
                       marker specifications, and backend preferences
    """
    return {
        "supported_modes": SUPPORTED_RENDER_MODES,
        "agent_marker_color": AGENT_MARKER_COLOR,
        "source_marker_color": SOURCE_MARKER_COLOR,
        "agent_marker_size": AGENT_MARKER_SIZE,
        "source_marker_size": SOURCE_MARKER_SIZE,
        "pixel_value_min": PIXEL_VALUE_MIN,
        "pixel_value_max": PIXEL_VALUE_MAX,
        "rgb_dtype": RGB_DTYPE,
        "rgb_render_target_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
        "backend_preferences": ["TkAgg", "Qt5Agg", "Agg"],
        "colormap_human_mode": "gray",
        "figure_size_inches": (8, 8),
        "coordinate_origin": "lower",
    }


def get_performance_constants() -> Dict[str, Any]:
    """
    Returns dictionary of performance constants.

    Includes timing targets, memory limits, and optimization parameters for system
    performance monitoring and validation.

    Returns:
        Dict[str, Any]: Dictionary containing performance constants with timing targets
                       and memory limits
    """
    return {
        "step_latency_target_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
        "rgb_render_target_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
        "plume_generation_target_ms": PERFORMANCE_TARGET_PLUME_GENERATION_MS,
        "memory_limit_total_mb": MEMORY_LIMIT_TOTAL_MB,
        "memory_limit_plume_field_mb": MEMORY_LIMIT_PLUME_FIELD_MB,
        "optimization_thresholds": {
            "step_warning_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS * 2,
            "render_warning_ms": PERFORMANCE_TARGET_RGB_RENDER_MS * 3,
            "memory_warning_mb": MEMORY_LIMIT_TOTAL_MB * 0.8,
        },
        "benchmark_iterations": {
            "step_timing": 1000,
            "render_timing": 100,
            "memory_profiling": 10,
        },
    }


def get_validation_constants() -> Dict[str, Any]:
    """
    Returns dictionary of validation constants.

    Includes parameter limits, precision values, and error message templates for
    comprehensive input validation and bounds checking.

    Returns:
        Dict[str, Any]: Dictionary containing validation constants with limits,
                       precision values, and error messages
    """
    return {
        "distance_precision": DISTANCE_PRECISION,
        "gaussian_precision": GAUSSIAN_PRECISION,
        "sigma_range": (MIN_PLUME_SIGMA, MAX_PLUME_SIGMA),
        "grid_size_limits": {
            "min_dimension": 16,
            "max_dimension": 1024,
            "recommended_max": 512,
        },
        "coordinate_bounds_checking": True,
        "error_messages": VALIDATION_ERROR_MESSAGES,
        "seed_validation": {"min_seed": 0, "max_seed": 2**32 - 1, "allow_none": True},
        "memory_validation_mb": MEMORY_LIMIT_TOTAL_MB,
        "action_validation_range": (0, ACTION_SPACE_SIZE - 1),
    }


def get_testing_constants() -> Dict[str, Any]:
    """
    Returns dictionary of testing constants.

    Includes test grid sizes, tolerance values, and test-specific parameters optimized
    for unit testing and validation performance.

    Returns:
        Dict[str, Any]: Dictionary containing testing constants optimized for test
                       execution performance and validation
    """
    return {
        "test_grid_size": (32, 32),  # Smaller grid for faster test execution
        "test_source_location": (16, 16),  # Center of test grid
        "test_max_steps": 100,  # Reduced steps for faster testing
        "test_tolerance": {
            "float_comparison": 1e-6,
            "coordinate_tolerance": 0,  # Exact integer matching required
            "timing_variance_percent": 20,  # 20% timing variance acceptable
        },
        "reproducibility_seeds": [42, 123, 456, 789, 999],
        "performance_test_iterations": {
            "step_timing": 100,  # Reduced for faster testing
            "render_timing": 10,
            "memory_check": 5,
        },
        "test_colors": {
            "agent_test_color": [128, 0, 0],  # Dimmed red for testing
            "source_test_color": [200, 200, 200],  # Light gray for testing
        },
        "expected_test_results": {
            "initial_concentration_at_source": 1.0,
            "zero_concentration_at_corners": 0.0,
            "default_reward_non_goal": 0.0,
            "goal_reward_at_source": 1.0,
        },
    }


TESTING_CONSTANTS = get_testing_constants()


def validate_constant_consistency(
    strict_mode: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates mathematical and logical consistency between related constants.

    Performs comprehensive cross-parameter checking with detailed error reporting
    to ensure all constant relationships remain mathematically sound and logically
    consistent.

    Args:
        strict_mode: Enable additional precision and bounds checking

    Returns:
        Tuple[bool, Dict[str, Any]]: Tuple of (is_valid, validation_report) with
                                    detailed consistency analysis
    """
    validation_report = {
        "overall_status": True,
        "mathematical_consistency": {},
        "logical_consistency": {},
        "performance_feasibility": {},
        "memory_relationships": {},
        "warnings": [],
        "errors": [],
    }

    # Mathematical consistency validation
    try:
        # Validate source location within grid bounds
        if (
            DEFAULT_SOURCE_LOCATION[0] >= DEFAULT_GRID_SIZE[0]
            or DEFAULT_SOURCE_LOCATION[1] >= DEFAULT_GRID_SIZE[1]
        ):
            validation_report["errors"].append(
                f"Source location {DEFAULT_SOURCE_LOCATION} outside grid {DEFAULT_GRID_SIZE}"
            )
            validation_report["overall_status"] = False

        # Validate sigma within acceptable range
        if not (MIN_PLUME_SIGMA <= DEFAULT_PLUME_SIGMA <= MAX_PLUME_SIGMA):
            validation_report["errors"].append(
                f"Default sigma {DEFAULT_PLUME_SIGMA} outside valid range [{MIN_PLUME_SIGMA}, {MAX_PLUME_SIGMA}]"
            )
            validation_report["overall_status"] = False

        validation_report["mathematical_consistency"]["source_bounds_check"] = True
        validation_report["mathematical_consistency"]["sigma_range_check"] = True

    except Exception as e:
        validation_report["errors"].append(
            f"Mathematical consistency check failed: {e}"
        )
        validation_report["overall_status"] = False

    # Action space consistency validation
    try:
        # Verify action constants match movement vector keys
        expected_actions = set(range(ACTION_SPACE_SIZE))
        actual_actions = set(MOVEMENT_VECTORS.keys())
        if expected_actions != actual_actions:
            validation_report["errors"].append(
                f"Action constants {expected_actions} do not match movement vector keys {actual_actions}"
            )
            validation_report["overall_status"] = False

        # Validate movement vectors are unit vectors in cardinal directions
        expected_vectors = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
        if MOVEMENT_VECTORS != expected_vectors:
            validation_report["errors"].append(
                f"Movement vectors {MOVEMENT_VECTORS} do not match expected cardinal directions"
            )
            validation_report["overall_status"] = False

        validation_report["logical_consistency"]["action_space_mapping"] = True

    except Exception as e:
        validation_report["errors"].append(
            f"Action space consistency check failed: {e}"
        )
        validation_report["overall_status"] = False

    # Performance target feasibility
    try:
        # Check if performance targets are achievable
        if PERFORMANCE_TARGET_STEP_LATENCY_MS <= 0:
            validation_report["errors"].append("Step latency target must be positive")
            validation_report["overall_status"] = False

        # Warn about very tight performance targets
        if PERFORMANCE_TARGET_STEP_LATENCY_MS < 0.1:
            validation_report["warnings"].append(
                f"Very tight step latency target: {PERFORMANCE_TARGET_STEP_LATENCY_MS}ms"
            )

        validation_report["performance_feasibility"]["targets_positive"] = True

    except Exception as e:
        validation_report["errors"].append(f"Performance feasibility check failed: {e}")
        validation_report["overall_status"] = False

    # Memory limit relationships
    try:
        # Validate that component limits don't exceed total
        if MEMORY_LIMIT_PLUME_FIELD_MB > MEMORY_LIMIT_TOTAL_MB:
            validation_report["errors"].append(
                f"Plume field limit {MEMORY_LIMIT_PLUME_FIELD_MB}MB exceeds total limit {MEMORY_LIMIT_TOTAL_MB}MB"
            )
            validation_report["overall_status"] = False

        # Calculate expected memory usage for default grid
        grid_elements = DEFAULT_GRID_SIZE[0] * DEFAULT_GRID_SIZE[1]
        expected_mb = (grid_elements * 4) / (1024 * 1024)  # float32 = 4 bytes

        if expected_mb > MEMORY_LIMIT_PLUME_FIELD_MB:
            validation_report["warnings"].append(
                f"Default grid size may exceed field memory limit: {expected_mb:.1f}MB > {MEMORY_LIMIT_PLUME_FIELD_MB}MB"
            )

        validation_report["memory_relationships"]["limits_consistent"] = True

    except Exception as e:
        validation_report["errors"].append(f"Memory relationships check failed: {e}")
        validation_report["overall_status"] = False

    # Rendering consistency validation
    try:
        # Validate color values are in valid RGB range
        for color_name, color_values in [
            ("agent", AGENT_MARKER_COLOR),
            ("source", SOURCE_MARKER_COLOR),
        ]:
            if not all(0 <= val <= 255 for val in color_values):
                validation_report["errors"].append(
                    f"{color_name} color values {color_values} outside RGB range [0, 255]"
                )
                validation_report["overall_status"] = False

        # Validate marker sizes are positive
        for size_name, size_tuple in [
            ("agent", AGENT_MARKER_SIZE),
            ("source", SOURCE_MARKER_SIZE),
        ]:
            if not all(val > 0 for val in size_tuple):
                validation_report["errors"].append(
                    f"{size_name} marker size {size_tuple} contains non-positive values"
                )
                validation_report["overall_status"] = False

        validation_report["logical_consistency"]["rendering_parameters"] = True

    except Exception as e:
        validation_report["errors"].append(f"Rendering consistency check failed: {e}")
        validation_report["overall_status"] = False

    # Strict mode additional validations
    if strict_mode:
        try:
            # Additional precision checking in strict mode
            if GAUSSIAN_PRECISION >= 1e-3:
                validation_report["warnings"].append(
                    f"Gaussian precision {GAUSSIAN_PRECISION} may be too loose for strict calculations"
                )

            if DISTANCE_PRECISION != GAUSSIAN_PRECISION:
                validation_report["warnings"].append(
                    "Distance and Gaussian precision differ - may cause numerical inconsistencies"
                )

            # Strict bounds checking
            if DEFAULT_GRID_SIZE[0] != DEFAULT_GRID_SIZE[1]:
                validation_report["warnings"].append(
                    "Non-square grid may affect plume symmetry"
                )

            validation_report["mathematical_consistency"]["strict_mode_checks"] = True

        except Exception as e:
            validation_report["warnings"].append(f"Strict mode validation warning: {e}")

    # Generate summary recommendations
    if validation_report["warnings"]:
        validation_report["recommendations"] = [
            "Review warning messages for potential improvements",
            "Consider adjusting parameters to resolve warnings",
            "Monitor performance impact of tight constraints",
        ]

    if not validation_report["overall_status"]:
        validation_report["recommendations"] = [
            "Fix all error conditions before using constants",
            "Verify mathematical relationships are correct",
            "Check parameter bounds and ranges",
        ]

    return validation_report["overall_status"], validation_report


# Export list for comprehensive public interface
__all__ = [
    # Component Names
    "COMPONENT_NAMES",
    # Logging
    "LOG_LEVEL_DEFAULT",
    # Package identification
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "ENVIRONMENT_ID",
    # Environment defaults
    "DEFAULT_GRID_SIZE",
    "DEFAULT_SOURCE_LOCATION",
    "DEFAULT_PLUME_SIGMA",
    "DEFAULT_GOAL_RADIUS",
    "DEFAULT_MAX_STEPS",
    "MIN_GRID_SIZE",
    "MAX_GRID_SIZE",
    # Action space constants
    "ACTION_UP",
    "ACTION_RIGHT",
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_SPACE_SIZE",
    "MOVEMENT_VECTORS",
    # Reward system
    "REWARD_GOAL_REACHED",
    "REWARD_DEFAULT",
    # Mathematical parameters
    "CONCENTRATION_RANGE",
    "GAUSSIAN_PRECISION",
    "DISTANCE_PRECISION",
    # Data types
    "FIELD_DTYPE",
    "OBSERVATION_DTYPE",
    "RGB_DTYPE",
    "COORDINATE_DTYPE",
    # Rendering parameters
    "AGENT_MARKER_COLOR",
    "SOURCE_MARKER_COLOR",
    "AGENT_MARKER_SIZE",
    "SOURCE_MARKER_SIZE",
    "PIXEL_VALUE_MIN",
    "PIXEL_VALUE_MAX",
    "SUPPORTED_RENDER_MODES",
    "MATPLOTLIB_DEFAULT_FIGSIZE",
    "BACKEND_PRIORITY_LIST",
    # Performance targets
    "PERFORMANCE_TARGET_STEP_LATENCY_MS",
    "PERFORMANCE_TARGET_RGB_RENDER_MS",
    "PERFORMANCE_TARGET_HUMAN_RENDER_MS",
    "PERFORMANCE_TARGET_EPISODE_RESET_MS",
    "PERFORMANCE_TARGET_PLUME_GENERATION_MS",
    # Memory constraints
    "MEMORY_LIMIT_TOTAL_MB",
    "MEMORY_LIMIT_PLUME_FIELD_MB",
    "TESTING_CONSTANTS",
    # Validation parameters
    "MIN_PLUME_SIGMA",
    "MAX_PLUME_SIGMA",
    "STATIC_GAUSSIAN_MODEL_TYPE",
    "VALIDATION_ERROR_MESSAGES",
    # Factory functions
    "get_default_environment_constants",
    "get_plume_model_constants",
    "get_action_space_constants",
    "get_rendering_constants",
    "get_performance_constants",
    "get_validation_constants",
    "get_testing_constants",
    "TESTING_CONSTANTS",
    # Validation function
    "validate_constant_consistency",
]
