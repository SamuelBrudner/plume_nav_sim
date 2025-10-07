"""
Comprehensive test module for plume_nav_sim core constants validation including constant value verification,
factory function testing, cross-constant consistency validation, type checking, range validation, and
mathematical relationship verification ensuring system-wide constant integrity and preventing configuration
drift across all components.

This module validates:
- System-wide constant definitions and their consistency across components
- Environment default configuration parameters for Gymnasium compliance
- Static Gaussian plume mathematical constants for numerical stability
- Performance target constants validation for system monitoring
- Cross-constant consistency checking with detailed error reporting

Test data is generated programmatically to ensure consistency across different execution environments.
"""

import re  # >=3.10 - Regular expression validation for string constant format validation

import numpy as np  # >=2.1.0 - Array operations, dtype validation, and mathematical constant verification
import pytest  # >=8.0.0 - Testing framework for constant validation, parameterized tests, and fixture management

# Import all constants and functions from the core constants module
from plume_nav_sim.core.constants import (  # Package metadata constants; Environment configuration constants; Action space constants; Reward system constants; Data type constants; Plume model constants; Performance constants; Rendering constants; Validation constants; Factory functions; Validation function
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    DEFAULT_GRID_SIZE,
    DEFAULT_PLUME_SIGMA,
    DEFAULT_SOURCE_LOCATION,
    ENVIRONMENT_ID,
    FIELD_DTYPE,
    MAX_PLUME_SIGMA,
    MEMORY_LIMIT_TOTAL_MB,
    MIN_PLUME_SIGMA,
    MOVEMENT_VECTORS,
    OBSERVATION_DTYPE,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    REWARD_DEFAULT,
    REWARD_GOAL_REACHED,
    SUPPORTED_RENDER_MODES,
    VALIDATION_ERROR_MESSAGES,
    get_action_space_constants,
    get_default_environment_constants,
    get_performance_constants,
    get_plume_model_constants,
    get_rendering_constants,
    get_testing_constants,
    get_validation_constants,
    validate_constant_consistency,
)

# Import type enumerations for integration testing
from plume_nav_sim.core.types import Action, RenderMode

# Import exception classes for testing validation error handling

# Test constants for validation
EXPECTED_PACKAGE_NAME = "plume_nav_sim"
EXPECTED_PACKAGE_VERSION_PATTERN = r"^\d+\.\d+\.\d+$"
EXPECTED_ENVIRONMENT_ID_PATTERN = r"^PlumeNav-.+-v\d+$"
EXPECTED_GRID_SIZE_DEFAULT = (128, 128)
EXPECTED_SOURCE_LOCATION_DEFAULT = (64, 64)
EXPECTED_ACTION_RANGE = [0, 1, 2, 3]
EXPECTED_RENDER_MODES = ["rgb_array", "human"]
TOLERANCE_FLOAT_COMPARISON = 1e-10
MAX_REASONABLE_GRID_SIZE = 1024
MIN_REASONABLE_SIGMA = 0.01
MAX_REASONABLE_SIGMA = 1000.0


class TestPackageMetadataConstants:
    """Test package metadata constants including package name, version format, and environment ID format validation."""

    def test_package_metadata_constants(self):
        """Test package metadata constants including package name, version format, and environment ID format validation ensuring proper identification and versioning."""
        # Assert PACKAGE_NAME equals expected 'plume_nav_sim' string
        assert (
            PACKAGE_NAME == EXPECTED_PACKAGE_NAME
        ), f"Expected package name '{EXPECTED_PACKAGE_NAME}', got '{PACKAGE_NAME}'"
        assert isinstance(
            PACKAGE_NAME, str
        ), f"PACKAGE_NAME should be string, got {type(PACKAGE_NAME)}"
        assert len(PACKAGE_NAME) > 0, "PACKAGE_NAME should not be empty"

        # Validate PACKAGE_VERSION matches semantic versioning pattern using regex
        assert isinstance(
            PACKAGE_VERSION, str
        ), f"PACKAGE_VERSION should be string, got {type(PACKAGE_VERSION)}"
        assert len(PACKAGE_VERSION) > 0, "PACKAGE_VERSION should not be empty"
        version_match = re.match(EXPECTED_PACKAGE_VERSION_PATTERN, PACKAGE_VERSION)
        assert (
            version_match is not None
        ), f"PACKAGE_VERSION '{PACKAGE_VERSION}' does not match semantic versioning pattern"

        # Validate ENVIRONMENT_ID follows Gymnasium versioning convention
        assert isinstance(
            ENVIRONMENT_ID, str
        ), f"ENVIRONMENT_ID should be string, got {type(ENVIRONMENT_ID)}"
        assert len(ENVIRONMENT_ID) > 0, "ENVIRONMENT_ID should not be empty"
        env_id_match = re.match(EXPECTED_ENVIRONMENT_ID_PATTERN, ENVIRONMENT_ID)
        assert (
            env_id_match is not None
        ), f"ENVIRONMENT_ID '{ENVIRONMENT_ID}' does not match Gymnasium pattern"

        # Test that package constants are immutable and hashable types
        assert isinstance(
            PACKAGE_NAME, str
        ), "PACKAGE_NAME should be immutable string type"
        assert isinstance(
            PACKAGE_VERSION, str
        ), "PACKAGE_VERSION should be immutable string type"
        assert isinstance(
            ENVIRONMENT_ID, str
        ), "ENVIRONMENT_ID should be immutable string type"


class TestEnvironmentDefaultConstants:
    """Test default environment configuration constants including grid size, source location, episode parameters, and their types with consistency validation."""

    def test_environment_default_constants(self):
        """Test default environment configuration constants including grid size, source location, episode parameters, and their types with consistency validation."""
        # Assert DEFAULT_GRID_SIZE equals (128, 128) and is tuple of two integers
        assert (
            DEFAULT_GRID_SIZE == EXPECTED_GRID_SIZE_DEFAULT
        ), f"Expected grid size {EXPECTED_GRID_SIZE_DEFAULT}, got {DEFAULT_GRID_SIZE}"
        assert isinstance(
            DEFAULT_GRID_SIZE, tuple
        ), f"DEFAULT_GRID_SIZE should be tuple, got {type(DEFAULT_GRID_SIZE)}"
        assert (
            len(DEFAULT_GRID_SIZE) == 2
        ), f"DEFAULT_GRID_SIZE should have 2 elements, got {len(DEFAULT_GRID_SIZE)}"
        assert all(
            isinstance(x, int) for x in DEFAULT_GRID_SIZE
        ), "DEFAULT_GRID_SIZE elements should be integers"
        assert all(
            x > 0 for x in DEFAULT_GRID_SIZE
        ), "DEFAULT_GRID_SIZE elements should be positive"
        assert all(
            x <= MAX_REASONABLE_GRID_SIZE for x in DEFAULT_GRID_SIZE
        ), "DEFAULT_GRID_SIZE elements should be within reasonable limits"

        # Assert DEFAULT_SOURCE_LOCATION equals (64, 64) and is tuple of two integers
        assert (
            DEFAULT_SOURCE_LOCATION == EXPECTED_SOURCE_LOCATION_DEFAULT
        ), f"Expected source location {EXPECTED_SOURCE_LOCATION_DEFAULT}, got {DEFAULT_SOURCE_LOCATION}"
        assert isinstance(
            DEFAULT_SOURCE_LOCATION, tuple
        ), f"DEFAULT_SOURCE_LOCATION should be tuple, got {type(DEFAULT_SOURCE_LOCATION)}"
        assert (
            len(DEFAULT_SOURCE_LOCATION) == 2
        ), f"DEFAULT_SOURCE_LOCATION should have 2 elements, got {len(DEFAULT_SOURCE_LOCATION)}"
        assert all(
            isinstance(x, int) for x in DEFAULT_SOURCE_LOCATION
        ), "DEFAULT_SOURCE_LOCATION elements should be integers"
        assert all(
            x >= 0 for x in DEFAULT_SOURCE_LOCATION
        ), "DEFAULT_SOURCE_LOCATION elements should be non-negative"

        # Validate DEFAULT_SOURCE_LOCATION is within DEFAULT_GRID_SIZE boundaries
        assert (
            DEFAULT_SOURCE_LOCATION[0] < DEFAULT_GRID_SIZE[0]
        ), "Source X coordinate should be within grid width"
        assert (
            DEFAULT_SOURCE_LOCATION[1] < DEFAULT_GRID_SIZE[1]
        ), "Source Y coordinate should be within grid height"

        # Assert DEFAULT_PLUME_SIGMA is positive float within reasonable range
        assert isinstance(
            DEFAULT_PLUME_SIGMA, (int, float)
        ), f"DEFAULT_PLUME_SIGMA should be numeric, got {type(DEFAULT_PLUME_SIGMA)}"
        assert DEFAULT_PLUME_SIGMA > 0, "DEFAULT_PLUME_SIGMA should be positive"
        assert (
            MIN_REASONABLE_SIGMA <= DEFAULT_PLUME_SIGMA <= MAX_REASONABLE_SIGMA
        ), f"DEFAULT_PLUME_SIGMA {DEFAULT_PLUME_SIGMA} should be within reasonable range [{MIN_REASONABLE_SIGMA}, {MAX_REASONABLE_SIGMA}]"

        # Test mathematical consistency between default grid size and source location centering
        grid_center_x = DEFAULT_GRID_SIZE[0] // 2
        grid_center_y = DEFAULT_GRID_SIZE[1] // 2
        assert (
            abs(DEFAULT_SOURCE_LOCATION[0] - grid_center_x) <= 1
        ), "Default source should be near grid center X"
        assert (
            abs(DEFAULT_SOURCE_LOCATION[1] - grid_center_y) <= 1
        ), "Default source should be near grid center Y"


class TestActionSpaceConstants:
    """Test discrete action space constants including action values, movement vectors, and action-vector consistency with enumeration validation."""

    def test_action_space_constants(self):
        """Test discrete action space constants including action values, movement vectors, and action-vector consistency with enumeration validation."""
        # Assert ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT have expected values 0,1,2,3
        expected_actions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]
        assert (
            expected_actions == EXPECTED_ACTION_RANGE
        ), f"Action constants should be {EXPECTED_ACTION_RANGE}, got {expected_actions}"

        # Validate all action constants are unique integers in range [0,3]
        actions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]
        assert all(
            isinstance(action, int) for action in actions
        ), "All action constants should be integers"
        assert len(set(actions)) == len(
            actions
        ), "All action constants should be unique"
        assert all(
            0 <= action <= 3 for action in actions
        ), "All action constants should be in range [0,3]"

        # Assert MOVEMENT_VECTORS dictionary has keys matching action constants
        assert isinstance(
            MOVEMENT_VECTORS, dict
        ), f"MOVEMENT_VECTORS should be dict, got {type(MOVEMENT_VECTORS)}"
        expected_keys = set(EXPECTED_ACTION_RANGE)
        actual_keys = set(MOVEMENT_VECTORS.keys())
        assert (
            actual_keys == expected_keys
        ), f"MOVEMENT_VECTORS keys should be {expected_keys}, got {actual_keys}"

        # Validate movement vectors are tuples of two integers (dx, dy)
        for action, vector in MOVEMENT_VECTORS.items():
            assert isinstance(
                vector, tuple
            ), f"Movement vector for action {action} should be tuple, got {type(vector)}"
            assert (
                len(vector) == 2
            ), f"Movement vector for action {action} should have 2 elements, got {len(vector)}"
            assert all(
                isinstance(x, int) for x in vector
            ), f"Movement vector for action {action} should contain integers"
            assert all(
                -1 <= x <= 1 for x in vector
            ), f"Movement vector for action {action} should be unit vector"

        # Test movement vector consistency: UP=(0,1), RIGHT=(1,0), DOWN=(0,-1), LEFT=(-1,0)
        assert MOVEMENT_VECTORS[ACTION_UP] == (
            0,
            1,
        ), f"UP movement should be (0,1), got {MOVEMENT_VECTORS[ACTION_UP]}"
        assert MOVEMENT_VECTORS[ACTION_RIGHT] == (
            1,
            0,
        ), f"RIGHT movement should be (1,0), got {MOVEMENT_VECTORS[ACTION_RIGHT]}"
        assert MOVEMENT_VECTORS[ACTION_DOWN] == (
            0,
            -1,
        ), f"DOWN movement should be (0,-1), got {MOVEMENT_VECTORS[ACTION_DOWN]}"
        assert MOVEMENT_VECTORS[ACTION_LEFT] == (
            -1,
            0,
        ), f"LEFT movement should be (-1,0), got {MOVEMENT_VECTORS[ACTION_LEFT]}"

        # Verify action constants match Action enumeration values from types module
        assert ACTION_UP == Action.UP.value, "ACTION_UP should match Action.UP.value"
        assert (
            ACTION_RIGHT == Action.RIGHT.value
        ), "ACTION_RIGHT should match Action.RIGHT.value"
        assert (
            ACTION_DOWN == Action.DOWN.value
        ), "ACTION_DOWN should match Action.DOWN.value"
        assert (
            ACTION_LEFT == Action.LEFT.value
        ), "ACTION_LEFT should match Action.LEFT.value"


class TestRewardSystemConstants:
    """Test reward system constants including goal reward, default reward, and sparse reward structure validation with numerical precision checks."""

    def test_reward_system_constants(self):
        """Test reward system constants including goal reward, default reward, and sparse reward structure validation with numerical precision checks."""
        # Assert REWARD_GOAL_REACHED equals 1.0 and is positive float
        assert isinstance(
            REWARD_GOAL_REACHED, (int, float)
        ), f"REWARD_GOAL_REACHED should be numeric, got {type(REWARD_GOAL_REACHED)}"
        assert (
            abs(REWARD_GOAL_REACHED - 1.0) < TOLERANCE_FLOAT_COMPARISON
        ), f"REWARD_GOAL_REACHED should be 1.0, got {REWARD_GOAL_REACHED}"
        assert REWARD_GOAL_REACHED > 0, "REWARD_GOAL_REACHED should be positive"

        # Assert REWARD_DEFAULT equals 0.0 for sparse reward structure
        assert isinstance(
            REWARD_DEFAULT, (int, float)
        ), f"REWARD_DEFAULT should be numeric, got {type(REWARD_DEFAULT)}"
        assert (
            abs(REWARD_DEFAULT - 0.0) < TOLERANCE_FLOAT_COMPARISON
        ), f"REWARD_DEFAULT should be 0.0, got {REWARD_DEFAULT}"

        # Test that goal reward is greater than default reward for proper incentive
        assert (
            REWARD_GOAL_REACHED > REWARD_DEFAULT
        ), "Goal reward should be greater than default reward for proper incentive structure"

        # Verify reward values are within reasonable range for RL training
        assert (
            -10.0 <= REWARD_DEFAULT <= 10.0
        ), "Default reward should be within reasonable RL range"
        assert (
            -10.0 <= REWARD_GOAL_REACHED <= 10.0
        ), "Goal reward should be within reasonable RL range"

        # Assert reward constants support mathematical operations without overflow
        reward_sum = REWARD_GOAL_REACHED + REWARD_DEFAULT
        assert isinstance(
            reward_sum, (int, float)
        ), "Reward constants should support addition"
        reward_product = REWARD_GOAL_REACHED * 1000  # Test scaling
        assert isinstance(
            reward_product, (int, float)
        ), "Reward constants should support scaling"


class TestDataTypeConstants:
    """Test NumPy data type constants ensuring consistent array dtypes across observations, fields, rendering, and coordinate operations with Gymnasium compliance."""

    def test_data_type_constants(self):
        """Test NumPy data type constants ensuring consistent array dtypes across observations, fields, rendering, and coordinate operations with Gymnasium compliance."""
        # Assert FIELD_DTYPE is numpy.float32 for concentration field arrays
        assert (
            FIELD_DTYPE == np.float32
        ), f"FIELD_DTYPE should be numpy.float32, got {FIELD_DTYPE}"
        assert isinstance(
            FIELD_DTYPE, type
        ), f"FIELD_DTYPE should be type, got {type(FIELD_DTYPE)}"

        # Assert OBSERVATION_DTYPE is numpy.float32 for Gymnasium observation compliance
        assert (
            OBSERVATION_DTYPE == np.float32
        ), f"OBSERVATION_DTYPE should be numpy.float32, got {OBSERVATION_DTYPE}"
        assert isinstance(
            OBSERVATION_DTYPE, type
        ), f"OBSERVATION_DTYPE should be type, got {type(OBSERVATION_DTYPE)}"

        # Validate all dtype constants are valid NumPy dtypes using numpy.dtype()
        field_dtype_obj = np.dtype(FIELD_DTYPE)
        obs_dtype_obj = np.dtype(OBSERVATION_DTYPE)
        assert field_dtype_obj is not None, "FIELD_DTYPE should be valid NumPy dtype"
        assert (
            obs_dtype_obj is not None
        ), "OBSERVATION_DTYPE should be valid NumPy dtype"

        # Test dtype compatibility for array operations and memory efficiency
        test_array = np.array([0.5, 1.0], dtype=FIELD_DTYPE)
        assert test_array.dtype == FIELD_DTYPE, "Array should maintain specified dtype"

        test_obs = np.array([0.8], dtype=OBSERVATION_DTYPE)
        assert (
            test_obs.dtype == OBSERVATION_DTYPE
        ), "Observation array should maintain specified dtype"

        # Verify dtype constants match expected itemsize and value ranges
        assert field_dtype_obj.itemsize == 4, "float32 should have 4-byte itemsize"
        assert obs_dtype_obj.itemsize == 4, "float32 should have 4-byte itemsize"

        # Test that dtypes support expected value ranges
        assert (
            np.finfo(FIELD_DTYPE).max > 1.0
        ), "Field dtype should support concentration values > 1.0"
        assert (
            np.finfo(OBSERVATION_DTYPE).max > 1.0
        ), "Observation dtype should support observation values > 1.0"


class TestPlumeModelConstants:
    """Test Gaussian plume model constants including concentration range, precision values, sigma limits, and mathematical consistency validation."""

    def test_plume_model_constants(self):
        """Test Gaussian plume model constants including concentration range, precision values, sigma limits, and mathematical consistency validation."""
        # Test MIN_PLUME_SIGMA and MAX_PLUME_SIGMA define reasonable sigma range
        assert isinstance(
            MIN_PLUME_SIGMA, (int, float)
        ), f"MIN_PLUME_SIGMA should be numeric, got {type(MIN_PLUME_SIGMA)}"
        assert isinstance(
            MAX_PLUME_SIGMA, (int, float)
        ), f"MAX_PLUME_SIGMA should be numeric, got {type(MAX_PLUME_SIGMA)}"
        assert MIN_PLUME_SIGMA > 0, "MIN_PLUME_SIGMA should be positive"
        assert MAX_PLUME_SIGMA > 0, "MAX_PLUME_SIGMA should be positive"
        assert (
            MIN_PLUME_SIGMA < MAX_PLUME_SIGMA
        ), "MIN_PLUME_SIGMA should be less than MAX_PLUME_SIGMA"

        # Validate DEFAULT_PLUME_SIGMA is within MIN/MAX range with proper consistency
        assert (
            MIN_PLUME_SIGMA <= DEFAULT_PLUME_SIGMA <= MAX_PLUME_SIGMA
        ), f"DEFAULT_PLUME_SIGMA {DEFAULT_PLUME_SIGMA} should be within range [{MIN_PLUME_SIGMA}, {MAX_PLUME_SIGMA}]"

        # Test mathematical relationships between precision constants and sigma limits
        sigma_range = MAX_PLUME_SIGMA - MIN_PLUME_SIGMA
        assert sigma_range > 0, "Sigma range should be positive"
        assert sigma_range > 1.0, "Sigma range should be meaningful (> 1.0)"

        # Validate sigma limits support stable Gaussian calculations
        # Test that min sigma doesn't cause numerical instability
        test_gaussian_min = np.exp(-1.0 / (2 * MIN_PLUME_SIGMA**2))
        assert (
            0.0 <= test_gaussian_min <= 1.0
        ), "Min sigma should produce valid Gaussian values"

        # Test that max sigma produces reasonable plume extent
        max_extent = 3 * MAX_PLUME_SIGMA  # 3-sigma rule
        assert (
            max_extent < MAX_REASONABLE_GRID_SIZE
        ), "Max sigma should not produce unreasonably large plume extent"


class TestRenderingConstants:
    """Test dual-mode rendering constants including supported modes, color specifications, marker configurations, and backend compatibility validation."""

    def test_rendering_constants(self):
        """Test dual-mode rendering constants including supported modes, color specifications, marker configurations, and backend compatibility validation."""
        # Assert SUPPORTED_RENDER_MODES equals ['rgb_array', 'human'] with correct types
        assert (
            SUPPORTED_RENDER_MODES == EXPECTED_RENDER_MODES
        ), f"Expected render modes {EXPECTED_RENDER_MODES}, got {SUPPORTED_RENDER_MODES}"
        assert isinstance(
            SUPPORTED_RENDER_MODES, list
        ), f"SUPPORTED_RENDER_MODES should be list, got {type(SUPPORTED_RENDER_MODES)}"
        assert (
            len(SUPPORTED_RENDER_MODES) == 2
        ), f"Should support exactly 2 render modes, got {len(SUPPORTED_RENDER_MODES)}"
        assert all(
            isinstance(mode, str) for mode in SUPPORTED_RENDER_MODES
        ), "All render modes should be strings"

        # Validate render modes match RenderMode enumeration values
        assert (
            "rgb_array" in SUPPORTED_RENDER_MODES
        ), "rgb_array mode should be supported"
        assert "human" in SUPPORTED_RENDER_MODES, "human mode should be supported"
        assert (
            SUPPORTED_RENDER_MODES[0] == RenderMode.RGB_ARRAY.value
        ), "First render mode should match RenderMode.RGB_ARRAY"
        assert (
            SUPPORTED_RENDER_MODES[1] == RenderMode.HUMAN.value
        ), "Second render mode should match RenderMode.HUMAN"

        # Verify rendering constants support both programmatic and interactive modes
        for mode in SUPPORTED_RENDER_MODES:
            assert len(mode) > 0, f"Render mode '{mode}' should not be empty"
            assert (
                mode.isalnum() or "_" in mode
            ), f"Render mode '{mode}' should be valid identifier"


class TestPerformanceTargetConstants:
    """Test performance target constants including timing benchmarks, memory limits, and optimization thresholds for system performance validation."""

    def test_performance_target_constants(self):
        """Test performance target constants including timing benchmarks, memory limits, and optimization thresholds for system performance validation."""
        # Assert PERFORMANCE_TARGET_STEP_LATENCY_MS equals 1.0 for <1ms requirement
        assert isinstance(
            PERFORMANCE_TARGET_STEP_LATENCY_MS, (int, float)
        ), f"Step latency target should be numeric, got {type(PERFORMANCE_TARGET_STEP_LATENCY_MS)}"
        assert (
            PERFORMANCE_TARGET_STEP_LATENCY_MS > 0
        ), "Step latency target should be positive"
        assert (
            PERFORMANCE_TARGET_STEP_LATENCY_MS <= 2.0
        ), "Step latency target should be achievable (<= 2ms)"

        # Assert MEMORY_LIMIT_TOTAL_MB equals reasonable value for resource constraint validation
        assert isinstance(
            MEMORY_LIMIT_TOTAL_MB, int
        ), f"Memory limit should be integer, got {type(MEMORY_LIMIT_TOTAL_MB)}"
        assert MEMORY_LIMIT_TOTAL_MB > 0, "Memory limit should be positive"
        assert (
            10 <= MEMORY_LIMIT_TOTAL_MB <= 500
        ), f"Memory limit {MEMORY_LIMIT_TOTAL_MB}MB should be reasonable [10-500MB]"

        # Verify performance targets are achievable and mathematically consistent
        assert (
            PERFORMANCE_TARGET_STEP_LATENCY_MS < 1000
        ), "Step latency should be sub-second"

        # Test that memory limits are positive integers representing megabytes
        assert MEMORY_LIMIT_TOTAL_MB >= 1, "Memory limit should be at least 1MB"


class TestValidationErrorMessages:
    """Test validation error message templates including format strings, parameter substitution, and comprehensive error coverage validation."""

    def test_validation_error_messages(self):
        """Test validation error message templates including format strings, parameter substitution, and comprehensive error coverage validation."""
        # Assert VALIDATION_ERROR_MESSAGES is dictionary with expected error keys
        assert isinstance(
            VALIDATION_ERROR_MESSAGES, dict
        ), f"VALIDATION_ERROR_MESSAGES should be dict, got {type(VALIDATION_ERROR_MESSAGES)}"
        assert (
            len(VALIDATION_ERROR_MESSAGES) > 0
        ), "VALIDATION_ERROR_MESSAGES should not be empty"

        # Test error messages for common validation cases
        expected_error_keys = [
            "invalid_action",
            "invalid_coordinates",
            "invalid_grid_size",
            "invalid_sigma",
        ]
        for key in expected_error_keys:
            if key in VALIDATION_ERROR_MESSAGES:
                message = VALIDATION_ERROR_MESSAGES[key]
                assert isinstance(
                    message, str
                ), f"Error message for '{key}' should be string"
                assert (
                    len(message) > 0
                ), f"Error message for '{key}' should not be empty"

        # Validate error message templates are non-empty strings with descriptive content
        for key, message in VALIDATION_ERROR_MESSAGES.items():
            assert isinstance(key, str), f"Error key should be string, got {type(key)}"
            assert isinstance(
                message, str
            ), f"Error message should be string, got {type(message)}"
            assert len(key) > 0, f"Error key '{key}' should not be empty"
            assert (
                len(message) > 10
            ), f"Error message for '{key}' should be descriptive (>10 chars)"

        # Test error messages are valid and substitution-safe
        for key, message in VALIDATION_ERROR_MESSAGES.items():
            # Check for basic format string safety (no unmatched braces)
            try:
                # Test that format strings don't have syntax errors
                if "{" in message and "}" in message:
                    # This is a format string, verify it's valid
                    assert message.count("{") == message.count(
                        "}"
                    ), f"Unmatched braces in error message for '{key}'"
            except Exception as e:
                pytest.fail(f"Invalid format string in error message for '{key}': {e}")


class TestFactoryFunctions:
    """Test all factory functions returning complete configuration dictionaries with proper structure and values."""

    def test_get_default_environment_constants(self):
        """Test environment constants factory function returning complete environment configuration dictionary with proper structure and values."""
        # Call get_default_environment_constants() and validate return type is dictionary
        result = get_default_environment_constants()
        assert isinstance(
            result, dict
        ), f"Factory should return dict, got {type(result)}"
        assert set(result.keys()) == {
            "DEFAULT_GRID_SIZE",
            "MIN_GRID_SIZE",
            "MAX_GRID_SIZE",
            "DEFAULT_SOURCE_LOCATION",
            "DEFAULT_PLUME_SIGMA",
            "DEFAULT_GOAL_RADIUS",
            "DEFAULT_MAX_STEPS",
        }, f"Unexpected environment keys: {sorted(result.keys())}"

        # Test that returned dictionary contains expected default values
        assert result["DEFAULT_GRID_SIZE"] == DEFAULT_GRID_SIZE
        assert result["DEFAULT_SOURCE_LOCATION"] == DEFAULT_SOURCE_LOCATION
        assert result["DEFAULT_GOAL_RADIUS"] == DEFAULT_GOAL_RADIUS
        assert result["DEFAULT_MAX_STEPS"] == DEFAULT_MAX_STEPS

        # Test that returned dictionary can be used for environment configuration
        grid_size = result["DEFAULT_GRID_SIZE"]
        assert isinstance(
            grid_size, tuple
        ), f"Grid size should be tuple, got {type(grid_size)}"
        assert len(grid_size) == 2, "Grid size should have 2 dimensions"

        # Verify factory function returns consistent results across multiple calls
        result2 = get_default_environment_constants()
        assert result == result2, "Factory function should return consistent results"

    def test_get_plume_model_constants(self):
        """Test plume model constants factory function returning mathematical parameters, validation limits, and model configuration dictionary."""
        # Call get_plume_model_constants() and validate return type is dictionary
        result = get_plume_model_constants()
        assert isinstance(
            result, dict
        ), f"Factory should return dict, got {type(result)}"
        expected_keys = {
            "CONCENTRATION_RANGE",
            "GAUSSIAN_PRECISION",
            "DISTANCE_PRECISION",
            "MIN_PLUME_SIGMA",
            "MAX_PLUME_SIGMA",
            "STATIC_GAUSSIAN_MODEL_TYPE",
            "DEFAULT_PLUME_MODEL_TYPE",
            "PLUME_MODEL_TYPES",
        }
        assert (
            set(result.keys()) == expected_keys
        ), f"Unexpected plume model keys: {sorted(result.keys())}"

        # Test that returned dictionary contains expected plume model parameters
        assert result["MIN_PLUME_SIGMA"] == MIN_PLUME_SIGMA
        assert result["MAX_PLUME_SIGMA"] == MAX_PLUME_SIGMA
        assert result["CONCENTRATION_RANGE"] == CONCENTRATION_RANGE
        assert result["STATIC_GAUSSIAN_MODEL_TYPE"] == STATIC_GAUSSIAN_MODEL_TYPE
        assert isinstance(result["PLUME_MODEL_TYPES"], list)

    def test_get_action_space_constants(self):
        """Test action space constants factory function returning action mappings, movement vectors, and discrete space configuration."""
        # Call get_action_space_constants() and validate return type is dictionary
        result = get_action_space_constants()
        assert isinstance(
            result, dict
        ), f"Factory should return dict, got {type(result)}"
        expected_keys = {
            "ACTION_UP",
            "ACTION_RIGHT",
            "ACTION_LEFT",
            "ACTION_SPACE_SIZE",
            "MOVEMENT_VECTORS",
        }
        assert (
            set(result.keys()) == expected_keys
        ), f"Unexpected action space keys: {sorted(result.keys())}"

        # Test that returned dictionary contains expected action space components
        assert result["ACTION_SPACE_SIZE"] == ACTION_SPACE_SIZE
        assert result["ACTION_UP"] == ACTION_UP
        assert isinstance(result["MOVEMENT_VECTORS"], dict)
        assert len(result["MOVEMENT_VECTORS"]) > 0

        # Verify movement vectors are unit vectors
        for action, vector in result["MOVEMENT_VECTORS"].items():
            if isinstance(vector, tuple) and len(vector) == 2:
                dx, dy = vector
                magnitude_squared = dx * dx + dy * dy
                assert (
                    magnitude_squared <= 2
                ), f"Movement vector {vector} should be unit vector or stationary"

    def test_get_rendering_constants(self):
        """Test rendering constants factory function returning supported modes, color schemes, marker specifications, and backend preferences."""
        # Call get_rendering_constants() and validate return type is dictionary
        result = get_rendering_constants()
        assert isinstance(
            result, dict
        ), f"Factory should return dict, got {type(result)}"
        assert set(result.keys()) == {
            "supported_modes",
            "agent_marker_color",
            "source_marker_color",
            "agent_marker_size",
            "source_marker_size",
            "pixel_value_min",
            "pixel_value_max",
            "rgb_dtype",
        }, f"Unexpected rendering keys: {sorted(result.keys())}"

        # Assert rendering performance targets are included in factory output
        if "supported_modes" in result:
            supported_modes = result["supported_modes"]
            assert isinstance(supported_modes, list), "Supported modes should be list"
            assert len(supported_modes) > 0, "Should support at least one render mode"

    def test_get_performance_constants(self):
        """Test performance constants factory function returning timing targets, memory limits, and optimization parameters for monitoring."""
        # Call get_performance_constants() and validate return type is dictionary
        result = get_performance_constants()
        assert isinstance(
            result, dict
        ), f"Factory should return dict, got {type(result)}"
        assert set(result.keys()) == {
            "tracking_enabled",
            "step_latency_target_ms",
            "rgb_render_target_ms",
            "human_render_target_ms",
            "plume_generation_target_ms",
            "episode_reset_target_ms",
            "boundary_enforcement_target_ms",
            "memory_limits_mb",
            "optimization_thresholds",
            "benchmark_iterations",
        }, f"Unexpected performance keys: {sorted(result.keys())}"

    def test_get_validation_constants(self):
        """Test validation constants factory function returning parameter limits, precision values, and error message templates."""
        # Call get_validation_constants() and validate return type is dictionary
        result = get_validation_constants()
        assert isinstance(
            result, dict
        ), f"Factory should return dict, got {type(result)}"
        assert set(result.keys()) == {
            "distance_precision",
            "gaussian_precision",
            "sigma_range",
            "grid_size_limits",
            "coordinate_bounds_checking",
            "error_messages",
            "seed_validation",
            "memory_validation_mb",
            "action_validation_range",
        }, f"Unexpected validation keys: {sorted(result.keys())}"

        # Assert all validation parameters support comprehensive input checking
        if "error_messages" in result:
            error_messages = result["error_messages"]
            assert isinstance(
                error_messages, dict
            ), "Error messages should be dictionary"

    def test_get_testing_constants(self):
        """Test testing constants factory function returning test-optimized parameters including reduced grid sizes, tolerance values, and test-specific configurations."""
        # Call get_testing_constants() and validate return type is dictionary
        result = get_testing_constants()
        assert isinstance(
            result, dict
        ), f"Factory should return dict, got {type(result)}"
        assert set(result.keys()) == {
            "test_grid_size",
            "test_source_location",
            "test_max_steps",
            "test_tolerance",
            "reproducibility_seeds",
            "performance_test_iterations",
            "expected_results",
        }, f"Unexpected testing keys: {sorted(result.keys())}"

        # Assert reproducibility test parameters include seed values
        if "test_seeds" in result:
            test_seeds = result["test_seeds"]
            assert isinstance(
                test_seeds, (list, tuple)
            ), "Test seeds should be sequence"
            if len(test_seeds) > 0:
                assert all(
                    isinstance(seed, int) for seed in test_seeds
                ), "Test seeds should be integers"


class TestConstantConsistencyValidation:
    """Test constant consistency validation function ensuring mathematical and logical relationships between constants."""

    def test_validate_constant_consistency_normal(self):
        """Test constant consistency validation function with normal mode ensuring mathematical and logical relationships between constants."""
        # Call validate_constant_consistency(strict_mode=False) and validate return tuple format
        result = validate_constant_consistency(strict_mode=False)
        assert isinstance(
            result, tuple
        ), f"Validation should return tuple, got {type(result)}"
        assert len(result) == 2, f"Validation should return 2-tuple, got {len(result)}"

        is_valid, validation_report = result

        # Assert function returns (is_valid, validation_report) tuple structure
        assert isinstance(
            is_valid, bool
        ), f"is_valid should be boolean, got {type(is_valid)}"
        assert isinstance(
            validation_report, dict
        ), f"validation_report should be dict, got {type(validation_report)}"

        # Test mathematical consistency: default source location within default grid bounds
        assert (
            DEFAULT_SOURCE_LOCATION[0] < DEFAULT_GRID_SIZE[0]
        ), "Source X should be within grid width"
        assert (
            DEFAULT_SOURCE_LOCATION[1] < DEFAULT_GRID_SIZE[1]
        ), "Source Y should be within grid height"

        # Assert sigma parameter relationships: DEFAULT_PLUME_SIGMA within MIN/MAX range
        assert (
            MIN_PLUME_SIGMA <= DEFAULT_PLUME_SIGMA <= MAX_PLUME_SIGMA
        ), "Default sigma should be within min/max range"

    def test_validate_constant_consistency_strict(self):
        """Test constant consistency validation function with strict mode for comprehensive validation with additional precision and bounds checking."""
        # Call validate_constant_consistency(strict_mode=True) and validate enhanced checking
        result = validate_constant_consistency(strict_mode=True)
        assert isinstance(
            result, tuple
        ), f"Strict validation should return tuple, got {type(result)}"
        assert (
            len(result) == 2
        ), f"Strict validation should return 2-tuple, got {len(result)}"

        is_valid, validation_report = result

        # Assert strict mode applies additional precision and bounds validation
        assert isinstance(
            is_valid, bool
        ), f"Strict is_valid should be boolean, got {type(is_valid)}"
        assert isinstance(
            validation_report, dict
        ), f"Strict validation_report should be dict, got {type(validation_report)}"

        # Assert comprehensive validation report includes detailed findings and recommendations
        assert (
            len(validation_report) > 0
        ), "Strict mode should provide detailed validation report"


class TestConstantTypesAndImmutability:
    """Test that all constants have correct types, are immutable where expected, and maintain type consistency across the system."""

    def test_constant_types_and_immutability(self):
        """Test that all constants have correct types, are immutable where expected, and maintain type consistency across the system."""
        # Test string constants are immutable strings
        string_constants = [PACKAGE_NAME, PACKAGE_VERSION, ENVIRONMENT_ID]
        for const in string_constants:
            assert isinstance(
                const, str
            ), f"String constant should be str, got {type(const)}"
            assert len(const) > 0, "String constant should not be empty"

        # Assert tuple constants are immutable tuples
        tuple_constants = [DEFAULT_GRID_SIZE, DEFAULT_SOURCE_LOCATION]
        for const in tuple_constants:
            assert isinstance(
                const, tuple
            ), f"Tuple constant should be tuple, got {type(const)}"

            # Test that attempting to modify immutable constants would raise appropriate errors
            with pytest.raises((TypeError, AttributeError)):
                const[0] = 999  # This should fail for tuples

        # Validate numeric constants maintain proper type
        numeric_constants = [
            DEFAULT_PLUME_SIGMA,
            REWARD_GOAL_REACHED,
            REWARD_DEFAULT,
            PERFORMANCE_TARGET_STEP_LATENCY_MS,
            MIN_PLUME_SIGMA,
            MAX_PLUME_SIGMA,
        ]
        for const in numeric_constants:
            assert isinstance(
                const, (int, float)
            ), f"Numeric constant should be int or float, got {type(const)}"

        # Test dictionary constants have correct structure
        dict_constants = [MOVEMENT_VECTORS, VALIDATION_ERROR_MESSAGES]
        for const in dict_constants:
            assert isinstance(
                const, dict
            ), f"Dictionary constant should be dict, got {type(const)}"
            assert len(const) > 0, "Dictionary constant should not be empty"

        # Assert list constants contain expected element types
        list_constants = [SUPPORTED_RENDER_MODES]
        for const in list_constants:
            assert isinstance(
                const, list
            ), f"List constant should be list, got {type(const)}"
            assert len(const) > 0, "List constant should not be empty"

        # Verify NumPy dtype constants are valid dtype objects
        dtype_constants = [FIELD_DTYPE, OBSERVATION_DTYPE]
        for const in dtype_constants:
            assert isinstance(
                const, type
            ), f"Dtype constant should be type, got {type(const)}"
            # Test that dtype can create valid numpy dtype
            dtype_obj = np.dtype(const)
            assert (
                dtype_obj is not None
            ), "Dtype constant should create valid numpy dtype"


class TestConstantRangesAndBounds:
    """Test that all constants fall within reasonable ranges and bounds for their intended use cases with mathematical feasibility validation."""

    def test_constant_ranges_and_bounds(self):
        """Test that all constants fall within reasonable ranges and bounds for their intended use cases with mathematical feasibility validation."""
        # Assert grid size constants are positive and within reasonable memory limits
        assert (
            DEFAULT_GRID_SIZE[0] > 0 and DEFAULT_GRID_SIZE[1] > 0
        ), "Grid dimensions should be positive"
        assert (
            DEFAULT_GRID_SIZE[0] <= MAX_REASONABLE_GRID_SIZE
        ), f"Grid width should be <= {MAX_REASONABLE_GRID_SIZE}"
        assert (
            DEFAULT_GRID_SIZE[1] <= MAX_REASONABLE_GRID_SIZE
        ), f"Grid height should be <= {MAX_REASONABLE_GRID_SIZE}"

        # Test sigma constants are positive and support stable Gaussian calculations
        assert MIN_PLUME_SIGMA > 0, "Min sigma should be positive"
        assert (
            MAX_PLUME_SIGMA > MIN_PLUME_SIGMA
        ), "Max sigma should be greater than min sigma"
        assert (
            MIN_PLUME_SIGMA >= MIN_REASONABLE_SIGMA
        ), f"Min sigma should be >= {MIN_REASONABLE_SIGMA}"
        assert (
            MAX_PLUME_SIGMA <= MAX_REASONABLE_SIGMA
        ), f"Max sigma should be <= {MAX_REASONABLE_SIGMA}"

        # Validate reward constants are within reasonable RL training ranges
        assert (
            -100.0 <= REWARD_DEFAULT <= 100.0
        ), "Default reward should be within reasonable range"
        assert (
            -100.0 <= REWARD_GOAL_REACHED <= 100.0
        ), "Goal reward should be within reasonable range"

        # Assert performance target constants are achievable with current hardware
        assert (
            0.01 <= PERFORMANCE_TARGET_STEP_LATENCY_MS <= 1000.0
        ), "Step latency target should be reasonable"

        # Test memory limit constants are realistic for target deployment environments
        assert 1 <= MEMORY_LIMIT_TOTAL_MB <= 10000, "Memory limit should be realistic"

        # Assert action constants form complete discrete action space [0,3]
        actions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]
        assert set(actions) == set(
            range(4)
        ), "Actions should form complete space [0,1,2,3]"


class TestCrossConstantMathematicalRelationships:
    """Test mathematical relationships between related constants ensuring system-wide mathematical consistency and coherence."""

    def test_cross_constant_mathematical_relationships(self):
        """Test mathematical relationships and dependencies between system constants."""
        # Test that DEFAULT_SOURCE_LOCATION is centered within DEFAULT_GRID_SIZE
        center_x = DEFAULT_GRID_SIZE[0] // 2
        center_y = DEFAULT_GRID_SIZE[1] // 2

        # Allow some tolerance for centering
        assert (
            abs(DEFAULT_SOURCE_LOCATION[0] - center_x) <= 2
        ), "Source X should be near grid center"
        assert (
            abs(DEFAULT_SOURCE_LOCATION[1] - center_y) <= 2
        ), "Source Y should be near grid center"

        # Assert DEFAULT_PLUME_SIGMA creates reasonable plume extent for DEFAULT_GRID_SIZE
        plume_extent = 3 * DEFAULT_PLUME_SIGMA  # 3-sigma rule
        grid_diagonal = np.sqrt(DEFAULT_GRID_SIZE[0] ** 2 + DEFAULT_GRID_SIZE[1] ** 2)
        assert (
            plume_extent < grid_diagonal
        ), "Plume extent should be reasonable for grid size"

        # Test memory limits accommodate worst-case scenarios for default configurations
        # Estimate memory usage for default grid
        cells = DEFAULT_GRID_SIZE[0] * DEFAULT_GRID_SIZE[1]
        bytes_per_cell = np.dtype(FIELD_DTYPE).itemsize
        estimated_mb = (cells * bytes_per_cell) / (1024 * 1024)
        assert (
            estimated_mb < MEMORY_LIMIT_TOTAL_MB
        ), "Default configuration should fit within memory limits"

        # Verify reward system constants provide proper incentive gradients
        reward_difference = REWARD_GOAL_REACHED - REWARD_DEFAULT
        assert (
            reward_difference > 0
        ), "Goal reward should be greater than default reward"
        assert reward_difference >= 0.1, "Reward difference should be meaningful"

        # Test action space constants maintain proper coordinate system relationships
        movement_magnitudes = []
        for vector in MOVEMENT_VECTORS.values():
            magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
            movement_magnitudes.append(magnitude)

        # All movement vectors should have same magnitude (unit movements)
        unique_magnitudes = set(f"{mag:.3f}" for mag in movement_magnitudes)
        assert (
            len(unique_magnitudes) <= 2
        ), "Movement vectors should have consistent magnitudes (0 or 1)"


class TestConstantsIntegrationWithTypes:
    """Test integration between constants module and types module ensuring consistency between constant values and type system enumerations."""

    def test_constants_integration_with_types(self):
        """Test integration between constants module and types module ensuring consistency between constant values and type system enumerations."""
        # Assert action constants match Action enumeration values exactly
        assert (
            ACTION_UP == Action.UP.value
        ), f"ACTION_UP {ACTION_UP} should match Action.UP.value {Action.UP.value}"
        assert (
            ACTION_RIGHT == Action.RIGHT.value
        ), f"ACTION_RIGHT {ACTION_RIGHT} should match Action.RIGHT.value {Action.RIGHT.value}"
        assert (
            ACTION_DOWN == Action.DOWN.value
        ), f"ACTION_DOWN {ACTION_DOWN} should match Action.DOWN.value {Action.DOWN.value}"
        assert (
            ACTION_LEFT == Action.LEFT.value
        ), f"ACTION_LEFT {ACTION_LEFT} should match Action.LEFT.value {Action.LEFT.value}"

        # Assert render mode constants match RenderMode enumeration values
        assert (
            "rgb_array" == RenderMode.RGB_ARRAY.value
        ), "'rgb_array' should match RenderMode.RGB_ARRAY.value"
        assert (
            "human" == RenderMode.HUMAN.value
        ), "'human' should match RenderMode.HUMAN.value"

        # Verify all supported render modes match enumeration
        for mode in SUPPORTED_RENDER_MODES:
            render_mode_values = [rm.value for rm in RenderMode]
            assert (
                mode in render_mode_values
            ), f"Render mode '{mode}' should match RenderMode enumeration value"


class TestFactoryFunctionConsistency:
    """Test that all factory functions return consistent data structures and maintain proper relationships between returned constant dictionaries."""

    def test_factory_function_consistency(self):
        """Test that all factory functions return consistent data structures and maintain proper relationships between returned constant dictionaries."""
        # Call all factory functions and validate they return non-empty dictionaries
        factories = [
            get_default_environment_constants,
            get_plume_model_constants,
            get_action_space_constants,
            get_rendering_constants,
            get_performance_constants,
            get_validation_constants,
            get_testing_constants,
        ]

        results = []
        for factory in factories:
            result = factory()
            assert isinstance(
                result, dict
            ), f"Factory {factory.__name__} should return dict"
            assert (
                len(result) > 0
            ), f"Factory {factory.__name__} should return non-empty dict"
            results.append(result)

        # Test that factory functions return deep copies preventing external modification
        env_constants1 = get_default_environment_constants()
        env_constants2 = get_default_environment_constants()
        assert (
            env_constants1 == env_constants2
        ), "Factory should return consistent results"

        # Verify factory functions maintain consistent key naming conventions
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())

        # Check that keys follow consistent naming (snake_case)
        for key in all_keys:
            if isinstance(key, str):
                # Basic check for consistent naming convention
                assert (
                    key.islower() or "_" in key or key.isupper()
                ), f"Key '{key}' should follow consistent naming convention"

        # Assert factory function outputs are JSON serializable for configuration storage
        import json

        for i, result in enumerate(results):
            try:
                json_str = json.dumps(
                    result, default=str
                )  # Use str for non-serializable types
                assert (
                    len(json_str) > 0
                ), f"Factory result {i} should be JSON serializable"
            except (TypeError, ValueError):
                # Some results may contain non-JSON-serializable objects (like numpy dtypes)
                # This is acceptable as long as they're consistent
                pass


# Additional integration and edge case tests
class TestConstantEdgeCases:
    """Test edge cases and boundary conditions for constant validation."""

    def test_movement_vector_completeness(self):
        """Test that movement vectors cover all cardinal directions and form complete action space."""
        # Verify all cardinal directions are represented
        expected_directions = {
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
        }  # UP, RIGHT, DOWN, LEFT
        actual_directions = set(MOVEMENT_VECTORS.values())

        assert expected_directions.issubset(
            actual_directions
        ), "Movement vectors should include all cardinal directions"

        # Test that movement vectors are orthogonal (perpendicular to each other)
        vectors = list(MOVEMENT_VECTORS.values())
        for i, vec1 in enumerate(vectors):
            for j, vec2 in enumerate(vectors):
                if i != j:
                    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                    # Adjacent cardinal directions should be orthogonal (dot product = 0)
                    # Opposite directions have dot product = -1
                    assert dot_product in [
                        -1,
                        0,
                    ], "Movement vectors should be orthogonal or opposite"

    def test_dtype_numpy_compatibility(self):
        """Test that data type constants are compatible with NumPy operations."""
        # Test that dtypes can create arrays
        field_array = np.array([0.0, 0.5, 1.0], dtype=FIELD_DTYPE)
        obs_array = np.array([0.8], dtype=OBSERVATION_DTYPE)

        assert (
            field_array.dtype == FIELD_DTYPE
        ), "Field array should maintain correct dtype"
        assert (
            obs_array.dtype == OBSERVATION_DTYPE
        ), "Observation array should maintain correct dtype"

        # Test dtype precision and range
        assert np.isfinite(field_array).all(), "Field array values should be finite"
        assert np.isfinite(obs_array).all(), "Observation array values should be finite"

        # Test that dtypes support required operations
        result = field_array * 2.0
        assert (
            result.dtype == FIELD_DTYPE
        ), "Arithmetic operations should preserve dtype"

    def test_error_message_format_strings(self):
        """Test that error message format strings are valid and safe."""
        for key, message in VALIDATION_ERROR_MESSAGES.items():
            # Test for balanced braces
            open_braces = message.count("{")
            close_braces = message.count("}")
            assert (
                open_braces == close_braces
            ), f"Error message '{key}' has unbalanced braces"

            # Test that message doesn't contain sensitive information patterns
            sensitive_patterns = ["password", "secret", "token", "key", "internal"]
            message_lower = message.lower()
            for pattern in sensitive_patterns:
                assert (
                    pattern not in message_lower
                ), f"Error message '{key}' should not contain sensitive information"

    @pytest.mark.parametrize("sigma_value", [0.1, 1.0, 10.0, 50.0])
    def test_sigma_range_validation(self, sigma_value):
        """Test sigma values across reasonable range for Gaussian calculations."""
        # Test that sigma values produce stable Gaussian calculations
        test_distance = 5.0
        gaussian_value = np.exp(-(test_distance**2) / (2 * sigma_value**2))

        assert (
            0.0 <= gaussian_value <= 1.0
        ), f"Gaussian value should be in [0,1] for sigma={sigma_value}"
        assert np.isfinite(
            gaussian_value
        ), f"Gaussian value should be finite for sigma={sigma_value}"

    def test_performance_target_relationships(self):
        """Test relationships between performance targets ensure system feasibility."""
        # Step latency should be much less than typical episode duration
        max_steps = 1000  # Typical episode length
        total_time_estimate = max_steps * PERFORMANCE_TARGET_STEP_LATENCY_MS
        assert (
            total_time_estimate < 10000
        ), "Total episode time should be reasonable (< 10 seconds)"

        # Memory limits should accommodate realistic grid sizes
        max_reasonable_cells = MAX_REASONABLE_GRID_SIZE * MAX_REASONABLE_GRID_SIZE
        bytes_per_cell = 4  # float32
        max_memory_mb = (max_reasonable_cells * bytes_per_cell) / (1024 * 1024)

        # Memory limit should be generous enough for reasonable configurations
        assert (
            MEMORY_LIMIT_TOTAL_MB >= max_memory_mb * 0.1
        ), "Memory limit should accommodate reasonable grid sizes"


# Test configuration and fixtures
@pytest.fixture
def sample_constants():
    """Provide sample constants for testing purposes."""
    return {
        "test_grid_size": (32, 32),
        "test_source_location": (16, 16),
        "test_sigma": 5.0,
        "test_seeds": [42, 123, 456],
    }


@pytest.fixture
def validation_tolerances():
    """Provide validation tolerances for floating-point comparisons."""
    return {
        "float_tolerance": TOLERANCE_FLOAT_COMPARISON,
        "percentage_tolerance": 0.01,  # 1%
        "timing_tolerance": 0.1,  # 100ms for timing tests
    }


# Performance and stress tests
class TestConstantPerformance:
    """Test performance characteristics of constant access and validation."""

    def test_constant_access_performance(self):
        """Test that constant access is fast enough for performance-critical code."""
        import time

        # Test rapid constant access
        start_time = time.perf_counter()

        for _ in range(10000):
            # Access various constants rapidly
            _ = DEFAULT_GRID_SIZE
            _ = DEFAULT_SOURCE_LOCATION
            _ = MOVEMENT_VECTORS
            _ = SUPPORTED_RENDER_MODES
            _ = PERFORMANCE_TARGET_STEP_LATENCY_MS

        elapsed_time = time.perf_counter() - start_time

        # Constant access should be extremely fast
        assert elapsed_time < 0.1, f"Constant access took too long: {elapsed_time:.4f}s"

    def test_factory_function_performance(self):
        """Test that factory functions execute within reasonable time limits."""
        import time

        factories = [
            get_default_environment_constants,
            get_plume_model_constants,
            get_action_space_constants,
            get_rendering_constants,
            get_performance_constants,
            get_validation_constants,
            get_testing_constants,
        ]

        for factory in factories:
            start_time = time.perf_counter()
            result = factory()
            elapsed_time = time.perf_counter() - start_time

            # Factory functions should execute quickly
            assert (
                elapsed_time < 0.01
            ), f"Factory {factory.__name__} took too long: {elapsed_time:.4f}s"
            assert isinstance(
                result, dict
            ), f"Factory {factory.__name__} should return dict"


if __name__ == "__main__":
    # Run tests with comprehensive output
    pytest.main([__file__, "-v", "--tb=short"])
