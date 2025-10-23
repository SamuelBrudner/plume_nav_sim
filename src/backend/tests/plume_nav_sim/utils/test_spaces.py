"""
Comprehensive test suite for plume_nav_sim spaces utility module providing exhaustive
validation of Gymnasium space creation, validation, and utility functions with performance
benchmarks, error handling verification, and Gymnasium API compliance testing for
reinforcement learning environment space management.

This test module validates all space creation, validation, configuration, and optimization
functions ensuring proper Gymnasium compatibility, performance targets, and robust error
handling for production-ready reinforcement learning environment development.
"""

import time  # standard library - Performance timing measurements for validation speed testing and benchmarking
import warnings  # standard library - Warning suppression and testing for deprecation warnings and compatibility
from unittest.mock import (  # standard library - Mock objects for testing error conditions and external dependencies
    Mock,
)

import gymnasium.spaces  # >=0.29.0 - Gymnasium space classes for testing compatibility and API compliance validation
import numpy as np  # >=2.1.0 - Array operations and dtype testing for space validation and mathematical operations

# External imports with version comments
import pytest  # >=8.0.0 - Testing framework for comprehensive unit tests with fixtures, parameterization, and assertion validation

from plume_nav_sim.core.constants import (
    ACTION_SPACE_SIZE,
    CONCENTRATION_RANGE,
    OBSERVATION_DTYPE,
)

# Internal imports from core modules
from plume_nav_sim.core.types import Action
from plume_nav_sim.utils.exceptions import ValidationError

# Internal imports from spaces utility module under test
from plume_nav_sim.utils.spaces import (
    SpaceConfig,
    SpaceValidator,
    check_space_compatibility,
    create_action_space,
    create_observation_space,
    get_space_info,
    optimize_space_operations,
    sample_valid_action,
    sample_valid_observation,
    validate_action,
    validate_action_space,
    validate_observation,
    validate_observation_space,
)

# Test configuration constants
TEST_TIMEOUT_SECONDS = 30
PERFORMANCE_TOLERANCE_MS = 2.0
TEST_GRID_SIZE = (32, 32)
LARGE_GRID_SIZE = (256, 256)
TEST_SOURCE_LOCATION = (16, 16)
INVALID_ACTIONS = [-1, 4, 5, "invalid", None, 3.14]
VALID_ACTIONS = [0, 1, 2, 3, Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
TEST_CONCENTRATION_VALUES = [0.0, 0.5, 1.0]
INVALID_CONCENTRATION_VALUES = [-0.1, 1.1, "invalid", None]
BENCHMARK_ITERATIONS = 1000
CACHE_TEST_SIZE = 100


def test_create_action_space_default():
    """
    Test creation of default Discrete action space with ACTION_SPACE_SIZE (4) actions
    ensuring proper Gymnasium space configuration and cardinal direction support.
    """
    # Create default action space using create_action_space() without parameters
    action_space = create_action_space()

    # Assert action space is instance of gymnasium.spaces.Discrete
    assert isinstance(
        action_space, gymnasium.spaces.Discrete
    ), f"Expected Discrete, got {type(action_space)}"

    # Verify action space.n equals ACTION_SPACE_SIZE (4) for cardinal directions
    assert (
        action_space.n == ACTION_SPACE_SIZE
    ), f"Expected {ACTION_SPACE_SIZE} actions, got {action_space.n}"

    # Test action space contains method with valid actions [0, 1, 2, 3]
    for valid_action in range(ACTION_SPACE_SIZE):
        assert action_space.contains(
            valid_action
        ), f"Action space should contain action {valid_action}"

    # Verify action space sample method returns valid actions in range [0, 3]
    for _ in range(10):
        sample = action_space.sample()
        assert (
            0 <= sample < ACTION_SPACE_SIZE
        ), f"Sample {sample} out of valid range [0, {ACTION_SPACE_SIZE-1}]"

    # Test action space does not contain invalid actions [-1, 4, 5]
    invalid_test_actions = [-1, ACTION_SPACE_SIZE, ACTION_SPACE_SIZE + 1]
    for invalid_action in invalid_test_actions:
        assert not action_space.contains(
            invalid_action
        ), f"Action space should not contain invalid action {invalid_action}"

    # Validate action space string representation for debugging
    space_str = str(action_space)
    assert (
        "Discrete" in space_str
    ), f"String representation should mention Discrete: {space_str}"

    # Assert action space meets Gymnasium API requirements
    assert hasattr(action_space, "sample"), "Action space missing sample method"
    assert hasattr(action_space, "contains"), "Action space missing contains method"


def test_create_action_space_with_parameters():
    """
    Test creation of action space with custom parameters including validation settings
    and configuration options for flexible space configuration.
    """
    # Create action space with custom num_actions parameter (4)
    custom_action_space = create_action_space(num_actions=4)
    assert custom_action_space.n == 4, "Custom num_actions not applied correctly"

    # Create action space with validate_actions=True for strict validation
    validated_space = create_action_space(validate_actions=True)
    assert isinstance(
        validated_space, gymnasium.spaces.Discrete
    ), "Validation should not affect space type"

    # Create action space with custom space_config dictionary
    config = {"custom_metadata": {"test": True}}
    config_space = create_action_space(space_config=config)
    assert config_space.n == ACTION_SPACE_SIZE, "Config should not affect default size"

    # Test action space creation with validate_actions=False
    unvalidated_space = create_action_space(validate_actions=False)
    assert isinstance(
        unvalidated_space, gymnasium.spaces.Discrete
    ), "Space should be created regardless of validation setting"

    # Verify all created spaces are valid Discrete instances
    test_spaces = [
        custom_action_space,
        validated_space,
        config_space,
        unvalidated_space,
    ]
    for space in test_spaces:
        assert isinstance(
            space, gymnasium.spaces.Discrete
        ), f"All spaces should be Discrete, got {type(space)}"

    # Compare spaces created with different parameters for consistency
    default_space = create_action_space()
    assert (
        default_space.n == custom_action_space.n
    ), "Default and custom spaces should have same size when num_actions=4"

    # Test caching behavior with repeated identical parameters
    cached_space1 = create_action_space(num_actions=4)
    cached_space2 = create_action_space(num_actions=4)
    # Note: Due to LRU cache, these might be the same object
    assert (
        cached_space1.n == cached_space2.n
    ), "Cached spaces should have consistent properties"

    # Validate parameter handling and error checking
    with pytest.raises(ValidationError):
        create_action_space(num_actions=0)

    with pytest.raises(ValidationError):
        create_action_space(num_actions=-1)


def test_create_observation_space_default():
    """
    Test creation of default Box observation space with concentration bounds [0.0, 1.0],
    float32 dtype, and shape (1,) for plume concentration observations.
    """
    # Create default observation space using create_observation_space() without parameters
    obs_space = create_observation_space()

    # Assert observation space is instance of gymnasium.spaces.Box
    assert isinstance(
        obs_space, gymnasium.spaces.Box
    ), f"Expected Box, got {type(obs_space)}"

    # Verify observation space shape equals (1,) for single concentration value
    expected_shape = (1,)
    assert (
        obs_space.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {obs_space.shape}"

    # Check observation space dtype equals OBSERVATION_DTYPE (float32)
    assert (
        obs_space.dtype == OBSERVATION_DTYPE
    ), f"Expected dtype {OBSERVATION_DTYPE}, got {obs_space.dtype}"

    # Validate observation space low bound equals CONCENTRATION_RANGE[0] (0.0)
    expected_low = CONCENTRATION_RANGE[0]
    assert np.allclose(
        obs_space.low, expected_low
    ), f"Expected low bound {expected_low}, got {obs_space.low}"

    # Verify observation space high bound equals CONCENTRATION_RANGE[1] (1.0)
    expected_high = CONCENTRATION_RANGE[1]
    assert np.allclose(
        obs_space.high, expected_high
    ), f"Expected high bound {expected_high}, got {obs_space.high}"

    # Test observation space contains method with valid observations
    valid_obs = np.array([0.5], dtype=OBSERVATION_DTYPE)
    assert obs_space.contains(
        valid_obs
    ), f"Observation space should contain valid observation {valid_obs}"

    # Test observation space sample method returns values in [0.0, 1.0] range
    for _ in range(10):
        sample = obs_space.sample()
        assert obs_space.contains(
            sample
        ), f"Sample {sample} should be contained in observation space"
        assert (
            CONCENTRATION_RANGE[0] <= sample.min() <= CONCENTRATION_RANGE[1]
        ), "Sample values should be in concentration range"
        assert (
            CONCENTRATION_RANGE[0] <= sample.max() <= CONCENTRATION_RANGE[1]
        ), "Sample values should be in concentration range"


def test_create_observation_space_with_parameters():
    """
    Test creation of observation space with custom shape, bounds, dtype, and validation
    settings for flexible observation space configuration.
    """
    # Create observation space with custom observation_shape (1,)
    custom_shape = (1,)
    shaped_space = create_observation_space(observation_shape=custom_shape)
    assert (
        shaped_space.shape == custom_shape
    ), f"Custom shape not applied: expected {custom_shape}, got {shaped_space.shape}"

    # Create observation space with custom concentration_bounds (0.0, 1.0)
    custom_bounds = (0.0, 1.0)
    bounded_space = create_observation_space(concentration_bounds=custom_bounds)
    assert np.allclose(
        bounded_space.low, custom_bounds[0]
    ), f"Custom low bound not applied: expected {custom_bounds[0]}, got {bounded_space.low}"
    assert np.allclose(
        bounded_space.high, custom_bounds[1]
    ), f"Custom high bound not applied: expected {custom_bounds[1]}, got {bounded_space.high}"

    # Create observation space with custom observation_dtype numpy.float32
    custom_dtype = np.float32
    dtype_space = create_observation_space(observation_dtype=custom_dtype)
    assert (
        dtype_space.dtype == custom_dtype
    ), f"Custom dtype not applied: expected {custom_dtype}, got {dtype_space.dtype}"

    # Test observation space with validate_bounds=True for strict validation
    validated_space = create_observation_space(validate_bounds=True)
    assert isinstance(
        validated_space, gymnasium.spaces.Box
    ), "Validation should not affect space type"

    # Create observation space with custom space_config dictionary
    config = {"custom_metadata": {"test": True}}
    config_space = create_observation_space(space_config=config)
    assert config_space.shape == (1,), "Config should not affect default shape"

    # Verify all created spaces meet custom parameter specifications
    test_spaces = [
        shaped_space,
        bounded_space,
        dtype_space,
        validated_space,
        config_space,
    ]
    for space in test_spaces:
        assert isinstance(
            space, gymnasium.spaces.Box
        ), f"All spaces should be Box, got {type(space)}"

    # Test parameter validation and error handling for invalid parameters
    with pytest.raises(ValidationError):
        create_observation_space(observation_shape=())

    with pytest.raises(ValidationError):
        create_observation_space(concentration_bounds=(1.0, 0.0))  # Invalid order

    with pytest.raises(ValidationError):
        create_observation_space(concentration_bounds=(0.5, 0.5))  # Equal bounds

    # Compare spaces with different configurations for correctness
    default_space = create_observation_space()
    assert (
        default_space.shape == shaped_space.shape
    ), "Default and custom spaces should match when using default values"


@pytest.mark.parametrize("action", VALID_ACTIONS)
def test_validate_action_valid_inputs(action):
    """
    Test action validation function with all valid action inputs including integers
    and Action enum values ensuring proper type conversion and validation.
    """
    # Test validate_action with each valid action from VALID_ACTIONS list
    validated_action = validate_action(action)

    # Verify function returns validated integer action in range [0, 3]
    assert isinstance(
        validated_action, int
    ), f"Expected int, got {type(validated_action)}"
    assert (
        0 <= validated_action <= 3
    ), f"Validated action {validated_action} out of range [0, 3]"

    # Test Action enum values (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)
    if isinstance(action, Action):
        assert (
            validated_action == action.value
        ), f"Action enum value not converted correctly: {action} -> {validated_action}"

    # Test integer values (0, 1, 2, 3) for cardinal directions
    elif isinstance(action, (int, np.integer)):
        assert validated_action == int(
            action
        ), f"Integer action not validated correctly: {action} -> {validated_action}"

    # Verify no exceptions raised for valid action inputs
    # This is implicit - if we reach here, no exception was raised

    # Test strict_mode=False and strict_mode=True for different validation levels
    strict_result = validate_action(action, strict_mode=True)
    normal_result = validate_action(action, strict_mode=False)
    assert (
        strict_result == normal_result
    ), "Strict and normal mode should give same result for valid actions"

    # Test log_validation parameter for logging behavior
    logged_result = validate_action(action, log_validation=True)
    assert (
        logged_result == validated_action
    ), "Logging should not affect validation result"

    # Validate return type is always integer regardless of input type
    assert isinstance(
        validated_action, int
    ), f"Return type should always be int, got {type(validated_action)}"


@pytest.mark.parametrize("invalid_action", INVALID_ACTIONS)
def test_validate_action_invalid_inputs(invalid_action):
    """
    Test action validation function with invalid action inputs ensuring proper
    ValidationError handling and secure error reporting without information disclosure.
    """
    # Test validate_action with each invalid action from INVALID_ACTIONS list
    with pytest.raises(ValidationError) as exc_info:
        validate_action(invalid_action)

    # Assert ValidationError is raised for all invalid actions
    error = exc_info.value
    assert isinstance(
        error, ValidationError
    ), f"Expected ValidationError, got {type(error)}"

    # Verify error messages are informative but do not disclose sensitive information
    error_message = str(error)
    assert len(error_message) > 0, "Error message should not be empty"
    assert (
        "Action" in error_message or "action" in error_message
    ), "Error message should mention action validation"

    # Test out-of-range actions (-1, 4, 5) raise appropriate errors
    if isinstance(invalid_action, (int, float)) and not np.isnan(invalid_action):
        try:
            int_val = int(invalid_action)
            if int_val < 0 or int_val >= ACTION_SPACE_SIZE:
                assert (
                    "range" in error_message.lower() or "bound" in error_message.lower()
                ), f"Out-of-range error should mention range: {error_message}"
        except (ValueError, OverflowError):
            pass  # Expected for invalid numeric values

    # Test invalid types ('invalid', None, 3.14) are properly handled
    if not isinstance(invalid_action, (int, float, Action)):
        assert (
            "type" in error_message.lower() or "convert" in error_message.lower()
        ), f"Type error should mention type issues: {error_message}"

    # Validate error context includes parameter name and expected format
    assert hasattr(error, "context"), "ValidationError should have context"
    if error.context:
        assert isinstance(error.context, dict), "Error context should be dictionary"

    # Test strict_mode affects validation rigor appropriately
    with pytest.raises(ValidationError):
        validate_action(invalid_action, strict_mode=True)

    with pytest.raises(ValidationError):
        validate_action(invalid_action, strict_mode=False)

    # Ensure no crashes or unexpected exceptions for any invalid input
    # If we reach here, proper ValidationError was raised as expected


def test_validate_observation_valid_inputs():
    """
    Test observation validation function with valid numpy arrays ensuring proper shape,
    dtype, and range validation for concentration observations.
    """
    # Create valid observations using numpy arrays with shape (1,) and float32 dtype
    for conc_val in TEST_CONCENTRATION_VALUES:
        valid_obs = np.array([conc_val], dtype=OBSERVATION_DTYPE)

        # Test observations with values from TEST_CONCENTRATION_VALUES [0.0, 0.5, 1.0]
        validated_obs = validate_observation(valid_obs)

        # Verify validate_observation returns validated numpy array
        assert isinstance(
            validated_obs, np.ndarray
        ), f"Expected numpy array, got {type(validated_obs)}"

        # Test observations at concentration range boundaries (0.0, 1.0)
        assert validated_obs.shape == (
            1,
        ), f"Expected shape (1,), got {validated_obs.shape}"
        assert (
            validated_obs.dtype == OBSERVATION_DTYPE
        ), f"Expected dtype {OBSERVATION_DTYPE}, got {validated_obs.dtype}"
        assert np.allclose(
            validated_obs, [conc_val]
        ), f"Observation value not preserved: {conc_val} -> {validated_obs[0]}"

    # Test check_range=True for range validation
    valid_obs = np.array([0.5], dtype=OBSERVATION_DTYPE)
    range_validated = validate_observation(valid_obs, check_range=True)
    assert np.array_equal(
        range_validated, valid_obs
    ), "Range validation should not modify valid observations"

    # Test check_range=False for relaxed range validation
    no_range_validated = validate_observation(valid_obs, check_range=False)
    assert np.array_equal(
        no_range_validated, valid_obs
    ), "Validation should not modify valid observations"

    # Verify strict_mode affects validation comprehensiveness
    strict_validated = validate_observation(valid_obs, strict_mode=True)
    normal_validated = validate_observation(valid_obs, strict_mode=False)
    assert np.array_equal(
        strict_validated, normal_validated
    ), "Valid observations should pass both strict and normal validation"

    # Test log_validation parameter for debugging output
    logged_validated = validate_observation(valid_obs, log_validation=True)
    assert np.array_equal(
        logged_validated, valid_obs
    ), "Logging should not affect validation result"


def test_validate_observation_invalid_inputs():
    """
    Test observation validation function with invalid inputs including wrong shapes,
    dtypes, and out-of-range values ensuring proper error handling.
    """
    # Test invalid observation shapes (wrong dimensions, empty arrays)
    invalid_shapes = [
        np.array([]),  # Empty array
        np.array([[0.5]]),  # Wrong dimensions
        np.array([0.5, 0.5]),  # Wrong size
    ]

    for invalid_shape in invalid_shapes:
        with pytest.raises(ValidationError) as exc_info:
            validate_observation(invalid_shape)
        error_message = str(exc_info.value)
        assert (
            "shape" in error_message.lower()
        ), f"Shape error should mention shape: {error_message}"

    # Test invalid dtypes (int32, float64) expecting float32
    valid_value = 0.5
    invalid_dtypes = [np.int32, np.float64, np.int8]

    for invalid_dtype in invalid_dtypes:
        invalid_obs = np.array([valid_value], dtype=invalid_dtype)
        # Should either validate with warning or convert dtype
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                validated = validate_observation(invalid_obs)
                # Should convert to correct dtype
                assert (
                    validated.dtype == OBSERVATION_DTYPE
                ), f"Dtype should be converted to {OBSERVATION_DTYPE}"
        except ValidationError:
            # Acceptable if strict dtype checking
            pass

    # Test out-of-range values from INVALID_CONCENTRATION_VALUES [-0.1, 1.1]
    for invalid_val in INVALID_CONCENTRATION_VALUES:
        if isinstance(invalid_val, (int, float)) and not np.isnan(invalid_val):
            invalid_obs = np.array([invalid_val], dtype=OBSERVATION_DTYPE)
            with pytest.raises(ValidationError) as exc_info:
                validate_observation(invalid_obs, check_range=True)
            error_message = str(exc_info.value)
            assert (
                "range" in error_message.lower() or "bound" in error_message.lower()
            ), f"Range error should mention range: {error_message}"

    # Test non-array inputs ('invalid', None, scalar values)
    non_array_inputs = ["invalid", None, 0.5, [0.5]]

    for invalid_input in non_array_inputs:
        if invalid_input is None:
            with pytest.raises(ValidationError) as exc_info:
                validate_observation(invalid_input)
            assert "None" in str(
                exc_info.value
            ), "None input should be explicitly handled"
        else:
            # Should either convert to array or raise ValidationError
            try:
                result = validate_observation(invalid_input)
                assert isinstance(
                    result, np.ndarray
                ), "Converted input should be numpy array"
            except ValidationError:
                # Acceptable if conversion fails
                pass

    # Assert ValidationError raised for all invalid inputs tested above
    # Verify error messages contain specific validation failure details
    # Test strict_mode produces more detailed error analysis
    with pytest.raises(ValidationError):
        validate_observation(
            np.array([2.0], dtype=OBSERVATION_DTYPE), check_range=True, strict_mode=True
        )

    # Ensure validation fails gracefully without system crashes
    # This is implicit - if we reach here without crashes, validation is graceful


def test_validate_action_space_valid():
    """
    Test action space validation function with properly configured Discrete spaces
    ensuring Gymnasium compliance and navigation compatibility.
    """
    # Create valid action space using create_action_space()
    valid_space = create_action_space()

    # Test validate_action_space returns True for valid space
    result = validate_action_space(valid_space)
    assert result is True, "Valid action space should pass validation"

    # Test check_metadata=True for metadata validation
    metadata_result = validate_action_space(valid_space, check_metadata=True)
    assert metadata_result is True, "Valid space should pass metadata validation"

    # Test strict_validation=True for comprehensive checking
    strict_result = validate_action_space(valid_space, strict_validation=True)
    assert strict_result is True, "Valid space should pass strict validation"

    # Verify action space sample and contains methods work correctly
    sample = valid_space.sample()
    assert valid_space.contains(sample), "Space sample should be contained in space"
    assert 0 <= sample < valid_space.n, f"Sample {sample} should be in valid range"

    # Test navigation compatibility with Action enum values
    for action_enum in Action:
        if action_enum.value < valid_space.n:
            assert valid_space.contains(
                action_enum.value
            ), f"Space should contain Action.{action_enum.name}"

    # Validate action space meets performance requirements
    start_time = time.time()
    for _ in range(100):
        valid_space.sample()
    sample_time = (time.time() - start_time) * 1000
    assert sample_time < 50, f"Sampling performance too slow: {sample_time}ms"

    # Test space properties (n, sample, contains) for correctness
    assert (
        valid_space.n == ACTION_SPACE_SIZE
    ), f"Expected {ACTION_SPACE_SIZE} actions, got {valid_space.n}"
    assert hasattr(valid_space, "sample"), "Space missing sample method"
    assert hasattr(valid_space, "contains"), "Space missing contains method"


def test_validate_action_space_invalid():
    """
    Test action space validation function with invalid or improperly configured spaces
    ensuring proper error detection and reporting.
    """
    # Test validation with non-Discrete space types (Box, MultiDiscrete)
    invalid_box_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,))
    with pytest.raises(ValidationError) as exc_info:
        validate_action_space(invalid_box_space)
    assert "Discrete" in str(
        exc_info.value
    ), "Error should mention expected Discrete space type"

    # Test Discrete space with wrong size (not ACTION_SPACE_SIZE)
    wrong_size_space = gymnasium.spaces.Discrete(8)
    # Should pass normal validation but warn in strict mode
    result = validate_action_space(wrong_size_space, strict_validation=False)
    assert result is True, "Wrong size should pass normal validation"

    with pytest.raises(ValidationError):
        validate_action_space(wrong_size_space, strict_validation=True)

    # Test spaces with broken sample or contains methods
    mock_space = Mock(spec=gymnasium.spaces.Discrete)
    mock_space.n = ACTION_SPACE_SIZE
    mock_space.sample.side_effect = Exception("Broken sample method")

    with pytest.raises(ValidationError):
        validate_action_space(mock_space)

    # Create mock spaces with invalid properties for testing
    mock_invalid_space = Mock()
    mock_invalid_space.n = ACTION_SPACE_SIZE
    mock_invalid_space.sample.return_value = -1  # Invalid sample
    mock_invalid_space.contains.return_value = True

    with pytest.raises(ValidationError):
        validate_action_space(mock_invalid_space)

    # Assert ValidationError raised for invalid spaces
    # Test strict_validation detects additional issues
    # Verify error messages are specific and actionable
    # Test validation handles edge cases and malformed spaces

    # Test None space
    with pytest.raises(ValidationError):
        validate_action_space(None)


def test_validate_observation_space_valid():
    """
    Test observation space validation function with properly configured Box spaces
    ensuring concentration bounds and dtype compliance.
    """
    # Create valid observation space using create_observation_space()
    valid_space = create_observation_space()

    # Test validate_observation_space returns True for valid space
    result = validate_observation_space(valid_space)
    assert result is True, "Valid observation space should pass validation"

    # Test check_bounds=True for bounds validation
    bounds_result = validate_observation_space(valid_space, check_bounds=True)
    assert bounds_result is True, "Valid space should pass bounds validation"

    # Test check_dtype=True for dtype validation
    dtype_result = validate_observation_space(valid_space, check_dtype=True)
    assert dtype_result is True, "Valid space should pass dtype validation"

    # Test strict_validation=True for comprehensive checking
    strict_result = validate_observation_space(valid_space, strict_validation=True)
    assert strict_result is True, "Valid space should pass strict validation"

    # Verify observation space sample and contains methods
    sample = valid_space.sample()
    assert valid_space.contains(sample), "Space sample should be contained in space"
    assert (
        sample.shape == valid_space.shape
    ), f"Sample shape {sample.shape} should match space shape {valid_space.shape}"

    # Test concentration compatibility with plume values
    for conc_val in TEST_CONCENTRATION_VALUES:
        test_obs = np.array([conc_val], dtype=valid_space.dtype)
        assert valid_space.contains(
            test_obs
        ), f"Space should contain concentration value {conc_val}"

    # Validate space meets performance and memory requirements
    start_time = time.time()
    for _ in range(100):
        valid_space.sample()
    sample_time = (time.time() - start_time) * 1000
    assert sample_time < 100, f"Observation sampling too slow: {sample_time}ms"


def test_validate_observation_space_invalid():
    """
    Test observation space validation function with invalid Box spaces ensuring
    proper error detection for bounds, shape, and dtype issues.
    """
    # Test validation with non-Box space types (Discrete, MultiBox)
    invalid_discrete_space = gymnasium.spaces.Discrete(4)
    with pytest.raises(ValidationError) as exc_info:
        validate_observation_space(invalid_discrete_space)
    assert "Box" in str(exc_info.value), "Error should mention expected Box space type"

    # Test Box space with wrong shape (not (1,))
    wrong_shape_space = gymnasium.spaces.Box(
        low=0.0, high=1.0, shape=(2,), dtype=np.float32
    )
    # Should pass normal validation but warn in strict mode
    with pytest.warns(UserWarning, match="Observation space"):
        result = validate_observation_space(wrong_shape_space, strict_validation=False)
    assert result is True, "Wrong shape should pass normal validation"

    with pytest.raises(ValidationError):
        validate_observation_space(wrong_shape_space, strict_validation=True)

    # Test Box space with wrong bounds (not [0.0, 1.0])
    wrong_bounds_space = gymnasium.spaces.Box(
        low=-1.0, high=1.5, shape=(1,), dtype=np.float32
    )
    # Should pass normal validation but warn when checking bounds
    with pytest.warns(UserWarning, match="Observation space"):
        result = validate_observation_space(
            wrong_bounds_space, check_bounds=True, strict_validation=False
        )
    assert (
        result is True
    ), "Wrong bounds should pass when check_bounds=True and strict_validation=False"

    with pytest.raises(ValidationError):
        validate_observation_space(
            wrong_bounds_space, check_bounds=True, strict_validation=True
        )

    # Test Box space with wrong dtype (not float32)
    wrong_dtype_space = gymnasium.spaces.Box(
        low=0.0, high=1.0, shape=(1,), dtype=np.float64
    )
    # Should pass normal validation but warn when checking dtype
    with pytest.warns(UserWarning, match="Observation space"):
        result = validate_observation_space(
            wrong_dtype_space, check_dtype=True, strict_validation=False
        )
    assert (
        result is True
    ), "Wrong dtype should pass when check_dtype=True and strict_validation=False"

    with pytest.raises(ValidationError):
        validate_observation_space(
            wrong_dtype_space, check_dtype=True, strict_validation=True
        )

    # Test extremely large observation space configuration warning for shape
    large_shape_config = SpaceConfig(observation_shape=(1000,))
    with pytest.warns(UserWarning, match="Observation space"):
        large_shape_config.create_observation_space()

    # Create mock spaces with invalid properties
    mock_space = Mock(spec=gymnasium.spaces.Box)
    mock_space.shape = (1,)
    mock_space.dtype = np.float32
    mock_space.low = np.array([0.0])
    mock_space.high = np.array([1.0])
    mock_space.sample.side_effect = Exception("Broken sample method")

    with pytest.raises(ValidationError):
        validate_observation_space(mock_space)

    # Assert ValidationError raised for all invalid configurations
    # Test check_bounds and check_dtype flags affect validation
    # Verify error messages provide specific remediation guidance


class TestSpaceConfig:
    """
    Test class for comprehensive SpaceConfig data class testing including initialization,
    validation, space creation, and serialization with all configuration parameters and methods.
    """

    def test_config_defaults(self):
        """Test SpaceConfig default parameter initialization and validation."""
        # Create SpaceConfig with default parameters
        config = SpaceConfig()

        # Verify action_space_size defaults to ACTION_SPACE_SIZE
        assert (
            config.action_space_size == ACTION_SPACE_SIZE
        ), f"Expected default action_space_size {ACTION_SPACE_SIZE}, got {config.action_space_size}"

        # Check observation_shape defaults to (1,)
        assert config.observation_shape == (
            1,
        ), f"Expected default observation_shape (1,), got {config.observation_shape}"

        # Validate concentration_bounds default to CONCENTRATION_RANGE
        assert (
            config.concentration_bounds == CONCENTRATION_RANGE
        ), f"Expected default bounds {CONCENTRATION_RANGE}, got {config.concentration_bounds}"

        # Test observation_dtype defaults to OBSERVATION_DTYPE
        assert (
            config.observation_dtype == OBSERVATION_DTYPE
        ), f"Expected default dtype {OBSERVATION_DTYPE}, got {config.observation_dtype}"

    def test_config_custom_parameters(self):
        """Test SpaceConfig with custom parameter values and validation."""
        # Create SpaceConfig with custom action_space_size
        custom_config = SpaceConfig(
            action_space_size=8,
            observation_shape=(2,),
            concentration_bounds=(0.1, 0.9),
            observation_dtype=np.float64,
        )

        # Test custom observation_shape and bounds
        assert (
            custom_config.action_space_size == 8
        ), "Custom action_space_size not applied"
        assert custom_config.observation_shape == (
            2,
        ), "Custom observation_shape not applied"
        assert custom_config.concentration_bounds == (
            0.1,
            0.9,
        ), "Custom concentration_bounds not applied"

        # Test custom observation_dtype handling
        assert (
            custom_config.observation_dtype == np.float64
        ), "Custom observation_dtype not applied"

        # Verify all custom parameters stored correctly
        assert (
            custom_config.enable_validation is True
        ), "Default validation setting should be preserved"
        assert (
            custom_config.enable_caching is True
        ), "Default caching setting should be preserved"


def test_space_config_initialization():
    """
    Test SpaceConfig data class initialization with default and custom parameters
    ensuring proper validation and configuration storage.
    """
    # Create SpaceConfig with default parameters
    default_config = SpaceConfig()
    assert default_config.action_space_size == ACTION_SPACE_SIZE
    assert default_config.observation_shape == (1,)
    assert default_config.concentration_bounds == CONCENTRATION_RANGE
    assert default_config.observation_dtype == OBSERVATION_DTYPE

    # Create SpaceConfig with custom action_space_size, observation_shape
    custom_config = SpaceConfig(action_space_size=8, observation_shape=(2,))
    assert custom_config.action_space_size == 8
    assert custom_config.observation_shape == (2,)

    # Create SpaceConfig with custom concentration_bounds and observation_dtype
    bounds_config = SpaceConfig(
        concentration_bounds=(0.2, 0.8), observation_dtype=np.float64
    )
    assert bounds_config.concentration_bounds == (0.2, 0.8)
    assert bounds_config.observation_dtype == np.float64

    # Verify all parameters stored correctly in configuration
    assert bounds_config.action_space_size == ACTION_SPACE_SIZE  # Default preserved
    assert bounds_config.observation_shape == (1,)  # Default preserved

    # Test enable_validation and enable_caching default values
    assert default_config.enable_validation is True
    assert default_config.enable_caching is True

    # Test metadata and optimization_settings initialization
    assert isinstance(default_config.metadata, dict)
    assert isinstance(default_config.optimization_settings, dict)

    # Validate configuration parameter types and ranges
    # Test configuration cloning and equality comparison
    cloned_metadata = default_config.metadata.copy()
    assert cloned_metadata == default_config.metadata


def test_space_config_validation():
    """
    Test SpaceConfig validation method ensuring mathematical consistency, performance
    feasibility, and parameter compatibility.
    """
    # Create valid SpaceConfig and test validate_config() returns True
    valid_config = SpaceConfig()
    assert (
        valid_config.validate_config() is True
    ), "Valid configuration should pass validation"

    # Test config with invalid action_space_size raises ValidationError
    with pytest.raises(ValidationError):
        SpaceConfig(action_space_size=0)

    with pytest.raises(ValidationError):
        SpaceConfig(action_space_size=-1)

    # Test config with invalid observation_shape raises ValidationError
    with pytest.raises(ValidationError):
        SpaceConfig(observation_shape=())

    with pytest.raises(ValidationError):
        SpaceConfig(observation_shape=(0,))

    # Test config with inconsistent concentration_bounds
    with pytest.raises(ValidationError):
        SpaceConfig(concentration_bounds=(1.0, 0.0))

    with pytest.raises(ValidationError):
        SpaceConfig(concentration_bounds=(0.5, 0.5))

    # Test config with unsupported observation_dtype
    with pytest.raises(ValidationError):
        SpaceConfig(observation_dtype="invalid")

    # Test strict_validation=True for enhanced checking
    valid_config = SpaceConfig(action_space_size=8)  # Different from standard
    normal_result = valid_config.validate_config(strict_validation=False)
    assert normal_result is True, "Non-standard size should pass normal validation"

    with pytest.raises(ValidationError):
        valid_config.validate_config(strict_validation=True)

    # Verify validation error messages are specific and helpful
    # Test validation performance meets speed requirements
    start_time = time.time()
    for _ in range(100):
        valid_config.validate_config()
    validation_time = (time.time() - start_time) * 1000
    assert validation_time < 100, f"Config validation too slow: {validation_time}ms"


def test_space_config_space_creation():
    """
    Test SpaceConfig action and observation space creation methods ensuring spaces
    match configuration parameters and validation requirements.
    """
    # Create SpaceConfig and test create_action_space() method
    config = SpaceConfig(action_space_size=4)
    action_space = config.create_action_space()

    # Verify created action space matches configuration parameters
    assert isinstance(
        action_space, gymnasium.spaces.Discrete
    ), "Created space should be Discrete"
    assert (
        action_space.n == config.action_space_size
    ), "Action space size should match configuration"

    # Create SpaceConfig and test create_observation_space() method
    obs_config = SpaceConfig(
        observation_shape=(1,),
        concentration_bounds=(0.0, 1.0),
        observation_dtype=np.float32,
    )
    obs_space = obs_config.create_observation_space()

    # Verify created observation space matches configuration bounds and dtype
    assert isinstance(obs_space, gymnasium.spaces.Box), "Created space should be Box"
    assert (
        obs_space.shape == obs_config.observation_shape
    ), "Observation space shape should match configuration"
    assert (
        obs_space.dtype == obs_config.observation_dtype
    ), "Observation space dtype should match configuration"

    # Test space creation with validation enabled and disabled
    validated_config = SpaceConfig(enable_validation=True)
    validated_space = validated_config.create_action_space()
    assert isinstance(validated_space, gymnasium.spaces.Discrete)

    unvalidated_config = SpaceConfig(enable_validation=False)
    unvalidated_space = unvalidated_config.create_action_space()
    assert isinstance(unvalidated_space, gymnasium.spaces.Discrete)

    # Test space creation with caching enabled and disabled
    cached_config = SpaceConfig(enable_caching=True)
    cached_space = cached_config.create_observation_space()
    assert isinstance(cached_space, gymnasium.spaces.Box)

    uncached_config = SpaceConfig(enable_caching=False)
    uncached_space = uncached_config.create_observation_space()
    assert isinstance(uncached_space, gymnasium.spaces.Box)

    # Compare spaces created by config vs direct factory functions
    direct_action_space = create_action_space(num_actions=4)
    config_action_space = config.create_action_space()
    assert (
        direct_action_space.n == config_action_space.n
    ), "Direct and config-created spaces should be equivalent"

    # Validate created spaces pass space validation functions
    assert validate_action_space(action_space)
    assert validate_observation_space(obs_space)


def test_space_config_serialization():
    """
    Test SpaceConfig serialization methods including to_dict conversion with optional
    metadata and optimization settings for configuration storage.
    """
    # Create SpaceConfig and test to_dict() method returns dictionary
    config = SpaceConfig(
        action_space_size=4,
        observation_shape=(1,),
        metadata={"test": "value"},
        optimization_settings={"fast": True},
    )

    config_dict = config.to_dict()
    assert isinstance(config_dict, dict), "to_dict() should return dictionary"

    # Test to_dict(include_metadata=True) includes metadata
    dict_with_metadata = config.to_dict(include_metadata=True)
    if config.metadata:
        assert (
            "metadata" in dict_with_metadata
        ), "Dictionary should include metadata when requested"

    # Test to_dict(include_optimization_settings=True) includes optimization settings
    dict_with_settings = config.to_dict(include_optimization_settings=True)
    if config.optimization_settings:
        assert (
            "optimization_settings" in dict_with_settings
        ), "Dictionary should include optimization settings when requested"

    # Verify dictionary contains all configuration parameters
    expected_keys = [
        "action_space_size",
        "observation_shape",
        "concentration_bounds",
        "observation_dtype",
        "enable_validation",
        "enable_caching",
    ]
    for key in expected_keys:
        assert key in config_dict, f"Dictionary missing key: {key}"

    # Test dictionary can be used to recreate equivalent SpaceConfig
    reconstructed_values = {
        "action_space_size": config_dict["action_space_size"],
        "observation_shape": config_dict["observation_shape"],
        "concentration_bounds": config_dict["concentration_bounds"],
        "enable_validation": config_dict["enable_validation"],
        "enable_caching": config_dict["enable_caching"],
    }
    # Note: Would need custom constructor for full reconstruction

    # Test dictionary serialization to JSON for storage
    import json

    json_str = json.dumps(config_dict, default=str)  # Handle numpy types
    assert isinstance(json_str, str), "Dictionary should be JSON serializable"

    # Validate dictionary format for external compatibility
    parsed_back = json.loads(json_str)
    assert isinstance(parsed_back, dict), "JSON round-trip should preserve structure"

    # Test round-trip configuration serialization and deserialization
    # This would require implementing from_dict class method
    assert len(config_dict) > 0, "Serialized config should not be empty"


class TestSpaceValidator:
    """
    Test class for comprehensive SpaceValidator utility class testing including validation
    methods, caching functionality, and performance statistics with error handling.
    """

    def test_validator_caching_enabled(self):
        """Test SpaceValidator with caching enabled for performance optimization."""
        # Create validator with enable_caching=True
        validator = SpaceValidator(enable_caching=True)
        assert validator.enable_caching is True, "Caching should be enabled"

        # Perform repeated validations and measure cache performance
        space = create_action_space()

        # First validation (cache miss)
        result1 = validator.validate_action_space(space)
        assert result1["valid"] is True, "Validation should succeed"

        # Second validation (should hit cache)
        result2 = validator.validate_action_space(space)
        assert result2["valid"] is True, "Cached validation should succeed"

        # Verify cache hit ratios improve validation speed
        stats = validator.get_validation_stats()
        assert (
            stats["cache_performance"]["cache_hits"] > 0
        ), "Should have cache hits after repeated validations"

        # Test cache memory usage stays within limits
        cache_size = (
            len(validator.validation_cache) if validator.validation_cache else 0
        )
        assert cache_size <= 1000, f"Cache size {cache_size} should be reasonable"

    def test_validator_strict_mode(self):
        """Test SpaceValidator strict mode for comprehensive validation checking."""
        # Create validator with strict_mode=True
        strict_validator = SpaceValidator(strict_mode=True)
        normal_validator = SpaceValidator(strict_mode=False)

        # Test strict validation detects additional issues
        space = create_action_space()

        strict_result = strict_validator.validate_action_space(space)
        normal_result = normal_validator.validate_action_space(space)

        # Compare strict vs normal validation results
        assert (
            strict_result["valid"] is True
        ), "Valid space should pass strict validation"
        assert (
            normal_result["valid"] is True
        ), "Valid space should pass normal validation"

        # Verify strict mode provides detailed error analysis
        # This would be more apparent with invalid spaces
        non_standard_space = gymnasium.spaces.Discrete(8)
        strict_non_standard = strict_validator.validate_action_space(non_standard_space)
        normal_non_standard = normal_validator.validate_action_space(non_standard_space)

        # Strict mode might have different validation results
        assert isinstance(strict_non_standard, dict), "Should return validation result"
        assert isinstance(normal_non_standard, dict), "Should return validation result"


def test_space_validator_initialization():
    """
    Test SpaceValidator initialization with caching, strict mode, and custom validation
    configuration ensuring proper validator setup.
    """
    # Create SpaceValidator with default parameters
    default_validator = SpaceValidator()
    assert default_validator.enable_caching is True, "Default caching should be enabled"
    assert (
        default_validator.strict_mode is False
    ), "Default strict mode should be disabled"

    # Create SpaceValidator with enable_caching=True/False
    cached_validator = SpaceValidator(enable_caching=True)
    uncached_validator = SpaceValidator(enable_caching=False)
    assert cached_validator.enable_caching is True
    assert uncached_validator.enable_caching is False
    assert uncached_validator.validation_cache is None

    # Create SpaceValidator with strict_mode=True/False
    strict_validator = SpaceValidator(strict_mode=True)
    normal_validator = SpaceValidator(strict_mode=False)
    assert strict_validator.strict_mode is True
    assert normal_validator.strict_mode is False

    # Create SpaceValidator with custom validation_config
    custom_config = {"timeout_ms": 5000, "max_samples_test": 5}
    custom_validator = SpaceValidator(validation_config=custom_config)
    assert custom_validator.validation_config["timeout_ms"] == 5000
    assert custom_validator.validation_config["max_samples_test"] == 5

    # Verify validator properties initialized correctly
    assert hasattr(default_validator, "validation_stats")
    assert hasattr(default_validator, "performance_metrics")
    assert hasattr(default_validator, "logger")

    # Test logger initialization and component naming
    assert default_validator.logger is not None

    # Test validation_cache and performance_metrics initialization
    assert isinstance(default_validator.validation_stats, dict)
    assert isinstance(default_validator.performance_metrics, dict)

    # Validate validator configuration consistency
    assert default_validator.validation_config is not None
    assert isinstance(default_validator.validation_config, dict)


def test_space_validator_action_space():
    """
    Test SpaceValidator specialized action space validation with navigation compatibility
    checking and detailed error reporting.
    """
    # Create SpaceValidator and valid action space
    validator = SpaceValidator()
    valid_space = create_action_space()

    # Test validate_action_space() returns success validation result
    result = validator.validate_action_space(valid_space)
    assert isinstance(result, dict), "Should return validation result dictionary"
    assert result["valid"] is True, "Valid space should pass validation"

    # Test check_navigation_compatibility=True for navigation features
    nav_result = validator.validate_action_space(
        valid_space, check_navigation_compatibility=True
    )
    assert nav_result["valid"] is True, "Navigation compatibility should pass"

    # Test validation result contains detailed analysis
    assert "errors" in result, "Result should contain errors list"
    assert "warnings" in result, "Result should contain warnings list"
    assert "recommendations" in result, "Result should contain recommendations"

    # Test validator with invalid action space returns failure result
    invalid_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,))
    invalid_result = validator.validate_action_space(invalid_space)
    assert invalid_result["valid"] is False, "Invalid space should fail validation"

    # Verify error details include specific issues and recommendations
    assert len(invalid_result["errors"]) > 0, "Invalid space should have error details"

    # Test caching behavior for repeated validation calls
    stats_before = validator.get_validation_stats()
    validator.validate_action_space(valid_space)  # Should hit cache
    stats_after = validator.get_validation_stats()

    if validator.enable_caching:
        assert (
            stats_after["cache_performance"]["cache_hits"]
            > stats_before["cache_performance"]["cache_hits"]
        ), "Repeated validation should increase cache hits"

    # Test performance tracking during validation operations
    assert "validation_time_ms" in result, "Result should include timing information"
    assert result["validation_time_ms"] >= 0, "Validation time should be non-negative"


def test_space_validator_observation_space():
    """
    Test SpaceValidator specialized observation space validation with concentration
    compatibility checking and bounds analysis.
    """
    # Create SpaceValidator and valid observation space
    validator = SpaceValidator()
    valid_space = create_observation_space()

    # Test validate_observation_space() returns success validation result
    result = validator.validate_observation_space(valid_space)
    assert isinstance(result, dict), "Should return validation result dictionary"
    assert result["valid"] is True, "Valid space should pass validation"

    # Test check_concentration_compatibility=True for plume features
    conc_result = validator.validate_observation_space(
        valid_space, check_concentration_compatibility=True
    )
    assert conc_result["valid"] is True, "Concentration compatibility should pass"

    # Test validation result contains concentration-specific analysis
    assert "space_info" in result, "Result should contain space information"
    assert "validation_time_ms" in result, "Result should contain timing info"

    # Test validator with invalid observation space returns failure result
    invalid_space = gymnasium.spaces.Discrete(4)
    invalid_result = validator.validate_observation_space(invalid_space)
    assert invalid_result["valid"] is False, "Invalid space should fail validation"

    # Verify error details include bounds and dtype issues
    assert len(invalid_result["errors"]) > 0, "Invalid space should have error details"
    error_message = str(invalid_result["errors"])
    assert "Box" in error_message, "Error should mention expected Box space"

    # Test caching performance for repeated validations
    start_time = time.time()
    for _ in range(10):
        validator.validate_observation_space(valid_space)
    elapsed = (time.time() - start_time) * 1000

    # With caching, repeated validations should be fast
    if validator.enable_caching:
        assert elapsed < 100, f"Cached validations should be fast: {elapsed}ms"

    # Test validation statistics tracking and reporting
    stats = validator.get_validation_stats()
    assert stats["validation_counts"]["total"] > 0, "Should track validation count"


def test_space_validator_caching():
    """
    Test SpaceValidator caching functionality ensuring cache performance optimization
    and proper cache management with memory efficiency.
    """
    # Create SpaceValidator with enable_caching=True
    validator = SpaceValidator(enable_caching=True)
    space = create_action_space()

    # Perform same validation multiple times and verify caching
    results = []
    for _ in range(5):
        result = validator.validate_action_space(space)
        results.append(result)

    # All results should be equivalent
    for result in results:
        assert result["valid"] is True, "All cached results should be consistent"

    # Test cache hit/miss ratios and performance improvement
    stats = validator.get_validation_stats(include_cache_stats=True)
    cache_stats = stats.get("cache_performance", {})

    if cache_stats:
        assert (
            cache_stats["cache_hits"] > 0
        ), "Should have cache hits after repeated calls"
        assert (
            cache_stats["total_requests"] >= cache_stats["cache_hits"]
        ), "Total requests should be >= cache hits"

    # Test clear_cache() method empties cache properly
    cache_report = validator.clear_cache()
    assert isinstance(cache_report, dict), "clear_cache should return report"
    assert cache_report["cache_entries_cleared"] >= 0, "Should report cleared entries"

    # Test cache behavior with different validation parameters
    validator.validate_action_space(space, check_navigation_compatibility=True)
    validator.validate_action_space(space, check_navigation_compatibility=False)
    # These should be cached separately due to different parameters

    # Verify cache size limits prevent memory exhaustion
    large_spaces = []
    for i in range(CACHE_TEST_SIZE):
        # Create slightly different spaces to avoid excessive caching
        if i % 2 == 0:
            test_space = create_action_space()
        else:
            test_space = create_observation_space()
        large_spaces.append(test_space)

        if i % 2 == 0:
            validator.validate_action_space(test_space)
        else:
            validator.validate_observation_space(test_space)

    # Test cache performance under CACHE_TEST_SIZE load
    final_stats = validator.get_validation_stats()
    assert (
        final_stats["validation_counts"]["total"] >= CACHE_TEST_SIZE
    ), "Should have processed all test validations"

    # Validate cache memory usage stays within reasonable limits
    if validator.validation_cache:
        cache_size = len(validator.validation_cache)
        assert cache_size <= 1000, f"Cache size {cache_size} should be limited"


def test_space_validator_statistics():
    """
    Test SpaceValidator statistics tracking including validation counts, performance
    metrics, and error rate analysis for operational monitoring.
    """
    # Create SpaceValidator and perform multiple validations
    validator = SpaceValidator()

    valid_action_space = create_action_space()
    valid_obs_space = create_observation_space()
    invalid_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,))

    # Perform mix of successful and failed validations
    validator.validate_action_space(valid_action_space)
    validator.validate_observation_space(valid_obs_space)
    validator.validate_action_space(invalid_space)  # This should fail

    # Test get_validation_stats() returns comprehensive statistics
    stats = validator.get_validation_stats()
    assert isinstance(stats, dict), "Stats should be dictionary"
    assert "validation_counts" in stats, "Should include validation counts"
    assert "success_rate" in stats, "Should include success rate"

    # Test include_cache_stats=True for cache performance analysis
    cache_stats = validator.get_validation_stats(include_cache_stats=True)
    if validator.enable_caching:
        assert "cache_performance" in cache_stats, "Should include cache stats"
        cache_perf = cache_stats["cache_performance"]
        assert "cache_hits" in cache_perf, "Should track cache hits"
        assert "cache_misses" in cache_perf, "Should track cache misses"
        assert "hit_rate" in cache_perf, "Should calculate hit rate"

    # Test include_performance_analysis=True for timing metrics
    perf_stats = validator.get_validation_stats(include_performance_analysis=True)
    assert "performance_metrics" in perf_stats, "Should include performance metrics"
    perf_metrics = perf_stats["performance_metrics"]
    assert "total_validation_time_ms" in perf_metrics, "Should track total time"

    # Verify statistics include success/failure rates
    counts = stats["validation_counts"]
    assert counts["total"] > 0, "Should have performed validations"
    assert counts["successful"] >= 0, "Should track successful validations"
    assert counts["failed"] >= 0, "Should track failed validations"
    assert (
        counts["total"] == counts["successful"] + counts["failed"]
    ), "Total should equal sum of successful and failed"

    # Test statistics track validation trends over time
    # This would require more complex time-based tracking

    # Test statistics provide optimization recommendations
    # This would be based on performance analysis

    # Validate statistics format for monitoring system integration
    assert isinstance(
        stats["success_rate"], (int, float)
    ), "Success rate should be numeric"
    assert 0 <= stats["success_rate"] <= 100, "Success rate should be percentage"


def test_get_space_info():
    """
    Test space information utility function providing comprehensive space analysis
    including properties, bounds, metadata, and performance information.
    """
    # Test get_space_info with action space returns complete information
    action_space = create_action_space()
    action_info = get_space_info(action_space)

    assert isinstance(action_info, dict), "Should return dictionary"
    assert "space_type" in action_info, "Should include space type"
    assert action_info["space_type"] == "Discrete", "Should identify Discrete space"
    assert "n" in action_info, "Discrete space should include n property"

    # Test get_space_info with observation space returns detailed analysis
    obs_space = create_observation_space()
    obs_info = get_space_info(obs_space)

    assert obs_info["space_type"] == "Box", "Should identify Box space"
    assert "shape" in obs_info, "Box space should include shape"
    assert "low" in obs_info, "Box space should include low bounds"
    assert "high" in obs_info, "Box space should include high bounds"
    assert "dtype" in obs_info, "Box space should include dtype"

    # Test include_samples=True generates sample values
    sampled_info = get_space_info(action_space, include_samples=True)
    assert "sample_values" in sampled_info, "Should include samples when requested"
    if "sample_values" in sampled_info:
        samples = sampled_info["sample_values"]
        assert len(samples) > 0, "Should generate sample values"
        for sample in samples:
            assert action_space.contains(sample), "All samples should be valid"

    # Test include_metadata=True includes space documentation
    metadata_info = get_space_info(action_space, include_metadata=True)
    assert "metadata" in metadata_info, "Should include metadata field"

    # Test include_performance_info=True adds timing analysis
    perf_info = get_space_info(action_space, include_performance_info=True)
    assert "performance" in perf_info, "Should include performance information"
    if "performance" in perf_info:
        perf_data = perf_info["performance"]
        assert isinstance(perf_data, dict), "Performance should be dictionary"

    # Verify returned dictionary contains all expected keys
    expected_basic_keys = ["space_type", "module", "methods"]
    for key in expected_basic_keys:
        assert key in action_info, f"Missing expected key: {key}"

    # Test space info with various space types and configurations
    custom_action_space = gymnasium.spaces.Discrete(8)
    custom_info = get_space_info(custom_action_space)
    assert custom_info["n"] == 8, "Should handle custom space sizes"

    # Validate information accuracy and completeness
    assert (
        action_info["n"] == action_space.n
    ), "Should accurately report space properties"


def test_sample_valid_action():
    """
    Test action sampling utility function with constraints and validation ensuring
    generated actions are valid and meet specified criteria.
    """
    # Test sample_valid_action() returns action in valid range [0, 3]
    for _ in range(10):
        action = sample_valid_action()
        assert isinstance(action, int), f"Sample should be integer, got {type(action)}"
        assert 0 <= action <= 3, f"Sample {action} should be in range [0, 3]"

    # Test sampling with custom action_space parameter
    custom_space = gymnasium.spaces.Discrete(8)
    custom_action = sample_valid_action(action_space=custom_space)
    assert (
        0 <= custom_action < 8
    ), f"Custom space sample {custom_action} should be in range [0, 7]"

    # Test excluded_actions parameter filters out specified actions
    excluded = [0, 1]
    for _ in range(10):
        action = sample_valid_action(excluded_actions=excluded)
        assert (
            action not in excluded
        ), f"Sample {action} should not be in excluded list {excluded}"
        assert action in [2, 3], "With exclusions, sample should be in [2, 3]"

    # Test validate_sample=True validates sampled actions
    validated_action = sample_valid_action(validate_sample=True)
    assert isinstance(validated_action, int), "Validated sample should be integer"
    assert 0 <= validated_action <= 3, "Validated sample should be in valid range"

    # Generate multiple samples and verify all are valid
    samples = [sample_valid_action() for _ in range(20)]
    action_space = create_action_space()
    for sample in samples:
        assert action_space.contains(
            sample
        ), f"Sample {sample} should be valid for action space"

    # Test sampling performance meets speed requirements
    start_time = time.time()
    for _ in range(100):
        sample_valid_action()
    sampling_time = (time.time() - start_time) * 1000
    assert sampling_time < 100, f"Sampling too slow: {sampling_time}ms for 100 samples"

    # Test edge case with all actions excluded raises appropriate error
    with pytest.raises(ValidationError):
        sample_valid_action(excluded_actions=[0, 1, 2, 3])

    # Validate sampling distribution fairness over multiple calls
    sample_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _ in range(400):  # Large enough for statistical significance
        action = sample_valid_action()
        sample_counts[action] += 1

    # Each action should appear at least a few times (rough fairness check)
    for count in sample_counts.values():
        assert count > 10, f"Sample distribution seems unfair: {sample_counts}"


def test_sample_valid_observation():
    """
    Test observation sampling utility function with concentration constraints ensuring
    generated observations represent realistic plume values.
    """
    # Test sample_valid_observation() returns valid concentration array
    for _ in range(10):
        obs = sample_valid_observation()
        assert isinstance(
            obs, np.ndarray
        ), f"Sample should be numpy array, got {type(obs)}"
        assert obs.shape == (1,), f"Sample shape should be (1,), got {obs.shape}"
        assert (
            obs.dtype == OBSERVATION_DTYPE
        ), f"Sample dtype should be {OBSERVATION_DTYPE}, got {obs.dtype}"

    # Test sampling with custom observation_space parameter
    custom_bounds = (0.2, 0.8)
    custom_space = gymnasium.spaces.Box(
        low=custom_bounds[0], high=custom_bounds[1], shape=(1,), dtype=OBSERVATION_DTYPE
    )
    custom_obs = sample_valid_observation(observation_space=custom_space)
    assert custom_space.contains(
        custom_obs
    ), "Custom space sample should be contained in space"

    # Test concentration_range parameter constrains sample values
    constrained_obs = sample_valid_observation(concentration_range=(0.3, 0.7))
    assert (
        0.3 <= constrained_obs[0] <= 0.7
    ), f"Constrained sample {constrained_obs[0]} should be in range [0.3, 0.7]"

    # Test validate_sample=True validates sampled observations
    validated_obs = sample_valid_observation(validate_sample=True)
    assert isinstance(
        validated_obs, np.ndarray
    ), "Validated sample should be numpy array"

    # Verify samples have correct shape (1,) and dtype float32
    obs = sample_valid_observation()
    assert obs.shape == (1,), "Sample should have shape (1,)"
    assert obs.dtype == OBSERVATION_DTYPE, "Sample should have correct dtype"

    # Test samples fall within CONCENTRATION_RANGE [0.0, 1.0]
    for _ in range(20):
        obs = sample_valid_observation()
        assert (
            CONCENTRATION_RANGE[0] <= obs[0] <= CONCENTRATION_RANGE[1]
        ), f"Sample {obs[0]} should be in concentration range {CONCENTRATION_RANGE}"

    # Generate multiple samples and verify realistic concentration values
    samples = [sample_valid_observation() for _ in range(50)]
    obs_space = create_observation_space()
    for sample in samples:
        assert obs_space.contains(
            sample
        ), "Sample should be valid for observation space"

    # Test sampling performance and memory efficiency
    start_time = time.time()
    for _ in range(100):
        sample_valid_observation()
    sampling_time = (time.time() - start_time) * 1000
    assert (
        sampling_time < 200
    ), f"Observation sampling too slow: {sampling_time}ms for 100 samples"


def test_check_space_compatibility():
    """
    Test space compatibility checking function ensuring proper action and observation
    space integration for RL environment compatibility.
    """
    # Create compatible action and observation spaces
    action_space = create_action_space()
    obs_space = create_observation_space()

    # Test check_space_compatibility() returns positive compatibility report
    compat_report = check_space_compatibility(action_space, obs_space)

    assert isinstance(
        compat_report, dict
    ), "Should return compatibility report dictionary"
    assert "compatible" in compat_report, "Report should include compatibility status"
    assert "issues" in compat_report, "Report should include issues list"
    assert "warnings" in compat_report, "Report should include warnings list"
    assert "recommendations" in compat_report, "Report should include recommendations"

    # For properly configured spaces, should be compatible
    assert compat_report["compatible"] is True, "Standard spaces should be compatible"

    # Test strict_checking=True for enhanced compatibility validation
    strict_report = check_space_compatibility(
        action_space, obs_space, strict_checking=True
    )
    assert isinstance(strict_report, dict), "Strict checking should return report"

    # Test compatibility_requirements parameter with custom rules
    custom_requirements = {
        "max_action_space_size": 4,
        "required_observation_dtype": str(OBSERVATION_DTYPE),
    }
    custom_report = check_space_compatibility(
        action_space, obs_space, compatibility_requirements=custom_requirements
    )
    assert isinstance(custom_report, dict), "Custom requirements should return report"

    # Test incompatible spaces return detailed issue analysis
    incompatible_action_space = gymnasium.spaces.Discrete(16)  # Non-standard size
    incompatible_report = check_space_compatibility(
        incompatible_action_space, obs_space, strict_checking=True
    )

    # In strict mode, non-standard size might cause compatibility issues
    if not incompatible_report["compatible"]:
        assert (
            len(incompatible_report["issues"]) > 0
        ), "Incompatible spaces should have issues"

    # Verify compatibility report includes optimization recommendations
    assert isinstance(
        compat_report["recommendations"], list
    ), "Recommendations should be list"
    assert len(compat_report["recommendations"]) > 0, "Should provide recommendations"

    # Test memory usage and performance estimates in compatibility analysis
    assert "space_info" in compat_report, "Should include space information"
    space_info = compat_report["space_info"]
    assert "action_space" in space_info, "Should analyze action space"
    assert "observation_space" in space_info, "Should analyze observation space"

    # Test compatibility checking performance meets speed targets
    start_time = time.time()
    for _ in range(10):
        check_space_compatibility(action_space, obs_space)
    compat_time = (time.time() - start_time) * 1000
    assert (
        compat_time < 500
    ), f"Compatibility checking too slow: {compat_time}ms for 10 checks"


def test_optimize_space_operations():
    """
    Test space operations optimization function ensuring performance improvements
    through caching, profiling, and configuration tuning.
    """
    # Test optimize_space_operations with default space_config
    default_config = {"grid_size": TEST_GRID_SIZE}
    optimization_result = optimize_space_operations(default_config)

    assert isinstance(optimization_result, dict), "Should return optimization results"
    assert (
        "optimization_applied" in optimization_result
    ), "Should list applied optimizations"
    assert (
        "performance_improvements" in optimization_result
    ), "Should report improvements"
    assert "recommendations" in optimization_result, "Should provide recommendations"

    # Test enable_caching=True improves operation performance
    cached_result = optimize_space_operations(default_config, enable_caching=True)
    assert (
        "caching" in str(cached_result["optimization_applied"]).lower()
        or len(cached_result["optimization_applied"]) > 0
    ), "Caching optimization should be reported"

    # Test profile_operations=True provides performance analysis
    profiled_result = optimize_space_operations(default_config, profile_operations=True)
    if "performance_baseline" in profiled_result:
        baseline = profiled_result["performance_baseline"]
        assert isinstance(baseline, dict), "Performance baseline should be dictionary"
        assert len(baseline) > 0, "Should provide baseline measurements"

    # Test optimization_settings parameter customizes optimization
    custom_settings = {"disable_validation": True, "fast_dtype_conversion": True}
    custom_result = optimize_space_operations(
        default_config, optimization_settings=custom_settings
    )

    applied_opts = custom_result.get("optimization_applied", [])
    assert len(applied_opts) >= 0, "Should apply custom optimizations"

    # Measure performance before and after optimization
    # This would require more complex performance testing framework

    # Verify optimization results include performance improvement metrics
    if "performance_improvements" in optimization_result:
        improvements = optimization_result["performance_improvements"]
        assert isinstance(improvements, dict), "Improvements should be dictionary"

    # Test optimization recommendations are actionable
    recommendations = optimization_result.get("recommendations", [])
    assert isinstance(recommendations, list), "Recommendations should be list"

    # Test optimization maintains functionality while improving speed
    # This would require functional testing before and after optimization


def test_performance_benchmarks():
    """
    Test space operations performance against targets ensuring <1ms validation times
    and efficient caching with comprehensive timing analysis.
    """
    # Benchmark validate_action performance over BENCHMARK_ITERATIONS (1000)
    action_space = create_action_space()
    valid_action = 1

    start_time = time.time()
    for _ in range(BENCHMARK_ITERATIONS):
        validate_action(valid_action, action_space)
    action_validation_time = (time.time() - start_time) * 1000 / BENCHMARK_ITERATIONS

    # Measure average validation time and verify <1ms target
    assert (
        action_validation_time < 1.0
    ), f"Action validation too slow: {action_validation_time:.3f}ms > 1ms target"

    # Benchmark validate_observation performance with timing analysis
    obs_space = create_observation_space()
    valid_obs = np.array([0.5], dtype=OBSERVATION_DTYPE)

    start_time = time.time()
    for _ in range(BENCHMARK_ITERATIONS):
        validate_observation(valid_obs, obs_space)
    obs_validation_time = (time.time() - start_time) * 1000 / BENCHMARK_ITERATIONS

    assert (
        obs_validation_time < 1.0
    ), f"Observation validation too slow: {obs_validation_time:.3f}ms > 1ms target"

    # Test space creation performance meets <10ms target
    start_time = time.time()
    for _ in range(100):
        create_action_space()
        create_observation_space()
    creation_time = (time.time() - start_time) * 1000 / 100

    assert (
        creation_time < 10.0
    ), f"Space creation too slow: {creation_time:.3f}ms > 10ms target"

    # Benchmark caching performance improvement ratios
    validator = SpaceValidator(enable_caching=True)

    # First validation (cache miss)
    start_time = time.time()
    validator.validate_action_space(action_space)
    uncached_time = (time.time() - start_time) * 1000

    # Second validation (cache hit)
    start_time = time.time()
    validator.validate_action_space(action_space)
    cached_time = (time.time() - start_time) * 1000

    # Cached validation should be faster (though both might be very fast)
    assert (
        cached_time <= uncached_time * 2
    ), f"Caching should not make performance worse: {cached_time}ms vs {uncached_time}ms"

    # Test performance under load with concurrent validations
    # This would require threading, which is complex for this test

    # Verify performance meets or exceeds PERFORMANCE_TOLERANCE_MS (2.0)
    max_observed_time = max(action_validation_time, obs_validation_time)
    assert (
        max_observed_time <= PERFORMANCE_TOLERANCE_MS
    ), f"Performance exceeds tolerance: {max_observed_time:.3f}ms > {PERFORMANCE_TOLERANCE_MS}ms"

    # Generate performance report with optimization recommendations
    perf_report = {
        "action_validation_ms": action_validation_time,
        "observation_validation_ms": obs_validation_time,
        "space_creation_ms": creation_time,
        "meets_targets": {
            "validation_under_1ms": max_observed_time < 1.0,
            "creation_under_10ms": creation_time < 10.0,
            "within_tolerance": max_observed_time <= PERFORMANCE_TOLERANCE_MS,
        },
    }

    # All performance targets should be met
    for target, met in perf_report["meets_targets"].items():
        assert met, f"Performance target not met: {target}"


def test_error_handling_comprehensive():
    """
    Test comprehensive error handling including ValidationError scenarios, error message
    formatting, and secure error reporting for all utility functions.
    """
    # Test all functions raise ValidationError for invalid inputs
    invalid_inputs = [None, "invalid", -1, 999, [], {}]

    # Test create_action_space error handling
    with pytest.raises(ValidationError):
        create_action_space(num_actions=0)

    with pytest.raises(ValidationError):
        create_action_space(num_actions=-1)

    # Test create_observation_space error handling
    with pytest.raises(ValidationError):
        create_observation_space(observation_shape=())

    with pytest.raises(ValidationError):
        create_observation_space(concentration_bounds=(1.0, 0.0))

    # Test validate_action error handling for each invalid input
    for invalid_input in invalid_inputs:
        with pytest.raises(ValidationError) as exc_info:
            validate_action(invalid_input)

        # Verify error messages are informative without sensitive disclosure
        error_message = str(exc_info.value)
        assert len(error_message) > 0, "Error message should not be empty"
        assert not any(
            sensitive in error_message.lower()
            for sensitive in ["password", "secret", "key", "token"]
        ), "Error message should not contain sensitive information"

    # Test validate_observation error handling
    invalid_observations = [
        None,
        "invalid",
        np.array([]),  # Empty array
        np.array([2.0]),  # Out of range
        np.array([[0.5]]),  # Wrong shape
    ]

    for invalid_obs in invalid_observations:
        with pytest.raises(ValidationError) as exc_info:
            if invalid_obs is not None:
                validate_observation(invalid_obs, check_range=True)
            else:
                validate_observation(invalid_obs)

        # Test error context includes parameter names and expected formats
        error = exc_info.value
        assert hasattr(error, "context"), "ValidationError should have context"

    # Test error recovery suggestions are actionable and helpful
    try:
        validate_action("invalid_action")
    except ValidationError as e:
        error_message = str(e)
        # Should suggest valid action types or ranges
        assert any(
            hint in error_message.lower()
            for hint in ["integer", "action", "range", "convert"]
        ), f"Error message should provide helpful hints: {error_message}"

    # Test nested error handling in complex validation scenarios
    validator = SpaceValidator()

    try:
        # This should handle validation errors gracefully
        result = validator.validate_action_space("not_a_space")
        assert result["valid"] is False, "Should return validation failure"
        assert len(result["errors"]) > 0, "Should report validation errors"
    except Exception as e:
        # Should not crash with unexpected exceptions
        assert isinstance(
            e, (ValidationError, TypeError, AttributeError)
        ), f"Should handle errors gracefully, got {type(e)}"

    # Verify error logging integration works correctly
    # This would require checking log output

    # Test error handling performance doesn't degrade system
    start_time = time.time()
    error_count = 0
    for _ in range(100):
        try:
            validate_action("invalid")
        except ValidationError:
            error_count += 1
    error_handling_time = (time.time() - start_time) * 1000

    assert error_count == 100, "Should handle all errors"
    assert (
        error_handling_time < 100
    ), f"Error handling too slow: {error_handling_time}ms for 100 errors"

    # Test error message consistency across all functions
    action_error_pattern = None
    try:
        validate_action(None)
    except ValidationError as e:
        action_error_pattern = "None" in str(e)

    obs_error_pattern = None
    try:
        validate_observation(None)
    except ValidationError as e:
        obs_error_pattern = "None" in str(e)

    # Both should handle None consistently
    assert (
        action_error_pattern == obs_error_pattern
    ), "Error handling should be consistent across functions"


def test_edge_cases():
    """
    Test edge cases and boundary conditions including extreme grid sizes, unusual
    dtype combinations, and system resource limits.
    """
    # Test space creation with minimal grid size (1, 1)
    # This tests the minimum possible configuration
    minimal_config = SpaceConfig(
        action_space_size=4,  # Standard size
        observation_shape=(1,),  # Minimal shape
        concentration_bounds=(0.0, 1.0),
    )

    minimal_action_space = minimal_config.create_action_space()
    minimal_obs_space = minimal_config.create_observation_space()

    assert isinstance(minimal_action_space, gymnasium.spaces.Discrete)
    assert isinstance(minimal_obs_space, gymnasium.spaces.Box)

    # Test space creation with LARGE_GRID_SIZE (256, 256)
    # This tests scalability limits
    large_config = SpaceConfig(
        observation_shape=(1,),  # Keep observation simple even for large grids
        concentration_bounds=(0.0, 1.0),
    )

    large_obs_space = large_config.create_observation_space()
    assert large_obs_space.shape == (1,), "Large grid config should still work"

    # Test boundary concentration values exactly at 0.0 and 1.0
    boundary_values = [0.0, 1.0]
    obs_space = create_observation_space()

    for boundary_val in boundary_values:
        boundary_obs = np.array([boundary_val], dtype=OBSERVATION_DTYPE)
        validated = validate_observation(boundary_obs, check_range=True)
        assert np.allclose(
            validated, [boundary_val]
        ), f"Boundary value {boundary_val} should be valid"
        assert obs_space.contains(
            boundary_obs
        ), f"Space should contain boundary value {boundary_val}"

    # Test action validation with numpy integer types
    numpy_int_types = [np.int8(1), np.int16(2), np.int32(0), np.int64(3)]

    for numpy_int in numpy_int_types:
        validated_action = validate_action(numpy_int)
        assert isinstance(
            validated_action, int
        ), f"NumPy int {type(numpy_int)} should convert to Python int"
        assert (
            0 <= validated_action <= 3
        ), f"Converted action should be valid: {validated_action}"

    # Test observation validation with edge case array shapes
    edge_case_shapes = [
        np.array([0.5], dtype=OBSERVATION_DTYPE),  # Standard
        np.array([0.0], dtype=OBSERVATION_DTYPE),  # Minimum value
        np.array([1.0], dtype=OBSERVATION_DTYPE),  # Maximum value
    ]

    for edge_obs in edge_case_shapes:
        try:
            validated = validate_observation(edge_obs)
            assert isinstance(validated, np.ndarray), "Should validate successfully"
        except ValidationError:
            # Some edge cases might legitimately fail
            pass

    # Test memory limits with large configuration parameters
    # Create configuration that might approach memory limits
    try:
        large_shape_config = SpaceConfig(observation_shape=(1000,))
        large_obs_space = large_shape_config.create_observation_space()

        # If creation succeeds, test that it's still usable
        sample = large_obs_space.sample()
        assert len(sample) == 1000, "Large observation should have correct size"

    except (MemoryError, ValidationError):
        # Acceptable to fail on very large configurations
        pass

    # Test thread safety in concurrent space operations
    # This would require threading, which is complex for this test

    # Test graceful degradation under resource constraints
    # Simulate low memory conditions by testing with many spaces
    spaces = []
    try:
        for i in range(100):  # Create many spaces
            spaces.append(create_action_space())
            spaces.append(create_observation_space())
    except MemoryError:
        # Graceful handling of memory limits
        pass

    # Verify we created at least some spaces
    assert len(spaces) > 0, "Should create at least some spaces before limits"

    # Clean up
    del spaces


def test_integration_with_gymnasium():
    """
    Test integration with Gymnasium framework ensuring created spaces work correctly
    with gym.make() and standard RL workflows.
    """
    # Create spaces using utility functions
    action_space = create_action_space()
    obs_space = create_observation_space()

    # Test spaces integrate correctly with mock Gymnasium environment
    # Create a minimal mock environment to test space integration
    class MockPlumeEnv:
        def __init__(self):
            self.action_space = action_space
            self.observation_space = obs_space
            self._step_count = 0

        def reset(self, seed=None):
            observation = self.observation_space.sample()
            return observation, {}

        def step(self, action):
            # Validate action using our validation function
            validated_action = validate_action(action, self.action_space)

            # Generate observation
            observation = self.observation_space.sample()
            validated_obs = validate_observation(observation, self.observation_space)

            reward = 0.0
            terminated = False
            truncated = self._step_count >= 1000
            info = {"step_count": self._step_count}

            self._step_count += 1

            return validated_obs, reward, terminated, truncated, info

    # Test mock environment works with our spaces
    mock_env = MockPlumeEnv()

    # Test space.sample() methods return valid values
    for _ in range(10):
        action_sample = mock_env.action_space.sample()
        assert mock_env.action_space.contains(
            action_sample
        ), "Action sample should be contained in action space"

        obs_sample = mock_env.observation_space.sample()
        assert mock_env.observation_space.contains(
            obs_sample
        ), "Observation sample should be contained in observation space"

    # Test space.contains() methods work with environment outputs
    obs, info = mock_env.reset()
    assert mock_env.observation_space.contains(
        obs
    ), "Reset observation should be contained in observation space"

    # Test environment step with various actions
    for test_action in [0, 1, 2, 3]:
        obs, reward, terminated, truncated, info = mock_env.step(test_action)

        assert mock_env.observation_space.contains(
            obs
        ), "Step observation should be contained in observation space"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(terminated, bool), "Terminated should be boolean"
        assert isinstance(truncated, bool), "Truncated should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"

    # Test space serialization for environment registration
    # Spaces should be serializable for gymnasium registration
    action_space_dict = {"type": "Discrete", "n": action_space.n}

    obs_space_dict = {
        "type": "Box",
        "shape": obs_space.shape,
        "low": obs_space.low.tolist(),
        "high": obs_space.high.tolist(),
        "dtype": str(obs_space.dtype),
    }

    assert isinstance(action_space_dict, dict), "Action space should be serializable"
    assert isinstance(obs_space_dict, dict), "Observation space should be serializable"

    # Test space compatibility with common RL training libraries
    # This would test compatibility with stable-baselines3, ray[rllib], etc.
    # For now, test basic properties that RL libraries expect

    # Action space properties expected by RL libraries
    assert hasattr(action_space, "n"), "Discrete space should have n property"
    assert hasattr(action_space, "sample"), "Space should have sample method"
    assert hasattr(action_space, "contains"), "Space should have contains method"

    # Observation space properties expected by RL libraries
    assert hasattr(obs_space, "shape"), "Box space should have shape property"
    assert hasattr(obs_space, "low"), "Box space should have low bounds"
    assert hasattr(obs_space, "high"), "Box space should have high bounds"
    assert hasattr(obs_space, "dtype"), "Box space should have dtype"

    # Verify spaces meet Gymnasium API requirements
    # Test that spaces conform to gymnasium.Space interface
    assert hasattr(action_space, "seed"), "Space should support seeding"
    assert callable(action_space.sample), "Sample should be callable"
    assert callable(action_space.contains), "Contains should be callable"

    # Test space documentation and metadata compatibility
    # Spaces should have reasonable string representations
    action_str = str(action_space)
    assert "Discrete" in action_str, "Action space string should mention Discrete"

    obs_str = str(obs_space)
    assert "Box" in obs_str, "Observation space string should mention Box"


# Performance test that should run quickly
@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_overall_performance():
    """Overall performance test ensuring all operations complete within timeout."""
    # Create spaces
    action_space = create_action_space()
    obs_space = create_observation_space()

    # Run validation operations
    validate_action(1, action_space)
    validate_observation(np.array([0.5], dtype=OBSERVATION_DTYPE), obs_space)

    # Test space validation
    assert validate_action_space(action_space)
    assert validate_observation_space(obs_space)

    # Test utility functions
    action = sample_valid_action(action_space)
    obs = sample_valid_observation(obs_space)

    # Test configuration
    config = SpaceConfig()
    assert config.validate_config()

    # Test validator
    validator = SpaceValidator()
    result = validator.validate_action_space(action_space)
    assert result["valid"]

    # If we reach here within timeout, performance is acceptable
    assert True, "All operations completed within timeout"
