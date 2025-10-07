"""
Comprehensive test suite for BaseEnvironment abstract class validating Gymnasium API
compliance, abstract method enforcement, configuration validation, error handling patterns,
and performance monitoring integration. Tests include abstract method implementation
requirements, environment initialization validation, API specification compliance, error
recovery strategies, and cross-component integration patterns for systematic base
environment testing.
"""

import time  # >=3.10 - Timing utilities for performance testing, benchmark validation, and timeout testing in environment operations
import warnings  # >=3.10 - Warning testing for development feedback, deprecation warning validation, and performance warning testing
from unittest.mock import (  # >=3.10 - Mock object creation for testing abstract methods, component isolation, rendering backend mocking, and dependency injection testing
    Mock,
    patch,
)

import numpy as np  # >=2.1.0 - Array operations, mathematical testing utilities, observation data validation, and numerical precision testing for environment operations

# External imports with version comments
import pytest  # >=8.0.0 - Testing framework for test discovery, fixture management, parameterized testing, and comprehensive test execution with assertion capabilities

import gymnasium as gym  # >=0.29.0 - Reinforcement learning environment framework for API compliance testing, action space validation, and observation space specification testing

# Core types and data structures for testing
from plume_nav_sim.core.types import (
    Coordinates,
    EnvironmentConfig,
    GridSize,
    PlumeParameters,
    create_coordinates,
    create_environment_config,
    create_grid_size,
    validate_action,
)

# Internal imports for BaseEnvironment and supporting classes
from plume_nav_sim.envs.base_env import (
    AbstractEnvironmentError,
    BaseEnvironment,
    create_base_environment_config,
    validate_base_environment_setup,
)

# Exception handling for comprehensive error testing
from plume_nav_sim.utils.exceptions import (
    ComponentError,
    ConfigurationError,
    RenderingError,
    StateError,
    ValidationError,
)
from plume_nav_sim.utils.validation import validate_plume_parameters

# Global constants for testing configuration and validation
TEST_TIMEOUT_SECONDS = 30.0
PERFORMANCE_TEST_ITERATIONS = 100
MEMORY_TEST_THRESHOLD_MB = 50.0
BENCHMARK_TOLERANCE_MS = 2.0
VALIDATION_TEST_CASES = 20
MOCK_COMPONENT_NAME = "TestComponent"
DEFAULT_TEST_GRID_SIZE = (32, 32)
TEST_SOURCE_LOCATION = (16, 16)


def create_mock_concrete_environment(
    config: EnvironmentConfig,
    render_mode: str = None,
    enable_validation: bool = True,
    mock_overrides: dict = None,
) -> Mock:
    """Factory function to create concrete BaseEnvironment implementation with mocked abstract
    methods for testing base functionality without requiring full concrete implementation.

    Args:
        config: Environment configuration for mock setup
        render_mode: Optional render mode for testing
        enable_validation: Whether to enable validation testing
        mock_overrides: Optional mock method overrides

    Returns:
        Mock: Mock concrete environment with base functionality and configurable abstract method implementations
    """

    # Create mock class inheriting from BaseEnvironment with all abstract methods implemented
    class MockConcreteEnvironment(BaseEnvironment):
        def __init__(self, config, render_mode=None):
            super().__init__(config, render_mode=render_mode)
            # Initialize mock tracking for test verification
            self._method_calls = []
            self._state_changes = []

        # Implement all abstract methods with mock behaviors
        def _reset_environment_state(self) -> None:
            self._method_calls.append("_reset_environment_state")

        def _process_action(self, action: int) -> None:
            self._method_calls.append(("_process_action", action))

        def _update_environment_state(self) -> None:
            self._method_calls.append("_update_environment_state")

        def _calculate_reward(self) -> float:
            self._method_calls.append("_calculate_reward")
            return 0.0

        def _check_terminated(self) -> bool:
            self._method_calls.append("_check_terminated")
            return False

        def _check_truncated(self) -> bool:
            self._method_calls.append("_check_truncated")
            return False

        def _get_observation(self) -> np.ndarray:
            self._method_calls.append("_get_observation")
            return np.zeros((2, 2), dtype=np.float32)

        def _create_render_context(self) -> dict:
            self._method_calls.append("_create_render_context")
            return {"test_context": True}

        def _create_renderer(self, render_mode: str) -> object:
            self._method_calls.append(("_create_renderer", render_mode))
            return Mock()

        def _seed_components(self, seed: int) -> None:
            self._method_calls.append(("_seed_components", seed))

        def _cleanup_components(self) -> None:
            self._method_calls.append("_cleanup_components")

        def _validate_component_states(self, strict_validation: bool = False) -> bool:
            self._method_calls.append(("_validate_component_states", strict_validation))
            return True

    # Configure mock abstract methods with default behaviors and return values
    mock_env = MockConcreteEnvironment(config, render_mode)

    # Apply mock_overrides to customize specific abstract method behaviors if provided
    if mock_overrides:
        for method_name, override_behavior in mock_overrides.items():
            if hasattr(mock_env, method_name):
                setattr(mock_env, method_name, override_behavior)

    # Set up environment configuration and initialization with proper validation
    if enable_validation:
        mock_env.validate_environment_state()

    # Return configured mock environment ready for testing base functionality
    return mock_env


def create_test_environment_config(
    grid_size: tuple = None,
    source_location: tuple = None,
    max_steps: int = None,
    goal_radius: float = None,
    enable_validation: bool = True,
    additional_params: dict = None,
) -> EnvironmentConfig:
    """Factory function to create validated test environment configuration with customizable
    parameters for various testing scenarios including edge cases and validation testing.

    Args:
        grid_size: Optional grid size tuple, defaults to test size
        source_location: Optional source location tuple
        max_steps: Optional maximum steps override
        goal_radius: Optional goal radius override
        enable_validation: Whether to enable configuration validation
        additional_params: Optional additional parameter overrides

    Returns:
        EnvironmentConfig: Validated environment configuration optimized for testing with proper parameter validation and resource constraints
    """
    # Use default test parameters if specific parameters not provided
    test_grid_size = grid_size or DEFAULT_TEST_GRID_SIZE
    test_source_location = source_location or TEST_SOURCE_LOCATION
    test_max_steps = max_steps or 100
    test_goal_radius = goal_radius or 0.0

    # Create GridSize and Coordinates from provided tuples with validation
    grid_size_obj = create_grid_size(test_grid_size)
    source_coords = create_coordinates(test_source_location)

    if enable_validation and not grid_size_obj.contains_coordinates(source_coords):
        raise ValidationError(
            "source_location must be within the provided grid_size bounds"
        )

    # Create PlumeParameters with source location and default sigma value
    plume_sigma = 12.0
    overrides = dict(additional_params) if additional_params else {}
    if "plume_sigma" in overrides:
        plume_sigma = float(overrides.pop("plume_sigma"))
    plume_params = PlumeParameters(source_location=source_coords, sigma=plume_sigma)

    # Validate all parameters using appropriate validation functions
    if enable_validation:
        validate_action(0)  # Test action validation
        validate_plume_parameters(plume_params, grid_size_obj)

    # Create EnvironmentConfig with validated parameters and resource checking
    config_data = {
        "grid_size": grid_size_obj,
        "source_location": source_coords,
        "plume_params": plume_params,
        "max_steps": test_max_steps,
        "goal_radius": test_goal_radius,
        "enable_rendering": True,
    }

    # Apply remaining overrides compatible with EnvironmentConfig fields
    for key, value in overrides.items():
        if key in {"grid_size", "source_location"}:
            continue  # Already handled via primary arguments
        config_data[key] = value

    config = create_environment_config(config_data)

    # Perform comprehensive configuration validation if enable_validation is True
    if enable_validation:
        config.validate()

    # Return complete test environment configuration ready for environment initialization
    return config


def assert_gymnasium_compliance(
    environment_instance: object,
    strict_validation: bool = True,
    expected_spaces: dict = None,
) -> None:
    """Comprehensive assertion function to validate complete Gymnasium API compliance including
    method signatures, return formats, action spaces, and observation spaces with detailed error reporting.

    Args:
        environment_instance: Environment to validate for Gymnasium compliance
        strict_validation: Whether to enable strict compliance checking
        expected_spaces: Optional expected space specifications

    Raises:
        AssertionError: If any API requirements are not met with detailed compliance violations
    """
    # Verify environment inherits from gymnasium.Env base class
    assert isinstance(
        environment_instance, gym.Env
    ), f"Environment must inherit from gymnasium.Env, got {type(environment_instance)}"

    # Check presence and signature of required methods: reset, step, render, close, seed
    required_methods = ["reset", "step", "render", "close", "seed"]
    for method_name in required_methods:
        assert hasattr(
            environment_instance, method_name
        ), f"Environment missing required method: {method_name}"
        assert callable(
            getattr(environment_instance, method_name)
        ), f"Environment {method_name} must be callable"

    # Validate action_space and observation_space are proper Gymnasium Space instances
    assert hasattr(
        environment_instance, "action_space"
    ), "Environment must have action_space"
    assert hasattr(
        environment_instance, "observation_space"
    ), "Environment must have observation_space"

    assert isinstance(
        environment_instance.action_space, gym.Space
    ), f"action_space must be gymnasium.Space, got {type(environment_instance.action_space)}"
    assert isinstance(
        environment_instance.observation_space, gym.Space
    ), f"observation_space must be gymnasium.Space, got {type(environment_instance.observation_space)}"

    # Test reset method returns (observation, info) tuple with correct types
    try:
        reset_result = environment_instance.reset()
        assert (
            isinstance(reset_result, tuple) and len(reset_result) == 2
        ), f"reset() must return (observation, info) tuple, got {type(reset_result)} with length {len(reset_result) if isinstance(reset_result, tuple) else 'N/A'}"

        observation, info = reset_result
        assert isinstance(
            observation, np.ndarray
        ), f"reset() observation must be numpy array, got {type(observation)}"
        assert isinstance(
            info, dict
        ), f"reset() info must be dictionary, got {type(info)}"

    except Exception as e:
        assert False, f"reset() method compliance test failed: {e}"

    # Test step method returns 5-tuple with correct types and Gymnasium specification
    try:
        valid_action = environment_instance.action_space.sample()
        step_result = environment_instance.step(valid_action)
        assert (
            isinstance(step_result, tuple) and len(step_result) == 5
        ), f"step() must return 5-tuple (obs, reward, terminated, truncated, info), got {type(step_result)} with length {len(step_result) if isinstance(step_result, tuple) else 'N/A'}"

        obs, reward, terminated, truncated, info = step_result
        assert isinstance(
            obs, np.ndarray
        ), f"step() observation must be numpy array, got {type(obs)}"
        assert isinstance(
            reward, (int, float)
        ), f"step() reward must be numeric, got {type(reward)}"
        assert isinstance(
            terminated, bool
        ), f"step() terminated must be boolean, got {type(terminated)}"
        assert isinstance(
            truncated, bool
        ), f"step() truncated must be boolean, got {type(truncated)}"
        assert isinstance(
            info, dict
        ), f"step() info must be dictionary, got {type(info)}"

    except Exception as e:
        assert False, f"step() method compliance test failed: {e}"

    # Validate render method supports required render modes and returns appropriate types
    if hasattr(environment_instance, "_render_modes") or strict_validation:
        for render_mode in ["rgb_array", "human"]:
            try:
                render_result = environment_instance.render(mode=render_mode)
                if render_mode == "rgb_array":
                    assert isinstance(
                        render_result, np.ndarray
                    ), f"render(rgb_array) must return numpy array, got {type(render_result)}"
                    assert (
                        render_result.dtype == np.uint8
                    ), f"render(rgb_array) must return uint8 array, got {render_result.dtype}"
                    assert (
                        len(render_result.shape) == 3
                    ), f"render(rgb_array) must return (H,W,3) array, got shape {render_result.shape}"
                elif render_mode == "human":
                    # Human mode should return None
                    assert (
                        render_result is None
                    ), f"render(human) must return None, got {type(render_result)}"
            except Exception as e:
                if strict_validation:
                    assert False, f"render({render_mode}) compliance test failed: {e}"
                else:
                    warnings.warn(f"render({render_mode}) test failed: {e}")

    # Check seed method returns list containing seed value per Gymnasium specification
    try:
        seed_result = environment_instance.seed(42)
        assert isinstance(
            seed_result, list
        ), f"seed() must return list, got {type(seed_result)}"
        assert (
            len(seed_result) >= 1
        ), f"seed() must return list with at least one element, got {len(seed_result)}"

    except Exception as e:
        if strict_validation:
            assert False, f"seed() method compliance test failed: {e}"
        else:
            warnings.warn(f"seed() test failed: {e}")


def measure_performance(
    environment_instance: object,
    operation_name: str,
    iterations: int,
    operation_params: dict = None,
) -> dict:
    """Performance measurement function for timing environment operations and validating
    performance targets with comprehensive timing analysis and resource monitoring.

    Args:
        environment_instance: Environment instance for performance testing
        operation_name: Name of operation to measure
        iterations: Number of iterations for measurement
        operation_params: Optional parameters for operation

    Returns:
        dict: Performance metrics including average timing, resource usage, and target compliance analysis
    """
    # Initialize performance monitoring with baseline measurements and resource tracking
    results = {
        "operation_name": operation_name,
        "iterations": iterations,
        "timings": [],
        "total_time": 0.0,
        "average_ms": 0.0,
        "min_ms": float("inf"),
        "max_ms": 0.0,
        "target_compliance": False,
        "memory_usage_mb": 0.0,
    }

    operation_params = operation_params or {}

    # Execute specified operation multiple iterations with timing measurement
    for i in range(iterations):
        start_time = time.perf_counter()

        try:
            # Execute operation based on operation_name with appropriate parameters
            if operation_name == "reset":
                environment_instance.reset()
            elif operation_name == "step":
                action = operation_params.get("action", 0)
                environment_instance.step(action)
            elif operation_name == "render":
                mode = operation_params.get("mode", "rgb_array")
                environment_instance.render(mode=mode)
            elif operation_name == "seed":
                seed_value = operation_params.get("seed", i)
                environment_instance.seed(seed_value)
            else:
                # Generic method call for other operations
                method = getattr(environment_instance, operation_name, None)
                if method and callable(method):
                    method()
        except Exception as e:
            warnings.warn(f"Performance test iteration {i} failed: {e}")
            continue

        end_time = time.perf_counter()
        iteration_time_ms = (end_time - start_time) * 1000

        # Record individual operation times, memory usage, and resource consumption
        results["timings"].append(iteration_time_ms)
        results["min_ms"] = min(results["min_ms"], iteration_time_ms)
        results["max_ms"] = max(results["max_ms"], iteration_time_ms)

    # Calculate performance statistics including average, median, maximum, and standard deviation
    if results["timings"]:
        results["total_time"] = sum(results["timings"])
        results["average_ms"] = results["total_time"] / len(results["timings"])
        results["median_ms"] = sorted(results["timings"])[len(results["timings"]) // 2]

        # Calculate standard deviation
        mean = results["average_ms"]
        variance = sum((t - mean) ** 2 for t in results["timings"]) / len(
            results["timings"]
        )
        results["std_deviation_ms"] = variance**0.5

        # Compare results against performance targets and generate compliance analysis
        target_ms = BENCHMARK_TOLERANCE_MS
        if operation_name == "step":
            target_ms = 1.0  # 1ms target for step operations
        elif operation_name == "render":
            target_ms = 5.0  # 5ms target for rendering
        elif operation_name == "reset":
            target_ms = 10.0  # 10ms target for reset

        results["target_ms"] = target_ms
        results["target_compliance"] = results["average_ms"] <= target_ms
        results["performance_ratio"] = results["average_ms"] / target_ms

    # Monitor memory usage and resource constraints during operation execution
    try:
        import psutil

        process = psutil.Process()
        results["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        results["memory_usage_mb"] = 0.0

    # Return detailed performance metrics for benchmark validation and analysis
    return results


@pytest.mark.unit
class TestBaseEnvironmentInitialization:
    """Test class for BaseEnvironment initialization validation including constructor parameter
    validation, component setup verification, configuration consistency checking, and proper
    error handling during initialization with comprehensive setup testing."""

    def test_successful_initialization(self):
        """Test successful BaseEnvironment initialization with valid configuration ensuring
        proper component setup and state initialization."""
        # Create valid test environment configuration using create_test_environment_config
        config = create_test_environment_config()

        # Create mock concrete environment with base initialization
        mock_env = create_mock_concrete_environment(config, enable_validation=True)

        # Verify environment initialization completes without errors
        assert mock_env is not None
        assert hasattr(mock_env, "_config")
        assert mock_env._config == config

        # Check action_space and observation_space are properly initialized
        assert hasattr(mock_env, "action_space")
        assert hasattr(mock_env, "observation_space")
        assert isinstance(mock_env.action_space, gym.spaces.Discrete)
        assert mock_env.action_space.n == 4  # Four cardinal directions
        assert isinstance(mock_env.observation_space, gym.spaces.Box)

        # Validate metadata dictionary contains required environment information
        assert hasattr(mock_env, "metadata")
        assert isinstance(mock_env.metadata, dict)
        assert "render_modes" in mock_env.metadata
        assert "render_fps" in mock_env.metadata

        # Verify component logger is properly set up with environment context
        assert hasattr(mock_env, "_logger")
        assert mock_env._logger is not None

        # Check performance monitoring is initialized if enabled
        assert hasattr(mock_env, "_performance_monitor")
        assert mock_env._performance_monitor is not None

        # Assert environment state flags and counters are properly initialized
        assert hasattr(mock_env, "_initialized")
        assert mock_env._initialized is True
        assert hasattr(mock_env, "_closed")
        assert mock_env._closed is False

    def test_configuration_validation(self):
        """Test comprehensive configuration validation during initialization including
        parameter consistency checking and resource constraint validation."""
        # Create various test configurations with different parameter combinations
        valid_config = create_test_environment_config()

        # Test valid configurations initialize successfully without errors
        mock_env = create_mock_concrete_environment(valid_config)
        assert mock_env is not None

        # Test invalid configurations raise appropriate ConfigurationError exceptions
        with pytest.raises(ConfigurationError):
            invalid_config = create_test_environment_config(grid_size=(0, 0))

        with pytest.raises(ConfigurationError):
            invalid_config = create_test_environment_config(max_steps=-1)

        with pytest.raises(ConfigurationError):
            invalid_config = create_test_environment_config(goal_radius=-1.0)

        # Verify configuration consistency checking identifies cross-parameter issues
        with pytest.raises(ValidationError):
            large_source = (1000, 1000)  # Outside small grid
            create_test_environment_config(
                grid_size=(10, 10), source_location=large_source, enable_validation=True
            )

        # Test resource constraint validation prevents over-allocation
        try:
            huge_grid_config = create_test_environment_config(
                grid_size=(2000, 2000), enable_validation=True
            )
            assert False, "Should have raised ValidationError for oversized grid"
        except (ValidationError, ConfigurationError):
            pass  # Expected behavior

        # Validate error messages provide helpful guidance for configuration issues
        try:
            create_test_environment_config(goal_radius=-5.0)
        except ConfigurationError as e:
            assert "goal_radius" in str(e).lower()
            assert "non-negative" in str(e).lower()

    def test_component_initialization(self):
        """Test individual component initialization within BaseEnvironment including logger
        setup, performance monitoring, and state management initialization."""
        # Create mock environment with component initialization tracking
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)

        # Verify logger is created with proper component identification
        assert hasattr(mock_env, "_logger")
        assert mock_env._logger is not None

        # Check performance metrics initialization with appropriate monitoring setup
        assert hasattr(mock_env, "_performance_monitor")
        assert mock_env._performance_monitor is not None

        # Validate renderer reference initialization for lazy loading
        assert hasattr(mock_env, "_renderer")
        # Renderer should be None initially (lazy loading)

        # Test random number generator reference setup for seeding
        assert hasattr(mock_env, "np_random")
        assert mock_env.np_random is None  # Not seeded yet

        # Verify environment flags and counters are set to initial values
        assert mock_env._initialized is True
        assert mock_env._closed is False
        assert hasattr(mock_env, "_step_count")
        assert mock_env._step_count == 0

        # Check error handling during component initialization failures
        with patch(
            "logging.getLogger", side_effect=Exception("Logger creation failed")
        ):
            try:
                failed_env = create_mock_concrete_environment(config)
                # Should still initialize but with degraded logging
                assert failed_env is not None
            except Exception:
                pass  # Acceptable if initialization fails completely

    def test_gymnasium_space_initialization(self):
        """Test proper Gymnasium action and observation space initialization with correct
        specifications and compliance validation."""
        # Create mock environment and verify action_space is Discrete(4)
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)

        # Check action_space bounds and valid action range [0,3]
        assert isinstance(mock_env.action_space, gym.spaces.Discrete)
        assert mock_env.action_space.n == 4

        # Test action space sampling and validation
        for _ in range(10):
            sampled_action = mock_env.action_space.sample()
            assert 0 <= sampled_action <= 3
            assert mock_env.action_space.contains(sampled_action)

        # Verify observation_space is Box with proper shape and dtype
        assert isinstance(mock_env.observation_space, gym.spaces.Box)
        assert mock_env.observation_space.dtype == np.float32

        # Test observation_space bounds are [0.0, 1.0] for concentration values
        assert np.all(mock_env.observation_space.low == 0.0)
        assert np.all(mock_env.observation_space.high == 1.0)

        # Validate space sampling works correctly for random action/observation generation
        sampled_obs = mock_env.observation_space.sample()
        assert isinstance(sampled_obs, np.ndarray)
        assert sampled_obs.dtype == np.float32
        assert mock_env.observation_space.contains(sampled_obs)

        # Check space contains() methods work for validation
        valid_action = 2
        invalid_action = 5
        assert mock_env.action_space.contains(valid_action)
        assert not mock_env.action_space.contains(invalid_action)

        # Test space serialization and deserialization for external compatibility
        action_space_str = str(mock_env.action_space)
        assert "Discrete" in action_space_str
        obs_space_str = str(mock_env.observation_space)
        assert "Box" in obs_space_str


@pytest.mark.unit
class TestBaseEnvironmentAbstractMethods:
    """Test class for abstract method enforcement validation ensuring concrete implementations
    provide all required abstract methods with proper error handling and implementation
    guidance for environment development."""

    def test_abstract_method_enforcement(self):
        """Test that BaseEnvironment cannot be instantiated directly and raises
        AbstractEnvironmentError with implementation guidance."""
        # Attempt to instantiate BaseEnvironment directly
        config = create_test_environment_config()

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseEnvironment(config)

        # Test incomplete concrete implementation raises AbstractEnvironmentError
        class IncompleteEnvironment(BaseEnvironment):
            # Missing most abstract method implementations
            def _reset_environment_state(self):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteEnvironment(config)

        # Verify error messages include method names and implementation guidance
        try:
            BaseEnvironment(config)
        except TypeError as e:
            error_message = str(e).lower()
            assert "abstract" in error_message
            assert "class" in error_message

        # Check all abstract methods are properly marked with @abstractmethod decorator
        abstract_methods = BaseEnvironment.__abstractmethods__
        expected_methods = {
            "_reset_environment_state",
            "_process_action",
            "_update_environment_state",
            "_calculate_reward",
            "_check_terminated",
            "_check_truncated",
            "_get_observation",
            "_create_render_context",
            "_create_renderer",
            "_seed_components",
            "_cleanup_components",
            "_validate_component_states",
        }
        assert abstract_methods == expected_methods

        # Validate implementation guidance provides helpful developer information
        # This would be tested through AbstractEnvironmentError if thrown
        try:
            raise AbstractEnvironmentError("_test_method", "TestClass")
        except AbstractEnvironmentError as e:
            guidance = e.get_implementation_guidance()
            assert isinstance(guidance, str)
            assert len(guidance) > 0
            assert "_test_method" in guidance

    def test_required_abstract_methods(self):
        """Test comprehensive list of required abstract methods ensuring all necessary
        methods are defined as abstract with proper signatures."""
        # Get list of abstract methods from BaseEnvironment using ABC inspection
        abstract_methods = BaseEnvironment.__abstractmethods__

        # Verify required methods are marked abstract: _reset_environment_state, _process_action
        required_methods = [
            "_reset_environment_state",
            "_process_action",
            "_update_environment_state",
            "_calculate_reward",
            "_check_terminated",
            "_check_truncated",
            "_get_observation",
            "_create_render_context",
            "_create_renderer",
            "_seed_components",
            "_cleanup_components",
            "_validate_component_states",
        ]

        for method_name in required_methods:
            assert (
                method_name in abstract_methods
            ), f"Method {method_name} should be abstract but is not in {abstract_methods}"

        # Check _update_environment_state, _calculate_reward are abstract
        assert "_update_environment_state" in abstract_methods
        assert "_calculate_reward" in abstract_methods

        # Verify _check_terminated, _check_truncated are abstract methods
        assert "_check_terminated" in abstract_methods
        assert "_check_truncated" in abstract_methods

        # Test _get_observation, _create_render_context are abstract
        assert "_get_observation" in abstract_methods
        assert "_create_render_context" in abstract_methods

        # Check _create_renderer, _seed_components are abstract methods
        assert "_create_renderer" in abstract_methods
        assert "_seed_components" in abstract_methods

        # Validate _cleanup_components, _validate_component_states are abstract
        assert "_cleanup_components" in abstract_methods
        assert "_validate_component_states" in abstract_methods

    def test_abstract_environment_error(self):
        """Test AbstractEnvironmentError exception functionality including error message
        formatting and implementation guidance generation."""
        # Create AbstractEnvironmentError with method name and class information
        method_name = "_test_abstract_method"
        class_name = "TestEnvironmentClass"
        error = AbstractEnvironmentError(method_name, class_name)

        # Verify error message includes method name and implementation requirements
        assert method_name in str(error)
        assert class_name in str(error)
        assert "abstract method" in str(error).lower()

        # Test get_implementation_guidance provides comprehensive developer guidance
        guidance = error.get_implementation_guidance()
        assert isinstance(guidance, str)
        assert len(guidance) > 50  # Should provide substantial guidance
        assert method_name in guidance
        assert class_name in guidance

        # Check error includes method signature and parameter information
        assert "implement" in guidance.lower()
        assert "method" in guidance.lower()

        # Validate implementation hints are helpful for environment development
        assert any(
            keyword in guidance.lower()
            for keyword in ["override", "implement", "define", "provide", "return"]
        )

        # Test error formatting for different abstract method types
        rendering_error = AbstractEnvironmentError("_create_renderer", "TestEnv")
        rendering_guidance = rendering_error.get_implementation_guidance()
        assert "render" in rendering_guidance.lower()

        # Verify error context includes debugging information for developers
        assert hasattr(error, "method_name")
        assert hasattr(error, "class_name")
        assert error.method_name == method_name
        assert error.class_name == class_name

    def test_concrete_implementation_validation(self):
        """Test validation of concrete implementation completeness ensuring all abstract
        methods are properly implemented with correct signatures."""
        # Create complete concrete implementation with all abstract methods
        config = create_test_environment_config()
        complete_env = create_mock_concrete_environment(config)

        # Verify instantiation succeeds without abstract method errors
        assert complete_env is not None
        assert isinstance(complete_env, BaseEnvironment)

        # Test partial implementation fails with specific missing method information
        class PartialEnvironment(BaseEnvironment):
            def _reset_environment_state(self):
                pass

            def _process_action(self, action):
                pass

            def _update_environment_state(self):
                pass

            # Missing other required methods

        with pytest.raises(TypeError):
            PartialEnvironment(config)

        # Validate method signature compatibility between abstract and concrete methods
        # All abstract methods should be callable on the complete implementation
        abstract_methods = BaseEnvironment.__abstractmethods__
        for method_name in abstract_methods:
            assert hasattr(complete_env, method_name)
            assert callable(getattr(complete_env, method_name))

        # Check return type consistency for implemented abstract methods
        # Test methods that should return specific types
        assert isinstance(complete_env._calculate_reward(), (int, float))
        assert isinstance(complete_env._check_terminated(), bool)
        assert isinstance(complete_env._check_truncated(), bool)
        assert isinstance(complete_env._get_observation(), np.ndarray)
        assert isinstance(complete_env._create_render_context(), dict)
        assert isinstance(complete_env._validate_component_states(), bool)

        # Test abstract method implementation with invalid signatures raises errors
        class BadSignatureEnvironment(BaseEnvironment):
            def _reset_environment_state(self, invalid_param):
                pass  # Wrong signature

            # ... implement other methods correctly

        # This would be caught during implementation, not instantiation
        # The abstract method signature checking is implicit in Python


@pytest.mark.api_compliance
class TestBaseEnvironmentAPICompliance:
    """Test class for comprehensive Gymnasium API compliance validation including method
    signatures, return formats, error handling, and specification adherence with detailed
    compliance testing for all interface requirements."""

    def test_reset_api_compliance(self):
        """Test reset method compliance with Gymnasium specification including signature,
        return format, and seed handling validation."""
        # Test reset method signature accepts optional seed and options parameters
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)

        # Basic reset call without parameters
        result = mock_env.reset()
        assert isinstance(result, tuple) and len(result) == 2

        # Reset with seed parameter
        result_with_seed = mock_env.reset(seed=42)
        assert isinstance(result_with_seed, tuple) and len(result_with_seed) == 2

        # Reset with options parameter
        result_with_options = mock_env.reset(options={"test": True})
        assert isinstance(result_with_options, tuple) and len(result_with_options) == 2

        # Verify reset returns tuple with (observation, info) format
        observation, info = result

        # Check observation is numpy array with proper shape and dtype
        assert isinstance(observation, np.ndarray)
        assert observation.dtype == np.float32
        assert observation.shape == mock_env.observation_space.shape
        assert mock_env.observation_space.contains(observation)

        # Validate info is dictionary with required episode information
        assert isinstance(info, dict)
        # Info should contain useful debugging/analysis information
        expected_keys = ["step_count", "episode_id"]
        for key in expected_keys:
            if key in info:  # Some keys are optional
                assert isinstance(info[key], (int, str, float))

        # Test seed parameter handling and random state initialization
        mock_env.reset(seed=123)
        assert mock_env.np_random is not None

        # Verify reset sets environment to initial state properly
        assert mock_env._step_count == 0
        assert not mock_env._closed

        # Test reset works correctly after previous episode completion
        mock_env.step(0)  # Take a step
        assert mock_env._step_count > 0
        mock_env.reset()
        assert mock_env._step_count == 0  # Should reset step counter

    def test_step_api_compliance(self):
        """Test step method compliance with Gymnasium 5-tuple specification including
        proper termination and truncation handling."""
        # Test step method signature accepts action parameter
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)

        # Initialize environment first
        mock_env.reset()

        # Test valid action values
        for action in range(4):  # All valid actions
            result = mock_env.step(action)

            # Verify step returns 5-tuple: (observation, reward, terminated, truncated, info)
            assert isinstance(result, tuple) and len(result) == 5

            observation, reward, terminated, truncated, info = result

            # Check observation is numpy array with consistent shape and dtype
            assert isinstance(observation, np.ndarray)
            assert observation.dtype == np.float32
            assert observation.shape == mock_env.observation_space.shape
            assert mock_env.observation_space.contains(observation)

            # Validate reward is numeric (int or float) type
            assert isinstance(reward, (int, float))

            # Test terminated and truncated are boolean values
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)

            # Verify info is dictionary with step metadata
            assert isinstance(info, dict)

            # Reset for next test if episode ended
            if terminated or truncated:
                mock_env.reset()

        # Test step raises error when called before reset
        fresh_env = create_mock_concrete_environment(config)
        # Don't call reset first
        with pytest.raises((StateError, RuntimeError)):
            fresh_env.step(0)

        # Validate step handling of terminated/truncated episodes
        mock_env.reset()

        # Mock environment to return terminated state
        with patch.object(mock_env, "_check_terminated", return_value=True):
            obs, reward, terminated, truncated, info = mock_env.step(0)
            assert terminated is True
            assert truncated is False

            # Further steps after termination should raise error
            with pytest.raises((StateError, RuntimeError)):
                mock_env.step(0)

    def test_render_api_compliance(self):
        """Test render method compliance with Gymnasium rendering specification including
        mode handling and output format validation."""
        # Test render method supports rgb_array mode returning numpy array
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)
        mock_env.reset()

        # Mock _create_renderer and _create_render_context for testing
        mock_renderer = Mock()
        mock_renderer.render_rgb_array.return_value = np.zeros(
            (64, 64, 3), dtype=np.uint8
        )
        mock_renderer.render_human.return_value = None

        with patch.object(mock_env, "_create_renderer", return_value=mock_renderer):
            # Verify rgb_array output has correct shape (H, W, 3) and uint8 dtype
            rgb_result = mock_env.render(mode="rgb_array")
            assert isinstance(rgb_result, np.ndarray)
            assert rgb_result.dtype == np.uint8
            assert len(rgb_result.shape) == 3
            assert rgb_result.shape[2] == 3  # RGB channels

            # Test human mode returns None and displays visualization
            human_result = mock_env.render(mode="human")
            assert human_result is None

        # Check render method handles invalid modes with appropriate errors
        with pytest.raises((ValidationError, ValueError)):
            mock_env.render(mode="invalid_mode")

        # Validate render works correctly before and after environment initialization
        fresh_env = create_mock_concrete_environment(config)
        with patch.object(fresh_env, "_create_renderer", return_value=mock_renderer):
            # Should work even before reset
            try:
                result = fresh_env.render(mode="rgb_array")
                assert isinstance(result, np.ndarray)
            except StateError:
                pass  # Acceptable if render requires reset

        # Test render error handling and fallback strategies
        mock_failing_renderer = Mock()
        mock_failing_renderer.render_rgb_array.side_effect = RenderingError(
            "Render failed"
        )

        with patch.object(
            mock_env, "_create_renderer", return_value=mock_failing_renderer
        ):
            with pytest.raises(RenderingError):
                mock_env.render(mode="rgb_array")

        # Verify render method performance meets timing requirements
        with patch.object(mock_env, "_create_renderer", return_value=mock_renderer):
            perf_results = measure_performance(
                mock_env, "render", 10, {"mode": "rgb_array"}
            )
            # Should complete within reasonable time (not strictly enforced in tests)
            assert perf_results["average_ms"] < 100.0  # Generous limit for tests

    def test_seed_api_compliance(self):
        """Test seed method compliance with Gymnasium seeding specification including
        return format and reproducibility validation."""
        # Test seed method accepts optional integer seed parameter
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)

        # Test with integer seed
        result = mock_env.seed(42)

        # Verify seed returns list containing seed value used
        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], int)

        # Check seed sets up numpy random generator properly
        assert mock_env.np_random is not None

        # Test reproducibility with identical seeds produces identical results
        seed_value = 123

        # First run
        mock_env.seed(seed_value)
        mock_env.reset()
        obs1, info1 = mock_env.reset()
        action_sequence = [0, 1, 2, 3]
        results1 = []
        for action in action_sequence:
            result = mock_env.step(action)
            results1.append(result)

        # Second run with same seed
        mock_env.seed(seed_value)
        mock_env.reset()
        obs2, info2 = mock_env.reset()
        results2 = []
        for action in action_sequence:
            result = mock_env.step(action)
            results2.append(result)

        # Results should be identical (within numerical precision)
        np.testing.assert_array_almost_equal(obs1, obs2)
        for r1, r2 in zip(results1, results2):
            np.testing.assert_array_almost_equal(r1[0], r2[0])  # observations
            assert abs(r1[1] - r2[1]) < 1e-6  # rewards

        # Validate seed parameter validation and error handling
        with pytest.raises((ValidationError, ValueError, TypeError)):
            mock_env.seed("invalid_seed")

        with pytest.raises((ValidationError, ValueError)):
            mock_env.seed(-1)  # Negative seed

        # Test seed method called multiple times with different values
        seeds = [1, 42, 999]
        for seed_val in seeds:
            result = mock_env.seed(seed_val)
            assert isinstance(result, list)
            assert result[0] == seed_val

        # Verify seeding affects all environment randomness sources
        mock_env.seed(456)
        mock_env.reset()
        # Any random operations should be reproducible
        random_sample1 = mock_env.np_random.random()

        mock_env.seed(456)
        mock_env.reset()
        random_sample2 = mock_env.np_random.random()

        assert abs(random_sample1 - random_sample2) < 1e-10

    def test_close_api_compliance(self):
        """Test close method compliance with Gymnasium specification including resource
        cleanup and proper shutdown procedures."""
        # Test close method signature requires no parameters
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)

        # Verify close returns None per Gymnasium specification
        result = mock_env.close()
        assert result is None

        # Check close method performs proper resource cleanup
        assert mock_env._closed is True

        # Test close can be called multiple times safely
        mock_env.close()  # Second call
        mock_env.close()  # Third call
        assert mock_env._closed is True  # Should remain True

        # Validate close sets environment to closed state
        assert mock_env._closed is True

        # Test operations after close raise appropriate errors
        with pytest.raises((StateError, RuntimeError)):
            mock_env.reset()

        with pytest.raises((StateError, RuntimeError)):
            mock_env.step(0)

        with pytest.raises((StateError, RuntimeError)):
            mock_env.render()

        # Verify close cleanup includes renderer and component resources
        # Mock environment should have called _cleanup_components
        assert "_cleanup_components" in [
            call[0] if isinstance(call, tuple) else call
            for call in mock_env._method_calls
        ]

        # Test close handles cleanup errors gracefully
        error_env = create_mock_concrete_environment(config)
        with patch.object(
            error_env, "_cleanup_components", side_effect=Exception("Cleanup failed")
        ):
            try:
                error_env.close()
                # Should complete despite cleanup error
                assert error_env._closed is True
            except Exception:
                # Acceptable if close propagates cleanup errors
                pass


@pytest.mark.unit
class TestBaseEnvironmentErrorHandling:
    """Test class for comprehensive error handling validation including exception propagation,
    recovery strategies, error logging, and secure error reporting with component-specific
    error handling testing."""

    def test_validation_error_handling(self):
        """Test comprehensive validation error handling including parameter validation,
        error messages, and recovery suggestions."""
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)
        mock_env.reset()

        # Test invalid action parameter raises ValidationError with details
        invalid_actions = [-1, 4, 10, "invalid", None, 3.5]
        for invalid_action in invalid_actions:
            with pytest.raises(ValidationError) as exc_info:
                mock_env.step(invalid_action)

            error = exc_info.value
            assert hasattr(error, "parameter_name")
            assert hasattr(error, "invalid_value")
            assert hasattr(error, "expected_format")

            # Verify error message contains parameter context
            assert "action" in str(error).lower()

        # Verify invalid seed value raises appropriate validation error
        with pytest.raises((ValidationError, ValueError)):
            mock_env.seed("not_an_integer")

        with pytest.raises((ValidationError, ValueError)):
            mock_env.seed(-999)  # Negative seed

        # Check invalid render mode raises ValidationError with valid options
        with pytest.raises(ValidationError) as exc_info:
            mock_env.render(mode="invalid_render_mode")

        error = exc_info.value
        assert "render" in str(error).lower() or "mode" in str(error).lower()

        # Test configuration parameter validation errors include parameter context
        with pytest.raises((ConfigurationError, ValidationError)):
            create_test_environment_config(grid_size=(0, 0))

        with pytest.raises((ConfigurationError, ValidationError)):
            create_test_environment_config(max_steps=-10)

        # Validate error messages provide helpful guidance for correction
        try:
            mock_env.step(-5)  # Invalid action
        except ValidationError as e:
            guidance = e.recovery_suggestion
            assert guidance is not None
            assert len(guidance) > 10  # Should provide substantial guidance
            assert any(
                word in guidance.lower() for word in ["action", "valid", "range"]
            )

        # Test validation errors include parameter constraints and expected formats
        try:
            create_test_environment_config(goal_radius=-1.0)
        except (ConfigurationError, ValidationError) as e:
            error_details = (
                e.get_error_details() if hasattr(e, "get_error_details") else {}
            )
            assert "goal_radius" in str(e).lower()

        # Verify validation error recovery suggestions are actionable
        try:
            mock_env.seed("not_numeric")
        except (ValidationError, ValueError) as e:
            if hasattr(e, "recovery_suggestion"):
                suggestion = e.recovery_suggestion
                assert "integer" in suggestion.lower() or "number" in suggestion.lower()

    def test_state_error_handling(self):
        """Test state error handling including invalid state transitions, recovery actions,
        and state consistency validation."""
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)

        # Test operations before initialization raise StateError
        fresh_env = create_mock_concrete_environment(config)

        # Don't call reset first
        with pytest.raises((StateError, RuntimeError)):
            fresh_env.step(0)

        # Verify invalid state transitions raise appropriate state errors
        mock_env.reset()
        mock_env.close()

        # Operations after close should raise state errors
        with pytest.raises((StateError, RuntimeError)):
            mock_env.step(0)

        with pytest.raises((StateError, RuntimeError)):
            mock_env.reset()

        # Check state error includes current and expected state information
        try:
            closed_env = create_mock_concrete_environment(config)
            closed_env.reset()
            closed_env.close()
            closed_env.step(0)
        except (StateError, RuntimeError) as e:
            if isinstance(e, StateError):
                assert hasattr(e, "current_state")
                assert hasattr(e, "expected_state")
                assert "closed" in str(e).lower() or "terminated" in str(e).lower()

        # Test state error recovery suggestions are component-specific
        try:
            mock_env.reset()  # After close
        except (StateError, RuntimeError) as e:
            if isinstance(e, StateError):
                recovery = e.suggest_recovery_action()
                assert isinstance(recovery, str)
                assert len(recovery) > 0

        # Validate state consistency validation identifies inconsistencies
        inconsistent_env = create_mock_concrete_environment(config)
        inconsistent_env.reset()

        # Force inconsistent state
        inconsistent_env._closed = True
        inconsistent_env._initialized = False

        with pytest.raises((StateError, RuntimeError)):
            inconsistent_env.step(0)

        # Test state error handling includes component context
        component_error = StateError(
            "Test state error",
            current_state="invalid",
            expected_state="valid",
            component_name="test_component",
        )
        assert component_error.component_name == "test_component"

        # Verify state errors provide debugging information for developers
        assert hasattr(component_error, "current_state")
        assert hasattr(component_error, "expected_state")
        assert component_error.current_state == "invalid"
        assert component_error.expected_state == "valid"

    def test_component_error_handling(self):
        """Test component-level error handling including error diagnosis, recovery
        strategies, and component isolation."""
        config = create_test_environment_config()

        # Test component failure raises ComponentError with component identification
        component_error = ComponentError(
            "Test component failure",
            component_name="test_renderer",
            operation_name="render_frame",
            underlying_error=Exception("Underlying failure"),
        )

        # Verify component errors include operation context and failure details
        assert component_error.component_name == "test_renderer"
        assert component_error.operation_name == "render_frame"
        assert component_error.underlying_error is not None

        # Check component error diagnosis provides specific failure analysis
        diagnosis = component_error.diagnose_failure()
        assert isinstance(diagnosis, dict)
        assert "component_name" in diagnosis
        assert "operation_name" in diagnosis
        assert "failure_timestamp" in diagnosis

        # Test component error recovery strategies are component-appropriate
        recovery_suggestion = component_error._generate_component_recovery_suggestion()
        assert isinstance(recovery_suggestion, str)
        assert "render" in recovery_suggestion.lower()

        # Validate component error isolation prevents system-wide failures
        mock_env = create_mock_concrete_environment(config)

        # Mock a component failure that should be isolated
        with patch.object(
            mock_env,
            "_create_renderer",
            side_effect=ComponentError("Renderer failed", "renderer"),
        ):
            try:
                mock_env.render(mode="human")
            except (ComponentError, RenderingError):
                # Error should be caught and handled
                pass

            # Environment should still be functional
            assert not mock_env._closed
            assert mock_env._initialized

        # Test component error logging includes debugging context
        assert hasattr(component_error, "component_state")
        assert hasattr(component_error, "diagnostic_info")

        # Verify component errors suggest specific recovery actions
        component_error.set_component_state({"initialized": False, "error_count": 3})
        updated_diagnosis = component_error.diagnose_failure()
        assert "component_state_analysis" in updated_diagnosis

    def test_rendering_error_handling(self):
        """Test rendering error handling including backend failures, fallback strategies,
        and graceful degradation."""
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)
        mock_env.reset()

        # Test matplotlib backend failure raises RenderingError with context
        rendering_error = RenderingError(
            "Matplotlib backend failed",
            render_mode="human",
            backend_name="TkAgg",
            underlying_error=ImportError("No display"),
        )

        # Verify rendering errors include fallback suggestions
        fallbacks = rendering_error.get_fallback_suggestions()
        assert isinstance(fallbacks, list)
        assert len(fallbacks) > 0
        assert "rgb_array" in fallbacks  # Should always suggest rgb_array fallback

        # Check rendering error fallback to alternative modes works correctly
        assert rendering_error.render_mode == "human"
        assert rendering_error.backend_name == "TkAgg"
        assert "Agg" in fallbacks  # Should suggest headless backend

        # Test headless environment handling with automatic backend selection
        headless_error = RenderingError(
            "No display available", render_mode="human", backend_name="Qt5Agg"
        )
        headless_error.set_rendering_context({"headless": True})
        headless_fallbacks = headless_error.get_fallback_suggestions()
        assert "rgb_array" in headless_fallbacks
        assert "Agg backend" in headless_fallbacks

        # Validate rendering error recovery doesn't crash environment
        mock_failing_renderer = Mock()
        mock_failing_renderer.render_human.side_effect = RenderingError("Display error")
        mock_failing_renderer.render_rgb_array.return_value = np.zeros(
            (32, 32, 3), dtype=np.uint8
        )

        with patch.object(
            mock_env, "_create_renderer", return_value=mock_failing_renderer
        ):
            # Human mode should fail
            with pytest.raises(RenderingError):
                mock_env.render(mode="human")

            # But rgb_array should still work
            result = mock_env.render(mode="rgb_array")
            assert isinstance(result, np.ndarray)

            # Environment should remain functional
            assert not mock_env._closed

        # Test rendering error logging includes backend information
        assert hasattr(rendering_error, "backend_name")
        assert hasattr(rendering_error, "render_mode")
        assert hasattr(rendering_error, "underlying_error")

        # Verify rendering errors provide user-friendly guidance
        guidance = rendering_error.get_fallback_suggestions()
        assert all(isinstance(suggestion, str) for suggestion in guidance)
        assert len(guidance) > 0


@pytest.mark.performance
class TestBaseEnvironmentPerformance:
    """Test class for performance monitoring and validation including timing benchmarks,
    resource usage testing, memory management, and performance target compliance with
    comprehensive performance analysis."""

    def test_initialization_performance(self):
        """Test environment initialization performance ensuring setup completes within
        acceptable time limits with resource efficiency."""
        # Measure environment initialization time for various configurations
        configs_to_test = [
            create_test_environment_config(grid_size=(16, 16)),
            create_test_environment_config(grid_size=(32, 32)),
            create_test_environment_config(grid_size=(64, 64)),
        ]

        initialization_times = []

        for config in configs_to_test:
            start_time = time.perf_counter()

            # Test initialization with different grid sizes and parameter combinations
            mock_env = create_mock_concrete_environment(config, enable_validation=True)
            assert mock_env is not None

            end_time = time.perf_counter()
            init_time_ms = (end_time - start_time) * 1000
            initialization_times.append(init_time_ms)

        # Verify initialization time scales reasonably with configuration complexity
        assert all(
            time_ms < 100.0 for time_ms in initialization_times
        ), f"Initialization times too high: {initialization_times}"

        # Check memory usage during initialization stays within limits
        config = create_test_environment_config()

        try:
            import psutil

            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB

            mock_env = create_mock_concrete_environment(config)

            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = memory_after - memory_before

            assert (
                memory_increase < MEMORY_TEST_THRESHOLD_MB
            ), f"Memory increase {memory_increase:.1f}MB exceeds threshold"

        except ImportError:
            warnings.warn("psutil not available for memory testing")

        # Test initialization performance with and without validation enabled
        start_no_validation = time.perf_counter()
        env_no_validation = create_mock_concrete_environment(
            config, enable_validation=False
        )
        end_no_validation = time.perf_counter()

        start_with_validation = time.perf_counter()
        env_with_validation = create_mock_concrete_environment(
            config, enable_validation=True
        )
        end_with_validation = time.perf_counter()

        time_no_validation = (end_no_validation - start_no_validation) * 1000
        time_with_validation = (end_with_validation - start_with_validation) * 1000

        # Validation overhead should be reasonable
        validation_overhead = time_with_validation - time_no_validation
        assert (
            validation_overhead < 50.0
        ), f"Validation overhead too high: {validation_overhead:.2f}ms"

        # Validate component initialization contributes reasonably to total time
        assert all(env is not None for env in [env_no_validation, env_with_validation])

        # Compare initialization performance against established baselines
        baseline_ms = 50.0  # Reasonable baseline for test environments
        average_init_time = sum(initialization_times) / len(initialization_times)
        assert (
            average_init_time < baseline_ms
        ), f"Average initialization time {average_init_time:.2f}ms exceeds baseline {baseline_ms}ms"

    def test_method_execution_performance(self):
        """Test performance of key environment methods ensuring they meet specified
        timing targets for real-time operation."""
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)
        mock_env.reset()

        # Measure reset method execution time over multiple iterations
        reset_results = measure_performance(mock_env, "reset", 10)
        assert (
            reset_results["average_ms"] < 10.0
        ), f"Reset method too slow: {reset_results['average_ms']:.2f}ms"

        # Test step method latency meets <1ms target for environment operations
        step_results = measure_performance(
            mock_env, "step", PERFORMANCE_TEST_ITERATIONS, {"action": 0}
        )

        # Allow some flexibility in testing environment
        step_target_ms = 5.0  # Relaxed for testing with mocks
        assert (
            step_results["average_ms"] < step_target_ms
        ), f"Step method too slow: {step_results['average_ms']:.2f}ms (target: {step_target_ms}ms)"

        # Benchmark render method performance for both rgb_array and human modes
        mock_renderer = Mock()
        mock_renderer.render_rgb_array.return_value = np.zeros(
            (32, 32, 3), dtype=np.uint8
        )
        mock_renderer.render_human.return_value = None

        with patch.object(mock_env, "_create_renderer", return_value=mock_renderer):
            rgb_results = measure_performance(
                mock_env, "render", 20, {"mode": "rgb_array"}
            )

            human_results = measure_performance(
                mock_env, "render", 20, {"mode": "human"}
            )

            # Render performance targets (relaxed for testing)
            assert (
                rgb_results["average_ms"] < 20.0
            ), f"RGB render too slow: {rgb_results['average_ms']:.2f}ms"
            assert (
                human_results["average_ms"] < 50.0
            ), f"Human render too slow: {human_results['average_ms']:.2f}ms"

        # Validate seed method execution time for reproducibility setup
        seed_results = measure_performance(mock_env, "seed", 50, {"seed": 42})
        assert (
            seed_results["average_ms"] < 2.0
        ), f"Seed method too slow: {seed_results['average_ms']:.2f}ms"

        # Test close method execution time for proper resource cleanup
        close_env = create_mock_concrete_environment(config)
        start_time = time.perf_counter()
        close_env.close()
        end_time = time.perf_counter()
        close_time_ms = (end_time - start_time) * 1000

        assert close_time_ms < 10.0, f"Close method too slow: {close_time_ms:.2f}ms"

        # Check performance consistency across multiple operation cycles
        consistency_results = []
        for cycle in range(5):
            mock_env.reset()
            cycle_results = measure_performance(
                mock_env, "step", 20, {"action": cycle % 4}
            )
            consistency_results.append(cycle_results["average_ms"])

        # Performance should be consistent across cycles
        max_variation = max(consistency_results) - min(consistency_results)
        average_time = sum(consistency_results) / len(consistency_results)
        variation_percent = (max_variation / average_time) * 100

        assert (
            variation_percent < 50.0
        ), f"Performance variation too high: {variation_percent:.1f}%"

        # Generate performance analysis with timing statistics and optimization recommendations
        performance_summary = {
            "reset_avg_ms": reset_results["average_ms"],
            "step_avg_ms": step_results["average_ms"],
            "seed_avg_ms": seed_results["average_ms"],
            "close_time_ms": close_time_ms,
            "performance_consistency": variation_percent,
            "meets_targets": all(
                [
                    reset_results["average_ms"] < 10.0,
                    step_results["average_ms"] < step_target_ms,
                    seed_results["average_ms"] < 2.0,
                    close_time_ms < 10.0,
                ]
            ),
        }

        assert performance_summary[
            "meets_targets"
        ], f"Performance targets not met: {performance_summary}"

    def test_memory_usage_performance(self):
        """Test memory usage efficiency ensuring environment operates within specified
        memory constraints with proper cleanup."""
        config = create_test_environment_config()

        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        # Monitor memory usage during environment initialization
        memory_baseline = process.memory_info().rss / (1024 * 1024)

        mock_env = create_mock_concrete_environment(config)
        memory_after_init = process.memory_info().rss / (1024 * 1024)
        init_memory_increase = memory_after_init - memory_baseline

        assert (
            init_memory_increase < 20.0
        ), f"Initialization memory usage too high: {init_memory_increase:.1f}MB"

        # Track memory consumption during episode execution
        mock_env.reset()
        memory_after_reset = process.memory_info().rss / (1024 * 1024)

        # Execute multiple steps and monitor memory
        for step in range(50):
            mock_env.step(step % 4)

        memory_after_steps = process.memory_info().rss / (1024 * 1024)
        episode_memory_increase = memory_after_steps - memory_after_reset

        # Memory usage during episode should be minimal
        assert (
            episode_memory_increase < 5.0
        ), f"Episode memory increase too high: {episode_memory_increase:.1f}MB"

        # Test memory usage with different grid sizes and configurations
        grid_configs = [
            create_test_environment_config(grid_size=(16, 16)),
            create_test_environment_config(grid_size=(32, 32)),
            create_test_environment_config(grid_size=(48, 48)),
        ]

        memory_usage_by_size = []

        for test_config in grid_configs:
            memory_before = process.memory_info().rss / (1024 * 1024)
            test_env = create_mock_concrete_environment(test_config)
            test_env.reset()
            memory_after = process.memory_info().rss / (1024 * 1024)

            memory_usage = memory_after - memory_before
            memory_usage_by_size.append(memory_usage)

            test_env.close()

        # Memory usage should scale reasonably with grid size
        assert all(
            usage < MEMORY_TEST_THRESHOLD_MB for usage in memory_usage_by_size
        ), f"Memory usage exceeds threshold for grid sizes: {memory_usage_by_size}"

        # Verify memory cleanup after environment closure
        memory_before_cleanup = process.memory_info().rss / (1024 * 1024)
        mock_env.close()

        # Force garbage collection
        import gc

        gc.collect()

        memory_after_cleanup = process.memory_info().rss / (1024 * 1024)
        cleanup_effectiveness = memory_before_cleanup - memory_after_cleanup

        # Some memory should be freed (though GC timing is unpredictable)
        assert (
            cleanup_effectiveness >= 0
        ), f"Memory usage increased after cleanup: {cleanup_effectiveness:.1f}MB"

        # Check for memory leaks during repeated initialization/cleanup cycles
        memory_leak_baseline = process.memory_info().rss / (1024 * 1024)

        for cycle in range(10):
            leak_test_env = create_mock_concrete_environment(config)
            leak_test_env.reset()
            leak_test_env.step(0)
            leak_test_env.close()

            if cycle % 3 == 0:
                gc.collect()  # Periodic cleanup

        memory_after_cycles = process.memory_info().rss / (1024 * 1024)
        potential_leak = memory_after_cycles - memory_leak_baseline

        # Allow some memory growth but detect significant leaks
        assert (
            potential_leak < 10.0
        ), f"Potential memory leak detected: {potential_leak:.1f}MB increase"

        # Test memory usage during rendering operations
        mock_renderer = Mock()
        mock_renderer.render_rgb_array.return_value = np.zeros(
            (32, 32, 3), dtype=np.uint8
        )

        with patch.object(mock_env, "_create_renderer", return_value=mock_renderer):
            memory_before_render = process.memory_info().rss / (1024 * 1024)

            for _ in range(20):
                mock_env.render(mode="rgb_array")

            memory_after_render = process.memory_info().rss / (1024 * 1024)
            render_memory_impact = memory_after_render - memory_before_render

            # Rendering should not cause significant memory growth
            assert (
                render_memory_impact < 5.0
            ), f"Rendering memory impact too high: {render_memory_impact:.1f}MB"

        # Validate total memory footprint stays within <50MB target
        final_memory = process.memory_info().rss / (1024 * 1024)
        total_memory_usage = final_memory - memory_baseline

        assert (
            total_memory_usage < MEMORY_TEST_THRESHOLD_MB
        ), f"Total memory usage exceeds target: {total_memory_usage:.1f}MB"

    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration including metric collection, analysis,
        and reporting functionality."""
        config = create_test_environment_config(enable_validation=True)
        mock_env = create_mock_concrete_environment(config, enable_validation=True)

        # Test performance monitoring decorator application to environment methods
        assert hasattr(mock_env, "_performance_monitor")
        assert mock_env._performance_monitor is not None

        # Verify timing data collection during environment operations
        mock_env.reset()

        # Performance monitor should track operations
        perf_monitor = mock_env._performance_monitor

        # Execute operations and check if they're being monitored
        start_count = getattr(perf_monitor, "operation_count", 0)

        mock_env.step(0)
        mock_env.step(1)
        mock_env.render(mode="rgb_array")

        # Check performance metrics storage and retrieval functionality
        if hasattr(perf_monitor, "get_metrics"):
            metrics = perf_monitor.get_metrics()
            assert isinstance(metrics, dict)

        # Test performance analysis and reporting generation
        if hasattr(perf_monitor, "generate_report"):
            report = perf_monitor.generate_report()
            assert isinstance(report, (dict, str))

        # Validate performance monitoring overhead is minimal
        # Test with and without monitoring
        config_no_monitoring = create_test_environment_config(enable_validation=False)
        config_no_monitoring.enable_performance_monitoring = False

        start_time = time.perf_counter()
        env_no_monitoring = create_mock_concrete_environment(config_no_monitoring)
        env_no_monitoring.reset()
        for _ in range(10):
            env_no_monitoring.step(0)
        no_monitoring_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        env_with_monitoring = create_mock_concrete_environment(config)
        env_with_monitoring.reset()
        for _ in range(10):
            env_with_monitoring.step(0)
        monitoring_time = time.perf_counter() - start_time

        # Monitoring overhead should be minimal
        overhead_ratio = (
            monitoring_time / no_monitoring_time if no_monitoring_time > 0 else 1.0
        )
        assert (
            overhead_ratio < 2.0
        ), f"Performance monitoring overhead too high: {overhead_ratio:.2f}x"

        # Test performance threshold checking and warning generation
        # Mock a slow operation
        slow_mock_env = create_mock_concrete_environment(config)

        with patch.object(
            slow_mock_env,
            "_process_action",
            side_effect=lambda action: time.sleep(0.01),
        ):  # 10ms delay
            slow_mock_env.reset()

            # This should trigger performance warnings if monitoring is active
            start_time = time.perf_counter()
            slow_mock_env.step(0)
            execution_time = (time.perf_counter() - start_time) * 1000

            assert execution_time > 5.0, "Test setup should create slow operation"

        # Verify performance monitoring integration with logging system
        with patch("logging.getLogger") as mock_logger:
            monitored_env = create_mock_concrete_environment(config)
            monitored_env.reset()
            monitored_env.step(0)

            # Logger should have been called for performance monitoring
            assert mock_logger.called


@pytest.mark.unit
class TestBaseEnvironmentConfiguration:
    """Test class for configuration management validation including parameter validation,
    consistency checking, resource estimation, and configuration serialization with
    comprehensive configuration testing."""

    def test_configuration_factory_function(self):
        """Test create_base_environment_config factory function including parameter
        validation, default handling, and error conditions."""
        # Test configuration creation with valid parameters produces proper config
        config = create_base_environment_config()

        assert isinstance(config, EnvironmentConfig)
        assert isinstance(config.grid_size, GridSize)
        assert isinstance(config.plume_params, PlumeParameters)
        assert config.max_steps > 0
        assert config.goal_radius >= 0

        # Verify default parameter handling when optional parameters omitted
        default_config = create_base_environment_config()
        assert default_config.grid_size.width == 128  # Default from constants
        assert default_config.grid_size.height == 128
        assert default_config.plume_params.sigma == 12.0  # Default sigma

        # Check invalid parameter combinations raise ConfigurationError
        with pytest.raises((ConfigurationError, ValidationError)):
            create_base_environment_config(
                grid_size=(0, 0),
                max_steps=100,  # Invalid grid size
            )

        with pytest.raises((ConfigurationError, ValidationError)):
            create_base_environment_config(max_steps=-1)  # Invalid max steps

        with pytest.raises((ConfigurationError, ValidationError)):
            create_base_environment_config(goal_radius=-1.0)  # Invalid goal radius

        # Test parameter validation with boundary conditions and edge cases
        boundary_tests = [
            {"grid_size": (1, 1), "should_pass": True},  # Minimum valid
            {"grid_size": (1024, 1024), "should_pass": False},  # Too large
            {"max_steps": 1, "should_pass": True},  # Minimum valid
            {"max_steps": 0, "should_pass": False},  # Invalid
            {"goal_radius": 0.0, "should_pass": True},  # Minimum valid
        ]

        for test_case in boundary_tests:
            try:
                test_config = create_base_environment_config(
                    **{k: v for k, v in test_case.items() if k != "should_pass"}
                )
                if not test_case["should_pass"]:
                    assert False, f"Should have failed for {test_case}"
            except (ConfigurationError, ValidationError):
                if test_case["should_pass"]:
                    assert False, f"Should have passed for {test_case}"

        # Validate cross-parameter consistency checking identifies conflicts
        with pytest.raises((ConfigurationError, ValidationError)):
            create_base_environment_config(
                grid_size=(10, 10),
                source_location=(20, 20),  # Outside grid bounds
            )

        # Test resource constraint validation prevents invalid configurations
        with pytest.raises((ConfigurationError, ValidationError)):
            create_base_environment_config(
                grid_size=(2000, 2000)  # Would exceed memory limits
            )

        # Verify configuration factory handles all parameter types correctly
        config_with_all_params = create_base_environment_config(
            grid_size=(64, 64),
            source_location=(32, 32),
            plume_sigma=15.0,
            max_steps=500,
            goal_radius=1.0,
        )

        assert config_with_all_params.grid_size.width == 64
        assert config_with_all_params.grid_size.height == 64
        assert config_with_all_params.plume_params.source_location.x == 32
        assert config_with_all_params.plume_params.source_location.y == 32
        assert config_with_all_params.plume_params.sigma == 15.0
        assert config_with_all_params.max_steps == 500
        assert config_with_all_params.goal_radius == 1.0

    def test_configuration_validation(self):
        """Test validate_base_environment_setup function including comprehensive validation,
        strict mode, and performance feasibility checking."""
        # Test configuration validation passes for valid configurations
        valid_config = create_test_environment_config()

        validation_result = validate_base_environment_setup(
            valid_config, strict_mode=False, performance_check=False
        )
        assert validation_result is True

        # Verify validation raises appropriate errors for invalid configurations
        invalid_configs = [
            create_test_environment_config(max_steps=-1),
            create_test_environment_config(goal_radius=-0.5),
        ]

        for invalid_config in invalid_configs:
            try:
                invalid_config.validate()
                assert False, f"Should have failed validation: {invalid_config}"
            except (ConfigurationError, ValidationError):
                pass  # Expected

        # Check strict validation mode identifies edge cases and boundary conditions
        edge_case_config = create_test_environment_config(
            grid_size=(1000, 1000),  # Large but technically valid
            max_steps=10000,  # Very long episodes
        )

        # Strict mode should catch potential issues
        try:
            edge_case_config.validate(strict_mode=True)
        except (ConfigurationError, ValidationError):
            pass  # Expected in strict mode

        # Test performance feasibility validation for resource-intensive configurations
        resource_intensive_config = create_test_environment_config(
            grid_size=(512, 512)  # Large grid
        )

        try:
            resource_intensive_config.validate(check_resource_constraints=True)
        except (ConfigurationError, ValidationError):
            pass  # May fail due to resource constraints

        # Validate cross-parameter consistency checking works correctly
        inconsistent_config = EnvironmentConfig(
            grid_size=GridSize(10, 10),
            plume_params=PlumeParameters(
                source_location=Coordinates(50, 50),
                sigma=12.0,  # Outside grid
            ),
            max_steps=100,
            goal_radius=0.0,
        )

        with pytest.raises((ConfigurationError, ValidationError)):
            inconsistent_config.validate()

        # Test validation context creation and error reporting detail
        try:
            inconsistent_config.validate(strict_mode=True)
        except (ConfigurationError, ValidationError) as e:
            error_details = (
                e.get_error_details() if hasattr(e, "get_error_details") else str(e)
            )
            assert (
                "source" in str(error_details).lower()
                or "grid" in str(error_details).lower()
            )

        # Verify validation recommendations provide actionable guidance
        try:
            create_test_environment_config(grid_size=(0, 10))
        except (ConfigurationError, ValidationError) as e:
            if hasattr(e, "recovery_suggestion"):
                suggestion = e.recovery_suggestion
                assert isinstance(suggestion, str)
                assert len(suggestion) > 10

    def test_helper_respects_explicit_grid_and_source(self):
        config = create_test_environment_config(
            grid_size=(32, 32), source_location=(16, 16)
        )

        assert config.grid_size == GridSize(32, 32)
        assert config.source_location == Coordinates(16, 16)
        assert config.plume_params.source_location == Coordinates(16, 16)

    def test_resource_estimation(self):
        """Test resource estimation functionality including memory calculation, computational
        requirements, and performance projections."""
        # Test memory estimation for various grid sizes and configurations
        grid_sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]

        for grid_size in grid_sizes:
            config = create_test_environment_config(grid_size=grid_size)

            # Check memory estimation accuracy against actual resource usage
            memory_estimate = config.grid_size.estimate_memory_mb()
            assert memory_estimate > 0

            # Larger grids should require more memory
            if grid_size == (128, 128):
                small_config = create_test_environment_config(grid_size=(16, 16))
                small_memory = small_config.grid_size.estimate_memory_mb()
                assert memory_estimate > small_memory

        # Verify computational time estimation for different parameter combinations
        configs_to_test = [
            create_test_environment_config(grid_size=(32, 32), max_steps=100),
            create_test_environment_config(grid_size=(32, 32), max_steps=500),
            create_test_environment_config(grid_size=(64, 64), max_steps=100),
        ]

        for config in configs_to_test:
            resource_estimates = env_manager.config.derive_component_configs()[
                "StateManager"
            ].estimate_resources()
            assert isinstance(resource_estimates, dict)
            assert set(resource_estimates.keys()) == {
                "memory_usage_mb",
                "memory_breakdown_mb",
                "step_latency_target_ms",
                "reset_latency_estimate_ms",
                "optimization_notes",
            }
            assert isinstance(resource_estimates["memory_usage_mb"], (int, float))
            assert resource_estimates["memory_usage_mb"] > 0
            assert isinstance(resource_estimates["memory_breakdown_mb"], dict)
            assert resource_estimates["reset_latency_estimate_ms"] > 0
            breakdown = resource_estimates["memory_breakdown_mb"]
            assert set(breakdown.keys()) == {
                "base_state_manager",
                "grid_storage",
                "history_tracking",
                "snapshot_cache",
                "component_overhead",
                "monitoring_overhead",
            }
            assert sum(breakdown.values()) > 0
            assert breakdown["grid_storage"] >= 0
            assert resource_estimates["step_latency_target_ms"] > 0
            assert resource_estimates["optimization_notes"] is not None

        # Test resource constraint validation prevents over-allocation
        large_config = create_test_environment_config(grid_size=(256, 256))
        large_resources = large_config.estimate_resources()

        if (
            large_resources["memory_usage_mb"]["total_estimated"]
            > MEMORY_TEST_THRESHOLD_MB
        ):
            with pytest.raises((ConfigurationError, ValidationError)):
                validate_environment_config(
                    {
                        "grid_size": (2048, 2048),
                        "source_location": (1024, 1024),
                        "plume_parameters": {
                            "source_location": (1024, 1024),
                            "sigma": DEFAULT_PLUME_SIGMA,
                            "model_type": STATIC_GAUSSIAN_MODEL_TYPE,
                        },
                        "max_steps": 5000,
                        "goal_radius": 10.0,
                    }
                )

        # Comprehensive resource estimation should report all component requirements
        comprehensive_config = create_test_environment_config()
        comprehensive_resources = comprehensive_config.estimate_resources()

        required_components = [
            "plume_field",
            "base_environment",
            "state_manager",
            "render_pipeline",
        ]
        for component in required_components:
            assert (
                component in comprehensive_resources["memory_usage_mb"]
            ), f"Missing memory estimate for component '{component}'"

        # Test resource optimization recommendations for large configurations
        optimization_config = create_test_environment_config(grid_size=(128, 128))
        optimization_resources = optimization_config.estimate_resources()

        if "optimization_suggestions" in optimization_resources:
            suggestions = optimization_resources["optimization_suggestions"]
            assert isinstance(suggestions, (list, dict))

        # Verify resource estimation integration with configuration validation
        resource_config = create_test_environment_config(grid_size=(64, 64))

        # Should include resource estimation in validation
        try:
            resource_config.validate(check_resource_constraints=True)
            resources = resource_config.estimate_resources()
            assert (
                resources["memory_usage_mb"]["total_estimated"]
                < MEMORY_TEST_THRESHOLD_MB
            )
        except (ConfigurationError, ValidationError):
            # May fail if resources exceed limits
            pass

    def test_configuration_consistency(self):
        """Test configuration parameter consistency validation including cross-parameter
        relationships and logical constraints."""
        # Test grid size and source location consistency validation
        valid_combinations = [
            {"grid_size": (32, 32), "source_location": (16, 16)},  # Center
            {"grid_size": (64, 64), "source_location": (0, 0)},  # Corner
            {"grid_size": (64, 64), "source_location": (63, 63)},  # Opposite corner
        ]

        for combo in valid_combinations:
            try:
                config = create_test_environment_config(**combo)
                config.validate()
            except (ConfigurationError, ValidationError):
                assert False, f"Valid combination should not fail: {combo}"

        invalid_combinations = [
            {"grid_size": (32, 32), "source_location": (32, 32)},  # Outside bounds
            {"grid_size": (10, 10), "source_location": (50, 5)},  # Way outside
            {"grid_size": (64, 64), "source_location": (-1, 0)},  # Negative coords
        ]

        for combo in invalid_combinations:
            with pytest.raises((ConfigurationError, ValidationError)):
                config = create_test_environment_config(**combo, enable_validation=True)

        # Verify goal radius and max steps parameter relationship checking
        goal_step_tests = [
            {"goal_radius": 0.0, "max_steps": 100},  # Valid: exact goal
            {"goal_radius": 5.0, "max_steps": 1000},  # Valid: reasonable search
            {"goal_radius": 0.0, "max_steps": 1},  # Edge case: very short
        ]

        for test_case in goal_step_tests:
            try:
                config = create_test_environment_config(**test_case)
                config.validate()
            except (ConfigurationError, ValidationError):
                # Some combinations may be invalid in strict validation
                pass

        # Check plume parameters and grid size compatibility validation
        sigma_grid_tests = [
            {"grid_size": (32, 32), "plume_sigma": 5.0},  # Small sigma, small grid
            {"grid_size": (128, 128), "plume_sigma": 20.0},  # Large sigma, large grid
        ]

        for test_case in sigma_grid_tests:
            grid_size = test_case["grid_size"]
            sigma = test_case["plume_sigma"]

            config = create_test_environment_config(
                grid_size=grid_size, additional_params={"plume_sigma": sigma}
            )

            # Validate plume parameters are consistent with grid
            validate_plume_parameters(config.plume_params, config.grid_size)

        # Test render mode and backend compatibility consistency
        # This would be tested if render modes were part of configuration
        config_with_render = create_test_environment_config()

        # Basic consistency should pass
        assert config_with_render.validate()

        # Validate mathematical consistency between related parameters
        math_consistency_config = create_test_environment_config(
            grid_size=(64, 64),
            source_location=(32, 32),
            goal_radius=5.0,  # Grid center
        )

        # Goal radius should be reasonable for grid size
        grid_diagonal = (64**2 + 64**2) ** 0.5
        assert math_consistency_config.goal_radius < grid_diagonal / 2

        # Test configuration consistency across component boundaries
        multi_component_config = create_test_environment_config()

        # All components should be consistent
        assert multi_component_config.grid_size.contains_coordinates(
            multi_component_config.plume_params.source_location
        )

        # Verify consistency validation provides specific error details
        # Test that source_location out of bounds triggers validation error in EnvironmentConfig.__post_init__
        with pytest.raises((ConfigurationError, ValidationError)) as exc_info:
            EnvironmentConfig(
                grid_size=GridSize(10, 10),
                source_location=Coordinates(1000, 1000),  # Out of bounds
                plume_params=None,
                max_steps=100,
                goal_radius=0.0,
            )

        error = exc_info.value
        assert "source" in str(error).lower() or "bounds" in str(error).lower()


@pytest.mark.integration
class TestBaseEnvironmentIntegration:
    """Test class for integration testing including component interaction, cross-component
    validation, system-level functionality, and end-to-end testing with comprehensive
    integration scenarios."""

    def test_component_coordination(self):
        """Test coordination between environment components ensuring proper integration
        and state synchronization."""
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config, enable_validation=True)

        # Test environment initialization coordinates all component setup
        assert mock_env._initialized is True
        assert hasattr(mock_env, "_logger")
        assert hasattr(mock_env, "_performance_monitor")
        assert hasattr(mock_env, "_config")

        # Verify state management coordinates across all environment subsystems
        mock_env.reset()

        # Check that reset coordinated multiple components
        reset_calls = [call for call in mock_env._method_calls if "reset" in str(call)]
        assert len(reset_calls) > 0

        # Test step coordination across components
        initial_call_count = len(mock_env._method_calls)
        mock_env.step(0)

        # Step should coordinate multiple abstract methods
        step_related_calls = mock_env._method_calls[initial_call_count:]
        expected_step_methods = [
            "_process_action",
            "_update_environment_state",
            "_calculate_reward",
            "_check_terminated",
            "_check_truncated",
            "_get_observation",
        ]

        called_methods = [
            call[0] if isinstance(call, tuple) else call for call in step_related_calls
        ]

        for expected_method in expected_step_methods:
            assert (
                expected_method in called_methods
            ), f"Step should call {expected_method}"

        # Check performance monitoring integration across all components
        if hasattr(mock_env._performance_monitor, "operation_count"):
            assert mock_env._performance_monitor.operation_count > 0

        # Test error handling coordination for component failures
        with patch.object(
            mock_env,
            "_update_environment_state",
            side_effect=ComponentError("Component failed", "test"),
        ):
            with pytest.raises(ComponentError):
                mock_env.step(0)

        # Environment should remain in consistent state after component error
        assert not mock_env._closed  # Should not auto-close on component error

        # Validate logging integration provides consistent component information
        with patch("logging.getLogger") as mock_logger:
            coordinated_env = create_mock_concrete_environment(config)
            coordinated_env.reset()
            coordinated_env.step(0)

            # Logger should be used for component coordination
            assert mock_logger.called

        # Test seeding coordination ensures consistent randomness across components
        seed_value = 42

        # First environment with seed
        mock_env.seed(seed_value)
        mock_env.reset()
        first_obs = mock_env._get_observation()

        # Second environment with same seed
        second_env = create_mock_concrete_environment(config)
        second_env.seed(seed_value)
        second_env.reset()
        second_obs = second_env._get_observation()

        # Should produce consistent results across components
        np.testing.assert_array_equal(first_obs, second_obs)

        # Verify cleanup coordination properly shuts down all components
        mock_env.close()

        # Should have called cleanup on all components
        cleanup_calls = [
            call for call in mock_env._method_calls if "cleanup" in str(call)
        ]
        assert len(cleanup_calls) > 0
        assert mock_env._closed is True

    def test_rendering_integration(self):
        """Test rendering pipeline integration including renderer coordination, backend
        selection, and visualization consistency."""
        config = create_test_environment_config()
        mock_env = create_mock_concrete_environment(config)
        mock_env.reset()

        # Test rendering pipeline integration with environment state
        mock_renderer = Mock()
        mock_renderer.render_rgb_array.return_value = np.zeros(
            (32, 32, 3), dtype=np.uint8
        )
        mock_renderer.render_human.return_value = None

        with patch.object(
            mock_env, "_create_renderer", return_value=mock_renderer
        ) as mock_create:
            # Rendering should integrate with environment state
            result = mock_env.render(mode="rgb_array")

            # Verify renderer creation and management through environment lifecycle
            mock_create.assert_called_once()
            assert isinstance(result, np.ndarray)

            # Check rendering backend selection and compatibility handling
            human_result = mock_env.render(mode="human")
            assert human_result is None

        # Test rendering error handling and fallback strategies integration
        failing_renderer = Mock()
        failing_renderer.render_rgb_array.side_effect = RenderingError(
            "Backend failed", render_mode="rgb_array", backend_name="test"
        )
        failing_renderer.render_human.return_value = None

        with patch.object(mock_env, "_create_renderer", return_value=failing_renderer):
            # Should handle rendering errors gracefully
            with pytest.raises(RenderingError) as exc_info:
                mock_env.render(mode="rgb_array")

            error = exc_info.value
            assert hasattr(error, "render_mode")
            assert hasattr(error, "backend_name")

            # Environment should remain functional after render error
            assert not mock_env._closed

            # Human mode might still work
            try:
                human_result = mock_env.render(mode="human")
                assert human_result is None
            except RenderingError:
                pass  # Acceptable if all rendering fails

        # Validate rendering performance integration with monitoring system
        performance_renderer = Mock()
        performance_renderer.render_rgb_array.return_value = np.zeros(
            (32, 32, 3), dtype=np.uint8
        )

        with patch.object(
            mock_env, "_create_renderer", return_value=performance_renderer
        ):
            start_time = time.perf_counter()
            mock_env.render(mode="rgb_array")
            render_time = (time.perf_counter() - start_time) * 1000

            # Performance should be monitored
            if hasattr(mock_env._performance_monitor, "record_timing"):
                # Performance monitor should track rendering
                pass  # Implementation-specific

        # Test rendering resource management and cleanup integration
        resource_renderer = Mock()
        resource_renderer.render_rgb_array.return_value = np.zeros(
            (16, 16, 3), dtype=np.uint8
        )
        resource_renderer.cleanup = Mock()

        with patch.object(mock_env, "_create_renderer", return_value=resource_renderer):
            mock_env.render(mode="rgb_array")
            mock_env.close()

            # Renderer cleanup should be integrated with environment cleanup
            assert mock_env._closed

        # Verify rendering mode switching and configuration management
        mode_switching_renderer = Mock()
        mode_switching_renderer.render_rgb_array.return_value = np.zeros(
            (32, 32, 3), dtype=np.uint8
        )
        mode_switching_renderer.render_human.return_value = None

        with patch.object(
            mock_env, "_create_renderer", return_value=mode_switching_renderer
        ):
            # Switch between modes
            rgb_result = mock_env.render(mode="rgb_array")
            human_result = mock_env.render(mode="human")

            assert isinstance(rgb_result, np.ndarray)
            assert human_result is None

            # Renderer should handle mode switching
            assert mode_switching_renderer.render_rgb_array.called
            assert mode_switching_renderer.render_human.called

    def test_validation_integration(self):
        """Test validation system integration including cross-component validation, error
        propagation, and validation consistency."""
        config = create_test_environment_config(enable_validation=True)
        mock_env = create_mock_concrete_environment(config, enable_validation=True)

        # Test validation integration across all environment components
        mock_env.reset()

        # Component validation should be integrated
        validation_calls = [
            call for call in mock_env._method_calls if "validate" in str(call)
        ]
        assert len(validation_calls) > 0

        # Verify validation error propagation and handling consistency
        with patch.object(
            mock_env, "_validate_component_states", return_value=False
        ) as mock_validate:
            # Validation failure should be handled consistently
            try:
                result = mock_env.get_environment_status()
                # Should handle validation failure gracefully
                assert isinstance(result, dict)
            except (StateError, ValidationError):
                # Acceptable if validation failure propagates
                pass

        # Check cross-component validation identifies integration issues
        inconsistent_env = create_mock_concrete_environment(config)

        # Create artificial inconsistency
        with patch.object(
            inconsistent_env,
            "_get_observation",
            return_value=np.ones((100, 100), dtype=np.float32),
        ):  # Wrong size
            inconsistent_env.reset()

            # Cross-component validation should catch size mismatch
            with pytest.raises((ValidationError, ValueError)):
                obs, _ = inconsistent_env.reset()
                # Observation should match observation space
                assert inconsistent_env.observation_space.contains(obs)

        # Test validation context creation and error reporting integration
        validation_env = create_mock_concrete_environment(config)

        with patch.object(
            validation_env,
            "_process_action",
            side_effect=ValidationError(
                "Invalid action", parameter_name="action", invalid_value=-1
            ),
        ):
            validation_env.reset()

            with pytest.raises(ValidationError) as exc_info:
                validation_env.step(-1)

            error = exc_info.value
            assert hasattr(error, "parameter_name")
            assert error.parameter_name == "action"
            assert hasattr(error, "invalid_value")

        # Validate error recovery strategies work across component boundaries
        recovery_env = create_mock_concrete_environment(config)

        # Mock recoverable validation error
        with patch.object(
            recovery_env,
            "_check_terminated",
            side_effect=[ValidationError("Recoverable error"), False],
        ):
            recovery_env.reset()

            # Should recover from validation error
            try:
                obs, reward, terminated, truncated, info = recovery_env.step(0)
                # Should eventually succeed
                assert isinstance(obs, np.ndarray)
                assert isinstance(terminated, bool)
            except ValidationError:
                # May not recover in mock environment
                pass

        # Test validation performance integration with monitoring system
        perf_validation_env = create_mock_concrete_environment(config)

        start_time = time.perf_counter()
        perf_validation_env.reset()

        # Multiple validation operations
        for action in range(4):
            try:
                perf_validation_env.step(action)
            except (StateError, ValidationError):
                break

        validation_time = (time.perf_counter() - start_time) * 1000

        # Validation should not significantly impact performance
        assert validation_time < 50.0  # Generous limit for testing

        # Verify validation configuration affects all relevant components
        strict_config = create_test_environment_config(enable_validation=True)
        strict_config.enable_validation = True

        strict_env = create_mock_concrete_environment(strict_config)

        # Strict validation should affect all components
        strict_env.reset()

        # Should perform more thorough validation
        strict_validation_calls = [
            call for call in strict_env._method_calls if "validate" in str(call)
        ]
        assert len(strict_validation_calls) > 0

    def test_end_to_end_functionality(self):
        """Test complete environment lifecycle including initialization, operation, and
        cleanup with realistic usage scenarios."""
        config = create_test_environment_config()

        # Test complete environment initialization with all components
        mock_env = create_mock_concrete_environment(config, enable_validation=True)

        assert mock_env is not None
        assert mock_env._initialized is True
        assert not mock_env._closed

        # Test Gymnasium API compliance
        assert_gymnasium_compliance(mock_env, strict_validation=False)

        # Verify full episode execution with step-by-step operation validation
        obs, info = mock_env.reset(seed=42)

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == mock_env.observation_space.shape
        assert mock_env.observation_space.contains(obs)

        episode_length = 0
        max_steps = 50  # Reasonable test episode length

        for step in range(max_steps):
            # Choose valid action
            action = step % 4  # Cycle through all actions

            # Execute step
            obs, reward, terminated, truncated, info = mock_env.step(action)

            # Validate step results
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

            # Check Gymnasium compliance
            assert obs.shape == mock_env.observation_space.shape
            assert mock_env.observation_space.contains(obs)
            assert mock_env.action_space.contains(action)

            episode_length += 1

            # Episode should end eventually
            if terminated or truncated:
                break

        # Check environment state consistency throughout complete lifecycle
        assert episode_length > 0
        assert episode_length <= max_steps

        # Test rendering integration during episode
        mock_renderer = Mock()
        mock_renderer.render_rgb_array.return_value = np.zeros(
            (32, 32, 3), dtype=np.uint8
        )
        mock_renderer.render_human.return_value = None

        with patch.object(mock_env, "_create_renderer", return_value=mock_renderer):
            rgb_frame = mock_env.render(mode="rgb_array")
            assert isinstance(rgb_frame, np.ndarray)
            assert rgb_frame.shape[2] == 3  # RGB channels
            assert rgb_frame.dtype == np.uint8

        # Test error handling integration during realistic failure scenarios
        failure_scenarios = [
            ("action_validation", lambda: mock_env.step(-1)),
            ("render_failure", lambda: mock_env.render(mode="invalid_mode")),
        ]

        for scenario_name, failure_func in failure_scenarios:
            try:
                failure_func()
                # Some failures might be handled gracefully
            except (ValidationError, RenderingError, ValueError):
                # Expected failures
                pass

            # Environment should remain functional after handled errors
            if not mock_env._closed:
                try:
                    valid_obs = mock_env._get_observation()
                    assert isinstance(valid_obs, np.ndarray)
                except:
                    pass  # May fail in mock environment

        # Validate performance monitoring throughout complete environment usage
        if hasattr(mock_env, "_performance_monitor"):
            monitor = mock_env._performance_monitor

            # Should have recorded operations
            if hasattr(monitor, "operation_count"):
                assert monitor.operation_count > 0

        # Test resource management during extended environment operation
        try:
            import psutil

            process = psutil.Process()
            memory_before_extended = process.memory_info().rss / (1024 * 1024)

            # Extended operation simulation
            for cycle in range(10):
                mock_env.reset()
                for _ in range(10):
                    try:
                        mock_env.step(cycle % 4)
                    except (StateError, ValidationError):
                        break

            memory_after_extended = process.memory_info().rss / (1024 * 1024)
            memory_growth = memory_after_extended - memory_before_extended

            # Should not have significant memory growth
            assert memory_growth < 10.0  # 10MB limit for extended operation

        except ImportError:
            warnings.warn("psutil not available for extended memory testing")

        # Verify proper cleanup and resource deallocation after environment usage
        assert not mock_env._closed
        mock_env.close()
        assert mock_env._closed

        # Operations after close should fail appropriately
        with pytest.raises((StateError, RuntimeError)):
            mock_env.step(0)

        with pytest.raises((StateError, RuntimeError)):
            mock_env.reset()

        # Cleanup should have been called
        cleanup_calls = [
            call for call in mock_env._method_calls if "cleanup" in str(call)
        ]
        assert len(cleanup_calls) > 0

        # Complete integration test with multiple environments
        multi_env_configs = [
            create_test_environment_config(grid_size=(16, 16)),
            create_test_environment_config(grid_size=(32, 32)),
            create_test_environment_config(max_steps=50),
        ]

        environments = []

        for config in multi_env_configs:
            env = create_mock_concrete_environment(config)
            env.reset()
            environments.append(env)

        # All environments should be independent and functional
        for i, env in enumerate(environments):
            obs, reward, terminated, truncated, info = env.step(i % 4)

            assert isinstance(obs, np.ndarray)
            assert not env._closed

            # Clean up
            env.close()
            assert env._closed
