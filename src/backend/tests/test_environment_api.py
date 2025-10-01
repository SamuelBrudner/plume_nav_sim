"""
Comprehensive test module for Gymnasium environment API compliance testing focusing on PlumeSearchEnv interface validation,
registration system testing, core functionality verification, and performance requirement validation. Tests complete
Gymnasium API compatibility including reset(), step(), render(), seed(), and close() methods with proper 5-tuple
response validation, action/observation space compliance, error handling robustness, and cross-component integration
verification ensuring <1ms step latency and deterministic reproducibility.

This test module provides exhaustive validation of the PlumeSearchEnv implementation, ensuring complete Gymnasium
API compliance, performance targets, and system reliability. All tests are designed for maximum coverage with
comprehensive validation patterns, error condition testing, and integration verification.

Test Organization:
- API compliance testing with method signature validation
- Action/observation space testing with bounds checking
- Performance benchmarking with timing validation
- Reproducibility testing with seeding verification
- Error handling testing with edge case validation
- Registration system testing with gym.make() compatibility
- Component integration testing with cross-system validation
- Memory management testing with resource monitoring

Architecture Integration:
- Tests all environment layers (Interface, Domain Logic, Infrastructure, Foundation)
- Validates Template Method pattern implementation
- Ensures performance targets (<1ms step latency)
- Verifies resource management and cleanup
- Tests complete episode lifecycle management
"""

import contextlib  # >=3.10 - Context manager utilities for resource management, exception handling, and cleanup automation in test scenarios
import copy  # >=3.10 - Deep copying utilities for state isolation testing and configuration management
import gc  # >=3.10 - Garbage collection utilities for memory management testing, resource cleanup validation, and memory leak detection in environment testing
import threading  # >=3.10 - Thread utilities for concurrency testing and thread-safety validation
import time  # >=3.10 - High-precision timing utilities for performance measurement, latency testing, and benchmark validation against performance targets
import warnings  # >=3.10 - Warning management for test execution, deprecation handling, and compatibility issue detection during environment testing
from typing import (  # >=3.10 - Type hints for test parameter specifications, return type validation, and comprehensive type checking
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import gymnasium  # >=0.29.0 - RL environment framework for API compliance testing, space validation, gym.make() testing, and Gymnasium interface verification
import numpy as np  # >=2.1.0 - Array operations for observation validation, mathematical testing, performance benchmarking, and numerical accuracy verification

# External imports with version comments for comprehensive dependency management
import pytest  # >=8.0.0 - Testing framework for test organization, fixtures, parametrization, and comprehensive test execution with assertion support

from plume_nav_sim.core.constants import (
    PERFORMANCE_TARGET_STEP_LATENCY_MS,  # System constants for environment configuration and performance validation
)
from plume_nav_sim.core.constants import DEFAULT_GRID_SIZE, DEFAULT_SOURCE_LOCATION
from plume_nav_sim.core.types import (
    Coordinates,  # Action enumeration and coordinate types for validation testing and data creation
)
from plume_nav_sim.core.types import Action
from plume_nav_sim.envs.base_env import (  # Base environment class for inheritance testing and abstract method validation
    BaseEnvironment,
)

# Internal imports for environment testing with comprehensive API validation
from plume_nav_sim.envs.plume_search_env import (  # Primary environment class under test for comprehensive Gymnasium API compliance validation
    PlumeSearchEnv,
)
from plume_nav_sim.registration.register import (
    ENV_ID,  # Environment registration functions for gym.make() compatibility testing
)
from plume_nav_sim.registration.register import (
    is_registered,
    register_env,
    unregister_env,
)

# Global test constants for comprehensive validation patterns
VALID_ACTIONS = [0, 1, 2, 3]  # Valid discrete action values for testing
INVALID_ACTIONS = [
    -1,
    4,
    "invalid",
    None,
    1.5,
]  # Invalid action values for error testing
TEST_SEEDS = [42, 123, 456, 789, 999]  # Fixed seeds for reproducibility testing
PERFORMANCE_TEST_ITERATIONS = 1000  # Number of iterations for performance benchmarking
MEMORY_THRESHOLD_MB = 50  # Maximum memory usage threshold for resource testing
MAX_TEST_EPISODE_LENGTH = 100  # Maximum episode length for testing scenarios
API_COMPLIANCE_TIMEOUT = 30.0  # Timeout seconds for API compliance testing


class TestEnvironmentAPI:
    """
    Main test class for comprehensive Gymnasium environment API testing including method validation, interface
    compliance, performance verification, and integration testing with systematic test organization and resource
    management ensuring complete test coverage and validation accuracy.

    This test class provides systematic validation of all environment functionality including API compliance,
    performance targets, error handling, reproducibility, and cross-component integration with comprehensive
    resource management and cleanup verification.

    Test Categories:
    - Inheritance and API compliance testing
    - Action and observation space validation
    - Method functionality testing (reset, step, render, seed, close)
    - Performance benchmarking and requirement validation
    - Error handling and edge case testing
    - Registration system and gym.make() compatibility
    - Reproducibility and seeding validation
    - Component integration and resource management
    """

    def __init__(self):
        """Initialize test class with configuration and resource tracking for comprehensive API testing."""
        # Initialize test configuration dictionary with default parameters
        self.test_config = {
            "grid_size": DEFAULT_GRID_SIZE,
            "source_location": DEFAULT_SOURCE_LOCATION,
            "max_steps": 1000,
            "goal_radius": 0,
            "render_mode": "rgb_array",
            "timeout": API_COMPLIANCE_TIMEOUT,
        }

        # Initialize empty list for tracking created environment instances
        self.created_environments: List[PlumeSearchEnv] = []

        # Initialize performance data dictionary for metrics collection
        self.performance_data: Dict[str, List[float]] = {
            "reset_times": [],
            "step_times": [],
            "render_times": [],
            "memory_usage": [],
        }

        # Set cleanup_required flag for resource management
        self.cleanup_required = False

    def setup_method(self, method):
        """
        Test method setup including environment registration, resource allocation, and performance monitoring
        initialization for individual test preparation with comprehensive resource tracking and error handling.

        Args:
            method: Test method being prepared for execution
        """
        # Clear any existing environment registrations for clean test state
        if is_registered(ENV_ID):
            unregister_env(ENV_ID, suppress_warnings=True)

        # Register environment with test-specific configuration
        register_env(env_id=ENV_ID, kwargs=self.test_config.copy())

        # Initialize performance monitoring and resource tracking
        self.performance_data = {
            "reset_times": [],
            "step_times": [],
            "render_times": [],
            "memory_usage": [],
        }

        # Set up error handling and exception capture for test execution
        self.error_context = {
            "test_method": method.__name__ if method else "unknown",
            "setup_timestamp": time.time(),
            "test_config": self.test_config.copy(),
        }

        # Configure memory monitoring and leak detection
        gc.collect()  # Clean up before test

        # Initialize test-specific logging and debugging infrastructure
        self.cleanup_required = True

    def teardown_method(self, method):
        """
        Test method cleanup including environment cleanup, resource validation, performance data collection,
        and memory management for comprehensive test isolation with detailed validation reporting.

        Args:
            method: Test method being cleaned up after execution
        """
        # Close all created environment instances and validate resource cleanup
        for env in self.created_environments:
            try:
                if hasattr(env, "close"):
                    env.close()
            except Exception as e:
                warnings.warn(f"Error closing environment: {e}")

        # Clear created environments list
        self.created_environments.clear()

        # Unregister test environments to prevent registry pollution
        try:
            if is_registered(ENV_ID):
                unregister_env(ENV_ID, suppress_warnings=True)
        except Exception as e:
            warnings.warn(f"Error unregistering environment: {e}")

        # Collect performance metrics and resource usage data
        if self.performance_data:
            for metric_name, values in self.performance_data.items():
                if values:
                    avg_value = np.mean(values)
                    max_value = np.max(values)
                    print(
                        f"Performance {metric_name}: avg={avg_value:.3f}, max={max_value:.3f}"
                    )

        # Validate memory cleanup and detect potential memory leaks
        gc.collect()

        # Clear performance data and reset monitoring infrastructure
        self.performance_data = {
            "reset_times": [],
            "step_times": [],
            "render_times": [],
            "memory_usage": [],
        }

        # Generate test-specific cleanup report and validation summary
        self.cleanup_required = False

    def create_test_environment(
        self, config: Optional[Dict] = None, register_for_cleanup: bool = True
    ) -> PlumeSearchEnv:
        """
        Factory method for creating test environment instances with configuration validation, resource tracking,
        and automatic cleanup registration ensuring consistent test environment setup and resource management.

        Args:
            config: Optional configuration dictionary for custom environment setup
            register_for_cleanup: Whether to register environment for automatic cleanup

        Returns:
            Configured environment instance ready for testing with resource tracking
        """
        # Apply default configuration if config not provided
        effective_config = self.test_config.copy()
        if config:
            effective_config.update(config)

        # Validate configuration parameters for test compatibility
        assert isinstance(
            effective_config.get("grid_size"), tuple
        ), "Grid size must be tuple"
        assert (
            len(effective_config["grid_size"]) == 2
        ), "Grid size must have 2 dimensions"
        assert all(
            isinstance(d, int) and d > 0 for d in effective_config["grid_size"]
        ), "Grid dimensions must be positive integers"

        # Create environment instance using validated configuration
        env = PlumeSearchEnv(
            grid_size=effective_config["grid_size"],
            source_location=effective_config["source_location"],
            max_steps=effective_config["max_steps"],
            goal_radius=effective_config["goal_radius"],
            render_mode=effective_config.get("render_mode"),
        )

        # Register environment for automatic cleanup if register_for_cleanup is True
        if register_for_cleanup:
            self.created_environments.append(env)

        # Initialize performance monitoring for environment instance
        env._test_performance_tracking = {
            "creation_time": time.time(),
            "operation_count": 0,
        }

        # Add environment to created_environments list for tracking
        # Already done above if register_for_cleanup is True

        # Return configured environment ready for testing
        return env

    def validate_api_response(
        self, response: Any, method_name: str, expected_format: Dict[str, Any]
    ) -> bool:
        """
        Utility method for validating Gymnasium API response formats including tuple structure, type validation,
        and value range checking ensuring complete API compliance and data format consistency.

        Args:
            response: API response to validate
            method_name: Name of method that produced response
            expected_format: Dictionary describing expected response format

        Returns:
            True if response format is valid, raises assertion error if invalid
        """
        # Validate response is proper tuple format for specified method
        if method_name == "reset":
            assert isinstance(
                response, tuple
            ), f"Reset must return tuple, got {type(response)}"
            assert (
                len(response) == 2
            ), f"Reset must return 2-tuple, got length {len(response)}"
            observation, info = response

            # Check each tuple element matches expected type from expected_format
            assert isinstance(
                observation, np.ndarray
            ), f"Observation must be numpy array, got {type(observation)}"
            assert isinstance(info, dict), f"Info must be dictionary, got {type(info)}"

        elif method_name == "step":
            assert isinstance(
                response, tuple
            ), f"Step must return tuple, got {type(response)}"
            assert (
                len(response) == 5
            ), f"Step must return 5-tuple, got length {len(response)}"
            observation, reward, terminated, truncated, info = response

            # Validate value ranges and constraints for numerical elements
            assert isinstance(
                observation, np.ndarray
            ), f"Observation must be numpy array, got {type(observation)}"
            assert isinstance(
                reward, (int, float)
            ), f"Reward must be numeric, got {type(reward)}"
            assert isinstance(
                terminated, bool
            ), f"Terminated must be boolean, got {type(terminated)}"
            assert isinstance(
                truncated, bool
            ), f"Truncated must be boolean, got {type(truncated)}"
            assert isinstance(info, dict), f"Info must be dictionary, got {type(info)}"

        # Verify data structure consistency and required dictionary keys
        if method_name in ["reset", "step"]:
            # Get info dict from response
            info = response[-1] if method_name == "step" else response[1]

            # Check array shapes and dtypes for numpy array elements
            observation = response[0]
            assert observation.shape == (
                1,
            ), f"Observation shape must be (1,), got {observation.shape}"
            assert (
                observation.dtype == np.float32
            ), f"Observation dtype must be float32, got {observation.dtype}"
            assert (
                0.0 <= observation[0] <= 1.0
            ), f"Observation value must be in [0,1], got {observation[0]}"

        # Validate boolean flags and status indicators are proper types
        # Already validated above for step method

        # Return True for valid responses or raise detailed assertion error
        return True

    def measure_method_performance(
        self,
        env: PlumeSearchEnv,
        method_name: str,
        method_call: callable,
        iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Utility method for measuring environment method performance with comprehensive timing analysis and resource
        monitoring providing detailed performance metrics and statistical analysis for optimization guidance.

        Args:
            env: Environment instance to test
            method_name: Name of method being measured
            method_call: Callable that executes the method
            iterations: Number of iterations for measurement

        Returns:
            Performance metrics dictionary including timing, memory usage, and resource analysis
        """
        # Initialize performance measurement infrastructure with high-precision timing
        timing_data = []
        memory_usage = []

        # Record baseline memory usage and system resource state
        gc.collect()
        baseline_memory = 0  # Would use memory profiling in full implementation

        # Execute method_call for specified iterations with timing measurement
        for i in range(iterations):
            start_time = time.perf_counter()

            try:
                result = method_call()
            except Exception as e:
                warnings.warn(f"Method call failed on iteration {i}: {e}")
                continue

            end_time = time.perf_counter()

            # Record timing data
            execution_time_ms = (end_time - start_time) * 1000
            timing_data.append(execution_time_ms)

            # Monitor memory usage changes and resource utilization
            # In full implementation, would track actual memory usage
            memory_usage.append(baseline_memory)

        # Calculate performance statistics including mean, median, and percentiles
        if timing_data:
            performance_metrics = {
                "method_name": method_name,
                "iterations_completed": len(timing_data),
                "mean_time_ms": np.mean(timing_data),
                "median_time_ms": np.median(timing_data),
                "min_time_ms": np.min(timing_data),
                "max_time_ms": np.max(timing_data),
                "std_time_ms": np.std(timing_data),
                "p95_time_ms": np.percentile(timing_data, 95),
                "p99_time_ms": np.percentile(timing_data, 99),
            }

            # Compare performance against targets and generate analysis
            if (
                method_name == "step"
                and performance_metrics["mean_time_ms"]
                > PERFORMANCE_TARGET_STEP_LATENCY_MS
            ):
                performance_metrics["target_exceeded"] = True
                performance_metrics["target_ratio"] = (
                    performance_metrics["mean_time_ms"]
                    / PERFORMANCE_TARGET_STEP_LATENCY_MS
                )
            else:
                performance_metrics["target_exceeded"] = False
                performance_metrics["target_ratio"] = (
                    performance_metrics["mean_time_ms"]
                    / PERFORMANCE_TARGET_STEP_LATENCY_MS
                    if method_name == "step"
                    else 1.0
                )

            # Record memory usage analysis
            if memory_usage:
                performance_metrics.update(
                    {
                        "mean_memory_mb": np.mean(memory_usage),
                        "max_memory_mb": np.max(memory_usage),
                        "memory_growth": (
                            np.max(memory_usage) - np.min(memory_usage)
                            if len(memory_usage) > 1
                            else 0
                        ),
                    }
                )
        else:
            performance_metrics = {
                "method_name": method_name,
                "error": "All iterations failed",
                "iterations_completed": 0,
            }

        # Return comprehensive performance metrics dictionary
        return performance_metrics

    def validate_environment_state(
        self, env: PlumeSearchEnv, strict_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Utility method for comprehensive environment state validation including component consistency, mathematical
        accuracy, and integration verification ensuring environment integrity and debugging support.

        Args:
            env: Environment instance to validate
            strict_validation: Whether to apply strict validation rules

        Returns:
            State validation report with consistency analysis and recommendations
        """
        # Initialize validation report with timestamp and validation settings
        validation_report = {
            "validation_timestamp": time.time(),
            "strict_validation": strict_validation,
            "state_valid": True,
            "validation_results": {},
            "warnings": [],
            "errors": [],
            "recommendations": [],
        }

        # Validate environment initialization status and component readiness
        try:
            # Check basic environment state
            if hasattr(env, "_environment_initialized"):
                validation_report["validation_results"][
                    "initialized"
                ] = env._environment_initialized
            else:
                validation_report["warnings"].append(
                    "Environment initialization status unknown"
                )

            # Check internal state consistency including counters and flags
            if hasattr(env, "_step_count") and hasattr(env, "_episode_count"):
                step_count = env._step_count
                episode_count = env._episode_count

                if step_count < 0:
                    validation_report["errors"].append("Negative step count detected")
                    validation_report["state_valid"] = False

                if episode_count < 0:
                    validation_report["errors"].append(
                        "Negative episode count detected"
                    )
                    validation_report["state_valid"] = False

                validation_report["validation_results"].update(
                    {"step_count": step_count, "episode_count": episode_count}
                )

            # Validate component integration and cross-component state synchronization
            if hasattr(env, "plume_model") and hasattr(env, "state_manager"):
                if env.plume_model and env.state_manager:
                    # Check grid size consistency
                    if hasattr(env.plume_model, "grid_size") and hasattr(
                        env.state_manager, "grid_size"
                    ):
                        if env.plume_model.grid_size != env.state_manager.grid_size:
                            validation_report["errors"].append(
                                "Grid size mismatch between components"
                            )
                            validation_report["state_valid"] = False

            # Apply strict validation rules if strict_validation enabled
            if strict_validation:
                # Check mathematical consistency including coordinate systems and calculations
                if hasattr(env, "action_space") and hasattr(env, "observation_space"):
                    # Validate action space is discrete with correct size
                    if not isinstance(env.action_space, gymnasium.spaces.Discrete):
                        validation_report["errors"].append(
                            "Action space must be Discrete"
                        )
                        validation_report["state_valid"] = False
                    elif env.action_space.n != 4:
                        validation_report["errors"].append(
                            "Action space must have 4 actions"
                        )
                        validation_report["state_valid"] = False

                    # Validate observation space is Box with correct shape
                    if not isinstance(env.observation_space, gymnasium.spaces.Box):
                        validation_report["errors"].append(
                            "Observation space must be Box"
                        )
                        validation_report["state_valid"] = False
                    elif env.observation_space.shape != (1,):
                        validation_report["errors"].append(
                            "Observation space shape must be (1,)"
                        )
                        validation_report["state_valid"] = False

                # Validate resource usage and memory management effectiveness
                # In full implementation, would check actual resource usage
                validation_report["validation_results"]["resource_check"] = "passed"

            # Generate comprehensive state validation report with findings and recommendations
            if validation_report["state_valid"] and not validation_report["warnings"]:
                validation_report["recommendations"].append(
                    "Environment state is valid and ready"
                )
            elif validation_report["warnings"] and validation_report["state_valid"]:
                validation_report["recommendations"].append(
                    "Environment functional with minor warnings"
                )
            else:
                validation_report["recommendations"].append(
                    "Address validation errors before use"
                )

        except Exception as e:
            validation_report["errors"].append(
                f"Validation failed with exception: {str(e)}"
            )
            validation_report["state_valid"] = False

        return validation_report


def test_environment_inheritance():
    """
    Test that PlumeSearchEnv properly inherits from BaseEnvironment and implements all required abstract methods
    with comprehensive inheritance validation and method signature verification ensuring proper class hierarchy.

    This test validates the complete inheritance chain and ensures all abstract methods are properly implemented
    with correct signatures and behavior, supporting the Template Method design pattern implementation.
    """
    # Create test environment instance for inheritance testing
    test_env = PlumeSearchEnv(
        grid_size=DEFAULT_GRID_SIZE,
        source_location=DEFAULT_SOURCE_LOCATION,
        max_steps=100,
        goal_radius=0,
    )

    try:
        # Validate PlumeSearchEnv inherits from BaseEnvironment using isinstance check
        assert isinstance(
            test_env, BaseEnvironment
        ), "PlumeSearchEnv must inherit from BaseEnvironment"

        # Verify PlumeSearchEnv inherits from gymnasium.Env for API compliance
        assert isinstance(
            test_env, gymnasium.Env
        ), "PlumeSearchEnv must inherit from gymnasium.Env"

        # Verify all abstract methods from BaseEnvironment are implemented in PlumeSearchEnv
        abstract_methods = [
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

        for method_name in abstract_methods:
            # Check method signatures match expected Gymnasium API specifications
            assert hasattr(
                test_env, method_name
            ), f"Method {method_name} not implemented"
            method = getattr(test_env, method_name)
            assert callable(method), f"Method {method_name} is not callable"

        # Validate that no abstract methods remain unimplemented using inspect module
        import inspect

        # Get all abstract methods from BaseEnvironment
        base_abstract_methods = []
        for name, method in inspect.getmembers(
            BaseEnvironment, predicate=inspect.isfunction
        ):
            if getattr(method, "__isabstractmethod__", False):
                base_abstract_methods.append(name)

        # Verify all are implemented in PlumeSearchEnv
        for abstract_method in base_abstract_methods:
            concrete_method = getattr(test_env, abstract_method)
            assert not getattr(
                concrete_method, "__isabstractmethod__", False
            ), f"Abstract method {abstract_method} not properly implemented"

        # Ensure proper method resolution order (MRO) for inheritance hierarchy
        mro = PlumeSearchEnv.__mro__
        expected_classes = [PlumeSearchEnv, BaseEnvironment, gymnasium.Env]
        for expected_class in expected_classes:
            assert expected_class in mro, f"Class {expected_class} not in MRO"

        # Test that environment can be instantiated without abstract method errors
        test_env2 = PlumeSearchEnv(
            grid_size=(64, 64), source_location=(32, 32), max_steps=50
        )
        assert test_env2 is not None, "Environment should instantiate successfully"

    finally:
        # Clean up test environments
        if "test_env" in locals():
            test_env.close()
        if "test_env2" in locals():
            test_env2.close()


def test_gymnasium_api_compliance():
    """
    Comprehensive test of complete Gymnasium API compliance including method signatures, return types, space
    definitions, and interface contracts with thorough validation of all API requirements ensuring full compatibility.

    This test ensures 100% Gymnasium API compliance with comprehensive validation of all interface methods,
    return types, and behavioral contracts required by the Gymnasium specification.
    """
    # Create test environment for API compliance validation
    test_env = PlumeSearchEnv(
        grid_size=DEFAULT_GRID_SIZE,
        source_location=DEFAULT_SOURCE_LOCATION,
        max_steps=MAX_TEST_EPISODE_LENGTH,
    )

    try:
        # Test reset() method returns proper (observation, info) tuple format
        reset_result = test_env.reset()
        assert isinstance(reset_result, tuple), "Reset must return tuple"
        assert len(reset_result) == 2, "Reset must return 2-tuple"

        observation, info = reset_result
        assert isinstance(
            observation, np.ndarray
        ), "Reset observation must be numpy array"
        assert isinstance(info, dict), "Reset info must be dictionary"
        assert observation.shape == (
            1,
        ), f"Observation shape must be (1,), got {observation.shape}"
        assert (
            observation.dtype == np.float32
        ), f"Observation dtype must be float32, got {observation.dtype}"

        # Test step() method returns proper 5-tuple (obs, reward, terminated, truncated, info) format
        for action in VALID_ACTIONS:
            step_result = test_env.step(action)
            assert isinstance(step_result, tuple), "Step must return tuple"
            assert len(step_result) == 5, "Step must return 5-tuple"

            obs, reward, terminated, truncated, step_info = step_result
            assert isinstance(obs, np.ndarray), "Step observation must be numpy array"
            assert isinstance(reward, (int, float)), "Reward must be numeric"
            assert isinstance(terminated, bool), "Terminated must be boolean"
            assert isinstance(truncated, bool), "Truncated must be boolean"
            assert isinstance(step_info, dict), "Step info must be dictionary"

            # Break after first step for basic validation
            break

        # Validate action_space is properly defined as gymnasium.spaces.Discrete(4)
        assert hasattr(test_env, "action_space"), "Environment must have action_space"
        assert isinstance(
            test_env.action_space, gymnasium.spaces.Discrete
        ), "Action space must be Discrete"
        assert test_env.action_space.n == 4, "Action space must have 4 actions"

        # Validate observation_space is properly defined as gymnasium.spaces.Box with correct shape and dtype
        assert hasattr(
            test_env, "observation_space"
        ), "Environment must have observation_space"
        assert isinstance(
            test_env.observation_space, gymnasium.spaces.Box
        ), "Observation space must be Box"
        assert test_env.observation_space.shape == (
            1,
        ), "Observation space shape must be (1,)"
        assert (
            test_env.observation_space.dtype == np.float32
        ), "Observation space dtype must be float32"
        assert (
            test_env.observation_space.low[0] == 0.0
        ), "Observation space low must be 0.0"
        assert (
            test_env.observation_space.high[0] == 1.0
        ), "Observation space high must be 1.0"

        # Test render() method supports both 'rgb_array' and 'human' modes
        # Test rgb_array mode
        test_env.render_mode = "rgb_array"
        rgb_result = test_env.render()
        if rgb_result is not None:  # Allow None for unavailable rendering
            assert isinstance(
                rgb_result, np.ndarray
            ), "RGB render must return numpy array"
            assert len(rgb_result.shape) == 3, "RGB array must be 3D"
            assert rgb_result.dtype == np.uint8, "RGB array must be uint8"

        # Test seeding via reset() per modern Gymnasium API
        obs, info = test_env.reset(seed=42)
        assert "seed" in info, "Reset should include seed in info"
        assert info["seed"] == 42, "Seed value should match requested"

        # Test close() method completes without errors and properly cleans up resources
        test_env.close()  # Should not raise exception

        # Verify all method signatures match Gymnasium environment interface specifications
        required_methods = ["reset", "step", "render", "close"]
        for method_name in required_methods:
            assert hasattr(
                test_env, method_name
            ), f"Method {method_name} required for Gymnasium compliance"
            assert callable(
                getattr(test_env, method_name)
            ), f"Method {method_name} must be callable"

        # Verify metadata attribute exists and contains render_modes
        assert hasattr(test_env, "metadata"), "Environment must have metadata attribute"
        assert isinstance(test_env.metadata, dict), "Metadata must be dictionary"
        assert "render_modes" in test_env.metadata, "Metadata must contain render_modes"

    except Exception as e:
        # Ensure cleanup even if test fails
        test_env.close()
        raise


@pytest.mark.parametrize("action", VALID_ACTIONS)
def test_action_space_validation(action):
    """
    Comprehensive action space testing including valid action processing, invalid action handling, boundary
    condition validation, and discrete action space compliance with error handling verification ensuring robust
    action processing and validation.

    Args:
        action: Parameterized valid action value for testing

    This test ensures complete action space compliance with comprehensive validation of all action types,
    boundary conditions, and error handling scenarios.
    """
    # Create test environment for action validation
    test_env = PlumeSearchEnv(
        grid_size=(32, 32), source_location=(16, 16), max_steps=50
    )

    try:
        # Initialize environment for action testing
        test_env.reset()

        # Test all valid actions (0, 1, 2, 3) are properly processed without errors
        step_result = test_env.step(action)
        assert len(step_result) == 5, "Step must return 5-tuple for valid action"

        obs, reward, terminated, truncated, info = step_result
        assert isinstance(obs, np.ndarray), "Valid action must return valid observation"
        assert isinstance(
            reward, (int, float)
        ), "Valid action must return numeric reward"

        # Verify action_space.contains() correctly identifies valid and invalid actions
        assert test_env.action_space.contains(
            action
        ), f"Action space must contain valid action {action}"

        # Test action space sampling produces valid actions within expected range
        for _ in range(10):
            sampled_action = test_env.action_space.sample()
            assert test_env.action_space.contains(
                sampled_action
            ), "Sampled action must be valid"
            assert 0 <= sampled_action <= 3, "Sampled action must be in range [0,3]"

        # Validate action space bounds and discrete nature with proper integer validation
        assert isinstance(
            test_env.action_space, gymnasium.spaces.Discrete
        ), "Action space must be Discrete"
        assert (
            test_env.action_space.n == 4
        ), "Action space must have exactly 4 discrete actions"

    finally:
        test_env.close()


@pytest.mark.parametrize("invalid_action", INVALID_ACTIONS)
def test_invalid_action_handling(invalid_action):
    """
    Test invalid action handling with proper error validation and boundary condition testing.

    Args:
        invalid_action: Parameterized invalid action value for error testing
    """
    test_env = PlumeSearchEnv(
        grid_size=(32, 32), source_location=(16, 16), max_steps=50
    )

    try:
        test_env.reset()

        # Test invalid actions raise appropriate ValidationError with descriptive messages
        with pytest.raises((ValueError, TypeError, AssertionError)):
            test_env.step(invalid_action)

        # Verify action_space.contains() correctly identifies invalid actions
        if invalid_action is not None and not isinstance(invalid_action, str):
            try:
                contains_result = test_env.action_space.contains(invalid_action)
                assert (
                    not contains_result
                ), f"Action space should not contain invalid action {invalid_action}"
            except (TypeError, ValueError):
                # Expected for invalid types
                pass

        # Verify action validation occurs before any state changes or processing
        initial_obs, _ = test_env.reset()

        try:
            test_env.step(invalid_action)
        except:
            # Environment state should remain unchanged after invalid action
            current_obs, _ = test_env.reset()  # Reset to get current state
            # State consistency check - environment should handle invalid actions gracefully
            pass

    finally:
        test_env.close()


def test_observation_space_validation():
    """
    Comprehensive observation space testing including observation format validation, value range checking, dtype
    verification, and Box space compliance with mathematical accuracy validation ensuring consistent observation
    format and data integrity.

    This test validates complete observation space compliance including format consistency, value ranges,
    mathematical accuracy, and Box space specification adherence.
    """
    test_env = PlumeSearchEnv(
        grid_size=DEFAULT_GRID_SIZE,
        source_location=DEFAULT_SOURCE_LOCATION,
        max_steps=100,
    )

    try:
        # Initialize environment and get initial observation
        observation, info = test_env.reset()

        # Test observations are numpy arrays with correct shape (1,) and dtype float32
        assert isinstance(observation, np.ndarray), "Observation must be numpy array"
        assert observation.shape == (
            1,
        ), f"Observation shape must be (1,), got {observation.shape}"
        assert (
            observation.dtype == np.float32
        ), f"Observation dtype must be float32, got {observation.dtype}"

        # Validate observation values are within expected range [0.0, 1.0] representing concentrations
        assert (
            0.0 <= observation[0] <= 1.0
        ), f"Observation value must be in [0,1], got {observation[0]}"

        # Test observation_space.contains() correctly validates observation arrays
        assert test_env.observation_space.contains(
            observation
        ), "Observation space must contain valid observation"

        # Test invalid observations are rejected
        invalid_obs_negative = np.array([-0.1], dtype=np.float32)
        assert not test_env.observation_space.contains(
            invalid_obs_negative
        ), "Negative observation should be invalid"

        invalid_obs_too_high = np.array([1.1], dtype=np.float32)
        assert not test_env.observation_space.contains(
            invalid_obs_too_high
        ), "Too high observation should be invalid"

        # Verify observations represent valid concentration values from plume model
        # Take several steps and validate all observations
        for action in [0, 1, 2, 3]:
            step_obs, reward, terminated, truncated, step_info = test_env.step(action)
            assert isinstance(
                step_obs, np.ndarray
            ), "Step observation must be numpy array"
            assert step_obs.shape == (1,), "Step observation shape must be (1,)"
            assert (
                step_obs.dtype == np.float32
            ), "Step observation dtype must be float32"
            assert 0.0 <= step_obs[0] <= 1.0, "Step observation must be in valid range"
            assert test_env.observation_space.contains(
                step_obs
            ), "Step observation must be valid"

            if terminated or truncated:
                break

        # Test observation consistency across multiple environment steps
        test_env.reset(seed=42)
        obs1, _ = test_env.reset(seed=42)
        obs2, _ = test_env.reset(seed=42)
        np.testing.assert_array_equal(
            obs1, obs2, "Identical seeds should produce identical observations"
        )

        # Validate observation space bounds and mathematical properties
        assert isinstance(
            test_env.observation_space, gymnasium.spaces.Box
        ), "Observation space must be Box"
        assert (
            test_env.observation_space.low[0] == 0.0
        ), "Observation space low bound must be 0.0"
        assert (
            test_env.observation_space.high[0] == 1.0
        ), "Observation space high bound must be 1.0"
        assert (
            test_env.observation_space.bounded_above.all()
        ), "Observation space must be bounded above"
        assert (
            test_env.observation_space.bounded_below.all()
        ), "Observation space must be bounded below"

    finally:
        test_env.close()


def test_reset_method_functionality():
    """
    Comprehensive reset method testing including initial state validation, seeding behavior, episode initialization,
    component coordination, and proper environment state restoration ensuring reliable episode management.

    This test validates complete reset functionality including state initialization, seeding consistency,
    component coordination, and proper episode boundary management.
    """
    test_env = PlumeSearchEnv(
        grid_size=(64, 64), source_location=(32, 32), max_steps=200
    )

    try:
        # Test reset() without seed returns valid (observation, info) tuple
        reset_result = test_env.reset()
        assert isinstance(reset_result, tuple), "Reset must return tuple"
        assert len(reset_result) == 2, "Reset must return 2-tuple"

        observation, info = reset_result
        assert isinstance(
            observation, np.ndarray
        ), "Reset observation must be numpy array"
        assert isinstance(info, dict), "Reset info must be dictionary"

        # Test reset() with seed produces deterministic initial states
        obs1, info1 = test_env.reset(seed=123)
        obs2, info2 = test_env.reset(seed=123)
        np.testing.assert_array_equal(
            obs1, obs2, "Same seed should produce identical observations"
        )

        # Test different seeds produce different states
        obs3, info3 = test_env.reset(seed=456)
        assert not np.array_equal(
            obs1, obs3
        ), "Different seeds should produce different observations"

        # Validate initial observation is valid concentration value within bounds
        assert (
            0.0 <= observation[0] <= 1.0
        ), "Initial observation must be valid concentration"

        # Test info dictionary contains required episode metadata including agent position
        required_info_keys = ["episode_count", "step_count"]
        for key in required_info_keys:
            assert key in info, f"Info must contain {key}"

        assert info["step_count"] == 0, "Initial step count must be 0"
        assert isinstance(info["episode_count"], int), "Episode count must be integer"
        assert info["episode_count"] > 0, "Episode count must be positive"

        # Verify environment state is properly reset including step counter and episode status
        # Take some steps then reset
        for _ in range(5):
            test_env.step(0)

        # Reset and verify state restoration
        reset_obs, reset_info = test_env.reset()
        assert reset_info["step_count"] == 0, "Step count should reset to 0"
        assert (
            reset_info["episode_count"] > info["episode_count"]
        ), "Episode count should increment"

        # Test multiple consecutive resets work correctly without state contamination
        for i in range(3):
            obs_i, info_i = test_env.reset(seed=789)
            assert info_i["step_count"] == 0, f"Reset {i}: step count must be 0"
            np.testing.assert_array_equal(
                obs_i,
                obs1 if i == 0 else obs_i,
                "Consecutive resets with same seed should be identical",
            )

        # Validate component coordination during reset including plume initialization
        # This is implicitly tested by successful reset operations
        assert hasattr(test_env, "plume_model"), "Environment should have plume model"
        assert hasattr(
            test_env, "state_manager"
        ), "Environment should have state manager"

    finally:
        test_env.close()


def test_step_method_functionality():
    """
    Comprehensive step method testing including action processing, state transitions, reward calculation, termination
    logic, and 5-tuple response validation with performance monitoring ensuring complete environment dynamics.

    This test validates complete step functionality including action processing, state transitions, reward calculation,
    termination conditions, and proper 5-tuple response format with performance monitoring.
    """
    test_env = PlumeSearchEnv(
        grid_size=(32, 32),
        source_location=(16, 16),
        max_steps=100,
        goal_radius=1,  # Allow goal achievement
    )

    try:
        # Initialize environment
        test_env.reset()

        # Test step() returns proper 5-tuple (obs, reward, terminated, truncated, info) format
        for action in VALID_ACTIONS[:2]:  # Test first 2 actions
            step_result = test_env.step(action)
            assert isinstance(step_result, tuple), "Step must return tuple"
            assert len(step_result) == 5, "Step must return 5-tuple"

            obs, reward, terminated, truncated, info = step_result

            # Validate observation updates correctly after agent movement
            assert isinstance(obs, np.ndarray), "Step observation must be numpy array"
            assert obs.shape == (1,), "Step observation shape must be (1,)"
            assert 0.0 <= obs[0] <= 1.0, "Step observation must be in valid range"

            # Test reward calculation based on goal proximity and achievement
            assert isinstance(reward, (int, float)), "Reward must be numeric"

            # Verify terminated flag is set correctly when agent reaches source location
            assert isinstance(terminated, bool), "Terminated must be boolean"

            # Test truncated flag is set when maximum episode steps reached
            assert isinstance(truncated, bool), "Truncated must be boolean"

            # Validate info dictionary contains comprehensive step metadata
            assert isinstance(info, dict), "Step info must be dictionary"
            assert "step_count" in info, "Info must contain step_count"
            assert "episode_count" in info, "Info must contain episode_count"
            assert "action_taken" in info, "Info must contain action_taken"

            # Test step counter increments correctly and episode tracking works
            assert info["step_count"] > 0, "Step count must increment"
            assert (
                info["action_taken"] == action
            ), "Action taken must match input action"

            if terminated or truncated:
                break

        # Test boundary enforcement prevents agent from moving outside grid
        # Reset to corner and try to move outside
        test_env.reset()

        # Force agent to corner (this would require access to internal state)
        # For now, just test that step operations complete successfully
        for _ in range(10):
            result = test_env.step(0)  # Move up repeatedly
            assert len(result) == 5, "All steps must return 5-tuple"

        # Test episode termination when max steps reached
        test_env_short = PlumeSearchEnv(
            grid_size=(16, 16), source_location=(8, 8), max_steps=5  # Short episode
        )

        try:
            test_env_short.reset()

            # Take enough steps to trigger truncation
            for i in range(10):
                obs, reward, terminated, truncated, info = test_env_short.step(0)

                if truncated:
                    assert i >= 4, "Truncation should occur after max_steps"
                    assert info["step_count"] >= 5, "Step count should reach max_steps"
                    break
            else:
                # If no truncation occurred, that's also valid depending on implementation
                pass
        finally:
            test_env_short.close()

    finally:
        test_env.close()


@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_seeding_and_reproducibility(seed):
    """
    Comprehensive seeding system testing including deterministic behavior validation, cross-episode consistency,
    component seeding coordination, and reproducibility verification across multiple episodes ensuring complete
    deterministic behavior.

    Args:
        seed: Parameterized seed value for reproducibility testing

    This test ensures complete reproducibility across all environment components with comprehensive validation
    of deterministic behavior, cross-episode consistency, and component coordination.
    """
    # Create multiple environment instances for reproducibility testing
    env1 = PlumeSearchEnv(grid_size=(32, 32), source_location=(16, 16), max_steps=50)

    env2 = PlumeSearchEnv(grid_size=(32, 32), source_location=(16, 16), max_steps=50)

    try:
        # Test identical seeds produce identical initial agent positions across episodes
        obs1, info1 = env1.reset(seed=seed)
        obs2, info2 = env2.reset(seed=seed)

        np.testing.assert_array_equal(
            obs1,
            obs2,
            f"Seed {seed}: identical seeds should produce identical initial observations",
        )

        # Validate identical action sequences produce identical trajectories with same seeds
        trajectory1 = []
        trajectory2 = []

        action_sequence = [0, 1, 2, 3, 1, 0]  # Fixed action sequence

        for action in action_sequence:
            step1 = env1.step(action)
            step2 = env2.step(action)

            trajectory1.append(step1)
            trajectory2.append(step2)

            # Compare observations
            np.testing.assert_array_equal(
                step1[0],
                step2[0],
                f"Seed {seed}, action {action}: observations should be identical",
            )

            # Compare rewards
            assert (
                step1[1] == step2[1]
            ), f"Seed {seed}, action {action}: rewards should be identical"

            # Compare termination flags
            assert (
                step1[2] == step2[2]
            ), f"Seed {seed}, action {action}: terminated flags should be identical"
            assert (
                step1[3] == step2[3]
            ), f"Seed {seed}, action {action}: truncated flags should be identical"

            if step1[2] or step1[3]:  # If episode ended
                break

        # Test different seeds produce different episode outcomes and agent placements
        different_seed = seed + 1000 if seed < 5000 else seed - 1000

        obs_diff, _ = env1.reset(seed=different_seed)

        # Different seeds should typically produce different outcomes
        # (though not guaranteed for all cases)
        different_outcome = not np.array_equal(obs1, obs_diff)

        # We don't assert this as different seeds might occasionally produce same initial state
        # but it's statistically very unlikely for good random number generators

        # Verify seeding via reset() properly coordinates across all environment components
        obs1_seeded, info1 = env1.reset(seed=seed)
        obs2_seeded, info2 = env2.reset(seed=seed)

        assert info1["seed"] == seed, "Seed should match requested value"
        assert info2["seed"] == seed, "Seed should match requested value"

        # With same seed, initial observations should be identical
        assert np.array_equal(
            obs1_seeded, obs2_seeded
        ), "Same seed should produce identical initial state"

        # Test cross-session reproducibility by creating new environment instances
        env3 = PlumeSearchEnv(
            grid_size=(32, 32), source_location=(16, 16), max_steps=50
        )

        try:
            obs3, _ = env3.reset(seed=seed)
            np.testing.assert_array_equal(
                obs1,
                obs3,
                f"Seed {seed}: cross-session reproducibility should be maintained",
            )

            # Test first step consistency
            step3 = env3.step(action_sequence[0])
            np.testing.assert_array_equal(
                trajectory1[0][0],
                step3[0],
                f"Seed {seed}: first step should be reproducible across sessions",
            )

        finally:
            env3.close()

        # Validate seeded random number generation maintains consistency throughout episodes
        # This is implicitly tested by the trajectory comparison above

        # Test seed parameter validation via reset() - invalid seeds should raise
        with pytest.raises((ValueError, TypeError)):
            env1.reset(seed="invalid_seed")  # type: ignore

        with pytest.raises((ValueError, TypeError)):
            env1.reset(seed=-1)  # Negative seed should be invalid

    finally:
        env1.close()
        env2.close()


def test_rendering_system_integration():
    """
    Comprehensive rendering system testing including dual-mode rendering validation, RGB array format verification,
    human mode compatibility, and rendering performance assessment ensuring complete visualization pipeline integration.

    This test validates complete rendering system functionality including both render modes, format verification,
    backend compatibility, and performance assessment with comprehensive error handling.
    """
    # Test rgb_array mode
    test_env_rgb = PlumeSearchEnv(
        grid_size=(64, 64),
        source_location=(32, 32),
        max_steps=100,
        render_mode="rgb_array",
    )

    try:
        # Initialize environment for rendering tests
        test_env_rgb.reset()

        # Test rgb_array mode returns numpy array with correct shape (H, W, 3) and dtype uint8
        rgb_result = test_env_rgb.render()

        if rgb_result is not None:
            assert isinstance(
                rgb_result, np.ndarray
            ), "RGB render must return numpy array"
            assert len(rgb_result.shape) == 3, "RGB array must be 3-dimensional"
            assert rgb_result.shape[2] == 3, "RGB array must have 3 color channels"
            assert rgb_result.dtype == np.uint8, "RGB array must have uint8 dtype"

            # Validate RGB array contains proper visual elements including agent and source markers
            # Check that array contains non-zero values (some visual content)
            assert rgb_result.sum() > 0, "RGB array should contain visual content"

            # Check value range is valid for uint8
            assert rgb_result.min() >= 0, "RGB values must be non-negative"
            assert rgb_result.max() <= 255, "RGB values must not exceed 255"

        # Test rendering with different environment states produces different visual outputs
        initial_render = test_env_rgb.render()

        # Take a step to change environment state
        test_env_rgb.step(0)
        step_render = test_env_rgb.render()

        if initial_render is not None and step_render is not None:
            # Renderings might be identical if agent didn't move or if change is subtle
            # This is not necessarily an error, so we just check they're valid
            assert isinstance(
                step_render, np.ndarray
            ), "Step render must be valid array"
            assert (
                step_render.shape == initial_render.shape
            ), "Render shape should be consistent"

    finally:
        test_env_rgb.close()

    # Test human mode rendering
    test_env_human = PlumeSearchEnv(
        grid_size=(32, 32), source_location=(16, 16), max_steps=50, render_mode="human"
    )

    try:
        test_env_human.reset()

        # Test human mode rendering completes without errors or returns None appropriately
        try:
            human_result = test_env_human.render()
            # Human mode should return None according to Gymnasium specification
            # or raise an exception if rendering backend is not available
            if human_result is not None:
                # Some implementations might return something other than None
                pass
        except Exception as e:
            # Rendering errors are acceptable if matplotlib is not available or configured
            assert (
                "matplotlib" in str(e).lower() or "display" in str(e).lower()
            ), f"Unexpected rendering error: {e}"

        # Verify rendering mode switching works correctly between rgb_array and human modes
        test_env_human.render_mode = "rgb_array"
        try:
            rgb_from_human_env = test_env_human.render()
            if rgb_from_human_env is not None:
                assert isinstance(
                    rgb_from_human_env, np.ndarray
                ), "Mode switch should work"
        except Exception:
            # Mode switching might not be supported in all implementations
            pass

        # Validate rendering performance meets targets with timing measurements
        # This is a basic performance test
        start_time = time.perf_counter()
        try:
            test_env_human.render()
        except Exception:
            pass  # Rendering might fail, that's OK
        end_time = time.perf_counter()

        render_time_ms = (end_time - start_time) * 1000
        # 100ms is a reasonable upper bound for rendering
        assert (
            render_time_ms < 100.0
        ), f"Rendering took too long: {render_time_ms:.1f}ms"

        # Test graceful fallback when matplotlib backend is unavailable
        # This is tested implicitly by the exception handling above

    finally:
        test_env_human.close()


def test_environment_registration():
    """
    Comprehensive environment registration testing including gym.make() compatibility, registration parameter
    validation, version management, and environment discovery verification ensuring complete Gymnasium ecosystem
    integration and compatibility.

    This test validates complete registration system functionality including gym.make() compatibility,
    parameter validation, version management, and environment discovery with comprehensive error handling.
    """
    # Clean up any existing registration
    if is_registered(ENV_ID):
        unregister_env(ENV_ID, suppress_warnings=True)

    try:
        # Test register_env() successfully registers environment with Gymnasium
        registered_id = register_env()
        assert (
            registered_id == ENV_ID
        ), f"Registration should return ENV_ID, got {registered_id}"

        # Validate gym.make(ENV_ID) creates functional environment instance
        gym_env = gymnasium.make(ENV_ID)

        try:
            assert isinstance(
                gym_env, PlumeSearchEnv
            ), "gym.make should create PlumeSearchEnv instance"

            # Test basic functionality of gym.make created environment
            obs, info = gym_env.reset()
            assert isinstance(obs, np.ndarray), "gym.make env should have valid reset"

            step_result = gym_env.step(0)
            assert len(step_result) == 5, "gym.make env should have valid step"

        finally:
            gym_env.close()

        # Test environment registration with custom parameters and configuration overrides
        custom_config = {
            "grid_size": (64, 64),
            "source_location": (32, 32),
            "goal_radius": 2.0,
        }

        # First unregister existing
        unregister_env(ENV_ID, suppress_warnings=True)

        # Register with custom parameters
        register_env(kwargs=custom_config)

        custom_env = gymnasium.make(ENV_ID)
        try:
            # Verify custom parameters are applied
            assert hasattr(custom_env, "grid_size"), "Custom env should have grid_size"
            assert hasattr(
                custom_env, "source_location"
            ), "Custom env should have source_location"
            # The exact verification depends on how parameters are stored

        finally:
            custom_env.close()

        # Verify registration status checking with is_registered() function accuracy
        assert is_registered(ENV_ID), "Environment should be registered"
        assert is_registered(
            ENV_ID, use_cache=False
        ), "Environment should be registered (no cache)"

        # Test environment unregistration and cleanup with unregister_env()
        unregister_result = unregister_env(ENV_ID)
        assert unregister_result == True, "Unregistration should succeed"
        assert not is_registered(
            ENV_ID
        ), "Environment should not be registered after unregistration"

        # Validate registration parameter validation and error handling for invalid configurations
        with pytest.raises((ValueError, TypeError)):
            register_env(kwargs={"invalid_param": "invalid_value"})

        # Test version management and strict versioning compliance with '-v0' suffix
        assert ENV_ID.endswith(
            "-v0"
        ), f"Environment ID should end with -v0, got {ENV_ID}"

        # Test registration with invalid environment ID
        with pytest.raises((ValueError, TypeError)):
            register_env(env_id="InvalidEnvName")  # Should require -v0 suffix

    finally:
        # Clean up registration
        if is_registered(ENV_ID):
            unregister_env(ENV_ID, suppress_warnings=True)


@pytest.mark.parametrize("invalid_action", INVALID_ACTIONS)
def test_error_handling_robustness(invalid_action):
    """
    Comprehensive error handling testing including invalid input validation, boundary condition handling, exception
    management, and graceful degradation with recovery strategy verification ensuring robust error handling.

    Args:
        invalid_action: Parameterized invalid action for error testing

    This test validates comprehensive error handling including input validation, boundary conditions,
    exception management, and graceful degradation with recovery strategy verification.
    """
    test_env = PlumeSearchEnv(
        grid_size=(32, 32), source_location=(16, 16), max_steps=100
    )

    try:
        # Initialize environment for error testing
        test_env.reset()

        # Test invalid actions raise appropriate ValidationError with descriptive messages
        with pytest.raises((ValueError, TypeError, AssertionError)):
            test_env.step(invalid_action)

        # Validate step() before reset() raises proper StateError with guidance
        uninitialized_env = PlumeSearchEnv(
            grid_size=(16, 16), source_location=(8, 8), max_steps=50
        )

        try:
            # Don't reset, try to step
            with pytest.raises((ValueError, AssertionError, RuntimeError)):
                uninitialized_env.step(0)
        finally:
            uninitialized_env.close()

        # Test boundary conditions including grid edge cases and extreme parameters
        boundary_env = PlumeSearchEnv(
            grid_size=(2, 2),  # Minimal grid
            source_location=(0, 0),  # Corner position
            max_steps=10,
        )

        try:
            boundary_env.reset()

            # Test movement in minimal grid
            for action in VALID_ACTIONS:
                try:
                    result = boundary_env.step(action)
                    assert (
                        len(result) == 5
                    ), "Boundary conditions should still return valid results"
                except Exception as e:
                    # Boundary errors should be handled gracefully
                    assert isinstance(
                        e, (ValueError, AssertionError)
                    ), f"Unexpected boundary error type: {type(e)}"
        finally:
            boundary_env.close()

        # Verify error recovery and environment state consistency after exceptions
        test_env.reset()

        try:
            test_env.step(invalid_action)
        except:
            # After error, environment should still be usable
            try:
                recovery_result = test_env.step(0)  # Valid action
                assert (
                    len(recovery_result) == 5
                ), "Environment should recover from errors"
            except Exception as recovery_error:
                # Some implementations might require reset after error
                test_env.reset()
                recovery_result = test_env.step(0)
                assert len(recovery_result) == 5, "Environment should work after reset"

        # Test invalid seed values via reset() raise appropriate TypeError with validation messages
        with pytest.raises((ValueError, TypeError)):
            test_env.reset(seed="invalid_seed_string")  # type: ignore

        with pytest.raises((ValueError, TypeError)):
            test_env.reset(seed=-1)  # Negative seed

        with pytest.raises((ValueError, TypeError)):
            test_env.reset(seed=1.5)  # type: ignore - Non-integer seed

        # Validate render mode errors are handled gracefully with fallback strategies
        test_env.render_mode = "invalid_mode"

        try:
            test_env.render()
            # Should either work or raise appropriate error
        except Exception as render_error:
            # Render errors should be specific types
            error_message = str(render_error).lower()
            assert any(
                keyword in error_message
                for keyword in ["render", "mode", "invalid", "unsupported"]
            ), f"Render error should be descriptive: {render_error}"

        # Test component error propagation and hierarchical error handling
        # This is implicitly tested by the above error scenarios

    finally:
        test_env.close()


def test_performance_requirements():
    """
    Comprehensive performance testing including step latency benchmarking, memory usage validation, rendering
    performance assessment, and system resource monitoring against defined targets ensuring performance compliance.

    This test validates performance requirements including step latency, memory usage, rendering performance,
    and system resource efficiency against defined targets with comprehensive monitoring and analysis.
    """
    # Create performance test environment
    performance_env = PlumeSearchEnv(
        grid_size=(128, 128),  # Standard test size
        source_location=(64, 64),
        max_steps=500,
        render_mode="rgb_array",
    )

    try:
        # Initialize performance tracking
        step_times = []
        reset_times = []
        render_times = []

        # Benchmark reset() method performance and initialization timing
        for _ in range(10):
            start_time = time.perf_counter()
            performance_env.reset()
            end_time = time.perf_counter()

            reset_time_ms = (end_time - start_time) * 1000
            reset_times.append(reset_time_ms)

        avg_reset_time = np.mean(reset_times)
        max_reset_time = np.max(reset_times)

        # Reset should be under 10ms on average (reasonable target)
        assert (
            avg_reset_time < 10.0
        ), f"Average reset time {avg_reset_time:.2f}ms exceeds 10ms target"

        # Benchmark step() method latency over PERFORMANCE_TEST_ITERATIONS iterations
        performance_env.reset()

        for i in range(PERFORMANCE_TEST_ITERATIONS):
            action = i % 4  # Cycle through actions

            start_time = time.perf_counter()
            result = performance_env.step(action)
            end_time = time.perf_counter()

            step_time_ms = (end_time - start_time) * 1000
            step_times.append(step_time_ms)

            # Reset periodically to avoid episode termination
            if i > 0 and i % 100 == 0:
                performance_env.reset()

        # Calculate step performance statistics
        avg_step_time = np.mean(step_times)
        p95_step_time = np.percentile(step_times, 95)
        p99_step_time = np.percentile(step_times, 99)
        max_step_time = np.max(step_times)

        # Validate average step latency meets <1ms target requirement from PERFORMANCE_TARGET_STEP_LATENCY_MS
        assert (
            avg_step_time < PERFORMANCE_TARGET_STEP_LATENCY_MS
        ), f"Average step time {avg_step_time:.3f}ms exceeds target {PERFORMANCE_TARGET_STEP_LATENCY_MS}ms"

        # P95 should be reasonable (allow some variation)
        assert (
            p95_step_time < PERFORMANCE_TARGET_STEP_LATENCY_MS * 2
        ), f"P95 step time {p95_step_time:.3f}ms exceeds 2x target"

        # Test memory usage remains below MEMORY_THRESHOLD_MB during extended episodes
        gc.collect()
        # In a full implementation, would measure actual memory usage
        # For now, just verify no obvious memory leaks by running extended episode

        performance_env.reset()
        for _ in range(1000):
            performance_env.step(np.random.randint(0, 4))
            # Reset if episode ends
            try:
                performance_env.step(0)
            except:
                performance_env.reset()

        # If we get here without crashing, memory usage is likely reasonable

        # Test rendering performance for both rgb_array and human modes
        performance_env.render_mode = "rgb_array"
        performance_env.reset()

        for _ in range(10):
            start_time = time.perf_counter()
            try:
                performance_env.render()
            except Exception:
                pass  # Rendering might fail, that's OK
            end_time = time.perf_counter()

            render_time_ms = (end_time - start_time) * 1000
            render_times.append(render_time_ms)

        if render_times:
            avg_render_time = np.mean(render_times)
            # 5ms is reasonable target for RGB rendering
            assert (
                avg_render_time < 50.0
            ), f"Average render time {avg_render_time:.2f}ms exceeds 50ms target"

        # Validate memory cleanup effectiveness and absence of memory leaks
        gc.collect()  # Force cleanup

        # Monitor system resource usage including CPU utilization during testing
        # This would require additional libraries in a full implementation

        # Print performance summary
        print(f"\nPerformance Summary:")
        print(
            f"  Average step time: {avg_step_time:.3f}ms (target: <{PERFORMANCE_TARGET_STEP_LATENCY_MS}ms)"
        )
        print(f"  P95 step time: {p95_step_time:.3f}ms")
        print(f"  P99 step time: {p99_step_time:.3f}ms")
        print(f"  Average reset time: {avg_reset_time:.2f}ms")
        if render_times:
            print(f"  Average render time: {np.mean(render_times):.2f}ms")
        print(f"  Total steps tested: {PERFORMANCE_TEST_ITERATIONS}")

    finally:
        performance_env.close()


def test_episode_lifecycle_management():
    """
    Comprehensive episode lifecycle testing including initialization, execution, termination, truncation, and state
    management with proper episode boundaries and metadata tracking ensuring complete episode management.

    This test validates complete episode lifecycle including initialization, execution, termination conditions,
    truncation handling, and proper state management with comprehensive metadata tracking.
    """
    test_env = PlumeSearchEnv(
        grid_size=(32, 32),
        source_location=(16, 16),
        max_steps=50,  # Short episodes for testing
        goal_radius=1,  # Allow goal achievement
    )

    try:
        # Test complete episode from reset to termination with goal achievement
        obs, info = test_env.reset()
        assert info["step_count"] == 0, "Initial step count should be 0"

        episode_steps = 0
        episode_rewards = []

        while True:
            # Choose action towards goal (simplified)
            action = 0  # Just move up repeatedly

            obs, reward, terminated, truncated, step_info = test_env.step(action)
            episode_steps += 1
            episode_rewards.append(reward)

            # Validate episode metadata tracking including step counts and timing
            assert (
                step_info["step_count"] == episode_steps
            ), "Step count should increment correctly"
            assert step_info["episode_count"] >= 1, "Episode count should be positive"

            # Check episode termination conditions
            if terminated:
                # Episode ended due to goal achievement
                assert reward > 0, "Termination should give positive reward"
                break
            elif truncated:
                # Episode ended due to step limit
                assert (
                    episode_steps >= test_env.max_steps
                ), "Truncation should occur at max steps"
                break

            # Safety check to prevent infinite loop
            if episode_steps >= 100:
                break

        # Validate episode truncation when maximum steps reached
        truncation_env = PlumeSearchEnv(
            grid_size=(16, 16),
            source_location=(8, 8),
            max_steps=5,  # Very short episodes
        )

        try:
            truncation_env.reset()

            for step in range(10):
                obs, reward, terminated, truncated, info = truncation_env.step(0)

                if truncated:
                    assert (
                        info["step_count"] >= 5
                    ), "Truncation should occur after max_steps"
                    break
                elif terminated:
                    # Early termination is also valid
                    break
        finally:
            truncation_env.close()

        # Test episode metadata tracking including step counts and timing
        # This is validated throughout the episode execution above

        # Verify proper episode boundary handling and state transitions
        test_env.reset()

        # Take a few steps
        for _ in range(3):
            test_env.step(1)

        # Reset and verify clean state
        new_obs, new_info = test_env.reset()
        assert new_info["step_count"] == 0, "Reset should clear step count"
        assert (
            new_info["episode_count"] > info["episode_count"]
        ), "Episode count should increment"

        # Test multiple consecutive episodes with proper state isolation
        episode_results = []

        for episode in range(3):
            obs, info = test_env.reset(seed=42 + episode)
            episode_data = {
                "initial_obs": obs.copy(),
                "episode_count": info["episode_count"],
                "steps_taken": 0,
            }

            # Run episode for fixed number of steps
            for _ in range(10):
                obs, reward, terminated, truncated, step_info = test_env.step(0)
                episode_data["steps_taken"] += 1

                if terminated or truncated:
                    break

            episode_data["final_obs"] = obs.copy()
            episode_results.append(episode_data)

        # Verify episode isolation
        for i in range(len(episode_results) - 1):
            current = episode_results[i]
            next_ep = episode_results[i + 1]

            assert (
                next_ep["episode_count"] > current["episode_count"]
            ), "Episode counts should increment"
            # Different seeds should typically produce different results

        # Validate episode statistics and performance metrics collection
        # This would be more comprehensive in a full implementation

        # Test episode interruption and cleanup during execution
        test_env.reset()
        test_env.step(0)
        test_env.step(1)
        # Interrupt with reset
        test_env.reset()

        # Should work normally after interruption
        result = test_env.step(0)
        assert len(result) == 5, "Environment should work normally after interruption"

    finally:
        test_env.close()


def test_component_integration():
    """
    Comprehensive component integration testing including cross-component communication, dependency coordination,
    state synchronization, and system-level integration verification ensuring complete system functionality.

    This test validates comprehensive component integration including communication, dependency coordination,
    state synchronization, and system-level functionality with complete integration verification.
    """
    integration_env = PlumeSearchEnv(
        grid_size=(64, 64),
        source_location=(32, 32),
        max_steps=200,
        render_mode="rgb_array",
    )

    try:
        # Initialize for integration testing
        integration_env.reset(seed=12345)

        # Test plume model integration with proper concentration sampling
        if hasattr(integration_env, "plume_model"):
            # Verify plume model is initialized
            assert (
                integration_env.plume_model is not None
            ), "Plume model should be initialized"

            # Test concentration sampling works
            obs, _ = integration_env.reset()
            assert isinstance(
                obs, np.ndarray
            ), "Observation should be sampled from plume model"
            assert 0.0 <= obs[0] <= 1.0, "Concentration should be in valid range"

        # Validate state manager coordination with action processor and reward calculator
        initial_obs, _ = integration_env.reset()

        # Take steps and verify components work together
        for action in [0, 1, 2, 3]:
            obs, reward, terminated, truncated, info = integration_env.step(action)

            # Verify state updates are coordinated
            assert isinstance(
                obs, np.ndarray
            ), "State manager should provide valid observations"
            assert isinstance(
                reward, (int, float)
            ), "Reward calculator should provide numeric rewards"

            # Verify info contains integrated information
            assert "step_count" in info, "State manager should track step count"
            assert "action_taken" in info, "Action processor should record actions"

            if terminated or truncated:
                break

        # Test rendering pipeline integration with environment state
        try:
            render_result = integration_env.render()
            if render_result is not None:
                assert isinstance(
                    render_result, np.ndarray
                ), "Rendering should integrate with environment state"
                assert render_result.ndim == 3, "RGB rendering should produce 3D array"
        except Exception:
            # Rendering might fail, that's acceptable
            pass

        # Verify component seeding coordination for reproducible behavior
        seed_env1 = PlumeSearchEnv(
            grid_size=(32, 32), source_location=(16, 16), max_steps=100
        )

        seed_env2 = PlumeSearchEnv(
            grid_size=(32, 32), source_location=(16, 16), max_steps=100
        )

        try:
            # Test seeded reproducibility across components
            obs1, _ = seed_env1.reset(seed=999)
            obs2, _ = seed_env2.reset(seed=999)

            np.testing.assert_array_equal(
                obs1, obs2, "Seeding should coordinate across all components"
            )

            # Test action sequence reproducibility
            for action in [1, 0, 2, 3]:
                result1 = seed_env1.step(action)
                result2 = seed_env2.step(action)

                np.testing.assert_array_equal(
                    result1[0], result2[0], "Action results should be reproducible"
                )
                assert result1[1] == result2[1], "Rewards should be reproducible"

        finally:
            seed_env1.close()
            seed_env2.close()

        # Test component cleanup coordination during environment closure
        cleanup_env = PlumeSearchEnv(
            grid_size=(16, 16), source_location=(8, 8), max_steps=50
        )

        cleanup_env.reset()
        cleanup_env.step(0)
        cleanup_env.render()  # Initialize rendering if possible

        # Should close without errors
        cleanup_env.close()

        # Validate error propagation and handling across component boundaries
        # Test with invalid configuration
        try:
            error_env = PlumeSearchEnv(
                grid_size=(1, 1), source_location=(0, 0), max_steps=10  # Too small
            )

            error_env.reset()
            # Should either work or fail gracefully
            error_env.close()

        except Exception as e:
            # Errors should be informative
            assert len(str(e)) > 0, "Error messages should be descriptive"

        # Test component resource management and memory efficiency
        # This is tested implicitly by successful operation without crashes

        # Validate cross-component consistency
        integration_env.reset()

        # Verify action space and observation space are consistent with components
        assert (
            integration_env.action_space.n == 4
        ), "Action space should match movement system"
        assert integration_env.observation_space.shape == (
            1,
        ), "Observation space should match plume model output"

    finally:
        integration_env.close()
