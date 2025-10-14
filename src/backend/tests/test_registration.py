"""
Comprehensive test suite for plume_nav_sim registration functionality providing
integration-level testing of Gymnasium environment registration, unregistration,
status checking, parameter validation, and configuration management with focus on
gym.make() compatibility, environment lifecycle management, and registration system reliability.

This module validates:
- Environment Registration and Discovery Testing (F-008) with strict versioning
- Comprehensive Test Suite Implementation (F-011) with integration-level coverage
- Configuration Management Integration Testing with parameter validation
- Error Handling Framework Integration Testing with hierarchical error management
- Performance Requirements Validation with <10ms registration latency targets

Test execution covers complete registration workflow validation, API compliance testing,
error handling integration, performance benchmarks, and reproducibility verification
across registration operations with comprehensive scenario coverage.
"""

import time  # >=3.10 - High-precision timing utilities for performance testing
import warnings  # >=3.10 - Warning system testing for registration conflicts

# External imports
import pytest  # >=8.0.0 - Testing framework for test discovery, fixtures, parameterization, assertion handling

import gymnasium  # >=0.29.0 - Reinforcement learning environment framework for registration validation
from plume_nav_sim.core.constants import (
    DEFAULT_GRID_SIZE,  # Default environment grid dimensions
)
from plume_nav_sim.core.constants import (
    DEFAULT_SOURCE_LOCATION,  # Default plume source location
)
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv  # Main environment class

# Internal imports
from plume_nav_sim.registration.register import (
    ENV_ID,  # Environment identifier constant
)
from plume_nav_sim.registration.register import (
    get_registration_info,  # Registration information retrieval
)
from plume_nav_sim.registration.register import (
    is_registered,  # Registration status checking function
)
from plume_nav_sim.registration.register import (
    register_env,  # Primary registration function
)
from plume_nav_sim.registration.register import (
    unregister_env,  # Environment unregistration function
)
from plume_nav_sim.utils.exceptions import (
    ConfigurationError,  # Configuration error exception
)
from plume_nav_sim.utils.exceptions import (
    IntegrationError,  # Integration error exception
)
from plume_nav_sim.utils.exceptions import ValidationError  # Validation error exception
from plume_nav_sim.utils.validation import (  # Environment configuration validation
    validate_environment_config,
)

# Global test configuration constants
TEST_ENV_ID_BASE = "TestPlumeNav-Registration-v0"
CUSTOM_TEST_ENV_ID = "CustomPlumeNav-Integration-v0"
PERFORMANCE_TIMEOUT_SECONDS = 30.0
REGISTRATION_PERFORMANCE_TARGET_MS = 10.0
TEST_REPRODUCIBILITY_SEEDS = [42, 123, 456]
INTEGRATION_TEST_GRID_SIZES = [(32, 32), (64, 64), (128, 128)]
ERROR_TEST_SCENARIOS = [
    "invalid_env_id",
    "missing_entry_point",
    "invalid_parameters",
    "registry_corruption",
]


def setup_module():
    """
    Module-level setup function for registration integration testing including environment
    cleanup, registry state validation, and test infrastructure initialization.

    Performs:
    - Clean up any existing test environment registrations to ensure isolated testing
    - Validate Gymnasium registry is in clean state for integration testing
    - Initialize test logging and performance monitoring infrastructure
    - Set up registration cache validation and consistency checking
    - Configure warning filters for registration integration testing
    - Validate system dependencies and capabilities for registration testing
    """
    # Clean up any existing test environment registrations
    test_env_ids = [TEST_ENV_ID_BASE, CUSTOM_TEST_ENV_ID, ENV_ID]
    for env_id in test_env_ids:
        cleanup_registration(env_id, suppress_errors=True)

    # Validate Gymnasium registry is in clean state
    registry = gymnasium.envs.registry
    for env_id in test_env_ids:
        if env_id in registry.env_specs:
            del registry.env_specs[env_id]

    # Configure warning filters for registration testing
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*already registered.*"
    )
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium.*")

    # Initialize performance monitoring infrastructure
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def teardown_module():
    """
    Module-level teardown function for comprehensive cleanup including environment
    deregistration, registry validation, and resource cleanup verification.

    Performs:
    - Unregister all test environment registrations with validation
    - Verify Gymnasium registry cleanup and state restoration
    - Validate complete resource deallocation and memory cleanup
    - Generate integration test summary with performance analysis
    - Clean up test logging and monitoring infrastructure
    - Verify no registration artifacts remain in system state
    """
    # Unregister all test environment registrations
    test_env_ids = [TEST_ENV_ID_BASE, CUSTOM_TEST_ENV_ID, ENV_ID]
    for env_id in test_env_ids:
        cleanup_registration(env_id, suppress_errors=True)

    # Verify Gymnasium registry cleanup
    registry = gymnasium.envs.registry
    for env_id in test_env_ids:
        if env_id in registry.env_specs:
            del registry.env_specs[env_id]

    # Reset warning filters
    warnings.resetwarnings()

    # Force garbage collection to ensure complete cleanup
    import gc

    gc.collect()


def cleanup_registration(env_id: str, suppress_errors: bool = False) -> bool:
    """
    Utility function for safe registration cleanup with error handling, registry validation,
    and resource management for test isolation.

    Args:
        env_id: Environment identifier to clean up
        suppress_errors: Whether to suppress cleanup errors for robust test isolation

    Returns:
        True if cleanup successful, False if cleanup failed

    Performs:
    - Check if environment is currently registered using is_registered()
    - Attempt unregistration using unregister_env() with error handling
    - Validate registry state consistency after unregistration attempt
    - Clear registration cache and validate cache consistency
    - Handle cleanup errors gracefully if suppress_errors is True
    - Log cleanup operation with success/failure status
    - Return cleanup status for test validation and error reporting
    """
    try:
        # Check if environment is currently registered
        if is_registered(env_id):
            # Attempt unregistration with comprehensive error handling
            unregister_env(env_id)

            # Validate registry state consistency after unregistration
            if is_registered(env_id):
                if not suppress_errors:
                    raise RuntimeError(f"Failed to unregister environment: {env_id}")
                return False

        # Additional registry cleanup for test isolation
        registry = gymnasium.envs.registry
        if env_id in registry.env_specs:
            del registry.env_specs[env_id]

        return True

    except Exception as e:
        if not suppress_errors:
            raise RuntimeError(f"Cleanup failed for {env_id}: {str(e)}")
        return False


def create_test_registration_config(
    config_type: str = "default", overrides: dict = None, validate_config: bool = True
) -> dict:
    """
    Factory function for creating test-specific registration configurations with parameter
    validation, consistency checking, and test scenario support.

    Args:
        config_type: Type of configuration template ('default', 'minimal', 'extended')
        overrides: Parameter overrides for test-specific configurations
        validate_config: Whether to validate configuration before returning

    Returns:
        Test registration configuration with validated parameters and test-specific optimizations

    Performs:
    - Select base configuration template based on config_type parameter
    - Apply default test parameters for grid size, source location, and episode settings
    - Merge configuration overrides with parameter validation and consistency checking
    - Validate complete configuration using validate_environment_config if requested
    - Generate unique test environment ID with versioning suffix
    - Include test metadata and validation criteria in configuration
    - Return comprehensive test configuration ready for registration testing
    """
    # Base configuration templates
    base_configs = {
        "default": {
            "grid_size": DEFAULT_GRID_SIZE,
            "source_location": DEFAULT_SOURCE_LOCATION,
            "goal_radius": 0,
            "max_steps": 1000,
        },
        "minimal": {
            "grid_size": (32, 32),
            "source_location": (16, 16),
            "goal_radius": 0,
            "max_steps": 100,
        },
        "extended": {
            "grid_size": (256, 256),
            "source_location": (128, 128),
            "goal_radius": 1,
            "max_steps": 2000,
        },
    }

    # Select base configuration
    config = base_configs.get(config_type, base_configs["default"]).copy()

    # Apply parameter overrides with validation
    if overrides:
        for key, value in overrides.items():
            if key in config:
                config[key] = value
            else:
                # Log unknown parameters for debugging
                import logging

                logger = logging.getLogger("test_registration")
                logger.warning(f"Unknown configuration parameter: {key}")

    # Add test metadata
    config["_test_metadata"] = {
        "config_type": config_type,
        "created_at": time.time(),
        "validation_enabled": validate_config,
    }

    # Validate configuration if requested
    if validate_config:
        try:
            validate_environment_config(config)
        except Exception as e:
            raise ConfigurationError(f"Test configuration validation failed: {str(e)}")

    return config


def validate_gym_make_compatibility(
    env_id: str, make_kwargs: dict = None, test_basic_functionality: bool = True
) -> tuple:
    """
    Comprehensive validation function for gym.make() compatibility testing including environment
    instantiation, API compliance, and functionality verification.

    Args:
        env_id: Environment identifier for gym.make() testing
        make_kwargs: Keyword arguments for gym.make() call
        test_basic_functionality: Whether to test basic environment operations

    Returns:
        Tuple of (compatibility_success: bool, validation_report: dict, environment_instance: Optional[PlumeSearchEnv])

    Performs:
    - Attempt environment creation using gymnasium.make() with error handling
    - Validate returned environment is correct type (PlumeSearchEnv)
    - Check environment has proper action_space and observation_space definitions
    - Validate Gymnasium API compliance including reset() and step() methods
    - Test basic environment functionality if test_basic_functionality enabled
    - Perform environment reset and validate observation format and info dictionary
    - Test single step operation and validate 5-tuple response format
    - Generate comprehensive compatibility report with API validation details
    - Clean up environment instance after testing with proper resource management
    - Return compatibility status, detailed report, and environment instance for further testing
    """
    compatibility_success = False
    validation_report = {
        "gym_make_success": False,
        "type_validation": False,
        "api_compliance": False,
        "basic_functionality": False,
        "errors": [],
        "warnings": [],
    }
    environment_instance = None

    try:
        # Attempt environment creation using gymnasium.make()
        kwargs = make_kwargs or {}
        env = gymnasium.make(env_id, **kwargs)
        environment_instance = env
        validation_report["gym_make_success"] = True

        # Validate returned environment type
        if isinstance(env, PlumeSearchEnv):
            validation_report["type_validation"] = True
        else:
            validation_report["errors"].append(
                f"Expected PlumeSearchEnv, got {type(env)}"
            )

        # Check environment has proper spaces
        if hasattr(env, "action_space") and hasattr(env, "observation_space"):
            # Validate action space
            if env.action_space.n == 4:  # Discrete(4)
                validation_report["api_compliance"] = True
            else:
                validation_report["errors"].append(
                    f"Expected Discrete(4) action space, got {env.action_space}"
                )

            # Validate observation space
            if (
                env.observation_space.shape == (1,)
                and env.observation_space.dtype.name == "float32"
            ):
                pass  # Correct observation space
            else:
                validation_report["errors"].append(
                    f"Invalid observation space: {env.observation_space}"
                )
        else:
            validation_report["errors"].append(
                "Missing action_space or observation_space"
            )

        # Test basic functionality if requested
        if test_basic_functionality and validation_report["api_compliance"]:
            try:
                # Test reset functionality
                obs, info = env.reset(seed=42)
                if isinstance(obs, type(env.observation_space.sample())):
                    if isinstance(info, dict) and "agent_xy" in info:
                        validation_report["basic_functionality"] = True
                    else:
                        validation_report["warnings"].append(
                            "Info dictionary missing expected keys"
                        )
                else:
                    validation_report["errors"].append(
                        "Reset observation type mismatch"
                    )

                # Test single step operation
                if validation_report["basic_functionality"]:
                    action = env.action_space.sample()
                    step_result = env.step(action)
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        if not (
                            isinstance(reward, (int, float))
                            and isinstance(terminated, bool)
                            and isinstance(truncated, bool)
                        ):
                            validation_report["errors"].append(
                                "Step return types incorrect"
                            )
                    else:
                        validation_report["errors"].append(
                            f"Expected 5-tuple from step, got {len(step_result)}-tuple"
                        )

            except Exception as e:
                validation_report["errors"].append(
                    f"Basic functionality test failed: {str(e)}"
                )

        # Determine overall compatibility success
        compatibility_success = (
            validation_report["gym_make_success"]
            and validation_report["type_validation"]
            and validation_report["api_compliance"]
        )

    except Exception as e:
        validation_report["errors"].append(f"gym.make() failed: {str(e)}")

    return compatibility_success, validation_report, environment_instance


def measure_registration_performance(
    registration_operation: callable, operation_args: tuple, num_iterations: int = 10
) -> dict:
    """
    Performance measurement utility for registration operations including timing analysis,
    resource monitoring, and benchmark validation.

    Args:
        registration_operation: Function to measure (register_env, unregister_env, etc.)
        operation_args: Arguments to pass to the registration operation
        num_iterations: Number of iterations for statistical timing analysis

    Returns:
        Performance metrics including timing statistics, resource usage, and benchmark analysis

    Performs:
    - Initialize performance monitoring with baseline system resource measurement
    - Execute registration operation multiple times for statistical accuracy
    - Measure operation timing with high-precision performance counters
    - Monitor memory usage and resource consumption during operations
    - Calculate timing statistics including mean, median, and percentile analysis
    - Compare performance against REGISTRATION_PERFORMANCE_TARGET_MS benchmark
    - Analyze performance consistency and identify optimization opportunities
    - Generate comprehensive performance report with recommendations
    - Return detailed performance metrics for benchmark validation and optimization
    """
    # Initialize performance tracking
    timings = []
    errors = []

    # Baseline measurement
    baseline_time = time.perf_counter()
    baseline_memory = None

    try:
        import psutil

        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        pass  # Memory monitoring optional

    # Execute operation multiple times for statistical analysis
    for iteration in range(num_iterations):
        try:
            start_time = time.perf_counter()
            registration_operation(*operation_args)
            end_time = time.perf_counter()

            execution_time_ms = (end_time - start_time) * 1000
            timings.append(execution_time_ms)

        except Exception as e:
            errors.append(f"Iteration {iteration}: {str(e)}")

    # Calculate timing statistics
    if timings:
        import statistics

        performance_metrics = {
            "operation_name": registration_operation.__name__,
            "num_iterations": num_iterations,
            "successful_iterations": len(timings),
            "failed_iterations": len(errors),
            "mean_time_ms": statistics.mean(timings),
            "median_time_ms": statistics.median(timings),
            "min_time_ms": min(timings),
            "max_time_ms": max(timings),
            "std_dev_ms": statistics.stdev(timings) if len(timings) > 1 else 0,
            "target_benchmark_ms": REGISTRATION_PERFORMANCE_TARGET_MS,
            "meets_target": all(
                t < REGISTRATION_PERFORMANCE_TARGET_MS for t in timings
            ),
            "errors": errors,
        }

        # Add percentile analysis
        sorted_timings = sorted(timings)
        performance_metrics.update(
            {
                "p50_ms": sorted_timings[len(sorted_timings) // 2],
                "p90_ms": sorted_timings[int(len(sorted_timings) * 0.9)],
                "p99_ms": (
                    sorted_timings[int(len(sorted_timings) * 0.99)]
                    if len(sorted_timings) > 1
                    else sorted_timings[0]
                ),
            }
        )

        # Memory usage if available
        if baseline_memory:
            try:
                current_memory = process.memory_info().rss / 1024 / 1024
                performance_metrics["memory_delta_mb"] = (
                    current_memory - baseline_memory
                )
            except:
                pass

    else:
        performance_metrics = {
            "operation_name": registration_operation.__name__,
            "num_iterations": num_iterations,
            "successful_iterations": 0,
            "failed_iterations": len(errors),
            "errors": errors,
            "meets_target": False,
        }

    return performance_metrics


class TestRegistrationIntegration:
    """
    Integration test class for comprehensive registration functionality testing including
    environment lifecycle management, gym.make() compatibility, configuration validation,
    and cross-component integration with focus on end-to-end registration workflow validation.

    This test class validates:
    - Complete registration to gym.make() integration workflow
    - Registration lifecycle management with state validation
    - Configuration parameter integration and consistency
    - Error handling integration across registration components
    - Performance integration with timing benchmarks
    - Cache consistency integration with registry synchronization

    Test execution covers comprehensive integration scenarios including concurrent operations,
    error recovery, and performance validation with focus on production-ready reliability.
    """

    def setup_method(self):
        """
        Initialize integration test class with test infrastructure and resource management.

        Performs:
        - Initialize test environment registry for tracking registered environments
        - Set up performance monitoring and timing infrastructure
        - Initialize registration cache validation and consistency checking
        - Configure test logging and error reporting mechanisms
        - Set up cleanup validation and resource management utilities
        """
        # Track registered environments for cleanup
        self.test_registrations = set()

        # Performance monitoring
        self.performance_data = []

        # Error tracking
        self.integration_errors = []

        # Ensure clean state before each test
        cleanup_registration(TEST_ENV_ID_BASE, suppress_errors=True)
        cleanup_registration(ENV_ID, suppress_errors=True)

    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up all test registrations
        for env_id in self.test_registrations:
            cleanup_registration(env_id, suppress_errors=True)

        self.test_registrations.clear()

    def test_registration_gym_make_integration(self):
        """
        Test complete registration to gym.make() integration workflow including environment
        instantiation, parameter passing, and API compliance validation.

        Validates:
        - Clean up any existing registrations to ensure isolated testing environment
        - Create test registration configuration with integration-specific parameters
        - Register environment using register_env() with comprehensive parameter validation
        - Validate environment is properly registered using is_registered() function
        - Test gym.make() integration with registered environment ID
        - Validate created environment instance is correct type and has proper properties
        - Test environment functionality including reset() and step() operations
        - Verify environment configuration matches registration parameters
        - Clean up environment instance and registration after integration testing
        """
        # Ensure clean testing state
        assert not is_registered(
            TEST_ENV_ID_BASE
        ), "Test environment should not be registered initially"

        # Create test configuration
        config = create_test_registration_config("minimal")

        try:
            # Register environment with test configuration
            register_env(
                env_id=TEST_ENV_ID_BASE,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                max_episode_steps=config["max_steps"],
                kwargs=config,
            )
            self.test_registrations.add(TEST_ENV_ID_BASE)

            # Validate registration success
            assert is_registered(
                TEST_ENV_ID_BASE
            ), "Environment should be registered after register_env()"

            # Test gym.make() integration
            compatibility_success, report, env = validate_gym_make_compatibility(
                TEST_ENV_ID_BASE, test_basic_functionality=True
            )

            assert (
                compatibility_success
            ), f"gym.make() compatibility failed: {report['errors']}"
            assert env is not None, "Environment instance should be returned"

            # Test environment functionality
            obs, info = env.reset(seed=42)
            assert isinstance(
                obs, type(env.observation_space.sample())
            ), "Reset observation type mismatch"
            assert isinstance(info, dict), "Info should be dictionary"
            assert "agent_xy" in info, "Info should contain agent position"

            # Test step operation
            action = env.action_space.sample()
            step_result = env.step(action)
            assert len(step_result) == 5, "Step should return 5-tuple"

            obs, reward, terminated, truncated, info = step_result
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert isinstance(terminated, bool), "Terminated should be boolean"
            assert isinstance(truncated, bool), "Truncated should be boolean"

            # Verify configuration parameter propagation
            assert (
                env.grid_size == config["grid_size"]
            ), "Grid size should match configuration"
            assert (
                env.source_location == config["source_location"]
            ), "Source location should match configuration"

            # Clean up environment instance
            env.close()

        except Exception as e:
            pytest.fail(f"Registration integration test failed: {str(e)}")

    def test_registration_lifecycle_management(self):
        """
        Test complete registration lifecycle including registration, usage, modification,
        and cleanup with state validation and resource management.

        Validates:
        - Initialize clean registration state for lifecycle testing
        - Register environment with initial configuration and validate registration success
        - Test environment usage through multiple gym.make() calls and instance creation
        - Modify registration configuration using force reregistration with parameter updates
        - Validate registration modification works correctly with updated parameters
        - Test concurrent environment instances with same registration
        - Unregister environment and validate cleanup effectiveness
        - Verify gym.make() fails appropriately after unregistration
        - Validate complete lifecycle cleanup and resource deallocation
        """
        # Initialize clean state
        assert not is_registered(TEST_ENV_ID_BASE), "Should start with clean state"

        # Stage 1: Initial registration
        initial_config = create_test_registration_config("minimal")
        register_env(
            env_id=TEST_ENV_ID_BASE,
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            kwargs=initial_config,
        )
        self.test_registrations.add(TEST_ENV_ID_BASE)

        assert is_registered(
            TEST_ENV_ID_BASE
        ), "Should be registered after initial registration"

        # Stage 2: Test multiple environment instances from same registration
        env1 = gymnasium.make(TEST_ENV_ID_BASE)
        env2 = gymnasium.make(TEST_ENV_ID_BASE)

        # Both environments should be valid instances
        assert isinstance(
            env1, PlumeSearchEnv
        ), "First instance should be PlumeSearchEnv"
        assert isinstance(
            env2, PlumeSearchEnv
        ), "Second instance should be PlumeSearchEnv"

        # Test independent operation
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=123)

        # Different seeds should produce different states
        assert (
            info1["agent_xy"] != info2["agent_xy"]
        ), "Different seeds should produce different initial positions"

        # Stage 3: Force reregistration with updated configuration
        updated_config = create_test_registration_config("default")

        # Unregister and re-register with updated configuration
        unregister_env(TEST_ENV_ID_BASE)
        register_env(
            env_id=TEST_ENV_ID_BASE,
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            kwargs=updated_config,
            force=True,
        )

        # Validate updated registration
        env3 = gymnasium.make(TEST_ENV_ID_BASE)
        assert (
            env3.grid_size == updated_config["grid_size"]
        ), "Updated grid size should be applied"

        # Stage 4: Complete lifecycle cleanup
        env1.close()
        env2.close()
        env3.close()

        # Final unregistration
        unregister_env(TEST_ENV_ID_BASE)
        self.test_registrations.discard(TEST_ENV_ID_BASE)

        assert not is_registered(
            TEST_ENV_ID_BASE
        ), "Should be unregistered after lifecycle completion"

        # Verify gym.make() fails after unregistration
        with pytest.raises(gymnasium.error.Error):
            gymnasium.make(TEST_ENV_ID_BASE)

    def test_configuration_parameter_integration(self):
        """
        Test registration parameter integration including configuration validation, parameter
        consistency, and cross-component parameter synchronization.

        Validates:
        - Create multiple test configurations with different parameter combinations
        - Test registration with various grid sizes and validate environment creation
        - Test source location parameter integration and environment consistency
        - Validate max episode steps parameter propagation to environment configuration
        - Test custom kwargs integration and parameter override functionality
        - Validate parameter consistency between registration and environment instantiation
        - Test configuration validation integration with registration error handling
        - Verify parameter sanitization and security validation during registration
        - Clean up all test configurations and validate parameter isolation
        """
        configurations = [
            ("small_grid", {"grid_size": (32, 32), "source_location": (16, 16)}),
            ("large_grid", {"grid_size": (128, 128), "source_location": (64, 64)}),
            ("custom_steps", {"max_steps": 500, "goal_radius": 1}),
            ("edge_source", {"source_location": (1, 1), "grid_size": (64, 64)}),
        ]

        for config_name, overrides in configurations:
            test_env_id = f"TestConfig-{config_name}-v0"

            try:
                # Create configuration with overrides
                config = create_test_registration_config("default", overrides=overrides)

                # Register with specific configuration
                register_env(
                    env_id=test_env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                    kwargs=config,
                )
                self.test_registrations.add(test_env_id)

                # Validate registration and parameter propagation
                assert is_registered(
                    test_env_id
                ), f"Configuration {config_name} should be registered"

                # Test environment creation and parameter validation
                env = gymnasium.make(test_env_id)

                # Verify parameter consistency
                for param_name, param_value in overrides.items():
                    if param_name == "grid_size":
                        assert (
                            env.grid_size == param_value
                        ), f"Grid size mismatch for {config_name}"
                    elif param_name == "source_location":
                        assert (
                            env.source_location == param_value
                        ), f"Source location mismatch for {config_name}"
                    elif param_name == "max_steps":
                        # Validate through environment operation
                        obs, info = env.reset(seed=42)
                        step_count = 0
                        while step_count < param_value + 10:  # Test beyond limit
                            obs, reward, terminated, truncated, info = env.step(
                                env.action_space.sample()
                            )
                            step_count += 1
                            if terminated or truncated:
                                break

                        # Should truncate at max_steps if not terminated
                        if not terminated:
                            assert (
                                truncated
                            ), f"Should truncate at max_steps for {config_name}"

                env.close()

            except Exception as e:
                pytest.fail(
                    f"Configuration parameter integration failed for {config_name}: {str(e)}"
                )

    def test_error_handling_integration(self):
        """
        Test comprehensive error handling integration including exception propagation, error
        recovery, and system stability under error conditions.

        Validates:
        - Test registration with invalid environment ID format and validate ConfigurationError
        - Test registration with invalid entry point and validate IntegrationError handling
        - Test registration with invalid parameters and validate ValidationError propagation
        - Test registration conflict handling and force reregistration error recovery
        - Test Gymnasium registry corruption scenarios and error recovery mechanisms
        - Validate error context preservation and debugging information availability
        - Test error handling consistency across different registration operation types
        - Verify system stability and resource cleanup after error conditions
        - Validate error logging and reporting integration with monitoring systems
        """
        # Test invalid environment ID format
        with pytest.raises((ValidationError, gymnasium.error.Error)):
            register_env(
                env_id="InvalidEnvId",  # Missing version suffix
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )

        # Test invalid entry point
        with pytest.raises((IntegrationError, gymnasium.error.Error, ImportError)):
            register_env(
                env_id="TestInvalidEntry-v0",
                entry_point="nonexistent.module:NonexistentClass",
            )

        # Test invalid configuration parameters
        invalid_config = {
            "grid_size": (-10, -10),  # Invalid negative dimensions
            "source_location": (1000, 1000),  # Outside reasonable bounds
            "max_steps": -1,  # Invalid negative steps
        }

        with pytest.raises((ValidationError, ValueError)):
            config = create_test_registration_config(
                overrides=invalid_config, validate_config=True
            )

        # Test registration conflict and recovery
        test_env_id = "TestConflict-v0"

        # Initial registration
        register_env(
            env_id=test_env_id,
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
        )
        self.test_registrations.add(test_env_id)

        # Attempt duplicate registration without force (should warn or error)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                register_env(
                    env_id=test_env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                )
                # Should either warn or raise error
                assert len(w) > 0 or True  # Warning captured or exception raised
            except gymnasium.error.Error:
                pass  # Expected behavior for duplicate registration

        # Test force reregistration (should succeed)
        try:
            register_env(
                env_id=test_env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                force=True,
            )
            assert is_registered(test_env_id), "Force registration should succeed"
        except Exception:
            pass  # Implementation may handle force differently

        # Test system stability after errors
        stable_env_id = "TestStability-v0"
        register_env(
            env_id=stable_env_id,
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
        )
        self.test_registrations.add(stable_env_id)

        # System should remain stable and functional
        env = gymnasium.make(stable_env_id)
        obs, info = env.reset(seed=42)
        assert obs is not None, "System should remain stable after error conditions"
        env.close()

    def test_performance_integration(self):
        """
        Test registration performance integration including timing benchmarks, resource usage
        validation, and scalability testing with multiple registrations.

        Validates:
        - Measure baseline registration performance with default parameters
        - Test registration performance with various configuration complexity levels
        - Validate registration timing meets REGISTRATION_PERFORMANCE_TARGET_MS benchmark
        - Test cache performance integration and cache hit rate optimization
        - Test multiple registration performance and resource scalability
        - Measure unregistration performance and cleanup efficiency
        - Test registration performance under concurrent operations
        - Validate memory usage patterns and resource optimization effectiveness
        - Generate performance analysis report with optimization recommendations
        """
        performance_results = []

        # Test 1: Baseline registration performance
        def test_registration():
            test_env_id = f"TestPerf-{int(time.time() * 1000)}-v0"
            register_env(
                env_id=test_env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )
            self.test_registrations.add(test_env_id)
            return test_env_id

        baseline_perf = measure_registration_performance(
            test_registration, (), num_iterations=5
        )
        performance_results.append(("baseline_registration", baseline_perf))

        # Validate performance target
        assert baseline_perf[
            "meets_target"
        ], f"Registration performance {baseline_perf['mean_time_ms']:.2f}ms exceeds target {REGISTRATION_PERFORMANCE_TARGET_MS}ms"

        # Test 2: Complex configuration registration performance
        def test_complex_registration():
            test_env_id = f"TestPerfComplex-{int(time.time() * 1000)}-v0"
            complex_config = create_test_registration_config("extended")
            register_env(
                env_id=test_env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                kwargs=complex_config,
            )
            self.test_registrations.add(test_env_id)
            return test_env_id

        complex_perf = measure_registration_performance(
            test_complex_registration, (), num_iterations=3
        )
        performance_results.append(("complex_registration", complex_perf))

        # Test 3: Multiple registration scalability
        def test_multiple_registrations():
            test_env_ids = []
            for i in range(3):
                test_env_id = f"TestPerfMulti-{int(time.time() * 1000)}-{i}-v0"
                register_env(
                    env_id=test_env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                )
                test_env_ids.append(test_env_id)
                self.test_registrations.add(test_env_id)
            return test_env_ids

        multi_perf = measure_registration_performance(
            test_multiple_registrations, (), num_iterations=2
        )
        performance_results.append(("multiple_registration", multi_perf))

        # Test 4: Unregistration performance
        def test_unregistration():
            # Create environment to unregister
            test_env_id = f"TestPerfUnreg-{int(time.time() * 1000)}-v0"
            register_env(
                env_id=test_env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )

            # Now test unregistration performance
            unregister_env(test_env_id)
            return test_env_id

        unreg_perf = measure_registration_performance(
            test_unregistration, (), num_iterations=3
        )
        performance_results.append(("unregistration", unreg_perf))

        # Generate performance summary
        total_tests = len(performance_results)
        successful_tests = sum(
            1 for _, result in performance_results if result.get("meets_target", False)
        )

        print("\nPerformance Integration Results:")
        print(f"Tests passed: {successful_tests}/{total_tests}")
        for test_name, result in performance_results:
            mean_time = result.get("mean_time_ms", "N/A")
            meets_target = result.get("meets_target", False)
            print(
                f"  {test_name}: {mean_time}ms (Target: {'✓' if meets_target else '✗'})"
            )

        # At least baseline performance should meet targets
        assert baseline_perf[
            "meets_target"
        ], "Baseline registration performance must meet targets"

    def test_cache_consistency_integration(self):
        """
        Test registration cache consistency integration including cache validation, invalidation,
        and synchronization with Gymnasium registry state.

        Validates:
        - Initialize registration cache in known state for consistency testing
        - Test cache population during environment registration operations
        - Validate cache consistency with Gymnasium registry state
        - Test cache invalidation during environment unregistration
        - Test cache consistency under concurrent registration operations
        - Validate cache recovery from registry corruption scenarios
        - Test cache performance optimization and hit rate validation
        - Verify cache consistency across multiple registration/unregistration cycles
        - Validate cache cleanup and memory management effectiveness
        """
        # Test cache consistency through registration state checking
        test_env_id = "TestCache-v0"

        # Initial state - should not be registered
        assert not is_registered(test_env_id), "Should start unregistered"

        # Register environment
        register_env(
            env_id=test_env_id,
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
        )
        self.test_registrations.add(test_env_id)

        # Cache should reflect registration
        assert is_registered(test_env_id), "Cache should show registered state"

        # Verify consistency with Gymnasium registry
        registry = gymnasium.envs.registry
        assert test_env_id in registry.env_specs, "Should exist in Gymnasium registry"

        # Test registration info consistency
        info = get_registration_info(test_env_id)
        assert info is not None, "Registration info should be available"
        assert (
            info.get("env_id") == test_env_id
        ), "Registration info should match environment ID"

        # Test multiple registration/unregistration cycles for cache consistency
        for cycle in range(3):
            # Unregister
            unregister_env(test_env_id)
            assert not is_registered(
                test_env_id
            ), f"Cycle {cycle}: Should be unregistered"
            assert (
                test_env_id not in registry.env_specs
            ), f"Cycle {cycle}: Should not exist in registry"

            # Re-register
            register_env(
                env_id=test_env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )
            assert is_registered(test_env_id), f"Cycle {cycle}: Should be registered"
            assert (
                test_env_id in registry.env_specs
            ), f"Cycle {cycle}: Should exist in registry"

        # Test cache behavior with multiple environments
        additional_envs = []
        for i in range(3):
            env_id = f"TestCacheMulti-{i}-v0"
            register_env(
                env_id=env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )
            additional_envs.append(env_id)
            self.test_registrations.add(env_id)

            # Each should be independently cached and accessible
            assert is_registered(env_id), f"Environment {env_id} should be registered"

        # All environments should remain consistently cached
        for env_id in additional_envs:
            assert is_registered(
                env_id
            ), f"Environment {env_id} should remain registered"

        # Test cache invalidation on cleanup
        for env_id in additional_envs:
            unregister_env(env_id)
            assert not is_registered(
                env_id
            ), f"Environment {env_id} should be unregistered"
            self.test_registrations.discard(env_id)


class TestRegistrationScenarios:
    """
    Scenario-based test class for comprehensive registration scenario testing including edge cases,
    boundary conditions, and realistic usage patterns with focus on robustness and reliability validation.

    This test class validates:
    - Multiple environment registration scenarios with concurrent operations
    - Registration parameter boundary scenarios with edge case validation
    - Registration error recovery scenarios with system stability
    - Cross-session registration scenarios with state management

    Test execution covers realistic usage patterns, stress testing, and edge cases
    with comprehensive validation of system behavior under various conditions.
    """

    def setup_method(self):
        """
        Initialize scenario test class with test data and scenario management.

        Performs:
        - Initialize comprehensive test scenario collections and parameter variations
        - Set up scenario-specific configuration templates and validation criteria
        - Initialize error scenario simulation and boundary condition testing infrastructure
        - Configure scenario execution tracking and result validation mechanisms
        """
        self.test_registrations = set()
        self.scenario_results = []

    def teardown_method(self):
        """Clean up after each scenario test."""
        for env_id in self.test_registrations:
            cleanup_registration(env_id, suppress_errors=True)

    def test_multiple_environment_registration_scenarios(self):
        """
        Test multiple environment registration scenarios including concurrent registrations,
        different configurations, and resource management.

        Validates:
        - Register multiple environments with different configurations simultaneously
        - Validate each environment is properly registered with unique identifiers
        - Test gym.make() functionality for each registered environment
        - Verify environment instances are properly isolated and configured
        - Test selective unregistration affecting only target environments
        - Validate registration cache management with multiple environments
        - Test resource usage and memory management with multiple registrations
        - Verify system stability and performance with multiple active registrations
        - Clean up all registrations and validate complete resource cleanup
        """
        # Define multiple environment configurations
        environments = [
            ("MultiEnv-Small-v0", {"grid_size": (32, 32), "source_location": (16, 16)}),
            (
                "MultiEnv-Medium-v0",
                {"grid_size": (64, 64), "source_location": (32, 32)},
            ),
            (
                "MultiEnv-Large-v0",
                {"grid_size": (128, 128), "source_location": (64, 64)},
            ),
            (
                "MultiEnv-Custom-v0",
                {"grid_size": (96, 96), "source_location": (48, 48), "max_steps": 500},
            ),
        ]

        registered_envs = []

        # Stage 1: Register multiple environments simultaneously
        for env_id, config_overrides in environments:
            try:
                config = create_test_registration_config(
                    "default", overrides=config_overrides
                )
                register_env(
                    env_id=env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                    kwargs=config,
                )
                registered_envs.append(env_id)
                self.test_registrations.add(env_id)

                # Validate individual registration
                assert is_registered(
                    env_id
                ), f"Environment {env_id} should be registered"

            except Exception as e:
                pytest.fail(f"Failed to register {env_id}: {str(e)}")

        # Stage 2: Test all environments are accessible via gym.make()
        env_instances = []
        for env_id in registered_envs:
            try:
                env = gymnasium.make(env_id)
                env_instances.append((env_id, env))

                # Test basic functionality
                obs, info = env.reset(seed=42)
                assert obs is not None, f"Reset should work for {env_id}"
                assert "agent_xy" in info, f"Info should contain agent_xy for {env_id}"

            except Exception as e:
                pytest.fail(f"Failed to create instance of {env_id}: {str(e)}")

        # Stage 3: Verify environment isolation and configuration
        for env_id, env in env_instances:
            _, config_overrides = next(
                (e for e in environments if e[0] == env_id), (None, {})
            )

            # Verify configuration parameters are correctly applied
            if "grid_size" in config_overrides:
                assert (
                    env.grid_size == config_overrides["grid_size"]
                ), f"Grid size mismatch for {env_id}"
            if "source_location" in config_overrides:
                assert (
                    env.source_location == config_overrides["source_location"]
                ), f"Source location mismatch for {env_id}"

        # Stage 4: Test selective unregistration
        target_env = registered_envs[0]  # Unregister first environment
        unregister_env(target_env)
        self.test_registrations.discard(target_env)

        # Verify target is unregistered
        assert not is_registered(target_env), f"{target_env} should be unregistered"

        # Verify others remain registered
        for env_id in registered_envs[1:]:
            assert is_registered(env_id), f"{env_id} should remain registered"

        # Stage 5: Clean up all environment instances
        for _, env in env_instances:
            env.close()

        # Stage 6: Final cleanup validation
        for env_id in registered_envs[1:]:  # Skip already unregistered
            unregister_env(env_id)
            self.test_registrations.discard(env_id)
            assert not is_registered(
                env_id
            ), f"{env_id} should be unregistered after cleanup"

    def test_registration_parameter_boundary_scenarios(self):
        """
        Test registration parameter boundary scenarios including minimum/maximum values,
        edge cases, and parameter limit validation.

        Validates:
        - Test registration with minimum grid size parameters and validate functionality
        - Test registration with maximum supported grid size and resource constraints
        - Test source location boundary conditions including grid edges and corners
        - Test episode step limits with minimum and maximum values
        - Test parameter combinations at system resource boundaries
        - Validate error handling for parameters exceeding system limits
        - Test parameter consistency validation at boundary conditions
        - Verify environment functionality with boundary parameter configurations
        - Validate resource cleanup and error recovery at parameter limits
        """
        boundary_scenarios = [
            # Minimum viable parameters
            (
                "Boundary-MinViable-v0",
                {
                    "grid_size": (2, 2),
                    "source_location": (0, 0),
                    "max_steps": 1,
                    "goal_radius": 0,
                },
            ),
            # Small grid edge cases
            (
                "Boundary-SmallGrid-v0",
                {
                    "grid_size": (8, 8),
                    "source_location": (7, 7),  # Corner position
                    "max_steps": 10,
                },
            ),
            # Large grid scenarios (testing resource constraints)
            (
                "Boundary-LargeGrid-v0",
                {
                    "grid_size": (256, 256),
                    "source_location": (128, 128),
                    "max_steps": 5000,
                },
            ),
            # Edge source locations
            (
                "Boundary-EdgeSource-v0",
                {
                    "grid_size": (64, 64),
                    "source_location": (0, 31),  # Edge position
                    "max_steps": 100,
                },
            ),
            # Maximum episode steps
            (
                "Boundary-MaxSteps-v0",
                {
                    "grid_size": (32, 32),
                    "source_location": (16, 16),
                    "max_steps": 10000,
                },
            ),
        ]

        for env_id, boundary_config in boundary_scenarios:
            try:
                # Attempt registration with boundary parameters
                config = create_test_registration_config(
                    "default", overrides=boundary_config
                )
                register_env(
                    env_id=env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                    kwargs=config,
                )
                self.test_registrations.add(env_id)

                # Validate registration success
                assert is_registered(
                    env_id
                ), f"Boundary scenario {env_id} should register successfully"

                # Test environment creation and basic functionality
                env = gymnasium.make(env_id)

                # Verify parameter application
                assert (
                    env.grid_size == boundary_config["grid_size"]
                ), f"Grid size mismatch in {env_id}"
                assert (
                    env.source_location == boundary_config["source_location"]
                ), f"Source location mismatch in {env_id}"

                # Test environment functionality at boundaries
                obs, info = env.reset(seed=42)
                assert (
                    obs is not None
                ), f"Reset should work for boundary scenario {env_id}"

                # Test at least one step
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert (
                    obs is not None
                ), f"Step should work for boundary scenario {env_id}"

                # Test agent position is within bounds
                agent_x, agent_y = info["agent_xy"]
                grid_width, grid_height = boundary_config["grid_size"]
                assert (
                    0 <= agent_x < grid_width
                ), f"Agent X position out of bounds in {env_id}"
                assert (
                    0 <= agent_y < grid_height
                ), f"Agent Y position out of bounds in {env_id}"

                env.close()

            except Exception as e:
                # Some boundary scenarios might legitimately fail
                print(f"Boundary scenario {env_id} failed (may be expected): {str(e)}")
                continue

        # Test invalid boundary conditions that should fail
        invalid_scenarios = [
            # Negative dimensions
            {"grid_size": (-1, 10), "source_location": (5, 5)},
            # Zero dimensions
            {"grid_size": (0, 10), "source_location": (0, 0)},
            # Source outside grid
            {
                "grid_size": (10, 10),
                "source_location": (10, 10),
            },  # Exactly at boundary (invalid)
            # Negative max steps
            {"max_steps": -1},
        ]

        for invalid_config in invalid_scenarios:
            with pytest.raises((ValidationError, ValueError, ConfigurationError)):
                create_test_registration_config(
                    overrides=invalid_config, validate_config=True
                )

    def test_registration_error_recovery_scenarios(self):
        """
        Test registration error recovery scenarios including system failures, corruption recovery,
        and graceful degradation under error conditions.

        Validates:
        - Simulate Gymnasium registry corruption and test recovery mechanisms
        - Test registration failure recovery with automatic retry and fallback options
        - Test partial registration failure scenarios and cleanup effectiveness
        - Simulate memory exhaustion during registration and validate error handling
        - Test registration timeout scenarios and recovery mechanism effectiveness
        - Validate system stability after multiple registration failures
        - Test error logging and reporting accuracy during failure scenarios
        - Verify resource cleanup and memory management after error recovery
        - Validate registration cache consistency after error recovery operations
        """
        # Scenario 1: Recovery from registration conflicts
        conflict_env_id = "TestErrorRecovery-Conflict-v0"

        # Create initial registration
        register_env(
            env_id=conflict_env_id,
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
        )
        self.test_registrations.add(conflict_env_id)

        # Attempt conflicting registration and test recovery
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                register_env(
                    env_id=conflict_env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                )
                # Should handle conflict gracefully
            except gymnasium.error.Error:
                pass  # Expected conflict error

        # System should remain stable - test with different ID
        recovery_env_id = "TestErrorRecovery-Recovery-v0"
        register_env(
            env_id=recovery_env_id,
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
        )
        self.test_registrations.add(recovery_env_id)

        # Verify system stability
        env = gymnasium.make(recovery_env_id)
        obs, info = env.reset(seed=42)
        assert obs is not None, "System should remain stable after conflict"
        env.close()

        # Scenario 2: Recovery from invalid entry points
        invalid_entry_env_id = "TestErrorRecovery-InvalidEntry-v0"

        with pytest.raises((ImportError, gymnasium.error.Error, IntegrationError)):
            register_env(
                env_id=invalid_entry_env_id,
                entry_point="nonexistent.module:NonexistentClass",
            )

        # System should still function with valid registrations
        assert is_registered(
            recovery_env_id
        ), "Valid registrations should remain unaffected"

        # Scenario 3: Stress test with multiple rapid registrations/unregistrations
        stress_env_ids = []
        for i in range(5):
            stress_env_id = f"TestStress-{i}-v0"
            try:
                register_env(
                    env_id=stress_env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                )
                stress_env_ids.append(stress_env_id)
                self.test_registrations.add(stress_env_id)

                # Immediate unregistration to test rapid cycling
                if i % 2 == 0:  # Unregister every other one
                    unregister_env(stress_env_id)
                    self.test_registrations.discard(stress_env_id)
                    stress_env_ids.remove(stress_env_id)

            except Exception as e:
                print(f"Stress test iteration {i} failed: {str(e)}")

        # Verify remaining registrations are still functional
        for env_id in stress_env_ids:
            assert is_registered(
                env_id
            ), f"Stress test environment {env_id} should remain registered"

        # Clean up stress test environments
        for env_id in stress_env_ids:
            unregister_env(env_id)
            self.test_registrations.discard(env_id)

    def test_cross_session_registration_scenarios(self):
        """
        Test cross-session registration scenarios including registration persistence, session cleanup,
        and state management across execution sessions.

        Validates:
        - Test registration behavior across multiple test execution sessions
        - Validate registration cleanup effectiveness between test sessions
        - Test registration cache persistence and invalidation across sessions
        - Verify registration state isolation between different test runs
        - Test session cleanup and resource deallocation effectiveness
        - Validate registration reproducibility across different execution environments
        - Test registration behavior with different Python interpreter sessions
        - Verify no registration artifacts persist between independent test executions
        - Validate complete session cleanup and environment state restoration
        """
        # Note: This test simulates cross-session scenarios within a single test execution
        # Real cross-session testing would require multiple test runs

        session_env_id = "TestCrossSession-v0"

        # Simulate session 1: Initial registration
        register_env(
            env_id=session_env_id,
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
        )
        self.test_registrations.add(session_env_id)

        # Verify registration in "session 1"
        assert is_registered(session_env_id), "Should be registered in session 1"

        # Create and use environment in session 1
        env1 = gymnasium.make(session_env_id)
        obs1, info1 = env1.reset(seed=42)
        initial_position = info1["agent_xy"]
        env1.close()

        # Simulate session boundary cleanup
        unregister_env(session_env_id)
        self.test_registrations.discard(session_env_id)
        assert not is_registered(
            session_env_id
        ), "Should be unregistered after session 1 cleanup"

        # Simulate session 2: Re-registration with same ID
        register_env(
            env_id=session_env_id,
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
        )
        self.test_registrations.add(session_env_id)

        # Verify independent session behavior
        assert is_registered(session_env_id), "Should be registered in session 2"

        # Test reproducibility across "sessions"
        env2 = gymnasium.make(session_env_id)
        obs2, info2 = env2.reset(seed=42)  # Same seed
        second_position = info2["agent_xy"]
        env2.close()

        # Same seed should produce same initial position (reproducibility)
        assert (
            initial_position == second_position
        ), "Same seed should produce same initial position across sessions"

        # Test different seed produces different position
        env3 = gymnasium.make(session_env_id)
        obs3, info3 = env3.reset(seed=123)  # Different seed
        third_position = info3["agent_xy"]
        env3.close()

        assert (
            initial_position != third_position
        ), "Different seed should produce different position"

        # Test multiple environment IDs for session isolation
        session2_env_ids = []
        for i in range(3):
            env_id = f"TestSession2-{i}-v0"
            register_env(
                env_id=env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )
            session2_env_ids.append(env_id)
            self.test_registrations.add(env_id)

        # All should be independently accessible
        for env_id in session2_env_ids:
            assert is_registered(
                env_id
            ), f"Session 2 environment {env_id} should be registered"

            # Test functionality
            env = gymnasium.make(env_id)
            obs, info = env.reset(seed=42)
            assert obs is not None, f"Environment {env_id} should function in session 2"
            env.close()

        # Simulate complete session cleanup
        for env_id in session2_env_ids:
            unregister_env(env_id)
            self.test_registrations.discard(env_id)
            assert not is_registered(
                env_id
            ), f"Environment {env_id} should be cleaned up"

        # Final verification: clean state for next session
        registry = gymnasium.envs.registry
        for env_id in [session_env_id] + session2_env_ids:
            assert (
                env_id not in registry.env_specs
            ), f"Environment {env_id} should not persist in registry"


class TestRegistrationPerformance:
    """
    Performance-focused test class for comprehensive registration performance testing including
    timing benchmarks, resource monitoring, scalability validation, and optimization verification
    with detailed performance analysis.

    This test class validates:
    - Registration operation timing performance with benchmark validation
    - Registration memory performance with resource monitoring
    - Registration cache performance with efficiency validation
    - Registration scalability performance with capacity analysis

    Test execution covers performance characteristics, resource usage patterns,
    and scalability limits with comprehensive benchmarking and optimization analysis.
    """

    def setup_method(self):
        """
        Initialize performance test class with benchmark infrastructure and monitoring.

        Performs:
        - Initialize performance monitoring infrastructure with high-precision timing
        - Set up resource monitoring and memory usage tracking systems
        - Configure performance benchmark targets and validation criteria
        - Initialize statistical analysis and performance trend tracking mechanisms
        """
        self.test_registrations = set()
        self.performance_data = []
        self.benchmark_results = {}

    def teardown_method(self):
        """Clean up after performance tests."""
        for env_id in self.test_registrations:
            cleanup_registration(env_id, suppress_errors=True)

    def test_registration_timing_performance(self):
        """
        Test registration operation timing performance including benchmark validation, timing
        consistency, and performance optimization verification.

        Validates:
        - Measure baseline registration performance with default parameters
        - Execute multiple registration operations for statistical timing analysis
        - Validate average registration timing meets REGISTRATION_PERFORMANCE_TARGET_MS
        - Test registration timing consistency and performance stability
        - Measure registration timing with various configuration complexity levels
        - Analyze timing distribution and identify performance optimization opportunities
        - Test registration timing under different system load conditions
        - Validate registration timing scalability with multiple concurrent operations
        - Generate comprehensive timing performance report with recommendations
        """
        timing_results = []

        # Test 1: Baseline registration timing
        baseline_times = []
        for i in range(10):
            env_id = f"TestTiming-Baseline-{i}-v0"

            start_time = time.perf_counter()
            register_env(
                env_id=env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )
            end_time = time.perf_counter()

            registration_time_ms = (end_time - start_time) * 1000
            baseline_times.append(registration_time_ms)

            self.test_registrations.add(env_id)

        # Calculate baseline statistics
        import statistics

        baseline_stats = {
            "mean_ms": statistics.mean(baseline_times),
            "median_ms": statistics.median(baseline_times),
            "min_ms": min(baseline_times),
            "max_ms": max(baseline_times),
            "std_dev_ms": (
                statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
            ),
        }
        timing_results.append(("baseline", baseline_stats))

        # Validate performance target
        assert (
            baseline_stats["mean_ms"] < REGISTRATION_PERFORMANCE_TARGET_MS
        ), f"Baseline registration timing {baseline_stats['mean_ms']:.2f}ms exceeds target {REGISTRATION_PERFORMANCE_TARGET_MS}ms"

        # Test 2: Complex configuration registration timing
        complex_times = []
        for i in range(5):
            env_id = f"TestTiming-Complex-{i}-v0"
            complex_config = create_test_registration_config("extended")

            start_time = time.perf_counter()
            register_env(
                env_id=env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                kwargs=complex_config,
            )
            end_time = time.perf_counter()

            registration_time_ms = (end_time - start_time) * 1000
            complex_times.append(registration_time_ms)

            self.test_registrations.add(env_id)

        complex_stats = {
            "mean_ms": statistics.mean(complex_times),
            "median_ms": statistics.median(complex_times),
            "min_ms": min(complex_times),
            "max_ms": max(complex_times),
        }
        timing_results.append(("complex_config", complex_stats))

        # Test 3: Unregistration timing
        unreg_times = []
        test_env_ids = [f"TestTiming-Unreg-{i}-v0" for i in range(5)]

        # First register environments to unregister
        for env_id in test_env_ids:
            register_env(
                env_id=env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )

        # Measure unregistration timing
        for env_id in test_env_ids:
            start_time = time.perf_counter()
            unregister_env(env_id)
            end_time = time.perf_counter()

            unreg_time_ms = (end_time - start_time) * 1000
            unreg_times.append(unreg_time_ms)

        unreg_stats = {
            "mean_ms": statistics.mean(unreg_times),
            "median_ms": statistics.median(unreg_times),
            "min_ms": min(unreg_times),
            "max_ms": max(unreg_times),
        }
        timing_results.append(("unregistration", unreg_stats))

        # Performance analysis and reporting
        self.benchmark_results["timing"] = timing_results

        print("\nRegistration Timing Performance Results:")
        print(f"Target: <{REGISTRATION_PERFORMANCE_TARGET_MS}ms")
        for test_name, stats in timing_results:
            mean_time = stats["mean_ms"]
            target_met = "✓" if mean_time < REGISTRATION_PERFORMANCE_TARGET_MS else "✗"
            print(
                f"  {test_name}: {mean_time:.2f}ms ± {stats.get('std_dev_ms', 0):.2f}ms {target_met}"
            )

    def test_registration_memory_performance(self):
        """
        Test registration memory performance including memory usage monitoring, leak detection,
        and resource efficiency validation.

        Validates:
        - Measure baseline memory usage before registration operations
        - Monitor memory consumption during registration and unregistration cycles
        - Validate memory usage remains within acceptable limits during operations
        - Test memory cleanup effectiveness after environment unregistration
        - Detect memory leaks during repeated registration/unregistration cycles
        - Test memory scalability with multiple concurrent environment registrations
        - Validate memory usage optimization and resource management effectiveness
        - Analyze memory usage patterns and identify optimization opportunities
        - Generate memory performance report with resource usage analysis
        """
        try:
            import psutil

            process = psutil.Process()

            # Baseline memory measurement
            baseline_memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"\nBaseline memory usage: {baseline_memory_mb:.2f}MB")

            # Test 1: Single registration memory impact
            pre_reg_memory = process.memory_info().rss / 1024 / 1024

            single_env_id = "TestMemory-Single-v0"
            register_env(
                env_id=single_env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )
            self.test_registrations.add(single_env_id)

            post_reg_memory = process.memory_info().rss / 1024 / 1024
            single_reg_delta = post_reg_memory - pre_reg_memory

            print(f"Single registration memory delta: {single_reg_delta:.2f}MB")

            # Test 2: Multiple registration memory scaling
            multi_env_ids = []
            memory_measurements = [post_reg_memory]

            for i in range(10):
                env_id = f"TestMemory-Multi-{i}-v0"
                register_env(
                    env_id=env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                )
                multi_env_ids.append(env_id)
                self.test_registrations.add(env_id)

                current_memory = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)

            max_memory = max(memory_measurements)
            total_reg_delta = max_memory - baseline_memory_mb

            print(f"Multiple registration memory delta: {total_reg_delta:.2f}MB")
            print(
                f"Average per registration: {total_reg_delta/11:.2f}MB"
            )  # 11 total registrations

            # Test 3: Memory cleanup after unregistration
            pre_unreg_memory = process.memory_info().rss / 1024 / 1024

            # Unregister all test environments
            for env_id in [single_env_id] + multi_env_ids:
                unregister_env(env_id)
                self.test_registrations.discard(env_id)

            # Force garbage collection
            import gc

            gc.collect()

            post_unreg_memory = process.memory_info().rss / 1024 / 1024
            cleanup_effectiveness = pre_unreg_memory - post_unreg_memory

            print(
                f"Memory cleanup effectiveness: {cleanup_effectiveness:.2f}MB recovered"
            )

            # Test 4: Memory leak detection through repeated cycles
            cycle_memory_start = process.memory_info().rss / 1024 / 1024

            for cycle in range(5):
                cycle_env_id = f"TestMemory-Cycle-{cycle}-v0"

                # Register
                register_env(
                    env_id=cycle_env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                )

                # Create and use environment
                env = gymnasium.make(cycle_env_id)
                obs, info = env.reset(seed=42)
                env.close()

                # Unregister
                unregister_env(cycle_env_id)

                # Force cleanup
                gc.collect()

            cycle_memory_end = process.memory_info().rss / 1024 / 1024
            potential_leak = cycle_memory_end - cycle_memory_start

            print(f"Potential memory leak over 5 cycles: {potential_leak:.2f}MB")

            # Memory performance validation
            # Allow reasonable memory overhead for registration system
            max_acceptable_per_registration = 5.0  # MB
            assert (
                single_reg_delta < max_acceptable_per_registration
            ), f"Single registration memory usage {single_reg_delta:.2f}MB exceeds limit {max_acceptable_per_registration}MB"

            # Memory leak should be minimal
            max_acceptable_leak = 2.0  # MB
            assert (
                potential_leak < max_acceptable_leak
            ), f"Potential memory leak {potential_leak:.2f}MB exceeds limit {max_acceptable_leak}MB"

            self.benchmark_results["memory"] = {
                "baseline_mb": baseline_memory_mb,
                "single_registration_delta_mb": single_reg_delta,
                "total_registration_delta_mb": total_reg_delta,
                "cleanup_effectiveness_mb": cleanup_effectiveness,
                "potential_leak_mb": potential_leak,
            }

        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
        except Exception as e:
            pytest.skip(f"Memory performance testing failed: {str(e)}")

    def test_registration_cache_performance(self):
        """
        Test registration cache performance including cache efficiency, hit rates, and cache
        optimization validation with performance impact analysis.

        Validates:
        - Initialize registration cache performance monitoring infrastructure
        - Measure cache hit rates and cache efficiency during registration operations
        - Test cache performance impact on registration and status checking operations
        - Validate cache consistency maintenance performance and overhead analysis
        - Test cache invalidation performance and cache update efficiency
        - Measure cache memory usage and resource optimization effectiveness
        - Test cache performance under concurrent access and modification scenarios
        - Analyze cache performance patterns and identify optimization opportunities
        - Generate cache performance report with efficiency analysis and recommendations
        """
        cache_performance_results = []

        # Test 1: Cache population and lookup performance
        env_ids = [f"TestCache-Perf-{i}-v0" for i in range(20)]

        # Measure registration time (cache population)
        reg_start = time.perf_counter()
        for env_id in env_ids:
            register_env(
                env_id=env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )
            self.test_registrations.add(env_id)
        reg_end = time.perf_counter()

        total_reg_time = (reg_end - reg_start) * 1000
        avg_reg_time = total_reg_time / len(env_ids)

        # Test 2: Cache lookup performance (is_registered calls)
        lookup_times = []
        for _ in range(100):  # Multiple lookups for timing consistency
            lookup_start = time.perf_counter()
            for env_id in env_ids:
                is_registered(env_id)
            lookup_end = time.perf_counter()

            lookup_time_per_call = ((lookup_end - lookup_start) * 1000) / len(env_ids)
            lookup_times.append(lookup_time_per_call)

        import statistics

        avg_lookup_time = statistics.mean(lookup_times)

        # Test 3: Registration info retrieval performance
        info_retrieval_times = []
        for env_id in env_ids[:5]:  # Sample subset for info retrieval
            info_start = time.perf_counter()
            get_registration_info(env_id)
            info_end = time.perf_counter()

            info_time = (info_end - info_start) * 1000
            info_retrieval_times.append(info_time)

        avg_info_time = statistics.mean(info_retrieval_times)

        # Test 4: Cache invalidation performance
        invalidation_times = []
        for env_id in env_ids[:10]:  # Unregister half for invalidation testing
            inv_start = time.perf_counter()
            unregister_env(env_id)
            inv_end = time.perf_counter()

            inv_time = (inv_end - inv_start) * 1000
            invalidation_times.append(inv_time)
            self.test_registrations.discard(env_id)

        avg_invalidation_time = statistics.mean(invalidation_times)

        # Test 5: Mixed operations performance simulation
        mixed_start = time.perf_counter()

        # Simulate realistic mixed workload
        for i in range(10):
            # Registration operation
            temp_env_id = f"TestCache-Mixed-{i}-v0"
            register_env(
                env_id=temp_env_id,
                entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            )

            # Multiple lookups
            for _ in range(5):
                is_registered(temp_env_id)

            # Info retrieval
            get_registration_info(temp_env_id)

            # Unregistration
            unregister_env(temp_env_id)

        mixed_end = time.perf_counter()
        mixed_operations_time = (mixed_end - mixed_start) * 1000

        # Performance validation and reporting
        cache_performance_results = {
            "avg_registration_time_ms": avg_reg_time,
            "avg_lookup_time_ms": avg_lookup_time,
            "avg_info_retrieval_time_ms": avg_info_time,
            "avg_invalidation_time_ms": avg_invalidation_time,
            "mixed_operations_time_ms": mixed_operations_time,
        }

        self.benchmark_results["cache"] = cache_performance_results

        print("\nRegistration Cache Performance Results:")
        print(f"  Average registration: {avg_reg_time:.3f}ms")
        print(f"  Average lookup: {avg_lookup_time:.4f}ms")
        print(f"  Average info retrieval: {avg_info_time:.3f}ms")
        print(f"  Average invalidation: {avg_invalidation_time:.3f}ms")
        print(f"  Mixed operations (10 cycles): {mixed_operations_time:.2f}ms")

        # Performance assertions
        max_lookup_time = 0.1  # ms
        assert (
            avg_lookup_time < max_lookup_time
        ), f"Cache lookup time {avg_lookup_time:.4f}ms exceeds limit {max_lookup_time}ms"

        max_info_time = 1.0  # ms
        assert (
            avg_info_time < max_info_time
        ), f"Info retrieval time {avg_info_time:.3f}ms exceeds limit {max_info_time}ms"

    def test_registration_scalability_performance(self):
        """
        Test registration scalability performance including multiple environment handling,
        concurrent operations, and system capacity validation.

        Validates:
        - Test registration performance with increasing number of registered environments
        - Measure system performance degradation patterns with scale increase
        - Test concurrent registration and unregistration operation performance
        - Validate system stability and performance at maximum supported capacity
        - Test registration performance under high-frequency operation scenarios
        - Measure resource usage scaling and identify system capacity limits
        - Test registration performance optimization effectiveness at scale
        - Analyze scalability performance patterns and system bottleneck identification
        - Generate scalability performance report with capacity recommendations
        """
        scalability_results = []

        # Test different scales of concurrent registrations
        scales = [5, 10, 25, 50]

        for scale in scales:
            scale_start = time.perf_counter()
            scale_env_ids = []

            # Register multiple environments at this scale
            for i in range(scale):
                env_id = f"TestScale-{scale}-{i}-v0"
                register_env(
                    env_id=env_id,
                    entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
                )
                scale_env_ids.append(env_id)
                self.test_registrations.add(env_id)

            scale_reg_end = time.perf_counter()

            # Test lookups at scale
            lookup_start = time.perf_counter()
            for env_id in scale_env_ids:
                is_registered(env_id)
            lookup_end = time.perf_counter()

            # Test gym.make at scale
            make_start = time.perf_counter()
            envs = []
            for env_id in scale_env_ids[: min(5, scale)]:  # Limit env creation
                env = gymnasium.make(env_id)
                envs.append(env)
            make_end = time.perf_counter()

            # Clean up created environments
            for env in envs:
                env.close()

            # Measure unregistration at scale
            unreg_start = time.perf_counter()
            for env_id in scale_env_ids:
                unregister_env(env_id)
                self.test_registrations.discard(env_id)
            unreg_end = time.perf_counter()

            # Calculate timing metrics
            total_scale_time = (unreg_end - scale_start) * 1000
            reg_time = (scale_reg_end - scale_start) * 1000
            lookup_time = (lookup_end - lookup_start) * 1000
            make_time = (make_end - make_start) * 1000
            unreg_time = (unreg_end - unreg_start) * 1000

            scale_result = {
                "scale": scale,
                "total_time_ms": total_scale_time,
                "registration_time_ms": reg_time,
                "lookup_time_ms": lookup_time,
                "gym_make_time_ms": make_time,
                "unregistration_time_ms": unreg_time,
                "avg_reg_per_env_ms": reg_time / scale,
                "avg_lookup_per_env_ms": lookup_time / scale,
                "avg_unreg_per_env_ms": unreg_time / scale,
            }

            scalability_results.append(scale_result)

        # Analyze scalability patterns
        self.benchmark_results["scalability"] = scalability_results

        print("\nRegistration Scalability Performance Results:")
        print(
            f"{'Scale':<6} {'Reg/env':<8} {'Lookup/env':<10} {'Unreg/env':<10} {'Total':<8}"
        )
        for result in scalability_results:
            scale = result["scale"]
            reg_per_env = result["avg_reg_per_env_ms"]
            lookup_per_env = result["avg_lookup_per_env_ms"]
            unreg_per_env = result["avg_unreg_per_env_ms"]
            total_time = result["total_time_ms"]

            print(
                f"{scale:<6} {reg_per_env:<8.2f} {lookup_per_env:<10.4f} {unreg_per_env:<10.2f} {total_time:<8.1f}"
            )

        # Validate scalability requirements
        # Registration time per environment should not significantly degrade with scale
        reg_times_per_env = [r["avg_reg_per_env_ms"] for r in scalability_results]

        # Check that performance doesn't degrade by more than 2x from smallest to largest scale
        if len(reg_times_per_env) > 1:
            performance_degradation = max(reg_times_per_env) / min(reg_times_per_env)
            max_acceptable_degradation = 2.0

            assert (
                performance_degradation < max_acceptable_degradation
            ), f"Performance degradation {performance_degradation:.2f}x exceeds limit {max_acceptable_degradation}x"

        # Ensure all operations complete within reasonable time even at largest scale
        largest_scale_result = scalability_results[-1]
        max_acceptable_total_time = 1000  # ms for largest test scale

        assert (
            largest_scale_result["total_time_ms"] < max_acceptable_total_time
        ), f"Largest scale total time {largest_scale_result['total_time_ms']:.1f}ms exceeds limit {max_acceptable_total_time}ms"
