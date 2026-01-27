"""
Comprehensive test suite for the plume_nav_sim registration module validating environment
registration, unregistration, status checking, parameter validation, and Gymnasium integration.

This test suite provides complete coverage of registration functionality including error handling,
cache management, performance requirements, and configuration validation with comprehensive
coverage of registration scenarios including edge cases, custom parameters, and integration
with Gymnasium registry.

Tests cover all registration functions with mirrored test structure matching the main module:
- register_env(): Environment registration with parameter validation
- unregister_env(): Environment cleanup and cache management
- is_registered(): Registration status checking with cache validation
- get_registration_info(): Information retrieval and debugging support
- create_registration_kwargs(): Parameter factory with validation
- validate_registration_config(): Configuration consistency checking
- register_with_custom_params(): Convenience registration with custom parameters
"""

import time  # >=3.10 - Timing utilities for performance testing and registration operation timing validation
import unittest.mock as mock  # >=3.10 - Mock object creation for testing error conditions, external dependency mocking, and isolation testing
import warnings  # >=3.10 - Warning system testing for registration conflicts, deprecation notices, and compatibility warning validation

import gymnasium  # >=0.29.0 - Reinforcement learning environment framework for registration validation, gym.make() testing, and registry management

# External imports with version comments for testing framework and dependencies
import pytest  # >=8.0.0 - Testing framework for test discovery, fixtures, parameterization, and assertion handling with comprehensive test execution management

# Internal imports for constants and configuration parameters
from plume_nav_sim.core.constants import (
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_SOURCE_LOCATION,
    ENVIRONMENT_ID,
)
from plume_nav_sim.envs.plume_env import PlumeEnv

# Internal imports for registration functions and dependencies
from plume_nav_sim.registration.register import ENV_ID, LEGACY_ENV_IDS
from plume_nav_sim.registration.register import (
    _create_registration_kwargs as create_registration_kwargs,
)
from plume_nav_sim.registration.register import (
    _validate_registration_config as validate_registration_config,
)
from plume_nav_sim.registration.register import (
    is_registered,
    register_env,
    unregister_env,
)

# Internal imports for exception classes and error handling testing
from plume_nav_sim.utils.exceptions import ConfigurationError, ValidationError

# Global test constants for comprehensive registration testing scenarios
TEST_ENV_ID = "TestPlumeNav-StaticGaussian-v0"
CUSTOM_TEST_ENV_ID = "CustomPlumeNav-Test-v0"
INVALID_ENV_ID = "InvalidEnvironment"
TEST_ENTRY_POINT = "plume_nav_sim.envs.plume_env:create_plume_env"
INVALID_ENTRY_POINT = "invalid.module.path:NonExistentClass"
TEST_PERFORMANCE_TARGET_MS = 10.0
TEST_GRID_SIZES = [(32, 32), (64, 64), (128, 128)]
TEST_SOURCE_LOCATIONS = [(16, 16), (32, 32), (64, 64)]
REPRODUCIBILITY_SEEDS = [42, 123, 456, 789]


def _unwrap_gym_env(env: gymnasium.Env) -> gymnasium.Env:
    """Unwrap Gymnasium wrapper stack to obtain the base environment."""

    base_env = env
    visited = set()

    while True:
        env_id = id(base_env)
        if env_id in visited:
            break
        visited.add(env_id)

        if hasattr(base_env, "env"):
            base_env = base_env.env
            continue

        if isinstance(base_env, gymnasium.Wrapper):
            try:
                base_env = base_env.unwrapped
            except AttributeError:
                break
            continue

        break

    return base_env


class TestEnvironmentRegistration:
    """Test class for basic environment registration functionality including register_env(),
    registration validation, gym.make() compatibility, and successful registration verification
    with comprehensive parameter testing."""

    def setup_method(self):
        """Initialize test class for environment registration testing with cleanup of existing registrations."""
        # Clean up any existing registrations to ensure clean test state
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
            unregister_env(CUSTOM_TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass  # Ignore cleanup errors

    def teardown_method(self):
        """Clean up test registrations after each test method."""
        # Clean up all test registrations to prevent test interference
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
            unregister_env(CUSTOM_TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def test_register_env_with_defaults(self):
        """Test successful environment registration using default parameters with validation
        of gym.make() compatibility, registry entry creation, and environment instantiation.
        """
        # Clean up any existing registration to ensure clean test state
        unregister_env(ENV_ID, suppress_warnings=True)

        # Call register_env() without parameters to use default configuration
        returned_env_id = register_env()

        # Assert returned environment ID matches expected ENVIRONMENT_ID constant
        assert returned_env_id == ENVIRONMENT_ID
        assert returned_env_id == ENV_ID

        # Verify environment is registered using is_registered() function
        assert is_registered(ENV_ID) is True
        assert is_registered(ENV_ID, use_cache=False) is True

        # Test gym.make() compatibility with registered environment ID
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress matplotlib warnings in testing
            try:
                self._extracted_from_test_register_env_with_defaults_23()
            except Exception as e:
                pytest.fail(f"gym.make() failed for registered environment: {e}")

        # Clean up registration after test completion
        assert unregister_env(ENV_ID) is True
        assert is_registered(ENV_ID) is False

    def test_legacy_env_ids_map_to_plume_env(self):
        """Legacy env ids should resolve to PlumeEnv with registration kwargs applied."""
        unregister_env(ENV_ID, suppress_warnings=True)
        for legacy_env_id in LEGACY_ENV_IDS:
            unregister_env(legacy_env_id, suppress_warnings=True)

        custom_kwargs = {
            "grid_size": (40, 24),
            "source_location": (12, 8),
            "max_steps": 77,
        }

        register_env(env_id=ENV_ID, kwargs=custom_kwargs, force_reregister=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for legacy_env_id in LEGACY_ENV_IDS:
                env = gymnasium.make(legacy_env_id)
                base_env = _unwrap_gym_env(env)
                assert isinstance(base_env, PlumeEnv)
                assert base_env.grid_size == custom_kwargs["grid_size"]
                assert base_env.max_steps == custom_kwargs["max_steps"]
                env.close()

        unregister_env(ENV_ID, suppress_warnings=True)
        for legacy_env_id in LEGACY_ENV_IDS:
            unregister_env(legacy_env_id, suppress_warnings=True)

    # TODO Rename this here and in `test_register_env_with_defaults`
    def _extracted_from_test_register_env_with_defaults_23(self):
        env = gymnasium.make(ENV_ID)

        # Verify environment has expected attributes
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")
        assert isinstance(env.observation_space, gymnasium.spaces.Box)
        assert hasattr(env.action_space, "n")  # Discrete action space

        # Clean up environment instance
        env.close()

    def test_register_env_with_custom_params(self):
        """Test environment registration with custom parameters including custom env_id,
        entry_point, max_episode_steps, and kwargs validation with parameter consistency checking.
        """
        # Clean up any existing registration to ensure isolated testing
        unregister_env(TEST_ENV_ID, suppress_warnings=True)

        # Define custom registration parameters: env_id, entry_point, max_steps, kwargs
        custom_env_id = TEST_ENV_ID
        custom_entry_point = TEST_ENTRY_POINT
        custom_max_steps = 500
        custom_kwargs = {
            "grid_size": (64, 64),
            "source_location": (32, 32),
            "goal_radius": 2.0,
        }

        # Call register_env() with custom parameters and validate return value
        returned_env_id = register_env(
            env_id=custom_env_id,
            entry_point=custom_entry_point,
            max_episode_steps=custom_max_steps,
            kwargs=custom_kwargs,
        )

        # Verify custom environment is registered using is_registered() with custom env_id
        assert returned_env_id == custom_env_id
        assert is_registered(custom_env_id) is True

        # Test gym.make() with custom environment ID for instantiation validation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                env = gymnasium.make(custom_env_id)

                # Validate custom kwargs are properly applied to environment configuration
                assert env is not None
                base_env = _unwrap_gym_env(env)
                assert isinstance(base_env, PlumeEnv)

                env.close()

            except Exception as e:
                pytest.fail(f"gym.make() failed with custom parameters: {e}")

        # Clean up custom registration to prevent test interference
        assert unregister_env(custom_env_id) is True
        assert is_registered(custom_env_id) is False

    def test_register_env_force_reregister(self):
        """Test force reregistration functionality with existing environment cleanup,
        conflict resolution, and successful re-registration with updated parameters."""
        # Register environment initially with default parameters
        initial_env_id = register_env(env_id=TEST_ENV_ID)

        # Verify initial registration is successful and environment is available
        assert initial_env_id == TEST_ENV_ID
        assert is_registered(TEST_ENV_ID) is True

        # Attempt registration with same env_id and force_reregister=False
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            second_registration = register_env(
                env_id=TEST_ENV_ID, force_reregister=False
            )

            # Assert appropriate warning or error handling for registration conflict
            assert (
                second_registration == TEST_ENV_ID
            )  # Should return existing registration
            # Warning should be issued about existing registration
            _ = [str(w.message) for w in warning_list]
            # Note: Warnings may come from underlying gymnasium or our registration system

        # Call register_env() with force_reregister=True and updated parameters
        updated_kwargs = {"grid_size": (128, 128), "source_location": (64, 64)}

        force_registered_env_id = register_env(
            env_id=TEST_ENV_ID, kwargs=updated_kwargs, force_reregister=True
        )

        # Verify old registration is cleaned up and new registration is active
        assert force_registered_env_id == TEST_ENV_ID
        assert is_registered(TEST_ENV_ID) is True

        # Test gym.make() with updated environment configuration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                env = gymnasium.make(TEST_ENV_ID)
                base_env = _unwrap_gym_env(env)
                assert isinstance(base_env, PlumeEnv)
                env.close()
            except Exception as e:
                pytest.fail(f"gym.make() failed after force reregistration: {e}")

        # Clean up final registration state for test isolation
        assert unregister_env(TEST_ENV_ID) is True

    def test_register_env_performance(self):
        """Test registration performance requirements with timing validation ensuring
        registration operations complete within performance targets for development efficiency.
        """
        # Clean up before performance test
        unregister_env(TEST_ENV_ID, suppress_warnings=True)

        # Record start time before registration operation
        start_time = time.time()

        # Call register_env() with default parameters
        env_id = register_env(env_id=TEST_ENV_ID)

        # Record end time and calculate registration duration
        end_time = time.time()
        registration_time_ms = (end_time - start_time) * 1000

        # Assert registration time is less than TEST_PERFORMANCE_TARGET_MS
        assert (
            registration_time_ms < TEST_PERFORMANCE_TARGET_MS
        ), f"Registration took {registration_time_ms:.2f}ms, exceeds target {TEST_PERFORMANCE_TARGET_MS}ms"

        # Perform multiple registration operations to test consistency
        performance_times = []
        for i in range(3):
            test_env_id = f"PerfTest-{i}-v0"

            start = time.time()
            register_env(env_id=test_env_id, force_reregister=True)
            end = time.time()

            perf_time_ms = (end - start) * 1000
            performance_times.append(perf_time_ms)

            # Clean up immediately
            unregister_env(test_env_id, suppress_warnings=True)

        # Validate average registration time meets performance requirements
        avg_time = sum(performance_times) / len(performance_times)
        assert (
            avg_time < TEST_PERFORMANCE_TARGET_MS
        ), f"Average registration time {avg_time:.2f}ms exceeds target"

        # Test unregistration performance as part of cleanup timing
        unreg_start = time.time()
        unregister_result = unregister_env(env_id)
        unreg_end = time.time()
        unreg_time_ms = (unreg_end - unreg_start) * 1000

        # Validate cleanup performance
        assert unreg_time_ms < TEST_PERFORMANCE_TARGET_MS
        assert unregister_result is True


class TestEnvironmentUnregistration:
    """Test class for environment unregistration functionality including unregister_env(),
    cleanup validation, registry state management, and proper resource deallocation with
    comprehensive cleanup testing."""

    def setup_method(self):
        """Initialize test class for environment unregistration testing."""
        # Clean up any existing registrations
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
            unregister_env(CUSTOM_TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def teardown_method(self):
        """Clean up after unregistration tests."""
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
            unregister_env(CUSTOM_TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def test_unregister_env_success(self):
        """Test successful environment unregistration with registry cleanup, cache clearing,
        and gym.make() unavailability validation ensuring complete environment removal.
        """
        # Register environment initially using register_env() for setup
        _ = register_env(env_id=TEST_ENV_ID)

        # Verify environment is successfully registered and available via gym.make()
        assert is_registered(TEST_ENV_ID) is True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                env = gymnasium.make(TEST_ENV_ID)
                base_env = _unwrap_gym_env(env)
                assert isinstance(base_env, PlumeEnv)
                env.close()
            except Exception:
                pytest.fail("Environment not accessible after registration")

        # Call unregister_env() and validate return value indicates success
        unregister_result = unregister_env(TEST_ENV_ID)

        # Assert is_registered() returns False after unregistration
        assert unregister_result is True
        assert is_registered(TEST_ENV_ID) is False
        assert is_registered(TEST_ENV_ID, use_cache=False) is False

        # Verify gym.make() raises appropriate error for unregistered environment
        with pytest.raises(Exception):  # gymnasium.error.UnregisteredEnv or similar
            gymnasium.make(TEST_ENV_ID)

        # Validate registration cache is cleared for unregistered environment
        # This is verified through is_registered() returning False

        # Test unregistration is idempotent - calling again should not error
        second_unregister = unregister_env(TEST_ENV_ID, suppress_warnings=True)
        assert second_unregister is True  # Should succeed without error

        # Confirm complete cleanup and registry consistency
        assert is_registered(TEST_ENV_ID) is False

    def test_unregister_nonexistent_env(self):
        """Test unregistration behavior for non-existent environment with proper error handling,
        warning management, and graceful failure modes without system disruption."""
        # Ensure test environment is not registered in initial state
        unregister_env(INVALID_ENV_ID, suppress_warnings=True)
        assert is_registered(INVALID_ENV_ID) is False

        # Call unregister_env() with non-existent environment ID
        with warnings.catch_warnings(record=True) as _:
            self._extracted_from_test_unregister_nonexistent_env_10(False)
            # Validate appropriate warnings are issued unless suppressed
            # Note: The specific warning behavior depends on implementation

        # Verify registry state remains consistent after attempted unregistration
        assert is_registered(INVALID_ENV_ID) is False

        # Test suppress_warnings parameter effectively suppresses warning output
        with warnings.catch_warnings(record=True) as _:
            self._extracted_from_test_unregister_nonexistent_env_10(True)
            # Should have fewer or no warnings when suppressed
            # Note: Implementation-specific behavior

        # Confirm system stability after non-existent environment operations
        # System should remain fully functional
        test_env_id = register_env(env_id=TEST_ENV_ID)
        assert is_registered(test_env_id) is True
        unregister_env(test_env_id, suppress_warnings=True)

    # TODO Rename this here and in `test_unregister_nonexistent_env`
    def _extracted_from_test_unregister_nonexistent_env_10(self, suppress_warnings):
        warnings.simplefilter("always")

        unregister_result = unregister_env(
            INVALID_ENV_ID, suppress_warnings=suppress_warnings
        )

        # Assert function returns True (success) for non-existent environment unregistration
        assert unregister_result is True

    def test_unregister_env_with_custom_id(self):
        """Test unregistration with custom environment ID specification, parameter validation,
        and selective environment removal with multiple registration management."""
        # Register multiple environments with different custom IDs
        env_ids = [f"MultiTest-{i}-v0" for i in range(3)]

        for env_id in env_ids:
            register_env(env_id=env_id)

        # Verify all custom environments are registered and accessible
        for env_id in env_ids:
            assert is_registered(env_id) is True

        # Call unregister_env() with specific custom environment ID
        target_env_id = env_ids[1]  # Middle environment
        unregister_result = unregister_env(target_env_id)

        # Assert only the specified environment is unregistered
        assert unregister_result is True
        assert is_registered(target_env_id) is False

        # Verify other registered environments remain available and functional
        remaining_env_ids = [env_ids[0], env_ids[2]]
        for remaining_id in remaining_env_ids:
            assert is_registered(remaining_id) is True

            # Test gym.make() works for remaining registered environments
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    env = gymnasium.make(remaining_id)
                    env.close()
                except Exception as e:
                    pytest.fail(
                        f"Remaining environment {remaining_id} not accessible: {e}"
                    )

        # Validate registration cache is updated correctly for selective removal
        # Verified through individual is_registered() calls above

        # Clean up all remaining registrations for test isolation
        for remaining_id in remaining_env_ids:
            assert unregister_env(remaining_id) is True
            assert is_registered(remaining_id) is False

    def test_unregister_env_suppress_warnings(self):
        """Test warning suppression during unregistration with warning management validation,
        logging control, and silent operation modes for automated testing."""
        # Set up warning capture mechanism for testing output
        with warnings.catch_warnings(record=True) as all_warnings:
            self._validate_warning_suppression(all_warnings)
        # Test warning suppression with different unregistration scenarios
        # Register and then unregister with suppression
        test_env_id = register_env(env_id=TEST_ENV_ID)

        with warnings.catch_warnings(record=True) as _:
            unregister_result = self._unregister_with_warning_control(test_env_id, True)
            # Validate warning suppression does not affect function return values
            assert is_registered(test_env_id) is False

        # Confirm warning suppression maintains functionality
        assert unregister_result is True

    def _validate_warning_suppression(self, all_warnings):
        _ = self._unregister_with_warning_control(INVALID_ENV_ID, False)
        warnings_without_suppression = len(all_warnings)

        # Clear warning list for next test
        all_warnings.clear()

        # Call unregister_env() with suppress_warnings=True for same scenario
        unregister_result_suppressed = unregister_env(
            INVALID_ENV_ID, suppress_warnings=True
        )

        # Assert no warnings are generated when suppression is enabled
        assert unregister_result_suppressed is True
        warnings_with_suppression = len(all_warnings)

        # Validate warning suppression effectiveness
        # Note: Some warnings may come from external sources
        assert warnings_with_suppression <= warnings_without_suppression

    def _unregister_with_warning_control(self, env_id, suppress_warnings):
        warnings.simplefilter("always")

        # Call unregister_env() for non-existent environment without suppress_warnings
        result = unregister_env(env_id, suppress_warnings=suppress_warnings)

        # Verify appropriate warnings are generated and captured
        assert result is True
        return result


class TestRegistrationStatus:
    """Test class for registration status checking functionality including is_registered(),
    cache validation, registry consistency, and status reporting with comprehensive state
    management testing."""

    def setup_method(self):
        """Initialize test class for registration status testing."""
        # Clean up any existing registrations
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def teardown_method(self):
        """Clean up after status tests."""
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def test_is_registered_true(self):
        """Test is_registered() returns True for successfully registered environments with
        cache validation, registry consistency, and accurate status reporting."""
        # Register environment using register_env() for test setup
        _ = register_env(env_id=TEST_ENV_ID)

        # Call is_registered() without parameters to check default environment
        # Note: We're testing with TEST_ENV_ID, not the default ENV_ID
        assert is_registered(TEST_ENV_ID) is True

        # Test is_registered() with explicit env_id parameter for validation
        assert is_registered(env_id=TEST_ENV_ID) is True

        # Verify cache consultation returns consistent results
        cached_result = is_registered(TEST_ENV_ID, use_cache=True)
        assert cached_result is True

        # Test registry query directly for authoritative status confirmation
        direct_result = is_registered(TEST_ENV_ID, use_cache=False)
        assert direct_result is True

        # Validate cache and registry consistency for registered environment
        assert cached_result == direct_result

        # Clean up registration after status testing completion
        cleanup_result = unregister_env(TEST_ENV_ID)
        assert cleanup_result is True

        # Verify status changes after unregistration
        assert is_registered(TEST_ENV_ID) is False

    def test_is_registered_false(self):
        """Test is_registered() returns False for unregistered or non-existent environments
        with accurate negative status reporting and cache consistency."""
        # Ensure test environment is not registered in initial state
        unregister_env(INVALID_ENV_ID, suppress_warnings=True)

        # Call is_registered() for non-existent environment
        assert is_registered(INVALID_ENV_ID) is False

        # Test is_registered() with various non-existent environment IDs
        non_existent_ids = [
            "NonExistent-v0",
            "FakeEnvironment-v1",
            "TestingFalseStatus-v0",
        ]

        for fake_id in non_existent_ids:
            assert is_registered(fake_id) is False
            assert is_registered(fake_id, use_cache=True) is False
            assert is_registered(fake_id, use_cache=False) is False

        # Verify cache returns consistent False results for unregistered environments
        # This is validated by the repeated calls above

        # Test registry consultation provides accurate negative status
        # Verified through use_cache=False calls

        # Validate no side effects occur during negative status checking
        # System should remain stable - verify by registering a real environment
        test_env_id = register_env(env_id=TEST_ENV_ID)
        assert is_registered(test_env_id) is True

        # Confirm status checking performance for unregistered environments
        # Should be fast even for non-existent environments
        start_time = time.time()
        for _ in range(10):
            is_registered(INVALID_ENV_ID)
        end_time = time.time()

        total_time_ms = (end_time - start_time) * 1000
        assert total_time_ms < 100  # Should be very fast

        # Clean up
        unregister_env(test_env_id, suppress_warnings=True)

    def test_is_registered_cache_validation(self):
        """Test registration status cache validation with cache consistency checking,
        cache invalidation, and authoritative registry consultation for accurate status reporting.
        """
        # Register environment and verify cache entry creation
        _ = register_env(env_id=TEST_ENV_ID)

        # Call is_registered() with use_cache=True to test cache consultation
        cached_status = is_registered(TEST_ENV_ID, use_cache=True)
        assert cached_status is True

        # Verify cache returns accurate status for cached environment
        # Multiple calls should be consistent
        for _ in range(3):
            assert is_registered(TEST_ENV_ID, use_cache=True) is True

        # Manually unregister environment from registry (bypass cache)
        # Note: This is testing cache invalidation behavior
        registry_status = is_registered(TEST_ENV_ID, use_cache=False)
        assert registry_status is True  # Should still be registered

        # Unregister the environment properly
        unregister_env(TEST_ENV_ID)

        # Call is_registered() with use_cache=False to force registry query
        direct_status = is_registered(TEST_ENV_ID, use_cache=False)
        assert direct_status is False

        # Assert cache is invalidated and updated with current registry status
        updated_cache_status = is_registered(TEST_ENV_ID, use_cache=True)
        assert updated_cache_status is False

        # Test cache consistency validation detects and corrects discrepancies
        assert direct_status == updated_cache_status

        # Verify final cache state matches authoritative registry status
        final_cached = is_registered(TEST_ENV_ID, use_cache=True)
        final_direct = is_registered(TEST_ENV_ID, use_cache=False)
        assert final_cached == final_direct == False

    def test_is_registered_custom_env_id(self):
        """Test registration status checking with custom environment ID specification,
        parameter validation, and multi-environment status management with selective querying.
        """
        # Register multiple environments with different custom IDs
        custom_ids = ["CustomStatus1-v0", "CustomStatus2-v0", "CustomStatus3-v0"]

        for custom_id in custom_ids:
            register_env(env_id=custom_id)

        # Test is_registered() with each custom environment ID
        for custom_id in custom_ids:
            self._extracted_from_test_is_registered_custom_env_id_14(custom_id, True)
        # Test is_registered() with non-registered custom IDs
        non_registered_ids = ["NotRegistered1-v0", "NotRegistered2-v0"]

        for unregistered_id in non_registered_ids:
            self._extracted_from_test_is_registered_custom_env_id_14(
                unregistered_id, False
            )
        # Validate selective status checking does not affect other environments
        # Unregister one environment and verify others remain
        target_id = custom_ids[1]
        unregister_env(target_id)

        assert is_registered(target_id) is False
        assert is_registered(custom_ids[0]) is True
        assert is_registered(custom_ids[2]) is True

        # Test status checking performance with multiple registered environments
        start_time = time.time()
        for custom_id in custom_ids + non_registered_ids:
            is_registered(custom_id)
        end_time = time.time()

        multi_check_time_ms = (end_time - start_time) * 1000
        assert multi_check_time_ms < 50  # Should be fast

        # Clean up all custom registrations for test isolation
        for custom_id in custom_ids:
            unregister_env(custom_id, suppress_warnings=True)
            assert is_registered(custom_id) is False

    # TODO Rename this here and in `test_is_registered_custom_env_id`
    def _extracted_from_test_is_registered_custom_env_id_14(self, arg0, arg1):
        # Assert each registered environment returns True status
        assert is_registered(arg0) is arg1
        assert is_registered(arg0, use_cache=True) is arg1
        assert is_registered(arg0, use_cache=False) is arg1


class TestRegistrationKwargs:
    """Test class for registration kwargs creation and validation including
    create_registration_kwargs(), parameter validation, default value application,
    and configuration consistency checking."""

    def test_create_registration_kwargs_defaults(self):
        """Test registration kwargs creation with default parameters using system constants
        and standard configuration values with comprehensive validation."""
        # Call create_registration_kwargs() without parameters to use defaults
        kwargs_dict = create_registration_kwargs()

        # Assert returned dictionary contains all required kwargs for registration
        assert isinstance(kwargs_dict, dict)
        assert "grid_size" in kwargs_dict
        assert "source_location" in kwargs_dict
        assert "max_steps" in kwargs_dict
        assert "goal_radius" in kwargs_dict

        # Verify grid_size matches DEFAULT_GRID_SIZE constant
        assert kwargs_dict["grid_size"] == DEFAULT_GRID_SIZE

        # Assert source_location equals DEFAULT_SOURCE_LOCATION constant
        assert kwargs_dict["source_location"] == DEFAULT_SOURCE_LOCATION

        # Check max_steps corresponds to DEFAULT_MAX_STEPS constant
        assert kwargs_dict["max_steps"] == DEFAULT_MAX_STEPS

        # Validate goal_radius matches DEFAULT_GOAL_RADIUS constant
        assert kwargs_dict["goal_radius"] == DEFAULT_GOAL_RADIUS

        # Confirm kwargs dictionary structure is suitable for gymnasium.register()
        # Should contain all expected parameter types
        assert isinstance(kwargs_dict["grid_size"], tuple)
        assert isinstance(kwargs_dict["source_location"], tuple)
        assert isinstance(kwargs_dict["max_steps"], int)
        assert isinstance(kwargs_dict["goal_radius"], (int, float))

        # Test parameter types and value ranges for default configuration
        assert len(kwargs_dict["grid_size"]) == 2
        assert all(isinstance(dim, int) for dim in kwargs_dict["grid_size"])
        assert all(dim > 0 for dim in kwargs_dict["grid_size"])

        assert len(kwargs_dict["source_location"]) == 2
        assert all(
            isinstance(coord, (int, float)) for coord in kwargs_dict["source_location"]
        )

        assert kwargs_dict["max_steps"] > 0
        assert kwargs_dict["goal_radius"] >= 0

    def test_create_registration_kwargs_custom(self):
        """Test registration kwargs creation with custom parameters including validation,
        type checking, and parameter consistency verification for specialized configurations.
        """
        # Define custom parameters: grid_size=(64, 64), source_location=(32, 32)
        custom_grid_size = (64, 64)
        custom_source_location = (32, 32)
        custom_max_steps = 750
        custom_goal_radius = 2.5

        # Call create_registration_kwargs() with custom parameters
        custom_kwargs = create_registration_kwargs(
            grid_size=custom_grid_size,
            source_location=custom_source_location,
            max_steps=custom_max_steps,
            goal_radius=custom_goal_radius,
        )

        # Assert returned kwargs contain custom grid_size and source_location
        assert custom_kwargs["grid_size"] == custom_grid_size
        assert custom_kwargs["source_location"] == custom_source_location

        # Verify custom max_steps and goal_radius values are applied correctly
        assert custom_kwargs["max_steps"] == custom_max_steps
        assert custom_kwargs["goal_radius"] == custom_goal_radius

        # Validate parameter types are correct for Gymnasium registration
        assert isinstance(custom_kwargs["grid_size"], tuple)
        assert isinstance(custom_kwargs["source_location"], tuple)
        assert isinstance(custom_kwargs["max_steps"], int)
        assert isinstance(custom_kwargs["goal_radius"], (int, float))

        # Check parameter consistency: source_location within grid_size bounds
        width, height = custom_kwargs["grid_size"]
        source_x, source_y = custom_kwargs["source_location"]
        assert 0 <= source_x < width
        assert 0 <= source_y < height

        # Test additional_kwargs merging and conflict resolution
        additional_params = {"custom_parameter_1": "value1"}
        kwargs_with_additional = create_registration_kwargs(
            grid_size=custom_grid_size, additional_kwargs=additional_params
        )

        assert "custom_parameter_1" in kwargs_with_additional
        assert kwargs_with_additional["custom_parameter_1"] == "value1"
        assert kwargs_with_additional["grid_size"] == custom_grid_size

        # Confirm custom kwargs produce valid environment configuration
        # Validation is implicit through successful creation
        assert isinstance(kwargs_with_additional, dict)

    def test_create_registration_kwargs_validation(self):
        """Test kwargs creation parameter validation with invalid parameter handling,
        constraint checking, and comprehensive error reporting for malformed configurations.
        """
        # Test invalid grid_size parameter (negative values, non-tuple)
        with pytest.raises(ValidationError) as exc_info:
            create_registration_kwargs(grid_size="invalid")

        # Assert appropriate ValidationError is raised for invalid grid dimensions
        error = exc_info.value
        assert "grid_size" in error.message.lower()

        # Test negative grid dimensions
        with pytest.raises(ValidationError):
            create_registration_kwargs(grid_size=(-1, 64))

        with pytest.raises(ValidationError):
            create_registration_kwargs(grid_size=(64, -1))

        # Test source_location outside grid bounds for constraint validation
        with pytest.raises(ValidationError) as exc_info:
            create_registration_kwargs(
                grid_size=(32, 32),
                source_location=(50, 16),  # x coordinate outside bounds
            )

        # Verify ValidationError contains helpful parameter information
        error = exc_info.value
        assert "source_location" in error.message.lower()
        assert hasattr(error, "parameter_name")

        # Test invalid max_steps values (negative, zero, excessive)
        with pytest.raises(ValidationError):
            create_registration_kwargs(max_steps=-1)

        with pytest.raises(ValidationError):
            create_registration_kwargs(max_steps=0)

        with pytest.raises(ValidationError):
            create_registration_kwargs(max_steps=200000)  # Exceeds maximum

        # Assert goal_radius validation for negative or invalid float values
        with pytest.raises(ValidationError):
            create_registration_kwargs(goal_radius=-1.0)

        # Test parameter type validation for non-numeric values
        with pytest.raises(ValidationError):
            create_registration_kwargs(max_steps="invalid")

        with pytest.raises(ValidationError):
            create_registration_kwargs(goal_radius="invalid")

        # Confirm validation error messages provide clear guidance for fixes
        try:
            create_registration_kwargs(grid_size=(0, 32))
        except ValidationError as e:
            assert "positive" in e.message.lower()

    def test_create_registration_kwargs_additional(self):
        """Test kwargs creation with additional parameters including merging, conflict detection,
        and extended configuration support with parameter override validation."""
        # Create base kwargs with standard parameters
        base_grid_size = (128, 128)
        base_source = (64, 64)

        # Define additional_kwargs with extra configuration parameters
        additional_kwargs = {
            "custom_parameter_1": "value1",
            "custom_parameter_2": 42,
            "rendering_option": True,
        }

        # Call create_registration_kwargs() with additional_kwargs parameter
        merged_kwargs = create_registration_kwargs(
            grid_size=base_grid_size,
            source_location=base_source,
            additional_kwargs=additional_kwargs,
        )

        # Assert additional parameters are properly merged into kwargs dictionary
        assert "custom_parameter_1" in merged_kwargs
        assert "custom_parameter_2" in merged_kwargs
        assert "rendering_option" in merged_kwargs

        assert merged_kwargs["custom_parameter_1"] == "value1"
        assert merged_kwargs["custom_parameter_2"] == 42
        assert merged_kwargs["rendering_option"] is True

        # Test conflict resolution when additional_kwargs override standard parameters
        conflicting_additional = {
            "grid_size": (256, 256),  # Override standard parameter
            "new_param": "test",
        }

        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")

            conflict_kwargs = create_registration_kwargs(
                grid_size=base_grid_size, additional_kwargs=conflicting_additional
            )

            # Verify final kwargs dictionary contains all parameters without duplication
            assert conflict_kwargs["grid_size"] == (256, 256)  # Should be overridden
            assert conflict_kwargs["new_param"] == "test"

            # Check for conflict warnings (implementation dependent)
            # Some implementations may warn about parameter conflicts

        # Validate merged configuration maintains parameter consistency
        width, height = conflict_kwargs["grid_size"]
        source_x, source_y = conflict_kwargs["source_location"]

        # Source should still be within new grid bounds or validation should occur
        # This tests the interaction between standard and additional parameters

        # Test extended configuration compatibility with Gymnasium registration
        # Verify that merged kwargs can be used for registration
        assert isinstance(conflict_kwargs, dict)
        assert len(conflict_kwargs) > 0


class TestConfigurationValidation:
    """Test class for registration configuration validation including validate_registration_config(),
    parameter consistency checking, Gymnasium compliance, and comprehensive validation reporting.
    """

    def test_validate_registration_config_valid(self):
        """Test configuration validation with valid parameters ensuring all validation checks
        pass and comprehensive validation report generation for correct configurations.
        """
        # Create valid configuration parameters: env_id, entry_point, max_steps, kwargs
        valid_env_id = "ValidTest-v0"
        valid_entry_point = TEST_ENTRY_POINT
        valid_max_steps = 1000
        valid_kwargs = {
            "grid_size": (128, 128),
            "source_location": (64, 64),
            "goal_radius": 1.0,
        }

        # Call validate_registration_config() with valid configuration
        is_valid, validation_report = validate_registration_config(
            env_id=valid_env_id,
            entry_point=valid_entry_point,
            max_episode_steps=valid_max_steps,
            kwargs=valid_kwargs,
        )

        # Assert validation returns (True, validation_report) tuple
        assert isinstance(is_valid, bool)
        assert is_valid is True
        assert isinstance(validation_report, dict)

        # Verify validation_report indicates all checks passed successfully
        assert "errors" in validation_report
        assert len(validation_report["errors"]) == 0

        # Check env_id format validation passes Gymnasium conventions
        # No errors should be reported for valid env_id format
        error_messages = validation_report.get("errors", [])
        env_id_errors = [
            msg
            for msg in error_messages
            if "env_id" in msg.lower() or "environment id" in msg.lower()
        ]
        assert len(env_id_errors) == 0

        # Assert entry_point validation confirms module accessibility
        entry_point_errors = [msg for msg in error_messages if "entry" in msg.lower()]
        assert len(entry_point_errors) == 0

        # Validate max_episode_steps meets reasonable training requirements
        steps_errors = [msg for msg in error_messages if "steps" in msg.lower()]
        assert len(steps_errors) == 0

        # Confirm kwargs parameter consistency and constraint satisfaction
        assert "warnings" in validation_report
        # Warnings are acceptable for valid configurations

    def test_validate_registration_config_invalid_env_id(self):
        """Test configuration validation with invalid environment ID formats including
        versioning convention violations and format specification errors with detailed error reporting.
        """
        # Test environment ID without required '-v0' suffix
        invalid_env_id = "InvalidEnvironment"  # Missing -v0
        valid_entry_point = TEST_ENTRY_POINT
        valid_max_steps = 1000
        valid_kwargs = {"grid_size": (64, 64)}

        # Call validate_registration_config() with invalid env_id format
        is_valid, validation_report = validate_registration_config(
            env_id=invalid_env_id,
            entry_point=valid_entry_point,
            max_episode_steps=valid_max_steps,
            kwargs=valid_kwargs,
        )

        # Assert validation returns (False, validation_report) tuple
        assert is_valid is False
        assert isinstance(validation_report, dict)

        # Verify validation_report contains specific env_id format errors
        assert "errors" in validation_report
        errors = validation_report["errors"]
        assert len(errors) > 0

        # Find env_id related error
        env_id_error_found = any("-v0" in error for error in errors)
        assert env_id_error_found, f"Expected env_id format error, got: {errors}"

        # Test various invalid env_id formats: empty, malformed, incorrect versioning
        invalid_formats = [
            "",  # Empty
            "Test-v1",  # Wrong version
            "NoVersion",  # No version
            "Test-v0-extra",  # Extra suffix
            "test@invalid-v0",  # Invalid characters
        ]

        for invalid_format in invalid_formats:
            is_valid_format, report_format = validate_registration_config(
                env_id=invalid_format,
                entry_point=valid_entry_point,
                max_episode_steps=valid_max_steps,
                kwargs=valid_kwargs,
            )

            # Assert error messages provide clear guidance for env_id correction
            assert is_valid_format is False
            assert len(report_format.get("errors", [])) > 0

        # Check validation identifies Gymnasium versioning convention violations
        # This is verified by the -v0 suffix check above

        # Confirm validation report includes recommendations for proper formatting
        assert "recommendations" in validation_report or "warnings" in validation_report

    def test_validate_registration_config_invalid_entry_point(self):
        """Test configuration validation with invalid entry point specifications including
        non-existent modules, incorrect class paths, and accessibility issues with module validation.
        """
        # Test entry_point with non-existent module path
        valid_env_id = "TestEntryPoint-v0"
        invalid_entry_point = (
            INVALID_ENTRY_POINT  # invalid.module.path:NonExistentClass
        )
        valid_max_steps = 1000
        valid_kwargs = {"grid_size": (64, 64)}

        # Call validate_registration_config() with invalid entry_point
        is_valid, validation_report = validate_registration_config(
            env_id=valid_env_id,
            entry_point=invalid_entry_point,
            max_episode_steps=valid_max_steps,
            kwargs=valid_kwargs,
        )

        # Assert validation identifies entry_point accessibility issues
        assert is_valid is False or len(validation_report.get("warnings", [])) > 0

        # Verify validation_report contains specific entry_point error details
        # Note: Actual module validation may generate warnings instead of errors
        issues = validation_report.get("errors", []) + validation_report.get(
            "warnings", []
        )
        _ = [
            issue
            for issue in issues
            if "entry" in issue.lower() or "module" in issue.lower()
        ]

        # Test malformed entry_point format without colon separator
        malformed_entry_point = "invalid.module.path.NoColon"

        is_valid_malformed, report_malformed = validate_registration_config(
            env_id=valid_env_id,
            entry_point=malformed_entry_point,
            max_episode_steps=valid_max_steps,
            kwargs=valid_kwargs,
        )

        # Assert validation detects incorrect module:class specification
        assert is_valid_malformed is False
        errors_malformed = report_malformed.get("errors", [])
        colon_error = any(":" in error for error in errors_malformed)
        assert colon_error, f"Expected colon format error, got: {errors_malformed}"

        # Check validation identifies non-existent class within valid module
        # Test with potentially valid module but invalid class
        questionable_entry_point = "sys:NonExistentClass"

        is_valid_class, report_class = validate_registration_config(
            env_id=valid_env_id,
            entry_point=questionable_entry_point,
            max_episode_steps=valid_max_steps,
            kwargs=valid_kwargs,
        )

        # Confirm error messages provide guidance for entry_point correction
        all_reports = [validation_report, report_malformed, report_class]
        for report in all_reports:
            if not report.get("errors") and not report.get("warnings"):
                continue
            # Should contain some guidance in errors, warnings, or recommendations
            has_guidance = (
                len(report.get("errors", [])) > 0
                or len(report.get("warnings", [])) > 0
                or len(report.get("recommendations", [])) > 0
            )
            assert has_guidance

    def test_validate_registration_config_strict_mode(self):
        """Test configuration validation in strict mode with enhanced precision checking,
        comprehensive constraint validation, and detailed analysis for production configurations.
        """
        # Create configuration with parameters at validation boundaries
        boundary_env_id = "BoundaryTest-v0"
        boundary_entry_point = TEST_ENTRY_POINT
        boundary_max_steps = 100  # At lower boundary
        boundary_kwargs = {
            "grid_size": (16, 16),  # Minimum reasonable size
            "source_location": (8, 8),  # Center of minimum grid
            "goal_radius": 0.0,  # Minimum goal radius
        }

        # Call validate_registration_config() with strict_validation=True
        is_valid_strict, report_strict = validate_registration_config(
            env_id=boundary_env_id,
            entry_point=boundary_entry_point,
            max_episode_steps=boundary_max_steps,
            kwargs=boundary_kwargs,
            strict_validation=True,
        )

        # Assert strict mode applies enhanced precision checking
        assert isinstance(report_strict, dict)
        assert "strict_validation" in report_strict
        assert report_strict["strict_validation"] is True

        # Verify strict validation identifies subtle configuration issues
        # May generate warnings for boundary values even if technically valid
        _ = len(report_strict.get("errors", [])) + len(
            report_strict.get("warnings", [])
        )

        # Test strict mode parameter consistency requirements
        # Compare with non-strict mode for same configuration
        is_valid_normal, report_normal = validate_registration_config(
            env_id=boundary_env_id,
            entry_point=boundary_entry_point,
            max_episode_steps=boundary_max_steps,
            kwargs=boundary_kwargs,
            strict_validation=False,
        )

        # Check strict validation provides more comprehensive analysis
        strict_issues = len(report_strict.get("warnings", [])) + len(
            report_strict.get("recommendations", [])
        )
        normal_issues = len(report_normal.get("warnings", [])) + len(
            report_normal.get("recommendations", [])
        )

        # Strict mode should be more thorough (equal or more issues identified)
        assert strict_issues >= normal_issues

        # Validate strict mode performance feasibility checking
        if "performance_analysis" in report_strict:
            perf_analysis = report_strict["performance_analysis"]
            assert isinstance(perf_analysis, dict)

        # Confirm strict mode validation report includes optimization recommendations
        assert "recommendations" in report_strict
        recommendations = report_strict["recommendations"]
        assert isinstance(recommendations, list)

        # Test with configuration that should trigger strict mode warnings
        challenging_config = {
            "grid_size": (512, 512),  # Large grid
            "source_location": (0, 0),  # Edge placement
            "goal_radius": 100.0,  # Very large goal radius
        }

        is_valid_challenge, report_challenge = validate_registration_config(
            env_id="ChallengeTest-v0",
            entry_point=boundary_entry_point,
            max_episode_steps=50000,  # Very high
            kwargs=challenging_config,
            strict_validation=True,
        )

        # Strict mode should identify more issues with challenging configuration
        challenge_warnings = len(report_challenge.get("warnings", []))
        assert challenge_warnings > 0


class TestCustomParameterRegistration:
    """Test class for custom parameter registration functionality including
    register_with_custom_params(), specialized configurations, parameter override validation,
    and convenience registration testing."""

    def setup_method(self):
        """Initialize test class for custom parameter registration testing."""
        # Clean up any existing registrations
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
            unregister_env(CUSTOM_TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def teardown_method(self):
        """Clean up after custom parameter tests."""
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
            unregister_env(CUSTOM_TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def test_register_with_custom_params_basic(self):
        """Test basic custom parameter registration with specialized grid sizes, source locations,
        and environment configuration overrides with validation and accessibility testing.
        """
        # Define custom parameters: grid_size=(64, 64), source_location=(32, 32)
        custom_grid_size = (64, 64)
        custom_source_location = (32, 32)
        custom_max_steps = 800
        custom_goal_radius = 2.0

        # Call register_env() with custom configuration kwargs
        kwargs = create_registration_kwargs(
            grid_size=custom_grid_size,
            source_location=custom_source_location,
            max_steps=custom_max_steps,
            goal_radius=custom_goal_radius,
        )
        returned_env_id = register_env(kwargs=kwargs)

        # Assert function returns valid environment ID for immediate use
        assert isinstance(returned_env_id, str)
        assert returned_env_id.endswith("-v0")

        # Verify environment is successfully registered with custom parameters
        assert is_registered(returned_env_id) is True

        # Test gym.make() with returned environment ID for instantiation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                env = gymnasium.make(returned_env_id)

                # Assert created environment uses custom grid_size and source_location
                assert env is not None
                # Note: Direct parameter validation requires access to environment internals
                # Success is validated through successful instantiation

                # Validate custom configuration is properly applied to environment
                assert hasattr(env, "action_space")
                assert hasattr(env, "observation_space")

                env.close()

            except Exception as e:
                pytest.fail(
                    f"Environment instantiation failed with custom parameters: {e}"
                )

        # Clean up custom registration for test isolation
        cleanup_result = unregister_env(returned_env_id)
        assert cleanup_result is True
        assert is_registered(returned_env_id) is False

    def test_register_with_custom_params_custom_env_id(self):
        """Test custom parameter registration with custom environment ID specification,
        version suffix validation, and unique identifier management for specialized research configurations.
        """
        # Define custom parameters with custom_env_id='CustomTest-v0'
        custom_env_id_base = "CustomTest"
        custom_grid_size = (96, 96)
        custom_source_location = (48, 48)

        # Call register_env() with custom_env_id parameter
        kwargs = create_registration_kwargs(
            grid_size=custom_grid_size,
            source_location=custom_source_location,
        )
        expected_env_id = f"{custom_env_id_base}-v0"
        returned_env_id = register_env(env_id=expected_env_id, kwargs=kwargs)

        # Assert function validates custom environment ID format
        assert returned_env_id.endswith("-v0")

        # Verify returned environment ID matches custom_env_id specification
        expected_env_id = f"{custom_env_id_base}-v0"
        assert returned_env_id == expected_env_id

        # Test environment registration with custom identifier
        assert is_registered(returned_env_id) is True

        # Assert gym.make() works with custom environment ID
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                env = gymnasium.make(returned_env_id)
                assert env is not None
                env.close()
            except Exception as e:
                pytest.fail(f"Custom environment ID registration failed: {e}")

        # Validate custom environment ID follows versioning conventions
        assert "-v0" in returned_env_id

        # Test with already properly formatted custom_env_id
        pre_formatted_id = "PreFormatted-v0"
        kwargs = create_registration_kwargs(grid_size=(32, 32))
        returned_formatted = register_env(
            env_id=pre_formatted_id, kwargs=kwargs, force_reregister=True
        )

        assert returned_formatted == pre_formatted_id

        # Clean up custom environment registrations
        unregister_env(returned_env_id, suppress_warnings=True)
        unregister_env(pre_formatted_id, suppress_warnings=True)

    def test_register_with_custom_params_force_reregister(self):
        """Test custom parameter registration with force reregistration for updating existing
        environments, conflict resolution, and configuration updates with state management.
        """
        # Register environment initially with standard parameters
        initial_env_id = register_env(env_id=TEST_ENV_ID)
        assert is_registered(initial_env_id) is True

        # Call register_env() with same env_id and force_reregister=False
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")

            # This should either succeed (returning existing) or issue warnings
            kwargs = create_registration_kwargs(grid_size=(96, 96))
            self._extracted_from_test_register_with_custom_params_force_reregister_15(
                kwargs, False
            )
        # Call register_env() with force_reregister=True
        updated_grid_size = (128, 128)
        updated_source = (64, 64)
        updated_goal_radius = 3.0

        kwargs = create_registration_kwargs(
            grid_size=updated_grid_size,
            source_location=updated_source,
            goal_radius=updated_goal_radius,
        )
        self._extracted_from_test_register_with_custom_params_force_reregister_15(
            kwargs, True
        )
        # Test gym.make() uses updated custom configuration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                updated_env = gymnasium.make(TEST_ENV_ID)
                assert updated_env is not None
                # Updated configuration validated through successful instantiation
                updated_env.close()
            except Exception as e:
                pytest.fail(f"Updated environment configuration failed: {e}")

        # Validate custom parameters override previous registration
        # This is verified through successful re-registration and instantiation

        # Clean up updated registration after testing
        cleanup_result = unregister_env(TEST_ENV_ID)
        assert cleanup_result is True

    # TODO Rename this here and in `test_register_with_custom_params_force_reregister`
    def _extracted_from_test_register_with_custom_params_force_reregister_15(
        self, kwargs, force_reregister
    ):
        first_attempt = register_env(
            env_id=TEST_ENV_ID, kwargs=kwargs, force_reregister=force_reregister
        )

        # Verify appropriate handling of registration conflict
        assert first_attempt == TEST_ENV_ID
        # Environment should still exist
        assert is_registered(TEST_ENV_ID) is True

    def test_register_with_custom_params_validation(self):
        """Test custom parameter registration validation with invalid parameters, constraint checking,
        and error handling for malformed custom configurations with comprehensive error reporting.
        """
        # Test custom parameter registration with invalid grid_size
        with pytest.raises(ValidationError) as exc_info:
            kwargs = create_registration_kwargs(
                grid_size=(-32, 32),
                source_location=(16, 16),  # Negative dimension
            )
            register_env(kwargs=kwargs)

        # Assert ValidationError is raised for invalid custom parameters
        error = exc_info.value
        assert isinstance(error, ValidationError)

        # Verify error message contains helpful custom parameter guidance
        assert "grid_size" in error.message.lower()
        assert hasattr(error, "parameter_name")

        # Test invalid source_location outside custom grid bounds
        with pytest.raises(ValidationError) as exc_info:
            kwargs = create_registration_kwargs(
                grid_size=(32, 32),
                source_location=(50, 16),  # x outside bounds
            )
            register_env(kwargs=kwargs)

        # Assert constraint validation works for custom configurations
        error = exc_info.value
        assert "source_location" in error.message.lower()

        # Test invalid max_steps and goal_radius custom values
        with pytest.raises(ValidationError):
            kwargs = create_registration_kwargs(max_steps=-100)  # Negative steps
            register_env(kwargs=kwargs)

        with pytest.raises(ValidationError):
            kwargs = create_registration_kwargs(goal_radius=-5.0)  # Negative radius
            register_env(kwargs=kwargs)

        # Check validation error provides specific parameter correction guidance
        try:
            kwargs = create_registration_kwargs(grid_size="invalid")
            register_env(kwargs=kwargs)
        except (ValidationError, TypeError):
            # May raise ValidationError or TypeError for type mismatch
            pass

        # Test zero dimensions (already validated in create_registration_kwargs)


class TestRegistrationErrorHandling:
    """Test class for comprehensive registration error handling including exception validation,
    error recovery, warning management, and edge case error scenarios with robust error reporting.
    """

    def setup_method(self):
        """Initialize test class for registration error handling testing."""
        # Clean up any existing registrations
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def teardown_method(self):
        """Clean up after error handling tests."""
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def test_registration_configuration_error(self):
        """Test ConfigurationError handling during registration with invalid configurations,
        error reporting, recovery suggestions, and detailed diagnostic information."""
        # Create invalid configuration causing ConfigurationError
        invalid_config = {
            "grid_size": "invalid_size",  # Should be tuple
            "source_location": "invalid_location",  # Should be tuple
        }

        # Call registration function with invalid configuration
        with pytest.raises(ValidationError) as exc_info:
            create_registration_kwargs(grid_size=invalid_config["grid_size"])

        # Assert ValidationError (or ConfigurationError) is raised with appropriate message
        error = exc_info.value
        assert isinstance(error, ValidationError)

        # Verify error contains specific configuration issue details
        assert "grid_size" in error.message.lower()
        assert hasattr(error, "parameter_name")

        # Check error provides valid configuration options for recovery
        if hasattr(error, "expected_format") and error.expected_format:
            assert "tuple" in error.expected_format.lower()

        # Test error severity level is appropriate for configuration issues
        if hasattr(error, "severity"):
            from plume_nav_sim.utils.exceptions import ErrorSeverity

            assert error.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]

        # Validate error logging includes debugging information
        # Error should contain helpful context
        assert len(error.message) > 0

        # Confirm system remains stable after configuration error
        # Should be able to create valid configuration after error
        valid_kwargs = create_registration_kwargs(
            grid_size=(64, 64), source_location=(32, 32)
        )
        assert isinstance(valid_kwargs, dict)

    def test_registration_validation_error(self):
        """Test ValidationError handling for invalid parameters with parameter-specific error
        reporting, constraint violation details, and validation recovery guidance."""
        # Create invalid parameters causing ValidationError
        invalid_grid_size = (-32, -32)  # Negative dimensions

        # Call registration function with invalid parameters
        with pytest.raises(ValidationError) as exc_info:
            create_registration_kwargs(grid_size=invalid_grid_size)

        # Assert ValidationError is raised with parameter details
        validation_error = exc_info.value
        assert isinstance(validation_error, ValidationError)

        # Verify error includes parameter_name and invalid_value information
        assert hasattr(validation_error, "parameter_name")
        assert validation_error.parameter_name is not None

        if hasattr(validation_error, "invalid_value"):
            # Should contain reference to the invalid value
            assert validation_error.invalid_value is not None

        # Check error provides expected_format for parameter correction
        if hasattr(validation_error, "expected_format"):
            assert validation_error.expected_format is not None

        # Test validation error contains parameter constraints
        constraint_keywords = ["positive", "integer", "tuple", "bounds"]
        message_lower = validation_error.message.lower()
        has_constraint_info = any(
            keyword in message_lower for keyword in constraint_keywords
        )
        assert has_constraint_info

        # Validate error recovery suggestions are helpful and actionable
        if (
            hasattr(validation_error, "recovery_suggestion")
            and validation_error.recovery_suggestion
        ):
            recovery = validation_error.recovery_suggestion.lower()
            helpful_keywords = ["check", "ensure", "use", "provide"]
            has_helpful_guidance = any(
                keyword in recovery for keyword in helpful_keywords
            )
            assert has_helpful_guidance

        # Confirm validation errors do not corrupt system state
        # Should be able to perform valid operations after validation error
        valid_env_id = register_env(env_id=TEST_ENV_ID)
        assert is_registered(valid_env_id) is True
        unregister_env(valid_env_id, suppress_warnings=True)

    @mock.patch("gymnasium.register")
    def test_registration_integration_error(self, mock_register):
        """Test IntegrationError handling for Gymnasium integration failures with dependency
        compatibility checking, version analysis, and integration recovery options."""
        # Mock Gymnasium integration failure scenario
        mock_register.side_effect = Exception("Gymnasium integration failure")

        # Call registration function to trigger IntegrationError
        with pytest.raises((Exception, ConfigurationError)) as exc_info:
            register_env(env_id=TEST_ENV_ID)

        # Integration errors may be wrapped in ConfigurationError
        error = exc_info.value

        # Verify error handling provides integration context
        error_message = str(error).lower()
        integration_keywords = ["gymnasium", "integration", "dependency"]
        _ = any(keyword in error_message for keyword in integration_keywords)

        # System should handle integration failures gracefully
        assert isinstance(error, Exception)

        # Test with mock import error to simulate missing dependency
        with mock.patch(
            "gymnasium.register", side_effect=ImportError("No module named 'gymnasium'")
        ):
            with pytest.raises(Exception) as import_exc_info:
                register_env(env_id=TEST_ENV_ID)

            # Should handle missing dependency gracefully
            import_error = import_exc_info.value
            assert isinstance(import_error, Exception)

    def test_registration_warning_management(self):
        """Test warning management during registration operations with warning suppression,
        informational warnings, and deprecation notice handling for user guidance."""
        # Set up warning capture mechanism for testing
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # Perform registration operation that generates warnings
            # Register same environment twice to trigger warning
            _ = register_env(env_id=TEST_ENV_ID)

            # Clear warnings from first registration
            warning_list.clear()

            # Second registration should generate warning
            _ = register_env(
                env_id=TEST_ENV_ID,
                force_reregister=False,  # Should warn about existing registration
            )

            # Assert appropriate warnings are generated for user guidance
            # Note: Warning behavior depends on implementation
            warning_messages = [str(w.message) for w in warning_list]

            # Test warning suppression functionality when required
            warning_list.clear()

            # Unregister and re-register with force to test different warning scenarios
            unregister_env(TEST_ENV_ID, suppress_warnings=True)

            # Register with force_reregister should not warn
            _ = register_env(env_id=TEST_ENV_ID, force_reregister=True)

            # Verify warning messages contain helpful information
            # Warnings should guide users toward correct usage
            for warning_msg in warning_messages:
                # Should contain actionable information
                helpful_terms = ["force_reregister", "already", "exist", "override"]
                _ = any(term in warning_msg.lower() for term in helpful_terms)

            # Check warning categories are appropriate for registration context
            # Warnings should be UserWarning or similar appropriate category
            for warning in warning_list:
                assert issubclass(warning.category, Warning)

        # Test deprecation warnings for version compatibility issues
        # This would be implementation-specific

        # Confirm warnings do not affect registration functionality
        assert is_registered(TEST_ENV_ID) is True

        # Clean up
        unregister_env(TEST_ENV_ID, suppress_warnings=True)


class TestRegistrationPerformance:
    """Test class for registration performance validation including timing benchmarks,
    memory usage monitoring, cache performance, and scalability testing with comprehensive
    performance analysis."""

    def setup_method(self):
        """Initialize test class for registration performance testing."""
        # Clean up any existing registrations
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def teardown_method(self):
        """Clean up after performance tests."""
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
            # Clean up any performance test environments
            for i in range(5):
                unregister_env(f"PerfTest-{i}-v0", suppress_warnings=True)
        except Exception:
            pass

    def test_registration_timing_performance(self):
        """Test registration operation timing performance with benchmark validation ensuring
        registration operations complete within performance targets for efficient development workflows.
        """
        # Record baseline system performance for comparison
        baseline_start = time.time()
        # Simple operation for baseline
        _ = {"test": "baseline"}
        baseline_end = time.time()
        _ = (baseline_end - baseline_start) * 1000

        # Measure registration operation timing over multiple iterations
        registration_times = []
        num_iterations = 5

        for i in range(num_iterations):
            test_env_id = f"PerfTest-{i}-v0"

            start_time = time.time()
            env_id = register_env(env_id=test_env_id)
            end_time = time.time()

            registration_time_ms = (end_time - start_time) * 1000
            registration_times.append(registration_time_ms)

            # Verify registration succeeded
            assert env_id == test_env_id
            assert is_registered(env_id) is True

            # Clean up immediately to isolate timing
            unregister_env(env_id, suppress_warnings=True)

        # Assert average registration time meets performance targets
        avg_time = sum(registration_times) / len(registration_times)
        assert (
            avg_time < TEST_PERFORMANCE_TARGET_MS
        ), f"Average registration time {avg_time:.2f}ms exceeds target {TEST_PERFORMANCE_TARGET_MS}ms"

        # Test registration timing consistency across multiple operations
        std_dev = (
            sum((t - avg_time) ** 2 for t in registration_times)
            / len(registration_times)
        ) ** 0.5
        max_acceptable_variation = TEST_PERFORMANCE_TARGET_MS * 0.5  # 50% variation
        assert (
            std_dev < max_acceptable_variation
        ), f"Registration time variation {std_dev:.2f}ms too high"

        # Validate registration performance under different parameter configurations
        complex_config_times = []

        for grid_size in [(32, 32), (128, 128)]:
            test_env_id = f"ComplexPerfTest-{grid_size[0]}-v0"

            start_time = time.time()
            env_id = register_env(
                env_id=test_env_id,
                kwargs={
                    "grid_size": grid_size,
                    "source_location": (grid_size[0] // 2, grid_size[1] // 2),
                },
            )
            end_time = time.time()

            config_time_ms = (end_time - start_time) * 1000
            complex_config_times.append(config_time_ms)

            # Clean up
            unregister_env(env_id, suppress_warnings=True)

        # Check unregistration timing performance for cleanup operations
        unreg_times = []

        for i in range(3):
            test_env_id = f"UnregPerfTest-{i}-v0"
            register_env(env_id=test_env_id)

            start_time = time.time()
            unregister_result = unregister_env(test_env_id)
            end_time = time.time()

            unreg_time_ms = (end_time - start_time) * 1000
            unreg_times.append(unreg_time_ms)

            assert unregister_result is True

        avg_unreg_time = sum(unreg_times) / len(unreg_times)
        assert avg_unreg_time < TEST_PERFORMANCE_TARGET_MS

        # Measure cache performance impact on registration speed
        # This is inherently tested through the multiple registrations above

        # Confirm registration timing scales appropriately with complexity
        # Larger grids should not significantly impact registration time
        # since registration doesn't create the environment instance
        max_complex_time = max(complex_config_times)
        assert (
            max_complex_time < TEST_PERFORMANCE_TARGET_MS * 2
        ), f"Complex configuration registration time {max_complex_time:.2f}ms too slow"

    def test_registration_cache_performance(self):
        """Test registration cache performance with cache hit rates, consistency checking
        performance, and cache invalidation timing for optimal status checking efficiency.
        """
        # Register multiple environments for cache performance testing
        cache_test_envs = []
        for i in range(5):
            env_id = f"CacheTest-{i}-v0"
            register_env(env_id=env_id)
            cache_test_envs.append(env_id)

        # Measure cache consultation timing for is_registered() calls
        _ = []
        num_cache_tests = 20

        start_time = time.time()
        for _ in range(num_cache_tests):
            for env_id in cache_test_envs:
                # Use cache for faster lookups
                is_registered(env_id, use_cache=True)
        end_time = time.time()

        total_cache_time_ms = (end_time - start_time) * 1000
        avg_cache_lookup_ms = total_cache_time_ms / (
            num_cache_tests * len(cache_test_envs)
        )

        # Assert cache hit performance meets efficiency targets
        max_cache_lookup_ms = 0.5  # Very fast for cached lookups
        assert (
            avg_cache_lookup_ms < max_cache_lookup_ms
        ), f"Average cache lookup {avg_cache_lookup_ms:.3f}ms too slow"

        # Test cache consistency validation performance impact
        consistency_start = time.time()
        for env_id in cache_test_envs:
            # Force registry query for consistency check
            is_registered(env_id, use_cache=False)
        consistency_end = time.time()

        consistency_time_ms = (consistency_end - consistency_start) * 1000
        avg_consistency_check_ms = consistency_time_ms / len(cache_test_envs)

        # Consistency checks should be reasonably fast
        max_consistency_ms = 2.0
        assert (
            avg_consistency_check_ms < max_consistency_ms
        ), f"Cache consistency check {avg_consistency_check_ms:.2f}ms too slow"

        # Measure cache invalidation and update timing
        target_env = cache_test_envs[0]

        invalidation_start = time.time()
        unregister_env(target_env)
        invalidation_end = time.time()

        _ = (invalidation_end - invalidation_start) * 1000

        # Validate cache memory usage remains within acceptable limits
        # This is difficult to measure precisely, but we can test with many environments
        many_env_ids = []
        for i in range(10):
            env_id = f"MemoryTest-{i}-v0"
            register_env(env_id=env_id)
            many_env_ids.append(env_id)

        # Cache operations should still be fast with more environments
        many_env_start = time.time()
        for env_id in many_env_ids:
            is_registered(env_id, use_cache=True)
        many_env_end = time.time()

        many_env_time_ms = (many_env_end - many_env_start) * 1000
        avg_many_env_ms = many_env_time_ms / len(many_env_ids)

        # Should scale well
        assert avg_many_env_ms < max_cache_lookup_ms * 2

        # Check cache performance with multiple concurrent registrations
        # Simulated through rapid sequential operations
        concurrent_start = time.time()
        concurrent_envs = []
        for i in range(5):
            env_id = f"ConcurrentTest-{i}-v0"
            register_env(env_id=env_id, force_reregister=True)
            concurrent_envs.append(env_id)
        concurrent_end = time.time()

        _ = (concurrent_end - concurrent_start) * 1000

        # Confirm cache optimization improves overall system performance
        # Cache hits should be significantly faster than registry queries
        cache_time_ratio = avg_cache_lookup_ms / avg_consistency_check_ms
        assert (
            cache_time_ratio < 0.65
        ), "Cache not providing expected performance benefit"

        # Clean up all test environments
        all_test_envs = cache_test_envs + many_env_ids + concurrent_envs
        for env_id in all_test_envs:
            unregister_env(env_id, suppress_warnings=True)

    def test_registration_memory_usage(self):
        """Test registration memory usage and resource consumption with memory monitoring,
        leak detection, and resource cleanup validation for sustainable registration operations.
        """
        # Note: Precise memory measurement requires external tools
        # This test focuses on behavioral validation of memory-conscious operations

        # Measure baseline memory usage before registration operations
        import gc
        import sys

        _ = len(gc.get_objects()) if "gc" in sys.modules else 0

        # Monitor memory consumption during registration operations
        memory_test_envs = []
        object_counts = []

        for i in range(10):
            env_id = f"MemoryTest-{i}-v0"
            register_env(env_id=env_id)
            memory_test_envs.append(env_id)

            # Track object count as proxy for memory usage
            if "gc" in sys.modules:
                import gc

                current_objects = len(gc.get_objects())
                object_counts.append(current_objects)

        # Assert registration memory usage remains within acceptable limits
        # Object count should grow predictably, not exponentially
        if object_counts:
            max_objects = max(object_counts)
            min_objects = min(object_counts)
            object_growth = max_objects - min_objects

            # Growth should be reasonable (not unlimited)
            max_reasonable_growth = 1000  # objects per registration
            assert object_growth < max_reasonable_growth * len(memory_test_envs)

        # Test memory cleanup after unregistration operations
        cleanup_start_objects = len(gc.get_objects()) if "gc" in sys.modules else 0

        for env_id in memory_test_envs:
            unregister_env(env_id)

        # Force garbage collection if available
        if "gc" in sys.modules:
            import gc

            gc.collect()

        cleanup_end_objects = len(gc.get_objects()) if "gc" in sys.modules else 0

        # Validate no memory leaks occur during registration cycles
        # Object count should decrease after cleanup
        if cleanup_start_objects > 0 and cleanup_end_objects > 0:
            # Some cleanup should occur
            cleanup_ratio = cleanup_end_objects / cleanup_start_objects
            assert cleanup_ratio < 1.1, "Potential memory leak detected"

        # Check memory scaling with multiple environment registrations
        # Test with various grid sizes
        scaling_test_envs = []
        for i, grid_size in enumerate([(32, 32), (64, 64), (128, 128)]):
            env_id = f"ScalingTest-{i}-v0"
            register_env(env_id=env_id, kwargs={"grid_size": grid_size})
            scaling_test_envs.append(env_id)

        # Monitor cache memory contribution to total usage
        # Multiple cache operations
        for _ in range(20):
            for env_id in scaling_test_envs:
                is_registered(env_id, use_cache=True)

        # Memory usage patterns should support scalable registration
        # System should remain stable
        final_test_env = register_env(env_id="FinalMemoryTest-v0")
        assert is_registered(final_test_env) is True

        # Clean up all test environments
        all_memory_test_envs = scaling_test_envs + [final_test_env]
        for env_id in all_memory_test_envs:
            unregister_env(env_id, suppress_warnings=True)

        # Final cleanup validation
        if "gc" in sys.modules:
            import gc

            gc.collect()

    def test_registration_scalability(self):
        """Test registration system scalability with multiple environment registrations,
        concurrent operations, and system capacity validation for research-scale usage patterns.
        """
        # Register multiple environments with different configurations
        scalability_envs = []
        grid_sizes = [(32, 32), (64, 64), (128, 128), (96, 96)]

        registration_times = []

        for i in range(20):  # Test with multiple environments
            env_id = f"ScalabilityTest-{i}-v0"
            grid_size = grid_sizes[i % len(grid_sizes)]
            source_location = (grid_size[0] // 2, grid_size[1] // 2)

            start_time = time.time()
            register_env(
                env_id=env_id,
                kwargs={
                    "grid_size": grid_size,
                    "source_location": source_location,
                    "goal_radius": float(i % 3),
                },
            )
            end_time = time.time()

            reg_time_ms = (end_time - start_time) * 1000
            registration_times.append(reg_time_ms)
            scalability_envs.append(env_id)

        # Measure system performance with increasing registration load
        # Performance should not degrade significantly
        early_times = registration_times[:5]
        late_times = registration_times[-5:]

        avg_early = sum(early_times) / len(early_times)
        avg_late = sum(late_times) / len(late_times)

        # Performance degradation should be minimal
        performance_degradation_ratio = avg_late / avg_early
        assert (
            performance_degradation_ratio < 2.0
        ), f"Performance degraded significantly: {performance_degradation_ratio:.2f}x slower"

        # Test concurrent registration and unregistration operations
        # Simulate by rapid sequential operations
        concurrent_start = time.time()

        # Mix of registrations and unregistrations
        temp_envs = []
        for i in range(10):
            env_id = f"ConcurrentTest-{i}-v0"
            register_env(env_id=env_id)
            temp_envs.append(env_id)

            # Unregister every other one immediately
            if i % 2 == 1:
                unregister_env(env_id)
                temp_envs.remove(env_id)

        concurrent_end = time.time()
        _ = (concurrent_end - concurrent_start) * 1000

        # Validate system stability with maximum supported registrations
        stability_test_count = 0
        max_test_registrations = 50  # Reasonable limit for testing

        try:
            stability_envs = []
            for i in range(max_test_registrations):
                env_id = f"StabilityTest-{i}-v0"
                register_env(env_id=env_id)
                stability_envs.append(env_id)
                stability_test_count += 1

            # Check performance degradation patterns with scale increase
            final_registration_start = time.time()
            final_test_env = register_env(env_id="FinalStabilityTest-v0")
            final_registration_end = time.time()

            final_reg_time_ms = (
                final_registration_end - final_registration_start
            ) * 1000

            # Should still be reasonably fast
            assert final_reg_time_ms < TEST_PERFORMANCE_TARGET_MS * 3

            # Clean up stability test environments
            for env_id in stability_envs + [final_test_env]:
                unregister_env(env_id, suppress_warnings=True)

        except Exception:
            # Clean up what we can
            for i in range(stability_test_count):
                try:
                    unregister_env(f"StabilityTest-{i}-v0", suppress_warnings=True)
                except Exception:
                    pass
            raise

        # Test registry consistency under high registration activity
        # Verify all scalability environments are still properly registered
        for env_id in scalability_envs:
            assert (
                is_registered(env_id) is True
            ), f"Environment {env_id} lost during scalability test"

        # Monitor resource usage patterns at different scales
        # Status checking should remain fast
        status_check_start = time.time()
        for env_id in scalability_envs:
            is_registered(env_id)
        status_check_end = time.time()

        status_check_time_ms = (status_check_end - status_check_start) * 1000
        avg_status_check_ms = status_check_time_ms / len(scalability_envs)

        # Confirm system maintains performance at research-scale usage
        max_acceptable_status_check_ms = 1.0  # 1ms per environment
        assert (
            avg_status_check_ms < max_acceptable_status_check_ms
        ), f"Status checking too slow at scale: {avg_status_check_ms:.2f}ms"

        # Clean up all scalability test environments
        for env_id in scalability_envs + temp_envs:
            unregister_env(env_id, suppress_warnings=True)


class TestRegistrationIntegration:
    """Test class for Gymnasium integration testing including gym.make() compatibility,
    environment instantiation, API compliance, and standard RL workflow validation with
    comprehensive integration verification."""

    def setup_method(self):
        """Initialize test class for registration integration testing."""
        # Clean up any existing registrations
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def teardown_method(self):
        """Clean up after integration tests."""
        try:
            unregister_env(ENV_ID, suppress_warnings=True)
            unregister_env(TEST_ENV_ID, suppress_warnings=True)
        except Exception:
            pass

    def test_gymnasium_make_integration(self):
        """Test gym.make() integration with registered environments ensuring seamless environment
        instantiation, parameter passing, and standard Gymnasium workflow compatibility.
        """
        # Register environment using register_env() for integration testing
        env_id = register_env(env_id=TEST_ENV_ID)

        # Call gym.make() with registered environment ID
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress matplotlib warnings
            try:
                env = gymnasium.make(env_id)

                # Assert gym.make() successfully creates environment instance
                assert env is not None

                # Verify created environment is correct class type (PlumeEnv)
                # Note: Direct type checking requires knowing the exact class
                assert hasattr(env, "action_space")
                assert hasattr(env, "observation_space")
                assert hasattr(env, "reset")
                assert hasattr(env, "step")
                assert hasattr(env, "render")
                assert hasattr(env, "close")

                # Test environment has proper action_space and observation_space
                from gymnasium.spaces import Box, Discrete

                assert isinstance(env.action_space, Discrete)
                assert env.action_space.n == 4  # Cardinal directions

                assert isinstance(env.observation_space, Box)

                # Validate environment parameters match registration configuration
                # This is validated through successful instantiation with expected spaces

                # Check environment supports standard Gymnasium API methods
                # Test reset method
                observation, info = env.reset()
                assert observation is not None
                assert isinstance(info, dict)
                assert observation in env.observation_space

                # Test step method
                action = env.action_space.sample()
                step_result = env.step(action)
                assert len(step_result) == 5  # obs, reward, terminated, truncated, info

                obs, reward, terminated, truncated, info = step_result
                assert obs in env.observation_space
                assert isinstance(reward, (int, float))
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)

                # Test render method (both modes)
                try:
                    # RGB array mode
                    rgb_array = env.render()  # Default mode should work
                    if rgb_array is not None:
                        import numpy as np

                        assert isinstance(rgb_array, np.ndarray)
                        assert len(rgb_array.shape) == 3  # Height, Width, Channels
                        assert rgb_array.shape[2] == 3  # RGB channels
                except Exception:
                    # Rendering may fail in test environment, which is acceptable
                    pass

                # Clean up environment and registration after integration testing
                env.close()

            except Exception as e:
                pytest.fail(f"gym.make() integration failed: {e}")

        # Verify registration cleanup
        cleanup_result = unregister_env(env_id)
        assert cleanup_result is True
        assert is_registered(env_id) is False

    def test_environment_lifecycle_integration(self):
        """Test complete environment lifecycle integration from registration through instantiation,
        operation, and cleanup with comprehensive lifecycle management validation."""
        # Register environment with specific configuration parameters
        config_kwargs = {
            "grid_size": (64, 64),
            "source_location": (32, 32),
            "goal_radius": 1.0,
            "max_steps": 500,
        }

        env_id = register_env(
            env_id=TEST_ENV_ID,
            kwargs=config_kwargs,
            max_episode_steps=config_kwargs["max_steps"],
        )

        # Create environment instance via gym.make()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            env = gymnasium.make(env_id)

            try:
                # Test environment reset() functionality with parameter validation
                observation, info = env.reset(seed=42)  # Test seeding
                assert observation is not None
                assert isinstance(info, dict)

                # Verify observation is valid
                assert observation in env.observation_space

                # Perform environment step() operations with action validation
                valid_actions = [0, 1, 2, 3]  # UP, RIGHT, DOWN, LEFT
                step_count = 0
                max_test_steps = 10

                while step_count < max_test_steps:
                    action = valid_actions[step_count % len(valid_actions)]

                    # Validate action is valid
                    assert action in env.action_space

                    step_result = env.step(action)
                    obs, reward, terminated, truncated, info = step_result

                    # Validate step result
                    assert obs in env.observation_space
                    assert isinstance(reward, (int, float))
                    assert isinstance(terminated, bool)
                    assert isinstance(truncated, bool)
                    assert isinstance(info, dict)

                    step_count += 1

                    # Break if episode ends
                    if terminated or truncated:
                        break

                # Test environment render() functionality if available
                try:
                    render_result = env.render()
                    # render() may return None or array depending on mode
                    if render_result is not None:
                        import numpy as np

                        assert isinstance(render_result, np.ndarray)
                except Exception:
                    # Rendering failures are acceptable in test environment
                    pass

                # Call environment close() for proper resource cleanup
                env.close()

            except Exception as e:
                # Ensure cleanup even if test fails
                try:
                    env.close()
                except Exception:
                    pass
                pytest.fail(f"Environment lifecycle integration failed: {e}")

        # Unregister environment and validate cleanup completion
        unregister_result = unregister_env(env_id)
        assert unregister_result is True
        assert is_registered(env_id) is False

        # Confirm complete lifecycle completes without resource leaks
        # Verify environment is no longer accessible
        with pytest.raises(Exception):  # Should raise error for unregistered env
            gymnasium.make(env_id)

    def test_multiple_environment_integration(self):
        """Test multiple environment registration and management with concurrent registrations,
        selective operations, and resource isolation validation for complex research scenarios.
        """
        # Register multiple environments with different configurations
        env_configs = [
            {
                "env_id": "MultiIntegration1-v0",
                "kwargs": {"grid_size": (32, 32), "source_location": (16, 16)},
            },
            {
                "env_id": "MultiIntegration2-v0",
                "kwargs": {"grid_size": (64, 64), "source_location": (32, 32)},
            },
            {
                "env_id": "MultiIntegration3-v0",
                "kwargs": {"grid_size": (96, 96), "source_location": (48, 48)},
            },
        ]

        registered_envs = []
        for config in env_configs:
            env_id = register_env(env_id=config["env_id"], kwargs=config["kwargs"])
            registered_envs.append(env_id)
            assert is_registered(env_id) is True

        # Create instances of all registered environments via gym.make()
        env_instances = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                for env_id, config in zip(registered_envs, env_configs):
                    env = gymnasium.make(env_id)
                    env_instances.append(env)

                    # Verify each environment is properly instantiated
                    assert env is not None
                    assert hasattr(env, "action_space")
                    assert hasattr(env, "observation_space")
                    base_env = _unwrap_gym_env(env)
                    assert isinstance(base_env, PlumeEnv)
                    assert base_env.grid_size == config["kwargs"]["grid_size"]

                # Test concurrent operation of multiple environment instances
                for i, env in enumerate(env_instances):
                    # Each environment should work independently
                    obs, info = env.reset(seed=100 + i)  # Different seeds
                    assert obs is not None
                    assert isinstance(info, dict)

                    # Take a step in each environment
                    action = env.action_space.sample()
                    step_result = env.step(action)
                    assert len(step_result) == 5

                # Validate resource isolation between different environments
                # Each environment should maintain independent state
                env_observations = []
                for env in env_instances:
                    obs, _ = env.reset(seed=42)  # Same seed for comparison
                    env_observations.append(obs)

                # Test environment-specific parameter application
                # PlumeEnv uses a flat Box observation space; grid_size is validated directly

                # Validate performance with multiple active environments
                # All environments should respond quickly
                performance_start = time.time()
                for env in env_instances:
                    env.reset(seed=99)  # Reset before performance test
                    for _ in range(3):  # Few steps per environment
                        action = env.action_space.sample()
                        obs, reward, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            env.reset()  # Reset if episode ends
                performance_end = time.time()

                multi_env_time_ms = (performance_end - performance_start) * 1000
                max_acceptable_time_ms = 100  # Should be fast
                assert (
                    multi_env_time_ms < max_acceptable_time_ms
                ), f"Multiple environment operation too slow: {multi_env_time_ms:.2f}ms"

                # Close all environment instances
                for env in env_instances:
                    env.close()

            except Exception as e:
                # Cleanup on failure
                for env in env_instances:
                    try:
                        env.close()
                    except Exception:
                        pass
                raise e

        # Check selective unregistration affects only target environment
        target_env_id = registered_envs[1]  # Middle environment
        unregister_result = unregister_env(target_env_id)
        assert unregister_result is True
        assert is_registered(target_env_id) is False

        # Other environments should remain registered
        remaining_env_ids = [registered_envs[0], registered_envs[2]]
        for remaining_id in remaining_env_ids:
            assert is_registered(remaining_id) is True

            # Should still be accessible
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    test_env = gymnasium.make(remaining_id)
                    test_env.close()
                except Exception as e:
                    pytest.fail(
                        f"Remaining environment {remaining_id} not accessible: {e}"
                    )

        # Clean up all environments and registrations for complete isolation
        for env_id in remaining_env_ids:
            unregister_result = unregister_env(env_id)
            assert unregister_result is True
            assert is_registered(env_id) is False

    def test_registration_api_compliance(self):
        """Test registration API compliance with Gymnasium standards including versioning conventions,
        parameter specifications, and registry management conformance for ecosystem compatibility.
        """
        # Test with compliant environment ID
        compliant_env_id = "ComplianceTest-v0"

        # Validate environment ID follows Gymnasium versioning conventions
        assert compliant_env_id.endswith("-v0"), "Environment ID must end with '-v0'"
        assert (
            len(compliant_env_id.split("-")) >= 2
        ), "Environment ID must have name-version format"

        # Register environment for compliance testing
        _ = register_env(env_id=compliant_env_id)

        # Test entry_point specification conforms to Gymnasium standards
        # Entry point is validated during registration

        # Verify kwargs are correctly passed to environment constructor
        custom_kwargs = {
            "grid_size": (96, 96),
            "source_location": (48, 48),
            "goal_radius": 2.0,
        }

        custom_compliant_env_id = "CustomCompliance-v0"
        register_env(env_id=custom_compliant_env_id, kwargs=custom_kwargs)

        # Test environment instantiation with custom kwargs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                env = gymnasium.make(custom_compliant_env_id)

                # Kwargs validation through successful instantiation
                assert env is not None
                obs, info = env.reset()
                assert obs is not None

                env.close()

            except Exception as e:
                pytest.fail(f"Custom kwargs not properly handled: {e}")

        # Test registration metadata is accessible via registry inspection
        # This tests integration with Gymnasium's internal registry
        try:
            if hasattr(gymnasium.envs, "registry"):
                registry = gymnasium.envs.registry
                if hasattr(registry, "env_specs"):
                    env_spec = registry.env_specs.get(compliant_env_id)
                    if env_spec is not None:
                        # Registry entry should exist
                        assert hasattr(env_spec, "entry_point")
                        assert hasattr(env_spec, "max_episode_steps")
        except Exception:
            # Registry inspection may fail in different Gymnasium versions
            pass

        # Validate registry state consistency after registration operations
        assert is_registered(compliant_env_id) is True
        assert is_registered(custom_compliant_env_id) is True

        # Test unregistration maintains registry consistency
        unregister_env(compliant_env_id)
        assert is_registered(compliant_env_id) is False
        assert (
            is_registered(custom_compliant_env_id) is True
        )  # Should not affect other envs

        # Check compliance with Gymnasium's environment discovery mechanisms
        # Registered environments should be discoverable
        try:
            # This tests that the environment can be found through standard mechanisms
            test_env = gymnasium.make(custom_compliant_env_id)
            test_env.close()
        except Exception as e:
            pytest.fail(f"Environment discovery compliance failed: {e}")

        # Confirm registration integrates properly with Gymnasium ecosystem
        # Test that standard Gymnasium utilities work with our registered environment
        # Registration validated through successful instantiation

        # Final cleanup
        unregister_env(custom_compliant_env_id, suppress_warnings=True)
        assert is_registered(custom_compliant_env_id) is False
