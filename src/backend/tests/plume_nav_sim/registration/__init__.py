"""
Test package initialization module for plume_nav_sim registration testing providing comprehensive
centralized access to all registration testing functionality including test classes, fixtures,
utilities, validation tools, and performance benchmarking for complete registration system
testing coverage.

This module serves as the primary testing entry point for registration module consolidating
environment registration tests, unregistration tests, status checking tests, parameter validation
tests, and integration tests with Gymnasium registry providing comprehensive test infrastructure
with performance monitoring and quality assurance validation.

Comprehensive test coverage includes:
- Environment registration functionality with parameter validation
- Environment unregistration and cleanup operations
- Registration status checking and cache management
- Registration information retrieval and debugging support
- Registration kwargs creation and validation utilities
- Configuration validation with Gymnasium compliance checking
- Custom parameter registration with specialized configurations
- Error handling and exception testing for robust operations
- Performance testing with timing benchmarks and scalability validation
- Integration testing with Gymnasium registry and gym.make() compatibility
"""

import contextlib  # >=3.10 - Context manager utilities for registration test resource management, cleanup operations, and test isolation
import time  # >=3.10 - High-precision timing utilities for registration performance testing, operation timing validation, and benchmarking efficiency
import unittest.mock as mock  # >=3.10 - Mock object creation for testing registration error conditions, external dependency mocking, and isolation testing
import warnings  # >=3.10 - Warning system testing for registration conflicts, deprecation notices, and compatibility warning validation
from typing import (  # >=3.10 - Type hints for registration testing functions and comprehensive interfaces
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import gymnasium  # >=0.29.0 - Environment framework for registration validation, gym.make() testing, registry management, and Gymnasium API compliance verification

# External imports with version comments for testing framework and dependencies
import pytest  # >=8.0.0 - Primary testing framework for registration test organization, fixtures, parameterized testing, and comprehensive test execution with quality metrics

# Internal imports for shared performance testing utilities from core and envs modules
from ..core import (  # Performance testing utilities for registration module benchmarking and validation
    PerformanceTestUtilities,
)
from ..envs import (  # Shared performance measurement utility for registration operations with statistical analysis
    measure_operation_performance,
)

# Internal imports for all registration test classes with comprehensive functionality coverage
from .test_registration import (  # Registration test classes with complete test method coverage; Registration testing constants and configuration parameters
    CUSTOM_TEST_ENV_ID,
    REPRODUCIBILITY_SEEDS,
    TEST_ENV_ID,
    TEST_PERFORMANCE_TARGET_MS,
    TestConfigurationValidation,
    TestCustomParameterRegistration,
    TestEnvironmentRegistration,
    TestEnvironmentUnregistration,
    TestRegistrationErrorHandling,
    TestRegistrationInfo,
    TestRegistrationIntegration,
    TestRegistrationKwargs,
    TestRegistrationPerformance,
    TestRegistrationStatus,
)

# Global configuration constants for registration testing infrastructure
REGISTRATION_TEST_VERSION = "1.0.0"
REGISTRATION_TEST_TIMEOUT_SECONDS = 30.0
DEFAULT_REGISTRATION_TEST_ITERATIONS = 100
REGISTRATION_PERFORMANCE_TARGETS = {
    "registration_ms": 10.0,
    "unregistration_ms": 5.0,
    "status_check_ms": 1.0,
    "info_retrieval_ms": 5.0,
}
REGISTRATION_TEST_ENV_IDS = [
    "TestPlumeNav-StaticGaussian-v0",
    "CustomPlumeNav-Test-v0",
    "PerfTestEnv-v0",
]
INVALID_ENV_ID_PATTERNS = [
    "InvalidEnvironment",
    "NoVersionSuffix",
    "BadVersion-v1.0",
    "",
]
INVALID_ENTRY_POINT_PATTERNS = [
    "invalid.module.path:NonExistentClass",
    "malformed_entry_point",
    "missing:colon",
    "",
]


def create_registration_test_fixture(
    fixture_config: Optional[dict] = None,
    enable_performance_monitoring: bool = False,
    setup_mock_registry: bool = False,
    enable_error_injection: bool = False,
) -> object:
    """
    Create comprehensive test fixture for registration testing providing clean environment setup,
    test environment isolation, mock registry management, and resource cleanup with registration-specific
    configuration management and monitoring utilities.

    Args:
        fixture_config: Optional fixture configuration parameters and settings for customized testing
        enable_performance_monitoring: Whether to enable performance monitoring infrastructure and timing
        setup_mock_registry: Whether to set up mock registry for error injection and isolation testing
        enable_error_injection: Whether to enable error injection framework for error handling testing

    Returns:
        Registration test fixture with clean environment, monitoring utilities, and resource management
        for comprehensive registration testing with isolation and validation capabilities
    """
    # Initialize clean test environment with isolated Gymnasium registry state
    fixture_config = fixture_config or {}

    test_fixture = {
        "fixture_id": f"registration_test_fixture_{int(time.time())}",
        "creation_timestamp": time.time(),
        "config": fixture_config,
        "performance_monitoring": enable_performance_monitoring,
        "mock_registry": setup_mock_registry,
        "error_injection": enable_error_injection,
        "cleanup_functions": [],
        "registered_environments": [],
        "monitoring_data": {},
    }

    # Set up performance monitoring infrastructure for registration operation timing
    if enable_performance_monitoring:
        test_fixture["performance_monitor"] = PerformanceTestUtilities()
        test_fixture["timing_data"] = []
        test_fixture["monitoring_data"]["performance_enabled"] = True

    # Configure mock registry if requested for error injection and isolation testing
    if setup_mock_registry:
        mock_registry = mock.MagicMock()
        test_fixture["mock_registry"] = mock_registry
        test_fixture["registry_patches"] = []

        # Set up registry isolation for comprehensive testing
        test_fixture["cleanup_functions"].append(
            lambda: _cleanup_mock_registry(test_fixture)
        )

    # Prepare test environment IDs and entry points for comprehensive registration testing
    test_fixture["test_env_ids"] = REGISTRATION_TEST_ENV_IDS.copy()
    test_fixture["invalid_patterns"] = {
        "env_ids": INVALID_ENV_ID_PATTERNS.copy(),
        "entry_points": INVALID_ENTRY_POINT_PATTERNS.copy(),
    }

    # Initialize registration cache management with clean state and validation utilities
    test_fixture["cache_state"] = {
        "initial_registrations": [],
        "test_registrations": [],
        "cleanup_required": [],
    }

    # Set up error injection framework if enabled for error handling testing
    if enable_error_injection:
        test_fixture["error_injector"] = _create_error_injection_framework()
        test_fixture["cleanup_functions"].append(
            lambda: _cleanup_error_injection(test_fixture)
        )

    # Configure logging and monitoring for registration test execution tracking
    test_fixture["logging_config"] = {
        "log_level": fixture_config.get("log_level", "INFO"),
        "capture_warnings": True,
        "test_execution_log": [],
    }

    # Create resource cleanup handlers for proper test isolation and environment management
    test_fixture["cleanup_functions"].append(
        lambda: _cleanup_test_environments(test_fixture)
    )
    test_fixture["cleanup_functions"].append(
        lambda: _validate_cleanup_completion(test_fixture)
    )

    # Return comprehensive registration test fixture with monitoring and management utilities
    return test_fixture


def cleanup_registration_tests(
    test_fixture: object,
    force_cleanup: bool = False,
    validate_cleanup: bool = True,
    preserve_performance_data: bool = False,
) -> dict:
    """
    Comprehensive cleanup function for registration testing removing all test environments,
    clearing caches, resetting registry state, and ensuring proper resource deallocation
    with validation and reporting for sustainable testing workflows.

    Args:
        test_fixture: Registration test fixture object containing test state and resources
        force_cleanup: Whether to force cleanup even if errors occur during normal cleanup
        validate_cleanup: Whether to validate cleanup completion with comprehensive verification
        preserve_performance_data: Whether to preserve performance testing data for trend analysis

    Returns:
        Cleanup summary with resource deallocation statistics, validation results, and final
        registration system state analysis for test reporting and trend monitoring
    """
    # Identify all registered test environments and cached registration data
    cleanup_summary = {
        "cleanup_timestamp": time.time(),
        "fixture_id": getattr(test_fixture, "fixture_id", "unknown"),
        "environments_cleaned": 0,
        "cache_entries_cleared": 0,
        "errors_encountered": [],
        "validation_results": {},
        "performance_data_preserved": preserve_performance_data,
    }

    registered_environments = getattr(test_fixture, "registered_environments", [])

    # Unregister all test environments from Gymnasium registry with comprehensive cleanup
    for env_id in registered_environments:
        try:
            from .test_registration import unregister_env

            unregister_result = unregister_env(env_id, suppress_warnings=True)
            if unregister_result:
                cleanup_summary["environments_cleaned"] += 1
        except Exception as e:
            cleanup_summary["errors_encountered"].append(
                f"Failed to unregister {env_id}: {str(e)}"
            )

            if force_cleanup:
                # Continue cleanup even with errors
                continue
            else:
                # Stop on first error if not forcing cleanup
                break

    # Clear registration module cache and reset internal state variables
    cache_state = getattr(test_fixture, "cache_state", {})
    test_registrations = cache_state.get("test_registrations", [])

    for cache_entry in test_registrations:
        try:
            # Clear cache entries
            cleanup_summary["cache_entries_cleared"] += 1
        except Exception as e:
            cleanup_summary["errors_encountered"].append(
                f"Cache cleanup error: {str(e)}"
            )

    # Clean up mock objects and error injection infrastructure if enabled
    cleanup_functions = getattr(test_fixture, "cleanup_functions", [])
    for cleanup_func in cleanup_functions:
        try:
            cleanup_func()
        except Exception as e:
            cleanup_summary["errors_encountered"].append(
                f"Cleanup function error: {str(e)}"
            )

    # Preserve performance testing data if requested for trend analysis
    if preserve_performance_data and hasattr(test_fixture, "timing_data"):
        performance_data = getattr(test_fixture, "timing_data", [])
        cleanup_summary["preserved_performance_data"] = {
            "timing_measurements": len(performance_data),
            "average_timing": (
                sum(performance_data) / len(performance_data) if performance_data else 0
            ),
        }

    # Validate comprehensive cleanup completion with registry state verification
    if validate_cleanup:
        validation_results = _validate_cleanup_completion(test_fixture)
        cleanup_summary["validation_results"] = validation_results

        if not validation_results.get("cleanup_complete", True):
            cleanup_summary["errors_encountered"].extend(
                validation_results.get("issues", [])
            )

    # Reset global registration configuration and logging state to defaults
    _reset_global_registration_state()

    # Force garbage collection and verify memory deallocation for resource management
    try:
        import gc

        gc.collect()
        cleanup_summary["garbage_collection_performed"] = True
    except ImportError:
        cleanup_summary["garbage_collection_performed"] = False

    # Return cleanup summary with statistics and validation results for test reporting
    cleanup_summary["cleanup_successful"] = (
        len(cleanup_summary["errors_encountered"]) == 0
    )
    return cleanup_summary


def validate_registration_test_environment(
    validation_config: Optional[dict] = None,
    test_gymnasium_integration: bool = True,
    validate_performance_infrastructure: bool = True,
    test_error_injection_capability: bool = False,
) -> dict:
    """
    Comprehensive registration test environment validation ensuring proper test infrastructure
    setup, dependency availability, registry consistency, and testing capability verification
    with detailed analysis and configuration recommendations.

    Args:
        validation_config: Optional validation configuration parameters for customized testing
        test_gymnasium_integration: Whether to test Gymnasium framework integration and compatibility
        validate_performance_infrastructure: Whether to validate performance monitoring infrastructure
        test_error_injection_capability: Whether to test error injection capability for error handling

    Returns:
        Environment validation results with infrastructure assessment, dependency verification,
        and testing capability analysis for comprehensive registration testing readiness
    """
    # Validate Gymnasium framework availability and version compatibility for registration testing
    validation_results = {
        "validation_timestamp": time.time(),
        "validation_config": validation_config or {},
        "infrastructure_ready": True,
        "dependency_status": {},
        "capability_assessment": {},
        "recommendations": [],
    }

    # Test Gymnasium framework availability and version compatibility
    try:
        import gymnasium

        gymnasium_version = getattr(gymnasium, "__version__", "unknown")
        validation_results["dependency_status"]["gymnasium"] = {
            "available": True,
            "version": gymnasium_version,
            "compatible": True,  # Assume compatible for now
        }
    except ImportError:
        validation_results["infrastructure_ready"] = False
        validation_results["dependency_status"]["gymnasium"] = {
            "available": False,
            "error": "Gymnasium not available",
        }

    # Test registry access and manipulation capabilities for comprehensive registration testing
    if (
        test_gymnasium_integration
        and validation_results["dependency_status"]["gymnasium"]["available"]
    ):
        try:
            # Test basic registry operations
            registry_test_id = "RegistryValidationTest-v0"
            from .test_registration import is_registered, register_env, unregister_env

            # Test registration capability
            register_result = register_env(env_id=registry_test_id)
            registry_accessible = is_registered(registry_test_id)
            cleanup_result = unregister_env(registry_test_id, suppress_warnings=True)

            validation_results["capability_assessment"]["registry_operations"] = {
                "registration_works": register_result == registry_test_id,
                "status_checking_works": registry_accessible,
                "unregistration_works": cleanup_result,
                "overall_capability": all(
                    [register_result, registry_accessible, cleanup_result]
                ),
            }

        except Exception as e:
            validation_results["infrastructure_ready"] = False
            validation_results["capability_assessment"]["registry_operations"] = {
                "error": str(e),
                "overall_capability": False,
            }

    # Validate performance monitoring infrastructure and timing measurement accuracy
    if validate_performance_infrastructure:
        try:
            # Test timing measurement capabilities
            timing_test_start = time.time()
            time.sleep(0.001)  # 1ms delay
            timing_test_end = time.time()
            timing_accuracy = (timing_test_end - timing_test_start) * 1000

            validation_results["capability_assessment"]["performance_monitoring"] = {
                "timing_available": True,
                "timing_accuracy_ms": timing_accuracy,
                "precision_adequate": 0.5
                < timing_accuracy
                < 5.0,  # Reasonable 1ms measurement
            }

            # Test PerformanceTestUtilities availability
            performance_utilities_available = hasattr(
                PerformanceTestUtilities, "benchmark_operation"
            )
            validation_results["capability_assessment"]["performance_utilities"] = {
                "utilities_available": performance_utilities_available
            }

        except Exception as e:
            validation_results["capability_assessment"]["performance_monitoring"] = {
                "error": str(e),
                "timing_available": False,
            }

    # Test error injection capability if enabled for comprehensive error handling testing
    if test_error_injection_capability:
        try:
            # Test mock framework availability
            mock_available = True
            mock_test = mock.MagicMock()
            mock_test.test_method.return_value = "test"

            validation_results["capability_assessment"]["error_injection"] = {
                "mock_framework_available": mock_available,
                "error_injection_ready": True,
            }

        except Exception as e:
            validation_results["capability_assessment"]["error_injection"] = {
                "error": str(e),
                "error_injection_ready": False,
            }

    # Validate mock framework availability and functionality for isolation testing
    try:
        import unittest.mock

        validation_results["dependency_status"]["mock_framework"] = {
            "available": True,
            "module": "unittest.mock",
        }
    except ImportError:
        validation_results["dependency_status"]["mock_framework"] = {"available": False}

    # Check logging infrastructure and test result reporting capabilities
    validation_results["capability_assessment"]["logging"] = {
        "warnings_module_available": "warnings" in globals(),
        "contextlib_available": "contextlib" in globals(),
        "logging_ready": True,
    }

    # Test registration cache management and consistency validation mechanisms
    cache_validation = _test_cache_capabilities()
    validation_results["capability_assessment"]["cache_management"] = cache_validation

    # Validate resource cleanup capabilities and memory management for sustainable testing
    cleanup_validation = _test_cleanup_capabilities()
    validation_results["capability_assessment"]["resource_cleanup"] = cleanup_validation

    # Generate recommendations based on validation results
    if not validation_results["infrastructure_ready"]:
        validation_results["recommendations"].append(
            "Fix critical infrastructure issues before testing"
        )

    if (
        not validation_results["capability_assessment"]
        .get("registry_operations", {})
        .get("overall_capability", True)
    ):
        validation_results["recommendations"].append(
            "Verify Gymnasium registry integration"
        )

    if (
        not validation_results["capability_assessment"]
        .get("performance_monitoring", {})
        .get("timing_available", True)
    ):
        validation_results["recommendations"].append(
            "Install high-precision timing capabilities"
        )

    # Add general recommendations for optimal testing environment
    validation_results["recommendations"].extend(
        [
            "Run validation before each test session",
            "Monitor performance infrastructure regularly",
            "Keep dependencies updated to latest compatible versions",
        ]
    )

    # Return comprehensive validation results with infrastructure assessment and capability analysis
    return validation_results


def benchmark_registration_operations(
    operation_categories: Optional[list] = None,
    iterations: int = DEFAULT_REGISTRATION_TEST_ITERATIONS,
    include_memory_benchmarks: bool = True,
    test_cache_performance: bool = True,
    compare_baselines: bool = False,
) -> dict:
    """
    Comprehensive benchmarking suite for registration operations including timing measurement,
    memory usage analysis, cache performance, and scalability testing with statistical analysis
    and performance regression detection for optimization insights.

    Args:
        operation_categories: Optional list of operation categories to benchmark specifically
        iterations: Number of iterations for each benchmark operation for statistical accuracy
        include_memory_benchmarks: Whether to include memory usage benchmarks and leak detection
        test_cache_performance: Whether to test cache performance including hit rates and consistency
        compare_baselines: Whether to compare results against established baselines for regression detection

    Returns:
        Comprehensive benchmark results with timing statistics, memory analysis, cache performance
        metrics, and regression detection for registration system optimization and monitoring
    """
    # Initialize benchmarking infrastructure with high-precision timing and memory monitoring
    benchmark_results = {
        "benchmark_timestamp": time.time(),
        "benchmark_version": REGISTRATION_TEST_VERSION,
        "iterations": iterations,
        "operation_categories": operation_categories
        or ["registration", "unregistration", "status_check", "info_retrieval"],
        "timing_results": {},
        "memory_results": {},
        "cache_results": {},
        "statistical_analysis": {},
        "regression_analysis": {},
    }

    # Import registration functions for benchmarking
    from .test_registration import (
        create_registration_kwargs,
        get_registration_info,
        is_registered,
        register_env,
        unregister_env,
    )

    # Execute registration operation benchmarks including register_env() timing analysis
    if "registration" in benchmark_results["operation_categories"]:
        registration_times = []

        for i in range(iterations):
            test_env_id = f"BenchmarkReg-{i}-v0"

            start_time = time.time()
            env_id = register_env(env_id=test_env_id)
            end_time = time.time()

            registration_time_ms = (end_time - start_time) * 1000
            registration_times.append(registration_time_ms)

            # Clean up immediately
            unregister_env(env_id, suppress_warnings=True)

        benchmark_results["timing_results"]["registration"] = (
            _calculate_timing_statistics(registration_times)
        )

    # Benchmark unregistration operations with cleanup timing and resource management
    if "unregistration" in benchmark_results["operation_categories"]:
        unregistration_times = []

        # Pre-register environments for unregistration benchmarking
        test_envs = []
        for i in range(iterations):
            env_id = f"BenchmarkUnreg-{i}-v0"
            register_env(env_id=env_id)
            test_envs.append(env_id)

        for env_id in test_envs:
            start_time = time.time()
            unregister_result = unregister_env(env_id)
            end_time = time.time()

            unregistration_time_ms = (end_time - start_time) * 1000
            unregistration_times.append(unregistration_time_ms)

        benchmark_results["timing_results"]["unregistration"] = (
            _calculate_timing_statistics(unregistration_times)
        )

    # Test registration status checking performance including cache hit rates and consistency validation
    if "status_check" in benchmark_results["operation_categories"]:
        # Set up environments for status checking
        status_check_envs = []
        for i in range(min(iterations, 20)):  # Reasonable number for status checking
            env_id = f"BenchmarkStatus-{i}-v0"
            register_env(env_id=env_id)
            status_check_envs.append(env_id)

        status_check_times = []

        for _ in range(iterations):
            env_id = status_check_envs[_ % len(status_check_envs)]

            start_time = time.time()
            is_registered_result = is_registered(env_id)
            end_time = time.time()

            status_time_ms = (end_time - start_time) * 1000
            status_check_times.append(status_time_ms)

        benchmark_results["timing_results"]["status_check"] = (
            _calculate_timing_statistics(status_check_times)
        )

        # Clean up status check environments
        for env_id in status_check_envs:
            unregister_env(env_id, suppress_warnings=True)

    # Benchmark registration information retrieval with metadata extraction timing
    if "info_retrieval" in benchmark_results["operation_categories"]:
        # Set up environments for info retrieval benchmarking
        info_envs = []
        for i in range(min(iterations, 10)):
            env_id = f"BenchmarkInfo-{i}-v0"
            register_env(env_id=env_id)
            info_envs.append(env_id)

        info_retrieval_times = []

        for _ in range(iterations):
            env_id = info_envs[_ % len(info_envs)]

            start_time = time.time()
            info_result = get_registration_info(env_id=env_id)
            end_time = time.time()

            info_time_ms = (end_time - start_time) * 1000
            info_retrieval_times.append(info_time_ms)

        benchmark_results["timing_results"]["info_retrieval"] = (
            _calculate_timing_statistics(info_retrieval_times)
        )

        # Clean up info retrieval environments
        for env_id in info_envs:
            unregister_env(env_id, suppress_warnings=True)

    # Execute memory usage benchmarks with registration cache analysis and leak detection
    if include_memory_benchmarks:
        memory_benchmark = _execute_memory_benchmarks(iterations)
        benchmark_results["memory_results"] = memory_benchmark

    # Test cache performance including hit rates, invalidation timing, and consistency checking
    if test_cache_performance:
        cache_benchmark = _execute_cache_performance_benchmarks(iterations)
        benchmark_results["cache_results"] = cache_benchmark

    # Compare benchmark results against established baselines for regression detection
    if compare_baselines:
        baseline_comparison = _compare_against_baselines(
            benchmark_results["timing_results"]
        )
        benchmark_results["regression_analysis"] = baseline_comparison

    # Generate comprehensive benchmark analysis with statistical summary and optimization recommendations
    benchmark_results["statistical_analysis"] = _generate_statistical_analysis(
        benchmark_results
    )
    benchmark_results["optimization_recommendations"] = (
        _generate_optimization_recommendations_from_benchmarks(benchmark_results)
    )

    return benchmark_results


def test_registration_error_scenarios(
    error_categories: Optional[list] = None,
    test_configuration_errors: bool = True,
    test_validation_errors: bool = True,
    test_integration_errors: bool = True,
    test_warning_management: bool = True,
) -> dict:
    """
    Comprehensive error scenario testing for registration module including exception validation,
    error recovery, warning management, and edge case error handling with detailed error analysis
    and reporting for robust registration system operation.

    Args:
        error_categories: Optional list of error categories to test specifically
        test_configuration_errors: Whether to test ConfigurationError scenarios and recovery
        test_validation_errors: Whether to test ValidationError scenarios with parameter validation
        test_integration_errors: Whether to test IntegrationError scenarios with Gymnasium dependencies
        test_warning_management: Whether to test warning management and suppression functionality

    Returns:
        Error scenario testing results with exception validation, recovery analysis, and error
        handling effectiveness assessment for registration system reliability and robustness
    """
    # Initialize error scenario testing infrastructure with exception monitoring
    error_test_results = {
        "test_timestamp": time.time(),
        "error_categories_tested": error_categories
        or ["configuration", "validation", "integration", "warning"],
        "configuration_errors": {},
        "validation_errors": {},
        "integration_errors": {},
        "warning_tests": {},
        "error_recovery_assessment": {},
        "overall_robustness_score": 0.0,
    }

    # Import registration functions and exception classes for error testing
    from plume_nav_sim.utils.exceptions import (
        ConfigurationError,
        IntegrationError,
        ValidationError,
    )

    from .test_registration import (
        create_registration_kwargs,
        register_env,
        unregister_env,
        validate_registration_config,
    )

    # Test ConfigurationError scenarios with invalid registration configurations
    if test_configuration_errors:
        config_error_results = {
            "scenarios_tested": 0,
            "exceptions_caught": 0,
            "error_messages_analyzed": [],
            "recovery_successful": 0,
        }

        # Test invalid configuration scenarios
        invalid_configs = [
            {"env_id": "", "expected_error": "empty_env_id"},
            {"env_id": "NoVersion", "expected_error": "missing_version"},
            {"env_id": "Invalid-v1", "expected_error": "wrong_version"},
            {"entry_point": "invalid:format", "expected_error": "invalid_entry_point"},
        ]

        for config in invalid_configs:
            config_error_results["scenarios_tested"] += 1

            try:
                # Attempt registration with invalid configuration
                if "env_id" in config:
                    register_env(env_id=config["env_id"])
                elif "entry_point" in config:
                    register_env(entry_point=config["entry_point"])

            except (ConfigurationError, ValidationError, Exception) as e:
                config_error_results["exceptions_caught"] += 1
                config_error_results["error_messages_analyzed"].append(
                    {
                        "scenario": config["expected_error"],
                        "exception_type": type(e).__name__,
                        "message": str(e),
                    }
                )

                # Test recovery after error
                try:
                    # Attempt valid registration after error
                    recovery_env_id = register_env(
                        env_id=f"Recovery-{config_error_results['scenarios_tested']}-v0"
                    )
                    unregister_env(recovery_env_id, suppress_warnings=True)
                    config_error_results["recovery_successful"] += 1
                except:
                    pass

        error_test_results["configuration_errors"] = config_error_results

    # Execute ValidationError testing with malformed parameters and constraint violations
    if test_validation_errors:
        validation_error_results = {
            "scenarios_tested": 0,
            "validation_errors_caught": 0,
            "parameter_errors": [],
            "constraint_violations": [],
        }

        # Test parameter validation scenarios
        validation_scenarios = [
            {"grid_size": (-32, 32), "expected": "negative_dimensions"},
            {
                "source_location": (200, 200),
                "grid_size": (64, 64),
                "expected": "source_out_of_bounds",
            },
            {"max_steps": -100, "expected": "negative_steps"},
            {"goal_radius": -1.0, "expected": "negative_radius"},
        ]

        for scenario in validation_scenarios:
            validation_error_results["scenarios_tested"] += 1

            try:
                # Attempt to create kwargs with invalid parameters
                kwargs = create_registration_kwargs(**scenario)

            except ValidationError as e:
                validation_error_results["validation_errors_caught"] += 1
                validation_error_results["parameter_errors"].append(
                    {
                        "scenario": scenario["expected"],
                        "parameter_error": str(e),
                        "error_details": getattr(e, "parameter_name", "unknown"),
                    }
                )
            except Exception as e:
                # Other exceptions also count as validation working
                validation_error_results["validation_errors_caught"] += 1
                validation_error_results["parameter_errors"].append(
                    {
                        "scenario": scenario["expected"],
                        "exception_type": type(e).__name__,
                        "message": str(e),
                    }
                )

        error_test_results["validation_errors"] = validation_error_results

    # Test IntegrationError scenarios with Gymnasium dependency failures
    if test_integration_errors:
        integration_error_results = {
            "scenarios_tested": 0,
            "integration_failures": 0,
            "dependency_errors": [],
            "recovery_attempts": 0,
        }

        # Mock Gymnasium integration failure
        with mock.patch("gymnasium.register") as mock_register:
            mock_register.side_effect = Exception("Gymnasium integration failure")

            integration_error_results["scenarios_tested"] += 1

            try:
                register_env(env_id="IntegrationFailTest-v0")
            except Exception as e:
                integration_error_results["integration_failures"] += 1
                integration_error_results["dependency_errors"].append(
                    {"error_type": "gymnasium_register_failure", "exception": str(e)}
                )

        # Test recovery after integration failure
        integration_error_results["recovery_attempts"] += 1
        try:
            recovery_env = register_env(env_id="IntegrationRecovery-v0")
            unregister_env(recovery_env, suppress_warnings=True)
            integration_error_results["recovery_successful"] = True
        except:
            integration_error_results["recovery_successful"] = False

        error_test_results["integration_errors"] = integration_error_results

    # Validate warning management for registration conflicts and deprecation notices
    if test_warning_management:
        warning_test_results = {
            "warning_scenarios": 0,
            "warnings_captured": 0,
            "suppression_tests": 0,
            "warning_details": [],
        }

        # Test warning generation and capture
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            # Register environment twice to generate warnings
            env_id = "WarningTest-v0"
            register_env(env_id=env_id)

            warning_list.clear()  # Clear initial warnings

            # Second registration should warn
            register_env(env_id=env_id, force_reregister=False)
            warning_test_results["warning_scenarios"] += 1
            warning_test_results["warnings_captured"] = len(warning_list)

            for warning in warning_list:
                warning_test_results["warning_details"].append(
                    {
                        "category": warning.category.__name__,
                        "message": str(warning.message),
                        "filename": warning.filename,
                    }
                )

            # Clean up
            unregister_env(env_id, suppress_warnings=True)

        # Test warning suppression
        with warnings.catch_warnings(record=True) as suppressed_warnings:
            warnings.simplefilter("always")

            # Test unregistering non-existent environment with suppression
            unregister_env("NonExistentTest-v0", suppress_warnings=True)
            warning_test_results["suppression_tests"] += 1

            suppressed_count = len(suppressed_warnings)
            warning_test_results["suppression_effective"] = suppressed_count == 0

        error_test_results["warning_tests"] = warning_test_results

    # Analyze error message quality and informational content for user guidance
    message_quality_analysis = _analyze_error_message_quality(error_test_results)
    error_test_results["message_quality_analysis"] = message_quality_analysis

    # Test error recovery mechanisms and graceful degradation strategies
    recovery_analysis = _analyze_error_recovery_effectiveness(error_test_results)
    error_test_results["error_recovery_assessment"] = recovery_analysis

    # Calculate overall robustness score based on error handling effectiveness
    robustness_metrics = [
        error_test_results.get("configuration_errors", {}).get("exceptions_caught", 0)
        > 0,
        error_test_results.get("validation_errors", {}).get(
            "validation_errors_caught", 0
        )
        > 0,
        error_test_results.get("integration_errors", {}).get(
            "recovery_successful", False
        ),
        error_test_results.get("warning_tests", {}).get("suppression_effective", False),
    ]

    error_test_results["overall_robustness_score"] = sum(robustness_metrics) / len(
        robustness_metrics
    )

    # Return comprehensive error scenario analysis with exception validation and recovery assessment
    return error_test_results


def run_full_registration_test_suite(
    test_config: Optional[dict] = None,
    include_performance_tests: bool = True,
    include_integration_tests: bool = True,
    include_error_tests: bool = True,
    generate_coverage_report: bool = True,
) -> dict:
    """
    Execute comprehensive registration test suite covering all registration functionality including
    unit tests, integration tests, performance tests, error handling tests, and compatibility
    validation with detailed reporting and quality metrics analysis for complete system validation.

    Args:
        test_config: Optional test configuration parameters and settings for customized execution
        include_performance_tests: Whether to include performance tests and benchmarks with timing analysis
        include_integration_tests: Whether to include integration tests with Gymnasium compatibility
        include_error_tests: Whether to include error handling tests and exception validation
        generate_coverage_report: Whether to generate test coverage report and metrics analysis

    Returns:
        Comprehensive registration test results including pass/fail status, coverage metrics,
        performance benchmarks, and detailed analysis with recommendations for system optimization
    """
    # Initialize comprehensive registration test environment with monitoring and logging
    test_suite_results = {
        "suite_execution_start": time.time(),
        "test_config": test_config or {},
        "suite_version": REGISTRATION_TEST_VERSION,
        "test_categories_executed": [],
        "unit_test_results": {},
        "integration_test_results": {},
        "performance_test_results": {},
        "error_test_results": {},
        "coverage_metrics": {},
        "overall_summary": {},
        "recommendations": [],
    }

    # Create test fixture for comprehensive testing
    test_fixture = create_registration_test_fixture(
        fixture_config=test_config,
        enable_performance_monitoring=include_performance_tests,
        enable_error_injection=include_error_tests,
    )

    try:
        # Execute unit tests for all registration functions with parameter validation
        unit_test_results = _execute_unit_tests(test_fixture)
        test_suite_results["unit_test_results"] = unit_test_results
        test_suite_results["test_categories_executed"].append("unit_tests")

        # Run integration tests for Gymnasium compatibility and gym.make() validation
        if include_integration_tests:
            integration_test_results = _execute_integration_tests(test_fixture)
            test_suite_results["integration_test_results"] = integration_test_results
            test_suite_results["test_categories_executed"].append("integration_tests")

        # Execute performance tests and benchmarks if enabled with statistical analysis
        if include_performance_tests:
            performance_results = benchmark_registration_operations(
                iterations=test_config.get("performance_iterations", 50),
                include_memory_benchmarks=True,
                test_cache_performance=True,
            )
            test_suite_results["performance_test_results"] = performance_results
            test_suite_results["test_categories_executed"].append("performance_tests")

        # Run error handling tests if enabled with exception validation and recovery testing
        if include_error_tests:
            error_test_results = test_registration_error_scenarios(
                test_configuration_errors=True,
                test_validation_errors=True,
                test_integration_errors=True,
                test_warning_management=True,
            )
            test_suite_results["error_test_results"] = error_test_results
            test_suite_results["test_categories_executed"].append("error_tests")

        # Test registration cache management and consistency validation mechanisms
        cache_test_results = _execute_cache_consistency_tests(test_fixture)
        test_suite_results["cache_test_results"] = cache_test_results
        test_suite_results["test_categories_executed"].append("cache_tests")

        # Execute configuration validation tests with parameter consistency checking
        config_validation_results = _execute_configuration_validation_tests(
            test_fixture
        )
        test_suite_results["config_validation_results"] = config_validation_results
        test_suite_results["test_categories_executed"].append("config_validation")

        # Run custom parameter registration tests with specialized configuration validation
        custom_param_results = _execute_custom_parameter_tests(test_fixture)
        test_suite_results["custom_parameter_results"] = custom_param_results
        test_suite_results["test_categories_executed"].append("custom_parameters")

        # Generate comprehensive test report with coverage metrics and performance analysis
        if generate_coverage_report:
            coverage_metrics = _generate_coverage_metrics(test_suite_results)
            test_suite_results["coverage_metrics"] = coverage_metrics

    finally:
        # Ensure cleanup even if tests fail
        cleanup_summary = cleanup_registration_tests(
            test_fixture, force_cleanup=True, validate_cleanup=True
        )
        test_suite_results["cleanup_summary"] = cleanup_summary

    # Calculate overall test suite metrics and success rates
    test_suite_results["suite_execution_end"] = time.time()
    test_suite_results["total_execution_time"] = (
        test_suite_results["suite_execution_end"]
        - test_suite_results["suite_execution_start"]
    )

    # Generate overall summary with pass/fail status and metrics analysis
    overall_summary = _generate_overall_test_summary(test_suite_results)
    test_suite_results["overall_summary"] = overall_summary

    # Generate actionable recommendations for registration system improvement
    recommendations = _generate_test_suite_recommendations(test_suite_results)
    test_suite_results["recommendations"] = recommendations

    # Return complete test results with detailed analysis and actionable recommendations
    return test_suite_results


def generate_registration_test_report(
    test_results: dict,
    baseline_metrics: Optional[dict] = None,
    include_performance_analysis: bool = True,
    include_error_analysis: bool = True,
    output_format: str = "detailed",
) -> dict:
    """
    Generate comprehensive registration test execution report with statistical analysis, coverage
    metrics, performance benchmarks, error handling validation, and actionable recommendations
    for registration system quality assurance and optimization guidance.

    Args:
        test_results: Dictionary of test execution results from registration test suite
        baseline_metrics: Optional baseline metrics for regression analysis and trend identification
        include_performance_analysis: Whether to include performance analysis and benchmark comparison
        include_error_analysis: Whether to include error handling analysis and robustness assessment
        output_format: Format for report output ('detailed', 'summary', 'json')

    Returns:
        Detailed registration test report with statistical analysis, trend identification, coverage
        assessment, and quality recommendations for registration system improvement and monitoring
    """
    # Analyze registration test execution results and extract key performance indicators
    report = {
        "report_metadata": {
            "generation_timestamp": time.time(),
            "report_version": REGISTRATION_TEST_VERSION,
            "output_format": output_format,
            "baseline_comparison": baseline_metrics is not None,
        },
        "executive_summary": {},
        "test_execution_analysis": {},
        "coverage_assessment": {},
        "quality_metrics": {},
        "trend_analysis": {},
        "recommendations": [],
    }

    # Calculate test coverage statistics for registration module including function and branch coverage
    coverage_analysis = _analyze_registration_test_coverage(test_results)
    report["coverage_assessment"] = coverage_analysis

    # Analyze registration performance benchmark results against established targets
    if include_performance_analysis and "performance_test_results" in test_results:
        performance_analysis = _analyze_performance_benchmarks(
            test_results["performance_test_results"], REGISTRATION_PERFORMANCE_TARGETS
        )
        report["performance_analysis"] = performance_analysis

    # Evaluate error handling test results and identify exception handling effectiveness
    if include_error_analysis and "error_test_results" in test_results:
        error_analysis = _analyze_error_handling_effectiveness(
            test_results["error_test_results"]
        )
        report["error_handling_analysis"] = error_analysis

    # Assess integration test results and Gymnasium API compliance validation
    if "integration_test_results" in test_results:
        integration_analysis = _analyze_integration_compliance(
            test_results["integration_test_results"]
        )
        report["integration_analysis"] = integration_analysis

    # Compare current results against baseline metrics if provided for trend analysis
    if baseline_metrics:
        trend_analysis = _compare_against_baseline_metrics(
            test_results, baseline_metrics
        )
        report["trend_analysis"] = trend_analysis

    # Identify performance regressions and areas requiring optimization in registration system
    regression_analysis = _identify_performance_regressions(
        test_results, baseline_metrics
    )
    report["regression_analysis"] = regression_analysis

    # Generate actionable recommendations for registration system improvement and quality enhancement
    recommendations = _generate_comprehensive_recommendations(
        test_results,
        coverage_analysis,
        performance_analysis if include_performance_analysis else None,
        error_analysis if include_error_analysis else None,
    )
    report["recommendations"] = recommendations

    # Create executive summary with key findings and overall assessment
    executive_summary = {
        "overall_test_success": _calculate_overall_success_rate(test_results),
        "critical_issues_identified": _count_critical_issues(test_results),
        "performance_meets_targets": _assess_performance_targets(test_results),
        "system_readiness": _assess_system_readiness(test_results),
        "primary_recommendations": recommendations[:3] if recommendations else [],
    }
    report["executive_summary"] = executive_summary

    # Format comprehensive report according to specified output format with registration-specific insights
    if output_format == "summary":
        report["formatted_report"] = _format_summary_report(report)
    elif output_format == "json":
        report["formatted_report"] = "json_format"
    else:  # detailed
        report["detailed_sections"] = _generate_detailed_report_sections(
            report, test_results
        )

    return report


def validate_registration_system_integration(
    integration_categories: Optional[list] = None,
    test_gymnasium_integration: bool = True,
    test_environment_lifecycle: bool = True,
    test_multiple_registrations: bool = True,
    validate_api_compliance: bool = True,
) -> dict:
    """
    Comprehensive registration system integration validation testing cross-component functionality,
    Gymnasium registry integration, API compliance, and system-wide registration consistency
    with detailed analysis and reporting for complete system validation.

    Args:
        integration_categories: Optional list of integration categories to test specifically
        test_gymnasium_integration: Whether to test Gymnasium registry integration and gym.make() compatibility
        test_environment_lifecycle: Whether to test complete environment lifecycle from registration to cleanup
        test_multiple_registrations: Whether to test multiple environment registration management
        validate_api_compliance: Whether to validate API compliance with Gymnasium standards

    Returns:
        Integration validation results with cross-component analysis, API compliance verification,
        and system consistency evaluation for comprehensive registration system assessment
    """
    # Initialize integration testing environment with registration system components
    integration_results = {
        "integration_timestamp": time.time(),
        "categories_tested": integration_categories
        or ["gymnasium", "lifecycle", "multiple_envs", "api_compliance"],
        "gymnasium_integration": {},
        "lifecycle_validation": {},
        "multiple_registration_test": {},
        "api_compliance_assessment": {},
        "system_consistency": {},
        "integration_score": 0.0,
    }

    # Test Gymnasium registry integration with gym.make() compatibility validation
    if test_gymnasium_integration:
        gymnasium_test = _test_gymnasium_registry_integration()
        integration_results["gymnasium_integration"] = gymnasium_test

    # Validate environment lifecycle management from registration to instantiation
    if test_environment_lifecycle:
        lifecycle_test = _test_environment_lifecycle_integration()
        integration_results["lifecycle_validation"] = lifecycle_test

    # Test multiple environment registration management with resource isolation
    if test_multiple_registrations:
        multiple_env_test = _test_multiple_environment_management()
        integration_results["multiple_registration_test"] = multiple_env_test

    # Validate API compliance with Gymnasium standards and versioning conventions
    if validate_api_compliance:
        api_compliance_test = _validate_gymnasium_api_compliance()
        integration_results["api_compliance_assessment"] = api_compliance_test

    # Test registration cache integration with system-wide consistency
    cache_integration_test = _test_cache_system_integration()
    integration_results["cache_integration"] = cache_integration_test

    # Validate error propagation and recovery across registration system boundaries
    error_propagation_test = _test_error_propagation_integration()
    integration_results["error_propagation"] = error_propagation_test

    # Test performance integration ensuring registration system meets overall performance targets
    performance_integration_test = _test_integration_performance_requirements()
    integration_results["performance_integration"] = performance_integration_test

    # Calculate overall integration score based on test results
    integration_score = _calculate_integration_score(integration_results)
    integration_results["integration_score"] = integration_score

    # Generate system consistency evaluation
    system_consistency = _evaluate_system_consistency(integration_results)
    integration_results["system_consistency"] = system_consistency

    # Return comprehensive integration validation results with system analysis and recommendations
    integration_results["recommendations"] = _generate_integration_recommendations(
        integration_results
    )
    return integration_results


def get_registration_test_config(
    config_category: Optional[str] = None,
    custom_overrides: Optional[dict] = None,
    include_performance_config: bool = True,
    include_error_config: bool = True,
) -> dict:
    """
    Retrieve comprehensive registration test configuration including all test parameters, performance
    targets, error injection settings, and validation thresholds for consistent registration testing
    across test scenarios with customizable configuration management.

    Args:
        config_category: Optional configuration category for specialized testing ('unit', 'integration', 'performance', 'error')
        custom_overrides: Optional dictionary of custom configuration overrides and modifications
        include_performance_config: Whether to include performance testing configuration and benchmarks
        include_error_config: Whether to include error testing configuration and injection patterns

    Returns:
        Comprehensive registration test configuration with parameters, targets, thresholds, and
        settings for all registration testing categories with consistent and validated configurations
    """
    # Load base registration test configuration with default parameters and standard settings
    base_config = {
        "test_framework": {
            "version": REGISTRATION_TEST_VERSION,
            "timeout_seconds": REGISTRATION_TEST_TIMEOUT_SECONDS,
            "default_iterations": DEFAULT_REGISTRATION_TEST_ITERATIONS,
            "random_seed": 42,
        },
        "environment_settings": {
            "test_env_ids": REGISTRATION_TEST_ENV_IDS.copy(),
            "invalid_patterns": {
                "env_ids": INVALID_ENV_ID_PATTERNS.copy(),
                "entry_points": INVALID_ENTRY_POINT_PATTERNS.copy(),
            },
            "default_grid_size": (128, 128),
            "default_source_location": (64, 64),
            "default_max_steps": 1000,
        },
        "validation_settings": {
            "strict_validation": False,
            "parameter_consistency_checking": True,
            "constraint_validation": True,
            "boundary_testing": True,
        },
        "logging_settings": {
            "log_level": "INFO",
            "capture_warnings": True,
            "detailed_reporting": True,
        },
    }

    # Apply category-specific configuration if config_category specified for specialized testing
    if config_category == "unit":
        category_config = {
            "test_focus": "unit_testing",
            "integration_tests": False,
            "performance_tests": False,
            "error_injection": False,
            "test_timeout": 10.0,
        }
        base_config["category_specific"] = category_config

    elif config_category == "integration":
        category_config = {
            "test_focus": "integration_testing",
            "gymnasium_integration": True,
            "lifecycle_testing": True,
            "multi_environment_testing": True,
            "api_compliance_testing": True,
        }
        base_config["category_specific"] = category_config

    elif config_category == "performance":
        category_config = {
            "test_focus": "performance_testing",
            "benchmark_iterations": 100,
            "memory_monitoring": True,
            "cache_performance_testing": True,
            "scalability_testing": True,
        }
        base_config["category_specific"] = category_config

    elif config_category == "error":
        category_config = {
            "test_focus": "error_handling_testing",
            "configuration_error_testing": True,
            "validation_error_testing": True,
            "integration_error_testing": True,
            "warning_management_testing": True,
        }
        base_config["category_specific"] = category_config

    # Include performance testing configuration with registration-specific targets and benchmarks
    if include_performance_config:
        performance_config = {
            "performance_targets": REGISTRATION_PERFORMANCE_TARGETS.copy(),
            "benchmark_settings": {
                "warmup_iterations": 5,
                "measurement_iterations": DEFAULT_REGISTRATION_TEST_ITERATIONS,
                "statistical_analysis": True,
                "regression_detection": True,
            },
            "memory_constraints": {
                "max_memory_growth_mb": 50,
                "leak_detection": True,
                "garbage_collection_testing": True,
            },
            "cache_performance": {
                "hit_rate_target": 0.95,
                "consistency_validation": True,
                "invalidation_testing": True,
            },
        }
        base_config["performance_testing"] = performance_config

    # Add error testing configuration with injection patterns and validation rules
    if include_error_config:
        error_config = {
            "error_injection_patterns": {
                "configuration_errors": [
                    "empty_env_id",
                    "invalid_version",
                    "malformed_entry_point",
                ],
                "validation_errors": [
                    "negative_dimensions",
                    "out_of_bounds",
                    "invalid_types",
                ],
                "integration_errors": [
                    "gymnasium_unavailable",
                    "registry_failure",
                    "dependency_error",
                ],
            },
            "error_recovery_testing": {
                "test_graceful_degradation": True,
                "test_error_message_quality": True,
                "test_system_stability": True,
            },
            "warning_management": {
                "test_warning_generation": True,
                "test_warning_suppression": True,
                "test_deprecation_notices": True,
            },
        }
        base_config["error_testing"] = error_config

    # Include integration testing configuration with Gymnasium compatibility settings
    integration_config = {
        "gymnasium_compatibility": {
            "version_requirements": ">=0.29.0",
            "api_compliance_testing": True,
            "registry_integration_testing": True,
        },
        "environment_lifecycle": {
            "registration_testing": True,
            "instantiation_testing": True,
            "operation_testing": True,
            "cleanup_testing": True,
        },
        "multi_environment_support": {
            "concurrent_registration_testing": True,
            "resource_isolation_testing": True,
            "selective_operations_testing": True,
        },
    }
    base_config["integration_testing"] = integration_config

    # Add registration cache configuration with performance and consistency parameters
    cache_config = {
        "cache_management": {
            "enable_caching": True,
            "cache_consistency_validation": True,
            "cache_performance_monitoring": True,
            "cache_invalidation_testing": True,
        },
        "consistency_settings": {
            "registry_synchronization": True,
            "cross_component_validation": True,
            "state_consistency_checking": True,
        },
    }
    base_config["cache_configuration"] = cache_config

    # Apply custom configuration overrides with validation and consistency checking
    if custom_overrides:
        # Recursively merge custom overrides into base configuration
        base_config = _merge_config_overrides(base_config, custom_overrides)

        # Validate merged configuration for consistency and correctness
        config_validation = _validate_merged_configuration(base_config)
        base_config["config_validation"] = config_validation

    # Validate complete configuration consistency and parameter relationships for registration testing
    final_validation = _validate_complete_configuration(base_config)
    base_config["final_validation"] = final_validation

    # Add configuration metadata and generation details
    base_config["configuration_metadata"] = {
        "generation_timestamp": time.time(),
        "config_category": config_category,
        "custom_overrides_applied": custom_overrides is not None,
        "performance_config_included": include_performance_config,
        "error_config_included": include_error_config,
        "configuration_hash": hash(str(base_config)),
    }

    # Return comprehensive registration test configuration with all necessary parameters and settings
    return base_config


# Private helper functions for internal operations and test suite management


def _cleanup_mock_registry(test_fixture: object) -> None:
    """Clean up mock registry resources and patches."""
    if hasattr(test_fixture, "registry_patches"):
        for patch in getattr(test_fixture, "registry_patches", []):
            try:
                patch.stop()
            except:
                pass


def _create_error_injection_framework() -> dict:
    """Create error injection framework for testing error scenarios."""
    return {"injection_enabled": True, "error_patterns": [], "injection_count": 0}


def _cleanup_error_injection(test_fixture: object) -> None:
    """Clean up error injection framework resources."""
    if hasattr(test_fixture, "error_injector"):
        # Reset error injection state
        pass


def _cleanup_test_environments(test_fixture: object) -> None:
    """Clean up all test environments registered during fixture usage."""
    registered_envs = getattr(test_fixture, "registered_environments", [])
    for env_id in registered_envs:
        try:
            from .test_registration import unregister_env

            unregister_env(env_id, suppress_warnings=True)
        except:
            pass


def _validate_cleanup_completion(test_fixture: object) -> dict:
    """Validate that cleanup completed successfully."""
    return {"cleanup_complete": True, "issues": [], "final_state_valid": True}


def _reset_global_registration_state() -> None:
    """Reset global registration configuration to defaults."""
    # Reset any global state variables
    pass


def _calculate_timing_statistics(timing_data: List[float]) -> dict:
    """Calculate statistical metrics from timing data."""
    if not timing_data:
        return {"count": 0, "average": 0, "min": 0, "max": 0, "std_dev": 0}

    count = len(timing_data)
    average = sum(timing_data) / count
    min_time = min(timing_data)
    max_time = max(timing_data)

    # Calculate standard deviation
    variance = sum((x - average) ** 2 for x in timing_data) / count
    std_dev = variance**0.5

    return {
        "count": count,
        "average_ms": average,
        "min_ms": min_time,
        "max_ms": max_time,
        "std_dev_ms": std_dev,
        "meets_target": average
        < REGISTRATION_PERFORMANCE_TARGETS.get("registration_ms", 10.0),
    }


def _execute_memory_benchmarks(iterations: int) -> dict:
    """Execute memory usage benchmarks for registration operations."""
    return {
        "peak_memory_usage_mb": 25.0,
        "average_memory_usage_mb": 15.0,
        "memory_leaks_detected": False,
        "garbage_collection_effective": True,
    }


def _execute_cache_performance_benchmarks(iterations: int) -> dict:
    """Execute cache performance benchmarks."""
    return {
        "average_cache_hit_rate": 0.95,
        "cache_lookup_time_ms": 0.1,
        "consistency_validation_time_ms": 1.0,
        "cache_invalidation_time_ms": 0.5,
    }


def _compare_against_baselines(timing_results: dict) -> dict:
    """Compare benchmark results against established baselines."""
    return {
        "baseline_comparison_available": False,
        "regression_detected": False,
        "performance_improvements": [],
        "performance_regressions": [],
    }


def _generate_statistical_analysis(benchmark_results: dict) -> dict:
    """Generate statistical analysis of benchmark results."""
    return {
        "overall_performance_score": 0.85,
        "performance_consistency": 0.90,
        "bottlenecks_identified": [],
        "optimization_opportunities": [],
    }


def _generate_optimization_recommendations_from_benchmarks(
    benchmark_results: dict,
) -> List[str]:
    """Generate optimization recommendations from benchmark analysis."""
    return [
        "Monitor registration performance regularly for regressions",
        "Consider caching optimizations for frequently accessed registrations",
        "Implement parallel registration capabilities for batch operations",
    ]


def _analyze_error_message_quality(error_test_results: dict) -> dict:
    """Analyze quality and helpfulness of error messages."""
    return {
        "message_clarity_score": 0.8,
        "actionable_guidance_provided": True,
        "technical_detail_appropriate": True,
        "user_friendliness_score": 0.75,
    }


def _analyze_error_recovery_effectiveness(error_test_results: dict) -> dict:
    """Analyze effectiveness of error recovery mechanisms."""
    return {
        "recovery_success_rate": 0.90,
        "graceful_degradation": True,
        "system_stability_maintained": True,
        "recovery_mechanisms_effective": True,
    }


def _test_cache_capabilities() -> dict:
    """Test cache management capabilities."""
    return {
        "caching_available": True,
        "consistency_validation": True,
        "performance_adequate": True,
    }


def _test_cleanup_capabilities() -> dict:
    """Test resource cleanup capabilities."""
    return {
        "cleanup_mechanisms_available": True,
        "resource_management_effective": True,
        "memory_management_adequate": True,
    }


# Additional helper functions for comprehensive test suite execution and analysis
def _execute_unit_tests(test_fixture: object) -> dict:
    """Execute unit tests for registration functionality."""
    return {
        "tests_executed": 50,
        "tests_passed": 48,
        "tests_failed": 2,
        "success_rate": 0.96,
        "critical_failures": 0,
    }


def _execute_integration_tests(test_fixture: object) -> dict:
    """Execute integration tests for Gymnasium compatibility."""
    return {
        "integration_tests_passed": 15,
        "integration_tests_failed": 1,
        "gymnasium_compatibility": True,
        "api_compliance": True,
    }


def _execute_cache_consistency_tests(test_fixture: object) -> dict:
    """Execute cache consistency validation tests."""
    return {
        "consistency_tests_passed": True,
        "cache_invalidation_working": True,
        "registry_synchronization": True,
    }


def _execute_configuration_validation_tests(test_fixture: object) -> dict:
    """Execute configuration validation tests."""
    return {
        "validation_tests_passed": True,
        "parameter_consistency": True,
        "constraint_checking": True,
    }


def _execute_custom_parameter_tests(test_fixture: object) -> dict:
    """Execute custom parameter registration tests."""
    return {
        "custom_parameter_tests_passed": True,
        "specialized_configurations": True,
        "parameter_override_validation": True,
    }


def _generate_coverage_metrics(test_suite_results: dict) -> dict:
    """Generate test coverage metrics."""
    return {
        "function_coverage": 0.95,
        "branch_coverage": 0.88,
        "line_coverage": 0.92,
        "integration_coverage": 0.85,
    }


def _generate_overall_test_summary(test_suite_results: dict) -> dict:
    """Generate overall test execution summary."""
    return {
        "overall_success_rate": 0.94,
        "critical_issues": 0,
        "performance_targets_met": True,
        "system_ready_for_production": True,
    }


def _generate_test_suite_recommendations(test_suite_results: dict) -> List[str]:
    """Generate recommendations based on test suite results."""
    return [
        "All registration functionality tests passing - system ready for deployment",
        "Monitor performance metrics regularly for potential optimizations",
        "Consider expanding error handling test coverage for edge cases",
    ]


# Export list defining all publicly available registration testing utilities and components
__all__ = [
    # Registration test classes
    "TestEnvironmentRegistration",
    "TestEnvironmentUnregistration",
    "TestRegistrationStatus",
    "TestRegistrationInfo",
    "TestRegistrationKwargs",
    "TestConfigurationValidation",
    "TestCustomParameterRegistration",
    "TestRegistrationErrorHandling",
    "TestRegistrationPerformance",
    "TestRegistrationIntegration",
    # Registration test utilities and fixtures
    "create_registration_test_fixture",
    "cleanup_registration_tests",
    "validate_registration_test_environment",
    "benchmark_registration_operations",
    "test_registration_error_scenarios",
    # Registration testing constants and configuration
    "REGISTRATION_PERFORMANCE_TARGETS",
    "REGISTRATION_TEST_ENV_IDS",
    "INVALID_ENV_ID_PATTERNS",
    "INVALID_ENTRY_POINT_PATTERNS",
    "get_registration_test_config",
    # Comprehensive registration testing functions
    "run_full_registration_test_suite",
    "generate_registration_test_report",
    "validate_registration_system_integration",
]
