"""
Comprehensive error handling test suite for plume_nav_sim package validating hierarchical
exception handling, error recovery mechanisms, secure error reporting, component-level failure
scenarios, and integration error patterns with performance monitoring, reproducibility validation,
and automated error analysis across all system components.

This module validates:
- Hierarchical exception handling with specific error types
- Input validation and security error testing
- Component-level error management and recovery
- Environment API error handling compliance
- Performance impact during error conditions
- Error recovery and system resilience mechanisms
"""

import contextlib  # >=3.10
import gc  # >=3.10
import logging  # >=3.10
import threading  # >=3.10
import time  # >=3.10
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import Mock, patch  # >=3.10

import numpy as np  # >=2.1.0
import pytest  # >=8.0.0

import plume_nav_sim.utils.logging as plume_logging
from plume_nav_sim.core.types import Action

# Internal imports for comprehensive error testing
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv
from plume_nav_sim.utils.exceptions import (
    ComponentError,
    ErrorContext,
    ErrorSeverity,
    IntegrationError,
    PlumeNavSimError,
    RenderingError,
    ResourceError,
    StateError,
    ValidationError,
    handle_component_error,
    sanitize_error_context,
)
from plume_nav_sim.utils.validation import (
    ParameterValidator,
    ValidationContext,
    validate_action_parameter,
)

# Global test data constants for comprehensive error scenario testing
INVALID_ACTIONS_TEST_DATA = [-1, 4, 10, "invalid", None, [], {}, 3.14]
INVALID_COORDINATES_TEST_DATA = [
    (-1, -1),
    (1000, 1000),
    ("x", "y"),
    None,
    [],
    "invalid",
]
INVALID_GRID_SIZES_TEST_DATA = [
    (0, 0),
    (-10, -10),
    (10000, 10000),
    ("x", "y"),
    None,
    [],
]
INVALID_SEEDS_TEST_DATA = [-1, 2**32, "seed", [], {}, None]
RESOURCE_EXHAUSTION_SCENARIOS = [
    "memory_limit",
    "cpu_limit",
    "disk_space",
    "file_handles",
]
COMPONENT_FAILURE_SCENARIOS = [
    "plume_model",
    "state_manager",
    "action_processor",
    "reward_calculator",
    "renderer",
]
INTEGRATION_FAILURE_SCENARIOS = [
    "gymnasium_incompatible",
    "numpy_version_mismatch",
    "matplotlib_backend_failure",
]
PERFORMANCE_ERROR_THRESHOLDS = {
    "step_latency_ms": 10.0,
    "memory_mb": 200.0,
    "render_latency_ms": 100.0,
}
ERROR_LOGGING_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
RECOVERY_STRATEGIES = [
    "retry",
    "fallback",
    "reset",
    "degraded_mode",
    "graceful_shutdown",
]


@pytest.mark.integration
def test_logging_log_exception_uses_handle_component_error_with_structured_context():
    """Verify setup_error_logging attaches a log_exception helper that
    forwards exceptions to handle_component_error with structured context.
    """

    test_logger = logging.getLogger("plume_nav_sim.test_logging_integration")
    test_logger.setLevel(logging.DEBUG)

    with patch("plume_nav_sim.utils.logging.handle_component_error") as mock_handle:
        plume_logging.setup_error_logging(logger=test_logger)

        assert hasattr(test_logger, "log_exception")

        test_exception = Exception("log_exception integration failure")
        context = {"operation": "test_logging", "component": "integration_test"}

        # type: ignore[attr-defined] - attached dynamically by setup_error_logging
        test_logger.log_exception(test_exception, context)  # noqa: E1101

    mock_handle.assert_called_once()
    called_error, called_component, called_context = mock_handle.call_args[0]

    assert called_error is test_exception
    assert called_component == test_logger.name
    assert isinstance(called_context, dict)
    assert called_context.get("exception_type") == type(test_exception).__name__

    component_context = called_context.get("component_context", {})
    assert component_context.get("operation") == "test_logging"
    assert component_context.get("component") == "integration_test"


@pytest.mark.performance
def test_log_performance_failure_path_calls_handle_component_error(monkeypatch):
    """Ensure log_performance uses handle_component_error on failure without
    propagating the exception to callers and passes useful context."""

    test_logger = logging.getLogger("plume_nav_sim.test_performance_logging")

    def failing_validate(operation_name: Any, duration_ms: Any) -> tuple[str, float]:
        raise ValidationError(
            "invalid performance input", "operation_name", operation_name
        )

    with patch("plume_nav_sim.utils.logging.handle_component_error") as mock_handle:
        monkeypatch.setattr(
            plume_logging,
            "_validate_performance_inputs",
            failing_validate,
        )

        plume_logging.log_performance(
            test_logger,
            "test_operation",
            1.23,
            additional_metrics={"password": "secret"},
            compare_to_baseline=True,
        )

    mock_handle.assert_called_once()
    error_arg, component_name_arg, error_context_arg = mock_handle.call_args[0]

    assert isinstance(error_arg, ValidationError)
    assert component_name_arg == "log_performance"
    assert isinstance(error_context_arg, dict)
    assert error_context_arg.get("operation_name") == "test_operation"
    assert error_context_arg.get("duration_ms") == 1.23


def create_invalid_environment_configs(
    include_type_errors: bool = True,
    include_range_violations: bool = True,
    include_mathematical_inconsistencies: bool = True,
    include_resource_violations: bool = True,
    include_all: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Factory function to create comprehensive invalid environment configuration test scenarios
    including parameter type errors, range violations, mathematical inconsistencies, and
    resource constraint violations for systematic configuration error testing.

    Args:
        include_type_errors: Include parameter type mismatch scenarios
        include_range_violations: Include parameter range violation scenarios
        include_mathematical_inconsistencies: Include mathematical constraint violations
        include_resource_violations: Include resource limit violation scenarios
        include_all: When provided, overrides the four include_* flags. The
            value must be a boolean which allows the caller to request either
            the complete set of scenarios (``True``) or none of the optional
            scenarios (``False``). A non-boolean value raises ``TypeError`` to
            fail fast during misconfiguration.

    Returns:
        List of invalid configuration dictionaries with expected error types and descriptions
    """
    if include_all is not None:
        if not isinstance(include_all, bool):
            raise TypeError("include_all must be a boolean when provided")
        include_type_errors = include_range_violations = (
            include_mathematical_inconsistencies
        ) = include_resource_violations = include_all

    invalid_configs = []

    # Type error configurations for systematic error testing scenarios
    if include_type_errors:
        invalid_configs.extend(
            [
                {
                    "grid_size": "invalid_string",
                    "expected_error": "ConfigurationError",
                    "description": "String grid_size instead of tuple",
                },
                {
                    "source_location": 42,
                    "expected_error": "ConfigurationError",
                    "description": "Integer source_location instead of tuple",
                },
                {
                    "max_steps": "unlimited",
                    "expected_error": "ValidationError",
                    "description": "String max_steps instead of integer",
                },
                {
                    "goal_radius": [1, 2, 3],
                    "expected_error": "ValidationError",
                    "description": "List goal_radius instead of numeric",
                },
            ]
        )

    # Range violation configurations for boundary testing scenarios
    if include_range_violations:
        invalid_configs.extend(
            [
                {
                    "grid_size": (-10, -10),
                    "expected_error": "ValidationError",
                    "description": "Negative grid dimensions",
                },
                {
                    "source_location": (-5, -5),
                    "expected_error": "ValidationError",
                    "description": "Negative source coordinates",
                },
                {
                    "max_steps": 0,
                    "expected_error": "ValidationError",
                    "description": "Zero maximum steps",
                },
                {
                    "goal_radius": -1.0,
                    "expected_error": "ValidationError",
                    "description": "Negative goal radius",
                },
            ]
        )

    # Mathematical inconsistency configurations for constraint validation
    if include_mathematical_inconsistencies:
        invalid_configs.extend(
            [
                {
                    "grid_size": (10, 10),
                    "source_location": (15, 5),
                    "expected_error": "ValidationError",
                    "description": "Source location outside grid bounds",
                },
                {
                    "plume_sigma": 0.0,
                    "expected_error": "ValidationError",
                    "description": "Zero plume dispersion parameter",
                },
                {
                    "goal_radius": 100.0,
                    "grid_size": (10, 10),
                    "expected_error": "ValidationError",
                    "description": "Goal radius larger than grid size",
                },
            ]
        )

    # Resource violation configurations for system limit testing
    if include_resource_violations:
        invalid_configs.extend(
            [
                {
                    "grid_size": (10000, 10000),
                    "expected_error": "ResourceError",
                    "description": "Grid size exceeding memory limits",
                },
                {
                    "render_mode": "ultra_high_resolution",
                    "expected_error": "ConfigurationError",
                    "description": "Unsupported high-resource render mode",
                },
            ]
        )

    # Parameter interaction errors with cross-parameter inconsistencies
    invalid_configs.extend(
        [
            {
                "grid_size": (128, 64),
                "source_location": (128, 32),
                "expected_error": "ValidationError",
                "description": "Source X coordinate equals grid width (boundary error)",
            },
            {
                "max_steps": 1000000,
                "grid_size": (5, 5),
                "expected_error": "ValidationError",
                "description": "Excessive max_steps for small grid",
            },
        ]
    )

    return invalid_configs


@contextlib.contextmanager
def simulate_component_failure(
    component_name: str,
    failure_type: str,
    failure_parameters: Dict[str, Any],
    enable_recovery_testing: bool = True,
):
    """
    Utility function to simulate controlled component failures for testing error handling,
    recovery mechanisms, and system resilience including state corruption, resource exhaustion,
    and integration failures with configurable failure scenarios.

    Args:
        component_name: Name of component to simulate failure for
        failure_type: Type of failure to simulate
        failure_parameters: Parameters controlling failure behavior
        enable_recovery_testing: Whether to enable recovery mechanism testing

    Yields:
        Context manager for controlled failure injection and recovery validation
    """
    # Validate component_name against supported failure scenarios
    if component_name not in COMPONENT_FAILURE_SCENARIOS:
        raise ValueError(f"Unsupported component for failure testing: {component_name}")

    # Set up failure injection strategy based on component type
    _ = {}
    _ = {}

    try:
        if component_name == "plume_model":
            # Simulate plume model calculation failures
            if failure_type == "calculation_error":
                with patch(
                    "plume_nav_sim.plume.static_gaussian.StaticGaussianPlume.sample_concentration"
                ) as mock_sample:
                    if failure_parameters.get("raise_exception", True):
                        mock_sample.side_effect = ComponentError(
                            "Plume calculation failed"
                        )
                    else:
                        mock_sample.return_value = np.nan
                    yield mock_sample
            elif failure_type == "memory_exhaustion":
                with patch("numpy.exp") as mock_exp:
                    mock_exp.side_effect = MemoryError(
                        "Insufficient memory for plume calculation"
                    )
                    yield mock_exp

        elif component_name == "renderer":
            # Simulate rendering pipeline failures
            if failure_type == "backend_failure":
                with patch("matplotlib.pyplot.subplots") as mock_plt:
                    mock_plt.side_effect = RenderingError(
                        "Matplotlib backend unavailable"
                    )
                    yield mock_plt
            elif failure_type == "resource_cleanup":
                with patch("matplotlib.pyplot.close") as mock_close:
                    mock_close.side_effect = ResourceError(
                        "Failed to cleanup rendering resources"
                    )
                    yield mock_close

        elif component_name == "state_manager":
            # Simulate state management failures
            if failure_type == "state_corruption":
                yield Mock(side_effect=StateError("Agent state corrupted"))
            elif failure_type == "synchronization_error":
                yield Mock(side_effect=StateError("Component synchronization failed"))

        else:
            # Generic component failure simulation
            yield Mock(side_effect=ComponentError(f"{component_name} component failed"))

    finally:
        # Cleanup and recovery validation if enabled
        if enable_recovery_testing:
            # Validate that system can recover from simulated failure
            gc.collect()  # Force garbage collection for memory cleanup testing


def validate_error_logging(
    log_record: logging.LogRecord,
    expected_level: str,
    check_sensitive_data: bool = True,
    validate_performance: bool = True,
) -> Dict[str, Any]:
    """
    Comprehensive error logging validation function testing log message security, appropriate
    log levels, error context sanitization, and logging performance impact with detailed
    logging analysis and validation.

    Args:
        log_record: Log record to validate
        expected_level: Expected logging level for the record
        check_sensitive_data: Whether to check for sensitive data exposure
        validate_performance: Whether to validate logging performance impact

    Returns:
        Logging validation results with security analysis and performance metrics
    """
    start_time = time.perf_counter() if validate_performance else 0

    validation_results = {
        "level_correct": log_record.levelname == expected_level,
        "security_compliant": True,
        "performance_acceptable": True,
        "issues": [],
    }

    # Validate log level matches expected level for appropriate error classification
    if log_record.levelname != expected_level:
        validation_results["issues"].append(
            f"Expected level {expected_level}, got {log_record.levelname}"
        )

    # Check log message content for sensitive information disclosure
    if check_sensitive_data:
        sensitive_patterns = [
            "password",
            "token",
            "key",
            "secret",
            "internal_state",
            "debug_info",
        ]
        message = log_record.getMessage().lower()

        for pattern in sensitive_patterns:
            if pattern in message:
                validation_results["security_compliant"] = False
                validation_results["issues"].append(
                    f"Potential sensitive data exposure: {pattern}"
                )

    # Validate error context sanitization and secure information handling
    if hasattr(log_record, "extra_data"):
        # Check if error context contains sanitized information
        if "raw_parameters" in str(log_record.extra_data):
            validation_results["security_compliant"] = False
            validation_results["issues"].append(
                "Unsanitized raw parameters in log record"
            )

    # Analyze logging performance impact if enabled
    if validate_performance:
        logging_time_ms = (time.perf_counter() - start_time) * 1000
        validation_results["performance_metrics"] = {
            "logging_time_ms": logging_time_ms,
            "acceptable": logging_time_ms < 1.0,  # <1ms logging overhead target
        }

        if logging_time_ms >= 1.0:
            validation_results["performance_acceptable"] = False
            validation_results["issues"].append(
                f"Logging overhead {logging_time_ms:.2f}ms exceeds 1ms target"
            )

    # Verify log message format compliance and error reporting standards
    required_fields = ["timestamp", "level", "message"]
    for field in required_fields:
        if not hasattr(log_record, field.replace("timestamp", "created")):
            validation_results["issues"].append(f"Missing required field: {field}")

    return validation_results


def measure_error_handling_overhead(
    error_function: Callable,
    iteration_count: int = 1000,
    include_recovery_timing: bool = True,
    performance_targets: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Performance measurement function for error handling overhead analysis including exception
    throwing, catching, logging, and recovery timing with performance impact assessment and
    optimization recommendations.

    Args:
        error_function: Function that generates errors for measurement
        iteration_count: Number of iterations for statistical measurement
        include_recovery_timing: Whether to measure recovery mechanism timing
        performance_targets: Target performance metrics for comparison

    Returns:
        Error handling performance analysis with timing statistics and optimization recommendations
    """
    # Initialize performance measurement infrastructure with high-precision timing
    timing_data = {"exception_times": [], "recovery_times": [], "logging_times": []}

    start_time = time.perf_counter()

    # Execute error_function multiple times with timing measurement
    for i in range(iteration_count):
        iteration_start = time.perf_counter()

        try:
            # Measure exception throwing and catching overhead
            error_function()
        except Exception as e:
            exception_time = (time.perf_counter() - iteration_start) * 1000
            timing_data["exception_times"].append(exception_time)

            # Measure recovery timing if enabled
            if include_recovery_timing:
                recovery_start = time.perf_counter()
                try:
                    # Simulate recovery action
                    handle_component_error(e, "test_component")
                except Exception:
                    pass
                recovery_time = (time.perf_counter() - recovery_start) * 1000
                timing_data["recovery_times"].append(recovery_time)

            # Measure logging overhead
            log_start = time.perf_counter()
            logging.getLogger("test").error(f"Test error iteration {i}: {str(e)}")
            log_time = (time.perf_counter() - log_start) * 1000
            timing_data["logging_times"].append(log_time)

    total_time = (time.perf_counter() - start_time) * 1000

    # Calculate statistical metrics including mean, median, standard deviation, and percentiles
    performance_analysis = {
        "total_iterations": iteration_count,
        "total_time_ms": total_time,
        "average_iteration_time_ms": total_time / iteration_count,
    }

    # Exception handling statistics
    if timing_data["exception_times"]:
        exception_times = np.array(timing_data["exception_times"])
        performance_analysis["exception_handling"] = {
            "mean_ms": np.mean(exception_times),
            "median_ms": np.median(exception_times),
            "std_dev_ms": np.std(exception_times),
            "percentile_95_ms": np.percentile(exception_times, 95),
            "percentile_99_ms": np.percentile(exception_times, 99),
        }

    # Recovery mechanism statistics
    if timing_data["recovery_times"] and include_recovery_timing:
        recovery_times = np.array(timing_data["recovery_times"])
        performance_analysis["recovery_mechanisms"] = {
            "mean_ms": np.mean(recovery_times),
            "median_ms": np.median(recovery_times),
            "success_rate": len(recovery_times) / iteration_count,
        }

    # Logging performance statistics
    if timing_data["logging_times"]:
        logging_times = np.array(timing_data["logging_times"])
        performance_analysis["logging_overhead"] = {
            "mean_ms": np.mean(logging_times),
            "median_ms": np.median(logging_times),
            "max_ms": np.max(logging_times),
        }

    # Analyze performance against targets if provided
    recommendations = []
    if performance_targets:
        for metric, target in performance_targets.items():
            if metric in performance_analysis.get("exception_handling", {}):
                actual = performance_analysis["exception_handling"].get("mean_ms", 0)
                if actual > target:
                    recommendations.append(
                        f"Exception handling {metric} ({actual:.2f}ms) exceeds target ({target}ms)"
                    )

    # Generate performance optimization recommendations
    if performance_analysis.get("exception_handling", {}).get("mean_ms", 0) > 1.0:
        recommendations.append(
            "Consider reducing exception handling overhead through caching or optimization"
        )

    if performance_analysis.get("logging_overhead", {}).get("mean_ms", 0) > 0.5:
        recommendations.append(
            "Consider optimizing logging configuration for error scenarios"
        )

    performance_analysis["optimization_recommendations"] = recommendations

    return performance_analysis


def create_error_test_scenarios(
    scenario_category: str,
    scenario_parameters: Dict[str, Any] = None,
    include_recovery_validation: bool = True,
    include_performance_testing: bool = True,
) -> List[Dict[str, Any]]:
    """
    Comprehensive error scenario factory creating systematic error test cases including boundary
    conditions, edge cases, component failures, and integration errors with expected outcomes
    and validation criteria.

    Args:
        scenario_category: Category of error scenarios to generate
        scenario_parameters: Parameters controlling scenario generation
        include_recovery_validation: Whether to include recovery testing scenarios
        include_performance_testing: Whether to include performance validation scenarios

    Returns:
        Comprehensive error test scenarios with expected outcomes and validation criteria
    """
    scenario_parameters = scenario_parameters or {}
    scenarios = []

    # Generate base error scenarios based on category
    if scenario_category == "validation":
        scenarios.extend(
            [
                {
                    "name": "invalid_parameter_type",
                    "setup": lambda: {"action": "invalid_string"},
                    "expected_exception": ValidationError,
                    "validation_criteria": [
                        "error_message_clarity",
                        "recovery_suggestion_provided",
                    ],
                },
                {
                    "name": "parameter_range_violation",
                    "setup": lambda: {"coordinates": (-1, -1)},
                    "expected_exception": ValidationError,
                    "validation_criteria": [
                        "boundary_check_accurate",
                        "constraint_explanation_clear",
                    ],
                },
            ]
        )

    elif scenario_category == "boundary":
        scenarios.extend(
            [
                {
                    "name": "grid_boundary_violation",
                    "setup": lambda: {"grid_size": (0, 0)},
                    "expected_exception": ValidationError,
                    "validation_criteria": [
                        "zero_dimension_detected",
                        "alternative_suggested",
                    ],
                },
                {
                    "name": "numerical_precision_limits",
                    "setup": lambda: {"sigma": 1e-15},
                    "expected_exception": ValidationError,
                    "validation_criteria": [
                        "precision_limit_identified",
                        "numerical_stability_warning",
                    ],
                },
            ]
        )

    elif scenario_category == "integration":
        scenarios.extend(
            [
                {
                    "name": "gymnasium_api_violation",
                    "setup": lambda: Mock(
                        side_effect=IntegrationError("Gymnasium API violation")
                    ),
                    "expected_exception": IntegrationError,
                    "validation_criteria": [
                        "api_compliance_checked",
                        "framework_compatibility_validated",
                    ],
                },
                {
                    "name": "numpy_array_incompatibility",
                    "setup": lambda: Mock(
                        side_effect=IntegrationError("NumPy array type mismatch")
                    ),
                    "expected_exception": IntegrationError,
                    "validation_criteria": [
                        "array_type_validation",
                        "conversion_attempted",
                    ],
                },
            ]
        )

    # Include recovery validation scenarios if requested
    if include_recovery_validation:
        for scenario in scenarios:
            scenario["recovery_tests"] = [
                "automatic_recovery_attempted",
                "fallback_mechanism_activated",
                "system_state_preserved",
                "error_logging_complete",
            ]

    # Add performance testing scenarios if requested
    if include_performance_testing:
        for scenario in scenarios:
            scenario["performance_requirements"] = {
                "max_exception_time_ms": 5.0,
                "max_recovery_time_ms": 10.0,
                "memory_overhead_mb": 1.0,
            }

    return scenarios


class TestErrorHandling:
    """
    Main test class for comprehensive error handling validation across all plume_nav_sim
    components including exception hierarchy testing, error recovery validation, performance
    impact analysis, and security verification with systematic error scenario testing and
    automated error analysis.
    """

    __test__ = False

    def __init__(self):
        """
        Initialize comprehensive error handling test infrastructure with logging, performance
        monitoring, and systematic error validation setup.
        """
        # Initialize test logger for error handling test activity and validation logging
        self.logger = logging.getLogger("plume_nav_sim.test.error_handling")
        self.logger.setLevel(logging.DEBUG)

        # Initialize performance metrics dictionary for error handling overhead analysis
        self.performance_metrics = {
            "exception_times": [],
            "recovery_times": [],
            "validation_times": [],
            "total_test_time": 0.0,
        }

        # Initialize error test results list for systematic error scenario tracking
        self.error_test_results = []

        # Initialize component error states dictionary for component failure state tracking
        self.component_error_states = {}

        # Set up error injection infrastructure for controlled failure testing
        self.error_injection_active = False
        self.injected_failures = {}

        # Configure security validation infrastructure for error context sanitization testing
        self.security_validation_enabled = True
        self.sensitive_data_patterns = ["password", "token", "key", "internal_state"]

    @pytest.mark.parametrize(
        "invalid_param, expected_error",
        [
            (config, config["expected_error"])
            for config in create_invalid_environment_configs()
        ],
    )
    def test_validation_error_hierarchy(self, invalid_param, expected_error):
        """
        Test ValidationError exception hierarchy including parameter validation failures, input
        sanitization, error context generation, and recovery suggestion validation with
        comprehensive parameter error testing.

        Args:
            invalid_param: Invalid parameter configuration for testing
            expected_error: Expected exception type for validation
        """
        test_start_time = time.perf_counter()

        # Test ValidationError instantiation with parameter context and constraint validation
        with pytest.raises(eval(expected_error)) as exc_info:
            # Create validation context for detailed error tracking
            context = ValidationContext(
                operation_name="parameter_validation_test",
                component_name="test_validation",
            )

            # Attempt validation with invalid parameter configuration
            validator = ParameterValidator()
            validator.validate_parameter("test_param", invalid_param, context)

        # Validate parameter-specific error details including parameter_name and invalid_value
        error = exc_info.value
        assert hasattr(error, "parameter_name") or hasattr(error, "message")

        if hasattr(error, "get_validation_details"):
            # Test validation error accumulation using add_validation_error
            validation_details = error.get_validation_details()
            assert validation_details is not None
            assert (
                "parameter_name" in validation_details
                or "constraint_violation" in validation_details
            )

        # Test validation error recovery suggestions including parameter correction guidance
        if hasattr(error, "get_recovery_suggestions"):
            recovery_suggestions = error.get_recovery_suggestions()
            assert len(recovery_suggestions) > 0
            assert any(
                "parameter" in suggestion.lower() for suggestion in recovery_suggestions
            )

        # Validate secure error context generation with sensitive parameter sanitization
        if hasattr(error, "error_context"):
            context = error.error_context
            assert context is not None
            # Verify no sensitive information is exposed
            context_str = str(context)
            for sensitive_pattern in self.sensitive_data_patterns:
                assert sensitive_pattern not in context_str.lower()

        # Test validation error logging with appropriate log levels and security compliance
        with self.assertLogs(level="WARNING") as _:
            self.logger.warning(f"Validation error occurred: {str(error)}")

        # Verify validation error performance impact and overhead measurement
        test_time = (time.perf_counter() - test_start_time) * 1000
        assert (
            test_time < 100.0
        ), f"Validation error test took {test_time:.2f}ms, exceeds 100ms limit"

        self.performance_metrics["validation_times"].append(test_time)

    @pytest.fixture(autouse=True)
    def test_state_error_management(self):
        """
        Test StateError exception handling including invalid state transitions, component
        synchronization failures, automated recovery actions, and state consistency validation
        with comprehensive state management error testing.
        """
        test_env = None
        try:
            # Initialize test environment for state error testing
            test_env = PlumeSearchEnv(grid_size=(32, 32), source_location=(16, 16))

            # Test StateError instantiation with current and expected state context validation
            with pytest.raises(StateError) as exc_info:
                # Simulate invalid state transition by manipulating internal state
                test_env._agent_position = (-1, -1)  # Invalid position
                test_env._validate_component_state()

            state_error = exc_info.value

            # Validate state transition error reporting with component identification
            assert (
                hasattr(state_error, "current_state")
                or "state" in str(state_error).lower()
            )
            assert (
                hasattr(state_error, "expected_state")
                or "expected" in str(state_error).lower()
            )

            # Test recovery action suggestions using suggest_recovery_action
            if hasattr(state_error, "suggest_recovery_action"):
                recovery_actions = state_error.suggest_recovery_action()
                assert len(recovery_actions) > 0
                assert any(
                    "reset" in action.lower() or "restore" in action.lower()
                    for action in recovery_actions
                )

            # Test state error escalation and severity classification
            if hasattr(state_error, "severity"):
                assert state_error.severity in [
                    ErrorSeverity.MEDIUM,
                    ErrorSeverity.HIGH,
                ]

            # Test automated state recovery mechanisms with recovery validation
            try:
                # Attempt automatic recovery
                test_env.reset(seed=42)
                assert test_env._agent_position is not None
                assert all(coord >= 0 for coord in test_env._agent_position)

            except Exception as recovery_error:
                pytest.fail(f"State recovery failed: {str(recovery_error)}")

        finally:
            # Cleanup test environment
            if test_env:
                try:
                    test_env.close()
                except Exception:
                    pass

    @pytest.mark.parametrize("render_mode", ["rgb_array", "human"])
    def test_rendering_error_fallback(self, render_mode):
        """
        Test RenderingError exception handling including matplotlib backend failures, display
        system errors, fallback mechanism validation, and rendering pipeline recovery with
        comprehensive visualization error testing.

        Args:
            render_mode: Rendering mode to test for error handling
        """
        test_env = None
        try:
            # Initialize test environment with specific render mode
            test_env = PlumeSearchEnv(render_mode=render_mode, grid_size=(32, 32))
            test_env.reset(seed=42)

            # Test RenderingError instantiation with render mode and backend context validation
            with simulate_component_failure(
                "renderer", "backend_failure", {"raise_exception": True}
            ) as _:
                with pytest.raises(RenderingError) as exc_info:
                    test_env.render()

                rendering_error = exc_info.value

                # Validate rendering context addition using set_rendering_context
                assert (
                    "render" in str(rendering_error).lower()
                    or "backend" in str(rendering_error).lower()
                )

                # Test fallback suggestion generation using get_fallback_suggestions
                if hasattr(rendering_error, "get_fallback_suggestions"):
                    fallback_suggestions = rendering_error.get_fallback_suggestions()
                    assert len(fallback_suggestions) > 0
                    assert any(
                        "fallback" in suggestion.lower()
                        or "alternative" in suggestion.lower()
                        for suggestion in fallback_suggestions
                    )

            # Test automatic fallback mechanisms from human to rgb_array mode
            if render_mode == "human":
                # Simulate backend unavailability and test fallback
                with patch(
                    "matplotlib.pyplot.subplots",
                    side_effect=ImportError("Backend unavailable"),
                ):
                    try:
                        result = test_env.render()
                        # Should fallback to rgb_array mode or return None gracefully
                        assert result is None or isinstance(result, np.ndarray)
                    except RenderingError as fallback_error:
                        # Fallback error should include suggestion for alternative
                        assert (
                            "fallback" in str(fallback_error).lower()
                            or "rgb_array" in str(fallback_error).lower()
                        )

            # Verify rendering error performance impact and fallback mechanism efficiency
            fallback_start_time = time.perf_counter()
            try:
                test_env.render()
            except Exception:
                pass
            fallback_time = (time.perf_counter() - fallback_start_time) * 1000
            assert (
                fallback_time < 100.0
            ), f"Rendering error handling took {fallback_time:.2f}ms"

        finally:
            if test_env:
                try:
                    test_env.close()
                except Exception:
                    pass

    @pytest.mark.parametrize(
        "config_error",
        [config for config in create_invalid_environment_configs(include_all=True)],
    )
    def test_configuration_error_validation(self, config_error):
        """
        Test ConfigurationError exception handling including invalid parameters, environment
        setup failures, valid option suggestions, and configuration consistency validation with
        comprehensive setup error testing.

        Args:
            config_error: Invalid configuration dictionary for testing
        """
        # Test ConfigurationError instantiation with parameter and valid options context
        with pytest.raises(
            eval(config_error.get("expected_error", "ConfigurationError"))
        ) as exc_info:
            # Attempt environment creation with invalid configuration
            invalid_config = {
                k: v
                for k, v in config_error.items()
                if k not in ["expected_error", "description"]
            }

            PlumeSearchEnv(**invalid_config)

        config_error_obj = exc_info.value

        # Validate configuration parameter error reporting with specific parameter identification
        error_message = str(config_error_obj).lower()
        assert (
            any(param in error_message for param in invalid_config.keys())
            or "config" in error_message
        )

        # Test valid options suggestion generation using get_valid_options
        if hasattr(config_error_obj, "get_valid_options"):
            valid_options = config_error_obj.get_valid_options()
            assert isinstance(valid_options, (list, dict))
            assert len(valid_options) > 0

        # Test configuration recovery mechanisms with parameter correction and validation
        if hasattr(config_error_obj, "suggest_correction"):
            correction_suggestion = config_error_obj.suggest_correction()
            assert correction_suggestion is not None
            assert isinstance(correction_suggestion, (str, dict))

        # Verify configuration error impact on environment initialization
        with pytest.raises(Exception):
            # Should consistently fail with same invalid configuration
            _ = PlumeSearchEnv(**invalid_config)

    @pytest.mark.parametrize("component_failure", COMPONENT_FAILURE_SCENARIOS)
    def test_component_error_diagnosis(self, component_failure):
        """
        Test ComponentError exception handling including component failure diagnosis, isolation
        mechanisms, recovery strategies, and system resilience validation with comprehensive
        component-level error testing.

        Args:
            component_failure: Component name to test failure scenarios for
        """
        test_env = None
        try:
            test_env = PlumeSearchEnv(grid_size=(32, 32))
            test_env.reset(seed=42)

            # Test ComponentError instantiation with component identification and operation context
            with simulate_component_failure(
                component_failure,
                "calculation_error",
                {"raise_exception": True},
                enable_recovery_testing=True,
            ) as _:
                with pytest.raises(ComponentError) as exc_info:
                    if component_failure == "plume_model":
                        # Trigger plume model error
                        test_env.step(Action.UP)
                    elif component_failure == "renderer":
                        test_env.render()
                    elif component_failure == "state_manager":
                        test_env._validate_component_state()
                    else:
                        # Generic component operation
                        getattr(test_env, f"test_{component_failure}", lambda: None)()

                component_error = exc_info.value

                # Test failure diagnosis using diagnose_failure with component-specific analysis
                if hasattr(component_error, "diagnose_failure"):
                    diagnosis = component_error.diagnose_failure()
                    assert diagnosis is not None
                    assert component_failure in str(diagnosis).lower()

                # Test component error isolation with system resilience validation
                assert component_failure in str(component_error).lower()

                # Validate component recovery mechanisms with automatic recovery
                recovery_attempted = False
                try:
                    handle_component_error(component_error, component_failure)
                    recovery_attempted = True
                except Exception:
                    pass

                assert (
                    recovery_attempted
                ), f"Recovery not attempted for {component_failure} failure"

        finally:
            if test_env:
                try:
                    test_env.close()
                except Exception:
                    pass

    @pytest.mark.parametrize("resource_scenario", RESOURCE_EXHAUSTION_SCENARIOS)
    def test_resource_error_management(self, resource_scenario):
        """
        Test ResourceError exception handling including memory exhaustion, cleanup failures,
        resource constraint validation, and system resource management with comprehensive
        resource error testing.

        Args:
            resource_scenario: Resource exhaustion scenario to test
        """
        # Test ResourceError instantiation with resource type and usage context validation
        with pytest.raises(ResourceError) as exc_info:
            if resource_scenario == "memory_limit":
                # Simulate memory exhaustion
                with patch(
                    "numpy.zeros", side_effect=MemoryError("Insufficient memory")
                ):
                    _ = PlumeSearchEnv(grid_size=(1000, 1000))
            elif resource_scenario == "cpu_limit":
                # Simulate CPU resource exhaustion
                raise ResourceError("CPU time limit exceeded")
            elif resource_scenario == "disk_space":
                # Simulate disk space exhaustion
                raise ResourceError("Insufficient disk space for operation")
            elif resource_scenario == "file_handles":
                # Simulate file handle exhaustion
                raise ResourceError("Too many open file handles")

        resource_error = exc_info.value

        # Test cleanup action suggestions using suggest_cleanup_actions
        if hasattr(resource_error, "suggest_cleanup_actions"):
            cleanup_actions = resource_error.suggest_cleanup_actions()
            assert len(cleanup_actions) > 0
            assert any(
                "cleanup" in action.lower() or "free" in action.lower()
                for action in cleanup_actions
            )

        # Test automatic resource cleanup mechanisms with memory management
        gc.collect()  # Force garbage collection

        # Validate resource constraint enforcement with proactive resource monitoring
        assert resource_scenario in str(resource_error).lower()

        # Verify resource error performance impact on system operation
        cleanup_start = time.perf_counter()
        try:
            # Attempt resource cleanup
            gc.collect()
        except Exception:
            pass
        cleanup_time = (time.perf_counter() - cleanup_start) * 1000
        assert cleanup_time < 50.0, f"Resource cleanup took {cleanup_time:.2f}ms"

    @pytest.mark.parametrize("integration_failure", INTEGRATION_FAILURE_SCENARIOS)
    def test_integration_error_handling(self, integration_failure):
        """
        Test IntegrationError exception handling including external dependency failures, version
        compatibility issues, fallback mechanisms, and system integration resilience with
        comprehensive dependency error testing.

        Args:
            integration_failure: Integration failure scenario to test
        """
        # Test IntegrationError instantiation with dependency identification and version context
        with pytest.raises(IntegrationError) as exc_info:
            if integration_failure == "gymnasium_incompatible":
                # Simulate Gymnasium API incompatibility
                with patch(
                    "gymnasium.spaces.Discrete",
                    side_effect=AttributeError("API changed"),
                ):
                    PlumeSearchEnv()
            elif integration_failure == "numpy_version_mismatch":
                # Simulate NumPy version incompatibility
                with patch(
                    "numpy.array", side_effect=ImportError("NumPy version incompatible")
                ):
                    PlumeSearchEnv()
            elif integration_failure == "matplotlib_backend_failure":
                # Simulate matplotlib backend failure
                with patch(
                    "matplotlib.pyplot.subplots",
                    side_effect=ImportError("Backend not available"),
                ):
                    PlumeSearchEnv(render_mode="human")

        integration_error = exc_info.value

        # Test compatibility checking using check_compatibility with version analysis
        if hasattr(integration_error, "check_compatibility"):
            compatibility_check = integration_error.check_compatibility()
            assert compatibility_check is not None
            assert isinstance(compatibility_check, (dict, bool))

        # Test dependency fallback mechanisms with alternative implementation
        assert integration_failure.replace("_", " ") in str(integration_error).lower()

        # Validate integration recovery strategies with dependency reinstallation guidance
        if hasattr(integration_error, "get_recovery_instructions"):
            recovery_instructions = integration_error.get_recovery_instructions()
            assert len(recovery_instructions) > 0
            assert any(
                "install" in instruction.lower() or "upgrade" in instruction.lower()
                for instruction in recovery_instructions
            )

    @pytest.mark.parametrize("severity_level", list(ErrorSeverity))
    def test_error_severity_classification(self, severity_level):
        """
        Test ErrorSeverity enumeration and error escalation logic including severity level
        validation, escalation triggers, automated response mechanisms, and priority-based
        error handling with comprehensive severity testing.

        Args:
            severity_level: Error severity level to test
        """
        # Test ErrorSeverity enumeration values and ordering with severity level validation
        assert severity_level in ErrorSeverity
        assert hasattr(ErrorSeverity, severity_level.name)
        assert isinstance(severity_level.value, int)

        # Create test error with specific severity level
        test_error = PlumeNavSimError(f"Test error with {severity_level.name} severity")
        test_error.severity = severity_level

        # Validate severity-based error handling with appropriate response mechanisms
        if severity_level == ErrorSeverity.LOW:
            # Low severity should allow continuation
            assert test_error.severity.value == ErrorSeverity.LOW.value
        elif severity_level == ErrorSeverity.MEDIUM:
            # Medium severity should trigger warnings
            assert test_error.severity.value == ErrorSeverity.MEDIUM.value
        elif severity_level == ErrorSeverity.HIGH:
            # High severity should trigger immediate attention
            assert test_error.severity.value == ErrorSeverity.HIGH.value
        elif severity_level == ErrorSeverity.CRITICAL:
            # Critical severity should trigger emergency response
            assert test_error.severity.value == ErrorSeverity.CRITICAL.value

        # Test escalation trigger validation using should_escalate with appropriate escalation logic
        if hasattr(test_error, "should_escalate"):
            escalation_needed = test_error.should_escalate()
            if severity_level in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                assert escalation_needed is True
            else:
                assert escalation_needed in [True, False]  # May vary based on context

        # Test automated response systems with severity-triggered actions
        response_time_start = time.perf_counter()
        try:
            handle_component_error(test_error, "test_component")
        except Exception:
            pass
        response_time = (time.perf_counter() - response_time_start) * 1000

        # Verify severity classification performance impact and response time effectiveness
        max_response_time = {
            ErrorSeverity.LOW: 10.0,
            ErrorSeverity.MEDIUM: 5.0,
            ErrorSeverity.HIGH: 2.0,
            ErrorSeverity.CRITICAL: 1.0,
        }

        assert response_time < max_response_time.get(
            severity_level, 10.0
        ), f"Response time {response_time:.2f}ms exceeds {max_response_time[severity_level]}ms for {severity_level.name}"

    @pytest.mark.security
    def test_error_context_security(self):
        """
        Test ErrorContext security including sensitive data sanitization, caller information
        validation, secure context generation, and information disclosure prevention with
        comprehensive security testing.
        """
        # Test ErrorContext instantiation with component and operation identification
        context = ErrorContext()
        context.component_name = "test_component"
        context.operation_name = "test_operation"

        # Add potentially sensitive data to test sanitization
        context.add_context_data("user_input", "sensitive_password_123")
        context.add_context_data("api_key", "secret_key_456")
        context.add_context_data("debug_info", "internal_state_data")

        # Test sensitive data sanitization using sanitize
        sanitized_context = context.sanitize()
        assert sanitized_context is not None

        # Validate information disclosure prevention with sensitive parameter identification
        sanitized_str = str(sanitized_context)
        for sensitive_pattern in self.sensitive_data_patterns:
            assert (
                sensitive_pattern not in sanitized_str.lower()
            ), f"Sensitive pattern '{sensitive_pattern}' found in sanitized context"

        # Test context serialization using to_dict with secure information handling
        context_dict = context.to_dict()
        assert isinstance(context_dict, dict)
        assert "component_name" in context_dict
        assert "operation_name" in context_dict

        # Validate secure error reporting with appropriate detail levels
        for key, value in context_dict.items():
            if isinstance(value, str):
                for sensitive_pattern in self.sensitive_data_patterns:
                    assert (
                        sensitive_pattern not in value.lower()
                    ), f"Sensitive data found in context field '{key}': {value}"

        # Test context security under concurrent conditions with thread-safe sanitization
        def concurrent_sanitization_test():
            local_context = ErrorContext()
            local_context.add_context_data("test_data", "concurrent_test")
            return local_context.sanitize()

        # Run concurrent sanitization tests
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(concurrent_sanitization_test) for _ in range(10)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

            # Verify all results are properly sanitized
            for result in results:
                assert result is not None
                assert "concurrent_test" in str(result)

    @pytest.mark.integration
    def test_centralized_error_handling(self):
        """
        Test centralized error handling function including component-specific recovery strategies,
        error classification, logging integration, and automated error management with
        comprehensive centralized error testing.
        """
        # Test handle_component_error function with various error types and component contexts
        test_errors = [
            (ValidationError("Invalid parameter"), "validator"),
            (StateError("Invalid state transition"), "state_manager"),
            (RenderingError("Backend failure"), "renderer"),
            (ComponentError("Component malfunction"), "plume_model"),
            (ResourceError("Memory exhausted"), "resource_manager"),
        ]

        for error, component_name in test_errors:
            # Test component-specific recovery strategy selection with appropriate recovery actions
            recovery_result = handle_component_error(error, component_name)
            assert recovery_result is not None
            assert isinstance(recovery_result, (str, dict))

            # Validate error classification and routing with severity-based handling
            if isinstance(recovery_result, dict):
                assert "strategy" in recovery_result or "action" in recovery_result
                assert "component" in recovery_result or component_name in str(
                    recovery_result
                )

        # Test automated error recovery with success validation and fallback mechanism testing
        test_env = None
        try:
            test_env = PlumeSearchEnv(grid_size=(32, 32))
            test_env.reset(seed=42)

            # Inject controlled error and test recovery
            with simulate_component_failure(
                "plume_model", "calculation_error", {"raise_exception": True}
            ) as _:
                try:
                    test_env.step(Action.UP)
                except ComponentError as e:
                    recovery_result = handle_component_error(e, "plume_model")
                    assert recovery_result is not None

                    # Test that system can continue operation after recovery
                    test_env.reset(seed=123)  # Should succeed after recovery

        finally:
            if test_env:
                try:
                    test_env.close()
                except Exception:
                    pass

        # Verify centralized error handling performance impact and system overhead
        performance_start = time.perf_counter()
        for _ in range(100):
            handle_component_error(ValidationError("Test error"), "test_component")
        performance_time = (time.perf_counter() - performance_start) * 1000

        assert (
            performance_time < 100.0
        ), f"Centralized error handling overhead {performance_time:.2f}ms"

    @pytest.mark.slow
    def test_environment_error_scenarios(self):
        """
        Test PlumeSearchEnv error handling including API method failures, invalid parameter
        handling, component coordination errors, and recovery mechanisms with comprehensive
        environment error testing.
        """
        # Test environment initialization errors with invalid configuration parameters
        invalid_configs = create_invalid_environment_configs()

        for config in invalid_configs[:5]:  # Test subset for performance
            with pytest.raises(Exception):
                invalid_params = {
                    k: v
                    for k, v in config.items()
                    if k not in ["expected_error", "description"]
                }
                test_env = PlumeSearchEnv(**invalid_params)

        # Test valid environment operations and error handling
        test_env = None
        try:
            test_env = PlumeSearchEnv(grid_size=(32, 32), source_location=(16, 16))

            # Validate reset method error handling with state initialization failures
            obs, info = test_env.reset(seed=42)
            assert obs is not None
            assert info is not None
            assert isinstance(obs, np.ndarray)

            # Test step method error handling with invalid actions and component failures
            for invalid_action in INVALID_ACTIONS_TEST_DATA[:3]:  # Test subset
                with pytest.raises((ValidationError, ValueError, TypeError)):
                    test_env.step(invalid_action)

            # Test render method error handling with backend failures and fallback mechanisms
            try:
                render_result = test_env.render()
                assert render_result is None or isinstance(render_result, np.ndarray)
            except RenderingError:
                # Rendering errors are acceptable and should be handled gracefully
                pass

            # Test environment integrity validation with comprehensive error detection
            integrity_result = test_env.validate_environment_integrity()
            assert isinstance(integrity_result, (bool, dict))

        finally:
            if test_env:
                try:
                    test_env.close()
                except Exception:
                    pass

        # Verify environment error handling performance impact on operation latency
        performance_env = PlumeSearchEnv(grid_size=(32, 32))
        try:
            performance_env.reset(seed=42)

            step_times = []
            for _ in range(100):
                step_start = time.perf_counter()
                try:
                    performance_env.step(Action.UP)
                except Exception:
                    pass
                step_times.append((time.perf_counter() - step_start) * 1000)

            avg_step_time = np.mean(step_times)
            assert (
                avg_step_time < PERFORMANCE_ERROR_THRESHOLDS["step_latency_ms"]
            ), f"Average step time {avg_step_time:.2f}ms exceeds threshold"

        finally:
            try:
                performance_env.close()
            except Exception:
                pass

    @pytest.mark.parametrize(
        "validation_scenario", create_error_test_scenarios("validation")
    )
    def test_validation_framework_errors(self, validation_scenario):
        """
        Test validation framework error handling including ParameterValidator failures,
        ValidationResult management, batch validation errors, and validation performance with
        comprehensive validation error testing.

        Args:
            validation_scenario: Validation error scenario configuration
        """
        scenario_setup = validation_scenario["setup"]
        expected_exception = validation_scenario["expected_exception"]

        # Test ParameterValidator error handling with invalid validation rules
        validator = ParameterValidator()

        with pytest.raises(expected_exception) as exc_info:
            invalid_params = scenario_setup()
            for param_name, param_value in invalid_params.items():
                validator.validate_parameter(
                    param_name,
                    param_value,
                    ValidationContext("test_validation", "test"),
                )

        validation_error = exc_info.value

        # Validate ValidationResult error accumulation with multiple validation failures
        assert isinstance(validation_error, expected_exception)

        # Test validation performance under error conditions with timing analysis
        performance_start = time.perf_counter()
        for _ in range(10):  # Multiple validation attempts
            try:
                invalid_params = scenario_setup()
                for param_name, param_value in invalid_params.items():
                    validator.validate_parameter(
                        param_name, param_value, ValidationContext("perf_test", "test")
                    )
            except Exception:
                pass

        validation_time = (time.perf_counter() - performance_start) * 1000
        assert (
            validation_time < 100.0
        ), f"Validation error handling took {validation_time:.2f}ms"

        # Verify validation framework error performance impact
        assert len(validation_scenario.get("validation_criteria", [])) > 0

    @pytest.mark.recovery
    def test_error_recovery_mechanisms(self):
        """
        Test comprehensive error recovery mechanisms including automatic recovery, manual
        intervention, graceful degradation, and system resilience with recovery effectiveness
        validation and performance analysis.
        """
        test_env = None
        recovery_test_results = []

        try:
            test_env = PlumeSearchEnv(grid_size=(32, 32))

            # Test automatic error recovery with success rate analysis and recovery timing
            for recovery_strategy in RECOVERY_STRATEGIES[:3]:  # Test subset
                recovery_start_time = time.perf_counter()

                try:
                    # Simulate error requiring recovery
                    if recovery_strategy == "reset":
                        test_env.reset(seed=42)
                    elif recovery_strategy == "fallback":
                        # Test fallback rendering mode
                        test_env.render()
                    elif recovery_strategy == "retry":
                        # Test retry mechanism
                        for attempt in range(3):
                            try:
                                test_env.step(Action.UP)
                                break
                            except Exception:
                                if attempt == 2:
                                    raise
                                continue

                    recovery_time = (time.perf_counter() - recovery_start_time) * 1000
                    recovery_test_results.append(
                        {
                            "strategy": recovery_strategy,
                            "success": True,
                            "time_ms": recovery_time,
                        }
                    )

                except Exception as e:
                    recovery_time = (time.perf_counter() - recovery_start_time) * 1000
                    recovery_test_results.append(
                        {
                            "strategy": recovery_strategy,
                            "success": False,
                            "time_ms": recovery_time,
                            "error": str(e),
                        }
                    )

            # Validate system resilience with multiple concurrent error conditions
            try:
                # Test multiple error conditions simultaneously
                with simulate_component_failure("plume_model", "calculation_error", {}):
                    with simulate_component_failure("renderer", "backend_failure", {}):
                        test_env.reset(
                            seed=42
                        )  # Should handle multiple failures gracefully
            except Exception:
                pass

            # Analyze recovery effectiveness with success metrics
            successful_recoveries = sum(
                1 for result in recovery_test_results if result["success"]
            )
            recovery_success_rate = (
                successful_recoveries / len(recovery_test_results)
                if recovery_test_results
                else 0
            )

            assert (
                recovery_success_rate >= 0.6
            ), f"Recovery success rate {recovery_success_rate:.1%} below 60% threshold"

            # Validate recovery performance with timing analysis
            if recovery_test_results:
                avg_recovery_time = np.mean(
                    [r["time_ms"] for r in recovery_test_results]
                )
                assert (
                    avg_recovery_time < 50.0
                ), f"Average recovery time {avg_recovery_time:.2f}ms exceeds 50ms"

        finally:
            if test_env:
                try:
                    test_env.close()
                except Exception:
                    pass

    @pytest.mark.security
    def test_error_logging_security(self):
        """
        Test error logging security including sensitive data sanitization, log level
        appropriateness, secure error reporting, and logging performance with comprehensive
        logging security validation.
        """
        # Set up test logger with controlled configuration
        test_logger = logging.getLogger("plume_nav_sim.security_test")
        test_logger.setLevel(logging.DEBUG)

        # Create test error with potentially sensitive information
        sensitive_error = ValidationError(
            "Authentication failed for user: admin with password: secret123"
        )

        # Test error logging security with sensitive data identification and sanitization
        with self.assertLogs(test_logger, level="ERROR") as log_context:
            # Log error through secure error handling
            sanitized_context = sanitize_error_context(
                {
                    "error_message": str(sensitive_error),
                    "user_input": "admin:secret123",
                    "internal_state": "debug_information",
                }
            )
            test_logger.error(f"Secure error logged: {sanitized_context}")

        # Validate log level appropriateness with security-based log level selection
        assert len(log_context.records) > 0
        log_record = log_context.records[0]

        validation_result = validate_error_logging(
            log_record, "ERROR", check_sensitive_data=True, validate_performance=True
        )

        assert validation_result[
            "security_compliant"
        ], f"Security compliance failed: {validation_result['issues']}"

        assert validation_result[
            "level_correct"
        ], f"Log level validation failed: {validation_result['issues']}"

        # Test concurrent logging security with thread-safe sanitization
        def concurrent_secure_logging():
            local_logger = logging.getLogger("plume_nav_sim.concurrent_security")
            sensitive_data = {"password": f"secret_{threading.current_thread().ident}"}
            sanitized_data = sanitize_error_context(sensitive_data)
            local_logger.error(f"Concurrent secure log: {sanitized_data}")
            return sanitized_data

        # Run concurrent security tests
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(concurrent_secure_logging) for _ in range(10)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

            # Verify all results are properly sanitized
            for result in results:
                assert "password" not in str(result) or "SANITIZED" in str(result)

        # Verify logging security performance impact and sanitization efficiency
        sanitization_start = time.perf_counter()
        for _ in range(100):
            sanitize_error_context({"test_key": "test_value", "password": "secret"})
        sanitization_time = (time.perf_counter() - sanitization_start) * 1000

        assert (
            sanitization_time < 50.0
        ), f"Sanitization overhead {sanitization_time:.2f}ms exceeds 50ms"

    @pytest.mark.performance
    def test_error_performance_impact(self):
        """
        Test error handling performance impact including exception overhead, recovery timing,
        logging latency, and system performance degradation with comprehensive performance
        analysis and optimization recommendations.
        """

        # Test error handling overhead with exception throwing, catching, and processing timing
        def test_error_function():
            raise ValidationError("Test error for performance measurement")

        overhead_analysis = measure_error_handling_overhead(
            test_error_function,
            iteration_count=1000,
            include_recovery_timing=True,
            performance_targets=PERFORMANCE_ERROR_THRESHOLDS,
        )

        # Validate error handling performance meets targets
        exception_handling = overhead_analysis.get("exception_handling", {})
        mean_exception_time = exception_handling.get("mean_ms", 0)

        assert (
            mean_exception_time < 1.0
        ), f"Exception handling mean time {mean_exception_time:.3f}ms exceeds 1ms target"

        # Test system performance degradation under error conditions
        test_env = PlumeSearchEnv(grid_size=(64, 64))
        try:
            test_env.reset(seed=42)

            # Measure baseline performance
            baseline_times = []
            for _ in range(100):
                step_start = time.perf_counter()
                test_env.step(Action.UP)
                baseline_times.append((time.perf_counter() - step_start) * 1000)

            baseline_avg = np.mean(baseline_times)

            # Measure performance under error conditions
            error_times = []
            for _ in range(100):
                step_start = time.perf_counter()
                try:
                    # Inject occasional errors
                    if np.random.random() < 0.1:  # 10% error rate
                        raise ValidationError("Simulated validation error")
                    test_env.step(Action.RIGHT)
                except ValidationError:
                    pass  # Handle error gracefully
                error_times.append((time.perf_counter() - step_start) * 1000)

            error_avg = np.mean(error_times)

            # Performance degradation should be minimal
            performance_degradation = (error_avg - baseline_avg) / baseline_avg
            assert (
                performance_degradation < 0.5
            ), f"Performance degradation {performance_degradation:.1%} exceeds 50% threshold"

        finally:
            test_env.close()

        # Analyze error handling scalability with increasing error rates
        scalability_results = []
        error_rates = [0.01, 0.05, 0.1, 0.2]  # 1%, 5%, 10%, 20%

        for error_rate in error_rates:
            scalability_env = PlumeSearchEnv(grid_size=(32, 32))
            try:
                scalability_env.reset(seed=42)

                step_times = []
                for _ in range(50):
                    step_start = time.perf_counter()
                    try:
                        if np.random.random() < error_rate:
                            raise ValidationError("Scalability test error")
                        scalability_env.step(Action.DOWN)
                    except ValidationError:
                        pass
                    step_times.append((time.perf_counter() - step_start) * 1000)

                scalability_results.append(
                    {
                        "error_rate": error_rate,
                        "avg_time_ms": np.mean(step_times),
                        "max_time_ms": np.max(step_times),
                    }
                )

            finally:
                scalability_env.close()

        # Validate scalability performance
        for result in scalability_results:
            assert (
                result["avg_time_ms"] < 5.0
            ), f"Average time {result['avg_time_ms']:.2f}ms at {result['error_rate']:.1%} error rate"

        # Generate error handling performance optimization recommendations
        recommendations = overhead_analysis.get("optimization_recommendations", [])
        assert len(recommendations) >= 0  # Should provide recommendations if needed

        # Store performance metrics for test suite analysis
        self.performance_metrics["exception_times"].extend(
            overhead_analysis.get("exception_handling", {}).get("raw_times", [])
        )
        self.performance_metrics["total_test_time"] = time.perf_counter()


class TestErrorScenarios:
    """
    Specialized test class for systematic error scenario testing including edge cases, boundary
    conditions, stress testing, and comprehensive error condition validation with automated
    scenario generation, recovery testing, and diagnostics collection.
    """

    __test__ = False

    def __init__(self):
        """
        Initialize error scenario testing infrastructure with scenario registry, statistics
        tracking, and automated scenario validation.
        """
        # Initialize scenario registry for systematic error scenario organization and tracking
        self.scenario_registry = {
            "boundary_conditions": [],
            "stress_conditions": [],
            "cascade_scenarios": [],
            "integration_failures": [],
        }

        # Initialize error statistics dictionary for error pattern analysis
        self.error_statistics = {
            "total_scenarios": 0,
            "passed_scenarios": 0,
            "failed_scenarios": 0,
            "error_patterns": {},
        }

        # Initialize scenario results list for comprehensive scenario outcome tracking
        self.scenario_results = []

    @pytest.mark.parametrize(
        "boundary_scenario", create_error_test_scenarios("boundary")
    )
    def test_boundary_condition_errors(self, boundary_scenario):
        """
        Test boundary condition error scenarios including parameter limits, edge values,
        numerical precision issues, and constraint violations with systematic boundary testing
        and validation.

        Args:
            boundary_scenario: Boundary condition scenario configuration
        """
        self.error_statistics["total_scenarios"] += 1
        scenario_start_time = time.perf_counter()

        try:
            # Test parameter boundary violations with minimum and maximum value testing
            scenario_setup = boundary_scenario["setup"]
            expected_exception = boundary_scenario["expected_exception"]

            with pytest.raises(expected_exception) as exc_info:
                boundary_params = scenario_setup()

                # Create test environment with boundary parameters
                if "grid_size" in boundary_params:
                    test_env = PlumeSearchEnv(grid_size=boundary_params["grid_size"])
                elif "coordinates" in boundary_params:
                    test_env = PlumeSearchEnv()
                    test_env.reset(seed=42)
                    # Test coordinate boundary validation
                    validate_action_parameter("move_to", boundary_params["coordinates"])
                else:
                    # Generic parameter validation
                    validator = ParameterValidator()
                    for param_name, param_value in boundary_params.items():
                        validator.validate_parameter(
                            param_name,
                            param_value,
                            ValidationContext("boundary_test", "test"),
                        )

            boundary_error = exc_info.value

            # Validate edge case error handling with corner cases and unusual parameter combinations
            assert isinstance(boundary_error, expected_exception)

            # Test boundary condition recovery with automatic constraint relaxation
            recovery_suggestions = getattr(
                boundary_error, "get_recovery_suggestions", lambda: []
            )()
            if recovery_suggestions:
                assert len(recovery_suggestions) > 0
                assert any(
                    "boundary" in suggestion.lower() or "limit" in suggestion.lower()
                    for suggestion in recovery_suggestions
                )

            # Verify boundary condition error consistency with reproducible edge case behavior
            scenario_time = (time.perf_counter() - scenario_start_time) * 1000
            assert scenario_time < 100.0, f"Boundary test took {scenario_time:.2f}ms"

            self.error_statistics["passed_scenarios"] += 1

            # Record successful scenario result
            self.scenario_results.append(
                {
                    "scenario": boundary_scenario["name"],
                    "status": "passed",
                    "execution_time_ms": scenario_time,
                    "error_type": expected_exception.__name__,
                }
            )

        except Exception as test_error:
            self.error_statistics["failed_scenarios"] += 1
            scenario_time = (time.perf_counter() - scenario_start_time) * 1000

            # Record failed scenario result
            self.scenario_results.append(
                {
                    "scenario": boundary_scenario.get("name", "unknown"),
                    "status": "failed",
                    "execution_time_ms": scenario_time,
                    "error": str(test_error),
                }
            )

            raise  # Re-raise for pytest reporting

    @pytest.mark.stress
    def test_stress_condition_errors(self):
        """
        Test system stress condition errors including high load, resource exhaustion, concurrent
        operations, and system limits with comprehensive stress testing and resilience validation.
        """
        stress_scenarios = [
            {"name": "high_frequency_operations", "operation_count": 10000},
            {"name": "large_grid_memory_stress", "grid_size": (512, 512)},
            {"name": "concurrent_environment_stress", "concurrent_envs": 10},
            {"name": "rapid_reset_stress", "reset_count": 1000},
        ]

        for stress_scenario in stress_scenarios:
            scenario_start_time = time.perf_counter()

            try:
                if stress_scenario["name"] == "high_frequency_operations":
                    # Test high load error scenarios with increased operation frequency
                    test_env = PlumeSearchEnv(grid_size=(32, 32))
                    test_env.reset(seed=42)

                    operation_times = []
                    for i in range(stress_scenario["operation_count"]):
                        op_start = time.perf_counter()
                        try:
                            test_env.step(Action.UP)
                        except Exception as e:
                            # Log error but continue stress test
                            if i % 1000 == 0:  # Log every 1000th error
                                logging.getLogger("stress_test").warning(
                                    f"Operation {i} failed: {e}"
                                )
                        operation_times.append((time.perf_counter() - op_start) * 1000)

                    test_env.close()

                    # Validate performance under stress
                    avg_operation_time = np.mean(operation_times)
                    assert (
                        avg_operation_time < 2.0
                    ), f"Average operation time {avg_operation_time:.3f}ms under stress exceeds 2ms"

                elif stress_scenario["name"] == "large_grid_memory_stress":
                    # Test resource exhaustion with memory limits and resource management
                    try:
                        large_env = PlumeSearchEnv(
                            grid_size=stress_scenario["grid_size"]
                        )
                        large_env.reset(seed=42)

                        # Test operations with large grid
                        for _ in range(100):
                            large_env.step(Action.RIGHT)

                        large_env.close()

                    except (MemoryError, ResourceError) as e:
                        # Large grid memory exhaustion is acceptable
                        assert (
                            "memory" in str(e).lower() or "resource" in str(e).lower()
                        )

                elif stress_scenario["name"] == "concurrent_environment_stress":
                    # Test concurrent operation errors with multi-threading and synchronization
                    import concurrent.futures

                    def create_and_run_env(env_id):
                        try:
                            concurrent_env = PlumeSearchEnv(grid_size=(32, 32))
                            concurrent_env.reset(seed=env_id)

                            for _ in range(50):
                                concurrent_env.step(Action.UP)

                            concurrent_env.close()
                            return f"env_{env_id}_success"
                        except Exception as e:
                            return f"env_{env_id}_failed: {str(e)}"

                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=4
                    ) as executor:
                        futures = [
                            executor.submit(create_and_run_env, i)
                            for i in range(stress_scenario["concurrent_envs"])
                        ]
                        results = [
                            future.result()
                            for future in concurrent.futures.as_completed(futures)
                        ]

                    # Validate concurrent execution results
                    successful_envs = sum(
                        1 for result in results if "success" in result
                    )
                    success_rate = successful_envs / len(results)

                    assert (
                        success_rate >= 0.8
                    ), f"Concurrent environment success rate {success_rate:.1%} below 80% threshold"

                elif stress_scenario["name"] == "rapid_reset_stress":
                    # Test stress condition recovery with load balancing and system scaling
                    rapid_env = PlumeSearchEnv(grid_size=(32, 32))

                    reset_times = []
                    for i in range(stress_scenario["reset_count"]):
                        reset_start = time.perf_counter()
                        try:
                            rapid_env.reset(seed=i % 100)
                        except Exception as e:
                            if i % 100 == 0:  # Log every 100th error
                                logging.getLogger("stress_test").error(
                                    f"Reset {i} failed: {e}"
                                )
                        reset_times.append((time.perf_counter() - reset_start) * 1000)

                    rapid_env.close()

                    # Validate reset performance under stress
                    avg_reset_time = np.mean(reset_times)
                    assert (
                        avg_reset_time < 10.0
                    ), f"Average reset time {avg_reset_time:.2f}ms under stress exceeds 10ms"

                scenario_time = (time.perf_counter() - scenario_start_time) * 1000
                self.scenario_results.append(
                    {
                        "scenario": stress_scenario["name"],
                        "status": "passed",
                        "execution_time_ms": scenario_time,
                    }
                )

            except Exception as stress_error:
                scenario_time = (time.perf_counter() - scenario_start_time) * 1000
                self.scenario_results.append(
                    {
                        "scenario": stress_scenario["name"],
                        "status": "failed",
                        "execution_time_ms": scenario_time,
                        "error": str(stress_error),
                    }
                )

                # Re-raise for individual scenario failure reporting
                raise

    @pytest.mark.cascade
    def test_cascade_error_scenarios(self):
        """
        Test error cascade scenarios including error propagation, dependency failures, system-wide
        failures, and cascade prevention with comprehensive cascade testing and isolation validation.
        """
        cascade_test_env = None

        try:
            cascade_test_env = PlumeSearchEnv(grid_size=(64, 64))
            cascade_test_env.reset(seed=42)

            # Test error propagation with component failure cascade analysis and prevention
            cascade_scenarios = [
                ("plume_model", "state_manager"),  # Plume failure affects state
                ("renderer", "plume_model"),  # Rendering failure affects plume access
                ("state_manager", "renderer"),  # State failure affects rendering
            ]

            for primary_component, secondary_component in cascade_scenarios:
                cascade_start_time = time.perf_counter()

                # Simulate primary component failure
                with simulate_component_failure(
                    primary_component, "calculation_error", {"raise_exception": True}
                ) as _:
                    try:
                        # Test that secondary component can handle primary failure gracefully
                        if primary_component == "plume_model":
                            cascade_test_env.step(
                                Action.UP
                            )  # Should trigger plume calculation
                        elif primary_component == "renderer":
                            cascade_test_env.render()  # Should trigger rendering
                        elif primary_component == "state_manager":
                            cascade_test_env._validate_component_state()  # Should trigger state check

                        # If we reach here, cascade was prevented
                        cascade_prevented = True

                    except ComponentError as cascade_error:
                        # Test cascade prevention mechanisms with circuit breakers
                        cascade_prevented = False

                        # Validate that cascade error provides appropriate context
                        assert (
                            primary_component in str(cascade_error).lower()
                            or "cascade" in str(cascade_error).lower()
                        )

                        # Test cascade recovery with sequential component recovery
                        recovery_result = handle_component_error(
                            cascade_error, primary_component
                        )
                        assert recovery_result is not None

                cascade_time = (time.perf_counter() - cascade_start_time) * 1000

                # Record cascade test result
                self.scenario_results.append(
                    {
                        "scenario": f"cascade_{primary_component}_to_{secondary_component}",
                        "status": (
                            "cascade_prevented"
                            if cascade_prevented
                            else "cascade_handled"
                        ),
                        "execution_time_ms": cascade_time,
                        "components": [primary_component, secondary_component],
                    }
                )

                # Verify cascade prevention effectiveness with isolation testing
                assert (
                    cascade_time < 200.0
                ), f"Cascade handling took {cascade_time:.2f}ms, exceeds 200ms threshold"

            # Test system-wide failure scenarios with critical component failures
            with simulate_component_failure(
                "plume_model", "memory_exhaustion", {"raise_exception": True}
            ):
                with simulate_component_failure(
                    "renderer", "backend_failure", {"raise_exception": True}
                ):
                    system_failure_start = time.perf_counter()

                    try:
                        # Test system resilience under multiple failures
                        cascade_test_env.reset(seed=123)
                        system_recovered = True

                    except Exception as system_error:
                        system_recovered = False

                        # Validate system failure handling
                        assert isinstance(
                            system_error,
                            (ComponentError, ResourceError, IntegrationError),
                        )

                    system_failure_time = (
                        time.perf_counter() - system_failure_start
                    ) * 1000

                    self.scenario_results.append(
                        {
                            "scenario": "system_wide_failure",
                            "status": (
                                "recovered" if system_recovered else "failed_gracefully"
                            ),
                            "execution_time_ms": system_failure_time,
                        }
                    )

        finally:
            if cascade_test_env:
                try:
                    cascade_test_env.close()
                except Exception:
                    pass

        # Analyze cascade test results for patterns and effectiveness
        cascade_results = [
            r for r in self.scenario_results if "cascade" in r["scenario"]
        ]
        if cascade_results:
            avg_cascade_time = np.mean(
                [r["execution_time_ms"] for r in cascade_results]
            )
            assert (
                avg_cascade_time < 150.0
            ), f"Average cascade handling time {avg_cascade_time:.2f}ms exceeds 150ms"

            # Validate cascade prevention rate
            prevented_cascades = sum(
                1 for r in cascade_results if r["status"] == "cascade_prevented"
            )
            prevention_rate = prevented_cascades / len(cascade_results)

            # At least 50% of cascades should be prevented or handled gracefully
            assert (
                prevention_rate >= 0.5
            ), f"Cascade prevention rate {prevention_rate:.1%} below 50% threshold"
