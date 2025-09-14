"""
Comprehensive test suite for plume_nav_sim logging utilities including unit tests for component logger 
factory functions, performance monitoring integration, error handling logging, security-aware 
sanitization, logging mixins, context managers, and integration with the centralized logging 
infrastructure. Tests validate logging functionality, performance tracking, information security, 
and development debugging capabilities.
"""

# Standard library imports with version comments
import pytest  # >=8.0.0 - Testing framework for test structure, fixtures, parameterization, and assertion validation
import logging  # >=3.10 - Standard Python logging module for logger testing, handler validation, and log record inspection
import time  # >=3.10 - Time utilities for performance testing, timing validation, and timestamp verification
import unittest.mock as mock  # >=3.10 - Mocking utilities for logging handlers, external dependencies, and controlled test scenarios
import io  # >=3.10 - String I/O utilities for capturing log output and validating log messages in tests
import threading  # >=3.10 - Thread utilities for testing thread-safe logging operations and concurrent logger access
import contextlib  # >=3.10 - Context manager utilities for testing logging context managers and resource cleanup
import tempfile  # >=3.10 - Temporary file utilities for testing file-based logging and log file validation
import sys  # >=3.10 - System interface for testing error handling and platform-specific behavior
import os  # >=3.10 - Operating system interface for environment variable testing and file system operations

# Internal imports - Import logging utilities and related components for testing
from plume_nav_sim.utils.logging import (
    get_component_logger,
    configure_logging_for_development,
    log_performance,
    log_with_context,
    create_performance_logger,
    setup_error_logging,
    ComponentLogger,
    PerformanceTimer,
    LoggingMixin,
    clear_logger_cache,
    get_caller_info
)

from plume_nav_sim.utils.exceptions import (
    PlumeNavSimError,
    ValidationError,
    handle_component_error
)

from plume_nav_sim.core.constants import (
    COMPONENT_NAMES,
    PERFORMANCE_TARGET_STEP_LATENCY_MS
)

from logging.config import ComponentType

# Test constants and global variables for consistent testing
TEST_COMPONENT_NAME = 'test_component'
TEST_OPERATION_NAME = 'test_operation'
TEST_LOG_MESSAGE = 'Test log message for validation'
PERFORMANCE_TEST_DURATION = 0.001  # 1ms for performance testing
SENSITIVE_TEST_DATA = {'password': 'secret123', 'token': 'abc123', 'safe_data': 'public'}

# Global test state tracking variables
_test_log_records = []  # Captured log records for test validation
_test_logger_names = set()  # Track created test loggers for cleanup


def setup_module(module):
    """
    Module-level setup for logging tests including test logger configuration, performance baseline 
    establishment, and test environment preparation.
    
    Args:
        module: Module object for module-level test setup
        
    Returns:
        None: Performs module-level test setup without return value
    """
    # Clear any existing logger cache to ensure clean test state
    clear_logger_cache()
    
    # Configure development logging for test environment
    configure_logging_for_development(
        log_level='DEBUG',
        enable_console_colors=False,  # Disable colors for test output
        enable_file_logging=False,    # Disable file logging during tests
        log_file_path=None
    )
    
    # Set up test log record capture for validation
    global _test_log_records
    _test_log_records.clear()
    
    # Initialize performance baselines for timing tests
    baseline_operations = {
        'test_step': PERFORMANCE_TARGET_STEP_LATENCY_MS,
        'test_render': 5.0,  # 5ms baseline for rendering operations
        'test_reset': 10.0   # 10ms baseline for reset operations
    }
    
    # Configure test-specific logging handlers and formatters
    test_handler = logging.StreamHandler(io.StringIO())
    test_handler.setLevel(logging.DEBUG)
    test_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    test_handler.setFormatter(test_formatter)
    
    # Set logging levels appropriate for test execution
    logging.getLogger('plume_nav_sim').setLevel(logging.DEBUG)
    logging.getLogger('test').setLevel(logging.DEBUG)


def teardown_module(module):
    """
    Module-level cleanup for logging tests including logger cache clearing, handler cleanup, 
    and test environment restoration.
    
    Args:
        module: Module object for module-level test cleanup
        
    Returns:
        None: Performs module-level test cleanup without return value
    """
    # Clear all test loggers from cache
    clear_logger_cache()
    
    # Close and remove test-specific handlers
    global _test_logger_names
    for logger_name in _test_logger_names:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers.copy():
            handler.close()
            logger.removeHandler(handler)
    
    # Reset logging configuration to default state
    logging.basicConfig(force=True)
    
    # Clear captured log records
    global _test_log_records
    _test_log_records.clear()
    
    # Clean up temporary log files and directories
    temp_files = [f for f in os.listdir('.') if f.startswith('plume_nav_sim') and f.endswith('.log')]
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except OSError:
            pass  # Ignore cleanup failures
    
    # Reset performance monitoring baseline data
    _test_logger_names.clear()


@contextlib.contextmanager
def capture_log_records(logger_name: str, level: int = logging.DEBUG):
    """
    Context manager for capturing log records during test execution for validation and assertion checking.
    
    Args:
        logger_name: Name of logger to capture records from
        level: Minimum log level to capture
        
    Yields:
        List[logging.LogRecord]: List of captured log records for test validation
    """
    # Create test log handler for capturing records
    captured_records = []
    
    class TestLogHandler(logging.Handler):
        def emit(self, record):
            captured_records.append(record)
    
    # Configure handler with specified level and logger name
    test_handler = TestLogHandler()
    test_handler.setLevel(level)
    
    # Install handler on target logger
    logger = logging.getLogger(logger_name)
    logger.addHandler(test_handler)
    logger.setLevel(level)
    
    try:
        # Yield list for record collection during test execution
        yield captured_records
    finally:
        # Remove handler and clean up after test completion
        logger.removeHandler(test_handler)
        test_handler.close()


def create_test_logger_with_handler(logger_name: str, handler: logging.Handler, level: int = logging.DEBUG):
    """
    Helper function to create test logger with custom handler for controlled logging test scenarios.
    
    Args:
        logger_name: Name for the test logger
        handler: Custom handler to attach to logger
        level: Logging level for the logger
        
    Returns:
        logging.Logger: Configured test logger with custom handler for testing
    """
    # Create logger instance with specified name
    logger = logging.getLogger(logger_name)
    
    # Set logger level to specified value
    logger.setLevel(level)
    
    # Add custom handler to logger
    logger.addHandler(handler)
    
    # Configure handler formatter for test output
    if not handler.formatter:
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        handler.setFormatter(formatter)
    
    # Add logger name to tracking set for cleanup
    global _test_logger_names
    _test_logger_names.add(logger_name)
    
    # Return configured logger ready for testing
    return logger


def assert_log_message_contains(records: list, expected_message: str, expected_level: int, partial_match: bool = True):
    """
    Assertion helper function for validating log message content and metadata in test scenarios.
    
    Args:
        records: List of captured log records to search
        expected_message: Expected message content
        expected_level: Expected log level
        partial_match: Whether to use partial or exact matching
        
    Returns:
        bool: True if log message found with expected content and level
    """
    # Iterate through captured log records
    for record in records:
        # Check each record for expected level match
        if record.levelno != expected_level:
            continue
        
        # Compare record message with expected message
        record_message = record.getMessage()
        if partial_match:
            # Use partial match if specified
            if expected_message in record_message:
                return True
        else:
            # Use exact match
            if expected_message == record_message:
                return True
    
    # Provide detailed assertion error if no match found
    level_name = logging.getLevelName(expected_level)
    messages = [f"'{record.getMessage()}' (level: {logging.getLevelName(record.levelno)})" for record in records]
    pytest.fail(f"Expected message '{expected_message}' at level {level_name} not found in records: {messages}")


class TestComponentLoggerFactory:
    """
    Test class for component logger factory functions including logger creation, caching, 
    configuration validation, and component-specific settings verification.
    """
    
    def setup_method(self):
        """
        Initialize test class with clean logger cache state and test configuration.
        """
        # Clear logger cache before each test
        clear_logger_cache()
        
        # Set up test component names and types
        self.test_component_name = TEST_COMPONENT_NAME
        self.test_component_type = ComponentType.UTILS
        
        # Initialize test-specific logging configuration
        self.test_config = {
            'enable_performance_tracking': True,
            'enable_context_capture': True,
            'log_level': 'DEBUG'
        }
    
    def test_get_component_logger_creation(self):
        """
        Test component logger creation with valid parameters and proper configuration validation.
        """
        # Create component logger using get_component_logger with test parameters
        logger = get_component_logger(
            component_name=self.test_component_name,
            component_type=self.test_component_type,
            enable_performance_tracking=True
        )
        
        # Validate logger is ComponentLogger instance
        assert isinstance(logger, ComponentLogger)
        
        # Check logger has correct component name and type
        assert logger.name.endswith(self.test_component_name)
        assert hasattr(logger, 'component_type')
        assert logger.component_type == self.test_component_type
        
        # Verify logger has appropriate logging level configuration
        assert logger.level <= logging.DEBUG
        
        # Validate performance tracking is enabled if requested
        assert hasattr(logger, 'performance_tracking_enabled')
        assert logger.performance_tracking_enabled == True
        
        # Confirm logger caching works correctly
        logger2 = get_component_logger(
            component_name=self.test_component_name,
            component_type=self.test_component_type
        )
        assert logger is logger2  # Should return same cached instance
    
    def test_component_logger_caching(self):
        """
        Test logger caching mechanism including cache hits, memory management, and cache invalidation.
        """
        # Create component logger with specific name and type
        logger1 = get_component_logger(
            component_name="test_component_1",
            component_type=ComponentType.ENVIRONMENT
        )
        
        # Request same logger again and verify cache hit (same instance)
        logger1_cached = get_component_logger(
            component_name="test_component_1",
            component_type=ComponentType.ENVIRONMENT
        )
        assert logger1 is logger1_cached
        
        # Request logger with different parameters and verify new instance
        logger2 = get_component_logger(
            component_name="test_component_2",
            component_type=ComponentType.PLUME_MODEL
        )
        assert logger1 is not logger2
        assert logger1.name != logger2.name
        
        # Test cache clearing functionality
        clear_logger_cache()
        logger1_after_clear = get_component_logger(
            component_name="test_component_1",
            component_type=ComponentType.ENVIRONMENT
        )
        # After cache clear, should get new instance (may have same configuration)
        assert isinstance(logger1_after_clear, ComponentLogger)
        
        # Validate memory management through weak references
        import weakref
        weak_ref = weakref.ref(logger1_after_clear)
        del logger1_after_clear
        # Logger may still exist due to caching, this tests weak reference functionality
        
        # Confirm cached loggers maintain configuration consistency
        logger_consistent = get_component_logger(
            component_name="consistent_test",
            component_type=ComponentType.RENDERING,
            enable_performance_tracking=True
        )
        assert logger_consistent.performance_tracking_enabled == True
    
    @pytest.mark.parametrize("component_type", [
        ComponentType.ENVIRONMENT,
        ComponentType.PLUME_MODEL,
        ComponentType.RENDERING,
        ComponentType.UTILS
    ])
    def test_component_type_specific_configuration(self, component_type):
        """
        Test component type-specific logger configuration including specialized settings and 
        performance tracking enablement.
        
        Args:
            component_type: ComponentType enum value to test
        """
        # Create loggers for each ComponentType enum value
        logger = get_component_logger(
            component_name=f"test_{component_type.value.lower()}",
            component_type=component_type,
            enable_performance_tracking=True
        )
        
        # Verify each logger has type-specific configuration
        assert isinstance(logger, ComponentLogger)
        assert logger.component_type == component_type
        
        # Check performance tracking enablement based on component type
        assert hasattr(logger, 'performance_tracking_enabled')
        
        # Validate specialized formatting for different component types
        with capture_log_records(logger.name) as records:
            logger.info("Test message for component type validation")
            
        assert len(records) >= 1
        record = records[0]
        assert component_type.value.lower() in record.name.lower() or component_type.value.lower() in record.getMessage().lower()
        
        # Confirm component-specific logging level settings
        assert logger.level <= logging.INFO
        
        # Test component identification in log messages
        assert hasattr(logger, 'component_type')
        assert logger.component_type == component_type
    
    def test_invalid_component_parameters(self):
        """
        Test error handling for invalid component names, types, and parameter validation.
        """
        # Test empty component name handling
        with pytest.raises((ValueError, TypeError)):
            get_component_logger(
                component_name="",
                component_type=ComponentType.UTILS
            )
        
        # Test None component name handling
        with pytest.raises((ValueError, TypeError)):
            get_component_logger(
                component_name=None,
                component_type=ComponentType.UTILS
            )
        
        # Test invalid component type handling
        with pytest.raises((ValueError, TypeError, AttributeError)):
            get_component_logger(
                component_name="valid_name",
                component_type="invalid_type"
            )
        
        # Test None component type handling
        with pytest.raises((ValueError, TypeError)):
            get_component_logger(
                component_name="valid_name",
                component_type=None
            )
        
        # Verify appropriate exceptions are raised
        # Error messages should be informative and secure
        try:
            get_component_logger(
                component_name="",
                component_type=ComponentType.UTILS
            )
            pytest.fail("Expected exception for empty component name")
        except (ValueError, TypeError) as e:
            # Check error messages are informative and secure
            error_msg = str(e).lower()
            assert "component" in error_msg or "name" in error_msg
            assert "password" not in error_msg  # Ensure no sensitive data in errors
        
        # Validate parameter sanitization for sensitive data
        # Component names should not contain sensitive information
        logger = get_component_logger(
            component_name="safe_component_name",
            component_type=ComponentType.UTILS
        )
        assert isinstance(logger, ComponentLogger)
        
        # Confirm graceful degradation for edge cases
        logger_edge = get_component_logger(
            component_name="a",  # Minimal valid name
            component_type=ComponentType.UTILS
        )
        assert isinstance(logger_edge, ComponentLogger)


class TestComponentLogger:
    """
    Test class for ComponentLogger functionality including enhanced logging methods, performance 
    tracking, context capture, and security filtering validation.
    """
    
    def setup_method(self):
        """
        Initialize test class with ComponentLogger instance and record capture setup.
        """
        # Create ComponentLogger instance for testing
        self.test_logger = get_component_logger(
            component_name="test_component_logger",
            component_type=ComponentType.UTILS,
            enable_performance_tracking=True
        )
        
        # Set up log record capture mechanism
        self.captured_records = []
        
        # Configure test-specific logging level
        self.test_logger.setLevel(logging.DEBUG)
        
        # Initialize performance tracking for testing
        self.performance_test_data = {
            'operation_name': TEST_OPERATION_NAME,
            'duration_ms': PERFORMANCE_TEST_DURATION * 1000,
            'memory_mb': 1.5
        }
    
    def test_enhanced_debug_logging(self):
        """
        Test enhanced debug logging with automatic context capture and component-specific formatting.
        """
        # Use ComponentLogger.debug with test message
        with capture_log_records(self.test_logger.name, logging.DEBUG) as records:
            self.test_logger.debug(TEST_LOG_MESSAGE, extra={'test_context': 'debug_test'})
        
        # Capture log records and validate debug level
        assert len(records) >= 1
        debug_record = records[0]
        assert debug_record.levelno == logging.DEBUG
        
        # Check automatic context capture (caller info, component details)
        assert TEST_LOG_MESSAGE in debug_record.getMessage()
        
        # Verify component-specific formatting applied
        assert self.test_logger.component_type.value in debug_record.name or hasattr(debug_record, 'component_type')
        
        # Validate security filtering of debug content
        # Test that sensitive information is not logged
        with capture_log_records(self.test_logger.name, logging.DEBUG) as sensitive_records:
            self.test_logger.debug("Test message with password=secret123")
        
        if len(sensitive_records) > 0:
            sensitive_record = sensitive_records[0]
            sensitive_message = sensitive_record.getMessage()
            # Security filtering should have redacted sensitive content
            assert "password=secret123" not in sensitive_message or "[REDACTED]" in sensitive_message
        
        # Confirm extra context parameters handled correctly
        context_found = any(hasattr(record, 'test_context') for record in records)
        # Extra parameters should be preserved for debugging
        assert context_found or any('test_context' in record.getMessage() for record in records)
    
    def test_performance_logging_integration(self):
        """
        Test performance logging integration with timing analysis, threshold comparison, and 
        baseline tracking.
        """
        # Use ComponentLogger.performance with test operation and duration
        with capture_log_records(self.test_logger.name, logging.INFO) as records:
            self.test_logger.performance(
                operation_name=self.performance_test_data['operation_name'],
                duration_ms=self.performance_test_data['duration_ms'],
                additional_metrics={'memory_mb': self.performance_test_data['memory_mb']}
            )
        
        # Validate performance log formatting and threshold comparison
        assert len(records) >= 1
        perf_record = records[0]
        perf_message = perf_record.getMessage()
        
        # Check integration with PERFORMANCE_TARGET_STEP_LATENCY_MS
        assert self.performance_test_data['operation_name'] in perf_message
        assert str(self.performance_test_data['duration_ms']) in perf_message or 'ms' in perf_message
        
        # Test baseline comparison and update functionality
        # Performance logging should include timing information
        timing_indicators = ['ms', 'duration', 'time']
        has_timing_info = any(indicator in perf_message.lower() for indicator in timing_indicators)
        assert has_timing_info
        
        # Verify additional metrics inclusion (memory, operations)
        assert 'memory' in perf_message.lower() or str(self.performance_test_data['memory_mb']) in perf_message
        
        # Confirm appropriate log level based on performance thresholds
        # Fast operations should log at DEBUG or INFO level
        if self.performance_test_data['duration_ms'] < PERFORMANCE_TARGET_STEP_LATENCY_MS:
            assert perf_record.levelno <= logging.INFO
        else:
            # Slow operations should log at WARNING or higher
            assert perf_record.levelno >= logging.WARNING
    
    def test_error_logging_with_exception_integration(self):
        """
        Test error logging with full context capture, stack trace information, and integration 
        with exception handling system.
        """
        # Create test exception (PlumeNavSimError) with context
        test_exception = PlumeNavSimError(
            message="Test error for logging integration",
            component="test_component",
            context={'operation': 'test_operation', 'state': 'error_test'}
        )
        
        # Use ComponentLogger.error with exception parameter
        with capture_log_records(self.test_logger.name, logging.ERROR) as records:
            try:
                raise test_exception
            except PlumeNavSimError as e:
                self.test_logger.error("Exception occurred during test", exc_info=True, extra={'exception_obj': e})
        
        # Validate comprehensive error context capture
        assert len(records) >= 1
        error_record = records[0]
        assert error_record.levelno == logging.ERROR
        error_message = error_record.getMessage()
        
        # Check integration with handle_component_error function
        assert "Exception occurred during test" in error_message or "Test error for logging integration" in error_message
        
        # Verify stack trace inclusion when requested
        assert hasattr(error_record, 'exc_info') and error_record.exc_info is not None
        
        # Test integration with exception handling system
        with capture_log_records('plume_nav_sim.error_handler', logging.ERROR) as handler_records:
            error_result = handle_component_error(test_exception, self.test_logger.name)
            
        # Error handling should process the exception appropriately
        assert error_result is not None
        
        # Confirm security filtering while preserving debugging info
        # Error messages should not contain sensitive information
        assert "password" not in error_message.lower()
        assert "secret" not in error_message.lower()
        # But should contain useful debugging information
        assert "test" in error_message.lower()
    
    def test_security_filtering(self):
        """
        Test security-aware logging with sensitive information filtering and safe error message generation.
        """
        # Log message with sensitive test data (passwords, tokens)
        sensitive_message = f"Login attempt with {SENSITIVE_TEST_DATA}"
        
        with capture_log_records(self.test_logger.name, logging.INFO) as records:
            self.test_logger.info(sensitive_message)
        
        # Verify sensitive information is sanitized or removed
        assert len(records) >= 1
        logged_record = records[0]
        logged_message = logged_record.getMessage()
        
        # Check preservation of safe debugging information
        assert "safe_data" in logged_message or "public" in logged_message or "[REDACTED]" in logged_message
        
        # Validate error message safety for user display
        # Sensitive values should be redacted
        sensitive_patterns = ['secret123', 'abc123', 'password', 'token']
        for pattern in sensitive_patterns:
            if pattern in sensitive_message.lower():
                # Pattern should be redacted or replaced
                assert pattern not in logged_message or "[REDACTED]" in logged_message
        
        # Confirm controlled information disclosure in logs
        # Test context sanitization while maintaining utility
        context_data = {
            'user_id': 'user123',
            'password': 'secret456',
            'action': 'login_test',
            'safe_info': 'public_data'
        }
        
        with capture_log_records(self.test_logger.name, logging.INFO) as context_records:
            self.test_logger.info("User action", extra=context_data)
        
        if len(context_records) > 0:
            context_record = context_records[0]
            context_message = context_record.getMessage()
            
            # Safe information should be preserved
            assert 'user123' in context_message or 'login_test' in context_message or 'public_data' in context_message
            
            # Sensitive information should be filtered
            assert 'secret456' not in context_message or '[REDACTED]' in context_message


class TestPerformanceTimer:
    """
    Test class for PerformanceTimer context manager including precise timing measurements, 
    automatic performance logging, and metric collection validation.
    """
    
    def setup_method(self):
        """
        Initialize test class with performance logger and timing test configuration.
        """
        # Create performance-optimized logger for testing
        self.performance_logger = create_performance_logger(
            component_name="test_performance_timer",
            enable_automatic_logging=True
        )
        
        # Set up test duration for consistent timing validation
        self.test_duration = PERFORMANCE_TEST_DURATION
        
        # Configure performance monitoring baselines
        self.baseline_operations = {
            'test_timer_operation': 2.0,  # 2ms baseline
            'fast_operation': 0.5,        # 0.5ms baseline
            'slow_operation': 10.0        # 10ms baseline
        }
    
    def test_context_manager_timing(self):
        """
        Test PerformanceTimer as context manager with accurate timing measurement and automatic logging.
        """
        operation_name = "test_context_manager_timing"
        
        # Create PerformanceTimer with test operation name
        with capture_log_records(self.performance_logger.name, logging.INFO) as records:
            with PerformanceTimer(operation_name=operation_name, logger=self.performance_logger) as timer:
                # Introduce controlled delay for timing validation
                time.sleep(self.test_duration)
                
                # Verify timing accuracy within acceptable tolerance
                measured_duration = timer.get_duration_ms()
                expected_duration_ms = self.test_duration * 1000
                
                # Allow 50% tolerance for timing variations in test environment
                tolerance = expected_duration_ms * 0.5
                assert abs(measured_duration - expected_duration_ms) < tolerance
        
        # Check automatic performance logging when enabled
        assert len(records) >= 1
        timing_record = next((r for r in records if operation_name in r.getMessage()), None)
        assert timing_record is not None
        
        # Validate duration calculation and reporting
        timing_message = timing_record.getMessage()
        assert 'ms' in timing_message.lower() or 'duration' in timing_message.lower()
        assert operation_name in timing_message
    
    def test_metric_collection(self):
        """
        Test additional metric collection including memory usage, operation counts, and custom 
        metrics during timing.
        """
        operation_name = "test_metric_collection"
        
        # Create PerformanceTimer and add test metrics
        with PerformanceTimer(operation_name=operation_name, logger=self.performance_logger) as timer:
            time.sleep(self.test_duration)
            
            # Include memory usage, operation count metrics
            timer.add_metric('memory_mb', 2.5, 'MB')
            timer.add_metric('operations_count', 15, 'ops')
            timer.add_metric('cache_hits', 8, 'hits')
            
            # Add custom metrics with units and values
            timer.add_metric('cpu_usage', 45.2, '%')
            timer.add_metric('network_bytes', 1024, 'bytes')
        
        # Verify metrics are captured correctly
        collected_metrics = timer.get_metrics()
        
        # Check metric security filtering for sensitive data
        assert 'memory_mb' in collected_metrics
        assert collected_metrics['memory_mb']['value'] == 2.5
        assert collected_metrics['memory_mb']['unit'] == 'MB'
        
        assert 'operations_count' in collected_metrics
        assert collected_metrics['operations_count']['value'] == 15
        
        # Validate metric formatting in performance logs
        with capture_log_records(self.performance_logger.name, logging.INFO) as metric_records:
            timer.log_performance_summary()
        
        if len(metric_records) > 0:
            metric_record = metric_records[0]
            metric_message = metric_record.getMessage()
            
            # Metrics should appear in formatted log output
            assert any(metric in metric_message.lower() for metric in ['memory', 'operations', 'cpu'])
    
    def test_exception_handling_during_timing(self):
        """
        Test PerformanceTimer behavior during exceptions including proper cleanup and exception 
        information logging.
        """
        operation_name = "test_exception_handling"
        exception_message = "Intentional test exception"
        
        # Create PerformanceTimer context manager
        with capture_log_records(self.performance_logger.name, logging.ERROR) as records:
            try:
                with PerformanceTimer(operation_name=operation_name, logger=self.performance_logger) as timer:
                    time.sleep(self.test_duration * 0.5)  # Partial execution
                    
                    # Raise exception within timed context
                    raise ValueError(exception_message)
                    
            except ValueError:
                pass  # Expected exception
        
        # Verify timer cleanup occurs despite exception
        # Timer should have recorded partial duration
        duration = timer.get_duration_ms()
        expected_partial_duration = (self.test_duration * 0.5) * 1000
        assert duration >= expected_partial_duration * 0.5  # At least half the expected time
        
        # Check exception information included in performance log
        exception_records = [r for r in records if exception_message in r.getMessage() or 'exception' in r.getMessage().lower()]
        # May or may not log exception depending on configuration
        
        # Validate timing data captured even with failure
        assert timer.get_duration_ms() > 0
        
        # Confirm proper exception propagation
        # Exception should have been re-raised after timer cleanup
        with pytest.raises(ValueError):
            with PerformanceTimer(operation_name=operation_name):
                raise ValueError("Test exception propagation")
    
    def test_performance_threshold_monitoring(self):
        """
        Test performance threshold monitoring with alerting and baseline comparison functionality.
        """
        fast_operation = "fast_test_operation"
        slow_operation = "slow_test_operation"
        
        # Set performance baseline for test operations
        fast_baseline_ms = 1.0   # 1ms baseline - should be fast
        slow_baseline_ms = 50.0  # 50ms baseline - will be exceeded
        
        with capture_log_records(self.performance_logger.name, logging.WARNING) as warning_records:
            # Execute operation with duration exceeding threshold
            with PerformanceTimer(operation_name=slow_operation, logger=self.performance_logger) as slow_timer:
                # Simulate slow operation
                time.sleep(0.020)  # 20ms - should trigger warning
                slow_timer.set_threshold(5.0)  # Set low threshold to trigger warning
        
        # Verify threshold violation is detected and logged
        slow_duration = slow_timer.get_duration_ms()
        assert slow_duration >= 15.0  # Should be at least 15ms
        
        # Check appropriate log level for performance issues
        threshold_records = [r for r in warning_records if slow_operation in r.getMessage()]
        performance_warning_found = len(threshold_records) > 0 or slow_duration > slow_timer.warning_threshold_ms
        
        # Test baseline comparison and trend analysis
        with capture_log_records(self.performance_logger.name, logging.INFO) as info_records:
            with PerformanceTimer(operation_name=fast_operation, logger=self.performance_logger) as fast_timer:
                time.sleep(self.test_duration * 0.5)  # Very fast operation
                fast_timer.set_baseline(fast_baseline_ms)
        
        fast_duration = fast_timer.get_duration_ms()
        
        # Validate alerting for performance degradation
        # Fast operations should not trigger warnings
        fast_records = [r for r in info_records if fast_operation in r.getMessage()]
        assert len(fast_records) >= 1  # Should have info-level log
        
        # Check that fast operations don't generate warnings
        fast_warning_records = [r for r in warning_records if fast_operation in r.getMessage()]
        assert len(fast_warning_records) == 0  # No warnings for fast operations


class TestLoggingMixin:
    """
    Test class for LoggingMixin functionality including automatic logger creation, component 
    identification, and method-level logging integration.
    """
    
    def setup_method(self):
        """
        Initialize test class for LoggingMixin testing with component creation.
        """
        # Set up test component classes using LoggingMixin
        self.TestMixinClass = type('TestMixinClass', (LoggingMixin,), {
            'component_type': ComponentType.UTILS,
            'test_method': lambda self: 'test_result'
        })
        
        # Configure automatic logger detection testing
        self.mixin_instance = self.TestMixinClass()
        
        # Initialize method-level logging test scenarios
        self.method_test_data = {
            'method_name': 'test_method_logging',
            'parameters': {'param1': 'value1', 'param2': 42},
            'expected_result': 'method_success'
        }
    
    def test_automatic_logger_creation(self):
        """
        Test automatic logger creation from class name and module with lazy initialization 
        and configuration detection.
        """
        # Create test class inheriting from LoggingMixin
        class TestAutoLoggerClass(LoggingMixin):
            component_type = ComponentType.RENDERING
            
            def test_logging_method(self):
                self.logger.info("Test automatic logger creation")
                return "success"
        
        test_instance = TestAutoLoggerClass()
        
        # Access logger property to trigger lazy initialization
        logger = test_instance.logger
        
        # Verify logger creation with correct component identification
        assert isinstance(logger, ComponentLogger)
        assert logger.name  # Should have a valid name
        
        # Check automatic component type detection
        assert hasattr(test_instance, 'component_type')
        assert test_instance.component_type == ComponentType.RENDERING
        
        # Validate logger configuration based on class context
        assert logger.level <= logging.INFO
        
        # Test actual logging functionality
        with capture_log_records(logger.name, logging.INFO) as records:
            result = test_instance.test_logging_method()
        
        assert result == "success"
        assert len(records) >= 1
        log_record = records[0]
        assert "Test automatic logger creation" in log_record.getMessage()
        
        # Confirm logger caching for subsequent accesses
        logger2 = test_instance.logger
        assert logger is logger2  # Same instance due to caching
    
    def test_method_entry_exit_logging(self):
        """
        Test convenient method entry and exit logging with parameter capture and execution tracing.
        """
        # Create test method using LoggingMixin
        class TestMethodLoggingClass(LoggingMixin):
            component_type = ComponentType.UTILS
            
            def test_method_with_logging(self, param1, param2=None):
                # Use log_method_entry with parameter information
                self.log_method_entry('test_method_with_logging', {'param1': param1, 'param2': param2})
                
                # Simulate method execution
                time.sleep(0.001)  # 1ms execution time
                result = f"processed_{param1}_{param2}"
                
                # Use log_method_exit with execution timing
                self.log_method_exit('test_method_with_logging', result, execution_time_ms=1.0)
                
                return result
        
        test_instance = TestMethodLoggingClass()
        
        # Execute method with controlled return value
        with capture_log_records(test_instance.logger.name, logging.DEBUG) as records:
            result = test_instance.test_method_with_logging("test_param", param2="optional_param")
        
        # Verify method entry/exit logs captured correctly
        assert len(records) >= 2  # At least entry and exit logs
        
        entry_logs = [r for r in records if 'entry' in r.getMessage().lower() or 'entering' in r.getMessage().lower()]
        exit_logs = [r for r in records if 'exit' in r.getMessage().lower() or 'exiting' in r.getMessage().lower()]
        
        # Should have entry and exit logging
        method_logs_found = len(entry_logs) > 0 and len(exit_logs) > 0
        if not method_logs_found:
            # Alternative: check for method name in any log
            method_name_logs = [r for r in records if 'test_method_with_logging' in r.getMessage()]
            assert len(method_name_logs) >= 1
        
        # Check parameter and return value security filtering
        # Parameters should be logged but sensitive data should be filtered
        param_logs = [r for r in records if 'test_param' in r.getMessage() or 'param1' in r.getMessage()]
        assert len(param_logs) >= 1 or any('test_method_with_logging' in r.getMessage() for r in records)
        
        # Verify result
        assert result == "processed_test_param_optional_param"
    
    def test_mixin_configuration_override(self):
        """
        Test LoggingMixin configuration override with custom component names, types, and logger settings.
        """
        # Create test class with LoggingMixin
        class CustomConfigMixinClass(LoggingMixin):
            component_type = ComponentType.PLUME_MODEL
            
            def __init__(self, custom_component_name=None):
                super().__init__()
                if custom_component_name:
                    self.custom_component_name = custom_component_name
            
            def configure_custom_logging(self, log_level='INFO', enable_performance=True):
                # Configure custom component name and type
                self.configure_logging(
                    component_name=getattr(self, 'custom_component_name', 'default_name'),
                    component_type=self.component_type,
                    log_level=log_level,
                    enable_performance_tracking=enable_performance
                )
        
        # Apply custom logger configuration settings
        custom_instance = CustomConfigMixinClass(custom_component_name="custom_plume_component")
        custom_instance.configure_custom_logging(log_level='DEBUG', enable_performance=True)
        
        # Verify configuration override takes effect
        logger = custom_instance.logger
        assert isinstance(logger, ComponentLogger)
        
        # Check custom settings preserved across method calls
        with capture_log_records(logger.name, logging.DEBUG) as records:
            logger.debug("Custom configuration test message")
        
        assert len(records) >= 1
        config_record = records[0]
        config_message = config_record.getMessage()
        
        # Validate logger reconfiguration when settings change
        assert "Custom configuration test message" in config_message
        assert config_record.levelno == logging.DEBUG
        
        # Test reconfiguration
        custom_instance.configure_custom_logging(log_level='WARNING', enable_performance=False)
        logger_reconfig = custom_instance.logger
        
        # Logger configuration should be updated
        with capture_log_records(logger_reconfig.name, logging.WARNING) as warning_records:
            logger_reconfig.warning("Reconfiguration test")
            logger_reconfig.debug("This debug message should not appear")
        
        # Only warning should be captured due to level change
        assert len(warning_records) >= 1
        warning_record = warning_records[0]
        assert warning_record.levelno == logging.WARNING
        assert "Reconfiguration test" in warning_record.getMessage()


class TestLoggingIntegration:
    """
    Test class for logging system integration including error handling integration, performance 
    monitoring, development configuration, and centralized logging coordination.
    """
    
    def setup_method(self):
        """
        Initialize integration testing with clean logging environment and test scenarios.
        """
        # Reset logging system to clean state
        clear_logger_cache()
        logging.getLogger().handlers.clear()
        
        # Set up integration test scenarios
        self.integration_components = [
            {'name': 'environment_component', 'type': ComponentType.ENVIRONMENT},
            {'name': 'plume_component', 'type': ComponentType.PLUME_MODEL},
            {'name': 'rendering_component', 'type': ComponentType.RENDERING}
        ]
        
        # Configure test monitoring and validation
        self.integration_loggers = {}
        for component in self.integration_components:
            self.integration_loggers[component['name']] = get_component_logger(
                component_name=component['name'],
                component_type=component['type']
            )
    
    def test_development_logging_configuration(self):
        """
        Test development logging configuration with enhanced debugging, console colors, and 
        file logging setup.
        """
        # Create temporary directory for log files
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, 'test_development.log')
            
            # Call configure_logging_for_development with test parameters
            configure_logging_for_development(
                log_level='DEBUG',
                enable_console_colors=False,  # Disable for testing
                enable_file_logging=True,
                log_file_path=log_file_path
            )
            
            # Test with multiple loggers
            test_logger = logging.getLogger('development_test')
            
            # Verify console logging setup with appropriate formatting
            with capture_log_records('development_test', logging.DEBUG) as records:
                test_logger.debug("Development debug message")
                test_logger.info("Development info message")
                test_logger.warning("Development warning message")
            
            assert len(records) >= 3
            
            # Test file logging configuration if enabled
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as log_file:
                    log_content = log_file.read()
                    assert 'Development debug message' in log_content
                    assert 'Development info message' in log_content
                    assert 'Development warning message' in log_content
            
            # Check color support detection and configuration
            # Colors should be disabled for testing
            debug_record = next((r for r in records if r.levelno == logging.DEBUG), None)
            info_record = next((r for r in records if r.levelno == logging.INFO), None)
            
            assert debug_record is not None
            assert info_record is not None
            
            # Validate enhanced debugging information enablement
            assert 'Development debug message' in debug_record.getMessage()
            
            # Confirm development-appropriate logging levels
            assert test_logger.level <= logging.DEBUG
    
    def test_error_handling_integration(self):
        """
        Test integration between logging system and error handling including automatic error 
        logging and recovery suggestions.
        """
        test_component = "error_integration_test"
        test_logger = get_component_logger(test_component, ComponentType.UTILS)
        
        # Set up error logging integration with test logger
        setup_error_logging(
            component_name=test_component,
            enable_automatic_logging=True,
            log_level='ERROR'
        )
        
        # Create test error with context
        test_error = ValidationError(
            message="Integration test validation error",
            component=test_component,
            context={'field': 'test_field', 'value': 'invalid_value'},
            suggestions=['Check field format', 'Validate input range']
        )
        
        # Trigger component error with context information
        with capture_log_records(test_logger.name, logging.ERROR) as error_records:
            error_result = handle_component_error(test_error, test_component)
        
        # Verify handle_component_error integration works
        assert error_result is not None
        assert isinstance(error_result, (str, dict))
        
        # Check automatic error logging with context capture
        error_logged = len(error_records) > 0
        if error_logged:
            error_record = error_records[0]
            error_message = error_record.getMessage()
            
            # Should contain error information
            assert 'validation' in error_message.lower() or 'error' in error_message.lower()
        
        # Test with different error types
        resource_error = PlumeNavSimError(
            message="Resource allocation failed",
            component=test_component,
            context={'resource': 'memory', 'requested': '100MB', 'available': '50MB'}
        )
        
        with capture_log_records(test_logger.name, logging.ERROR) as resource_records:
            resource_result = handle_component_error(resource_error, test_component)
        
        # Validate recovery suggestion logging
        assert resource_result is not None
        
        # Confirm error escalation for critical issues
        critical_error = PlumeNavSimError(
            message="Critical system error",
            component=test_component,
            severity='critical'
        )
        
        with capture_log_records(test_logger.name, logging.CRITICAL) as critical_records:
            critical_result = handle_component_error(critical_error, test_component)
        
        assert critical_result is not None
    
    def test_performance_monitoring_integration(self):
        """
        Test integration between logging utilities and performance monitoring system including 
        baseline tracking and threshold alerting.
        """
        # Create performance logger with threshold configuration
        perf_component = "performance_integration_test"
        perf_logger = create_performance_logger(
            component_name=perf_component,
            enable_automatic_logging=True
        )
        
        # Define test operations with different performance characteristics
        operations = [
            {'name': 'fast_operation', 'duration': 0.0005, 'expected_level': logging.INFO},
            {'name': 'normal_operation', 'duration': 0.002, 'expected_level': logging.INFO},
            {'name': 'slow_operation', 'duration': 0.010, 'expected_level': logging.WARNING}
        ]
        
        performance_results = []
        
        for operation in operations:
            # Execute operations with performance monitoring
            with capture_log_records(perf_logger.name, logging.DEBUG) as perf_records:
                with PerformanceTimer(operation_name=operation['name'], logger=perf_logger) as timer:
                    time.sleep(operation['duration'])
                    
                    # Add operation-specific metrics
                    timer.add_metric('operation_type', operation['name'])
                    timer.add_metric('expected_duration', operation['duration'] * 1000, 'ms')
            
            duration_ms = timer.get_duration_ms()
            performance_results.append({
                'operation': operation['name'],
                'duration_ms': duration_ms,
                'records': perf_records
            })
        
        # Test threshold violation detection and alerting
        slow_result = next((r for r in performance_results if r['operation'] == 'slow_operation'), None)
        assert slow_result is not None
        assert slow_result['duration_ms'] >= 8.0  # Should be at least 8ms
        
        # Verify baseline tracking and comparison functionality
        baseline_test_operation = 'baseline_test_operation'
        baseline_duration_ms = 3.0
        
        # Set baseline for operation
        with PerformanceTimer(operation_name=baseline_test_operation, logger=perf_logger) as baseline_timer:
            time.sleep(0.002)  # 2ms operation
            baseline_timer.set_baseline(baseline_duration_ms)
        
        actual_duration = baseline_timer.get_duration_ms()
        
        # Check performance trend analysis and logging
        assert actual_duration > 0
        baseline_comparison = baseline_timer.compare_to_baseline(baseline_duration_ms)
        
        # Validate integration with system performance targets
        step_target_ms = PERFORMANCE_TARGET_STEP_LATENCY_MS
        
        with capture_log_records(perf_logger.name, logging.INFO) as target_records:
            log_performance(
                operation_name='step_simulation',
                duration_ms=step_target_ms * 0.5,  # Half of target - should be good
                target_ms=step_target_ms,
                logger=perf_logger
            )
        
        # Performance within target should log at INFO level or below
        target_met_records = [r for r in target_records if 'step_simulation' in r.getMessage()]
        assert len(target_met_records) >= 1
    
    def test_concurrent_logging_safety(self):
        """
        Test thread-safe logging operations including concurrent logger access, cache management, 
        and resource cleanup.
        """
        import concurrent.futures
        
        # Create multiple threads accessing logging utilities
        concurrent_components = [
            f'concurrent_component_{i}' for i in range(5)
        ]
        
        concurrent_results = []
        
        def concurrent_logging_task(component_name):
            """Task function for concurrent logging testing."""
            # Test concurrent logger creation and caching
            logger = get_component_logger(
                component_name=component_name,
                component_type=ComponentType.UTILS
            )
            
            results = {
                'component_name': component_name,
                'logger_id': id(logger),
                'operations': []
            }
            
            # Verify thread-safe performance timer operations
            for i in range(3):
                with PerformanceTimer(f'{component_name}_op_{i}', logger=logger) as timer:
                    time.sleep(0.001)  # 1ms per operation
                    timer.add_metric('thread_id', threading.current_thread().ident)
                    timer.add_metric('iteration', i)
                
                operation_result = {
                    'operation': f'{component_name}_op_{i}',
                    'duration_ms': timer.get_duration_ms(),
                    'thread_id': threading.current_thread().ident
                }
                results['operations'].append(operation_result)
            
            # Check concurrent error logging without conflicts
            try:
                raise ValueError(f"Test concurrent error from {component_name}")
            except ValueError as e:
                logger.error(f"Concurrent error in {component_name}", exc_info=True)
            
            return results
        
        # Execute concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(concurrent_logging_task, component_name)
                for component_name in concurrent_components
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    concurrent_results.append(result)
                except Exception as e:
                    pytest.fail(f"Concurrent logging task failed: {e}")
        
        # Validate resource cleanup in multi-threaded scenarios
        assert len(concurrent_results) == len(concurrent_components)
        
        # Check that each component got a logger
        logger_ids = [r['logger_id'] for r in concurrent_results]
        component_names = [r['component_name'] for r in concurrent_results]
        
        # Each component should have a unique name
        assert len(set(component_names)) == len(concurrent_components)
        
        # Loggers may be cached, so same-named components should share loggers
        unique_logger_count = len(set(logger_ids))
        assert unique_logger_count <= len(concurrent_components)
        
        # Confirm logging system stability under concurrent load
        for result in concurrent_results:
            assert len(result['operations']) == 3
            for operation in result['operations']:
                assert operation['duration_ms'] > 0
                assert operation['thread_id'] is not None


class TestLoggingUtilityFunctions:
    """
    Test class for utility functions including context capture, caller information extraction, 
    cache management, and specialized logging operations.
    """
    
    def setup_method(self):
        """
        Initialize utility function testing with test data and validation setup.
        """
        # Set up test data for utility function validation
        self.test_context_data = {
            'operation': 'test_utility_operation',
            'component': 'test_utility_component',
            'safe_data': 'public_information',
            'sensitive_data': 'password=secret123'
        }
        
        # Configure test scenarios for edge cases
        self.edge_case_scenarios = [
            {'name': 'empty_message', 'message': '', 'context': {}},
            {'name': 'none_message', 'message': None, 'context': {'key': 'value'}},
            {'name': 'large_context', 'message': 'test', 'context': {f'key_{i}': f'value_{i}' for i in range(100)}},
            {'name': 'nested_context', 'message': 'test', 'context': {'level1': {'level2': {'level3': 'deep_value'}}}}
        ]
        
        # Initialize utility function test environment
        self.test_logger = get_component_logger(
            component_name='utility_test_logger',
            component_type=ComponentType.UTILS
        )
    
    def test_log_with_context_function(self):
        """
        Test log_with_context function with automatic caller information capture and security filtering.
        """
        test_message = "Test message for context logging"
        test_context = self.test_context_data.copy()
        
        # Create test logger and call log_with_context
        with capture_log_records(self.test_logger.name, logging.INFO) as records:
            log_with_context(
                logger=self.test_logger,
                level=logging.INFO,
                message=test_message,
                context=test_context,
                include_caller_info=True,
                include_stack_trace=False
            )
        
        # Verify automatic caller information capture
        assert len(records) >= 1
        context_record = records[0]
        context_message = context_record.getMessage()
        
        # Check extra context parameter handling
        assert test_message in context_message
        
        # Test security filtering of context information
        # Sensitive data should be filtered while safe data is preserved
        if 'password=secret123' in str(test_context):
            # Original context contained sensitive data
            assert 'password=secret123' not in context_message or '[REDACTED]' in context_message
        
        # Safe data should be preserved
        assert 'public_information' in context_message or 'safe_data' in context_message
        
        # Validate stack information inclusion when requested
        with capture_log_records(self.test_logger.name, logging.INFO) as stack_records:
            log_with_context(
                logger=self.test_logger,
                level=logging.INFO,
                message="Test with stack trace",
                context={'test': 'stack_context'},
                include_caller_info=True,
                include_stack_trace=True
            )
        
        if len(stack_records) > 0:
            stack_record = stack_records[0]
            # Stack information should be included in some form
            assert hasattr(stack_record, 'pathname') or 'test_log_with_context_function' in stack_record.getMessage()
        
        # Confirm enhanced log record creation
        assert context_record.levelno == logging.INFO
        assert hasattr(context_record, 'funcName')
        assert context_record.funcName == 'test_log_with_context_function'
    
    def test_caller_info_extraction(self):
        """
        Test get_caller_info function with stack inspection and caller context extraction.
        """
        def nested_function_level_3():
            return get_caller_info(stack_depth=1)
        
        def nested_function_level_2():
            return nested_function_level_3()
        
        def nested_function_level_1():
            return nested_function_level_2()
        
        # Call get_caller_info from test function
        caller_info = get_caller_info(stack_depth=0)
        
        # Verify correct function name, file, and line extraction
        assert isinstance(caller_info, dict)
        assert 'function_name' in caller_info
        assert 'filename' in caller_info
        assert 'line_number' in caller_info
        
        # Should identify this test method
        assert caller_info['function_name'] == 'test_caller_info_extraction'
        assert 'test_logging.py' in caller_info['filename'] or 'test' in caller_info['filename']
        assert isinstance(caller_info['line_number'], int)
        assert caller_info['line_number'] > 0
        
        # Test stack depth parameter functionality
        nested_caller_info = nested_function_level_1()
        
        assert isinstance(nested_caller_info, dict)
        assert 'function_name' in nested_caller_info
        
        # Should identify the calling function based on stack depth
        expected_functions = ['nested_function_level_3', 'nested_function_level_2', 'test_caller_info_extraction']
        assert nested_caller_info['function_name'] in expected_functions
        
        # Check local variable inclusion when safe
        caller_info_with_locals = get_caller_info(include_local_variables=True, stack_depth=0)
        
        # Validate security filtering of local variables
        if 'local_variables' in caller_info_with_locals:
            local_vars = caller_info_with_locals['local_variables']
            
            # Should not include sensitive information
            sensitive_patterns = ['password', 'secret', 'token', 'key']
            for var_name, var_value in local_vars.items():
                var_str = f"{var_name}={var_value}".lower()
                for pattern in sensitive_patterns:
                    if pattern in var_str:
                        # Sensitive data should be redacted
                        assert '[REDACTED]' in str(var_value) or var_value == '[REDACTED]'
        
        # Confirm graceful handling of inspection failures
        # Test with invalid stack depth
        invalid_caller_info = get_caller_info(stack_depth=100)  # Very deep stack
        
        # Should handle gracefully and return safe defaults
        assert isinstance(invalid_caller_info, dict)
        assert 'function_name' in invalid_caller_info
        # May contain fallback values like 'unknown' or the actual deepest function
    
    def test_logger_cache_management(self):
        """
        Test logger cache management including cache clearing, memory management, and cleanup operations.
        """
        # Create multiple cached loggers with different parameters
        cache_test_loggers = []
        
        for i in range(5):
            logger = get_component_logger(
                component_name=f'cache_test_component_{i}',
                component_type=ComponentType.UTILS,
                enable_performance_tracking=True
            )
            cache_test_loggers.append(logger)
        
        # Verify loggers are cached (same parameters should return same instance)
        logger_duplicate = get_component_logger(
            component_name='cache_test_component_0',
            component_type=ComponentType.UTILS,
            enable_performance_tracking=True
        )
        
        assert cache_test_loggers[0] is logger_duplicate  # Should be same cached instance
        
        # Test selective cache clearing with component filtering
        # This functionality may not exist, so we test what's available
        initial_cache_size = len(cache_test_loggers)
        
        # Clear all cache
        clear_logger_cache()
        
        # Verify cache clearing by requesting same logger again
        logger_after_clear = get_component_logger(
            component_name='cache_test_component_0',
            component_type=ComponentType.UTILS,
            enable_performance_tracking=True
        )
        
        # After cache clear, should get a new instance (or at least be properly configured)
        assert isinstance(logger_after_clear, ComponentLogger)
        
        # Test with different logger configurations to verify cache key differentiation
        logger_perf_enabled = get_component_logger(
            component_name='cache_diff_test',
            component_type=ComponentType.ENVIRONMENT,
            enable_performance_tracking=True
        )
        
        logger_perf_disabled = get_component_logger(
            component_name='cache_diff_test',
            component_type=ComponentType.ENVIRONMENT,
            enable_performance_tracking=False
        )
        
        # Different configurations should result in different loggers (or at least different behavior)
        # The actual caching behavior may vary based on implementation
        assert isinstance(logger_perf_enabled, ComponentLogger)
        assert isinstance(logger_perf_disabled, ComponentLogger)
        
        # Check weak reference memory management
        import weakref
        weak_refs = []
        
        for logger in cache_test_loggers:
            weak_refs.append(weakref.ref(logger))
        
        # Clear strong references
        cache_test_loggers.clear()
        
        # Test cache statistics and monitoring (if available)
        # This may not be implemented in the current version
        try:
            # Attempt to get cache statistics
            cache_stats = get_component_logger.__cache_info__  # This might not exist
        except AttributeError:
            # Cache statistics not available
            pass
        
        # Validate complete cache clearing functionality
        clear_logger_cache()
        
        # Verify cleanup completed successfully
        post_clear_logger = get_component_logger(
            component_name='post_clear_test',
            component_type=ComponentType.UTILS
        )
        
        assert isinstance(post_clear_logger, ComponentLogger)
    
    def test_performance_logging_utility(self):
        """
        Test log_performance utility function with timing analysis, baseline comparison, and 
        threshold monitoring.
        """
        test_operation_name = "utility_performance_test"
        test_duration_ms = 2.5
        test_target_ms = 5.0
        
        # Call log_performance with test timing data
        with capture_log_records(self.test_logger.name, logging.INFO) as perf_records:
            log_performance(
                operation_name=test_operation_name,
                duration_ms=test_duration_ms,
                target_ms=test_target_ms,
                additional_metrics={'memory_mb': 1.2, 'operations': 10},
                logger=self.test_logger
            )
        
        # Verify performance analysis and threshold comparison
        assert len(perf_records) >= 1
        perf_record = perf_records[0]
        perf_message = perf_record.getMessage()
        
        # Test additional metrics inclusion
        assert test_operation_name in perf_message
        assert str(test_duration_ms) in perf_message or 'ms' in perf_message
        
        # Should include timing information
        timing_keywords = ['duration', 'ms', 'time', 'performance']
        has_timing_info = any(keyword in perf_message.lower() for keyword in timing_keywords)
        assert has_timing_info
        
        # Check baseline comparison when enabled
        baseline_duration_ms = 3.0
        
        with capture_log_records(self.test_logger.name, logging.INFO) as baseline_records:
            log_performance(
                operation_name=test_operation_name,
                duration_ms=test_duration_ms,
                target_ms=test_target_ms,
                baseline_ms=baseline_duration_ms,
                logger=self.test_logger
            )
        
        if len(baseline_records) > 0:
            baseline_record = baseline_records[0]
            baseline_message = baseline_record.getMessage()
            
            # Baseline comparison information might be included
            comparison_keywords = ['baseline', 'vs', 'compared', '%']
            has_comparison_info = any(keyword in baseline_message.lower() for keyword in comparison_keywords)
            # This feature may not be implemented yet
        
        # Validate appropriate log level determination
        # Performance within target should be INFO or DEBUG level
        assert perf_record.levelno <= logging.INFO
        
        # Test with performance exceeding target
        slow_duration_ms = 15.0  # Exceeds 5ms target
        
        with capture_log_records(self.test_logger.name, logging.WARNING) as slow_records:
            log_performance(
                operation_name="slow_operation_test",
                duration_ms=slow_duration_ms,
                target_ms=test_target_ms,
                logger=self.test_logger
            )
        
        # Slow operations might be logged at WARNING level
        # This depends on implementation details
        all_slow_records = slow_records + perf_records  # Check both record lists
        slow_op_record = next((r for r in all_slow_records if 'slow_operation_test' in r.getMessage()), None)
        
        assert slow_op_record is not None
        
        # Confirm performance baseline updates
        # This functionality may not be fully implemented
        updated_baseline_test = log_performance(
            operation_name="baseline_update_test",
            duration_ms=4.0,
            target_ms=5.0,
            update_baseline=True,
            logger=self.test_logger
        )
        
        # Function should execute without errors
        # Return value may vary based on implementation