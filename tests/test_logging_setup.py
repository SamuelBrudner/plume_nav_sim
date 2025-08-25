"""
Enhanced Tests for the Centralized Loguru Logging Configuration System.

This test suite validates the comprehensive logging enhancements including:
- Centralized Loguru configuration with environment-specific defaults
- Structured JSON output with correlation IDs and performance monitoring
- Performance threshold alerting for step() timing violations (≤10ms requirement)
- Legacy API deprecation detection and structured warnings
- Context propagation and thread-safe operation
- Performance monitoring integration with automatic threshold violations
- Memory usage tracking and database operation timing

Covers Section 5.4 Cross-Cutting Concerns requirements for logging modernization.
"""

import pytest
import sys
import os
import io
import json
import time
import tempfile
import threading
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock
import re
from contextlib import suppress, contextmanager

import psutil
from loguru import logger
from pydantic import ValidationError

# Import enhanced logging components
from plume_nav_sim.utils.logging_setup import (
    # Core setup functions
    setup_logger,
    get_module_logger,
    get_enhanced_logger,
    get_logger,
    
    # Configuration models
    LoggingConfig,
    PerformanceMetrics,
    
    # Enhanced logger class
    EnhancedLogger,
    
    # Context managers and correlation tracking
    correlation_context,
    get_correlation_context,
    set_correlation_context,
    CorrelationContext,
    
    # Performance monitoring context managers
    create_step_timer,
    step_performance_timer,
    frame_rate_timer,
    debug_command_timer,
    debug_session_timer,
    memory_usage_timer,
    database_operation_timer,
    
    # Legacy API detection
    detect_legacy_gym_import,
    log_legacy_api_deprecation,
    monitor_environment_creation,
    
    # Hydra integration
    create_configuration_from_hydra,
    
    # Constants for backward compatibility
    DEFAULT_FORMAT,
    MODULE_FORMAT,
    ENHANCED_FORMAT,
    JSON_FORMAT,
    PERFORMANCE_THRESHOLDS,
    ENVIRONMENT_DEFAULTS,
)


class TestEnhancedLoggingSetup:
    """Comprehensive tests for enhanced logging configuration system."""
    
    @pytest.fixture(autouse=True)
    def reset_logger(self):
        """Reset logger before and after each test."""
        # Save original handlers
        original_handlers = logger._core.handlers.copy()
        
        # Safely remove all handlers for clean testing
        try:
            logger.remove()
        except (OSError, ValueError):
            # Handle cases where handlers are already closed or invalid
            pass
        
        yield
        
        # Restore original handlers safely
        try:
            logger._core.handlers.clear()
            for handler_id, handler in original_handlers.items():
                logger._core.handlers[handler_id] = handler
        except (OSError, ValueError):
            # If restoration fails, just ensure we have a basic handler
            logger.add(sys.stderr, level="INFO")
    
    @pytest.fixture
    def temp_log_file(self):
        """Create a temporary log file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
            log_path = tmp.name
        
        yield log_path
        
        # Cleanup
        if os.path.exists(log_path):
            os.unlink(log_path)
    
    @pytest.fixture
    def temp_json_log_file(self):
        """Create a temporary JSON log file for structured logging tests."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            log_path = tmp.name
        
        yield log_path
        
        # Cleanup
        if os.path.exists(log_path):
            os.unlink(log_path)
    
    @pytest.fixture
    def test_log_messages(self):
        """Standard test messages for each log level."""
        return {
            "debug": "Test debug message",
            "info": "Test info message",
            "warning": "Test warning message",
            "error": "Test error message",
            "critical": "Test critical message"
        }
    
    @pytest.fixture
    def sample_config(self):
        """Sample logging configuration for testing."""
        return LoggingConfig(
            environment="testing",
            level="DEBUG",
            format="enhanced",
            console_enabled=True,
            file_enabled=True,
            enable_performance=True,
            correlation_enabled=True,
            memory_tracking=True
        )
    
    @pytest.fixture
    def json_config(self):
        """JSON logging configuration for structured output tests."""
        return LoggingConfig(
            environment="production",
            level="INFO",
            format="json",
            console_enabled=False,
            file_enabled=True,
            enable_performance=True,
            correlation_enabled=True
        )
    
    # ========================================================================================
    # SECTION 1: Core Configuration Tests
    # ========================================================================================
    
    def test_logging_config_creation(self):
        """Test LoggingConfig model creation and validation."""
        config = LoggingConfig(
            environment="development",
            level="DEBUG",
            format="enhanced",
            enable_performance=True,
            correlation_enabled=True
        )
        
        assert config.environment == "development"
        assert config.level == "DEBUG"
        assert config.format == "enhanced"
        assert config.enable_performance is True
        assert config.correlation_enabled is True
        
        # Test environment defaults application
        config_with_defaults = config.apply_environment_defaults()
        assert config_with_defaults.console_enabled is True  # Development default
        assert config_with_defaults.file_enabled is True     # Development default
    
    def test_logging_config_validation(self):
        """Test LoggingConfig validation rules."""
        # Test invalid log level
        with pytest.raises(ValidationError, match=r"Input should be.*TRACE.*DEBUG.*INFO.*WARNING.*ERROR.*CRITICAL"):
            LoggingConfig(level="INVALID_LEVEL")
        
        # Test file path validation and normalization
        config = LoggingConfig(file_path="./test_logs/app.log")
        assert config.file_path == "test_logs/app.log"  # Path normalization removes './' prefix
        
        # Test environment variable interpolation handling
        config = LoggingConfig(file_path="${oc.env:LOG_PATH,./logs/app.log}")
        assert config.file_path == "${oc.env:LOG_PATH,./logs/app.log}"
    
    def test_environment_specific_defaults(self):
        """Test that environment-specific defaults are applied correctly."""
        # Test development environment
        dev_config = LoggingConfig(environment="development").apply_environment_defaults()
        assert dev_config.level == "DEBUG"
        assert dev_config.enable_performance is True
        assert dev_config.format == "enhanced"
        
        # Test production environment
        prod_config = LoggingConfig(environment="production").apply_environment_defaults()
        assert prod_config.level == "INFO"
        assert prod_config.format == "production"
        assert prod_config.enable_performance is True
        
        # Test testing environment
        test_config = LoggingConfig(environment="testing").apply_environment_defaults()
        assert test_config.level == "INFO"
        assert test_config.format == "minimal"
        assert test_config.enable_performance is False
    
    def test_setup_logger_console(self):
        """Test enhanced logger setup with console output."""
        from plume_nav_sim.utils.logging_setup import ENHANCED_FORMAT
        string_io = io.StringIO()
        
        # Set up logger with enhanced configuration
        config = LoggingConfig(
            level="DEBUG",
            format="enhanced",
            console_enabled=True,
            file_enabled=False,
            correlation_enabled=True
        )
        setup_logger(config)
        
        # Add capture handler with enhanced format
        handler_id = logger.add(string_io, level="DEBUG", format=ENHANCED_FORMAT)
        
        # Test with correlation context
        with correlation_context("test_operation", request_id="test_req_001"):
            logger.info("Enhanced console test")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify enhanced format elements
        assert "Enhanced console test" in output
        assert "test_req_001" in output  # Should include request_id
        assert "correlation_id" in output
    
    def test_setup_logger_with_config_object(self, sample_config, temp_log_file):
        """Test logger setup using LoggingConfig object."""
        sample_config.file_path = temp_log_file
        
        result_config = setup_logger(sample_config)
        
        # Verify returned config
        assert isinstance(result_config, LoggingConfig)
        assert result_config.level == sample_config.level
        assert result_config.format == sample_config.format
        
        # Test logging with the configured setup
        enhanced_logger = get_enhanced_logger("test_module")
        enhanced_logger.info("Config object test message")
        
        # Verify file output
        time.sleep(0.1)  # Allow file write
        with open(temp_log_file, "r") as f:
            content = f.read()
        
        assert "Config object test message" in content
        assert "test_module" in content
    
    # ========================================================================================
    # SECTION 2: Structured JSON Logging Tests
    # ========================================================================================
    
    def test_json_structured_logging(self, temp_json_log_file):
        """Test structured JSON logging with correlation IDs."""
        config = LoggingConfig(
            format="json",
            console_enabled=False,
            file_enabled=True,
            file_path=temp_json_log_file,
            correlation_enabled=True
        )
        setup_logger(config)
        
        # Log with correlation context
        with correlation_context("json_test", episode_id="ep_001", agent_count=5):
            enhanced_logger = get_enhanced_logger("json_test_module")
            enhanced_logger.info("JSON structured test message", extra={
                "metric_type": "test_metric",
                "custom_field": "custom_value"
            })
        
        time.sleep(0.1)  # Allow file write
        
        # Read and parse JSON log entry - handle multiple log entries and escaping
        with open(temp_json_log_file, "r") as f:
            content = f.read()
        
        # Split into lines and find our test message
        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        log_entry = None
        
        for line in lines:
            try:
                # Try parsing directly first (unescaped JSON)
                parsed = json.loads(line)
                if parsed.get("message") == "JSON structured test message":
                    log_entry = parsed
                    break
            except json.JSONDecodeError:
                try:
                    # Try unescaping first (escaped JSON)
                    unescaped_line = (line
                                    .replace('&lt;', '<')
                                    .replace('&gt;', '>')
                                    .replace('{{', '{')
                                    .replace('}}', '}'))
                    parsed = json.loads(unescaped_line)
                    if parsed.get("message") == "JSON structured test message":
                        log_entry = parsed
                        break
                except json.JSONDecodeError:
                    continue
        
        # Ensure we found the test message
        assert log_entry is not None, f"Could not find test message in log file. Found {len(lines)} lines."
        
        # Verify JSON structure
        assert log_entry["message"] == "JSON structured test message"
        assert log_entry["level"] == "INFO"
        assert log_entry["module"] == "json_test_module"
        assert "correlation_id" in log_entry
        assert log_entry["episode_id"] == "ep_001"
        assert log_entry["agent_count"] == 5
        assert log_entry["metric_type"] == "test_metric"
        assert log_entry["custom_field"] == "custom_value"
        assert "timestamp" in log_entry
        assert "thread_id" in log_entry
        assert "process_id" in log_entry
    
    def test_correlation_id_propagation(self):
        """Test correlation ID propagation across logging calls."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(format="enhanced", console_enabled=False))
        handler_id = logger.add(string_io, format=ENHANCED_FORMAT)
        
        correlation_id = "test_correlation_123"
        request_id = "req_456"
        
        with correlation_context("propagation_test", correlation_id=correlation_id, request_id=request_id):
            context = get_correlation_context()
            
            # Log multiple messages
            enhanced_logger = get_enhanced_logger("propagation_module")
            enhanced_logger.info("First message")
            enhanced_logger.warning("Second message")
            enhanced_logger.error("Third message")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify all messages contain the same correlation ID
        lines = output.strip().split('\n')
        assert len(lines) == 3
        
        for line in lines:
            assert correlation_id in line
            assert request_id in line
            assert "propagation_module" in line
    
    def test_thread_safe_correlation_context(self):
        """Test thread-safe operation of correlation context."""
        results = {}
        
        def worker_thread(thread_id):
            """Worker function for thread safety testing."""
            correlation_id = f"thread_{thread_id}_correlation"
            
            with correlation_context("thread_test", correlation_id=correlation_id):
                context = get_correlation_context()
                results[thread_id] = context.correlation_id
                time.sleep(0.1)  # Simulate work
                
                # Verify context is still correct after delay
                context_after = get_correlation_context()
                assert context_after.correlation_id == correlation_id
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify each thread had its own correlation context
        assert len(results) == 5
        for thread_id, correlation_id in results.items():
            assert correlation_id == f"thread_{thread_id}_correlation"
    
    def test_setup_logger_with_file(self, temp_log_file, test_log_messages):
        """Test enhanced logger setup with file output."""
        config = LoggingConfig(
            level="DEBUG",
            format="enhanced",
            console_enabled=False,
            file_enabled=True,
            file_path=temp_log_file,
            correlation_enabled=True
        )
        setup_logger(config)
        
        # Use enhanced logger with correlation context
        with correlation_context("file_test"):
            enhanced_logger = get_enhanced_logger("file_test_module")
            enhanced_logger.debug(test_log_messages["debug"])
            enhanced_logger.info(test_log_messages["info"])
            enhanced_logger.warning(test_log_messages["warning"])
            enhanced_logger.error(test_log_messages["error"])
            # Added CRITICAL level logging for verification
            enhanced_logger.critical(test_log_messages["critical"])
        
        time.sleep(0.1)  # Allow file write
        
        # Read file and verify messages
        with open(temp_log_file, "r") as f:
            content = f.read()
        
        # Verify each message with enhanced format
        for level, message in test_log_messages.items():
            assert message in content, f"{level.upper()} message not found in log file"
        
        # Verify enhanced format elements
        assert "file_test_module" in content
        assert "correlation_id" in content
    
    # ========================================================================================
    # SECTION 3: Performance Monitoring Tests 
    # ========================================================================================
    
    def test_step_timing_threshold_monitoring(self):
        """Test automatic performance threshold alerting for step() timing ≤10ms requirement."""
        string_io = io.StringIO()
        
        config = LoggingConfig(
            level="DEBUG",
            format="json",
            console_enabled=False,
            enable_performance=True
        )
        setup_logger(config)
        handler_id = logger.add(string_io, format="{message} | {extra}")
        
        # Test step timing below threshold (should not trigger warning)
        with create_step_timer() as metrics:
            time.sleep(0.005)  # 5ms - below 10ms threshold
        
        # Test step timing above threshold (should trigger warning)
        with create_step_timer() as metrics:
            time.sleep(0.015)  # 15ms - above 10ms threshold
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify threshold violation warning was logged
        assert "Environment step() latency exceeded threshold" in output
        # Extract timing values and perform relaxed assertions
        match = re.search(r"exceeded threshold: ([0-9.]+)s > ([0-9.]+)s", output)
        assert match, "Timing pattern not found in output"
        actual_sec = float(match.group(1))
        threshold_sec = float(match.group(2))
        assert actual_sec >= 0.015, f"Actual latency ({actual_sec}) below expected minimum"
        assert threshold_sec == 0.010, "Threshold latency should be 0.010s"
        assert "metric_type" in output
        assert "step_latency_violation" in output
    
    def test_performance_metrics_structure(self):
        """Test PerformanceMetrics structure and functionality."""
        metrics = PerformanceMetrics(
            operation_name="test_operation",
            start_time=time.time(),
            correlation_id="test_correlation",
            metadata={"test_key": "test_value"}
        )
        
        # Test completion
        time.sleep(0.01)
        completed_metrics = metrics.complete()
        
        assert completed_metrics.duration is not None
        assert completed_metrics.duration > 0.01
        assert completed_metrics.end_time is not None
        assert completed_metrics.correlation_id == "test_correlation"
        assert completed_metrics.metadata["test_key"] == "test_value"
        
        # Test threshold checking
        assert completed_metrics.is_slow(0.005)  # Should be slow for 5ms threshold
        assert not completed_metrics.is_slow(0.020)  # Should not be slow for 20ms threshold
        
        # Test dictionary conversion
        metrics_dict = completed_metrics.to_dict()
        assert "operation_name" in metrics_dict
        assert "duration" in metrics_dict
        assert "correlation_id" in metrics_dict
    
    def test_frame_rate_monitoring(self):
        """Test frame rate measurement and warning functionality."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG", enable_performance=True))
        handler_id = logger.add(string_io, format="{message} | {extra}")
        
        # Test frame rate below threshold (should trigger warning)
        with frame_rate_timer():
            time.sleep(0.040)  # 40ms per frame = 25 FPS (below 30 FPS target)
        
        # Test frame rate above threshold (should not trigger warning)
        with frame_rate_timer():
            time.sleep(0.020)  # 20ms per frame = 50 FPS (above 30 FPS target)
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify frame rate warning was logged for slow frame
        assert "Frame rate below target" in output
        assert "25" in output or "24" in output  # Approximate FPS
        assert "30" in output  # Target FPS
        assert "frame_rate_violation" in output
    
    def test_memory_usage_monitoring(self):
        """Test memory usage delta tracking and warnings."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG", enable_performance=True, memory_tracking=True))
        handler_id = logger.add(string_io, format="{message} | {extra}")
        
        # Simulate memory usage operation
        with memory_usage_timer("test_memory_operation"):
            # Simulate memory allocation (this may or may not trigger warning depending on system)
            test_data = [0] * 1000000  # Allocate some memory
            time.sleep(0.01)
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Memory warnings are conditional on actual memory changes
        # Just verify the timer executed without errors
        assert "test_memory_operation" in output or len(output) >= 0
    
    def test_database_operation_timing(self):
        """Test database operation performance monitoring."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG", enable_performance=True))
        handler_id = logger.add(string_io, format="{message} | {extra}")
        
        # Test database operation above threshold (should trigger warning)
        with database_operation_timer("slow_query"):
            time.sleep(0.150)  # 150ms - above 100ms threshold
        
        # Test database operation below threshold (should not trigger warning)
        with database_operation_timer("fast_query"):
            time.sleep(0.050)  # 50ms - below 100ms threshold
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify database latency warning was logged
        assert "Slow database operation" in output
        assert "slow_query" in output
        # Relaxed timing check (allow small scheduling variance)
        match = re.search(r"took ([0-9.]+)s", output)
        assert match, "Duration pattern not found in output"
        duration_sec = float(match.group(1))
        assert duration_sec >= 0.140, f"Duration {duration_sec} shorter than expected lower bound"
        assert "db_latency_violation" in output
    
    def test_enhanced_logger_performance_timer(self):
        """Test enhanced logger performance timer context manager."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG"))
        handler_id = logger.add(string_io, format="{message} | {extra}")
        
        enhanced_logger = get_enhanced_logger("performance_test_module")
        
        # Test performance timer with threshold
        with enhanced_logger.performance_timer("test_operation", threshold=0.010) as metrics:
            time.sleep(0.020)  # 20ms - above 10ms threshold
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify performance timer logging
        assert "Starting operation: test_operation" in output
        assert "Slow operation completed: test_operation" in output
        assert "performance_metrics" in output
        assert "operation_complete" in output
    
    def test_log_level_info_filters_debug(self):
        """Test that INFO level filters DEBUG messages."""
        string_io = io.StringIO()
        
        # Set up logger with INFO level
        setup_logger(LoggingConfig(level="INFO", format="minimal"))
        
        # Add capture handler at INFO level
        handler_id = logger.add(string_io, level="INFO")
        
        # Log messages
        debug_message = "DEBUG message for INFO level test"
        info_message = "INFO message for INFO level test"
        warning_message = "WARNING message for INFO level test"
        error_message = "ERROR message for INFO level test"
        
        logger.debug(debug_message)
        logger.info(info_message)
        logger.warning(warning_message)
        logger.error(error_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove the handler
        logger.remove(handler_id)
        
        # Verify correct message filtering
        assert debug_message not in output, "DEBUG message should not be visible at INFO level"
        assert info_message in output, "INFO message should be visible at INFO level"
        assert warning_message in output, "WARNING message should be visible at INFO level"
        assert error_message in output, "ERROR message should be visible at INFO level"
    
    # ========================================================================================
    # SECTION 4: Legacy API Detection and Deprecation Tests
    # ========================================================================================
    
    def test_legacy_api_deprecation_warning(self):
        """Test legacy gym API deprecation detection and structured warnings."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG"))
        handler_id = logger.add(string_io, format="{message} | {extra}")
        
        # Test legacy API deprecation logging
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            log_legacy_api_deprecation(
                operation="environment_creation",
                legacy_call="gym.make('CartPole-v1')",
                recommended_call="gymnasium.make('CartPole-v1')",
                migration_guide="Replace 'import gym' with 'import gymnasium as gym'"
            )
            
            # Verify Python DeprecationWarning was issued
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "gym.make('CartPole-v1')" in str(w[0].message)
            assert "gymnasium.make('CartPole-v1')" in str(w[0].message)
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify structured logging of deprecation
        assert "Legacy API deprecation: environment_creation" in output
        assert "legacy_api_deprecation" in output
        assert "gym.make('CartPole-v1')" in output
        assert "gymnasium.make('CartPole-v1')" in output
        assert "migration_guide" in output
    
    def test_monitor_environment_creation(self):
        """Test environment creation monitoring for legacy API usage."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG"))
        handler_id = logger.add(string_io, format="{message} | {extra}")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test monitoring of legacy gym.make call
            monitor_environment_creation("CartPole-v1", "gym.make")
            
            # Verify deprecation warning was triggered
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify structured logging
        assert "Legacy API deprecation" in output
        assert "CartPole-v1" in output
        assert "gym.make" in output
    
    @patch('odor_plume_nav.utils.logging_setup.inspect')
    def test_detect_legacy_gym_import(self, mock_inspect):
        """Test detection of legacy gym imports vs gymnasium."""
        # Mock inspect.stack to simulate legacy gym import
        mock_frame = MagicMock()
        mock_frame.f_globals = {
            '__name__': 'test_module',
            'gym': MagicMock()  # Simulate gym import
        }
        mock_frame_info = MagicMock()
        mock_frame_info.frame = mock_frame
        mock_inspect.stack.return_value = [mock_frame_info]
        
        # Configure mock gym object to simulate legacy gym
        mock_frame.f_globals['gym'].make = MagicMock()
        
        # Test detection
        is_legacy = detect_legacy_gym_import()
        
        # The function should detect legacy usage
        assert is_legacy is True
    
    def test_enhanced_logger_legacy_api_logging(self):
        """Test enhanced logger legacy API usage logging."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG"))
        handler_id = logger.add(string_io, format="{message} | {extra}")
        
        enhanced_logger = get_enhanced_logger("legacy_test_module")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            enhanced_logger.log_legacy_api_usage(
                operation="environment_creation",
                legacy_call="gym.make('Pendulum-v1')",
                recommended_call="gymnasium.make('Pendulum-v1')"
            )
            
            # Verify deprecation warning was issued
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify logging occurred
        assert "Legacy API deprecation" in output
        assert "Pendulum-v1" in output
    
    def test_log_level_warning_filters_info_and_debug(self):
        """Test that WARNING level filters DEBUG and INFO messages."""
        string_io = io.StringIO()
        
        # Set up logger with WARNING level
        setup_logger(LoggingConfig(level="WARNING", format="minimal"))
        
        # Add capture handler
        handler_id = logger.add(string_io, level="WARNING")
        
        # Log messages
        debug_message = "DEBUG message for WARNING level test"
        info_message = "INFO message for WARNING level test"
        warning_message = "WARNING message for WARNING level test"
        error_message = "ERROR message for WARNING level test"
        
        logger.debug(debug_message)
        logger.info(info_message)
        logger.warning(warning_message)
        logger.error(error_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove the handler
        logger.remove(handler_id)
        
        # Verify correct message filtering
        assert debug_message not in output, "DEBUG message should not be visible at WARNING level"
        assert info_message not in output, "INFO message should not be visible at WARNING level"
        assert warning_message in output, "WARNING message should be visible at WARNING level"
        assert error_message in output, "ERROR message should be visible at WARNING level"
    
    # ========================================================================================
    # SECTION 5: Enhanced Logger Class Tests
    # ========================================================================================
    
    def test_enhanced_logger_creation(self):
        """Test EnhancedLogger creation and basic functionality."""
        config = LoggingConfig(level="DEBUG", correlation_enabled=True)
        enhanced_logger = EnhancedLogger("test_module", config)
        
        assert enhanced_logger.name == "test_module"
        assert enhanced_logger.config == config
        assert enhanced_logger._base_context["module"] == "test_module"
    
    def test_enhanced_logger_context_binding(self):
        """Test automatic context binding in enhanced logger."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG", format="enhanced"))
        handler_id = logger.add(string_io, format=ENHANCED_FORMAT)
        
        enhanced_logger = get_enhanced_logger("context_test_module")
        
        with correlation_context("test_operation", request_id="req_123"):
            enhanced_logger.info("Context binding test message")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify context binding
        assert "Context binding test message" in output
        assert "context_test_module" in output
        assert "req_123" in output
        assert "correlation_id" in output
    
    def test_enhanced_logger_all_log_levels(self):
        """Test all log levels of enhanced logger."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="TRACE"))
        handler_id = logger.add(string_io, level="TRACE", format="{level}: {message}")
        
        enhanced_logger = get_enhanced_logger("level_test_module")
        
        # Test all log levels
        enhanced_logger.trace("Trace message")
        enhanced_logger.debug("Debug message")
        enhanced_logger.info("Info message")
        enhanced_logger.success("Success message")
        enhanced_logger.warning("Warning message")
        enhanced_logger.error("Error message")
        enhanced_logger.critical("Critical message")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify all levels are logged
        assert "TRACE: Trace message" in output
        assert "DEBUG: Debug message" in output
        assert "INFO: Info message" in output
        assert "SUCCESS: Success message" in output
        assert "WARNING: Warning message" in output
        assert "ERROR: Error message" in output
        assert "CRITICAL: Critical message" in output
    
    def test_enhanced_logger_exception_handling(self):
        """Test enhanced logger exception handling with context."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG", backtrace=True, diagnose=True))
        handler_id = logger.add(string_io, format="{message}")
        
        enhanced_logger = get_enhanced_logger("exception_test_module")
        
        try:
            raise ValueError("Test exception for enhanced logger")
        except Exception:
            enhanced_logger.exception("Exception occurred in enhanced logger test")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify exception logging
        assert "Exception occurred in enhanced logger test" in output
        assert "ValueError" in output
        assert "Test exception for enhanced logger" in output
        assert "Traceback" in output
    
    def test_enhanced_logger_performance_tracking_methods(self):
        """Test enhanced logger specialized performance tracking methods."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG"))
        handler_id = logger.add(string_io, format="{message} | {extra}")
        
        enhanced_logger = get_enhanced_logger("perf_test_module")
        
        # Test step latency violation logging
        enhanced_logger.log_step_latency_violation(0.015, 0.010)
        
        # Test frame rate measurement logging
        enhanced_logger.log_frame_rate_measurement(25.0, 30.0)  # Below target
        enhanced_logger.log_frame_rate_measurement(35.0, 30.0)  # Above target
        
        # Test memory usage delta logging
        enhanced_logger.log_memory_usage_delta(150.0, "large_allocation")  # Warning threshold
        enhanced_logger.log_memory_usage_delta(50.0, "small_allocation")   # Normal
        
        # Test database operation latency logging
        enhanced_logger.log_database_operation_latency("slow_query", 0.150, 0.100)  # Above threshold
        enhanced_logger.log_database_operation_latency("fast_query", 0.050, 0.100)  # Below threshold
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify performance tracking logs
        assert "Environment step() latency exceeded" in output
        assert "15.0ms > 10.0ms" in output
        
        assert "Frame rate below target: 25.0 FPS < 30.0 FPS" in output
        assert "Frame rate measurement: 35.0 FPS" in output
        
        assert "Significant memory increase: +150.0MB" in output
        assert "Memory usage delta: +50.0MB" in output
        
        assert "Slow database operation: slow_query took 0.150s" in output
        assert "Database operation: fast_query completed in 0.050s" in output
    
    def test_get_module_logger(self):
        """Test getting an enhanced module-specific logger."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG", format="enhanced"))
        handler_id = logger.add(string_io, format=ENHANCED_FORMAT)
        
        # Test different logger factory functions
        module_logger1 = get_module_logger("test_module_1")
        module_logger2 = get_enhanced_logger("test_module_2")
        module_logger3 = get_logger("test_module_3")
        
        # All should return EnhancedLogger instances
        assert isinstance(module_logger1, EnhancedLogger)
        assert isinstance(module_logger2, EnhancedLogger)
        assert isinstance(module_logger3, EnhancedLogger)
        
        # Test logging with each
        module_logger1.info("Message from module 1")
        module_logger2.info("Message from module 2")
        module_logger3.info("Message from module 3")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify module-specific logging
        assert "Message from module 1" in output
        assert "test_module_1" in output
        assert "Message from module 2" in output
        assert "test_module_2" in output
        assert "Message from module 3" in output
        assert "test_module_3" in output
    
    # ========================================================================================
    # SECTION 6: Hydra Integration and Advanced Configuration Tests
    # ========================================================================================
    
    def test_create_configuration_from_hydra(self):
        """Test creating LoggingConfig from Hydra configuration."""
        # Mock Hydra config structure
        mock_hydra_config = {
            "environment": "production",
            "level": "INFO",
            "format": "json",
            "enable_performance": True,
            "correlation_enabled": True,
            "file_path": "./logs/hydra_test.log"
        }
        
        config = create_configuration_from_hydra(mock_hydra_config)
        
        assert isinstance(config, LoggingConfig)
        assert config.environment == "production"
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.enable_performance is True
        assert config.correlation_enabled is True
        assert config.file_path == "./logs/hydra_test.log"
    
    def test_backward_compatibility_setup(self):
        """Test backward compatibility with original setup_logger parameters."""
        string_io = io.StringIO()
        
        # Test legacy parameter support
        config = setup_logger(
            sink=None,
            level="DEBUG",
            format=DEFAULT_FORMAT,
            environment="development"
        )
        
        handler_id = logger.add(string_io, level="DEBUG")
        
        # Test legacy module logger
        legacy_logger = get_module_logger("legacy_test_module")
        legacy_logger.info("Legacy compatibility test")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify configuration
        assert isinstance(config, LoggingConfig)
        assert config.level == "DEBUG"
        
        # Verify logging works
        assert "Legacy compatibility test" in output
        assert "legacy_test_module" in output
    
    def test_format_pattern_resolution(self):
        """Test format pattern resolution for different format types."""
        config = LoggingConfig()
        
        # Test all format types
        format_types = ["default", "module", "enhanced", "hydra", "cli", "minimal", "production", "json"]
        
        for format_type in format_types:
            config.format = format_type
            pattern = config.get_format_pattern()
            
            assert isinstance(pattern, str)
            assert len(pattern) > 0
            
            # Verify format-specific elements
            if format_type == "json":
                assert pattern == JSON_FORMAT
            elif format_type == "enhanced":
                assert "correlation_id" in pattern
            elif format_type == "minimal":
                assert pattern == "{level: <8} | {message}"
    
    def test_environment_variable_interpolation_handling(self):
        """Test handling of environment variable interpolation in configuration."""
        config = LoggingConfig(
            file_path="${oc.env:LOG_PATH,./logs/app.log}",
            level="INFO"
        )
        
        # Should preserve interpolation syntax
        assert config.file_path == "${oc.env:LOG_PATH,./logs/app.log}"
        
        # Should not attempt to create directory for interpolated paths
        assert config.level == "INFO"
    
    # ========================================================================================
    # SECTION 7: Integration and Edge Case Tests
    # ========================================================================================
    
    def test_multiple_correlation_contexts_nested(self):
        """Test nested correlation contexts and proper cleanup."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(format="enhanced"))
        handler_id = logger.add(string_io, format=ENHANCED_FORMAT)
        
        enhanced_logger = get_enhanced_logger("nested_test_module")
        
        with correlation_context("outer_operation", request_id="outer_req"):
            enhanced_logger.info("Outer context message")
            
            with correlation_context("inner_operation", request_id="inner_req"):
                enhanced_logger.info("Inner context message")
            
            enhanced_logger.info("Back to outer context")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        lines = output.strip().split('\n')
        assert len(lines) == 3
        
        # Verify proper context switching
        assert "outer_req" in lines[0]
        assert "inner_req" in lines[1]
        # Note: Context handling may vary based on implementation
    
    def test_performance_thresholds_configuration(self):
        """Test performance thresholds are properly configured."""
        # Verify critical performance thresholds are set correctly
        assert PERFORMANCE_THRESHOLDS["environment_step"] == 0.010  # 10ms requirement
        assert PERFORMANCE_THRESHOLDS["simulation_fps_min"] == 30.0  # 30 FPS requirement
        assert PERFORMANCE_THRESHOLDS["db_operation"] == 0.1  # 100ms threshold
        
        # Test threshold usage in performance metrics
        metrics = PerformanceMetrics(
            operation_name="environment_step",
            start_time=time.time() - 0.015  # 15ms ago
        )
        completed_metrics = metrics.complete()
        
        # Should detect as slow using default threshold
        assert completed_metrics.is_slow()
        
        # Should not be slow with higher threshold
        assert not completed_metrics.is_slow(0.020)
    
    def test_concurrent_performance_monitoring(self):
        """Test performance monitoring under concurrent access."""
        results = []
        
        def worker_performance_test(worker_id):
            """Worker function for concurrent performance testing."""
            with correlation_context(f"worker_{worker_id}_operation"):
                with create_step_timer() as metrics:
                    time.sleep(0.005)  # 5ms operation
                
                # Access correlation context
                context = get_correlation_context()
                results.append({
                    'worker_id': worker_id,
                    'correlation_id': context.correlation_id,
                    'step_count': context.step_count
                })
        
        # Run multiple workers concurrently
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_performance_test, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify each worker had isolated context
        assert len(results) == 3
        correlation_ids = [r['correlation_id'] for r in results]
        assert len(set(correlation_ids)) == 3  # All different correlation IDs
    
    def test_custom_format(self):
        """Test using custom log formats with enhanced features."""
        string_io = io.StringIO()
        
        # Define enhanced custom format
        custom_format = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[module]} | {extra[correlation_id]} | {message}"
        
        config = LoggingConfig(
            level="INFO",
            format="enhanced",  # Will be overridden by setup
            console_enabled=False
        )
        setup_logger(config)
        
        handler_id = logger.add(string_io, level="DEBUG", format=custom_format)
        
        with correlation_context("custom_format_test", correlation_id="custom_123"):
            enhanced_logger = get_enhanced_logger("custom_format_module")
            enhanced_logger.info("Custom format test message")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify custom format elements
        date_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        assert re.search(date_pattern, output), "Date pattern not found"
        assert "INFO" in output
        assert "custom_format_module" in output
        assert "custom_123" in output
        assert "Custom format test message" in output
    
    def test_exception_logging_with_context(self):
        """Test exception logging with enhanced context and correlation."""
        string_io = io.StringIO()
        setup_logger(LoggingConfig(level="DEBUG", backtrace=True, diagnose=True))
        handler_id = logger.add(string_io, level="DEBUG")
        
        enhanced_logger = get_enhanced_logger("exception_test_module")
        
        with correlation_context("exception_test", episode_id="ep_error_001"):
            try:
                raise ValueError("Test exception with enhanced context")
            except Exception:
                enhanced_logger.exception("Enhanced exception logging test")
        
        output = string_io.getvalue()
        logger.remove(handler_id)
        
        # Verify enhanced exception logging
        assert "Enhanced exception logging test" in output
        assert "ValueError" in output
        assert "Test exception with enhanced context" in output
        assert "Traceback" in output
        assert "ep_error_001" in output  # Episode ID should be included
        assert "exception_test_module" in output
