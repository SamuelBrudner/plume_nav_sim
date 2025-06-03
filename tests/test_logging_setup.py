"""Tests for the logging configuration system."""

import pytest
import sys
import os
import io
import tempfile
from pathlib import Path
import re
from contextlib import suppress

from loguru import logger

# Import the module to test
from odor_plume_nav.utils.logging_setup import (
    setup_logger,
    get_module_logger,
    DEFAULT_FORMAT,
    MODULE_FORMAT
)


class TestLoggingSetup:
    """Tests for logging setup functionality."""
    
    @pytest.fixture(autouse=True)
    def reset_logger(self):
        """Reset logger before and after each test."""
        # Save original handlers
        original_handlers = logger._core.handlers.copy()
        
        # Remove all handlers for clean testing
        logger.remove()
        
        yield
        
        # Restore original handlers
        logger._core.handlers.clear()
        for handler_id, handler in original_handlers.items():
            logger._core.handlers[handler_id] = handler
    
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
    def test_log_messages(self):
        """Standard test messages for each log level."""
        return {
            "debug": "Test debug message",
            "info": "Test info message",
            "warning": "Test warning message",
            "error": "Test error message",
            "critical": "Test critical message"
        }
    
    def test_setup_logger_console(self):
        """Test setting up logger with console output."""
        # Set up a string IO to capture output
        string_io = io.StringIO()
        
        # Set up logger with our string_io as the sink instead of stderr
        setup_logger(sink=None, level="DEBUG")
        
        # Now add our capture handler AFTER setup_logger has run
        handler_id = logger.add(string_io, level="DEBUG")
        
        # Test messages dictionary for different log levels
        test_messages = {
            "debug": "Console debug test",
            "info": "Console info test",
            "warning": "Console warning test"
        }
        
        # Log messages at each level
        logger.debug(test_messages["debug"])
        logger.info(test_messages["info"])
        logger.warning(test_messages["warning"])
        
        # Get captured output
        output = string_io.getvalue()
        
        # Remove the handler we added
        logger.remove(handler_id)
        
        # Assert messages appear in output
        assert test_messages["debug"] in output, "Debug message not found in logs"
        assert test_messages["info"] in output, "Info message not found in logs"
        assert test_messages["warning"] in output, "Warning message not found in logs"
    
    def test_setup_logger_with_file(self, temp_log_file, test_log_messages):
        """Test setting up logger with file output."""
        # Set up logger with file sink
        setup_logger(sink=temp_log_file, level="DEBUG")
        
        # Log messages directly
        logger.debug(test_log_messages["debug"])
        logger.info(test_log_messages["info"])
        logger.warning(test_log_messages["warning"])
        logger.error(test_log_messages["error"])
        
        # Ensure writes are flushed
        import time
import yaml
        time.sleep(0.1)
        
        # Read file and verify messages
        with open(temp_log_file, "r") as f:
            content = f.read()
            
        # Verify each message individually
        assert test_log_messages["debug"] in content, "Debug message not found in log file"
        assert test_log_messages["info"] in content, "Info message not found in log file"
        assert test_log_messages["warning"] in content, "Warning message not found in log file"
        assert test_log_messages["error"] in content, "Error message not found in log file"
    
    def test_log_level_info_filters_debug(self):
        """Test that INFO level filters DEBUG messages."""
        string_io = io.StringIO()
        
        # Set up logger with INFO level
        setup_logger(sink=None, level="INFO")
        
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
    
    def test_log_level_warning_filters_info_and_debug(self):
        """Test that WARNING level filters DEBUG and INFO messages."""
        string_io = io.StringIO()
        
        # Set up logger with WARNING level
        setup_logger(sink=None, level="WARNING")
        
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
    
    def test_get_module_logger(self):
        """Test getting a module-specific logger."""
        # Module name to test
        module_name = "test_module"
        
        # Set up string IO capture
        string_io = io.StringIO()
        
        # Setup format that clearly shows module binding
        module_format = "{level}|module={extra[module]}|{message}"
        setup_logger(sink=None, level="DEBUG", format=module_format)
        
        # Add capture handler after setup
        handler_id = logger.add(string_io, level="DEBUG", format=module_format)
        
        # Get module logger and log a test message
        test_message = "Module logger test message"
        module_logger = get_module_logger(module_name)
        module_logger.info(test_message)
        
        # Get output and verify module name and message
        output = string_io.getvalue()
        
        # Remove the handler we added
        logger.remove(handler_id)
        
        # Assert module logger functionality
        assert test_message in output, "Module message not found in output"
        assert f"module={module_name}" in output, "Module name not found in output"
    
    def test_custom_format(self):
        """Test using a custom log format."""
        # Define a simple format without colors
        custom_format = "{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}"
        
        # Set up string IO capture
        string_io = io.StringIO()
        
        # Setup logger with custom format
        setup_logger(sink=None, level="INFO", format=custom_format)
        
        # Add capture handler after setup with the same format
        handler_id = logger.add(string_io, level="DEBUG", format=custom_format)
        
        # Log a test message
        test_message = "Custom format test message"
        logger.info(test_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove the handler we added
        logger.remove(handler_id)
        
        # Define pattern for date-time format and verify format
        date_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        expected_pattern = f"{date_pattern} - INFO - {test_message}"
        
        assert re.search(expected_pattern, output), "Format pattern not found in output"
        assert "<green>" not in output, "Default color codes should not be present"
        assert "<level>" not in output, "Default level markup should not be present"
    
    def test_exception_logging(self):
        """Test logging exceptions with traceback."""
        # Set up string IO capture
        string_io = io.StringIO()
        
        # Setup logger with backtrace enabled
        setup_logger(sink=None, level="DEBUG", backtrace=True, diagnose=True)
        
        # Add capture handler after setup
        handler_id = logger.add(string_io, level="DEBUG")
        
        # Test data
        error_message = "Test exception occurred"
        exception_details = "Test exception details"
        
        # Use contextlib.suppress to safely catch the exception
        # without letting pytest intercept it
        with suppress(ValueError):
            # Create and log an exception
            try:
                raise ValueError(exception_details)
            except Exception:
                # Log the caught exception
                logger.exception(error_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove the handler we added
        logger.remove(handler_id)
        
        # Verify exception logging details
        assert error_message in output, "Error message missing from exception log"
        assert "ValueError" in output, "Exception type missing from exception log"
        assert exception_details in output, "Exception details missing from exception log"
        assert "Traceback" in output, "Traceback missing from exception log"
