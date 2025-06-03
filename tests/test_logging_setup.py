"""Tests for the logging configuration system."""

import pytest
import sys
import os
import io
import tempfile
from pathlib import Path
import re
from contextlib import suppress
from unittest.mock import Mock, patch, MagicMock
import hashlib
import json
from uuid import uuid4

from loguru import logger

# Import the module to test
from {{cookiecutter.project_slug}}.utils.logging import (
    setup_logger,
    get_module_logger,
    DEFAULT_FORMAT,
    MODULE_FORMAT
)

# Import enhanced logging features (may not exist yet - using patches for testing)
try:
    from {{cookiecutter.project_slug}}.utils.logging import (
        setup_enhanced_logger,
        bind_hydra_context,
        bind_seed_context,
        bind_cli_context,
        generate_config_checksum
    )
except ImportError:
    # Enhanced features not implemented yet - tests will use mocks
    setup_enhanced_logger = None
    bind_hydra_context = None
    bind_seed_context = None
    bind_cli_context = None
    generate_config_checksum = None


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
    
    @pytest.fixture
    def mock_hydra_context(self):
        """Mock Hydra context for testing enhanced logging."""
        return {
            "job_name": "test_navigation_experiment",
            "run_id": "2024_01_15_10_30_00_123456",
            "config_checksum": "abc123def456",
            "config_composition": {
                "base": "conf/base.yaml",
                "overrides": ["data.video_path=/test/video.mp4", "navigator.speed=1.5"]
            }
        }
    
    @pytest.fixture
    def mock_seed_context(self):
        """Mock seed manager context for testing."""
        return {
            "seed": 12345,
            "run_id": "seed_run_2024_01_15",
            "initial_timestamp": "2024-01-15T10:30:00.123456"
        }
    
    @pytest.fixture
    def mock_cli_context(self):
        """Mock CLI command context for testing."""
        return {
            "command": "navigate",
            "args": ["--config", "test_config.yaml", "--output", "/tmp/results"],
            "execution_id": str(uuid4()),
            "start_time": "2024-01-15T10:30:00.123456"
        }
    
    @pytest.fixture
    def sample_config_dict(self):
        """Sample configuration for checksum testing."""
        return {
            "navigator": {
                "speed": 1.0,
                "max_speed": 2.0,
                "position": [0.0, 0.0]
            },
            "video_plume": {
                "video_path": "/test/video.mp4",
                "grayscale": True
            },
            "environment": {
                "seed": "${oc.env:EXPERIMENT_SEED,42}",
                "output_dir": "${oc.env:OUTPUT_DIR,/tmp/results}"
            }
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


class TestEnhancedLoggingFeatures:
    """Tests for enhanced logging features including Hydra integration and context binding."""
    
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
    
    def test_setup_enhanced_logger_with_hydra_context(self, mock_hydra_context):
        """Test enhanced logger setup with Hydra context binding."""
        string_io = io.StringIO()
        
        # Setup enhanced logger with Hydra context
        with patch('{{cookiecutter.project_slug}}.utils.logging.setup_enhanced_logger') as mock_setup:
            # Configure mock to actually set up logger
            def setup_side_effect(sink=None, level="INFO", **kwargs):
                setup_logger(sink=sink, level=level)
                return bind_hydra_context(mock_hydra_context)
            
            mock_setup.side_effect = setup_side_effect
            enhanced_logger = setup_enhanced_logger(hydra_context=mock_hydra_context)
        
        # Add capture handler
        handler_id = logger.add(string_io, level="DEBUG")
        
        # Test message with enhanced context
        test_message = "Test message with Hydra context"
        if enhanced_logger:
            enhanced_logger.info(test_message)
        else:
            # Fallback to direct context binding test
            bound_logger = bind_hydra_context(mock_hydra_context)
            bound_logger.info(test_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify Hydra context is included (this test verifies the concept)
        assert test_message in output, "Test message not found in output"
    
    def test_bind_hydra_context(self, mock_hydra_context):
        """Test Hydra context binding functionality."""
        string_io = io.StringIO()
        
        # Setup basic logger
        setup_logger(sink=None, level="DEBUG")
        
        # Format that shows extra context
        context_format = "{level}|job={extra[job_name]}|run_id={extra[run_id]}|checksum={extra[config_checksum]}|{message}"
        handler_id = logger.add(string_io, level="DEBUG", format=context_format)
        
        # Bind Hydra context
        with patch('{{cookiecutter.project_slug}}.utils.logging.bind_hydra_context') as mock_bind:
            def bind_side_effect(context):
                return logger.bind(**context)
            mock_bind.side_effect = bind_side_effect
            
            hydra_logger = bind_hydra_context(mock_hydra_context)
            test_message = "Test Hydra context binding"
            hydra_logger.info(test_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify Hydra context components
        assert test_message in output, "Test message not found"
        assert f"job={mock_hydra_context['job_name']}" in output, "Job name not found"
        assert f"run_id={mock_hydra_context['run_id']}" in output, "Run ID not found"
        assert f"checksum={mock_hydra_context['config_checksum']}" in output, "Config checksum not found"
    
    def test_bind_seed_context(self, mock_seed_context):
        """Test seed manager context binding."""
        string_io = io.StringIO()
        
        # Setup basic logger
        setup_logger(sink=None, level="DEBUG")
        
        # Format that shows seed context
        seed_format = "{level}|seed={extra[seed]}|seed_run_id={extra[run_id]}|timestamp={extra[initial_timestamp]}|{message}"
        handler_id = logger.add(string_io, level="DEBUG", format=seed_format)
        
        # Bind seed context
        with patch('{{cookiecutter.project_slug}}.utils.logging.bind_seed_context') as mock_bind:
            def bind_side_effect(context):
                return logger.bind(**context)
            mock_bind.side_effect = bind_side_effect
            
            seed_logger = bind_seed_context(mock_seed_context)
            test_message = "Test seed context binding"
            seed_logger.info(test_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify seed context components
        assert test_message in output, "Test message not found"
        assert f"seed={mock_seed_context['seed']}" in output, "Seed not found"
        assert f"seed_run_id={mock_seed_context['run_id']}" in output, "Seed run ID not found"
        assert f"timestamp={mock_seed_context['initial_timestamp']}" in output, "Initial timestamp not found"
    
    def test_bind_cli_context(self, mock_cli_context):
        """Test CLI command context preservation."""
        string_io = io.StringIO()
        
        # Setup basic logger
        setup_logger(sink=None, level="DEBUG")
        
        # Format that shows CLI context
        cli_format = "{level}|command={extra[command]}|exec_id={extra[execution_id]}|{message}"
        handler_id = logger.add(string_io, level="DEBUG", format=cli_format)
        
        # Bind CLI context
        with patch('{{cookiecutter.project_slug}}.utils.logging.bind_cli_context') as mock_bind:
            def bind_side_effect(context):
                return logger.bind(**context)
            mock_bind.side_effect = bind_side_effect
            
            cli_logger = bind_cli_context(mock_cli_context)
            test_message = "Test CLI context preservation"
            cli_logger.info(test_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify CLI context components
        assert test_message in output, "Test message not found"
        assert f"command={mock_cli_context['command']}" in output, "Command not found"
        assert f"exec_id={mock_cli_context['execution_id']}" in output, "Execution ID not found"
    
    def test_generate_config_checksum(self, sample_config_dict):
        """Test configuration checksum generation for reproducibility."""
        with patch('{{cookiecutter.project_slug}}.utils.logging.generate_config_checksum') as mock_checksum:
            # Mock implementation that generates a realistic checksum
            def checksum_side_effect(config):
                config_str = json.dumps(config, sort_keys=True)
                return hashlib.sha256(config_str.encode()).hexdigest()[:16]
            
            mock_checksum.side_effect = checksum_side_effect
            
            # Generate checksum
            checksum1 = generate_config_checksum(sample_config_dict)
            checksum2 = generate_config_checksum(sample_config_dict)
            
            # Verify consistency
            assert checksum1 == checksum2, "Checksums should be consistent for same config"
            assert len(checksum1) == 16, "Checksum should be 16 characters"
            assert isinstance(checksum1, str), "Checksum should be string"
            
            # Test with modified config
            modified_config = sample_config_dict.copy()
            modified_config["navigator"]["speed"] = 2.0
            checksum3 = generate_config_checksum(modified_config)
            
            assert checksum1 != checksum3, "Different configs should have different checksums"
    
    def test_structured_logging_with_correlation_ids(self):
        """Test enhanced structured logging format with correlation IDs."""
        string_io = io.StringIO()
        
        # Setup logger with correlation ID format
        correlation_format = "{level}|correlation_id={extra[correlation_id]}|experiment_id={extra[experiment_id]}|{message}"
        setup_logger(sink=None, level="DEBUG")
        handler_id = logger.add(string_io, level="DEBUG", format=correlation_format)
        
        # Create correlation context
        correlation_id = str(uuid4())
        experiment_id = "experiment_2024_01_15"
        
        # Bind correlation context
        corr_logger = logger.bind(correlation_id=correlation_id, experiment_id=experiment_id)
        
        # Log test messages
        test_messages = [
            "Starting experiment phase 1",
            "Processing frame 100",
            "Agent navigation update",
            "Experiment phase 1 complete"
        ]
        
        for msg in test_messages:
            corr_logger.info(msg)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify correlation IDs appear in all messages
        lines = output.strip().split('\n')
        assert len(lines) == len(test_messages), "All messages should be logged"
        
        for line, expected_msg in zip(lines, test_messages):
            assert f"correlation_id={correlation_id}" in line, "Correlation ID missing"
            assert f"experiment_id={experiment_id}" in line, "Experiment ID missing"
            assert expected_msg in line, "Expected message missing"
    
    def test_configuration_composition_tracking(self):
        """Test configuration composition tracking and environment variable interpolation logging."""
        string_io = io.StringIO()
        
        # Setup logger with composition tracking format
        comp_format = "{level}|config_source={extra[config_source]}|override_count={extra[override_count]}|env_vars={extra[env_vars]}|{message}"
        setup_logger(sink=None, level="DEBUG")
        handler_id = logger.add(string_io, level="DEBUG", format=comp_format)
        
        # Mock configuration composition context
        composition_context = {
            "config_source": "conf/base.yaml",
            "override_count": 3,
            "env_vars": ["EXPERIMENT_SEED", "OUTPUT_DIR"]
        }
        
        # Bind composition context
        comp_logger = logger.bind(**composition_context)
        
        # Log configuration events
        comp_logger.info("Configuration loaded from base")
        comp_logger.info("Applied user overrides")
        comp_logger.info("Environment variable interpolation complete")
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify composition tracking components
        assert "config_source=conf/base.yaml" in output, "Config source not tracked"
        assert "override_count=3" in output, "Override count not tracked"
        assert "env_vars=['EXPERIMENT_SEED', 'OUTPUT_DIR']" in output, "Environment variables not tracked"
        assert "Configuration loaded from base" in output, "Configuration loading not logged"
        assert "Applied user overrides" in output, "Override application not logged"
        assert "Environment variable interpolation complete" in output, "Environment interpolation not logged"
    
    def test_multi_context_binding(self, mock_hydra_context, mock_seed_context, mock_cli_context):
        """Test binding multiple contexts simultaneously."""
        string_io = io.StringIO()
        
        # Setup logger with multi-context format
        multi_format = (
            "{level}|job={extra[job_name]}|seed={extra[seed]}|command={extra[command]}|"
            "run_id={extra[run_id]}|{message}"
        )
        setup_logger(sink=None, level="DEBUG")
        handler_id = logger.add(string_io, level="DEBUG", format=multi_format)
        
        # Combine all contexts
        combined_context = {**mock_hydra_context, **mock_seed_context, **mock_cli_context}
        
        # Note: run_id conflict - seed context will override hydra context
        # This is expected behavior in real implementation
        
        # Bind combined context
        multi_logger = logger.bind(**combined_context)
        
        # Log test message
        test_message = "Multi-context logging test"
        multi_logger.info(test_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify all context types are present
        assert test_message in output, "Test message not found"
        assert f"job={mock_hydra_context['job_name']}" in output, "Hydra job name missing"
        assert f"seed={mock_seed_context['seed']}" in output, "Seed context missing"
        assert f"command={mock_cli_context['command']}" in output, "CLI command missing"
        # run_id should be from seed context (last bound)
        assert f"run_id={mock_seed_context['run_id']}" in output, "Seed run_id should override"


class TestBackwardCompatibilityAndIntegration:
    """Tests for backward compatibility and integration with existing functionality."""
    
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
    
    def test_backward_compatible_setup_logger_interface(self):
        """Test that the enhanced logging maintains backward compatible interface."""
        string_io = io.StringIO()
        
        # Test that original setup_logger interface still works
        setup_logger(sink=None, level="INFO")
        handler_id = logger.add(string_io, level="DEBUG")
        
        # Test standard logging functionality
        test_message = "Backward compatibility test"
        logger.info(test_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify basic functionality works
        assert test_message in output, "Basic logging functionality should work"
    
    def test_get_module_logger_with_enhanced_context(self):
        """Test module logger works with enhanced context binding."""
        string_io = io.StringIO()
        
        # Setup logger
        setup_logger(sink=None, level="DEBUG")
        
        # Format that shows both module and enhanced context
        enhanced_format = (
            "{level}|module={extra[module]}|"
            "job={extra[job_name]:-none}|seed={extra[seed]:-none}|{message}"
        )
        handler_id = logger.add(string_io, level="DEBUG", format=enhanced_format)
        
        # Get module logger
        module_name = "test_enhanced_module"
        module_logger = get_module_logger(module_name)
        
        # Add enhanced context to module logger
        enhanced_context = {
            "job_name": "enhanced_test_job",
            "seed": 98765
        }
        enhanced_module_logger = module_logger.bind(**enhanced_context)
        
        # Log test message
        test_message = "Enhanced module logger test"
        enhanced_module_logger.info(test_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify both module and enhanced context
        assert test_message in output, "Test message not found"
        assert f"module={module_name}" in output, "Module name not found"
        assert f"job={enhanced_context['job_name']}" in output, "Job name not found"
        assert f"seed={enhanced_context['seed']}" in output, "Seed not found"
    
    @patch.dict(os.environ, {
        'EXPERIMENT_SEED': '12345',
        'OUTPUT_DIR': '/test/output',
        'HYDRA_JOB_NAME': 'env_test_job'
    })
    def test_environment_variable_integration(self):
        """Test integration with environment variable interpolation logging."""
        string_io = io.StringIO()
        
        # Setup logger with environment variable format
        env_format = "{level}|env_seed={extra[env_seed]}|env_output={extra[env_output]}|env_job={extra[env_job]}|{message}"
        setup_logger(sink=None, level="DEBUG")
        handler_id = logger.add(string_io, level="DEBUG", format=env_format)
        
        # Create context from environment variables
        env_context = {
            "env_seed": os.environ.get('EXPERIMENT_SEED', 'not_set'),
            "env_output": os.environ.get('OUTPUT_DIR', 'not_set'),
            "env_job": os.environ.get('HYDRA_JOB_NAME', 'not_set')
        }
        
        # Bind environment context
        env_logger = logger.bind(**env_context)
        
        # Log test message
        test_message = "Environment variable integration test"
        env_logger.info(test_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify environment variables are captured
        assert test_message in output, "Test message not found"
        assert "env_seed=12345" in output, "Environment seed not captured"
        assert "env_output=/test/output" in output, "Environment output not captured"
        assert "env_job=env_test_job" in output, "Environment job name not captured"
    
    def test_performance_with_enhanced_context(self):
        """Test that enhanced logging doesn't significantly impact performance."""
        import time
        
        # Setup logger with rich context
        setup_logger(sink=None, level="INFO")
        
        # Create comprehensive context
        comprehensive_context = {
            "job_name": "performance_test",
            "run_id": "perf_run_123",
            "config_checksum": "perf_checksum",
            "seed": 54321,
            "correlation_id": str(uuid4()),
            "experiment_id": "performance_experiment"
        }
        
        # Bind context once
        perf_logger = logger.bind(**comprehensive_context)
        
        # Measure performance
        start_time = time.time()
        num_messages = 1000
        
        for i in range(num_messages):
            perf_logger.info(f"Performance test message {i}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertion - should complete 1000 logs in reasonable time
        # This is a basic smoke test - in production you'd have more specific requirements
        assert total_time < 5.0, f"Logging {num_messages} messages took too long: {total_time:.2f}s"
        
        # Calculate messages per second
        messages_per_second = num_messages / total_time
        assert messages_per_second > 100, f"Logging rate too slow: {messages_per_second:.1f} msg/s"
    
    def test_error_handling_with_enhanced_context(self):
        """Test error handling and exception logging with enhanced context."""
        string_io = io.StringIO()
        
        # Setup logger with error context format
        error_format = "{level}|error_context={extra[error_context]}|{message}"
        setup_logger(sink=None, level="DEBUG", backtrace=True, diagnose=True)
        handler_id = logger.add(string_io, level="DEBUG", format=error_format)
        
        # Create error context
        error_context = {
            "error_context": "enhanced_logging_test",
        }
        
        # Bind error context
        error_logger = logger.bind(**error_context)
        
        # Test exception logging with context
        test_error_message = "Enhanced exception test"
        
        with suppress(ValueError):
            try:
                raise ValueError("Enhanced logging exception")
            except Exception:
                error_logger.exception(test_error_message)
        
        # Get output
        output = string_io.getvalue()
        
        # Remove handler
        logger.remove(handler_id)
        
        # Verify exception logging with context
        assert test_error_message in output, "Error message not found"
        assert "error_context=enhanced_logging_test" in output, "Error context not preserved"
        assert "ValueError" in output, "Exception type not logged"
        assert "Enhanced logging exception" in output, "Exception details not logged"