"""
Comprehensive test suite for enhanced logging configuration with Hydra integration.

This module provides extensive testing for the enhanced logging system including
structured output formats, configuration-driven setup, log level management,
CLI integration, and environment-specific configuration validation.

Test Categories:
- Configuration management with Hydra integration (Feature F-006)
- Structured output format validation for research workflows
- Log level management and dynamic configuration override testing
- CLI command execution context and performance metrics tracking
- Environment-specific logging configuration (development, production, testing)
- Log file management, rotation, and retention capabilities
- Correlation ID and seed manager integration validation
- Performance monitoring and threshold-based warning systems
"""

import pytest
import sys
import os
import tempfile
import json
import time
import threading
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock, call, mock_open
from contextlib import contextmanager
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, Any, List, Optional

import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

# Import the logging module under test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from {{cookiecutter.project_slug}}.utils.logging import (
    LoggingConfig,
    EnhancedLoggingManager,
    CorrelationContext,
    HydraJobTracker,
    CLIMetricsTracker,
    get_logging_manager,
    setup_enhanced_logging,
    get_module_logger,
    create_correlation_scope,
    create_cli_command_scope,
    create_parameter_validation_scope,
    log_environment_variables,
    log_configuration_composition,
)


# =============================================================================
# ENHANCED FIXTURES FOR LOGGING SYSTEM TESTING
# =============================================================================

@pytest.fixture
def enhanced_config_files():
    """
    Enhanced configuration fixture specifically for logging system testing.
    
    Provides comprehensive configuration data including logging-specific settings
    for all environments and testing scenarios.
    """
    # Base logging configuration
    base_config = {
        "logging": {
            "level": "INFO",
            "console_format": "enhanced",
            "file_format": "structured",
            "enable_file_logging": True,
            "log_directory": None,
            "log_filename": "experiment.log",
            "rotation": "100 MB",
            "retention": "30 days",
            "compression": "gz",
            "enable_hydra_integration": True,
            "enable_seed_context": True,
            "enable_cli_metrics": True,
            "enable_config_tracking": True,
            "enable_correlation_ids": True,
            "enable_performance_monitoring": True,
            "performance_threshold_ms": 100.0,
            "enable_exception_diagnostics": True,
            "enable_environment_logging": True,
            "enable_module_filtering": False,
            "filtered_modules": [],
            "debug_hydra_composition": False
        }
    }
    
    # User configuration with overrides
    user_config = {
        "logging": {
            "level": "DEBUG",
            "console_format": "simple",
            "enable_cli_metrics": True,
            "performance_threshold_ms": 50.0,
            "debug_hydra_composition": True
        }
    }
    
    # Local development configuration
    local_config = {
        "logging": {
            "level": "DEBUG",
            "console_format": "enhanced",
            "file_format": "json",
            "enable_file_logging": True,
            "enable_performance_monitoring": True,
            "enable_exception_diagnostics": True,
            "debug_hydra_composition": True
        }
    }
    
    # Production configuration
    production_config = {
        "logging": {
            "level": "WARNING",
            "console_format": "structured",
            "file_format": "json",
            "enable_file_logging": True,
            "log_directory": "/var/log/odor_plume_nav",
            "rotation": "50 MB",
            "retention": "90 days",
            "enable_hydra_integration": True,
            "enable_seed_context": True,
            "enable_cli_metrics": True,
            "enable_config_tracking": True,
            "enable_correlation_ids": True,
            "enable_performance_monitoring": True,
            "performance_threshold_ms": 200.0,
            "enable_exception_diagnostics": False,
            "enable_environment_logging": True,
            "debug_hydra_composition": False
        }
    }
    
    # Testing configuration
    testing_config = {
        "logging": {
            "level": "DEBUG",
            "console_format": "simple",
            "file_format": "enhanced",
            "enable_file_logging": False,
            "enable_hydra_integration": False,
            "enable_seed_context": False,
            "enable_cli_metrics": False,
            "enable_config_tracking": False,
            "enable_correlation_ids": False,
            "enable_performance_monitoring": False,
            "enable_exception_diagnostics": True,
            "enable_environment_logging": False,
            "debug_hydra_composition": False
        }
    }
    
    return {
        "base_config": base_config,
        "user_config": user_config,
        "local_config": local_config,
        "production_config": production_config,
        "testing_config": testing_config
    }


@pytest.fixture
def temp_logging_config_files(tmp_path, enhanced_config_files):
    """
    Create temporary Hydra-compatible configuration files for logging system testing.
    
    Creates a complete conf/ directory structure with logging-specific configurations
    for comprehensive integration testing with Hydra configuration composition.
    """
    import yaml
    
    # Get configuration data
    configs = enhanced_config_files
    
    # Create Hydra-compatible directory structure
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    
    local_dir = config_dir / "local"
    local_dir.mkdir()
    
    # Create base configuration file (conf/base.yaml)
    base_path = config_dir / "base.yaml"
    with open(base_path, 'w') as f:
        yaml.dump(configs["base_config"], f, default_flow_style=False)
    
    # Create main configuration file (conf/config.yaml)  
    config_path = config_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(configs["user_config"], f, default_flow_style=False)
    
    # Create local development configuration (conf/local/development.yaml)
    local_dev_path = local_dir / "development.yaml"
    with open(local_dev_path, 'w') as f:
        yaml.dump(configs["local_config"], f, default_flow_style=False)
    
    # Create production configuration (conf/local/production.yaml)
    local_prod_path = local_dir / "production.yaml"
    with open(local_prod_path, 'w') as f:
        yaml.dump(configs["production_config"], f, default_flow_style=False)
    
    # Create testing configuration (conf/local/testing.yaml)
    local_test_path = local_dir / "testing.yaml"
    with open(local_test_path, 'w') as f:
        yaml.dump(configs["testing_config"], f, default_flow_style=False)
    
    return {
        "config_dir": config_dir,
        "base_path": base_path,
        "config_path": config_path,
        "local_dev_path": local_dev_path,
        "local_prod_path": local_prod_path,
        "local_test_path": local_test_path,
        **configs
    }


@pytest.fixture
def mock_hydra_global():
    """
    Mock Hydra global state for testing Hydra integration functionality.
    
    Provides comprehensive mocking of Hydra GlobalHydra instance with realistic
    configuration structure for testing job tracking and configuration composition.
    """
    with patch('{{cookiecutter.project_slug}}.utils.logging.GlobalHydra') as mock_global:
        # Create mock Hydra configuration
        mock_instance = MagicMock()
        mock_global.instance.return_value = mock_instance
        mock_global.is_initialized.return_value = True
        
        # Mock Hydra configuration structure
        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.job.name = "test_job"
        mock_hydra_cfg.hydra.version = "1.3.0"
        mock_hydra_cfg.runtime.choices = {"experiment": "test", "model": "simple"}
        mock_hydra_cfg.hydra.overrides.task = ["experiment=test", "model.learning_rate=0.01"]
        mock_hydra_cfg.hydra.overrides.hydra = ["job.name=test_job"]
        mock_hydra_cfg.hydra.searchpath = ["/conf", "/conf/local"]
        mock_hydra_cfg.defaults = [{"_self_": True}, {"experiment": "test"}]
        
        # Mock runtime configuration
        mock_runtime_cfg = DictConfig({
            "experiment": {
                "seed": 42,
                "max_iterations": 1000
            },
            "model": {
                "learning_rate": "${oc.env:LEARNING_RATE,0.01}",
                "hidden_units": 128
            },
            "logging": {
                "level": "DEBUG",
                "enable_hydra_integration": True
            }
        })
        
        mock_instance.hydra_cfg = mock_hydra_cfg
        mock_instance.cfg = mock_runtime_cfg
        
        yield mock_global, mock_instance, mock_hydra_cfg, mock_runtime_cfg


@pytest.fixture
def mock_seed_manager():
    """
    Mock seed manager for testing seed context integration.
    
    Provides realistic seed manager behavior for testing reproducibility
    context binding in the logging system.
    """
    with patch('{{cookiecutter.project_slug}}.utils.logging.get_seed_manager') as mock_get:
        mock_manager = MagicMock()
        mock_manager.current_seed = 42
        mock_manager.run_id = "run_20240603_123456"
        mock_manager.environment_hash = "env_abc123def456"
        mock_get.return_value = mock_manager
        yield mock_manager


@pytest.fixture
def isolated_logging_manager():
    """
    Provide isolated logging manager instance for each test.
    
    Ensures test isolation by resetting the singleton logging manager
    before and after each test execution.
    """
    # Reset singleton before test
    EnhancedLoggingManager.reset()
    
    yield
    
    # Reset singleton after test
    EnhancedLoggingManager.reset()


@pytest.fixture
def captured_logs():
    """
    Capture log output for validation in tests.
    
    Provides both console and file log capture capabilities with
    structured access to log records and formatted output.
    """
    captured_records = []
    captured_console = StringIO()
    
    def log_sink(message):
        """Custom sink to capture log messages."""
        captured_records.append(message.record)
        captured_console.write(str(message))
        captured_console.write('\n')
    
    # Add custom sink to capture logs
    logger.remove()  # Remove default handlers
    sink_id = logger.add(log_sink, level="TRACE")
    
    yield {
        "records": captured_records,
        "console": captured_console,
        "get_logs": lambda: captured_console.getvalue(),
        "get_records": lambda: captured_records.copy(),
        "clear": lambda: (captured_records.clear(), captured_console.seek(0), captured_console.truncate(0))
    }
    
    # Cleanup
    logger.remove(sink_id)


@pytest.fixture
def environment_variables():
    """
    Provide controlled environment variable management for testing.
    
    Sets up and tears down environment variables needed for testing
    environment variable interpolation and logging.
    """
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env_vars = {
        "LOG_LEVEL": "DEBUG",
        "LOG_DIR": "/tmp/test_logs",
        "DB_USER": "test_user",
        "DB_PASSWORD": "secret123",
        "API_KEY": "test_api_key_456",
        "LEARNING_RATE": "0.001",
        "EXPERIMENT_NAME": "test_experiment"
    }
    
    for key, value in test_env_vars.items():
        os.environ[key] = value
    
    yield test_env_vars
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# =============================================================================
# LOGGING CONFIGURATION TESTING
# =============================================================================

class TestLoggingConfiguration:
    """Test logging configuration management and validation."""
    
    def test_logging_config_creation_with_defaults(self):
        """Test LoggingConfig creation with default values."""
        config = LoggingConfig()
        
        # Validate default configuration values
        assert config.level == "INFO"
        assert config.console_format == "enhanced"
        assert config.file_format == "structured"
        assert config.enable_file_logging is True
        assert config.log_directory is None
        assert config.log_filename == "experiment.log"
        assert config.rotation == "100 MB"
        assert config.retention == "30 days"
        assert config.compression == "gz"
        assert config.enable_hydra_integration is True
        assert config.enable_seed_context is True
        assert config.enable_cli_metrics is True
        assert config.enable_config_tracking is True
        assert config.enable_correlation_ids is True
        assert config.enable_performance_monitoring is True
        assert config.performance_threshold_ms == 100.0
        assert config.enable_exception_diagnostics is True
        assert config.enable_environment_logging is True
        assert config.enable_module_filtering is False
        assert config.filtered_modules == []
        assert config.debug_hydra_composition is False
    
    def test_logging_config_custom_values(self):
        """Test LoggingConfig creation with custom values."""
        custom_config = {
            "level": "DEBUG",
            "console_format": "json",
            "file_format": "simple",
            "enable_file_logging": False,
            "log_directory": "/custom/log/dir",
            "log_filename": "custom.log",
            "rotation": "50 MB",
            "retention": "7 days",
            "compression": "bz2",
            "enable_hydra_integration": False,
            "enable_seed_context": False,
            "enable_cli_metrics": False,
            "enable_config_tracking": False,
            "enable_correlation_ids": False,
            "enable_performance_monitoring": False,
            "performance_threshold_ms": 250.0,
            "enable_exception_diagnostics": False,
            "enable_environment_logging": False,
            "enable_module_filtering": True,
            "filtered_modules": ["test_module", "debug_module"],
            "debug_hydra_composition": True
        }
        
        config = LoggingConfig(**custom_config)
        
        # Validate all custom values are set correctly
        for key, expected_value in custom_config.items():
            assert getattr(config, key) == expected_value
    
    def test_logging_config_validation_errors(self):
        """Test LoggingConfig validation with invalid values."""
        # Test invalid log level
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID_LEVEL")
        
        # Test invalid threshold type
        with pytest.raises(ValueError):
            LoggingConfig(performance_threshold_ms="not_a_number")
    
    def test_logging_config_from_dict_config(self):
        """Test LoggingConfig creation from OmegaConf DictConfig."""
        config_dict = {
            "level": "WARNING",
            "console_format": "structured",
            "enable_cli_metrics": False,
            "performance_threshold_ms": 150.0
        }
        
        dict_config = DictConfig(config_dict)
        config = LoggingConfig(**dict_config)
        
        assert config.level == "WARNING"
        assert config.console_format == "structured" 
        assert config.enable_cli_metrics is False
        assert config.performance_threshold_ms == 150.0
        # Ensure defaults are preserved for non-specified values
        assert config.enable_hydra_integration is True


class TestEnhancedLoggingManagerInitialization:
    """Test enhanced logging manager initialization and configuration."""
    
    def test_singleton_behavior(self, isolated_logging_manager):
        """Test that EnhancedLoggingManager follows singleton pattern."""
        manager1 = EnhancedLoggingManager()
        manager2 = EnhancedLoggingManager()
        manager3 = get_logging_manager()
        
        # All instances should be the same object
        assert manager1 is manager2
        assert manager2 is manager3
        assert id(manager1) == id(manager2) == id(manager3)
    
    def test_reset_singleton(self, isolated_logging_manager):
        """Test singleton reset functionality for testing."""
        manager1 = EnhancedLoggingManager()
        manager1_id = id(manager1)
        
        # Reset singleton
        EnhancedLoggingManager.reset()
        
        manager2 = EnhancedLoggingManager()
        manager2_id = id(manager2)
        
        # Should be different instances after reset
        assert manager1_id != manager2_id
    
    def test_initialization_with_logging_config(self, isolated_logging_manager, captured_logs):
        """Test logging manager initialization with LoggingConfig object."""
        config = LoggingConfig(
            level="DEBUG",
            console_format="simple",
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Test that configuration was applied
        logger.info("Test message")
        
        logs = captured_logs["get_logs"]()
        assert "Test message" in logs
        assert "Enhanced logging system initialized successfully" in logs
    
    def test_initialization_with_dict_config(self, isolated_logging_manager, captured_logs):
        """Test logging manager initialization with dictionary configuration."""
        config_dict = {
            "level": "WARNING",
            "console_format": "enhanced", 
            "enable_file_logging": False,
            "enable_hydra_integration": False,
            "performance_threshold_ms": 75.0
        }
        
        manager = get_logging_manager()
        manager.initialize(config_dict)
        
        # Test that configuration was applied
        logger.warning("Test warning message")
        logger.info("This should not appear")  # Below WARNING level
        
        logs = captured_logs["get_logs"]()
        assert "Test warning message" in logs
        assert "This should not appear" not in logs
    
    def test_initialization_with_hydra_config(self, isolated_logging_manager, captured_logs, mock_hydra_global):
        """Test logging manager initialization with Hydra configuration loading."""
        mock_global, mock_instance, mock_hydra_cfg, mock_runtime_cfg = mock_hydra_global
        
        # Add logging config to mock runtime config
        mock_runtime_cfg.logging = DictConfig({
            "level": "DEBUG",
            "console_format": "structured",
            "enable_hydra_integration": True,
            "enable_seed_context": False
        })
        
        manager = get_logging_manager()
        manager.initialize()  # Should load from Hydra
        
        # Verify Hydra integration was enabled
        assert mock_global.is_initialized.called
        assert mock_global.instance.called
        
        # Test logging with Hydra context
        logger.info("Test Hydra integration")
        
        logs = captured_logs["get_logs"]()
        assert "Test Hydra integration" in logs
        assert "Enhanced logging system initialized successfully" in logs
    
    def test_initialization_fallback_on_error(self, isolated_logging_manager, captured_logs):
        """Test fallback logging setup when initialization fails."""
        # Create invalid configuration that will cause initialization to fail
        invalid_config = MagicMock()
        invalid_config.level = "INVALID"
        
        manager = get_logging_manager()
        
        with pytest.raises(RuntimeError):
            manager.initialize(invalid_config)
        
        # Should still be able to log with fallback configuration
        logger.info("Fallback test message")
        
        logs = captured_logs["get_logs"]()
        assert "Fallback test message" in logs
    
    def test_performance_tracking_during_initialization(self, isolated_logging_manager, captured_logs):
        """Test that initialization performance is tracked and logged."""
        config = LoggingConfig(
            enable_file_logging=False,
            enable_hydra_integration=False,
            performance_threshold_ms=1.0  # Very low threshold to trigger warning
        )
        
        manager = get_logging_manager()
        
        # Mock a slow initialization
        with patch('time.perf_counter', side_effect=[0.0, 0.15]):  # 150ms initialization
            manager.initialize(config)
        
        logs = captured_logs["get_logs"]()
        assert "initialization exceeded performance threshold" in logs.lower()


# =============================================================================
# STRUCTURED OUTPUT FORMAT TESTING
# =============================================================================

class TestStructuredOutputFormats:
    """Test structured output formats and message formatting."""
    
    def test_simple_console_format(self, isolated_logging_manager, captured_logs):
        """Test simple console output format."""
        config = LoggingConfig(
            console_format="simple",
            enable_file_logging=False,
            enable_hydra_integration=False,
            enable_correlation_ids=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        logger.info("Simple format test message")
        
        logs = captured_logs["get_logs"]()
        
        # Simple format should contain time, level, and message
        assert "INFO" in logs
        assert "Simple format test message" in logs
        # Should not contain extra context like correlation IDs
        assert "corr=" not in logs
        assert "job=" not in logs
    
    def test_enhanced_console_format(self, isolated_logging_manager, captured_logs):
        """Test enhanced console output format with context."""
        config = LoggingConfig(
            console_format="enhanced",
            enable_file_logging=False,
            enable_hydra_integration=False,
            enable_correlation_ids=True
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        logger.info("Enhanced format test message")
        
        logs = captured_logs["get_logs"]()
        
        # Enhanced format should contain additional context
        assert "INFO" in logs
        assert "Enhanced format test message" in logs
        assert "corr=" in logs  # Correlation ID should be present
        # Should contain module and function context
        assert ":test_enhanced_console_format:" in logs or "test_logging" in logs
    
    def test_structured_console_format(self, isolated_logging_manager, captured_logs, mock_seed_manager):
        """Test structured console output format with full context."""
        config = LoggingConfig(
            console_format="structured",
            enable_file_logging=False,
            enable_hydra_integration=False,
            enable_correlation_ids=True,
            enable_seed_context=True
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        logger.info("Structured format test message")
        
        logs = captured_logs["get_logs"]()
        
        # Structured format should contain all context elements
        assert "INFO" in logs
        assert "Structured format test message" in logs
        assert "corr=" in logs  # Correlation ID
        assert "job=" in logs   # Job name
        assert "seed=" in logs  # Seed context
    
    def test_json_console_format(self, isolated_logging_manager, captured_logs):
        """Test JSON console output format for machine-readable logs."""
        config = LoggingConfig(
            console_format="json",
            enable_file_logging=False,
            enable_hydra_integration=False,
            enable_correlation_ids=True
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        logger.info("JSON format test message")
        
        logs = captured_logs["get_logs"]()
        
        # Find JSON log lines
        log_lines = [line.strip() for line in logs.split('\n') if line.strip()]
        json_logs = []
        
        for line in log_lines:
            if line.startswith('{"timestamp"'):
                try:
                    json_log = json.loads(line)
                    json_logs.append(json_log)
                except json.JSONDecodeError:
                    continue
        
        # Should have at least one valid JSON log entry
        assert len(json_logs) > 0
        
        # Find the test message log
        test_log = None
        for log_entry in json_logs:
            if "JSON format test message" in log_entry.get("message", ""):
                test_log = log_entry
                break
        
        assert test_log is not None
        
        # Validate JSON structure
        assert "timestamp" in test_log
        assert "level" in test_log
        assert "module" in test_log
        assert "function" in test_log
        assert "line" in test_log
        assert "correlation_id" in test_log
        assert "message" in test_log
        assert test_log["level"] == "INFO"
        assert test_log["message"] == "JSON format test message"
    
    def test_file_format_differs_from_console(self, isolated_logging_manager, tmp_path):
        """Test that file format can be different from console format."""
        log_file = tmp_path / "test.log"
        
        config = LoggingConfig(
            console_format="simple",
            file_format="json",
            enable_file_logging=True,
            log_directory=str(tmp_path),
            log_filename="test.log",
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        logger.info("Format difference test message")
        
        # Wait for file write
        import time
        time.sleep(0.1)
        
        # Check that log file exists and contains JSON format
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            file_content = f.read()
        
        # File should contain JSON format
        assert '{"timestamp"' in file_content
        assert '"message": "Format difference test message"' in file_content


class TestLogLevelManagement:
    """Test log level management and dynamic configuration."""
    
    def test_log_level_filtering(self, isolated_logging_manager, captured_logs):
        """Test that log level filtering works correctly."""
        config = LoggingConfig(
            level="WARNING",
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log messages at different levels
        logger.debug("Debug message - should not appear")
        logger.info("Info message - should not appear")
        logger.warning("Warning message - should appear")
        logger.error("Error message - should appear")
        logger.critical("Critical message - should appear")
        
        logs = captured_logs["get_logs"]()
        
        # Only WARNING and above should appear
        assert "Debug message" not in logs
        assert "Info message" not in logs
        assert "Warning message" in logs
        assert "Error message" in logs
        assert "Critical message" in logs
    
    def test_trace_level_logging(self, isolated_logging_manager, captured_logs):
        """Test TRACE level logging for detailed debugging."""
        config = LoggingConfig(
            level="TRACE",
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Test all log levels
        logger.trace("Trace message")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        logs = captured_logs["get_logs"]()
        
        # All messages should appear with TRACE level
        assert "Trace message" in logs
        assert "Debug message" in logs
        assert "Info message" in logs
        assert "Warning message" in logs
    
    def test_critical_level_only(self, isolated_logging_manager, captured_logs):
        """Test CRITICAL level filtering for production environments."""
        config = LoggingConfig(
            level="CRITICAL",
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log messages at different levels
        logger.info("Info message - should not appear")
        logger.warning("Warning message - should not appear")
        logger.error("Error message - should not appear")
        logger.critical("Critical message - should appear")
        
        logs = captured_logs["get_logs"]()
        
        # Only CRITICAL should appear
        assert "Info message" not in logs
        assert "Warning message" not in logs
        assert "Error message" not in logs
        assert "Critical message" in logs


# =============================================================================
# HYDRA INTEGRATION TESTING
# =============================================================================

class TestHydraIntegration:
    """Test Hydra configuration integration and job tracking."""
    
    def test_hydra_job_tracker_initialization(self, mock_hydra_global):
        """Test HydraJobTracker initialization with mock Hydra state."""
        mock_global, mock_instance, mock_hydra_cfg, mock_runtime_cfg = mock_hydra_global
        
        tracker = HydraJobTracker()
        job_info = tracker.initialize()
        
        # Verify job information extraction
        assert job_info["job_name"] == "test_job"
        assert job_info["hydra_version"] == "1.3.0"
        assert "config_checksum" in job_info
        assert "composition_info" in job_info
        assert "override_info" in job_info
        assert "environment_variables" in job_info
        assert "job_start_time" in job_info
        
        # Verify runtime choices extraction
        assert job_info["runtime_choices"]["experiment"] == "test"
        assert job_info["runtime_choices"]["model"] == "simple"
    
    def test_hydra_job_tracker_without_hydra(self):
        """Test HydraJobTracker behavior when Hydra is not initialized."""
        with patch('{{cookiecutter.project_slug}}.utils.logging.GlobalHydra') as mock_global:
            mock_global.is_initialized.return_value = False
            
            tracker = HydraJobTracker()
            job_info = tracker.initialize()
            
            # Should handle gracefully when Hydra is not available
            assert job_info["status"] == "hydra_not_initialized"
    
    def test_config_checksum_generation(self, mock_hydra_global):
        """Test configuration checksum generation for reproducibility."""
        mock_global, mock_instance, mock_hydra_cfg, mock_runtime_cfg = mock_hydra_global
        
        tracker = HydraJobTracker()
        tracker.initialize()
        
        # Get job metrics
        metrics = tracker.get_job_metrics()
        
        assert "config_checksum" in metrics
        assert metrics["config_checksum"] is not None
        assert len(metrics["config_checksum"]) == 12  # MD5 hash first 12 chars
    
    def test_environment_variable_extraction(self, mock_hydra_global, environment_variables):
        """Test environment variable usage extraction from configuration."""
        mock_global, mock_instance, mock_hydra_cfg, mock_runtime_cfg = mock_hydra_global
        
        # Add environment variable references to mock config
        mock_runtime_cfg.model.learning_rate = "${oc.env:LEARNING_RATE,0.01}"
        mock_runtime_cfg.database.url = "${DATABASE_URL}"
        
        tracker = HydraJobTracker()
        job_info = tracker.initialize()
        
        # Check that environment variables were extracted
        env_vars = job_info["environment_variables"]
        assert "LEARNING_RATE" in env_vars
        assert "DATABASE_URL" in env_vars
    
    def test_configuration_change_logging(self, isolated_logging_manager, captured_logs, mock_hydra_global):
        """Test configuration change logging during runtime."""
        mock_global, mock_instance, mock_hydra_cfg, mock_runtime_cfg = mock_hydra_global
        
        config = LoggingConfig(
            enable_hydra_integration=True,
            enable_config_tracking=True,
            enable_file_logging=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log a configuration change
        tracker = manager.get_hydra_tracker()
        tracker.log_configuration_change("parameter_override", {
            "parameter": "model.learning_rate",
            "old_value": 0.01,
            "new_value": 0.001,
            "source": "cli_override"
        })
        
        logs = captured_logs["get_logs"]()
        
        # Verify configuration change was logged
        assert "Configuration change detected: parameter_override" in logs
    
    def test_hydra_integration_in_logging_manager(self, isolated_logging_manager, captured_logs, mock_hydra_global):
        """Test complete Hydra integration in logging manager."""
        mock_global, mock_instance, mock_hydra_cfg, mock_runtime_cfg = mock_hydra_global
        
        config = LoggingConfig(
            enable_hydra_integration=True,
            enable_file_logging=False,
            console_format="structured"
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log a message with Hydra context
        logger.info("Test message with Hydra context")
        
        logs = captured_logs["get_logs"]()
        
        # Should contain job name from Hydra
        assert "job=test_job" in logs
        assert "Test message with Hydra context" in logs


# =============================================================================
# CLI INTEGRATION TESTING
# =============================================================================

class TestCLIIntegration:
    """Test CLI command tracking and performance metrics."""
    
    def test_cli_command_tracking(self, isolated_logging_manager, captured_logs):
        """Test CLI command execution tracking."""
        config = LoggingConfig(
            enable_cli_metrics=True,
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        cli_tracker = manager.get_cli_tracker()
        
        # Track a CLI command
        with cli_tracker.track_command("simulate") as metrics:
            # Simulate command execution
            time.sleep(0.01)  # Small delay to measure
            metrics["parameter_count"] = 5
            metrics["config_files_loaded"] = 3
        
        logs = captured_logs["get_logs"]()
        
        # Verify command tracking logs
        assert "CLI command completed: simulate" in logs
        records = captured_logs["get_records"]()
        
        # Find the completion record
        completion_record = None
        for record in records:
            if "CLI command completed" in record["message"]:
                completion_record = record
                break
        
        assert completion_record is not None
        assert "command_metrics" in completion_record["extra"]
        assert completion_record["extra"]["command_metrics"]["command_name"] == "simulate"
        assert "total_execution_time_ms" in completion_record["extra"]["command_metrics"]
    
    def test_parameter_validation_tracking(self, isolated_logging_manager, captured_logs):
        """Test parameter validation performance tracking."""
        config = LoggingConfig(
            enable_cli_metrics=True,
            enable_file_logging=False,
            enable_hydra_integration=False,
            performance_threshold_ms=5.0  # Low threshold for testing
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        cli_tracker = manager.get_cli_tracker()
        
        # Track parameter validation with artificial delay
        with cli_tracker.track_parameter_validation() as validation_metrics:
            time.sleep(0.01)  # Exceed threshold
            validation_metrics["parameters_validated"] = 10
            validation_metrics["validation_errors"] = 0
        
        logs = captured_logs["get_logs"]()
        
        # Should log performance warning due to exceeded threshold
        assert "Parameter validation exceeded performance threshold" in logs
    
    def test_parameter_validation_below_threshold(self, isolated_logging_manager, captured_logs):
        """Test parameter validation tracking when below performance threshold."""
        config = LoggingConfig(
            enable_cli_metrics=True,
            enable_file_logging=False,
            enable_hydra_integration=False,
            performance_threshold_ms=100.0  # High threshold
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        cli_tracker = manager.get_cli_tracker()
        
        # Track fast parameter validation
        with cli_tracker.track_parameter_validation() as validation_metrics:
            validation_metrics["parameters_validated"] = 5
        
        records = captured_logs["get_records"]()
        
        # Should have debug log entry, not warning
        validation_records = [r for r in records if "parameter_validation" in r.get("extra", {}).get("event_type", "")]
        assert len(validation_records) > 0
        
        validation_record = validation_records[0]
        assert validation_record["level"].name == "DEBUG"
    
    def test_cli_command_scope_convenience_function(self, isolated_logging_manager, captured_logs):
        """Test convenience function for CLI command scope creation."""
        config = LoggingConfig(
            enable_cli_metrics=True,
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        setup_enhanced_logging(config)
        
        # Use convenience function
        with create_cli_command_scope("analyze") as metrics:
            metrics["analysis_type"] = "trajectory"
            metrics["data_points"] = 1000
        
        logs = captured_logs["get_logs"]()
        assert "CLI command completed: analyze" in logs
    
    def test_parameter_validation_scope_convenience_function(self, isolated_logging_manager, captured_logs):
        """Test convenience function for parameter validation scope."""
        config = LoggingConfig(
            enable_cli_metrics=True,
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        setup_enhanced_logging(config)
        
        # Use convenience function
        with create_parameter_validation_scope() as validation_metrics:
            validation_metrics["config_keys_validated"] = 15
            validation_metrics["schema_checks"] = 8
        
        records = captured_logs["get_records"]()
        
        # Should have parameter validation record
        validation_records = [r for r in records if "parameter_validation" in r.get("extra", {}).get("event_type", "")]
        assert len(validation_records) > 0


# =============================================================================
# CORRELATION ID AND CONTEXT TESTING
# =============================================================================

class TestCorrelationContext:
    """Test correlation ID management and context tracking."""
    
    def test_correlation_id_generation(self):
        """Test automatic correlation ID generation."""
        # Clear any existing correlation ID
        CorrelationContext.clear_correlation_id()
        
        # Get correlation ID (should generate new one)
        corr_id1 = CorrelationContext.get_correlation_id()
        corr_id2 = CorrelationContext.get_correlation_id()
        
        # Should be the same within thread
        assert corr_id1 == corr_id2
        assert len(corr_id1) == 8  # Short UUID format
    
    def test_correlation_id_setting(self):
        """Test setting custom correlation ID."""
        custom_id = "custom123"
        CorrelationContext.set_correlation_id(custom_id)
        
        retrieved_id = CorrelationContext.get_correlation_id()
        assert retrieved_id == custom_id
    
    def test_correlation_id_thread_isolation(self):
        """Test that correlation IDs are isolated between threads."""
        results = {}
        
        def thread_function(thread_id):
            CorrelationContext.clear_correlation_id()
            corr_id = CorrelationContext.get_correlation_id()
            results[thread_id] = corr_id
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Each thread should have different correlation ID
        correlation_ids = list(results.values())
        assert len(set(correlation_ids)) == 3  # All unique
    
    def test_correlation_scope_context_manager(self):
        """Test correlation scope context manager."""
        # Clear existing correlation ID
        CorrelationContext.clear_correlation_id()
        
        original_id = CorrelationContext.get_correlation_id()
        
        # Use correlation scope
        with CorrelationContext.correlation_scope("scope123") as scope_id:
            assert scope_id == "scope123"
            assert CorrelationContext.get_correlation_id() == "scope123"
        
        # Should restore previous ID after scope
        current_id = CorrelationContext.get_correlation_id()
        assert current_id == original_id
    
    def test_nested_correlation_scopes(self):
        """Test nested correlation scope behavior."""
        CorrelationContext.clear_correlation_id()
        
        with CorrelationContext.correlation_scope("outer") as outer_id:
            assert CorrelationContext.get_correlation_id() == "outer"
            
            with CorrelationContext.correlation_scope("inner") as inner_id:
                assert CorrelationContext.get_correlation_id() == "inner"
            
            # Should restore outer scope
            assert CorrelationContext.get_correlation_id() == "outer"
    
    def test_correlation_id_in_logs(self, isolated_logging_manager, captured_logs):
        """Test correlation ID inclusion in log messages."""
        config = LoggingConfig(
            enable_correlation_ids=True,
            console_format="enhanced",
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log with specific correlation ID
        with create_correlation_scope("test_corr_123"):
            logger.info("Message with correlation ID")
        
        logs = captured_logs["get_logs"]()
        assert "corr=test_corr_123" in logs
        assert "Message with correlation ID" in logs
    
    def test_correlation_id_convenience_function(self, isolated_logging_manager, captured_logs):
        """Test correlation scope convenience function."""
        config = LoggingConfig(
            enable_correlation_ids=True,
            console_format="enhanced",
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        setup_enhanced_logging(config)
        
        # Use convenience function
        with create_correlation_scope("conv_test_456") as corr_id:
            assert corr_id == "conv_test_456"
            logger.info("Convenience function test")
        
        logs = captured_logs["get_logs"]()
        assert "corr=conv_test_456" in logs
        assert "Convenience function test" in logs


# =============================================================================
# ENVIRONMENT AND CONFIGURATION COMPOSITION TESTING
# =============================================================================

class TestEnvironmentConfiguration:
    """Test environment-specific configuration and composition."""
    
    def test_development_environment_config(self, isolated_logging_manager, captured_logs, temp_logging_config_files):
        """Test development environment logging configuration."""
        configs = temp_logging_config_files
        
        # Load development configuration
        dev_config = configs["local_config"]["logging"]
        config = LoggingConfig(**dev_config)
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Development should have DEBUG level and enhanced diagnostics
        assert config.level == "DEBUG"
        assert config.enable_exception_diagnostics is True
        assert config.debug_hydra_composition is True
        
        # Test debug logging
        logger.debug("Development debug message")
        
        logs = captured_logs["get_logs"]()
        assert "Development debug message" in logs
    
    def test_production_environment_config(self, isolated_logging_manager, captured_logs, temp_logging_config_files):
        """Test production environment logging configuration."""
        configs = temp_logging_config_files
        
        # Load production configuration
        prod_config = configs["production_config"]["logging"]
        config = LoggingConfig(**prod_config)
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Production should have WARNING level and minimal diagnostics
        assert config.level == "WARNING"
        assert config.enable_exception_diagnostics is False
        assert config.debug_hydra_composition is False
        assert config.performance_threshold_ms == 200.0
        
        # Test that only warnings and above are logged
        logger.info("Info message - should not appear")
        logger.warning("Warning message - should appear")
        
        logs = captured_logs["get_logs"]()
        assert "Info message" not in logs
        assert "Warning message" in logs
    
    def test_testing_environment_config(self, isolated_logging_manager, captured_logs, temp_logging_config_files):
        """Test testing environment logging configuration."""
        configs = temp_logging_config_files
        
        # Load testing configuration
        test_config = configs["testing_config"]["logging"]
        config = LoggingConfig(**test_config)
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Testing should disable most features for clean test output
        assert config.enable_file_logging is False
        assert config.enable_hydra_integration is False
        assert config.enable_seed_context is False
        assert config.enable_cli_metrics is False
        assert config.enable_correlation_ids is False
        assert config.enable_performance_monitoring is False
        
        # Should still allow debug level for test debugging
        assert config.level == "DEBUG"
        assert config.enable_exception_diagnostics is True
    
    def test_environment_variable_logging(self, isolated_logging_manager, captured_logs, environment_variables):
        """Test environment variable usage logging."""
        config = LoggingConfig(
            enable_environment_logging=True,
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log environment variable usage
        env_vars = ["LOG_LEVEL", "DB_USER", "DB_PASSWORD", "API_KEY"]
        log_environment_variables(env_vars)
        
        records = captured_logs["get_records"]()
        
        # Find environment access record
        env_record = None
        for record in records:
            if record.get("extra", {}).get("event_type") == "environment_access":
                env_record = record
                break
        
        assert env_record is not None
        
        env_info = env_record["extra"]["environment_variables"]
        
        # Check that values are properly masked for sensitive variables
        assert env_info["LOG_LEVEL"] == "DEBUG"  # Not sensitive
        assert env_info["DB_USER"] == "test_user"  # Not sensitive
        assert env_info["DB_PASSWORD"] == "***123"  # Masked sensitive value
        assert env_info["API_KEY"] == "***456"  # Masked sensitive value
    
    def test_configuration_composition_logging(self, isolated_logging_manager, captured_logs):
        """Test configuration composition tracking."""
        config = LoggingConfig(
            enable_config_tracking=True,
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log configuration composition
        composition_details = {
            "base_config": "base.yaml",
            "overrides": ["experiment=test", "model.learning_rate=0.001"],
            "resolved_keys": ["navigator.max_speed", "video_plume.kernel_size"],
            "composition_source": "hierarchical_merge"
        }
        
        log_configuration_composition(composition_details)
        
        records = captured_logs["get_records"]()
        
        # Find configuration composition record
        config_record = None
        for record in records:
            if record.get("extra", {}).get("event_type") == "config_composition":
                config_record = record
                break
        
        assert config_record is not None
        assert config_record["extra"]["composition_details"] == composition_details


# =============================================================================
# FILE LOGGING AND ROTATION TESTING
# =============================================================================

class TestFileLoggingAndRotation:
    """Test file logging, rotation, and retention capabilities."""
    
    def test_file_logging_basic(self, isolated_logging_manager, tmp_path):
        """Test basic file logging functionality."""
        log_file = tmp_path / "test.log"
        
        config = LoggingConfig(
            enable_file_logging=True,
            log_directory=str(tmp_path),
            log_filename="test.log",
            file_format="simple",
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log some messages
        logger.info("File logging test message 1")
        logger.warning("File logging test message 2")
        
        # Wait for file write
        time.sleep(0.1)
        
        # Check that log file was created and contains messages
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        assert "File logging test message 1" in content
        assert "File logging test message 2" in content
        assert "INFO" in content
        assert "WARNING" in content
    
    def test_file_logging_json_format(self, isolated_logging_manager, tmp_path):
        """Test file logging with JSON format."""
        log_file = tmp_path / "json_test.log"
        
        config = LoggingConfig(
            enable_file_logging=True,
            log_directory=str(tmp_path),
            log_filename="json_test.log",
            file_format="json",
            enable_hydra_integration=False,
            enable_correlation_ids=True
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log a test message
        with create_correlation_scope("file_json_test"):
            logger.info("JSON file format test")
        
        # Wait for file write
        time.sleep(0.1)
        
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Should contain JSON formatted logs
        assert '{"timestamp"' in content
        assert '"message": "JSON file format test"' in content
        assert '"correlation_id": "file_json_test"' in content
    
    def test_log_directory_creation(self, isolated_logging_manager, tmp_path):
        """Test that log directory is created if it doesn't exist."""
        nested_dir = tmp_path / "logs" / "nested" / "deep"
        
        config = LoggingConfig(
            enable_file_logging=True,
            log_directory=str(nested_dir),
            log_filename="nested.log",
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        logger.info("Directory creation test")
        
        # Wait for file write
        time.sleep(0.1)
        
        # Check that nested directory was created
        assert nested_dir.exists()
        assert (nested_dir / "nested.log").exists()
    
    def test_log_file_compression_config(self, isolated_logging_manager, tmp_path):
        """Test log file compression configuration."""
        config = LoggingConfig(
            enable_file_logging=True,
            log_directory=str(tmp_path),
            log_filename="compression_test.log",
            rotation="1 MB",
            compression="bz2",
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Configuration should be applied correctly
        assert config.compression == "bz2"
        assert config.rotation == "1 MB"
    
    def test_log_retention_config(self, isolated_logging_manager, tmp_path):
        """Test log retention configuration."""
        config = LoggingConfig(
            enable_file_logging=True,
            log_directory=str(tmp_path),
            log_filename="retention_test.log",
            retention="7 days",
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Configuration should be applied correctly
        assert config.retention == "7 days"
    
    def test_disable_file_logging(self, isolated_logging_manager, tmp_path):
        """Test that file logging can be disabled."""
        config = LoggingConfig(
            enable_file_logging=False,
            log_directory=str(tmp_path),
            log_filename="should_not_exist.log",
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        logger.info("This should not go to file")
        
        # Wait to ensure no file is created
        time.sleep(0.1)
        
        # Log file should not exist
        log_file = tmp_path / "should_not_exist.log"
        assert not log_file.exists()


# =============================================================================
# SEED MANAGER INTEGRATION TESTING
# =============================================================================

class TestSeedManagerIntegration:
    """Test integration with seed manager for reproducibility context."""
    
    def test_seed_context_integration(self, isolated_logging_manager, captured_logs, mock_seed_manager):
        """Test seed manager context integration in logging."""
        config = LoggingConfig(
            enable_seed_context=True,
            console_format="structured",
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log a message with seed context
        logger.info("Message with seed context")
        
        logs = captured_logs["get_logs"]()
        
        # Should contain seed information from mock
        assert "seed=42" in logs
        assert "Message with seed context" in logs
    
    def test_seed_context_disabled(self, isolated_logging_manager, captured_logs):
        """Test logging when seed context is disabled."""
        config = LoggingConfig(
            enable_seed_context=False,
            console_format="structured",
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        logger.info("Message without seed context")
        
        logs = captured_logs["get_logs"]()
        
        # Should contain unknown seed value when disabled
        assert "seed=unknown" in logs
        assert "Message without seed context" in logs
    
    def test_seed_context_unavailable(self, isolated_logging_manager, captured_logs):
        """Test logging when seed manager is unavailable."""
        # Mock import error for seed manager
        with patch('{{cookiecutter.project_slug}}.utils.logging.get_seed_manager', side_effect=ImportError("Seed manager not available")):
            config = LoggingConfig(
                enable_seed_context=True,
                console_format="structured", 
                enable_file_logging=False,
                enable_hydra_integration=False
            )
            
            manager = get_logging_manager()
            manager.initialize(config)
            
            logger.info("Message with unavailable seed context")
            
            logs = captured_logs["get_logs"]()
            
            # Should handle gracefully and use unknown seed
            assert "seed=unknown" in logs
            assert "Message with unavailable seed context" in logs


# =============================================================================
# PERFORMANCE MONITORING TESTING
# =============================================================================

class TestPerformanceMonitoring:
    """Test performance monitoring and threshold-based warnings."""
    
    def test_performance_threshold_warning(self, isolated_logging_manager, captured_logs):
        """Test performance threshold warning generation."""
        config = LoggingConfig(
            enable_performance_monitoring=True,
            performance_threshold_ms=10.0,  # Low threshold for testing
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Simulate a slow operation
        logger.info(
            "Slow operation completed",
            extra={
                "event_type": "parameter_validation",
                "validation_time_ms": 50.0  # Exceeds threshold
            }
        )
        
        logs = captured_logs["get_logs"]()
        
        # Should be upgraded to WARNING due to performance threshold
        assert "PERFORMANCE:" in logs or "WARNING" in logs
    
    def test_performance_monitoring_disabled(self, isolated_logging_manager, captured_logs):
        """Test that performance monitoring can be disabled."""
        config = LoggingConfig(
            enable_performance_monitoring=False,
            performance_threshold_ms=10.0,
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log a slow operation
        logger.info(
            "Slow operation completed",
            extra={
                "event_type": "parameter_validation",
                "validation_time_ms": 50.0
            }
        )
        
        logs = captured_logs["get_logs"]()
        
        # Should remain INFO level when performance monitoring is disabled
        assert "PERFORMANCE:" not in logs
        assert "INFO" in logs
    
    def test_initialization_performance_tracking(self, isolated_logging_manager, captured_logs):
        """Test that initialization performance is tracked."""
        config = LoggingConfig(
            enable_performance_monitoring=True,
            performance_threshold_ms=1.0,  # Very low threshold
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        # Mock slow initialization
        with patch('time.perf_counter', side_effect=[0.0, 0.15]):  # 150ms
            manager = get_logging_manager()
            manager.initialize(config)
        
        logs = captured_logs["get_logs"]()
        
        # Should warn about slow initialization
        assert "initialization exceeded performance threshold" in logs.lower()


# =============================================================================
# EXCEPTION HANDLING AND DIAGNOSTICS TESTING
# =============================================================================

class TestExceptionHandling:
    """Test exception handling and diagnostic capabilities."""
    
    def test_exception_diagnostics_enabled(self, isolated_logging_manager, captured_logs):
        """Test exception diagnostics when enabled."""
        config = LoggingConfig(
            enable_exception_diagnostics=True,
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log an exception
        try:
            raise ValueError("Test exception for diagnostics")
        except ValueError as e:
            logger.exception("Exception occurred during test")
        
        logs = captured_logs["get_logs"]()
        
        # Should contain exception details and traceback
        assert "Exception occurred during test" in logs
        assert "ValueError: Test exception for diagnostics" in logs
        assert "Traceback" in logs or "Exception" in logs
    
    def test_exception_diagnostics_disabled(self, isolated_logging_manager, captured_logs):
        """Test exception handling when diagnostics are disabled."""
        config = LoggingConfig(
            enable_exception_diagnostics=False,
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Log an exception
        try:
            raise ValueError("Test exception without diagnostics")
        except ValueError as e:
            logger.exception("Exception occurred during test")
        
        logs = captured_logs["get_logs"]()
        
        # Should contain basic exception message but minimal diagnostics
        assert "Exception occurred during test" in logs


# =============================================================================
# CONVENIENCE FUNCTIONS AND MODULE LOGGER TESTING
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions and module-specific loggers."""
    
    def test_setup_enhanced_logging_function(self, isolated_logging_manager, captured_logs):
        """Test setup_enhanced_logging convenience function."""
        config = LoggingConfig(
            level="DEBUG",
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        # Use convenience function
        setup_enhanced_logging(config)
        
        logger.info("Setup function test")
        
        logs = captured_logs["get_logs"]()
        assert "Setup function test" in logs
        assert "Enhanced logging system initialized successfully" in logs
    
    def test_get_module_logger(self, isolated_logging_manager, captured_logs):
        """Test module-specific logger creation."""
        config = LoggingConfig(
            enable_file_logging=False,
            enable_hydra_integration=False
        )
        
        setup_enhanced_logging(config)
        
        # Get module logger
        module_logger = get_module_logger("test_module")
        module_logger.info("Module-specific message")
        
        records = captured_logs["get_records"]()
        
        # Find the module-specific record
        module_record = None
        for record in records:
            if "Module-specific message" in record["message"]:
                module_record = record
                break
        
        assert module_record is not None
        assert module_record["extra"]["module"] == "test_module"
    
    def test_setup_enhanced_logging_with_dict(self, isolated_logging_manager, captured_logs):
        """Test setup_enhanced_logging with dictionary configuration."""
        config_dict = {
            "level": "WARNING",
            "console_format": "simple",
            "enable_file_logging": False,
            "enable_hydra_integration": False
        }
        
        setup_enhanced_logging(config_dict)
        
        # Test that configuration was applied
        logger.info("This should not appear")  # Below WARNING
        logger.warning("This should appear")
        
        logs = captured_logs["get_logs"]()
        assert "This should not appear" not in logs
        assert "This should appear" in logs
    
    def test_setup_enhanced_logging_with_none(self, isolated_logging_manager, captured_logs):
        """Test setup_enhanced_logging with no configuration (defaults)."""
        setup_enhanced_logging(None)
        
        logger.info("Default configuration test")
        
        logs = captured_logs["get_logs"]()
        assert "Default configuration test" in logs
        assert "Enhanced logging system initialized successfully" in logs


# =============================================================================
# INTEGRATION AND END-TO-END TESTING
# =============================================================================

class TestIntegrationScenarios:
    """Test complete integration scenarios and workflows."""
    
    def test_complete_workflow_simulation(self, isolated_logging_manager, captured_logs, mock_hydra_global, mock_seed_manager, environment_variables):
        """Test complete workflow with all features enabled."""
        mock_global, mock_instance, mock_hydra_cfg, mock_runtime_cfg = mock_hydra_global
        
        # Complete configuration with all features
        config = LoggingConfig(
            level="DEBUG",
            console_format="structured",
            file_format="json",
            enable_file_logging=False,  # Disable for test simplicity
            enable_hydra_integration=True,
            enable_seed_context=True,
            enable_cli_metrics=True,
            enable_config_tracking=True,
            enable_correlation_ids=True,
            enable_performance_monitoring=True,
            enable_environment_logging=True,
            performance_threshold_ms=50.0
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Simulate complete workflow
        with create_correlation_scope("workflow_123") as corr_id:
            with create_cli_command_scope("simulate") as cmd_metrics:
                with create_parameter_validation_scope() as val_metrics:
                    # Simulate parameter validation
                    val_metrics["parameters_validated"] = 15
                    logger.info("Parameters validated successfully")
                
                # Simulate main execution
                cmd_metrics["simulation_steps"] = 1000
                cmd_metrics["convergence_achieved"] = True
                logger.info("Simulation completed successfully")
                
                # Log environment variable usage
                log_environment_variables(["EXPERIMENT_NAME", "API_KEY"])
                
                # Log configuration composition
                log_configuration_composition({
                    "base_config": "base.yaml",
                    "experiment_config": "test_experiment.yaml",
                    "overrides": ["seed=42", "max_steps=1000"]
                })
        
        logs = captured_logs["get_logs"]()
        records = captured_logs["get_records"]()
        
        # Verify complete workflow was logged
        assert "Parameters validated successfully" in logs
        assert "Simulation completed successfully" in logs
        assert "CLI command completed: simulate" in logs
        assert "corr=workflow_123" in logs
        assert "seed=42" in logs
        assert "job=test_job" in logs
        
        # Verify event types were logged
        event_types = [r.get("extra", {}).get("event_type") for r in records]
        assert "parameter_validation" in event_types
        assert "cli_command_complete" in event_types
        assert "environment_access" in event_types
        assert "config_composition" in event_types
    
    def test_research_workflow_compatibility(self, isolated_logging_manager, tmp_path, mock_seed_manager):
        """Test compatibility with research workflow requirements."""
        log_file = tmp_path / "research.log"
        
        # Research-focused configuration
        config = LoggingConfig(
            level="INFO",
            console_format="simple",  # Clean console for researchers
            file_format="json",       # Structured data for analysis
            enable_file_logging=True,
            log_directory=str(tmp_path),
            log_filename="research.log",
            enable_hydra_integration=False,  # Simplified for research
            enable_seed_context=True,        # Important for reproducibility
            enable_cli_metrics=True,         # Track experiment execution
            enable_config_tracking=True,     # Track configuration changes
            enable_correlation_ids=True,     # Link related operations
            enable_performance_monitoring=True,  # Performance analysis
            performance_threshold_ms=100.0
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Simulate research workflow
        with create_correlation_scope("experiment_001"):
            logger.info("Starting experiment", extra={
                "experiment_id": "exp_001",
                "algorithm": "gradient_ascent",
                "dataset": "wind_tunnel_data"
            })
            
            logger.info("Training iteration completed", extra={
                "iteration": 50,
                "loss": 0.123,
                "accuracy": 0.876
            })
            
            logger.info("Experiment completed", extra={
                "final_accuracy": 0.912,
                "total_iterations": 100,
                "convergence_time": 45.2
            })
        
        # Wait for file write
        time.sleep(0.1)
        
        # Verify research data is properly logged
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Should contain JSON formatted research data
        assert '"experiment_id": "exp_001"' in content
        assert '"algorithm": "gradient_ascent"' in content
        assert '"final_accuracy": 0.912' in content
        assert '"correlation_id": "experiment_001"' in content
    
    def test_production_deployment_scenario(self, isolated_logging_manager, tmp_path, environment_variables):
        """Test production deployment logging scenario."""
        log_file = tmp_path / "production.log"
        
        # Production configuration
        config = LoggingConfig(
            level="WARNING",  # Minimal logging in production
            console_format="structured",
            file_format="json",
            enable_file_logging=True,
            log_directory=str(tmp_path),
            log_filename="production.log",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            enable_hydra_integration=True,
            enable_seed_context=True,
            enable_cli_metrics=True,
            enable_config_tracking=True,
            enable_correlation_ids=True,
            enable_performance_monitoring=True,
            performance_threshold_ms=200.0,  # Higher threshold for production
            enable_exception_diagnostics=False,  # Disable for security
            enable_environment_logging=True
        )
        
        manager = get_logging_manager()
        manager.initialize(config)
        
        # Simulate production operations
        logger.info("Info message - should not appear in production")
        logger.warning("Production warning - resource utilization high")
        logger.error("Production error - processing failed for batch 123")
        
        # Log environment access (should mask sensitive values)
        log_environment_variables(["API_KEY", "DB_PASSWORD"])
        
        # Wait for file write
        time.sleep(0.1)
        
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Only warnings and errors should appear
        assert "Info message" not in content
        assert "Production warning" in content
        assert "Production error" in content
        
        # Sensitive values should be masked
        assert '"API_KEY": "***456"' in content
        assert '"DB_PASSWORD": "***123"' in content


if __name__ == "__main__":
    pytest.main([__file__])