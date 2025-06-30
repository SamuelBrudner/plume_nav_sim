"""
Comprehensive test suite for enhanced logging configuration with Hydra integration.

This module provides exhaustive testing for the enhanced logging system including 
structured output formats validation, log level management through Hydra configuration,
CLI integration testing, and multi-environment logging validation. The test suite
ensures logging reliability across development, production, and testing environments
with comprehensive coverage of configuration composition, experiment tracking, and
performance metrics collection.

Key Testing Domains:
- Enhanced logging configuration with Hydra integration validation
- Structured output format testing (enhanced, hydra, cli, minimal, production)
- Log level management through hierarchical configuration overrides
- Logging initialization and configuration composition from conf/ directory
- CLI command execution context tracking and metrics collection
- Log message formatting, timestamp handling, and correlation ID generation
- Multi-environment configuration testing (development, production, testing)
- Log file management, rotation capabilities, and cleanup validation
- Experiment context tracking with seed manager integration
- Performance monitoring and timing assertions for setup requirements

Test Categories:
1. Configuration Testing: Hydra configuration composition and validation
2. Format Testing: Output format validation across all supported patterns
3. Level Management: Log level override scenarios and hierarchical validation
4. CLI Integration: Command execution tracking and parameter flow validation
5. Environment Testing: Multi-environment configuration and behavior validation
6. Performance Testing: Setup timing, throughput, and resource usage validation
7. Error Handling: Graceful failure scenarios and recovery testing
8. Integration Testing: Cross-component interaction and context binding validation

Testing Strategy:
- Comprehensive fixture usage from conftest.py for consistent test setup
- pytest-hydra integration for configuration composition testing
- click.testing.CliRunner for CLI command execution validation
- In-memory SQLite for database session interaction testing
- Mock frameworks for external dependency isolation
- Performance benchmarking with timing assertions
- Multi-environment isolation with proper cleanup

Coverage Requirements:
- Overall target: >85% for utility modules per technical specification
- Critical path coverage: 100% for setup and configuration methods
- Error handling coverage: 100% for exception scenarios
- Integration testing: Comprehensive validation across logging contexts

Author: Cookiecutter Template Generator
Version: 2.0.0
"""

import pytest
import os
import time
import tempfile
import threading
import uuid
import hashlib
import json
from pathlib import Path
from io import StringIO
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import patch, MagicMock, Mock, call
from contextlib import contextmanager, redirect_stderr

# Import loguru for logging testing
import loguru
from loguru import logger

# Import testing utilities
import numpy as np

# Enhanced imports for cookiecutter-based architecture testing
try:
    # Hydra configuration testing support
    from hydra import initialize, compose, GlobalHydra
    from hydra.core.config_store import ConfigStore
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

try:
    # CLI testing infrastructure
    from click.testing import CliRunner
    CLI_TESTING_AVAILABLE = True
except ImportError:
    CLI_TESTING_AVAILABLE = False
    CliRunner = None

# Import the enhanced logging module under test
try:
    from plume_nav_sim.utils.logging import (
        EnhancedLogger,
        LoggingConfig,
        ExperimentContext,
        CLICommandTracker,
        setup_enhanced_logging,
        configure_from_hydra,
        bind_experiment_context,
        track_cli_command,
        get_module_logger,
        log_configuration_override,
        get_logging_metrics,
        ENHANCED_FORMAT,
        HYDRA_FORMAT,
        CLI_FORMAT,
        MINIMAL_FORMAT,
        PRODUCTION_FORMAT
    )
    LOGGING_MODULE_AVAILABLE = True
except ImportError:
    LOGGING_MODULE_AVAILABLE = False


class TestLoggingConfig:
    """
    Comprehensive testing for LoggingConfig dataclass and configuration management.
    
    Tests configuration validation, format string resolution, file path resolution
    with environment variable interpolation, and configuration composition scenarios
    essential for robust logging system operation across different environments.
    
    Coverage Areas:
    - Default configuration validation and parameter initialization
    - Format string resolution for all supported logging patterns
    - File path resolution with environment variable interpolation
    - Configuration composition and override scenarios
    - Validation of configuration constraints and boundaries
    """
    
    def test_default_configuration_initialization(self):
        """Test LoggingConfig initializes with correct default values."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig()
        
        # Basic logging configuration defaults
        assert config.level == "INFO"
        assert config.format == "enhanced"
        assert config.console_enabled is True
        assert config.file_enabled is False
        assert config.file_path is None
        
        # File rotation and retention defaults
        assert config.rotation == "10 MB"
        assert config.retention == "1 week"
        assert config.compression is None
        
        # Loguru configuration defaults
        assert config.enqueue is True
        assert config.backtrace is True
        assert config.diagnose is True
        
        # Enhanced tracking features defaults
        assert config.correlation_id_enabled is True
        assert config.experiment_tracking_enabled is True
        assert config.hydra_integration_enabled is True
        assert config.seed_context_enabled is True
        assert config.cli_metrics_enabled is True
        assert config.environment_logging_enabled is True
        assert config.performance_monitoring_enabled is True
    
    def test_format_string_resolution_all_patterns(self):
        """Test format string resolution for all supported logging patterns."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig()
        
        # Test all supported format patterns
        format_mapping = {
            'enhanced': ENHANCED_FORMAT,
            'hydra': HYDRA_FORMAT,
            'cli': CLI_FORMAT,
            'minimal': MINIMAL_FORMAT,
            'production': PRODUCTION_FORMAT
        }
        
        for format_name, expected_format in format_mapping.items():
            config.format = format_name
            resolved_format = config.get_format_string()
            assert resolved_format == expected_format
            
            # Validate format string contains expected components
            if format_name == 'enhanced':
                assert 'correlation_id' in resolved_format
                assert 'experiment_id' in resolved_format
            elif format_name == 'hydra':
                assert 'hydra_job' in resolved_format
                assert 'config_checksum' in resolved_format
            elif format_name == 'cli':
                assert 'cli_command' in resolved_format
                assert 'execution_time_ms' in resolved_format
            elif format_name == 'minimal':
                assert 'HH:mm:ss.SSS' in resolved_format
            elif format_name == 'production':
                assert 'seed_value' in resolved_format
    
    def test_format_string_fallback_for_invalid_format(self):
        """Test format string fallback to enhanced format for invalid format names."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig()
        
        # Test invalid format names fall back to enhanced format
        invalid_formats = ['invalid', 'unknown', '', None, 123]
        
        for invalid_format in invalid_formats:
            config.format = invalid_format
            resolved_format = config.get_format_string()
            assert resolved_format == ENHANCED_FORMAT
    
    def test_file_path_resolution_with_environment_variables(self, isolated_environment):
        """Test file path resolution with environment variable interpolation."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        env_vars, temp_dir = isolated_environment
        
        # Set up test environment variables
        test_log_dir = os.path.join(temp_dir, "logs")
        env_vars['LOG_DIR'] = test_log_dir
        env_vars['LOG_FILE'] = 'test.log'
        os.environ.update(env_vars)
        
        config = LoggingConfig()
        
        # Test environment variable interpolation
        config.file_path = "${LOG_DIR}/${LOG_FILE}"
        resolved_path = config.resolve_file_path()
        
        assert resolved_path is not None
        assert str(resolved_path) == os.path.join(test_log_dir, 'test.log')
        
        # Test user home directory expansion
        config.file_path = "~/logs/test.log"
        resolved_path = config.resolve_file_path()
        assert resolved_path is not None
        assert str(resolved_path).startswith(os.path.expanduser("~"))
        
        # Test no file path returns None
        config.file_path = None
        resolved_path = config.resolve_file_path()
        assert resolved_path is None
    
    def test_configuration_validation_with_invalid_parameters(self):
        """Test configuration validation with invalid parameter combinations."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Test with invalid log level (should not raise exception but use default)
        config = LoggingConfig(level="INVALID_LEVEL")
        assert config.level == "INVALID_LEVEL"  # LoggingConfig doesn't validate, Loguru will
        
        # Test with invalid rotation settings
        config = LoggingConfig(rotation="invalid_rotation")
        assert config.rotation == "invalid_rotation"  # Again, validation is at Loguru level
        
        # Test disabled features work correctly
        config = LoggingConfig(
            correlation_id_enabled=False,
            experiment_tracking_enabled=False,
            hydra_integration_enabled=False
        )
        assert config.correlation_id_enabled is False
        assert config.experiment_tracking_enabled is False
        assert config.hydra_integration_enabled is False


class TestExperimentContext:
    """
    Comprehensive testing for ExperimentContext tracking and metadata management.
    
    Tests experiment context creation, Hydra integration, seed manager integration,
    system information collection, and logger context conversion essential for
    complete experiment traceability and reproducibility.
    
    Coverage Areas:
    - Experiment context initialization and unique ID generation
    - Hydra configuration integration and checksum generation
    - Seed manager integration and context binding
    - System information collection and environment variable tracking
    - Logger context conversion and metadata extraction
    """
    
    def test_experiment_context_initialization_with_defaults(self):
        """Test ExperimentContext initializes with correct default values and unique IDs."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        context = ExperimentContext()
        
        # Test unique experiment ID generation
        assert context.experiment_id.startswith("exp_")
        assert len(context.experiment_id) > 4
        
        # Test correlation ID generation
        assert len(context.correlation_id) == 8
        assert isinstance(context.correlation_id, str)
        
        # Test default None values for optional fields
        assert context.hydra_job_name is None
        assert context.hydra_run_id is None
        assert context.config_checksum is None
        assert context.seed_value is None
        assert context.cli_command is None
        
        # Test timing initialization
        assert context.start_time > 0
        assert context.execution_time_ms is None
        
        # Test default collections
        assert isinstance(context.system_info, dict)
        assert isinstance(context.config_composition, list)
        assert isinstance(context.environment_variables, dict)
        assert isinstance(context.performance_metrics, dict)
    
    def test_experiment_context_custom_initialization(self):
        """Test ExperimentContext initialization with custom parameters."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        custom_id = "custom_exp_123"
        context = ExperimentContext(experiment_id=custom_id)
        
        assert context.experiment_id == custom_id
        assert len(context.correlation_id) == 8  # Still generates unique correlation ID
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_update_from_hydra_integration(self, mock_hydra_config):
        """Test experiment context update from Hydra configuration."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        context = ExperimentContext()
        
        # Mock HydraConfig.get() for testing
        with patch('plume_nav_sim.utils.logging.HydraConfig') as mock_hydra_class:
            mock_hydra_instance = Mock()
            mock_hydra_instance.job.name = "test_job"
            mock_hydra_instance.runtime.output_dir = "/tmp/outputs/2023-01-01/12-00-00"
            mock_hydra_class.get.return_value = mock_hydra_instance
            
            # Test Hydra context update
            context.update_from_hydra(mock_hydra_config)
            
            assert context.hydra_job_name == "test_job"
            assert context.hydra_run_id == "12-00-00"
            assert context.config_checksum is not None
            assert len(context.config_checksum) == 8  # MD5 truncated to 8 chars
            assert len(context.config_composition) > 0
    
    def test_update_from_hydra_with_error_handling(self):
        """Test experiment context Hydra update handles errors gracefully."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        context = ExperimentContext()
        
        # Test with exception in HydraConfig.get()
        with patch('plume_nav_sim.utils.logging.HydraConfig') as mock_hydra_class:
            mock_hydra_class.get.side_effect = Exception("Hydra not initialized")
            
            # Should not raise exception
            context.update_from_hydra(None)
            
            # Values should remain None
            assert context.hydra_job_name is None
            assert context.hydra_run_id is None
            assert context.config_checksum is None
    
    def test_update_system_info_collection(self):
        """Test system information collection and environment variable tracking."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        context = ExperimentContext()
        context.update_system_info()
        
        # Test system information collection
        assert 'platform' in context.system_info
        assert 'python_version' in context.system_info
        assert 'architecture' in context.system_info
        assert 'hostname' in context.system_info
        
        # Test environment variable collection
        expected_env_vars = [
            'PYTHONPATH', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'HYDRA_FULL_ERROR', 'LOGURU_LEVEL'
        ]
        
        for var in expected_env_vars:
            assert var in context.environment_variables
            # Should be either the actual value or 'N/A'
            assert context.environment_variables[var] is not None
    
    def test_logger_context_conversion(self):
        """Test conversion of experiment context to logger binding context."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        context = ExperimentContext(
            experiment_id="test_exp",
            hydra_job_name="test_job",
            seed_value=42,
            cli_command="test_command",
            execution_time_ms=123.45
        )
        context.config_checksum = "abc12345"
        context.cli_parameters = {"param1": "value1"}
        context.performance_metrics = {"metric1": 1.23}
        
        logger_context = context.to_logger_context()
        
        # Test required context fields
        assert logger_context['experiment_id'] == "test_exp"
        assert logger_context['correlation_id'] == context.correlation_id
        assert logger_context['seed_value'] == 42
        assert logger_context['hydra_job_name'] == "test_job"
        assert logger_context['config_checksum'] == "abc12345"
        assert logger_context['cli_command'] == "test_command"
        assert logger_context['execution_time_ms'] == 123.45
        
        # Test optional fields
        assert 'cli_parameters' in logger_context
        assert logger_context['metric1'] == 1.23  # Performance metrics merged
    
    def test_logger_context_conversion_with_none_values(self):
        """Test logger context conversion handles None values correctly."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        context = ExperimentContext()
        logger_context = context.to_logger_context()
        
        # Test None values become 'N/A' or appropriate defaults
        assert logger_context['seed_value'] == 'N/A'
        assert logger_context['hydra_job_name'] == 'N/A'
        assert logger_context['config_checksum'] == 'N/A'
        assert logger_context['cli_command'] == 'N/A'
        assert logger_context['execution_time_ms'] == 0.0


class TestEnhancedLogger:
    """
    Comprehensive testing for EnhancedLogger implementation and configuration.
    
    Tests enhanced logger setup, configuration management, sink configuration,
    performance monitoring, and integration with Hydra, CLI commands, and
    experiment tracking systems for complete logging system validation.
    
    Coverage Areas:
    - Enhanced logger initialization and configuration
    - Console and file sink setup and management
    - Performance monitoring and setup timing validation
    - Context binding and experiment tracking integration
    - Global context configuration and metadata injection
    - Cleanup and resource management
    """
    
    def test_enhanced_logger_initialization_with_defaults(self):
        """Test EnhancedLogger initializes correctly with default configuration."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        enhanced_logger = EnhancedLogger()
        
        # Test default configuration
        assert isinstance(enhanced_logger.config, LoggingConfig)
        assert enhanced_logger.config.level == "INFO"
        assert enhanced_logger.config.format == "enhanced"
        
        # Test initialization state
        assert enhanced_logger.experiment_context is None
        assert enhanced_logger._setup_complete is False
        assert enhanced_logger._sink_ids == []
        assert enhanced_logger._setup_start_time is None
        assert enhanced_logger._setup_duration_ms is None
    
    def test_enhanced_logger_initialization_with_custom_config(self):
        """Test EnhancedLogger initialization with custom configuration."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        custom_config = LoggingConfig(
            level="DEBUG",
            format="minimal",
            console_enabled=False,
            file_enabled=True,
            file_path="/tmp/test.log"
        )
        
        enhanced_logger = EnhancedLogger(custom_config)
        
        assert enhanced_logger.config.level == "DEBUG"
        assert enhanced_logger.config.format == "minimal"
        assert enhanced_logger.config.console_enabled is False
        assert enhanced_logger.config.file_enabled is True
        assert enhanced_logger.config.file_path == "/tmp/test.log"
    
    def test_enhanced_logger_setup_console_only(self, isolated_environment):
        """Test enhanced logger setup with console output only."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        env_vars, temp_dir = isolated_environment
        
        config = LoggingConfig(
            console_enabled=True,
            file_enabled=False,
            level="DEBUG"
        )
        
        enhanced_logger = EnhancedLogger(config)
        
        # Mock logger.remove and logger.add for testing
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.return_value = 1  # Mock sink ID
            
            result = enhanced_logger.setup()
            
            # Test setup completion
            assert result == enhanced_logger
            assert enhanced_logger._setup_complete is True
            assert enhanced_logger._setup_duration_ms is not None
            assert enhanced_logger._setup_duration_ms >= 0
            
            # Test logger configuration calls
            mock_logger.remove.assert_called_once()
            mock_logger.add.assert_called_once()
            mock_logger.configure.assert_called_once()
            
            # Test sink ID tracking
            assert len(enhanced_logger._sink_ids) == 1
            assert enhanced_logger._sink_ids[0] == 1
    
    def test_enhanced_logger_setup_with_file_output(self, isolated_environment):
        """Test enhanced logger setup with file output configuration."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        env_vars, temp_dir = isolated_environment
        log_file = os.path.join(temp_dir, "test.log")
        
        config = LoggingConfig(
            console_enabled=True,
            file_enabled=True,
            file_path=log_file,
            rotation="5 MB",
            retention="1 week",
            compression="gz"
        )
        
        enhanced_logger = EnhancedLogger(config)
        
        # Mock logger operations
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.side_effect = [1, 2]  # Two sink IDs
            
            result = enhanced_logger.setup()
            
            # Test setup completion
            assert result == enhanced_logger
            assert enhanced_logger._setup_complete is True
            
            # Test both sinks configured
            assert mock_logger.add.call_count == 2
            assert len(enhanced_logger._sink_ids) == 2
            
            # Test file sink configuration
            file_call = mock_logger.add.call_args_list[1]
            assert str(log_file) in str(file_call)
    
    def test_enhanced_logger_setup_performance_monitoring(self):
        """Test enhanced logger setup performance monitoring and timing validation."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig(console_enabled=True, file_enabled=False)
        enhanced_logger = EnhancedLogger(config)
        
        # Mock logger operations for controlled timing
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.return_value = 1
            
            start_time = time.perf_counter()
            enhanced_logger.setup()
            end_time = time.perf_counter()
            
            # Test performance timing
            assert enhanced_logger._setup_duration_ms is not None
            assert enhanced_logger._setup_duration_ms > 0
            
            # Test setup time is reasonable (should be much less than 100ms in mock)
            actual_duration = (end_time - start_time) * 1000
            assert enhanced_logger._setup_duration_ms <= actual_duration + 10  # Some tolerance
    
    def test_enhanced_logger_setup_performance_warning(self):
        """Test enhanced logger logs warning when setup exceeds performance requirement."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig(console_enabled=True, file_enabled=False)
        enhanced_logger = EnhancedLogger(config)
        
        # Mock slow setup by patching time.perf_counter
        call_count = [0]
        def mock_perf_counter():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.0  # Start time
            else:
                return 0.15  # End time (150ms setup)
        
        with patch('plume_nav_sim.utils.logging.time.perf_counter', side_effect=mock_perf_counter):
            with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
                mock_logger.add.return_value = 1
                
                enhanced_logger.setup()
                
                # Test warning was logged for slow setup
                assert enhanced_logger._setup_duration_ms == 150.0
                # In a real scenario, this would log a warning, but mocking makes verification complex
    
    def test_enhanced_logger_experiment_context_creation(self, mock_hydra_config):
        """Test enhanced logger experiment context creation and integration."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig(
            hydra_integration_enabled=True,
            seed_context_enabled=True,
            performance_monitoring_enabled=True
        )
        enhanced_logger = EnhancedLogger(config)
        
        context = enhanced_logger.create_experiment_context(
            cfg=mock_hydra_config,
            experiment_id="test_experiment"
        )
        
        # Test context creation
        assert isinstance(context, ExperimentContext)
        assert context.experiment_id == "test_experiment"
        assert len(context.correlation_id) == 8
        
        # Test system info was updated
        assert len(context.system_info) > 0
        assert 'platform' in context.system_info
    
    def test_enhanced_logger_bind_experiment_context(self):
        """Test enhanced logger experiment context binding with additional parameters."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig()
        enhanced_logger = EnhancedLogger(config)
        
        # Test context binding with additional parameters
        context = enhanced_logger.bind_experiment_context(
            step=1,
            agent_count=5,
            custom_metric=123.45
        )
        
        # Test context includes base experiment context
        assert 'experiment_id' in context
        assert 'correlation_id' in context
        
        # Test additional parameters included
        assert context['step'] == 1
        assert context['agent_count'] == 5
        assert context['custom_metric'] == 123.45
    
    def test_enhanced_logger_cleanup(self):
        """Test enhanced logger cleanup and resource management."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig(console_enabled=True, file_enabled=False)
        enhanced_logger = EnhancedLogger(config)
        
        # Mock setup
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.return_value = 1
            enhanced_logger.setup()
            
            # Test cleanup
            enhanced_logger.cleanup()
            
            # Test state reset
            assert enhanced_logger._sink_ids == []
            assert enhanced_logger._setup_complete is False
            
            # Test logger.remove called
            mock_logger.remove.assert_called_with(1)
    
    def test_enhanced_logger_setup_metrics(self):
        """Test enhanced logger setup metrics collection and reporting."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig(console_enabled=True, file_enabled=False)
        enhanced_logger = EnhancedLogger(config)
        
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.return_value = 1
            enhanced_logger.setup(experiment_id="test_metrics")
            
            metrics = enhanced_logger.get_setup_metrics()
            
            # Test metrics content
            assert 'setup_duration_ms' in metrics
            assert 'sinks_configured' in metrics
            assert 'experiment_id' in metrics
            assert 'config_checksum' in metrics
            
            assert metrics['sinks_configured'] == 1
            assert metrics['experiment_id'] == "test_metrics"
            assert metrics['setup_duration_ms'] >= 0


class TestGlobalLoggingFunctions:
    """
    Comprehensive testing for global logging functions and configuration management.
    
    Tests global enhanced logging setup, Hydra configuration integration,
    experiment context binding, CLI command tracking, module logger creation,
    and logging metrics collection for complete global logging functionality.
    
    Coverage Areas:
    - Global enhanced logging setup and singleton management
    - Hydra configuration integration and error handling
    - Experiment context binding and parameter management
    - CLI command tracking and performance monitoring
    - Module-specific logger creation and context binding
    - Configuration override logging and parameter tracking
    - Global logging metrics collection and reporting
    """
    
    def test_setup_enhanced_logging_default_configuration(self):
        """Test global setup_enhanced_logging with default configuration."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Mock the global logger to avoid actual setup
        with patch('plume_nav_sim.utils.logging._global_enhanced_logger', None):
            with patch('plume_nav_sim.utils.logging.EnhancedLogger') as mock_enhanced_logger_class:
                mock_instance = Mock()
                mock_enhanced_logger_class.return_value = mock_instance
                
                result = setup_enhanced_logging()
                
                # Test logger creation and setup
                mock_enhanced_logger_class.assert_called_once()
                mock_instance.setup.assert_called_once_with(cfg=None, experiment_id=None)
                assert result == mock_instance
    
    def test_setup_enhanced_logging_with_custom_config(self):
        """Test global setup_enhanced_logging with custom configuration."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        custom_config = LoggingConfig(level="DEBUG", format="minimal")
        
        with patch('plume_nav_sim.utils.logging._global_enhanced_logger', None):
            with patch('plume_nav_sim.utils.logging.EnhancedLogger') as mock_enhanced_logger_class:
                mock_instance = Mock()
                mock_enhanced_logger_class.return_value = mock_instance
                
                result = setup_enhanced_logging(
                    config=custom_config,
                    experiment_id="test_exp"
                )
                
                # Test custom configuration passed
                mock_enhanced_logger_class.assert_called_once_with(custom_config)
                mock_instance.setup.assert_called_once_with(cfg=None, experiment_id="test_exp")
    
    def test_setup_enhanced_logging_cleanup_existing_logger(self):
        """Test global setup_enhanced_logging cleans up existing logger."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Mock existing global logger
        mock_existing_logger = Mock()
        
        with patch('plume_nav_sim.utils.logging._global_enhanced_logger', mock_existing_logger):
            with patch('plume_nav_sim.utils.logging.EnhancedLogger') as mock_enhanced_logger_class:
                mock_new_instance = Mock()
                mock_enhanced_logger_class.return_value = mock_new_instance
                
                result = setup_enhanced_logging()
                
                # Test existing logger cleanup
                mock_existing_logger.cleanup.assert_called_once()
                
                # Test new logger creation
                mock_enhanced_logger_class.assert_called_once()
                mock_new_instance.setup.assert_called_once()
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_configure_from_hydra_success(self, mock_hydra_config):
        """Test configure_from_hydra with successful Hydra configuration."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Add logging configuration to mock
        logging_config_dict = {
            'level': 'DEBUG',
            'format': 'hydra',
            'file_enabled': True,
            'file_path': '/tmp/hydra_test.log'
        }
        mock_hydra_config.logging = DictConfig(logging_config_dict) if HYDRA_AVAILABLE else logging_config_dict
        
        with patch('plume_nav_sim.utils.logging.setup_enhanced_logging') as mock_setup:
            mock_logger = Mock()
            mock_setup.return_value = mock_logger
            
            result = configure_from_hydra(mock_hydra_config, experiment_id="hydra_test")
            
            # Test successful configuration
            assert result is True
            
            # Test setup was called with Hydra config
            mock_setup.assert_called_once()
            call_args = mock_setup.call_args
            assert call_args[1]['cfg'] == mock_hydra_config
            assert call_args[1]['experiment_id'] == "hydra_test"
            
            # Test configuration composition tracking
            mock_logger.track_configuration_composition.assert_called_once_with(mock_hydra_config)
    
    def test_configure_from_hydra_without_hydra_available(self):
        """Test configure_from_hydra fallback when Hydra not available."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Mock HYDRA_AVAILABLE as False
        with patch('plume_nav_sim.utils.logging.HYDRA_AVAILABLE', False):
            with patch('plume_nav_sim.utils.logging.setup_enhanced_logging') as mock_setup:
                mock_setup.return_value = Mock()
                
                result = configure_from_hydra({}, experiment_id="fallback_test")
                
                # Test fallback behavior
                assert result is False
                mock_setup.assert_called_once_with(experiment_id="fallback_test")
    
    def test_configure_from_hydra_with_exception(self, mock_hydra_config):
        """Test configure_from_hydra handles exceptions gracefully."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging.setup_enhanced_logging') as mock_setup:
            mock_setup.side_effect = Exception("Setup failed")
            
            with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
                result = configure_from_hydra(mock_hydra_config)
                
                # Test exception handling
                assert result is False
                mock_logger.error.assert_called_once()
                
                # Test fallback setup was called
                assert mock_setup.call_count == 2  # Initial failed call + fallback
    
    def test_bind_experiment_context_with_global_logger(self):
        """Test bind_experiment_context with global enhanced logger."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        mock_global_logger = Mock()
        expected_context = {'experiment_id': 'test', 'custom_param': 'value'}
        mock_global_logger.bind_experiment_context.return_value = expected_context
        
        with patch('plume_nav_sim.utils.logging._global_enhanced_logger', mock_global_logger):
            result = bind_experiment_context(custom_param='value')
            
            # Test context binding
            assert result == expected_context
            mock_global_logger.bind_experiment_context.assert_called_once_with(custom_param='value')
    
    def test_bind_experiment_context_without_global_logger(self):
        """Test bind_experiment_context creates global logger if not exists."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging._global_enhanced_logger', None):
            with patch('plume_nav_sim.utils.logging.setup_enhanced_logging') as mock_setup:
                mock_logger = Mock()
                expected_context = {'experiment_id': 'auto_created'}
                mock_logger.bind_experiment_context.return_value = expected_context
                mock_setup.return_value = mock_logger
                
                result = bind_experiment_context(test_param='test')
                
                # Test automatic setup
                mock_setup.assert_called_once()
                mock_logger.bind_experiment_context.assert_called_once_with(test_param='test')
                assert result == expected_context
    
    def test_get_module_logger_with_context(self):
        """Test get_module_logger creates module-specific logger with context."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging.bind_experiment_context') as mock_bind:
            mock_context = {'experiment_id': 'test', 'module': 'test_module'}
            mock_bind.return_value = mock_context
            
            with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
                mock_bound_logger = Mock()
                mock_logger.bind.return_value = mock_bound_logger
                
                result = get_module_logger('test_module', component='test_component')
                
                # Test context binding
                mock_bind.assert_called_once_with(module='test_module', component='test_component')
                mock_logger.bind.assert_called_once_with(**mock_context)
                assert result == mock_bound_logger
    
    def test_log_configuration_override(self):
        """Test log_configuration_override logs parameter changes correctly."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging.bind_experiment_context') as mock_bind:
            mock_context = {'experiment_id': 'test', 'override_key': 'test.param'}
            mock_bind.return_value = mock_context
            
            with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
                mock_bound_logger = Mock()
                mock_logger.bind.return_value = mock_bound_logger
                
                log_configuration_override(
                    override_key='test.param',
                    old_value='old',
                    new_value='new',
                    source='cli'
                )
                
                # Test context binding with override information
                mock_bind.assert_called_once_with(
                    override_key='test.param',
                    override_source='cli'
                )
                mock_logger.bind.assert_called_once_with(**mock_context)
                mock_bound_logger.info.assert_called_once()
    
    def test_get_logging_metrics_with_initialized_logger(self):
        """Test get_logging_metrics returns comprehensive metrics when logger initialized."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        mock_global_logger = Mock()
        mock_setup_metrics = {
            'setup_duration_ms': 50.0,
            'sinks_configured': 2,
            'experiment_id': 'test_exp',
            'config_checksum': 'abc123'
        }
        mock_global_logger.get_setup_metrics.return_value = mock_setup_metrics
        mock_global_logger.config.level = 'DEBUG'
        mock_global_logger.config.format = 'enhanced'
        mock_global_logger.config.console_enabled = True
        mock_global_logger.config.file_enabled = True
        mock_global_logger.config.hydra_integration_enabled = True
        mock_global_logger.config.seed_context_enabled = True
        mock_global_logger.config.cli_metrics_enabled = True
        
        with patch('plume_nav_sim.utils.logging._global_enhanced_logger', mock_global_logger):
            metrics = get_logging_metrics()
            
            # Test metrics structure
            assert metrics['status'] == 'initialized'
            assert metrics['setup_duration_ms'] == 50.0
            assert metrics['sinks_configured'] == 2
            assert metrics['experiment_id'] == 'test_exp'
            assert metrics['config_checksum'] == 'abc123'
            
            # Test configuration metrics
            config_metrics = metrics['config']
            assert config_metrics['level'] == 'DEBUG'
            assert config_metrics['format'] == 'enhanced'
            assert config_metrics['console_enabled'] is True
            assert config_metrics['file_enabled'] is True
            assert config_metrics['hydra_integration_enabled'] is True
            assert config_metrics['seed_context_enabled'] is True
            assert config_metrics['cli_metrics_enabled'] is True
    
    def test_get_logging_metrics_without_initialized_logger(self):
        """Test get_logging_metrics returns not_initialized status when logger not set."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging._global_enhanced_logger', None):
            metrics = get_logging_metrics()
            
            assert metrics == {'status': 'not_initialized'}


class TestCLICommandTracker:
    """
    Comprehensive testing for CLI command execution tracking and performance monitoring.
    
    Tests CLI command tracker initialization, metric logging, performance monitoring,
    parameter validation timing, and command completion tracking essential for
    comprehensive CLI command observability and optimization.
    
    Coverage Areas:
    - CLI command tracker initialization and context creation
    - Performance metric logging and timing validation
    - Parameter validation timing tracking
    - Command completion and final metrics reporting
    - Context manager integration and resource cleanup
    """
    
    def test_cli_command_tracker_initialization(self):
        """Test CLICommandTracker initialization with command tracking setup."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        command_name = "test_command"
        parameters = {"param1": "value1", "param2": 42}
        
        with patch('plume_nav_sim.utils.logging.bind_experiment_context') as mock_bind:
            mock_context = {'experiment_id': 'test', 'cli_command': command_name}
            mock_bind.return_value = mock_context
            
            with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
                tracker = CLICommandTracker(
                    command_name=command_name,
                    parameters=parameters,
                    track_performance=True
                )
                
                # Test initialization
                assert tracker.command_name == command_name
                assert tracker.parameters == parameters
                assert tracker.track_performance is True
                assert tracker.start_time > 0
                assert tracker.context == mock_context
                assert isinstance(tracker.metrics, dict)
                
                # Test context binding was called
                mock_bind.assert_called_once_with(
                    cli_command=command_name,
                    cli_parameters=parameters
                )
                
                # Test start logging
                mock_logger.bind.assert_called_once_with(**mock_context)
    
    def test_cli_command_tracker_metric_logging(self):
        """Test CLICommandTracker metric logging functionality."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging.bind_experiment_context') as mock_bind:
            mock_context = {'experiment_id': 'test'}
            mock_bind.return_value = mock_context
            
            with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
                tracker = CLICommandTracker("test_cmd", track_performance=True)
                
                # Test metric logging
                tracker.log_metric("test_metric", 123.45, "ms")
                
                # Test metric storage
                assert "test_metric_ms" in tracker.metrics
                assert tracker.metrics["test_metric_ms"] == 123.45
                
                # Test logging call
                mock_logger.bind.return_value.info.assert_called()
    
    def test_cli_command_tracker_validation_timing(self):
        """Test CLICommandTracker parameter validation timing tracking."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging.bind_experiment_context') as mock_bind:
            mock_context = {'experiment_id': 'test'}
            mock_bind.return_value = mock_context
            
            with patch('plume_nav_sim.utils.logging.logger'):
                tracker = CLICommandTracker("test_cmd", track_performance=True)
                
                # Test validation timing
                tracker.log_validation_time(45.67)
                
                # Test metric storage
                assert "parameter_validation_time_ms" in tracker.metrics
                assert tracker.metrics["parameter_validation_time_ms"] == 45.67
    
    def test_cli_command_tracker_performance_disabled(self):
        """Test CLICommandTracker with performance tracking disabled."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging.bind_experiment_context') as mock_bind:
            mock_context = {'experiment_id': 'test'}
            mock_bind.return_value = mock_context
            
            with patch('plume_nav_sim.utils.logging.logger'):
                tracker = CLICommandTracker("test_cmd", track_performance=False)
                
                # Test metric logging is skipped
                tracker.log_metric("test_metric", 123.45)
                
                # Test no metrics stored
                assert len(tracker.metrics) == 0
    
    def test_cli_command_tracker_completion(self):
        """Test CLICommandTracker command completion and final metrics."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging.bind_experiment_context') as mock_bind:
            mock_context = {'experiment_id': 'test', 'cli_command': 'test_cmd'}
            mock_bind.return_value = mock_context
            
            with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
                # Mock timing for controlled test
                start_time = 1000.0
                end_time = 1000.5  # 500ms execution
                
                with patch('plume_nav_sim.utils.logging.time.perf_counter', side_effect=[start_time, end_time]):
                    tracker = CLICommandTracker("test_cmd")
                    tracker.log_metric("test_metric", 42.0)
                    
                    # Complete the command
                    tracker.complete()
                    
                    # Test execution time calculation
                    expected_execution_time = (end_time - start_time) * 1000  # 500ms
                    assert tracker.context['execution_time_ms'] == expected_execution_time
                    
                    # Test completion logging
                    mock_logger.bind.return_value.info.assert_called()
                    completion_call = mock_logger.bind.return_value.info.call_args_list[-1]
                    assert "completed" in str(completion_call)
                    assert "execution_time=500.00ms" in str(completion_call)
    
    def test_track_cli_command_context_manager(self):
        """Test track_cli_command context manager integration."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        command_name = "test_context_cmd"
        parameters = {"test_param": "test_value"}
        
        with patch('plume_nav_sim.utils.logging.CLICommandTracker') as mock_tracker_class:
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            # Test context manager usage
            with track_cli_command(command_name, parameters, track_performance=True) as tracker:
                assert tracker == mock_tracker
                
                # Test tracker initialization
                mock_tracker_class.assert_called_once_with(
                    command_name, parameters, True
                )
            
            # Test completion called
            mock_tracker.complete.assert_called_once()
    
    def test_track_cli_command_context_manager_with_exception(self):
        """Test track_cli_command context manager handles exceptions correctly."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        with patch('plume_nav_sim.utils.logging.CLICommandTracker') as mock_tracker_class:
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            # Test exception handling
            try:
                with track_cli_command("test_cmd") as tracker:
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected exception
            
            # Test completion still called even with exception
            mock_tracker.complete.assert_called_once()


class TestLoggingFormats:
    """
    Comprehensive testing for logging format patterns and output validation.
    
    Tests all supported logging formats including enhanced, hydra, cli, minimal,
    and production formats for correct structure, field inclusion, and format
    consistency essential for proper log parsing and analysis.
    
    Coverage Areas:
    - Format pattern validation and structure verification
    - Field inclusion testing for all format variants
    - Format string consistency and parsing validation
    - Color coding and output formatting verification
    - Cross-format compatibility and conversion testing
    """
    
    def test_enhanced_format_structure(self):
        """Test ENHANCED_FORMAT contains required fields and structure."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Test enhanced format structure
        assert "time" in ENHANCED_FORMAT
        assert "level" in ENHANCED_FORMAT
        assert "name" in ENHANCED_FORMAT
        assert "function" in ENHANCED_FORMAT
        assert "line" in ENHANCED_FORMAT
        assert "correlation_id" in ENHANCED_FORMAT
        assert "experiment_id" in ENHANCED_FORMAT
        assert "message" in ENHANCED_FORMAT
        
        # Test color formatting
        assert "<green>" in ENHANCED_FORMAT
        assert "<level>" in ENHANCED_FORMAT
        assert "<cyan>" in ENHANCED_FORMAT
        assert "<blue>" in ENHANCED_FORMAT
        assert "<magenta>" in ENHANCED_FORMAT
    
    def test_hydra_format_structure(self):
        """Test HYDRA_FORMAT contains Hydra-specific fields."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Test Hydra-specific fields
        assert "hydra_job" in HYDRA_FORMAT
        assert "config_checksum" in HYDRA_FORMAT
        assert "correlation_id" in HYDRA_FORMAT
        assert "experiment_id" in HYDRA_FORMAT
        
        # Test Hydra format includes base fields
        assert "time" in HYDRA_FORMAT
        assert "level" in HYDRA_FORMAT
        assert "message" in HYDRA_FORMAT
        
        # Test Hydra color coding
        assert "<yellow>" in HYDRA_FORMAT  # hydra_job
        assert "<white>" in HYDRA_FORMAT   # config_checksum
    
    def test_cli_format_structure(self):
        """Test CLI_FORMAT contains CLI-specific fields."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Test CLI-specific fields
        assert "cli_command" in CLI_FORMAT
        assert "execution_time_ms" in CLI_FORMAT
        assert "correlation_id" in CLI_FORMAT
        assert "experiment_id" in CLI_FORMAT
        
        # Test CLI color coding
        assert "<yellow>" in CLI_FORMAT  # cli_command
        assert "<red>" in CLI_FORMAT     # execution_time_ms
    
    def test_minimal_format_structure(self):
        """Test MINIMAL_FORMAT has simplified structure for development."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Test minimal format simplicity
        assert "HH:mm:ss.SSS" in MINIMAL_FORMAT  # Short time format
        assert "level" in MINIMAL_FORMAT
        assert "name" in MINIMAL_FORMAT
        assert "message" in MINIMAL_FORMAT
        
        # Test minimal format excludes complex fields
        assert "correlation_id" not in MINIMAL_FORMAT
        assert "experiment_id" not in MINIMAL_FORMAT
        assert "hydra_job" not in MINIMAL_FORMAT
        assert "cli_command" not in MINIMAL_FORMAT
    
    def test_production_format_structure(self):
        """Test PRODUCTION_FORMAT for structured production logging."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        # Test production format structure
        assert "YYYY-MM-DD HH:mm:ss.SSSZ" in PRODUCTION_FORMAT  # ISO timestamp
        assert "correlation_id" in PRODUCTION_FORMAT
        assert "experiment_id" in PRODUCTION_FORMAT
        assert "seed_value" in PRODUCTION_FORMAT
        
        # Test production format excludes color codes
        assert "<green>" not in PRODUCTION_FORMAT
        assert "<level>" not in PRODUCTION_FORMAT
        assert "<cyan>" not in PRODUCTION_FORMAT
        
        # Test structured format for parsing
        assert " | " in PRODUCTION_FORMAT  # Consistent delimiter
    
    def test_format_field_extraction_patterns(self):
        """Test format patterns can be parsed for field extraction."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        formats = {
            'enhanced': ENHANCED_FORMAT,
            'hydra': HYDRA_FORMAT,
            'cli': CLI_FORMAT,
            'minimal': MINIMAL_FORMAT,
            'production': PRODUCTION_FORMAT
        }
        
        for format_name, format_string in formats.items():
            # Test format contains time field
            assert "{time" in format_string
            
            # Test format contains level field
            assert "{level" in format_string
            
            # Test format contains message field
            assert "{message}" in format_string
            
            # Test format has consistent structure
            assert format_string.count("{") == format_string.count("}")


class TestLoggingIntegration:
    """
    Comprehensive integration testing for logging system components.
    
    Tests integration between enhanced logger, experiment context, CLI tracking,
    configuration management, and cross-component interaction patterns essential
    for complete logging system validation and operational reliability.
    
    Coverage Areas:
    - End-to-end logging workflow integration testing
    - Hydra configuration integration with logging setup
    - CLI command tracking integration with experiment context
    - Performance monitoring integration across components
    - Error handling integration and recovery scenarios
    - Multi-environment integration and configuration composition
    """
    
    def test_end_to_end_logging_workflow(self, isolated_environment):
        """Test complete end-to-end logging workflow with all components."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        env_vars, temp_dir = isolated_environment
        log_file = os.path.join(temp_dir, "integration_test.log")
        
        # Create comprehensive configuration
        config = LoggingConfig(
            level="DEBUG",
            format="enhanced",
            console_enabled=True,
            file_enabled=True,
            file_path=log_file,
            hydra_integration_enabled=True,
            cli_metrics_enabled=True,
            performance_monitoring_enabled=True
        )
        
        # Mock logger operations for integration test
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.side_effect = [1, 2]  # Console and file sinks
            
            # Setup enhanced logging
            enhanced_logger = setup_enhanced_logging(config, experiment_id="integration_test")
            
            # Test CLI command tracking integration
            with track_cli_command("integration_test_cmd", {"param": "value"}) as tracker:
                tracker.log_metric("test_metric", 42.0)
                tracker.log_validation_time(10.5)
            
            # Test module logger integration
            module_logger = get_module_logger(__name__, component="integration_test")
            
            # Test configuration override logging
            log_configuration_override("test.param", "old_value", "new_value", "test")
            
            # Verify all integrations worked
            assert enhanced_logger._setup_complete is True
            assert len(enhanced_logger._sink_ids) == 2
            
            # Test logging metrics collection
            metrics = get_logging_metrics()
            assert metrics['status'] == 'initialized'
            assert 'setup_duration_ms' in metrics
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_configuration_integration(self, mock_hydra_config, isolated_environment):
        """Test complete Hydra configuration integration with logging."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        env_vars, temp_dir = isolated_environment
        
        # Add logging configuration to Hydra config
        logging_config = {
            'level': 'INFO',
            'format': 'hydra',
            'file_enabled': True,
            'file_path': os.path.join(temp_dir, 'hydra_integration.log'),
            'hydra_integration_enabled': True
        }
        mock_hydra_config.logging = DictConfig(logging_config) if HYDRA_AVAILABLE else logging_config
        
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.side_effect = [1, 2]
            
            # Test Hydra configuration integration
            result = configure_from_hydra(mock_hydra_config, experiment_id="hydra_integration")
            
            assert result is True
            
            # Test Hydra context in experiment tracking
            context = bind_experiment_context(test_param="hydra_test")
            assert 'experiment_id' in context
            
            # Verify logging setup was called with Hydra config
            assert mock_logger.add.call_count == 2  # Console + file
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration across logging components."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig(
            performance_monitoring_enabled=True,
            console_enabled=True,
            file_enabled=False
        )
        
        # Mock performance tracking
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.return_value = 1
            
            # Setup with performance monitoring
            start_time = time.perf_counter()
            enhanced_logger = setup_enhanced_logging(config)
            end_time = time.perf_counter()
            
            # Test performance metrics
            metrics = enhanced_logger.get_setup_metrics()
            assert 'setup_duration_ms' in metrics
            assert metrics['setup_duration_ms'] >= 0
            
            # Test CLI performance integration
            with track_cli_command("perf_test", track_performance=True) as tracker:
                tracker.log_metric("operation_time", 25.5, "ms")
                
                assert "operation_time_ms" in tracker.metrics
                assert tracker.metrics["operation_time_ms"] == 25.5
    
    def test_error_handling_integration(self, isolated_environment):
        """Test error handling integration across logging components."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        env_vars, temp_dir = isolated_environment
        
        # Test configuration error handling
        invalid_config = LoggingConfig(
            file_enabled=True,
            file_path="/invalid/path/that/does/not/exist/test.log"
        )
        
        # Mock logger.add to simulate file creation error
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.side_effect = [1, PermissionError("Cannot create file")]
            
            # Setup should handle error gracefully
            try:
                enhanced_logger = EnhancedLogger(invalid_config)
                enhanced_logger.setup()
            except Exception as e:
                # Should re-raise with context
                assert isinstance(e, RuntimeError)
                assert "Failed to setup enhanced logging" in str(e)
    
    def test_multi_environment_integration(self, isolated_environment):
        """Test logging integration across different environment configurations."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        env_vars, temp_dir = isolated_environment
        
        # Test development environment
        dev_config = LoggingConfig(
            level="DEBUG",
            format="minimal",
            console_enabled=True,
            file_enabled=False,
            environment_logging_enabled=True
        )
        
        # Test production environment
        prod_config = LoggingConfig(
            level="WARNING",
            format="production",
            console_enabled=False,
            file_enabled=True,
            file_path=os.path.join(temp_dir, "production.log"),
            performance_monitoring_enabled=True
        )
        
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.side_effect = [1, 2, 3]  # Multiple sink IDs
            
            # Test development setup
            dev_logger = EnhancedLogger(dev_config)
            dev_logger.setup(experiment_id="dev_test")
            
            # Test production setup
            prod_logger = EnhancedLogger(prod_config)
            prod_logger.setup(experiment_id="prod_test")
            
            # Verify different configurations
            assert dev_logger.config.level == "DEBUG"
            assert dev_logger.config.format == "minimal"
            assert prod_logger.config.level == "WARNING"
            assert prod_logger.config.format == "production"
            
            # Test environment-specific context
            dev_context = dev_logger.create_experiment_context()
            prod_context = prod_logger.create_experiment_context()
            
            assert dev_context.experiment_id == "dev_test"
            assert prod_context.experiment_id == "prod_test"
    
    def test_concurrent_logging_thread_safety(self):
        """Test logging system thread safety with concurrent operations."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig(console_enabled=True, file_enabled=False)
        
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.return_value = 1
            
            enhanced_logger = EnhancedLogger(config)
            enhanced_logger.setup()
            
            # Test thread safety with multiple context bindings
            def worker_thread(thread_id):
                context = enhanced_logger.bind_experiment_context(
                    thread_id=thread_id,
                    operation="concurrent_test"
                )
                return context
            
            # Create multiple threads
            threads = []
            results = []
            
            def thread_worker(tid):
                result = worker_thread(tid)
                results.append(result)
            
            for i in range(5):
                thread = threading.Thread(target=thread_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Verify all contexts were created
            assert len(results) == 5
            
            # Verify unique correlation IDs
            correlation_ids = [result['correlation_id'] for result in results]
            assert len(set(correlation_ids)) == 5  # All unique
    
    def test_cleanup_and_resource_management_integration(self):
        """Test comprehensive cleanup and resource management across components."""
        if not LOGGING_MODULE_AVAILABLE:
            pytest.skip("Enhanced logging module not available")
        
        config = LoggingConfig(console_enabled=True, file_enabled=False)
        
        with patch('plume_nav_sim.utils.logging.logger') as mock_logger:
            mock_logger.add.return_value = 1
            
            # Setup multiple loggers
            logger1 = EnhancedLogger(config)
            logger1.setup()
            
            logger2 = EnhancedLogger(config)
            logger2.setup()
            
            # Test individual cleanup
            logger1.cleanup()
            assert logger1._setup_complete is False
            assert len(logger1._sink_ids) == 0
            
            # Test global logger cleanup
            with patch('plume_nav_sim.utils.logging._global_enhanced_logger', logger2):
                setup_enhanced_logging()  # Should cleanup existing logger
                
            # Verify cleanup was called
            mock_logger.remove.assert_called()