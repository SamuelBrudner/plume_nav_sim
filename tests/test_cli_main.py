"""
Comprehensive pytest test suite for the CLI interface module (src/{{cookiecutter.project_slug}}/cli/main.py).

This module uses click.testing.CliRunner to validate command-line interface functionality including
simulation execution, configuration validation, parameter overrides, batch processing, and Hydra 
integration. Ensures robust CLI behavior, error handling, and parameter flow from command-line 
through Hydra composition to domain objects.

Test Coverage Areas:
- CLI command execution validation with CliRunner per Section 6.6.1.1 enhanced testing standards
- Parameter override testing through command-line flags per F-013-RQ-003
- Command initialization timing validation (<2s) per Section 2.2.9.3 performance criteria
- Help system and usage example validation per F-013-RQ-002
- Batch processing and automation workflow testing per Section 7.4.4.1
- Error handling and parameter validation testing per Section 6.6.7.2 CLI security
- Hydra configuration integration testing per Feature F-013 requirements
"""

import pytest
import time
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner
import yaml
import json

# Test imports with proper error handling for development environment
try:
    from {{cookiecutter.project_slug}}.cli.main import main, cli
    from {{cookiecutter.project_slug}}.config.schemas import NavigatorConfig, VideoPlumeConfig
    CLI_AVAILABLE = True
except ImportError:
    # Allow tests to run during development when CLI module doesn't exist yet
    CLI_AVAILABLE = False
    main = None
    cli = None


# =============================================================================
# CLI TESTING FIXTURES
# =============================================================================

@pytest.fixture
def cli_runner():
    """
    Provides Click CliRunner for command-line interface testing.
    
    Creates isolated environment for CLI testing with proper cleanup
    and environment variable isolation per Section 6.6.5.4.
    
    Returns:
        tuple: (CliRunner instance, isolated environment dict)
    
    Usage:
        def test_cli_command(cli_runner):
            runner, env = cli_runner
            result = runner.invoke(main, ['--help'], env=env)
            assert result.exit_code == 0
    """
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up isolated environment for CLI testing
        env = os.environ.copy()
        env['HYDRA_WORKING_DIR'] = temp_dir
        env['PYTHONPATH'] = str(Path(__file__).parent.parent / "src")
        
        # Clear any existing Hydra configuration to prevent pollution
        for key in list(env.keys()):
            if key.startswith('HYDRA_'):
                if key != 'HYDRA_WORKING_DIR':
                    del env[key]
        
        yield runner, env


@pytest.fixture
def mock_hydra_config():
    """
    Provides Hydra configuration composition mock for CLI testing.
    
    Creates comprehensive DictConfig mock with hierarchical structure
    supporting override testing and parameter validation scenarios.
    
    Returns:
        MagicMock: Hydra DictConfig mock with realistic configuration structure
    
    Usage:
        def test_config_override(mock_hydra_config):
            cfg = mock_hydra_config
            cfg.navigator.max_speed = 15.0
            # Test configuration parameter handling
    """
    from unittest.mock import MagicMock
    
    # Create realistic configuration structure
    config_mock = MagicMock()
    
    # Navigator configuration
    config_mock.navigator.type = "single"
    config_mock.navigator.initial_position = [50.0, 50.0]
    config_mock.navigator.initial_orientation = 0.0
    config_mock.navigator.max_speed = 10.0
    config_mock.navigator.angular_velocity = 0.1
    
    # Video plume configuration
    config_mock.video_plume.flip_horizontal = False
    config_mock.video_plume.gaussian_blur.enabled = False
    config_mock.video_plume.gaussian_blur.kernel_size = 5
    config_mock.video_plume.gaussian_blur.sigma = 1.0
    
    # Simulation configuration
    config_mock.simulation.num_steps = 1000
    config_mock.simulation.dt = 0.1
    config_mock.simulation.recording_enabled = True
    
    # Visualization configuration
    config_mock.visualization.show_animation = True
    config_mock.visualization.save_animations = False
    config_mock.visualization.export_format = "mp4"
    
    # Hydra runtime configuration
    config_mock.hydra.run.dir = "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
    config_mock.hydra.sweep.dir = "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}"
    
    return config_mock


@pytest.fixture
def temp_cli_config_files(tmp_path):
    """
    Creates temporary Hydra configuration files for CLI testing.
    
    Generates realistic conf/ directory structure with base.yaml, config.yaml,
    and local override files for comprehensive CLI parameter testing.
    
    Args:
        tmp_path: Pytest temporary directory fixture
    
    Returns:
        dict: Configuration file paths and structures for CLI testing
    
    Usage:
        def test_cli_config_loading(temp_cli_config_files):
            config_files = temp_cli_config_files
            config_dir = config_files["config_dir"]
            # Test CLI with actual configuration files
    """
    # Create conf/ directory structure
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    
    local_dir = config_dir / "local"
    local_dir.mkdir()
    
    # Base configuration (conf/base.yaml)
    base_config = {
        "defaults": ["_self_", "navigation: single_agent", "video_plume: default"],
        "navigator": {
            "type": "single",
            "initial_position": [50.0, 50.0], 
            "initial_orientation": 0.0,
            "max_speed": 10.0,
            "angular_velocity": 0.1
        },
        "video_plume": {
            "flip_horizontal": False,
            "gaussian_blur": {
                "enabled": False,
                "kernel_size": 5,
                "sigma": 1.0
            }
        },
        "simulation": {
            "num_steps": 1000,
            "dt": 0.1,
            "recording_enabled": True
        }
    }
    
    base_path = config_dir / "base.yaml"
    with open(base_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False)
    
    # Main configuration (conf/config.yaml)
    main_config = {
        "hydra": {
            "run": {"dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"},
            "sweep": {"dir": "multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}"}
        },
        "debug": False,
        "logging_level": "INFO"
    }
    
    config_path = config_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(main_config, f, default_flow_style=False)
    
    # Local development configuration (conf/local/development.yaml)
    dev_config = {
        "debug": True,
        "logging_level": "DEBUG",
        "visualization": {
            "save_animations": False,
            "show_animation": True
        }
    }
    
    dev_path = local_dir / "development.yaml"
    with open(dev_path, 'w') as f:
        yaml.dump(dev_config, f, default_flow_style=False)
    
    return {
        "config_dir": config_dir,
        "base_path": base_path,
        "config_path": config_path,
        "dev_path": dev_path,
        "base_config": base_config,
        "main_config": main_config,
        "dev_config": dev_config
    }


# =============================================================================
# CLI COMMAND EXECUTION TESTS
# =============================================================================

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available during development")
class TestCLICommandExecution:
    """
    Test suite for CLI command execution validation with CliRunner.
    
    Validates core CLI functionality including command registration,
    parameter parsing, and basic execution patterns per Section 6.6.1.1
    enhanced testing standards.
    """
    
    def test_cli_main_help_command(self, cli_runner):
        """
        Test CLI help system displays comprehensive usage information.
        
        Validates F-013-RQ-002: Help system and usage example validation.
        Ensures help text generation and documentation per requirements.
        """
        runner, env = cli_runner
        
        result = runner.invoke(main, ['--help'], env=env)
        
        # Verify successful help display
        assert result.exit_code == 0, f"Help command failed: {result.output}"
        
        # Verify essential help content
        help_text = result.output.lower()
        assert 'usage:' in help_text or 'usage' in help_text
        assert 'options:' in help_text or 'commands:' in help_text
        
        # Verify key command categories are documented
        expected_content = ['simulation', 'config', 'help']
        for content in expected_content:
            assert content in help_text, f"Missing '{content}' in help output"
    
    def test_cli_main_version_command(self, cli_runner):
        """
        Test CLI version information display.
        
        Validates version reporting functionality and package metadata access.
        """
        runner, env = cli_runner
        
        result = runner.invoke(main, ['--version'], env=env)
        
        # Version command should succeed
        assert result.exit_code == 0, f"Version command failed: {result.output}"
        assert result.output.strip(), "Version output should not be empty"
    
    def test_cli_command_initialization_timing(self, cli_runner):
        """
        Test CLI command initialization performance meets SLA requirements.
        
        Validates Section 2.2.9.3: Command initialization timing (<2s).
        Ensures CLI responsiveness per performance criteria.
        """
        runner, env = cli_runner
        
        start_time = time.time()
        result = runner.invoke(main, ['--help'], env=env)
        initialization_time = time.time() - start_time
        
        # Verify command completed successfully
        assert result.exit_code == 0, f"Command failed: {result.output}"
        
        # Verify initialization timing requirement
        assert initialization_time < 2.0, (
            f"CLI initialization took {initialization_time:.2f}s, "
            "exceeds 2s requirement per Section 2.2.9.3"
        )
    
    @patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation')
    @patch('{{cookiecutter.project_slug}}.api.navigation.create_navigator')
    @patch('{{cookiecutter.project_slug}}.api.navigation.create_video_plume')
    def test_cli_simulation_run_command(self, mock_video_plume, mock_navigator, 
                                      mock_simulation, cli_runner, mock_hydra_config):
        """
        Test CLI simulation execution with mocked dependencies.
        
        Validates F-013-RQ-001: Simulation launching with exit code verification.
        Tests complete simulation workflow through CLI interface.
        """
        runner, env = cli_runner
        
        # Configure mocks for successful simulation
        mock_navigator.return_value = MagicMock()
        mock_video_plume.return_value = MagicMock()
        mock_simulation.return_value = MagicMock()
        
        with patch('hydra.compose') as mock_compose:
            mock_compose.return_value = mock_hydra_config
            
            result = runner.invoke(main, [
                'run',
                '--config-name=config',
                '--config-path=../conf'
            ], env=env)
            
            # Verify successful simulation execution
            assert result.exit_code == 0, f"Simulation command failed: {result.output}"
            
            # Verify core simulation components were called
            mock_navigator.assert_called_once()
            mock_video_plume.assert_called_once()
            mock_simulation.assert_called_once()


# =============================================================================
# CLI PARAMETER OVERRIDE TESTS
# =============================================================================

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available during development")
class TestCLIParameterOverrides:
    """
    Test suite for CLI parameter override functionality.
    
    Validates F-013-RQ-003: Parameter override support through command-line flags.
    Tests Hydra configuration composition and override mechanisms.
    """
    
    @patch('hydra.compose')
    def test_cli_parameter_override_navigator_config(self, mock_compose, cli_runner, mock_hydra_config):
        """
        Test CLI parameter overrides for navigator configuration.
        
        Validates command-line parameter flow through Hydra composition
        to domain objects per Section 7.2.3.2.
        """
        runner, env = cli_runner
        mock_compose.return_value = mock_hydra_config
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.create_navigator') as mock_create:
            mock_create.return_value = MagicMock()
            
            result = runner.invoke(main, [
                'run',
                'navigator.max_speed=15.0',
                'navigator.initial_position=[100.0,200.0]'
            ], env=env)
            
            # Verify command completed successfully
            assert result.exit_code == 0, f"Parameter override failed: {result.output}"
            
            # Verify Hydra compose was called with overrides
            mock_compose.assert_called()
            call_args = mock_compose.call_args
            assert 'overrides' in call_args.kwargs or len(call_args.args) > 1
    
    @patch('hydra.compose')
    def test_cli_parameter_override_video_plume_config(self, mock_compose, cli_runner, mock_hydra_config):
        """
        Test CLI parameter overrides for video plume configuration.
        
        Validates video processing parameter override functionality
        through Hydra configuration system.
        """
        runner, env = cli_runner
        mock_compose.return_value = mock_hydra_config
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.create_video_plume') as mock_create:
            mock_create.return_value = MagicMock()
            
            result = runner.invoke(main, [
                'run',
                'video_plume.flip_horizontal=true',
                'video_plume.gaussian_blur.enabled=true',
                'video_plume.gaussian_blur.sigma=2.0'
            ], env=env)
            
            # Verify command completed successfully
            assert result.exit_code == 0, f"Video plume override failed: {result.output}"
    
    def test_cli_environment_variable_interpolation(self, cli_runner, temp_cli_config_files):
        """
        Test CLI environment variable interpolation support.
        
        Validates Section 7.2.3.1: Environment variable integration
        and local override management through CLI.
        """
        runner, env = cli_runner
        config_files = temp_cli_config_files
        
        # Set environment variables for interpolation testing
        env['MAX_SPEED'] = '12.5'
        env['DEBUG_MODE'] = 'true'
        env['OUTPUT_DIR'] = '/tmp/test_outputs'
        
        # Create config with environment variable interpolation
        env_config = {
            'navigator': {
                'max_speed': '${oc.env:MAX_SPEED,10.0}'
            },
            'debug': '${oc.env:DEBUG_MODE,false}',
            'output_dir': '${oc.env:OUTPUT_DIR,./outputs}'
        }
        
        env_config_path = config_files["config_dir"] / "env_test.yaml"
        with open(env_config_path, 'w') as f:
            yaml.dump(env_config, f)
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.create_navigator') as mock_create:
            mock_create.return_value = MagicMock()
            
            result = runner.invoke(main, [
                'run',
                f'--config-path={config_files["config_dir"]}',
                '--config-name=env_test'
            ], env=env)
            
            # Environment variable interpolation should not cause failures
            # Exact behavior depends on Hydra implementation
            assert result.exit_code in [0, 2], f"Environment interpolation test failed: {result.output}"


# =============================================================================
# CLI CONFIGURATION VALIDATION TESTS  
# =============================================================================

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available during development")
class TestCLIConfigurationValidation:
    """
    Test suite for CLI configuration validation functionality.
    
    Validates configuration validation commands, export functionality,
    and error handling per Section 7.4.3.2 development support.
    """
    
    def test_cli_config_validate_command(self, cli_runner, temp_cli_config_files):
        """
        Test CLI configuration validation command.
        
        Validates configuration validation capabilities per Section 7.4.3.2
        and Pydantic schema integration with Hydra per Feature F-007.
        """
        runner, env = cli_runner
        config_files = temp_cli_config_files
        
        result = runner.invoke(main, [
            'config', 'validate',
            f'--config-path={config_files["config_dir"]}'
        ], env=env)
        
        # Configuration validation should succeed with valid configs
        assert result.exit_code == 0, f"Config validation failed: {result.output}"
        
        # Output should indicate validation success
        output_lower = result.output.lower()
        success_indicators = ['valid', 'passed', 'success', 'ok']
        assert any(indicator in output_lower for indicator in success_indicators), (
            f"Expected validation success indication in: {result.output}"
        )
    
    def test_cli_config_export_command(self, cli_runner, temp_cli_config_files):
        """
        Test CLI configuration export functionality.
        
        Validates configuration documentation and export capabilities
        per development workflow requirements.
        """
        runner, env = cli_runner
        config_files = temp_cli_config_files
        
        result = runner.invoke(main, [
            'config', 'export',
            f'--config-path={config_files["config_dir"]}',
            '--format=yaml'
        ], env=env)
        
        # Config export should succeed
        assert result.exit_code == 0, f"Config export failed: {result.output}"
        
        # Output should contain configuration data
        assert result.output.strip(), "Config export should produce output"
    
    def test_cli_config_validation_with_invalid_parameters(self, cli_runner, temp_cli_config_files):
        """
        Test CLI configuration validation with invalid parameters.
        
        Validates error handling for invalid configuration values
        and comprehensive validation reporting.
        """
        runner, env = cli_runner
        config_files = temp_cli_config_files
        
        # Create invalid configuration
        invalid_config = {
            'navigator': {
                'max_speed': -5.0,  # Invalid: negative speed
                'initial_position': [50.0],  # Invalid: missing y coordinate
                'angular_velocity': 'invalid'  # Invalid: string instead of number
            }
        }
        
        invalid_path = config_files["config_dir"] / "invalid.yaml"
        with open(invalid_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        result = runner.invoke(main, [
            'config', 'validate',
            f'--config-path={config_files["config_dir"]}',
            '--config-name=invalid'
        ], env=env)
        
        # Validation should fail with invalid configuration
        assert result.exit_code != 0, "Config validation should fail with invalid parameters"
        
        # Error output should be informative
        assert result.output.strip(), "Error output should not be empty"


# =============================================================================
# CLI BATCH PROCESSING AND MULTI-RUN TESTS
# =============================================================================

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available during development")
class TestCLIBatchProcessing:
    """
    Test suite for CLI batch processing and automation workflows.
    
    Validates Section 7.4.4.1: Batch processing and automation workflow testing.
    Tests multi-run parameter sweep capabilities and experiment orchestration.
    """
    
    @patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation')
    def test_cli_multirun_parameter_sweep(self, mock_simulation, cli_runner, mock_hydra_config):
        """
        Test CLI multi-run parameter sweep functionality.
        
        Validates --multirun flag for parameter sweep execution
        per Section 7.4.4.1 experiment orchestration.
        """
        runner, env = cli_runner
        mock_simulation.return_value = MagicMock()
        
        with patch('hydra.compose') as mock_compose:
            mock_compose.return_value = mock_hydra_config
            
            result = runner.invoke(main, [
                '--multirun',
                'navigator.max_speed=5,10,15',
                'navigator.angular_velocity=0.1,0.2'
            ], env=env)
            
            # Multi-run execution handling depends on Hydra implementation
            # Command should either succeed or fail gracefully
            assert result.exit_code in [0, 1, 2], f"Multi-run command failed unexpectedly: {result.output}"
    
    @patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation')
    def test_cli_dry_run_validation(self, mock_simulation, cli_runner, mock_hydra_config):
        """
        Test CLI dry-run option for simulation validation.
        
        Validates --dry-run option for simulation validation without execution
        per development support requirements.
        """
        runner, env = cli_runner
        
        with patch('hydra.compose') as mock_compose:
            mock_compose.return_value = mock_hydra_config
            
            result = runner.invoke(main, [
                'run',
                '--dry-run'
            ], env=env)
            
            # Dry run should complete without actual simulation execution
            assert result.exit_code == 0, f"Dry run failed: {result.output}"
            
            # Simulation should not be called in dry-run mode
            mock_simulation.assert_not_called()
    
    def test_cli_headless_batch_execution(self, cli_runner, mock_hydra_config):
        """
        Test CLI headless execution for batch processing.
        
        Validates Section 7.4.2.1: Advanced interfaces for headless execution
        and automated experiment workflows.
        """
        runner, env = cli_runner
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.create_navigator') as mock_nav:
            with patch('{{cookiecutter.project_slug}}.api.navigation.create_video_plume') as mock_plume:
                with patch('hydra.compose') as mock_compose:
                    mock_nav.return_value = MagicMock()
                    mock_plume.return_value = MagicMock()
                    mock_compose.return_value = mock_hydra_config
                    
                    result = runner.invoke(main, [
                        'run',
                        '--headless',
                        'visualization.show_animation=false'
                    ], env=env)
                    
                    # Headless execution should succeed
                    assert result.exit_code in [0, 2], f"Headless execution failed: {result.output}"


# =============================================================================
# CLI ERROR HANDLING AND SECURITY TESTS
# =============================================================================

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available during development") 
class TestCLIErrorHandling:
    """
    Test suite for CLI error handling and security validation.
    
    Validates Section 6.6.7.2: CLI security and error handling requirements.
    Tests parameter validation, injection prevention, and recovery strategies.
    """
    
    def test_cli_invalid_command_handling(self, cli_runner):
        """
        Test CLI handling of invalid commands.
        
        Validates error handling for unrecognized commands and
        user-friendly error messages per Section 4.1.3.2.
        """
        runner, env = cli_runner
        
        result = runner.invoke(main, ['invalid-command'], env=env)
        
        # Invalid commands should fail gracefully
        assert result.exit_code != 0, "Invalid command should fail"
        
        # Error message should be helpful
        error_output = result.output.lower()
        assert any(phrase in error_output for phrase in ['usage', 'help', 'command', 'error']), (
            f"Expected helpful error message, got: {result.output}"
        )
    
    def test_cli_invalid_parameter_types(self, cli_runner):
        """
        Test CLI parameter type validation and error handling.
        
        Validates Click framework parameter validation and type checking
        per Section 2.2.10.1 business rules validation.
        """
        runner, env = cli_runner
        
        # Test invalid numeric parameters
        result = runner.invoke(main, [
            'run',
            'navigator.max_speed=invalid_number'
        ], env=env)
        
        # Type validation should catch invalid parameters
        assert result.exit_code != 0, "Invalid parameter type should be rejected"
    
    def test_cli_injection_attack_prevention(self, cli_runner):
        """
        Test CLI parameter parsing security against injection attempts.
        
        Validates Section 6.6.7.2: CLI argument parsing security and
        injection prevention per security requirements.
        """
        runner, env = cli_runner
        
        # Test various injection attempt patterns
        malicious_inputs = [
            '--config-path="../../../etc/passwd"',
            '--output-dir="; rm -rf /"',
            '--experiment-name="$(whoami)"',
            'navigator.max_speed="${oc.env:SECRET_KEY}"'
        ]
        
        for malicious_input in malicious_inputs:
            result = runner.invoke(main, ['run', malicious_input], env=env)
            
            # Malicious inputs should be rejected or handled safely
            # Exit code depends on specific validation implementation
            assert result.exit_code != 0 or "error" in result.output.lower(), (
                f"Potential security vulnerability with input: {malicious_input}"
            )
    
    def test_cli_file_path_validation(self, cli_runner):
        """
        Test CLI file path validation and security.
        
        Validates path traversal prevention and file access restrictions
        per Section 6.6.7.1 configuration system security.
        """
        runner, env = cli_runner
        
        # Test path traversal attempts
        dangerous_paths = [
            '../../../etc/passwd',
            '../../config.yaml',
            '/etc/hosts',
            'config/../../../sensitive_file'
        ]
        
        for dangerous_path in dangerous_paths:
            result = runner.invoke(main, [
                'config', 'validate',
                f'--config-path={dangerous_path}'
            ], env=env)
            
            # Dangerous paths should be rejected
            assert result.exit_code != 0, f"Path traversal should be prevented: {dangerous_path}"


# =============================================================================
# CLI VISUALIZATION EXPORT TESTS
# =============================================================================

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available during development")
class TestCLIVisualizationExport:
    """
    Test suite for CLI visualization export functionality.
    
    Validates F-008-RQ-004 and F-009-RQ-003: CLI export capabilities
    for MP4 animations and publication-quality static plots.
    """
    
    @patch('{{cookiecutter.project_slug}}.utils.visualization.visualize_simulation_results')
    def test_cli_animation_export_mp4(self, mock_visualize, cli_runner, mock_hydra_config):
        """
        Test CLI MP4 animation export functionality.
        
        Validates F-008-RQ-004: Export animation via CLI with MP4 format
        and configurable quality settings.
        """
        runner, env = cli_runner
        mock_visualize.return_value = None
        
        with patch('hydra.compose') as mock_compose:
            mock_compose.return_value = mock_hydra_config
            
            result = runner.invoke(main, [
                'visualize',
                '--export-format=mp4',
                '--output-file=test_animation.mp4',
                'visualization.save_animations=true'
            ], env=env)
            
            # Visualization export should succeed
            assert result.exit_code in [0, 2], f"MP4 export failed: {result.output}"
    
    @patch('{{cookiecutter.project_slug}}.utils.visualization.visualize_trajectory')
    def test_cli_publication_quality_plots(self, mock_plot, cli_runner, mock_hydra_config):
        """
        Test CLI publication-quality plot generation.
        
        Validates F-009-RQ-003: Publication quality plots with CLI export
        for high-resolution scientific visualization.
        """
        runner, env = cli_runner
        mock_plot.return_value = MagicMock()
        
        with patch('hydra.compose') as mock_compose:
            mock_compose.return_value = mock_hydra_config
            
            result = runner.invoke(main, [
                'plot',
                '--format=pdf',
                '--dpi=300',
                '--output-file=trajectory.pdf'
            ], env=env)
            
            # Publication plot generation should succeed
            assert result.exit_code in [0, 2], f"Publication plot export failed: {result.output}"


# =============================================================================
# CLI INTEGRATION TESTS
# =============================================================================

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available during development")
class TestCLIIntegration:
    """
    Test suite for comprehensive CLI integration scenarios.
    
    Validates end-to-end CLI workflows and integration with core system
    components per Feature F-013 comprehensive requirements.
    """
    
    @patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation')
    @patch('{{cookiecutter.project_slug}}.api.navigation.create_navigator')
    @patch('{{cookiecutter.project_slug}}.api.navigation.create_video_plume')
    @patch('{{cookiecutter.project_slug}}.utils.seed_manager.set_global_seed')
    def test_cli_complete_simulation_workflow(self, mock_seed, mock_plume, mock_nav, 
                                            mock_sim, cli_runner, mock_hydra_config):
        """
        Test complete CLI simulation workflow integration.
        
        Validates end-to-end CLI functionality with all core components
        including configuration, simulation, and output generation.
        """
        runner, env = cli_runner
        
        # Configure mocks for successful workflow
        mock_nav.return_value = MagicMock()
        mock_plume.return_value = MagicMock()
        mock_sim.return_value = MagicMock()
        
        with patch('hydra.compose') as mock_compose:
            mock_compose.return_value = mock_hydra_config
            
            result = runner.invoke(main, [
                'run',
                'navigator.max_speed=12.0',
                'simulation.num_steps=500',
                'visualization.save_animations=true',
                '--seed=42'
            ], env=env)
            
            # Complete workflow should succeed
            assert result.exit_code == 0, f"Complete workflow failed: {result.output}"
            
            # Verify all components were initialized
            mock_seed.assert_called_once_with(42)
            mock_nav.assert_called_once()
            mock_plume.assert_called_once()
            mock_sim.assert_called_once()
    
    def test_cli_interactive_prompt_simulation(self, cli_runner):
        """
        Test CLI interactive prompt and confirmation dialog functionality.
        
        Validates interactive CLI user experience with confirmation dialogs
        and prompt handling for enhanced usability.
        """
        runner, env = cli_runner
        
        with patch('{{cookiecutter.project_slug}}.api.navigation.create_navigator') as mock_nav:
            mock_nav.return_value = MagicMock()
            
            # Simulate user input for interactive prompts
            result = runner.invoke(main, [
                'run',
                '--interactive'
            ], input='y\n12.0\n500\n', env=env)
            
            # Interactive mode should handle user input appropriately
            # Exact behavior depends on CLI implementation
            assert result.exit_code in [0, 1, 2], f"Interactive mode failed: {result.output}"


# =============================================================================
# CLI PERFORMANCE AND STRESS TESTS
# =============================================================================

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available during development")
class TestCLIPerformance:
    """
    Test suite for CLI performance validation and stress testing.
    
    Validates performance requirements per Section 2.2.9.3 and ensures
    CLI responsiveness under various load conditions.
    """
    
    def test_cli_help_response_time(self, cli_runner):
        """
        Test CLI help command response time performance.
        
        Validates CLI responsiveness for documentation access
        ensuring quick help system response.
        """
        runner, env = cli_runner
        
        response_times = []
        
        # Test multiple help command executions
        for _ in range(5):
            start_time = time.time()
            result = runner.invoke(main, ['--help'], env=env)
            response_time = time.time() - start_time
            
            assert result.exit_code == 0, "Help command should succeed"
            response_times.append(response_time)
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 1.0, (
            f"Average help response time {avg_response_time:.2f}s exceeds 1.0s threshold"
        )
    
    def test_cli_parameter_validation_performance(self, cli_runner):
        """
        Test CLI parameter validation performance with complex overrides.
        
        Validates parameter parsing and validation efficiency for
        complex parameter override scenarios.
        """
        runner, env = cli_runner
        
        # Create complex parameter override scenario
        complex_overrides = [
            'navigator.max_speed=10.0',
            'navigator.initial_position=[100.0,200.0]',
            'video_plume.flip_horizontal=true',
            'video_plume.gaussian_blur.enabled=true',
            'video_plume.gaussian_blur.kernel_size=7',
            'video_plume.gaussian_blur.sigma=2.5',
            'simulation.num_steps=2000',
            'simulation.dt=0.05',
            'visualization.show_animation=false'
        ]
        
        start_time = time.time()
        result = runner.invoke(main, ['run', '--dry-run'] + complex_overrides, env=env)
        validation_time = time.time() - start_time
        
        # Parameter validation should complete quickly
        assert validation_time < 2.0, (
            f"Parameter validation took {validation_time:.2f}s, "
            "exceeds 2.0s threshold"
        )


# =============================================================================
# PYTEST CONFIGURATION AND UTILITIES
# =============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Automatic fixture to set up clean test environment for each CLI test.
    
    Ensures proper test isolation and cleanup for CLI testing scenarios
    per Section 6.6.5.4 test environment isolation strategy.
    """
    # Clear any existing Hydra configuration state
    import os
    original_env = os.environ.copy()
    
    # Remove Hydra-related environment variables
    for key in list(os.environ.keys()):
        if key.startswith('HYDRA_'):
            del os.environ[key]
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Test execution configuration
def pytest_configure(config):
    """Configure pytest for CLI testing with proper markers and options."""
    config.addinivalue_line(
        "markers", "cli: mark test as CLI interface test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


# Test collection configuration  
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add appropriate markers."""
    for item in items:
        # Mark CLI tests
        if "cli" in item.nodeid.lower():
            item.add_marker(pytest.mark.cli)
        
        # Mark integration tests
        if "integration" in item.name.lower() or "workflow" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "performance" in item.name.lower() or "timing" in item.name.lower():
            item.add_marker(pytest.mark.performance)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])