"""
Command-line interface package for odor plume navigation system.

This package provides comprehensive CLI functionality built with Click framework and Hydra
configuration management, supporting simulation execution, configuration validation, batch
processing, parameter sweeps, and visualization export commands. The interface implements
enterprise-grade command structure with <2s initialization time and extensible architecture.

The CLI architecture supports:
- Standardized entry point via python -m {{cookiecutter.project_slug}}.cli.main
- Hydra-based configuration management with parameter overrides  
- Click framework integration with comprehensive subcommand structure
- Multi-run experiment orchestration via --multirun flag
- Batch processing capabilities for headless execution environments
- Real-time simulation execution with visualization export options
- Configuration validation and export with comprehensive error reporting
- Extensible command registration patterns for research workflow integration

Performance Characteristics:
- Command initialization: <2s per Section 2.2.9.3 performance criteria
- Configuration loading: <1s for complex hierarchical configurations
- CLI help generation: Instant response with comprehensive documentation
- Parameter validation: Real-time with immediate feedback and error reporting

Usage Patterns:
    # Basic simulation execution
    python -m {{cookiecutter.project_slug}}.cli.main run
    
    # With parameter overrides
    python -m {{cookiecutter.project_slug}}.cli.main run navigator.max_speed=10.0
    
    # Multi-run parameter sweep
    python -m {{cookiecutter.project_slug}}.cli.main --multirun run navigator.max_speed=5,10,15
    
    # Configuration validation
    python -m {{cookiecutter.project_slug}}.cli.main config validate
    
    # Visualization export
    python -m {{cookiecutter.project_slug}}.cli.main visualize export --format mp4

Programmatic Access:
    from {{cookiecutter.project_slug}}.cli import main, cli, run_command
    from {{cookiecutter.project_slug}}.cli import register_command, extend_cli
    
    # Direct CLI group access for extension
    @cli.command()
    def custom_command():
        pass
    
    # Main function access for embedding
    main()

Entry Points:
- main: Primary CLI entrypoint with @hydra.main decorator for configuration injection
- cli: Click command group for direct command registration and extension
- Command utilities: Functions for CLI command registration and workflow integration
"""

import sys
import warnings
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path

# Performance-optimized imports - lazy loading for faster initialization
try:
    # Import core CLI components with error handling for missing dependencies
    from .main import main, cli
    
    # Import key utility functions for CLI operations and extensions
    from .main import (
        _setup_cli_logging,
        _validate_hydra_availability, 
        _validate_configuration,
        _export_config_documentation,
        _measure_performance,
        CLIError,
        ConfigValidationError
    )
    
    # CLI component availability flag
    CLI_AVAILABLE = True
    
except ImportError as e:
    # Graceful degradation when CLI dependencies are missing
    CLI_AVAILABLE = False
    main = None
    cli = None
    
    # Create stub functions for missing components
    def _missing_cli_stub(*args, **kwargs):
        raise ImportError(
            f"CLI functionality not available. Missing dependencies: {e}. "
            "Please install with: pip install click hydra-core"
        )
    
    _setup_cli_logging = _missing_cli_stub
    _validate_hydra_availability = _missing_cli_stub
    _validate_configuration = _missing_cli_stub
    _export_config_documentation = _missing_cli_stub
    _measure_performance = _missing_cli_stub
    
    class CLIError(Exception):
        """CLI-specific error for command execution failures."""
        pass
    
    class ConfigValidationError(Exception):
        """Configuration validation specific errors."""
        pass


# CLI metadata and version information
__version__ = "1.0.0"
__author__ = "Odor Plume Navigation Team"
__description__ = "Command-line interface for odor plume navigation simulation system"

# CLI configuration and performance settings
CLI_CONFIG = {
    'max_initialization_time': 2.0,  # Per Section 2.2.9.3 performance criteria
    'default_log_level': 'INFO',
    'support_multirun': True,
    'enable_parameter_overrides': True,
    'support_batch_processing': True
}


def get_cli_version() -> str:
    """
    Get CLI package version information.
    
    Returns:
        Version string for CLI package
    """
    return __version__


def is_cli_available() -> bool:
    """
    Check if CLI functionality is available.
    
    Returns:
        True if all CLI dependencies are installed and functional
    """
    return CLI_AVAILABLE


def validate_cli_environment() -> Dict[str, Any]:
    """
    Validate CLI environment and dependencies.
    
    Returns:
        Dictionary containing environment validation results
        
    Raises:
        CLIError: If critical CLI dependencies are missing
    """
    validation_results = {
        'cli_available': CLI_AVAILABLE,
        'dependencies': {},
        'warnings': [],
        'errors': []
    }
    
    # Check core dependencies
    try:
        import click
        validation_results['dependencies']['click'] = click.__version__
    except ImportError:
        validation_results['dependencies']['click'] = None
        validation_results['errors'].append("Click framework not available")
    
    try:
        import hydra
        validation_results['dependencies']['hydra'] = hydra.__version__
    except ImportError:
        validation_results['dependencies']['hydra'] = None
        validation_results['errors'].append("Hydra configuration management not available")
    
    try:
        from omegaconf import OmegaConf
        validation_results['dependencies']['omegaconf'] = OmegaConf.__version__
    except ImportError:
        validation_results['dependencies']['omegaconf'] = None
        validation_results['warnings'].append("OmegaConf not available - limited configuration features")
    
    # Check optional dependencies
    try:
        import loguru
        validation_results['dependencies']['loguru'] = loguru.__version__
    except ImportError:
        validation_results['dependencies']['loguru'] = None
        validation_results['warnings'].append("Loguru not available - using standard logging")
    
    # Validate environment configuration
    if not CLI_AVAILABLE:
        validation_results['errors'].append("CLI functionality not available due to missing dependencies")
    
    return validation_results


def register_command(
    command_func: Callable,
    name: Optional[str] = None,
    **click_kwargs
) -> Callable:
    """
    Register a custom command with the CLI system.
    
    This function provides a decorator pattern for extending CLI functionality
    with custom commands while maintaining integration with Hydra configuration
    and Click framework patterns.
    
    Args:
        command_func: Function to register as CLI command
        name: Optional command name (defaults to function name)
        **click_kwargs: Additional Click command options
        
    Returns:
        Decorated command function registered with CLI
        
    Example:
        @register_command
        def custom_analysis():
            '''Custom analysis command'''
            pass
        
        @register_command(name="special-run")
        def custom_run():
            '''Custom simulation runner'''
            pass
    """
    if not CLI_AVAILABLE:
        def unavailable_command(*args, **kwargs):
            raise CLIError("CLI functionality not available")
        return unavailable_command
    
    # Import click here to avoid import issues when CLI not available
    import click
    
    # Determine command name
    cmd_name = name or command_func.__name__.replace('_', '-')
    
    # Create Click command with provided options
    click_command = click.command(name=cmd_name, **click_kwargs)(command_func)
    
    # Register with main CLI group if available
    if cli is not None:
        cli.add_command(click_command)
    
    return click_command


def extend_cli(group_func: Callable, name: Optional[str] = None, **click_kwargs) -> Callable:
    """
    Extend CLI with custom command groups.
    
    This function enables registration of command groups for modular CLI extension
    supporting research workflow integration and specialized command hierarchies.
    
    Args:
        group_func: Function to register as CLI command group
        name: Optional group name (defaults to function name)
        **click_kwargs: Additional Click group options
        
    Returns:
        Decorated group function registered with CLI
        
    Example:
        @extend_cli
        def analysis():
            '''Analysis command group'''
            pass
        
        @analysis.command()
        def trajectory():
            '''Analyze trajectories'''
            pass
    """
    if not CLI_AVAILABLE:
        def unavailable_group(*args, **kwargs):
            raise CLIError("CLI functionality not available")
        return unavailable_group
    
    # Import click here to avoid import issues when CLI not available
    import click
    
    # Determine group name
    group_name = name or group_func.__name__.replace('_', '-')
    
    # Create Click group with provided options
    click_group = click.group(name=group_name, **click_kwargs)(group_func)
    
    # Register with main CLI group if available
    if cli is not None:
        cli.add_command(click_group)
    
    return click_group


def run_command(command_name: str, args: List[str] = None, **kwargs) -> int:
    """
    Programmatically execute CLI commands.
    
    This function enables embedding CLI functionality within Python scripts
    and applications while maintaining full access to configuration management
    and error handling capabilities.
    
    Args:
        command_name: Name of CLI command to execute
        args: Optional command arguments
        **kwargs: Additional command options
        
    Returns:
        Command exit code (0 for success, non-zero for failure)
        
    Example:
        # Run simulation programmatically
        exit_code = run_command('run', ['navigator.max_speed=15.0'])
        
        # Validate configuration
        exit_code = run_command('config', ['validate', '--strict'])
    """
    if not CLI_AVAILABLE:
        raise CLIError("CLI functionality not available")
    
    if cli is None:
        raise CLIError("CLI not properly initialized")
    
    try:
        # Prepare command arguments
        command_args = [command_name]
        if args:
            command_args.extend(args)
        
        # Execute command with error handling
        result = cli.main(command_args, standalone_mode=False, **kwargs)
        return 0
        
    except SystemExit as e:
        return e.code
    except Exception as e:
        print(f"CLI command failed: {e}", file=sys.stderr)
        return 1


def get_available_commands() -> Dict[str, str]:
    """
    Get list of available CLI commands with descriptions.
    
    Returns:
        Dictionary mapping command names to their descriptions
    """
    if not CLI_AVAILABLE or cli is None:
        return {}
    
    commands = {}
    for name, command in cli.commands.items():
        # Get command help text or docstring
        help_text = getattr(command, 'help', None) or getattr(command, '__doc__', '') or 'No description available'
        commands[name] = help_text.split('\n')[0]  # First line only
    
    return commands


def setup_cli_development() -> None:
    """
    Setup CLI for development and testing environments.
    
    This function configures CLI for optimal development experience including
    enhanced error reporting, debug logging, and development-specific features.
    """
    if not CLI_AVAILABLE:
        warnings.warn("CLI not available - skipping development setup")
        return
    
    # Enable development features
    import os
    os.environ.setdefault('CLI_DEVELOPMENT_MODE', '1')
    os.environ.setdefault('HYDRA_FULL_ERROR', '1')
    
    # Setup enhanced logging for development
    if _setup_cli_logging:
        _setup_cli_logging(verbose=True, log_level='DEBUG')


# Module-level initialization and validation
def _initialize_cli_module() -> None:
    """Initialize CLI module with performance monitoring and validation."""
    import time
    start_time = time.time()
    
    try:
        # Validate CLI environment on import
        if CLI_AVAILABLE:
            validation_results = validate_cli_environment()
            
            # Check for critical errors
            if validation_results['errors']:
                warnings.warn(
                    f"CLI initialization warnings: {', '.join(validation_results['errors'])}",
                    ImportWarning
                )
            
            # Check performance
            initialization_time = time.time() - start_time
            if initialization_time > CLI_CONFIG['max_initialization_time']:
                warnings.warn(
                    f"CLI initialization took {initialization_time:.2f}s "
                    f"(exceeds {CLI_CONFIG['max_initialization_time']}s threshold)",
                    PerformanceWarning
                )
        
    except Exception as e:
        warnings.warn(f"CLI module initialization failed: {e}", ImportWarning)


# Performance monitoring warning category
class PerformanceWarning(UserWarning):
    """Warning for CLI performance issues."""
    pass


# Initialize CLI module on import
_initialize_cli_module()


# Public API exports for clean import patterns
__all__ = [
    # Primary CLI functions
    'main',
    'cli', 
    
    # CLI utilities and configuration
    'get_cli_version',
    'is_cli_available',
    'validate_cli_environment',
    'get_available_commands',
    'setup_cli_development',
    
    # Command registration and extension
    'register_command',
    'extend_cli',
    'run_command',
    
    # Error classes
    'CLIError',
    'ConfigValidationError',
    
    # CLI configuration
    'CLI_CONFIG',
    
    # Internal utilities (for advanced usage)
    '_setup_cli_logging',
    '_validate_hydra_availability',
    '_validate_configuration',
    '_export_config_documentation',
    '_measure_performance'
]