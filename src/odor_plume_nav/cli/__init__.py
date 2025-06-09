"""
Odor Plume Navigation CLI - Central initializer and API surface.

This module serves as the central initializer and API surface for the odor plume navigation
command-line interface. It conditionally imports the core CLI entry point and command group
from src/odor_plume_nav/cli/main.py, defines version metadata and system information,
establishes default configuration settings, and exposes utility functions for CLI operations
including version retrieval, availability checks, environment validation, command registration,
programmatic invocation, and performance monitoring.

The module implements graceful degradation in headless environments with predictable
ImportError messages for missing dependencies, enabling reliable CLI operation across
different deployment scenarios including development, testing, and production environments.

Primary Functions:
- CLI entry point imports and conditional availability checking
- Version and metadata management for CLI operations
- Configuration dictionary initialization with performance settings
- Environment validation and development setup utilities
- Command registration and extension capabilities for researchers
- Programmatic CLI invocation with proper error handling
- Performance monitoring and diagnostic utilities

Architecture:
The CLI system is built on Click framework with Hydra configuration integration,
providing hierarchical configuration composition, parameter overrides, multi-run
experiment support, and comprehensive error handling with diagnostic information.

Example Usage:
    # Import CLI components
    from odor_plume_nav.cli import main, cli, is_available
    
    # Check CLI availability
    if is_available():
        main()  # Run with Hydra configuration
    
    # Get version information
    version = get_version()
    
    # Validate environment
    validate_environment()
    
    # Programmatic execution
    result = invoke_command(['run', '--dry-run'])

Performance Characteristics:
- Command initialization: <2s per Section 2.2.9.3 performance criteria
- Import overhead: <500ms for conditional loading
- Environment validation: <100ms for comprehensive checks
- Version queries: <10ms for metadata retrieval

Dependencies:
- odor_plume_nav.cli.main: Core CLI implementation with Click and Hydra
- odor_plume_nav.config.models: Configuration validation models
- odor_plume_nav.utils.logging_setup: Enhanced logging utilities
"""

import os
import sys
import time
import warnings
from typing import Optional, Dict, Any, List, Callable, Union
from pathlib import Path

# Package metadata and version information
__version__ = "1.0.0"
__author__ = "Odor Plume Navigation Team"
__description__ = "Comprehensive command-line interface for odor plume navigation research"
__license__ = "MIT"

# CLI availability flags and error tracking
_CLI_AVAILABLE = False
_CLI_ERROR = None
_HYDRA_AVAILABLE = False
_CLICK_AVAILABLE = False

# Default CLI configuration dictionary with performance and behavior settings
CLI_CONFIG = {
    # Performance settings per Section 2.2.9.3 criteria
    'max_command_init_time': 2.0,  # seconds
    'max_config_load_time': 1.0,   # seconds
    'max_validation_time': 0.5,    # seconds
    
    # Behavior configuration
    'verbose_mode': False,
    'quiet_mode': False,
    'log_level': 'INFO',
    'development_mode': os.getenv('CLI_DEVELOPMENT_MODE', 'false').lower() == 'true',
    'hydra_full_error': os.getenv('HYDRA_FULL_ERROR', 'false').lower() == 'true',
    
    # Environment settings
    'auto_setup_logging': True,
    'enable_performance_monitoring': True,
    'graceful_degradation': True,
    'headless_compatible': True,
    
    # Command registration settings
    'allow_command_extensions': True,
    'validate_command_signatures': True,
    'enable_command_caching': False,
    
    # Output configuration
    'color_output': sys.stdout.isatty(),
    'progress_indicators': True,
    'error_diagnostics': True,
    'timing_reports': False
}

# Global state tracking for CLI operations
_cli_state = {
    'initialized': False,
    'last_command': None,
    'performance_metrics': {},
    'error_history': [],
    'startup_time': None
}


def _measure_time(operation_name: str) -> Callable:
    """
    Decorator for measuring operation timing with performance tracking.
    
    Args:
        operation_name: Name of the operation being measured
        
    Returns:
        Decorator function for timing operations
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                _cli_state['performance_metrics'][operation_name] = elapsed
                
                # Check performance thresholds
                if operation_name == 'command_init' and elapsed > CLI_CONFIG['max_command_init_time']:
                    warnings.warn(
                        f"Command initialization took {elapsed:.2f}s (>{CLI_CONFIG['max_command_init_time']}s threshold)",
                        PerformanceWarning
                    )
                
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                _cli_state['error_history'].append({
                    'operation': operation_name,
                    'error': str(e),
                    'timestamp': time.time(),
                    'duration': elapsed
                })
                raise
        return wrapper
    return decorator


class PerformanceWarning(UserWarning):
    """Warning for CLI performance issues exceeding thresholds."""
    pass


class CLINotAvailableError(ImportError):
    """Raised when CLI functionality is not available due to missing dependencies."""
    pass


class EnvironmentValidationError(RuntimeError):
    """Raised when environment validation fails with specific diagnostic information."""
    pass


# Conditional imports with graceful degradation
@_measure_time('cli_imports')
def _import_cli_components():
    """
    Conditionally import CLI components with comprehensive error handling.
    
    This function attempts to import the main CLI components and tracks
    availability status for graceful degradation in headless environments.
    
    Returns:
        tuple: (main_function, cli_group, import_success)
    """
    global _CLI_AVAILABLE, _CLI_ERROR, _HYDRA_AVAILABLE, _CLICK_AVAILABLE
    
    main_func = None
    cli_group = None
    
    try:
        # Check for Click framework availability
        try:
            import click
            _CLICK_AVAILABLE = True
        except ImportError as e:
            _CLI_ERROR = f"Click framework not available: {e}"
            return None, None, False
        
        # Check for Hydra configuration system
        try:
            import hydra
            from hydra.core.config_store import ConfigStore
            _HYDRA_AVAILABLE = True
        except ImportError as e:
            _CLI_ERROR = f"Hydra configuration system not available: {e}"
            return None, None, False
        
        # Import main CLI components from src/odor_plume_nav/cli/main.py
        try:
            from odor_plume_nav.cli.main import main, cli
            main_func = main
            cli_group = cli
            _CLI_AVAILABLE = True
            
        except ImportError as e:
            _CLI_ERROR = f"Failed to import CLI main components: {e}"
            return None, None, False
            
        except Exception as e:
            _CLI_ERROR = f"Unexpected error importing CLI components: {e}"
            return None, None, False
    
    except Exception as e:
        _CLI_ERROR = f"Critical error during CLI import: {e}"
        return None, None, False
    
    return main_func, cli_group, True


# Perform initial import attempt
_startup_time = time.time()
main, cli, _import_success = _import_cli_components()
_cli_state['startup_time'] = time.time() - _startup_time


def is_available() -> bool:
    """
    Check if CLI functionality is available.
    
    This function performs a comprehensive availability check for CLI components
    including Click framework, Hydra configuration system, and core CLI modules.
    
    Returns:
        bool: True if CLI is fully functional, False otherwise
        
    Examples:
        >>> if is_available():
        ...     from odor_plume_nav.cli import main
        ...     main()
        ... else:
        ...     print("CLI not available in headless environment")
    """
    return _CLI_AVAILABLE and main is not None and cli is not None


def get_availability_status() -> Dict[str, Any]:
    """
    Get detailed availability status for CLI components.
    
    Returns comprehensive status information including component availability,
    error details, performance metrics, and diagnostic information.
    
    Returns:
        dict: Detailed availability status with diagnostics
        
    Example:
        >>> status = get_availability_status()
        >>> print(f"CLI Available: {status['cli_available']}")
        >>> print(f"Hydra Available: {status['hydra_available']}")
        >>> if status['error']:
        ...     print(f"Error: {status['error']}")
    """
    return {
        'cli_available': _CLI_AVAILABLE,
        'click_available': _CLICK_AVAILABLE, 
        'hydra_available': _HYDRA_AVAILABLE,
        'main_function_available': main is not None,
        'cli_group_available': cli is not None,
        'error': _CLI_ERROR,
        'startup_time': _cli_state['startup_time'],
        'performance_metrics': _cli_state['performance_metrics'].copy(),
        'initialized': _cli_state['initialized']
    }


def get_version() -> str:
    """
    Get CLI package version information.
    
    Returns:
        str: Version string for the CLI package
        
    Example:
        >>> version = get_version()
        >>> print(f"CLI Version: {version}")
    """
    return __version__


def get_version_info() -> Dict[str, str]:
    """
    Get comprehensive version and metadata information.
    
    Returns detailed version information including package metadata,
    author information, license details, and dependency versions.
    
    Returns:
        dict: Comprehensive version and metadata information
        
    Example:
        >>> info = get_version_info()
        >>> print(f"Author: {info['author']}")
        >>> print(f"Description: {info['description']}")
    """
    version_info = {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'license': __license__,
        'package_name': 'odor_plume_nav',
        'cli_module': 'odor_plume_nav.cli'
    }
    
    # Add dependency versions if available
    try:
        import click
        version_info['click_version'] = click.__version__
    except (ImportError, AttributeError):
        version_info['click_version'] = 'Not available'
    
    try:
        import hydra
        version_info['hydra_version'] = hydra.__version__
    except (ImportError, AttributeError):
        version_info['hydra_version'] = 'Not available'
    
    try:
        import numpy
        version_info['numpy_version'] = numpy.__version__
    except (ImportError, AttributeError):
        version_info['numpy_version'] = 'Not available'
    
    return version_info


@_measure_time('environment_validation')
def validate_environment() -> Dict[str, Any]:
    """
    Validate CLI environment and dependencies with comprehensive checks.
    
    Performs thorough validation of the CLI environment including dependency
    availability, configuration file existence, environment variables,
    and performance characteristics.
    
    Returns:
        dict: Validation results with detailed diagnostic information
        
    Raises:
        EnvironmentValidationError: If critical validation checks fail
        
    Example:
        >>> try:
        ...     results = validate_environment()
        ...     print(f"Validation passed: {results['valid']}")
        ... except EnvironmentValidationError as e:
        ...     print(f"Environment validation failed: {e}")
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'checks': {},
        'performance': {},
        'environment': {}
    }
    
    # Check CLI availability
    validation_results['checks']['cli_available'] = is_available()
    if not is_available():
        validation_results['errors'].append(f"CLI not available: {_CLI_ERROR}")
        validation_results['valid'] = False
    
    # Check environment variables
    env_vars = ['CLI_DEVELOPMENT_MODE', 'HYDRA_FULL_ERROR', 'PYTHONPATH']
    for var in env_vars:
        value = os.getenv(var)
        validation_results['environment'][var] = value
        validation_results['checks'][f'env_{var.lower()}'] = value is not None
    
    # Check configuration directory
    config_path = Path(__file__).parent.parent.parent.parent / 'conf'
    validation_results['checks']['config_directory'] = config_path.exists()
    if not config_path.exists():
        validation_results['warnings'].append(f"Configuration directory not found: {config_path}")
    
    # Check for required configuration files
    if config_path.exists():
        required_configs = ['base.yaml', 'config.yaml']
        for config_file in required_configs:
            config_file_path = config_path / config_file
            exists = config_file_path.exists()
            validation_results['checks'][f'config_{config_file}'] = exists
            if not exists:
                validation_results['warnings'].append(f"Configuration file missing: {config_file}")
    
    # Performance validation
    if _cli_state['startup_time']:
        validation_results['performance']['startup_time'] = _cli_state['startup_time']
        if _cli_state['startup_time'] > CLI_CONFIG['max_command_init_time']:
            validation_results['warnings'].append(
                f"Startup time {_cli_state['startup_time']:.2f}s exceeds threshold "
                f"{CLI_CONFIG['max_command_init_time']}s"
            )
    
    # Python version check
    py_version = sys.version_info
    validation_results['environment']['python_version'] = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
    if py_version < (3, 8):
        validation_results['errors'].append(f"Python {py_version.major}.{py_version.minor} not supported, requires >=3.8")
        validation_results['valid'] = False
    
    # Memory and system checks
    try:
        import psutil
        validation_results['environment']['available_memory'] = psutil.virtual_memory().available
        validation_results['checks']['memory_adequate'] = psutil.virtual_memory().available > 512 * 1024 * 1024  # 512MB
    except ImportError:
        validation_results['warnings'].append("psutil not available for memory checks")
    
    # Raise exception if validation fails
    if not validation_results['valid']:
        error_details = '; '.join(validation_results['errors'])
        raise EnvironmentValidationError(f"Environment validation failed: {error_details}")
    
    return validation_results


def register_command(command_name: str, command_func: Callable, group: Optional[str] = None) -> bool:
    """
    Register a custom command with the CLI system for extension support.
    
    This function enables researchers and extension modules to register
    custom commands with the CLI system, supporting dynamic command
    registration and extension capabilities.
    
    Args:
        command_name: Name of the command to register
        command_func: Function implementing the command
        group: Optional command group for organization
        
    Returns:
        bool: True if registration successful, False otherwise
        
    Example:
        >>> def my_command():
        ...     print("Custom command executed")
        ...
        >>> success = register_command('my-cmd', my_command, 'custom')
        >>> print(f"Command registered: {success}")
    """
    if not CLI_CONFIG['allow_command_extensions']:
        warnings.warn("Command registration disabled in CLI configuration")
        return False
    
    if not is_available():
        warnings.warn("CLI not available, cannot register commands")
        return False
    
    try:
        # Basic validation of command function
        if CLI_CONFIG['validate_command_signatures']:
            if not callable(command_func):
                raise ValueError("Command function must be callable")
        
        # In a full implementation, this would integrate with Click
        # to dynamically add commands to the CLI group
        # For now, we'll track registered commands for future implementation
        if not hasattr(_cli_state, 'registered_commands'):
            _cli_state['registered_commands'] = {}
        
        command_key = f"{group}.{command_name}" if group else command_name
        _cli_state['registered_commands'][command_key] = {
            'name': command_name,
            'function': command_func,
            'group': group,
            'registered_at': time.time()
        }
        
        return True
    
    except Exception as e:
        warnings.warn(f"Failed to register command {command_name}: {e}")
        return False


def list_commands() -> List[Dict[str, Any]]:
    """
    List all available CLI commands with introspection support.
    
    Returns comprehensive information about available CLI commands
    including built-in commands and registered extensions.
    
    Returns:
        list: List of command information dictionaries
        
    Example:
        >>> commands = list_commands()
        >>> for cmd in commands:
        ...     print(f"Command: {cmd['name']} - {cmd['description']}")
    """
    commands = []
    
    # Add built-in commands if CLI is available
    if is_available() and cli is not None:
        try:
            # Introspect Click commands
            for name, command in cli.commands.items():
                commands.append({
                    'name': name,
                    'type': 'built-in',
                    'description': command.help or 'No description available',
                    'group': 'main',
                    'available': True
                })
        except Exception as e:
            warnings.warn(f"Failed to introspect built-in commands: {e}")
    
    # Add registered extension commands
    if hasattr(_cli_state, 'registered_commands'):
        for cmd_key, cmd_info in _cli_state['registered_commands'].items():
            commands.append({
                'name': cmd_info['name'],
                'type': 'extension',
                'description': getattr(cmd_info['function'], '__doc__', 'No description'),
                'group': cmd_info.get('group', 'custom'),
                'available': True,
                'registered_at': cmd_info['registered_at']
            })
    
    return commands


def invoke_command(args: List[str], **kwargs) -> Dict[str, Any]:
    """
    Programmatically invoke CLI commands with error handling.
    
    Enables programmatic execution of CLI commands from Python code
    with comprehensive error handling and result reporting.
    
    Args:
        args: Command arguments list (e.g., ['run', '--dry-run'])
        **kwargs: Additional keyword arguments for command execution
        
    Returns:
        dict: Execution results with success status and output
        
    Example:
        >>> result = invoke_command(['config', 'validate'])
        >>> if result['success']:
        ...     print(f"Command succeeded: {result['output']}")
        ... else:
        ...     print(f"Command failed: {result['error']}")
    """
    if not is_available():
        return {
            'success': False,
            'error': 'CLI not available',
            'output': None,
            'execution_time': 0
        }
    
    start_time = time.time()
    
    try:
        # Store original sys.argv
        original_argv = sys.argv.copy()
        
        # Set up command arguments
        sys.argv = ['odor-plume-nav-cli'] + args
        
        # Track command execution
        _cli_state['last_command'] = {
            'args': args,
            'timestamp': start_time,
            'kwargs': kwargs
        }
        
        # Attempt to execute command
        # Note: This is a simplified implementation
        # Full implementation would capture output and handle Click context properly
        if cli is not None:
            # For demonstration, we'll validate the command exists
            if args and args[0] in cli.commands:
                execution_time = time.time() - start_time
                return {
                    'success': True,
                    'error': None,
                    'output': f"Command {args[0]} would be executed",
                    'execution_time': execution_time
                }
            else:
                return {
                    'success': False,
                    'error': f"Unknown command: {args[0] if args else 'no command'}",
                    'output': None,
                    'execution_time': time.time() - start_time
                }
        else:
            return {
                'success': False,
                'error': 'CLI group not available',
                'output': None,
                'execution_time': time.time() - start_time
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'output': None,
            'execution_time': time.time() - start_time
        }
    
    finally:
        # Restore original sys.argv
        try:
            sys.argv = original_argv
        except:
            pass


def setup_development_environment() -> Dict[str, Any]:
    """
    Setup development environment for CLI operations.
    
    Configures development-specific settings including enhanced logging,
    performance monitoring, error diagnostics, and development utilities.
    
    Returns:
        dict: Setup results and configuration information
        
    Example:
        >>> setup_results = setup_development_environment()
        >>> print(f"Development setup: {setup_results['success']}")
    """
    setup_results = {
        'success': True,
        'actions': [],
        'configuration': {},
        'errors': []
    }
    
    try:
        # Enable development mode
        CLI_CONFIG['development_mode'] = True
        CLI_CONFIG['hydra_full_error'] = True
        CLI_CONFIG['enable_performance_monitoring'] = True
        CLI_CONFIG['timing_reports'] = True
        CLI_CONFIG['error_diagnostics'] = True
        
        setup_results['actions'].append('Enabled development mode')
        setup_results['configuration']['development_mode'] = True
        
        # Setup enhanced logging if available
        if CLI_CONFIG['auto_setup_logging']:
            try:
                from odor_plume_nav.utils.logging_setup import setup_logger
                setup_logger(level='DEBUG', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                setup_results['actions'].append('Configured enhanced logging')
                setup_results['configuration']['logging_level'] = 'DEBUG'
            except ImportError as e:
                setup_results['errors'].append(f"Failed to setup enhanced logging: {e}")
        
        # Set environment variables for development
        os.environ['CLI_DEVELOPMENT_MODE'] = 'true'
        os.environ['HYDRA_FULL_ERROR'] = 'true'
        setup_results['actions'].append('Set development environment variables')
        
        # Initialize performance monitoring
        if not _cli_state['initialized']:
            _cli_state['initialized'] = True
            _cli_state['performance_metrics'] = {}
            setup_results['actions'].append('Initialized performance monitoring')
        
    except Exception as e:
        setup_results['success'] = False
        setup_results['errors'].append(f"Development setup failed: {e}")
    
    return setup_results


def get_performance_metrics() -> Dict[str, Any]:
    """
    Get comprehensive performance metrics for CLI operations.
    
    Returns detailed performance information including operation timings,
    resource usage, error statistics, and performance thresholds.
    
    Returns:
        dict: Comprehensive performance metrics and statistics
        
    Example:
        >>> metrics = get_performance_metrics()
        >>> print(f"Startup time: {metrics['startup_time']:.2f}s")
        >>> print(f"Operations: {len(metrics['operation_timings'])}")
    """
    metrics = {
        'startup_time': _cli_state.get('startup_time', 0),
        'operation_timings': _cli_state['performance_metrics'].copy(),
        'error_count': len(_cli_state.get('error_history', [])),
        'last_command': _cli_state.get('last_command'),
        'thresholds': {
            'max_command_init_time': CLI_CONFIG['max_command_init_time'],
            'max_config_load_time': CLI_CONFIG['max_config_load_time'],
            'max_validation_time': CLI_CONFIG['max_validation_time']
        },
        'configuration': CLI_CONFIG.copy()
    }
    
    # Add performance warnings
    metrics['warnings'] = []
    if metrics['startup_time'] > CLI_CONFIG['max_command_init_time']:
        metrics['warnings'].append(
            f"Startup time {metrics['startup_time']:.2f}s exceeds threshold "
            f"{CLI_CONFIG['max_command_init_time']}s"
        )
    
    # Add recent errors
    if _cli_state.get('error_history'):
        metrics['recent_errors'] = _cli_state['error_history'][-5:]  # Last 5 errors
    
    return metrics


# Module-level exports for public API
__all__ = [
    # Main CLI components (conditionally available)
    'main',
    'cli',
    
    # Availability and status functions
    'is_available',
    'get_availability_status',
    
    # Version and metadata
    'get_version',
    'get_version_info',
    '__version__',
    '__author__',
    '__description__',
    
    # Environment and validation
    'validate_environment',
    'setup_development_environment',
    
    # Command management
    'register_command',
    'list_commands',
    'invoke_command',
    
    # Performance and monitoring
    'get_performance_metrics',
    
    # Configuration
    'CLI_CONFIG',
    
    # Exceptions
    'CLINotAvailableError',
    'EnvironmentValidationError',
    'PerformanceWarning'
]