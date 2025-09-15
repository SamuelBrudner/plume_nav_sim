"""
Main package initialization module for plume_nav_sim providing comprehensive public API interface, 
environment registration, type definitions, exception handling, and package metadata management. 
Exposes PlumeSearchEnv, registration utilities, core types, constants, and error handling for 
Gymnasium-compatible reinforcement learning plume navigation research with streamlined imports 
and consistent versioning.

This module serves as the primary entry point for the plume_nav_sim package, providing a unified
interface to all core functionality including environment creation, registration with Gymnasium,
type definitions, error handling, and package metadata. It implements comprehensive initialization
procedures and convenience functions for rapid research environment setup.
"""

# External imports with version comments for dependency management and compatibility tracking
import warnings  # >=3.10 - Warning system for deprecation notices and compatibility issues during package initialization
import logging  # >=3.10 - Package-level logger setup and initialization logging for development debugging
import sys
import os
from typing import Optional, Dict, Any, Tuple, Union
import time

# Internal imports for core environment functionality and complete API exposure
from .envs.plume_search_env import (
    PlumeSearchEnv,
    create_plume_search_env,
    validate_plume_search_config
)

# Internal imports for environment registration and Gymnasium integration
from .registration.register import (
    register_env,
    unregister_env,
    is_registered,
    get_registration_info,
    register_with_custom_params,
    ENV_ID,
    create_registration_kwargs,
    validate_registration_config
)

# Internal imports for core type definitions and data structures
from .core.types import (
    Action,
    RenderMode,
    Coordinates,
    GridSize,
    AgentState,
    EnvironmentConfig,
    create_coordinates,
    create_grid_size,
    validate_action,
    PlumeParameters,
    EpisodeState,
    StateSnapshot,
    create_agent_state,
    create_environment_config,
    validate_coordinates,
    validate_grid_size,
    calculate_euclidean_distance,
    create_state_snapshot
)

# Internal imports for system constants and configuration values
from .core.constants import (
    PACKAGE_NAME,
    PACKAGE_VERSION,
    ENVIRONMENT_ID,
    DEFAULT_GRID_SIZE,
    DEFAULT_SOURCE_LOCATION,
    DEFAULT_MAX_STEPS,
    DEFAULT_GOAL_RADIUS,
    DEFAULT_SIGMA,
    get_default_environment_constants,
    get_default_plume_constants,
    get_performance_constants,
    validate_constant_consistency
)

# Internal imports for comprehensive exception handling
from .utils.exceptions import (
    PlumeNavSimError,
    ValidationError,
    StateError,
    RenderingError,
    ConfigurationError,
    ComponentError,
    ResourceError,
    IntegrationError,
    ErrorSeverity,
    ErrorContext,
    handle_component_error,
    sanitize_error_context,
    format_error_details,
    create_error_context,
    log_exception_with_recovery
)

# Package metadata and version information for programmatic access
__version__ = PACKAGE_VERSION
__author__ = 'plume_nav_sim Development Team'
__email__ = 'plume-nav-sim@example.com'
__license__ = 'MIT'
__status__ = 'Development'
__package_name__ = PACKAGE_NAME

# Package initialization state tracking and logger setup
_package_initialized = False
_logger = logging.getLogger(__name__)

# Comprehensive public API exports for complete package functionality
__all__ = [
    # Core environment class and factory functions
    'PlumeSearchEnv',
    'create_plume_search_env', 
    'validate_plume_search_config',
    
    # Environment registration and Gymnasium integration
    'register_env',
    'unregister_env',
    'is_registered',
    'get_registration_info',
    'register_with_custom_params',
    'create_registration_kwargs',
    'validate_registration_config',
    'ENV_ID',
    'ENVIRONMENT_ID',
    
    # Core type definitions and data structures
    'Action',
    'RenderMode', 
    'Coordinates',
    'GridSize',
    'AgentState',
    'EnvironmentConfig',
    'PlumeParameters',
    'EpisodeState',
    'StateSnapshot',
    
    # Type factory functions and validation utilities
    'create_coordinates',
    'create_grid_size',
    'create_agent_state',
    'create_environment_config',
    'create_state_snapshot',
    'validate_action',
    'validate_coordinates',
    'validate_grid_size',
    'calculate_euclidean_distance',
    
    # Exception handling classes and utilities
    'PlumeNavSimError',
    'ValidationError',
    'StateError', 
    'RenderingError',
    'ConfigurationError',
    'ComponentError',
    'ResourceError',
    'IntegrationError',
    'ErrorSeverity',
    'ErrorContext',
    'handle_component_error',
    'sanitize_error_context',
    'format_error_details',
    'create_error_context',
    'log_exception_with_recovery',
    
    # System constants and configuration values
    'DEFAULT_GRID_SIZE',
    'DEFAULT_SOURCE_LOCATION',
    'DEFAULT_MAX_STEPS', 
    'DEFAULT_GOAL_RADIUS',
    'DEFAULT_SIGMA',
    'get_default_environment_constants',
    'get_default_plume_constants',
    'get_performance_constants',
    'validate_constant_consistency',
    
    # Package management and convenience functions
    'get_version',
    'get_package_info',
    'initialize_package',
    'quick_start',
    'create_example_environment'
]


def get_version() -> str:
    """
    Returns the current package version string for programmatic version checking and compatibility 
    validation in research environments and external integrations.
    
    This function provides programmatic access to the package version following semantic versioning
    standards for compatibility checking and system integration requirements.
    
    Returns:
        str: Package version string following semantic versioning (e.g., '0.0.1') for compatibility checking
        
    Example:
        # Check package version programmatically
        version = get_version()
        print(f"plume_nav_sim version: {version}")
        
        # Version compatibility checking
        if get_version().startswith('0.0'):
            print("Development version detected")
    """
    # Return __version__ global containing PACKAGE_VERSION constant
    return __version__


def get_package_info(include_environment_info: bool = False) -> Dict[str, Any]:
    """
    Returns comprehensive package metadata dictionary including version, author, license, and 
    environment information for debugging and system analysis.
    
    This function provides detailed package information for debugging, system analysis, and
    administrative purposes, optionally including runtime environment details.
    
    Args:
        include_environment_info (bool): Whether to include Python version and dependencies information
        
    Returns:
        dict: Package metadata dictionary with version, author, license, and optional environment information
        
    Example:
        # Basic package information
        info = get_package_info()
        print(f"Package: {info['package_name']} v{info['version']}")
        
        # Detailed environment analysis
        detailed_info = get_package_info(include_environment_info=True)
        print(f"Python: {detailed_info['python_version']}")
    """
    # Create base package info dictionary with version, author, email, license, and status
    package_info = {
        'package_name': __package_name__,
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'status': __status__
    }
    
    # Add package name and environment ID for identification
    package_info.update({
        'environment_id': ENVIRONMENT_ID,
        'default_grid_size': DEFAULT_GRID_SIZE,
        'default_source_location': DEFAULT_SOURCE_LOCATION,
        'package_initialized': _package_initialized
    })
    
    # Include Python version and dependencies information if include_environment_info is True
    if include_environment_info:
        try:
            # Add Python version and platform information
            import platform
            package_info.update({
                'python_version': sys.version,
                'platform': platform.platform(),
                'python_executable': sys.executable
            })
            
            # Add dependency versions if available
            try:
                import gymnasium
                package_info['gymnasium_version'] = gymnasium.__version__
            except ImportError:
                package_info['gymnasium_version'] = 'not available'
            
            try:
                import numpy
                package_info['numpy_version'] = numpy.__version__
            except ImportError:
                package_info['numpy_version'] = 'not available'
                
            try:
                import matplotlib
                package_info['matplotlib_version'] = matplotlib.__version__
            except ImportError:
                package_info['matplotlib_version'] = 'not available'
                
        except Exception as e:
            package_info['environment_info_error'] = str(e)
    
    # Add installation path and configuration details for debugging
    package_info['module_path'] = os.path.dirname(__file__)
    
    # Return comprehensive package information dictionary
    return package_info


def initialize_package(
    enable_warnings: bool = True, 
    validate_dependencies: bool = True, 
    log_level: Optional[str] = None
) -> bool:
    """
    Initializes package-level configuration, logging setup, and compatibility checking with optional 
    validation and warning management for research environment setup.
    
    This function performs comprehensive package initialization including logging configuration,
    dependency validation, and system compatibility checking with configurable warning management.
    
    Args:
        enable_warnings (bool): Whether to enable warning messages for compatibility and deprecation issues
        validate_dependencies (bool): Whether to validate external dependencies (gymnasium, numpy, matplotlib)
        log_level (Optional[str]): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR'), defaults to 'INFO'
        
    Returns:
        bool: True if package initialization successful, False if issues detected but recoverable
        
    Raises:
        IntegrationError: If critical dependencies are missing or incompatible
        ConfigurationError: If package configuration is invalid
        
    Example:
        # Basic package initialization
        success = initialize_package()
        if success:
            print("Package ready for use")
            
        # Development initialization with debugging
        initialize_package(
            enable_warnings=True,
            validate_dependencies=True,
            log_level='DEBUG'
        )
    """
    global _package_initialized
    
    try:
        # Set up package-level logger with specified log_level or default INFO level
        effective_log_level = log_level or 'INFO'
        log_level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        if effective_log_level in log_level_map:
            _logger.setLevel(log_level_map[effective_log_level])
        
        # Log package initialization start with version and configuration information
        _logger.info(f"Initializing {__package_name__} v{__version__}")
        _logger.debug(f"Initialization parameters: warnings={enable_warnings}, validate_deps={validate_dependencies}")
        
        # Validate external dependencies (gymnasium, numpy, matplotlib) if validate_dependencies is True
        if validate_dependencies:
            missing_dependencies = []
            
            try:
                import gymnasium
                _logger.debug(f"Gymnasium version: {gymnasium.__version__}")
            except ImportError:
                missing_dependencies.append('gymnasium')
                
            try:
                import numpy
                _logger.debug(f"NumPy version: {numpy.__version__}")
            except ImportError:
                missing_dependencies.append('numpy')
                
            try:
                import matplotlib
                _logger.debug(f"Matplotlib version: {matplotlib.__version__}")
            except ImportError:
                # Matplotlib is optional for some use cases
                _logger.warning("Matplotlib not available - rendering capabilities will be limited")
            
            if missing_dependencies:
                error_msg = f"Critical dependencies missing: {missing_dependencies}"
                _logger.error(error_msg)
                raise IntegrationError(
                    error_msg,
                    dependency_name=', '.join(missing_dependencies)
                )
        
        # Check Python version compatibility against minimum requirements (3.10+)
        python_version = sys.version_info
        if python_version < (3, 10):
            error_msg = f"Python {python_version.major}.{python_version.minor} is not supported. Minimum required: Python 3.10"
            _logger.error(error_msg)
            raise IntegrationError(
                error_msg,
                dependency_name="python",
                required_version="3.10+",
                current_version=f"{python_version.major}.{python_version.minor}"
            )
        
        # Configure warning filters if enable_warnings is False to suppress non-critical warnings
        if not enable_warnings:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            _logger.debug("Non-critical warnings suppressed")
        
        # Validate core constants consistency using validate_constant_consistency function
        try:
            is_valid = validate_constant_consistency()
            if not is_valid:
                _logger.warning("Constant consistency validation detected potential issues")
        except Exception as const_error:
            _logger.warning(f"Constant validation failed: {const_error}")
        
        # Set _package_initialized global flag to True upon successful initialization
        _package_initialized = True
        
        # Log successful initialization with package readiness confirmation
        _logger.info(f"{__package_name__} v{__version__} initialization completed successfully")
        _logger.debug(f"Environment ID: {ENVIRONMENT_ID}")
        
        # Return True if initialization completed successfully
        return True
        
    except Exception as e:
        _logger.error(f"Package initialization failed: {e}")
        _package_initialized = False
        raise


def quick_start(
    env_config: Optional[Dict[str, Any]] = None,
    auto_register: bool = True,
    validate_setup: bool = True
) -> PlumeSearchEnv:
    """
    Convenience function for rapid environment setup and registration providing streamlined access 
    to PlumeSearchEnv for educational and demonstration purposes.
    
    This function provides a streamlined interface for creating and configuring plume navigation
    environments with automatic registration and validation for immediate research use.
    
    Args:
        env_config (Optional[dict]): Environment configuration parameters, uses defaults if not provided
        auto_register (bool): Whether to automatically register environment with Gymnasium
        validate_setup (bool): Whether to validate environment configuration and setup
        
    Returns:
        PlumeSearchEnv: Ready-to-use environment instance with registration and validation complete
        
    Raises:
        ConfigurationError: If environment configuration is invalid
        ValidationError: If setup validation fails
        
    Example:
        # Quick setup with defaults
        env = quick_start()
        obs, info = env.reset()
        
        # Custom configuration with validation
        env = quick_start(
            env_config={"grid_size": (256, 256), "source_location": (128, 128)},
            auto_register=True,
            validate_setup=True
        )
    """
    try:
        # Initialize package using initialize_package() if not already initialized
        if not _package_initialized:
            _logger.info("Package not initialized, initializing with default settings")
            initialize_package(enable_warnings=False, validate_dependencies=True)
        
        # Apply default environment configuration and merge with env_config if provided
        default_config = {
            'grid_size': DEFAULT_GRID_SIZE,
            'source_location': DEFAULT_SOURCE_LOCATION,
            'max_steps': DEFAULT_MAX_STEPS,
            'goal_radius': DEFAULT_GOAL_RADIUS,
            'sigma': DEFAULT_SIGMA
        }
        
        effective_config = default_config.copy()
        if env_config:
            effective_config.update(env_config)
        
        _logger.debug(f"Quick start configuration: {effective_config}")
        
        # Validate environment configuration using validate_plume_search_config if validate_setup is True
        if validate_setup:
            try:
                is_valid = validate_plume_search_config(effective_config)
                if not is_valid:
                    raise ConfigurationError(
                        "Environment configuration validation failed",
                        config_parameter="env_config",
                        invalid_value=effective_config
                    )
            except Exception as validation_error:
                _logger.error(f"Configuration validation failed: {validation_error}")
                raise
        
        # Register environment with Gymnasium using register_env() if auto_register is True
        if auto_register:
            if not is_registered(ENV_ID):
                _logger.info(f"Auto-registering environment: {ENV_ID}")
                register_env(kwargs=effective_config)
            else:
                _logger.debug(f"Environment {ENV_ID} already registered")
        
        # Create PlumeSearchEnv instance using create_plume_search_env with validated configuration
        environment = create_plume_search_env(**effective_config)
        
        # Log quick start completion with environment ID and configuration summary
        _logger.info(f"Quick start completed for environment with config: {effective_config}")
        
        # Return ready-to-use PlumeSearchEnv instance for immediate research use
        return environment
        
    except Exception as e:
        _logger.error(f"Quick start failed: {e}")
        raise


def create_example_environment(
    example_type: str = 'tutorial',
    custom_params: Optional[Dict[str, Any]] = None,
    include_visualization: bool = False
) -> Tuple[PlumeSearchEnv, Dict[str, Any]]:
    """
    Factory function to create pre-configured example environments for tutorials, demonstrations, 
    and educational purposes with various difficulty levels and configurations.
    
    This function provides pre-configured example environments designed for different use cases
    including tutorials, research demonstrations, and performance testing with comprehensive
    configuration documentation.
    
    Args:
        example_type (str): Type of example environment ('tutorial', 'research', 'performance_test')
        custom_params (Optional[dict]): Custom parameter overrides for example configuration
        include_visualization (bool): Whether to enable visualization modes for interactive use
        
    Returns:
        tuple: Tuple of (environment, configuration_info) with ready environment and setup details
        
    Raises:
        ValidationError: If example_type is not recognized or parameters are invalid
        ConfigurationError: If example configuration cannot be created
        
    Example:
        # Create tutorial environment
        env, info = create_example_environment('tutorial')
        print(f"Tutorial setup: {info['description']}")
        
        # Create research environment with custom parameters
        env, info = create_example_environment(
            example_type='research',
            custom_params={'grid_size': (256, 256)},
            include_visualization=True
        )
    """
    try:
        # Validate example_type against available examples: 'tutorial', 'research', 'performance_test'
        valid_examples = ['tutorial', 'research', 'performance_test']
        if example_type not in valid_examples:
            raise ValidationError(
                f"Invalid example_type '{example_type}'. Must be one of: {valid_examples}",
                parameter_name="example_type",
                invalid_value=example_type,
                expected_format=f"One of: {valid_examples}"
            )
        
        # Load pre-configured parameters for specified example_type
        example_configs = {
            'tutorial': {
                'grid_size': (64, 64),
                'source_location': (32, 32),
                'max_steps': 500,
                'goal_radius': 3.0,
                'sigma': 8.0,
                'description': 'Small grid tutorial environment for learning basic navigation',
                'difficulty': 'beginner'
            },
            'research': {
                'grid_size': DEFAULT_GRID_SIZE,
                'source_location': DEFAULT_SOURCE_LOCATION,
                'max_steps': DEFAULT_MAX_STEPS,
                'goal_radius': DEFAULT_GOAL_RADIUS,
                'sigma': DEFAULT_SIGMA,
                'description': 'Standard research environment with default parameters',
                'difficulty': 'intermediate'
            },
            'performance_test': {
                'grid_size': (512, 512),
                'source_location': (256, 256),
                'max_steps': 5000,
                'goal_radius': 5.0,
                'sigma': 20.0,
                'description': 'Large grid environment for performance testing and scalability analysis',
                'difficulty': 'advanced'
            }
        }
        
        base_config = example_configs[example_type].copy()
        
        # Apply custom_params overrides if provided with parameter validation
        if custom_params:
            if not isinstance(custom_params, dict):
                raise ValidationError(
                    "custom_params must be a dictionary",
                    parameter_name="custom_params",
                    invalid_value=custom_params
                )
            
            # Validate and merge custom parameters
            for key, value in custom_params.items():
                if key in ['description', 'difficulty']:
                    base_config[key] = value
                elif key == 'grid_size':
                    base_config[key] = validate_grid_size(value) if value else base_config[key]
                elif key == 'source_location':
                    base_config[key] = validate_coordinates(value) if value else base_config[key]
                else:
                    base_config[key] = value
        
        # Set appropriate render_mode based on include_visualization flag
        render_mode = RenderMode.HUMAN if include_visualization else RenderMode.RGB_ARRAY
        
        # Create environment using create_plume_search_env with example configuration
        env_params = {k: v for k, v in base_config.items() 
                     if k not in ['description', 'difficulty']}
        
        environment = create_plume_search_env(**env_params)
        
        # Register environment with example-specific ID if needed
        example_env_id = f"PlumeNav-{example_type.title()}-v0"
        try:
            if not is_registered(example_env_id):
                register_env(
                    env_id=example_env_id,
                    kwargs=env_params
                )
                _logger.debug(f"Registered example environment: {example_env_id}")
        except Exception as reg_error:
            _logger.warning(f"Failed to register example environment: {reg_error}")
        
        # Generate configuration_info dictionary with example description and parameters
        configuration_info = {
            'example_type': example_type,
            'description': base_config.get('description', f'{example_type} example environment'),
            'difficulty': base_config.get('difficulty', 'unknown'),
            'parameters': env_params,
            'render_mode': render_mode.value,
            'environment_id': example_env_id,
            'creation_timestamp': time.time(),
            'include_visualization': include_visualization
        }
        
        _logger.info(f"Created {example_type} example environment: {configuration_info['description']}")
        
        # Return tuple of configured environment and detailed configuration information
        return environment, configuration_info
        
    except Exception as e:
        _logger.error(f"Failed to create example environment '{example_type}': {e}")
        raise


# Automatic package initialization on import with error handling and logging
try:
    # Check if running in development or test environment
    is_dev_environment = (
        'pytest' in sys.modules or 
        'unittest' in sys.modules or
        os.environ.get('PLUME_NAV_SIM_DEV', '').lower() == 'true'
    )
    
    # Initialize package with appropriate settings for development vs production
    if is_dev_environment:
        # Development initialization with enhanced logging
        initialize_package(
            enable_warnings=True,
            validate_dependencies=True,
            log_level='DEBUG'
        )
    else:
        # Production initialization with standard settings
        initialize_package(
            enable_warnings=False,
            validate_dependencies=True,
            log_level='INFO'
        )
        
    _logger.debug(f"Package {__package_name__} v{__version__} imported successfully")
    
except Exception as init_error:
    # Log initialization errors but allow import to continue
    logging.getLogger(__name__).warning(f"Package initialization failed during import: {init_error}")
    _package_initialized = False