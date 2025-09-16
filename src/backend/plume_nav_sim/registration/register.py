"""
Core Gymnasium environment registration module implementing the complete registration system 
for PlumeNav-StaticGaussian-v0 environment with comprehensive parameter validation, configuration 
management, version control, and error handling. Provides primary registration functions for 
Gymnasium compatibility including register_env(), unregister_env(), is_registered(), and 
registration status management with strict versioning compliance and entry point specification.

This module serves as the primary interface for registering and managing the PlumeNav-StaticGaussian-v0
environment within the Gymnasium ecosystem, ensuring proper integration with gym.make() calls and
providing comprehensive parameter validation, error handling, and configuration management.
"""

# External imports with version comments for dependency management and compatibility tracking
import gymnasium  # >=0.29.0 - Reinforcement learning environment framework providing register() function, environment registry, and gym.make() compatibility for standard RL environment registration
from typing import Optional, Dict, Any, Tuple, Union  # >=3.10 - Type hints for function parameters, return types, and optional parameter specifications ensuring type safety and documentation clarity
import warnings  # >=3.10 - Warning system for registration conflicts, deprecation notices, and compatibility issues during environment registration
import time
import copy

# Internal imports for configuration constants and system integration
from ..core.constants import (
    ENVIRONMENT_ID,
    DEFAULT_GRID_SIZE, 
    DEFAULT_SOURCE_LOCATION,
    DEFAULT_MAX_STEPS,
    DEFAULT_GOAL_RADIUS
)

# Internal imports for error handling and logging integration
from ..utils.exceptions import (
    ConfigurationError,
    ValidationError
)
from ..utils.logging import get_component_logger

# Global constants for registration system configuration and environment identification
ENV_ID = ENVIRONMENT_ID  # Primary environment identifier 'PlumeNav-StaticGaussian-v0' for Gymnasium registration compliance
ENTRY_POINT = 'plume_nav_sim.envs.plume_search_env:PlumeSearchEnv'  # Entry point specification string for Gymnasium registration defining exact module path and class location
MAX_EPISODE_STEPS = DEFAULT_MAX_STEPS  # Default maximum episode steps (1000) for registration parameter configuration

# Component logger for registration system debugging and operation tracking
_logger = get_component_logger('registration')

# Registration cache for tracking environment status and preventing duplicate registrations
_registration_cache: Dict[str, Dict[str, Any]] = {}

# Public API exports for comprehensive registration functionality
__all__ = [
    'register_env',
    'unregister_env', 
    'is_registered',
    'get_registration_info',
    'create_registration_kwargs',
    'validate_registration_config',
    'register_with_custom_params',
    'ENV_ID',
    'ENTRY_POINT'
]


def register_env(
    env_id: Optional[str] = None,
    entry_point: Optional[str] = None, 
    max_episode_steps: Optional[int] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    force_reregister: bool = False
) -> str:
    """
    Main environment registration function for Gymnasium compatibility with comprehensive parameter 
    validation, configuration management, version control, and error handling ensuring proper 
    PlumeNav-StaticGaussian-v0 environment registration.
    
    This function serves as the primary interface for registering the plume navigation environment
    with Gymnasium, providing comprehensive parameter validation, error handling, and integration
    with the gym.make() ecosystem. It supports both default and custom configuration parameters
    while ensuring strict compliance with Gymnasium versioning conventions.
    
    Args:
        env_id: Environment identifier string, defaults to ENV_ID constant if not provided
        entry_point: Module path string for environment class, defaults to ENTRY_POINT if not provided  
        max_episode_steps: Maximum steps per episode, defaults to MAX_EPISODE_STEPS if not provided
        kwargs: Additional environment parameters dictionary for customization
        force_reregister: Whether to force re-registration if environment already exists
        
    Returns:
        Registered environment ID string ready for immediate use with gym.make() calls
        
    Raises:
        ValidationError: If environment ID format is invalid or parameters fail validation
        ConfigurationError: If registration configuration is invalid or conflicts exist
        
    Example:
        # Basic registration with defaults
        env_id = register_env()
        env = gym.make(env_id)
        
        # Custom registration with parameters  
        env_id = register_env(
            env_id="CustomPlume-v0",
            kwargs={"grid_size": (256, 256), "source_location": (128, 128)}
        )
    """
    try:
        # Apply default values using ENV_ID, ENTRY_POINT, MAX_EPISODE_STEPS if parameters not provided
        effective_env_id = env_id or ENV_ID
        effective_entry_point = entry_point or ENTRY_POINT  
        effective_max_steps = max_episode_steps or MAX_EPISODE_STEPS
        effective_kwargs = kwargs or {}
        
        _logger.debug(f"Starting registration for environment: {effective_env_id}")
        
        # Validate environment ID follows Gymnasium versioning conventions with '-v0' suffix pattern
        if not effective_env_id.endswith('-v0'):
            raise ValidationError(
                f"Environment ID '{effective_env_id}' must end with '-v0' suffix for Gymnasium versioning compliance",
                parameter_name="env_id",
                invalid_value=effective_env_id,
                expected_format="environment_name-v0"
            )
        
        # Check if environment already registered using is_registered() with cache validation
        if is_registered(effective_env_id, use_cache=True):
            if not force_reregister:
                _logger.warning(f"Environment '{effective_env_id}' already registered. Use force_reregister=True to override.")
                return effective_env_id
            else:
                # Handle force_reregister flag by calling unregister_env() if environment exists and force requested
                _logger.info(f"Force re-registration requested for '{effective_env_id}', unregistering existing...")
                unregister_env(effective_env_id, suppress_warnings=True)
        
        # Create complete kwargs dictionary using create_registration_kwargs() with parameter validation
        registration_kwargs = create_registration_kwargs(
            grid_size=effective_kwargs.get('grid_size'),
            source_location=effective_kwargs.get('source_location'), 
            max_steps=effective_kwargs.get('max_steps'),
            goal_radius=effective_kwargs.get('goal_radius'),
            additional_kwargs=effective_kwargs
        )
        
        # Validate registration configuration using validate_registration_config() for consistency checking
        is_valid, validation_report = validate_registration_config(
            env_id=effective_env_id,
            entry_point=effective_entry_point,
            max_episode_steps=None,
            kwargs=registration_kwargs,
            strict_validation=True
        )
        
        if not is_valid:
            error_details = validation_report.get('errors', [])
            raise ConfigurationError(
                f"Registration configuration validation failed: {error_details}",
                config_parameter="registration_config",
                invalid_value=validation_report
            )
        
        # Call gymnasium.register() with validated env_id, entry_point, max_episode_steps, and kwargs
        gymnasium.register(
            id=effective_env_id,
            entry_point=effective_entry_point,
            max_episode_steps=None,
            kwargs=registration_kwargs
        )
        
        # Update registration cache with successful registration information and timestamp
        _registration_cache[effective_env_id] = {
            'registered': True,
            'entry_point': effective_entry_point,
            'max_episode_steps': effective_max_steps,
            'kwargs': copy.deepcopy(registration_kwargs),
            'registration_timestamp': time.time(),
            'validation_report': validation_report
        }
        
        # Log successful environment registration with configuration details and usage instructions
        _logger.info(f"Successfully registered environment '{effective_env_id}' with entry_point '{effective_entry_point}'")
        _logger.debug(f"Registration parameters: max_steps={effective_max_steps}, kwargs={registration_kwargs}")
        
        # Test environment creation using gym.make() to verify registration integrity and functionality
        try:
            test_env = gymnasium.make(effective_env_id)
            test_env.close()
            _logger.debug(f"Registration integrity verified for '{effective_env_id}'")
        except Exception as test_error:
            _logger.error(f"Registration verification failed for '{effective_env_id}': {test_error}")
            # Unregister the faulty environment
            unregister_env(effective_env_id, suppress_warnings=True)
            raise ConfigurationError(
                f"Environment registration verification failed: {test_error}",
                config_parameter="registration_integrity"
            ) from test_error
        
        # Return registered environment ID for immediate use with comprehensive success confirmation
        return effective_env_id
        
    except Exception as e:
        _logger.error(f"Environment registration failed for '{env_id or ENV_ID}': {e}")
        raise


def unregister_env(
    env_id: Optional[str] = None,
    suppress_warnings: bool = False
) -> bool:
    """
    Environment unregistration function for cleanup and testing workflows with comprehensive 
    cache management and error handling ensuring proper environment removal from Gymnasium registry.
    
    This function provides a clean mechanism for removing environments from the Gymnasium registry,
    supporting testing workflows and cleanup operations. It handles both cache management and
    registry cleanup with comprehensive error handling and validation.
    
    Args:
        env_id: Environment identifier to unregister, defaults to ENV_ID if not provided
        suppress_warnings: Whether to suppress warnings about unregistration operations
        
    Returns:
        True if environment was successfully unregistered or was not registered, False if unregistration failed
        
    Raises:
        ValidationError: If environment ID format is invalid
        
    Example:
        # Unregister default environment
        success = unregister_env()
        
        # Unregister specific environment with warnings suppressed
        success = unregister_env("CustomPlume-v0", suppress_warnings=True)
    """
    try:
        # Apply default env_id using ENV_ID constant if parameter not provided
        effective_env_id = env_id or ENV_ID
        
        _logger.debug(f"Starting unregistration for environment: {effective_env_id}")
        
        # Check current registration status using is_registered() with cache consultation
        currently_registered = is_registered(effective_env_id, use_cache=True)
        
        # Log unregistration attempt with environment ID and current registration status
        if not currently_registered:
            if not suppress_warnings:
                _logger.warning(f"Environment '{effective_env_id}' is not currently registered")
            return True
        
        # Remove environment from Gymnasium registry using gymnasium.envs.registry.env_specs.pop()
        try:
            if hasattr(gymnasium.envs, 'registry') and hasattr(gymnasium.envs.registry, 'env_specs'):
                removed_spec = gymnasium.envs.registry.env_specs.pop(effective_env_id, None)
                if removed_spec:
                    _logger.debug(f"Removed environment spec for '{effective_env_id}' from Gymnasium registry")
            else:
                # Fallback method for different Gymnasium versions
                if hasattr(gymnasium, 'envs') and hasattr(gymnasium.envs, 'registration'):
                    if effective_env_id in gymnasium.envs.registration.registry.env_specs:
                        del gymnasium.envs.registration.registry.env_specs[effective_env_id]
                        
        except Exception as registry_error:
            _logger.warning(f"Error accessing Gymnasium registry during unregistration: {registry_error}")
            # Continue with cache cleanup even if registry access fails
        
        # Clear registration cache entry for the specified environment ID
        if effective_env_id in _registration_cache:
            del _registration_cache[effective_env_id]
            _logger.debug(f"Cleared cache entry for '{effective_env_id}'")
        
        # Issue warnings about unregistration unless suppress_warnings is True
        if not suppress_warnings:
            _logger.info(f"Environment '{effective_env_id}' has been unregistered")
        
        # Verify successful unregistration by checking registry and cache consistency
        still_registered = is_registered(effective_env_id, use_cache=False)
        if still_registered:
            _logger.error(f"Unregistration verification failed for '{effective_env_id}'")
            return False
        
        # Log successful unregistration with cleanup confirmation and status update
        _logger.debug(f"Successfully unregistered environment '{effective_env_id}'")
        
        # Return unregistration status with success/failure indication for caller feedback
        return True
        
    except Exception as e:
        _logger.error(f"Environment unregistration failed for '{env_id or ENV_ID}': {e}")
        return False


def is_registered(
    env_id: Optional[str] = None,
    use_cache: bool = True
) -> bool:
    """
    Registration status checking function with comprehensive cache validation, registry consistency 
    checking, and error handling providing accurate environment availability information.
    
    This function provides reliable status checking for environment registration, supporting both
    cached and authoritative registry queries with comprehensive validation and consistency checking.
    
    Args:
        env_id: Environment identifier to check, defaults to ENV_ID if not provided
        use_cache: Whether to use cached registration information for faster queries
        
    Returns:
        True if environment is properly registered and available, False otherwise
        
    Example:
        # Quick cache-based check
        if is_registered():
            env = gym.make(ENV_ID)
            
        # Authoritative registry check
        if is_registered("CustomPlume-v0", use_cache=False):
            print("Environment confirmed in registry")
    """
    try:
        # Apply default env_id using ENV_ID constant if parameter not provided  
        effective_env_id = env_id or ENV_ID
        
        # Check registration cache if use_cache is True and cache entry exists with validation
        if use_cache and effective_env_id in _registration_cache:
            cached_info = _registration_cache[effective_env_id]
            if cached_info.get('registered', False):
                _logger.debug(f"Cache hit: '{effective_env_id}' is registered")
                return True
        
        # Query Gymnasium registry directly using gymnasium.envs.registry.env_specs.get() for authoritative status
        try:
            if hasattr(gymnasium.envs, 'registry') and hasattr(gymnasium.envs.registry, 'env_specs'):
                registry_entry = gymnasium.envs.registry.env_specs.get(effective_env_id)
                is_in_registry = registry_entry is not None
            else:
                # Fallback method for different Gymnasium versions
                try:
                    test_env = gymnasium.make(effective_env_id)
                    test_env.close()
                    is_in_registry = True
                except Exception:
                    is_in_registry = False
        except Exception as registry_error:
            _logger.warning(f"Error querying Gymnasium registry: {registry_error}")
            is_in_registry = False
        
        # Validate cache consistency with registry state and update cache if discrepancies found
        if use_cache and effective_env_id in _registration_cache:
            cached_status = _registration_cache[effective_env_id].get('registered', False)
            if cached_status != is_in_registry:
                _logger.debug(f"Cache inconsistency detected for '{effective_env_id}', updating cache")
                if is_in_registry:
                    _registration_cache[effective_env_id]['registered'] = True
                    _registration_cache[effective_env_id]['last_verified'] = time.time()
                else:
                    if effective_env_id in _registration_cache:
                        del _registration_cache[effective_env_id]
        
        # Update registration cache with current registry state and timestamp for future queries
        if is_in_registry and effective_env_id not in _registration_cache:
            _registration_cache[effective_env_id] = {
                'registered': True,
                'last_verified': time.time(),
                'cache_source': 'registry_query'
            }
        
        # Log registration status check with environment ID and result for debugging support
        _logger.debug(f"Registration status for '{effective_env_id}': {is_in_registry}")
        
        # Return accurate registration status based on authoritative registry consultation
        return is_in_registry
        
    except Exception as e:
        _logger.error(f"Registration status check failed for '{env_id or ENV_ID}': {e}")
        return False


def get_registration_info(
    env_id: Optional[str] = None,
    include_config_details: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive registration information retrieval function providing detailed environment 
    metadata, configuration parameters, registration status, and debugging information for 
    monitoring and troubleshooting.
    
    This function provides complete registration information for debugging, monitoring, and
    administrative purposes, including detailed configuration analysis and system status.
    
    Args:
        env_id: Environment identifier to get information for, defaults to ENV_ID if not provided
        include_config_details: Whether to include detailed configuration parameter breakdown
        
    Returns:
        Complete registration information dictionary including status, configuration, metadata, and debugging details
        
    Example:
        # Basic registration info
        info = get_registration_info()
        print(f"Status: {info['registered']}")
        
        # Detailed configuration analysis
        detailed_info = get_registration_info(include_config_details=True)
        print(f"Config: {detailed_info['config_details']}")
    """
    try:
        # Apply default env_id using ENV_ID constant if parameter not provided
        effective_env_id = env_id or ENV_ID
        
        # Initialize registration info dictionary with environment ID and timestamp
        registration_info = {
            'env_id': effective_env_id,
            'query_timestamp': time.time(),
            'cache_available': effective_env_id in _registration_cache
        }
        
        # Check registration status using is_registered() and include in information dictionary
        is_currently_registered = is_registered(effective_env_id, use_cache=False)
        registration_info['registered'] = is_currently_registered
        
        if is_currently_registered:
            # Retrieve environment specification from Gymnasium registry if registered
            try:
                if hasattr(gymnasium.envs, 'registry') and hasattr(gymnasium.envs.registry, 'env_specs'):
                    env_spec = gymnasium.envs.registry.env_specs.get(effective_env_id)
                    if env_spec:
                        # Extract entry point, max_episode_steps, and kwargs from environment specification
                        registration_info.update({
                            'entry_point': getattr(env_spec, 'entry_point', 'unknown'),
                            'max_episode_steps': getattr(env_spec, 'max_episode_steps', None),
                            'spec_kwargs': getattr(env_spec, 'kwargs', {}),
                            'reward_threshold': getattr(env_spec, 'reward_threshold', None)
                        })
            except Exception as spec_error:
                registration_info['spec_retrieval_error'] = str(spec_error)
        
        # Include configuration details if include_config_details enabled with parameter breakdown
        if include_config_details:
            config_details = {
                'default_parameters': {
                    'grid_size': DEFAULT_GRID_SIZE,
                    'source_location': DEFAULT_SOURCE_LOCATION,
                    'max_steps': DEFAULT_MAX_STEPS,
                    'goal_radius': DEFAULT_GOAL_RADIUS
                },
                'entry_point_default': ENTRY_POINT,
                'env_id_default': ENV_ID
            }
            
            # Include validation information if available
            if effective_env_id in _registration_cache:
                cached_info = _registration_cache[effective_env_id]
                if 'validation_report' in cached_info:
                    config_details['last_validation'] = cached_info['validation_report']
                    
            registration_info['config_details'] = config_details
        
        # Add cache information including last check timestamp and cache validity status
        if effective_env_id in _registration_cache:
            cache_info = _registration_cache[effective_env_id].copy()
            registration_info['cache_info'] = {
                'registration_timestamp': cache_info.get('registration_timestamp'),
                'last_verified': cache_info.get('last_verified'),
                'cached_status': cache_info.get('registered', False),
                'cache_source': cache_info.get('cache_source', 'unknown')
            }
        
        # Include Gymnasium version and compatibility information for debugging support
        try:
            registration_info['gymnasium_version'] = gymnasium.__version__
            registration_info['registry_available'] = hasattr(gymnasium.envs, 'registry')
        except Exception:
            registration_info['gymnasium_info_error'] = 'Failed to retrieve Gymnasium information'
        
        # Compile comprehensive registration dictionary with metadata and operational details
        registration_info['system_info'] = {
            'total_cached_environments': len(_registration_cache),
            'query_method': 'authoritative_registry' if not registration_info['cache_available'] else 'cached_with_verification'
        }
        
        # Log information retrieval request with environment ID and detail level for monitoring
        _logger.debug(f"Retrieved registration info for '{effective_env_id}', detailed={include_config_details}")
        
        # Return complete registration information dictionary for debugging and administration
        return registration_info
        
    except Exception as e:
        _logger.error(f"Failed to retrieve registration info for '{env_id or ENV_ID}': {e}")
        return {
            'env_id': env_id or ENV_ID,
            'error': str(e),
            'query_timestamp': time.time(),
            'registered': False
        }


def create_registration_kwargs(
    grid_size: Optional[Tuple[int, int]] = None,
    source_location: Optional[Tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    goal_radius: Optional[float] = None,
    additional_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Registration kwargs factory function for Gymnasium register() calls with comprehensive 
    parameter validation, default value application, and configuration consistency ensuring 
    proper environment setup.
    
    This function creates validated parameter dictionaries for environment registration,
    providing comprehensive validation, default application, and consistency checking for
    all configuration parameters.
    
    Args:
        grid_size: Grid dimensions as (width, height) tuple, defaults to DEFAULT_GRID_SIZE
        source_location: Plume source coordinates as (x, y) tuple, defaults to DEFAULT_SOURCE_LOCATION
        max_steps: Maximum episode steps, defaults to DEFAULT_MAX_STEPS
        goal_radius: Goal detection radius, defaults to DEFAULT_GOAL_RADIUS
        additional_kwargs: Additional parameters dictionary for custom configuration
        
    Returns:
        Complete kwargs dictionary ready for gymnasium.register() call with validated parameters
        
    Raises:
        ValidationError: If parameter validation fails or constraints are violated
        
    Example:
        # Default parameters
        kwargs = create_registration_kwargs()
        
        # Custom configuration
        kwargs = create_registration_kwargs(
            grid_size=(256, 256),
            source_location=(128, 128),
            goal_radius=5.0
        )
    """
    try:
        # Apply default values using DEFAULT_GRID_SIZE, DEFAULT_SOURCE_LOCATION, DEFAULT_MAX_STEPS, DEFAULT_GOAL_RADIUS if not provided
        effective_grid_size = grid_size or DEFAULT_GRID_SIZE
        effective_source_location = source_location or DEFAULT_SOURCE_LOCATION
        effective_max_steps = max_steps or DEFAULT_MAX_STEPS
        effective_goal_radius = goal_radius if goal_radius is not None else DEFAULT_GOAL_RADIUS
        
        _logger.debug(f"Creating registration kwargs with parameters: grid_size={effective_grid_size}, source_location={effective_source_location}")
        
        # Validate grid_size parameter for positive integer tuple with reasonable dimensions
        if not isinstance(effective_grid_size, (tuple, list)) or len(effective_grid_size) != 2:
            raise ValidationError(
                "grid_size must be a tuple or list of exactly 2 elements",
                parameter_name="grid_size",
                invalid_value=effective_grid_size,
                expected_format="(width, height) tuple with positive integers"
            )
        
        width, height = effective_grid_size
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValidationError(
                "grid_size dimensions must be integers",
                parameter_name="grid_size",
                invalid_value=effective_grid_size
            )
        
        if width <= 0 or height <= 0:
            raise ValidationError(
                "grid_size dimensions must be positive integers",
                parameter_name="grid_size", 
                invalid_value=effective_grid_size
            )
        
        if width > 1024 or height > 1024:
            raise ValidationError(
                "grid_size dimensions exceed maximum allowed size (1024x1024)",
                parameter_name="grid_size",
                invalid_value=effective_grid_size
            )
        
        # Validate source_location coordinates within grid bounds with mathematical consistency
        if not isinstance(effective_source_location, (tuple, list)) or len(effective_source_location) != 2:
            raise ValidationError(
                "source_location must be a tuple or list of exactly 2 elements",
                parameter_name="source_location",
                invalid_value=effective_source_location,
                expected_format="(x, y) tuple with coordinates within grid bounds"
            )
        
        source_x, source_y = effective_source_location
        if not isinstance(source_x, (int, float)) or not isinstance(source_y, (int, float)):
            raise ValidationError(
                "source_location coordinates must be numeric",
                parameter_name="source_location",
                invalid_value=effective_source_location
            )
        
        if source_x < 0 or source_x >= width or source_y < 0 or source_y >= height:
            raise ValidationError(
                f"source_location coordinates must be within grid bounds: (0,0) to ({width-1},{height-1})",
                parameter_name="source_location",
                invalid_value=effective_source_location
            )
        
        # Validate max_steps parameter for positive integer within performance and memory constraints
        if not isinstance(effective_max_steps, int):
            raise ValidationError(
                "max_steps must be an integer",
                parameter_name="max_steps",
                invalid_value=effective_max_steps
            )
        
        if effective_max_steps <= 0:
            raise ValidationError(
                "max_steps must be a positive integer",
                parameter_name="max_steps",
                invalid_value=effective_max_steps
            )
        
        if effective_max_steps > 100000:
            raise ValidationError(
                "max_steps exceeds maximum allowed value (100000) for performance constraints",
                parameter_name="max_steps",
                invalid_value=effective_max_steps
            )
        
        # Validate goal_radius parameter for non-negative float with mathematical feasibility
        if not isinstance(effective_goal_radius, (int, float)):
            raise ValidationError(
                "goal_radius must be numeric",
                parameter_name="goal_radius", 
                invalid_value=effective_goal_radius
            )
        
        if effective_goal_radius < 0:
            raise ValidationError(
                "goal_radius must be non-negative",
                parameter_name="goal_radius",
                invalid_value=effective_goal_radius
            )
        
        max_grid_dimension = max(width, height)
        if effective_goal_radius > max_grid_dimension:
            raise ValidationError(
                f"goal_radius ({effective_goal_radius}) exceeds maximum grid dimension ({max_grid_dimension})",
                parameter_name="goal_radius",
                invalid_value=effective_goal_radius
            )
        
        # Create base kwargs dictionary with validated parameters and proper parameter names
        base_kwargs = {
            'grid_size': effective_grid_size,
            'source_location': effective_source_location,
            'max_steps': effective_max_steps,
            'goal_radius': effective_goal_radius
        }
        
        # Merge additional_kwargs if provided with conflict detection and resolution
        if additional_kwargs:
            if not isinstance(additional_kwargs, dict):
                raise ValidationError(
                    "additional_kwargs must be a dictionary",
                    parameter_name="additional_kwargs",
                    invalid_value=additional_kwargs
                )
            
            # Check for parameter conflicts
            conflicts = set(base_kwargs.keys()) & set(additional_kwargs.keys())
            if conflicts:
                _logger.warning(f"Parameter conflicts detected, additional_kwargs will override: {conflicts}")
            
            base_kwargs.update(additional_kwargs)

            # Drop testing/internal metadata keys that the environment constructor doesn't accept
            for _k in list(base_kwargs.keys()):
                if isinstance(_k, str) and _k.startswith('_'):
                    del base_kwargs[_k]
        
        # Validate complete kwargs dictionary for parameter consistency and constraint satisfaction
        # Cross-validate source location with goal radius
        if effective_goal_radius > 0:
            min_distance_to_edge = min(
                source_x, source_y,
                width - source_x - 1, height - source_y - 1
            )
            if effective_goal_radius > min_distance_to_edge:
                _logger.warning(f"Goal radius ({effective_goal_radius}) extends beyond grid edges from source location")
        
        # Log kwargs creation with parameter summary and validation status for debugging
        _logger.debug(f"Successfully created registration kwargs: {base_kwargs}")
        
        # Return complete kwargs dictionary ready for Gymnasium registration with comprehensive validation
        return base_kwargs
        
    except Exception as e:
        _logger.error(f"Failed to create registration kwargs: {e}")
        raise


def validate_registration_config(
    env_id: str,
    entry_point: str,
    max_episode_steps: int,
    kwargs: Dict[str, Any],
    strict_validation: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Configuration validation function ensuring parameter consistency, Gymnasium compliance, 
    mathematical feasibility, and performance requirements for robust environment registration.
    
    This function provides comprehensive validation of all registration parameters, ensuring
    compatibility with Gymnasium requirements, mathematical feasibility, and performance constraints.
    
    Args:
        env_id: Environment identifier string to validate
        entry_point: Entry point specification to validate  
        max_episode_steps: Maximum episode steps parameter to validate
        kwargs: Environment parameters dictionary to validate
        strict_validation: Whether to apply enhanced validation rules and constraints
        
    Returns:
        Tuple of (is_valid: bool, validation_report: dict) with detailed configuration analysis and recommendations
        
    Example:
        # Validate registration configuration
        is_valid, report = validate_registration_config(
            env_id="PlumeNav-StaticGaussian-v0",
            entry_point="plume_nav_sim.envs.plume_search_env:PlumeSearchEnv",
            max_episode_steps=1000,
            kwargs={"grid_size": (128, 128)}
        )
        
        if not is_valid:
            print(f"Validation errors: {report['errors']}")
    """
    try:
        # Initialize validation report dictionary with categories for different validation types
        validation_report = {
            'timestamp': time.time(),
            'strict_validation': strict_validation,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'performance_analysis': {},
            'compatibility_check': {}
        }
        
        is_valid = True
        
        # Validate environment ID format following Gymnasium versioning conventions with '-v0' suffix
        if not isinstance(env_id, str) or not env_id.strip():
            validation_report['errors'].append("Environment ID must be a non-empty string")
            is_valid = False
        elif not env_id.endswith('-v0'):
            validation_report['errors'].append("Environment ID must end with '-v0' suffix for Gymnasium versioning compliance")
            is_valid = False
        elif len(env_id) > 100:
            validation_report['warnings'].append("Environment ID is unusually long, consider shorter name")
        
        # Validate entry point specification with module path verification and class accessibility
        if not isinstance(entry_point, str) or not entry_point.strip():
            validation_report['errors'].append("Entry point must be a non-empty string")
            is_valid = False
        elif ':' not in entry_point:
            validation_report['errors'].append("Entry point must contain ':' separator between module and class")
            is_valid = False
        else:
            module_path, class_name = entry_point.rsplit(':', 1)
            if not module_path or not class_name:
                validation_report['errors'].append("Entry point must specify both module path and class name")
                is_valid = False
            elif strict_validation:
                # Try to validate module path format
                if not all(part.isidentifier() or part == '.' for part in module_path.replace('.', ' ').split()):
                    validation_report['warnings'].append("Entry point module path may not be valid Python module")
                if not class_name.isidentifier():
                    validation_report['warnings'].append("Entry point class name may not be valid Python identifier")
        
        # Validate max_episode_steps parameter for positive integer within reasonable training limits
        if not isinstance(max_episode_steps, int):
            validation_report['errors'].append("max_episode_steps must be an integer")
            is_valid = False
        elif max_episode_steps <= 0:
            validation_report['errors'].append("max_episode_steps must be positive")
            is_valid = False
        elif max_episode_steps > 100000:
            validation_report['errors'].append("max_episode_steps exceeds recommended maximum (100000)")
            is_valid = False
        elif max_episode_steps < 100:
            validation_report['warnings'].append("max_episode_steps is quite low, may limit learning")
        
        # Validate kwargs dictionary structure and parameter types with comprehensive type checking
        if not isinstance(kwargs, dict):
            validation_report['errors'].append("kwargs must be a dictionary")
            is_valid = False
        else:
            # Validate specific parameters if present
            if 'grid_size' in kwargs:
                grid_size = kwargs['grid_size']
                if not isinstance(grid_size, (tuple, list)) or len(grid_size) != 2:
                    validation_report['errors'].append("grid_size must be a tuple/list of 2 elements")
                    is_valid = False
                else:
                    width, height = grid_size
                    if not isinstance(width, int) or not isinstance(height, int):
                        validation_report['errors'].append("grid_size dimensions must be integers")
                        is_valid = False
                    elif width <= 0 or height <= 0:
                        validation_report['errors'].append("grid_size dimensions must be positive")
                        is_valid = False
                    elif strict_validation and (width > 1024 or height > 1024):
                        validation_report['warnings'].append("Large grid_size may impact performance")
                    elif width < 16 or height < 16:
                        validation_report['warnings'].append("Small grid_size may limit environment complexity")
            
            if 'source_location' in kwargs:
                source_location = kwargs['source_location']
                if not isinstance(source_location, (tuple, list)) or len(source_location) != 2:
                    validation_report['errors'].append("source_location must be a tuple/list of 2 elements")
                    is_valid = False
                else:
                    source_x, source_y = source_location
                    if not isinstance(source_x, (int, float)) or not isinstance(source_y, (int, float)):
                        validation_report['errors'].append("source_location coordinates must be numeric")
                        is_valid = False
            
            if 'goal_radius' in kwargs:
                goal_radius = kwargs['goal_radius']
                if not isinstance(goal_radius, (int, float)):
                    validation_report['errors'].append("goal_radius must be numeric")
                    is_valid = False
                elif goal_radius < 0:
                    validation_report['errors'].append("goal_radius must be non-negative")
                    is_valid = False
        
        # Cross-validate parameters for mathematical consistency and constraint satisfaction
        if 'grid_size' in kwargs and 'source_location' in kwargs:
            try:
                grid_size = kwargs['grid_size']
                source_location = kwargs['source_location']
                width, height = grid_size
                source_x, source_y = source_location
                
                if source_x < 0 or source_x >= width or source_y < 0 or source_y >= height:
                    validation_report['errors'].append("source_location must be within grid_size bounds")
                    is_valid = False
                    
            except (ValueError, TypeError):
                validation_report['warnings'].append("Could not cross-validate grid_size and source_location")
        
        # Check performance feasibility including memory usage estimation and computational requirements
        if 'grid_size' in kwargs:
            try:
                width, height = kwargs['grid_size']
                grid_cells = width * height
                estimated_memory_mb = (grid_cells * 4) / (1024 * 1024)  # float32 array
                
                validation_report['performance_analysis'] = {
                    'grid_cells': grid_cells,
                    'estimated_memory_mb': round(estimated_memory_mb, 2),
                    'performance_tier': 'high' if grid_cells > 262144 else 'medium' if grid_cells > 16384 else 'low'
                }
                
                if estimated_memory_mb > 100:
                    validation_report['warnings'].append(f"High memory usage estimated: {estimated_memory_mb:.1f}MB")
                    
            except Exception:
                validation_report['warnings'].append("Could not estimate performance characteristics")
        
        # Apply strict validation rules if strict_validation enabled with enhanced precision checking
        if strict_validation:
            # Check for potential naming conflicts
            if env_id.lower().startswith('gym'):
                validation_report['warnings'].append("Environment ID starting with 'gym' may conflict with official environments")
            
            # Validate parameter completeness
            expected_params = {'grid_size', 'source_location', 'goal_radius'}
            missing_params = expected_params - set(kwargs.keys())
            if missing_params:
                validation_report['recommendations'].append(f"Consider specifying parameters: {missing_params}")
            
            # Check for unusual parameter combinations
            if 'goal_radius' in kwargs and kwargs['goal_radius'] > 0:
                if 'source_location' in kwargs and 'grid_size' in kwargs:
                    try:
                        width, height = kwargs['grid_size']
                        source_x, source_y = kwargs['source_location']
                        goal_radius = kwargs['goal_radius']
                        
                        min_distance_to_edge = min(source_x, source_y, width - source_x - 1, height - source_y - 1)
                        if goal_radius > min_distance_to_edge:
                            validation_report['warnings'].append("Goal radius extends beyond grid boundaries")
                    except Exception:
                        pass
        
        # Test environment instantiation possibility without actual registration for validation
        validation_report['compatibility_check'] = {
            'gymnasium_available': True,
            'entry_point_format_valid': ':' in entry_point,
            'parameter_types_valid': is_valid
        }
        
        # Compile comprehensive validation report with findings, warnings, and optimization recommendations
        if not validation_report['errors']:
            validation_report['recommendations'].append("Configuration appears valid for registration")
        
        if validation_report['warnings']:
            validation_report['recommendations'].append("Review warnings for potential improvements")
        
        # Return validation status tuple with detailed analysis for configuration improvement and error resolution
        return is_valid, validation_report
        
    except Exception as e:
        _logger.error(f"Configuration validation failed: {e}")
        return False, {
            'timestamp': time.time(),
            'errors': [f"Validation process failed: {str(e)}"],
            'warnings': [],
            'recommendations': ['Check validation parameters and try again']
        }


def register_with_custom_params(
    grid_size: Optional[Tuple[int, int]] = None,
    source_location: Optional[Tuple[int, int]] = None,
    max_steps: Optional[int] = None,
    goal_radius: Optional[float] = None,
    custom_env_id: Optional[str] = None,
    force_reregister: bool = False
) -> str:
    """
    Convenience registration function with custom parameter overrides providing streamlined 
    environment registration with validation, error handling, and immediate availability for 
    specialized research configurations.
    
    This function provides a convenient interface for registering environments with custom
    parameters, handling all validation and configuration automatically while supporting
    specialized research requirements.
    
    Args:
        grid_size: Custom grid dimensions, defaults to DEFAULT_GRID_SIZE if not provided
        source_location: Custom source location, defaults to DEFAULT_SOURCE_LOCATION if not provided
        max_steps: Custom maximum steps, defaults to DEFAULT_MAX_STEPS if not provided
        goal_radius: Custom goal radius, defaults to DEFAULT_GOAL_RADIUS if not provided
        custom_env_id: Custom environment identifier, defaults to ENV_ID if not provided
        force_reregister: Whether to force re-registration if environment already exists
        
    Returns:
        Registered environment ID ready for immediate use with gym.make() calls
        
    Raises:
        ValidationError: If custom parameters fail validation
        ConfigurationError: If registration fails due to configuration issues
        
    Example:
        # Register with larger grid
        env_id = register_with_custom_params(
            grid_size=(256, 256),
            source_location=(128, 128)
        )
        env = gym.make(env_id)
        
        # Register completely custom environment
        env_id = register_with_custom_params(
            grid_size=(64, 64),
            source_location=(32, 32), 
            goal_radius=3.0,
            custom_env_id="SmallPlume-v0"
        )
    """
    try:
        # Generate custom environment ID if custom_env_id provided with version suffix validation
        if custom_env_id:
            if not custom_env_id.endswith('-v0'):
                custom_env_id = custom_env_id + '-v0'
                _logger.info(f"Added version suffix to custom environment ID: {custom_env_id}")
            effective_env_id = custom_env_id
        else:
            effective_env_id = ENV_ID
        
        _logger.info(f"Registering environment with custom parameters: {effective_env_id}")
        
        # Create complete kwargs using create_registration_kwargs() with provided parameters
        registration_kwargs = create_registration_kwargs(
            grid_size=grid_size,
            source_location=source_location,
            max_steps=max_steps,
            goal_radius=goal_radius
        )
        
        # Apply custom environment ID or use default ENV_ID for registration
        # Call register_env() with custom parameters and force_reregister flag
        registered_env_id = register_env(
            env_id=effective_env_id,
            entry_point=ENTRY_POINT,
            max_episode_steps=max_steps or MAX_EPISODE_STEPS,
            kwargs=registration_kwargs,
            force_reregister=force_reregister
        )
        
        # Validate successful registration with immediate gym.make() test
        try:
            test_env = gymnasium.make(registered_env_id)
            test_env.close()
            _logger.debug(f"Custom registration verified for '{registered_env_id}'")
        except Exception as test_error:
            _logger.error(f"Custom registration verification failed: {test_error}")
            raise ConfigurationError(
                f"Custom registration verification failed: {test_error}",
                config_parameter="custom_registration"
            ) from test_error
        
        # Log custom registration with parameter overrides and configuration summary
        param_summary = {
            'grid_size': grid_size or DEFAULT_GRID_SIZE,
            'source_location': source_location or DEFAULT_SOURCE_LOCATION,
            'max_steps': max_steps or DEFAULT_MAX_STEPS,
            'goal_radius': goal_radius if goal_radius is not None else DEFAULT_GOAL_RADIUS
        }
        _logger.info(f"Successfully registered custom environment '{registered_env_id}' with parameters: {param_summary}")
        
        # Return registered environment ID for immediate use with success confirmation
        return registered_env_id
        
    except Exception as e:
        _logger.error(f"Custom registration failed: {e}")
        raise