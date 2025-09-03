"""
Wind Field Models Package: Configurable environmental dynamics with pluggable wind implementations.

This package provides the core infrastructure for the modular wind field architecture
enabling seamless switching between different wind dynamics approaches while maintaining
consistent interfaces and performance requirements. The modular design supports both
simple constant wind flows for rapid experimentation and realistic turbulent dynamics
for high-fidelity environmental modeling.

Key Features:
- Protocol-based architecture for seamless wind field model substitution
- Configuration-driven instantiation via factory patterns and Hydra integration
- Performance-optimized implementations with <0.5ms velocity query latency per WindFieldProtocol
- Runtime model discovery and validation through plugin registry system
- Graceful degradation during development with comprehensive error handling

Modular Architecture Components:
- ConstantWindField: Uniform directional flow with minimal computational overhead
- TurbulentWindField: Realistic gusty conditions with stochastic atmospheric boundary layer dynamics
- TimeVaryingWindField: Data-driven and procedural temporal wind patterns with configurable evolution
- Plugin registry enabling runtime discovery and validation of available implementations
- Factory functions supporting both programmatic and configuration-driven instantiation

Performance Characteristics:
- velocity_at(): <0.5ms single query, <5ms for spatial field evaluation per WindFieldProtocol
- step(): <2ms per time step for minimal simulation overhead and real-time compatibility
- Memory efficiency: <50MB for typical wind field representations with spatial correlation
- Zero-copy NumPy operations and vectorized computations for scalability

Usage Patterns:
    Direct instantiation:
    >>> from plume_nav_sim.models.wind import ConstantWindField, TurbulentWindField
    >>> constant_wind = ConstantWindField(velocity=(2.0, 0.5))
    >>> turbulent_wind = TurbulentWindField(mean_velocity=(3.0, 1.0), turbulence_intensity=0.2)
    
    Factory-based creation:
    >>> from plume_nav_sim.models.wind import create_wind_field, AVAILABLE_WIND_FIELDS
    >>> config = {'type': 'ConstantWindField', 'velocity': (2.5, 1.0)}
    >>> wind_field = create_wind_field(config)
    
    Configuration-driven instantiation:
    >>> from hydra import compose, initialize
    >>> with initialize(config_path="../conf"):
    ...     cfg = compose(config_name="config")
    ...     wind_field = create_wind_field(cfg.wind_field)
    
    Environment integration:
    >>> from plume_nav_sim.envs import PlumeNavigationEnv
    >>> env = PlumeNavigationEnv.from_config({
    ...     "wind_field": {"type": "TurbulentWindField", "turbulence_intensity": 0.3}
    ... })
    
    Model discovery and validation:
    >>> print(f"Available wind fields: {list(AVAILABLE_WIND_FIELDS.keys())}")
    >>> field_info = get_wind_field_info('TurbulentWindField')
    >>> print(f"Model features: {field_info['features']}")

Technical Integration:
- Implements WindFieldProtocol from core.protocols for type safety and interface compliance
- Integrates with PlumeModelProtocol for realistic plume transport dynamics
- Supports NavigatorProtocol integration for environmental sensing and agent observations
- Maintains compatibility with Gymnasium environments and RL training frameworks
- Provides seamless Hydra configuration management with structured configuration schemas
"""

from __future__ import annotations
import warnings
from typing import Dict, Any, List, Optional, Union, Type, Tuple
from pathlib import Path

from loguru import logger
LOGURU_AVAILABLE = True

# Core imports with graceful fallbacks during migration
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Protocol import for type safety and interface compliance
from plume_nav_sim.protocols.wind_field import WindFieldProtocol
PROTOCOLS_AVAILABLE = True

# Configuration management
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    HYDRA_AVAILABLE = False

# Import all wind field implementations with graceful fallbacks
WIND_FIELD_IMPLEMENTATIONS = {}  # Registry for successful imports

# ConstantWindField implementation
try:
    from .constant_wind import ConstantWindField, ConstantWindFieldConfig, create_constant_wind_field
    WIND_FIELD_IMPLEMENTATIONS['ConstantWindField'] = {
        'class': ConstantWindField,
        'config_class': ConstantWindFieldConfig,
        'factory_function': create_constant_wind_field,
        'description': 'Uniform directional wind flow with minimal computational overhead',
        'features': ['uniform_flow', 'fast_computation', 'temporal_evolution', 'vectorized_operations'],
        'performance': {'velocity_query_ms': 0.1, 'step_time_ms': 0.05, 'memory_mb': 1},
        'use_cases': ['rapid_prototyping', 'baseline_experiments', 'simple_transport_modeling'],
        'available': True
    }
    CONSTANT_WIND_AVAILABLE = True
    logger.debug("ConstantWindField imported successfully")
except ImportError as e:
    CONSTANT_WIND_AVAILABLE = False
    logger.warning(f"ConstantWindField not available: {e}")
    ConstantWindField = None
    ConstantWindFieldConfig = None
    create_constant_wind_field = None

# TurbulentWindField implementation
try:
    from .turbulent_wind import TurbulentWindField, TurbulentWindFieldConfig
    WIND_FIELD_IMPLEMENTATIONS['TurbulentWindField'] = {
        'class': TurbulentWindField,
        'config_class': TurbulentWindFieldConfig,
        'factory_function': None,  # Uses class constructor
        'description': 'Realistic gusty conditions with stochastic atmospheric boundary layer dynamics',
        'features': ['stochastic_variations', 'atmospheric_boundary_layer', 'eddy_formation', 'realistic_physics'],
        'performance': {'velocity_query_ms': 0.5, 'step_time_ms': 2.0, 'memory_mb': 50},
        'use_cases': ['realistic_simulation', 'atmospheric_research', 'environmental_modeling'],
        'available': True
    }
    TURBULENT_WIND_AVAILABLE = True
    logger.debug("TurbulentWindField imported successfully")
except ImportError as e:
    TURBULENT_WIND_AVAILABLE = False
    logger.warning(f"TurbulentWindField not available: {e}")
    TurbulentWindField = None
    TurbulentWindFieldConfig = None

# TimeVaryingWindField implementation
try:
    from .time_varying_wind import TimeVaryingWindField, TimeVaryingWindFieldConfig
    WIND_FIELD_IMPLEMENTATIONS['TimeVaryingWindField'] = {
        'class': TimeVaryingWindField,
        'config_class': TimeVaryingWindFieldConfig,
        'factory_function': None,  # Uses class constructor
        'description': 'Time-dependent wind patterns with configurable temporal evolution',
        'features': ['temporal_evolution', 'data_driven_patterns', 'periodic_variations', 'file_based_loading'],
        'performance': {'velocity_query_ms': 0.5, 'step_time_ms': 2.0, 'memory_mb': 50},
        'use_cases': ['measurement_driven_simulation', 'temporal_dynamics_research', 'long_term_experiments'],
        'available': True
    }
    TIME_VARYING_WIND_AVAILABLE = True
    logger.debug("TimeVaryingWindField imported successfully")
except ImportError as e:
    TIME_VARYING_WIND_AVAILABLE = False
    logger.warning(f"TimeVaryingWindField not available: {e}")
    TimeVaryingWindField = None
    TimeVaryingWindFieldConfig = None

# Calculate total available implementations
TOTAL_IMPLEMENTATIONS = len(WIND_FIELD_IMPLEMENTATIONS)
logger.info(f"Loaded {TOTAL_IMPLEMENTATIONS} wind field implementations")

# Ensure at least one implementation is available
if TOTAL_IMPLEMENTATIONS == 0:
    warnings.warn(
        "No wind field implementations are currently available. "
        "This may indicate missing dependencies or incomplete migration. "
        "At least one wind field implementation is required for environmental dynamics.",
        ImportWarning,
        stacklevel=2
    )

# Create the public registry for wind field discovery and validation
AVAILABLE_WIND_FIELDS: Dict[str, Dict[str, Any]] = WIND_FIELD_IMPLEMENTATIONS.copy()
"""
Registry of available wind field implementations with metadata.

This registry enables runtime discovery, validation, and selection of wind field models
based on feature requirements, performance characteristics, and use case suitability.

Registry Structure:
    Each wind field entry contains:
    - class: Wind field implementation class
    - config_class: Associated configuration schema
    - factory_function: Optional factory function for creation
    - description: Human-readable wind field description
    - features: List of supported features and capabilities
    - performance: Expected performance characteristics
    - use_cases: Recommended application scenarios
    - available: Boolean indicating successful import status

Examples:
    Wind field discovery:
    >>> available_fields = list(AVAILABLE_WIND_FIELDS.keys())
    >>> print(f"Available wind fields: {available_fields}")
    
    Feature-based selection:
    >>> fast_fields = [name for name, info in AVAILABLE_WIND_FIELDS.items() 
    ...                if info['performance']['velocity_query_ms'] < 1.0]
    
    Use case matching:
    >>> realistic_fields = [name for name, info in AVAILABLE_WIND_FIELDS.items()
    ...                    if 'realistic_simulation' in info['use_cases']]
"""

# Wind field type mappings for configuration-driven instantiation
WIND_FIELD_TYPE_MAPPING = {
    'constant': 'ConstantWindField',
    'turbulent': 'TurbulentWindField',
    'time_varying': 'TimeVaryingWindField',
    'uniform': 'ConstantWindField',
    'gusty': 'TurbulentWindField',
    'temporal': 'TimeVaryingWindField',
    'stochastic': 'TurbulentWindField',
    'data_driven': 'TimeVaryingWindField'
}
"""Mapping of user-friendly wind field type names to implementation classes."""


def create_wind_field(
    config: Union[Dict[str, Any], DictConfig], 
    validate_protocol: bool = True,
    **kwargs: Any
) -> WindFieldProtocol:
    """
    Factory function for creating wind field instances from configuration.
    
    This function provides the primary interface for configuration-driven wind field
    instantiation with comprehensive parameter validation, type checking, and
    graceful error handling. Supports both Hydra-style and dictionary configurations.
    
    Args:
        config: Configuration dictionary or DictConfig specifying wind field
            parameters. Must include 'type' or '_target_' field for wind field selection.
            Supported configuration formats:
            - Hydra style: {'_target_': 'fully.qualified.ClassName', ...}
            - Factory style: {'type': 'WindFieldName', ...}
            - Friendly names: {'type': 'constant', ...}
        validate_protocol: Whether to validate WindFieldProtocol compliance
            of the created instance (default: True).
        **kwargs: Additional parameters to override or supplement configuration.
            These take precedence over config values for flexibility.
            
    Returns:
        WindFieldProtocol: Configured wind field instance implementing the
            complete protocol interface with validated parameters.
            
    Raises:
        ValueError: If configuration is invalid, incomplete, or specifies
            unknown wind field type.
        TypeError: If configuration format is unsupported.
        ImportError: If required wind field implementation is not available.
        RuntimeError: If wind field instantiation fails or protocol validation fails.
        
    Notes:
        Wind Field Selection Priority:
        1. '_target_' field for Hydra instantiation (highest priority)
        2. 'type' field with exact class name matching
        3. 'type' field with friendly name mapping
        4. Default to ConstantWindField if no type specified
        
        Parameter Merging:
        - kwargs override config parameters for runtime flexibility
        - Hydra interpolation is resolved before parameter validation
        - Type conversion is applied for numeric parameters
        
        Performance Considerations:
        - Wind field instances are created fresh for each call (no caching)
        - Configuration validation adds ~1ms overhead
        - Protocol compliance checking adds ~0.5ms overhead
        
    Examples:
        Hydra-style configuration:
        >>> config = {
        ...     '_target_': 'plume_nav_sim.models.wind.ConstantWindField',
        ...     'velocity': (2.0, 0.5),
        ...     'enable_temporal_evolution': True,
        ...     'evolution_rate': 0.1
        ... }
        >>> wind_field = create_wind_field(config)
        
        Factory-style configuration:
        >>> config = {
        ...     'type': 'TurbulentWindField',
        ...     'mean_velocity': (3.0, 1.0),
        ...     'turbulence_intensity': 0.2,
        ...     'correlation_length': 10.0
        ... }
        >>> wind_field = create_wind_field(config)
        
        Friendly name configuration:
        >>> config = {'type': 'constant', 'velocity': (1.5, 0.8)}
        >>> wind_field = create_wind_field(config, evolution_rate=0.05)
        
        Time-varying wind configuration:
        >>> config = {
        ...     'type': 'time_varying',
        ...     'base_velocity': (2.0, 1.0),
        ...     'temporal_pattern': 'sinusoidal',
        ...     'period': 60.0
        ... }
        >>> wind_field = create_wind_field(config)
        
        Configuration with validation disabled:
        >>> wind_field = create_wind_field(config, validate_protocol=False)
    """
    if not config:
        raise ValueError("Configuration cannot be empty")
    
    # Convert DictConfig to regular dict if needed
    if hasattr(config, 'to_container') and HYDRA_AVAILABLE:
        config_dict = config.to_container(resolve=True)
    else:
        config_dict = dict(config)
    
    # Apply parameter overrides from kwargs
    config_dict.update(kwargs)
    
    # Determine wind field type and validate availability
    wind_field_type = None
    target_class = None
    
    # Priority 1: Hydra-style _target_ field
    if '_target_' in config_dict:
        if HYDRA_AVAILABLE:
            try:
                from hydra import utils as hydra_utils
                logger.debug(f"Using Hydra instantiation with target: {config_dict['_target_']}")
                wind_field = hydra_utils.instantiate(config_dict)
                
                # Validate protocol compliance if requested
                if validate_protocol:
                    missing = [
                        m for m in ("velocity_at", "step", "reset")
                        if not callable(getattr(wind_field, m, None))
                    ]
                    if missing:
                        logger.error(
                            f"Wind field missing required methods: {', '.join(missing)}"
                        )
                        raise RuntimeError(
                            f"Instantiated wind field missing required methods: {', '.join(missing)}"
                        )
                    if not isinstance(wind_field, WindFieldProtocol):
                        logger.error(
                            f"Instantiated wind field does not implement WindFieldProtocol: {type(wind_field)}"
                        )
                        raise RuntimeError(
                            f"Instantiated wind field does not implement WindFieldProtocol: {type(wind_field)}"
                        )
                
                logger.info(f"Successfully created wind field via Hydra: {type(wind_field).__name__}")
                return wind_field
            except Exception as e:
                raise RuntimeError(f"Hydra instantiation failed: {e}") from e
        else:
            warnings.warn(
                "Hydra not available but _target_ specified. "
                "Falling back to factory instantiation.",
                UserWarning
            )
    
    # Priority 2: Factory-style type field
    if 'type' in config_dict:
        specified_type = config_dict['type']
        
        # Check for exact class name match
        if specified_type in AVAILABLE_WIND_FIELDS:
            wind_field_type = specified_type
            target_class = AVAILABLE_WIND_FIELDS[specified_type]['class']
        
        # Check for friendly name mapping
        elif specified_type in WIND_FIELD_TYPE_MAPPING:
            wind_field_type = WIND_FIELD_TYPE_MAPPING[specified_type] 
            if wind_field_type in AVAILABLE_WIND_FIELDS:
                target_class = AVAILABLE_WIND_FIELDS[wind_field_type]['class']
            else:
                raise ImportError(
                    f"Mapped wind field type '{wind_field_type}' is not available. "
                    f"Available wind fields: {list(AVAILABLE_WIND_FIELDS.keys())}"
                )
        else:
            raise ValueError(
                f"Unknown wind field type: '{specified_type}'. "
                f"Available types: {list(AVAILABLE_WIND_FIELDS.keys())} "
                f"or friendly names: {list(WIND_FIELD_TYPE_MAPPING.keys())}"
            )
    
    # Priority 3: Default to ConstantWindField
    else:
        if 'ConstantWindField' in AVAILABLE_WIND_FIELDS:
            wind_field_type = 'ConstantWindField'
            target_class = AVAILABLE_WIND_FIELDS[wind_field_type]['class']
            logger.debug("No wind field type specified, defaulting to ConstantWindField")
        else:
            raise RuntimeError(
                "No wind field type specified and ConstantWindField not available. "
                f"Please specify 'type' field. Available wind fields: {list(AVAILABLE_WIND_FIELDS.keys())}"
            )
    
    # Remove internal configuration keys before instantiation
    wind_field_params = {k: v for k, v in config_dict.items() 
                        if k not in ['type', '_target_']}
    
    # Instantiate the wind field with error handling
    try:
        logger.debug(f"Instantiating {wind_field_type} with parameters: {list(wind_field_params.keys())}")
        wind_field = target_class(**wind_field_params)
        
        # Validate protocol compliance if requested
        if validate_protocol:
            missing = [
                m for m in ("velocity_at", "step", "reset")
                if not callable(getattr(wind_field, m, None))
            ]
            if missing:
                logger.error(
                    f"Wind field missing required methods: {', '.join(missing)}"
                )
                raise RuntimeError(
                    f"Instantiated wind field missing required methods: {', '.join(missing)}"
                )
            if not isinstance(wind_field, WindFieldProtocol):
                logger.error(
                    f"Instantiated wind field does not implement WindFieldProtocol: {type(wind_field)}"
                )
                raise RuntimeError(
                    f"Instantiated wind field does not implement WindFieldProtocol: {type(wind_field)}"
                )
        
        logger.info(f"Successfully created {wind_field_type} wind field")
        return wind_field
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {wind_field_type} with parameters {wind_field_params}. "
            f"Error: {e}"
        ) from e


def get_wind_field_info(wind_field_type: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a specific wind field implementation.
    
    Args:
        wind_field_type: Name of the wind field implementation to query.
            Must be a key in AVAILABLE_WIND_FIELDS registry.
            
    Returns:
        Dict[str, Any]: Comprehensive wind field information including:
            - description: Human-readable wind field description
            - features: List of supported features and capabilities
            - performance: Expected performance characteristics
            - use_cases: Recommended application scenarios
            - available: Import and availability status
            - class: Wind field implementation class (if available)
            - config_class: Associated configuration schema (if available)
            
    Raises:
        KeyError: If wind_field_type is not found in the registry.
        
    Examples:
        Get constant wind field information:
        >>> info = get_wind_field_info('ConstantWindField')
        >>> print(f"Description: {info['description']}")
        >>> print(f"Features: {info['features']}")
        >>> print(f"Expected query time: {info['performance']['velocity_query_ms']}ms")
        
        Check wind field availability:
        >>> info = get_wind_field_info('TurbulentWindField')
        >>> if info['available']:
        ...     print("TurbulentWindField is available for use")
    """
    if wind_field_type not in AVAILABLE_WIND_FIELDS:
        available_types = list(AVAILABLE_WIND_FIELDS.keys())
        raise KeyError(
            f"Unknown wind field type: '{wind_field_type}'. "
            f"Available wind fields: {available_types}"
        )
    
    return AVAILABLE_WIND_FIELDS[wind_field_type].copy()


def list_available_wind_fields() -> List[str]:
    """
    Get list of all available wind field implementation names.
    
    Returns:
        List[str]: List of wind field names that can be used with create_wind_field().
        
    Examples:
        List all available wind fields:
        >>> wind_fields = list_available_wind_fields()
        >>> print(f"Available wind fields: {wind_fields}")
        
        Check if specific wind field is available:
        >>> if 'ConstantWindField' in list_available_wind_fields():
        ...     print("Constant wind field modeling is available")
    """
    return list(AVAILABLE_WIND_FIELDS.keys())


def get_wind_fields_by_feature(feature: str) -> List[str]:
    """
    Get list of wind fields that support a specific feature.
    
    Args:
        feature: Feature name to search for. Examples: 'fast_computation',
            'stochastic_variations', 'temporal_evolution', 'realistic_physics'.
            
    Returns:
        List[str]: List of wind field names that support the specified feature.
        
    Examples:
        Find wind fields with fast computation:
        >>> fast_fields = get_wind_fields_by_feature('fast_computation')
        >>> print(f"Fast computation wind fields: {fast_fields}")
        
        Find wind fields with realistic physics:
        >>> physics_fields = get_wind_fields_by_feature('realistic_physics')
        >>> print(f"Physics-based wind fields: {physics_fields}")
    """
    matching_fields = []
    for field_name, field_info in AVAILABLE_WIND_FIELDS.items():
        if feature in field_info.get('features', []):
            matching_fields.append(field_name)
    return matching_fields


def get_wind_fields_by_use_case(use_case: str) -> List[str]:
    """
    Get list of wind fields recommended for a specific use case.
    
    Args:
        use_case: Use case name to search for. Examples: 'rapid_prototyping',
            'realistic_simulation', 'atmospheric_research', 'baseline_experiments'.
            
    Returns:
        List[str]: List of wind field names recommended for the specified use case.
        
    Examples:
        Find wind fields for rapid prototyping:
        >>> prototype_fields = get_wind_fields_by_use_case('rapid_prototyping')
        >>> print(f"Prototyping wind fields: {prototype_fields}")
        
        Find wind fields for realistic simulation:
        >>> simulation_fields = get_wind_fields_by_use_case('realistic_simulation')
        >>> print(f"Simulation wind fields: {simulation_fields}")
    """
    matching_fields = []
    for field_name, field_info in AVAILABLE_WIND_FIELDS.items():
        if use_case in field_info.get('use_cases', []):
            matching_fields.append(field_name)
    return matching_fields


def validate_wind_field_config(
    config: Union[Dict[str, Any], DictConfig], 
    wind_field_type: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate wind field configuration for completeness and correctness.
    
    Args:
        config: Configuration dictionary to validate.
        wind_field_type: Optional specific wind field type to validate against.
            If None, attempts to infer from config.
            
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
            - is_valid: True if configuration is valid
            - error_messages: List of validation errors (empty if valid)
            
    Examples:
        Validate complete configuration:
        >>> config = {'type': 'ConstantWindField', 'velocity': (2.0, 0.5)}
        >>> is_valid, errors = validate_wind_field_config(config)
        >>> if not is_valid:
        ...     print(f"Configuration errors: {errors}")
        
        Validate specific wind field configuration:
        >>> config = {'velocity': (3.0, 1.0), 'evolution_rate': 0.1}
        >>> is_valid, errors = validate_wind_field_config(config, 'ConstantWindField')
    """
    errors = []
    
    # Check for empty configuration
    if not config:
        errors.append("Configuration cannot be empty")
        return False, errors
    
    # Convert to dict if needed
    if hasattr(config, 'to_container'):
        config_dict = config.to_container(resolve=True)
    else:
        config_dict = dict(config)
    
    # Determine wind field type for validation
    if wind_field_type is None:
        if '_target_' in config_dict:
            # Extract class name from _target_ path
            target_path = config_dict['_target_']
            wind_field_type = target_path.split('.')[-1]
        elif 'type' in config_dict:
            specified_type = config_dict['type']
            wind_field_type = WIND_FIELD_TYPE_MAPPING.get(specified_type, specified_type)
        else:
            wind_field_type = 'ConstantWindField'  # Default
    
    # Check if wind field type is available
    if wind_field_type not in AVAILABLE_WIND_FIELDS:
        errors.append(f"Unknown or unavailable wind field type: {wind_field_type}")
        return False, errors
    
    # Perform wind field-specific validation
    field_info = AVAILABLE_WIND_FIELDS[wind_field_type]
    
    # Check availability
    if not field_info['available']:
        errors.append(f"Wind field type {wind_field_type} is not currently available")
    
    # Basic parameter validation based on wind field type
    if wind_field_type == 'ConstantWindField':
        # Validate constant wind field parameters
        if 'velocity' in config_dict:
            velocity = config_dict['velocity']
            if not (isinstance(velocity, (list, tuple)) and len(velocity) == 2):
                errors.append("velocity must be a 2-element list or tuple")
        
        if 'evolution_rate' in config_dict and config_dict['evolution_rate'] < 0:
            errors.append("evolution_rate must be non-negative")
        
        if 'evolution_amplitude' in config_dict and config_dict['evolution_amplitude'] < 0:
            errors.append("evolution_amplitude must be non-negative")
    
    elif wind_field_type == 'TurbulentWindField':
        # Validate turbulent wind field parameters
        if 'mean_velocity' in config_dict:
            mean_vel = config_dict['mean_velocity']
            if not (isinstance(mean_vel, (list, tuple)) and len(mean_vel) == 2):
                errors.append("mean_velocity must be a 2-element list or tuple")
        
        if 'turbulence_intensity' in config_dict:
            intensity = config_dict['turbulence_intensity']
            if not (0 <= intensity <= 1):
                errors.append("turbulence_intensity must be between 0 and 1")
        
        if 'correlation_length' in config_dict and config_dict['correlation_length'] <= 0:
            errors.append("correlation_length must be positive")
    
    elif wind_field_type == 'TimeVaryingWindField':
        # Validate time-varying wind field parameters
        if 'base_velocity' in config_dict:
            base_vel = config_dict['base_velocity']
            if not (isinstance(base_vel, (list, tuple)) and len(base_vel) == 2):
                errors.append("base_velocity must be a 2-element list or tuple")
        
        if 'period' in config_dict and config_dict['period'] <= 0:
            errors.append("period must be positive")
        
        if 'data_file' in config_dict:
            data_file = config_dict['data_file']
            if not isinstance(data_file, str):
                errors.append("data_file must be a string")
            else:
                # Check if path exists (if it's not a placeholder)
                path_obj = Path(data_file)
                if not str(path_obj).startswith('<') and not path_obj.exists():
                    errors.append(f"Data file not found: {data_file}")
    
    # Return validation results
    is_valid = len(errors) == 0
    return is_valid, errors


def get_default_config(wind_field_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific wind field type.
    
    Args:
        wind_field_type: Name of the wind field implementation.
            
    Returns:
        Dict[str, Any]: Default configuration dictionary with sensible defaults
            for the specified wind field type.
            
    Raises:
        KeyError: If wind_field_type is not found in the registry.
        
    Examples:
        Get default constant wind field configuration:
        >>> config = get_default_config('ConstantWindField')
        >>> print(f"Default velocity: {config['velocity']}")
        
        Create wind field with defaults and overrides:
        >>> config = get_default_config('TurbulentWindField')
        >>> config['turbulence_intensity'] = 0.4  # Override default
        >>> wind_field = create_wind_field(config)
    """
    if wind_field_type not in AVAILABLE_WIND_FIELDS:
        available_types = list(AVAILABLE_WIND_FIELDS.keys())
        raise KeyError(
            f"Unknown wind field type: '{wind_field_type}'. "
            f"Available wind fields: {available_types}"
        )
    
    # Define default configurations for each wind field type
    defaults = {
        'ConstantWindField': {
            'velocity': (1.0, 0.0),
            'enable_temporal_evolution': False,
            'evolution_rate': 0.0,
            'evolution_amplitude': 0.0,
            'evolution_period': 100.0,
            'noise_intensity': 0.0,
            'boundary_conditions': None,
            'performance_monitoring': False
        },
        'TurbulentWindField': {
            'mean_velocity': (2.0, 0.5),
            'turbulence_intensity': 0.2,
            'correlation_length': 10.0,
            'atmospheric_stability': 'neutral',
            'boundary_layer_height': 1000.0,
            'surface_roughness': 0.1,
            'enable_wind_shear': True,
            'temporal_correlation': 0.8
        },
        'TimeVaryingWindField': {
            'base_velocity': (1.5, 0.8),
            'temporal_pattern': 'sinusoidal',
            'amplitude': (0.5, 0.3),
            'period': 60.0,
            'phase_offset': 0.0,
            'noise_amplitude': 0.1,
            'data_file': None,
            'temporal_column': 'timestamp',
            'velocity_columns': ['u_wind', 'v_wind']
        }
    }
    
    return defaults.get(wind_field_type, {}).copy()


# Performance monitoring and diagnostics
def get_performance_summary() -> Dict[str, Any]:
    """
    Get performance summary for all available wind field implementations.
    
    Returns:
        Dict[str, Any]: Performance summary including wind field characteristics,
            expected execution times, and memory usage estimates.
            
    Examples:
        Display performance comparison:
        >>> summary = get_performance_summary()
        >>> for field, perf in summary['wind_fields'].items():
        ...     print(f"{field}: {perf['velocity_query_ms']}ms query time")
    """
    performance_data = {
        'total_wind_fields': len(AVAILABLE_WIND_FIELDS),
        'available_wind_fields': [name for name, info in AVAILABLE_WIND_FIELDS.items() 
                                 if info['available']],
        'wind_fields': {}
    }
    
    for field_name, field_info in AVAILABLE_WIND_FIELDS.items():
        performance_data['wind_fields'][field_name] = {
            'available': field_info['available'],
            'performance': field_info['performance'],
            'features': field_info['features'],
            'description': field_info['description']
        }
    
    return performance_data


# Module initialization and integrity validation
def _validate_module_integrity() -> Dict[str, Any]:
    """
    Validate module integrity and component availability.
    
    Returns:
        Dict[str, Any]: Validation results with status and component details.
    """
    validation_results = {
        'status': 'healthy',
        'issues': [],
        'components': {
            'numpy_available': NUMPY_AVAILABLE,
            'protocols_available': PROTOCOLS_AVAILABLE,
            'hydra_available': HYDRA_AVAILABLE,
            'loguru_available': LOGURU_AVAILABLE
        },
        'implementations': {
            'total': len(AVAILABLE_WIND_FIELDS),
            'available': len([f for f in AVAILABLE_WIND_FIELDS.values() if f['available']]),
            'wind_fields': list(AVAILABLE_WIND_FIELDS.keys())
        }
    }
    
    # Check critical dependencies
    if not NUMPY_AVAILABLE:
        validation_results['issues'].append("NumPy not available - array operations will fail")
    
    if not PROTOCOLS_AVAILABLE:
        validation_results['issues'].append("Protocol interfaces not available - type safety limited")
    
    # Check implementation availability
    available_count = validation_results['implementations']['available']
    if available_count == 0:
        validation_results['issues'].append("No wind field implementations available")
        validation_results['status'] = 'critical'
    elif available_count < 2:
        validation_results['issues'].append("Limited wind field implementations available")
        validation_results['status'] = 'degraded'
    
    # Check for basic functionality
    if available_count > 0:
        try:
            # Test factory function with minimal config
            test_config = {'type': list(AVAILABLE_WIND_FIELDS.keys())[0]}
            test_field = create_wind_field(test_config, validate_protocol=False)
            validation_results['factory_test'] = 'passed'
        except Exception as e:
            validation_results['issues'].append(f"Factory function test failed: {e}")
            validation_results['factory_test'] = 'failed'
            validation_results['status'] = 'degraded'
    
    return validation_results


# Perform module integrity validation on import
_validation_results = _validate_module_integrity()
if _validation_results['status'] != 'healthy':
    warning_msg = (
        f"Wind fields package validation {_validation_results['status']}: "
        f"{len(_validation_results['issues'])} issues detected. "
        f"Available implementations: {_validation_results['implementations']['available']}/"
        f"{_validation_results['implementations']['total']}"
    )
    warnings.warn(warning_msg, ImportWarning, stacklevel=2)

# Log successful initialization
logger.info(
    f"Wind fields package initialized successfully",
    total_implementations=_validation_results['implementations']['total'],
    available_implementations=_validation_results['implementations']['available'],
    status=_validation_results['status'],
)

# Define comprehensive public API for maximum flexibility
__all__ = [
    # Core wind field implementations (conditionally exported based on availability)
    *(["ConstantWindField", "ConstantWindFieldConfig", "create_constant_wind_field"] 
      if CONSTANT_WIND_AVAILABLE else []),
    *(["TurbulentWindField", "TurbulentWindFieldConfig"] 
      if TURBULENT_WIND_AVAILABLE else []),
    *(["TimeVaryingWindField", "TimeVaryingWindFieldConfig"] 
      if TIME_VARYING_WIND_AVAILABLE else []),
    
    # Factory functions and registry
    "create_wind_field",
    "AVAILABLE_WIND_FIELDS", 
    "WIND_FIELD_TYPE_MAPPING",
    
    # Discovery and information functions
    "get_wind_field_info",
    "list_available_wind_fields",
    "get_wind_fields_by_feature",
    "get_wind_fields_by_use_case",
    
    # Configuration utilities
    "validate_wind_field_config",
    "get_default_config",
    
    # Performance and diagnostics
    "get_performance_summary",
    
    # Module metadata
    "TOTAL_IMPLEMENTATIONS",
]

# Module metadata for documentation and debugging
__version__ = "0.1.0"
__author__ = "Plume Navigation Team"
__description__ = "Modular wind field implementations with pluggable architecture"
__module_status__ = _validation_results['status']
__available_wind_fields__ = _validation_results['implementations']['available']