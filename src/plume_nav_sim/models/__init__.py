"""
Models package for modular plume physics and wind dynamics.

This module establishes the plugin architecture for modular plume physics and wind dynamics,
providing factory functions and import structure for PlumeModelProtocol and WindFieldProtocol
implementations. Acts as the central registry for all plume and wind field models, supporting
the transformation from rigid simulator to configurable research platform.

The models package enables seamless switching between different plume simulation approaches
and environmental dynamics through configuration-driven component selection, supporting both
simple analytical models for rapid experimentation and realistic turbulent physics for
high-fidelity research scenarios.

Architecture Overview:
- PlumeModelProtocol implementations: GaussianPlumeModel, TurbulentPlumeModel, VideoPlumeAdapter
- WindFieldProtocol implementations: ConstantWindField, TurbulentWindField, TimeVaryingWindField
- Factory-based instantiation with Hydra integration for dependency injection
- Component registry system enabling runtime discovery and validation
- Graceful degradation during development with informative error messages

Key Design Principles:
- Protocol-based extensibility ensuring uniform interfaces across implementations
- Configuration-driven instantiation supporting both YAML and programmatic creation
- Performance optimization with lazy loading and optional dependency management
- Type safety through runtime protocol compliance validation
- Seamless integration with existing NavigatorProtocol and environment systems

Performance Requirements:
- Model instantiation: <100ms for complex configurations
- Protocol validation: <1ms per component for runtime checks
- Memory efficiency: <50MB overhead for model registry and factory infrastructure
- Lazy loading: Import overhead only when specific models are requested

Examples:
    Basic model instantiation:
    >>> from plume_nav_sim.models import create_plume_model, create_wind_field
    >>> plume_model = create_plume_model({'type': 'GaussianPlumeModel', 'source_position': (50, 50)})
    >>> wind_field = create_wind_field({'type': 'ConstantWindField', 'velocity': (2.0, 0.5)})
    
    Configuration-driven creation:
    >>> config = {
    ...     'plume_model': {'type': 'TurbulentPlumeModel', 'filament_count': 500},
    ...     'wind_field': {'type': 'TurbulentWindField', 'turbulence_intensity': 0.3}
    ... }
    >>> components = create_modular_environment(config)
    
    Model discovery and validation:
    >>> available_models = list_available_plume_models()
    >>> print(f"Available plume models: {available_models}")
    >>> 
    >>> model = create_plume_model({'type': 'GaussianPlumeModel'})
    >>> is_valid = validate_protocol_compliance(model, PlumeModelProtocol)
    >>> assert is_valid, "Model must implement PlumeModelProtocol"
    
    Hydra integration:
    >>> from hydra import compose, initialize
    >>> with initialize(config_path="../conf"):
    ...     cfg = compose(config_name="config")
    ...     plume_model = create_plume_model(cfg.plume_model)
    ...     wind_field = create_wind_field(cfg.wind_field)
"""

from __future__ import annotations
import warnings
from typing import Dict, List, Any, Optional, Union, Type, Callable, TYPE_CHECKING
import importlib
from pathlib import Path

# Import core protocols for type annotations and validation
from ..core.protocols import (
    PlumeModelProtocol,
    WindFieldProtocol,
    SensorProtocol,
    ComponentConfigType
)

# Hydra configuration support
try:
    from omegaconf import DictConfig
    from hydra import utils as hydra_utils
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    hydra_utils = None
    HYDRA_AVAILABLE = False

# Enhanced logging support
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Type imports for protocols
if TYPE_CHECKING:
    from ..core.protocols import PlumeModelProtocol, WindFieldProtocol, SensorProtocol

# Global model registries for runtime discovery and validation
_PLUME_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
_WIND_FIELD_REGISTRY: Dict[str, Dict[str, Any]] = {}
_SENSOR_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Import status tracking for graceful degradation during development
_IMPORT_STATUS: Dict[str, Dict[str, Any]] = {
    'plume_models': {},
    'wind_fields': {},
    'sensors': {}
}


def register_plume_model(
    model_name: str,
    model_class: Type['PlumeModelProtocol'],
    description: str,
    config_schema: Optional[Dict[str, Any]] = None,
    performance_characteristics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a plume model implementation in the global registry.
    
    Args:
        model_name: Unique identifier for the model (e.g., 'GaussianPlumeModel')
        model_class: Class implementing PlumeModelProtocol
        description: Human-readable description of the model's capabilities
        config_schema: Optional schema defining expected configuration parameters
        performance_characteristics: Optional performance metrics and requirements
        
    Examples:
        Register a new plume model:
        >>> register_plume_model(
        ...     'CustomPlumeModel',
        ...     CustomPlumeModel,
        ...     'Custom analytical plume model with enhanced dispersion physics',
        ...     config_schema={'source_strength': float, 'dispersion_coeff': float},
        ...     performance_characteristics={'complexity': 'medium', 'memory_usage': 'low'}
        ... )
    """
    _PLUME_MODEL_REGISTRY[model_name] = {
        'class': model_class,
        'description': description,
        'config_schema': config_schema or {},
        'performance_characteristics': performance_characteristics or {},
        'module_path': f"{model_class.__module__}.{model_class.__name__}"
    }
    
    # Log registration for monitoring
    if LOGURU_AVAILABLE:
        logger.debug(
            f"Registered plume model: {model_name}",
            model_class=model_class.__name__,
            module=model_class.__module__,
            description=description
        )


def register_wind_field(
    field_name: str,
    field_class: Type['WindFieldProtocol'],
    description: str,
    config_schema: Optional[Dict[str, Any]] = None,
    performance_characteristics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a wind field implementation in the global registry.
    
    Args:
        field_name: Unique identifier for the wind field (e.g., 'ConstantWindField')
        field_class: Class implementing WindFieldProtocol
        description: Human-readable description of the field's capabilities
        config_schema: Optional schema defining expected configuration parameters
        performance_characteristics: Optional performance metrics and requirements
    """
    _WIND_FIELD_REGISTRY[field_name] = {
        'class': field_class,
        'description': description,
        'config_schema': config_schema or {},
        'performance_characteristics': performance_characteristics or {},
        'module_path': f"{field_class.__module__}.{field_class.__name__}"
    }
    
    if LOGURU_AVAILABLE:
        logger.debug(
            f"Registered wind field: {field_name}",
            field_class=field_class.__name__,
            module=field_class.__module__,
            description=description
        )


def register_sensor(
    sensor_name: str,
    sensor_class: Type['SensorProtocol'],
    description: str,
    config_schema: Optional[Dict[str, Any]] = None,
    performance_characteristics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a sensor implementation in the global registry.
    
    Args:
        sensor_name: Unique identifier for the sensor (e.g., 'BinarySensor')
        sensor_class: Class implementing SensorProtocol
        description: Human-readable description of the sensor's capabilities
        config_schema: Optional schema defining expected configuration parameters
        performance_characteristics: Optional performance metrics and requirements
    """
    _SENSOR_REGISTRY[sensor_name] = {
        'class': sensor_class,
        'description': description,
        'config_schema': config_schema or {},
        'performance_characteristics': performance_characteristics or {},
        'module_path': f"{sensor_class.__module__}.{sensor_class.__name__}"
    }
    
    if LOGURU_AVAILABLE:
        logger.debug(
            f"Registered sensor: {sensor_name}",
            sensor_class=sensor_class.__name__,
            module=sensor_class.__module__,
            description=description
        )


def create_plume_model(config: Union[Dict[str, Any], DictConfig]) -> 'PlumeModelProtocol':
    """
    Factory function to create plume model instances from configuration.
    
    Supports both Hydra instantiation patterns and registry-based creation for maximum
    flexibility across different deployment scenarios. Provides comprehensive error
    handling and validation for robust component creation.
    
    Args:
        config: Configuration dictionary containing model type and parameters.
            Supports two patterns:
            1. Hydra pattern: {'_target_': 'module.ClassName', 'param1': value1, ...}
            2. Registry pattern: {'type': 'ModelName', 'param1': value1, ...}
            
    Returns:
        PlumeModelProtocol: Configured plume model instance implementing the protocol
        
    Raises:
        ValueError: If model type is unknown or configuration is invalid
        ImportError: If required model implementation is not available
        TypeError: If created instance doesn't implement PlumeModelProtocol
        
    Examples:
        Hydra-style instantiation:
        >>> config = {
        ...     '_target_': 'plume_nav_sim.models.plume.GaussianPlumeModel',
        ...     'source_position': (50, 50),
        ...     'source_strength': 1000.0,
        ...     'dispersion_coefficients': (2.0, 1.5)
        ... }
        >>> model = create_plume_model(config)
        
        Registry-style instantiation:
        >>> config = {
        ...     'type': 'TurbulentPlumeModel',
        ...     'filament_count': 500,
        ...     'turbulence_intensity': 0.3,
        ...     'wind_field_coupling': True
        ... }
        >>> model = create_plume_model(config)
        
        Simple configuration:
        >>> model = create_plume_model({'type': 'GaussianPlumeModel'})
    """
    # Handle Hydra instantiation pattern
    if HYDRA_AVAILABLE and isinstance(config, (dict, DictConfig)) and '_target_' in config:
        try:
            model = hydra_utils.instantiate(config)
            
            # Validate protocol compliance
            if not isinstance(model, PlumeModelProtocol):
                raise TypeError(
                    f"Created model {type(model)} does not implement PlumeModelProtocol"
                )
            
            if LOGURU_AVAILABLE:
                logger.info(
                    f"Created plume model via Hydra instantiation",
                    target=config['_target_'],
                    model_type=type(model).__name__
                )
            
            return model
            
        except Exception as e:
            raise ImportError(
                f"Failed to instantiate plume model via Hydra: {config.get('_target_', 'unknown')}. "
                f"Error: {e}"
            )
    
    # Handle registry-based instantiation
    model_type = config.get('type', config.get('model_type', 'GaussianPlumeModel'))
    
    # Try registry lookup first
    if model_type in _PLUME_MODEL_REGISTRY:
        try:
            model_info = _PLUME_MODEL_REGISTRY[model_type]
            model_class = model_info['class']
            
            # Extract configuration parameters (exclude 'type' field)
            model_config = {k: v for k, v in config.items() if k not in ['type', 'model_type']}
            
            model = model_class(**model_config)
            
            if LOGURU_AVAILABLE:
                logger.info(
                    f"Created plume model via registry",
                    model_type=model_type,
                    config_keys=list(model_config.keys())
                )
            
            return model
            
        except Exception as e:
            raise ValueError(
                f"Failed to create plume model '{model_type}' from registry. "
                f"Error: {e}"
            )
    
    # Fallback to direct import attempts for common models
    try:
        if model_type == 'GaussianPlumeModel':
            from .plume.gaussian_plume import GaussianPlumeModel
            model_class = GaussianPlumeModel
        elif model_type == 'TurbulentPlumeModel':
            from .plume.turbulent_plume import TurbulentPlumeModel
            model_class = TurbulentPlumeModel
        elif model_type == 'VideoPlumeAdapter':
            from .plume.video_plume_adapter import VideoPlumeAdapter
            model_class = VideoPlumeAdapter
        else:
            raise ValueError(f"Unknown plume model type: {model_type}")
        
        # Create model instance
        model_config = {k: v for k, v in config.items() if k not in ['type', 'model_type']}
        model = model_class(**model_config)
        
        if LOGURU_AVAILABLE:
            logger.info(
                f"Created plume model via direct import",
                model_type=model_type,
                module=model_class.__module__
            )
        
        return model
        
    except ImportError as e:
        # Update import status for diagnostics
        _IMPORT_STATUS['plume_models'][model_type] = {
            'available': False,
            'error': str(e),
            'attempted_import': True
        }
        
        raise ImportError(
            f"Plume model implementation not available: {model_type}. "
            f"Ensure the corresponding module has been created. "
            f"Available models: {list_available_plume_models()}. "
            f"Error: {e}"
        )


def create_wind_field(config: Union[Dict[str, Any], DictConfig]) -> 'WindFieldProtocol':
    """
    Factory function to create wind field instances from configuration.
    
    Args:
        config: Configuration dictionary containing wind field type and parameters
        
    Returns:
        WindFieldProtocol: Configured wind field instance implementing the protocol
        
    Examples:
        Constant wind field:
        >>> config = {'type': 'ConstantWindField', 'velocity': (2.0, 0.5)}
        >>> wind_field = create_wind_field(config)
        
        Turbulent wind field:
        >>> config = {
        ...     'type': 'TurbulentWindField',
        ...     'mean_velocity': (3.0, 1.0),
        ...     'turbulence_intensity': 0.2,
        ...     'correlation_length': 10.0
        ... }
        >>> wind_field = create_wind_field(config)
    """
    # Handle Hydra instantiation pattern
    if HYDRA_AVAILABLE and isinstance(config, (dict, DictConfig)) and '_target_' in config:
        try:
            wind_field = hydra_utils.instantiate(config)
            
            if not isinstance(wind_field, WindFieldProtocol):
                raise TypeError(
                    f"Created wind field {type(wind_field)} does not implement WindFieldProtocol"
                )
            
            if LOGURU_AVAILABLE:
                logger.info(
                    f"Created wind field via Hydra instantiation",
                    target=config['_target_'],
                    field_type=type(wind_field).__name__
                )
            
            return wind_field
            
        except Exception as e:
            raise ImportError(
                f"Failed to instantiate wind field via Hydra: {config.get('_target_', 'unknown')}. "
                f"Error: {e}"
            )
    
    # Handle registry-based instantiation
    field_type = config.get('type', config.get('field_type', 'ConstantWindField'))
    
    # Try registry lookup first
    if field_type in _WIND_FIELD_REGISTRY:
        try:
            field_info = _WIND_FIELD_REGISTRY[field_type]
            field_class = field_info['class']
            
            field_config = {k: v for k, v in config.items() if k not in ['type', 'field_type']}
            wind_field = field_class(**field_config)
            
            if LOGURU_AVAILABLE:
                logger.info(
                    f"Created wind field via registry",
                    field_type=field_type,
                    config_keys=list(field_config.keys())
                )
            
            return wind_field
            
        except Exception as e:
            raise ValueError(
                f"Failed to create wind field '{field_type}' from registry. "
                f"Error: {e}"
            )
    
    # Fallback to direct import attempts
    try:
        if field_type == 'ConstantWindField':
            from .wind.constant_wind import ConstantWindField
            field_class = ConstantWindField
        elif field_type == 'TurbulentWindField':
            from .wind.turbulent_wind import TurbulentWindField
            field_class = TurbulentWindField
        elif field_type == 'TimeVaryingWindField':
            from .wind.time_varying_wind import TimeVaryingWindField
            field_class = TimeVaryingWindField
        else:
            raise ValueError(f"Unknown wind field type: {field_type}")
        
        field_config = {k: v for k, v in config.items() if k not in ['type', 'field_type']}
        wind_field = field_class(**field_config)
        
        if LOGURU_AVAILABLE:
            logger.info(
                f"Created wind field via direct import",
                field_type=field_type,
                module=field_class.__module__
            )
        
        return wind_field
        
    except ImportError as e:
        _IMPORT_STATUS['wind_fields'][field_type] = {
            'available': False,
            'error': str(e),
            'attempted_import': True
        }
        
        raise ImportError(
            f"Wind field implementation not available: {field_type}. "
            f"Ensure the corresponding module has been created. "
            f"Available fields: {list_available_wind_fields()}. "
            f"Error: {e}"
        )


def create_sensors(sensor_configs: List[Union[Dict[str, Any], DictConfig]]) -> List['SensorProtocol']:
    """
    Factory function to create multiple sensor instances from configuration list.
    
    Args:
        sensor_configs: List of sensor configuration dictionaries
        
    Returns:
        List[SensorProtocol]: List of configured sensor instances
        
    Examples:
        Multi-sensor setup:
        >>> sensor_configs = [
        ...     {'type': 'BinarySensor', 'threshold': 0.1, 'false_positive_rate': 0.02},
        ...     {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1), 'resolution': 0.001},
        ...     {'type': 'GradientSensor', 'spatial_resolution': (0.5, 0.5)}
        ... ]
        >>> sensors = create_sensors(sensor_configs)
    """
    sensors = []
    
    for i, config in enumerate(sensor_configs):
        try:
            # Handle Hydra instantiation
            if HYDRA_AVAILABLE and isinstance(config, (dict, DictConfig)) and '_target_' in config:
                sensor = hydra_utils.instantiate(config)
                
                if not isinstance(sensor, SensorProtocol):
                    raise TypeError(
                        f"Created sensor {type(sensor)} does not implement SensorProtocol"
                    )
                
                sensors.append(sensor)
                continue
            
            # Handle registry-based instantiation
            sensor_type = config.get('type', config.get('sensor_type', 'ConcentrationSensor'))
            
            if sensor_type in _SENSOR_REGISTRY:
                sensor_info = _SENSOR_REGISTRY[sensor_type]
                sensor_class = sensor_info['class']
                
                sensor_config = {k: v for k, v in config.items() if k not in ['type', 'sensor_type']}
                sensor = sensor_class(**sensor_config)
                sensors.append(sensor)
                continue
            
            # Fallback to direct import
            try:
                if sensor_type == 'BinarySensor':
                    from ..core.sensors.binary_sensor import BinarySensor
                    sensor_class = BinarySensor
                elif sensor_type == 'ConcentrationSensor':
                    from ..core.sensors.concentration_sensor import ConcentrationSensor
                    sensor_class = ConcentrationSensor
                elif sensor_type == 'GradientSensor':
                    from ..core.sensors.gradient_sensor import GradientSensor
                    sensor_class = GradientSensor
                else:
                    raise ValueError(f"Unknown sensor type: {sensor_type}")
                
                sensor_config = {k: v for k, v in config.items() if k not in ['type', 'sensor_type']}
                sensor = sensor_class(**sensor_config)
                sensors.append(sensor)
                
            except ImportError as e:
                _IMPORT_STATUS['sensors'][sensor_type] = {
                    'available': False,
                    'error': str(e),
                    'attempted_import': True
                }
                
                raise ImportError(
                    f"Sensor implementation not available: {sensor_type} (config {i}). "
                    f"Ensure the corresponding module has been created. "
                    f"Error: {e}"
                )
                
        except Exception as e:
            raise ValueError(
                f"Failed to create sensor from config {i}: {config}. Error: {e}"
            )
    
    if LOGURU_AVAILABLE:
        logger.info(
            f"Created {len(sensors)} sensors",
            sensor_types=[type(s).__name__ for s in sensors]
        )
    
    return sensors


def create_modular_environment(config: Union[Dict[str, Any], DictConfig]) -> Dict[str, Any]:
    """
    Create complete modular environment with all specified components.
    
    Args:
        config: Configuration dictionary containing component specifications:
            - plume_model: Plume model configuration
            - wind_field: Optional wind field configuration  
            - sensors: Optional list of sensor configurations
            - environment: Optional environment-specific settings
            
    Returns:
        Dict[str, Any]: Dictionary containing all created components
        
    Examples:
        Complete modular environment:
        >>> config = {
        ...     'plume_model': {
        ...         'type': 'GaussianPlumeModel',
        ...         'source_position': (50, 50),
        ...         'source_strength': 1000.0
        ...     },
        ...     'wind_field': {
        ...         'type': 'ConstantWindField',
        ...         'velocity': (2.0, 0.5)
        ...     },
        ...     'sensors': [
        ...         {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)}
        ...     ]
        ... }
        >>> components = create_modular_environment(config)
        >>> plume_model = components['plume_model']
        >>> wind_field = components['wind_field']
        >>> sensors = components['sensors']
    """
    components = {}
    
    # Create plume model (required)
    if 'plume_model' not in config:
        raise ValueError("Configuration must include 'plume_model' specification")
    
    components['plume_model'] = create_plume_model(config['plume_model'])
    
    # Create wind field (optional)
    if 'wind_field' in config:
        components['wind_field'] = create_wind_field(config['wind_field'])
    else:
        components['wind_field'] = None
    
    # Create sensors (optional)
    if 'sensors' in config:
        components['sensors'] = create_sensors(config['sensors'])
    else:
        components['sensors'] = []
    
    # Store additional configuration
    components['environment_config'] = config.get('environment', {})
    components['metadata'] = {
        'created_components': list(components.keys()),
        'has_wind_field': components['wind_field'] is not None,
        'sensor_count': len(components['sensors']),
        'plume_model_type': type(components['plume_model']).__name__,
        'wind_field_type': type(components['wind_field']).__name__ if components['wind_field'] else None
    }
    
    if LOGURU_AVAILABLE:
        logger.info(
            "Created modular environment",
            **components['metadata']
        )
    
    return components


def list_available_plume_models() -> List[str]:
    """
    Get list of available plume model types.
    
    Returns:
        List[str]: List of plume model type names that can be instantiated
        
    Examples:
        Discover available models:
        >>> models = list_available_plume_models()
        >>> print(f"Available plume models: {models}")
    """
    available_models = list(_PLUME_MODEL_REGISTRY.keys())
    
    # Add fallback models that have direct import support
    fallback_models = ['GaussianPlumeModel', 'TurbulentPlumeModel', 'VideoPlumeAdapter']
    for model in fallback_models:
        if model not in available_models:
            # Check if import status indicates availability
            status = _IMPORT_STATUS['plume_models'].get(model, {})
            if not status.get('attempted_import', False) or status.get('available', False):
                available_models.append(model)
    
    return sorted(available_models)


def list_available_wind_fields() -> List[str]:
    """
    Get list of available wind field types.
    
    Returns:
        List[str]: List of wind field type names that can be instantiated
    """
    available_fields = list(_WIND_FIELD_REGISTRY.keys())
    
    fallback_fields = ['ConstantWindField', 'TurbulentWindField', 'TimeVaryingWindField']
    for field in fallback_fields:
        if field not in available_fields:
            status = _IMPORT_STATUS['wind_fields'].get(field, {})
            if not status.get('attempted_import', False) or status.get('available', False):
                available_fields.append(field)
    
    return sorted(available_fields)


def list_available_sensors() -> List[str]:
    """
    Get list of available sensor types.
    
    Returns:
        List[str]: List of sensor type names that can be instantiated
    """
    available_sensors = list(_SENSOR_REGISTRY.keys())
    
    fallback_sensors = ['BinarySensor', 'ConcentrationSensor', 'GradientSensor']
    for sensor in fallback_sensors:
        if sensor not in available_sensors:
            status = _IMPORT_STATUS['sensors'].get(sensor, {})
            if not status.get('attempted_import', False) or status.get('available', False):
                available_sensors.append(sensor)
    
    return sorted(available_sensors)


def get_model_registry() -> Dict[str, Dict[str, Any]]:
    """
    Get complete model registry for inspection and debugging.
    
    Returns:
        Dict containing all registered plume models, wind fields, and sensors
        
    Examples:
        Inspect model registry:
        >>> registry = get_model_registry()
        >>> for model_type, models in registry.items():
        ...     print(f"{model_type}: {list(models.keys())}")
    """
    return {
        'plume_models': _PLUME_MODEL_REGISTRY.copy(),
        'wind_fields': _WIND_FIELD_REGISTRY.copy(),
        'sensors': _SENSOR_REGISTRY.copy(),
        'import_status': _IMPORT_STATUS.copy()
    }


def validate_protocol_compliance(
    component: Any,
    protocol_type: Type
) -> bool:
    """
    Validate that a component implements the specified protocol.
    
    Args:
        component: Component instance to validate
        protocol_type: Protocol class to check against (e.g., PlumeModelProtocol)
        
    Returns:
        bool: True if component implements the protocol
        
    Examples:
        Validate component compliance:
        >>> model = create_plume_model({'type': 'GaussianPlumeModel'})
        >>> is_valid = validate_protocol_compliance(model, PlumeModelProtocol)
        >>> assert is_valid, "Model must implement PlumeModelProtocol"
    """
    try:
        return isinstance(component, protocol_type)
    except Exception:
        # Fallback to duck-typing validation for protocol checking
        if protocol_type == PlumeModelProtocol:
            required_methods = ['concentration_at', 'step', 'reset']
        elif protocol_type == WindFieldProtocol:
            required_methods = ['velocity_at', 'step', 'reset']
        elif protocol_type == SensorProtocol:
            required_methods = ['detect', 'measure', 'compute_gradient', 'configure']
        else:
            return False
        
        return all(hasattr(component, method) for method in required_methods)


def get_import_diagnostics() -> Dict[str, Any]:
    """
    Get diagnostic information about component import status and availability.
    
    Returns:
        Dict containing import status, available components, and error information
        
    Examples:
        Check import diagnostics:
        >>> diagnostics = get_import_diagnostics()
        >>> if diagnostics['errors']:
        ...     print(f"Import errors detected: {diagnostics['errors']}")
    """
    diagnostics = {
        'available_components': {
            'plume_models': list_available_plume_models(),
            'wind_fields': list_available_wind_fields(),
            'sensors': list_available_sensors()
        },
        'registry_status': {
            'plume_models': len(_PLUME_MODEL_REGISTRY),
            'wind_fields': len(_WIND_FIELD_REGISTRY),
            'sensors': len(_SENSOR_REGISTRY)
        },
        'import_status': _IMPORT_STATUS.copy(),
        'errors': [],
        'warnings': []
    }
    
    # Collect import errors
    for component_type, components in _IMPORT_STATUS.items():
        for component_name, status in components.items():
            if not status.get('available', True) and status.get('attempted_import', False):
                diagnostics['errors'].append({
                    'component_type': component_type,
                    'component_name': component_name,
                    'error': status.get('error', 'Unknown error')
                })
    
    # Generate warnings for missing implementations
    if not diagnostics['available_components']['plume_models']:
        diagnostics['warnings'].append(
            "No plume models available. Ensure plume model implementations are created."
        )
    
    if not diagnostics['available_components']['wind_fields']:
        diagnostics['warnings'].append(
            "No wind fields available. Consider creating basic wind field implementations."
        )
    
    return diagnostics


def auto_discover_models() -> None:
    """
    Automatically discover and register model implementations from the models package.
    
    This function attempts to import and register all available model implementations
    to populate the registries without requiring manual registration calls.
    """
    # Get the models package directory
    models_package_path = Path(__file__).parent
    
    # Try to auto-discover plume models
    plume_path = models_package_path / 'plume'
    if plume_path.exists():
        for module_file in plume_path.glob('*.py'):
            if module_file.name.startswith('_'):
                continue
            
            module_name = module_file.stem
            try:
                module = importlib.import_module(f'.plume.{module_name}', package=__name__.rsplit('.', 1)[0])
                
                # Look for classes that might implement PlumeModelProtocol
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name.endswith('PlumeModel') and
                        hasattr(attr, 'concentration_at')):
                        
                        # Auto-register if not already registered
                        if attr_name not in _PLUME_MODEL_REGISTRY:
                            register_plume_model(
                                attr_name,
                                attr,
                                f"Auto-discovered plume model from {module_name}",
                                performance_characteristics={'auto_discovered': True}
                            )
                            
            except ImportError:
                pass  # Skip modules that can't be imported
    
    # Try to auto-discover wind fields
    wind_path = models_package_path / 'wind'
    if wind_path.exists():
        for module_file in wind_path.glob('*.py'):
            if module_file.name.startswith('_'):
                continue
            
            module_name = module_file.stem
            try:
                module = importlib.import_module(f'.wind.{module_name}', package=__name__.rsplit('.', 1)[0])
                
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name.endswith('WindField') and
                        hasattr(attr, 'velocity_at')):
                        
                        if attr_name not in _WIND_FIELD_REGISTRY:
                            register_wind_field(
                                attr_name,
                                attr,
                                f"Auto-discovered wind field from {module_name}",
                                performance_characteristics={'auto_discovered': True}
                            )
                            
            except ImportError:
                pass


# Initialize the package by attempting auto-discovery
try:
    auto_discover_models()
    
    if LOGURU_AVAILABLE:
        logger.debug(
            "Models package initialized",
            plume_models=len(_PLUME_MODEL_REGISTRY),
            wind_fields=len(_WIND_FIELD_REGISTRY),
            sensors=len(_SENSOR_REGISTRY)
        )
        
except Exception as e:
    # Don't fail package import if auto-discovery fails
    if LOGURU_AVAILABLE:
        logger.warning(f"Auto-discovery failed, continuing with manual registration: {e}")
    else:
        warnings.warn(f"Models auto-discovery failed: {e}", UserWarning, stacklevel=2)


# Public API exports
__all__ = [
    # Core protocols for type annotations
    'PlumeModelProtocol',
    'WindFieldProtocol',
    'SensorProtocol',
    
    # Factory functions for component creation
    'create_plume_model',
    'create_wind_field',
    'create_sensors',
    'create_modular_environment',
    
    # Registry management functions
    'register_plume_model',
    'register_wind_field',
    'register_sensor',
    
    # Discovery and validation functions
    'list_available_plume_models',
    'list_available_wind_fields',
    'list_available_sensors',
    'get_model_registry',
    'validate_protocol_compliance',
    
    # Diagnostic and utility functions
    'get_import_diagnostics',
    'auto_discover_models'
]

# Package metadata
__version__ = "0.1.0"
__description__ = "Modular plume physics and wind dynamics package with plugin architecture"
__author__ = "Plume Navigation Team"

# Performance monitoring metadata
__performance_characteristics__ = {
    "factory_instantiation_ms": 100,  # Target <100ms for complex configurations
    "protocol_validation_ms": 1,     # Target <1ms for runtime validation
    "registry_overhead_mb": 50,      # Target <50MB for registry infrastructure
    "auto_discovery_ms": 2000        # Target <2s for full auto-discovery
}

# Compatibility features
__compatibility_features__ = {
    "hydra_integration": HYDRA_AVAILABLE,
    "enhanced_logging": LOGURU_AVAILABLE,
    "auto_discovery": True,
    "protocol_validation": True,
    "registry_based_creation": True,
    "graceful_degradation": True,
    "development_mode_support": True
}