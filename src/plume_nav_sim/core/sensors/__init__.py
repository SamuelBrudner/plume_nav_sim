"""
Sensor abstraction layer providing configurable detection models for flexible agent perception.

This module serves as the public API facade for the sensor abstraction layer, implementing
the modular architecture requirements from Section 0.2.1 of the technical specification.
The sensor system enables researchers to switch between different sensing modalities
(binary detection, quantitative concentration, spatial gradients) without code changes,
supporting both memory-based and non-memory-based navigation strategies.

Key Architectural Components:
- SensorProtocol implementations: BinarySensor, ConcentrationSensor, GradientSensor
- BaseSensor infrastructure: Shared functionality and performance optimizations
- HistoricalSensor wrapper: Optional temporal history integration for memory-based agents
- Sensor factory functions: Hydra-based configuration instantiation with dependency injection
- Performance monitoring: Sub-10ms step latency compliance with comprehensive metrics
- Configuration utilities: Runtime component swapping without code modifications

The sensor abstraction layer replaces direct field sampling with protocol-based observation
processing, enabling agent-agnostic design where the simulator core makes no assumptions
about sensing capabilities or temporal integration requirements. This supports the goal
of configuration-driven component selection while maintaining enterprise-grade performance.

Sensor Modality Overview:

Binary Detection (BinarySensor):
    - Threshold-based detection with configurable false positive/negative rates
    - Hysteresis support to prevent detection oscillations
    - Ideal for simple presence/absence navigation strategies
    - Sub-microsecond per-agent processing for real-time performance

Quantitative Measurement (ConcentrationSensor):
    - Calibrated concentration readings with dynamic range configuration
    - Temporal filtering and response delay modeling for realism
    - Saturation effects and calibration drift simulation
    - Linear scaling with agent count through vectorized operations

Spatial Gradient Computation (GradientSensor):
    - Finite difference gradient estimation with adaptive step sizing
    - Multi-point sampling for noise suppression and accuracy
    - Configurable spatial resolution and derivative order
    - Essential for gradient-following navigation algorithms

Historical Integration (HistoricalSensor):
    - Temporal observation sequences for memory-based navigation
    - Configurable history length and temporal sampling strategies
    - Zero-copy integration with base sensor implementations
    - Optional feature for agents requiring temporal context

Performance Requirements:
- Single sensor operation: <0.1ms per agent per sensor
- Multi-agent batch processing: <1ms for 100 agents with multiple sensors
- Memory efficiency: <10MB for historical data with configurable limits
- Vectorized operations: Linear scaling with agent count
- Integration overhead: <0.1ms additional latency per sensor protocol call

Configuration Integration:
- Hydra instantiate() support for dependency injection
- Runtime component swapping via configuration updates
- Type-safe configuration schemas with validation
- Multi-sensor setup with heterogeneous sensor types
- Wind field integration when environmental dynamics are enabled

Examples:
    Basic sensor instantiation:
        >>> from plume_nav_sim.core.sensors import BinarySensor, ConcentrationSensor
        >>> binary_sensor = BinarySensor(threshold=0.1, false_positive_rate=0.02)
        >>> conc_sensor = ConcentrationSensor(dynamic_range=(0, 1), resolution=0.001)
        
    Configuration-driven sensor factory:
        >>> from plume_nav_sim.core.sensors import create_sensor_from_config
        >>> sensor_config = {
        ...     'type': 'GradientSensor',
        ...     'spatial_resolution': (0.5, 0.5),
        ...     'method': 'central'
        ... }
        >>> gradient_sensor = create_sensor_from_config(sensor_config)
        
    Multi-sensor environment setup:
        >>> from plume_nav_sim.core.sensors import create_sensor_suite
        >>> sensor_configs = [
        ...     {'type': 'BinarySensor', 'threshold': 0.1},
        ...     {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)},
        ...     {'type': 'GradientSensor', 'spatial_resolution': (0.2, 0.2)}
        ... ]
        >>> sensors = create_sensor_suite(sensor_configs)
        
    Historical sensor wrapper:
        >>> from plume_nav_sim.core.sensors import HistoricalSensor, ConcentrationSensor
        >>> base_sensor = ConcentrationSensor(dynamic_range=(0, 1))
        >>> historical_sensor = HistoricalSensor(
        ...     base_sensor=base_sensor,
        ...     history_length=10,
        ...     sampling_interval=1.0
        ... )
        
    Hydra integration:
        >>> import hydra
        >>> from plume_nav_sim.core.sensors import validate_sensor_config
        >>> 
        >>> # In your configuration
        >>> sensor_cfg = {
        ...     '_target_': 'plume_nav_sim.core.sensors.BinarySensor',
        ...     'threshold': 0.1,
        ...     'false_positive_rate': 0.02
        ... }
        >>> sensor = hydra.utils.instantiate(sensor_cfg)
        >>> assert validate_sensor_config(sensor_cfg)

Notes:
    The sensor abstraction layer is designed to enable the transition from rigid simulator
    architecture to highly configurable research platform supporting diverse navigation
    strategies. All sensor implementations maintain protocol compliance for seamless
    interoperability while providing specialized functionality for different research
    requirements.
    
    Performance monitoring is integrated throughout the sensor system to ensure sub-10ms
    step execution requirements are maintained even with complex multi-sensor setups.
    Memory management includes automatic cleanup of temporal data and configurable limits
    to prevent memory leaks in long-duration experiments.
    
    Wind field integration is supported through sensor configuration when environmental
    dynamics are enabled, allowing sensors to account for transport effects in their
    measurements without breaking agent-agnostic design principles.
"""

from __future__ import annotations
from typing import Optional, Union, List, Dict, Any, Type, Callable
import numpy as np

# Core protocol import for type safety
from plume_nav_sim.protocols.sensor import SensorProtocol
PROTOCOLS_AVAILABLE = True

# Hydra integration for configuration management
try:
    from hydra import utils as hydra_utils
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    hydra_utils = None
    DictConfig = dict
    OmegaConf = None
    HYDRA_AVAILABLE = False

# Performance monitoring integration
try:
    import time
    import psutil
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    psutil = None
    PERFORMANCE_MONITORING_AVAILABLE = False

# Enhanced logging support
from loguru import logger
LOGURU_AVAILABLE = True


# =============================================================================
# Sensor Implementation Imports
# =============================================================================

try:
    from .base_sensor import BaseSensor
    BASE_SENSOR_AVAILABLE = True
except ImportError as exc:
    logger.exception("Failed to import BaseSensor", exc_info=exc)
    raise

try:
    from .binary_sensor import BinarySensor
    BINARY_SENSOR_AVAILABLE = True
except ImportError as exc:
    logger.exception("Failed to import BinarySensor", exc_info=exc)
    raise

try:
    from .concentration_sensor import ConcentrationSensor
    CONCENTRATION_SENSOR_AVAILABLE = True
except ImportError as exc:
    logger.exception("Failed to import ConcentrationSensor", exc_info=exc)
    raise

try:
    from .gradient_sensor import GradientSensor
    GRADIENT_SENSOR_AVAILABLE = True
except ImportError as exc:
    logger.exception("Failed to import GradientSensor", exc_info=exc)
    raise

try:
    from .historical_sensor import HistoricalSensor
    HISTORICAL_SENSOR_AVAILABLE = True
except ImportError as exc:
    logger.exception("Failed to import HistoricalSensor", exc_info=exc)
    raise


# =============================================================================
# Sensor Factory Functions
# =============================================================================

def create_sensor_from_config(config: Union[Dict[str, Any], DictConfig]) -> SensorProtocol:
    """
    Create sensor instance from configuration dictionary or Hydra config.
    
    This factory function enables configuration-driven sensor instantiation supporting
    both explicit factory methods and Hydra dependency injection patterns. Supports
    all sensor types with automatic validation and performance monitoring setup.
    
    Args:
        config: Sensor configuration dictionary or DictConfig containing:
            - type: Sensor type ('BinarySensor', 'ConcentrationSensor', 'GradientSensor')
            - _target_: Optional Hydra target for dependency injection
            - Additional sensor-specific parameters
            
    Returns:
        SensorProtocol: Configured sensor instance implementing full protocol
        
    Raises:
        ValueError: If sensor type is unknown or configuration is invalid
        ImportError: If sensor implementation is not available
        TypeError: If configuration format is unsupported
        
    Performance:
        Configuration parsing: <1ms for standard sensor types
        Validation overhead: <0.1ms additional per sensor
        
    Examples:
        Binary sensor creation:
        >>> config = {
        ...     'type': 'BinarySensor',
        ...     'threshold': 0.1,
        ...     'false_positive_rate': 0.02
        ... }
        >>> sensor = create_sensor_from_config(config)
        
        Hydra instantiation:
        >>> config = {
        ...     '_target_': 'plume_nav_sim.core.sensors.ConcentrationSensor',
        ...     'dynamic_range': (0, 2.0),
        ...     'resolution': 0.001
        ... }
        >>> sensor = create_sensor_from_config(config)
        
        Gradient sensor with advanced configuration:
        >>> config = {
        ...     'type': 'GradientSensor',
        ...     'spatial_resolution': (0.2, 0.2),
        ...     'method': 'central',
        ...     'order': 2
        ... }
        >>> sensor = create_sensor_from_config(config)
    """
    if HYDRA_AVAILABLE and '_target_' in config:
        # Use Hydra instantiation for dependency injection
        try:
            sensor = hydra_utils.instantiate(config)

            # Validate protocol compliance
            if not hasattr(sensor, 'configure'):
                raise TypeError(f"Instantiated sensor {type(sensor)} does not implement SensorProtocol")
            if LOGURU_AVAILABLE:
                metadata = sensor.get_metadata() if hasattr(sensor, 'get_metadata') else {}
                observation_shape = getattr(sensor, 'observation_shape', None)
                logger.debug(
                    "Sensor created",
                    sensor_type=type(sensor).__name__,
                    observation_shape=observation_shape,
                    metadata=metadata,
                    target=config.get('_target_'),
                    config_keys=list(config.keys())
                )

            return sensor
            
        except Exception as e:
            raise ImportError(f"Failed to instantiate sensor from Hydra target: {e}")
    
    # Factory method for explicit sensor type configuration
    sensor_type = config.get('type', 'ConcentrationSensor')
    sensor_params = {k: v for k, v in config.items() if k not in ['type', '_target_']}
    
    # Create sensor based on type
    if sensor_type == 'BinarySensor':
        if not BINARY_SENSOR_AVAILABLE:
            logger.error("BinarySensor implementation not available")
            raise ImportError("BinarySensor implementation not available")
        sensor = BinarySensor(**sensor_params)

    elif sensor_type == 'ConcentrationSensor':
        if not CONCENTRATION_SENSOR_AVAILABLE:
            logger.error("ConcentrationSensor implementation not available")
            raise ImportError("ConcentrationSensor implementation not available")
        sensor = ConcentrationSensor(**sensor_params)

    elif sensor_type == 'GradientSensor':
        if not GRADIENT_SENSOR_AVAILABLE:
            logger.error("GradientSensor implementation not available")
            raise ImportError("GradientSensor implementation not available")
        sensor = GradientSensor(**sensor_params)

    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")

    # Validate protocol compliance
    if not isinstance(sensor, SensorProtocol):
        raise TypeError(f"Instantiated sensor {type(sensor)} does not implement SensorProtocol")

    if LOGURU_AVAILABLE:
        metadata = sensor.get_metadata() if hasattr(sensor, 'get_metadata') else {}
        observation_shape = getattr(sensor, 'observation_shape', None)
        logger.debug(
            "Sensor created",
            sensor_type=type(sensor).__name__,
            observation_shape=observation_shape,
            metadata=metadata,
            parameters=sensor_params
        )

    return sensor


def create_sensor_suite(sensor_configs: List[Union[Dict[str, Any], DictConfig]]) -> List[SensorProtocol]:
    """
    Create multiple sensors from a list of configurations for multi-modal sensing.
    
    This function enables easy setup of heterogeneous sensor suites supporting different
    detection modalities within a single environment. Each sensor is configured
    independently while maintaining consistent performance characteristics.
    
    Args:
        sensor_configs: List of sensor configuration dictionaries
        
    Returns:
        List[SensorProtocol]: List of configured sensor instances
        
    Raises:
        ValueError: If any sensor configuration is invalid
        ImportError: If required sensor implementations are not available
        
    Performance:
        Sensor creation: <1ms per sensor for standard configurations
        Validation overhead: <0.1ms per sensor
        
    Examples:
        Multi-modal sensor setup:
        >>> sensor_configs = [
        ...     {'type': 'BinarySensor', 'threshold': 0.1},
        ...     {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)},
        ...     {'type': 'GradientSensor', 'spatial_resolution': (0.5, 0.5)}
        ... ]
        >>> sensors = create_sensor_suite(sensor_configs)
        >>> assert len(sensors) == 3
        
        Hydra-based sensor suite:
        >>> sensor_configs = [
        ...     {
        ...         '_target_': 'plume_nav_sim.core.sensors.BinarySensor',
        ...         'threshold': 0.05,
        ...         'false_positive_rate': 0.01
        ...     },
        ...     {
        ...         '_target_': 'plume_nav_sim.core.sensors.ConcentrationSensor',
        ...         'dynamic_range': (0, 2.0),
        ...         'noise_level': 0.05
        ...     }
        ... ]
        >>> sensors = create_sensor_suite(sensor_configs)
    """
    sensors = []

    for i, config in enumerate(sensor_configs):
        try:
            sensor = create_sensor_from_config(config)
            sensors.append(sensor)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to create sensor {i} from config {config}: {e}")
    
    if LOGURU_AVAILABLE:
        sensor_types = [type(sensor).__name__ for sensor in sensors]
        logger.info(
            "Sensor suite created successfully",
            sensor_count=len(sensors),
            sensor_types=sensor_types
        )
    
    return sensors


def create_historical_sensor(
    base_config: Union[Dict[str, Any], DictConfig],
    history_length: int = 10,
    sampling_interval: float = 1.0,
    **kwargs: Any
) -> HistoricalSensor:
    """
    Create a HistoricalSensor wrapper around a base sensor configuration.
    
    This convenience function creates a base sensor from configuration and wraps it
    with temporal history capabilities for memory-based navigation strategies.
    
    Args:
        base_config: Configuration for the base sensor
        history_length: Number of historical observations to maintain
        sampling_interval: Time interval between historical samples (seconds)
        **kwargs: Additional HistoricalSensor parameters
        
    Returns:
        HistoricalSensor: Configured historical sensor wrapper
        
    Examples:
        Historical concentration sensor:
        >>> base_config = {
        ...     'type': 'ConcentrationSensor',
        ...     'dynamic_range': (0, 1),
        ...     'resolution': 0.001
        ... }
        >>> historical_sensor = create_historical_sensor(
        ...     base_config,
        ...     history_length=20,
        ...     sampling_interval=0.5
        ... )
        
        Historical gradient sensor:
        >>> base_config = {
        ...     'type': 'GradientSensor',
        ...     'spatial_resolution': (0.2, 0.2)
        ... }
        >>> historical_sensor = create_historical_sensor(base_config, history_length=5)
    """
    if not HISTORICAL_SENSOR_AVAILABLE:
        logger.error("HistoricalSensor implementation not available")
        raise ImportError("HistoricalSensor implementation not available")

    base_sensor = create_sensor_from_config(base_config)

    historical_sensor = HistoricalSensor(
        base_sensor=base_sensor,
        history_length=history_length,
        sampling_interval=sampling_interval,
        **kwargs
    )
    
    if LOGURU_AVAILABLE:
        logger.debug(
            "Historical sensor created",
            base_sensor_type=type(base_sensor).__name__,
            history_length=history_length,
            sampling_interval=sampling_interval
        )
    
    return historical_sensor


# =============================================================================
# Sensor Validation and Utilities
# =============================================================================

def validate_sensor_config(config: Union[Dict[str, Any], DictConfig]) -> bool:
    """
    Validate sensor configuration for completeness and parameter constraints.
    
    This function performs comprehensive validation of sensor configurations to catch
    common errors before sensor instantiation, improving debugging experience.
    
    Args:
        config: Sensor configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration contains invalid parameters
        TypeError: If configuration type is unsupported
        
    Examples:
        Valid configuration validation:
        >>> config = {
        ...     'type': 'BinarySensor',
        ...     'threshold': 0.1,
        ...     'false_positive_rate': 0.02
        ... }
        >>> assert validate_sensor_config(config)
        
        Invalid configuration detection:
        >>> config = {
        ...     'type': 'BinarySensor',
        ...     'threshold': -0.1  # Invalid: negative threshold
        ... }
        >>> try:
        ...     validate_sensor_config(config)
        ... except ValueError as e:
        ...     print(f"Validation failed: {e}")
    """
    if not isinstance(config, (dict, DictConfig)):
        raise TypeError(f"Configuration must be dict or DictConfig, got {type(config)}")
    
    # Check for required fields
    if '_target_' not in config and 'type' not in config:
        raise ValueError("Configuration must specify either 'type' or '_target_'")
    
    sensor_type = config.get('type', 'Unknown')
    
    # Type-specific validation
    if sensor_type == 'BinarySensor':
        threshold = config.get('threshold', 0.1)
        if threshold < 0:
            raise ValueError(f"BinarySensor threshold must be >= 0, got {threshold}")
        
        false_positive_rate = config.get('false_positive_rate', 0.0)
        if not 0 <= false_positive_rate <= 1:
            raise ValueError(f"false_positive_rate must be in [0, 1], got {false_positive_rate}")
        
        false_negative_rate = config.get('false_negative_rate', 0.0)
        if not 0 <= false_negative_rate <= 1:
            raise ValueError(f"false_negative_rate must be in [0, 1], got {false_negative_rate}")
    
    elif sensor_type == 'ConcentrationSensor':
        dynamic_range = config.get('dynamic_range', (0.0, 1.0))
        if not isinstance(dynamic_range, (tuple, list)) or len(dynamic_range) != 2:
            raise ValueError(f"dynamic_range must be a 2-tuple, got {dynamic_range}")
        
        if dynamic_range[0] >= dynamic_range[1]:
            raise ValueError(f"Invalid dynamic_range: min >= max ({dynamic_range})")
        
        resolution = config.get('resolution', 0.001)
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
    
    elif sensor_type == 'GradientSensor':
        spatial_resolution = config.get('spatial_resolution', (0.5, 0.5))
        if not isinstance(spatial_resolution, (tuple, list)) or len(spatial_resolution) != 2:
            raise ValueError(f"spatial_resolution must be a 2-tuple, got {spatial_resolution}")
        
        if any(r <= 0 for r in spatial_resolution):
            raise ValueError(f"spatial_resolution components must be > 0, got {spatial_resolution}")
        
        method = config.get('method', 'central')
        if method not in ['forward', 'backward', 'central']:
            raise ValueError(f"method must be 'forward', 'backward', or 'central', got {method}")
    
    elif sensor_type != 'Unknown':
        # Allow unknown types if _target_ is specified (Hydra will validate)
        if '_target_' not in config:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
    
    return True


def get_sensor_performance_metrics(sensors: Union[SensorProtocol, List[SensorProtocol]]) -> Dict[str, Any]:
    """
    Collect performance metrics from sensors for monitoring and optimization.
    
    This function aggregates performance data from sensor instances to support
    system monitoring and performance optimization according to sub-10ms requirements.
    
    Args:
        sensors: Single sensor or list of sensors to collect metrics from
        
    Returns:
        Dict[str, Any]: Aggregated performance metrics including:
            - operation_count: Number of sensor operations performed
            - total_execution_time: Total time spent in sensor operations
            - average_latency: Average operation latency in milliseconds
            - min_latency: Minimum operation latency
            - max_latency: Maximum operation latency
            - memory_usage: Current memory usage (if psutil available)
            
    Examples:
        Single sensor metrics:
        >>> sensor = create_sensor_from_config({'type': 'ConcentrationSensor'})
        >>> # ... perform some operations ...
        >>> metrics = get_sensor_performance_metrics(sensor)
        >>> print(f"Average latency: {metrics['average_latency']:.3f} ms")
        
        Multi-sensor metrics:
        >>> sensors = create_sensor_suite([
        ...     {'type': 'BinarySensor', 'threshold': 0.1},
        ...     {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)}
        ... ])
        >>> metrics = get_sensor_performance_metrics(sensors)
    """
    if isinstance(sensors, list):
        sensor_list = sensors
    else:
        sensor_list = [sensors]
    
    aggregated_metrics = {
        'sensor_count': len(sensor_list),
        'total_operation_count': 0,
        'total_execution_time': 0.0,
        'individual_metrics': {}
    }
    
    execution_times = []
    
    for i, sensor in enumerate(sensor_list):
        sensor_name = f"{type(sensor).__name__}_{i}"
        
        if hasattr(sensor, 'get_performance_metrics'):
            metrics = sensor.get_performance_metrics()
            aggregated_metrics['individual_metrics'][sensor_name] = metrics
            
            # Aggregate timing data
            if 'operation_count' in metrics:
                aggregated_metrics['total_operation_count'] += metrics['operation_count']
            
            if 'total_execution_time' in metrics:
                aggregated_metrics['total_execution_time'] += metrics['total_execution_time']
            
            if 'execution_times' in metrics:
                execution_times.extend(metrics['execution_times'])
        else:
            aggregated_metrics['individual_metrics'][sensor_name] = {
                'note': 'Performance metrics not available for this sensor'
            }
    
    # Calculate aggregate statistics
    if execution_times:
        aggregated_metrics.update({
            'average_latency_ms': np.mean(execution_times) * 1000,
            'min_latency_ms': np.min(execution_times) * 1000,
            'max_latency_ms': np.max(execution_times) * 1000,
            'std_latency_ms': np.std(execution_times) * 1000,
            'latency_percentiles': {
                'p50': np.percentile(execution_times, 50) * 1000,
                'p95': np.percentile(execution_times, 95) * 1000,
                'p99': np.percentile(execution_times, 99) * 1000
            }
        })
    
    # Add system memory information if available
    if PERFORMANCE_MONITORING_AVAILABLE and psutil:
        process = psutil.Process()
        memory_info = process.memory_info()
        aggregated_metrics['system_metrics'] = {
            'memory_rss_mb': memory_info.rss / 1024 / 1024,
            'memory_vms_mb': memory_info.vms / 1024 / 1024,
            'cpu_percent': process.cpu_percent()
        }
    
    return aggregated_metrics


def reset_sensor_performance_metrics(sensors: Union[SensorProtocol, List[SensorProtocol]]) -> None:
    """
    Reset performance metrics for sensors to start fresh measurement periods.
    
    Args:
        sensors: Single sensor or list of sensors to reset metrics for
        
    Examples:
        Reset metrics for sensor suite:
        >>> sensors = create_sensor_suite([
        ...     {'type': 'BinarySensor', 'threshold': 0.1},
        ...     {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)}
        ... ])
        >>> reset_sensor_performance_metrics(sensors)
    """
    if isinstance(sensors, list):
        sensor_list = sensors
    else:
        sensor_list = [sensors]
    
    reset_count = 0
    for sensor in sensor_list:
        if hasattr(sensor, 'reset_performance_metrics'):
            sensor.reset_performance_metrics()
            reset_count += 1
    
    if LOGURU_AVAILABLE:
        logger.debug(
            "Performance metrics reset",
            total_sensors=len(sensor_list),
            reset_count=reset_count
        )


def configure_sensor_suite(
    sensors: List[SensorProtocol],
    configurations: List[Dict[str, Any]]
) -> None:
    """
    Configure multiple sensors with individual parameter updates.
    
    This utility function enables batch configuration updates for sensor suites,
    useful for runtime parameter adjustment during experiments.
    
    Args:
        sensors: List of sensor instances to configure
        configurations: List of configuration dictionaries (one per sensor)
        
    Raises:
        ValueError: If sensors and configurations lists have different lengths
        
    Examples:
        Batch sensor configuration:
        >>> sensors = create_sensor_suite([
        ...     {'type': 'BinarySensor', 'threshold': 0.1},
        ...     {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)}
        ... ])
        >>> new_configs = [
        ...     {'threshold': 0.05, 'false_positive_rate': 0.01},
        ...     {'resolution': 0.0005, 'noise_level': 0.02}
        ... ]
        >>> configure_sensor_suite(sensors, new_configs)
    """
    if len(sensors) != len(configurations):
        raise ValueError(
            f"Number of sensors ({len(sensors)}) must match number of "
            f"configurations ({len(configurations)})"
        )
    
    for sensor, config in zip(sensors, configurations):
        sensor.configure(**config)
    
    if LOGURU_AVAILABLE:
        logger.debug(
            "Sensor suite configured",
            sensor_count=len(sensors),
            configuration_keys=[list(config.keys()) for config in configurations]
        )


# =============================================================================
# Module Availability Check
# =============================================================================

def check_sensor_module_availability() -> Dict[str, bool]:
    """
    Check availability of sensor module components for diagnostics.
    
    Returns:
        Dict[str, bool]: Availability status of each component
        
    Examples:
        Check module status:
        >>> status = check_sensor_module_availability()
        >>> if not status['BinarySensor']:
        ...     print("BinarySensor implementation not yet available")
    """
    return {
        'SensorProtocol': PROTOCOLS_AVAILABLE,
        'BaseSensor': BASE_SENSOR_AVAILABLE,
        'BinarySensor': BINARY_SENSOR_AVAILABLE,
        'ConcentrationSensor': CONCENTRATION_SENSOR_AVAILABLE,
        'GradientSensor': GRADIENT_SENSOR_AVAILABLE,
        'HistoricalSensor': HISTORICAL_SENSOR_AVAILABLE,
        'HydraIntegration': HYDRA_AVAILABLE,
        'PerformanceMonitoring': PERFORMANCE_MONITORING_AVAILABLE,
        'StructuredLogging': LOGURU_AVAILABLE
    }


def get_sensor_module_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the sensor module for debugging.
    
    Returns:
        Dict[str, Any]: Module information including versions and capabilities
    """
    availability = check_sensor_module_availability()
    
    info = {
        'module_name': 'plume_nav_sim.core.sensors',
        'version': '0.3.0',
        'component_availability': availability,
        'supported_sensor_types': ['BinarySensor', 'ConcentrationSensor', 'GradientSensor'],
        'supported_features': {
            'configuration_driven_instantiation': True,
            'hydra_integration': HYDRA_AVAILABLE,
            'performance_monitoring': PERFORMANCE_MONITORING_AVAILABLE,
            'historical_sensor_wrapper': True,
            'multi_sensor_suites': True,
            'runtime_configuration': True,
            'protocol_compliance_validation': True
        },
        'performance_requirements': {
            'single_sensor_operation_ms': 0.1,
            'multi_agent_batch_processing_ms': 1.0,
            'memory_efficiency_mb': 10,
            'integration_overhead_ms': 0.1
        }
    }
    
    # Add implementation warnings
    unavailable_components = [k for k, v in availability.items() if not v]
    if unavailable_components:
        info['warnings'] = [
            f"Components not yet available: {', '.join(unavailable_components)}"
        ]
    
    return info


# =============================================================================
# Public API Exports
# =============================================================================

__all__ = [
    # Core sensor protocol and base infrastructure
    'SensorProtocol',
    'BaseSensor',
    
    # Sensor implementations
    'BinarySensor',
    'ConcentrationSensor', 
    'GradientSensor',
    'HistoricalSensor',
    
    # Sensor factory functions
    'create_sensor_from_config',
    'create_sensor_suite',
    'create_historical_sensor',
    
    # Sensor validation and configuration utilities
    'validate_sensor_config',
    'configure_sensor_suite',
    
    # Performance monitoring and utilities
    'get_sensor_performance_metrics',
    'reset_sensor_performance_metrics',
    
    # Module diagnostics and information
    'check_sensor_module_availability',
    'get_sensor_module_info'
]

# Module metadata
__version__ = '0.3.0'
__author__ = 'Plume Navigation Team'
__description__ = 'Sensor abstraction layer for configurable agent perception modeling'

# Performance requirements (per Section 0.2.1)
__performance_requirements__ = {
    'single_sensor_operation_ms': 0.1,
    'multi_agent_batch_processing_ms': 1.0,
    'memory_efficiency_mb': 10,
    'integration_overhead_ms': 0.1,
    'vectorized_scaling': 'linear'
}

# Module initialization logging
if LOGURU_AVAILABLE:
    logger.info(
        "Sensor abstraction layer initialized",
        available_components=sum(check_sensor_module_availability().values()),
        total_components=len(check_sensor_module_availability()),
        hydra_available=HYDRA_AVAILABLE,
        performance_monitoring=PERFORMANCE_MONITORING_AVAILABLE
    )