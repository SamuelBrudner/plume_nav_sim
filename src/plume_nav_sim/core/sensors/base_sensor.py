"""
Enhanced sensor infrastructure providing BaseSensor abstract base class with enterprise-grade features.

This module provides shared sensor infrastructure following the pattern established in core/controllers.py,
implementing common functionality including logging integration, performance monitoring, configuration
management, error handling, vectorized operations support, and extensibility hooks that all concrete 
sensor implementations inherit to ensure consistent behavior and performance characteristics.

Key enhancements for modular plume navigation architecture:
- Protocol-based sensor abstraction supporting flexible sensing modalities (Binary, Concentration, Gradient)
- PlumeModel integration for standardized environmental sampling through concentration_at() interface
- WindField awareness for realistic sensor response modeling in dynamic environmental conditions
- Vectorized operations supporting 100+ concurrent agents with linear memory scaling guarantees
- Performance monitoring ensuring sub-0.1ms sensor operation latency requirements per agent
- Structured logging with correlation IDs and comprehensive performance metrics collection

The BaseSensor abstract class ensures consistent behavior across sensor implementations while providing
extensibility hooks for custom noise modeling, calibration drift simulation, and advanced sensing
strategies required for realistic navigation research scenarios.

Performance Requirements:
- Sensor operations: <0.1ms per agent per sensor for minimal overhead
- Batch processing: <1ms for 100 agents with multiple sensors
- Memory efficiency: <10KB per sensor for historical data with configurable limits
- Vectorized sampling: Linear scaling with agent count through optimized NumPy operations

Examples:
    Custom sensor implementation with shared infrastructure:
        >>> class CustomOdorSensor(BaseSensor):
        ...     def __init__(self, threshold=0.1, **kwargs):
        ...         super().__init__(sensor_type="custom_odor", **kwargs)
        ...         self._threshold = threshold
        ...     
        ...     def detect(self, plume_state, positions):
        ...         # Leverage shared infrastructure for performance monitoring
        ...         return self._execute_with_monitoring(
        ...             self._detect_impl, plume_state, positions
        ...         )
        ...     
        ...     def _detect_impl(self, plume_state, positions):
        ...         concentrations = self._sample_concentrations(plume_state, positions)
        ...         return concentrations > self._threshold
        
    Configuration-driven sensor instantiation:
        >>> from omegaconf import DictConfig
        >>> sensor_config = DictConfig({
        ...     "sensor_type": "concentration",
        ...     "dynamic_range": (0.0, 1.0),
        ...     "enable_logging": True,
        ...     "enable_performance_monitoring": True
        ... })
        >>> sensor = create_sensor_from_config(sensor_config)
        
    Multi-agent vectorized sensing:
        >>> positions = np.array([[10, 20], [15, 25], [20, 30]])  # 3 agents
        >>> readings = sensor.measure(plume_state, positions)
        >>> assert readings.shape == (3,)  # One reading per agent

Notes:
    All sensor implementations must inherit from BaseSensor and implement the SensorProtocol
    interface methods. The shared infrastructure handles performance monitoring, logging,
    error handling, and configuration management automatically.
    
    Sensor implementations integrate with PlumeModel components through standardized
    concentration_at() method calls, enabling seamless switching between different
    environmental modeling approaches (Gaussian, Turbulent, Video-based).
"""

import contextlib
import time
import warnings
import uuid
from typing import Optional, Union, Any, Tuple, List, Dict, TypeVar, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

# Core protocol imports
from plume_nav_sim.protocols.sensor import SensorProtocol

# Hydra integration for configuration management
try:
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    HydraConfig = None
    DictConfig = dict
    OmegaConf = None

# Loguru integration for enhanced logging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Configuration schemas - handle case where they don't exist yet
try:
    from ...config.schemas import SensorConfig, BinarySensorConfig, ConcentrationSensorConfig
    SCHEMAS_AVAILABLE = True
except ImportError:
    # These will be created by other agents - use minimal fallback types
    SensorConfig = Dict[str, Any]
    BinarySensorConfig = Dict[str, Any] 
    ConcentrationSensorConfig = Dict[str, Any]
    SCHEMAS_AVAILABLE = False

# PSUtil for memory monitoring (optional dependency)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

# Type variable for sensor types
SensorType = TypeVar('SensorType', bound='BaseSensor')


@dataclass
class SensorParams:
    """
    Type-safe parameters for sensor configuration and runtime updates.
    
    This dataclass provides stronger type checking than kwargs-based configuration
    and integrates with Hydra's structured configuration system for validation.
    Enhanced for modular sensor architecture with support for noise modeling,
    calibration parameters, and performance optimization settings.
    
    Attributes:
        sensor_type: Type identifier for sensor implementation
        dynamic_range: Measurement range as (min, max) tuple for concentration sensors
        threshold: Detection threshold for binary sensors
        spatial_resolution: Finite difference step size for gradient sensors as (dx, dy)
        temporal_filtering: Time constant for response dynamics simulation
        noise_parameters: Dict containing false positive/negative rates, measurement noise
        calibration_drift: Parameters for sensor calibration drift simulation
        enable_history: Whether to maintain temporal observation history
        history_length: Maximum number of historical readings to store
        performance_monitoring: Enable detailed performance metrics collection
        vectorized_ops: Enable vectorized operations for multi-agent scenarios
    
    Examples:
        Binary sensor configuration:
            >>> params = SensorParams(
            ...     sensor_type="binary",
            ...     threshold=0.1,
            ...     noise_parameters={"false_positive_rate": 0.02, "false_negative_rate": 0.01}
            ... )
            >>> sensor.configure(**params.to_kwargs())
            
        Concentration sensor with calibration drift:
            >>> params = SensorParams(
            ...     sensor_type="concentration",
            ...     dynamic_range=(0.0, 1.0),
            ...     calibration_drift={"drift_rate": 0.001, "recalibration_interval": 1000},
            ...     temporal_filtering=5.0
            ... )
            >>> sensor.configure(**params.to_kwargs())
    """
    sensor_type: Optional[str] = None
    dynamic_range: Optional[Tuple[float, float]] = None
    threshold: Optional[float] = None
    spatial_resolution: Optional[Tuple[float, float]] = None
    temporal_filtering: Optional[float] = None
    noise_parameters: Optional[Dict[str, float]] = None
    calibration_drift: Optional[Dict[str, Any]] = None
    enable_history: bool = False
    history_length: int = 100
    performance_monitoring: bool = True
    vectorized_ops: bool = True
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_kwargs(self) -> Dict[str, Any]:
        """Convert dataclass to kwargs dictionary for sensor configuration methods."""
        return {k: v for k, v in self.__dict__.items() if v is not None and v != {} and v != []}


class BaseSensor(ABC):
    """
    Abstract base sensor class with shared infrastructure for enterprise-grade sensing implementations.
    
    Provides common infrastructure for performance monitoring, extensibility hooks,
    configuration management, and structured logging with comprehensive context information. 
    This base class ensures consistent behavior across sensor implementations while enabling
    flexible sensing modalities for diverse research scenarios.
    
    Enhanced Features:
    - PlumeModel integration through standardized concentration_at() interface
    - WindField awareness for realistic environmental response modeling
    - Vectorized operations for efficient multi-agent sensing with linear scaling
    - Performance monitoring with sub-0.1ms operation latency guarantees
    - Structured logging with sensor-specific context and correlation IDs
    - Extensible architecture for custom noise modeling and calibration drift
    - Memory-efficient historical data management with configurable retention
    - Configuration-driven instantiation through Hydra integration
    
    Performance Requirements:
    - Sensor operations: <0.1ms per agent for minimal sensing overhead
    - Memory usage: <10KB per sensor for historical data management
    - Vectorized processing: Linear scaling with agent count up to 100+
    - Configuration updates: <1ms for runtime parameter changes
    
    Examples:
        Basic sensor initialization:
            >>> sensor = CustomSensor(
            ...     sensor_type="custom",
            ...     enable_logging=True,
            ...     enable_performance_monitoring=True
            ... )
            
        Performance monitoring integration:
            >>> readings = sensor.measure(plume_state, positions)
            >>> metrics = sensor.get_performance_metrics()
            >>> assert metrics['operation_time_mean_ms'] < 0.1
            
        Configuration-driven instantiation:
            >>> config = {"sensor_type": "binary", "threshold": 0.05}
            >>> sensor = create_sensor_from_config(config)
    """
    
    def __init__(
        self,
        sensor_type: str = "base",
        sensor_id: Optional[str] = None,
        enable_logging: bool = True,
        enable_performance_monitoring: bool = True,
        enable_history: bool = False,
        history_length: int = 100,
        vectorized_ops: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize base sensor with enhanced monitoring and configuration capabilities.
        
        Args:
            sensor_type: Type identifier for sensor implementation
            sensor_id: Unique sensor identifier for correlation and debugging
            enable_logging: Enable comprehensive logging integration
            enable_performance_monitoring: Enable detailed performance metrics collection
            enable_history: Whether to maintain temporal observation history
            history_length: Maximum number of historical readings to store
            vectorized_ops: Enable vectorized operations for multi-agent scenarios
            **kwargs: Additional sensor-specific configuration options
                
        Raises:
            ValueError: If sensor configuration parameters are invalid
            TypeError: If parameter types are incorrect
        """
        self._sensor_type = sensor_type
        self._sensor_id = sensor_id or f"{sensor_type}_{uuid.uuid4().hex[:8]}"
        self._enable_logging = enable_logging
        self._enable_performance_monitoring = enable_performance_monitoring
        self._enable_history = enable_history
        self._history_length = history_length
        self._vectorized_ops = vectorized_ops
        
        # Performance metrics tracking
        self._performance_metrics = {
            'operation_times': [],
            'total_operations': 0,
            'detection_count': 0,
            'measurement_count': 0,
            'gradient_computation_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0
        }
        
        # Historical data storage (optional)
        self._observation_history = [] if enable_history else None
        self._position_history = [] if enable_history else None
        self._timestamp_history = [] if enable_history else None
        
        # Sensor-specific configuration storage
        self._sensor_config = kwargs.copy()
        
        # Memory monitoring for large-scale scenarios
        if PSUTIL_AVAILABLE and enable_performance_monitoring:
            self._monitor_memory = True
            self._base_memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        else:
            self._monitor_memory = False
            self._base_memory_usage = 0
        
        # Setup structured logging with sensor context binding
        if self._enable_logging and LOGURU_AVAILABLE:
            self._logger = logger.bind(
                sensor_type=self._sensor_type,
                sensor_id=self._sensor_id,
                performance_monitoring=self._enable_performance_monitoring,
                vectorized_ops=self._vectorized_ops,
                history_enabled=self._enable_history
            )
            
            # Add Hydra context if available
            if HYDRA_AVAILABLE:
                try:
                    hydra_cfg = HydraConfig.get()
                    self._logger = self._logger.bind(
                        hydra_job_name=hydra_cfg.job.name,
                        hydra_output_dir=hydra_cfg.runtime.output_dir
                    )
                except Exception:
                    # Hydra context not available, continue without it
                    pass
        else:
            self._logger = None
        
        # Log initialization with sensor context
        if self._logger:
            self._logger.info(
                f"{self._sensor_type.title()}Sensor initialized with enhanced features",
                sensor_config=self._sensor_config,
                performance_monitoring=self._enable_performance_monitoring,
                history_enabled=self._enable_history,
                vectorized_ops=self._vectorized_ops,
                memory_monitoring=self._monitor_memory
            )
    
    # Abstract methods that must be implemented by concrete sensor classes
    
    @abstractmethod
    def detect(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Perform binary detection at specified agent positions (BinarySensor implementation).
        
        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Boolean detection results with shape (n_agents,) or scalar for single agent
            
        Notes:
            Concrete implementations must provide detection logic appropriate for their
            sensing modality. The base class handles performance monitoring, logging,
            and error handling automatically.
            
        Performance:
            Must execute in <0.1ms per agent for minimal sensing overhead.
        """
        pass
    
    @abstractmethod
    def measure(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Perform quantitative measurements at specified agent positions (ConcentrationSensor).
        
        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Quantitative measurement values with shape (n_agents,) or scalar for single agent
            
        Notes:
            Concrete implementations must provide measurement logic with appropriate
            dynamic range, resolution, and noise characteristics. The base class
            handles performance monitoring and data validation.
            
        Performance:
            Must execute in <0.1ms per agent for minimal sensing overhead.
        """
        pass
    
    @abstractmethod
    def compute_gradient(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Compute spatial gradients at specified agent positions (GradientSensor implementation).
        
        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Gradient vectors with shape (n_agents, 2) or (2,) for single agent
            
        Notes:
            Concrete implementations must provide gradient computation using appropriate
            finite difference methods and spatial resolution. The base class handles
            performance monitoring and numerical stability checks.
            
        Performance:
            Must execute in <0.2ms per agent due to multi-point sampling requirements.
        """
        pass
    
    # Shared infrastructure methods
    
    def configure(self, **kwargs: Any) -> None:
        """
        Update sensor configuration parameters during runtime with validation.
        
        Args:
            **kwargs: Sensor-specific configuration parameters including:
                - threshold: Detection threshold for binary sensors
                - dynamic_range: Measurement range for concentration sensors
                - spatial_resolution: Finite difference step size for gradient sensors
                - noise_parameters: False positive/negative rates, measurement noise
                - temporal_filtering: Response time constants and history length
                - calibration_drift: Drift simulation parameters
                
        Notes:
            Configuration updates apply immediately to subsequent sensor operations.
            Parameter validation ensures physical consistency and performance requirements.
            Temporal parameters may trigger reset of internal state buffers.
            
        Raises:
            ValueError: If configuration parameters are invalid or incompatible
            TypeError: If parameter types are incorrect
            
        Examples:
            Update binary sensor threshold:
                >>> sensor.configure(threshold=0.05, noise_parameters={"false_positive_rate": 0.01})
                
            Adjust concentration sensor range:
                >>> sensor.configure(dynamic_range=(0, 2.0), temporal_filtering=3.0)
                
            Configure gradient sensor resolution:
                >>> sensor.configure(spatial_resolution=(0.2, 0.2), vectorized_ops=True)
        """
        start_time = time.perf_counter() if self._enable_performance_monitoring else None
        
        try:
            # Validate basic parameter types
            if 'threshold' in kwargs and not isinstance(kwargs['threshold'], (int, float)):
                raise TypeError(f"threshold must be numeric, got {type(kwargs['threshold'])}")
            
            if 'dynamic_range' in kwargs:
                range_val = kwargs['dynamic_range']
                if not isinstance(range_val, (tuple, list)) or len(range_val) != 2:
                    raise ValueError(f"dynamic_range must be (min, max) tuple, got {range_val}")
                if range_val[0] >= range_val[1]:
                    raise ValueError(f"dynamic_range min ({range_val[0]}) must be < max ({range_val[1]})")
            
            if 'spatial_resolution' in kwargs:
                res_val = kwargs['spatial_resolution']
                if not isinstance(res_val, (tuple, list)) or len(res_val) != 2:
                    raise ValueError(f"spatial_resolution must be (dx, dy) tuple, got {res_val}")
                if any(r <= 0 for r in res_val):
                    raise ValueError(f"spatial_resolution values must be positive, got {res_val}")
            
            if 'history_length' in kwargs:
                length = kwargs['history_length']
                if not isinstance(length, int) or length <= 0:
                    raise ValueError(f"history_length must be positive integer, got {length}")
                
                # Update history length and potentially truncate existing data
                if self._enable_history and length != self._history_length:
                    self._history_length = length
                    if self._observation_history and len(self._observation_history) > length:
                        self._observation_history = self._observation_history[-length:]
                        self._position_history = self._position_history[-length:]
                        self._timestamp_history = self._timestamp_history[-length:]
            
            # Handle history enable/disable
            if 'enable_history' in kwargs:
                enable_hist = kwargs['enable_history']
                if enable_hist and not self._enable_history:
                    # Enable history tracking
                    self._enable_history = True
                    self._observation_history = []
                    self._position_history = []
                    self._timestamp_history = []
                elif not enable_hist and self._enable_history:
                    # Disable history tracking and free memory
                    self._enable_history = False
                    self._observation_history = None
                    self._position_history = None
                    self._timestamp_history = None
            
            # Update sensor configuration
            self._sensor_config.update(kwargs)
            
            # Call sensor-specific configuration update
            self._configure_sensor_specific(**kwargs)
            
            # Log configuration update
            if self._logger:
                config_time = (time.perf_counter() - start_time) * 1000 if start_time else 0
                self._logger.info(
                    "Sensor configuration updated",
                    updated_params=list(kwargs.keys()),
                    config_time_ms=config_time,
                    current_config=self._sensor_config
                )
                
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Sensor configuration failed: {str(e)}",
                    error_type=type(e).__name__,
                    attempted_params=kwargs
                )
            raise
    
    def _configure_sensor_specific(self, **kwargs: Any) -> None:
        """
        Override this method in concrete implementations for sensor-specific configuration.
        
        Args:
            **kwargs: Configuration parameters specific to the sensor implementation
            
        Notes:
            Default implementation is a no-op. Concrete sensor classes should override
            this method to handle their specific configuration parameters.
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and optimization.
        
        Returns:
            Dict[str, Any]: Dictionary containing detailed performance statistics including:
                - operation_time_mean_ms: Average operation time in milliseconds
                - operation_time_max_ms: Maximum operation time in milliseconds
                - total_operations: Total number of sensor operations performed
                - operation_type_counts: Breakdown by detection/measurement/gradient operations
                - cache_hit_rate: Cache efficiency metrics
                - error_rate: Fraction of operations resulting in errors
                - memory_usage_mb: Current memory usage (if monitoring enabled)
                
        Examples:
            Performance monitoring and validation:
                >>> metrics = sensor.get_performance_metrics()
                >>> assert metrics['operation_time_mean_ms'] < 0.1
                >>> print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
                
            Performance optimization decision-making:
                >>> if metrics['operation_time_max_ms'] > 1.0:
                ...     sensor.configure(vectorized_ops=True)
        """
        if not self._enable_performance_monitoring:
            return {}
        
        metrics = {
            'sensor_type': self._sensor_type,
            'sensor_id': self._sensor_id,
            'total_operations': self._performance_metrics['total_operations'],
            'vectorized_ops_enabled': self._vectorized_ops,
            'history_enabled': self._enable_history,
            'memory_monitoring_enabled': self._monitor_memory
        }
        
        # Operation time statistics
        if self._performance_metrics['operation_times']:
            op_times = np.array(self._performance_metrics['operation_times'])
            metrics.update({
                'operation_time_mean_ms': float(np.mean(op_times)),
                'operation_time_std_ms': float(np.std(op_times)),
                'operation_time_max_ms': float(np.max(op_times)),
                'operation_time_p95_ms': float(np.percentile(op_times, 95)),
                'performance_violations': int(np.sum(op_times > 0.1))  # >0.1ms threshold
            })
        
        # Operation type breakdown
        metrics.update({
            'detection_count': self._performance_metrics['detection_count'],
            'measurement_count': self._performance_metrics['measurement_count'],
            'gradient_computation_count': self._performance_metrics['gradient_computation_count'],
            'error_count': self._performance_metrics['error_count']
        })
        
        # Cache statistics
        total_cache_ops = (self._performance_metrics['cache_hits'] + 
                          self._performance_metrics['cache_misses'])
        if total_cache_ops > 0:
            metrics.update({
                'cache_hit_rate': self._performance_metrics['cache_hits'] / total_cache_ops,
                'cache_total_operations': total_cache_ops,
                'cache_misses': self._performance_metrics['cache_misses']
            })
        
        # Error rate calculation
        if self._performance_metrics['total_operations'] > 0:
            metrics['error_rate'] = (self._performance_metrics['error_count'] / 
                                   self._performance_metrics['total_operations'])
        
        # Memory usage monitoring
        if self._monitor_memory and PSUTIL_AVAILABLE:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_growth = current_memory - self._base_memory_usage
            metrics.update({
                'memory_usage_mb': current_memory,
                'memory_growth_mb': memory_growth,
                'memory_per_operation_kb': (memory_growth * 1024 / 
                                          max(self._performance_metrics['total_operations'], 1))
            })
        
        # History statistics
        if self._enable_history and self._observation_history:
            metrics.update({
                'history_length': len(self._observation_history),
                'history_capacity': self._history_length,
                'history_utilization': len(self._observation_history) / self._history_length
            })
        
        return metrics
    
    def get_observation_history(self, count: Optional[int] = None) -> Dict[str, List[Any]]:
        """
        Retrieve historical observation data for memory-based navigation strategies.
        
        Args:
            count: Number of recent observations to return (default: all available)
            
        Returns:
            Dict[str, List[Any]]: Dictionary containing historical data with keys:
                - observations: List of historical sensor readings
                - positions: List of agent positions for each observation
                - timestamps: List of timestamps for each observation
                
        Notes:
            Returns empty dict if history is disabled. Historical data is maintained
            as a circular buffer with configurable capacity for memory efficiency.
            
        Examples:
            Retrieve recent observation history:
                >>> history = sensor.get_observation_history(count=10)
                >>> recent_concentrations = history['observations'][-5:]
                
            Full history for trajectory analysis:
                >>> full_history = sensor.get_observation_history()
                >>> trajectory = history['positions']
        """
        if not self._enable_history or not self._observation_history:
            return {'observations': [], 'positions': [], 'timestamps': []}
        
        if count is None:
            return {
                'observations': self._observation_history.copy(),
                'positions': self._position_history.copy(),
                'timestamps': self._timestamp_history.copy()
            }
        else:
            count = min(count, len(self._observation_history))
            return {
                'observations': self._observation_history[-count:],
                'positions': self._position_history[-count:],
                'timestamps': self._timestamp_history[-count:]
            }
    
    def reset_history(self) -> None:
        """
        Clear all historical observation data to reset sensor memory.
        
        Notes:
            This operation frees memory used by historical data storage and
            resets the sensor to a clean state. Useful for episode boundaries
            in RL training or when switching between different experimental conditions.
            
        Examples:
            Reset between episodes:
                >>> sensor.reset_history()
                >>> assert len(sensor.get_observation_history()['observations']) == 0
        """
        if self._enable_history:
            self._observation_history = []
            self._position_history = []
            self._timestamp_history = []
            
            if self._logger:
                self._logger.debug("Sensor observation history reset")
    
    def reset(self, **kwargs: Any) -> None:
        """
        Reset sensor to initial state, clearing history and performance metrics.
        
        Args:
            **kwargs: Optional parameters for sensor-specific reset behavior.
                Common options include:
                - clear_history: Whether to reset observation history (default: True)
                - clear_metrics: Whether to reset performance metrics (default: True)
                - reset_config: Whether to reset sensor configuration (default: False)
                
        Notes:
            This method provides comprehensive sensor reset functionality by clearing
            both observation history and performance metrics. Sensor configuration
            is preserved unless explicitly requested via reset_config parameter.
            
            Subclasses can override this method to implement sensor-specific reset
            behavior while calling super().reset() for standard reset operations.
            
        Examples:
            Complete sensor reset:
                >>> sensor.reset()
                
            Reset only performance metrics:
                >>> sensor.reset(clear_history=False, clear_metrics=True)
        """
        clear_history = kwargs.get('clear_history', True)
        clear_metrics = kwargs.get('clear_metrics', True)
        
        if clear_history:
            self.reset_history()
            
        if clear_metrics:
            self.reset_performance_metrics()
            
        if self._logger:
            self._logger.debug(f"Sensor {self._sensor_id} reset completed")
    
    def reset_performance_metrics(self) -> None:
        """
        Reset all performance metrics to initial state for clean measurement periods.
        
        Notes:
            Useful for measuring performance over specific time windows or
            experimental conditions. Does not affect sensor configuration or
            observation history.
            
        Examples:
            Measure performance for specific experiment:
                >>> sensor.reset_performance_metrics()
                >>> # ... run experiment ...
                >>> metrics = sensor.get_performance_metrics()
        """
        if self._enable_performance_monitoring:
            self._performance_metrics = {
                'operation_times': [],
                'total_operations': 0,
                'detection_count': 0,
                'measurement_count': 0,
                'gradient_computation_count': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'error_count': 0
            }

            if self._logger:
                self._logger.debug("Sensor performance metrics reset")

    def get_observation_space_info(self) -> Dict[str, Any]:
        """Return generic observation space information for the sensor.

        Returns
        -------
        Dict[str, Any]
            Dictionary with ``shape`` and ``dtype`` describing the sensor's
            observation output. Subclasses may override this for custom outputs.
        """
        return {"shape": (1,), "dtype": np.float64}

    # Protected helper methods for use by concrete implementations
    
    def _execute_with_monitoring(
        self, 
        operation: Callable[..., Any], 
        operation_type: str,
        *args: Any, 
        **kwargs: Any
    ) -> Any:
        """
        Execute sensor operation with automatic performance monitoring and error handling.
        
        Args:
            operation: The sensor operation function to execute
            operation_type: Type of operation for metrics tracking ("detect", "measure", "gradient")
            *args: Arguments to pass to the operation function
            **kwargs: Keyword arguments to pass to the operation function
            
        Returns:
            Any: Result from the operation function
            
        Notes:
            This method provides centralized performance monitoring, error handling,
            and logging for all sensor operations. Concrete implementations should
            use this method to wrap their core sensing logic.
            
        Raises:
            Exception: Re-raises any exception from the operation after logging
        """
        start_time = time.perf_counter() if self._enable_performance_monitoring else None
        
        try:
            # Execute the sensor operation
            result = operation(*args, **kwargs)
            
            # Update performance metrics on success
            if self._enable_performance_monitoring:
                operation_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['operation_times'].append(operation_time)
                self._performance_metrics['total_operations'] += 1
                
                # Update operation-specific counters
                if operation_type == "detect":
                    self._performance_metrics['detection_count'] += 1
                elif operation_type == "measure":
                    self._performance_metrics['measurement_count'] += 1
                elif operation_type == "gradient":
                    self._performance_metrics['gradient_computation_count'] += 1
                
                # Log performance warning if operation exceeds threshold
                if operation_time > 0.1 and self._logger:
                    self._logger.warning(
                        f"Sensor operation exceeded 0.1ms threshold",
                        operation_type=operation_type,
                        operation_time_ms=operation_time,
                        sensor_type=self._sensor_type,
                        performance_degradation=True
                    )
                
                # Log periodic performance summary
                if (self._performance_metrics['total_operations'] % 1000 == 0 and 
                    self._logger and len(self._performance_metrics['operation_times']) >= 100):
                    recent_times = self._performance_metrics['operation_times'][-100:]
                    avg_time = np.mean(recent_times)
                    self._logger.debug(
                        "Sensor performance summary",
                        total_operations=self._performance_metrics['total_operations'],
                        avg_operation_time_ms=avg_time,
                        operation_type=operation_type,
                        sensor_type=self._sensor_type
                    )
            
            return result
            
        except Exception as e:
            # Update error metrics
            if self._enable_performance_monitoring:
                self._performance_metrics['error_count'] += 1
                self._performance_metrics['total_operations'] += 1
            
            # Log error with context
            if self._logger:
                self._logger.error(
                    f"Sensor operation failed: {str(e)}",
                    operation_type=operation_type,
                    error_type=type(e).__name__,
                    sensor_type=self._sensor_type,
                    sensor_id=self._sensor_id
                )
            
            # Re-raise the exception
            raise
    
    def _sample_concentrations(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Sample odor concentrations from plume state at specified positions.
        
        Args:
            plume_state: Current plume model state implementing concentration_at() method
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Concentration values at specified positions
            
        Notes:
            This method provides a standardized interface for sampling concentrations
            from different plume model implementations (Gaussian, Turbulent, Video-based).
            Handles both single and multi-agent position arrays automatically.
            
        Raises:
            AttributeError: If plume_state doesn't implement concentration_at() method
            ValueError: If positions array has invalid shape
        """
        # Validate positions array
        positions = np.asarray(positions)
        if positions.ndim == 1 and positions.shape[0] == 2:
            # Single agent position - reshape to (1, 2)
            positions = positions.reshape(1, 2)
        elif positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"positions must have shape (n_agents, 2) or (2,), got {positions.shape}")
        
        # Sample concentrations from plume model
        if hasattr(plume_state, 'concentration_at'):
            concentrations = plume_state.concentration_at(positions)
        else:
            # Fallback for legacy plume interfaces
            if hasattr(plume_state, 'current_frame'):
                # Video-based plume
                concentrations = self._sample_from_video_frame(plume_state.current_frame, positions)
            else:
                # Direct array access
                concentrations = self._sample_from_array(plume_state, positions)
        
        # Ensure output is numpy array
        concentrations = np.asarray(concentrations)
        
        # Validate and clean concentration values
        if np.any(np.isnan(concentrations)) or np.any(np.isinf(concentrations)):
            if self._logger:
                invalid_count = np.sum(np.isnan(concentrations) | np.isinf(concentrations))
                self._logger.warning(
                    "Invalid concentration values detected",
                    invalid_count=invalid_count,
                    total_samples=concentrations.size,
                    applying_cleanup=True
                )
            concentrations = np.nan_to_num(concentrations, nan=0.0, posinf=1.0, neginf=0.0)
        
        return concentrations
    
    def _sample_from_video_frame(self, frame: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Sample concentrations from video frame data at specified positions.
        
        Args:
            frame: Video frame array with shape (height, width) or (height, width, channels)
            positions: Agent positions array with shape (n_agents, 2)
            
        Returns:
            np.ndarray: Sampled concentration values
        """
        if not hasattr(frame, 'shape') or len(frame.shape) < 2:
            return np.zeros(len(positions))
        
        height, width = frame.shape[:2]
        num_positions = len(positions)
        
        # Convert positions to pixel coordinates
        x_pos = np.clip(np.floor(positions[:, 0]).astype(int), 0, width - 1)
        y_pos = np.clip(np.floor(positions[:, 1]).astype(int), 0, height - 1)
        
        # Sample values at positions
        if frame.ndim == 2:
            # Grayscale frame
            values = frame[y_pos, x_pos]
        else:
            # Color frame - use first channel
            values = frame[y_pos, x_pos, 0]
        
        # Normalize if uint8
        if hasattr(frame, 'dtype') and frame.dtype == np.uint8:
            values = values.astype(np.float64) / 255.0
        
        return values
    
    def _sample_from_array(self, array: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Sample concentrations from generic array data at specified positions.
        
        Args:
            array: Concentration array with shape (height, width)
            positions: Agent positions array with shape (n_agents, 2)
            
        Returns:
            np.ndarray: Sampled concentration values
        """
        if not hasattr(array, 'shape') or len(array.shape) < 2:
            return np.zeros(len(positions))
        
        height, width = array.shape[:2]
        
        # Convert positions to array indices
        x_indices = np.clip(np.floor(positions[:, 0]).astype(int), 0, width - 1)
        y_indices = np.clip(np.floor(positions[:, 1]).astype(int), 0, height - 1)
        
        # Sample values
        return array[y_indices, x_indices]
    
    def _add_to_history(self, observations: np.ndarray, positions: np.ndarray) -> None:
        """
        Add observations to historical data with automatic capacity management.
        
        Args:
            observations: Sensor observations to store
            positions: Agent positions corresponding to observations
            
        Notes:
            Maintains circular buffer with configurable capacity. Automatically
            removes oldest entries when capacity is exceeded.
        """
        if not self._enable_history:
            return
        
        timestamp = time.time()
        
        # Add new data
        self._observation_history.append(observations.copy() if hasattr(observations, 'copy') else observations)
        self._position_history.append(positions.copy() if hasattr(positions, 'copy') else positions)
        self._timestamp_history.append(timestamp)
        
        # Maintain capacity limit
        if len(self._observation_history) > self._history_length:
            self._observation_history.pop(0)
            self._position_history.pop(0)
            self._timestamp_history.pop(0)
    
    def _validate_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Validate and normalize position array format.
        
        Args:
            positions: Input position array
            
        Returns:
            np.ndarray: Validated position array with shape (n_agents, 2)
            
        Raises:
            ValueError: If position array has invalid shape or contains invalid values
        """
        positions = np.asarray(positions)
        
        # Handle single agent case
        if positions.ndim == 1 and positions.shape[0] == 2:
            positions = positions.reshape(1, 2)
        
        # Validate shape
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"positions must have shape (n_agents, 2) or (2,), got {positions.shape}")
        
        # Check for invalid values
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            raise ValueError("positions array contains NaN or infinite values")
        
        return positions


# Factory functions for configuration-driven sensor instantiation

def create_sensor_from_config(
    config: Union[DictConfig, Dict[str, Any], SensorConfig],
    sensor_id: Optional[str] = None,
    enable_logging: bool = True
) -> BaseSensor:
    """
    Create a sensor from configuration with automatic type detection and validation.
    
    Args:
        config: Configuration object containing sensor parameters including type
        sensor_id: Unique identifier for the sensor instance
        enable_logging: Enable comprehensive logging integration
        
    Returns:
        BaseSensor: Configured sensor instance implementing SensorProtocol
        
    Raises:
        ValueError: If configuration is invalid or sensor type is unknown
        TypeError: If configuration type is not supported
        ImportError: If sensor implementation is not available
        
    Examples:
        Binary sensor from configuration:
            >>> config = {
            ...     "sensor_type": "binary",
            ...     "threshold": 0.1,
            ...     "noise_parameters": {"false_positive_rate": 0.02}
            ... }
            >>> sensor = create_sensor_from_config(config)
            
        Concentration sensor with history:
            >>> config = {
            ...     "sensor_type": "concentration",
            ...     "dynamic_range": (0.0, 1.0),
            ...     "enable_history": True,
            ...     "history_length": 200
            ... }
            >>> sensor = create_sensor_from_config(config)
    """
    start_time = time.perf_counter() if enable_logging else None
    
    try:
        # Handle different configuration types
        if SCHEMAS_AVAILABLE and isinstance(config, SensorConfig):
            # Pydantic model - extract relevant parameters
            config_dict = config.model_dump(exclude_none=True)
        elif isinstance(config, DictConfig) and HYDRA_AVAILABLE:
            # Hydra OmegaConf configuration
            config_dict = OmegaConf.to_container(config, resolve=True)
        elif isinstance(config, dict):
            # Regular dictionary
            config_dict = config.copy()
        else:
            raise TypeError(
                f"Unsupported configuration type: {type(config)}. "
                f"Expected DictConfig, dict, or SensorConfig"
            )
        
        # Extract sensor type
        sensor_type = config_dict.get('sensor_type', config_dict.get('type', 'concentration'))
        
        # Import sensor implementations
        if sensor_type == 'binary':
            try:
                from .binary_sensor import BinarySensor
                return BinarySensor(sensor_id=sensor_id, enable_logging=enable_logging, **config_dict)
            except ImportError:
                raise ImportError(
                    "BinarySensor implementation not available. Ensure "
                    "plume_nav_sim.core.sensors.binary_sensor module has been created."
                )
        
        elif sensor_type == 'concentration':
            try:
                from .concentration_sensor import ConcentrationSensor
                return ConcentrationSensor(sensor_id=sensor_id, enable_logging=enable_logging, **config_dict)
            except ImportError:
                raise ImportError(
                    "ConcentrationSensor implementation not available. Ensure "
                    "plume_nav_sim.core.sensors.concentration_sensor module has been created."
                )
        
        elif sensor_type == 'gradient':
            try:
                from .gradient_sensor import GradientSensor
                return GradientSensor(sensor_id=sensor_id, enable_logging=enable_logging, **config_dict)
            except ImportError:
                raise ImportError(
                    "GradientSensor implementation not available. Ensure "
                    "plume_nav_sim.core.sensors.gradient_sensor module has been created."
                )
        
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        
    except Exception as e:
        if enable_logging and LOGURU_AVAILABLE:
            logger.error(
                f"Sensor creation failed: {str(e)}",
                error_type=type(e).__name__,
                config_type=type(config).__name__,
                sensor_id=sensor_id
            )
        raise


def validate_sensor_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate sensor configuration and return validation results.
    
    Args:
        config: Sensor configuration to validate
        
    Returns:
        Tuple[bool, List[str]]: Tuple of (is_valid, list_of_errors)
        
    Examples:
        Configuration validation:
            >>> config = {"sensor_type": "binary", "threshold": 0.1}
            >>> is_valid, errors = validate_sensor_config(config)
            >>> assert is_valid and len(errors) == 0
    """
    errors = []
    
    # Check for required sensor type
    if 'sensor_type' not in config and 'type' not in config:
        errors.append("Configuration must specify 'sensor_type' or 'type'")
    
    # Validate sensor type
    sensor_type = config.get('sensor_type', config.get('type'))
    valid_types = ['binary', 'concentration', 'gradient']
    if sensor_type not in valid_types:
        errors.append(f"sensor_type must be one of {valid_types}, got {sensor_type}")
    
    # Type-specific validation
    if sensor_type == 'binary':
        if 'threshold' in config and not isinstance(config['threshold'], (int, float)):
            errors.append("threshold must be numeric for binary sensors")
        if 'threshold' in config and config['threshold'] < 0:
            errors.append("threshold must be non-negative for binary sensors")
    
    elif sensor_type == 'concentration':
        if 'dynamic_range' in config:
            range_val = config['dynamic_range']
            if not isinstance(range_val, (tuple, list)) or len(range_val) != 2:
                errors.append("dynamic_range must be (min, max) tuple for concentration sensors")
            elif range_val[0] >= range_val[1]:
                errors.append("dynamic_range min must be < max for concentration sensors")
    
    elif sensor_type == 'gradient':
        if 'spatial_resolution' in config:
            res_val = config['spatial_resolution']
            if not isinstance(res_val, (tuple, list)) or len(res_val) != 2:
                errors.append("spatial_resolution must be (dx, dy) tuple for gradient sensors")
            elif any(r <= 0 for r in res_val):
                errors.append("spatial_resolution values must be positive for gradient sensors")
    
    # Common parameter validation
    if 'history_length' in config:
        length = config['history_length']
        if not isinstance(length, int) or length <= 0:
            errors.append("history_length must be positive integer")
    
    if 'enable_history' in config and not isinstance(config['enable_history'], bool):
        errors.append("enable_history must be boolean")
    
    if 'vectorized_ops' in config and not isinstance(config['vectorized_ops'], bool):
        errors.append("vectorized_ops must be boolean")
    
    return len(errors) == 0, errors


def get_sensor_info(sensor: BaseSensor) -> Dict[str, Any]:
    """
    Get comprehensive information about a sensor instance.
    
    Args:
        sensor: Sensor instance to analyze
        
    Returns:
        Dict[str, Any]: Dictionary containing sensor information and statistics
        
    Examples:
        Sensor analysis:
            >>> info = get_sensor_info(sensor)
            >>> print(f"Sensor type: {info['sensor_type']}")
            >>> print(f"Operations performed: {info['total_operations']}")
    """
    info = {
        'sensor_type': sensor._sensor_type,
        'sensor_id': sensor._sensor_id,
        'has_performance_monitoring': sensor._enable_performance_monitoring,
        'has_logging': sensor._enable_logging,
        'history_enabled': sensor._enable_history,
        'vectorized_ops_enabled': sensor._vectorized_ops,
        'memory_monitoring_enabled': sensor._monitor_memory
    }
    
    # Add configuration information
    info['sensor_config'] = sensor._sensor_config.copy()
    
    # Add performance metrics if available
    if sensor._enable_performance_monitoring:
        try:
            metrics = sensor.get_performance_metrics()
            info['performance_metrics'] = metrics
        except Exception:
            info['performance_metrics'] = "Error retrieving metrics"
    
    # Add history information
    if sensor._enable_history:
        info.update({
            'history_length': sensor._history_length,
            'current_history_size': len(sensor._observation_history) if sensor._observation_history else 0,
            'history_utilization': (len(sensor._observation_history) / sensor._history_length 
                                  if sensor._observation_history else 0.0)
        })
    
    return info


# Export public API
__all__ = [
    # Base sensor infrastructure
    "BaseSensor",
    "SensorParams",
    
    # Factory functions
    "create_sensor_from_config",
    
    # Utility functions
    "validate_sensor_config",
    "get_sensor_info",
]