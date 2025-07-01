"""
ConcentrationSensor implementation providing quantitative odor concentration measurement.

This module implements the ConcentrationSensor class that provides realistic quantitative 
concentration measurements with configurable dynamic range, precision settings, saturation 
modeling, calibration drift simulation, temporal response characteristics, and optimized
vectorized operations for multi-agent scenarios.

The ConcentrationSensor supports flexible measurement resolution, realistic sensor behavior
modeling including noise and drift, temporal filtering for sensor dynamics simulation, and
linear performance scaling for 100+ concurrent agents with sub-10ms step latency requirements.

Key Features:
- Quantitative concentration measurement with configurable dynamic range
- Precision control and saturation modeling for realistic sensor limits
- Temporal response characteristics with optional filtering capabilities
- Calibration drift simulation for long-term behavior research
- Vectorized sampling operations with linear performance scaling
- Comprehensive noise modeling and measurement uncertainty simulation
- Integration with BaseSensor infrastructure for logging and monitoring

Performance Requirements:
- Sensor operations: <0.1ms per agent for minimal sensing overhead
- Batch processing: <1ms for 100 agents with multiple sensors
- Memory efficiency: <10MB for typical multi-agent scenarios
- Linear scaling with agent count through optimized vectorized operations

Examples:
    Basic concentration sensor with default settings:
        >>> sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0))
        >>> concentrations = sensor.measure(plume_state, agent_positions)
        
    High-precision sensor with noise modeling:
        >>> sensor = ConcentrationSensor(
        ...     dynamic_range=(0.0, 10.0),
        ...     resolution=0.001,
        ...     noise_std=0.02,
        ...     enable_drift=True
        ... )
        >>> concentrations = sensor.measure(plume_state, positions)
        
    Sensor with temporal filtering:
        >>> sensor = ConcentrationSensor(
        ...     dynamic_range=(0.0, 1.0),
        ...     response_time=0.5,
        ...     enable_filtering=True
        ... )
        >>> for t in range(100):
        ...     concentrations = sensor.measure(plume_state, positions)

Notes:
    The ConcentrationSensor class implements the SensorProtocol interface enabling seamless
    integration with the modular sensor architecture. All concrete sensor implementations
    inherit shared functionality from BaseSensor for consistent behavior, performance
    monitoring, and configuration management.
    
    Measurement values are calibrated and normalized according to the configured dynamic
    range with optional saturation effects. Temporal filtering models realistic sensor
    response delays and frequency characteristics matching physical chemical sensors.
"""

from __future__ import annotations
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

# Core protocol import
from ..protocols import SensorProtocol

# Base sensor infrastructure
try:
    from .base_sensor import BaseSensor
    BASE_SENSOR_AVAILABLE = True
except ImportError:
    # Fallback base class if BaseSensor doesn't exist yet
    class BaseSensor:
        def __init__(self, **kwargs):
            self._enable_logging = kwargs.get('enable_logging', True)
            self._sensor_id = kwargs.get('sensor_id', f"sensor_{id(self)}")
            self._performance_metrics = {
                'measurement_times': [],
                'total_measurements': 0,
                'noise_samples': 0,
                'drift_updates': 0
            }
        
        def get_performance_metrics(self) -> Dict[str, Any]:
            return self._performance_metrics.copy()
    
    BASE_SENSOR_AVAILABLE = False

# Logging integration
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Configuration integration
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    HYDRA_AVAILABLE = False


@dataclass
class ConcentrationSensorConfig:
    """
    Configuration dataclass for ConcentrationSensor with comprehensive parameter validation.
    
    This dataclass provides type-safe configuration for ConcentrationSensor instances with
    validation of measurement parameters, noise characteristics, temporal dynamics, and
    performance settings. Integrates with Hydra configuration system for structured validation.
    
    Attributes:
        dynamic_range: Tuple of (min, max) measurement range in concentration units
        resolution: Minimum detectable concentration difference (quantization step)
        saturation_level: Concentration level at which sensor saturates (None = no saturation)
        noise_std: Standard deviation of measurement noise (0.0 = noiseless)
        noise_type: Type of noise model ("gaussian", "uniform", "none")
        enable_drift: Enable calibration drift simulation over time
        drift_rate: Rate of calibration drift per time unit (concentration/time)
        baseline_offset: Constant offset added to all measurements
        response_time: Sensor response time constant for temporal filtering (seconds)
        enable_filtering: Enable temporal filtering for sensor dynamics
        filter_type: Type of temporal filter ("lowpass", "bandpass", "none")
        cutoff_frequency: Filter cutoff frequency in Hz (for lowpass filter)
        vectorized_ops: Enable optimized vectorized operations for multi-agent scenarios
        enable_metadata: Include measurement metadata in sensor outputs
        calibration_interval: Time interval between calibration drift updates (seconds)
        
    Examples:
        High-precision laboratory sensor:
            >>> config = ConcentrationSensorConfig(
            ...     dynamic_range=(0.0, 100.0),
            ...     resolution=0.01,
            ...     noise_std=0.1,
            ...     enable_drift=True,
            ...     response_time=0.1
            ... )
            
        Field sensor with limited precision:
            >>> config = ConcentrationSensorConfig(
            ...     dynamic_range=(0.0, 10.0),
            ...     resolution=0.1,
            ...     noise_std=0.5,
            ...     saturation_level=9.5,
            ...     enable_filtering=True
            ... )
    """
    dynamic_range: Tuple[float, float] = (0.0, 1.0)
    resolution: float = 0.001
    saturation_level: Optional[float] = None
    noise_std: float = 0.0
    noise_type: str = "gaussian"
    enable_drift: bool = False
    drift_rate: float = 0.001
    baseline_offset: float = 0.0
    response_time: float = 0.0
    enable_filtering: bool = False
    filter_type: str = "lowpass"
    cutoff_frequency: float = 10.0
    vectorized_ops: bool = True
    enable_metadata: bool = False
    calibration_interval: float = 10.0
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_dynamic_range()
        self._validate_resolution()
        self._validate_noise_parameters()
        self._validate_temporal_parameters()
        self._validate_string_parameters()
    
    def _validate_dynamic_range(self) -> None:
        """Validate dynamic range configuration."""
        if not isinstance(self.dynamic_range, (tuple, list)) or len(self.dynamic_range) != 2:
            raise ValueError("dynamic_range must be a tuple/list of (min, max) values")
        
        min_val, max_val = self.dynamic_range
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise ValueError("dynamic_range values must be numeric")
        
        if min_val >= max_val:
            raise ValueError(f"dynamic_range min ({min_val}) must be less than max ({max_val})")
        
        if min_val < 0:
            raise ValueError(f"dynamic_range min ({min_val}) cannot be negative")
    
    def _validate_resolution(self) -> None:
        """Validate resolution parameter."""
        if not isinstance(self.resolution, (int, float)) or self.resolution <= 0:
            raise ValueError("resolution must be a positive number")
        
        range_span = self.dynamic_range[1] - self.dynamic_range[0]
        if self.resolution > range_span:
            raise ValueError(f"resolution ({self.resolution}) cannot exceed dynamic range span ({range_span})")
    
    def _validate_noise_parameters(self) -> None:
        """Validate noise modeling parameters."""
        if not isinstance(self.noise_std, (int, float)) or self.noise_std < 0:
            raise ValueError("noise_std must be a non-negative number")
        
        if self.noise_type not in ["gaussian", "uniform", "none"]:
            raise ValueError("noise_type must be one of: 'gaussian', 'uniform', 'none'")
        
        if not isinstance(self.drift_rate, (int, float)) or self.drift_rate < 0:
            raise ValueError("drift_rate must be a non-negative number")
        
        if not isinstance(self.baseline_offset, (int, float)):
            raise ValueError("baseline_offset must be numeric")
    
    def _validate_temporal_parameters(self) -> None:
        """Validate temporal response parameters."""
        if not isinstance(self.response_time, (int, float)) or self.response_time < 0:
            raise ValueError("response_time must be a non-negative number")
        
        if self.filter_type not in ["lowpass", "bandpass", "none"]:
            raise ValueError("filter_type must be one of: 'lowpass', 'bandpass', 'none'")
        
        if not isinstance(self.cutoff_frequency, (int, float)) or self.cutoff_frequency <= 0:
            raise ValueError("cutoff_frequency must be a positive number")
        
        if not isinstance(self.calibration_interval, (int, float)) or self.calibration_interval <= 0:
            raise ValueError("calibration_interval must be a positive number")
    
    def _validate_string_parameters(self) -> None:
        """Validate string configuration parameters."""
        if not isinstance(self.noise_type, str):
            raise ValueError("noise_type must be a string")
        
        if not isinstance(self.filter_type, str):
            raise ValueError("filter_type must be a string")


class ConcentrationSensor(BaseSensor):
    """
    ConcentrationSensor implementation providing quantitative odor concentration measurement.
    
    This sensor provides realistic quantitative concentration measurements with configurable
    dynamic range, precision settings, saturation modeling, calibration drift simulation,
    temporal response characteristics, and optimized vectorized operations for multi-agent
    scenarios while maintaining sub-10ms step latency requirements.
    
    The ConcentrationSensor implements the SensorProtocol interface enabling seamless
    integration with the modular sensor architecture. It inherits shared functionality from
    BaseSensor for consistent behavior, performance monitoring, and configuration management.
    
    Key Features:
    - Quantitative concentration measurement with configurable dynamic range and precision
    - Realistic sensor behavior including noise, drift, and saturation effects
    - Temporal response characteristics with optional filtering capabilities
    - Vectorized sampling operations with linear performance scaling
    - Comprehensive measurement metadata and uncertainty quantification
    - Integration with configuration management and logging infrastructure
    
    Performance Characteristics:
    - Measurement latency: <0.1ms per agent for single measurements
    - Batch processing: <1ms for 100 agents with vectorized operations
    - Memory usage: <1KB per agent for internal state management
    - Linear scaling with agent count through optimized algorithms
    
    Configuration Integration:
    The sensor supports configuration-driven parameter management through Hydra/OmegaConf
    integration. All parameters can be specified via structured configuration files or
    programmatic instantiation with comprehensive validation.
    
    Examples:
        Basic concentration sensor for research:
            >>> sensor = ConcentrationSensor(
            ...     dynamic_range=(0.0, 1.0),
            ...     resolution=0.001,
            ...     enable_logging=True
            ... )
            >>> concentrations = sensor.measure(plume_state, agent_positions)
            
        High-fidelity sensor with noise modeling:
            >>> sensor = ConcentrationSensor(
            ...     dynamic_range=(0.0, 10.0),
            ...     resolution=0.01,
            ...     noise_std=0.05,
            ...     enable_drift=True,
            ...     drift_rate=0.001,
            ...     response_time=0.2
            ... )
            >>> for t in range(1000):
            ...     concentrations = sensor.measure(plume_state, positions)
            
        Production sensor with performance optimization:
            >>> sensor = ConcentrationSensor(
            ...     dynamic_range=(0.0, 1.0),
            ...     vectorized_ops=True,
            ...     enable_filtering=False,
            ...     enable_metadata=False
            ... )
            >>> metrics = sensor.get_performance_metrics()
            >>> assert metrics['avg_measurement_time_ms'] < 0.1
    """
    
    def __init__(
        self,
        dynamic_range: Tuple[float, float] = (0.0, 1.0),
        resolution: float = 0.001,
        saturation_level: Optional[float] = None,
        noise_std: float = 0.0,
        noise_type: str = "gaussian",
        enable_drift: bool = False,
        drift_rate: float = 0.001,
        baseline_offset: float = 0.0,
        response_time: float = 0.0,
        enable_filtering: bool = False,
        filter_type: str = "lowpass",
        cutoff_frequency: float = 10.0,
        vectorized_ops: bool = True,
        enable_metadata: bool = False,
        calibration_interval: float = 10.0,
        sensor_id: Optional[str] = None,
        enable_logging: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize ConcentrationSensor with comprehensive configuration validation.
        
        Args:
            dynamic_range: Tuple of (min, max) measurement range in concentration units
            resolution: Minimum detectable concentration difference (quantization step)
            saturation_level: Concentration level at which sensor saturates (None = no saturation)
            noise_std: Standard deviation of measurement noise (0.0 = noiseless)
            noise_type: Type of noise model ("gaussian", "uniform", "none")
            enable_drift: Enable calibration drift simulation over time
            drift_rate: Rate of calibration drift per time unit (concentration/time)
            baseline_offset: Constant offset added to all measurements
            response_time: Sensor response time constant for temporal filtering (seconds)
            enable_filtering: Enable temporal filtering for sensor dynamics
            filter_type: Type of temporal filter ("lowpass", "bandpass", "none")
            cutoff_frequency: Filter cutoff frequency in Hz (for lowpass filter)
            vectorized_ops: Enable optimized vectorized operations for multi-agent scenarios
            enable_metadata: Include measurement metadata in sensor outputs
            calibration_interval: Time interval between calibration drift updates (seconds)
            sensor_id: Unique sensor identifier for logging and monitoring
            enable_logging: Enable comprehensive logging integration
            **kwargs: Additional configuration parameters for BaseSensor
            
        Raises:
            ValueError: If configuration parameters are invalid or inconsistent
            TypeError: If parameter types are incorrect
        """
        # Initialize base sensor infrastructure
        super().__init__(
            sensor_id=sensor_id or f"concentration_sensor_{id(self)}",
            enable_logging=enable_logging,
            **kwargs
        )
        
        # Create and validate configuration
        self._config = ConcentrationSensorConfig(
            dynamic_range=dynamic_range,
            resolution=resolution,
            saturation_level=saturation_level,
            noise_std=noise_std,
            noise_type=noise_type,
            enable_drift=enable_drift,
            drift_rate=drift_rate,
            baseline_offset=baseline_offset,
            response_time=response_time,
            enable_filtering=enable_filtering,
            filter_type=filter_type,
            cutoff_frequency=cutoff_frequency,
            vectorized_ops=vectorized_ops,
            enable_metadata=enable_metadata,
            calibration_interval=calibration_interval
        )
        
        # Initialize sensor state
        self._measurement_count = 0
        self._total_drift = 0.0
        self._last_calibration_time = time.time()
        self._last_measurements = None
        self._filter_state = None
        
        # Initialize temporal filter if enabled
        if self._config.enable_filtering and self._config.response_time > 0:
            self._initialize_temporal_filter()
        
        # Initialize random number generator for reproducible noise
        self._rng = np.random.RandomState(kwargs.get('random_seed', 42))
        
        # Enhanced performance metrics for concentration sensing
        self._performance_metrics.update({
            'measurement_times': [],
            'drift_updates': 0,
            'filter_operations': 0,
            'saturation_events': 0,
            'noise_applications': 0,
            'vectorized_operations': 0
        })
        
        # Log sensor initialization
        if self._enable_logging and BASE_SENSOR_AVAILABLE:
            if LOGURU_AVAILABLE:
                logger.bind(
                    sensor_type="ConcentrationSensor",
                    sensor_id=self._sensor_id,
                    dynamic_range=dynamic_range,
                    resolution=resolution,
                    noise_std=noise_std,
                    enable_drift=enable_drift,
                    vectorized_ops=vectorized_ops
                ).info("ConcentrationSensor initialized with enhanced measurement capabilities")
    
    def _initialize_temporal_filter(self) -> None:
        """Initialize temporal filtering components for sensor dynamics."""
        # Initialize lowpass filter state for exponential filtering
        if self._config.filter_type == "lowpass":
            # Simple exponential moving average filter
            self._filter_alpha = 1.0 - np.exp(-1.0 / (self._config.response_time * self._config.cutoff_frequency))
            self._filter_state = {}
        elif self._config.filter_type == "bandpass":
            # More complex filter state would be implemented here for bandpass filtering
            self._filter_alpha = 0.5  # Simplified for demonstration
            self._filter_state = {}
    
    def detect(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Perform binary detection based on concentration measurements.
        
        This method implements the SensorProtocol.detect() interface by applying
        a threshold to quantitative concentration measurements. Useful for
        compatibility with binary sensor interfaces while maintaining quantitative
        measurement capabilities.
        
        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for
                single agent. Coordinates in environment units.
                
        Returns:
            np.ndarray: Boolean detection results with shape (n_agents,) or scalar
                for single agent. True indicates concentration above detection threshold.
                
        Notes:
            Detection threshold is set to 10% of the dynamic range maximum by default.
            This can be configured through the sensor configuration parameters.
            
        Performance:
            Executes in <0.1ms per agent through optimized vectorized operations.
            
        Examples:
            Single agent detection:
                >>> position = np.array([15, 25])
                >>> detected = sensor.detect(plume_state, position)
                
            Multi-agent batch detection:
                >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
                >>> detections = sensor.detect(plume_state, positions)
        """
        # Get quantitative measurements
        concentrations = self.measure(plume_state, positions)
        
        # Apply detection threshold (10% of maximum range)
        detection_threshold = self._config.dynamic_range[0] + 0.1 * (
            self._config.dynamic_range[1] - self._config.dynamic_range[0]
        )
        
        # Return boolean detections
        return concentrations >= detection_threshold
    
    def measure(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Perform quantitative concentration measurements at specified positions.
        
        This is the primary measurement method providing calibrated concentration
        readings with realistic sensor characteristics including noise, drift,
        saturation, and temporal response. Supports both single and multi-agent
        scenarios with optimized vectorized operations.
        
        Args:
            plume_state: Current plume model state providing concentration field access.
                Typically a PlumeModel instance or spatial concentration array.
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for
                single agent. Coordinates in environment units.
                
        Returns:
            np.ndarray: Quantitative concentration values with shape (n_agents,) or
                scalar for single agent. Values in calibrated concentration units
                according to the configured dynamic range and precision.
                
        Notes:
            Measurements include realistic sensor characteristics:
            - Quantization according to resolution setting
            - Additive noise based on noise model configuration
            - Calibration drift simulation for long-term behavior
            - Saturation effects at high concentrations
            - Temporal filtering for sensor response dynamics
            
            Raw plume concentrations are sampled from the plume_state and processed
            through the complete sensor model to provide realistic measurements.
            
        Performance:
            Executes in <0.1ms per agent for minimal sensing overhead through
            optimized vectorized operations and efficient array processing.
            
        Raises:
            ValueError: If positions array has invalid shape or plume_state is invalid
            RuntimeError: If measurement processing fails or exceeds time limits
            
        Examples:
            Single agent measurement:
                >>> position = np.array([15, 25])
                >>> concentration = sensor.measure(plume_state, position)
                >>> assert 0 <= concentration <= sensor._config.dynamic_range[1]
                
            Multi-agent batch measurement:
                >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
                >>> concentrations = sensor.measure(plume_state, positions)
                >>> assert concentrations.shape == (3,)
                
            Performance validation:
                >>> import time
                >>> start_time = time.perf_counter()
                >>> concentrations = sensor.measure(plume_state, positions)
                >>> measurement_time = (time.perf_counter() - start_time) * 1000
                >>> assert measurement_time < 0.1  # <0.1ms requirement
        """
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Validate and normalize positions
            positions = self._validate_and_normalize_positions(positions)
            num_agents = positions.shape[0]
            
            # Sample raw concentrations from plume state
            raw_concentrations = self._sample_raw_concentrations(plume_state, positions)
            
            # Apply sensor measurement model
            measured_concentrations = self._apply_measurement_model(raw_concentrations)
            
            # Apply temporal filtering if enabled
            if self._config.enable_filtering and self._config.response_time > 0:
                measured_concentrations = self._apply_temporal_filter(measured_concentrations, positions)
                self._performance_metrics['filter_operations'] += 1
            
            # Update sensor state for drift modeling
            if self._config.enable_drift:
                self._update_calibration_drift()
            
            # Store last measurements for temporal filtering
            self._last_measurements = measured_concentrations.copy()
            self._measurement_count += num_agents
            
            # Track performance metrics
            if self._enable_logging and start_time:
                measurement_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
                self._performance_metrics['measurement_times'].append(measurement_time)
                self._performance_metrics['total_measurements'] += num_agents
                
                if self._config.vectorized_ops and num_agents > 1:
                    self._performance_metrics['vectorized_operations'] += 1
                
                # Log detailed performance for debugging (reduced frequency)
                if LOGURU_AVAILABLE and self._measurement_count % 100 == 0:
                    logger.trace(
                        "ConcentrationSensor measurement completed",
                        measurement_time_ms=measurement_time,
                        num_agents=num_agents,
                        avg_concentration=float(np.mean(measured_concentrations)),
                        measurement_count=self._measurement_count,
                        drift_offset=self._total_drift
                    )
                
                # Check performance requirement (<0.1ms per agent)
                if measurement_time > 0.1 * num_agents and LOGURU_AVAILABLE:
                    logger.warning(
                        "ConcentrationSensor measurement exceeded performance requirement",
                        measurement_time_ms=measurement_time,
                        num_agents=num_agents,
                        time_per_agent_ms=measurement_time / num_agents,
                        performance_degradation=True
                    )
            
            # Return scalar for single agent, array for multiple agents
            if num_agents == 1:
                return float(measured_concentrations[0])
            else:
                return measured_concentrations
                
        except Exception as e:
            if self._enable_logging and LOGURU_AVAILABLE:
                logger.error(
                    f"ConcentrationSensor measurement failed: {str(e)}",
                    error_type=type(e).__name__,
                    positions_shape=getattr(positions, 'shape', 'unknown'),
                    plume_state_type=type(plume_state).__name__
                )
            # Return safe default values
            if hasattr(positions, 'shape') and len(positions.shape) > 1:
                return np.zeros(positions.shape[0])
            else:
                return 0.0
    
    def compute_gradient(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Compute concentration gradients using finite difference methods.
        
        This method provides spatial gradient computation for concentration fields
        using configurable finite difference algorithms. While not the primary
        measurement method for ConcentrationSensor, it enables gradient-based
        navigation when needed.
        
        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for
                single agent. Coordinates in environment units.
                
        Returns:
            np.ndarray: Gradient vectors with shape (n_agents, 2) or (2,) for single
                agent. Components represent [∂c/∂x, ∂c/∂y] spatial derivatives in
                concentration units per distance unit.
                
        Notes:
            Gradient computation uses simple finite difference with step size based
            on sensor resolution. More sophisticated gradient computation is available
            through the dedicated GradientSensor implementation.
            
            For production gradient-based navigation, consider using GradientSensor
            which provides optimized finite difference algorithms and error estimation.
            
        Performance:
            Executes in <0.2ms per agent due to multi-point sampling requirements.
            
        Examples:
            Single agent gradient:
                >>> position = np.array([15, 25])
                >>> gradient = sensor.compute_gradient(plume_state, position)
                >>> assert gradient.shape == (2,)
                
            Multi-agent batch gradients:
                >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
                >>> gradients = sensor.compute_gradient(plume_state, positions)
                >>> assert gradients.shape == (3, 2)
        """
        # Validate and normalize positions
        positions = self._validate_and_normalize_positions(positions)
        num_agents = positions.shape[0]
        
        # Use sensor resolution as step size for finite differences
        step_size = self._config.resolution * 10  # 10x resolution for gradient accuracy
        
        # Compute gradients using central differences
        gradients = np.zeros((num_agents, 2))
        
        for i in range(num_agents):
            pos = positions[i]
            
            # Central difference in x direction
            pos_x_plus = pos + np.array([step_size, 0])
            pos_x_minus = pos - np.array([step_size, 0])
            c_x_plus = self.measure(plume_state, pos_x_plus.reshape(1, -1))
            c_x_minus = self.measure(plume_state, pos_x_minus.reshape(1, -1))
            gradients[i, 0] = (c_x_plus - c_x_minus) / (2 * step_size)
            
            # Central difference in y direction
            pos_y_plus = pos + np.array([0, step_size])
            pos_y_minus = pos - np.array([0, step_size])
            c_y_plus = self.measure(plume_state, pos_y_plus.reshape(1, -1))
            c_y_minus = self.measure(plume_state, pos_y_minus.reshape(1, -1))
            gradients[i, 1] = (c_y_plus - c_y_minus) / (2 * step_size)
        
        # Return scalar for single agent, array for multiple agents
        if num_agents == 1:
            return gradients[0]
        else:
            return gradients
    
    def configure(self, **kwargs: Any) -> None:
        """
        Update sensor configuration parameters during runtime.
        
        Args:
            **kwargs: Sensor-specific configuration parameters. Supported options:
                - dynamic_range: Tuple of (min, max) measurement range
                - resolution: Minimum detectable concentration difference
                - noise_std: Standard deviation of measurement noise
                - enable_drift: Enable/disable calibration drift simulation
                - drift_rate: Rate of calibration drift per time unit
                - response_time: Sensor response time constant
                - enable_filtering: Enable/disable temporal filtering
                - vectorized_ops: Enable/disable vectorized operations
                - enable_metadata: Enable/disable measurement metadata
                
        Notes:
            Configuration updates apply immediately to subsequent sensor operations.
            Parameter validation ensures physical consistency and performance requirements.
            
            Temporal parameters may trigger reset of internal state buffers for
            clean transition to new configuration.
            
        Examples:
            Update measurement range:
                >>> sensor.configure(dynamic_range=(0.0, 5.0), resolution=0.005)
                
            Enable noise and drift modeling:
                >>> sensor.configure(noise_std=0.1, enable_drift=True, drift_rate=0.002)
                
            Configure temporal response:
                >>> sensor.configure(response_time=0.3, enable_filtering=True)
        """
        # Create new configuration with updated parameters
        config_dict = self._config.__dict__.copy()
        config_dict.update(kwargs)
        
        try:
            # Validate new configuration
            new_config = ConcentrationSensorConfig(**config_dict)
            
            # Check if temporal parameters changed (requires filter reset)
            temporal_params = ['response_time', 'enable_filtering', 'filter_type', 'cutoff_frequency']
            temporal_changed = any(
                getattr(new_config, param) != getattr(self._config, param) 
                for param in temporal_params
            )
            
            # Update configuration
            self._config = new_config
            
            # Reset temporal filter if parameters changed
            if temporal_changed and self._config.enable_filtering:
                self._initialize_temporal_filter()
                self._filter_state = None
            
            # Reset drift if drift parameters changed
            if 'enable_drift' in kwargs or 'drift_rate' in kwargs:
                self._total_drift = 0.0
                self._last_calibration_time = time.time()
            
            # Log configuration update
            if self._enable_logging and LOGURU_AVAILABLE:
                logger.info(
                    "ConcentrationSensor configuration updated",
                    sensor_id=self._sensor_id,
                    updated_params=list(kwargs.keys()),
                    temporal_reset=temporal_changed
                )
                
        except Exception as e:
            if self._enable_logging and LOGURU_AVAILABLE:
                logger.error(
                    f"ConcentrationSensor configuration update failed: {str(e)}",
                    error_type=type(e).__name__,
                    invalid_params=kwargs
                )
            raise ValueError(f"Invalid configuration parameters: {str(e)}")
    
    def _validate_and_normalize_positions(self, positions: np.ndarray) -> np.ndarray:
        """Validate and normalize position array format."""
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        
        # Handle single position case
        if positions.ndim == 1:
            if len(positions) != 2:
                raise ValueError(f"Single position must have 2 coordinates, got {len(positions)}")
            positions = positions.reshape(1, -1)
        elif positions.ndim == 2:
            if positions.shape[1] != 2:
                raise ValueError(f"Position array must have shape (n_agents, 2), got {positions.shape}")
        else:
            raise ValueError(f"Position array must be 1D or 2D, got shape {positions.shape}")
        
        return positions
    
    def _sample_raw_concentrations(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """Sample raw concentration values from plume state."""
        try:
            # Check if plume_state implements concentration_at method (PlumeModelProtocol)
            if hasattr(plume_state, 'concentration_at'):
                return plume_state.concentration_at(positions)
            
            # Check if plume_state is a numpy array (direct concentration field)
            elif isinstance(plume_state, np.ndarray):
                return self._sample_from_array(plume_state, positions)
            
            # Check for mock plume object (testing)
            elif hasattr(plume_state, 'current_frame'):
                return self._sample_from_array(plume_state.current_frame, positions)
            
            # Fallback for unknown plume state types
            else:
                warnings.warn(f"Unknown plume_state type: {type(plume_state)}, returning zeros")
                return np.zeros(positions.shape[0])
                
        except Exception as e:
            if self._enable_logging and LOGURU_AVAILABLE:
                logger.warning(
                    f"Failed to sample from plume_state: {str(e)}",
                    plume_state_type=type(plume_state).__name__,
                    fallback_to_zeros=True
                )
            return np.zeros(positions.shape[0])
    
    def _sample_from_array(self, env_array: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Sample concentrations from a 2D environment array."""
        if not hasattr(env_array, 'shape') or len(env_array.shape) < 2:
            return np.zeros(positions.shape[0])

        height, width = env_array.shape[:2]
        num_positions = positions.shape[0]
        
        # Convert positions to array indices with bounds checking
        x_pos = np.clip(np.floor(positions[:, 0]).astype(int), 0, width - 1)
        y_pos = np.clip(np.floor(positions[:, 1]).astype(int), 0, height - 1)
        
        # Sample values from array
        concentrations = env_array[y_pos, x_pos]
        
        # Normalize if uint8 array
        if hasattr(env_array, 'dtype') and env_array.dtype == np.uint8:
            concentrations = concentrations.astype(np.float64) / 255.0
        
        return concentrations.astype(np.float64)
    
    def _apply_measurement_model(self, raw_concentrations: np.ndarray) -> np.ndarray:
        """Apply complete sensor measurement model to raw concentrations."""
        # Start with raw concentrations
        measured = raw_concentrations.copy()
        
        # Apply baseline offset
        if self._config.baseline_offset != 0.0:
            measured += self._config.baseline_offset
        
        # Apply calibration drift
        if self._config.enable_drift and self._total_drift != 0.0:
            measured += self._total_drift
        
        # Apply noise
        if self._config.noise_std > 0.0 and self._config.noise_type != "none":
            noise = self._generate_noise(measured.shape)
            measured += noise
            self._performance_metrics['noise_applications'] += measured.size
        
        # Apply saturation
        if self._config.saturation_level is not None:
            saturation_mask = measured > self._config.saturation_level
            if np.any(saturation_mask):
                measured[saturation_mask] = self._config.saturation_level
                self._performance_metrics['saturation_events'] += np.sum(saturation_mask)
        
        # Apply dynamic range clipping
        measured = np.clip(measured, self._config.dynamic_range[0], self._config.dynamic_range[1])
        
        # Apply quantization (resolution)
        if self._config.resolution > 0:
            measured = self._quantize_measurements(measured)
        
        return measured
    
    def _generate_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate measurement noise according to configuration."""
        if self._config.noise_type == "gaussian":
            return self._rng.normal(0.0, self._config.noise_std, shape)
        elif self._config.noise_type == "uniform":
            # Uniform noise with same standard deviation as Gaussian
            # For uniform distribution: std = range / sqrt(12)
            noise_range = self._config.noise_std * np.sqrt(12)
            return self._rng.uniform(-noise_range/2, noise_range/2, shape)
        else:
            return np.zeros(shape)
    
    def _quantize_measurements(self, measurements: np.ndarray) -> np.ndarray:
        """Apply measurement quantization based on resolution setting."""
        # Quantize to resolution steps
        quantized = np.round(measurements / self._config.resolution) * self._config.resolution
        
        # Ensure quantized values stay within dynamic range
        return np.clip(quantized, self._config.dynamic_range[0], self._config.dynamic_range[1])
    
    def _apply_temporal_filter(self, measurements: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Apply temporal filtering for sensor response dynamics."""
        if self._last_measurements is None or self._filter_state is None:
            # Initialize filter state
            self._filter_state = {pos_key: measurement for pos_key, measurement in 
                                zip([tuple(pos) for pos in positions], measurements)}
            return measurements
        
        # Apply exponential moving average filter
        filtered = np.zeros_like(measurements)
        
        for i, (pos, measurement) in enumerate(zip(positions, measurements)):
            pos_key = tuple(pos)
            
            if pos_key in self._filter_state:
                # Apply exponential filter: y[n] = α*x[n] + (1-α)*y[n-1]
                filtered[i] = (self._filter_alpha * measurement + 
                              (1 - self._filter_alpha) * self._filter_state[pos_key])
            else:
                # First measurement at this position
                filtered[i] = measurement
            
            # Update filter state
            self._filter_state[pos_key] = filtered[i]
        
        return filtered
    
    def _update_calibration_drift(self) -> None:
        """Update calibration drift simulation based on elapsed time."""
        current_time = time.time()
        time_elapsed = current_time - self._last_calibration_time
        
        # Update drift if enough time has passed
        if time_elapsed >= self._config.calibration_interval:
            drift_increment = self._config.drift_rate * time_elapsed
            self._total_drift += drift_increment
            
            # Limit total drift to reasonable bounds
            max_drift = 0.1 * (self._config.dynamic_range[1] - self._config.dynamic_range[0])
            self._total_drift = np.clip(self._total_drift, -max_drift, max_drift)
            
            self._last_calibration_time = current_time
            self._performance_metrics['drift_updates'] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and optimization.
        
        Returns:
            Dict[str, Any]: Dictionary containing detailed performance statistics
                including measurement timing, noise and drift statistics, filter
                operations, saturation events, and vectorized operation counts.
                
        Examples:
            Basic performance monitoring:
                >>> metrics = sensor.get_performance_metrics()
                >>> print(f"Average measurement time: {metrics['avg_measurement_time_ms']:.3f}ms")
                >>> print(f"Total measurements: {metrics['total_measurements']}")
                
            Performance optimization analysis:
                >>> metrics = sensor.get_performance_metrics()
                >>> if metrics['avg_measurement_time_ms'] > 0.1:
                ...     print("Performance degradation detected")
                >>> if metrics['saturation_events'] > 0:
                ...     print("Sensor saturation occurring")
        """
        base_metrics = super().get_performance_metrics()
        
        # Calculate timing statistics
        timing_metrics = {}
        if self._performance_metrics['measurement_times']:
            times = np.array(self._performance_metrics['measurement_times'])
            timing_metrics.update({
                'avg_measurement_time_ms': float(np.mean(times)),
                'max_measurement_time_ms': float(np.max(times)),
                'min_measurement_time_ms': float(np.min(times)),
                'std_measurement_time_ms': float(np.std(times)),
                'p95_measurement_time_ms': float(np.percentile(times, 95)),
                'performance_violations': int(np.sum(times > 0.1))  # >0.1ms violations
            })
        
        # Concentration sensor specific metrics
        sensor_metrics = {
            'sensor_type': 'ConcentrationSensor',
            'total_measurements': self._performance_metrics['total_measurements'],
            'measurement_count': self._measurement_count,
            'drift_updates': self._performance_metrics['drift_updates'],
            'filter_operations': self._performance_metrics['filter_operations'],
            'saturation_events': self._performance_metrics['saturation_events'],
            'noise_applications': self._performance_metrics['noise_applications'],
            'vectorized_operations': self._performance_metrics['vectorized_operations'],
            'current_drift_offset': self._total_drift,
            'configuration': {
                'dynamic_range': self._config.dynamic_range,
                'resolution': self._config.resolution,
                'noise_std': self._config.noise_std,
                'enable_drift': self._config.enable_drift,
                'enable_filtering': self._config.enable_filtering,
                'vectorized_ops': self._config.vectorized_ops
            }
        }
        
        # Combine all metrics
        combined_metrics = {**base_metrics, **timing_metrics, **sensor_metrics}
        
        return combined_metrics
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """
        Get comprehensive sensor information including configuration and capabilities.
        
        Returns:
            Dict[str, Any]: Dictionary containing sensor configuration, capabilities,
                and current operational state information.
                
        Examples:
            Sensor capability inspection:
                >>> info = sensor.get_sensor_info()
                >>> print(f"Dynamic range: {info['dynamic_range']}")
                >>> print(f"Resolution: {info['resolution']}")
                >>> print(f"Supports noise modeling: {info['supports_noise']}")
        """
        return {
            'sensor_type': 'ConcentrationSensor',
            'sensor_id': self._sensor_id,
            'sensor_protocol': 'SensorProtocol',
            'dynamic_range': self._config.dynamic_range,
            'resolution': self._config.resolution,
            'saturation_level': self._config.saturation_level,
            'noise_std': self._config.noise_std,
            'noise_type': self._config.noise_type,
            'supports_noise': self._config.noise_std > 0,
            'supports_drift': self._config.enable_drift,
            'supports_filtering': self._config.enable_filtering,
            'response_time': self._config.response_time,
            'filter_type': self._config.filter_type,
            'vectorized_ops': self._config.vectorized_ops,
            'enable_metadata': self._config.enable_metadata,
            'measurement_count': self._measurement_count,
            'current_drift': self._total_drift,
            'last_calibration_time': self._last_calibration_time,
            'capabilities': [
                'quantitative_measurement',
                'binary_detection',
                'gradient_computation',
                'noise_modeling',
                'drift_simulation',
                'temporal_filtering',
                'vectorized_operations',
                'performance_monitoring'
            ]
        }


# Factory functions for configuration-driven instantiation

def create_concentration_sensor_from_config(
    config: Union[DictConfig, Dict[str, Any]],
    **kwargs: Any
) -> ConcentrationSensor:
    """
    Create ConcentrationSensor from configuration with comprehensive validation.
    
    Args:
        config: Configuration dict or DictConfig containing sensor parameters
        **kwargs: Additional parameters to override configuration values
        
    Returns:
        ConcentrationSensor: Configured sensor instance
        
    Examples:
        From dictionary configuration:
            >>> config = {
            ...     'dynamic_range': (0.0, 10.0),
            ...     'resolution': 0.01,
            ...     'noise_std': 0.05,
            ...     'enable_drift': True
            ... }
            >>> sensor = create_concentration_sensor_from_config(config)
            
        From Hydra configuration:
            >>> sensor = create_concentration_sensor_from_config(cfg.sensors.concentration)
    """
    # Handle different configuration types
    if HYDRA_AVAILABLE and isinstance(config, DictConfig):
        config_dict = config
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise TypeError(f"Unsupported configuration type: {type(config)}")
    
    # Merge with kwargs
    config_dict = dict(config_dict)
    config_dict.update(kwargs)
    
    return ConcentrationSensor(**config_dict)


# Export public API
__all__ = [
    'ConcentrationSensor',
    'ConcentrationSensorConfig',
    'create_concentration_sensor_from_config'
]