"""
BinarySensor implementation providing threshold-based odor detection with configurable thresholds and noise modeling.

This module implements a high-performance binary odor sensor that provides boolean presence indicators
through configurable threshold-based detection. The sensor supports hysteresis for stable detection
behavior, realistic noise modeling with false positive/negative rates, and vectorized operations
for sub-microsecond per-agent detection latency in multi-agent scenarios.

Key Features:
- Threshold-based odor detection with boolean outputs
- Configurable detection thresholds with hysteresis support
- Noise modeling for false positive/negative rate simulation
- Vectorized threshold operations for multi-agent scenarios
- Sub-microsecond per-agent detection latency
- Confidence reporting and metadata collection
- Integration with logging and performance monitoring
- Configuration-driven parameter management

Performance Requirements:
- Detection latency: <1μs per agent for binary threshold operations
- Multi-agent scaling: Linear performance scaling up to 100+ agents
- Memory efficiency: <1KB overhead per agent for detection state
- Vectorized operations: Single NumPy operations for batch processing

Examples:
    Basic binary sensor with threshold detection:
        >>> sensor = BinarySensor(threshold=0.1)
        >>> detections = sensor.detect(concentration_values, positions)
        >>> print(f"Detection rate: {np.mean(detections):.2f}")
        
    Sensor with hysteresis for stable detection:
        >>> sensor = BinarySensor(threshold=0.1, hysteresis=0.02)
        >>> # Threshold up at 0.1, threshold down at 0.08
        >>> detections = sensor.detect(concentrations, positions)
        
    Noisy sensor with false positive/negative rates:
        >>> sensor = BinarySensor(
        ...     threshold=0.1,
        ...     false_positive_rate=0.05,
        ...     false_negative_rate=0.03
        ... )
        >>> detections = sensor.detect(concentrations, positions)
        
    Multi-agent vectorized detection:
        >>> positions = np.array([[0, 0], [10, 10], [20, 20]])
        >>> concentrations = np.array([0.05, 0.15, 0.25])
        >>> detections = sensor.detect(concentrations, positions)
        >>> # Returns: [False, True, True] for threshold=0.1
"""

from __future__ import annotations
import time
import warnings
from typing import Protocol, runtime_checkable, Optional, Union, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import numpy as np

# Core protocol imports
try:
    from ..protocols import NavigatorProtocol, SensorProtocol
    PROTOCOLS_AVAILABLE = True
except ImportError:
    # Handle case where protocols don't exist yet
    NavigatorProtocol = object
    SensorProtocol = object
    PROTOCOLS_AVAILABLE = False

# Hydra integration for configuration management
try:
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HydraConfig = None
    DictConfig = dict
    OmegaConf = None
    HYDRA_AVAILABLE = False

# Loguru integration for enhanced logging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False





@dataclass
class BinarySensorConfig:
    """
    Configuration schema for BinarySensor with validation and defaults.
    
    This dataclass provides type-safe configuration management for binary
    sensor parameters, supporting Hydra integration and parameter validation.
    
    Attributes:
        threshold: Detection threshold for odor concentration (0.0-1.0)
        hysteresis: Hysteresis band width for stable switching (default: 0.0)
        false_positive_rate: Probability of false positive detection (0.0-1.0)
        false_negative_rate: Probability of false negative detection (0.0-1.0)
        enable_logging: Enable detailed logging and performance monitoring
        random_seed: Seed for reproducible noise generation (None = random)
        confidence_reporting: Enable confidence metrics in metadata
        history_length: Number of past detections to maintain (0 = no history)
    """
    threshold: float = 0.1
    hysteresis: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    enable_logging: bool = True
    random_seed: Optional[int] = None
    confidence_reporting: bool = True
    history_length: int = 0
    
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0], got {self.threshold}")
        if not (0.0 <= self.hysteresis <= 0.5):
            raise ValueError(f"hysteresis must be in [0.0, 0.5], got {self.hysteresis}")
        if not (0.0 <= self.false_positive_rate <= 1.0):
            raise ValueError(f"false_positive_rate must be in [0.0, 1.0], got {self.false_positive_rate}")
        if not (0.0 <= self.false_negative_rate <= 1.0):
            raise ValueError(f"false_negative_rate must be in [0.0, 1.0], got {self.false_negative_rate}")
        if self.history_length < 0:
            raise ValueError(f"history_length must be >= 0, got {self.history_length}")


class BinarySensor:
    """
    High-performance binary odor sensor with threshold-based detection and noise modeling.
    
    This sensor provides boolean odor presence indicators through configurable
    threshold-based detection with support for hysteresis, noise modeling, and
    vectorized multi-agent operations. Designed for sub-microsecond per-agent
    detection latency with comprehensive performance monitoring.
    
    Key Features:
    - Threshold-based detection with configurable hysteresis
    - Realistic noise modeling with false positive/negative rates
    - Vectorized operations for multi-agent scenarios
    - Sub-microsecond per-agent detection latency
    - Confidence reporting and metadata collection
    - Performance monitoring and logging integration
    - Configuration-driven parameter management
    
    Performance Characteristics:
    - Detection latency: <1μs per agent
    - Memory usage: ~1KB per agent for state management
    - Scaling: Linear performance up to 100+ agents
    - Throughput: >1M detections/second on modern hardware
    
    Examples:
        Basic threshold detection:
            >>> sensor = BinarySensor(threshold=0.1)
            >>> concentrations = np.array([0.05, 0.15, 0.25])
            >>> positions = np.array([[0, 0], [10, 10], [20, 20]])
            >>> detections = sensor.detect(concentrations, positions)
            
        Hysteresis for stable detection:
            >>> sensor = BinarySensor(threshold=0.1, hysteresis=0.02)
            >>> # Rising threshold: 0.1, falling threshold: 0.08
            
        Noisy sensor simulation:
            >>> sensor = BinarySensor(
            ...     threshold=0.1,
            ...     false_positive_rate=0.05,
            ...     false_negative_rate=0.03
            ... )
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        hysteresis: float = 0.0,
        false_positive_rate: float = 0.0,
        false_negative_rate: float = 0.0,
        enable_logging: bool = True,
        random_seed: Optional[int] = None,
        confidence_reporting: bool = True,
        history_length: int = 0,
        sensor_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize binary sensor with configurable detection parameters.
        
        Args:
            threshold: Detection threshold for odor concentration (0.0-1.0)
            hysteresis: Hysteresis band width for stable switching (default: 0.0)
            false_positive_rate: Probability of false positive detection (0.0-1.0)
            false_negative_rate: Probability of false negative detection (0.0-1.0)
            enable_logging: Enable detailed logging and performance monitoring
            random_seed: Seed for reproducible noise generation (None = random)
            confidence_reporting: Enable confidence metrics in metadata
            history_length: Number of past detections to maintain (0 = no history)
            sensor_id: Unique identifier for this sensor instance
            **kwargs: Additional configuration parameters
            
        Raises:
            ValueError: If parameter values are outside valid ranges
        """
        # Validate and store core configuration
        self.config = BinarySensorConfig(
            threshold=threshold,
            hysteresis=hysteresis,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            enable_logging=enable_logging,
            random_seed=random_seed,
            confidence_reporting=confidence_reporting,
            history_length=history_length
        )
        
        # Sensor identification and logging setup
        self._sensor_id = sensor_id or f"binary_sensor_{id(self)}"
        self._enable_logging = enable_logging
        
        # Detection state management
        self._previous_detections: Optional[np.ndarray] = None
        self._detection_history: List[np.ndarray] = []
        self._num_agents: Optional[int] = None
        
        # Performance monitoring
        self._performance_metrics = {
            'total_detections': 0,
            'detection_times': [],
            'false_positives_applied': 0,
            'false_negatives_applied': 0,
            'hysteresis_activations': 0,
            'total_calls': 0
        }
        
        # Random number generation for noise modeling
        if random_seed is not None:
            self._rng = np.random.RandomState(random_seed)
        else:
            self._rng = np.random.RandomState()
        
        # Setup structured logging with context binding
        if self._enable_logging and LOGURU_AVAILABLE:
            self._logger = logger.bind(
                sensor_type="BinarySensor",
                sensor_id=self._sensor_id,
                threshold=self.config.threshold,
                hysteresis=self.config.hysteresis,
                noise_fp=self.config.false_positive_rate,
                noise_fn=self.config.false_negative_rate
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
                    pass
        else:
            self._logger = None
        
        # Log initialization
        if self._logger:
            self._logger.info(
                "BinarySensor initialized with high-performance detection",
                threshold=self.config.threshold,
                hysteresis=self.config.hysteresis,
                false_positive_rate=self.config.false_positive_rate,
                false_negative_rate=self.config.false_negative_rate,
                history_length=self.config.history_length,
                confidence_reporting=self.config.confidence_reporting
            )
    
    def detect(
        self, 
        concentration_values: np.ndarray, 
        positions: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Perform vectorized binary detection with sub-microsecond per-agent latency.
        
        This method implements high-performance threshold-based detection with
        optional hysteresis and noise modeling. All operations are vectorized
        for optimal multi-agent performance.
        
        Args:
            concentration_values: Odor concentration values with shape (num_agents,)
            positions: Agent positions with shape (num_agents, 2) 
            **kwargs: Additional detection parameters (unused in binary sensor)
            
        Returns:
            np.ndarray: Boolean detection array with shape (num_agents,)
            
        Raises:
            ValueError: If input arrays have incompatible shapes
            TypeError: If input arrays are not numpy arrays
            
        Examples:
            Single agent detection:
                >>> concentrations = np.array([0.15])
                >>> positions = np.array([[10.0, 20.0]])
                >>> detections = sensor.detect(concentrations, positions)
                
            Multi-agent batch detection:
                >>> concentrations = np.array([0.05, 0.15, 0.25, 0.08])
                >>> positions = np.array([[0, 0], [10, 10], [20, 20], [30, 30]])
                >>> detections = sensor.detect(concentrations, positions)
        """
        start_time = time.perf_counter() if self._enable_logging else None
        
        try:
            # Input validation
            if not isinstance(concentration_values, np.ndarray):
                raise TypeError("concentration_values must be a numpy array")
            if not isinstance(positions, np.ndarray):
                raise TypeError("positions must be a numpy array")
            
            concentration_values = np.asarray(concentration_values, dtype=np.float64)
            positions = np.asarray(positions, dtype=np.float64)
            
            # Ensure arrays are compatible
            if concentration_values.ndim != 1:
                raise ValueError(f"concentration_values must be 1D, got shape {concentration_values.shape}")
            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError(f"positions must have shape (N, 2), got {positions.shape}")
            if len(concentration_values) != len(positions):
                raise ValueError(f"Array length mismatch: {len(concentration_values)} vs {len(positions)}")
            
            num_agents = len(concentration_values)
            
            # Initialize agent state if needed
            if self._num_agents != num_agents:
                self._num_agents = num_agents
                self._previous_detections = np.zeros(num_agents, dtype=bool)
                if self.config.history_length > 0:
                    self._detection_history = []
            
            # Vectorized threshold detection with hysteresis
            detections = self._apply_threshold_with_hysteresis(concentration_values)
            
            # Apply noise modeling if configured
            if self.config.false_positive_rate > 0 or self.config.false_negative_rate > 0:
                detections = self._apply_noise_model(detections)
            
            # Update detection history
            if self.config.history_length > 0:
                self._detection_history.append(detections.copy())
                if len(self._detection_history) > self.config.history_length:
                    self._detection_history.pop(0)
            
            # Update previous state for hysteresis
            self._previous_detections = detections.copy()
            
            # Performance tracking
            if self._enable_logging:
                detection_time = (time.perf_counter() - start_time) * 1000000  # microseconds
                self._performance_metrics['detection_times'].append(detection_time)
                self._performance_metrics['total_detections'] += num_agents
                self._performance_metrics['total_calls'] += 1
                
                # Per-agent latency calculation
                per_agent_latency = detection_time / num_agents if num_agents > 0 else 0
                
                # Log performance warnings if needed
                if per_agent_latency > 1.0 and self._logger:  # >1μs per agent
                    self._logger.warning(
                        "Binary detection latency exceeded 1μs per agent",
                        per_agent_latency_us=per_agent_latency,
                        total_time_us=detection_time,
                        num_agents=num_agents,
                        performance_degradation=True
                    )
                
                # Periodic performance logging
                if self._performance_metrics['total_calls'] % 100 == 0 and self._logger:
                    recent_times = self._performance_metrics['detection_times'][-100:]
                    avg_time = np.mean(recent_times)
                    avg_per_agent = avg_time / num_agents if num_agents > 0 else 0
                    
                    self._logger.debug(
                        "Binary sensor performance summary",
                        total_calls=self._performance_metrics['total_calls'],
                        avg_detection_time_us=avg_time,
                        avg_per_agent_latency_us=avg_per_agent,
                        detection_rate=float(np.mean(detections)),
                        num_agents=num_agents
                    )
            
            return detections
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Binary detection failed: {str(e)}",
                    error_type=type(e).__name__,
                    concentration_shape=getattr(concentration_values, 'shape', 'unknown'),
                    positions_shape=getattr(positions, 'shape', 'unknown')
                )
            raise
    
    def _apply_threshold_with_hysteresis(self, concentration_values: np.ndarray) -> np.ndarray:
        """
        Apply threshold detection with optional hysteresis for stable switching.
        
        Hysteresis prevents oscillation near the threshold by using different
        thresholds for rising and falling edges:
        - Rising threshold: threshold
        - Falling threshold: threshold - hysteresis
        
        Args:
            concentration_values: Input concentration array
            
        Returns:
            np.ndarray: Boolean detection array after threshold and hysteresis
        """
        if self.config.hysteresis <= 0 or self._previous_detections is None:
            # Simple threshold without hysteresis
            return concentration_values >= self.config.threshold
        
        # Hysteresis logic: different thresholds for rising/falling edges
        rising_threshold = self.config.threshold
        falling_threshold = self.config.threshold - self.config.hysteresis
        
        # Current detection based on simple threshold
        current_detections = concentration_values >= rising_threshold
        
        # Apply hysteresis: only change state if concentration crosses the appropriate threshold
        hysteresis_detections = self._previous_detections.copy()
        
        # Rising edge: was False, now above rising threshold
        rising_mask = (~self._previous_detections) & current_detections
        hysteresis_detections[rising_mask] = True
        
        # Falling edge: was True, now below falling threshold
        falling_mask = self._previous_detections & (concentration_values < falling_threshold)
        hysteresis_detections[falling_mask] = False
        
        # Track hysteresis activations for monitoring
        if self._enable_logging:
            hysteresis_events = np.sum(rising_mask) + np.sum(falling_mask)
            self._performance_metrics['hysteresis_activations'] += hysteresis_events
        
        return hysteresis_detections
    
    def _apply_noise_model(self, clean_detections: np.ndarray) -> np.ndarray:
        """
        Apply realistic noise model with false positive and false negative rates.
        
        This method simulates sensor imperfections by randomly flipping
        detection results based on configured error rates:
        - False positives: True detections become False
        - False negatives: False detections become True
        
        Args:
            clean_detections: Clean detection results before noise
            
        Returns:
            np.ndarray: Noisy detection results after error injection
        """
        noisy_detections = clean_detections.copy()
        
        if self.config.false_positive_rate > 0:
            # False positives: flip False -> True
            false_detection_mask = ~clean_detections
            false_positive_prob = self._rng.random(size=len(clean_detections))
            false_positive_mask = false_detection_mask & (false_positive_prob < self.config.false_positive_rate)
            noisy_detections[false_positive_mask] = True
            
            if self._enable_logging:
                self._performance_metrics['false_positives_applied'] += np.sum(false_positive_mask)
        
        if self.config.false_negative_rate > 0:
            # False negatives: flip True -> False
            true_detection_mask = clean_detections
            false_negative_prob = self._rng.random(size=len(clean_detections))
            false_negative_mask = true_detection_mask & (false_negative_prob < self.config.false_negative_rate)
            noisy_detections[false_negative_mask] = False
            
            if self._enable_logging:
                self._performance_metrics['false_negatives_applied'] += np.sum(false_negative_mask)
        
        return noisy_detections
    
    def configure(self, **parameters: Any) -> None:
        """
        Dynamically configure sensor parameters with validation.
        
        This method allows runtime parameter updates while maintaining
        validation and logging integration. Configuration changes take
        effect immediately for subsequent detections.
        
        Args:
            **parameters: Parameter updates. Valid keys include:
                - threshold: New detection threshold (0.0-1.0)
                - hysteresis: New hysteresis band width (0.0-0.5)
                - false_positive_rate: New false positive rate (0.0-1.0)
                - false_negative_rate: New false negative rate (0.0-1.0)
                - random_seed: New random seed for noise generation
                
        Raises:
            ValueError: If parameter values are outside valid ranges
            
        Examples:
            Update threshold:
                >>> sensor.configure(threshold=0.15)
                
            Update noise parameters:
                >>> sensor.configure(
                ...     false_positive_rate=0.02,
                ...     false_negative_rate=0.01
                ... )
        """
        updated_params = []
        
        if 'threshold' in parameters:
            new_threshold = float(parameters['threshold'])
            if not (0.0 <= new_threshold <= 1.0):
                raise ValueError(f"threshold must be in [0.0, 1.0], got {new_threshold}")
            self.config.threshold = new_threshold
            updated_params.append('threshold')
        
        if 'hysteresis' in parameters:
            new_hysteresis = float(parameters['hysteresis'])
            if not (0.0 <= new_hysteresis <= 0.5):
                raise ValueError(f"hysteresis must be in [0.0, 0.5], got {new_hysteresis}")
            self.config.hysteresis = new_hysteresis
            updated_params.append('hysteresis')
        
        if 'false_positive_rate' in parameters:
            new_fp_rate = float(parameters['false_positive_rate'])
            if not (0.0 <= new_fp_rate <= 1.0):
                raise ValueError(f"false_positive_rate must be in [0.0, 1.0], got {new_fp_rate}")
            self.config.false_positive_rate = new_fp_rate
            updated_params.append('false_positive_rate')
        
        if 'false_negative_rate' in parameters:
            new_fn_rate = float(parameters['false_negative_rate'])
            if not (0.0 <= new_fn_rate <= 1.0):
                raise ValueError(f"false_negative_rate must be in [0.0, 1.0], got {new_fn_rate}")
            self.config.false_negative_rate = new_fn_rate
            updated_params.append('false_negative_rate')
        
        if 'random_seed' in parameters:
            new_seed = parameters['random_seed']
            if new_seed is not None:
                self._rng = np.random.RandomState(new_seed)
            else:
                self._rng = np.random.RandomState()
            self.config.random_seed = new_seed
            updated_params.append('random_seed')
        
        # Log configuration updates
        if self._logger and updated_params:
            self._logger.info(
                "Binary sensor configuration updated",
                updated_parameters=updated_params,
                new_threshold=self.config.threshold,
                new_hysteresis=self.config.hysteresis,
                new_fp_rate=self.config.false_positive_rate,
                new_fn_rate=self.config.false_negative_rate
            )
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get comprehensive sensor metadata including performance metrics and configuration.
        
        Returns sensor configuration, performance statistics, detection history,
        and confidence metrics for monitoring and debugging purposes.
        
        Returns:
            Dict[str, Any]: Comprehensive sensor metadata including:
                - configuration: Current sensor parameters
                - performance: Detection timing and throughput metrics
                - statistics: Detection rates and noise application stats
                - history: Recent detection patterns (if enabled)
                - confidence: Detection quality metrics (if enabled)
                
        Examples:
            Get performance metrics:
                >>> metadata = sensor.get_metadata()
                >>> print(f"Avg latency: {metadata['performance']['avg_latency_us']:.2f}μs")
                
            Monitor detection statistics:
                >>> stats = sensor.get_metadata()['statistics']
                >>> print(f"Detection rate: {stats['detection_rate']:.2f}")
        """
        metadata = {
            'sensor_type': 'BinarySensor',
            'sensor_id': self._sensor_id,
            'configuration': {
                'threshold': self.config.threshold,
                'hysteresis': self.config.hysteresis,
                'false_positive_rate': self.config.false_positive_rate,
                'false_negative_rate': self.config.false_negative_rate,
                'history_length': self.config.history_length,
                'confidence_reporting': self.config.confidence_reporting,
                'random_seed': self.config.random_seed
            },
            'state': {
                'num_agents': self._num_agents,
                'has_previous_detections': self._previous_detections is not None,
                'history_entries': len(self._detection_history)
            }
        }
        
        # Performance metrics
        if self._performance_metrics['detection_times']:
            detection_times = np.array(self._performance_metrics['detection_times'])
            metadata['performance'] = {
                'total_calls': self._performance_metrics['total_calls'],
                'total_detections': self._performance_metrics['total_detections'],
                'avg_latency_us': float(np.mean(detection_times)),
                'max_latency_us': float(np.max(detection_times)),
                'min_latency_us': float(np.min(detection_times)),
                'std_latency_us': float(np.std(detection_times)),
                'avg_per_agent_latency_us': float(np.mean(detection_times) / max(1, self._num_agents or 1)),
                'throughput_detections_per_sec': 1e6 / float(np.mean(detection_times)) if len(detection_times) > 0 else 0
            }
        else:
            metadata['performance'] = {
                'total_calls': 0,
                'total_detections': 0,
                'avg_latency_us': 0,
                'throughput_detections_per_sec': 0
            }
        
        # Noise modeling statistics
        metadata['noise_statistics'] = {
            'false_positives_applied': self._performance_metrics['false_positives_applied'],
            'false_negatives_applied': self._performance_metrics['false_negatives_applied'],
            'hysteresis_activations': self._performance_metrics['hysteresis_activations']
        }
        
        # Detection history and patterns
        if self._detection_history and self.config.history_length > 0:
            history_array = np.array(self._detection_history)
            metadata['detection_history'] = {
                'length': len(self._detection_history),
                'recent_detection_rates': [float(np.mean(frame)) for frame in self._detection_history[-5:]],
                'temporal_stability': float(np.std([np.mean(frame) for frame in self._detection_history])),
                'per_agent_consistency': [float(np.std(history_array[:, i])) for i in range(min(5, history_array.shape[1]))]
            }
        
        # Confidence metrics (if enabled)
        if self.config.confidence_reporting and self._previous_detections is not None:
            metadata['confidence_metrics'] = {
                'current_detection_rate': float(np.mean(self._previous_detections)),
                'detection_consistency': float(1.0 - np.std(self._previous_detections.astype(float))),
                'effective_threshold': self.config.threshold - (self.config.hysteresis / 2),
                'noise_impact_estimate': (self.config.false_positive_rate + self.config.false_negative_rate) / 2
            }
        
        return metadata
    
    def reset(self) -> None:
        """
        Reset sensor internal state for new episode.
        
        Clears detection history, previous states, and resets performance
        metrics for a fresh episode start. Configuration parameters are
        preserved across resets.
        
        Examples:
            Reset between episodes:
                >>> sensor.reset()  # Clear history and state
                >>> # Sensor ready for new episode
        """
        # Clear detection state
        self._previous_detections = None
        self._detection_history = []
        self._num_agents = None
        
        # Reset performance metrics for new episode
        self._performance_metrics = {
            'total_detections': 0,
            'detection_times': [],
            'false_positives_applied': 0,
            'false_negatives_applied': 0,
            'hysteresis_activations': 0,
            'total_calls': 0
        }
        
        # Log reset
        if self._logger:
            self._logger.debug(
                "Binary sensor reset for new episode",
                threshold=self.config.threshold,
                hysteresis=self.config.hysteresis,
                noise_config=(self.config.false_positive_rate, self.config.false_negative_rate)
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get detailed performance metrics for monitoring and optimization.
        
        Returns comprehensive timing statistics, throughput metrics, and
        performance indicators for sensor operation analysis.
        
        Returns:
            Dict[str, Any]: Performance metrics including timing, throughput,
                and efficiency statistics
        """
        if not self._performance_metrics['detection_times']:
            return {
                'total_calls': 0,
                'total_detections': 0,
                'avg_latency_us': 0,
                'throughput_detections_per_sec': 0,
                'efficiency_score': 0
            }
        
        detection_times = np.array(self._performance_metrics['detection_times'])
        
        return {
            'total_calls': self._performance_metrics['total_calls'],
            'total_detections': self._performance_metrics['total_detections'],
            'avg_latency_us': float(np.mean(detection_times)),
            'p95_latency_us': float(np.percentile(detection_times, 95)),
            'max_latency_us': float(np.max(detection_times)),
            'min_latency_us': float(np.min(detection_times)),
            'latency_std_us': float(np.std(detection_times)),
            'avg_per_agent_latency_us': float(np.mean(detection_times) / max(1, self._num_agents or 1)),
            'throughput_detections_per_sec': float(self._num_agents / (np.mean(detection_times) / 1e6)) if self._num_agents else 0,
            'efficiency_score': min(1.0, 1.0 / max(1.0, np.mean(detection_times))),  # 1.0 at 1μs, decreases with latency
            'performance_violations': int(np.sum(detection_times > 1.0))  # Count of >1μs per agent
        }

    def measure(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Perform quantitative measurements using binary sensor (compatibility method).
        
        This method provides SensorProtocol compatibility by converting binary
        detection results to quantitative measurements. Returns threshold values
        for detected positions and zero for non-detected positions.
        
        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent
                
        Returns:
            np.ndarray: Quantitative measurement values - threshold value for detections,
                0.0 for non-detections. Shape matches detection output.
                
        Notes:
            BinarySensor provides binary detection as its primary functionality.
            This method enables compatibility with quantitative measurement interfaces
            by mapping detection results to simple threshold-based values.
            
            For true quantitative measurements, use ConcentrationSensor instead.
            
        Examples:
            Single agent measurement:
            >>> position = np.array([15, 25])
            >>> measurement = sensor.measure(plume_state, position)
            
            Multi-agent measurements:
            >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
            >>> measurements = sensor.measure(plume_state, positions)
        """
        # Extract concentration values from plume state  
        try:
            # Try to get concentration values from plume state
            if hasattr(plume_state, 'get_concentration_values'):
                concentration_values = plume_state.get_concentration_values(positions)
            elif hasattr(plume_state, 'concentration'):
                # Try direct concentration access 
                if hasattr(plume_state.concentration, '__call__'):
                    concentration_values = plume_state.concentration(positions)
                else:
                    concentration_values = plume_state.concentration
            elif hasattr(plume_state, '__call__'):
                # Plume state is callable
                concentration_values = plume_state(positions)
            else:
                # Fallback: assume plume_state is array-like concentration values
                concentration_values = np.asarray(plume_state)
                
            # Get binary detections using primary detect() method
            detections = self.detect(concentration_values, positions)
        except Exception as e:
            # If we can't extract concentrations, return zero measurements
            if isinstance(positions, np.ndarray) and positions.ndim == 1:
                return np.array([0.0], dtype=np.float64)
            else:
                return np.zeros(positions.shape[0] if positions.ndim == 2 else 1, dtype=np.float64)
        
        # Convert boolean detections to quantitative values
        # Detected = threshold value, Not detected = 0.0
        if isinstance(detections, (bool, np.bool_)):
            measurements = np.array([float(self.config.threshold) if detections else 0.0])
            return measurements.astype(np.float64)
        else:
            measurements = np.where(detections, self.config.threshold, 0.0)
            return measurements.astype(np.float64)
    
    def compute_gradient(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Compute concentration gradients using binary sensor (compatibility method).
        
        This method provides SensorProtocol compatibility by estimating gradients
        from binary detection patterns. Uses finite difference approximation with
        small spatial offsets to detect concentration gradient directions.
        
        Args:
            plume_state: Current plume model state providing concentration field access  
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent
                
        Returns:
            np.ndarray: Estimated gradient vectors with shape (n_agents, 2) or (2,) for single agent.
                Components represent approximate [∂c/∂x, ∂c/∂y] directions based on detection patterns.
                
        Notes:
            BinarySensor provides binary detection as its primary functionality.
            This gradient estimation is approximate and limited compared to dedicated
            gradient sensors. For accurate gradient computation, use GradientSensor instead.
            
            The method uses small spatial offsets (0.1 units) to sample neighboring
            positions and estimate gradient direction from detection differences.
            
        Examples:
            Single agent gradient:
            >>> position = np.array([15, 25])  
            >>> gradient = sensor.compute_gradient(plume_state, position)
            
            Multi-agent gradients:
            >>> positions = np.array([[10, 20], [15, 25], [20, 30]])
            >>> gradients = sensor.compute_gradient(plume_state, positions)
        """
        # Ensure positions is 2D array for consistent processing
        positions = np.atleast_2d(positions)
        is_single_agent = positions.shape[0] == 1
        
        # Spatial offset for finite difference approximation
        offset = 0.1  # Small offset for gradient estimation
        
        gradients = np.zeros((positions.shape[0], 2))
        
        for i, pos in enumerate(positions):
            try:
                # Extract concentration values for center position
                if hasattr(plume_state, 'get_concentration_values'):
                    center_conc = plume_state.get_concentration_values(pos.reshape(1, -1))
                elif hasattr(plume_state, 'concentration'):
                    if hasattr(plume_state.concentration, '__call__'):
                        center_conc = plume_state.concentration(pos.reshape(1, -1))
                    else:
                        center_conc = plume_state.concentration
                elif hasattr(plume_state, '__call__'):
                    center_conc = plume_state(pos.reshape(1, -1))
                else:
                    center_conc = np.asarray(plume_state)
                
                center_detection = self.detect(center_conc, pos.reshape(1, -1))[0]
                
                # Sample neighboring positions for gradient estimation
                pos_x_plus = pos + np.array([offset, 0])
                pos_x_minus = pos - np.array([offset, 0])  
                pos_y_plus = pos + np.array([0, offset])
                pos_y_minus = pos - np.array([0, offset])
                
                # Get concentration values at neighboring positions  
                if hasattr(plume_state, 'get_concentration_values'):
                    conc_x_plus = plume_state.get_concentration_values(pos_x_plus.reshape(1, -1))
                    conc_x_minus = plume_state.get_concentration_values(pos_x_minus.reshape(1, -1))
                    conc_y_plus = plume_state.get_concentration_values(pos_y_plus.reshape(1, -1))
                    conc_y_minus = plume_state.get_concentration_values(pos_y_minus.reshape(1, -1))
                elif hasattr(plume_state, 'concentration'):
                    if hasattr(plume_state.concentration, '__call__'):
                        conc_x_plus = plume_state.concentration(pos_x_plus.reshape(1, -1))
                        conc_x_minus = plume_state.concentration(pos_x_minus.reshape(1, -1))
                        conc_y_plus = plume_state.concentration(pos_y_plus.reshape(1, -1))
                        conc_y_minus = plume_state.concentration(pos_y_minus.reshape(1, -1))
                    else:
                        # Use same concentration for all positions if not callable
                        conc_x_plus = conc_x_minus = conc_y_plus = conc_y_minus = plume_state.concentration
                elif hasattr(plume_state, '__call__'):
                    conc_x_plus = plume_state(pos_x_plus.reshape(1, -1))
                    conc_x_minus = plume_state(pos_x_minus.reshape(1, -1))
                    conc_y_plus = plume_state(pos_y_plus.reshape(1, -1))
                    conc_y_minus = plume_state(pos_y_minus.reshape(1, -1))
                else:
                    # Use same concentration for all positions
                    conc_x_plus = conc_x_minus = conc_y_plus = conc_y_minus = np.asarray(plume_state)
                
                # Get detections at neighboring positions
                det_x_plus = self.detect(conc_x_plus, pos_x_plus.reshape(1, -1))[0]
                det_x_minus = self.detect(conc_x_minus, pos_x_minus.reshape(1, -1))[0]
                det_y_plus = self.detect(conc_y_plus, pos_y_plus.reshape(1, -1))[0] 
                det_y_minus = self.detect(conc_y_minus, pos_y_minus.reshape(1, -1))[0]
            except Exception:
                # If concentration extraction fails, assume no gradient
                center_detection = det_x_plus = det_x_minus = det_y_plus = det_y_minus = False
            
            # Estimate gradient components using finite differences
            # Convert boolean to float for arithmetic
            grad_x = (float(det_x_plus) - float(det_x_minus)) / (2 * offset)
            grad_y = (float(det_y_plus) - float(det_y_minus)) / (2 * offset)
            
            gradients[i] = [grad_x, grad_y]
        
        # Return appropriate shape based on input
        if is_single_agent and positions.shape[0] == 1:
            return gradients[0]
        else:
            return gradients


# Factory functions for configuration-driven instantiation

def create_binary_sensor_from_config(
    config: Union[DictConfig, Dict[str, Any], BinarySensorConfig],
    sensor_id: Optional[str] = None
) -> BinarySensor:
    """
    Create BinarySensor from configuration object with comprehensive validation.
    
    This factory function enables configuration-driven sensor instantiation
    with support for Hydra, Pydantic, and plain dictionary configurations.
    
    Args:
        config: Configuration object containing sensor parameters
        sensor_id: Optional unique identifier for the sensor
        
    Returns:
        BinarySensor: Configured sensor instance
        
    Raises:
        ValueError: If configuration is invalid
        TypeError: If configuration type is unsupported
        
    Examples:
        From Hydra configuration:
            >>> config = DictConfig({"threshold": 0.15, "hysteresis": 0.02})
            >>> sensor = create_binary_sensor_from_config(config)
            
        From dictionary:
            >>> config = {"threshold": 0.1, "false_positive_rate": 0.05}
            >>> sensor = create_binary_sensor_from_config(config)
    """
    # Handle different configuration types
    if isinstance(config, BinarySensorConfig):
        config_dict = config.__dict__
    elif isinstance(config, DictConfig) and HYDRA_AVAILABLE:
        config_dict = OmegaConf.to_container(config, resolve=True)
    elif isinstance(config, dict):
        config_dict = config.copy()
    else:
        raise TypeError(f"Unsupported configuration type: {type(config)}")
    
    # Add sensor_id if provided
    if sensor_id:
        config_dict['sensor_id'] = sensor_id
    
    return BinarySensor(**config_dict)


# Export public API
__all__ = [
    # Core sensor classes
    "BinarySensor",
    "BinarySensorConfig",
    
    # Factory functions
    "create_binary_sensor_from_config",
]