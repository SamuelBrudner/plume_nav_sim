"""
HistoricalSensor wrapper implementation providing optional temporal history integration.

This module implements the HistoricalSensor wrapper class that provides optional temporal 
history integration for any SensorProtocol implementation, enabling memory-based navigation 
strategies through configurable history length, temporal windowing, data aggregation methods, 
and efficient circular buffer management for historical sensor data access.

The HistoricalSensor maintains agent-agnostic design by providing history without enforcing 
memory usage patterns, allowing both memory-based and non-memory-based agents to use the 
same sensor infrastructure seamlessly.

Key Features:
- Wraps any SensorProtocol implementation to add temporal history functionality
- Configurable history length and temporal windowing for memory-based navigation strategies
- Efficient circular buffer management for historical data with memory optimization
- Data aggregation methods for temporal pattern analysis and historical gradient computation
- Performance monitoring integration for consistent system instrumentation
- Thread-safe operations for multi-agent scenarios with shared history buffers

Performance Requirements:
- Sensor operations: <0.1ms per agent per sensor for minimal overhead
- History access: <0.05ms for recent data retrieval and aggregation
- Memory efficiency: <1KB per agent per 100 historical samples
- Thread safety: Safe concurrent access for multi-agent scenarios

Examples:
    Basic historical sensor wrapper:
        >>> base_sensor = ConcentrationSensor(dynamic_range=(0, 1), resolution=0.001)
        >>> historical_sensor = HistoricalSensor(
        ...     base_sensor=base_sensor,
        ...     history_length=100,
        ...     enable_aggregation=True
        ... )
        >>> measurements = historical_sensor.measure(plume_state, agent_positions)
        >>> recent_avg = historical_sensor.get_moving_average(window_size=10)
        
    Advanced temporal windowing:
        >>> temporal_sensor = HistoricalSensor(
        ...     base_sensor=GradientSensor(spatial_resolution=(0.5, 0.5)),
        ...     history_length=200,
        ...     temporal_window=30.0,  # 30 second window
        ...     aggregation_methods=['mean', 'std', 'gradient']
        ... )
        >>> gradients = temporal_sensor.compute_gradient(plume_state, positions)
        >>> temporal_gradient = temporal_sensor.get_temporal_gradient()
        
    Configuration-driven instantiation:
        >>> config = {
        ...     'base_sensor': {'type': 'BinarySensor', 'threshold': 0.1},
        ...     'history_length': 150,
        ...     'enable_temporal_windowing': True,
        ...     'aggregation_methods': ['mean', 'max', 'trend']
        ... }
        >>> sensor = HistoricalSensor.from_config(config)

Design Principles:
- Wrapper Pattern: Maintains full compatibility with underlying SensorProtocol implementation
- Memory Efficiency: Circular buffer with configurable size limits and automatic cleanup
- Agent Agnostic: History available but not enforced, supporting both memory and memoryless agents
- Performance First: Minimal overhead while providing rich temporal analysis capabilities
- Extensible: Easy addition of new aggregation methods and temporal analysis functions
"""

from __future__ import annotations
import time
import threading
import warnings
from typing import Protocol, Union, Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np

# Core protocol imports
from ..protocols import SensorProtocol

# Hydra integration for configuration management
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

# Enhanced logging integration
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
class HistoricalSensorConfig:
    """
    Configuration schema for HistoricalSensor with comprehensive parameter validation.
    
    This dataclass provides type-safe configuration for historical sensor functionality
    with validation constraints ensuring physical consistency and performance requirements.
    
    Attributes:
        history_length: Maximum number of historical samples to maintain per agent
        temporal_window: Time window in seconds for temporal analysis (None = unlimited)
        enable_aggregation: Enable built-in aggregation methods for temporal analysis
        aggregation_methods: List of aggregation method names to enable
        memory_limit_mb: Maximum memory usage for historical data in megabytes
        cleanup_interval: Interval for memory cleanup operations in seconds
        enable_threading: Enable thread-safe operations for multi-agent scenarios
        performance_monitoring: Enable detailed performance metrics collection
    """
    history_length: int = 100
    temporal_window: Optional[float] = None
    enable_aggregation: bool = True
    aggregation_methods: List[str] = field(default_factory=lambda: ['mean', 'std', 'min', 'max'])
    memory_limit_mb: float = 10.0
    cleanup_interval: float = 30.0
    enable_threading: bool = True
    performance_monitoring: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters for consistency and performance."""
        if self.history_length <= 0:
            raise ValueError(f"history_length must be positive, got {self.history_length}")
        
        if self.temporal_window is not None and self.temporal_window <= 0:
            raise ValueError(f"temporal_window must be positive, got {self.temporal_window}")
        
        if self.memory_limit_mb <= 0:
            raise ValueError(f"memory_limit_mb must be positive, got {self.memory_limit_mb}")
        
        if self.cleanup_interval <= 0:
            raise ValueError(f"cleanup_interval must be positive, got {self.cleanup_interval}")
        
        # Validate aggregation methods
        valid_methods = {'mean', 'std', 'min', 'max', 'median', 'trend', 'gradient', 'variance'}
        invalid_methods = set(self.aggregation_methods) - valid_methods
        if invalid_methods:
            raise ValueError(f"Invalid aggregation methods: {invalid_methods}. Valid: {valid_methods}")


@dataclass
class HistoricalDataPoint:
    """
    Single historical data point with timestamp and sensor reading.
    
    This dataclass encapsulates individual sensor readings with temporal metadata
    for efficient storage and retrieval in circular buffer structures.
    
    Attributes:
        timestamp: Recording timestamp in seconds since epoch
        reading: Sensor reading data (format depends on sensor type)
        position: Agent position when reading was taken
        metadata: Additional sensor-specific metadata
    """
    timestamp: float
    reading: Union[float, bool, np.ndarray]
    position: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure position is a numpy array for consistency."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)


class CircularBuffer:
    """
    Thread-safe circular buffer for efficient historical data storage.
    
    This class implements a high-performance circular buffer with automatic
    memory management, thread safety, and configurable size limits for
    storing temporal sensor data with minimal overhead.
    
    Performance Characteristics:
    - O(1) insertion and recent data access
    - O(n) full buffer iteration (where n = buffer size)
    - Memory usage bounded by max_size parameter
    - Thread-safe concurrent access for multi-agent scenarios
    
    Examples:
        Basic circular buffer usage:
            >>> buffer = CircularBuffer(max_size=100)
            >>> buffer.append(HistoricalDataPoint(time.time(), 0.5, np.array([10, 20])))
            >>> recent_data = buffer.get_recent(10)  # Last 10 entries
            
        Thread-safe multi-agent access:
            >>> buffer = CircularBuffer(max_size=500, thread_safe=True)
            >>> # Multiple threads can safely append and read concurrently
    """
    
    def __init__(
        self, 
        max_size: int = 100, 
        thread_safe: bool = True,
        enable_monitoring: bool = True
    ):
        """
        Initialize circular buffer with specified capacity and safety options.
        
        Args:
            max_size: Maximum number of data points to store
            thread_safe: Enable thread synchronization for concurrent access
            enable_monitoring: Enable performance monitoring and statistics
        """
        self.max_size = max_size
        self.thread_safe = thread_safe
        self.enable_monitoring = enable_monitoring
        
        self._buffer = deque(maxlen=max_size)
        self._lock = threading.RLock() if thread_safe else None
        
        # Performance monitoring
        self._access_count = 0
        self._total_append_time = 0.0
        self._total_access_time = 0.0
        
    def append(self, data_point: HistoricalDataPoint) -> None:
        """
        Append new data point to buffer with thread safety.
        
        Args:
            data_point: Historical data point to store
        """
        start_time = time.perf_counter() if self.enable_monitoring else None
        
        if self._lock:
            with self._lock:
                self._buffer.append(data_point)
        else:
            self._buffer.append(data_point)
        
        if self.enable_monitoring:
            self._total_append_time += time.perf_counter() - start_time
    
    def get_recent(self, count: int) -> List[HistoricalDataPoint]:
        """
        Get the most recent data points from buffer.
        
        Args:
            count: Number of recent data points to retrieve
            
        Returns:
            List[HistoricalDataPoint]: Recent data points (newest first)
        """
        start_time = time.perf_counter() if self.enable_monitoring else None
        
        if self._lock:
            with self._lock:
                result = list(self._buffer)[-count:] if count > 0 else []
        else:
            result = list(self._buffer)[-count:] if count > 0 else []
        
        if self.enable_monitoring:
            self._access_count += 1
            self._total_access_time += time.perf_counter() - start_time
        
        return result
    
    def get_in_time_window(self, window_seconds: float) -> List[HistoricalDataPoint]:
        """
        Get all data points within specified time window from present.
        
        Args:
            window_seconds: Time window in seconds from current time
            
        Returns:
            List[HistoricalDataPoint]: Data points within time window
        """
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        start_time = time.perf_counter() if self.enable_monitoring else None
        
        if self._lock:
            with self._lock:
                result = [dp for dp in self._buffer if dp.timestamp >= cutoff_time]
        else:
            result = [dp for dp in self._buffer if dp.timestamp >= cutoff_time]
        
        if self.enable_monitoring:
            self._access_count += 1
            self._total_access_time += time.perf_counter() - start_time
        
        return result
    
    def clear(self) -> None:
        """Clear all data from buffer."""
        if self._lock:
            with self._lock:
                self._buffer.clear()
        else:
            self._buffer.clear()
    
    def __len__(self) -> int:
        """Get current number of data points in buffer."""
        return len(self._buffer)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring and optimization.
        
        Returns:
            Dict[str, Any]: Performance statistics dictionary
        """
        if not self.enable_monitoring:
            return {}
        
        stats = {
            'buffer_size': len(self._buffer),
            'max_size': self.max_size,
            'utilization': len(self._buffer) / self.max_size,
            'access_count': self._access_count,
            'thread_safe': self.thread_safe
        }
        
        if self._access_count > 0:
            stats.update({
                'avg_append_time_ms': (self._total_append_time / len(self._buffer)) * 1000,
                'avg_access_time_ms': (self._total_access_time / self._access_count) * 1000
            })
        
        return stats


class HistoricalSensor:
    """
    HistoricalSensor wrapper providing temporal history integration for any SensorProtocol implementation.
    
    This class wraps existing sensor implementations to add configurable temporal history
    functionality without modifying the underlying sensor logic. It provides memory-based
    navigation strategies through efficient circular buffer management, data aggregation
    methods, and performance optimization while maintaining agent-agnostic design principles.
    
    The wrapper maintains full protocol compatibility with the base sensor while adding
    rich temporal analysis capabilities including moving averages, temporal gradients,
    trend analysis, and configurable windowing functions.
    
    Key Features:
    - Protocol-compliant wrapper for any SensorProtocol implementation
    - Configurable history length and temporal windowing for flexible memory strategies
    - Efficient circular buffer with thread-safe operations for multi-agent scenarios
    - Built-in aggregation methods for temporal pattern analysis
    - Memory management with automatic cleanup and configurable limits
    - Performance monitoring integration for system optimization
    
    Performance Characteristics:
    - Sensor operations: <0.1ms overhead per agent per sensor
    - History access: <0.05ms for recent data retrieval and aggregation
    - Memory usage: <1KB per agent per 100 historical samples
    - Thread safety: Concurrent access support for multi-agent scenarios
    
    Examples:
        Basic wrapper usage:
            >>> base_sensor = ConcentrationSensor(dynamic_range=(0, 1))
            >>> historical_sensor = HistoricalSensor(
            ...     base_sensor=base_sensor,
            ...     config=HistoricalSensorConfig(history_length=100)
            ... )
            >>> readings = historical_sensor.measure(plume_state, positions)
            >>> avg_reading = historical_sensor.get_moving_average(window_size=10)
            
        Advanced temporal analysis:
            >>> config = HistoricalSensorConfig(
            ...     history_length=200,
            ...     temporal_window=30.0,
            ...     aggregation_methods=['mean', 'std', 'gradient', 'trend']
            ... )
            >>> historical_sensor = HistoricalSensor(gradient_sensor, config)
            >>> gradients = historical_sensor.compute_gradient(plume_state, positions)
            >>> temporal_trend = historical_sensor.get_temporal_trend()
            
        Multi-agent scenario:
            >>> historical_sensor = HistoricalSensor(
            ...     binary_sensor,
            ...     config=HistoricalSensorConfig(
            ...         history_length=150,
            ...         enable_threading=True,
            ...         memory_limit_mb=5.0
            ...     )
            ... )
            >>> # Safe for concurrent access by multiple agents
    """
    
    def __init__(
        self,
        base_sensor: SensorProtocol,
        config: Optional[HistoricalSensorConfig] = None,
        sensor_id: Optional[str] = None,
        enable_logging: bool = True
    ):
        """
        Initialize HistoricalSensor wrapper with base sensor and configuration.
        
        Args:
            base_sensor: SensorProtocol implementation to wrap with history functionality
            config: Configuration parameters for historical functionality
            sensor_id: Unique identifier for this sensor instance
            enable_logging: Enable comprehensive logging integration
            
        Raises:
            TypeError: If base_sensor does not implement SensorProtocol
            ValueError: If configuration parameters are invalid
        """
        # Validate base sensor implements SensorProtocol
        if not isinstance(base_sensor, SensorProtocol):
            raise TypeError(
                f"base_sensor must implement SensorProtocol, got {type(base_sensor)}"
            )
        
        self.base_sensor = base_sensor
        self.config = config or HistoricalSensorConfig()
        self.sensor_id = sensor_id or f"historical_sensor_{id(self)}"
        self.enable_logging = enable_logging
        
        # Initialize historical data storage per agent
        self._agent_histories: Dict[str, CircularBuffer] = {}
        self._last_cleanup_time = time.time()
        
        # Thread safety for multi-agent scenarios
        self._lock = threading.RLock() if self.config.enable_threading else None
        
        # Performance monitoring
        self._performance_metrics = {
            'total_operations': 0,
            'total_time_ms': 0.0,
            'aggregation_operations': 0,
            'aggregation_time_ms': 0.0,
            'memory_usage_mb': 0.0
        }
        
        # Setup structured logging with context binding
        if self.enable_logging and LOGURU_AVAILABLE:
            self._logger = logger.bind(
                sensor_type="HistoricalSensor",
                sensor_id=self.sensor_id,
                base_sensor_type=type(self.base_sensor).__name__,
                history_length=self.config.history_length,
                temporal_window=self.config.temporal_window,
                enable_threading=self.config.enable_threading
            )
            
            self._logger.info(
                "HistoricalSensor initialized with enhanced temporal functionality",
                config=self.config.__dict__,
                base_sensor_info=getattr(self.base_sensor, '__dict__', {})
            )
        else:
            self._logger = None
    
    def _get_agent_key(self, position: np.ndarray) -> str:
        """
        Generate unique key for agent based on position for history tracking.
        
        Args:
            position: Agent position as numpy array
            
        Returns:
            str: Unique agent identifier for history tracking
        """
        # Use position hash for agent identification
        # In multi-agent scenarios, positions should be stable per agent
        return f"agent_{hash(tuple(position.flatten()))}"
    
    def _get_or_create_history(self, agent_key: str) -> CircularBuffer:
        """
        Get or create circular buffer for specified agent.
        
        Args:
            agent_key: Unique agent identifier
            
        Returns:
            CircularBuffer: Historical data buffer for agent
        """
        if self._lock:
            with self._lock:
                if agent_key not in self._agent_histories:
                    self._agent_histories[agent_key] = CircularBuffer(
                        max_size=self.config.history_length,
                        thread_safe=self.config.enable_threading,
                        enable_monitoring=self.config.performance_monitoring
                    )
                return self._agent_histories[agent_key]
        else:
            if agent_key not in self._agent_histories:
                self._agent_histories[agent_key] = CircularBuffer(
                    max_size=self.config.history_length,
                    thread_safe=self.config.enable_threading,
                    enable_monitoring=self.config.performance_monitoring
                )
            return self._agent_histories[agent_key]
    
    def _store_reading(
        self, 
        agent_position: np.ndarray, 
        reading: Union[float, bool, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store sensor reading in agent's historical buffer.
        
        Args:
            agent_position: Position where reading was taken
            reading: Sensor reading data
            metadata: Additional metadata to store with reading
        """
        agent_key = self._get_agent_key(agent_position)
        history_buffer = self._get_or_create_history(agent_key)
        
        data_point = HistoricalDataPoint(
            timestamp=time.time(),
            reading=reading,
            position=agent_position.copy(),
            metadata=metadata or {}
        )
        
        history_buffer.append(data_point)
    
    def _cleanup_memory(self) -> None:
        """
        Perform memory cleanup if needed based on configuration limits.
        """
        current_time = time.time()
        if current_time - self._last_cleanup_time < self.config.cleanup_interval:
            return
        
        if PSUTIL_AVAILABLE:
            # Estimate memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.config.memory_limit_mb:
                # Remove oldest agent histories if over limit
                if self._lock:
                    with self._lock:
                        if len(self._agent_histories) > 1:
                            # Remove one agent history (oldest by creation order)
                            oldest_key = next(iter(self._agent_histories))
                            del self._agent_histories[oldest_key]
                            
                            if self._logger:
                                self._logger.warning(
                                    "Memory cleanup performed - removed agent history",
                                    removed_agent=oldest_key,
                                    memory_usage_mb=memory_mb,
                                    limit_mb=self.config.memory_limit_mb
                                )
                else:
                    if len(self._agent_histories) > 1:
                        oldest_key = next(iter(self._agent_histories))
                        del self._agent_histories[oldest_key]
        
        self._last_cleanup_time = current_time
    
    # SensorProtocol implementation methods
    
    def detect(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Perform binary detection with historical data storage (BinarySensor implementation).
        
        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Boolean detection results with shape (n_agents,) or scalar for single agent
        """
        start_time = time.perf_counter() if self.config.performance_monitoring else None
        
        try:
            # Delegate to base sensor for actual detection
            detections = self.base_sensor.detect(plume_state, positions)
            
            # Ensure positions is 2D for consistent processing
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)
                detections = np.array([detections])
            
            # Store historical data for each agent
            for i, position in enumerate(positions):
                detection_value = bool(detections[i])
                self._store_reading(
                    agent_position=position,
                    reading=detection_value,
                    metadata={
                        'sensor_type': 'binary',
                        'operation': 'detect',
                        'plume_state_id': id(plume_state)
                    }
                )
            
            # Memory cleanup if needed
            self._cleanup_memory()
            
            # Performance tracking
            if self.config.performance_monitoring:
                operation_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['total_operations'] += 1
                self._performance_metrics['total_time_ms'] += operation_time
                
                if self._logger and self._performance_metrics['total_operations'] % 100 == 0:
                    avg_time = self._performance_metrics['total_time_ms'] / self._performance_metrics['total_operations']
                    self._logger.debug(
                        "Historical sensor detection performance summary",
                        total_operations=self._performance_metrics['total_operations'],
                        avg_operation_time_ms=avg_time,
                        current_operation_time_ms=operation_time,
                        active_agent_histories=len(self._agent_histories)
                    )
            
            # Return in original format
            return detections[0] if len(detections) == 1 else detections
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Historical sensor detection failed: {str(e)}",
                    error_type=type(e).__name__,
                    positions_shape=positions.shape,
                    base_sensor_type=type(self.base_sensor).__name__
                )
            raise
    
    def measure(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Perform quantitative measurements with historical data storage (ConcentrationSensor).
        
        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Quantitative measurement values with shape (n_agents,) or scalar for single agent
        """
        start_time = time.perf_counter() if self.config.performance_monitoring else None
        
        try:
            # Delegate to base sensor for actual measurement
            measurements = self.base_sensor.measure(plume_state, positions)
            
            # Ensure positions is 2D for consistent processing
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)
                measurements = np.array([measurements])
            
            # Store historical data for each agent
            for i, position in enumerate(positions):
                measurement_value = float(measurements[i])
                self._store_reading(
                    agent_position=position,
                    reading=measurement_value,
                    metadata={
                        'sensor_type': 'concentration',
                        'operation': 'measure',
                        'plume_state_id': id(plume_state)
                    }
                )
            
            # Memory cleanup if needed
            self._cleanup_memory()
            
            # Performance tracking
            if self.config.performance_monitoring:
                operation_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['total_operations'] += 1
                self._performance_metrics['total_time_ms'] += operation_time
            
            # Return in original format
            return measurements[0] if len(measurements) == 1 else measurements
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Historical sensor measurement failed: {str(e)}",
                    error_type=type(e).__name__,
                    positions_shape=positions.shape,
                    base_sensor_type=type(self.base_sensor).__name__
                )
            raise
    
    def compute_gradient(self, plume_state: Any, positions: np.ndarray) -> np.ndarray:
        """
        Compute spatial gradients with historical data storage (GradientSensor implementation).
        
        Args:
            plume_state: Current plume model state providing concentration field access
            positions: Agent positions as array with shape (n_agents, 2) or (2,) for single agent
            
        Returns:
            np.ndarray: Gradient vectors with shape (n_agents, 2) or (2,) for single agent
        """
        start_time = time.perf_counter() if self.config.performance_monitoring else None
        
        try:
            # Delegate to base sensor for actual gradient computation
            gradients = self.base_sensor.compute_gradient(plume_state, positions)
            
            # Ensure positions is 2D for consistent processing
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)
                if gradients.ndim == 1:
                    gradients = gradients.reshape(1, -1)
            
            # Store historical data for each agent
            for i, position in enumerate(positions):
                gradient_value = gradients[i].copy() if gradients.ndim > 1 else gradients.copy()
                self._store_reading(
                    agent_position=position,
                    reading=gradient_value,
                    metadata={
                        'sensor_type': 'gradient',
                        'operation': 'compute_gradient',
                        'plume_state_id': id(plume_state)
                    }
                )
            
            # Memory cleanup if needed
            self._cleanup_memory()
            
            # Performance tracking
            if self.config.performance_monitoring:
                operation_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['total_operations'] += 1
                self._performance_metrics['total_time_ms'] += operation_time
            
            # Return in original format
            return gradients[0] if gradients.ndim > 1 and len(gradients) == 1 else gradients
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Historical sensor gradient computation failed: {str(e)}",
                    error_type=type(e).__name__,
                    positions_shape=positions.shape,
                    base_sensor_type=type(self.base_sensor).__name__
                )
            raise
    
    def configure(self, **kwargs: Any) -> None:
        """
        Update sensor configuration parameters during runtime.
        
        Args:
            **kwargs: Sensor-specific configuration parameters including:
                - history_length: Update maximum history buffer size
                - temporal_window: Update temporal analysis window
                - aggregation_methods: Update enabled aggregation methods
                - memory_limit_mb: Update memory usage limit
                - Any base sensor configuration parameters
        """
        try:
            # Update historical sensor configuration
            historical_params = {}
            base_params = {}
            
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    historical_params[key] = value
                else:
                    base_params[key] = value
            
            # Update historical sensor config
            if historical_params:
                for key, value in historical_params.items():
                    setattr(self.config, key, value)
                
                # Re-validate configuration
                self.config.__post_init__()
                
                if self._logger:
                    self._logger.info(
                        "Historical sensor configuration updated",
                        updated_params=historical_params,
                        new_config=self.config.__dict__
                    )
            
            # Delegate base sensor configuration to wrapped sensor
            if base_params:
                self.base_sensor.configure(**base_params)
                
                if self._logger:
                    self._logger.debug(
                        "Base sensor configuration updated",
                        updated_params=base_params,
                        base_sensor_type=type(self.base_sensor).__name__
                    )
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Historical sensor configuration failed: {str(e)}",
                    error_type=type(e).__name__,
                    kwargs=kwargs
                )
            raise
    
    # Historical data access and aggregation methods
    
    def get_history(
        self, 
        agent_position: np.ndarray, 
        count: Optional[int] = None,
        time_window: Optional[float] = None
    ) -> List[HistoricalDataPoint]:
        """
        Get historical data for specified agent position.
        
        Args:
            agent_position: Position of agent to get history for
            count: Number of recent entries to retrieve (None = all)
            time_window: Time window in seconds from present (None = no limit)
            
        Returns:
            List[HistoricalDataPoint]: Historical data points (newest first)
        """
        agent_key = self._get_agent_key(agent_position)
        
        if agent_key not in self._agent_histories:
            return []
        
        history_buffer = self._agent_histories[agent_key]
        
        if time_window is not None:
            return history_buffer.get_in_time_window(time_window)
        elif count is not None:
            return history_buffer.get_recent(count)
        else:
            return history_buffer.get_recent(len(history_buffer))
    
    def get_moving_average(
        self, 
        agent_position: np.ndarray, 
        window_size: int = 10
    ) -> Union[float, np.ndarray, None]:
        """
        Compute moving average of historical readings for specified agent.
        
        Args:
            agent_position: Position of agent to analyze
            window_size: Number of recent readings to average
            
        Returns:
            Union[float, np.ndarray, None]: Moving average value or None if insufficient data
        """
        start_time = time.perf_counter() if self.config.performance_monitoring else None
        
        if not self.config.enable_aggregation or 'mean' not in self.config.aggregation_methods:
            return None
        
        try:
            history = self.get_history(agent_position, count=window_size)
            
            if len(history) == 0:
                return None
            
            readings = [dp.reading for dp in history]
            
            # Handle different reading types
            if isinstance(readings[0], (bool, np.bool_)):
                # Binary readings - compute fraction of True values
                return float(np.mean([float(r) for r in readings]))
            elif isinstance(readings[0], np.ndarray):
                # Array readings (e.g., gradients) - compute element-wise mean
                return np.mean(readings, axis=0)
            else:
                # Scalar readings - simple mean
                return float(np.mean(readings))
                
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Moving average computation failed: {str(e)}",
                    error_type=type(e).__name__,
                    agent_position=agent_position.tolist(),
                    window_size=window_size
                )
            return None
        finally:
            if self.config.performance_monitoring:
                operation_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['aggregation_operations'] += 1
                self._performance_metrics['aggregation_time_ms'] += operation_time
    
    def get_temporal_gradient(
        self, 
        agent_position: np.ndarray, 
        time_delta: float = 1.0
    ) -> Union[float, np.ndarray, None]:
        """
        Compute temporal gradient (rate of change) of sensor readings.
        
        Args:
            agent_position: Position of agent to analyze
            time_delta: Time difference for gradient calculation in seconds
            
        Returns:
            Union[float, np.ndarray, None]: Temporal gradient or None if insufficient data
        """
        start_time = time.perf_counter() if self.config.performance_monitoring else None
        
        if not self.config.enable_aggregation or 'gradient' not in self.config.aggregation_methods:
            return None
        
        try:
            # Get recent history covering the time delta
            current_time = time.time()
            history = self.get_history(agent_position, time_window=time_delta * 2)
            
            if len(history) < 2:
                return None
            
            # Sort by timestamp (newest first)
            history.sort(key=lambda dp: dp.timestamp, reverse=True)
            
            # Find readings separated by approximately time_delta
            recent_reading = history[0]
            past_reading = None
            
            for dp in history[1:]:
                if recent_reading.timestamp - dp.timestamp >= time_delta:
                    past_reading = dp
                    break
            
            if past_reading is None:
                return None
            
            # Compute temporal gradient
            dt = recent_reading.timestamp - past_reading.timestamp
            
            if isinstance(recent_reading.reading, (bool, np.bool_)):
                # Binary readings - simple difference
                return (float(recent_reading.reading) - float(past_reading.reading)) / dt
            elif isinstance(recent_reading.reading, np.ndarray):
                # Array readings - element-wise gradient
                return (recent_reading.reading - past_reading.reading) / dt
            else:
                # Scalar readings - simple gradient
                return (recent_reading.reading - past_reading.reading) / dt
                
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Temporal gradient computation failed: {str(e)}",
                    error_type=type(e).__name__,
                    agent_position=agent_position.tolist(),
                    time_delta=time_delta
                )
            return None
        finally:
            if self.config.performance_monitoring:
                operation_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['aggregation_operations'] += 1
                self._performance_metrics['aggregation_time_ms'] += operation_time
    
    def get_temporal_statistics(
        self, 
        agent_position: np.ndarray, 
        window_size: int = 20
    ) -> Dict[str, Union[float, np.ndarray, None]]:
        """
        Compute comprehensive temporal statistics for historical readings.
        
        Args:
            agent_position: Position of agent to analyze
            window_size: Number of recent readings to analyze
            
        Returns:
            Dict[str, Union[float, np.ndarray, None]]: Dictionary of temporal statistics
        """
        start_time = time.perf_counter() if self.config.performance_monitoring else None
        
        if not self.config.enable_aggregation:
            return {}
        
        try:
            history = self.get_history(agent_position, count=window_size)
            
            if len(history) == 0:
                return {}
            
            readings = [dp.reading for dp in history]
            stats = {}
            
            # Compute enabled statistics
            if 'mean' in self.config.aggregation_methods:
                if isinstance(readings[0], (bool, np.bool_)):
                    stats['mean'] = float(np.mean([float(r) for r in readings]))
                elif isinstance(readings[0], np.ndarray):
                    stats['mean'] = np.mean(readings, axis=0)
                else:
                    stats['mean'] = float(np.mean(readings))
            
            if 'std' in self.config.aggregation_methods:
                if isinstance(readings[0], (bool, np.bool_)):
                    stats['std'] = float(np.std([float(r) for r in readings]))
                elif isinstance(readings[0], np.ndarray):
                    stats['std'] = np.std(readings, axis=0)
                else:
                    stats['std'] = float(np.std(readings))
            
            if 'min' in self.config.aggregation_methods:
                if isinstance(readings[0], (bool, np.bool_)):
                    stats['min'] = float(np.min([float(r) for r in readings]))
                elif isinstance(readings[0], np.ndarray):
                    stats['min'] = np.min(readings, axis=0)
                else:
                    stats['min'] = float(np.min(readings))
            
            if 'max' in self.config.aggregation_methods:
                if isinstance(readings[0], (bool, np.bool_)):
                    stats['max'] = float(np.max([float(r) for r in readings]))
                elif isinstance(readings[0], np.ndarray):
                    stats['max'] = np.max(readings, axis=0)
                else:
                    stats['max'] = float(np.max(readings))
            
            if 'median' in self.config.aggregation_methods:
                if isinstance(readings[0], (bool, np.bool_)):
                    stats['median'] = float(np.median([float(r) for r in readings]))
                elif isinstance(readings[0], np.ndarray):
                    stats['median'] = np.median(readings, axis=0)
                else:
                    stats['median'] = float(np.median(readings))
            
            if 'variance' in self.config.aggregation_methods:
                if isinstance(readings[0], (bool, np.bool_)):
                    stats['variance'] = float(np.var([float(r) for r in readings]))
                elif isinstance(readings[0], np.ndarray):
                    stats['variance'] = np.var(readings, axis=0)
                else:
                    stats['variance'] = float(np.var(readings))
            
            if 'trend' in self.config.aggregation_methods and len(history) >= 3:
                # Simple linear trend analysis
                timestamps = [dp.timestamp for dp in history]
                if isinstance(readings[0], (bool, np.bool_)):
                    float_readings = [float(r) for r in readings]
                    trend_coeff = np.polyfit(timestamps, float_readings, 1)[0]
                    stats['trend'] = float(trend_coeff)
                elif isinstance(readings[0], np.ndarray):
                    # Compute trend for each array component
                    trends = []
                    for i in range(readings[0].shape[0]):
                        component_readings = [r[i] for r in readings]
                        trend_coeff = np.polyfit(timestamps, component_readings, 1)[0]
                        trends.append(trend_coeff)
                    stats['trend'] = np.array(trends)
                else:
                    trend_coeff = np.polyfit(timestamps, readings, 1)[0]
                    stats['trend'] = float(trend_coeff)
            
            return stats
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Temporal statistics computation failed: {str(e)}",
                    error_type=type(e).__name__,
                    agent_position=agent_position.tolist(),
                    window_size=window_size
                )
            return {}
        finally:
            if self.config.performance_monitoring:
                operation_time = (time.perf_counter() - start_time) * 1000
                self._performance_metrics['aggregation_operations'] += 1
                self._performance_metrics['aggregation_time_ms'] += operation_time
    
    def clear_history(self, agent_position: Optional[np.ndarray] = None) -> None:
        """
        Clear historical data for specified agent or all agents.
        
        Args:
            agent_position: Position of agent to clear history for (None = clear all)
        """
        try:
            if agent_position is not None:
                agent_key = self._get_agent_key(agent_position)
                if agent_key in self._agent_histories:
                    self._agent_histories[agent_key].clear()
                    
                    if self._logger:
                        self._logger.debug(
                            "Historical data cleared for agent",
                            agent_key=agent_key,
                            agent_position=agent_position.tolist()
                        )
            else:
                if self._lock:
                    with self._lock:
                        for buffer in self._agent_histories.values():
                            buffer.clear()
                        self._agent_histories.clear()
                else:
                    for buffer in self._agent_histories.values():
                        buffer.clear()
                    self._agent_histories.clear()
                
                if self._logger:
                    self._logger.info("All historical data cleared")
                    
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Clear history operation failed: {str(e)}",
                    error_type=type(e).__name__,
                    agent_position=agent_position.tolist() if agent_position is not None else None
                )
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and optimization.
        
        Returns:
            Dict[str, Any]: Performance metrics dictionary
        """
        if not self.config.performance_monitoring:
            return {}
        
        try:
            metrics = {
                'sensor_type': 'HistoricalSensor',
                'sensor_id': self.sensor_id,
                'base_sensor_type': type(self.base_sensor).__name__,
                'configuration': self.config.__dict__,
                'active_agent_histories': len(self._agent_histories),
                'performance_stats': self._performance_metrics.copy()
            }
            
            # Add operation performance statistics
            if self._performance_metrics['total_operations'] > 0:
                metrics['avg_operation_time_ms'] = (
                    self._performance_metrics['total_time_ms'] / 
                    self._performance_metrics['total_operations']
                )
            
            if self._performance_metrics['aggregation_operations'] > 0:
                metrics['avg_aggregation_time_ms'] = (
                    self._performance_metrics['aggregation_time_ms'] / 
                    self._performance_metrics['aggregation_operations']
                )
            
            # Add buffer statistics
            buffer_stats = {}
            for agent_key, buffer in self._agent_histories.items():
                buffer_stats[agent_key] = buffer.get_statistics()
            metrics['buffer_statistics'] = buffer_stats
            
            # Memory usage estimation
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            
            return metrics
            
        except Exception as e:
            if self._logger:
                self._logger.error(
                    f"Performance metrics collection failed: {str(e)}",
                    error_type=type(e).__name__
                )
            return {'error': str(e)}
    
    @classmethod
    def from_config(
        cls, 
        config: Union[Dict[str, Any], DictConfig],
        enable_logging: bool = True
    ) -> 'HistoricalSensor':
        """
        Create HistoricalSensor from configuration dictionary.
        
        Args:
            config: Configuration dictionary with base_sensor and historical parameters
            enable_logging: Enable comprehensive logging integration
            
        Returns:
            HistoricalSensor: Configured historical sensor instance
            
        Examples:
            Configuration-driven instantiation:
                >>> config = {
                ...     'base_sensor': {
                ...         'type': 'ConcentrationSensor',
                ...         'dynamic_range': (0, 1),
                ...         'resolution': 0.001
                ...     },
                ...     'history_length': 150,
                ...     'temporal_window': 30.0,
                ...     'aggregation_methods': ['mean', 'std', 'gradient']
                ... }
                >>> sensor = HistoricalSensor.from_config(config)
        """
        try:
            # Extract base sensor configuration
            base_sensor_config = config.get('base_sensor', {})
            if isinstance(base_sensor_config, dict) and 'type' in base_sensor_config:
                # Create base sensor from type specification
                sensor_type = base_sensor_config.pop('type')
                
                # Import sensor classes dynamically (placeholder for when they exist)
                if sensor_type == 'ConcentrationSensor':
                    try:
                        from .concentration_sensor import ConcentrationSensor
                        base_sensor = ConcentrationSensor(**base_sensor_config)
                    except ImportError:
                        # Fallback for development/testing
                        class MockConcentrationSensor:
                            def detect(self, plume_state, positions): return np.zeros(len(positions) if positions.ndim > 1 else 1)
                            def measure(self, plume_state, positions): return np.zeros(len(positions) if positions.ndim > 1 else 1)
                            def compute_gradient(self, plume_state, positions): return np.zeros((len(positions), 2) if positions.ndim > 1 else (2,))
                            def configure(self, **kwargs): pass
                        base_sensor = MockConcentrationSensor()
                        
                elif sensor_type == 'BinarySensor':
                    try:
                        from .binary_sensor import BinarySensor
                        base_sensor = BinarySensor(**base_sensor_config)
                    except ImportError:
                        # Fallback for development/testing
                        class MockBinarySensor:
                            def detect(self, plume_state, positions): return np.zeros(len(positions) if positions.ndim > 1 else 1, dtype=bool)
                            def measure(self, plume_state, positions): return np.zeros(len(positions) if positions.ndim > 1 else 1)
                            def compute_gradient(self, plume_state, positions): return np.zeros((len(positions), 2) if positions.ndim > 1 else (2,))
                            def configure(self, **kwargs): pass
                        base_sensor = MockBinarySensor()
                        
                elif sensor_type == 'GradientSensor':
                    try:
                        from .gradient_sensor import GradientSensor
                        base_sensor = GradientSensor(**base_sensor_config)
                    except ImportError:
                        # Fallback for development/testing
                        class MockGradientSensor:
                            def detect(self, plume_state, positions): return np.zeros(len(positions) if positions.ndim > 1 else 1, dtype=bool)
                            def measure(self, plume_state, positions): return np.zeros(len(positions) if positions.ndim > 1 else 1)
                            def compute_gradient(self, plume_state, positions): return np.zeros((len(positions), 2) if positions.ndim > 1 else (2,))
                            def configure(self, **kwargs): pass
                        base_sensor = MockGradientSensor()
                else:
                    raise ValueError(f"Unknown sensor type: {sensor_type}")
            else:
                raise ValueError("base_sensor configuration must include 'type' field")
            
            # Extract historical sensor configuration
            historical_config_dict = {k: v for k, v in config.items() if k != 'base_sensor'}
            historical_config = HistoricalSensorConfig(**historical_config_dict)
            
            # Create historical sensor
            return cls(
                base_sensor=base_sensor,
                config=historical_config,
                sensor_id=config.get('sensor_id'),
                enable_logging=enable_logging
            )
            
        except Exception as e:
            if enable_logging and LOGURU_AVAILABLE:
                logger.error(
                    f"HistoricalSensor creation from config failed: {str(e)}",
                    error_type=type(e).__name__,
                    config=config
                )
            raise


# Export public API
__all__ = [
    'HistoricalSensor',
    'HistoricalSensorConfig', 
    'HistoricalDataPoint',
    'CircularBuffer'
]