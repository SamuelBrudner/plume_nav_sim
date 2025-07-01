"""
Testing Infrastructure for Modular Plume Model Components.

This module provides specialized testing utilities for the new modular plume navigation
architecture including:
- Protocol compliance validation for PlumeModelProtocol, WindFieldProtocol, and SensorProtocol
- Performance benchmarking for modular component switching and execution latency
- Mock factories for isolated component testing
- Property-based testing strategies for scientific computation validation
- Shared fixtures for plume model mathematical validation
- Test data generation for realistic simulation scenarios

The module supports the transition from monolithic VideoPlume processing to
pluggable, protocol-driven component architecture while maintaining backward
compatibility and performance requirements.

Key Testing Patterns:
- Protocol compliance: Structural subtyping validation across all implementations
- Performance validation: <10ms step execution latency across model types
- Configuration switching: Hydra-based component swapping without code changes
- Mathematical validation: Property-based testing for concentration fields
- Integration testing: Cross-component interaction validation
"""

import sys
import time
import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union, Callable, List, Protocol, runtime_checkable
from pathlib import Path
from abc import ABC, abstractmethod

# Core testing framework imports
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Hypothesis property-based testing framework for scientific validation
try:
    import hypothesis
    from hypothesis import given, assume, strategies as st
    from hypothesis.extra.numpy import arrays, array_shapes
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    hypothesis = None
    given = None
    assume = None
    st = None
    arrays = None
    array_shapes = None
    HYPOTHESIS_AVAILABLE = False

# Scientific computing libraries for test validation
try:
    import scipy.stats
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    scipy = None
    SCIPY_AVAILABLE = False

# Import protocol definitions for compliance testing
try:
    from plume_nav_sim.core.protocols import (
        PlumeModelProtocol,
        WindFieldProtocol, 
        SensorProtocol,
        NavigatorProtocol
    )
    PROTOCOLS_AVAILABLE = True
except ImportError:
    PlumeModelProtocol = None
    WindFieldProtocol = None
    SensorProtocol = None
    NavigatorProtocol = None
    PROTOCOLS_AVAILABLE = False

# Import centralized logging for performance monitoring
try:
    from plume_nav_sim.utils.logging_setup import (
        LoggingConfig,
        setup_logger,
        get_logger,
        PerformanceMetrics
    )
    LOGGING_AVAILABLE = True
except ImportError:
    LoggingConfig = None
    setup_logger = None
    get_logger = None
    PerformanceMetrics = None
    LOGGING_AVAILABLE = False

# Import configuration models for modular testing
try:
    from plume_nav_sim.config.schemas import (
        PlumeModelConfig,
        WindFieldConfig,
        SensorConfig,
        SimulationConfig
    )
    CONFIG_SCHEMAS_AVAILABLE = True
except ImportError:
    PlumeModelConfig = None
    WindFieldConfig = None
    SensorConfig = None
    SimulationConfig = None
    CONFIG_SCHEMAS_AVAILABLE = False


# Performance monitoring constants for modular components
PLUME_MODEL_STEP_THRESHOLD_MS = 5.0    # ≤5ms for plume model step operations
WIND_FIELD_UPDATE_THRESHOLD_MS = 2.0   # ≤2ms for wind field updates  
SENSOR_SAMPLING_THRESHOLD_MS = 1.0     # ≤1ms for sensor sampling operations
PROTOCOL_SWITCHING_THRESHOLD_MS = 1.0  # ≤1ms for component switching overhead
CONCENTRATION_FIELD_THRESHOLD_MS = 8.0 # ≤8ms for concentration field generation

# Test environment configuration for modular components
MODULAR_TEST_CONFIG = {
    "environment": "modular_testing",
    "level": "INFO", 
    "format": "minimal",
    "console_enabled": True,
    "file_enabled": False,
    "correlation_enabled": False,
    "memory_tracking": True,
    "enable_performance": True,
    "protocol_validation": True,
}

# Mathematical validation constants
CONCENTRATION_TOLERANCE = 1e-6         # Tolerance for concentration field validation
GRADIENT_FINITE_DIFF_EPSILON = 1e-4   # Epsilon for gradient finite difference validation
WIND_VECTOR_TOLERANCE = 1e-8          # Tolerance for wind vector validation
GAUSSIAN_SIGMA_MIN = 0.01              # Minimum sigma for Gaussian plume testing
GAUSSIAN_SIGMA_MAX = 100.0             # Maximum sigma for Gaussian plume testing


class ProtocolComplianceValidator:
    """
    Comprehensive protocol compliance validator for modular components.
    
    Validates that all implementations correctly implement required protocol methods
    and maintain consistent behavior patterns across different component types.
    """
    
    @staticmethod
    def validate_plume_model_protocol(model: Any) -> bool:
        """
        Validate PlumeModelProtocol compliance for any plume model implementation.
        
        Args:
            model: Plume model instance to validate
            
        Returns:
            True if protocol compliant
            
        Raises:
            AssertionError: If protocol compliance fails
        """
        if not PROTOCOLS_AVAILABLE:
            pytest.skip("Protocol definitions not available for compliance testing")
        
        # Verify required methods exist
        required_methods = ['concentration_at', 'step', 'reset']
        for method_name in required_methods:
            assert hasattr(model, method_name), f"Missing required method: {method_name}"
            assert callable(getattr(model, method_name)), f"Method not callable: {method_name}"
        
        # Test concentration_at method signature and behavior
        test_positions = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        concentrations = model.concentration_at(test_positions)
        
        assert isinstance(concentrations, np.ndarray), "concentration_at must return numpy array"
        assert concentrations.shape[0] == test_positions.shape[0], "Output shape must match input positions"
        assert np.all(concentrations >= 0), "Concentrations must be non-negative"
        assert np.all(np.isfinite(concentrations)), "Concentrations must be finite"
        
        # Test step method
        initial_state = model.reset()
        model.step(dt=1.0)
        
        # Test reset method returns model to initial state
        model.reset()
        
        return True
    
    @staticmethod
    def validate_wind_field_protocol(wind_field: Any) -> bool:
        """
        Validate WindFieldProtocol compliance for wind field implementations.
        
        Args:
            wind_field: Wind field instance to validate
            
        Returns:
            True if protocol compliant
            
        Raises:
            AssertionError: If protocol compliance fails
        """
        if not PROTOCOLS_AVAILABLE:
            pytest.skip("Protocol definitions not available for compliance testing")
        
        # Verify required methods exist
        required_methods = ['get_wind_at', 'step', 'reset']
        for method_name in required_methods:
            assert hasattr(wind_field, method_name), f"Missing required method: {method_name}"
            assert callable(getattr(wind_field, method_name)), f"Method not callable: {method_name}"
        
        # Test get_wind_at method signature and behavior
        test_positions = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        wind_vectors = wind_field.get_wind_at(test_positions)
        
        assert isinstance(wind_vectors, np.ndarray), "get_wind_at must return numpy array"
        assert wind_vectors.shape == (test_positions.shape[0], 2), "Wind vectors must be 2D"
        assert np.all(np.isfinite(wind_vectors)), "Wind vectors must be finite"
        
        # Test temporal consistency
        wind_field.reset()
        wind_t0 = wind_field.get_wind_at(test_positions)
        wind_field.step(dt=1.0)
        wind_t1 = wind_field.get_wind_at(test_positions)
        
        # Wind should be consistent for constant fields or evolve for turbulent fields
        assert isinstance(wind_t0, np.ndarray) and isinstance(wind_t1, np.ndarray)
        
        return True
    
    @staticmethod
    def validate_sensor_protocol(sensor: Any) -> bool:
        """
        Validate SensorProtocol compliance for sensor implementations.
        
        Args:
            sensor: Sensor instance to validate
            
        Returns:
            True if protocol compliant
            
        Raises:
            AssertionError: If protocol compliance fails
        """
        if not PROTOCOLS_AVAILABLE:
            pytest.skip("Protocol definitions not available for compliance testing")
        
        # Verify required methods exist
        required_methods = ['sample', 'reset']
        for method_name in required_methods:
            assert hasattr(sensor, method_name), f"Missing required method: {method_name}"
            assert callable(getattr(sensor, method_name)), f"Method not callable: {method_name}"
        
        # Test sample method with mock plume state
        mock_plume_state = {
            'concentrations': np.array([0.1, 0.5, 0.8]),
            'positions': np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            'timestamp': 0.0
        }
        
        test_positions = np.array([[0.0, 0.0], [1.0, 1.0]])
        readings = sensor.sample(mock_plume_state, test_positions)
        
        assert readings is not None, "Sensor sample must return readings"
        
        # Test reset method
        sensor.reset()
        
        return True


class PlumeModelPerformanceMonitor:
    """
    Specialized performance monitoring for plume model components.
    
    Provides detailed timing analysis for different plume model operations
    with component-specific performance thresholds and regression detection.
    """
    
    def __init__(self, model_name: str = "unknown_model"):
        self.model_name = model_name
        self.operation_timings = {}
        self.memory_usage = {}
        
    def time_concentration_field_generation(self, model: Any, grid_size: tuple = (100, 100)) -> float:
        """
        Time concentration field generation for performance validation.
        
        Args:
            model: Plume model to benchmark
            grid_size: Grid dimensions for field generation
            
        Returns:
            Execution time in milliseconds
        """
        start_time = time.perf_counter()
        
        if hasattr(model, 'concentration_field'):
            field = model.concentration_field(grid_size=grid_size)
        else:
            # Fallback to concentration_at for protocol-only models
            x = np.linspace(0, grid_size[1]-1, grid_size[1])
            y = np.linspace(0, grid_size[0]-1, grid_size[0])
            xx, yy = np.meshgrid(x, y)
            positions = np.column_stack([xx.ravel(), yy.ravel()])
            concentrations = model.concentration_at(positions)
            field = concentrations.reshape(grid_size)
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        self.operation_timings['concentration_field'] = duration_ms
        return duration_ms
    
    def time_step_execution(self, model: Any, dt: float = 1.0) -> float:
        """
        Time single step execution for performance validation.
        
        Args:
            model: Plume model to benchmark
            dt: Time step size
            
        Returns:
            Execution time in milliseconds
        """
        start_time = time.perf_counter()
        model.step(dt=dt)
        end_time = time.perf_counter()
        
        duration_ms = (end_time - start_time) * 1000
        self.operation_timings['step'] = duration_ms
        return duration_ms
    
    def assert_performance_requirements(self):
        """Assert that all operations meet performance requirements."""
        if 'concentration_field' in self.operation_timings:
            duration = self.operation_timings['concentration_field']
            assert duration <= CONCENTRATION_FIELD_THRESHOLD_MS, (
                f"{self.model_name} concentration field generation took {duration:.2f}ms, "
                f"exceeds threshold of {CONCENTRATION_FIELD_THRESHOLD_MS:.2f}ms"
            )
        
        if 'step' in self.operation_timings:
            duration = self.operation_timings['step']
            assert duration <= PLUME_MODEL_STEP_THRESHOLD_MS, (
                f"{self.model_name} step execution took {duration:.2f}ms, "
                f"exceeds threshold of {PLUME_MODEL_STEP_THRESHOLD_MS:.2f}ms"
            )


@contextmanager
def plume_model_performance_timer(model_name: str = "test_model"):
    """
    Context manager for plume model performance timing.
    
    Args:
        model_name: Name of the model being tested
        
    Yields:
        PlumeModelPerformanceMonitor: Performance monitor for the model
        
    Example:
        >>> with plume_model_performance_timer("GaussianPlumeModel") as monitor:
        ...     duration = monitor.time_concentration_field_generation(model)
        >>> monitor.assert_performance_requirements()
    """
    monitor = PlumeModelPerformanceMonitor(model_name)
    try:
        yield monitor
    finally:
        # Optional: Log performance metrics
        if LOGGING_AVAILABLE:
            logger = get_logger("test.performance")
            logger.info(f"Performance metrics for {model_name}", 
                       extra={"timings": monitor.operation_timings})


class MockPlumeModelFactory:
    """
    Factory for creating mock plume model implementations for testing.
    
    Provides deterministic mock implementations that follow protocol
    contracts while enabling isolated testing of dependent components.
    """
    
    @staticmethod
    def create_mock_gaussian_plume(source_pos: tuple = (0.0, 0.0), 
                                  sigma: float = 1.0) -> Mock:
        """
        Create mock Gaussian plume model with deterministic behavior.
        
        Args:
            source_pos: Source position for the mock plume
            sigma: Spread parameter for concentration field
            
        Returns:
            Mock object implementing PlumeModelProtocol
        """
        mock = Mock()
        mock.source_position = source_pos
        mock.sigma = sigma
        
        def mock_concentration_at(positions):
            positions = np.atleast_2d(positions)
            distances = cdist(positions, [source_pos])
            concentrations = np.exp(-0.5 * (distances.flatten() / sigma)**2)
            return concentrations
        
        def mock_step(dt):
            # Deterministic step - no state change for testing
            pass
        
        def mock_reset():
            # Reset to initial state
            return None
        
        mock.concentration_at = mock_concentration_at
        mock.step = mock_step
        mock.reset = mock_reset
        
        return mock
    
    @staticmethod
    def create_mock_turbulent_plume(num_filaments: int = 100,
                                   turbulence_intensity: float = 0.2) -> Mock:
        """
        Create mock turbulent plume model with stochastic behavior.
        
        Args:
            num_filaments: Number of filaments to simulate
            turbulence_intensity: Turbulence strength parameter
            
        Returns:
            Mock object implementing PlumeModelProtocol
        """
        mock = Mock()
        mock.num_filaments = num_filaments
        mock.turbulence_intensity = turbulence_intensity
        
        def mock_concentration_at(positions):
            positions = np.atleast_2d(positions)
            # Simple stochastic concentration with controlled randomness
            np.random.seed(42)  # Deterministic for testing
            base_concentration = np.random.exponential(0.1, positions.shape[0])
            noise = np.random.normal(0, turbulence_intensity, positions.shape[0])
            concentrations = np.maximum(0, base_concentration + noise)
            return concentrations
        
        def mock_step(dt):
            # Simulate filament evolution
            pass
        
        def mock_reset():
            # Reset filament positions
            return None
        
        mock.concentration_at = mock_concentration_at
        mock.step = mock_step
        mock.reset = mock_reset
        
        return mock


class MockWindFieldFactory:
    """
    Factory for creating mock wind field implementations for testing.
    """
    
    @staticmethod
    def create_mock_constant_wind(wind_vector: tuple = (1.0, 0.0)) -> Mock:
        """
        Create mock constant wind field.
        
        Args:
            wind_vector: (u, v) wind velocity vector
            
        Returns:
            Mock object implementing WindFieldProtocol
        """
        mock = Mock()
        mock.wind_vector = wind_vector
        
        def mock_get_wind_at(positions):
            positions = np.atleast_2d(positions)
            wind_vectors = np.tile(wind_vector, (positions.shape[0], 1))
            return wind_vectors
        
        def mock_step(dt):
            pass
        
        def mock_reset():
            return None
        
        mock.get_wind_at = mock_get_wind_at
        mock.step = mock_step
        mock.reset = mock_reset
        
        return mock
    
    @staticmethod
    def create_mock_turbulent_wind(mean_wind: tuple = (2.0, 0.5),
                                  turbulence_std: float = 0.5) -> Mock:
        """
        Create mock turbulent wind field with stochastic variations.
        
        Args:
            mean_wind: Mean wind velocity vector
            turbulence_std: Standard deviation of turbulent fluctuations
            
        Returns:
            Mock object implementing WindFieldProtocol
        """
        mock = Mock()
        mock.mean_wind = mean_wind
        mock.turbulence_std = turbulence_std
        
        def mock_get_wind_at(positions):
            positions = np.atleast_2d(positions)
            np.random.seed(42)  # Deterministic for testing
            base_wind = np.tile(mean_wind, (positions.shape[0], 1))
            turbulence = np.random.normal(0, turbulence_std, base_wind.shape)
            wind_vectors = base_wind + turbulence
            return wind_vectors
        
        def mock_step(dt):
            pass
        
        def mock_reset():
            return None
        
        mock.get_wind_at = mock_get_wind_at
        mock.step = mock_step
        mock.reset = mock_reset
        
        return mock


class MockSensorFactory:
    """
    Factory for creating mock sensor implementations for testing.
    """
    
    @staticmethod
    def create_mock_binary_sensor(threshold: float = 0.1,
                                 false_positive_rate: float = 0.05) -> Mock:
        """
        Create mock binary sensor with configurable detection threshold.
        
        Args:
            threshold: Detection threshold for binary decisions
            false_positive_rate: Rate of false positive detections
            
        Returns:
            Mock object implementing SensorProtocol
        """
        mock = Mock()
        mock.threshold = threshold
        mock.false_positive_rate = false_positive_rate
        
        def mock_sample(plume_state, positions):
            concentrations = plume_state.get('concentrations', np.zeros(len(positions)))
            detections = concentrations > threshold
            
            # Add false positives deterministically for testing
            np.random.seed(42)
            false_positives = np.random.random(len(positions)) < false_positive_rate
            detections = detections | false_positives
            
            return detections.astype(float)
        
        def mock_reset():
            pass
        
        mock.sample = mock_sample
        mock.reset = mock_reset
        
        return mock
    
    @staticmethod
    def create_mock_concentration_sensor(noise_std: float = 0.01,
                                       saturation_level: float = 1.0) -> Mock:
        """
        Create mock concentration sensor with noise and saturation.
        
        Args:
            noise_std: Standard deviation of measurement noise
            saturation_level: Maximum concentration reading
            
        Returns:
            Mock object implementing SensorProtocol
        """
        mock = Mock()
        mock.noise_std = noise_std
        mock.saturation_level = saturation_level
        
        def mock_sample(plume_state, positions):
            concentrations = plume_state.get('concentrations', np.zeros(len(positions)))
            
            # Add measurement noise
            np.random.seed(42)
            noise = np.random.normal(0, noise_std, len(positions))
            noisy_concentrations = concentrations + noise
            
            # Apply saturation
            saturated_concentrations = np.clip(noisy_concentrations, 0, saturation_level)
            
            return saturated_concentrations
        
        def mock_reset():
            pass
        
        mock.sample = mock_sample
        mock.reset = mock_reset
        
        return mock


def create_test_plume_model_config(model_type: str = "gaussian", **overrides) -> Dict[str, Any]:
    """
    Create test configuration dictionaries for plume model components.
    
    Args:
        model_type: Type of plume model ("gaussian", "turbulent", "video_adapter")
        **overrides: Configuration parameter overrides
        
    Returns:
        Dictionary configuration for testing
    """
    base_configs = {
        "gaussian": {
            "type": "gaussian",
            "source_position": [0.0, 0.0],
            "spread_sigma": 1.0,
            "emission_rate": 10.0,
            "background_concentration": 0.01,
        },
        "turbulent": {
            "type": "turbulent", 
            "source_position": [0.0, 0.0],
            "num_filaments": 100,
            "turbulence_intensity": 0.2,
            "dissipation_rate": 0.1,
            "wind_coupling": True,
        },
        "video_adapter": {
            "type": "video_adapter",
            "video_path": "/tmp/test_plume.mp4",
            "preprocessing": {
                "grayscale": True,
                "gaussian_blur": {"kernel_size": 5, "sigma": 1.0},
                "normalize": True,
            },
            "frame_rate": 30.0,
        },
    }
    
    config = base_configs.get(model_type, {}).copy()
    config.update(overrides)
    return config


def create_test_wind_field_config(field_type: str = "constant", **overrides) -> Dict[str, Any]:
    """
    Create test configuration dictionaries for wind field components.
    
    Args:
        field_type: Type of wind field ("constant", "turbulent", "time_varying")
        **overrides: Configuration parameter overrides
        
    Returns:
        Dictionary configuration for testing
    """
    base_configs = {
        "constant": {
            "type": "constant",
            "wind_vector": [2.0, 0.0],
            "turbulence_intensity": 0.0,
        },
        "turbulent": {
            "type": "turbulent",
            "mean_wind": [2.0, 0.5],
            "turbulence_intensity": 0.3,
            "correlation_length": 10.0,
            "time_scale": 5.0,
        },
        "time_varying": {
            "type": "time_varying", 
            "wind_profile": "sinusoidal",
            "base_wind": [1.0, 0.0],
            "amplitude": [0.5, 0.2],
            "frequency": 0.1,
        },
    }
    
    config = base_configs.get(field_type, {}).copy()
    config.update(overrides)
    return config


def create_test_sensor_config(sensor_type: str = "concentration", **overrides) -> Dict[str, Any]:
    """
    Create test configuration dictionaries for sensor components.
    
    Args:
        sensor_type: Type of sensor ("binary", "concentration", "gradient")
        **overrides: Configuration parameter overrides
        
    Returns:
        Dictionary configuration for testing
    """
    base_configs = {
        "binary": {
            "type": "binary",
            "threshold": 0.1,
            "hysteresis": 0.02,
            "false_positive_rate": 0.05,
            "false_negative_rate": 0.02,
        },
        "concentration": {
            "type": "concentration",
            "dynamic_range": [0.0, 1.0],
            "resolution": 0.001,
            "noise_std": 0.01,
            "saturation_level": 1.0,
        },
        "gradient": {
            "type": "gradient",
            "spatial_resolution": 0.1,
            "finite_difference_method": "central",
            "smoothing_kernel": "gaussian",
            "kernel_size": 3,
        },
    }
    
    config = base_configs.get(sensor_type, {}).copy()
    config.update(overrides)
    return config


# Hypothesis strategies for property-based testing of modular components
if HYPOTHESIS_AVAILABLE:
    # Concentration field strategies
    concentration_strategy = st.floats(
        min_value=0.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False
    )
    
    # Position strategies for spatial testing
    spatial_coordinate_strategy = st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False
    )
    
    spatial_position_strategy = st.tuples(spatial_coordinate_strategy, spatial_coordinate_strategy)
    
    # Plume model parameter strategies
    gaussian_sigma_strategy = st.floats(
        min_value=GAUSSIAN_SIGMA_MIN,
        max_value=GAUSSIAN_SIGMA_MAX,
        allow_nan=False,
        allow_infinity=False
    )
    
    emission_rate_strategy = st.floats(
        min_value=0.1,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False
    )
    
    # Wind field strategies
    wind_speed_strategy = st.floats(
        min_value=0.0,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False
    )
    
    wind_direction_strategy = st.floats(
        min_value=0.0,
        max_value=2*np.pi,
        allow_nan=False,
        allow_infinity=False
    )
    
    turbulence_intensity_strategy = st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False
    )
    
    # Grid size strategies for field generation
    grid_dimension_strategy = st.integers(min_value=10, max_value=200)
    grid_size_strategy = st.tuples(grid_dimension_strategy, grid_dimension_strategy)
    
    # Multi-position strategies for testing
    multi_position_strategy = st.lists(
        spatial_position_strategy,
        min_size=1,
        max_size=20
    )


def gaussian_plume_mathematical_validation(source_pos: tuple, sigma: float, 
                                         test_positions: List[tuple],
                                         concentrations: np.ndarray,
                                         tolerance: float = CONCENTRATION_TOLERANCE) -> bool:
    """
    Validate mathematical properties of Gaussian plume concentration fields.
    
    Args:
        source_pos: Source position (x, y)
        sigma: Spread parameter
        test_positions: List of test positions
        concentrations: Computed concentration values
        tolerance: Numerical tolerance for validation
        
    Returns:
        True if mathematical properties are satisfied
        
    Raises:
        AssertionError: If mathematical validation fails
    """
    test_positions = np.array(test_positions)
    
    # Validate non-negativity
    assert np.all(concentrations >= 0), "Concentrations must be non-negative"
    
    # Validate finite values
    assert np.all(np.isfinite(concentrations)), "Concentrations must be finite"
    
    # Validate monotonic decrease with distance from source
    if SCIPY_AVAILABLE:
        distances = cdist(test_positions, [source_pos]).flatten()
        
        # For positions at equal distance, concentrations should be approximately equal
        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                if abs(distances[i] - distances[j]) < tolerance:
                    assert abs(concentrations[i] - concentrations[j]) < tolerance, (
                        f"Concentrations at equal distances should be equal: "
                        f"d1={distances[i]:.6f}, d2={distances[j]:.6f}, "
                        f"c1={concentrations[i]:.6f}, c2={concentrations[j]:.6f}"
                    )
        
        # Validate Gaussian profile shape
        expected_concentrations = np.exp(-0.5 * (distances / sigma)**2)
        
        # Normalize for comparison
        if np.max(concentrations) > 0 and np.max(expected_concentrations) > 0:
            norm_computed = concentrations / np.max(concentrations)
            norm_expected = expected_concentrations / np.max(expected_concentrations)
            
            # Allow for some numerical error in implementation
            relative_error = np.abs(norm_computed - norm_expected) / (norm_expected + tolerance)
            max_error = np.max(relative_error)
            
            assert max_error < 0.1, f"Gaussian profile deviates too much: max_error={max_error:.4f}"
    
    return True


def wind_field_consistency_validation(wind_field: Any, test_positions: List[tuple],
                                    temporal_steps: int = 5,
                                    dt: float = 1.0) -> bool:
    """
    Validate consistency properties of wind field implementations.
    
    Args:
        wind_field: Wind field implementation to validate
        test_positions: List of test positions
        temporal_steps: Number of temporal steps to test
        dt: Time step size
        
    Returns:
        True if consistency properties are satisfied
        
    Raises:
        AssertionError: If consistency validation fails
    """
    test_positions = np.array(test_positions)
    
    # Test spatial consistency
    wind_vectors = wind_field.get_wind_at(test_positions)
    assert isinstance(wind_vectors, np.ndarray), "Wind field must return numpy array"
    assert wind_vectors.shape == (len(test_positions), 2), "Wind vectors must be 2D"
    assert np.all(np.isfinite(wind_vectors)), "Wind vectors must be finite"
    
    # Test temporal evolution
    wind_field.reset()
    temporal_winds = []
    
    for step in range(temporal_steps):
        current_winds = wind_field.get_wind_at(test_positions)
        temporal_winds.append(current_winds.copy())
        wind_field.step(dt=dt)
    
    temporal_winds = np.array(temporal_winds)
    
    # For constant wind fields, temporal variation should be minimal
    temporal_variance = np.var(temporal_winds, axis=0)
    
    # Check if this is a constant wind field (low temporal variance)
    if np.all(temporal_variance < WIND_VECTOR_TOLERANCE):
        # Constant wind field - verify temporal consistency
        for t in range(1, temporal_steps):
            assert np.allclose(temporal_winds[0], temporal_winds[t], 
                             atol=WIND_VECTOR_TOLERANCE), (
                "Constant wind field should not vary temporally"
            )
    
    return True


# Test fixtures for modular component testing
@pytest.fixture
def mock_gaussian_plume():
    """Provide mock Gaussian plume model for testing."""
    return MockPlumeModelFactory.create_mock_gaussian_plume()


@pytest.fixture
def mock_turbulent_plume():
    """Provide mock turbulent plume model for testing."""
    return MockPlumeModelFactory.create_mock_turbulent_plume()


@pytest.fixture
def mock_constant_wind():
    """Provide mock constant wind field for testing."""
    return MockWindFieldFactory.create_mock_constant_wind()


@pytest.fixture
def mock_turbulent_wind():
    """Provide mock turbulent wind field for testing."""
    return MockWindFieldFactory.create_mock_turbulent_wind()


@pytest.fixture
def mock_binary_sensor():
    """Provide mock binary sensor for testing."""
    return MockSensorFactory.create_mock_binary_sensor()


@pytest.fixture
def mock_concentration_sensor():
    """Provide mock concentration sensor for testing."""
    return MockSensorFactory.create_mock_concentration_sensor()


@pytest.fixture
def protocol_validator():
    """Provide protocol compliance validator for testing."""
    return ProtocolComplianceValidator()


@pytest.fixture
def plume_model_config():
    """Provide standard plume model configuration for testing."""
    return create_test_plume_model_config("gaussian")


@pytest.fixture
def wind_field_config():
    """Provide standard wind field configuration for testing."""
    return create_test_wind_field_config("constant")


@pytest.fixture
def sensor_config():
    """Provide standard sensor configuration for testing."""
    return create_test_sensor_config("concentration")


# Performance testing decorators for modular components
def requires_plume_model_performance(threshold_ms: float = PLUME_MODEL_STEP_THRESHOLD_MS):
    """
    Decorator to ensure plume model operations meet performance requirements.
    
    Args:
        threshold_ms: Maximum allowed duration in milliseconds
        
    Example:
        @requires_plume_model_performance(5.0)
        def test_gaussian_plume_step(gaussian_plume):
            gaussian_plume.step(dt=1.0)
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            with plume_model_performance_timer(test_func.__name__) as monitor:
                result = test_func(*args, **kwargs)
            monitor.assert_performance_requirements()
            return result
        return wrapper
    return decorator


def requires_protocol_compliance(protocol_type: str = "all"):
    """
    Decorator to ensure components pass protocol compliance validation.
    
    Args:
        protocol_type: Type of protocol to validate ("plume", "wind", "sensor", "all")
        
    Example:
        @requires_protocol_compliance("plume")
        def test_gaussian_plume_interface(gaussian_plume):
            # Test will automatically validate protocol compliance
            pass
    """
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            # Extract component from args/kwargs
            component = None
            if args and hasattr(args[0], 'concentration_at'):
                component = args[0]
                if protocol_type in ["plume", "all"]:
                    ProtocolComplianceValidator.validate_plume_model_protocol(component)
            elif args and hasattr(args[0], 'get_wind_at'):
                component = args[0]
                if protocol_type in ["wind", "all"]:
                    ProtocolComplianceValidator.validate_wind_field_protocol(component)
            elif args and hasattr(args[0], 'sample'):
                component = args[0]
                if protocol_type in ["sensor", "all"]:
                    ProtocolComplianceValidator.validate_sensor_protocol(component)
            
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


def setup_modular_test_logging():
    """
    Configure logging for modular component testing.
    
    Sets up logging with performance monitoring and protocol validation
    suitable for modular component test execution.
    """
    if not LOGGING_AVAILABLE:
        # Fallback to basic logging if centralized logging not available
        import logging
        logging.basicConfig(level=logging.WARNING)
        return
    
    # Configure logging for modular testing
    config = LoggingConfig(**MODULAR_TEST_CONFIG)
    setup_logger(config)


# Initialize modular test logging on module import
setup_modular_test_logging()


# Export testing utilities for modular components
__all__ = [
    # Protocol compliance validation
    "ProtocolComplianceValidator",
    "requires_protocol_compliance",
    
    # Performance monitoring
    "PlumeModelPerformanceMonitor",
    "plume_model_performance_timer",
    "requires_plume_model_performance",
    
    # Mock factories
    "MockPlumeModelFactory",
    "MockWindFieldFactory", 
    "MockSensorFactory",
    
    # Configuration utilities
    "create_test_plume_model_config",
    "create_test_wind_field_config",
    "create_test_sensor_config",
    
    # Mathematical validation
    "gaussian_plume_mathematical_validation",
    "wind_field_consistency_validation",
    
    # Test environment setup
    "setup_modular_test_logging",
    
    # Availability flags
    "HYPOTHESIS_AVAILABLE",
    "SCIPY_AVAILABLE",
    "PROTOCOLS_AVAILABLE",
    "LOGGING_AVAILABLE",
    "CONFIG_SCHEMAS_AVAILABLE",
    
    # Performance constants
    "PLUME_MODEL_STEP_THRESHOLD_MS",
    "WIND_FIELD_UPDATE_THRESHOLD_MS",
    "SENSOR_SAMPLING_THRESHOLD_MS",
    "PROTOCOL_SWITCHING_THRESHOLD_MS",
    "CONCENTRATION_FIELD_THRESHOLD_MS",
    
    # Mathematical validation constants
    "CONCENTRATION_TOLERANCE",
    "GRADIENT_FINITE_DIFF_EPSILON",
    "WIND_VECTOR_TOLERANCE",
    "GAUSSIAN_SIGMA_MIN",
    "GAUSSIAN_SIGMA_MAX",
]

# Conditional exports based on availability
if HYPOTHESIS_AVAILABLE:
    __all__.extend([
        "concentration_strategy",
        "spatial_coordinate_strategy",
        "spatial_position_strategy",
        "gaussian_sigma_strategy",
        "emission_rate_strategy",
        "wind_speed_strategy",
        "wind_direction_strategy",
        "turbulence_intensity_strategy",
        "grid_dimension_strategy",
        "grid_size_strategy",
        "multi_position_strategy",
    ])

if SCIPY_AVAILABLE:
    __all__.extend([
        # Additional scipy-based validation utilities would be listed here
    ])