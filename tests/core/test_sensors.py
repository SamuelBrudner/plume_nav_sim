"""
Comprehensive test suite for the sensor abstraction layer validating SensorProtocol compliance and integration.

This module provides extensive testing for the modular sensor architecture including BinarySensor,
ConcentrationSensor, and GradientSensor implementations. Tests validate protocol interface compliance,
multi-modal observation processing, performance requirements, and integration with all plume model types.

Key Test Categories:
- SensorProtocol compliance validation across all sensor implementations
- Multi-modal observation validation for dynamic observation space construction
- Sensor-plume model integration tests with GaussianPlumeModel, TurbulentPlumeModel, and VideoPlumeAdapter
- Performance validation ensuring <0.1ms per observation overhead and <10ms step execution targets
- Optional history wrapper functionality tests for HistoricalSensor integration
- Vectorized sensor sampling tests for multi-agent scenarios with linear scaling validation

Performance Requirements Validated:
- Sensor operations: <0.1ms per agent for minimal sensing overhead
- Multi-agent scaling: Linear performance scaling up to 100+ agents
- Memory efficiency: <1KB overhead per agent for sensor state management
- Step execution: Complete sensor pipeline within <10ms step latency requirements

Examples:
    Run sensor protocol compliance tests:
        pytest tests/core/test_sensors.py::TestSensorProtocolCompliance -v
        
    Run performance validation tests:
        pytest tests/core/test_sensors.py::TestSensorPerformance -v
        
    Run multi-agent integration tests:
        pytest tests/core/test_sensors.py::TestMultiAgentSensorIntegration -v
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, MagicMock, patch
from contextlib import contextmanager

import pytest
import numpy as np
import numpy.testing as npt

# Import test infrastructure and utilities
from tests.core.test_helpers import (
    PerformanceMonitor,
    performance_timer,
    requires_performance_validation,
    STEP_LATENCY_THRESHOLD_MS,
    create_test_config
)

# Import sensor implementations and protocols
try:
    from src.plume_nav_sim.core.protocols import SensorProtocol
    from src.plume_nav_sim.core.sensors.binary_sensor import BinarySensor, BinarySensorConfig
    from src.plume_nav_sim.core.sensors.concentration_sensor import ConcentrationSensor, ConcentrationSensorConfig
    from src.plume_nav_sim.core.sensors.gradient_sensor import GradientSensor, GradientSensorConfig, GradientResult
    from src.plume_nav_sim.core.controllers import DirectOdorSensor
    SENSORS_AVAILABLE = True
except ImportError as e:
    pytest.skip(f"Sensor implementations not available: {e}", allow_module_level=True)

# Import plume models for integration testing
try:
    from src.plume_nav_sim.models.plume.gaussian_plume import GaussianPlumeModel
    from src.plume_nav_sim.models.plume.turbulent_plume import TurbulentPlumeModel
    PLUME_MODELS_AVAILABLE = True
except ImportError:
    PLUME_MODELS_AVAILABLE = False

# Optional imports for enhanced testing
try:
    import hypothesis
    from hypothesis import given, assume, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


# Test fixtures and utilities

@pytest.fixture
def sample_positions():
    """Standard set of agent positions for testing."""
    return np.array([
        [10.0, 20.0],
        [15.0, 25.0], 
        [20.0, 30.0],
        [25.0, 35.0],
        [30.0, 40.0]
    ])


@pytest.fixture
def single_position():
    """Single agent position for testing."""
    return np.array([15.0, 25.0])


@pytest.fixture
def mock_plume_state():
    """Mock plume state for sensor testing."""
    mock_plume = Mock()
    
    # Mock concentration field as 2D array
    concentration_field = np.random.rand(100, 100) * 0.5
    mock_plume.shape = concentration_field.shape
    mock_plume.current_frame = concentration_field
    
    # Mock concentration_at method
    def mock_concentration_at(positions):
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        # Simple distance-based concentration for testing
        source_pos = np.array([50.0, 50.0])
        distances = np.linalg.norm(positions - source_pos, axis=1)
        concentrations = np.exp(-distances / 20.0)  # Exponential falloff
        
        return concentrations
    
    mock_plume.concentration_at = mock_concentration_at
    mock_plume.state_id = "test_plume_state"
    
    return mock_plume


@pytest.fixture
def gaussian_plume():
    """Real GaussianPlumeModel for integration testing."""
    if not PLUME_MODELS_AVAILABLE:
        pytest.skip("Plume models not available")
    
    return GaussianPlumeModel(
        source_position=(50.0, 50.0),
        source_strength=1000.0,
        sigma_x=10.0,
        sigma_y=8.0
    )


@pytest.fixture
def concentration_array():
    """2D concentration array for direct array testing."""
    # Create a simple Gaussian-like concentration field
    y, x = np.ogrid[:100, :100]
    center_x, center_y = 50, 50
    concentration = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 15**2))
    return concentration.astype(np.float64)


def test_direct_odor_sensor_interface():
    """DirectOdorSensor should expose modern sensing interface."""
    sensor = DirectOdorSensor()

    assert hasattr(sensor, "detect") and callable(sensor.detect)
    assert hasattr(sensor, "measure") and callable(sensor.measure)
    assert hasattr(sensor, "compute_gradient") and callable(sensor.compute_gradient)


class TestSensorProtocolCompliance:
    """Test suite validating SensorProtocol interface compliance across all sensor implementations."""
    
    def test_binary_sensor_protocol_compliance(self):
        """Test BinarySensor implements SensorProtocol correctly."""
        sensor = BinarySensor(threshold=0.1, enable_logging=False)
        
        # Verify protocol compliance
        assert isinstance(sensor, SensorProtocol)
        
        # Test required methods exist and are callable
        assert hasattr(sensor, 'detect') and callable(sensor.detect)
        assert hasattr(sensor, 'configure') and callable(sensor.configure) 
        assert hasattr(sensor, 'get_metadata') and callable(sensor.get_metadata)
        assert hasattr(sensor, 'reset') and callable(sensor.reset)
        
        # Test method signatures accept required parameters
        positions = np.array([[10, 20], [15, 25]])
        concentration_values = np.array([0.05, 0.15])
        
        # Should not raise exceptions with proper inputs
        try:
            detections = sensor.detect(concentration_values, positions)
            assert isinstance(detections, np.ndarray)
            assert detections.shape == (2,)
            assert detections.dtype == bool
        except Exception as e:
            pytest.fail(f"BinarySensor.detect() failed with valid inputs: {e}")
    
    def test_concentration_sensor_protocol_compliance(self):
        """Test ConcentrationSensor implements SensorProtocol correctly."""
        sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        
        # Verify protocol compliance
        assert isinstance(sensor, SensorProtocol)
        
        # Test required methods exist
        assert hasattr(sensor, 'detect') and callable(sensor.detect)
        assert hasattr(sensor, 'configure') and callable(sensor.configure)
        assert hasattr(sensor, 'reset') and callable(sensor.reset)
        
        # Test additional measurement method specific to ConcentrationSensor
        assert hasattr(sensor, 'measure') and callable(sensor.measure)
        
        # Test method behavior with mock plume state
        mock_plume = Mock()
        mock_plume.concentration_at = Mock(return_value=np.array([0.3, 0.7]))
        
        positions = np.array([[10, 20], [15, 25]])
        
        # Test detect method
        detections = sensor.detect(mock_plume, positions)
        assert isinstance(detections, np.ndarray)
        assert detections.dtype == bool
        
        # Test measure method
        measurements = sensor.measure(mock_plume, positions)
        assert isinstance(measurements, np.ndarray)
        assert measurements.dtype == np.float64
    
    def test_gradient_sensor_protocol_compliance(self):
        """Test GradientSensor implements SensorProtocol correctly."""
        sensor = GradientSensor(enable_logging=False)
        
        # Verify protocol compliance
        assert isinstance(sensor, SensorProtocol)
        
        # Test required methods exist
        assert hasattr(sensor, 'detect') and callable(sensor.detect)
        assert hasattr(sensor, 'configure') and callable(sensor.configure)
        assert hasattr(sensor, 'reset') and callable(sensor.reset)
        
        # Test gradient-specific methods
        assert hasattr(sensor, 'compute_gradient') and callable(sensor.compute_gradient)
        assert hasattr(sensor, 'measure') and callable(sensor.measure)
        
        # Test with mock plume state
        mock_plume = Mock()
        mock_plume.concentration_at = Mock(return_value=np.array([0.3, 0.7]))
        
        positions = np.array([[10, 20], [15, 25]])
        
        # Test detect method returns boolean array
        detections = sensor.detect(mock_plume, positions)
        assert isinstance(detections, np.ndarray)
        assert detections.dtype == bool
        
        # Test measure method returns gradient magnitudes
        magnitudes = sensor.measure(mock_plume, positions)
        assert isinstance(magnitudes, np.ndarray)
        assert magnitudes.dtype == np.float64
        
        # Test compute_gradient returns gradient vectors
        gradients = sensor.compute_gradient(mock_plume, positions)
        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == (2, 2)  # (n_agents, 2) for gradient components


class TestSensorFunctionality:
    """Test specific sensor functionality and behavior."""
    
    def test_binary_sensor_threshold_detection(self, mock_plume_state):
        """Test binary sensor threshold-based detection logic."""
        threshold = 0.3
        sensor = BinarySensor(threshold=threshold, enable_logging=False)
        
        positions = np.array([[10, 20], [50, 50], [90, 90]])  # Near source, at source, far from source
        concentration_values = np.array([0.1, 0.8, 0.05])  # Below, above, below threshold
        
        detections = sensor.detect(concentration_values, positions)
        
        # Verify threshold logic
        expected_detections = concentration_values >= threshold
        npt.assert_array_equal(detections, expected_detections)
        
        # Test single agent case
        single_pos = np.array([50, 50])
        single_conc = np.array([0.5])
        single_detection = sensor.detect(single_conc, single_pos.reshape(1, -1))
        
        assert isinstance(single_detection, np.ndarray)
        assert single_detection.shape == (1,)
        assert single_detection[0] == (0.5 >= threshold)
    
    def test_binary_sensor_hysteresis(self, mock_plume_state):
        """Test binary sensor hysteresis functionality."""
        threshold = 0.3
        hysteresis = 0.1
        sensor = BinarySensor(threshold=threshold, hysteresis=hysteresis, enable_logging=False)
        
        positions = np.array([[50, 50]])
        
        # Test rising edge (was False, now above rising threshold)
        concentration_values = np.array([0.35])  # Above threshold
        detections = sensor.detect(concentration_values, positions)
        assert detections[0] == True
        
        # Test falling edge (was True, now below falling threshold but above falling threshold)
        concentration_values = np.array([0.25])  # Between falling and rising threshold
        detections = sensor.detect(concentration_values, positions)
        assert detections[0] == True  # Should remain True due to hysteresis
        
        # Test falling below falling threshold
        concentration_values = np.array([0.15])  # Below falling threshold (0.2)
        detections = sensor.detect(concentration_values, positions)
        assert detections[0] == False  # Should switch to False
    
    def test_binary_sensor_noise_modeling(self, mock_plume_state):
        """Test binary sensor noise modeling with false positive/negative rates."""
        sensor = BinarySensor(
            threshold=0.3,
            false_positive_rate=0.1,
            false_negative_rate=0.1,
            random_seed=42,  # For reproducible testing
            enable_logging=False
        )

        positions = np.array([[i, i] for i in range(100)])  # 100 test positions
        concentration_values = np.zeros(100)  # All below threshold

        detections = sensor.detect(concentration_values, positions)

        # With false positive rate, some zeros should become True
        false_positives = np.sum(detections)
        assert false_positives > 0, "Expected some false positives with 10% rate"
        assert false_positives < 30, "Too many false positives detected"

    def test_binary_sensor_scalar_detection(self, single_position):
        """BinarySensor.detect should accept scalar inputs and return bool."""
        sensor = BinarySensor(threshold=0.1, enable_logging=False)

        concentration = 0.2  # Above threshold
        detection = sensor.detect(concentration, single_position)

        assert isinstance(detection, (bool, np.bool_))
        assert detection is True
    
    def test_concentration_sensor_quantitative_measurement(self, mock_plume_state):
        """Test concentration sensor quantitative measurement capabilities."""
        sensor = ConcentrationSensor(
            dynamic_range=(0.0, 10.0),
            resolution=0.01,
            enable_logging=False
        )
        
        positions = np.array([[10, 20], [50, 50], [90, 90]])
        measurements = sensor.measure(mock_plume_state, positions)
        
        # Verify return format
        assert isinstance(measurements, np.ndarray)
        assert measurements.shape == (3,)
        assert measurements.dtype == np.float64
        
        # Verify measurements are within dynamic range
        assert np.all(measurements >= 0.0)
        assert np.all(measurements <= 10.0)

        # Test single agent measurement
        single_pos = np.array([50, 50])
        single_measurement = sensor.measure(mock_plume_state, single_pos)
        assert isinstance(single_measurement, (float, np.floating))

    def test_concentration_sensor_scalar_dtype(self, mock_plume_state, single_position):
        """ConcentrationSensor.measure should return float64 for single position."""
        sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)

        measurement = sensor.measure(mock_plume_state, single_position)

        assert np.isscalar(measurement)
        assert np.asarray(measurement).dtype == np.float64
    
    def test_concentration_sensor_noise_and_drift(self, mock_plume_state):
        """Test concentration sensor noise and drift modeling."""
        sensor = ConcentrationSensor(
            dynamic_range=(0.0, 1.0),
            noise_std=0.05,
            enable_drift=True,
            drift_rate=0.01,
            random_seed=42,
            enable_logging=False
        )
        
        positions = np.array([[50, 50]])
        
        # Take multiple measurements to test noise and drift
        measurements = []
        for i in range(10):
            time.sleep(0.1)  # Small delay to allow drift accumulation
            measurement = sensor.measure(mock_plume_state, positions)
            measurements.append(measurement)
        
        measurements = np.array(measurements)
        
        # Verify noise introduces variability
        measurement_std = np.std(measurements)
        assert measurement_std > 0, "Expected measurement variability due to noise"
        
        # Verify drift affects measurements over time
        metadata = sensor.get_performance_metrics()
        assert 'current_drift_offset' in metadata
    
    def test_concentration_sensor_saturation(self, mock_plume_state):
        """Test concentration sensor saturation behavior."""
        saturation_level = 0.5
        sensor = ConcentrationSensor(
            dynamic_range=(0.0, 1.0),
            saturation_level=saturation_level,
            enable_logging=False
        )
        
        # Mock high concentration input
        high_concentration_plume = Mock()
        high_concentration_plume.concentration_at = Mock(return_value=np.array([0.8, 1.2]))
        
        positions = np.array([[10, 20], [50, 50]])
        measurements = sensor.measure(high_concentration_plume, positions)
        
        # Verify saturation limiting
        assert np.all(measurements <= saturation_level)
        
        # Check performance metrics for saturation events
        metrics = sensor.get_performance_metrics()
        assert 'saturation_events' in metrics
    
    def test_gradient_sensor_spatial_derivatives(self, mock_plume_state):
        """Test gradient sensor spatial derivative computation."""
        sensor = GradientSensor(
            config=GradientSensorConfig(
                spatial_resolution=(0.5, 0.5),
                method="central",
                order=2
            ),
            enable_logging=False
        )
        
        positions = np.array([[45, 45], [50, 50], [55, 55]])
        gradients = sensor.compute_gradient(mock_plume_state, positions)
        
        # Verify gradient shape and type
        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == (3, 2)  # (n_agents, 2) for [dx, dy]
        assert gradients.dtype == np.float64
        
        # Test single agent case
        single_pos = np.array([50, 50])
        single_gradient = sensor.compute_gradient(mock_plume_state, single_pos)
        assert isinstance(single_gradient, np.ndarray)
        assert single_gradient.shape == (2,)
    
    def test_gradient_sensor_comprehensive_metadata(self, mock_plume_state):
        """Test gradient sensor comprehensive metadata generation."""
        sensor = GradientSensor(
            config=GradientSensorConfig(enable_metadata=True),
            enable_logging=False
        )
        
        positions = np.array([[45, 45], [50, 50]])
        result = sensor.compute_gradient_with_metadata(mock_plume_state, positions)
        
        # Verify GradientResult structure
        assert isinstance(result, GradientResult)
        assert hasattr(result, 'gradient')
        assert hasattr(result, 'magnitude')
        assert hasattr(result, 'direction')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'metadata')
        
        # Verify metadata content
        assert isinstance(result.metadata, dict)
        assert 'sensor_id' in result.metadata
        assert 'timestamp' in result.metadata
        
        # Verify computed quantities
        expected_magnitude = np.sqrt(np.sum(result.gradient**2, axis=1))
        npt.assert_array_almost_equal(result.magnitude, expected_magnitude)
        
        expected_direction = np.degrees(np.arctan2(result.gradient[:, 1], result.gradient[:, 0]))
        expected_direction = (expected_direction + 360) % 360
        npt.assert_array_almost_equal(result.direction, expected_direction)


class TestSensorConfiguration:
    """Test sensor configuration management and runtime updates."""
    
    def test_binary_sensor_configuration_updates(self):
        """Test BinarySensor configuration parameter updates."""
        sensor = BinarySensor(threshold=0.1, enable_logging=False)
        
        # Test parameter updates
        sensor.configure(threshold=0.2, hysteresis=0.05)
        assert sensor.config.threshold == 0.2
        assert sensor.config.hysteresis == 0.05
        
        # Test invalid parameter validation
        with pytest.raises(ValueError):
            sensor.configure(threshold=-0.1)  # Negative threshold
        
        with pytest.raises(ValueError):
            sensor.configure(threshold=1.5)  # Threshold > 1.0
    
    def test_concentration_sensor_configuration_updates(self, mock_plume_state):
        """Test ConcentrationSensor configuration parameter updates."""
        sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        
        # Test dynamic range update
        sensor.configure(dynamic_range=(0.0, 5.0), resolution=0.001)
        assert sensor._config.dynamic_range == (0.0, 5.0)
        assert sensor._config.resolution == 0.001
        
        # Test noise configuration
        sensor.configure(noise_std=0.1, enable_drift=True)
        assert sensor._config.noise_std == 0.1
        assert sensor._config.enable_drift == True
        
        # Test invalid configuration
        with pytest.raises(ValueError):
            sensor.configure(dynamic_range=(5.0, 1.0))  # Invalid range
    
    def test_gradient_sensor_configuration_updates(self):
        """Test GradientSensor configuration parameter updates."""
        sensor = GradientSensor(enable_logging=False)
        
        # Test spatial resolution update
        sensor.configure(spatial_resolution=(0.1, 0.1))
        assert sensor.config.spatial_resolution == (0.1, 0.1)
        
        # Test method update
        sensor.configure(method='forward', order=4)
        assert sensor.config.method.value == 'forward'
        assert sensor.config.order == 4
        
        # Test adaptive parameters
        sensor.configure(adaptive_step_size=True, min_step_size=0.05)
        assert sensor.config.adaptive_step_size == True
        assert sensor.config.min_step_size == 0.05
        
        # Test invalid order
        with pytest.raises(ValueError):
            sensor.configure(order=3)  # Invalid order


class TestSensorPlumeModeIntegration:
    """Test sensor integration with different plume model types."""
    
    @pytest.mark.skipif(not PLUME_MODELS_AVAILABLE, reason="Plume models not available")
    def test_sensors_with_gaussian_plume(self, gaussian_plume):
        """Test all sensors work correctly with GaussianPlumeModel."""
        positions = np.array([[40, 40], [50, 50], [60, 60]])
        
        # Test BinarySensor
        binary_sensor = BinarySensor(threshold=0.1, enable_logging=False)
        binary_detections = binary_sensor.detect(
            gaussian_plume.concentration_at(positions), positions
        )
        assert isinstance(binary_detections, np.ndarray)
        assert binary_detections.dtype == bool
        
        # Test ConcentrationSensor  
        conc_sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        concentrations = conc_sensor.measure(gaussian_plume, positions)
        assert isinstance(concentrations, np.ndarray)
        assert concentrations.dtype == np.float64
        assert concentrations.shape == (3,)
        
        # Test GradientSensor
        grad_sensor = GradientSensor(enable_logging=False)
        gradients = grad_sensor.compute_gradient(gaussian_plume, positions)
        assert isinstance(gradients, np.ndarray)
        assert gradients.shape == (3, 2)
    
    def test_sensors_with_concentration_array(self, concentration_array):
        """Test sensors work with direct concentration arrays."""
        positions = np.array([[25, 25], [50, 50], [75, 75]])
        
        # Test BinarySensor
        binary_sensor = BinarySensor(threshold=0.3, enable_logging=False)
        
        # Sample concentrations manually from array
        sampled_concentrations = []
        for pos in positions:
            x, y = int(pos[0]), int(pos[1])
            x = np.clip(x, 0, concentration_array.shape[1] - 1)
            y = np.clip(y, 0, concentration_array.shape[0] - 1)
            sampled_concentrations.append(concentration_array[y, x])
        sampled_concentrations = np.array(sampled_concentrations)
        
        detections = binary_sensor.detect(sampled_concentrations, positions)
        assert isinstance(detections, np.ndarray)
        assert detections.dtype == bool
        
        # Test ConcentrationSensor with array-like plume state
        conc_sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        measurements = conc_sensor.measure(concentration_array, positions)
        assert isinstance(measurements, np.ndarray)
        assert measurements.dtype == np.float64
    
    def test_sensors_with_mock_plume_state(self, mock_plume_state):
        """Test sensors work with mock plume state objects."""
        positions = np.array([[10, 20], [30, 40], [50, 60]])
        
        # All sensors should handle mock plume state gracefully
        binary_sensor = BinarySensor(threshold=0.2, enable_logging=False)
        conc_sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        grad_sensor = GradientSensor(enable_logging=False)
        
        # Test each sensor type
        binary_result = binary_sensor.detect(
            mock_plume_state.concentration_at(positions), positions
        )
        assert isinstance(binary_result, np.ndarray)
        
        conc_result = conc_sensor.measure(mock_plume_state, positions)
        assert isinstance(conc_result, np.ndarray)
        
        grad_result = grad_sensor.compute_gradient(mock_plume_state, positions)
        assert isinstance(grad_result, np.ndarray)


class TestSensorPerformance:
    """Test sensor performance requirements and optimization."""
    
    @requires_performance_validation(0.1)  # <0.1ms per agent requirement
    def test_binary_sensor_performance_single_agent(self, mock_plume_state):
        """Test BinarySensor meets single-agent performance requirements."""
        sensor = BinarySensor(threshold=0.1, enable_logging=False)
        positions = np.array([[50, 50]])
        concentration_values = np.array([0.5])
        
        # Performance test is handled by decorator
        detections = sensor.detect(concentration_values, positions)
        assert len(detections) == 1
    
    @requires_performance_validation(1.0)  # <1ms for 100 agents requirement
    def test_concentration_sensor_performance_multi_agent(self, mock_plume_state):
        """Test ConcentrationSensor performance with multiple agents."""
        sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        
        # Test with 100 agents
        positions = np.random.rand(100, 2) * 100
        
        # Performance test is handled by decorator
        measurements = sensor.measure(mock_plume_state, positions)
        assert len(measurements) == 100
    
    @requires_performance_validation(2.0)  # <0.2ms per agent for gradients (more expensive)
    def test_gradient_sensor_performance_single_agent(self, mock_plume_state):
        """Test GradientSensor meets performance requirements."""
        sensor = GradientSensor(
            config=GradientSensorConfig(
                method="central",
                order=2,
                adaptive_step_size=False  # Disable for performance
            ),
            enable_logging=False
        )
        
        positions = np.array([[50, 50]])
        
        # Performance test is handled by decorator  
        gradients = sensor.compute_gradient(mock_plume_state, positions)
        assert gradients.shape == (2,)
    
    def test_sensor_memory_efficiency(self, mock_plume_state):
        """Test sensor memory usage remains within efficiency requirements."""
        import psutil
        process = psutil.Process()
        
        # Baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create sensors and perform operations
        sensors = [
            BinarySensor(threshold=0.1, enable_logging=False),
            ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False),
            GradientSensor(enable_logging=False)
        ]
        
        positions = np.random.rand(100, 2) * 100
        
        # Perform operations with all sensors
        for sensor in sensors:
            if hasattr(sensor, 'detect'):
                if isinstance(sensor, BinarySensor):
                    concentration_values = np.random.rand(100)
                    sensor.detect(concentration_values, positions)
                else:
                    sensor.detect(mock_plume_state, positions)
            
            if hasattr(sensor, 'measure'):
                sensor.measure(mock_plume_state, positions)
                
            if hasattr(sensor, 'compute_gradient'):
                sensor.compute_gradient(mock_plume_state, positions)
        
        # Check memory usage after operations
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - baseline_memory
        
        # Memory requirement: <1KB per agent = <0.1MB for 100 agents
        max_memory_mb = 100 * 0.001  # 0.1MB
        assert memory_delta < max_memory_mb, (
            f"Memory usage {memory_delta:.3f}MB exceeds limit {max_memory_mb:.3f}MB"
        )
    
    def test_vectorized_operations_scaling(self, mock_plume_state):
        """Test sensor operations scale linearly with agent count."""
        sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        
        agent_counts = [1, 10, 50, 100]
        execution_times = []
        
        for count in agent_counts:
            positions = np.random.rand(count, 2) * 100
            
            start_time = time.perf_counter()
            measurements = sensor.measure(mock_plume_state, positions)
            end_time = time.perf_counter()
            
            execution_times.append((end_time - start_time) * 1000)  # Convert to ms
            assert len(measurements) == count
        
        # Verify linear scaling (within reasonable tolerance)
        time_per_agent = [t / c for t, c in zip(execution_times, agent_counts)]
        
        # Time per agent should be relatively constant for vectorized operations
        max_time_per_agent = max(time_per_agent)
        min_time_per_agent = min(time_per_agent)
        scaling_ratio = max_time_per_agent / min_time_per_agent
        
        # Allow for some overhead but should be roughly linear
        assert scaling_ratio < 3.0, (
            f"Poor vectorization scaling: {scaling_ratio:.2f}x variation in per-agent time"
        )


class TestMultiModalObservationConstruction:
    """Test multi-modal observation space construction and sensor combination."""
    
    def test_sensor_combination_for_observation_space(self, mock_plume_state):
        """Test combining multiple sensors for rich observation spaces."""
        positions = np.array([[40, 40], [50, 50], [60, 60]])
        
        # Create multiple sensors
        binary_sensor = BinarySensor(threshold=0.2, enable_logging=False)
        conc_sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        grad_sensor = GradientSensor(enable_logging=False)
        
        # Simulate multi-modal observation construction
        observations = {}
        
        # Binary detection
        concentration_values = mock_plume_state.concentration_at(positions)
        observations['odor_detected'] = binary_sensor.detect(concentration_values, positions)
        
        # Quantitative measurements
        observations['concentration'] = conc_sensor.measure(mock_plume_state, positions)
        
        # Gradient information
        gradients = grad_sensor.compute_gradient(mock_plume_state, positions)
        observations['gradient_x'] = gradients[:, 0]
        observations['gradient_y'] = gradients[:, 1]
        observations['gradient_magnitude'] = np.sqrt(np.sum(gradients**2, axis=1))
        observations['gradient_direction'] = np.degrees(np.arctan2(gradients[:, 1], gradients[:, 0]))
        
        # Verify observation structure
        assert all(isinstance(obs, np.ndarray) for obs in observations.values())
        assert all(len(obs) == 3 for obs in observations.values())
        
        # Verify data types
        assert observations['odor_detected'].dtype == bool
        assert observations['concentration'].dtype == np.float64
        assert observations['gradient_magnitude'].dtype == np.float64
    
    def test_sensor_configuration_for_different_modalities(self, mock_plume_state):
        """Test sensor configurations for different sensing modalities."""
        positions = np.array([[50, 50]])
        
        # High-sensitivity binary detection
        sensitive_binary = BinarySensor(threshold=0.01, hysteresis=0.005, enable_logging=False)
        
        # High-precision concentration measurement
        precise_conc = ConcentrationSensor(
            dynamic_range=(0.0, 1.0),
            resolution=0.0001,
            noise_std=0.001,
            enable_logging=False
        )
        
        # High-accuracy gradient computation
        accurate_grad = GradientSensor(
            config=GradientSensorConfig(
                method="central",
                order=4,
                spatial_resolution=(0.1, 0.1),
                adaptive_step_size=True
            ),
            enable_logging=False
        )
        
        # Test each modality
        concentration_values = mock_plume_state.concentration_at(positions)
        
        sensitive_detection = sensitive_binary.detect(concentration_values, positions)
        precise_measurement = precise_conc.measure(mock_plume_state, positions)
        accurate_gradient = accurate_grad.compute_gradient(mock_plume_state, positions)
        
        # Verify results
        assert isinstance(sensitive_detection, np.ndarray)
        assert isinstance(precise_measurement, (float, np.floating))
        assert isinstance(accurate_gradient, np.ndarray)
        assert accurate_gradient.shape == (2,)


class TestSensorErrorHandling:
    """Test sensor error handling and edge cases."""
    
    def test_invalid_position_arrays(self, mock_plume_state):
        """Test sensor handling of invalid position arrays."""
        sensor = BinarySensor(threshold=0.1, enable_logging=False)
        
        # Test invalid shapes
        with pytest.raises(ValueError):
            invalid_positions = np.array([1, 2, 3])  # Wrong shape
            concentration_values = np.array([0.5])
            sensor.detect(concentration_values, invalid_positions)
        
        with pytest.raises(ValueError):
            invalid_positions = np.array([[1, 2, 3]])  # Wrong number of coordinates
            concentration_values = np.array([0.5])
            sensor.detect(concentration_values, invalid_positions)
    
    def test_array_length_mismatch(self, mock_plume_state):
        """Test sensor handling of mismatched array lengths."""
        sensor = BinarySensor(threshold=0.1, enable_logging=False)
        
        with pytest.raises(ValueError):
            positions = np.array([[10, 20], [30, 40]])  # 2 positions
            concentration_values = np.array([0.1, 0.2, 0.3])  # 3 values
            sensor.detect(concentration_values, positions)
    
    def test_sensor_with_invalid_plume_state(self):
        """Test sensor graceful handling of invalid plume states."""
        sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        positions = np.array([[50, 50]])
        
        # Test with None plume state
        measurements = sensor.measure(None, positions)
        assert isinstance(measurements, (float, np.floating, np.ndarray))
        
        # Test with invalid plume state object
        invalid_plume = object()
        measurements = sensor.measure(invalid_plume, positions)
        assert isinstance(measurements, (float, np.floating, np.ndarray))
    
    def test_sensor_reset_functionality(self, mock_plume_state):
        """Test sensor reset functionality clears internal state."""
        # Test BinarySensor reset
        binary_sensor = BinarySensor(threshold=0.1, hysteresis=0.05, enable_logging=False)
        
        positions = np.array([[50, 50]])
        concentration_values = np.array([0.5])
        
        # Establish some internal state
        binary_sensor.detect(concentration_values, positions)
        assert binary_sensor._previous_detections is not None
        
        # Reset and verify state is cleared
        binary_sensor.reset()
        assert binary_sensor._previous_detections is None
        
        # Test ConcentrationSensor reset
        conc_sensor = ConcentrationSensor(
            dynamic_range=(0.0, 1.0),
            enable_drift=True,
            enable_logging=False
        )
        
        # Establish some state
        conc_sensor.measure(mock_plume_state, positions)
        initial_metrics = conc_sensor.get_performance_metrics()
        
        # Reset and verify metrics are cleared
        conc_sensor.reset()
        reset_metrics = conc_sensor.get_performance_metrics()
        assert reset_metrics['total_measurements'] == 0


class TestSensorMetadata:
    """Test sensor metadata collection and reporting."""
    
    def test_binary_sensor_metadata(self, mock_plume_state):
        """Test BinarySensor metadata collection."""
        sensor = BinarySensor(
            threshold=0.1,
            confidence_reporting=True,
            history_length=5,
            enable_logging=False
        )
        
        positions = np.array([[50, 50]])
        concentration_values = np.array([0.5])
        
        # Perform some detections to generate metadata
        for i in range(10):
            sensor.detect(concentration_values, positions)
        
        metadata = sensor.get_metadata()
        
        # Verify metadata structure
        assert isinstance(metadata, dict)
        assert 'sensor_type' in metadata
        assert 'sensor_id' in metadata
        assert 'configuration' in metadata
        assert 'performance' in metadata
        
        # Verify configuration metadata
        config = metadata['configuration']
        assert config['threshold'] == 0.1
        assert config['confidence_reporting'] == True
        
        # Verify performance metadata
        performance = metadata['performance']
        assert 'total_calls' in performance
        assert 'total_detections' in performance
        assert performance['total_calls'] == 10
    
    def test_concentration_sensor_metadata(self, mock_plume_state):
        """Test ConcentrationSensor metadata collection."""
        sensor = ConcentrationSensor(
            dynamic_range=(0.0, 1.0),
            enable_metadata=True,
            enable_logging=False
        )
        
        positions = np.array([[50, 50]])
        
        # Perform measurements
        for i in range(5):
            sensor.measure(mock_plume_state, positions)
        
        info = sensor.get_sensor_info()
        
        # Verify sensor info structure
        assert isinstance(info, dict)
        assert info['sensor_type'] == 'ConcentrationSensor'
        assert 'capabilities' in info
        assert 'quantitative_measurement' in info['capabilities']
        assert 'noise_modeling' in info['capabilities']
        
        # Test performance metrics
        metrics = sensor.get_performance_metrics()
        assert 'total_measurements' in metrics
        assert metrics['total_measurements'] >= 5
    
    def test_gradient_sensor_metadata(self, mock_plume_state):
        """Test GradientSensor metadata and performance reporting."""
        sensor = GradientSensor(
            config=GradientSensorConfig(enable_metadata=True),
            enable_logging=False
        )
        
        positions = np.array([[45, 45], [50, 50]])
        
        # Perform gradient computations
        for i in range(3):
            result = sensor.compute_gradient_with_metadata(mock_plume_state, positions)
        
        # Test comprehensive metadata from gradient result
        assert isinstance(result.metadata, dict)
        assert 'sensor_id' in result.metadata
        assert 'timestamp' in result.metadata
        
        # Test performance metrics
        metrics = sensor.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'sensor_type' in metrics
        assert metrics['sensor_type'] == 'GradientSensor'
        assert 'total_computations' in metrics


class TestSensorIntrospection:
    """Tests for sensor capability and metadata introspection utilities."""

    def test_binary_sensor_info_and_observation_space(self):
        sensor = BinarySensor(threshold=0.1, enable_logging=False)

        info = sensor.get_sensor_info()
        assert info['sensor_type'] == 'BinarySensor'
        assert 'binary_detection' in info['capabilities']

        obs_info = sensor.get_observation_space_info()
        assert obs_info['shape'] == (1,)
        assert obs_info['dtype'] == np.bool_

    def test_concentration_sensor_metadata_accessor(self, mock_plume_state):
        sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        position = np.array([50.0, 50.0])

        sensor.measure(mock_plume_state, position)
        metadata = sensor.get_metadata()

        assert metadata['sensor_type'] == 'ConcentrationSensor'
        assert metadata['performance']['total_measurements'] == 1

    def test_gradient_sensor_introspection(self, mock_plume_state, single_position):
        sensor = GradientSensor(enable_logging=False)

        sensor.compute_gradient(mock_plume_state, single_position)

        info = sensor.get_sensor_info()
        assert info['sensor_type'] == 'GradientSensor'
        assert 'gradient_computation' in info['capabilities']

        obs_info = sensor.get_observation_space_info()
        assert obs_info['shape'] == (2,)
        assert obs_info['dtype'] == np.float64

        metadata = sensor.get_metadata()
        assert metadata['sensor_type'] == 'GradientSensor'
        assert metadata['performance']['total_operations'] >= 1

    def test_concentration_sensor_metrics_after_reset(self, mock_plume_state):
        sensor = ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False)
        position = np.array([10.0, 10.0])
        sensor.measure(mock_plume_state, position)

        sensor.reset()
        metrics = sensor.get_performance_metrics()
        assert metrics['sensor_type'] == 'ConcentrationSensor'
        assert 'total_operations' in metrics

    def test_performance_metrics_persist_after_reset(self):
        sensor = BinarySensor(threshold=0.1, enable_logging=False)
        sensor.reset()
        metrics = sensor.get_performance_metrics()
        assert metrics['sensor_type'] == 'BinarySensor'
        assert 'total_operations' in metrics


@pytest.mark.xfail(reason="Vectorization performance requires further optimization")
def test_binary_sensor_vectorization_scaling(sample_positions):
    sensor = BinarySensor(threshold=0.1, enable_logging=False)
    batch_sizes = [1, 8, 64]
    times_per_agent = []
    for n in batch_sizes:
        concentrations = np.linspace(0.0, 1.0, n)
        positions = np.tile(sample_positions[0], (n, 1))
        sensor.detect(concentrations, positions)  # warm-up
        start = time.perf_counter()
        for _ in range(10):
            sensor.detect(concentrations, positions)
        times_per_agent.append((time.perf_counter() - start) / (10 * n))

    # Ignore single-agent warm-up which includes initialization overhead
    ratio = max(times_per_agent[1:]) / min(times_per_agent[1:])
    assert ratio < 3


# Integration tests requiring hypothesis property-based testing
if HYPOTHESIS_AVAILABLE:
    
    class TestSensorPropertyBasedTesting:
        """Property-based testing using Hypothesis for comprehensive validation."""
        
        @given(
            positions=st.lists(
                st.tuples(
                    st.floats(min_value=0, max_value=100, allow_nan=False),
                    st.floats(min_value=0, max_value=100, allow_nan=False)
                ),
                min_size=1,
                max_size=20
            ),
            threshold=st.floats(min_value=0.01, max_value=0.99, allow_nan=False)
        )
        def test_binary_sensor_threshold_property(self, positions, threshold):
            """Property test: Binary sensor always respects threshold logic."""
            assume(len(positions) > 0)
            
            sensor = BinarySensor(threshold=threshold, enable_logging=False)
            positions_array = np.array(positions)
            
            # Generate concentration values around threshold
            concentration_values = np.random.rand(len(positions))
            
            detections = sensor.detect(concentration_values, positions_array)
            
            # Property: detections should match threshold comparison
            expected_detections = concentration_values >= threshold
            npt.assert_array_equal(detections, expected_detections)
        
        @given(
            positions=st.lists(
                st.tuples(
                    st.floats(min_value=0, max_value=100, allow_nan=False),
                    st.floats(min_value=0, max_value=100, allow_nan=False)
                ),
                min_size=1,
                max_size=50
            ),
            dynamic_range=st.tuples(
                st.floats(min_value=0, max_value=1, allow_nan=False),
                st.floats(min_value=1, max_value=10, allow_nan=False)
            )
        )
        def test_concentration_sensor_range_property(self, positions, dynamic_range):
            """Property test: Concentration sensor respects dynamic range."""
            assume(len(positions) > 0)
            assume(dynamic_range[1] > dynamic_range[0])
            
            sensor = ConcentrationSensor(
                dynamic_range=dynamic_range,
                enable_logging=False
            )
            
            # Create simple mock plume
            mock_plume = Mock()
            mock_plume.concentration_at = Mock(
                return_value=np.random.rand(len(positions)) * dynamic_range[1]
            )
            
            positions_array = np.array(positions)
            measurements = sensor.measure(mock_plume, positions_array)
            
            # Property: measurements should be within dynamic range
            assert np.all(measurements >= dynamic_range[0])
            assert np.all(measurements <= dynamic_range[1])


# Performance benchmark tests
class TestSensorBenchmarks:
    """Benchmark tests for sensor performance optimization."""
    
    def test_sensor_operation_benchmark(self, mock_plume_state):
        """Benchmark all sensor operations for performance analysis."""
        agent_counts = [1, 10, 50, 100]
        sensors = {
            'binary': BinarySensor(threshold=0.1, enable_logging=False),
            'concentration': ConcentrationSensor(dynamic_range=(0.0, 1.0), enable_logging=False), 
            'gradient': GradientSensor(enable_logging=False)
        }
        
        benchmark_results = {}
        
        for sensor_name, sensor in sensors.items():
            benchmark_results[sensor_name] = {}
            
            for count in agent_counts:
                positions = np.random.rand(count, 2) * 100
                
                # Benchmark each sensor operation
                if sensor_name == 'binary':
                    concentration_values = np.random.rand(count)
                    
                    start_time = time.perf_counter()
                    for _ in range(10):  # Average over multiple runs
                        sensor.detect(concentration_values, positions)
                    end_time = time.perf_counter()
                    
                    avg_time = (end_time - start_time) / 10 * 1000  # ms
                    
                elif sensor_name == 'concentration':
                    start_time = time.perf_counter()
                    for _ in range(10):
                        sensor.measure(mock_plume_state, positions)
                    end_time = time.perf_counter()
                    
                    avg_time = (end_time - start_time) / 10 * 1000  # ms
                    
                elif sensor_name == 'gradient':
                    start_time = time.perf_counter()
                    for _ in range(10):
                        sensor.compute_gradient(mock_plume_state, positions)
                    end_time = time.perf_counter()
                    
                    avg_time = (end_time - start_time) / 10 * 1000  # ms
                
                benchmark_results[sensor_name][count] = {
                    'total_time_ms': avg_time,
                    'time_per_agent_ms': avg_time / count
                }
        
        # Verify performance requirements
        for sensor_name, results in benchmark_results.items():
            for count, metrics in results.items():
                if sensor_name in ['binary', 'concentration']:
                    # <0.1ms per agent requirement
                    assert metrics['time_per_agent_ms'] < 0.1, (
                        f"{sensor_name} sensor exceeds 0.1ms per agent: "
                        f"{metrics['time_per_agent_ms']:.3f}ms"
                    )
                elif sensor_name == 'gradient':
                    # <0.2ms per agent for gradient computation
                    assert metrics['time_per_agent_ms'] < 0.2, (
                        f"{sensor_name} sensor exceeds 0.2ms per agent: "
                        f"{metrics['time_per_agent_ms']:.3f}ms"
                    )


# Test execution entry point
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])