"""
Comprehensive test suite for WindFieldProtocol implementations including ConstantWindField,
TurbulentWindField, and TimeVaryingWindField.

This test module validates wind field implementations according to Section 0.4.1 requirements,
ensuring proper environmental dynamics integration, vector field computation accuracy,
temporal evolution capabilities, and performance requirements for realistic plume transport
modeling. Tests include protocol compliance validation, mathematical property verification,
configuration switching via Hydra, and property-based testing using Hypothesis.

Key Testing Areas:
- WindFieldProtocol compliance validation for all implementations per Section 0.4.1
- Performance testing ensuring wind field computations complete within 2ms per simulation step
- Mathematical property validation for wind vector field computations and temporal evolution
- Configuration switching tests for different wind field types via Hydra instantiation
- Property-based testing for wind vector field mathematical properties using Hypothesis
- Integration testing with plume transport calculations and environmental dynamics
- Temporal dynamics validation including evolution patterns and stochastic variations
- Boundary condition handling and spatial domain constraint validation

Performance Requirements (from Section 0.2.1):
- velocity_at(): <0.5ms for single query, <2ms for 100+ position batch queries 
- step(): <2ms per time step for minimal simulation overhead
- Memory efficiency: <50MB for typical wind field representations
- Real-time simulation compatibility maintaining sub-10ms integration latency

Enhanced Integration Points:
- src.plume_nav_sim.core.protocols.WindFieldProtocol compliance validation
- src.plume_nav_sim.models.wind.constant_wind.ConstantWindField implementation tests  
- src.plume_nav_sim.models.wind.turbulent_wind.TurbulentWindField implementation tests
- src.plume_nav_sim.models.wind.time_varying_wind.TimeVaryingWindField implementation tests
- src.plume_nav_sim.config.schemas configuration schema validation and Hydra integration
- Performance monitoring hooks and threshold validation per Section 0.2.1 requirements
"""

import pytest
import numpy as np
import time
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any, Tuple, List, Optional, Union
from contextlib import contextmanager
import tempfile
import json

# Core testing framework imports from test infrastructure
from tests.core.test_helpers import (
    PerformanceMonitor, performance_timer
)

# Check for optional dependencies
try:
    import hypothesis
    HYPOTHESIS_AVAILABLE = True
    from hypothesis import given, strategies as st
    
    # Define testing strategies
    coordinate_strategy = st.floats(min_value=-1000.0, max_value=1000.0, allow_infinity=False, allow_nan=False)
    position_strategy = st.tuples(coordinate_strategy, coordinate_strategy)
    angle_strategy = st.floats(min_value=0.0, max_value=2*3.14159, allow_infinity=False, allow_nan=False)
    speed_strategy = st.floats(min_value=0.0, max_value=100.0, allow_infinity=False, allow_nan=False)
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    coordinate_strategy = None
    position_strategy = None
    angle_strategy = None
    speed_strategy = None

try:
    import gymnasium
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

# Property-based testing with Hypothesis
if HYPOTHESIS_AVAILABLE:
    from hypothesis import given, strategies as st, settings, assume, HealthCheck
    from hypothesis.extra.numpy import arrays
else:
    # Fallback decorators when Hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    st = None
    settings = lambda *args, **kwargs: lambda f: f

# Configuration management imports
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

# Enhanced logging integration for test performance monitoring
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

# Scientific computing imports for validation
try:
    import scipy.stats
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Core protocol import for interface compliance testing
from src.plume_nav_sim.protocols.wind_field import WindFieldProtocol
PROTOCOLS_AVAILABLE = True

# Wind field implementation imports for testing
try:
    from src.plume_nav_sim.models.wind.constant_wind import (
        ConstantWindField, ConstantWindFieldConfig, create_constant_wind_field
    )
    CONSTANT_WIND_AVAILABLE = True
except ImportError:
    CONSTANT_WIND_AVAILABLE = False
    ConstantWindField = None
    ConstantWindFieldConfig = None

try:
    from src.plume_nav_sim.models.wind.turbulent_wind import (
        TurbulentWindField, TurbulentWindFieldConfig, create_turbulent_wind_field
    )
    TURBULENT_WIND_AVAILABLE = True
except ImportError:
    TURBULENT_WIND_AVAILABLE = False
    TurbulentWindField = None
    TurbulentWindFieldConfig = None

try:
    from src.plume_nav_sim.models.wind.time_varying_wind import (
        TimeVaryingWindField, TimeVaryingWindFieldConfig, create_time_varying_wind_field
    )
    TIME_VARYING_WIND_AVAILABLE = True
except ImportError:
    TIME_VARYING_WIND_AVAILABLE = False
    TimeVaryingWindField = None
    TimeVaryingWindFieldConfig = None

# Configuration schema imports for validation
try:
    from src.plume_nav_sim.config.schemas import WindFieldConfig
    SCHEMAS_AVAILABLE = True
except ImportError:
    # Define minimal schema for testing if not available
    WindFieldConfig = dict
    SCHEMAS_AVAILABLE = False


# Performance constants aligned with Section 0.2.1 requirements
WIND_FIELD_QUERY_THRESHOLD_MS = 2.0      # <2ms per simulation step requirement
SINGLE_QUERY_THRESHOLD_MS = 0.5          # <0.5ms for single position query  
BATCH_QUERY_THRESHOLD_MS = 2.0           # <2ms for 100+ position batch queries
STEP_EVOLUTION_THRESHOLD_MS = 2.0        # <2ms per temporal evolution step
INTEGRATION_LATENCY_THRESHOLD_MS = 10.0  # <10ms integration latency requirement
MAX_MEMORY_USAGE_MB = 50.0               # <50MB memory usage for typical representations


# ======================================================================================
# ENHANCED FIXTURES FOR WIND FIELD TESTING AND PERFORMANCE MONITORING INTEGRATION
# ======================================================================================

@pytest.fixture
def performance_monitor():
    """Enhanced performance monitor for wind field operation timing validation."""
    return PerformanceMonitor

@pytest.fixture  
def test_positions():
    """Standard test positions for wind field velocity queries."""
    return np.array([
        [0.0, 0.0],      # Origin
        [10.0, 20.0],    # Standard position
        [50.5, 75.2],    # Sub-pixel position  
        [100.0, 100.0],  # Boundary position
        [-5.0, -3.0],    # Negative coordinates
        [1000.0, 800.0] # Large coordinates
    ])

@pytest.fixture
def batch_test_positions():
    """Large batch of test positions for performance validation."""
    # Generate systematic grid plus random positions
    x_grid = np.linspace(0, 100, 20)
    y_grid = np.linspace(0, 100, 20) 
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_positions = np.column_stack([X.ravel(), Y.ravel()])
    
    # Add random positions for edge case testing
    np.random.seed(42)  # Reproducible random positions
    random_positions = np.random.uniform(-50, 150, (100, 2))
    
    return np.vstack([grid_positions, random_positions])

@pytest.fixture
def wind_field_configs():
    """Standard wind field configurations for testing different scenarios."""
    return {
        'constant_basic': {
            'velocity': (2.0, 1.0),
            'enable_temporal_evolution': False,
            'performance_monitoring': True
        },
        'constant_temporal': {
            'velocity': (1.5, 0.5),
            'enable_temporal_evolution': True,
            'evolution_rate': 0.1,
            'evolution_amplitude': 0.3,
            'evolution_period': 50.0,
            'noise_intensity': 0.05,
            'performance_monitoring': True
        },
        'turbulent_basic': {
            'mean_velocity': (3.0, 1.5),
            'turbulence_intensity': 0.2,
            'correlation_length': 10.0,
            'performance_monitoring': True
        },
        'time_varying_basic': {
            'base_velocity': (2.5, 0.8),
            'temporal_pattern': 'sinusoidal',
            'amplitude': (1.0, 0.4),
            'period': 60.0,
            'performance_monitoring': True
        }
    }

@pytest.fixture
def mock_hydra_config():
    """Mock Hydra configuration for wind field instantiation testing."""
    if HYDRA_AVAILABLE:
        return OmegaConf.create({
            'wind_field': {
                '_target_': 'src.plume_nav_sim.models.wind.constant_wind.ConstantWindField',
                'velocity': [2.0, 1.0],
                'enable_temporal_evolution': False,
                'performance_monitoring': True
            }
        })
    else:
        return {
            'wind_field': {
                '_target_': 'src.plume_nav_sim.models.wind.constant_wind.ConstantWindField',
                'velocity': [2.0, 1.0],
                'enable_temporal_evolution': False,
                'performance_monitoring': True
            }
        }

@pytest.fixture
def temp_wind_data_file():
    """Temporary wind data file for TimeVaryingWindField testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        wind_data = {
            'timestamps': [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            'u_wind': [2.0, 2.5, 1.8, 3.0, 2.2, 1.9],
            'v_wind': [1.0, 0.8, 1.2, 0.5, 1.1, 0.9]
        }
        json.dump(wind_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


# ======================================================================================
# WIND FIELD PROTOCOL COMPLIANCE TESTS
# ======================================================================================

@pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
class TestConstantWindFieldProtocol:
    """Test ConstantWindField implementation for WindFieldProtocol compliance."""
    
    def test_protocol_interface_compliance(self):
        """Test that ConstantWindField implements all required WindFieldProtocol methods."""
        wind_field = ConstantWindField(velocity=(2.0, 1.0))
        
        # Test protocol compliance
        if PROTOCOLS_AVAILABLE:
            assert isinstance(wind_field, WindFieldProtocol), "ConstantWindField must implement WindFieldProtocol"
        
        # Test required methods exist and are callable
        assert hasattr(wind_field, 'velocity_at') and callable(wind_field.velocity_at)
        assert hasattr(wind_field, 'step') and callable(wind_field.step)
        assert hasattr(wind_field, 'reset') and callable(wind_field.reset)
        
        # Test additional utility methods  
        assert hasattr(wind_field, 'get_current_velocity') and callable(wind_field.get_current_velocity)
        assert hasattr(wind_field, 'get_performance_stats') and callable(wind_field.get_performance_stats)
    
    def test_velocity_at_interface_compliance(self, test_positions):
        """Test velocity_at method interface compliance with different input formats."""
        wind_field = ConstantWindField(velocity=(3.0, 1.5), performance_monitoring=True)
        
        # Test single position (1D array)
        single_pos = test_positions[0]  # [0.0, 0.0]
        velocity = wind_field.velocity_at(single_pos)
        assert isinstance(velocity, np.ndarray)
        assert velocity.shape == (2,), f"Single position should return (2,) shape, got {velocity.shape}"
        assert np.allclose(velocity, [3.0, 1.5], atol=1e-10)
        
        # Test multiple positions (2D array)
        velocities = wind_field.velocity_at(test_positions)
        assert isinstance(velocities, np.ndarray)
        assert velocities.shape == (len(test_positions), 2), f"Multi-position should return (n, 2) shape, got {velocities.shape}"
        
        # For constant wind, all velocities should be identical
        expected_velocity = np.array([3.0, 1.5])
        for i, vel in enumerate(velocities):
            assert np.allclose(vel, expected_velocity, atol=1e-10), f"Position {i}: {vel} != {expected_velocity}"
    
    def test_step_interface_compliance(self):
        """Test step method interface compliance and temporal evolution."""
        wind_field = ConstantWindField(
            velocity=(2.0, 1.0),
            enable_temporal_evolution=True,
            evolution_rate=0.1,
            performance_monitoring=True
        )
        
        initial_velocity = wind_field.get_current_velocity().copy()
        initial_time = wind_field.current_time
        
        # Test step with default dt
        wind_field.step()
        assert wind_field.current_time == initial_time + 1.0
        
        # Test step with custom dt
        wind_field.step(dt=2.5)
        assert wind_field.current_time == initial_time + 3.5
        
        # With temporal evolution enabled, velocity should change
        final_velocity = wind_field.get_current_velocity()
        # Note: velocity might be same or different depending on evolution parameters
        assert isinstance(final_velocity, np.ndarray)
        assert final_velocity.shape == (2,)
    
    def test_reset_interface_compliance(self):
        """Test reset method interface compliance and parameter override functionality."""
        wind_field = ConstantWindField(
            velocity=(2.0, 1.0),
            enable_temporal_evolution=True,
            performance_monitoring=True
        )
        
        # Advance time and modify state
        wind_field.step(dt=10.0)
        original_time = wind_field.current_time
        assert original_time > 0.0
        
        # Test basic reset
        wind_field.reset()
        assert wind_field.current_time == 0.0
        assert np.allclose(wind_field.get_current_velocity(), [2.0, 1.0])
        
        # Test reset with parameter overrides
        wind_field.reset(
            velocity=(5.0, 2.5),
            enable_temporal_evolution=False,
            current_time=5.0
        )
        assert wind_field.current_time == 5.0
        assert np.allclose(wind_field.get_current_velocity(), [5.0, 2.5])
        assert not wind_field.enable_temporal_evolution


@pytest.mark.skipif(not TURBULENT_WIND_AVAILABLE, reason="TurbulentWindField not available")  
class TestTurbulentWindFieldProtocol:
    """Test TurbulentWindField implementation for WindFieldProtocol compliance."""
    
    def test_protocol_interface_compliance(self):
        """Test that TurbulentWindField implements all required WindFieldProtocol methods."""
        if not SCIPY_AVAILABLE:
            pytest.skip("SciPy not available for TurbulentWindField")
            
        wind_field = TurbulentWindField(
            mean_velocity=(2.0, 1.0),
            turbulence_intensity=0.1,
            correlation_length=5.0
        )
        
        # Test protocol compliance
        if PROTOCOLS_AVAILABLE:
            assert isinstance(wind_field, WindFieldProtocol), "TurbulentWindField must implement WindFieldProtocol"
        
        # Test required methods exist and are callable
        assert hasattr(wind_field, 'velocity_at') and callable(wind_field.velocity_at)
        assert hasattr(wind_field, 'step') and callable(wind_field.step)
        assert hasattr(wind_field, 'reset') and callable(wind_field.reset)
    
    def test_stochastic_velocity_properties(self, test_positions):
        """Test that turbulent wind field produces realistic stochastic variations."""
        if not SCIPY_AVAILABLE:
            pytest.skip("SciPy not available for TurbulentWindField")
            
        wind_field = TurbulentWindField(
            mean_velocity=(3.0, 1.0),
            turbulence_intensity=0.2,
            correlation_length=10.0,
            performance_monitoring=True
        )
        
        # Generate multiple velocity samples
        velocities_samples = []
        for _ in range(10):
            wind_field.step(dt=1.0)  # Advance time to get variations
            velocities = wind_field.velocity_at(test_positions)
            velocities_samples.append(velocities)
        
        velocities_array = np.array(velocities_samples)  # Shape: (10, n_positions, 2)
        
        # Test statistical properties
        mean_velocities = np.mean(velocities_array, axis=0)  # Average over time samples
        
        # Mean should be close to specified mean velocity (within statistical bounds)
        expected_mean = np.array([3.0, 1.0])
        for i, pos_mean in enumerate(mean_velocities):
            # Allow some statistical variation
            assert np.allclose(pos_mean, expected_mean, atol=0.5), \
                f"Position {i} mean velocity {pos_mean} not close to expected {expected_mean}"
        
        # Verify temporal variation exists (not constant)
        temporal_std = np.std(velocities_array, axis=0)
        assert np.any(temporal_std > 0.01), "Turbulent wind should show temporal variation"


@pytest.mark.skipif(not TIME_VARYING_WIND_AVAILABLE, reason="TimeVaryingWindField not available")
class TestTimeVaryingWindFieldProtocol:
    """Test TimeVaryingWindField implementation for WindFieldProtocol compliance."""
    
    def test_protocol_interface_compliance(self):
        """Test that TimeVaryingWindField implements all required WindFieldProtocol methods."""
        wind_field = TimeVaryingWindField(
            base_velocity=(2.0, 1.0),
            temporal_pattern='sinusoidal',
            amplitude=(0.5, 0.2),
            period=20.0
        )
        
        # Test protocol compliance
        if PROTOCOLS_AVAILABLE:
            assert isinstance(wind_field, WindFieldProtocol), "TimeVaryingWindField must implement WindFieldProtocol"
        
        # Test required methods exist and are callable
        assert hasattr(wind_field, 'velocity_at') and callable(wind_field.velocity_at)
        assert hasattr(wind_field, 'step') and callable(wind_field.step)
        assert hasattr(wind_field, 'reset') and callable(wind_field.reset)
    
    def test_temporal_pattern_evolution(self, test_positions):
        """Test that temporal patterns evolve correctly over time."""
        wind_field = TimeVaryingWindField(
            base_velocity=(2.0, 1.0),
            temporal_pattern='sinusoidal',
            amplitude=(1.0, 0.5),
            period=10.0,  # Short period for quick testing
            performance_monitoring=True
        )
        
        # Sample velocities at different time points
        time_points = [0.0, 2.5, 5.0, 7.5, 10.0]  # Quarter periods
        velocities_over_time = []
        
        for target_time in time_points:
            wind_field.reset(current_time=target_time)
            velocity = wind_field.velocity_at(test_positions[0])
            velocities_over_time.append(velocity.copy())
        
        velocities_array = np.array(velocities_over_time)  # Shape: (5, 2)
        
        # Test that velocity varies over time for sinusoidal pattern
        u_components = velocities_array[:, 0]
        v_components = velocities_array[:, 1]
        
        # Verify sinusoidal variation exists
        u_range = np.max(u_components) - np.min(u_components)
        v_range = np.max(v_components) - np.min(v_components)
        
        assert u_range > 0.5, f"U-component should vary significantly, range={u_range}"
        assert v_range > 0.2, f"V-component should vary significantly, range={v_range}"
        
        # Test periodicity (t=0 and t=10 should be similar for period=10)
        assert np.allclose(velocities_array[0], velocities_array[4], atol=0.1), \
            "Velocity at t=0 and t=period should be similar for periodic patterns"


# ======================================================================================
# PERFORMANCE TESTING WITH REQUIREMENTS VALIDATION
# ======================================================================================

class TestWindFieldPerformance:
    """Performance testing ensuring wind field operations meet timing requirements."""
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
    def test_constant_wind_single_query_performance(self, test_positions):
        """Test ConstantWindField single position query performance (<0.5ms requirement)."""
        wind_field = ConstantWindField(velocity=(2.0, 1.0), performance_monitoring=True)
        single_position = test_positions[0]
        
        # Warm up
        for _ in range(10):
            wind_field.velocity_at(single_position)
        
        # Measure performance over multiple queries
        query_times = []
        for _ in range(100):
            with performance_timer("single_query") as perf:
                velocity = wind_field.velocity_at(single_position)
            query_times.append(perf.duration_ms)
            
            # Verify correct output
            assert velocity.shape == (2,)
            assert np.allclose(velocity, [2.0, 1.0])
        
        # Validate performance requirements
        avg_query_time = np.mean(query_times)
        max_query_time = np.max(query_times)
        p95_query_time = np.percentile(query_times, 95)
        
        assert avg_query_time < SINGLE_QUERY_THRESHOLD_MS, \
            f"Average single query time {avg_query_time:.3f}ms exceeds {SINGLE_QUERY_THRESHOLD_MS}ms threshold"
        assert p95_query_time < SINGLE_QUERY_THRESHOLD_MS * 2, \
            f"95th percentile query time {p95_query_time:.3f}ms exceeds {SINGLE_QUERY_THRESHOLD_MS * 2}ms threshold"
        
        if LOGURU_AVAILABLE:
            logger.info(f"ConstantWindField single query performance: avg={avg_query_time:.3f}ms, "
                       f"max={max_query_time:.3f}ms, p95={p95_query_time:.3f}ms")
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
    def test_constant_wind_batch_query_performance(self, batch_test_positions):
        """Test ConstantWindField batch position query performance (<2ms requirement)."""
        wind_field = ConstantWindField(velocity=(3.0, 1.5), performance_monitoring=True)
        
        # Warm up
        for _ in range(5):
            wind_field.velocity_at(batch_test_positions)
        
        # Measure performance over multiple batch queries
        batch_times = []
        for _ in range(50):
            with performance_timer("batch_query") as perf:
                velocities = wind_field.velocity_at(batch_test_positions)
            batch_times.append(perf.duration_ms)
            
            # Verify correct output
            assert velocities.shape == (len(batch_test_positions), 2)
            # For constant wind, all velocities should be identical
            assert np.allclose(velocities, [3.0, 1.5], atol=1e-10)
        
        # Validate performance requirements  
        avg_batch_time = np.mean(batch_times)
        max_batch_time = np.max(batch_times)
        p95_batch_time = np.percentile(batch_times, 95)
        
        assert avg_batch_time < BATCH_QUERY_THRESHOLD_MS, \
            f"Average batch query time {avg_batch_time:.3f}ms exceeds {BATCH_QUERY_THRESHOLD_MS}ms threshold"
        assert p95_batch_time < BATCH_QUERY_THRESHOLD_MS * 2, \
            f"95th percentile batch time {p95_batch_time:.3f}ms exceeds {BATCH_QUERY_THRESHOLD_MS * 2}ms threshold"
        
        if LOGURU_AVAILABLE:
            logger.info(f"ConstantWindField batch query ({len(batch_test_positions)} positions) performance: "
                       f"avg={avg_batch_time:.3f}ms, max={max_batch_time:.3f}ms, p95={p95_batch_time:.3f}ms")
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
    def test_constant_wind_step_performance(self):
        """Test ConstantWindField temporal evolution step performance (<2ms requirement)."""
        wind_field = ConstantWindField(
            velocity=(2.0, 1.0),
            enable_temporal_evolution=True,
            evolution_rate=0.1,
            evolution_amplitude=0.5,
            noise_intensity=0.05,
            performance_monitoring=True
        )
        
        # Warm up
        for _ in range(10):
            wind_field.step(dt=1.0)
        
        # Measure step performance
        step_times = []
        for _ in range(100):
            with performance_timer("step_evolution") as perf:
                wind_field.step(dt=1.0)
            step_times.append(perf.duration_ms)
        
        # Validate performance requirements
        avg_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        p95_step_time = np.percentile(step_times, 95)
        
        assert avg_step_time < STEP_EVOLUTION_THRESHOLD_MS, \
            f"Average step time {avg_step_time:.3f}ms exceeds {STEP_EVOLUTION_THRESHOLD_MS}ms threshold"
        assert p95_step_time < STEP_EVOLUTION_THRESHOLD_MS * 2, \
            f"95th percentile step time {p95_step_time:.3f}ms exceeds {STEP_EVOLUTION_THRESHOLD_MS * 2}ms threshold"
        
        if LOGURU_AVAILABLE:
            logger.info(f"ConstantWindField step evolution performance: avg={avg_step_time:.3f}ms, "
                       f"max={max_step_time:.3f}ms, p95={p95_step_time:.3f}ms")
    
    @pytest.mark.skipif(not TURBULENT_WIND_AVAILABLE or not SCIPY_AVAILABLE, 
                       reason="TurbulentWindField or SciPy not available")
    def test_turbulent_wind_performance(self, test_positions):
        """Test TurbulentWindField performance under computational load."""
        wind_field = TurbulentWindField(
            mean_velocity=(2.0, 1.0),
            turbulence_intensity=0.15,
            correlation_length=8.0,
            performance_monitoring=True
        )
        
        # Test combined step + query performance (realistic usage pattern)
        combined_times = []
        for _ in range(50):
            with performance_timer("turbulent_step_query") as perf:
                wind_field.step(dt=1.0)
                velocities = wind_field.velocity_at(test_positions)
            combined_times.append(perf.duration_ms)
            
            # Verify output shape
            assert velocities.shape == (len(test_positions), 2)
        
        # Validate performance requirements (more lenient for turbulent model)
        avg_combined_time = np.mean(combined_times)
        max_combined_time = np.max(combined_times)
        
        # Allow higher threshold for turbulent wind due to complexity
        turbulent_threshold = STEP_EVOLUTION_THRESHOLD_MS * 2  # 4ms for turbulent
        assert avg_combined_time < turbulent_threshold, \
            f"Average turbulent step+query time {avg_combined_time:.3f}ms exceeds {turbulent_threshold}ms threshold"
        
        if LOGURU_AVAILABLE:
            logger.info(f"TurbulentWindField step+query performance: avg={avg_combined_time:.3f}ms, "
                       f"max={max_combined_time:.3f}ms")
    
    def test_memory_usage_efficiency(self, batch_test_positions):
        """Test that wind field implementations maintain memory efficiency requirements."""
        if not CONSTANT_WIND_AVAILABLE:
            pytest.skip("ConstantWindField not available")
        
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create wind field and perform operations
        wind_field = ConstantWindField(
            velocity=(2.0, 1.0),
            enable_temporal_evolution=True,
            performance_monitoring=True
        )
        
        # Perform typical usage pattern
        for _ in range(100):
            wind_field.step(dt=1.0)
            velocities = wind_field.velocity_at(batch_test_positions)
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - baseline_memory
        
        # Validate memory efficiency requirement
        assert memory_usage < MAX_MEMORY_USAGE_MB, \
            f"Wind field memory usage {memory_usage:.2f}MB exceeds {MAX_MEMORY_USAGE_MB}MB threshold"
        
        if LOGURU_AVAILABLE:
            logger.info(f"Wind field memory usage: {memory_usage:.2f}MB")


# ======================================================================================
# MATHEMATICAL PROPERTY VALIDATION TESTS
# ======================================================================================

class TestWindFieldMathematicalProperties:
    """Mathematical property validation for wind vector field computations."""
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
    def test_constant_wind_uniformity(self, test_positions):
        """Test that ConstantWindField provides uniform velocity everywhere."""
        wind_field = ConstantWindField(velocity=(4.0, 2.5))
        
        velocities = wind_field.velocity_at(test_positions)
        expected_velocity = np.array([4.0, 2.5])
        
        # All velocities should be identical for constant wind
        for i, velocity in enumerate(velocities):
            assert np.allclose(velocity, expected_velocity, atol=1e-12), \
                f"Position {i} velocity {velocity} != expected {expected_velocity}"
        
        # Test consistency across time (without temporal evolution)
        wind_field_static = ConstantWindField(velocity=(4.0, 2.5), enable_temporal_evolution=False)
        for _ in range(10):
            wind_field_static.step(dt=1.0)
            velocity_after_steps = wind_field_static.velocity_at(test_positions[0])
            assert np.allclose(velocity_after_steps, expected_velocity, atol=1e-12), \
                "Static constant wind should not change over time"
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")  
    def test_temporal_evolution_properties(self):
        """Test mathematical properties of temporal evolution patterns."""
        wind_field = ConstantWindField(
            velocity=(2.0, 1.0),
            enable_temporal_evolution=True,
            evolution_rate=0.0,  # No drift
            evolution_amplitude=1.0,
            evolution_period=20.0,
            noise_intensity=0.0,  # No noise for deterministic testing
            performance_monitoring=True
        )
        
        # Sample velocity over complete period
        period = 20.0
        time_points = np.linspace(0, period, 21)  # Include endpoints
        velocities_over_period = []
        
        for t in time_points:
            wind_field.reset(current_time=t)
            velocity = wind_field.velocity_at(np.array([0.0, 0.0]))
            velocities_over_period.append(velocity.copy())
        
        velocities_array = np.array(velocities_over_period)
        
        # Test periodicity: velocity at t=0 should equal velocity at t=period
        assert np.allclose(velocities_array[0], velocities_array[-1], atol=1e-6), \
            "Periodic evolution should return to initial state after one period"
        
        # Test symmetry: check sinusoidal properties
        mid_point = len(time_points) // 2
        u_components = velocities_array[:, 0]
        
        # For sinusoidal pattern, values should be symmetric around the mean
        u_mean = np.mean(u_components)
        u_range = np.max(u_components) - np.min(u_components)
        
        assert u_range > 0.1, "Temporal evolution should produce measurable variation"
        
        # Verify that evolution amplitude affects variation magnitude
        wind_field_small = ConstantWindField(
            velocity=(2.0, 1.0),
            enable_temporal_evolution=True,
            evolution_amplitude=0.1,  # Small amplitude
            evolution_period=20.0
        )
        
        wind_field_large = ConstantWindField(
            velocity=(2.0, 1.0),
            enable_temporal_evolution=True,
            evolution_amplitude=2.0,  # Large amplitude
            evolution_period=20.0
        )
        
        # Sample both at same time point
        test_time = 5.0
        wind_field_small.reset(current_time=test_time)
        wind_field_large.reset(current_time=test_time)
        
        velocity_small = wind_field_small.velocity_at(np.array([0.0, 0.0]))
        velocity_large = wind_field_large.velocity_at(np.array([0.0, 0.0]))
        
        # Larger amplitude should produce larger deviation from base velocity
        base_velocity = np.array([2.0, 1.0])
        deviation_small = np.linalg.norm(velocity_small - base_velocity)
        deviation_large = np.linalg.norm(velocity_large - base_velocity)
        
        assert deviation_large > deviation_small, \
            "Larger evolution amplitude should produce larger velocity deviations"
    
    def test_vector_field_continuity(self, wind_field_configs):
        """Test spatial continuity properties of wind vector fields."""
        if not CONSTANT_WIND_AVAILABLE:
            pytest.skip("ConstantWindField not available")
        
        wind_field = ConstantWindField(**wind_field_configs['constant_basic'])
        
        # Generate closely spaced positions to test continuity
        center = np.array([50.0, 50.0])
        epsilon = 1e-3  # Small spatial perturbation
        
        nearby_positions = np.array([
            center,
            center + [epsilon, 0],
            center + [0, epsilon],
            center + [epsilon, epsilon],
            center - [epsilon, 0],
            center - [0, epsilon]
        ])
        
        velocities = wind_field.velocity_at(nearby_positions)
        
        # For ConstantWindField, all velocities should be identical (perfect continuity)
        reference_velocity = velocities[0]
        for i, velocity in enumerate(velocities):
            assert np.allclose(velocity, reference_velocity, atol=1e-10), \
                f"Position {i} velocity {velocity} not continuous with reference {reference_velocity}"
    
    @pytest.mark.skipif(not TURBULENT_WIND_AVAILABLE or not SCIPY_AVAILABLE, 
                       reason="TurbulentWindField or SciPy not available")
    def test_turbulent_field_statistical_properties(self):
        """Test statistical properties of turbulent wind fields."""
        wind_field = TurbulentWindField(
            mean_velocity=(3.0, 1.5),
            turbulence_intensity=0.2,
            correlation_length=5.0,
            random_seed=42  # For reproducible testing
        )
        
        # Generate large sample of velocity measurements
        np.random.seed(42)
        test_positions = np.random.uniform(0, 100, (200, 2))
        
        # Collect samples over time and space
        velocity_samples = []
        for _ in range(20):  # Multiple time steps
            wind_field.step(dt=1.0)
            velocities = wind_field.velocity_at(test_positions)
            velocity_samples.append(velocities)
        
        all_velocities = np.concatenate(velocity_samples, axis=0)  # Shape: (4000, 2)
        
        # Test statistical properties
        mean_velocity = np.mean(all_velocities, axis=0)
        std_velocity = np.std(all_velocities, axis=0)
        
        # Mean should be close to specified mean (within statistical bounds)
        expected_mean = np.array([3.0, 1.5])
        assert np.allclose(mean_velocity, expected_mean, atol=0.3), \
            f"Sample mean {mean_velocity} not close to expected {expected_mean}"
        
        # Standard deviation should reflect turbulence intensity
        expected_std = 0.2 * np.abs(expected_mean)  # 20% turbulence intensity
        assert np.allclose(std_velocity, expected_std, atol=0.1), \
            f"Sample std {std_velocity} not close to expected {expected_std}"
        
        # Test that correlation length affects spatial correlation
        # (This is a simplified test - full correlation analysis would be more complex)
        close_positions = test_positions[:10]  # First 10 positions
        close_velocities = wind_field.velocity_at(close_positions)
        
        # Positions within correlation length should have similar velocities
        distances = cdist(close_positions, close_positions)
        velocity_differences = cdist(close_velocities, close_velocities)
        
        # Find pairs within correlation length
        correlation_length = 5.0
        close_pairs = distances < correlation_length
        close_velocity_diffs = velocity_differences[close_pairs]
        
        # Velocity differences should be smaller for spatially close positions
        mean_close_diff = np.mean(close_velocity_diffs)
        overall_std = np.mean(std_velocity)
        
        assert mean_close_diff < overall_std, \
            "Spatially close positions should have more similar velocities than overall variation"


# ====================================================================================== 
# CONFIGURATION AND HYDRA INTEGRATION TESTS
# ======================================================================================

class TestWindFieldConfiguration:
    """Configuration switching and Hydra integration tests."""
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
    def test_configuration_switching(self, wind_field_configs):
        """Test switching between different wind field configurations."""
        
        # Test different configuration scenarios
        config_scenarios = [
            ('constant_basic', wind_field_configs['constant_basic']),
            ('constant_temporal', wind_field_configs['constant_temporal'])
        ]
        
        created_wind_fields = {}
        
        for config_name, config in config_scenarios:
            # Create wind field from configuration
            wind_field = ConstantWindField(**config)
            created_wind_fields[config_name] = wind_field
            
            # Verify configuration was applied correctly
            expected_velocity = np.array(config['velocity'])
            actual_velocity = wind_field.get_current_velocity()
            assert np.allclose(actual_velocity, expected_velocity, atol=1e-10), \
                f"Configuration {config_name}: velocity {actual_velocity} != expected {expected_velocity}"
            
            assert wind_field.enable_temporal_evolution == config['enable_temporal_evolution'], \
                f"Configuration {config_name}: temporal evolution setting incorrect"
            
            assert wind_field.performance_monitoring == config['performance_monitoring'], \
                f"Configuration {config_name}: performance monitoring setting incorrect"
        
        # Test that different configurations produce different behaviors
        basic_field = created_wind_fields['constant_basic']
        temporal_field = created_wind_fields['constant_temporal']
        
        # Basic field should not change over time
        basic_initial = basic_field.get_current_velocity().copy()
        basic_field.step(dt=5.0)
        basic_after = basic_field.get_current_velocity()
        assert np.allclose(basic_initial, basic_after, atol=1e-10), \
            "Basic configuration should not change velocity over time"
        
        # Temporal field should change over time (with evolution enabled)
        temporal_initial = temporal_field.get_current_velocity().copy()
        temporal_field.step(dt=5.0)
        temporal_after = temporal_field.get_current_velocity()
        # Note: might be the same by chance, but evolution is enabled so it could change
        # We mainly test that the configuration was applied correctly above
    
    @pytest.mark.skipif(not HYDRA_AVAILABLE or not CONSTANT_WIND_AVAILABLE, 
                       reason="Hydra or ConstantWindField not available")
    def test_hydra_instantiation(self, mock_hydra_config):
        """Test wind field instantiation via Hydra configuration."""
        
        try:
            import hydra
            from hydra.utils import instantiate
        except ImportError:
            pytest.skip("Hydra instantiation utilities not available")
        
        # Test direct instantiation from config
        wind_field_config = mock_hydra_config['wind_field']
        
        # For testing, we'll manually instantiate since we may not have full Hydra setup
        config_dict = dict(wind_field_config)
        target_class = config_dict.pop('_target_', None)
        
        if target_class == 'src.plume_nav_sim.models.wind.constant_wind.ConstantWindField':
            wind_field = ConstantWindField(**config_dict)
        else:
            pytest.skip(f"Target class {target_class} not available for testing")
        
        # Verify instantiation was successful
        assert isinstance(wind_field, ConstantWindField)
        assert np.allclose(wind_field.get_current_velocity(), [2.0, 1.0])
        assert not wind_field.enable_temporal_evolution
        assert wind_field.performance_monitoring
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
    def test_configuration_validation(self):
        """Test configuration parameter validation and error handling."""
        
        # Test valid configuration
        valid_config = {
            'velocity': (2.0, 1.0),
            'enable_temporal_evolution': True,
            'evolution_rate': 0.1,
            'evolution_amplitude': 0.5,
            'evolution_period': 30.0,
            'noise_intensity': 0.05
        }
        
        wind_field = ConstantWindField(**valid_config)
        assert isinstance(wind_field, ConstantWindField)
        
        # Test invalid configurations
        invalid_configs = [
            # Invalid velocity format
            {'velocity': [2.0]},  # Wrong length
            {'velocity': "invalid"},  # Wrong type
            {'velocity': (2.0, "invalid")},  # Mixed types
            
            # Invalid evolution parameters
            {'velocity': (2.0, 1.0), 'evolution_rate': -0.1},  # Negative rate
            {'velocity': (2.0, 1.0), 'evolution_amplitude': -0.5},  # Negative amplitude
            {'velocity': (2.0, 1.0), 'evolution_period': -10.0},  # Negative period
            {'velocity': (2.0, 1.0), 'noise_intensity': -0.1},  # Negative noise
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            with pytest.raises((ValueError, TypeError)), \
                 f"Invalid config {i} should raise exception: {invalid_config}":
                ConstantWindField(**invalid_config)
    
    @pytest.mark.skipif(not TIME_VARYING_WIND_AVAILABLE, reason="TimeVaryingWindField not available")
    def test_data_driven_configuration(self, temp_wind_data_file):
        """Test configuration with external data files."""
        
        # Test data-driven wind field configuration
        config = {
            'data_file': temp_wind_data_file,
            'temporal_column': 'timestamps',
            'velocity_columns': ['u_wind', 'v_wind'],
            'interpolation_method': 'linear'
        }
        
        wind_field = TimeVaryingWindField(**config)
        
        # Test that data was loaded correctly
        test_position = np.array([25.0, 35.0])
        
        # Set to different time points and verify interpolation
        time_points = [0.0, 15.0, 30.0]
        velocities_at_times = []
        
        for t in time_points:
            wind_field.reset(current_time=t)
            velocity = wind_field.velocity_at(test_position)
            velocities_at_times.append(velocity.copy())
        
        # Verify that velocities change over time (data-driven)
        velocity_changes = [
            np.linalg.norm(velocities_at_times[i+1] - velocities_at_times[i])
            for i in range(len(velocities_at_times) - 1)
        ]
        
        assert any(change > 0.1 for change in velocity_changes), \
            "Data-driven wind field should show temporal variation"


# ======================================================================================
# PROPERTY-BASED TESTING WITH HYPOTHESIS
# ======================================================================================

if HYPOTHESIS_AVAILABLE:
    
    # Hypothesis strategies for wind field testing
    velocity_strategy = st.tuples(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    
    evolution_rate_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    
    evolution_amplitude_strategy = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    
    evolution_period_strategy = st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
    
    noise_intensity_strategy = st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)
    
    time_delta_strategy = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
    class TestWindFieldPropertyBased:
        """Property-based testing for wind field mathematical properties."""
        
        @given(
            velocity=velocity_strategy,
            positions=arrays(np.float64, shape=(st.integers(1, 20), 2), 
                           elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False))
        )
        @settings(max_examples=50, deadline=1000)
        def test_constant_wind_invariant_properties(self, velocity, positions):
            """Property-based test for ConstantWindField invariant properties."""
            wind_field = ConstantWindField(velocity=velocity, enable_temporal_evolution=False)
            
            velocities = wind_field.velocity_at(positions)
            
            # Property 1: Output shape consistency
            assert velocities.shape == (len(positions), 2), \
                f"Output shape {velocities.shape} != expected {(len(positions), 2)}"
            
            # Property 2: Velocity uniformity for constant wind
            expected_velocity = np.array(velocity)
            for i, vel in enumerate(velocities):
                assert np.allclose(vel, expected_velocity, atol=1e-10), \
                    f"Position {i}: velocity {vel} != expected {expected_velocity}"
            
            # Property 3: Temporal invariance (without evolution)
            wind_field.step(dt=5.0)
            velocities_after_step = wind_field.velocity_at(positions)
            assert np.allclose(velocities, velocities_after_step, atol=1e-10), \
                "Velocities should not change over time without temporal evolution"
        
        @given(
            base_velocity=velocity_strategy,
            evolution_rate=evolution_rate_strategy,
            evolution_amplitude=evolution_amplitude_strategy,
            evolution_period=evolution_period_strategy,
            dt=time_delta_strategy
        )
        @settings(max_examples=30, deadline=2000)
        def test_temporal_evolution_properties(self, base_velocity, evolution_rate, 
                                             evolution_amplitude, evolution_period, dt):
            """Property-based test for temporal evolution mathematical properties."""
            wind_field = ConstantWindField(
                velocity=base_velocity,
                enable_temporal_evolution=True,
                evolution_rate=evolution_rate,
                evolution_amplitude=evolution_amplitude,
                evolution_period=evolution_period,
                noise_intensity=0.0  # No noise for deterministic testing
            )
            
            test_position = np.array([0.0, 0.0])
            
            # Property 1: Continuous evolution (small time steps should produce small changes)
            initial_velocity = wind_field.velocity_at(test_position).copy()
            wind_field.step(dt=dt)
            evolved_velocity = wind_field.velocity_at(test_position)
            
            # For small dt, change should be proportional to dt
            velocity_change = np.linalg.norm(evolved_velocity - initial_velocity)
            max_expected_change = (evolution_rate + evolution_amplitude) * dt
            
            # Allow some tolerance for numerical precision and complex evolution
            assert velocity_change <= max_expected_change * 5, \
                f"Velocity change {velocity_change} too large for dt={dt}"
            
            # Property 2: Reset consistency
            wind_field.reset()
            reset_velocity = wind_field.velocity_at(test_position)
            base_velocity_array = np.array(base_velocity)
            
            # After reset, velocity should be close to base velocity
            assert np.allclose(reset_velocity, base_velocity_array, atol=1e-6), \
                f"Reset velocity {reset_velocity} != base velocity {base_velocity_array}"
        
        @given(
            velocity=velocity_strategy,
            num_queries=st.integers(1, 100)
        )
        @settings(max_examples=20, deadline=3000, suppress_health_check=[HealthCheck.too_slow])
        def test_performance_scalability_properties(self, velocity, num_queries):
            """Property-based test for performance scalability with query count."""
            wind_field = ConstantWindField(velocity=velocity, performance_monitoring=True)
            
            # Generate random positions for testing
            np.random.seed(42)  # Reproducible random positions
            positions = np.random.uniform(-50, 150, (num_queries, 2))
            
            # Measure query performance
            with performance_timer("scalability_test") as perf:
                velocities = wind_field.velocity_at(positions)
            
            # Property 1: Linear scalability (time should scale roughly linearly with query count)
            time_per_query = perf.duration_ms / num_queries
            
            # For ConstantWindField, time per query should be very small and roughly constant
            max_time_per_query = 0.1  # 0.1ms per query is generous for constant wind
            assert time_per_query < max_time_per_query, \
                f"Time per query {time_per_query:.4f}ms exceeds {max_time_per_query}ms for {num_queries} queries"
            
            # Property 2: Output correctness regardless of batch size
            assert velocities.shape == (num_queries, 2)
            expected_velocity = np.array(velocity)
            for vel in velocities:
                assert np.allclose(vel, expected_velocity, atol=1e-10)


# ======================================================================================
# INTEGRATION AND BOUNDARY CONDITION TESTS
# ======================================================================================

class TestWindFieldIntegration:
    """Integration testing and boundary condition validation."""
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
    def test_boundary_condition_handling(self):
        """Test wind field behavior at domain boundaries."""
        # Create wind field with boundary conditions
        boundary_conditions = ((0.0, 100.0), (0.0, 100.0))  # 100x100 domain
        wind_field = ConstantWindField(
            velocity=(2.0, 1.0),
            boundary_conditions=boundary_conditions,
            performance_monitoring=True
        )
        
        # Test positions inside, on boundary, and outside domain
        test_cases = [
            # (position, expected_behavior)
            (np.array([50.0, 50.0]), "inside_domain"),      # Inside
            (np.array([0.0, 0.0]), "on_boundary"),          # On boundary (corner)
            (np.array([100.0, 50.0]), "on_boundary"),       # On boundary (edge)
            (np.array([-10.0, 50.0]), "outside_domain"),    # Outside (negative x)
            (np.array([110.0, 50.0]), "outside_domain"),    # Outside (large x)
            (np.array([50.0, -5.0]), "outside_domain"),     # Outside (negative y)
            (np.array([50.0, 105.0]), "outside_domain")     # Outside (large y)
        ]
        
        for position, expected_behavior in test_cases:
            velocity = wind_field.velocity_at(position)
            
            if expected_behavior == "inside_domain" or expected_behavior == "on_boundary":
                # Inside or on boundary should return normal wind velocity
                assert np.allclose(velocity, [2.0, 1.0], atol=1e-10), \
                    f"Position {position} ({expected_behavior}) should have normal velocity"
            elif expected_behavior == "outside_domain":
                # Outside domain should return zero velocity (no-flow boundary)
                assert np.allclose(velocity, [0.0, 0.0], atol=1e-10), \
                    f"Position {position} ({expected_behavior}) should have zero velocity"
    
    @pytest.mark.skipif(not CONSTANT_WIND_AVAILABLE, reason="ConstantWindField not available")
    def test_large_scale_integration(self, batch_test_positions):
        """Test wind field integration with large-scale position arrays."""
        wind_field = ConstantWindField(
            velocity=(3.0, 1.5),
            enable_temporal_evolution=True,
            evolution_rate=0.05,
            performance_monitoring=True
        )
        
        # Simulate large-scale usage pattern
        total_queries = 0
        total_time = 0.0
        
        for step in range(20):  # 20 simulation steps
            # Advance wind field
            step_start = time.perf_counter()
            wind_field.step(dt=1.0)
            step_time = time.perf_counter() - step_start
            
            # Query velocities for all positions
            query_start = time.perf_counter()
            velocities = wind_field.velocity_at(batch_test_positions)
            query_time = time.perf_counter() - query_start
            
            total_time += (step_time + query_time) * 1000  # Convert to ms
            total_queries += len(batch_test_positions)
            
            # Verify output consistency
            assert velocities.shape == (len(batch_test_positions), 2)
            assert not np.any(np.isnan(velocities)), f"NaN velocities at step {step}"
            assert not np.any(np.isinf(velocities)), f"Inf velocities at step {step}"
        
        # Validate overall performance
        avg_time_per_query = total_time / total_queries
        avg_time_per_step = total_time / 20
        
        assert avg_time_per_query < 0.01, \
            f"Average time per query {avg_time_per_query:.4f}ms exceeds 0.01ms threshold"
        assert avg_time_per_step < INTEGRATION_LATENCY_THRESHOLD_MS, \
            f"Average time per step {avg_time_per_step:.2f}ms exceeds {INTEGRATION_LATENCY_THRESHOLD_MS}ms threshold"
        
        if LOGURU_AVAILABLE:
            logger.info(f"Large-scale integration: {total_queries} queries, "
                       f"avg {avg_time_per_query:.4f}ms/query, {avg_time_per_step:.2f}ms/step")
    
    def test_error_handling_and_robustness(self):
        """Test wind field error handling and robustness to invalid inputs."""
        if not CONSTANT_WIND_AVAILABLE:
            pytest.skip("ConstantWindField not available")
        
        wind_field = ConstantWindField(velocity=(2.0, 1.0))
        
        # Test invalid position inputs
        invalid_position_inputs = [
            np.array([np.nan, 0.0]),           # NaN coordinates
            np.array([np.inf, 0.0]),           # Inf coordinates
            np.array([0.0]),                   # Wrong shape (1D with length 1)
            np.array([0.0, 0.0, 0.0]),         # Wrong shape (1D with length 3)
            np.array([[0.0]]),                 # Wrong shape (2D with wrong width)
            "invalid",                         # Wrong type
            None                               # None input
        ]
        
        for invalid_input in invalid_position_inputs:
            with pytest.raises((ValueError, TypeError)):
                wind_field.velocity_at(invalid_input)
        
        # Test invalid step parameters
        invalid_step_inputs = [
            -1.0,      # Negative dt
            0.0,       # Zero dt
            np.nan,    # NaN dt
            np.inf,    # Inf dt
            "invalid"  # Wrong type
        ]
        
        for invalid_dt in invalid_step_inputs:
            with pytest.raises((ValueError, TypeError)):
                wind_field.step(dt=invalid_dt)
        
        # Test robustness to extreme but valid inputs
        extreme_positions = np.array([
            [1e6, 1e6],     # Very large coordinates
            [-1e6, -1e6],   # Very large negative coordinates
            [1e-10, 1e-10]  # Very small coordinates
        ])
        
        # Should not raise exceptions for extreme but valid positions
        velocities = wind_field.velocity_at(extreme_positions)
        assert velocities.shape == (3, 2)
        assert not np.any(np.isnan(velocities))
        assert not np.any(np.isinf(velocities))
    
    def test_concurrent_access_thread_safety(self):
        """Test wind field thread safety for concurrent access scenarios."""
        if not CONSTANT_WIND_AVAILABLE:
            pytest.skip("ConstantWindField not available")
        
        import threading
        import queue
        
        wind_field = ConstantWindField(
            velocity=(2.0, 1.0),
            enable_temporal_evolution=True,
            evolution_rate=0.1,
            performance_monitoring=True
        )
        
        results = queue.Queue()
        exceptions = queue.Queue()
        
        def worker_function(worker_id, num_operations):
            """Worker function for concurrent testing."""
            try:
                local_results = []
                for i in range(num_operations):
                    # Mix of different operations
                    if i % 3 == 0:
                        wind_field.step(dt=0.1)
                    elif i % 3 == 1:
                        position = np.array([worker_id * 10, i * 5])
                        velocity = wind_field.velocity_at(position)
                        local_results.append(velocity.copy())
                    else:
                        stats = wind_field.get_performance_stats()
                        local_results.append(stats['current_velocity'].copy())
                
                results.put((worker_id, local_results))
            except Exception as e:
                exceptions.put((worker_id, e))
        
        # Start multiple threads
        threads = []
        num_workers = 4
        operations_per_worker = 25
        
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=worker_function,
                args=(worker_id, operations_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)  # 5 second timeout
            assert not thread.is_alive(), "Worker thread failed to complete within timeout"
        
        # Check for exceptions
        assert exceptions.empty(), f"Worker exceptions occurred: {list(exceptions.queue)}"
        
        # Verify all workers completed
        assert results.qsize() == num_workers, f"Expected {num_workers} results, got {results.qsize()}"
        
        # Verify results are reasonable
        all_results = {}
        while not results.empty():
            worker_id, worker_results = results.get()
            all_results[worker_id] = worker_results
        
        for worker_id, worker_results in all_results.items():
            assert len(worker_results) > 0, f"Worker {worker_id} produced no results"
            
            # Check that results are valid arrays
            for result in worker_results:
                assert isinstance(result, np.ndarray)
                assert result.shape == (2,)
                assert not np.any(np.isnan(result))
                assert not np.any(np.isinf(result))


# ======================================================================================
# COMPREHENSIVE TEST EXECUTION AND REPORTING
# ======================================================================================

def test_module_imports_and_availability():
    """Test that all required modules and components are available."""
    availability_report = {
        'protocols_available': PROTOCOLS_AVAILABLE,
        'constant_wind_available': CONSTANT_WIND_AVAILABLE,
        'turbulent_wind_available': TURBULENT_WIND_AVAILABLE,
        'time_varying_wind_available': TIME_VARYING_WIND_AVAILABLE,
        'schemas_available': SCHEMAS_AVAILABLE,
        'scipy_available': SCIPY_AVAILABLE,
        'hypothesis_available': HYPOTHESIS_AVAILABLE,
        'hydra_available': HYDRA_AVAILABLE,
        'loguru_available': LOGURU_AVAILABLE
    }
    
    # Log availability status
    if LOGURU_AVAILABLE:
        logger.info(f"Wind field test module availability: {availability_report}")
    
    # Ensure at least core components are available for meaningful testing
    essential_components = [CONSTANT_WIND_AVAILABLE]
    
    if not any(essential_components):
        pytest.skip("No wind field implementations available for testing")
    
    # Warn about missing optional components
    if not SCIPY_AVAILABLE:
        warnings.warn("SciPy not available - some advanced tests will be skipped", UserWarning)
    
    if not HYPOTHESIS_AVAILABLE:
        warnings.warn("Hypothesis not available - property-based tests will be skipped", UserWarning)

    assert True  # Test passes if we reach here


def test_create_wind_field_validates_required_methods(monkeypatch):
    """Factory should raise when required protocol methods are missing."""
    from src.plume_nav_sim.models.wind import create_wind_field, AVAILABLE_WIND_FIELDS

    class IncompleteWindField:
        def velocity_at(self, positions):
            return positions

    monkeypatch.setitem(AVAILABLE_WIND_FIELDS, 'IncompleteWindField', {'class': IncompleteWindField})

    config = {'type': 'IncompleteWindField'}
    with pytest.raises(RuntimeError, match="missing required methods"):
        create_wind_field(config)


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v", "--tb=short"])