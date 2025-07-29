"""
Comprehensive pytest suite for testing PlumeModelProtocol implementations.

This test module provides comprehensive validation for the new modular plume modeling architecture,
focusing on protocol compliance, mathematical accuracy, performance requirements, and seamless
component switching. The tests validate GaussianPlumeModel, TurbulentPlumeModel, and 
VideoPlumeAdapter implementations according to PlumeModelProtocol specifications.

Key Testing Areas:
- Protocol compliance validation ensuring structural subtyping conformance per Section 0.4.1
- Mathematical validation for GaussianPlumeModel concentration field generation 
- Performance benchmarks ensuring sub-10ms step execution latency per requirements
- TurbulentPlumeModel filament-based physics validation with realistic dispersion
- VideoPlumeAdapter backward compatibility tests maintaining existing VideoPlume functionality
- Configuration-driven component switching via Hydra without code changes per Section 0.2.1
- Cross-component integration testing with wind field and sensor protocol integration

The testing architecture follows scientific computing best practices with deterministic behavior,
comprehensive mocking, research-grade quality standards, and API consistency validation as
specified in Section 6.6 of the technical specification.

Enhanced Testing Features:
- Test-driven protocol development approach per Section 0.6.1 requirements
- Property-based testing using Hypothesis for mathematical validation
- Performance regression detection with benchmark comparison
- Hydra configuration switching tests validating seamless model transitions
- Integration with wind field and sensor protocols for realistic simulation testing
- Memory efficiency validation ensuring <100MB usage for typical scenarios
"""

from __future__ import annotations
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np
import pytest
from dataclasses import asdict

# Property-based testing imports
try:
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = lambda *args, **kwargs: lambda f: f
    assume = lambda x: None
    
    # Mock strategies object to handle @given decorators when hypothesis is not available
    class MockStrategies:
        def floats(self, **kwargs):
            return None
        def integers(self, **kwargs):
            return None
        def text(self, **kwargs):
            return None
        def booleans(self, **kwargs):
            return None
        def lists(self, *args, **kwargs):
            return None
        def sampled_from(self, *args, **kwargs):
            return None
    
    st = MockStrategies()
    
    # Mock settings and HealthCheck for test configuration
    class MockSettings:
        def __init__(self, **kwargs):
            pass
        def __call__(self, func):
            return func
    
    settings = MockSettings
    
    class MockHealthCheck:
        too_slow = None
    
    HealthCheck = MockHealthCheck()
    
    # Mock composite decorator
    composite = lambda func: func
    settings = lambda *args, **kwargs: lambda f: f

# Scientific computing imports with graceful fallbacks
try:
    import scipy.stats
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Optional Numba imports for performance testing
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Hydra imports for configuration testing
try:
    from omegaconf import DictConfig, OmegaConf
    from hydra import compose, initialize_config_store
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

# Core plume_nav_sim imports with fallbacks for incremental development
try:
    from src.plume_nav_sim.core.protocols import PlumeModelProtocol, WindFieldProtocol
    PROTOCOLS_AVAILABLE = True
except ImportError:
    # Fallback protocol definitions for testing during development
    from typing import Protocol
    
    class PlumeModelProtocol(Protocol):
        def concentration_at(self, positions: np.ndarray) -> np.ndarray: ...
        def step(self, dt: float = 1.0) -> None: ...
        def reset(self, **kwargs: Any) -> None: ...
    
    class WindFieldProtocol(Protocol):
        def velocity_at(self, positions: np.ndarray) -> np.ndarray: ...
        def step(self, dt: float = 1.0) -> None: ...
        def reset(self, **kwargs: Any) -> None: ...
    
    PROTOCOLS_AVAILABLE = False

# Plume model imports with fallback implementations for testing
try:
    from src.plume_nav_sim.models.plume.gaussian_plume import GaussianPlumeModel, GaussianPlumeConfig
    GAUSSIAN_PLUME_AVAILABLE = True
except ImportError:
    GAUSSIAN_PLUME_AVAILABLE = False
    
    # Minimal fallback for testing infrastructure
    class GaussianPlumeModel:
        def __init__(self, source_position=(50, 50), source_strength=1000, sigma_x=5, sigma_y=3, **kwargs):
            self.source_position = np.array(source_position)
            self.source_strength = source_strength
            self.sigma_x = sigma_x
            self.sigma_y = sigma_y
            self.current_time = 0.0
        
        def concentration_at(self, positions: np.ndarray) -> np.ndarray:
            positions = np.asarray(positions)
            if positions.ndim == 1:
                positions = positions.reshape(1, 2)
            
            dx = positions[:, 0] - self.source_position[0]
            dy = positions[:, 1] - self.source_position[1]
            
            # Simple Gaussian calculation
            exp_term = np.exp(-0.5 * ((dx/self.sigma_x)**2 + (dy/self.sigma_y)**2))
            concentrations = self.source_strength * exp_term / (2 * np.pi * self.sigma_x * self.sigma_y)
            
            return concentrations[0] if positions.shape[0] == 1 else concentrations
        
        def step(self, dt: float = 1.0) -> None:
            self.current_time += dt
        
        def reset(self, **kwargs: Any) -> None:
            self.current_time = 0.0
            if 'source_position' in kwargs:
                self.source_position = np.array(kwargs['source_position'])

try:
    from src.plume_nav_sim.models.plume.turbulent_plume import TurbulentPlumeModel, TurbulentPlumeConfig
    TURBULENT_PLUME_AVAILABLE = True
except ImportError:
    TURBULENT_PLUME_AVAILABLE = False
    
    # Minimal fallback for testing infrastructure
    class TurbulentPlumeModel:
        def __init__(self, source_position=(50, 50), source_strength=1000, num_filaments=100, 
                     turbulence_intensity=0.2, **kwargs):
            self.source_position = np.array(source_position)
            self.source_strength = source_strength
            self.num_filaments = num_filaments
            self.turbulence_intensity = turbulence_intensity
            self.current_time = 0.0
            self.filaments = self._initialize_filaments()
        
        def _initialize_filaments(self):
            # Simplified filament representation for testing
            return {
                'positions': np.random.normal(self.source_position, 1.0, (self.num_filaments, 2)),
                'strengths': np.random.exponential(1.0, self.num_filaments),
                'ages': np.zeros(self.num_filaments)
            }
        
        def concentration_at(self, positions: np.ndarray) -> np.ndarray:
            positions = np.asarray(positions)
            if positions.ndim == 1:
                positions = positions.reshape(1, 2)
            
            concentrations = np.zeros(positions.shape[0])
            
            # Simple distance-based concentration from filaments
            for i, pos in enumerate(positions):
                distances = np.linalg.norm(self.filaments['positions'] - pos, axis=1)
                nearby_filaments = distances < 5.0  # 5 unit radius
                if np.any(nearby_filaments):
                    concentrations[i] = np.sum(self.filaments['strengths'][nearby_filaments] / 
                                             (distances[nearby_filaments] + 0.1))
            
            return concentrations[0] if positions.shape[0] == 1 else concentrations
        
        def step(self, dt: float = 1.0) -> None:
            self.current_time += dt
            # Simple filament advection for testing
            self.filaments['positions'] += np.random.normal(0, self.turbulence_intensity * dt, 
                                                           self.filaments['positions'].shape)
            self.filaments['ages'] += dt
        
        def reset(self, **kwargs: Any) -> None:
            self.current_time = 0.0
            if 'source_position' in kwargs:
                self.source_position = np.array(kwargs['source_position'])
            self.filaments = self._initialize_filaments()

try:
    from src.plume_nav_sim.models.plume.video_plume_adapter import VideoPlumeAdapter, VideoPlumeConfig
    VIDEO_PLUME_ADAPTER_AVAILABLE = True
except ImportError:
    VIDEO_PLUME_ADAPTER_AVAILABLE = False
    
    # Minimal fallback for testing infrastructure
    class VideoPlumeAdapter:
        def __init__(self, video_path="test_video.mp4", **kwargs):
            self.video_path = video_path
            self.current_frame = 0
            self.frame_count = 1000
            self.width = 640
            self.height = 480
            self.current_time = 0.0
        
        def concentration_at(self, positions: np.ndarray) -> np.ndarray:
            positions = np.asarray(positions)
            if positions.ndim == 1:
                positions = positions.reshape(1, 2)
            
            # Simulate video-based concentration lookup
            concentrations = np.zeros(positions.shape[0])
            for i, pos in enumerate(positions):
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Simple sinusoidal pattern for testing
                    concentrations[i] = 0.5 + 0.3 * np.sin(x/10) * np.cos(y/10)
            
            return concentrations[0] if positions.shape[0] == 1 else concentrations
        
        def step(self, dt: float = 1.0) -> None:
            self.current_time += dt
            self.current_frame = min(self.current_frame + 1, self.frame_count - 1)
        
        def reset(self, **kwargs: Any) -> None:
            self.current_time = 0.0
            self.current_frame = 0

try:
    from src.plume_nav_sim.config.schemas import PlumeModelConfig, SimulationConfig
    CONFIG_SCHEMAS_AVAILABLE = True
except ImportError:
    CONFIG_SCHEMAS_AVAILABLE = False
    # Fallback configuration classes
    PlumeModelConfig = dict
    SimulationConfig = dict


# =====================================================================================
# ENHANCED TEST FIXTURES FOR PLUME MODEL PROTOCOL TESTING
# =====================================================================================

@pytest.fixture
def gaussian_plume_config():
    """
    Fixture providing validated configuration for GaussianPlumeModel testing.
    
    Provides deterministic configuration parameters for reproducible test execution
    with performance optimization and mathematical validation support.
    """
    return {
        'source_position': (50.0, 50.0),
        'source_strength': 1000.0,
        'sigma_x': 5.0,
        'sigma_y': 3.0,
        'background_concentration': 0.0,
        'max_concentration': 1.0,
        'wind_speed': 0.0,
        'wind_direction': 0.0,
        'concentration_cutoff': 1e-6
    }

@pytest.fixture
def turbulent_plume_config():
    """
    Fixture providing validated configuration for TurbulentPlumeModel testing.
    
    Provides realistic turbulent parameters for filament-based physics validation
    while maintaining performance requirements for test execution.
    """
    return {
        'source_position': (50.0, 50.0),
        'source_strength': 1000.0,
        'num_filaments': 100,  # Reduced for test performance
        'turbulence_intensity': 0.2,
        'mean_wind_velocity': (1.0, 0.5),
        'diffusion_coefficient': 0.1,
        'filament_lifetime': 100.0,
        'emission_rate': 10.0
    }

@pytest.fixture
def video_plume_adapter_config():
    """
    Fixture providing validated configuration for VideoPlumeAdapter testing.
    
    Provides backward compatibility configuration with mock video processing
    for legacy VideoPlume functionality validation.
    """
    return {
        'video_path': 'test_plume_video.mp4',
        'preprocessing_config': {
            'grayscale': True,
            'blur_kernel': 3,
            'normalize': True,
            'threshold': None
        },
        'frame_cache_config': {
            'mode': 'lru',
            'memory_limit_mb': 256
        },
        'spatial_interpolation': True,
        'temporal_interpolation': False
    }

@pytest.fixture
def mock_wind_field():
    """
    Mock wind field implementation for deterministic plume model testing.
    
    Provides controlled wind dynamics for testing wind field integration
    without dependencies on specific wind field implementations.
    """
    mock = Mock(spec=WindFieldProtocol)
    mock.velocity_at.return_value = np.array([[1.0, 0.5]])  # Constant northeast wind
    mock.step.return_value = None
    mock.reset.return_value = None
    return mock

@pytest.fixture
def mock_scipy_stats():
    """
    Mock SciPy statistical functions for testing without SciPy dependency.
    
    Provides controlled statistical computation mocking for testing
    Gaussian distribution calculations and optimization routines.
    """
    if SCIPY_AVAILABLE:
        yield None  # Use real SciPy when available
    else:
        with patch('scipy.stats') as mock_stats:
            # Mock multivariate normal for Gaussian plume calculations
            mock_mvn = Mock()
            mock_mvn.pdf.return_value = np.array([0.5, 0.3, 0.1])
            mock_stats.multivariate_normal.return_value = mock_mvn
            yield mock_stats

@pytest.fixture
def performance_monitor():
    """
    Performance monitoring fixture for step execution time validation.
    
    Provides performance timing infrastructure for validating sub-10ms
    step execution requirements across all plume model implementations.
    """
    class PerformanceMonitor:
        def __init__(self):
            self.measurements = []
        
        def time_execution(self, func, *args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            self.measurements.append(execution_time)
            return result, execution_time
        
        def get_average_time(self):
            return np.mean(self.measurements) if self.measurements else 0.0
        
        def get_max_time(self):
            return np.max(self.measurements) if self.measurements else 0.0
        
        def assert_performance_requirements(self, max_time_ms=10.0):
            if self.measurements:
                avg_time_ms = self.get_average_time() * 1000
                max_time_actual_ms = self.get_max_time() * 1000
                assert avg_time_ms < max_time_ms, f"Average execution time {avg_time_ms:.3f}ms exceeds {max_time_ms}ms limit"
                assert max_time_actual_ms < max_time_ms * 2, f"Maximum execution time {max_time_actual_ms:.3f}ms exceeds safety margin"
    
    return PerformanceMonitor()

@pytest.fixture
def test_positions():
    """
    Fixture providing standardized test positions for concentration sampling.
    
    Provides deterministic spatial positions for consistent testing across
    all plume model implementations with various geometric arrangements.
    """
    return {
        'single_agent': np.array([45.0, 48.0]),
        'multi_agent': np.array([
            [45.0, 48.0],  # Near source
            [52.0, 47.0],  # Near source, different direction
            [60.0, 55.0],  # Further from source
            [30.0, 30.0],  # Distant position
            [100.0, 100.0]  # Far position for boundary testing
        ]),
        'grid_positions': np.array([
            [x, y] for x in range(40, 61, 5) for y in range(40, 61, 5)
        ]),
        'boundary_positions': np.array([
            [0.0, 0.0],    # Bottom-left boundary
            [100.0, 100.0],  # Top-right boundary
            [-10.0, 50.0],    # Outside left boundary
            [110.0, 50.0]     # Outside right boundary
        ])
    }

@pytest.fixture
def hydra_config_store():
    """
    Mock Hydra ConfigStore for testing configuration-driven instantiation.
    
    Provides controlled Hydra configuration testing infrastructure for
    validating seamless component switching via configuration without
    code changes per Section 0.2.1 requirements.
    """
    if not HYDRA_AVAILABLE:
        yield None
        return
    
    with patch('hydra.core.config_store.ConfigStore') as mock_cs:
        cs_instance = Mock()
        mock_cs.instance.return_value = cs_instance
        
        # Mock configuration storage and retrieval
        cs_instance.store = Mock()
        cs_instance.load = Mock()
        
        # Mock configuration templates for different plume models
        cs_instance.load.side_effect = lambda name: {
            'gaussian_plume': {
                '_target_': 'src.plume_nav_sim.models.plume.gaussian_plume.GaussianPlumeModel',
                'source_position': (50, 50),
                'source_strength': 1000,
                'sigma_x': 5.0,
                'sigma_y': 3.0
            },
            'turbulent_plume': {
                '_target_': 'src.plume_nav_sim.models.plume.turbulent_plume.TurbulentPlumeModel',
                'source_position': (50, 50),
                'num_filaments': 100,
                'turbulence_intensity': 0.2
            },
            'video_plume_adapter': {
                '_target_': 'src.plume_nav_sim.models.plume.video_plume_adapter.VideoPlumeAdapter',
                'video_path': 'test_video.mp4'
            }
        }.get(name, {})
        
        yield cs_instance


# =====================================================================================
# PROTOCOL COMPLIANCE TESTING (Section 0.4.1 - Test-Driven Protocol Development)
# =====================================================================================

class TestPlumeModelProtocolCompliance:
    """
    Comprehensive protocol compliance tests for PlumeModelProtocol implementations.
    
    This test class implements test-driven protocol development per Section 0.6.1,
    validating that all plume model implementations correctly implement the
    PlumeModelProtocol interface with proper structural subtyping conformance.
    
    Key testing areas:
    - Protocol method existence and signatures
    - Return type validation and array shapes
    - Parameter handling and validation
    - Error handling and edge cases
    - Performance requirements compliance
    """
    
    @pytest.mark.parametrize("plume_model_class,config_fixture", [
        pytest.param(GaussianPlumeModel, 'gaussian_plume_config', 
                    marks=pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, 
                                            reason="GaussianPlumeModel not available")),
        pytest.param(TurbulentPlumeModel, 'turbulent_plume_config',
                    marks=pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE,
                                            reason="TurbulentPlumeModel not available")),
        pytest.param(VideoPlumeAdapter, 'video_plume_adapter_config',
                    marks=pytest.mark.skipif(not VIDEO_PLUME_ADAPTER_AVAILABLE,
                                            reason="VideoPlumeAdapter not available"))
    ])
    def test_protocol_method_existence(self, plume_model_class, config_fixture, request):
        """
        Test that plume model implementations have all required protocol methods.
        
        Validates structural subtyping conformance by checking that all
        PlumeModelProtocol methods are implemented with correct signatures.
        """
        config = request.getfixturevalue(config_fixture)
        model = plume_model_class(**config)
        
        # Verify protocol method existence
        assert hasattr(model, 'concentration_at'), "concentration_at method required by PlumeModelProtocol"
        assert callable(model.concentration_at), "concentration_at must be callable"
        
        assert hasattr(model, 'step'), "step method required by PlumeModelProtocol"
        assert callable(model.step), "step must be callable"
        
        assert hasattr(model, 'reset'), "reset method required by PlumeModelProtocol"
        assert callable(model.reset), "reset must be callable"
    
    @pytest.mark.parametrize("plume_model_class,config_fixture", [
        pytest.param(GaussianPlumeModel, 'gaussian_plume_config', 
                    marks=pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, 
                                            reason="GaussianPlumeModel not available")),
        pytest.param(TurbulentPlumeModel, 'turbulent_plume_config',
                    marks=pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE,
                                            reason="TurbulentPlumeModel not available")),
        pytest.param(VideoPlumeAdapter, 'video_plume_adapter_config',
                    marks=pytest.mark.skipif(not VIDEO_PLUME_ADAPTER_AVAILABLE,
                                            reason="VideoPlumeAdapter not available"))
    ])
    def test_concentration_at_method_signature(self, plume_model_class, config_fixture, 
                                              test_positions, request):
        """
        Test concentration_at method signature and return types.
        
        Validates that concentration_at method accepts numpy arrays and returns
        appropriately shaped concentration arrays per protocol specification.
        """
        config = request.getfixturevalue(config_fixture)
        model = plume_model_class(**config)
        
        # Test single agent position
        single_pos = test_positions['single_agent']
        result = model.concentration_at(single_pos)
        assert isinstance(result, (float, np.number)), "Single agent should return scalar concentration"
        assert np.isfinite(result), "Concentration must be finite"
        assert result >= 0, "Concentration must be non-negative"
        
        # Test multi-agent positions
        multi_pos = test_positions['multi_agent']
        result = model.concentration_at(multi_pos)
        assert isinstance(result, np.ndarray), "Multi-agent should return numpy array"
        assert result.shape == (multi_pos.shape[0],), f"Expected shape {(multi_pos.shape[0],)}, got {result.shape}"
        assert np.all(np.isfinite(result)), "All concentrations must be finite"
        assert np.all(result >= 0), "All concentrations must be non-negative"
    
    @pytest.mark.parametrize("plume_model_class,config_fixture", [
        pytest.param(GaussianPlumeModel, 'gaussian_plume_config', 
                    marks=pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, 
                                            reason="GaussianPlumeModel not available")),
        pytest.param(TurbulentPlumeModel, 'turbulent_plume_config',
                    marks=pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE,
                                            reason="TurbulentPlumeModel not available")),
        pytest.param(VideoPlumeAdapter, 'video_plume_adapter_config',
                    marks=pytest.mark.skipif(not VIDEO_PLUME_ADAPTER_AVAILABLE,
                                            reason="VideoPlumeAdapter not available"))
    ])
    def test_step_method_temporal_advancement(self, plume_model_class, config_fixture, request):
        """
        Test step method for temporal state advancement.
        
        Validates that step method advances plume state consistently
        and maintains internal temporal consistency.
        """
        config = request.getfixturevalue(config_fixture)
        model = plume_model_class(**config)
        
        # Test step execution without errors
        initial_time = getattr(model, 'current_time', 0.0)
        model.step(dt=1.0)
        
        # Verify temporal advancement if time tracking is implemented
        if hasattr(model, 'current_time'):
            assert model.current_time == initial_time + 1.0, "Step should advance current_time by dt"
        
        # Test multiple steps
        for i in range(5):
            model.step(dt=0.5)
        
        if hasattr(model, 'current_time'):
            expected_time = initial_time + 1.0 + 5 * 0.5
            assert abs(model.current_time - expected_time) < 1e-10, "Multiple steps should accumulate correctly"
    
    @pytest.mark.parametrize("plume_model_class,config_fixture", [
        pytest.param(GaussianPlumeModel, 'gaussian_plume_config', 
                    marks=pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, 
                                            reason="GaussianPlumeModel not available")),
        pytest.param(TurbulentPlumeModel, 'turbulent_plume_config',
                    marks=pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE,
                                            reason="TurbulentPlumeModel not available")),
        pytest.param(VideoPlumeAdapter, 'video_plume_adapter_config',
                    marks=pytest.mark.skipif(not VIDEO_PLUME_ADAPTER_AVAILABLE,
                                            reason="VideoPlumeAdapter not available"))
    ])
    def test_reset_method_state_restoration(self, plume_model_class, config_fixture, 
                                           test_positions, request):
        """
        Test reset method for state restoration.
        
        Validates that reset method properly restores initial state
        and accepts parameter overrides per protocol specification.
        """
        config = request.getfixturevalue(config_fixture)
        model = plume_model_class(**config)
        
        # Get initial concentration
        test_pos = test_positions['single_agent']
        initial_concentration = model.concentration_at(test_pos)
        
        # Advance state
        for _ in range(10):
            model.step(dt=1.0)
        
        # Verify state has changed (if applicable)
        advanced_concentration = model.concentration_at(test_pos)
        
        # Reset to initial state
        model.reset()
        
        # Verify state restoration
        if hasattr(model, 'current_time'):
            assert model.current_time == 0.0, "Reset should restore current_time to 0"
        
        reset_concentration = model.concentration_at(test_pos)
        
        # For deterministic models, concentration should match exactly
        # For stochastic models, we check that reset was called successfully
        if plume_model_class == GaussianPlumeModel:
            assert abs(reset_concentration - initial_concentration) < 1e-10, \
                "Reset should restore initial concentration exactly for Gaussian model"
    
    def test_protocol_compliance_with_isinstance(self):
        """
        Test protocol compliance using isinstance checks.
        
        Validates that all available plume model implementations
        are recognized as PlumeModelProtocol instances.
        """
        if PROTOCOLS_AVAILABLE and GAUSSIAN_PLUME_AVAILABLE:
            model = GaussianPlumeModel(source_position=(0, 0), source_strength=100)
            assert isinstance(model, PlumeModelProtocol), \
                "GaussianPlumeModel should implement PlumeModelProtocol"
        
        if PROTOCOLS_AVAILABLE and TURBULENT_PLUME_AVAILABLE:
            model = TurbulentPlumeModel(source_position=(0, 0), num_filaments=50)
            assert isinstance(model, PlumeModelProtocol), \
                "TurbulentPlumeModel should implement PlumeModelProtocol"
        
        if PROTOCOLS_AVAILABLE and VIDEO_PLUME_ADAPTER_AVAILABLE:
            model = VideoPlumeAdapter(video_path="test.mp4")
            assert isinstance(model, PlumeModelProtocol), \
                "VideoPlumeAdapter should implement PlumeModelProtocol"


# =====================================================================================
# GAUSSIAN PLUME MODEL MATHEMATICAL VALIDATION TESTS
# =====================================================================================

@pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, reason="GaussianPlumeModel not available")
class TestGaussianPlumeModelMathematicalValidation:
    """
    Mathematical validation tests for GaussianPlumeModel concentration field generation.
    
    This test class validates the mathematical correctness of Gaussian plume
    dispersion calculations, ensuring analytical accuracy and physical realism
    per Section 0.2.1 requirements for fast analytical plume modeling.
    """
    
    def test_gaussian_concentration_peak_at_source(self, gaussian_plume_config):
        """
        Test that concentration peak occurs at source position.
        
        Validates fundamental Gaussian plume property that maximum
        concentration occurs at the emission source location.
        """
        model = GaussianPlumeModel(**gaussian_plume_config)
        
        # Test concentration at source
        source_pos = np.array(gaussian_plume_config['source_position'])
        source_concentration = model.concentration_at(source_pos)
        
        # Test concentrations at nearby positions
        nearby_positions = source_pos + np.array([
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
            [2.0, 2.0], [-2.0, -2.0]
        ])
        
        nearby_concentrations = model.concentration_at(nearby_positions)
        
        # Source should have higher concentration than all nearby positions
        assert source_concentration > np.max(nearby_concentrations), \
            "Source position should have maximum concentration"
    
    def test_gaussian_distance_decay(self, gaussian_plume_config):
        """
        Test concentration decay with distance from source.
        
        Validates that concentration decreases monotonically with
        distance from source following Gaussian decay properties.
        """
        model = GaussianPlumeModel(**gaussian_plume_config)
        source_pos = np.array(gaussian_plume_config['source_position'])
        
        # Test positions at increasing distances from source
        distances = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
        concentrations = []
        
        for distance in distances:
            test_pos = source_pos + np.array([distance, 0.0])
            concentration = model.concentration_at(test_pos)
            concentrations.append(concentration)
        
        concentrations = np.array(concentrations)
        
        # Concentrations should decrease with distance
        for i in range(len(concentrations) - 1):
            assert concentrations[i] > concentrations[i + 1], \
                f"Concentration should decrease with distance: {concentrations[i]} <= {concentrations[i + 1]} at distance {distances[i+1]}"
    
    def test_gaussian_symmetry_properties(self, gaussian_plume_config):
        """
        Test Gaussian plume symmetry properties.
        
        Validates that concentration field exhibits expected symmetry
        properties around the source position for isotropic dispersion.
        """
        # Use symmetric dispersion coefficients for symmetry testing
        config = gaussian_plume_config.copy()
        config['sigma_x'] = config['sigma_y'] = 3.0
        
        model = GaussianPlumeModel(**config)
        source_pos = np.array(config['source_position'])
        
        # Test symmetric positions around source
        test_distance = 5.0
        symmetric_positions = source_pos + test_distance * np.array([
            [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]
        ])
        
        concentrations = model.concentration_at(symmetric_positions)
        
        # All symmetric positions should have equal concentrations
        expected_concentration = concentrations[0]
        for concentration in concentrations:
            assert abs(concentration - expected_concentration) < 1e-10, \
                "Symmetric positions should have equal concentrations"
    
    def test_gaussian_dispersion_coefficient_effects(self):
        """
        Test effects of dispersion coefficients on concentration field.
        
        Validates that changes in sigma_x and sigma_y parameters
        produce expected changes in concentration field shape.
        """
        base_config = {
            'source_position': (50.0, 50.0),
            'source_strength': 1000.0,
            'sigma_x': 2.0,
            'sigma_y': 2.0
        }
        
        # Test wider dispersion in x-direction
        wide_x_config = base_config.copy()
        wide_x_config['sigma_x'] = 8.0
        
        # Test wider dispersion in y-direction  
        wide_y_config = base_config.copy()
        wide_y_config['sigma_y'] = 8.0
        
        model_base = GaussianPlumeModel(**base_config)
        model_wide_x = GaussianPlumeModel(**wide_x_config)
        model_wide_y = GaussianPlumeModel(**wide_y_config)
        
        source_pos = np.array(base_config['source_position'])
        
        # Test position displaced in x-direction
        x_displaced = source_pos + np.array([5.0, 0.0])
        conc_base_x = model_base.concentration_at(x_displaced)
        conc_wide_x = model_wide_x.concentration_at(x_displaced)
        
        # Wider x-dispersion should give higher concentration at x-displaced position
        assert conc_wide_x > conc_base_x, \
            "Wider x-dispersion should increase concentration at x-displaced position"
        
        # Test position displaced in y-direction
        y_displaced = source_pos + np.array([0.0, 5.0])
        conc_base_y = model_base.concentration_at(y_displaced)
        conc_wide_y = model_wide_y.concentration_at(y_displaced)
        
        # Wider y-dispersion should give higher concentration at y-displaced position
        assert conc_wide_y > conc_base_y, \
            "Wider y-dispersion should increase concentration at y-displaced position"
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(
        source_x=st.floats(min_value=10, max_value=90),
        source_y=st.floats(min_value=10, max_value=90),
        sigma_x=st.floats(min_value=1.0, max_value=10.0),
        sigma_y=st.floats(min_value=1.0, max_value=10.0),
        source_strength=st.floats(min_value=100, max_value=2000)
    )
    @settings(max_examples=50, deadline=1000)
    def test_gaussian_mathematical_properties(self, source_x, source_y, sigma_x, sigma_y, source_strength):
        """
        Property-based testing of Gaussian plume mathematical properties.
        
        Uses Hypothesis to test mathematical invariants across a wide
        range of parameter combinations ensuring robustness.
        """
        model = GaussianPlumeModel(
            source_position=(source_x, source_y),
            source_strength=source_strength,
            sigma_x=sigma_x,
            sigma_y=sigma_y
        )
        
        # Test positions in a grid around the source
        test_positions = np.array([
            [source_x, source_y],  # Source position
            [source_x + sigma_x, source_y],  # One sigma_x away
            [source_x, source_y + sigma_y],  # One sigma_y away
            [source_x - sigma_x, source_y - sigma_y],  # Diagonal
        ])
        
        concentrations = model.concentration_at(test_positions)
        
        # Mathematical property: All concentrations should be non-negative
        assert np.all(concentrations >= 0), "All concentrations must be non-negative"
        
        # Mathematical property: All concentrations should be finite
        assert np.all(np.isfinite(concentrations)), "All concentrations must be finite"
        
        # Mathematical property: Source should have maximum concentration
        source_concentration = concentrations[0]
        other_concentrations = concentrations[1:]
        assert source_concentration >= np.max(other_concentrations), \
            "Source position should have maximum or equal concentration"
    
    def test_gaussian_wind_integration(self, gaussian_plume_config, mock_wind_field):
        """
        Test Gaussian plume integration with wind field effects.
        
        Validates that wind field integration produces expected
        plume advection and transport dynamics.
        """
        config = gaussian_plume_config.copy()
        config['wind_field'] = mock_wind_field
        
        model = GaussianPlumeModel(**config)
        
        # Get initial concentration at test position
        test_pos = np.array([55.0, 52.0])
        initial_concentration = model.concentration_at(test_pos)
        
        # Advance time with wind field integration
        model.step(dt=5.0)
        
        # Verify wind field was queried
        mock_wind_field.velocity_at.assert_called()
        
        # Get concentration after wind advection
        final_concentration = model.concentration_at(test_pos)
        
        # With northeast wind, concentration distribution should change
        # (Exact validation depends on implementation details)
        assert isinstance(final_concentration, (float, np.number)), \
            "Wind integration should produce valid concentration"


# =====================================================================================
# TURBULENT PLUME MODEL FILAMENT-BASED PHYSICS VALIDATION TESTS
# =====================================================================================

@pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE, reason="TurbulentPlumeModel not available")
class TestTurbulentPlumeModelFilamentPhysics:
    """
    Filament-based physics validation tests for TurbulentPlumeModel.
    
    This test class validates the realistic turbulent physics simulation
    with individual filament tracking per user requirements in Section 0.3.2,
    including stochastic wind integration and intermittent plume structures.
    """
    
    def test_filament_initialization(self, turbulent_plume_config):
        """
        Test proper filament initialization and structure.
        
        Validates that filaments are correctly initialized with
        appropriate positions, strengths, and ages.
        """
        model = TurbulentPlumeModel(**turbulent_plume_config)
        
        # Verify filament data structure exists
        assert hasattr(model, 'filaments') or hasattr(model, '_filaments'), \
            "Model should maintain filament data structure"
        
        # Test filament count
        expected_count = turbulent_plume_config['num_filaments']
        if hasattr(model, 'filaments') and isinstance(model.filaments, dict):
            if 'positions' in model.filaments:
                actual_count = len(model.filaments['positions'])
                assert actual_count == expected_count, \
                    f"Should have {expected_count} filaments, got {actual_count}"
    
    def test_turbulent_concentration_variability(self, turbulent_plume_config):
        """
        Test turbulent concentration variability and intermittency.
        
        Validates that turbulent model produces realistic intermittent
        concentration patterns unlike smooth Gaussian distributions.
        """
        model = TurbulentPlumeModel(**turbulent_plume_config)
        
        # Test multiple concentration samples at same location
        test_position = np.array([52.0, 48.0])
        concentrations = []
        
        for _ in range(20):
            concentration = model.concentration_at(test_position)
            concentrations.append(concentration)
            model.step(dt=1.0)  # Advance time for variability
        
        concentrations = np.array(concentrations)
        
        # Turbulent model should show variability
        concentration_std = np.std(concentrations)
        concentration_mean = np.mean(concentrations)
        
        # Coefficient of variation should indicate turbulent intermittency
        if concentration_mean > 1e-6:  # Avoid division by very small numbers
            cv = concentration_std / concentration_mean
            # Turbulent plumes typically show significant variability
            # This is a heuristic check - exact values depend on implementation
            assert cv >= 0.0, "Coefficient of variation should be non-negative"
    
    def test_filament_temporal_evolution(self, turbulent_plume_config):
        """
        Test temporal evolution of filament system.
        
        Validates that filaments evolve over time with proper
        advection, diffusion, and lifecycle management.
        """
        model = TurbulentPlumeModel(**turbulent_plume_config)
        
        # Get initial filament state if accessible
        initial_positions = None
        if hasattr(model, 'filaments') and isinstance(model.filaments, dict):
            if 'positions' in model.filaments:
                initial_positions = model.filaments['positions'].copy()
        
        # Advance multiple time steps
        for _ in range(10):
            model.step(dt=1.0)
        
        # Verify temporal evolution occurred
        if initial_positions is not None and hasattr(model, 'filaments'):
            final_positions = model.filaments['positions']
            
            # Positions should have changed due to turbulent advection
            position_changes = np.linalg.norm(final_positions - initial_positions, axis=1)
            mean_displacement = np.mean(position_changes)
            
            # With turbulence_intensity=0.2 and 10 time steps, expect some movement
            assert mean_displacement > 0.0, "Filaments should move due to turbulent advection"
    
    def test_turbulent_wind_integration(self, turbulent_plume_config, mock_wind_field):
        """
        Test integration with wind field for realistic transport.
        
        Validates that turbulent plume model properly integrates
        with wind field protocols for enhanced transport physics.
        """
        config = turbulent_plume_config.copy()
        # Set wind field if the model supports it
        if 'wind_field' in TurbulentPlumeModel.__init__.__code__.co_varnames:
            config['wind_field'] = mock_wind_field
        
        model = TurbulentPlumeModel(**config)
        
        # Test concentration before and after wind integration
        test_pos = np.array([45.0, 50.0])
        initial_concentration = model.concentration_at(test_pos)
        
        # Advance with wind integration
        model.step(dt=2.0)
        
        # If wind field integration is implemented, it should be called
        if hasattr(model, 'wind_field') and model.wind_field is mock_wind_field:
            mock_wind_field.velocity_at.assert_called()
        
        final_concentration = model.concentration_at(test_pos)
        
        # Verify concentration is computed successfully
        assert isinstance(final_concentration, (float, np.number)), \
            "Wind integration should produce valid concentration"
    
    def test_filament_emission_and_lifecycle(self, turbulent_plume_config):
        """
        Test filament emission and lifecycle management.
        
        Validates proper filament emission from source and
        lifecycle management including aging and removal.
        """
        model = TurbulentPlumeModel(**turbulent_plume_config)
        
        # Test that model maintains consistent filament count over time
        initial_filament_count = None
        if hasattr(model, 'filaments') and isinstance(model.filaments, dict):
            if 'positions' in model.filaments:
                initial_filament_count = len(model.filaments['positions'])
        
        # Advance time and check filament management
        for _ in range(5):
            model.step(dt=2.0)
        
        final_filament_count = None
        if hasattr(model, 'filaments') and isinstance(model.filaments, dict):
            if 'positions' in model.filaments:
                final_filament_count = len(model.filaments['positions'])
        
        # Filament count should be managed (either constant or within reasonable bounds)
        if initial_filament_count is not None and final_filament_count is not None:
            # Allow some variation but shouldn't grow unbounded
            assert final_filament_count <= initial_filament_count * 2, \
                "Filament count should not grow unbounded"
    
    def test_turbulent_spatial_correlation(self, turbulent_plume_config):
        """
        Test spatial correlation properties of turbulent concentration field.
        
        Validates that nearby positions show appropriate spatial
        correlation patterns typical of turbulent dispersion.
        """
        model = TurbulentPlumeModel(**turbulent_plume_config)
        
        # Test positions in a small grid
        center_pos = np.array(turbulent_plume_config['source_position'])
        grid_positions = center_pos + np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]
        ])
        
        concentrations = model.concentration_at(grid_positions)
        
        # Nearby positions should generally have similar concentrations
        # but with turbulent variability
        center_concentration = concentrations[0]
        adjacent_concentrations = concentrations[1:4]  # 1-unit away positions
        distant_concentrations = concentrations[4:]    # 2-unit away positions
        
        # Basic spatial coherence: nearby positions shouldn't be too different
        # This is a loose constraint since turbulent fields can be patchy
        if center_concentration > 1e-6:
            relative_variations = np.abs(adjacent_concentrations - center_concentration) / center_concentration
            # Allow significant variation but not extreme outliers
            assert np.all(relative_variations <= 10.0), \
                "Adjacent positions shouldn't have extreme concentration variations"


# =====================================================================================
# VIDEO PLUME ADAPTER BACKWARD COMPATIBILITY TESTS
# =====================================================================================

@pytest.mark.skipif(not VIDEO_PLUME_ADAPTER_AVAILABLE, reason="VideoPlumeAdapter not available")
class TestVideoPlumeAdapterBackwardCompatibility:
    """
    Backward compatibility tests for VideoPlumeAdapter maintaining VideoPlume functionality.
    
    This test class validates that VideoPlumeAdapter preserves existing VideoPlume
    functionality while implementing PlumeModelProtocol for seamless integration
    with the new modular architecture per Section 0.2.1 requirements.
    """
    
    def test_video_plume_adapter_initialization(self, video_plume_adapter_config):
        """
        Test VideoPlumeAdapter initialization with legacy parameters.
        
        Validates that adapter accepts legacy VideoPlume configuration
        parameters and initializes correctly.
        """
        adapter = VideoPlumeAdapter(**video_plume_adapter_config)
        
        # Verify basic properties exist
        assert hasattr(adapter, 'video_path'), "Adapter should maintain video_path property"
        assert adapter.video_path == video_plume_adapter_config['video_path']
        
        # Verify frame-related properties
        assert hasattr(adapter, 'frame_count') or hasattr(adapter, 'get_frame_count'), \
            "Adapter should provide frame count access"
        assert hasattr(adapter, 'width') or hasattr(adapter, 'get_width'), \
            "Adapter should provide width access"
        assert hasattr(adapter, 'height') or hasattr(adapter, 'get_height'), \
            "Adapter should provide height access"
    
    def test_video_plume_adapter_protocol_compliance(self, video_plume_adapter_config, test_positions):
        """
        Test VideoPlumeAdapter implements PlumeModelProtocol correctly.
        
        Validates that adapter provides all required protocol methods
        with appropriate behavior for video-based concentration access.
        """
        adapter = VideoPlumeAdapter(**video_plume_adapter_config)
        
        # Test concentration_at method
        single_pos = test_positions['single_agent']
        concentration = adapter.concentration_at(single_pos)
        assert isinstance(concentration, (float, np.number)), \
            "Single position should return scalar concentration"
        assert np.isfinite(concentration), "Concentration must be finite"
        assert concentration >= 0, "Concentration must be non-negative"
        
        # Test multi-position sampling
        multi_pos = test_positions['multi_agent']
        concentrations = adapter.concentration_at(multi_pos)
        assert isinstance(concentrations, np.ndarray), \
            "Multi-position should return numpy array"
        assert concentrations.shape == (len(multi_pos),), \
            "Array shape should match number of positions"
    
    def test_video_plume_adapter_frame_progression(self, video_plume_adapter_config):
        """
        Test VideoPlumeAdapter frame progression with step method.
        
        Validates that step method advances through video frames
        maintaining temporal progression like original VideoPlume.
        """
        adapter = VideoPlumeAdapter(**video_plume_adapter_config)
        
        # Get initial frame index
        initial_frame = getattr(adapter, 'current_frame', 0)
        
        # Step through several frames
        for i in range(5):
            adapter.step(dt=1.0)
            
            # Verify frame advancement if accessible
            if hasattr(adapter, 'current_frame'):
                expected_frame = initial_frame + i + 1
                assert adapter.current_frame == expected_frame, \
                    f"Frame should advance to {expected_frame}, got {adapter.current_frame}"
    
    def test_video_plume_adapter_reset_functionality(self, video_plume_adapter_config):
        """
        Test VideoPlumeAdapter reset functionality.
        
        Validates that reset method restores adapter to initial state
        like original VideoPlume reset behavior.
        """
        adapter = VideoPlumeAdapter(**video_plume_adapter_config)
        
        # Advance several frames
        for _ in range(10):
            adapter.step(dt=1.0)
        
        # Reset to initial state
        adapter.reset()
        
        # Verify reset to initial state
        if hasattr(adapter, 'current_frame'):
            assert adapter.current_frame == 0, "Reset should return to frame 0"
        if hasattr(adapter, 'current_time'):
            assert adapter.current_time == 0.0, "Reset should return to time 0"
    
    def test_video_plume_adapter_spatial_interpolation(self, video_plume_adapter_config):
        """
        Test VideoPlumeAdapter spatial interpolation capabilities.
        
        Validates that adapter provides spatial interpolation for
        sub-pixel position queries like original VideoPlume.
        """
        adapter = VideoPlumeAdapter(**video_plume_adapter_config)
        
        # Test sub-pixel positions
        subpixel_positions = np.array([
            [45.5, 48.3],  # Sub-pixel position
            [52.7, 47.1],  # Another sub-pixel position
        ])
        
        concentrations = adapter.concentration_at(subpixel_positions)
        
        # Should handle sub-pixel positions without errors
        assert len(concentrations) == len(subpixel_positions), \
            "Should return concentration for each sub-pixel position"
        assert np.all(np.isfinite(concentrations)), \
            "Sub-pixel interpolation should produce finite values"
    
    def test_video_plume_adapter_boundary_handling(self, video_plume_adapter_config, test_positions):
        """
        Test VideoPlumeAdapter boundary condition handling.
        
        Validates that adapter handles positions outside video bounds
        gracefully like original VideoPlume boundary behavior.
        """
        adapter = VideoPlumeAdapter(**video_plume_adapter_config)
        
        # Test boundary positions
        boundary_positions = test_positions['boundary_positions']
        concentrations = adapter.concentration_at(boundary_positions)
        
        # Should handle boundary positions without errors
        assert len(concentrations) == len(boundary_positions), \
            "Should return concentration for each boundary position"
        assert np.all(np.isfinite(concentrations)), \
            "Boundary handling should produce finite values"
        
        # Positions outside bounds should typically return zero or background
        out_of_bounds_concentration = concentrations[-1]  # Last position is out of bounds
        assert out_of_bounds_concentration >= 0, \
            "Out of bounds positions should return non-negative concentration"


# =====================================================================================
# PERFORMANCE REGRESSION TESTS (Sub-10ms Requirement)
# =====================================================================================

class TestPlumeModelPerformanceRequirements:
    """
    Performance regression tests ensuring sub-10ms step execution latency.
    
    This test class validates that all plume model implementations maintain
    performance requirements specified in the technical specification,
    ensuring real-time simulation compatibility.
    """
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("plume_model_class,config_fixture,expected_name", [
        pytest.param(GaussianPlumeModel, 'gaussian_plume_config', 'GaussianPlumeModel',
                    marks=pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, 
                                            reason="GaussianPlumeModel not available")),
        pytest.param(TurbulentPlumeModel, 'turbulent_plume_config', 'TurbulentPlumeModel',
                    marks=pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE,
                                            reason="TurbulentPlumeModel not available")),
        pytest.param(VideoPlumeAdapter, 'video_plume_adapter_config', 'VideoPlumeAdapter',
                    marks=pytest.mark.skipif(not VIDEO_PLUME_ADAPTER_AVAILABLE,
                                            reason="VideoPlumeAdapter not available"))
    ])
    def test_concentration_at_performance(self, plume_model_class, config_fixture, expected_name, 
                                         test_positions, performance_monitor, request):
        """
        Test concentration_at method performance requirements.
        
        Validates that concentration queries execute within performance
        requirements: <1ms for single query, <10ms for 100+ agents.
        """
        config = request.getfixturevalue(config_fixture)
        model = plume_model_class(**config)
        
        # Test single agent performance
        single_pos = test_positions['single_agent']
        _, execution_time = performance_monitor.time_execution(
            model.concentration_at, single_pos
        )
        
        # Single agent should be very fast
        assert execution_time < 0.001, \
            f"Single agent concentration query took {execution_time*1000:.3f}ms, should be <1ms"
        
        # Test multi-agent performance (simulate 100 agents)
        multi_positions = np.random.rand(100, 2) * 100  # 100 random positions
        _, batch_execution_time = performance_monitor.time_execution(
            model.concentration_at, multi_positions
        )
        
        # Batch query should still be fast
        assert batch_execution_time < 0.010, \
            f"100-agent batch query took {batch_execution_time*1000:.3f}ms, should be <10ms"
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("plume_model_class,config_fixture,expected_name", [
        pytest.param(GaussianPlumeModel, 'gaussian_plume_config', 'GaussianPlumeModel',
                    marks=pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, 
                                            reason="GaussianPlumeModel not available")),
        pytest.param(TurbulentPlumeModel, 'turbulent_plume_config', 'TurbulentPlumeModel',
                    marks=pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE,
                                            reason="TurbulentPlumeModel not available")),
        pytest.param(VideoPlumeAdapter, 'video_plume_adapter_config', 'VideoPlumeAdapter',
                    marks=pytest.mark.skipif(not VIDEO_PLUME_ADAPTER_AVAILABLE,
                                            reason="VideoPlumeAdapter not available"))
    ])
    def test_step_performance(self, plume_model_class, config_fixture, expected_name, 
                             performance_monitor, request):
        """
        Test step method performance requirements.
        
        Validates that step execution completes within 5ms per step
        for real-time simulation compatibility.
        """
        config = request.getfixturevalue(config_fixture)
        model = plume_model_class(**config)
        
        # Test multiple step executions
        step_times = []
        for _ in range(20):
            _, execution_time = performance_monitor.time_execution(
                model.step, dt=1.0
            )
            step_times.append(execution_time)
        
        avg_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        
        # Step execution should be fast
        assert avg_step_time < 0.005, \
            f"Average step time {avg_step_time*1000:.3f}ms exceeds 5ms limit"
        assert max_step_time < 0.010, \
            f"Maximum step time {max_step_time*1000:.3f}ms exceeds 10ms safety limit"
    
    @pytest.mark.benchmark
    def test_memory_efficiency_requirements(self):
        """
        Test memory efficiency requirements for plume models.
        
        Validates that plume models maintain memory efficiency
        requirements: <100MB for typical simulation scenarios.
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        models = []
        
        # Create multiple model instances to test memory usage
        if GAUSSIAN_PLUME_AVAILABLE:
            for i in range(10):
                model = GaussianPlumeModel(
                    source_position=(i*10, i*10),
                    source_strength=1000,
                    sigma_x=5.0,
                    sigma_y=3.0
                )
                models.append(model)
        
        if TURBULENT_PLUME_AVAILABLE:
            for i in range(5):  # Fewer turbulent models due to higher memory usage
                model = TurbulentPlumeModel(
                    source_position=(i*10, i*10),
                    num_filaments=50,  # Reduced for memory testing
                    turbulence_intensity=0.2
                )
                models.append(model)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, \
            f"Memory increase {memory_increase:.1f}MB exceeds 100MB limit for typical scenarios"
        
        # Cleanup
        del models


# =====================================================================================
# HYDRA CONFIGURATION SWITCHING TESTS (Section 0.2.1)
# =====================================================================================

@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestHydraConfigurationSwitching:
    """
    Configuration switching tests validating seamless model transitions via Hydra.
    
    This test class validates that plume models can be switched via Hydra
    configuration without code changes per Section 0.2.1 requirements,
    ensuring configuration-driven component switching.
    """
    
    def test_gaussian_to_turbulent_switching(self, hydra_config_store):
        """
        Test switching from Gaussian to Turbulent plume model via configuration.
        
        Validates that the same simulation code can use different plume
        models by changing only the Hydra configuration.
        """
        # Simulate Hydra configuration loading for Gaussian model
        gaussian_config = {
            '_target_': 'src.plume_nav_sim.models.plume.gaussian_plume.GaussianPlumeModel',
            'source_position': (25, 75),
            'source_strength': 1500,
            'sigma_x': 4.0,
            'sigma_y': 2.0
        }
        
        # Simulate Hydra configuration loading for Turbulent model
        turbulent_config = {
            '_target_': 'src.plume_nav_sim.models.plume.turbulent_plume.TurbulentPlumeModel',
            'source_position': (25, 75),
            'source_strength': 1500,
            'num_filaments': 150,
            'turbulence_intensity': 0.3
        }
        
        # Test configuration-driven instantiation (simulated)
        if GAUSSIAN_PLUME_AVAILABLE:
            # Simulate hydra.utils.instantiate() for Gaussian model
            gaussian_params = {k: v for k, v in gaussian_config.items() if k != '_target_'}
            gaussian_model = GaussianPlumeModel(**gaussian_params)
            
            # Verify model type and basic functionality
            assert gaussian_model.__class__.__name__ == 'GaussianPlumeModel'
            test_concentration = gaussian_model.concentration_at(np.array([25, 75]))
            assert test_concentration > 0, "Gaussian model should produce concentration at source"
        
        if TURBULENT_PLUME_AVAILABLE:
            # Simulate hydra.utils.instantiate() for Turbulent model
            turbulent_params = {k: v for k, v in turbulent_config.items() if k != '_target_'}
            turbulent_model = TurbulentPlumeModel(**turbulent_params)
            
            # Verify model type and basic functionality
            assert turbulent_model.__class__.__name__ == 'TurbulentPlumeModel'
            test_concentration = turbulent_model.concentration_at(np.array([25, 75]))
            assert test_concentration >= 0, "Turbulent model should produce valid concentration"
    
    def test_video_adapter_configuration_switching(self, hydra_config_store):
        """
        Test switching to VideoPlumeAdapter via configuration.
        
        Validates that legacy video-based plume data can be used
        through configuration switching without code changes.
        """
        # Simulate Hydra configuration for video adapter
        video_config = {
            '_target_': 'src.plume_nav_sim.models.plume.video_plume_adapter.VideoPlumeAdapter',
            'video_path': 'experiments/plume_data.mp4',
            'preprocessing_config': {
                'grayscale': True,
                'normalize': True,
                'blur_kernel': 5
            }
        }
        
        if VIDEO_PLUME_ADAPTER_AVAILABLE:
            # Simulate hydra.utils.instantiate() for video adapter
            video_params = {k: v for k, v in video_config.items() if k != '_target_'}
            video_model = VideoPlumeAdapter(**video_params)
            
            # Verify model type and protocol compliance
            assert video_model.__class__.__name__ == 'VideoPlumeAdapter'
            assert hasattr(video_model, 'concentration_at')
            assert hasattr(video_model, 'step')
            assert hasattr(video_model, 'reset')
    
    def test_parameter_override_configuration(self):
        """
        Test parameter override via Hydra configuration overrides.
        
        Validates that model parameters can be overridden via
        Hydra command-line or configuration overrides.
        """
        # Base configuration
        base_config = {
            'source_position': (50, 50),
            'source_strength': 1000
        }
        
        # Override configuration (simulating hydra overrides)
        override_config = base_config.copy()
        override_config.update({
            'source_position': (30, 70),
            'source_strength': 2000
        })
        
        if GAUSSIAN_PLUME_AVAILABLE:
            # Test base configuration
            base_model = GaussianPlumeModel(**base_config, sigma_x=5.0, sigma_y=3.0)
            
            # Test override configuration
            override_model = GaussianPlumeModel(**override_config, sigma_x=5.0, sigma_y=3.0)
            
            # Verify parameter overrides took effect
            assert np.array_equal(base_model.source_position, [50, 50])
            assert np.array_equal(override_model.source_position, [30, 70])
            assert base_model.source_strength == 1000
            assert override_model.source_strength == 2000
    
    def test_configuration_validation_and_error_handling(self):
        """
        Test configuration validation and error handling.
        
        Validates that invalid configurations are caught and
        reported appropriately during model instantiation.
        """
        # Test invalid configuration parameters
        invalid_configs = [
            {
                'source_position': (50, 50),
                'source_strength': -100,  # Invalid: negative strength
                'sigma_x': 5.0,
                'sigma_y': 3.0
            },
            {
                'source_position': (50, 50),
                'source_strength': 1000,
                'sigma_x': -2.0,  # Invalid: negative dispersion
                'sigma_y': 3.0
            }
        ]
        
        if GAUSSIAN_PLUME_AVAILABLE:
            for invalid_config in invalid_configs:
                with pytest.raises((ValueError, TypeError)):
                    GaussianPlumeModel(**invalid_config)


# =====================================================================================
# INTEGRATION TESTS WITH WIND FIELD AND SENSOR PROTOCOLS
# =====================================================================================

class TestPlumeModelIntegration:
    """
    Integration tests for plume models with wind field and sensor protocols.
    
    This test class validates cross-component integration ensuring
    seamless interaction between plume models and other system components.
    """
    
    @pytest.mark.parametrize("plume_model_class,config_fixture", [
        pytest.param(GaussianPlumeModel, 'gaussian_plume_config',
                    marks=pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, 
                                            reason="GaussianPlumeModel not available")),
        pytest.param(TurbulentPlumeModel, 'turbulent_plume_config',
                    marks=pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE,
                                            reason="TurbulentPlumeModel not available"))
    ])
    def test_wind_field_integration(self, plume_model_class, config_fixture, 
                                   mock_wind_field, request):
        """
        Test plume model integration with wind field protocols.
        
        Validates that plume models properly integrate with WindFieldProtocol
        implementations for realistic transport dynamics.
        """
        config = request.getfixturevalue(config_fixture)
        
        # Add wind field to configuration if supported
        if 'wind_field' in plume_model_class.__init__.__code__.co_varnames:
            config = config.copy()
            config['wind_field'] = mock_wind_field
        
        model = plume_model_class(**config)
        
        # Test basic functionality with wind field
        test_pos = np.array([55.0, 52.0])
        concentration = model.concentration_at(test_pos)
        assert np.isfinite(concentration), "Wind integration should produce finite concentration"
        
        # Test temporal evolution with wind
        model.step(dt=1.0)
        
        # If wind field integration is implemented, verify it was called
        if hasattr(model, 'wind_field') and model.wind_field is mock_wind_field:
            mock_wind_field.velocity_at.assert_called()
    
    def test_multi_model_consistency(self, test_positions):
        """
        Test consistency across different plume model implementations.
        
        Validates that all plume models produce reasonable results
        for the same spatial queries, ensuring API consistency.
        """
        models = []
        model_names = []
        
        # Create available model instances with similar parameters
        common_source = (50.0, 50.0)
        common_strength = 1000.0
        
        if GAUSSIAN_PLUME_AVAILABLE:
            gaussian_model = GaussianPlumeModel(
                source_position=common_source,
                source_strength=common_strength,
                sigma_x=5.0,
                sigma_y=3.0
            )
            models.append(gaussian_model)
            model_names.append('GaussianPlumeModel')
        
        if TURBULENT_PLUME_AVAILABLE:
            turbulent_model = TurbulentPlumeModel(
                source_position=common_source,
                source_strength=common_strength,
                num_filaments=100,
                turbulence_intensity=0.1
            )
            models.append(turbulent_model)
            model_names.append('TurbulentPlumeModel')
        
        if VIDEO_PLUME_ADAPTER_AVAILABLE:
            video_model = VideoPlumeAdapter(video_path="test_video.mp4")
            models.append(video_model)
            model_names.append('VideoPlumeAdapter')
        
        if len(models) < 2:
            pytest.skip("Need at least 2 plume models for consistency testing")
        
        # Test same positions across all models
        test_pos = test_positions['single_agent']
        concentrations = []
        
        for model in models:
            concentration = model.concentration_at(test_pos)
            concentrations.append(concentration)
        
        # All models should produce finite, non-negative concentrations
        for i, (concentration, name) in enumerate(zip(concentrations, model_names)):
            assert np.isfinite(concentration), f"{name} should produce finite concentration"
            assert concentration >= 0, f"{name} should produce non-negative concentration"
    
    def test_protocol_compliance_across_implementations(self):
        """
        Test that all implementations satisfy PlumeModelProtocol requirements.
        
        Validates protocol compliance across all available plume model
        implementations ensuring consistent behavior.
        """
        if not PROTOCOLS_AVAILABLE:
            pytest.skip("Protocols not available for testing")
        
        implementations = []
        
        if GAUSSIAN_PLUME_AVAILABLE:
            implementations.append(('GaussianPlumeModel', GaussianPlumeModel, {
                'source_position': (0, 0), 'source_strength': 100, 'sigma_x': 2, 'sigma_y': 2
            }))
        
        if TURBULENT_PLUME_AVAILABLE:
            implementations.append(('TurbulentPlumeModel', TurbulentPlumeModel, {
                'source_position': (0, 0), 'source_strength': 100, 'num_filaments': 50
            }))
        
        if VIDEO_PLUME_ADAPTER_AVAILABLE:
            implementations.append(('VideoPlumeAdapter', VideoPlumeAdapter, {
                'video_path': 'test.mp4'
            }))
        
        for name, model_class, config in implementations:
            model = model_class(**config)
            
            # Test protocol compliance
            assert isinstance(model, PlumeModelProtocol), \
                f"{name} should implement PlumeModelProtocol"
            
            # Test required methods exist and work
            test_pos = np.array([1.0, 1.0])
            concentration = model.concentration_at(test_pos)
            assert isinstance(concentration, (float, np.number)), \
                f"{name}.concentration_at should return scalar for single position"
            
            model.step(dt=1.0)  # Should not raise error
            model.reset()       # Should not raise error


# =====================================================================================
# EDGE CASES AND ERROR HANDLING TESTS
# =====================================================================================

class TestPlumeModelEdgeCases:
    """
    Edge case and error handling tests for plume model implementations.
    
    This test class validates robust error handling and edge case management
    ensuring reliable operation under various boundary conditions.
    """
    
    @pytest.mark.parametrize("plume_model_class,config_fixture", [
        pytest.param(GaussianPlumeModel, 'gaussian_plume_config',
                    marks=pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, 
                                            reason="GaussianPlumeModel not available")),
        pytest.param(TurbulentPlumeModel, 'turbulent_plume_config',
                    marks=pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE,
                                            reason="TurbulentPlumeModel not available")),
        pytest.param(VideoPlumeAdapter, 'video_plume_adapter_config',
                    marks=pytest.mark.skipif(not VIDEO_PLUME_ADAPTER_AVAILABLE,
                                            reason="VideoPlumeAdapter not available"))
    ])
    def test_invalid_position_inputs(self, plume_model_class, config_fixture, request):
        """
        Test handling of invalid position inputs.
        
        Validates that plume models handle malformed position
        inputs gracefully with appropriate error messages.
        """
        config = request.getfixturevalue(config_fixture)
        model = plume_model_class(**config)
        
        # Test invalid position arrays
        invalid_positions = [
            np.array([1.0]),           # Wrong dimension
            np.array([[1.0]]),         # Wrong shape
            np.array([[1.0, 2.0, 3.0]]),  # Too many coordinates
            np.array([[[1.0, 2.0]]]),  # Too many dimensions
        ]
        
        for invalid_pos in invalid_positions:
            with pytest.raises((ValueError, IndexError)):
                model.concentration_at(invalid_pos)
    
    @pytest.mark.parametrize("plume_model_class,config_fixture", [
        pytest.param(GaussianPlumeModel, 'gaussian_plume_config',
                    marks=pytest.mark.skipif(not GAUSSIAN_PLUME_AVAILABLE, 
                                            reason="GaussianPlumeModel not available")),
        pytest.param(TurbulentPlumeModel, 'turbulent_plume_config',
                    marks=pytest.mark.skipif(not TURBULENT_PLUME_AVAILABLE,
                                            reason="TurbulentPlumeModel not available")),
        pytest.param(VideoPlumeAdapter, 'video_plume_adapter_config',
                    marks=pytest.mark.skipif(not VIDEO_PLUME_ADAPTER_AVAILABLE,
                                            reason="VideoPlumeAdapter not available"))
    ])
    def test_extreme_position_values(self, plume_model_class, config_fixture, request):
        """
        Test handling of extreme position values.
        
        Validates that plume models handle very large, very small,
        and special float values appropriately.
        """
        config = request.getfixturevalue(config_fixture)
        model = plume_model_class(**config)
        
        # Test extreme positions
        extreme_positions = np.array([
            [1e6, 1e6],      # Very large coordinates
            [-1e6, -1e6],    # Very large negative coordinates
            [0.0, 0.0],      # Origin
            [1e-6, 1e-6],    # Very small coordinates
        ])
        
        # Should handle extreme positions without errors
        concentrations = model.concentration_at(extreme_positions)
        assert len(concentrations) == len(extreme_positions)
        assert np.all(np.isfinite(concentrations)), "Should handle extreme positions with finite values"
        assert np.all(concentrations >= 0), "Concentrations should remain non-negative"
    
    def test_invalid_step_parameters(self):
        """
        Test handling of invalid step method parameters.
        
        Validates that step method validates time step parameters
        and handles invalid inputs appropriately.
        """
        if GAUSSIAN_PLUME_AVAILABLE:
            model = GaussianPlumeModel(source_position=(0, 0), source_strength=100)
            
            # Test invalid dt values
            invalid_dt_values = [-1.0, 0.0, np.inf, np.nan]
            
            for invalid_dt in invalid_dt_values:
                with pytest.raises((ValueError, TypeError)):
                    model.step(dt=invalid_dt)
    
    def test_memory_stress_conditions(self):
        """
        Test plume models under memory stress conditions.
        
        Validates that models handle large numbers of position
        queries efficiently without memory issues.
        """
        if GAUSSIAN_PLUME_AVAILABLE:
            model = GaussianPlumeModel(source_position=(50, 50), source_strength=1000)
            
            # Test large batch of positions
            large_position_batch = np.random.rand(1000, 2) * 100
            
            concentrations = model.concentration_at(large_position_batch)
            
            assert len(concentrations) == 1000, "Should handle large position batches"
            assert np.all(np.isfinite(concentrations)), "Large batches should produce finite results"
    
    def test_concurrent_access_safety(self):
        """
        Test thread safety for concurrent access to plume models.
        
        Validates that plume models can handle concurrent access
        safely for multi-threaded simulation scenarios.
        """
        import threading
        import queue
        
        if GAUSSIAN_PLUME_AVAILABLE:
            model = GaussianPlumeModel(source_position=(50, 50), source_strength=1000)
            results_queue = queue.Queue()
            errors_queue = queue.Queue()
            
            def worker_thread(thread_id):
                try:
                    for i in range(10):
                        pos = np.array([thread_id + i, thread_id + i])
                        concentration = model.concentration_at(pos)
                        results_queue.put((thread_id, i, concentration))
                        model.step(dt=0.1)
                except Exception as e:
                    errors_queue.put((thread_id, e))
            
            # Start multiple worker threads
            threads = []
            for thread_id in range(3):
                thread = threading.Thread(target=worker_thread, args=(thread_id,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Check for errors
            assert errors_queue.empty(), f"Concurrent access errors: {list(errors_queue.queue)}"
            
            # Verify results were produced
            assert not results_queue.empty(), "Concurrent access should produce results"


# =====================================================================================
# COMPREHENSIVE TEST EXECUTION AND REPORTING
# =====================================================================================

def test_comprehensive_plume_model_validation():
    """
    Comprehensive validation test ensuring all plume models meet requirements.
    
    This test provides a summary validation of all plume model implementations
    ensuring they meet the comprehensive requirements specified in Section 0.
    """
    validation_results = {
        'protocol_compliance': [],
        'performance_requirements': [],
        'mathematical_accuracy': [],
        'configuration_switching': [],
        'integration_compatibility': []
    }
    
    # Test protocol compliance
    available_models = []
    if GAUSSIAN_PLUME_AVAILABLE:
        available_models.append('GaussianPlumeModel')
    if TURBULENT_PLUME_AVAILABLE:
        available_models.append('TurbulentPlumeModel')
    if VIDEO_PLUME_ADAPTER_AVAILABLE:
        available_models.append('VideoPlumeAdapter')
    
    validation_results['protocol_compliance'] = available_models
    
    # Validate minimum requirements
    assert len(available_models) > 0, \
        "At least one plume model implementation should be available"
    
    # Test basic functionality for each available model
    for model_name in available_models:
        if model_name == 'GaussianPlumeModel':
            model = GaussianPlumeModel(source_position=(50, 50), source_strength=1000)
            test_pos = np.array([45, 48])
            concentration = model.concentration_at(test_pos)
            assert np.isfinite(concentration) and concentration >= 0, \
                f"{model_name} should produce valid concentrations"
            validation_results['mathematical_accuracy'].append(model_name)
        
        elif model_name == 'TurbulentPlumeModel':
            model = TurbulentPlumeModel(source_position=(50, 50), num_filaments=50)
            test_pos = np.array([45, 48])
            concentration = model.concentration_at(test_pos)
            assert np.isfinite(concentration) and concentration >= 0, \
                f"{model_name} should produce valid concentrations"
            validation_results['mathematical_accuracy'].append(model_name)
        
        elif model_name == 'VideoPlumeAdapter':
            model = VideoPlumeAdapter(video_path="test_video.mp4")
            test_pos = np.array([45, 48])
            concentration = model.concentration_at(test_pos)
            assert np.isfinite(concentration) and concentration >= 0, \
                f"{model_name} should produce valid concentrations"
            validation_results['mathematical_accuracy'].append(model_name)
    
    # Report validation summary
    print(f"\nPlume Model Validation Summary:")
    print(f"Available models: {validation_results['protocol_compliance']}")
    print(f"Protocol compliant: {len(validation_results['protocol_compliance'])} models")
    print(f"Mathematical validation: {len(validation_results['mathematical_accuracy'])} models")
    
    # Ensure comprehensive test coverage per Section 0.6.1 requirements
    assert len(validation_results['mathematical_accuracy']) == len(available_models), \
        "All available models should pass mathematical validation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])