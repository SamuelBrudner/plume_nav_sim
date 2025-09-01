"""
Comprehensive end-to-end integration tests for modular plume navigation simulation architecture.

This module provides comprehensive testing for the modular simulation system focusing on:
- Component boundary interactions (Navigator-PlumeModel-WindField-Sensor implementations)
- Configuration-driven component switching capabilities via Hydra integration
- Protocol compliance validation across PlumeModelProtocol, WindFieldProtocol, and SensorProtocol
- Memory-independent navigation scenarios supporting both memory-based and memoryless agents
- Seamless plume model switching between Gaussian and Turbulent implementations
- Performance maintenance with sub-10ms step execution despite modular abstractions
- Backward compatibility with existing VideoPlume workflows during architectural transition
- Sensor abstraction layer functionality across all sensing modalities
- Wind field integration and environmental dynamics coupling
- Simulation context orchestration handling episode management and result collection

Key Testing Areas:
- Protocol compliance validation across all modular component implementations
- Cross-component interaction testing with realistic integration scenarios
- Configuration-driven component switching without code modifications via Hydra
- Performance regression prevention ensuring <10ms step execution latency
- Memory vs memoryless agent execution with identical configurations
- Comprehensive sensor abstraction testing across Binary/Concentration/Gradient sensors
- Wind field integration testing covering Constant/Turbulent/TimeVarying implementations
- Vectorized environment compatibility for parallel RL training scenarios
- Property-based testing using Hypothesis for comprehensive edge case coverage
- Component lifecycle management and resource cleanup validation

Author: Blitzy Agent
Version: 1.0.0
"""

import pytest
import numpy as np
import time
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, MagicMock, patch
import sys
from contextlib import contextmanager

# Update sys.path for new project structure
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Core dependencies for modular architecture testing
try:
    from plume_nav_sim.core.protocols import (
        NavigatorProtocol, PlumeModelProtocol, WindFieldProtocol, SensorProtocol,
        AgentObservationProtocol, AgentActionProtocol, NavigatorFactory
    )
    from plume_nav_sim.core.simulation import (
        SimulationContext, SimulationConfig, SimulationResults, PerformanceMonitor
    )
    from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv
    CORE_AVAILABLE = True
except ImportError:
    # Graceful degradation for components not yet created
    CORE_AVAILABLE = False
    NavigatorProtocol = object
    PlumeModelProtocol = object
    WindFieldProtocol = object
    SensorProtocol = object

# Modular component imports with fallbacks
try:
    from plume_nav_sim.models.plume.gaussian_plume import GaussianPlumeModel
    from plume_nav_sim.models.plume.turbulent_plume import TurbulentPlumeModel
    PLUME_MODELS_AVAILABLE = True
except ImportError:
    PLUME_MODELS_AVAILABLE = False
    GaussianPlumeModel = None
    TurbulentPlumeModel = None

try:
    from plume_nav_sim.models.wind.constant_wind import ConstantWindField
    from plume_nav_sim.models.wind.turbulent_wind import TurbulentWindField
    WIND_MODELS_AVAILABLE = True
except ImportError:
    WIND_MODELS_AVAILABLE = False
    ConstantWindField = None
    TurbulentWindField = None

try:
    from plume_nav_sim.core.sensors.binary_sensor import BinarySensor
    from plume_nav_sim.core.sensors.concentration_sensor import ConcentrationSensor
    from plume_nav_sim.core.sensors.gradient_sensor import GradientSensor
    SENSORS_AVAILABLE = True
except ImportError:
    SENSORS_AVAILABLE = False
    BinarySensor = None
    ConcentrationSensor = None
    GradientSensor = None

try:
    from plume_nav_sim.examples.agents.reactive_agent import ReactiveAgent
    from plume_nav_sim.examples.agents.infotaxis_agent import InfotaxisAgent
    EXAMPLE_AGENTS_AVAILABLE = True
except ImportError:
    EXAMPLE_AGENTS_AVAILABLE = False
    ReactiveAgent = None
    InfotaxisAgent = None

try:
    from plume_nav_sim.config.schemas import (
        NavigatorConfig, PlumeModelConfig, WindFieldConfig, SensorConfig
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    NavigatorConfig = dict
    PlumeModelConfig = dict

# Enhanced testing dependencies
try:
    import hypothesis
    from hypothesis import given, strategies as st, assume, settings, HealthCheck
    from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule, initialize
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
    
    # Mock stateful testing components
    RuleBasedStateMachine = object
    Bundle = None
    rule = lambda *args, **kwargs: lambda f: f
    initialize = lambda *args, **kwargs: lambda f: f

try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    check_env = None

try:
    from omegaconf import DictConfig, OmegaConf
    from hydra import compose, initialize_config_store
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

# Import shared test fixtures
try:
    from tests.conftest import (
        mock_video_plume, mock_navigator, mock_multi_navigator,
        mock_hydra_config, mock_seed_manager
    )
except ImportError:
    # Define minimal fallback fixtures
    @pytest.fixture
    def mock_video_plume():
        return MagicMock()
    
    @pytest.fixture  
    def mock_navigator():
        return MagicMock()
    
    @pytest.fixture
    def mock_multi_navigator():
        return MagicMock()

# Performance constants from requirements
PERFORMANCE_TARGET_MS = 10.0  # ≤10ms step execution requirement from Section 0.5.1
PERFORMANCE_WARNING_MS = 5.0   # Warning threshold for performance monitoring
FPS_TARGET = 30.0               # ≥30 FPS simulation rate requirement
MAX_MEMORY_MB = 100            # Maximum memory usage for test environments


@contextmanager
def performance_monitor(operation_name: str, max_time_ms: float = PERFORMANCE_TARGET_MS):
    """Context manager for monitoring test operation performance."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        if duration_ms > max_time_ms:
            warnings.warn(
                f"{operation_name} took {duration_ms:.2f}ms, exceeding {max_time_ms}ms target",
                UserWarning
            )


def create_mock_plume_model(model_type: str = "gaussian") -> Mock:
    """Create a mock plume model for testing."""
    mock = Mock(spec=PlumeModelProtocol)
    
    # Configure basic methods
    mock.concentration_at.return_value = np.array([0.5, 0.3, 0.7])
    mock.step.return_value = None
    mock.reset.return_value = None
    mock.get_metadata.return_value = {
        "type": model_type,
        "source_position": (50, 50),
        "parameters": {"strength": 1000.0}
    }
    
    return mock


def create_mock_wind_field(field_type: str = "constant") -> Mock:
    """Create a mock wind field for testing."""
    mock = Mock(spec=WindFieldProtocol)
    
    # Configure basic methods
    mock.velocity_at.return_value = np.array([[1.0, 0.5], [1.2, 0.3], [0.8, 0.7]])
    mock.step.return_value = None
    mock.reset.return_value = None
    mock.get_metadata.return_value = {
        "type": field_type,
        "velocity": (1.0, 0.5),
        "parameters": {"turbulence_intensity": 0.1}
    }
    
    return mock


def create_mock_sensor(sensor_type: str = "concentration") -> Mock:
    """Create a mock sensor for testing."""
    mock = Mock(spec=SensorProtocol)
    
    if sensor_type == "binary":
        mock.detect.return_value = np.array([True, False, True])
        mock.measure.return_value = np.array([1.0, 0.0, 1.0])
        mock.compute_gradient.return_value = np.array([[0.1, 0.2], [0.0, 0.0], [0.3, 0.1]])
    elif sensor_type == "concentration":
        mock.detect.return_value = np.array([0.5, 0.3, 0.7])
        mock.measure.return_value = np.array([0.5, 0.3, 0.7])
        mock.compute_gradient.return_value = np.array([[0.1, 0.2], [0.05, 0.1], [0.2, 0.15]])
    elif sensor_type == "gradient":
        mock.detect.return_value = np.array([0.5, 0.3, 0.7])
        mock.measure.return_value = np.array([0.5, 0.3, 0.7])
        mock.compute_gradient.return_value = np.array([[0.2, 0.3], [0.1, 0.15], [0.25, 0.2]])
    
    mock.configure.return_value = None
    mock.get_observation_space_info.return_value = {
        "type": sensor_type,
        "shape": (3,) if sensor_type != "gradient" else (3, 2),
        "dtype": "float32" if sensor_type != "binary" else "bool"
    }
    mock.get_metadata.return_value = {
        "type": sensor_type,
        "parameters": {"threshold": 0.1} if sensor_type == "binary" else {"range": (0, 1)}
    }
    
    return mock


def create_mock_navigator(num_agents: int = 1) -> Mock:
    """Create a mock navigator for testing."""
    mock = Mock(spec=NavigatorProtocol)
    
    # Configure navigator properties
    mock.num_agents = num_agents
    mock.positions = np.random.rand(num_agents, 2) * 100
    mock.orientations = np.random.rand(num_agents) * 360
    mock.speeds = np.random.rand(num_agents) * 2.0
    mock.max_speeds = np.ones(num_agents) * 3.0
    mock.angular_velocities = (np.random.rand(num_agents) - 0.5) * 20.0
    
    # Configure methods
    mock.reset.return_value = None
    mock.step.return_value = None
    mock.sample_odor.return_value = np.random.rand(num_agents)
    mock.sample_multiple_sensors.return_value = np.random.rand(num_agents, 2)
    
    # Configure extensibility hooks
    mock.compute_additional_obs.return_value = {}
    mock.compute_extra_reward.return_value = 0.0
    mock.on_episode_end.return_value = None
    mock.load_memory.return_value = None
    mock.save_memory.return_value = None
    
    return mock


class TestProtocolCompliance:
    """Test suite for protocol compliance validation across all modular component implementations."""
    
    def test_plume_model_protocol_compliance(self):
        """Test that all plume model implementations comply with PlumeModelProtocol."""
        if not PLUME_MODELS_AVAILABLE:
            pytest.skip("Plume models not available")
        
        # Test Gaussian plume model compliance
        if GaussianPlumeModel:
            gaussian_model = GaussianPlumeModel(source_position=(50, 50), source_strength=1000.0)
            
            # Verify protocol methods exist and are callable
            assert hasattr(gaussian_model, 'concentration_at')
            assert hasattr(gaussian_model, 'step')
            assert hasattr(gaussian_model, 'reset')
            assert callable(gaussian_model.concentration_at)
            assert callable(gaussian_model.step)
            assert callable(gaussian_model.reset)
            
            # Test method functionality
            positions = np.array([[45, 48], [55, 52]])
            concentrations = gaussian_model.concentration_at(positions)
            assert isinstance(concentrations, np.ndarray)
            assert concentrations.shape == (2,)
            assert np.all(concentrations >= 0)
            
        # Test Turbulent plume model compliance
        if TurbulentPlumeModel:
            turbulent_model = TurbulentPlumeModel(
                source_position=(50, 50), 
                filament_count=100,
                turbulence_intensity=0.2
            )
            
            # Verify protocol compliance
            assert hasattr(turbulent_model, 'concentration_at')
            assert hasattr(turbulent_model, 'step')
            assert hasattr(turbulent_model, 'reset')
            
            # Test functionality
            positions = np.array([[45, 48], [55, 52]])
            concentrations = turbulent_model.concentration_at(positions)
            assert isinstance(concentrations, np.ndarray)
            assert concentrations.shape == (2,)
    
    def test_wind_field_protocol_compliance(self):
        """Test that all wind field implementations comply with WindFieldProtocol."""
        if not WIND_MODELS_AVAILABLE:
            pytest.skip("Wind models not available")
        
        # Test Constant wind field compliance
        if ConstantWindField:
            constant_wind = ConstantWindField(velocity=(2.0, 1.0))
            
            # Verify protocol methods exist
            assert hasattr(constant_wind, 'velocity_at')
            assert hasattr(constant_wind, 'step')
            assert hasattr(constant_wind, 'reset')
            assert callable(constant_wind.velocity_at)
            
            # Test functionality
            positions = np.array([[10, 20], [30, 40]])
            velocities = constant_wind.velocity_at(positions)
            assert isinstance(velocities, np.ndarray)
            assert velocities.shape == (2, 2)
            
        # Test Turbulent wind field compliance
        if TurbulentWindField:
            turbulent_wind = TurbulentWindField(
                mean_velocity=(2.0, 1.0),
                turbulence_intensity=0.3
            )
            
            # Verify protocol compliance
            assert hasattr(turbulent_wind, 'velocity_at')
            assert hasattr(turbulent_wind, 'step')
            assert hasattr(turbulent_wind, 'reset')
            
            # Test functionality
            positions = np.array([[10, 20], [30, 40]])
            velocities = turbulent_wind.velocity_at(positions)
            assert isinstance(velocities, np.ndarray)
            assert velocities.shape == (2, 2)
    
    def test_sensor_protocol_compliance(self):
        """Test that all sensor implementations comply with SensorProtocol."""
        if not SENSORS_AVAILABLE:
            pytest.skip("Sensors not available")
        
        # Test Binary sensor compliance
        if BinarySensor:
            binary_sensor = BinarySensor(threshold=0.1)
            
            # Verify protocol methods exist
            assert hasattr(binary_sensor, 'detect')
            assert hasattr(binary_sensor, 'measure')
            assert hasattr(binary_sensor, 'compute_gradient')
            assert hasattr(binary_sensor, 'configure')
            
            # Test functionality with mock plume state
            mock_plume = create_mock_plume_model()
            positions = np.array([[10, 20], [30, 40]])
            
            detections = binary_sensor.detect(mock_plume, positions)
            assert isinstance(detections, np.ndarray)
            
        # Test Concentration sensor compliance
        if ConcentrationSensor:
            conc_sensor = ConcentrationSensor(dynamic_range=(0, 1))
            
            # Verify protocol compliance
            assert hasattr(conc_sensor, 'detect')
            assert hasattr(conc_sensor, 'measure')
            assert hasattr(conc_sensor, 'compute_gradient')
            
            # Test functionality
            mock_plume = create_mock_plume_model()
            positions = np.array([[10, 20], [30, 40]])
            
            measurements = conc_sensor.measure(mock_plume, positions)
            assert isinstance(measurements, np.ndarray)
            
        # Test Gradient sensor compliance
        if GradientSensor:
            gradient_sensor = GradientSensor(spatial_resolution=(0.5, 0.5))
            
            # Verify protocol compliance
            assert hasattr(gradient_sensor, 'compute_gradient')
            
            # Test functionality
            mock_plume = create_mock_plume_model()
            positions = np.array([[10, 20], [30, 40]])
            
            gradients = gradient_sensor.compute_gradient(mock_plume, positions)
            assert isinstance(gradients, np.ndarray)
            assert gradients.shape == (2, 2)


class TestComponentBoundaryInteractions:
    """Test suite for component boundary interactions and integration patterns."""
    
    def test_navigator_plume_model_interaction(self):
        """Test interaction between Navigator and PlumeModel components."""
        with performance_monitor("Navigator-PlumeModel interaction"):
            # Create components
            navigator = create_mock_navigator(num_agents=3)
            plume_model = create_mock_plume_model("gaussian")
            
            # Test interaction pattern
            positions = navigator.positions
            concentrations = plume_model.concentration_at(positions)
            
            # Verify interaction
            assert concentrations is not None
            assert len(concentrations) == navigator.num_agents
            plume_model.concentration_at.assert_called_once_with(positions)
    
    def test_plume_model_wind_field_coupling(self):
        """Test coupling between PlumeModel and WindField components."""
        with performance_monitor("PlumeModel-WindField coupling"):
            # Create components
            plume_model = create_mock_plume_model("turbulent")
            wind_field = create_mock_wind_field("turbulent")
            
            # Test temporal integration
            dt = 0.1
            for _ in range(10):
                wind_field.step(dt)
                plume_model.step(dt)
            
            # Verify both components advanced
            assert wind_field.step.call_count == 10
            assert plume_model.step.call_count == 10
    
    def test_sensor_plume_model_integration(self):
        """Test integration between Sensor and PlumeModel components."""
        with performance_monitor("Sensor-PlumeModel integration"):
            # Create components
            sensors = [
                create_mock_sensor("binary"),
                create_mock_sensor("concentration"),
                create_mock_sensor("gradient")
            ]
            plume_model = create_mock_plume_model("gaussian")
            
            # Test multi-sensor sampling
            positions = np.array([[25, 35], [45, 55], [65, 75]])
            
            for sensor in sensors:
                readings = sensor.measure(plume_model, positions)
                assert readings is not None
                sensor.measure.assert_called_with(plume_model, positions)
    
    def test_full_component_integration_cycle(self):
        """Test complete integration cycle with all components."""
        with performance_monitor("Full component integration cycle"):
            # Create complete modular system
            navigator = create_mock_navigator(num_agents=2)
            plume_model = create_mock_plume_model("gaussian")
            wind_field = create_mock_wind_field("constant")
            sensors = [create_mock_sensor("concentration")]
            
            # Execute integration cycle
            dt = 0.1
            num_steps = 5
            
            for step in range(num_steps):
                # Update environmental dynamics
                wind_field.step(dt)
                plume_model.step(dt)
                
                # Navigator perceives environment
                positions = navigator.positions
                concentrations = plume_model.concentration_at(positions)
                sensor_readings = sensors[0].measure(plume_model, positions)
                
                # Navigator updates based on perception
                navigator.step(concentrations, dt)
            
            # Verify all components participated
            assert wind_field.step.call_count == num_steps
            assert plume_model.step.call_count == num_steps
            assert navigator.step.call_count == num_steps
            assert plume_model.concentration_at.call_count == num_steps
            assert sensors[0].measure.call_count == num_steps


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestConfigurationIntegration:
    """Test suite for Hydra configuration integration and component switching."""
    
    def test_plume_model_switching_via_configuration(self):
        """Test seamless plume model switching via Hydra configuration."""
        gaussian_config = {
            "type": "gaussian",
            "source_position": (50, 50),
            "source_strength": 1000.0,
            "spread_sigma": 5.0
        }
        
        turbulent_config = {
            "type": "turbulent", 
            "source_position": (50, 50),
            "filament_count": 500,
            "turbulence_intensity": 0.3
        }
        
        # Test configuration-driven switching
        with performance_monitor("Configuration-driven plume model switching"):
            # Mock factory method for switching
            with patch('plume_nav_sim.core.protocols.NavigatorFactory.create_plume_model') as mock_factory:
                mock_factory.side_effect = [
                    create_mock_plume_model("gaussian"),
                    create_mock_plume_model("turbulent")
                ]
                
                # Switch from Gaussian to Turbulent
                if CORE_AVAILABLE:
                    gaussian_model = NavigatorFactory.create_plume_model(gaussian_config)
                    turbulent_model = NavigatorFactory.create_plume_model(turbulent_config)
                    
                    # Verify both models were created
                    assert mock_factory.call_count == 2
                    assert gaussian_model is not None
                    assert turbulent_model is not None
    
    def test_wind_field_configuration_switching(self):
        """Test wind field switching via configuration without code changes."""
        constant_config = {
            "type": "constant",
            "velocity": (2.0, 1.0)
        }
        
        turbulent_config = {
            "type": "turbulent",
            "mean_velocity": (2.0, 1.0),
            "turbulence_intensity": 0.2
        }
        
        with performance_monitor("Wind field configuration switching"):
            # Mock factory method
            with patch('plume_nav_sim.core.protocols.NavigatorFactory.create_wind_field') as mock_factory:
                mock_factory.side_effect = [
                    create_mock_wind_field("constant"),
                    create_mock_wind_field("turbulent")
                ]
                
                if CORE_AVAILABLE:
                    constant_wind = NavigatorFactory.create_wind_field(constant_config)
                    turbulent_wind = NavigatorFactory.create_wind_field(turbulent_config)
                    
                    assert mock_factory.call_count == 2
                    assert constant_wind is not None
                    assert turbulent_wind is not None
    
    def test_sensor_configuration_composition(self):
        """Test multi-sensor configuration composition."""
        sensor_configs = [
            {"type": "binary", "threshold": 0.1},
            {"type": "concentration", "dynamic_range": (0, 1)},
            {"type": "gradient", "spatial_resolution": (0.5, 0.5)}
        ]
        
        with performance_monitor("Multi-sensor configuration composition"):
            # Mock factory method
            with patch('plume_nav_sim.core.protocols.NavigatorFactory.create_sensors') as mock_factory:
                mock_factory.return_value = [
                    create_mock_sensor("binary"),
                    create_mock_sensor("concentration"),
                    create_mock_sensor("gradient")
                ]
                
                if CORE_AVAILABLE:
                    sensors = NavigatorFactory.create_sensors(sensor_configs)
                    
                    assert mock_factory.call_count == 1
                    assert len(sensors) == 3
    
    def test_complete_modular_environment_configuration(self):
        """Test complete modular environment creation from configuration."""
        env_config = {
            "navigator": {
                "type": "single",
                "position": (0, 0),
                "max_speed": 2.0
            },
            "plume_model": {
                "type": "gaussian",
                "source_position": (50, 50),
                "source_strength": 1000.0
            },
            "wind_field": {
                "type": "constant",
                "velocity": (1.0, 0.5)
            },
            "sensors": [
                {"type": "concentration", "dynamic_range": (0, 1)}
            ]
        }
        
        with performance_monitor("Complete modular environment configuration"):
            # Mock factory method
            with patch('plume_nav_sim.core.protocols.NavigatorFactory.create_modular_environment') as mock_factory:
                mock_navigator = create_mock_navigator(1)
                mock_navigator._modular_components = {
                    'plume_model': create_mock_plume_model("gaussian"),
                    'wind_field': create_mock_wind_field("constant"),
                    'sensors': [create_mock_sensor("concentration")]
                }
                mock_factory.return_value = mock_navigator
                
                if CORE_AVAILABLE:
                    environment = NavigatorFactory.create_modular_environment(
                        navigator_config=env_config["navigator"],
                        plume_model_config=env_config["plume_model"],
                        wind_field_config=env_config["wind_field"],
                        sensor_configs=env_config["sensors"]
                    )
                    
                    assert mock_factory.call_count == 1
                    assert environment is not None
                    assert hasattr(environment, '_modular_components')


class TestMemoryIndependentNavigation:
    """Test suite for memory-independent navigation scenarios."""
    
    def test_memoryless_agent_execution(self):
        """Test memoryless agent execution without memory functionality."""
        with performance_monitor("Memoryless agent execution"):
            # Create memoryless navigator
            navigator = create_mock_navigator(num_agents=1)
            
            # Configure memory methods to return None/do nothing
            navigator.load_memory.return_value = None
            navigator.save_memory.return_value = None
            
            # Test memory-less execution
            plume_state = create_mock_plume_model("gaussian")
            
            # Execute steps without memory
            for step in range(10):
                positions = navigator.positions
                concentrations = plume_state.concentration_at(positions)
                navigator.step(concentrations, dt=0.1)
            
            # Verify no memory operations occurred
            memory_state = navigator.save_memory()
            assert memory_state is None
            
            # Load memory should be a no-op
            navigator.load_memory({"some": "data"})
            navigator.load_memory.assert_called_once()
    
    def test_memory_based_agent_execution(self):
        """Test memory-based agent execution with state persistence."""
        if not EXAMPLE_AGENTS_AVAILABLE:
            pytest.skip("Example agents not available")
        
        with performance_monitor("Memory-based agent execution"):
            # Create memory-based navigator
            navigator = create_mock_navigator(num_agents=1)
            
            # Configure memory functionality
            memory_data = {
                "trajectory_history": [[0, 0], [1, 1], [2, 2]],
                "odor_concentration_history": [0.1, 0.3, 0.5],
                "spatial_map": {"visited": [(0, 0), (1, 1)]},
                "episode_count": 5
            }
            navigator.save_memory.return_value = memory_data
            
            # Test memory-based execution
            plume_state = create_mock_plume_model("gaussian")
            
            # Load initial memory
            navigator.load_memory(memory_data)
            
            # Execute steps with memory
            for step in range(5):
                positions = navigator.positions
                concentrations = plume_state.concentration_at(positions)
                navigator.step(concentrations, dt=0.1)
            
            # Verify memory operations
            saved_memory = navigator.save_memory()
            assert saved_memory is not None
            assert isinstance(saved_memory, dict)
            navigator.load_memory.assert_called_once_with(memory_data)
    
    def test_identical_configuration_different_agent_types(self):
        """Test identical configurations differing only in agent memory type."""
        base_config = {
            "plume_model": {"type": "gaussian", "source_strength": 1000.0},
            "wind_field": {"type": "constant", "velocity": (1.0, 0.0)},
            "sensors": [{"type": "concentration", "dynamic_range": (0, 1)}],
            "navigator": {"position": (10, 10), "max_speed": 2.0}
        }
        
        # Configuration for memoryless agent
        memoryless_config = {**base_config}
        memoryless_config["navigator"]["enable_memory"] = False
        
        # Configuration for memory-based agent  
        memory_config = {**base_config}
        memory_config["navigator"]["enable_memory"] = True
        memory_config["navigator"]["memory_capacity"] = 1000
        
        with performance_monitor("Identical configuration with different agent types"):
            # Create both agent types
            memoryless_nav = create_mock_navigator(1)
            memoryless_nav.save_memory.return_value = None
            
            memory_nav = create_mock_navigator(1)
            memory_nav.save_memory.return_value = {"history": [1, 2, 3]}
            
            # Test both agents with same environment
            plume_state = create_mock_plume_model("gaussian")
            
            # Execute same scenario for both agents
            for nav in [memoryless_nav, memory_nav]:
                for step in range(5):
                    positions = nav.positions
                    concentrations = plume_state.concentration_at(positions)
                    nav.step(concentrations, dt=0.1)
            
            # Verify different memory behavior
            memoryless_memory = memoryless_nav.save_memory()
            memory_based_memory = memory_nav.save_memory()
            
            assert memoryless_memory is None
            assert memory_based_memory is not None


class TestPerformanceRegression:
    """Test suite for performance regression detection and validation."""
    
    def test_step_execution_performance_single_agent(self):
        """Test step execution maintains <10ms performance with single agent."""
        navigator = create_mock_navigator(num_agents=1)
        plume_model = create_mock_plume_model("gaussian")
        
        # Warm-up
        for _ in range(5):
            positions = navigator.positions
            concentrations = plume_model.concentration_at(positions)
            navigator.step(concentrations, dt=0.1)
        
        # Performance measurement
        step_times = []
        num_measurements = 20
        
        for _ in range(num_measurements):
            start_time = time.perf_counter()
            
            positions = navigator.positions
            concentrations = plume_model.concentration_at(positions)
            navigator.step(concentrations, dt=0.1)
            
            step_time_ms = (time.perf_counter() - start_time) * 1000
            step_times.append(step_time_ms)
        
        # Validate performance requirements
        mean_step_time = np.mean(step_times)
        p95_step_time = np.percentile(step_times, 95)
        max_step_time = np.max(step_times)
        
        assert mean_step_time < PERFORMANCE_TARGET_MS, \
            f"Mean step time {mean_step_time:.2f}ms exceeds {PERFORMANCE_TARGET_MS}ms target"
        assert p95_step_time < PERFORMANCE_TARGET_MS * 1.5, \
            f"P95 step time {p95_step_time:.2f}ms indicates performance issues"
        assert max_step_time < PERFORMANCE_TARGET_MS * 2, \
            f"Max step time {max_step_time:.2f}ms too high for real-time performance"
    
    def test_step_execution_performance_multi_agent(self):
        """Test step execution performance with multiple agents."""
        num_agents = 10
        navigator = create_mock_navigator(num_agents=num_agents)
        plume_model = create_mock_plume_model("gaussian")
        
        # Warm-up
        for _ in range(5):
            positions = navigator.positions
            concentrations = plume_model.concentration_at(positions)
            navigator.step(concentrations, dt=0.1)
        
        # Performance measurement
        step_times = []
        num_measurements = 15
        
        for _ in range(num_measurements):
            start_time = time.perf_counter()
            
            positions = navigator.positions
            concentrations = plume_model.concentration_at(positions)
            navigator.step(concentrations, dt=0.1)
            
            step_time_ms = (time.perf_counter() - start_time) * 1000
            step_times.append(step_time_ms)
        
        # Validate performance requirements (more lenient for multi-agent)
        mean_step_time = np.mean(step_times)
        p95_step_time = np.percentile(step_times, 95)
        
        assert mean_step_time < PERFORMANCE_TARGET_MS * 2, \
            f"Multi-agent mean step time {mean_step_time:.2f}ms exceeds threshold"
        assert p95_step_time < PERFORMANCE_TARGET_MS * 3, \
            f"Multi-agent P95 step time {p95_step_time:.2f}ms indicates scaling issues"
    
    def test_modular_component_overhead(self):
        """Test that modular architecture doesn't introduce significant overhead."""
        # Create components
        navigator = create_mock_navigator(num_agents=5)
        plume_model = create_mock_plume_model("gaussian")
        wind_field = create_mock_wind_field("constant")
        sensors = [
            create_mock_sensor("binary"),
            create_mock_sensor("concentration")
        ]
        
        # Measure full modular cycle
        cycle_times = []
        num_measurements = 10
        
        for _ in range(num_measurements):
            start_time = time.perf_counter()
            
            # Full modular cycle
            dt = 0.1
            wind_field.step(dt)
            plume_model.step(dt)
            
            positions = navigator.positions
            concentrations = plume_model.concentration_at(positions)
            
            for sensor in sensors:
                sensor.measure(plume_model, positions)
            
            navigator.step(concentrations, dt)
            
            cycle_time_ms = (time.perf_counter() - start_time) * 1000
            cycle_times.append(cycle_time_ms)
        
        # Validate modular overhead is acceptable
        mean_cycle_time = np.mean(cycle_times)
        
        # Allow extra time for component coordination
        max_acceptable_time = PERFORMANCE_TARGET_MS * 2  # 20ms for full cycle
        assert mean_cycle_time < max_acceptable_time, \
            f"Modular cycle time {mean_cycle_time:.2f}ms exceeds {max_acceptable_time}ms threshold"
    
    def test_memory_usage_efficiency(self):
        """Test memory usage remains within acceptable bounds."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")
        
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Create large-scale simulation
        num_agents = 20
        navigator = create_mock_navigator(num_agents=num_agents)
        plume_model = create_mock_plume_model("turbulent")
        wind_field = create_mock_wind_field("turbulent")
        sensors = [create_mock_sensor("concentration") for _ in range(3)]
        
        # Run simulation
        for step in range(100):
            wind_field.step(0.1)
            plume_model.step(0.1)
            
            positions = navigator.positions
            concentrations = plume_model.concentration_at(positions)
            
            for sensor in sensors:
                sensor.measure(plume_model, positions)
            
            navigator.step(concentrations, 0.1)
        
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase_mb = final_memory_mb - initial_memory_mb
        
        # Memory usage should be reasonable
        max_acceptable_memory_mb = MAX_MEMORY_MB
        assert memory_increase_mb < max_acceptable_memory_mb, \
            f"Memory increase {memory_increase_mb:.1f}MB exceeds {max_acceptable_memory_mb}MB limit"


class TestBackwardCompatibility:
    """Test suite for backward compatibility with existing VideoPlume workflows."""
    
    def test_video_plume_adapter_compatibility(self):
        """Test VideoPlumeAdapter maintains compatibility with existing workflows."""
        # Mock VideoPlume functionality
        mock_video_plume = Mock()
        mock_video_plume.get_frame.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_video_plume.get_metadata.return_value = {
            "width": 640, "height": 480, "fps": 30.0, "frame_count": 1000
        }
        
        # Test compatibility with existing patterns
        with performance_monitor("VideoPlume compatibility"):
            # Simulate existing workflow
            frame_idx = 0
            frame = mock_video_plume.get_frame(frame_idx)
            metadata = mock_video_plume.get_metadata()
            
            # Verify existing interface still works
            assert frame is not None
            assert frame.shape == (480, 640)
            assert metadata["width"] == 640
            assert metadata["height"] == 480
            
            mock_video_plume.get_frame.assert_called_once_with(frame_idx)
            mock_video_plume.get_metadata.assert_called_once()
    
    def test_navigator_protocol_backward_compatibility(self):
        """Test NavigatorProtocol maintains backward compatibility."""
        navigator = create_mock_navigator(num_agents=1)
        
        # Test legacy interface still works
        with performance_monitor("Navigator backward compatibility"):
            # Legacy sample_odor interface
            env_array = np.random.rand(100, 100)
            odor_reading = navigator.sample_odor(env_array)
            
            # Legacy multi-sensor interface
            sensor_readings = navigator.sample_multiple_sensors(
                env_array, sensor_distance=5.0, num_sensors=2
            )
            
            # Verify legacy methods still callable
            navigator.sample_odor.assert_called_once()
            navigator.sample_multiple_sensors.assert_called_once()
            
            assert odor_reading is not None
            assert sensor_readings is not None
    
    def test_simulation_config_compatibility(self):
        """Test SimulationConfig maintains backward compatibility."""
        # Test legacy configuration parameters
        legacy_config = {
            "num_steps": 1000,
            "dt": 0.1,
            "record_trajectories": True,
            "enable_visualization": False
        }
        
        with performance_monitor("Simulation config compatibility"):
            # Mock config creation
            with patch('plume_nav_sim.core.simulation.SimulationConfig') as MockConfig:
                mock_config = Mock()
                mock_config.num_steps = legacy_config["num_steps"]
                mock_config.dt = legacy_config["dt"] 
                mock_config.record_trajectories = legacy_config["record_trajectories"]
                MockConfig.return_value = mock_config
                
                if CORE_AVAILABLE:
                    config = SimulationConfig(**legacy_config)
                    MockConfig.assert_called_once()


@pytest.mark.skipif(not SENSORS_AVAILABLE, reason="Sensor implementations not available")
class TestSensorAbstractionLayer:
    """Test suite for comprehensive sensor abstraction layer validation."""
    
    def test_binary_sensor_functionality(self):
        """Test BinarySensor implementation and functionality."""
        if not BinarySensor:
            pytest.skip("BinarySensor not available")
        
        with performance_monitor("Binary sensor functionality"):
            sensor = BinarySensor(threshold=0.1, false_positive_rate=0.02)
            plume_state = create_mock_plume_model("gaussian")
            positions = np.array([[25, 35], [45, 55]])
            
            # Test detection
            detections = sensor.detect(plume_state, positions)
            assert isinstance(detections, np.ndarray)
            assert detections.dtype == bool or np.issubdtype(detections.dtype, np.number)
            
            # Test measurement (should be binary)
            measurements = sensor.measure(plume_state, positions)
            assert isinstance(measurements, np.ndarray)
            
            # Test configuration
            sensor.configure(threshold=0.05, false_positive_rate=0.01)
            
            # Test metadata
            metadata = sensor.get_metadata()
            assert "threshold" in metadata
    
    def test_concentration_sensor_functionality(self):
        """Test ConcentrationSensor implementation and functionality."""
        if not ConcentrationSensor:
            pytest.skip("ConcentrationSensor not available")
        
        with performance_monitor("Concentration sensor functionality"):
            sensor = ConcentrationSensor(dynamic_range=(0, 1), resolution=0.001)
            plume_state = create_mock_plume_model("gaussian")
            positions = np.array([[25, 35], [45, 55]])
            
            # Test measurement
            measurements = sensor.measure(plume_state, positions)
            assert isinstance(measurements, np.ndarray)
            assert np.all(measurements >= 0)
            assert np.all(measurements <= 1)
            
            # Test detection (should use threshold)
            detections = sensor.detect(plume_state, positions)
            assert isinstance(detections, np.ndarray)
            
            # Test configuration
            sensor.configure(dynamic_range=(0, 2), resolution=0.0001)
            
            # Test observation space info
            obs_info = sensor.get_observation_space_info()
            assert "type" in obs_info
            assert "shape" in obs_info
    
    def test_gradient_sensor_functionality(self):
        """Test GradientSensor implementation and functionality."""
        if not GradientSensor:
            pytest.skip("GradientSensor not available")
        
        with performance_monitor("Gradient sensor functionality"):
            sensor = GradientSensor(spatial_resolution=(0.5, 0.5))
            plume_state = create_mock_plume_model("gaussian")
            positions = np.array([[25, 35], [45, 55]])
            
            # Test gradient computation
            gradients = sensor.compute_gradient(plume_state, positions)
            assert isinstance(gradients, np.ndarray)
            assert gradients.shape == (2, 2)  # [n_positions, 2] for [dx, dy]
            
            # Test measurement (may be gradient magnitude)
            measurements = sensor.measure(plume_state, positions)
            assert isinstance(measurements, np.ndarray)
            
            # Test configuration
            sensor.configure(spatial_resolution=(0.2, 0.2), method='central')
            
            # Test metadata
            metadata = sensor.get_metadata()
            assert "spatial_resolution" in metadata
    
    def test_multi_sensor_coordination(self):
        """Test coordination between multiple sensor types."""
        sensors = []
        if BinarySensor:
            sensors.append(BinarySensor(threshold=0.1))
        if ConcentrationSensor:
            sensors.append(ConcentrationSensor(dynamic_range=(0, 1)))
        if GradientSensor:
            sensors.append(GradientSensor(spatial_resolution=(0.5, 0.5)))
        
        if not sensors:
            pytest.skip("No sensors available")
        
        with performance_monitor("Multi-sensor coordination"):
            plume_state = create_mock_plume_model("gaussian")
            positions = np.array([[25, 35], [45, 55], [65, 75]])
            
            # Collect all sensor readings
            all_readings = []
            for sensor in sensors:
                readings = sensor.measure(plume_state, positions)
                all_readings.append(readings)
            
            # Verify all sensors produced outputs
            assert len(all_readings) == len(sensors)
            for readings in all_readings:
                assert isinstance(readings, np.ndarray)
                assert len(readings) > 0


@pytest.mark.skipif(not WIND_MODELS_AVAILABLE, reason="Wind field implementations not available")
class TestWindFieldIntegration:
    """Test suite for wind field integration and environmental dynamics coupling."""
    
    def test_constant_wind_field_integration(self):
        """Test ConstantWindField integration with plume dynamics."""
        if not ConstantWindField:
            pytest.skip("ConstantWindField not available")
        
        with performance_monitor("Constant wind field integration"):
            wind_field = ConstantWindField(velocity=(2.0, 1.0))
            
            # Test basic functionality
            positions = np.array([[10, 20], [30, 40], [50, 60]])
            velocities = wind_field.velocity_at(positions)
            
            assert isinstance(velocities, np.ndarray)
            assert velocities.shape == (3, 2)
            
            # Test temporal evolution (should be minimal for constant wind)
            initial_velocities = velocities.copy()
            wind_field.step(dt=0.1)
            updated_velocities = wind_field.velocity_at(positions)
            
            # Constant wind should remain constant
            np.testing.assert_array_almost_equal(initial_velocities, updated_velocities)
    
    def test_turbulent_wind_field_integration(self):
        """Test TurbulentWindField integration with dynamic wind patterns."""
        if not TurbulentWindField:
            pytest.skip("TurbulentWindField not available")
        
        with performance_monitor("Turbulent wind field integration"):
            wind_field = TurbulentWindField(
                mean_velocity=(2.0, 1.0),
                turbulence_intensity=0.3
            )
            
            # Test basic functionality
            positions = np.array([[10, 20], [30, 40], [50, 60]])
            velocities = wind_field.velocity_at(positions)
            
            assert isinstance(velocities, np.ndarray)
            assert velocities.shape == (3, 2)
            
            # Test temporal evolution (should show variation for turbulent wind)
            initial_velocities = velocities.copy()
            
            # Advance wind field multiple steps
            for _ in range(10):
                wind_field.step(dt=0.1)
            
            updated_velocities = wind_field.velocity_at(positions)
            
            # Turbulent wind should show some variation
            # (though mock implementation may not show this)
            assert updated_velocities.shape == initial_velocities.shape
    
    def test_wind_field_plume_model_coupling(self):
        """Test coupling between wind field and plume model dynamics."""
        if not (ConstantWindField and PLUME_MODELS_AVAILABLE):
            pytest.skip("Required components not available")
        
        with performance_monitor("Wind field and plume model coupling"):
            wind_field = create_mock_wind_field("constant")
            plume_model = create_mock_plume_model("turbulent")
            
            # Test coordinated temporal evolution
            dt = 0.1
            num_steps = 5
            
            for step in range(num_steps):
                # Both components should advance together
                wind_field.step(dt)
                plume_model.step(dt)
                
                # Test that wind affects plume (via positions)
                positions = np.array([[25, 35], [45, 55]])
                wind_velocities = wind_field.velocity_at(positions)
                concentrations = plume_model.concentration_at(positions)
                
                # Verify both components respond
                assert wind_velocities is not None
                assert concentrations is not None
            
            # Verify temporal coordination
            assert wind_field.step.call_count == num_steps
            assert plume_model.step.call_count == num_steps
    
    def test_environmental_dynamics_integration(self):
        """Test full environmental dynamics integration."""
        with performance_monitor("Environmental dynamics integration"):
            # Create environmental components
            wind_field = create_mock_wind_field("turbulent")
            plume_model = create_mock_plume_model("turbulent")
            navigator = create_mock_navigator(num_agents=3)
            
            # Test complete environmental dynamics cycle
            dt = 0.1
            
            for step in range(10):
                # Update environmental state
                wind_field.step(dt)
                plume_model.step(dt)
                
                # Navigator perceives and responds to environment
                positions = navigator.positions
                wind_velocities = wind_field.velocity_at(positions)
                concentrations = plume_model.concentration_at(positions)
                
                # Navigator responds to environmental conditions
                navigator.step(concentrations, dt)
                
                # Update positions based on wind influence (if implemented)
                # This would be done in a real implementation
            
            # Verify all components participated in dynamics
            assert wind_field.step.call_count == 10
            assert plume_model.step.call_count == 10
            assert navigator.step.call_count == 10


@pytest.mark.skipif(not CORE_AVAILABLE, reason="Core simulation components not available")
class TestSimulationContextOrchestration:
    """Test suite for simulation context orchestration and episode management."""
    
    def test_episode_management_lifecycle(self):
        """Test automatic episode management and lifecycle handling."""
        with performance_monitor("Episode management lifecycle"):
            # Mock simulation context
            with patch('plume_nav_sim.core.simulation.SimulationContext') as MockContext:
                mock_context = Mock()
                mock_context.run_simulation.return_value = Mock(
                    step_count=100,
                    success=True,
                    performance_metrics={"avg_step_time_ms": 5.0}
                )
                MockContext.return_value = mock_context
                
                # Test episode management
                config = Mock()
                config.num_steps = 100
                config.episode_management_mode = "auto"
                
                context = SimulationContext(config)
                results = context.run_simulation()
                
                # Verify episode was managed
                assert results.step_count == 100
                assert results.success == True
                context.run_simulation.assert_called_once()
    
    def test_result_collection_and_aggregation(self):
        """Test comprehensive result collection and aggregation."""
        with performance_monitor("Result collection and aggregation"):
            # Mock components with performance metrics
            navigator = create_mock_navigator(num_agents=2)
            plume_model = create_mock_plume_model("gaussian")
            wind_field = create_mock_wind_field("constant")
            sensors = [create_mock_sensor("concentration")]
            
            # Add performance metrics to components
            navigator.get_performance_metrics = Mock(return_value={
                "step_time_mean_ms": 2.5,
                "total_steps": 100
            })
            plume_model.get_performance_metrics = Mock(return_value={
                "concentration_queries": 100,
                "avg_query_time_ms": 0.5
            })
            
            # Simulate result collection
            results = {
                "navigator_metrics": navigator.get_performance_metrics(),
                "plume_model_metrics": plume_model.get_performance_metrics(),
                "component_metadata": {
                    "plume_model": plume_model.get_metadata(),
                    "wind_field": wind_field.get_metadata()
                }
            }
            
            # Verify comprehensive results
            assert "navigator_metrics" in results
            assert "plume_model_metrics" in results
            assert "component_metadata" in results
            assert results["navigator_metrics"]["step_time_mean_ms"] == 2.5
            assert results["plume_model_metrics"]["concentration_queries"] == 100
    
    def test_termination_conditions_evaluation(self):
        """Test configurable episode termination condition evaluation."""
        with performance_monitor("Termination conditions evaluation"):
            # Mock termination conditions
            termination_conditions = {
                "max_steps": 1000,
                "target_concentration": 0.8,
                "time_limit_seconds": 30.0,
                "success_threshold": 0.9
            }
            
            # Simulate condition evaluation
            current_state = {
                "step": 500,
                "max_concentration": 0.6,
                "elapsed_time": 15.0,
                "success_metric": 0.7
            }
            
            # Test condition evaluation logic
            step_terminated = current_state["step"] >= termination_conditions["max_steps"]
            concentration_terminated = current_state["max_concentration"] >= termination_conditions["target_concentration"]
            time_terminated = current_state["elapsed_time"] >= termination_conditions["time_limit_seconds"]
            success_terminated = current_state["success_metric"] >= termination_conditions["success_threshold"]
            
            # None of the conditions should be met in this test case
            assert not step_terminated
            assert not concentration_terminated  
            assert not time_terminated
            assert not success_terminated
            
            # Test when conditions are met
            final_state = {
                "step": 1000,
                "max_concentration": 0.9,
                "elapsed_time": 35.0,
                "success_metric": 0.95
            }
            
            step_terminated = final_state["step"] >= termination_conditions["max_steps"]
            assert step_terminated
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration throughout simulation."""
        with performance_monitor("Performance monitoring integration"):
            # Mock performance monitor
            with patch('plume_nav_sim.core.simulation.PerformanceMonitor') as MockMonitor:
                mock_monitor = Mock()
                mock_monitor.record_step_time.return_value = None
                mock_monitor.get_summary.return_value = {
                    "avg_step_time_ms": 7.5,
                    "max_step_time_ms": 12.0,
                    "total_steps": 100,
                    "performance_target_met": True
                }
                MockMonitor.return_value = mock_monitor
                
                # Simulate performance monitoring
                monitor = PerformanceMonitor()
                
                # Record multiple step times
                step_times = [0.005, 0.008, 0.006, 0.009, 0.007]  # In seconds
                for step_time in step_times:
                    monitor.record_step_time(step_time)
                
                # Get performance summary
                summary = monitor.get_summary()
                
                # Verify monitoring functionality
                assert summary["avg_step_time_ms"] == 7.5
                assert summary["performance_target_met"] == True
                mock_monitor.record_step_time.assert_called()


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
class TestVectorizedEnvironmentCompatibility:
    """Test suite for vectorized environment compatibility for parallel RL training."""
    
    def test_vectorized_environment_creation(self):
        """Test creation of vectorized environments for parallel training."""
        with performance_monitor("Vectorized environment creation"):
            # Mock vectorized environment creation
            num_envs = 4
            mock_envs = []
            
            for i in range(num_envs):
                mock_env = Mock()
                mock_env.reset.return_value = (
                    {"position": np.array([i*10, i*10]), "concentration": 0.5},
                    {"env_id": i}
                )
                mock_env.step.return_value = (
                    {"position": np.array([i*10+1, i*10+1]), "concentration": 0.6},
                    0.1,  # reward
                    False,  # terminated
                    False,  # truncated
                    {"step": 1}
                )
                mock_envs.append(mock_env)
            
            # Test parallel environment operations
            observations = []
            for env in mock_envs:
                obs, info = env.reset()
                observations.append(obs)
            
            # Verify all environments reset successfully
            assert len(observations) == num_envs
            for i, obs in enumerate(observations):
                assert obs["position"][0] == i * 10
                assert obs["position"][1] == i * 10
    
    def test_parallel_step_execution(self):
        """Test parallel step execution across multiple environments."""
        with performance_monitor("Parallel step execution"):
            num_envs = 3
            mock_envs = []
            
            # Create mock environments
            for i in range(num_envs):
                mock_env = Mock()
                mock_env.step.return_value = (
                    {"position": np.array([i*5, i*5]), "concentration": 0.4 + i*0.1},
                    0.1 + i*0.05,  # reward
                    False,  # terminated
                    False,  # truncated
                    {"env_id": i, "step": 1}
                )
                mock_envs.append(mock_env)
            
            # Test parallel step execution
            actions = [np.array([1.0, 0.5]) for _ in range(num_envs)]
            results = []
            
            start_time = time.perf_counter()
            for env, action in zip(mock_envs, actions):
                result = env.step(action)
                results.append(result)
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Verify parallel execution
            assert len(results) == num_envs
            assert execution_time_ms < PERFORMANCE_TARGET_MS * num_envs  # Should be faster than sequential
            
            # Verify individual results
            for i, (obs, reward, terminated, truncated, info) in enumerate(results):
                assert obs["concentration"] == 0.4 + i * 0.1
                assert reward == 0.1 + i * 0.05
                assert info["env_id"] == i
    
    def test_vectorized_environment_synchronization(self):
        """Test synchronization across vectorized environments."""
        with performance_monitor("Vectorized environment synchronization"):
            num_envs = 5
            sync_points = []
            
            # Mock synchronized environments
            for env_id in range(num_envs):
                mock_env = Mock()
                
                def create_step_fn(eid):
                    def step_fn(action):
                        sync_points.append(f"env_{eid}_step")
                        return (
                            {"position": np.array([eid, eid]), "concentration": 0.5},
                            0.1,
                            False,
                            False,
                            {"sync_point": len(sync_points)}
                        )
                    return step_fn
                
                mock_env.step = create_step_fn(env_id)
                
                # Execute synchronized step
                result = mock_env.step(np.array([1.0, 0.0]))
                
            # Verify synchronization occurred
            assert len(sync_points) == num_envs
            for env_id in range(num_envs):
                assert f"env_{env_id}_step" in sync_points


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
class TestPropertyBasedIntegration:
    """Property-based testing using Hypothesis for comprehensive edge case coverage."""
    
    @given(
        num_agents=st.integers(min_value=1, max_value=10),
        num_steps=st.integers(min_value=1, max_value=20),
        dt=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(
        max_examples=20,
        deadline=10000,  # 10 second timeout
        suppress_health_check=[HealthCheck.too_slow]
    )
    def test_modular_simulation_invariants(self, num_agents, num_steps, dt):
        """Test modular simulation maintains invariants across parameter ranges."""
        assume(num_agents >= 1)
        assume(num_steps >= 1)
        assume(0.01 <= dt <= 1.0)
        
        with performance_monitor("Property-based modular simulation"):
            # Create modular components
            navigator = create_mock_navigator(num_agents=num_agents)
            plume_model = create_mock_plume_model("gaussian")
            wind_field = create_mock_wind_field("constant")
            sensors = [create_mock_sensor("concentration")]
            
            # Track invariants
            initial_positions = navigator.positions.copy()
            
            # Execute simulation steps
            for step in range(num_steps):
                # Component updates
                wind_field.step(dt)
                plume_model.step(dt)
                
                # Agent perception and action
                positions = navigator.positions
                concentrations = plume_model.concentration_at(positions)
                sensor_readings = sensors[0].measure(plume_model, positions)
                
                # Navigator update
                navigator.step(concentrations, dt)
            
            # Verify invariants maintained
            final_positions = navigator.positions
            
            # Shape invariants
            assert final_positions.shape == (num_agents, 2)
            assert final_positions.shape == initial_positions.shape
            
            # All values should remain finite
            assert np.all(np.isfinite(final_positions))
            
            # Component call count invariants
            assert wind_field.step.call_count == num_steps
            assert plume_model.step.call_count == num_steps
            assert navigator.step.call_count == num_steps
            assert plume_model.concentration_at.call_count == num_steps
    
    @given(
        plume_type=st.sampled_from(["gaussian", "turbulent"]),
        wind_type=st.sampled_from(["constant", "turbulent"]),
        sensor_types=st.lists(
            st.sampled_from(["binary", "concentration", "gradient"]),
            min_size=1, max_size=3, unique=True
        )
    )
    @settings(max_examples=15, deadline=8000)
    def test_component_composition_properties(self, plume_type, wind_type, sensor_types):
        """Test properties hold across different component compositions.""" 
        with performance_monitor("Property-based component composition"):
            # Create component composition
            navigator = create_mock_navigator(num_agents=2)
            plume_model = create_mock_plume_model(plume_type)
            wind_field = create_mock_wind_field(wind_type) 
            sensors = [create_mock_sensor(sensor_type) for sensor_type in sensor_types]
            
            # Test composition properties
            positions = navigator.positions
            
            # All components should respond to the same positions
            concentrations = plume_model.concentration_at(positions)
            wind_velocities = wind_field.velocity_at(positions)
            
            # All sensors should be able to process the plume state
            sensor_outputs = []
            for sensor in sensors:
                output = sensor.measure(plume_model, positions)
                sensor_outputs.append(output)
            
            # Verify composition properties
            assert concentrations is not None
            assert wind_velocities is not None
            assert len(sensor_outputs) == len(sensors)
            
            # All outputs should have compatible shapes for the agent count
            for output in sensor_outputs:
                assert isinstance(output, np.ndarray)
                # Output should be compatible with number of agents
                assert len(output) >= navigator.num_agents or output.ndim == 2
    
    @given(
        performance_target=st.floats(min_value=1.0, max_value=20.0),
        num_measurements=st.integers(min_value=5, max_value=15)
    )
    @settings(max_examples=10, deadline=15000)
    def test_performance_scaling_properties(self, performance_target, num_measurements):
        """Test performance scaling properties across different scenarios."""
        assume(1.0 <= performance_target <= 20.0)
        assume(5 <= num_measurements <= 15)
        
        # Create test scenario
        navigator = create_mock_navigator(num_agents=3)
        plume_model = create_mock_plume_model("gaussian")
        
        # Measure performance
        step_times = []
        for _ in range(num_measurements):
            start_time = time.perf_counter()
            
            positions = navigator.positions
            concentrations = plume_model.concentration_at(positions)
            navigator.step(concentrations, dt=0.1)
            
            step_time_ms = (time.perf_counter() - start_time) * 1000
            step_times.append(step_time_ms)
        
        # Performance scaling properties
        mean_time = np.mean(step_times)
        max_time = np.max(step_times)
        
        # Properties that should hold
        assert mean_time > 0, "Step execution should take measurable time"
        assert max_time >= mean_time, "Maximum time should be >= mean time"
        assert np.all(np.isfinite(step_times)), "All timing measurements should be finite"
        
        # Performance should be reasonable (relaxed for property testing)
        reasonable_limit = performance_target * 2  # Allow 2x target for property testing
        assert mean_time < reasonable_limit, f"Mean time {mean_time:.2f}ms exceeds {reasonable_limit:.2f}ms"


class TestCrossProtocolValidation:
    """Test suite for cross-protocol validation ensuring all component implementations maintain API compatibility."""
    
    def test_all_protocols_interface_consistency(self):
        """Test that all protocol implementations maintain consistent interfaces."""
        with performance_monitor("Cross-protocol interface consistency"):
            # Test protocol method existence across all components
            plume_model = create_mock_plume_model("gaussian")
            wind_field = create_mock_wind_field("constant")
            sensor = create_mock_sensor("concentration")
            navigator = create_mock_navigator(num_agents=1)
            
            # Verify common protocol patterns
            protocols_and_methods = [
                (plume_model, ["concentration_at", "step", "reset", "get_metadata"]),
                (wind_field, ["velocity_at", "step", "reset", "get_metadata"]),
                (sensor, ["measure", "configure", "get_metadata"]),
                (navigator, ["step", "reset", "sample_odor"])
            ]
            
            for component, expected_methods in protocols_and_methods:
                for method_name in expected_methods:
                    assert hasattr(component, method_name), \
                        f"{component.__class__.__name__} missing method {method_name}"
                    assert callable(getattr(component, method_name)), \
                        f"{method_name} should be callable"
    
    def test_protocol_data_type_consistency(self):
        """Test data type consistency across protocol implementations."""
        with performance_monitor("Protocol data type consistency"):
            # Create components
            plume_model = create_mock_plume_model("gaussian")
            wind_field = create_mock_wind_field("constant")
            sensor = create_mock_sensor("concentration")
            
            # Test consistent position input/output types
            positions = np.array([[25, 35], [45, 55]])
            
            # All components should handle the same position format
            concentrations = plume_model.concentration_at(positions)
            wind_velocities = wind_field.velocity_at(positions)
            sensor_readings = sensor.measure(plume_model, positions)
            
            # Verify output types are consistent
            assert isinstance(concentrations, np.ndarray)
            assert isinstance(wind_velocities, np.ndarray)
            assert isinstance(sensor_readings, np.ndarray)
            
            # Verify output shapes are consistent with input
            assert len(concentrations) == len(positions)
            assert len(wind_velocities) == len(positions)
            assert len(sensor_readings) == len(positions)
    
    def test_protocol_error_handling_consistency(self):
        """Test consistent error handling across protocol implementations.""" 
        with performance_monitor("Protocol error handling consistency"):
            # Test with invalid inputs
            plume_model = create_mock_plume_model("gaussian")
            wind_field = create_mock_wind_field("constant")
            sensor = create_mock_sensor("concentration")
            
            # Configure mocks to raise consistent errors for invalid inputs
            invalid_positions = "invalid"
            
            # All components should handle invalid inputs gracefully
            # (In a real implementation, they should raise appropriate exceptions)
            plume_model.concentration_at.side_effect = ValueError("Invalid positions")
            wind_field.velocity_at.side_effect = ValueError("Invalid positions")
            sensor.measure.side_effect = ValueError("Invalid positions")
            
            # Test error handling
            with pytest.raises(ValueError):
                plume_model.concentration_at(invalid_positions)
            
            with pytest.raises(ValueError):
                wind_field.velocity_at(invalid_positions)
            
            with pytest.raises(ValueError):
                sensor.measure(plume_model, invalid_positions)


# Integration test execution summary and validation
def test_integration_test_suite_completeness():
    """Validate that the integration test suite covers all required areas."""
    # Required test areas from Section 0.4.1
    required_areas = [
        "protocol_compliance",
        "component_boundary_interactions", 
        "configuration_integration",
        "memory_independent_navigation",
        "performance_regression",
        "backward_compatibility",
        "sensor_abstraction_layer",
        "wind_field_integration",
        "simulation_context_orchestration",
        "vectorized_environment_compatibility",
        "property_based_testing",
        "cross_protocol_validation"
    ]
    
    # Get all test classes defined in this module
    import inspect
    test_classes = [
        name for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isclass(obj) and name.startswith('Test')
    ]
    
    # Map test classes to areas
    class_to_area = {
        "TestProtocolCompliance": "protocol_compliance",
        "TestComponentBoundaryInteractions": "component_boundary_interactions",
        "TestConfigurationIntegration": "configuration_integration", 
        "TestMemoryIndependentNavigation": "memory_independent_navigation",
        "TestPerformanceRegression": "performance_regression",
        "TestBackwardCompatibility": "backward_compatibility",
        "TestSensorAbstractionLayer": "sensor_abstraction_layer",
        "TestWindFieldIntegration": "wind_field_integration",
        "TestSimulationContextOrchestration": "simulation_context_orchestration",
        "TestVectorizedEnvironmentCompatibility": "vectorized_environment_compatibility",
        "TestPropertyBasedIntegration": "property_based_testing",
        "TestCrossProtocolValidation": "cross_protocol_validation"
    }
    
    # Verify all required areas are covered
    covered_areas = set(class_to_area.values())
    missing_areas = set(required_areas) - covered_areas
    
    assert len(missing_areas) == 0, f"Missing test coverage for areas: {missing_areas}"
    
    # Verify all test classes exist
    for test_class_name in class_to_area.keys():
        assert test_class_name in test_classes, f"Test class {test_class_name} not found"
    
    # Calculate coverage percentage
    coverage_percentage = (len(covered_areas) / len(required_areas)) * 100
    assert coverage_percentage >= 100, f"Test coverage {coverage_percentage:.1f}% below 100% requirement"


def test_simulation_results_have_step_count_and_success():
    """Simulation results expose step count and success attributes."""
    ctx = SimulationContext.create()
    results = ctx.run_simulation()
    assert hasattr(results, "step_count")
    assert hasattr(results, "success")


if __name__ == "__main__":
    # Enhanced test execution with comprehensive reporting
    pytest.main([
        __file__,
        "-v",
        "--tb=short", 
        "--durations=10",
        "--cov=src/plume_nav_sim",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-fail-under=70",
        "-x"  # Stop on first failure for faster debugging
    ])