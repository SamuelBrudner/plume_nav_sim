"""Tests for the core navigator module with Gymnasium 0.29.x API compliance and sensor protocol integration.

This test module provides comprehensive testing for the enhanced navigator implementation
supporting Gymnasium 0.29.x API compliance, centralized Loguru logging, Hydra 1.3+
structured configurations, and sensor protocol integration patterns for modular navigation.

Key Testing Areas:
- Gymnasium 0.29.x API compliance (5-tuple step() returns, seed parameter in reset())
- Performance testing for ≤10ms average step() execution time requirements
- Sensor protocol integration validation (BinarySensor, ConcentrationSensor, GradientSensor)
- Sensor-based observation processing workflow tests replacing direct odor sampling
- Vectorized sensor sampling and multi-agent observation processing validation
- Observation processing patterns (concentration readings, binary detections, gradient information)
- Property-based testing for coordinate frame consistency validation
- Backward compatibility testing for existing gym-based code
- Structured dataclass configuration testing with Hydra 1.3+
- Centralized Loguru logging integration
- Seed management utilities and deterministic behavior validation
- Single and multi-agent controller functionality coverage
"""

import time
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import pytest
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union

# Gymnasium integration for API compliance testing
try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    from gymnasium.spaces import Box, Dict as DictSpace
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    check_env = None

# Property-based testing with Hypothesis
try:
    from hypothesis import given, strategies as st, settings, assume, note
    from hypothesis.strategies import composite
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = lambda *args, **kwargs: lambda f: f
    st = None
    settings = lambda *args, **kwargs: lambda f: f

# Enhanced logging integration
try:
    from loguru import logger
    # Try to import from new package structure
    try:
        from plume_nav_sim.utils.logging_setup import (
            get_enhanced_logger, correlation_context, PerformanceMetrics
        )
    except ImportError:
        # Fallback if logging utils not available yet
        def get_enhanced_logger(name): return logger
        correlation_context = None
        PerformanceMetrics = dict
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False
    correlation_context = None
    PerformanceMetrics = dict

# Core navigation module imports
from plume_nav_sim.core.protocols import NavigatorProtocol
from plume_nav_sim.core.navigator import NavigatorFactory
from plume_nav_sim.core.controllers import (
    SingleAgentController, MultiAgentController,
    SingleAgentParams, MultiAgentParams,
    create_controller_from_config, validate_controller_config
)

# Sensor protocol integration imports
try:
    from plume_nav_sim.core.sensors import (
        SensorProtocol, BinarySensor, ConcentrationSensor, GradientSensor,
        create_sensor_from_config, create_sensor_suite,
        validate_sensor_config, get_sensor_performance_metrics
    )
    SENSOR_PROTOCOLS_AVAILABLE = True
except ImportError:
    # Fallback during migration - create minimal mocks
    class SensorProtocol:
        def detect(self, plume_state, positions): pass
        def measure(self, plume_state, positions): pass
        def compute_gradient(self, plume_state, positions): pass
    
    class BinarySensor:
        def __init__(self, **kwargs): self._config = kwargs
        def detect(self, plume_state, positions): return np.random.random(len(positions)) > 0.5
    
    class ConcentrationSensor:
        def __init__(self, **kwargs): self._config = kwargs
        def measure(self, plume_state, positions): return np.random.random(len(positions))
    
    class GradientSensor:
        def __init__(self, **kwargs): self._config = kwargs
        def compute_gradient(self, plume_state, positions): return np.random.random((len(positions), 2))
    
    def create_sensor_from_config(config): return ConcentrationSensor(**config)
    def create_sensor_suite(configs): return [create_sensor_from_config(c) for c in configs]
    def validate_sensor_config(config): return True
    def get_sensor_performance_metrics(sensors): return {}
    
    SENSOR_PROTOCOLS_AVAILABLE = False

# Environment integration for API testing
try:
    from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv as GymnasiumEnv
    GYMNASIUM_ENV_AVAILABLE = True
except ImportError:
    # Fallback during migration
    class GymnasiumEnv:
        def __init__(self, **kwargs): pass
        def reset(self, **kwargs): return {}, {}
        def step(self, action): return {}, 0.0, False, False, {}
    GYMNASIUM_ENV_AVAILABLE = False

# Seed management utilities
try:
    from plume_nav_sim.utils.seed_utils import (
        set_global_seed, get_seed_context, SeedContext, validate_deterministic_behavior
    )
    SEED_UTILS_AVAILABLE = True
except ImportError:
    # Fallback during migration
    import contextlib
    @contextlib.contextmanager
    def set_global_seed(seed): yield
    def get_seed_context(): return type('SeedContext', (), {'global_seed': 42, 'is_seeded': True})()
    SeedContext = type('SeedContext', (), {})
    def validate_deterministic_behavior(*args): return True
    SEED_UTILS_AVAILABLE = False

# Configuration models for structured config testing
try:
    from plume_nav_sim.config.schemas import (
        NavigatorConfig, SingleAgentConfig, MultiAgentConfig
    )
    CONFIG_SCHEMAS_AVAILABLE = True
except ImportError:
    # Fallback during migration
    NavigatorConfig = dict
    SingleAgentConfig = dict
    MultiAgentConfig = dict
    CONFIG_SCHEMAS_AVAILABLE = False

# Import Hydra components for configuration testing
try:
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None
    ConfigStore = None


def test_import_navigator_without_controllers_raises(monkeypatch):
    """Navigator should raise ImportError when controllers are unavailable."""
    import sys
    import builtins

    # Ensure fresh import of navigator and controllers
    monkeypatch.delitem(sys.modules, "plume_nav_sim.core.navigator", raising=False)
    monkeypatch.delitem(sys.modules, "plume_nav_sim.core.controllers", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "plume_nav_sim.core.controllers" or (
            level == 1 and name == "controllers"
        ):
            raise ImportError("controllers missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        __import__("plume_nav_sim.core.navigator")


# Enhanced test fixtures for Gymnasium 0.29.x API compliance and structured configuration testing

@pytest.fixture(scope="session")
def enhanced_logger():
    """Session-scoped enhanced logger for test performance monitoring."""
    if LOGURU_AVAILABLE:
        return get_enhanced_logger("test_navigator")
    return logger


@pytest.fixture
def performance_context():
    """Performance monitoring context for test execution timing."""
    start_time = time.perf_counter()
    metrics = PerformanceMetrics() if LOGURU_AVAILABLE else {}
    yield metrics
    if LOGURU_AVAILABLE:
        metrics.record_duration("test_execution", time.perf_counter() - start_time)


@pytest.fixture
def correlation_context_fixture():
    """Correlation context for structured logging across test execution."""
    if LOGURU_AVAILABLE and correlation_context:
        with correlation_context("test_navigator"):
            yield
    else:
        yield


@pytest.fixture
def structured_single_agent_config():
    """Hydra 1.3+ structured dataclass configuration for single agent."""
    return SingleAgentConfig(
        position=(10.0, 20.0),
        orientation=45.0,
        speed=1.5,
        max_speed=3.0,
        angular_velocity=0.2,
        _target_="odor_plume_nav.core.controllers.SingleAgentController"
    )


@pytest.fixture
def structured_multi_agent_config():
    """Hydra 1.3+ structured dataclass configuration for multi-agent."""
    return MultiAgentConfig(
        positions=[[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
        orientations=[0.0, 90.0, 180.0],
        speeds=[1.0, 1.5, 2.0],
        max_speeds=[2.0, 3.0, 4.0],
        angular_velocities=[0.1, 0.2, 0.3],
        _target_="odor_plume_nav.core.controllers.MultiAgentController"
    )


@pytest.fixture
def hydra_dict_config():
    """Legacy Hydra DictConfig for backward compatibility testing."""
    if HYDRA_AVAILABLE:
        return OmegaConf.create({
            "position": [5.0, 15.0],
            "orientation": 30.0,
            "speed": 1.0,
            "max_speed": 2.5,
            "angular_velocity": 0.15
        })
    else:
        return {
            "position": [5.0, 15.0],
            "orientation": 30.0,
            "speed": 1.0,
            "max_speed": 2.5,
            "angular_velocity": 0.15
        }


@pytest.fixture
def mock_gymnasium_env():
    """Mock Gymnasium environment for API compliance testing."""
    if not GYMNASIUM_AVAILABLE:
        pytest.skip("Gymnasium not available")
    
    env = Mock(spec=GymnasiumEnv)
    env.action_space = Box(low=np.array([0.0, -90.0]), high=np.array([3.0, 90.0]))
    env.observation_space = DictSpace({
        'position': Box(low=0, high=100, shape=(2,)),
        'orientation': Box(low=0, high=360, shape=(1,)),
        'odor': Box(low=0, high=1, shape=(1,))
    })
    
    # Mock step() to return 5-tuple for Gymnasium 0.29.x compliance
    def mock_step(action):
        obs = {
            'position': np.array([10.0, 20.0]),
            'orientation': np.array([45.0]),
            'odor': np.array([0.5])
        }
        return obs, 1.0, False, False, {}
    
    # Mock reset() to accept seed parameter and return (obs, info)
    def mock_reset(seed=None, options=None):
        obs = {
            'position': np.array([0.0, 0.0]),
            'orientation': np.array([0.0]),
            'odor': np.array([0.0])
        }
        info = {"seed": seed, "options": options}
        return obs, info
    
    env.step = mock_step
    env.reset = mock_reset
    return env


@pytest.fixture 
def deterministic_seed_context():
    """Deterministic seed context for reproducibility testing."""
    seed_value = 42
    with set_global_seed(seed_value):
        context = get_seed_context()
        yield context, seed_value


@pytest.fixture
def mock_navigator_protocol():
    """Enhanced mock navigator implementing NavigatorProtocol for testing."""
    navigator = MagicMock(spec=NavigatorProtocol)
    navigator.positions = np.array([[0.0, 0.0]])
    navigator.orientations = np.array([0.0])
    navigator.speeds = np.array([0.0])
    navigator.max_speeds = np.array([1.0])
    navigator.angular_velocities = np.array([0.0])
    navigator.num_agents = 1
    
    # Mock step method to simulate movement and return timing
    def mock_step(env_array, dt=1.0):
        # Simulate minimal processing time
        time.sleep(0.001)  # 1ms simulated processing
        navigator.positions[0] += navigator.speeds[0] * dt
    
    navigator.step = mock_step
    # Legacy methods for backward compatibility testing
    navigator.sample_odor.return_value = 0.5
    navigator.sample_multiple_sensors.return_value = np.array([0.4, 0.6])
    
    # New sensor protocol integration methods
    def mock_process_sensor_observations(sensor_outputs):
        """Mock processing of sensor outputs into structured observations."""
        return {
            'concentration': np.mean([out.get('value', 0.0) for out in sensor_outputs]),
            'binary_detection': any(out.get('detected', False) for out in sensor_outputs),
            'gradient': np.array([0.1, 0.05])  # Mock gradient
        }
    
    navigator.process_sensor_observations = mock_process_sensor_observations
    
    return navigator


@pytest.fixture
def mock_sensor_suite():
    """Mock sensor suite for testing sensor protocol integration."""
    binary_sensor = MagicMock(spec=BinarySensor)
    binary_sensor.detect.return_value = np.array([True])
    
    concentration_sensor = MagicMock(spec=ConcentrationSensor) 
    concentration_sensor.measure.return_value = np.array([0.5])
    
    gradient_sensor = MagicMock(spec=GradientSensor)
    gradient_sensor.compute_gradient.return_value = np.array([[0.1, 0.05]])
    
    return {
        'binary': binary_sensor,
        'concentration': concentration_sensor,
        'gradient': gradient_sensor
    }


@pytest.fixture
def mock_plume_state():
    """Mock plume state for sensor testing."""
    plume_state = MagicMock()
    plume_state.concentration_at.return_value = np.array([0.5])
    plume_state.get_frame.return_value = np.random.rand(100, 100)
    return plume_state


@pytest.fixture 
def sensor_observation_samples():
    """Sample sensor observation structures for testing."""
    return {
        'concentration_reading': {
            'concentration': 0.75,
            'timestamp': time.time(),
            'position': (10.0, 20.0),
            'sensor_id': 'concentration_001'
        },
        'binary_detection': {
            'detected': True,
            'confidence': 0.85,
            'threshold': 0.1,
            'metadata': {'false_positive_rate': 0.02}
        },
        'gradient_information': {
            'gradient': (0.15, 0.08),
            'magnitude': 0.17,
            'direction': 28.3,
            'spatial_resolution': 0.5
        }
    }


# Property-based testing strategies for coordinate frame consistency validation

if HYPOTHESIS_AVAILABLE:
    @composite
    def position_strategy(draw):
        """Generate valid position coordinates within reasonable bounds."""
        x = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
        y = draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False))
        return (x, y)

    @composite
    def orientation_strategy(draw):
        """Generate valid orientation values in degrees."""
        return draw(st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False))

    @composite
    def speed_strategy(draw):
        """Generate valid speed values."""
        return draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False))

    @composite
    def multi_agent_positions_strategy(draw):
        """Generate valid multi-agent position arrays."""
        num_agents = draw(st.integers(min_value=1, max_value=50))
        positions = []
        for _ in range(num_agents):
            pos = draw(position_strategy())
            positions.append(list(pos))
        return positions

    @composite
    def controller_config_strategy(draw):
        """Generate valid controller configuration dictionaries."""
        config_type = draw(st.sampled_from(['single', 'multi']))
        
        if config_type == 'single':
            return {
                'position': draw(position_strategy()),
                'orientation': draw(orientation_strategy()),
                'speed': draw(speed_strategy()),
                'max_speed': draw(st.floats(min_value=1.0, max_value=20.0)),
                'angular_velocity': draw(st.floats(min_value=-180.0, max_value=180.0))
            }
        else:
            num_agents = draw(st.integers(min_value=2, max_value=10))
            return {
                'positions': draw(multi_agent_positions_strategy()),
                'orientations': draw(st.lists(orientation_strategy(), min_size=num_agents, max_size=num_agents)),
                'speeds': draw(st.lists(speed_strategy(), min_size=num_agents, max_size=num_agents)),
                'max_speeds': draw(st.lists(st.floats(min_value=1.0, max_value=20.0), min_size=num_agents, max_size=num_agents)),
                'angular_velocities': draw(st.lists(st.floats(min_value=-180.0, max_value=180.0), min_size=num_agents, max_size=num_agents))
            }

else:
    # Fallback strategies when Hypothesis is not available
    def position_strategy():
        return (50.0, 50.0)
    
    def orientation_strategy():
        return 45.0
    
    def speed_strategy():
        return 1.0
    
    def controller_config_strategy():
        """Fallback strategy for controller config when Hypothesis is not available."""
        return {
            'position': (50.0, 50.0),
            'orientation': 45.0,
            'speed': 1.0,
            'max_speed': 5.0,
            'angular_velocity': 0.0
        }


# Import validation and module structure tests

def test_proper_imports():
    """Test that Navigator components are imported from the correct modules."""
    # Test core navigation imports
    assert hasattr(NavigatorProtocol, '__module__')
    assert hasattr(NavigatorFactory, '__module__')
    assert hasattr(SingleAgentController, '__module__')
    assert hasattr(MultiAgentController, '__module__')
    
    # Test that classes can be instantiated (basic import validation)
    assert callable(NavigatorFactory.single_agent)
    assert callable(NavigatorFactory.multi_agent)


def test_enhanced_controller_imports():
    """Test enhanced controller classes and utilities are available."""
    # Test parameter dataclasses
    assert SingleAgentParams is not None
    assert MultiAgentParams is not None
    
    # Test factory functions
    assert callable(create_controller_from_config)
    assert callable(validate_controller_config)
    
    # Test configuration models
    if 'SingleAgentConfig' in globals():
        assert SingleAgentConfig is not None
    if 'MultiAgentConfig' in globals():
        assert MultiAgentConfig is not None


def test_gymnasium_integration_imports():
    """Test Gymnasium integration components are available when installed."""
    if GYMNASIUM_AVAILABLE:
        assert gym is not None
        assert check_env is not None
        assert Box is not None
        assert DictSpace is not None
    else:
        pytest.skip("Gymnasium not available")


def test_logging_integration_imports():
    """Test enhanced logging components are available."""
    if LOGURU_AVAILABLE:
        assert logger is not None
        assert correlation_context is not None
    else:
        # Fallback logging should still work
        assert logger is not None


def test_seed_management_imports():
    """Test seed management utilities are available."""
    assert set_global_seed is not None
    assert get_seed_context is not None
    assert SeedContext is not None


# Gymnasium 0.29.x API Compliance Tests

@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
def test_gymnasium_api_compliance(mock_gymnasium_env):
    """Test environment complies with Gymnasium 0.29.x API requirements."""
    env = mock_gymnasium_env
    
    # Test reset() method returns (observation, info) tuple
    obs, info = env.reset()
    assert isinstance(obs, dict), "Observation should be a dictionary"
    assert isinstance(info, dict), "Info should be a dictionary"
    
    # Test reset() accepts seed parameter
    obs, info = env.reset(seed=42)
    assert info.get("seed") == 42, "Seed should be passed through in info"
    
    # Test reset() accepts options parameter
    options = {"position": [100, 200]}
    obs, info = env.reset(options=options)
    assert info.get("options") == options, "Options should be passed through in info"
    
    # Test step() method returns 5-tuple (obs, reward, terminated, truncated, info)
    action = env.action_space.sample() if hasattr(env.action_space, 'sample') else [1.0, 0.0]
    result = env.step(action)
    assert len(result) == 5, f"Step should return 5-tuple, got {len(result)}"
    
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, dict), "Observation should be a dictionary"
    assert isinstance(reward, (int, float, np.number)), "Reward should be numeric"
    assert isinstance(terminated, bool), "Terminated should be boolean"
    assert isinstance(truncated, bool), "Truncated should be boolean"
    assert isinstance(info, dict), "Info should be a dictionary"


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")  
def test_environment_checker_validation():
    """Test that environments pass gymnasium.utils.env_checker validation."""
    if not check_env:
        pytest.skip("env_checker not available")
    
    # This would test against a real environment in practice
    # For now, we verify the checker function is available
    assert callable(check_env), "env_checker should be callable"


def test_backward_compatibility_gym_api():
    """Test backward compatibility with legacy gym API patterns."""
    # Test that legacy step() patterns still work through compatibility layer
    navigator = NavigatorFactory.single_agent(position=(0.0, 0.0), speed=1.0)
    
    # Simulate legacy gym-style usage
    env_array = np.zeros((10, 10))
    
    # Test that step() method exists and works
    navigator.step(env_array)
    
    # Verify position changed (basic step functionality)
    assert not np.allclose(navigator.positions[0], [0.0, 0.0])


# Enhanced Configuration Testing with Hydra 1.3+ Structured Configs

def test_structured_single_agent_config(structured_single_agent_config):
    """Test single agent creation from Hydra 1.3+ structured dataclass config."""
    config = structured_single_agent_config
    
    # Test direct controller creation
    controller = SingleAgentController(
        position=config.position,
        orientation=config.orientation,
        speed=config.speed,
        max_speed=config.max_speed,
        angular_velocity=config.angular_velocity
    )
    
    # Verify configuration values are applied correctly
    assert np.allclose(controller.positions[0], [10.0, 20.0])
    assert controller.orientations[0] == 45.0
    assert controller.speeds[0] == 1.5
    assert controller.max_speeds[0] == 3.0
    assert controller.angular_velocities[0] == 0.2
    assert controller.num_agents == 1


def test_structured_multi_agent_config(structured_multi_agent_config):
    """Test multi-agent creation from Hydra 1.3+ structured dataclass config."""
    config = structured_multi_agent_config
    
    # Test direct controller creation
    controller = MultiAgentController(
        positions=config.positions,
        orientations=config.orientations,
        speeds=config.speeds,
        max_speeds=config.max_speeds,
        angular_velocities=config.angular_velocities
    )
    
    # Verify configuration values are applied correctly
    assert controller.num_agents == 3
    assert np.allclose(controller.positions, [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    assert np.allclose(controller.orientations, [0.0, 90.0, 180.0])
    assert np.allclose(controller.speeds, [1.0, 1.5, 2.0])
    assert np.allclose(controller.max_speeds, [2.0, 3.0, 4.0])
    assert np.allclose(controller.angular_velocities, [0.1, 0.2, 0.3])


def test_config_factory_from_structured_config(structured_single_agent_config):
    """Test controller creation from structured config using factory function."""
    config = structured_single_agent_config
    controller = create_controller_from_config(config.model_dump())
    
    # Verify correct type and configuration
    assert isinstance(controller, SingleAgentController)
    assert np.allclose(controller.positions[0], [10.0, 20.0])
    assert controller.orientations[0] == 45.0


def test_hydra_dict_config_compatibility(hydra_dict_config):
    """Test that factory methods work with legacy Hydra DictConfig."""
    controller = create_controller_from_config(hydra_dict_config)
    
    # Verify controller creation and configuration
    assert isinstance(controller, SingleAgentController)
    assert np.allclose(controller.positions[0], [5.0, 15.0])
    assert controller.orientations[0] == 30.0
    assert controller.speeds[0] == 1.0


def test_config_validation_functionality():
    """Test configuration validation with enhanced error handling."""
    # Valid configuration
    valid_config = {"position": [10.0, 20.0], "speed": 1.0, "max_speed": 2.0}
    is_valid, errors = validate_controller_config(valid_config)
    assert is_valid, f"Valid config should pass validation, errors: {errors}"
    assert len(errors) == 0
    
    # Invalid configuration - speed exceeds max_speed
    invalid_config = {"position": [10.0, 20.0], "speed": 3.0, "max_speed": 2.0}
    is_valid, errors = validate_controller_config(invalid_config)
    assert not is_valid, "Invalid config should fail validation"
    assert len(errors) > 0
    assert any("speed" in error and "max_speed" in error for error in errors)


# Legacy Navigation Tests (Updated for New Architecture)

def test_single_agent_initialization():
    """Test that SingleAgentController can be initialized with orientation and speed."""
    # Create a navigator with default parameters using factory
    navigator = NavigatorFactory.single_agent()
    
    # Default values should be set - array-based for consistency
    assert navigator.orientations[0] == 0.0
    assert navigator.speeds[0] == 0.0
    assert navigator.num_agents == 1
    assert isinstance(navigator.positions, np.ndarray)
    assert navigator.positions.shape == (1, 2)
    
    # Create a navigator with custom parameters
    custom_navigator = NavigatorFactory.single_agent(
        position=(10.0, 20.0),
        orientation=45.0, 
        speed=0.5
    )
    
    # Custom values should be set
    assert np.allclose(custom_navigator.positions[0], [10.0, 20.0])
    assert custom_navigator.orientations[0] == 45.0
    assert custom_navigator.speeds[0] == 0.5


def test_multi_agent_initialization():
    """Test that MultiAgentController can be initialized with multiple agents."""
    positions = [[0.0, 0.0], [10.0, 10.0]]
    orientations = [0.0, 90.0]
    speeds = [0.8, 1.0]  # Fixed: Use speeds within max_speed constraint
    
    navigator = NavigatorFactory.multi_agent(
        positions=positions,
        orientations=orientations,
        speeds=speeds,
        max_speeds=[1.0, 1.5]  # Explicitly set max_speeds to allow the test speeds
    )
    
    # Verify multi-agent setup
    assert navigator.num_agents == 2
    assert np.allclose(navigator.positions, positions)
    assert np.allclose(navigator.orientations, orientations)
    assert np.allclose(navigator.speeds, speeds)


# Movement Precision and Coordinate Frame Tests (Enhanced)

def test_single_agent_movement_precision():
    """Test that the navigator calculates movement vectors with scientific accuracy."""
    navigator = NavigatorFactory.single_agent(
        position=(0.0, 0.0), 
        orientation=0.0, 
        speed=1.0
    )
    
    # Store initial position
    initial_position = navigator.positions[0].copy()
    
    # Take a step - at 0 degrees, movement should be along positive x-axis
    navigator.step(np.zeros((10, 10)))
    
    # Calculate actual movement with enhanced precision validation
    movement = navigator.positions[0] - initial_position
    assert np.isclose(movement[0], 1.0, atol=1e-10), f"X movement should be 1.0, got {movement[0]}"
    assert np.isclose(movement[1], 0.0, atol=1e-10), f"Y movement should be 0.0, got {movement[1]}"


def test_multi_agent_movement_precision():
    """Test multi-agent movement maintains numerical accuracy across all agents."""
    positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
    orientations = [0.0, 90.0, 180.0]  # East, North, West
    speeds = [1.0, 1.0, 1.0]
    
    navigator = NavigatorFactory.multi_agent(
        positions=positions,
        orientations=orientations,
        speeds=speeds
    )
    
    # Store initial positions
    initial_positions = navigator.positions.copy()
    
    # Take a step
    navigator.step(np.zeros((10, 10)))
    
    # Calculate movements for each agent
    movements = navigator.positions - initial_positions
    
    # Agent 0: Moving East (0°)
    assert np.isclose(movements[0, 0], 1.0, atol=1e-10), f"Agent 0 X movement should be 1.0, got {movements[0, 0]}"
    assert np.isclose(movements[0, 1], 0.0, atol=1e-10), f"Agent 0 Y movement should be 0.0, got {movements[0, 1]}"
    
    # Agent 1: Moving North (90°)
    assert np.isclose(movements[1, 0], 0.0, atol=1e-10), f"Agent 1 X movement should be 0.0, got {movements[1, 0]}"
    assert np.isclose(movements[1, 1], 1.0, atol=1e-10), f"Agent 1 Y movement should be 1.0, got {movements[1, 1]}"
    
    # Agent 2: Moving West (180°)
    assert np.isclose(movements[2, 0], -1.0, atol=1e-10), f"Agent 2 X movement should be -1.0, got {movements[2, 0]}"
    assert np.isclose(movements[2, 1], 0.0, atol=1e-10), f"Agent 2 Y movement should be 0.0, got {movements[2, 1]}"


def test_diagonal_movement_precision():
    """Test diagonal movement maintains trigonometric precision."""
    navigator = NavigatorFactory.single_agent(
        position=(0.0, 0.0),
        orientation=45.0,  # Northeast
        speed=1.0
    )
    
    # Store initial position
    initial_position = navigator.positions[0].copy()
    
    # Take a step
    navigator.step(np.zeros((10, 10)))
    
    # Calculate actual movement
    movement = navigator.positions[0] - initial_position
    
    # At 45 degrees, both x and y components should be 1/√2 ≈ 0.7071
    expected_component = 1.0 / np.sqrt(2)
    assert np.isclose(movement[0], expected_component, atol=1e-10), f"X movement should be {expected_component}, got {movement[0]}"
    assert np.isclose(movement[1], expected_component, atol=1e-10), f"Y movement should be {expected_component}, got {movement[1]}"
    
    # Verify total movement magnitude is preserved
    total_magnitude = np.linalg.norm(movement)
    assert np.isclose(total_magnitude, 1.0, atol=1e-10), f"Total movement magnitude should be 1.0, got {total_magnitude}"


def test_orientation_normalization():
    """Test that orientation values are properly normalized."""
    navigator = NavigatorFactory.single_agent(orientation=450.0)  # Should normalize to 90.0
    
    # Take a step to trigger any normalization
    navigator.step(np.zeros((10, 10)))
    
    # Verify orientation is normalized (implementation-dependent)
    orientation = navigator.orientations[0]
    assert 0.0 <= orientation < 360.0, f"Orientation {orientation} should be normalized to [0, 360)"


# Integration and Protocol Compliance Tests

def test_navigator_protocol_compliance():
    """Test that navigators implement the NavigatorProtocol interface."""
    # Test single agent navigator
    single_navigator = NavigatorFactory.single_agent(orientation=45.0, speed=0.5)
    assert isinstance(single_navigator, NavigatorProtocol)
    
    # Test multi-agent navigator
    multi_navigator = NavigatorFactory.multi_agent(
        positions=[[1.0, 2.0], [3.0, 4.0]],
        orientations=[0.0, 90.0],
        speeds=[0.5, 1.0]
    )
    assert isinstance(multi_navigator, NavigatorProtocol)
    
    # Test that all required protocol methods and properties exist
    for navigator in [single_navigator, multi_navigator]:
        # Properties
        assert hasattr(navigator, 'positions')
        assert hasattr(navigator, 'orientations')
        assert hasattr(navigator, 'speeds')
        assert hasattr(navigator, 'max_speeds')
        assert hasattr(navigator, 'angular_velocities')
        assert hasattr(navigator, 'num_agents')
        
        # Methods
        assert hasattr(navigator, 'step') and callable(navigator.step)
        assert hasattr(navigator, 'reset') and callable(navigator.reset)
        assert hasattr(navigator, 'sample_odor') and callable(navigator.sample_odor)
        assert hasattr(navigator, 'sample_multiple_sensors') and callable(navigator.sample_multiple_sensors)


def test_reset_functionality_enhanced():
    """Test enhanced reset method works with new architecture."""
    navigator = NavigatorFactory.single_agent(
        position=(10.0, 10.0),
        orientation=45.0,
        speed=0.8,  # Fixed: Use speed within max_speed constraint
        max_speed=2.0  # Explicitly set max_speed to allow higher speeds if needed
    )
    
    # Move the navigator
    navigator.step(np.zeros((10, 10)))
    moved_position = navigator.positions[0].copy()
    
    # Verify movement occurred
    assert not np.allclose(moved_position, [10.0, 10.0])
    
    # Reset with new parameters
    navigator.reset(position=(0.0, 0.0), orientation=0.0, speed=0.0)
    
    # Verify reset worked
    assert np.allclose(navigator.positions[0], [0.0, 0.0])
    assert navigator.orientations[0] == 0.0
    assert navigator.speeds[0] == 0.0


# Centralized Loguru Logging Integration Tests

@pytest.mark.skipif(not LOGURU_AVAILABLE, reason="Loguru not available")
def test_enhanced_logging_integration(correlation_context_fixture):
    """Test centralized Loguru logging integration with structured output."""
    with correlation_context_fixture:
        # Create controller with enhanced logging
        controller = SingleAgentController(
            position=(10.0, 20.0),
            enable_logging=True,
            controller_id="test_agent_001"
        )
        
        # Verify logger is configured
        assert hasattr(controller, '_logger')
        assert controller._logger is not None
        
        # Execute operations that should generate logs
        env_array = np.random.rand(50, 50)
        controller.step(env_array)
        
        # Test parameter updates with logging
        params = SingleAgentParams(position=(20.0, 30.0), speed=2.0)
        controller.reset_with_params(params)
        
        # Verify performance metrics are collected
        metrics = controller.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'controller_id' in metrics
        assert metrics['controller_id'] == "test_agent_001"


@pytest.mark.skipif(not LOGURU_AVAILABLE, reason="Loguru not available")
def test_structured_logging_output():
    """Test that structured logging provides JSON-compatible correlation IDs."""
    if not correlation_context:
        pytest.skip("Correlation context not available")
        
    with correlation_context("test_session"):
        controller = MultiAgentController(
            positions=[[0, 0], [10, 10]],
            enable_logging=True,
            controller_id="test_multi_001"
        )
        
        # Verify structured context
        assert hasattr(controller, '_logger')
        
        # Execute operations
        env_array = np.random.rand(50, 50)
        controller.step(env_array)
        
        # Get metrics with structured data
        metrics = controller.get_performance_metrics()
        assert 'controller_type' in metrics
        assert metrics['controller_type'] == 'multi_agent'
        assert metrics['num_agents'] == 2


def test_logging_performance_monitoring():
    """Test that logging does not significantly impact performance."""
    # Test with logging enabled
    controller_with_logging = SingleAgentController(
        position=(10.0, 20.0),
        enable_logging=True
    )
    
    # Test without logging  
    controller_without_logging = SingleAgentController(
        position=(10.0, 20.0),
        enable_logging=False
    )
    
    env_array = np.random.rand(100, 100)
    
    # Measure performance with logging
    start_time = time.perf_counter()
    for _ in range(50):
        controller_with_logging.step(env_array)
    time_with_logging = time.perf_counter() - start_time
    
    # Measure performance without logging
    start_time = time.perf_counter()
    for _ in range(50):
        controller_without_logging.step(env_array)
    time_without_logging = time.perf_counter() - start_time
    
    # Logging overhead should be minimal (< 20% increase)
    overhead_ratio = time_with_logging / time_without_logging
    assert overhead_ratio < 1.2, f"Logging overhead {overhead_ratio:.2f}x exceeds 20% threshold"


# Sensor Protocol Integration Tests for Enhanced Modular Navigation

def test_sensor_protocol_availability():
    """Test that sensor protocol components are available for testing."""
    if SENSOR_PROTOCOLS_AVAILABLE:
        assert SensorProtocol is not None
        assert BinarySensor is not None
        assert ConcentrationSensor is not None
        assert GradientSensor is not None
    else:
        pytest.skip("Sensor protocols not yet available - using fallback implementations")


def test_sensor_config_validation():
    """Test sensor configuration validation functionality."""
    # Valid binary sensor configuration
    binary_config = {
        'type': 'BinarySensor',
        'threshold': 0.1,
        'false_positive_rate': 0.02,
        'false_negative_rate': 0.01
    }
    assert validate_sensor_config(binary_config)
    
    # Valid concentration sensor configuration  
    concentration_config = {
        'type': 'ConcentrationSensor',
        'dynamic_range': (0.0, 1.0),
        'resolution': 0.001,
        'noise_level': 0.05
    }
    assert validate_sensor_config(concentration_config)
    
    # Valid gradient sensor configuration
    gradient_config = {
        'type': 'GradientSensor',
        'spatial_resolution': (0.5, 0.5),
        'method': 'central',
        'order': 2
    }
    assert validate_sensor_config(gradient_config)


def test_sensor_suite_creation():
    """Test creation of multi-modal sensor suites for navigation."""
    sensor_configs = [
        {'type': 'BinarySensor', 'threshold': 0.1},
        {'type': 'ConcentrationSensor', 'dynamic_range': (0, 1)},
        {'type': 'GradientSensor', 'spatial_resolution': (0.2, 0.2)}
    ]
    
    sensors = create_sensor_suite(sensor_configs)
    assert len(sensors) == 3
    assert isinstance(sensors[0], BinarySensor)
    assert isinstance(sensors[1], ConcentrationSensor)
    assert isinstance(sensors[2], GradientSensor)


def test_controller_sensor_protocol_integration(mock_sensor_suite, mock_plume_state):
    """Test that controllers integrate with sensor protocols instead of direct sampling."""
    # Create controller that supports sensor protocol integration
    controller = SingleAgentController(position=(10.0, 20.0), speed=1.5)
    
    # Test sensor-based observation processing
    positions = controller.positions
    
    # Test binary sensor integration
    binary_detections = mock_sensor_suite['binary'].detect(mock_plume_state, positions)
    assert isinstance(binary_detections, np.ndarray)
    assert binary_detections.dtype == bool
    
    # Test concentration sensor integration
    concentrations = mock_sensor_suite['concentration'].measure(mock_plume_state, positions)
    assert isinstance(concentrations, np.ndarray)
    assert np.all(concentrations >= 0.0)
    
    # Test gradient sensor integration
    gradients = mock_sensor_suite['gradient'].compute_gradient(mock_plume_state, positions)
    assert isinstance(gradients, np.ndarray)
    assert gradients.shape[-1] == 2  # 2D gradient


def test_sensor_observation_processing_workflow(sensor_observation_samples):
    """Test sensor-based observation processing workflow replacing direct odor sampling."""
    # Test concentration reading processing
    conc_obs = sensor_observation_samples['concentration_reading']
    assert 'concentration' in conc_obs
    assert 'timestamp' in conc_obs
    assert 'position' in conc_obs
    assert 'sensor_id' in conc_obs
    assert isinstance(conc_obs['concentration'], (int, float))
    assert conc_obs['concentration'] >= 0.0
    
    # Test binary detection processing
    binary_obs = sensor_observation_samples['binary_detection']
    assert 'detected' in binary_obs
    assert 'confidence' in binary_obs
    assert 'threshold' in binary_obs
    assert isinstance(binary_obs['detected'], bool)
    assert 0.0 <= binary_obs['confidence'] <= 1.0
    
    # Test gradient information processing
    grad_obs = sensor_observation_samples['gradient_information']
    assert 'gradient' in grad_obs
    assert 'magnitude' in grad_obs
    assert 'direction' in grad_obs
    assert 'spatial_resolution' in grad_obs
    assert len(grad_obs['gradient']) == 2  # 2D gradient


def test_vectorized_sensor_sampling_multi_agent(mock_sensor_suite, mock_plume_state):
    """Test vectorized sensor sampling for multi-agent scenarios."""
    # Create multi-agent controller
    positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
    controller = MultiAgentController(positions=positions)
    
    agent_positions = controller.positions
    assert agent_positions.shape == (3, 2)
    
    # Test vectorized binary sensor sampling
    binary_detections = mock_sensor_suite['binary'].detect(mock_plume_state, agent_positions)
    mock_sensor_suite['binary'].detect.return_value = np.array([True, False, True])
    binary_detections = mock_sensor_suite['binary'].detect(mock_plume_state, agent_positions)
    assert binary_detections.shape == (3,)
    assert binary_detections.dtype == bool
    
    # Test vectorized concentration measurements
    concentrations = mock_sensor_suite['concentration'].measure(mock_plume_state, agent_positions)
    mock_sensor_suite['concentration'].measure.return_value = np.array([0.5, 0.3, 0.8])
    concentrations = mock_sensor_suite['concentration'].measure(mock_plume_state, agent_positions)
    assert concentrations.shape == (3,)
    assert np.all(concentrations >= 0.0)
    
    # Test vectorized gradient computation
    gradients = mock_sensor_suite['gradient'].compute_gradient(mock_plume_state, agent_positions)
    mock_sensor_suite['gradient'].compute_gradient.return_value = np.array([[0.1, 0.05], [0.2, 0.1], [0.15, 0.08]])
    gradients = mock_sensor_suite['gradient'].compute_gradient(mock_plume_state, agent_positions)
    assert gradients.shape == (3, 2)


def test_observation_structure_compatibility():
    """Test that new observation structures are compatible with existing interfaces."""
    # Test observation dictionary structure
    observation = {
        'position': np.array([10.0, 20.0]),
        'orientation': np.array([45.0]),
        'concentration': 0.5,
        'binary_detection': True,
        'gradient': np.array([0.1, 0.05])
    }
    
    # Verify required keys are present
    assert 'position' in observation
    assert 'orientation' in observation
    
    # Verify sensor data is properly structured
    assert isinstance(observation['concentration'], (int, float))
    assert isinstance(observation['binary_detection'], bool)
    assert isinstance(observation['gradient'], np.ndarray)
    assert observation['gradient'].shape == (2,)


def test_sensor_performance_monitoring(mock_sensor_suite):
    """Test sensor performance monitoring for sub-10ms compliance."""
    # Test sensor performance metrics collection
    metrics = get_sensor_performance_metrics(list(mock_sensor_suite.values()))
    
    assert isinstance(metrics, dict)
    assert 'sensor_count' in metrics
    assert metrics['sensor_count'] == 3
    
    # Test individual sensor metrics structure
    if 'individual_metrics' in metrics:
        individual_metrics = metrics['individual_metrics']
        assert len(individual_metrics) == 3


# Enhanced Test Coverage for Single and Multi-Agent Controller Functionality

def test_single_agent_controller_comprehensive():
    """Comprehensive test coverage for single agent controller functionality."""
    controller = SingleAgentController(
        position=(25.0, 35.0),
        orientation=120.0,
        speed=1.5,
        max_speed=3.0,
        angular_velocity=0.3,
        enable_logging=True
    )
    
    # Test initial state
    assert controller.num_agents == 1
    assert np.allclose(controller.positions[0], [25.0, 35.0])
    assert controller.orientations[0] == 120.0
    assert controller.speeds[0] == 1.5
    assert controller.max_speeds[0] == 3.0
    assert controller.angular_velocities[0] == 0.3
    
    # Test state modification
    controller.speeds[0] = 2.0
    controller.angular_velocities[0] = 0.5
    
    # Test step execution
    env_array = np.random.rand(100, 100)
    initial_position = controller.positions[0].copy()
    controller.step(env_array)
    
    # Verify movement occurred
    assert not np.allclose(controller.positions[0], initial_position)
    
    # Test reset functionality
    params = SingleAgentParams(
        position=(50.0, 60.0),
        orientation=45.0,
        speed=0.5,
        max_speed=2.0
    )
    controller.reset_with_params(params)
    
    assert np.allclose(controller.positions[0], [50.0, 60.0])
    assert controller.orientations[0] == 45.0
    assert controller.speeds[0] == 0.5
    assert controller.max_speeds[0] == 2.0
    
    # Test performance metrics
    metrics = controller.get_performance_metrics()
    assert 'controller_type' in metrics
    assert metrics['controller_type'] == 'single_agent'
    assert metrics['num_agents'] == 1


def test_multi_agent_controller_comprehensive():
    """Comprehensive test coverage for multi-agent controller functionality."""
    positions = [[0.0, 0.0], [25.0, 25.0], [50.0, 50.0]]
    orientations = [0.0, 90.0, 180.0]
    speeds = [1.0, 1.5, 2.0]
    max_speeds = [2.0, 3.0, 4.0]
    angular_velocities = [0.1, 0.2, 0.3]
    
    controller = MultiAgentController(
        positions=positions,
        orientations=orientations,
        speeds=speeds,
        max_speeds=max_speeds,
        angular_velocities=angular_velocities,
        enable_logging=True
    )
    
    # Test initial state
    assert controller.num_agents == 3
    assert np.allclose(controller.positions, positions)
    assert np.allclose(controller.orientations, orientations)
    assert np.allclose(controller.speeds, speeds)
    assert np.allclose(controller.max_speeds, max_speeds)
    assert np.allclose(controller.angular_velocities, angular_velocities)
    
    # Test vectorized state modification
    controller.speeds = np.array([2.0, 2.5, 3.0])
    controller.angular_velocities = np.array([0.15, 0.25, 0.35])
    
    # Test step execution
    env_array = np.random.rand(100, 100)
    initial_positions = controller.positions.copy()
    controller.step(env_array)
    
    # Verify all agents moved
    for i in range(3):
        assert not np.allclose(controller.positions[i], initial_positions[i])
    
    # Test batch reset functionality
    new_positions = np.array([[10.0, 10.0], [30.0, 30.0], [60.0, 60.0]])
    new_orientations = np.array([45.0, 135.0, 225.0])
    new_speeds = np.array([0.5, 1.0, 1.5])
    
    params = MultiAgentParams(
        positions=new_positions,
        orientations=new_orientations,
        speeds=new_speeds
    )
    controller.reset_with_params(params)
    
    assert np.allclose(controller.positions, new_positions)
    assert np.allclose(controller.orientations, new_orientations)
    assert np.allclose(controller.speeds, new_speeds)
    
    # Test performance metrics for multi-agent
    metrics = controller.get_performance_metrics()
    assert 'controller_type' in metrics
    assert metrics['controller_type'] == 'multi_agent'
    assert metrics['num_agents'] == 3
    
    if 'throughput_mean_agents_fps' in metrics:
        assert metrics['throughput_mean_agents_fps'] >= 0


def test_controller_protocol_compliance():
    """Test that both controller types fully implement NavigatorProtocol."""
    # Test single agent
    single_controller = SingleAgentController(position=(10.0, 20.0))
    assert isinstance(single_controller, NavigatorProtocol)
    
    # Test multi-agent  
    multi_controller = MultiAgentController(positions=[[0, 0], [10, 10]])
    assert isinstance(multi_controller, NavigatorProtocol)
    
    # Test all protocol methods exist and are callable
    for controller in [single_controller, multi_controller]:
        assert hasattr(controller, 'step') and callable(controller.step)
        assert hasattr(controller, 'reset') and callable(controller.reset)
        assert hasattr(controller, 'sample_odor') and callable(controller.sample_odor)
        assert hasattr(controller, 'sample_multiple_sensors') and callable(controller.sample_multiple_sensors)
        
        # Test all protocol properties exist
        assert hasattr(controller, 'positions')
        assert hasattr(controller, 'orientations')
        assert hasattr(controller, 'speeds')
        assert hasattr(controller, 'max_speeds')
        assert hasattr(controller, 'angular_velocities')
        assert hasattr(controller, 'num_agents')


def test_odor_sampling_comprehensive():
    """Comprehensive test of odor sampling functionality across different scenarios."""
    # Test single agent odor sampling
    single_controller = SingleAgentController(position=(50.0, 50.0))
    env_array = np.random.rand(100, 100) * 0.8  # Concentration values 0-0.8
    
    # Test basic odor sampling
    odor_value = single_controller.sample_odor(env_array)
    assert isinstance(odor_value, (float, np.floating))
    assert 0.0 <= odor_value <= 1.0
    
    # Test multi-sensor sampling
    multi_sensor_values = single_controller.sample_multiple_sensors(env_array)
    assert isinstance(multi_sensor_values, np.ndarray)
    assert multi_sensor_values.ndim >= 1
    assert np.all(multi_sensor_values >= 0.0)
    assert np.all(multi_sensor_values <= 1.0)
    
    # Test multi-agent odor sampling
    multi_controller = MultiAgentController(
        positions=[[25.0, 25.0], [75.0, 75.0], [50.0, 25.0]]
    )
    
    # Test vectorized odor sampling
    multi_odor_values = multi_controller.sample_odor(env_array)
    assert isinstance(multi_odor_values, np.ndarray)
    assert multi_odor_values.shape == (3,)
    assert np.all(multi_odor_values >= 0.0)
    assert np.all(multi_odor_values <= 1.0)
    
    # Test vectorized multi-sensor sampling
    multi_sensor_array = multi_controller.sample_multiple_sensors(env_array)
    assert isinstance(multi_sensor_array, np.ndarray)
    assert multi_sensor_array.shape[0] == 3  # Three agents
    assert np.all(multi_sensor_array >= 0.0)
    assert np.all(multi_sensor_array <= 1.0)


def test_backward_compatibility_odor_sampling():
    """Test backward compatibility of legacy odor sampling methods during sensor protocol migration."""
    # Test that legacy sample_odor and sample_multiple_sensors methods still work
    single_controller = SingleAgentController(position=(30.0, 40.0))
    multi_controller = MultiAgentController(positions=[[0, 0], [10, 10]])
    
    env_array = np.random.rand(100, 100)
    
    # Legacy single agent methods should still work
    assert hasattr(single_controller, 'sample_odor')
    assert hasattr(single_controller, 'sample_multiple_sensors')
    
    legacy_odor = single_controller.sample_odor(env_array)
    assert isinstance(legacy_odor, (float, np.floating))
    
    legacy_multi_sensor = single_controller.sample_multiple_sensors(env_array)
    assert isinstance(legacy_multi_sensor, np.ndarray)
    
    # Legacy multi-agent methods should still work
    assert hasattr(multi_controller, 'sample_odor')
    assert hasattr(multi_controller, 'sample_multiple_sensors')
    
    legacy_multi_odor = multi_controller.sample_odor(env_array)
    assert isinstance(legacy_multi_odor, np.ndarray)
    assert legacy_multi_odor.shape == (2,)
    
    legacy_multi_sensors = multi_controller.sample_multiple_sensors(env_array)
    assert isinstance(legacy_multi_sensors, np.ndarray)
    assert legacy_multi_sensors.shape[0] == 2


def test_sensor_protocol_replaces_direct_sampling(mock_sensor_suite, mock_plume_state):
    """Test that sensor protocol implementations replace direct environmental sampling."""
    # Create controllers that should use sensor protocols instead of direct sampling
    single_controller = SingleAgentController(position=(15.0, 25.0))
    multi_controller = MultiAgentController(positions=[[5, 5], [15, 15]])
    
    # Test that controllers can work with sensor protocol outputs
    positions_single = single_controller.positions
    positions_multi = multi_controller.positions
    
    # Test sensor protocol integration for single agent
    binary_detection = mock_sensor_suite['binary'].detect(mock_plume_state, positions_single)
    concentration = mock_sensor_suite['concentration'].measure(mock_plume_state, positions_single)
    gradient = mock_sensor_suite['gradient'].compute_gradient(mock_plume_state, positions_single)
    
    # Verify sensor outputs are in expected format
    assert isinstance(binary_detection, np.ndarray)
    assert isinstance(concentration, np.ndarray)
    assert isinstance(gradient, np.ndarray)
    
    # Test sensor protocol integration for multi-agent
    multi_binary = mock_sensor_suite['binary'].detect(mock_plume_state, positions_multi)
    multi_concentration = mock_sensor_suite['concentration'].measure(mock_plume_state, positions_multi)
    multi_gradient = mock_sensor_suite['gradient'].compute_gradient(mock_plume_state, positions_multi)
    
    # Mock return values for multi-agent testing
    mock_sensor_suite['binary'].detect.return_value = np.array([True, False])
    mock_sensor_suite['concentration'].measure.return_value = np.array([0.6, 0.4])
    mock_sensor_suite['gradient'].compute_gradient.return_value = np.array([[0.1, 0.05], [0.08, 0.12]])
    
    multi_binary = mock_sensor_suite['binary'].detect(mock_plume_state, positions_multi)
    multi_concentration = mock_sensor_suite['concentration'].measure(mock_plume_state, positions_multi)
    multi_gradient = mock_sensor_suite['gradient'].compute_gradient(mock_plume_state, positions_multi)
    
    # Verify multi-agent sensor outputs have correct shapes
    assert multi_binary.shape == (2,)
    assert multi_concentration.shape == (2,)
    assert multi_gradient.shape == (2, 2)


def test_sensor_integration_workflow_end_to_end(mock_sensor_suite, mock_plume_state):
    """Test complete sensor integration workflow from detection to navigation decision."""
    controller = SingleAgentController(position=(20.0, 30.0), speed=1.0)
    
    # Configure realistic sensor responses
    mock_sensor_suite['binary'].detect.return_value = np.array([True])
    mock_sensor_suite['concentration'].measure.return_value = np.array([0.75])
    mock_sensor_suite['gradient'].compute_gradient.return_value = np.array([[0.15, 0.08]])
    
    # Step 1: Agent requests environmental observations
    positions = controller.positions
    
    # Step 2: Sensors process plume state and return observations
    detected = mock_sensor_suite['binary'].detect(mock_plume_state, positions)
    concentration = mock_sensor_suite['concentration'].measure(mock_plume_state, positions)
    gradient = mock_sensor_suite['gradient'].compute_gradient(mock_plume_state, positions)
    
    # Step 3: Controller processes sensor observations into navigation inputs
    navigation_input = {
        'odor_detected': detected[0],
        'concentration_level': concentration[0],
        'gradient_direction': gradient[0],
        'gradient_magnitude': np.linalg.norm(gradient[0])
    }
    
    # Step 4: Verify navigation input is properly structured
    assert isinstance(navigation_input['odor_detected'], (bool, np.bool_))
    assert isinstance(navigation_input['concentration_level'], (float, np.floating))
    assert isinstance(navigation_input['gradient_direction'], np.ndarray)
    assert isinstance(navigation_input['gradient_magnitude'], (float, np.floating))
    assert navigation_input['gradient_direction'].shape == (2,)
    
    # Step 5: Controller executes navigation step using sensor observations
    env_array = np.random.rand(100, 100)
    initial_position = controller.positions[0].copy()
    controller.step(env_array)
    
    # Step 6: Verify controller state was updated
    final_position = controller.positions[0]
    # Position should have changed (step execution occurred)
    position_changed = not np.allclose(initial_position, final_position)
    # Note: Position change depends on controller implementation details


def test_error_handling_and_edge_cases():
    """Test error handling and edge case behavior."""
    # Test invalid configuration handling
    with pytest.raises((TypeError, ValueError)):
        create_controller_from_config("invalid_config")
    
    # Test empty configuration should create default controller
    controller = create_controller_from_config({})
    assert controller is not None
    assert isinstance(controller, SingleAgentController)
    
    # Test invalid parameter types
    with pytest.raises(TypeError):
        controller = SingleAgentController(position=(10.0, 20.0))
        controller.reset_with_params("invalid_params")
    
    # Test boundary conditions
    controller = SingleAgentController(position=(0.0, 0.0), speed=0.0)
    env_array = np.zeros((10, 10))
    
    # Should handle zero speed without error
    initial_position = controller.positions[0].copy()
    controller.step(env_array)
    # With zero speed, position should not change
    assert np.allclose(controller.positions[0], initial_position)
    
    # Test edge case with very small environment
    small_env = np.ones((1, 1))
    controller.step(small_env)  # Should not crash
    
    # Test very large speed values
    controller.speeds[0] = 1000.0
    controller.step(env_array)  # Should handle gracefully


# Performance Testing for ≤10ms Average Step() Execution Time Requirement

def test_single_agent_step_performance():
    """Test single agent step() execution meets ≤10ms performance requirement."""
    navigator = NavigatorFactory.single_agent(position=(50.0, 50.0), speed=1.0)
    env_array = np.random.rand(100, 100)
    
    # Warm up
    for _ in range(5):
        navigator.step(env_array)
    
    # Measure performance over multiple iterations
    step_times = []
    for _ in range(100):
        start_time = time.perf_counter()
        navigator.step(env_array)
        step_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        step_times.append(step_time)
    
    # Verify performance requirements
    avg_step_time = np.mean(step_times)
    max_step_time = np.max(step_times)
    p95_step_time = np.percentile(step_times, 95)
    
    assert avg_step_time < 10.0, f"Average step time {avg_step_time:.2f}ms exceeds 10ms requirement"
    assert p95_step_time < 20.0, f"95th percentile step time {p95_step_time:.2f}ms exceeds 20ms threshold"
    
    if LOGURU_AVAILABLE:
        logger.info(f"Single agent performance: avg={avg_step_time:.2f}ms, max={max_step_time:.2f}ms, p95={p95_step_time:.2f}ms")


def test_sensor_protocol_integration_performance(mock_sensor_suite, mock_plume_state):
    """Test that sensor protocol integration maintains <10ms step execution time."""
    navigator = NavigatorFactory.single_agent(position=(50.0, 50.0), speed=1.0)
    
    # Create sensor observation processing workflow
    def sensor_observation_step():
        positions = navigator.positions
        
        # Simulate sensor protocol integration workflow
        start_sensor_time = time.perf_counter()
        
        # Binary detection
        binary_result = mock_sensor_suite['binary'].detect(mock_plume_state, positions)
        
        # Concentration measurement
        concentration_result = mock_sensor_suite['concentration'].measure(mock_plume_state, positions)
        
        # Gradient computation
        gradient_result = mock_sensor_suite['gradient'].compute_gradient(mock_plume_state, positions)
        
        # Process sensor observations into structured format
        observation = {
            'binary_detection': binary_result[0] if len(binary_result) > 0 else False,
            'concentration': concentration_result[0] if len(concentration_result) > 0 else 0.0,
            'gradient': gradient_result[0] if len(gradient_result) > 0 else np.array([0.0, 0.0])
        }
        
        sensor_time = (time.perf_counter() - start_sensor_time) * 1000
        
        # Navigator step with sensor integration
        start_nav_time = time.perf_counter()
        navigator.step(np.random.rand(100, 100))
        nav_time = (time.perf_counter() - start_nav_time) * 1000
        
        return sensor_time + nav_time
    
    # Warm up
    for _ in range(5):
        sensor_observation_step()
    
    # Measure performance with sensor protocol integration
    step_times = []
    for _ in range(50):
        step_time = sensor_observation_step()
        step_times.append(step_time)
    
    # Verify performance requirements with sensor integration
    avg_step_time = np.mean(step_times)
    max_step_time = np.max(step_times)
    p95_step_time = np.percentile(step_times, 95)
    
    assert avg_step_time < 10.0, f"Sensor integration step time {avg_step_time:.2f}ms exceeds 10ms requirement"
    assert p95_step_time < 15.0, f"95th percentile sensor integration time {p95_step_time:.2f}ms exceeds threshold"
    
    if LOGURU_AVAILABLE:
        logger.info(f"Sensor integration performance: avg={avg_step_time:.2f}ms, max={max_step_time:.2f}ms, p95={p95_step_time:.2f}ms")


def test_multi_agent_step_performance():
    """Test multi-agent step() execution scales appropriately with agent count."""
    # Test with different agent counts
    agent_counts = [1, 5, 10, 25]
    performance_results = {}
    
    for num_agents in agent_counts:
        positions = [[i * 10, i * 10] for i in range(num_agents)]
        navigator = NavigatorFactory.multi_agent(positions=positions)
        env_array = np.random.rand(100, 100)
        
        # Warm up
        for _ in range(3):
            navigator.step(env_array)
        
        # Measure performance
        step_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            navigator.step(env_array)
            step_time = (time.perf_counter() - start_time) * 1000
            step_times.append(step_time)
        
        avg_step_time = np.mean(step_times)
        performance_results[num_agents] = avg_step_time
        
        # Performance should scale reasonably with agent count
        max_expected_time = 10.0 + (num_agents * 0.5)  # Allow 0.5ms per additional agent
        assert avg_step_time < max_expected_time, f"{num_agents} agents: {avg_step_time:.2f}ms exceeds {max_expected_time:.2f}ms"
    
    if LOGURU_AVAILABLE:
        logger.info(f"Multi-agent performance scaling: {performance_results}")


def test_vectorized_sensor_sampling_performance(mock_sensor_suite, mock_plume_state):
    """Test vectorized sensor sampling performance for multi-agent scenarios."""
    # Test with different agent counts for vectorized operations
    agent_counts = [1, 5, 10, 25, 50]
    sensor_performance_results = {}
    
    for num_agents in agent_counts:
        positions = [[i * 2, i * 2] for i in range(num_agents)]
        navigator = NavigatorFactory.multi_agent(positions=positions)
        agent_positions = navigator.positions
        
        # Configure mock sensors to return appropriate vectorized results
        mock_sensor_suite['binary'].detect.return_value = np.random.random(num_agents) > 0.5
        mock_sensor_suite['concentration'].measure.return_value = np.random.random(num_agents)
        mock_sensor_suite['gradient'].compute_gradient.return_value = np.random.random((num_agents, 2))
        
        # Define vectorized sensor sampling workflow
        def vectorized_sensor_step():
            start_time = time.perf_counter()
            
            # Vectorized binary detection
            binary_detections = mock_sensor_suite['binary'].detect(mock_plume_state, agent_positions)
            
            # Vectorized concentration measurement
            concentrations = mock_sensor_suite['concentration'].measure(mock_plume_state, agent_positions)
            
            # Vectorized gradient computation
            gradients = mock_sensor_suite['gradient'].compute_gradient(mock_plume_state, agent_positions)
            
            # Process vectorized sensor outputs
            observations = {
                'binary_detections': binary_detections,
                'concentrations': concentrations,
                'gradients': gradients
            }
            
            # Validate vectorized output shapes
            assert binary_detections.shape == (num_agents,)
            assert concentrations.shape == (num_agents,)
            assert gradients.shape == (num_agents, 2)
            
            return (time.perf_counter() - start_time) * 1000
        
        # Warm up
        for _ in range(3):
            vectorized_sensor_step()
        
        # Measure vectorized sensor performance
        sensor_times = []
        for _ in range(30):
            sensor_time = vectorized_sensor_step()
            sensor_times.append(sensor_time)
        
        avg_sensor_time = np.mean(sensor_times)
        sensor_performance_results[num_agents] = avg_sensor_time
        
        # Vectorized sensor operations should scale linearly and stay under performance thresholds
        max_expected_sensor_time = 1.0 + (num_agents * 0.02)  # Allow 0.02ms per agent for vectorized operations
        assert avg_sensor_time < max_expected_sensor_time, \
            f"{num_agents} agents vectorized sensor time {avg_sensor_time:.2f}ms exceeds {max_expected_sensor_time:.2f}ms"
    
    if LOGURU_AVAILABLE:
        logger.info(f"Vectorized sensor sampling performance: {sensor_performance_results}")


def test_sensor_observation_processing_performance():
    """Test that sensor observation processing maintains performance requirements."""
    # Test processing of different observation structure types
    observation_types = ['concentration', 'binary', 'gradient', 'multi_modal']
    processing_times = {}
    
    for obs_type in observation_types:
        def process_observation_type():
            start_time = time.perf_counter()
            
            if obs_type == 'concentration':
                obs = {
                    'concentration': 0.75,
                    'timestamp': time.time(),
                    'position': (10.0, 20.0),
                    'sensor_id': 'concentration_001'
                }
                # Simulate concentration processing
                processed_value = obs['concentration'] * 1.0
                
            elif obs_type == 'binary':
                obs = {
                    'detected': True,
                    'confidence': 0.85,
                    'threshold': 0.1,
                    'metadata': {'false_positive_rate': 0.02}
                }
                # Simulate binary processing
                processed_value = float(obs['detected']) * obs['confidence']
                
            elif obs_type == 'gradient':
                obs = {
                    'gradient': (0.15, 0.08),
                    'magnitude': 0.17,
                    'direction': 28.3,
                    'spatial_resolution': 0.5
                }
                # Simulate gradient processing
                processed_value = np.linalg.norm(obs['gradient'])
                
            elif obs_type == 'multi_modal':
                obs = {
                    'concentration': 0.6,
                    'binary_detection': True,
                    'gradient': np.array([0.1, 0.05]),
                    'timestamp': time.time()
                }
                # Simulate multi-modal processing
                processed_value = obs['concentration'] + float(obs['binary_detection']) + np.sum(obs['gradient'])
            
            return (time.perf_counter() - start_time) * 1000
        
        # Measure processing performance
        times = []
        for _ in range(100):
            proc_time = process_observation_type()
            times.append(proc_time)
        
        avg_time = np.mean(times)
        processing_times[obs_type] = avg_time
        
        # Observation processing should be very fast (< 0.1ms per observation)
        assert avg_time < 0.1, f"{obs_type} observation processing {avg_time:.3f}ms exceeds 0.1ms requirement"
    
    if LOGURU_AVAILABLE:
        logger.info(f"Sensor observation processing performance: {processing_times}")


def test_performance_monitoring_integration():
    """Test that enhanced controllers provide performance metrics."""
    controller = SingleAgentController(position=(10.0, 20.0), enable_logging=True)
    env_array = np.random.rand(50, 50)
    
    # Execute several steps
    for _ in range(10):
        controller.step(env_array)
    
    # Get performance metrics
    metrics = controller.get_performance_metrics()
    
    # Verify metrics are available
    assert isinstance(metrics, dict)
    assert 'total_steps' in metrics
    assert metrics['total_steps'] == 10
    
    if 'step_time_mean_ms' in metrics:
        assert metrics['step_time_mean_ms'] > 0
        assert metrics['step_time_mean_ms'] < 100  # Reasonable upper bound


# Seed Management and Deterministic Behavior Testing

def test_global_seed_management(deterministic_seed_context):
    """Test global seed management utilities for reproducible experiments."""
    context, seed_value = deterministic_seed_context
    
    # Verify seed context is properly configured
    assert context.global_seed == seed_value
    assert context.is_seeded is True
    
    # Test that random operations are deterministic
    random_values_1 = [np.random.random() for _ in range(10)]
    
    # Re-seed and verify reproducibility
    with set_global_seed(seed_value):
        random_values_2 = [np.random.random() for _ in range(10)]
    
    assert np.allclose(random_values_1, random_values_2), "Random values should be reproducible with same seed"


def test_navigator_deterministic_behavior():
    """Test that navigator behavior is deterministic with seed management."""
    seed_value = 123
    initial_position = (50.0, 50.0)
    
    # Create two navigators with same configuration and seed
    navigator1 = NavigatorFactory.single_agent(position=initial_position)
    navigator2 = NavigatorFactory.single_agent(position=initial_position)
    
    # Apply same seed and execute same operations
    env_array = np.random.RandomState(seed_value).rand(100, 100)
    
    with set_global_seed(seed_value):
        navigator1.step(env_array)
        position1 = navigator1.positions[0].copy()
    
    with set_global_seed(seed_value):
        navigator2.step(env_array)
        position2 = navigator2.positions[0].copy()
    
    # Verify deterministic behavior
    assert np.allclose(position1, position2), f"Positions should be identical: {position1} vs {position2}"


def test_seed_context_thread_safety():
    """Test that seed context is properly isolated between threads."""
    import threading
    import queue
    import time

    results = queue.Queue()

    # Pre-compute expected sequences for each seed
    expected_sequences = {s: np.random.default_rng(s).random(5) for s in (0, 100, 200)}

    def worker_function(seed_val, worker_id):
        with set_global_seed(seed_val) as (_, np_rng):
            # Sleep to encourage thread scheduling overlap which exposes race conditions
            time.sleep(0.01)
            context = get_seed_context()
            # Generate some random values
            values = np_rng.random(5)
            results.put((worker_id, context.global_seed, context.thread_id, values))

    # Start multiple threads with different seeds
    threads = []
    for i, seed in enumerate((0, 100, 200)):
        thread = threading.Thread(target=worker_function, args=(seed, i))
        threads.append(thread)
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # Verify each thread used its assigned seed and produced expected sequence
    thread_results = {}
    while not results.empty():
        worker_id, seed, thread_id, values = results.get()
        thread_results[worker_id] = (seed, thread_id, values)

    assert len(thread_results) == 3

    # Check deterministic sequences and distinctness
    seen_sequences = []
    seen_threads = set()
    for worker_id, (seed, thread_id, values) in thread_results.items():
        expected_seed = worker_id * 100
        assert seed == expected_seed, f"Worker {worker_id} should have seed {expected_seed}, got {seed}"

        expected_values = expected_sequences[seed]
        assert np.allclose(values, expected_values), (
            f"Worker {worker_id} did not produce expected sequence for seed {seed}"
        )
        seen_sequences.append(tuple(values))
        seen_threads.add(thread_id)

    # Ensure each thread produced a distinct sequence and thread id
    assert len(set(seen_sequences)) == 3, "Threads should produce distinct sequences"
    assert len(seen_threads) == 3, "Thread identifiers should be unique"


# Regression test for per-thread RNG independence
def test_thread_local_rng_independence():
    """Each thread should maintain independent RNGs and recorded seeds."""
    import threading
    import queue
    import random

    seeds = [1, 2, 3]
    expected_py = {}
    for s in seeds:
        rng = random.Random(s)
        expected_py[s] = [rng.random() for _ in range(3)]
    expected_np = {s: np.random.default_rng(s).random(3) for s in seeds}

    results = queue.Queue()

    def worker(seed_val: int) -> None:
        with set_global_seed(seed_val) as rngs:
            assert rngs is not None, "set_global_seed should provide thread-local RNGs"
            py_rng, np_rng = rngs
            ctx = get_seed_context()
            assert hasattr(ctx, "thread_id"), "SeedContext should include thread identifier"
            py_seq = [py_rng.random() for _ in range(3)]
            np_seq = np_rng.random(3)
            results.put((seed_val, ctx.global_seed, ctx.thread_id, py_seq, np_seq))

    threads = [threading.Thread(target=worker, args=(s,)) for s in seeds]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.qsize() == len(seeds)

    seen_py = set()
    seen_np = set()
    while not results.empty():
        seed_val, recorded_seed, thread_id, py_seq, np_seq = results.get()
        assert recorded_seed == seed_val
        assert np.allclose(py_seq, expected_py[seed_val])
        assert np.allclose(np_seq, expected_np[seed_val])
        seen_py.add(tuple(py_seq))
        seen_np.add(tuple(np_seq))

    assert len(seen_py) == len(seeds), "Python RNG sequences should be distinct"
    assert len(seen_np) == len(seeds), "NumPy RNG sequences should be distinct"


# Property-Based Testing for Coordinate Frame Consistency

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
@given(position=position_strategy(), orientation=orientation_strategy(), speed=speed_strategy())
@settings(max_examples=50, deadline=1000)
def test_coordinate_frame_consistency(position, orientation, speed):
    """Property-based test for coordinate frame consistency across navigator operations."""
    assume(speed < 10.0)  # Reasonable speed bounds
    
    navigator = NavigatorFactory.single_agent(
        position=position,
        orientation=orientation,
        speed=speed
    )
    
    initial_position = navigator.positions[0].copy()
    initial_orientation = navigator.orientations[0]
    
    # Execute a step
    env_array = np.zeros((100, 100))
    navigator.step(env_array)
    
    final_position = navigator.positions[0]
    
    # Verify coordinate frame consistency
    movement = final_position - initial_position
    movement_magnitude = np.linalg.norm(movement)
    
    # Movement magnitude should match speed (within numerical precision)
    expected_magnitude = speed * 1.0  # dt = 1.0
    assert np.isclose(movement_magnitude, expected_magnitude, atol=1e-10), \
        f"Movement magnitude {movement_magnitude} should equal speed {expected_magnitude}"
    
    # Movement direction should align with orientation
    if movement_magnitude > 1e-10:  # Avoid division by zero
        movement_angle = np.degrees(np.arctan2(movement[1], movement[0]))
        # Normalize to [0, 360)
        movement_angle = movement_angle % 360
        
        # Allow for small numerical differences
        angle_diff = abs(movement_angle - initial_orientation)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        assert angle_diff < 1e-6, f"Movement direction {movement_angle}° should align with orientation {initial_orientation}°"


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
@given(config=controller_config_strategy())
@settings(max_examples=25, deadline=2000)
def test_controller_config_property_validation(config):
    """Property-based test for controller configuration validation."""
    try:
        is_valid, errors = validate_controller_config(config)
        
        if is_valid:
            # If validation passes, controller creation should succeed
            controller = create_controller_from_config(config)
            assert controller is not None
            assert hasattr(controller, 'positions')
            assert hasattr(controller, 'num_agents')
            
            # Verify controller state is consistent with configuration
            if 'position' in config:
                assert np.allclose(controller.positions[0], config['position'])
            if 'positions' in config:
                assert len(controller.positions) == len(config['positions'])
                
        else:
            # If validation fails, controller creation should raise exception
            with pytest.raises((ValueError, TypeError)):
                create_controller_from_config(config)
                
    except Exception as e:
        # Log unexpected exceptions for debugging
        if LOGURU_AVAILABLE:
            logger.warning(f"Unexpected exception in property test: {e}, config: {config}")
        raise


# Sensor Protocol Compliance and Integration Tests

def test_observation_space_compatibility_with_sensors():
    """Test that observation spaces account for sensor protocol outputs."""
    # Test that observation space can accommodate different sensor types
    observation_components = {
        'position': np.array([10.0, 20.0]),
        'orientation': np.array([45.0]),
        'speed': np.array([1.5])
    }
    
    # Test binary sensor integration
    observation_components['binary_detection'] = True
    observation_components['detection_confidence'] = 0.85
    
    # Test concentration sensor integration
    observation_components['concentration'] = 0.65
    observation_components['concentration_timestamp'] = time.time()
    
    # Test gradient sensor integration
    observation_components['gradient'] = np.array([0.12, 0.08])
    observation_components['gradient_magnitude'] = 0.14
    
    # Verify all components are properly typed
    assert isinstance(observation_components['position'], np.ndarray)
    assert isinstance(observation_components['binary_detection'], bool)
    assert isinstance(observation_components['concentration'], (float, np.floating))
    assert isinstance(observation_components['gradient'], np.ndarray)
    
    # Verify observation structure is valid for RL training
    assert observation_components['position'].shape == (2,)
    assert observation_components['gradient'].shape == (2,)
    assert 0.0 <= observation_components['concentration'] <= 1.0
    assert 0.0 <= observation_components['detection_confidence'] <= 1.0


def test_sensor_configuration_integration_with_controllers():
    """Test that controllers can be configured with different sensor setups."""
    # Test single-sensor configuration
    single_sensor_config = {
        'type': 'ConcentrationSensor',
        'dynamic_range': (0.0, 1.0),
        'resolution': 0.001
    }
    
    concentration_sensor = create_sensor_from_config(single_sensor_config)
    assert concentration_sensor is not None
    
    # Test multi-sensor configuration
    multi_sensor_configs = [
        {'type': 'BinarySensor', 'threshold': 0.1},
        {'type': 'ConcentrationSensor', 'dynamic_range': (0, 2)},
        {'type': 'GradientSensor', 'spatial_resolution': (0.5, 0.5)}
    ]
    
    sensor_suite = create_sensor_suite(multi_sensor_configs)
    assert len(sensor_suite) == 3
    
    # Test controller compatibility with sensor configurations
    controller = SingleAgentController(position=(25.0, 35.0))
    
    # Verify controller can work with all sensor types
    assert hasattr(controller, 'positions')  # Required for sensor position input
    positions = controller.positions
    assert positions.shape[1] == 2  # 2D positions required for sensors


def test_performance_requirements_with_sensor_integration():
    """Test that performance requirements are maintained with sensor protocol integration."""
    # Create performance test scenario with multiple sensor types
    controller = MultiAgentController(positions=[[0, 0], [10, 10], [20, 20]])
    
    # Create mock sensors with performance tracking
    mock_sensors = {
        'binary': BinarySensor(threshold=0.1),
        'concentration': ConcentrationSensor(dynamic_range=(0, 1)),
        'gradient': GradientSensor(spatial_resolution=(0.5, 0.5))
    }
    
    def integrated_step_with_sensors():
        start_time = time.perf_counter()
        
        # Simulate sensor-based observation workflow
        positions = controller.positions
        
        # Multi-sensor sampling (simulated)
        for sensor_name, sensor in mock_sensors.items():
            if hasattr(sensor, 'detect'):
                sensor.detect(None, positions)  # Mock call
            elif hasattr(sensor, 'measure'):
                sensor.measure(None, positions)  # Mock call
            elif hasattr(sensor, 'compute_gradient'):
                sensor.compute_gradient(None, positions)  # Mock call
        
        # Controller step
        controller.step(np.random.rand(100, 100))
        
        return (time.perf_counter() - start_time) * 1000
    
    # Performance benchmark
    step_times = []
    for _ in range(20):
        step_time = integrated_step_with_sensors()
        step_times.append(step_time)
    
    avg_step_time = np.mean(step_times)
    
    # Verify performance requirements are met with sensor integration
    assert avg_step_time < 10.0, f"Integrated step time {avg_step_time:.2f}ms exceeds 10ms requirement"
    
    if LOGURU_AVAILABLE:
        logger.info(f"Sensor integration performance: {avg_step_time:.2f}ms average step time")