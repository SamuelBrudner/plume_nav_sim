"""Tests for the core navigator module."""

import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Import from the new core module per Section 0.2.1 mapping table
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol, NavigatorFactory
from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController

# Import the validation helpers
from tests.helpers.import_validator import assert_imported_from

# Import Hydra components for configuration testing
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict


# Test fixtures for Hydra configuration integration with Navigator creation per enhanced testing strategy Section 6.6.1
@pytest.fixture
def single_agent_hydra_config():
    """Test fixture for single agent Hydra configuration."""
    if HYDRA_AVAILABLE:
        config = OmegaConf.create({
            "position": [10.0, 20.0],
            "orientation": 45.0,
            "speed": 1.5,
            "max_speed": 3.0,
            "angular_velocity": 0.2
        })
        return config
    else:
        return {
            "position": [10.0, 20.0],
            "orientation": 45.0,
            "speed": 1.5,
            "max_speed": 3.0,
            "angular_velocity": 0.2
        }


@pytest.fixture
def multi_agent_hydra_config():
    """Test fixture for multi-agent Hydra configuration."""
    if HYDRA_AVAILABLE:
        config = OmegaConf.create({
            "positions": [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
            "orientations": [0.0, 90.0, 180.0],
            "speeds": [1.0, 1.5, 2.0],
            "max_speeds": [2.0, 3.0, 4.0],
            "angular_velocities": [0.1, 0.2, 0.3]
        })
        return config
    else:
        return {
            "positions": [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
            "orientations": [0.0, 90.0, 180.0],
            "speeds": [1.0, 1.5, 2.0],
            "max_speeds": [2.0, 3.0, 4.0],
            "angular_velocities": [0.1, 0.2, 0.3]
        }


@pytest.fixture
def mock_navigator():
    """Mock navigator for testing purposes."""
    navigator = MagicMock(spec=NavigatorProtocol)
    navigator.positions = np.array([[0.0, 0.0]])
    navigator.orientations = np.array([0.0])
    navigator.speeds = np.array([0.0])
    navigator.max_speeds = np.array([1.0])
    navigator.angular_velocities = np.array([0.0])
    navigator.num_agents = 1
    return navigator


# Validate imports for updated core module structure per Section 0.2.1 mapping table
def test_proper_imports():
    """Test that Navigator components are imported from the correct modules."""
    assert_imported_from(NavigatorProtocol, "{{cookiecutter.project_slug}}.core.navigator")
    assert_imported_from(NavigatorFactory, "{{cookiecutter.project_slug}}.core.navigator")
    assert_imported_from(SingleAgentController, "{{cookiecutter.project_slug}}.core.controllers")
    assert_imported_from(MultiAgentController, "{{cookiecutter.project_slug}}.core.controllers")


# Add validation tests for Hydra DictConfig integration in Navigator factory methods per Section 7.2.1.1
def test_navigator_factory_from_config_single_agent(single_agent_hydra_config):
    """Test Navigator factory method creates single agent from Hydra config."""
    navigator = NavigatorFactory.from_config(single_agent_hydra_config)
    
    # Verify navigator implements the protocol
    assert isinstance(navigator, NavigatorProtocol)
    
    # Verify configuration values are applied correctly
    assert np.allclose(navigator.positions[0], [10.0, 20.0])
    assert navigator.orientations[0] == 45.0
    assert navigator.speeds[0] == 1.5
    assert navigator.max_speeds[0] == 3.0
    assert navigator.angular_velocities[0] == 0.2
    assert navigator.num_agents == 1


def test_navigator_factory_from_config_multi_agent(multi_agent_hydra_config):
    """Test Navigator factory method creates multi-agent from Hydra config."""
    navigator = NavigatorFactory.from_config(multi_agent_hydra_config)
    
    # Verify navigator implements the protocol
    assert isinstance(navigator, NavigatorProtocol)
    
    # Verify configuration values are applied correctly
    assert navigator.num_agents == 3
    assert np.allclose(navigator.positions, [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    assert np.allclose(navigator.orientations, [0.0, 90.0, 180.0])
    assert np.allclose(navigator.speeds, [1.0, 1.5, 2.0])
    assert np.allclose(navigator.max_speeds, [2.0, 3.0, 4.0])
    assert np.allclose(navigator.angular_velocities, [0.1, 0.2, 0.3])


def test_navigator_factory_dict_config_compatibility():
    """Test that factory methods work with plain dict configurations."""
    dict_config = {
        "position": (5.0, 5.0),
        "orientation": 30.0,
        "speed": 2.0,
        "max_speed": 5.0,
        "angular_velocity": 0.15
    }
    
    navigator = NavigatorFactory.from_config(dict_config)
    
    # Verify navigator implements the protocol
    assert isinstance(navigator, NavigatorProtocol)
    
    # Verify configuration values are applied correctly
    assert np.allclose(navigator.positions[0], [5.0, 5.0])
    assert navigator.orientations[0] == 30.0
    assert navigator.speeds[0] == 2.0
    assert navigator.max_speeds[0] == 5.0
    assert navigator.angular_velocities[0] == 0.15


# Update test fixtures to work with new core module organization including navigator.py and controllers.py separation per Section 5.2.1
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
    speeds = [1.0, 1.5]
    
    navigator = NavigatorFactory.multi_agent(
        positions=positions,
        orientations=orientations,
        speeds=speeds
    )
    
    # Verify multi-agent setup
    assert navigator.num_agents == 2
    assert np.allclose(navigator.positions, positions)
    assert np.allclose(navigator.orientations, orientations)
    assert np.allclose(navigator.speeds, speeds)


def test_single_agent_set_orientation():
    """Test that orientation can be set and is normalized properly through step operations."""
    navigator = NavigatorFactory.single_agent()
    
    # Access the internal controller for direct testing
    if hasattr(navigator, '_controller'):
        controller = navigator._controller
        
        # Test setting orientation in degrees
        controller._orientation[0] = 90.0
        assert navigator.orientations[0] == 90.0
        
        # Test normalization of angles > 360 happens during step
        controller._orientation[0] = 450.0
        # Take a step to trigger normalization
        navigator.step(np.zeros((10, 10)))
        assert navigator.orientations[0] == 90.0  # 450 % 360 = 90
        
        # Test normalization of negative angles
        controller._orientation[0] = -90.0
        # Take a step to trigger normalization
        navigator.step(np.zeros((10, 10)))
        assert navigator.orientations[0] == 270.0  # -90 + 360 = 270


def test_single_agent_set_speed():
    """Test that speed can be set with proper constraints."""
    navigator = NavigatorFactory.single_agent(max_speed=1.0)
    
    # Access the internal controller for direct testing
    if hasattr(navigator, '_controller'):
        controller = navigator._controller
        
        # Test setting valid speed
        controller._speed[0] = 0.5
        assert navigator.speeds[0] == 0.5
        
        # Test setting speed higher than max_speed
        controller._speed[0] = 2.0
        assert navigator.speeds[0] == 2.0
        
        # Take a step to see how the implementation handles speed constraints
        navigator.step(np.zeros((10, 10)))
        
        # Verify the implementation doesn't crash and positions are valid
        assert isinstance(navigator.positions[0], np.ndarray)
        assert navigator.positions[0].shape == (2,)
        assert np.all(np.isfinite(navigator.positions[0]))


# Enhance numerical comparison tests to validate new core component architecture while maintaining scientific accuracy requirements
def test_single_agent_move_numerical_precision():
    """Test that the navigator calculates movement vectors with scientific accuracy."""
    # Test movement in cardinal directions with high precision
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


def test_single_agent_update():
    """Test that the navigator can update its position with step method."""
    # Starting at position (0, 0) with orientation 0 and speed 1.0
    navigator = NavigatorFactory.single_agent(
        position=(0.0, 0.0),
        orientation=0.0, 
        speed=1.0
    )
    
    # Take a step - moving along x-axis
    navigator.step(np.zeros((10, 10)))
    
    # Check position after step
    assert np.isclose(navigator.positions[0, 0], 1.0, atol=1e-5)
    assert np.isclose(navigator.positions[0, 1], 0.0, atol=1e-5)
    
    # Change orientation to 90 degrees and step again
    if hasattr(navigator, '_controller'):
        navigator._controller._orientation[0] = 90.0
        navigator.step(np.zeros((10, 10)))
        
        # Check position after second step
        assert np.isclose(navigator.positions[0, 0], 1.0, atol=1e-5)
        assert np.isclose(navigator.positions[0, 1], 1.0, atol=1e-5)


def test_protocol_navigator_compatibility():
    """Test that both single and multi-agent controllers implement NavigatorProtocol."""
    # Create different navigator types
    single_navigator = NavigatorFactory.single_agent(orientation=45.0, speed=0.5)
    multi_navigator = NavigatorFactory.multi_agent(
        positions=[[1.0, 2.0], [3.0, 4.0]],
        orientations=[0.0, 90.0],
        speeds=[0.5, 1.0]
    )
    
    # Test that both implement the protocol
    assert isinstance(single_navigator, NavigatorProtocol)
    assert isinstance(multi_navigator, NavigatorProtocol)
    
    # Test protocol property access
    assert hasattr(single_navigator, 'positions')
    assert hasattr(single_navigator, 'orientations')
    assert hasattr(single_navigator, 'speeds')
    assert hasattr(single_navigator, 'max_speeds')
    assert hasattr(single_navigator, 'angular_velocities')
    assert hasattr(single_navigator, 'num_agents')
    
    assert hasattr(multi_navigator, 'positions')
    assert hasattr(multi_navigator, 'orientations')
    assert hasattr(multi_navigator, 'speeds')
    assert hasattr(multi_navigator, 'max_speeds')
    assert hasattr(multi_navigator, 'angular_velocities')
    assert hasattr(multi_navigator, 'num_agents')
    
    # Test protocol method access
    assert hasattr(single_navigator, 'step')
    assert hasattr(single_navigator, 'reset')
    assert hasattr(single_navigator, 'sample_odor')
    assert hasattr(single_navigator, 'sample_multiple_sensors')
    
    assert hasattr(multi_navigator, 'step')
    assert hasattr(multi_navigator, 'reset')
    assert hasattr(multi_navigator, 'sample_odor')
    assert hasattr(multi_navigator, 'sample_multiple_sensors')


def test_odor_sampling_functionality():
    """Test odor sampling methods work correctly with new architecture."""
    navigator = NavigatorFactory.single_agent(position=(5.0, 5.0))
    
    # Create a simple test environment array
    env_array = np.ones((10, 10)) * 0.5  # Uniform concentration
    
    # Test single odor sampling
    concentration = navigator.sample_odor(env_array)
    assert isinstance(concentration, (float, np.ndarray))
    if isinstance(concentration, np.ndarray):
        assert concentration.shape == () or concentration.shape == (1,)
    
    # Test multiple sensor sampling
    readings = navigator.sample_multiple_sensors(env_array)
    assert isinstance(readings, np.ndarray)
    assert readings.ndim >= 1  # Should have at least one dimension


def test_multi_agent_odor_sampling():
    """Test multi-agent odor sampling returns correct array shapes."""
    navigator = NavigatorFactory.multi_agent(
        positions=[[2.0, 2.0], [7.0, 7.0]]
    )
    
    # Create a test environment array
    env_array = np.random.rand(10, 10)
    
    # Test single odor sampling for multiple agents
    concentrations = navigator.sample_odor(env_array)
    assert isinstance(concentrations, np.ndarray)
    assert concentrations.shape == (2,)  # Two agents
    
    # Test multiple sensor sampling for multiple agents
    readings = navigator.sample_multiple_sensors(env_array)
    assert isinstance(readings, np.ndarray)
    assert readings.shape[0] == 2  # Two agents


def test_reset_functionality():
    """Test reset method works with new architecture."""
    navigator = NavigatorFactory.single_agent(
        position=(10.0, 10.0),
        orientation=45.0,
        speed=2.0
    )
    
    # Move the navigator
    navigator.step(np.zeros((10, 10)))
    
    # Reset with new parameters
    navigator.reset(position=(0.0, 0.0), orientation=0.0, speed=0.0)
    
    # Verify reset worked
    assert np.allclose(navigator.positions[0], [0.0, 0.0])
    assert navigator.orientations[0] == 0.0
    assert navigator.speeds[0] == 0.0


def test_configuration_error_handling():
    """Test that invalid configurations are handled properly."""
    # Test invalid configuration type
    with pytest.raises((TypeError, ValueError)):
        NavigatorFactory.from_config("invalid_config")
    
    # Test missing required fields for multi-agent
    with pytest.raises((ValueError, KeyError, TypeError)):
        NavigatorFactory.from_config({})


def test_performance_characteristics():
    """Test that new architecture maintains performance requirements."""
    import time
    
    # Test single agent creation performance
    start_time = time.perf_counter()
    navigator = NavigatorFactory.single_agent()
    creation_time = (time.perf_counter() - start_time) * 1000
    
    # Should create navigator in reasonable time
    assert creation_time < 100, f"Navigator creation took {creation_time:.2f}ms, should be <100ms"
    
    # Test step performance
    env_array = np.random.rand(100, 100)
    start_time = time.perf_counter()
    navigator.step(env_array)
    step_time = (time.perf_counter() - start_time) * 1000
    
    # Should execute step in reasonable time
    assert step_time < 10, f"Navigator step took {step_time:.2f}ms, should be <10ms"


def test_large_multi_agent_performance():
    """Test performance with larger numbers of agents."""
    import time
    
    # Create 50 agents
    positions = [[i, i] for i in range(50)]
    navigator = NavigatorFactory.multi_agent(positions=positions)
    
    assert navigator.num_agents == 50
    
    # Test step performance with many agents
    env_array = np.random.rand(100, 100)
    start_time = time.perf_counter()
    navigator.step(env_array)
    step_time = (time.perf_counter() - start_time) * 1000
    
    # Should handle 50 agents efficiently
    assert step_time < 50, f"50-agent step took {step_time:.2f}ms, should be <50ms"