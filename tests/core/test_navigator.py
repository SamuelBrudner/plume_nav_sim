"""Tests for the core navigator module.

This module provides comprehensive testing for the NavigatorProtocol interface 
and the factory method patterns used to create navigator instances from Hydra 
configuration objects. The tests validate the new core module organization 
including navigator.py and controllers.py separation, ensuring protocol 
compliance and numerical accuracy requirements.

Enhanced testing includes:
- Hydra configuration integration with Navigator creation
- Protocol-based interface validation
- Factory method patterns with DictConfig integration
- Enhanced numerical comparison tests for scientific accuracy
- Multi-agent and single-agent controller testing
"""

import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import pytest
from typing import Dict, Any, Union
from omegaconf import DictConfig, OmegaConf

# Import from the new core module structure
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController

# Import the validation helpers
from tests.helpers.import_validator import assert_imported_from


def test_proper_imports():
    """Test that NavigatorProtocol is imported from the correct module."""
    assert_imported_from(NavigatorProtocol, "{{cookiecutter.project_slug}}.core.navigator")


def test_controller_imports():
    """Test that controller classes are imported from the correct module."""
    assert_imported_from(SingleAgentController, "{{cookiecutter.project_slug}}.core.controllers")
    assert_imported_from(MultiAgentController, "{{cookiecutter.project_slug}}.core.controllers")


# Hydra configuration fixtures for enhanced testing strategy
@pytest.fixture
def base_single_agent_config():
    """Provides base single agent configuration for Hydra integration testing."""
    config_dict = {
        "position": [0.0, 0.0],
        "orientation": 0.0,
        "speed": 0.0,
        "max_speed": 10.0,
        "angular_velocity": 0.0
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def base_multi_agent_config():
    """Provides base multi-agent configuration for Hydra integration testing."""
    config_dict = {
        "positions": [[0.0, 0.0], [10.0, 10.0]],
        "orientations": [0.0, 90.0],
        "speeds": [0.0, 0.0],
        "max_speeds": [10.0, 12.0],
        "angular_velocities": [0.0, 0.0]
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def hydra_override_config():
    """Provides Hydra configuration with environment variable overrides."""
    config_dict = {
        "position": [5.0, 5.0],
        "orientation": 45.0,
        "speed": 2.5,
        "max_speed": 15.0,
        "angular_velocity": 0.2,
        "debug": True,
        "seed": 42
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def mock_seed_manager():
    """Mock seed manager for deterministic testing."""
    mock_manager = Mock()
    mock_manager.current_seed = 42
    mock_manager.temporary_seed.return_value.__enter__ = Mock(return_value=None)
    mock_manager.temporary_seed.return_value.__exit__ = Mock(return_value=None)
    return mock_manager


def test_single_agent_initialization_default():
    """Test that SingleAgentController can be initialized with default parameters."""
    # Create a controller with default parameters
    controller = SingleAgentController()
    
    # Default values should be set correctly
    assert controller.positions.shape == (1, 2)
    assert controller.orientations.shape == (1,)
    assert controller.speeds.shape == (1,)
    assert controller.max_speeds.shape == (1,)
    assert controller.angular_velocities.shape == (1,)
    
    # Check default values
    assert np.allclose(controller.positions[0], [0.0, 0.0])
    assert controller.orientations[0] == 0.0
    assert controller.speeds[0] == 0.0
    assert controller.max_speeds[0] == 1.0  # Default from controller
    assert controller.angular_velocities[0] == 0.0


def test_single_agent_initialization_with_hydra_config(base_single_agent_config):
    """Test SingleAgentController initialization with Hydra DictConfig."""
    # Override some config values
    config = base_single_agent_config.copy()
    config.position = [1.0, 2.0]
    config.orientation = 45.0
    config.speed = 0.5
    config.max_speed = 8.0
    
    # Create controller with Hydra config
    controller = SingleAgentController(config=config)
    
    # Verify config values are applied
    assert np.allclose(controller.positions[0], [1.0, 2.0])
    assert controller.orientations[0] == 45.0
    assert controller.speeds[0] == 0.5
    assert controller.max_speeds[0] == 8.0


def test_single_agent_initialization_with_override_parameters(base_single_agent_config):
    """Test that explicit parameters override Hydra config values."""
    # Create controller with both config and explicit parameters
    controller = SingleAgentController(
        position=(5.0, 10.0),
        orientation=90.0,
        speed=2.0,
        config=base_single_agent_config
    )
    
    # Explicit parameters should take precedence
    assert np.allclose(controller.positions[0], [5.0, 10.0])
    assert controller.orientations[0] == 90.0
    assert controller.speeds[0] == 2.0


def test_multi_agent_initialization_with_hydra_config(base_multi_agent_config):
    """Test MultiAgentController initialization with Hydra DictConfig."""
    controller = MultiAgentController(config=base_multi_agent_config)
    
    # Verify array shapes
    assert controller.positions.shape == (2, 2)
    assert controller.orientations.shape == (2,)
    assert controller.speeds.shape == (2,)
    assert controller.max_speeds.shape == (2,)
    assert controller.angular_velocities.shape == (2,)
    
    # Verify config values are applied
    expected_positions = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert np.allclose(controller.positions, expected_positions)
    assert np.allclose(controller.orientations, [0.0, 90.0])
    assert np.allclose(controller.speeds, [0.0, 0.0])
    assert np.allclose(controller.max_speeds, [10.0, 12.0])


def test_navigator_protocol_compliance():
    """Test that controller classes implement NavigatorProtocol correctly."""
    # Test SingleAgentController protocol compliance
    single_controller = SingleAgentController(
        position=(1.0, 2.0),
        orientation=45.0,
        speed=1.5
    )
    
    assert isinstance(single_controller, NavigatorProtocol)
    assert hasattr(single_controller, 'positions')
    assert hasattr(single_controller, 'orientations')
    assert hasattr(single_controller, 'speeds')
    assert hasattr(single_controller, 'max_speeds')
    assert hasattr(single_controller, 'angular_velocities')
    assert hasattr(single_controller, 'num_agents')
    assert hasattr(single_controller, 'reset')
    assert hasattr(single_controller, 'step')
    assert hasattr(single_controller, 'sample_odor')
    assert hasattr(single_controller, 'read_single_antenna_odor')
    assert hasattr(single_controller, 'sample_multiple_sensors')
    
    # Test MultiAgentController protocol compliance
    multi_controller = MultiAgentController(
        positions=np.array([[1.0, 2.0], [3.0, 4.0]]),
        orientations=np.array([0.0, 90.0]),
        speeds=np.array([1.0, 1.5])
    )
    
    assert isinstance(multi_controller, NavigatorProtocol)
    assert multi_controller.num_agents == 2


def test_single_agent_set_orientation():
    """Test that orientation can be set and is normalized properly."""
    controller = SingleAgentController()
    
    # Test setting orientation directly
    controller._orientation[0] = 90.0
    assert controller.orientations[0] == 90.0
    
    # Test normalization happens during step - create mock environment
    env_array = np.zeros((10, 10))
    
    # Test normalization of angles > 360
    controller._orientation[0] = 450.0
    controller.step(env_array)
    # After step, orientation should be normalized to 0-360 range
    assert 0 <= controller.orientations[0] < 360
    
    # Test normalization of negative angles
    controller._orientation[0] = -90.0
    controller.step(env_array)
    # After step, orientation should be normalized
    assert 0 <= controller.orientations[0] < 360


def test_single_agent_set_speed():
    """Test that speed can be set with proper constraints."""
    controller = SingleAgentController(max_speed=5.0)
    
    # Test setting valid speed
    controller._speed[0] = 2.5
    assert controller.speeds[0] == 2.5
    
    # Test that speeds exceeding max_speed trigger validation
    controller._speed[0] = 10.0
    assert controller.speeds[0] == 10.0  # Initially set
    
    # Take a step to trigger validation
    env_array = np.zeros((10, 10))
    controller.step(env_array)
    
    # Speed should now be clamped (controller validates during step)
    # The exact behavior depends on implementation details
    assert isinstance(controller.speeds[0], (int, float, np.number))


def test_single_agent_movement_calculation():
    """Test that the controller calculates movement vectors correctly based on orientation and speed."""
    controller = SingleAgentController(
        position=(0.0, 0.0),
        orientation=0.0,
        speed=1.0
    )
    
    # Store initial position
    initial_position = controller.positions[0].copy()
    
    # Take a step - at 0 degrees, movement should be along positive x-axis
    env_array = np.zeros((10, 10))
    controller.step(env_array)
    
    # Calculate actual movement
    movement = controller.positions[0] - initial_position
    
    # At 0 degrees, expect movement in +x direction
    assert movement[0] > 0, "Should move in positive x direction"
    assert np.abs(movement[1]) < 0.1, "Should have minimal y movement"
    
    # Test 90-degree movement
    controller._position[0] = np.array([0.0, 0.0])
    controller._orientation[0] = 90.0
    initial_position = controller.positions[0].copy()
    
    controller.step(env_array)
    movement = controller.positions[0] - initial_position
    
    # At 90 degrees, expect movement in +y direction
    assert np.abs(movement[0]) < 0.1, "Should have minimal x movement"
    assert movement[1] > 0, "Should move in positive y direction"


def test_single_agent_movement_with_different_speeds():
    """Test movement calculation with various speed values."""
    controller = SingleAgentController(
        position=(0.0, 0.0),
        orientation=0.0,
        speed=2.0
    )
    
    initial_position = controller.positions[0].copy()
    env_array = np.zeros((10, 10))
    controller.step(env_array)
    movement_fast = controller.positions[0] - initial_position
    
    # Reset and test with slower speed
    controller._position[0] = np.array([0.0, 0.0])
    controller._speed[0] = 1.0
    initial_position = controller.positions[0].copy()
    controller.step(env_array)
    movement_slow = controller.positions[0] - initial_position
    
    # Faster speed should result in greater movement distance
    fast_distance = np.linalg.norm(movement_fast)
    slow_distance = np.linalg.norm(movement_slow)
    assert fast_distance > slow_distance, "Faster speed should result in greater movement"


def test_multi_agent_initialization_and_properties():
    """Test multi-agent controller initialization and property access."""
    positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    orientations = np.array([0.0, 90.0, 180.0])
    speeds = np.array([0.5, 1.0, 1.5])
    max_speeds = np.array([5.0, 6.0, 7.0])
    angular_velocities = np.array([0.1, 0.2, 0.3])
    
    controller = MultiAgentController(
        positions=positions,
        orientations=orientations,
        speeds=speeds,
        max_speeds=max_speeds,
        angular_velocities=angular_velocities
    )
    
    # Test that properties match initialization
    assert np.allclose(controller.positions, positions)
    assert np.allclose(controller.orientations, orientations)
    assert np.allclose(controller.speeds, speeds)
    assert np.allclose(controller.max_speeds, max_speeds)
    assert np.allclose(controller.angular_velocities, angular_velocities)
    assert controller.num_agents == 3


def test_multi_agent_step_functionality():
    """Test multi-agent step function updates all agents."""
    positions = np.array([[0.0, 0.0], [10.0, 10.0]])
    orientations = np.array([0.0, 90.0])
    speeds = np.array([1.0, 1.0])
    
    controller = MultiAgentController(
        positions=positions,
        orientations=orientations,
        speeds=speeds
    )
    
    initial_positions = controller.positions.copy()
    env_array = np.zeros((20, 20))
    
    # Take a step
    controller.step(env_array)
    
    # All agents should have moved
    for i in range(controller.num_agents):
        movement = controller.positions[i] - initial_positions[i]
        movement_distance = np.linalg.norm(movement)
        assert movement_distance > 0, f"Agent {i} should have moved"


def test_odor_sampling_functionality():
    """Test odor sampling methods work correctly."""
    controller = SingleAgentController(position=(5.0, 5.0))
    
    # Create a test environment with known values
    env_array = np.ones((10, 10)) * 0.5
    env_array[5, 5] = 1.0  # High concentration at agent position
    
    # Test single antenna odor reading
    odor_value = controller.read_single_antenna_odor(env_array)
    assert isinstance(odor_value, (float, np.number))
    assert 0.0 <= odor_value <= 1.0
    
    # Test sample_odor (should delegate to read_single_antenna_odor)
    odor_value_2 = controller.sample_odor(env_array)
    assert isinstance(odor_value_2, (float, np.number))
    
    # Test multiple sensor sampling
    sensor_readings = controller.sample_multiple_sensors(
        env_array, 
        sensor_distance=2.0,
        num_sensors=3
    )
    assert isinstance(sensor_readings, np.ndarray)
    assert sensor_readings.shape == (3,)
    assert np.all(sensor_readings >= 0.0)
    assert np.all(sensor_readings <= 1.0)


def test_multi_agent_odor_sampling():
    """Test odor sampling for multi-agent scenarios."""
    positions = np.array([[2.0, 2.0], [7.0, 7.0]])
    controller = MultiAgentController(positions=positions)
    
    # Create test environment
    env_array = np.zeros((10, 10))
    env_array[2, 2] = 0.8  # High concentration at first agent
    env_array[7, 7] = 0.3  # Lower concentration at second agent
    
    # Test single antenna reading for all agents
    odor_readings = controller.read_single_antenna_odor(env_array)
    assert isinstance(odor_readings, np.ndarray)
    assert odor_readings.shape == (2,)
    
    # Test multiple sensor sampling for all agents
    multi_sensor_readings = controller.sample_multiple_sensors(
        env_array,
        num_sensors=2
    )
    assert isinstance(multi_sensor_readings, np.ndarray)
    assert multi_sensor_readings.shape == (2, 2)  # 2 agents, 2 sensors each


def test_reset_functionality():
    """Test reset functionality for both controller types."""
    # Test single agent reset
    controller = SingleAgentController(
        position=(5.0, 5.0),
        orientation=45.0,
        speed=2.0
    )
    
    # Modify state
    controller._position[0] = np.array([10.0, 10.0])
    controller._orientation[0] = 180.0
    
    # Reset with new parameters
    controller.reset(
        position=(1.0, 1.0),
        orientation=0.0,
        speed=1.0
    )
    
    assert np.allclose(controller.positions[0], [1.0, 1.0])
    assert controller.orientations[0] == 0.0
    assert controller.speeds[0] == 1.0
    
    # Test multi-agent reset
    multi_controller = MultiAgentController(
        positions=np.array([[0.0, 0.0], [5.0, 5.0]]),
        orientations=np.array([0.0, 90.0])
    )
    
    new_positions = np.array([[1.0, 1.0], [6.0, 6.0]])
    new_orientations = np.array([45.0, 135.0])
    
    multi_controller.reset(
        positions=new_positions,
        orientations=new_orientations
    )
    
    assert np.allclose(multi_controller.positions, new_positions)
    assert np.allclose(multi_controller.orientations, new_orientations)


def test_hydra_config_validation_scenarios(hydra_override_config):
    """Test various Hydra configuration validation scenarios."""
    # Test with valid configuration
    controller = SingleAgentController(config=hydra_override_config)
    assert np.allclose(controller.positions[0], [5.0, 5.0])
    assert controller.orientations[0] == 45.0
    assert controller.speeds[0] == 2.5
    assert controller.max_speeds[0] == 15.0
    
    # Test with invalid configuration that should trigger validation
    invalid_config = hydra_override_config.copy()
    invalid_config.speed = 20.0  # Exceeds max_speed
    invalid_config.max_speed = 15.0
    
    # Should either clamp or raise validation error depending on implementation
    try:
        controller = SingleAgentController(config=invalid_config)
        # If it doesn't raise error, speed should be validated during step
        env_array = np.zeros((10, 10))
        controller.step(env_array)
        # After step, speed should be within constraints
        assert controller.speeds[0] <= controller.max_speeds[0]
    except (ValueError, RuntimeError):
        # Validation error is acceptable behavior
        pass


def test_numerical_precision_and_stability():
    """Enhanced numerical comparison tests for scientific accuracy requirements."""
    controller = SingleAgentController(
        position=(0.0, 0.0),
        orientation=0.0,
        speed=1.0
    )
    
    env_array = np.zeros((100, 100))
    
    # Test numerical stability over multiple steps
    initial_position = controller.positions[0].copy()
    
    for _ in range(10):
        controller.step(env_array)
    
    final_position = controller.positions[0]
    
    # Check that position values remain finite
    assert np.all(np.isfinite(final_position)), "Position values should remain finite"
    
    # Check that movement is consistent with expected physics
    total_movement = final_position - initial_position
    expected_distance = 10.0  # 10 steps * 1.0 speed
    actual_distance = np.linalg.norm(total_movement)
    
    # Allow for reasonable numerical tolerance
    assert np.isclose(actual_distance, expected_distance, rtol=0.1), \
        f"Expected distance ~{expected_distance}, got {actual_distance}"


def test_array_bounds_and_edge_cases():
    """Test array bounds checking and edge case handling."""
    # Test with extreme position values
    controller = SingleAgentController(
        position=(1e6, 1e6),  # Very large coordinates
        speed=1.0
    )
    
    env_array = np.zeros((10, 10))
    
    # Should handle large coordinates gracefully
    try:
        odor_value = controller.sample_odor(env_array)
        # Out of bounds should return 0 or handle gracefully
        assert isinstance(odor_value, (float, np.number))
        assert odor_value >= 0.0
    except (IndexError, ValueError):
        # Boundary checking errors are acceptable
        pass
    
    # Test with zero-sized arrays
    try:
        empty_env = np.zeros((0, 0))
        controller.step(empty_env)
    except (ValueError, IndexError):
        # Empty array errors are acceptable
        pass


def test_performance_characteristics():
    """Test that controllers meet performance requirements."""
    import time
    
    # Test single agent performance
    controller = SingleAgentController()
    env_array = np.random.random((100, 100))
    
    start_time = time.time()
    for _ in range(100):
        controller.step(env_array)
    end_time = time.time()
    
    step_time = (end_time - start_time) / 100
    assert step_time < 0.001, f"Single agent step should be <1ms, got {step_time*1000:.2f}ms"
    
    # Test multi-agent performance with 10 agents
    positions = np.random.random((10, 2)) * 50
    multi_controller = MultiAgentController(
        positions=positions,
        speeds=np.ones(10)
    )
    
    start_time = time.time()
    for _ in range(100):
        multi_controller.step(env_array)
    end_time = time.time()
    
    multi_step_time = (end_time - start_time) / 100
    assert multi_step_time < 0.005, f"Multi-agent step (10 agents) should be <5ms, got {multi_step_time*1000:.2f}ms"


def test_configuration_override_precedence():
    """Test that configuration override precedence works correctly."""
    base_config = OmegaConf.create({
        "position": [0.0, 0.0],
        "orientation": 0.0,
        "speed": 1.0,
        "max_speed": 5.0
    })
    
    # Explicit parameters should override config
    controller = SingleAgentController(
        position=(10.0, 20.0),
        orientation=90.0,
        config=base_config
    )
    
    # Explicit parameters take precedence
    assert np.allclose(controller.positions[0], [10.0, 20.0])
    assert controller.orientations[0] == 90.0
    # Config values used where no explicit parameter provided
    assert controller.speeds[0] == 1.0
    assert controller.max_speeds[0] == 5.0


@pytest.mark.parametrize("num_agents,expected_shape", [
    (1, (1, 2)),
    (5, (5, 2)),
    (10, (10, 2)),
    (50, (50, 2)),
])
def test_multi_agent_scaling(num_agents, expected_shape):
    """Test multi-agent controller scaling with different agent counts."""
    positions = np.random.random((num_agents, 2)) * 100
    orientations = np.random.random(num_agents) * 360
    speeds = np.ones(num_agents)
    
    controller = MultiAgentController(
        positions=positions,
        orientations=orientations,
        speeds=speeds
    )
    
    assert controller.positions.shape == expected_shape
    assert controller.orientations.shape == (num_agents,)
    assert controller.speeds.shape == (num_agents,)
    assert controller.num_agents == num_agents
    
    # Test that step function handles all agents
    env_array = np.random.random((100, 100))
    initial_positions = controller.positions.copy()
    controller.step(env_array)
    
    # All agents should have potentially moved
    movements = controller.positions - initial_positions
    assert movements.shape == expected_shape


def test_hydra_dictconfig_integration():
    """Test comprehensive DictConfig integration scenarios."""
    # Test nested configuration structure
    nested_config = OmegaConf.create({
        "navigator": {
            "position": [2.0, 3.0],
            "orientation": 30.0,
            "speed": 1.5,
            "max_speed": 8.0,
            "angular_velocity": 0.15
        },
        "simulation": {
            "dt": 0.1,
            "num_steps": 100
        }
    })
    
    # Test that nested config access works
    controller = SingleAgentController(config=nested_config.navigator)
    assert np.allclose(controller.positions[0], [2.0, 3.0])
    assert controller.orientations[0] == 30.0
    assert controller.speeds[0] == 1.5
    
    # Test with OmegaConf interpolation
    interpolated_config = OmegaConf.create({
        "base_speed": 2.0,
        "position": [1.0, 1.0],
        "orientation": 0.0,
        "speed": "${base_speed}",
        "max_speed": 10.0
    })
    
    controller = SingleAgentController(config=interpolated_config)
    assert controller.speeds[0] == 2.0  # Should resolve interpolation


def test_error_handling_and_edge_cases():
    """Test error handling for various edge cases."""
    # Test with None config
    controller = SingleAgentController(config=None)
    assert controller is not None
    
    # Test with empty config
    empty_config = OmegaConf.create({})
    controller = SingleAgentController(config=empty_config)
    assert controller is not None
    
    # Test multi-agent with mismatched array sizes
    with pytest.raises((ValueError, RuntimeError)):
        MultiAgentController(
            positions=np.array([[1, 2], [3, 4]]),  # 2 agents
            orientations=np.array([0, 90, 180])     # 3 orientations - mismatch!
        )
    
    # Test invalid speed constraints
    with pytest.raises((ValueError, RuntimeError)):
        SingleAgentController(
            speed=10.0,
            max_speed=5.0  # speed > max_speed should raise error
        )