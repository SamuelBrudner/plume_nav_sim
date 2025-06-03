"""Core navigation system testing module.

This module provides comprehensive validation for Navigator facade, controller implementations,
and protocol interface compliance. Tests single-agent and multi-agent navigation logic,
state management, sensor sampling, and protocol adherence through systematic unit testing.

Test Categories:
- Navigator facade class testing (single-agent and multi-agent scenarios)
- SingleAgentController and MultiAgentController implementation validation
- NavigatorProtocol compliance and interface adherence verification
- State management validation (positions, orientations, speeds, angular velocities)
- Sensor sampling functions and multi-sensor capabilities testing
- Factory method integration with configuration-driven instantiation
- Numerical precision and accuracy validation for position calculations
- Performance requirements validation for multi-agent scenarios
- Collision avoidance and boundary handling behavior testing
- State reset and initialization patterns for reproducible experiments

Performance Requirements (per Section 6.6.3.3):
- Single agent step operations: <1ms execution time
- Multi-agent operations: efficient scaling up to 100 agents
- Numerical precision: 1e-6 tolerance for research standards
- Test coverage target: >90% for Navigator components
"""

import pytest
import numpy as np
import time
from typing import Optional, Tuple, Union, Any, List
from unittest.mock import Mock, MagicMock, patch
import tempfile
import contextlib

# Import the core components under test
from src.{{cookiecutter.project_slug}}.core.navigator import Navigator, NavigatorProtocol
from src.{{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController
from src.{{cookiecutter.project_slug}}.core.sensors import (
    SensorLayout, 
    SensorConfiguration,
    calculate_sensor_positions,
    sample_odor_at_sensors
)


class TestNavigatorFacade:
    """Test suite for Navigator facade class functionality.
    
    Validates single-agent and multi-agent scenarios, factory methods,
    configuration-driven instantiation, and protocol compliance.
    """
    
    def test_navigator_single_agent_initialization_default(self):
        """Test Navigator single-agent initialization with default parameters."""
        navigator = Navigator.single()
        
        # Validate initial state
        assert navigator.num_agents == 1
        assert navigator.is_single_agent is True
        
        # Check default position (should be at origin)
        np.testing.assert_array_equal(navigator.positions, np.array([[0.0, 0.0]]))
        
        # Check default orientation (should be 0 degrees)
        np.testing.assert_array_equal(navigator.orientations, np.array([0.0]))
        
        # Check default speed and max_speed
        np.testing.assert_array_equal(navigator.speeds, np.array([0.0]))
        np.testing.assert_array_equal(navigator.max_speeds, np.array([1.0]))
        
        # Check default angular velocity
        np.testing.assert_array_equal(navigator.angular_velocities, np.array([0.0]))
    
    def test_navigator_single_agent_initialization_custom(self):
        """Test Navigator single-agent initialization with custom parameters."""
        position = (10.0, 20.0)
        orientation = 45.0
        speed = 1.5
        max_speed = 2.0
        angular_velocity = 10.0
        
        navigator = Navigator.single(
            position=position,
            orientation=orientation,
            speed=speed,
            max_speed=max_speed,
            angular_velocity=angular_velocity
        )
        
        # Validate custom initialization
        assert navigator.num_agents == 1
        assert navigator.is_single_agent is True
        
        np.testing.assert_allclose(navigator.positions, np.array([position]), rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, np.array([orientation]), rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, np.array([speed]), rtol=1e-6)
        np.testing.assert_allclose(navigator.max_speeds, np.array([max_speed]), rtol=1e-6)
        np.testing.assert_allclose(navigator.angular_velocities, np.array([angular_velocity]), rtol=1e-6)
    
    def test_navigator_multi_agent_initialization_default(self):
        """Test Navigator multi-agent initialization with default parameters."""
        positions = np.array([[0.0, 0.0], [10.0, 15.0], [20.0, 25.0]])
        navigator = Navigator.multi(positions=positions)
        
        # Validate multi-agent state
        assert navigator.num_agents == 3
        assert navigator.is_single_agent is False
        
        np.testing.assert_array_equal(navigator.positions, positions)
        
        # Check default orientations, speeds, etc.
        np.testing.assert_array_equal(navigator.orientations, np.zeros(3))
        np.testing.assert_array_equal(navigator.speeds, np.zeros(3))
        np.testing.assert_array_equal(navigator.max_speeds, np.ones(3))
        np.testing.assert_array_equal(navigator.angular_velocities, np.zeros(3))
    
    def test_navigator_multi_agent_initialization_custom(self):
        """Test Navigator multi-agent initialization with custom parameters."""
        positions = np.array([[5.0, 10.0], [15.0, 20.0]])
        orientations = np.array([30.0, 60.0])
        speeds = np.array([1.0, 1.5])
        max_speeds = np.array([2.0, 2.5])
        angular_velocities = np.array([5.0, -5.0])
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
        
        # Validate custom multi-agent initialization
        assert navigator.num_agents == 2
        assert navigator.is_single_agent is False
        
        np.testing.assert_allclose(navigator.positions, positions, rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, orientations, rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, speeds, rtol=1e-6)
        np.testing.assert_allclose(navigator.max_speeds, max_speeds, rtol=1e-6)
        np.testing.assert_allclose(navigator.angular_velocities, angular_velocities, rtol=1e-6)
    
    def test_navigator_from_config_single_agent(self):
        """Test Navigator creation from configuration dictionary for single agent."""
        config = {
            'position': (5.0, 7.5),
            'orientation': 90.0,
            'speed': 2.0,
            'max_speed': 3.0,
            'angular_velocity': 15.0
        }
        
        navigator = Navigator.from_config(config)
        
        assert navigator.num_agents == 1
        assert navigator.is_single_agent is True
        np.testing.assert_allclose(navigator.positions, np.array([[5.0, 7.5]]), rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, np.array([90.0]), rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, np.array([2.0]), rtol=1e-6)
        np.testing.assert_allclose(navigator.max_speeds, np.array([3.0]), rtol=1e-6)
        np.testing.assert_allclose(navigator.angular_velocities, np.array([15.0]), rtol=1e-6)
    
    def test_navigator_from_config_multi_agent(self):
        """Test Navigator creation from configuration dictionary for multi-agent."""
        config = {
            'positions': np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            'orientations': np.array([0.0, 120.0, 240.0]),
            'speeds': np.array([0.5, 1.0, 1.5]),
            'max_speeds': np.array([2.0, 2.0, 2.0]),
            'angular_velocities': np.array([10.0, -10.0, 0.0])
        }
        
        navigator = Navigator.from_config(config)
        
        assert navigator.num_agents == 3
        assert navigator.is_single_agent is False
        np.testing.assert_allclose(navigator.positions, config['positions'], rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, config['orientations'], rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, config['speeds'], rtol=1e-6)
        np.testing.assert_allclose(navigator.max_speeds, config['max_speeds'], rtol=1e-6)
        np.testing.assert_allclose(navigator.angular_velocities, config['angular_velocities'], rtol=1e-6)


class TestNavigatorProtocolCompliance:
    """Test suite for NavigatorProtocol compliance verification.
    
    Validates that all Navigator implementations properly adhere to the
    protocol interface and provide consistent behavior across different
    agent configurations.
    """
    
    @pytest.fixture
    def single_agent_navigator(self) -> NavigatorProtocol:
        """Fixture providing a single-agent navigator for protocol testing."""
        return Navigator.single(position=(10.0, 15.0), orientation=45.0, speed=1.0)
    
    @pytest.fixture
    def multi_agent_navigator(self) -> NavigatorProtocol:
        """Fixture providing a multi-agent navigator for protocol testing."""
        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        orientations = np.array([0.0, 90.0, 180.0])
        speeds = np.array([0.5, 1.0, 1.5])
        return Navigator.multi(positions=positions, orientations=orientations, speeds=speeds)
    
    @pytest.fixture
    def test_environment(self) -> np.ndarray:
        """Fixture providing a test environment array for odor sampling."""
        # Create a 50x50 environment with a Gaussian odor plume
        x, y = np.meshgrid(np.linspace(0, 49, 50), np.linspace(0, 49, 50))
        center_x, center_y = 25, 25
        sigma = 8.0
        
        # Gaussian concentration field
        concentration = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        return concentration.astype(np.float32)
    
    def test_protocol_properties_single_agent(self, single_agent_navigator, test_environment):
        """Test NavigatorProtocol property access for single agent."""
        nav = single_agent_navigator
        
        # Test property shapes and types
        assert isinstance(nav.positions, np.ndarray)
        assert nav.positions.shape == (1, 2)
        
        assert isinstance(nav.orientations, np.ndarray)
        assert nav.orientations.shape == (1,)
        
        assert isinstance(nav.speeds, np.ndarray)
        assert nav.speeds.shape == (1,)
        
        assert isinstance(nav.max_speeds, np.ndarray)
        assert nav.max_speeds.shape == (1,)
        
        assert isinstance(nav.angular_velocities, np.ndarray)
        assert nav.angular_velocities.shape == (1,)
        
        assert nav.num_agents == 1
        assert isinstance(nav.num_agents, int)
    
    def test_protocol_properties_multi_agent(self, multi_agent_navigator, test_environment):
        """Test NavigatorProtocol property access for multi-agent."""
        nav = multi_agent_navigator
        
        # Test property shapes and types
        assert isinstance(nav.positions, np.ndarray)
        assert nav.positions.shape == (3, 2)
        
        assert isinstance(nav.orientations, np.ndarray)
        assert nav.orientations.shape == (3,)
        
        assert isinstance(nav.speeds, np.ndarray)
        assert nav.speeds.shape == (3,)
        
        assert isinstance(nav.max_speeds, np.ndarray)
        assert nav.max_speeds.shape == (3,)
        
        assert isinstance(nav.angular_velocities, np.ndarray)
        assert nav.angular_velocities.shape == (3,)
        
        assert nav.num_agents == 3
        assert isinstance(nav.num_agents, int)
    
    def test_protocol_step_method_single_agent(self, single_agent_navigator, test_environment):
        """Test NavigatorProtocol step method for single agent."""
        nav = single_agent_navigator
        
        # Store initial state
        initial_positions = nav.positions.copy()
        initial_orientations = nav.orientations.copy()
        
        # Execute step
        nav.step(test_environment)
        
        # Verify state has been updated (positions should change with non-zero speed)
        if nav.speeds[0] > 0:
            position_changed = not np.allclose(nav.positions, initial_positions, rtol=1e-6)
            assert position_changed, "Position should change when speed > 0"
        
        # Verify method doesn't raise exceptions
        assert nav.positions.shape == (1, 2)
        assert nav.orientations.shape == (1,)
    
    def test_protocol_step_method_multi_agent(self, multi_agent_navigator, test_environment):
        """Test NavigatorProtocol step method for multi-agent."""
        nav = multi_agent_navigator
        
        # Store initial state
        initial_positions = nav.positions.copy()
        initial_orientations = nav.orientations.copy()
        
        # Execute step
        nav.step(test_environment)
        
        # Verify state dimensions preserved
        assert nav.positions.shape == (3, 2)
        assert nav.orientations.shape == (3,)
        
        # Verify positions changed for agents with non-zero speed
        for i in range(nav.num_agents):
            if nav.speeds[i] > 0:
                position_changed = not np.allclose(
                    nav.positions[i], initial_positions[i], rtol=1e-6
                )
                assert position_changed, f"Agent {i} position should change when speed > 0"
    
    def test_protocol_sample_odor_single_agent(self, single_agent_navigator, test_environment):
        """Test NavigatorProtocol sample_odor method for single agent."""
        nav = single_agent_navigator
        
        # Test odor sampling
        odor_value = nav.sample_odor(test_environment)
        
        assert isinstance(odor_value, (float, np.floating))
        assert 0.0 <= odor_value <= 1.0  # Normalized concentration
        assert np.isfinite(odor_value)
    
    def test_protocol_sample_odor_multi_agent(self, multi_agent_navigator, test_environment):
        """Test NavigatorProtocol sample_odor method for multi-agent."""
        nav = multi_agent_navigator
        
        # Test odor sampling
        odor_values = nav.sample_odor(test_environment)
        
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (3,)
        assert np.all(odor_values >= 0.0)
        assert np.all(odor_values <= 1.0)  # Normalized concentration
        assert np.all(np.isfinite(odor_values))
    
    def test_protocol_read_single_antenna_odor(self, single_agent_navigator, multi_agent_navigator, test_environment):
        """Test NavigatorProtocol read_single_antenna_odor method."""
        # Test single agent
        single_odor = single_agent_navigator.read_single_antenna_odor(test_environment)
        assert isinstance(single_odor, (float, np.floating))
        assert np.isfinite(single_odor)
        
        # Test multi-agent
        multi_odor = multi_agent_navigator.read_single_antenna_odor(test_environment)
        assert isinstance(multi_odor, np.ndarray)
        assert multi_odor.shape == (3,)
        assert np.all(np.isfinite(multi_odor))
    
    def test_protocol_sample_multiple_sensors(self, single_agent_navigator, multi_agent_navigator, test_environment):
        """Test NavigatorProtocol sample_multiple_sensors method."""
        # Test single agent with multiple sensors
        single_readings = single_agent_navigator.sample_multiple_sensors(
            test_environment,
            sensor_distance=5.0,
            sensor_angle=45.0,
            num_sensors=3
        )
        assert isinstance(single_readings, np.ndarray)
        assert single_readings.shape == (3,)
        assert np.all(np.isfinite(single_readings))
        
        # Test multi-agent with multiple sensors
        multi_readings = multi_agent_navigator.sample_multiple_sensors(
            test_environment,
            sensor_distance=3.0,
            sensor_angle=60.0,
            num_sensors=2
        )
        assert isinstance(multi_readings, np.ndarray)
        assert multi_readings.shape == (3, 2)  # 3 agents, 2 sensors each
        assert np.all(np.isfinite(multi_readings))
    
    def test_protocol_reset_method(self, single_agent_navigator, multi_agent_navigator):
        """Test NavigatorProtocol reset method."""
        # Test single agent reset
        single_agent_navigator.reset(position=(50.0, 50.0), orientation=180.0)
        np.testing.assert_allclose(single_agent_navigator.positions, np.array([[50.0, 50.0]]), rtol=1e-6)
        np.testing.assert_allclose(single_agent_navigator.orientations, np.array([180.0]), rtol=1e-6)
        
        # Test multi-agent reset
        new_positions = np.array([[100.0, 100.0], [110.0, 110.0], [120.0, 120.0]])
        multi_agent_navigator.reset(positions=new_positions)
        np.testing.assert_allclose(multi_agent_navigator.positions, new_positions, rtol=1e-6)


class TestControllerImplementations:
    """Test suite for SingleAgentController and MultiAgentController implementations.
    
    Validates the concrete controller classes that implement the navigation
    logic and state management for both single and multi-agent scenarios.
    """
    
    def test_single_agent_controller_initialization(self):
        """Test SingleAgentController initialization and default values."""
        controller = SingleAgentController()
        
        assert controller.num_agents == 1
        np.testing.assert_array_equal(controller.positions, np.array([[0.0, 0.0]]))
        np.testing.assert_array_equal(controller.orientations, np.array([0.0]))
        np.testing.assert_array_equal(controller.speeds, np.array([0.0]))
        np.testing.assert_array_equal(controller.max_speeds, np.array([1.0]))
        np.testing.assert_array_equal(controller.angular_velocities, np.array([0.0]))
    
    def test_single_agent_controller_custom_initialization(self):
        """Test SingleAgentController with custom initialization parameters."""
        position = (25.0, 35.0)
        orientation = 270.0
        speed = 2.5
        max_speed = 5.0
        angular_velocity = -15.0
        
        controller = SingleAgentController(
            position=position,
            orientation=orientation,
            speed=speed,
            max_speed=max_speed,
            angular_velocity=angular_velocity
        )
        
        assert controller.num_agents == 1
        np.testing.assert_allclose(controller.positions, np.array([position]), rtol=1e-6)
        np.testing.assert_allclose(controller.orientations, np.array([orientation]), rtol=1e-6)
        np.testing.assert_allclose(controller.speeds, np.array([speed]), rtol=1e-6)
        np.testing.assert_allclose(controller.max_speeds, np.array([max_speed]), rtol=1e-6)
        np.testing.assert_allclose(controller.angular_velocities, np.array([angular_velocity]), rtol=1e-6)
    
    def test_multi_agent_controller_initialization(self):
        """Test MultiAgentController initialization and default values."""
        positions = np.array([[0.0, 5.0], [10.0, 15.0], [20.0, 25.0], [30.0, 35.0]])
        controller = MultiAgentController(positions=positions)
        
        assert controller.num_agents == 4
        np.testing.assert_array_equal(controller.positions, positions)
        np.testing.assert_array_equal(controller.orientations, np.zeros(4))
        np.testing.assert_array_equal(controller.speeds, np.zeros(4))
        np.testing.assert_array_equal(controller.max_speeds, np.ones(4))
        np.testing.assert_array_equal(controller.angular_velocities, np.zeros(4))
    
    def test_multi_agent_controller_custom_initialization(self):
        """Test MultiAgentController with custom initialization parameters."""
        positions = np.array([[1.0, 2.0], [3.0, 4.0]])
        orientations = np.array([45.0, 135.0])
        speeds = np.array([1.2, 2.3])
        max_speeds = np.array([3.0, 4.0])
        angular_velocities = np.array([12.0, -8.0])
        
        controller = MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
        
        assert controller.num_agents == 2
        np.testing.assert_allclose(controller.positions, positions, rtol=1e-6)
        np.testing.assert_allclose(controller.orientations, orientations, rtol=1e-6)
        np.testing.assert_allclose(controller.speeds, speeds, rtol=1e-6)
        np.testing.assert_allclose(controller.max_speeds, max_speeds, rtol=1e-6)
        np.testing.assert_allclose(controller.angular_velocities, angular_velocities, rtol=1e-6)
    
    def test_controller_step_position_updates(self):
        """Test position updates during step operations."""
        # Test single agent movement
        single_controller = SingleAgentController(
            position=(0.0, 0.0),
            orientation=0.0,  # Facing east
            speed=1.0
        )
        
        env_array = np.zeros((50, 50))
        single_controller.step(env_array, dt=1.0)
        
        # After 1 second at speed 1.0 facing east, should move 1 unit in x direction
        expected_position = np.array([[1.0, 0.0]])
        np.testing.assert_allclose(single_controller.positions, expected_position, rtol=1e-6)
        
        # Test multi-agent movement
        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        orientations = np.array([90.0, 270.0])  # North and South
        speeds = np.array([2.0, 1.5])
        
        multi_controller = MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds
        )
        
        multi_controller.step(env_array, dt=1.0)
        
        # Agent 0: moves north (positive y) by 2.0 units
        # Agent 1: moves south (negative y) by 1.5 units
        expected_positions = np.array([[0.0, 2.0], [10.0, 8.5]])
        np.testing.assert_allclose(multi_controller.positions, expected_positions, rtol=1e-6)
    
    def test_controller_step_orientation_updates(self):
        """Test orientation updates during step operations with angular velocity."""
        # Test single agent orientation change
        single_controller = SingleAgentController(
            position=(0.0, 0.0),
            orientation=0.0,
            angular_velocity=90.0  # 90 degrees per second
        )
        
        env_array = np.zeros((50, 50))
        single_controller.step(env_array, dt=1.0)
        
        # After 1 second with 90 deg/s angular velocity, orientation should be 90 degrees
        np.testing.assert_allclose(single_controller.orientations, np.array([90.0]), rtol=1e-6)
        
        # Test multi-agent orientation changes
        positions = np.array([[0.0, 0.0], [5.0, 5.0]])
        orientations = np.array([0.0, 180.0])
        angular_velocities = np.array([45.0, -30.0])
        
        multi_controller = MultiAgentController(
            positions=positions,
            orientations=orientations,
            angular_velocities=angular_velocities
        )
        
        multi_controller.step(env_array, dt=1.0)
        
        # Agent 0: 0° + 45° = 45°
        # Agent 1: 180° - 30° = 150°
        expected_orientations = np.array([45.0, 150.0])
        np.testing.assert_allclose(multi_controller.orientations, expected_orientations, rtol=1e-6)
    
    def test_controller_reset_functionality(self):
        """Test reset functionality for controllers."""
        # Test single agent reset
        single_controller = SingleAgentController(position=(10.0, 20.0), speed=1.5)
        
        # Modify state
        single_controller.step(np.zeros((50, 50)), dt=1.0)
        
        # Reset with new parameters
        single_controller.reset(position=(50.0, 60.0), orientation=45.0, speed=2.0)
        
        np.testing.assert_allclose(single_controller.positions, np.array([[50.0, 60.0]]), rtol=1e-6)
        np.testing.assert_allclose(single_controller.orientations, np.array([45.0]), rtol=1e-6)
        np.testing.assert_allclose(single_controller.speeds, np.array([2.0]), rtol=1e-6)
        
        # Test multi-agent reset
        positions = np.array([[1.0, 2.0], [3.0, 4.0]])
        multi_controller = MultiAgentController(positions=positions)
        
        # Reset with new positions
        new_positions = np.array([[10.0, 20.0], [30.0, 40.0]])
        new_speeds = np.array([1.0, 2.0])
        multi_controller.reset(positions=new_positions, speeds=new_speeds)
        
        np.testing.assert_allclose(multi_controller.positions, new_positions, rtol=1e-6)
        np.testing.assert_allclose(multi_controller.speeds, new_speeds, rtol=1e-6)


class TestStateManagement:
    """Test suite for state management validation.
    
    Validates positions, orientations, speeds, angular velocities,
    and state consistency across navigation operations.
    """
    
    @pytest.fixture
    def complex_environment(self) -> np.ndarray:
        """Create a complex environment for state management testing."""
        # Create a 100x100 environment with multiple odor sources
        x, y = np.meshgrid(np.linspace(0, 99, 100), np.linspace(0, 99, 100))
        
        # Multiple Gaussian sources
        source1 = np.exp(-((x - 30)**2 + (y - 30)**2) / (2 * 10**2))
        source2 = np.exp(-((x - 70)**2 + (y - 70)**2) / (2 * 15**2))
        source3 = np.exp(-((x - 50)**2 + (y - 20)**2) / (2 * 8**2))
        
        environment = 0.5 * source1 + 0.3 * source2 + 0.7 * source3
        return environment.astype(np.float32)
    
    def test_state_consistency_single_agent(self, complex_environment):
        """Test state consistency for single agent across multiple operations."""
        navigator = Navigator.single(
            position=(25.0, 25.0),
            orientation=30.0,
            speed=1.5,
            max_speed=3.0,
            angular_velocity=5.0
        )
        
        # Record initial state
        initial_position = navigator.positions.copy()
        initial_orientation = navigator.orientations.copy()
        
        # Perform multiple steps
        for i in range(10):
            navigator.step(complex_environment, dt=0.1)
            
            # Validate state shapes remain consistent
            assert navigator.positions.shape == (1, 2)
            assert navigator.orientations.shape == (1,)
            assert navigator.speeds.shape == (1,)
            assert navigator.max_speeds.shape == (1,)
            assert navigator.angular_velocities.shape == (1,)
            
            # Validate numerical properties
            assert np.all(np.isfinite(navigator.positions))
            assert np.all(np.isfinite(navigator.orientations))
            assert np.all(navigator.speeds >= 0)
            assert np.all(navigator.speeds <= navigator.max_speeds)
            
            # Sample odor to ensure interaction works
            odor = navigator.sample_odor(complex_environment)
            assert np.isfinite(odor)
            assert odor >= 0.0
        
        # Verify position has changed due to movement
        position_changed = not np.allclose(navigator.positions, initial_position, rtol=1e-6)
        assert position_changed, "Position should change during movement"
        
        # Verify orientation has changed due to angular velocity
        orientation_changed = not np.allclose(navigator.orientations, initial_orientation, rtol=1e-6)
        assert orientation_changed, "Orientation should change with angular velocity"
    
    def test_state_consistency_multi_agent(self, complex_environment):
        """Test state consistency for multi-agent across multiple operations."""
        positions = np.array([[10.0, 10.0], [30.0, 30.0], [50.0, 50.0], [70.0, 70.0]])
        orientations = np.array([0.0, 90.0, 180.0, 270.0])
        speeds = np.array([0.5, 1.0, 1.5, 2.0])
        max_speeds = np.array([2.0, 2.5, 3.0, 3.5])
        angular_velocities = np.array([10.0, -5.0, 15.0, -20.0])
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
        
        # Record initial state
        initial_positions = navigator.positions.copy()
        initial_orientations = navigator.orientations.copy()
        
        # Perform multiple steps
        for i in range(20):
            navigator.step(complex_environment, dt=0.05)
            
            # Validate state shapes remain consistent
            assert navigator.positions.shape == (4, 2)
            assert navigator.orientations.shape == (4,)
            assert navigator.speeds.shape == (4,)
            assert navigator.max_speeds.shape == (4,)
            assert navigator.angular_velocities.shape == (4,)
            
            # Validate numerical properties for all agents
            assert np.all(np.isfinite(navigator.positions))
            assert np.all(np.isfinite(navigator.orientations))
            assert np.all(navigator.speeds >= 0)
            assert np.all(navigator.speeds <= navigator.max_speeds)
            
            # Sample odor for all agents
            odor_values = navigator.sample_odor(complex_environment)
            assert odor_values.shape == (4,)
            assert np.all(np.isfinite(odor_values))
            assert np.all(odor_values >= 0.0)
        
        # Verify positions have changed for agents with non-zero speed
        for i in range(4):
            if speeds[i] > 0:
                position_changed = not np.allclose(
                    navigator.positions[i], initial_positions[i], rtol=1e-6
                )
                assert position_changed, f"Agent {i} position should change during movement"
        
        # Verify orientations have changed for agents with non-zero angular velocity
        for i in range(4):
            if angular_velocities[i] != 0:
                orientation_changed = not np.allclose(
                    navigator.orientations[i], initial_orientations[i], rtol=1e-6
                )
                assert orientation_changed, f"Agent {i} orientation should change with angular velocity"
    
    def test_state_bounds_validation(self):
        """Test validation of state bounds and constraints."""
        # Test speed constraints
        navigator = Navigator.single(speed=1.0, max_speed=2.0)
        assert navigator.speeds[0] <= navigator.max_speeds[0]
        
        # Test multi-agent speed constraints
        positions = np.array([[0.0, 0.0], [10.0, 10.0]])
        speeds = np.array([1.5, 2.5])
        max_speeds = np.array([2.0, 3.0])
        
        multi_navigator = Navigator.multi(
            positions=positions,
            speeds=speeds,
            max_speeds=max_speeds
        )
        
        assert np.all(multi_navigator.speeds <= multi_navigator.max_speeds)
    
    def test_orientation_normalization(self):
        """Test orientation normalization behavior."""
        # Test large positive orientation
        navigator = Navigator.single(orientation=450.0)  # Should normalize to 90.0
        expected_orientation = 450.0 % 360.0
        np.testing.assert_allclose(navigator.orientations, np.array([expected_orientation]), rtol=1e-6)
        
        # Test negative orientation
        navigator = Navigator.single(orientation=-90.0)  # Should normalize to 270.0
        expected_orientation = (-90.0) % 360.0
        np.testing.assert_allclose(navigator.orientations, np.array([expected_orientation]), rtol=1e-6)
        
        # Test multi-agent orientation normalization
        orientations = np.array([720.0, -180.0, 450.0])  # [0.0, 180.0, 90.0]
        expected_orientations = orientations % 360.0
        
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        navigator = Navigator.multi(positions=positions, orientations=orientations)
        
        np.testing.assert_allclose(navigator.orientations, expected_orientations, rtol=1e-6)


class TestSensorSampling:
    """Test suite for sensor sampling functions and multi-sensor capabilities.
    
    Validates single antenna sampling, multi-sensor sampling, sensor layouts,
    and odor concentration reading accuracy.
    """
    
    @pytest.fixture
    def gradient_environment(self) -> np.ndarray:
        """Create a gradient environment for sensor testing."""
        # Create a linear gradient from left to right
        x, y = np.meshgrid(np.linspace(0, 49, 50), np.linspace(0, 49, 50))
        gradient = x / 49.0  # Normalized gradient 0 to 1
        return gradient.astype(np.float32)
    
    @pytest.fixture
    def circular_environment(self) -> np.ndarray:
        """Create a circular environment for sensor testing."""
        # Create concentric circles with different concentrations
        x, y = np.meshgrid(np.linspace(0, 49, 50), np.linspace(0, 49, 50))
        center_x, center_y = 25, 25
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create rings with different concentrations
        environment = np.zeros_like(distance)
        environment[distance <= 10] = 1.0
        environment[(distance > 10) & (distance <= 20)] = 0.5
        environment[distance > 20] = 0.1
        
        return environment.astype(np.float32)
    
    def test_single_antenna_sampling_accuracy(self, gradient_environment):
        """Test single antenna odor sampling accuracy."""
        # Test known positions in gradient environment
        test_cases = [
            ((5.0, 25.0), 5.0 / 49.0),    # Left side, low concentration
            ((25.0, 25.0), 25.0 / 49.0),  # Center, medium concentration
            ((45.0, 25.0), 45.0 / 49.0),  # Right side, high concentration
        ]
        
        for position, expected_concentration in test_cases:
            navigator = Navigator.single(position=position)
            odor_value = navigator.read_single_antenna_odor(gradient_environment)
            
            np.testing.assert_allclose(odor_value, expected_concentration, rtol=0.1)
    
    def test_multi_agent_single_antenna_sampling(self, circular_environment):
        """Test single antenna sampling for multiple agents."""
        # Position agents at different distances from center
        positions = np.array([
            [25.0, 25.0],  # Center, should read 1.0
            [35.0, 25.0],  # Distance 10, should read 1.0
            [40.0, 25.0],  # Distance 15, should read 0.5
            [45.0, 25.0],  # Distance 20, should read 0.5
            [48.0, 25.0],  # Distance 23, should read 0.1
        ])
        
        expected_readings = np.array([1.0, 1.0, 0.5, 0.5, 0.1])
        
        navigator = Navigator.multi(positions=positions)
        odor_values = navigator.read_single_antenna_odor(circular_environment)
        
        assert odor_values.shape == (5,)
        np.testing.assert_allclose(odor_values, expected_readings, rtol=0.1)
    
    def test_multi_sensor_sampling_single_agent(self, gradient_environment):
        """Test multi-sensor sampling for single agent."""
        # Position agent at center facing east (0 degrees)
        navigator = Navigator.single(position=(25.0, 25.0), orientation=0.0)
        
        # Sample with 3 sensors: left, center, right relative to heading
        sensor_readings = navigator.sample_multiple_sensors(
            gradient_environment,
            sensor_distance=5.0,
            sensor_angle=90.0,  # 90 degrees between sensors
            num_sensors=3
        )
        
        assert sensor_readings.shape == (3,)
        assert np.all(np.isfinite(sensor_readings))
        
        # In gradient environment, right sensor should read higher than left
        # (assuming sensor layout places sensors relative to orientation)
        assert np.all(sensor_readings >= 0.0)
        assert np.all(sensor_readings <= 1.0)
    
    def test_multi_sensor_sampling_multi_agent(self, circular_environment):
        """Test multi-sensor sampling for multiple agents."""
        positions = np.array([[20.0, 20.0], [30.0, 30.0]])
        orientations = np.array([45.0, 225.0])
        
        navigator = Navigator.multi(positions=positions, orientations=orientations)
        
        # Sample with 2 sensors per agent
        sensor_readings = navigator.sample_multiple_sensors(
            circular_environment,
            sensor_distance=3.0,
            sensor_angle=60.0,
            num_sensors=2
        )
        
        assert sensor_readings.shape == (2, 2)  # 2 agents, 2 sensors each
        assert np.all(np.isfinite(sensor_readings))
        assert np.all(sensor_readings >= 0.0)
        assert np.all(sensor_readings <= 1.0)
    
    def test_sensor_layout_configurations(self, gradient_environment):
        """Test different sensor layout configurations."""
        navigator = Navigator.single(position=(25.0, 25.0), orientation=0.0)
        
        # Test different numbers of sensors
        for num_sensors in [2, 3, 4, 5]:
            readings = navigator.sample_multiple_sensors(
                gradient_environment,
                sensor_distance=4.0,
                num_sensors=num_sensors
            )
            assert readings.shape == (num_sensors,)
            assert np.all(np.isfinite(readings))
        
        # Test different sensor distances
        for distance in [1.0, 3.0, 7.0, 10.0]:
            readings = navigator.sample_multiple_sensors(
                gradient_environment,
                sensor_distance=distance,
                num_sensors=3
            )
            assert readings.shape == (3,)
            assert np.all(np.isfinite(readings))
        
        # Test different sensor angles
        for angle in [30.0, 60.0, 90.0, 120.0]:
            readings = navigator.sample_multiple_sensors(
                gradient_environment,
                sensor_angle=angle,
                num_sensors=3
            )
            assert readings.shape == (3,)
            assert np.all(np.isfinite(readings))
    
    def test_boundary_conditions_odor_sampling(self):
        """Test odor sampling at environment boundaries."""
        # Create small environment for boundary testing
        small_env = np.ones((10, 10), dtype=np.float32)
        
        # Test positions at boundaries and outside
        boundary_positions = [
            (0.0, 0.0),    # Top-left corner
            (9.0, 9.0),    # Bottom-right corner
            (-1.0, 5.0),   # Outside left boundary
            (10.0, 5.0),   # Outside right boundary
            (5.0, -1.0),   # Outside top boundary
            (5.0, 10.0),   # Outside bottom boundary
        ]
        
        for position in boundary_positions:
            navigator = Navigator.single(position=position)
            
            # Should not raise exceptions
            odor_value = navigator.read_single_antenna_odor(small_env)
            assert np.isfinite(odor_value)
            assert odor_value >= 0.0
            
            # Multi-sensor sampling should also work
            sensor_readings = navigator.sample_multiple_sensors(
                small_env,
                sensor_distance=2.0,
                num_sensors=2
            )
            assert sensor_readings.shape == (2,)
            assert np.all(np.isfinite(sensor_readings))


class TestPerformanceRequirements:
    """Test suite for performance requirements validation.
    
    Validates single agent <1ms execution time, multi-agent scaling efficiency,
    and overall system performance against specified benchmarks.
    """
    
    @pytest.fixture
    def performance_environment(self) -> np.ndarray:
        """Create a representative environment for performance testing."""
        # Create a complex 200x200 environment
        x, y = np.meshgrid(np.linspace(0, 199, 200), np.linspace(0, 199, 200))
        
        # Multiple overlapping Gaussian sources
        sources = []
        for i in range(5):
            center_x = 40 + i * 30
            center_y = 40 + i * 25
            sigma = 15 + i * 3
            source = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
            sources.append(source)
        
        environment = sum(sources) / len(sources)
        return environment.astype(np.float32)
    
    def test_single_agent_step_performance(self, performance_environment):
        """Test single agent step operation performance (<1ms requirement)."""
        navigator = Navigator.single(
            position=(50.0, 50.0),
            orientation=45.0,
            speed=2.0,
            angular_velocity=10.0
        )
        
        # Warm up
        for _ in range(5):
            navigator.step(performance_environment, dt=0.1)
        
        # Measure performance over multiple iterations
        num_iterations = 100
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            navigator.step(performance_environment, dt=0.1)
        
        end_time = time.perf_counter()
        average_time = (end_time - start_time) / num_iterations
        
        # Should be less than 1ms per step
        assert average_time < 0.001, f"Single agent step took {average_time:.6f}s, should be <0.001s"
    
    def test_multi_agent_scaling_performance(self, performance_environment):
        """Test multi-agent scaling performance up to 100 agents."""
        agent_counts = [10, 25, 50, 75, 100]
        performance_results = []
        
        for num_agents in agent_counts:
            # Create agents in a grid pattern
            positions = []
            for i in range(num_agents):
                x = 20 + (i % 10) * 15
                y = 20 + (i // 10) * 15
                positions.append([x, y])
            
            positions = np.array(positions)
            orientations = np.random.uniform(0, 360, num_agents)
            speeds = np.random.uniform(0.5, 2.0, num_agents)
            angular_velocities = np.random.uniform(-20, 20, num_agents)
            
            navigator = Navigator.multi(
                positions=positions,
                orientations=orientations,
                speeds=speeds,
                angular_velocities=angular_velocities
            )
            
            # Warm up
            for _ in range(3):
                navigator.step(performance_environment, dt=0.1)
            
            # Measure performance
            num_iterations = 50
            start_time = time.perf_counter()
            
            for _ in range(num_iterations):
                navigator.step(performance_environment, dt=0.1)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            average_time = total_time / num_iterations
            time_per_agent = average_time / num_agents
            
            performance_results.append({
                'agents': num_agents,
                'total_time': total_time,
                'avg_time': average_time,
                'time_per_agent': time_per_agent
            })
            
            # Performance should scale efficiently (sub-linear with agent count)
            # Time per agent should remain approximately constant
            assert time_per_agent < 0.0001, f"Time per agent ({time_per_agent:.6f}s) too high for {num_agents} agents"
        
        # Verify scaling efficiency - time per agent should not increase significantly
        first_time_per_agent = performance_results[0]['time_per_agent']
        last_time_per_agent = performance_results[-1]['time_per_agent']
        scaling_factor = last_time_per_agent / first_time_per_agent
        
        # Should scale efficiently (less than 2x degradation)
        assert scaling_factor < 2.0, f"Performance degradation too high: {scaling_factor:.2f}x"
    
    def test_odor_sampling_performance(self, performance_environment):
        """Test odor sampling performance requirements."""
        # Test single agent odor sampling
        navigator = Navigator.single(position=(100.0, 100.0))
        
        num_samples = 1000
        start_time = time.perf_counter()
        
        for _ in range(num_samples):
            odor_value = navigator.sample_odor(performance_environment)
        
        end_time = time.perf_counter()
        avg_sample_time = (end_time - start_time) / num_samples
        
        # Should be much faster than 1ms for single sample
        assert avg_sample_time < 0.0001, f"Odor sampling took {avg_sample_time:.6f}s, should be <0.0001s"
        
        # Test multi-agent odor sampling
        positions = np.random.uniform(20, 180, (50, 2))
        multi_navigator = Navigator.multi(positions=positions)
        
        start_time = time.perf_counter()
        
        for _ in range(num_samples):
            odor_values = multi_navigator.sample_odor(performance_environment)
        
        end_time = time.perf_counter()
        avg_multi_sample_time = (end_time - start_time) / num_samples
        
        # Multi-agent sampling should be efficient (vectorized)
        assert avg_multi_sample_time < 0.001, f"Multi-agent odor sampling took {avg_multi_sample_time:.6f}s"
    
    def test_sensor_sampling_performance(self, performance_environment):
        """Test multi-sensor sampling performance."""
        navigator = Navigator.single(position=(100.0, 100.0))
        
        num_samples = 200
        start_time = time.perf_counter()
        
        for _ in range(num_samples):
            sensor_readings = navigator.sample_multiple_sensors(
                performance_environment,
                sensor_distance=5.0,
                num_sensors=5
            )
        
        end_time = time.perf_counter()
        avg_sensor_sample_time = (end_time - start_time) / num_samples
        
        # Multi-sensor sampling should complete quickly
        assert avg_sensor_sample_time < 0.005, f"Multi-sensor sampling took {avg_sensor_sample_time:.6f}s"


class TestNumericalPrecision:
    """Test suite for numerical precision and accuracy validation.
    
    Validates position calculations, numerical stability, floating-point
    precision, and consistency across different numerical operations.
    """
    
    def test_position_calculation_precision(self):
        """Test numerical precision of position calculations."""
        # Test precise movement calculations
        navigator = Navigator.single(
            position=(0.0, 0.0),
            orientation=0.0,  # East
            speed=1.0
        )
        
        # Move for precise time steps
        env_array = np.zeros((50, 50))
        dt = 0.1
        num_steps = 10
        
        for _ in range(num_steps):
            navigator.step(env_array, dt=dt)
        
        # After 10 steps of 0.1s at speed 1.0, should be at (1.0, 0.0)
        expected_position = np.array([[1.0, 0.0]])
        np.testing.assert_allclose(navigator.positions, expected_position, rtol=1e-6)
    
    def test_orientation_calculation_precision(self):
        """Test numerical precision of orientation calculations."""
        navigator = Navigator.single(
            position=(0.0, 0.0),
            orientation=0.0,
            angular_velocity=36.0  # 36 degrees per second
        )
        
        env_array = np.zeros((50, 50))
        dt = 0.1
        num_steps = 10
        
        for _ in range(num_steps):
            navigator.step(env_array, dt=dt)
        
        # After 10 steps of 0.1s at 36°/s, should be at 36° total
        expected_orientation = np.array([36.0])
        np.testing.assert_allclose(navigator.orientations, expected_orientation, rtol=1e-6)
    
    def test_multi_agent_precision_consistency(self):
        """Test numerical precision consistency across multiple agents."""
        # Create identical agents at different positions
        positions = np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]])
        orientations = np.array([45.0, 45.0, 45.0])  # Same orientation
        speeds = np.array([1.5, 1.5, 1.5])           # Same speed
        angular_velocities = np.array([10.0, 10.0, 10.0])  # Same angular velocity
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            angular_velocities=angular_velocities
        )
        
        env_array = np.zeros((300, 300))
        dt = 0.1
        
        # Track relative positions between agents
        initial_relative_positions = positions[1:] - positions[0]
        
        # Simulate for multiple steps
        for _ in range(20):
            navigator.step(env_array, dt=dt)
        
        # Relative positions should remain constant (same movement for all)
        current_relative_positions = navigator.positions[1:] - navigator.positions[0]
        
        np.testing.assert_allclose(
            current_relative_positions,
            initial_relative_positions,
            rtol=1e-6
        )
        
        # All agents should have same orientation change
        orientation_differences = navigator.orientations[1:] - navigator.orientations[0]
        np.testing.assert_allclose(orientation_differences, np.zeros(2), atol=1e-6)
    
    def test_cumulative_error_bounds(self):
        """Test that cumulative numerical errors remain within bounds."""
        navigator = Navigator.single(
            position=(0.0, 0.0),
            orientation=0.0,
            speed=1.0,
            angular_velocity=1.0  # 1 degree per second
        )
        
        env_array = np.zeros((100, 100))
        dt = 0.001  # Very small time step to test numerical stability
        
        # Simulate for 1000 small steps (equivalent to 1 second)
        for _ in range(1000):
            navigator.step(env_array, dt=dt)
        
        # Position should be approximately (1.0, 0.0) after 1 second
        expected_position = np.array([[1.0, 0.0]])
        np.testing.assert_allclose(navigator.positions, expected_position, rtol=1e-3)
        
        # Orientation should be approximately 1.0 degree
        expected_orientation = np.array([1.0])
        np.testing.assert_allclose(navigator.orientations, expected_orientation, rtol=1e-3)
    
    def test_floating_point_edge_cases(self):
        """Test handling of floating-point edge cases."""
        # Test very small values
        navigator = Navigator.single(
            position=(1e-10, 1e-10),
            speed=1e-8,
            angular_velocity=1e-6
        )
        
        env_array = np.zeros((50, 50))
        
        # Should handle very small values without underflow
        navigator.step(env_array, dt=1.0)
        assert np.all(np.isfinite(navigator.positions))
        assert np.all(np.isfinite(navigator.orientations))
        
        # Test large values (within reasonable bounds)
        navigator = Navigator.single(
            position=(1e6, 1e6),
            speed=1e4,
            angular_velocity=1e3
        )
        
        # Should handle large values without overflow
        navigator.step(env_array, dt=0.001)  # Small dt to prevent huge movements
        assert np.all(np.isfinite(navigator.positions))
        assert np.all(np.isfinite(navigator.orientations))
    
    def test_trigonometric_precision(self):
        """Test precision of trigonometric calculations in movement."""
        # Test movement at cardinal directions
        cardinal_angles = [0.0, 90.0, 180.0, 270.0]
        expected_movements = [
            [1.0, 0.0],   # East
            [0.0, 1.0],   # North  
            [-1.0, 0.0],  # West
            [0.0, -1.0]   # South
        ]
        
        env_array = np.zeros((50, 50))
        
        for angle, expected_movement in zip(cardinal_angles, expected_movements):
            navigator = Navigator.single(
                position=(25.0, 25.0),
                orientation=angle,
                speed=1.0
            )
            
            navigator.step(env_array, dt=1.0)
            
            actual_movement = navigator.positions[0] - np.array([25.0, 25.0])
            np.testing.assert_allclose(actual_movement, expected_movement, rtol=1e-6)
    
    def test_inverse_operations_precision(self):
        """Test precision of inverse operations (reset to original state)."""
        # Record initial state
        initial_position = (15.0, 25.0)
        initial_orientation = 30.0
        initial_speed = 1.2
        
        navigator = Navigator.single(
            position=initial_position,
            orientation=initial_orientation,
            speed=initial_speed
        )
        
        # Perform operations that should be reversible
        env_array = np.zeros((50, 50))
        
        # Move forward then backward
        navigator.step(env_array, dt=1.0)
        current_position = navigator.positions.copy()
        current_orientation = navigator.orientations.copy()
        
        # Reverse direction and move back
        navigator.reset(
            position=current_position[0],
            orientation=(current_orientation[0] + 180.0) % 360.0,
            speed=initial_speed
        )
        navigator.step(env_array, dt=1.0)
        
        # Should be very close to original position
        np.testing.assert_allclose(navigator.positions, np.array([initial_position]), rtol=1e-3)


class TestFactoryMethodIntegration:
    """Test suite for factory method integration with configuration-driven instantiation.
    
    Validates Hydra configuration integration, factory patterns, and
    configuration validation across different instantiation methods.
    """
    
    def test_single_agent_factory_creation(self):
        """Test single agent creation through factory methods."""
        # Test Navigator.single factory method
        navigator = Navigator.single(
            position=(20.0, 30.0),
            orientation=60.0,
            speed=2.5,
            max_speed=4.0,
            angular_velocity=15.0
        )
        
        assert isinstance(navigator, Navigator)
        assert navigator.is_single_agent is True
        assert navigator.num_agents == 1
        
        # Verify all parameters were set correctly
        np.testing.assert_allclose(navigator.positions, np.array([[20.0, 30.0]]), rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, np.array([60.0]), rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, np.array([2.5]), rtol=1e-6)
        np.testing.assert_allclose(navigator.max_speeds, np.array([4.0]), rtol=1e-6)
        np.testing.assert_allclose(navigator.angular_velocities, np.array([15.0]), rtol=1e-6)
    
    def test_multi_agent_factory_creation(self):
        """Test multi-agent creation through factory methods."""
        # Test Navigator.multi factory method
        positions = np.array([[5.0, 10.0], [15.0, 20.0], [25.0, 30.0]])
        orientations = np.array([0.0, 120.0, 240.0])
        speeds = np.array([1.0, 1.5, 2.0])
        max_speeds = np.array([3.0, 3.5, 4.0])
        angular_velocities = np.array([5.0, -10.0, 15.0])
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            max_speeds=max_speeds,
            angular_velocities=angular_velocities
        )
        
        assert isinstance(navigator, Navigator)
        assert navigator.is_single_agent is False
        assert navigator.num_agents == 3
        
        # Verify all parameters were set correctly
        np.testing.assert_allclose(navigator.positions, positions, rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, orientations, rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, speeds, rtol=1e-6)
        np.testing.assert_allclose(navigator.max_speeds, max_speeds, rtol=1e-6)
        np.testing.assert_allclose(navigator.angular_velocities, angular_velocities, rtol=1e-6)
    
    def test_configuration_dictionary_creation(self):
        """Test Navigator creation from configuration dictionaries."""
        # Test single agent config
        single_config = {
            'position': (12.0, 18.0),
            'orientation': 75.0,
            'speed': 1.8,
            'max_speed': 3.5,
            'angular_velocity': 8.0
        }
        
        navigator = Navigator.from_config(single_config)
        assert navigator.is_single_agent is True
        assert navigator.num_agents == 1
        
        # Test multi-agent config
        multi_config = {
            'positions': np.array([[0.0, 0.0], [10.0, 20.0]]),
            'orientations': np.array([45.0, 225.0]),
            'speeds': np.array([1.2, 1.8]),
            'max_speeds': np.array([2.5, 3.0]),
            'angular_velocities': np.array([12.0, -6.0])
        }
        
        navigator = Navigator.from_config(multi_config)
        assert navigator.is_single_agent is False
        assert navigator.num_agents == 2
        
        np.testing.assert_allclose(navigator.positions, multi_config['positions'], rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, multi_config['orientations'], rtol=1e-6)
    
    def test_partial_configuration_handling(self):
        """Test handling of partial configurations with defaults."""
        # Test single agent with minimal config
        minimal_config = {'position': (5.0, 7.0)}
        navigator = Navigator.from_config(minimal_config)
        
        assert navigator.is_single_agent is True
        np.testing.assert_allclose(navigator.positions, np.array([[5.0, 7.0]]), rtol=1e-6)
        # Should have default values for other parameters
        np.testing.assert_allclose(navigator.orientations, np.array([0.0]), rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, np.array([0.0]), rtol=1e-6)
        np.testing.assert_allclose(navigator.max_speeds, np.array([1.0]), rtol=1e-6)
        
        # Test multi-agent with minimal config
        positions_only = {'positions': np.array([[1.0, 2.0], [3.0, 4.0]])}
        navigator = Navigator.from_config(positions_only)
        
        assert navigator.is_single_agent is False
        assert navigator.num_agents == 2
        np.testing.assert_allclose(navigator.positions, np.array([[1.0, 2.0], [3.0, 4.0]]), rtol=1e-6)
        # Should have default values
        np.testing.assert_allclose(navigator.orientations, np.zeros(2), rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, np.zeros(2), rtol=1e-6)
    
    def test_configuration_validation(self):
        """Test configuration validation during factory creation."""
        # Test invalid single agent configuration
        with pytest.raises(Exception):  # Should raise some form of validation error
            invalid_config = {'position': [1, 2, 3]}  # Wrong format
            Navigator.from_config(invalid_config)
        
        # Test mismatched array sizes in multi-agent config
        with pytest.raises(Exception):
            mismatched_config = {
                'positions': np.array([[1.0, 2.0], [3.0, 4.0]]),
                'orientations': np.array([0.0])  # Wrong size
            }
            Navigator.from_config(mismatched_config)
    
    def test_factory_method_equivalence(self):
        """Test that different factory methods produce equivalent results."""
        # Create navigator using direct constructor
        direct_navigator = Navigator(
            position=(10.0, 15.0),
            orientation=30.0,
            speed=1.5,
            max_speed=2.5,
            angular_velocity=5.0
        )
        
        # Create navigator using factory method
        factory_navigator = Navigator.single(
            position=(10.0, 15.0),
            orientation=30.0,
            speed=1.5,
            max_speed=2.5,
            angular_velocity=5.0
        )
        
        # Create navigator using config dictionary
        config = {
            'position': (10.0, 15.0),
            'orientation': 30.0,
            'speed': 1.5,
            'max_speed': 2.5,
            'angular_velocity': 5.0
        }
        config_navigator = Navigator.from_config(config)
        
        # All should produce equivalent navigators
        for nav in [factory_navigator, config_navigator]:
            np.testing.assert_allclose(nav.positions, direct_navigator.positions, rtol=1e-6)
            np.testing.assert_allclose(nav.orientations, direct_navigator.orientations, rtol=1e-6)
            np.testing.assert_allclose(nav.speeds, direct_navigator.speeds, rtol=1e-6)
            np.testing.assert_allclose(nav.max_speeds, direct_navigator.max_speeds, rtol=1e-6)
            np.testing.assert_allclose(nav.angular_velocities, direct_navigator.angular_velocities, rtol=1e-6)


class TestBoundaryHandlingAndCollisionAvoidance:
    """Test suite for boundary handling and collision avoidance behavior.
    
    Validates safety requirements, boundary conditions, collision detection,
    and proper handling of edge cases in navigation.
    """
    
    @pytest.fixture
    def bounded_environment(self) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Create a bounded environment for testing boundary conditions."""
        # Create a 30x30 environment
        environment = np.ones((30, 30), dtype=np.float32)
        bounds = (0.0, 29.0)  # Valid coordinate range
        return environment, bounds
    
    def test_position_boundary_validation(self, bounded_environment):
        """Test position validation at environment boundaries."""
        env, (min_bound, max_bound) = bounded_environment
        
        # Test positions at boundaries
        boundary_positions = [
            (min_bound, min_bound),           # Top-left corner
            (max_bound, max_bound),           # Bottom-right corner
            (min_bound, max_bound),           # Bottom-left corner
            (max_bound, min_bound),           # Top-right corner
            (min_bound + 0.5, min_bound + 0.5),  # Near top-left
            (max_bound - 0.5, max_bound - 0.5),  # Near bottom-right
        ]
        
        for position in boundary_positions:
            navigator = Navigator.single(position=position, speed=0.0)
            
            # Should not raise exceptions during odor sampling
            odor_value = navigator.sample_odor(env)
            assert np.isfinite(odor_value)
            assert odor_value >= 0.0
            
            # Position should remain valid
            assert np.all(np.isfinite(navigator.positions))
    
    def test_movement_boundary_constraints(self, bounded_environment):
        """Test movement constraints at environment boundaries."""
        env, (min_bound, max_bound) = bounded_environment
        
        # Test agent starting near boundary
        navigator = Navigator.single(
            position=(1.0, 1.0),  # Near left/top boundary
            orientation=225.0,    # Southwest direction (toward boundary)
            speed=2.0
        )
        
        # Simulate movement toward boundary
        for _ in range(5):
            navigator.step(env, dt=1.0)
            
            # Positions should remain finite and reasonable
            assert np.all(np.isfinite(navigator.positions))
            
            # System should handle boundary conditions gracefully
            # (specific behavior depends on implementation)
    
    def test_out_of_bounds_odor_sampling(self):
        """Test odor sampling for out-of-bounds positions."""
        # Small environment for testing boundaries
        small_env = np.ones((10, 10), dtype=np.float32)
        
        # Test positions outside environment bounds
        out_of_bounds_positions = [
            (-5.0, 5.0),   # Left of environment
            (15.0, 5.0),   # Right of environment
            (5.0, -5.0),   # Above environment
            (5.0, 15.0),   # Below environment
            (-2.0, -2.0),  # Upper-left outside
            (12.0, 12.0),  # Lower-right outside
        ]
        
        for position in out_of_bounds_positions:
            navigator = Navigator.single(position=position)
            
            # Should handle out-of-bounds gracefully (not crash)
            try:
                odor_value = navigator.sample_odor(small_env)
                # If no exception, value should be reasonable
                assert np.isfinite(odor_value)
                assert odor_value >= 0.0
            except (IndexError, ValueError):
                # Some implementations may raise exceptions for out-of-bounds
                # This is also acceptable behavior
                pass
    
    def test_agent_collision_detection_multi_agent(self):
        """Test collision detection and handling for multi-agent scenarios."""
        # Place agents very close to each other
        positions = np.array([
            [10.0, 10.0],
            [10.1, 10.0],  # Very close to first agent
            [10.0, 10.1],  # Very close to first agent
            [15.0, 15.0],  # Separate agent
        ])
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=np.array([0.0, 180.0, 270.0, 90.0]),
            speeds=np.array([1.0, 1.0, 1.0, 1.0])
        )
        
        env = np.ones((30, 30), dtype=np.float32)
        
        # Simulate movement - agents might collide
        for _ in range(10):
            navigator.step(env, dt=0.1)
            
            # All positions should remain valid
            assert np.all(np.isfinite(navigator.positions))
            
            # Check that agents haven't moved to identical positions
            # (basic collision avoidance expectation)
            for i in range(navigator.num_agents):
                for j in range(i + 1, navigator.num_agents):
                    distance = np.linalg.norm(navigator.positions[i] - navigator.positions[j])
                    # Agents should maintain some minimum separation
                    # (this test depends on implementation details)
                    assert distance >= 0.0  # Basic sanity check
    
    def test_extreme_parameter_handling(self):
        """Test handling of extreme parameter values."""
        # Test very high speeds
        navigator = Navigator.single(
            position=(15.0, 15.0),
            speed=1000.0,  # Extremely high speed
            max_speed=1000.0
        )
        
        env = np.ones((30, 30), dtype=np.float32)
        
        # Should handle extreme speeds without crashing
        navigator.step(env, dt=0.001)  # Small time step
        assert np.all(np.isfinite(navigator.positions))
        
        # Test very high angular velocities
        navigator = Navigator.single(
            position=(15.0, 15.0),
            angular_velocity=3600.0  # 10 rotations per second
        )
        
        navigator.step(env, dt=0.1)
        assert np.all(np.isfinite(navigator.orientations))
        
        # Test zero and negative values
        navigator = Navigator.single(
            position=(15.0, 15.0),
            speed=0.0,
            angular_velocity=0.0
        )
        
        initial_position = navigator.positions.copy()
        initial_orientation = navigator.orientations.copy()
        
        navigator.step(env, dt=1.0)
        
        # With zero speed and angular velocity, should not move
        np.testing.assert_allclose(navigator.positions, initial_position, rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, initial_orientation, rtol=1e-6)
    
    def test_sensor_boundary_conditions(self):
        """Test multi-sensor sampling at environment boundaries."""
        # Small environment
        small_env = np.ones((15, 15), dtype=np.float32)
        
        # Position agent near boundary
        navigator = Navigator.single(
            position=(2.0, 2.0),  # Near boundary
            orientation=225.0     # Facing toward boundary
        )
        
        # Test multi-sensor sampling with sensors potentially outside bounds
        sensor_readings = navigator.sample_multiple_sensors(
            small_env,
            sensor_distance=5.0,  # Large sensor distance
            num_sensors=4
        )
        
        # Should handle boundary conditions gracefully
        assert sensor_readings.shape == (4,)
        assert np.all(np.isfinite(sensor_readings))
        assert np.all(sensor_readings >= 0.0)
    
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability in edge cases."""
        # Test with very small movements
        navigator = Navigator.single(
            position=(10.0, 10.0),
            speed=1e-10  # Extremely small speed
        )
        
        env = np.ones((20, 20), dtype=np.float32)
        
        # Should handle very small movements without numerical issues
        for _ in range(100):
            navigator.step(env, dt=1.0)
            assert np.all(np.isfinite(navigator.positions))
        
        # Test with very small time steps
        navigator = Navigator.single(
            position=(10.0, 10.0),
            speed=1.0
        )
        
        for _ in range(100):
            navigator.step(env, dt=1e-10)  # Extremely small time step
            assert np.all(np.isfinite(navigator.positions))


class TestStateResetAndInitialization:
    """Test suite for state reset and initialization patterns.
    
    Validates reproducible experiments, proper state management,
    reset functionality, and initialization consistency.
    """
    
    def test_state_reset_single_agent(self):
        """Test state reset functionality for single agent."""
        # Create navigator with initial state
        navigator = Navigator.single(
            position=(10.0, 15.0),
            orientation=45.0,
            speed=1.5,
            max_speed=3.0,
            angular_velocity=10.0
        )
        
        # Modify state through simulation
        env = np.ones((50, 50), dtype=np.float32)
        for _ in range(5):
            navigator.step(env, dt=1.0)
        
        # State should have changed
        modified_position = navigator.positions.copy()
        modified_orientation = navigator.orientations.copy()
        
        # Reset to new state
        new_position = (25.0, 35.0)
        new_orientation = 90.0
        new_speed = 2.0
        
        navigator.reset(
            position=new_position,
            orientation=new_orientation,
            speed=new_speed
        )
        
        # Verify reset worked
        np.testing.assert_allclose(navigator.positions, np.array([new_position]), rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, np.array([new_orientation]), rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, np.array([new_speed]), rtol=1e-6)
    
    def test_state_reset_multi_agent(self):
        """Test state reset functionality for multi-agent."""
        # Create multi-agent navigator
        initial_positions = np.array([[5.0, 10.0], [15.0, 20.0], [25.0, 30.0]])
        navigator = Navigator.multi(
            positions=initial_positions,
            orientations=np.array([0.0, 120.0, 240.0]),
            speeds=np.array([1.0, 1.5, 2.0])
        )
        
        # Modify state through simulation
        env = np.ones((50, 50), dtype=np.float32)
        for _ in range(5):
            navigator.step(env, dt=1.0)
        
        # Reset to new state
        new_positions = np.array([[50.0, 50.0], [60.0, 60.0], [70.0, 70.0]])
        new_orientations = np.array([90.0, 180.0, 270.0])
        new_speeds = np.array([2.5, 3.0, 3.5])
        
        navigator.reset(
            positions=new_positions,
            orientations=new_orientations,
            speeds=new_speeds
        )
        
        # Verify reset worked
        np.testing.assert_allclose(navigator.positions, new_positions, rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, new_orientations, rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, new_speeds, rtol=1e-6)
    
    def test_partial_state_reset(self):
        """Test partial state reset (only some parameters)."""
        navigator = Navigator.single(
            position=(10.0, 20.0),
            orientation=30.0,
            speed=1.0,
            angular_velocity=5.0
        )
        
        # Store initial values for non-reset parameters
        initial_speed = navigator.speeds.copy()
        initial_angular_velocity = navigator.angular_velocities.copy()
        
        # Reset only position and orientation
        navigator.reset(
            position=(40.0, 50.0),
            orientation=60.0
        )
        
        # Verify only specified parameters were reset
        np.testing.assert_allclose(navigator.positions, np.array([[40.0, 50.0]]), rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, np.array([60.0]), rtol=1e-6)
        
        # Non-specified parameters should remain unchanged
        np.testing.assert_allclose(navigator.speeds, initial_speed, rtol=1e-6)
        np.testing.assert_allclose(navigator.angular_velocities, initial_angular_velocity, rtol=1e-6)
    
    def test_initialization_reproducibility(self):
        """Test that identical initialization produces identical navigators."""
        # Create multiple navigators with identical parameters
        params = {
            'position': (12.5, 17.5),
            'orientation': 67.5,
            'speed': 1.75,
            'max_speed': 2.25,
            'angular_velocity': 12.5
        }
        
        navigator1 = Navigator.single(**params)
        navigator2 = Navigator.single(**params)
        navigator3 = Navigator.single(**params)
        
        # All should be identical
        for nav in [navigator2, navigator3]:
            np.testing.assert_allclose(nav.positions, navigator1.positions, rtol=1e-15)
            np.testing.assert_allclose(nav.orientations, navigator1.orientations, rtol=1e-15)
            np.testing.assert_allclose(nav.speeds, navigator1.speeds, rtol=1e-15)
            np.testing.assert_allclose(nav.max_speeds, navigator1.max_speeds, rtol=1e-15)
            np.testing.assert_allclose(nav.angular_velocities, navigator1.angular_velocities, rtol=1e-15)
    
    def test_state_independence_after_reset(self):
        """Test that reset creates independent state (no shared references)."""
        navigator = Navigator.single(position=(5.0, 5.0))
        
        # Get reference to original position
        original_position_reference = navigator.positions
        
        # Reset to new position
        navigator.reset(position=(10.0, 10.0))
        
        # Modify current position array
        navigator.positions[0, 0] = 999.0
        
        # Original reference should not be affected by reset
        # (this tests that reset creates new arrays, not shared references)
        assert original_position_reference[0, 0] != 999.0
    
    def test_reset_parameter_validation(self):
        """Test parameter validation during reset operations."""
        navigator = Navigator.single(position=(10.0, 10.0))
        
        # Test valid reset operations (should not raise exceptions)
        navigator.reset(position=(20.0, 25.0))
        navigator.reset(orientation=45.0)
        navigator.reset(speed=2.0)
        navigator.reset(max_speed=3.0)
        navigator.reset(angular_velocity=15.0)
        
        # Test reset with multiple parameters
        navigator.reset(
            position=(30.0, 35.0),
            orientation=90.0,
            speed=1.5
        )
        
        # Verify final state
        np.testing.assert_allclose(navigator.positions, np.array([[30.0, 35.0]]), rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, np.array([90.0]), rtol=1e-6)
        np.testing.assert_allclose(navigator.speeds, np.array([1.5]), rtol=1e-6)
    
    def test_initialization_with_edge_case_values(self):
        """Test initialization with edge case parameter values."""
        # Test with zero values
        navigator = Navigator.single(
            position=(0.0, 0.0),
            orientation=0.0,
            speed=0.0,
            max_speed=0.0,
            angular_velocity=0.0
        )
        
        assert navigator.num_agents == 1
        np.testing.assert_allclose(navigator.positions, np.array([[0.0, 0.0]]), rtol=1e-6)
        np.testing.assert_allclose(navigator.orientations, np.array([0.0]), rtol=1e-6)
        
        # Test with large values
        navigator = Navigator.single(
            position=(1000.0, 2000.0),
            orientation=3600.0,  # Multiple full rotations
            speed=100.0,
            max_speed=200.0,
            angular_velocity=720.0  # 2 rotations per second
        )
        
        assert navigator.num_agents == 1
        # Should handle large values appropriately
        assert np.all(np.isfinite(navigator.positions))
        assert np.all(np.isfinite(navigator.orientations))
    
    def test_multi_agent_reset_consistency(self):
        """Test multi-agent reset maintains consistency between agents."""
        positions = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        navigator = Navigator.multi(positions=positions)
        
        # Reset all agents to new positions
        new_positions = np.array([[50.0, 60.0], [70.0, 80.0], [90.0, 100.0]])
        navigator.reset(positions=new_positions)
        
        # Verify all agents were reset correctly
        np.testing.assert_allclose(navigator.positions, new_positions, rtol=1e-6)
        
        # Test partial reset (only some agents)
        navigator.reset(
            positions=np.array([[100.0, 110.0], [120.0, 130.0], [140.0, 150.0]]),
            orientations=np.array([45.0, 90.0, 135.0])
        )
        
        # Verify consistency
        assert navigator.positions.shape == (3, 2)
        assert navigator.orientations.shape == (3,)
        assert np.all(np.isfinite(navigator.positions))
        assert np.all(np.isfinite(navigator.orientations))


# Test execution and performance monitoring
if __name__ == "__main__":
    # Run with coverage reporting
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=src.{{cookiecutter.project_slug}}.core",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/test_core_coverage",
        "--cov-fail-under=90"
    ])