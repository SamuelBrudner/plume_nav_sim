"""Tests for the protocol-based navigator implementation."""

import pytest
import numpy as np
from typing import Any, Dict, Tuple

from odor_plume_nav.core.protocols import NavigatorProtocol
from odor_plume_nav.core.controllers import SingleAgentController, MultiAgentController
from odor_plume_nav.core.navigator import Navigator


class TestSingleAgentController:
    """Tests for the SingleAgentController."""
    
    def test_initialization(self) -> None:
        """Test that a SingleAgentController initializes correctly."""
        # Test default initialization
        controller = SingleAgentController()
        assert controller.num_agents == 1
        assert controller.positions.shape == (1, 2)
        assert controller.orientations.shape == (1,)
        assert controller.speeds.shape == (1,)
        assert controller.max_speeds.shape == (1,)
        assert controller.angular_velocities.shape == (1,)
        
        # Test with custom parameters
        controller = SingleAgentController(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=2.0,
            max_speed=5.0,
            angular_velocity=10.0
        )
        assert controller.positions[0, 0] == 10.0
        assert controller.positions[0, 1] == 20.0
        assert controller.orientations[0] == 45.0
        assert controller.speeds[0] == 2.0
        assert controller.max_speeds[0] == 5.0
        assert controller.angular_velocities[0] == 10.0
    
    def test_reset(self) -> None:
        """Test resetting the controller state."""
        controller = SingleAgentController(position=(10.0, 20.0), orientation=45.0)
        controller.reset(position=(30.0, 40.0), orientation=90.0)
        assert controller.positions[0, 0] == 30.0
        assert controller.positions[0, 1] == 40.0
        assert controller.orientations[0] == 90.0
    
    def test_step(self) -> None:
        """Test the step method updates position and orientation."""
        controller = SingleAgentController(
            position=(10.0, 10.0),
            orientation=0.0,  # Pointing along x-axis
            speed=1.0,
            angular_velocity=10.0
        )
        
        # Create a mock environment array
        env_array = np.zeros((100, 100))
        
        # Take a step
        controller.step(env_array)
        
        # Check that position was updated (should move along x-axis)
        assert controller.positions[0, 0] > 10.0
        assert np.isclose(controller.positions[0, 1], 10.0)
        
        # Check that orientation was updated
        assert controller.orientations[0] == 10.0
    
    def test_sample_odor(self) -> None:
        """Test sampling odor at the agent's position."""
        controller = SingleAgentController(position=(5, 5))
        
        # Create an environment with a known value at the agent's position
        env_array = np.zeros((10, 10))
        env_array[5, 5] = 1.0
        
        odor = controller.sample_odor(env_array)
        assert odor == 1.0
    
    def test_sample_multiple_sensors(self) -> None:
        """Test sampling odor at multiple sensor positions."""
        controller = SingleAgentController(position=(50, 50), orientation=0.0)
        
        # Create a gradient environment
        env_array = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        env_array += np.exp(-((x - 50)**2 + (y - 50)**2) / 100)
        
        # Sample with multiple sensors
        odor_values = controller.sample_multiple_sensors(
            env_array, 
            sensor_distance=10.0,
            sensor_angle=90.0,
            num_sensors=3
        )
        
        # Check result shape and values
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (3,)
        assert np.all(odor_values >= 0.0)


class TestMultiAgentController:
    """Tests for the MultiAgentController."""
    
    def test_initialization(self) -> None:
        """Test that a MultiAgentController initializes correctly."""
        # Test default initialization
        controller = MultiAgentController()
        assert controller.num_agents == 1
        assert controller.positions.shape == (1, 2)
        
        # Test with custom parameters
        positions = np.array([[10.0, 20.0], [30.0, 40.0]])
        orientations = np.array([0.0, 90.0])
        speeds = np.array([1.0, 2.0])
        
        controller = MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds
        )
        
        assert controller.num_agents == 2
        assert controller.positions.shape == (2, 2)
        assert controller.orientations.shape == (2,)
        assert controller.speeds.shape == (2,)
        assert np.array_equal(controller.positions, positions)
        assert np.array_equal(controller.orientations, orientations)
        assert np.array_equal(controller.speeds, speeds)
    
    def test_reset(self) -> None:
        """Test resetting the controller state."""
        controller = MultiAgentController(
            positions=np.array([[10.0, 20.0], [30.0, 40.0]]),
            orientations=np.array([0.0, 90.0])
        )
        
        new_positions = np.array([[50.0, 60.0], [70.0, 80.0], [90.0, 100.0]])
        controller.reset(positions=new_positions)
        
        assert controller.num_agents == 3
        assert np.array_equal(controller.positions, new_positions)
        assert controller.orientations.shape == (3,)
    
    def test_step(self) -> None:
        """Test the step method updates positions and orientations."""
        positions = np.array([[10.0, 10.0], [20.0, 20.0]])
        orientations = np.array([0.0, 90.0])  # First agent along x, second along y
        speeds = np.array([1.0, 2.0])
        
        controller = MultiAgentController(
            positions=positions,
            orientations=orientations,
            speeds=speeds,
            angular_velocities=np.array([5.0, 10.0])
        )
        
        # Create a mock environment array
        env_array = np.zeros((100, 100))
        
        # Take a step
        controller.step(env_array)
        
        # Check that first agent moved along x-axis
        assert controller.positions[0, 0] > 10.0
        assert np.isclose(controller.positions[0, 1], 10.0)
        
        # Check that second agent moved along y-axis
        assert np.isclose(controller.positions[1, 0], 20.0)
        assert controller.positions[1, 1] > 20.0
        
        # Check that orientations were updated
        assert controller.orientations[0] == 5.0
        assert controller.orientations[1] == 100.0
    
    def test_sample_odor(self) -> None:
        """Test sampling odor at multiple agent positions."""
        positions = np.array([[5, 5], [8, 8]])
        controller = MultiAgentController(positions=positions)
        
        # Create an environment with known values at agent positions
        env_array = np.zeros((10, 10))
        env_array[5, 5] = 1.0
        env_array[8, 8] = 0.5
        
        odor_values = controller.sample_odor(env_array)
        assert odor_values.shape == (2,)
        assert odor_values[0] == 1.0
        assert odor_values[1] == 0.5
    
    def test_sample_multiple_sensors(self) -> None:
        """Test sampling odor at multiple sensor positions for multiple agents."""
        positions = np.array([[25, 25], [75, 75]])
        controller = MultiAgentController(positions=positions)
        
        # Create a gradient environment
        env_array = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        env_array += np.exp(-((x - 50)**2 + (y - 50)**2) / 100)
        
        # Sample with multiple sensors
        odor_values = controller.sample_multiple_sensors(
            env_array, 
            sensor_distance=10.0,
            sensor_angle=90.0,
            num_sensors=2
        )
        
        # Check result shape and values
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (2, 2)
        assert np.all(odor_values >= 0.0)


class TestNavigator:
    """Tests for the main Navigator facade class."""
    
    def test_single_agent_initialization(self) -> None:
        """Test initializing a single-agent navigator."""
        # Test with default parameters
        nav = Navigator()
        assert nav.is_single_agent
        assert nav.num_agents == 1
        assert nav.positions.shape == (1, 2)
        
        # Test with custom parameters
        nav = Navigator(position=(10.0, 20.0), orientation=45.0)
        assert nav.positions[0, 0] == 10.0
        assert nav.positions[0, 1] == 20.0
        assert nav.orientations[0] == 45.0
    
    def test_multi_agent_initialization(self) -> None:
        """Test initializing a multi-agent navigator."""
        positions = np.array([[10.0, 20.0], [30.0, 40.0]])
        orientations = np.array([0.0, 90.0])

        nav = Navigator(positions=positions, orientations=orientations)
        self._extracted_from_test_factory_methods_7(nav, positions, orientations)
    
    def test_factory_methods(self) -> None:
        """Test the factory methods for creating navigators."""
        # Test single-agent factory
        nav = Navigator.single(position=(10.0, 20.0), orientation=45.0)
        assert nav.is_single_agent
        assert nav.positions[0, 0] == 10.0
        assert nav.positions[0, 1] == 20.0
        assert nav.orientations[0] == 45.0

        # Test multi-agent factory
        positions = np.array([[10.0, 20.0], [30.0, 40.0]])
        orientations = np.array([0.0, 90.0])

        nav = Navigator.multi(positions=positions, orientations=orientations)
        self._extracted_from_test_factory_methods_7(nav, positions, orientations)

    # TODO Rename this here and in `test_multi_agent_initialization` and `test_factory_methods`
    def _extracted_from_test_factory_methods_7(self, nav, positions, orientations):
        assert not nav.is_single_agent
        assert nav.num_agents == 2
        assert np.array_equal(nav.positions, positions)
        assert np.array_equal(nav.orientations, orientations)
    
    def test_from_config(self) -> None:
        """Test creating a navigator from configuration."""
        # Test single-agent config
        config = {
            'position': (10.0, 20.0),
            'orientation': 45.0
        }
        
        nav = Navigator.from_config(config)
        assert nav.is_single_agent
        assert nav.positions[0, 0] == 10.0
        assert nav.positions[0, 1] == 20.0
        assert nav.orientations[0] == 45.0
        
        # Test multi-agent config
        config = {
            'positions': np.array([[10.0, 20.0], [30.0, 40.0]]),
            'orientations': np.array([0.0, 90.0])
        }
        
        nav = Navigator.from_config(config)
        assert not nav.is_single_agent
        assert nav.num_agents == 2
    
    def test_step(self) -> None:
        """Test the step method delegates correctly."""
        nav = Navigator(position=(10.0, 10.0), orientation=0.0, speed=1.0)
        
        # Create a mock environment array
        env_array = np.zeros((100, 100))
        
        # Take a step
        nav.step(env_array)
        
        # Check that position was updated
        assert nav.positions[0, 0] > 10.0
        assert np.isclose(nav.positions[0, 1], 10.0)
    
    def test_sample_odor(self) -> None:
        """Test the sample_odor method delegates correctly."""
        # Test single-agent
        nav = Navigator(position=(5, 5))
        
        env_array = np.zeros((10, 10))
        env_array[5, 5] = 1.0
        
        odor = nav.sample_odor(env_array)
        assert odor == 1.0
        
        # Test multi-agent
        positions = np.array([[5, 5], [8, 8]])
        nav = Navigator(positions=positions)
        
        env_array = np.zeros((10, 10))
        env_array[5, 5] = 1.0
        env_array[8, 8] = 0.5
        
        odor_values = nav.sample_odor(env_array)
        assert odor_values.shape == (2,)
        assert odor_values[0] == 1.0
        assert odor_values[1] == 0.5
    
    def test_sample_multiple_sensors(self) -> None:
        """Test the sample_multiple_sensors method delegates correctly."""
        # Test single-agent
        nav = Navigator(position=(50, 50), orientation=0.0)
        
        env_array = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        env_array += np.exp(-((x - 50)**2 + (y - 50)**2) / 100)
        
        odor_values = nav.sample_multiple_sensors(
            env_array, 
            sensor_distance=10.0,
            sensor_angle=90.0,
            num_sensors=3
        )
        
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (3,)
        
        # Test multi-agent
        positions = np.array([[25, 25], [75, 75]])
        nav = Navigator(positions=positions)
        
        odor_values = nav.sample_multiple_sensors(
            env_array, 
            sensor_distance=10.0,
            sensor_angle=90.0,
            num_sensors=2
        )
        
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (2, 2)
    
    def test_predefined_sensor_layout(self) -> None:
        """Test using a predefined sensor layout."""
        nav = Navigator(position=(50, 50), orientation=0.0)
        
        env_array = np.zeros((100, 100))
        y, x = np.ogrid[:100, :100]
        env_array += np.exp(-((x - 50)**2 + (y - 50)**2) / 100)
        
        odor_values = nav.sample_multiple_sensors(
            env_array, 
            layout_name="LEFT_RIGHT",
            sensor_distance=10.0
        )
        
        # LEFT_RIGHT has 2 sensors
        assert isinstance(odor_values, np.ndarray)
        assert odor_values.shape == (2,)
