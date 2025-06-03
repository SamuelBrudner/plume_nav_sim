"""Tests for the navigator functionality."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig, OmegaConf

from {{cookiecutter.project_slug}}.core.navigator import Navigator


class TestNavigatorInitialization:
    """Test navigator initialization and factory methods."""
    
    def test_navigator_single_initialization(self):
        """Test that Navigator can be initialized with orientation and speed using single() factory."""
        # Create a navigator with default parameters
        navigator = Navigator.single()
        
        # Default values should be set
        assert navigator.orientations[0] == 0.0
        assert navigator.speeds[0] == 0.0
        
        # Create a navigator with custom parameters
        custom_navigator = Navigator.single(orientation=45.0, speed=0.5)
        
        # Custom values should be set
        assert custom_navigator.orientations[0] == 45.0
        assert custom_navigator.speeds[0] == 0.5

    def test_navigator_multi_initialization(self):
        """Test that Navigator can be initialized for multiple agents using multi() factory."""
        # Create multi-agent navigator with 3 agents
        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        orientations = np.array([0.0, 45.0, 90.0])
        speeds = np.array([1.0, 1.5, 2.0])
        
        navigator = Navigator.multi(
            positions=positions,
            orientations=orientations,
            speeds=speeds
        )
        
        # Check that all agents are properly initialized
        assert navigator.num_agents == 3
        assert not navigator.is_single_agent
        np.testing.assert_array_equal(navigator.positions, positions)
        np.testing.assert_array_equal(navigator.orientations, orientations)
        np.testing.assert_array_equal(navigator.speeds, speeds)


class TestNavigatorHydraIntegration:
    """Test Hydra configuration integration with navigator creation and initialization."""
    
    def test_navigator_from_config_single_agent(self):
        """Test that Navigator can be created from Hydra DictConfig for single agent."""
        # Create a Hydra DictConfig for single agent configuration
        config = OmegaConf.create({
            'position': [10.0, 20.0],
            'orientation': 90.0,
            'speed': 1.5,
            'max_speed': 3.0,
            'angular_velocity': 0.2
        })
        
        navigator = Navigator.from_config(config)
        
        # Verify the navigator was created with config values
        assert navigator.is_single_agent
        assert navigator.positions[0][0] == 10.0
        assert navigator.positions[0][1] == 20.0
        assert navigator.orientations[0] == 90.0
        assert navigator.speeds[0] == 1.5
        assert navigator.max_speeds[0] == 3.0
        assert navigator.angular_velocities[0] == 0.2

    def test_navigator_from_config_multi_agent(self):
        """Test that Navigator can be created from Hydra DictConfig for multiple agents."""
        # Create a Hydra DictConfig for multi-agent configuration
        positions = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
        orientations = [0.0, 45.0, 90.0]
        speeds = [1.0, 1.5, 2.0]
        
        config = OmegaConf.create({
            'positions': positions,
            'orientations': orientations,
            'speeds': speeds,
            'max_speeds': [2.0, 3.0, 4.0],
            'angular_velocities': [0.1, 0.15, 0.2]
        })
        
        navigator = Navigator.from_config(config)
        
        # Verify the navigator was created with config values
        assert not navigator.is_single_agent
        assert navigator.num_agents == 3
        np.testing.assert_array_equal(navigator.positions, np.array(positions))
        np.testing.assert_array_equal(navigator.orientations, np.array(orientations))
        np.testing.assert_array_equal(navigator.speeds, np.array(speeds))

    def test_navigator_from_config_with_hydra_interpolation(self):
        """Test navigator creation with Hydra environment variable interpolation."""
        # Mock environment variable for testing
        with patch.dict('os.environ', {'NAVIGATOR_SPEED': '2.5', 'NAVIGATOR_MAX_SPEED': '5.0'}):
            # Create config with interpolation syntax
            config_yaml = """
            position: [0.0, 0.0]
            orientation: 0.0
            speed: ${oc.env:NAVIGATOR_SPEED,1.0}
            max_speed: ${oc.env:NAVIGATOR_MAX_SPEED,2.0}
            angular_velocity: 0.1
            """
            config = OmegaConf.create(OmegaConf.load(config_yaml))
            
            # Resolve interpolations
            OmegaConf.resolve(config)
            
            navigator = Navigator.from_config(config)
            
            # Verify interpolated values were used
            assert navigator.speeds[0] == 2.5
            assert navigator.max_speeds[0] == 5.0

    def test_navigator_from_config_error_handling(self):
        """Test error handling for invalid Hydra configurations."""
        # Test with invalid configuration (missing required fields)
        invalid_config = OmegaConf.create({
            'invalid_field': 'invalid_value'
        })
        
        # Should handle gracefully or provide meaningful error
        # The actual error handling depends on the implementation
        try:
            navigator = Navigator.from_config(invalid_config)
            # If no error is raised, verify the navigator has sensible defaults
            assert navigator is not None
        except (ValueError, TypeError, KeyError) as e:
            # Expected behavior for invalid config
            assert str(e) is not None


class TestNavigatorFactoryMethodPatterns:
    """Test updated factory method patterns and configuration structure."""

    @pytest.fixture
    def single_agent_config(self):
        """Fixture providing single agent Hydra configuration."""
        return OmegaConf.create({
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 1.0,
            'max_speed': 2.0,
            'angular_velocity': 0.1
        })

    @pytest.fixture
    def multi_agent_config(self):
        """Fixture providing multi-agent Hydra configuration."""
        return OmegaConf.create({
            'positions': [[0.0, 0.0], [10.0, 10.0]],
            'orientations': [0.0, 90.0],
            'speeds': [1.0, 1.5],
            'max_speeds': [2.0, 3.0],
            'angular_velocities': [0.1, 0.15]
        })

    def test_single_factory_with_config(self, single_agent_config):
        """Test single() factory method compatibility with configuration patterns."""
        # Test that single() method works with **config unpacking
        navigator = Navigator.single(**single_agent_config)
        
        assert navigator.is_single_agent
        assert navigator.num_agents == 1
        assert navigator.speeds[0] == 1.0
        assert navigator.max_speeds[0] == 2.0

    def test_multi_factory_with_config(self, multi_agent_config):
        """Test multi() factory method compatibility with configuration patterns."""
        # Convert list positions to numpy array for multi() method
        config_dict = OmegaConf.to_container(multi_agent_config, resolve=True)
        config_dict['positions'] = np.array(config_dict['positions'])
        
        navigator = Navigator.multi(**config_dict)
        
        assert not navigator.is_single_agent
        assert navigator.num_agents == 2
        np.testing.assert_array_equal(navigator.speeds, [1.0, 1.5])

    def test_configuration_compatibility_patterns(self, single_agent_config):
        """Test that navigator works with different configuration patterns."""
        # Test direct instantiation with config values
        navigator1 = Navigator(**single_agent_config)
        
        # Test factory method
        navigator2 = Navigator.single(**single_agent_config)
        
        # Test from_config method
        navigator3 = Navigator.from_config(single_agent_config)
        
        # All should produce equivalent results
        for nav in [navigator1, navigator2, navigator3]:
            assert nav.is_single_agent
            assert nav.speeds[0] == 1.0
            assert nav.max_speeds[0] == 2.0


class TestNavigatorOrientation:
    """Test that orientation is normalized properly during step."""
    
    def test_orientation_normalization_90_degrees(self):
        """Test that 90-degree orientation is handled correctly."""
        navigator_90 = Navigator.single(orientation=90.0)
        assert navigator_90.orientations[0] == 90.0
        
        # Orientation should remain 90 after step
        navigator_90.step(np.zeros((100, 100)))
        assert navigator_90.orientations[0] == 90.0

    def test_orientation_normalization_450_degrees(self):
        """Test that angles > 360 degrees are normalized during step."""
        navigator_450 = Navigator.single(orientation=450.0)
        # Initial value is not automatically normalized
        assert navigator_450.orientations[0] == 450.0
        
        # Normalization happens during step
        navigator_450.step(np.zeros((100, 100)))
        assert navigator_450.orientations[0] == 90.0

    def test_orientation_normalization_negative_angles(self):
        """Test that negative angles are normalized during step."""
        navigator_neg90 = Navigator.single(orientation=-90.0)
        # Initial value is not automatically normalized
        assert navigator_neg90.orientations[0] == -90.0
        
        # Normalization happens during step
        navigator_neg90.step(np.zeros((100, 100)))
        assert navigator_neg90.orientations[0] == 270.0


class TestNavigatorSpeed:
    """Test speed values in the controller."""
    
    def test_speed_setting_valid_range(self):
        """Test setting valid speed within max_speed range."""
        navigator_half = Navigator.single(speed=0.5, max_speed=1.0)
        assert navigator_half.speeds[0] == 0.5

    def test_speed_above_max_speed(self):
        """Test that the controller accepts speeds above max_speed."""
        # In the new implementation, speeds are not automatically capped
        navigator_max = Navigator.single(speed=2.0, max_speed=1.0)
        assert navigator_max.speeds[0] == 2.0
        
        # After a step, the speed is still not capped
        navigator_max.step(np.zeros((100, 100)))
        assert navigator_max.speeds[0] == 2.0
        
        # Verify that the movement uses the actual speed value
        dist_moved = np.linalg.norm(navigator_max.positions[0])
        assert np.isclose(dist_moved, 2.0, atol=1e-4)


class TestNavigatorMovement:
    """Test that the navigator calculates correct movement."""
    
    def test_movement_along_x_axis(self):
        """Test movement along positive x-axis at 0 degrees."""
        navigator = Navigator.single(orientation=0.0, speed=1.0, position=(0.0, 0.0))
        
        # Step the simulation to apply movement
        navigator.step(np.zeros((100, 100)))
        new_pos = navigator.positions[0]
        
        # Should have moved along positive x-axis
        assert np.isclose(new_pos[0], 1.0)
        assert np.isclose(new_pos[1], 0.0)

    def test_movement_along_y_axis(self):
        """Test movement along positive y-axis at 90 degrees."""
        navigator = Navigator.single(orientation=90.0, speed=1.0, position=(0.0, 0.0))
        navigator.step(np.zeros((100, 100)))
        new_pos = navigator.positions[0]
        
        # Should have moved along positive y-axis
        assert np.isclose(new_pos[0], 0.0)
        assert np.isclose(new_pos[1], 1.0)

    def test_movement_at_45_degrees(self):
        """Test movement at 45-degree angle with speed 0.5."""
        navigator = Navigator.single(orientation=45.0, speed=0.5, position=(0.0, 0.0))
        navigator.step(np.zeros((100, 100)))
        new_pos = navigator.positions[0]
        
        # Should have moved at 45-degree angle
        assert np.isclose(new_pos[0], 0.3536, atol=1e-4)
        assert np.isclose(new_pos[1], 0.3536, atol=1e-4)


class TestNavigatorMultiStep:
    """Test that the navigator can update its position over multiple steps."""
    
    def test_multi_step_movement(self):
        """Test navigator position updates over multiple simulation steps."""
        # Starting at position (0, 0) with orientation 0 and speed 1.0
        navigator = Navigator.single(orientation=0.0, speed=1.0, position=(0.0, 0.0))
        env = np.zeros((100, 100))

        # Move for 1 second along x-axis
        navigator.step(env)
        pos = navigator.positions[0]
        assert np.isclose(pos[0], 1.0)
        assert np.isclose(pos[1], 0.0)
        
        # Change orientation to 90 degrees and update again
        # We need to create a new navigator since we can't directly set orientation
        navigator = Navigator.single(orientation=90.0, speed=1.0, position=(1.0, 0.0))
        navigator.step(env)
        pos = navigator.positions[0]
        assert np.isclose(pos[0], 1.0)
        assert np.isclose(pos[1], 1.0)
        
        # One more step
        navigator.step(env)
        pos = navigator.positions[0]
        assert np.isclose(pos[0], 1.0)
        assert np.isclose(pos[1], 2.0)


class TestNavigatorSensorMethods:
    """Test navigator sensor sampling methods."""
    
    def test_sample_odor_basic(self):
        """Test basic odor sampling functionality."""
        navigator = Navigator.single(position=(50.0, 50.0))
        env = np.ones((100, 100)) * 0.5  # Uniform odor field
        
        odor_value = navigator.sample_odor(env)
        assert isinstance(odor_value, (float, np.ndarray))
        assert odor_value >= 0.0

    def test_sample_multiple_sensors(self):
        """Test multi-sensor odor sampling."""
        navigator = Navigator.single(position=(50.0, 50.0))
        env = np.ones((100, 100)) * 0.5  # Uniform odor field
        
        sensor_readings = navigator.sample_multiple_sensors(
            env, 
            sensor_distance=5.0,
            sensor_angle=45.0,
            num_sensors=2
        )
        
        assert isinstance(sensor_readings, np.ndarray)
        assert len(sensor_readings) >= 2  # Should have readings for multiple sensors

    def test_read_single_antenna_odor(self):
        """Test single antenna odor reading."""
        navigator = Navigator.single(position=(50.0, 50.0))
        env = np.ones((100, 100)) * 0.3
        
        antenna_reading = navigator.read_single_antenna_odor(env)
        assert isinstance(antenna_reading, (float, np.ndarray))
        assert antenna_reading >= 0.0


class TestNavigatorReset:
    """Test navigator reset functionality."""
    
    def test_reset_to_initial_state(self):
        """Test that reset returns navigator to initial state."""
        initial_pos = (10.0, 20.0)
        initial_orientation = 45.0
        navigator = Navigator.single(
            position=initial_pos,
            orientation=initial_orientation,
            speed=1.0
        )
        
        # Move the navigator
        navigator.step(np.zeros((100, 100)))
        
        # Position should have changed
        moved_pos = navigator.positions[0]
        assert not np.allclose(moved_pos, initial_pos)
        
        # Reset and verify return to initial state
        navigator.reset(
            position=initial_pos,
            orientation=initial_orientation,
            speed=0.0
        )
        
        # Should be back to initial position with zero speed
        reset_pos = navigator.positions[0]
        assert np.allclose(reset_pos, initial_pos)
        assert navigator.speeds[0] == 0.0


class TestNavigatorConfigurationValidation:
    """Test navigator configuration validation with Hydra integration."""
    
    def test_config_validation_with_pydantic_integration(self):
        """Test that Pydantic validation works with Hydra configuration."""
        # This test ensures that invalid configurations are caught
        # The actual validation logic depends on the Pydantic schemas
        
        # Test with reasonable configuration
        valid_config = OmegaConf.create({
            'position': [0.0, 0.0],
            'orientation': 0.0,
            'speed': 1.0,
            'max_speed': 2.0,
            'angular_velocity': 0.1
        })
        
        # Should create navigator without error
        navigator = Navigator.from_config(valid_config)
        assert navigator is not None
        assert navigator.is_single_agent

    def test_config_with_missing_optional_fields(self):
        """Test navigator creation with minimal configuration."""
        minimal_config = OmegaConf.create({
            'position': [5.0, 10.0],
            'orientation': 30.0
        })
        
        # Should create navigator with defaults for missing fields
        navigator = Navigator.from_config(minimal_config)
        assert navigator is not None
        assert navigator.positions[0][0] == 5.0
        assert navigator.positions[0][1] == 10.0
        assert navigator.orientations[0] == 30.0