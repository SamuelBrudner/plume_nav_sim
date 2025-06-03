"""Tests for the navigator functionality with Hydra configuration support.

This test module validates the NavigatorProtocol implementation, factory method patterns,
and Hydra configuration integration for both single-agent and multi-agent navigation
scenarios. It ensures backward compatibility while testing new configuration features.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from unittest.mock import MagicMock, patch
from typing import Dict, Any
import warnings

# Import new navigator structure
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController
from {{cookiecutter.project_slug}}.api.navigation import (
    create_navigator,
    create_navigator_from_config,
    ConfigurationError
)

# Mock DictConfig for testing without requiring full Hydra installation
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    # Create mock DictConfig for testing
    class DictConfig(dict):
        """Mock DictConfig for testing without Hydra."""
        pass
    
    class OmegaConf:
        @staticmethod
        def to_container(cfg, resolve=True, throw_on_missing=True):
            return dict(cfg)
    
    HYDRA_AVAILABLE = False


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        'position': [50.0, 50.0],
        'orientation': 0.0,
        'speed': 1.0,
        'max_speed': 10.0,
        'angular_velocity': 0.1
    }


@pytest.fixture
def sample_multi_config():
    """Provide sample multi-agent configuration for testing."""
    return {
        'positions': [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
        'orientations': [0.0, 45.0, 90.0],
        'speeds': [1.0, 1.5, 2.0],
        'max_speeds': [5.0, 6.0, 7.0],
        'angular_velocities': [0.1, 0.15, 0.2]
    }


@pytest.fixture
def mock_hydra_config(sample_config):
    """Provide mock Hydra DictConfig for testing."""
    return DictConfig(sample_config)


@pytest.fixture
def mock_multi_hydra_config(sample_multi_config):
    """Provide mock multi-agent Hydra DictConfig for testing."""
    return DictConfig(sample_multi_config)


class TestNavigatorCreation:
    """Test navigator creation patterns and factory methods."""
    
    def test_create_navigator_direct_parameters(self):
        """Test navigator creation with direct parameters."""
        # Create a navigator with direct parameters
        navigator = create_navigator(
            position=(10.0, 20.0), 
            orientation=45.0, 
            speed=0.5,
            max_speed=5.0
        )
        
        # Verify it satisfies the NavigatorProtocol
        assert isinstance(navigator, NavigatorProtocol)
        
        # Default values should be set
        assert_allclose(navigator.positions[0], [10.0, 20.0])
        assert navigator.orientations[0] == 45.0
        assert navigator.speeds[0] == 0.5
        assert navigator.max_speeds[0] == 5.0
        assert navigator.num_agents == 1
    
    def test_create_navigator_single_default(self):
        """Test navigator creation with minimal parameters."""
        navigator = create_navigator()
        
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 1
        
        # Check that we get reasonable defaults
        assert len(navigator.positions) == 1
        assert len(navigator.orientations) == 1
        assert len(navigator.speeds) == 1
        assert len(navigator.max_speeds) == 1
    
    def test_create_navigator_multi_agent(self):
        """Test multi-agent navigator creation."""
        positions = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        orientations = [0.0, 45.0, 90.0]
        speeds = [1.0, 1.5, 2.0]
        
        navigator = create_navigator(
            positions=positions,
            orientations=orientations,
            speeds=speeds
        )
        
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 3
        assert_allclose(navigator.positions, positions)
        assert_allclose(navigator.orientations, orientations)
        assert_allclose(navigator.speeds, speeds)
    
    def test_create_navigator_hydra_config(self, mock_hydra_config):
        """Test navigator creation with Hydra configuration."""
        navigator = create_navigator(cfg=mock_hydra_config)
        
        assert isinstance(navigator, NavigatorProtocol)
        assert_allclose(navigator.positions[0], [50.0, 50.0])
        assert navigator.orientations[0] == 0.0
        assert navigator.speeds[0] == 1.0
        assert navigator.max_speeds[0] == 10.0
        assert navigator.angular_velocities[0] == 0.1
    
    def test_create_navigator_config_with_overrides(self, mock_hydra_config):
        """Test navigator creation with configuration overrides."""
        navigator = create_navigator(
            cfg=mock_hydra_config,
            max_speed=15.0,  # Override config value
            orientation=90.0  # Override config value
        )
        
        assert isinstance(navigator, NavigatorProtocol)
        assert_allclose(navigator.positions[0], [50.0, 50.0])  # From config
        assert navigator.orientations[0] == 90.0  # Overridden
        assert navigator.speeds[0] == 1.0  # From config
        assert navigator.max_speeds[0] == 15.0  # Overridden
    
    def test_create_navigator_multi_hydra_config(self, mock_multi_hydra_config):
        """Test multi-agent navigator creation with Hydra configuration."""
        navigator = create_navigator(cfg=mock_multi_hydra_config)
        
        assert isinstance(navigator, NavigatorProtocol)
        assert navigator.num_agents == 3
        assert_allclose(navigator.positions, [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        assert_allclose(navigator.orientations, [0.0, 45.0, 90.0])
        assert_allclose(navigator.speeds, [1.0, 1.5, 2.0])
        assert_allclose(navigator.max_speeds, [5.0, 6.0, 7.0])
    
    def test_create_navigator_invalid_position_parameters(self):
        """Test error handling for conflicting position parameters."""
        with pytest.raises(ConfigurationError, match="Cannot specify both 'position'"):
            create_navigator(
                position=(10.0, 20.0),
                positions=[[0.0, 0.0], [10.0, 10.0]]
            )
    
    def test_create_navigator_from_config_backward_compatibility(self, sample_config):
        """Test backward compatibility method."""
        navigator = create_navigator_from_config(cfg=sample_config)
        
        assert isinstance(navigator, NavigatorProtocol)
        assert_allclose(navigator.positions[0], [50.0, 50.0])
        assert navigator.orientations[0] == 0.0


class TestNavigatorProtocolCompliance:
    """Test NavigatorProtocol interface compliance."""
    
    @pytest.fixture
    def navigator(self):
        """Create a basic navigator for protocol testing."""
        return create_navigator(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=1.0,
            max_speed=5.0
        )
    
    def test_navigator_properties(self, navigator):
        """Test that navigator exposes required properties."""
        # Test all required properties exist
        assert hasattr(navigator, 'positions')
        assert hasattr(navigator, 'orientations')
        assert hasattr(navigator, 'speeds')
        assert hasattr(navigator, 'max_speeds')
        assert hasattr(navigator, 'angular_velocities')
        assert hasattr(navigator, 'num_agents')
        
        # Test property types
        assert isinstance(navigator.positions, np.ndarray)
        assert isinstance(navigator.orientations, np.ndarray)
        assert isinstance(navigator.speeds, np.ndarray)
        assert isinstance(navigator.max_speeds, np.ndarray)
        assert isinstance(navigator.angular_velocities, np.ndarray)
        assert isinstance(navigator.num_agents, int)
    
    def test_navigator_methods(self, navigator):
        """Test that navigator implements required methods."""
        # Test all required methods exist
        assert hasattr(navigator, 'reset')
        assert callable(navigator.reset)
        assert hasattr(navigator, 'step')
        assert callable(navigator.step)
        assert hasattr(navigator, 'sample_odor')
        assert callable(navigator.sample_odor)
        assert hasattr(navigator, 'read_single_antenna_odor')
        assert callable(navigator.read_single_antenna_odor)
        assert hasattr(navigator, 'sample_multiple_sensors')
        assert callable(navigator.sample_multiple_sensors)
    
    def test_navigator_runtime_protocol_check(self, navigator):
        """Test runtime protocol compliance checking."""
        assert isinstance(navigator, NavigatorProtocol)


class TestNavigatorMovement:
    """Test navigator movement mechanics and calculations."""
    
    def test_navigator_orientation_normalization(self):
        """Test that orientation is normalized properly during step."""
        # Create navigator with various orientations
        navigator_90 = create_navigator(position=(0.0, 0.0), orientation=90.0, speed=1.0)
        assert navigator_90.orientations[0] == 90.0
        
        # Test normalization of angles during step
        navigator_450 = create_navigator(position=(0.0, 0.0), orientation=450.0, speed=1.0)
        # Initial value may not be automatically normalized
        assert navigator_450.orientations[0] == 450.0
        
        # Normalization should happen during step
        env_array = np.zeros((100, 100))
        navigator_450.step(env_array)
        # After step, orientation should be normalized to [0, 360) range
        assert 0 <= navigator_450.orientations[0] < 360
        assert np.isclose(navigator_450.orientations[0], 90.0, atol=1e-10)
        
        # Test normalization of negative angles during step
        navigator_neg90 = create_navigator(position=(0.0, 0.0), orientation=-90.0, speed=1.0)
        assert navigator_neg90.orientations[0] == -90.0
        
        navigator_neg90.step(env_array)
        assert 0 <= navigator_neg90.orientations[0] < 360
        assert np.isclose(navigator_neg90.orientations[0], 270.0, atol=1e-10)
    
    def test_navigator_speed_handling(self):
        """Test speed values and constraints in the controller."""
        # Test setting valid speed
        navigator_half = create_navigator(
            position=(0.0, 0.0), 
            speed=0.5, 
            max_speed=1.0
        )
        assert navigator_half.speeds[0] == 0.5
        
        # Test that the controller accepts speeds above max_speed
        # (Speed capping may be implementation-dependent)
        navigator_max = create_navigator(
            position=(0.0, 0.0), 
            speed=2.0, 
            max_speed=1.0
        )
        assert navigator_max.speeds[0] == 2.0
        
        # After a step, verify the speed behavior
        env_array = np.zeros((100, 100))
        initial_pos = navigator_max.positions[0].copy()
        navigator_max.step(env_array)
        
        # Calculate distance moved
        distance_moved = np.linalg.norm(navigator_max.positions[0] - initial_pos)
        
        # Distance should be related to the speed (exact relationship depends on implementation)
        assert distance_moved > 0, "Navigator should move when speed > 0"
    
    def test_navigator_movement_directions(self):
        """Test that the navigator calculates correct movement directions."""
        env_array = np.zeros((100, 100))
        
        # Test movement at 0 degrees (positive x direction)
        navigator = create_navigator(
            position=(0.0, 0.0), 
            orientation=0.0, 
            speed=1.0
        )
        initial_pos = navigator.positions[0].copy()
        navigator.step(env_array)
        new_pos = navigator.positions[0]
        
        # Should have moved along positive x-axis
        assert new_pos[0] > initial_pos[0], "Should move in positive x direction at 0°"
        assert np.isclose(new_pos[1], initial_pos[1], atol=1e-10), "Should not move in y direction at 0°"
        
        # Test movement at 90 degrees (positive y direction)
        navigator = create_navigator(
            position=(0.0, 0.0), 
            orientation=90.0, 
            speed=1.0
        )
        initial_pos = navigator.positions[0].copy()
        navigator.step(env_array)
        new_pos = navigator.positions[0]
        
        # Should have moved along positive y-axis
        assert np.isclose(new_pos[0], initial_pos[0], atol=1e-10), "Should not move in x direction at 90°"
        assert new_pos[1] > initial_pos[1], "Should move in positive y direction at 90°"
        
        # Test movement at 45 degrees
        navigator = create_navigator(
            position=(0.0, 0.0), 
            orientation=45.0, 
            speed=1.0
        )
        initial_pos = navigator.positions[0].copy()
        navigator.step(env_array)
        new_pos = navigator.positions[0]
        
        # Should have moved at 45-degree angle (equal x and y components)
        delta_x = new_pos[0] - initial_pos[0]
        delta_y = new_pos[1] - initial_pos[1]
        assert delta_x > 0, "Should move in positive x direction at 45°"
        assert delta_y > 0, "Should move in positive y direction at 45°"
        assert np.isclose(delta_x, delta_y, atol=1e-10), "Should move equally in x and y at 45°"
    
    def test_navigator_multi_step_update(self):
        """Test that the navigator can update its position over multiple steps."""
        env_array = np.zeros((100, 100))
        
        # Create navigator and track movement
        navigator = create_navigator(
            position=(0.0, 0.0), 
            orientation=0.0, 
            speed=1.0
        )
        
        positions_history = []
        positions_history.append(navigator.positions[0].copy())
        
        # Move for several steps
        for _ in range(3):
            navigator.step(env_array)
            positions_history.append(navigator.positions[0].copy())
        
        # Verify progressive movement
        for i in range(1, len(positions_history)):
            # Each step should move further in the positive x direction
            assert positions_history[i][0] > positions_history[i-1][0], f"Step {i} should advance position"
            # Y position should remain approximately constant for 0° orientation
            assert np.isclose(
                positions_history[i][1], 
                positions_history[0][1], 
                atol=1e-10
            ), f"Y position should remain constant at step {i}"


class TestNavigatorSensorSampling:
    """Test navigator sensor sampling functionality."""
    
    @pytest.fixture
    def test_environment(self):
        """Create a test environment array."""
        env = np.zeros((100, 100))
        # Create a gradient from left to right
        for i in range(100):
            env[:, i] = i / 100.0
        return env
    
    def test_navigator_sample_odor(self, test_environment):
        """Test basic odor sampling functionality."""
        navigator = create_navigator(position=(50.0, 50.0))
        
        # Sample odor at center position
        odor_value = navigator.sample_odor(test_environment)
        
        # Should return a scalar for single agent
        assert isinstance(odor_value, (float, np.floating))
        assert 0.0 <= odor_value <= 1.0
        # Expected value should be around 0.5 (middle of gradient)
        assert np.isclose(odor_value, 0.5, atol=0.1)
    
    def test_navigator_sample_multiple_sensors(self, test_environment):
        """Test multiple sensor sampling."""
        navigator = create_navigator(position=(50.0, 50.0))
        
        # Sample with multiple sensors
        sensor_readings = navigator.sample_multiple_sensors(
            test_environment,
            sensor_distance=5.0,
            sensor_angle=45.0,
            num_sensors=2
        )
        
        # Should return array for single agent with multiple sensors
        assert isinstance(sensor_readings, np.ndarray)
        assert sensor_readings.shape == (2,) or sensor_readings.shape == (1, 2)
        
        # All readings should be valid
        assert np.all(sensor_readings >= 0.0)
        assert np.all(sensor_readings <= 1.0)
    
    def test_navigator_multi_agent_sampling(self, test_environment):
        """Test odor sampling with multi-agent navigator."""
        positions = [[25.0, 50.0], [50.0, 50.0], [75.0, 50.0]]
        navigator = create_navigator(positions=positions)
        
        # Sample odor for all agents
        odor_values = navigator.sample_odor(test_environment)
        
        # Should return array with one value per agent
        assert isinstance(odor_values, np.ndarray)
        assert len(odor_values) == 3
        
        # Values should reflect the gradient (left to right)
        assert odor_values[0] < odor_values[1] < odor_values[2]


class TestNavigatorConfigurationIntegration:
    """Test integration with configuration systems and error handling."""
    
    def test_seed_management_integration(self):
        """Test integration with seed management system."""
        # Test seed parameter
        navigator1 = create_navigator(seed=42, position=(0.0, 0.0), speed=1.0)
        navigator2 = create_navigator(seed=42, position=(0.0, 0.0), speed=1.0)
        
        # With same seed, initial states should be identical
        assert_allclose(navigator1.positions, navigator2.positions)
        assert_allclose(navigator1.orientations, navigator2.orientations)
    
    def test_configuration_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test invalid position format
        with pytest.raises(ConfigurationError):
            create_navigator(position=[1, 2, 3])  # Should be 2D
        
        # Test conflicting parameters
        with pytest.raises(ConfigurationError):
            create_navigator(
                position=(10.0, 20.0),
                positions=[[0.0, 0.0], [10.0, 10.0]]
            )
    
    def test_configuration_validation_with_hydra(self):
        """Test configuration validation with Hydra integration."""
        # Test valid configuration passes validation
        valid_config = DictConfig({
            'position': [10.0, 20.0],
            'orientation': 45.0,
            'speed': 1.0,
            'max_speed': 5.0
        })
        
        navigator = create_navigator(cfg=valid_config)
        assert isinstance(navigator, NavigatorProtocol)
    
    def test_mixed_configuration_sources(self):
        """Test mixing configuration sources with proper precedence."""
        base_config = DictConfig({
            'position': [10.0, 20.0],
            'orientation': 0.0,
            'speed': 1.0,
            'max_speed': 5.0
        })
        
        # Direct parameters should override config
        navigator = create_navigator(
            cfg=base_config,
            orientation=90.0,  # Override
            max_speed=8.0      # Override
        )
        
        assert_allclose(navigator.positions[0], [10.0, 20.0])  # From config
        assert navigator.orientations[0] == 90.0               # Overridden
        assert navigator.speeds[0] == 1.0                      # From config
        assert navigator.max_speeds[0] == 8.0                  # Overridden


class TestNavigatorReset:
    """Test navigator reset functionality."""
    
    def test_navigator_reset(self):
        """Test navigator reset to initial state."""
        navigator = create_navigator(
            position=(10.0, 20.0),
            orientation=45.0,
            speed=1.0
        )
        
        # Store initial state
        initial_pos = navigator.positions[0].copy()
        initial_orient = navigator.orientations[0]
        initial_speed = navigator.speeds[0]
        
        # Modify state by stepping
        env_array = np.zeros((100, 100))
        navigator.step(env_array)
        
        # State should have changed
        assert not np.allclose(navigator.positions[0], initial_pos)
        
        # Reset should restore initial state
        navigator.reset()
        
        assert_allclose(navigator.positions[0], initial_pos)
        assert navigator.orientations[0] == initial_orient
        assert navigator.speeds[0] == initial_speed
    
    def test_navigator_reset_with_overrides(self):
        """Test navigator reset with parameter overrides."""
        navigator = create_navigator(position=(10.0, 20.0))
        
        # Reset with new position
        new_position = np.array([[50.0, 60.0]])
        navigator.reset(positions=new_position)
        
        assert_allclose(navigator.positions[0], [50.0, 60.0])


class TestBackwardCompatibility:
    """Test backward compatibility with existing patterns."""
    
    def test_navigator_factory_patterns(self):
        """Test that old factory patterns still work."""
        # Test that we can create navigators using the old approach
        navigator = create_navigator(position=(0.0, 0.0), orientation=0.0, speed=1.0)
        
        # Should still satisfy protocol
        assert isinstance(navigator, NavigatorProtocol)
        
        # Should have expected properties
        assert hasattr(navigator, 'positions')
        assert hasattr(navigator, 'orientations')
        assert hasattr(navigator, 'speeds')
    
    def test_legacy_import_compatibility(self):
        """Test that legacy import patterns work."""
        # This test ensures that the refactored structure maintains API compatibility
        from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
        from {{cookiecutter.project_slug}}.api.navigation import create_navigator
        
        # Should be able to create and use navigator
        navigator = create_navigator()
        assert isinstance(navigator, NavigatorProtocol)
    
    def test_configuration_backward_compatibility(self):
        """Test that old configuration patterns are supported."""
        # Test dictionary-based configuration (non-Hydra)
        config = {
            'position': [10.0, 20.0],
            'orientation': 45.0,
            'speed': 1.0
        }
        
        navigator = create_navigator_from_config(cfg=config)
        assert isinstance(navigator, NavigatorProtocol)
        assert_allclose(navigator.positions[0], [10.0, 20.0])


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__])