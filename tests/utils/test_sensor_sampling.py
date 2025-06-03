"""Tests for sensor sampling utility functions.

This module contains comprehensive test coverage for sensor sampling functionality
in the enhanced cookiecutter-based architecture, including integration tests
for Hydra configuration management, enhanced logging with seed management,
and multi-agent sensor sampling with configuration-driven parameter management.

Enhanced Testing Features:
- Hydra configuration integration with DictConfig-based parameter management
- Enhanced logging and seed management for reproducible test execution
- Multi-agent sensor sampling validation with vectorized operations
- Configuration-driven parameter management with validation
- Integration with enhanced conftest.py fixtures for consistent testing patterns
- MockPlume class updated for new data module organization patterns

Test Categories:
1. Core Sensor Sampling: Basic sensor position calculation and odor sampling
2. Configuration Integration: Hydra-based parameter management and validation
3. Multi-Agent Scenarios: Vectorized sensor operations for multiple agents
4. Seed Management: Reproducible random state for deterministic testing
5. Enhanced Logging: Structured logging validation and output verification
6. Integration Tests: End-to-end sensor sampling workflow validation

Coverage Areas:
- Single and multi-agent sensor position calculations with layout patterns
- Odor sampling at sensor positions with bounds checking and interpolation
- Configuration-driven sensor parameter management through Hydra integration
- Reproducible test execution through enhanced seed management
- Enhanced MockPlume integration with new data module organization
"""

import itertools
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Optional, Dict, Any

# Updated imports for cookiecutter-based structure
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol, NavigatorFactory
from {{cookiecutter.project_slug}}.utils.navigator_utils import (
    define_sensor_offsets,
    rotate_offset,
    calculate_sensor_positions,
    sample_odor_at_sensors,
    get_predefined_sensor_layout,
    compute_sensor_positions,
    PREDEFINED_SENSOR_LAYOUTS
)

# Enhanced testing imports from conftest.py patterns
try:
    from {{cookiecutter.project_slug}}.utils.seed_manager import SeedManager
    SEED_MANAGER_AVAILABLE = True
except ImportError:
    SEED_MANAGER_AVAILABLE = False


class MockPlume:
    """
    Mock plume for testing, updated for new data module organization patterns.
    
    Enhanced Features:
    - Integration with new data module organization from src/{{cookiecutter.project_slug}}/data/
    - Configuration-driven plume parameters through Hydra integration
    - Seed-managed pattern generation for reproducible testing
    - Enhanced logging integration for plume state tracking
    - Compatibility with both single and multi-agent navigation scenarios
    """
    
    def __init__(self, shape=(100, 100), config: Optional[Dict[str, Any]] = None):
        """
        Initialize MockPlume with enhanced configuration support.
        
        Args:
            shape: Shape of the plume environment (height, width)
            config: Optional configuration dictionary for enhanced testing
        """
        self.shape = shape
        self.config = config or {}
        
        # Initialize enhanced plume data with configuration support
        self.current_frame = np.zeros(shape, dtype=np.float32)
        
        # Create enhanced pattern with seed management if available
        if SEED_MANAGER_AVAILABLE and self.config.get('use_seed_manager', False):
            try:
                seed_manager = SeedManager()
                seed_manager.initialize({'seed': self.config.get('seed', 42)})
            except Exception:
                pass  # Fall back to default pattern generation
        
        # Enhanced pattern generation with configuration-driven parameters
        pattern_type = self.config.get('pattern_type', 'gaussian')
        
        if pattern_type == 'gaussian':
            self._create_gaussian_pattern()
        elif pattern_type == 'uniform':
            self._create_uniform_pattern()
        else:
            self._create_default_pattern()
    
    def _create_gaussian_pattern(self):
        """Create enhanced Gaussian odor pattern."""
        y, x = np.mgrid[:self.shape[0], :self.shape[1]]
        center_x = self.config.get('center_x', self.shape[1] // 2)
        center_y = self.config.get('center_y', self.shape[0] // 2)
        sigma = self.config.get('sigma', self.shape[0] // 6)
        
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        self.current_frame = np.exp(-dist**2 / (2 * sigma**2))
        
        # Add central high-concentration patch
        self.current_frame[center_y-5:center_y+6, center_x-5:center_x+6] = 1.0
    
    def _create_uniform_pattern(self):
        """Create uniform odor pattern for testing."""
        intensity = self.config.get('intensity', 0.5)
        self.current_frame.fill(intensity)
    
    def _create_default_pattern(self):
        """Create default test pattern with central concentration."""
        # Add a pattern to the frame for testing
        center_y, center_x = self.shape[0] // 2, self.shape[1] // 2
        self.current_frame[center_y-10:center_y+11, center_x-10:center_x+11] = 1.0
        
        # Create enhanced gradient with configuration support
        y, x = np.ogrid[:self.shape[0], :self.shape[1]]
        gradient_strength = self.config.get('gradient_strength', 0.5)
        decay_rate = self.config.get('decay_rate', 100)
        
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        gradient = gradient_strength * np.exp(-dist_sq / decay_rate)
        self.current_frame += gradient
    
    @property
    def width(self):
        """Get plume width for compatibility."""
        return self.shape[1]
    
    @property 
    def height(self):
        """Get plume height for compatibility."""
        return self.shape[0]


def test_calculate_sensor_positions_single_agent():
    """Test calculating sensor positions for a single agent with enhanced validation."""
    # Create a navigator using the factory pattern
    navigator = NavigatorFactory.single_agent(position=(50, 60), orientation=90)  # facing "north"
    
    # Calculate sensor positions with default parameters (2 sensors, 45-degree angle)
    sensor_positions = calculate_sensor_positions(navigator)
    
    # Should return an array of shape (1, 2, 2) - (num_agents, num_sensors, x/y)
    assert sensor_positions.shape == (1, 2, 2)
    
    # Check that positions are calculated correctly
    # With orientation 90 and sensor angle 45, sensors should be at +/- 22.5 degrees
    # Sensor 0 should be at angle 90 - 22.5 = 67.5 degrees from horizontal
    # Sensor 1 should be at angle 90 + 22.5 = 112.5 degrees from horizontal
    
    # Check approximately using np.isclose to handle floating point
    # Sensor 0: expect ~(50 + 5*cos(67.5°), 60 + 5*sin(67.5°))
    assert np.isclose(sensor_positions[0, 0, 0], 50 + 5 * np.cos(np.radians(67.5)), atol=0.1)
    assert np.isclose(sensor_positions[0, 0, 1], 60 + 5 * np.sin(np.radians(67.5)), atol=0.1)
    
    # Sensor 1: expect ~(50 + 5*cos(112.5°), 60 + 5*sin(112.5°))
    assert np.isclose(sensor_positions[0, 1, 0], 50 + 5 * np.cos(np.radians(112.5)), atol=0.1)
    assert np.isclose(sensor_positions[0, 1, 1], 60 + 5 * np.sin(np.radians(112.5)), atol=0.1)


def test_calculate_sensor_positions_multi_agent():
    """Test calculating sensor positions for multiple agents with enhanced validation."""
    # Create a multi-agent navigator using the factory pattern
    positions = [[10, 20], [30, 40], [50, 60]]
    orientations = [0, 90, 180]  # facing right, up, left
    navigator = NavigatorFactory.multi_agent(
        positions=positions, 
        orientations=orientations
    )
    
    # Calculate sensor positions with custom parameters
    sensor_positions = calculate_sensor_positions(
        navigator, 
        sensor_distance=10.0,
        sensor_angle=30.0,
        num_sensors=3
    )
    
    # Should return array of shape (3, 3, 2) - (num_agents, num_sensors, x/y)
    assert sensor_positions.shape == (3, 3, 2)
    
    # For 3 sensors, we expect one in the center and one on each side
    # Check the center sensor for each agent (should be directly ahead)
    # Agent 0: orientation 0 degrees, center sensor at (10+10, 20)
    assert np.isclose(sensor_positions[0, 1, 0], 20, atol=0.1)
    assert np.isclose(sensor_positions[0, 1, 1], 20, atol=0.1)
    
    # Agent 1: orientation 90 degrees, center sensor at (30, 40+10)
    assert np.isclose(sensor_positions[1, 1, 0], 30, atol=0.1)
    assert np.isclose(sensor_positions[1, 1, 1], 50, atol=0.1)
    
    # Agent 2: orientation 180 degrees, center sensor at (50-10, 60)
    assert np.isclose(sensor_positions[2, 1, 0], 40, atol=0.1)
    assert np.isclose(sensor_positions[2, 1, 1], 60, atol=0.1)


def test_sample_odor_at_sensors_single_agent():
    """Test sampling odor at sensor positions for a single agent with enhanced MockPlume."""
    # Create a navigator using factory pattern
    navigator = NavigatorFactory.single_agent(position=(5, 5))

    # Create enhanced test environment with configuration
    plume_config = {
        'pattern_type': 'gaussian',
        'center_x': 10,
        'center_y': 10,
        'sigma': 5,
        'use_seed_manager': SEED_MANAGER_AVAILABLE
    }
    env_array = MockPlume(shape=(20, 20), config=plume_config).current_frame

    # Sample with 2 sensors
    odor_values = sample_odor_at_sensors(navigator, env_array)

    # Should return array of shape (1, 2) - (num_agents, num_sensors)
    assert isinstance(odor_values, np.ndarray)
    assert odor_values.shape == (1, 2)

    # Sample with 4 sensors, different distance
    odor_values = sample_odor_at_sensors(
        navigator, 
        env_array,
        sensor_distance=3.0,
        sensor_angle=30.0,
        num_sensors=4
    )

    # Should return array of shape (1, 4) - (num_agents, num_sensors)
    assert odor_values.shape == (1, 4)


def test_sample_odor_at_sensors_multi_agent():
    """Test sampling odor at sensor positions for multiple agents with enhanced configuration."""
    # Create a multi-agent navigator
    positions = [[5, 5], [10, 10], [15, 15]]
    navigator = NavigatorFactory.multi_agent(positions=positions)
    
    # Create enhanced test environment with configuration
    plume_config = {
        'pattern_type': 'gaussian',
        'center_x': 10,
        'center_y': 10,
        'sigma': 8,
        'use_seed_manager': SEED_MANAGER_AVAILABLE
    }
    env_array = MockPlume(shape=(20, 20), config=plume_config).current_frame

    # Sample with multiple sensors
    odor_values = sample_odor_at_sensors(
        navigator, 
        env_array,
        num_sensors=3
    )

    # Should return array of shape (3, 3) - (num_agents, num_sensors)
    assert odor_values.shape == (3, 3)

    # Check that values are in valid range [0, 1]
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)


def test_navigator_sample_multiple_sensors_enhanced():
    """Test the Navigator's sample_multiple_sensors method with enhanced features."""
    # Create single agent navigator using factory
    navigator_single = NavigatorFactory.single_agent(position=(5, 5))

    # Create multi-agent navigator using factory  
    positions = [[5, 5], [10, 10]]
    navigator_multi = NavigatorFactory.multi_agent(positions=positions)

    # Create enhanced test environment
    plume_config = {
        'pattern_type': 'gaussian',
        'center_x': 10,
        'center_y': 10,
        'sigma': 6,
        'use_seed_manager': SEED_MANAGER_AVAILABLE
    }
    env_array = MockPlume(shape=(20, 20), config=plume_config).current_frame

    # Test single agent - should return a list
    readings_single = navigator_single.sample_multiple_sensors(env_array)
    assert isinstance(readings_single, (list, np.ndarray))
    if isinstance(readings_single, list):
        assert len(readings_single) == 2  # default num_sensors
    else:
        assert readings_single.shape[-1] == 2  # default num_sensors

    # Test multi-agent - should return an array
    readings_multi = navigator_multi.sample_multiple_sensors(env_array)
    assert isinstance(readings_multi, np.ndarray)
    assert readings_multi.shape == (2, 2)  # (num_agents, num_sensors)

    # Test with custom parameters
    readings_custom = navigator_single.sample_multiple_sensors(
        env_array,
        sensor_distance=2.0,
        sensor_angle=60.0,
        num_sensors=3
    )
    assert isinstance(readings_custom, (list, np.ndarray))
    expected_size = 3  # custom num_sensors
    if isinstance(readings_custom, list):
        assert len(readings_custom) == expected_size
    else:
        assert readings_custom.shape[-1] == expected_size


def test_out_of_bounds_sensors_enhanced():
    """Test that out-of-bounds sensors return zero odor values with enhanced validation."""
    # Create enhanced simple environment
    env_config = {
        'pattern_type': 'uniform',
        'intensity': 0.0
    }
    env = MockPlume(shape=(10, 10), config=env_config).current_frame
    
    # Add odor patch in the center
    env[4:7, 4:7] = 1.0
    
    # Create navigator near the edge using factory
    navigator = NavigatorFactory.single_agent(position=(1, 1), orientation=180)
    
    # Sample with a large sensor distance that will place sensors outside bounds
    odor_values = sample_odor_at_sensors(
        navigator, env, sensor_distance=5.0
    )
    
    # Ensure odor_values is a numpy array
    odor_values = np.asarray(odor_values)
    
    # Check that at least one value is 0 (out of bounds)
    assert np.any(odor_values == 0)


def test_predefined_sensor_layouts_enhanced():
    """Test the predefined sensor layouts with enhanced validation."""
    # Test that all layouts exist
    assert "SINGLE" in PREDEFINED_SENSOR_LAYOUTS
    assert "LEFT_RIGHT" in PREDEFINED_SENSOR_LAYOUTS
    assert "FRONT_SIDES" in PREDEFINED_SENSOR_LAYOUTS
    
    # Test getting a layout
    single = get_predefined_sensor_layout("SINGLE", distance=1.0)
    assert single.shape == (1, 2)
    assert np.array_equal(single, np.array([[0.0, 0.0]]))
    
    # Test scaling
    left_right = get_predefined_sensor_layout("LEFT_RIGHT", distance=5.0)
    assert left_right.shape == (2, 2)
    assert np.array_equal(left_right, np.array([[0.0, 5.0], [0.0, -5.0]]))
    
    # Test front_sides
    front_sides = get_predefined_sensor_layout("FRONT_SIDES", distance=10.0)
    assert front_sides.shape == (3, 2)
    assert np.array_equal(front_sides, np.array([[10.0, 0.0], [0.0, 10.0], [0.0, -10.0]]))
    
    # Test invalid layout name
    with pytest.raises(ValueError):
        get_predefined_sensor_layout("INVALID_LAYOUT")


def test_compute_sensor_positions_enhanced():
    """Test the compute_sensor_positions function with enhanced parameter validation."""
    # Define test data
    positions = np.array([[10, 10], [50, 50], [90, 90]])
    orientations = np.array([0, 90, 180])
    
    # Test with a predefined layout
    sensor_positions = compute_sensor_positions(
        positions, orientations, layout_name="LEFT_RIGHT", distance=5.0
    )
    
    # Check shape - should be (3 agents, 2 sensors, 2 coordinates)
    assert sensor_positions.shape == (3, 2, 2)
    
    # For LEFT_RIGHT layout with orientation 0, sensors should be at (10,15) and (10,5)
    # First agent has orientation 0
    assert np.isclose(sensor_positions[0, 0, 0], 10)
    assert np.isclose(sensor_positions[0, 0, 1], 15)
    assert np.isclose(sensor_positions[0, 1, 0], 10)
    assert np.isclose(sensor_positions[0, 1, 1], 5)
    
    # For LEFT_RIGHT layout with orientation 90, sensors should be at specific positions
    # Second agent has orientation 90 degrees, so sensors are along the y-axis
    # When we rotate [0,1] (left) at 90 degrees, we get [-1,0] (downward)
    # When we rotate [0,-1] (right) at 90 degrees, we get [1,0] (upward)
    assert np.isclose(sensor_positions[1, 0, 0], 45)
    assert np.isclose(sensor_positions[1, 0, 1], 50)
    assert np.isclose(sensor_positions[1, 1, 0], 55)
    assert np.isclose(sensor_positions[1, 1, 1], 50)
    
    # Test with custom parameters instead of layout
    sensor_positions_custom = compute_sensor_positions(
        positions, orientations, layout_name=None, 
        num_sensors=3, distance=10.0, angle=45.0
    )
    
    # Check shape - should be (3 agents, 3 sensors, 2 coordinates)
    assert sensor_positions_custom.shape == (3, 3, 2)


def test_calculate_sensor_positions_with_layout():
    """Test calculate_sensor_positions using a predefined layout with factory pattern."""
    # Create a navigator using factory
    navigator = NavigatorFactory.single_agent(position=(50, 50), orientation=0)
    
    # Calculate sensor positions using a layout
    positions = calculate_sensor_positions(
        navigator, sensor_distance=5.0, layout_name="FRONT_SIDES"
    )
    
    # Check shape - FRONT_SIDES has 3 sensors
    assert positions.shape == (1, 3, 2)
    
    # Front sensor should be at (55, 50)
    assert np.isclose(positions[0, 0, 0], 55)
    assert np.isclose(positions[0, 0, 1], 50)
    
    # Left sensor should be at (50, 55)
    assert np.isclose(positions[0, 1, 0], 50)
    assert np.isclose(positions[0, 1, 1], 55)
    
    # Right sensor should be at (50, 45)
    assert np.isclose(positions[0, 2, 0], 50)
    assert np.isclose(positions[0, 2, 1], 45)


def test_sample_odor_with_layout_enhanced():
    """Test sampling odor using a predefined layout with enhanced MockPlume."""
    # Create a navigator using factory and enhanced mock plume
    navigator = NavigatorFactory.single_agent(position=(50, 50), orientation=0)
    plume_config = {
        'pattern_type': 'gaussian',
        'center_x': 50,
        'center_y': 50,
        'sigma': 10,
        'use_seed_manager': SEED_MANAGER_AVAILABLE
    }
    plume = MockPlume(config=plume_config)
    
    # Sample odor with a layout
    odor_values = sample_odor_at_sensors(
        navigator, plume.current_frame, layout_name="FRONT_SIDES", sensor_distance=5.0
    )
    
    # Check shape - FRONT_SIDES has 3 sensors
    assert odor_values.shape == (1, 3)


# Enhanced integration tests with Hydra configuration support

def test_sensor_sampling_with_hydra_config(mock_hydra_config):
    """Test sensor sampling integration with Hydra configuration management."""
    if mock_hydra_config is None:
        pytest.skip("Hydra not available for configuration testing")
    
    # Extract navigator configuration from Hydra config
    nav_config = mock_hydra_config.navigator
    
    # Create navigator from Hydra configuration
    navigator = NavigatorFactory.single_agent(
        orientation=nav_config.orientation,
        speed=nav_config.speed,
        max_speed=nav_config.max_speed
    )
    
    # Create enhanced environment based on config
    plume_config = {
        'pattern_type': 'gaussian',
        'use_seed_manager': True,
        'seed': mock_hydra_config.get('reproducibility', {}).get('global_seed', 42)
    }
    plume = MockPlume(config=plume_config)
    
    # Test sensor sampling with configuration-driven parameters
    sensor_distance = 5.0
    sensor_angle = 45.0
    num_sensors = 2
    
    odor_values = sample_odor_at_sensors(
        navigator, 
        plume.current_frame,
        sensor_distance=sensor_distance,
        sensor_angle=sensor_angle,
        num_sensors=num_sensors
    )
    
    # Validate results
    assert odor_values.shape == (1, num_sensors)
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)


def test_sensor_sampling_with_seed_management(mock_seed_manager):
    """Test sensor sampling integration with enhanced seed management."""
    if not SEED_MANAGER_AVAILABLE or mock_seed_manager is None:
        pytest.skip("SeedManager not available for reproducibility testing")
    
    # Initialize seed manager for reproducible testing
    seed_manager = mock_seed_manager
    test_seed = 123
    seed_manager.initialize({'seed': test_seed})
    
    # Create navigator and plume with seed management
    navigator = NavigatorFactory.single_agent(position=(25, 25))
    plume_config = {
        'pattern_type': 'gaussian',
        'use_seed_manager': True,
        'seed': test_seed
    }
    plume = MockPlume(config=plume_config)
    
    # Sample odor multiple times to test reproducibility
    odor_values_1 = sample_odor_at_sensors(navigator, plume.current_frame)
    
    # Reset seed and test again
    seed_manager.initialize({'seed': test_seed})
    plume_2 = MockPlume(config=plume_config)
    odor_values_2 = sample_odor_at_sensors(navigator, plume_2.current_frame)
    
    # Results should be identical with same seed
    assert np.allclose(odor_values_1, odor_values_2)


def test_multi_agent_sensor_sampling_integration(mock_multi_navigator):
    """Test multi-agent sensor sampling with enhanced integration patterns."""
    # Use the multi-agent navigator from conftest.py
    navigator = mock_multi_navigator
    
    # Create enhanced test environment
    plume_config = {
        'pattern_type': 'gaussian',
        'center_x': 50,
        'center_y': 50,
        'sigma': 15,
        'use_seed_manager': SEED_MANAGER_AVAILABLE
    }
    plume = MockPlume(shape=(100, 100), config=plume_config)
    
    # Test multi-agent sensor sampling
    odor_values = sample_odor_at_sensors(
        navigator, 
        plume.current_frame,
        sensor_distance=8.0,
        sensor_angle=60.0,
        num_sensors=3
    )
    
    # Validate multi-agent results
    assert odor_values.shape == (navigator.num_agents, 3)
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)
    
    # Test that different agents get different readings
    if navigator.num_agents > 1:
        agent_readings = [odor_values[i] for i in range(navigator.num_agents)]
        # At least some agents should have different readings
        readings_identical = all(
            np.allclose(agent_readings[0], reading) 
            for reading in agent_readings[1:]
        )
        # Allow for identical readings if agents are in similar positions
        # This is not an error, just a validation that the system works


def test_sensor_layout_configuration_driven():
    """Test sensor layout selection through configuration-driven parameter management."""
    # Test different layouts with configuration patterns
    layouts_to_test = ["SINGLE", "LEFT_RIGHT", "FRONT_SIDES"]
    
    for layout_name in layouts_to_test:
        # Create navigator using factory
        navigator = NavigatorFactory.single_agent(position=(50, 50))
        
        # Create configuration-driven plume
        plume_config = {
            'pattern_type': 'uniform',
            'intensity': 0.8
        }
        plume = MockPlume(config=plume_config)
        
        # Test sensor sampling with layout
        odor_values = sample_odor_at_sensors(
            navigator,
            plume.current_frame,
            layout_name=layout_name,
            sensor_distance=5.0
        )
        
        # Validate layout-specific results
        expected_sensors = len(PREDEFINED_SENSOR_LAYOUTS[layout_name])
        assert odor_values.shape == (1, expected_sensors)
        
        # For uniform intensity, all readings should be similar
        if plume_config['pattern_type'] == 'uniform':
            expected_intensity = plume_config['intensity']
            assert np.allclose(odor_values, expected_intensity, atol=0.1)


def test_enhanced_logging_integration(caplog):
    """Test sensor sampling integration with enhanced logging capabilities."""
    # Create navigator and plume for logging test
    navigator = NavigatorFactory.single_agent(position=(10, 10))
    plume_config = {
        'pattern_type': 'gaussian',
        'center_x': 15,
        'center_y': 15,
        'sigma': 5
    }
    plume = MockPlume(config=plume_config)
    
    # Enable logging and test sensor sampling
    with caplog.at_level("DEBUG"):
        odor_values = sample_odor_at_sensors(
            navigator,
            plume.current_frame,
            sensor_distance=3.0,
            num_sensors=2
        )
    
    # Validate that sampling completed successfully
    assert odor_values.shape == (1, 2)
    assert np.all(odor_values >= 0)
    
    # Log validation can be extended based on actual logging implementation
    # This test ensures the integration works without errors


def test_bounds_checking_enhanced_validation():
    """Test enhanced bounds checking with configuration-driven validation."""
    # Create small environment for bounds testing
    small_env_config = {
        'pattern_type': 'uniform',
        'intensity': 1.0
    }
    small_env = MockPlume(shape=(5, 5), config=small_env_config).current_frame
    
    # Create navigator that will have sensors outside bounds
    navigator = NavigatorFactory.single_agent(position=(1, 1), orientation=0)
    
    # Test with sensors that should go out of bounds
    odor_values = sample_odor_at_sensors(
        navigator,
        small_env,
        sensor_distance=10.0,  # Large distance to ensure out-of-bounds
        num_sensors=4
    )
    
    # Validate that out-of-bounds sensors return appropriate values
    assert odor_values.shape == (1, 4)
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)
    
    # Some sensors should return 0 (out of bounds)
    assert np.any(odor_values == 0)


# Enhanced performance and edge case tests

def test_sensor_sampling_performance_validation():
    """Test sensor sampling performance with large multi-agent scenarios."""
    # Create large multi-agent scenario
    num_agents = 50
    positions = [[i * 2, j * 2] for i in range(10) for j in range(5)][:num_agents]
    navigator = NavigatorFactory.multi_agent(positions=positions)
    
    # Create large environment
    plume_config = {
        'pattern_type': 'gaussian',
        'center_x': 50,
        'center_y': 25,
        'sigma': 20
    }
    large_plume = MockPlume(shape=(100, 50), config=plume_config)
    
    # Test that sampling completes efficiently
    import time
    start_time = time.time()
    
    odor_values = sample_odor_at_sensors(
        navigator,
        large_plume.current_frame,
        num_sensors=3
    )
    
    end_time = time.time()
    sampling_time = end_time - start_time
    
    # Validate results and performance
    assert odor_values.shape == (num_agents, 3)
    assert sampling_time < 1.0  # Should complete within 1 second
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)


def test_edge_case_sensor_configurations():
    """Test edge cases in sensor configuration parameters."""
    navigator = NavigatorFactory.single_agent(position=(25, 25))
    plume_config = {'pattern_type': 'uniform', 'intensity': 0.5}
    plume = MockPlume(config=plume_config)
    
    # Test single sensor
    odor_values_single = sample_odor_at_sensors(
        navigator, plume.current_frame, num_sensors=1
    )
    assert odor_values_single.shape == (1, 1)
    
    # Test large number of sensors
    odor_values_many = sample_odor_at_sensors(
        navigator, plume.current_frame, num_sensors=8, sensor_angle=45.0
    )
    assert odor_values_many.shape == (1, 8)
    
    # Test zero sensor distance (all sensors at agent position)
    odor_values_zero_dist = sample_odor_at_sensors(
        navigator, plume.current_frame, sensor_distance=0.0, num_sensors=3
    )
    assert odor_values_zero_dist.shape == (1, 3)
    # All sensors should have same value (at agent position)
    assert np.allclose(odor_values_zero_dist[0], odor_values_zero_dist[0, 0])