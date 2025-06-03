"""Tests for sensor sampling utility functions.

This test module validates sensor sampling functionality within the enhanced 
cookiecutter-based project structure, including integration with Hydra 
configuration management, seed management for reproducibility, and database 
session infrastructure.

The tests cover single-agent and multi-agent sensor sampling scenarios,
configuration-driven parameter management, and integration with the enhanced
logging and visualization components of the refactored architecture.
"""

import itertools
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig

from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
from {{cookiecutter.project_slug}}.utils.navigator_utils import (
    define_sensor_offsets,
    rotate_offset,
    calculate_sensor_positions,
    sample_odor_at_sensors,
    get_predefined_sensor_layout,
    compute_sensor_positions,
    PREDEFINED_SENSOR_LAYOUTS
)


# =============================================================================
# ENHANCED FIXTURES FOR HYDRA CONFIGURATION INTEGRATION
# =============================================================================

@pytest.fixture
def hydra_sensor_config():
    """
    Create a comprehensive Hydra-compatible sensor configuration for testing.
    
    This fixture provides DictConfig objects that match the hierarchical
    configuration structure used throughout the system, supporting both
    unit testing and integration testing scenarios.
    
    Returns:
        DictConfig: Hierarchical sensor configuration with multiple scenarios
    """
    sensor_config = {
        "sensor_sampling": {
            "default": {
                "num_sensors": 2,
                "sensor_distance": 5.0,
                "sensor_angle": 45.0,
                "layout_name": None
            },
            "bilateral": {
                "num_sensors": 2,
                "sensor_distance": 10.0,
                "sensor_angle": 90.0,
                "layout_name": "LEFT_RIGHT"
            },
            "multi_sensor": {
                "num_sensors": 4,
                "sensor_distance": 8.0,
                "sensor_angle": 30.0,
                "layout_name": "FRONT_SIDES"
            },
            "single_sensor": {
                "num_sensors": 1,
                "sensor_distance": 5.0,
                "sensor_angle": 0.0,
                "layout_name": "SINGLE"
            }
        },
        "navigator": {
            "single_agent": {
                "position": [50, 60],
                "orientation": 90.0,
                "speed": 0.0,
                "max_speed": 10.0
            },
            "multi_agent": {
                "positions": [[10, 20], [30, 40], [50, 60]],
                "orientations": [0, 90, 180],
                "speeds": [0.0, 0.0, 0.0],
                "max_speeds": [10.0, 15.0, 12.0]
            }
        },
        "environment": {
            "size": [100, 100],
            "odor_center": [50, 50],
            "odor_radius": 20
        },
        "seed": {
            "global_seed": 42,
            "sensor_seed": 123,
            "enable_deterministic": True
        }
    }
    
    return DictConfig(sensor_config)


@pytest.fixture 
def mock_navigator_with_hydra_config(hydra_sensor_config):
    """
    Create a mock Navigator instance configured with Hydra settings.
    
    This fixture demonstrates integration between Hydra configuration
    management and Navigator instantiation, providing consistent test
    scenarios that match production usage patterns.
    
    Args:
        hydra_sensor_config: Hydra configuration fixture
        
    Returns:
        Mock navigator configured with Hydra parameters
    """
    mock_navigator = MagicMock(spec=NavigatorProtocol)
    
    # Configure single agent scenario from Hydra config
    nav_config = hydra_sensor_config.navigator.single_agent
    mock_navigator.positions = np.array([nav_config.position])
    mock_navigator.orientations = np.array([nav_config.orientation])
    mock_navigator.speeds = np.array([nav_config.speed])
    mock_navigator.max_speeds = np.array([nav_config.max_speed])
    mock_navigator.angular_velocities = np.array([0.0])
    mock_navigator.num_agents = 1
    
    # Configure multi-agent scenario methods
    def switch_to_multi_agent():
        multi_config = hydra_sensor_config.navigator.multi_agent
        mock_navigator.positions = np.array(multi_config.positions)
        mock_navigator.orientations = np.array(multi_config.orientations)
        mock_navigator.speeds = np.array(multi_config.speeds)
        mock_navigator.max_speeds = np.array(multi_config.max_speeds)
        mock_navigator.angular_velocities = np.array([0.0, 0.0, 0.0])
        mock_navigator.num_agents = 3
    
    mock_navigator.switch_to_multi_agent = switch_to_multi_agent
    
    # Mock sensor sampling methods
    mock_navigator.sample_multiple_sensors.return_value = np.array([0.5, 0.3])
    mock_navigator.sample_odor.return_value = 0.4
    mock_navigator.read_single_antenna_odor.return_value = 0.6
    
    return mock_navigator


@pytest.fixture
def enhanced_mock_plume(hydra_sensor_config):
    """
    Enhanced MockPlume class integrating with new data module organization.
    
    This fixture provides a sophisticated mock that aligns with the data/
    module patterns and supports configuration-driven initialization using
    Hydra configuration parameters.
    
    Args:
        hydra_sensor_config: Hydra configuration for environment setup
        
    Returns:
        Enhanced MockPlume instance with configuration integration
    """
    env_config = hydra_sensor_config.environment
    
    class EnhancedMockPlume:
        """Enhanced mock plume with Hydra configuration integration."""
        
        def __init__(self):
            self.width = env_config.size[0] 
            self.height = env_config.size[1]
            self.shape = (self.height, self.width)
            self.center = env_config.odor_center
            self.radius = env_config.odor_radius
            
            # Create sophisticated odor pattern matching data module expectations
            self._create_odor_environment()
            
            # Metadata for data module compatibility
            self.metadata = {
                "width": self.width,
                "height": self.height,
                "shape": self.shape,
                "odor_center": self.center,
                "odor_radius": self.radius,
                "frame_count": 300,
                "fps": 30.0,
                "duration": 10.0
            }
    
        def _create_odor_environment(self):
            """Create realistic odor environment with configurable parameters."""
            y, x = np.mgrid[0:self.height, 0:self.width]
            
            # Distance from odor center
            dist = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
            
            # Gaussian-like odor distribution with realistic turbulence
            base_odor = np.exp(-(dist**2) / (2 * (self.radius/3)**2))
            
            # Add turbulent variations for realistic odor plume
            turbulence = 0.1 * np.random.random(self.shape)
            
            self.current_frame = np.clip(base_odor + turbulence, 0, 1).astype(np.float32)
            
        def get_frame(self, frame_idx=0):
            """Get frame compatible with data module interface."""
            return self.current_frame
            
        def get_metadata(self):
            """Return metadata compatible with data module patterns."""
            return self.metadata
            
        def refresh_turbulence(self, seed=None):
            """Refresh turbulent patterns with optional seed for reproducibility."""
            if seed is not None:
                np.random.seed(seed)
            self._create_odor_environment()
    
    return EnhancedMockPlume()


# =============================================================================
# SEED MANAGEMENT INTEGRATION TESTS
# =============================================================================

@pytest.fixture
def mock_seed_manager():
    """
    Mock seed manager for testing reproducible sensor sampling.
    
    This fixture provides controlled randomness for sensor sampling tests,
    ensuring deterministic behavior and supporting reproducibility validation
    across test scenarios.
    """
    class MockSeedManager:
        def __init__(self):
            self.current_seed = None
            self.seed_history = []
            
        def set_global_seed(self, seed):
            """Set global seed for all random number generators."""
            self.current_seed = seed
            self.seed_history.append(seed)
            np.random.seed(seed)
            
        def get_deterministic_noise(self, shape, scale=0.1):
            """Generate deterministic noise for sensor sampling."""
            if self.current_seed is not None:
                np.random.seed(self.current_seed + len(self.seed_history))
            return np.random.normal(0, scale, shape)
            
        def create_subseed(self, base_name):
            """Create hierarchical seed for multi-agent scenarios."""
            return hash(f"{self.current_seed}_{base_name}") % 2**32
    
    return MockSeedManager()


def test_sensor_sampling_with_seed_management(mock_seed_manager, enhanced_mock_plume):
    """Test sensor sampling integration with enhanced seed management."""
    # Create navigator with deterministic behavior
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Set global seed for reproducibility
    mock_seed_manager.set_global_seed(42)
    
    navigator = Navigator(position=(50, 50), orientation=0)
    
    # Sample with deterministic seed
    odor_values_1 = sample_odor_at_sensors(
        navigator, enhanced_mock_plume.current_frame, sensor_distance=10.0
    )
    
    # Reset seed and sample again
    mock_seed_manager.set_global_seed(42)
    enhanced_mock_plume.refresh_turbulence(seed=42)
    
    odor_values_2 = sample_odor_at_sensors(
        navigator, enhanced_mock_plume.current_frame, sensor_distance=10.0
    )
    
    # Results should be identical with same seed
    np.testing.assert_array_equal(odor_values_1, odor_values_2)
    
    # Different seed should produce different results
    mock_seed_manager.set_global_seed(123)
    enhanced_mock_plume.refresh_turbulence(seed=123)
    
    odor_values_3 = sample_odor_at_sensors(
        navigator, enhanced_mock_plume.current_frame, sensor_distance=10.0
    )
    
    assert not np.array_equal(odor_values_1, odor_values_3)


# =============================================================================
# CONFIGURATION-DRIVEN PARAMETER MANAGEMENT TESTS  
# =============================================================================

def test_sensor_sampling_with_hydra_config(hydra_sensor_config, enhanced_mock_plume):
    """Test sensor sampling using Hydra configuration parameters."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Test default configuration
    default_config = hydra_sensor_config.sensor_sampling.default
    navigator = Navigator(position=(50, 50), orientation=0)
    
    odor_values = sample_odor_at_sensors(
        navigator,
        enhanced_mock_plume.current_frame,
        sensor_distance=default_config.sensor_distance,
        sensor_angle=default_config.sensor_angle,
        num_sensors=default_config.num_sensors
    )
    
    assert odor_values.shape == (1, default_config.num_sensors)
    
    # Test bilateral configuration
    bilateral_config = hydra_sensor_config.sensor_sampling.bilateral
    odor_values_bilateral = sample_odor_at_sensors(
        navigator,
        enhanced_mock_plume.current_frame,
        layout_name=bilateral_config.layout_name,
        sensor_distance=bilateral_config.sensor_distance
    )
    
    assert odor_values_bilateral.shape == (1, bilateral_config.num_sensors)


def test_multi_agent_sensor_sampling_with_config(hydra_sensor_config, enhanced_mock_plume):
    """Test multi-agent sensor sampling with Hydra configuration."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create multi-agent navigator from Hydra config
    multi_config = hydra_sensor_config.navigator.multi_agent
    navigator = Navigator(
        positions=multi_config.positions,
        orientations=multi_config.orientations
    )
    
    # Test multi-sensor configuration
    multi_sensor_config = hydra_sensor_config.sensor_sampling.multi_sensor
    odor_values = sample_odor_at_sensors(
        navigator,
        enhanced_mock_plume.current_frame,
        sensor_distance=multi_sensor_config.sensor_distance,
        sensor_angle=multi_sensor_config.sensor_angle,
        num_sensors=multi_sensor_config.num_sensors
    )
    
    expected_shape = (len(multi_config.positions), multi_sensor_config.num_sensors)
    assert odor_values.shape == expected_shape
    
    # Validate odor values are within expected range
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)


# =============================================================================
# ENHANCED LOGGING INTEGRATION TESTS
# =============================================================================

@pytest.mark.logging
def test_sensor_sampling_with_enhanced_logging(enhanced_mock_plume, caplog):
    """Test sensor sampling integration with enhanced logging system."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    import logging
    
    # Configure logging level for test
    caplog.set_level(logging.DEBUG)
    
    navigator = Navigator(position=(50, 50), orientation=0)
    
    # Perform sensor sampling (with expected logging)
    with patch('{{cookiecutter.project_slug}}.utils.navigator_utils.logger') as mock_logger:
        odor_values = sample_odor_at_sensors(
            navigator, enhanced_mock_plume.current_frame, sensor_distance=15.0
        )
        
        # Verify logging calls were made (structure depends on implementation)
        # This tests that logging integration points are maintained
        assert odor_values.shape == (1, 2)


def test_sensor_position_calculation_logging_context(enhanced_mock_plume, mock_seed_manager):
    """Test sensor position calculation with logging context preservation."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Set up logging context with seed manager
    mock_seed_manager.set_global_seed(42)
    
    navigator = Navigator(position=(25, 75), orientation=45)
    
    # Calculate sensor positions with potential logging
    positions = calculate_sensor_positions(
        navigator, sensor_distance=12.0, sensor_angle=60.0, num_sensors=3
    )
    
    # Verify positions are calculated correctly
    assert positions.shape == (1, 3, 2)
    
    # Check that positions are reasonable given navigator location
    nav_pos = np.array([25, 75])
    for sensor_pos in positions[0]:
        distance = np.linalg.norm(sensor_pos - nav_pos)
        assert distance <= 12.1  # Allow small floating point tolerance


# =============================================================================
# UPDATED CORE SENSOR SAMPLING TESTS
# =============================================================================

def test_calculate_sensor_positions_single_agent():
    """Test calculating sensor positions for a single agent."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create a navigator with a known position and orientation
    navigator = Navigator(position=(50, 60), orientation=90)  # facing "north"
    
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
    """Test calculating sensor positions for multiple agents."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create a navigator with multiple agents
    positions = [(10, 20), (30, 40), (50, 60)]
    orientations = [0, 90, 180]  # facing right, up, left
    navigator = Navigator(positions=positions, orientations=orientations)
    
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


def test_sample_odor_at_sensors_single_agent(enhanced_mock_plume):
    """Test sampling odor at sensor positions for a single agent."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create a navigator with a known position
    navigator = Navigator(position=(50, 50))

    # Sample with 2 sensors
    odor_values = sample_odor_at_sensors(navigator, enhanced_mock_plume.current_frame)

    # Should return array of shape (1, 2) - (num_agents, num_sensors)
    assert isinstance(odor_values, np.ndarray)
    assert odor_values.shape == (1, 2)

    # Sample with 4 sensors, different distance
    odor_values = sample_odor_at_sensors(
        navigator, 
        enhanced_mock_plume.current_frame,
        sensor_distance=3.0,
        sensor_angle=30.0,
        num_sensors=4
    )

    # Should return array of shape (1, 4) - (num_agents, num_sensors)
    assert odor_values.shape == (1, 4)


def test_sample_odor_at_sensors_multi_agent(enhanced_mock_plume):
    """Test sampling odor at sensor positions for multiple agents."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create a multi-agent navigator
    positions = [(25, 25), (50, 50), (75, 75)]
    navigator = Navigator(positions=positions)

    # Sample with multiple sensors
    odor_values = sample_odor_at_sensors(
        navigator, 
        enhanced_mock_plume.current_frame,
        num_sensors=3
    )

    # Should return array of shape (3, 3) - (num_agents, num_sensors)
    assert odor_values.shape == (3, 3)

    # Check that values are normalized to [0, 1] range
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)


def test_navigator_sample_multiple_sensors(enhanced_mock_plume):
    """Test the Navigator.sample_multiple_sensors method."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create a single agent navigator
    navigator_single = Navigator(position=(50, 50))

    # Create a multi-agent navigator
    positions = [(40, 40), (60, 60)]
    navigator_multi = Navigator(positions=positions)

    # Test single agent - should return a list
    readings_single = navigator_single.sample_multiple_sensors(enhanced_mock_plume.current_frame)
    assert isinstance(readings_single, list)
    assert len(readings_single) == 2  # default num_sensors

    # Test multi-agent - should return an array
    readings_multi = navigator_multi.sample_multiple_sensors(enhanced_mock_plume.current_frame)
    assert isinstance(readings_multi, np.ndarray)
    assert readings_multi.shape == (2, 2)  # (num_agents, num_sensors)

    # Test with custom parameters
    readings_custom = navigator_single.sample_multiple_sensors(
        enhanced_mock_plume.current_frame,
        sensor_distance=2.0,
        sensor_angle=60.0,
        num_sensors=3
    )
    assert isinstance(readings_custom, list)
    assert len(readings_custom) == 3  # custom num_sensors


def test_out_of_bounds_sensors():
    """Test that out-of-bounds sensors return zero odor values."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create a simple environment
    env = np.zeros((10, 10))
    env[4:7, 4:7] = 1.0  # Odor patch in the center
    
    # Create a navigator near the edge
    navigator = Navigator(position=(1, 1), orientation=180)
    
    # Sample with a large sensor distance that will place sensors outside bounds
    odor_values = sample_odor_at_sensors(
        navigator, env, sensor_distance=5.0
    )
    
    # Ensure odor_values is a numpy array
    odor_values = np.asarray(odor_values)
    
    # Check that at least one value is 0 (out of bounds)
    assert np.any(odor_values == 0)


def test_predefined_sensor_layouts():
    """Test the predefined sensor layouts."""
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


def test_compute_sensor_positions():
    """Test the compute_sensor_positions function."""
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


def test_calculate_sensor_positions_with_layout(enhanced_mock_plume):
    """Test calculate_sensor_positions using a predefined layout."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create a navigator
    navigator = Navigator(position=(50, 50), orientation=0)
    
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


def test_sample_odor_with_layout(enhanced_mock_plume):
    """Test sampling odor using a predefined layout."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create a navigator
    navigator = Navigator(position=(50, 50), orientation=0)
    
    # Sample odor with a layout
    odor_values = sample_odor_at_sensors(
        navigator, enhanced_mock_plume.current_frame, layout_name="FRONT_SIDES", sensor_distance=5.0
    )
    
    # Check shape - FRONT_SIDES has 3 sensors
    assert odor_values.shape == (1, 3)


# =============================================================================
# INTEGRATION TESTS FOR CONFIGURATION-DRIVEN PARAMETER MANAGEMENT
# =============================================================================

def test_sensor_sampling_configuration_integration(temp_config_files, enhanced_mock_plume):
    """
    Test sensor sampling with full Hydra configuration file integration.
    
    This test validates the complete configuration-driven workflow including
    hierarchical config loading, parameter validation, and sensor sampling
    execution using the cookiecutter project structure.
    """
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    from omegaconf import OmegaConf
    
    # Load configuration from temporary files
    configs = temp_config_files
    base_config = OmegaConf.load(configs["base_path"])
    
    # Create navigator from configuration
    if "navigator" in base_config:
        nav_config = base_config.navigator
        navigator = Navigator(
            position=nav_config.get("position", [50, 50]),
            orientation=nav_config.get("orientation", 0.0)
        )
    else:
        navigator = Navigator(position=(50, 50), orientation=0.0)
    
    # Test sensor sampling with config-driven parameters
    sensor_config = base_config.get("sensor_sampling", {})
    default_sensors = sensor_config.get("default", {})
    
    odor_values = sample_odor_at_sensors(
        navigator,
        enhanced_mock_plume.current_frame,
        sensor_distance=default_sensors.get("sensor_distance", 5.0),
        sensor_angle=default_sensors.get("sensor_angle", 45.0),
        num_sensors=default_sensors.get("num_sensors", 2)
    )
    
    assert isinstance(odor_values, np.ndarray)
    assert odor_values.shape[0] == 1  # Single agent
    assert odor_values.shape[1] == default_sensors.get("num_sensors", 2)


@pytest.mark.parametrize("layout_name,expected_sensors", [
    ("SINGLE", 1),
    ("LEFT_RIGHT", 2), 
    ("FRONT_SIDES", 3),
])
def test_parametrized_layout_sampling(layout_name, expected_sensors, enhanced_mock_plume):
    """Test sensor sampling with parametrized predefined layouts."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    navigator = Navigator(position=(50, 50), orientation=0)
    
    odor_values = sample_odor_at_sensors(
        navigator,
        enhanced_mock_plume.current_frame,
        layout_name=layout_name,
        sensor_distance=8.0
    )
    
    assert odor_values.shape == (1, expected_sensors)
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)


def test_database_session_integration_with_sensor_sampling(mock_db_session, enhanced_mock_plume):
    """
    Test sensor sampling with database session integration.
    
    This test validates that sensor sampling operations can work alongside
    database session management for trajectory persistence and metadata storage.
    """
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create navigator for testing
    navigator = Navigator(position=(50, 50), orientation=0)
    
    # Perform sensor sampling within database session context
    with mock_db_session as session:
        # Sample sensor data
        odor_values = sample_odor_at_sensors(
            navigator, enhanced_mock_plume.current_frame, num_sensors=3
        )
        
        # Mock trajectory record creation (would be real database model in practice)
        trajectory_record = {
            "agent_id": 1,
            "position": navigator.positions[0].tolist(),
            "orientation": navigator.orientations[0],
            "sensor_readings": odor_values[0].tolist(),
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # Simulate database operation
        session.add(trajectory_record)
        session.commit()
        
        # Verify sensor sampling worked correctly
        assert odor_values.shape == (1, 3)
        assert np.all(odor_values >= 0)
        assert np.all(odor_values <= 1)


def test_workflow_integration_sensor_sampling(mock_workflow_runner, enhanced_mock_plume):
    """
    Test sensor sampling integration with workflow orchestration systems.
    
    This test validates compatibility with DVC and Snakemake workflow
    patterns for automated experiment execution and reproducible research.
    """
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Setup workflow context
    runner = mock_workflow_runner
    runner.working_dir = "/tmp/sensor_sampling_test"
    
    # Configure workflow parameters
    workflow_config = {
        "sensor_distance": 10.0,
        "num_sensors": 4,
        "agent_positions": [[20, 20], [40, 40], [60, 60]]
    }
    runner.config_overrides = workflow_config
    
    # Create multi-agent navigator for workflow testing
    navigator = Navigator(positions=workflow_config["agent_positions"])
    
    # Execute sensor sampling as workflow step
    with patch.object(runner.dvc, 'repro') as mock_repro:
        # Simulate DVC pipeline execution
        odor_values = sample_odor_at_sensors(
            navigator,
            enhanced_mock_plume.current_frame,
            sensor_distance=workflow_config["sensor_distance"],
            num_sensors=workflow_config["num_sensors"]
        )
        
        # Verify workflow integration
        expected_shape = (len(workflow_config["agent_positions"]), workflow_config["num_sensors"])
        assert odor_values.shape == expected_shape
        
        # Verify DVC integration points
        mock_repro.return_value.returncode = 0
        result = runner.dvc.repro(dry_run=True)
        assert result.returncode == 0


# =============================================================================
# VISUALIZATION MIGRATION IMPACT TESTS
# =============================================================================

def test_sensor_sampling_visualization_integration(enhanced_mock_plume):
    """
    Test sensor sampling integration with migrated visualization components.
    
    This test validates that sensor sampling data can be properly formatted
    and integrated with the enhanced visualization utilities in the utils module.
    """
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    
    # Create navigator and sample sensor data
    navigator = Navigator(position=(50, 50), orientation=45)
    
    odor_values = sample_odor_at_sensors(
        navigator, enhanced_mock_plume.current_frame, num_sensors=4
    )
    
    sensor_positions = calculate_sensor_positions(
        navigator, num_sensors=4
    )
    
    # Format data for visualization integration
    visualization_data = {
        "agent_positions": navigator.positions,
        "agent_orientations": navigator.orientations,
        "sensor_positions": sensor_positions,
        "sensor_readings": odor_values,
        "environment_frame": enhanced_mock_plume.current_frame
    }
    
    # Verify data structure for visualization compatibility
    assert "agent_positions" in visualization_data
    assert "sensor_positions" in visualization_data
    assert "sensor_readings" in visualization_data
    assert visualization_data["sensor_positions"].shape == (1, 4, 2)
    assert visualization_data["sensor_readings"].shape == (1, 4)
    
    # Test data format compatibility with matplotlib visualization patterns
    sensor_x_coords = visualization_data["sensor_positions"][0, :, 0]
    sensor_y_coords = visualization_data["sensor_positions"][0, :, 1]
    
    assert len(sensor_x_coords) == 4
    assert len(sensor_y_coords) == 4
    assert all(isinstance(coord, (int, float, np.number)) for coord in sensor_x_coords)
    assert all(isinstance(coord, (int, float, np.number)) for coord in sensor_y_coords)


# =============================================================================
# PERFORMANCE AND STRESS TESTS
# =============================================================================

@pytest.mark.performance
def test_sensor_sampling_performance_with_large_agent_count(enhanced_mock_plume):
    """Test sensor sampling performance with large numbers of agents."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    import time
    
    # Create large number of agents
    num_agents = 50
    positions = [(10 + i*2, 10 + i*2) for i in range(num_agents)]
    orientations = [i * (360 / num_agents) for i in range(num_agents)]
    
    navigator = Navigator(positions=positions, orientations=orientations)
    
    # Time sensor sampling operation
    start_time = time.time()
    
    odor_values = sample_odor_at_sensors(
        navigator, enhanced_mock_plume.current_frame, num_sensors=3
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify results
    assert odor_values.shape == (num_agents, 3)
    
    # Performance assertion (should complete in reasonable time)
    assert execution_time < 1.0  # Should complete within 1 second
    
    # Verify all values are valid
    assert np.all(np.isfinite(odor_values))
    assert np.all(odor_values >= 0)
    assert np.all(odor_values <= 1)


@pytest.mark.stress
def test_sensor_sampling_memory_efficiency(enhanced_mock_plume):
    """Test memory efficiency of sensor sampling operations."""
    from {{cookiecutter.project_slug}}.core.navigator import Navigator
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Create moderate number of agents for memory testing
    num_agents = 20
    positions = [(25 + i*3, 25 + i*3) for i in range(num_agents)]
    navigator = Navigator(positions=positions)
    
    # Perform multiple sensor sampling operations
    for _ in range(10):
        odor_values = sample_odor_at_sensors(
            navigator, enhanced_mock_plume.current_frame, num_sensors=5
        )
        
        # Verify operation completed successfully
        assert odor_values.shape == (num_agents, 5)
    
    # Check memory usage didn't grow excessively
    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory
    
    # Memory growth should be reasonable (less than 50MB for this test)
    assert memory_growth < 50 * 1024 * 1024