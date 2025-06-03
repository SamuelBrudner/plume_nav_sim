"""
Comprehensive API test suite for the odor plume navigation system.

This module provides extensive testing coverage for the public API functions with 
enhanced Hydra configuration management, CLI integration, pytest-hydra fixture 
support, and comprehensive validation of the refactored module structure.

The test suite covers:
- Navigator creation with Hydra DictConfig and traditional parameter patterns
- VideoPlume instantiation with hierarchical configuration composition
- Simulation execution with configuration override scenarios
- Visualization integration with enhanced plotting capabilities  
- Configuration validation using pytest-hydra fixtures
- Factory method patterns for Kedro integration
- Error handling and parameter validation
- Backward compatibility with legacy parameter patterns

Testing Infrastructure:
- pytest-hydra plugin for configuration composition testing
- Click testing utilities for CLI parameter validation
- In-memory database fixtures for persistence testing
- Comprehensive mock framework for external dependencies
- Performance validation for research-grade computing requirements
"""

import contextlib
import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from typing import Dict, Any, Optional, List

# Enhanced imports for Hydra configuration testing
try:
    from omegaconf import DictConfig, OmegaConf
    from hydra import compose, initialize
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    # Fallback for environments without Hydra
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

# Import the refactored API functions from the new structure
from {{cookiecutter.project_slug}}.api.navigation import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    create_navigator_from_config,
    create_video_plume_from_config,
    ConfigurationError,
    SimulationError
)

# Import visualization functions from the new utils structure
from {{cookiecutter.project_slug}}.utils.visualization import (
    visualize_trajectory,
    SimulationVisualization
)

# Import configuration schemas for validation
from {{cookiecutter.project_slug}}.config.schemas import (
    NavigatorConfig,
    VideoPlumeConfig, 
    SimulationConfig
)

# Import core navigation components
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
from {{cookiecutter.project_slug}}.utils.seed_manager import set_global_seed


# =============================================================================
# ENHANCED FIXTURES WITH HYDRA CONFIGURATION SUPPORT
# =============================================================================

@pytest.fixture
def hydra_navigator_config():
    """
    Provides Hydra DictConfig fixture for navigator configuration testing.
    
    Supports hierarchical configuration composition with override scenarios,
    environment variable interpolation, and schema validation through 
    pytest-hydra integration patterns.
    """
    if HYDRA_AVAILABLE:
        config_dict = {
            "type": "multi_agent",
            "positions": [[10.0, 20.0], [30.0, 40.0]],
            "orientations": [45.0, 90.0],
            "speeds": [0.5, 0.7],
            "max_speeds": [2.0, 2.5],
            "angular_velocities": [0.0, 0.0],
            "seed": 42
        }
        return OmegaConf.create(config_dict)
    else:
        # Fallback for environments without Hydra
        return {
            "type": "multi_agent",
            "positions": [[10.0, 20.0], [30.0, 40.0]],
            "orientations": [45.0, 90.0],
            "speeds": [0.5, 0.7],
            "max_speeds": [2.0, 2.5],
            "angular_velocities": [0.0, 0.0],
            "seed": 42
        }


@pytest.fixture
def hydra_single_agent_config():
    """
    Provides Hydra DictConfig fixture for single-agent configuration testing.
    
    Validates single-agent parameter patterns with Hydra configuration
    composition and override mechanisms.
    """
    if HYDRA_AVAILABLE:
        config_dict = {
            "type": "single_agent", 
            "position": [10.0, 20.0],
            "orientation": 45.0,
            "speed": 0.5,
            "max_speed": 2.0,
            "angular_velocity": 0.0,
            "seed": 123
        }
        return OmegaConf.create(config_dict)
    else:
        return {
            "type": "single_agent",
            "position": [10.0, 20.0],
            "orientation": 45.0,
            "speed": 0.5,
            "max_speed": 2.0,
            "angular_velocity": 0.0,
            "seed": 123
        }


@pytest.fixture
def hydra_video_plume_config():
    """
    Provides Hydra DictConfig fixture for video plume configuration testing.
    
    Supports file path validation, parameter override scenarios, and
    configuration composition patterns for video processing parameters.
    """
    if HYDRA_AVAILABLE:
        config_dict = {
            "video_path": "test_video.mp4",
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }
        return OmegaConf.create(config_dict)
    else:
        return {
            "video_path": "test_video.mp4", 
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }


@pytest.fixture
def hydra_simulation_config():
    """
    Provides Hydra DictConfig fixture for simulation configuration testing.
    
    Validates simulation parameter composition with performance constraints,
    timing requirements, and reproducibility settings.
    """
    if HYDRA_AVAILABLE:
        config_dict = {
            "num_steps": 100,
            "dt": 0.1,
            "sensor_distance": 5.0,
            "sensor_angle": 45.0,
            "record_trajectory": True,
            "seed": 456
        }
        return OmegaConf.create(config_dict)
    else:
        return {
            "num_steps": 100,
            "dt": 0.1,
            "sensor_distance": 5.0,
            "sensor_angle": 45.0,
            "record_trajectory": True,
            "seed": 456
        }


@pytest.fixture
def mock_video_capture():
    """
    Enhanced mock for cv2.VideoCapture with comprehensive frame simulation.
    
    Provides predictable video processing behavior for testing video plume
    functionality with configurable frame sequences and property validation.
    """
    with patch('cv2.VideoCapture') as mock_cap:
        # Configure the mock to return appropriate values
        mock_instance = MagicMock()
        mock_cap.return_value = mock_instance
        
        # Mock isOpened to return True by default
        mock_instance.isOpened.return_value = True
        
        # Configure property values for a synthetic video
        cap_properties = {
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30.0
        }
        
        # Configure get method to return values from the dictionary
        mock_instance.get.side_effect = lambda prop: cap_properties.get(prop, 0)
        
        # Mock read to return a valid BGR frame (3 channels)
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_instance.read.return_value = (True, mock_frame)
        
        yield mock_cap


@pytest.fixture
def mock_file_exists(monkeypatch):
    """
    Enhanced file existence mock supporting configuration file patterns.
    
    Provides controlled file system behavior for testing configuration
    loading, video file validation, and path resolution scenarios.
    """
    def patched_exists(self):
        # Allow certain test file patterns to exist
        test_patterns = [
            "test_video.mp4",
            "config.yaml", 
            "base.yaml",
            "test_config.yaml"
        ]
        return any(pattern in str(self) for pattern in test_patterns)
    
    monkeypatch.setattr(Path, "exists", patched_exists)
    return patched_exists


@pytest.fixture
def mock_simulation_runner():
    """
    Enhanced mock for simulation execution with realistic output patterns.
    
    Provides controlled simulation behavior for testing multi-agent scenarios,
    performance validation, and trajectory recording functionality.
    """
    with patch('{{cookiecutter.project_slug}}.api.navigation.run_plume_simulation') as mock_run:
        # Configure mock to return synthetic multi-agent simulation data
        def create_simulation_data(navigator, video_plume, **kwargs):
            # Determine number of agents from navigator
            num_agents = getattr(navigator, 'num_agents', 1)
            if hasattr(navigator, 'positions'):
                if hasattr(navigator.positions, 'shape'):
                    num_agents = navigator.positions.shape[0]
                elif hasattr(navigator.positions, '__len__'):
                    num_agents = len(navigator.positions)
            
            num_steps = kwargs.get('num_steps', 100)
            
            # Generate realistic trajectory data
            positions_history = np.random.random((num_agents, num_steps + 1, 2)) * 100
            orientations_history = np.random.random((num_agents, num_steps + 1)) * 360
            odor_readings = np.random.random((num_agents, num_steps + 1))
            
            return positions_history, orientations_history, odor_readings
        
        mock_run.side_effect = create_simulation_data
        yield mock_run


# =============================================================================
# NAVIGATOR CREATION TESTS WITH HYDRA CONFIGURATION SUPPORT
# =============================================================================

def test_create_navigator_default():
    """Test creating a navigator with default parameters."""
    navigator = create_navigator()
    
    # Should create a default Navigator instance (single agent)
    assert hasattr(navigator, 'positions')
    assert hasattr(navigator, 'orientations') 
    assert hasattr(navigator, 'speeds')
    assert hasattr(navigator, 'max_speeds')
    
    # Check default single-agent configuration
    assert navigator.positions.shape == (1, 2)  # Single agent with 2D position
    assert navigator.orientations.shape == (1,)  # Single agent orientation
    assert navigator.speeds.shape == (1,)  # Single agent speed
    
    # Verify protocol compliance
    assert isinstance(navigator, NavigatorProtocol)


def test_create_navigator_with_hydra_config(hydra_navigator_config):
    """Test creating a navigator using Hydra DictConfig."""
    navigator = create_navigator(cfg=hydra_navigator_config)
    
    # Validate multi-agent configuration from Hydra config
    assert navigator.positions.shape == (2, 2)  # Two agents with 2D positions
    assert navigator.orientations.shape == (2,)  # Two agent orientations
    assert navigator.speeds.shape == (2,)  # Two agent speeds
    
    # Check that config values were applied correctly
    assert np.allclose(navigator.orientations, [45.0, 90.0])
    assert np.allclose(navigator.speeds, [0.5, 0.7])
    assert np.allclose(navigator.max_speeds, [2.0, 2.5])
    
    # Verify positions were set from config
    expected_positions = np.array([[10.0, 20.0], [30.0, 40.0]])
    assert np.allclose(navigator.positions, expected_positions)


def test_create_navigator_single_agent_with_hydra_config(hydra_single_agent_config):
    """Test creating a single-agent navigator using Hydra DictConfig."""
    navigator = create_navigator(cfg=hydra_single_agent_config)
    
    # Validate single-agent configuration
    assert navigator.positions.shape == (1, 2)
    assert navigator.orientations.shape == (1,)
    assert navigator.speeds.shape == (1,)
    
    # Check that single-agent config values were applied
    assert navigator.orientations[0] == 45.0
    assert navigator.speeds[0] == 0.5
    assert navigator.max_speeds[0] == 2.0
    assert np.allclose(navigator.positions[0], [10.0, 20.0])


def test_create_navigator_config_with_overrides(hydra_navigator_config):
    """Test that direct parameters override Hydra configuration values."""
    # Override specific parameters
    navigator = create_navigator(
        cfg=hydra_navigator_config,
        max_speeds=[5.0, 6.0],  # Override max speeds
        seed=999  # Override seed
    )
    
    # Direct parameters should take precedence over config
    assert np.allclose(navigator.max_speeds, [5.0, 6.0])
    
    # Non-overridden values should come from config
    assert np.allclose(navigator.orientations, [45.0, 90.0])
    assert np.allclose(navigator.speeds, [0.5, 0.7])


def test_create_navigator_traditional_parameters():
    """Test creating navigator with traditional parameter patterns (backward compatibility)."""
    # Multi-agent with traditional parameters
    positions = [(10, 20), (30, 40), (50, 60)]
    orientations = [45, 90, 135]
    speeds = [0.5, 0.7, 0.9]
    
    navigator = create_navigator(
        positions=positions,
        orientations=orientations,
        speeds=speeds
    )
    
    # Check that the navigator has the correct number of agents
    assert navigator.positions.shape == (3, 2)
    
    # Check that all agents have correct positions
    assert np.allclose(navigator.positions, positions)
    
    # Verify each agent has correct orientation and speed
    assert np.allclose(navigator.orientations, orientations)
    assert np.allclose(navigator.speeds, speeds)


def test_create_navigator_single_agent_traditional():
    """Test creating single-agent navigator with traditional parameters."""
    navigator = create_navigator(
        position=(10, 20),
        orientation=45,
        speed=0.5,
        max_speed=2.0
    )
    
    # Check single-agent properties
    assert navigator.positions.shape == (1, 2)
    assert navigator.orientations[0] == 45
    assert navigator.speeds[0] == 0.5
    assert navigator.max_speeds[0] == 2.0
    assert np.allclose(navigator.positions[0], [10, 20])


def test_create_navigator_conflicting_position_parameters():
    """Test error handling when both position and positions are provided."""
    with pytest.raises(ConfigurationError, match=r"Cannot specify both 'position'.*and 'positions'"):
        create_navigator(position=(0, 0), positions=[(1, 2), (3, 4)])


def test_create_navigator_from_config_factory_method(hydra_navigator_config):
    """Test factory method for backward compatibility."""
    navigator = create_navigator_from_config(hydra_navigator_config)
    
    # Should produce same result as create_navigator with cfg parameter
    assert navigator.positions.shape == (2, 2)
    assert np.allclose(navigator.orientations, [45.0, 90.0])
    
    # Verify protocol compliance
    assert isinstance(navigator, NavigatorProtocol)


def test_create_navigator_invalid_config_type():
    """Test error handling for invalid configuration types."""
    with pytest.raises(ConfigurationError):
        create_navigator(cfg="invalid_config_string")


def test_create_navigator_seed_management():
    """Test global seed management integration."""
    # Test with seed parameter
    navigator1 = create_navigator(position=(0, 0), seed=42)
    navigator2 = create_navigator(position=(0, 0), seed=42)
    
    # Should create identical navigators with same seed
    assert np.allclose(navigator1.positions, navigator2.positions)


# =============================================================================
# VIDEO PLUME CREATION TESTS WITH ENHANCED CONFIGURATION
# =============================================================================

def test_create_video_plume_basic(mock_video_capture, mock_file_exists):
    """Test creating a video plume with basic parameters."""
    plume = create_video_plume("test_video.mp4", flip=True, kernel_size=5)
    
    # Check that the plume has the correct properties
    assert plume.video_path == Path("test_video.mp4")
    assert plume.flip is True
    assert plume.kernel_size == 5
    
    # Verify it's a VideoPlume instance
    assert isinstance(plume, VideoPlume)


def test_create_video_plume_with_hydra_config(mock_video_capture, mock_file_exists, hydra_video_plume_config):
    """Test creating video plume using Hydra DictConfig."""
    plume = create_video_plume(cfg=hydra_video_plume_config)
    
    # Verify config values were applied
    assert plume.video_path == Path("test_video.mp4")
    assert plume.flip is True
    assert plume.kernel_size == 5
    assert plume.kernel_sigma == 1.0


def test_create_video_plume_config_with_overrides(mock_video_capture, mock_file_exists, hydra_video_plume_config):
    """Test that direct parameters override Hydra configuration."""
    plume = create_video_plume(
        cfg=hydra_video_plume_config,
        flip=False,  # Override config value
        kernel_size=7  # Override config value
    )
    
    # Direct parameters should take precedence
    assert plume.flip is False
    assert plume.kernel_size == 7
    
    # Non-overridden values should come from config
    assert plume.kernel_sigma == 1.0


def test_create_video_plume_factory_method(mock_video_capture, mock_file_exists, hydra_video_plume_config):
    """Test factory method for backward compatibility."""
    plume = create_video_plume_from_config(
        cfg=hydra_video_plume_config,
        video_path="override_video.mp4"
    )
    
    # Should handle video_path override correctly
    assert plume.video_path == Path("override_video.mp4")
    assert plume.flip is True  # From config


def test_create_video_plume_missing_video_path():
    """Test error handling when video_path is missing."""
    with pytest.raises(ConfigurationError, match="video_path is required"):
        create_video_plume()


def test_create_video_plume_nonexistent_file(mock_video_capture, monkeypatch):
    """Test error handling for nonexistent video files."""
    # Mock file doesn't exist
    monkeypatch.setattr(Path, "exists", lambda self: False)
    
    with pytest.raises(FileNotFoundError, match="Video file does not exist"):
        create_video_plume("nonexistent.mp4")


def test_create_video_plume_invalid_parameters(mock_video_capture, mock_file_exists):
    """Test validation of video plume parameters."""
    # Test invalid flip parameter
    with pytest.raises(ConfigurationError, match="flip must be a boolean"):
        create_video_plume("test_video.mp4", flip="yes")
    
    # Test invalid kernel_size
    with pytest.raises(ConfigurationError, match="kernel_size must be a positive integer"):
        create_video_plume("test_video.mp4", kernel_size=-1)
    
    # Test invalid kernel_sigma
    with pytest.raises(ConfigurationError, match="kernel_sigma must be a positive number"):
        create_video_plume("test_video.mp4", kernel_sigma=-1.0)


# =============================================================================
# SIMULATION EXECUTION TESTS WITH ENHANCED CONFIGURATION
# =============================================================================

def test_run_plume_simulation_basic(mock_simulation_runner, mock_video_capture, mock_file_exists):
    """Test basic simulation execution with direct parameters."""
    # Create navigator and video plume
    navigator = create_navigator(position=(10, 20), orientation=45)
    plume = create_video_plume("test_video.mp4")
    
    # Run the simulation
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=100, dt=0.5
    )
    
    # Check that the simulation function was called with correct parameters
    mock_simulation_runner.assert_called_once()
    args, kwargs = mock_simulation_runner.call_args
    assert args[0] == navigator
    assert args[1] == plume
    assert kwargs["num_steps"] == 100
    assert kwargs["dt"] == 0.5
    
    # Validate return value shapes for single agent
    assert positions.shape[0] == 1  # Single agent
    assert orientations.shape[0] == 1
    assert readings.shape[0] == 1


def test_run_plume_simulation_with_hydra_config(mock_simulation_runner, mock_video_capture, mock_file_exists, hydra_simulation_config):
    """Test simulation execution using Hydra DictConfig."""
    # Create navigator and video plume
    navigator = create_navigator(positions=[(0, 0), (10, 10)])
    plume = create_video_plume("test_video.mp4")
    
    # Run simulation with Hydra config
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, cfg=hydra_simulation_config
    )
    
    # Verify config parameters were used
    args, kwargs = mock_simulation_runner.call_args
    assert kwargs["num_steps"] == 100  # From config
    assert kwargs["dt"] == 0.1  # From config
    assert kwargs["sensor_distance"] == 5.0  # From config
    assert kwargs["sensor_angle"] == 45.0  # From config
    
    # Validate multi-agent output shapes
    assert positions.shape[0] == 2  # Two agents
    assert orientations.shape[0] == 2
    assert readings.shape[0] == 2


def test_run_plume_simulation_config_with_overrides(mock_simulation_runner, mock_video_capture, mock_file_exists, hydra_simulation_config):
    """Test simulation with configuration overrides."""
    navigator = create_navigator(position=(0, 0))
    plume = create_video_plume("test_video.mp4")
    
    # Override some config parameters
    positions, orientations, readings = run_plume_simulation(
        navigator, plume,
        cfg=hydra_simulation_config,
        num_steps=200,  # Override config value
        dt=0.05  # Override config value
    )
    
    # Direct parameters should take precedence
    args, kwargs = mock_simulation_runner.call_args
    assert kwargs["num_steps"] == 200  # Overridden
    assert kwargs["dt"] == 0.05  # Overridden
    assert kwargs["sensor_distance"] == 5.0  # From config


def test_run_plume_simulation_backward_compatibility_step_size(mock_simulation_runner, mock_video_capture, mock_file_exists):
    """Test backward compatibility for step_size parameter."""
    navigator = create_navigator(position=(0, 0))
    plume = create_video_plume("test_video.mp4")
    
    # Use deprecated step_size parameter
    with pytest.warns(DeprecationWarning, match="Parameter 'step_size' is deprecated"):
        positions, orientations, readings = run_plume_simulation(
            navigator, plume, num_steps=50, step_size=0.2
        )
    
    # Should convert step_size to dt
    args, kwargs = mock_simulation_runner.call_args
    assert kwargs["dt"] == 0.2


def test_run_plume_simulation_parameter_validation():
    """Test validation of simulation parameters."""
    navigator = create_navigator(position=(0, 0))
    plume = create_video_plume("test_video.mp4")
    
    # Test invalid num_steps
    with pytest.raises(ConfigurationError, match="num_steps must be a positive integer"):
        run_plume_simulation(navigator, plume, num_steps=0, dt=0.1)
    
    # Test invalid dt
    with pytest.raises(ConfigurationError, match="dt must be a positive number"):
        run_plume_simulation(navigator, plume, num_steps=10, dt=-0.1)


def test_run_plume_simulation_missing_required_parameters():
    """Test error handling for missing required parameters."""
    with pytest.raises((TypeError, SimulationError)):
        run_plume_simulation(None, None, num_steps=5, dt=1.0)


def test_run_plume_simulation_seed_management(mock_simulation_runner, mock_video_capture, mock_file_exists):
    """Test seed management in simulation execution."""
    navigator = create_navigator(position=(0, 0))
    plume = create_video_plume("test_video.mp4")
    
    # Run with specific seed
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=10, dt=0.1, seed=123
    )
    
    # Verify seed was handled in parameters
    args, kwargs = mock_simulation_runner.call_args
    assert kwargs.get("seed") == 123


# =============================================================================
# VISUALIZATION INTEGRATION TESTS
# =============================================================================

@pytest.fixture
def mock_visualize_trajectory():
    """Mock the visualization functions from the new utils structure."""
    with patch('{{cookiecutter.project_slug}}.utils.visualization.visualize_trajectory') as mock_viz:
        yield mock_viz


def test_visualize_simulation_results(mock_visualize_trajectory):
    """Test visualization integration with the new utils structure."""
    # Create synthetic simulation results
    positions = np.array([[[0, 0], [1, 1], [2, 2]]])
    orientations = np.array([[0, 45, 90]])
    
    # Call visualization through the new import structure
    visualize_trajectory(
        positions, orientations, output_path="test_output.png", show_plot=False
    )
    
    # Check that the visualization function was called with correct parameters
    mock_visualize_trajectory.assert_called_once()
    args, kwargs = mock_visualize_trajectory.call_args
    assert np.array_equal(args[0], positions)
    assert np.array_equal(args[1], orientations)
    assert kwargs["output_path"] == "test_output.png"
    assert kwargs["show_plot"] is False


def test_simulation_visualization_class_integration():
    """Test SimulationVisualization class integration."""
    # Test that we can import and instantiate the visualization class
    try:
        viz = SimulationVisualization(headless=True)
        assert viz is not None
        assert hasattr(viz, 'setup_environment')
        assert hasattr(viz, 'update_visualization')
        viz.close()  # Clean up
    except Exception as e:
        pytest.fail(f"SimulationVisualization import/instantiation failed: {e}")


# =============================================================================
# CONFIGURATION VALIDATION AND ERROR HANDLING TESTS
# =============================================================================

def test_configuration_error_handling():
    """Test ConfigurationError exception handling."""
    # Test that ConfigurationError is properly raised and handled
    with pytest.raises(ConfigurationError):
        create_navigator(cfg="invalid_config")


def test_simulation_error_handling():
    """Test SimulationError exception handling.""" 
    # Test invalid navigator type
    with pytest.raises((TypeError, SimulationError)):
        run_plume_simulation("not_a_navigator", "not_a_plume", num_steps=5, dt=1.0)


def test_hydra_config_validation_integration():
    """Test integration with Hydra configuration validation."""
    if not HYDRA_AVAILABLE:
        pytest.skip("Hydra not available for configuration validation testing")
    
    # Test configuration composition and validation
    config_dict = {
        "navigator": {
            "positions": [[0, 0], [10, 10]],
            "max_speeds": [1.0, 1.5]
        },
        "video_plume": {
            "video_path": "test.mp4",
            "flip": True
        }
    }
    
    config = OmegaConf.create(config_dict)
    
    # Test that configuration can be properly composed and used
    try:
        navigator = create_navigator(cfg=config.navigator)
        assert navigator is not None
    except Exception as e:
        pytest.fail(f"Hydra configuration validation failed: {e}")


# =============================================================================
# PARAMETRIZED TESTS FOR COMPREHENSIVE COVERAGE
# =============================================================================

@pytest.mark.parametrize(
    "positions,expected_shape",
    [
        ([(1, 2), (3, 4)], (2, 2)),
        ([1, 2], (1, 2)),
        ((1, 2), (1, 2)),
        (np.array([[1, 2], [3, 4]]), (2, 2)),
    ]
)
def test_create_navigator_position_shapes(positions, expected_shape):
    """Test navigator creation with various position input formats."""
    navigator = create_navigator(positions=positions)
    assert navigator.positions.shape == expected_shape


@pytest.mark.parametrize(
    "invalid_positions",
    [
        np.array([[1], [2]]),  # Wrong shape
        [(1, 2, 3), (4, 5, 6)],  # 3D positions
        ["a", "b"],  # Non-numeric
    ]
)
def test_create_navigator_invalid_positions(invalid_positions):
    """Test error handling for invalid position formats."""
    with pytest.raises((ValueError, ConfigurationError)):
        create_navigator(positions=invalid_positions)


@pytest.mark.parametrize(
    "num_steps,dt,should_pass",
    [
        (100, 0.1, True),  # Valid parameters
        (0, 0.1, False),   # Invalid num_steps
        (100, 0, False),   # Invalid dt
        (-1, 0.1, False),  # Negative num_steps
        (100, -0.1, False), # Negative dt
    ]
)
def test_simulation_parameter_validation(mock_simulation_runner, mock_video_capture, mock_file_exists, num_steps, dt, should_pass):
    """Test simulation parameter validation with various inputs."""
    navigator = create_navigator(position=(0, 0))
    plume = create_video_plume("test_video.mp4")
    
    if should_pass:
        # Should not raise an exception
        run_plume_simulation(navigator, plume, num_steps=num_steps, dt=dt)
        mock_simulation_runner.assert_called_once()
    else:
        # Should raise ConfigurationError
        with pytest.raises(ConfigurationError):
            run_plume_simulation(navigator, plume, num_steps=num_steps, dt=dt)


# =============================================================================
# PERFORMANCE AND INTEGRATION TESTS
# =============================================================================

def test_api_performance_requirements(mock_simulation_runner, mock_video_capture, mock_file_exists):
    """Test that API functions meet performance requirements."""
    import time
    
    # Test navigator creation performance
    start_time = time.time()
    navigator = create_navigator(positions=[(0, 0)] * 10)  # 10 agents
    creation_time = time.time() - start_time
    
    # Should create navigator quickly (< 100ms for 10 agents)
    assert creation_time < 0.1, f"Navigator creation took {creation_time:.3f}s, expected < 0.1s"
    
    # Test video plume creation performance  
    start_time = time.time()
    plume = create_video_plume("test_video.mp4")
    plume_creation_time = time.time() - start_time
    
    # Should create video plume quickly
    assert plume_creation_time < 0.1, f"VideoPlume creation took {plume_creation_time:.3f}s, expected < 0.1s"


def test_multi_agent_scalability(mock_simulation_runner, mock_video_capture, mock_file_exists):
    """Test API scalability with multiple agents."""
    # Test with different numbers of agents
    agent_counts = [1, 5, 10, 50]
    
    for num_agents in agent_counts:
        positions = [(i * 10, i * 10) for i in range(num_agents)]
        navigator = create_navigator(positions=positions)
        
        # Verify correct number of agents
        assert navigator.positions.shape[0] == num_agents
        
        # Test simulation execution
        plume = create_video_plume("test_video.mp4")
        positions, orientations, readings = run_plume_simulation(
            navigator, plume, num_steps=10, dt=0.1
        )
        
        # Verify output shapes
        assert positions.shape[0] == num_agents
        assert orientations.shape[0] == num_agents
        assert readings.shape[0] == num_agents


def test_api_integration_with_external_components():
    """Test API integration with external scientific computing components."""
    # Test NumPy array compatibility
    positions = np.random.random((5, 2)) * 100
    navigator = create_navigator(positions=positions)
    
    # Verify NumPy integration works properly
    assert isinstance(navigator.positions, np.ndarray)
    assert navigator.positions.dtype == np.float64
    
    # Test that positions are properly formatted
    assert navigator.positions.shape == (5, 2)


# =============================================================================
# BACKWARD COMPATIBILITY TESTS
# =============================================================================

def test_legacy_parameter_patterns():
    """Test backward compatibility with legacy parameter patterns."""
    # Test old-style position parameter for single agent
    navigator = create_navigator(position=(10, 20), orientation=45, speed=0.5)
    
    # Should work with traditional patterns
    assert navigator.positions.shape == (1, 2)
    assert navigator.orientations[0] == 45
    assert navigator.speeds[0] == 0.5


def test_mixed_parameter_styles():
    """Test mixing traditional and new parameter styles."""
    # Create with traditional parameters
    navigator1 = create_navigator(position=(0, 0), max_speed=2.0)
    
    # Create with new config-style parameters
    navigator2 = create_navigator(positions=[(0, 0)], max_speeds=[2.0])
    
    # Should produce equivalent results
    assert np.allclose(navigator1.positions, navigator2.positions)
    assert np.allclose(navigator1.max_speeds, navigator2.max_speeds)


# =============================================================================
# COMPREHENSIVE EDGE CASE TESTING
# =============================================================================

def test_edge_case_single_step_simulation(mock_simulation_runner, mock_video_capture, mock_file_exists):
    """Test simulation with edge case parameters."""
    navigator = create_navigator(position=(0, 0))
    plume = create_video_plume("test_video.mp4")
    
    # Test single-step simulation
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=1, dt=1.0
    )
    
    # Should handle single step correctly
    args, kwargs = mock_simulation_runner.call_args
    assert kwargs["num_steps"] == 1
    assert kwargs["dt"] == 1.0


def test_zero_agent_edge_case():
    """Test handling of edge cases with no agents."""
    # Test with empty positions (should raise error)
    with pytest.raises((ValueError, ConfigurationError)):
        create_navigator(positions=[])


def test_configuration_override_precedence(hydra_navigator_config):
    """Test precedence of configuration overrides."""
    # Test that direct parameters override config values
    navigator = create_navigator(
        cfg=hydra_navigator_config,
        max_speeds=[10.0, 12.0],  # Override config values
        speeds=[1.0, 1.5]  # Override config values
    )
    
    # Direct parameters should take precedence
    assert np.allclose(navigator.max_speeds, [10.0, 12.0])
    assert np.allclose(navigator.speeds, [1.0, 1.5])
    
    # Non-overridden parameters should come from config
    assert np.allclose(navigator.orientations, [45.0, 90.0])


# =============================================================================
# DOCUMENTATION AND TYPE SAFETY TESTS
# =============================================================================

def test_api_function_signatures():
    """Test that API functions have correct signatures and type hints."""
    import inspect
    from typing import get_type_hints
    
    # Test create_navigator signature
    sig = inspect.signature(create_navigator)
    assert 'cfg' in sig.parameters
    assert 'positions' in sig.parameters
    assert 'seed' in sig.parameters
    
    # Test create_video_plume signature
    sig = inspect.signature(create_video_plume)
    assert 'cfg' in sig.parameters
    assert 'video_path' in sig.parameters
    
    # Test run_plume_simulation signature
    sig = inspect.signature(run_plume_simulation)
    assert 'navigator' in sig.parameters
    assert 'video_plume' in sig.parameters
    assert 'cfg' in sig.parameters


def test_return_type_validation():
    """Test that API functions return expected types."""
    # Test navigator creation return type
    navigator = create_navigator(position=(0, 0))
    assert isinstance(navigator, NavigatorProtocol)
    
    # Test video plume creation return type (with mocks)
    with patch('cv2.VideoCapture'), patch.object(Path, 'exists', return_value=True):
        plume = create_video_plume("test_video.mp4")
        assert isinstance(plume, VideoPlume)


if __name__ == "__main__":
    # Allow running tests directly for development
    pytest.main([__file__, "-v"])