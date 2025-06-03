"""Tests for the simulation module with enhanced Hydra integration and new package structure.

This module provides comprehensive testing for simulation functionality including:
- Core simulation execution with updated module imports
- Hydra configuration integration for DictConfig parameter injection
- Enhanced mock fixtures for separated navigator and controllers modules
- VideoPlume factory method validation with Hydra configuration
- CLI integration scenarios through simulation execution
- Multi-agent simulation scenarios with new controller architecture
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

# Updated import statements per new package structure
from {{cookiecutter.project_slug}}.api.navigation import run_plume_simulation
from {{cookiecutter.project_slug}}.core.navigator import NavigatorProtocol
from {{cookiecutter.project_slug}}.core.controllers import SingleAgentController, MultiAgentController
from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
from {{cookiecutter.project_slug}}.config.schemas import SimulationConfig, NavigatorConfig

# Hydra configuration integration imports
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict
    OmegaConf = None
    HYDRA_AVAILABLE = False


@pytest.fixture
def mock_navigator():
    """Create a mock Navigator instance with updated module structure."""
    # Mock based on NavigatorProtocol interface for new structure
    mock_nav = MagicMock(spec=NavigatorProtocol)
    
    # Configure the mock for a single agent using observable properties
    # rather than internal implementation details
    mock_nav.num_agents = 1
    mock_nav.positions = np.array([[0.0, 0.0]])
    mock_nav.orientations = np.array([0.0])
    mock_nav.speeds = np.array([1.0])
    
    # Mock the step method
    def mock_step(env_array, dt=1.0):
        # Simulate movement along the x-axis with dt parameter
        mock_nav.positions[0, 0] += mock_nav.speeds[0] * dt
    
    # Mock the sample_odor method
    def mock_sample_odor(env_array):
        # Return a constant odor value
        return np.array([0.5])  # Return as array for consistent API
    
    mock_nav.step.side_effect = mock_step
    mock_nav.sample_odor.side_effect = mock_sample_odor
    
    return mock_nav


@pytest.fixture
def mock_video_plume():
    """Create a mock VideoPlume instance with enhanced factory method support."""
    mock_plume = MagicMock(spec=VideoPlume)
    
    # Configure the mock for a synthetic video
    mock_plume.frame_count = 100
    mock_plume.width = 640
    mock_plume.height = 480
    
    # Mock the get_frame method
    def mock_get_frame(frame_idx):
        # Return a synthetic frame with a gradient
        frame = np.zeros((480, 640), dtype=np.uint8)
        for i in range(480):
            for j in range(640):
                frame[i, j] = (i + j) % 256
        return frame
    
    mock_plume.get_frame.side_effect = mock_get_frame
    
    return mock_plume


@pytest.fixture
def mock_multi_navigator():
    """Create a mock Navigator instance for multiple agents with new controller structure."""
    mock_nav = MagicMock(spec=NavigatorProtocol)
    
    # Configure the mock for multiple agents using observable properties
    # rather than internal implementation details
    mock_nav.num_agents = 2
    mock_nav.positions = np.array([[0.0, 0.0], [10.0, 10.0]])
    mock_nav.orientations = np.array([0.0, 90.0])
    mock_nav.speeds = np.array([1.0, 0.5])
    
    # Mock the step method
    def mock_step(env_array, dt=1.0):
        # Simulate movement along respective axes
        mock_nav.positions[0, 0] += mock_nav.speeds[0] * dt  # Agent 1 along x-axis
        mock_nav.positions[1, 1] += mock_nav.speeds[1] * dt  # Agent 2 along y-axis
    
    # Mock the sample_odor method
    def mock_sample_odor(env_array):
        # Return different odor values for each agent as numpy array
        return np.array([0.5, 0.7])
    
    mock_nav.step.side_effect = mock_step
    mock_nav.sample_odor.side_effect = mock_sample_odor
    
    return mock_nav


@pytest.fixture
def sample_hydra_config():
    """Provide a sample Hydra DictConfig for simulation testing."""
    if HYDRA_AVAILABLE:
        config_dict = {
            "num_steps": 10,
            "dt": 0.5,
            "sensor_distance": 5.0,
            "sensor_angle": 45.0,
            "record_trajectory": True,
            "seed": 42
        }
        return OmegaConf.create(config_dict)
    else:
        # Fallback for environments without Hydra
        return {
            "num_steps": 10,
            "dt": 0.5,
            "sensor_distance": 5.0,
            "sensor_angle": 45.0,
            "record_trajectory": True,
            "seed": 42
        }


@pytest.fixture
def navigator_hydra_config():
    """Provide a sample Navigator configuration via Hydra DictConfig."""
    if HYDRA_AVAILABLE:
        config_dict = {
            "type": "single",
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0,
            "angular_velocity": 0.1
        }
        return OmegaConf.create(config_dict)
    else:
        return {
            "type": "single",
            "position": [0.0, 0.0],
            "orientation": 0.0,
            "speed": 1.0,
            "max_speed": 2.0,
            "angular_velocity": 0.1
        }


def test_run_plume_simulation_single_agent(mock_navigator, mock_video_plume):
    """Test running a simulation with a single agent using updated function name."""
    # Run the simulation with updated function name
    num_steps = 10
    positions, orientations, odor_readings = run_plume_simulation(
        mock_navigator,
        mock_video_plume,
        num_steps=num_steps,
        dt=0.5
    )
    
    # Check that the output has the correct shape
    assert positions.shape == (1, num_steps + 1, 2)  # (num_agents, num_steps + 1, 2)
    assert orientations.shape == (1, num_steps + 1)  # (num_agents, num_steps + 1)
    assert odor_readings.shape == (1, num_steps + 1)  # (num_agents, num_steps + 1)
    
    # Check that step was called the correct number of times
    assert mock_navigator.step.call_count == num_steps
    
    # Check that sample_odor was called the correct number of times
    assert mock_navigator.sample_odor.call_count == num_steps + 1


def test_run_plume_simulation_multi_agent(mock_multi_navigator, mock_video_plume):
    """Test running a simulation with multiple agents using updated function name."""
    # Run the simulation with updated function name
    num_steps = 10
    positions, orientations, odor_readings = run_plume_simulation(
        mock_multi_navigator,
        mock_video_plume,
        num_steps=num_steps,
        dt=0.5
    )
    
    # Check that the output has the correct shape
    assert positions.shape == (2, num_steps + 1, 2)  # (num_agents, num_steps + 1, 2)
    assert orientations.shape == (2, num_steps + 1)  # (num_agents, num_steps + 1)
    assert odor_readings.shape == (2, num_steps + 1)  # (num_agents, num_steps + 1)
    
    # Check that step was called the correct number of times
    assert mock_multi_navigator.step.call_count == num_steps
    
    # Check that sample_odor was called the correct number of times
    assert mock_multi_navigator.sample_odor.call_count == num_steps + 1


def test_run_plume_simulation_with_hydra_config(mock_navigator, mock_video_plume, sample_hydra_config):
    """Test running simulation with Hydra DictConfig parameter injection."""
    # Run simulation using Hydra configuration
    positions, orientations, odor_readings = run_plume_simulation(
        mock_navigator,
        mock_video_plume,
        cfg=sample_hydra_config
    )
    
    # Verify configuration parameters were applied
    num_steps = sample_hydra_config["num_steps"] if isinstance(sample_hydra_config, dict) else sample_hydra_config.num_steps
    assert positions.shape == (1, num_steps + 1, 2)
    assert orientations.shape == (1, num_steps + 1)
    assert odor_readings.shape == (1, num_steps + 1)
    
    # Verify step method called with correct dt parameter
    assert mock_navigator.step.call_count == num_steps


def test_run_plume_simulation_with_config_overrides(mock_navigator, mock_video_plume, sample_hydra_config):
    """Test simulation with DictConfig and parameter overrides."""
    # Override some parameters while using Hydra config
    positions, orientations, odor_readings = run_plume_simulation(
        mock_navigator,
        mock_video_plume,
        cfg=sample_hydra_config,
        num_steps=5,  # Override the config value
        dt=1.0  # Override the config value
    )
    
    # Check that overrides took precedence
    assert positions.shape == (1, 6, 2)  # 5 steps + 1 initial = 6 timesteps
    assert mock_navigator.step.call_count == 5


def test_run_plume_simulation_with_custom_sensors(mock_navigator, mock_video_plume):
    """Test running a simulation with custom sensor parameters."""
    # Run the simulation with custom sensor parameters
    num_steps = 5
    positions, orientations, odor_readings = run_plume_simulation(
        mock_navigator,
        mock_video_plume,
        num_steps=num_steps,
        dt=0.5,
        sensor_distance=10.0,
        sensor_angle=60.0
    )
    
    # We can't easily test that the sensor parameters affected the outcome
    # without more complex mocking, but we can check that the simulation ran
    assert positions.shape == (1, num_steps + 1, 2)
    assert orientations.shape == (1, num_steps + 1)
    assert odor_readings.shape == (1, num_steps + 1)


def test_run_plume_simulation_with_seed_management(mock_navigator, mock_video_plume):
    """Test simulation with seed parameter for reproducibility."""
    # Run simulation with seed parameter
    seed_value = 42
    positions1, orientations1, odor_readings1 = run_plume_simulation(
        mock_navigator,
        mock_video_plume,
        num_steps=5,
        seed=seed_value
    )
    
    # Reset navigator state for second run
    mock_navigator.positions = np.array([[0.0, 0.0]])
    mock_navigator.orientations = np.array([0.0])
    
    # Run again with same seed
    positions2, orientations2, odor_readings2 = run_plume_simulation(
        mock_navigator,
        mock_video_plume,
        num_steps=5,
        seed=seed_value
    )
    
    # Results should be identical due to seeding
    np.testing.assert_array_equal(positions1, positions2)
    np.testing.assert_array_equal(orientations1, orientations2)
    np.testing.assert_array_equal(odor_readings1, odor_readings2)


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
def test_video_plume_factory_with_hydra_config():
    """Test VideoPlume factory method with Hydra configuration integration."""
    from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
    
    # Create Hydra config for VideoPlume
    video_config = OmegaConf.create({
        "video_path": "/tmp/test_video.mp4",
        "flip": True,
        "kernel_size": 5,
        "kernel_sigma": 1.5,
        "grayscale": True
    })
    
    # Mock file existence to avoid actual file dependency
    with patch('pathlib.Path.exists', return_value=True), \
         patch('cv2.VideoCapture') as mock_cv2:
        
        # Configure mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [100, 640, 480, 30.0]  # frame_count, width, height, fps
        mock_cv2.return_value = mock_cap
        
        # Test factory method with Hydra config
        video_plume = VideoPlume.from_config(video_config)
        
        # Verify configuration was applied
        assert video_plume.flip == True
        assert video_plume.kernel_size == 5
        assert video_plume.kernel_sigma == 1.5
        assert video_plume.grayscale == True


def test_simulation_config_validation_with_hydra():
    """Test simulation configuration validation using Pydantic schemas with Hydra."""
    from {{cookiecutter.project_slug}}.config.schemas import SimulationConfig
    
    # Test valid configuration
    valid_config = {
        "num_steps": 100,
        "dt": 0.1,
        "sensor_distance": 5.0,
        "sensor_angle": 45.0,
        "record_trajectory": True
    }
    
    # Should not raise validation errors
    sim_config = SimulationConfig.model_validate(valid_config)
    assert sim_config.num_steps == 100
    assert sim_config.dt == 0.1
    
    # Test invalid configuration
    invalid_config = {
        "num_steps": -10,  # Invalid: negative steps
        "dt": 0.0,  # Invalid: zero dt
        "sensor_distance": -5.0,  # Invalid: negative distance
    }
    
    # Should raise validation error
    with pytest.raises(Exception):  # Pydantic validation error
        SimulationConfig.model_validate(invalid_config)


def test_cli_integration_simulation_execution():
    """Test CLI integration scenarios through simulation execution."""
    # This test validates that simulation can be executed through CLI patterns
    # without actually invoking the CLI (which would be tested in tests/cli/)
    
    # Simulate CLI-like parameter processing
    cli_params = {
        "num_steps": 20,
        "dt": 0.2,
        "output_format": "json",
        "verbose": True
    }
    
    # Create mocks that would be created by CLI parameter processing
    mock_nav = MagicMock(spec=NavigatorProtocol)
    mock_nav.num_agents = 1
    mock_nav.positions = np.array([[0.0, 0.0]])
    mock_nav.orientations = np.array([0.0])
    mock_nav.speeds = np.array([1.0])
    
    mock_nav.step.side_effect = lambda env, dt=1.0: None
    mock_nav.sample_odor.return_value = np.array([0.5])
    
    mock_plume = MagicMock(spec=VideoPlume)
    mock_plume.frame_count = 100
    mock_plume.get_frame.return_value = np.zeros((480, 640), dtype=np.uint8)
    
    # Execute simulation with CLI-style parameters
    positions, orientations, odor_readings = run_plume_simulation(
        mock_nav,
        mock_plume,
        num_steps=cli_params["num_steps"],
        dt=cli_params["dt"]
    )
    
    # Verify results match CLI parameters
    expected_timesteps = cli_params["num_steps"] + 1
    assert positions.shape == (1, expected_timesteps, 2)
    assert mock_nav.step.call_count == cli_params["num_steps"]


def test_backwards_compatibility_step_size_parameter(mock_navigator, mock_video_plume):
    """Test backwards compatibility for step_size parameter (now dt)."""
    # Test that old step_size parameter still works but issues deprecation warning
    with pytest.warns(DeprecationWarning, match="step_size.*deprecated.*dt"):
        positions, orientations, odor_readings = run_plume_simulation(
            mock_navigator,
            mock_video_plume,
            num_steps=5,
            step_size=0.5  # Using deprecated parameter name
        )
    
    # Should still work correctly
    assert positions.shape == (1, 6, 2)
    assert mock_navigator.step.call_count == 5


def test_simulation_error_handling(mock_navigator, mock_video_plume):
    """Test simulation error handling and validation."""
    # Test invalid num_steps
    with pytest.raises(Exception):  # Should raise ConfigurationError
        run_plume_simulation(
            mock_navigator,
            mock_video_plume,
            num_steps=-1  # Invalid negative steps
        )
    
    # Test invalid dt
    with pytest.raises(Exception):  # Should raise ConfigurationError
        run_plume_simulation(
            mock_navigator,
            mock_video_plume,
            num_steps=10,
            dt=0.0  # Invalid zero dt
        )


def test_simulation_with_limited_frames(mock_navigator, mock_video_plume):
    """Test simulation behavior when video has fewer frames than requested steps."""
    # Simulate video with limited frames
    mock_video_plume.frame_count = 5
    
    # Request more steps than available frames
    with pytest.warns(UserWarning, match="only.*frames available"):
        positions, orientations, odor_readings = run_plume_simulation(
            mock_navigator,
            mock_video_plume,
            num_steps=10  # More than available frames
        )
    
    # Should only execute available steps
    assert positions.shape == (1, 5, 2)  # 4 steps + 1 initial = 5 timesteps
    assert mock_navigator.step.call_count == 4  # frame_count - 1