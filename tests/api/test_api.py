"""Tests for the public API functions."""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from pathlib import Path

from odor_plume_nav.api import (
    create_navigator,
    create_video_plume,
    run_plume_simulation,
    visualize_simulation_results,
)


def test_create_navigator_default():
    """Test creating a navigator with default parameters."""
    navigator = create_navigator()
    
    # Should create a default Navigator instance (single agent)
    # Check default values aligned with the protocol-based Navigator
    assert navigator.positions.shape == (1, 2)  # Single agent with 2D position
    assert navigator.orientations.shape == (1,)  # Single agent orientation
    assert navigator.speeds.shape == (1,)  # Single agent speed
    
    # Check default values
    assert navigator.orientations[0] == 0.0
    assert navigator.speeds[0] == 0.0
    assert navigator.max_speeds[0] == 1.0


def test_create_navigator_single_agent():
    """Test creating a navigator with single agent parameters."""
    # Create a navigator with single agent parameters
    navigator = create_navigator(
        positions=(10, 20),
        orientations=45,
        speeds=0.5,
        max_speeds=2.0
    )
    
    # Check that the navigator has the correct properties
    # In the protocol-based architecture, properties are array-based
    assert navigator.orientations[0] == 45
    assert navigator.speeds[0] == 0.5
    assert navigator.max_speeds[0] == 2.0
    assert np.allclose(navigator.positions[0], [10, 20])
    
    # Check that it's a single-agent navigator by verifying array lengths
    assert len(navigator.positions) == 1
    assert len(navigator.orientations) == 1
    assert len(navigator.speeds) == 1


def test_create_navigator_multi_agent():
    """Test creating a navigator with multi-agent parameters."""
    # Create a navigator with multi-agent parameters
    positions = [(10, 20), (30, 40), (50, 60)]
    orientations = [45, 90, 135]
    speeds = [0.5, 0.7, 0.9]
    
    navigator = create_navigator(
        positions=positions,
        orientations=orientations,
        speeds=speeds
    )
    
    # Check that the navigator has the correct number of agents
    assert len(navigator.positions) == 3
    
    # Check the properties of the first agent using array indexing
    assert navigator.orientations[0] == 45
    assert navigator.speeds[0] == 0.5
    
    # Check that all agents have correct positions
    for i, expected_pos in enumerate(positions):
        assert np.allclose(navigator.positions[i], expected_pos)
    
    # Verify each agent has correct orientation and speed
    assert np.allclose(navigator.orientations, orientations)
    assert np.allclose(navigator.speeds, speeds)


def test_create_navigator_numpy_array_positions():
    """Test creating a navigator with numpy array positions."""
    # Test with numpy array position data
    positions = np.array([[10, 20], [30, 40], [50, 60]])
    
    navigator = create_navigator(positions=positions)
    
    # Check it's a multi-agent navigator with the right number of agents
    assert len(navigator.positions) == 3
    
    # Verify positions were set correctly
    assert np.allclose(navigator.positions, positions)
    
    # Check default values for other properties
    assert np.allclose(navigator.orientations, np.zeros(3))
    assert np.allclose(navigator.speeds, np.zeros(3))


@pytest.fixture
def mock_config_load():
    """Mock the config loading function."""
    with patch('odor_plume_nav.config.utils.load_config') as mock_load:
        # Create valid config with numpy arrays since that's what the validator expects
        positions = np.array([[10, 20], [30, 40]])
        orientations = np.array([45, 90])
        speeds = np.array([0.5, 0.7])
        max_speeds = np.array([1.0, 1.0])
        
        mock_load.return_value = {
            "positions": positions,
            "orientations": orientations,
            "speeds": speeds,
            "max_speeds": max_speeds,
            "video_plume": {
                "flip": True,
                "kernel_size": 5,
                "kernel_sigma": 1.0
            }
        }
        yield mock_load


def test_create_navigator_from_config(mock_config_load):
    """Test creating a navigator from a configuration file."""
    # Create a navigator from a configuration file
    navigator = create_navigator(config_path="test_config.yaml")
    
    # Verify config was loaded
    mock_config_load.assert_called_once_with("test_config.yaml")
    
    # Check that the navigator has the correct properties with multi-agent protocol
    assert len(navigator.positions) == 2  # Two agents from config
    assert navigator.orientations[0] == 45  # First agent's orientation
    assert navigator.speeds[0] == 0.5  # First agent's speed
    assert np.allclose(navigator.positions[0], [10, 20])  # First agent's position


def test_create_navigator_from_config_single_agent(mock_config_load):
    """Test creating a single-agent navigator from config."""
    # Override the mock to return a single-agent config
    mock_config_load.return_value = {
        "position": (10, 20),
        "orientation": 45,
        "speed": 0.5,
        "max_speed": 1.0,
        "video_plume": {
            "flip": True,
            "kernel_size": 5,
            "kernel_sigma": 1.0
        }
    }
    
    # Create a navigator from the single-agent config
    navigator = create_navigator(config_path="single_agent_config.yaml")
    
    # Verify config was loaded
    mock_config_load.assert_called_once_with("single_agent_config.yaml")
    
    # Check that the navigator has the correct properties with single-agent protocol
    assert len(navigator.positions) == 1  # Single agent
    assert navigator.orientations[0] == 45
    assert navigator.speeds[0] == 0.5
    assert navigator.max_speeds[0] == 1.0
    assert np.allclose(navigator.positions[0], [10, 20])


@pytest.fixture
def mock_video_capture():
    """Create a mock for cv2.VideoCapture."""
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
def mock_exists(monkeypatch):
    """Mock the Path.exists method to return True for all paths."""
    def patched_exists(self):
        return True
    
    monkeypatch.setattr(Path, "exists", patched_exists)
    return patched_exists


def test_create_video_plume(mock_video_capture, mock_exists):
    """Test creating a video plume with the API function."""
    # Create a video plume
    plume = create_video_plume("test_video.mp4", flip=True, kernel_size=5)
    
    # Check that the plume has the correct properties
    assert plume.video_path == Path("test_video.mp4")
    assert plume.flip is True
    assert plume.kernel_size == 5


def test_create_video_plume_with_config(mock_video_capture, mock_exists, mock_config_load):
    """Test creating a video plume with a configuration file."""
    # Create a video plume from a configuration file
    plume = create_video_plume("test_video.mp4", config_path="test_config.yaml")
    
    # Verify config was loaded
    mock_config_load.assert_called_once_with("test_config.yaml")


@pytest.fixture
def mock_run_simulation():
    """Mock the run_simulation function."""
    # We need to patch the function where it's imported, not where it's defined
    with patch('odor_plume_nav.api.run_simulation') as mock_run:
        # Configure mock to return synthetic data
        positions_history = np.array([[[0, 0], [1, 1], [2, 2]]])
        orientations_history = np.array([[0, 45, 90]])
        odor_readings = np.array([[0.1, 0.2, 0.3]])
        
        mock_run.return_value = (positions_history, orientations_history, odor_readings)
        yield mock_run


def test_run_plume_simulation(mock_run_simulation, mock_video_capture, mock_exists):
    """Test running a plume simulation with the API function."""
    # Create a navigator and video plume
    navigator = create_navigator(positions=(10, 20), orientations=45)
    plume = create_video_plume("test_video.mp4")
    
    # Run the simulation
    positions, orientations, readings = run_plume_simulation(
        navigator, plume, num_steps=100, step_size=0.5
    )
    
    # Check that the simulation function was called with the correct parameters
    mock_run_simulation.assert_called_once()
    args, kwargs = mock_run_simulation.call_args
    assert args[0] == navigator
    assert args[1] == plume
    assert kwargs["num_steps"] == 100
    assert kwargs["step_size"] == 0.5
    
    # Check that the results were returned correctly
    assert positions.shape == (1, 3, 2)  # (num_agents, num_steps, 2)
    assert orientations.shape == (1, 3)  # (num_agents, num_steps)
    assert readings.shape == (1, 3)    # (num_agents, num_steps)


@pytest.fixture
def mock_visualize_trajectory():
    """Mock the visualize_trajectory function."""
    with patch('odor_plume_nav.visualization.trajectory.visualize_trajectory') as mock_viz:
        yield mock_viz


def test_visualize_simulation_results(mock_visualize_trajectory):
    """Test visualizing simulation results with the API function."""
    # Create synthetic simulation results
    positions = np.array([[[0, 0], [1, 1], [2, 2]]])
    orientations = np.array([[0, 45, 90]])
    
    # Visualize the results
    visualize_simulation_results(
        positions, orientations, output_path="test_output.png", show_plot=False
    )
    
    # Check that the visualization function was called with the correct parameters
    mock_visualize_trajectory.assert_called_once()
    args, kwargs = mock_visualize_trajectory.call_args
    assert np.array_equal(args[0], positions)
    assert np.array_equal(args[1], orientations)
    assert kwargs["output_path"] == "test_output.png"
    assert kwargs["show_plot"] is False
