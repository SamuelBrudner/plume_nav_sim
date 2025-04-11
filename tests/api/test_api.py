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
    
    # Should create a default Navigator instance
    assert navigator._single_agent is True
    # Check default values from Navigator class
    assert navigator.orientation == 0.0
    assert navigator.speed == 0.0
    assert navigator.max_speed == 1.0


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
    assert navigator.orientation == 45
    assert navigator.speed == 0.5
    assert navigator.max_speed == 2.0
    assert navigator.get_position() == (10, 20)
    
    # Check that it's in single-agent mode
    assert navigator._single_agent is True


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
    assert navigator.num_agents == 3
    
    # Check the properties of the first agent
    assert navigator.orientation == 45
    assert navigator.speed == 0.5
    
    # Check that all agents have correct positions
    for i, expected_pos in enumerate(positions):
        assert np.allclose(navigator.positions[i], expected_pos)
    
    # Check that it's not in single-agent mode
    assert navigator._single_agent is False


def test_create_navigator_numpy_array_positions():
    """Test creating a navigator with numpy array positions."""
    # Test with numpy array position data
    positions = np.array([[10, 20], [30, 40], [50, 60]])
    
    navigator = create_navigator(positions=positions)
    
    assert navigator._single_agent is False
    assert navigator.num_agents == 3
    assert np.allclose(navigator.positions, positions)


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
    
    # Check that the navigator has the correct properties
    assert navigator.num_agents == 2
    assert navigator.orientation == 45
    assert navigator.speed == 0.5
    assert np.allclose(navigator.positions[0], [10, 20])


def test_create_navigator_from_config_single_agent(mock_config_load):
    """Test creating a single-agent navigator from config."""
    # Modify the mock to return a single agent config
    mock_config_load.return_value = {
        "position": (5, 10),
        "orientation": 30,
        "speed": 0.8,
        "max_speed": 2.0,
        "video_plume": {
            "flip": False
        }
    }
    
    navigator = create_navigator(config_path="single_agent_config.yaml")
    
    # Verify single agent mode
    assert navigator._single_agent is True
    assert navigator.get_position() == (5, 10)
    assert navigator.orientation == 30
    assert navigator.speed == 0.8


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
