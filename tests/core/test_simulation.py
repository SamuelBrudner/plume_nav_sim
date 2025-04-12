"""Tests for the simulation module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from odor_plume_nav.core import Navigator, run_simulation
from odor_plume_nav.environments import VideoPlume


@pytest.fixture
def mock_navigator():
    """Create a mock Navigator instance."""
    mock_nav = MagicMock(spec=Navigator)
    
    # Configure the mock for a single agent
    mock_nav.is_single_agent = True
    mock_nav.num_agents = 1
    mock_nav.positions = np.array([[0.0, 0.0]])
    mock_nav.orientations = np.array([0.0])
    mock_nav.speeds = np.array([1.0])
    
    # Mock the step method
    def mock_step(env_array):
        # Simulate movement along the x-axis
        mock_nav.positions[0, 0] += mock_nav.speeds[0]
    
    # Mock the sample_odor method
    def mock_sample_odor(env_array):
        # Return a constant odor value
        return np.array([0.5])  # Return as array for consistent API
    
    mock_nav.step.side_effect = mock_step
    mock_nav.sample_odor.side_effect = mock_sample_odor
    
    return mock_nav


@pytest.fixture
def mock_plume():
    """Create a mock VideoPlume instance."""
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


def test_run_simulation_single_agent(mock_navigator, mock_plume):
    """Test running a simulation with a single agent."""
    # Run the simulation
    num_steps = 10
    positions, orientations, odor_readings = run_simulation(
        mock_navigator,
        mock_plume,
        num_steps=num_steps,
        step_size=0.5
    )
    
    # Check that the output has the correct shape
    assert positions.shape == (1, num_steps + 1, 2)  # (num_agents, num_steps + 1, 2)
    assert orientations.shape == (1, num_steps + 1)  # (num_agents, num_steps + 1)
    assert odor_readings.shape == (1, num_steps + 1)  # (num_agents, num_steps + 1)
    
    # Check that step was called the correct number of times
    assert mock_navigator.step.call_count == num_steps
    
    # Check that sample_odor was called the correct number of times
    assert mock_navigator.sample_odor.call_count == num_steps + 1


@pytest.fixture
def mock_multi_navigator():
    """Create a mock Navigator instance for multiple agents."""
    mock_nav = MagicMock(spec=Navigator)
    
    # Configure the mock for multiple agents
    mock_nav.is_single_agent = False
    mock_nav.num_agents = 2
    mock_nav.positions = np.array([[0.0, 0.0], [10.0, 10.0]])
    mock_nav.orientations = np.array([0.0, 90.0])
    mock_nav.speeds = np.array([1.0, 0.5])
    
    # Mock the step method
    def mock_step(env_array):
        # Simulate movement along respective axes
        mock_nav.positions[0, 0] += mock_nav.speeds[0]  # Agent 1 along x-axis
        mock_nav.positions[1, 1] += mock_nav.speeds[1]  # Agent 2 along y-axis
    
    # Mock the sample_odor method
    def mock_sample_odor(env_array):
        # Return different odor values for each agent as numpy array
        return np.array([0.5, 0.7])
    
    mock_nav.step.side_effect = mock_step
    mock_nav.sample_odor.side_effect = mock_sample_odor
    
    return mock_nav


def test_run_simulation_multi_agent(mock_multi_navigator, mock_plume):
    """Test running a simulation with multiple agents."""
    # Run the simulation
    num_steps = 10
    positions, orientations, odor_readings = run_simulation(
        mock_multi_navigator,
        mock_plume,
        num_steps=num_steps,
        step_size=0.5
    )
    
    # Check that the output has the correct shape
    assert positions.shape == (2, num_steps + 1, 2)  # (num_agents, num_steps + 1, 2)
    assert orientations.shape == (2, num_steps + 1)  # (num_agents, num_steps + 1)
    assert odor_readings.shape == (2, num_steps + 1)  # (num_agents, num_steps + 1)
    
    # Check that step was called the correct number of times
    assert mock_multi_navigator.step.call_count == num_steps
    
    # Check that sample_odor was called the correct number of times
    assert mock_multi_navigator.sample_odor.call_count == num_steps + 1


def test_run_simulation_with_custom_sensors(mock_navigator, mock_plume):
    """Test running a simulation with custom sensor parameters."""
    # Run the simulation with custom sensor parameters
    num_steps = 5
    positions, orientations, odor_readings = run_simulation(
        mock_navigator,
        mock_plume,
        num_steps=num_steps,
        step_size=0.5,
        sensor_distance=10.0,
        sensor_angle=60.0
    )
    
    # We can't easily test that the sensor parameters affected the outcome
    # without more complex mocking, but we can check that the simulation ran
    assert positions.shape == (1, num_steps + 1, 2)
    assert orientations.shape == (1, num_steps + 1)
    assert odor_readings.shape == (1, num_steps + 1)


def test_run_simulation_with_custom_step_size(mock_navigator, mock_plume):
    """Test simulation with different step configurations."""
    positions_few_steps = (
        _extracted_from_test_run_simulation_with_custom_step_size_(
            mock_navigator, 3, mock_plume
        )
    )
    positions_many_steps = (
        _extracted_from_test_run_simulation_with_custom_step_size_(
            mock_navigator, 6, mock_plume
        )
    )
    # With more steps, the agent should move farther
    assert positions_many_steps[0, -1, 0] > positions_few_steps[0, -1, 0]


# TODO Rename this here and in `test_run_simulation_with_custom_step_size`
def _extracted_from_test_run_simulation_with_custom_step_size_(mock_navigator, arg1, mock_plume):
    mock_navigator.positions = np.array([[0.0, 0.0]])
    num_steps_few = arg1
    result, _, _ = run_simulation(
        mock_navigator, mock_plume, num_steps=num_steps_few
    )

    return result
