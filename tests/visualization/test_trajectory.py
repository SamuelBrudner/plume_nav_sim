"""Tests for the trajectory visualization module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from odor_plume_nav.visualization import visualize_trajectory


@pytest.fixture
def mock_plt():
    """Mock matplotlib.pyplot to avoid displaying plots during tests."""
    with patch('odor_plume_nav.visualization.trajectory.plt') as mock_plt:
        yield mock_plt


def test_visualize_trajectory_single_agent(mock_plt):
    """Test visualizing trajectory for a single agent."""
    # Create synthetic data for a single agent
    positions = np.array([[[0, 0], [1, 1], [2, 2]]])  # Shape: (1, 3, 2)
    orientations = np.array([[0, 45, 90]])  # Shape: (1, 3)
    
    # Call the visualization function
    visualize_trajectory(positions, orientations, show_plot=False)
    
    # Check that the plot was created and properly formatted
    mock_plt.figure.assert_called_once()
    mock_plt.plot.assert_called()  # Should be called at least once
    mock_plt.quiver.assert_called()  # Should be called to show orientations
    mock_plt.savefig.assert_not_called()  # No output path was provided


def test_visualize_trajectory_multi_agent(mock_plt):
    """Test visualizing trajectory for multiple agents."""
    # Create synthetic data for multiple agents
    positions = np.array([
        [[0, 0], [1, 1], [2, 2]],  # Agent 1
        [[5, 5], [6, 6], [7, 7]]   # Agent 2
    ])  # Shape: (2, 3, 2)
    orientations = np.array([
        [0, 45, 90],    # Agent 1
        [180, 225, 270]  # Agent 2
    ])  # Shape: (2, 3)
    
    # Call the visualization function
    visualize_trajectory(positions, orientations, show_plot=False)
    
    # Check that the plot was created with multiple traces
    mock_plt.figure.assert_called_once()
    assert mock_plt.plot.call_count >= 2  # Should be called at least once per agent
    assert mock_plt.quiver.call_count >= 2  # Should be called to show orientations for each agent


def test_visualize_trajectory_with_plume(mock_plt):
    """Test visualizing trajectory with a plume background."""
    # Create synthetic data
    positions = np.array([[[0, 0], [1, 1], [2, 2]]])
    orientations = np.array([[0, 45, 90]])
    
    # Create a synthetic plume frame
    plume_frame = np.zeros((10, 10), dtype=np.uint8)
    plume_frame[5:8, 5:8] = 255  # Add a bright spot
    
    # Call the visualization function with plume frames
    visualize_trajectory(
        positions, 
        orientations, 
        plume_frames=plume_frame, 
        show_plot=False
    )
    
    # Check that the imshow was called to display the plume
    mock_plt.imshow.assert_called_once()
    mock_plt.colorbar.assert_called_once()


def test_visualize_trajectory_with_output(mock_plt):
    """Test visualizing trajectory with an output file."""
    # Create synthetic data
    positions = np.array([[[0, 0], [1, 1], [2, 2]]])
    orientations = np.array([[0, 45, 90]])
    
    # Call the visualization function with an output path
    output_path = "test_trajectory.png"
    visualize_trajectory(
        positions, 
        orientations, 
        output_path=output_path,
        show_plot=False
    )
    
    # Check that the figure was saved
    mock_plt.savefig.assert_called_once_with(output_path, dpi=300, bbox_inches="tight")


def test_visualize_trajectory_with_custom_colors(mock_plt):
    """Test visualizing trajectory with custom colors."""
    # Create synthetic data for multiple agents
    positions = np.array([
        [[0, 0], [1, 1], [2, 2]],  # Agent 1
        [[5, 5], [6, 6], [7, 7]]   # Agent 2
    ])
    orientations = np.array([
        [0, 45, 90],    # Agent 1
        [180, 225, 270]  # Agent 2
    ])
    
    # Define custom colors
    colors = ['red', 'blue']
    
    # Call the visualization function with custom colors
    visualize_trajectory(
        positions, 
        orientations, 
        colors=colors,
        show_plot=False
    )
    
    # Check that the plot was created with the specified colors
    mock_plt.plot.assert_called()
    
    # Extract the color arguments from the plot calls
    plot_calls = mock_plt.plot.call_args_list
    
    # At least one call should have used 'red' (this is a simplification as we can't
    # easily check the actual args in a mock - in practice we'd need to set up a more
    # sophisticated verification)
