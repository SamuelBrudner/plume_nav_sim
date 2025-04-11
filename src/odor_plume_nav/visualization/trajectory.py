"""
Trajectory visualization module for odor plume navigation.

This module provides functions for visualizing agent trajectories
from simulation results.
"""

from typing import Optional, Union, List, Tuple
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def visualize_trajectory(
    positions: np.ndarray, 
    orientations: Optional[np.ndarray] = None,
    plume_frames: Optional[np.ndarray] = None,
    output_path: Optional[Union[str, pathlib.Path]] = None,
    show_plot: bool = True,
    agent_colors: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,  # Alternate parameter name for backward compatibility
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300,
):
    """
    Visualize agent trajectory from simulation results.
    
    Args:
        positions: Array of shape (num_agents, time_steps, 2) or (time_steps, 2) for positions
        orientations: Array of shape (num_agents, time_steps) or (time_steps,) for orientations
        plume_frames: Optional array of plume frames to show in the background
        output_path: Optional path to save the visualization
        show_plot: Whether to display the plot
        agent_colors: Optional list of colors for each agent
        colors: Alternative parameter name for agent_colors (for backward compatibility)
        title: Optional title for the plot
        figsize: Figure size as (width, height) in inches
        dpi: DPI for saved figure
        
    Returns:
        None
    """
    # Use colors if agent_colors is not provided
    if agent_colors is None and colors is not None:
        agent_colors = colors
    
    # Reshape positions to ensure 3D format (num_agents, time_steps, 2)
    if len(positions.shape) == 2:
        # Single agent case - reshape to (1, time_steps, 2)
        positions = np.expand_dims(positions, axis=0)
    
    # Transpose if in (time_steps, num_agents, 2) format to (num_agents, time_steps, 2)
    if positions.shape[2] == 2 and positions.shape[0] < positions.shape[1]:
        positions = np.transpose(positions, (1, 0, 2))
    
    # Get the number of agents
    num_agents = positions.shape[0]
    
    # Set default colors if not provided
    if agent_colors is None:
        # Use tab10 colormap for up to 10 agents
        agent_colors = [f'C{i}' for i in range(num_agents)]
    
    # Reshape orientations if provided
    if orientations is not None:
        if len(orientations.shape) == 1:
            # Single agent case - reshape to (1, time_steps)
            orientations = np.expand_dims(orientations, axis=0)
        # Transpose if in (time_steps, num_agents) format to (num_agents, time_steps)
        elif orientations.shape[0] < orientations.shape[1]:
            orientations = orientations.T
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot the background plume if provided
    if plume_frames is not None:
        # Convert to numpy array if not already
        plume_array = np.array(plume_frames)
        
        # Ensure plume_frames is at least 2D
        if len(plume_array.shape) < 2:
            raise TypeError(f"Invalid shape {plume_array.shape} for image data")
            
        # Use the first frame as background if multiple frames
        if len(plume_array.shape) > 2 and plume_array.shape[0] > 1:
            plume_array = plume_array[0]
            
        # Display the plume
        plt.imshow(plume_array, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Odor concentration')
    
    # Plot each agent's trajectory
    for i in range(num_agents):
        agent_positions = positions[i]
        color = agent_colors[i % len(agent_colors)]  # Cycle through colors if needed
        
        # Plot the trajectory
        plt.plot(agent_positions[:, 0], agent_positions[:, 1], 
                color=color, alpha=0.7, linewidth=2, label=f'Agent {i+1}')
        
        # Mark the start and end points
        plt.scatter(agent_positions[0, 0], agent_positions[0, 1], 
                  color=color, marker='o', s=100)
        plt.scatter(agent_positions[-1, 0], agent_positions[-1, 1], 
                  color=color, marker='x', s=100)
        
        # Plot orientations if provided
        if orientations is not None:
            agent_orientations = orientations[i]
            # Convert orientations from degrees to radians for quiver
            orient_rad = np.deg2rad(agent_orientations)
            # Calculate direction vectors
            u = np.cos(orient_rad)
            v = np.sin(orient_rad)
            # Plot orientation arrows at selected positions
            step = max(1, len(agent_positions) // 10)  # Show arrows at 10% of points
            plt.quiver(agent_positions[::step, 0], agent_positions[::step, 1], 
                     u[::step], v[::step], color=color, alpha=0.6, scale=20)
    
    # Set plot title and labels
    if title:
        plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot if an output path is provided
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
