"""
Simulation module for odor plume navigation.

This module provides functions for running simulations with agents
navigating through plume environments.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np

from odor_plume_nav.core.navigator import Navigator
from odor_plume_nav.environments.video_plume import VideoPlume


def run_simulation(
    navigator: Navigator,
    video_plume: VideoPlume,
    num_steps: int,
    step_size: float = 1.0,
    sensor_distance: float = 5.0,
    sensor_angle: float = 45.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a simulation of agent(s) navigating through an odor plume.
    
    Args:
        navigator: Navigator instance controlling agent(s)
        video_plume: VideoPlume instance providing the environment
        num_steps: Number of simulation steps to run
        step_size: Size of each simulation step
        sensor_distance: Distance of the sensors from the agent
        sensor_angle: Angle of the sensors relative to the agent's orientation
        **kwargs: Additional parameters for the simulation
        
    Returns:
        Tuple of (positions_history, orientations_history, odor_readings)
    """
    # Determine if we're working with a single agent or multiple
    num_agents = navigator.num_agents
    
    # Initialize arrays to store simulation history
    positions_history = np.zeros((num_agents, num_steps + 1, 2))
    orientations_history = np.zeros((num_agents, num_steps + 1))
    odor_readings = np.zeros((num_agents, num_steps + 1))
    
    # Store initial state
    positions_history[:, 0] = navigator.positions
    orientations_history[:, 0] = navigator.orientations
    
    # Get initial odor readings
    current_frame = video_plume.get_frame(0)
    odor_readings[:, 0] = navigator.sample_odor(current_frame)
    
    # Run simulation steps
    for step in range(num_steps):
        # Update the navigator position
        navigator.update(dt=step_size)
        
        # Get the next frame
        frame_idx = min(step + 1, video_plume.frame_count - 1)
        current_frame = video_plume.get_frame(frame_idx)
        
        # Store the current state
        positions_history[:, step + 1] = navigator.positions
        orientations_history[:, step + 1] = navigator.orientations
        
        # Sample odor at the current position
        odor_readings[:, step + 1] = navigator.sample_odor(current_frame)
    
    return positions_history, orientations_history, odor_readings
