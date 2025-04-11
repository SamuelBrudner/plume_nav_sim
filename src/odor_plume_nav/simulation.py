"""
Simulation module for odor plume navigation.

This module integrates the video plume and navigator components to simulate 
agent navigation in an odor plume environment.
"""

from typing import Optional, Union, Tuple, Any, Dict
from pathlib import Path
import numpy as np

from odor_plume_nav.video_plume import VideoPlume
from odor_plume_nav.navigator import SimpleNavigator, Navigator
from odor_plume_nav.video_plume_factory import create_video_plume_from_config
from odor_plume_nav.navigator_factory import create_navigator_from_config
from odor_plume_nav.config_utils import load_config


class Simulation:
    """
    Simulation class that integrates video plume and navigator.
    
    This class coordinates the video plume advancement and agent movement
    during simulation steps.
    """
    
    def __init__(
        self, 
        video_path: Union[str, Path], 
        dt: float = 0.1, 
        config_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize a simulation with video plume and navigator components.
        
        Args:
            video_path: Path to the video file
            dt: Time step for simulation updates (in seconds)
            config_path: Optional path to a configuration file
        """
        # Initialize simulation state
        self.dt = dt
        self.time = 0.0
        self.frame_index = 0
        
        # If a config is provided, use factory functions to create components
        if config_path is not None:
            self.plume = create_video_plume_from_config(video_path, config_path=config_path)
            self.navigator = create_navigator_from_config(config_path=config_path)
        else:
            # Otherwise, create with default settings
            self.plume = VideoPlume(video_path)
            self.navigator = SimpleNavigator()
        
        # Store the initial frame
        self.current_frame = self.plume.get_frame()
    
    def step(self) -> None:
        """
        Perform a single simulation step.
        
        This advances the video plume and updates the navigator position.
        """
        # Advance the simulation time
        self.time += self.dt
        self.frame_index += 1
        
        # Get the next frame from the plume
        self.current_frame = self.plume.get_frame()
        
        # Update the navigator position
        self.navigator.update(dt=self.dt)
    
    def get_agent_position(self) -> Tuple[float, float]:
        """
        Get the current position of the navigator agent.
        
        Returns:
            (x, y) position of the agent
        """
        return self.navigator.get_position()
    
    def get_current_frame(self) -> np.ndarray:
        """
        Get the current frame from the video plume.
        
        Returns:
            Current video frame as a numpy array
        """
        return self.current_frame


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
