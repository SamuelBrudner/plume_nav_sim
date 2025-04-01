"""
Simulation module for odor plume navigation.

This module integrates the video plume and navigator components to simulate 
agent navigation in an odor plume environment.
"""

from typing import Optional, Union, Tuple, Any
from pathlib import Path
import numpy as np

from odor_plume_nav.video_plume import VideoPlume
from odor_plume_nav.navigator import SimpleNavigator
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
