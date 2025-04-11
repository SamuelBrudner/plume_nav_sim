"""
Navigator module for odor plume navigation.

This module provides the core Navigator class for navigating in odor plume environments.
"""

from typing import Tuple, Union, Any, Optional, List, Dict
import math
import numpy as np

# Import from the existing navigator module for now
# This will be refactored in a future update
from odor_plume_nav.navigator import Navigator as LegacyNavigator
from odor_plume_nav.config_models import NavigatorConfig


class Navigator(LegacyNavigator):
    """
    Navigator class for odor plume navigation that handles both single and multiple agents.
    
    This class inherits from the legacy Navigator for backward compatibility,
    but will be refactored in the future to live fully in the core module.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize a Navigator instance.
        
        Ensures positions are stored as float arrays to prevent casting errors.
        """
        super().__init__(*args, **kwargs)
        # Ensure positions are stored as float arrays to avoid casting errors
        if not np.issubdtype(self.positions.dtype, np.floating):
            self.positions = self.positions.astype(np.float64)
    
    @property
    def num_agents(self) -> int:
        """
        Get the number of agents being managed by this navigator.
        
        Returns:
            Number of agents
        """
        return self.positions.shape[0]
    
    def sample_odor(self, env_array: np.ndarray) -> Union[float, np.ndarray]:
        """
        Sample odor values at agent positions from the environment array.
        
        Args:
            env_array: 2D array representing the environment (e.g., video frame)
            
        Returns:
            For single agent: odor value as float
            For multiple agents: array of odor values
        """
        # Use the legacy implementation of odor reading
        return self.read_single_antenna_odor(env_array)


# Re-export legacy classes for backward compatibility
from odor_plume_nav.navigator import SimpleNavigator, VectorizedNavigator
