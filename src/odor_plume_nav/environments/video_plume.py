"""
Video plume environment for odor plume navigation.

This module provides a video-based plume environment implementation.
"""

from typing import Dict, Optional, Union
import pathlib
import cv2
import numpy as np

# Import from the existing video_plume module for now
from odor_plume_nav.video_plume import VideoPlume as LegacyVideoPlume


class VideoPlume(LegacyVideoPlume):
    """
    Video-based plume environment implementation.
    
    This class inherits from the legacy VideoPlume for backward compatibility,
    but will be refactored in the future to live fully in the environments module.
    """
    
    def get_frame(self, frame_idx):
        """
        Get a frame from the video at the specified index.
        
        Args:
            frame_idx: Index of the frame to retrieve
            
        Returns:
            Grayscale frame as numpy array, or None if frame cannot be retrieved
            
        Raises:
            ValueError: If the VideoPlume is closed
        """
        if self._is_closed:
            raise ValueError("VideoPlume is closed")
        
        return super().get_frame(frame_idx)
    
    def get_metadata_string(self):
        """
        Get a formatted string with video metadata.
        
        Returns:
            String with formatted metadata
        """
        metadata = self.get_metadata()
        
        return (
            f"Video: {self.video_path.name}\n"
            f"Dimensions: {metadata['width']}x{metadata['height']} pixels\n"
            f"Frame rate: {metadata['fps']:.1f} fps\n"
            f"{metadata['frame_count']} frames\n"
            f"Duration: {metadata['duration']:.2f} seconds"
        )
