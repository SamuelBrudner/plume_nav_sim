"""
VideoPlume module providing a simple implementation for video-based odor plume environments.
"""

import numpy as np
from pathlib import Path
import cv2
from contextlib import suppress
from typing import Dict, Union, Optional

from odor_plume_nav.video_plume_config import VideoPlumeConfig
from odor_plume_nav.config_utils import load_config

class VideoPlume:
    """Minimal VideoPlume for testing purposes."""
    
    @classmethod
    def from_config(cls, 
                    video_path: Union[str, Path], 
                    config_dict: Optional[Dict] = None,
                    config_path: Optional[Union[str, Path]] = None,
                    **kwargs):
        """
        Create a VideoPlume from configuration.
        
        This method provides a consistent configuration approach across the codebase.
        Configuration can be provided directly as a dictionary or loaded from a file.
        
        Args:
            video_path: Path to the video file
            config_dict: Optional configuration dictionary with VideoPlume parameters
            config_path: Optional path to a configuration file
            **kwargs: Additional parameters that override both config_dict and file config
            
        Returns:
            VideoPlume instance configured with the provided parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Start with empty config
        params = {}

        # Load from file if provided (lowest priority)
        if config_path is not None:
            file_config = load_config(config_path)
            video_plume_config = file_config.get("video_plume", {})
            params |= video_plume_config

        # Update with provided config_dict (middle priority)
        if config_dict is not None:
            params |= config_dict

        # Add video_path to params
        params["video_path"] = video_path

        # Update with kwargs (highest priority)
        params |= kwargs

        # Validate using Pydantic model
        try:
            validated_config = VideoPlumeConfig.model_validate(params)
            # Create instance with validated parameters
            return cls(**validated_config.model_dump())
        except Exception as e:
            # Re-raise with more context
            raise ValueError(f"Invalid VideoPlume configuration: {str(e)}") from e
    
    def __init__(self, video_path, flip=False, kernel_size=0, kernel_sigma=1.0, **kwargs):
        """Initialize with basic parameters needed for test.
        
        Args:
            video_path: Path to the video file
            flip: Whether to flip the frames horizontally
            kernel_size: Size of the smoothing kernel (0 means no smoothing)
            kernel_sigma: Sigma of the Gaussian kernel
            **kwargs: Additional parameters
            
        Raises:
            IOError: If the video file cannot be opened
        """
        self.video_path = Path(video_path)
        self.flip = flip
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self._is_closed = False
        self.cap = None
        
        # Validate video file exists, but allow skipping for testing
        skip_validation = kwargs.get('_skip_validation', False)
        if not skip_validation and not self.video_path.exists():
            raise IOError(f"Video file does not exist: {self.video_path}")
        
        # Open the video file
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        # Check if video opened successfully
        if not skip_validation and not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_path}")
        
        # Store video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        
    def get_frame(self, frame_idx):
        """Get a frame from the video at the specified index.
        
        Args:
            frame_idx: Index of the frame to retrieve
            
        Returns:
            Grayscale frame as numpy array, or None if frame cannot be retrieved
            
        Raises:
            ValueError: If the VideoPlume is closed
        """
        if self._is_closed:
            raise ValueError("Cannot get frame from closed VideoPlume")
        
        # Validate frame index
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        
        # Set frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame and convert to grayscale if successful
        ret, frame = self.cap.read()
        
        # Apply flip if needed
        if ret and self.flip:
            frame = cv2.flip(frame, 1)  # 1 means horizontal flip
            
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else None
    
    @property
    def duration(self):
        """Get the duration of the video in seconds."""
        return 0 if self.fps == 0 else self.frame_count / self.fps
        
    @property
    def shape(self):
        """Get the shape of the frames (height, width)."""
        return (self.height, self.width)
    
    def get_metadata(self):
        """Return a dictionary containing all video metadata."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "shape": self.shape
        }
    
    def close(self):
        """Release the video capture resource."""
        if not self._is_closed and hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self._is_closed = True
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        with suppress(Exception):
            self.close()
