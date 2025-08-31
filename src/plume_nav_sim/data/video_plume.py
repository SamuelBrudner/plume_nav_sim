"""
Video plume data handling and processing.

This module provides the VideoPlume class for loading and processing video-based
plume data for navigation simulations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, Any, Dict
from plume_nav_sim.api.navigation import ConfigurationError
from odor_plume_nav.data.video_plume import VIDEO_FILE_MISSING_MSG


class VideoPlume:
    """
    A class for handling video-based plume data.
    
    This class provides functionality to load and process video files containing
    plume visualizations for use in navigation simulations.
    
    Attributes:
        video_path (Path): Path to the video file
        flip (bool): Whether to flip the video frames
        kernel_size (int): Size of the processing kernel (0 for no processing)
        kernel_sigma (float): Sigma value for Gaussian kernel processing
        grayscale (bool): Whether to convert frames to grayscale
        normalize (bool): Whether to normalize frame values
    """
    
    def __init__(
        self,
        video_path: Union[str, Path],
        flip: bool = False,
        kernel_size: int = 0,
        kernel_sigma: float = 1.0,
        grayscale: bool = False,
        normalize: bool = False,
        **kwargs
    ):
        """
        Initialize the VideoPlume with the specified parameters.
        
        Args:
            video_path: Path to the video file
            flip: Whether to flip the video frames
            kernel_size: Size of the processing kernel (0 for no processing)
            kernel_sigma: Sigma value for Gaussian kernel processing
            grayscale: Whether to convert frames to grayscale
            normalize: Whether to normalize frame values
            **kwargs: Additional keyword arguments for extended functionality
        
        Raises:
            IOError: If the video file does not exist
            ConfigurationError: If the video file cannot be opened
        """
        # Convert to Path object and validate existence
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise IOError(VIDEO_FILE_MISSING_MSG)
        
        # Store configuration parameters
        self.flip = flip
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.grayscale = grayscale
        self.normalize = normalize
        
        # Store any additional keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Initialize video capture and validate
        self._init_video_capture()
    
    def _init_video_capture(self):
        """Initialize the OpenCV VideoCapture and validate it can be opened."""
        self._cap = cv2.VideoCapture(str(self.video_path))
        
        if not self._cap.isOpened():
            raise ConfigurationError(f"Failed to open video file: {self.video_path}")
        
        # Store video properties
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_frame(self, frame_index: Optional[int] = None) -> np.ndarray:
        """
        Get a frame from the video.
        
        Args:
            frame_index: Index of the frame to retrieve. If None, gets the next frame.
            
        Returns:
            The requested frame as a numpy array
            
        Raises:
            ValueError: If the frame index is out of range
            RuntimeError: If frame reading fails
        """
        if frame_index is not None:
            if frame_index < 0 or frame_index >= self.frame_count:
                raise ValueError(f"Frame index {frame_index} is out of range")
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video")
        
        # Apply processing based on configuration
        processed_frame = self._process_frame(frame)
        return processed_frame
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame according to the configured parameters.
        
        Args:
            frame: Raw frame from the video
            
        Returns:
            Processed frame
        """
        processed = frame.copy()
        
        # Apply flipping if requested
        if self.flip:
            processed = cv2.flip(processed, -1)  # Flip both horizontally and vertically
        
        # Convert to grayscale if requested
        if self.grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian kernel processing if specified
        if self.kernel_size > 0:
            processed = cv2.GaussianBlur(
                processed, 
                (self.kernel_size, self.kernel_size), 
                self.kernel_sigma
            )
        
        # Normalize if requested
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    def get_concentration(self, position: tuple) -> float:
        """
        Get the concentration value at a specific position in the current frame.
        
        Args:
            position: (x, y) position tuple
            
        Returns:
            Concentration value at the specified position
        """
        # This is a placeholder implementation
        # In a real implementation, this would extract concentration from the current frame
        return 0.0
    
    def reset(self):
        """Reset the video to the beginning."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def close(self):
        """Close the video capture and release resources."""
        if hasattr(self, '_cap') and self._cap is not None:
            self._cap.release()
    
    def __del__(self):
        """Destructor to ensure video capture is properly closed."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()