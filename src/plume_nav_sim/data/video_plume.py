"""Video plume data handling and processing.

This module provides the VideoPlume class for loading and processing video-based
plume data for navigation simulations.
"""

import logging
import hashlib
import uuid
from pathlib import Path
from typing import Optional, Union, Any, Dict

import cv2
import numpy as np
from loguru import logger
from plume_nav_sim.api.navigation import ConfigurationError
from odor_plume_nav.data.video_plume import VIDEO_FILE_MISSING_MSG
from plume_nav_sim.utils.logging_setup import get_correlation_context

py_logger = logging.getLogger(__name__)


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
    
    @classmethod
    def from_config(cls, cfg: Any) -> "VideoPlume":
        """Create a VideoPlume from a configuration object or mapping."""
        if hasattr(cfg, "model_dump"):
            params = cfg.model_dump()
        elif hasattr(cfg, "dict") and not isinstance(cfg, dict):
            params = cfg.dict()
        else:
            params = dict(cfg)
        return cls(**params)

    def __init__(
        self,
        video_path: Union[str, Path],
        flip: bool = False,
        kernel_size: int = 0,
        kernel_sigma: float = 1.0,
        grayscale: bool = False,
        normalize: bool = False,
        frame_skip: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        **kwargs,
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
            frame_skip: Number of frames to skip between reads (>=1)
            start_frame: Starting frame index (>=0)
            end_frame: Optional ending frame index (exclusive)
            **kwargs: Additional keyword arguments for extended functionality
        
        Raises:
            IOError: If the video file does not exist
            ConfigurationError: If the video file cannot be opened
        """
        # Convert to Path object and validate existence
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise IOError(VIDEO_FILE_MISSING_MSG)
        
        # Validate frame range parameters early
        if frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if end_frame is not None and end_frame <= start_frame:
            raise ValueError("end_frame must be greater than start_frame")

        # Store configuration parameters
        self.flip = flip
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.grayscale = grayscale
        self.normalize = normalize
        self.frame_skip = frame_skip
        self.start_frame = start_frame
        self.end_frame = end_frame

        if self.kernel_size == 0:
            # Log explicitly when smoothing is turned off
            py_logger.info("Kernel smoothing disabled")
        
        # Store any additional keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Correlation context for logging
        self._correlation_ctx = get_correlation_context()
        self._correlation_ctx.bind_context()

        # Initialize video capture and validate
        self._init_video_capture()

        # Validate frame range against video properties
        if self.start_frame >= self.frame_count:
            raise ValueError("start_frame is beyond video length")
        if self.end_frame is not None and self.end_frame > self.frame_count:
            raise ValueError("end_frame is beyond video length")

        # Track current frame for sequential reads
        self._current_frame = self.start_frame
    
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
    
    def get_frame(self, frame_index: Optional[int] = None) -> Optional[np.ndarray]:
        """Retrieve a processed frame using frame skip logic."""
        # Determine the actual frame to fetch
        max_frame = self.end_frame if self.end_frame is not None else self.frame_count

        if frame_index is None:
            actual_idx = self._current_frame
            self._current_frame += self.frame_skip
        else:
            if frame_index < 0:
                raise ValueError(f"Frame index {frame_index} is out of range")
            actual_idx = self.start_frame + frame_index * self.frame_skip

        if actual_idx < self.start_frame or actual_idx >= max_frame:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, actual_idx)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video")
        processed = self._process_frame(frame)
        if getattr(self, "_correlation_ctx", None) is not None:
            self._correlation_ctx.bind_context()
        logger.bind(
            correlation_id=uuid.uuid4().hex,
            frame_index=actual_idx,
            video_path=str(self.video_path),
        ).info("frame_retrieved")
        return processed
    
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

    @property
    def duration(self) -> float:
        return 0.0 if self.fps == 0 else self.frame_count / self.fps

    @property
    def shape(self) -> tuple:
        return (self.height, self.width)

    def get_metadata(self) -> Dict[str, Any]:
        """Return basic video metadata including frame controls."""
        metadata = {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "shape": self.shape,
            "frame_skip": self.frame_skip,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }
        if getattr(self, "_correlation_ctx", None) is not None:
            self._correlation_ctx.bind_context()
        logger.bind(
            correlation_id=uuid.uuid4().hex,
            operation="get_metadata",
            video_path=str(self.video_path),
        ).info("metadata_retrieved")
        return metadata

    def get_workflow_metadata(self) -> Dict[str, Any]:
        """Return extended metadata with file details."""
        if getattr(self, "_correlation_ctx", None) is not None:
            self._correlation_ctx.bind_context()
        with open(self.video_path, "rb") as f:
            data = f.read()
        file_hash = hashlib.md5(data).hexdigest()
        metadata = self.get_metadata()
        metadata.update({"file_hash": file_hash, "file_size": len(data)})
        logger.info("workflow_metadata_generated")
        return metadata

    def reset(self):
        """Reset the video to the beginning."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self._current_frame = self.start_frame
    
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