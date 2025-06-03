"""
VideoPlume module providing comprehensive video-based odor plume environment functionality.

This module consolidates the video processing capabilities from the original adapters and 
environments modules, providing enhanced Hydra configuration integration, OpenCV processing,
and workflow orchestration support for research pipelines.

The VideoPlume class serves as the primary interface for loading and processing video-based
odor plume data, with support for various preprocessing operations, metadata extraction,
and integration with DVC/Snakemake workflows.
"""

import numpy as np
from pathlib import Path
import cv2
import os
from contextlib import suppress
from typing import Dict, Union, Optional, Any, Tuple
import threading
import warnings

try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    # Graceful fallback when Hydra is not available
    DictConfig = dict
    HYDRA_AVAILABLE = False
    warnings.warn(
        "Hydra not available. Some configuration features may be limited.",
        ImportWarning
    )

from {{cookiecutter.project_slug}}.config.schemas import VideoPlumeConfig


class VideoPlume:
    """
    Comprehensive video-based odor plume environment implementation.
    
    This class provides frame-by-frame access to video data with advanced preprocessing
    capabilities, metadata extraction, and integration with research workflows. It supports
    both traditional dictionary-based configuration and modern Hydra DictConfig objects
    for enhanced experiment orchestration.
    
    Features:
    - OpenCV-based video processing with frame-by-frame access
    - Gaussian smoothing and horizontal flipping preprocessing
    - Automatic resource management with thread-safe operations
    - Environment variable interpolation for secure path management
    - Metadata extraction for research documentation
    - DVC/Snakemake workflow integration points
    - Hydra configuration system integration
    
    Example:
        Basic usage with file path:
        >>> plume = VideoPlume(video_path="path/to/video.mp4")
        >>> frame = plume.get_frame(0)
        >>> metadata = plume.get_metadata()
        
        Usage with Hydra configuration:
        >>> from omegaconf import DictConfig
        >>> cfg = DictConfig({
        ...     "video_path": "path/to/video.mp4",
        ...     "flip": True,
        ...     "kernel_size": 5,
        ...     "kernel_sigma": 1.5
        ... })
        >>> plume = VideoPlume.from_config(cfg)
        
        Advanced metadata extraction:
        >>> metadata_str = plume.get_metadata_string()
        >>> print(metadata_str)  # Formatted research documentation
    """
    
    def __init__(
        self,
        video_path: Union[str, Path],
        flip: bool = False,
        kernel_size: int = 0,
        kernel_sigma: float = 1.0,
        grayscale: bool = True,
        threshold: Optional[float] = None,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize VideoPlume with comprehensive configuration options.
        
        Args:
            video_path: Path to video file (supports environment variable interpolation)
            flip: Whether to apply horizontal flipping to frames
            kernel_size: Size of Gaussian kernel for smoothing (0 disables, must be odd)
            kernel_sigma: Standard deviation for Gaussian kernel
            grayscale: Whether to convert frames to grayscale
            threshold: Optional threshold value for frame processing
            normalize: Whether to normalize frame values
            **kwargs: Additional parameters for future extensibility
            
        Raises:
            IOError: If video file does not exist or cannot be opened
            ValueError: If kernel_size is even or negative
            
        Note:
            The video_path supports environment variable interpolation when used
            with Hydra configuration. For example: "${oc.env:VIDEO_DATA_PATH}/plume.mp4"
        """
        # Resolve environment variables in video path if present
        self.video_path = Path(self._resolve_env_path(str(video_path)))
        self.flip = flip
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.grayscale = grayscale
        self.threshold = threshold
        self.normalize = normalize
        
        # Validate kernel_size parameter
        if self.kernel_size < 0:
            raise ValueError("kernel_size must be non-negative")
        if self.kernel_size > 0 and self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd when greater than 0")
        
        # Validate kernel_sigma parameter
        if self.kernel_sigma <= 0:
            raise ValueError("kernel_sigma must be positive")
        
        # Thread safety for concurrent access
        self._lock = threading.RLock()
        self._is_closed = False
        self.cap = None
        
        # Initialize video capture and extract metadata
        self._initialize_video_capture()
        
        # Store additional configuration for workflow integration
        self._config_metadata = kwargs.copy()
        self._config_metadata.update({
            'flip': flip,
            'kernel_size': kernel_size,
            'kernel_sigma': kernel_sigma,
            'grayscale': grayscale,
            'threshold': threshold,
            'normalize': normalize
        })
    
    @classmethod
    def from_config(
        cls,
        cfg: Union[DictConfig, Dict[str, Any]],
        **kwargs
    ) -> 'VideoPlume':
        """
        Create VideoPlume instance from Hydra DictConfig or dictionary configuration.
        
        This factory method provides enhanced integration with Hydra's configuration
        management system, supporting environment variable interpolation, structured
        configuration composition, and parameter validation through Pydantic schemas.
        
        Args:
            cfg: Hydra DictConfig or dictionary containing configuration parameters
            **kwargs: Additional parameters to override configuration values
            
        Returns:
            Configured VideoPlume instance
            
        Raises:
            ValueError: If configuration validation fails
            KeyError: If required configuration parameters are missing
            
        Example:
            With Hydra DictConfig:
            >>> from omegaconf import DictConfig
            >>> cfg = DictConfig({
            ...     "video_path": "${oc.env:DATA_PATH}/experiment_001.mp4",
            ...     "flip": True,
            ...     "gaussian_blur": {"kernel_size": 5, "sigma": 1.5}
            ... })
            >>> plume = VideoPlume.from_config(cfg)
            
            With dictionary configuration:
            >>> config = {
            ...     "video_path": "data/plume_video.mp4",
            ...     "flip": False,
            ...     "kernel_size": 3,
            ...     "kernel_sigma": 1.0
            ... }
            >>> plume = VideoPlume.from_config(config)
        """
        try:
            # Convert DictConfig to dictionary if needed
            if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
                # Resolve environment variable interpolations
                config_dict = OmegaConf.to_container(cfg, resolve=True)
            else:
                config_dict = dict(cfg) if not isinstance(cfg, dict) else cfg.copy()
            
            # Apply any override parameters
            config_dict.update(kwargs)
            
            # Validate configuration using Pydantic schema
            validated_config = VideoPlumeConfig.model_validate(config_dict)
            
            # Create instance with validated parameters
            return cls(**validated_config.model_dump())
            
        except Exception as e:
            raise ValueError(f"Invalid VideoPlume configuration: {e}") from e
    
    def _resolve_env_path(self, path_str: str) -> str:
        """
        Resolve environment variable interpolations in path string.
        
        Supports both standard environment variable syntax ($VAR, ${VAR})
        and Hydra interpolation syntax (${oc.env:VAR}).
        
        Args:
            path_str: Path string potentially containing environment variables
            
        Returns:
            Resolved path string with environment variables expanded
        """
        # Handle Hydra-style environment variable interpolation
        if "${oc.env:" in path_str:
            # Simple fallback for Hydra syntax when OmegaConf is not available
            import re
            def replace_hydra_env(match):
                var_name = match.group(1)
                default = match.group(2) if match.group(2) else None
                return os.getenv(var_name, default or "")
            
            # Pattern: ${oc.env:VAR_NAME,default_value} or ${oc.env:VAR_NAME}
            pattern = r'\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}'
            path_str = re.sub(pattern, replace_hydra_env, path_str)
        
        # Handle standard environment variable expansion
        return os.path.expandvars(path_str)
    
    def _initialize_video_capture(self) -> None:
        """
        Initialize OpenCV video capture with comprehensive error handling.
        
        Raises:
            IOError: If video file does not exist or cannot be opened
        """
        if not self.video_path.exists():
            raise IOError(f"Video file does not exist: {self.video_path}")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_path}")
        
        # Extract video metadata
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Validate video properties
        if self.frame_count <= 0:
            raise IOError(f"Video file contains no frames: {self.video_path}")
        if self.width <= 0 or self.height <= 0:
            raise IOError(f"Invalid video dimensions: {self.width}x{self.height}")
        if self.fps <= 0:
            warnings.warn(f"Invalid or missing FPS information for: {self.video_path}")
            self.fps = 30.0  # Default fallback
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Retrieve and process a specific frame from the video.
        
        This method provides frame-by-frame access with optional preprocessing including
        horizontal flipping, Gaussian smoothing, grayscale conversion, and normalization.
        All operations are applied efficiently using OpenCV's optimized implementations.
        
        Args:
            frame_idx: Index of the frame to retrieve (0-based)
            
        Returns:
            Processed frame as numpy array, or None if frame cannot be retrieved
            
        Raises:
            ValueError: If VideoPlume is closed
            
        Note:
            Frame processing is applied in the following order:
            1. Frame extraction from video
            2. Horizontal flipping (if enabled)
            3. Grayscale conversion (if enabled)
            4. Gaussian smoothing (if kernel_size > 0)
            5. Thresholding (if threshold is set)
            6. Normalization (if enabled)
        """
        with self._lock:
            if self._is_closed:
                raise ValueError("Cannot get frame from closed VideoPlume")
            
            # Validate frame index
            if frame_idx < 0 or frame_idx >= self.frame_count:
                return None
            
            # Seek to requested frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                return None
            
            # Apply preprocessing pipeline
            frame = self._preprocess_frame(frame)
            
            return frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to a raw video frame.
        
        Args:
            frame: Raw frame from video capture
            
        Returns:
            Processed frame with applied transformations
        """
        # Apply horizontal flipping if requested
        if self.flip:
            frame = cv2.flip(frame, 1)
        
        # Convert to grayscale if requested
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian smoothing if kernel_size > 0
        if self.kernel_size > 0:
            frame = cv2.GaussianBlur(
                frame,
                (self.kernel_size, self.kernel_size),
                self.kernel_sigma
            )
        
        # Apply threshold if specified
        if self.threshold is not None:
            if len(frame.shape) == 3:
                # Convert to grayscale for thresholding
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(frame, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Normalize frame values if requested
        if self.normalize and frame.dtype != np.float32:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from the video file.
        
        Returns comprehensive information about the video including dimensions,
        frame rate, duration, and processing configuration. This metadata is
        essential for research documentation and workflow integration.
        
        Returns:
            Dictionary containing video and configuration metadata
            
        Example:
            >>> metadata = plume.get_metadata()
            >>> print(f"Video duration: {metadata['duration']:.2f} seconds")
            >>> print(f"Processing config: {metadata['processing_config']}")
        """
        with self._lock:
            return {
                # Video file properties
                "video_path": str(self.video_path),
                "filename": self.video_path.name,
                "file_size_bytes": self.video_path.stat().st_size if self.video_path.exists() else 0,
                
                # Video stream properties
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
                "frame_count": self.frame_count,
                "duration": self.duration,
                "shape": self.shape,
                
                # Processing configuration
                "processing_config": self._config_metadata.copy(),
                
                # Workflow integration metadata
                "dvc_compatible": True,  # Indicates DVC workflow compatibility
                "snakemake_ready": True,  # Indicates Snakemake rule compatibility
                "hydra_managed": HYDRA_AVAILABLE,  # Indicates Hydra configuration support
            }
    
    def get_metadata_string(self) -> str:
        """
        Generate formatted metadata string for research documentation.
        
        Creates a human-readable string containing comprehensive video and processing
        information suitable for research logs, reports, and documentation. This
        method consolidates functionality from the original environments module.
        
        Returns:
            Formatted string with video metadata and processing configuration
            
        Example:
            >>> metadata_str = plume.get_metadata_string()
            >>> print(metadata_str)
            Video: experiment_001.mp4
            Dimensions: 1920x1080 pixels
            Frame rate: 30.0 fps
            1800 frames
            Duration: 60.00 seconds
            Processing: flip=True, gaussian_blur=5x5 (σ=1.5)
        """
        metadata = self.get_metadata()
        
        # Format processing configuration for display
        processing_parts = []
        if self.flip:
            processing_parts.append("horizontal_flip=True")
        if self.kernel_size > 0:
            processing_parts.append(f"gaussian_blur={self.kernel_size}x{self.kernel_size} (σ={self.kernel_sigma})")
        if self.threshold is not None:
            processing_parts.append(f"threshold={self.threshold}")
        if self.normalize:
            processing_parts.append("normalize=True")
        
        processing_info = ", ".join(processing_parts) if processing_parts else "none"
        
        return (
            f"Video: {metadata['filename']}\n"
            f"Dimensions: {metadata['width']}x{metadata['height']} pixels\n"
            f"Frame rate: {metadata['fps']:.1f} fps\n"
            f"{metadata['frame_count']} frames\n"
            f"Duration: {metadata['duration']:.2f} seconds\n"
            f"Processing: {processing_info}"
        )
    
    @property
    def duration(self) -> float:
        """
        Calculate video duration in seconds.
        
        Returns:
            Duration in seconds, or 0.0 if FPS is unavailable
        """
        return 0.0 if self.fps == 0 else self.frame_count / self.fps
    
    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get video frame dimensions as (height, width) tuple.
        
        Returns:
            Tuple of (height, width) matching NumPy array convention
        """
        return (self.height, self.width)
    
    @property
    def is_closed(self) -> bool:
        """
        Check if the VideoPlume is closed.
        
        Returns:
            True if the VideoPlume has been closed and cannot be used
        """
        return self._is_closed
    
    def close(self) -> None:
        """
        Close video capture and release resources.
        
        This method ensures proper cleanup of OpenCV video capture resources
        and marks the instance as closed. It is thread-safe and can be called
        multiple times without error.
        
        Note:
            After calling close(), the VideoPlume instance cannot be used
            for frame retrieval. This method is automatically called by
            the destructor, but explicit calls are recommended for
            deterministic resource management.
        """
        with self._lock:
            if not self._is_closed and self.cap is not None:
                self.cap.release()
                self._is_closed = True
    
    def __del__(self):
        """
        Destructor ensuring automatic resource cleanup.
        
        Provides automatic resource management by calling close() when the
        object is garbage collected. This prevents resource leaks in case
        explicit close() calls are missed.
        """
        with suppress(Exception):
            self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()
    
    def __repr__(self) -> str:
        """
        Provide detailed string representation for debugging.
        
        Returns:
            String representation including key configuration and status
        """
        status = "closed" if self._is_closed else "open"
        return (
            f"VideoPlume(video_path='{self.video_path}', "
            f"frames={self.frame_count}, shape={self.shape}, "
            f"fps={self.fps:.1f}, status={status})"
        )
    
    def __str__(self) -> str:
        """
        Provide user-friendly string representation.
        
        Returns:
            Concise string representation for display
        """
        return f"VideoPlume({self.video_path.name}, {self.frame_count} frames)"


# Workflow Integration Utilities
# These functions provide integration points for DVC and Snakemake workflows

def create_video_plume_from_dvc_path(dvc_path: str, **kwargs) -> VideoPlume:
    """
    Create VideoPlume instance from DVC-managed data path.
    
    This utility function provides integration with DVC (Data Version Control)
    workflows by resolving DVC-managed paths and creating VideoPlume instances
    with appropriate configuration.
    
    Args:
        dvc_path: DVC path specification (e.g., "data/videos/experiment.mp4")
        **kwargs: Additional VideoPlume configuration parameters
        
    Returns:
        Configured VideoPlume instance
        
    Example:
        >>> # In a DVC pipeline
        >>> plume = create_video_plume_from_dvc_path(
        ...     "data/plume_videos/experiment_001.mp4",
        ...     flip=True,
        ...     kernel_size=5
        ... )
    """
    # For DVC integration, we assume the path is already resolved
    # In a real DVC workflow, this would interface with DVC's Python API
    resolved_path = Path(dvc_path)
    
    return VideoPlume(video_path=resolved_path, **kwargs)


def create_snakemake_rule_config(
    input_video: str,
    output_metadata: str,
    processing_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate Snakemake rule configuration for VideoPlume processing.
    
    This utility creates standardized configuration dictionaries for use in
    Snakemake workflows, enabling reproducible video processing pipelines.
    
    Args:
        input_video: Path to input video file
        output_metadata: Path for output metadata file
        processing_params: Optional processing parameters
        
    Returns:
        Dictionary suitable for Snakemake rule configuration
        
    Example:
        >>> # In a Snakemake workflow
        >>> rule_config = create_snakemake_rule_config(
        ...     input_video="data/raw/experiment.mp4",
        ...     output_metadata="data/processed/experiment_meta.json",
        ...     processing_params={"flip": True, "kernel_size": 5}
        ... )
    """
    config = {
        "input": {
            "video": input_video
        },
        "output": {
            "metadata": output_metadata
        },
        "params": processing_params or {},
        "resources": {
            "mem_mb": 2048,  # Reasonable default for video processing
            "runtime": 300   # 5 minutes default timeout
        }
    }
    
    return config


# Export the main class and utility functions
__all__ = [
    "VideoPlume",
    "create_video_plume_from_dvc_path", 
    "create_snakemake_rule_config"
]