"""
Consolidated VideoPlume class implementation for video-based odor plume environments.

This module provides comprehensive video processing capabilities with Hydra configuration 
integration, OpenCV frame processing, and workflow orchestration support. It merges 
functionality from legacy adapters and environments into a unified implementation 
supporting modern configuration management and research workflows.

Key Features:
    - Hydra DictConfig factory method for seamless configuration integration
    - OpenCV-based video processing with preprocessing options
    - Pydantic schema validation within Hydra structured config system
    - DVC/Snakemake workflow integration points
    - Environment variable interpolation for secure video path management
    - Enhanced metadata extraction for research documentation
    - Thread-safe resource management with automatic cleanup

Example Usage:
    Basic factory method creation:
        >>> from hydra import compose, initialize
        >>> from {{cookiecutter.project_slug}}.data.video_plume import VideoPlume
        >>> 
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     plume = VideoPlume.from_config(cfg.video_plume)
    
    Direct instantiation with preprocessing:
        >>> plume = VideoPlume(
        ...     video_path="data/plume_video.mp4",
        ...     flip=True,
        ...     grayscale=True,
        ...     kernel_size=5,
        ...     kernel_sigma=1.0
        ... )
    
    Workflow integration:
        >>> # DVC data versioning compatible
        >>> plume = VideoPlume.from_config(cfg.video_plume)
        >>> metadata = plume.get_workflow_metadata()
        >>> # Snakemake rule input compatible
        >>> frame = plume.get_frame(frame_idx=100)
"""

import threading
from contextlib import suppress
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple

import cv2
import numpy as np
from loguru import logger

# Hydra imports for configuration management
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    logger.warning("Hydra not available. Falling back to dictionary configuration.")

# Import configuration schema for validation
from ..config.schemas import VideoPlumeConfig


class VideoPlume:
    """
    Unified VideoPlume implementation with comprehensive video processing capabilities.
    
    This class consolidates video-based odor plume environment functionality with 
    enhanced Hydra configuration integration, OpenCV processing, and workflow 
    orchestration support. Designed for research workflows requiring reproducible 
    video processing with flexible configuration management.
    
    Features:
        - Factory method creation with Hydra DictConfig support
        - OpenCV video processing with frame-by-frame access
        - Configurable preprocessing (grayscale, flipping, Gaussian smoothing)
        - Thread-safe resource management with automatic cleanup
        - Environment variable interpolation for secure path management
        - Workflow integration points for DVC and Snakemake
        - Enhanced metadata extraction for research documentation
        - Memory-efficient frame processing without caching
    
    Attributes:
        video_path (Path): Path to the video file
        flip (bool): Whether to flip frames horizontally
        grayscale (bool): Whether to convert frames to grayscale
        kernel_size (Optional[int]): Gaussian kernel size for smoothing
        kernel_sigma (Optional[float]): Gaussian kernel sigma for smoothing
        threshold (Optional[float]): Threshold value for binary detection
        normalize (bool): Whether to normalize frame values to [0, 1] range
        width (int): Video frame width in pixels
        height (int): Video frame height in pixels
        fps (float): Video frame rate
        frame_count (int): Total number of frames in video
        duration (float): Video duration in seconds
    
    Thread Safety:
        All public methods are thread-safe. Internal OpenCV VideoCapture 
        operations are protected by locks to prevent concurrent access issues.
    """
    
    def __init__(
        self,
        video_path: Union[str, Path],
        flip: bool = False,
        grayscale: bool = True,
        kernel_size: Optional[int] = None,
        kernel_sigma: Optional[float] = None,
        threshold: Optional[float] = None,
        normalize: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize VideoPlume with specified parameters.
        
        Args:
            video_path: Path to the video file containing plume data
            flip: Whether to flip video frames horizontally for coordinate system alignment
            grayscale: Whether to convert frames to grayscale for processing efficiency
            kernel_size: Gaussian kernel size for smoothing (must be odd and positive)
            kernel_sigma: Gaussian kernel sigma for smoothing (must be positive)
            threshold: Threshold value for binary plume detection (0.0 to 1.0)
            normalize: Whether to normalize frame values to [0, 1] range
            **kwargs: Additional parameters for future extensibility
            
        Raises:
            IOError: If video file does not exist or cannot be opened
            ValueError: If kernel parameters are invalid or inconsistent
            
        Note:
            Gaussian smoothing is applied only when both kernel_size and 
            kernel_sigma are specified. All preprocessing operations are 
            applied in sequence: grayscale → flip → gaussian → threshold → normalize.
        """
        # Convert and validate video path
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise IOError(f"Video file does not exist: {self.video_path}")
        if not self.video_path.is_file():
            raise IOError(f"Video path is not a file: {self.video_path}")
        
        # Store preprocessing parameters
        self.flip = flip
        self.grayscale = grayscale
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.threshold = threshold
        self.normalize = normalize
        
        # Validate Gaussian parameters consistency
        if (kernel_size is not None) != (kernel_sigma is not None):
            raise ValueError(
                "Both kernel_size and kernel_sigma must be specified together or not at all"
            )
        
        if kernel_size is not None:
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError("kernel_size must be positive and odd")
            if kernel_sigma <= 0:
                raise ValueError("kernel_sigma must be positive")
        
        if threshold is not None and (threshold < 0.0 or threshold > 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")
        
        # Initialize OpenCV video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {self.video_path}")
        
        # Extract video metadata
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Validate extracted metadata
        if self.frame_count <= 0:
            raise IOError(f"Invalid frame count: {self.frame_count}")
        if self.width <= 0 or self.height <= 0:
            raise IOError(f"Invalid video dimensions: {self.width}x{self.height}")
        if self.fps <= 0:
            logger.warning(f"Invalid or missing FPS value: {self.fps}, defaulting to 30.0")
            self.fps = 30.0
        
        # Thread safety for OpenCV operations
        self._lock = threading.RLock()
        self._is_closed = False
        
        logger.info(
            f"VideoPlume initialized: {self.video_path.name} "
            f"({self.width}x{self.height}, {self.frame_count} frames, {self.fps:.1f} fps)"
        )
    
    @classmethod
    def from_config(cls, cfg: Union[DictConfig, Dict]) -> 'VideoPlume':
        """
        Create VideoPlume instance from Hydra configuration.
        
        This factory method supports Hydra's structured configuration system
        with automatic Pydantic validation and environment variable interpolation.
        Compatible with DVC data versioning and Snakemake workflow definitions.
        
        Args:
            cfg: Hydra DictConfig object or dictionary containing video plume parameters.
                 Expected configuration structure:
                 ```yaml
                 video_path: ${oc.env:PLUME_VIDEO_PATH,data/default_plume.mp4}
                 flip: false
                 grayscale: true
                 kernel_size: 5
                 kernel_sigma: 1.0
                 threshold: 0.5
                 normalize: true
                 ```
        
        Returns:
            VideoPlume: Configured instance ready for frame processing
            
        Raises:
            ValueError: If configuration validation fails
            IOError: If video file cannot be accessed or opened
            
        Example:
            >>> from hydra import compose, initialize
            >>> with initialize(config_path="../conf"):
            ...     cfg = compose(config_name="config")
            ...     plume = VideoPlume.from_config(cfg.video_plume)
            
        Note:
            Environment variable interpolation is handled automatically by Hydra.
            For example, ${oc.env:PLUME_VIDEO_PATH} resolves to the value of
            the PLUME_VIDEO_PATH environment variable.
        """
        # Convert DictConfig to regular dict if needed for Pydantic validation
        if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
            # Resolve any remaining interpolations and convert to dict
            config_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            config_dict = dict(cfg)
        
        try:
            # Validate configuration through Pydantic schema
            # Note: _skip_validation=True prevents file existence check during validation
            # since we want to handle that in the main constructor
            validated_config = VideoPlumeConfig.model_validate(
                {**config_dict, "_skip_validation": True}
            )
            
            # Create instance with validated parameters
            return cls(**validated_config.model_dump(exclude={"_skip_validation"}))
            
        except Exception as e:
            logger.error(f"VideoPlume configuration validation failed: {e}")
            raise ValueError(f"Invalid VideoPlume configuration: {e}") from e
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Retrieve and preprocess a video frame by index.
        
        Applies all configured preprocessing operations in sequence:
        1. Frame extraction from video
        2. Grayscale conversion (if enabled)
        3. Horizontal flipping (if enabled)
        4. Gaussian smoothing (if configured)
        5. Threshold application (if configured)
        6. Normalization to [0, 1] range (if enabled)
        
        Args:
            frame_idx: Zero-based frame index to retrieve
            
        Returns:
            Preprocessed frame as numpy array, or None if frame unavailable.
            Frame shape depends on preprocessing options:
            - Grayscale: (height, width) with dtype float32 or uint8
            - Color: (height, width, 3) with dtype float32 or uint8
            
        Raises:
            ValueError: If VideoPlume has been closed
            
        Note:
            This method is thread-safe and can be called concurrently.
            Out-of-bounds frame indices return None rather than raising exceptions.
            
        Example:
            >>> frame = plume.get_frame(100)
            >>> if frame is not None:
            ...     print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        """
        with self._lock:
            if self._is_closed:
                raise ValueError("Cannot get frame from closed VideoPlume")
            
            # Validate frame index bounds
            if frame_idx < 0 or frame_idx >= self.frame_count:
                logger.debug(f"Frame index {frame_idx} out of bounds [0, {self.frame_count})")
                return None
            
            # Seek to the requested frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                return None
            
            # Apply preprocessing pipeline
            frame = self._preprocess_frame(frame)
            return frame
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline to a raw video frame.
        
        Args:
            frame: Raw frame from OpenCV VideoCapture
            
        Returns:
            Preprocessed frame according to instance configuration
        """
        # Step 1: Grayscale conversion
        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Horizontal flipping for coordinate system alignment
        if self.flip:
            frame = cv2.flip(frame, 1)  # 1 = horizontal flip
        
        # Step 3: Gaussian smoothing for noise reduction
        if self.kernel_size is not None and self.kernel_sigma is not None:
            frame = cv2.GaussianBlur(
                frame, 
                (self.kernel_size, self.kernel_size), 
                self.kernel_sigma
            )
        
        # Step 4: Normalization to [0, 1] range
        if self.normalize:
            if frame.dtype != np.float32:
                frame = frame.astype(np.float32)
            frame = frame / 255.0
        
        # Step 5: Threshold application for binary detection
        if self.threshold is not None:
            if not self.normalize:
                # Convert to float and normalize for thresholding
                if frame.dtype != np.float32:
                    frame = frame.astype(np.float32) / 255.0
            # Apply threshold
            frame = (frame > self.threshold).astype(np.float32)
        
        return frame
    
    @property
    def duration(self) -> float:
        """
        Get video duration in seconds.
        
        Returns:
            Video duration calculated from frame count and FPS.
            Returns 0.0 if FPS is invalid.
        """
        return 0.0 if self.fps <= 0 else self.frame_count / self.fps
    
    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get video frame shape as (height, width).
        
        Returns:
            Tuple of (height, width) representing frame dimensions
        """
        return (self.height, self.width)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract comprehensive video metadata for analysis and documentation.
        
        Returns:
            Dictionary containing video properties and preprocessing configuration:
            - width: Frame width in pixels
            - height: Frame height in pixels  
            - fps: Frame rate in frames per second
            - frame_count: Total number of frames
            - duration: Video duration in seconds
            - shape: Frame shape as (height, width) tuple
            - preprocessing: Applied preprocessing configuration
            
        Example:
            >>> metadata = plume.get_metadata()
            >>> print(f"Video: {metadata['width']}x{metadata['height']}")
            >>> print(f"Duration: {metadata['duration']:.2f}s")
        """
        return {
            "video_path": str(self.video_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "shape": self.shape,
            "preprocessing": {
                "flip": self.flip,
                "grayscale": self.grayscale,
                "kernel_size": self.kernel_size,
                "kernel_sigma": self.kernel_sigma,
                "threshold": self.threshold,
                "normalize": self.normalize,
            }
        }
    
    def get_metadata_string(self) -> str:
        """
        Generate formatted metadata string for research documentation.
        
        Creates a human-readable summary of video properties and preprocessing
        configuration suitable for research logs, experiment documentation,
        and automated reporting systems.
        
        Returns:
            Formatted string with comprehensive video information
            
        Example:
            >>> print(plume.get_metadata_string())
            Video: plume_video.mp4
            Dimensions: 640x480 pixels
            Frame rate: 30.0 fps
            1500 frames
            Duration: 50.00 seconds
            Preprocessing: grayscale, flip, gaussian(5,1.0)
        """
        metadata = self.get_metadata()
        
        # Build preprocessing description
        preprocessing_parts = []
        if metadata["preprocessing"]["grayscale"]:
            preprocessing_parts.append("grayscale")
        if metadata["preprocessing"]["flip"]:
            preprocessing_parts.append("flip")
        if metadata["preprocessing"]["kernel_size"] is not None:
            kernel_size = metadata["preprocessing"]["kernel_size"]
            kernel_sigma = metadata["preprocessing"]["kernel_sigma"]
            preprocessing_parts.append(f"gaussian({kernel_size},{kernel_sigma})")
        if metadata["preprocessing"]["threshold"] is not None:
            threshold = metadata["preprocessing"]["threshold"]
            preprocessing_parts.append(f"threshold({threshold})")
        if metadata["preprocessing"]["normalize"]:
            preprocessing_parts.append("normalize")
        
        preprocessing_str = ", ".join(preprocessing_parts) if preprocessing_parts else "none"
        
        return (
            f"Video: {self.video_path.name}\n"
            f"Dimensions: {metadata['width']}x{metadata['height']} pixels\n"
            f"Frame rate: {metadata['fps']:.1f} fps\n"
            f"{metadata['frame_count']} frames\n"
            f"Duration: {metadata['duration']:.2f} seconds\n"
            f"Preprocessing: {preprocessing_str}"
        )
    
    def get_workflow_metadata(self) -> Dict[str, Any]:
        """
        Generate workflow-compatible metadata for DVC and Snakemake integration.
        
        Provides metadata in a structure compatible with workflow management
        systems for reproducible data processing pipelines.
        
        Returns:
            Dictionary with workflow-specific metadata including file hashes,
            processing parameters, and dependency information
            
        Example:
            >>> metadata = plume.get_workflow_metadata()
            >>> # Use in Snakemake rule
            >>> with open("metadata.yaml", "w") as f:
            ...     yaml.dump(metadata, f)
        """
        import hashlib
        
        # Calculate file hash for DVC compatibility
        file_hash = hashlib.md5()
        with open(self.video_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        
        base_metadata = self.get_metadata()
        
        # Add workflow-specific fields
        workflow_metadata = {
            **base_metadata,
            "file_hash": file_hash.hexdigest(),
            "file_size": self.video_path.stat().st_size,
            "workflow_version": "1.0",
            "dependencies": {
                "opencv_version": cv2.__version__,
                "numpy_version": np.__version__,
            }
        }
        
        return workflow_metadata
    
    def close(self) -> None:
        """
        Close video capture and release resources.
        
        Releases the OpenCV VideoCapture object and marks the instance as closed.
        After calling this method, attempts to get frames will raise ValueError.
        This method is idempotent and thread-safe.
        
        Note:
            This method is automatically called by __del__ for cleanup,
            but explicit calling is recommended for deterministic resource management.
        """
        with self._lock:
            if not self._is_closed and self.cap is not None:
                self.cap.release()
                self._is_closed = True
                logger.debug(f"VideoPlume closed: {self.video_path.name}")
    
    def __del__(self) -> None:
        """
        Automatic cleanup when instance is garbage collected.
        
        Ensures OpenCV resources are properly released even if close() 
        was not called explicitly. Uses suppress to handle any exceptions
        during cleanup to prevent issues during interpreter shutdown.
        """
        with suppress(Exception):
            self.close()
    
    def __enter__(self) -> 'VideoPlume':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "closed" if self._is_closed else "open"
        return (
            f"VideoPlume({self.video_path.name}, {self.width}x{self.height}, "
            f"{self.frame_count} frames, {status})"
        )


# Factory function for API compatibility
def create_video_plume(cfg: Union[DictConfig, Dict], **kwargs) -> VideoPlume:
    """
    Factory function for creating VideoPlume instances with configuration override.
    
    Provides an alternative factory method compatible with the API layer
    while supporting parameter overrides for dynamic configuration.
    
    Args:
        cfg: Hydra DictConfig or dictionary with video plume configuration
        **kwargs: Additional parameters to override configuration values
        
    Returns:
        VideoPlume: Configured instance with applied overrides
        
    Example:
        >>> from {{cookiecutter.project_slug}}.api.navigation import create_video_plume
        >>> plume = create_video_plume(cfg.video_plume, flip=True)
    """
    # Merge configuration with overrides
    if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
        config_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        config_dict = dict(cfg)
    
    # Apply overrides
    config_dict.update(kwargs)
    
    return VideoPlume.from_config(config_dict)


# Re-export main class for public API
__all__ = ["VideoPlume", "create_video_plume"]