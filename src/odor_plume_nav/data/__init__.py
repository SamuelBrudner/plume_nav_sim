"""
Data processing package for video-based odor plume environments.

This package provides unified access to video plume environments and data handling 
components with comprehensive framework integration support. It serves as the primary
entry point for video-based odor plume data processing across different research 
workflows including Kedro pipelines, reinforcement learning frameworks, and 
machine learning analysis tools.

Key Features:
    - Clean import patterns for different framework integrations
    - Hydra configuration support with DictConfig factory methods  
    - Workflow orchestration integration (DVC/Snakemake compatibility)
    - NumPy array interfaces for ML framework compatibility
    - Kedro pipeline-compatible data processing interfaces
    - Enhanced metadata extraction for research documentation

Supported Import Patterns:
    
    Basic VideoPlume usage:
        >>> from odor_plume_nav.data import VideoPlume
        >>> plume = VideoPlume("data/plume_video.mp4")
    
    Hydra configuration integration:
        >>> from odor_plume_nav.data import VideoPlume, create_video_plume
        >>> from hydra import compose, initialize
        >>> 
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     plume = VideoPlume.from_config(cfg.video_plume)
        ...     # Alternative factory method
        ...     plume2 = create_video_plume(cfg.video_plume, flip=True)
    
    Kedro pipeline integration:
        >>> from odor_plume_nav.data import get_video_plume_catalog_entry
        >>> # Use in Kedro data catalog definition
        >>> catalog_entry = get_video_plume_catalog_entry("plume_video")
    
    RL framework compatibility:
        >>> from odor_plume_nav.data import VideoPlume
        >>> plume = VideoPlume("environment.mp4", normalize=True)
        >>> frame = plume.get_frame(step_idx)  # Returns NumPy array for RL agent
    
    ML/Neural network analysis:
        >>> from odor_plume_nav.data import VideoPlume, get_frame_batch
        >>> plume = VideoPlume("dataset.mp4", grayscale=True, normalize=True)
        >>> frames = get_frame_batch(plume, indices=[10, 20, 30])  # Batch processing

Workflow Integration:
    
    DVC data versioning:
        >>> from odor_plume_nav.data import get_workflow_metadata
        >>> metadata = get_workflow_metadata(plume)
        >>> # Compatible with DVC stage definition
    
    Snakemake rules:
        >>> # In Snakefile
        >>> from odor_plume_nav.data import VideoPlume
        >>> plume = VideoPlume(input.video_file)
        >>> frames = [plume.get_frame(i) for i in range(plume.frame_count)]

Environment Variable Support:
    The package supports secure path management through environment variables:
    
    >>> import os
    >>> os.environ['PLUME_VIDEO_PATH'] = '/secure/path/to/videos'
    >>> # In Hydra config: video_path: ${oc.env:PLUME_VIDEO_PATH}/dataset.mp4
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from pathlib import Path

# Core video plume functionality
from .video_plume import VideoPlume, create_video_plume

# Type imports for better IDE support and documentation
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    # Fallback type hint for when Hydra is not available
    DictConfig = Dict[str, Any]


def get_frame_batch(
    video_plume: VideoPlume, 
    indices: List[int], 
    skip_missing: bool = True
) -> List[Optional[np.ndarray]]:
    """
    Extract multiple frames from a VideoPlume instance for batch processing.
    
    Optimized for ML/neural network analysis workflows requiring batch data loading.
    Provides efficient frame extraction with optional missing frame handling for
    robust dataset processing.
    
    Args:
        video_plume: VideoPlume instance to extract frames from
        indices: List of frame indices to extract
        skip_missing: If True, missing frames return None; if False, raises error
        
    Returns:
        List of frames as NumPy arrays, with None for missing frames if skip_missing=True
        
    Raises:
        ValueError: If skip_missing=False and any frame is missing
        
    Example:
        >>> plume = VideoPlume("dataset.mp4", normalize=True)
        >>> frames = get_frame_batch(plume, [0, 10, 20, 30])
        >>> valid_frames = [f for f in frames if f is not None]
        >>> batch_array = np.stack(valid_frames)  # Shape: (batch_size, height, width)
        
    Note:
        For large batch sizes, consider processing in chunks to manage memory usage.
        Frame order in the output list corresponds to the input indices order.
    """
    frames = []
    
    for idx in indices:
        try:
            frame = video_plume.get_frame(idx)
            if frame is None and not skip_missing:
                raise ValueError(f"Frame {idx} not available and skip_missing=False")
            frames.append(frame)
        except Exception as e:
            if not skip_missing:
                raise ValueError(f"Failed to extract frame {idx}: {e}") from e
            frames.append(None)
    
    return frames


def get_workflow_metadata(video_plume: VideoPlume) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for workflow integration systems.
    
    Creates workflow-compatible metadata supporting DVC data versioning,
    Snakemake pipeline definitions, and automated research documentation.
    Extends the base VideoPlume metadata with additional workflow-specific
    information for enhanced reproducibility and tracking.
    
    Args:
        video_plume: VideoPlume instance to extract metadata from
        
    Returns:
        Dictionary containing comprehensive workflow metadata including:
        - File metadata (path, size, hash)
        - Video properties (dimensions, frame count, duration)
        - Processing configuration (preprocessing parameters)
        - Workflow compatibility information
        - Dependency versions and requirements
        
    Example:
        >>> plume = VideoPlume("experiment_data.mp4")
        >>> metadata = get_workflow_metadata(plume)
        >>> # Use in DVC stage definition
        >>> with open("video_metadata.yaml", "w") as f:
        ...     yaml.dump(metadata, f)
        
        >>> # Use in Snakemake rule
        >>> with open("processing_log.json", "w") as f:
        ...     json.dump(metadata, f, indent=2)
    """
    # Get base workflow metadata from VideoPlume
    base_metadata = video_plume.get_workflow_metadata()
    
    # Add package-level workflow information
    workflow_metadata = {
        **base_metadata,
        "package_info": {
            "data_module": "odor_plume_nav.data",
            "video_plume_class": "VideoPlume",
            "api_version": "1.0.0",
            "workflow_compatibility": {
                "dvc": True,
                "snakemake": True,
                "kedro": True,
                "hydra": HYDRA_AVAILABLE
            }
        },
        "processing_capabilities": {
            "batch_processing": True,
            "frame_indexing": True,
            "metadata_extraction": True,
            "preprocessing_pipeline": True,
            "memory_efficient": True
        },
        "integration_patterns": {
            "numpy_interface": True,
            "hydra_config": HYDRA_AVAILABLE,
            "environment_variables": True,
            "path_interpolation": True
        }
    }
    
    return workflow_metadata


def get_video_plume_catalog_entry(
    name: str,
    video_path: Optional[Union[str, Path]] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate Kedro data catalog entry for VideoPlume integration.
    
    Creates standardized Kedro catalog entries for seamless integration with
    Kedro pipelines, supporting both direct instantiation and configuration-based
    creation with Hydra parameter management.
    
    Args:
        name: Dataset name for Kedro catalog
        video_path: Optional path to video file (can use Hydra interpolation)
        config_overrides: Optional configuration parameters to override defaults
        
    Returns:
        Dictionary formatted for Kedro data catalog with VideoPlume configuration
        
    Example:
        >>> # Basic catalog entry
        >>> entry = get_video_plume_catalog_entry(
        ...     "experiment_video",
        ...     video_path="data/experiment.mp4"
        ... )
        
        >>> # With configuration overrides
        >>> entry = get_video_plume_catalog_entry(
        ...     "processed_video",
        ...     video_path="${oc.env:DATA_PATH}/video.mp4",
        ...     config_overrides={"flip": True, "grayscale": True}
        ... )
        
        >>> # Use in Kedro catalog.yml
        >>> # experiment_video:
        >>> #   type: odor_plume_nav.data.VideoPlume
        >>> #   video_path: data/experiment.mp4
        >>> #   flip: false
        >>> #   grayscale: true
    """
    # Base catalog entry structure
    catalog_entry = {
        "type": "odor_plume_nav.data.VideoPlume",
        "description": f"VideoPlume dataset: {name}",
        "metadata": {
            "kedro_compatible": True,
            "hydra_configurable": HYDRA_AVAILABLE,
            "workflow_ready": True
        }
    }
    
    # Add video path if provided
    if video_path is not None:
        catalog_entry["video_path"] = str(video_path)
    
    # Apply configuration overrides
    if config_overrides:
        catalog_entry.update(config_overrides)
    
    # Add default processing parameters for Kedro compatibility
    default_params = {
        "flip": False,
        "grayscale": True,
        "normalize": True
    }
    
    # Only add defaults that aren't already specified
    for key, value in default_params.items():
        if key not in catalog_entry:
            catalog_entry[key] = value
    
    return catalog_entry


def validate_video_file(video_path: Union[str, Path]) -> bool:
    """
    Validate video file accessibility and format compatibility.
    
    Performs comprehensive validation of video files for VideoPlume compatibility,
    checking file existence, format support, and basic metadata accessibility.
    Useful for workflow validation and error prevention in automated pipelines.
    
    Args:
        video_path: Path to video file for validation
        
    Returns:
        True if video file is valid and compatible, False otherwise
        
    Example:
        >>> if validate_video_file("data/experiment.mp4"):
        ...     plume = VideoPlume("data/experiment.mp4")
        ... else:
        ...     print("Invalid video file")
        
        >>> # Use in workflow validation
        >>> video_files = ["video1.mp4", "video2.avi"]
        >>> valid_files = [f for f in video_files if validate_video_file(f)]
    """
    try:
        video_path = Path(video_path)
        
        # Check file existence and accessibility
        if not video_path.exists():
            return False
        
        if not video_path.is_file():
            return False
        
        # Try to create VideoPlume instance for format validation
        test_plume = VideoPlume(video_path)
        
        # Basic validation of video properties
        if test_plume.frame_count <= 0:
            test_plume.close()
            return False
        
        if test_plume.width <= 0 or test_plume.height <= 0:
            test_plume.close()
            return False
        
        # Clean up test instance
        test_plume.close()
        return True
        
    except Exception:
        # Any exception during validation indicates incompatible file
        return False


def create_video_plume_from_env(
    env_var_name: str = "PLUME_VIDEO_PATH",
    config_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> VideoPlume:
    """
    Create VideoPlume instance from environment variable path.
    
    Convenient factory method for creating VideoPlume instances using environment
    variable path resolution, supporting secure deployment and configuration
    management in production environments.
    
    Args:
        env_var_name: Environment variable containing video file path
        config_overrides: Optional configuration parameters to override
        **kwargs: Additional parameters passed to VideoPlume constructor
        
    Returns:
        VideoPlume instance configured with environment variable path
        
    Raises:
        EnvironmentError: If environment variable is not set
        IOError: If video file path is invalid or inaccessible
        
    Example:
        >>> import os
        >>> os.environ['PLUME_VIDEO_PATH'] = '/data/experiment.mp4'
        >>> plume = create_video_plume_from_env()
        
        >>> # With custom environment variable
        >>> os.environ['CUSTOM_VIDEO'] = '/videos/dataset.mp4'
        >>> plume = create_video_plume_from_env("CUSTOM_VIDEO", {"flip": True})
    """
    import os
    
    # Get video path from environment variable
    video_path = os.getenv(env_var_name)
    if video_path is None:
        raise EnvironmentError(
            f"Environment variable '{env_var_name}' not set. "
            f"Please set it to the path of your video file."
        )
    
    # Merge configuration overrides with kwargs
    config = kwargs.copy()
    if config_overrides:
        config.update(config_overrides)
    
    # Create VideoPlume instance
    return VideoPlume(video_path, **config)


# Package metadata and version info
__version__ = "1.0.0"
__author__ = "Blitzy Engineering Team"

# Comprehensive public API exports
__all__ = [
    # Core classes and functions
    "VideoPlume",
    "create_video_plume", 
    
    # Batch processing utilities
    "get_frame_batch",
    
    # Workflow integration
    "get_workflow_metadata",
    "get_video_plume_catalog_entry",
    
    # Validation utilities
    "validate_video_file",
    
    # Environment-based creation
    "create_video_plume_from_env",
    
    # Type hints for external use
    "DictConfig" if HYDRA_AVAILABLE else None,
]

# Clean up None values from __all__ when Hydra is not available
__all__ = [item for item in __all__ if item is not None]


# Package-level configuration for improved usability
def _configure_logging():
    """Configure package-level logging for better debugging and monitoring."""
    from loguru import logger
    import sys
    
    # Add package-specific log formatting
    logger.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                         "<level>{level: <8}</level> | "
                         "<cyan>odor_plume_nav.data</cyan> | "
                         "<level>{message}</level>",
                "level": "INFO"
            }
        ]
    )


# Initialize package-level configuration
_configure_logging()

# Package documentation and metadata for automated tools
__doc_format__ = "restructuredtext"
__package_metadata__ = {
    "name": "odor_plume_nav.data",
    "description": "Video-based odor plume data processing with workflow integration",
    "version": __version__,
    "author": __author__,
    "compatibility": {
        "kedro": ">=0.18.0",
        "hydra": ">=1.1.0" if HYDRA_AVAILABLE else "not available",
        "opencv": ">=4.5.0",
        "numpy": ">=1.20.0"
    },
    "workflow_support": {
        "dvc": True,
        "snakemake": True,
        "kedro_pipelines": True,
        "hydra_configs": HYDRA_AVAILABLE
    }
}