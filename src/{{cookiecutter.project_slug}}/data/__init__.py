"""
Data processing module for odor plume navigation simulation.

This module provides unified access to video-based odor plume environments and data
handling components, serving as the primary entry point for data processing functionality
within the simulation framework. The module supports multiple integration patterns for
scientific computing workflows including Kedro pipelines, reinforcement learning
frameworks, and machine learning analysis tools.

The module is designed to support modern research workflows through:
- Clean import patterns for different use cases
- DVC/Snakemake workflow integration 
- Hydra configuration management integration
- NumPy array interfaces for ML frameworks
- Protocol-based extensibility for custom environments

Import Examples:
    # For Kedro projects
    from {{cookiecutter.project_slug}}.data import VideoPlume
    
    # For RL frameworks  
    from {{cookiecutter.project_slug}}.data import VideoPlume, create_video_plume_from_dvc_path
    
    # For ML/neural network analyses
    from {{cookiecutter.project_slug}}.data import VideoPlume
    
    # For workflow orchestration
    from {{cookiecutter.project_slug}}.data import create_snakemake_rule_config

Architecture:
    The data module follows a clean architecture pattern with clear separation of concerns:
    - VideoPlume: Core environment class with frame-by-frame video processing
    - Workflow utilities: Integration functions for DVC and Snakemake
    - Configuration support: Hydra DictConfig factory methods
    - Resource management: Automatic cleanup and thread-safe operations
"""

from typing import Dict, Any, Optional, Union
import warnings

# Core video plume environment implementation
from .video_plume import (
    VideoPlume,
    create_video_plume_from_dvc_path,
    create_snakemake_rule_config
)

# Ensure compatibility with optional dependencies
try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict  # Fallback type
    HYDRA_AVAILABLE = False
    warnings.warn(
        "Hydra not available. Some configuration features may be limited.",
        ImportWarning,
        stacklevel=2
    )


def create_video_plume(
    cfg: Optional[Union[DictConfig, Dict[str, Any]]] = None,
    **kwargs
) -> VideoPlume:
    """
    Create VideoPlume instance with flexible configuration support.
    
    This factory function provides a unified interface for creating VideoPlume
    instances that supports both Hydra DictConfig objects and traditional
    dictionary-based configuration. It serves as the primary entry point for
    data processing consumers as specified in Section 7.2.1.2.
    
    Args:
        cfg: Hydra DictConfig or dictionary containing configuration parameters.
             If None, parameters must be provided via kwargs.
        **kwargs: Additional configuration parameters that override cfg values
        
    Returns:
        Configured VideoPlume instance ready for simulation integration
        
    Raises:
        ValueError: If configuration validation fails
        IOError: If specified video file cannot be accessed
        
    Examples:
        # Using Hydra configuration (recommended for Kedro pipelines)
        >>> from omegaconf import DictConfig
        >>> cfg = DictConfig({
        ...     "video_path": "data/experiment_001.mp4",
        ...     "flip": True,
        ...     "kernel_size": 5
        ... })
        >>> plume = create_video_plume(cfg)
        
        # Using dictionary configuration (RL frameworks)
        >>> config = {
        ...     "video_path": "videos/plume_data.mp4",
        ...     "grayscale": True,
        ...     "normalize": True
        ... }
        >>> plume = create_video_plume(config)
        
        # Using kwargs (ML analysis workflows)
        >>> plume = create_video_plume(
        ...     video_path="analysis_video.mp4",
        ...     kernel_size=3,
        ...     kernel_sigma=1.5
        ... )
        
        # DVC integration pattern
        >>> plume = create_video_plume_from_dvc_path(
        ...     "data/plume_videos/experiment.mp4"
        ... )
    """
    if cfg is not None:
        # Use factory method for configuration-based creation
        return VideoPlume.from_config(cfg, **kwargs)
    elif kwargs:
        # Direct instantiation from kwargs
        return VideoPlume(**kwargs)
    else:
        raise ValueError(
            "Either cfg parameter or keyword arguments must be provided "
            "to create VideoPlume instance"
        )


# Kedro pipeline integration utilities
def get_video_plume_catalog_entry(
    dataset_name: str,
    video_path: str,
    processing_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate Kedro catalog entry for VideoPlume dataset integration.
    
    This utility function creates standardized Kedro catalog entries that enable
    seamless integration with data science pipelines. The generated catalog entry
    follows Kedro conventions for dataset definition and parameter management.
    
    Args:
        dataset_name: Name for the dataset in Kedro catalog
        video_path: Path to video file (supports templating)
        processing_params: Optional processing configuration parameters
        
    Returns:
        Dictionary suitable for Kedro catalog.yml configuration
        
    Example:
        >>> catalog_entry = get_video_plume_catalog_entry(
        ...     dataset_name="experiment_video",
        ...     video_path="data/01_raw/experiment_001.mp4",
        ...     processing_params={
        ...         "flip": True,
        ...         "kernel_size": 5,
        ...         "grayscale": True
        ...     }
        ... )
        >>> # Add to conf/base/catalog.yml:
        >>> # experiment_video:
        >>> #   type: {{cookiecutter.project_slug}}.data.VideoPlume
        >>> #   filepath: data/01_raw/experiment_001.mp4
        >>> #   ...
    """
    params = processing_params or {}
    
    catalog_entry = {
        f"{dataset_name}": {
            "type": f"{__name__.split('.')[0]}.data.VideoPlume",
            "filepath": video_path,
            "load_args": params.copy(),
            "save_args": {},  # VideoPlume is read-only
        }
    }
    
    return catalog_entry


# ML framework integration utilities  
def get_frame_iterator(video_plume: VideoPlume, batch_size: int = 1):
    """
    Create frame iterator for ML framework integration.
    
    This generator function provides efficient frame-by-frame iteration over
    VideoPlume data with batching support for machine learning workflows.
    The iterator yields NumPy arrays compatible with tensor frameworks.
    
    Args:
        video_plume: VideoPlume instance to iterate over
        batch_size: Number of frames per batch (default: 1)
        
    Yields:
        NumPy arrays containing frame data with shape (batch_size, height, width)
        or (batch_size, height, width, channels) depending on grayscale setting
        
    Example:
        >>> plume = VideoPlume("experiment.mp4")
        >>> for frame_batch in get_frame_iterator(plume, batch_size=8):
        ...     # Process batch with PyTorch/TensorFlow
        ...     predictions = model(torch.from_numpy(frame_batch))
    """
    import numpy as np
    
    frame_count = video_plume.frame_count
    
    for start_idx in range(0, frame_count, batch_size):
        end_idx = min(start_idx + batch_size, frame_count)
        batch_frames = []
        
        for frame_idx in range(start_idx, end_idx):
            frame = video_plume.get_frame(frame_idx)
            if frame is not None:
                batch_frames.append(frame)
        
        if batch_frames:
            # Stack frames into batch tensor
            yield np.stack(batch_frames, axis=0)


# Workflow integration status checking
def check_workflow_integration() -> Dict[str, bool]:
    """
    Check availability of workflow integration dependencies.
    
    This utility function provides runtime detection of workflow orchestration
    capabilities, enabling conditional workflow integration based on available
    dependencies. Supports DVC, Snakemake, and Hydra availability checking.
    
    Returns:
        Dictionary indicating availability of integration components
        
    Example:
        >>> integration_status = check_workflow_integration()
        >>> if integration_status['dvc_available']:
        ...     # Use DVC integration features
        ...     plume = create_video_plume_from_dvc_path("data/video.mp4")
        >>> if integration_status['hydra_available']:
        ...     # Use Hydra configuration features
        ...     plume = VideoPlume.from_config(hydra_cfg)
    """
    status = {
        'hydra_available': HYDRA_AVAILABLE,
        'dvc_available': False,
        'snakemake_available': False,
        'kedro_available': False
    }
    
    # Check DVC availability
    try:
        import dvc
        status['dvc_available'] = True
    except ImportError:
        pass
    
    # Check Snakemake availability
    try:
        import snakemake
        status['snakemake_available'] = True
    except ImportError:
        pass
        
    # Check Kedro availability
    try:
        import kedro
        status['kedro_available'] = True
    except ImportError:
        pass
    
    return status


# Package metadata for discovery and integration
__version__ = "1.0.0"
__author__ = "Odor Plume Navigation Team"

# Define public API surface following Section 7.2.1.2 environment creation interface
__all__ = [
    # Core video plume environment
    "VideoPlume",
    
    # Factory and creation functions  
    "create_video_plume",
    "create_video_plume_from_dvc_path",
    
    # Workflow integration utilities
    "create_snakemake_rule_config",
    "get_video_plume_catalog_entry",
    
    # ML framework integration
    "get_frame_iterator",
    
    # System integration utilities
    "check_workflow_integration",
    
    # Configuration support
    "HYDRA_AVAILABLE",
]


# Module-level documentation for integration patterns
_INTEGRATION_DOCS = """
Integration Patterns:

1. Kedro Pipeline Integration:
   ```python
   # In pipeline nodes
   from {{cookiecutter.project_slug}}.data import VideoPlume
   
   def process_video_node(video_dataset: VideoPlume) -> np.ndarray:
       frames = []
       for i in range(video_dataset.frame_count):
           frame = video_dataset.get_frame(i)
           if frame is not None:
               frames.append(frame)
       return np.stack(frames)
   ```

2. Reinforcement Learning Integration:
   ```python
   # RL environment wrapper
   from {{cookiecutter.project_slug}}.data import VideoPlume
   
   class VideoPlumeEnv(gym.Env):
       def __init__(self, video_path: str):
           self.plume = VideoPlume(video_path=video_path)
           # Define observation/action spaces
           
       def reset(self):
           return self.plume.get_frame(0)
   ```

3. Machine Learning Analysis:
   ```python
   # Batch processing for ML models
   from {{cookiecutter.project_slug}}.data import VideoPlume, get_frame_iterator
   
   plume = VideoPlume("data.mp4", normalize=True)
   for batch in get_frame_iterator(plume, batch_size=32):
       predictions = model.predict(batch)
   ```

4. DVC Workflow Integration:
   ```yaml
   # dvc.yaml pipeline stage
   stages:
     process_video:
       cmd: python process.py ${video_path}
       deps:
       - ${video_path}
       params:
       - flip: true
       - kernel_size: 5
   ```

5. Snakemake Workflow Integration:
   ```python
   # Snakemake rule
   rule process_video:
       input: "data/raw/experiment.mp4"
       output: "data/processed/frames.npz"
       script: "scripts/process_video.py"
   ```
"""


def get_integration_documentation() -> str:
    """
    Get comprehensive integration documentation.
    
    Returns:
        String containing detailed integration patterns and examples
    """
    return _INTEGRATION_DOCS


# Enhanced import support for different usage patterns
def _configure_import_hooks():
    """
    Configure import hooks for enhanced integration support.
    
    This function sets up module-level configuration that enables
    seamless integration with various scientific computing frameworks
    while maintaining backward compatibility.
    """
    # Add VideoPlume to module namespace for direct access
    globals()['VideoPlume'] = VideoPlume
    
    # Enable shortened import patterns when possible
    if HYDRA_AVAILABLE:
        globals()['DictConfig'] = DictConfig


# Initialize import hooks
_configure_import_hooks()