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
        >>> from odor_plume_nav.data.video_plume import VideoPlume
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
import time
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

# Import configuration schema for validation from unified package structure
from odor_plume_nav.config.models import VideoPlumeConfig

# Import cache and logging dependencies for frame caching integration
try:
    from odor_plume_nav.cache.frame_cache import FrameCache
    FRAME_CACHE_AVAILABLE = True
except ImportError:
    FRAME_CACHE_AVAILABLE = False
    FrameCache = None
    logger.debug("FrameCache not available, running without cache support")

try:
    from odor_plume_nav.utils.logging_setup import (
        PerformanceMetrics, 
        get_correlation_context,
        create_step_timer
    )
    LOGGING_UTILS_AVAILABLE = True
except ImportError:
    LOGGING_UTILS_AVAILABLE = False
    logger.debug("Enhanced logging utilities not available, using basic logging")


class VideoPlume:
    """
    Unified VideoPlume implementation with comprehensive video processing capabilities.
    
    This class consolidates video-based odor plume environment functionality with 
    enhanced Hydra configuration integration, OpenCV processing, workflow 
    orchestration support, and intelligent frame caching for performance optimization.
    Designed for research workflows requiring reproducible video processing with 
    flexible configuration management and sub-10ms frame retrieval performance.
    
    Features:
        - Factory method creation with Hydra DictConfig support
        - OpenCV video processing with frame-by-frame access
        - Configurable preprocessing (grayscale, flipping, Gaussian smoothing)
        - Thread-safe resource management with automatic cleanup
        - Environment variable interpolation for secure path management
        - Workflow integration points for DVC and Snakemake
        - Enhanced metadata extraction for research documentation
        - Intelligent frame caching with LRU and full-preload modes
        - Zero-copy frame retrieval for performance optimization
        - Cache hit/miss statistics and performance metrics tracking
        - Sub-10ms frame access performance for RL training workflows
    
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
        cache (Optional[FrameCache]): Frame cache instance for performance optimization
        cache_stats (Dict): Cache performance statistics (hits, misses, retrieval times)
    
    Thread Safety:
        All public methods are thread-safe. Internal OpenCV VideoCapture 
        operations and cache access are protected by locks to prevent 
        concurrent access issues in multi-agent scenarios.
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
        cache: Optional[Any] = None,
        **kwargs
    ) -> None:
        """
        Initialize VideoPlume with specified parameters and optional frame caching.
        
        Args:
            video_path: Path to the video file containing plume data
            flip: Whether to flip video frames horizontally for coordinate system alignment
            grayscale: Whether to convert frames to grayscale for processing efficiency
            kernel_size: Gaussian kernel size for smoothing (must be odd and positive)
            kernel_sigma: Gaussian kernel sigma for smoothing (must be positive)
            threshold: Threshold value for binary plume detection (0.0 to 1.0)
            normalize: Whether to normalize frame values to [0, 1] range
            cache: Optional FrameCache instance for performance optimization
            **kwargs: Additional parameters for future extensibility
            
        Raises:
            IOError: If video file does not exist or cannot be opened
            ValueError: If kernel parameters are invalid or inconsistent
            
        Note:
            Gaussian smoothing is applied only when both kernel_size and 
            kernel_sigma are specified. All preprocessing operations are 
            applied in sequence: grayscale → flip → gaussian → threshold → normalize.
            When cache is provided, frame retrieval performance is optimized
            for sub-10ms access times with hit/miss statistics tracking.
        """
        # Convert and validate video path
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.video_path}")
        
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
        
        # Initialize frame caching system for performance optimization
        self.cache = cache
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "hit_rate": 0.0,
            "total_retrieval_time": 0.0,
            "cache_retrieval_time": 0.0,
            "disk_retrieval_time": 0.0,
            "memory_usage_mb": 0.0
        }
        
        # Initialize cache warming for preload modes
        if self.cache is not None and FRAME_CACHE_AVAILABLE:
            logger.info(f"Frame cache enabled for {self.video_path.name}")
            self._warm_cache_if_needed()
        else:
            logger.debug(f"Frame cache disabled for {self.video_path.name}")
        
        # Thread safety for OpenCV operations and cache access
        self._lock = threading.RLock()
        self._is_closed = False
        
        logger.info(
            f"VideoPlume initialized: {self.video_path.name} "
            f"({self.width}x{self.height}, {self.frame_count} frames, {self.fps:.1f} fps)"
        )
    
    def _warm_cache_if_needed(self) -> None:
        """
        Perform cache warming based on cache mode for full-preload functionality.
        
        This method checks the cache mode and performs appropriate initialization:
        - For LRU mode: Basic cache setup with memory monitoring
        - For full-preload mode: Sequential loading of all frames
        
        Implements cache warming as specified in Section 0.2.2 for batch 
        processing workflows and predictable performance.
        """
        if self.cache is None:
            return
            
        try:
            # Check if cache supports preload mode (full cache mode)
            if hasattr(self.cache, 'is_preload_mode') and self.cache.is_preload_mode():
                logger.info(f"Starting cache preload for {self.frame_count} frames")
                start_time = time.time()
                
                # Preload all frames sequentially for maximum performance
                for frame_idx in range(self.frame_count):
                    # Check if frame is already cached to avoid duplicate work
                    if not self.cache.contains(frame_idx):
                        # Read frame directly for caching (bypass normal preprocessing)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, raw_frame = self.cap.read()
                        
                        if ret:
                            # Process frame and store in cache
                            processed_frame = self._preprocess_frame(raw_frame)
                            self.cache.put(frame_idx, processed_frame)
                        else:
                            logger.warning(f"Failed to read frame {frame_idx} during preload")
                            break
                    
                    # Log progress every 1000 frames
                    if frame_idx % 1000 == 0 and frame_idx > 0:
                        progress = (frame_idx / self.frame_count) * 100
                        logger.debug(f"Cache preload progress: {progress:.1f}% ({frame_idx}/{self.frame_count})")
                
                preload_time = time.time() - start_time
                logger.info(
                    f"Cache preload completed in {preload_time:.2f}s "
                    f"({self.frame_count} frames, {self.frame_count/preload_time:.1f} fps)"
                )
                
            elif hasattr(self.cache, 'is_lru_mode') and self.cache.is_lru_mode():
                logger.info("LRU cache mode initialized - frames will be cached on demand")
                
            else:
                logger.debug("Cache warming not needed for current cache mode")
                
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
            # Continue without cache rather than failing completely
            self.cache = None
    
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
        elif hasattr(cfg, 'model_dump'):  # Pydantic v2 model
            # Already a validated Pydantic model, convert to dict
            config_dict = cfg.model_dump()
        elif hasattr(cfg, 'dict'):  # Pydantic v1 model
            # Already a validated Pydantic model, convert to dict
            config_dict = cfg.dict()
        elif hasattr(cfg, '__dict__'):  # Generic object with attributes
            # Convert object attributes to dict
            config_dict = cfg.__dict__
        else:
            config_dict = dict(cfg)
        
        try:
            # Validate configuration through Pydantic schema
            # Note: _skip_validation=True prevents file existence check during validation
            # since we want to handle that in the main constructor
            config_for_validation = {**config_dict, "_skip_validation": True}
            
            # Handle different Pydantic versions for validation
            if hasattr(VideoPlumeConfig, 'model_validate'):  # Pydantic v2
                validated_config = VideoPlumeConfig.model_validate(config_for_validation)
                validated_params = validated_config.model_dump(exclude={"_skip_validation"})
            elif hasattr(VideoPlumeConfig, 'parse_obj'):  # Pydantic v1
                validated_config = VideoPlumeConfig.parse_obj(config_for_validation)
                validated_params = validated_config.dict(exclude={"_skip_validation"})
            else:
                # Fallback: direct instantiation (exclude _skip_validation as it's not a real field)
                config_for_direct_validation = {k: v for k, v in config_for_validation.items() 
                                              if k != "_skip_validation"}
                validated_config = VideoPlumeConfig(**config_for_direct_validation)
                if hasattr(validated_config, 'dict'):
                    validated_params = validated_config.dict()
                elif hasattr(validated_config, 'model_dump'):
                    validated_params = validated_config.model_dump()
                else:
                    validated_params = validated_config.__dict__.copy()
            
            # Handle cache parameter separately if present in original config
            if isinstance(cfg, dict) and 'cache' in cfg:
                validated_params['cache'] = cfg['cache']
            elif HYDRA_AVAILABLE and isinstance(cfg, DictConfig) and 'cache' in cfg:
                validated_params['cache'] = cfg['cache']
            
            return cls(**validated_params)
            
        except Exception as e:
            logger.error(f"VideoPlume configuration validation failed: {e}")
            raise ValueError(f"Invalid VideoPlume configuration: {e}") from e
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Retrieve and preprocess a video frame by index with intelligent caching.
        
        Implements high-performance frame retrieval with cache integration:
        1. Check frame cache for sub-10ms retrieval (if enabled)
        2. Frame extraction from video (on cache miss)
        3. Grayscale conversion (if enabled)
        4. Horizontal flipping (if enabled)
        5. Gaussian smoothing (if configured)
        6. Threshold application (if configured)
        7. Normalization to [0, 1] range (if enabled)
        8. Cache storage and statistics update
        
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
            Cache hit/miss statistics are tracked and available via get_cache_stats().
            
        Example:
            >>> frame = plume.get_frame(100)
            >>> if frame is not None:
            ...     print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            >>> stats = plume.get_cache_stats()
            >>> print(f"Cache hit rate: {stats['hit_rate']:.2%}")
        """
        with self._lock:
            if self._is_closed:
                raise ValueError("Cannot get frame from closed VideoPlume")
            
            # Validate frame index bounds
            if frame_idx < 0 or frame_idx >= self.frame_count:
                logger.debug(f"Frame index {frame_idx} out of bounds [0, {self.frame_count})")
                return None
            
            # Start performance timing for metrics collection
            retrieval_start = time.time()
            
            # Update total request counter
            self.cache_stats["total_requests"] += 1
            
            # Check cache first for optimized frame retrieval
            if self.cache is not None and FRAME_CACHE_AVAILABLE:
                try:
                    # Attempt zero-copy frame retrieval from cache
                    cached_frame = self.cache.get(frame_idx)
                    
                    if cached_frame is not None:
                        # Cache hit - update statistics
                        cache_time = time.time() - retrieval_start
                        self._update_cache_hit_stats(cache_time)
                        
                        # Return numpy array view for zero-copy access
                        if isinstance(cached_frame, np.ndarray):
                            return cached_frame.view()  # Zero-copy view
                        else:
                            return cached_frame
                            
                except Exception as e:
                    logger.debug(f"Cache access failed for frame {frame_idx}: {e}")
                    # Fall through to disk I/O on cache errors
            
            # Cache miss or no cache - read from video file
            disk_start = time.time()
            
            # Seek to the requested frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}")
                return None
            
            # Apply preprocessing pipeline
            processed_frame = self._preprocess_frame(frame)
            
            # Store in cache for future access
            if self.cache is not None and FRAME_CACHE_AVAILABLE and processed_frame is not None:
                try:
                    self.cache.put(frame_idx, processed_frame)
                except Exception as e:
                    logger.debug(f"Cache storage failed for frame {frame_idx}: {e}")
            
            # Update cache miss statistics
            disk_time = time.time() - disk_start
            total_time = time.time() - retrieval_start
            self._update_cache_miss_stats(disk_time, total_time)
            
            return processed_frame
    
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
    
    def _update_cache_hit_stats(self, retrieval_time: float) -> None:
        """
        Update cache hit statistics with timing information.
        
        Args:
            retrieval_time: Time taken for cache retrieval in seconds
        """
        self.cache_stats["hits"] += 1
        self.cache_stats["cache_retrieval_time"] += retrieval_time
        self.cache_stats["total_retrieval_time"] += retrieval_time
        self._recalculate_hit_rate()
        
        # Log performance violations for sub-10ms requirement
        if retrieval_time > 0.010 and LOGGING_UTILS_AVAILABLE:
            logger.warning(
                f"Cache hit exceeded 10ms threshold: {retrieval_time*1000:.1f}ms",
                extra={
                    "metric_type": "cache_hit_latency_violation",
                    "actual_latency_ms": retrieval_time * 1000,
                    "threshold_latency_ms": 10.0
                }
            )
    
    def _update_cache_miss_stats(self, disk_time: float, total_time: float) -> None:
        """
        Update cache miss statistics with timing information.
        
        Args:
            disk_time: Time taken for disk I/O in seconds
            total_time: Total time including processing in seconds
        """
        self.cache_stats["misses"] += 1
        self.cache_stats["disk_retrieval_time"] += disk_time
        self.cache_stats["total_retrieval_time"] += total_time
        self._recalculate_hit_rate()
    
    def _recalculate_hit_rate(self) -> None:
        """Recalculate cache hit rate and update memory usage."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total > 0:
            self.cache_stats["hit_rate"] = self.cache_stats["hits"] / total
        
        # Update memory usage if cache is available
        if self.cache is not None and hasattr(self.cache, 'get_memory_usage'):
            try:
                self.cache_stats["memory_usage_mb"] = self.cache.get_memory_usage()
            except Exception:
                pass  # Ignore memory usage collection errors
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache performance statistics.
        
        Returns:
            Dictionary containing cache performance metrics including:
            - hit_rate: Percentage of cache hits (0.0 to 1.0)
            - hits/misses: Raw counters for cache operations
            - average retrieval times for cache vs disk access
            - memory usage information
            
        Example:
            >>> stats = plume.get_cache_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
            >>> print(f"Avg cache time: {stats['avg_cache_time_ms']:.1f}ms")
        """
        stats = self.cache_stats.copy()
        
        # Calculate average retrieval times
        if stats["hits"] > 0:
            stats["avg_cache_time_ms"] = (stats["cache_retrieval_time"] / stats["hits"]) * 1000
        else:
            stats["avg_cache_time_ms"] = 0.0
            
        if stats["misses"] > 0:
            stats["avg_disk_time_ms"] = (stats["disk_retrieval_time"] / stats["misses"]) * 1000
        else:
            stats["avg_disk_time_ms"] = 0.0
            
        if stats["total_requests"] > 0:
            stats["avg_total_time_ms"] = (stats["total_retrieval_time"] / stats["total_requests"]) * 1000
        else:
            stats["avg_total_time_ms"] = 0.0
        
        # Add cache-specific metrics if available
        if self.cache is not None and hasattr(self.cache, 'get_stats'):
            try:
                cache_internal_stats = self.cache.get_stats()
                stats.update(cache_internal_stats)
            except Exception:
                pass  # Ignore cache internal stats errors
        
        return stats
    
    def warm_cache(self, start_frame: int = 0, end_frame: Optional[int] = None) -> None:
        """
        Warm cache by preloading a range of frames for predictable performance.
        
        This method allows explicit cache warming for specific frame ranges,
        useful for batch processing workflows where predictable access patterns
        are known in advance.
        
        Args:
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (exclusive), None for all remaining frames
            
        Example:
            >>> # Warm cache for first 1000 frames
            >>> plume.warm_cache(0, 1000)
            >>> 
            >>> # Warm cache for specific range
            >>> plume.warm_cache(500, 1500)
        """
        if self.cache is None:
            logger.debug("Cache warming skipped - no cache instance available")
            return
            
        if not FRAME_CACHE_AVAILABLE:
            logger.debug("Cache warming skipped - FrameCache not available")
            return
        
        end_frame = end_frame or self.frame_count
        end_frame = min(end_frame, self.frame_count)
        
        if start_frame >= end_frame:
            logger.warning(f"Invalid frame range for warming: {start_frame} >= {end_frame}")
            return
        
        logger.info(f"Warming cache for frames {start_frame} to {end_frame-1}")
        warm_start = time.time()
        
        with self._lock:
            for frame_idx in range(start_frame, end_frame):
                try:
                    # Check if frame is already cached
                    if not self.cache.contains(frame_idx):
                        # Read and cache the frame
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, raw_frame = self.cap.read()
                        
                        if ret:
                            processed_frame = self._preprocess_frame(raw_frame)
                            self.cache.put(frame_idx, processed_frame)
                        else:
                            logger.warning(f"Failed to read frame {frame_idx} during warming")
                    
                    # Log progress every 500 frames
                    if (frame_idx - start_frame) % 500 == 0 and frame_idx > start_frame:
                        progress = ((frame_idx - start_frame) / (end_frame - start_frame)) * 100
                        logger.debug(f"Cache warming progress: {progress:.1f}%")
                        
                except Exception as e:
                    logger.error(f"Cache warming failed for frame {frame_idx}: {e}")
                    break
        
        warm_time = time.time() - warm_start
        frame_count = end_frame - start_frame
        logger.info(
            f"Cache warming completed: {frame_count} frames in {warm_time:.2f}s "
            f"({frame_count/warm_time:.1f} fps)"
        )
    
    def clear_cache(self) -> None:
        """
        Clear all cached frames and reset cache statistics.
        
        Useful for memory management in long-running processes or when
        switching between different video processing configurations.
        """
        if self.cache is not None and hasattr(self.cache, 'clear'):
            try:
                self.cache.clear()
                logger.info("Frame cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
        
        # Reset statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "hit_rate": 0.0,
            "total_retrieval_time": 0.0,
            "cache_retrieval_time": 0.0,
            "disk_retrieval_time": 0.0,
            "memory_usage_mb": 0.0
        }
    
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
            Dictionary containing video properties, preprocessing configuration,
            and cache performance statistics:
            - width: Frame width in pixels
            - height: Frame height in pixels  
            - fps: Frame rate in frames per second
            - frame_count: Total number of frames
            - duration: Video duration in seconds
            - shape: Frame shape as (height, width) tuple
            - preprocessing: Applied preprocessing configuration
            - cache_enabled: Whether frame caching is active
            - cache_stats: Cache performance metrics (if enabled)
            
        Example:
            >>> metadata = plume.get_metadata()
            >>> print(f"Video: {metadata['width']}x{metadata['height']}")
            >>> print(f"Duration: {metadata['duration']:.2f}s")
            >>> if metadata['cache_enabled']:
            ...     print(f"Cache hit rate: {metadata['cache_stats']['hit_rate']:.2%}")
        """
        metadata = {
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
            },
            "cache_enabled": self.cache is not None,
        }
        
        # Add cache statistics if caching is enabled
        if self.cache is not None:
            metadata["cache_stats"] = self.get_cache_stats()
            if hasattr(self.cache, 'get_config'):
                try:
                    metadata["cache_config"] = self.cache.get_config()
                except Exception:
                    pass  # Ignore cache config errors
        
        return metadata
    
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
        
        # Build cache information string
        cache_str = "disabled"
        if metadata["cache_enabled"]:
            cache_stats = metadata.get("cache_stats", {})
            hit_rate = cache_stats.get("hit_rate", 0.0)
            memory_mb = cache_stats.get("memory_usage_mb", 0.0)
            cache_str = f"enabled (hit rate: {hit_rate:.1%}, memory: {memory_mb:.1f}MB)"
        
        return (
            f"Video: {self.video_path.name}\n"
            f"Dimensions: {metadata['width']}x{metadata['height']} pixels\n"
            f"Frame rate: {metadata['fps']:.1f} fps\n"
            f"{metadata['frame_count']} frames\n"
            f"Duration: {metadata['duration']:.2f} seconds\n"
            f"Preprocessing: {preprocessing_str}\n"
            f"Frame cache: {cache_str}"
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
        Close video capture and release resources including frame cache.
        
        Releases the OpenCV VideoCapture object, clears the frame cache,
        and marks the instance as closed. After calling this method, 
        attempts to get frames will raise ValueError.
        This method is idempotent and thread-safe.
        
        Note:
            This method is automatically called by __del__ for cleanup,
            but explicit calling is recommended for deterministic resource management.
        """
        with self._lock:
            if not self._is_closed:
                # Close video capture
                if self.cap is not None:
                    self.cap.release()
                
                # Clear cache to free memory
                if self.cache is not None:
                    try:
                        self.clear_cache()
                    except Exception as e:
                        logger.debug(f"Error clearing cache during close: {e}")
                
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
def create_video_plume(cfg: Union[DictConfig, Dict], cache: Optional[Any] = None, **kwargs) -> VideoPlume:
    """
    Factory function for creating VideoPlume instances with configuration override and cache support.
    
    Provides an alternative factory method compatible with the API layer
    while supporting parameter overrides for dynamic configuration and
    optional frame caching for performance optimization.
    
    Args:
        cfg: Hydra DictConfig or dictionary with video plume configuration
        cache: Optional FrameCache instance for performance optimization
        **kwargs: Additional parameters to override configuration values
        
    Returns:
        VideoPlume: Configured instance with applied overrides and cache integration
        
    Example:
        >>> from odor_plume_nav.api.navigation import create_video_plume
        >>> from odor_plume_nav.cache.frame_cache import FrameCache
        >>> 
        >>> cache = FrameCache(mode="lru", memory_limit="1GiB")
        >>> plume = create_video_plume(cfg.video_plume, cache=cache, flip=True)
    """
    # Merge configuration with overrides
    if HYDRA_AVAILABLE and isinstance(cfg, DictConfig):
        config_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        config_dict = dict(cfg)
    
    # Apply overrides
    config_dict.update(kwargs)
    
    # Add cache to configuration if provided
    if cache is not None:
        config_dict['cache'] = cache
    
    return VideoPlume.from_config(config_dict)


# Re-export main class and factory functions for public API
__all__ = [
    "VideoPlume", 
    "create_video_plume",
    "FRAME_CACHE_AVAILABLE",
    "LOGGING_UTILS_AVAILABLE"
]