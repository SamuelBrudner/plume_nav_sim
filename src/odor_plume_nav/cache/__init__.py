"""
High-Performance Frame Caching System for Odor Plume Navigation.

This package provides a comprehensive in-memory frame caching system with dual-mode 
operation (LRU and full-preload), zero-copy operations, and extensive performance 
monitoring. Designed to achieve sub-10ms step latency requirements for reinforcement 
learning workflows through intelligent memory management and optimized data structures.

The cache system integrates seamlessly with VideoPlume and GymnasiumEnv components 
while maintaining graceful degradation when cache dependencies are unavailable. It 
provides structured logging capabilities and comprehensive performance metrics 
suitable for production RL training environments.

Key Features:
    - Dual-mode caching: LRU eviction and full-preload strategies
    - Thread-safe operations supporting 100+ concurrent agents
    - Zero-copy NumPy frame access with memory-mapped operations
    - Comprehensive statistics tracking (hit rate >90% target)
    - Memory pressure monitoring with psutil integration
    - Configurable 2 GiB default memory limit with intelligent overflow handling
    - Automatic cache warming and preload capabilities
    - Integration with Loguru structured logging and performance metrics

Architecture:
    - LRU Mode: OrderedDict-based O(1) access with intelligent eviction
    - Preload Mode: Sequential frame loading with integrity validation
    - Memory Management: Real-time pressure monitoring with graceful degradation
    - Statistics: Atomic counters with thread-safe access patterns

Performance Targets:
    - <10ms frame retrieval latency (cache hits)
    - >90% cache hit rate for sequential access patterns
    - Support for 100+ concurrent agent access
    - 2 GiB memory limit with automatic pressure management

Supported Import Patterns:

    Basic FrameCache usage:
        >>> from odor_plume_nav.cache import FrameCache, CacheMode
        >>> cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=2048)
        >>> frame = cache.get(frame_id=100, video_plume=plume_instance)
    
    Factory method creation:
        >>> from odor_plume_nav.cache import create_frame_cache
        >>> cache = create_frame_cache("lru", memory_limit_mb=1024)
        >>> # Alternative factory methods
        >>> lru_cache = create_lru_cache(memory_limit_mb=2048)
        >>> preload_cache = create_preload_cache(memory_limit_mb=4096)
    
    Configuration integration:
        >>> from odor_plume_nav.cache import create_frame_cache_from_config
        >>> from hydra import compose, initialize
        >>> 
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     cache = create_frame_cache_from_config(cfg.frame_cache)
    
    Environment variable integration:
        >>> import os
        >>> os.environ['FRAME_CACHE_MODE'] = 'lru'
        >>> os.environ['FRAME_CACHE_SIZE_MB'] = '1024'
        >>> cache = create_frame_cache_from_env()
    
    Performance monitoring:
        >>> cache = FrameCache(mode="lru")
        >>> stats = cache.get_performance_stats()
        >>> hit_rate = cache.hit_rate  # Real-time hit rate
        >>> print(f"Cache efficiency: {hit_rate:.1%}")
    
    Contextual usage with automatic cleanup:
        >>> with FrameCache(mode="lru", memory_limit_mb=512) as cache:
        ...     frame = cache.get(frame_id=42, video_plume=plume)
        ...     # Cache automatically cleared on exit

CLI Integration:
    The cache system integrates with the command-line interface through the 
    `--frame_cache` parameter:
    
        >>> # CLI usage examples
        >>> # odor_plume_nav run --frame_cache lru --config experiment.yaml
        >>> # odor_plume_nav train --frame_cache all --memory_limit 4096
    
    Supported cache modes:
        - none: Direct I/O with no caching for memory-constrained environments
        - lru: Intelligent caching with automatic eviction for balanced performance
        - all: Full preload strategy for maximum throughput scenarios

Environment Variable Support:
    The package supports comprehensive environment variable configuration:
    
        >>> import os
        >>> os.environ['FRAME_CACHE_MODE'] = 'lru'
        >>> os.environ['FRAME_CACHE_SIZE_MB'] = '2048'
        >>> os.environ['FRAME_CACHE_PRESSURE_THRESHOLD'] = '0.9'
        >>> # In Hydra config: frame_cache: ${oc.env:FRAME_CACHE_MODE}

Workflow Integration:

    Gymnasium Environment integration:
        >>> from odor_plume_nav.environments import GymnasiumEnv
        >>> cache = create_frame_cache("lru", memory_limit_mb=1024)
        >>> env = GymnasiumEnv(video_path="plume.mp4", frame_cache=cache)
        >>> obs, info = env.reset()
        >>> # Access performance metrics
        >>> perf_stats = info["perf_stats"]
        >>> cache_hit_rate = perf_stats["cache_hit_rate"]
    
    VideoPlume integration:
        >>> from odor_plume_nav.data import VideoPlume
        >>> from odor_plume_nav.cache import FrameCache
        >>> cache = FrameCache(mode="all", memory_limit_mb=4096)
        >>> plume = VideoPlume("experiment.mp4", frame_cache=cache)
        >>> # Frames automatically cached during access
    
    Batch processing optimization:
        >>> cache = create_preload_cache(memory_limit_mb=8192)
        >>> cache.preload(range(0, 1000), video_plume)
        >>> # Process frames with <10ms latency
        >>> for frame_id in range(1000):
        ...     frame = cache.get(frame_id, video_plume)

Monitoring and Observability:
    
    Performance metrics access:
        >>> stats = cache.get_performance_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
        >>> print(f"Memory usage: {stats['memory_usage_mb']:.1f}MB")
        >>> print(f"Average hit time: {stats['average_hit_time_ms']:.2f}ms")
    
    Structured logging integration:
        >>> # Automatic JSON logging with correlation IDs
        >>> # Logs include cache hit/miss metrics embedded in perf_stats
        >>> # Memory pressure warnings with psutil integration
    
    Diagnostic utilities:
        >>> diagnostics = diagnose_cache_setup()
        >>> print(f"Cache available: {diagnostics['cache_available']}")
        >>> print(f"Memory monitoring: {diagnostics['psutil_available']}")
"""

import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from pathlib import Path

# Import centralized logging setup first for structured diagnostics
try:
    from odor_plume_nav.utils.logging_setup import get_enhanced_logger
    logger = get_enhanced_logger(__name__)
    LOGGING_AVAILABLE = True
except ImportError:
from loguru import logger
    LOGGING_AVAILABLE = False

# Core cache functionality with graceful degradation
try:
    from .frame_cache import (
        FrameCache,
        CacheMode,
        CacheStatistics,
        create_lru_cache,
        create_preload_cache,
        create_no_cache
    )
    CACHE_AVAILABLE = True
    logger.info(
        "Frame cache system available for high-performance video processing",
        extra={
            "metric_type": "cache_capability",
            "feature": "frame_caching",
            "modes_available": ["none", "lru", "all"],
            "performance_target": "sub_10ms"
        }
    ) if LOGGING_AVAILABLE else None
except ImportError as e:
    # Graceful degradation when cache dependencies are unavailable
    FrameCache = None
    CacheMode = None
    CacheStatistics = None
    create_lru_cache = None
    create_preload_cache = None
    create_no_cache = None
    CACHE_AVAILABLE = False
    logger.warning(
        f"Frame cache system not available: {e}",
        extra={
            "metric_type": "cache_limitation",
            "missing_feature": "frame_caching",
            "fallback_mode": "direct_io_only"
        }
    ) if LOGGING_AVAILABLE else None

# Check for optional dependencies with availability flags
try:
    import psutil
    MEMORY_MONITORING_AVAILABLE = True
except ImportError:
    psutil = None
    MEMORY_MONITORING_AVAILABLE = False

try:
    from omegaconf import DictConfig
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    # Fallback type hint for when Hydra is not available
    DictConfig = Dict[str, Any]

# List of all available exports - will be updated based on successful imports
__all__ = []

# Add cache exports if available
if CACHE_AVAILABLE:
    __all__.extend([
        "FrameCache",
        "CacheMode", 
        "CacheStatistics",
        "create_lru_cache",
        "create_preload_cache",
        "create_no_cache"
    ])

def create_frame_cache(
    mode: Union[str, "CacheMode"] = "lru",
    memory_limit_mb: float = 2048.0,
    **kwargs
) -> Optional["FrameCache"]:
    """
    Factory function for creating FrameCache instances with intelligent defaults.
    
    Provides a unified interface for cache creation supporting both string and enum
    mode specifications. Includes comprehensive error handling and graceful
    degradation when cache system is unavailable.
    
    Args:
        mode: Cache operational mode ("none", "lru", "all" or CacheMode enum)
        memory_limit_mb: Maximum memory allocation in MB (default: 2048 MB = 2 GiB)
        **kwargs: Additional parameters passed to FrameCache constructor
        
    Returns:
        Configured FrameCache instance or None if cache system unavailable
        
    Raises:
        ValueError: If mode is invalid or memory_limit_mb is non-positive
        ImportError: If cache dependencies are unavailable (with graceful fallback)
        
    Example:
        >>> # Basic LRU cache
        >>> cache = create_frame_cache("lru", memory_limit_mb=1024)
        >>> 
        >>> # High-memory preload cache
        >>> cache = create_frame_cache("all", memory_limit_mb=4096)
        >>> 
        >>> # Memory-constrained direct I/O
        >>> cache = create_frame_cache("none")
        
        >>> # With additional configuration
        >>> cache = create_frame_cache(
        ...     mode="lru",
        ...     memory_limit_mb=2048,
        ...     memory_pressure_threshold=0.85,
        ...     enable_statistics=True
        ... )
    """
    if not CACHE_AVAILABLE:
        logger.warning(
            "Frame cache system unavailable, returning None",
            extra={
                "metric_type": "cache_creation_failure",
                "requested_mode": str(mode),
                "fallback": "direct_io"
            }
        ) if LOGGING_AVAILABLE else None
        return None
    
    if memory_limit_mb <= 0:
        raise ValueError("memory_limit_mb must be positive")
    
    try:
        cache = FrameCache(mode=mode, memory_limit_mb=memory_limit_mb, **kwargs)
        
        logger.info(
            f"Created FrameCache: mode={cache.mode.value}, limit={memory_limit_mb}MB",
            extra={
                "metric_type": "cache_creation_success",
                "cache_mode": cache.mode.value,
                "memory_limit_mb": memory_limit_mb,
                "additional_params": len(kwargs)
            }
        ) if LOGGING_AVAILABLE else None
        
        return cache
        
    except Exception as e:
        logger.error(
            f"Failed to create FrameCache: {e}",
            extra={
                "metric_type": "cache_creation_error",
                "requested_mode": str(mode),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        ) if LOGGING_AVAILABLE else None
        raise

__all__.append("create_frame_cache")


def create_frame_cache_from_config(
    config: Union[Dict[str, Any], "DictConfig"],
    **kwargs
) -> Optional["FrameCache"]:
    """
    Create FrameCache instance from Hydra configuration.
    
    Supports comprehensive configuration loading with parameter validation
    and intelligent defaults. Integrates seamlessly with Hydra's configuration
    composition system and environment variable interpolation.
    
    Args:
        config: Configuration dictionary or DictConfig containing cache settings
        **kwargs: Additional parameters to override configuration values
        
    Returns:
        Configured FrameCache instance or None if unavailable
        
    Raises:
        ValueError: If configuration contains invalid parameters
        KeyError: If required configuration keys are missing
        
    Example:
        >>> # From Hydra configuration
        >>> with initialize(config_path="../conf"):
        ...     cfg = compose(config_name="config")
        ...     cache = create_frame_cache_from_config(cfg.frame_cache)
        
        >>> # From dictionary configuration
        >>> config = {
        ...     "mode": "lru",
        ...     "memory_limit_mb": 2048,
        ...     "memory_pressure_threshold": 0.9
        ... }
        >>> cache = create_frame_cache_from_config(config)
        
        >>> # With configuration overrides
        >>> cache = create_frame_cache_from_config(
        ...     cfg.frame_cache, 
        ...     memory_limit_mb=4096
        ... )
    """
    if not CACHE_AVAILABLE:
        logger.warning(
            "Frame cache system unavailable, cannot create from config",
            extra={
                "metric_type": "cache_config_creation_failure",
                "config_provided": config is not None
            }
        ) if LOGGING_AVAILABLE else None
        return None
    
    if config is None:
        raise ValueError("config cannot be None")
    
    # Convert DictConfig to regular dict if needed
    if hasattr(config, '_content'):
        # OmegaConf DictConfig
        config_dict = dict(config)
    elif hasattr(config, 'items'):
        # Regular dict-like object
        config_dict = dict(config)
    else:
        raise ValueError(f"Invalid config type: {type(config)}")
    
    # Merge configuration with kwargs overrides
    merged_config = config_dict.copy()
    merged_config.update(kwargs)
    
    # Extract and validate required parameters
    mode = merged_config.get("mode", "lru")
    memory_limit_mb = merged_config.get("memory_limit_mb", 2048.0)
    
    # Pass all configuration to create_frame_cache
    return create_frame_cache(mode=mode, memory_limit_mb=memory_limit_mb, **{
        k: v for k, v in merged_config.items() 
        if k not in ["mode", "memory_limit_mb"]
    })

__all__.append("create_frame_cache_from_config")


def create_frame_cache_from_env(
    mode_env_var: str = "FRAME_CACHE_MODE",
    size_env_var: str = "FRAME_CACHE_SIZE_MB",
    **kwargs
) -> Optional["FrameCache"]:
    """
    Create FrameCache instance from environment variables.
    
    Convenient factory method for creating FrameCache instances using environment
    variable configuration, supporting secure deployment and configuration
    management in production environments.
    
    Args:
        mode_env_var: Environment variable containing cache mode
        size_env_var: Environment variable containing memory limit in MB
        **kwargs: Additional parameters passed to FrameCache constructor
        
    Returns:
        FrameCache instance configured from environment variables
        
    Raises:
        EnvironmentError: If required environment variables are not set
        ValueError: If environment variable values are invalid
        
    Example:
        >>> import os
        >>> os.environ['FRAME_CACHE_MODE'] = 'lru'
        >>> os.environ['FRAME_CACHE_SIZE_MB'] = '1024'
        >>> cache = create_frame_cache_from_env()
        
        >>> # With custom environment variables
        >>> os.environ['MY_CACHE_MODE'] = 'all'
        >>> os.environ['MY_CACHE_SIZE'] = '4096'
        >>> cache = create_frame_cache_from_env(
        ...     mode_env_var="MY_CACHE_MODE",
        ...     size_env_var="MY_CACHE_SIZE"
        ... )
    """
    if not CACHE_AVAILABLE:
        logger.warning(
            "Frame cache system unavailable, cannot create from environment",
            extra={
                "metric_type": "cache_env_creation_failure",
                "mode_env_var": mode_env_var,
                "size_env_var": size_env_var
            }
        ) if LOGGING_AVAILABLE else None
        return None
    
    import os
    
    # Get cache mode from environment variable
    mode = os.getenv(mode_env_var)
    if mode is None:
        raise EnvironmentError(
            f"Environment variable '{mode_env_var}' not set. "
            f"Please set it to one of: none, lru, all"
        )
    
    # Get memory limit from environment variable
    memory_limit_str = os.getenv(size_env_var)
    if memory_limit_str is None:
        raise EnvironmentError(
            f"Environment variable '{size_env_var}' not set. "
            f"Please set it to the memory limit in MB (e.g., 2048)"
        )
    
    try:
        memory_limit_mb = float(memory_limit_str)
    except ValueError:
        raise ValueError(
            f"Invalid memory limit in '{size_env_var}': {memory_limit_str}. "
            f"Must be a positive number representing MB"
        )
    
    # Create cache with environment configuration
    return create_frame_cache(mode=mode, memory_limit_mb=memory_limit_mb, **kwargs)

__all__.append("create_frame_cache_from_env")


def get_cache_modes() -> List[str]:
    """
    Get list of available cache modes.
    
    Returns:
        List of supported cache mode strings
        
    Example:
        >>> modes = get_cache_modes()
        >>> print(f"Available modes: {', '.join(modes)}")
    """
    if CACHE_AVAILABLE and CacheMode:
        return [mode.value for mode in CacheMode]
    return ["none", "lru", "all"]  # Fallback list

__all__.append("get_cache_modes")


def validate_cache_config(
    mode: str,
    memory_limit_mb: float,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate cache configuration parameters.
    
    Performs comprehensive validation of cache configuration including
    mode validity, memory constraints, and parameter compatibility.
    Useful for configuration validation in automated pipelines.
    
    Args:
        mode: Cache mode string to validate
        memory_limit_mb: Memory limit to validate
        **kwargs: Additional cache parameters to validate
        
    Returns:
        Dictionary containing validation results and recommendations
        
    Example:
        >>> result = validate_cache_config("lru", 2048)
        >>> if result["valid"]:
        ...     cache = create_frame_cache(mode, memory_limit_mb)
        ... else:
        ...     print(f"Invalid config: {result['errors']}")
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "recommendations": []
    }
    
    # Validate cache mode
    valid_modes = get_cache_modes()
    if mode not in valid_modes:
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"Invalid cache mode '{mode}'. Valid modes: {valid_modes}"
        )
    
    # Validate memory limit
    if memory_limit_mb <= 0:
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"Memory limit must be positive, got {memory_limit_mb}"
        )
    elif memory_limit_mb < 128:
        validation_result["warnings"].append(
            f"Memory limit {memory_limit_mb}MB is very low, may cause frequent evictions"
        )
    elif memory_limit_mb > 8192:
        validation_result["warnings"].append(
            f"Memory limit {memory_limit_mb}MB is very high, monitor system memory usage"
        )
    
    # Check memory monitoring availability
    if not MEMORY_MONITORING_AVAILABLE:
        validation_result["warnings"].append(
            "psutil not available - memory pressure monitoring disabled"
        )
        validation_result["recommendations"].append(
            "Install psutil for enhanced memory monitoring: pip install psutil"
        )
    
    # Check cache system availability
    if not CACHE_AVAILABLE:
        validation_result["valid"] = False
        validation_result["errors"].append(
            "Frame cache system not available - missing dependencies"
        )
        validation_result["recommendations"].append(
            "Install cache dependencies: pip install numpy>=1.24.0"
        )
    
    # Mode-specific validations
    if mode == "all" and memory_limit_mb < 1024:
        validation_result["warnings"].append(
            "Preload mode with low memory limit may fail for large videos"
        )
        validation_result["recommendations"].append(
            "Consider increasing memory limit or using 'lru' mode"
        )
    
    return validation_result

__all__.append("validate_cache_config")


def diagnose_cache_setup() -> Dict[str, Any]:
    """
    Comprehensive diagnostic information about cache system setup and capabilities.
    
    Returns:
        Dictionary containing diagnostic information for troubleshooting
        
    Example:
        >>> diagnostics = diagnose_cache_setup()
        >>> print(f"Cache available: {diagnostics['cache_available']}")
        >>> for recommendation in diagnostics['recommendations']:
        ...     print(f"Recommendation: {recommendation}")
    """
    diagnostics = {
        "cache_available": CACHE_AVAILABLE,
        "memory_monitoring_available": MEMORY_MONITORING_AVAILABLE,
        "logging_available": LOGGING_AVAILABLE,
        "hydra_available": HYDRA_AVAILABLE,
        "supported_modes": get_cache_modes(),
        "default_memory_limit_mb": 2048.0,
        "recommendations": []
    }
    
    # Generate recommendations based on setup
    if not CACHE_AVAILABLE:
        diagnostics["recommendations"].append(
            "Install cache dependencies: pip install numpy>=1.24.0"
        )
    
    if not MEMORY_MONITORING_AVAILABLE:
        diagnostics["recommendations"].append(
            "Install psutil for memory monitoring: pip install psutil>=5.9.0"
        )
    
    if not LOGGING_AVAILABLE:
        diagnostics["recommendations"].append(
            "Install enhanced logging for better diagnostics: pip install loguru>=0.7.0"
        )
    
    if not HYDRA_AVAILABLE:
        diagnostics["recommendations"].append(
            "Install Hydra for configuration management: pip install hydra-core>=1.3.0"
        )
    
    # System-specific recommendations
    if MEMORY_MONITORING_AVAILABLE and psutil:
        try:
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            
            if available_gb < 4:
                diagnostics["recommendations"].append(
                    f"Low system memory ({available_gb:.1f}GB available). Consider using mode='none' or lower memory limits"
                )
            elif available_gb > 16:
                diagnostics["recommendations"].append(
                    f"High system memory ({available_gb:.1f}GB available). Consider using mode='all' for maximum performance"
                )
        except Exception:
            pass  # Ignore psutil errors
    
    return diagnostics

__all__.append("diagnose_cache_setup")


def estimate_cache_memory_usage(
    video_frame_count: int,
    frame_width: int,
    frame_height: int,
    channels: int = 1,
    dtype_size: int = 1
) -> Dict[str, float]:
    """
    Estimate memory requirements for video caching.
    
    Calculates memory usage estimates for different cache modes to help
    with configuration planning and memory allocation decisions.
    
    Args:
        video_frame_count: Total number of frames in video
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        channels: Number of channels (1 for grayscale, 3 for RGB)
        dtype_size: Size of data type in bytes (1 for uint8, 4 for float32)
        
    Returns:
        Dictionary containing memory estimates for different scenarios
        
    Example:
        >>> # Estimate for 1000-frame video, 640x480 grayscale
        >>> estimates = estimate_cache_memory_usage(1000, 640, 480, 1, 1)
        >>> print(f"Full cache would need: {estimates['full_cache_mb']:.1f}MB")
        >>> print(f"50% cache would need: {estimates['partial_cache_50_mb']:.1f}MB")
    """
    frame_size_bytes = frame_width * frame_height * channels * dtype_size
    total_video_bytes = frame_size_bytes * video_frame_count
    
    estimates = {
        "frame_size_bytes": frame_size_bytes,
        "frame_size_kb": frame_size_bytes / 1024,
        "frame_size_mb": frame_size_bytes / (1024 * 1024),
        "total_video_bytes": total_video_bytes,
        "total_video_mb": total_video_bytes / (1024 * 1024),
        "full_cache_mb": total_video_bytes / (1024 * 1024),
        "partial_cache_50_mb": (total_video_bytes * 0.5) / (1024 * 1024),
        "partial_cache_25_mb": (total_video_bytes * 0.25) / (1024 * 1024),
        "partial_cache_10_mb": (total_video_bytes * 0.1) / (1024 * 1024),
        "recommended_memory_limit_mb": min(
            max(total_video_bytes / (1024 * 1024), 512),  # At least 512MB
            4096  # At most 4GB
        ),
        "video_info": {
            "frame_count": video_frame_count,
            "frame_dimensions": f"{frame_width}x{frame_height}",
            "channels": channels,
            "dtype_size_bytes": dtype_size
        }
    }
    
    # Add recommendations based on size
    if estimates["total_video_mb"] > 8192:  # >8GB
        estimates["recommendation"] = "Use mode='lru' with memory_limit_mb=2048 or lower"
    elif estimates["total_video_mb"] > 2048:  # >2GB
        estimates["recommendation"] = "Use mode='lru' with memory_limit_mb=4096"
    else:
        estimates["recommendation"] = "Use mode='all' for full preload if memory permits"
    
    return estimates

__all__.append("estimate_cache_memory_usage")


# Package-level configuration for improved usability
def _configure_logging():
    """Configure package-level logging for better debugging and monitoring."""
    if not LOGGING_AVAILABLE:
        return
    
    try:
        from loguru import logger as loguru_logger
        import sys
        
        # Add package-specific log formatting
        loguru_logger.configure(
            handlers=[
                {
                    "sink": sys.stderr,
                    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                             "<level>{level: <8}</level> | "
                             "<cyan>odor_plume_nav.cache</cyan> | "
                             "<level>{message}</level>",
                    "level": "INFO"
                }
            ]
        )
    except Exception:
        # Fallback to basic configuration
        pass


# Initialize package-level configuration
_configure_logging()

# Package metadata and version info
__version__ = "1.0.0"
__author__ = "Blitzy Engineering Team"

# Add diagnostic and utility functions to exports
__all__.extend([
    "validate_cache_config",
    "diagnose_cache_setup", 
    "estimate_cache_memory_usage",
    "get_cache_modes"
])

# Clean up None values from __all__ when dependencies are not available
__all__ = [item for item in __all__ if item is not None]

# Availability flags for external use
__all__.extend([
    "CACHE_AVAILABLE",
    "MEMORY_MONITORING_AVAILABLE", 
    "LOGGING_AVAILABLE",
    "HYDRA_AVAILABLE"
])

# Package-level configuration for improved usability
__doc_format__ = "restructuredtext"
__package_metadata__ = {
    "name": "odor_plume_nav.cache",
    "description": "High-performance frame caching system with dual-mode operation",
    "version": __version__,
    "author": __author__,
    "performance_targets": {
        "frame_retrieval_latency_ms": "<10",
        "cache_hit_rate_target": ">90%",
        "concurrent_agents_supported": "100+",
        "default_memory_limit_mb": 2048
    },
    "compatibility": {
        "numpy": ">=1.24.0",
        "psutil": ">=5.9.0 (optional)",
        "loguru": ">=0.7.0 (optional)",
        "hydra": ">=1.3.0 (optional)"
    },
    "cache_modes": {
        "none": "Direct I/O with no caching",
        "lru": "LRU eviction with intelligent memory management", 
        "all": "Full preload strategy for maximum throughput"
    },
    "integration_support": {
        "gymnasium_env": True,
        "video_plume": True,
        "hydra_configs": HYDRA_AVAILABLE,
        "structured_logging": LOGGING_AVAILABLE,
        "cli_integration": True
    }
}

# Package provides the following cache modes for configuration:
#
# Cache Modes:
#   none - Direct I/O with no caching for memory-constrained environments
#   lru  - LRU eviction with intelligent memory management for balanced performance
#   all  - Full preload strategy for maximum throughput scenarios
#
# Factory Functions:
#   create_frame_cache() - Primary factory with mode and memory configuration
#   create_frame_cache_from_config() - Hydra configuration integration
#   create_frame_cache_from_env() - Environment variable configuration
#   create_lru_cache() - LRU-specific factory
#   create_preload_cache() - Preload-specific factory
#   create_no_cache() - Direct I/O factory
#
# Diagnostic Functions:
#   diagnose_cache_setup() - Comprehensive system diagnostics
#   validate_cache_config() - Configuration validation
#   estimate_cache_memory_usage() - Memory requirement estimation
#
# For detailed usage examples and integration guidance, see the project documentation.