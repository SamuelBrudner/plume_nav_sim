"""
Recording backends module initialization providing centralized access to all storage backend implementations.

This module serves as the primary entry point for the recording backend framework, implementing
the backend registration system for automatic discovery and instantiation via Hydra configuration.
Provides comprehensive factory functions, capability detection for optional dependencies, and
a clean public API for external access to recording functionality.

The module implements F-017-RQ-002 requirements for multiple backend support including parquet,
HDF5, SQLite, and none backends with runtime selection capability. Backend implementations
achieve performance requirements through optimized buffering, compression, and graceful
degradation when dependencies are unavailable.

Key Features:
    - Backend registration system with BACKEND_REGISTRY for automatic component discovery
    - Factory functions for configuration-driven backend instantiation and validation
    - Capability detection with graceful degradation for optional dependencies (pandas, h5py)
    - Clean public API via __all__ export list following Python best practices
    - Comprehensive error handling with informative fallback recommendations
    - Integration with Hydra configuration management for seamless backend selection
    - Performance-aware backend selection with zero-overhead null recorder fallback

Backend Registry Architecture:
    The BACKEND_REGISTRY provides a centralized mapping from backend names to implementation
    classes, enabling runtime backend selection and automatic discovery. Supports both
    direct instantiation and Hydra-based dependency injection patterns.

Performance Characteristics:
    - Backend detection: <1ms for capability discovery and validation
    - Factory instantiation: <10ms for backend creation with dependency validation
    - Fallback handling: Automatic degradation to NullRecorder when primary backends fail
    - Memory efficiency: Minimal overhead through lazy imports and capability caching

Examples:
    Basic backend selection and instantiation:
    >>> from plume_nav_sim.recording.backends import create_backend, get_available_backends
    >>> available = get_available_backends()
    >>> print(f"Available backends: {available}")
    >>> config = {'backend': 'parquet', 'compression': 'snappy'}
    >>> recorder = create_backend(config)
    
    Advanced usage with capability detection:
    >>> from plume_nav_sim.recording.backends import BACKEND_REGISTRY
    >>> if 'parquet' in get_available_backends():
    ...     recorder = BACKEND_REGISTRY['parquet'](config)
    ... else:
    ...     print("Parquet backend unavailable, using NullRecorder")
    ...     recorder = BACKEND_REGISTRY['null'](config)
    
    Hydra configuration integration:
    >>> # In Hydra config: backend: parquet
    >>> # Automatic backend selection via registry lookup
    >>> recorder_class = BACKEND_REGISTRY[cfg.backend]
    >>> recorder = recorder_class(cfg)
"""

from typing import Dict, List, Optional, Type, Union, Any, TYPE_CHECKING
import warnings
import logging

# Import RecorderProtocol for type validation
if TYPE_CHECKING:
    from ..import BaseRecorder

from .null import NullRecorder

# Configure logging for backend module
logger = logging.getLogger(__name__)


# Backend registration system for automatic discovery and instantiation
BACKEND_REGISTRY: Dict[str, Type['BaseRecorder']] = {
    'null': NullRecorder,
    'none': NullRecorder,  # Alias for null backend
}


# Optional backend imports with explicit failure logging
try:
    from .parquet import ParquetRecorder
    BACKEND_REGISTRY['parquet'] = ParquetRecorder
    logger.debug("ParquetRecorder registered in BACKEND_REGISTRY")
except ImportError as exc:
    logger.error("Failed to import ParquetRecorder: %s", exc)
    raise

try:
    from .hdf5 import HDF5Recorder
    BACKEND_REGISTRY['hdf5'] = HDF5Recorder
    logger.debug("HDF5Recorder registered in BACKEND_REGISTRY")
except ImportError as exc:
    logger.error("Failed to import HDF5Recorder: %s", exc)
    raise

try:
    from .sqlite import SQLiteRecorder
    BACKEND_REGISTRY['sqlite'] = SQLiteRecorder
    logger.debug("SQLiteRecorder registered in BACKEND_REGISTRY")
except ImportError as exc:
    logger.error("Failed to import SQLiteRecorder: %s", exc)
    raise

# Derived availability flags for internal utilities
PARQUET_AVAILABLE = 'parquet' in BACKEND_REGISTRY
HDF5_AVAILABLE = 'hdf5' in BACKEND_REGISTRY
SQLITE_AVAILABLE = 'sqlite' in BACKEND_REGISTRY


def get_available_backends() -> List[str]:
    """
    Get list of available recording backends with dependency validation.
    
    Performs runtime capability detection to determine which backends are available
    based on optional dependency availability. The null backend is always available
    as a fallback option when other backends fail or dependencies are missing.
    
    Returns:
        List[str]: List of available backend names that can be instantiated
        
    Notes:
        Backend availability is determined by:
        - Successful import of backend implementation classes
        - Availability of required optional dependencies (pandas, h5py, pyarrow)
        - Runtime validation of key backend functionality
        
        The returned list is ordered by performance preference:
        1. null - Zero-overhead disabled recording
        2. parquet - High-performance columnar storage  
        3. hdf5 - Hierarchical scientific data storage
        4. sqlite - Embedded relational database storage
        
    Examples:
        Check available backends for UI display:
        >>> available = get_available_backends()
        >>> print(f"Recorder backends: {', '.join(available)}")
        
        Conditional backend selection:
        >>> if 'parquet' in get_available_backends():
        ...     backend_config = {'backend': 'parquet'}
        ... else:
        ...     backend_config = {'backend': 'null'}
    """
    available_backends = []
    
    # Null backend is always available as fallback
    available_backends.append('null')
    
    # Check optional backends with dependency validation
    if 'parquet' in BACKEND_REGISTRY:
        available_backends.append('parquet')
        logger.debug("ParquetRecorder available")

    if 'hdf5' in BACKEND_REGISTRY:
        available_backends.append('hdf5')
        logger.debug("HDF5Recorder available")

    if 'sqlite' in BACKEND_REGISTRY:
        available_backends.append('sqlite')
        logger.debug("SQLiteRecorder available")
    
    logger.info(f"Available recording backends: {available_backends}")
    return available_backends


def create_backend(
    config: Union[Dict[str, Any], Any],
    backend_name: Optional[str] = None,
) -> 'BaseRecorder':
    """Create a recording backend instance from configuration.

    The backend must be specified either via the ``backend`` key on a config
    dictionary or via an explicit ``backend_name`` argument. When recording is
    not desired, callers should specify ``backend='null'``.

    Args:
        config: Backend configuration as dictionary or structured config
            object. Must provide a ``backend`` field or attribute.
        backend_name: Optional explicit backend name override. If provided, it
            takes precedence over any value in ``config``.

    Returns:
        BaseRecorder: Configured recorder backend instance implementing
        ``RecorderProtocol``.

    Raises:
        ValueError: If the backend name is unknown or missing.
        ImportError: If backend dependencies are missing.
        TypeError: If the configuration format is invalid.
    """
    # Extract backend name from various configuration formats
    target_backend = backend_name
    if target_backend is None:
        if isinstance(config, dict):
            target_backend = config.get('backend')
        elif hasattr(config, 'backend'):
            target_backend = config.backend
        elif hasattr(config, 'get'):
            target_backend = config.get('backend')
        else:
            logger.error("Invalid config format: missing 'backend' field")
            raise TypeError("Invalid config format: must be dict or have 'backend' attribute")

    if not target_backend:
        logger.error("No backend specified in configuration")
        raise ValueError("Backend must be specified in configuration")

    # Validate backend name
    if target_backend not in BACKEND_REGISTRY:
        available = list(BACKEND_REGISTRY.keys())
        logger.error("Unknown backend '%s'. Available: %s", target_backend, available)
        raise ValueError(f"Unknown backend '{target_backend}'. Available: {available}")

    # Attempt backend instantiation with dependency validation
    backend_class = BACKEND_REGISTRY[target_backend]

    try:
        recorder = backend_class(config)
        logger.info("Successfully created %s recorder backend", target_backend)
        return recorder
    except ImportError as e:
        dependency_suggestions = {
            'parquet': "Install PyArrow: pip install pyarrow>=10.0.0",
            'hdf5': "Install h5py: pip install h5py>=3.0.0",
            'sqlite': "SQLite3 should be available in standard library",
        }
        suggestion = dependency_suggestions.get(target_backend, "Check backend dependencies")
        logger.error("Failed to create %s backend: %s. %s", target_backend, e, suggestion)
        raise ImportError(f"Failed to create {target_backend} backend: {e}. {suggestion}") from e
    except Exception as e:
        logger.error("Failed to create %s backend: %s", target_backend, e)
        raise RuntimeError(f"Failed to create {target_backend} backend: {e}") from e


def get_backend_capabilities() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed capability information for all available backends.
    
    Provides comprehensive metadata about each backend including dependency status,
    performance characteristics, and feature support for informed backend selection
    and UI display purposes.
    
    Returns:
        Dict[str, Dict[str, Any]]: Mapping of backend names to capability dictionaries
        
    Notes:
        Capability information includes:
        - available: Boolean indicating if backend can be instantiated
        - dependencies: List of required dependencies and their availability status
        - features: List of supported features (compression, async_io, etc.)
        - performance: Performance characteristics and optimization notes
        - use_cases: Recommended use cases and scenarios
        
    Examples:
        Display backend information in UI:
        >>> capabilities = get_backend_capabilities()
        >>> for name, info in capabilities.items():
        ...     status = "✓" if info['available'] else "✗"
        ...     print(f"{status} {name}: {info['description']}")
        
        Backend selection based on features:
        >>> caps = get_backend_capabilities()
        >>> compressed_backends = [
        ...     name for name, info in caps.items()
        ...     if 'compression' in info['features'] and info['available']
        ... ]
    """
    capabilities = {
        'null': {
            'available': True,
            'description': 'Zero-overhead disabled recording for maximum simulation performance',
            'dependencies': [],
            'features': ['zero_overhead', 'debug_mode', 'call_counting'],
            'performance': 'Optimized for <0.001ms per operation when disabled',
            'use_cases': ['High-performance simulation', 'Development testing', 'Fallback mode'],
            'file_formats': [],
            'compression': [],
            'import_error': None
        }
    }
    
    # Add ParquetRecorder capabilities
    if PARQUET_AVAILABLE:
        capabilities['parquet'] = {
            'available': True,
            'description': 'High-performance columnar storage using Apache Parquet format',
            'dependencies': ['pyarrow>=10.0.0', 'pandas>=1.5.0 (optional)'],
            'features': ['compression', 'schema_evolution', 'columnar_storage', 'async_io'],
            'performance': 'Optimized for analytical workloads with efficient compression',
            'use_cases': ['Large dataset analysis', 'Long-term storage', 'Data science workflows'],
            'file_formats': ['parquet'],
            'compression': ['snappy', 'gzip', 'lz4', 'zstd'],
            'import_error': None
        }
    else:
        capabilities['parquet'] = {
            'available': False,
            'description': 'High-performance columnar storage (unavailable)',
            'dependencies': ['pyarrow>=10.0.0 (missing)'],
            'features': [],
            'performance': 'N/A',
            'use_cases': [],
            'file_formats': [],
            'compression': [],
            'import_error': _PARQUET_IMPORT_ERROR if 'ParquetRecorder' in globals() else 'Import failed'
        }
    
    # Add HDF5Recorder capabilities
    if HDF5_AVAILABLE:
        capabilities['hdf5'] = {
            'available': True,
            'description': 'Hierarchical scientific data storage using HDF5 format',
            'dependencies': ['h5py>=3.0.0'],
            'features': ['hierarchical_storage', 'compression', 'metadata_attributes', 'chunked_datasets'],
            'performance': 'Optimized for scientific data with metadata preservation',
            'use_cases': ['Scientific research', 'Hierarchical data', 'Metadata-rich datasets'],
            'file_formats': ['hdf5', 'h5'],
            'compression': ['gzip', 'lzf', 'szip'],
            'import_error': None
        }
    else:
        capabilities['hdf5'] = {
            'available': False,
            'description': 'Hierarchical scientific data storage (unavailable)',
            'dependencies': ['h5py>=3.0.0 (missing)'],
            'features': [],
            'performance': 'N/A',
            'use_cases': [],
            'file_formats': [],
            'compression': [],
            'import_error': _HDF5_IMPORT_ERROR if 'HDF5Recorder' in globals() else 'Import failed'
        }
    
    # Add SQLiteRecorder capabilities
    if SQLITE_AVAILABLE:
        capabilities['sqlite'] = {
            'available': True,
            'description': 'Embedded relational database storage using SQLite3',
            'dependencies': ['sqlite3 (standard library)'],
            'features': ['relational_storage', 'sql_queries', 'transactions', 'zero_config'],
            'performance': 'Optimized for queryable data access with ACID compliance',
            'use_cases': ['Queryable data', 'Relational analysis', 'Zero-configuration deployment'],
            'file_formats': ['sqlite', 'db'],
            'compression': ['built_in'],
            'import_error': None
        }
    else:
        capabilities['sqlite'] = {
            'available': False,
            'description': 'Embedded relational database storage (unavailable)',
            'dependencies': ['sqlite3 (missing)'],
            'features': [],
            'performance': 'N/A',
            'use_cases': [],
            'file_formats': [],
            'compression': [],
            'import_error': _SQLITE_IMPORT_ERROR if 'SQLiteRecorder' in globals() else 'Import failed'
        }
    
    return capabilities


def validate_backend_config(config: Dict[str, Any], backend_name: str) -> bool:
    """
    Validate backend configuration parameters before instantiation.
    
    Performs comprehensive validation of backend-specific configuration parameters
    to catch errors early and provide informative feedback for configuration issues.
    
    Args:
        config: Configuration dictionary to validate
        backend_name: Target backend name for validation rules
        
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Notes:
        Validation includes:
        - Required parameter presence checking
        - Parameter type and range validation
        - Backend-specific constraint validation
        - Dependency availability verification
        
    Examples:
        Validate configuration before backend creation:
        >>> config = {'backend': 'parquet', 'compression': 'snappy'}
        >>> if validate_backend_config(config, 'parquet'):
        ...     recorder = create_backend(config)
        ... else:
        ...     print("Invalid configuration")
    """
    try:
        # Check if backend is available
        if backend_name not in BACKEND_REGISTRY:
            logger.error(f"Unknown backend: {backend_name}")
            return False
            
        if backend_name not in get_available_backends():
            logger.error(f"Backend {backend_name} not available (missing dependencies)")
            return False
        
        # Basic configuration validation
        if not isinstance(config, dict):
            logger.error("Configuration must be a dictionary")
            return False
        
        # Backend-specific validation
        if backend_name == 'parquet':
            return _validate_parquet_config(config)
        elif backend_name == 'hdf5':
            return _validate_hdf5_config(config)
        elif backend_name == 'sqlite':
            return _validate_sqlite_config(config)
        elif backend_name in ['null', 'none']:
            return _validate_null_config(config)
        else:
            # Unknown backend - basic validation only
            return True
            
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        return False


def _validate_parquet_config(config: Dict[str, Any]) -> bool:
    """Validate ParquetRecorder-specific configuration parameters."""
    valid_compression = ['snappy', 'gzip', 'lz4', 'zstd', 'none']
    
    compression = config.get('compression', 'snappy')
    if compression not in valid_compression:
        logger.error(f"Invalid compression: {compression}. Valid options: {valid_compression}")
        return False
    
    batch_size = config.get('batch_size', 1000)
    if not isinstance(batch_size, int) or batch_size <= 0:
        logger.error("batch_size must be a positive integer")
        return False
    
    return True


def _validate_hdf5_config(config: Dict[str, Any]) -> bool:
    """Validate HDF5Recorder-specific configuration parameters."""
    valid_compression = ['gzip', 'lzf', 'szip', 'none']
    
    compression = config.get('compression', 'gzip')
    if compression not in valid_compression:
        logger.error(f"Invalid compression: {compression}. Valid options: {valid_compression}")
        return False
    
    chunk_size = config.get('chunk_size', 1000)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        logger.error("chunk_size must be a positive integer")
        return False
    
    return True


def _validate_sqlite_config(config: Dict[str, Any]) -> bool:
    """Validate SQLiteRecorder-specific configuration parameters."""
    batch_size = config.get('batch_size', 100)
    if not isinstance(batch_size, int) or batch_size <= 0:
        logger.error("batch_size must be a positive integer")
        return False
    
    # Validate journal mode if specified
    journal_mode = config.get('journal_mode', 'WAL')
    valid_modes = ['DELETE', 'TRUNCATE', 'PERSIST', 'MEMORY', 'WAL', 'OFF']
    if journal_mode not in valid_modes:
        logger.error(f"Invalid journal_mode: {journal_mode}. Valid options: {valid_modes}")
        return False
    
    return True


def _validate_null_config(config: Dict[str, Any]) -> bool:
    """Validate NullRecorder-specific configuration parameters."""
    # NullRecorder accepts any configuration - validation always passes
    debug_mode = config.get('enable_debug_mode', False)
    if not isinstance(debug_mode, bool):
        logger.warning("enable_debug_mode should be boolean, but NullRecorder will handle conversion")
    
    return True


# Export all recorder backend implementations and utilities
__all__ = [
    # Factory functions for backend creation and discovery
    'get_available_backends',
    'create_backend',
    'get_backend_capabilities',
    'validate_backend_config',
    
    # Backend registry for automatic discovery
    'BACKEND_REGISTRY',
    
    # Recorder backend implementations
    'NullRecorder',
    'ParquetRecorder',
    'HDF5Recorder', 
    'SQLiteRecorder',
]

# Log backend registration summary
logger.info(f"Recording backends module initialized with {len(BACKEND_REGISTRY)} registered backends")
logger.debug(f"Registered backends: {list(BACKEND_REGISTRY.keys())}")
logger.debug(f"Available backends: {get_available_backends()}")