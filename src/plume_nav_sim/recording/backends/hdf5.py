"""
HDF5Recorder implementation providing hierarchical scientific data storage using HDF5 format via h5py.

This module implements the HDF5Recorder class extending BaseRecorder with hierarchical scientific 
data storage capabilities optimized for research workflows. Features efficient compression, 
chunked datasets, and comprehensive metadata attribution for experimental reproducibility and 
long-term scientific data management.

Key Features:
    - Hierarchical data organization with groups for runs/episodes matching scientific standards
    - Buffered dataset writing with chunk-based I/O to maintain ≤33ms step latency requirements
    - h5py integration with automatic dataset creation, compression, and metadata attribution
    - Compression support (gzip, lzf, szip) with configurable chunk sizes for time-series data
    - Structured group hierarchy (/run_id/episode_id/datasets) for scalable multi-experiment organization
    - Attribute metadata preservation with HDF5 attributes for experimental parameters

Performance Characteristics:
    - F-017-RQ-001: <1ms overhead per 1000 steps when disabled for minimal simulation impact
    - F-017-RQ-002: Multiple backend support with HDF5 hierarchical scientific data storage
    - F-017-RQ-003: Buffered asynchronous I/O for non-blocking data persistence
    - Section 5.2.8: H5py ≥3.0.0 integration with compression and multi-threaded I/O

Scientific Data Organization:
    - Self-describing format with embedded documentation and metadata
    - Cross-platform compatibility for data sharing and long-term preservation
    - Chunked datasets optimized for time-series trajectory data access patterns
    - Scalable architecture supporting large-scale multi-experiment datasets

Examples:
    Basic HDF5 recorder usage:
    >>> config = HDF5Config(
    ...     file_path="./data/experiment.h5",
    ...     compression="gzip",
    ...     compression_opts=6,
    ...     chunk_size=1000
    ... )
    >>> recorder = HDF5Recorder(config)
    >>> with recorder.recording_session(episode_id=1):
    ...     recorder.record_step({'position': [0, 0], 'concentration': 0.5}, step_number=0)
    
    Scientific metadata integration:
    >>> recorder.set_attributes('/experiment_1/episode_001', {
    ...     'experiment_date': '2024-01-15',
    ...     'researcher': 'Dr. Smith',
    ...     'environmental_conditions': 'controlled_lab',
    ...     'equipment_calibration': '2024-01-10'
    ... })
    
    Multi-experiment hierarchical organization:
    >>> # Data organized as: /run_id/episode_id/step_data
    >>> # Each level contains comprehensive metadata attributes
    >>> # Enables efficient querying and cross-experiment analysis
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, Any, Optional, Union, List, Tuple, TYPE_CHECKING

import numpy as np
import h5py  # type: ignore

# Import BaseRecorder from recording framework
from .. import BaseRecorder, RecorderConfig

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class HDF5Config:
    """
    Configuration dataclass for HDF5Recorder with scientific data management parameters.
    
    This dataclass provides comprehensive configuration for HDF5-based hierarchical data storage
    supporting scientific research workflows, performance optimization, and experimental 
    reproducibility requirements.
    
    File Organization Configuration:
        file_path: HDF5 file path for hierarchical data storage (default: "./data/recording.h5")
        create_intermediate_groups: Automatically create parent groups (default: True)
        track_order: Maintain creation order for groups and datasets (default: True)
        
    Compression Configuration:
        compression: Compression algorithm ('gzip', 'lzf', 'szip', None) (default: 'gzip')
        compression_opts: Algorithm-specific compression options (default: 6)
        shuffle: Enable byte-shuffle filter for better compression (default: True)
        fletcher32: Enable Fletcher32 checksum for data integrity (default: True)
        
    Performance Configuration:
        chunk_size: Dataset chunk size for optimal I/O performance (default: 1000)
        buffer_size: Memory buffer size for batched writes (default: 10000)
        
    Scientific Metadata Configuration:
        store_metadata: Include comprehensive experimental metadata (default: True)
        metadata_format: Format for metadata storage ('attributes', 'datasets') (default: 'attributes')
        
    Advanced Configuration:
        swmr_mode: Single Writer Multiple Reader mode for concurrent access (default: False)
        libver: HDF5 library version compatibility ('earliest', 'latest') (default: 'latest')
    """
    # File organization
    file_path: str = "./data/recording.h5"
    create_intermediate_groups: bool = True
    track_order: bool = True
    
    # Compression settings
    compression: Optional[str] = 'gzip'
    compression_opts: Optional[int] = 6
    shuffle: bool = True
    fletcher32: bool = True
    
    # Performance settings
    chunk_size: int = 1000
    buffer_size: int = 10000
    
    # Scientific metadata
    store_metadata: bool = True
    metadata_format: str = 'attributes'  # 'attributes' or 'datasets'
    
    # Advanced HDF5 settings
    swmr_mode: bool = False
    libver: str = 'latest'
    
    def __post_init__(self):
        """Validate HDF5 configuration parameters after initialization."""
        if self.compression and self.compression not in ['gzip', 'lzf', 'szip']:
            raise ValueError("compression must be one of: gzip, lzf, szip, or None")
        
        if self.compression_opts is not None and not isinstance(self.compression_opts, int):
            raise ValueError("compression_opts must be an integer or None")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        
        if self.metadata_format not in ['attributes', 'datasets']:
            raise ValueError("metadata_format must be 'attributes' or 'datasets'")
        
        if self.libver not in ['earliest', 'latest']:
            raise ValueError("libver must be 'earliest' or 'latest'")
        
        # Ensure file path has .h5 extension
        if not self.file_path.endswith(('.h5', '.hdf5')):
            self.file_path += '.h5'


class HDF5Recorder(BaseRecorder):
    """
    HDF5Recorder implementation providing hierarchical scientific data storage via h5py.
    
    The HDF5Recorder extends BaseRecorder to provide comprehensive hierarchical data storage
    capabilities optimized for scientific research workflows. Features efficient compression,
    chunked datasets, structured group organization, and comprehensive metadata attribution
    for experimental reproducibility and long-term scientific data management.
    
    Key Features:
    - Hierarchical data organization: /run_id/episode_id/step_data structure
    - Chunked dataset writing optimized for time-series trajectory data
    - Comprehensive compression support with configurable algorithms and options
    - Scientific metadata preservation with HDF5 attributes and embedded documentation
    - Buffered asynchronous I/O coordination for minimal simulation performance impact
    - Graceful fallback handling when h5py dependencies are unavailable
    - Cross-platform compatibility and long-term data preservation capabilities
    
    Performance Characteristics:
    - F-017-RQ-001: <1ms overhead when disabled per 1000 steps requirement
    - F-017-RQ-003: Buffered asynchronous I/O for non-blocking data persistence
    - Section 5.2.8: Maintains ≤33ms step latency with 100 agents through chunked I/O
    - Efficient memory management with configurable buffer sizes and compression
    
    Scientific Data Standards:
    - Self-describing format with comprehensive metadata attribution
    - Structured group hierarchy enabling efficient multi-experiment analysis
    - Cross-experiment compatibility through standardized data organization
    - Long-term preservation with embedded documentation and configuration snapshots
    
    Thread Safety:
    - All public methods are thread-safe with proper HDF5 file locking
    - Background I/O thread handles asynchronous dataset writes safely
    - Resource cleanup ensures proper file closure without data loss
    - Concurrent access support through optional SWMR mode configuration
    
    Examples:
        Basic scientific recording workflow:
        >>> config = HDF5Config(
        ...     file_path="./experiments/plume_navigation.h5",
        ...     compression="gzip",
        ...     compression_opts=6,
        ...     chunk_size=1000
        ... )
        >>> recorder = HDF5Recorder(config)
        >>> 
        >>> with recorder.recording_session(episode_id=1):
        ...     recorder.record_step({
        ...         'agent_position': [10.5, 20.3],
        ...         'odor_concentration': 0.75,
        ...         'wind_velocity': [1.2, 0.8]
        ...     }, step_number=0)
        ...     
        ...     recorder.record_episode({
        ...         'total_steps': 1000,
        ...         'success': True,
        ...         'path_efficiency': 0.82
        ...     }, episode_id=1)
        
        Multi-experiment hierarchical organization:
        >>> # Automatic creation of structured hierarchy:
        >>> # /run_20240115_143022/episode_001/step_data
        >>> # /run_20240115_143022/episode_001/episode_summary
        >>> # Each level contains comprehensive metadata attributes
        
        Scientific metadata integration:
        >>> recorder.set_attributes('/run_20240115_143022', {
        ...     'experiment_title': 'Turbulent Plume Navigation Study',
        ...     'researcher': 'Dr. Jane Smith',
        ...     'institution': 'University Research Lab',
        ...     'experiment_date': '2024-01-15T14:30:22Z',
        ...     'environmental_conditions': {
        ...         'temperature': 22.5,
        ...         'humidity': 45,
        ...         'wind_speed': 2.1
        ...     }
        ... })
    """
    
    def __init__(self, config: Union[HDF5Config, RecorderConfig]):
        """
        Initialize HDF5Recorder with configuration and hierarchical data infrastructure.
        
        Args:
            config: HDF5Recorder configuration with compression and performance settings.
                   Can be HDF5Config or RecorderConfig with HDF5-specific parameters.
        
        Raises:
            ImportError: If h5py is not available and graceful fallback is disabled
            ValueError: If configuration parameters are invalid or incompatible
            OSError: If file system permissions prevent HDF5 file creation
        """
        # Handle configuration conversion
        if isinstance(config, HDF5Config):
            # Convert HDF5Config to RecorderConfig for BaseRecorder
            base_config = RecorderConfig(
                backend='hdf5',
                output_dir=str(Path(config.file_path).parent),
                buffer_size=config.buffer_size,
                compression=config.compression if config.compression else 'none'
            )
            super().__init__(base_config)
            self.hdf5_config = config
        else:
            # Use provided RecorderConfig
            super().__init__(config)
            # Create default HDF5Config from RecorderConfig
            self.hdf5_config = HDF5Config(
                file_path=str(Path(config.output_dir) / f"{self.run_id}.h5"),
                compression=config.compression if config.compression != 'none' else None,
                buffer_size=config.buffer_size
            )
        
        # HDF5-specific state
        self._h5_file: Optional["h5py.File"] = None
        self._file_lock = threading.RLock()
        self._dataset_cache: Dict[str, "h5py.Dataset"] = {}
        self._group_cache: Dict[str, "h5py.Group"] = {}
        
        # Performance monitoring
        self._write_times: List[float] = []
        self._compression_ratios: List[float] = []
        
        # Initialize HDF5 file structure
        self._initialize_hdf5_file()

        logger.info(f"HDF5Recorder initialized with file: {self.hdf5_config.file_path}")
    
    def _initialize_hdf5_file(self) -> None:
        """Initialize HDF5 file with proper configuration and metadata."""
        try:
            # Ensure parent directory exists
            file_path = Path(self.hdf5_config.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open HDF5 file with configuration
            with self._file_lock:
                self._h5_file = h5py.File(
                    str(file_path),
                    mode='a',  # Create if doesn't exist, append if exists
                    libver=self.hdf5_config.libver,
                    swmr=self.hdf5_config.swmr_mode,
                    track_order=self.hdf5_config.track_order
                )
                
                # Set file-level metadata
                if self.hdf5_config.store_metadata:
                    self._set_file_metadata()
                
                logger.debug(f"HDF5 file initialized: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize HDF5 file: {e}")
            raise
    
    def _set_file_metadata(self) -> None:
        """Set comprehensive file-level metadata for scientific reproducibility."""
        if not self._h5_file:
            raise RuntimeError("HDF5 file is not initialized")
        
        metadata = {
            'created_timestamp': time.time(),
            'created_date': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'recorder_type': 'HDF5Recorder',
            'recorder_version': '1.0.0',
            'plume_nav_sim_version': '1.0.0',
            'h5py_version': h5py.__version__ if h5py else 'unavailable',
            'numpy_version': np.__version__,
            'data_format': 'hierarchical_scientific',
            'compression_algorithm': self.hdf5_config.compression,
            'compression_level': self.hdf5_config.compression_opts,
            'chunk_size': self.hdf5_config.chunk_size,
            'buffer_size': self.hdf5_config.buffer_size,
            'run_id': self.run_id
        }
        
        # Set attributes
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                self._h5_file.attrs[key] = value
            else:
                # JSON encode complex types
                self._h5_file.attrs[key] = json.dumps(value)
    
    def create_group(self, group_path: str, **metadata: Any) -> Optional["h5py.Group"]:
        """
        Create HDF5 group with optional metadata attributes.
        
        Args:
            group_path: HDF5 group path (e.g., '/run_id/episode_001')
            **metadata: Metadata attributes to attach to the group
            
        Returns:
            Optional["h5py.Group"]: Created group
        """
        if not self._h5_file:
            raise RuntimeError("HDF5 file is not initialized")
        
        try:
            with self._file_lock:
                # Create group with intermediate groups if needed
                if self.hdf5_config.create_intermediate_groups:
                    group = self._h5_file.require_group(group_path)
                else:
                    group = self._h5_file.create_group(group_path)
                
                # Set metadata attributes
                if metadata and self.hdf5_config.store_metadata:
                    self.set_attributes(group_path, metadata)
                
                # Cache group for performance
                self._group_cache[group_path] = group
                
                logger.debug(f"Created HDF5 group: {group_path}")
                return group
                
        except Exception as e:
            logger.error(f"Failed to create HDF5 group {group_path}: {e}")
            return None
    
    def create_dataset(
        self, 
        dataset_path: str, 
        data: np.ndarray,
        **metadata: Any
    ) -> Optional["h5py.Dataset"]:
        """
        Create HDF5 dataset with chunking, compression, and metadata.
        
        Args:
            dataset_path: HDF5 dataset path (e.g., '/run_id/episode_001/step_data')
            data: Initial data for the dataset
            **metadata: Metadata attributes to attach to the dataset
            
        Returns:
            Optional["h5py.Dataset"]: Created dataset
        """
        if not self._h5_file:
            raise RuntimeError("HDF5 file is not initialized")
        
        try:
            with self._file_lock:
                # Determine dataset parameters
                dtype = data.dtype if hasattr(data, 'dtype') else np.float32
                shape = data.shape if hasattr(data, 'shape') else (len(data),)
                
                # Configure chunking for time-series data
                chunks = self._calculate_optimal_chunks(shape)
                
                # Create dataset with compression and chunking
                compression_kwargs = {}
                if self.hdf5_config.compression:
                    compression_kwargs['compression'] = self.hdf5_config.compression
                    # Only add compression_opts for algorithms that support it
                    if self.hdf5_config.compression == 'gzip' and self.hdf5_config.compression_opts is not None:
                        compression_kwargs['compression_opts'] = self.hdf5_config.compression_opts
                
                dataset = self._h5_file.create_dataset(
                    dataset_path,
                    data=data,
                    dtype=dtype,
                    chunks=chunks,
                    shuffle=self.hdf5_config.shuffle,
                    fletcher32=self.hdf5_config.fletcher32,
                    maxshape=(None,) + shape[1:] if len(shape) > 1 else (None,),  # Unlimited first dimension
                    **compression_kwargs
                )
                
                # Set metadata attributes
                if metadata and self.hdf5_config.store_metadata:
                    self.set_attributes(dataset_path, metadata)
                
                # Cache dataset for performance
                self._dataset_cache[dataset_path] = dataset
                
                logger.debug(f"Created HDF5 dataset: {dataset_path} with shape {shape}")
                return dataset
                
        except Exception as e:
            logger.error(f"Failed to create HDF5 dataset {dataset_path}: {e}")
            return None
    
    def set_attributes(self, path: str, attributes: Dict[str, Any]) -> None:
        """
        Set HDF5 attributes for scientific metadata preservation.
        
        Args:
            path: HDF5 path (group or dataset) to attach attributes
            attributes: Dictionary of attributes to set
        """
        if not self._h5_file:
            raise RuntimeError("HDF5 file is not initialized")
        
        try:
            with self._file_lock:
                obj = self._h5_file[path]
                
                for key, value in attributes.items():
                    if value is None:
                        continue
                        
                    # Handle different value types
                    if isinstance(value, (str, int, float, bool)):
                        obj.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        obj.attrs[key] = value
                    elif isinstance(value, (list, tuple, dict)):
                        # JSON encode complex types
                        obj.attrs[key] = json.dumps(value)
                    else:
                        # Convert to string representation
                        obj.attrs[key] = str(value)
                
                logger.debug(f"Set {len(attributes)} attributes for {path}")
                
        except Exception as e:
            logger.error(f"Failed to set attributes for {path}: {e}")
    
    def flush_buffer(self) -> None:
        """Force immediate flush of HDF5 file buffers to disk."""
        if not self._h5_file:
            raise RuntimeError("HDF5 file is not initialized")
        
        try:
            with self._file_lock:
                self._h5_file.flush()
                logger.debug("HDF5 file buffers flushed to disk")
                
        except Exception as e:
            logger.error(f"Failed to flush HDF5 buffers: {e}")
    
    def close_file(self) -> None:
        """Close HDF5 file with proper resource cleanup."""
        if not self._h5_file:
            raise RuntimeError("HDF5 file is not initialized")
        
        try:
            with self._file_lock:
                # Clear caches
                self._dataset_cache.clear()
                self._group_cache.clear()
                
                # Close HDF5 file
                if self._h5_file:
                    self._h5_file.close()
                    self._h5_file = None
                
                logger.info(f"HDF5 file closed: {self.hdf5_config.file_path}")
                
        except Exception as e:
            logger.error(f"Failed to close HDF5 file: {e}")
    
    def _calculate_optimal_chunks(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate optimal chunk sizes for time-series trajectory data."""
        if len(shape) == 1:
            # 1D time series
            return (min(self.hdf5_config.chunk_size, shape[0]),)
        elif len(shape) == 2:
            # 2D data (steps, features)
            chunk_steps = min(self.hdf5_config.chunk_size, shape[0])
            return (chunk_steps, shape[1])
        else:
            # Higher dimensional data
            chunk_steps = min(self.hdf5_config.chunk_size, shape[0])
            return (chunk_steps,) + shape[1:]
    
    def _write_step_data(self, data: List[Dict[str, Any]]) -> int:
        """
        Write step-level data using HDF5 hierarchical storage with chunking.
        
        Args:
            data: List of step data dictionaries from buffer
            
        Returns:
            int: Number of bytes written (estimated)
        """
        if not data or not self._h5_file:
            return 0
        
        start_time = time.perf_counter()
        bytes_written = 0
        
        try:
            with self._file_lock:
                # Group data by episode
                episode_data = {}
                for step in data:
                    episode_id = step.get('episode_id', self.current_episode_id)
                    if episode_id not in episode_data:
                        episode_data[episode_id] = []
                    episode_data[episode_id].append(step)
                
                # Write data for each episode
                for episode_id, episode_steps in episode_data.items():
                    episode_path = f"/{self.run_id}/episode_{episode_id:06d}"
                    
                    # Create episode group if needed
                    if episode_path not in self._group_cache:
                        self.create_group(episode_path, episode_id=episode_id)
                    
                    # Convert step data to structured arrays
                    step_arrays = self._convert_steps_to_arrays(episode_steps)
                    
                    # Write each data type as separate dataset
                    for data_type, array_data in step_arrays.items():
                        dataset_path = f"{episode_path}/step_{data_type}"
                        
                        if dataset_path in self._dataset_cache:
                            # Append to existing dataset
                            dataset = self._dataset_cache[dataset_path]
                            current_size = dataset.shape[0]
                            new_size = current_size + len(array_data)
                            dataset.resize((new_size,) + dataset.shape[1:])
                            dataset[current_size:new_size] = array_data
                        else:
                            # Create new dataset
                            dataset = self.create_dataset(
                                dataset_path,
                                array_data,
                                data_type=data_type,
                                episode_id=episode_id
                            )
                        
                        bytes_written += array_data.nbytes
                
                # Force flush for immediate persistence
                if self.hdf5_config.buffer_size < 1000:  # Small buffer = immediate flush
                    self.flush_buffer()
                
        except Exception as e:
            logger.error(f"Error writing step data to HDF5: {e}")
            self._metrics['io_errors'] += 1
            
        finally:
            # Performance tracking
            if self.config.enable_metrics:
                write_time = time.perf_counter() - start_time
                self._write_times.append(write_time)
                
                # Keep only recent timings
                if len(self._write_times) > 1000:
                    self._write_times = self._write_times[-1000:]
        
        return bytes_written
    
    def _write_episode_data(self, data: List[Dict[str, Any]]) -> int:
        """
        Write episode-level data using HDF5 hierarchical storage.
        
        Args:
            data: List of episode data dictionaries from buffer
            
        Returns:
            int: Number of bytes written (estimated)
        """
        if not data or not self._h5_file:
            return 0
        
        bytes_written = 0
        
        try:
            with self._file_lock:
                for episode in data:
                    episode_id = episode.get('episode_id', self.current_episode_id)
                    episode_path = f"/{self.run_id}/episode_{episode_id:06d}"
                    
                    # Create episode group if needed
                    if episode_path not in self._group_cache:
                        self.create_group(episode_path, episode_id=episode_id)
                    
                    # Store episode summary data
                    summary_path = f"{episode_path}/episode_summary"
                    
                    # Convert episode data to arrays
                    episode_arrays = self._convert_episode_to_arrays(episode)
                    
                    for data_type, array_data in episode_arrays.items():
                        dataset_path = f"{summary_path}_{data_type}"
                        
                        # Create dataset for episode summary
                        dataset = self.create_dataset(
                            dataset_path,
                            array_data,
                            data_type=data_type,
                            episode_id=episode_id,
                            data_category='episode_summary'
                        )
                        
                        if dataset:
                            bytes_written += array_data.nbytes
                
        except Exception as e:
            logger.error(f"Error writing episode data to HDF5: {e}")
            self._metrics['io_errors'] += 1
        
        return bytes_written
    
    def _export_data_backend(
        self, 
        output_path: str,
        format: str,
        compression: Optional[str] = None,
        filter_episodes: Optional[List[int]] = None,
        **export_options: Any
    ) -> bool:
        """
        Export HDF5 data with format conversion and filtering options.
        
        Args:
            output_path: Target file path for exported data
            format: Export format ('hdf5', 'parquet', 'csv', 'json')
            compression: Optional compression for export format
            filter_episodes: Optional list of episode IDs to export
            **export_options: Additional format-specific export parameters
            
        Returns:
            bool: True if export completed successfully, False otherwise
        """
        if not self._h5_file:
            logger.error("No HDF5 file open for export")
            return False
        
        try:
            with self._file_lock:
                if format.lower() == 'hdf5':
                    # Copy HDF5 file with optional filtering
                    return self._export_hdf5(output_path, filter_episodes, **export_options)
                elif format.lower() == 'parquet':
                    # Convert to Parquet format
                    return self._export_parquet(output_path, compression, filter_episodes, **export_options)
                elif format.lower() == 'csv':
                    # Convert to CSV format
                    return self._export_csv(output_path, filter_episodes, **export_options)
                elif format.lower() == 'json':
                    # Convert to JSON format
                    return self._export_json(output_path, filter_episodes, **export_options)
                else:
                    logger.error(f"Unsupported export format: {format}")
                    return False
                    
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def _convert_steps_to_arrays(self, steps: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Convert step data to structured numpy arrays for HDF5 storage."""
        if not steps:
            return {}
        
        # Group data by type
        data_types = {}
        for step in steps:
            for key, value in step.items():
                if key not in data_types:
                    data_types[key] = []
                data_types[key].append(value)
        
        # Convert to numpy arrays
        arrays = {}
        for key, values in data_types.items():
            try:
                # Convert to appropriate numpy array
                array = np.array(values)
                if array.dtype.kind in 'UO':  # Unicode or object
                    # Convert strings/objects to fixed-length strings with HDF5-compatible encoding
                    if all(isinstance(v, str) for v in values):
                        max_len = max(len(str(v).encode('utf-8')) for v in values) if values else 1
                        # Use S (bytes) dtype instead of U (unicode) for better HDF5 compatibility
                        byte_values = [str(v).encode('utf-8') for v in values]
                        array = np.array(byte_values, dtype=f'S{max_len}')
                    else:
                        # Convert to JSON strings for complex objects
                        json_values = [json.dumps(v) if not isinstance(v, str) else str(v) for v in values]
                        max_len = max(len(s.encode('utf-8')) for s in json_values) if json_values else 1
                        byte_values = [s.encode('utf-8') for s in json_values]
                        array = np.array(byte_values, dtype=f'S{max_len}')
                
                arrays[key] = array
                
            except Exception as e:
                logger.warning(f"Failed to convert {key} to array: {e}")
                # Skip problematic fields to avoid blocking the entire write
                continue
        
        return arrays
    
    def _convert_episode_to_arrays(self, episode: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert episode data to structured numpy arrays for HDF5 storage."""
        arrays = {}
        
        for key, value in episode.items():
            try:
                if isinstance(value, (list, tuple)):
                    array = np.array(value)
                    # Handle string arrays in lists
                    if array.dtype.kind in 'UO':
                        # Convert to bytes for HDF5 compatibility
                        if array.size > 0:
                            str_values = [str(v) for v in value]
                            max_len = max(len(s.encode('utf-8')) for s in str_values) if str_values else 1
                            byte_values = [s.encode('utf-8') for s in str_values]
                            array = np.array(byte_values, dtype=f'S{max_len}')
                elif isinstance(value, (int, float, bool)):
                    array = np.array([value])
                elif isinstance(value, str):
                    # Use bytes encoding for HDF5 compatibility
                    byte_value = value.encode('utf-8')
                    array = np.array([byte_value], dtype=f'S{len(byte_value)}')
                else:
                    # Convert complex objects to JSON strings
                    json_str = json.dumps(value)
                    byte_value = json_str.encode('utf-8')
                    array = np.array([byte_value], dtype=f'S{len(byte_value)}')
                
                arrays[key] = array
                
            except Exception as e:
                logger.warning(f"Failed to convert episode {key} to array: {e}")
                continue
        
        return arrays
    
    def _export_hdf5(self, output_path: str, filter_episodes: Optional[List[int]] = None, **options: Any) -> bool:
        """Export to HDF5 format with optional episode filtering."""
        try:
            # For HDF5 to HDF5, copy file with optional filtering
            import shutil
            
            if filter_episodes is None:
                # Simple file copy
                shutil.copy2(self.hdf5_config.file_path, output_path)
                logger.info(f"HDF5 file copied to {output_path}")
                return True
            else:
                # Filtered copy - would need to implement selective copying
                logger.warning("Episode filtering for HDF5 export not yet implemented")
                return False
                
        except Exception as e:
            logger.error(f"HDF5 export failed: {e}")
            return False
    
    def _export_parquet(self, output_path: str, compression: Optional[str] = None, filter_episodes: Optional[List[int]] = None, **options: Any) -> bool:
        """Export to Parquet format with compression."""
        try:
            # Would need pandas/pyarrow for Parquet export
            logger.warning("Parquet export requires pandas/pyarrow - not yet implemented")
            return False
            
        except Exception as e:
            logger.error(f"Parquet export failed: {e}")
            return False
    
    def _export_csv(self, output_path: str, filter_episodes: Optional[List[int]] = None, **options: Any) -> bool:
        """Export to CSV format."""
        try:
            # Would need pandas for CSV export
            logger.warning("CSV export requires pandas - not yet implemented") 
            return False
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    def _export_json(self, output_path: str, filter_episodes: Optional[List[int]] = None, **options: Any) -> bool:
        """Export to JSON format."""
        try:
            # JSON export implementation
            logger.warning("JSON export not yet implemented")
            return False
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False
    
    @contextmanager
    def hdf5_transaction(self):
        """
        Context manager for HDF5 transaction-like operations with automatic cleanup.
        
        Examples:
            >>> with recorder.hdf5_transaction():
            ...     recorder.create_group('/experiment_1')
            ...     recorder.create_dataset('/experiment_1/data', data_array)
        """
        if not self._h5_file:
            raise RuntimeError("HDF5 file is not initialized")

        try:
            with self._file_lock:
                yield self._h5_file
        except Exception as e:
            logger.error(f"HDF5 transaction failed: {e}")
            raise
        finally:
            # Ensure flush after transaction
            self.flush_buffer()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive HDF5-specific performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics including HDF5-specific timing and compression stats
        """
        base_metrics = super().get_performance_metrics()
        
        # Add HDF5-specific metrics
        hdf5_metrics = {
            'hdf5_file_path': self.hdf5_config.file_path,
            'hdf5_file_size_mb': 0.0,
            'compression_algorithm': self.hdf5_config.compression,
            'compression_level': self.hdf5_config.compression_opts,
            'average_write_time_ms': 0.0,
            'total_datasets': len(self._dataset_cache),
            'total_groups': len(self._group_cache),
            'chunk_size': self.hdf5_config.chunk_size
        }
        
        # Calculate file size
        try:
            if Path(self.hdf5_config.file_path).exists():
                file_size = Path(self.hdf5_config.file_path).stat().st_size
                hdf5_metrics['hdf5_file_size_mb'] = file_size / (1024 * 1024)
        except Exception:
            pass
        
        # Calculate average write time
        if self._write_times:
            hdf5_metrics['average_write_time_ms'] = np.mean(self._write_times) * 1000
        
        # Merge with base metrics
        base_metrics['hdf5_specific'] = hdf5_metrics
        
        return base_metrics
    
    def __del__(self):
        """Destructor with automatic HDF5 file cleanup."""
        try:
            self.close_file()
        except Exception:
            pass  # Avoid exceptions in destructor


# Export classes for public API
__all__ = [
    'HDF5Recorder',
    'HDF5Config'
]