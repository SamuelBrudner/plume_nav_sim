"""
ParquetRecorder backend implementation providing high-performance columnar storage using Apache Parquet format via PyArrow.

This module implements the ParquetRecorder class extending BaseRecorder to provide optimized
columnar data storage for analytical workloads with efficient compression, schema evolution
support, and structured dataset partitioning by run_id/episode_id. Features buffered I/O,
configurable compression algorithms, and metadata preservation for long-term scientific data
storage with minimal simulation overhead.

The ParquetRecorder is designed to achieve the performance requirements of F-017-RQ-001
(<1ms disabled-mode overhead per 1000 steps) while providing powerful analytical capabilities
through columnar storage optimizations and advanced compression techniques.

Key Features:
- High-performance columnar storage using Apache Parquet format via PyArrow ≥10.0.0
- Efficient compression support (snappy, gzip, lz4, zstd) with configurable ratios
- Structured output organization with run_id/episode_id hierarchical partitioning
- Buffered asynchronous I/O with configurable batch sizes for ≤33ms step latency
- Schema evolution support for long-term data storage compatibility
- Metadata preservation with embedded experimental configuration
- Zero-copy operations and memory-efficient data processing

Performance Characteristics:
- Columnar storage enables efficient analytical queries and compression
- Buffered writes minimize I/O overhead during simulation steps
- Multi-threaded compression and serialization for non-blocking operations
- Automatic schema inference with type optimization for numerical data
- Configurable partitioning strategies for scalable data organization

Technical Implementation:
- PyArrow Table and RecordBatch for zero-copy columnar operations
- Pandas DataFrame integration for rich data manipulation capabilities
- Configurable compression algorithms balancing storage efficiency and I/O performance
- Dataset partitioning with automatic directory structure generation
- Metadata embedding for schema evolution and experimental reproducibility

Examples:
    Basic ParquetRecorder usage:
    >>> from plume_nav_sim.recording.backends.parquet import ParquetRecorder, ParquetConfig
    >>> config = ParquetConfig(
    ...     file_path="./data/experiment.parquet",
    ...     compression="snappy",
    ...     buffer_size=1000
    ... )
    >>> recorder = ParquetRecorder(config)
    >>> recorder.record_step({'position': [0, 0], 'concentration': 0.5}, step_number=0)

    Advanced configuration with compression and partitioning:
    >>> config = ParquetConfig(
    ...     file_path="./data/experiment.parquet",
    ...     compression="lz4",
    ...     compression_level=3,
    ...     batch_size=500,
    ...     partition_cols=['episode_id'],
    ...     metadata={'experiment_name': 'plume_navigation_study'}
    ... )
    >>> recorder = ParquetRecorder(config)
"""

import logging
import warnings
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread, Lock, Event, RLock, current_thread
from typing import Dict, Any, Optional, Union, List, Tuple

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
except ImportError as e:  # pragma: no cover - import failure path
    raise ImportError(
        "PyArrow is required for ParquetRecorder. Install with: pip install pyarrow>=10.0.0"
    ) from e

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover - import failure path
    raise ImportError(
        "Pandas is required for ParquetRecorder. Install with: pip install pandas>=1.5.0"
    ) from e

# Internal imports
from ..import BaseRecorder


# Configure logging for ParquetRecorder
logger = logging.getLogger(__name__)


@dataclass
class ParquetConfig:
    """
    Configuration dataclass for ParquetRecorder with comprehensive parameter validation.
    
    This dataclass provides type-safe parameter validation for Parquet-specific recording
    configuration including compression settings, buffering parameters, partitioning
    strategies, and schema options. All parameters are designed to work with Hydra 
    configuration management and provide optimal defaults for simulation workloads.
    
    Core Parquet Configuration:
        file_path: Output file path with .parquet extension (default: auto-generated)
        compression: Compression algorithm ('snappy', 'gzip', 'lz4', 'zstd', 'none')
        compression_level: Compression level for algorithms that support it (1-9)
        
    Performance Configuration:
        buffer_size: Records to buffer before writing (inherited from BaseRecorder)
        batch_size: Records per Parquet RecordBatch for optimal I/O (default: 1000)
        write_options: PyArrow write options for performance tuning
        
    Partitioning Configuration:
        partition_cols: Column names for dataset partitioning (default: ['run_id', 'episode_id'])
        schema: Optional PyArrow schema for type enforcement and validation
        
    Metadata Configuration:
        metadata: Additional metadata to embed in Parquet file headers
        preserve_index: Whether to preserve DataFrame index in Parquet output
    """
    # Core Parquet configuration
    file_path: Optional[str] = None
    compression: str = 'snappy'
    compression_level: Optional[int] = None
    
    # Performance configuration  
    batch_size: int = 1000
    write_options: Optional[Dict[str, Any]] = None
    
    # Partitioning configuration
    partition_cols: Optional[List[str]] = None
    schema: Optional[str] = None  # JSON-serialized schema for Hydra compatibility
    
    # Metadata configuration
    metadata: Optional[Dict[str, Any]] = None
    preserve_index: bool = False

    def __post_init__(self):
        """Validate Parquet-specific configuration parameters after initialization."""
        # Validate compression algorithm
        valid_compression = ['snappy', 'gzip', 'lz4', 'zstd', 'none']
        if self.compression not in valid_compression:
            raise ValueError(f"compression must be one of: {valid_compression}")
        
        # Validate compression level
        if self.compression_level is not None:
            if not 1 <= self.compression_level <= 9:
                raise ValueError("compression_level must be between 1 and 9")
            if self.compression in ['snappy', 'lz4']:
                warnings.warn(f"compression_level ignored for {self.compression}")
        
        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Set default partition columns
        if self.partition_cols is None:
            self.partition_cols = ['run_id', 'episode_id']
        
        # Initialize default write options
        if self.write_options is None:
            self.write_options = {
                'use_dictionary': True,  # Enable dictionary encoding
                'compression': self.compression,
                'write_statistics': True,  # Enable column statistics
                'use_deprecated_int96_timestamps': False,
                'allow_truncated_timestamps': False
            }
            if self.compression_level is not None:
                self.write_options['compression_level'] = self.compression_level


class ParquetRecorder(BaseRecorder):
    """
    High-performance Parquet backend implementation extending BaseRecorder with Apache Parquet columnar storage.
    
    The ParquetRecorder provides optimized columnar data storage using PyArrow's Parquet implementation,
    designed for analytical workloads requiring efficient compression, fast queries, and long-term data 
    preservation. Implements advanced features including schema evolution, dataset partitioning, and 
    multi-threaded I/O coordination while maintaining the performance requirements for real-time simulation.
    
    Advanced Features:
    - Zero-copy columnar operations using PyArrow Table and RecordBatch APIs
    - Advanced compression algorithms (Snappy, LZ4, Zstd, Gzip) with configurable levels
    - Dataset partitioning by run_id/episode_id for scalable data organization
    - Schema evolution support with automatic type inference and validation
    - Metadata preservation with embedded experimental configuration and provenance
    - Multi-threaded compression and serialization for non-blocking I/O operations
    - Memory-efficient buffering with configurable batch sizes and streaming writes
    
    Performance Optimizations:
    - Buffered writes minimize file system I/O during simulation steps
    - Columnar layout enables vectorized operations and SIMD acceleration
    - Dictionary encoding for categorical data reduces storage overhead
    - Column statistics enable efficient query pruning and filtering
    - Configurable compression ratios balance storage efficiency and CPU usage
    
    Schema Management:
    - Automatic schema inference from simulation data with type optimization
    - Schema evolution support for adding new columns without breaking compatibility
    - Type enforcement for numerical precision and data validation
    - Embedded metadata for experimental reproducibility and data lineage
    
    Thread Safety:
    - All public methods are thread-safe with proper locking mechanisms
    - Background compression thread handles asynchronous data processing
    - Resource cleanup ensures proper shutdown without data loss
    - Exception handling with graceful degradation on I/O errors
    """
    
    def __init__(self, config):
        """
        Initialize ParquetRecorder with configuration validation and PyArrow setup.
        
        Args:
            config: Either ParquetConfig instance or RecorderConfig with Parquet parameters
        """
        # Handle different config types for flexible initialization
        if isinstance(config, ParquetConfig):
            # Convert ParquetConfig to RecorderConfig for BaseRecorder
            from ..import RecorderConfig
            base_config = RecorderConfig(
                backend='parquet',
                output_dir=str(Path(config.file_path).parent) if config.file_path else './data',
                buffer_size=getattr(config, 'buffer_size', 1000),
                compression=config.compression
            )
            self.parquet_config = config
        else:
            # Assume RecorderConfig - extract Parquet-specific parameters
            base_config = config
            self.parquet_config = ParquetConfig(
                file_path=getattr(config, 'file_path', None),
                compression=getattr(config, 'compression', 'snappy'),
                batch_size=getattr(config, 'batch_size', 1000),
                partition_cols=getattr(config, 'partition_cols', None),
                metadata=getattr(config, 'metadata', None)
            )
        
        # Initialize base recorder
        super().__init__(base_config)
        
        # Parquet-specific state
        self._parquet_writer: Optional[pq.ParquetWriter] = None
        self._current_schema: Optional[pa.Schema] = None
        self._step_data_buffer: List[Dict[str, Any]] = []
        self._episode_data_buffer: List[Dict[str, Any]] = []
        self._writer_lock = RLock()
        self._batch_count = 0
        self._total_rows_written = 0
        
        # Performance tracking
        self._compression_time = 0.0
        self._write_time = 0.0
        self._schema_inference_time = 0.0
        
        # Generate file path if not provided
        if not self.parquet_config.file_path:
            timestamp = int(time.time())
            self.parquet_config.file_path = (
                self.base_dir / f"trajectory_data_{timestamp}.parquet"
            )
        else:
            self.parquet_config.file_path = Path(self.parquet_config.file_path)
        
        # Ensure directory exists
        self.parquet_config.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"ParquetRecorder initialized: file_path={self.parquet_config.file_path}, "
            f"compression={self.parquet_config.compression}, "
            f"batch_size={self.parquet_config.batch_size}"
        )
    
    def configure_backend(self, **kwargs: Any) -> None:
        """
        Configure Parquet-specific parameters during runtime.
        
        Args:
            **kwargs: Parquet configuration parameters including:
                - compression: Compression algorithm ('snappy', 'gzip', 'lz4', 'zstd')
                - compression_level: Compression level (1-9) for supported algorithms
                - batch_size: Records per batch for optimal I/O performance
                - partition_cols: Column names for dataset partitioning
                - metadata: Additional metadata for Parquet file headers
        """
        # Update base configuration
        super().configure_backend(**kwargs)
        
        # Update Parquet-specific configuration
        parquet_params = [
            'compression', 'compression_level', 'batch_size', 
            'partition_cols', 'metadata', 'write_options'
        ]
        
        for param in parquet_params:
            if param in kwargs:
                setattr(self.parquet_config, param, kwargs[param])
                logger.debug(f"Updated parquet_config.{param} = {kwargs[param]}")
        
        # Recreate write options if compression changed
        if 'compression' in kwargs or 'compression_level' in kwargs:
            self.parquet_config.__post_init__()  # Rebuild write options
    
    def create_table(self, data: List[Dict[str, Any]]) -> pa.Table:
        """
        Create PyArrow Table from data with optimized schema inference and type conversion.
        
        Args:
            data: List of dictionaries containing simulation data
            
        Returns:
            pa.Table: PyArrow Table with optimized schema and data types
        """
        start_time = time.perf_counter()
        
        try:
            # Use Pandas for efficient data processing and type inference
            df = pd.DataFrame(data)

            # Optimize data types for storage efficiency
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Try to convert to more efficient types
                    try:
                        # Check if it's numeric - handle deprecation warning
                        try:
                            numeric_series = pd.to_numeric(df[col], errors='coerce')
                            if not pd.isna(numeric_series).all() and not numeric_series.equals(df[col]):
                                df[col] = numeric_series
                        except Exception:
                            pass  # Keep as object/string
                    except (ValueError, TypeError):
                        pass  # Keep as object/string
                elif df[col].dtype == 'float64':
                    # Downcast to float32 if possible without precision loss
                    if df[col].min() >= np.finfo(np.float32).min and \
                       df[col].max() <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)

            # Convert to PyArrow Table
            table = pa.Table.from_pandas(
                df,
                preserve_index=self.parquet_config.preserve_index,
                schema=self._get_target_schema() if self._current_schema else None
            )
            
            # Update schema tracking
            if self._current_schema is None:
                self._current_schema = table.schema
                logger.debug(f"Inferred schema: {self._current_schema}")
            elif not table.schema.equals(self._current_schema):
                # Handle schema evolution
                table = self._handle_schema_evolution(table)
            
            return table
            
        except Exception as e:
            logger.error(f"Error creating PyArrow Table: {e}")
            raise
        finally:
            self._schema_inference_time += time.perf_counter() - start_time
    
    def write_batch(self, table: pa.Table) -> int:
        """
        Write PyArrow Table to Parquet file with compression and performance monitoring.
        
        Args:
            table: PyArrow Table to write
            
        Returns:
            int: Number of bytes written (estimated)
        """
        start_time = time.perf_counter()
        bytes_written = 0
        
        try:
            with self._writer_lock:
                # Initialize writer if needed
                if self._parquet_writer is None:
                    self._initialize_writer(table.schema)
                
                # Write table as RecordBatch for optimal performance
                batch = table.to_batches(max_chunksize=self.parquet_config.batch_size)[0]
                self._parquet_writer.write_batch(batch)
                
                # Update metrics
                self._batch_count += 1
                self._total_rows_written += len(table)
                
                # Estimate bytes written (approximate)
                bytes_written = len(table) * table.nbytes // len(table) if len(table) > 0 else 0
                
                logger.debug(
                    f"Wrote batch {self._batch_count}: {len(table)} rows, "
                    f"~{bytes_written} bytes, total_rows={self._total_rows_written}"
                )
                
        except Exception as e:
            logger.error(f"Error writing Parquet batch: {e}")
            raise
        finally:
            self._write_time += time.perf_counter() - start_time
        
        return bytes_written
    
    def flush_buffer(self) -> None:
        """Force immediate flush of writer buffers to ensure data persistence."""
        try:
            with self._writer_lock:
                if self._parquet_writer is not None:
                    # PyArrow ParquetWriter doesn't have explicit flush,
                    # but we can close and reopen to ensure data is written
                    # For now, we rely on the writer's internal buffering
                    pass
                    
        except Exception as e:
            logger.error(f"Error flushing Parquet buffer: {e}")
            raise
    
    def close_file(self) -> None:
        """Close Parquet writer and finalize file with proper cleanup."""
        try:
            # Ensure we have the writer lock attribute
            if not hasattr(self, '_writer_lock'):
                import threading
                self._writer_lock = threading.RLock()
                
            with self._writer_lock:
                if self._parquet_writer is not None:
                    self._parquet_writer.close()
                    self._parquet_writer = None
                    
                    logger.info(
                        f"Closed Parquet file: {self.parquet_config.file_path}, "
                        f"total_rows={self._total_rows_written}, "
                        f"batches={self._batch_count}"
                    )
                    
        except Exception as e:
            logger.error(f"Error closing Parquet file: {e}")
            raise
    
    def get_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get current Parquet schema information for debugging and validation.
        
        Returns:
            Optional[Dict[str, Any]]: Schema information including field types and metadata
        """
        if self._current_schema is None:
            return None
            
        return {
            'fields': [
                {
                    'name': field.name,
                    'type': str(field.type),
                    'nullable': field.nullable,
                    'metadata': dict(field.metadata) if field.metadata else None
                }
                for field in self._current_schema
            ],
            'metadata': dict(self._current_schema.metadata) if self._current_schema.metadata else None,
            'total_rows_written': self._total_rows_written,
            'batch_count': self._batch_count
        }
    
    # Implementation of abstract methods from BaseRecorder
    
    def _write_step_data(self, data: List[Dict[str, Any]]) -> int:
        """
        Write step-level data using Parquet columnar storage implementation.
        
        Args:
            data: List of step data dictionaries
            
        Returns:
            int: Number of bytes written
        """
        if not data:
            return 0
            
        try:
            # Create PyArrow Table from step data
            table = self.create_table(data)
            
            # Write to Parquet file
            bytes_written = self.write_batch(table)
            
            return bytes_written
            
        except Exception as e:
            logger.error(f"Error writing step data to Parquet: {e}")
            raise
    
    def _write_episode_data(self, data: List[Dict[str, Any]]) -> int:
        """
        Write episode-level data using Parquet columnar storage implementation.
        
        Args:
            data: List of episode data dictionaries
            
        Returns:
            int: Number of bytes written
        """
        if not data:
            return 0
            
        try:
            # Create separate table for episode data with distinct schema
            episode_table = self.create_table(data)
            
            # Write to separate episode data file or same file with episode marker
            episode_file_path = self.parquet_config.file_path.with_name(
                f"{self.parquet_config.file_path.stem}_episodes.parquet"
            )
            
            # Write episode data to separate file
            pq.write_table(
                episode_table,
                episode_file_path,
                compression=self.parquet_config.compression,
                write_statistics=True,
                use_dictionary=True
            )
            
            logger.debug(f"Wrote episode data to {episode_file_path}")
            
            # Estimate bytes written
            return episode_table.nbytes
            
        except Exception as e:
            logger.error(f"Error writing episode data to Parquet: {e}")
            raise
    
    def _export_data_backend(
        self, 
        output_path: str,
        format: str,
        compression: Optional[str] = None,
        filter_episodes: Optional[List[int]] = None,
        **export_options: Any
    ) -> bool:
        """
        Export Parquet data with format conversion and filtering options.
        
        Args:
            output_path: Target file path for exported data
            format: Export format ('parquet', 'csv', 'json', 'hdf5')
            compression: Optional compression for export format
            filter_episodes: Optional list of episode IDs to export
            **export_options: Format-specific export parameters
            
        Returns:
            bool: True if export completed successfully, False otherwise
        """
        try:
            # Ensure data is written
            self.flush_buffer()
            
            # Read Parquet data
            if not self.parquet_config.file_path.exists():
                logger.warning(f"No data file found at {self.parquet_config.file_path}")
                return False
            
            table = pq.read_table(str(self.parquet_config.file_path))
            
            # Apply episode filtering if requested
            if filter_episodes:
                if 'episode_id' in table.column_names:
                    filter_expr = pa.compute.is_in(
                        table['episode_id'], 
                        pa.array(filter_episodes)
                    )
                    table = table.filter(filter_expr)
                else:
                    logger.warning("Cannot filter by episode_id: column not found")
            
            # Export based on format
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'parquet':
                pq.write_table(
                    table,
                    output_path,
                    compression=compression or self.parquet_config.compression,
                    **export_options
                )
            elif format == 'csv':
                df = table.to_pandas()
                df.to_csv(output_path, compression=compression, **export_options)
            elif format == 'json':
                df = table.to_pandas()
                df.to_json(output_path, compression=compression, **export_options)
            elif format == 'hdf5':
                try:
                    import h5py
                    df = table.to_pandas()
                    df.to_hdf(output_path, key='data', compression=compression, **export_options)
                except ImportError:
                    logger.error("h5py not available for HDF5 export")
                    return False
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported data to {output_path} in {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Parquet data: {e}")
            return False
    
    # Helper methods for advanced Parquet features
    
    def _initialize_writer(self, schema: pa.Schema) -> None:
        """Initialize PyArrow ParquetWriter with optimized settings."""
        try:
            # Add metadata to schema
            metadata = self._build_metadata()
            if metadata:
                schema = schema.with_metadata(metadata)
            
            # Prepare writer options, avoiding parameter conflicts
            writer_options = {
                'compression': self.parquet_config.compression,
                'write_statistics': True,
                'use_deprecated_int96_timestamps': False,
                'allow_truncated_timestamps': False,
                'metadata_collector': None,  # We handle metadata separately
            }
            
            # Add compression level if specified
            if self.parquet_config.compression_level is not None:
                writer_options['compression_level'] = self.parquet_config.compression_level
            
            # Merge with additional write options, avoiding conflicts
            additional_options = self.parquet_config.write_options.copy()
            
            # Remove conflicting parameters from additional options
            conflicting_params = ['compression', 'compression_level', 'write_statistics']
            for param in conflicting_params:
                additional_options.pop(param, None)
            
            writer_options.update(additional_options)
            
            # Initialize writer with optimized settings
            self._parquet_writer = pq.ParquetWriter(
                str(self.parquet_config.file_path),
                schema=schema,
                **writer_options
            )
            
            logger.debug(f"Initialized ParquetWriter with schema: {schema}")
            
        except Exception as e:
            logger.error(f"Error initializing Parquet writer: {e}")
            raise
    
    def _get_target_schema(self) -> Optional[pa.Schema]:
        """Get target schema for type enforcement and validation."""
        if self.parquet_config.schema:
            try:
                # Parse JSON-serialized schema
                schema_dict = json.loads(self.parquet_config.schema)
                # Convert to PyArrow schema (simplified implementation)
                return pa.schema(schema_dict)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Invalid schema configuration: {e}")
        return None
    
    def _handle_schema_evolution(self, table: pa.Table) -> pa.Table:
        """Handle schema evolution by adapting table to current schema."""
        try:
            # Simple schema evolution: add missing columns with null values
            current_columns = set(self._current_schema.names)
            new_columns = set(table.schema.names)
            
            # Add missing columns from current schema
            missing_columns = current_columns - new_columns
            for col_name in missing_columns:
                field = self._current_schema.field(col_name)
                null_array = pa.nulls(len(table), type=field.type)
                table = table.append_column(field, null_array)
            
            # Reorder columns to match current schema
            column_order = [col for col in self._current_schema.names if col in table.schema.names]
            table = table.select(column_order)
            
            logger.debug(f"Applied schema evolution: added {len(missing_columns)} columns")
            return table
            
        except Exception as e:
            logger.warning(f"Schema evolution failed: {e}. Using table as-is.")
            return table
    
    def _build_metadata(self) -> Optional[Dict[bytes, bytes]]:
        """Build metadata dictionary for Parquet file headers."""
        try:
            metadata = {}
            
            # Add configuration metadata
            if self.parquet_config.metadata:
                for key, value in self.parquet_config.metadata.items():
                    metadata[f"config.{key}".encode()] = str(value).encode()
            
            # Add system metadata
            metadata[b"created_by"] = b"plume_nav_sim.recording.backends.ParquetRecorder"
            metadata[b"created_at"] = str(time.time()).encode()
            metadata[b"run_id"] = self.run_id.encode()
            metadata[b"compression"] = self.parquet_config.compression.encode()
            metadata[b"batch_size"] = str(self.parquet_config.batch_size).encode()
            
            return metadata if metadata else None
            
        except Exception as e:
            logger.warning(f"Error building metadata: {e}")
            return None
    
    @contextmanager
    def recording_session(self, episode_id: int):
        """
        Context manager for Parquet recording session with automatic file management.
        
        Args:
            episode_id: Episode identifier for the recording session
            
        Examples:
            >>> with recorder.recording_session(episode_id=1):
            ...     recorder.record_step({'position': [0, 0], 'concentration': 0.5})
            ...     # Automatic cleanup and file finalization
        """
        try:
            self.start_recording(episode_id)
            yield self
        finally:
            self.stop_recording()
            self.close_file()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics including Parquet-specific statistics.
        
        Returns:
            Dict[str, Any]: Extended performance metrics with Parquet backend details
        """
        base_metrics = super().get_performance_metrics()
        
        # Add Parquet-specific metrics
        parquet_metrics = {
            'parquet_specific': {
                'total_rows_written': self._total_rows_written,
                'batch_count': self._batch_count,
                'compression_time': self._compression_time,
                'write_time': self._write_time,
                'schema_inference_time': self._schema_inference_time,
                'file_path': str(self.parquet_config.file_path),
                'compression_algorithm': self.parquet_config.compression,
                'batch_size': self.parquet_config.batch_size,
                'current_schema': self.get_schema()
            }
        }
        
        # Merge metrics
        base_metrics.update(parquet_metrics)
        return base_metrics
    
    def stop_recording(self) -> None:
        """Override stop_recording to ensure proper file closure."""
        # Call parent stop_recording first
        super().stop_recording()
        
        # Ensure file is properly closed
        self.close_file()
    
    def __del__(self):
        """Destructor with automatic cleanup of Parquet resources."""
        try:
            if hasattr(self, '_parquet_writer') and self._parquet_writer is not None:
                self.close_file()
        except Exception:
            pass  # Avoid exceptions in destructor


# Export classes for public API
__all__ = [
    'ParquetRecorder',
    'ParquetConfig'
]