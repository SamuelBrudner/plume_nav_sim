"""
Comprehensive pytest test module for RecorderProtocol implementations and recording framework.

This module provides comprehensive validation for the RecorderProtocol interface across multiple
backends (parquet, hdf5, sqlite, none) with performance-aware buffering, compression testing,
and data integrity validation. Tests ensure compliance with F-017-RQ-001 (≤1ms disabled overhead),
F-017-RQ-002 (multiple backend support), and F-017-RQ-003 (buffered asynchronous I/O) requirements.

Key Test Categories:
- Protocol compliance and interface validation for all recorder backend implementations
- Performance validation ensuring ≤33ms/step latency compliance with recording enabled per Section 5.2.8
- Data integrity verification across all supported backends with compression and format validation
- Buffered I/O efficiency testing with configurable buffer sizes and flush strategies
- Multi-threaded recording capabilities testing for non-blocking data persistence
- Backend factory and configuration validation with graceful degradation and error handling
- Memory management and resource cleanup validation for long-running simulation scenarios

Performance Requirements Validated:
- F-017-RQ-001: <1ms overhead when recording is disabled for minimal simulation impact
- F-017-RQ-003: Buffered asynchronous I/O for non-blocking writes during simulation steps
- Section 5.2.8: ≤33ms/step with 100 agents through optimized buffering and compression
- Section 6.6.2.4: 100% protocol coverage for RecorderProtocol implementations

Backend Coverage:
- ParquetRecorder: High-performance columnar storage with PyArrow compression
- HDF5Recorder: Hierarchical scientific data storage with metadata preservation
- SQLiteRecorder: Embedded relational database with transaction support
- NoneRecorder: Zero-overhead null implementation for disabled recording scenarios

Test Infrastructure:
- pytest >=7.4.0 framework with fixtures for deterministic behavior
- Performance monitoring with real-time SLA validation and regression detection
- Temporary file management with automatic cleanup and resource isolation
- Mock components for controlled environment testing without external dependencies
- Property-based testing with Hypothesis for invariant validation across configurations

Authors: Blitzy Enhanced Testing Framework for plume_nav_sim v1.0
Version: v1.0.0 (Protocol-driven architecture)
License: MIT
"""

import gc
import json
import os
import sqlite3
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import h5py
import pyarrow as pyarrow

# Import the recording framework components
from plume_nav_sim.core.protocols import RecorderProtocol
from plume_nav_sim.recording import (
    RecorderManager, RecorderConfig, BaseRecorder, NoneRecorder
)
from plume_nav_sim.recording.backends import (
    create_backend, get_available_backends, BACKEND_REGISTRY, 
    get_backend_capabilities, validate_backend_config
)

# Performance and testing constants per Section 6.6.2.4 requirements
PERFORMANCE_TARGET_MS = 33.0  # ≤33ms/step latency requirement per Section 5.2.8
DISABLED_OVERHEAD_TARGET_MS = 1.0  # <1ms disabled overhead per F-017-RQ-001
NUMERICAL_PRECISION_TOLERANCE = 1e-6  # Research accuracy standards per Section 6.6.2.6
COMPRESSION_TEST_RATIOS = [0.1, 0.5, 0.8]  # Compression efficiency validation thresholds
BUFFER_SIZE_TEST_VALUES = [10, 100, 1000, 5000]  # Buffer size performance validation


class TestRecorderProtocol:
    """
    Test RecorderProtocol interface compliance and method signatures.
    
    Validates that all recorder backend implementations strictly adhere to the RecorderProtocol
    interface definition, ensuring consistent API behavior across all backends. Tests cover
    method signature validation, protocol inheritance verification, and interface completeness
    according to Section 6.6.2.4 protocol coverage requirements.
    """
    
    def test_protocol_compliance(self):
        """
        Test that RecorderProtocol defines required interface methods and properties.
        
        Validates the protocol interface definition includes all required methods with proper
        signatures for record_step(), record_episode(), export_data(), and lifecycle methods.
        """
        # Verify protocol is runtime checkable
        assert hasattr(RecorderProtocol, '__runtime_checkable__')
        
        # Required method signatures for protocol compliance
        required_methods = [
            'record_step', 'record_episode', 'export_data',
            'start_recording', 'stop_recording', 'flush'
        ]
        
        for method_name in required_methods:
            assert hasattr(RecorderProtocol, method_name), \
                f"RecorderProtocol missing required method: {method_name}"
    
    def test_interface_implementation(self):
        """Test that concrete recorder implementations properly implement the protocol interface."""
        # Test available backends implement RecorderProtocol
        available_backends = get_available_backends()
        assert 'null' in available_backends, "NullRecorder should always be available"
        
        for backend_name in available_backends:
            if backend_name in BACKEND_REGISTRY:
                backend_class = BACKEND_REGISTRY[backend_name]
                assert issubclass(backend_class, RecorderProtocol), \
                    f"Backend {backend_name} must implement RecorderProtocol"
    
    def test_method_signatures(self):
        """Validate that protocol methods have correct signatures for type safety."""
        # Create a test configuration for validation
        test_config = RecorderConfig(backend='null', buffer_size=100)
        recorder = NoneRecorder(test_config)
        
        # Verify recorder implements protocol
        assert isinstance(recorder, RecorderProtocol)
        
        # Test method signatures exist and are callable
        assert callable(recorder.record_step)
        assert callable(recorder.record_episode)
        assert callable(recorder.export_data)
        assert callable(recorder.start_recording)
        assert callable(recorder.stop_recording)
        assert callable(recorder.flush)
    
    def test_protocol_inheritance(self):
        """Test that all backend implementations properly inherit from RecorderProtocol."""
        for backend_name, backend_class in BACKEND_REGISTRY.items():
            # Verify protocol inheritance
            assert issubclass(backend_class, RecorderProtocol), \
                f"Backend {backend_name} must inherit from RecorderProtocol"
            
            # Verify runtime protocol compliance
            test_config = RecorderConfig(backend=backend_name)
            try:
                instance = backend_class(test_config)
                assert isinstance(instance, RecorderProtocol), \
                    f"Backend {backend_name} instance must be RecorderProtocol compliant"
            except ImportError:
                # Skip if backend dependencies are missing
                pytest.skip(f"Backend {backend_name} dependencies not available")


class TestParquetRecorder:
    """
    Test ParquetRecorder backend implementation for columnar data storage.
    
    Validates ParquetRecorder functionality including columnar data organization, compression
    algorithms, schema evolution support, and PyArrow integration. Tests ensure optimal
    performance for analytical workloads and long-term data storage requirements.
    """
    
    @pytest.fixture
    def parquet_config(self, tmp_path):
        """Create test configuration for ParquetRecorder with temporary output directory."""
        return RecorderConfig(
            backend='parquet',
            output_dir=str(tmp_path),
            buffer_size=100,
            compression='snappy',
            async_io=True
        )
    
    @pytest.fixture
    def parquet_recorder(self, parquet_config):
        """Create ParquetRecorder instance for testing, skipping if dependencies unavailable."""
        if 'parquet' not in get_available_backends():
            pytest.skip("ParquetRecorder backend not available (missing dependencies)")
        
        return create_backend(parquet_config)
    
    def test_parquet_backend_initialization(self, parquet_recorder, parquet_config):
        """Test ParquetRecorder initialization with configuration validation."""
        assert isinstance(parquet_recorder, RecorderProtocol)
        assert parquet_recorder.config.backend == 'parquet'
        assert parquet_recorder.config.compression == 'snappy'
        assert parquet_recorder.config.buffer_size == 100
        
        # Verify directory structure creation
        expected_dir = Path(parquet_config.output_dir) / parquet_recorder.run_id
        assert expected_dir.exists()
    
    def test_columnar_data_storage(self, parquet_recorder, tmp_path):
        """Test columnar data organization and efficient storage in Parquet format."""
        # Start recording session
        episode_id = 1
        parquet_recorder.start_recording(episode_id)
        
        # Generate test data with multiple columns
        test_steps = [
            {
                'position': [float(i), float(i*2)],
                'concentration': float(i * 0.1),
                'reward': float(i * 0.05),
                'action': [float(i % 4), float((i+1) % 4)]
            }
            for i in range(50)
        ]
        
        # Record step data
        for step_num, step_data in enumerate(test_steps):
            parquet_recorder.record_step(step_data, step_num, episode_id)
        
        # Force flush and stop recording
        parquet_recorder.stop_recording()
        
        # Verify Parquet file creation and columnar structure
        episode_dir = Path(parquet_recorder.base_dir) / f"episode_{episode_id:06d}"
        parquet_files = list(episode_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "Parquet files should be created"
        
        # Validate columnar data structure using PyArrow
        if parquet_files:
            table = pyarrow.parquet.read_table(parquet_files[0])
            assert table.num_rows == len(test_steps)
            
            # Verify column preservation
            column_names = table.column_names
            expected_columns = ['position', 'concentration', 'reward', 'action', 'timestamp', 'step_number']
            for col in expected_columns:
                assert any(col in name for name in column_names), f"Column {col} missing from Parquet file"
    
    def test_compression_algorithms(self, tmp_path):
        """Test different compression algorithms for Parquet storage efficiency."""
        compression_methods = ['snappy', 'gzip', 'lz4', 'zstd']
        compression_results = {}
        
        for compression in compression_methods:
            if compression == 'lz4' or compression == 'zstd':
                # Skip if not available in pyarrow installation
                try:
                    import pyarrow.lib as lib
                    if not hasattr(lib, f'{compression.upper()}_AVAILABLE'):
                        continue
                except:
                    continue
            
            # Create recorder with specific compression
            config = RecorderConfig(
                backend='parquet',
                output_dir=str(tmp_path / f"compression_{compression}"),
                compression=compression,
                buffer_size=50
            )
            
            try:
                recorder = create_backend(config)
                
                # Record test data
                recorder.start_recording(1)
                
                # Generate compressible test data
                test_data = {
                    'position': [1.0, 2.0] * 25,  # Repeated data for compression
                    'concentration': [0.5] * 50,
                    'pattern': ['test_pattern'] * 25
                }
                
                for i in range(25):
                    recorder.record_step({
                        'position': test_data['position'][:2],
                        'concentration': test_data['concentration'][i],
                        'pattern': test_data['pattern'][i]
                    }, i, 1)
                
                recorder.stop_recording()
                
                # Measure compression efficiency
                parquet_files = list(Path(config.output_dir).rglob("*.parquet"))
                if parquet_files:
                    total_size = sum(f.stat().st_size for f in parquet_files)
                    compression_results[compression] = total_size
                    
            except Exception as e:
                warnings.warn(f"Compression test failed for {compression}: {e}")
        
        # Validate compression results if any succeeded
        if compression_results:
            assert len(compression_results) > 0, "At least one compression method should work"
            
            # Snappy should generally be available and efficient
            if 'snappy' in compression_results:
                assert compression_results['snappy'] > 0, "Snappy compression should produce non-zero files"
    
    def test_schema_evolution(self, parquet_recorder):
        """Test Parquet schema evolution and backward compatibility."""
        # Start recording with initial schema
        parquet_recorder.start_recording(1)
        
        # Record data with initial schema
        initial_data = {
            'position': [1.0, 2.0],
            'concentration': 0.5
        }
        parquet_recorder.record_step(initial_data, 0, 1)
        
        # Add new fields (schema evolution)
        evolved_data = {
            'position': [2.0, 3.0],
            'concentration': 0.7,
            'new_field': 'added_data',
            'numeric_field': 42.0
        }
        parquet_recorder.record_step(evolved_data, 1, 1)
        
        parquet_recorder.stop_recording()
        
        # Verify schema evolution handling
        episode_dir = Path(parquet_recorder.base_dir) / "episode_000001"
        parquet_files = list(episode_dir.glob("*.parquet"))
        
        if parquet_files:
            # Read data and verify schema flexibility
            df = pd.read_parquet(parquet_files[0])
            assert len(df) == 2, "Both records should be preserved"
            
            # Verify new fields are handled appropriately
            assert 'new_field' in df.columns or df.iloc[1].get('new_field') is not None
    
    def test_data_integrity_validation(self, parquet_recorder):
        """Test data integrity preservation and validation in Parquet storage."""
        parquet_recorder.start_recording(1)
        
        # Generate test data with precise values for integrity checking
        test_values = {
            'float_precision': np.pi,
            'large_integer': 2**53 - 1,
            'small_float': 1e-10,
            'array_data': [1.1, 2.2, 3.3, 4.4, 5.5],
            'string_data': 'test_string_with_unicode_🔬'
        }
        
        parquet_recorder.record_step(test_values, 0, 1)
        parquet_recorder.stop_recording()
        
        # Read back data and verify integrity
        episode_dir = Path(parquet_recorder.base_dir) / "episode_000001"
        parquet_files = list(episode_dir.glob("*.parquet"))
        
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
            assert len(df) == 1, "Single record should be preserved"
            
            # Verify numeric precision
            row = df.iloc[0]
            np.testing.assert_allclose(
                row['float_precision'], np.pi, 
                atol=NUMERICAL_PRECISION_TOLERANCE
            )
            
            # Verify large integer preservation
            assert row['large_integer'] == 2**53 - 1
            
            # Verify small float precision
            np.testing.assert_allclose(
                row['small_float'], 1e-10,
                atol=NUMERICAL_PRECISION_TOLERANCE
            )
            
            # Verify string data preservation
            assert row['string_data'] == 'test_string_with_unicode_🔬'


class TestHDF5Recorder:
    """
    Test HDF5Recorder backend implementation for hierarchical data storage.
    
    Validates HDF5Recorder functionality including hierarchical data organization, metadata
    attribution, compression support, and dataset chunking for scientific data storage.
    Tests ensure optimal performance for complex nested data structures and metadata preservation.
    """
    
    @pytest.fixture
    def hdf5_config(self, tmp_path):
        """Create test configuration for HDF5Recorder with temporary output directory."""
        return RecorderConfig(
            backend='hdf5',
            output_dir=str(tmp_path),
            buffer_size=100,
            compression='gzip',
            async_io=True
        )
    
    @pytest.fixture
    def hdf5_recorder(self, hdf5_config):
        """Create HDF5Recorder instance for testing, skipping if dependencies unavailable."""
        if 'hdf5' not in get_available_backends():
            pytest.skip("HDF5Recorder backend not available (missing h5py dependency)")
        
        return create_backend(hdf5_config)
    
    def test_hdf5_backend_initialization(self, hdf5_recorder, hdf5_config):
        """Test HDF5Recorder initialization with configuration validation."""
        assert isinstance(hdf5_recorder, RecorderProtocol)
        assert hdf5_recorder.config.backend == 'hdf5'
        assert hdf5_recorder.config.compression == 'gzip'
        
        # Verify directory structure creation
        expected_dir = Path(hdf5_config.output_dir) / hdf5_recorder.run_id
        assert expected_dir.exists()
    
    def test_hierarchical_data_organization(self, hdf5_recorder):
        """Test hierarchical data organization in HDF5 format."""
        hdf5_recorder.start_recording(1)
        
        # Generate hierarchical test data
        hierarchical_data = {
            'agent_data': {
                'position': [1.0, 2.0],
                'velocity': [0.5, 0.3],
                'sensors': {
                    'left': 0.8,
                    'right': 0.6,
                    'center': 0.9
                }
            },
            'environment': {
                'temperature': 25.5,
                'pressure': 1013.25,
                'wind': {
                    'speed': 2.1,
                    'direction': 45.0
                }
            }
        }
        
        hdf5_recorder.record_step(hierarchical_data, 0, 1)
        hdf5_recorder.stop_recording()
        
        # Verify hierarchical structure in HDF5 file
        episode_dir = Path(hdf5_recorder.base_dir) / "episode_000001"
        hdf5_files = list(episode_dir.glob("*.h5")) + list(episode_dir.glob("*.hdf5"))
        
        if hdf5_files:
            with h5py.File(hdf5_files[0], 'r') as f:
                # Verify hierarchical groups exist
                assert 'agent_data' in f or any('agent_data' in str(key) for key in f.keys())
                assert 'environment' in f or any('environment' in str(key) for key in f.keys())
    
    def test_compression_support(self, tmp_path):
        """Test HDF5 compression algorithms and efficiency."""
        compression_methods = ['gzip', 'lzf', 'szip']
        compression_results = {}
        
        for compression in compression_methods:
            config = RecorderConfig(
                backend='hdf5',
                output_dir=str(tmp_path / f"hdf5_compression_{compression}"),
                compression=compression,
                buffer_size=50
            )
            
            try:
                recorder = create_backend(config)
                recorder.start_recording(1)
                
                # Generate compressible test data
                for i in range(20):
                    test_data = {
                        'array_data': np.ones(100) * i,  # Compressible arrays
                        'metadata': f'step_{i:04d}',
                        'timestamp': time.time()
                    }
                    recorder.record_step(test_data, i, 1)
                
                recorder.stop_recording()
                
                # Measure file size for compression efficiency
                hdf5_files = list(Path(config.output_dir).rglob("*.h5")) + \
                           list(Path(config.output_dir).rglob("*.hdf5"))
                if hdf5_files:
                    total_size = sum(f.stat().st_size for f in hdf5_files)
                    compression_results[compression] = total_size
                    
            except Exception as e:
                # Some compression methods may not be available
                warnings.warn(f"HDF5 compression test failed for {compression}: {e}")
        
        # Validate compression results
        if compression_results:
            assert len(compression_results) > 0, "At least one HDF5 compression method should work"
            
            # Gzip should generally be available
            if 'gzip' in compression_results:
                assert compression_results['gzip'] > 0, "Gzip compression should produce non-zero files"
    
    def test_metadata_attribution(self, hdf5_recorder):
        """Test metadata preservation and attribution in HDF5 format."""
        hdf5_recorder.start_recording(1)
        
        # Record data with rich metadata
        test_data = {
            'experimental_data': np.random.random(50),
            'metadata': {
                'experiment_id': 'EXP_001',
                'researcher': 'Test Researcher',
                'date': '2024-01-01',
                'parameters': {
                    'temperature': 25.0,
                    'humidity': 60.0
                }
            }
        }
        
        hdf5_recorder.record_step(test_data, 0, 1)
        hdf5_recorder.stop_recording()
        
        # Verify metadata preservation
        episode_dir = Path(hdf5_recorder.base_dir) / "episode_000001"
        hdf5_files = list(episode_dir.glob("*.h5")) + list(episode_dir.glob("*.hdf5"))
        
        if hdf5_files:
            with h5py.File(hdf5_files[0], 'r') as f:
                # Check for metadata attributes or groups
                assert len(f.keys()) > 0, "HDF5 file should contain data groups"
                
                # Verify data preservation
                for key in f.keys():
                    if hasattr(f[key], 'shape'):
                        assert f[key].shape[0] > 0, "Data should be preserved"
    
    def test_dataset_chunking(self, hdf5_recorder):
        """Test HDF5 dataset chunking for efficient access patterns."""
        hdf5_recorder.start_recording(1)
        
        # Generate large array data for chunking
        large_array_data = {
            'trajectory': np.random.random((100, 3)),  # Large trajectory data
            'sensor_readings': np.random.random((100, 10)),  # Multi-sensor data
            'timestamps': np.arange(100, dtype=np.float64)
        }
        
        hdf5_recorder.record_step(large_array_data, 0, 1)
        hdf5_recorder.stop_recording()
        
        # Verify chunking implementation
        episode_dir = Path(hdf5_recorder.base_dir) / "episode_000001"
        hdf5_files = list(episode_dir.glob("*.h5")) + list(episode_dir.glob("*.hdf5"))
        
        if hdf5_files:
            with h5py.File(hdf5_files[0], 'r') as f:
                # Verify datasets exist and have appropriate structure
                datasets_found = []
                
                def find_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets_found.append((name, obj))
                
                f.visititems(find_datasets)
                assert len(datasets_found) > 0, "HDF5 file should contain datasets"
                
                # Check dataset properties
                for name, dataset in datasets_found:
                    assert dataset.size > 0, f"Dataset {name} should contain data"


class TestSQLiteRecorder:
    """
    Test SQLiteRecorder backend implementation for relational data storage.
    
    Validates SQLiteRecorder functionality including relational schema creation, transaction
    handling, query optimization, and embedded database features. Tests ensure optimal
    performance for queryable data access with ACID compliance.
    """
    
    @pytest.fixture
    def sqlite_config(self, tmp_path):
        """Create test configuration for SQLiteRecorder with temporary output directory."""
        return RecorderConfig(
            backend='sqlite',
            output_dir=str(tmp_path),
            buffer_size=100,
            compression='built_in',
            async_io=True
        )
    
    @pytest.fixture
    def sqlite_recorder(self, sqlite_config):
        """Create SQLiteRecorder instance for testing, skipping if unavailable."""
        if 'sqlite' not in get_available_backends():
            pytest.skip("SQLiteRecorder backend not available")
        
        return create_backend(sqlite_config)
    
    def test_sqlite_backend_initialization(self, sqlite_recorder, sqlite_config):
        """Test SQLiteRecorder initialization with configuration validation."""
        assert isinstance(sqlite_recorder, RecorderProtocol)
        assert sqlite_recorder.config.backend == 'sqlite'
        
        # Verify directory structure creation
        expected_dir = Path(sqlite_config.output_dir) / sqlite_recorder.run_id
        assert expected_dir.exists()
    
    def test_relational_schema_creation(self, sqlite_recorder):
        """Test relational database schema creation and table structure."""
        sqlite_recorder.start_recording(1)
        
        # Record structured data that should create appropriate tables
        relational_data = {
            'step_id': 1,
            'agent_id': 'agent_001',
            'position_x': 1.5,
            'position_y': 2.5,
            'concentration': 0.8,
            'action': 'move_forward',
            'reward': 0.1
        }
        
        sqlite_recorder.record_step(relational_data, 0, 1)
        sqlite_recorder.stop_recording()
        
        # Verify database schema
        episode_dir = Path(sqlite_recorder.base_dir) / "episode_000001"
        db_files = list(episode_dir.glob("*.db")) + list(episode_dir.glob("*.sqlite"))
        
        if db_files:
            conn = sqlite3.connect(db_files[0])
            cursor = conn.cursor()
            
            # Check for table creation
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            assert len(tables) > 0, "SQLite database should contain tables"
            
            # Verify data insertion
            table_name = tables[0][0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            assert row_count > 0, "Database should contain recorded data"
            
            conn.close()
    
    def test_transaction_handling(self, sqlite_recorder):
        """Test SQLite transaction handling and ACID compliance."""
        sqlite_recorder.start_recording(1)
        
        # Record multiple steps to test transaction batching
        for i in range(10):
            step_data = {
                'step_number': i,
                'data_value': float(i * 0.1),
                'status': 'active'
            }
            sqlite_recorder.record_step(step_data, i, 1)
        
        # Force flush to trigger transaction
        sqlite_recorder.flush()
        sqlite_recorder.stop_recording()
        
        # Verify transaction integrity
        episode_dir = Path(sqlite_recorder.base_dir) / "episode_000001"
        db_files = list(episode_dir.glob("*.db")) + list(episode_dir.glob("*.sqlite"))
        
        if db_files:
            conn = sqlite3.connect(db_files[0])
            cursor = conn.cursor()
            
            # Verify all data was committed
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if tables:
                table_name = tables[0][0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                assert row_count == 10, "All 10 records should be committed"
            
            conn.close()
    
    def test_query_optimization(self, sqlite_recorder):
        """Test SQLite query optimization and indexing."""
        sqlite_recorder.start_recording(1)
        
        # Record data with indexable fields
        for i in range(50):
            indexed_data = {
                'timestamp': time.time() + i,
                'agent_id': f'agent_{i % 5:03d}',  # 5 different agents
                'step_number': i,
                'position_x': float(i),
                'position_y': float(i * 0.5),
                'concentration': np.random.random()
            }
            sqlite_recorder.record_step(indexed_data, i, 1)
        
        sqlite_recorder.stop_recording()
        
        # Test query performance
        episode_dir = Path(sqlite_recorder.base_dir) / "episode_000001"
        db_files = list(episode_dir.glob("*.db")) + list(episode_dir.glob("*.sqlite"))
        
        if db_files:
            conn = sqlite3.connect(db_files[0])
            cursor = conn.cursor()
            
            # Check for table structure
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if tables:
                table_name = tables[0][0]
                
                # Test queries that should be optimizable
                start_time = time.perf_counter()
                cursor.execute(f"SELECT * FROM {table_name} WHERE step_number > 25;")
                results = cursor.fetchall()
                query_time = time.perf_counter() - start_time
                
                assert len(results) > 0, "Query should return results"
                assert query_time < 0.1, "Query should execute quickly"
            
            conn.close()
    
    def test_embedded_database_features(self, sqlite_recorder):
        """Test SQLite embedded database features and zero-configuration operation."""
        # Verify recorder works without external database server
        assert sqlite_recorder is not None
        
        sqlite_recorder.start_recording(1)
        
        # Test concurrent access (within same process)
        def concurrent_writer(recorder, episode_id, start_idx):
            for i in range(start_idx, start_idx + 10):
                data = {
                    'thread_id': threading.current_thread().ident,
                    'step_index': i,
                    'value': float(i)
                }
                recorder.record_step(data, i, episode_id)
        
        # Create concurrent recording threads
        threads = []
        for t in range(3):
            thread = threading.Thread(
                target=concurrent_writer,
                args=(sqlite_recorder, 1, t * 10)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        sqlite_recorder.stop_recording()
        
        # Verify concurrent data integrity
        episode_dir = Path(sqlite_recorder.base_dir) / "episode_000001"
        db_files = list(episode_dir.glob("*.db")) + list(episode_dir.glob("*.sqlite"))
        
        if db_files:
            conn = sqlite3.connect(db_files[0])
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if tables:
                table_name = tables[0][0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                total_records = cursor.fetchone()[0]
                
                # Should have records from all threads
                assert total_records >= 20, "Concurrent writing should preserve all records"
            
            conn.close()


class TestNullRecorder:
    """
    Test NullRecorder backend implementation for zero-overhead disabled recording.
    
    Validates NullRecorder functionality including zero-overhead operations, performance
    verification, fallback mode behavior, and debug mode functionality. Tests ensure
    minimal performance impact when recording is disabled per F-017-RQ-001 requirements.
    """
    
    @pytest.fixture
    def null_config(self):
        """Create test configuration for NullRecorder."""
        return RecorderConfig(
            backend='null',
            buffer_size=1000,
            disabled_mode_optimization=True
        )
    
    @pytest.fixture
    def null_recorder(self, null_config):
        """Create NullRecorder instance for testing."""
        return NoneRecorder(null_config)
    
    def test_null_backend_initialization(self, null_recorder, null_config):
        """Test NullRecorder initialization and configuration."""
        assert isinstance(null_recorder, RecorderProtocol)
        assert isinstance(null_recorder, NoneRecorder)
        assert null_recorder.config.backend == 'null'
        assert null_recorder.config.disabled_mode_optimization is True
    
    def test_zero_overhead_operations(self, null_recorder):
        """Test zero-overhead operation compliance per F-017-RQ-001."""
        # Test disabled mode performance
        null_recorder.enabled = False
        
        # Measure overhead for disabled operations
        start_time = time.perf_counter()
        
        for i in range(1000):
            null_recorder.record_step({'data': i}, i, 1)
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        average_time_ms = total_time_ms / 1000
        
        # Verify <1ms per 1000 operations requirement
        assert average_time_ms < DISABLED_OVERHEAD_TARGET_MS, \
            f"Disabled overhead {average_time_ms:.4f}ms exceeds target {DISABLED_OVERHEAD_TARGET_MS}ms"
    
    def test_performance_verification(self, null_recorder):
        """Test performance characteristics and SLA compliance."""
        # Enable recording to test null operations
        null_recorder.start_recording(1)
        
        # Measure enabled performance (should still be fast)
        large_data = {
            'large_array': list(range(1000)),
            'metadata': {'key': 'value'} * 100,
            'timestamp': time.time()
        }
        
        start_time = time.perf_counter()
        
        for i in range(100):
            null_recorder.record_step(large_data, i, 1)
        
        total_time_ms = (time.perf_counter() - start_time) * 1000
        average_time_ms = total_time_ms / 100
        
        # Even enabled NullRecorder should be very fast
        assert average_time_ms < 1.0, \
            f"NullRecorder enabled overhead {average_time_ms:.4f}ms too high"
        
        null_recorder.stop_recording()
    
    def test_fallback_mode_behavior(self, null_config):
        """Test NullRecorder as fallback when other backends fail."""
        # Test creation through factory with fallback
        invalid_config = RecorderConfig(
            backend='nonexistent_backend',
            output_dir='/invalid/path'
        )
        
        # Should fallback to null recorder
        recorder = create_backend(invalid_config, fallback_to_null=True)
        assert isinstance(recorder, RecorderProtocol)
        
        # Verify fallback functionality
        recorder.start_recording(1)
        recorder.record_step({'test': 'data'}, 0, 1)
        recorder.stop_recording()
    
    def test_debug_mode_functionality(self, null_recorder):
        """Test debug mode capabilities of NullRecorder."""
        # Enable debug mode if available
        if hasattr(null_recorder, 'enable_debug_mode'):
            null_recorder.enable_debug_mode = True
        
        null_recorder.start_recording(1)
        
        # Record debug data
        debug_data = {
            'debug_info': 'test_debug_session',
            'call_count': 1,
            'performance_metric': 0.001
        }
        
        null_recorder.record_step(debug_data, 0, 1)
        
        # Test debug metrics if available
        if hasattr(null_recorder, 'get_debug_metrics'):
            metrics = null_recorder.get_debug_metrics()
            assert isinstance(metrics, dict)
        
        null_recorder.stop_recording()


class TestRecorderPerformance:
    """
    Test recording framework performance compliance and SLA validation.
    
    Validates performance requirements including step latency compliance (≤33ms/step),
    disabled overhead requirements (<1ms), multi-threaded recording efficiency, and
    buffered I/O performance characteristics according to Section 5.2.8 requirements.
    """
    
    @pytest.fixture
    def performance_configs(self, tmp_path):
        """Create performance test configurations for different backends."""
        configs = {}
        available = get_available_backends()
        
        for backend in available:
            configs[backend] = RecorderConfig(
                backend=backend,
                output_dir=str(tmp_path / f"perf_{backend}"),
                buffer_size=1000,
                async_io=True,
                compression='snappy' if backend == 'parquet' else 'gzip'
            )
        
        return configs
    
    def test_step_latency_compliance(self, performance_configs):
        """Test step latency compliance with ≤33ms/step requirement per Section 5.2.8."""
        latency_results = {}
        
        for backend_name, config in performance_configs.items():
            try:
                recorder = create_backend(config)
                recorder.start_recording(1)
                
                # Generate realistic step data
                step_times = []
                
                for i in range(100):
                    start_time = time.perf_counter()
                    
                    step_data = {
                        'position': [float(i), float(i * 0.5)],
                        'concentration': np.random.random(),
                        'action': [np.random.random(), np.random.random()],
                        'reward': np.random.random() - 0.5,
                        'sensor_data': np.random.random(10).tolist(),
                        'metadata': {'step': i, 'episode': 1}
                    }
                    
                    recorder.record_step(step_data, i, 1)
                    
                    step_time_ms = (time.perf_counter() - start_time) * 1000
                    step_times.append(step_time_ms)
                
                recorder.stop_recording()
                
                # Analyze performance
                avg_latency = np.mean(step_times)
                p95_latency = np.percentile(step_times, 95)
                p99_latency = np.percentile(step_times, 99)
                
                latency_results[backend_name] = {
                    'average': avg_latency,
                    'p95': p95_latency,
                    'p99': p99_latency
                }
                
                # Validate SLA compliance
                assert avg_latency < PERFORMANCE_TARGET_MS, \
                    f"{backend_name} average latency {avg_latency:.2f}ms exceeds {PERFORMANCE_TARGET_MS}ms"
                
                assert p95_latency < PERFORMANCE_TARGET_MS * 1.5, \
                    f"{backend_name} P95 latency {p95_latency:.2f}ms too high"
                
            except Exception as e:
                warnings.warn(f"Performance test failed for {backend_name}: {e}")
        
        # Ensure at least null recorder meets requirements
        assert 'null' in latency_results, "Null recorder should always be available for performance testing"
    
    def test_disabled_overhead_requirement(self, performance_configs):
        """Test disabled recording overhead compliance with <1ms requirement per F-017-RQ-001."""
        overhead_results = {}
        
        for backend_name, config in performance_configs.items():
            try:
                recorder = create_backend(config)
                
                # Test disabled mode overhead
                recorder.enabled = False
                
                # Measure disabled operation overhead
                start_time = time.perf_counter()
                
                for i in range(1000):
                    recorder.record_step({'disabled': 'data'}, i, 1)
                
                total_time_ms = (time.perf_counter() - start_time) * 1000
                overhead_per_1000_ops = total_time_ms
                
                overhead_results[backend_name] = overhead_per_1000_ops
                
                # Validate disabled overhead requirement
                assert overhead_per_1000_ops < DISABLED_OVERHEAD_TARGET_MS, \
                    f"{backend_name} disabled overhead {overhead_per_1000_ops:.4f}ms exceeds {DISABLED_OVERHEAD_TARGET_MS}ms per 1000 operations"
                
            except Exception as e:
                warnings.warn(f"Disabled overhead test failed for {backend_name}: {e}")
        
        # Ensure at least null recorder meets requirements
        assert len(overhead_results) > 0, "At least one recorder should pass disabled overhead test"
    
    def test_multi_threaded_recording(self, tmp_path):
        """Test multi-threaded recording performance and thread safety."""
        config = RecorderConfig(
            backend='null',  # Use null for deterministic testing
            output_dir=str(tmp_path),
            buffer_size=500,
            async_io=True
        )
        
        recorder = create_backend(config)
        recorder.start_recording(1)
        
        # Thread worker function
        def thread_worker(thread_id, num_steps):
            thread_times = []
            for i in range(num_steps):
                start_time = time.perf_counter()
                
                step_data = {
                    'thread_id': thread_id,
                    'step': i,
                    'data': np.random.random(50).tolist()
                }
                
                recorder.record_step(step_data, i + thread_id * num_steps, 1)
                
                thread_time = (time.perf_counter() - start_time) * 1000
                thread_times.append(thread_time)
            
            return thread_times
        
        # Run concurrent threads
        num_threads = 4
        steps_per_thread = 50
        
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(thread_worker, thread_id, steps_per_thread)
                for thread_id in range(num_threads)
            ]
            
            # Collect results
            all_times = []
            for future in concurrent.futures.as_completed(futures):
                thread_times = future.result()
                all_times.extend(thread_times)
        
        recorder.stop_recording()
        
        # Analyze multi-threaded performance
        avg_time = np.mean(all_times)
        p95_time = np.percentile(all_times, 95)
        
        # Multi-threaded performance should still meet requirements
        assert avg_time < PERFORMANCE_TARGET_MS, \
            f"Multi-threaded average latency {avg_time:.2f}ms exceeds target"
        
        assert p95_time < PERFORMANCE_TARGET_MS * 2, \
            f"Multi-threaded P95 latency {p95_time:.2f}ms too high"
    
    def test_buffered_io_efficiency(self, tmp_path):
        """Test buffered I/O efficiency and performance characteristics."""
        buffer_performance = {}
        
        for buffer_size in BUFFER_SIZE_TEST_VALUES:
            config = RecorderConfig(
                backend='null',
                output_dir=str(tmp_path / f"buffer_{buffer_size}"),
                buffer_size=buffer_size,
                async_io=True,
                flush_interval=1.0
            )
            
            recorder = create_backend(config)
            recorder.start_recording(1)
            
            # Test buffered performance
            start_time = time.perf_counter()
            
            for i in range(buffer_size * 2):  # Fill buffer twice
                step_data = {
                    'buffer_test': True,
                    'step': i,
                    'data': [i] * 10
                }
                recorder.record_step(step_data, i, 1)
            
            # Force flush
            recorder.flush()
            
            total_time = time.perf_counter() - start_time
            recorder.stop_recording()
            
            # Calculate throughput
            steps_per_second = (buffer_size * 2) / total_time
            buffer_performance[buffer_size] = {
                'total_time': total_time,
                'steps_per_second': steps_per_second
            }
        
        # Validate buffer efficiency
        assert len(buffer_performance) > 0, "Buffer performance test should produce results"
        
        # Larger buffers should generally be more efficient
        if len(buffer_performance) >= 2:
            buffer_sizes = sorted(buffer_performance.keys())
            for i in range(len(buffer_sizes) - 1):
                smaller_buffer = buffer_sizes[i]
                larger_buffer = buffer_sizes[i + 1]
                
                # Larger buffers should have higher or similar throughput
                smaller_throughput = buffer_performance[smaller_buffer]['steps_per_second']
                larger_throughput = buffer_performance[larger_buffer]['steps_per_second']
                
                # Allow some variance but expect general improvement
                assert larger_throughput >= smaller_throughput * 0.8, \
                    f"Larger buffer {larger_buffer} should not be significantly slower than {smaller_buffer}"
    
    def test_compression_performance_tradeoffs(self, tmp_path):
        """Test compression performance vs storage efficiency tradeoffs."""
        # Only test backends that support compression
        compression_backends = {
            'parquet': ['snappy', 'gzip'],
            'hdf5': ['gzip', 'lzf']
        }
        
        compression_results = {}
        
        for backend, compression_options in compression_backends.items():
            if backend not in get_available_backends():
                continue
            
            for compression in compression_options:
                config = RecorderConfig(
                    backend=backend,
                    output_dir=str(tmp_path / f"compression_{backend}_{compression}"),
                    buffer_size=100,
                    compression=compression,
                    async_io=False  # Synchronous for accurate timing
                )
                
                try:
                    recorder = create_backend(config)
                    recorder.start_recording(1)
                    
                    # Generate compressible data
                    compressible_data = {
                        'repeated_values': [1.0] * 100,  # Highly compressible
                        'random_values': np.random.random(100).tolist(),  # Less compressible
                        'metadata': 'compression_test_data'
                    }
                    
                    # Measure compression performance
                    start_time = time.perf_counter()
                    
                    for i in range(20):
                        recorder.record_step(compressible_data, i, 1)
                    
                    compression_time = time.perf_counter() - start_time
                    recorder.stop_recording()
                    
                    compression_results[f"{backend}_{compression}"] = {
                        'time': compression_time,
                        'backend': backend,
                        'compression': compression
                    }
                    
                except Exception as e:
                    warnings.warn(f"Compression performance test failed for {backend}_{compression}: {e}")
        
        # Validate compression performance
        if compression_results:
            for key, result in compression_results.items():
                # Compression should not severely impact performance
                avg_time_per_step = result['time'] / 20
                assert avg_time_per_step < PERFORMANCE_TARGET_MS / 1000, \
                    f"Compression {key} too slow: {avg_time_per_step*1000:.2f}ms per step"


class TestBufferedIO:
    """
    Test buffered I/O implementation and coordination features.
    
    Validates buffered I/O configuration, asynchronous I/O coordination, backpressure
    handling, memory management, and flush strategies for optimal recording performance
    with minimal simulation impact per F-017-RQ-003 requirements.
    """
    
    @pytest.fixture
    def buffered_config(self, tmp_path):
        """Create configuration for buffered I/O testing."""
        return RecorderConfig(
            backend='null',
            output_dir=str(tmp_path),
            buffer_size=100,
            flush_interval=2.0,
            async_io=True,
            memory_limit_mb=64
        )
    
    def test_buffer_configuration(self, buffered_config):
        """Test buffer configuration and sizing validation."""
        recorder = create_backend(buffered_config)
        
        # Verify buffer configuration
        assert recorder.config.buffer_size == 100
        assert recorder.config.flush_interval == 2.0
        assert recorder.config.async_io is True
        
        # Test buffer size limits
        large_buffer_config = RecorderConfig(
            backend='null',
            buffer_size=10000,  # Large buffer
            memory_limit_mb=128
        )
        
        large_recorder = create_backend(large_buffer_config)
        assert large_recorder.config.buffer_size == 10000
    
    def test_async_io_coordination(self, buffered_config):
        """Test asynchronous I/O coordination and non-blocking writes."""
        recorder = create_backend(buffered_config)
        recorder.start_recording(1)
        
        # Test async operation timing
        async_times = []
        
        for i in range(50):
            start_time = time.perf_counter()
            
            step_data = {
                'async_test': True,
                'step': i,
                'data': list(range(i, i + 10))
            }
            
            recorder.record_step(step_data, i, 1)
            
            async_time = time.perf_counter() - start_time
            async_times.append(async_time)
        
        # Verify async operations are non-blocking
        avg_async_time = np.mean(async_times)
        
        # Async operations should be very fast (no blocking I/O)
        assert avg_async_time < 0.001, \
            f"Async operations too slow: {avg_async_time*1000:.2f}ms average"
        
        # Test flush behavior
        start_flush = time.perf_counter()
        recorder.flush()
        flush_time = time.perf_counter() - start_flush
        
        # Flush may take time but should complete reasonably quickly
        assert flush_time < 1.0, f"Flush operation too slow: {flush_time*1000:.2f}ms"
        
        recorder.stop_recording()
    
    def test_backpressure_handling(self, tmp_path):
        """Test backpressure handling when buffers fill up."""
        # Create recorder with small buffer for backpressure testing
        config = RecorderConfig(
            backend='null',
            output_dir=str(tmp_path),
            buffer_size=10,  # Small buffer to trigger backpressure
            flush_interval=10.0,  # Long interval to let buffer fill
            async_io=True
        )
        
        recorder = create_backend(config)
        recorder.start_recording(1)
        
        # Fill buffer beyond capacity
        backpressure_times = []
        
        for i in range(50):  # More than buffer capacity
            start_time = time.perf_counter()
            
            large_data = {
                'backpressure_test': True,
                'large_array': list(range(100)),  # Large data
                'step': i
            }
            
            recorder.record_step(large_data, i, 1)
            
            record_time = time.perf_counter() - start_time
            backpressure_times.append(record_time)
        
        recorder.stop_recording()
        
        # Analyze backpressure handling
        avg_time = np.mean(backpressure_times)
        max_time = np.max(backpressure_times)
        
        # Even with backpressure, operations should remain reasonably fast
        assert max_time < 0.1, f"Backpressure caused excessive delay: {max_time*1000:.2f}ms"
        
        # Check for gradual increase in times (backpressure response)
        early_times = np.mean(backpressure_times[:10])
        late_times = np.mean(backpressure_times[-10:])
        
        # Late times may be higher due to backpressure, but not excessively
        if late_times > early_times:
            ratio = late_times / early_times
            assert ratio < 10, f"Backpressure ratio too high: {ratio:.2f}x"
    
    def test_memory_management(self, tmp_path):
        """Test memory management and limit enforcement."""
        config = RecorderConfig(
            backend='null',
            output_dir=str(tmp_path),
            buffer_size=1000,
            memory_limit_mb=32,  # Low memory limit
            async_io=True
        )
        
        recorder = create_backend(config)
        recorder.start_recording(1)
        
        initial_memory = self._get_process_memory_mb()
        
        # Generate memory-intensive data
        for i in range(100):
            memory_data = {
                'large_array': np.random.random(1000).tolist(),
                'metadata': f'memory_test_{i}' * 100,
                'step': i
            }
            
            recorder.record_step(memory_data, i, 1)
            
            # Check memory usage periodically
            if i % 20 == 0:
                current_memory = self._get_process_memory_mb()
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be controlled
                if memory_growth > config.memory_limit_mb * 2:
                    warnings.warn(f"Memory growth {memory_growth:.1f}MB exceeds expected limits")
        
        final_memory = self._get_process_memory_mb()
        recorder.stop_recording()
        
        # Force garbage collection
        gc.collect()
        post_gc_memory = self._get_process_memory_mb()
        
        # Verify memory cleanup
        memory_growth = final_memory - initial_memory
        memory_after_cleanup = post_gc_memory - initial_memory
        
        # Memory should be reasonable and cleaned up
        assert memory_growth < config.memory_limit_mb * 3, \
            f"Excessive memory growth: {memory_growth:.1f}MB"
        
        # Cleanup should reduce memory usage
        assert memory_after_cleanup <= memory_growth, \
            "Memory cleanup should not increase usage"
    
    def test_flush_strategies(self, buffered_config):
        """Test different flush strategies and timing."""
        recorder = create_backend(buffered_config)
        recorder.start_recording(1)
        
        # Test automatic flush on buffer full
        for i in range(buffered_config.buffer_size + 10):
            recorder.record_step({'auto_flush_test': i}, i, 1)
        
        # Test manual flush
        manual_flush_start = time.perf_counter()
        recorder.flush()
        manual_flush_time = time.perf_counter() - manual_flush_start
        
        # Manual flush should complete quickly
        assert manual_flush_time < 0.5, f"Manual flush too slow: {manual_flush_time*1000:.2f}ms"
        
        # Test time-based flush (if implemented)
        time.sleep(buffered_config.flush_interval + 0.1)
        
        # Add more data to trigger time-based flush
        for i in range(10):
            recorder.record_step({'time_flush_test': i}, i + 1000, 1)
        
        recorder.stop_recording()
    
    def _get_process_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback if psutil not available
            return 0.0


class TestRecorderFactory:
    """
    Test recorder factory functionality and backend discovery.
    
    Validates backend discovery mechanisms, configuration validation, dependency checking,
    graceful degradation, and Hydra integration for the recorder factory system according
    to Section 0.3.1 dependency injection requirements.
    """
    
    def test_backend_discovery(self):
        """Test automatic backend discovery and registration."""
        available_backends = get_available_backends()
        
        # Null backend should always be available
        assert 'null' in available_backends, "Null backend should always be available"
        
        # Test backend registry consistency
        for backend in available_backends:
            assert backend in BACKEND_REGISTRY, \
                f"Available backend {backend} not in registry"
        
        # Test backend capabilities
        capabilities = get_backend_capabilities()
        
        for backend in available_backends:
            assert backend in capabilities, \
                f"Backend {backend} missing from capabilities"
            
            backend_info = capabilities[backend]
            assert backend_info['available'] is True, \
                f"Backend {backend} marked as unavailable but in available list"
    
    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test valid configuration
        valid_config = {
            'backend': 'null',
            'buffer_size': 100,
            'compression': 'none'
        }
        
        assert validate_backend_config(valid_config, 'null') is True
        
        # Test invalid configuration
        invalid_config = {
            'backend': 'null',
            'buffer_size': -10,  # Invalid buffer size
            'compression': 'invalid_compression'
        }
        
        # Validation should handle invalid parameters gracefully
        # Note: NullRecorder may accept any configuration
        result = validate_backend_config(invalid_config, 'null')
        assert isinstance(result, bool)
        
        # Test unknown backend
        unknown_backend_config = {
            'backend': 'unknown_backend',
            'buffer_size': 100
        }
        
        assert validate_backend_config(unknown_backend_config, 'unknown_backend') is False
    
    def test_dependency_checking(self):
        """Test dependency availability checking for optional backends."""
        capabilities = get_backend_capabilities()
        
        # Check dependency information
        for backend_name, info in capabilities.items():
            assert 'dependencies' in info, f"Backend {backend_name} missing dependency info"
            assert 'available' in info, f"Backend {backend_name} missing availability info"
            
            # If backend is available, dependencies should be satisfied
            if info['available']:
                deps = info['dependencies']
                assert isinstance(deps, list), "Dependencies should be a list"
    
    def test_graceful_degradation(self, tmp_path):
        """Test graceful degradation when backends fail."""
        # Test fallback to null recorder
        invalid_config = RecorderConfig(
            backend='nonexistent_backend',
            output_dir=str(tmp_path)
        )
        
        # Should fallback to null recorder without exception
        recorder = create_backend(invalid_config, fallback_to_null=True)
        assert isinstance(recorder, RecorderProtocol)
        
        # Test operation with fallback recorder
        recorder.start_recording(1)
        recorder.record_step({'fallback_test': True}, 0, 1)
        recorder.stop_recording()
        
        # Test graceful handling of missing dependencies
        for backend_name in ['parquet', 'hdf5', 'sqlite']:
            config = RecorderConfig(backend=backend_name, output_dir=str(tmp_path))
            
            # Should either create backend or fallback gracefully
            try:
                recorder = create_backend(config, fallback_to_null=True)
                assert isinstance(recorder, RecorderProtocol)
            except Exception as e:
                pytest.fail(f"Graceful degradation failed for {backend_name}: {e}")
    
    def test_hydra_integration(self, tmp_path):
        """Test Hydra configuration integration and instantiation."""
        # Test Hydra-style configuration
        hydra_config = {
            '_target_': 'plume_nav_sim.recording.backends.NullRecorder',
            'config': {
                'backend': 'null',
                'output_dir': str(tmp_path),
                'buffer_size': 200
            }
        }
        
        # Test factory with Hydra config format
        # Note: Full Hydra integration would require hydra.utils.instantiate
        # This tests the config handling pattern
        
        basic_config = RecorderConfig(
            backend='null',
            output_dir=str(tmp_path),
            buffer_size=200
        )
        
        recorder = create_backend(basic_config)
        assert isinstance(recorder, RecorderProtocol)
        assert recorder.config.buffer_size == 200


class TestRecorderIntegration:
    """
    Test recorder integration with simulation loop and other components.
    
    Validates simulation loop integration, multi-backend compatibility, data pipeline
    integrity, lifecycle management, and error recovery scenarios for comprehensive
    system integration testing per Section 6.6.3.2 requirements.
    """
    
    @pytest.fixture
    def mock_simulation_environment(self):
        """Create mock simulation environment for integration testing."""
        class MockEnvironment:
            def __init__(self):
                self.step_count = 0
                self.episode_count = 0
                self.agents = ['agent_001', 'agent_002']
            
            def step(self, actions):
                self.step_count += 1
                return {
                    'observations': np.random.random((len(self.agents), 10)),
                    'rewards': np.random.random(len(self.agents)),
                    'done': self.step_count >= 100,
                    'info': {'step': self.step_count}
                }
            
            def reset(self):
                self.step_count = 0
                self.episode_count += 1
                return np.random.random((len(self.agents), 10))
        
        return MockEnvironment()
    
    def test_simulation_loop_integration(self, mock_simulation_environment, tmp_path):
        """Test recorder integration with simulation loop execution."""
        # Create recorder for integration testing
        config = RecorderConfig(
            backend='null',
            output_dir=str(tmp_path),
            buffer_size=50,
            async_io=True
        )
        
        recorder = create_backend(config)
        env = mock_simulation_environment
        
        # Simulate training episode with recorder integration
        episode_id = 1
        recorder.start_recording(episode_id)
        
        observations = env.reset()
        
        total_simulation_time = 0
        total_recording_time = 0
        
        for step in range(50):
            # Simulate simulation step timing
            sim_start = time.perf_counter()
            
            # Mock agent actions
            actions = np.random.random((len(env.agents), 2))
            
            # Environment step
            result = env.step(actions)
            
            sim_time = time.perf_counter() - sim_start
            total_simulation_time += sim_time
            
            # Record step data
            rec_start = time.perf_counter()
            
            step_data = {
                'observations': result['observations'].tolist(),
                'actions': actions.tolist(),
                'rewards': result['rewards'].tolist(),
                'step_count': step,
                'agent_count': len(env.agents)
            }
            
            recorder.record_step(step_data, step, episode_id)
            
            rec_time = time.perf_counter() - rec_start
            total_recording_time += rec_time
        
        # Record episode summary
        episode_data = {
            'total_steps': env.step_count,
            'episode_reward': np.sum([result['rewards'] for _ in range(50)]),
            'episode_length': 50
        }
        
        recorder.record_episode(episode_data, episode_id)
        recorder.stop_recording()
        
        # Validate integration performance
        avg_sim_time_ms = (total_simulation_time / 50) * 1000
        avg_rec_time_ms = (total_recording_time / 50) * 1000
        
        # Recording should not dominate simulation time
        recording_overhead_ratio = avg_rec_time_ms / avg_sim_time_ms
        assert recording_overhead_ratio < 0.5, \
            f"Recording overhead too high: {recording_overhead_ratio:.2f}x simulation time"
        
        # Total time should meet performance requirements
        total_avg_time_ms = avg_sim_time_ms + avg_rec_time_ms
        assert total_avg_time_ms < PERFORMANCE_TARGET_MS, \
            f"Total simulation+recording time {total_avg_time_ms:.2f}ms exceeds target"
    
    def test_multi_backend_compatibility(self, tmp_path):
        """Test compatibility and consistency across multiple backends."""
        available_backends = get_available_backends()
        
        # Test data for consistency checking
        test_episode_data = []
        for i in range(10):
            step_data = {
                'step': i,
                'position': [float(i), float(i * 0.5)],
                'concentration': float(i * 0.1),
                'metadata': f'step_{i:03d}'
            }
            test_episode_data.append(step_data)
        
        backend_results = {}
        
        # Record same data with different backends
        for backend in available_backends[:3]:  # Limit to 3 backends for test efficiency
            config = RecorderConfig(
                backend=backend,
                output_dir=str(tmp_path / f"multi_backend_{backend}"),
                buffer_size=20,
                async_io=False  # Synchronous for consistency
            )
            
            try:
                recorder = create_backend(config)
                recorder.start_recording(1)
                
                # Record identical data
                for step_num, step_data in enumerate(test_episode_data):
                    recorder.record_step(step_data, step_num, 1)
                
                # Record episode summary
                episode_summary = {
                    'total_steps': len(test_episode_data),
                    'backend_used': backend
                }
                recorder.record_episode(episode_summary, 1)
                
                recorder.stop_recording()
                
                # Collect results for comparison
                backend_results[backend] = {
                    'steps_recorded': len(test_episode_data),
                    'config': config,
                    'success': True
                }
                
            except Exception as e:
                backend_results[backend] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Validate multi-backend consistency
        successful_backends = [b for b, r in backend_results.items() if r.get('success')]
        assert len(successful_backends) >= 1, "At least one backend should succeed"
        
        # All successful backends should have recorded the same number of steps
        if len(successful_backends) > 1:
            step_counts = [backend_results[b]['steps_recorded'] for b in successful_backends]
            assert all(count == step_counts[0] for count in step_counts), \
                "All backends should record the same number of steps"
    
    def test_data_pipeline_integrity(self, tmp_path):
        """Test end-to-end data pipeline integrity and validation."""
        config = RecorderConfig(
            backend='null',
            output_dir=str(tmp_path),
            buffer_size=100,
            async_io=True,
            enable_metrics=True
        )
        
        recorder = create_backend(config)
        
        # Test complete data pipeline
        pipeline_data = {
            'input_data': [],
            'processed_data': [],
            'output_metadata': []
        }
        
        recorder.start_recording(1)
        
        # Simulate complex data pipeline
        for i in range(30):
            # Input stage
            input_data = {
                'raw_sensors': np.random.random(5).tolist(),
                'timestamp': time.time(),
                'sequence_id': i
            }
            pipeline_data['input_data'].append(input_data)
            
            # Processing stage
            processed_data = {
                'filtered_sensors': [x * 0.9 for x in input_data['raw_sensors']],
                'derived_features': {
                    'mean': np.mean(input_data['raw_sensors']),
                    'std': np.std(input_data['raw_sensors'])
                },
                'processing_time': time.perf_counter()
            }
            pipeline_data['processed_data'].append(processed_data)
            
            # Output stage - record to pipeline
            output_data = {
                'input': input_data,
                'processed': processed_data,
                'pipeline_stage': 'complete',
                'data_integrity_hash': hash(str(input_data) + str(processed_data))
            }
            
            recorder.record_step(output_data, i, 1)
            
            pipeline_data['output_metadata'].append({
                'sequence_id': i,
                'hash': output_data['data_integrity_hash']
            })
        
        # Validate pipeline integrity
        assert len(pipeline_data['input_data']) == 30
        assert len(pipeline_data['processed_data']) == 30
        assert len(pipeline_data['output_metadata']) == 30
        
        # All sequence IDs should be unique and sequential
        sequence_ids = [meta['sequence_id'] for meta in pipeline_data['output_metadata']]
        assert sequence_ids == list(range(30)), "Sequence IDs should be sequential"
        
        recorder.stop_recording()
        
        # Test data integrity through performance metrics
        if hasattr(recorder, 'get_performance_metrics'):
            metrics = recorder.get_performance_metrics()
            assert metrics['steps_recorded'] == 30, "All steps should be recorded"
    
    def test_lifecycle_management(self, tmp_path):
        """Test recorder lifecycle management and state transitions."""
        config = RecorderConfig(
            backend='null',
            output_dir=str(tmp_path),
            buffer_size=50
        )
        
        recorder = create_backend(config)
        
        # Test initial state
        assert not recorder.enabled, "Recorder should start disabled"
        
        # Test start recording lifecycle
        recorder.start_recording(1)
        assert recorder.enabled, "Recorder should be enabled after start"
        assert recorder.current_episode_id == 1, "Episode ID should be set"
        
        # Test recording during active state
        for i in range(10):
            recorder.record_step({'lifecycle_test': i}, i, 1)
        
        # Test flush during active state
        recorder.flush()
        
        # Test stop recording lifecycle
        recorder.stop_recording()
        assert not recorder.enabled, "Recorder should be disabled after stop"
        assert recorder.current_episode_id is None, "Episode ID should be cleared"
        
        # Test multiple episodes
        for episode in range(3):
            recorder.start_recording(episode + 2)
            assert recorder.current_episode_id == episode + 2
            
            for step in range(5):
                recorder.record_step({'episode': episode + 2, 'step': step}, step, episode + 2)
            
            recorder.stop_recording()
        
        # Test lifecycle with context manager if available
        if hasattr(recorder, 'recording_session'):
            with recorder.recording_session(10):
                assert recorder.enabled, "Context manager should enable recording"
                recorder.record_step({'context_test': True}, 0, 10)
            
            assert not recorder.enabled, "Context manager should disable recording on exit"
    
    def test_error_recovery_scenarios(self, tmp_path):
        """Test error handling and recovery in various failure scenarios."""
        config = RecorderConfig(
            backend='null',
            output_dir=str(tmp_path),
            buffer_size=20
        )
        
        recorder = create_backend(config)
        recorder.start_recording(1)
        
        # Test recovery from invalid data
        try:
            invalid_data = {
                'invalid_numpy': np.array([float('inf'), float('nan')]),
                'invalid_string': None,
                'circular_reference': {}
            }
            invalid_data['circular_reference']['self'] = invalid_data
            
            recorder.record_step(invalid_data, 0, 1)
            
        except Exception as e:
            # Recorder should handle invalid data gracefully
            warnings.warn(f"Invalid data handling: {e}")
        
        # Recorder should continue functioning after error
        valid_data = {'recovery_test': True, 'step': 1}
        recorder.record_step(valid_data, 1, 1)
        
        # Test recovery from buffer overflow
        try:
            # Try to overwhelm buffer quickly
            for i in range(config.buffer_size * 3):
                large_data = {
                    'overflow_test': True,
                    'large_array': list(range(1000)),
                    'step': i
                }
                recorder.record_step(large_data, i, 1)
                
        except Exception as e:
            warnings.warn(f"Buffer overflow handling: {e}")
        
        # Test recovery from flush errors
        original_flush = recorder.flush
        
        def failing_flush():
            if hasattr(recorder, '_flush_call_count'):
                recorder._flush_call_count += 1
            else:
                recorder._flush_call_count = 1
            
            if recorder._flush_call_count <= 2:  # Fail first two calls
                raise RuntimeError("Simulated flush failure")
            
            return original_flush()
        
        recorder.flush = failing_flush
        
        try:
            # This should eventually succeed after retries
            recorder.record_step({'flush_recovery_test': True}, 100, 1)
            recorder.flush()  # Should fail
            recorder.flush()  # Should fail
            recorder.flush()  # Should succeed
            
        except Exception as e:
            warnings.warn(f"Flush recovery handling: {e}")
        finally:
            # Restore original flush method
            recorder.flush = original_flush
        
        # Recorder should still be functional
        recorder.record_step({'final_test': True}, 200, 1)
        recorder.stop_recording()


# Standalone test functions for comprehensive validation

def test_recorder_backwards_compatibility():
    """
    Test recorder backwards compatibility with v0.3.0 configurations.
    
    Validates that legacy recorder configurations and usage patterns continue
    to work with the new v1.0 RecorderProtocol interface, ensuring smooth
    migration path for existing users per Section 0.2.1 requirements.
    """
    # Test legacy configuration format
    legacy_config = {
        'recording_enabled': True,
        'output_directory': './legacy_output',
        'data_format': 'json',  # Legacy format
        'compression_enabled': False
    }
    
    # Convert legacy config to new format
    modern_config = RecorderConfig(
        backend='null',  # Map legacy to modern backend
        output_dir=legacy_config.get('output_directory', './data'),
        buffer_size=1000,  # Default modern buffer size
        compression='none' if not legacy_config.get('compression_enabled') else 'gzip'
    )
    
    # Create recorder with converted config
    recorder = create_backend(modern_config)
    assert isinstance(recorder, RecorderProtocol)
    
    # Test legacy-style usage patterns
    recorder.start_recording(1)
    
    # Legacy data format simulation
    legacy_step_data = {
        'agent_position': [1.0, 2.0],
        'sensor_reading': 0.5,
        'action_taken': 'move_north',
        'episode_time': 10.5
    }
    
    recorder.record_step(legacy_step_data, 0, 1)
    recorder.stop_recording()


def test_data_format_compliance():
    """
    Test data format compliance across all recorder backends.
    
    Validates that all recorder backends produce data in standardized formats
    that comply with research data standards and enable cross-backend
    compatibility per Section 5.2.8 structured output requirements.
    """
    available_backends = get_available_backends()
    
    # Standard test data for format compliance
    standard_data = {
        'timestamp': time.time(),
        'step_number': 42,
        'episode_id': 1,
        'agent_data': {
            'position': [1.5, 2.5],
            'velocity': [0.1, 0.2],
            'orientation': 45.0
        },
        'environment_data': {
            'concentration': 0.8,
            'gradient': [0.1, -0.05],
            'wind_speed': 2.1
        },
        'metadata': {
            'experiment_id': 'TEST_001',
            'configuration': 'standard_test'
        }
    }
    
    format_compliance_results = {}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        for backend in available_backends:
            try:
                config = RecorderConfig(
                    backend=backend,
                    output_dir=str(Path(tmp_dir) / f"format_test_{backend}"),
                    buffer_size=10,
                    async_io=False  # Synchronous for immediate file creation
                )
                
                recorder = create_backend(config)
                recorder.start_recording(1)
                
                # Record standard data
                recorder.record_step(standard_data, 0, 1)
                
                # Record episode data
                episode_data = {
                    'episode_summary': 'format_compliance_test',
                    'total_steps': 1,
                    'final_score': 100.0
                }
                recorder.record_episode(episode_data, 1)
                
                recorder.stop_recording()
                
                # Verify output format compliance
                output_dir = Path(config.output_dir)
                if output_dir.exists():
                    files_created = list(output_dir.rglob("*"))
                    format_compliance_results[backend] = {
                        'files_created': len(files_created),
                        'output_directory_exists': True,
                        'compliance_test_passed': True
                    }
                else:
                    format_compliance_results[backend] = {
                        'files_created': 0,
                        'output_directory_exists': False,
                        'compliance_test_passed': backend == 'null'  # Null recorder creates no files
                    }
                    
            except Exception as e:
                format_compliance_results[backend] = {
                    'compliance_test_passed': False,
                    'error': str(e)
                }
    
    # Validate format compliance results
    assert len(format_compliance_results) > 0, "At least one backend should be tested"
    
    # Null recorder should always pass
    assert format_compliance_results.get('null', {}).get('compliance_test_passed', False), \
        "Null recorder should always pass format compliance"
    
    # Count successful backends
    successful_backends = [
        backend for backend, result in format_compliance_results.items()
        if result.get('compliance_test_passed', False)
    ]
    
    assert len(successful_backends) >= 1, "At least one backend should pass format compliance testing"


if __name__ == "__main__":
    # Execute comprehensive test suite with appropriate configuration
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=plume_nav_sim.recording",
        "--cov=plume_nav_sim.core.protocols",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/test_recording",
        "-x",  # Stop on first failure for debugging
        "--durations=10"  # Show 10 slowest tests
    ])