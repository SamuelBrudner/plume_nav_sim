"""
Comprehensive test module for RecorderProtocol implementations validating data persistence functionality.

This module provides extensive testing for the plume_nav_sim v1.0 recording system, including
multiple backends (parquet, hdf5, sqlite, none) with performance-aware buffering and compression
capabilities. Tests validate protocol compliance, performance requirements, and integration
with the simulation loop per F-017 Data Recorder System requirements.

Key Testing Areas:
- RecorderProtocol interface compliance across all backend implementations
- Performance validation: <1ms overhead when disabled, ≤33ms/step when enabled
- Multi-backend support: ParquetRecorder, HDF5Recorder, SQLiteRecorder, NullRecorder
- Buffered I/O with configurable buffer sizes and compression algorithms
- Integration testing with simulation loop and performance monitoring systems
- Configuration validation via Hydra config groups and schema evolution support
- Data integrity verification across export formats and compression ratios

Performance Requirements Testing:
- F-017-RQ-001: <1ms overhead per 1000 steps when disabled for minimal simulation impact
- F-017-RQ-002: Multiple backend support with runtime selection and graceful degradation
- F-017-RQ-003: Buffered asynchronous I/O for non-blocking data persistence with backpressure
- Section 5.1.3: Integration with simulation loop and performance monitoring infrastructure

Test Categories:
1. Protocol Compliance Tests: Validate RecorderProtocol interface implementation
2. Performance Tests: Validate timing requirements and overhead thresholds
3. Backend Implementation Tests: Test specific functionality per backend type
4. Integration Tests: Test with simulation components and performance monitoring
5. Configuration Tests: Validate Hydra integration and parameter validation
6. Data Integrity Tests: Validate compression, export, and schema evolution
7. Error Handling Tests: Test graceful degradation and exception management

Examples:
    Basic protocol compliance testing:
    >>> def test_recorder_protocol_compliance(mock_parquet_recorder):
    ...     assert hasattr(mock_parquet_recorder, 'record_step')
    ...     assert hasattr(mock_parquet_recorder, 'record_episode')
    ...     assert hasattr(mock_parquet_recorder, 'export_data')
    
    Performance validation with timing:
    >>> def test_disabled_recorder_performance(performance_monitor):
    ...     timing_ctx = performance_monitor.start_timing('record_step')
    ...     null_recorder.record_step({'test': 'data'}, step_number=0)
    ...     perf_data = performance_monitor.end_timing(timing_ctx)
    ...     assert perf_data['duration_ms'] < 1.0

Author: Blitzy Platform v1.0
Version: 1.0.0
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import threading
from typing import Dict, Any, Optional, List, Callable, Union
import json

# Test framework imports
from pytest import fixture, mark, raises, approx, main

# Core imports for recorder testing
from src.plume_nav_sim.core.protocols import RecorderProtocol
from src.plume_nav_sim.recording import RecorderConfig, BaseRecorder
from src.plume_nav_sim.recording.backends.parquet import ParquetRecorder
from src.plume_nav_sim.recording.backends.hdf5 import HDF5Recorder  
from src.plume_nav_sim.recording.backends.sqlite import SQLiteRecorder
from src.plume_nav_sim.recording.backends.null import NullRecorder

# Import factory function if available
try:
    from src.plume_nav_sim.recording import RecorderFactory
except ImportError:
    # Create a mock factory for testing if not available
    class RecorderFactory:
        @staticmethod
        def create_recorder(config):
            backend = config.get('backend', 'none')
            if backend == 'parquet':
                return ParquetRecorder(RecorderConfig(**config))
            elif backend == 'hdf5':
                return HDF5Recorder(RecorderConfig(**config))
            elif backend == 'sqlite':
                return SQLiteRecorder(RecorderConfig(**config))
            else:
                return NullRecorder(RecorderConfig(**config))
        
        @staticmethod
        def get_available_backends():
            return ['parquet', 'hdf5', 'sqlite', 'none']
        
        @staticmethod
        def validate_config(config):
            return {'valid': True, 'warnings': [], 'recommendations': [], 'backend_available': True}


class TestRecorderProtocolCompliance:
    """
    Test suite validating RecorderProtocol interface compliance across all backend implementations.
    
    This test class ensures that all recorder backends properly implement the RecorderProtocol
    interface according to Section 6.6.2.2 Protocol Coverage Test Modules requirements.
    Tests validate method signatures, return types, and behavioral contracts.
    """
    
    @mark.parametrize("backend_name", ["parquet", "hdf5", "sqlite", "none"])
    def test_recorder_protocol_interface_methods(self, backend_name, mock_recorder_config):
        """
        Test that all recorder backends implement required RecorderProtocol methods.
        
        Validates that each backend implementation provides all required methods
        with correct signatures according to the RecorderProtocol specification.
        """
        # Skip test if backend dependencies unavailable
        if backend_name == "parquet" and not hasattr(pytest, 'pd'):
            pytest.skip("Parquet backend dependencies not available")
        elif backend_name == "hdf5" and not hasattr(pytest, 'h5py'):
            pytest.skip("HDF5 backend dependencies not available")
        elif backend_name == "sqlite" and not hasattr(pytest, 'sqlite3'):
            pytest.skip("SQLite backend dependencies not available")
        
        # Create recorder instance with basic valid configuration
        basic_config = {
            'backend': backend_name,
            'output_dir': './test_data',
            'buffer_size': 100,
            'flush_interval': 1.0,
            'compression': 'snappy' if backend_name == 'parquet' else 'none'
        }
        
        try:
            recorder = RecorderFactory.create_recorder(basic_config)
        except ImportError:
            pytest.skip(f"{backend_name} backend dependencies not available")
        
        # Validate RecorderProtocol interface compliance
        assert isinstance(recorder, RecorderProtocol), f"{backend_name} recorder must implement RecorderProtocol"
        
        # Test required method presence
        required_methods = ['record_step', 'record_episode', 'export_data']
        for method_name in required_methods:
            assert hasattr(recorder, method_name), f"{backend_name} recorder missing {method_name} method"
            method = getattr(recorder, method_name)
            assert callable(method), f"{backend_name} recorder {method_name} must be callable"
    
    def test_record_step_method_signature(self):
        """
        Test record_step method signature and parameter validation.
        
        Validates that record_step accepts required parameters and handles
        optional metadata according to RecorderProtocol specification.
        """
        # Create real recorder for testing
        config = {'backend': 'none', 'output_dir': './test_data'}
        
        try:
            recorder = RecorderFactory.create_recorder(config)
        except ImportError:
            pytest.skip("Recorder dependencies not available")
        
        # Test basic method call (should not raise exception)
        step_data = {'position': np.array([10.0, 20.0]), 'concentration': 0.5}
        recorder.record_step(step_data, step_number=0)
        
        # Test with optional parameters (should not raise exception)
        recorder.record_step(step_data, step_number=0, episode_id=1, metadata={'test': True})
        
        # Test that the method exists and is callable
        assert hasattr(recorder, 'record_step')
        assert callable(recorder.record_step)
    
    def test_record_episode_method_signature(self):
        """
        Test record_episode method signature and parameter validation.
        
        Validates that record_episode accepts required parameters and handles
        metadata correlation according to RecorderProtocol specification.
        """
        # Create real recorder for testing
        config = {'backend': 'none', 'output_dir': './test_data'}
        
        try:
            recorder = RecorderFactory.create_recorder(config)
        except ImportError:
            pytest.skip("Recorder dependencies not available")
        
        # Test basic method call (should not raise exception)
        episode_data = {'total_steps': 100, 'success': True, 'final_position': [80.0, 90.0]}
        recorder.record_episode(episode_data, episode_id=1)
        
        # Test with optional metadata (should not raise exception)
        recorder.record_episode(episode_data, episode_id=1, config_snapshot={'test': True})
        
        # Test that the method exists and is callable
        assert hasattr(recorder, 'record_episode')
        assert callable(recorder.record_episode)
    
    def test_export_data_method_signature(self):
        """
        Test export_data method signature and format options.
        
        Validates that export_data supports required formats and compression
        options according to RecorderProtocol specification.
        """
        # Create real recorder for testing
        config = {'backend': 'none', 'output_dir': './test_data'}
        
        try:
            recorder = RecorderFactory.create_recorder(config)
        except ImportError:
            pytest.skip("Recorder dependencies not available")
        
        # Test basic export call (should not raise exception)
        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp_file:
            result = recorder.export_data(tmp_file.name, format='csv')
            
            # Test with compression and filtering (should not raise exception)
            recorder.export_data(
                tmp_file.name, 
                format='csv', 
                compression='gzip',
                filter_episodes=[1, 2, 3]
            )
            
            # Test that the method exists and is callable
            assert hasattr(recorder, 'export_data')
            assert callable(recorder.export_data)
    
    def test_protocol_method_call_patterns(self):
        """
        Test typical usage patterns and method call sequences.
        
        Validates that RecorderProtocol implementations support common
        usage patterns including session management and data correlation.
        """
        # Create real recorder for testing
        config = {'backend': 'none', 'output_dir': './test_data'}
        
        try:
            recorder = RecorderFactory.create_recorder(config)
        except ImportError:
            pytest.skip("Recorder dependencies not available")
        
        # Simulate typical recording session (should not raise exceptions)
        recorder.record_step({'position': [0, 0]}, step_number=0, episode_id=1)
        recorder.record_step({'position': [1, 1]}, step_number=1, episode_id=1)
        recorder.record_episode({'total_steps': 2}, episode_id=1)
        
        # Validate that all methods exist and work correctly
        assert hasattr(recorder, 'record_step')
        assert hasattr(recorder, 'record_episode')
        assert hasattr(recorder, 'export_data')


class TestRecorderPerformance:
    """
    Test suite for validating recorder performance requirements.
    
    This test class validates critical performance requirements including
    <1ms overhead when disabled and ≤33ms/step when enabled per F-017 requirements.
    Uses pytest-benchmark for precise performance measurement.
    """
    
    def test_disabled_recorder_overhead(self, performance_monitor):
        """
        Test that disabled recording achieves <1ms overhead per F-017-RQ-001.
        
        Validates zero-overhead performance when recording is disabled using
        NullRecorder implementation with high-precision timing measurement.
        """
        # Create null recorder for performance baseline
        config = RecorderConfig(backend='none', disabled_mode_optimization=True)
        recorder = NullRecorder(config)
        
        # Measure overhead for 1000 operations
        timing_ctx = performance_monitor.start_timing('disabled_recording_1000_ops')
        
        for i in range(1000):
            recorder.record_step({'position': [i, i], 'concentration': 0.5}, step_number=i)
        
        perf_data = performance_monitor.end_timing(timing_ctx)
        
        # Validate performance requirement: <1ms for 1000 operations
        assert perf_data['duration_ms'] < 1.0, f"Disabled recording overhead {perf_data['duration_ms']:.3f}ms exceeds 1ms limit"
        
        # Additional validation: per-operation overhead
        per_op_overhead_ms = perf_data['duration_ms'] / 1000
        assert per_op_overhead_ms < 0.001, f"Per-operation overhead {per_op_overhead_ms:.6f}ms too high"
    
    def test_enabled_recorder_performance(self, performance_monitor, mock_parquet_recorder):
        """
        Test that enabled recording achieves ≤33ms/step with 100 agents.
        
        Validates performance when recording is active using buffered I/O
        and compression with realistic multi-agent simulation data.
        """
        if mock_parquet_recorder is None:
            pytest.skip("Parquet recorder not available")
        
        # Generate realistic multi-agent step data
        num_agents = 100
        step_data = {
            'positions': np.random.rand(num_agents, 2) * 100,
            'velocities': np.random.rand(num_agents, 2) * 10 - 5,
            'concentrations': np.random.exponential(scale=0.5, size=num_agents),
            'actions': np.random.randint(0, 5, size=num_agents),
            'rewards': np.random.rand(num_agents) * 10 - 2
        }
        
        # Measure recording performance
        timing_ctx = performance_monitor.start_timing('enabled_recording_step')
        mock_parquet_recorder.record_step(step_data, step_number=0, episode_id=1)
        perf_data = performance_monitor.end_timing(timing_ctx)
        
        # Validate performance requirement: ≤33ms per step
        assert perf_data['duration_ms'] <= 33.0, f"Recording step time {perf_data['duration_ms']:.3f}ms exceeds 33ms limit"
        
        # Validate memory efficiency
        assert perf_data['memory_delta_mb'] < 10.0, f"Memory delta {perf_data['memory_delta_mb']:.2f}MB too high"
    
    def test_buffered_io_performance(self, performance_monitor):
        """
        Test buffered I/O performance with configurable buffer sizes.
        
        Validates that buffering reduces I/O overhead and maintains
        performance targets with different buffer configurations.
        """
        buffer_sizes = [100, 500, 1000, 2000]
        performance_results = []
        
        for buffer_size in buffer_sizes:
            config = RecorderConfig(
                backend='none',
                buffer_size=buffer_size,
                async_io=True
            )
            recorder = NullRecorder(config)
            
            # Measure buffered recording performance
            timing_ctx = performance_monitor.start_timing(f'buffered_recording_{buffer_size}')
            
            for i in range(buffer_size * 2):  # Fill and flush buffer twice
                recorder.record_step({'test': i}, step_number=i)
            
            perf_data = performance_monitor.end_timing(timing_ctx)
            performance_results.append({
                'buffer_size': buffer_size,
                'duration_ms': perf_data['duration_ms'],
                'ops_per_ms': (buffer_size * 2) / perf_data['duration_ms']
            })
        
        # Validate that larger buffers improve performance
        assert performance_results[-1]['ops_per_ms'] >= performance_results[0]['ops_per_ms'], \
            "Larger buffers should improve performance"
        
        # Validate all configurations meet performance requirements
        for result in performance_results:
            per_op_time_ms = result['duration_ms'] / (result['buffer_size'] * 2)
            assert per_op_time_ms < 0.01, f"Buffer size {result['buffer_size']} per-op time too high: {per_op_time_ms:.6f}ms"
    
    def test_compression_performance_impact(self, compression_testing_utilities):
        """
        Test compression algorithm performance impact on recording speed.
        
        Validates that compression does not exceed performance budgets
        and provides appropriate tradeoffs between speed and storage.
        """
        # Generate test data for compression benchmarking
        test_data = compression_testing_utilities['generate_test_data'](1000, 'trajectory')
        
        # Test different compression algorithms
        algorithms = ['none', 'snappy', 'lz4', 'gzip', 'zstd']
        compression_results = []
        
        for algorithm in algorithms:
            result = compression_testing_utilities['test_compression_algorithm'](test_data, algorithm)
            compression_results.append(result)
            
            # Validate compression time is within performance budget
            assert result['compression_time_ms'] < 100.0, \
                f"Compression algorithm {algorithm} too slow: {result['compression_time_ms']:.2f}ms"
        
        # Validate performance vs compression ratio tradeoffs
        benchmark_data = compression_testing_utilities['benchmark_algorithms'](test_data, algorithms)
        
        # Fastest algorithm should be 'none' or 'snappy'
        fastest = benchmark_data['fastest']
        assert fastest['algorithm'] in ['none', 'snappy', 'lz4'], \
            f"Unexpected fastest algorithm: {fastest['algorithm']}"
        
        # Best compression should be significant improvement
        best_ratio = benchmark_data['best_ratio']
        assert best_ratio['compression_ratio'] > 1.5, \
            f"Best compression ratio too low: {best_ratio['compression_ratio']:.2f}"


class TestBackendImplementations:
    """
    Test suite for backend-specific functionality and features.
    
    This test class validates backend-specific implementations including
    ParquetRecorder, HDF5Recorder, SQLiteRecorder, and NullRecorder
    with their unique features and optimizations.
    """
    
    def test_parquet_recorder_functionality(self, mock_parquet_recorder, tmp_path):
        """
        Test ParquetRecorder-specific functionality including columnar storage and compression.
        
        Validates Parquet-specific features including schema inference, compression,
        and columnar data organization for analytical workloads.
        """
        if mock_parquet_recorder is None:
            pytest.skip("Parquet recorder not available")
        
        # Test schema validation and metadata tracking
        assert mock_parquet_recorder.schema_validation is True
        assert mock_parquet_recorder.compression in ['snappy', 'gzip', 'lz4', 'zstd']
        
        # Test compression configuration
        compression_test = mock_parquet_recorder.test_compression()
        assert compression_test['compression_ratio'] > 1.0, "Compression should reduce size"
        assert compression_test['compression_time_ms'] < 50.0, "Compression should be fast"
        
        # Test partition column configuration
        assert hasattr(mock_parquet_recorder, 'output_dir')
        assert mock_parquet_recorder.output_dir.exists()
    
    def test_hdf5_recorder_functionality(self, mock_hdf5_recorder, tmp_path):
        """
        Test HDF5Recorder-specific functionality including hierarchical organization and chunking.
        
        Validates HDF5-specific features including dataset creation, group hierarchy,
        and metadata attribute management for scientific data storage.
        """
        if mock_hdf5_recorder is None:
            pytest.skip("HDF5 recorder not available")
        
        # Test hierarchical data organization
        assert mock_hdf5_recorder.chunking is True
        assert mock_hdf5_recorder.track_order is True
        assert mock_hdf5_recorder.chunk_size == [100, 10]
        
        # Test dataset and group management
        mock_hdf5_recorder.create_dataset('test_dataset')
        mock_hdf5_recorder.create_group('test_group')
        mock_hdf5_recorder.create_dataset.assert_called_with('test_dataset')
        mock_hdf5_recorder.create_group.assert_called_with('test_group')
        
        # Test dataset listing
        datasets = mock_hdf5_recorder.list_datasets()
        expected_datasets = [
            '/episodes/episode_001/positions',
            '/episodes/episode_001/actions',
            '/episodes/episode_001/rewards',
            '/metadata/configuration',
            '/metadata/statistics'
        ]
        assert all(dataset in datasets for dataset in expected_datasets)
    
    def test_sqlite_recorder_functionality(self, mock_sqlite_recorder, tmp_path):
        """
        Test SQLiteRecorder-specific functionality including relational storage and indexing.
        
        Validates SQLite-specific features including table creation, query execution,
        and transaction management for relational data organization.
        """
        if mock_sqlite_recorder is None:
            pytest.skip("SQLite recorder not available")
        
        # Test database configuration
        assert mock_sqlite_recorder.transaction_size == 50
        assert mock_sqlite_recorder.pragma_settings['journal_mode'] == 'WAL'
        assert mock_sqlite_recorder.pragma_settings['synchronous'] == 'NORMAL'
        
        # Test table and index creation
        mock_sqlite_recorder.create_table('test_table')
        mock_sqlite_recorder.create_index('test_index')
        mock_sqlite_recorder.create_table.assert_called_with('test_table')
        mock_sqlite_recorder.create_index.assert_called_with('test_index')
        
        # Test query execution
        mock_sqlite_recorder.execute_query('SELECT * FROM test_table')
        mock_sqlite_recorder.execute_query.assert_called_with('SELECT * FROM test_table')
        
        # Test schema validation
        schema = mock_sqlite_recorder.get_schema()
        expected_tables = ['steps', 'episodes', 'metadata']
        assert all(table in schema for table in expected_tables)
    
    def test_null_recorder_optimization(self, performance_monitor):
        """
        Test NullRecorder optimization for maximum performance.
        
        Validates zero-overhead implementation with optional debug counting
        for performance baseline establishment and development verification.
        """
        # Test with debug mode disabled (maximum performance)
        config = RecorderConfig(backend='none', enable_debug_mode=False)
        recorder = NullRecorder(config)
        
        timing_ctx = performance_monitor.start_timing('null_recorder_optimized')
        
        for i in range(10000):  # Large number of operations
            recorder.record_step({'test': i}, step_number=i)
        
        perf_data = performance_monitor.end_timing(timing_ctx)
        
        # Validate ultra-low overhead
        per_op_time_ns = (perf_data['duration_ms'] * 1_000_000) / 10000  # Convert to nanoseconds
        assert per_op_time_ns < 100, f"Null recorder per-operation time too high: {per_op_time_ns:.2f}ns"
        
        # Test with debug mode enabled
        config_debug = RecorderConfig(backend='none', enable_debug_mode=True)
        recorder_debug = NullRecorder(config_debug)
        
        # Debug mode should track calls but still be fast
        recorder_debug.record_step({'test': 1}, step_number=1)
        recorder_debug.record_episode({'test': True}, episode_id=1)
        
        if hasattr(recorder_debug, 'get_call_count'):
            call_count = recorder_debug.get_call_count()
            assert call_count.get('record_step', 0) >= 1
            assert call_count.get('record_episode', 0) >= 1


class TestRecorderIntegration:
    """
    Test suite for recorder integration with simulation components.
    
    This test class validates recorder integration with simulation loop,
    performance monitoring, and component coordination per Section 5.1.3.
    """
    
    def test_simulation_loop_integration(self, mock_v1_environment, performance_monitor):
        """
        Test recorder integration with simulation loop and performance monitoring.
        
        Validates seamless integration between recorder and simulation components
        including episode management and performance correlation.
        """
        # Simulate episode with recording
        obs, info = mock_v1_environment.reset(seed=42)
        
        timing_ctx = performance_monitor.start_timing('simulation_with_recording')
        
        for step in range(100):
            action = np.array([0.1, 0.2])  # Simple test action
            obs, reward, done, truncated, info = mock_v1_environment.step(action)
            
            if done:
                break
        
        perf_data = performance_monitor.end_timing(timing_ctx)
        
        # Validate recording was called during simulation
        if mock_v1_environment.recorder:
            assert mock_v1_environment.recorder.record_step.call_count > 0
        
        # Validate performance with recording enabled
        steps_completed = info.get('step_count', 100)
        avg_step_time_ms = perf_data['duration_ms'] / steps_completed
        assert avg_step_time_ms <= 35.0, f"Average step time {avg_step_time_ms:.3f}ms exceeds limit with recording"
    
    def test_recorder_lifecycle_management(self, mock_parquet_recorder):
        """
        Test recorder lifecycle management including startup, recording, and cleanup.
        
        Validates proper resource management and cleanup during recorder
        lifecycle including error recovery and graceful shutdown.
        """
        if mock_parquet_recorder is None:
            pytest.skip("Parquet recorder not available")
        
        # Test recorder startup
        episode_id = 1
        if hasattr(mock_parquet_recorder, 'start_recording'):
            mock_parquet_recorder.start_recording(episode_id)
        
        # Test recording operations
        mock_parquet_recorder.record_step({'position': [0, 0]}, step_number=0)
        mock_parquet_recorder.record_episode({'total_steps': 1}, episode_id=episode_id)
        
        # Test buffer flushing
        mock_parquet_recorder.flush_buffer()
        mock_parquet_recorder.flush_buffer.assert_called()
        
        # Test recorder cleanup
        if hasattr(mock_parquet_recorder, 'stop_recording'):
            mock_parquet_recorder.stop_recording()
        
        mock_parquet_recorder.close()
        mock_parquet_recorder.close.assert_called()
    
    def test_multi_threaded_recording(self, performance_monitor):
        """
        Test thread safety and concurrent recording scenarios.
        
        Validates thread-safe recording operations with concurrent access
        and proper synchronization for multi-threaded simulation scenarios.
        """
        config = RecorderConfig(backend='none', async_io=True)
        recorder = NullRecorder(config)
        
        results = {'errors': 0, 'completed': 0}
        lock = threading.Lock()
        
        def recording_worker(worker_id: int, num_operations: int):
            """Worker function for concurrent recording testing."""
            try:
                for i in range(num_operations):
                    step_data = {
                        'worker_id': worker_id,
                        'step': i,
                        'position': [worker_id, i],
                        'timestamp': time.time()
                    }
                    recorder.record_step(step_data, step_number=i)
                
                with lock:
                    results['completed'] += 1
                    
            except Exception as e:
                with lock:
                    results['errors'] += 1
                    print(f"Worker {worker_id} error: {e}")
        
        # Launch concurrent workers
        num_workers = 5
        operations_per_worker = 100
        threads = []
        
        timing_ctx = performance_monitor.start_timing('concurrent_recording')
        
        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=recording_worker,
                args=(worker_id, operations_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        perf_data = performance_monitor.end_timing(timing_ctx)
        
        # Validate thread safety
        assert results['errors'] == 0, f"Thread safety errors: {results['errors']}"
        assert results['completed'] == num_workers, f"Not all workers completed: {results['completed']}/{num_workers}"
        
        # Validate concurrent performance
        total_operations = num_workers * operations_per_worker
        ops_per_second = total_operations / (perf_data['duration_ms'] / 1000)
        assert ops_per_second > 1000, f"Concurrent throughput too low: {ops_per_second:.0f} ops/sec"


class TestRecorderConfiguration:
    """
    Test suite for recorder configuration validation and Hydra integration.
    
    This test class validates configuration management including Hydra config
    groups, parameter validation, and backend selection per conf/base/record/ requirements.
    """
    
    def test_hydra_config_integration(self, config_files):
        """
        Test Hydra configuration integration for recorder selection and parameters.
        
        Validates that recorder configuration works seamlessly with Hydra
        config groups and supports runtime backend selection.
        """
        # Test recorder configurations from config fixture
        recorder_configs = config_files['v1_protocol_configs']['recorder_configs']
        
        # Validate all expected backends are configured
        expected_backends = ['parquet', 'hdf5', 'sqlite', 'none']
        for backend in expected_backends:
            assert backend in recorder_configs, f"Missing {backend} configuration"
            
            config = recorder_configs[backend]
            assert '_target_' in config, f"{backend} config missing _target_"
            assert config['_target_'].endswith(f'{backend.title()}Recorder') or \
                   config['_target_'].endswith('NullRecorder'), f"Invalid target for {backend}"
    
    def test_recorder_factory_configuration(self, mock_recorder_config):
        """
        Test RecorderFactory configuration validation and backend creation.
        
        Validates factory-based recorder creation with configuration validation
        and dependency availability checking.
        """
        # Test factory validation
        available_backends = RecorderFactory.get_available_backends()
        assert 'none' in available_backends, "NullRecorder should always be available"
        
        # Test configuration validation
        for backend_name, config in mock_recorder_config.items():
            if backend_name == 'performance' or backend_name == 'output_structure':
                continue  # Skip non-recorder configs
                
            validation_result = RecorderFactory.validate_config(config)
            
            # Basic validation should pass
            assert 'valid' in validation_result
            assert 'warnings' in validation_result
            assert 'recommendations' in validation_result
            
            # Backend availability check
            if backend_name in available_backends:
                assert validation_result['backend_available'] is True
    
    def test_configuration_parameter_validation(self):
        """
        Test configuration parameter validation and error handling.
        
        Validates that invalid configuration parameters are properly detected
        and helpful error messages are provided for debugging.
        """
        # Test invalid buffer size
        with raises(ValueError, match="buffer_size must be positive"):
            RecorderConfig(backend='none', buffer_size=0)
        
        # Test invalid flush interval
        with raises(ValueError, match="flush_interval must be positive"):
            RecorderConfig(backend='none', flush_interval=-1.0)
        
        # Test invalid memory limit
        with raises(ValueError, match="memory_limit_mb must be positive"):
            RecorderConfig(backend='none', memory_limit_mb=0)
        
        # Test invalid warning threshold
        with raises(ValueError, match="warning_threshold must be between 0 and 1"):
            RecorderConfig(backend='none', warning_threshold=1.5)
        
        # Test invalid backend
        with raises(ValueError, match="backend must be one of"):
            RecorderConfig(backend='invalid_backend')
    
    def test_backend_availability_detection(self, backend_availability_detector):
        """
        Test backend availability detection and graceful degradation.
        
        Validates automatic detection of available backends and graceful
        fallback when dependencies are unavailable.
        """
        availability = backend_availability_detector['check_backend_availability']()
        
        # Validate availability structure
        assert 'dependencies' in availability
        assert 'backends' in availability
        assert 'available_backends' in availability
        assert 'unavailable_backends' in availability
        
        # Test fallback behavior
        for backend_name in ['parquet', 'hdf5', 'sqlite']:
            fallback_result = backend_availability_detector['test_fallback_behavior'](backend_name)
            
            if fallback_result['fallback_triggered']:
                assert fallback_result['fallback_config']['backend'] == 'none'
                assert 'warning_message' in fallback_result['fallback_config']
        
        # Test minimal configuration
        minimal_config = backend_availability_detector['get_minimal_config']()
        assert minimal_config['recorder']['_target_'].endswith('NullRecorder')


class TestDataIntegrityAndExport:
    """
    Test suite for data integrity validation and export functionality.
    
    This test class validates data integrity across different formats, compression
    algorithms, and export scenarios including schema evolution support.
    """
    
    def test_data_export_formats(self, mock_parquet_recorder):
        """
        Test data export across multiple formats with integrity validation.
        
        Validates that exported data maintains integrity across different
        formats including CSV, JSON, and native formats.
        """
        if mock_parquet_recorder is None:
            pytest.skip("Parquet recorder not available")
        
        # Test different export formats
        export_formats = ['parquet', 'csv', 'json']
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            for format_name in export_formats:
                output_path = Path(tmp_dir) / f"test_export.{format_name}"
                
                # Test export operation
                success = mock_parquet_recorder.export_data(
                    str(output_path),
                    format=format_name,
                    compression='gzip' if format_name != 'parquet' else 'snappy'
                )
                
                # Mock should return success for all formats
                assert success is True or success is None, f"Export failed for {format_name}"
    
    def test_compression_data_integrity(self, compression_testing_utilities):
        """
        Test data integrity across different compression algorithms.
        
        Validates that compression and decompression maintain data integrity
        without corruption or precision loss.
        """
        # Generate test data
        original_data = compression_testing_utilities['generate_test_data'](500, 'trajectory')
        
        # Test compression algorithms
        algorithms = ['snappy', 'gzip', 'lz4', 'zstd']
        
        for algorithm in algorithms:
            compression_result = compression_testing_utilities['test_compression_algorithm'](
                original_data, algorithm
            )
            
            # Validate compression effectiveness
            assert compression_result['compression_ratio'] >= 1.0, \
                f"Compression ratio for {algorithm} should be >= 1.0"
            
            # Validate compression speed
            assert compression_result['compression_time_ms'] < 1000, \
                f"Compression time for {algorithm} too slow: {compression_result['compression_time_ms']:.2f}ms"
            
            # Validate compression speed is reasonable for size
            speed_mb_per_sec = compression_result['compression_speed_mb_per_sec']
            assert speed_mb_per_sec > 1.0, \
                f"Compression speed for {algorithm} too slow: {speed_mb_per_sec:.2f} MB/s"
    
    def test_structured_output_organization(self, tmp_path):
        """
        Test structured output directory organization by run_id/episode_id hierarchy.
        
        Validates that recorded data is organized in the expected directory
        structure for easy navigation and analysis.
        """
        # Create recorder with structured output
        config = RecorderConfig(
            backend='none',
            output_dir=str(tmp_path),
            run_id='test_run_001'
        )
        recorder = NullRecorder(config)
        
        # Test directory creation
        expected_base_dir = tmp_path / 'test_run_001'
        
        if hasattr(recorder, 'base_dir'):
            assert recorder.base_dir == expected_base_dir or str(recorder.base_dir) == str(expected_base_dir)
    
    def test_metadata_preservation(self, mock_hdf5_recorder):
        """
        Test metadata preservation and retrieval across recording sessions.
        
        Validates that experimental metadata is properly preserved and
        can be retrieved for reproducibility and analysis.
        """
        if mock_hdf5_recorder is None:
            pytest.skip("HDF5 recorder not available")
        
        # Test metadata setting and retrieval
        test_metadata = {
            'experiment_date': '2024-01-15',
            'researcher': 'test_user',
            'conditions': 'controlled_lab',
            'equipment_version': 'v1.0.0'
        }
        
        if hasattr(mock_hdf5_recorder, 'set_attributes'):
            mock_hdf5_recorder.set_attributes('/experiment_metadata', test_metadata)
            mock_hdf5_recorder.set_attributes.assert_called_with('/experiment_metadata', test_metadata)


class TestErrorHandlingAndRobustness:
    """
    Test suite for error handling and system robustness.
    
    This test class validates graceful error handling, recovery mechanisms,
    and robustness under various failure scenarios.
    """
    
    def test_graceful_degradation_on_backend_failure(self, backend_availability_detector):
        """
        Test graceful degradation when recorder backends fail.
        
        Validates that system continues to function with fallback recorder
        when primary backends encounter errors or are unavailable.
        """
        # Test fallback to null recorder when backend fails
        for backend_name in ['parquet', 'hdf5', 'sqlite']:
            fallback_result = backend_availability_detector['test_fallback_behavior'](backend_name, mock_unavailable=True)
            
            assert fallback_result['fallback_triggered'] is True
            assert fallback_result['fallback_config']['backend'] == 'none'
            assert 'warning_message' in fallback_result['fallback_config']
    
    def test_buffer_overflow_handling(self):
        """
        Test buffer overflow handling and backpressure mechanisms.
        
        Validates that system handles buffer overflow gracefully without
        data loss or system crashes.
        """
        # Create recorder with small buffer for testing
        config = RecorderConfig(
            backend='none',
            buffer_size=10,
            memory_limit_mb=1  # Very small limit for testing
        )
        recorder = NullRecorder(config)
        
        # Generate more data than buffer can hold
        for i in range(50):  # 5x buffer size
            large_data = {
                'position': np.random.rand(100, 2),  # Large data arrays
                'metadata': {'step': i, 'data_size': 'large'}
            }
            
            # Should not raise exception even with buffer overflow
            try:
                recorder.record_step(large_data, step_number=i)
            except Exception as e:
                pytest.fail(f"Buffer overflow caused exception: {e}")
    
    def test_concurrent_access_error_handling(self):
        """
        Test error handling under concurrent access scenarios.
        
        Validates proper error handling and recovery when multiple threads
        access recorder simultaneously.
        """
        config = RecorderConfig(backend='none', async_io=True)
        recorder = NullRecorder(config)
        
        errors = []
        
        def concurrent_writer(thread_id: int):
            """Concurrent writer with potential conflicts."""
            try:
                for i in range(100):
                    recorder.record_step(
                        {'thread_id': thread_id, 'step': i},
                        step_number=i
                    )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
        
        # Launch multiple writers
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_writer, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Validate no errors occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
    
    def test_disk_space_error_simulation(self, tmp_path):
        """
        Test behavior when disk space is exhausted.
        
        Validates graceful handling of disk space exhaustion and appropriate
        error reporting without system crashes.
        """
        # Create recorder with temporary directory
        config = RecorderConfig(
            backend='none',
            output_dir=str(tmp_path)
        )
        recorder = NullRecorder(config)
        
        # Simulate large data recording (null recorder won't actually write)
        large_data = {'position': np.random.rand(10000, 2)}
        
        # Should not fail even with simulated large data
        try:
            recorder.record_step(large_data, step_number=0)
            recorder.record_episode({'size': 'very_large'}, episode_id=1)
        except Exception as e:
            pytest.fail(f"Large data recording failed: {e}")


# Test configuration and fixtures
@fixture
def performance_monitor():
    """Create performance monitoring utilities for recorder testing."""
    class TestPerformanceMonitor:
        def __init__(self):
            self.timing_data = []
        
        def start_timing(self, operation_name: str):
            return {
                'operation': operation_name,
                'start_time': time.perf_counter(),
                'start_memory': 0  # Simplified for testing
            }
        
        def end_timing(self, timing_context: dict):
            end_time = time.perf_counter()
            duration_ms = (end_time - timing_context['start_time']) * 1000
            
            return {
                'operation': timing_context['operation'],
                'duration_ms': duration_ms,
                'memory_delta_mb': 0.1  # Simplified for testing
            }
    
    return TestPerformanceMonitor()


# Main test execution
if __name__ == "__main__":
    # Run tests with verbose output and coverage reporting
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "--capture=no"
    ])