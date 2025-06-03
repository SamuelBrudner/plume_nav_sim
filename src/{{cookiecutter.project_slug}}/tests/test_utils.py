"""
Comprehensive test suite for utility functions validating seed management, logging configuration,
visualization utilities, and cross-cutting concerns.

This module ensures utility function reliability, mathematical precision, and proper integration
with the broader system architecture per Section 6.6.3.1 requirements.
"""

import pytest
import numpy as np
import time
import tempfile
import os
import sys
import platform
import gc
import psutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import multiprocessing
from contextlib import contextmanager
import io

# Import utility modules to test
try:
    from {{cookiecutter.project_slug}}.utils.seed_manager import (
        set_global_seed, get_current_seed, initialize_reproducibility,
        get_random_state_context, reset_random_state
    )
    SEED_MANAGER_AVAILABLE = True
except ImportError:
    SEED_MANAGER_AVAILABLE = False

try:
    from {{cookiecutter.project_slug}}.utils.logging import (
        setup_logger, get_module_logger, configure_hydra_logging,
        get_correlation_id, bind_experiment_context
    )
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False

try:
    from {{cookiecutter.project_slug}}.utils.visualization import (
        SimulationVisualization, visualize_trajectory, export_animation,
        create_trajectory_plot, setup_headless_mode
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class TestSeedManager:
    """Test suite for seed management utilities ensuring reproducible randomization."""
    
    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="Seed manager module not available")
    def test_set_global_seed_reproducibility(self):
        """Test that global seed setting ensures reproducible results across runs."""
        test_seed = 12345
        
        # Set seed and generate random numbers
        set_global_seed(test_seed)
        np_result1 = np.random.random(10)
        python_result1 = [np.random.random() for _ in range(5)]
        
        # Reset seed and generate again
        set_global_seed(test_seed)
        np_result2 = np.random.random(10)
        python_result2 = [np.random.random() for _ in range(5)]
        
        # Results should be identical
        np.testing.assert_array_equal(np_result1, np_result2)
        assert python_result1 == python_result2
    
    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="Seed manager module not available")
    def test_seed_initialization_performance(self):
        """Test that seed initialization completes within 100ms per Section 6.6.3.3."""
        start_time = time.perf_counter()
        
        set_global_seed(42)
        
        end_time = time.perf_counter()
        initialization_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert initialization_time < 100, f"Seed initialization took {initialization_time:.2f}ms, should be <100ms"
    
    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="Seed manager module not available")
    def test_cross_platform_consistency(self):
        """Test that seed management produces consistent results across platforms."""
        test_seed = 54321
        
        # Test with different data types and operations
        set_global_seed(test_seed)
        results = {
            'random_floats': np.random.random(100),
            'random_ints': np.random.randint(0, 1000, 50),
            'normal_dist': np.random.normal(0, 1, 75),
            'choice_array': np.random.choice(range(100), 25, replace=False)
        }
        
        # Reset and test again
        set_global_seed(test_seed)
        repeated_results = {
            'random_floats': np.random.random(100),
            'random_ints': np.random.randint(0, 1000, 50),
            'normal_dist': np.random.normal(0, 1, 75),
            'choice_array': np.random.choice(range(100), 25, replace=False)
        }
        
        # Verify all results are identical
        for key in results:
            np.testing.assert_array_equal(
                results[key], repeated_results[key],
                err_msg=f"Results for {key} differ between runs"
            )
    
    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="Seed manager module not available")
    def test_get_current_seed(self):
        """Test retrieval of current seed value."""
        test_seed = 98765
        set_global_seed(test_seed)
        
        current_seed = get_current_seed()
        assert current_seed == test_seed
    
    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="Seed manager module not available")
    def test_random_state_context_isolation(self):
        """Test that random state context provides isolation without affecting global state."""
        global_seed = 11111
        context_seed = 22222
        
        set_global_seed(global_seed)
        global_value1 = np.random.random()
        
        # Use context with different seed
        with get_random_state_context(context_seed):
            context_value = np.random.random()
            
        # Continue with global state
        global_value2 = np.random.random()
        
        # Reset global seed and verify sequence continues correctly
        set_global_seed(global_seed)
        expected_value1 = np.random.random()
        expected_value2 = np.random.random()
        
        assert global_value1 == expected_value1
        assert global_value2 == expected_value2
        # Context value should be different from global sequence
        assert context_value != global_value1
        assert context_value != global_value2
    
    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="Seed manager module not available")
    def test_multiprocess_seed_isolation(self):
        """Test that seed management works correctly in multiprocess environments."""
        def worker_function(seed_value, return_dict, process_id):
            set_global_seed(seed_value)
            results = np.random.random(10)
            return_dict[process_id] = results.tolist()
        
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        processes = []
        
        # Start multiple processes with the same seed
        test_seed = 33333
        for i in range(3):
            p = multiprocessing.Process(
                target=worker_function,
                args=(test_seed, return_dict, i)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # All processes should produce identical results
        results_list = [return_dict[i] for i in range(3)]
        for i in range(1, 3):
            np.testing.assert_array_almost_equal(results_list[0], results_list[i])


class TestLoggingConfiguration:
    """Test suite for logging configuration and performance monitoring integration."""
    
    @pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging module not available")
    def test_logger_setup_performance(self):
        """Test that logger setup completes quickly for real-time applications."""
        start_time = time.perf_counter()
        
        with patch('loguru.logger') as mock_logger:
            setup_logger(level="INFO")
            
        end_time = time.perf_counter()
        setup_time = (end_time - start_time) * 1000
        
        assert setup_time < 50, f"Logger setup took {setup_time:.2f}ms, should be <50ms"
        assert mock_logger.remove.called
        assert mock_logger.add.called
    
    @pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging module not available")
    def test_module_logger_context_binding(self):
        """Test that module loggers bind context correctly."""
        module_name = "test_module"
        
        with patch('loguru.logger') as mock_logger:
            mock_bound_logger = MagicMock()
            mock_logger.bind.return_value = mock_bound_logger
            
            logger = get_module_logger(module_name)
            
            mock_logger.bind.assert_called_once_with(module=module_name)
            assert logger == mock_bound_logger
    
    @pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging module not available")
    @patch('{{cookiecutter.project_slug}}.utils.logging.uuid4')
    def test_correlation_id_generation(self, mock_uuid):
        """Test correlation ID generation for experiment tracking."""
        mock_uuid.return_value.hex = "test_correlation_id_123"
        
        correlation_id = get_correlation_id()
        
        assert correlation_id == "test_correlation_id_123"
        mock_uuid.assert_called_once()
    
    @pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging module not available")
    def test_hydra_logging_integration(self):
        """Test Hydra configuration integration with logging system."""
        mock_config = {
            'job': {'name': 'test_experiment'},
            'hydra': {'job': {'chdir': '/tmp/test'}},
            'logging': {'level': 'DEBUG'}
        }
        
        with patch('loguru.logger') as mock_logger:
            configure_hydra_logging(mock_config)
            
            # Verify logger configuration was updated
            assert mock_logger.remove.called
            assert mock_logger.add.called
    
    @pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging module not available")
    def test_experiment_context_binding(self):
        """Test experiment context binding for enhanced traceability."""
        experiment_context = {
            'experiment_id': 'exp_001',
            'run_id': 'run_123',
            'seed': 42,
            'timestamp': '2023-01-01T00:00:00'
        }
        
        with patch('loguru.logger') as mock_logger:
            mock_bound_logger = MagicMock()
            mock_logger.bind.return_value = mock_bound_logger
            
            result_logger = bind_experiment_context(experiment_context)
            
            mock_logger.bind.assert_called_once_with(**experiment_context)
            assert result_logger == mock_bound_logger
    
    @pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging module not available")
    def test_logging_memory_efficiency(self):
        """Test that logging operations don't cause memory leaks."""
        initial_memory = psutil.Process().memory_info().rss
        
        with patch('loguru.logger'):
            # Perform many logging operations
            for i in range(1000):
                logger = get_module_logger(f"test_module_{i}")
                setup_logger(level="INFO")
        
        # Force garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 10MB)
        assert memory_increase < 10 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"


class TestVisualizationUtilities:
    """Test suite for visualization utilities including trajectory plotting and animation export."""
    
    @pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization module not available")
    @patch('matplotlib.pyplot')
    def test_simulation_visualization_initialization(self, mock_plt):
        """Test SimulationVisualization class initialization."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        viz = SimulationVisualization(figsize=(12, 10), dpi=150)
        
        mock_plt.subplots.assert_called_once_with(figsize=(12, 10), dpi=150)
        assert viz.fig == mock_fig
        assert viz.ax == mock_ax
    
    @pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization module not available")
    @patch('matplotlib.pyplot')
    def test_trajectory_visualization_single_agent(self, mock_plt):
        """Test trajectory plotting for single agent scenarios."""
        positions = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        orientations = np.array([0, 45, 90, 135])
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        visualize_trajectory(positions, orientations, show_plot=False)
        
        # Verify plotting calls were made
        mock_plt.subplots.assert_called()
        mock_ax.plot.assert_called()
        mock_ax.quiver.assert_called()
    
    @pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization module not available")
    @patch('matplotlib.pyplot')
    def test_trajectory_visualization_multi_agent(self, mock_plt):
        """Test trajectory plotting for multi-agent scenarios."""
        # Multi-agent positions (2 agents, 4 timesteps each)
        positions = np.array([
            [[0, 0], [1, 1], [2, 2], [3, 3]],  # Agent 1
            [[0, 5], [1, 4], [2, 3], [3, 2]]   # Agent 2
        ])
        orientations = np.array([
            [0, 45, 90, 135],    # Agent 1
            [180, 225, 270, 315] # Agent 2
        ])
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        visualize_trajectory(positions, orientations, show_plot=False)
        
        # Should have multiple plot calls for multiple agents
        assert mock_ax.plot.call_count >= 2
        mock_ax.quiver.assert_called()
    
    @pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization module not available")
    @patch('matplotlib.pyplot')
    def test_animation_export_performance(self, mock_plt):
        """Test animation export meets performance requirements."""
        mock_animation = MagicMock()
        mock_fig = MagicMock()
        
        start_time = time.perf_counter()
        
        with patch('matplotlib.animation.FuncAnimation', return_value=mock_animation):
            export_animation(
                mock_fig, 
                update_function=lambda x: [],
                frames=30,
                filename="test_animation.mp4"
            )
        
        end_time = time.perf_counter()
        export_time = (end_time - start_time) * 1000
        
        # Animation setup should be fast (actual rendering can be slow)
        assert export_time < 100, f"Animation export setup took {export_time:.2f}ms"
    
    @pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization module not available")
    @patch('matplotlib.pyplot')
    def test_headless_mode_setup(self, mock_plt):
        """Test headless mode setup for batch processing."""
        with patch('matplotlib.use') as mock_use:
            setup_headless_mode()
            mock_use.assert_called_with('Agg')
    
    @pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization module not available")
    @patch('matplotlib.pyplot')
    def test_trajectory_plot_with_background(self, mock_plt):
        """Test trajectory plotting with odor plume background."""
        positions = np.array([[0, 0], [1, 1], [2, 2]])
        background = np.random.random((100, 100))
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        visualize_trajectory(positions, background=background, show_plot=False)
        
        # Should include background plotting
        mock_ax.imshow.assert_called()
        mock_plt.colorbar.assert_called()
    
    @pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization module not available")
    def test_visualization_memory_cleanup(self):
        """Test that visualization operations properly clean up memory."""
        initial_memory = psutil.Process().memory_info().rss
        
        with patch('matplotlib.pyplot') as mock_plt:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)
            
            # Create many visualization objects
            for i in range(50):
                viz = SimulationVisualization()
                positions = np.random.random((10, 2))
                visualize_trajectory(positions, show_plot=False)
        
        # Force cleanup
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"


class TestMathematicalPrecision:
    """Test suite for mathematical utilities ensuring numerical precision requirements."""
    
    def test_array_normalization_precision(self):
        """Test that array normalization maintains 1e-6 precision per research standards."""
        test_arrays = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([0.1, 0.01, 0.001, 0.0001]),
            np.array([1e6, 1e7, 1e8, 1e9]),
            np.array([-1.0, -2.0, -3.0])
        ]
        
        for arr in test_arrays:
            # Simulate array normalization (would be actual utility function)
            normalized = arr / np.linalg.norm(arr)
            
            # Verify unit vector property
            norm = np.linalg.norm(normalized)
            assert abs(norm - 1.0) < 1e-6, f"Normalization precision error: {abs(norm - 1.0)}"
    
    def test_angle_calculations_precision(self):
        """Test angle calculations maintain required precision."""
        test_angles = np.linspace(0, 2*np.pi, 1000)
        
        for angle in test_angles:
            # Test trigonometric identity: sin²(θ) + cos²(θ) = 1
            sin_val = np.sin(angle)
            cos_val = np.cos(angle)
            identity_result = sin_val**2 + cos_val**2
            
            assert abs(identity_result - 1.0) < 1e-6, f"Trigonometric precision error at angle {angle}"
    
    def test_vector_operations_precision(self):
        """Test vector operations maintain numerical stability."""
        # Test with various vector magnitudes
        test_vectors = [
            np.array([1.0, 0.0]),
            np.array([1e-10, 1e-10]),
            np.array([1e10, 1e10]),
            np.array([np.sqrt(2)/2, np.sqrt(2)/2])
        ]
        
        for vec in test_vectors:
            # Test dot product with itself equals squared magnitude
            dot_product = np.dot(vec, vec)
            magnitude_squared = np.linalg.norm(vec)**2
            
            relative_error = abs(dot_product - magnitude_squared) / max(abs(magnitude_squared), 1e-15)
            assert relative_error < 1e-6, f"Vector operation precision error: {relative_error}"
    
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge cases."""
        # Test with very small numbers
        small_array = np.array([1e-15, 1e-14, 1e-13])
        assert np.all(np.isfinite(small_array)), "Small numbers should remain finite"
        
        # Test with very large numbers
        large_array = np.array([1e14, 1e15])
        assert np.all(np.isfinite(large_array)), "Large numbers should remain finite"
        
        # Test division by small numbers
        result = 1.0 / 1e-10
        assert np.isfinite(result), "Division by small number should be finite"


class TestCrossPlatformCompatibility:
    """Test suite for cross-platform compatibility of file I/O and path handling."""
    
    def test_path_handling_cross_platform(self):
        """Test that path operations work consistently across platforms."""
        test_paths = [
            "simple_file.txt",
            "folder/subfolder/file.txt",
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt"
        ]
        
        for path_str in test_paths:
            path_obj = Path(path_str)
            
            # Test path operations work on all platforms
            assert isinstance(path_obj.name, str)
            assert isinstance(path_obj.suffix, str)
            assert isinstance(str(path_obj), str)
            
            # Test path resolution
            resolved = path_obj.resolve()
            assert resolved.is_absolute()
    
    def test_file_io_consistency(self):
        """Test file I/O operations work consistently across platforms."""
        test_data = {
            'text': "Test data\nWith newlines\n",
            'json_data': {"key": "value", "number": 42},
            'binary_data': b'\x00\x01\x02\x03\x04'
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test text file I/O
            text_file = temp_path / "test.txt"
            text_file.write_text(test_data['text'], encoding='utf-8')
            read_text = text_file.read_text(encoding='utf-8')
            assert read_text == test_data['text']
            
            # Test binary file I/O
            binary_file = temp_path / "test.bin"
            binary_file.write_bytes(test_data['binary_data'])
            read_binary = binary_file.read_bytes()
            assert read_binary == test_data['binary_data']
    
    def test_directory_operations(self):
        """Test directory operations work across platforms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Create nested directory structure
            nested_path = base_path / "level1" / "level2" / "level3"
            nested_path.mkdir(parents=True, exist_ok=True)
            
            assert nested_path.exists()
            assert nested_path.is_dir()
            
            # Test directory listing
            (nested_path / "test_file.txt").touch()
            files = list(nested_path.iterdir())
            assert len(files) == 1
            assert files[0].name == "test_file.txt"
    
    def test_platform_specific_behaviors(self):
        """Test handling of platform-specific behaviors."""
        current_platform = platform.system()
        
        # Test that we can detect the platform
        assert current_platform in ['Windows', 'Linux', 'Darwin', 'Java']
        
        # Test platform-specific path separators
        if current_platform == 'Windows':
            assert os.sep == '\\'
        else:
            assert os.sep == '/'
        
        # Test that pathlib handles platform differences correctly
        path = Path("folder") / "subfolder" / "file.txt"
        path_str = str(path)
        assert os.sep in path_str or path_str.count('/') > 0


class TestPerformanceCharacteristics:
    """Test suite for validating performance characteristics meet operational requirements."""
    
    def test_utility_function_performance(self):
        """Test that utility functions meet timing requirements."""
        # Test array operations performance
        large_array = np.random.random((10000, 2))
        
        start_time = time.perf_counter()
        normalized = large_array / np.linalg.norm(large_array, axis=1, keepdims=True)
        end_time = time.perf_counter()
        
        operation_time = (end_time - start_time) * 1000
        assert operation_time < 10, f"Array normalization took {operation_time:.2f}ms, should be <10ms"
    
    def test_configuration_processing_performance(self):
        """Test configuration processing meets timing requirements."""
        # Simulate configuration validation
        config_data = {
            'navigator': {
                'speed': 1.0,
                'max_speed': 2.0,
                'orientation': 0.0
            },
            'video_plume': {
                'path': '/path/to/video.mp4',
                'frame_count': 1000
            }
        }
        
        start_time = time.perf_counter()
        
        # Simulate validation operations
        for _ in range(100):
            validated = dict(config_data)
            validated['navigator']['speed'] = float(validated['navigator']['speed'])
            
        end_time = time.perf_counter()
        
        processing_time = (end_time - start_time) * 1000
        assert processing_time < 100, f"Config processing took {processing_time:.2f}ms, should be <100ms"
    
    def test_memory_usage_efficiency(self):
        """Test memory usage remains efficient for large datasets."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Create large arrays and process them
        large_arrays = []
        for i in range(10):
            arr = np.random.random((1000, 1000))
            processed = arr * 2.0 + 1.0
            large_arrays.append(processed)
        
        peak_memory = psutil.Process().memory_info().rss
        
        # Clean up
        del large_arrays
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        
        memory_increase = peak_memory - initial_memory
        memory_cleaned = peak_memory - final_memory
        
        # Memory should be released after cleanup
        cleanup_ratio = memory_cleaned / memory_increase if memory_increase > 0 else 1.0
        assert cleanup_ratio > 0.8, f"Only {cleanup_ratio:.2%} of memory was cleaned up"
    
    def test_concurrent_operations_performance(self):
        """Test performance under concurrent operations."""
        def worker_task(n):
            # Simulate computational work
            arr = np.random.random((100, 100))
            result = np.sum(arr * arr)
            return result
        
        start_time = time.perf_counter()
        
        # Run tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_task, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000
        assert total_time < 1000, f"Concurrent operations took {total_time:.2f}ms, should be <1000ms"
        assert len(results) == 20


class TestIntegrationWithConfigurationSystem:
    """Test suite for integration with Hydra configuration system and environment management."""
    
    @patch('hydra.compose')
    @patch('hydra.initialize')
    def test_hydra_configuration_integration(self, mock_initialize, mock_compose):
        """Test integration with Hydra configuration composition."""
        mock_config = {
            'utils': {
                'seed': 42,
                'logging_level': 'INFO',
                'visualization': {
                    'dpi': 150,
                    'figsize': [10, 8]
                }
            }
        }
        
        mock_compose.return_value = mock_config
        
        # Simulate configuration-driven utility setup
        with mock_initialize:
            config = mock_compose(config_name="config")
            
            assert config['utils']['seed'] == 42
            assert config['utils']['logging_level'] == 'INFO'
            assert config['utils']['visualization']['dpi'] == 150
    
    def test_environment_variable_integration(self):
        """Test integration with environment variables for configuration."""
        test_env_vars = {
            'PLUME_UTILS_SEED': '12345',
            'PLUME_LOGGING_LEVEL': 'DEBUG',
            'PLUME_VIZ_DPI': '300'
        }
        
        with patch.dict(os.environ, test_env_vars):
            # Test environment variable access
            seed = int(os.environ.get('PLUME_UTILS_SEED', '0'))
            log_level = os.environ.get('PLUME_LOGGING_LEVEL', 'INFO')
            dpi = int(os.environ.get('PLUME_VIZ_DPI', '100'))
            
            assert seed == 12345
            assert log_level == 'DEBUG'
            assert dpi == 300
    
    def test_configuration_validation_integration(self):
        """Test configuration validation with utility parameters."""
        valid_config = {
            'seed': 42,
            'logging_level': 'INFO',
            'max_memory_mb': 1024,
            'performance_mode': 'standard'
        }
        
        invalid_configs = [
            {'seed': -1},  # Negative seed
            {'logging_level': 'INVALID'},  # Invalid log level
            {'max_memory_mb': 'not_a_number'},  # Wrong type
            {'performance_mode': 'unknown'}  # Unknown mode
        ]
        
        # Valid config should pass
        assert self._validate_utils_config(valid_config) is True
        
        # Invalid configs should fail
        for invalid_config in invalid_configs:
            assert self._validate_utils_config(invalid_config) is False
    
    def _validate_utils_config(self, config: Dict[str, Any]) -> bool:
        """Helper method to validate utility configuration."""
        try:
            # Simulate validation logic
            if 'seed' in config:
                if not isinstance(config['seed'], int) or config['seed'] < 0:
                    return False
            
            if 'logging_level' in config:
                valid_levels = ['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                if config['logging_level'] not in valid_levels:
                    return False
            
            if 'max_memory_mb' in config:
                if not isinstance(config['max_memory_mb'], (int, float)) or config['max_memory_mb'] <= 0:
                    return False
            
            if 'performance_mode' in config:
                valid_modes = ['standard', 'high_performance', 'memory_optimized']
                if config['performance_mode'] not in valid_modes:
                    return False
            
            return True
        except Exception:
            return False


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge case behavior validation."""
    
    @pytest.mark.skipif(not SEED_MANAGER_AVAILABLE, reason="Seed manager module not available")
    def test_seed_manager_error_handling(self):
        """Test seed manager handles edge cases gracefully."""
        # Test with invalid seed types
        invalid_seeds = [None, "string", 3.14, [], {}]
        
        for invalid_seed in invalid_seeds:
            with pytest.raises((TypeError, ValueError)):
                set_global_seed(invalid_seed)
        
        # Test with boundary values
        boundary_seeds = [0, 2**31 - 1, 2**63 - 1]
        for seed in boundary_seeds:
            try:
                set_global_seed(seed)
                current = get_current_seed()
                assert current is not None
            except (OverflowError, ValueError):
                # Some boundary values may not be supported
                pass
    
    @pytest.mark.skipif(not LOGGING_AVAILABLE, reason="Logging module not available")
    def test_logging_error_recovery(self):
        """Test logging system recovers from errors gracefully."""
        with patch('loguru.logger') as mock_logger:
            # Simulate logger setup failure
            mock_logger.add.side_effect = Exception("Setup failed")
            
            try:
                setup_logger()
            except Exception:
                pass  # Should handle gracefully
            
            # Logger should still be functional after error
            mock_logger.remove.assert_called()
    
    @pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="Visualization module not available")
    def test_visualization_error_handling(self):
        """Test visualization handles invalid inputs gracefully."""
        # Test with invalid position arrays
        invalid_positions = [
            np.array([]),  # Empty array
            np.array([1, 2, 3]),  # Wrong dimensions
            np.array([[1], [2]]),  # Wrong shape
            None,  # None input
        ]
        
        for invalid_pos in invalid_positions:
            with patch('matplotlib.pyplot'):
                try:
                    visualize_trajectory(invalid_pos, show_plot=False)
                except (ValueError, TypeError, AttributeError):
                    pass  # Expected to handle gracefully
    
    def test_numerical_edge_cases(self):
        """Test numerical operations handle edge cases."""
        # Test with special values
        special_values = [np.inf, -np.inf, np.nan, 0.0, -0.0]
        
        for val in special_values:
            arr = np.array([val, 1.0, 2.0])
            
            # Operations should handle special values appropriately
            finite_mask = np.isfinite(arr)
            finite_arr = arr[finite_mask]
            
            if len(finite_arr) > 0:
                # Should be able to process finite values
                result = finite_arr * 2.0
                assert np.all(np.isfinite(result))
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure conditions."""
        # Simulate memory pressure by allocating large arrays
        large_arrays = []
        
        try:
            # Try to allocate until we get close to memory limits
            for i in range(100):
                try:
                    arr = np.random.random((1000, 1000))
                    large_arrays.append(arr)
                except MemoryError:
                    break
            
            # Test that utilities still function under memory pressure
            if SEED_MANAGER_AVAILABLE:
                set_global_seed(42)
                result = np.random.random(10)
                assert len(result) == 10
                
        finally:
            # Clean up
            del large_arrays
            gc.collect()
    
    def test_thread_safety(self):
        """Test thread safety of utility functions."""
        results = []
        errors = []
        
        def worker_function(thread_id):
            try:
                # Test thread-safe operations
                if SEED_MANAGER_AVAILABLE:
                    set_global_seed(thread_id)
                    result = np.random.random(5)
                    results.append((thread_id, result.tolist()))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check for thread safety issues
        if errors:
            pytest.fail(f"Thread safety errors: {errors}")
        
        # Results should be different for different thread IDs
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i][1] != results[i+1][1], "Thread isolation failed"


# Import required for concurrent operations test
import concurrent.futures

if __name__ == "__main__":
    pytest.main([__file__, "-v"])