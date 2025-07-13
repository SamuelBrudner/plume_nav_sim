"""
Comprehensive test module for StatsAggregatorProtocol implementations.

This module provides thorough validation of automated statistics collection via
StatsAggregatorProtocol interface, ensuring protocol compliance, performance targets,
and integration with the recording system per F-018 Statistics Aggregator requirements.

Test Coverage:
- Protocol interface compliance and method validation
- Episode-level and run-level statistics calculation accuracy
- Custom metrics definitions and calculation frameworks
- Performance validation: ≤33 ms/step with 100 agents
- Integration with Recorder system for data connectivity
- Summary.json export with standardized research metrics
- Error handling and edge case management
- Memory optimization and parallel processing capabilities

Performance Requirements:
- F-018-RQ-001: ≤33 ms/step processing time with 100 agents
- F-018-RQ-002: Memory efficiency for large dataset processing  
- F-018-RQ-003: Integration with episode completion hooks
- F-018-RQ-004: Standardized summary.json export format

Integration Testing:
- Recorder system connectivity and data source access
- Episode completion hook integration for automated collection
- Batch processing capabilities for comparative studies
- Custom statistics definitions with validation and error handling

Examples:
    Basic protocol compliance testing:
    >>> pytest tests/test_stats.py::test_stats_aggregator_protocol_compliance
    
    Performance validation:
    >>> pytest tests/test_stats.py::test_performance_33ms_target -v
    
    Integration testing:
    >>> pytest tests/test_stats.py::TestStatsAggregatorIntegration -v
"""

import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pytest

# Import the protocols and implementations under test
from src.plume_nav_sim.core.protocols import StatsAggregatorProtocol, RecorderProtocol
from src.plume_nav_sim.analysis.stats import (
    StatsAggregator, 
    StatsAggregatorConfig,
    calculate_basic_stats,
    calculate_advanced_stats,
    create_stats_aggregator,
    generate_summary_report
)


class TestStatsAggregatorProtocol:
    """Test suite for StatsAggregatorProtocol interface compliance and basic functionality."""
    
    @pytest.fixture
    def basic_config(self) -> StatsAggregatorConfig:
        """Create basic configuration for testing."""
        return StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std', 'min', 'max'],
                'concentration': ['mean', 'std', 'detection_rate'],
                'speed': ['mean', 'total_distance'],
                'timing': ['episode_length']
            },
            aggregation_levels=['episode', 'run'],
            output_format='json',
            performance_tracking=True,
            data_validation=True
        )
    
    @pytest.fixture
    def sample_episode_data(self) -> Dict[str, Any]:
        """Create sample episode data for testing."""
        np.random.seed(42)  # Deterministic test data
        trajectory = np.random.rand(100, 2) * 100  # 100 steps in 100x100 domain
        concentrations = np.random.exponential(0.1, 100)  # Realistic concentration distribution
        speeds = np.random.gamma(2, 0.5, 99)  # Speed data (one less than trajectory)
        
        return {
            'episode_id': 1,
            'trajectory': trajectory,
            'concentrations': concentrations,
            'speeds': speeds,
            'episode_length': 100,
            'step_durations': np.full(100, 0.025),  # 25ms per step
            'success': True,
            'final_position': trajectory[-1].tolist(),
            'total_reward': 15.7
        }
    
    @pytest.fixture
    def sample_episodes_list(self, sample_episode_data) -> List[Dict[str, Any]]:
        """Create list of sample episodes for run-level testing."""
        episodes = []
        for i in range(10):
            episode = sample_episode_data.copy()
            episode['episode_id'] = i
            # Add some variation
            np.random.seed(42 + i)
            episode['trajectory'] = np.random.rand(100, 2) * 100
            episode['concentrations'] = np.random.exponential(0.1, 100)
            episode['total_reward'] = np.random.normal(15.0, 3.0)
            episodes.append(episode)
        return episodes
    
    @pytest.fixture
    def stats_aggregator(self, basic_config) -> StatsAggregator:
        """Create StatsAggregator instance for testing."""
        return StatsAggregator(basic_config)
    
    @pytest.fixture
    def mock_recorder(self) -> Mock:
        """Create mock recorder for integration testing."""
        recorder = Mock(spec=RecorderProtocol)
        recorder.record_step.return_value = None
        recorder.record_episode.return_value = None
        recorder.export_data.return_value = True
        return recorder
    
    def test_stats_aggregator_protocol_compliance(self, stats_aggregator):
        """Test that StatsAggregator implements StatsAggregatorProtocol correctly."""
        # Verify protocol compliance
        assert isinstance(stats_aggregator, StatsAggregatorProtocol)
        
        # Verify required methods exist
        assert hasattr(stats_aggregator, 'calculate_episode_stats')
        assert hasattr(stats_aggregator, 'calculate_run_stats')
        assert hasattr(stats_aggregator, 'export_summary')
        
        # Verify methods are callable
        assert callable(stats_aggregator.calculate_episode_stats)
        assert callable(stats_aggregator.calculate_run_stats)
        assert callable(stats_aggregator.export_summary)
    
    def test_calculate_episode_stats_basic(self, stats_aggregator, sample_episode_data):
        """Test basic episode statistics calculation functionality."""
        # Calculate episode statistics
        stats = stats_aggregator.calculate_episode_stats(sample_episode_data, episode_id=1)
        
        # Verify return type and structure
        assert isinstance(stats, dict)
        assert 'episode_id' in stats
        assert 'timestamp' in stats
        assert 'trajectory' in stats
        assert 'concentration' in stats
        assert 'speed' in stats
        assert 'timing' in stats
        
        # Verify trajectory statistics
        traj_stats = stats['trajectory']
        assert 'mean_position' in traj_stats
        assert 'total_distance' in traj_stats
        assert 'net_displacement' in traj_stats
        assert 'displacement_efficiency' in traj_stats
        
        # Verify concentration statistics
        conc_stats = stats['concentration']
        assert 'mean' in conc_stats
        assert 'std' in conc_stats
        assert 'detection_rate' in conc_stats
        
        # Verify speed statistics
        speed_stats = stats['speed']
        assert 'mean_speed' in speed_stats
        assert 'total_distance' in speed_stats
        
        # Verify timing statistics
        timing_stats = stats['timing']
        assert 'episode_length' in timing_stats
        assert timing_stats['episode_length'] == 100
    
    def test_calculate_episode_stats_custom_metrics(self, basic_config, sample_episode_data):
        """Test episode statistics with custom metrics calculation."""
        # Define custom tortuosity metric
        def calculate_tortuosity(trajectory_data):
            """Calculate path tortuosity metric."""
            trajectory = np.array(trajectory_data['trajectory'])
            if len(trajectory) < 2:
                return float('inf')
            
            # Calculate path length
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            path_length = np.sum(distances)
            
            # Calculate direct distance
            direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            
            return path_length / direct_distance if direct_distance > 0 else float('inf')
        
        # Configure aggregator with custom calculation
        config = basic_config
        config.custom_calculations = {'tortuosity': calculate_tortuosity}
        aggregator = StatsAggregator(config)
        
        # Calculate statistics with custom metrics
        stats = aggregator.calculate_episode_stats(sample_episode_data, episode_id=1)
        
        # Verify custom metrics are included
        assert 'custom' in stats
        assert 'tortuosity' in stats['custom']
        assert isinstance(stats['custom']['tortuosity'], (int, float))
        assert stats['custom']['tortuosity'] > 1.0  # Tortuosity should be > 1
    
    def test_calculate_run_stats_basic(self, stats_aggregator, sample_episodes_list):
        """Test basic run-level statistics aggregation."""
        # Calculate run statistics
        run_stats = stats_aggregator.calculate_run_stats(sample_episodes_list, run_id='test_run')
        
        # Verify return type and structure
        assert isinstance(run_stats, dict)
        assert 'run_id' in run_stats
        assert 'episode_count' in run_stats
        assert 'timestamp' in run_stats
        
        # Verify aggregation categories
        assert 'trajectory_aggregation' in run_stats
        assert 'concentration_aggregation' in run_stats
        assert 'speed_aggregation' in run_stats
        assert 'timing_aggregation' in run_stats
        assert 'summary' in run_stats
        
        # Verify episode count is correct
        assert run_stats['episode_count'] == 10
        
        # Verify summary contains expected fields
        summary = run_stats['summary']
        assert 'total_episodes' in summary
        assert summary['total_episodes'] == 10
        assert 'average_performance' in summary
    
    def test_calculate_run_stats_correlations(self, stats_aggregator, sample_episodes_list):
        """Test cross-episode correlation analysis in run statistics."""
        # Calculate run statistics
        run_stats = stats_aggregator.calculate_run_stats(sample_episodes_list, run_id='correlation_test')
        
        # Verify correlations are computed
        assert 'correlations' in run_stats
        correlations = run_stats['correlations']
        
        # Check for expected correlation analyses
        if correlations:  # Only if sufficient data for correlations
            for corr_key, corr_data in correlations.items():
                assert 'correlation' in corr_data
                assert 'p_value' in corr_data
                assert 'significant' in corr_data
                assert isinstance(corr_data['correlation'], (int, float))
                assert isinstance(corr_data['p_value'], (int, float))
                assert isinstance(corr_data['significant'], bool)
    
    def test_export_summary_json(self, stats_aggregator, sample_episodes_list):
        """Test summary export in JSON format."""
        # Calculate some statistics first
        run_stats = stats_aggregator.calculate_run_stats(sample_episodes_list, run_id='export_test')
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / 'test_summary.json'
            
            # Export summary
            success = stats_aggregator.export_summary(str(output_path))
            
            # Verify export success
            assert success is True
            assert output_path.exists()
            
            # Verify JSON structure
            with open(output_path, 'r') as f:
                summary_data = json.load(f)
            
            # Verify required top-level fields
            assert 'metadata' in summary_data
            assert 'performance_metrics' in summary_data
            assert 'run_statistics' in summary_data
            assert 'global_summary' in summary_data
            
            # Verify metadata structure
            metadata = summary_data['metadata']
            assert 'export_timestamp' in metadata
            assert 'aggregator_version' in metadata
            assert 'format_version' in metadata
            assert 'configuration' in metadata
    
    def test_export_summary_file_creation(self, stats_aggregator):
        """Test that export creates proper file structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Test with nested directory structure
            output_path = Path(tmp_dir) / 'results' / 'experiment_001' / 'summary.json'
            
            # Export should create directories
            success = stats_aggregator.export_summary(str(output_path))
            
            assert success is True
            assert output_path.exists()
            assert output_path.parent.exists()


class TestStatsAggregatorPerformance:
    """Test suite for performance requirements and optimization validation."""
    
    @pytest.fixture
    def performance_config(self) -> StatsAggregatorConfig:
        """Create performance-optimized configuration."""
        return StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std', 'total_distance'],
                'concentration': ['mean', 'detection_rate'],
                'speed': ['mean'],
                'timing': ['episode_length']
            },
            aggregation_levels=['episode'],
            performance_tracking=True,
            parallel_processing=False,  # Single-threaded for consistent timing
            precision_mode='float64'
        )
    
    @pytest.fixture
    def large_episode_data(self) -> Dict[str, Any]:
        """Create large episode data for performance testing."""
        np.random.seed(42)
        # Simulate 100 agents with 1000 steps each
        trajectory = np.random.rand(1000, 2) * 100
        concentrations = np.random.exponential(0.1, 1000)
        speeds = np.random.gamma(2, 0.5, 999)
        
        return {
            'episode_id': 1,
            'trajectory': trajectory,
            'concentrations': concentrations,
            'speeds': speeds,
            'episode_length': 1000,
            'step_durations': np.full(1000, 0.030),  # 30ms per step
            'agent_count': 100
        }
    
    def test_performance_33ms_target(self, performance_config, large_episode_data):
        """Test that episode statistics calculation meets ≤33ms target with 100 agents."""
        aggregator = StatsAggregator(performance_config)
        
        # Measure performance over multiple runs for stability
        execution_times = []
        
        for run in range(5):
            start_time = time.perf_counter()
            
            # Calculate episode statistics
            stats = aggregator.calculate_episode_stats(large_episode_data, episode_id=run)
            
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            execution_times.append(execution_time_ms)
            
            # Verify statistics were calculated
            assert isinstance(stats, dict)
            assert 'trajectory' in stats
            assert 'concentration' in stats
        
        # Performance validation
        avg_time = np.mean(execution_times)
        max_time = np.max(execution_times)
        
        # Log performance metrics for debugging
        print(f"Average execution time: {avg_time:.2f}ms")
        print(f"Maximum execution time: {max_time:.2f}ms")
        print(f"All execution times: {execution_times}")
        
        # Verify performance requirement: ≤33ms average
        assert avg_time <= 33.0, f"Average execution time {avg_time:.2f}ms exceeds 33ms target"
        
        # Allow some tolerance for maximum time (50ms)
        assert max_time <= 50.0, f"Maximum execution time {max_time:.2f}ms exceeds tolerance"
    
    @pytest.mark.benchmark
    def test_benchmark_episode_calculation(self, benchmark, performance_config, large_episode_data):
        """Benchmark episode statistics calculation using pytest-benchmark."""
        aggregator = StatsAggregator(performance_config)
        
        # Benchmark the calculation
        result = benchmark(aggregator.calculate_episode_stats, large_episode_data, episode_id=1)
        
        # Verify result is valid
        assert isinstance(result, dict)
        assert 'trajectory' in result
        assert 'concentration' in result
    
    def test_memory_efficiency_large_dataset(self, performance_config):
        """Test memory efficiency with large datasets."""
        aggregator = StatsAggregator(performance_config)
        
        # Create very large dataset
        np.random.seed(42)
        large_trajectory = np.random.rand(10000, 2) * 100  # 10k steps
        large_concentrations = np.random.exponential(0.1, 10000)
        
        episode_data = {
            'episode_id': 1,
            'trajectory': large_trajectory,
            'concentrations': large_concentrations,
            'episode_length': 10000
        }
        
        # Monitor memory usage (simplified - in real implementation could use psutil)
        start_time = time.perf_counter()
        
        stats = aggregator.calculate_episode_stats(episode_data, episode_id=1)
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Verify calculations completed successfully
        assert isinstance(stats, dict)
        assert 'trajectory' in stats
        
        # Basic performance check (should still be reasonable for large data)
        assert execution_time_ms < 500.0, f"Large dataset processing took {execution_time_ms:.2f}ms"
    
    def test_parallel_processing_configuration(self):
        """Test that parallel processing configuration works correctly."""
        config = StatsAggregatorConfig(
            parallel_processing=True,
            performance_tracking=True
        )
        
        aggregator = StatsAggregator(config)
        
        # Verify configuration is set
        assert aggregator.config.parallel_processing is True
        
        # Note: Actual parallel processing testing would require more complex setup
        # This test verifies the configuration is accepted and stored correctly


class TestStatsAggregatorIntegration:
    """Test suite for integration with recorder system and episode completion hooks."""
    
    @pytest.fixture
    def integration_config(self) -> StatsAggregatorConfig:
        """Create configuration for integration testing."""
        return StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'total_distance'],
                'concentration': ['mean', 'detection_rate'],
                'timing': ['episode_length']
            },
            aggregation_levels=['episode', 'run'],
            output_format='json',
            data_validation=True
        )
    
    @pytest.fixture
    def mock_recorder_with_data(self) -> Mock:
        """Create mock recorder with realistic data responses."""
        recorder = Mock(spec=RecorderProtocol)
        
        # Mock data responses
        def mock_record_step(step_data, step_number, episode_id=None, **metadata):
            return None
        
        def mock_record_episode(episode_data, episode_id, **metadata):
            return None
        
        def mock_export_data(output_path, format='parquet', compression=None, 
                           filter_episodes=None, **export_options):
            return True
        
        recorder.record_step.side_effect = mock_record_step
        recorder.record_episode.side_effect = mock_record_episode
        recorder.export_data.side_effect = mock_export_data
        
        return recorder
    
    def test_recorder_integration_basic(self, integration_config, mock_recorder_with_data):
        """Test basic integration with recorder system."""
        # Create aggregator with recorder
        aggregator = StatsAggregator(integration_config, recorder=mock_recorder_with_data)
        
        # Verify recorder is attached
        assert aggregator.recorder is mock_recorder_with_data
        
        # Test that aggregator can interact with recorder
        episode_data = {
            'episode_id': 1,
            'trajectory': np.random.rand(50, 2),
            'concentrations': np.random.rand(50),
            'episode_length': 50
        }
        
        # Calculate statistics (should not fail with recorder attached)
        stats = aggregator.calculate_episode_stats(episode_data, episode_id=1)
        
        assert isinstance(stats, dict)
        assert 'trajectory' in stats
    
    def test_recorder_data_connectivity(self, integration_config, mock_recorder_with_data):
        """Test statistics aggregator data source connectivity with recorder."""
        aggregator = StatsAggregator(integration_config, recorder=mock_recorder_with_data)
        
        # Simulate recorder providing data for statistics
        simulated_recorded_data = {
            'step_data': [
                {'position': [10, 20], 'concentration': 0.5, 'step_number': i}
                for i in range(100)
            ],
            'episode_data': {
                'episode_id': 1,
                'total_steps': 100,
                'success': True
            }
        }
        
        # Test that aggregator can process recorder data format
        episode_data = {
            'episode_id': 1,
            'trajectory': np.array([[step['position'][0], step['position'][1]] 
                                  for step in simulated_recorded_data['step_data']]),
            'concentrations': np.array([step['concentration'] 
                                      for step in simulated_recorded_data['step_data']]),
            'episode_length': 100
        }
        
        stats = aggregator.calculate_episode_stats(episode_data, episode_id=1)
        
        # Verify statistics are calculated correctly from recorder data
        assert isinstance(stats, dict)
        assert stats['timing']['episode_length'] == 100
        assert 'trajectory' in stats
        assert 'concentration' in stats
    
    def test_episode_completion_hook_integration(self, integration_config):
        """Test integration with episode completion hooks."""
        aggregator = StatsAggregator(integration_config)
        
        # Simulate episode completion hook
        def episode_completion_hook(episode_data):
            """Simulate hook that calls statistics aggregator."""
            return aggregator.calculate_episode_stats(episode_data)
        
        # Test episode data
        episode_data = {
            'episode_id': 42,
            'trajectory': np.random.rand(75, 2),
            'concentrations': np.random.rand(75),
            'episode_length': 75,
            'final_reward': 12.5
        }
        
        # Call hook
        hook_result = episode_completion_hook(episode_data)
        
        # Verify hook integration works
        assert isinstance(hook_result, dict)
        assert hook_result['episode_id'] == 42
        assert 'trajectory' in hook_result
    
    def test_batch_processing_capability(self, integration_config):
        """Test batch processing for comparative studies."""
        aggregator = StatsAggregator(integration_config)
        
        # Create batch of episodes for comparative study
        batch_episodes = []
        for i in range(5):
            np.random.seed(42 + i)  # Different but reproducible data
            episode = {
                'episode_id': i,
                'trajectory': np.random.rand(50, 2) * 100,
                'concentrations': np.random.exponential(0.1, 50),
                'episode_length': 50,
                'condition': 'treatment' if i % 2 == 0 else 'control'
            }
            batch_episodes.append(episode)
        
        # Process batch
        run_stats = aggregator.calculate_run_stats(batch_episodes, run_id='batch_test')
        
        # Verify batch processing results
        assert isinstance(run_stats, dict)
        assert run_stats['episode_count'] == 5
        assert 'trajectory_aggregation' in run_stats
        assert 'summary' in run_stats


class TestStatsAggregatorErrorHandling:
    """Test suite for error handling and edge case management."""
    
    @pytest.fixture
    def error_config(self) -> StatsAggregatorConfig:
        """Create configuration for error handling testing."""
        return StatsAggregatorConfig(
            metrics_definitions={'trajectory': ['mean', 'std']},
            aggregation_levels=['episode'],
            data_validation=True,
            error_handling='warn'  # Don't raise errors, just warn
        )
    
    def test_empty_episode_data_handling(self, error_config):
        """Test handling of empty episode data."""
        aggregator = StatsAggregator(error_config)
        
        # Test with empty trajectory
        empty_data = {
            'episode_id': 1,
            'trajectory': np.array([]).reshape(0, 2),
            'concentrations': np.array([]),
            'episode_length': 0
        }
        
        stats = aggregator.calculate_episode_stats(empty_data, episode_id=1)
        
        # Should handle gracefully and return valid structure
        assert isinstance(stats, dict)
        assert 'episode_id' in stats
        assert 'trajectory' in stats
    
    def test_invalid_trajectory_data_handling(self, error_config):
        """Test handling of invalid trajectory data."""
        aggregator = StatsAggregator(error_config)
        
        # Test with invalid trajectory shape
        invalid_data = {
            'episode_id': 1,
            'trajectory': np.array([[1, 2, 3]]),  # Wrong shape (should be Nx2)
            'concentrations': np.array([0.5]),
            'episode_length': 1
        }
        
        # Should handle invalid data gracefully with error_handling='warn'
        stats = aggregator.calculate_episode_stats(invalid_data, episode_id=1)
        
        # Verify error handling works
        assert isinstance(stats, dict)
    
    def test_nan_and_inf_data_handling(self, error_config):
        """Test handling of NaN and infinite values in data."""
        aggregator = StatsAggregator(error_config)
        
        # Create data with NaN and inf values
        trajectory = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, np.inf], [5.0, 6.0]])
        concentrations = np.array([0.1, np.nan, 0.3, np.inf])
        
        problematic_data = {
            'episode_id': 1,
            'trajectory': trajectory,
            'concentrations': concentrations,
            'episode_length': 4
        }
        
        # Should handle NaN/inf values gracefully
        stats = aggregator.calculate_episode_stats(problematic_data, episode_id=1)
        
        assert isinstance(stats, dict)
        # Verify no NaN values in output statistics
        def check_no_nan(data_dict):
            for key, value in data_dict.items():
                if isinstance(value, dict):
                    check_no_nan(value)
                elif isinstance(value, (int, float)):
                    assert not np.isnan(value), f"NaN found in output: {key} = {value}"
        
        # Check trajectory and concentration stats
        if 'trajectory' in stats:
            check_no_nan(stats['trajectory'])
        if 'concentration' in stats:
            check_no_nan(stats['concentration'])
    
    def test_custom_calculation_error_handling(self, error_config):
        """Test error handling in custom calculations."""
        # Add failing custom calculation
        def failing_calculation(data):
            raise ValueError("Intentional test error")
        
        config = error_config
        config.custom_calculations = {'failing_metric': failing_calculation}
        
        aggregator = StatsAggregator(config)
        
        episode_data = {
            'episode_id': 1,
            'trajectory': np.random.rand(10, 2),
            'concentrations': np.random.rand(10),
            'episode_length': 10
        }
        
        # Should handle custom calculation error gracefully
        stats = aggregator.calculate_episode_stats(episode_data, episode_id=1)
        
        assert isinstance(stats, dict)
        # Custom calculation should be None due to error
        if 'custom' in stats:
            assert stats['custom']['failing_metric'] is None
    
    def test_export_error_handling(self, error_config):
        """Test error handling during export operations."""
        aggregator = StatsAggregator(error_config)
        
        # Try to export to invalid path
        invalid_path = "/invalid/readonly/path/summary.json"
        
        success = aggregator.export_summary(invalid_path)
        
        # Should return False on failure
        assert success is False


class TestStatsAggregatorUtilityFunctions:
    """Test suite for utility functions and helper methods."""
    
    def test_calculate_basic_stats_function(self):
        """Test standalone calculate_basic_stats function."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        stats = calculate_basic_stats(data)
        
        assert isinstance(stats, dict)
        assert stats['mean'] == pytest.approx(3.0)
        assert stats['std'] == pytest.approx(np.std(data))
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['median'] == 3.0
    
    def test_calculate_basic_stats_empty_array(self):
        """Test calculate_basic_stats with empty array."""
        empty_data = np.array([])
        
        stats = calculate_basic_stats(empty_data)
        
        assert isinstance(stats, dict)
        assert stats['mean'] == 0.0
        assert stats['std'] == 0.0
        assert stats['min'] == 0.0
        assert stats['max'] == 0.0
        assert stats['median'] == 0.0
    
    def test_calculate_advanced_stats_function(self):
        """Test standalone calculate_advanced_stats function."""
        # Create data with known statistical properties
        np.random.seed(42)
        data = np.random.normal(5.0, 2.0, 1000)  # Normal distribution
        
        stats = calculate_advanced_stats(data)
        
        assert isinstance(stats, dict)
        assert 'skewness' in stats
        assert 'kurtosis' in stats
        assert 'p25' in stats
        assert 'p75' in stats
        assert 'p95' in stats
        
        # Verify percentiles are ordered correctly
        assert stats['p25'] <= stats['p75'] <= stats['p95']
    
    def test_create_stats_aggregator_factory(self):
        """Test create_stats_aggregator factory function."""
        config_dict = {
            'metrics_definitions': {'trajectory': ['mean']},
            'aggregation_levels': ['episode'],
            'output_format': 'json'
        }
        
        # Test with dictionary
        aggregator = create_stats_aggregator(config_dict)
        
        assert isinstance(aggregator, StatsAggregator)
        assert aggregator.config.output_format == 'json'
        
        # Test with StatsAggregatorConfig object
        config_obj = StatsAggregatorConfig(**config_dict)
        aggregator2 = create_stats_aggregator(config_obj)
        
        assert isinstance(aggregator2, StatsAggregator)
        assert aggregator2.config.output_format == 'json'
    
    def test_generate_summary_report_function(self):
        """Test generate_summary_report convenience function."""
        # Create sample episodes data
        episodes_data = []
        for i in range(3):
            np.random.seed(42 + i)
            episode = {
                'episode_id': i,
                'trajectory': np.random.rand(20, 2),
                'concentrations': np.random.rand(20),
                'episode_length': 20
            }
            episodes_data.append(episode)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = str(Path(tmp_dir) / 'test_report.json')
            
            # Generate summary report
            success = generate_summary_report(episodes_data, output_path)
            
            assert success is True
            assert Path(output_path).exists()
            
            # Verify report content
            with open(output_path, 'r') as f:
                report_data = json.load(f)
            
            assert 'metadata' in report_data
            assert 'run_statistics' in report_data


class TestStatsAggregatorConfiguration:
    """Test suite for configuration validation and management."""
    
    def test_config_validation_valid(self):
        """Test valid configuration creation."""
        config = StatsAggregatorConfig(
            metrics_definitions={'trajectory': ['mean', 'std']},
            aggregation_levels=['episode', 'run'],
            output_format='json',
            performance_tracking=True,
            memory_limit_mb=512
        )
        
        assert config.metrics_definitions == {'trajectory': ['mean', 'std']}
        assert config.aggregation_levels == ['episode', 'run']
        assert config.output_format == 'json'
        assert config.memory_limit_mb == 512
    
    def test_config_validation_invalid_memory_limit(self):
        """Test configuration validation with invalid memory limit."""
        with pytest.raises(ValueError, match="memory_limit_mb must be positive"):
            StatsAggregatorConfig(memory_limit_mb=-1)
    
    def test_config_validation_invalid_timeout(self):
        """Test configuration validation with invalid timeout."""
        with pytest.raises(ValueError, match="computation_timeout_s must be positive"):
            StatsAggregatorConfig(computation_timeout_s=-5.0)
    
    def test_config_validation_invalid_output_format(self):
        """Test configuration validation with invalid output format."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            StatsAggregatorConfig(output_format='invalid_format')
    
    def test_config_validation_invalid_error_handling(self):
        """Test configuration validation with invalid error handling."""
        with pytest.raises(ValueError, match="error_handling must be one of"):
            StatsAggregatorConfig(error_handling='invalid_mode')
    
    def test_runtime_configuration_update(self):
        """Test runtime configuration updates."""
        config = StatsAggregatorConfig()
        aggregator = StatsAggregator(config)
        
        # Update configuration
        aggregator.configure_metrics(
            performance_tracking=False,
            output_format='yaml'
        )
        
        assert aggregator.config.performance_tracking is False
        assert aggregator.config.output_format == 'yaml'


# Integration test to verify the complete workflow
class TestStatsAggregatorWorkflow:
    """Integration test for complete statistics aggregator workflow."""
    
    def test_complete_workflow(self):
        """Test complete workflow from configuration to export."""
        # Step 1: Create configuration
        config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'total_distance', 'displacement_efficiency'],
                'concentration': ['mean', 'detection_rate'],
                'speed': ['mean', 'movement_efficiency'],
                'timing': ['episode_length']
            },
            aggregation_levels=['episode', 'run'],
            output_format='json',
            performance_tracking=True
        )
        
        # Step 2: Create aggregator
        aggregator = StatsAggregator(config)
        
        # Step 3: Generate test data
        episodes_data = []
        for i in range(5):
            np.random.seed(42 + i)
            episode = {
                'episode_id': i,
                'trajectory': np.random.rand(100, 2) * 100,
                'concentrations': np.random.exponential(0.1, 100),
                'speeds': np.random.gamma(2, 0.5, 99),
                'episode_length': 100,
                'step_durations': np.full(100, 0.025),
                'success': i % 2 == 0  # Alternate success/failure
            }
            episodes_data.append(episode)
        
        # Step 4: Calculate episode statistics
        episode_stats_list = []
        for episode_data in episodes_data:
            stats = aggregator.calculate_episode_stats(
                episode_data, 
                episode_id=episode_data['episode_id']
            )
            episode_stats_list.append(stats)
            
            # Verify each episode calculation
            assert isinstance(stats, dict)
            assert 'trajectory' in stats
            assert 'concentration' in stats
            assert 'speed' in stats
            assert 'timing' in stats
        
        # Step 5: Calculate run statistics
        run_stats = aggregator.calculate_run_stats(episodes_data, run_id='workflow_test')
        
        # Verify run statistics
        assert isinstance(run_stats, dict)
        assert run_stats['episode_count'] == 5
        assert 'trajectory_aggregation' in run_stats
        assert 'summary' in run_stats
        
        # Step 6: Export summary
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / 'workflow_summary.json'
            
            success = aggregator.export_summary(str(output_path))
            
            assert success is True
            assert output_path.exists()
            
            # Verify exported data
            with open(output_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'metadata' in exported_data
            assert 'performance_metrics' in exported_data
            assert 'run_statistics' in exported_data
        
        # Step 7: Verify performance metrics
        performance_metrics = aggregator.get_performance_metrics()
        
        assert 'episodes_processed' in performance_metrics
        assert performance_metrics['episodes_processed'] == 5
        assert 'computation_time_ms' in performance_metrics
        assert 'performance_compliance' in performance_metrics


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__, '-v'])