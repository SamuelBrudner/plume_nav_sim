"""
Comprehensive pytest test module for validating StatsAggregatorProtocol implementations.

This module provides comprehensive validation for the statistics aggregation functionality
including automated metrics calculation, summary generation, and export functionality with 
performance compliance validation. Tests cover protocol interface compliance, episode-level 
and run-level statistical analysis, custom metrics support, and integration with the 
recorder system as specified in Section 0.4.1 of the technical specification.

Test Categories:
- StatsAggregatorProtocol compliance and interface verification per Section 6.6.2.4
- Episode-level statistics calculation with error analysis and validation
- Run-level aggregation with hierarchical statistical analysis and correlation metrics  
- Summary.json export functionality with standardized research metrics format
- Performance validation ensuring 100ms post-episode computation compliance
- Configuration testing for metrics definitions and custom calculation functions
- Integration testing with recorder system connectivity and data pipeline validation
- Error handling and edge case validation for robust operation

Performance Requirements Tested:
- Statistics computation: ≤100ms post-episode per Section 0.5.1 validation checklist
- Memory efficiency: Configurable limits with validation per Section 5.2.2
- Protocol compliance: 100% coverage for StatsAggregatorProtocol methods per Section 6.6.2.4
- Data validation: Comprehensive input validation and schema compliance testing
- Export validation: Summary.json format compliance and standardized metrics generation

Architecture Validation:
- Protocol-based design ensuring interface compliance and extensibility
- Configuration-driven metrics definitions with Hydra integration support
- Recorder system integration for automated data collection workflows
- Performance monitoring with detailed timing and resource tracking
- Error handling strategies with configurable error management policies

Authors: Blitzy Enhanced Testing Framework
Version: v1.0 (Statistics Aggregation Testing)
License: MIT
"""

import pytest
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from unittest.mock import patch, Mock, MagicMock
import tempfile
from pathlib import Path
import json

# Import components to test
from plume_nav_sim.core.protocols import StatsAggregatorProtocol
from plume_nav_sim.analysis.stats import StatsAggregator, StatsAggregatorConfig
from plume_nav_sim.analysis import generate_summary

# Test configuration constants
PERFORMANCE_TIMEOUT_MS = 100.0  # 100ms post-episode requirement
NUMERICAL_PRECISION_TOLERANCE = 1e-6
LARGE_DATASET_SIZE = 1000
STRESS_TEST_EPISODES = 100


class TestStatsAggregatorProtocol:
    """
    Test StatsAggregatorProtocol interface compliance and type checking.
    
    Validates that the StatsAggregatorProtocol defines the correct interface and that
    concrete implementations properly implement all required methods with correct
    signatures and return types per Section 6.6.2.4 protocol coverage requirements.
    """
    
    def test_protocol_compliance(self):
        """Test that StatsAggregatorProtocol defines required interface methods."""
        # Verify protocol is runtime checkable
        assert hasattr(StatsAggregatorProtocol, '__runtime_checkable__')
        
        # Required methods for protocol compliance
        required_methods = [
            'calculate_episode_stats',
            'calculate_run_stats', 
            'export_summary',
            'configure_metrics'
        ]
        
        for method_name in required_methods:
            assert hasattr(StatsAggregatorProtocol, method_name), \
                f"Protocol missing required method: {method_name}"
    
    def test_interface_implementation(self):
        """Test that StatsAggregator implements StatsAggregatorProtocol interface."""
        config = StatsAggregatorConfig()
        aggregator = StatsAggregator(config)
        
        # Test protocol compliance
        assert isinstance(aggregator, StatsAggregatorProtocol)
        
        # Test all required methods are callable
        assert callable(aggregator.calculate_episode_stats)
        assert callable(aggregator.calculate_run_stats)
        assert callable(aggregator.export_summary)
        assert callable(aggregator.configure_metrics)
    
    def test_method_signatures(self):
        """Test that protocol methods have correct signatures."""
        config = StatsAggregatorConfig()
        aggregator = StatsAggregator(config)
        
        # Test method signatures accept expected parameters
        import inspect
        
        # calculate_episode_stats signature
        calc_episode_sig = inspect.signature(aggregator.calculate_episode_stats)
        expected_params = ['episode_data', 'episode_id']
        for param in expected_params:
            assert param in calc_episode_sig.parameters, \
                f"calculate_episode_stats missing parameter: {param}"
        
        # calculate_run_stats signature  
        calc_run_sig = inspect.signature(aggregator.calculate_run_stats)
        run_params = ['episodes_data', 'run_id']
        for param in run_params:
            assert param in calc_run_sig.parameters, \
                f"calculate_run_stats missing parameter: {param}"
        
        # export_summary signature
        export_sig = inspect.signature(aggregator.export_summary)
        export_params = ['output_path', 'format']
        for param in export_params:
            assert param in export_sig.parameters, \
                f"export_summary missing parameter: {param}"
    
    def test_protocol_inheritance(self):
        """Test protocol inheritance and isinstance checks."""
        config = StatsAggregatorConfig()
        aggregator = StatsAggregator(config)
        
        # Test isinstance with protocol
        assert isinstance(aggregator, StatsAggregatorProtocol)
        
        # Test that protocol methods exist and are bound
        assert hasattr(aggregator, 'calculate_episode_stats')
        assert hasattr(aggregator, 'calculate_run_stats')
        assert hasattr(aggregator, 'export_summary')
        assert hasattr(aggregator, 'configure_metrics')


class TestEpisodeStatistics:
    """
    Test episode-level statistics calculation functionality.
    
    Validates episode statistics calculation accuracy, metrics computation,
    error analysis, and data integrity per automated statistics collection
    requirements and performance compliance targets.
    """
    
    @pytest.fixture
    def sample_episode_data(self):
        """Create sample episode data for testing."""
        np.random.seed(42)  # Deterministic test data
        
        # Generate realistic trajectory
        n_steps = 100
        trajectory = np.cumsum(np.random.randn(n_steps, 2) * 0.1, axis=0)
        trajectory += np.array([10.0, 15.0])  # Starting position
        
        # Generate concentration data
        concentrations = np.random.exponential(0.5, n_steps)
        concentrations[concentrations > 2.0] = 0.0  # Some no-detection periods
        
        # Generate speed data
        speeds = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        speeds = np.concatenate([[0.0], speeds])  # Include initial speed
        
        return {
            'episode_id': 42,
            'trajectory': trajectory,
            'concentrations': concentrations,
            'speeds': speeds,
            'episode_length': n_steps,
            'step_durations': np.random.normal(0.02, 0.005, n_steps),  # ~20ms ± 5ms
            'timestamp': time.time()
        }
    
    @pytest.fixture
    def basic_aggregator(self):
        """Create basic statistics aggregator for testing."""
        config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std', 'min', 'max'],
                'concentration': ['mean', 'std', 'detection_rate'],
                'speed': ['mean', 'total_distance'],
                'timing': ['episode_length']
            },
            performance_tracking=True
        )
        return StatsAggregator(config)
    
    def test_episode_stats_calculation(self, basic_aggregator, sample_episode_data):
        """Test basic episode statistics calculation."""
        stats = basic_aggregator.calculate_episode_stats(sample_episode_data)
        
        # Verify basic structure
        assert isinstance(stats, dict)
        assert 'episode_id' in stats
        assert stats['episode_id'] == 42
        assert 'timestamp' in stats
        assert 'computation_metadata' in stats
        
        # Verify trajectory statistics
        assert 'trajectory' in stats
        traj_stats = stats['trajectory']
        assert 'mean_position' in traj_stats
        assert 'std_position' in traj_stats
        assert 'total_distance' in traj_stats
        assert 'net_displacement' in traj_stats
        
        # Verify concentration statistics
        assert 'concentration' in stats
        conc_stats = stats['concentration']
        assert 'mean' in conc_stats
        assert 'std' in conc_stats
        assert 'detection_rate' in conc_stats
        
        # Verify speed statistics
        assert 'speed' in stats
        speed_stats = stats['speed']
        assert 'mean_speed' in speed_stats
        assert 'total_distance' in speed_stats
    
    def test_metrics_accuracy(self, basic_aggregator, sample_episode_data):
        """Test accuracy of calculated metrics against known values."""
        stats = basic_aggregator.calculate_episode_stats(sample_episode_data)
        
        # Verify trajectory metrics accuracy
        trajectory = sample_episode_data['trajectory']
        expected_mean = np.mean(trajectory, axis=0)
        calculated_mean = stats['trajectory']['mean_position']
        
        np.testing.assert_allclose(
            calculated_mean, 
            expected_mean, 
            atol=NUMERICAL_PRECISION_TOLERANCE
        )
        
        # Verify concentration metrics
        concentrations = sample_episode_data['concentrations']
        expected_mean_conc = np.mean(concentrations)
        calculated_mean_conc = stats['concentration']['mean']
        
        assert abs(calculated_mean_conc - expected_mean_conc) < NUMERICAL_PRECISION_TOLERANCE
        
        # Verify detection rate calculation
        detection_threshold = 0.01
        expected_detection_rate = np.mean(concentrations > detection_threshold)
        calculated_detection_rate = stats['concentration']['detection_rate']
        
        assert abs(calculated_detection_rate - expected_detection_rate) < NUMERICAL_PRECISION_TOLERANCE
    
    def test_error_analysis(self, basic_aggregator):
        """Test error analysis and uncertainty metrics calculation."""
        # Create episode data with error information
        episode_data = {
            'episode_id': 1,
            'trajectory': np.random.randn(50, 2),
            'concentrations': np.random.exponential(0.3, 50),
            'position_errors': np.random.exponential(0.1, 50),
            'measurement_uncertainties': np.random.normal(0.02, 0.01, 50)
        }
        
        stats = basic_aggregator.calculate_episode_stats(episode_data)
        
        # Verify error analysis is included
        assert 'error_analysis' in stats
        error_stats = stats['error_analysis']
        
        assert 'mean_position_error' in error_stats
        assert 'max_position_error' in error_stats
        assert 'mean_uncertainty' in error_stats
        
        # Verify error metrics accuracy
        expected_mean_error = np.mean(episode_data['position_errors'])
        assert abs(error_stats['mean_position_error'] - expected_mean_error) < NUMERICAL_PRECISION_TOLERANCE
    
    def test_statistical_measures(self, basic_aggregator, sample_episode_data):
        """Test advanced statistical measures and distribution analysis."""
        stats = basic_aggregator.calculate_episode_stats(sample_episode_data)
        
        # Verify advanced trajectory metrics
        traj_stats = stats['trajectory']
        assert 'tortuosity' in traj_stats
        assert 'displacement_efficiency' in traj_stats
        assert 'exploration_area' in traj_stats
        
        # Verify values are reasonable
        assert traj_stats['tortuosity'] >= 1.0  # Should be >= 1 by definition
        assert 0.0 <= traj_stats['displacement_efficiency'] <= 1.0
        assert traj_stats['exploration_area'] >= 0.0
        
        # Verify concentration distribution analysis
        conc_stats = stats['concentration']
        assert 'percentiles' in conc_stats
        assert 'skewness' in conc_stats
        assert 'kurtosis' in conc_stats
        
        # Verify percentiles are ordered
        percentiles = conc_stats['percentiles']
        assert percentiles['p10'] <= percentiles['p25'] <= percentiles['p75'] <= percentiles['p90']
    
    def test_episode_data_integrity(self, basic_aggregator):
        """Test data integrity validation and error handling."""
        # Test with invalid trajectory data
        invalid_data = {
            'episode_id': 1,
            'trajectory': np.array([[np.inf, 0], [0, np.nan]]),  # Invalid values
            'concentrations': np.array([0.1, -0.5, 0.3])  # Negative concentration
        }
        
        # Should handle gracefully with error reporting
        stats = basic_aggregator.calculate_episode_stats(invalid_data)
        
        # Either returns error stats or handles gracefully
        assert isinstance(stats, dict)
        assert 'episode_id' in stats
        
        # Test with empty data
        empty_data = {
            'episode_id': 2,
            'trajectory': np.array([]).reshape(0, 2),
            'concentrations': np.array([])
        }
        
        stats = basic_aggregator.calculate_episode_stats(empty_data)
        assert isinstance(stats, dict)
        
        # Should have default/empty values for empty data
        if 'trajectory' in stats:
            traj_stats = stats['trajectory']
            assert traj_stats['total_distance'] == 0.0


class TestRunStatistics:
    """
    Test run-level statistics aggregation functionality.
    
    Validates multi-episode aggregation, hierarchical statistical analysis,
    cross-episode correlation metrics, and run-level performance validation
    per automated statistics collection requirements.
    """
    
    @pytest.fixture
    def sample_episodes_data(self):
        """Create sample multi-episode data for run testing."""
        episodes = []
        np.random.seed(123)  # Deterministic data
        
        for i in range(10):
            n_steps = np.random.randint(50, 150)
            
            # Generate trajectory with some variation
            trajectory = np.cumsum(np.random.randn(n_steps, 2) * 0.1, axis=0)
            trajectory += np.random.randn(2) * 5.0  # Random starting position
            
            # Generate concentrations with trend
            base_concentration = 0.3 + i * 0.05  # Increasing trend
            concentrations = np.random.exponential(base_concentration, n_steps)
            
            # Generate speeds
            speeds = np.linalg.norm(np.diff(trajectory, axis=0), axis=1) if n_steps > 1 else np.array([0.0])
            if n_steps > 1:
                speeds = np.concatenate([[0.0], speeds])
            
            episodes.append({
                'episode_id': i,
                'trajectory': trajectory,
                'concentrations': concentrations,
                'speeds': speeds,
                'episode_length': n_steps,
                'step_durations': np.random.normal(0.025, 0.008, n_steps),
                'performance_score': 0.5 + i * 0.04  # Improving performance
            })
        
        return episodes
    
    @pytest.fixture
    def advanced_aggregator(self):
        """Create advanced aggregator for run-level testing."""
        config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std', 'total_distance', 'displacement_efficiency'],
                'concentration': ['mean', 'std', 'detection_rate', 'percentiles'],
                'speed': ['mean', 'std', 'total_distance'],
                'timing': ['episode_length', 'performance_compliance_rate']
            },
            aggregation_levels=['episode', 'run'],
            performance_tracking=True
        )
        return StatsAggregator(config)
    
    def test_run_aggregation(self, advanced_aggregator, sample_episodes_data):
        """Test basic run-level aggregation functionality."""
        run_stats = advanced_aggregator.calculate_run_stats(sample_episodes_data)
        
        # Verify basic structure
        assert isinstance(run_stats, dict)
        assert 'run_id' in run_stats
        assert 'episode_count' in run_stats
        assert run_stats['episode_count'] == len(sample_episodes_data)
        assert 'timestamp' in run_stats
        assert 'computation_metadata' in run_stats
        
        # Verify aggregation sections
        assert 'trajectory_aggregation' in run_stats
        assert 'concentration_aggregation' in run_stats
        assert 'speed_aggregation' in run_stats
        assert 'timing_aggregation' in run_stats
        
        # Verify summary section
        assert 'summary' in run_stats
        summary = run_stats['summary']
        assert 'total_episodes' in summary
        assert summary['total_episodes'] == len(sample_episodes_data)
    
    def test_multi_episode_analysis(self, advanced_aggregator, sample_episodes_data):
        """Test multi-episode analysis and cross-episode statistics."""
        run_stats = advanced_aggregator.calculate_run_stats(sample_episodes_data)
        
        # Test trajectory aggregation
        traj_agg = run_stats['trajectory_aggregation']
        assert 'total_distance_mean' in traj_agg
        assert 'total_distance_std' in traj_agg
        assert 'displacement_efficiency_mean' in traj_agg
        
        # Verify aggregated values are reasonable
        assert traj_agg['total_distance_mean'] > 0.0
        assert traj_agg['total_distance_std'] >= 0.0
        assert 0.0 <= traj_agg['displacement_efficiency_mean'] <= 1.0
        
        # Test concentration aggregation
        conc_agg = run_stats['concentration_aggregation']
        assert 'detection_rate_mean' in conc_agg
        assert 'concentration_mean' in conc_agg
        
        # Verify reasonable values
        assert 0.0 <= conc_agg['detection_rate_mean'] <= 1.0
        assert conc_agg['concentration_mean'] >= 0.0
    
    def test_hierarchical_statistics(self, advanced_aggregator, sample_episodes_data):
        """Test hierarchical aggregation levels and nested statistics."""
        run_stats = advanced_aggregator.calculate_run_stats(sample_episodes_data)
        
        # Verify correlations are calculated
        assert 'correlations' in run_stats
        correlations = run_stats['correlations']
        
        # Should have cross-metric correlations
        for correlation_key in correlations.keys():
            if isinstance(correlations[correlation_key], dict):
                assert 'correlation' in correlations[correlation_key]
                assert 'p_value' in correlations[correlation_key]
                assert 'significant' in correlations[correlation_key]
                
                # Correlation should be in valid range
                corr_value = correlations[correlation_key]['correlation']
                assert -1.0 <= corr_value <= 1.0
    
    def test_cross_episode_correlation(self, advanced_aggregator, sample_episodes_data):
        """Test cross-episode correlation analysis and trend detection."""
        run_stats = advanced_aggregator.calculate_run_stats(sample_episodes_data)
        
        # Verify trends analysis
        assert 'trends' in run_stats
        trends = run_stats['trends']
        
        # Check for trend analysis results
        if 'detection_rate_trend' in trends:
            trend_data = trends['detection_rate_trend']
            assert 'slope' in trend_data
            assert 'r_squared' in trend_data
            assert 'p_value' in trend_data
            assert 'improving' in trend_data
            
            # R-squared should be in valid range
            assert 0.0 <= trend_data['r_squared'] <= 1.0
    
    def test_run_level_metrics(self, advanced_aggregator, sample_episodes_data):
        """Test run-level specific metrics and summary statistics."""
        run_stats = advanced_aggregator.calculate_run_stats(sample_episodes_data)
        
        # Test timing aggregation
        timing_agg = run_stats['timing_aggregation']
        assert 'episode_length_mean' in timing_agg
        assert 'episode_length_std' in timing_agg
        assert 'episode_length_min' in timing_agg
        assert 'episode_length_max' in timing_agg
        
        # Verify logical relationships
        assert timing_agg['episode_length_min'] <= timing_agg['episode_length_mean']
        assert timing_agg['episode_length_mean'] <= timing_agg['episode_length_max']
        
        # Test summary statistics
        summary = run_stats['summary']
        assert 'successful_episodes' in summary
        assert 'average_performance' in summary
        assert 'best_episode' in summary
        assert 'worst_episode' in summary
        
        # Verify episode counts are reasonable
        assert 0 <= summary['successful_episodes'] <= len(sample_episodes_data)


class TestSummaryExport:
    """
    Test summary export functionality and standardized metrics format.
    
    Validates summary.json generation, export format compliance, standardized
    research metrics output, file integrity, and metadata inclusion per
    research reproducibility requirements.
    """
    
    @pytest.fixture
    def export_aggregator(self):
        """Create aggregator configured for export testing."""
        config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std', 'efficiency'],
                'concentration': ['mean', 'detection_rate']
            },
            output_format='json',
            output_validation=True
        )
        return StatsAggregator(config)
    
    @pytest.fixture
    def sample_run_data(self):
        """Create sample data for export testing."""
        episodes = []
        for i in range(5):
            episodes.append({
                'episode_id': i,
                'trajectory': np.random.randn(30, 2),
                'concentrations': np.random.exponential(0.4, 30),
                'episode_length': 30
            })
        return episodes
    
    def test_summary_json_generation(self, export_aggregator, sample_run_data):
        """Test summary.json generation with standard format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.json"
            
            # Calculate run stats first
            run_stats = export_aggregator.calculate_run_stats(sample_run_data)
            
            # Export summary
            success = export_aggregator.export_summary(output_path)
            assert success is True
            assert output_path.exists()
            
            # Verify file is valid JSON
            with open(output_path, 'r') as f:
                summary_data = json.load(f)
            
            assert isinstance(summary_data, dict)
            
            # Verify required top-level sections
            required_sections = ['metadata', 'performance_metrics', 'run_statistics']
            for section in required_sections:
                assert section in summary_data, f"Missing section: {section}"
    
    def test_export_format_validation(self, export_aggregator, sample_run_data):
        """Test export format validation and compliance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON export
            json_path = Path(temp_dir) / "summary.json"
            export_aggregator.calculate_run_stats(sample_run_data)
            
            success = export_aggregator.export_summary(json_path, format='json')
            assert success is True
            
            # Verify JSON structure
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            assert 'metadata' in json_data
            metadata = json_data['metadata']
            assert 'export_timestamp' in metadata
            assert 'aggregator_version' in metadata
            assert 'format_version' in metadata
            
            # Test YAML export if available
            try:
                yaml_path = Path(temp_dir) / "summary.yaml"
                success = export_aggregator.export_summary(yaml_path, format='yaml')
                if success:
                    assert yaml_path.exists()
            except ImportError:
                # YAML not available, skip test
                pass
    
    def test_standardized_metrics(self, export_aggregator, sample_run_data):
        """Test standardized research metrics format compliance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "standardized_summary.json"
            
            # Generate run statistics
            run_stats = export_aggregator.calculate_run_stats(sample_run_data)
            
            # Export with standardized format
            success = export_aggregator.export_summary(output_path)
            assert success is True
            
            # Load and validate standardized metrics
            with open(output_path, 'r') as f:
                summary = json.load(f)
            
            # Verify configuration section
            assert 'configuration' in summary['metadata']
            config = summary['metadata']['configuration']
            assert 'metrics_definitions' in config
            assert 'aggregation_levels' in config
            
            # Verify performance metrics section
            assert 'performance_metrics' in summary
            perf = summary['performance_metrics']
            assert 'episodes_processed' in perf
            assert 'computation_time_ms' in perf
            assert 'performance_compliance' in perf
    
    def test_file_output_integrity(self, export_aggregator, sample_run_data):
        """Test file output integrity and error handling."""
        # Test export to non-existent directory
        invalid_path = Path("/non/existent/directory/summary.json")
        export_aggregator.calculate_run_stats(sample_run_data)
        
        # Should handle gracefully
        success = export_aggregator.export_summary(invalid_path)
        # May succeed if path is created or fail gracefully
        assert isinstance(success, bool)
        
        # Test with valid directory
        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "valid_summary.json"
            success = export_aggregator.export_summary(valid_path)
            
            if success:
                assert valid_path.exists()
                assert valid_path.stat().st_size > 0  # File has content
                
                # Verify file is valid JSON
                with open(valid_path, 'r') as f:
                    data = json.load(f)
                assert isinstance(data, dict)
    
    def test_metadata_inclusion(self, export_aggregator, sample_run_data):
        """Test comprehensive metadata inclusion in export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metadata_test.json"
            
            # Calculate stats and export
            export_aggregator.calculate_run_stats(sample_run_data)
            success = export_aggregator.export_summary(output_path)
            assert success is True
            
            # Load and verify metadata
            with open(output_path, 'r') as f:
                summary = json.load(f)
            
            metadata = summary['metadata']
            
            # Verify timestamp information
            assert 'export_timestamp' in metadata
            assert isinstance(metadata['export_timestamp'], (int, float))
            assert metadata['export_timestamp'] > 0
            
            # Verify version information
            assert 'aggregator_version' in metadata
            assert 'format_version' in metadata
            
            # Verify configuration metadata
            assert 'configuration' in metadata
            config = metadata['configuration']
            assert 'metrics_definitions' in config
            assert 'performance_tracking' in config


class TestStatsPerformance:
    """
    Test statistics computation performance and compliance validation.
    
    Validates computation latency, 100ms post-episode requirement compliance,
    memory efficiency, large dataset performance, and concurrent processing
    capabilities per Section 6.6.6.3 performance targets.
    """
    
    @pytest.fixture
    def performance_aggregator(self):
        """Create aggregator optimized for performance testing."""
        config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std'],  # Minimal metrics for speed
                'concentration': ['mean', 'detection_rate']
            },
            performance_tracking=True,
            precision_mode='float32'  # Faster processing
        )
        return StatsAggregator(config)
    
    def test_computation_latency(self, performance_aggregator):
        """Test computation latency for various data sizes."""
        data_sizes = [10, 50, 100, 500]
        
        for size in data_sizes:
            # Generate test data
            episode_data = {
                'episode_id': 1,
                'trajectory': np.random.randn(size, 2),
                'concentrations': np.random.exponential(0.3, size),
                'episode_length': size
            }
            
            # Measure computation time
            start_time = time.perf_counter()
            stats = performance_aggregator.calculate_episode_stats(episode_data)
            computation_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Verify computation completed
            assert isinstance(stats, dict)
            assert 'episode_id' in stats
            
            # Log performance for analysis
            print(f"Episode size {size}: {computation_time_ms:.2f}ms")
            
            # For reasonable sizes, should be well under 100ms
            if size <= 100:
                assert computation_time_ms < PERFORMANCE_TIMEOUT_MS / 2, \
                    f"Computation took {computation_time_ms:.2f}ms for size {size}"
    
    def test_100ms_post_episode_requirement(self, performance_aggregator):
        """Test 100ms post-episode computation requirement compliance."""
        # Create realistic episode data
        episode_data = {
            'episode_id': 1,
            'trajectory': np.random.randn(200, 2),  # Typical episode length
            'concentrations': np.random.exponential(0.4, 200),
            'speeds': np.random.exponential(0.5, 200),
            'episode_length': 200,
            'step_durations': np.random.normal(0.025, 0.005, 200)
        }
        
        # Measure performance multiple times for reliability
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            stats = performance_aggregator.calculate_episode_stats(episode_data)
            end_time = time.perf_counter()
            
            computation_time_ms = (end_time - start_time) * 1000
            times.append(computation_time_ms)
            
            # Verify computation completed successfully
            assert isinstance(stats, dict)
            assert 'computation_metadata' in stats
        
        # Check performance statistics
        mean_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        max_time = np.max(times)
        
        print(f"Performance stats: mean={mean_time:.2f}ms, p95={p95_time:.2f}ms, max={max_time:.2f}ms")
        
        # Performance requirements
        assert mean_time < PERFORMANCE_TIMEOUT_MS, \
            f"Mean computation time {mean_time:.2f}ms exceeds {PERFORMANCE_TIMEOUT_MS}ms target"
        assert p95_time < PERFORMANCE_TIMEOUT_MS * 1.5, \
            f"P95 computation time {p95_time:.2f}ms exceeds tolerance"
    
    def test_memory_efficiency(self, performance_aggregator):
        """Test memory efficiency and resource management."""
        import psutil
        import gc
        
        # Get baseline memory
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        
        # Process multiple episodes
        for i in range(20):
            episode_data = {
                'episode_id': i,
                'trajectory': np.random.randn(100, 2),
                'concentrations': np.random.exponential(0.3, 100),
                'episode_length': 100
            }
            
            stats = performance_aggregator.calculate_episode_stats(episode_data)
            assert isinstance(stats, dict)
            
            # Force garbage collection periodically
            if i % 5 == 0:
                gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss
        memory_growth_mb = (final_memory - baseline_memory) / (1024 * 1024)
        
        print(f"Memory growth: {memory_growth_mb:.2f}MB for 20 episodes")
        
        # Should not grow excessively
        assert memory_growth_mb < 50, \
            f"Memory growth {memory_growth_mb:.2f}MB too high"
    
    def test_large_dataset_performance(self, performance_aggregator):
        """Test performance with large datasets."""
        # Create large episode
        large_episode = {
            'episode_id': 1,
            'trajectory': np.random.randn(LARGE_DATASET_SIZE, 2),
            'concentrations': np.random.exponential(0.3, LARGE_DATASET_SIZE),
            'speeds': np.random.exponential(0.5, LARGE_DATASET_SIZE),
            'episode_length': LARGE_DATASET_SIZE
        }
        
        # Test episode statistics
        start_time = time.perf_counter()
        episode_stats = performance_aggregator.calculate_episode_stats(large_episode)
        episode_time_ms = (time.perf_counter() - start_time) * 1000
        
        assert isinstance(episode_stats, dict)
        print(f"Large episode ({LARGE_DATASET_SIZE} steps): {episode_time_ms:.2f}ms")
        
        # Should still be reasonable for large datasets
        assert episode_time_ms < PERFORMANCE_TIMEOUT_MS * 3, \
            f"Large episode processing too slow: {episode_time_ms:.2f}ms"
        
        # Test run statistics with multiple episodes
        episodes = [large_episode.copy() for _ in range(5)]
        for i, episode in enumerate(episodes):
            episode['episode_id'] = i
        
        start_time = time.perf_counter()
        run_stats = performance_aggregator.calculate_run_stats(episodes)
        run_time_ms = (time.perf_counter() - start_time) * 1000
        
        assert isinstance(run_stats, dict)
        print(f"Run stats (5 large episodes): {run_time_ms:.2f}ms")
    
    def test_concurrent_processing(self, performance_aggregator):
        """Test concurrent statistics processing capabilities."""
        import threading
        import queue
        
        # Create test data
        episodes = []
        for i in range(20):
            episodes.append({
                'episode_id': i,
                'trajectory': np.random.randn(50, 2),
                'concentrations': np.random.exponential(0.3, 50),
                'episode_length': 50
            })
        
        # Test concurrent processing
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker(episode_data):
            try:
                stats = performance_aggregator.calculate_episode_stats(episode_data)
                results_queue.put(stats)
            except Exception as e:
                errors_queue.put(e)
        
        # Launch threads
        threads = []
        start_time = time.perf_counter()
        
        for episode in episodes[:10]:  # Process 10 concurrently
            thread = threading.Thread(target=worker, args=(episode,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        concurrent_time_ms = (end_time - start_time) * 1000
        
        # Verify results
        assert errors_queue.empty(), f"Errors in concurrent processing: {list(errors_queue.queue)}"
        assert results_queue.qsize() == 10, f"Expected 10 results, got {results_queue.qsize()}"
        
        print(f"Concurrent processing (10 episodes): {concurrent_time_ms:.2f}ms")
        
        # Concurrent processing should be efficient
        assert concurrent_time_ms < PERFORMANCE_TIMEOUT_MS * 2, \
            f"Concurrent processing too slow: {concurrent_time_ms:.2f}ms"


class TestStatsConfiguration:
    """
    Test statistics configuration and customization functionality.
    
    Validates configurable metrics definitions, custom calculation functions,
    hierarchical aggregation settings, metrics validation, and schema
    compliance per configuration management requirements.
    """
    
    def test_configurable_metrics(self):
        """Test configurable metrics definitions and validation."""
        # Test custom metrics configuration
        custom_config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std', 'custom_efficiency'],
                'concentration': ['mean', 'percentiles', 'custom_detection'],
                'custom_category': ['mean', 'variance']
            },
            aggregation_levels=['episode', 'run', 'batch']
        )
        
        aggregator = StatsAggregator(custom_config)
        
        # Verify configuration was applied
        assert aggregator.config.metrics_definitions['trajectory'] == ['mean', 'std', 'custom_efficiency']
        assert 'custom_category' in aggregator.config.metrics_definitions
        assert aggregator.config.aggregation_levels == ['episode', 'run', 'batch']
    
    def test_custom_calculation_functions(self):
        """Test custom calculation functions integration."""
        # Define custom calculation functions
        def custom_tortuosity(trajectory_data):
            """Calculate path tortuosity."""
            trajectory = np.asarray(trajectory_data)
            if len(trajectory) < 2:
                return float('inf')
            
            path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
            direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
            
            return path_length / direct_distance if direct_distance > 0 else float('inf')
        
        def custom_exploration_index(trajectory_data):
            """Calculate exploration diversity index."""
            trajectory = np.asarray(trajectory_data)
            if len(trajectory) < 2:
                return 0.0
            
            # Simple exploration index based on visited area
            x_range = np.max(trajectory[:, 0]) - np.min(trajectory[:, 0])
            y_range = np.max(trajectory[:, 1]) - np.min(trajectory[:, 1])
            return float(x_range * y_range)
        
        # Create config with custom functions
        config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std'],
                'custom': ['tortuosity', 'exploration']
            },
            custom_calculations={
                'tortuosity': custom_tortuosity,
                'exploration_index': custom_exploration_index
            }
        )
        
        aggregator = StatsAggregator(config)
        
        # Test with sample data
        episode_data = {
            'episode_id': 1,
            'trajectory': np.array([[0, 0], [1, 0], [2, 1], [3, 2], [1, 3]]),
            'concentrations': np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        }
        
        stats = aggregator.calculate_episode_stats(episode_data)
        
        # Verify custom calculations were applied
        assert 'custom' in stats
        custom_stats = stats['custom']
        
        # Should have results from custom functions
        assert 'tortuosity' in custom_stats
        assert 'exploration_index' in custom_stats
        
        # Verify reasonable values
        if custom_stats['tortuosity'] is not None:
            assert custom_stats['tortuosity'] >= 1.0  # Tortuosity should be >= 1
        
        if custom_stats['exploration_index'] is not None:
            assert custom_stats['exploration_index'] >= 0.0
    
    def test_hierarchical_aggregation(self):
        """Test hierarchical aggregation level configuration."""
        # Test different aggregation levels
        for levels in [['episode'], ['episode', 'run'], ['episode', 'run', 'batch']]:
            config = StatsAggregatorConfig(
                metrics_definitions={'trajectory': ['mean']},
                aggregation_levels=levels
            )
            
            aggregator = StatsAggregator(config)
            assert aggregator.get_aggregation_levels() == levels
    
    def test_metrics_validation(self):
        """Test metrics configuration validation."""
        # Test invalid configuration
        with pytest.raises(ValueError):
            StatsAggregatorConfig(
                metrics_definitions={},  # Empty metrics
                aggregation_levels=['episode']
            )
        
        # Test invalid aggregation levels
        with pytest.raises(ValueError):
            StatsAggregatorConfig(
                metrics_definitions={'trajectory': ['mean']},
                aggregation_levels=[]  # Empty levels
            )
        
        # Test invalid memory limit
        with pytest.raises(ValueError):
            StatsAggregatorConfig(
                metrics_definitions={'trajectory': ['mean']},
                memory_limit_mb=-1  # Negative memory
            )
    
    def test_schema_compliance(self):
        """Test configuration schema compliance and validation."""
        # Test all valid configuration options
        config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std', 'min', 'max'],
                'concentration': ['mean', 'std', 'percentiles'],
                'speed': ['mean', 'total_distance']
            },
            aggregation_levels=['episode', 'run'],
            output_format='json',
            custom_calculations={},
            performance_tracking=True,
            parallel_processing=False,
            memory_limit_mb=256,
            computation_timeout_s=30.0,
            data_validation=True,
            output_validation=True,
            error_handling='warn',
            precision_mode='float64'
        )
        
        # Should create without errors
        aggregator = StatsAggregator(config)
        assert isinstance(aggregator, StatsAggregator)
        
        # Verify all configuration was applied
        assert aggregator.config.performance_tracking is True
        assert aggregator.config.memory_limit_mb == 256
        assert aggregator.config.error_handling == 'warn'
        assert aggregator.config.precision_mode == 'float64'


class TestStatsIntegration:
    """
    Test statistics integration with other system components.
    
    Validates recorder integration, hook system connectivity, environment
    lifecycle integration, multi-component workflow validation, and data
    pipeline integrity per system integration requirements.
    """
    
    @pytest.fixture
    def mock_recorder(self):
        """Create mock recorder for integration testing."""
        recorder = Mock()
        recorder.config = Mock()
        recorder.config.backend = 'parquet'
        recorder.get_performance_metrics.return_value = {
            'average_write_time': 0.015,  # 15ms write time
            'total_records': 100
        }
        return recorder
    
    def test_recorder_integration(self, mock_recorder):
        """Test integration with recorder system."""
        config = StatsAggregatorConfig(
            metrics_definitions={'trajectory': ['mean', 'std']},
            performance_tracking=True
        )
        
        # Create aggregator with recorder
        aggregator = StatsAggregator(config, recorder=mock_recorder)
        
        # Verify recorder was assigned
        assert aggregator.recorder is mock_recorder
        
        # Test statistics calculation with recorder
        episode_data = {
            'episode_id': 1,
            'trajectory': np.random.randn(50, 2),
            'concentrations': np.random.exponential(0.3, 50)
        }
        
        stats = aggregator.calculate_episode_stats(episode_data)
        
        # Should complete successfully with recorder integration
        assert isinstance(stats, dict)
        assert 'episode_id' in stats
    
    def test_hook_system_integration(self):
        """Test integration with hook system for extensions."""
        # Create aggregator with hook-compatible configuration
        config = StatsAggregatorConfig(
            metrics_definitions={'trajectory': ['mean']},
            performance_tracking=True
        )
        
        aggregator = StatsAggregator(config)
        
        # Test that aggregator can be used as a hook
        episode_data = {
            'episode_id': 1,
            'trajectory': np.random.randn(30, 2),
            'concentrations': np.random.exponential(0.3, 30)
        }
        
        # Calculate stats (simulating hook call)
        stats = aggregator.calculate_episode_stats(episode_data)
        
        # Verify hook integration compatibility
        assert isinstance(stats, dict)
        assert 'episode_id' in stats
        assert 'timestamp' in stats
        
        # Test performance metrics access (for hook monitoring)
        metrics = aggregator.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert 'episodes_processed' in metrics
    
    def test_environment_lifecycle(self):
        """Test statistics aggregator through environment lifecycle."""
        config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'total_distance'],
                'concentration': ['mean', 'detection_rate']
            }
        )
        
        aggregator = StatsAggregator(config)
        
        # Simulate environment lifecycle
        episodes_data = []
        
        # Episode 1
        episode1 = {
            'episode_id': 0,
            'trajectory': np.random.randn(40, 2),
            'concentrations': np.random.exponential(0.4, 40),
            'episode_length': 40
        }
        episodes_data.append(episode1)
        stats1 = aggregator.calculate_episode_stats(episode1)
        
        # Episode 2  
        episode2 = {
            'episode_id': 1,
            'trajectory': np.random.randn(60, 2),
            'concentrations': np.random.exponential(0.3, 60),
            'episode_length': 60
        }
        episodes_data.append(episode2)
        stats2 = aggregator.calculate_episode_stats(episode2)
        
        # Run-level aggregation
        run_stats = aggregator.calculate_run_stats(episodes_data)
        
        # Verify lifecycle integration
        assert isinstance(stats1, dict)
        assert isinstance(stats2, dict)
        assert isinstance(run_stats, dict)
        
        # Verify episode tracking
        assert aggregator.episode_count >= 2
        
        # Test reset functionality
        aggregator.reset()
        assert aggregator.episode_count == 0
        assert len(aggregator.run_data) == 0
    
    def test_multi_component_workflow(self, mock_recorder):
        """Test workflow with multiple integrated components."""
        # Create aggregator with full configuration
        config = StatsAggregatorConfig(
            metrics_definitions={
                'trajectory': ['mean', 'std', 'total_distance'],
                'concentration': ['mean', 'detection_rate', 'percentiles'],
                'speed': ['mean', 'total_distance']
            },
            aggregation_levels=['episode', 'run'],
            performance_tracking=True,
            output_format='json'
        )
        
        aggregator = StatsAggregator(config, recorder=mock_recorder)
        
        # Simulate complete workflow
        episodes = []
        for i in range(5):
            episode = {
                'episode_id': i,
                'trajectory': np.random.randn(50, 2),
                'concentrations': np.random.exponential(0.4, 50),
                'speeds': np.random.exponential(0.5, 50),
                'episode_length': 50
            }
            episodes.append(episode)
            
            # Process episode
            episode_stats = aggregator.calculate_episode_stats(episode)
            assert isinstance(episode_stats, dict)
        
        # Process run
        run_stats = aggregator.calculate_run_stats(episodes)
        assert isinstance(run_stats, dict)
        
        # Export summary
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "workflow_summary.json"
            success = aggregator.export_summary(output_path)
            
            if success:
                assert output_path.exists()
        
        # Verify performance metrics
        metrics = aggregator.get_performance_metrics()
        assert metrics['episodes_processed'] >= 5
        assert metrics['runs_processed'] >= 1
    
    def test_data_pipeline_integrity(self):
        """Test data pipeline integrity and validation."""
        config = StatsAggregatorConfig(
            metrics_definitions={'trajectory': ['mean', 'std']},
            data_validation=True,
            output_validation=True
        )
        
        aggregator = StatsAggregator(config)
        
        # Test with valid data
        valid_data = {
            'episode_id': 1,
            'trajectory': np.random.randn(30, 2),
            'concentrations': np.random.exponential(0.3, 30)
        }
        
        # Validate data
        validation = aggregator.validate_data(valid_data)
        assert isinstance(validation, dict)
        assert 'valid' in validation
        
        # Process valid data
        stats = aggregator.calculate_episode_stats(valid_data)
        assert isinstance(stats, dict)
        
        # Test with invalid data
        invalid_data = {
            'episode_id': 2,
            'trajectory': np.array([[np.inf, 0], [0, np.nan]]),  # Invalid values
            'concentrations': np.array([-0.1, 0.2])  # Invalid negative
        }
        
        # Validation should catch issues
        validation = aggregator.validate_data(invalid_data)
        # May be valid=False or handle gracefully depending on implementation
        assert isinstance(validation, dict)


# Standalone test functions for factory and compatibility

def test_stats_aggregator_factory():
    """Test statistics aggregator factory function."""
    from plume_nav_sim.analysis.stats import create_stats_aggregator
    
    # Test with dictionary configuration
    config_dict = {
        'metrics_definitions': {
            'trajectory': ['mean', 'std'],
            'concentration': ['mean']
        },
        'aggregation_levels': ['episode', 'run']
    }
    
    aggregator = create_stats_aggregator(config_dict)
    assert isinstance(aggregator, StatsAggregator)
    assert isinstance(aggregator, StatsAggregatorProtocol)
    
    # Test with StatsAggregatorConfig
    config_obj = StatsAggregatorConfig(**config_dict)
    aggregator2 = create_stats_aggregator(config_obj)
    assert isinstance(aggregator2, StatsAggregator)


def test_stats_backwards_compatibility():
    """Test backwards compatibility with existing interfaces."""
    # Test generate_summary function
    from plume_nav_sim.analysis import generate_summary
    
    config = StatsAggregatorConfig(
        metrics_definitions={'trajectory': ['mean']},
        performance_tracking=True
    )
    aggregator = StatsAggregator(config)
    
    # Create sample episodes data
    episodes_data = [
        {
            'episode_id': 0,
            'trajectory': np.random.randn(20, 2),
            'concentrations': np.random.exponential(0.3, 20)
        },
        {
            'episode_id': 1,
            'trajectory': np.random.randn(25, 2),
            'concentrations': np.random.exponential(0.4, 25)
        }
    ]
    
    # Test summary generation
    summary = generate_summary(aggregator, episodes_data)
    
    # Verify summary structure
    assert isinstance(summary, dict)
    assert 'run_statistics' in summary
    assert 'performance_metrics' in summary
    assert 'episode_count' in summary
    assert summary['episode_count'] == 2
    
    # Test with export
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "test_summary.json")
        summary_with_export = generate_summary(
            aggregator, 
            episodes_data, 
            output_path=output_path
        )
        
        assert isinstance(summary_with_export, dict)
        assert 'export_status' in summary_with_export


if __name__ == "__main__":
    # Run tests with appropriate verbosity and coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        f"--cov=src.plume_nav_sim.analysis",
        "--cov-report=term-missing",
        "--durations=10"
    ])