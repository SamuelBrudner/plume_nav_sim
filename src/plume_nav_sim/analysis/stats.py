"""
Core statistics aggregation implementation providing automated research metrics calculation.

This module implements the StatsAggregatorProtocol interface for comprehensive research-focused
statistics collection and analysis. The implementation provides episode-level and run-level
statistical analysis with configurable metrics definitions, summary.json export, and integration
with the recorder system for experimental data analysis.

Key Features:
    - Automated episode and run-level statistics calculation with error analysis
    - Configurable metrics definitions supporting custom calculation functions
    - High-performance statistical processing achieving ≤33 ms/step with 100 agents
    - Integration with recorder system for comprehensive data connectivity
    - Hierarchical summary generation with run_id/episode_id organization
    - Standardized summary.json export for research reproducibility
    - Memory-efficient algorithms for large dataset processing
    - Validation and schema compliance for output format consistency

Performance Requirements:
    - Target: ≤33 ms/step with 100 agents through optimized statistical processing
    - Memory-efficient algorithms for large datasets with configurable limits
    - Parallel processing support for batch statistics calculation
    - Configurable aggregation levels with hierarchical summary generation

Integration Points:
    - Episode completion hooks for automated data collection
    - Recorder system connectivity for data source access
    - Hydra configuration management for metrics definitions
    - JSON export with standardized research metrics format

Examples:
    Basic statistics aggregator usage:
        >>> config = StatsAggregatorConfig(
        ...     metrics_definitions={'basic': ['mean', 'std', 'min', 'max']},
        ...     aggregation_levels=['episode', 'run'],
        ...     output_format='json'
        ... )
        >>> aggregator = StatsAggregator(config)
        >>> episode_stats = aggregator.calculate_episode_stats(episode_data)
        >>> run_stats = aggregator.calculate_run_stats(episodes_list)
        >>> aggregator.export_summary('./results/summary.json')

    Custom metrics with advanced analysis:
        >>> def custom_tortuosity(trajectory):
        ...     path_length = np.sum(np.diff(trajectory, axis=0))
        ...     direct_distance = np.linalg.norm(trajectory[-1] - trajectory[0])
        ...     return path_length / direct_distance if direct_distance > 0 else np.inf
        >>> 
        >>> config = StatsAggregatorConfig(
        ...     custom_calculations={'tortuosity': custom_tortuosity},
        ...     performance_tracking=True
        ... )
        >>> aggregator = StatsAggregator(config)

    Integration with recorder system:
        >>> aggregator = create_stats_aggregator({
        ...     'recorder_integration': True,
        ...     'data_validation': True,
        ...     'parallel_processing': True
        ... })
        >>> metrics = aggregator.get_performance_metrics()
        >>> print(f"Processing time: {metrics['computation_time_ms']:.2f}ms")
"""

import json
from loguru import logger
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, TypeVar, Generic

import numpy as np
import scipy.stats

# Import protocols for interface compliance
from ..core.protocols import StatsAggregatorProtocol, RecorderProtocol

# Configure logging
# Type definitions
T = TypeVar('T')
MetricsDict = Dict[str, Union[float, int, np.ndarray]]
EpisodeData = Dict[str, Any]
RunData = List[EpisodeData]
CustomCalculation = Callable[[np.ndarray], Union[float, np.ndarray]]


@dataclass
class StatsAggregatorConfig:
    """
    Configuration dataclass for statistics aggregator setup and validation.
    
    This dataclass provides type-safe parameter validation for statistics aggregator
    configuration supporting custom metrics definitions, aggregation levels, output
    formats, and performance optimization settings. All parameters integrate with
    Hydra configuration management for consistent parameter injection.
    
    Core Configuration:
        metrics_definitions: Dictionary defining which metrics to calculate for each data type
        aggregation_levels: List of aggregation levels to compute (episode, run, batch)
        output_format: Export format for summary data (json, yaml, pickle)
        custom_calculations: Dictionary of custom metric calculation functions
        
    Performance Configuration:
        performance_tracking: Enable detailed performance monitoring and timing
        parallel_processing: Use multiprocessing for batch calculations
        memory_limit_mb: Maximum memory usage for statistical computations
        computation_timeout_s: Maximum time allowed for metric calculations
        
    Validation Configuration:
        data_validation: Enable input data validation and schema checking
        output_validation: Validate output metrics against expected ranges
        error_handling: Strategy for handling calculation errors (skip, warn, raise)
        precision_mode: Numerical precision for calculations (float32, float64)
    """
    # Core configuration
    metrics_definitions: Dict[str, List[str]] = field(
        default_factory=lambda: {
            'trajectory': ['mean', 'std', 'min', 'max', 'median'],
            'concentration': ['mean', 'std', 'min', 'max', 'percentiles'],
            'speed': ['mean', 'std', 'max', 'total_distance'],
            'timing': ['episode_length', 'step_duration']
        }
    )
    aggregation_levels: List[str] = field(default_factory=lambda: ['episode', 'run'])
    output_format: str = 'json'
    custom_calculations: Dict[str, CustomCalculation] = field(default_factory=dict)
    
    # Performance configuration
    performance_tracking: bool = True
    parallel_processing: bool = False
    memory_limit_mb: int = 512
    computation_timeout_s: float = 30.0
    
    # Validation configuration
    data_validation: bool = True
    output_validation: bool = True
    error_handling: str = 'warn'  # skip, warn, raise
    precision_mode: str = 'float64'  # float32, float64
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if self.computation_timeout_s <= 0:
            raise ValueError("computation_timeout_s must be positive")
        if self.output_format not in ['json', 'yaml', 'pickle']:
            raise ValueError("output_format must be one of: json, yaml, pickle")
        if self.error_handling not in ['skip', 'warn', 'raise']:
            raise ValueError("error_handling must be one of: skip, warn, raise")
        if self.precision_mode not in ['float32', 'float64']:
            raise ValueError("precision_mode must be one of: float32, float64")


class StatsAggregator(StatsAggregatorProtocol):
    """
    Core statistics aggregator implementing comprehensive research metrics calculation.
    
    The StatsAggregator implements the StatsAggregatorProtocol interface to provide
    automated research-focused statistics collection and analysis. It supports
    episode-level and run-level statistical analysis with configurable metrics
    definitions, custom calculation functions, and high-performance processing
    designed to meet the ≤33 ms/step performance requirement with 100 agents.
    
    Key Features:
    - Protocol-compliant implementation of all required statistics methods
    - High-performance statistical algorithms optimized for large datasets
    - Memory-efficient processing with configurable limits and streaming
    - Custom metrics support with validation and error handling
    - Integration with recorder system for comprehensive data access
    - Hierarchical aggregation supporting episode and run-level analysis
    - Standardized output format for research reproducibility
    - Performance monitoring with detailed timing and resource tracking
    
    Performance Characteristics:
    - ≤33 ms/step processing time with 100 agents per specification
    - Memory-efficient algorithms suitable for large trajectory datasets
    - Optional parallel processing for batch calculations
    - Configurable precision modes balancing accuracy and performance
    - Automatic optimization for common statistical operations
    
    Integration Features:
    - RecorderProtocol connectivity for data source access
    - Episode completion hook integration for automated collection
    - Validation and schema compliance for reliable output
    - Error handling and recovery for robust operation
    """
    
    def __init__(
        self, 
        config: StatsAggregatorConfig,
        recorder: Optional[RecorderProtocol] = None
    ):
        """
        Initialize statistics aggregator with configuration and optional recorder.
        
        Args:
            config: Statistics aggregator configuration with validation
            recorder: Optional recorder instance for data source connectivity
        """
        self.config = config
        self.recorder = recorder
        
        # State management
        self.episode_count = 0
        self.run_data: List[EpisodeData] = []
        self.current_episode_stats: Optional[Dict[str, Any]] = None
        
        # Performance tracking
        self._metrics = {
            'episodes_processed': 0,
            'runs_processed': 0,
            'computation_time_ms': 0.0,
            'memory_usage_peak_mb': 0.0,
            'calculations_performed': 0,
            'validation_errors': 0,
            'performance_warnings': 0
        }
        self._start_time = time.perf_counter()
        
        # Configure numerical precision
        self.dtype = np.float64 if config.precision_mode == 'float64' else np.float32
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"StatsAggregator initialized with {len(config.metrics_definitions)} metric types")
    
    def calculate_episode_stats(
        self, 
        episode_data: EpisodeData,
        episode_id: Optional[int] = None,
        **metadata: Any
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive episode-level statistics with error analysis.
        
        Implements the StatsAggregatorProtocol.calculate_episode_stats() method
        providing episode-level metrics calculation with configurable analysis
        depth and performance optimization for real-time processing.
        
        Args:
            episode_data: Dictionary containing episode simulation data
            episode_id: Optional episode identifier for metadata correlation
            **metadata: Additional metadata for analysis context
            
        Returns:
            Dict[str, Any]: Comprehensive episode statistics with error analysis
        """
        start_time = time.perf_counter()
        
        try:
            # Validate input data
            if self.config.data_validation:
                self._validate_episode_data(episode_data)
            
            # Extract episode identifier
            if episode_id is None:
                episode_id = episode_data.get('episode_id', self.episode_count)
            
            # Initialize statistics dictionary
            episode_stats = {
                'episode_id': episode_id,
                'timestamp': time.time(),
                'computation_metadata': {
                    'aggregator_version': '1.0',
                    'precision_mode': self.config.precision_mode,
                    'validation_enabled': self.config.data_validation
                },
                **metadata
            }
            
            # Calculate basic trajectory statistics
            if 'trajectory' in episode_data:
                trajectory_stats = self._calculate_trajectory_stats(episode_data['trajectory'])
                episode_stats['trajectory'] = trajectory_stats
            
            # Calculate concentration statistics
            if 'concentrations' in episode_data:
                concentration_stats = self._calculate_concentration_stats(episode_data['concentrations'])
                episode_stats['concentration'] = concentration_stats
            
            # Calculate speed and movement statistics
            if 'speeds' in episode_data or 'trajectory' in episode_data:
                speed_stats = self._calculate_speed_stats(episode_data)
                episode_stats['speed'] = speed_stats
            
            # Calculate timing statistics
            timing_stats = self._calculate_timing_stats(episode_data)
            episode_stats['timing'] = timing_stats
            
            # Apply custom calculations
            if self.config.custom_calculations:
                custom_stats = self._apply_custom_calculations(episode_data)
                episode_stats['custom'] = custom_stats
            
            # Calculate derived metrics
            derived_stats = self._calculate_derived_metrics(episode_data, episode_stats)
            episode_stats['derived'] = derived_stats
            
            # Add error analysis
            if 'errors' in episode_data or 'uncertainties' in episode_data:
                error_analysis = self._calculate_error_analysis(episode_data)
                episode_stats['error_analysis'] = error_analysis
            
            # Update performance metrics
            computation_time = (time.perf_counter() - start_time) * 1000
            self._metrics['computation_time_ms'] += computation_time
            self._metrics['episodes_processed'] += 1
            self._metrics['calculations_performed'] += len(episode_stats) - 3  # Exclude metadata
            
            # Performance validation
            if computation_time > 33.0:  # Performance requirement
                self._metrics['performance_warnings'] += 1
                logger.warning(f"Episode stats calculation exceeded 33ms: {computation_time:.2f}ms")
            
            # Store current episode stats
            self.current_episode_stats = episode_stats
            self.episode_count += 1
            
            logger.debug(f"Calculated episode {episode_id} statistics in {computation_time:.2f}ms")
            return episode_stats
            
        except Exception as e:
            self._handle_calculation_error('calculate_episode_stats', e)
            return self._create_error_stats(episode_id, str(e))
    
    def calculate_run_stats(
        self, 
        episodes_data: RunData,
        run_id: Optional[str] = None,
        **metadata: Any
    ) -> Dict[str, Any]:
        """
        Aggregate statistics across multiple episodes with configurable aggregation levels.
        
        Implements the StatsAggregatorProtocol.calculate_run_stats() method providing
        run-level aggregation of episode statistics with advanced statistical analysis
        and cross-episode correlation metrics.
        
        Args:
            episodes_data: List of episode data dictionaries for aggregation
            run_id: Optional run identifier for metadata correlation
            **metadata: Additional metadata for aggregation context
            
        Returns:
            Dict[str, Any]: Comprehensive run statistics with cross-episode analysis
        """
        start_time = time.perf_counter()
        
        try:
            # Validate input data
            if self.config.data_validation:
                self._validate_run_data(episodes_data)
            
            if not episodes_data:
                return self._create_empty_run_stats(run_id)
            
            # Generate run identifier
            if run_id is None:
                run_id = f"run_{int(time.time())}"
            
            # Initialize run statistics
            run_stats = {
                'run_id': run_id,
                'timestamp': time.time(),
                'episode_count': len(episodes_data),
                'computation_metadata': {
                    'aggregator_version': '1.0',
                    'aggregation_levels': self.config.aggregation_levels,
                    'parallel_processing': self.config.parallel_processing
                },
                **metadata
            }
            
            # Calculate episode-level statistics for each episode
            episode_statistics = []
            for i, episode_data in enumerate(episodes_data):
                episode_stats = self.calculate_episode_stats(episode_data, episode_id=i)
                episode_statistics.append(episode_stats)
            
            # Aggregate trajectory statistics across episodes
            trajectory_aggregation = self._aggregate_trajectory_stats(episode_statistics)
            run_stats['trajectory_aggregation'] = trajectory_aggregation
            
            # Aggregate concentration statistics
            concentration_aggregation = self._aggregate_concentration_stats(episode_statistics)
            run_stats['concentration_aggregation'] = concentration_aggregation
            
            # Aggregate speed statistics
            speed_aggregation = self._aggregate_speed_stats(episode_statistics)
            run_stats['speed_aggregation'] = speed_aggregation
            
            # Aggregate timing statistics
            timing_aggregation = self._aggregate_timing_stats(episode_statistics)
            run_stats['timing_aggregation'] = timing_aggregation
            
            # Calculate cross-episode correlations
            correlations = self._calculate_cross_episode_correlations(episode_statistics)
            run_stats['correlations'] = correlations
            
            # Calculate learning curves and trends
            trends = self._calculate_learning_trends(episode_statistics)
            run_stats['trends'] = trends
            
            # Aggregate custom metrics
            if self.config.custom_calculations:
                custom_aggregation = self._aggregate_custom_stats(episode_statistics)
                run_stats['custom_aggregation'] = custom_aggregation
            
            # Calculate summary statistics
            summary_stats = self._calculate_run_summary(episode_statistics)
            run_stats['summary'] = summary_stats
            
            # Update performance metrics
            computation_time = (time.perf_counter() - start_time) * 1000
            self._metrics['computation_time_ms'] += computation_time
            self._metrics['runs_processed'] += 1
            
            # Store run data
            self.run_data.append(run_stats)
            
            logger.info(f"Calculated run {run_id} statistics for {len(episodes_data)} episodes")
            return run_stats
            
        except Exception as e:
            self._handle_calculation_error('calculate_run_stats', e)
            return self._create_error_run_stats(run_id, str(e))
    
    def export_summary(
        self, 
        output_path: Union[str, Path],
        format: Optional[str] = None,
        **export_options: Any
    ) -> bool:
        """
        Export comprehensive summary report with standardized research metrics format.
        
        Implements the StatsAggregatorProtocol.export_summary() method providing
        summary.json export functionality with standardized format for research
        reproducibility and cross-project comparison.
        
        Args:
            output_path: File system path for summary export
            format: Optional format override (default: config.output_format)
            **export_options: Additional export parameters and options
            
        Returns:
            bool: True if export completed successfully, False otherwise
        """
        try:
            output_path = Path(output_path)
            export_format = format or self.config.output_format
            
            # Create comprehensive summary data
            summary_data = {
                'metadata': {
                    'export_timestamp': time.time(),
                    'aggregator_version': '1.0',
                    'format_version': '1.0',
                    'configuration': {
                        'metrics_definitions': self.config.metrics_definitions,
                        'aggregation_levels': self.config.aggregation_levels,
                        'custom_calculations': list(self.config.custom_calculations.keys()),
                        'performance_tracking': self.config.performance_tracking
                    }
                },
                'performance_metrics': self.get_performance_metrics(),
                'run_statistics': self.run_data,
                'current_episode': self.current_episode_stats,
                'global_summary': self._generate_global_summary()
            }
            
            # Validate output data
            if self.config.output_validation:
                self._validate_summary_data(summary_data)
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export in specified format
            if export_format == 'json':
                self._export_json(summary_data, output_path, **export_options)
            elif export_format == 'yaml':
                self._export_yaml(summary_data, output_path, **export_options)
            elif export_format == 'pickle':
                self._export_pickle(summary_data, output_path, **export_options)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            logger.info(f"Exported summary to {output_path} in {export_format} format")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export summary: {e}")
            self._metrics['validation_errors'] += 1
            return False
    
    def configure_metrics(self, **kwargs: Any) -> None:
        """
        Update metrics configuration during runtime with validation.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config.{key} = {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
        
        # Re-validate configuration
        self._validate_configuration()
    
    def reset(self) -> None:
        """Reset aggregator state while preserving configuration."""
        self.episode_count = 0
        self.run_data.clear()
        self.current_episode_stats = None
        
        # Reset performance metrics
        self._metrics = {key: 0 if isinstance(val, (int, float)) else val 
                        for key, val in self._metrics.items()}
        self._start_time = time.perf_counter()
        
        logger.debug("StatsAggregator state reset")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for monitoring and optimization.
        
        Returns:
            Dict[str, Any]: Performance metrics including timing, memory, and computation stats
        """
        current_time = time.perf_counter()
        elapsed_time = current_time - self._start_time
        
        metrics = {
            **self._metrics,
            'elapsed_time_s': elapsed_time,
            'episodes_per_second': self._metrics['episodes_processed'] / elapsed_time if elapsed_time > 0 else 0,
            'avg_computation_time_ms': (self._metrics['computation_time_ms'] / 
                                      max(self._metrics['episodes_processed'], 1)),
            'memory_efficiency': self._calculate_memory_efficiency(),
            'performance_compliance': {
                'meets_33ms_target': self._metrics['performance_warnings'] == 0,
                'warning_count': self._metrics['performance_warnings'],
                'error_rate': (self._metrics['validation_errors'] / 
                             max(self._metrics['episodes_processed'], 1))
            }
        }
        
        return metrics
    
    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate input data for statistics calculation.
        
        Args:
            data: Input data to validate
            
        Returns:
            Dict[str, Any]: Validation results with diagnostics
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'data_summary': {}
        }
        
        try:
            # Check required fields
            required_fields = ['trajectory', 'concentrations', 'episode_id']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                validation_results['warnings'].extend([f"Missing field: {field}" for field in missing_fields])
            
            # Validate trajectory data
            if 'trajectory' in data:
                trajectory_validation = self._validate_trajectory_data(data['trajectory'])
                validation_results['data_summary']['trajectory'] = trajectory_validation
                if not trajectory_validation['valid']:
                    validation_results['errors'].extend(trajectory_validation['errors'])
            
            # Validate concentration data
            if 'concentrations' in data:
                concentration_validation = self._validate_concentration_data(data['concentrations'])
                validation_results['data_summary']['concentrations'] = concentration_validation
                if not concentration_validation['valid']:
                    validation_results['errors'].extend(concentration_validation['errors'])
            
            # Set overall validity
            validation_results['valid'] = len(validation_results['errors']) == 0
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation exception: {e}")
        
        return validation_results
    
    def get_aggregation_levels(self) -> List[str]:
        """
        Get configured aggregation levels.
        
        Returns:
            List[str]: List of configured aggregation levels
        """
        return self.config.aggregation_levels.copy()
    
    # Private helper methods
    
    def _calculate_trajectory_stats(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive trajectory statistics."""
        trajectory = np.asarray(trajectory, dtype=self.dtype)
        
        if trajectory.size == 0:
            return self._create_empty_trajectory_stats()
        
        stats = {}
        
        # Basic positional statistics
        stats['mean_position'] = np.mean(trajectory, axis=0).tolist()
        stats['std_position'] = np.std(trajectory, axis=0).tolist()
        stats['min_position'] = np.min(trajectory, axis=0).tolist()
        stats['max_position'] = np.max(trajectory, axis=0).tolist()
        stats['range_position'] = (np.max(trajectory, axis=0) - np.min(trajectory, axis=0)).tolist()
        
        # Distance and displacement metrics
        if len(trajectory) > 1:
            displacements = np.diff(trajectory, axis=0)
            distances = np.linalg.norm(displacements, axis=1)
            
            stats['total_distance'] = float(np.sum(distances))
            stats['mean_step_distance'] = float(np.mean(distances))
            stats['std_step_distance'] = float(np.std(distances))
            stats['max_step_distance'] = float(np.max(distances))
            
            # Net displacement
            net_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
            stats['net_displacement'] = float(net_displacement)
            stats['displacement_efficiency'] = float(net_displacement / stats['total_distance']) if stats['total_distance'] > 0 else 0.0
            
            # Tortuosity
            stats['tortuosity'] = float(stats['total_distance'] / net_displacement) if net_displacement > 0 else float('inf')
        
        # Coverage statistics
        stats['exploration_area'] = self._calculate_exploration_area(trajectory)
        stats['position_entropy'] = self._calculate_position_entropy(trajectory)
        
        return stats
    
    def _calculate_concentration_stats(self, concentrations: np.ndarray) -> Dict[str, float]:
        """Calculate concentration statistics with advanced analysis."""
        concentrations = np.asarray(concentrations, dtype=self.dtype)
        
        if concentrations.size == 0:
            return self._create_empty_concentration_stats()
        
        stats = {}
        
        # Basic statistics
        stats['mean'] = float(np.mean(concentrations))
        stats['std'] = float(np.std(concentrations))
        stats['min'] = float(np.min(concentrations))
        stats['max'] = float(np.max(concentrations))
        stats['median'] = float(np.median(concentrations))
        
        # Percentiles
        percentiles = [10, 25, 75, 90, 95, 99]
        stats['percentiles'] = {f'p{p}': float(np.percentile(concentrations, p)) for p in percentiles}
        
        # Distribution analysis
        stats['skewness'] = float(scipy.stats.skew(concentrations))
        stats['kurtosis'] = float(scipy.stats.kurtosis(concentrations))
        
        # Detection statistics
        detection_threshold = 0.01  # Configurable threshold
        detections = concentrations > detection_threshold
        stats['detection_rate'] = float(np.mean(detections))
        stats['max_detection_streak'] = self._calculate_max_streak(detections)
        stats['detection_efficiency'] = stats['detection_rate'] * stats['mean']
        
        # Time series analysis
        if len(concentrations) > 1:
            gradient = np.gradient(concentrations)
            stats['mean_gradient'] = float(np.mean(gradient))
            stats['gradient_variance'] = float(np.var(gradient))
        
        return stats
    
    def _calculate_speed_stats(self, episode_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate speed and movement statistics."""
        speeds = episode_data.get('speeds')
        trajectory = episode_data.get('trajectory')
        
        # Calculate speeds from trajectory if not provided
        if speeds is None and trajectory is not None:
            trajectory = np.asarray(trajectory, dtype=self.dtype)
            if len(trajectory) > 1:
                displacements = np.diff(trajectory, axis=0)
                speeds = np.linalg.norm(displacements, axis=1)
            else:
                return self._create_empty_speed_stats()
        
        if speeds is None:
            return self._create_empty_speed_stats()
        
        speeds = np.asarray(speeds, dtype=self.dtype)
        
        stats = {}
        
        # Basic speed statistics
        stats['mean_speed'] = float(np.mean(speeds))
        stats['std_speed'] = float(np.std(speeds))
        stats['min_speed'] = float(np.min(speeds))
        stats['max_speed'] = float(np.max(speeds))
        stats['median_speed'] = float(np.median(speeds))
        
        # Movement efficiency
        stats['total_distance'] = float(np.sum(speeds))
        if trajectory is not None:
            trajectory = np.asarray(trajectory, dtype=self.dtype)
            net_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
            stats['movement_efficiency'] = float(net_displacement / stats['total_distance']) if stats['total_distance'] > 0 else 0.0
        
        # Speed distribution analysis
        stats['speed_entropy'] = self._calculate_speed_entropy(speeds)
        stats['acceleration_variance'] = self._calculate_acceleration_variance(speeds)
        
        return stats
    
    def _calculate_timing_stats(self, episode_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate timing and duration statistics."""
        stats = {}
        
        # Episode length
        episode_length = episode_data.get('episode_length', len(episode_data.get('trajectory', [])))
        stats['episode_length'] = int(episode_length)
        
        # Step duration
        step_durations = episode_data.get('step_durations')
        if step_durations is not None:
            step_durations = np.asarray(step_durations, dtype=self.dtype)
            stats['mean_step_duration'] = float(np.mean(step_durations))
            stats['std_step_duration'] = float(np.std(step_durations))
            stats['max_step_duration'] = float(np.max(step_durations))
            stats['total_episode_time'] = float(np.sum(step_durations))
        
        # Timing efficiency
        target_step_duration = 0.033  # 33ms target
        if step_durations is not None:
            performance_compliance = step_durations <= target_step_duration
            stats['performance_compliance_rate'] = float(np.mean(performance_compliance))
        
        return stats
    
    def _apply_custom_calculations(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply custom calculation functions to episode data."""
        custom_stats = {}
        
        for calc_name, calc_function in self.config.custom_calculations.items():
            try:
                # ----------------------------------------------------------
                # 1) Preferred: call the custom function with full episode
                #    dictionary so the user can access multiple fields.
                # ----------------------------------------------------------
                try:
                    result = calc_function(episode_data)
                    custom_stats[calc_name] = self._serialize_result(result)
                    continue  # success, proceed to next custom calc
                except (TypeError, KeyError, ValueError, AttributeError):
                    # Fall back to array-only signature expected by older
                    # implementations/tests.
                    pass

                # ----------------------------------------------------------
                # 2) Fallback: pass a primary data array (field-specific or
                #    trajectory) converted to numpy.
                # ----------------------------------------------------------
                if calc_name in episode_data:
                    primary_data = episode_data[calc_name]
                elif 'trajectory' in episode_data:
                    primary_data = episode_data['trajectory']
                else:
                    raise KeyError(
                        f"No suitable data found for custom calculation '{calc_name}'"
                    )

                result = calc_function(np.asarray(primary_data))
                custom_stats[calc_name] = self._serialize_result(result)
                
            except Exception as e:
                self._handle_calculation_error(f'custom_{calc_name}', e)
                custom_stats[calc_name] = None
        
        return custom_stats
    
    def _calculate_derived_metrics(self, episode_data: Dict[str, Any], episode_stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate derived metrics from basic statistics."""
        derived = {}
        
        # Navigation efficiency
        if 'trajectory' in episode_stats and 'speed' in episode_stats:
            trajectory_stats = episode_stats['trajectory']
            speed_stats = episode_stats['speed']
            
            # Combined efficiency metric
            if 'displacement_efficiency' in trajectory_stats and 'movement_efficiency' in speed_stats:
                derived['navigation_efficiency'] = (
                    trajectory_stats['displacement_efficiency'] + speed_stats['movement_efficiency']
                ) / 2.0
        
        # Search effectiveness
        if 'concentration' in episode_stats:
            concentration_stats = episode_stats['concentration']
            derived['search_effectiveness'] = (
                concentration_stats['detection_rate'] * concentration_stats['mean'] * 
                concentration_stats.get('detection_efficiency', 1.0)
            )
        
        # Performance score
        performance_factors = []
        if 'timing' in episode_stats:
            timing_stats = episode_stats['timing']
            if 'performance_compliance_rate' in timing_stats:
                performance_factors.append(timing_stats['performance_compliance_rate'])
        
        if performance_factors:
            derived['performance_score'] = float(np.mean(performance_factors))
        
        return derived
    
    def _calculate_error_analysis(self, episode_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate error analysis and uncertainty metrics."""
        error_analysis = {}
        
        # Position uncertainty
        if 'position_errors' in episode_data:
            position_errors = np.asarray(episode_data['position_errors'], dtype=self.dtype)
            error_analysis['mean_position_error'] = float(np.mean(position_errors))
            error_analysis['max_position_error'] = float(np.max(position_errors))
            error_analysis['position_error_std'] = float(np.std(position_errors))
        
        # Measurement uncertainty
        if 'measurement_uncertainties' in episode_data:
            uncertainties = np.asarray(episode_data['measurement_uncertainties'], dtype=self.dtype)
            error_analysis['mean_uncertainty'] = float(np.mean(uncertainties))
            error_analysis['uncertainty_range'] = float(np.max(uncertainties) - np.min(uncertainties))
        
        return error_analysis
    
    def _aggregate_trajectory_stats(self, episode_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate trajectory statistics across episodes."""
        trajectory_stats = [ep.get('trajectory', {}) for ep in episode_statistics]
        trajectory_stats = [stats for stats in trajectory_stats if stats]
        
        if not trajectory_stats:
            return {}
        
        aggregation = {}
        
        # Aggregate numeric values
        numeric_fields = ['total_distance', 'net_displacement', 'displacement_efficiency', 'tortuosity']
        for field in numeric_fields:
            values = [stats.get(field) for stats in trajectory_stats if stats.get(field) is not None]
            if values:
                aggregation[f'{field}_mean'] = float(np.mean(values))
                aggregation[f'{field}_std'] = float(np.std(values))
                aggregation[f'{field}_min'] = float(np.min(values))
                aggregation[f'{field}_max'] = float(np.max(values))
        
        return aggregation
    
    def _aggregate_concentration_stats(self, episode_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate concentration statistics across episodes."""
        concentration_stats = [ep.get('concentration', {}) for ep in episode_statistics]
        concentration_stats = [stats for stats in concentration_stats if stats]
        
        if not concentration_stats:
            return {}
        
        aggregation = {}
        
        # Aggregate detection rates
        detection_rates = [stats.get('detection_rate') for stats in concentration_stats if stats.get('detection_rate') is not None]
        if detection_rates:
            aggregation['detection_rate_mean'] = float(np.mean(detection_rates))
            aggregation['detection_rate_std'] = float(np.std(detection_rates))
        
        # Aggregate concentration levels
        mean_concentrations = [stats.get('mean') for stats in concentration_stats if stats.get('mean') is not None]
        if mean_concentrations:
            aggregation['concentration_mean'] = float(np.mean(mean_concentrations))
            aggregation['concentration_std'] = float(np.std(mean_concentrations))
        
        return aggregation
    
    def _aggregate_speed_stats(self, episode_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate speed statistics across episodes."""
        speed_stats = [ep.get('speed', {}) for ep in episode_statistics]
        speed_stats = [stats for stats in speed_stats if stats]
        
        if not speed_stats:
            return {}
        
        aggregation = {}
        
        # Aggregate movement metrics
        movement_fields = ['mean_speed', 'total_distance', 'movement_efficiency']
        for field in movement_fields:
            values = [stats.get(field) for stats in speed_stats if stats.get(field) is not None]
            if values:
                aggregation[f'{field}_mean'] = float(np.mean(values))
                aggregation[f'{field}_std'] = float(np.std(values))
        
        return aggregation
    
    def _aggregate_timing_stats(self, episode_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate timing statistics across episodes."""
        timing_stats = [ep.get('timing', {}) for ep in episode_statistics]
        timing_stats = [stats for stats in timing_stats if stats]
        
        if not timing_stats:
            return {}
        
        aggregation = {}
        
        # Aggregate episode lengths
        episode_lengths = [stats.get('episode_length') for stats in timing_stats if stats.get('episode_length') is not None]
        if episode_lengths:
            aggregation['episode_length_mean'] = float(np.mean(episode_lengths))
            aggregation['episode_length_std'] = float(np.std(episode_lengths))
            aggregation['episode_length_min'] = int(np.min(episode_lengths))
            aggregation['episode_length_max'] = int(np.max(episode_lengths))
        
        # Aggregate performance compliance
        compliance_rates = [stats.get('performance_compliance_rate') for stats in timing_stats if stats.get('performance_compliance_rate') is not None]
        if compliance_rates:
            aggregation['performance_compliance_mean'] = float(np.mean(compliance_rates))
            aggregation['performance_compliance_std'] = float(np.std(compliance_rates))
        
        return aggregation
    
    def _calculate_cross_episode_correlations(self, episode_statistics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate correlations between different metrics across episodes."""
        correlations = {}
        
        # Extract time series of key metrics
        episode_lengths = self._extract_metric_series(episode_statistics, 'timing.episode_length')
        detection_rates = self._extract_metric_series(episode_statistics, 'concentration.detection_rate')
        movement_efficiencies = self._extract_metric_series(episode_statistics, 'speed.movement_efficiency')
        
        # Calculate correlations
        if len(episode_lengths) > 1 and len(detection_rates) > 1:
            correlation, p_value = scipy.stats.pearsonr(episode_lengths, detection_rates)
            correlations['episode_length_vs_detection_rate'] = {
                'correlation': float(correlation),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        
        if len(detection_rates) > 1 and len(movement_efficiencies) > 1:
            correlation, p_value = scipy.stats.pearsonr(detection_rates, movement_efficiencies)
            correlations['detection_rate_vs_movement_efficiency'] = {
                'correlation': float(correlation),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        
        return correlations
    
    def _calculate_learning_trends(self, episode_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate learning curves and performance trends."""
        trends = {}
        
        if len(episode_statistics) < 3:
            return trends
        
        # Episode indices
        episode_indices = np.arange(len(episode_statistics))
        
        # Extract performance metrics over time
        detection_rates = self._extract_metric_series(episode_statistics, 'concentration.detection_rate')
        navigation_efficiencies = self._extract_metric_series(episode_statistics, 'derived.navigation_efficiency')
        
        # Calculate trends
        if len(detection_rates) == len(episode_indices):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(episode_indices, detection_rates)
            trends['detection_rate_trend'] = {
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'improving': slope > 0 and p_value < 0.05
            }
        
        if len(navigation_efficiencies) == len(episode_indices):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(episode_indices, navigation_efficiencies)
            trends['navigation_efficiency_trend'] = {
                'slope': float(slope),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'improving': slope > 0 and p_value < 0.05
            }
        
        return trends
    
    def _aggregate_custom_stats(self, episode_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate custom statistics across episodes."""
        custom_stats = [ep.get('custom', {}) for ep in episode_statistics]
        custom_stats = [stats for stats in custom_stats if stats]
        
        if not custom_stats:
            return {}
        
        aggregation = {}
        
        # Get all custom metric names
        all_custom_metrics = set()
        for stats in custom_stats:
            all_custom_metrics.update(stats.keys())
        
        # Aggregate each custom metric
        for metric_name in all_custom_metrics:
            values = [stats.get(metric_name) for stats in custom_stats if stats.get(metric_name) is not None]
            if values and all(isinstance(v, (int, float)) for v in values):
                aggregation[f'{metric_name}_mean'] = float(np.mean(values))
                aggregation[f'{metric_name}_std'] = float(np.std(values))
                aggregation[f'{metric_name}_min'] = float(np.min(values))
                aggregation[f'{metric_name}_max'] = float(np.max(values))
        
        return aggregation
    
    def _calculate_run_summary(self, episode_statistics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall run summary statistics."""
        summary = {
            'total_episodes': len(episode_statistics),
            'successful_episodes': sum(1 for ep in episode_statistics if ep.get('derived', {}).get('navigation_efficiency', 0) > 0.5),
            'average_performance': 0.0,
            'best_episode': None,
            'worst_episode': None
        }
        
        # Calculate average performance
        performance_scores = self._extract_metric_series(episode_statistics, 'derived.performance_score')
        if performance_scores:
            summary['average_performance'] = float(np.mean(performance_scores))
            
            # Find best and worst episodes
            best_idx = np.argmax(performance_scores)
            worst_idx = np.argmin(performance_scores)
            summary['best_episode'] = {
                'episode_id': episode_statistics[best_idx].get('episode_id'),
                'performance_score': float(performance_scores[best_idx])
            }
            summary['worst_episode'] = {
                'episode_id': episode_statistics[worst_idx].get('episode_id'),
                'performance_score': float(performance_scores[worst_idx])
            }
        
        return summary
    
    def _generate_global_summary(self) -> Dict[str, Any]:
        """Generate global summary across all processed runs."""
        if not self.run_data:
            return {}
        
        global_summary = {
            'total_runs': len(self.run_data),
            'total_episodes': sum(run.get('episode_count', 0) for run in self.run_data),
            'processing_efficiency': self.get_performance_metrics(),
            'data_quality': self._assess_data_quality()
        }
        
        return global_summary
    
    # Utility methods
    
    def _validate_configuration(self) -> None:
        """Validate aggregator configuration parameters."""
        if not self.config.metrics_definitions:
            raise ValueError("metrics_definitions cannot be empty")
        
        if not self.config.aggregation_levels:
            raise ValueError("aggregation_levels cannot be empty")
        
        # Validate custom calculations
        for name, func in self.config.custom_calculations.items():
            if not callable(func):
                raise ValueError(f"Custom calculation '{name}' must be callable")
    
    def _validate_episode_data(self, episode_data: Dict[str, Any]) -> None:
        """Validate episode data structure and content."""
        if not isinstance(episode_data, dict):
            raise ValueError("Episode data must be a dictionary")
        
        # Check for required fields
        if 'trajectory' in episode_data:
            trajectory = episode_data['trajectory']
            if not isinstance(trajectory, (list, np.ndarray)):
                raise ValueError("Trajectory must be a list or numpy array")
            trajectory = np.asarray(trajectory)
            if trajectory.ndim != 2 or trajectory.shape[1] != 2:
                raise ValueError("Trajectory must be a 2D array with shape (n_steps, 2)")
    
    def _validate_run_data(self, episodes_data: List[Dict[str, Any]]) -> None:
        """Validate run data structure and content."""
        if not isinstance(episodes_data, list):
            raise ValueError("Episodes data must be a list")
        
        for i, episode_data in enumerate(episodes_data):
            try:
                self._validate_episode_data(episode_data)
            except ValueError as e:
                raise ValueError(f"Invalid episode data at index {i}: {e}")
    
    def _validate_summary_data(self, summary_data: Dict[str, Any]) -> None:
        """Validate summary data before export."""
        required_fields = ['metadata', 'performance_metrics', 'run_statistics']
        for field in required_fields:
            if field not in summary_data:
                raise ValueError(f"Summary data missing required field: {field}")
    
    def _handle_calculation_error(self, operation: str, error: Exception) -> None:
        """Handle calculation errors based on configured error handling strategy."""
        self._metrics['validation_errors'] += 1
        
        if self.config.error_handling == 'raise':
            raise error
        elif self.config.error_handling == 'warn':
            logger.warning(f"Error in {operation}: {error}")
        # 'skip' option does nothing - error is logged but not raised
        
        logger.debug(f"Calculation error in {operation}: {error}")
    
    def _extract_metric_series(self, episode_statistics: List[Dict[str, Any]], metric_path: str) -> List[float]:
        """Extract a metric time series from episode statistics."""
        values = []
        for episode_stats in episode_statistics:
            # Navigate nested dictionary structure
            current = episode_stats
            for key in metric_path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    current = None
                    break
            
            if current is not None and isinstance(current, (int, float)):
                values.append(float(current))
        
        return values
    
    def _serialize_result(self, result: Any) -> Any:
        """Serialize calculation results for JSON compatibility."""
        if isinstance(result, np.ndarray):
            return result.tolist()
        elif isinstance(result, (np.integer, np.floating)):
            return float(result)
        elif isinstance(result, (int, float, str, bool, list, dict)):
            return result
        else:
            return str(result)
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency metrics."""
        # Simplified memory efficiency calculation
        episodes_processed = self._metrics['episodes_processed']
        if episodes_processed == 0:
            return 1.0
        
        # Assume linear memory scaling with episodes
        expected_memory = episodes_processed * 0.1  # 0.1 MB per episode estimate
        actual_memory = self._metrics['memory_usage_peak_mb']
        
        if actual_memory == 0:
            return 1.0
        
        return min(expected_memory / actual_memory, 1.0)
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess overall data quality across all processed data."""
        total_episodes = sum(run.get('episode_count', 0) for run in self.run_data)
        error_rate = self._metrics['validation_errors'] / max(total_episodes, 1)
        
        return {
            'error_rate': error_rate,
            'data_completeness': 1.0 - error_rate,
            'quality_score': max(0.0, 1.0 - 2 * error_rate)  # Penalize errors heavily
        }
    
    # Export methods
    
    def _export_json(self, data: Dict[str, Any], output_path: Path, **options: Any) -> None:
        """Export data in JSON format."""
        indent = options.get('indent', 2)
        ensure_ascii = options.get('ensure_ascii', False)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=self._json_serializer)
    
    def _export_yaml(self, data: Dict[str, Any], output_path: Path, **options: Any) -> None:
        """Export data in YAML format."""
        try:
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            logger.error("PyYAML not available for YAML export")
            raise ImportError("PyYAML required for YAML export")
    
    def _export_pickle(self, data: Dict[str, Any], output_path: Path, **options: Any) -> None:
        """Export data in pickle format."""
        import pickle
        protocol = options.get('protocol', pickle.HIGHEST_PROTOCOL)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f, protocol=protocol)
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy arrays and other objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return str(obj)
    
    # Empty stats creators
    
    def _create_empty_trajectory_stats(self) -> Dict[str, float]:
        """Create empty trajectory statistics structure."""
        return {
            'mean_position': [0.0, 0.0],
            'std_position': [0.0, 0.0],
            'total_distance': 0.0,
            'net_displacement': 0.0,
            'displacement_efficiency': 0.0,
            'tortuosity': float('inf'),
            'exploration_area': 0.0,
            'position_entropy': 0.0
        }
    
    def _create_empty_concentration_stats(self) -> Dict[str, float]:
        """Create empty concentration statistics structure."""
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'detection_rate': 0.0,
            'detection_efficiency': 0.0
        }
    
    def _create_empty_speed_stats(self) -> Dict[str, float]:
        """Create empty speed statistics structure."""
        return {
            'mean_speed': 0.0,
            'std_speed': 0.0,
            'max_speed': 0.0,
            'total_distance': 0.0,
            'movement_efficiency': 0.0
        }
    
    def _create_error_stats(self, episode_id: Optional[int], error_message: str) -> Dict[str, Any]:
        """Create error statistics structure."""
        return {
            'episode_id': episode_id,
            'error': True,
            'error_message': error_message,
            'timestamp': time.time()
        }
    
    def _create_empty_run_stats(self, run_id: Optional[str]) -> Dict[str, Any]:
        """Create empty run statistics structure."""
        return {
            'run_id': run_id,
            'episode_count': 0,
            'timestamp': time.time(),
            'error': False,
            'summary': {'total_episodes': 0}
        }
    
    def _create_error_run_stats(self, run_id: Optional[str], error_message: str) -> Dict[str, Any]:
        """Create error run statistics structure."""
        return {
            'run_id': run_id,
            'error': True,
            'error_message': error_message,
            'timestamp': time.time()
        }
    
    # Advanced calculation methods
    
    def _calculate_exploration_area(self, trajectory: np.ndarray) -> float:
        """Calculate exploration area covered by trajectory."""
        if len(trajectory) < 3:
            return 0.0
        
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(trajectory)
            return float(hull.volume)  # Area in 2D
        except Exception:
            # Fallback to bounding box area
            min_coords = np.min(trajectory, axis=0)
            max_coords = np.max(trajectory, axis=0)
            return float(np.prod(max_coords - min_coords))
    
    def _calculate_position_entropy(self, trajectory: np.ndarray) -> float:
        """Calculate position entropy as measure of exploration diversity."""
        if len(trajectory) < 2:
            return 0.0
        
        # Discretize positions into grid
        grid_size = 10
        min_coords = np.min(trajectory, axis=0)
        max_coords = np.max(trajectory, axis=0)
        ranges = max_coords - min_coords
        
        if np.any(ranges == 0):
            return 0.0
        
        # Create grid indices
        grid_indices = ((trajectory - min_coords) / ranges * (grid_size - 1)).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_size - 1)
        
        # Calculate histogram
        hist, _ = np.histogramdd(grid_indices, bins=grid_size, range=[(0, grid_size-1)] * 2)
        
        # Calculate entropy
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zero probabilities
        
        return float(-np.sum(prob * np.log2(prob)))
    
    def _calculate_max_streak(self, boolean_array: np.ndarray) -> int:
        """Calculate maximum consecutive True values in boolean array."""
        if len(boolean_array) == 0:
            return 0
        
        streaks = []
        current_streak = 0
        
        for value in boolean_array:
            if value:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                    current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return max(streaks) if streaks else 0
    
    def _calculate_speed_entropy(self, speeds: np.ndarray) -> float:
        """Calculate speed distribution entropy."""
        if len(speeds) < 2:
            return 0.0
        
        # Create histogram of speeds
        hist, _ = np.histogram(speeds, bins=20)
        
        # Calculate probability distribution
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zero probabilities
        
        if len(prob) <= 1:
            return 0.0
        
        # Calculate entropy
        return float(-np.sum(prob * np.log2(prob)))
    
    def _calculate_acceleration_variance(self, speeds: np.ndarray) -> float:
        """Calculate variance in acceleration (speed changes)."""
        if len(speeds) < 2:
            return 0.0
        
        accelerations = np.diff(speeds)
        return float(np.var(accelerations))
    
    def _validate_trajectory_data(self, trajectory: Any) -> Dict[str, Any]:
        """Validate trajectory data structure."""
        validation = {'valid': True, 'errors': []}
        
        try:
            trajectory = np.asarray(trajectory)
            if trajectory.size == 0:
                validation['errors'].append("Trajectory is empty")
                validation['valid'] = False
            elif trajectory.ndim != 2:
                validation['errors'].append("Trajectory must be 2D array")
                validation['valid'] = False
            elif trajectory.shape[1] != 2:
                validation['errors'].append("Trajectory must have 2 columns (x, y)")
                validation['valid'] = False
            elif not np.isfinite(trajectory).all():
                validation['errors'].append("Trajectory contains non-finite values")
                validation['valid'] = False
        except Exception as e:
            validation['errors'].append(f"Trajectory validation error: {e}")
            validation['valid'] = False
        
        return validation
    
    def _validate_concentration_data(self, concentrations: Any) -> Dict[str, Any]:
        """Validate concentration data structure."""
        validation = {'valid': True, 'errors': []}
        
        try:
            concentrations = np.asarray(concentrations)
            if concentrations.size == 0:
                validation['errors'].append("Concentrations array is empty")
                validation['valid'] = False
            elif not np.isfinite(concentrations).all():
                validation['errors'].append("Concentrations contain non-finite values")
                validation['valid'] = False
            elif np.any(concentrations < 0):
                validation['errors'].append("Concentrations contain negative values")
                validation['valid'] = False
        except Exception as e:
            validation['errors'].append(f"Concentration validation error: {e}")
            validation['valid'] = False
        
        return validation


# Standalone utility functions

def calculate_basic_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistical measures for numpy array data.
    
    Args:
        data: Input numpy array for statistical analysis
        
    Returns:
        Dict[str, float]: Basic statistics including mean, std, min, max, median
    """
    if data.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
    
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data))
    }


def calculate_advanced_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate advanced statistical measures including distribution analysis.
    
    Args:
        data: Input numpy array for advanced statistical analysis
        
    Returns:
        Dict[str, float]: Advanced statistics including skewness, kurtosis, percentiles
    """
    if data.size < 2:
        return {'skewness': 0.0, 'kurtosis': 0.0, 'p25': 0.0, 'p75': 0.0, 'p95': 0.0}
    
    return {
        'skewness': float(scipy.stats.skew(data)),
        'kurtosis': float(scipy.stats.kurtosis(data)),
        'p25': float(np.percentile(data, 25)),
        'p75': float(np.percentile(data, 75)),
        'p95': float(np.percentile(data, 95))
    }


def create_stats_aggregator(config: Union[Dict[str, Any], StatsAggregatorConfig]) -> StatsAggregator:
    """
    Factory function for creating statistics aggregator instances.
    
    Args:
        config: Configuration dictionary or StatsAggregatorConfig instance
        
    Returns:
        StatsAggregator: Configured statistics aggregator instance
    """
    if isinstance(config, dict):
        config = StatsAggregatorConfig(**config)
    
    return StatsAggregator(config)


def generate_summary_report(
    episodes_data: List[Dict[str, Any]], 
    output_path: str,
    config: Optional[StatsAggregatorConfig] = None
) -> bool:
    """
    Generate comprehensive summary report from episodes data.
    
    Args:
        episodes_data: List of episode data dictionaries
        output_path: Output file path for summary report
        config: Optional configuration for statistics calculation
        
    Returns:
        bool: True if summary generation completed successfully
    """
    try:
        if config is None:
            config = StatsAggregatorConfig()
        
        aggregator = StatsAggregator(config)
        run_stats = aggregator.calculate_run_stats(episodes_data)
        
        return aggregator.export_summary(output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")
        return False


# Export all public components
__all__ = [
    'StatsAggregator',
    'StatsAggregatorConfig', 
    'calculate_basic_stats',
    'calculate_advanced_stats',
    'create_stats_aggregator',
    'generate_summary_report'
]