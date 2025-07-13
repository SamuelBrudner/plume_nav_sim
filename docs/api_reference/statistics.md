# Statistics API Reference

The statistics aggregation system provides automated research-focused metrics calculation, summary.json generation, and standardized analysis through the `StatsAggregatorProtocol` interface. This system is designed to support comprehensive experimental analysis with publication-quality results while maintaining high performance (≤33ms/step with 100 agents).

## Overview

The statistics system consists of two main components:

1. **StatsAggregatorProtocol**: Protocol interface defining the contract for statistics aggregation implementations
2. **StatsAggregator**: Comprehensive implementation providing automated research metrics calculation

The system integrates seamlessly with the recording framework and episode completion hooks to provide automated data collection and analysis capabilities.

## StatsAggregatorProtocol Interface

The `StatsAggregatorProtocol` defines the standardized interface for all statistics aggregation implementations, ensuring consistent API across different analysis approaches.

### Protocol Methods

#### `calculate_episode_stats()`

Calculate comprehensive statistics for a single episode with configurable analysis depth.

```python
def calculate_episode_stats(
    self, 
    trajectory_data: Dict[str, Any],
    episode_id: int,
    custom_metrics: Optional[Dict[str, callable]] = None
) -> Dict[str, float]:
```

**Parameters:**
- `trajectory_data`: Dictionary containing episode trajectory information including:
  - `'positions'`: Agent position time series as numpy array (n_steps, 2)
  - `'concentrations'`: Odor concentration measurements as numpy array
  - `'actions'`: Applied navigation commands
  - `'rewards'`: Step-wise reward values
  - `'timestamps'`: Temporal information
- `episode_id`: Unique episode identifier for metric correlation
- `custom_metrics`: Optional dictionary of custom metric calculation functions

**Returns:**
- Dictionary of episode-level metrics including:
  - `'path_efficiency'`: Ratio of direct distance to actual path length
  - `'exploration_coverage'`: Fraction of domain area explored
  - `'mean_concentration'`: Average odor concentration encountered
  - `'success_indicator'`: Binary success metric (1.0 if successful)
  - `'total_reward'`: Cumulative episode reward
  - `'episode_length'`: Number of simulation steps

**Performance Requirements:**
- Must execute in <10ms for episode-level metric computation

**Example:**
```python
trajectory_data = {
    'positions': position_time_series,
    'concentrations': concentration_measurements,
    'actions': action_sequence,
    'rewards': reward_time_series
}
metrics = aggregator.calculate_episode_stats(trajectory_data, episode_id=42)
print(f"Path efficiency: {metrics['path_efficiency']:.3f}")
print(f"Success: {bool(metrics['success_indicator'])}")
```

#### `calculate_run_stats()`

Calculate aggregate statistics across multiple episodes for run-level analysis.

```python
def calculate_run_stats(
    self, 
    episode_data_list: List[Dict[str, Any]],
    run_id: str,
    statistical_tests: Optional[List[str]] = None
) -> Dict[str, float]:
```

**Parameters:**
- `episode_data_list`: List of episode data dictionaries from `calculate_episode_stats()`
- `run_id`: Unique run identifier for experimental tracking
- `statistical_tests`: Optional list of statistical tests to perform ("t_test", "anova", "ks_test", "wilcoxon")

**Returns:**
- Dictionary of run-level aggregate metrics including:
  - `'success_rate'`: Fraction of successful episodes
  - `'mean_path_efficiency'`: Average path efficiency across episodes
  - `'std_path_efficiency'`: Standard deviation of path efficiency
  - `'mean_episode_length'`: Average episode duration
  - `'total_episodes'`: Number of episodes in run
  - `'confidence_intervals'`: Statistical confidence bounds (if requested)

**Performance Requirements:**
- Must execute in <100ms for multi-episode aggregation analysis

**Example:**
```python
episode_data_list = [episode_metrics_1, episode_metrics_2, ...]
run_metrics = aggregator.calculate_run_stats(
    episode_data_list, run_id="experiment_001"
)
print(f"Success rate: {run_metrics['success_rate']:.2%}")
print(f"Mean efficiency: {run_metrics['mean_path_efficiency']:.3f}")
```

#### `export_summary()`

Generate and export standardized summary report for research publication.

```python
def export_summary(
    self, 
    output_path: str,
    run_data: Optional[Dict[str, Any]] = None,
    include_distributions: bool = False,
    format: str = "json"
) -> bool:
```

**Parameters:**
- `output_path`: File system path for summary report output
- `run_data`: Optional run-level data from `calculate_run_stats()` for inclusion
- `include_distributions`: Include distribution plots and histograms in summary
- `format`: Output format specification ("json", "yaml", "markdown", "latex")

**Returns:**
- `True` if summary export completed successfully, `False` otherwise

**Performance Requirements:**
- Must execute in <50ms for summary generation and file output

**Example:**
```python
success = aggregator.export_summary(
    output_path="./results/experiment_summary.json",
    run_data=run_metrics,
    include_distributions=False
)
assert success == True
```

## StatsAggregator Implementation

The `StatsAggregator` class provides a comprehensive implementation of the `StatsAggregatorProtocol` with advanced statistical analysis capabilities, custom metric support, and research-grade performance.

### Class Definition

```python
class StatsAggregator(StatsAggregatorProtocol):
    """
    Core statistics aggregator implementing comprehensive research metrics calculation.
    
    Provides automated research-focused statistics collection and analysis with
    episode-level and run-level statistical analysis, configurable metrics
    definitions, and high-performance processing designed to meet the ≤33 ms/step
    performance requirement with 100 agents.
    """
```

### Configuration

#### StatsAggregatorConfig

Configuration dataclass for statistics aggregator setup and validation.

```python
@dataclass
class StatsAggregatorConfig:
    """Configuration for statistics aggregator setup and validation."""
    
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
```

**Configuration Parameters:**

- **Core Configuration:**
  - `metrics_definitions`: Dictionary defining which metrics to calculate for each data type
  - `aggregation_levels`: List of aggregation levels to compute (episode, run, batch)
  - `output_format`: Export format for summary data (json, yaml, pickle)
  - `custom_calculations`: Dictionary of custom metric calculation functions

- **Performance Configuration:**
  - `performance_tracking`: Enable detailed performance monitoring and timing
  - `parallel_processing`: Use multiprocessing for batch calculations
  - `memory_limit_mb`: Maximum memory usage for statistical computations
  - `computation_timeout_s`: Maximum time allowed for metric calculations

- **Validation Configuration:**
  - `data_validation`: Enable input data validation and schema checking
  - `output_validation`: Validate output metrics against expected ranges
  - `error_handling`: Strategy for handling calculation errors (skip, warn, raise)
  - `precision_mode`: Numerical precision for calculations (float32, float64)

### Constructor

```python
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
```

### Core Methods

#### Episode-Level Analysis

The `calculate_episode_stats()` method provides comprehensive episode-level metrics:

**Calculated Metrics:**

1. **Trajectory Statistics:**
   - Basic positional statistics (mean, std, min, max, range)
   - Distance and displacement metrics (total distance, net displacement, displacement efficiency)
   - Tortuosity (ratio of path length to direct distance)
   - Exploration area coverage and position entropy

2. **Concentration Statistics:**
   - Basic statistics (mean, std, min, max, median)
   - Percentiles (10th, 25th, 75th, 90th, 95th, 99th)
   - Distribution analysis (skewness, kurtosis)
   - Detection statistics (detection rate, max detection streak, detection efficiency)

3. **Speed and Movement Statistics:**
   - Speed distribution (mean, std, min, max, median)
   - Movement efficiency metrics
   - Acceleration variance and speed entropy

4. **Timing Statistics:**
   - Episode length and step durations
   - Performance compliance rate (≤33ms target)
   - Timing efficiency metrics

5. **Custom Metrics:**
   - User-defined calculation functions
   - Configurable metric application
   - Error handling for custom calculations

6. **Derived Metrics:**
   - Navigation efficiency (combined trajectory and speed metrics)
   - Search effectiveness (concentration-based performance)
   - Performance score (timing compliance assessment)

7. **Error Analysis:**
   - Position uncertainty quantification
   - Measurement uncertainty analysis

#### Run-Level Aggregation

The `calculate_run_stats()` method provides cross-episode analysis:

**Aggregation Features:**

1. **Statistical Aggregation:**
   - Mean, standard deviation, min, max for all episode metrics
   - Cross-episode correlation analysis
   - Learning trend detection with regression analysis

2. **Performance Trends:**
   - Episode-by-episode improvement detection
   - Statistical significance testing
   - Learning curve analysis

3. **Comparative Analysis:**
   - Success rate calculation
   - Performance distribution analysis
   - Outlier detection and analysis

#### Summary Export

The `export_summary()` method generates standardized research reports:

**Summary.json Format:**

```json
{
  "metadata": {
    "export_timestamp": 1640995200.0,
    "aggregator_version": "1.0",
    "format_version": "1.0",
    "configuration": {
      "metrics_definitions": {...},
      "aggregation_levels": ["episode", "run"],
      "custom_calculations": ["tortuosity", "exploration_index"],
      "performance_tracking": true
    }
  },
  "performance_metrics": {
    "episodes_processed": 100,
    "runs_processed": 5,
    "computation_time_ms": 156.7,
    "memory_usage_peak_mb": 45.2,
    "performance_compliance": {
      "meets_33ms_target": true,
      "warning_count": 0,
      "error_rate": 0.0
    }
  },
  "run_statistics": [...],
  "current_episode": {...},
  "global_summary": {
    "total_runs": 5,
    "total_episodes": 100,
    "processing_efficiency": {...},
    "data_quality": {
      "error_rate": 0.0,
      "data_completeness": 1.0,
      "quality_score": 1.0
    }
  }
}
```

### Performance Monitoring

#### Performance Metrics

The `get_performance_metrics()` method provides comprehensive performance tracking:

```python
def get_performance_metrics(self) -> Dict[str, Any]:
    """
    Get comprehensive performance metrics for monitoring and optimization.
    
    Returns:
        Performance metrics including timing, memory, and computation stats
    """
```

**Returned Metrics:**
- `episodes_processed`: Number of episodes analyzed
- `runs_processed`: Number of runs completed
- `computation_time_ms`: Total computation time
- `memory_usage_peak_mb`: Peak memory usage
- `episodes_per_second`: Processing throughput
- `avg_computation_time_ms`: Average per-episode computation time
- `performance_compliance`: Performance target compliance status

### Utility Functions

#### Factory Functions

```python
def create_stats_aggregator(config: Union[Dict[str, Any], StatsAggregatorConfig]) -> StatsAggregator:
    """
    Factory function for creating statistics aggregator instances.
    
    Args:
        config: Configuration dictionary or StatsAggregatorConfig instance
        
    Returns:
        Configured statistics aggregator instance
    """
```

#### Summary Generation

```python
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
        True if summary generation completed successfully
    """
```

#### Basic Statistics

```python
def calculate_basic_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistical measures for numpy array data.
    
    Args:
        data: Input numpy array for statistical analysis
        
    Returns:
        Basic statistics including mean, std, min, max, median
    """
```

```python
def calculate_advanced_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate advanced statistical measures including distribution analysis.
    
    Args:
        data: Input numpy array for advanced statistical analysis
        
    Returns:
        Advanced statistics including skewness, kurtosis, percentiles
    """
```

## Integration Patterns

### Recorder Integration

The statistics aggregator integrates with the recorder system for automated data collection:

```python
# Integration with recorder for data source connectivity
aggregator = StatsAggregator(config, recorder=recorder_instance)

# Automated episode analysis
episode_stats = aggregator.calculate_episode_stats(episode_data)

# Access recorded data through recorder integration
recorded_data = recorder.export_data("./temp_data.parquet")
```

### Episode Completion Hooks

Integration with episode completion for automated analysis:

```python
# Hook integration in environment
def on_episode_end(self, final_info: dict) -> None:
    episode_stats = self.stats_aggregator.calculate_episode_stats(
        final_info['trajectory_data'], 
        episode_id=final_info['episode_id']
    )
    # Store or process episode statistics
```

### Batch Processing Workflows

Support for batch analysis across multiple experimental runs:

```python
# Batch processing configuration
config = StatsAggregatorConfig(
    parallel_processing=True,
    performance_tracking=True,
    output_format='json'
)

aggregator = StatsAggregator(config)

# Process multiple runs
for run_id, episodes_data in experimental_runs.items():
    run_stats = aggregator.calculate_run_stats(episodes_data, run_id)
    aggregator.export_summary(f"./results/{run_id}_summary.json")
```

## Configuration Examples

### Basic Configuration

```yaml
# conf/base/stats/basic.yaml
_target_: plume_nav_sim.analysis.stats.StatsAggregator
config:
  metrics_definitions:
    trajectory: ['mean', 'std', 'total_distance', 'displacement_efficiency']
    concentration: ['mean', 'std', 'detection_rate']
    timing: ['episode_length']
  aggregation_levels: ['episode', 'run']
  output_format: 'json'
  performance_tracking: true
```

### Advanced Configuration

```yaml
# conf/base/stats/advanced.yaml
_target_: plume_nav_sim.analysis.stats.StatsAggregator
config:
  metrics_definitions:
    trajectory: ['mean', 'std', 'min', 'max', 'median', 'total_distance', 'displacement_efficiency', 'tortuosity']
    concentration: ['mean', 'std', 'min', 'max', 'percentiles', 'detection_rate', 'detection_efficiency']
    speed: ['mean_speed', 'std_speed', 'movement_efficiency']
    timing: ['episode_length', 'performance_compliance_rate']
  aggregation_levels: ['episode', 'run', 'batch']
  output_format: 'json'
  custom_calculations: {}
  performance_tracking: true
  parallel_processing: true
  data_validation: true
  error_handling: 'warn'
  precision_mode: 'float64'
```

### Custom Metrics Configuration

```python
# Custom tortuosity calculation
def custom_tortuosity(trajectory_data):
    positions = trajectory_data['positions']
    path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    direct_distance = np.linalg.norm(positions[-1] - positions[0])
    return path_length / direct_distance if direct_distance > 0 else np.inf

# Configuration with custom metrics
config = StatsAggregatorConfig(
    custom_calculations={'tortuosity': custom_tortuosity},
    performance_tracking=True
)
```

## Advanced Usage

### Custom Statistics Definitions

Create domain-specific metrics for specialized research:

```python
def calculate_search_efficiency(trajectory_data):
    """Calculate search efficiency based on concentration gradient following."""
    positions = trajectory_data['positions']
    concentrations = trajectory_data['concentrations']
    
    # Calculate gradient following efficiency
    gradients = np.gradient(concentrations)
    movement_vectors = np.diff(positions, axis=0)
    
    # Dot product between movement and gradient
    alignment = np.sum(movement_vectors * gradients[:-1, np.newaxis])
    total_movement = np.sum(np.linalg.norm(movement_vectors, axis=1))
    
    return alignment / total_movement if total_movement > 0 else 0.0

# Add to configuration
custom_metrics = {'search_efficiency': calculate_search_efficiency}
```

### Comparative Studies

Framework for comparing different experimental conditions:

```python
def compare_experimental_conditions(control_data, treatment_data, stats_config):
    """Compare two experimental conditions with statistical testing."""
    aggregator = StatsAggregator(stats_config)
    
    # Calculate statistics for both conditions
    control_stats = aggregator.calculate_run_stats(control_data, "control")
    treatment_stats = aggregator.calculate_run_stats(treatment_data, "treatment")
    
    # Statistical comparison
    improvement = ((treatment_stats['success_rate'] - control_stats['success_rate']) 
                   / control_stats['success_rate'])
    
    # Export comparative summary
    comparative_summary = {
        'control': control_stats,
        'treatment': treatment_stats,
        'improvement': improvement,
        'statistical_significance': treatment_stats.get('t_test_p_value', None)
    }
    
    return comparative_summary
```

### Publication-Quality Results

Generate publication-ready statistical summaries:

```python
def generate_publication_summary(episodes_data, output_path):
    """Generate publication-quality statistical summary."""
    config = StatsAggregatorConfig(
        metrics_definitions={
            'trajectory': ['mean', 'std', 'displacement_efficiency', 'tortuosity'],
            'concentration': ['mean', 'std', 'detection_rate'],
            'performance': ['success_rate', 'episode_length']
        },
        output_format='markdown',
        data_validation=True,
        performance_tracking=True
    )
    
    aggregator = StatsAggregator(config)
    run_stats = aggregator.calculate_run_stats(episodes_data, "publication_run")
    
    # Export with publication formatting
    success = aggregator.export_summary(
        output_path=output_path,
        run_data=run_stats,
        include_distributions=True,
        format="markdown"
    )
    
    return success
```

## Dependencies

### Required Dependencies

The statistics system requires the following external dependencies:

- **NumPy (>=1.26.0)**: Core numerical computing for statistical calculations, array operations, and vectorized computations
- **SciPy (>=1.11.0)**: Advanced statistical functions, hypothesis testing, and specialized statistical algorithms

### NumPy Operations

Key NumPy functions used in statistical calculations:

```python
import numpy as np

# Array operations
np.array, np.mean, np.std, np.median
np.percentile, np.histogram
np.min, np.max, np.sum
np.diff, np.gradient
np.linalg.norm

# Example usage in statistics
trajectory = np.array(positions)
distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
mean_distance = np.mean(distances)
```

### SciPy Statistical Tests

Statistical analysis functions for hypothesis testing:

```python
from scipy import stats

# Statistical tests
stats.ttest_1samp    # One-sample t-test
stats.mannwhitneyu   # Mann-Whitney U test
stats.spearmanr      # Spearman correlation
stats.pearsonr       # Pearson correlation
stats.normaltest     # Normality testing

# Example usage in run statistics
correlation, p_value = stats.pearsonr(detection_rates, movement_efficiencies)
```

## Error Handling

The statistics system provides comprehensive error handling:

### Configuration Validation

```python
# Automatic validation in StatsAggregatorConfig
config = StatsAggregatorConfig(
    memory_limit_mb=-1  # Raises ValueError
)
```

### Data Validation

```python
# Input data validation
validation_results = aggregator.validate_data(episode_data)
if not validation_results['valid']:
    print("Validation errors:", validation_results['errors'])
```

### Calculation Error Handling

```python
# Configurable error handling strategies
config = StatsAggregatorConfig(
    error_handling='warn'  # 'skip', 'warn', or 'raise'
)
```

## Performance Considerations

### Performance Requirements

The statistics system is designed to meet strict performance requirements:

- **Episode Statistics**: ≤10ms per episode
- **Run Statistics**: ≤100ms for multi-episode aggregation  
- **Summary Export**: ≤50ms for file generation
- **Memory Usage**: ≤50MB for typical experimental datasets
- **Step-level Impact**: ≤33ms per simulation step with 100 agents

### Performance Monitoring

```python
# Monitor performance compliance
metrics = aggregator.get_performance_metrics()
print(f"Average computation time: {metrics['avg_computation_time_ms']:.2f}ms")
print(f"Meets 33ms target: {metrics['performance_compliance']['meets_33ms_target']}")
```

### Optimization Strategies

- **Vectorized Operations**: Use NumPy vectorization for multi-agent scenarios
- **Lazy Evaluation**: Compute metrics only when requested
- **Memory Management**: Configurable precision modes and memory limits
- **Parallel Processing**: Optional multiprocessing for batch calculations

This comprehensive API reference provides complete documentation for the statistics aggregation system, supporting automated research metrics collection, publication-quality analysis, and high-performance statistical processing in the plume_nav_sim v1.0 framework.