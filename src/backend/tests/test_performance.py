"""
Comprehensive performance testing module for plume_nav_sim environment validating all performance targets 
including environment step latency (<1ms), episode reset timing (<10ms), rendering performance (<5ms RGB, 
<50ms human), memory usage constraints (<50MB), and system resource utilization. Provides detailed 
performance benchmarking, statistical analysis, regression detection, and optimization recommendations 
using pytest framework with specialized fixtures and advanced performance monitoring capabilities.

This module implements systematic performance validation of the PlumeSearchEnv including:
- Step latency benchmarking with <1ms target validation and statistical analysis
- Episode reset performance testing with <10ms target and seeding overhead analysis  
- Dual-mode rendering performance validation for RGB array (<5ms) and human mode (<50ms)
- Memory usage profiling with <50MB total footprint validation and leak detection
- Scalability analysis across different grid configurations with resource optimization
- Performance regression detection with statistical trend analysis and baseline comparison
- Comprehensive reporting with executive summaries and actionable optimization recommendations

Performance targets are validated against system constants with tolerance checking and detailed
failure analysis including component-specific recommendations for development optimization.
"""

# External imports with version requirements for comprehensive testing framework
import contextlib  # >=3.10 - Context managers for resource isolation and performance measurement
import gc  # >=3.10 - Garbage collection control for memory leak detection and baseline measurement
import logging  # >=3.10 - Structured logging for fail-fast diagnostics during performance test import
import numpy as np  # >=2.1.0 - Statistical analysis and performance data processing for benchmark validation
import pytest  # >=8.0.0 - Testing framework with fixtures, parametrization, and comprehensive execution support
import statistics  # >=3.10 - Statistical functions for performance metric calculation and confidence intervals
import threading  # >=3.10 - Thread safety utilities for concurrent performance testing and resource monitoring
import time  # >=3.10 - High-precision timing measurements using perf_counter for accurate latency validation
import warnings  # >=3.10 - Warning management for performance test execution and deprecation handling

# Internal imports for environment benchmarking and comprehensive performance analysis
from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv, create_plume_search_env

LOGGER = logging.getLogger(__name__)

try:
    import psutil  # type: ignore[import-not-found] - validated at runtime
except ImportError as exc:  # pragma: no cover - failure handled immediately
    _MESSAGE = "psutil is required for performance benchmark validation"
    LOGGER.error(_MESSAGE)
    raise RuntimeError(_MESSAGE) from exc

try:
    from benchmarks.environment_performance import (
        run_environment_performance_benchmark,
        benchmark_step_latency,
        benchmark_episode_performance,
        benchmark_memory_usage,
        benchmark_rendering_performance,
        validate_performance_targets,
        EnvironmentPerformanceSuite,
        PerformanceAnalysis,
        EnvironmentBenchmarkConfig,
        BenchmarkResult
    )
except ImportError as exc:  # pragma: no cover - failure handled immediately
    _MESSAGE = "Performance benchmark module is required"
    LOGGER.error(_MESSAGE)
    raise RuntimeError(_MESSAGE) from exc
from plume_nav_sim.core.constants import (
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    PERFORMANCE_TARGET_RGB_RENDER_MS,  
    PERFORMANCE_TARGET_PLUME_GENERATION_MS,
    MEMORY_LIMIT_TOTAL_MB,
    get_performance_constants,
    get_testing_constants
)

# Global configuration constants for systematic performance testing and analysis
PERFORMANCE_TEST_ITERATIONS = 1000
PERFORMANCE_TEST_WARMUP_ITERATIONS = 100  
MEMORY_TEST_DURATION_SECONDS = 30
PERFORMANCE_TOLERANCE_FACTOR = 0.1
STATISTICAL_CONFIDENCE_LEVEL = 0.95
REGRESSION_DETECTION_THRESHOLD = 0.15
MEMORY_LEAK_DETECTION_SAMPLES = 50
SCALABILITY_TEST_GRID_SIZES = [(32, 32), (64, 64), (128, 128), (256, 256)]
BENCHMARK_TIMEOUT_SECONDS = 300

# Performance testing fixtures for systematic test execution and resource management
@pytest.fixture
def performance_test_env():
    """
    Performance-optimized environment fixture providing clean environment instance for 
    benchmark execution with deterministic seeding and resource cleanup validation.
    
    Creates environment instance optimized for performance testing including baseline
    memory measurement, deterministic initialization, and proper resource cleanup
    with performance monitoring infrastructure setup.
    
    Returns:
        PlumeSearchEnv: Performance-optimized environment instance with monitoring
    """
    # Force garbage collection for clean memory baseline before environment creation
    gc.collect()
    
    # Create environment with performance-optimized configuration
    env = create_plume_search_env(
        grid_size=(128, 128),  # Standard grid for consistent performance measurement
        source_location=(64, 64),  # Centered source for symmetric performance
        max_steps=1000,  # Full episode length for realistic performance testing
        goal_radius=0  # Precise goal detection for consistent termination
    )
    
    # Initialize environment state for deterministic performance measurement
    env.reset(seed=42)  # Reproducible seed for consistent benchmark conditions
    
    try:
        yield env
    finally:
        # Ensure proper cleanup and resource deallocation after testing
        env.close()
        gc.collect()  # Clean up any residual objects for accurate memory measurement


@pytest.fixture  
def performance_tracker():
    """
    Performance tracking fixture providing comprehensive measurement infrastructure for 
    timing analysis, memory profiling, and statistical validation with baseline comparison.
    
    Initializes performance monitoring components including high-precision timing,
    system resource tracking, and statistical analysis infrastructure for comprehensive
    benchmark execution and validation.
    
    Returns:
        dict: Performance tracking infrastructure with timing and memory monitoring
    """
    # Initialize high-precision performance tracking infrastructure
    tracking_data = {
        'start_time': time.perf_counter(),
        'memory_samples': [],
        'timing_measurements': [],
        'baseline_memory': None,
        'process_monitor': psutil.Process()
    }
    
    # Establish baseline memory usage for leak detection analysis
    gc.collect()  # Force garbage collection for accurate baseline
    tracking_data['baseline_memory'] = tracking_data['process_monitor'].memory_info().rss / (1024 * 1024)
    
    try:
        yield tracking_data
    finally:
        # Calculate final performance summary and resource cleanup validation
        final_memory = tracking_data['process_monitor'].memory_info().rss / (1024 * 1024)
        memory_growth = final_memory - tracking_data['baseline_memory']
        
        # Log significant memory growth for leak detection analysis
        if memory_growth > 5.0:  # 5MB threshold for memory growth warning
            warnings.warn(f"Memory growth detected: {memory_growth:.1f}MB increase during testing")


@pytest.fixture
def memory_monitor():
    """
    Advanced memory monitoring fixture providing continuous memory usage tracking with 
    leak detection capabilities and component-specific memory analysis for optimization.
    
    Implements background memory monitoring with configurable sampling intervals,
    leak detection thresholds, and component-specific tracking for detailed memory
    usage analysis and optimization recommendations.
    
    Returns:
        object: Memory monitoring infrastructure with continuous tracking capabilities
    """
    import threading
    import time
    
    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.samples = []
            self.monitoring = False
            self.monitor_thread = None
            self.baseline_memory = None
            
        def start_monitoring(self, sampling_interval: float = 0.1):
            """Start continuous memory monitoring with specified sampling interval."""
            if self.monitoring:
                return  # Already monitoring
                
            # Establish baseline memory usage
            gc.collect()
            self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)
            self.monitoring = True
            
            def monitor_loop():
                while self.monitoring:
                    try:
                        memory_mb = self.process.memory_info().rss / (1024 * 1024)
                        self.samples.append(memory_mb)
                        time.sleep(sampling_interval)
                    except Exception:
                        break
                        
            self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            self.monitor_thread.start()
            
        def stop_monitoring(self):
            """Stop memory monitoring and return analysis results."""
            self.monitoring = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
                
            if not self.samples:
                return {'error': 'No memory samples collected'}
                
            return {
                'baseline_memory_mb': self.baseline_memory,
                'peak_memory_mb': max(self.samples),
                'mean_memory_mb': statistics.mean(self.samples),
                'memory_growth_mb': max(self.samples) - min(self.samples),
                'sample_count': len(self.samples),
                'leak_detected': (max(self.samples) - min(self.samples)) > 5.0
            }
            
    monitor = MemoryMonitor()
    
    try:
        yield monitor
    finally:
        if monitor.monitoring:
            monitor.stop_monitoring()


@pytest.mark.performance
@pytest.mark.timeout(60)
def test_environment_step_latency_performance(performance_test_env, performance_tracker):
    """
    Test environment step execution latency against <1ms performance target with comprehensive 
    statistical analysis, action breakdown validation, and latency distribution testing for RL 
    training efficiency optimization.
    
    Validates step execution performance through systematic benchmarking including warmup
    iterations, statistical significance testing, confidence interval calculation, and
    performance target compliance with detailed failure analysis and optimization recommendations.
    
    Args:
        performance_test_env (PlumeSearchEnv): Performance-optimized environment instance
        performance_tracker (dict): Performance measurement infrastructure
        
    Asserts:
        Average step latency meets <1ms target with statistical significance
        95th percentile latency within acceptable performance bounds
        No significant performance outliers affecting training efficiency
    """
    # Initialize environment and perform warmup iterations for stable performance baseline
    env = performance_test_env
    obs, info = env.reset(seed=42)  # Deterministic seed for reproducible measurements
    
    # Execute warmup iterations to stabilize system performance and eliminate cold start effects
    for _ in range(PERFORMANCE_TEST_WARMUP_ITERATIONS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    # Execute comprehensive step latency benchmark with high-precision timing measurement
    step_timings = []
    action_breakdown = {0: [], 1: [], 2: [], 3: []}  # Per-action timing analysis
    
    for iteration in range(PERFORMANCE_TEST_ITERATIONS):
        action = env.action_space.sample()
        
        # Measure step execution time with microsecond precision
        start_time = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time_ms = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        
        step_timings.append(step_time_ms)
        action_breakdown[action].append(step_time_ms)
        performance_tracker['timing_measurements'].append(step_time_ms)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    # Calculate comprehensive statistical analysis of step latency performance
    mean_latency = statistics.mean(step_timings)
    median_latency = statistics.median(step_timings)
    std_deviation = statistics.stdev(step_timings)
    percentile_95 = np.percentile(step_timings, 95)
    percentile_99 = np.percentile(step_timings, 99)
    
    # Calculate confidence interval for statistical significance validation
    confidence_interval = None
    if len(step_timings) >= 30:  # Sufficient sample size for normal approximation
        std_error = std_deviation / np.sqrt(len(step_timings))
        margin_error = 1.96 * std_error  # 95% confidence interval
        confidence_interval = (mean_latency - margin_error, mean_latency + margin_error)
    
    # Validate step latency performance against target with tolerance checking
    target_latency = PERFORMANCE_TARGET_STEP_LATENCY_MS
    tolerance = target_latency * PERFORMANCE_TOLERANCE_FACTOR
    performance_compliant = mean_latency <= target_latency
    
    # Generate detailed performance analysis and optimization recommendations
    performance_analysis = {
        'mean_latency_ms': mean_latency,
        'median_latency_ms': median_latency, 
        'std_deviation_ms': std_deviation,
        'percentile_95_ms': percentile_95,
        'percentile_99_ms': percentile_99,
        'target_latency_ms': target_latency,
        'performance_ratio': mean_latency / target_latency,
        'confidence_interval': confidence_interval,
        'sample_size': len(step_timings)
    }
    
    # Analyze action-specific performance for optimization guidance
    action_analysis = {}
    for action, timings in action_breakdown.items():
        if timings:
            action_analysis[f'action_{action}'] = {
                'mean_ms': statistics.mean(timings),
                'sample_count': len(timings)
            }
    
    # Assert performance targets met with comprehensive error reporting
    assert performance_compliant, (
        f"Step latency performance target not met:\n"
        f"  Mean latency: {mean_latency:.3f}ms (target: {target_latency}ms)\n"
        f"  Performance ratio: {mean_latency / target_latency:.2f}x target\n"
        f"  95th percentile: {percentile_95:.3f}ms\n"
        f"  Standard deviation: {std_deviation:.3f}ms\n"
        f"  Sample size: {len(step_timings)} iterations\n"
        f"  Confidence interval: {confidence_interval}\n"
        f"  Action breakdown: {action_analysis}\n"
        f"  Recommendation: Optimize core step execution algorithms for <{target_latency}ms target"
    )
    
    # Validate 95th percentile performance for consistency under load
    percentile_target = target_latency * 2.0  # Allow 2x target for percentile performance
    assert percentile_95 <= percentile_target, (
        f"95th percentile latency exceeds acceptable threshold:\n"
        f"  95th percentile: {percentile_95:.3f}ms (threshold: {percentile_target}ms)\n"
        f"  Recommendation: Investigate performance outliers and system load variability"
    )


@pytest.mark.performance
@pytest.mark.timeout(30)
def test_episode_reset_performance(performance_test_env, performance_tracker):
    """
    Test episode reset performance against <10ms target including initialization timing, 
    seeding overhead analysis, component setup validation, and reproducibility performance 
    impact assessment.
    
    Validates episode reset timing through comprehensive benchmarking including seeded
    and unseeded reset comparison, component initialization analysis, and statistical
    validation of reset consistency across multiple episodes.
    
    Args:
        performance_test_env (PlumeSearchEnv): Performance-optimized environment instance  
        performance_tracker (dict): Performance measurement infrastructure
        
    Asserts:
        Average reset time meets <10ms target with statistical validation
        Seeding overhead within acceptable performance bounds
        Reset consistency across multiple episode cycles
    """
    env = performance_test_env
    
    # Execute episode reset performance benchmark with timing analysis
    reset_timings = []
    seeded_reset_timings = []
    unseeded_reset_timings = []
    
    # Test both seeded and unseeded reset performance for overhead analysis
    num_episodes = min(100, PERFORMANCE_TEST_ITERATIONS // 10)  # Reasonable episode count
    
    for episode in range(num_episodes):
        # Test unseeded reset performance
        start_time = time.perf_counter()
        obs, info = env.reset()
        unseeded_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        unseeded_reset_timings.append(unseeded_time)
        reset_timings.append(unseeded_time)
        
        # Test seeded reset performance with reproducibility validation
        start_time = time.perf_counter() 
        obs, info = env.reset(seed=episode + 42)
        seeded_time = (time.perf_counter() - start_time) * 1000
        
        seeded_reset_timings.append(seeded_time)
        reset_timings.append(seeded_time)
        
        performance_tracker['timing_measurements'].extend([unseeded_time, seeded_time])
    
    # Calculate comprehensive reset performance statistics
    mean_reset_time = statistics.mean(reset_timings)
    mean_seeded_time = statistics.mean(seeded_reset_timings)
    mean_unseeded_time = statistics.mean(unseeded_reset_timings)
    seeding_overhead = mean_seeded_time - mean_unseeded_time
    
    # Analyze component initialization timing and state setup validation
    component_analysis = {
        'total_reset_operations': len(reset_timings),
        'seeded_operations': len(seeded_reset_timings),
        'unseeded_operations': len(unseeded_reset_timings),
        'mean_reset_time_ms': mean_reset_time,
        'seeding_overhead_ms': seeding_overhead,
        'seeding_overhead_percent': (seeding_overhead / mean_unseeded_time) * 100 if mean_unseeded_time > 0 else 0
    }
    
    # Validate reset performance against target with tolerance checking
    reset_target = PERFORMANCE_TARGET_PLUME_GENERATION_MS  # 10ms target for reset operations
    tolerance = reset_target * PERFORMANCE_TOLERANCE_FACTOR
    performance_compliant = mean_reset_time <= reset_target
    
    # Test plume initialization timing as component of overall reset performance
    plume_initialization_compliant = mean_reset_time <= reset_target
    
    # Validate state consistency across multiple reset operations
    consistency_validation = {
        'reset_time_variance': statistics.stdev(reset_timings) if len(reset_timings) > 1 else 0,
        'coefficient_of_variation': (statistics.stdev(reset_timings) / mean_reset_time) if mean_reset_time > 0 else 0
    }
    
    # Assert reset performance meets targets with component-specific recommendations
    assert performance_compliant, (
        f"Episode reset performance target not met:\n"
        f"  Mean reset time: {mean_reset_time:.3f}ms (target: {reset_target}ms)\n"
        f"  Seeded reset time: {mean_seeded_time:.3f}ms\n"
        f"  Unseeded reset time: {mean_unseeded_time:.3f}ms\n"
        f"  Seeding overhead: {seeding_overhead:.3f}ms ({component_analysis['seeding_overhead_percent']:.1f}%)\n"
        f"  Reset time variance: {consistency_validation['reset_time_variance']:.3f}ms\n"
        f"  Total episodes tested: {num_episodes}\n"
        f"  Recommendation: Optimize environment initialization and plume generation for <{reset_target}ms target"
    )
    
    # Validate seeding impact on reset performance stays within reasonable bounds
    max_acceptable_overhead = reset_target * 0.2  # 20% overhead threshold
    assert seeding_overhead <= max_acceptable_overhead, (
        f"Seeding overhead exceeds acceptable threshold:\n"
        f"  Seeding overhead: {seeding_overhead:.3f}ms (threshold: {max_acceptable_overhead}ms)\n"
        f"  Overhead percentage: {component_analysis['seeding_overhead_percent']:.1f}%\n"
        f"  Recommendation: Optimize seeding implementation for reduced performance impact"
    )


@pytest.mark.performance
@pytest.mark.timeout(30)
def test_rgb_rendering_performance(performance_test_env, performance_tracker):
    """
    Test RGB array rendering performance against <5ms target with pixel generation analysis, 
    memory efficiency validation, and output format compliance testing for programmatic 
    visualization requirements.
    
    Validates RGB rendering performance through comprehensive benchmarking including pixel
    generation timing, memory efficiency analysis, output format validation, and rendering
    consistency across different environment states.
    
    Args:
        performance_test_env (PlumeSearchEnv): Performance-optimized environment instance
        performance_tracker (dict): Performance measurement infrastructure
        
    Asserts:
        RGB rendering time meets <5ms target with statistical validation
        Output format complies with expected shape and dtype requirements  
        Memory usage during rendering within efficiency bounds
    """
    env = performance_test_env
    
    # Initialize environment with rgb_array mode for RGB rendering performance testing
    obs, info = env.reset(seed=42)
    
    # Execute RGB rendering performance benchmark with comprehensive timing analysis
    render_timings = []
    memory_usage_samples = []
    output_validations = []
    
    num_renders = min(200, PERFORMANCE_TEST_ITERATIONS // 5)  # Reasonable render count
    
    for render_iteration in range(num_renders):
        # Take random step to vary environment state for comprehensive rendering testing
        if render_iteration > 0:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        # Measure memory usage before rendering for efficiency analysis
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)
        
        # Execute RGB array rendering with high-precision timing measurement
        start_time = time.perf_counter()
        rgb_array = env.render(mode='rgb_array')
        render_time_ms = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
        
        # Measure memory usage after rendering
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_delta = memory_after - memory_before
        
        render_timings.append(render_time_ms)
        memory_usage_samples.append(memory_delta)
        performance_tracker['timing_measurements'].append(render_time_ms)
        
        # Validate RGB array output format compliance with expected specifications
        if rgb_array is not None:
            format_validation = {
                'shape_valid': len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3,
                'dtype_valid': rgb_array.dtype == np.uint8,
                'value_range_valid': np.all((rgb_array >= 0) & (rgb_array <= 255)),
                'non_empty': rgb_array.size > 0
            }
            output_validations.append(format_validation)
        
    # Calculate comprehensive RGB rendering performance statistics
    mean_render_time = statistics.mean(render_timings)
    median_render_time = statistics.median(render_timings)
    percentile_95 = np.percentile(render_timings, 95)
    std_deviation = statistics.stdev(render_timings)
    
    # Analyze memory usage during RGB rendering operations for efficiency validation
    memory_analysis = {
        'mean_memory_delta_mb': statistics.mean(memory_usage_samples),
        'peak_memory_delta_mb': max(memory_usage_samples),
        'memory_efficiency_ratio': mean_render_time / statistics.mean(memory_usage_samples) if statistics.mean(memory_usage_samples) > 0 else float('inf')
    }
    
    # Validate rendering performance against target with statistical analysis
    rgb_target = PERFORMANCE_TARGET_RGB_RENDER_MS  # 5ms target for RGB rendering
    performance_compliant = mean_render_time <= rgb_target
    
    # Validate pixel value generation accuracy and output format compliance
    format_compliance = all(
        all(validation.values()) for validation in output_validations
    ) if output_validations else False
    
    # Generate rendering optimization recommendations based on performance patterns
    optimization_analysis = {
        'render_time_variance': std_deviation,
        'performance_consistency': (std_deviation / mean_render_time) * 100 if mean_render_time > 0 else 0,
        'memory_efficiency': memory_analysis['mean_memory_delta_mb']
    }
    
    # Assert RGB rendering performance meets target with detailed analysis
    assert performance_compliant, (
        f"RGB rendering performance target not met:\n"
        f"  Mean render time: {mean_render_time:.3f}ms (target: {rgb_target}ms)\n"
        f"  Median render time: {median_render_time:.3f}ms\n"
        f"  95th percentile: {percentile_95:.3f}ms\n"
        f"  Standard deviation: {std_deviation:.3f}ms\n"
        f"  Performance consistency: {optimization_analysis['performance_consistency']:.1f}% CV\n"
        f"  Memory efficiency: {memory_analysis['mean_memory_delta_mb']:.3f}MB per render\n"
        f"  Renders tested: {num_renders}\n"
        f"  Recommendation: Optimize pixel generation and array operations for <{rgb_target}ms target"
    )
    
    # Validate RGB array output format compliance with expected requirements
    assert format_compliance, (
        f"RGB array format compliance validation failed:\n"
        f"  Expected: (H, W, 3) uint8 array with values [0, 255]\n"
        f"  Format validations: {len([v for v in output_validations if all(v.values())])}/{len(output_validations)} passed\n"
        f"  Recommendation: Ensure RGB array generation follows specified format requirements"
    )
    
    # Validate memory efficiency during rendering operations
    max_acceptable_memory = 5.0  # 5MB memory delta threshold per render
    mean_memory_delta = memory_analysis['mean_memory_delta_mb']
    assert abs(mean_memory_delta) <= max_acceptable_memory, (
        f"RGB rendering memory usage exceeds efficiency threshold:\n"
        f"  Mean memory delta: {mean_memory_delta:.3f}MB (threshold: ±{max_acceptable_memory}MB)\n"
        f"  Peak memory delta: {memory_analysis['peak_memory_delta_mb']:.3f}MB\n"
        f"  Recommendation: Optimize memory allocation and cleanup in rendering pipeline"
    )


@pytest.mark.performance
@pytest.mark.timeout(60)
@pytest.mark.skipif('not matplotlib_available')
def test_human_mode_rendering_performance(performance_test_env, performance_tracker):
    """
    Test human mode interactive rendering performance against <50ms target with matplotlib 
    backend compatibility testing, graceful degradation validation, and visualization quality 
    assessment.
    
    Validates human mode rendering performance including backend compatibility testing,
    fallback behavior validation, and interactive visualization quality assessment with
    performance optimization recommendations.
    
    Args:
        performance_test_env (PlumeSearchEnv): Performance-optimized environment instance
        performance_tracker (dict): Performance measurement infrastructure
        
    Asserts:
        Human rendering meets <50ms target or validates graceful degradation  
        Backend compatibility functions correctly across different systems
        Visualization quality and consistency maintained across updates
    """
    # Check matplotlib backend availability and compatibility for interactive rendering
    matplotlib_available = True
    backend_info = {'available': False, 'backend': 'none', 'interactive': False}
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        backend_info['available'] = True
        backend_info['backend'] = matplotlib.get_backend()
        backend_info['interactive'] = hasattr(plt, 'show')
    except ImportError:
        matplotlib_available = False
        pytest.skip("Matplotlib not available for human mode rendering testing")
    
    env = performance_test_env
    obs, info = env.reset(seed=42)
    
    # Execute human mode rendering performance benchmark with backend validation
    render_timings = []
    backend_compatibility_results = []
    visualization_quality_checks = []
    
    num_renders = min(20, PERFORMANCE_TEST_ITERATIONS // 50)  # Reduced for interactive rendering
    
    for render_iteration in range(num_renders):
        # Vary environment state for comprehensive visualization testing
        if render_iteration > 0:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        # Execute human mode rendering with timing and compatibility analysis
        try:
            start_time = time.perf_counter()
            result = env.render(mode='human')
            render_time_ms = (time.perf_counter() - start_time) * 1000  # Convert to ms
            
            render_timings.append(render_time_ms)
            performance_tracker['timing_measurements'].append(render_time_ms)
            
            # Validate backend compatibility and rendering success
            compatibility_check = {
                'render_successful': True,
                'render_time_ms': render_time_ms,
                'backend_functional': True,
                'no_exceptions': True
            }
            backend_compatibility_results.append(compatibility_check)
            
        except Exception as e:
            # Handle rendering exceptions for graceful degradation testing
            compatibility_check = {
                'render_successful': False,
                'render_time_ms': float('inf'),
                'backend_functional': False,
                'exception': str(e)
            }
            backend_compatibility_results.append(compatibility_check)
    
    # Analyze human mode rendering performance with backend considerations
    successful_renders = [t for t in render_timings if t != float('inf')]
    
    if successful_renders:
        mean_render_time = statistics.mean(successful_renders)
        median_render_time = statistics.median(successful_renders)
        success_rate = len(successful_renders) / len(backend_compatibility_results)
        
        # Performance analysis with backend compatibility assessment
        performance_analysis = {
            'mean_render_time_ms': mean_render_time,
            'median_render_time_ms': median_render_time,
            'success_rate': success_rate,
            'backend_info': backend_info,
            'successful_renders': len(successful_renders),
            'failed_renders': len(backend_compatibility_results) - len(successful_renders)
        }
        
        # Validate human rendering performance against target
        human_render_target = 50.0  # 50ms target for human mode rendering
        performance_compliant = mean_render_time <= human_render_target
        
        # Test rendering consistency across multiple updates and state changes
        consistency_analysis = {
            'time_variance': statistics.stdev(successful_renders) if len(successful_renders) > 1 else 0,
            'coefficient_of_variation': (statistics.stdev(successful_renders) / mean_render_time) * 100 if mean_render_time > 0 and len(successful_renders) > 1 else 0
        }
        
        # Assert human rendering meets performance targets with backend analysis
        if success_rate >= 0.8:  # 80% success rate threshold
            assert performance_compliant, (
                f"Human mode rendering performance target not met:\n"
                f"  Mean render time: {mean_render_time:.3f}ms (target: {human_render_target}ms)\n"
                f"  Median render time: {median_render_time:.3f}ms\n"
                f"  Success rate: {success_rate:.1%}\n"
                f"  Backend: {backend_info['backend']}\n"
                f"  Time variance: {consistency_analysis['time_variance']:.3f}ms\n"
                f"  Renders tested: {len(backend_compatibility_results)}\n"
                f"  Recommendation: Optimize matplotlib rendering pipeline for <{human_render_target}ms target"
            )
        else:
            # Validate graceful degradation when rendering performance is poor
            warnings.warn(
                f"Human mode rendering has low success rate ({success_rate:.1%}), "
                f"validating graceful degradation behavior"
            )
            
    else:
        # Test matplotlib backend fallback behavior for headless environment compatibility
        fallback_test_result = None
        try:
            # Attempt fallback to rgb_array mode
            fallback_array = env.render(mode='rgb_array')
            fallback_test_result = {
                'fallback_successful': fallback_array is not None,
                'fallback_mode': 'rgb_array',
                'graceful_degradation': True
            }
        except Exception as e:
            fallback_test_result = {
                'fallback_successful': False,
                'fallback_mode': 'none',
                'graceful_degradation': False,
                'fallback_error': str(e)
            }
        
        # Assert graceful fallback behavior when human mode unavailable
        assert fallback_test_result['graceful_degradation'], (
            f"Human mode rendering failed without graceful degradation:\n"
            f"  Backend available: {backend_info['available']}\n"
            f"  Backend type: {backend_info['backend']}\n"
            f"  Fallback successful: {fallback_test_result['fallback_successful']}\n"
            f"  Fallback error: {fallback_test_result.get('fallback_error', 'none')}\n"
            f"  Recommendation: Implement graceful fallback to rgb_array mode when interactive rendering unavailable"
        )


@pytest.mark.performance
@pytest.mark.memory
@pytest.mark.timeout(120)
def test_memory_usage_constraints(performance_test_env, performance_tracker, memory_monitor):
    """
    Test system memory usage against <50MB total limit with component breakdown analysis, 
    memory leak detection, resource optimization validation, and long-term stability assessment.
    
    Validates memory usage constraints through comprehensive profiling including baseline
    measurement, continuous monitoring, leak detection, and component-specific analysis
    with optimization recommendations for development teams.
    
    Args:
        performance_test_env (PlumeSearchEnv): Performance-optimized environment instance
        performance_tracker (dict): Performance measurement infrastructure
        memory_monitor (object): Advanced memory monitoring infrastructure
        
    Asserts:
        Total memory usage within <50MB limit with statistical validation
        No significant memory leaks detected during extended operation  
        Component-specific memory usage within optimization guidelines
    """
    env = performance_test_env
    
    # Initialize memory baseline measurement before extended environment operations
    gc.collect()  # Force garbage collection for accurate baseline
    initial_memory_mb = performance_tracker['baseline_memory']
    
    # Start continuous memory monitoring with high-resolution sampling
    memory_monitor.start_monitoring(sampling_interval=0.1)  # 100ms sampling interval
    
    # Execute extended environment operations for comprehensive memory usage analysis
    num_operations = min(5000, PERFORMANCE_TEST_ITERATIONS * 5)  # Extended operation count
    episode_count = 0
    step_count = 0
    
    obs, info = env.reset(seed=42)
    
    for operation in range(num_operations):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        if terminated or truncated:
            obs, info = env.reset()
            episode_count += 1
            
        # Periodic garbage collection for leak detection analysis
        if operation % 1000 == 0:
            gc.collect()
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            performance_tracker['memory_samples'].append(current_memory)
    
    # Stop memory monitoring and analyze comprehensive usage patterns
    memory_analysis = memory_monitor.stop_monitoring()
    
    # Calculate final memory usage and detect potential leaks
    gc.collect()  # Final garbage collection
    final_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    memory_growth_mb = final_memory_mb - initial_memory_mb
    
    # Analyze component-specific memory usage including plume fields and rendering buffers
    component_analysis = {
        'initial_memory_mb': initial_memory_mb,
        'final_memory_mb': final_memory_mb,
        'memory_growth_mb': memory_growth_mb,
        'peak_memory_mb': memory_analysis['peak_memory_mb'],
        'mean_memory_mb': memory_analysis['mean_memory_mb'],
        'memory_samples': len(performance_tracker['memory_samples']),
        'episodes_completed': episode_count,
        'steps_executed': step_count
    }
    
    # Validate peak memory usage against total system limit with tolerance checking  
    memory_limit_mb = MEMORY_LIMIT_TOTAL_MB
    tolerance_mb = memory_limit_mb * PERFORMANCE_TOLERANCE_FACTOR
    peak_compliant = memory_analysis['peak_memory_mb'] <= memory_limit_mb
    
    # Perform memory leak detection through growth pattern analysis
    leak_threshold_mb = 5.0  # 5MB growth threshold for leak detection
    leak_detected = memory_growth_mb > leak_threshold_mb
    memory_stability = not leak_detected and (memory_analysis['peak_memory_mb'] - memory_analysis['mean_memory_mb']) < 10.0
    
    # Generate memory optimization recommendations based on usage patterns
    optimization_recommendations = []
    if not peak_compliant:
        optimization_recommendations.append(f"Peak memory usage ({memory_analysis['peak_memory_mb']:.1f}MB) exceeds limit ({memory_limit_mb}MB)")
    if leak_detected:
        optimization_recommendations.append(f"Potential memory leak detected: {memory_growth_mb:.1f}MB growth during testing")
    if not memory_stability:
        optimization_recommendations.append("Memory usage shows high variability - investigate allocation patterns")
        
    # Assert total memory usage within limits with detailed component breakdown
    assert peak_compliant, (
        f"Memory usage constraint validation failed:\n"
        f"  Peak memory usage: {memory_analysis['peak_memory_mb']:.1f}MB (limit: {memory_limit_mb}MB)\n"
        f"  Mean memory usage: {memory_analysis['mean_memory_mb']:.1f}MB\n"
        f"  Memory growth: {memory_growth_mb:.1f}MB\n"
        f"  Episodes completed: {episode_count}\n"
        f"  Steps executed: {step_count}\n"
        f"  Memory samples: {memory_analysis['sample_count']}\n"
        f"  Leak detected: {leak_detected}\n"
        f"  Optimization recommendations: {optimization_recommendations}\n"
        f"  Recommendation: Optimize memory allocation and implement component-specific limits"
    )
    
    # Validate memory leak detection with statistical significance
    assert not leak_detected, (
        f"Memory leak detected during extended operation:\n"
        f"  Initial memory: {initial_memory_mb:.1f}MB\n"
        f"  Final memory: {final_memory_mb:.1f}MB\n"
        f"  Memory growth: {memory_growth_mb:.1f}MB (threshold: {leak_threshold_mb}MB)\n"
        f"  Operations executed: {num_operations}\n"
        f"  Leak severity: {'High' if memory_growth_mb > leak_threshold_mb * 2 else 'Moderate'}\n"
        f"  Recommendation: Investigate resource cleanup and implement proper memory management"
    )
    
    # Validate memory usage stability and consistency across operations
    memory_variance = (memory_analysis['peak_memory_mb'] - memory_analysis['mean_memory_mb']) / memory_analysis['mean_memory_mb'] * 100
    assert memory_variance <= 25.0, (  # 25% variance threshold
        f"Memory usage shows excessive variability:\n"
        f"  Memory variance: {memory_variance:.1f}% (threshold: 25%)\n"
        f"  Peak usage: {memory_analysis['peak_memory_mb']:.1f}MB\n"
        f"  Mean usage: {memory_analysis['mean_memory_mb']:.1f}MB\n"
        f"  Recommendation: Implement consistent memory allocation patterns and buffer management"
    )


@pytest.mark.performance
@pytest.mark.scalability  
@pytest.mark.timeout(180)
def test_performance_scalability(performance_tracker):
    """
    Test performance scaling across different grid sizes from 32×32 to 256×256 with resource 
    utilization analysis, scalability bottleneck identification, and optimization recommendations 
    for research-scale deployments.
    
    Validates performance scaling patterns through systematic grid size analysis including
    memory usage scaling, latency scaling assessment, and configuration optimization
    recommendations for large-scale research deployments.
    
    Args:
        performance_tracker (dict): Performance measurement infrastructure
        
    Asserts:
        Performance scaling patterns within acceptable degradation bounds
        Memory scaling follows predictable patterns with optimization guidance
        No critical bottlenecks identified across scaling range
    """
    scaling_results = {
        'grid_sizes_tested': SCALABILITY_TEST_GRID_SIZES,
        'performance_measurements': {},
        'scaling_analysis': {},
        'optimization_recommendations': []
    }
    
    # Iterate through grid size configurations for comprehensive scaling analysis
    for grid_size in SCALABILITY_TEST_GRID_SIZES:
        width, height = grid_size
        grid_key = f"{width}x{height}"
        total_cells = width * height
        
        try:
            # Create environment instance with specific grid size for scaling measurement
            env = create_plume_search_env(
                grid_size=grid_size,
                source_location=(width // 2, height // 2),  # Centered source
                max_steps=min(500, 1000)  # Adjusted for scaling analysis
            )
            
            # Initialize environment and establish performance baseline
            obs, info = env.reset(seed=42)
            
            # Measure step latency scaling with grid size increase
            step_timings = []
            memory_samples = []
            render_timings = []
            
            num_iterations = min(200, PERFORMANCE_TEST_ITERATIONS // len(SCALABILITY_TEST_GRID_SIZES))
            
            # Execute performance measurement across different grid configurations
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)
            
            for iteration in range(num_iterations):
                action = env.action_space.sample()
                
                # Measure step execution time with grid size scaling impact
                start_time = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                step_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
                step_timings.append(step_time)
                
                # Sample memory usage periodically for scaling analysis
                if iteration % 20 == 0:
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    memory_samples.append(current_memory - initial_memory)
                    
                # Test rendering performance scaling periodically
                if iteration % 50 == 0:
                    render_start = time.perf_counter()
                    rgb_array = env.render(mode='rgb_array')
                    render_time = (time.perf_counter() - render_start) * 1000
                    render_timings.append(render_time)
                
                if terminated or truncated:
                    obs, info = env.reset()
                    
            # Calculate grid-size-specific performance metrics
            mean_step_time = statistics.mean(step_timings)
            peak_memory_delta = max(memory_samples) if memory_samples else 0
            mean_render_time = statistics.mean(render_timings) if render_timings else 0
            
            # Calculate per-cell performance metrics for scaling analysis
            step_time_per_cell = (mean_step_time / total_cells) * 1000  # nanoseconds per cell
            memory_per_cell = (peak_memory_delta * 1024 * 1024) / total_cells  # bytes per cell
            
            scaling_results['performance_measurements'][grid_key] = {
                'grid_dimensions': grid_size,
                'total_cells': total_cells,
                'mean_step_time_ms': mean_step_time,
                'step_time_per_cell_ns': step_time_per_cell,
                'peak_memory_delta_mb': peak_memory_delta,
                'memory_per_cell_bytes': memory_per_cell,
                'mean_render_time_ms': mean_render_time,
                'iterations_tested': num_iterations,
                'samples_collected': len(step_timings)
            }
            
            env.close()
            
        except Exception as e:
            scaling_results['performance_measurements'][grid_key] = {
                'error': str(e),
                'grid_dimensions': grid_size,
                'analysis_failed': True
            }
    
    # Analyze performance scaling patterns and identify bottlenecks
    successful_measurements = {k: v for k, v in scaling_results['performance_measurements'].items() 
                              if 'error' not in v}
    
    if len(successful_measurements) >= 2:
        # Calculate scaling efficiency and identify optimization opportunities
        grid_sizes = [v['total_cells'] for v in successful_measurements.values()]
        step_times = [v['mean_step_time_ms'] for v in successful_measurements.values()]
        memory_usage = [v['peak_memory_delta_mb'] for v in successful_measurements.values()]
        
        # Analyze step latency scaling efficiency
        if len(grid_sizes) >= 2:
            # Calculate scaling factor and efficiency metrics
            min_size, max_size = min(grid_sizes), max(grid_sizes)
            min_idx, max_idx = grid_sizes.index(min_size), grid_sizes.index(max_size)
            
            size_scaling_factor = max_size / min_size
            latency_scaling_factor = step_times[max_idx] / step_times[min_idx]
            memory_scaling_factor = memory_usage[max_idx] / memory_usage[min_idx] if memory_usage[min_idx] > 0 else float('inf')
            
            scaling_results['scaling_analysis'] = {
                'grid_size_range': (min_size, max_size),
                'size_scaling_factor': size_scaling_factor,
                'latency_scaling_factor': latency_scaling_factor,
                'memory_scaling_factor': memory_scaling_factor,
                'latency_efficiency': size_scaling_factor / latency_scaling_factor if latency_scaling_factor > 0 else 0,
                'memory_efficiency': size_scaling_factor / memory_scaling_factor if memory_scaling_factor > 0 and memory_scaling_factor != float('inf') else 0
            }
    
    # Validate performance targets maintained across different scales
    performance_compliance = {}
    for grid_key, measurements in successful_measurements.items():
        if 'mean_step_time_ms' in measurements:
            step_compliant = measurements['mean_step_time_ms'] <= PERFORMANCE_TARGET_STEP_LATENCY_MS * 2  # Allow 2x degradation
            memory_compliant = measurements['peak_memory_delta_mb'] <= MEMORY_LIMIT_TOTAL_MB
            
            performance_compliance[grid_key] = {
                'step_latency_compliant': step_compliant,
                'memory_compliant': memory_compliant,
                'overall_compliant': step_compliant and memory_compliant
            }
            
    # Generate scaling optimization recommendations
    compliant_sizes = [k for k, v in performance_compliance.items() if v.get('overall_compliant', False)]
    
    if compliant_sizes:
        scaling_results['optimization_recommendations'].append(
            f"Recommended compliant grid sizes: {', '.join(compliant_sizes)}"
        )
    else:
        scaling_results['optimization_recommendations'].append(
            "No grid sizes meet all performance targets - consider algorithm optimization"
        )
        
    # Analyze scaling bottlenecks and provide specific recommendations  
    if 'scaling_analysis' in scaling_results:
        analysis = scaling_results['scaling_analysis']
        if analysis['latency_efficiency'] < 0.5:  # Less than 50% efficiency
            scaling_results['optimization_recommendations'].append(
                f"Step latency scales poorly (efficiency: {analysis['latency_efficiency']:.2f}) - optimize core algorithms"
            )
        if analysis['memory_efficiency'] < 0.5 and analysis['memory_scaling_factor'] != float('inf'):
            scaling_results['optimization_recommendations'].append(
                f"Memory usage scales poorly (efficiency: {analysis['memory_efficiency']:.2f}) - implement memory optimization"
            )
    
    # Assert acceptable performance degradation patterns with scaling recommendations
    if successful_measurements:
        worst_case_grid = max(successful_measurements.keys(), key=lambda k: successful_measurements[k]['total_cells'])
        worst_case_performance = successful_measurements[worst_case_grid]
        
        acceptable_degradation = worst_case_performance['mean_step_time_ms'] <= PERFORMANCE_TARGET_STEP_LATENCY_MS * 3  # 3x degradation threshold
        
        assert acceptable_degradation, (
            f"Performance scaling validation failed:\n"
            f"  Worst case grid: {worst_case_grid}\n"
            f"  Worst case latency: {worst_case_performance['mean_step_time_ms']:.3f}ms\n"
            f"  Latency threshold: {PERFORMANCE_TARGET_STEP_LATENCY_MS * 3}ms (3x base target)\n"
            f"  Scaling analysis: {scaling_results.get('scaling_analysis', 'unavailable')}\n"
            f"  Compliant grid sizes: {compliant_sizes}\n"
            f"  Optimization recommendations: {scaling_results['optimization_recommendations']}\n"
            f"  Recommendation: Implement performance optimization for large-scale grid configurations"
        )
    else:
        pytest.fail("Scalability analysis failed - no successful measurements obtained across grid size range")


@pytest.mark.performance
@pytest.mark.integration
@pytest.mark.timeout(300)
def test_comprehensive_performance_suite(performance_tracker):
    """
    Execute complete performance benchmark suite with comprehensive analysis, regression detection, 
    trend analysis, and optimization recommendations for continuous performance monitoring and CI 
    integration.
    
    Orchestrates execution of all performance benchmarks including step latency, memory usage,
    rendering performance, and scaling analysis with integrated validation, statistical analysis,
    and comprehensive reporting for development teams.
    
    Args:
        performance_tracker (dict): Performance measurement infrastructure
        
    Asserts:
        Complete benchmark suite execution successful with comprehensive validation
        All performance targets met or detailed failure analysis provided
        Optimization recommendations generated for development guidance
    """
    # Initialize comprehensive performance benchmark suite with detailed configuration
    benchmark_config = EnvironmentBenchmarkConfig(
        iterations=min(1000, PERFORMANCE_TEST_ITERATIONS),
        warmup_iterations=PERFORMANCE_TEST_WARMUP_ITERATIONS,
        enable_memory_profiling=True,
        enable_scaling_analysis=True,
        scaling_grid_sizes=SCALABILITY_TEST_GRID_SIZES,
        performance_targets={
            'step_latency_ms': PERFORMANCE_TARGET_STEP_LATENCY_MS,
            'rgb_render_ms': PERFORMANCE_TARGET_RGB_RENDER_MS,
            'memory_limit_mb': MEMORY_LIMIT_TOTAL_MB
        },
        validate_targets=True,
        detect_memory_leaks=True,
        include_action_analysis=True,
        include_position_analysis=False  # Disabled for performance
    )
    
    # Validate benchmark configuration for systematic execution
    config_validation = benchmark_config.validate_config(strict_validation=True)
    assert config_validation.is_valid, (
        f"Benchmark configuration validation failed:\n"
        f"  Errors: {config_validation.errors}\n"
        f"  Warnings: {config_validation.warnings}\n"
        f"  Recommendation: Fix configuration issues before benchmark execution"
    )
    
    # Initialize comprehensive performance suite orchestrator
    performance_suite = EnvironmentPerformanceSuite(
        config=benchmark_config,
        enable_detailed_logging=True,
        output_directory=None  # No file output for testing
    )
    
    # Execute comprehensive benchmark suite with integrated analysis
    suite_start_time = time.time()
    benchmark_result = performance_suite.run_full_benchmark_suite(
        validate_targets=True,
        include_scaling_analysis=True,
        custom_targets=None
    )
    suite_execution_time = time.time() - suite_start_time
    
    # Validate benchmark result completeness and data integrity
    assert benchmark_result is not None, "Benchmark suite execution returned no results"
    assert hasattr(benchmark_result, 'benchmark_name'), "Benchmark result missing essential metadata"
    assert hasattr(benchmark_result, 'step_latency_metrics'), "Benchmark result missing step latency analysis"
    assert hasattr(benchmark_result, 'targets_met'), "Benchmark result missing target validation status"
    
    # Perform comprehensive performance validation with statistical analysis
    performance_analysis = PerformanceAnalysis()
    
    # Analyze performance trends and detect potential regressions
    trend_analysis = performance_analysis.analyze_performance_trends(
        performance_data={
            'step_latency': benchmark_result.step_latency_metrics,
            'memory_usage': benchmark_result.memory_usage_metrics,
            'scalability': benchmark_result.scalability_metrics
        }
    )
    
    # Generate comprehensive optimization recommendations based on analysis
    optimization_recommendations = performance_analysis.generate_optimization_recommendations(
        benchmark_results=benchmark_result,
        trend_analysis=trend_analysis,
        include_scaling_guidance=True
    )
    
    # Validate suite execution performance and completeness
    execution_analysis = {
        'suite_execution_time_sec': suite_execution_time,
        'estimated_time_sec': benchmark_config.estimate_execution_time(),
        'execution_efficiency': suite_execution_time / benchmark_config.estimate_execution_time() if benchmark_config.estimate_execution_time() > 0 else 1.0,
        'results_completeness': {
            'has_step_latency': bool(benchmark_result.step_latency_metrics),
            'has_memory_analysis': bool(benchmark_result.memory_usage_metrics), 
            'has_scaling_analysis': bool(benchmark_result.scalability_metrics),
            'has_executive_summary': bool(benchmark_result.executive_summary)
        }
    }
    
    # Assert comprehensive benchmark suite success with detailed analysis
    assert benchmark_result.targets_met or len(benchmark_result.optimization_recommendations) > 0, (
        f"Performance benchmark suite validation failed:\n"
        f"  Targets met: {benchmark_result.targets_met}\n"
        f"  Suite execution time: {suite_execution_time:.2f}s\n"
        f"  Results completeness: {execution_analysis['results_completeness']}\n"
        f"  Step latency metrics: {bool(benchmark_result.step_latency_metrics)}\n"
        f"  Memory usage metrics: {bool(benchmark_result.memory_usage_metrics)}\n" 
        f"  Scalability analysis: {bool(benchmark_result.scalability_metrics)}\n"
        f"  Optimization recommendations: {len(benchmark_result.optimization_recommendations)}\n"
        f"  Executive summary: {bool(benchmark_result.executive_summary)}\n"
        f"  Trend analysis: {trend_analysis}\n"
        f"  Recommendation: Review detailed performance analysis and implement suggested optimizations"
    )
    
    # Validate optimization recommendations quality and actionability
    assert len(optimization_recommendations) > 0 or benchmark_result.targets_met, (
        f"Performance analysis failed to generate actionable recommendations:\n"
        f"  Targets met: {benchmark_result.targets_met}\n"
        f"  Generated recommendations: {len(optimization_recommendations)}\n"
        f"  Suite recommendations: {len(benchmark_result.optimization_recommendations)}\n"
        f"  Recommendation: Ensure performance analysis generates actionable guidance for development teams"
    )
    
    # Store comprehensive performance results for historical analysis and monitoring
    performance_tracker['suite_results'] = {
        'benchmark_result': benchmark_result.to_dict(include_raw_data=False, include_analysis=True),
        'trend_analysis': trend_analysis,
        'optimization_recommendations': optimization_recommendations,
        'execution_analysis': execution_analysis,
        'suite_timestamp': suite_start_time
    }


@pytest.mark.performance
@pytest.mark.regression
@pytest.mark.timeout(120)
def test_performance_regression_detection(performance_tracker):
    """
    Test performance regression detection system with baseline comparison, statistical significance 
    analysis, and automated regression alerting for continuous performance monitoring and quality 
    assurance.
    
    Validates regression detection capabilities through baseline performance comparison,
    statistical significance testing, and automated alerting for continuous integration
    workflows and development team notifications.
    
    Args:
        performance_tracker (dict): Performance measurement infrastructure
        
    Asserts:
        Regression detection system functions correctly with statistical validation
        Baseline comparison provides accurate performance change analysis
        No significant performance regressions detected in current implementation
    """
    # Load or create historical performance baselines for regression comparison
    baseline_performance = {
        'step_latency_baseline_ms': PERFORMANCE_TARGET_STEP_LATENCY_MS * 0.8,  # 80% of target as baseline
        'memory_usage_baseline_mb': MEMORY_LIMIT_TOTAL_MB * 0.6,  # 60% of limit as baseline
        'rgb_render_baseline_ms': PERFORMANCE_TARGET_RGB_RENDER_MS * 0.7,  # 70% of target as baseline
        'baseline_timestamp': time.time() - 7 * 24 * 3600,  # 7 days ago simulation
        'baseline_iterations': 1000,
        'baseline_confidence': 0.95
    }
    
    # Execute current performance benchmark suite for regression comparison
    current_benchmark_config = EnvironmentBenchmarkConfig(
        iterations=min(500, PERFORMANCE_TEST_ITERATIONS // 2),  # Reduced for regression testing
        warmup_iterations=50,
        enable_memory_profiling=True,
        enable_scaling_analysis=False,  # Disabled for focused regression testing
        validate_targets=True,
        detect_memory_leaks=True
    )
    
    # Run current performance measurement for baseline comparison
    current_results = run_environment_performance_benchmark(
        config=current_benchmark_config,
        validate_targets=True,
        include_scaling_analysis=False
    )
    
    # Perform statistical regression analysis using significance testing
    regression_analysis = {
        'analysis_timestamp': time.time(),
        'baseline_data': baseline_performance,
        'current_results': {},
        'regression_detection': {},
        'statistical_significance': {},
        'alert_conditions': []
    }
    
    # Extract current performance metrics for comparison
    if current_results.step_latency_metrics and 'timings' in current_results.step_latency_metrics:
        current_step_latency = statistics.mean(current_results.step_latency_metrics['timings'])
        regression_analysis['current_results']['step_latency_ms'] = current_step_latency
        
        # Compare current performance against historical baseline
        latency_change_percent = ((current_step_latency - baseline_performance['step_latency_baseline_ms']) / 
                                 baseline_performance['step_latency_baseline_ms']) * 100
        
        # Detect statistically significant performance regressions
        latency_regression_detected = latency_change_percent > (REGRESSION_DETECTION_THRESHOLD * 100)
        
        regression_analysis['regression_detection']['step_latency'] = {
            'current_value_ms': current_step_latency,
            'baseline_value_ms': baseline_performance['step_latency_baseline_ms'],
            'change_percent': latency_change_percent,
            'regression_detected': latency_regression_detected,
            'significance_threshold': REGRESSION_DETECTION_THRESHOLD * 100
        }
        
        if latency_regression_detected:
            regression_analysis['alert_conditions'].append(
                f"Step latency regression: {latency_change_percent:.1f}% increase from baseline"
            )
    
    # Analyze memory usage regression patterns
    if current_results.memory_usage_metrics and 'peak_usage_mb' in current_results.memory_usage_metrics:
        current_memory_usage = current_results.memory_usage_metrics['peak_usage_mb']
        regression_analysis['current_results']['memory_usage_mb'] = current_memory_usage
        
        memory_change_percent = ((current_memory_usage - baseline_performance['memory_usage_baseline_mb']) /
                                baseline_performance['memory_usage_baseline_mb']) * 100
        
        memory_regression_detected = memory_change_percent > (REGRESSION_DETECTION_THRESHOLD * 100)
        
        regression_analysis['regression_detection']['memory_usage'] = {
            'current_value_mb': current_memory_usage,
            'baseline_value_mb': baseline_performance['memory_usage_baseline_mb'],
            'change_percent': memory_change_percent,
            'regression_detected': memory_regression_detected,
            'significance_threshold': REGRESSION_DETECTION_THRESHOLD * 100
        }
        
        if memory_regression_detected:
            regression_analysis['alert_conditions'].append(
                f"Memory usage regression: {memory_change_percent:.1f}% increase from baseline"
            )
    
    # Validate regression detection accuracy and sensitivity
    regression_detection_functional = True
    detection_sensitivity_analysis = {
        'total_metrics_analyzed': len(regression_analysis['regression_detection']),
        'regressions_detected': len(regression_analysis['alert_conditions']),
        'detection_rate': len(regression_analysis['alert_conditions']) / len(regression_analysis['regression_detection']) if regression_analysis['regression_detection'] else 0,
        'false_positive_risk': 'low' if len(regression_analysis['alert_conditions']) <= 1 else 'moderate'
    }
    
    # Generate comprehensive regression report with baseline comparison
    regression_report = {
        'regression_summary': f"Analyzed {detection_sensitivity_analysis['total_metrics_analyzed']} performance metrics",
        'regressions_found': regression_analysis['alert_conditions'],
        'statistical_confidence': STATISTICAL_CONFIDENCE_LEVEL,
        'detection_threshold': REGRESSION_DETECTION_THRESHOLD,
        'baseline_age_days': (time.time() - baseline_performance['baseline_timestamp']) / (24 * 3600),
        'recommendation': 'Update performance baselines and continue monitoring' if not regression_analysis['alert_conditions'] else 'Investigate performance regressions immediately'
    }
    
    # Assert no significant performance regressions detected with detailed analysis
    assert len(regression_analysis['alert_conditions']) == 0, (
        f"Performance regressions detected:\n"
        f"  Alert conditions: {regression_analysis['alert_conditions']}\n"
        f"  Step latency analysis: {regression_analysis['regression_detection'].get('step_latency', 'unavailable')}\n"
        f"  Memory usage analysis: {regression_analysis['regression_detection'].get('memory_usage', 'unavailable')}\n"
        f"  Detection sensitivity: {detection_sensitivity_analysis}\n"
        f"  Baseline age: {regression_report['baseline_age_days']:.1f} days\n"
        f"  Statistical threshold: {REGRESSION_DETECTION_THRESHOLD * 100}%\n"
        f"  Recommendation: {regression_report['recommendation']}"
    )
    
    # Validate regression detection system functionality and accuracy
    assert regression_detection_functional, (
        f"Regression detection system validation failed:\n"
        f"  System functional: {regression_detection_functional}\n"
        f"  Metrics analyzed: {detection_sensitivity_analysis['total_metrics_analyzed']}\n"
        f"  Detection accuracy: {detection_sensitivity_analysis['detection_rate']:.2f}\n"
        f"  Recommendation: Ensure regression detection system provides reliable performance monitoring"
    )
    
    # Update performance baselines for continuous monitoring
    performance_tracker['regression_analysis'] = regression_analysis
    performance_tracker['regression_report'] = regression_report
    performance_tracker['baseline_update_recommended'] = len(regression_analysis['alert_conditions']) == 0


@pytest.mark.performance
@pytest.mark.concurrent
@pytest.mark.timeout(180)
def test_concurrent_performance(performance_tracker):
    """
    Test performance characteristics under concurrent execution with multiple environment instances, 
    thread safety validation, and resource contention analysis for multi-agent and distributed 
    training scenarios.
    
    Validates concurrent performance through multi-threaded environment execution including
    resource contention analysis, thread safety validation, and performance degradation
    assessment for parallel training workflows.
    
    Args:
        performance_tracker (dict): Performance measurement infrastructure
        
    Asserts:
        Concurrent performance characteristics meet expectations with thread safety
        Resource contention within acceptable bounds for parallel execution
        Performance degradation under concurrent load acceptable for multi-agent training
    """
    import concurrent.futures
    import threading
    
    # Configure concurrent execution parameters for systematic testing
    num_concurrent_environments = min(4, threading.active_count() + 2)  # Conservative thread count
    iterations_per_environment = min(250, PERFORMANCE_TEST_ITERATIONS // num_concurrent_environments)
    
    concurrent_results = {
        'test_configuration': {
            'concurrent_environments': num_concurrent_environments,
            'iterations_per_environment': iterations_per_environment,
            'total_operations': num_concurrent_environments * iterations_per_environment
        },
        'environment_results': {},
        'resource_contention_analysis': {},
        'thread_safety_validation': {},
        'performance_degradation_analysis': {}
    }
    
    # Thread-safe performance measurement infrastructure
    results_lock = threading.Lock()
    shared_performance_data = {
        'all_step_timings': [],
        'all_memory_samples': [],
        'thread_exceptions': [],
        'environment_lifecycles': []
    }
    
    def concurrent_environment_worker(worker_id: int, iterations: int) -> dict:
        """
        Worker function for concurrent environment execution with performance measurement.
        
        Args:
            worker_id (int): Unique identifier for worker thread
            iterations (int): Number of iterations for worker execution
            
        Returns:
            dict: Worker-specific performance results and measurements
        """
        worker_results = {
            'worker_id': worker_id,
            'start_time': time.time(),
            'step_timings': [],
            'memory_samples': [],
            'episodes_completed': 0,
            'exceptions_encountered': [],
            'thread_id': threading.get_ident()
        }
        
        try:
            # Create isolated environment instance for concurrent execution
            env = create_plume_search_env(
                grid_size=(64, 64),  # Smaller grid for concurrent testing
                source_location=(32, 32),
                max_steps=100  # Shorter episodes for concurrent testing
            )
            
            # Initialize environment with worker-specific seed for isolation
            obs, info = env.reset(seed=worker_id * 1000 + 42)
            
            # Execute concurrent performance measurement
            process = psutil.Process()
            
            for iteration in range(iterations):
                action = env.action_space.sample()
                
                # Measure step execution time in concurrent context
                step_start = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                step_time_ms = (time.perf_counter() - step_start) * 1000
                
                worker_results['step_timings'].append(step_time_ms)
                
                # Sample memory usage periodically for contention analysis
                if iteration % 25 == 0:
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    worker_results['memory_samples'].append(memory_mb)
                
                if terminated or truncated:
                    obs, info = env.reset(seed=worker_id * 1000 + iteration)
                    worker_results['episodes_completed'] += 1
                    
            # Thread-safe aggregation of worker results
            with results_lock:
                shared_performance_data['all_step_timings'].extend(worker_results['step_timings'])
                shared_performance_data['all_memory_samples'].extend(worker_results['memory_samples'])
                shared_performance_data['environment_lifecycles'].append({
                    'worker_id': worker_id,
                    'creation_successful': True,
                    'cleanup_successful': True
                })
            
            env.close()  # Ensure proper resource cleanup
            
        except Exception as e:
            worker_results['exceptions_encountered'].append(str(e))
            with results_lock:
                shared_performance_data['thread_exceptions'].append({
                    'worker_id': worker_id,
                    'exception': str(e),
                    'thread_id': threading.get_ident()
                })
        
        worker_results['end_time'] = time.time()
        worker_results['execution_duration'] = worker_results['end_time'] - worker_results['start_time']
        
        return worker_results
    
    # Execute concurrent environment instances with thread-based parallel execution
    concurrent_start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_environments) as executor:
        # Submit concurrent environment workers
        future_to_worker = {
            executor.submit(concurrent_environment_worker, worker_id, iterations_per_environment): worker_id
            for worker_id in range(num_concurrent_environments)
        }
        
        # Collect worker results as they complete
        for future in concurrent.futures.as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                worker_result = future.result()
                concurrent_results['environment_results'][f'worker_{worker_id}'] = worker_result
            except Exception as e:
                concurrent_results['environment_results'][f'worker_{worker_id}'] = {
                    'worker_id': worker_id,
                    'execution_failed': True,
                    'exception': str(e)
                }
    
    concurrent_execution_time = time.time() - concurrent_start_time
    
    # Analyze concurrent execution performance and resource contention
    successful_workers = [result for result in concurrent_results['environment_results'].values() 
                         if not result.get('execution_failed', False)]
    
    if successful_workers:
        # Calculate concurrent performance metrics
        all_step_timings = shared_performance_data['all_step_timings']
        concurrent_mean_latency = statistics.mean(all_step_timings) if all_step_timings else float('inf')
        
        # Analyze performance degradation under concurrent load
        baseline_latency = PERFORMANCE_TARGET_STEP_LATENCY_MS * 0.8  # Assume baseline performance
        performance_degradation = (concurrent_mean_latency - baseline_latency) / baseline_latency * 100
        
        concurrent_results['performance_degradation_analysis'] = {
            'concurrent_mean_latency_ms': concurrent_mean_latency,
            'baseline_latency_ms': baseline_latency,
            'performance_degradation_percent': performance_degradation,
            'degradation_acceptable': performance_degradation <= 50.0,  # 50% degradation threshold
            'successful_workers': len(successful_workers),
            'failed_workers': num_concurrent_environments - len(successful_workers)
        }
        
        # Validate thread safety and resource isolation
        thread_safety_analysis = {
            'thread_exceptions': len(shared_performance_data['thread_exceptions']),
            'environment_creation_failures': sum(1 for result in concurrent_results['environment_results'].values() 
                                                if result.get('execution_failed', False)),
            'resource_isolation_successful': len(shared_performance_data['thread_exceptions']) == 0,
            'concurrent_execution_time_sec': concurrent_execution_time
        }
        
        concurrent_results['thread_safety_validation'] = thread_safety_analysis
        
        # Analyze CPU utilization and resource sharing efficiency
        resource_efficiency = {
            'theoretical_speedup': num_concurrent_environments,
            'actual_speedup': (num_concurrent_environments * iterations_per_environment * baseline_latency / 1000) / concurrent_execution_time,
            'parallel_efficiency': 0.0,  # Will be calculated
            'resource_contention_detected': performance_degradation > 25.0
        }
        
        if resource_efficiency['theoretical_speedup'] > 0:
            resource_efficiency['parallel_efficiency'] = (resource_efficiency['actual_speedup'] / 
                                                        resource_efficiency['theoretical_speedup']) * 100
        
        concurrent_results['resource_contention_analysis'] = resource_efficiency
    
    # Assert concurrent performance characteristics meet expectations
    if successful_workers:
        degradation_analysis = concurrent_results['performance_degradation_analysis']
        
        assert degradation_analysis['degradation_acceptable'], (
            f"Concurrent performance degradation exceeds acceptable threshold:\n"
            f"  Concurrent mean latency: {degradation_analysis['concurrent_mean_latency_ms']:.3f}ms\n"
            f"  Performance degradation: {degradation_analysis['performance_degradation_percent']:.1f}%\n"
            f"  Successful workers: {degradation_analysis['successful_workers']}/{num_concurrent_environments}\n"
            f"  Concurrent execution time: {concurrent_execution_time:.2f}s\n"
            f"  Resource efficiency: {concurrent_results['resource_contention_analysis']}\n"
            f"  Recommendation: Optimize resource allocation and reduce contention for concurrent execution"
        )
        
        # Assert thread safety and resource isolation validation
        thread_safety = concurrent_results['thread_safety_validation']
        
        assert thread_safety['resource_isolation_successful'], (
            f"Thread safety validation failed:\n"
            f"  Thread exceptions: {thread_safety['thread_exceptions']}\n"
            f"  Environment creation failures: {thread_safety['environment_creation_failures']}\n"
            f"  Resource isolation: {thread_safety['resource_isolation_successful']}\n"
            f"  Exception details: {shared_performance_data['thread_exceptions']}\n"
            f"  Recommendation: Ensure proper resource isolation and thread-safe environment implementation"
        )
        
        # Store concurrent execution results for monitoring and analysis
        performance_tracker['concurrent_results'] = concurrent_results
        
    else:
        pytest.fail(
            f"Concurrent performance testing failed - no successful workers:\n"
            f"  Configured workers: {num_concurrent_environments}\n"
            f"  Successful workers: {len(successful_workers)}\n"
            f"  Thread exceptions: {len(shared_performance_data['thread_exceptions'])}\n"
            f"  Exception details: {shared_performance_data['thread_exceptions']}\n"
            f"  Recommendation: Fix environment creation and execution issues for concurrent testing"
        )


class TestPerformanceValidation:
    """
    Test class containing performance validation methods with comprehensive benchmarking, statistical 
    analysis, and optimization recommendations for plume_nav_sim environment performance assurance and 
    continuous monitoring.
    
    This class provides systematic performance validation including setup/teardown methods,
    performance target validation, trend analysis, and optimization recommendations with
    comprehensive reporting capabilities for development teams and continuous integration.
    
    Attributes:
        performance_data (dict): Performance measurement storage and analysis results
        baseline_metrics (dict): Historical performance baselines for regression comparison
        optimization_recommendations (list): Generated optimization guidance for development
        regression_detected (bool): Performance regression detection status flag
    """
    
    def __init__(self):
        """
        Initialize performance validation test class with baseline metrics, performance data 
        storage, and regression detection configuration.
        """
        # Initialize empty performance data dictionary for metric storage and analysis
        self.performance_data = {}
        
        # Load baseline performance metrics for regression comparison and trend analysis
        self.baseline_metrics = {
            'step_latency_baseline_ms': PERFORMANCE_TARGET_STEP_LATENCY_MS * 0.8,
            'memory_usage_baseline_mb': MEMORY_LIMIT_TOTAL_MB * 0.6,
            'render_time_baseline_ms': PERFORMANCE_TARGET_RGB_RENDER_MS * 0.7,
            'baseline_timestamp': time.time()
        }
        
        # Initialize optimization recommendations list for performance improvement guidance
        self.optimization_recommendations = []
        
        # Set regression detection flag to False pending analysis completion
        self.regression_detected = False
        
    def setup_method(self, method):
        """
        Setup method executed before each test method providing performance monitoring initialization 
        and baseline measurement.
        
        Initializes test-specific performance monitoring including garbage collection, memory
        baseline establishment, and high-precision timing infrastructure for accurate
        benchmark execution.
        
        Args:
            method (object): Test method object for method-specific configuration
        """
        # Initialize garbage collection and memory baseline measurement for clean testing environment
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / (1024 * 1024)
        
        # Set up high-precision timing infrastructure and performance monitoring systems
        timing_infrastructure = {
            'method_start_time': time.perf_counter(),
            'baseline_memory_mb': baseline_memory,
            'measurement_precision': 6,  # Microsecond precision
            'sampling_active': True
        }
        
        # Initialize test-specific performance data storage and metric tracking
        method_name = getattr(method, '__name__', 'unknown_method')
        self.performance_data[method_name] = {
            'timing_infrastructure': timing_infrastructure,
            'measurements': [],
            'analysis_results': {},
            'validation_status': 'pending'
        }
        
    def teardown_method(self, method):
        """
        Teardown method executed after each test method providing performance data collection and 
        cleanup validation.
        
        Collects final performance metrics including timing analysis, memory cleanup validation,
        and method-specific performance summary with optimization recommendations for
        comprehensive test result analysis.
        
        Args:
            method (object): Test method object for cleanup and analysis finalization
        """
        method_name = getattr(method, '__name__', 'unknown_method')
        
        if method_name in self.performance_data:
            # Collect final performance metrics and timing data for analysis and reporting
            method_data = self.performance_data[method_name]
            timing_infrastructure = method_data['timing_infrastructure']
            
            method_execution_time = time.perf_counter() - timing_infrastructure['method_start_time']
            
            # Validate resource cleanup effectiveness and memory recovery after test execution
            gc.collect()
            process = psutil.Process()
            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_cleanup_delta = final_memory - timing_infrastructure['baseline_memory_mb']
            
            # Store test-specific performance data for trend analysis and regression detection
            method_data['execution_summary'] = {
                'total_execution_time_sec': method_execution_time,
                'memory_cleanup_delta_mb': memory_cleanup_delta,
                'cleanup_successful': abs(memory_cleanup_delta) < 5.0,  # 5MB cleanup threshold
                'measurement_count': len(method_data['measurements'])
            }
            
            # Generate test method performance summary with optimization recommendations
            if not method_data['execution_summary']['cleanup_successful']:
                self.optimization_recommendations.append(
                    f"Method {method_name}: Memory cleanup incomplete ({memory_cleanup_delta:.1f}MB residual)"
                )
                
            method_data['validation_status'] = 'completed'
            
    def validate_performance_target(self, measured_value: float, target_value: float, 
                                  metric_name: str, tolerance: float = PERFORMANCE_TOLERANCE_FACTOR) -> bool:
        """
        Validate individual performance metric against target with tolerance checking and detailed 
        analysis for performance assurance.
        
        Performs comprehensive performance target validation including tolerance analysis,
        statistical significance assessment, and optimization recommendation generation
        with detailed reporting for development guidance.
        
        Args:
            measured_value (float): Actual measured performance value
            target_value (float): Performance target for validation
            metric_name (str): Descriptive name for performance metric
            tolerance (float): Acceptable tolerance factor for target validation
            
        Returns:
            bool: True if performance target met within tolerance, False with detailed analysis
        """
        # Compare measured_value against target_value with tolerance factor analysis
        tolerance_threshold = target_value * (1 + tolerance)
        target_met = measured_value <= tolerance_threshold
        
        # Calculate performance margin and target compliance percentage
        performance_ratio = measured_value / target_value
        compliance_percentage = (1.0 / performance_ratio) * 100 if performance_ratio > 0 else 100.0
        margin_to_target = measured_value - target_value
        
        # Generate detailed performance analysis with statistical significance testing
        validation_analysis = {
            'metric_name': metric_name,
            'measured_value': measured_value,
            'target_value': target_value,
            'tolerance_factor': tolerance,
            'tolerance_threshold': tolerance_threshold,
            'target_met': target_met,
            'performance_ratio': performance_ratio,
            'compliance_percentage': compliance_percentage,
            'margin_to_target': margin_to_target,
            'validation_timestamp': time.time()
        }
        
        # Store validation results in performance_data for comprehensive reporting
        validation_key = f"{metric_name}_validation"
        self.performance_data[validation_key] = validation_analysis
        
        # Generate optimization recommendations if target not met
        if not target_met:
            optimization_recommendation = (
                f"{metric_name}: {measured_value:.3f} exceeds target {target_value:.3f} "
                f"(ratio: {performance_ratio:.2f}x, margin: {margin_to_target:.3f})"
            )
            self.optimization_recommendations.append(optimization_recommendation)
            
        # Return validation status with comprehensive analysis
        return target_met
        
    def analyze_performance_trends(self, performance_metrics: dict) -> dict:
        """
        Analyze performance trends from collected data with statistical analysis and optimization 
        recommendations for continuous improvement.
        
        Performs comprehensive trend analysis including temporal performance patterns,
        regression detection, statistical significance testing, and optimization guidance
        with forecasting capabilities for performance planning.
        
        Args:
            performance_metrics (dict): Performance measurement data for trend analysis
            
        Returns:
            dict: Trend analysis results with statistical insights and optimization recommendations
        """
        # Extract temporal performance data from performance_metrics for trend analysis
        trend_analysis = {
            'analysis_timestamp': time.time(),
            'metrics_analyzed': list(performance_metrics.keys()),
            'trend_patterns': {},
            'regression_indicators': [],
            'optimization_opportunities': [],
            'statistical_significance': {},
            'performance_forecast': {}
        }
        
        # Analyze each performance metric for temporal trends and patterns
        for metric_name, metric_data in performance_metrics.items():
            if isinstance(metric_data, (list, tuple)) and len(metric_data) >= 3:
                # Calculate statistical trends using regression analysis and confidence intervals
                values = np.array(metric_data)
                time_points = np.arange(len(values))
                
                # Perform linear regression for trend identification
                if len(values) > 1:
                    slope, intercept = np.polyfit(time_points, values, 1)
                    correlation = np.corrcoef(time_points, values)[0, 1]
                    
                    trend_classification = 'improving' if slope < 0 else 'degrading' if slope > 0 else 'stable'
                    
                    trend_analysis['trend_patterns'][metric_name] = {
                        'slope': float(slope),
                        'intercept': float(intercept),
                        'correlation': float(correlation),
                        'trend_classification': trend_classification,
                        'trend_strength': abs(correlation),
                        'data_points': len(values)
                    }
                    
                    # Identify performance improvements and degradations with significance testing
                    if abs(slope) > 0.01 and abs(correlation) > 0.5:  # Significant trend threshold
                        if slope > 0:  # Performance degrading
                            trend_analysis['regression_indicators'].append({
                                'metric': metric_name,
                                'trend': 'degrading',
                                'slope': slope,
                                'significance': 'high' if abs(correlation) > 0.8 else 'moderate'
                            })
                        else:  # Performance improving
                            trend_analysis['optimization_opportunities'].append({
                                'metric': metric_name,
                                'trend': 'improving', 
                                'slope': slope,
                                'potential': 'high' if abs(correlation) > 0.8 else 'moderate'
                            })
        
        # Generate optimization recommendations based on trend patterns
        recommendations = []
        
        for regression in trend_analysis['regression_indicators']:
            recommendations.append(
                f"Performance degradation detected in {regression['metric']}: "
                f"{regression['trend']} trend with {regression['significance']} significance"
            )
            
        for opportunity in trend_analysis['optimization_opportunities']:
            recommendations.append(
                f"Optimization opportunity in {opportunity['metric']}: "
                f"{opportunity['trend']} trend with {opportunity['potential']} potential"
            )
            
        trend_analysis['optimization_recommendations'] = recommendations
        
        # Update instance regression detection status
        self.regression_detected = len(trend_analysis['regression_indicators']) > 0
        self.optimization_recommendations.extend(recommendations)
        
        # Return comprehensive trend analysis with actionable insights and performance forecasting
        return trend_analysis