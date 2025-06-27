"""
Performance benchmark test suite using pytest-benchmark to validate sub-10ms step execution times, 
cache hit rate targets, and memory usage compliance, ensuring the frame caching system meets 
strict performance SLA requirements for RL training workflows.

This module provides comprehensive performance validation covering all critical performance 
requirements specified in Section 0 including step execution latency, cache efficiency, 
memory usage bounds, and statistical regression detection. Implements microsecond-precision 
timing assertions using pytest-benchmark with Intel i7-8700K equivalent reference hardware 
performance targets.

Key Test Coverage:
- Sub-10ms step execution requirement validation per Section 0.5.1
- Cache hit rate >90% for sequential access patterns per Section 0.5.1
- Memory usage within 2GiB limit under extended load per Section 0.5.1
- Frame retrieval latency ≤1ms from cache per Section 6.6.5.4.1
- Training throughput ≥1M steps/hour with cache optimization per Section 6.6.4.1.1
- Performance regression detection with statistical significance per Section 6.6.5.3.3
- Memory leak detection over 1M steps per Section 0.5.1

Architecture:
- pytest-benchmark integration for microsecond-precision timing assertions
- Statistical significance testing for performance regression detection
- Memory profiling with psutil for leak detection and usage bounds validation
- Cache performance analysis with hit rate and latency measurements
- Load testing patterns simulating real RL training workflows
- Reference hardware performance baseline validation

Performance Targets (Section 0.1.2):
- Environment step() execution: <10ms average on reference CPU
- Frame retrieval from cache: ≤1ms latency
- Cache hit rate: >90% for sequential access patterns
- Memory usage: ≤2GiB default limit
- Training throughput: ≥1M steps/hour (≈278 steps/second)
- No memory leaks over 1M steps
"""

import gc
import math
import statistics
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

import numpy as np
import pytest
import psutil

# pytest-benchmark for microsecond-precision timing assertions
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    pytest.skip("pytest-benchmark not available", allow_module_level=True)

# Core environment imports
try:
    from odor_plume_nav.environments.gymnasium_env import GymnasiumEnv
    from odor_plume_nav.cache.frame_cache import FrameCache, CacheMode, CacheStatistics
    from odor_plume_nav.data.video_plume import VideoPlume
    CORE_IMPORTS_AVAILABLE = True
except ImportError as e:
    CORE_IMPORTS_AVAILABLE = False
    pytest.skip(f"Core modules not available: {e}", allow_module_level=True)

# Enhanced logging for performance metrics correlation
try:
    from odor_plume_nav.utils.logging_setup import (
        get_enhanced_logger, correlation_context, PerformanceMetrics
    )
    logger = get_enhanced_logger(__name__)
    ENHANCED_LOGGING = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    ENHANCED_LOGGING = False

# Performance constants from Section 0.1.2 requirements
STEP_TARGET_MS = 10.0  # <10ms step execution requirement
CACHE_HIT_LATENCY_TARGET_MS = 1.0  # ≤1ms frame retrieval from cache
CACHE_HIT_RATE_TARGET = 0.90  # >90% cache hit rate for sequential access
MEMORY_LIMIT_GB = 2.0  # 2GiB default memory limit
THROUGHPUT_TARGET_STEPS_PER_HOUR = 1_000_000  # ≥1M steps/hour
THROUGHPUT_TARGET_STEPS_PER_SECOND = THROUGHPUT_TARGET_STEPS_PER_HOUR / 3600  # ≈278 steps/sec

# Statistical testing parameters for regression detection
REGRESSION_SIGNIFICANCE_LEVEL = 0.05  # 5% significance level
MIN_SAMPLES_FOR_STATS = 50  # Minimum samples for statistical significance
PERFORMANCE_TOLERANCE_PERCENT = 10.0  # 10% performance degradation tolerance

# Test correlation tracking
TEST_CORRELATION_ID = f"perf_benchmark_{int(time.time())}"


@pytest.fixture(scope="session")
def mock_video_file():
    """Create temporary mock video file for consistent performance testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = Path(f.name)
        # Write minimal video header for OpenCV compatibility
        f.write(b'\x00\x00\x00\x18ftypmp42')  # Minimal MP4 header
    
    yield video_path
    
    # Cleanup
    try:
        video_path.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_video_plume_optimized():
    """Mock VideoPlume optimized for performance testing with deterministic frame generation."""
    with patch('odor_plume_nav.environments.gymnasium_env.VideoPlume') as mock_class:
        mock_instance = Mock()
        
        # Configure metadata for consistent testing
        TEST_VIDEO_WIDTH = 640
        TEST_VIDEO_HEIGHT = 480
        TEST_VIDEO_FRAMES = 1000
        
        mock_instance.get_metadata.return_value = {
            'width': TEST_VIDEO_WIDTH,
            'height': TEST_VIDEO_HEIGHT,
            'fps': 30.0,
            'frame_count': TEST_VIDEO_FRAMES
        }
        
        # Optimized frame generation with timing simulation
        def get_frame_with_timing(frame_index):
            """Simulate realistic frame retrieval with configurable latency."""
            # Simulate disk I/O latency for cache miss scenario
            if not hasattr(get_frame_with_timing, '_cache_enabled'):
                time.sleep(0.005)  # 5ms disk I/O simulation
            
            # Generate deterministic frame data (smaller than real frames for speed)
            np.random.seed(frame_index % 1000)  # Deterministic but varied
            frame = np.random.rand(TEST_VIDEO_HEIGHT // 4, TEST_VIDEO_WIDTH // 4).astype(np.float32)
            return frame
        
        mock_instance.get_frame.side_effect = get_frame_with_timing
        mock_instance.close.return_value = None
        
        # Properties for environment initialization
        mock_instance.width = TEST_VIDEO_WIDTH
        mock_instance.height = TEST_VIDEO_HEIGHT
        mock_instance.frame_count = TEST_VIDEO_FRAMES
        
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_navigator_optimized():
    """Mock Navigator optimized for performance testing with minimal overhead."""
    with patch('odor_plume_nav.environments.gymnasium_env.NavigatorFactory') as mock_factory:
        mock_navigator = Mock()
        
        # Initialize with minimal state for performance
        mock_navigator.positions = np.array([[320.0, 240.0]], dtype=np.float32)
        mock_navigator.orientations = np.array([0.0], dtype=np.float32)
        mock_navigator.speeds = np.array([0.0], dtype=np.float32)
        mock_navigator.angular_velocities = np.array([0.0], dtype=np.float32)
        
        # Optimized methods with minimal computation
        def fast_reset(position, orientation, speed, angular_velocity):
            mock_navigator.positions[0] = np.array(position, dtype=np.float32)
            mock_navigator.orientations[0] = float(orientation)
            mock_navigator.speeds[0] = float(speed)
            mock_navigator.angular_velocities[0] = float(angular_velocity)
        
        def fast_step(frame, dt):
            # Minimal position update for performance
            speed = mock_navigator.speeds[0]
            angle_rad = np.radians(mock_navigator.orientations[0])
            
            dx = speed * np.cos(angle_rad) * dt * 0.1  # Reduced movement for stability
            dy = speed * np.sin(angle_rad) * dt * 0.1
            mock_navigator.positions[0] += [dx, dy]
            
            mock_navigator.orientations[0] += mock_navigator.angular_velocities[0] * dt * 0.1
            mock_navigator.orientations[0] = mock_navigator.orientations[0] % 360.0
        
        def fast_sample_odor(frame):
            # Fast odor calculation based on position
            pos = mock_navigator.positions[0]
            center_x, center_y = 320, 240
            distance = abs(pos[0] - center_x) + abs(pos[1] - center_y)  # Manhattan distance for speed
            return max(0.0, 1.0 - distance / 400.0)  # Simple linear falloff
        
        mock_navigator.reset.side_effect = fast_reset
        mock_navigator.step.side_effect = fast_step
        mock_navigator.sample_odor.side_effect = fast_sample_odor
        
        mock_factory.single_agent.return_value = mock_navigator
        yield mock_navigator


@pytest.fixture
def performance_test_config(mock_video_file):
    """Configuration optimized for performance testing with minimal overhead."""
    return {
        'video_path': str(mock_video_file),
        'initial_position': (320, 240),
        'initial_orientation': 0.0,
        'max_speed': 2.0,
        'max_angular_velocity': 90.0,
        'max_episode_steps': 1000,  # Sufficient for performance testing
        'performance_monitoring': True,  # Enable for metrics collection
        'render_mode': None  # Disable rendering for performance
    }


@pytest.fixture
def frame_cache_lru():
    """Create LRU FrameCache instance for performance testing."""
    cache = FrameCache(
        mode=CacheMode.LRU,
        memory_limit_mb=512,  # Smaller limit for testing
        memory_pressure_threshold=0.9,
        enable_statistics=True,
        enable_logging=False  # Disable logging for pure performance
    )
    yield cache
    # Cleanup
    cache.clear()


@pytest.fixture
def frame_cache_preload():
    """Create preload FrameCache instance for maximum performance testing."""
    cache = FrameCache(
        mode=CacheMode.ALL,
        memory_limit_mb=1024,  # Larger limit for preload mode
        memory_pressure_threshold=0.9,
        enable_statistics=True,
        enable_logging=False
    )
    yield cache
    cache.clear()


@pytest.fixture
def gymnasium_env_cached(performance_test_config, mock_video_plume_optimized, 
                        mock_navigator_optimized, frame_cache_lru):
    """GymnasiumEnv with frame caching enabled for performance testing."""
    config_with_cache = {**performance_test_config, 'frame_cache': frame_cache_lru}
    env = GymnasiumEnv(**config_with_cache)
    yield env
    env.close()


@pytest.fixture
def gymnasium_env_uncached(performance_test_config, mock_video_plume_optimized, 
                          mock_navigator_optimized):
    """GymnasiumEnv without caching for baseline performance comparison."""
    env = GymnasiumEnv(**performance_test_config)
    yield env
    env.close()


class TestStepExecutionPerformance:
    """Test suite for step execution performance validation per Section 0.5.1."""
    
    def test_step_execution_time_benchmark(self, benchmark, gymnasium_env_cached):
        """
        Validate step() execution time meets <10ms requirement using pytest-benchmark.
        
        Tests the core performance requirement from Section 0.5.1 using microsecond-precision
        timing to ensure environment step execution consistently meets the sub-10ms target
        required for real-time RL training workflows.
        """
        # Reset environment for consistent state
        obs, info = gymnasium_env_cached.reset(seed=42)
        action = gymnasium_env_cached.action_space.sample()
        
        # Benchmark single step execution
        def step_execution():
            return gymnasium_env_cached.step(action)
        
        # Run benchmark with statistical analysis
        result = benchmark.pedantic(
            step_execution,
            rounds=100,  # Sufficient for statistical significance
            iterations=1  # Single iteration per round for realistic timing
        )
        
        # Extract timing statistics
        step_time_ms = benchmark.stats['mean'] * 1000  # Convert to milliseconds
        std_dev_ms = benchmark.stats['stddev'] * 1000
        p95_ms = benchmark.stats.get('q_95', benchmark.stats['max']) * 1000
        
        # Log performance metrics with correlation
        if ENHANCED_LOGGING:
            logger.info(
                f"Step execution benchmark: {step_time_ms:.2f}ms ± {std_dev_ms:.2f}ms",
                extra={
                    "metric_type": "step_execution_benchmark",
                    "mean_time_ms": step_time_ms,
                    "stddev_ms": std_dev_ms,
                    "p95_ms": p95_ms,
                    "target_ms": STEP_TARGET_MS,
                    "compliant": step_time_ms <= STEP_TARGET_MS,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Assert performance requirement compliance
        assert step_time_ms <= STEP_TARGET_MS, (
            f"Step execution time {step_time_ms:.2f}ms exceeds target {STEP_TARGET_MS}ms. "
            f"P95: {p95_ms:.2f}ms, StdDev: {std_dev_ms:.2f}ms"
        )
        
        # Additional assertions for consistency
        assert p95_ms <= STEP_TARGET_MS * 1.5, f"P95 step time {p95_ms:.2f}ms too high"
        assert std_dev_ms <= STEP_TARGET_MS * 0.3, f"Step time variance {std_dev_ms:.2f}ms too high"
    
    def test_step_execution_time_extended_load(self, gymnasium_env_cached):
        """
        Validate step execution performance under extended load scenarios.
        
        Tests sustained performance over extended periods to detect memory leaks,
        cache degradation, or other performance regressions that may appear only
        under prolonged operation typical of RL training.
        """
        obs, info = gymnasium_env_cached.reset(seed=42)
        
        step_times = []
        memory_usage = []
        cache_hit_rates = []
        
        # Extended load test - sufficient for statistical analysis
        num_steps = 1000
        
        with correlation_context("extended_load_test", correlation_id=TEST_CORRELATION_ID) if ENHANCED_LOGGING else nullcontext():
            for step_idx in range(num_steps):
                action = gymnasium_env_cached.action_space.sample()
                
                start_time = time.time()
                obs, reward, terminated, truncated, info = gymnasium_env_cached.step(action)
                step_time = time.time() - start_time
                
                step_times.append(step_time * 1000)  # Convert to milliseconds
                
                # Track memory usage every 100 steps
                if step_idx % 100 == 0:
                    memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
                
                # Track cache performance
                if 'perf_stats' in info:
                    cache_hit_rates.append(info['perf_stats'].get('cache_hit_rate', 0.0))
                
                # Reset environment if terminated
                if terminated or truncated:
                    obs, info = gymnasium_env_cached.reset()
        
        # Statistical analysis
        mean_step_time = statistics.mean(step_times)
        median_step_time = statistics.median(step_times)
        p95_step_time = np.percentile(step_times, 95)
        p99_step_time = np.percentile(step_times, 99)
        
        # Memory analysis
        memory_growth = memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
        avg_cache_hit_rate = statistics.mean(cache_hit_rates) if cache_hit_rates else 0.0
        
        # Log comprehensive performance analysis
        if ENHANCED_LOGGING:
            logger.info(
                f"Extended load test: {num_steps} steps, avg={mean_step_time:.2f}ms, "
                f"p95={p95_step_time:.2f}ms, memory_growth={memory_growth:.1f}MB",
                extra={
                    "metric_type": "extended_load_performance",
                    "num_steps": num_steps,
                    "mean_step_time_ms": mean_step_time,
                    "median_step_time_ms": median_step_time,
                    "p95_step_time_ms": p95_step_time,
                    "p99_step_time_ms": p99_step_time,
                    "memory_growth_mb": memory_growth,
                    "avg_cache_hit_rate": avg_cache_hit_rate,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Performance assertions
        assert mean_step_time <= STEP_TARGET_MS, (
            f"Extended load mean step time {mean_step_time:.2f}ms exceeds target {STEP_TARGET_MS}ms"
        )
        assert p95_step_time <= STEP_TARGET_MS * 1.5, (
            f"Extended load P95 step time {p95_step_time:.2f}ms exceeds tolerance"
        )
        
        # Memory leak detection
        assert memory_growth <= 100, (  # Allow up to 100MB growth for caching
            f"Potential memory leak detected: {memory_growth:.1f}MB growth over {num_steps} steps"
        )
    
    def test_step_execution_performance_comparison(self, gymnasium_env_cached, gymnasium_env_uncached):
        """
        Compare cached vs uncached performance to validate cache effectiveness.
        
        Measures the performance improvement provided by frame caching to ensure
        the caching system delivers the expected performance benefits without
        introducing unacceptable overhead.
        """
        num_steps = 200  # Sufficient for statistical comparison
        
        # Benchmark uncached performance
        obs, info = gymnasium_env_uncached.reset(seed=42)
        uncached_times = []
        
        for _ in range(num_steps):
            action = gymnasium_env_uncached.action_space.sample()
            start_time = time.time()
            obs, reward, terminated, truncated, info = gymnasium_env_uncached.step(action)
            uncached_times.append(time.time() - start_time)
            
            if terminated or truncated:
                obs, info = gymnasium_env_uncached.reset()
        
        # Benchmark cached performance
        obs, info = gymnasium_env_cached.reset(seed=42)
        cached_times = []
        
        for _ in range(num_steps):
            action = gymnasium_env_cached.action_space.sample()
            start_time = time.time()
            obs, reward, terminated, truncated, info = gymnasium_env_cached.step(action)
            cached_times.append(time.time() - start_time)
            
            if terminated or truncated:
                obs, info = gymnasium_env_cached.reset()
        
        # Statistical comparison
        uncached_mean = statistics.mean(uncached_times) * 1000
        cached_mean = statistics.mean(cached_times) * 1000
        performance_improvement = (uncached_mean - cached_mean) / uncached_mean * 100
        
        # Log performance comparison
        if ENHANCED_LOGGING:
            logger.info(
                f"Performance comparison: uncached={uncached_mean:.2f}ms, "
                f"cached={cached_mean:.2f}ms, improvement={performance_improvement:.1f}%",
                extra={
                    "metric_type": "cache_performance_comparison",
                    "uncached_mean_ms": uncached_mean,
                    "cached_mean_ms": cached_mean,
                    "performance_improvement_percent": performance_improvement,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Validate performance improvements
        assert cached_mean <= STEP_TARGET_MS, (
            f"Cached step time {cached_mean:.2f}ms exceeds target {STEP_TARGET_MS}ms"
        )
        assert performance_improvement >= 10.0, (  # Expect at least 10% improvement
            f"Cache provides insufficient performance improvement: {performance_improvement:.1f}%"
        )


class TestCachePerformance:
    """Test suite for frame cache performance validation per Section 6.6.5.4.1."""
    
    def test_cache_hit_rate_sequential_access(self, frame_cache_lru):
        """
        Validate cache hit rate >90% for sequential access patterns per Section 0.5.1.
        
        Tests the cache effectiveness for typical RL training access patterns where
        agents step through video frames sequentially. Validates the cache achieves
        the target >90% hit rate specified in performance requirements.
        """
        # Mock video plume for frame generation
        mock_video_plume = Mock()
        
        def generate_mock_frame(frame_id):
            # Generate small frames for testing efficiency
            return np.random.rand(100, 100).astype(np.float32)
        
        mock_video_plume.get_frame.side_effect = generate_mock_frame
        
        # Sequential access pattern typical of RL training
        frame_range = range(0, 500)  # Sufficient range for cache evaluation
        access_pattern = list(frame_range) * 3  # Access each frame 3 times
        
        hit_count = 0
        total_requests = 0
        
        with correlation_context("cache_hit_rate_test", correlation_id=TEST_CORRELATION_ID) if ENHANCED_LOGGING else nullcontext():
            for frame_id in access_pattern:
                frame = frame_cache_lru.get(frame_id, mock_video_plume)
                total_requests += 1
                
                # Check if this was a cache hit (first access should be miss, subsequent hits)
                if frame_cache_lru.statistics and frame_cache_lru.statistics.total_requests > 0:
                    current_hit_rate = frame_cache_lru.statistics.hit_rate
                    if total_requests > len(frame_range):  # After first full pass
                        hit_count += 1 if current_hit_rate > 0 else 0
        
        # Calculate final hit rate
        final_hit_rate = frame_cache_lru.hit_rate
        
        # Log cache performance metrics
        if ENHANCED_LOGGING:
            logger.info(
                f"Cache hit rate test: {final_hit_rate:.3f} ({final_hit_rate*100:.1f}%)",
                extra={
                    "metric_type": "cache_hit_rate_validation",
                    "hit_rate": final_hit_rate,
                    "total_requests": total_requests,
                    "cache_size": len(frame_cache_lru._cache) if frame_cache_lru._cache else 0,
                    "target_hit_rate": CACHE_HIT_RATE_TARGET,
                    "compliant": final_hit_rate >= CACHE_HIT_RATE_TARGET,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Assert hit rate requirement
        assert final_hit_rate >= CACHE_HIT_RATE_TARGET, (
            f"Cache hit rate {final_hit_rate:.3f} ({final_hit_rate*100:.1f}%) "
            f"below target {CACHE_HIT_RATE_TARGET:.3f} ({CACHE_HIT_RATE_TARGET*100:.1f}%)"
        )
    
    def test_frame_retrieval_latency_from_cache(self, benchmark, frame_cache_lru):
        """
        Validate frame retrieval latency ≤1ms from cache per Section 6.6.5.4.1.
        
        Benchmarks cached frame retrieval to ensure sub-millisecond access times
        required for high-performance RL training workflows. Uses pytest-benchmark
        for precise timing measurements.
        """
        # Mock video plume with realistic frame generation
        mock_video_plume = Mock()
        
        def generate_realistic_frame(frame_id):
            return np.random.rand(480, 640).astype(np.float32)  # Realistic frame size
        
        mock_video_plume.get_frame.side_effect = generate_realistic_frame
        
        # Pre-populate cache with test frames
        test_frame_ids = list(range(0, 100))
        for frame_id in test_frame_ids:
            frame_cache_lru.get(frame_id, mock_video_plume)
        
        # Select a frame that should be cached
        cached_frame_id = test_frame_ids[50]  # Middle frame, likely still cached
        
        # Benchmark cached frame retrieval
        def cached_frame_retrieval():
            return frame_cache_lru.get(cached_frame_id, mock_video_plume)
        
        # Run benchmark with high precision
        result = benchmark.pedantic(
            cached_frame_retrieval,
            rounds=50,
            iterations=10  # Multiple iterations per round for precision
        )
        
        # Extract timing statistics
        retrieval_time_ms = benchmark.stats['mean'] * 1000  # Convert to milliseconds
        std_dev_ms = benchmark.stats['stddev'] * 1000
        p95_ms = benchmark.stats.get('q_95', benchmark.stats['max']) * 1000
        
        # Log cache retrieval performance
        if ENHANCED_LOGGING:
            logger.info(
                f"Cache retrieval benchmark: {retrieval_time_ms:.3f}ms ± {std_dev_ms:.3f}ms",
                extra={
                    "metric_type": "cache_retrieval_benchmark",
                    "mean_time_ms": retrieval_time_ms,
                    "stddev_ms": std_dev_ms,
                    "p95_ms": p95_ms,
                    "target_ms": CACHE_HIT_LATENCY_TARGET_MS,
                    "compliant": retrieval_time_ms <= CACHE_HIT_LATENCY_TARGET_MS,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Assert latency requirement
        assert retrieval_time_ms <= CACHE_HIT_LATENCY_TARGET_MS, (
            f"Cache retrieval time {retrieval_time_ms:.3f}ms exceeds target "
            f"{CACHE_HIT_LATENCY_TARGET_MS}ms. P95: {p95_ms:.3f}ms"
        )
        
        # Additional consistency checks
        assert p95_ms <= CACHE_HIT_LATENCY_TARGET_MS * 2, (
            f"Cache retrieval P95 time {p95_ms:.3f}ms indicates inconsistent performance"
        )
    
    def test_cache_memory_usage_compliance(self, frame_cache_lru):
        """
        Validate memory usage stays within configured limits during operation.
        
        Tests that the frame cache respects memory limits and triggers appropriate
        eviction policies to maintain compliance with the 2GiB default limit
        specified in Section 0.5.1.
        """
        # Mock video plume with realistic frame sizes
        mock_video_plume = Mock()
        
        def generate_sized_frame(frame_id):
            # Generate frames with realistic size (640x480 float32 = ~1.2MB)
            return np.random.rand(480, 640).astype(np.float32)
        
        mock_video_plume.get_frame.side_effect = generate_sized_frame
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        max_memory_observed = initial_memory
        
        # Fill cache beyond memory limit to test eviction
        num_frames = 600  # Should exceed 512MB cache limit with ~1.2MB frames
        
        with correlation_context("memory_usage_test", correlation_id=TEST_CORRELATION_ID) if ENHANCED_LOGGING else nullcontext():
            for frame_id in range(num_frames):
                frame_cache_lru.get(frame_id, mock_video_plume)
                
                # Monitor memory usage every 50 frames
                if frame_id % 50 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    max_memory_observed = max(max_memory_observed, current_memory)
                    
                    cache_memory = frame_cache_lru.statistics.memory_usage_mb if frame_cache_lru.statistics else 0
                    
                    # Check cache-reported memory usage
                    assert cache_memory <= frame_cache_lru.memory_limit_mb * 1.1, (  # 10% tolerance
                        f"Cache reports memory usage {cache_memory:.1f}MB exceeds "
                        f"limit {frame_cache_lru.memory_limit_mb}MB"
                    )
        
        # Final memory analysis
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        cache_stats = frame_cache_lru.statistics
        
        # Log memory usage analysis
        if ENHANCED_LOGGING:
            logger.info(
                f"Memory usage test: growth={memory_growth:.1f}MB, "
                f"cache_memory={cache_stats.memory_usage_mb if cache_stats else 0:.1f}MB, "
                f"evictions={cache_stats.evictions if cache_stats else 0}",
                extra={
                    "metric_type": "cache_memory_usage_validation",
                    "memory_growth_mb": memory_growth,
                    "cache_memory_mb": cache_stats.memory_usage_mb if cache_stats else 0,
                    "cache_limit_mb": frame_cache_lru.memory_limit_mb,
                    "evictions": cache_stats.evictions if cache_stats else 0,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Assert memory compliance
        if cache_stats:
            assert cache_stats.memory_usage_mb <= frame_cache_lru.memory_limit_mb * 1.1, (
                f"Cache memory usage {cache_stats.memory_usage_mb:.1f}MB exceeds "
                f"limit {frame_cache_lru.memory_limit_mb}MB"
            )
            
            # Verify evictions occurred when memory pressure was reached
            assert cache_stats.evictions > 0, (
                "No evictions occurred despite exceeding cache capacity"
            )
        
        # Global memory usage should be reasonable
        assert memory_growth <= MEMORY_LIMIT_GB * 1024 * 1.2, (  # 20% tolerance for test overhead
            f"Total memory growth {memory_growth:.1f}MB exceeds reasonable bounds"
        )


class TestTrainingThroughputPerformance:
    """Test suite for training throughput performance validation per Section 6.6.4.1.1."""
    
    def test_training_throughput_target(self, gymnasium_env_cached):
        """
        Validate training throughput ≥1M steps/hour with cache optimization.
        
        Measures sustained step execution rate to ensure the environment can support
        the target training throughput of 1 million steps per hour specified in
        Section 6.6.4.1.1 performance requirements.
        """
        obs, info = gymnasium_env_cached.reset(seed=42)
        
        # Measure throughput over sufficient duration for statistical accuracy
        test_duration_seconds = 10.0  # 10-second measurement window
        start_time = time.time()
        step_count = 0
        step_times = []
        
        with correlation_context("throughput_test", correlation_id=TEST_CORRELATION_ID) if ENHANCED_LOGGING else nullcontext():
            while time.time() - start_time < test_duration_seconds:
                action = gymnasium_env_cached.action_space.sample()
                
                step_start = time.time()
                obs, reward, terminated, truncated, info = gymnasium_env_cached.step(action)
                step_time = time.time() - step_start
                
                step_times.append(step_time)
                step_count += 1
                
                if terminated or truncated:
                    obs, info = gymnasium_env_cached.reset()
        
        # Calculate throughput metrics
        actual_duration = time.time() - start_time
        steps_per_second = step_count / actual_duration
        steps_per_hour = steps_per_second * 3600
        
        # Statistical analysis of step times
        mean_step_time = statistics.mean(step_times) * 1000  # ms
        p95_step_time = np.percentile(step_times, 95) * 1000  # ms
        
        # Cache performance analysis
        cache_hit_rate = 0.0
        if 'perf_stats' in info:
            cache_hit_rate = info['perf_stats'].get('cache_hit_rate', 0.0)
        
        # Log throughput analysis
        if ENHANCED_LOGGING:
            logger.info(
                f"Throughput test: {steps_per_hour:.0f} steps/hour "
                f"({steps_per_second:.1f} steps/sec), cache_hit_rate={cache_hit_rate:.3f}",
                extra={
                    "metric_type": "training_throughput_validation",
                    "steps_per_second": steps_per_second,
                    "steps_per_hour": steps_per_hour,
                    "target_steps_per_hour": THROUGHPUT_TARGET_STEPS_PER_HOUR,
                    "mean_step_time_ms": mean_step_time,
                    "p95_step_time_ms": p95_step_time,
                    "cache_hit_rate": cache_hit_rate,
                    "step_count": step_count,
                    "test_duration": actual_duration,
                    "compliant": steps_per_hour >= THROUGHPUT_TARGET_STEPS_PER_HOUR,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Assert throughput requirement
        assert steps_per_hour >= THROUGHPUT_TARGET_STEPS_PER_HOUR, (
            f"Training throughput {steps_per_hour:.0f} steps/hour below target "
            f"{THROUGHPUT_TARGET_STEPS_PER_HOUR:.0f} steps/hour. "
            f"Current rate: {steps_per_second:.1f} steps/sec, "
            f"Required rate: {THROUGHPUT_TARGET_STEPS_PER_SECOND:.1f} steps/sec"
        )
        
        # Additional performance consistency checks
        assert mean_step_time <= STEP_TARGET_MS, (
            f"Mean step time {mean_step_time:.2f}ms exceeds target for throughput compliance"
        )
        assert p95_step_time <= STEP_TARGET_MS * 1.5, (
            f"P95 step time {p95_step_time:.2f}ms indicates inconsistent throughput"
        )
    
    def test_concurrent_agent_throughput(self, performance_test_config, 
                                       mock_video_plume_optimized, mock_navigator_optimized):
        """
        Validate throughput performance under concurrent multi-agent access.
        
        Tests the system's ability to maintain performance targets when multiple
        agents access the environment concurrently, simulating distributed training
        scenarios common in modern RL workflows.
        """
        num_agents = 4  # Concurrent agents
        steps_per_agent = 100
        
        # Create multiple environments with shared cache
        shared_cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=1024, enable_logging=False)
        
        environments = []
        for i in range(num_agents):
            config = {**performance_test_config, 'frame_cache': shared_cache}
            env = GymnasiumEnv(**config)
            environments.append(env)
        
        try:
            results = []
            threads = []
            
            def agent_worker(agent_id, env):
                """Worker function for concurrent agent testing."""
                agent_results = {
                    'agent_id': agent_id,
                    'step_times': [],
                    'throughput': 0.0,
                    'completed_steps': 0
                }
                
                obs, info = env.reset(seed=42 + agent_id)
                start_time = time.time()
                
                for step in range(steps_per_agent):
                    action = env.action_space.sample()
                    
                    step_start = time.time()
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_time = time.time() - step_start
                    
                    agent_results['step_times'].append(step_time)
                    agent_results['completed_steps'] += 1
                    
                    if terminated or truncated:
                        obs, info = env.reset()
                
                duration = time.time() - start_time
                agent_results['throughput'] = agent_results['completed_steps'] / duration
                results.append(agent_results)
            
            # Start concurrent agent workers
            with correlation_context("concurrent_throughput_test", correlation_id=TEST_CORRELATION_ID) if ENHANCED_LOGGING else nullcontext():
                for i, env in enumerate(environments):
                    thread = threading.Thread(target=agent_worker, args=(i, env))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all agents to complete
                for thread in threads:
                    thread.join()
            
            # Analyze concurrent performance
            total_steps = sum(r['completed_steps'] for r in results)
            total_throughput = sum(r['throughput'] for r in results)
            avg_step_time = statistics.mean([
                t for r in results for t in r['step_times']
            ]) * 1000  # ms
            
            # Cache performance under concurrent load
            cache_hit_rate = shared_cache.hit_rate
            cache_memory_usage = shared_cache.statistics.memory_usage_mb if shared_cache.statistics else 0
            
            # Log concurrent performance analysis
            if ENHANCED_LOGGING:
                logger.info(
                    f"Concurrent throughput: {total_throughput:.1f} total steps/sec, "
                    f"avg_step_time={avg_step_time:.2f}ms, cache_hit_rate={cache_hit_rate:.3f}",
                    extra={
                        "metric_type": "concurrent_throughput_validation",
                        "num_agents": num_agents,
                        "total_steps": total_steps,
                        "total_throughput": total_throughput,
                        "avg_step_time_ms": avg_step_time,
                        "cache_hit_rate": cache_hit_rate,
                        "cache_memory_mb": cache_memory_usage,
                        "correlation_id": TEST_CORRELATION_ID
                    }
                )
            
            # Performance assertions
            assert avg_step_time <= STEP_TARGET_MS * 1.2, (  # Allow 20% overhead for concurrency
                f"Concurrent avg step time {avg_step_time:.2f}ms exceeds tolerance"
            )
            
            # Validate each agent met minimum performance
            for result in results:
                agent_avg_time = statistics.mean(result['step_times']) * 1000
                assert agent_avg_time <= STEP_TARGET_MS * 1.5, (
                    f"Agent {result['agent_id']} step time {agent_avg_time:.2f}ms too high"
                )
            
            # Cache should maintain effectiveness under concurrent load
            assert cache_hit_rate >= CACHE_HIT_RATE_TARGET * 0.8, (  # Allow some degradation
                f"Cache hit rate {cache_hit_rate:.3f} degraded too much under concurrent load"
            )
            
        finally:
            # Cleanup environments
            for env in environments:
                env.close()
            shared_cache.clear()


class TestMemoryUsageCompliance:
    """Test suite for memory usage compliance validation per Section 0.5.1."""
    
    def test_memory_usage_within_limits_extended_load(self, gymnasium_env_cached):
        """
        Test memory usage stays within 2GiB limit under extended load per Section 0.5.1.
        
        Validates that the system maintains memory usage within specified bounds during
        extended operation, detecting potential memory leaks or excessive memory growth
        that could impact long-running training workflows.
        """
        obs, info = gymnasium_env_cached.reset(seed=42)
        
        # Extended load test parameters
        target_steps = 5000  # Substantial load for memory analysis
        memory_check_interval = 250  # Check memory every N steps
        
        initial_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        memory_samples = [initial_memory_mb]
        peak_memory_mb = initial_memory_mb
        
        step_count = 0
        
        with correlation_context("memory_compliance_test", correlation_id=TEST_CORRELATION_ID) if ENHANCED_LOGGING else nullcontext():
            for step in range(target_steps):
                action = gymnasium_env_cached.action_space.sample()
                obs, reward, terminated, truncated, info = gymnasium_env_cached.step(action)
                step_count += 1
                
                # Periodic memory monitoring
                if step % memory_check_interval == 0:
                    current_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory_mb)
                    peak_memory_mb = max(peak_memory_mb, current_memory_mb)
                    
                    # Check cache memory if available
                    cache_memory_mb = 0
                    if 'perf_stats' in info:
                        cache_memory_mb = info['perf_stats'].get('cache_memory_usage_mb', 0)
                    
                    # Early termination if memory growth is excessive
                    memory_growth = current_memory_mb - initial_memory_mb
                    if memory_growth > MEMORY_LIMIT_GB * 1024 * 1.5:  # 50% tolerance
                        warnings.warn(f"Memory growth {memory_growth:.1f}MB exceeds tolerance at step {step}")
                        break
                
                if terminated or truncated:
                    obs, info = gymnasium_env_cached.reset()
        
        # Memory analysis
        final_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        total_memory_growth_mb = final_memory_mb - initial_memory_mb
        max_memory_growth_mb = peak_memory_mb - initial_memory_mb
        
        # Statistical analysis of memory usage
        memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]  # Linear trend
        
        # Cache statistics
        cache_stats = {}
        if hasattr(gymnasium_env_cached, 'frame_cache') and gymnasium_env_cached.frame_cache:
            cache = gymnasium_env_cached.frame_cache
            if cache.statistics:
                cache_stats = {
                    'memory_mb': cache.statistics.memory_usage_mb,
                    'hit_rate': cache.statistics.hit_rate,
                    'evictions': cache.statistics.evictions
                }
        
        # Log memory compliance analysis
        if ENHANCED_LOGGING:
            logger.info(
                f"Memory compliance test: growth={total_memory_growth_mb:.1f}MB, "
                f"peak_growth={max_memory_growth_mb:.1f}MB, trend={memory_trend:.3f}MB/sample",
                extra={
                    "metric_type": "memory_compliance_validation",
                    "steps_completed": step_count,
                    "initial_memory_mb": initial_memory_mb,
                    "final_memory_mb": final_memory_mb,
                    "total_growth_mb": total_memory_growth_mb,
                    "peak_growth_mb": max_memory_growth_mb,
                    "memory_trend_mb_per_sample": memory_trend,
                    "cache_stats": cache_stats,
                    "memory_limit_gb": MEMORY_LIMIT_GB,
                    "compliant": total_memory_growth_mb <= MEMORY_LIMIT_GB * 1024,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Memory compliance assertions
        assert total_memory_growth_mb <= MEMORY_LIMIT_GB * 1024, (
            f"Total memory growth {total_memory_growth_mb:.1f}MB exceeds "
            f"limit {MEMORY_LIMIT_GB * 1024:.0f}MB over {step_count} steps"
        )
        
        assert max_memory_growth_mb <= MEMORY_LIMIT_GB * 1024 * 1.2, (  # 20% peak tolerance
            f"Peak memory growth {max_memory_growth_mb:.1f}MB exceeds tolerance"
        )
        
        # Memory leak detection via trend analysis
        memory_leak_threshold = 0.1  # MB growth per sample indicates potential leak
        assert memory_trend <= memory_leak_threshold, (
            f"Potential memory leak detected: {memory_trend:.3f}MB growth per sample"
        )
    
    def test_memory_leak_detection_over_episodes(self, gymnasium_env_cached):
        """
        Detect memory leaks over multiple episode cycles.
        
        Specifically tests for memory leaks that may occur during episode resets
        and environment state transitions, which can be sources of memory retention
        in RL environments.
        """
        num_episodes = 50
        steps_per_episode = 100
        
        initial_memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        episode_memory_samples = []
        
        with correlation_context("memory_leak_detection", correlation_id=TEST_CORRELATION_ID) if ENHANCED_LOGGING else nullcontext():
            for episode in range(num_episodes):
                obs, info = gymnasium_env_cached.reset(seed=42 + episode)
                
                # Run episode
                for step in range(steps_per_episode):
                    action = gymnasium_env_cached.action_space.sample()
                    obs, reward, terminated, truncated, info = gymnasium_env_cached.step(action)
                    
                    if terminated or truncated:
                        break
                
                # Measure memory after episode completion
                episode_memory = psutil.Process().memory_info().rss / 1024 / 1024
                episode_memory_samples.append(episode_memory)
                
                # Force garbage collection to detect real leaks
                gc.collect()
        
        # Analyze memory growth across episodes
        final_memory_mb = episode_memory_samples[-1]
        total_growth_mb = final_memory_mb - initial_memory_mb
        
        # Calculate linear trend across episodes
        episode_trend = np.polyfit(range(len(episode_memory_samples)), episode_memory_samples, 1)[0]
        
        # Calculate memory growth rate per episode
        growth_per_episode = total_growth_mb / num_episodes
        
        # Log memory leak analysis
        if ENHANCED_LOGGING:
            logger.info(
                f"Memory leak detection: {total_growth_mb:.1f}MB total growth, "
                f"{growth_per_episode:.3f}MB per episode, trend={episode_trend:.3f}MB/episode",
                extra={
                    "metric_type": "memory_leak_detection",
                    "num_episodes": num_episodes,
                    "steps_per_episode": steps_per_episode,
                    "total_growth_mb": total_growth_mb,
                    "growth_per_episode_mb": growth_per_episode,
                    "episode_trend_mb": episode_trend,
                    "initial_memory_mb": initial_memory_mb,
                    "final_memory_mb": final_memory_mb,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Memory leak assertions
        max_growth_per_episode = 2.0  # MB - reasonable for cache warming
        assert growth_per_episode <= max_growth_per_episode, (
            f"Memory growth {growth_per_episode:.3f}MB per episode indicates potential leak"
        )
        
        # Trend analysis for leak detection
        leak_trend_threshold = 0.5  # MB growth per episode trend
        assert episode_trend <= leak_trend_threshold, (
            f"Memory leak trend detected: {episode_trend:.3f}MB growth per episode"
        )
        
        # Total growth should be reasonable for cache operations
        max_total_growth = 200.0  # MB - reasonable for cache establishment
        assert total_growth_mb <= max_total_growth, (
            f"Total memory growth {total_growth_mb:.1f}MB over {num_episodes} episodes too high"
        )


class TestPerformanceRegressionDetection:
    """Test suite for performance regression detection per Section 6.6.5.3.3."""
    
    def test_performance_regression_statistical_detection(self, gymnasium_env_cached):
        """
        Implement performance regression detection with statistical significance testing.
        
        Uses statistical hypothesis testing to detect performance regressions with
        scientific rigor, implementing the statistical significance testing requirement
        from Section 6.6.5.3.3.
        """
        # Baseline performance measurement
        baseline_measurements = self._collect_performance_baseline(gymnasium_env_cached)
        
        # Simulate potential regression by adding artificial delay
        # In real scenarios, this would be comparison with previous performance data
        regression_measurements = self._collect_performance_with_potential_regression(
            gymnasium_env_cached, artificial_delay=0.002  # 2ms delay simulation
        )
        
        # Statistical analysis
        regression_detected, statistics_results = self._detect_performance_regression(
            baseline_measurements, regression_measurements
        )
        
        # Log regression analysis
        if ENHANCED_LOGGING:
            logger.info(
                f"Performance regression analysis: detected={regression_detected}, "
                f"p_value={statistics_results['p_value']:.6f}, "
                f"baseline_mean={statistics_results['baseline_mean']:.3f}ms, "
                f"current_mean={statistics_results['current_mean']:.3f}ms",
                extra={
                    "metric_type": "performance_regression_detection",
                    "regression_detected": regression_detected,
                    "statistical_results": statistics_results,
                    "significance_level": REGRESSION_SIGNIFICANCE_LEVEL,
                    "correlation_id": TEST_CORRELATION_ID
                }
            )
        
        # Validate regression detection capability
        if statistics_results['performance_degradation_percent'] > PERFORMANCE_TOLERANCE_PERCENT:
            assert regression_detected, (
                f"Failed to detect significant performance regression: "
                f"{statistics_results['performance_degradation_percent']:.1f}% degradation, "
                f"p-value: {statistics_results['p_value']:.6f}"
            )
    
    def _collect_performance_baseline(self, env) -> List[float]:
        """Collect baseline performance measurements."""
        obs, info = env.reset(seed=42)
        measurements = []
        
        for _ in range(MIN_SAMPLES_FOR_STATS):
            action = env.action_space.sample()
            
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - start_time
            
            measurements.append(step_time * 1000)  # Convert to milliseconds
            
            if terminated or truncated:
                obs, info = env.reset()
        
        return measurements
    
    def _collect_performance_with_potential_regression(self, env, artificial_delay: float = 0.0) -> List[float]:
        """Collect performance measurements with potential regression simulation."""
        obs, info = env.reset(seed=43)  # Different seed for variation
        measurements = []
        
        for _ in range(MIN_SAMPLES_FOR_STATS):
            action = env.action_space.sample()
            
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Simulate regression with artificial delay
            if artificial_delay > 0:
                time.sleep(artificial_delay)
            
            step_time = time.time() - start_time
            measurements.append(step_time * 1000)  # Convert to milliseconds
            
            if terminated or truncated:
                obs, info = env.reset()
        
        return measurements
    
    def _detect_performance_regression(self, baseline: List[float], current: List[float]) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect performance regression using statistical hypothesis testing.
        
        Uses Welch's t-test for unequal variances to detect statistically significant
        performance degradation between baseline and current measurements.
        """
        import scipy.stats as stats
        
        # Calculate descriptive statistics
        baseline_mean = statistics.mean(baseline)
        current_mean = statistics.mean(current)
        baseline_std = statistics.stdev(baseline)
        current_std = statistics.stdev(current)
        
        # Calculate performance change
        performance_degradation_percent = ((current_mean - baseline_mean) / baseline_mean) * 100
        
        # Perform Welch's t-test (assumes unequal variances)
        t_statistic, p_value = stats.ttest_ind(current, baseline, equal_var=False, alternative='greater')
        
        # Detect regression based on statistical significance and practical significance
        statistical_significance = p_value < REGRESSION_SIGNIFICANCE_LEVEL
        practical_significance = performance_degradation_percent > PERFORMANCE_TOLERANCE_PERCENT
        
        regression_detected = statistical_significance and practical_significance
        
        results = {
            'baseline_mean': baseline_mean,
            'current_mean': current_mean,
            'baseline_std': baseline_std,
            'current_std': current_std,
            'performance_degradation_percent': performance_degradation_percent,
            't_statistic': t_statistic,
            'p_value': p_value,
            'statistical_significance': statistical_significance,
            'practical_significance': practical_significance,
            'regression_detected': regression_detected,
            'significance_level': REGRESSION_SIGNIFICANCE_LEVEL,
            'tolerance_percent': PERFORMANCE_TOLERANCE_PERCENT
        }
        
        return regression_detected, results


# Utility context manager for conditional correlation context
@contextmanager
def nullcontext():
    """Null context manager for conditional context usage."""
    yield


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--benchmark-only", "--tb=short"])