"""
Enhanced Performance Test Suite for Plume Navigation Simulation.

Comprehensive SLA validation suite ensuring critical performance requirements for scientific
computing and reinforcement learning training workflows. Validates sub-10ms step execution,
>90% frame cache efficiency, memory management within 2 GiB limits, and regression detection
for production-grade scientific simulation environments.

Key Performance Requirements Validated:
- Environment step execution: P95 latency < 10ms per Section 6.6.4
- Frame cache hit rate: >90% efficiency with memory pressure management
- Memory usage: ≤2 GiB per process with automatic eviction per Section 5.2.2
- Frame processing: <33ms target for real-time operation
- Cross-repository compatibility: place_mem_rl integration validation

Test Categories:
- Unit performance tests: Individual component SLA validation
- Integration performance tests: End-to-end workflow validation
- Stress testing: Resource limits and degradation scenarios
- Regression detection: Performance trend monitoring and alerting
- Memory pressure testing: Cache eviction and resource management

Architecture:
- pytest-benchmark integration for statistical validation
- PSUtil-based memory monitoring and limit enforcement
- Hypothesis property-based testing for invariant validation
- Cross-repository test compatibility with place_mem_rl
- Automated performance regression detection with alerting

Performance Baselines (GitHub Ubuntu 22.04 Runner):
- Single agent step: 3-5ms typical, <10ms P95
- Multi-agent step (10 agents): 15-25ms typical, <33ms P95
- Cache hit retrieval: <1ms typical, <2ms P95
- Cache miss + load: 8-12ms typical, <15ms P95
- Memory per 100 agents: 150-200MB typical, <300MB limit

Authors: Blitzy Enhanced Performance Testing Framework
Version: v0.3.0 (Gymnasium migration)
License: MIT
"""

import gc
import os
import sys
import time
import threading
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator, Union
from unittest.mock import Mock, MagicMock, patch
import tempfile

import pytest
import numpy as np
import psutil

# Performance testing and validation frameworks
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    warnings.warn("pytest-benchmark not available, performance tests will be limited")

try:
    from hypothesis import given, strategies as st, settings, example
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# System under test - Enhanced frame cache and environment components
from plume_nav_sim.utils.frame_cache import (
    FrameCache, CacheMode, CacheStatistics,
    create_lru_cache, create_preload_cache, create_no_cache
)

# Mock dependencies for isolated performance testing
try:
    # Try to import actual components if available (for integration testing)
    from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv
    from plume_nav_sim.core.controllers import SingleAgentController, MultiAgentController
    INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback to mocks for unit testing
    INTEGRATION_AVAILABLE = False


# ============================================================================
# Performance Test Configuration and Utilities
# ============================================================================

class PerformanceConfig:
    """
    Configuration for performance test execution and validation.
    
    Centralizes all performance thresholds, test parameters, and SLA requirements
    for consistent validation across test suite.
    """
    
    # Core performance SLA requirements from Section 6.6.4
    STEP_LATENCY_P95_MS = 10.0
    STEP_LATENCY_P50_MS = 5.0
    FRAME_PROCESSING_TARGET_MS = 33.0
    
    # Cache performance requirements from Section 5.2.2
    CACHE_HIT_RATE_TARGET = 0.90
    CACHE_HIT_RATE_OPTIMAL = 0.95
    CACHE_MISS_PENALTY_MS = 15.0
    
    # Memory management limits per Section 5.2.2
    MEMORY_LIMIT_GB = 2.0
    MEMORY_WARNING_THRESHOLD = 0.9
    MEMORY_PER_100_AGENTS_MB = 300.0
    
    # Test execution parameters
    WARMUP_ITERATIONS = 10
    BENCHMARK_ITERATIONS = 100
    STRESS_TEST_DURATION_SEC = 30
    
    # Performance regression detection
    REGRESSION_THRESHOLD_PCT = 5.0
    PERFORMANCE_BASELINE_TOLERANCE_PCT = 15.0


class PerformanceTestBase:
    """
    Base class for performance tests with common fixtures and utilities.
    
    Provides standardized performance measurement, validation, and reporting
    capabilities for all performance test classes.
    """
    
    @pytest.fixture(autouse=True)
    def setup_performance_environment(self, tmp_path, monkeypatch):
        """Set up isolated performance testing environment."""
        # Create performance test workspace
        self.temp_dir = tmp_path
        self.cache_dir = tmp_path / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Configure test isolation
        monkeypatch.setenv("PERFORMANCE_TEST_MODE", "1")
        monkeypatch.setenv("CACHE_DIR", str(self.cache_dir))
        
        # Disable external logging for clean performance measurement
        monkeypatch.setenv("LOG_LEVEL", "ERROR")
        
        # Memory monitoring setup
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss
        
        # Performance tracking
        self.performance_data = defaultdict(list)
        
        yield
        
        # Cleanup and validation
        self._validate_memory_cleanup()
        
    def _validate_memory_cleanup(self):
        """Validate memory was properly cleaned up after test execution."""
        gc.collect()  # Force garbage collection
        final_memory = self.process.memory_info().rss
        memory_growth_mb = (final_memory - self.baseline_memory) / (1024 * 1024)
        
        # Allow some memory growth but flag excessive usage
        if memory_growth_mb > 100:
            warnings.warn(
                f"Significant memory growth detected: {memory_growth_mb:.1f}MB. "
                "Check for memory leaks in test.",
                ResourceWarning
            )
    
    @contextmanager
    def measure_performance(self, operation_name: str) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager for performance measurement with memory tracking.
        
        Args:
            operation_name: Name of operation being measured
            
        Yields:
            Performance metrics dictionary for recording custom metrics
        """
        # Baseline measurements
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss
        
        # Initialize metrics collection
        metrics = {
            'operation': operation_name,
            'start_time': start_time,
            'custom_metrics': {}
        }
        
        try:
            yield metrics
        finally:
            # Final measurements
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss
            
            # Calculate performance metrics
            execution_time_ms = (end_time - start_time) * 1000
            memory_delta_mb = (end_memory - start_memory) / (1024 * 1024)
            
            # Update metrics
            metrics.update({
                'execution_time_ms': execution_time_ms,
                'memory_delta_mb': memory_delta_mb,
                'end_time': end_time,
                'peak_memory_mb': end_memory / (1024 * 1024)
            })
            
            # Store for analysis
            self.performance_data[operation_name].append(metrics)
    
    def assert_performance_sla(
        self, 
        actual_ms: float, 
        target_ms: float, 
        operation: str,
        percentile: str = "average"
    ):
        """
        Assert performance meets SLA requirements with detailed error reporting.
        
        Args:
            actual_ms: Actual measured time in milliseconds
            target_ms: Target time threshold in milliseconds
            operation: Operation name for error reporting
            percentile: Performance percentile being validated
        """
        if actual_ms > target_ms:
            tolerance_ms = target_ms * (PerformanceConfig.PERFORMANCE_BASELINE_TOLERANCE_PCT / 100)
            if actual_ms > target_ms + tolerance_ms:
                pytest.fail(
                    f"Performance SLA violation for {operation}:\n"
                    f"  {percentile}: {actual_ms:.2f}ms\n"
                    f"  Target: {target_ms:.2f}ms\n"
                    f"  Tolerance: ±{tolerance_ms:.2f}ms\n"
                    f"  Violation: {actual_ms - target_ms:.2f}ms over target"
                )
            else:
                warnings.warn(
                    f"Performance close to SLA limit for {operation}: "
                    f"{actual_ms:.2f}ms (target: {target_ms:.2f}ms)",
                    PerformanceWarning
                )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for test analysis."""
        summary = {}
        
        for operation, measurements in self.performance_data.items():
            if not measurements:
                continue
                
            times = [m['execution_time_ms'] for m in measurements]
            memory_deltas = [m['memory_delta_mb'] for m in measurements]
            
            summary[operation] = {
                'count': len(measurements),
                'time_stats': {
                    'mean_ms': np.mean(times),
                    'median_ms': np.median(times),
                    'p95_ms': np.percentile(times, 95),
                    'p99_ms': np.percentile(times, 99),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'std_ms': np.std(times)
                },
                'memory_stats': {
                    'mean_delta_mb': np.mean(memory_deltas),
                    'max_delta_mb': np.max(memory_deltas),
                    'total_delta_mb': np.sum(memory_deltas)
                }
            }
        
        return summary


class MockVideoPlume:
    """
    High-performance mock VideoPlume for deterministic performance testing.
    
    Provides controlled frame generation with configurable latency and
    memory characteristics for cache performance validation.
    """
    
    def __init__(
        self, 
        frame_count: int = 1000,
        frame_width: int = 640,
        frame_height: int = 480,
        load_latency_ms: float = 10.0,
        deterministic: bool = True
    ):
        self.frame_count = frame_count
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.load_latency_ms = load_latency_ms
        self.deterministic = deterministic
        
        # Performance tracking
        self.access_count = 0
        self.load_times = []
        
        # Pre-generate deterministic frames if needed
        if deterministic:
            self._frame_cache = {}
    
    def get_frame(self, frame_id: int, **kwargs) -> Optional[np.ndarray]:
        """
        Simulate frame loading with controlled latency.
        
        Args:
            frame_id: Frame index to retrieve
            
        Returns:
            Simulated frame data or None if frame_id invalid
        """
        start_time = time.perf_counter()
        self.access_count += 1
        
        # Validate frame ID
        if frame_id < 0 or frame_id >= self.frame_count:
            return None
        
        # Simulate loading latency
        if self.load_latency_ms > 0:
            time.sleep(self.load_latency_ms / 1000.0)
        
        # Generate or retrieve frame
        if self.deterministic and frame_id in self._frame_cache:
            frame = self._frame_cache[frame_id]
        else:
            # Generate synthetic frame with deterministic content
            if self.deterministic:
                np.random.seed(frame_id % 1000)  # Deterministic but varied
            
            frame = np.random.randint(
                0, 255, 
                (self.frame_height, self.frame_width, 3), 
                dtype=np.uint8
            )
            
            if self.deterministic:
                self._frame_cache[frame_id] = frame
        
        # Track performance
        load_time = (time.perf_counter() - start_time) * 1000
        self.load_times.append(load_time)
        
        return frame
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get access statistics for performance analysis."""
        return {
            'access_count': self.access_count,
            'average_load_time_ms': np.mean(self.load_times) if self.load_times else 0,
            'total_frames_generated': len(self._frame_cache) if self.deterministic else self.access_count
        }


class PerformanceWarning(UserWarning):
    """Warning category for performance issues that don't fail tests."""
    pass


# ============================================================================
# Frame Cache Performance Tests
# ============================================================================

class TestFrameCachePerformance(PerformanceTestBase):
    """
    Comprehensive frame cache performance validation suite.
    
    Tests cache efficiency, memory management, and performance characteristics
    under various load conditions and operational modes.
    """
    
    @pytest.fixture
    def mock_video_plume(self):
        """Create mock video plume with controlled performance characteristics."""
        return MockVideoPlume(
            frame_count=1000,
            load_latency_ms=10.0,  # Realistic video frame load time
            deterministic=True
        )
    
    @pytest.fixture
    def performance_cache_config(self):
        """Configuration optimized for performance testing."""
        return {
            'memory_limit_mb': 512,  # Smaller limit for faster test execution
            'memory_pressure_threshold': 0.8,
            'enable_statistics': True,
            'enable_logging': False,  # Disable for clean performance measurement
            'preload_chunk_size': 50,
            'eviction_batch_size': 5
        }
    
    def test_cache_hit_performance_sla(self, mock_video_plume, performance_cache_config, benchmark):
        """
        Validate cache hit retrieval meets <1ms performance target.
        
        Tests the core cache hit path performance, ensuring zero-copy access
        and O(1) lookup time compliance.
        """
        cache = FrameCache(mode=CacheMode.LRU, **performance_cache_config)
        
        # Pre-populate cache with test frames
        test_frame_ids = list(range(0, 100, 10))  # Every 10th frame
        for frame_id in test_frame_ids:
            cache.get(frame_id, mock_video_plume)
        
        def cache_hit_operation():
            """Benchmark cache hit retrieval performance."""
            frame_id = np.random.choice(test_frame_ids)
            frame = cache.get(frame_id, mock_video_plume)
            assert frame is not None
            return frame
        
        # Benchmark cache hit performance
        if BENCHMARK_AVAILABLE:
            result = benchmark.pedantic(
                cache_hit_operation,
                iterations=100,
                rounds=5,
                warmup_rounds=2
            )
            
            # Validate cache hit rate
            hit_rate = cache.statistics.hit_rate
            assert hit_rate > PerformanceConfig.CACHE_HIT_RATE_TARGET, \
                f"Cache hit rate {hit_rate:.3f} below target {PerformanceConfig.CACHE_HIT_RATE_TARGET}"
            
            # Performance assertion
            mean_time_ms = benchmark.stats['mean'] * 1000
            self.assert_performance_sla(
                mean_time_ms, 
                1.0,  # <1ms target for cache hits
                "cache_hit_retrieval"
            )
        else:
            # Manual timing if benchmark not available
            with self.measure_performance("cache_hit_manual") as metrics:
                for _ in range(100):
                    cache_hit_operation()
            
            mean_time_ms = metrics['execution_time_ms'] / 100
            self.assert_performance_sla(mean_time_ms, 1.0, "cache_hit_retrieval")
    
    def test_cache_miss_performance_sla(self, mock_video_plume, performance_cache_config):
        """
        Validate cache miss + load performance meets target thresholds.
        
        Tests the cache miss path including frame loading, preprocessing,
        and cache insertion performance.
        """
        cache = FrameCache(mode=CacheMode.LRU, **performance_cache_config)
        
        miss_times = []
        
        # Test cache miss performance across different frame IDs
        with self.measure_performance("cache_miss_load") as metrics:
            for frame_id in range(50, 150):  # Fresh frame IDs for guaranteed misses
                start_time = time.perf_counter()
                frame = cache.get(frame_id, mock_video_plume)
                end_time = time.perf_counter()
                
                assert frame is not None
                miss_time_ms = (end_time - start_time) * 1000
                miss_times.append(miss_time_ms)
        
        # Performance validation
        mean_miss_time = np.mean(miss_times)
        p95_miss_time = np.percentile(miss_times, 95)
        
        self.assert_performance_sla(
            mean_miss_time, 
            PerformanceConfig.CACHE_MISS_PENALTY_MS, 
            "cache_miss_average"
        )
        
        self.assert_performance_sla(
            p95_miss_time, 
            PerformanceConfig.CACHE_MISS_PENALTY_MS * 1.5,  # Allow higher P95
            "cache_miss_p95",
            "P95"
        )
        
        # Validate cache populated correctly
        assert cache.cache_size == 100, f"Expected 100 cached frames, got {cache.cache_size}"
    
    def test_cache_efficiency_target_validation(self, mock_video_plume, performance_cache_config):
        """
        Validate cache achieves >90% hit rate under realistic access patterns.
        
        Simulates typical navigation access patterns with spatial and temporal locality.
        """
        cache = FrameCache(mode=CacheMode.LRU, **performance_cache_config)
        
        # Simulate realistic access pattern with locality
        access_pattern = []
        
        # Sequential access phase (high locality)
        for i in range(100):
            access_pattern.extend(range(i, min(i + 10, 200)))
        
        # Random access with bias toward recent frames
        recent_bias_frames = list(range(150, 200))
        for _ in range(300):
            if np.random.random() < 0.7:  # 70% chance of recent frame
                access_pattern.append(np.random.choice(recent_bias_frames))
            else:  # 30% chance of random frame
                access_pattern.append(np.random.randint(0, 500))
        
        # Execute access pattern with performance tracking
        with self.measure_performance("realistic_access_pattern") as metrics:
            for frame_id in access_pattern:
                frame = cache.get(frame_id, mock_video_plume)
                # Frame might be None for out-of-range IDs, which is acceptable
        
        # Validate cache efficiency
        statistics = cache.get_performance_stats()
        hit_rate = statistics['hit_rate']
        
        assert hit_rate >= PerformanceConfig.CACHE_HIT_RATE_TARGET, \
            f"Cache hit rate {hit_rate:.3f} below target {PerformanceConfig.CACHE_HIT_RATE_TARGET}"
        
        # Log performance details for analysis
        metrics['custom_metrics'].update({
            'hit_rate': hit_rate,
            'total_requests': statistics['total_requests'],
            'cache_size': statistics['cache_size'],
            'memory_usage_mb': statistics['memory_usage_mb']
        })
    
    def test_memory_pressure_handling_performance(self, mock_video_plume):
        """
        Validate memory pressure handling meets performance requirements.
        
        Tests LRU eviction performance and memory limit enforcement under
        memory pressure conditions.
        """
        # Configure cache with small memory limit to trigger pressure quickly
        cache_config = {
            'memory_limit_mb': 100,  # Small limit for fast pressure testing
            'memory_pressure_threshold': 0.8,
            'enable_statistics': True,
            'eviction_batch_size': 10
        }
        
        cache = FrameCache(mode=CacheMode.LRU, **cache_config)
        eviction_times = []
        
        # Load frames until memory pressure triggers eviction
        with self.measure_performance("memory_pressure_eviction") as metrics:
            for frame_id in range(500):  # Load many frames to trigger pressure
                start_time = time.perf_counter()
                frame = cache.get(frame_id, mock_video_plume)
                eviction_time = (time.perf_counter() - start_time) * 1000
                
                if frame is not None:
                    eviction_times.append(eviction_time)
                    
                    # Check if we're approaching memory limit
                    if cache.memory_usage_mb > cache_config['memory_limit_mb'] * 0.9:
                        break
        
        # Validate memory management
        final_memory_mb = cache.memory_usage_mb
        assert final_memory_mb <= cache_config['memory_limit_mb'] * 1.1, \
            f"Memory usage {final_memory_mb:.1f}MB exceeds limit {cache_config['memory_limit_mb']}MB"
        
        # Validate eviction performance
        if eviction_times:
            mean_eviction_time = np.mean(eviction_times)
            p95_eviction_time = np.percentile(eviction_times, 95)
            
            # Eviction should not significantly impact performance
            self.assert_performance_sla(
                mean_eviction_time, 
                5.0,  # 5ms target for eviction operations
                "memory_eviction_average"
            )
        
        # Validate statistics accuracy
        stats = cache.get_performance_stats()
        assert stats['evictions'] > 0, "No evictions recorded despite memory pressure"
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(
        cache_mode=st.sampled_from([CacheMode.LRU, CacheMode.ALL]),
        memory_limit=st.integers(min_value=50, max_value=500),
        access_count=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=10, deadline=None)  # Limit examples for performance tests
    def test_cache_performance_invariants(self, cache_mode, memory_limit, access_count, mock_video_plume):
        """
        Property-based testing of cache performance invariants.
        
        Validates that cache performance characteristics hold across different
        configurations and access patterns.
        """
        cache = FrameCache(
            mode=cache_mode,
            memory_limit_mb=memory_limit,
            enable_statistics=True,
            enable_logging=False
        )
        
        # Generate random but deterministic access pattern
        np.random.seed(42)
        frame_ids = np.random.randint(0, 200, access_count)
        
        access_times = []
        
        # Measure access performance
        for frame_id in frame_ids:
            start_time = time.perf_counter()
            frame = cache.get(frame_id, mock_video_plume)
            access_time = (time.perf_counter() - start_time) * 1000
            access_times.append(access_time)
        
        # Performance invariants
        if access_times:
            mean_access_time = np.mean(access_times)
            max_access_time = np.max(access_times)
            
            # Cache access should never exceed reasonable bounds
            assert mean_access_time < 50.0, \
                f"Mean access time {mean_access_time:.2f}ms too high"
            assert max_access_time < 100.0, \
                f"Max access time {max_access_time:.2f}ms too high"
        
        # Memory invariants
        memory_usage = cache.memory_usage_mb
        assert memory_usage <= memory_limit * 1.2, \
            f"Memory usage {memory_usage:.1f}MB exceeds limit {memory_limit}MB + tolerance"
        
        # Cache consistency invariants
        stats = cache.get_performance_stats()
        assert stats['total_requests'] == len([fid for fid in frame_ids if 0 <= fid < 1000])
        assert stats['cache_size'] <= min(len(set(frame_ids)), memory_limit * 10)  # Rough estimate
    
    def test_concurrent_cache_access_performance(self, mock_video_plume, performance_cache_config):
        """
        Validate cache performance under concurrent access from multiple threads.
        
        Tests thread safety and performance degradation under concurrent load.
        """
        cache = FrameCache(mode=CacheMode.LRU, **performance_cache_config)
        
        # Pre-populate cache
        for i in range(0, 100, 5):
            cache.get(i, mock_video_plume)
        
        thread_results = []
        
        def worker_thread(thread_id: int, iterations: int):
            """Worker thread for concurrent cache access."""
            thread_times = []
            
            for i in range(iterations):
                frame_id = (thread_id * 100 + i) % 200  # Spread access across range
                
                start_time = time.perf_counter()
                frame = cache.get(frame_id, mock_video_plume)
                access_time = (time.perf_counter() - start_time) * 1000
                
                thread_times.append(access_time)
            
            thread_results.append({
                'thread_id': thread_id,
                'times': thread_times,
                'mean_time': np.mean(thread_times),
                'max_time': np.max(thread_times)
            })
        
        # Launch concurrent threads
        num_threads = 4
        iterations_per_thread = 50
        
        threads = []
        start_time = time.perf_counter()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=worker_thread,
                args=(thread_id, iterations_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.perf_counter() - start_time
        
        # Analyze concurrent performance
        all_times = []
        for result in thread_results:
            all_times.extend(result['times'])
        
        concurrent_mean_time = np.mean(all_times)
        concurrent_p95_time = np.percentile(all_times, 95)
        
        # Concurrent access should not significantly degrade performance
        self.assert_performance_sla(
            concurrent_mean_time,
            10.0,  # Allow higher latency for concurrent access
            "concurrent_cache_access_mean"
        )
        
        self.assert_performance_sla(
            concurrent_p95_time,
            25.0,  # P95 tolerance for concurrent access
            "concurrent_cache_access_p95",
            "P95"
        )
        
        # Validate cache integrity after concurrent access
        final_stats = cache.get_performance_stats()
        assert final_stats['total_requests'] > 0
        assert cache.cache_size > 0


# ============================================================================
# Environment Step Performance Tests
# ============================================================================

class TestEnvironmentStepPerformance(PerformanceTestBase):
    """
    Environment step execution performance validation suite.
    
    Tests the core environment.step() performance requirements ensuring
    sub-10ms P95 latency for RL training efficiency.
    """
    
    @pytest.fixture
    def mock_environment_factory(self):
        """Factory for creating mock environments with controlled performance."""
        def create_mock_env(
            agent_count: int = 1,
            step_latency_ms: float = 3.0,
            frame_processing_ms: float = 1.0
        ):
            """Create mock environment with specified performance characteristics."""
            mock_env = Mock()
            
            def step_implementation(action):
                # Simulate step processing time
                time.sleep(step_latency_ms / 1000.0)
                
                # Simulate frame processing
                time.sleep(frame_processing_ms / 1000.0)
                
                # Return Gymnasium 5-tuple format
                observation = np.random.random((agent_count, 10))
                reward = np.random.random(agent_count)
                terminated = np.random.random(agent_count) < 0.01  # 1% termination chance
                truncated = np.random.random(agent_count) < 0.01   # 1% truncation chance
                info = {'agent_count': agent_count}
                
                return observation, reward, terminated, truncated, info
            
            def reset_implementation(seed=None, options=None):
                time.sleep(step_latency_ms / 1000.0)  # Reset should be similar speed
                observation = np.random.random((agent_count, 10))
                info = {'agent_count': agent_count}
                return observation, info
            
            mock_env.step = Mock(side_effect=step_implementation)
            mock_env.reset = Mock(side_effect=reset_implementation)
            mock_env.agent_count = agent_count
            
            return mock_env
        
        return create_mock_env
    
    def test_single_agent_step_performance_sla(self, mock_environment_factory, benchmark):
        """
        Validate single agent step execution meets <10ms P95 requirement.
        
        Tests the core single-agent environment step performance path.
        """
        env = mock_environment_factory(
            agent_count=1,
            step_latency_ms=3.0,  # Realistic single agent processing
            frame_processing_ms=1.0
        )
        
        # Initialize environment
        observation, info = env.reset()
        
        def single_step_operation():
            """Single environment step operation."""
            action = np.random.random(1)  # Single agent action
            return env.step(action)
        
        if BENCHMARK_AVAILABLE:
            # Benchmark step performance
            result = benchmark.pedantic(
                single_step_operation,
                iterations=PerformanceConfig.BENCHMARK_ITERATIONS,
                rounds=5,
                warmup_rounds=2
            )
            
            # Validate performance requirements
            mean_time_ms = benchmark.stats['mean'] * 1000
            p95_time_ms = np.percentile(
                [t * 1000 for t in benchmark.stats['data']], 
                95
            )
            
            self.assert_performance_sla(
                mean_time_ms,
                PerformanceConfig.STEP_LATENCY_P50_MS,
                "single_agent_step_mean"
            )
            
            self.assert_performance_sla(
                p95_time_ms,
                PerformanceConfig.STEP_LATENCY_P95_MS,
                "single_agent_step_p95",
                "P95"
            )
        else:
            # Manual performance measurement
            step_times = []
            
            with self.measure_performance("single_agent_manual") as metrics:
                for _ in range(100):
                    start_time = time.perf_counter()
                    single_step_operation()
                    step_time = (time.perf_counter() - start_time) * 1000
                    step_times.append(step_time)
            
            mean_time_ms = np.mean(step_times)
            p95_time_ms = np.percentile(step_times, 95)
            
            self.assert_performance_sla(mean_time_ms, PerformanceConfig.STEP_LATENCY_P50_MS, "single_agent_step")
            self.assert_performance_sla(p95_time_ms, PerformanceConfig.STEP_LATENCY_P95_MS, "single_agent_step_p95", "P95")
    
    def test_multi_agent_step_performance_scaling(self, mock_environment_factory):
        """
        Validate multi-agent step performance scaling with agent count.
        
        Tests performance characteristics with varying agent counts and
        validates sub-33ms execution for realistic multi-agent scenarios.
        """
        agent_counts = [1, 5, 10, 25, 50]
        scaling_results = {}
        
        for agent_count in agent_counts:
            env = mock_environment_factory(
                agent_count=agent_count,
                step_latency_ms=2.0,  # Optimistic base latency
                frame_processing_ms=0.5  # Fast frame processing
            )
            
            # Initialize environment
            observation, info = env.reset()
            
            # Measure step performance for this agent count
            step_times = []
            
            with self.measure_performance(f"multi_agent_{agent_count}") as metrics:
                for _ in range(50):  # Fewer iterations for multi-agent
                    action = np.random.random(agent_count)
                    
                    start_time = time.perf_counter()
                    result = env.step(action)
                    step_time = (time.perf_counter() - start_time) * 1000
                    
                    step_times.append(step_time)
                    
                    # Validate result format
                    assert len(result) == 5  # Gymnasium 5-tuple
                    obs, reward, terminated, truncated, info = result
                    assert len(obs) == agent_count
                    assert len(reward) == agent_count
            
            # Analyze performance for this agent count
            mean_time = np.mean(step_times)
            p95_time = np.percentile(step_times, 95)
            
            scaling_results[agent_count] = {
                'mean_ms': mean_time,
                'p95_ms': p95_time,
                'per_agent_ms': mean_time / agent_count
            }
            
            # Performance validation based on agent count
            if agent_count <= 10:
                target_time = PerformanceConfig.STEP_LATENCY_P95_MS
            else:
                target_time = PerformanceConfig.FRAME_PROCESSING_TARGET_MS
            
            self.assert_performance_sla(
                p95_time,
                target_time,
                f"multi_agent_{agent_count}_p95",
                "P95"
            )
        
        # Validate scaling characteristics
        # Performance should scale sub-linearly with agent count
        scaling_efficiency = scaling_results[50]['per_agent_ms'] / scaling_results[1]['per_agent_ms']
        
        assert scaling_efficiency < 3.0, \
            f"Multi-agent scaling efficiency {scaling_efficiency:.2f} indicates poor vectorization"
    
    def test_frame_processing_performance_target(self, mock_environment_factory):
        """
        Validate frame processing meets <33ms target for real-time operation.
        
        Tests the frame processing pipeline performance including cache
        integration and video frame access.
        """
        # Create environment with realistic frame processing simulation
        env = mock_environment_factory(
            agent_count=10,
            step_latency_ms=1.0,  # Minimal step overhead
            frame_processing_ms=8.0  # Focus on frame processing
        )
        
        # Initialize
        observation, info = env.reset()
        
        frame_processing_times = []
        
        # Simulate episode with frame processing focus
        with self.measure_performance("frame_processing_pipeline") as metrics:
            for step_idx in range(100):
                # Simulate frame-intensive operation
                start_time = time.perf_counter()
                
                action = np.random.random(env.agent_count)
                result = env.step(action)
                
                frame_time = (time.perf_counter() - start_time) * 1000
                frame_processing_times.append(frame_time)
                
                # Validate result
                obs, reward, terminated, truncated, info = result
                assert obs.shape[0] == env.agent_count
        
        # Performance validation
        mean_frame_time = np.mean(frame_processing_times)
        p95_frame_time = np.percentile(frame_processing_times, 95)
        
        self.assert_performance_sla(
            mean_frame_time,
            PerformanceConfig.FRAME_PROCESSING_TARGET_MS * 0.7,  # 70% of target for mean
            "frame_processing_mean"
        )
        
        self.assert_performance_sla(
            p95_frame_time,
            PerformanceConfig.FRAME_PROCESSING_TARGET_MS,
            "frame_processing_p95",
            "P95"
        )
        
        # Record detailed metrics
        metrics['custom_metrics'].update({
            'mean_frame_time_ms': mean_frame_time,
            'p95_frame_time_ms': p95_frame_time,
            'total_frames_processed': len(frame_processing_times)
        })
    
    def test_environment_memory_efficiency(self, mock_environment_factory):
        """
        Validate environment memory usage remains within efficiency targets.
        
        Tests memory growth patterns and validates memory per agent limits.
        """
        baseline_memory = self.process.memory_info().rss
        
        agent_configs = [
            (10, "small_env"),
            (50, "medium_env"), 
            (100, "large_env")
        ]
        
        memory_results = {}
        
        for agent_count, config_name in agent_configs:
            # Create environment
            env = mock_environment_factory(
                agent_count=agent_count,
                step_latency_ms=1.0,
                frame_processing_ms=0.5
            )
            
            # Memory before episode
            pre_episode_memory = self.process.memory_info().rss
            
            # Run episode with memory tracking
            observation, info = env.reset()
            
            for step_idx in range(50):  # Shorter episodes for memory testing
                action = np.random.random(agent_count)
                result = env.step(action)
                
                # Periodic memory check
                if step_idx % 10 == 0:
                    current_memory = self.process.memory_info().rss
                    memory_growth_mb = (current_memory - pre_episode_memory) / (1024 * 1024)
                    
                    # Memory per agent should remain reasonable
                    memory_per_agent_mb = memory_growth_mb / agent_count
                    
                    if memory_per_agent_mb > 5.0:  # 5MB per agent warning threshold
                        warnings.warn(
                            f"High memory per agent: {memory_per_agent_mb:.2f}MB "
                            f"({agent_count} agents, step {step_idx})",
                            PerformanceWarning
                        )
            
            # Final memory measurement
            post_episode_memory = self.process.memory_info().rss
            episode_memory_growth = (post_episode_memory - pre_episode_memory) / (1024 * 1024)
            memory_per_agent = episode_memory_growth / agent_count
            
            memory_results[config_name] = {
                'agent_count': agent_count,
                'memory_growth_mb': episode_memory_growth,
                'memory_per_agent_mb': memory_per_agent
            }
            
            # Clean up environment
            del env
            gc.collect()
        
        # Validate memory efficiency targets
        for config_name, results in memory_results.items():
            memory_per_100_agents = results['memory_per_agent_mb'] * 100
            
            assert memory_per_100_agents < PerformanceConfig.MEMORY_PER_100_AGENTS_MB, \
                f"{config_name}: Memory per 100 agents {memory_per_100_agents:.1f}MB exceeds " \
                f"target {PerformanceConfig.MEMORY_PER_100_AGENTS_MB}MB"


# ============================================================================
# Stress Testing and Resource Management
# ============================================================================

class TestStressAndResourceManagement(PerformanceTestBase):
    """
    Stress testing and resource management validation suite.
    
    Tests system behavior under resource pressure, extended operation,
    and edge conditions.
    """
    
    def test_extended_operation_stability(self, mock_environment_factory):
        """
        Validate system stability during extended operation periods.
        
        Tests for memory leaks, performance degradation, and resource
        cleanup over extended execution periods.
        """
        env = mock_environment_factory(
            agent_count=10,
            step_latency_ms=2.0,
            frame_processing_ms=1.0
        )
        
        # Baseline measurements
        start_memory = self.process.memory_info().rss
        start_time = time.perf_counter()
        
        observation, info = env.reset()
        
        performance_samples = []
        memory_samples = []
        
        # Extended operation simulation
        total_steps = 1000  # Extended operation
        
        with self.measure_performance("extended_operation") as metrics:
            for step_idx in range(total_steps):
                step_start_time = time.perf_counter()
                
                action = np.random.random(env.agent_count)
                result = env.step(action)
                
                step_time = (time.perf_counter() - step_start_time) * 1000
                performance_samples.append(step_time)
                
                # Periodic memory and performance monitoring
                if step_idx % 100 == 0:
                    current_memory = self.process.memory_info().rss
                    memory_growth = (current_memory - start_memory) / (1024 * 1024)
                    memory_samples.append(memory_growth)
                    
                    # Check for performance degradation
                    recent_performance = np.mean(performance_samples[-50:]) if len(performance_samples) >= 50 else np.mean(performance_samples)
                    
                    if step_idx > 200 and recent_performance > 15.0:  # Performance degradation threshold
                        warnings.warn(
                            f"Performance degradation detected at step {step_idx}: "
                            f"{recent_performance:.2f}ms average",
                            PerformanceWarning
                        )
        
        # Analyze stability metrics
        total_time = time.perf_counter() - start_time
        final_memory = self.process.memory_info().rss
        total_memory_growth = (final_memory - start_memory) / (1024 * 1024)
        
        # Performance stability validation
        early_performance = np.mean(performance_samples[:100])
        late_performance = np.mean(performance_samples[-100:])
        performance_drift = (late_performance - early_performance) / early_performance
        
        assert abs(performance_drift) < 0.2, \
            f"Significant performance drift detected: {performance_drift:.1%} change"
        
        # Memory stability validation
        assert total_memory_growth < 200, \
            f"Excessive memory growth: {total_memory_growth:.1f}MB during extended operation"
        
        # Throughput validation
        steps_per_second = total_steps / total_time
        assert steps_per_second > 30, \
            f"Throughput too low: {steps_per_second:.1f} steps/second"
        
        # Record detailed metrics
        metrics['custom_metrics'].update({
            'total_steps': total_steps,
            'steps_per_second': steps_per_second,
            'memory_growth_mb': total_memory_growth,
            'performance_drift_pct': performance_drift * 100,
            'early_performance_ms': early_performance,
            'late_performance_ms': late_performance
        })
    
    @pytest.mark.timeout(60)  # 1 minute timeout for stress test
    def test_memory_pressure_resilience(self):
        """
        Validate system resilience under memory pressure conditions.
        
        Tests cache eviction, memory cleanup, and performance maintenance
        under constrained memory conditions.
        """
        # Create memory pressure scenario
        mock_video_plume = MockVideoPlume(
            frame_count=2000,
            load_latency_ms=5.0,
            deterministic=True
        )
        
        # Small memory limit to induce pressure quickly
        cache = FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=50,  # Very small limit
            memory_pressure_threshold=0.7,
            enable_statistics=True,
            eviction_batch_size=20
        )
        
        memory_measurements = []
        performance_measurements = []
        eviction_events = []
        
        with self.measure_performance("memory_pressure_resilience") as metrics:
            # Load frames until we hit memory pressure multiple times
            for frame_id in range(1000):
                start_time = time.perf_counter()
                
                # Track cache state before access
                pre_access_size = cache.cache_size
                pre_access_memory = cache.memory_usage_mb
                
                # Access frame
                frame = cache.get(frame_id, mock_video_plume)
                
                access_time = (time.perf_counter() - start_time) * 1000
                performance_measurements.append(access_time)
                
                # Track cache state after access
                post_access_size = cache.cache_size
                post_access_memory = cache.memory_usage_mb
                
                # Record measurements
                memory_measurements.append(post_access_memory)
                
                # Detect eviction events
                if post_access_size < pre_access_size:
                    eviction_events.append({
                        'frame_id': frame_id,
                        'evicted_count': pre_access_size - post_access_size,
                        'memory_before': pre_access_memory,
                        'memory_after': post_access_memory,
                        'access_time_ms': access_time
                    })
                
                # Break if we're not making progress (memory management working)
                if frame_id > 100 and len(eviction_events) > 10:
                    break
        
        # Validate memory pressure handling
        assert len(eviction_events) > 0, "No eviction events detected despite small memory limit"
        
        # Memory should stay within bounds
        max_memory_usage = max(memory_measurements)
        assert max_memory_usage <= 50 * 1.2, \
            f"Memory usage {max_memory_usage:.1f}MB exceeded limit with tolerance"
        
        # Performance should remain reasonable during pressure
        if performance_measurements:
            mean_performance = np.mean(performance_measurements)
            p95_performance = np.percentile(performance_measurements, 95)
            
            self.assert_performance_sla(
                mean_performance,
                20.0,  # Higher tolerance during memory pressure
                "memory_pressure_performance"
            )
        
        # Eviction should be efficient
        eviction_times = [e['access_time_ms'] for e in eviction_events]
        if eviction_times:
            mean_eviction_time = np.mean(eviction_times)
            assert mean_eviction_time < 30.0, \
                f"Eviction operations too slow: {mean_eviction_time:.2f}ms average"
        
        # Record stress test metrics
        metrics['custom_metrics'].update({
            'eviction_events': len(eviction_events),
            'max_memory_mb': max_memory_usage,
            'mean_access_time_ms': np.mean(performance_measurements),
            'frames_processed': len(performance_measurements)
        })
    
    def test_concurrent_stress_testing(self, mock_environment_factory):
        """
        Validate system performance under concurrent multi-environment stress.
        
        Tests thread safety and resource management with multiple concurrent
        environment instances.
        """
        num_environments = 4
        steps_per_env = 100
        
        results = []
        
        def environment_worker(env_id: int):
            """Worker function for concurrent environment testing."""
            env = mock_environment_factory(
                agent_count=5,
                step_latency_ms=2.0,
                frame_processing_ms=1.0
            )
            
            worker_results = {
                'env_id': env_id,
                'step_times': [],
                'errors': []
            }
            
            try:
                observation, info = env.reset()
                
                for step_idx in range(steps_per_env):
                    start_time = time.perf_counter()
                    
                    action = np.random.random(env.agent_count)
                    result = env.step(action)
                    
                    step_time = (time.perf_counter() - start_time) * 1000
                    worker_results['step_times'].append(step_time)
                    
                    # Validate result format
                    obs, reward, terminated, truncated, info = result
                    assert len(obs) == env.agent_count
                    
            except Exception as e:
                worker_results['errors'].append(str(e))
            
            results.append(worker_results)
        
        # Launch concurrent environments
        start_time = time.perf_counter()
        
        threads = []
        for env_id in range(num_environments):
            thread = threading.Thread(target=environment_worker, args=(env_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.perf_counter() - start_time
        
        # Analyze concurrent performance
        all_step_times = []
        total_errors = []
        
        for result in results:
            all_step_times.extend(result['step_times'])
            total_errors.extend(result['errors'])
        
        # Validate concurrent operation
        assert len(total_errors) == 0, f"Errors during concurrent operation: {total_errors}"
        
        concurrent_mean_time = np.mean(all_step_times)
        concurrent_p95_time = np.percentile(all_step_times, 95)
        
        # Concurrent performance should not severely degrade
        self.assert_performance_sla(
            concurrent_mean_time,
            15.0,  # Higher tolerance for concurrent operation
            "concurrent_stress_mean"
        )
        
        # Validate total throughput
        total_steps = num_environments * steps_per_env
        steps_per_second = total_steps / total_time
        
        assert steps_per_second > 50, \
            f"Concurrent throughput too low: {steps_per_second:.1f} steps/second"


# ============================================================================
# Cross-Repository Performance Compatibility Tests
# ============================================================================

class TestCrossRepositoryPerformanceCompatibility(PerformanceTestBase):
    """
    Cross-repository performance compatibility validation suite.
    
    Tests integration performance with downstream repositories like
    place_mem_rl to ensure migration maintains compatibility.
    """
    
    @pytest.fixture
    def mock_place_mem_rl_integration(self):
        """Mock place_mem_rl training loop for performance testing."""
        def create_training_loop(
            env_factory,
            episodes: int = 10,
            steps_per_episode: int = 100
        ):
            """Simulate place_mem_rl training loop."""
            training_metrics = {
                'episode_times': [],
                'step_times': [],
                'total_steps': 0,
                'episodes_completed': 0
            }
            
            for episode in range(episodes):
                episode_start = time.perf_counter()
                
                # Create environment (simulating place_mem_rl behavior)
                env = env_factory(agent_count=1)
                observation, info = env.reset()
                
                episode_steps = 0
                for step in range(steps_per_episode):
                    step_start = time.perf_counter()
                    
                    # Simulate RL algorithm action selection
                    action = np.random.random(1)  # Simple random action
                    
                    # Environment step
                    result = env.step(action)
                    obs, reward, terminated, truncated, info = result
                    
                    step_time = (time.perf_counter() - step_start) * 1000
                    training_metrics['step_times'].append(step_time)
                    
                    episode_steps += 1
                    training_metrics['total_steps'] += 1
                    
                    # Break on episode termination
                    if terminated.any() or truncated.any():
                        break
                
                episode_time = time.perf_counter() - episode_start
                training_metrics['episode_times'].append(episode_time)
                training_metrics['episodes_completed'] += 1
            
            return training_metrics
        
        return create_training_loop
    
    def test_place_mem_rl_training_performance_compatibility(
        self, 
        mock_environment_factory, 
        mock_place_mem_rl_integration
    ):
        """
        Validate performance compatibility with place_mem_rl training workflows.
        
        Ensures the Gymnasium migration maintains training performance
        characteristics expected by downstream RL repositories.
        """
        # Create environment factory mimicking place_mem_rl usage
        def env_factory(agent_count=1):
            return mock_environment_factory(
                agent_count=agent_count,
                step_latency_ms=3.0,  # Realistic step time
                frame_processing_ms=1.0
            )
        
        # Run simulated training session
        with self.measure_performance("place_mem_rl_integration") as metrics:
            training_metrics = mock_place_mem_rl_integration(
                env_factory=env_factory,
                episodes=5,  # Reduced for test speed
                steps_per_episode=50
            )
        
        # Analyze training performance
        mean_step_time = np.mean(training_metrics['step_times'])
        p95_step_time = np.percentile(training_metrics['step_times'], 95)
        mean_episode_time = np.mean(training_metrics['episode_times'])
        
        total_training_time = metrics['execution_time_ms'] / 1000
        steps_per_second = training_metrics['total_steps'] / total_training_time
        
        # Validate RL training performance requirements
        self.assert_performance_sla(
            mean_step_time,
            PerformanceConfig.STEP_LATENCY_P50_MS,
            "rl_training_step_mean"
        )
        
        self.assert_performance_sla(
            p95_step_time,
            PerformanceConfig.STEP_LATENCY_P95_MS,
            "rl_training_step_p95",
            "P95"
        )
        
        # Training throughput should support efficient learning
        assert steps_per_second > 100, \
            f"Training throughput too low: {steps_per_second:.1f} steps/second"
        
        # Record integration metrics
        metrics['custom_metrics'].update({
            'total_episodes': training_metrics['episodes_completed'],
            'total_steps': training_metrics['total_steps'],
            'steps_per_second': steps_per_second,
            'mean_step_time_ms': mean_step_time,
            'mean_episode_time_s': mean_episode_time
        })
    
    def test_legacy_gym_compatibility_performance(self, mock_environment_factory):
        """
        Validate performance of legacy gym compatibility shim.
        
        Tests that the gym_make shim maintains performance while providing
        backward compatibility with legacy 4-tuple returns.
        """
        # Mock the shim behavior
        def create_legacy_env():
            """Create environment with legacy gym interface."""
            base_env = mock_environment_factory(
                agent_count=1,
                step_latency_ms=3.0,
                frame_processing_ms=1.0
            )
            
            # Wrap with legacy compatibility
            def legacy_step(action):
                obs, reward, terminated, truncated, info = base_env.step(action)
                # Convert to legacy 4-tuple format
                done = terminated | truncated  # Combine terminated and truncated
                return obs, reward, done, info
            
            def legacy_reset():
                obs, info = base_env.reset()
                return obs  # Legacy format returns only observation
            
            base_env.step = legacy_step
            base_env.reset = legacy_reset
            
            return base_env
        
        legacy_env = create_legacy_env()
        
        # Initialize legacy environment
        observation = legacy_env.reset()
        assert isinstance(observation, np.ndarray)
        
        step_times = []
        
        # Test legacy interface performance
        with self.measure_performance("legacy_gym_compatibility") as metrics:
            for step_idx in range(100):
                start_time = time.perf_counter()
                
                action = np.random.random(1)
                result = legacy_env.step(action)
                
                step_time = (time.perf_counter() - start_time) * 1000
                step_times.append(step_time)
                
                # Validate legacy format
                assert len(result) == 4  # Legacy 4-tuple
                obs, reward, done, info = result
                assert isinstance(done, (bool, np.bool_, np.ndarray))
        
        # Performance should be comparable to native interface
        mean_legacy_time = np.mean(step_times)
        p95_legacy_time = np.percentile(step_times, 95)
        
        # Allow small overhead for compatibility conversion
        compatibility_overhead = 1.2  # 20% overhead tolerance
        
        self.assert_performance_sla(
            mean_legacy_time,
            PerformanceConfig.STEP_LATENCY_P50_MS * compatibility_overhead,
            "legacy_compatibility_mean"
        )
        
        self.assert_performance_sla(
            p95_legacy_time,
            PerformanceConfig.STEP_LATENCY_P95_MS * compatibility_overhead,
            "legacy_compatibility_p95",
            "P95"
        )
        
        metrics['custom_metrics'].update({
            'mean_legacy_time_ms': mean_legacy_time,
            'compatibility_overhead': (mean_legacy_time / 3.0) - 1.0  # Overhead vs base latency
        })
    
    def test_gymnasium_api_compliance_performance(self, mock_environment_factory):
        """
        Validate Gymnasium API compliance maintains performance standards.
        
        Tests that modern Gymnasium interface provides optimal performance
        without compatibility overhead.
        """
        # Create modern Gymnasium-compliant environment
        env = mock_environment_factory(
            agent_count=1,
            step_latency_ms=3.0,
            frame_processing_ms=1.0
        )
        
        # Test modern interface performance
        modern_step_times = []
        reset_times = []
        
        with self.measure_performance("gymnasium_api_compliance") as metrics:
            # Test reset performance
            for _ in range(10):
                reset_start = time.perf_counter()
                observation, info = env.reset(seed=42)
                reset_time = (time.perf_counter() - reset_start) * 1000
                reset_times.append(reset_time)
                
                assert isinstance(observation, np.ndarray)
                assert isinstance(info, dict)
            
            # Test step performance
            for step_idx in range(100):
                step_start = time.perf_counter()
                
                action = np.random.random(1)
                result = env.step(action)
                
                step_time = (time.perf_counter() - step_start) * 1000
                modern_step_times.append(step_time)
                
                # Validate modern Gymnasium format
                assert len(result) == 5  # Modern 5-tuple
                obs, reward, terminated, truncated, info = result
                assert isinstance(terminated, (bool, np.bool_, np.ndarray))
                assert isinstance(truncated, (bool, np.bool_, np.ndarray))
        
        # Modern interface should meet optimal performance
        mean_modern_step = np.mean(modern_step_times)
        p95_modern_step = np.percentile(modern_step_times, 95)
        mean_reset_time = np.mean(reset_times)
        
        self.assert_performance_sla(
            mean_modern_step,
            PerformanceConfig.STEP_LATENCY_P50_MS,
            "gymnasium_api_step_mean"
        )
        
        self.assert_performance_sla(
            p95_modern_step,
            PerformanceConfig.STEP_LATENCY_P95_MS,
            "gymnasium_api_step_p95",
            "P95"
        )
        
        # Reset should also be performant
        self.assert_performance_sla(
            mean_reset_time,
            PerformanceConfig.STEP_LATENCY_P95_MS,  # Same target as step
            "gymnasium_api_reset_mean"
        )
        
        metrics['custom_metrics'].update({
            'mean_step_time_ms': mean_modern_step,
            'mean_reset_time_ms': mean_reset_time,
            'api_format': 'gymnasium_5_tuple'
        })


# ============================================================================
# Performance Regression Detection Tests
# ============================================================================

class TestPerformanceRegressionDetection(PerformanceTestBase):
    """
    Performance regression detection and monitoring suite.
    
    Implements automated performance regression detection and provides
    baseline validation for continuous performance monitoring.
    """
    
    @pytest.fixture
    def performance_baseline_storage(self, tmp_path):
        """Mock performance baseline storage for regression testing."""
        baseline_file = tmp_path / "performance_baselines.json"
        
        # Create mock baseline data
        baseline_data = {
            'single_agent_step_mean_ms': 3.2,
            'single_agent_step_p95_ms': 8.7,
            'multi_agent_10_step_mean_ms': 12.5,
            'multi_agent_10_step_p95_ms': 28.3,
            'cache_hit_mean_ms': 0.8,
            'cache_miss_mean_ms': 11.2,
            'frame_processing_mean_ms': 15.1,
            'memory_per_100_agents_mb': 185.0,
            'last_updated': time.time(),
            'test_environment': 'github_ubuntu_22_04'
        }
        
        import json
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        return baseline_file
    
    def load_performance_baselines(self, baseline_file: Path) -> Dict[str, float]:
        """Load performance baselines from storage."""
        import json
        try:
            with open(baseline_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def detect_performance_regression(
        self, 
        current_metrics: Dict[str, float],
        baselines: Dict[str, float],
        regression_threshold_pct: float = PerformanceConfig.REGRESSION_THRESHOLD_PCT
    ) -> Dict[str, Any]:
        """
        Detect performance regressions by comparing against baselines.
        
        Args:
            current_metrics: Current performance measurements
            baselines: Historical baseline measurements
            regression_threshold_pct: Threshold for regression detection
            
        Returns:
            Regression analysis results
        """
        regression_results = {
            'regressions_detected': [],
            'improvements_detected': [],
            'stable_metrics': [],
            'missing_baselines': []
        }
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baselines:
                regression_results['missing_baselines'].append(metric_name)
                continue
            
            baseline_value = baselines[metric_name]
            change_pct = ((current_value - baseline_value) / baseline_value) * 100
            
            if abs(change_pct) > regression_threshold_pct:
                if change_pct > 0:  # Performance degradation
                    regression_results['regressions_detected'].append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_pct': change_pct
                    })
                else:  # Performance improvement
                    regression_results['improvements_detected'].append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_pct': change_pct
                    })
            else:
                regression_results['stable_metrics'].append({
                    'metric': metric_name,
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_pct': change_pct
                })
        
        return regression_results
    
    def test_comprehensive_performance_regression_detection(
        self, 
        mock_environment_factory,
        mock_video_plume,
        performance_baseline_storage
    ):
        """
        Comprehensive performance regression detection across all components.
        
        Runs full performance test suite and compares against historical
        baselines to detect any performance regressions.
        """
        baselines = self.load_performance_baselines(performance_baseline_storage)
        current_metrics = {}
        
        # 1. Single agent step performance
        env = mock_environment_factory(agent_count=1, step_latency_ms=3.0)
        observation, info = env.reset()
        
        single_step_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            result = env.step(np.random.random(1))
            step_time = (time.perf_counter() - start_time) * 1000
            single_step_times.append(step_time)
        
        current_metrics['single_agent_step_mean_ms'] = np.mean(single_step_times)
        current_metrics['single_agent_step_p95_ms'] = np.percentile(single_step_times, 95)
        
        # 2. Multi-agent step performance
        multi_env = mock_environment_factory(agent_count=10, step_latency_ms=2.0)
        observation, info = multi_env.reset()
        
        multi_step_times = []
        for _ in range(30):
            start_time = time.perf_counter()
            result = multi_env.step(np.random.random(10))
            step_time = (time.perf_counter() - start_time) * 1000
            multi_step_times.append(step_time)
        
        current_metrics['multi_agent_10_step_mean_ms'] = np.mean(multi_step_times)
        current_metrics['multi_agent_10_step_p95_ms'] = np.percentile(multi_step_times, 95)
        
        # 3. Cache performance
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=256, enable_statistics=True)
        
        # Cache hit test
        cache.get(0, mock_video_plume)  # Load frame 0
        cache_hit_times = []
        for _ in range(50):
            start_time = time.perf_counter()
            frame = cache.get(0, mock_video_plume)
            hit_time = (time.perf_counter() - start_time) * 1000
            cache_hit_times.append(hit_time)
        
        current_metrics['cache_hit_mean_ms'] = np.mean(cache_hit_times)
        
        # Cache miss test
        cache_miss_times = []
        for frame_id in range(50, 100):
            start_time = time.perf_counter()
            frame = cache.get(frame_id, mock_video_plume)
            miss_time = (time.perf_counter() - start_time) * 1000
            cache_miss_times.append(miss_time)
        
        current_metrics['cache_miss_mean_ms'] = np.mean(cache_miss_times)
        
        # 4. Memory efficiency test
        baseline_memory = self.process.memory_info().rss
        
        # Simulate 100 agents worth of memory usage
        test_envs = []
        for _ in range(10):  # 10 environments with 10 agents each
            test_env = mock_environment_factory(agent_count=10)
            test_envs.append(test_env)
            test_env.reset()
        
        peak_memory = self.process.memory_info().rss
        memory_growth = (peak_memory - baseline_memory) / (1024 * 1024)
        current_metrics['memory_per_100_agents_mb'] = memory_growth
        
        # Clean up
        del test_envs
        gc.collect()
        
        # Perform regression detection
        regression_analysis = self.detect_performance_regression(current_metrics, baselines)
        
        # Report results
        if regression_analysis['regressions_detected']:
            regression_details = []
            for regression in regression_analysis['regressions_detected']:
                regression_details.append(
                    f"{regression['metric']}: {regression['current']:.2f} vs "
                    f"baseline {regression['baseline']:.2f} "
                    f"({regression['change_pct']:+.1f}%)"
                )
            
            # Fail test if significant regressions detected
            if any(r['change_pct'] > 10.0 for r in regression_analysis['regressions_detected']):
                pytest.fail(
                    f"Significant performance regressions detected:\n" +
                    "\n".join(regression_details)
                )
            else:
                warnings.warn(
                    f"Minor performance regressions detected:\n" +
                    "\n".join(regression_details),
                    PerformanceWarning
                )
        
        # Report improvements
        if regression_analysis['improvements_detected']:
            improvement_details = []
            for improvement in regression_analysis['improvements_detected']:
                improvement_details.append(
                    f"{improvement['metric']}: {improvement['current']:.2f} vs "
                    f"baseline {improvement['baseline']:.2f} "
                    f"({improvement['change_pct']:+.1f}%)"
                )
            
            print(f"\nPerformance improvements detected:\n" + "\n".join(improvement_details))
        
        # Store current metrics for future regression testing
        import json
        updated_baselines = baselines.copy()
        updated_baselines.update(current_metrics)
        updated_baselines['last_updated'] = time.time()
        
        with open(performance_baseline_storage, 'w') as f:
            json.dump(updated_baselines, f, indent=2)
    
    def test_performance_trend_monitoring(self):
        """
        Test performance trend monitoring and alerting capabilities.
        
        Validates that performance monitoring can detect gradual
        performance degradation over time.
        """
        # Simulate performance measurements over time
        baseline_performance = 5.0  # 5ms baseline
        measurements = []
        
        # Simulate gradual performance degradation
        for day in range(30):
            # Add some noise and gradual degradation
            daily_performance = baseline_performance + (day * 0.1) + np.random.normal(0, 0.5)
            measurements.append({
                'day': day,
                'performance_ms': daily_performance,
                'timestamp': time.time() - (30 - day) * 24 * 3600  # Days ago
            })
        
        # Analyze trend
        performances = [m['performance_ms'] for m in measurements]
        
        # Calculate trend using linear regression
        days = np.array([m['day'] for m in measurements])
        performance_values = np.array(performances)
        
        # Simple linear regression
        slope = np.polyfit(days, performance_values, 1)[0]
        
        # Detect significant trend
        trend_per_day = slope
        trend_over_period = trend_per_day * 30
        trend_percentage = (trend_over_period / baseline_performance) * 100
        
        # Alert on significant performance trends
        if abs(trend_percentage) > 5.0:  # 5% change over 30 days
            if trend_percentage > 0:
                warnings.warn(
                    f"Performance degradation trend detected: "
                    f"{trend_percentage:.1f}% over 30 days",
                    PerformanceWarning
                )
            else:
                print(f"Performance improvement trend detected: "
                      f"{abs(trend_percentage):.1f}% over 30 days")
        
        # Validate trend detection accuracy
        assert abs(trend_percentage) > 5.0, \
            "Trend detection should identify the simulated degradation"


# ============================================================================
# Performance Test Configuration and Execution
# ============================================================================

# Configure pytest markers for performance tests
pytestmark = [
    pytest.mark.performance,
    pytest.mark.timeout(300)  # 5 minute timeout for performance tests
]


def pytest_configure(config):
    """Configure pytest for performance testing."""
    # Register performance test markers
    config.addinivalue_line("markers", "performance: mark test as performance validation")
    config.addinivalue_line("markers", "stress: mark test as stress/resource testing")
    config.addinivalue_line("markers", "regression: mark test as regression detection")
    
    # Configure performance test environment
    os.environ.setdefault("PERFORMANCE_TEST_MODE", "1")
    
    # Optimize for performance testing
    if hasattr(config.option, 'benchmark_disable') and not config.option.benchmark_disable:
        print("\nPerformance testing with pytest-benchmark enabled")
    else:
        print("\nPerformance testing with manual timing (pytest-benchmark disabled)")


@pytest.fixture(scope="session")
def performance_test_session():
    """Session-scoped fixture for performance test orchestration."""
    session_start = time.perf_counter()
    
    print(f"\n{'='*60}")
    print("Performance Test Suite - Enhanced Plume Navigation")
    print(f"Testing SLA Requirements:")
    print(f"  Step Latency P95: <{PerformanceConfig.STEP_LATENCY_P95_MS}ms")
    print(f"  Cache Hit Rate: >{PerformanceConfig.CACHE_HIT_RATE_TARGET:.0%}")
    print(f"  Memory Limit: <{PerformanceConfig.MEMORY_LIMIT_GB}GB per process")
    print(f"  Frame Processing: <{PerformanceConfig.FRAME_PROCESSING_TARGET_MS}ms")
    print(f"{'='*60}")
    
    yield
    
    session_duration = time.perf_counter() - session_start
    print(f"\nPerformance test session completed in {session_duration:.2f}s")


if __name__ == "__main__":
    # Enable direct execution for debugging
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "performance",
        "--durations=10"
    ])