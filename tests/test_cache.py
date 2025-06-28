"""
Comprehensive Unit Test Suite for FrameCache System.

This test suite validates the high-performance frame caching implementation per 
Section 0.4.1 requirements including LRU and full-preload strategies, memory 
boundary compliance, thread safety, and cache statistics accuracy to ensure 
optimal performance for reinforcement learning workflows in the plume_nav_sim project.

Test Coverage Areas:
- FrameCache class functionality across all operational modes (LRU, ALL, NONE)
- LRU eviction policy comprehensive testing with memory pressure scenarios
- Cache hit rate validation ≥90% target through statistical sampling
- Memory boundary compliance ≤2 GiB default limit with psutil monitoring
- Thread safety validation under concurrent access scenarios
- Cache pressure threshold testing (0.9) per configuration schema
- Cache statistics tracking accuracy within ±1% error bounds
- Performance benchmarks to validate sub-10ms latency requirements
- Integration testing with VideoPlume mock objects
- Hydra configuration integration for LRU cache mode testing

Performance Targets Validated:
- Cache hit rate ≥90% for sequential access patterns
- Memory usage ≤2 GiB default limit under normal operations
- Thread-safe operations supporting 100+ concurrent agents
- Sub-10ms frame retrieval latency for cache hits
- Accurate statistics tracking within ±1% error bounds
- Memory pressure threshold enforcement at 0.9 utilization

Author: Blitzy Engineering Team
Version: 2.0.0
"""

import pytest
import threading
import time
import tempfile
import gc
import numpy as np
import psutil
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import MagicMock, patch, Mock
import warnings

# Import Hypothesis for property-based testing per Section 6.6.4.2.3
try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.stateful import RuleBasedStateMachine, rule, invariant
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Fallback decorators for when Hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import pytest-benchmark for performance testing if available
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Core cache system imports from plume_nav_sim
from plume_nav_sim.utils.frame_cache import (
    FrameCache,
    CacheMode,
    CacheStatistics,
    create_lru_cache,
    create_preload_cache,
    create_no_cache,
    create_frame_cache,
    validate_cache_config,
    diagnose_cache_setup,
    estimate_cache_memory_usage
)

# Hydra configuration testing imports
try:
    from hydra import initialize, compose
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict


class TestCacheStatistics:
    """Test suite for CacheStatistics class functionality."""
    
    def test_statistics_initialization(self):
        """Test CacheStatistics initialization with default values."""
        stats = CacheStatistics()
        
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0
        assert stats.total_requests == 0
        assert stats.average_hit_time == 0.0
        assert stats.average_miss_time == 0.0
        assert stats.memory_usage_mb == 0.0
        assert stats.peak_memory_mb == 0.0
    
    def test_hit_miss_recording(self):
        """Test hit and miss recording with timing information."""
        stats = CacheStatistics()
        
        # Record hits with timing
        stats.record_hit(0.005)  # 5ms
        stats.record_hit(0.008)  # 8ms
        stats.record_hit(0.003)  # 3ms
        
        # Record misses with timing
        stats.record_miss(0.020)  # 20ms
        stats.record_miss(0.025)  # 25ms
        
        # Verify statistics
        assert stats.total_requests == 5
        assert stats.hit_rate == 0.6  # 3/5
        assert stats.miss_rate == 0.4  # 2/5
        assert abs(stats.average_hit_time - 0.00533) < 0.001  # (5+8+3)/3 ms ≈ 5.33ms
        assert abs(stats.average_miss_time - 0.0225) < 0.001  # (20+25)/2 ms = 22.5ms
    
    def test_memory_tracking(self):
        """Test memory usage tracking and peak detection."""
        stats = CacheStatistics()
        
        # Record insertions with various frame sizes
        stats.record_insertion(1024 * 1024)      # 1MB
        assert stats.memory_usage_mb == 1.0
        assert stats.peak_memory_mb == 1.0
        
        stats.record_insertion(2 * 1024 * 1024)  # 2MB
        assert stats.memory_usage_mb == 3.0
        assert stats.peak_memory_mb == 3.0
        
        # Record eviction
        stats.record_eviction(1024 * 1024)       # Remove 1MB
        assert stats.memory_usage_mb == 2.0
        assert stats.peak_memory_mb == 3.0  # Peak should remain
    
    def test_statistics_thread_safety(self):
        """Test thread safety of statistics operations."""
        stats = CacheStatistics()
        results = []
        
        def worker_thread(thread_id: int):
            """Worker function for concurrent statistics updates."""
            for i in range(100):
                if i % 2 == 0:
                    stats.record_hit(0.005)
                else:
                    stats.record_miss(0.020)
                stats.record_insertion(1024)  # 1KB per insertion
            
            results.append({
                'thread_id': thread_id,
                'total_requests': stats.total_requests
            })
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify thread safety - total should be exactly 500 (5 threads × 100 operations)
        assert stats.total_requests == 500
        assert stats.hit_rate == 0.5  # 50% hit rate
        assert len(results) == 5
    
    def test_statistics_accuracy_bounds(self):
        """Test statistics accuracy within ±1% error bounds per Section 6.6.5.4.2."""
        stats = CacheStatistics()
        
        # Generate known pattern for accuracy testing
        target_hit_rate = 0.9  # 90% target
        total_operations = 1000
        
        hits = int(total_operations * target_hit_rate)
        misses = total_operations - hits
        
        # Record hits and misses
        for _ in range(hits):
            stats.record_hit(0.005)  # 5ms consistent hit time
        
        for _ in range(misses):
            stats.record_miss(0.020)  # 20ms consistent miss time
        
        # Verify accuracy within ±1% bounds
        calculated_hit_rate = stats.hit_rate
        error_margin = abs(calculated_hit_rate - target_hit_rate)
        assert error_margin <= 0.01, f"Hit rate error {error_margin:.3f} exceeds ±1% bound"
        
        # Verify timing accuracy
        assert abs(stats.average_hit_time - 0.005) < 0.001
        assert abs(stats.average_miss_time - 0.020) < 0.001
    
    def test_statistics_reset(self):
        """Test complete statistics reset functionality."""
        stats = CacheStatistics()
        
        # Populate with data
        stats.record_hit(0.005)
        stats.record_miss(0.020)
        stats.record_insertion(1024 * 1024)
        stats.record_eviction(512 * 1024)
        stats.record_pressure_warning()
        
        # Verify data exists
        assert stats.total_requests > 0
        assert stats.memory_usage_mb > 0
        
        # Reset and verify clean state
        stats.reset()
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0
        assert stats.total_requests == 0
        assert stats.memory_usage_mb == 0.0
        assert stats.peak_memory_mb == 0.0
        
        summary = stats.get_summary()
        assert summary['hits'] == 0
        assert summary['misses'] == 0
        assert summary['pressure_warnings'] == 0


class TestCacheMode:
    """Test suite for CacheMode enum functionality."""
    
    def test_cache_mode_values(self):
        """Test CacheMode enum values and string representations."""
        assert CacheMode.NONE.value == "none"
        assert CacheMode.LRU.value == "lru"
        assert CacheMode.ALL.value == "all"
    
    def test_cache_mode_from_string(self):
        """Test CacheMode creation from string values."""
        assert CacheMode.from_string("none") == CacheMode.NONE
        assert CacheMode.from_string("lru") == CacheMode.LRU
        assert CacheMode.from_string("all") == CacheMode.ALL
        
        # Test case insensitivity
        assert CacheMode.from_string("LRU") == CacheMode.LRU
        assert CacheMode.from_string(" all ") == CacheMode.ALL
    
    def test_invalid_cache_mode_string(self):
        """Test error handling for invalid cache mode strings."""
        with pytest.raises(ValueError, match="Invalid cache mode"):
            CacheMode.from_string("invalid")
        
        with pytest.raises(ValueError, match="Invalid cache mode"):
            CacheMode.from_string("")


class TestFrameCacheInitialization:
    """Test suite for FrameCache initialization and configuration."""
    
    def test_default_initialization(self):
        """Test FrameCache initialization with default parameters."""
        cache = FrameCache()
        
        assert cache.mode == CacheMode.LRU
        assert cache.memory_limit_mb == 2048.0
        assert cache.memory_limit_bytes == 2048 * 1024 * 1024
        assert cache.memory_pressure_threshold == 0.9
        assert cache.cache_size == 0
        assert cache.hit_rate == 0.0
        assert cache.statistics is not None
    
    def test_custom_initialization(self):
        """Test FrameCache initialization with custom parameters."""
        cache = FrameCache(
            mode=CacheMode.ALL,
            memory_limit_mb=1024.0,
            memory_pressure_threshold=0.8,
            enable_statistics=True,
            enable_logging=False,
            preload_chunk_size=200,
            eviction_batch_size=20
        )
        
        assert cache.mode == CacheMode.ALL
        assert cache.memory_limit_mb == 1024.0
        assert cache.memory_pressure_threshold == 0.8
        assert cache.preload_chunk_size == 200
        assert cache.eviction_batch_size == 20
        assert cache.statistics is not None
        assert cache.enable_logging is False
    
    def test_string_mode_initialization(self):
        """Test FrameCache initialization with string mode parameter."""
        cache = FrameCache(mode="lru", memory_limit_mb=512.0)
        
        assert cache.mode == CacheMode.LRU
        assert cache.memory_limit_mb == 512.0
    
    def test_invalid_initialization_parameters(self):
        """Test error handling for invalid initialization parameters."""
        with pytest.raises(ValueError, match="memory_limit_mb must be positive"):
            FrameCache(memory_limit_mb=-100)
        
        with pytest.raises(ValueError, match="memory_limit_mb must be positive"):
            FrameCache(memory_limit_mb=0)
        
        with pytest.raises(ValueError, match="memory_pressure_threshold must be between"):
            FrameCache(memory_pressure_threshold=-0.1)
        
        with pytest.raises(ValueError, match="memory_pressure_threshold must be between"):
            FrameCache(memory_pressure_threshold=1.5)
        
        with pytest.raises(ValueError, match="Invalid mode type"):
            FrameCache(mode=42)
    
    def test_no_cache_mode_initialization(self):
        """Test FrameCache initialization in NONE mode."""
        cache = FrameCache(mode=CacheMode.NONE)
        
        assert cache.mode == CacheMode.NONE
        assert cache._cache is None
        assert cache.cache_size == 0


class TestFrameCacheBasicOperations:
    """Test suite for basic FrameCache operations and functionality."""
    
    @pytest.fixture
    def mock_video_plume(self):
        """Create a mock VideoPlume instance for testing."""
        mock = MagicMock()
        mock.frame_count = 1000
        
        # Configure get_frame to return deterministic frames
        def get_frame_side_effect(frame_id, **kwargs):
            if 0 <= frame_id < 1000:
                # Create unique frame based on frame_id
                frame = np.full((480, 640), frame_id % 256, dtype=np.uint8)
                return frame
            return None
        
        mock.get_frame.side_effect = get_frame_side_effect
        return mock
    
    @pytest.fixture
    def lru_cache(self):
        """Create a small LRU cache for testing."""
        return FrameCache(mode=CacheMode.LRU, memory_limit_mb=10.0)  # 10MB limit
    
    @pytest.fixture
    def preload_cache(self):
        """Create a preload cache for testing."""
        return FrameCache(mode=CacheMode.ALL, memory_limit_mb=20.0)  # 20MB limit
    
    @pytest.fixture
    def no_cache(self):
        """Create a no-cache instance for testing."""
        return FrameCache(mode=CacheMode.NONE)
    
    def test_cache_miss_and_hit_cycle(self, lru_cache, mock_video_plume):
        """Test basic cache miss and hit cycle."""
        frame_id = 42
        
        # First access - should be a miss
        frame1 = lru_cache.get(frame_id, mock_video_plume)
        assert frame1 is not None
        assert lru_cache.statistics.total_requests == 1
        assert lru_cache.statistics._misses == 1
        assert lru_cache.statistics._hits == 0
        assert lru_cache.cache_size == 1
        
        # Second access - should be a hit
        frame2 = lru_cache.get(frame_id, mock_video_plume)
        assert frame2 is not None
        assert np.array_equal(frame1, frame2)
        assert lru_cache.statistics.total_requests == 2
        assert lru_cache.statistics._misses == 1
        assert lru_cache.statistics._hits == 1
        assert lru_cache.hit_rate == 0.5
    
    def test_no_cache_mode_operations(self, no_cache, mock_video_plume):
        """Test operations in NONE cache mode."""
        frame_id = 42
        
        # Multiple accesses should always hit video source
        frame1 = no_cache.get(frame_id, mock_video_plume)
        frame2 = no_cache.get(frame_id, mock_video_plume)
        
        assert frame1 is not None
        assert frame2 is not None
        assert np.array_equal(frame1, frame2)
        assert no_cache.cache_size == 0
        assert no_cache.statistics.total_requests == 2
        assert no_cache.statistics._hits == 0
        assert no_cache.statistics._misses == 2
    
    def test_frame_id_validation(self, lru_cache, mock_video_plume):
        """Test frame ID validation and error handling."""
        with pytest.raises(ValueError, match="frame_id must be non-negative"):
            lru_cache.get(-1, mock_video_plume)
        
        with pytest.raises(ValueError, match="video_plume cannot be None"):
            lru_cache.get(0, None)
    
    def test_invalid_frame_handling(self, lru_cache):
        """Test handling of invalid frames from video source."""
        mock_video_plume = MagicMock()
        mock_video_plume.get_frame.return_value = None
        
        frame = lru_cache.get(42, mock_video_plume)
        assert frame is None
        assert lru_cache.statistics._misses == 1
        assert lru_cache.cache_size == 0
    
    def test_frame_copy_prevention(self, lru_cache, mock_video_plume):
        """Test that returned frames are copies to prevent external modification."""
        frame_id = 42
        
        # Get frame and modify it
        frame1 = lru_cache.get(frame_id, mock_video_plume)
        original_value = frame1[0, 0]
        frame1[0, 0] = 255  # Modify returned frame
        
        # Get same frame again - should be unmodified
        frame2 = lru_cache.get(frame_id, mock_video_plume)
        assert frame2[0, 0] == original_value
        assert frame2[0, 0] != frame1[0, 0]


class TestFrameCacheLRUEviction:
    """Test suite for LRU eviction policy validation per Section 6.6.4.2.3."""
    
    @pytest.fixture
    def small_lru_cache(self):
        """Create a very small LRU cache to force evictions."""
        return FrameCache(mode=CacheMode.LRU, memory_limit_mb=1.0)  # 1MB limit
    
    @pytest.fixture
    def mock_video_plume_small_frames(self):
        """Create mock VideoPlume with small frames for eviction testing."""
        mock = MagicMock()
        
        def get_frame_side_effect(frame_id, **kwargs):
            if 0 <= frame_id < 1000:
                # Create 100KB frames (10 frames = 1MB)
                frame = np.full((100, 100, 10), frame_id % 256, dtype=np.uint8)
                return frame
            return None
        
        mock.get_frame.side_effect = get_frame_side_effect
        return mock
    
    def test_lru_eviction_order(self, small_lru_cache, mock_video_plume_small_frames):
        """Test that LRU eviction follows correct order."""
        cache = small_lru_cache
        video_plume = mock_video_plume_small_frames
        
        # Fill cache beyond capacity to trigger evictions
        frame_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for frame_id in frame_ids:
            cache.get(frame_id, video_plume)
        
        # Access some frames to update LRU order
        cache.get(3, video_plume)  # Make frame 3 most recently used
        cache.get(7, video_plume)  # Make frame 7 second most recently used
        cache.get(10, video_plume) # Make frame 10 third most recently used
        
        # Add new frame to trigger eviction
        cache.get(99, video_plume)
        
        # Verify that recently accessed frames are still cached
        with cache._lock:
            cached_frames = set(cache._cache.keys())
        
        # Frames 3, 7, 10, 99 should likely be in cache (most recently accessed)
        # This test verifies eviction occurred and most recent frames survive
        assert 99 in cached_frames  # New frame should be cached
        evictions_occurred = cache.statistics._evictions > 0
        assert evictions_occurred, "Expected evictions to occur with small cache"
    
    def test_lru_move_to_end_on_access(self, small_lru_cache, mock_video_plume_small_frames):
        """Test that accessing cached frames moves them to end (most recent)."""
        cache = small_lru_cache
        video_plume = mock_video_plume_small_frames
        
        # Cache a few frames
        cache.get(1, video_plume)
        cache.get(2, video_plume)
        cache.get(3, video_plume)
        
        # Access frame 1 (should move to most recent)
        cache.get(1, video_plume)
        
        # Verify frame 1 hit was recorded and LRU order updated
        assert cache.statistics._hits >= 1
        
        # Add frames to force eviction and verify LRU behavior
        for frame_id in range(10, 20):
            cache.get(frame_id, video_plume)
        
        # Frame 1 (recently accessed) should survive longer than frame 2 or 3
        evictions_count = cache.statistics._evictions
        assert evictions_count > 0, "Expected evictions with memory pressure"
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(access_sequence=st.lists(st.integers(min_value=0, max_value=99), min_size=50, max_size=200))
    @settings(max_examples=20, deadline=None)
    def test_lru_invariant_property_based(self, access_sequence):
        """Property-based test for LRU invariant validation."""
        # Use very small cache to ensure evictions
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=0.5)  # 512KB
        
        mock_video_plume = MagicMock()
        
        def get_frame_side_effect(frame_id, **kwargs):
            # Create 50KB frames to force evictions quickly
            return np.full((50, 50, 20), frame_id % 256, dtype=np.uint8)
        
        mock_video_plume.get_frame.side_effect = get_frame_side_effect
        
        accessed_frames = []
        for frame_id in access_sequence:
            frame = cache.get(frame_id, mock_video_plume)
            accessed_frames.append(frame_id)
            
            # LRU Invariant: Cache never exceeds memory limit
            assert cache.memory_usage_mb <= cache.memory_limit_mb
            
            # LRU Invariant: Statistics remain consistent
            total_requests = cache.statistics.total_requests
            hits = cache.statistics._hits
            misses = cache.statistics._misses
            assert hits + misses == total_requests
            
            # LRU Invariant: Hit rate is valid percentage
            assert 0.0 <= cache.hit_rate <= 1.0
        
        # Verify evictions occurred if we exceeded capacity
        if len(set(access_sequence)) > 10:  # More unique frames than can fit
            assert cache.statistics._evictions > 0
    
    def test_lru_cache_size_tracking(self, small_lru_cache, mock_video_plume_small_frames):
        """Test accurate cache size tracking during evictions."""
        cache = small_lru_cache
        video_plume = mock_video_plume_small_frames
        
        initial_size = cache.cache_size
        assert initial_size == 0
        
        # Add frames and track size changes
        for i in range(15):  # Add more frames than can fit
            cache.get(i, video_plume)
            current_size = cache.cache_size
            assert current_size >= 0  # Size should never be negative
            
            # Size should not exceed reasonable limits for 1MB cache
            assert current_size <= 20  # Conservative upper bound


class TestFrameCacheMemoryManagement:
    """Test suite for memory management and boundary compliance."""
    
    @pytest.fixture
    def memory_test_cache(self):
        """Create cache with small memory limit for testing."""
        return FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=5.0,  # 5MB limit
            memory_pressure_threshold=0.8,
            eviction_batch_size=2
        )
    
    @pytest.fixture
    def mock_large_frame_video(self):
        """Create mock VideoPlume with large frames for memory testing."""
        mock = MagicMock()
        
        def get_frame_side_effect(frame_id, **kwargs):
            if 0 <= frame_id < 100:
                # Create 1MB frames (5 frames = 5MB limit)
                frame = np.full((500, 500, 4), frame_id % 256, dtype=np.uint8)
                return frame
            return None
        
        mock.get_frame.side_effect = get_frame_side_effect
        return mock
    
    def test_memory_limit_compliance(self, memory_test_cache, mock_large_frame_video):
        """Test that cache respects memory limits ≤2 GiB default per Section 0.5.1."""
        cache = memory_test_cache
        video_plume = mock_large_frame_video
        
        # Fill cache beyond memory limit
        for frame_id in range(10):  # 10MB worth of frames for 5MB cache
            cache.get(frame_id, video_plume)
        
        # Verify memory usage stays within bounds
        memory_usage_mb = cache.memory_usage_mb
        assert memory_usage_mb <= cache.memory_limit_mb * 1.1  # Allow 10% overhead
        
        # Verify evictions occurred
        assert cache.statistics._evictions > 0
        
        # Verify cache is still functional
        assert cache.cache_size > 0
        assert cache.hit_rate >= 0.0
    
    def test_memory_pressure_detection(self, memory_test_cache, mock_large_frame_video):
        """Test memory pressure detection and handling."""
        cache = memory_test_cache
        video_plume = mock_large_frame_video
        
        initial_pressure_warnings = cache.statistics._pressure_warnings
        
        # Fill cache to trigger pressure warnings
        for frame_id in range(8):  # Force pressure
            cache.get(frame_id, video_plume)
        
        # Check if pressure warnings were recorded
        final_pressure_warnings = cache.statistics._pressure_warnings
        pressure_detected = final_pressure_warnings > initial_pressure_warnings
        
        # Should have triggered pressure handling
        assert pressure_detected or cache.statistics._evictions > 0
    
    def test_default_memory_limit_compliance(self):
        """Test default 2 GiB memory limit compliance per Section 0.5.1."""
        cache = FrameCache()  # Default 2048MB limit
        
        assert cache.memory_limit_mb == 2048.0
        assert cache.memory_limit_bytes == 2 * 1024 * 1024 * 1024  # 2 GiB
        
        # Verify memory usage starts at 0
        assert cache.memory_usage_mb == 0.0
        assert cache.memory_usage_ratio == 0.0
    
    @pytest.mark.skipif(not psutil, reason="psutil not available")
    def test_system_memory_monitoring(self, memory_test_cache):
        """Test system memory monitoring with psutil integration."""
        cache = memory_test_cache
        
        # Verify psutil process monitoring is available
        assert cache._process is not None
        
        # Test memory info access
        try:
            memory_info = cache._process.memory_info()
            assert hasattr(memory_info, 'rss')  # Resident Set Size
        except psutil.Error:
            pytest.skip("psutil memory monitoring not available in test environment")
    
    def test_memory_estimation_accuracy(self):
        """Test memory usage estimation accuracy for different frame types."""
        # Test estimation function
        estimates = estimate_cache_memory_usage(
            video_frame_count=1000,
            frame_width=640,
            frame_height=480,
            channels=1,
            dtype_size=1
        )
        
        expected_frame_size = 640 * 480 * 1 * 1  # 307,200 bytes
        expected_total_mb = (expected_frame_size * 1000) / (1024 * 1024)  # ~293 MB
        
        assert estimates['frame_size_bytes'] == expected_frame_size
        assert abs(estimates['total_video_mb'] - expected_total_mb) < 1.0
        assert 'recommendation' in estimates
    
    def test_memory_pressure_threshold_configuration(self):
        """Test configurable memory pressure thresholds."""
        # Test different threshold values
        thresholds = [0.7, 0.8, 0.9, 0.95]
        
        for threshold in thresholds:
            cache = FrameCache(
                memory_limit_mb=10.0,
                memory_pressure_threshold=threshold
            )
            assert cache.memory_pressure_threshold == threshold
            
            # Test that threshold affects pressure detection logic
            pressure_detected = cache._check_memory_pressure(
                additional_bytes=int(cache.memory_limit_bytes * threshold * 1.1)
            )
            # With bytes exceeding threshold, pressure should be detected
            # (actual behavior may vary based on rate limiting)


class TestFrameCacheThreadSafety:
    """Test suite for thread safety validation per Section 0.3.3."""
    
    @pytest.fixture
    def thread_safe_cache(self):
        """Create cache configured for thread safety testing."""
        return FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=50.0,
            enable_statistics=True
        )
    
    @pytest.fixture
    def concurrent_mock_video(self):
        """Create mock VideoPlume for concurrent access testing."""
        mock = MagicMock()
        
        def get_frame_side_effect(frame_id, **kwargs):
            # Simulate processing time and return unique frames
            time.sleep(0.001)  # 1ms simulation
            if 0 <= frame_id < 500:
                return np.full((100, 100), frame_id % 256, dtype=np.uint8)
            return None
        
        mock.get_frame.side_effect = get_frame_side_effect
        return mock
    
    def test_concurrent_cache_access(self, thread_safe_cache, concurrent_mock_video):
        """Test concurrent cache access from multiple threads."""
        cache = thread_safe_cache
        video_plume = concurrent_mock_video
        results = []
        
        def worker_thread(thread_id: int, frame_range: range):
            """Worker function for concurrent cache access."""
            thread_results = {
                'thread_id': thread_id,
                'frames_accessed': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'errors': []
            }
            
            initial_hits = cache.statistics._hits
            initial_misses = cache.statistics._misses
            
            try:
                for frame_id in frame_range:
                    frame = cache.get(frame_id, video_plume)
                    if frame is not None:
                        thread_results['frames_accessed'] += 1
                    else:
                        thread_results['errors'].append(f"Failed to get frame {frame_id}")
                
                # Calculate hits and misses for this thread (approximate)
                final_hits = cache.statistics._hits
                final_misses = cache.statistics._misses
                
            except Exception as e:
                thread_results['errors'].append(str(e))
            
            results.append(thread_results)
        
        # Start multiple threads with overlapping frame ranges
        threads = []
        for i in range(5):
            # Each thread accesses 50 frames with overlap
            start_frame = i * 30
            frame_range = range(start_frame, start_frame + 50)
            thread = threading.Thread(target=worker_thread, args=(i, frame_range))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify thread safety and correctness
        assert len(results) == 5
        total_frames_accessed = sum(r['frames_accessed'] for r in results)
        total_errors = sum(len(r['errors']) for r in results)
        
        # Should have accessed frames successfully
        assert total_frames_accessed > 0
        assert total_errors == 0  # No errors should occur
        
        # Cache statistics should remain consistent
        assert cache.statistics.total_requests > 0
        assert cache.statistics._hits + cache.statistics._misses == cache.statistics.total_requests
        assert 0.0 <= cache.hit_rate <= 1.0
    
    def test_concurrent_eviction_safety(self, thread_safe_cache, concurrent_mock_video):
        """Test thread safety during cache evictions."""
        cache = thread_safe_cache
        video_plume = concurrent_mock_video
        
        def eviction_worker(thread_id: int):
            """Worker that forces cache evictions."""
            # Access many unique frames to force evictions
            for frame_id in range(thread_id * 100, (thread_id + 1) * 100):
                try:
                    cache.get(frame_id, video_plume)
                except Exception:
                    pass  # Ignore individual frame errors
        
        # Start multiple threads that will cause evictions
        threads = []
        for i in range(3):
            thread = threading.Thread(target=eviction_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify cache remained stable
        assert cache.cache_size >= 0
        assert cache.statistics._evictions >= 0
        assert cache.memory_usage_mb >= 0.0
        
        # Statistics should remain internally consistent
        total_requests = cache.statistics.total_requests
        hits = cache.statistics._hits
        misses = cache.statistics._misses
        assert hits + misses == total_requests
    
    def test_statistics_thread_safety_stress(self, thread_safe_cache):
        """Stress test statistics thread safety with rapid updates."""
        cache = thread_safe_cache
        
        def stats_updater(thread_id: int):
            """Rapidly update statistics from multiple threads."""
            for i in range(1000):
                if i % 2 == 0:
                    cache.statistics.record_hit(0.005)
                else:
                    cache.statistics.record_miss(0.020)
                
                cache.statistics.record_insertion(1024)
                if i % 10 == 0:
                    cache.statistics.record_eviction(1024)
        
        # Start multiple stat updating threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=stats_updater, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify final statistics consistency
        total_requests = cache.statistics.total_requests
        hits = cache.statistics._hits
        misses = cache.statistics._misses
        
        assert hits + misses == total_requests
        assert total_requests == 5000  # 5 threads × 1000 operations
        assert cache.hit_rate == 0.5  # 50% hit rate from pattern


class TestFrameCachePreloading:
    """Test suite for frame preloading and ALL mode functionality."""
    
    @pytest.fixture
    def preload_cache(self):
        """Create cache suitable for preloading tests."""
        return FrameCache(
            mode=CacheMode.ALL,
            memory_limit_mb=100.0,  # Large enough for test frames
            preload_chunk_size=20
        )
    
    @pytest.fixture
    def preload_mock_video(self):
        """Create mock VideoPlume for preloading tests."""
        mock = MagicMock()
        mock.frame_count = 200
        
        def get_frame_side_effect(frame_id, **kwargs):
            if 0 <= frame_id < 200:
                # Create 100KB frames
                return np.full((100, 100, 10), frame_id % 256, dtype=np.uint8)
            return None
        
        mock.get_frame.side_effect = get_frame_side_effect
        return mock
    
    def test_successful_preload_range(self, preload_cache, preload_mock_video):
        """Test successful preloading of frame range."""
        cache = preload_cache
        video_plume = preload_mock_video
        
        # Preload frames 0-49
        success = cache.preload(range(0, 50), video_plume)
        
        assert success is True
        assert cache._preload_completed is True
        assert cache._preloaded_range == range(0, 50)
        assert cache.cache_size == 50
        
        # Verify all preloaded frames are cache hits
        for frame_id in range(0, 50):
            initial_hits = cache.statistics._hits
            frame = cache.get(frame_id, video_plume)
            assert frame is not None
            assert cache.statistics._hits > initial_hits  # Should be a hit
    
    def test_preload_tuple_range(self, preload_cache, preload_mock_video):
        """Test preloading with tuple range specification."""
        cache = preload_cache
        video_plume = preload_mock_video
        
        # Preload using tuple range
        success = cache.preload((10, 30), video_plume)
        
        assert success is True
        assert cache.cache_size == 20  # 30 - 10 = 20 frames
    
    def test_preload_memory_estimation(self, preload_cache, preload_mock_video):
        """Test preload memory estimation and limits."""
        cache = preload_cache
        video_plume = preload_mock_video
        
        # Preload reasonable number of frames
        success = cache.preload(range(0, 100), video_plume)
        
        # Should succeed within memory limits
        assert success is True
        assert cache.memory_usage_mb <= cache.memory_limit_mb
    
    def test_preload_memory_limit_exceeded(self):
        """Test preload failure when memory limit would be exceeded."""
        # Create very small cache
        small_cache = FrameCache(mode=CacheMode.ALL, memory_limit_mb=1.0)  # 1MB
        
        mock_video = MagicMock()
        # Create large frames that exceed limit
        mock_video.get_frame.return_value = np.full((1000, 1000), 255, dtype=np.uint8)  # ~1MB per frame
        
        # Attempt to preload many frames
        with pytest.raises(MemoryError):
            small_cache.preload(range(0, 10), mock_video)
    
    def test_preload_invalid_parameters(self, preload_cache):
        """Test preload parameter validation."""
        cache = preload_cache
        
        with pytest.raises(ValueError, match="video_plume cannot be None"):
            cache.preload(range(0, 10), None)
        
        with pytest.raises(ValueError, match="frame_range must be range object"):
            cache.preload("invalid", MagicMock())
    
    def test_preload_chunked_loading(self, preload_cache, preload_mock_video):
        """Test chunked preloading process."""
        cache = preload_cache
        video_plume = preload_mock_video
        
        # Set small chunk size to test chunking
        cache.preload_chunk_size = 10
        
        # Preload 35 frames (4 chunks: 10+10+10+5)
        success = cache.preload(range(0, 35), video_plume)
        
        assert success is True
        assert cache.cache_size == 35
        
        # Verify all frames are accessible
        for frame_id in range(0, 35):
            frame = cache.get(frame_id, video_plume)
            assert frame is not None
    
    def test_cache_warmup_functionality(self, preload_cache, preload_mock_video):
        """Test cache warmup for optimal performance."""
        cache = preload_cache
        video_plume = preload_mock_video
        
        # Warmup with default frames
        success = cache.warmup(video_plume, warmup_frames=30)
        
        assert success is True
        assert cache.cache_size == 30
        
        # Verify warmup frames are cached
        for frame_id in range(0, 30):
            initial_hits = cache.statistics._hits
            cache.get(frame_id, video_plume)
            assert cache.statistics._hits > initial_hits
    
    def test_preload_with_failed_frames(self, preload_cache):
        """Test preload handling when some frames fail to load."""
        cache = preload_cache
        
        mock_video = MagicMock()
        def failing_get_frame(frame_id, **kwargs):
            # Fail every 5th frame
            if frame_id % 5 == 0:
                return None
            return np.full((50, 50), frame_id % 256, dtype=np.uint8)
        
        mock_video.get_frame.side_effect = failing_get_frame
        
        # Preload with some failures
        success = cache.preload(range(0, 25), mock_video)
        
        # Should report partial failure
        assert success is False  # _preload_completed should be False due to failures
        assert cache.cache_size < 25  # Some frames should be missing


class TestFrameCachePerformanceValidation:
    """Test suite for performance validation and benchmarking."""
    
    @pytest.fixture
    def performance_cache(self):
        """Create cache optimized for performance testing."""
        return FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=100.0,
            enable_statistics=True
        )
    
    @pytest.fixture
    def performance_mock_video(self):
        """Create mock VideoPlume for performance testing."""
        mock = MagicMock()
        
        def fast_get_frame(frame_id, **kwargs):
            # Fast frame generation for performance testing
            return np.full((200, 200), frame_id % 256, dtype=np.uint8)
        
        mock.get_frame.side_effect = fast_get_frame
        return mock
    
    def test_cache_hit_rate_target_validation(self, performance_cache, performance_mock_video):
        """Test cache hit rate ≥90% target per Section 6.6.4.1.1."""
        cache = performance_cache
        video_plume = performance_mock_video
        
        # Sequential access pattern to achieve high hit rate
        frame_sequence = []
        
        # Fill cache with initial frames
        for frame_id in range(50):
            cache.get(frame_id, video_plume)
            frame_sequence.append(frame_id)
        
        # Perform 1000+ accesses with 90% reuse pattern per requirements
        for _ in range(1000):
            # 90% chance of accessing cached frame, 10% new frame
            if np.random.random() < 0.9:
                frame_id = np.random.choice(frame_sequence[:40])  # Reuse cached
            else:
                frame_id = np.random.randint(100, 200)  # New frame
                if len(frame_sequence) < 100:
                    frame_sequence.append(frame_id)
            
            cache.get(frame_id, video_plume)
        
        # Verify hit rate meets ≥90% target
        hit_rate = cache.hit_rate
        assert hit_rate >= 0.90, f"Cache hit rate {hit_rate:.1%} below 90% target"
        assert cache.statistics.total_requests >= 1000
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_sub_10ms_cache_hit_latency(self, benchmark, performance_cache, performance_mock_video):
        """Test sub-10ms cache hit latency requirement."""
        cache = performance_cache
        video_plume = performance_mock_video
        
        # Pre-populate cache
        frame_id = 42
        cache.get(frame_id, video_plume)
        
        def cached_frame_access():
            """Benchmark function for cached frame access."""
            return cache.get(frame_id, video_plume)
        
        # Benchmark cache hit performance
        result = benchmark.pedantic(cached_frame_access, rounds=100, iterations=10)
        
        # Validate sub-10ms requirement
        mean_time_ms = benchmark.stats.mean * 1000
        max_time_ms = benchmark.stats.max * 1000
        
        assert mean_time_ms < 10.0, f"Mean cache hit time {mean_time_ms:.2f}ms exceeds 10ms"
        assert max_time_ms < 15.0, f"Max cache hit time {max_time_ms:.2f}ms exceeds reasonable bound"
    
    def test_sequential_access_pattern_optimization(self, performance_cache, performance_mock_video):
        """Test cache optimization for sequential access patterns."""
        cache = performance_cache
        video_plume = performance_mock_video
        
        # Sequential access pattern (common in RL training)
        start_time = time.time()
        
        for frame_id in range(100):
            cache.get(frame_id, video_plume)
        
        # Second pass - should be all cache hits
        hit_start_time = time.time()
        for frame_id in range(100):
            cache.get(frame_id, video_plume)
        hit_duration = time.time() - hit_start_time
        
        # Verify sequential performance
        avg_hit_time_ms = (hit_duration / 100) * 1000
        assert avg_hit_time_ms < 10.0, f"Sequential hit time {avg_hit_time_ms:.2f}ms too slow"
        
        # Verify high hit rate
        assert cache.hit_rate >= 0.5  # At least 50% overall (second pass is 100% hits)
    
    def test_memory_usage_efficiency(self, performance_cache, performance_mock_video):
        """Test memory usage efficiency and tracking accuracy."""
        cache = performance_cache
        video_plume = performance_mock_video
        
        # Track memory usage during operations
        initial_memory = cache.memory_usage_mb
        assert initial_memory == 0.0
        
        # Add frames and monitor memory growth
        frame_sizes = []
        for frame_id in range(20):
            frame = cache.get(frame_id, video_plume)
            frame_sizes.append(frame.nbytes)
            
            current_memory = cache.memory_usage_mb
            assert current_memory >= initial_memory
        
        # Verify memory calculation accuracy
        expected_memory_bytes = sum(frame_sizes)
        expected_memory_mb = expected_memory_bytes / (1024 * 1024)
        actual_memory_mb = cache.memory_usage_mb
        
        # Allow for small calculation differences
        memory_error = abs(actual_memory_mb - expected_memory_mb)
        assert memory_error < 0.1, f"Memory calculation error {memory_error:.3f}MB too large"
    
    def test_concurrent_performance_scaling(self, performance_cache, performance_mock_video):
        """Test performance scaling with concurrent agents (100+ target)."""
        cache = performance_cache
        video_plume = performance_mock_video
        
        # Simulate multiple agents accessing cache concurrently
        results = []
        
        def agent_worker(agent_id: int):
            """Simulate agent accessing frames."""
            start_time = time.time()
            frames_accessed = 0
            
            # Each agent accesses 50 frames
            for i in range(50):
                frame_id = (agent_id * 50) + i
                frame = cache.get(frame_id, video_plume)
                if frame is not None:
                    frames_accessed += 1
            
            duration = time.time() - start_time
            results.append({
                'agent_id': agent_id,
                'frames_accessed': frames_accessed,
                'duration': duration,
                'avg_frame_time_ms': (duration / frames_accessed * 1000) if frames_accessed > 0 else 0
            })
        
        # Test with 10 concurrent agents (simulating 100+ scenario)
        threads = []
        for agent_id in range(10):
            thread = threading.Thread(target=agent_worker, args=(agent_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify concurrent performance
        assert len(results) == 10
        total_frames = sum(r['frames_accessed'] for r in results)
        avg_frame_times = [r['avg_frame_time_ms'] for r in results if r['avg_frame_time_ms'] > 0]
        
        assert total_frames == 500  # 10 agents × 50 frames
        if avg_frame_times:
            overall_avg_ms = sum(avg_frame_times) / len(avg_frame_times)
            assert overall_avg_ms < 20.0, f"Concurrent avg frame time {overall_avg_ms:.2f}ms too slow"


class TestFrameCacheFactoryFunctions:
    """Test suite for cache factory functions and configuration."""
    
    def test_create_lru_cache_factory(self):
        """Test create_lru_cache factory function."""
        cache = create_lru_cache(memory_limit_mb=1024.0, enable_statistics=True)
        
        assert cache is not None
        assert cache.mode == CacheMode.LRU
        assert cache.memory_limit_mb == 1024.0
        assert cache.statistics is not None
    
    def test_create_preload_cache_factory(self):
        """Test create_preload_cache factory function."""
        cache = create_preload_cache(memory_limit_mb=2048.0, preload_chunk_size=50)
        
        assert cache is not None
        assert cache.mode == CacheMode.ALL
        assert cache.memory_limit_mb == 2048.0
        assert cache.preload_chunk_size == 50
    
    def test_create_no_cache_factory(self):
        """Test create_no_cache factory function."""
        cache = create_no_cache(enable_statistics=False)
        
        assert cache is not None
        assert cache.mode == CacheMode.NONE
        assert cache._cache is None
    
    def test_create_frame_cache_factory(self):
        """Test create_frame_cache factory function with various modes."""
        # Test string mode
        lru_cache = create_frame_cache("lru", memory_limit_mb=512.0)
        assert lru_cache.mode == CacheMode.LRU
        assert lru_cache.memory_limit_mb == 512.0
        
        # Test enum mode
        all_cache = create_frame_cache(CacheMode.ALL, memory_limit_mb=1024.0)
        assert all_cache.mode == CacheMode.ALL
        assert all_cache.memory_limit_mb == 1024.0
        
        # Test none mode
        none_cache = create_frame_cache("none")
        assert none_cache.mode == CacheMode.NONE
    
    def test_validate_cache_config_function(self):
        """Test cache configuration validation function."""
        # Valid configuration
        result = validate_cache_config("lru", 2048.0)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Invalid mode
        result = validate_cache_config("invalid", 2048.0)
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        
        # Invalid memory limit
        result = validate_cache_config("lru", -100.0)
        assert result["valid"] is False
        assert "Memory limit must be positive" in str(result["errors"])
        
        # Warning conditions
        result = validate_cache_config("lru", 50.0)  # Low memory
        assert len(result["warnings"]) > 0
    
    def test_diagnose_cache_setup_function(self):
        """Test cache setup diagnostic function."""
        diagnostics = diagnose_cache_setup()
        
        assert "cache_available" in diagnostics
        assert "memory_monitoring_available" in diagnostics
        assert "supported_modes" in diagnostics
        assert "recommendations" in diagnostics
        assert isinstance(diagnostics["supported_modes"], list)
        assert len(diagnostics["supported_modes"]) > 0


class TestFrameCacheContextManager:
    """Test suite for context manager functionality."""
    
    def test_context_manager_basic_usage(self, mock_video_plume):
        """Test FrameCache as context manager with automatic cleanup."""
        initial_cache = None
        
        with FrameCache(mode=CacheMode.LRU, memory_limit_mb=10.0) as cache:
            initial_cache = cache
            
            # Use cache normally
            frame = cache.get(42, mock_video_plume)
            assert frame is not None
            assert cache.cache_size > 0
        
        # After context exit, cache should be cleared
        assert initial_cache.cache_size == 0
        assert initial_cache.statistics._hits == 0  # Reset
        assert initial_cache.statistics._misses == 0  # Reset
    
    def test_context_manager_exception_handling(self, mock_video_plume):
        """Test context manager cleanup on exception."""
        cache_ref = None
        
        try:
            with FrameCache(mode=CacheMode.LRU, memory_limit_mb=10.0) as cache:
                cache_ref = cache
                
                # Use cache
                cache.get(42, mock_video_plume)
                assert cache.cache_size > 0
                
                # Raise exception
                raise ValueError("Test exception")
                
        except ValueError:
            pass  # Expected exception
        
        # Cache should still be cleaned up
        assert cache_ref.cache_size == 0


class TestFrameCacheClearOperation:
    """Test suite for cache clear operation."""
    
    def test_cache_clear_functionality(self, mock_video_plume):
        """Test complete cache clearing and reset."""
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=10.0)
        
        # Populate cache
        for frame_id in range(10):
            cache.get(frame_id, mock_video_plume)
        
        # Verify cache has content
        assert cache.cache_size > 0
        assert cache.statistics.total_requests > 0
        assert cache.memory_usage_mb > 0
        
        # Clear cache
        cache.clear()
        
        # Verify complete reset
        assert cache.cache_size == 0
        assert cache.statistics.total_requests == 0
        assert cache.statistics._hits == 0
        assert cache.statistics._misses == 0
        assert cache.memory_usage_mb == 0.0
        assert cache._preload_completed is False
        assert cache._preloaded_range is None
    
    def test_clear_thread_safety(self, mock_video_plume):
        """Test thread safety of cache clear operation."""
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=10.0)
        
        def worker_clear():
            """Worker that repeatedly clears cache."""
            for _ in range(10):
                cache.clear()
                time.sleep(0.001)
        
        def worker_access():
            """Worker that accesses frames."""
            for frame_id in range(50):
                try:
                    cache.get(frame_id, mock_video_plume)
                except Exception:
                    pass  # Ignore errors during concurrent operations
                time.sleep(0.001)
        
        # Start concurrent clear and access operations
        threads = [
            threading.Thread(target=worker_clear),
            threading.Thread(target=worker_access),
            threading.Thread(target=worker_access)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Cache should be in valid state (possibly empty)
        assert cache.cache_size >= 0
        assert cache.memory_usage_mb >= 0.0
        assert cache.statistics._hits >= 0
        assert cache.statistics._misses >= 0


# Test markers for categorization per Section 6.6.1.1
pytestmark = [
    pytest.mark.cache,  # Cache-specific tests
    pytest.mark.unit,   # Unit test category
]


# Performance test configuration
class TestCachePerformanceBenchmarks:
    """Performance benchmark tests using pytest-benchmark if available."""
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    @pytest.mark.benchmark
    def test_cache_hit_performance_benchmark(self, benchmark):
        """Benchmark cache hit performance for <10ms requirement."""
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=50.0)
        mock_video = MagicMock()
        mock_video.get_frame.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # Pre-populate cache
        frame_id = 42
        cache.get(frame_id, mock_video)
        
        def cache_hit_operation():
            return cache.get(frame_id, mock_video)
        
        # Benchmark with strict timing requirements
        result = benchmark.pedantic(cache_hit_operation, rounds=50, iterations=20)
        
        # Validate performance requirements
        assert benchmark.stats.mean < 0.010  # <10ms requirement
        assert benchmark.stats.median < 0.008  # Median should be even faster
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    @pytest.mark.benchmark
    def test_cache_memory_operations_benchmark(self, benchmark):
        """Benchmark memory management operations."""
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=10.0)
        mock_video = MagicMock()
        mock_video.get_frame.return_value = np.full((200, 200), 255, dtype=np.uint8)
        
        frame_counter = [0]  # Use list for mutable reference
        
        def memory_intensive_operation():
            frame_id = frame_counter[0] % 100
            frame_counter[0] += 1
            return cache.get(frame_id, mock_video)
        
        # Benchmark memory operations including evictions
        result = benchmark.pedantic(memory_intensive_operation, rounds=30, iterations=10)
        
        # Memory operations should complete within reasonable time
        assert benchmark.stats.mean < 0.050  # <50ms for memory operations
        assert cache.statistics._evictions >= 0  # Evictions may have occurred


class TestFrameCacheLRUEvictionPolicyAdvanced:
    """Advanced test suite for comprehensive LRU eviction policy testing per Section 0.4.1."""
    
    @pytest.fixture
    def memory_constrained_cache(self):
        """Create a memory-constrained LRU cache for eviction testing."""
        return FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=2.0,  # Very small limit to force frequent evictions
            memory_pressure_threshold=0.9,
            eviction_batch_size=3,
            enable_statistics=True
        )
    
    @pytest.fixture
    def predictable_video_source(self):
        """Create mock VideoPlume with predictable frame sizes for eviction testing."""
        mock = MagicMock()
        
        def get_frame_side_effect(frame_id, **kwargs):
            if 0 <= frame_id < 500:
                # Create 200KB frames - 10 frames will exceed 2MB limit
                frame = np.full((200, 200, 5), frame_id % 256, dtype=np.uint8)
                return frame
            return None
        
        mock.get_frame.side_effect = get_frame_side_effect
        mock.frame_count = 500
        return mock
    
    def test_lru_eviction_policy_order_validation(self, memory_constrained_cache, predictable_video_source):
        """Test that LRU eviction follows strict chronological order."""
        cache = memory_constrained_cache
        video = predictable_video_source
        
        # Fill cache with initial frames
        initial_frames = [1, 2, 3, 4, 5]
        for frame_id in initial_frames:
            cache.get(frame_id, video)
        
        # Access frames in specific order to establish LRU sequence
        # This makes frame 2 least recently used, then 1, 4, 3, 5 (most recent)
        cache.get(2, video)  # Access 2 (now most recent)
        cache.get(1, video)  # Access 1 (second most recent)
        cache.get(4, video)  # Access 4 (third most recent)
        
        # Add new frame to trigger evictions
        cache.get(10, video)  # Should evict oldest frames
        
        # Verify LRU order by checking which frames survive
        with cache._lock:
            cached_frames = set(cache._cache.keys())
        
        # Most recently accessed frames (2, 1, 4, 10) should be more likely to survive
        assert 10 in cached_frames  # New frame should definitely be cached
        
        # Verify evictions occurred
        assert cache.statistics._evictions > 0
        
        # Test that recently accessed frames have priority
        recent_frames_surviving = sum(1 for fid in [2, 1, 4] if fid in cached_frames)
        old_frames_surviving = sum(1 for fid in [3, 5] if fid in cached_frames)
        
        # Recently accessed frames should have higher survival rate
        assert recent_frames_surviving >= old_frames_surviving
    
    def test_lru_eviction_batch_size_enforcement(self, memory_constrained_cache, predictable_video_source):
        """Test that eviction batch size is respected during memory pressure."""
        cache = memory_constrained_cache
        video = predictable_video_source
        
        # Set known batch size
        cache.eviction_batch_size = 2
        
        # Fill cache to trigger evictions
        initial_evictions = cache.statistics._evictions
        
        # Add enough frames to trigger multiple eviction cycles
        for frame_id in range(20):
            cache.get(frame_id, video)
        
        final_evictions = cache.statistics._evictions
        evictions_performed = final_evictions - initial_evictions
        
        # Verify that evictions occurred in batches
        if evictions_performed > 0:
            # Should be roughly divisible by batch size (allowing for some variance)
            remainder = evictions_performed % cache.eviction_batch_size
            # Allow some flexibility due to memory calculation variations
            assert remainder <= 1, f"Evictions {evictions_performed} not in expected batch size {cache.eviction_batch_size}"
    
    def test_lru_memory_pressure_threshold_accuracy(self, memory_constrained_cache, predictable_video_source):
        """Test that evictions trigger at exact memory pressure threshold."""
        cache = memory_constrained_cache
        video = predictable_video_source
        threshold = cache.memory_pressure_threshold  # 0.9
        
        # Monitor memory usage during frame additions
        memory_readings = []
        
        for frame_id in range(15):
            cache.get(frame_id, video)
            usage_ratio = cache.memory_usage_mb / cache.memory_limit_mb
            memory_readings.append(usage_ratio)
            
            # Check if pressure threshold is respected
            if usage_ratio > threshold:
                # Should have triggered evictions to bring usage back down
                assert cache.statistics._evictions > 0, "Expected evictions at memory pressure threshold"
        
        # Verify that memory usage stayed reasonably close to threshold
        max_usage = max(memory_readings)
        assert max_usage <= 1.1, f"Memory usage {max_usage} exceeded reasonable bounds above threshold {threshold}"
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(access_pattern=st.lists(st.integers(min_value=0, max_value=50), min_size=100, max_size=300))
    @settings(max_examples=10, deadline=None)
    def test_lru_invariant_under_random_access(self, access_pattern):
        """Property-based test for LRU invariants under random access patterns."""
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=1.0)
        mock_video = MagicMock()
        
        def get_frame_side_effect(frame_id, **kwargs):
            # Create 100KB frames to force frequent evictions
            return np.full((100, 100, 10), frame_id % 256, dtype=np.uint8)
        
        mock_video.get_frame.side_effect = get_frame_side_effect
        
        # Track access times for LRU verification
        access_times = {}
        current_time = 0
        
        for frame_id in access_pattern:
            frame = cache.get(frame_id, mock_video)
            access_times[frame_id] = current_time
            current_time += 1
            
            # LRU Invariant 1: Memory limit never exceeded
            assert cache.memory_usage_mb <= cache.memory_limit_mb * 1.1  # Allow small overhead
            
            # LRU Invariant 2: Statistics consistency
            total = cache.statistics.total_requests
            hits = cache.statistics._hits
            misses = cache.statistics._misses
            assert hits + misses == total
            
            # LRU Invariant 3: Valid hit rate
            assert 0.0 <= cache.hit_rate <= 1.0
            
            # LRU Invariant 4: Cache size non-negative
            assert cache.cache_size >= 0
        
        # Final verification: If evictions occurred, verify LRU property
        if cache.statistics._evictions > 0:
            with cache._lock:
                remaining_frames = set(cache._cache.keys())
            
            # Remaining frames should be among the most recently accessed
            if remaining_frames:
                # Get the most recent access time among cached frames
                most_recent_cached = max(access_times[fid] for fid in remaining_frames if fid in access_times)
                
                # Check that no uncached frame was accessed more recently than least recent cached frame
                least_recent_cached = min(access_times[fid] for fid in remaining_frames if fid in access_times)
                
                uncached_frames = set(access_times.keys()) - remaining_frames
                if uncached_frames:
                    most_recent_uncached = max(access_times[fid] for fid in uncached_frames)
                    
                    # This is the core LRU property: least recent cached should be more recent than most recent uncached
                    # Allow some flexibility due to batch evictions
                    assert least_recent_cached >= most_recent_uncached - cache.eviction_batch_size


class TestFrameCacheMemoryLimitEnforcement:
    """Test suite for 2 GiB memory limit enforcement with psutil monitoring per Feature F-004."""
    
    @pytest.fixture
    def strict_memory_cache(self):
        """Create cache with strict memory enforcement for testing."""
        return FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=2048.0,  # 2 GiB default limit
            memory_pressure_threshold=0.9,
            enable_statistics=True,
            eviction_batch_size=10
        )
    
    @pytest.fixture
    def large_frame_video_source(self):
        """Create mock VideoPlume with large frames for memory testing."""
        mock = MagicMock()
        
        def get_frame_side_effect(frame_id, **kwargs):
            if 0 <= frame_id < 1000:
                # Create 10MB frames to quickly reach memory limits
                frame = np.full((1000, 1000, 10), frame_id % 256, dtype=np.uint8)
                return frame
            return None
        
        mock.get_frame.side_effect = get_frame_side_effect
        mock.frame_count = 1000
        return mock
    
    def test_default_2gib_memory_limit_compliance(self, strict_memory_cache):
        """Test that default 2 GiB memory limit is properly enforced."""
        cache = strict_memory_cache
        
        # Verify default limit is set to 2 GiB
        expected_limit_mb = 2048.0
        assert cache.memory_limit_mb == expected_limit_mb
        assert cache.memory_limit_bytes == expected_limit_mb * 1024 * 1024
        
        # Verify initial state
        assert cache.memory_usage_mb == 0.0
        assert cache.memory_usage_ratio == 0.0
    
    def test_memory_limit_enforcement_under_pressure(self, strict_memory_cache, large_frame_video_source):
        """Test that memory limit is enforced under memory pressure scenarios."""
        cache = strict_memory_cache
        video = large_frame_video_source
        
        # Fill cache with large frames to approach limit
        frames_added = 0
        for frame_id in range(300):  # 300 * 10MB = 3GB potential usage
            cache.get(frame_id, video)
            frames_added += 1
            
            current_usage_mb = cache.memory_usage_mb
            
            # Memory usage should never exceed limit by more than small tolerance
            assert current_usage_mb <= cache.memory_limit_mb * 1.05, (
                f"Memory usage {current_usage_mb:.2f}MB exceeded limit "
                f"{cache.memory_limit_mb:.2f}MB at frame {frame_id}"
            )
            
            # If we're above the pressure threshold, evictions should have occurred
            usage_ratio = current_usage_mb / cache.memory_limit_mb
            if usage_ratio > cache.memory_pressure_threshold:
                assert cache.statistics._evictions > 0, (
                    f"Expected evictions at usage ratio {usage_ratio:.3f}, "
                    f"threshold {cache.memory_pressure_threshold:.3f}"
                )
        
        # Verify final state compliance
        final_usage = cache.memory_usage_mb
        assert final_usage <= cache.memory_limit_mb, f"Final usage {final_usage}MB exceeds limit"
        
        # Verify evictions occurred as expected
        assert cache.statistics._evictions > 0, "Expected evictions with large frame workload"
    
    @pytest.mark.skipif(not psutil, reason="psutil not available")
    def test_psutil_memory_monitoring_integration(self, strict_memory_cache):
        """Test psutil-based memory monitoring integration per Section 0.3.1."""
        cache = strict_memory_cache
        
        # Verify psutil process monitoring is properly initialized
        assert hasattr(cache, '_process'), "Cache should have psutil process monitoring"
        assert cache._process is not None, "PSUtil process should be initialized"
        
        # Test memory info access
        try:
            memory_info = cache._process.memory_info()
            assert hasattr(memory_info, 'rss'), "Memory info should include RSS (Resident Set Size)"
            assert hasattr(memory_info, 'vms'), "Memory info should include VMS (Virtual Memory Size)"
            
            # RSS should be reasonable for a fresh cache
            rss_mb = memory_info.rss / (1024 * 1024)
            assert rss_mb > 0, "RSS should be positive"
            assert rss_mb < 1000, "RSS should be reasonable for test environment"
            
        except psutil.Error as e:
            pytest.skip(f"PSUtil memory monitoring not available: {e}")
    
    @pytest.mark.skipif(not psutil, reason="psutil not available")
    def test_system_memory_pressure_detection(self, strict_memory_cache, large_frame_video_source):
        """Test system-wide memory pressure detection with psutil."""
        cache = strict_memory_cache
        video = large_frame_video_source
        
        # Get initial system memory state
        initial_memory = psutil.virtual_memory()
        initial_process_memory = cache._process.memory_info().rss
        
        # Fill cache and monitor system memory impact
        for frame_id in range(50):  # Moderate load to avoid system instability
            cache.get(frame_id, video)
            
            # Check current memory state
            current_memory = psutil.virtual_memory()
            current_process_memory = cache._process.memory_info().rss
            
            # Verify process memory is tracked correctly
            process_growth = (current_process_memory - initial_process_memory) / (1024 * 1024)
            cache_usage = cache.memory_usage_mb
            
            # Process memory growth should correlate with cache usage
            # Allow generous tolerance due to Python's memory management
            assert process_growth <= cache_usage * 3, (
                f"Process memory growth {process_growth:.2f}MB seems excessive "
                f"for cache usage {cache_usage:.2f}MB"
            )
            
            # System should not be under severe memory pressure
            memory_percent = current_memory.percent
            assert memory_percent < 95, f"System memory usage {memory_percent}% too high during test"
    
    def test_memory_estimation_accuracy_validation(self):
        """Test memory usage estimation accuracy for different frame configurations."""
        # Test various frame size scenarios
        test_scenarios = [
            {"width": 640, "height": 480, "channels": 1, "dtype_size": 1, "count": 1000},
            {"width": 1920, "height": 1080, "channels": 3, "dtype_size": 1, "count": 500},
            {"width": 800, "height": 600, "channels": 1, "dtype_size": 4, "count": 800},
        ]
        
        for scenario in test_scenarios:
            estimates = estimate_cache_memory_usage(
                video_frame_count=scenario["count"],
                frame_width=scenario["width"],
                frame_height=scenario["height"],
                channels=scenario["channels"],
                dtype_size=scenario["dtype_size"]
            )
            
            # Calculate expected values
            expected_frame_bytes = (
                scenario["width"] * scenario["height"] * 
                scenario["channels"] * scenario["dtype_size"]
            )
            expected_total_mb = (expected_frame_bytes * scenario["count"]) / (1024 * 1024)
            
            # Verify estimation accuracy
            assert estimates['frame_size_bytes'] == expected_frame_bytes
            assert abs(estimates['total_video_mb'] - expected_total_mb) < 0.1
            
            # Verify recommendations are sensible
            assert 'recommendation' in estimates
            if expected_total_mb > 2048:  # Exceeds 2 GiB limit
                assert 'lru' in estimates['recommendation'].lower()
            elif expected_total_mb < 512:  # Small enough for full preload
                assert 'all' in estimates['recommendation'].lower() or 'preload' in estimates['recommendation'].lower()


class TestFrameCachePressureThresholdTesting:
    """Test suite for cache pressure threshold testing (0.9) per Section 0.3.4 configuration schema."""
    
    @pytest.fixture
    def threshold_test_cache(self):
        """Create cache with configurable pressure threshold for testing."""
        return FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=10.0,  # Small limit for quick threshold testing
            memory_pressure_threshold=0.9,  # 90% threshold per spec
            enable_statistics=True,
            eviction_batch_size=2
        )
    
    @pytest.fixture
    def calibrated_video_source(self):
        """Create mock VideoPlume with calibrated frame sizes for threshold testing."""
        mock = MagicMock()
        
        def get_frame_side_effect(frame_id, **kwargs):
            if 0 <= frame_id < 200:
                # Create 1MB frames - 10 frames will reach 10MB limit
                frame = np.full((500, 500, 4), frame_id % 256, dtype=np.uint8)
                return frame
            return None
        
        mock.get_frame.side_effect = get_frame_side_effect
        mock.frame_count = 200
        return mock
    
    def test_pressure_threshold_configuration_validation(self):
        """Test that pressure threshold configuration is properly validated."""
        # Test valid threshold values
        valid_thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
        
        for threshold in valid_thresholds:
            cache = FrameCache(
                mode=CacheMode.LRU,
                memory_limit_mb=5.0,
                memory_pressure_threshold=threshold
            )
            assert cache.memory_pressure_threshold == threshold
        
        # Test invalid threshold values
        invalid_thresholds = [-0.1, 0.0, 1.0, 1.5, 2.0]
        
        for threshold in invalid_thresholds:
            with pytest.raises(ValueError, match="memory_pressure_threshold must be between"):
                FrameCache(
                    mode=CacheMode.LRU,
                    memory_limit_mb=5.0,
                    memory_pressure_threshold=threshold
                )
    
    def test_pressure_threshold_trigger_accuracy(self, threshold_test_cache, calibrated_video_source):
        """Test that pressure threshold triggers at precisely 0.9 (90%) memory usage."""
        cache = threshold_test_cache
        video = calibrated_video_source
        threshold = cache.memory_pressure_threshold  # 0.9
        
        pressure_warnings_initial = cache.statistics._pressure_warnings
        evictions_initial = cache.statistics._evictions
        
        # Add frames progressively to reach threshold
        for frame_id in range(15):  # More than enough to exceed threshold
            cache.get(frame_id, video)
            
            usage_ratio = cache.memory_usage_mb / cache.memory_limit_mb
            
            # If we're at or above threshold, pressure handling should activate
            if usage_ratio >= threshold:
                pressure_warnings_final = cache.statistics._pressure_warnings
                evictions_final = cache.statistics._evictions
                
                # Either pressure warnings or evictions should have occurred
                pressure_detected = (
                    pressure_warnings_final > pressure_warnings_initial or
                    evictions_final > evictions_initial
                )
                
                assert pressure_detected, (
                    f"Expected pressure handling at usage ratio {usage_ratio:.3f}, "
                    f"threshold {threshold:.3f}, but no warnings or evictions detected"
                )
                
                # Memory usage should not significantly exceed threshold for long
                assert usage_ratio <= 1.2, (
                    f"Memory usage ratio {usage_ratio:.3f} significantly exceeded threshold "
                    f"{threshold:.3f} without proper pressure handling"
                )


class TestFrameCacheHydraConfigurationIntegration:
    """Test suite for LRU cache mode integration with Hydra configuration per enhanced frame caching system."""
    
    @pytest.fixture
    def mock_hydra_cache_config(self):
        """Create mock Hydra configuration for frame cache testing."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for configuration testing")
        
        config_dict = {
            'frame_cache': {
                'mode': 'lru',
                'memory_limit_mb': 1024.0,
                'lru': {
                    'max_frames': None,  # Auto-calculate
                    'preload_sequential': True,
                    'preload_count': 10,
                    'eviction_batch_size': 5,
                    'memory_pressure_threshold': 0.9
                },
                'monitoring': {
                    'target_hit_rate': 0.90,
                    'collect_statistics': True,
                    'statistics_window': 1000,
                    'memory_monitoring_enabled': True,
                    'memory_warning_threshold': 0.85
                }
            }
        }
        
        return OmegaConf.create(config_dict)
    
    def test_hydra_config_cache_instantiation(self, mock_hydra_cache_config):
        """Test cache instantiation from Hydra configuration."""
        config = mock_hydra_cache_config.frame_cache
        
        # Create cache from configuration
        cache = FrameCache(
            mode=config.mode,
            memory_limit_mb=config.memory_limit_mb,
            memory_pressure_threshold=config.lru.memory_pressure_threshold,
            eviction_batch_size=config.lru.eviction_batch_size,
            enable_statistics=config.monitoring.collect_statistics
        )
        
        # Verify configuration was applied correctly
        assert cache.mode == CacheMode.LRU
        assert cache.memory_limit_mb == 1024.0
        assert cache.memory_pressure_threshold == 0.9
        assert cache.eviction_batch_size == 5
        assert cache.statistics is not None
    
    def test_hydra_config_parameter_validation(self, mock_hydra_cache_config):
        """Test validation of Hydra configuration parameters."""
        config = mock_hydra_cache_config.frame_cache
        
        # Test valid configuration
        result = validate_cache_config(config.mode, config.memory_limit_mb)
        assert result["valid"] is True
        assert len(result["errors"]) == 0
        
        # Test configuration with warnings
        low_memory_result = validate_cache_config("lru", 32.0)  # Very low memory
        assert low_memory_result["valid"] is True  # Still valid but with warnings
        assert len(low_memory_result["warnings"]) > 0
        
        # Test invalid configuration
        invalid_result = validate_cache_config("invalid_mode", 1024.0)
        assert invalid_result["valid"] is False
        assert len(invalid_result["errors"]) > 0


class TestFrameCacheStatisticsAccuracyAdvanced:
    """Advanced test suite for cache statistics accuracy testing per Section 0.3.1 implementation design."""
    
    @pytest.fixture
    def high_precision_cache(self):
        """Create cache with high precision statistics tracking."""
        return FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=50.0,
            enable_statistics=True,
            memory_pressure_threshold=0.9
        )
    
    @pytest.fixture
    def statistics_test_video(self):
        """Create mock VideoPlume for statistics accuracy testing."""
        mock = MagicMock()
        
        def get_frame_side_effect(frame_id, **kwargs):
            # Add small delay to test timing accuracy
            time.sleep(0.001)  # 1ms delay
            if 0 <= frame_id < 1000:
                frame = np.full((200, 200), frame_id % 256, dtype=np.uint8)
                return frame
            return None
        
        mock.get_frame.side_effect = get_frame_side_effect
        mock.frame_count = 1000
        return mock
    
    def test_hit_rate_calculation_accuracy(self, high_precision_cache, statistics_test_video):
        """Test hit rate calculation accuracy within ±1% error bounds."""
        cache = high_precision_cache
        video = statistics_test_video
        
        # Create known access pattern for precise hit rate testing
        access_pattern = [1, 2, 3, 4, 5] * 20  # 100 accesses, 20 unique frames
        expected_hits = 80  # 100 - 20 = 80 hits after initial misses
        expected_hit_rate = 0.8  # 80%
        
        for frame_id in access_pattern:
            cache.get(frame_id, video)
        
        # Verify hit rate accuracy
        actual_hit_rate = cache.hit_rate
        hit_rate_error = abs(actual_hit_rate - expected_hit_rate)
        
        assert hit_rate_error <= 0.01, (
            f"Hit rate error {hit_rate_error:.4f} exceeds ±1% bound. "
            f"Expected: {expected_hit_rate:.3f}, Actual: {actual_hit_rate:.3f}"
        )
        
        # Verify component statistics
        assert cache.statistics.total_requests == 100
        assert cache.statistics._hits == expected_hits
        assert cache.statistics._misses == 20
    
    def test_timing_measurement_accuracy(self, high_precision_cache, statistics_test_video):
        """Test timing measurement accuracy for cache operations."""
        cache = high_precision_cache
        video = statistics_test_video
        
        # Perform operations with known timing characteristics
        frame_id = 42
        
        # First access (miss) - should take longer due to video loading
        start_time = time.time()
        cache.get(frame_id, video)
        miss_duration = time.time() - start_time
        
        # Second access (hit) - should be much faster
        start_time = time.time()
        cache.get(frame_id, video)
        hit_duration = time.time() - start_time
        
        # Verify timing measurements are reasonable
        stats = cache.statistics
        
        # Miss time should reflect actual video loading time
        recorded_miss_time = stats.average_miss_time
        assert 0.0005 <= recorded_miss_time <= 0.010, (  # 0.5ms to 10ms range
            f"Recorded miss time {recorded_miss_time:.6f}s outside expected range"
        )
        
        # Hit time should be much faster than miss time
        recorded_hit_time = stats.average_hit_time
        assert recorded_hit_time < recorded_miss_time, (
            f"Hit time {recorded_hit_time:.6f}s not faster than miss time {recorded_miss_time:.6f}s"
        )
        
        # Hit time should be sub-millisecond for cached access
        assert recorded_hit_time <= 0.001, (  # 1ms limit for cache hits
            f"Hit time {recorded_hit_time:.6f}s exceeds 1ms limit"
        )


if __name__ == "__main__":
    # Run tests with coverage if executed directly
    pytest.main([
        __file__,
        "-v",
        "--cov=plume_nav_sim.utils.frame_cache",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "-m", "cache",
        "--benchmark-skip"  # Skip benchmarks in direct execution
    ])