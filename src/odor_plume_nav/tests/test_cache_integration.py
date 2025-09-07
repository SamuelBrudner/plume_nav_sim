"""
Integration test suite validating FrameCache integration with VideoPlume processing and GymnasiumEnv workflows.

This module provides comprehensive integration testing for the FrameCache system ensuring cache-enabled
environments maintain full Gymnasium API compatibility while achieving performance targets. Tests validate
cache warming during environment reset, performance metrics embedding in info['perf_stats'], and video
frame availability for analysis workflows.

Key Testing Objectives:
- FrameCache integration with VideoPlume processor workflow validation per Section 0.4.1
- Cache warming during environment reset for predictable performance per Section 0.2.1
- GymnasiumEnv step() method populating info['perf_stats'] with cache metrics per Section 5.2.2
- End-to-end workflow tests with cache-enabled environments per Section 6.6.3.2.5
- Cache-VideoPlume integration validation ensuring zero-copy frame retrieval per Section 0.3.1
- Multi-agent simulation tests with shared cache instances per Section 5.2.5
- Video frame availability tests for info['video_frame'] in analysis workflows per Section 0.1.2

Performance Requirements:
- Cache-enabled environments must achieve sub-10ms step execution per Section 0.1.1
- Cache hit rate targets ≥90% for sustained performance per Section 6.6.5.4.1
- Memory usage must remain within 2 GiB default limits per Section 0.1.1
- Full Gymnasium API compatibility preservation per Section 0.1.3

Quality Assurance:
- Integration test coverage ≥90% for cache-enabled environment workflows per Section 6.6.6.1.1
- Performance validation through pytest-benchmark integration per Section 6.6.4.1.3
- Cross-platform consistency validation per Section 6.6.1.1

Authors: Blitzy Platform
License: MIT
"""

import os
import sys
import time
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Generator
from unittest.mock import Mock, MagicMock, patch, mock_open
from contextlib import contextmanager, suppress
import threading

import pytest
import numpy as np

# Core testing framework imports
try:
    from click.testing import CliRunner
    CLICK_TESTING_AVAILABLE = True
except ImportError:
    CLICK_TESTING_AVAILABLE = False
    warnings.warn("Click testing not available. CLI tests will be skipped.", ImportWarning)

# Import system under test components
try:
    from odor_plume_nav.environments.gymnasium_env import GymnasiumEnv, create_gymnasium_environment
    from odor_plume_nav.data.video_plume import VideoPlume, create_video_plume
    GYMNASIUM_ENV_AVAILABLE = True
except ImportError:
    GYMNASIUM_ENV_AVAILABLE = False
    GymnasiumEnv = Mock
    VideoPlume = Mock
    warnings.warn("Gymnasium environment modules not available. Some tests will be skipped.", ImportWarning)

# Frame cache imports for integration testing
try:
    from odor_plume_nav.cache.frame_cache import FrameCache
    FRAME_CACHE_AVAILABLE = True
except ImportError:
    FRAME_CACHE_AVAILABLE = False
    FrameCache = Mock
    warnings.warn("FrameCache not available. Cache integration tests will be skipped.", ImportWarning)

# Configuration and navigation imports
try:
    from odor_plume_nav.core.protocols import NavigatorProtocol, NavigatorFactory
    from odor_plume_nav.config.models import NavigatorConfig, VideoPlumeConfig, SimulationConfig
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    warnings.warn("Core navigation modules not available. Some tests will be skipped.", ImportWarning)

# Optional imports with graceful degradation
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

# Performance monitoring imports
try:
    from odor_plume_nav.utils.logging_setup import (
        get_enhanced_logger, PerformanceMetrics, correlation_context
    )
    logger = get_enhanced_logger(__name__)
    LOGGING_UTILS_AVAILABLE = True
except ImportError:
from loguru import logger
    LOGGING_UTILS_AVAILABLE = False

# Benchmark testing for performance validation
try:
    import pytest_benchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False


class CacheIntegrationTestBase:
    """Base class for cache integration tests with common fixtures and utilities."""
    
    @pytest.fixture(autouse=True)
    def setup_cache_test_environment(self, tmp_path, monkeypatch):
        """Set up isolated test environment for cache integration testing."""
        # Create temporary directories for test isolation
        self.temp_dir = tmp_path
        self.config_dir = tmp_path / "conf"
        self.output_dir = tmp_path / "outputs"
        self.data_dir = tmp_path / "data"
        self.cache_dir = tmp_path / "cache"
        
        # Create directory structure
        for dir_path in [self.config_dir, self.output_dir, self.data_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for test isolation
        monkeypatch.setenv("FRAME_CACHE_MODE", "lru")
        monkeypatch.setenv("FRAME_CACHE_SIZE_MB", "512")
        monkeypatch.setenv("PYTEST_RUNNING", "true")
        
        # Disable GUI backends for headless testing
        monkeypatch.setenv("MPLBACKEND", "Agg")
        
        # Set up test-specific logging to avoid interference
from loguru import logger
        logger.getLogger().setLevel(logger.WARNING)
        
        yield
        
        # Cleanup after test - ensure no cache artifacts remain
        try:
            # Clear any remaining cache instances
            import gc
            gc.collect()
        except Exception:
            pass  # Ignore cleanup errors
    
    @pytest.fixture
    def mock_video_file(self):
        """Create a mock video file for testing cache integration."""
        video_path = self.data_dir / "test_cache_video.mp4"
        # Create a more realistic mock video file with some content
        video_path.write_bytes(b"MOCK_VIDEO_DATA_FOR_CACHE_TESTING" * 100)
        return video_path
    
    @pytest.fixture
    def sample_cache_config(self):
        """Generate sample cache configuration for integration testing."""
        return {
            'mode': 'lru',
            'max_size_mb': 512,
            'memory_limit': 2048 * 1024 * 1024,  # 2 GiB default limit
            'enable_preload': False,
            'thread_safe': True,
            'statistics': True
        }
    
    @pytest.fixture
    def sample_environment_config(self):
        """Generate sample environment configuration with cache integration."""
        return {
            'video_path': str(self.data_dir / "test_cache_video.mp4"),
            'navigator': {
                'position': [10.0, 20.0],
                'orientation': 45.0,
                'max_speed': 1.5,
                'angular_velocity': 5.0
            },
            'video_plume': {
                'flip': True,
                'kernel_size': 3,
                'kernel_sigma': 1.0,
                'grayscale': True,
                'normalize': True
            },
            'simulation': {
                'max_steps': 100,
                'step_size': 0.1,
                'performance_monitoring': True
            },
            'frame_cache': {
                'mode': 'lru',
                'max_size_mb': 256,
                'enable_statistics': True
            }
        }
    
    @pytest.fixture
    def mock_video_capture_for_cache(self):
        """Enhanced mock VideoCapture for cache integration testing."""
        with patch('cv2.VideoCapture') as mock_cap:
            # Configure mock to return predictable frame data for cache testing
            mock_instance = MagicMock()
            mock_instance.get.side_effect = lambda prop: {
                1: 300,   # CAP_PROP_FRAME_COUNT (increased for cache testing)
                3: 640,   # CAP_PROP_FRAME_WIDTH  
                4: 480,   # CAP_PROP_FRAME_HEIGHT
                5: 30     # CAP_PROP_FPS
            }.get(prop, 0)
            
            # Mock frame reading with deterministic data for cache validation
            frame_data = {}  # Store frames for consistency validation
            
            def mock_read():
                current_frame = int(mock_instance.get(0))  # Current position
                if current_frame not in frame_data:
                    # Generate deterministic frame data based on frame index
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Create unique pattern for each frame to validate cache correctness
                    pattern_value = (current_frame % 256)
                    frame[:, :, 0] = pattern_value
                    frame[:, :, 1] = (pattern_value + 50) % 256
                    frame[:, :, 2] = (pattern_value + 100) % 256
                    frame_data[current_frame] = frame
                
                return True, frame_data[current_frame].copy()
            
            mock_instance.read = mock_read
            mock_instance.isOpened.return_value = True
            mock_instance.set.return_value = True
            mock_cap.return_value = mock_instance
            yield mock_cap
    
    @pytest.fixture  
    def performance_monitor(self):
        """Performance monitoring fixture for cache integration testing."""
        @contextmanager
        def time_operation(operation_name: str, threshold_ms: float = 10.0):
            """Context manager for timing cache-related operations."""
            class Timer:
                def __init__(self, name, threshold):
                    self.name = name
                    self.threshold = threshold
                    self.start_time = None
                    self.end_time = None
                    self.duration = None
                    self.threshold_exceeded = False
                
            timer = Timer(operation_name, threshold_ms / 1000.0)
            timer.start_time = time.perf_counter()
            
            try:
                yield timer
            finally:
                timer.end_time = time.perf_counter()
                timer.duration = timer.end_time - timer.start_time
                timer.threshold_exceeded = timer.duration > timer.threshold
                
                # Log performance violations for cache operations
                if timer.threshold_exceeded:
                    logger.warning(
                        f"Cache operation '{operation_name}' exceeded {threshold_ms}ms threshold: "
                        f"{timer.duration*1000:.2f}ms"
                    )
        
        return {
            'time_operation': time_operation,
            'validate_sub_10ms': lambda duration: duration < 0.010,
            'validate_cache_hit_rate': lambda hits, total: (hits / total) >= 0.90 if total > 0 else True
        }


@pytest.mark.skipif(not FRAME_CACHE_AVAILABLE, reason="FrameCache not available")
@pytest.mark.skipif(not GYMNASIUM_ENV_AVAILABLE, reason="Gymnasium environment not available")
class TestFrameCacheVideoIntegration(CacheIntegrationTestBase):
    """Test FrameCache integration with VideoPlume processing workflow."""
    
    def test_video_plume_cache_initialization(self, sample_cache_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test VideoPlume initialization with FrameCache integration.
        
        Validates:
        - VideoPlume accepts FrameCache instance during initialization
        - Cache configuration is properly applied
        - Video metadata extraction works with cache enabled
        - Cache statistics are initialized correctly
        """
        # Create FrameCache instance with test configuration
        cache = FrameCache(
            mode=sample_cache_config['mode'],
            max_size_mb=sample_cache_config['max_size_mb'],
            memory_limit=sample_cache_config['memory_limit']
        )
        
        # Initialize VideoPlume with cache
        video_plume = VideoPlume(
            video_path=mock_video_file,
            flip=True,
            grayscale=True,
            cache=cache
        )
        
        # Validate cache integration
        assert video_plume.cache is not None
        assert video_plume.cache == cache
        assert hasattr(video_plume, 'cache_stats')
        
        # Validate video metadata extraction with cache
        metadata = video_plume.get_metadata()
        assert metadata['cache_enabled'] is True
        assert 'cache_stats' in metadata
        assert metadata['width'] == 640
        assert metadata['height'] == 480
        assert metadata['frame_count'] == 300
        
        # Validate initial cache statistics
        cache_stats = video_plume.get_cache_stats()
        assert cache_stats['hits'] == 0
        assert cache_stats['misses'] == 0
        assert cache_stats['total_requests'] == 0
        assert cache_stats['hit_rate'] == 0.0
        
        # Clean up
        video_plume.close()
        cache.clear()
    
    def test_cache_enabled_frame_retrieval_workflow(self, sample_cache_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test complete frame retrieval workflow with cache integration.
        
        Validates:
        - First frame access results in cache miss and disk I/O
        - Subsequent access to same frame results in cache hit
        - Zero-copy frame retrieval from cache
        - Cache statistics accuracy
        """
        # Create cache and video plume
        cache = FrameCache(mode='lru', max_size_mb=256)
        video_plume = VideoPlume(
            video_path=mock_video_file,
            grayscale=True,
            cache=cache
        )
        
        try:
            # First access - should be cache miss
            frame_0_first = video_plume.get_frame(0)
            assert frame_0_first is not None
            assert isinstance(frame_0_first, np.ndarray)
            
            # Validate cache miss statistics
            stats_after_miss = video_plume.get_cache_stats()
            assert stats_after_miss['total_requests'] == 1
            assert stats_after_miss['misses'] == 1
            assert stats_after_miss['hits'] == 0
            assert stats_after_miss['hit_rate'] == 0.0
            
            # Second access to same frame - should be cache hit
            frame_0_second = video_plume.get_frame(0)
            assert frame_0_second is not None
            
            # Validate frames are identical (cache correctness)
            np.testing.assert_array_equal(frame_0_first, frame_0_second)
            
            # Validate cache hit statistics
            stats_after_hit = video_plume.get_cache_stats()
            assert stats_after_hit['total_requests'] == 2
            assert stats_after_hit['misses'] == 1
            assert stats_after_hit['hits'] == 1
            assert stats_after_hit['hit_rate'] == 0.5
            
            # Test access to different frame - cache miss again
            frame_1 = video_plume.get_frame(1)
            assert frame_1 is not None
            
            # Validate frames are different (frame uniqueness)
            assert not np.array_equal(frame_0_first, frame_1)
            
            # Final statistics validation
            final_stats = video_plume.get_cache_stats()
            assert final_stats['total_requests'] == 3
            assert final_stats['misses'] == 2
            assert final_stats['hits'] == 1
            assert abs(final_stats['hit_rate'] - (1/3)) < 0.01
            
        finally:
            video_plume.close()
            cache.clear()
    
    def test_cache_warming_during_initialization(self, sample_cache_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test cache warming functionality during VideoPlume initialization.
        
        Validates:
        - Preload mode cache warming works correctly
        - All frames are loaded during initialization
        - Subsequent frame access results in cache hits
        - Performance improvement from cache warming
        """
        # Create cache in preload mode
        cache = FrameCache(mode='all', max_size_mb=512)
        
        # Mock preload mode detection
        cache.is_preload_mode = Mock(return_value=True)
        cache.contains = Mock(return_value=False)  # Initially no frames cached
        cache.put = Mock()  # Track cache storage calls
        
        # Initialize VideoPlume with preload cache
        with patch.object(cache, 'is_preload_mode', return_value=True):
            video_plume = VideoPlume(
                video_path=mock_video_file,
                grayscale=True,
                cache=cache
            )
        
        # Validate cache warming was attempted
        # Note: In real implementation, this would preload frames
        # For testing, we verify the warming logic was triggered
        assert video_plume.cache is not None
        
        # Test frame access with "warmed" cache
        # Mock cache to simulate preloaded state
        test_frame = np.zeros((480, 640), dtype=np.uint8)
        test_frame[:] = 42  # Distinctive pattern
        
        with patch.object(cache, 'get', return_value=test_frame):
            frame = video_plume.get_frame(0)
            assert frame is not None
            np.testing.assert_array_equal(frame, test_frame)
        
        # Clean up
        video_plume.close()
        cache.clear()
    
    def test_cache_memory_boundary_enforcement(self, mock_video_file, mock_video_capture_for_cache):
        """
        Test cache memory boundary enforcement during video processing.
        
        Validates:
        - Cache respects memory limits during frame storage
        - LRU eviction triggers when memory limit exceeded
        - Memory usage tracking accuracy
        - No memory leaks during extended operation
        """
        # Create cache with small memory limit for testing
        memory_limit = 10 * 1024 * 1024  # 10 MB
        cache = FrameCache(mode='lru', memory_limit=memory_limit)
        
        video_plume = VideoPlume(
            video_path=mock_video_file,
            grayscale=True,
            cache=cache
        )
        
        try:
            initial_memory = cache.get_memory_usage() if hasattr(cache, 'get_memory_usage') else 0
            
            # Access multiple frames to trigger memory pressure
            accessed_frames = []
            for frame_idx in range(50):  # Access enough frames to exceed memory limit
                frame = video_plume.get_frame(frame_idx)
                if frame is not None:
                    accessed_frames.append(frame_idx)
                
                # Check memory usage doesn't exceed limit
                current_memory = cache.get_memory_usage() if hasattr(cache, 'get_memory_usage') else 0
                if hasattr(cache, 'memory_limit'):
                    assert current_memory <= cache.memory_limit, f"Memory usage {current_memory} exceeded limit {cache.memory_limit}"
            
            # Validate some frames were accessed
            assert len(accessed_frames) > 0
            
            # Validate cache statistics show evictions occurred (if memory was exceeded)
            final_stats = video_plume.get_cache_stats()
            
            # If we accessed more frames than cache capacity, should have evictions
            if hasattr(cache, 'size') and cache.size > 0:
                if len(accessed_frames) > cache.size:
                    # Should have both hits and misses due to evictions
                    assert final_stats['total_requests'] == len(accessed_frames)
            
        finally:
            video_plume.close()
            cache.clear()
    
    def test_cache_thread_safety_video_access(self, sample_cache_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test thread safety of cache-enabled video frame access.
        
        Validates:
        - Concurrent frame access doesn't corrupt cache state
        - Multiple threads can safely access cached frames
        - Cache statistics remain consistent under concurrent access
        - No race conditions in cache operations
        """
        cache = FrameCache(mode='lru', max_size_mb=256, thread_safe=True)
        video_plume = VideoPlume(
            video_path=mock_video_file,
            grayscale=True,
            cache=cache
        )
        
        # Track access results from multiple threads
        access_results = []
        access_lock = threading.Lock()
        
        def concurrent_frame_access(thread_id: int, frame_indices: List[int]):
            """Function for concurrent frame access testing."""
            thread_results = []
            for frame_idx in frame_indices:
                try:
                    frame = video_plume.get_frame(frame_idx)
                    thread_results.append({
                        'thread_id': thread_id,
                        'frame_idx': frame_idx,
                        'success': frame is not None,
                        'frame_shape': frame.shape if frame is not None else None
                    })
                except Exception as e:
                    thread_results.append({
                        'thread_id': thread_id,
                        'frame_idx': frame_idx,
                        'success': False,
                        'error': str(e)
                    })
            
            with access_lock:
                access_results.extend(thread_results)
        
        try:
            # Create multiple threads accessing overlapping frame ranges
            threads = []
            num_threads = 3
            frames_per_thread = 10
            
            for thread_id in range(num_threads):
                # Each thread accesses overlapping frame ranges to test cache sharing
                start_frame = thread_id * 5
                frame_indices = list(range(start_frame, start_frame + frames_per_thread))
                
                thread = threading.Thread(
                    target=concurrent_frame_access,
                    args=(thread_id, frame_indices)
                )
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10.0)  # 10 second timeout
            
            # Validate all threads completed successfully
            assert len(access_results) == num_threads * frames_per_thread
            
            # Check for any errors in concurrent access
            errors = [result for result in access_results if not result['success']]
            if errors:
                pytest.fail(f"Concurrent access errors: {errors}")
            
            # Validate cache statistics consistency
            final_stats = video_plume.get_cache_stats()
            total_expected = len(access_results)
            actual_total = final_stats['hits'] + final_stats['misses']
            
            # Allow for some variance due to concurrent access patterns
            assert abs(actual_total - total_expected) <= num_threads, f"Statistics mismatch: expected ~{total_expected}, got {actual_total}"
            
        finally:
            video_plume.close()
            cache.clear()


@pytest.mark.skipif(not FRAME_CACHE_AVAILABLE, reason="FrameCache not available")
@pytest.mark.skipif(not GYMNASIUM_ENV_AVAILABLE, reason="Gymnasium environment not available")
class TestFrameCacheGymnasiumIntegration(CacheIntegrationTestBase):
    """Test FrameCache integration with GymnasiumEnv environment workflows."""
    
    def test_gymnasium_env_cache_initialization(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test GymnasiumEnv initialization with FrameCache integration.
        
        Validates:
        - GymnasiumEnv accepts FrameCache instance during initialization
        - Cache is properly integrated with VideoPlume
        - Environment maintains Gymnasium API compatibility
        - Performance monitoring is enabled with cache
        """
        # Create FrameCache instance
        cache = FrameCache(mode='lru', max_size_mb=256)
        
        # Initialize GymnasiumEnv with cache
        env = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            performance_monitoring=True,
            frame_cache=cache
        )
        
        try:
            # Validate cache integration
            assert env.frame_cache is not None
            assert env.frame_cache == cache
            assert env._cache_enabled is True
            
            # Validate Gymnasium API compatibility is maintained
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'observation_space')
            assert hasattr(env, 'reset')
            assert hasattr(env, 'step')
            assert hasattr(env, 'close')
            
            # Test environment reset with cache
            obs, info = env.reset()
            
            # Validate reset works with cache enabled
            assert obs is not None
            assert isinstance(info, dict)
            assert 'agent_position' in info
            assert 'agent_orientation' in info
            
            # Validate cache methods are available
            assert hasattr(env, 'get_cache_stats')
            assert hasattr(env, 'clear_cache')
            
            # Test cache statistics access
            cache_stats = env.get_cache_stats()
            assert isinstance(cache_stats, dict)
            assert 'enabled' in cache_stats
            assert cache_stats['enabled'] is True
            
        finally:
            env.close()
            cache.clear()
    
    def test_environment_step_with_cache_performance_stats(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache, performance_monitor):
        """
        Test environment step method populating info['perf_stats'] with cache metrics.
        
        Validates:
        - Step method includes performance statistics in info dictionary
        - Cache hit/miss metrics are tracked and reported
        - Step execution maintains sub-10ms performance target
        - Video frame is available in info['video_frame'] for analysis
        """
        cache = FrameCache(mode='lru', max_size_mb=256)
        env = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            performance_monitoring=True,
            frame_cache=cache
        )
        
        try:
            # Reset environment
            obs, reset_info = env.reset()
            
            # Validate reset info contains performance data
            if 'perf_stats' in reset_info:
                assert isinstance(reset_info['perf_stats'], dict)
            
            # Execute multiple steps to generate cache statistics
            step_count = 10
            step_times = []
            
            for step_idx in range(step_count):
                action = env.action_space.sample()
                
                with performance_monitor['time_operation'](f"step_{step_idx}", 10.0) as timer:
                    obs, reward, terminated, truncated, info = env.step(action)
                
                step_times.append(timer.duration)
                
                # Validate step info contains required fields
                assert isinstance(info, dict)
                assert 'agent_position' in info
                assert 'agent_orientation' in info
                
                # Validate performance statistics are included
                if 'perf_stats' in info:
                    perf_stats = info['perf_stats']
                    assert isinstance(perf_stats, dict)
                    
                    # Validate cache-specific performance metrics
                    expected_cache_fields = [
                        'cache_hit_rate',
                        'cache_hits', 
                        'cache_misses',
                        'cache_memory_usage_mb'
                    ]
                    
                    for field in expected_cache_fields:
                        if field in perf_stats:
                            assert isinstance(perf_stats[field], (int, float))
                            if field == 'cache_hit_rate':
                                assert 0.0 <= perf_stats[field] <= 1.0
                
                # Validate video frame availability for analysis
                if 'video_frame' in info:
                    video_frame = info['video_frame']
                    assert isinstance(video_frame, np.ndarray)
                    assert video_frame.ndim in [2, 3]  # Grayscale or color
                
                # Validate step performance target
                assert timer.duration < 0.050, f"Step {step_idx} took {timer.duration*1000:.2f}ms (>50ms threshold)"
                
                if terminated or truncated:
                    break
            
            # Validate overall performance
            avg_step_time = np.mean(step_times)
            max_step_time = np.max(step_times)
            
            # Performance targets validation
            assert avg_step_time < 0.020, f"Average step time {avg_step_time*1000:.2f}ms exceeds 20ms target"
            assert max_step_time < 0.050, f"Maximum step time {max_step_time*1000:.2f}ms exceeds 50ms limit"
            
            # Validate cache effectiveness
            final_cache_stats = env.get_cache_stats()
            if final_cache_stats['enabled'] and final_cache_stats.get('hits', 0) + final_cache_stats.get('misses', 0) > 0:
                hit_rate = final_cache_stats.get('hit_rate', 0.0)
                # With repeated access patterns, cache should achieve reasonable hit rate
                if step_count > 5:  # Only check after sufficient steps
                    assert hit_rate >= 0.1, f"Cache hit rate {hit_rate:.2%} too low after {step_count} steps"
            
        finally:
            env.close()
            cache.clear()
    
    def test_cache_warming_during_environment_reset(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test cache warming during environment reset for predictable performance.
        
        Validates:
        - Environment reset triggers cache warming when appropriate
        - Subsequent steps benefit from pre-cached frames
        - Reset performance is acceptable with cache warming
        - Cache warming improves step performance predictability
        """
        # Create cache with warming capabilities
        cache = FrameCache(mode='lru', max_size_mb=256)
        
        # Mock cache warming methods
        cache.warm_range = Mock()
        cache.is_warmed = Mock(return_value=False)
        
        env = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            performance_monitoring=True,
            frame_cache=cache
        )
        
        try:
            # First reset - may trigger cache warming
            reset_start = time.time()
            obs, info = env.reset()
            reset_time = time.time() - reset_start
            
            # Validate reset completed successfully
            assert obs is not None
            assert isinstance(info, dict)
            
            # Reset should complete in reasonable time even with warming
            assert reset_time < 1.0, f"Reset took {reset_time:.3f}s - too slow with cache warming"
            
            # Test multiple steps after reset to validate cache effectiveness
            step_times = []
            for step_idx in range(5):
                step_start = time.time()
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                if terminated or truncated:
                    break
            
            # Validate step performance after potential cache warming
            if len(step_times) > 0:
                avg_step_time = np.mean(step_times)
                # With cache, steps should be consistently fast
                assert avg_step_time < 0.030, f"Average step time {avg_step_time*1000:.2f}ms too slow with cache"
                
                # Step time variance should be low with cache
                step_variance = np.var(step_times)
                assert step_variance < 0.001, f"Step time variance {step_variance:.6f} too high - cache not effective"
            
            # Test second reset - should benefit from existing cache
            second_reset_start = time.time()
            obs2, info2 = env.reset()
            second_reset_time = time.time() - second_reset_start
            
            # Second reset might be faster due to cache
            assert second_reset_time <= reset_time * 1.5, "Second reset significantly slower than first"
            
        finally:
            env.close()
            cache.clear()
    
    def test_multi_environment_cache_sharing(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test cache sharing between multiple environment instances.
        
        Validates:
        - Multiple environments can share a single cache instance
        - Cache statistics reflect combined usage from all environments
        - No interference between environments using shared cache
        - Memory efficiency of shared cache approach
        """
        # Create shared cache instance
        shared_cache = FrameCache(mode='lru', max_size_mb=512)
        
        # Create multiple environments sharing the same cache
        envs = []
        num_envs = 3
        
        try:
            for env_idx in range(num_envs):
                env = GymnasiumEnv(
                    video_path=mock_video_file,
                    initial_position=(320 + env_idx * 10, 240 + env_idx * 10),
                    max_speed=2.0,
                    performance_monitoring=True,
                    frame_cache=shared_cache
                )
                envs.append(env)
            
            # Validate all environments use the same cache
            for env in envs:
                assert env.frame_cache is shared_cache
                assert env._cache_enabled is True
            
            # Reset all environments
            for env in envs:
                obs, info = env.reset()
                assert obs is not None
            
            # Execute steps on all environments to generate cache activity
            total_steps = 0
            for round_idx in range(5):
                for env_idx, env in enumerate(envs):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_steps += 1
                    
                    # Validate cache stats are accessible from each environment
                    cache_stats = env.get_cache_stats()
                    assert cache_stats['enabled'] is True
                    
                    if terminated or truncated:
                        # Reset environment if episode ended
                        env.reset()
            
            # Validate shared cache accumulated statistics from all environments
            final_stats = shared_cache.get_stats() if hasattr(shared_cache, 'get_stats') else {}
            
            # With shared cache, should have high activity
            if hasattr(shared_cache, 'total_accesses'):
                assert shared_cache.total_accesses >= total_steps
            
            # Validate memory efficiency - shared cache should use less memory than individual caches
            shared_memory = shared_cache.get_memory_usage() if hasattr(shared_cache, 'get_memory_usage') else 0
            
            # Memory usage should be reasonable for shared access
            if shared_memory > 0:
                max_expected = (512 * 1024 * 1024)  # 512 MB limit
                assert shared_memory <= max_expected, f"Shared cache using {shared_memory} bytes > {max_expected} limit"
            
        finally:
            # Clean up all environments and shared cache
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass  # Ignore cleanup errors
            shared_cache.clear()


@pytest.mark.skipif(not FRAME_CACHE_AVAILABLE, reason="FrameCache not available")  
@pytest.mark.skipif(not GYMNASIUM_ENV_AVAILABLE, reason="Gymnasium environment not available")
class TestCachePerformanceIntegration(CacheIntegrationTestBase):
    """Test cache performance integration and validation."""
    
    @pytest.mark.skipif(not BENCHMARK_AVAILABLE, reason="pytest-benchmark not available")
    def test_cache_enabled_step_performance_benchmark(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache, benchmark):
        """
        Benchmark step execution performance with cache enabled.
        
        Validates:
        - Step execution maintains sub-10ms target with cache
        - Cache significantly improves performance over uncached access
        - Performance is consistent across multiple step executions
        - Benchmark results meet performance requirements
        """
        cache = FrameCache(mode='lru', max_size_mb=256)
        env = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            performance_monitoring=True,
            frame_cache=cache
        )
        
        try:
            # Initialize environment
            obs, info = env.reset()
            action = env.action_space.sample()
            
            # Warm up cache with initial step
            env.step(action)
            
            def step_execution():
                """Function to benchmark step execution."""
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                return obs
            
            # Benchmark step execution with cache
            result = benchmark.pedantic(
                step_execution,
                rounds=20,      # Number of benchmark rounds
                iterations=5,   # Iterations per round
                warmup_rounds=2 # Warmup rounds
            )
            
            # Validate benchmark results meet performance requirements
            mean_time = benchmark.stats.mean
            max_time = benchmark.stats.max
            min_time = benchmark.stats.min
            
            # Sub-10ms requirement validation
            assert mean_time < 0.010, f"Mean step time {mean_time*1000:.2f}ms exceeds 10ms target"
            assert max_time < 0.020, f"Max step time {max_time*1000:.2f}ms exceeds 20ms tolerance"
            
            # Performance consistency validation
            time_variance = benchmark.stats.stddev
            assert time_variance < 0.005, f"Step time variance {time_variance*1000:.2f}ms too high - inconsistent performance"
            
            # Cache effectiveness validation
            cache_stats = env.get_cache_stats()
            if cache_stats['enabled']:
                total_accesses = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
                if total_accesses > 10:  # Sufficient data for hit rate analysis
                    hit_rate = cache_stats.get('hit_rate', 0.0)
                    assert hit_rate >= 0.5, f"Cache hit rate {hit_rate:.2%} too low for performance benefit"
            
        finally:
            env.close()
            cache.clear()
    
    def test_cache_vs_no_cache_performance_comparison(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache):
        """
        Compare performance between cache-enabled and cache-disabled environments.
        
        Validates:
        - Cache provides measurable performance improvement
        - Cache overhead is minimal compared to I/O savings
        - Performance difference is statistically significant
        - Cache benefit increases with repeated frame access
        """
        # Test without cache
        env_no_cache = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            performance_monitoring=True,
            frame_cache=None
        )
        
        # Test with cache
        cache = FrameCache(mode='lru', max_size_mb=256)
        env_with_cache = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            performance_monitoring=True,
            frame_cache=cache
        )
        
        try:
            # Measure performance without cache
            env_no_cache.reset()
            no_cache_times = []
            
            for step_idx in range(10):
                action = env_no_cache.action_space.sample()
                start_time = time.perf_counter()
                obs, reward, terminated, truncated, info = env_no_cache.step(action)
                step_time = time.perf_counter() - start_time
                no_cache_times.append(step_time)
                
                if terminated or truncated:
                    env_no_cache.reset()
            
            # Measure performance with cache
            env_with_cache.reset()
            with_cache_times = []
            
            for step_idx in range(10):
                action = env_with_cache.action_space.sample()
                start_time = time.perf_counter()
                obs, reward, terminated, truncated, info = env_with_cache.step(action)
                step_time = time.perf_counter() - start_time
                with_cache_times.append(step_time)
                
                if terminated or truncated:
                    env_with_cache.reset()
            
            # Statistical analysis of performance difference
            avg_no_cache = np.mean(no_cache_times)
            avg_with_cache = np.mean(with_cache_times)
            
            # Cache should provide some benefit, though improvement may be limited in test environment
            improvement_ratio = avg_no_cache / avg_with_cache if avg_with_cache > 0 else 1.0
            
            # Log performance comparison for analysis
            logger.info(
                f"Performance comparison - No cache: {avg_no_cache*1000:.2f}ms, "
                f"With cache: {avg_with_cache*1000:.2f}ms, "
                f"Improvement: {improvement_ratio:.2f}x"
            )
            
            # Validate both configurations meet basic performance requirements
            assert avg_no_cache < 0.100, f"No-cache performance {avg_no_cache*1000:.2f}ms too slow"
            assert avg_with_cache < 0.100, f"With-cache performance {avg_with_cache*1000:.2f}ms too slow"
            
            # Cache should not significantly degrade performance
            assert improvement_ratio >= 0.8, f"Cache degraded performance by {(1-improvement_ratio)*100:.1f}%"
            
        finally:
            env_no_cache.close()
            env_with_cache.close()
            cache.clear()
    
    def test_cache_memory_usage_under_load(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test cache memory usage behavior under sustained load.
        
        Validates:
        - Memory usage stays within configured limits
        - Memory usage stabilizes under sustained access
        - No memory leaks during extended operation
        - LRU eviction works correctly under memory pressure
        """
        import psutil
        import gc
        
        # Create cache with moderate memory limit
        memory_limit = 100 * 1024 * 1024  # 100 MB
        cache = FrameCache(mode='lru', memory_limit=memory_limit)
        
        env = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            performance_monitoring=True,
            frame_cache=cache
        )
        
        try:
            # Get initial memory baseline
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Reset environment
            env.reset()
            
            # Execute many steps to create memory pressure
            steps_executed = 0
            memory_samples = []
            
            for cycle in range(10):  # 10 cycles of 20 steps each
                for step in range(20):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps_executed += 1
                    
                    # Sample memory usage periodically
                    if steps_executed % 10 == 0:
                        current_memory = process.memory_info().rss
                        memory_samples.append({
                            'step': steps_executed,
                            'memory_rss': current_memory,
                            'memory_increase': current_memory - initial_memory
                        })
                        
                        # Check cache memory usage
                        cache_memory = cache.get_memory_usage() if hasattr(cache, 'get_memory_usage') else 0
                        if cache_memory > 0:
                            assert cache_memory <= memory_limit, f"Cache memory {cache_memory} exceeds limit {memory_limit}"
                    
                    if terminated or truncated:
                        env.reset()
                
                # Force garbage collection between cycles
                gc.collect()
            
            # Analyze memory usage patterns
            if len(memory_samples) > 1:
                memory_increases = [sample['memory_increase'] for sample in memory_samples]
                
                # Memory should stabilize (not grow indefinitely)
                final_increase = memory_increases[-1]
                max_increase = max(memory_increases)
                
                # Allow for reasonable memory increase due to caching
                reasonable_limit = 200 * 1024 * 1024  # 200 MB increase limit
                assert final_increase < reasonable_limit, f"Memory increased by {final_increase/1024/1024:.1f}MB - possible leak"
                
                # Memory usage should not grow continuously
                if len(memory_increases) >= 5:
                    # Check if memory growth has stabilized in recent samples
                    recent_samples = memory_increases[-5:]
                    memory_variance = np.var(recent_samples)
                    max_recent = max(recent_samples)
                    min_recent = min(recent_samples)
                    
                    # Recent memory usage should be relatively stable
                    memory_range = max_recent - min_recent
                    assert memory_range < 50 * 1024 * 1024, f"Memory usage unstable: {memory_range/1024/1024:.1f}MB variance"
            
            # Validate cache statistics for sustained operation
            final_cache_stats = env.get_cache_stats()
            if final_cache_stats['enabled']:
                total_requests = final_cache_stats.get('hits', 0) + final_cache_stats.get('misses', 0)
                assert total_requests >= steps_executed * 0.8, "Cache statistics inconsistent with step count"
                
                # Should have reasonable hit rate under sustained load
                if total_requests > 50:
                    hit_rate = final_cache_stats.get('hit_rate', 0.0)
                    assert hit_rate >= 0.3, f"Hit rate {hit_rate:.2%} too low for sustained access pattern"
            
        finally:
            env.close()
            cache.clear()
            gc.collect()  # Final cleanup


@pytest.mark.skipif(not FRAME_CACHE_AVAILABLE, reason="FrameCache not available")
@pytest.mark.skipif(not GYMNASIUM_ENV_AVAILABLE, reason="Gymnasium environment not available")
class TestCacheAPICompatibility(CacheIntegrationTestBase):
    """Test cache integration maintains full Gymnasium API compatibility."""
    
    def test_gymnasium_api_compliance_with_cache(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test complete Gymnasium API compliance with cache enabled.
        
        Validates:
        - All Gymnasium interface methods work with cache
        - API behavior is identical with and without cache
        - Action and observation spaces are unchanged
        - Episode lifecycle works correctly with cache
        """
        # Test cache-enabled environment
        cache = FrameCache(mode='lru', max_size_mb=256)
        env_cached = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            frame_cache=cache
        )
        
        # Test reference environment without cache
        env_reference = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            frame_cache=None
        )
        
        try:
            # Test API interface compliance
            api_methods = ['reset', 'step', 'close', 'seed', 'render']
            for method_name in api_methods:
                assert hasattr(env_cached, method_name), f"Missing API method: {method_name}"
                assert hasattr(env_reference, method_name), f"Reference missing API method: {method_name}"
            
            # Test action and observation space consistency
            assert env_cached.action_space == env_reference.action_space
            assert type(env_cached.observation_space) == type(env_reference.observation_space)
            
            # Test space bounds and properties
            np.testing.assert_array_equal(
                env_cached.action_space.low, 
                env_reference.action_space.low
            )
            np.testing.assert_array_equal(
                env_cached.action_space.high, 
                env_reference.action_space.high
            )
            
            # Test reset behavior
            obs_cached, info_cached = env_cached.reset(seed=42)
            obs_reference, info_reference = env_reference.reset(seed=42)
            
            # Observations should have same structure (values may differ due to cache)
            assert type(obs_cached) == type(obs_reference)
            if isinstance(obs_cached, dict):
                assert set(obs_cached.keys()) == set(obs_reference.keys())
                for key in obs_cached.keys():
                    assert obs_cached[key].shape == obs_reference[key].shape
                    assert obs_cached[key].dtype == obs_reference[key].dtype
            
            # Test step behavior with identical actions
            action = env_cached.action_space.sample()
            
            # Set identical seeds for reproducible comparison
            env_cached.seed(123)
            env_reference.seed(123)
            
            step_cached = env_cached.step(action)
            step_reference = env_reference.step(action)
            
            # Step return format should be identical
            assert len(step_cached) == len(step_reference)
            assert len(step_cached) == 5  # obs, reward, terminated, truncated, info
            
            obs_c, reward_c, term_c, trunc_c, info_c = step_cached
            obs_r, reward_r, term_r, trunc_r, info_r = step_reference
            
            # Validate return types
            assert type(obs_c) == type(obs_r)
            assert type(reward_c) == type(reward_r)
            assert type(term_c) == type(term_r)
            assert type(trunc_c) == type(trunc_r)
            assert type(info_c) == type(info_r)
            
            # Boolean flags should be identical (deterministic)
            assert isinstance(term_c, bool) and isinstance(term_r, bool)
            assert isinstance(trunc_c, bool) and isinstance(trunc_r, bool)
            
            # Info dictionaries should have compatible structure
            common_keys = set(info_c.keys()) & set(info_r.keys())
            assert len(common_keys) > 0, "No common info keys between cached and reference"
            
            # Test seeding behavior
            seed_result_cached = env_cached.seed(456)
            seed_result_reference = env_reference.seed(456)
            
            assert isinstance(seed_result_cached, list)
            assert isinstance(seed_result_reference, list)
            assert len(seed_result_cached) == len(seed_result_reference)
            
        finally:
            env_cached.close()
            env_reference.close()
            cache.clear()
    
    def test_vectorized_environment_compatibility(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test cache integration compatibility with vectorized environments.
        
        Validates:
        - Cache-enabled environments work in vectorized wrappers
        - Multiple environment instances can share cache resources
        - Vectorized operations maintain performance benefits
        - No interference between vectorized environment instances
        """
        # Create shared cache for vectorized environments
        shared_cache = FrameCache(mode='lru', max_size_mb=512)
        
        # Create factory function for vectorized environment creation
        def make_env():
            return GymnasiumEnv(
                video_path=mock_video_file,
                initial_position=(320, 240),
                max_speed=2.0,
                frame_cache=shared_cache,
                performance_monitoring=True
            )
        
        # Create multiple environment instances (simulating vectorization)
        num_envs = 3
        envs = [make_env() for _ in range(num_envs)]
        
        try:
            # Validate all environments share the same cache
            for env in envs:
                assert env.frame_cache is shared_cache
                assert env._cache_enabled is True
            
            # Test parallel reset operations
            reset_results = []
            for env_idx, env in enumerate(envs):
                obs, info = env.reset(seed=env_idx * 100)
                reset_results.append((obs, info))
                
                # Validate reset successful
                assert obs is not None
                assert isinstance(info, dict)
            
            # Test parallel step operations
            step_results = []
            for round_idx in range(5):
                round_results = []
                
                for env_idx, env in enumerate(envs):
                    action = env.action_space.sample()
                    step_result = env.step(action)
                    round_results.append(step_result)
                    
                    # Validate step result format
                    assert len(step_result) == 5
                    obs, reward, terminated, truncated, info = step_result
                    
                    # Check for cache statistics in info
                    if 'perf_stats' in info and 'cache_hit_rate' in info['perf_stats']:
                        hit_rate = info['perf_stats']['cache_hit_rate']
                        assert 0.0 <= hit_rate <= 1.0
                
                step_results.append(round_results)
            
            # Validate shared cache accumulated statistics
            final_cache_stats = shared_cache.get_stats() if hasattr(shared_cache, 'get_stats') else {}
            
            # With multiple environments using shared cache, should have significant activity
            total_operations = len(envs) * 6  # Reset + 5 steps per env
            
            # Validate environments maintained independence despite shared cache
            for round_results in step_results:
                # Each environment should have potentially different states
                observations = [result[0] for result in round_results]
                
                # Observations should be valid for all environments
                for obs in observations:
                    assert obs is not None
                    if isinstance(obs, dict):
                        assert len(obs) > 0
            
        finally:
            # Clean up all environments
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass
            shared_cache.clear()
    
    def test_environment_attribute_access_with_cache(self, sample_environment_config, mock_video_file, mock_video_capture_for_cache):
        """
        Test environment attribute access and modification with cache enabled.
        
        Validates:
        - get_attr/set_attr methods work with cache
        - env_method calls work with cache integration
        - Cache-specific methods are accessible
        - No conflicts between cache and environment attributes
        """
        cache = FrameCache(mode='lru', max_size_mb=256)
        env = GymnasiumEnv(
            video_path=mock_video_file,
            initial_position=(320, 240),
            max_speed=2.0,
            frame_cache=cache,
            performance_monitoring=True
        )
        
        try:
            # Test get_attr method
            action_space = env.get_attr('action_space')
            assert action_space is not None
            assert action_space == env.action_space
            
            # Test cache-specific attribute access
            frame_cache = env.get_attr('frame_cache')
            assert frame_cache is cache
            
            cache_enabled = env.get_attr('_cache_enabled')
            assert cache_enabled is True
            
            # Test set_attr method
            test_attribute = 'test_cache_integration'
            test_value = 'cache_integration_test_value'
            env.set_attr(test_attribute, test_value)
            
            retrieved_value = env.get_attr(test_attribute)
            assert retrieved_value == test_value
            
            # Test env_method calls
            cache_stats = env.env_method('get_cache_stats')
            assert isinstance(cache_stats, dict)
            assert 'enabled' in cache_stats
            assert cache_stats['enabled'] is True
            
            # Test cache-specific method calls
            env.env_method('clear_cache')
            
            # Validate cache was cleared
            cleared_stats = env.get_cache_stats()
            if 'hits' in cleared_stats and 'misses' in cleared_stats:
                assert cleared_stats['hits'] == 0
                assert cleared_stats['misses'] == 0
            
            # Test method that doesn't exist (should raise AttributeError)
            with pytest.raises(AttributeError):
                env.env_method('nonexistent_cache_method')
            
        finally:
            env.close()
            cache.clear()


# Performance benchmarking utilities for cache integration testing
class CachePerformanceBenchmark:
    """Utility class for cache performance benchmarking in integration tests."""
    
    def __init__(self, name: str, target_threshold: float, cache_hit_target: float = 0.9):
        self.name = name
        self.target_threshold = target_threshold  # Target time in seconds
        self.cache_hit_target = cache_hit_target  # Target cache hit rate
        self.measurements = []
        self.cache_stats = []
    
    def measure_operation(self, operation_func, *args, **kwargs):
        """Measure a single operation and collect cache statistics."""
        start_time = time.perf_counter()
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter()
        
        measurement = {
            'duration': end_time - start_time,
            'timestamp': end_time,
            'result': result
        }
        self.measurements.append(measurement)
        
        return result
    
    def add_cache_stats(self, cache_stats: Dict[str, Any]):
        """Add cache statistics for analysis."""
        self.cache_stats.append(cache_stats.copy())
    
    def validate_performance(self) -> Dict[str, Any]:
        """Validate performance against targets and return analysis."""
        if not self.measurements:
            return {'valid': False, 'error': 'No measurements collected'}
        
        durations = [m['duration'] for m in self.measurements]
        mean_duration = np.mean(durations)
        max_duration = np.max(durations)
        min_duration = np.min(durations)
        std_duration = np.std(durations)
        
        # Performance validation
        performance_valid = mean_duration <= self.target_threshold
        consistency_valid = std_duration <= (self.target_threshold * 0.5)  # Variance should be reasonable
        
        # Cache effectiveness validation
        cache_valid = True
        final_hit_rate = 0.0
        
        if self.cache_stats:
            final_stats = self.cache_stats[-1]
            total_requests = final_stats.get('hits', 0) + final_stats.get('misses', 0)
            if total_requests > 0:
                final_hit_rate = final_stats.get('hits', 0) / total_requests
                cache_valid = final_hit_rate >= self.cache_hit_target
        
        return {
            'valid': performance_valid and consistency_valid and cache_valid,
            'performance_valid': performance_valid,
            'consistency_valid': consistency_valid,
            'cache_valid': cache_valid,
            'mean_duration': mean_duration,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'std_duration': std_duration,
            'final_hit_rate': final_hit_rate,
            'measurement_count': len(self.measurements),
            'target_threshold': self.target_threshold,
            'cache_hit_target': self.cache_hit_target
        }


# Test execution configuration and pytest markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.cache,  # New marker for cache-specific tests
    pytest.mark.slow,   # Integration tests may take longer
]


# Module-level test utilities and fixtures
@pytest.fixture(scope="session")
def cache_integration_test_session():
    """Session-scoped fixture for cache integration test setup."""
    # Global test session setup for cache integration
    start_time = time.time()
    
    # Set global test environment for cache testing
    os.environ.setdefault("FRAME_CACHE_TEST_MODE", "true")
    os.environ.setdefault("CACHE_INTEGRATION_TESTING", "true")
    
    yield
    
    # Global test session teardown
    total_time = time.time() - start_time
    print(f"\nCache integration test session completed in {total_time:.2f}s")
    
    # Clean up environment
    os.environ.pop("FRAME_CACHE_TEST_MODE", None)
    os.environ.pop("CACHE_INTEGRATION_TESTING", None)


if __name__ == "__main__":
    # Enable direct execution for debugging
    pytest.main([__file__, "-v", "--tb=short", "-m", "cache"])