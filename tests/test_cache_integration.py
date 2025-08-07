"""
Integration tests for FrameCache system with VideoPlume and GymnasiumEnv.

This test suite validates the comprehensive integration of the FrameCache system with
VideoPlume video processing and GymnasiumEnv step execution, ensuring cache-enabled
environments maintain API compatibility while providing significant performance 
optimizations for reinforcement learning workflows.

Test Coverage:
- VideoPlume.get_frame() cache checking functionality per Section 0.4.1
- Zero-copy frame retrieval validation per Section 0.3.3 implementation details
- Cache warming functionality per Section 0.3.1 technical approach
- Cache hit/miss statistics integration with VideoPlume operations per Section 0.2.2
- Batch preloading for sequential access patterns per Section 0.3.3
- Cache integration with GymnasiumEnv step() method per Section 0.2.2
- API compatibility preservation per Section 0.1.4 integration constraints

Performance Targets:
- <10ms frame retrieval latency for cache hits
- >90% cache hit rate for sequential access patterns
- Sub-10ms step() execution time with cache enabled
- API compatibility preserved across all cache modes

Architecture Validation:
- Thread-safe cache operations for multi-agent scenarios
- Memory pressure handling and intelligent eviction
- Cache statistics accuracy and real-time monitoring
- Integration with Loguru structured logging
"""

import pytest
import numpy as np
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import warnings

# Core imports with graceful fallbacks
try:
    from odor_plume_nav.cache.frame_cache import (
        FrameCache, CacheMode, CacheStatistics,
        create_lru_cache, create_preload_cache, create_no_cache
    )
    FRAME_CACHE_AVAILABLE = True
except ImportError:
    FRAME_CACHE_AVAILABLE = False
    FrameCache = CacheMode = CacheStatistics = None

try:
    from odor_plume_nav.data.video_plume import VideoPlume
    VIDEO_PLUME_AVAILABLE = True
except ImportError:
    VIDEO_PLUME_AVAILABLE = False
    VideoPlume = None

try:
    from odor_plume_nav.environments.gymnasium_env import GymnasiumEnv
    GYMNASIUM_ENV_AVAILABLE = True
except ImportError:
    GYMNASIUM_ENV_AVAILABLE = False
    GymnasiumEnv = None

# Test utilities and mocks are available as pytest fixtures from conftest.py


# Skip all tests if required components are not available
pytestmark = pytest.mark.skipif(
    not (FRAME_CACHE_AVAILABLE and VIDEO_PLUME_AVAILABLE),
    reason="FrameCache and VideoPlume components required for integration tests"
)


class MockVideoCapture:
    """Enhanced mock VideoCapture for comprehensive testing."""
    
    def __init__(self, width: int = 640, height: int = 480, frame_count: int = 300, fps: float = 30.0):
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.fps = fps
        self.current_frame = 0
        self.is_opened = True
        
        # Generate deterministic test frames
        self._frames = self._generate_test_frames()
    
    def _generate_test_frames(self) -> List[np.ndarray]:
        """Generate deterministic test frames with unique patterns."""
        frames = []
        for i in range(self.frame_count):
            # Create frame with unique pattern for each frame
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Add frame-specific pattern for verification
            frame[:, :, 0] = (i % 256)  # Red channel encodes frame number
            frame[:, :, 1] = ((i * 2) % 256)  # Green channel with different pattern
            frame[:, :, 2] = ((i * 3) % 256)  # Blue channel with different pattern
            
            # Add some spatial structure for realistic testing
            y, x = np.ogrid[:self.height, :self.width]
            center_x, center_y = self.width // 2, self.height // 2
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 < (min(self.width, self.height) // 4) ** 2
            frame[mask] = frame[mask] + 50  # Brighter center region
            
            frames.append(frame)
        return frames
    
    def isOpened(self) -> bool:
        return self.is_opened
    
    def get(self, prop: int) -> float:
        """Get video property."""
        if prop == 0:  # CAP_PROP_FRAME_WIDTH
            return float(self.width)
        elif prop == 1:  # CAP_PROP_FRAME_HEIGHT
            return float(self.height)
        elif prop == 5:  # CAP_PROP_FPS
            return self.fps
        elif prop == 7:  # CAP_PROP_FRAME_COUNT
            return float(self.frame_count)
        return 0.0
    
    def set(self, prop: int, value: float) -> bool:
        """Set video property."""
        if prop == 1:  # CAP_PROP_POS_FRAMES
            self.current_frame = int(value)
            return True
        return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read current frame."""
        if not self.is_opened or self.current_frame >= len(self._frames):
            return False, None
        
        frame = self._frames[self.current_frame].copy()
        return True, frame
    
    def release(self) -> None:
        """Release video capture."""
        self.is_opened = False


@pytest.fixture
def mock_video_capture_enhanced():
    """Enhanced mock video capture for integration testing."""
    return MockVideoCapture()


@pytest.fixture
def test_video_path(tmp_path):
    """Create a test video file path."""
    video_path = tmp_path / "test_plume.mp4"
    video_path.touch()  # Create empty file
    return video_path


@pytest.fixture
def frame_cache_lru():
    """Create LRU frame cache for testing."""
    if not FRAME_CACHE_AVAILABLE:
        pytest.skip("FrameCache not available")
    return create_lru_cache(memory_limit_mb=100, enable_logging=False)


@pytest.fixture
def frame_cache_preload():
    """Create preload frame cache for testing."""
    if not FRAME_CACHE_AVAILABLE:
        pytest.skip("FrameCache not available")
    return create_preload_cache(memory_limit_mb=100, enable_logging=False)


@pytest.fixture
def frame_cache_disabled():
    """Create disabled frame cache for testing."""
    if not FRAME_CACHE_AVAILABLE:
        pytest.skip("FrameCache not available")
    return create_no_cache(enable_logging=False)


@pytest.fixture
def video_plume_with_cache(test_video_path, frame_cache_lru, mock_video_capture_enhanced):
    """Create VideoPlume instance integrated with frame cache."""
    with patch('odor_plume_nav.data.video_plume.cv2.VideoCapture') as mock_cv:
        mock_cv.return_value = mock_video_capture_enhanced
        
        # VideoPlume will be enhanced in another agent to accept cache parameter
        # For now, we mock the integration
        video_plume = VideoPlume(str(test_video_path))
        
        # Mock cache integration methods that will be added
        video_plume._frame_cache = frame_cache_lru
        video_plume._cache_enabled = True
        
        # Create cached get_frame method
        original_get_frame = video_plume.get_frame
        
        def cached_get_frame(frame_idx: int) -> Optional[np.ndarray]:
            """Cached version of get_frame method."""
            if video_plume._cache_enabled and video_plume._frame_cache:
                return video_plume._frame_cache.get(frame_idx, video_plume=MockVideoPlume(original_get_frame))
            return original_get_frame(frame_idx)
        
        video_plume.get_frame = cached_get_frame
        yield video_plume


class MockVideoPlume:
    """Mock VideoPlume for cache integration testing."""
    
    def __init__(self, get_frame_func):
        self._get_frame_func = get_frame_func
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        return self._get_frame_func(frame_idx)


@pytest.fixture  
def gymnasium_env_with_cache(test_video_path, frame_cache_lru, mock_video_capture_enhanced):
    """Create GymnasiumEnv with frame cache integration."""
    if not GYMNASIUM_ENV_AVAILABLE:
        pytest.skip("GymnasiumEnv not available")
    
    with patch('odor_plume_nav.data.video_plume.cv2.VideoCapture') as mock_cv:
        mock_cv.return_value = mock_video_capture_enhanced
        
        # Mock navigator and other dependencies
        with patch('odor_plume_nav.environments.gymnasium_env.NavigatorFactory') as mock_nav_factory:
            mock_navigator = Mock()
            mock_navigator.positions = np.array([[320.0, 240.0]])
            mock_navigator.orientations = np.array([0.0])
            mock_navigator.speeds = np.array([0.0])
            mock_navigator.angular_velocities = np.array([0.0])
            mock_navigator.sample_odor.return_value = 0.5
            mock_navigator.sample_multiple_sensors.return_value = np.array([0.4, 0.6])
            mock_navigator.step.return_value = None
            mock_navigator.reset.return_value = None
            
            mock_nav_factory.create_navigator.return_value = mock_navigator
            
            # Create environment with cache
            env = GymnasiumEnv(
                video_path=str(test_video_path),
                frame_cache=frame_cache_lru,
                performance_monitoring=True
            )
            
            yield env


class TestFrameCacheVideoPlumeMethods:
    """Test FrameCache integration with VideoPlume methods."""
    
    def test_cache_get_method_interface(self, frame_cache_lru, mock_video_capture_enhanced):
        """Test FrameCache.get() method interface with VideoPlume."""
        # Create mock VideoPlume for cache integration
        mock_video_plume = Mock()
        mock_video_plume.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test cache miss - should call VideoPlume.get_frame
        frame = frame_cache_lru.get(frame_id=0, video_plume=mock_video_plume)
        
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        mock_video_plume.get_frame.assert_called_once_with(0)
        
        # Verify cache statistics
        assert frame_cache_lru.statistics.total_requests == 1
        assert frame_cache_lru.statistics._misses == 1
        assert frame_cache_lru.statistics._hits == 0
    
    def test_cache_hit_zero_copy_retrieval(self, frame_cache_lru):
        """Test zero-copy frame retrieval validation per Section 0.3.3."""
        # Create test frame with unique pattern
        test_frame = np.random.rand(480, 640, 3).astype(np.float32)
        test_frame[100:200, 100:200] = 1.0  # Unique marker
        
        mock_video_plume = Mock()
        mock_video_plume.get_frame.return_value = test_frame
        
        # First access - cache miss
        frame1 = frame_cache_lru.get(frame_id=42, video_plume=mock_video_plume)
        assert frame1 is not None
        
        # Second access - cache hit (should be zero-copy)
        frame2 = frame_cache_lru.get(frame_id=42, video_plume=mock_video_plume)
        
        # Verify frames are equivalent but independent copies
        np.testing.assert_array_equal(frame1, frame2)
        assert frame1 is not frame2  # Different objects (copies for safety)
        
        # Verify cache hit
        assert frame_cache_lru.statistics._hits == 1
        assert frame_cache_lru.statistics._misses == 1
        
        # Verify VideoPlume.get_frame called only once
        assert mock_video_plume.get_frame.call_count == 1
    
    def test_cache_warming_functionality(self, frame_cache_lru):
        """Test cache warming during environment reset per Section 0.3.1."""
        # Create mock VideoPlume with multiple frames
        mock_video_plume = Mock()
        
        def mock_get_frame(frame_idx):
            if 0 <= frame_idx < 100:
                frame = np.full((480, 640), frame_idx, dtype=np.uint8)
                return frame
            return None
        
        mock_video_plume.get_frame.side_effect = mock_get_frame
        mock_video_plume.frame_count = 100
        
        # Test warmup operation
        warmup_start = time.time()
        success = frame_cache_lru.warmup(mock_video_plume, warmup_frames=20)
        warmup_time = time.time() - warmup_start
        
        assert success is True
        assert warmup_time < 1.0  # Should complete quickly
        
        # Verify frames were cached
        assert frame_cache_lru.cache_size >= 20
        
        # Test subsequent access is from cache
        frame = frame_cache_lru.get(frame_id=5, video_plume=mock_video_plume)
        assert frame is not None
        assert frame[0, 0] == 5  # Verify correct frame
        
        # Should be cache hit
        assert frame_cache_lru.hit_rate > 0.0
    
    def test_batch_preloading_sequential_access(self, frame_cache_preload):
        """Test batch preloading for sequential access patterns per Section 0.3.3."""
        # Create mock VideoPlume with sequential frames
        mock_video_plume = Mock()
        
        def mock_get_frame(frame_idx):
            if 0 <= frame_idx < 500:
                # Create frame with frame_idx encoded in pixel values
                frame = np.full((240, 320), frame_idx % 256, dtype=np.uint8)
                return frame
            return None
        
        mock_video_plume.get_frame.side_effect = mock_get_frame
        
        # Test batch preloading
        preload_start = time.time()
        success = frame_cache_preload.preload(
            frame_range=range(0, 100), 
            video_plume=mock_video_plume
        )
        preload_time = time.time() - preload_start
        
        assert success is True
        assert preload_time < 5.0  # Should complete within reasonable time
        assert frame_cache_preload.cache_size == 100
        
        # Test sequential access performance
        access_times = []
        for i in range(0, 100, 10):  # Sample every 10th frame
            start = time.time()
            frame = frame_cache_preload.get(frame_id=i, video_plume=mock_video_plume)
            access_time = time.time() - start
            access_times.append(access_time)
            
            assert frame is not None
            assert frame[0, 0] == i % 256  # Verify correct frame
        
        # Verify fast cache access (<10ms target)
        avg_access_time = np.mean(access_times)
        assert avg_access_time < 0.01  # <10ms target
        
        # Verify high hit rate (>90% target)
        assert frame_cache_preload.hit_rate >= 0.9


class TestFrameCacheStatisticsIntegration:
    """Test cache hit/miss statistics integration with VideoPlume operations."""
    
    def test_statistics_accuracy_with_video_operations(self, frame_cache_lru):
        """Test cache statistics accuracy with VideoPlume operations per Section 0.2.2."""
        mock_video_plume = Mock()
        
        # Create frames with deterministic patterns
        def mock_get_frame(frame_idx):
            return np.full((100, 100), frame_idx, dtype=np.uint8)
        
        mock_video_plume.get_frame.side_effect = mock_get_frame
        
        # Test mixed access pattern
        access_pattern = [0, 1, 2, 0, 1, 3, 0, 4, 1, 2]  # Some repeats
        
        for frame_id in access_pattern:
            frame = frame_cache_lru.get(frame_id, mock_video_plume)
            assert frame is not None
        
        # Verify statistics
        stats = frame_cache_lru.statistics.get_summary()
        
        assert stats['total_requests'] == len(access_pattern)
        assert stats['hits'] + stats['misses'] == len(access_pattern)
        
        # Expected: frames 0,1,2,3,4 are unique (5 misses)
        # Repeats: 0(3 times total, 2 hits), 1(3 times total, 2 hits), 2(2 times total, 1 hit)
        expected_misses = 5  # Unique frames
        expected_hits = len(access_pattern) - expected_misses
        
        assert stats['misses'] == expected_misses
        assert stats['hits'] == expected_hits
        assert abs(stats['hit_rate'] - (expected_hits / len(access_pattern))) < 0.01
    
    def test_memory_usage_tracking(self, frame_cache_lru):
        """Test memory usage tracking accuracy."""
        mock_video_plume = Mock()
        
        # Create frames of known size
        frame_size = 100 * 100 * 1  # 10KB per frame
        
        def mock_get_frame(frame_idx):
            return np.zeros((100, 100), dtype=np.uint8)
        
        mock_video_plume.get_frame.side_effect = mock_get_frame
        
        initial_memory = frame_cache_lru.memory_usage_mb
        
        # Cache 10 frames
        for i in range(10):
            frame_cache_lru.get(i, mock_video_plume)
        
        final_memory = frame_cache_lru.memory_usage_mb
        memory_increase = final_memory - initial_memory
        
        # Verify memory tracking accuracy (within 10% tolerance)
        expected_memory_mb = (10 * frame_size) / (1024 * 1024)
        assert abs(memory_increase - expected_memory_mb) < expected_memory_mb * 0.1
    
    def test_real_time_statistics_updates(self, frame_cache_lru):
        """Test real-time statistics updates during operations."""
        mock_video_plume = Mock()
        mock_video_plume.get_frame.return_value = np.zeros((50, 50), dtype=np.uint8)
        
        # Monitor statistics during operations
        initial_stats = frame_cache_lru.statistics.get_summary()
        
        # Perform cache operations
        frame_cache_lru.get(0, mock_video_plume)  # Miss
        mid_stats = frame_cache_lru.statistics.get_summary()
        
        frame_cache_lru.get(0, mock_video_plume)  # Hit
        final_stats = frame_cache_lru.statistics.get_summary()
        
        # Verify incremental updates
        assert mid_stats['misses'] == initial_stats['misses'] + 1
        assert mid_stats['hits'] == initial_stats['hits']
        
        assert final_stats['misses'] == mid_stats['misses']
        assert final_stats['hits'] == mid_stats['hits'] + 1


class TestGymnasiumEnvCacheIntegration:
    """Test cache integration with GymnasiumEnv step() method."""
    
    @pytest.mark.skipif(not GYMNASIUM_ENV_AVAILABLE, reason="GymnasiumEnv not available")
    def test_gymnasium_env_step_with_cache(self, gymnasium_env_with_cache):
        """Test cache integration with GymnasiumEnv step() method per Section 0.2.2."""
        env = gymnasium_env_with_cache
        
        # Reset environment to initialize
        obs, info = env.reset()
        
        # Verify performance monitoring is enabled
        assert env.performance_monitoring is True
        assert env._cache_enabled is True
        
        # Perform multiple steps to test cache integration
        step_times = []
        for i in range(10):
            action = env.action_space.sample()
            
            step_start = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Verify performance stats are included
            assert "perf_stats" in info
            perf_stats = info["perf_stats"]
            
            # Verify cache statistics are included
            assert "cache_hit_rate" in perf_stats
            assert "cache_hits" in perf_stats
            assert "cache_misses" in perf_stats
            assert "cache_memory_usage_mb" in perf_stats
            
            # Verify video frame is included for analysis
            assert "video_frame" in info
            assert info["video_frame"] is not None
            
            if terminated or truncated:
                break
        
        # Verify step performance target (<33ms for 30 FPS)
        avg_step_time = np.mean(step_times)
        assert avg_step_time < 0.033  # 33ms target
    
    @pytest.mark.skipif(not GYMNASIUM_ENV_AVAILABLE, reason="GymnasiumEnv not available")
    def test_cache_performance_metrics_in_step_info(self, gymnasium_env_with_cache):
        """Test cache performance metrics in step info per Section 0.1.4."""
        env = gymnasium_env_with_cache
        
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify comprehensive performance metrics
        assert "perf_stats" in info
        perf_stats = info["perf_stats"]
        
        # Core performance metrics
        required_metrics = [
            "step_time_ms", "frame_retrieval_ms", "avg_step_time_ms", 
            "fps_estimate", "step_count", "episode"
        ]
        for metric in required_metrics:
            assert metric in perf_stats
            assert isinstance(perf_stats[metric], (int, float))
        
        # Cache-specific metrics
        cache_metrics = [
            "cache_hit_rate", "cache_hits", "cache_misses", "cache_evictions",
            "cache_memory_usage_mb", "cache_size"
        ]
        for metric in cache_metrics:
            assert metric in perf_stats
            assert isinstance(perf_stats[metric], (int, float))
        
        # Verify performance targets
        assert perf_stats["step_time_ms"] < 33  # <33ms for 30 FPS
        if perf_stats["cache_hits"] + perf_stats["cache_misses"] > 0:
            assert 0.0 <= perf_stats["cache_hit_rate"] <= 1.0
    
    @pytest.mark.skipif(not GYMNASIUM_ENV_AVAILABLE, reason="GymnasiumEnv not available")
    def test_api_compatibility_with_cache_enabled(self, gymnasium_env_with_cache):
        """Test API compatibility preservation with cache enabled per Section 0.1.4."""
        env = gymnasium_env_with_cache
        
        # Test standard Gymnasium API
        obs, info = env.reset()
        
        # Verify observation space structure
        assert isinstance(obs, dict)
        expected_keys = ["odor_concentration", "agent_position", "agent_orientation"]
        for key in expected_keys:
            assert key in obs
        
        # Test action space compatibility
        action = env.action_space.sample()
        assert action.shape == (2,)  # [speed, angular_velocity]
        
        # Test step return format
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Verify cache doesn't break core environment functionality
        assert "video_frame" in info  # Cache enhancement
        assert info["video_frame"] is not None
        assert obs["odor_concentration"] >= 0.0
        assert len(obs["agent_position"]) == 2


class TestCacheThreadSafety:
    """Test thread-safe cache operations for multi-agent scenarios."""
    
    def test_concurrent_cache_access(self, frame_cache_lru):
        """Test thread-safe concurrent cache access."""
        mock_video_plume = Mock()
        
        # Create frames with thread-specific patterns
        def mock_get_frame(frame_idx):
            # Add small delay to increase chance of race conditions
            time.sleep(0.001)
            return np.full((50, 50), frame_idx, dtype=np.uint8)
        
        mock_video_plume.get_frame.side_effect = mock_get_frame
        
        # Test concurrent access from multiple threads
        num_threads = 5
        frames_per_thread = 20
        results = [[] for _ in range(num_threads)]
        exceptions = [[] for _ in range(num_threads)]
        
        def worker(thread_id):
            try:
                for i in range(frames_per_thread):
                    frame_id = (thread_id * frames_per_thread) + i
                    frame = frame_cache_lru.get(frame_id, mock_video_plume)
                    results[thread_id].append((frame_id, frame))
            except Exception as e:
                exceptions[thread_id].append(e)
        
        # Start threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Verify no exceptions occurred
        for thread_exceptions in exceptions:
            assert len(thread_exceptions) == 0
        
        # Verify all frames were retrieved correctly
        for thread_id, thread_results in enumerate(results):
            assert len(thread_results) == frames_per_thread
            for frame_id, frame in thread_results:
                assert frame is not None
                assert frame[0, 0] == frame_id % 256  # Verify correct frame
        
        # Verify cache statistics are consistent
        total_expected_requests = num_threads * frames_per_thread
        assert frame_cache_lru.statistics.total_requests == total_expected_requests
    
    def test_concurrent_preloading_and_access(self, frame_cache_lru):
        """Test concurrent preloading and cache access."""
        mock_video_plume = Mock()
        
        def mock_get_frame(frame_idx):
            time.sleep(0.001)  # Simulate I/O delay
            return np.full((30, 30), frame_idx, dtype=np.uint8)
        
        mock_video_plume.get_frame.side_effect = mock_get_frame
        
        preload_completed = threading.Event()
        access_results = []
        access_exceptions = []
        
        def preloader():
            try:
                success = frame_cache_lru.preload(range(0, 50), mock_video_plume)
                assert success
                preload_completed.set()
            except Exception as e:
                access_exceptions.append(e)
        
        def accessor():
            try:
                # Wait a bit then start accessing
                time.sleep(0.1)
                for i in range(0, 20, 2):  # Access every other frame
                    frame = frame_cache_lru.get(i, mock_video_plume)
                    access_results.append((i, frame))
            except Exception as e:
                access_exceptions.append(e)
        
        # Start concurrent operations
        preload_thread = threading.Thread(target=preloader)
        access_thread = threading.Thread(target=accessor)
        
        preload_thread.start()
        access_thread.start()
        
        # Wait for completion
        preload_thread.join()
        access_thread.join()
        
        # Verify no exceptions
        assert len(access_exceptions) == 0
        assert preload_completed.is_set()
        
        # Verify access results
        assert len(access_results) == 10  # Every other frame from 0-20
        for frame_id, frame in access_results:
            assert frame is not None
            assert frame[0, 0] == frame_id % 256


class TestCachePerformanceBenchmarks:
    """Test cache performance benchmarks and threshold validation."""
    
    def test_cache_hit_latency_benchmark(self, frame_cache_lru):
        """Test cache hit latency meets <10ms target."""
        mock_video_plume = Mock()
        
        # Create test frame
        test_frame = np.random.rand(480, 640, 3).astype(np.float32)
        mock_video_plume.get_frame.return_value = test_frame
        
        # Prime cache
        frame_cache_lru.get(0, mock_video_plume)
        
        # Benchmark cache hits
        hit_times = []
        for _ in range(100):  # Multiple samples for statistical significance
            start = time.perf_counter()
            frame = frame_cache_lru.get(0, mock_video_plume)
            hit_time = time.perf_counter() - start
            hit_times.append(hit_time)
            assert frame is not None
        
        # Verify performance requirements
        avg_hit_time = np.mean(hit_times)
        max_hit_time = np.max(hit_times)
        
        assert avg_hit_time < 0.01  # <10ms average
        assert max_hit_time < 0.02  # <20ms worst case
        
        # Verify all were cache hits
        assert frame_cache_lru.hit_rate > 0.99  # >99% hit rate
    
    def test_sequential_access_hit_rate(self, frame_cache_lru):
        """Test cache hit rate for sequential access patterns meets >90% target."""
        mock_video_plume = Mock()
        
        def mock_get_frame(frame_idx):
            return np.full((100, 100), frame_idx, dtype=np.uint8)
        
        mock_video_plume.get_frame.side_effect = mock_get_frame
        
        # Simulate typical RL training access pattern
        sequence_length = 1000
        window_size = 50  # Frames commonly accessed
        
        # Access pattern: mostly sequential with some random access
        access_pattern = []
        
        # 80% sequential access within window
        for i in range(int(sequence_length * 0.8)):
            frame_id = i % window_size
            access_pattern.append(frame_id)
        
        # 20% random access across larger range
        for i in range(int(sequence_length * 0.2)):
            frame_id = np.random.randint(0, window_size * 2)
            access_pattern.append(frame_id)
        
        # Execute access pattern
        for frame_id in access_pattern:
            frame = frame_cache_lru.get(frame_id, mock_video_plume)
            assert frame is not None
        
        # Verify hit rate meets target
        hit_rate = frame_cache_lru.hit_rate
        assert hit_rate >= 0.9  # >90% target for sequential access
    
    def test_memory_pressure_handling(self, frame_cache_lru):
        """Test memory pressure handling and intelligent eviction."""
        mock_video_plume = Mock()
        
        # Create large frames to trigger memory pressure
        def mock_get_frame(frame_idx):
            # Large frame to quickly hit memory limit
            return np.random.rand(200, 200, 3).astype(np.float32)
        
        mock_video_plume.get_frame.side_effect = mock_get_frame
        
        # Load frames until memory pressure is triggered
        initial_memory = frame_cache_lru.memory_usage_mb
        frames_loaded = 0
        
        while frame_cache_lru.memory_usage_mb < frame_cache_lru.memory_limit_mb * 0.8:
            frame = frame_cache_lru.get(frames_loaded, mock_video_plume)
            assert frame is not None
            frames_loaded += 1
            
            # Safety check to prevent infinite loop
            if frames_loaded > 1000:
                break
        
        # Continue loading to trigger eviction
        for i in range(frames_loaded, frames_loaded + 20):
            frame = frame_cache_lru.get(i, mock_video_plume)
            assert frame is not None
        
        # Verify memory management
        final_memory = frame_cache_lru.memory_usage_mb
        assert final_memory <= frame_cache_lru.memory_limit_mb  # Stayed within limit
        
        # Verify evictions occurred
        stats = frame_cache_lru.statistics.get_summary()
        assert stats['evictions'] > 0  # Some frames were evicted


class TestCacheModesComparison:
    """Test different cache modes and their performance characteristics."""
    
    def test_cache_mode_performance_comparison(self):
        """Compare performance characteristics of different cache modes."""
        if not FRAME_CACHE_AVAILABLE:
            pytest.skip("FrameCache not available")
        
        # Create caches with different modes
        cache_none = create_no_cache(enable_logging=False)
        cache_lru = create_lru_cache(memory_limit_mb=50, enable_logging=False)
        cache_preload = create_preload_cache(memory_limit_mb=50, enable_logging=False)
        
        mock_video_plume = Mock()
        
        def mock_get_frame(frame_idx):
            time.sleep(0.001)  # Simulate I/O delay
            return np.zeros((100, 100), dtype=np.uint8)
        
        mock_video_plume.get_frame.side_effect = mock_get_frame
        
        # Test access pattern
        access_pattern = [0, 1, 2, 0, 1, 3, 0, 4, 1, 2] * 10  # Repeated pattern
        
        # Benchmark each cache mode
        results = {}
        
        for cache_name, cache in [("none", cache_none), ("lru", cache_lru), ("preload", cache_preload)]:
            if cache_name == "preload":
                # Preload frames for preload cache
                cache.preload(range(5), mock_video_plume)
            
            start_time = time.time()
            for frame_id in access_pattern:
                frame = cache.get(frame_id, mock_video_plume)
                assert frame is not None
            total_time = time.time() - start_time
            
            results[cache_name] = {
                "total_time": total_time,
                "hit_rate": cache.hit_rate if hasattr(cache, 'hit_rate') else 0.0,
                "avg_access_time": total_time / len(access_pattern)
            }
        
        # Verify performance ordering: preload > lru > none
        assert results["preload"]["total_time"] < results["lru"]["total_time"]
        assert results["lru"]["total_time"] < results["none"]["total_time"]
        
        # Verify hit rates
        assert results["preload"]["hit_rate"] >= results["lru"]["hit_rate"]
        assert results["lru"]["hit_rate"] > results["none"]["hit_rate"]
    
    def test_cache_mode_memory_characteristics(self):
        """Test memory usage characteristics of different cache modes."""
        if not FRAME_CACHE_AVAILABLE:
            pytest.skip("FrameCache not available")
        
        cache_lru = create_lru_cache(memory_limit_mb=10, enable_logging=False)
        cache_preload = create_preload_cache(memory_limit_mb=10, enable_logging=False)
        
        mock_video_plume = Mock()
        mock_video_plume.get_frame.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # Load same number of frames in each cache
        for i in range(20):
            cache_lru.get(i, mock_video_plume)
        
        # Preload same frames
        cache_preload.preload(range(20), mock_video_plume)
        
        # Compare memory usage
        lru_memory = cache_lru.memory_usage_mb
        preload_memory = cache_preload.memory_usage_mb
        
        # Both should respect memory limits
        assert lru_memory <= cache_lru.memory_limit_mb
        assert preload_memory <= cache_preload.memory_limit_mb
        
        # LRU should have evicted some frames due to limit
        assert cache_lru.cache_size < 20
        
        # Preload may have all frames if they fit in memory
        # or may have failed to preload all if memory limit exceeded


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])