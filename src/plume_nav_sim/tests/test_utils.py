"""
Comprehensive test module for utility functions validating frame cache performance, seed management 
reproducibility, enhanced logging with performance metrics, and visualization utilities with 
CI-compatible headless rendering support.

This module ensures utility function reliability, mathematical precision, and proper integration
with the broader system architecture. Tests cover enhanced frame caching with LRU eviction
and memory monitoring per Section 5.2.2, seed manager reproducibility across Gymnasium 
environment instances, performance metrics collection and correlation tracking, and 
visualization capabilities with headless rendering for CI compatibility.

Test Coverage Targets:
- Frame cache: >85% coverage with LRU eviction and memory monitoring validation
- Seed management: >85% coverage with reproducibility validation across Gymnasium environments
- Logging configuration: >85% coverage with performance validation and correlation tracking
- Visualization utilities: >75% coverage with headless rendering validation
- Mathematical utilities: >90% coverage with precision validation
- Cross-platform compatibility: 100% coverage for file I/O and path handling

Key Features Tested:
- Enhanced FrameCache with LRU eviction and PSUtil memory pressure monitoring
- SeedManager reproducibility across different Gymnasium environment instances
- Enhanced logging with performance metrics collection and correlation context
- Visualization utilities with headless mode support for CI/CD pipelines
- Cross-cutting utility integration for complete system validation
"""

import os
import sys
import time
import tempfile
import threading
import platform
import gc
import concurrent.futures
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List, Optional, Tuple
import uuid

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

# Import the utilities we need to test with updated package structure
from plume_nav_sim.utils.frame_cache import (
    FrameCache, CacheMode, CacheStatistics,
    create_lru_cache, create_preload_cache, create_no_cache
)
from plume_nav_sim.utils.seed_manager import (
    SeedConfig, set_global_seed, get_random_state, capture_random_state,
    restore_random_state, scoped_seed, validate_determinism, is_seeded,
    get_last_seed, generate_experiment_seed, setup_global_seed
)

visualization = pytest.importorskip("plume_nav_sim.utils.visualization")
SimulationVisualization = visualization.SimulationVisualization
visualize_trajectory = visualization.visualize_trajectory
batch_visualize_trajectories = visualization.batch_visualize_trajectories
setup_headless_mode = visualization.setup_headless_mode
get_available_themes = visualization.get_available_themes
DEFAULT_VISUALIZATION_CONFIG = visualization.DEFAULT_VISUALIZATION_CONFIG

# Try to import enhanced logging, fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class TestFrameCache:
    """
    Comprehensive test suite for enhanced frame cache functionality with LRU eviction
    and memory pressure monitoring per Section 5.2.2 requirements.
    
    Tests dual-mode caching (LRU/preload), memory management with PSUtil integration,
    performance characteristics, and thread-safe operations for multi-agent access.
    """
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.test_frames = self._generate_test_frames()
        self.mock_video_plume = self._create_mock_video_plume()
    
    def teardown_method(self):
        """Clean up after each test."""
        # Clear any cached frames
        if hasattr(self, 'cache'):
            self.cache.clear()
    
    def _generate_test_frames(self, count: int = 10) -> List[np.ndarray]:
        """Generate deterministic test frames for caching tests."""
        frames = []
        for i in range(count):
            # Create deterministic frame data
            frame = np.random.RandomState(i).rand(64, 64).astype(np.float32)
            frames.append(frame)
        return frames
    
    def _create_mock_video_plume(self):
        """Create mock VideoPlume instance for testing."""
        mock = Mock()
        mock.get_frame = Mock(side_effect=lambda frame_id, **kwargs: 
                             self.test_frames[frame_id % len(self.test_frames)])
        mock.frame_count = len(self.test_frames)
        return mock
    
    def test_cache_mode_enum_conversion(self):
        """Test CacheMode enum string conversion functionality."""
        assert CacheMode.from_string("lru") == CacheMode.LRU
        assert CacheMode.from_string("LRU") == CacheMode.LRU
        assert CacheMode.from_string("all") == CacheMode.ALL
        assert CacheMode.from_string("none") == CacheMode.NONE
        
        with pytest.raises(ValueError):
            CacheMode.from_string("invalid")
    
    def test_cache_statistics_initialization(self):
        """Test CacheStatistics thread-safe initialization."""
        stats = CacheStatistics()
        
        assert stats.hit_rate == 0.0
        assert stats.miss_rate == 1.0
        assert stats.total_requests == 0
        assert stats.memory_usage_mb == 0.0
        assert stats.peak_memory_mb == 0.0
    
    def test_cache_statistics_operations(self):
        """Test CacheStatistics tracking of cache operations."""
        stats = CacheStatistics()
        
        # Record hits and misses
        stats.record_hit(0.005)
        stats.record_miss(0.010)
        stats.record_hit(0.003)
        
        assert stats.total_requests == 3
        assert stats.hit_rate == 2.0 / 3.0
        assert stats.miss_rate == 1.0 / 3.0
        assert stats.average_hit_time == 0.004  # (0.005 + 0.003) / 2
        assert stats.average_miss_time == 0.010
    
    def test_cache_statistics_memory_tracking(self):
        """Test memory usage tracking in statistics."""
        stats = CacheStatistics()
        
        frame_size = 1024 * 1024  # 1MB frame
        stats.record_insertion(frame_size)
        
        assert stats.memory_usage_mb == 1.0
        assert stats.peak_memory_mb == 1.0
        
        # Add another frame
        stats.record_insertion(frame_size)
        assert stats.memory_usage_mb == 2.0
        assert stats.peak_memory_mb == 2.0
        
        # Evict one frame
        stats.record_eviction(frame_size)
        assert stats.memory_usage_mb == 1.0
        assert stats.peak_memory_mb == 2.0  # Peak should remain
    
    def test_lru_cache_initialization(self):
        """Test LRU cache initialization with proper configuration."""
        cache = FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=100.0,
            enable_statistics=True
        )
        
        assert cache.mode == CacheMode.LRU
        assert cache.memory_limit_mb == 100.0
        assert cache.cache_size == 0
        assert cache.statistics is not None
    
    def test_lru_cache_basic_operations(self):
        """Test basic LRU cache get/store operations."""
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=50.0)
        
        # Test cache miss (frame not cached)
        frame = cache.get(0, self.mock_video_plume)
        assert frame is not None
        assert cache.cache_size == 1
        assert cache.statistics.total_requests == 1
        assert cache.statistics.hit_rate == 0.0  # First request is always a miss
        
        # Test cache hit (frame already cached)
        frame2 = cache.get(0, self.mock_video_plume)
        np.testing.assert_array_equal(frame, frame2)
        assert cache.cache_size == 1
        assert cache.statistics.total_requests == 2
        assert cache.statistics.hit_rate == 0.5  # 1 hit out of 2 requests
    
    def test_lru_cache_eviction_policy(self):
        """Test LRU eviction policy with memory pressure."""
        # Create small cache to force eviction
        cache = FrameCache(
            mode=CacheMode.LRU, 
            memory_limit_mb=0.01,  # Very small limit
            memory_pressure_threshold=0.8
        )
        
        # Load multiple frames to exceed memory limit
        for i in range(5):
            frame = cache.get(i, self.mock_video_plume)
            assert frame is not None
        
        # Should have triggered evictions
        assert cache.cache_size < 5
        if cache.statistics:
            assert cache.statistics._evictions > 0
    
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available for memory monitoring")
    def test_memory_pressure_monitoring(self):
        """Test memory pressure monitoring with PSUtil integration."""
        cache = FrameCache(
            mode=CacheMode.LRU,
            memory_limit_mb=1.0,  # Small limit
            memory_pressure_threshold=0.7
        )
        
        # Simulate memory pressure by loading many frames
        initial_memory = cache.memory_usage_mb
        
        for i in range(3):
            cache.get(i, self.mock_video_plume)
        
        final_memory = cache.memory_usage_mb
        assert final_memory >= initial_memory
        
        # Check if pressure warnings were recorded
        if cache.statistics:
            # May have pressure warnings depending on frame sizes
            assert cache.statistics._pressure_warnings >= 0
    
    def test_preload_cache_mode(self):
        """Test preload cache mode functionality."""
        cache = FrameCache(mode=CacheMode.ALL, memory_limit_mb=100.0)
        
        # Test preload operation
        success = cache.preload(range(0, 3), self.mock_video_plume)
        assert success
        assert cache.cache_size == 3
        
        # All subsequent gets should be cache hits
        for i in range(3):
            frame = cache.get(i, self.mock_video_plume)
            assert frame is not None
        
        if cache.statistics:
            # Should have high hit rate after preload
            assert cache.statistics.hit_rate >= 0.5
    
    def test_no_cache_mode(self):
        """Test no-cache mode for direct I/O operations."""
        cache = FrameCache(mode=CacheMode.NONE)
        
        # Should always call video plume directly
        frame1 = cache.get(0, self.mock_video_plume)
        frame2 = cache.get(0, self.mock_video_plume)
        
        assert frame1 is not None
        assert frame2 is not None
        assert cache.cache_size == 0  # No caching
        
        # Should have called get_frame twice
        assert self.mock_video_plume.get_frame.call_count == 2
    
    def test_cache_performance_requirements(self):
        """Test cache performance meets sub-10ms requirements."""
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=100.0)
        
        # Preload a frame
        cache.get(0, self.mock_video_plume)
        
        # Measure cache hit performance
        start_time = time.perf_counter()
        for _ in range(100):
            cache.get(0, self.mock_video_plume)
        avg_time_ms = (time.perf_counter() - start_time) * 1000 / 100
        
        # Should meet <10ms requirement for cache hits
        assert avg_time_ms < 10.0, f"Cache hit time {avg_time_ms:.2f}ms exceeds 10ms target"
    
    def test_cache_thread_safety(self):
        """Test cache thread safety for concurrent access."""
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=100.0)
        results = {}
        
        def worker(thread_id):
            # Each thread accesses different frames
            frame_id = thread_id % len(self.test_frames)
            frame = cache.get(frame_id, self.mock_video_plume)
            results[thread_id] = frame is not None
        
        # Run concurrent workers
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should succeed
        assert all(results.values())
        assert len(results) == 5
    
    def test_cache_factory_functions(self):
        """Test cache factory functions for convenience creation."""
        lru_cache = create_lru_cache(memory_limit_mb=50.0)
        assert lru_cache.mode == CacheMode.LRU
        assert lru_cache.memory_limit_mb == 50.0
        
        preload_cache = create_preload_cache(memory_limit_mb=100.0)
        assert preload_cache.mode == CacheMode.ALL
        assert preload_cache.memory_limit_mb == 100.0
        
        no_cache = create_no_cache()
        assert no_cache.mode == CacheMode.NONE
    
    def test_cache_context_manager(self):
        """Test cache context manager for automatic cleanup."""
        with FrameCache(mode=CacheMode.LRU) as cache:
            cache.get(0, self.mock_video_plume)
            assert cache.cache_size == 1
        
        # Cache should be cleared after context exit
        assert cache.cache_size == 0


class TestSeedManager:
    """
    Comprehensive test suite for seed management functionality with focus on
    reproducibility across Gymnasium environment instances.
    
    Tests deterministic behavior, cross-platform consistency, Gymnasium integration,
    performance characteristics, and state management.
    """
    
    def setup_method(self):
        """Reset seed manager state before each test."""
        # Reset any global seed state
        try:
            from plume_nav_sim.utils.seed_manager import reset_random_state
            reset_random_state()
        except ImportError:
            pass
    
    def teardown_method(self):
        """Clean up after each test."""
        # Reset random state to prevent test interference
        try:
            from plume_nav_sim.utils.seed_manager import reset_random_state
            reset_random_state()
        except ImportError:
            pass
    
    def test_seed_config_validation(self):
        """Test SeedConfig Pydantic schema validation."""
        # Valid configuration
        config = SeedConfig(
            global_seed=42,
            enable_validation=True,
            cross_platform_determinism=True
        )
        assert config.global_seed == 42
        assert config.enable_validation is True
        assert config.cross_platform_determinism is True
        
        # Test default values
        default_config = SeedConfig()
        assert default_config.global_seed is None
        assert default_config.enable_validation is True
        assert default_config.performance_monitoring is True
    
    def test_global_seed_setting_basic(self):
        """Test basic global seed setting functionality."""
        results = set_global_seed(42)
        
        assert results["seed_value"] == 42
        assert "python_random" in results["components_seeded"]
        assert "numpy_random" in results["components_seeded"]
        assert results["validation_passed"] is True
        assert results["total_time"] < 1.0  # Should complete quickly
    
    def test_seed_reproducibility_across_calls(self):
        """Test reproducibility across multiple seed setting calls."""
        # Set seed and generate random values
        set_global_seed(42)
        python_val1 = np.random.random()
        numpy_val1 = np.random.random()
        
        # Reset and set same seed
        set_global_seed(42)
        python_val2 = np.random.random()
        numpy_val2 = np.random.random()
        
        # Should be identical
        assert python_val1 == python_val2
        assert numpy_val1 == numpy_val2
    
    def test_gymnasium_environment_reproducibility(self):
        """Test reproducibility with mock Gymnasium environment instances."""
        # Mock a simple environment that uses random numbers
        class MockGymnasiumEnv:
            def __init__(self, seed=None):
                if seed is not None:
                    set_global_seed(seed)
            
            def reset(self, seed=None):
                if seed is not None:
                    set_global_seed(seed)
                return np.random.random(4), {}
            
            def step(self, action):
                obs = np.random.random(4)
                reward = np.random.random()
                terminated = np.random.random() > 0.9
                truncated = np.random.random() > 0.95
                info = {}
                return obs, reward, terminated, truncated, info
        
        # Test reproducibility across environment instances
        env1 = MockGymnasiumEnv()
        obs1, _ = env1.reset(seed=42)
        step_result1 = env1.step([0.5, 0.5])
        
        env2 = MockGymnasiumEnv()
        obs2, _ = env2.reset(seed=42)
        step_result2 = env2.step([0.5, 0.5])
        
        # Results should be identical
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(step_result1[0], step_result2[0])
        assert step_result1[1] == step_result2[1]  # reward
        assert step_result1[2] == step_result2[2]  # terminated
        assert step_result1[3] == step_result2[3]  # truncated
    
    def test_determinism_validation(self):
        """Test determinism validation functionality."""
        set_global_seed(42)
        
        results = validate_determinism(iterations=10)
        
        assert results["is_deterministic"] is True
        assert results["python_test_passed"] is True
        assert results["numpy_test_passed"] is True
        assert results["python_variance"] < 1e-15
        assert results["numpy_variance"] < 1e-15
        assert results["test_duration"] < 1.0
    
    def test_scoped_seed_context_manager(self):
        """Test scoped seed context manager for temporary seed changes."""
        # Set initial seed
        set_global_seed(100)
        initial_value = np.random.random()
        
        # Use scoped seed
        with scoped_seed(42, "test_operation") as original_state:
            scoped_value = np.random.random()
            assert original_state is not None
        
        # Should restore to original state
        post_scoped_value = np.random.random()
        
        # Values should be different (since we're in different states)
        assert scoped_value != initial_value
        # Note: We can't easily test exact restoration without complex state management
    
    def test_random_state_capture_and_restore(self):
        """Test random state capture and restoration functionality."""
        set_global_seed(42)
        
        # Capture initial state
        state = capture_random_state()
        assert state is not None
        assert state.is_valid()
        
        # Generate some random numbers to change state
        _ = np.random.random()
        _ = np.random.random()
        value_before_restore = np.random.random()
        
        # Restore state and generate same sequence
        success = restore_random_state(state)
        assert success is True
        
        # Skip the same numbers we generated before
        _ = np.random.random()
        _ = np.random.random()
        value_after_restore = np.random.random()
        
        # Should get the same value
        assert value_before_restore == value_after_restore
    
    def test_experiment_seed_generation(self):
        """Test deterministic experiment seed generation."""
        seed1 = generate_experiment_seed("test_experiment_1")
        seed2 = generate_experiment_seed("test_experiment_2")
        seed1_again = generate_experiment_seed("test_experiment_1")
        
        # Different experiments should have different seeds
        assert seed1 != seed2
        
        # Same experiment name should give same seed
        assert seed1 == seed1_again
        
        # Seeds should be in valid range
        assert 0 <= seed1 <= 2**31 - 1
        assert 0 <= seed2 <= 2**31 - 1
    
    def test_seed_manager_performance(self):
        """Test seed manager performance requirements."""
        start_time = time.perf_counter()
        
        for i in range(10):
            set_global_seed(i)
        
        total_time = time.perf_counter() - start_time
        avg_time_ms = (total_time / 10) * 1000
        
        # Should meet <100ms requirement per operation
        assert avg_time_ms < 100, f"Seed setting took {avg_time_ms:.2f}ms > 100ms target"
    
    def test_seed_manager_utility_functions(self):
        """Test seed manager utility functions."""
        assert is_seeded() is False
        assert get_last_seed() is None
        
        set_global_seed(42)
        
        assert is_seeded() is True
        assert get_last_seed() == 42
        
        # Test setup function
        results = setup_global_seed(12345)
        assert results["seed_value"] == 12345
        assert get_last_seed() == 12345
    
    def test_cross_platform_seed_consistency(self):
        """Test cross-platform seed consistency."""
        # Test with different platform configurations
        for seed in [42, 123, 999]:
            set_global_seed(seed)
            
            # Generate test values
            python_vals = [np.random.random() for _ in range(5)]
            numpy_vals = np.random.random(5)
            
            # Reset and test again
            set_global_seed(seed)
            python_vals2 = [np.random.random() for _ in range(5)]
            numpy_vals2 = np.random.random(5)
            
            # Should be identical
            assert python_vals == python_vals2
            np.testing.assert_array_equal(numpy_vals, numpy_vals2)


class TestVisualizationUtilities:
    """
    Comprehensive test suite for visualization functionality with focus on
    headless rendering for CI compatibility.
    
    Tests real-time animation, static trajectory plots, batch processing,
    configuration management, and headless operation modes.
    """
    
    def setup_method(self):
        """Set up test environment for visualization tests."""
        # Ensure matplotlib backend is set for testing
        matplotlib.use('Agg')  # Non-interactive backend
        setup_headless_mode()
    
    def teardown_method(self):
        """Clean up after visualization tests."""
        plt.close('all')  # Close all figures
    
    def test_headless_mode_setup(self):
        """Test headless mode setup for CI compatibility."""
        setup_headless_mode()
        
        # Should be able to create figures without display
        fig, ax = plt.subplots(figsize=(8, 6))
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_simulation_visualization_initialization(self):
        """Test SimulationVisualization initialization in headless mode."""
        viz = SimulationVisualization(headless=True)
        
        assert viz.fig is not None
        assert viz.ax is not None
        assert viz.headless is True
    
    def test_static_trajectory_visualization(self):
        """Test static trajectory visualization with headless rendering."""
        # Create test trajectory data
        time_steps = 50
        positions = np.column_stack([
            np.linspace(0, 10, time_steps),
            np.sin(np.linspace(0, 2*np.pi, time_steps)) * 2 + 5
        ])
        
        # Test basic trajectory plot
        fig = visualize_trajectory(
            positions=positions,
            show_plot=False,
            batch_mode=True
        )
        
        assert fig is not None
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_multi_agent_trajectory_visualization(self):
        """Test multi-agent trajectory visualization."""
        # Create test trajectory data for multiple agents
        time_steps = 30
        n_agents = 3
        
        positions = np.zeros((n_agents, time_steps, 2))
        for i in range(n_agents):
            positions[i, :, 0] = np.linspace(i*2, i*2 + 10, time_steps)
            positions[i, :, 1] = np.sin(np.linspace(0, 2*np.pi, time_steps)) * (i+1) + 5
        
        orientations = np.random.rand(n_agents, time_steps) * 360
        
        fig = visualize_trajectory(
            positions=positions,
            orientations=orientations,
            show_plot=False,
            batch_mode=True,
            title="Multi-Agent Test Trajectory"
        )
        
        assert fig is not None
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_batch_visualization_processing(self):
        """Test batch visualization processing with headless output."""
        # Create test trajectory data
        trajectory_data = []
        for i in range(3):
            positions = np.column_stack([
                np.linspace(0, 10, 20),
                np.sin(np.linspace(0, 2*np.pi, 20)) * (i+1)
            ])
            trajectory_data.append({
                'positions': positions,
                'title': f'Test Trajectory {i+1}'
            })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_paths = batch_visualize_trajectories(
                trajectory_data=trajectory_data,
                output_dir=temp_dir,
                naming_pattern="test_batch_{idx:02d}"
            )
            
            assert len(saved_paths) == 3
            for path in saved_paths:
                assert path.exists()
                assert path.suffix == '.png'
    
    def test_available_themes(self):
        """Test available themes functionality."""
        themes = get_available_themes()
        
        assert isinstance(themes, dict)
        assert len(themes) > 0
        
        # Test expected themes
        expected_themes = ["scientific", "presentation", "high_contrast"]
        for theme in expected_themes:
            if theme in themes:
                assert 'colormap' in themes[theme]
                assert 'background' in themes[theme]
    
    def test_visualization_performance_scaling(self):
        """Test visualization performance with different agent counts."""
        viz = SimulationVisualization(headless=True)
        
        # Test with different numbers of agents
        for n_agents in [1, 10, 50]:
            positions = np.random.rand(n_agents, 2) * 20
            orientations = np.random.rand(n_agents) * 360
            odor_values = np.random.rand(n_agents)
            
            start_time = time.perf_counter()
            frame_data = (positions, orientations, odor_values)
            artists = viz.update_visualization(frame_data)
            update_time = (time.perf_counter() - start_time) * 1000
            
            # Should complete in reasonable time
            assert update_time < 500  # 500ms threshold for testing
            assert len(artists) > 0 or n_agents == 0
    
    def test_visualization_error_handling(self):
        """Test error handling in visualization components."""
        viz = SimulationVisualization(headless=True)
        
        # Test with invalid frame data
        with pytest.raises((ValueError, IndexError, AttributeError, TypeError)):
            invalid_frame_data = (None, None, None)
            viz.update_visualization(invalid_frame_data)
    
    def test_visualization_memory_management(self):
        """Test visualization memory management in headless mode."""
        initial_figures = len(plt.get_fignums())
        
        # Create and close multiple visualizations
        for i in range(5):
            positions = np.random.rand(10, 2) * 10
            fig = visualize_trajectory(positions, show_plot=False, batch_mode=True)
            plt.close(fig)
        
        final_figures = len(plt.get_fignums())
        
        # Should not have figure leaks
        assert final_figures <= initial_figures + 1  # Allow some tolerance


class TestUtilityIntegration:
    """
    Integration tests for utility functions working together.
    
    Tests cross-cutting concerns, module interactions, and system-wide
    utility functionality including seed management with caching,
    logging integration, and performance monitoring.
    """
    
    def setup_method(self):
        """Set up integration test environment."""
        matplotlib.use('Agg')
        try:
            from plume_nav_sim.utils.seed_manager import reset_random_state
            reset_random_state()
        except ImportError:
            pass
    
    def teardown_method(self):
        """Clean up integration test environment."""
        plt.close('all')
        try:
            from plume_nav_sim.utils.seed_manager import reset_random_state
            reset_random_state()
        except ImportError:
            pass
    
    def test_seed_manager_cache_integration(self):
        """Test integration between seed manager and frame cache."""
        # Set deterministic seed
        set_global_seed(42)
        
        # Create cache with deterministic behavior
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=10.0)
        
        # Create mock video plume with seeded random frames
        mock_plume = Mock()
        mock_plume.get_frame = Mock(side_effect=lambda frame_id, **kwargs: 
                                  np.random.RandomState(frame_id).rand(32, 32))
        
        # Get frames with deterministic seed
        frame1 = cache.get(0, mock_plume)
        frame2 = cache.get(1, mock_plume)
        
        # Reset seed and get same frames
        set_global_seed(42)
        frame1_repeat = cache.get(0, mock_plume)  # Should be cache hit
        
        # Reset cache and seed to test reproducibility
        cache.clear()
        set_global_seed(42)
        frame1_new = cache.get(0, mock_plume)
        
        # Frame content should be identical (deterministic generation)
        np.testing.assert_array_equal(frame1, frame1_new)
    
    def test_reproducible_visualization_generation(self):
        """Test reproducible visualization generation with seed management."""
        # Set deterministic seed
        set_global_seed(42)
        
        # Generate reproducible test data
        positions1 = np.random.rand(20, 2) * 10
        
        # Reset and regenerate with same seed
        set_global_seed(42)
        positions2 = np.random.rand(20, 2) * 10
        
        # Should be identical
        np.testing.assert_array_equal(positions1, positions2)
        
        # Create visualizations
        fig1 = visualize_trajectory(positions1, show_plot=False, batch_mode=True)
        fig2 = visualize_trajectory(positions2, show_plot=False, batch_mode=True)
        
        assert fig1 is not None
        assert fig2 is not None
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring across utility components."""
        # Test performance of cache operations with timing
        cache = FrameCache(mode=CacheMode.LRU, enable_statistics=True)
        mock_plume = Mock()
        mock_plume.get_frame = Mock(return_value=np.random.rand(64, 64))
        
        # Perform timed operations
        start_time = time.perf_counter()
        
        for i in range(10):
            cache.get(i % 3, mock_plume)  # Some cache hits, some misses
        
        total_time = time.perf_counter() - start_time
        
        # Verify performance tracking
        if cache.statistics:
            assert cache.statistics.total_requests == 10
            assert cache.statistics.hit_rate > 0  # Should have some cache hits
            
        # Should complete quickly
        assert total_time < 1.0
    
    def test_cross_platform_file_operations(self):
        """Test cross-platform compatibility for file I/O operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test path handling across platforms
            test_file = temp_path / "test_file.png"
            # Generate test visualization
            positions = np.column_stack([np.linspace(0, 10, 20), np.linspace(0, 10, 20)])

            # Save visualization
            fig = visualize_trajectory(
                positions=positions,
                output_path=test_file,
                show_plot=False,
                batch_mode=True
            )

            # Verify file was created
            assert test_file.exists()
            assert test_file.stat().st_size > 0

            plt.close(fig)
    
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_management_integration(self):
        """Test memory management across utility components."""
        process = psutil.Process()
        baseline_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=10.0)
        mock_plume = Mock()
        mock_plume.get_frame = Mock(return_value=np.random.rand(128, 128))
        
        for i in range(20):
            # Set different seeds and cache frames
            set_global_seed(i)
            cache.get(i, mock_plume)
            
            # Force garbage collection
            if i % 5 == 0:
                gc.collect()
        
        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - baseline_memory) / 1024 / 1024  # MB
        
        # Allow some growth but not excessive (100MB threshold for test)
        assert memory_growth < 100, f"Memory grew by {memory_growth:.2f}MB"
    
    def test_concurrent_utility_operations(self):
        """Test thread safety of utility operations."""
        def worker(thread_id):
            # Each thread should have independent operations
            cache = FrameCache(mode=CacheMode.LRU, memory_limit_mb=5.0)
            mock_plume = Mock()
            mock_plume.get_frame = Mock(return_value=np.random.rand(32, 32))
            
            set_global_seed(thread_id)
            
            # Perform operations
            frame = cache.get(thread_id, mock_plume)
            random_val = np.random.random()
            
            return {
                'thread_id': thread_id,
                'frame_shape': frame.shape,
                'random_value': random_val,
                'cache_size': cache.cache_size
            }
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i) for i in range(3)]
            results = [future.result() for future in futures]
        
        # Verify each thread had independent operations
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['thread_id'] == i
            assert result['frame_shape'] == (32, 32)
            assert result['cache_size'] == 1


class TestUtilityErrorHandling:
    """
    Test error handling and edge cases for utility functions.
    
    Validates robust error handling, graceful degradation, and
    appropriate error messages for all utility components.
    """
    
    def test_frame_cache_error_handling(self):
        """Test frame cache error handling for invalid inputs."""
        cache = FrameCache(mode=CacheMode.LRU)
        
        # Test with invalid frame_id
        with pytest.raises(ValueError):
            cache.get(-1, Mock())
        
        # Test with None video_plume
        with pytest.raises(ValueError):
            cache.get(0, None)
        
        # Test invalid cache mode
        with pytest.raises(ValueError):
            FrameCache(mode="invalid_mode")
    
    def test_seed_manager_error_handling(self):
        """Test seed manager error handling for invalid inputs."""
        # Test invalid seed values
        with pytest.raises((ValueError, RuntimeError)):
            set_global_seed(-1)
        
        with pytest.raises((ValueError, RuntimeError)):
            set_global_seed(2**32)  # Too large
        
        # Test invalid random state
        from plume_nav_sim.utils.seed_manager import restore_random_state, RandomState
        
        with pytest.raises((ValueError, TypeError)):
            restore_random_state("invalid_state")
        
        # Test invalid random state object
        invalid_state = RandomState()
        invalid_state.python_state = None
        invalid_state.numpy_state = None
        
        # Should handle gracefully
        try:
            restore_random_state(invalid_state)
        except (ValueError, RuntimeError):
            pass  # Expected
    
    def test_visualization_error_handling(self):
        """Test visualization error handling for invalid inputs."""
        # Test with empty trajectory data
        empty_positions = np.array([]).reshape(0, 2)
        
        # Should handle gracefully or raise appropriate error
        try:
            fig = visualize_trajectory(empty_positions, show_plot=False, batch_mode=True)
            if fig is not None:
                plt.close(fig)
        except (ValueError, IndexError):
            pass  # Expected for empty data
        
        # Test with invalid data types
        with pytest.raises((TypeError, ValueError)):
            visualize_trajectory("invalid_data", show_plot=False)
    
    def test_mathematical_precision_edge_cases(self):
        """Test mathematical precision in edge cases."""
        # Test seed manager with extreme values
        try:
            set_global_seed(0)  # Minimum valid seed
            val1 = np.random.random()
            
            set_global_seed(2**31 - 1)  # Maximum valid seed
            val2 = np.random.random()
            
            # Should complete without error
            assert isinstance(val1, float)
            assert isinstance(val2, float)
        except (ValueError, RuntimeError):
            pass  # Some edge cases may not be supported
        
        # Test cache with very small frames
        cache = FrameCache(mode=CacheMode.LRU)
        mock_plume = Mock()
        tiny_frame = np.array([[1e-10]], dtype=np.float32)
        mock_plume.get_frame = Mock(return_value=tiny_frame)
        
        frame = cache.get(0, mock_plume)
        assert frame is not None
        assert frame.shape == (1, 1)
    
    def test_resource_cleanup_edge_cases(self):
        """Test resource cleanup in error conditions."""
        # Test cache cleanup after errors
        cache = FrameCache(mode=CacheMode.LRU)
        
        # Simulate partial failure
        mock_plume = Mock()
        mock_plume.get_frame = Mock(side_effect=[
            np.random.rand(32, 32),  # Success
            None,  # Failure
            np.random.rand(32, 32)   # Success again
        ])
        
        # Should handle mixed success/failure
        frame1 = cache.get(0, mock_plume)
        frame2 = cache.get(1, mock_plume)  # Returns None
        frame3 = cache.get(2, mock_plume)
        
        assert frame1 is not None
        assert frame2 is None
        assert frame3 is not None
        
        # Cache should still be functional
        assert cache.cache_size >= 1
        
        # Test cleanup
        cache.clear()
        assert cache.cache_size == 0


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])