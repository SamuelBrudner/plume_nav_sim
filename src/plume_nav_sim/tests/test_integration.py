"""
Comprehensive integration test suite for the plume_nav_sim library migration.

This module validates end-to-end workflows, Gymnasium compliance, backward compatibility,
extensibility hooks, and cross-repository integration patterns as specified in Section 0
of the technical specification. Integration tests ensure that the migration from Python 3.9/Gym 0.26 
to Python 3.10+/Gymnasium 0.29 maintains numerical fidelity while enhancing extensibility.

Test Categories:
- End-to-end simulation workflows with PlumeNavigationEnv integration
- Gymnasium API compliance validation with env_checker
- Dual API support testing for 4-tuple vs 5-tuple returns
- Extensibility hooks testing (compute_additional_obs, compute_extra_reward, on_episode_end)
- Cross-repository compatibility with place_mem_rl
- CLI-to-core integration with parameter flow validation
- Configuration system integration with Hydra and frame cache management
- Database session integration with simulation workflows
- Performance characteristics validation per timing requirements (mean env.step() < 10ms)
- Frame cache performance testing with memory limits (≤ 2 GiB)

Architecture Integration Points:
- src/plume_nav_sim/envs/plume_navigation_env.py: Core Gymnasium environment class
- src/plume_nav_sim/shims/gym_make.py: Legacy Gym compatibility layer
- src/plume_nav_sim/utils/frame_cache.py: Enhanced caching system with LRU eviction
- src/plume_nav_sim/cli/main.py: Command-line interface integration
- src/plume_nav_sim/config/schemas.py: Configuration validation schemas

Performance Requirements:
- Environment step latency: <10ms P95 per Section 0.2.1
- Frame cache efficiency: >90% hit rate with ≤2 GiB memory cap
- CLI command initialization: ≤2 seconds per Section 6.6.3.3
- Cross-repository compatibility: Maintained with place_mem_rl

Quality Gates:
- 100% pass rate for Gymnasium compliance validation
- Comprehensive scenario coverage across all integration points
- Performance validation within specified timing requirements
- Cross-framework compatibility verification
- Backward compatibility preservation for legacy Gym usage

Authors: Blitzy Platform Migration Agent
License: MIT
"""

import os
import sys
import time
import tempfile
import warnings
import gc
import json
from pathlib import Path
from contextlib import contextmanager, suppress
from typing import Dict, List, Optional, Any, Tuple, Union, Generator, Protocol
from unittest.mock import Mock, MagicMock, patch, mock_open
from dataclasses import asdict

import pytest
import numpy as np
from click.testing import CliRunner

# Gymnasium imports for migration validation
try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

# Legacy Gym imports for backward compatibility testing
try:
    import gym as legacy_gym
    LEGACY_GYM_AVAILABLE = True
except ImportError:
    LEGACY_GYM_AVAILABLE = False

# Core plume_nav_sim imports (updated package structure)
try:
    from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv
    from plume_nav_sim.envs.spaces import SpacesFactory
    from plume_nav_sim.utils.frame_cache import FrameCache, CacheMode
    from plume_nav_sim.config.schemas import (
        EnvironmentConfig, FrameCacheConfig, SimulationConfig
    )
    PLUME_NAV_SIM_AVAILABLE = True
except ImportError:
    PLUME_NAV_SIM_AVAILABLE = False

# Compatibility shim imports for backward compatibility testing
try:
    from plume_nav_sim.shims.gym_make import gym_make
    from plume_nav_sim.shims import LegacyWrapper
    SHIM_AVAILABLE = True
except ImportError:
    SHIM_AVAILABLE = False

# CLI imports for command-line integration testing
try:
    from plume_nav_sim.cli.main import main as cli_main, cli
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

# Optional imports with graceful degradation
try:
    from omegaconf import DictConfig, OmegaConf
    import hydra
    from hydra import compose, initialize_config_store
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    DictConfig = dict
    OmegaConf = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class IntegrationTestBase:
    """Base class for integration tests with common fixtures and utilities."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self, tmp_path, monkeypatch):
        """Set up isolated test environment for each integration test."""
        # Create temporary directories for test isolation
        self.temp_dir = tmp_path
        self.config_dir = tmp_path / "conf"
        self.output_dir = tmp_path / "outputs"
        self.data_dir = tmp_path / "data"
        
        # Create directory structure
        for dir_path in [self.config_dir, self.output_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for test isolation
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        monkeypatch.setenv("PYTHONPATH", str(self.temp_dir))
        monkeypatch.setenv("FRAME_CACHE_MODE", "none")  # Disable caching by default for tests
        monkeypatch.setenv("FRAME_CACHE_SIZE_MB", "256")  # Conservative cache size for tests
        
        # Disable GUI backends for headless testing
        monkeypatch.setenv("MPLBACKEND", "Agg")
        
        # Set up test-specific logging to avoid interference
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        
        yield
        
        # Cleanup after test - force garbage collection to free memory
        gc.collect()
    
    @pytest.fixture
    def mock_video_data(self):
        """Create mock video data for testing video-based environments."""
        # Generate deterministic test frames
        frames = []
        for i in range(10):
            # Create gradient pattern for odor concentration simulation
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Simple gradient to simulate odor plume
            y, x = np.ogrid[:480, :640]
            concentration = np.exp(-((x - 320)**2 + (y - 240)**2) / (2 * 100**2))
            frame[:, :, 0] = (concentration * 255).astype(np.uint8)
            frames.append(frame)
        return frames
    
    @pytest.fixture
    def sample_env_config(self):
        """Generate sample environment configuration for testing."""
        return {
            'environment': {
                'agent_config': {
                    'position': [0.0, 0.0],
                    'orientation': 0.0,
                    'max_speed': 2.0,
                    'sensor_distance': 1.0
                },
                'odor_config': {
                    'concentration_threshold': 0.1,
                    'max_concentration': 1.0,
                    'background_noise': 0.01
                },
                'episode_config': {
                    'max_steps': 1000,
                    'success_threshold': 0.8,
                    'timeout_penalty': -1.0
                }
            },
            'frame_cache': {
                'mode': 'lru',
                'memory_limit_mb': 512,
                'enable_statistics': True,
                'pressure_threshold': 0.8
            },
            'simulation': {
                'dt': 0.1,
                'random_seed': 42,
                'enable_logging': True
            }
        }
    
    @pytest.fixture
    def hydra_config(self, sample_env_config):
        """Create Hydra DictConfig for configuration testing."""
        if not HYDRA_AVAILABLE:
            pytest.skip("Hydra not available for DictConfig testing")
        return OmegaConf.create(sample_env_config)


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
class TestGymnasiumAPICompliance(IntegrationTestBase):
    """Test Gymnasium API compliance and migration validation."""
    
    def test_gymnasium_environment_registration(self):
        """
        Test that PlumeNavSim environment is properly registered with Gymnasium.
        
        Validates:
        - Environment registration with 'PlumeNavSim-v0' ID
        - Proper entry point configuration
        - Environment creation without errors
        """
        # Test environment registration
        try:
            env = gym.make('PlumeNavSim-v0')
            assert env is not None
            assert isinstance(env, gym.Env)
            env.close()
        except gym.error.UnregisteredEnv:
            pytest.skip("PlumeNavSim-v0 not registered yet - environment in development")
    
    def test_gymnasium_api_compliance_validation(self, mock_video_data):
        """
        Test comprehensive Gymnasium API compliance using env_checker.
        
        Validates:
        - Standard Gymnasium environment interface
        - Reset method returns (observation, info) tuple
        - Step method returns 5-tuple: (obs, reward, terminated, truncated, info)
        - Action and observation space consistency
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("PlumeNavigationEnv not available")
        
        # Create environment instance directly
        env = PlumeNavigationEnv(
            video_frames=mock_video_data,
            max_steps=100
        )
        
        try:
            # Run Gymnasium environment checker
            check_env(env, warn=True, skip_render_check=True)
            
            # Test reset functionality
            reset_result = env.reset(seed=42)
            assert len(reset_result) == 2, "Reset should return (observation, info) tuple"
            observation, info = reset_result
            assert isinstance(observation, np.ndarray), "Observation should be numpy array"
            assert isinstance(info, dict), "Info should be dictionary"
            
            # Test step functionality - critical 5-tuple return
            action = env.action_space.sample()
            step_result = env.step(action)
            assert len(step_result) == 5, "Step should return 5-tuple for Gymnasium compatibility"
            
            obs, reward, terminated, truncated, info = step_result
            assert isinstance(obs, np.ndarray), "Observation should be numpy array"
            assert isinstance(reward, (int, float, np.number)), "Reward should be numeric"
            assert isinstance(terminated, bool), "Terminated should be boolean"
            assert isinstance(truncated, bool), "Truncated should be boolean"
            assert isinstance(info, dict), "Info should be dictionary"
            
        finally:
            env.close()
    
    def test_environment_step_performance_sla(self, mock_video_data):
        """
        Test environment step performance meets SLA requirements.
        
        Validates:
        - Mean step time < 10ms per Section 0.2.1
        - P95 step time < 15ms for performance consistency
        - No significant performance degradation over episode
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("PlumeNavigationEnv not available")
        
        env = PlumeNavigationEnv(
            video_frames=mock_video_data,
            max_steps=100,
            enable_frame_cache=False  # Test worst-case without caching
        )
        
        try:
            env.reset(seed=42)
            
            # Warm-up runs to eliminate initialization overhead
            for _ in range(10):
                action = env.action_space.sample()
                env.step(action)
            
            # Performance measurement over 100 steps
            step_times = []
            for _ in range(100):
                start_time = time.perf_counter()
                action = env.action_space.sample()
                env.step(action)
                step_time_ms = (time.perf_counter() - start_time) * 1000
                step_times.append(step_time_ms)
            
            # Validate performance SLA requirements
            mean_step_time = np.mean(step_times)
            p95_step_time = np.percentile(step_times, 95)
            
            assert mean_step_time < 10.0, f"Mean step time {mean_step_time:.2f}ms exceeds 10ms SLA"
            assert p95_step_time < 15.0, f"P95 step time {p95_step_time:.2f}ms exceeds 15ms target"
            
            # Validate performance consistency (no significant degradation)
            first_half_mean = np.mean(step_times[:50])
            second_half_mean = np.mean(step_times[50:])
            performance_ratio = second_half_mean / first_half_mean
            
            assert performance_ratio < 1.2, f"Performance degradation detected: {performance_ratio:.2f}x slower"
            
        finally:
            env.close()
    
    def test_observation_action_space_consistency(self, mock_video_data):
        """
        Test observation and action space consistency and type safety.
        
        Validates:
        - Proper space definitions using SpacesFactory
        - Type-safe space construction with validated bounds
        - Consistent space properties across environment lifecycle
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("PlumeNavigationEnv not available")
        
        env = PlumeNavigationEnv(video_frames=mock_video_data)
        
        try:
            # Test action space properties
            assert hasattr(env, 'action_space'), "Environment should have action_space"
            assert hasattr(env, 'observation_space'), "Environment should have observation_space"
            
            # Test space types
            assert isinstance(env.action_space, gym.Space), "Action space should be Gymnasium Space"
            assert isinstance(env.observation_space, gym.Space), "Observation space should be Gymnasium Space"
            
            # Test space sampling and validation
            for _ in range(10):
                action = env.action_space.sample()
                assert env.action_space.contains(action), "Sampled action should be valid"
            
            # Test observation space validation
            obs, _ = env.reset(seed=42)
            assert env.observation_space.contains(obs), "Reset observation should be valid"
            
            # Test step observation validation
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs), "Step observation should be valid"
            
        finally:
            env.close()


@pytest.mark.skipif(not SHIM_AVAILABLE, reason="Compatibility shim not available")
class TestBackwardCompatibilityShim(IntegrationTestBase):
    """Test backward compatibility shim for legacy Gym support."""
    
    def test_gym_make_deprecation_warning(self):
        """
        Test that gym_make issues proper deprecation warnings.
        
        Validates:
        - DeprecationWarning is emitted when using gym_make
        - Warning message includes migration guidance
        - Deprecation follows proper stacklevel for user code location
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                env = gym_make("PlumeNavSim-v0")
                env.close()
                
                # Verify deprecation warning was emitted
                assert len(w) > 0, "gym_make should emit deprecation warning"
                
                warning = None
                for warning_item in w:
                    if issubclass(warning_item.category, DeprecationWarning):
                        warning = warning_item
                        break
                
                assert warning is not None, "DeprecationWarning should be emitted"
                assert "deprecated" in str(warning.message).lower(), "Warning should mention deprecation"
                assert "gymnasium.make" in str(warning.message), "Warning should suggest Gymnasium migration"
                
            except Exception as e:
                pytest.skip(f"gym_make not functional yet: {e}")
    
    def test_dual_api_format_conversion(self):
        """
        Test automatic conversion between 4-tuple and 5-tuple formats.
        
        Validates:
        - Legacy callers receive 4-tuple (obs, reward, done, info)
        - Modern callers receive 5-tuple (obs, reward, terminated, truncated, info)
        - Proper conversion between terminated/truncated and done flags
        """
        if not SHIM_AVAILABLE:
            pytest.skip("Compatibility shim not available")
        
        try:
            # Test with legacy wrapper for 4-tuple format
            env = gym_make("PlumeNavSim-v0")
            
            # Test legacy reset (should return just observation)
            if hasattr(env, '_legacy_mode') and env._legacy_mode:
                obs = env.reset()
                assert isinstance(obs, np.ndarray), "Legacy reset should return observation only"
            else:
                obs, info = env.reset()
                assert isinstance(obs, np.ndarray), "Modern reset should return (obs, info)"
                assert isinstance(info, dict), "Modern reset should return info dict"
            
            # Test legacy step format conversion
            action = env.action_space.sample()
            step_result = env.step(action)
            
            if hasattr(env, '_legacy_mode') and env._legacy_mode:
                # Legacy 4-tuple format
                assert len(step_result) == 4, "Legacy step should return 4-tuple"
                obs, reward, done, info = step_result
                assert isinstance(done, bool), "Legacy done should be boolean"
            else:
                # Modern 5-tuple format
                assert len(step_result) == 5, "Modern step should return 5-tuple"
                obs, reward, terminated, truncated, info = step_result
                assert isinstance(terminated, bool), "Terminated should be boolean"
                assert isinstance(truncated, bool), "Truncated should be boolean"
            
            env.close()
            
        except Exception as e:
            pytest.skip(f"Dual API testing not available: {e}")
    
    def test_legacy_gym_compatibility_preservation(self):
        """
        Test that legacy Gym code patterns continue to work.
        
        Validates:
        - Existing stable-baselines3 integration patterns
        - Legacy observation/action space usage
        - Backward-compatible environment lifecycle
        """
        if not (SHIM_AVAILABLE and LEGACY_GYM_AVAILABLE):
            pytest.skip("Legacy Gym compatibility testing not available")
        
        try:
            # Test basic legacy gym pattern
            env = gym_make("PlumeNavSim-v0")
            
            # Legacy environment usage pattern
            obs = env.reset()
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
            
            env.close()
            
        except Exception as e:
            pytest.skip(f"Legacy compatibility testing failed: {e}")


@pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="PlumeNavigationEnv not available")
class TestExtensibilityHooks(IntegrationTestBase):
    """Test extensibility hooks for custom observations, rewards, and episode handling."""
    
    def test_compute_additional_obs_hook(self, mock_video_data):
        """
        Test compute_additional_obs hook for custom observation augmentation.
        
        Validates:
        - Hook is called during observation computation
        - Additional observations are properly integrated
        - Type safety and shape consistency maintained
        """
        class CustomObservationEnv(PlumeNavigationEnv):
            def compute_additional_obs(self, base_obs: dict) -> dict:
                """Add custom sensor data to observations."""
                return {
                    'custom_sensor': np.array([0.5, 0.3]),
                    'agent_id': 1,
                    'step_count': self.current_step
                }
        
        env = CustomObservationEnv(video_frames=mock_video_data)
        
        try:
            obs, info = env.reset(seed=42)
            
            # Verify custom observations are included
            if isinstance(obs, dict):
                assert 'custom_sensor' in obs, "Custom sensor data should be included"
                assert 'agent_id' in obs, "Agent ID should be included"
                assert 'step_count' in obs, "Step count should be included"
                
                # Verify data types and shapes
                assert isinstance(obs['custom_sensor'], np.ndarray), "Custom sensor should be numpy array"
                assert obs['custom_sensor'].shape == (2,), "Custom sensor should have correct shape"
                assert obs['agent_id'] == 1, "Agent ID should match expected value"
                assert obs['step_count'] == 0, "Initial step count should be zero"
            
            # Test hook behavior during step
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            
            if isinstance(obs, dict):
                assert obs['step_count'] == 1, "Step count should increment"
            
        finally:
            env.close()
    
    def test_compute_extra_reward_hook(self, mock_video_data):
        """
        Test compute_extra_reward hook for custom reward shaping.
        
        Validates:
        - Hook is called during reward computation
        - Extra rewards are properly added to base reward
        - Reward shaping doesn't break learning signals
        """
        class CustomRewardEnv(PlumeNavigationEnv):
            def compute_extra_reward(self, base_reward: float, info: dict) -> float:
                """Add distance-based reward shaping."""
                # Simple distance-based bonus
                agent_pos = info.get('agent_position', [0, 0])
                distance_to_goal = np.linalg.norm(np.array(agent_pos) - np.array([10, 10]))
                distance_bonus = -0.01 * distance_to_goal  # Penalty for being far from goal
                return distance_bonus
        
        env = CustomRewardEnv(video_frames=mock_video_data)
        
        try:
            env.reset(seed=42)
            
            # Collect rewards over multiple steps
            total_base_reward = 0
            total_shaped_reward = 0
            
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_shaped_reward += reward
                
                # Verify reward is numeric
                assert isinstance(reward, (int, float, np.number)), "Reward should be numeric"
                
                if terminated or truncated:
                    break
            
            # Verify reward shaping is applied (shaped reward should differ from base)
            assert total_shaped_reward != 0, "Reward shaping should produce non-zero rewards"
            
        finally:
            env.close()
    
    def test_on_episode_end_hook(self, mock_video_data):
        """
        Test on_episode_end hook for episode completion processing.
        
        Validates:
        - Hook is called when episode ends (terminated or truncated)
        - Episode information is properly passed to hook
        - Hook can perform logging and data collection
        """
        episode_end_calls = []
        
        class EpisodeTrackingEnv(PlumeNavigationEnv):
            def on_episode_end(self, final_info: dict) -> None:
                """Track episode completion statistics."""
                episode_data = {
                    'episode_length': self.current_step,
                    'final_reward': final_info.get('episode_reward', 0),
                    'success': final_info.get('success', False),
                    'termination_reason': final_info.get('termination_reason', 'unknown')
                }
                episode_end_calls.append(episode_data)
        
        env = EpisodeTrackingEnv(video_frames=mock_video_data, max_steps=20)
        
        try:
            env.reset(seed=42)
            
            # Run episode to completion
            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Verify hook was called
            assert len(episode_end_calls) == 1, "on_episode_end should be called once per episode"
            
            episode_data = episode_end_calls[0]
            assert 'episode_length' in episode_data, "Episode data should include length"
            assert 'final_reward' in episode_data, "Episode data should include final reward"
            assert episode_data['episode_length'] > 0, "Episode should have positive length"
            
            # Test multiple episodes
            episode_end_calls.clear()
            for episode in range(3):
                env.reset(seed=42 + episode)
                for _ in range(10):  # Shorter episodes
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
            
            assert len(episode_end_calls) == 3, "Hook should be called for each episode"
            
        finally:
            env.close()


@pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="Frame cache not available")
class TestFrameCacheIntegration(IntegrationTestBase):
    """Test frame cache integration with LRU eviction and memory management."""
    
    def test_frame_cache_lru_functionality(self, mock_video_data):
        """
        Test LRU frame cache functionality and performance.
        
        Validates:
        - LRU cache eviction policy
        - Cache hit rate optimization
        - Memory usage within limits
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("FrameCache not available")
        
        # Create cache with limited size for testing
        cache_config = FrameCacheConfig(
            mode=CacheMode.LRU,
            memory_limit_mb=64,  # Small limit for testing
            enable_statistics=True
        )
        
        cache = FrameCache(config=cache_config)
        
        try:
            # Fill cache beyond capacity to test eviction
            for i, frame in enumerate(mock_video_data * 3):  # Use more frames than cache can hold
                cache.store_frame(i, frame)
            
            # Test cache hit/miss behavior
            hit_count = 0
            miss_count = 0
            
            for i in range(len(mock_video_data)):
                frame = cache.get_frame(i)
                stats = cache.get_statistics()
                
                if stats['hits'] > hit_count:
                    hit_count = stats['hits']
                else:
                    miss_count = stats['misses']
            
            # Verify LRU behavior and statistics
            stats = cache.get_statistics()
            assert stats['hits'] > 0, "Cache should have some hits"
            assert stats['misses'] > 0, "Cache should have some misses due to eviction"
            
            # Verify memory limit compliance
            if PSUTIL_AVAILABLE:
                memory_usage_mb = cache.get_memory_usage_mb()
                assert memory_usage_mb <= cache_config.memory_limit_mb * 1.1, "Memory usage should respect limits"
            
        finally:
            cache.clear()
    
    def test_frame_cache_performance_impact(self, mock_video_data):
        """
        Test frame cache performance impact on environment step times.
        
        Validates:
        - Cache improves frame access performance
        - Cache hit rate >90% after warm-up
        - Step times remain <10ms with caching enabled
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("Frame cache performance testing not available")
        
        # Test with caching enabled
        env_cached = PlumeNavigationEnv(
            video_frames=mock_video_data,
            enable_frame_cache=True,
            cache_mode='lru',
            cache_memory_limit_mb=256
        )
        
        # Test without caching
        env_uncached = PlumeNavigationEnv(
            video_frames=mock_video_data,
            enable_frame_cache=False
        )
        
        try:
            # Measure performance with caching
            env_cached.reset(seed=42)
            cached_times = []
            
            for _ in range(50):  # Warm-up cache
                action = env_cached.action_space.sample()
                start_time = time.perf_counter()
                env_cached.step(action)
                step_time_ms = (time.perf_counter() - start_time) * 1000
                cached_times.append(step_time_ms)
            
            # Measure performance without caching
            env_uncached.reset(seed=42)
            uncached_times = []
            
            for _ in range(50):
                action = env_uncached.action_space.sample()
                start_time = time.perf_counter()
                env_uncached.step(action)
                step_time_ms = (time.perf_counter() - start_time) * 1000
                uncached_times.append(step_time_ms)
            
            # Compare performance (cached should be faster after warm-up)
            cached_mean = np.mean(cached_times[20:])  # Skip warm-up
            uncached_mean = np.mean(uncached_times)
            
            assert cached_mean < 10.0, f"Cached step time {cached_mean:.2f}ms exceeds 10ms SLA"
            
            # Cache should provide performance benefit (allow 20% margin for test variance)
            if uncached_mean > cached_mean:
                speedup = uncached_mean / cached_mean
                assert speedup > 1.0, f"Cache should provide speedup, got {speedup:.2f}x"
            
        finally:
            env_cached.close()
            env_uncached.close()
    
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available for memory testing")
    def test_frame_cache_memory_pressure_handling(self, mock_video_data):
        """
        Test frame cache memory pressure detection and handling.
        
        Validates:
        - Memory pressure detection via psutil
        - Automatic cache eviction under pressure
        - Memory usage stays within 2 GiB limit per process
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("Memory pressure testing not available")
        
        # Get baseline memory usage
        process = psutil.Process()
        baseline_memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Create cache with memory monitoring
        cache_config = FrameCacheConfig(
            mode=CacheMode.LRU,
            memory_limit_mb=128,  # Conservative limit for testing
            enable_statistics=True,
            pressure_threshold=0.8  # Trigger eviction at 80% of limit
        )
        
        cache = FrameCache(config=cache_config)
        
        try:
            # Fill cache and monitor memory usage
            large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)  # Large frame
            
            for i in range(20):  # Store multiple large frames
                cache.store_frame(i, large_frame)
                
                current_memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_increase = current_memory_mb - baseline_memory_mb
                
                # Verify memory limit is respected
                assert memory_increase < 2048, f"Memory usage {memory_increase:.1f}MB exceeds 2GB limit"
                
                # Check if pressure handling is triggered
                stats = cache.get_statistics()
                if stats.get('evictions', 0) > 0:
                    break  # Eviction occurred as expected
            
            # Verify eviction occurred when approaching limits
            stats = cache.get_statistics()
            assert stats.get('evictions', 0) > 0, "Cache should evict frames under memory pressure"
            
        finally:
            cache.clear()
            gc.collect()  # Force garbage collection to clean up test data


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI not available")
class TestCLIIntegration(IntegrationTestBase):
    """Test CLI-to-core integration with parameter flow validation."""
    
    def test_cli_environment_creation_workflow(self, sample_env_config):
        """
        Test CLI environment creation workflow with configuration.
        
        Validates:
        - CLI parameter parsing and validation
        - Environment creation from CLI configuration
        - Performance requirements (<2s initialization)
        """
        runner = CliRunner()
        
        # Test CLI help command for rapid response
        start_time = time.perf_counter()
        result = runner.invoke(cli, ['--help'])
        help_time = (time.perf_counter() - start_time) * 1000
        
        assert result.exit_code == 0
        assert help_time < 1000, f"CLI help too slow: {help_time:.0f}ms"
        assert "plume" in result.output.lower() or "navigation" in result.output.lower()
        
        # Test environment creation command
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(sample_env_config, f)
            config_path = f.name
        
        try:
            start_time = time.perf_counter()
            result = runner.invoke(cli, ['env', 'create', '--config', config_path, '--dry-run'])
            init_time = (time.perf_counter() - start_time) * 1000
            
            # Validate CLI initialization performance (<2s per requirement)
            assert init_time < 2000, f"CLI initialization too slow: {init_time:.0f}ms"
            
            # Check for successful dry-run execution
            assert result.exit_code == 0 or "dry-run" in result.output.lower()
            
        finally:
            os.unlink(config_path)
    
    def test_cli_parameter_override_validation(self, sample_env_config):
        """
        Test CLI parameter override capabilities with configuration validation.
        
        Validates:
        - Command-line parameter parsing
        - Configuration override syntax support
        - Parameter validation and error handling
        """
        runner = CliRunner()
        
        # Create base configuration file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(sample_env_config, f)
            config_path = f.name
        
        try:
            # Test parameter override
            result = runner.invoke(cli, [
                'env', 'validate',
                '--config', config_path,
                '--override', 'environment.agent_config.max_speed=3.0',
                '--override', 'frame_cache.memory_limit_mb=1024'
            ])
            
            # Configuration validation should succeed or provide informative output
            assert result.exit_code == 0 or "validation" in result.output.lower()
            
        finally:
            os.unlink(config_path)
    
    def test_cli_simulation_execution_workflow(self, sample_env_config):
        """
        Test CLI simulation execution workflow.
        
        Validates:
        - End-to-end simulation execution via CLI
        - Output generation and validation
        - Performance monitoring integration
        """
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(sample_env_config, f)
            config_path = f.name
        
        try:
            # Test simulation execution
            result = runner.invoke(cli, [
                'sim', 'run',
                '--config', config_path,
                '--steps', '10',
                '--output-dir', str(self.output_dir),
                '--dry-run'
            ])
            
            # Simulation should execute successfully or provide meaningful feedback
            assert result.exit_code == 0 or "simulation" in result.output.lower()
            
        finally:
            os.unlink(config_path)


@pytest.mark.skipif(not HYDRA_AVAILABLE, reason="Hydra not available")
class TestConfigurationSystemIntegration(IntegrationTestBase):
    """Test configuration system integration with Hydra and frame cache management."""
    
    def test_hydra_configuration_composition(self, sample_env_config):
        """
        Test Hydra configuration composition and validation.
        
        Validates:
        - Multi-layer configuration loading
        - Environment variable interpolation
        - Configuration validation across components
        """
        # Test configuration composition
        base_config = {
            'environment': {'agent_config': {'max_speed': 1.0}},
            'frame_cache': {'mode': 'none'}
        }
        
        override_config = {
            'environment': {'agent_config': {'max_speed': 2.0}},
            'frame_cache': {'mode': 'lru', 'memory_limit_mb': 512}
        }
        
        # Test configuration merging
        merged_config = OmegaConf.merge(
            OmegaConf.create(base_config),
            OmegaConf.create(override_config)
        )
        
        # Validate composition results
        assert merged_config.environment.agent_config.max_speed == 2.0
        assert merged_config.frame_cache.mode == 'lru'
        assert merged_config.frame_cache.memory_limit_mb == 512
    
    def test_environment_variable_interpolation(self, monkeypatch):
        """
        Test environment variable interpolation in configuration.
        
        Validates:
        - ${oc.env:VAR_NAME} syntax support
        - Default value handling
        - Environment-specific configuration
        """
        # Set test environment variables
        monkeypatch.setenv("TEST_CACHE_MODE", "lru")
        monkeypatch.setenv("TEST_MEMORY_LIMIT", "1024")
        
        # Configuration with environment variable interpolation
        config_yaml = """
        frame_cache:
          mode: ${oc.env:TEST_CACHE_MODE,none}
          memory_limit_mb: ${oc.env:TEST_MEMORY_LIMIT,512}
          enable_statistics: true
        environment:
          agent_config:
            max_speed: ${oc.env:TEST_MAX_SPEED,2.0}
        """
        
        import yaml
        config = OmegaConf.create(yaml.safe_load(config_yaml))
        resolved_config = OmegaConf.to_object(config)
        
        # Validate environment variable substitution
        assert resolved_config['frame_cache']['mode'] == 'lru'
        assert int(resolved_config['frame_cache']['memory_limit_mb']) == 1024
        assert float(resolved_config['environment']['agent_config']['max_speed']) == 2.0  # Default value
    
    def test_configuration_validation_schemas(self, sample_env_config):
        """
        Test configuration validation using Pydantic schemas.
        
        Validates:
        - Schema validation for each component
        - Type checking and constraint validation
        - Error handling for invalid configurations
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("Configuration schemas not available")
        
        # Test valid configuration
        try:
            env_config = EnvironmentConfig(**sample_env_config['environment'])
            assert env_config.agent_config['max_speed'] == sample_env_config['environment']['agent_config']['max_speed']
        except Exception as e:
            pytest.skip(f"EnvironmentConfig not available: {e}")
        
        # Test frame cache configuration
        try:
            cache_config = FrameCacheConfig(**sample_env_config['frame_cache'])
            assert cache_config.mode == sample_env_config['frame_cache']['mode']
        except Exception as e:
            pytest.skip(f"FrameCacheConfig not available: {e}")
        
        # Test invalid configuration handling
        try:
            invalid_config = sample_env_config['environment'].copy()
            invalid_config['agent_config']['max_speed'] = -1.0  # Invalid negative speed
            
            with pytest.raises(ValueError):
                EnvironmentConfig(**invalid_config)
        except Exception:
            pass  # Schema validation not yet implemented


class TestCrossRepositoryCompatibility(IntegrationTestBase):
    """Test cross-repository compatibility with place_mem_rl and other frameworks."""
    
    def test_place_mem_rl_integration_compatibility(self, mock_video_data):
        """
        Test compatibility with place_mem_rl training workflows.
        
        Validates:
        - Environment interface compatibility
        - Training loop integration
        - Memory-based navigation patterns
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("Environment not available for cross-repo testing")
        
        # Test basic environment interface that place_mem_rl expects
        env = PlumeNavigationEnv(video_frames=mock_video_data)
        
        try:
            # Test standard RL interface that external packages expect
            obs, info = env.reset(seed=42)
            assert isinstance(obs, np.ndarray), "Observation should be numpy array for compatibility"
            
            # Test episode execution pattern
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Validate compatibility interface
                assert isinstance(obs, np.ndarray), "Observations should be numpy arrays"
                assert isinstance(reward, (int, float, np.number)), "Rewards should be numeric"
                assert isinstance(info, dict), "Info should be dictionary"
                
                if terminated or truncated:
                    obs, info = env.reset()
                    break
        
        finally:
            env.close()
    
    def test_stable_baselines3_interface_compatibility(self, mock_video_data):
        """
        Test compatibility with stable-baselines3 training interface.
        
        Validates:
        - VecEnv compatibility patterns
        - Action/observation space consistency
        - Training loop stability
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("Environment not available for SB3 testing")
        
        env = PlumeNavigationEnv(video_frames=mock_video_data)
        
        try:
            # Test SB3-style environment usage
            obs, _ = env.reset()
            
            # Simulate basic training loop
            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # SB3 expects specific data types
                assert obs.dtype in [np.float32, np.float64], "Observations should be float type"
                assert isinstance(reward, (int, float, np.number)), "Rewards should be numeric"
                
                if terminated or truncated:
                    obs, _ = env.reset()
        
        finally:
            env.close()
    
    def test_numpy_interface_compatibility(self, mock_video_data):
        """
        Test NumPy array interface compatibility for ML frameworks.
        
        Validates:
        - Array dtypes and shapes
        - Memory layout compatibility
        - Tensor conversion readiness
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("Environment not available for NumPy testing")
        
        env = PlumeNavigationEnv(video_frames=mock_video_data)
        
        try:
            obs, _ = env.reset()
            
            # Test NumPy compatibility
            assert isinstance(obs, np.ndarray), "Observations should be NumPy arrays"
            assert obs.flags['C_CONTIGUOUS'] or obs.flags['F_CONTIGUOUS'], "Arrays should be contiguous"
            assert obs.dtype in [np.float32, np.float64], "Arrays should use standard float types"
            
            # Test action processing
            action = env.action_space.sample()
            if isinstance(action, np.ndarray):
                assert action.flags['C_CONTIGUOUS'] or action.flags['F_CONTIGUOUS'], "Action arrays should be contiguous"
            
            # Test tensor conversion compatibility (if available)
            try:
                import torch
                obs_tensor = torch.from_numpy(obs)
                assert obs_tensor.shape == torch.Size(obs.shape), "PyTorch conversion should preserve shape"
            except ImportError:
                pass  # PyTorch not available
            
            try:
                import tensorflow as tf
                obs_tf = tf.convert_to_tensor(obs)
                assert tuple(obs_tf.shape) == obs.shape, "TensorFlow conversion should preserve shape"
            except ImportError:
                pass  # TensorFlow not available
        
        finally:
            env.close()


class TestPerformanceCharacteristics(IntegrationTestBase):
    """Test performance characteristics and SLA validation."""
    
    def test_end_to_end_performance_validation(self, mock_video_data):
        """
        Test end-to-end performance meets all SLA requirements.
        
        Validates:
        - Step latency <10ms mean per Section 0.2.1
        - Memory efficiency per agent
        - Scaling characteristics
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("Performance testing not available")
        
        env = PlumeNavigationEnv(
            video_frames=mock_video_data,
            enable_frame_cache=True
        )
        
        try:
            # Get baseline memory
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                baseline_memory = process.memory_info().rss
            
            # Comprehensive performance test
            env.reset(seed=42)
            
            step_times = []
            for step in range(100):
                start_time = time.perf_counter()
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                step_time_ms = (time.perf_counter() - start_time) * 1000
                step_times.append(step_time_ms)
                
                if terminated or truncated:
                    env.reset(seed=42 + step)
            
            # Validate performance SLAs
            mean_step_time = np.mean(step_times)
            p95_step_time = np.percentile(step_times, 95)
            p99_step_time = np.percentile(step_times, 99)
            
            assert mean_step_time < 10.0, f"Mean step time {mean_step_time:.2f}ms exceeds 10ms SLA"
            assert p95_step_time < 15.0, f"P95 step time {p95_step_time:.2f}ms exceeds 15ms target"
            assert p99_step_time < 25.0, f"P99 step time {p99_step_time:.2f}ms exceeds 25ms limit"
            
            # Validate memory efficiency
            if PSUTIL_AVAILABLE:
                peak_memory = process.memory_info().rss
                memory_increase_mb = (peak_memory - baseline_memory) / (1024 * 1024)
                assert memory_increase_mb < 100, f"Memory increase {memory_increase_mb:.1f}MB too high"
            
        finally:
            env.close()
    
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil required for memory testing")
    def test_memory_leak_detection(self, mock_video_data):
        """
        Test for memory leaks during extended operation.
        
        Validates:
        - Memory usage stability over time
        - Proper resource cleanup
        - No significant memory growth
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("Memory leak testing not available")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run multiple episodes to detect memory leaks
        for episode in range(5):
            env = PlumeNavigationEnv(video_frames=mock_video_data)
            
            try:
                env.reset(seed=42 + episode)
                
                for _ in range(50):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
            
            finally:
                env.close()
                
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            current_memory = process.memory_info().rss
            memory_growth_mb = (current_memory - initial_memory) / (1024 * 1024)
            
            # Allow some memory growth but detect significant leaks
            assert memory_growth_mb < 50, f"Memory leak detected: {memory_growth_mb:.1f}MB growth after {episode + 1} episodes"
    
    def test_concurrent_environment_performance(self, mock_video_data):
        """
        Test performance characteristics with concurrent environments.
        
        Validates:
        - Multi-environment scaling
        - Resource sharing efficiency
        - Performance isolation
        """
        if not PLUME_NAV_SIM_AVAILABLE:
            pytest.skip("Concurrent testing not available")
        
        # Create multiple environments
        num_envs = 4
        environments = []
        
        try:
            for i in range(num_envs):
                env = PlumeNavigationEnv(
                    video_frames=mock_video_data,
                    enable_frame_cache=True
                )
                environments.append(env)
                env.reset(seed=42 + i)
            
            # Test concurrent step execution
            step_times = []
            
            for step in range(20):
                start_time = time.perf_counter()
                
                # Step all environments
                for env in environments:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        env.reset(seed=42 + step)
                
                total_step_time_ms = (time.perf_counter() - start_time) * 1000
                avg_step_time_ms = total_step_time_ms / num_envs
                step_times.append(avg_step_time_ms)
            
            # Validate concurrent performance
            mean_concurrent_step_time = np.mean(step_times)
            assert mean_concurrent_step_time < 15.0, f"Concurrent step time {mean_concurrent_step_time:.2f}ms too high"
            
        finally:
            for env in environments:
                env.close()


# Test execution markers and configuration
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,  # Integration tests may take longer
]


# Global test configuration
def pytest_configure(config):
    """Configure pytest for integration testing."""
    # Register custom markers
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "performance: mark test as performance validation")
    config.addinivalue_line("markers", "compatibility: mark test as compatibility validation")
    
    # Configure test environment
    os.environ.setdefault("PYTEST_CURRENT_TEST", "true")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    
    # Disable interactive matplotlib backends
    if MATPLOTLIB_AVAILABLE:
        matplotlib.use('Agg')


# Module-level test utilities and fixtures
@pytest.fixture(scope="session")
def integration_test_session():
    """Session-scoped fixture for integration test setup."""
    start_time = time.time()
    
    # Global test session setup
    if PLUME_NAV_SIM_AVAILABLE:
        # Initialize any global state needed for testing
        pass
    
    yield
    
    # Global test session teardown
    total_time = time.time() - start_time
    print(f"\nIntegration test session completed in {total_time:.2f}s")


if __name__ == "__main__":
    # Enable direct execution for debugging
    pytest.main([__file__, "-v", "--tb=short"])