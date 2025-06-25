"""
Comprehensive test suite for Gymnasium 0.29.x API compliance and dual API support.

This module validates the enhanced RL environment interfaces per Section 0 requirements,
ensuring 100% API compliance with gymnasium.utils.env_checker while maintaining backward
compatibility with legacy gym usage patterns. Tests cover dual API support, environment
registration, performance requirements, and cross-repository integration scenarios.

Key Test Coverage:
- Gymnasium 0.29.x API compliance validation via env_checker
- Dual API support: 5-tuple (obs, reward, terminated, truncated, info) vs 4-tuple legacy
- New environment ID 'PlumeNavSim-v0' registration and functionality
- reset() method with seed parameter support per Gymnasium requirements
- Compatibility layer detection and proper step() return signature adaptation
- Performance validation: step() ≤10ms average execution time
- Property-based tests for API contract compliance across configurations
- Cross-repository integration scenarios for place_mem_rl compatibility

Architecture:
- pytest-based test framework with comprehensive fixtures
- Property-based testing using hypothesis for API contract validation
- Performance benchmarking with timing thresholds and statistical validation
- Mock video environments for consistent test execution
- Correlation ID tracking for enhanced debugging and test observability
- Comprehensive error handling and edge case validation

Technical Requirements Validated:
- F-004-RQ-005: Environment step() returns 5-tuple for Gymnasium callers
- F-004-RQ-006: reset(seed=...) parameter support with deterministic behavior
- F-004-RQ-007: Legacy gym callers receive 4-tuple without code changes
- F-004-RQ-008: New environment ID 'PlumeNavSim-v0' registration
- F-011-RQ-003: Performance warning if step() average >10ms
"""

from __future__ import annotations

import time
import warnings
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import uuid

import pytest
import numpy as np

# Property-based testing for API contract validation
try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, Bundle
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Mock decorators if hypothesis not available
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    st = Mock()
    settings = Mock()

# Core imports for environment testing
try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    from gymnasium.spaces import Box, Dict as DictSpace
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = Mock()
    check_env = Mock()
    Box = DictSpace = Mock()

# Legacy gym for compatibility testing
try:
    import gym as legacy_gym
    LEGACY_GYM_AVAILABLE = True
except ImportError:
    LEGACY_GYM_AVAILABLE = False
    legacy_gym = Mock()

# Test target imports
try:
    from odor_plume_nav.environments.gymnasium_env import (
        GymnasiumEnv, create_gymnasium_environment, validate_gymnasium_environment,
        _detect_legacy_gym_caller
    )
    from odor_plume_nav.environments.compat import (
        detect_api_version, format_step_return, create_compatibility_mode,
        wrap_environment, validate_compatibility, compatibility_context,
        CompatibilityWrapper, APIDetectionResult
    )
    from odor_plume_nav.environments import (
        register_environments, make_environment, get_available_environments,
        diagnose_environment_setup
    )
    MAIN_IMPORTS_AVAILABLE = True
except ImportError as e:
    MAIN_IMPORTS_AVAILABLE = False
    pytest.skip(f"Core environment modules not available: {e}", allow_module_level=True)

# Enhanced logging for test correlation tracking
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

# Test constants and configuration
PERFORMANCE_TARGET_MS = 10.0  # 10ms step() target from requirements
TEST_VIDEO_WIDTH = 640
TEST_VIDEO_HEIGHT = 480
TEST_VIDEO_FRAMES = 100
MAX_EPISODE_STEPS = 50  # Short episodes for faster testing
DEFAULT_TEST_SEED = 42

# Test correlation tracking
TEST_CORRELATION_ID = f"test_gymnasium_api_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def mock_video_file():
    """Create temporary mock video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        # Create minimal mock video data
        video_path = Path(f.name)
    
    yield video_path
    
    # Cleanup
    try:
        video_path.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_video_plume():
    """Mock VideoPlume class for consistent testing."""
    with patch('odor_plume_nav.environments.gymnasium_env.VideoPlume') as mock_class:
        mock_instance = Mock()
        mock_instance.get_metadata.return_value = {
            'width': TEST_VIDEO_WIDTH,
            'height': TEST_VIDEO_HEIGHT,
            'fps': 30.0,
            'frame_count': TEST_VIDEO_FRAMES
        }
        
        # Generate consistent mock frames
        def get_frame(frame_index):
            # Create deterministic frame data
            frame = np.random.RandomState(frame_index).rand(
                TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH
            ).astype(np.float32)
            return frame
        
        mock_instance.get_frame.side_effect = get_frame
        mock_instance.close.return_value = None
        mock_class.return_value = mock_instance
        
        yield mock_instance


@pytest.fixture
def mock_navigator():
    """Mock Navigator for controlled testing."""
    with patch('odor_plume_nav.environments.gymnasium_env.NavigatorFactory') as mock_factory:
        mock_navigator = Mock()
        
        # Set up deterministic navigator state
        mock_navigator.positions = np.array([[320.0, 240.0]], dtype=np.float32)
        mock_navigator.orientations = np.array([0.0], dtype=np.float32)
        mock_navigator.speeds = np.array([0.0], dtype=np.float32)
        mock_navigator.angular_velocities = np.array([0.0], dtype=np.float32)
        
        def reset(position, orientation, speed, angular_velocity):
            mock_navigator.positions[0] = np.array(position, dtype=np.float32)
            mock_navigator.orientations[0] = float(orientation)
            mock_navigator.speeds[0] = float(speed)
            mock_navigator.angular_velocities[0] = float(angular_velocity)
        
        mock_navigator.reset.side_effect = reset
        
        def step(frame, dt):
            # Simple movement simulation
            speed = mock_navigator.speeds[0]
            angle_rad = np.radians(mock_navigator.orientations[0])
            
            # Update position based on speed and orientation
            dx = speed * np.cos(angle_rad) * dt
            dy = speed * np.sin(angle_rad) * dt
            mock_navigator.positions[0] += [dx, dy]
            
            # Update orientation based on angular velocity
            mock_navigator.orientations[0] += mock_navigator.angular_velocities[0] * dt
            mock_navigator.orientations[0] = mock_navigator.orientations[0] % 360.0
        
        mock_navigator.step.side_effect = step
        
        def sample_odor(frame):
            # Return odor based on position (closer to center = higher odor)
            pos = mock_navigator.positions[0]
            center_x, center_y = TEST_VIDEO_WIDTH / 2, TEST_VIDEO_HEIGHT / 2
            distance = np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            odor = max(0.0, 1.0 - distance / max_distance)
            return odor
        
        mock_navigator.sample_odor.side_effect = sample_odor
        
        def sample_multiple_sensors(frame, **kwargs):
            # Return multiple sensor readings
            num_sensors = kwargs.get('num_sensors', 2)
            base_odor = sample_odor(frame)
            return np.array([base_odor + np.random.rand() * 0.1 for _ in range(num_sensors)])
        
        mock_navigator.sample_multiple_sensors.side_effect = sample_multiple_sensors
        
        mock_factory.single_agent.return_value = mock_navigator
        yield mock_navigator


@pytest.fixture
def test_config(mock_video_file):
    """Standard test configuration for environment creation."""
    return {
        'video_path': str(mock_video_file),
        'initial_position': (320, 240),
        'initial_orientation': 0.0,
        'max_speed': 2.0,
        'max_angular_velocity': 90.0,
        'max_episode_steps': MAX_EPISODE_STEPS,
        'performance_monitoring': True,
        'render_mode': None
    }


@pytest.fixture
def gymnasium_env(test_config, mock_video_plume, mock_navigator):
    """Create GymnasiumEnv instance for testing."""
    if not GYMNASIUM_AVAILABLE:
        pytest.skip("Gymnasium not available")
    
    env = GymnasiumEnv(**test_config)
    yield env
    env.close()


@pytest.fixture
def legacy_env(test_config, mock_video_plume, mock_navigator):
    """Create GymnasiumEnv in legacy compatibility mode."""
    if not GYMNASIUM_AVAILABLE:
        pytest.skip("Gymnasium not available")
    
    config_with_legacy = {**test_config, '_force_legacy_api': True}
    env = GymnasiumEnv(**config_with_legacy)
    yield env
    env.close()


class TestGymnasiumAPICompliance:
    """Test suite for Gymnasium 0.29.x API compliance validation."""
    
    def test_environment_api_checker_validation(self, gymnasium_env):
        """Test F-004-RQ-001: Environment passes gymnasium.utils.env_checker validation."""
        if not GYMNASIUM_AVAILABLE:
            pytest.skip("Gymnasium not available")
        
        with correlation_context("api_checker_validation", correlation_id=TEST_CORRELATION_ID) if ENHANCED_LOGGING else nullcontext():
            # Gymnasium env_checker performs comprehensive API validation
            try:
                check_env(gymnasium_env, warn=True, skip_render_check=True)
                logger.info(
                    "Environment passed gymnasium env_checker validation",
                    extra={
                        "metric_type": "api_compliance_success",
                        "test_type": "env_checker_validation"
                    }
                ) if ENHANCED_LOGGING else None
            except Exception as e:
                pytest.fail(f"Environment failed gymnasium env_checker validation: {e}")
    
    def test_environment_validates_own_compliance(self, gymnasium_env):
        """Test built-in API compliance validation method."""
        results = gymnasium_env.validate_api_compliance()
        
        assert results["compliant"], f"Self-validation failed: {results['errors']}"
        assert results["api_mode"] == "gymnasium"
        assert results["validation_time"] > 0
        assert isinstance(results["errors"], list)
        assert isinstance(results["warnings"], list)
    
    def test_environment_step_returns_five_tuple(self, gymnasium_env):
        """Test F-004-RQ-005: step() returns 5-tuple for Gymnasium callers."""
        obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        action = gymnasium_env.action_space.sample()
        step_result = gymnasium_env.step(action)
        
        # Verify 5-tuple format
        assert len(step_result) == 5, f"Expected 5-tuple, got {len(step_result)}-tuple"
        
        obs, reward, terminated, truncated, info = step_result
        
        # Validate types and structures
        assert gymnasium_env.observation_space.contains(obs), "Observation not in observation space"
        assert isinstance(reward, (int, float)), f"Reward must be numeric, got {type(reward)}"
        assert isinstance(terminated, bool), f"Terminated must be boolean, got {type(terminated)}"
        assert isinstance(truncated, bool), f"Truncated must be boolean, got {type(truncated)}"
        assert isinstance(info, dict), f"Info must be dict, got {type(info)}"
        
        logger.info(
            "Environment step returns valid 5-tuple",
            extra={
                "metric_type": "api_format_validation",
                "tuple_length": 5,
                "terminated": terminated,
                "truncated": truncated
            }
        ) if ENHANCED_LOGGING else None
    
    def test_environment_reset_with_seed_parameter(self, gymnasium_env):
        """Test F-004-RQ-006: reset(seed=...) parameter support with deterministic behavior."""
        # Test reset with seed
        obs1, info1 = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        # Verify return format
        assert gymnasium_env.observation_space.contains(obs1), "Reset observation not in space"
        assert isinstance(info1, dict), "Reset info must be dict"
        assert "seed" in info1, "Seed should be recorded in info"
        assert info1["seed"] == DEFAULT_TEST_SEED
        
        # Test deterministic behavior - same seed should give same initial state
        obs2, info2 = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        # Compare observations for deterministic behavior
        if isinstance(obs1, dict):
            for key in obs1:
                if isinstance(obs1[key], np.ndarray):
                    np.testing.assert_array_equal(
                        obs1[key], obs2[key], 
                        f"Non-deterministic reset for {key}"
                    )
                else:
                    assert obs1[key] == obs2[key], f"Non-deterministic reset for {key}"
        else:
            np.testing.assert_array_equal(obs1, obs2, "Non-deterministic reset")
        
        logger.info(
            "Reset with seed parameter works deterministically",
            extra={
                "metric_type": "seed_determinism_validation",
                "test_seed": DEFAULT_TEST_SEED
            }
        ) if ENHANCED_LOGGING else None
    
    def test_observation_space_compliance(self, gymnasium_env):
        """Test observation space structure and compliance."""
        obs_space = gymnasium_env.observation_space
        
        # Should be a Dict space with required keys
        assert isinstance(obs_space, DictSpace), "Observation space must be Dict"
        
        required_keys = {"odor_concentration", "agent_position", "agent_orientation"}
        actual_keys = set(obs_space.spaces.keys())
        
        assert required_keys.issubset(actual_keys), f"Missing required keys: {required_keys - actual_keys}"
        
        # Test observation generation
        obs, _ = gymnasium_env.reset()
        assert obs_space.contains(obs), "Generated observation not in observation space"
        
        # Validate individual components
        assert isinstance(obs["odor_concentration"], (np.ndarray, float)), "Odor concentration type invalid"
        assert isinstance(obs["agent_position"], np.ndarray), "Agent position must be numpy array"
        assert len(obs["agent_position"]) == 2, "Agent position must be 2D"
        assert isinstance(obs["agent_orientation"], (np.ndarray, float)), "Agent orientation type invalid"
    
    def test_action_space_compliance(self, gymnasium_env):
        """Test action space structure and bounds."""
        action_space = gymnasium_env.action_space
        
        # Should be Box space for continuous control
        assert isinstance(action_space, Box), "Action space must be Box for continuous control"
        assert len(action_space.shape) == 1, "Action space must be 1D"
        assert action_space.shape[0] == 2, "Action space must have 2 dimensions [speed, angular_velocity]"
        
        # Test action sampling and clipping
        action = action_space.sample()
        assert action_space.contains(action), "Sampled action not in action space"
        
        # Test bounds
        assert np.all(action >= action_space.low), "Action below lower bound"
        assert np.all(action <= action_space.high), "Action above upper bound"
    
    def test_episode_termination_logic(self, gymnasium_env):
        """Test proper episode termination with terminated/truncated distinction."""
        obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS + 10:
            action = gymnasium_env.action_space.sample()
            obs, reward, terminated, truncated, info = gymnasium_env.step(action)
            step_count += 1
            
            # Validate termination logic
            assert not (terminated and truncated), "Episode cannot be both terminated and truncated"
            
            if terminated:
                logger.info(
                    f"Episode terminated naturally at step {step_count}",
                    extra={"metric_type": "episode_termination", "reason": "terminated"}
                ) if ENHANCED_LOGGING else None
                break
            
            if truncated:
                logger.info(
                    f"Episode truncated at step {step_count}",
                    extra={"metric_type": "episode_termination", "reason": "truncated"}
                ) if ENHANCED_LOGGING else None
                break
        
        # Should terminate within reasonable time
        assert step_count <= MAX_EPISODE_STEPS + 5, "Episode ran too long without termination"


class TestDualAPISupport:
    """Test suite for dual API support (legacy gym vs modern Gymnasium)."""
    
    def test_legacy_api_returns_four_tuple(self, legacy_env):
        """Test F-004-RQ-007: Legacy gym callers receive 4-tuple without code changes."""
        obs = legacy_env.reset(seed=DEFAULT_TEST_SEED)
        
        # Legacy reset returns observation only (not tuple)
        assert not isinstance(obs, tuple) or len(obs) == 2, "Legacy reset format error"
        
        action = legacy_env.action_space.sample()
        step_result = legacy_env.step(action)
        
        # Verify 4-tuple format for legacy compatibility
        assert len(step_result) == 4, f"Expected 4-tuple for legacy API, got {len(step_result)}-tuple"
        
        obs, reward, done, info = step_result
        
        # Validate types
        assert legacy_env.observation_space.contains(obs), "Observation not in observation space"
        assert isinstance(reward, (int, float)), f"Reward must be numeric, got {type(reward)}"
        assert isinstance(done, bool), f"Done must be boolean, got {type(done)}"
        assert isinstance(info, dict), f"Info must be dict, got {type(info)}"
        
        # Verify termination information is preserved in info
        assert "terminated" in info, "Terminated flag should be in info for debugging"
        assert "truncated" in info, "Truncated flag should be in info for debugging"
        
        logger.info(
            "Legacy API returns valid 4-tuple",
            extra={
                "metric_type": "legacy_api_validation",
                "tuple_length": 4,
                "done": done
            }
        ) if ENHANCED_LOGGING else None
    
    def test_api_detection_mechanism(self):
        """Test compatibility layer API detection."""
        # Test detection function
        detection_result = detect_api_version(depth=2, performance_monitoring=True)
        
        assert isinstance(detection_result, APIDetectionResult)
        assert isinstance(detection_result.is_legacy, bool)
        assert 0.0 <= detection_result.confidence <= 1.0
        assert detection_result.detection_method is not None
        assert isinstance(detection_result.debug_info, dict)
        
        # Test detection consistency
        results = [detect_api_version() for _ in range(5)]
        legacy_votes = sum(1 for r in results if r.is_legacy)
        consistency = max(legacy_votes, 5 - legacy_votes) / 5
        
        assert consistency >= 0.6, f"API detection inconsistent: {consistency:.2f}"
        
        logger.info(
            f"API detection consistency: {consistency:.2f}",
            extra={
                "metric_type": "api_detection_validation",
                "consistency": consistency,
                "detection_method": detection_result.detection_method
            }
        ) if ENHANCED_LOGGING else None
    
    def test_format_step_return_function(self):
        """Test step return format conversion utility."""
        obs = {"test": np.array([1, 2, 3])}
        reward = 1.5
        terminated = True
        truncated = False
        info = {"test_info": "value"}
        
        # Test legacy format conversion
        legacy_result = format_step_return(obs, reward, terminated, truncated, info, use_legacy_api=True)
        assert len(legacy_result) == 4
        obs_l, reward_l, done_l, info_l = legacy_result
        assert obs_l is obs
        assert reward_l == reward
        assert done_l == (terminated or truncated)  # Combined flag
        assert info_l["terminated"] == terminated  # Preserved in info
        assert info_l["truncated"] == truncated
        
        # Test gymnasium format conversion  
        gym_result = format_step_return(obs, reward, terminated, truncated, info, use_legacy_api=False)
        assert len(gym_result) == 5
        obs_g, reward_g, terminated_g, truncated_g, info_g = gym_result
        assert obs_g is obs
        assert reward_g == reward
        assert terminated_g == terminated
        assert truncated_g == truncated
        assert info_g is info
    
    def test_compatibility_wrapper(self, gymnasium_env):
        """Test CompatibilityWrapper functionality."""
        # Test wrapping environment
        wrapped_env = wrap_environment(gymnasium_env)
        assert isinstance(wrapped_env, CompatibilityWrapper)
        
        # Test API mode detection
        assert hasattr(wrapped_env, 'use_legacy_api')
        assert hasattr(wrapped_env, 'detection_result')
        
        # Test step method delegation
        obs, info = wrapped_env.reset(seed=DEFAULT_TEST_SEED)
        action = wrapped_env.action_space.sample()
        step_result = wrapped_env.step(action)
        
        # Return format should match detected API mode
        if wrapped_env.use_legacy_api:
            assert len(step_result) == 4
        else:
            assert len(step_result) == 5
        
        # Test performance metrics
        metrics = wrapped_env.get_performance_metrics()
        assert "total_steps" in metrics
        assert "avg_step_time_ms" in metrics
        assert "api_mode" in metrics
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
    @given(
        use_legacy=st.booleans(),
        terminated=st.booleans(),
        truncated=st.booleans(),
        reward=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=1000)
    def test_format_conversion_properties(self, use_legacy, terminated, truncated, reward):
        """Property-based test for format conversion consistency."""
        obs = {"test": np.array([1.0, 2.0])}
        info = {"test": "info"}
        
        result = format_step_return(obs, reward, terminated, truncated, info, use_legacy)
        
        if use_legacy:
            assert len(result) == 4
            obs_r, reward_r, done_r, info_r = result
            assert done_r == (terminated or truncated)
        else:
            assert len(result) == 5
            obs_r, reward_r, terminated_r, truncated_r, info_r = result
            assert terminated_r == terminated
            assert truncated_r == truncated
        
        # Common assertions
        assert obs_r is obs
        assert reward_r == reward
        assert isinstance(info_r, dict)


class TestEnvironmentRegistration:
    """Test suite for environment registration and factory functions."""
    
    def test_new_environment_id_registration(self):
        """Test F-004-RQ-008: Register new environment ID 'PlumeNavSim-v0'."""
        if not GYMNASIUM_AVAILABLE:
            pytest.skip("Gymnasium not available")
        
        # Test environment registration
        registration_results = register_environments()
        
        assert "PlumeNavSim-v0" in registration_results, "PlumeNavSim-v0 not registered"
        assert registration_results["PlumeNavSim-v0"], "PlumeNavSim-v0 registration failed"
        
        # Test environment creation via gymnasium.make()
        try:
            env = gym.make("PlumeNavSim-v0", video_path="test.mp4")
            assert env is not None, "Failed to create PlumeNavSim-v0"
            env.close()
        except Exception as e:
            # Expected to fail due to missing video file, but registration should work
            assert "not found" in str(e).lower() or "no such file" in str(e).lower()
        
        logger.info(
            "PlumeNavSim-v0 environment registration successful",
            extra={"metric_type": "environment_registration", "env_id": "PlumeNavSim-v0"}
        ) if ENHANCED_LOGGING else None
    
    def test_legacy_environment_id_compatibility(self):
        """Test backward compatibility with OdorPlumeNavigation-v1."""
        registration_results = register_environments()
        
        if "OdorPlumeNavigation-v1" in registration_results:
            assert registration_results["OdorPlumeNavigation-v1"], "Legacy environment registration failed"
            
            # Test creation with available gym implementation
            try:
                target_gym = gym if GYMNASIUM_AVAILABLE else (legacy_gym if LEGACY_GYM_AVAILABLE else None)
                if target_gym:
                    env = target_gym.make("OdorPlumeNavigation-v1", video_path="test.mp4")
                    assert env is not None, "Failed to create OdorPlumeNavigation-v1"
                    env.close()
            except Exception as e:
                # Expected to fail due to missing video file
                assert "not found" in str(e).lower() or "no such file" in str(e).lower()
    
    def test_get_available_environments(self):
        """Test environment discovery functionality."""
        available_envs = get_available_environments()
        
        assert isinstance(available_envs, dict), "Available environments must be dict"
        
        if GYMNASIUM_AVAILABLE:
            assert "PlumeNavSim-v0" in available_envs, "PlumeNavSim-v0 not in available environments"
            
            env_info = available_envs["PlumeNavSim-v0"]
            assert env_info["api_type"] == "gymnasium"
            assert env_info["step_returns"] == "5-tuple"
            assert env_info["supports_terminated_truncated"] == True
            assert env_info["recommended"] == True
        
        if GYMNASIUM_AVAILABLE or LEGACY_GYM_AVAILABLE:
            assert "OdorPlumeNavigation-v1" in available_envs, "OdorPlumeNavigation-v1 not available"
            
            legacy_info = available_envs["OdorPlumeNavigation-v1"]
            assert legacy_info["api_type"] == "legacy_compatible"
            assert legacy_info["step_returns"] == "4-tuple"
            assert legacy_info["supports_terminated_truncated"] == False
    
    def test_make_environment_factory(self, test_config):
        """Test make_environment factory function."""
        if not GYMNASIUM_AVAILABLE:
            pytest.skip("Gymnasium not available")
        
        with patch('odor_plume_nav.environments.video_plume.VideoPlume'):
            with patch('odor_plume_nav.environments.gymnasium_env.NavigatorFactory'):
                # Test environment creation with config
                env = make_environment("PlumeNavSim-v0", config=test_config)
                
                if env is not None:  # Successful creation
                    assert hasattr(env, 'step'), "Environment missing step method"
                    assert hasattr(env, 'reset'), "Environment missing reset method"
                    assert hasattr(env, 'action_space'), "Environment missing action_space"
                    assert hasattr(env, 'observation_space'), "Environment missing observation_space"
                    env.close()
    
    def test_diagnose_environment_setup(self):
        """Test diagnostic functionality."""
        diagnostics = diagnose_environment_setup()
        
        assert isinstance(diagnostics, dict), "Diagnostics must be dict"
        assert "packages" in diagnostics, "Missing packages info"
        assert "environments" in diagnostics, "Missing environments info"
        assert "recommendations" in diagnostics, "Missing recommendations"
        
        packages = diagnostics["packages"]
        assert "gymnasium_available" in packages
        assert "rl_env_available" in packages
        assert "compat_layer_available" in packages
        
        environments = diagnostics["environments"]
        assert isinstance(environments, dict), "Environments info must be dict"
        
        recommendations = diagnostics["recommendations"]
        assert isinstance(recommendations, list), "Recommendations must be list"


class TestPerformanceRequirements:
    """Test suite for performance requirement validation."""
    
    def test_step_performance_target(self, gymnasium_env):
        """Test F-011-RQ-003: Performance warning if step() average >10ms."""
        obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        step_times = []
        num_steps = 100  # Sufficient for statistical significance
        
        for _ in range(num_steps):
            action = gymnasium_env.action_space.sample()
            
            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = gymnasium_env.step(action)
            step_time = time.perf_counter() - start_time
            
            step_times.append(step_time * 1000)  # Convert to milliseconds
            
            if terminated or truncated:
                obs, info = gymnasium_env.reset()
        
        avg_step_time = np.mean(step_times)
        median_step_time = np.median(step_times)
        p95_step_time = np.percentile(step_times, 95)
        
        logger.info(
            f"Step performance: avg={avg_step_time:.2f}ms, median={median_step_time:.2f}ms, p95={p95_step_time:.2f}ms",
            extra={
                "metric_type": "performance_validation",
                "avg_step_time_ms": avg_step_time,
                "median_step_time_ms": median_step_time,
                "p95_step_time_ms": p95_step_time,
                "target_ms": PERFORMANCE_TARGET_MS,
                "compliant": avg_step_time <= PERFORMANCE_TARGET_MS
            }
        ) if ENHANCED_LOGGING else None
        
        # Performance requirement: average step time ≤ 10ms
        if avg_step_time > PERFORMANCE_TARGET_MS:
            warnings.warn(
                f"Step time {avg_step_time:.2f}ms exceeds target {PERFORMANCE_TARGET_MS}ms",
                UserWarning
            )
            # Don't fail test, but log warning as per requirements
        
        # Assert that most steps are reasonable (allowing some outliers)
        assert p95_step_time <= PERFORMANCE_TARGET_MS * 2, f"95th percentile step time too high: {p95_step_time:.2f}ms"
    
    def test_reset_performance(self, gymnasium_env):
        """Test reset() performance requirements."""
        reset_times = []
        num_resets = 20
        
        for i in range(num_resets):
            start_time = time.perf_counter()
            obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED + i)
            reset_time = time.perf_counter() - start_time
            
            reset_times.append(reset_time * 1000)  # Convert to milliseconds
        
        avg_reset_time = np.mean(reset_times)
        
        logger.info(
            f"Reset performance: avg={avg_reset_time:.2f}ms",
            extra={
                "metric_type": "reset_performance_validation",
                "avg_reset_time_ms": avg_reset_time
            }
        ) if ENHANCED_LOGGING else None
        
        # Reset should be fast (target <20ms)
        assert avg_reset_time <= 20.0, f"Reset time too high: {avg_reset_time:.2f}ms"
    
    def test_compatibility_overhead(self, gymnasium_env):
        """Test that compatibility layer adds minimal overhead."""
        # Test direct environment
        obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        direct_times = []
        for _ in range(50):
            action = gymnasium_env.action_space.sample()
            start_time = time.perf_counter()
            gymnasium_env.step(action)
            direct_times.append(time.perf_counter() - start_time)
        
        # Test wrapped environment
        wrapped_env = wrap_environment(gymnasium_env)
        obs, info = wrapped_env.reset(seed=DEFAULT_TEST_SEED)
        
        wrapped_times = []
        for _ in range(50):
            action = wrapped_env.action_space.sample()
            start_time = time.perf_counter()
            wrapped_env.step(action)
            wrapped_times.append(time.perf_counter() - start_time)
        
        avg_direct = np.mean(direct_times) * 1000
        avg_wrapped = np.mean(wrapped_times) * 1000
        overhead = avg_wrapped - avg_direct
        overhead_pct = (overhead / avg_direct) * 100 if avg_direct > 0 else 0
        
        logger.info(
            f"Compatibility overhead: {overhead:.2f}ms ({overhead_pct:.1f}%)",
            extra={
                "metric_type": "compatibility_overhead",
                "direct_time_ms": avg_direct,
                "wrapped_time_ms": avg_wrapped,
                "overhead_ms": overhead,
                "overhead_percent": overhead_pct
            }
        ) if ENHANCED_LOGGING else None
        
        # Overhead should be minimal (<5% or <1ms)
        assert overhead <= 1.0 or overhead_pct <= 5.0, f"Compatibility overhead too high: {overhead:.2f}ms ({overhead_pct:.1f}%)"


class TestCrossRepositoryIntegration:
    """Test suite for cross-repository integration scenarios."""
    
    def test_stable_baselines3_compatibility(self, gymnasium_env):
        """Test compatibility with stable-baselines3 patterns."""
        # Test vectorized environment pattern
        try:
            from gymnasium.vector import SyncVectorEnv
            
            def make_env():
                return gymnasium_env
            
            # Test vectorized environment creation (common SB3 pattern)
            vec_env = SyncVectorEnv([make_env])
            
            # Test vectorized operations
            obs = vec_env.reset()
            assert len(obs) == 1, "Vectorized reset should return array of observations"
            
            actions = [vec_env.action_space.sample()]
            step_result = vec_env.step(actions)
            assert len(step_result) == 5, "Vectorized step should return 5-tuple"
            
            vec_env.close()
            
        except ImportError:
            pytest.skip("gymnasium.vector not available")
    
    def test_place_mem_rl_integration_pattern(self, gymnasium_env):
        """Test integration patterns expected by place_mem_rl repository."""
        # Test typical RL training loop pattern
        obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(3):  # Short test episodes
            episode_reward = 0
            episode_length = 0
            
            obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED + episode)
            done = False
            
            while not done and episode_length < MAX_EPISODE_STEPS:
                # Simulate policy action selection
                action = gymnasium_env.action_space.sample()
                
                # Environment step
                obs, reward, terminated, truncated, info = gymnasium_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                # Validate observation structure (place_mem_rl expectation)
                assert "odor_concentration" in obs, "Missing odor_concentration in observation"
                assert "agent_position" in obs, "Missing agent_position in observation"
                assert "agent_orientation" in obs, "Missing agent_orientation in observation"
                
                # Validate info structure
                assert isinstance(info, dict), "Info must be dictionary"
                assert "step" in info, "Missing step count in info"
                assert "episode" in info, "Missing episode count in info"
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Validate training metrics
        assert len(episode_rewards) == 3, "Missing episode rewards"
        assert len(episode_lengths) == 3, "Missing episode lengths"
        assert all(length > 0 for length in episode_lengths), "Zero-length episodes"
        
        logger.info(
            "place_mem_rl integration pattern test successful",
            extra={
                "metric_type": "integration_validation",
                "avg_episode_reward": np.mean(episode_rewards),
                "avg_episode_length": np.mean(episode_lengths)
            }
        ) if ENHANCED_LOGGING else None


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestPropertyBasedAPIContracts:
    """Property-based tests for API contract compliance across configurations."""
    
    @given(
        max_speed=st.floats(min_value=0.1, max_value=10.0),
        max_angular_velocity=st.floats(min_value=1.0, max_value=180.0),
        initial_position=st.tuples(
            st.floats(min_value=50, max_value=590),  # Within video bounds
            st.floats(min_value=50, max_value=430)
        ),
        initial_orientation=st.floats(min_value=0, max_value=360),
        include_multi_sensor=st.booleans(),
        num_sensors=st.integers(min_value=1, max_value=8)
    )
    @settings(max_examples=20, deadline=5000)
    def test_environment_configuration_properties(
        self, mock_video_file, mock_video_plume, mock_navigator,
        max_speed, max_angular_velocity, initial_position, initial_orientation,
        include_multi_sensor, num_sensors
    ):
        """Property-based test for environment configuration robustness."""
        if not GYMNASIUM_AVAILABLE:
            pytest.skip("Gymnasium not available")
        
        config = {
            'video_path': str(mock_video_file),
            'initial_position': initial_position,
            'initial_orientation': initial_orientation,
            'max_speed': max_speed,
            'max_angular_velocity': max_angular_velocity,
            'include_multi_sensor': include_multi_sensor,
            'num_sensors': num_sensors,
            'max_episode_steps': 10,  # Short for property testing
            'performance_monitoring': False  # Disable for speed
        }
        
        try:
            env = GymnasiumEnv(**config)
            
            # Test basic functionality
            obs, info = env.reset(seed=DEFAULT_TEST_SEED)
            
            # Validate observation structure
            assert env.observation_space.contains(obs), "Invalid observation"
            
            if include_multi_sensor:
                assert "multi_sensor_readings" in obs, "Missing multi-sensor readings"
                assert len(obs["multi_sensor_readings"]) == num_sensors
            
            # Test step
            action = env.action_space.sample()
            step_result = env.step(action)
            assert len(step_result) == 5, "Invalid step return format"
            
            env.close()
            
        except Exception as e:
            pytest.fail(f"Environment configuration failed: {e}")
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31-1),
        num_steps=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=10, deadline=3000)
    def test_deterministic_behavior_properties(
        self, gymnasium_env, seed, num_steps
    ):
        """Property-based test for deterministic behavior with seeding."""
        # Run episode 1
        obs1, info1 = gymnasium_env.reset(seed=seed)
        trajectory1 = [obs1]
        
        for _ in range(num_steps):
            action = np.array([1.0, 0.0], dtype=np.float32)  # Deterministic action
            obs, reward, terminated, truncated, info = gymnasium_env.step(action)
            trajectory1.append(obs)
            if terminated or truncated:
                break
        
        # Run episode 2 with same seed
        obs2, info2 = gymnasium_env.reset(seed=seed)
        trajectory2 = [obs2]
        
        for _ in range(len(trajectory1) - 1):
            action = np.array([1.0, 0.0], dtype=np.float32)  # Same deterministic action
            obs, reward, terminated, truncated, info = gymnasium_env.step(action)
            trajectory2.append(obs)
            if terminated or truncated:
                break
        
        # Compare trajectories
        assert len(trajectory1) == len(trajectory2), "Trajectory lengths differ"
        
        for i, (obs1, obs2) in enumerate(zip(trajectory1, trajectory2)):
            for key in obs1:
                if isinstance(obs1[key], np.ndarray):
                    np.testing.assert_allclose(
                        obs1[key], obs2[key], rtol=1e-6,
                        err_msg=f"Non-deterministic behavior at step {i}, key {key}"
                    )
                else:
                    assert abs(obs1[key] - obs2[key]) < 1e-6, f"Non-deterministic at step {i}, key {key}"


class TestComprehensiveValidation:
    """Comprehensive validation scenarios combining multiple test aspects."""
    
    def test_full_compatibility_validation(self, test_config, mock_video_plume, mock_navigator):
        """Comprehensive test validating all compatibility aspects."""
        if not GYMNASIUM_AVAILABLE:
            pytest.skip("Gymnasium not available")
        
        # Create both API modes
        modern_env = GymnasiumEnv(**test_config)
        legacy_config = {**test_config, '_force_legacy_api': True}
        legacy_env = GymnasiumEnv(**legacy_config)
        
        try:
            # Test compatibility validation utility
            modern_results = validate_compatibility(modern_env, test_episodes=2)
            legacy_results = validate_compatibility(legacy_env, test_episodes=2)
            
            # Validate results structure
            for results in [modern_results, legacy_results]:
                assert "overall_status" in results
                assert "legacy_api_tests" in results
                assert "gymnasium_api_tests" in results
                assert "performance" in results
                assert "recommendations" in results
            
            # Both should pass basic functionality
            assert modern_results["overall_status"] in ["passed", "passed_with_performance_warnings"]
            assert legacy_results["overall_status"] in ["passed", "passed_with_performance_warnings"]
            
            logger.info(
                "Comprehensive compatibility validation successful",
                extra={
                    "metric_type": "comprehensive_validation",
                    "modern_status": modern_results["overall_status"],
                    "legacy_status": legacy_results["overall_status"]
                }
            ) if ENHANCED_LOGGING else None
            
        finally:
            modern_env.close()
            legacy_env.close()
    
    def test_end_to_end_rl_workflow(self, gymnasium_env):
        """Test complete RL training workflow simulation."""
        # Simulate typical RL training workflow
        total_steps = 0
        total_episodes = 0
        total_reward = 0
        
        for episode in range(5):  # Multiple episodes
            obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED + episode)
            episode_reward = 0
            episode_steps = 0
            
            while episode_steps < MAX_EPISODE_STEPS:
                # Simulate policy (random for testing)
                action = gymnasium_env.action_space.sample()
                
                # Environment step
                obs, reward, terminated, truncated, info = gymnasium_env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Simulate training data collection
                training_data = {
                    'obs': obs,
                    'action': action,
                    'reward': reward,
                    'terminated': terminated,
                    'truncated': truncated,
                    'info': info
                }
                
                # Validate training data structure
                assert all(key in training_data for key in ['obs', 'action', 'reward', 'terminated', 'truncated'])
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
            total_episodes += 1
        
        avg_episode_reward = total_reward / total_episodes
        avg_episode_length = total_steps / total_episodes
        
        logger.info(
            f"End-to-end RL workflow: {total_episodes} episodes, avg reward={avg_episode_reward:.2f}, avg length={avg_episode_length:.1f}",
            extra={
                "metric_type": "rl_workflow_validation",
                "total_episodes": total_episodes,
                "total_steps": total_steps,
                "avg_episode_reward": avg_episode_reward,
                "avg_episode_length": avg_episode_length
            }
        ) if ENHANCED_LOGGING else None
        
        # Validate workflow metrics
        assert total_episodes == 5, "Incorrect episode count"
        assert total_steps > 0, "No steps executed"
        assert avg_episode_length > 0, "Zero average episode length"


# Utility context manager for tests
@contextmanager
def nullcontext():
    """Null context manager for conditional context usage."""
    yield


# Performance monitoring decorator for test methods
def monitor_test_performance(threshold_ms: float = 1000):
    """Decorator to monitor test execution performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = (time.perf_counter() - start_time) * 1000
                if execution_time > threshold_ms:
                    warnings.warn(
                        f"Test {func.__name__} took {execution_time:.2f}ms (threshold: {threshold_ms}ms)",
                        UserWarning
                    )
                if ENHANCED_LOGGING:
                    logger.debug(
                        f"Test {func.__name__} execution time: {execution_time:.2f}ms",
                        extra={
                            "metric_type": "test_performance",
                            "test_name": func.__name__,
                            "execution_time_ms": execution_time,
                            "threshold_ms": threshold_ms,
                            "compliant": execution_time <= threshold_ms
                        }
                    )
        return wrapper
    return decorator


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])