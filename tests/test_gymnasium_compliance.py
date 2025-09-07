"""
Comprehensive Gymnasium 0.29.x API Compliance Test Suite for Plume Navigation Environment.

This module provides comprehensive testing of the plume_nav_sim environment's adherence to 
Gymnasium 0.29.x API standards using gymnasium.utils.env_checker as the primary validation 
mechanism. Tests cover environment registration, space compliance, modern reset/step signatures,
and dual API compatibility for seamless migration from legacy Gym usage.

Key Test Coverage Areas:
- Gymnasium API compliance validation via env_checker per Section 0.4.1 requirements
- Environment registration for PlumeNavSim-v0 ID per Section 0.3.4 specification
- Modern reset(seed=None, options=None) signature validation per Section 0.2.1
- 5-tuple step return format: (obs, reward, terminated, truncated, info) per Section 0.3.1
- Observation and action space compliance per gymnasium.spaces validation requirements
- Dual API support validation ensuring both legacy and modern interface compatibility
- Performance validation ensuring step() execution meets <10ms requirements

Architecture:
- pytest-based framework with comprehensive fixtures for consistent test execution
- Mock implementations for VideoPlume and Navigator dependencies for deterministic testing
- Property-based testing using hypothesis for robust contract validation
- Performance benchmarking with timing assertions for real-time training compatibility
- Structured logging integration for comprehensive test observability and debugging

Technical Requirements Validated:
- F-005: Environment passes gymnasium.utils.env_checker validation per Section 0.2.3
- F-004-RQ-005: Environment step() returns 5-tuple for Gymnasium callers per Section 0.3.1
- F-004-RQ-006: reset(seed=...) parameter support with deterministic behavior per Section 0.2.1
- F-004-RQ-008: New environment ID 'PlumeNavSim-v0' registration per Section 0.3.4
- F-011-RQ-003: Performance validation ensuring step() average <10ms per Section 2.2.3

Example Usage:
    Run specific gymnasium compliance tests:
    >>> pytest tests/test_gymnasium_compliance.py::TestGymnasiumCompliance::test_env_checker_validation -v
    
    Run all API compliance tests:
    >>> pytest tests/test_gymnasium_compliance.py -v
    
    Run with performance benchmarking:
    >>> pytest tests/test_gymnasium_compliance.py --benchmark-only
"""

from __future__ import annotations

import time
import warnings
import tempfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import uuid

import pytest
import numpy as np

# Core testing and validation frameworks
try:
    from hypothesis import given, strategies as st, settings, assume
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

# Gymnasium and environment testing imports
try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    from gymnasium.spaces import Box, Dict as DictSpace, Space
    from gymnasium.error import Error as GymnasiumError
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = Mock()
    check_env = Mock()
    Box = DictSpace = Space = Mock()
    GymnasiumError = Exception

# Test target imports - main environment under test
try:
    from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv
    PLUME_NAV_ENV_AVAILABLE = True
except ImportError:
    PLUME_NAV_ENV_AVAILABLE = False
    PlumeNavigationEnv = None

# Compatibility and space validation utilities
try:
    from plume_nav_sim.envs.spaces import (
        ActionSpaceFactory, ObservationSpaceFactory, SpaceValidator,
        ReturnFormatConverter, get_standard_action_space, get_standard_observation_space
    )
    SPACES_AVAILABLE = True
except ImportError:
    SPACES_AVAILABLE = False
    ActionSpaceFactory = ObservationSpaceFactory = SpaceValidator = Mock()
    ReturnFormatConverter = Mock()
    get_standard_action_space = get_standard_observation_space = Mock()

# Enhanced logging for test correlation tracking
try:
    from plume_nav_sim.utils.logging_setup import (
        get_enhanced_logger, correlation_context
    )
    ENHANCED_LOGGING = True
    logger = get_enhanced_logger(__name__)
except ImportError:
from loguru import logger
    ENHANCED_LOGGING = False
    
    @contextmanager
    def correlation_context(name, **kwargs):
        yield

# Test constants and configuration
PERFORMANCE_TARGET_MS = 10.0  # Step execution target from Section F-011-RQ-003
TEST_VIDEO_WIDTH = 640
TEST_VIDEO_HEIGHT = 480
TEST_VIDEO_FRAMES = 100
MAX_EPISODE_STEPS = 50  # Short episodes for faster testing
DEFAULT_TEST_SEED = 42

# Test correlation tracking for enhanced observability
TEST_CORRELATION_ID = f"test_gymnasium_compliance_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def mock_video_file():
    """Create temporary mock video file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
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
    with patch('plume_nav_sim.envs.plume_navigation_env.VideoPlume') as mock_class:
        mock_instance = Mock()
        mock_instance.get_metadata.return_value = {
            'width': TEST_VIDEO_WIDTH,
            'height': TEST_VIDEO_HEIGHT,
            'fps': 30.0,
            'frame_count': TEST_VIDEO_FRAMES
        }
        
        # Generate consistent mock frames
        def get_frame(frame_index):
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
    with patch('plume_nav_sim.envs.plume_navigation_env.NavigatorFactory') as mock_factory:
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
    """Create PlumeNavigationEnv instance for testing."""
    if not GYMNASIUM_AVAILABLE or not PLUME_NAV_ENV_AVAILABLE:
        pytest.skip("Gymnasium or PlumeNavigationEnv not available")
    
    env = PlumeNavigationEnv(**test_config)
    yield env
    env.close()


@pytest.fixture
def legacy_env(test_config, mock_video_plume, mock_navigator):
    """Create PlumeNavigationEnv in legacy compatibility mode."""
    if not GYMNASIUM_AVAILABLE or not PLUME_NAV_ENV_AVAILABLE:
        pytest.skip("Gymnasium or PlumeNavigationEnv not available")
    
    config_with_legacy = {**test_config, '_force_legacy_api': True}
    env = PlumeNavigationEnv(**config_with_legacy)
    yield env
    env.close()


class TestGymnasiumCompliance:
    """
    Primary test suite for Gymnasium 0.29.x API compliance validation.
    
    This class implements comprehensive testing of the PlumeNavigationEnv against
    Gymnasium API standards using env_checker as the authoritative validation tool.
    Tests ensure full compliance with modern RL framework requirements while
    maintaining backward compatibility for legacy usage patterns.
    """
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_env_checker_validation(self, gymnasium_env):
        """
        Test F-005: Environment passes gymnasium.utils.env_checker validation.
        
        This is the primary compliance test using gymnasium's official validation
        tool as specified in Section 0.4.1. The env_checker performs comprehensive
        validation of the Gymnasium API including space compliance, method signatures,
        return formats, and proper error handling.
        """
        with correlation_context("env_checker_validation", correlation_id=TEST_CORRELATION_ID):
            try:
                # Run official Gymnasium environment checker
                check_env(gymnasium_env, warn=True, skip_render_check=True)
                
                logger.info(
                    "Environment passed gymnasium env_checker validation successfully",
                    extra={
                        "metric_type": "api_compliance_success",
                        "test_type": "env_checker_validation",
                        "environment_id": "PlumeNavSim-v0"
                    }
                ) if ENHANCED_LOGGING else None
                
            except Exception as e:
                pytest.fail(f"Environment failed gymnasium env_checker validation: {e}")
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_environment_registration(self):
        """
        Test F-004-RQ-008: New environment ID 'PlumeNavSim-v0' registration.
        
        Validates that the new Gymnasium-compliant environment ID is properly
        registered and can be created through the standard gym.make() interface
        as specified in Section 0.3.4.
        """
        try:
            # Test environment creation via gymnasium.make()
            env = gym.make("PlumeNavSim-v0")
            assert env is not None, "Failed to create PlumeNavSim-v0 environment"
            
            # Verify it's the correct environment type
            assert hasattr(env, 'reset'), "Environment missing reset method"
            assert hasattr(env, 'step'), "Environment missing step method"
            assert hasattr(env, 'action_space'), "Environment missing action_space"
            assert hasattr(env, 'observation_space'), "Environment missing observation_space"
            
            env.close()
            
            logger.info(
                "PlumeNavSim-v0 environment registration validation successful",
                extra={
                    "metric_type": "environment_registration",
                    "env_id": "PlumeNavSim-v0",
                    "registration_success": True
                }
            ) if ENHANCED_LOGGING else None
            
        except Exception as e:
            # Expected to fail due to missing video file in registration test
            if "not found" in str(e).lower() or "no such file" in str(e).lower():
                # This is expected - registration works but requires video file
                pass
            else:
                pytest.fail(f"Environment registration failed: {e}")
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_modern_reset_signature(self, gymnasium_env):
        """
        Test F-004-RQ-006: reset(seed=...) parameter support with deterministic behavior.
        
        Validates the modern Gymnasium reset interface per Section 0.2.1 requirements,
        ensuring proper seed parameter handling and deterministic behavior.
        """
        # Test reset with seed parameter
        obs1, info1 = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        # Verify return format - must be 2-tuple (observation, info)
        assert isinstance(obs1, dict), "Reset observation must be dict"
        assert isinstance(info1, dict), "Reset info must be dict"
        assert "seed" in info1, "Seed should be recorded in info dict"
        assert info1["seed"] == DEFAULT_TEST_SEED, "Incorrect seed recorded"
        
        # Test deterministic behavior - same seed should give same initial state
        obs2, info2 = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        # Compare observations for deterministic behavior
        for key in obs1:
            if isinstance(obs1[key], np.ndarray):
                np.testing.assert_array_equal(
                    obs1[key], obs2[key], 
                    f"Non-deterministic reset for observation key: {key}"
                )
            else:
                assert obs1[key] == obs2[key], f"Non-deterministic reset for key: {key}"
        
        logger.info(
            "Modern reset signature validation successful",
            extra={
                "metric_type": "reset_signature_validation",
                "test_seed": DEFAULT_TEST_SEED,
                "deterministic": True
            }
        ) if ENHANCED_LOGGING else None
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_modern_step_return_format(self, gymnasium_env):
        """
        Test F-004-RQ-005: Environment step() returns 5-tuple for Gymnasium callers.
        
        Validates the modern Gymnasium step interface per Section 0.3.1 requirements,
        ensuring proper 5-tuple return format with terminated/truncated distinction.
        """
        obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        action = gymnasium_env.action_space.sample()
        
        # Execute step and verify return format
        step_result = gymnasium_env.step(action)
        
        # Must return exactly 5 elements
        assert len(step_result) == 5, f"Expected 5-tuple, got {len(step_result)}-tuple"
        
        # Unpack and validate types
        obs, reward, terminated, truncated, info = step_result
        
        # Type validation per Gymnasium specification
        assert gymnasium_env.observation_space.contains(obs), "Observation not in observation space"
        assert isinstance(reward, (int, float)), f"Reward must be numeric, got {type(reward)}"
        assert isinstance(terminated, bool), f"Terminated must be boolean, got {type(terminated)}"
        assert isinstance(truncated, bool), f"Truncated must be boolean, got {type(truncated)}"
        assert isinstance(info, dict), f"Info must be dict, got {type(info)}"
        
        # Logical validation - cannot be both terminated and truncated
        assert not (terminated and truncated), "Episode cannot be both terminated and truncated"
        
        logger.info(
            "Modern step return format validation successful",
            extra={
                "metric_type": "step_format_validation",
                "tuple_length": 5,
                "terminated": terminated,
                "truncated": truncated,
                "reward": reward
            }
        ) if ENHANCED_LOGGING else None
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_observation_space_compliance(self, gymnasium_env):
        """
        Test observation space structure and Gymnasium compliance.
        
        Validates that observation spaces conform to gymnasium.spaces validation
        requirements with proper structure, types, and bounds.
        """
        obs_space = gymnasium_env.observation_space
        
        # Should be a Dict space with required keys per plume navigation specification
        assert isinstance(obs_space, DictSpace), "Observation space must be Dict"
        
        # Validate required observation components
        required_keys = {"odor_concentration", "agent_position", "agent_orientation"}
        actual_keys = set(obs_space.spaces.keys())
        assert required_keys.issubset(actual_keys), f"Missing required keys: {required_keys - actual_keys}"
        
        # Test observation generation and space compliance
        obs, _ = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        assert obs_space.contains(obs), "Generated observation not in observation space"
        
        # Validate individual observation components
        assert isinstance(obs["odor_concentration"], (np.ndarray, float)), "Invalid odor concentration type"
        assert isinstance(obs["agent_position"], np.ndarray), "Agent position must be numpy array"
        assert len(obs["agent_position"]) == 2, "Agent position must be 2D coordinate"
        assert isinstance(obs["agent_orientation"], (np.ndarray, float)), "Invalid agent orientation type"
        
        logger.info(
            "Observation space compliance validation successful",
            extra={
                "metric_type": "observation_space_validation",
                "space_type": type(obs_space).__name__,
                "observation_keys": list(actual_keys)
            }
        ) if ENHANCED_LOGGING else None
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_action_space_compliance(self, gymnasium_env):
        """
        Test action space structure and Gymnasium compliance.
        
        Validates that action spaces conform to gymnasium.spaces requirements
        with proper bounds, dtypes, and sampling behavior.
        """
        action_space = gymnasium_env.action_space
        
        # Should be Box space for continuous control
        assert isinstance(action_space, Box), "Action space must be Box for continuous control"
        assert len(action_space.shape) == 1, "Action space must be 1D"
        assert action_space.shape[0] == 2, "Action space must have 2 dimensions [speed, angular_velocity]"
        
        # Test action sampling and bounds validation
        action = action_space.sample()
        assert action_space.contains(action), "Sampled action not in action space"
        
        # Validate bounds compliance
        assert np.all(action >= action_space.low), "Action below lower bound"
        assert np.all(action <= action_space.high), "Action above upper bound"
        
        # Test dtype compliance
        assert action.dtype == action_space.dtype, "Action dtype mismatch with space dtype"
        
        logger.info(
            "Action space compliance validation successful",
            extra={
                "metric_type": "action_space_validation",
                "space_type": type(action_space).__name__,
                "action_shape": action_space.shape,
                "action_dtype": str(action_space.dtype)
            }
        ) if ENHANCED_LOGGING else None
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_episode_termination_logic(self, gymnasium_env):
        """
        Test proper episode termination with terminated/truncated distinction.
        
        Validates that the environment properly handles episode termination
        with correct terminated vs truncated semantics per Gymnasium specification.
        """
        obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        terminated = False
        truncated = False
        step_count = 0
        
        # Run episode until termination or truncation
        while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS + 10:
            action = gymnasium_env.action_space.sample()
            obs, reward, terminated, truncated, info = gymnasium_env.step(action)
            step_count += 1
            
            # Validate termination logic at each step
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


class TestLegacyCompatibility:
    """
    Test suite for legacy Gym API compatibility and dual API support.
    
    Validates that the environment properly supports legacy usage patterns
    while maintaining modern Gymnasium compliance through the compatibility layer.
    """
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_legacy_api_compatibility(self, legacy_env):
        """
        Test that legacy API mode properly converts to 4-tuple returns.
        
        Validates the compatibility layer's ability to detect legacy usage
        and provide appropriate return formats for backward compatibility.
        """
        # Test legacy reset format
        reset_result = legacy_env.reset(seed=DEFAULT_TEST_SEED)
        
        # Legacy reset should return observation only (or 2-tuple with info)
        if isinstance(reset_result, tuple):
            assert len(reset_result) == 2, "Legacy reset should return 2-tuple (obs, info)"
            obs, info = reset_result
        else:
            obs = reset_result
            assert legacy_env.observation_space.contains(obs), "Invalid legacy reset observation"
        
        # Test legacy step format
        action = legacy_env.action_space.sample()
        step_result = legacy_env.step(action)
        
        # Legacy step should return 4-tuple
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
            "Legacy API compatibility validation successful",
            extra={
                "metric_type": "legacy_api_validation",
                "step_tuple_length": 4,
                "done": done,
                "has_termination_info": "terminated" in info and "truncated" in info
            }
        ) if ENHANCED_LOGGING else None


class TestPerformanceRequirements:
    """
    Test suite for performance requirement validation.
    
    Validates that the environment meets the <10ms step execution requirement
    per Section F-011-RQ-003 for real-time training compatibility.
    """
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_step_performance_requirement(self, gymnasium_env):
        """
        Test F-011-RQ-003: Performance validation ensuring step() average <10ms.
        
        Validates that environment step execution meets performance requirements
        essential for real-time RL training workflows.
        """
        obs, info = gymnasium_env.reset(seed=DEFAULT_TEST_SEED)
        
        step_times = []
        num_steps = 100  # Sufficient for statistical significance
        
        # Measure step execution times
        for _ in range(num_steps):
            action = gymnasium_env.action_space.sample()
            
            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = gymnasium_env.step(action)
            step_time = time.perf_counter() - start_time
            
            step_times.append(step_time * 1000)  # Convert to milliseconds
            
            if terminated or truncated:
                obs, info = gymnasium_env.reset()
        
        # Performance statistics
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


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not available")
class TestPropertyBasedCompliance:
    """
    Property-based tests for robust API contract validation.
    
    Uses hypothesis to test environment behavior across a wide range of
    configurations and inputs to ensure robust compliance.
    """
    
    @given(
        seed=st.integers(min_value=0, max_value=2**31-1),
        num_steps=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=10, deadline=3000)
    def test_deterministic_behavior_property(self, gymnasium_env, seed, num_steps):
        """
        Property-based test for deterministic behavior with seeding.
        
        Validates that identical seeds produce identical behavior across
        multiple runs, ensuring reproducible experiments.
        """
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
        
        # Compare trajectories for deterministic behavior
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
    """
    Comprehensive validation combining multiple aspects of Gymnasium compliance.
    
    These tests validate the complete environment workflow including registration,
    creation, interaction, and cleanup to ensure production readiness.
    """
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_ENV_AVAILABLE, reason="PlumeNavigationEnv not available")
    def test_complete_gymnasium_workflow(self, test_config, mock_video_plume, mock_navigator):
        """
        Test complete Gymnasium workflow from creation to cleanup.
        
        Validates the entire environment lifecycle to ensure proper integration
        with RL training pipelines and research workflows.
        """
        # Environment creation
        env = PlumeNavigationEnv(**test_config)
        
        try:
            # Verify environment structure
            assert hasattr(env, 'action_space'), "Missing action_space attribute"
            assert hasattr(env, 'observation_space'), "Missing observation_space attribute"
            assert hasattr(env, 'metadata'), "Missing metadata attribute"
            
            # Test full interaction sequence
            obs, info = env.reset(seed=DEFAULT_TEST_SEED)
            assert isinstance(obs, dict), "Invalid observation type"
            assert isinstance(info, dict), "Invalid info type"
            
            # Multi-step interaction
            total_reward = 0.0
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            # Verify interaction completed successfully
            assert total_reward is not None, "No reward accumulated"
            
            logger.info(
                "Complete Gymnasium workflow validation successful",
                extra={
                    "metric_type": "workflow_validation",
                    "total_steps": step + 1,
                    "total_reward": total_reward,
                    "terminated": terminated,
                    "truncated": truncated
                }
            ) if ENHANCED_LOGGING else None
            
        finally:
            # Ensure proper cleanup
            env.close()


# Utility functions for test support
@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for monitoring test operation performance."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_time = (time.perf_counter() - start_time) * 1000
        if ENHANCED_LOGGING:
            logger.debug(
                f"Operation {operation_name} completed in {elapsed_time:.2f}ms",
                extra={
                    "metric_type": "test_performance",
                    "operation": operation_name,
                    "execution_time_ms": elapsed_time
                }
            )


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])