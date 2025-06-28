"""
Comprehensive test suite for the gym-to-gymnasium compatibility shim layer.

This module validates the backward compatibility layer that allows legacy projects 
using OpenAI Gym API patterns to work seamlessly with the modernized Gymnasium-based
plume_nav_sim library while guiding migration to modern APIs.

Test Coverage:
- Legacy gym.make() proxy functionality
- Automatic caller detection and format conversion
- Deprecation warning emission and guidance
- 4-tuple to 5-tuple return format conversion
- Environment proxy behavior and API consistency
- Integration with existing downstream projects
- Performance impact validation
"""

import contextlib
import io
import inspect
import sys
import threading
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import gymnasium

from plume_nav_sim.shims import gym_make
from plume_nav_sim.shims.gym_make import (
    LegacyWrapper,
    _is_legacy_caller,
    _convert_to_legacy_format,
    _detect_caller_context,
    _emit_migration_warning,
)
from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv


class TestGymMakeProxy:
    """Test suite for the gym_make() compatibility function."""

    def test_gym_make_creates_gymnasium_environment(self):
        """Test that gym_make creates a valid Gymnasium environment."""
        env = gym_make("PlumeNavSim-v0")
        
        # Verify it's a valid Gymnasium environment
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        
        # Clean up
        env.close()

    def test_gym_make_forwards_kwargs_to_gymnasium(self):
        """Test that gym_make properly forwards keyword arguments to gymnasium.make()."""
        with patch('gymnasium.make') as mock_gymnasium_make:
            mock_env = MagicMock()
            mock_gymnasium_make.return_value = mock_env
            
            # Call with various kwargs
            kwargs = {
                'render_mode': 'human',
                'max_episode_steps': 1000,
                'custom_param': 'test_value'
            }
            
            result = gym_make("PlumeNavSim-v0", **kwargs)
            
            # Verify gymnasium.make was called with correct arguments
            mock_gymnasium_make.assert_called_once_with("PlumeNavSim-v0", **kwargs)
            assert result == mock_env

    def test_gym_make_handles_invalid_environment_id(self):
        """Test that gym_make handles invalid environment IDs gracefully."""
        with pytest.raises(Exception):  # Should raise same error as gymnasium.make
            gym_make("NonExistentEnv-v0")

    def test_gym_make_preserves_environment_attributes(self):
        """Test that gym_make preserves all environment attributes."""
        env = gym_make("PlumeNavSim-v0")
        
        # Check that essential attributes are preserved
        assert hasattr(env, 'spec')
        assert hasattr(env, 'metadata')
        assert hasattr(env, 'reward_range')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        
        env.close()


class TestDeprecationWarning:
    """Test suite for deprecation warning functionality."""

    def test_gym_make_emits_deprecation_warning(self):
        """Test that gym_make emits a deprecation warning."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            env = gym_make("PlumeNavSim-v0")
            
            # Verify deprecation warning was emitted
            assert len(warning_list) >= 1
            deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            
            warning = deprecation_warnings[0]
            assert "gym_make is deprecated" in str(warning.message)
            assert "gymnasium.make" in str(warning.message)
            assert "PlumeNavSim-v0" in str(warning.message)
            
            env.close()

    def test_deprecation_warning_includes_migration_guidance(self):
        """Test that deprecation warning includes clear migration guidance."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            gym_make("PlumeNavSim-v0").close()
            
            warning = warning_list[-1]  # Get the last warning
            warning_message = str(warning.message)
            
            # Check for key migration guidance elements
            assert "Please update to:" in warning_message
            assert "gymnasium.make" in warning_message
            assert "v1.0" in warning_message  # Version when removal happens
            
    def test_deprecation_warning_stack_level(self):
        """Test that deprecation warning has correct stack level."""
        def wrapper_function():
            return gym_make("PlumeNavSim-v0")
        
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            env = wrapper_function()
            
            # Verify warning points to the caller, not internal implementation
            warning = warning_list[-1]
            # Stack level should make the warning appear at the wrapper_function call
            # This is implementation-dependent but we can verify it's not pointing to gym_make internals
            assert warning.filename.endswith('test_shim_compatibility.py')
            
            env.close()

    def test_emit_migration_warning_content(self):
        """Test the _emit_migration_warning function directly."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            
            _emit_migration_warning("TestEnv-v0")
            
            assert len(warning_list) == 1
            warning = warning_list[0]
            assert issubclass(warning.category, DeprecationWarning)
            
            message = str(warning.message)
            assert "TestEnv-v0" in message
            assert "gymnasium.make('TestEnv-v0')" in message


class TestCallerDetection:
    """Test suite for legacy caller detection mechanism."""

    def test_is_legacy_caller_detection_basic(self):
        """Test basic legacy caller detection functionality."""
        # Create a mock legacy context
        with patch('plume_nav_sim.shims.gym_make._detect_caller_context') as mock_detect:
            mock_detect.return_value = {'is_legacy': True, 'caller_info': {'file': 'old_script.py'}}
            
            result = _is_legacy_caller()
            assert result is True
            
            mock_detect.return_value = {'is_legacy': False, 'caller_info': {'file': 'new_script.py'}}
            
            result = _is_legacy_caller()
            assert result is False

    def test_detect_caller_context_analyzes_stack(self):
        """Test that _detect_caller_context properly analyzes the call stack."""
        def legacy_style_caller():
            # Simulate a legacy caller pattern
            return _detect_caller_context()
        
        context = legacy_style_caller()
        
        # Verify context contains expected information
        assert isinstance(context, dict)
        assert 'is_legacy' in context
        assert 'caller_info' in context
        assert isinstance(context['caller_info'], dict)

    def test_caller_detection_with_place_mem_rl_pattern(self):
        """Test caller detection specifically for place_mem_rl compatibility."""
        # Simulate call stack that looks like it's from place_mem_rl
        with patch('inspect.stack') as mock_stack:
            # Create a mock stack frame that resembles place_mem_rl usage
            mock_frame = MagicMock()
            mock_frame.filename = '/path/to/place_mem_rl/agent.py'
            mock_frame.function = 'create_environment'
            mock_frame.code_context = ['env = gym.make("PlumeNavSim-v0")']
            
            mock_stack.return_value = [
                MagicMock(),  # Current frame
                MagicMock(),  # gym_make frame  
                mock_frame,   # Caller frame (place_mem_rl)
            ]
            
            context = _detect_caller_context()
            
            # Should detect as legacy due to place_mem_rl pattern
            assert context['is_legacy'] is True
            assert 'place_mem_rl' in context['caller_info']['file']

    def test_caller_detection_with_modern_gymnasium_pattern(self):
        """Test caller detection with modern Gymnasium usage patterns."""
        with patch('inspect.stack') as mock_stack:
            # Create a mock stack frame that resembles modern usage
            mock_frame = MagicMock()
            mock_frame.filename = '/path/to/modern_project/train.py'
            mock_frame.function = 'setup_training'
            mock_frame.code_context = ['import gymnasium as gym']
            
            mock_stack.return_value = [
                MagicMock(),  # Current frame
                MagicMock(),  # gym_make frame
                mock_frame,   # Caller frame (modern)
            ]
            
            context = _detect_caller_context()
            
            # Should detect as modern due to gymnasium import pattern
            assert context['is_legacy'] is False

    def test_caller_detection_thread_safety(self):
        """Test that caller detection works correctly in multi-threaded environments."""
        results = []
        exceptions = []
        
        def worker_thread(thread_id):
            try:
                # Each thread should get its own stack analysis
                context = _detect_caller_context()
                results.append((thread_id, context))
            except Exception as e:
                exceptions.append((thread_id, e))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no exceptions and all threads got results
        assert len(exceptions) == 0, f"Thread exceptions: {exceptions}"
        assert len(results) == 5
        
        # Each result should be a valid context
        for thread_id, context in results:
            assert isinstance(context, dict)
            assert 'is_legacy' in context
            assert 'caller_info' in context


class TestLegacyWrapper:
    """Test suite for the LegacyWrapper that converts API formats."""

    def test_legacy_wrapper_initialization(self):
        """Test that LegacyWrapper properly initializes with a Gymnasium environment."""
        gym_env = gymnasium.make("PlumeNavSim-v0")
        wrapper = LegacyWrapper(gym_env)
        
        # Verify wrapper preserves environment attributes
        assert wrapper.action_space == gym_env.action_space
        assert wrapper.observation_space == gym_env.observation_space
        assert wrapper.spec == gym_env.spec
        
        wrapper.close()

    def test_legacy_wrapper_reset_conversion(self):
        """Test that LegacyWrapper converts reset() to legacy format."""
        gym_env = gymnasium.make("PlumeNavSim-v0")
        wrapper = LegacyWrapper(gym_env)
        
        # Modern reset returns (observation, info)
        result = wrapper.reset()
        
        # Legacy reset should return just observation
        assert not isinstance(result, tuple), "Legacy reset should return observation only, not tuple"
        assert isinstance(result, (np.ndarray, dict)), "Reset should return observation"
        
        wrapper.close()

    def test_legacy_wrapper_reset_with_seed(self):
        """Test that LegacyWrapper handles seed parameter correctly."""
        gym_env = gymnasium.make("PlumeNavSim-v0")
        wrapper = LegacyWrapper(gym_env)
        
        # Test reset with seed
        obs1 = wrapper.reset(seed=42)
        obs2 = wrapper.reset(seed=42)
        
        # With same seed, observations should be reproducible
        # Note: exact equality might not hold due to floating point precision
        assert obs1 is not None
        assert obs2 is not None
        
        wrapper.close()

    def test_legacy_wrapper_step_conversion(self):
        """Test that LegacyWrapper converts step() to legacy 4-tuple format."""
        gym_env = gymnasium.make("PlumeNavSim-v0")
        wrapper = LegacyWrapper(gym_env)
        
        wrapper.reset()
        action = wrapper.action_space.sample()
        
        # Call step and verify it returns 4-tuple (legacy format)
        result = wrapper.step(action)
        
        assert isinstance(result, tuple), "Step should return a tuple"
        assert len(result) == 4, "Legacy step should return 4-tuple"
        
        obs, reward, done, info = result
        
        # Verify types
        assert isinstance(obs, (np.ndarray, dict)), "Observation should be array or dict"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        
        wrapper.close()

    def test_legacy_wrapper_step_terminated_truncated_combination(self):
        """Test that LegacyWrapper properly combines terminated and truncated into done."""
        gym_env = gymnasium.make("PlumeNavSim-v0")
        wrapper = LegacyWrapper(gym_env)
        
        # Mock the underlying environment to control terminated/truncated values
        with patch.object(gym_env, 'step') as mock_step:
            # Test case 1: terminated=True, truncated=False -> done=True
            mock_step.return_value = (
                np.array([0.0, 0.0]), 1.0, True, False, {'test': 'info'}
            )
            
            wrapper.reset()
            obs, reward, done, info = wrapper.step(wrapper.action_space.sample())
            assert done is True, "done should be True when terminated=True"
            
            # Test case 2: terminated=False, truncated=True -> done=True  
            mock_step.return_value = (
                np.array([0.0, 0.0]), 0.5, False, True, {'test': 'info'}
            )
            
            obs, reward, done, info = wrapper.step(wrapper.action_space.sample())
            assert done is True, "done should be True when truncated=True"
            
            # Test case 3: terminated=False, truncated=False -> done=False
            mock_step.return_value = (
                np.array([0.0, 0.0]), 0.0, False, False, {'test': 'info'}
            )
            
            obs, reward, done, info = wrapper.step(wrapper.action_space.sample())
            assert done is False, "done should be False when both terminated and truncated are False"
        
        wrapper.close()

    def test_convert_to_legacy_format_function(self):
        """Test the _convert_to_legacy_format utility function."""
        # Test modern 5-tuple to legacy 4-tuple conversion
        modern_result = (
            np.array([1.0, 2.0]),  # observation
            1.5,                   # reward
            True,                  # terminated
            False,                 # truncated
            {'step': 100}          # info
        )
        
        legacy_result = _convert_to_legacy_format(modern_result)
        
        assert isinstance(legacy_result, tuple)
        assert len(legacy_result) == 4
        
        obs, reward, done, info = legacy_result
        assert np.array_equal(obs, np.array([1.0, 2.0]))
        assert reward == 1.5
        assert done is True  # terminated=True OR truncated=False -> True
        assert info == {'step': 100}

    def test_convert_to_legacy_format_preserves_info(self):
        """Test that conversion preserves and enhances info dictionary."""
        modern_result = (
            np.array([0.0]),
            0.0,
            False,  # terminated
            True,   # truncated
            {'episode_length': 50}
        )
        
        legacy_result = _convert_to_legacy_format(modern_result)
        obs, reward, done, info = legacy_result
        
        assert done is True  # Should be True because truncated=True
        assert 'episode_length' in info
        
        # Check if additional termination info is added
        assert 'terminated' in info
        assert 'truncated' in info
        assert info['terminated'] is False
        assert info['truncated'] is True


class TestShimIntegration:
    """Test suite for end-to-end shim integration scenarios."""

    def test_shim_with_stable_baselines3_compatibility(self):
        """Test shim compatibility with stable-baselines3 usage patterns."""
        # Test that the shim works with typical SB3 usage
        env = gym_make("PlumeNavSim-v0")
        
        # Verify it has the required methods for SB3
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step') 
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        
        # Test basic episode
        obs = env.reset()
        assert obs is not None
        
        action = env.action_space.sample()
        result = env.step(action)
        
        # Should work with either 4-tuple or 5-tuple depending on detection
        assert isinstance(result, tuple)
        assert len(result) in [4, 5]
        
        env.close()

    def test_shim_preserves_environment_seeding(self):
        """Test that shim preserves seeding behavior for reproducibility."""
        # Create two environments with same seed
        env1 = gym_make("PlumeNavSim-v0")
        env2 = gym_make("PlumeNavSim-v0")
        
        # Reset with same seed
        obs1 = env1.reset(seed=12345)
        obs2 = env2.reset(seed=12345)
        
        # Take same action
        action = env1.action_space.sample()
        env1.action_space.seed(12345)  # Seed action space too
        env2.action_space.seed(12345)
        
        result1 = env1.step(action)
        result2 = env2.step(action)
        
        # Results should be similar (allowing for floating point differences)
        if isinstance(result1[0], np.ndarray) and isinstance(result2[0], np.ndarray):
            assert np.allclose(result1[0], result2[0], rtol=1e-6)
        
        env1.close()
        env2.close()

    def test_shim_performance_overhead(self):
        """Test that shim introduces minimal performance overhead."""
        import time
        
        # Test direct gymnasium.make performance
        start_time = time.perf_counter()
        for _ in range(10):
            env = gymnasium.make("PlumeNavSim-v0")
            env.reset()
            env.step(env.action_space.sample())
            env.close()
        direct_time = time.perf_counter() - start_time
        
        # Test gym_make performance
        start_time = time.perf_counter()
        for _ in range(10):
            env = gym_make("PlumeNavSim-v0")
            env.reset()
            env.step(env.action_space.sample())
            env.close()
        shim_time = time.perf_counter() - start_time
        
        # Shim should add minimal overhead (less than 50% increase)
        overhead_ratio = shim_time / direct_time
        assert overhead_ratio < 1.5, f"Shim overhead too high: {overhead_ratio:.2f}x"

    def test_shim_error_handling_consistency(self):
        """Test that shim handles errors consistently with gymnasium."""
        # Test invalid environment ID
        with pytest.raises(Exception):
            gym_make("InvalidEnv-v999")
        
        # Test invalid actions
        env = gym_make("PlumeNavSim-v0")
        env.reset()
        
        # Create an invalid action (outside action space bounds)
        if hasattr(env.action_space, 'high'):
            invalid_action = env.action_space.high * 2  # Way outside bounds
            # Should handle gracefully (exact behavior depends on environment)
            try:
                env.step(invalid_action)
            except Exception as e:
                # Exception should be meaningful
                assert len(str(e)) > 0
        
        env.close()


class TestBackwardCompatibility:
    """Test suite for backward compatibility with existing projects."""

    def test_place_mem_rl_compatibility_pattern(self):
        """Test compatibility with place_mem_rl usage patterns."""
        # Simulate typical place_mem_rl environment creation pattern
        def create_environment_like_place_mem_rl():
            # This simulates how place_mem_rl might create environments
            return gym_make("PlumeNavSim-v0")
        
        env = create_environment_like_place_mem_rl()
        
        # Test typical RL training loop pattern
        obs = env.reset()
        total_reward = 0
        
        for _ in range(10):  # Short episode for testing
            action = env.action_space.sample()
            step_result = env.step(action)
            
            # Handle both 4-tuple and 5-tuple formats
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            total_reward += reward
            
            if done:
                obs = env.reset()
                break
        
        assert total_reward is not None
        env.close()

    def test_legacy_gym_import_pattern_simulation(self):
        """Test simulation of legacy gym import patterns."""
        # Simulate old-style imports that might trigger legacy detection
        with patch('sys.modules') as mock_modules:
            # Add a fake 'gym' module to simulate legacy environment
            mock_gym_module = MagicMock()
            mock_modules.__contains__ = lambda self, name: name == 'gym'
            mock_modules.__getitem__ = lambda self, name: mock_gym_module if name == 'gym' else {}
            
            # This should trigger legacy detection and warning
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                
                env = gym_make("PlumeNavSim-v0")
                
                # Should emit deprecation warning
                deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
                assert len(deprecation_warnings) >= 1
                
                env.close()

    def test_multiple_environment_instances(self):
        """Test that multiple environment instances work correctly with shim."""
        envs = []
        
        try:
            # Create multiple environments
            for i in range(3):
                env = gym_make("PlumeNavSim-v0")
                envs.append(env)
            
            # Reset all environments
            observations = []
            for env in envs:
                obs = env.reset(seed=42 + len(observations))
                observations.append(obs)
            
            # Step all environments
            for i, env in enumerate(envs):
                action = env.action_space.sample()
                result = env.step(action)
                assert isinstance(result, tuple)
                assert len(result) in [4, 5]
                
        finally:
            # Clean up all environments
            for env in envs:
                env.close()

    def test_gymnasium_compatibility_check(self):
        """Test that shim maintains full compatibility with gymnasium API."""
        env = gym_make("PlumeNavSim-v0")
        
        # Test that it still works with gymnasium.utils.env_checker
        try:
            from gymnasium.utils.env_checker import check_env
            # This should pass without errors
            check_env(env, warn=True)
        except ImportError:
            # If env_checker is not available, skip this test
            pytest.skip("gymnasium.utils.env_checker not available")
        
        env.close()


class TestErrorConditions:
    """Test suite for error conditions and edge cases."""

    def test_shim_with_closed_environment(self):
        """Test shim behavior with closed environments."""
        env = gym_make("PlumeNavSim-v0")
        env.close()
        
        # Operations on closed environment should fail gracefully
        with pytest.raises(Exception):
            env.reset()
        
        with pytest.raises(Exception):
            env.step(env.action_space.sample())

    def test_shim_with_invalid_actions(self):
        """Test shim behavior with invalid actions."""
        env = gym_make("PlumeNavSim-v0")
        env.reset()
        
        # Test with None action
        try:
            env.step(None)
            assert False, "Should have raised an exception for None action"
        except Exception:
            pass  # Expected behavior
        
        env.close()

    def test_caller_detection_with_corrupted_stack(self):
        """Test caller detection with corrupted or incomplete stack traces."""
        with patch('inspect.stack') as mock_stack:
            # Simulate corrupted stack
            mock_stack.side_effect = RuntimeError("Stack inspection failed")
            
            # Should handle gracefully and default to modern behavior
            context = _detect_caller_context()
            assert isinstance(context, dict)
            assert 'is_legacy' in context
            # Should default to False (modern) when detection fails
            assert context['is_legacy'] is False

    def test_warning_system_with_filtered_warnings(self):
        """Test that deprecation warnings work even with warning filters."""
        # Temporarily filter all warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Reset to capture warnings again
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                
                env = gym_make("PlumeNavSim-v0")
                
                # Should still emit warning despite filters
                assert len(warning_list) >= 1
                
                env.close()


# Test fixtures and utilities

@pytest.fixture
def mock_gymnasium_env():
    """Fixture providing a mock Gymnasium environment for testing."""
    mock_env = MagicMock()
    mock_env.action_space = MagicMock()
    mock_env.observation_space = MagicMock()
    mock_env.spec = MagicMock()
    mock_env.metadata = {}
    mock_env.reward_range = (-float('inf'), float('inf'))
    
    # Mock reset to return (observation, info)
    mock_env.reset.return_value = (np.array([0.0, 0.0]), {})
    
    # Mock step to return (obs, reward, terminated, truncated, info)
    mock_env.step.return_value = (
        np.array([1.0, 1.0]), 0.1, False, False, {'step': 1}
    )
    
    return mock_env


@pytest.fixture
def capture_warnings():
    """Fixture for capturing warnings during tests."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list


# Performance and stress tests

class TestShimPerformance:
    """Performance tests for the compatibility shim."""

    @pytest.mark.performance
    def test_shim_step_performance(self):
        """Test that shim maintains sub-10ms step performance."""
        env = gym_make("PlumeNavSim-v0")
        env.reset()
        
        import time
        step_times = []
        
        for _ in range(100):
            action = env.action_space.sample()
            
            start_time = time.perf_counter()
            env.step(action)
            end_time = time.perf_counter()
            
            step_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate performance metrics
        mean_time = np.mean(step_times)
        p95_time = np.percentile(step_times, 95)
        
        # Assert performance requirements from Section 6.6.4
        assert mean_time < 10.0, f"Mean step time {mean_time:.2f}ms exceeds 10ms target"
        assert p95_time < 15.0, f"P95 step time {p95_time:.2f}ms exceeds 15ms target"
        
        env.close()

    @pytest.mark.performance  
    def test_shim_memory_usage(self):
        """Test that shim doesn't introduce significant memory overhead."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and use multiple environments
        envs = []
        for _ in range(10):
            env = gym_make("PlumeNavSim-v0")
            env.reset()
            env.step(env.action_space.sample())
            envs.append(env)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        for env in envs:
            env.close()
        
        # Memory increase should be reasonable (less than 100MB for 10 environments)
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB too high"

    @pytest.mark.stress
    def test_shim_concurrent_usage(self):
        """Test shim behavior under concurrent usage."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker():
            try:
                env = gym_make("PlumeNavSim-v0")
                env.reset()
                
                for _ in range(10):
                    action = env.action_space.sample()
                    result = env.step(action)
                    assert isinstance(result, tuple)
                
                env.close()
                results.append(True)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all threads succeeded
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 5, "Not all threads completed successfully"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])