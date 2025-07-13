"""
Comprehensive pytest test module for validating extension system hooks.

This module provides complete validation of the v1.0 extension system hooks including
extra_obs_fn, extra_reward_fn, and episode_end_fn integration with lifecycle management
and performance compliance validation. The test suite ensures proper hook registration,
execution order, error handling, and integration with recording and statistics systems
while maintaining the ≤33ms step latency requirement with 100 agents.

Test Categories:
- Hook registration and lifecycle management with validation and constraints
- extra_obs_fn hook testing for custom observation augmentation and space compatibility
- extra_reward_fn hook testing for reward modification and calculation accuracy
- episode_end_fn hook testing for episode completion handling and cleanup
- Performance compliance validation ensuring hook execution meets timing requirements
- Integration testing with recording and statistics systems for automated data collection
- Backwards compatibility testing for v0.3.0 migration support
- Error handling and recovery testing for robust hook system operation

Architecture Integration:
- Tests hook integration with PlumeNavigationEnv per Section 5.2.3 environment requirements
- Validates NavigatorProtocol hook methods implementation per Section 5.2.1 navigation engine
- Tests RecorderManager integration per Section 5.2.8 recording infrastructure
- Validates StatsAggregator integration per Section 5.2.9 analysis framework
- Ensures SingleAgentController and MultiAgentController hook support per navigation requirements

Performance Requirements:
- Hook execution overhead: <1ms per agent for minimal simulation impact
- Multi-agent hook performance: ≤33ms with 100 agents per specification
- Memory efficiency: <10MB overhead for hook system with 100 agents
- Recording integration: <1ms additional overhead for hook data collection

Quality Gates:
- 100% hook system test coverage with comprehensive scenario validation
- Performance regression testing with automated benchmark validation
- Integration testing across recording and statistics systems
- Error condition testing with graceful degradation validation
- Backwards compatibility testing for seamless v0.3.0 to v1.0 migration

Examples:
    Basic hook system testing:
    >>> def test_extra_obs_hook():
    ...     env = create_test_environment()
    ...     env.set_hooks(extra_obs_fn=lambda state: {"custom_data": 1.0})
    ...     obs, info = env.reset()
    ...     assert "custom_data" in obs

    Performance compliance testing:
    >>> def test_hook_performance():
    ...     env = create_multi_agent_environment(num_agents=100)
    ...     env.set_hooks(extra_obs_fn=custom_obs_fn)
    ...     start_time = time.perf_counter()
    ...     env.step(env.action_space.sample())
    ...     duration = time.perf_counter() - start_time
    ...     assert duration < 0.033  # ≤33ms requirement

Authors: Blitzy Platform Hook System Agent
License: MIT
"""

import pytest
import numpy as np
import time
import warnings
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
from unittest.mock import patch, Mock, MagicMock
from contextlib import contextmanager

# Core testing framework imports per external import specifications
import gymnasium
from gymnasium import spaces

# Internal imports per import specifications
from plume_nav_sim.core.protocols import StatsAggregatorProtocol
from plume_nav_sim.envs.plume_navigation_env import PlumeNavigationEnv
from plume_nav_sim.core.controllers import SingleAgentController
from plume_nav_sim.recording import RecorderManager
from plume_nav_sim.analysis import StatsAggregator


# Test configuration constants for performance validation
SINGLE_AGENT_HOOK_THRESHOLD_MS = 1.0  # <1ms per agent requirement
MULTI_AGENT_HOOK_THRESHOLD_MS = 33.0  # ≤33ms with 100 agents requirement
HOOK_MEMORY_THRESHOLD_MB = 10.0  # <10MB overhead requirement
RECORDING_OVERHEAD_THRESHOLD_MS = 1.0  # <1ms additional overhead requirement


class TestHookRegistration:
    """Test hook registration and lifecycle management functionality."""
    
    def test_hook_registration_interface(self):
        """Test that hook registration interface is properly implemented."""
        # Create test environment with hook support
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        # Test hook registration methods exist
        assert hasattr(env, 'set_hooks'), "Environment must have set_hooks method"
        assert hasattr(env, 'configure_hooks'), "Environment must have configure_hooks method"
        assert hasattr(env, 'get_hooks'), "Environment must have get_hooks method"
        assert hasattr(env, 'clear_hooks'), "Environment must have clear_hooks method"
        
        # Test individual hook setters
        assert hasattr(env, 'set_extra_obs_fn'), "Environment must support extra_obs_fn registration"
        assert hasattr(env, 'set_extra_reward_fn'), "Environment must support extra_reward_fn registration"
        assert hasattr(env, 'set_episode_end_fn'), "Environment must support episode_end_fn registration"
    
    def test_hook_validation_and_constraints(self):
        """Test hook validation and constraint enforcement during registration."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        # Test valid hook function registration
        def valid_extra_obs_fn(state):
            return {"test_obs": 1.0}
        
        def valid_extra_reward_fn(base_reward, info):
            return 0.1
        
        def valid_episode_end_fn(final_info):
            pass
        
        # These should succeed without error
        env.set_extra_obs_fn(valid_extra_obs_fn)
        env.set_extra_reward_fn(valid_extra_reward_fn)
        env.set_episode_end_fn(valid_episode_end_fn)
        
        # Test invalid hook function rejection
        with pytest.raises(TypeError):
            env.set_extra_obs_fn("not_a_function")
        
        with pytest.raises(TypeError):
            env.set_extra_reward_fn(123)
        
        with pytest.raises(TypeError):
            env.set_episode_end_fn(None)
        
        # Test hook function signature validation
        def invalid_obs_fn():  # Wrong signature
            return {}
        
        with pytest.raises(ValueError):
            env.set_extra_obs_fn(invalid_obs_fn)
    
    def test_hook_lifecycle_management(self):
        """Test hook lifecycle management across environment resets and episodes."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        hook_call_count = {"obs": 0, "reward": 0, "episode_end": 0}
        
        def counting_obs_fn(state):
            hook_call_count["obs"] += 1
            return {"call_count": hook_call_count["obs"]}
        
        def counting_reward_fn(base_reward, info):
            hook_call_count["reward"] += 1
            return 0.1
        
        def counting_episode_end_fn(final_info):
            hook_call_count["episode_end"] += 1
        
        # Register hooks
        env.set_hooks(
            extra_obs_fn=counting_obs_fn,
            extra_reward_fn=counting_reward_fn,
            episode_end_fn=counting_episode_end_fn
        )
        
        # Test hooks persist across resets
        obs, info = env.reset()
        assert hook_call_count["obs"] > 0, "Extra obs hook should be called during reset"
        
        # Test hooks are called during steps
        initial_obs_count = hook_call_count["obs"]
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert hook_call_count["obs"] > initial_obs_count, "Extra obs hook should be called during step"
        assert hook_call_count["reward"] > 0, "Extra reward hook should be called during step"
        
        # Test episode end hook is called when episode terminates
        env._terminate_episode()
        assert hook_call_count["episode_end"] > 0, "Episode end hook should be called on termination"
    
    def test_multiple_hook_registration(self):
        """Test registration and management of multiple hooks with different configurations."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        # Register multiple hooks with different purposes
        def exploration_obs_fn(state):
            return {"exploration_bonus": 0.1, "visited_positions": len(state.get("trajectory", []))}
        
        def efficiency_reward_fn(base_reward, info):
            path_length = info.get("episode_length", 1)
            return base_reward + (1.0 / path_length)  # Efficiency bonus
        
        def logging_episode_end_fn(final_info):
            print(f"Episode completed with reward: {final_info.get('total_reward', 0)}")
        
        # Register all hooks simultaneously
        env.configure_hooks({
            "extra_obs_fn": exploration_obs_fn,
            "extra_reward_fn": efficiency_reward_fn,
            "episode_end_fn": logging_episode_end_fn,
            "enable_timing": True,
            "enable_validation": True
        })
        
        # Verify all hooks are registered and functional
        hooks = env.get_hooks()
        assert hooks["extra_obs_fn"] is not None, "Extra obs hook should be registered"
        assert hooks["extra_reward_fn"] is not None, "Extra reward hook should be registered"
        assert hooks["episode_end_fn"] is not None, "Episode end hook should be registered"
        
        # Test concurrent hook execution
        obs, info = env.reset()
        assert "exploration_bonus" in obs, "Exploration obs hook should contribute to observations"
        assert "visited_positions" in obs, "Position tracking obs hook should contribute to observations"
        
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert reward > 0, "Efficiency reward hook should modify reward"


class TestExtraObsHooks:
    """Test extra_obs_fn hook functionality for custom observation augmentation."""
    
    def test_extra_obs_fn_integration(self):
        """Test extra_obs_fn integration with observation space and data flow."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        def custom_obs_fn(state):
            """Add custom observations including wind direction and energy level."""
            return {
                "wind_direction": np.array([1.0, 0.5]),  # Wind vector
                "energy_level": 0.8,  # Agent energy
                "exploration_score": len(state.get("trajectory", [])) * 0.1,
                "time_since_last_detection": state.get("steps_since_detection", 0)
            }
        
        env.set_extra_obs_fn(custom_obs_fn)
        
        # Test that custom observations are included in reset
        obs, info = env.reset()
        assert "wind_direction" in obs, "Custom wind direction should be in observations"
        assert "energy_level" in obs, "Custom energy level should be in observations"
        assert "exploration_score" in obs, "Custom exploration score should be in observations"
        assert "time_since_last_detection" in obs, "Custom timing data should be in observations"
        
        # Test that custom observations are included in step
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert "wind_direction" in obs, "Custom observations should persist in step"
        assert isinstance(obs["wind_direction"], np.ndarray), "Wind direction should be numpy array"
        assert isinstance(obs["energy_level"], (int, float)), "Energy level should be numeric"
    
    def test_observation_modification_validation(self):
        """Test validation of observation modifications and data type consistency."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        def type_consistent_obs_fn(state):
            """Return type-consistent observations."""
            return {
                "float_value": 3.14,
                "int_value": 42,
                "array_value": np.array([1.0, 2.0, 3.0]),
                "bool_value": True,
                "string_value": "test"
            }
        
        env.set_extra_obs_fn(type_consistent_obs_fn)
        
        # Test multiple resets and steps for type consistency
        for _ in range(3):
            obs, info = env.reset()
            
            # Validate types remain consistent
            assert isinstance(obs["float_value"], (int, float)), "Float values should be numeric"
            assert isinstance(obs["int_value"], (int, float)), "Int values should be numeric"
            assert isinstance(obs["array_value"], np.ndarray), "Array values should be numpy arrays"
            assert isinstance(obs["bool_value"], (bool, np.bool_)), "Bool values should be boolean"
            assert isinstance(obs["string_value"], str), "String values should be strings"
            
            # Test step consistency
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert all(key in obs for key in ["float_value", "int_value", "array_value", "bool_value", "string_value"]), \
                "All custom observation keys should be present in step observations"
    
    def test_extra_obs_fn_error_handling(self):
        """Test error handling and graceful degradation for extra_obs_fn failures."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True,
            "hook_error_mode": "warn"  # Don't crash on hook errors
        })
        
        call_count = {"count": 0}
        
        def failing_obs_fn(state):
            """Obs function that fails after a few calls."""
            call_count["count"] += 1
            if call_count["count"] > 2:
                raise ValueError("Simulated observation hook failure")
            return {"working_obs": call_count["count"]}
        
        env.set_extra_obs_fn(failing_obs_fn)
        
        # First few calls should work
        obs, info = env.reset()
        assert "working_obs" in obs, "Hook should work initially"
        
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert "working_obs" in obs, "Hook should still work"
        
        # Next call should fail but not crash the environment
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Should have generated a warning about hook failure
            assert len(w) > 0, "Should generate warning on hook failure"
            assert "hook" in str(w[0].message).lower(), "Warning should mention hook failure"
        
        # Environment should continue functioning without the hook
        assert "working_obs" not in obs, "Failed hook should not contribute observations"
        assert "odor_concentration" in obs, "Base observations should still work"
    
    def test_observation_space_compatibility(self):
        """Test that extra observations are compatible with environment observation space."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True,
            "dynamic_observation_space": True
        })
        
        def bounded_obs_fn(state):
            """Return observations with known bounds for space compatibility."""
            return {
                "normalized_energy": np.clip(state.get("energy", 1.0), 0.0, 1.0),
                "direction_vector": np.array([np.cos(state.get("angle", 0)), np.sin(state.get("angle", 0))]),
                "step_count": min(state.get("step_count", 0), 1000)  # Bounded step count
            }
        
        env.set_extra_obs_fn(bounded_obs_fn)
        
        # Get updated observation space
        obs_space = env.observation_space
        
        # Test that custom observations fit within space bounds
        obs, info = env.reset()
        
        if isinstance(obs_space, spaces.Dict):
            # Check that extra observations are within bounds
            if "normalized_energy" in obs_space.spaces:
                assert obs_space.spaces["normalized_energy"].contains(obs["normalized_energy"]), \
                    "Normalized energy should be within observation space bounds"
            
            if "direction_vector" in obs_space.spaces:
                assert obs_space.spaces["direction_vector"].contains(obs["direction_vector"]), \
                    "Direction vector should be within observation space bounds"
        
        # Test multiple steps for continued compatibility
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Validate bounds are maintained
            assert 0.0 <= obs["normalized_energy"] <= 1.0, "Energy should remain normalized"
            assert np.allclose(np.linalg.norm(obs["direction_vector"]), 1.0, atol=1e-6), \
                "Direction vector should be unit length"
            assert obs["step_count"] <= 1000, "Step count should be bounded"


class TestExtraRewardHooks:
    """Test extra_reward_fn hook functionality for reward modification."""
    
    def test_extra_reward_fn_integration(self):
        """Test extra_reward_fn integration with reward calculation and data flow."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        reward_history = {"calls": [], "base_rewards": [], "extra_rewards": []}
        
        def exploration_reward_fn(base_reward, info):
            """Add exploration bonus based on novelty and efficiency."""
            reward_history["calls"].append(len(reward_history["calls"]))
            reward_history["base_rewards"].append(base_reward)
            
            # Calculate exploration bonus
            novelty_bonus = 0.1 if info.get("novel_position", False) else 0.0
            efficiency_bonus = 0.05 if info.get("path_efficiency", 0) > 0.5 else 0.0
            step_penalty = -0.001  # Small penalty for long episodes
            
            extra_reward = novelty_bonus + efficiency_bonus + step_penalty
            reward_history["extra_rewards"].append(extra_reward)
            
            return extra_reward
        
        env.set_extra_reward_fn(exploration_reward_fn)
        
        # Test reward modification during environment interaction
        obs, info = env.reset()
        assert len(reward_history["calls"]) == 0, "Reward hook should not be called during reset"
        
        # Take several steps and verify reward modifications
        for step_num in range(5):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Verify hook was called
            assert len(reward_history["calls"]) == step_num + 1, "Reward hook should be called each step"
            
            # Verify reward structure
            assert isinstance(reward, (int, float)), "Modified reward should be numeric"
            assert np.isfinite(reward), "Modified reward should be finite"
            
            # Verify extra reward was applied
            base_reward = reward_history["base_rewards"][-1]
            extra_reward = reward_history["extra_rewards"][-1]
            expected_total = base_reward + extra_reward
            
            assert np.isclose(reward, expected_total, atol=1e-6), \
                f"Total reward should equal base + extra: {reward} vs {expected_total}"
    
    def test_reward_modification_validation(self):
        """Test validation of reward modifications and numerical stability."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        def stable_reward_fn(base_reward, info):
            """Return numerically stable reward modifications."""
            # Test various reward modification patterns
            step = info.get("step_count", 0)
            
            if step % 4 == 0:
                return 0.1  # Positive bonus
            elif step % 4 == 1:
                return -0.05  # Small penalty
            elif step % 4 == 2:
                return 0.0  # No modification
            else:
                return base_reward * 0.1  # Proportional bonus
        
        env.set_extra_reward_fn(stable_reward_fn)
        
        # Test reward stability over many steps
        rewards = []
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            rewards.append(reward)
            
            # Validate numerical properties
            assert np.isfinite(reward), "All rewards should be finite"
            assert not np.isnan(reward), "No rewards should be NaN"
            assert isinstance(reward, (int, float)), "All rewards should be numeric"
        
        # Test reward distribution properties
        rewards = np.array(rewards)
        assert len(rewards) == 20, "Should have collected all rewards"
        assert np.all(np.isfinite(rewards)), "All rewards should be finite"
        assert not np.any(np.isnan(rewards)), "No rewards should be NaN"
    
    def test_reward_calculation_accuracy(self):
        """Test accuracy of reward calculations with mathematical precision."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        calculation_log = {"operations": []}
        
        def precise_reward_fn(base_reward, info):
            """Perform precise mathematical reward calculations."""
            position = info.get("position", [0, 0])
            target = info.get("target_position", [50, 50])
            
            # Calculate distance-based reward with high precision
            distance = np.sqrt((position[0] - target[0])**2 + (position[1] - target[1])**2)
            distance_reward = -distance * 0.01  # Negative distance penalty
            
            # Calculate concentration-based reward
            concentration = info.get("odor_concentration", 0.0)
            concentration_reward = concentration * 10.0  # Amplify concentration signal
            
            # Calculate time-based penalty
            time_penalty = -info.get("step_count", 0) * 0.001
            
            total_extra = distance_reward + concentration_reward + time_penalty
            
            calculation_log["operations"].append({
                "base_reward": base_reward,
                "distance_reward": distance_reward,
                "concentration_reward": concentration_reward,
                "time_penalty": time_penalty,
                "total_extra": total_extra
            })
            
            return total_extra
        
        env.set_extra_reward_fn(precise_reward_fn)
        
        # Test mathematical precision over multiple steps
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Verify calculation accuracy
            last_calc = calculation_log["operations"][-1]
            expected_total = (last_calc["base_reward"] + last_calc["distance_reward"] + 
                            last_calc["concentration_reward"] + last_calc["time_penalty"])
            
            assert np.isclose(reward, expected_total, atol=1e-10), \
                f"Reward calculation should be mathematically precise: {reward} vs {expected_total}"
        
        # Verify all operations were logged
        assert len(calculation_log["operations"]) == 10, "Should have logged all calculations"
    
    def test_reward_hook_performance(self):
        """Test performance characteristics of reward hook execution."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        timing_data = {"durations": []}
        
        def timed_reward_fn(base_reward, info):
            """Reward function with performance timing."""
            start_time = time.perf_counter()
            
            # Simulate various computational operations
            position = info.get("position", [0, 0])
            
            # Vector operations
            distance_vec = np.array(position) - np.array([25, 25])
            distance = np.linalg.norm(distance_vec)
            
            # Trigonometric operations
            angle = np.arctan2(distance_vec[1], distance_vec[0])
            directional_bonus = np.cos(angle) * 0.1
            
            # Statistical operations
            recent_rewards = info.get("recent_rewards", [base_reward])
            mean_reward = np.mean(recent_rewards)
            
            end_time = time.perf_counter()
            timing_data["durations"].append((end_time - start_time) * 1000)  # Convert to ms
            
            return directional_bonus + (base_reward - mean_reward) * 0.1
        
        env.set_extra_reward_fn(timed_reward_fn)
        
        # Warm up the system
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        # Performance test over multiple steps
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        # Analyze performance characteristics
        durations = np.array(timing_data["durations"])
        mean_duration = np.mean(durations)
        p95_duration = np.percentile(durations, 95)
        max_duration = np.max(durations)
        
        # Validate performance requirements
        assert mean_duration < SINGLE_AGENT_HOOK_THRESHOLD_MS, \
            f"Mean reward hook duration {mean_duration:.3f}ms should be <{SINGLE_AGENT_HOOK_THRESHOLD_MS}ms"
        assert p95_duration < SINGLE_AGENT_HOOK_THRESHOLD_MS * 2, \
            f"P95 reward hook duration {p95_duration:.3f}ms should be reasonable"
        assert max_duration < SINGLE_AGENT_HOOK_THRESHOLD_MS * 5, \
            f"Max reward hook duration {max_duration:.3f}ms should not be excessive"


class TestEpisodeEndHooks:
    """Test episode_end_fn hook functionality for episode completion handling."""
    
    def test_episode_end_fn_integration(self):
        """Test episode_end_fn integration with episode lifecycle and cleanup."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        episode_data = {"episodes": [], "cleanup_called": 0}
        
        def comprehensive_episode_end_fn(final_info):
            """Comprehensive episode completion handler."""
            episode_data["episodes"].append({
                "total_reward": final_info.get("total_reward", 0.0),
                "episode_length": final_info.get("episode_length", 0),
                "success": final_info.get("success", False),
                "final_position": final_info.get("final_position", [0, 0]),
                "termination_reason": final_info.get("termination_reason", "unknown"),
                "timestamp": time.time()
            })
            episode_data["cleanup_called"] += 1
        
        env.set_episode_end_fn(comprehensive_episode_end_fn)
        
        # Run a complete episode
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < 50:  # Limit steps to prevent infinite episodes
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        # Manually trigger episode end if needed
        if not (terminated or truncated):
            env._end_episode(final_info={
                "total_reward": total_reward,
                "episode_length": steps,
                "success": False,
                "termination_reason": "manual"
            })
        
        # Verify episode end hook was called
        assert episode_data["cleanup_called"] > 0, "Episode end hook should be called"
        assert len(episode_data["episodes"]) > 0, "Episode data should be collected"
        
        # Verify episode data structure
        last_episode = episode_data["episodes"][-1]
        assert "total_reward" in last_episode, "Episode should contain total reward"
        assert "episode_length" in last_episode, "Episode should contain length"
        assert "success" in last_episode, "Episode should contain success indicator"
        assert "final_position" in last_episode, "Episode should contain final position"
        assert "termination_reason" in last_episode, "Episode should contain termination reason"
        assert isinstance(last_episode["timestamp"], (int, float)), "Episode should contain timestamp"
    
    def test_stats_aggregator_hook_integration(self):
        """Test integration between episode_end_fn and StatsAggregator for automated metrics."""
        # Create mock stats aggregator
        mock_aggregator = Mock(spec=StatsAggregator)
        mock_aggregator.calculate_episode_stats.return_value = {
            "path_efficiency": 0.75,
            "exploration_coverage": 0.60,
            "mean_concentration": 0.45,
            "success_indicator": 1.0
        }
        mock_aggregator.export_summary.return_value = True
        
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        aggregation_data = {"episodes_processed": 0, "metrics_calculated": []}
        
        def stats_integration_fn(final_info):
            """Episode end handler with stats aggregator integration."""
            # Extract trajectory data for analysis
            trajectory_data = {
                "positions": final_info.get("trajectory_positions", []),
                "concentrations": final_info.get("trajectory_concentrations", []),
                "actions": final_info.get("trajectory_actions", []),
                "rewards": final_info.get("trajectory_rewards", [])
            }
            
            # Calculate episode statistics
            episode_metrics = mock_aggregator.calculate_episode_stats(
                trajectory_data=trajectory_data,
                episode_id=aggregation_data["episodes_processed"]
            )
            
            aggregation_data["metrics_calculated"].append(episode_metrics)
            aggregation_data["episodes_processed"] += 1
            
            # Export summary if multiple episodes completed
            if aggregation_data["episodes_processed"] % 5 == 0:
                mock_aggregator.export_summary(
                    output_path=f"./test_summary_{aggregation_data['episodes_processed']}.json"
                )
        
        env.set_episode_end_fn(stats_integration_fn)
        
        # Simulate multiple episode completions
        for episode_num in range(3):
            obs, info = env.reset()
            
            # Run episode steps
            for _ in range(10):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                if terminated or truncated:
                    break
            
            # Manually end episode with trajectory data
            env._end_episode(final_info={
                "episode_id": episode_num,
                "trajectory_positions": [[0, 0], [1, 1], [2, 2]],
                "trajectory_concentrations": [0.1, 0.2, 0.3],
                "trajectory_actions": [[0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]],
                "trajectory_rewards": [0.1, 0.2, 0.1]
            })
        
        # Verify stats aggregator integration
        assert aggregation_data["episodes_processed"] == 3, "Should have processed 3 episodes"
        assert len(aggregation_data["metrics_calculated"]) == 3, "Should have calculated metrics for 3 episodes"
        
        # Verify mock aggregator was called correctly
        assert mock_aggregator.calculate_episode_stats.call_count == 3, "Should have called stats calculation 3 times"
        
        # Verify episode metrics structure
        for metrics in aggregation_data["metrics_calculated"]:
            assert "path_efficiency" in metrics, "Should contain path efficiency metric"
            assert "exploration_coverage" in metrics, "Should contain exploration coverage metric"
            assert "mean_concentration" in metrics, "Should contain mean concentration metric"
            assert "success_indicator" in metrics, "Should contain success indicator metric"
    
    def test_episode_completion_lifecycle(self):
        """Test episode completion lifecycle with proper timing and order."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        lifecycle_events = {"events": [], "timestamps": []}
        
        def lifecycle_tracking_fn(final_info):
            """Track episode completion lifecycle events."""
            event_time = time.perf_counter()
            
            lifecycle_events["events"].append({
                "event_type": "episode_end",
                "episode_id": final_info.get("episode_id", "unknown"),
                "termination_reason": final_info.get("termination_reason", "unknown"),
                "total_steps": final_info.get("episode_length", 0),
                "final_reward": final_info.get("total_reward", 0.0)
            })
            lifecycle_events["timestamps"].append(event_time)
        
        env.set_episode_end_fn(lifecycle_tracking_fn)
        
        # Test different episode termination scenarios
        termination_scenarios = [
            {"name": "success", "max_steps": 20, "force_success": True},
            {"name": "timeout", "max_steps": 100, "force_success": False},
            {"name": "early_termination", "max_steps": 5, "force_success": False}
        ]
        
        for scenario in termination_scenarios:
            obs, info = env.reset()
            steps = 0
            
            while steps < scenario["max_steps"]:
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                steps += 1
                
                # Force termination for test scenarios
                if scenario["force_success"] and steps >= 10:
                    terminated = True
                    break
                
                if terminated or truncated:
                    break
            
            # Ensure episode end is triggered
            if not (terminated or truncated):
                env._end_episode(final_info={
                    "episode_id": scenario["name"],
                    "termination_reason": scenario["name"],
                    "episode_length": steps,
                    "total_reward": steps * 0.1
                })
        
        # Verify lifecycle events were recorded
        assert len(lifecycle_events["events"]) >= len(termination_scenarios), \
            "Should have recorded episode end events for all scenarios"
        
        # Verify event timing and order
        timestamps = lifecycle_events["timestamps"]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1], "Events should be in chronological order"
        
        # Verify event data structure
        for event in lifecycle_events["events"]:
            assert "event_type" in event, "Event should contain type"
            assert "episode_id" in event, "Event should contain episode ID"
            assert "termination_reason" in event, "Event should contain termination reason"
            assert "total_steps" in event, "Event should contain step count"
            assert "final_reward" in event, "Event should contain final reward"
    
    def test_cleanup_and_finalization(self):
        """Test cleanup and finalization operations in episode end hooks."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        cleanup_state = {
            "resources_freed": 0,
            "data_exported": 0,
            "caches_cleared": 0,
            "logs_written": 0
        }
        
        def cleanup_episode_end_fn(final_info):
            """Episode end handler with comprehensive cleanup operations."""
            # Simulate resource cleanup
            if final_info.get("trajectory_positions"):
                cleanup_state["resources_freed"] += 1
            
            # Simulate data export
            if final_info.get("total_reward", 0) > 0:
                cleanup_state["data_exported"] += 1
            
            # Simulate cache clearing
            if final_info.get("episode_length", 0) > 50:
                cleanup_state["caches_cleared"] += 1
            
            # Simulate log writing
            cleanup_state["logs_written"] += 1
        
        env.set_episode_end_fn(cleanup_episode_end_fn)
        
        # Test cleanup with different episode outcomes
        test_episodes = [
            {"steps": 25, "reward": 0.5, "has_trajectory": True},
            {"steps": 75, "reward": 1.2, "has_trajectory": True},
            {"steps": 10, "reward": 0.0, "has_trajectory": False}
        ]
        
        for i, episode_config in enumerate(test_episodes):
            obs, info = env.reset()
            
            # Simulate episode with configured parameters
            for _ in range(episode_config["steps"]):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                if terminated or truncated:
                    break
            
            # End episode with configured final info
            final_info = {
                "episode_id": i,
                "total_reward": episode_config["reward"],
                "episode_length": episode_config["steps"]
            }
            
            if episode_config["has_trajectory"]:
                final_info["trajectory_positions"] = [[j, j] for j in range(episode_config["steps"])]
            
            env._end_episode(final_info=final_info)
        
        # Verify cleanup operations were performed
        assert cleanup_state["logs_written"] == len(test_episodes), \
            "Should have written logs for all episodes"
        assert cleanup_state["resources_freed"] == 2, \
            "Should have freed resources for episodes with trajectories"
        assert cleanup_state["data_exported"] == 2, \
            "Should have exported data for episodes with positive rewards"
        assert cleanup_state["caches_cleared"] == 1, \
            "Should have cleared caches for long episodes"


class TestHookPerformance:
    """Test performance characteristics of hook system execution."""
    
    def test_hook_execution_overhead(self):
        """Test hook execution overhead with single agent scenarios."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        timing_results = {"with_hooks": [], "without_hooks": []}
        
        # Test without hooks first
        for _ in range(50):
            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            end_time = time.perf_counter()
            timing_results["without_hooks"].append((end_time - start_time) * 1000)  # Convert to ms
        
        # Add minimal hooks
        def minimal_obs_fn(state):
            return {"hook_timestamp": time.time()}
        
        def minimal_reward_fn(base_reward, info):
            return 0.001  # Tiny modification
        
        def minimal_episode_end_fn(final_info):
            pass  # No operation
        
        env.set_hooks(
            extra_obs_fn=minimal_obs_fn,
            extra_reward_fn=minimal_reward_fn,
            episode_end_fn=minimal_episode_end_fn
        )
        
        # Test with hooks
        for _ in range(50):
            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            end_time = time.perf_counter()
            timing_results["with_hooks"].append((end_time - start_time) * 1000)  # Convert to ms
        
        # Analyze performance impact
        without_hooks_mean = np.mean(timing_results["without_hooks"])
        with_hooks_mean = np.mean(timing_results["with_hooks"])
        overhead = with_hooks_mean - without_hooks_mean
        
        # Validate performance requirements
        assert with_hooks_mean < SINGLE_AGENT_HOOK_THRESHOLD_MS, \
            f"Hook execution time {with_hooks_mean:.3f}ms should be <{SINGLE_AGENT_HOOK_THRESHOLD_MS}ms"
        assert overhead < SINGLE_AGENT_HOOK_THRESHOLD_MS * 0.5, \
            f"Hook overhead {overhead:.3f}ms should be minimal"
        
        # Statistical validation
        assert len(timing_results["without_hooks"]) == 50, "Should have baseline measurements"
        assert len(timing_results["with_hooks"]) == 50, "Should have hook measurements"
    
    def test_multi_agent_hook_performance(self):
        """Test hook performance with multi-agent scenarios up to 100 agents."""
        # Test with different agent counts
        agent_counts = [1, 10, 25, 50, 100]
        performance_results = {}
        
        for num_agents in agent_counts:
            # Create multi-agent environment
            positions = [[i * 2.0, i * 2.0] for i in range(num_agents)]
            
            env = PlumeNavigationEnv.from_config({
                "video_path": "test_video.mp4",
                "navigator": {
                    "positions": positions,
                    "max_speeds": [1.0] * num_agents
                },
                "enable_hooks": True
            })
            
            # Add vectorized hooks
            def vectorized_obs_fn(state):
                num_agents = len(state.get("positions", []))
                return {
                    "agent_count": num_agents,
                    "collective_energy": np.sum([1.0] * num_agents),
                    "center_of_mass": np.mean(state.get("positions", [[0, 0]]), axis=0)
                }
            
            def vectorized_reward_fn(base_reward, info):
                # Simple collective reward modification
                return base_reward * 1.01 + 0.001
            
            def vectorized_episode_end_fn(final_info):
                # Log collective metrics
                pass
            
            env.set_hooks(
                extra_obs_fn=vectorized_obs_fn,
                extra_reward_fn=vectorized_reward_fn,
                episode_end_fn=vectorized_episode_end_fn
            )
            
            # Warm up
            obs, info = env.reset()
            
            # Performance test
            step_times = []
            for _ in range(20):
                start_time = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                end_time = time.perf_counter()
                step_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Store results
            performance_results[num_agents] = {
                "mean_time": np.mean(step_times),
                "p95_time": np.percentile(step_times, 95),
                "max_time": np.max(step_times),
                "step_times": step_times
            }
        
        # Validate performance requirements
        for num_agents, results in performance_results.items():
            if num_agents <= 10:
                # Stricter requirements for small agent counts
                assert results["mean_time"] < SINGLE_AGENT_HOOK_THRESHOLD_MS * 2, \
                    f"Mean time for {num_agents} agents: {results['mean_time']:.3f}ms should be <{SINGLE_AGENT_HOOK_THRESHOLD_MS * 2}ms"
            elif num_agents == 100:
                # Main performance requirement
                assert results["mean_time"] < MULTI_AGENT_HOOK_THRESHOLD_MS, \
                    f"Mean time for 100 agents: {results['mean_time']:.3f}ms should be <{MULTI_AGENT_HOOK_THRESHOLD_MS}ms"
            
            # General stability requirements
            assert results["p95_time"] < results["mean_time"] * 3, \
                f"P95 time should not be excessive compared to mean for {num_agents} agents"
    
    def test_hook_performance_compliance(self):
        """Test compliance with performance requirements across different hook complexities."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        # Test different complexity levels
        complexity_tests = {
            "minimal": {
                "obs_fn": lambda state: {"simple": 1},
                "reward_fn": lambda base_reward, info: 0.01,
                "episode_end_fn": lambda final_info: None
            },
            "moderate": {
                "obs_fn": lambda state: {
                    "position_norm": np.linalg.norm(state.get("position", [0, 0])),
                    "velocity_angle": np.arctan2(state.get("velocity", [0, 1])[1], state.get("velocity", [1, 0])[0]),
                    "step_efficiency": 1.0 / (state.get("step_count", 1) + 1)
                },
                "reward_fn": lambda base_reward, info: base_reward * 0.95 + np.random.normal(0, 0.01),
                "episode_end_fn": lambda final_info: print(f"Episode ended: {final_info.get('total_reward', 0)}")
            },
            "complex": {
                "obs_fn": lambda state: {
                    "distance_matrix": np.random.rand(10, 10),
                    "fourier_features": np.fft.fft(np.random.rand(16)).real,
                    "statistical_summary": {
                        "mean": np.mean(state.get("trajectory", [0])),
                        "std": np.std(state.get("trajectory", [0])),
                        "skew": 0.0
                    }
                },
                "reward_fn": lambda base_reward, info: base_reward + np.sum(np.random.rand(100)) * 0.001,
                "episode_end_fn": lambda final_info: [print(f"Analysis {i}") for i in range(5)]
            }
        }
        
        performance_by_complexity = {}
        
        for complexity_name, hook_functions in complexity_tests.items():
            env.set_hooks(**hook_functions)
            
            # Performance test
            step_times = []
            for _ in range(30):
                start_time = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                end_time = time.perf_counter()
                step_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            performance_by_complexity[complexity_name] = {
                "mean_time": np.mean(step_times),
                "p95_time": np.percentile(step_times, 95),
                "max_time": np.max(step_times)
            }
        
        # Validate performance compliance
        for complexity, perf in performance_by_complexity.items():
            if complexity == "minimal":
                assert perf["mean_time"] < SINGLE_AGENT_HOOK_THRESHOLD_MS * 0.5, \
                    f"Minimal hooks should be very fast: {perf['mean_time']:.3f}ms"
            elif complexity == "moderate":
                assert perf["mean_time"] < SINGLE_AGENT_HOOK_THRESHOLD_MS, \
                    f"Moderate hooks should meet basic requirements: {perf['mean_time']:.3f}ms"
            elif complexity == "complex":
                assert perf["mean_time"] < SINGLE_AGENT_HOOK_THRESHOLD_MS * 2, \
                    f"Complex hooks should still be reasonable: {perf['mean_time']:.3f}ms"
            
            # All hooks should complete in reasonable time
            assert perf["max_time"] < SINGLE_AGENT_HOOK_THRESHOLD_MS * 10, \
                f"Maximum hook time should not be excessive for {complexity}: {perf['max_time']:.3f}ms"
    
    def test_hook_memory_efficiency(self):
        """Test memory efficiency of hook system operations."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        # Create hooks that may use memory
        trajectory_data = {"positions": [], "observations": [], "rewards": []}
        
        def memory_tracking_obs_fn(state):
            # Store some data
            trajectory_data["positions"].append(state.get("position", [0, 0]))
            trajectory_data["observations"].append({
                "large_array": np.random.rand(100),  # 800 bytes
                "metadata": {"timestamp": time.time(), "step": len(trajectory_data["positions"])}
            })
            return {"trajectory_length": len(trajectory_data["positions"])}
        
        def memory_tracking_reward_fn(base_reward, info):
            trajectory_data["rewards"].append(base_reward)
            return base_reward + 0.01
        
        def memory_cleanup_episode_end_fn(final_info):
            # Cleanup some data periodically
            if len(trajectory_data["positions"]) > 1000:
                trajectory_data["positions"] = trajectory_data["positions"][-500:]
                trajectory_data["observations"] = trajectory_data["observations"][-500:]
                trajectory_data["rewards"] = trajectory_data["rewards"][-500:]
                gc.collect()  # Force garbage collection
        
        env.set_hooks(
            extra_obs_fn=memory_tracking_obs_fn,
            extra_reward_fn=memory_tracking_reward_fn,
            episode_end_fn=memory_cleanup_episode_end_fn
        )
        
        # Run many steps to test memory growth
        for step in range(500):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Check memory periodically
            if step % 100 == 0:
                current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable
                assert memory_growth < HOOK_MEMORY_THRESHOLD_MB * (step / 100 + 1), \
                    f"Memory growth {memory_growth:.1f}MB should be controlled at step {step}"
        
        # Final memory check
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        total_growth = final_memory - initial_memory
        
        assert total_growth < HOOK_MEMORY_THRESHOLD_MB * 10, \
            f"Total memory growth {total_growth:.1f}MB should be within limits"
        
        # Cleanup check
        env._end_episode(final_info={"cleanup_test": True})
        gc.collect()
        
        cleanup_memory = process.memory_info().rss / (1024 * 1024)  # MB
        assert cleanup_memory <= final_memory, "Memory should not increase after cleanup"


class TestHookIntegration:
    """Test integration of hook system with other plume_nav_sim components."""
    
    def test_recorder_hook_integration(self):
        """Test integration between hook system and recording infrastructure."""
        # Create mock recorder manager
        mock_recorder = Mock(spec=RecorderManager)
        mock_recorder.start_recording.return_value = None
        mock_recorder.stop_recording.return_value = None
        mock_recorder.get_performance_metrics.return_value = {
            "recording_overhead_ms": 0.5,
            "buffer_utilization": 0.25,
            "total_records": 100
        }
        
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True,
            "recorder": mock_recorder
        })
        
        recorded_data = {"step_records": [], "episode_records": []}
        
        def recording_obs_fn(state):
            """Obs hook that integrates with recording system."""
            step_data = {
                "timestamp": time.time(),
                "position": state.get("position", [0, 0]),
                "custom_obs": {"exploration_state": "active"}
            }
            recorded_data["step_records"].append(step_data)
            return {"recording_active": True, "records_count": len(recorded_data["step_records"])}
        
        def recording_reward_fn(base_reward, info):
            """Reward hook that logs to recording system."""
            return base_reward + 0.01  # Small modification for testing
        
        def recording_episode_end_fn(final_info):
            """Episode end hook that triggers recording finalization."""
            episode_record = {
                "episode_id": final_info.get("episode_id", "unknown"),
                "total_steps": len(recorded_data["step_records"]),
                "final_reward": final_info.get("total_reward", 0.0),
                "recording_metrics": mock_recorder.get_performance_metrics()
            }
            recorded_data["episode_records"].append(episode_record)
            
            # Trigger recording operations
            mock_recorder.stop_recording()
            mock_recorder.start_recording()
        
        env.set_hooks(
            extra_obs_fn=recording_obs_fn,
            extra_reward_fn=recording_reward_fn,
            episode_end_fn=recording_episode_end_fn
        )
        
        # Test recorder integration over multiple steps
        obs, info = env.reset()
        assert "recording_active" in obs, "Recording hook should be active"
        
        # Run several steps
        for step in range(20):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Verify recording integration
            assert "recording_active" in obs, "Recording should remain active"
            assert obs["records_count"] == step + 2, "Record count should increment"  # +1 for reset, +1 for current step
        
        # Trigger episode end
        env._end_episode(final_info={
            "episode_id": "test_episode",
            "total_reward": 2.0,
            "episode_length": 20
        })
        
        # Verify recording integration
        assert len(recorded_data["step_records"]) > 0, "Should have recorded step data"
        assert len(recorded_data["episode_records"]) > 0, "Should have recorded episode data"
        
        # Verify mock recorder interactions
        assert mock_recorder.get_performance_metrics.call_count > 0, "Should have queried recording metrics"
        assert mock_recorder.stop_recording.call_count > 0, "Should have stopped recording"
        assert mock_recorder.start_recording.call_count > 0, "Should have restarted recording"
    
    def test_stats_aggregator_integration(self):
        """Test integration between hook system and statistics aggregation."""
        # Create mock stats aggregator with proper protocol compliance
        mock_aggregator = Mock(spec=StatsAggregator)
        mock_aggregator.calculate_episode_stats.return_value = {
            "path_efficiency": 0.85,
            "exploration_coverage": 0.70,
            "mean_concentration": 0.35,
            "success_indicator": 1.0,
            "total_reward": 5.5,
            "episode_length": 150
        }
        mock_aggregator.calculate_run_stats.return_value = {
            "success_rate": 0.80,
            "mean_path_efficiency": 0.78,
            "std_path_efficiency": 0.12,
            "mean_episode_length": 145.5
        }
        mock_aggregator.export_summary.return_value = True
        
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True,
            "stats_aggregator": mock_aggregator
        })
        
        stats_data = {"episode_metrics": [], "run_metrics": None}
        
        def stats_obs_fn(state):
            """Obs hook that collects data for statistics."""
            return {
                "stats_integration": True,
                "trajectory_point": {
                    "position": state.get("position", [0, 0]),
                    "timestamp": time.time()
                }
            }
        
        def stats_reward_fn(base_reward, info):
            """Reward hook that tracks performance metrics."""
            # Simple performance tracking
            return base_reward + (0.1 if info.get("novel_position", False) else 0.0)
        
        def stats_episode_end_fn(final_info):
            """Episode end hook that triggers statistics calculation."""
            # Prepare trajectory data for analysis
            trajectory_data = {
                "positions": final_info.get("trajectory", []),
                "concentrations": final_info.get("concentrations", []),
                "actions": final_info.get("actions", []),
                "rewards": final_info.get("rewards", [])
            }
            
            # Calculate episode statistics
            episode_metrics = mock_aggregator.calculate_episode_stats(
                trajectory_data=trajectory_data,
                episode_id=len(stats_data["episode_metrics"])
            )
            
            stats_data["episode_metrics"].append(episode_metrics)
            
            # Calculate run statistics every 3 episodes
            if len(stats_data["episode_metrics"]) % 3 == 0:
                run_metrics = mock_aggregator.calculate_run_stats(
                    episode_data_list=stats_data["episode_metrics"],
                    run_id=f"test_run_{len(stats_data['episode_metrics']) // 3}"
                )
                stats_data["run_metrics"] = run_metrics
                
                # Export summary
                mock_aggregator.export_summary(
                    output_path=f"./test_stats_{len(stats_data['episode_metrics'])}.json"
                )
        
        env.set_hooks(
            extra_obs_fn=stats_obs_fn,
            extra_reward_fn=stats_reward_fn,
            episode_end_fn=stats_episode_end_fn
        )
        
        # Run multiple episodes to test statistics integration
        for episode_num in range(5):
            obs, info = env.reset()
            assert "stats_integration" in obs, "Statistics integration should be active"
            
            # Run episode steps
            trajectory = []
            rewards = []
            for step in range(10):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                trajectory.append(obs.get("trajectory_point", {"position": [0, 0]}))
                rewards.append(reward)
                
                if terminated or truncated:
                    break
            
            # End episode with statistics data
            env._end_episode(final_info={
                "episode_id": episode_num,
                "trajectory": trajectory,
                "rewards": rewards,
                "total_reward": sum(rewards)
            })
        
        # Verify statistics integration
        assert len(stats_data["episode_metrics"]) == 5, "Should have calculated metrics for all episodes"
        assert stats_data["run_metrics"] is not None, "Should have calculated run statistics"
        
        # Verify mock aggregator interactions
        assert mock_aggregator.calculate_episode_stats.call_count == 5, "Should have calculated episode stats 5 times"
        assert mock_aggregator.calculate_run_stats.call_count >= 1, "Should have calculated run stats"
        assert mock_aggregator.export_summary.call_count >= 1, "Should have exported summary"
        
        # Verify statistics data structure
        for metrics in stats_data["episode_metrics"]:
            assert "path_efficiency" in metrics, "Episode metrics should contain path efficiency"
            assert "exploration_coverage" in metrics, "Episode metrics should contain exploration coverage"
            assert "success_indicator" in metrics, "Episode metrics should contain success indicator"
        
        run_metrics = stats_data["run_metrics"]
        assert "success_rate" in run_metrics, "Run metrics should contain success rate"
        assert "mean_path_efficiency" in run_metrics, "Run metrics should contain mean path efficiency"
    
    def test_environment_hook_coordination(self):
        """Test coordination between hooks and environment state management."""
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True
        })
        
        coordination_state = {
            "environment_state": {},
            "hook_interactions": [],
            "state_consistency_checks": []
        }
        
        def state_aware_obs_fn(state):
            """Hook that maintains consistency with environment state."""
            env_position = state.get("position", [0, 0])
            env_orientation = state.get("orientation", 0.0)
            env_speed = state.get("speed", 0.0)
            
            # Store current environment state
            coordination_state["environment_state"] = {
                "position": env_position,
                "orientation": env_orientation,
                "speed": env_speed,
                "timestamp": time.time()
            }
            
            # Log hook interaction
            coordination_state["hook_interactions"].append({
                "hook_type": "obs",
                "env_position": env_position,
                "env_orientation": env_orientation
            })
            
            return {
                "env_coordination": True,
                "position_magnitude": np.linalg.norm(env_position),
                "orientation_radians": np.radians(env_orientation),
                "speed_normalized": env_speed / 5.0  # Normalize by max speed
            }
        
        def state_consistent_reward_fn(base_reward, info):
            """Hook that maintains reward consistency with environment state."""
            # Check state consistency
            current_position = coordination_state["environment_state"].get("position", [0, 0])
            info_position = info.get("position", [0, 0])
            
            position_consistent = np.allclose(current_position, info_position, atol=1e-6)
            coordination_state["state_consistency_checks"].append({
                "position_consistent": position_consistent,
                "hook_type": "reward",
                "timestamp": time.time()
            })
            
            coordination_state["hook_interactions"].append({
                "hook_type": "reward",
                "base_reward": base_reward,
                "position_consistent": position_consistent
            })
            
            # Reward modification based on state consistency
            consistency_bonus = 0.01 if position_consistent else -0.01
            return base_reward + consistency_bonus
        
        def state_cleanup_episode_end_fn(final_info):
            """Hook that performs state cleanup and validation."""
            # Validate final state consistency
            final_position = final_info.get("final_position", [0, 0])
            stored_position = coordination_state["environment_state"].get("position", [0, 0])
            
            final_consistent = np.allclose(final_position, stored_position, atol=1e-6)
            
            coordination_state["hook_interactions"].append({
                "hook_type": "episode_end",
                "final_consistent": final_consistent,
                "total_interactions": len(coordination_state["hook_interactions"])
            })
            
            # Clear state for next episode
            coordination_state["environment_state"] = {}
        
        env.set_hooks(
            extra_obs_fn=state_aware_obs_fn,
            extra_reward_fn=state_consistent_reward_fn,
            episode_end_fn=state_cleanup_episode_end_fn
        )
        
        # Test state coordination over episode lifecycle
        obs, info = env.reset()
        assert "env_coordination" in obs, "Environment coordination should be active"
        
        # Run steps and verify state consistency
        for step in range(15):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Verify coordination observations
            assert "env_coordination" in obs, "Coordination should remain active"
            assert "position_magnitude" in obs, "Position data should be available"
            assert "orientation_radians" in obs, "Orientation data should be available"
            assert "speed_normalized" in obs, "Speed data should be available"
            
            if terminated or truncated:
                break
        
        # End episode
        env._end_episode(final_info={
            "final_position": coordination_state["environment_state"].get("position", [0, 0]),
            "episode_length": 15
        })
        
        # Verify coordination results
        assert len(coordination_state["hook_interactions"]) > 0, "Should have recorded hook interactions"
        assert len(coordination_state["state_consistency_checks"]) > 0, "Should have performed consistency checks"
        
        # Verify state consistency
        consistency_checks = coordination_state["state_consistency_checks"]
        consistent_count = sum(1 for check in consistency_checks if check["position_consistent"])
        consistency_rate = consistent_count / len(consistency_checks) if consistency_checks else 0
        
        assert consistency_rate >= 0.8, f"State consistency rate {consistency_rate:.2f} should be high"
        
        # Verify hook interaction types
        interaction_types = set(interaction["hook_type"] for interaction in coordination_state["hook_interactions"])
        assert "obs" in interaction_types, "Should have observation hook interactions"
        assert "reward" in interaction_types, "Should have reward hook interactions"
        assert "episode_end" in interaction_types, "Should have episode end hook interactions"
    
    def test_full_system_hook_workflow(self):
        """Test complete hook system workflow with all components integrated."""
        # Create comprehensive test environment with all integrations
        mock_recorder = Mock(spec=RecorderManager)
        mock_recorder.get_performance_metrics.return_value = {"overhead_ms": 0.3}
        
        mock_aggregator = Mock(spec=StatsAggregator)
        mock_aggregator.calculate_episode_stats.return_value = {"efficiency": 0.9}
        mock_aggregator.export_summary.return_value = True
        
        env = PlumeNavigationEnv.from_config({
            "video_path": "test_video.mp4",
            "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
            "enable_hooks": True,
            "recorder": mock_recorder,
            "stats_aggregator": mock_aggregator
        })
        
        workflow_state = {
            "workflow_steps": [],
            "performance_metrics": [],
            "integration_checks": []
        }
        
        def comprehensive_obs_fn(state):
            """Comprehensive observation hook integrating all systems."""
            workflow_state["workflow_steps"].append("obs_hook_called")
            
            # Performance measurement
            start_time = time.perf_counter()
            
            # Complex observation computation
            position = state.get("position", [0, 0])
            observation_data = {
                "workflow_active": True,
                "position_distance": np.linalg.norm(position),
                "exploration_progress": len(workflow_state["workflow_steps"]) / 100.0,
                "system_timestamp": time.time(),
                "integration_status": "active"
            }
            
            # Record performance
            end_time = time.perf_counter()
            workflow_state["performance_metrics"].append({
                "hook_type": "obs",
                "duration_ms": (end_time - start_time) * 1000,
                "timestamp": end_time
            })
            
            return observation_data
        
        def comprehensive_reward_fn(base_reward, info):
            """Comprehensive reward hook with full system integration."""
            workflow_state["workflow_steps"].append("reward_hook_called")
            
            start_time = time.perf_counter()
            
            # Multi-factor reward computation
            efficiency_bonus = 0.1 if info.get("efficient_movement", False) else 0.0
            exploration_bonus = 0.05 if info.get("novel_area", False) else 0.0
            recorder_penalty = -0.01 if mock_recorder.get_performance_metrics()["overhead_ms"] > 1.0 else 0.0
            
            total_extra = efficiency_bonus + exploration_bonus + recorder_penalty
            
            end_time = time.perf_counter()
            workflow_state["performance_metrics"].append({
                "hook_type": "reward",
                "duration_ms": (end_time - start_time) * 1000,
                "timestamp": end_time
            })
            
            return total_extra
        
        def comprehensive_episode_end_fn(final_info):
            """Comprehensive episode end hook with full cleanup."""
            workflow_state["workflow_steps"].append("episode_end_hook_called")
            
            start_time = time.perf_counter()
            
            # Integration checks
            workflow_state["integration_checks"].append({
                "recorder_available": mock_recorder is not None,
                "aggregator_available": mock_aggregator is not None,
                "workflow_steps_count": len(workflow_state["workflow_steps"]),
                "performance_samples": len(workflow_state["performance_metrics"])
            })
            
            # Trigger integrated operations
            if len(workflow_state["workflow_steps"]) > 10:
                mock_aggregator.calculate_episode_stats(
                    trajectory_data={"steps": workflow_state["workflow_steps"]},
                    episode_id="workflow_test"
                )
                mock_aggregator.export_summary("./workflow_summary.json")
            
            end_time = time.perf_counter()
            workflow_state["performance_metrics"].append({
                "hook_type": "episode_end",
                "duration_ms": (end_time - start_time) * 1000,
                "timestamp": end_time
            })
        
        # Register comprehensive hooks
        env.set_hooks(
            extra_obs_fn=comprehensive_obs_fn,
            extra_reward_fn=comprehensive_reward_fn,
            episode_end_fn=comprehensive_episode_end_fn
        )
        
        # Execute full workflow
        obs, info = env.reset()
        assert "workflow_active" in obs, "Workflow should be active"
        
        # Run comprehensive episode
        for step in range(25):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Verify ongoing workflow
            assert "workflow_active" in obs, "Workflow should remain active"
            assert "integration_status" in obs, "Integration status should be tracked"
            
            if terminated or truncated:
                break
        
        # Complete episode
        env._end_episode(final_info={
            "workflow_test": True,
            "total_steps": 25,
            "integration_complete": True
        })
        
        # Verify comprehensive workflow results
        workflow_steps = workflow_state["workflow_steps"]
        assert "obs_hook_called" in workflow_steps, "Observation hooks should be called"
        assert "reward_hook_called" in workflow_steps, "Reward hooks should be called"
        assert "episode_end_hook_called" in workflow_steps, "Episode end hooks should be called"
        
        # Verify performance metrics
        performance_metrics = workflow_state["performance_metrics"]
        assert len(performance_metrics) > 0, "Should have collected performance metrics"
        
        # Verify integration checks
        integration_checks = workflow_state["integration_checks"]
        assert len(integration_checks) > 0, "Should have performed integration checks"
        
        final_check = integration_checks[-1]
        assert final_check["recorder_available"], "Recorder should be available"
        assert final_check["aggregator_available"], "Stats aggregator should be available"
        assert final_check["workflow_steps_count"] > 20, "Should have sufficient workflow steps"
        
        # Verify system integration calls
        assert mock_aggregator.calculate_episode_stats.call_count > 0, "Should have calculated episode stats"
        assert mock_aggregator.export_summary.call_count > 0, "Should have exported summary"
        
        # Performance validation
        hook_durations = [m["duration_ms"] for m in performance_metrics]
        mean_duration = np.mean(hook_durations)
        max_duration = np.max(hook_durations)
        
        assert mean_duration < SINGLE_AGENT_HOOK_THRESHOLD_MS, \
            f"Mean hook duration {mean_duration:.3f}ms should meet performance requirements"
        assert max_duration < SINGLE_AGENT_HOOK_THRESHOLD_MS * 5, \
            f"Maximum hook duration {max_duration:.3f}ms should be reasonable"


def test_hook_system_backwards_compatibility():
    """Test backwards compatibility of hook system with v0.3.0 environments."""
    # Test that environments without hooks enabled still function
    env = PlumeNavigationEnv.from_config({
        "video_path": "test_video.mp4",
        "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
        "enable_hooks": False  # Explicitly disable hooks
    })
    
    # Environment should function normally without hooks
    obs, info = env.reset()
    assert isinstance(obs, dict), "Observations should be dict even without hooks"
    
    # Steps should work without hook modifications
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, (int, float)), "Rewards should be numeric without hooks"
    assert isinstance(obs, dict), "Observations should remain dict format"
    
    # Test that hook methods exist but are no-ops when disabled
    if hasattr(env, 'set_hooks'):
        # Should be able to call but should not affect behavior
        env.set_hooks(
            extra_obs_fn=lambda state: {"should_not_appear": True},
            extra_reward_fn=lambda r, i: 999.0,
            episode_end_fn=lambda i: print("should not execute")
        )
        
        # Observations should not include hook data
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert "should_not_appear" not in obs, "Hook data should not appear when hooks disabled"
        assert reward != 999.0, "Hook reward modifications should not apply when disabled"


def test_hook_error_handling_and_recovery():
    """Test error handling and recovery mechanisms in the hook system."""
    env = PlumeNavigationEnv.from_config({
        "video_path": "test_video.mp4",
        "navigator": {"position": [0.0, 0.0], "max_speed": 1.0},
        "enable_hooks": True,
        "hook_error_mode": "recover"  # Continue on errors
    })
    
    error_log = {"errors": [], "recoveries": []}
    
    def error_prone_obs_fn(state):
        """Hook that sometimes fails."""
        step_count = len(error_log["errors"]) + len(error_log["recoveries"])
        
        if step_count == 3:
            error_log["errors"].append("obs_hook_division_by_zero")
            raise ZeroDivisionError("Simulated division by zero")
        elif step_count == 7:
            error_log["errors"].append("obs_hook_value_error")
            raise ValueError("Simulated value error")
        elif step_count == 12:
            error_log["errors"].append("obs_hook_runtime_error")
            raise RuntimeError("Simulated runtime error")
        else:
            error_log["recoveries"].append("obs_hook_success")
            return {"error_test": True, "step_count": step_count}
    
    def error_prone_reward_fn(base_reward, info):
        """Reward hook that sometimes fails."""
        step_count = len(error_log["errors"]) + len(error_log["recoveries"])
        
        if step_count == 5:
            error_log["errors"].append("reward_hook_type_error")
            raise TypeError("Simulated type error")
        elif step_count == 10:
            error_log["errors"].append("reward_hook_key_error")
            raise KeyError("Simulated key error")
        else:
            error_log["recoveries"].append("reward_hook_success")
            return 0.01
    
    def error_prone_episode_end_fn(final_info):
        """Episode end hook that sometimes fails."""
        if len(error_log["errors"]) % 3 == 0:
            error_log["errors"].append("episode_end_hook_error")
            raise Exception("Simulated episode end error")
        else:
            error_log["recoveries"].append("episode_end_hook_success")
    
    env.set_hooks(
        extra_obs_fn=error_prone_obs_fn,
        extra_reward_fn=error_prone_reward_fn,
        episode_end_fn=error_prone_episode_end_fn
    )
    
    # Test error recovery over multiple steps
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        
        obs, info = env.reset()
        
        # Run steps that will trigger various errors
        for step in range(15):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            
            # Environment should continue functioning despite hook errors
            assert isinstance(obs, dict), "Observations should remain valid after hook errors"
            assert isinstance(reward, (int, float)), "Rewards should remain valid after hook errors"
            assert np.isfinite(reward), "Rewards should be finite after hook errors"
        
        # Trigger episode end (which may also error)
        env._end_episode(final_info={"test_episode": True})
    
    # Verify error handling
    assert len(error_log["errors"]) > 0, "Should have encountered hook errors"
    assert len(error_log["recoveries"]) > 0, "Should have successful hook calls"
    assert len(warning_list) >= len(error_log["errors"]), "Should have generated warnings for errors"
    
    # Verify error types were handled
    error_types = set(error_log["errors"])
    expected_errors = {
        "obs_hook_division_by_zero", "obs_hook_value_error", "obs_hook_runtime_error",
        "reward_hook_type_error", "reward_hook_key_error", "episode_end_hook_error"
    }
    
    # Should have encountered at least some of the expected error types
    assert len(error_types.intersection(expected_errors)) > 0, "Should have handled multiple error types"
    
    # Verify warnings mention hook failures
    hook_warnings = [w for w in warning_list if "hook" in str(w.message).lower()]
    assert len(hook_warnings) > 0, "Should have generated warnings about hook failures"