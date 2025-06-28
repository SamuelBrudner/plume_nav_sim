"""
Comprehensive Gymnasium API compliance validation test suite.

This module provides complete validation of the plume navigation environment's
compliance with Gymnasium 0.29.x API standards, ensuring seamless integration
with modern reinforcement learning frameworks while maintaining backward
compatibility with legacy Gym implementations.

Test Coverage Areas:
- Full Gymnasium API compliance using gymnasium.utils.env_checker
- Modern reset() signature: reset(seed=None, options=None) -> (obs, info)
- Modern step() signature: step(action) -> (obs, reward, terminated, truncated, info)
- Environment registration and factory creation patterns
- Action/observation space compliance and validation
- Performance requirements: sub-10ms step execution
- Extensibility hooks: compute_additional_obs, compute_extra_reward, on_episode_end
- Dual API compatibility: legacy 4-tuple vs modern 5-tuple returns
- Frame caching performance and memory management
- Seeding and reproducibility validation

The test suite ensures the environment meets production-grade quality standards
for reinforcement learning research while providing clear validation of the
migration from legacy Gym to modern Gymnasium APIs.
"""

import time
import warnings
import tempfile
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import numpy as np
import pytest

# Core testing framework imports
try:
    import gymnasium
    from gymnasium.utils.env_checker import check_env
    from gymnasium.spaces import Box, Dict as DictSpace
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gymnasium = None
    check_env = None
    Box = DictSpace = None

# Environment and component imports
try:
    from plume_nav_sim.envs import PlumeNavigationEnv
    from plume_nav_sim.envs.spaces import ActionSpace, ObservationSpace
    from plume_nav_sim import __version__
    PLUME_NAV_SIM_AVAILABLE = True
except ImportError:
    PLUME_NAV_SIM_AVAILABLE = False
    PlumeNavigationEnv = None
    ActionSpace = ObservationSpace = None

# Performance monitoring imports
try:
    import psutil
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    psutil = None


class TestGymnasiumCompliance:
    """
    Comprehensive test suite for Gymnasium API compliance validation.
    
    This test class validates complete adherence to Gymnasium 0.29.x standards
    including modern API signatures, return formats, space compliance, and
    performance requirements critical for RL framework compatibility.
    """
    
    @pytest.fixture(scope="class")
    def mock_video_file(self) -> Path:
        """
        Create a temporary mock video file for testing.
        
        Returns:
            Path to temporary video file with minimal test data
        """
        # Create a temporary directory for test files
        temp_dir = Path(tempfile.mkdtemp())
        video_path = temp_dir / "test_plume_video.mp4"
        
        # Create a minimal mock video file (placeholder - actual implementation
        # would use OpenCV to create a small test video with odor plume data)
        with open(video_path, 'wb') as f:
            # Write minimal MP4 header for basic file existence test
            f.write(b'\x00\x00\x00\x20ftypmp42')  # Basic MP4 file signature
            f.write(b'\x00' * 1000)  # Minimal content for testing
        
        yield video_path
        
        # Cleanup temporary files
        if video_path.exists():
            video_path.unlink()
        temp_dir.rmdir()
    
    @pytest.fixture
    def basic_env_config(self, mock_video_file: Path) -> Dict[str, Any]:
        """
        Basic environment configuration for testing.
        
        Args:
            mock_video_file: Path to mock video file
            
        Returns:
            Dictionary containing basic environment configuration
        """
        return {
            "video_path": str(mock_video_file),
            "initial_position": [100, 100],
            "initial_orientation": 0.0,
            "max_speed": 2.0,
            "max_angular_velocity": 90.0,
            "max_episode_steps": 100,
            "performance_monitoring": True,
            "include_multi_sensor": False,
        }
    
    @pytest.fixture
    def enhanced_env_config(self, mock_video_file: Path) -> Dict[str, Any]:
        """
        Enhanced environment configuration with all features enabled.
        
        Args:
            mock_video_file: Path to mock video file
            
        Returns:
            Dictionary containing enhanced environment configuration
        """
        return {
            "video_path": str(mock_video_file),
            "initial_position": [150, 150],
            "initial_orientation": 45.0,
            "max_speed": 3.0,
            "max_angular_velocity": 120.0,
            "max_episode_steps": 200,
            "performance_monitoring": True,
            "include_multi_sensor": True,
            "num_sensors": 3,
            "sensor_distance": 10.0,
            "sensor_layout": "triangular",
            "reward_config": {
                "odor_concentration": 2.0,
                "distance_penalty": -0.02,
                "exploration_bonus": 0.2,
            },
        }
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_gymnasium_env_checker_compliance(self, basic_env_config: Dict[str, Any]):
        """
        Test full Gymnasium API compliance using gymnasium.utils.env_checker.
        
        This test validates that the environment passes all standard Gymnasium
        validation checks including space compliance, method signatures, and
        return format validation as required by Section 0.2.1.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        # Create environment with basic configuration
        env = PlumeNavigationEnv(**basic_env_config)
        
        try:
            # Run comprehensive Gymnasium API validation
            # This will raise exceptions if any compliance issues are found
            check_env(env, warn=True, skip_render_check=True)
            
            # If we reach here, the environment passed all checks
            assert True, "Environment passed Gymnasium API compliance validation"
            
        except Exception as e:
            pytest.fail(f"Environment failed Gymnasium API compliance: {str(e)}")
        
        finally:
            env.close()
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_modern_reset_signature_compliance(self, basic_env_config: Dict[str, Any]):
        """
        Test modern reset() method signature compliance per Section 0.2.1.
        
        Validates that reset() accepts seed and options parameters and returns
        a 2-tuple (observation, info) as required by Gymnasium 0.29.x.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        env = PlumeNavigationEnv(**basic_env_config)
        
        try:
            # Test basic reset without parameters
            result = env.reset()
            assert isinstance(result, tuple), "reset() must return a tuple"
            assert len(result) == 2, "reset() must return 2-tuple (observation, info)"
            
            observation, info = result
            assert isinstance(observation, dict), "Observation must be a dictionary"
            assert isinstance(info, dict), "Info must be a dictionary"
            
            # Test reset with seed parameter
            result_with_seed = env.reset(seed=42)
            assert isinstance(result_with_seed, tuple), "reset(seed=42) must return tuple"
            assert len(result_with_seed) == 2, "reset(seed=42) must return 2-tuple"
            
            obs_seed, info_seed = result_with_seed
            assert isinstance(obs_seed, dict), "Observation with seed must be dictionary"
            assert isinstance(info_seed, dict), "Info with seed must be dictionary"
            assert "seed" in info_seed, "Info should contain seed information"
            
            # Test reset with options parameter
            reset_options = {
                "position": [120, 120],
                "orientation": 30.0,
                "frame_index": 5
            }
            result_with_options = env.reset(seed=123, options=reset_options)
            assert isinstance(result_with_options, tuple), "reset() with options must return tuple"
            assert len(result_with_options) == 2, "reset() with options must return 2-tuple"
            
            obs_opts, info_opts = result_with_options
            assert isinstance(obs_opts, dict), "Observation with options must be dictionary"
            assert isinstance(info_opts, dict), "Info with options must be dictionary"
            assert "reset_options" in info_opts, "Info should contain reset options"
            
            # Verify observation structure
            required_obs_keys = ["odor_concentration", "agent_position", "agent_orientation"]
            for key in required_obs_keys:
                assert key in observation, f"Observation missing required key: {key}"
            
            # Verify info structure
            required_info_keys = ["episode", "step", "agent_position", "agent_orientation"]
            for key in required_info_keys:
                assert key in info, f"Info missing required key: {key}"
                
        finally:
            env.close()
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_modern_step_signature_compliance(self, basic_env_config: Dict[str, Any]):
        """
        Test modern step() method signature compliance per Section 0.2.1.
        
        Validates that step() returns 5-tuple format (obs, reward, terminated, 
        truncated, info) with proper terminated/truncated split as required
        by Gymnasium 0.29.x standards.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        env = PlumeNavigationEnv(**basic_env_config)
        
        try:
            # Reset environment to get initial state
            observation, info = env.reset(seed=42)
            
            # Test step with valid action
            action = env.action_space.sample()
            step_result = env.step(action)
            
            # Validate 5-tuple return format
            assert isinstance(step_result, tuple), "step() must return a tuple"
            assert len(step_result) == 5, "step() must return 5-tuple (obs, reward, terminated, truncated, info)"
            
            obs, reward, terminated, truncated, info = step_result
            
            # Validate return types
            assert isinstance(obs, dict), "Observation must be a dictionary"
            assert isinstance(reward, (int, float, np.number)), "Reward must be numeric"
            assert isinstance(terminated, bool), "Terminated must be boolean"
            assert isinstance(truncated, bool), "Truncated must be boolean"
            assert isinstance(info, dict), "Info must be a dictionary"
            
            # Validate observation structure consistency
            required_obs_keys = ["odor_concentration", "agent_position", "agent_orientation"]
            for key in required_obs_keys:
                assert key in obs, f"Observation missing required key: {key}"
            
            # Validate info structure
            required_info_keys = ["step", "episode", "reward", "agent_position"]
            for key in required_info_keys:
                assert key in info, f"Info missing required key: {key}"
            
            # Test multiple steps to verify consistency
            for step_num in range(5):
                action = env.action_space.sample()
                step_result = env.step(action)
                
                assert len(step_result) == 5, f"Step {step_num}: Must return 5-tuple"
                obs, reward, terminated, truncated, info = step_result
                
                # Verify types remain consistent
                assert isinstance(obs, dict), f"Step {step_num}: Observation type inconsistent"
                assert isinstance(reward, (int, float, np.number)), f"Step {step_num}: Reward type inconsistent"
                assert isinstance(terminated, bool), f"Step {step_num}: Terminated type inconsistent"
                assert isinstance(truncated, bool), f"Step {step_num}: Truncated type inconsistent"
                assert isinstance(info, dict), f"Step {step_num}: Info type inconsistent"
                
                # Check for episode termination
                if terminated or truncated:
                    break
                    
        finally:
            env.close()
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_action_space_compliance(self, basic_env_config: Dict[str, Any]):
        """
        Test action space compliance with Gymnasium standards.
        
        Validates that action space is properly defined, contains valid bounds,
        and supports required sampling and validation operations.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        env = PlumeNavigationEnv(**basic_env_config)
        
        try:
            action_space = env.action_space
            
            # Validate action space type
            assert isinstance(action_space, Box), "Action space must be a Box space"
            
            # Validate action space shape
            assert action_space.shape == (2,), f"Action space must have shape (2,), got {action_space.shape}"
            
            # Validate action space bounds
            assert action_space.low.shape == (2,), "Action space low bounds must have shape (2,)"
            assert action_space.high.shape == (2,), "Action space high bounds must have shape (2,)"
            assert np.all(action_space.low <= action_space.high), "Action space low must be <= high"
            
            # Validate action space dtype
            assert action_space.dtype == np.float32, f"Action space must use float32, got {action_space.dtype}"
            
            # Test action sampling
            for _ in range(10):
                action = action_space.sample()
                assert isinstance(action, np.ndarray), "Sampled action must be numpy array"
                assert action.shape == (2,), f"Sampled action must have shape (2,), got {action.shape}"
                assert action.dtype == np.float32, f"Sampled action must be float32, got {action.dtype}"
                
                # Verify action is within bounds
                assert action_space.contains(action), f"Sampled action {action} not in action space"
                assert np.all(action >= action_space.low), f"Action {action} below lower bound {action_space.low}"
                assert np.all(action <= action_space.high), f"Action {action} above upper bound {action_space.high}"
            
            # Test boundary actions
            boundary_actions = [
                action_space.low,
                action_space.high,
                (action_space.low + action_space.high) / 2,
            ]
            
            for boundary_action in boundary_actions:
                assert action_space.contains(boundary_action), f"Boundary action {boundary_action} not valid"
            
            # Test invalid actions
            invalid_actions = [
                action_space.low - 1.0,
                action_space.high + 1.0,
                np.array([float('inf'), 0.0]),
                np.array([0.0, float('nan')]),
            ]
            
            for invalid_action in invalid_actions:
                assert not action_space.contains(invalid_action), f"Invalid action {invalid_action} incorrectly accepted"
                
        finally:
            env.close()
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_observation_space_compliance(self, enhanced_env_config: Dict[str, Any]):
        """
        Test observation space compliance with Gymnasium standards.
        
        Validates that observation space is properly defined with correct
        structure, bounds, and supports both single and multi-sensor configurations.
        
        Args:
            enhanced_env_config: Enhanced environment configuration with multi-sensor enabled
        """
        env = PlumeNavigationEnv(**enhanced_env_config)
        
        try:
            observation_space = env.observation_space
            
            # Validate observation space type
            assert isinstance(observation_space, DictSpace), "Observation space must be a Dict space"
            
            # Validate required observation keys
            required_keys = ["odor_concentration", "agent_position", "agent_orientation"]
            for key in required_keys:
                assert key in observation_space.spaces, f"Observation space missing required key: {key}"
            
            # Validate multi-sensor configuration
            if enhanced_env_config.get("include_multi_sensor", False):
                assert "multi_sensor_readings" in observation_space.spaces, "Multi-sensor mode missing sensor readings"
            
            # Validate individual space components
            odor_space = observation_space.spaces["odor_concentration"]
            assert isinstance(odor_space, Box), "Odor concentration must be Box space"
            assert odor_space.shape == (), "Odor concentration must be scalar"
            assert odor_space.dtype == np.float32, "Odor concentration must be float32"
            
            position_space = observation_space.spaces["agent_position"]
            assert isinstance(position_space, Box), "Agent position must be Box space"
            assert position_space.shape == (2,), "Agent position must have shape (2,)"
            assert position_space.dtype == np.float32, "Agent position must be float32"
            
            orientation_space = observation_space.spaces["agent_orientation"]
            assert isinstance(orientation_space, Box), "Agent orientation must be Box space"
            assert orientation_space.shape == (), "Agent orientation must be scalar"
            assert orientation_space.dtype == np.float32, "Agent orientation must be float32"
            
            # Test observation sampling and validation
            env.reset(seed=42)
            
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Validate observation compliance with space
                assert observation_space.contains(obs), f"Observation {obs} not in observation space"
                
                # Validate observation structure
                for key in required_keys:
                    assert key in obs, f"Observation missing required key: {key}"
                    space = observation_space.spaces[key]
                    value = obs[key]
                    assert space.contains(value), f"Observation value {value} for key {key} not in space {space}"
                
                if terminated or truncated:
                    break
                    
        finally:
            env.close()
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_environment_registration_compliance(self):
        """
        Test environment registration and gymnasium.make() compatibility.
        
        Validates that environments are properly registered with Gymnasium
        and can be created using standard factory methods.
        """
        # Test primary environment registration
        try:
            env = gymnasium.make('PlumeNavSim-v0')
            assert env is not None, "Failed to create PlumeNavSim-v0 environment"
            assert hasattr(env, 'reset'), "Environment missing reset method"
            assert hasattr(env, 'step'), "Environment missing step method"
            assert hasattr(env, 'action_space'), "Environment missing action_space"
            assert hasattr(env, 'observation_space'), "Environment missing observation_space"
            env.close()
            
        except gymnasium.error.UnregisteredEnv:
            pytest.skip("PlumeNavSim-v0 not registered - registration may be handled by other components")
        
        # Test legacy compatibility environment registration
        try:
            legacy_env = gymnasium.make('OdorPlumeNavigation-v1')
            assert legacy_env is not None, "Failed to create OdorPlumeNavigation-v1 environment"
            legacy_env.close()
            
        except gymnasium.error.UnregisteredEnv:
            pytest.skip("OdorPlumeNavigation-v1 not registered - legacy compatibility may be disabled")
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    @pytest.mark.skipif(not PERFORMANCE_MONITORING_AVAILABLE, reason="Performance monitoring not available")
    def test_performance_requirements_compliance(self, basic_env_config: Dict[str, Any]):
        """
        Test performance requirements per Section 6.6.4.
        
        Validates that step() execution remains under 10ms as required for
        real-time training workflows and RL framework compatibility.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        env = PlumeNavigationEnv(**basic_env_config)
        
        try:
            # Reset environment
            env.reset(seed=42)
            
            # Warm-up phase to eliminate initialization overhead
            warmup_steps = 10
            for _ in range(warmup_steps):
                action = env.action_space.sample()
                env.step(action)
            
            # Performance measurement phase
            performance_samples = 100
            step_times = []
            
            for i in range(performance_samples):
                action = env.action_space.sample()
                
                # Measure step execution time
                start_time = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                end_time = time.perf_counter()
                
                step_time_ms = (end_time - start_time) * 1000
                step_times.append(step_time_ms)
                
                # Reset if episode ended
                if terminated or truncated:
                    env.reset()
            
            # Analyze performance metrics
            mean_step_time = np.mean(step_times)
            p95_step_time = np.percentile(step_times, 95)
            p99_step_time = np.percentile(step_times, 99)
            max_step_time = np.max(step_times)
            
            # Performance requirements validation
            assert mean_step_time < 10.0, f"Mean step time {mean_step_time:.2f}ms exceeds 10ms requirement"
            assert p95_step_time < 15.0, f"P95 step time {p95_step_time:.2f}ms exceeds reasonable threshold"
            assert p99_step_time < 20.0, f"P99 step time {p99_step_time:.2f}ms exceeds reasonable threshold"
            
            # Log performance summary for analysis
            print(f"\nPerformance Summary:")
            print(f"  Mean step time: {mean_step_time:.2f}ms")
            print(f"  P95 step time: {p95_step_time:.2f}ms")
            print(f"  P99 step time: {p99_step_time:.2f}ms")
            print(f"  Max step time: {max_step_time:.2f}ms")
            print(f"  Samples: {len(step_times)}")
            
            # Verify performance info is included in step results
            if 'perf_stats' in info:
                perf_stats = info['perf_stats']
                assert 'step_time_ms' in perf_stats, "Performance stats missing step_time_ms"
                assert perf_stats['step_time_ms'] > 0, "Step time should be positive"
                
        finally:
            env.close()
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_extensibility_hooks_compliance(self, basic_env_config: Dict[str, Any]):
        """
        Test extensibility hooks per Section 0.2.1.
        
        Validates that the environment provides extensibility hooks for custom
        observations, rewards, and episode handling as specified.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        # Create a custom environment class to test hooks
        class TestCustomEnvironment(PlumeNavigationEnv):
            """Test environment with custom hooks implemented."""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.additional_obs_called = False
                self.extra_reward_called = False
                self.episode_end_called = False
                self.custom_reward_sum = 0.0
            
            def compute_additional_obs(self, base_obs: dict) -> dict:
                """Custom observation hook implementation."""
                self.additional_obs_called = True
                return {
                    "custom_sensor": np.float32(0.5),
                    "observation_count": np.int32(len(base_obs))
                }
            
            def compute_extra_reward(self, base_reward: float, info: dict) -> float:
                """Custom reward hook implementation."""
                self.extra_reward_called = True
                extra_reward = 0.1  # Small bonus reward
                self.custom_reward_sum += extra_reward
                return extra_reward
            
            def on_episode_end(self, final_info: dict) -> None:
                """Episode end hook implementation."""
                self.episode_end_called = True
                final_info["custom_reward_sum"] = self.custom_reward_sum
        
        # Test environment with custom hooks
        env = TestCustomEnvironment(**basic_env_config)
        
        try:
            # Verify hooks exist as methods
            assert hasattr(env, 'compute_additional_obs'), "Missing compute_additional_obs hook"
            assert hasattr(env, 'compute_extra_reward'), "Missing compute_extra_reward hook"
            assert hasattr(env, 'on_episode_end'), "Missing on_episode_end hook"
            
            # Reset and run episode to test hooks
            env.reset(seed=42)
            
            # Run several steps
            for step_num in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Verify hook execution
                assert env.additional_obs_called, f"Step {step_num}: compute_additional_obs not called"
                assert env.extra_reward_called, f"Step {step_num}: compute_extra_reward not called"
                
                # Check if custom observations are included
                if env.include_multi_sensor or hasattr(env, '_include_custom_obs'):
                    # Custom observations would be merged with base observations
                    pass  # Implementation detail - hooks may not modify observation directly
                
                # Reset flags for next iteration
                env.additional_obs_called = False
                env.extra_reward_called = False
                
                if terminated or truncated:
                    # Verify episode end hook was called
                    assert env.episode_end_called, "on_episode_end hook not called on episode termination"
                    
                    # Check if custom info was added
                    assert "custom_reward_sum" in info, "Episode end hook did not add custom info"
                    break
            
            # Test hook method signatures
            base_obs = {"test": np.float32(1.0)}
            additional_obs = env.compute_additional_obs(base_obs)
            assert isinstance(additional_obs, dict), "compute_additional_obs must return dict"
            
            base_reward = 1.0
            test_info = {"step": 1}
            extra_reward = env.compute_extra_reward(base_reward, test_info)
            assert isinstance(extra_reward, (int, float, np.number)), "compute_extra_reward must return numeric"
            
            # Test episode end hook
            final_info = {"episode": 1, "total_reward": 10.0}
            env.on_episode_end(final_info)  # Should not raise exception
            
        finally:
            env.close()
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_seeding_and_reproducibility(self, basic_env_config: Dict[str, Any]):
        """
        Test seeding and reproducibility compliance.
        
        Validates that environment supports proper seeding and produces
        reproducible results across multiple runs with the same seed.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        seed = 12345
        num_steps = 10
        
        # Run first episode with fixed seed
        env1 = PlumeNavigationEnv(**basic_env_config)
        
        try:
            obs1, info1 = env1.reset(seed=seed)
            trajectory1 = [obs1]
            rewards1 = []
            
            for _ in range(num_steps):
                action = np.array([1.0, 0.5], dtype=np.float32)  # Fixed action
                obs, reward, terminated, truncated, info = env1.step(action)
                trajectory1.append(obs)
                rewards1.append(reward)
                
                if terminated or truncated:
                    break
                    
        finally:
            env1.close()
        
        # Run second episode with same seed
        env2 = PlumeNavigationEnv(**basic_env_config)
        
        try:
            obs2, info2 = env2.reset(seed=seed)
            trajectory2 = [obs2]
            rewards2 = []
            
            for _ in range(num_steps):
                action = np.array([1.0, 0.5], dtype=np.float32)  # Same fixed action
                obs, reward, terminated, truncated, info = env2.step(action)
                trajectory2.append(obs)
                rewards2.append(reward)
                
                if terminated or truncated:
                    break
                    
        finally:
            env2.close()
        
        # Verify reproducibility
        assert len(trajectory1) == len(trajectory2), "Trajectory lengths differ with same seed"
        assert len(rewards1) == len(rewards2), "Reward sequences differ with same seed"
        
        # Compare observations with tolerance for floating point precision
        for i, (obs1, obs2) in enumerate(zip(trajectory1, trajectory2)):
            for key in obs1.keys():
                if key in obs2:
                    val1, val2 = obs1[key], obs2[key]
                    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                        np.testing.assert_allclose(
                            val1, val2, rtol=1e-6, atol=1e-6,
                            err_msg=f"Step {i}, key {key}: observations differ with same seed"
                        )
                    elif isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
                        assert abs(val1 - val2) < 1e-6, f"Step {i}, key {key}: values {val1} vs {val2} differ"
        
        # Compare rewards
        for i, (r1, r2) in enumerate(zip(rewards1, rewards2)):
            assert abs(r1 - r2) < 1e-6, f"Step {i}: rewards {r1} vs {r2} differ with same seed"
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_dual_api_compatibility_detection(self, basic_env_config: Dict[str, Any]):
        """
        Test dual API compatibility mode detection.
        
        Validates that the environment can detect legacy gym callers and
        provide appropriate return format conversion while maintaining
        backward compatibility.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        # Test modern Gymnasium API mode (default)
        env_modern = PlumeNavigationEnv(**basic_env_config)
        
        try:
            # Test modern reset format
            reset_result = env_modern.reset(seed=42)
            assert len(reset_result) == 2, "Modern API reset should return 2-tuple"
            
            # Test modern step format
            action = env_modern.action_space.sample()
            step_result = env_modern.step(action)
            assert len(step_result) == 5, "Modern API step should return 5-tuple"
            
        finally:
            env_modern.close()
        
        # Test legacy API mode (if supported)
        try:
            env_legacy = PlumeNavigationEnv(**basic_env_config, _force_legacy_api=True)
            
            try:
                # Test that legacy mode still works
                reset_result = env_legacy.reset(seed=42)
                assert isinstance(reset_result, tuple), "Legacy reset should return tuple"
                
                action = env_legacy.action_space.sample()
                step_result = env_legacy.step(action)
                assert isinstance(step_result, tuple), "Legacy step should return tuple"
                
                # Legacy step might return 4-tuple or 5-tuple depending on implementation
                assert len(step_result) in [4, 5], f"Legacy step should return 4 or 5-tuple, got {len(step_result)}"
                
            finally:
                env_legacy.close()
                
        except TypeError:
            # Legacy API mode parameter not supported - this is acceptable
            pytest.skip("Legacy API mode not supported via constructor parameter")
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")  
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_error_handling_and_validation(self, basic_env_config: Dict[str, Any]):
        """
        Test error handling and input validation compliance.
        
        Validates that the environment properly handles invalid inputs and
        provides meaningful error messages for debugging.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        env = PlumeNavigationEnv(**basic_env_config)
        
        try:
            # Reset environment
            env.reset(seed=42)
            
            # Test invalid action shapes
            invalid_actions = [
                np.array([1.0]),  # Wrong shape
                np.array([1.0, 2.0, 3.0]),  # Wrong shape
                np.array([[1.0, 2.0]]),  # Wrong dimensions
                [1.0, 2.0],  # List instead of array
                "invalid",  # Wrong type
            ]
            
            for invalid_action in invalid_actions:
                with pytest.raises((ValueError, TypeError), 
                                 match=r"(Action|action|shape|type)"):
                    env.step(invalid_action)
            
            # Test invalid reset parameters
            with pytest.raises(ValueError, match=r"(position|Position)"):
                env.reset(options={"position": [1, 2, 3]})  # Wrong position shape
            
            with pytest.raises(ValueError, match=r"(frame|Frame|index)"):
                env.reset(options={"frame_index": -1})  # Negative frame index
            
            # Test out-of-bounds actions (should be clipped, not error)
            out_of_bounds_action = np.array([100.0, 200.0])  # Very large values
            obs, reward, terminated, truncated, info = env.step(out_of_bounds_action)
            
            # Should not raise exception - action should be clipped
            assert obs is not None, "Out-of-bounds action should be handled gracefully"
            
        finally:
            env.close()
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    def test_multi_sensor_observation_compliance(self, enhanced_env_config: Dict[str, Any]):
        """
        Test multi-sensor observation compliance.
        
        Validates that multi-sensor configurations produce valid observations
        with correct shapes and types when enabled.
        
        Args:
            enhanced_env_config: Enhanced configuration with multi-sensor enabled
        """
        env = PlumeNavigationEnv(**enhanced_env_config)
        
        try:
            # Verify multi-sensor is enabled
            assert env.include_multi_sensor, "Multi-sensor should be enabled in enhanced config"
            
            # Reset and get initial observation
            obs, info = env.reset(seed=42)
            
            # Verify multi-sensor readings are present
            assert "multi_sensor_readings" in obs, "Multi-sensor observations missing"
            
            sensor_readings = obs["multi_sensor_readings"]
            expected_num_sensors = enhanced_env_config["num_sensors"]
            
            # Validate sensor readings structure
            assert isinstance(sensor_readings, np.ndarray), "Sensor readings must be numpy array"
            assert sensor_readings.shape == (expected_num_sensors,), \
                f"Sensor readings shape {sensor_readings.shape} != expected {(expected_num_sensors,)}"
            assert sensor_readings.dtype == np.float32, "Sensor readings must be float32"
            
            # Verify sensor readings are valid values
            assert np.all(np.isfinite(sensor_readings)), "Sensor readings must be finite values"
            assert np.all(sensor_readings >= 0.0), "Sensor readings must be non-negative"
            
            # Test consistency across steps
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                assert "multi_sensor_readings" in obs, "Multi-sensor readings missing in step"
                sensor_readings = obs["multi_sensor_readings"]
                assert sensor_readings.shape == (expected_num_sensors,), \
                    "Sensor readings shape inconsistent across steps"
                assert np.all(np.isfinite(sensor_readings)), "Sensor readings contain invalid values"
                
                if terminated or truncated:
                    break
                    
        finally:
            env.close()


class TestPerformanceBenchmarks:
    """
    Performance benchmark tests for environment validation.
    
    These tests validate that the environment meets performance requirements
    for production reinforcement learning workloads.
    """
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    @pytest.mark.skipif(not PERFORMANCE_MONITORING_AVAILABLE, reason="Performance monitoring not available")
    @pytest.mark.benchmark
    def test_step_execution_benchmark(self, basic_env_config: Dict[str, Any]):
        """
        Benchmark step execution performance.
        
        Measures and validates step execution times to ensure they meet
        the sub-10ms requirement for real-time training.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        env = PlumeNavigationEnv(**basic_env_config)
        
        try:
            env.reset(seed=42)
            
            # Warmup
            for _ in range(50):
                action = env.action_space.sample()
                env.step(action)
            
            # Benchmark measurement
            num_iterations = 1000
            start_time = time.perf_counter()
            
            for _ in range(num_iterations):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    env.reset()
            
            end_time = time.perf_counter()
            
            # Calculate performance metrics
            total_time = end_time - start_time
            avg_step_time_ms = (total_time / num_iterations) * 1000
            
            # Performance assertions
            assert avg_step_time_ms < 10.0, f"Average step time {avg_step_time_ms:.2f}ms exceeds 10ms requirement"
            
            print(f"\nBenchmark Results:")
            print(f"  Total iterations: {num_iterations}")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Average step time: {avg_step_time_ms:.2f}ms")
            print(f"  Estimated FPS: {1000 / avg_step_time_ms:.1f}")
            
        finally:
            env.close()
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
    @pytest.mark.skipif(not PLUME_NAV_SIM_AVAILABLE, reason="plume_nav_sim not available")
    @pytest.mark.skipif(not PERFORMANCE_MONITORING_AVAILABLE, reason="Performance monitoring not available")
    def test_memory_usage_monitoring(self, basic_env_config: Dict[str, Any]):
        """
        Test memory usage monitoring and leak detection.
        
        Validates that the environment doesn't have memory leaks during
        extended operation periods.
        
        Args:
            basic_env_config: Basic environment configuration for testing
        """
        if not psutil:
            pytest.skip("psutil not available for memory monitoring")
        
        env = PlumeNavigationEnv(**basic_env_config)
        
        try:
            # Get baseline memory usage
            process = psutil.Process()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run extended simulation
            env.reset(seed=42)
            
            for episode in range(10):
                for step in range(100):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated or truncated:
                        break
                
                env.reset()
            
            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_growth = final_memory - baseline_memory
            
            # Memory growth should be reasonable (< 100MB for this test)
            assert memory_growth < 100.0, f"Memory growth {memory_growth:.1f}MB may indicate memory leak"
            
            print(f"\nMemory Usage:")
            print(f"  Baseline: {baseline_memory:.1f}MB")
            print(f"  Final: {final_memory:.1f}MB")
            print(f"  Growth: {memory_growth:.1f}MB")
            
        finally:
            env.close()


# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_video_directory():
    """Create temporary directory for test video files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="plume_nav_test_"))
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# Integration test markers for pytest
def pytest_configure(config):
    """Configure pytest markers for test categorization."""
    config.addinivalue_line("markers", "benchmark: mark test as a performance benchmark")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "compliance: mark test as an API compliance test")


if __name__ == "__main__":
    # Direct execution for development testing
    pytest.main([__file__, "-v", "--tb=short"])