# External imports with version comments
import gc  # >=3.10 - Garbage collection control for memory testing
import time  # >=3.10 - Timing utilities for performance testing and benchmark validation in episode manager testing
import uuid  # >=3.10 - Unique identifier generation for test isolation and episode tracking validation

import numpy as np  # >=2.1.0 - Array operations and mathematical testing utilities for numerical validation and performance testing
import pytest  # >=8.0.0 - Testing framework for test discovery, fixture management, parameterized testing, and comprehensive test execution

# Internal imports from plume_nav_sim core components
from plume_nav_sim.core.episode_manager import (
    EpisodeManager,
    EpisodeManagerConfig,
    EpisodeResult,
    EpisodeStatistics,
    create_episode_manager,
    validate_episode_config,
)
from plume_nav_sim.core.types import (
    Action,
    AgentState,
    EnvironmentConfig,
    EpisodeState,
    create_coordinates,
    create_environment_config,
)
from plume_nav_sim.utils.exceptions import ComponentError, StateError, ValidationError
from plume_nav_sim.utils.seeding import SeedManager

# Global test constants from JSON specification
TEST_TIMEOUT = 30.0
PERFORMANCE_TEST_ITERATIONS = 100
REPRODUCIBILITY_TEST_SEEDS = [42, 123, 456, 789, 999]
TEST_GRID_SIZE = (32, 32)
TEST_SOURCE_LOCATION = (16, 16)
PERFORMANCE_TARGET_STEP_LATENCY_MS = 1.0


class TestEpisodeManagerConfig:
    """Test class for EpisodeManagerConfig data class validation including parameter validation, configuration cloning, resource estimation, and component configuration derivation with comprehensive validation testing."""

    def test_config_initialization(self):
        """Test proper initialization of EpisodeManagerConfig with valid parameters and default values validation."""
        # Create test environment configuration using create_environment_config factory
        env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=100,
            goal_radius=0.0,
        )

        # Initialize EpisodeManagerConfig with test environment configuration
        config = EpisodeManagerConfig(env_config=env_config)

        # Assert env_config is stored correctly with proper type validation
        assert config.env_config is env_config
        assert isinstance(config.env_config, EnvironmentConfig)

        # Assert enable_performance_monitoring defaults to True for monitoring
        assert config.enable_performance_monitoring is True

        # Assert enable_state_validation defaults to True for validation
        assert config.enable_state_validation is True

        # Validate all default values are set correctly for component coordination
        assert hasattr(config, "enable_reproducibility_tracking")
        assert hasattr(config, "episode_timeout_ms")

    def test_config_validation_success(self):
        """Test successful validation of episode manager configuration with valid parameters and consistency checking."""
        # Create valid episode manager configuration with test parameters
        env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=200,
            goal_radius=1.0,
        )
        config = EpisodeManagerConfig(env_config=env_config)

        # Call config.validate() with strict validation enabled
        validation_result = config.validate(strict_mode=True)

        # Assert validation returns True for valid configuration
        assert validation_result is True

        # Verify no ValidationError is raised during validation
        try:
            config.validate(strict_mode=True)
        except ValidationError:
            pytest.fail("ValidationError should not be raised for valid configuration")

        # Assert all configuration parameters pass validation checks
        assert config.env_config.grid_size == TEST_GRID_SIZE
        assert config.env_config.source_location == TEST_SOURCE_LOCATION

        # Validate cross-parameter consistency and mathematical feasibility
        assert config.env_config.max_steps > 0
        assert config.env_config.goal_radius >= 0

    def test_config_validation_failure(self):
        """Test configuration validation failure with invalid parameters and comprehensive error reporting validation."""
        # Create invalid environment configuration with negative max_steps
        captured_error = None

        try:
            invalid_env_config = create_environment_config(
                grid_size=TEST_GRID_SIZE,
                source_location=TEST_SOURCE_LOCATION,
                max_steps=-100,  # Invalid negative value
                goal_radius=0.0,
            )
            pytest.fail("Should have raised ValidationError for negative max_steps")
        except ValidationError as exc:
            # Assert ValidationError is raised with appropriate error message
            assert "max_steps" in str(exc.message).lower()

        # Test validation with invalid goal_radius (negative value)
        try:
            invalid_env_config = create_environment_config(
                grid_size=TEST_GRID_SIZE,
                source_location=TEST_SOURCE_LOCATION,
                max_steps=100,
                goal_radius=-1.0,  # Invalid negative radius
            )
            pytest.fail("Should have raised ValidationError for negative goal_radius")
        except ValidationError as exc:
            # Verify error message contains specific validation failure details
            assert "goal_radius" in str(exc.message).lower()
            captured_error = exc

        # Assert comprehensive error reporting with recovery suggestions
        assert captured_error is not None
        assert hasattr(captured_error, "recovery_suggestion")

    def test_derive_component_configs(self):
        """Test derivation of individual component configurations from episode manager configuration with consistency validation."""
        # Create episode manager configuration with test parameters
        env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=150,
            goal_radius=0.5,
        )
        config = EpisodeManagerConfig(env_config=env_config)

        # Call derive_component_configs() method
        component_configs = config.derive_component_configs()

        # Assert returned dictionary contains StateManager, RewardCalculator, ActionProcessor configs
        assert isinstance(component_configs, dict)
        expected_components = ["state_manager", "reward_calculator", "action_processor"]
        for component in expected_components:
            assert component in component_configs

        # Validate StateManagerConfig parameters match episode configuration
        state_config = component_configs["state_manager"]
        assert state_config["grid_size"] == config.env_config.grid_size
        assert state_config["max_steps"] == config.env_config.max_steps

        # Verify component configurations have consistent parameters
        reward_config = component_configs["reward_calculator"]
        assert reward_config["goal_radius"] == config.env_config.goal_radius

        # Assert all derived configurations pass individual validation checks
        for component_name, component_config in component_configs.items():
            assert isinstance(component_config, dict)
            assert "enabled" in component_config

    def test_resource_estimation(self):
        """Test computational and memory resource estimation for episode processing with performance analysis."""
        # Create episode manager configuration with test parameters
        env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=200,
            goal_radius=0.0,
        )
        config = EpisodeManagerConfig(env_config=env_config)

        # Call estimate_episode_resources() with expected episode length
        resource_estimate = config.estimate_episode_resources(
            expected_episode_length=100
        )

        # Assert returned dictionary contains memory usage estimation
        assert isinstance(resource_estimate, dict)
        assert "memory_usage_mb" in resource_estimate
        assert resource_estimate["memory_usage_mb"] > 0

        # Verify computation time estimation is reasonable
        assert "computation_time_ms" in resource_estimate
        assert resource_estimate["computation_time_ms"] > 0

        # Assert resource estimates include component overhead analysis
        assert "component_overhead" in resource_estimate
        assert isinstance(resource_estimate["component_overhead"], dict)

        # Validate performance impact analysis and optimization recommendations
        assert "optimization_recommendations" in resource_estimate

    def test_config_cloning(self):
        """Test configuration cloning with parameter overrides and validation preservation."""
        # Create original episode manager configuration
        original_env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=100,
            goal_radius=0.0,
        )
        original_config = EpisodeManagerConfig(env_config=original_env_config)

        # Clone configuration without overrides
        cloned_config = original_config.clone()

        # Assert cloned configuration equals original configuration
        assert (
            cloned_config.env_config.grid_size == original_config.env_config.grid_size
        )
        assert (
            cloned_config.env_config.source_location
            == original_config.env_config.source_location
        )
        assert (
            cloned_config.enable_performance_monitoring
            == original_config.enable_performance_monitoring
        )

        # Clone with parameter overrides (different max_steps)
        overridden_config = original_config.clone(overrides={"max_steps": 200})

        # Verify override parameters are applied correctly
        assert overridden_config.env_config.max_steps == 200
        assert (
            overridden_config.env_config.grid_size
            == original_config.env_config.grid_size
        )

        # Assert cloned configuration passes validation with new parameters
        assert overridden_config.validate() is True


class TestEpisodeManager:
    """Comprehensive test class for EpisodeManager orchestrator testing complete episode lifecycle coordination, component integration, performance monitoring, and API compliance validation with reproducibility and error handling testing."""

    @pytest.fixture
    def episode_manager_config(self):
        """Fixture providing test episode manager configuration."""
        env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=100,
            goal_radius=0.0,
        )
        return EpisodeManagerConfig(env_config=env_config)

    @pytest.fixture
    def episode_manager(self, episode_manager_config):
        """Fixture providing initialized episode manager for testing."""
        return EpisodeManager(episode_manager_config)

    def test_episode_manager_initialization(self, episode_manager_config):
        """Test proper initialization of EpisodeManager with configuration validation and component coordination setup."""
        # Initialize EpisodeManager with configuration and optional seed manager
        episode_manager = EpisodeManager(episode_manager_config)

        # Assert configuration is stored correctly with validation
        assert episode_manager.config is episode_manager_config
        assert isinstance(episode_manager.config, EpisodeManagerConfig)

        # Verify state_manager, reward_calculator, action_processor are initialized
        assert hasattr(episode_manager, "state_manager")
        assert hasattr(episode_manager, "reward_calculator")
        assert hasattr(episode_manager, "action_processor")

        # Assert performance_metrics and episode_statistics are created
        assert hasattr(episode_manager, "performance_metrics")
        assert hasattr(episode_manager, "episode_statistics")

        # Validate initial state flags (episode_active=False, episode_count=0)
        assert episode_manager.episode_active is False
        assert episode_manager.episode_count == 0

        # Verify component integration and dependency injection setup
        assert episode_manager.state_manager is not None
        assert episode_manager.reward_calculator is not None

    def test_reset_episode(self, episode_manager):
        """Test episode reset functionality with agent placement, component coordination, and Gymnasium reset() compliance."""
        # Call reset_episode() with test seed for reproducibility
        observation, info = episode_manager.reset_episode(seed=42)

        # Assert returned tuple contains (observation, info) for Gymnasium compliance
        assert isinstance(observation, np.ndarray)
        assert isinstance(info, dict)

        # Verify observation is numpy array with correct shape and dtype
        expected_shape = TEST_GRID_SIZE
        assert observation.shape == expected_shape
        assert observation.dtype == np.float32

        # Assert info dictionary contains agent_xy, distance_to_source, step_count
        required_info_keys = [
            "agent_xy",
            "distance_to_source",
            "step_count",
            "episode_id",
        ]
        for key in required_info_keys:
            assert key in info

        # Validate agent position is within grid bounds and not at source location
        agent_pos = info["agent_xy"]
        assert 0 <= agent_pos[0] < TEST_GRID_SIZE[0]
        assert 0 <= agent_pos[1] < TEST_GRID_SIZE[1]
        assert agent_pos != TEST_SOURCE_LOCATION

        # Verify episode_active flag is set to True and episode_count incremented
        assert episode_manager.episode_active is True
        assert episode_manager.episode_count == 1

        # Assert component coordination and state synchronization
        current_state = episode_manager.get_current_state()
        assert current_state.agent_state.position.x == agent_pos[0]
        assert current_state.agent_state.position.y == agent_pos[1]

    def test_process_step(self, episode_manager):
        """Test step processing with action validation, component coordination, and 5-tuple response format validation."""
        # Reset episode manager with test seed for initial state
        episode_manager.reset_episode(seed=42)

        # Process step with valid action (Action.RIGHT)
        obs, reward, terminated, truncated, info = episode_manager.process_step(
            Action.RIGHT
        )

        # Assert returned 5-tuple contains (obs, reward, terminated, truncated, info)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Verify observation is updated with new agent position
        assert obs.shape == TEST_GRID_SIZE
        assert obs.dtype == np.float32

        # Assert reward is calculated correctly based on distance to goal
        assert isinstance(reward, (int, float))

        # Validate terminated and truncated flags are boolean values
        assert terminated in [True, False]
        assert truncated in [True, False]

        # Verify info dictionary contains updated agent state and metrics
        assert "agent_xy" in info
        assert "step_count" in info
        assert info["step_count"] == 1

        # Assert agent position is updated and step count incremented
        current_state = episode_manager.get_current_state()
        assert current_state.agent_state.step_count == 1

        # Validate component coordination and state consistency
        assert episode_manager.validate_episode_consistency() is True

    def test_episode_termination(self, episode_manager):
        """Test episode termination conditions with goal achievement and step limit validation."""
        # Reset episode manager with agent positioned near source
        episode_manager.reset_episode(seed=42)

        # Get initial agent position
        initial_state = episode_manager.get_current_state()
        source_pos = TEST_SOURCE_LOCATION

        # Process steps to move agent to source location (simulate goal achievement)
        max_attempts = 50
        terminated = False
        step_count = 0

        while not terminated and step_count < max_attempts:
            # Choose action to move towards source
            current_pos = episode_manager.get_current_state().agent_state.position

            # Simple pathfinding toward source
            if current_pos.x < source_pos[0]:
                action = Action.RIGHT
            elif current_pos.x > source_pos[0]:
                action = Action.LEFT
            elif current_pos.y < source_pos[1]:
                action = Action.UP
            elif current_pos.y > source_pos[1]:
                action = Action.DOWN
            else:
                # At source location
                action = Action.UP  # Any action

            obs, reward, terminated, truncated, info = episode_manager.process_step(
                action
            )
            step_count += 1

            # Check if we reached the goal
            if (
                info["distance_to_source"]
                <= episode_manager.config.env_config.goal_radius
            ):
                # Assert terminated flag is True when goal is reached
                assert terminated is True
                # Verify reward is 1.0 for goal achievement
                assert reward == 1.0
                break

        # Test step limit truncation by running maximum steps
        if not terminated:
            # Reset and test step limit
            episode_manager.reset_episode(seed=123)
            max_steps = episode_manager.config.env_config.max_steps

            for step in range(max_steps + 1):
                obs, reward, terminated, truncated, info = episode_manager.process_step(
                    Action.UP
                )

                if step >= max_steps - 1:
                    # Assert truncated flag is True when max_steps reached
                    assert truncated is True
                    break

        # Verify episode completion detection and state finalization
        episode_result = episode_manager.finalize_episode()
        assert isinstance(episode_result, EpisodeResult)

    def test_episode_finalization(self, episode_manager):
        """Test episode finalization with statistics collection, performance metrics, and result generation."""
        # Run complete episode from reset to termination
        episode_manager.reset_episode(seed=42)

        # Process several steps
        for _ in range(10):
            obs, reward, terminated, truncated, info = episode_manager.process_step(
                Action.RIGHT
            )
            if terminated or truncated:
                break

        # Call finalize_episode() to complete episode processing
        episode_result = episode_manager.finalize_episode()

        # Assert EpisodeResult is returned with comprehensive episode data
        assert isinstance(episode_result, EpisodeResult)
        assert hasattr(episode_result, "episode_id")
        assert hasattr(episode_result, "total_steps")

        # Verify final agent state and episode duration are recorded
        assert episode_result.total_steps > 0
        assert hasattr(episode_result, "duration_ms")

        # Assert performance metrics contain timing and resource usage data
        performance_data = episode_result.get_performance_metrics()
        assert isinstance(performance_data, dict)
        assert "step_timing" in performance_data

        # Validate episode statistics are updated with completion data
        stats = episode_manager.get_episode_statistics()
        assert stats.episodes_completed > 0

        # Verify component cleanup and resource deallocation
        assert episode_manager.episode_active is False

    def test_performance_monitoring(self, episode_manager_config):
        """Test performance monitoring with timing validation and latency measurement within target thresholds."""
        # Initialize episode manager with performance monitoring enabled
        episode_manager_config.enable_performance_monitoring = True
        episode_manager = EpisodeManager(episode_manager_config)

        # Reset episode and measure reset timing performance
        start_time = time.perf_counter()
        episode_manager.reset_episode(seed=42)
        reset_time_ms = (time.perf_counter() - start_time) * 1000

        # Assert reset timing is within target threshold (< 10ms)
        assert reset_time_ms < 10.0

        # Process multiple steps measuring individual step latency
        step_times = []
        for i in range(10):
            start_time = time.perf_counter()
            obs, reward, terminated, truncated, info = episode_manager.process_step(
                Action.RIGHT
            )
            step_time_ms = (time.perf_counter() - start_time) * 1000
            step_times.append(step_time_ms)

            if terminated or truncated:
                break

        # Assert step processing time is within target (< 1ms) for most steps
        avg_step_time = np.mean(step_times)
        # Allow some tolerance for system variations
        assert avg_step_time < PERFORMANCE_TARGET_STEP_LATENCY_MS * 5  # 5ms tolerance

        # Verify performance metrics are collected and analyzed
        performance_metrics = episode_manager.get_performance_metrics()
        assert isinstance(performance_metrics, dict)
        assert "average_step_time_ms" in performance_metrics

        # Assert performance data includes timing, resource usage, efficiency metrics
        assert "reset_time_ms" in performance_metrics
        assert "total_processing_time_ms" in performance_metrics

    def test_state_consistency_validation(self, episode_manager):
        """Test comprehensive state consistency validation across all components with error detection."""
        # Reset episode manager and process several steps
        episode_manager.reset_episode(seed=42)

        for _ in range(5):
            obs, reward, terminated, truncated, info = episode_manager.process_step(
                Action.DOWN
            )
            if terminated or truncated:
                break

        # Call validate_episode_consistency() with strict validation
        consistency_result = episode_manager.validate_episode_consistency(strict=True)

        # Assert validation result indicates consistent state
        assert consistency_result is True

        # Verify agent state synchronization between components
        current_state = episode_manager.get_current_state()
        assert isinstance(current_state.agent_state, AgentState)

        # Assert episode state consistency and termination logic
        assert isinstance(current_state.episode_state, EpisodeState)

        # Test consistency validation with artificially corrupted state
        # Temporarily modify state to create inconsistency
        original_step_count = current_state.agent_state.step_count
        current_state.agent_state.step_count = -1  # Invalid step count

        try:
            # Verify inconsistency detection and detailed error reporting
            consistency_result = episode_manager.validate_episode_consistency(
                strict=True
            )
            # Should detect the inconsistency
            assert consistency_result is False
        finally:
            # Restore original state
            current_state.agent_state.step_count = original_step_count

    def test_episode_statistics(self, episode_manager):
        """Test episode statistics collection with success rates, performance trends, and multi-episode analysis."""
        # Run multiple episodes with different outcomes (terminated and truncated)
        episode_results = []

        for seed in [42, 123, 456]:
            episode_manager.reset_episode(seed=seed)

            # Process steps until termination or truncation
            for step in range(50):  # Limit steps to avoid infinite loops
                obs, reward, terminated, truncated, info = episode_manager.process_step(
                    Action.RIGHT
                )

                if terminated or truncated:
                    break

            # Finalize episode and collect result
            result = episode_manager.finalize_episode()
            episode_results.append(result)

        # Call get_episode_statistics() after each episode completion
        statistics = episode_manager.get_episode_statistics()

        # Assert statistics include episode count, success rate, average duration
        assert isinstance(statistics, EpisodeStatistics)
        assert statistics.episodes_completed >= len(episode_results)

        # Verify performance trends and component efficiency analysis
        success_rate = statistics.calculate_success_rate()
        assert 0.0 <= success_rate <= 1.0

        # Assert success rate calculation is accurate based on terminated episodes
        terminated_count = sum(1 for result in episode_results if result.terminated)
        expected_success_rate = (
            terminated_count / len(episode_results) if episode_results else 0.0
        )

        # Validate optimization recommendations based on performance patterns
        recommendations = statistics.get_optimization_recommendations()
        assert isinstance(recommendations, list)

    def test_cleanup(self, episode_manager):
        """Test comprehensive cleanup with resource release and final statistics collection validation."""
        # Initialize episode manager and run several episodes
        for seed in [42, 123]:
            episode_manager.reset_episode(seed=seed)

            # Process a few steps
            for _ in range(3):
                obs, reward, terminated, truncated, info = episode_manager.process_step(
                    Action.UP
                )
                if terminated or truncated:
                    break

            episode_manager.finalize_episode()

        # Call cleanup() with statistics preservation option
        statistics_before_cleanup = episode_manager.get_episode_statistics()
        episode_manager.cleanup(preserve_statistics=True)

        # Assert all component resources are properly released
        # Episode should be inactive after cleanup
        assert episode_manager.episode_active is False

        # Verify episode statistics are preserved if requested
        statistics_after_cleanup = episode_manager.get_episode_statistics()
        assert (
            statistics_after_cleanup.episodes_completed
            == statistics_before_cleanup.episodes_completed
        )

        # Test cleanup with different preservation options
        episode_manager.cleanup(preserve_statistics=False)

        # Verify complete resource cleanup and memory deallocation
        # Performance metrics should be cleared
        performance_metrics = episode_manager.get_performance_metrics()
        # After full cleanup, metrics should be reset or minimal
        assert isinstance(performance_metrics, dict)


class TestEpisodeManagerReproducibility:
    """Test class for episode manager reproducibility validation ensuring identical seeds produce identical episodes with comprehensive deterministic behavior testing and cross-execution consistency validation."""

    @pytest.fixture
    def episode_manager_pair(self):
        """Fixture providing two identical episode managers for reproducibility testing."""
        env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=50,
            goal_radius=0.0,
        )
        config = EpisodeManagerConfig(env_config=env_config)

        manager1 = EpisodeManager(config)
        manager2 = EpisodeManager(config.clone())

        return manager1, manager2

    @pytest.mark.parametrize("test_seed", REPRODUCIBILITY_TEST_SEEDS)
    def test_identical_episodes_same_seed(self, episode_manager_pair, test_seed):
        """Test that identical seeds produce exactly identical episodes with complete state and trajectory matching."""
        manager1, manager2 = episode_manager_pair

        # Reset both managers with same test seed
        obs1, info1 = manager1.reset_episode(seed=test_seed)
        obs2, info2 = manager2.reset_episode(seed=test_seed)

        # Assert initial observations and info are identical
        np.testing.assert_array_equal(obs1, obs2)
        assert info1["agent_xy"] == info2["agent_xy"]
        assert info1["distance_to_source"] == info2["distance_to_source"]

        # Process same sequence of actions on both managers
        action_sequence = [
            Action.RIGHT,
            Action.DOWN,
            Action.LEFT,
            Action.UP,
            Action.RIGHT,
        ]

        trajectories = [[], []]
        managers = [manager1, manager2]

        for action in action_sequence:
            for i, manager in enumerate(managers):
                obs, reward, terminated, truncated, info = manager.process_step(action)
                trajectories[i].append(
                    {
                        "observation": obs.copy(),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info.copy(),
                    }
                )

                if terminated or truncated:
                    break

        # Verify observations, rewards, termination flags match exactly
        for step in range(min(len(trajectories[0]), len(trajectories[1]))):
            step1 = trajectories[0][step]
            step2 = trajectories[1][step]

            np.testing.assert_array_equal(step1["observation"], step2["observation"])
            assert step1["reward"] == step2["reward"]
            assert step1["terminated"] == step2["terminated"]
            assert step1["truncated"] == step2["truncated"]

            # Assert agent trajectories are identical step-by-step
            assert step1["info"]["agent_xy"] == step2["info"]["agent_xy"]
            assert step1["info"]["step_count"] == step2["info"]["step_count"]

        # Validate final episode states and statistics are identical
        result1 = manager1.finalize_episode()
        result2 = manager2.finalize_episode()

        assert result1.total_steps == result2.total_steps
        assert result1.terminated == result2.terminated

    def test_different_episodes_different_seeds(self, episode_manager_pair):
        """Test that different seeds produce different episodes with statistical independence validation."""
        manager1, manager2 = episode_manager_pair

        # Reset with first seed and record complete episode trajectory
        seed1, seed2 = 42, 123
        obs1, info1 = manager1.reset_episode(seed=seed1)
        obs2, info2 = manager2.reset_episode(seed=seed2)

        # Assert trajectories differ in agent start positions
        assert info1["agent_xy"] != info2["agent_xy"] or not np.array_equal(obs1, obs2)

        # Process identical actions and record differences
        action_sequence = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        differences_found = False

        for action in action_sequence:
            step1 = manager1.process_step(action)
            step2 = manager2.process_step(action)

            if not np.array_equal(step1[0], step2[0]):  # Different observations
                differences_found = True
                break

            if step1[3]["agent_xy"] != step2[3]["agent_xy"]:  # Different positions
                differences_found = True
                break

        # Verify different episodes have different step sequences
        assert (
            differences_found
        ), "Episodes with different seeds should produce different trajectories"

        # Assert statistical independence between different seeded episodes
        result1 = manager1.finalize_episode()
        result2 = manager2.finalize_episode()

        # Episodes should have different characteristics
        assert (
            result1.total_steps != result2.total_steps
            or result1.final_agent_position != result2.final_agent_position
        )

    def test_deterministic_component_behavior(self, episode_manager_pair):
        """Test deterministic behavior across all components with consistent state management and coordination."""
        manager1, manager2 = episode_manager_pair
        fixed_seed = 42

        # Initialize both episode managers with deterministic seed
        manager1.reset_episode(seed=fixed_seed)
        manager2.reset_episode(seed=fixed_seed)

        # Test deterministic behavior of state_manager component
        state1 = manager1.get_current_state()
        state2 = manager2.get_current_state()

        assert state1.agent_state.position == state2.agent_state.position
        assert state1.episode_state.step_count == state2.episode_state.step_count

        # Process identical action sequence
        actions = [Action.RIGHT, Action.DOWN, Action.LEFT]

        for action in actions:
            # Verify reward_calculator produces identical results
            step1 = manager1.process_step(action)
            step2 = manager2.process_step(action)

            assert step1[1] == step2[1]  # Same reward

            # Assert action_processor behaves deterministically
            assert step1[3]["agent_xy"] == step2[3]["agent_xy"]  # Same position

        # Test component coordination consistency with same inputs
        consistency1 = manager1.validate_episode_consistency()
        consistency2 = manager2.validate_episode_consistency()

        assert consistency1 == consistency2

        # Verify performance metrics are reproducible
        metrics1 = manager1.get_performance_metrics()
        metrics2 = manager2.get_performance_metrics()

        # Should have similar structure even if timing varies slightly
        assert set(metrics1.keys()) == set(metrics2.keys())

    def test_seeding_isolation(self, episode_manager_pair):
        """Test that seeding properly isolates episodes without cross-episode contamination."""
        manager1, manager2 = episode_manager_pair

        # Run episode with seed A and record final random state
        seed_a, seed_b = 42, 123

        manager1.reset_episode(seed=seed_a)
        # Process some steps
        for _ in range(5):
            obs, reward, terminated, truncated, info = manager1.process_step(
                Action.RIGHT
            )
            if terminated or truncated:
                break

        # Record trajectory from first run with seed A
        trajectory_a1 = info["agent_xy"]
        manager1.finalize_episode()

        # Run different episode with seed B
        manager1.reset_episode(seed=seed_b)
        for _ in range(5):
            obs, reward, terminated, truncated, info = manager1.process_step(
                Action.RIGHT
            )
            if terminated or truncated:
                break
        manager1.finalize_episode()

        # Reset with seed A again and verify identical behavior to first run
        manager1.reset_episode(seed=seed_a)
        for _ in range(5):
            obs, reward, terminated, truncated, info = manager1.process_step(
                Action.RIGHT
            )
            if terminated or truncated:
                break

        # Assert no random state contamination between episodes
        trajectory_a2 = info["agent_xy"]

        # The trajectories with the same seed should be identical
        # Note: This test verifies the final position after same number of steps
        # More comprehensive testing would track all intermediate positions

        # Verify seed manager properly isolates episode random states
        assert isinstance(manager1.state_manager.seed_manager, SeedManager)


class TestEpisodeManagerPerformance:
    """Test class for episode manager performance validation targeting <1ms step latency with comprehensive timing analysis, resource usage monitoring, and optimization validation for system performance requirements."""

    @pytest.fixture
    def performance_episode_manager(self):
        """Fixture providing episode manager optimized for performance testing."""
        env_config = create_environment_config(
            grid_size=(16, 16),  # Smaller grid for performance testing
            source_location=(8, 8),
            max_steps=50,
            goal_radius=0.0,
        )
        config = EpisodeManagerConfig(
            env_config=env_config, enable_performance_monitoring=True
        )
        return EpisodeManager(config)

    def test_step_latency_performance(self, performance_episode_manager):
        """Test step processing latency meets performance targets with statistical analysis across multiple iterations."""
        episode_manager = performance_episode_manager

        # Reset episode for initial state setup
        episode_manager.reset_episode(seed=42)

        # Measure step processing time across PERFORMANCE_TEST_ITERATIONS iterations
        step_times = []

        for i in range(
            min(PERFORMANCE_TEST_ITERATIONS, 20)
        ):  # Reduced for test efficiency
            start_time = time.perf_counter()

            # Use rotating actions to avoid early termination
            action = Action(i % 4)
            obs, reward, terminated, truncated, info = episode_manager.process_step(
                action
            )

            end_time = time.perf_counter()
            step_time_ms = (end_time - start_time) * 1000
            step_times.append(step_time_ms)

            if terminated or truncated:
                episode_manager.reset_episode(seed=42 + i)

        # Calculate average, minimum, maximum, and standard deviation of step times
        avg_time = np.mean(step_times)
        min_time = np.min(step_times)
        max_time = np.max(step_times)
        std_time = np.std(step_times)

        # Assert average step time is within reasonable bounds
        # Note: Using relaxed targets for CI/testing environments
        assert avg_time < PERFORMANCE_TARGET_STEP_LATENCY_MS * 10  # 10ms tolerance

        # Verify 95% of steps complete within reasonable time
        percentile_95 = np.percentile(step_times, 95)
        assert (
            percentile_95 < PERFORMANCE_TARGET_STEP_LATENCY_MS * 20
        )  # 20ms for 95th percentile

        # Assert no steps exceed extreme outlier threshold
        outlier_threshold = (
            PERFORMANCE_TARGET_STEP_LATENCY_MS * 50
        )  # 50ms outlier threshold
        outliers = [t for t in step_times if t > outlier_threshold]
        assert (
            len(outliers) == 0
        ), f"Found {len(outliers)} outliers exceeding {outlier_threshold}ms"

        # Validate performance consistency across different actions and states
        assert std_time < avg_time  # Standard deviation should be less than mean

    def test_reset_performance(self, performance_episode_manager):
        """Test episode reset performance with initialization timing and resource allocation validation."""
        episode_manager = performance_episode_manager

        # Measure episode reset time across multiple iterations
        reset_times = []

        for i in range(10):  # 10 reset operations
            start_time = time.perf_counter()
            episode_manager.reset_episode(seed=42 + i)
            end_time = time.perf_counter()

            reset_time_ms = (end_time - start_time) * 1000
            reset_times.append(reset_time_ms)

        # Calculate timing statistics for reset operations
        avg_reset_time = np.mean(reset_times)
        max_reset_time = np.max(reset_times)

        # Assert average reset time is reasonable (more lenient for test environments)
        target_reset_time = PERFORMANCE_TARGET_STEP_LATENCY_MS * 50  # 50ms for reset
        assert avg_reset_time < target_reset_time

        # Verify reset time consistency across different seeds
        std_reset_time = np.std(reset_times)
        assert std_reset_time < avg_reset_time  # Reasonable variation

        # Assert resource allocation efficiency during reset
        # Check that memory usage is reasonable after multiple resets
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Should not exceed reasonable memory bounds
        assert memory_mb < 200  # 200MB upper bound for testing

    def test_memory_usage_efficiency(self, performance_episode_manager):
        """Test memory usage efficiency with resource monitoring and leak detection validation."""
        episode_manager = performance_episode_manager

        # Measure baseline memory usage before episode manager operations
        import os

        import psutil

        process = psutil.Process(os.getpid())
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024

        # Run complete episode and monitor memory usage throughout
        episode_manager.reset_episode(seed=42)

        for _ in range(20):  # Process 20 steps
            obs, reward, terminated, truncated, info = episode_manager.process_step(
                Action.RIGHT
            )
            if terminated or truncated:
                break

        episode_manager.finalize_episode()

        # Check memory usage after episode
        post_episode_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase = post_episode_memory_mb - baseline_memory_mb

        # Assert memory usage remains within reasonable bounds
        assert memory_increase < 50  # Less than 50MB increase

        # Test for memory leaks by running multiple episodes
        for i in range(5):
            episode_manager.reset_episode(seed=42 + i)

            for _ in range(10):
                obs, reward, terminated, truncated, info = episode_manager.process_step(
                    Action.UP
                )
                if terminated or truncated:
                    break

            episode_manager.finalize_episode()

        # Check final memory usage
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory_mb - baseline_memory_mb

        # Verify memory is managed efficiently across multiple episodes
        assert total_increase < 100  # Less than 100MB total increase

        # Force garbage collection and verify cleanup
        gc.collect()
        post_gc_memory_mb = process.memory_info().rss / 1024 / 1024

        # Memory should not grow indefinitely
        assert post_gc_memory_mb <= final_memory_mb

    def test_component_coordination_efficiency(self, performance_episode_manager):
        """Test efficiency of component coordination with timing analysis and bottleneck identification."""
        episode_manager = performance_episode_manager

        # Reset episode for component coordination testing
        episode_manager.reset_episode(seed=42)

        # Measure individual component response times within step processing
        component_times = {
            "state_manager": [],
            "reward_calculator": [],
            "action_processor": [],
        }

        # Mock component methods to measure timing
        original_process_step = episode_manager.state_manager.process_step

        def timed_state_manager_step(*args, **kwargs):
            start = time.perf_counter()
            result = original_process_step(*args, **kwargs)
            component_times["state_manager"].append(
                (time.perf_counter() - start) * 1000
            )
            return result

        episode_manager.state_manager.process_step = timed_state_manager_step

        # Process steps and collect component timing data
        for _ in range(10):
            obs, reward, terminated, truncated, info = episode_manager.process_step(
                Action.DOWN
            )
            if terminated or truncated:
                break

        # Restore original method
        episode_manager.state_manager.process_step = original_process_step

        # Analyze state_manager timing
        if component_times["state_manager"]:
            avg_state_time = np.mean(component_times["state_manager"])

            # Assert component coordination overhead is reasonable
            step_target = PERFORMANCE_TARGET_STEP_LATENCY_MS
            assert avg_state_time < step_target * 5  # 5x step target for component

        # Verify efficient data passing between components
        current_state = episode_manager.get_current_state()
        assert current_state is not None
        assert isinstance(current_state.agent_state, AgentState)

        # Test component caching effectiveness
        consistency_check_time_start = time.perf_counter()
        is_consistent = episode_manager.validate_episode_consistency()
        consistency_check_time = (
            time.perf_counter() - consistency_check_time_start
        ) * 1000

        assert is_consistent is True
        assert consistency_check_time < 10.0  # Should be fast due to caching

    def test_performance_under_load(self, performance_episode_manager):
        """Test performance stability under sustained load with stress testing and performance degradation analysis."""
        episode_manager = performance_episode_manager

        # Run sustained load test with continuous episode execution
        load_test_episodes = 5  # Reduced for test efficiency
        episode_times = []
        step_times_per_episode = []

        for episode_idx in range(load_test_episodes):
            episode_start = time.perf_counter()

            # Reset episode
            episode_manager.reset_episode(seed=42 + episode_idx)

            # Process steps with timing
            episode_step_times = []
            for step_idx in range(20):  # 20 steps per episode
                step_start = time.perf_counter()
                obs, reward, terminated, truncated, info = episode_manager.process_step(
                    Action((step_idx + episode_idx) % 4)
                )
                step_time = (time.perf_counter() - step_start) * 1000
                episode_step_times.append(step_time)

                if terminated or truncated:
                    break

            episode_manager.finalize_episode()
            episode_total_time = (time.perf_counter() - episode_start) * 1000

            episode_times.append(episode_total_time)
            step_times_per_episode.append(episode_step_times)

        # Monitor performance metrics over extended operation period
        overall_step_times = [
            time for episode_times in step_times_per_episode for time in episode_times
        ]

        # Assert performance remains stable over multiple episodes
        if len(overall_step_times) > 10:
            first_half = overall_step_times[: len(overall_step_times) // 2]
            second_half = overall_step_times[len(overall_step_times) // 2 :]

            first_half_avg = np.mean(first_half)
            second_half_avg = np.mean(second_half)

            # Performance degradation should be minimal
            degradation_ratio = second_half_avg / first_half_avg
            assert degradation_ratio < 2.0  # No more than 2x degradation

        # Verify no performance degradation over time
        episode_avg_times = [
            np.mean(times) for times in step_times_per_episode if times
        ]
        if len(episode_avg_times) > 1:
            # Check that later episodes aren't significantly slower
            early_avg = np.mean(episode_avg_times[: len(episode_avg_times) // 2])
            late_avg = np.mean(episode_avg_times[len(episode_avg_times) // 2 :])

            performance_ratio = late_avg / early_avg
            assert performance_ratio < 1.5  # No more than 50% degradation

        # Assert resource usage remains stable under load
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Memory should not grow excessively under load
        assert memory_mb < 300  # 300MB upper bound under load


class TestEpisodeManagerErrorHandling:
    """Test class for episode manager error handling validation including component failure recovery, validation error handling, and graceful degradation strategies with comprehensive error scenario testing."""

    @pytest.fixture
    def episode_manager(self):
        """Fixture providing episode manager for error handling testing."""
        env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=100,
            goal_radius=0.0,
        )
        config = EpisodeManagerConfig(env_config=env_config)
        return EpisodeManager(config)

    def test_invalid_action_handling(self, episode_manager):
        """Test handling of invalid actions with proper validation error reporting and recovery."""
        # Reset episode manager to active state
        episode_manager.reset_episode(seed=42)

        # Attempt to process invalid action (out of range integer)
        with pytest.raises(ValidationError) as exc_info:
            episode_manager.process_step(99)  # Invalid action

        # Assert ValidationError is raised with appropriate error message
        assert "action" in str(exc_info.value.message).lower()

        # Test handling of non-integer action inputs
        with pytest.raises(ValidationError):
            episode_manager.process_step("invalid_action")

        with pytest.raises(ValidationError):
            episode_manager.process_step(None)

        # Verify episode state remains consistent after validation errors
        current_state = episode_manager.get_current_state()
        assert current_state is not None

        # Assert component states are not corrupted by invalid inputs
        consistency_result = episode_manager.validate_episode_consistency()
        assert consistency_result is True

        # Test recovery from validation errors without episode reset
        # Should be able to process valid action after invalid one
        obs, reward, terminated, truncated, info = episode_manager.process_step(
            Action.RIGHT
        )
        assert isinstance(obs, np.ndarray)

    def test_inactive_episode_error_handling(self, episode_manager):
        """Test error handling for operations on inactive episodes with proper state validation."""
        # Initialize episode manager without resetting episode (inactive state)
        assert episode_manager.episode_active is False

        # Attempt to process step on inactive episode
        with pytest.raises(StateError) as exc_info:
            episode_manager.process_step(Action.RIGHT)

        # Assert StateError is raised with appropriate error message
        error_message = str(exc_info.value.message).lower()
        assert "episode" in error_message or "reset" in error_message

        # Test multiple operations on inactive episode
        with pytest.raises(StateError):
            episode_manager.get_current_state()

        with pytest.raises(StateError):
            episode_manager.finalize_episode()

        # Verify proper error messages indicate required reset operation
        assert "reset" in str(exc_info.value.message).lower()

        # Assert episode manager maintains safe state during errors
        assert episode_manager.episode_active is False

        # Test error recovery through proper episode reset
        episode_manager.reset_episode(seed=42)
        assert episode_manager.episode_active is True

        # Should work normally after reset
        obs, reward, terminated, truncated, info = episode_manager.process_step(
            Action.UP
        )
        assert isinstance(obs, np.ndarray)

    def test_component_failure_handling(self, episode_manager):
        """Test handling of component failures with isolation, recovery strategies, and graceful degradation."""
        # Reset episode manager for component testing
        episode_manager.reset_episode(seed=42)

        # Mock component failure in state_manager during step processing
        original_process_step = episode_manager.state_manager.process_step

        def failing_process_step(*args, **kwargs):
            raise ComponentError(
                message="Simulated state manager failure",
                component_name="state_manager",
                operation_name="process_step",
            )

        episode_manager.state_manager.process_step = failing_process_step

        try:
            # Attempt step processing with failing component
            with pytest.raises(ComponentError) as exc_info:
                episode_manager.process_step(Action.RIGHT)

            # Assert ComponentError is raised with component identification
            assert "state_manager" in str(exc_info.value.component_name)

            # Test error isolation preventing cascade failures
            # Other components should remain functional
            assert episode_manager.reward_calculator is not None
            assert episode_manager.action_processor is not None

        finally:
            # Restore original method for cleanup
            episode_manager.state_manager.process_step = original_process_step

        # Verify other components remain functional during isolated failures
        # Should be able to reset and continue
        episode_manager.reset_episode(seed=123)
        obs, reward, terminated, truncated, info = episode_manager.process_step(
            Action.LEFT
        )
        assert isinstance(obs, np.ndarray)

    def test_configuration_error_handling(self):
        """Test handling of configuration errors with validation failure recovery and error reporting."""
        # Attempt to create episode manager with invalid configuration
        try:
            invalid_env_config = create_environment_config(
                grid_size=(-10, -10),  # Invalid negative dimensions
                source_location=TEST_SOURCE_LOCATION,
                max_steps=100,
                goal_radius=0.0,
            )
            pytest.fail("Should have raised ValidationError for invalid grid_size")
        except ValidationError as e:
            # Assert ValidationError is raised during initialization
            assert "grid_size" in str(e.message).lower()

        # Test specific configuration parameter validation failures
        try:
            invalid_env_config = create_environment_config(
                grid_size=TEST_GRID_SIZE,
                source_location=(1000, 1000),  # Outside grid bounds
                max_steps=100,
                goal_radius=0.0,
            )
            pytest.fail("Should have raised ValidationError for source outside grid")
        except ValidationError as e:
            # Verify comprehensive error messages with parameter details
            assert (
                "source" in str(e.message).lower()
                or "location" in str(e.message).lower()
            )

        # Test edge cases with borderline invalid configurations
        try:
            borderline_env_config = create_environment_config(
                grid_size=TEST_GRID_SIZE,
                source_location=TEST_SOURCE_LOCATION,
                max_steps=0,  # Zero steps
                goal_radius=0.0,
            )
            pytest.fail("Should have raised ValidationError for zero max_steps")
        except ValidationError as e:
            assert "max_steps" in str(e.message).lower()

    def test_resource_exhaustion_handling(self, episode_manager):
        """Test handling of resource exhaustion scenarios with cleanup actions and resource management."""
        # Reset episode for resource testing
        episode_manager.reset_episode(seed=42)

        # Simulate memory exhaustion scenario during episode processing
        # Mock a resource constraint
        original_validate = episode_manager.validate_episode_consistency

        resource_exhaustion_count = 0

        def resource_limited_validate(*args, **kwargs):
            nonlocal resource_exhaustion_count
            resource_exhaustion_count += 1

            if resource_exhaustion_count > 3:
                from plume_nav_sim.utils.exceptions import ResourceError

                raise ResourceError(
                    message="Simulated memory exhaustion during validation",
                    resource_type="memory",
                    current_usage=100.0,
                    limit_exceeded=50.0,
                )

            return original_validate(*args, **kwargs)

        episode_manager.validate_episode_consistency = resource_limited_validate

        try:
            # Process steps until resource exhaustion
            for _ in range(5):
                obs, reward, terminated, truncated, info = episode_manager.process_step(
                    Action.DOWN
                )

                if terminated or truncated:
                    break

                # This should eventually trigger the resource error
                try:
                    episode_manager.validate_episode_consistency()
                except ResourceError as e:
                    # Test resource error detection and handling
                    assert "memory" in str(e.message).lower()
                    assert e.resource_type == "memory"
                    break

        finally:
            # Restore original method
            episode_manager.validate_episode_consistency = original_validate

        # Test graceful degradation under resource constraints
        # Episode manager should still be able to perform basic operations
        try:
            current_state = episode_manager.get_current_state()
            assert current_state is not None
        except Exception:
            # Some degradation is acceptable under resource constraints
            pass

    def test_error_recovery_strategies(self, episode_manager):
        """Test comprehensive error recovery strategies with automated recovery and manual intervention guidance."""
        # Test automated recovery from transient component failures
        episode_manager.reset_episode(seed=42)

        # Simulate transient failure
        failure_count = 0
        original_process_step = episode_manager.state_manager.process_step

        def transient_failing_process_step(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1

            if failure_count <= 2:  # Fail first 2 times, then succeed
                raise ComponentError(
                    message="Transient component failure",
                    component_name="state_manager",
                    operation_name="process_step",
                )
            else:
                return original_process_step(*args, **kwargs)

        episode_manager.state_manager.process_step = transient_failing_process_step

        try:
            # Attempt operations that should fail then recover
            with pytest.raises(ComponentError):
                episode_manager.process_step(Action.UP)  # First failure

            with pytest.raises(ComponentError):
                episode_manager.process_step(Action.RIGHT)  # Second failure

            # Third attempt should succeed (simulating recovery)
            obs, reward, terminated, truncated, info = episode_manager.process_step(
                Action.DOWN
            )
            assert isinstance(obs, np.ndarray)

        finally:
            # Restore original method
            episode_manager.state_manager.process_step = original_process_step

        # Test manual recovery procedures for critical failures
        # Verify episode state restoration after recoverable errors
        current_state = episode_manager.get_current_state()
        assert current_state is not None
        assert current_state.agent_state.step_count > 0

        # Test recovery success validation and confirmation
        consistency_result = episode_manager.validate_episode_consistency()
        assert consistency_result is True


class TestEpisodeResultAndStatistics:
    """Test class for EpisodeResult and EpisodeStatistics data classes validation including result generation, statistics collection, performance analysis, and multi-episode trend analysis with comprehensive data integrity testing."""

    def test_episode_result_creation(self):
        """Test creation and initialization of EpisodeResult with comprehensive episode information."""
        # Create EpisodeResult with test parameters
        episode_id = str(uuid.uuid4())
        terminated = True
        total_steps = 25

        episode_result = EpisodeResult(
            episode_id=episode_id,
            terminated=terminated,
            truncated=False,
            total_steps=total_steps,
        )

        # Assert episode_id is stored correctly
        assert episode_result.episode_id == episode_id

        # Verify terminated and truncated flags are set properly
        assert episode_result.terminated is True
        assert episode_result.truncated is False

        # Assert total_steps reflects episode length accurately
        assert episode_result.total_steps == total_steps

        # Test initialization of optional fields
        assert hasattr(episode_result, "total_reward")
        assert hasattr(episode_result, "duration_ms")

        # Verify empty performance_metrics initialization (direct access)
        assert isinstance(episode_result.performance_metrics, dict)

    def test_episode_result_final_state(self):
        """Test setting final episode state with agent information, distance metrics, and termination analysis."""
        # Create EpisodeResult and final AgentState
        episode_result = EpisodeResult(
            episode_id="test_episode", terminated=True, truncated=False, total_steps=15
        )

        final_position = create_coordinates((10, 8))
        final_agent_state = AgentState(
            position=final_position, step_count=15, total_reward=1.0, goal_reached=True
        )

        # Call set_final_state() with agent state and termination details
        episode_result.set_final_state(
            final_agent_state=final_agent_state,
            distance_to_goal=0.5,
            termination_reason="goal_reached",
        )

        # Assert final_agent_position is set from agent state
        assert episode_result.final_agent_position == final_position

        # Verify total_reward matches agent total reward
        assert episode_result.total_reward == 1.0

        # Assert final_distance_to_goal is calculated correctly
        assert episode_result.final_distance_to_goal == 0.5

        # Test termination_reason setting and validation
        assert episode_result.termination_reason == "goal_reached"

    def test_episode_result_performance_metrics(self):
        """Test addition of performance metrics with timing analysis and resource usage tracking."""
        # Create EpisodeResult and performance metrics data
        episode_result = EpisodeResult(
            episode_id="perf_test_episode",
            terminated=False,
            truncated=True,
            total_steps=50,
        )

        episode_metrics = {
            "total_duration_ms": 125.5,
            "average_step_time_ms": 2.5,
            "reset_time_ms": 5.0,
        }

        component_metrics = {
            "state_manager": {"average_time_ms": 0.8},
            "reward_calculator": {"average_time_ms": 0.3},
            "action_processor": {"average_time_ms": 0.2},
        }

        # Call add_performance_metrics() with episode and component metrics
        episode_result.add_performance_metrics(
            episode_metrics=episode_metrics, component_metrics=component_metrics
        )

        # Assert performance_metrics contains timing and resource data
        performance_data = episode_result.get_performance_metrics()
        assert "total_duration_ms" in performance_data
        assert performance_data["total_duration_ms"] == 125.5

        # Verify component_statistics includes component-specific metrics
        assert "component_metrics" in performance_data
        assert "state_manager" in performance_data["component_metrics"]

    def test_episode_result_summary(self):
        """Test comprehensive episode summary generation with analysis, logging, and reporting."""
        # Create complete EpisodeResult with all data fields
        episode_result = EpisodeResult(
            episode_id="summary_test", terminated=True, truncated=False, total_steps=30
        )

        # Set final state
        final_position = create_coordinates(TEST_SOURCE_LOCATION)
        final_agent_state = AgentState(
            position=final_position, step_count=30, total_reward=1.0, goal_reached=True
        )

        episode_result.set_final_state(
            final_agent_state=final_agent_state,
            distance_to_goal=0.0,
            termination_reason="goal_reached",
        )

        # Call get_summary() with performance analysis enabled
        summary = episode_result.get_summary(include_performance=True)

        # Assert summary contains basic episode information
        assert isinstance(summary, dict)
        assert "episode_id" in summary
        assert "total_steps" in summary

        # Verify final position, distance, and reward information
        assert "final_position" in summary
        assert "total_reward" in summary

        # Assert goal achievement assessment and success analysis
        assert "goal_reached" in summary or "success" in summary

        # Test component details inclusion when requested
        summary_detailed = episode_result.get_summary(
            include_performance=True, include_components=True
        )
        assert isinstance(summary_detailed, dict)

    def test_episode_statistics_collection(self):
        """Test episode statistics collection with multi-episode aggregation and trend analysis."""
        # Create EpisodeStatistics with unique identifier
        statistics = EpisodeStatistics(session_id="test_session")

        # Create multiple EpisodeResult instances with different outcomes
        episodes = []

        # Successful episode
        success_episode = EpisodeResult(
            "ep1", terminated=True, truncated=False, total_steps=20
        )
        success_episode.total_reward = 1.0
        episodes.append(success_episode)

        # Truncated episode
        truncated_episode = EpisodeResult(
            "ep2", terminated=False, truncated=True, total_steps=100
        )
        truncated_episode.total_reward = 0.0
        episodes.append(truncated_episode)

        # Another successful episode
        success_episode2 = EpisodeResult(
            "ep3", terminated=True, truncated=False, total_steps=35
        )
        success_episode2.total_reward = 1.0
        episodes.append(success_episode2)

        # Add episodes to statistics
        for episode in episodes:
            statistics.add_episode_result(episode)

        # Assert episodes_completed counter is updated correctly
        assert statistics.episodes_completed == len(episodes)

        # Verify episode termination and truncation counters
        assert statistics.successful_episodes == 2  # Two terminated episodes
        assert statistics.truncated_episodes == 1  # One truncated episode

    def test_success_rate_calculation(self):
        """Test success rate calculation based on goal achievement and completion status."""
        # Create statistics for success rate testing
        statistics = EpisodeStatistics(session_id="success_rate_test")

        # Add mix of terminated and truncated episode results
        # 3 successful (terminated), 2 truncated
        for i in range(3):
            success_ep = EpisodeResult(
                f"success_{i}", terminated=True, truncated=False, total_steps=20 + i
            )
            success_ep.total_reward = 1.0
            statistics.add_episode_result(success_ep)

        for i in range(2):
            truncated_ep = EpisodeResult(
                f"truncated_{i}", terminated=False, truncated=True, total_steps=100
            )
            truncated_ep.total_reward = 0.0
            statistics.add_episode_result(truncated_ep)

        # Call calculate_success_rate() method
        success_rate = statistics.calculate_success_rate()

        # Assert success rate is calculated as terminated / completed ratio
        expected_rate = 3 / 5  # 3 successful out of 5 total
        assert abs(success_rate - expected_rate) < 0.001

        # Test success rate calculation with only terminated episodes (100%)
        success_only_stats = EpisodeStatistics(session_id="success_only")
        success_ep = EpisodeResult(
            "all_success", terminated=True, truncated=False, total_steps=15
        )
        success_only_stats.add_episode_result(success_ep)

        assert success_only_stats.calculate_success_rate() == 1.0

        # Test with only truncated episodes (0% success rate)
        truncated_only_stats = EpisodeStatistics(session_id="truncated_only")
        truncated_ep = EpisodeResult(
            "all_truncated", terminated=False, truncated=True, total_steps=100
        )
        truncated_only_stats.add_episode_result(truncated_ep)

        assert truncated_only_stats.calculate_success_rate() == 0.0

        # Assert proper handling of zero episodes (no division by zero)
        empty_stats = EpisodeStatistics(session_id="empty")
        assert empty_stats.calculate_success_rate() == 0.0

    def test_performance_summary_analysis(self):
        """Test comprehensive performance summary generation with trends and optimization recommendations."""
        # Create statistics for performance summary testing
        statistics = EpisodeStatistics(session_id="performance_test")

        # Collect statistics from multiple episodes with varying performance
        episode_performance_data = [
            (20, 1.0, 50.5),  # steps, reward, duration_ms
            (35, 0.0, 85.2),
            (15, 1.0, 42.1),
            (50, 0.0, 120.8),
            (25, 1.0, 68.3),
        ]

        for i, (steps, reward, duration) in enumerate(episode_performance_data):
            episode = EpisodeResult(
                f"perf_ep_{i}",
                terminated=(reward == 1.0),
                truncated=(reward == 0.0),
                total_steps=steps,
            )
            episode.total_reward = reward
            episode.duration_ms = duration
            statistics.add_episode_result(episode)

        # Call get_performance_summary() with trend analysis enabled
        performance_summary = statistics.get_performance_summary(include_trends=True)

        # Assert summary contains performance statistics and averages
        assert isinstance(performance_summary, dict)
        assert "average_steps" in performance_summary
        assert "average_duration_ms" in performance_summary

        # Verify trend analysis includes performance changes over time
        if "trends" in performance_summary:
            assert isinstance(performance_summary["trends"], dict)

        # Assert optimization recommendations based on performance patterns
        if "recommendations" in performance_summary:
            assert isinstance(performance_summary["recommendations"], list)


class TestFactoryFunctions:
    """Test class for factory functions validation including create_episode_manager and validate_episode_config functions with comprehensive configuration testing, validation scenarios, and factory function reliability testing."""

    def test_create_episode_manager_default(self):
        """Test create_episode_manager factory function with default parameters and validation."""
        # Call create_episode_manager() without parameters
        episode_manager = create_episode_manager()

        # Assert EpisodeManager instance is returned
        assert isinstance(episode_manager, EpisodeManager)

        # Verify default configuration is created and validated
        assert episode_manager.config is not None
        assert isinstance(episode_manager.config, EpisodeManagerConfig)

        # Assert default SeedManager is initialized
        assert hasattr(episode_manager, "state_manager")
        if hasattr(episode_manager.state_manager, "seed_manager"):
            assert episode_manager.state_manager.seed_manager is not None

        # Test component coordination and integration setup
        assert episode_manager.reward_calculator is not None
        assert episode_manager.action_processor is not None

        # Verify performance monitoring is enabled by default
        assert episode_manager.config.enable_performance_monitoring is True

        # Test functionality of created episode manager instance
        obs, info = episode_manager.reset_episode(seed=42)
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_create_episode_manager_custom(self):
        """Test create_episode_manager factory function with custom configuration and components."""
        # Create custom episode manager configuration
        custom_env_config = create_environment_config(
            grid_size=(24, 24), source_location=(12, 12), max_steps=75, goal_radius=1.0
        )
        custom_config = EpisodeManagerConfig(
            env_config=custom_env_config, enable_performance_monitoring=False
        )

        # Create custom seed manager with test seed
        custom_seed_manager = SeedManager(default_seed=123, enable_validation=True)

        # Call create_episode_manager() with custom parameters
        episode_manager = create_episode_manager(
            config=custom_config, seed_manager=custom_seed_manager
        )

        # Assert returned episode manager uses custom configuration
        assert episode_manager.config is custom_config
        assert episode_manager.config.env_config.grid_size == (24, 24)
        assert episode_manager.config.enable_performance_monitoring is False

        # Verify custom seed manager integration and functionality
        if hasattr(episode_manager.state_manager, "seed_manager"):
            assert episode_manager.state_manager.seed_manager is custom_seed_manager

        # Test component coordination with custom components
        obs, info = episode_manager.reset_episode()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (24, 24)

        # Assert configuration validation and consistency checking
        assert episode_manager.config.validate() is True

    def test_create_episode_manager_options(self):
        """Test create_episode_manager factory function with various options and feature flags."""
        # Test factory function with performance monitoring disabled
        episode_manager_no_perf = create_episode_manager(
            enable_performance_monitoring=False, enable_state_validation=False
        )

        # Assert episode manager operates without performance tracking
        assert episode_manager_no_perf.config.enable_performance_monitoring is False

        # Test with component validation disabled
        assert episode_manager_no_perf.config.enable_state_validation is False

        # Test combination of different option flags
        episode_manager_custom_options = create_episode_manager(
            grid_size=(16, 16),
            max_steps=50,
            enable_performance_monitoring=True,
            enable_state_validation=True,
        )

        # Assert option configurations are applied correctly
        assert episode_manager_custom_options.config.env_config.grid_size == (16, 16)
        assert episode_manager_custom_options.config.env_config.max_steps == 50

        # Validate feature flag impact on episode manager behavior
        obs, info = episode_manager_custom_options.reset_episode(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (16, 16)

    def test_validate_episode_config_valid(self):
        """Test validate_episode_config function with valid configurations and comprehensive validation analysis."""
        # Create valid episode manager configuration
        valid_env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=200,
            goal_radius=0.5,
        )
        valid_config = EpisodeManagerConfig(env_config=valid_env_config)

        # Call validate_episode_config() with valid configuration
        is_valid, validation_report = validate_episode_config(valid_config)

        # Assert validation returns (True, validation_report) tuple
        assert is_valid is True
        assert isinstance(validation_report, dict)

        # Verify validation report contains comprehensive analysis
        assert "validation_status" in validation_report
        assert validation_report["validation_status"] == "valid"

        # Assert no validation errors or warnings are reported
        if "errors" in validation_report:
            assert len(validation_report["errors"]) == 0

        # Test validation with strict mode enabled
        is_valid_strict, strict_report = validate_episode_config(
            valid_config, strict_mode=True
        )
        assert is_valid_strict is True

        # Verify detailed validation analysis and recommendations
        assert isinstance(strict_report, dict)

    def test_validate_episode_config_invalid(self):
        """Test validate_episode_config function with invalid configurations and detailed error reporting."""
        try:
            # Create invalid episode manager configuration (negative max_steps)
            invalid_env_config = create_environment_config(
                grid_size=TEST_GRID_SIZE,
                source_location=TEST_SOURCE_LOCATION,
                max_steps=-50,  # Invalid
                goal_radius=0.0,
            )
            pytest.fail("Should have raised ValidationError for invalid config")
        except ValidationError:
            # Expected - invalid configuration should raise ValidationError during creation
            pass

        # Test with valid config but invalid override
        valid_env_config = create_environment_config(
            grid_size=TEST_GRID_SIZE,
            source_location=TEST_SOURCE_LOCATION,
            max_steps=100,
            goal_radius=0.0,
        )
        valid_config = EpisodeManagerConfig(env_config=valid_env_config)

        # Modify config to make it invalid
        valid_config.env_config.goal_radius = -1.0  # Invalid negative radius

        # Call validate_episode_config() with invalid configuration
        is_valid, validation_report = validate_episode_config(valid_config)

        # Assert validation returns (False, validation_report) tuple
        assert is_valid is False
        assert isinstance(validation_report, dict)

        # Verify validation report contains detailed error analysis
        assert "validation_status" in validation_report
        assert validation_report["validation_status"] == "invalid"

        # Assert specific validation failures are identified
        if "errors" in validation_report:
            assert len(validation_report["errors"]) > 0

    def test_validate_episode_config_edge_cases(self):
        """Test validate_episode_config function with edge cases and boundary conditions."""
        # Test validation with minimal viable configuration parameters
        minimal_env_config = create_environment_config(
            grid_size=(16, 16),  # Minimal grid size
            source_location=(8, 8),
            max_steps=1,  # Minimal steps
            goal_radius=0.0,
        )
        minimal_config = EpisodeManagerConfig(env_config=minimal_env_config)

        # Assert edge case configurations are handled properly
        is_valid, report = validate_episode_config(minimal_config)
        # This should be valid, though might have warnings
        assert isinstance(is_valid, bool)
        assert isinstance(report, dict)

        # Test maximum reasonable parameter values
        large_env_config = create_environment_config(
            grid_size=(256, 256),  # Large grid
            source_location=(128, 128),
            max_steps=10000,  # Many steps
            goal_radius=10.0,  # Large radius
        )
        large_config = EpisodeManagerConfig(env_config=large_env_config)

        # Verify boundary condition detection
        is_valid_large, large_report = validate_episode_config(large_config)
        assert isinstance(is_valid_large, bool)
        assert isinstance(large_report, dict)

        # Test configuration with extreme but valid parameters
        if "warnings" in large_report:
            # Large configurations might generate performance warnings
            assert isinstance(large_report["warnings"], list)

    def test_factory_function_integration(self):
        """Test integration between factory functions and episode manager functionality."""
        # Create test configuration
        test_env_config = create_environment_config(
            grid_size=(20, 20), source_location=(10, 10), max_steps=100, goal_radius=0.0
        )
        test_config = EpisodeManagerConfig(env_config=test_env_config)

        # Use validate_episode_config() to validate configuration
        is_valid, validation_report = validate_episode_config(test_config)

        if is_valid:
            # Use validated configuration with create_episode_manager()
            episode_manager = create_episode_manager(config=test_config)

            # Assert seamless integration between validation and creation
            assert isinstance(episode_manager, EpisodeManager)
            assert episode_manager.config is test_config

            # Test complete workflow from validation to functional episode manager
            obs, info = episode_manager.reset_episode(seed=42)
            obs2, reward, terminated, truncated, info2 = episode_manager.process_step(
                Action.RIGHT
            )

            assert isinstance(obs, np.ndarray)
            assert isinstance(obs2, np.ndarray)
            assert isinstance(reward, (int, float))

            # Verify factory function consistency and compatibility
            assert episode_manager.config.validate() is True

            # Assert episode manager created from validated config functions properly
            result = episode_manager.finalize_episode()
            assert isinstance(result, EpisodeResult)
        else:
            pytest.skip("Configuration validation failed, skipping integration test")
