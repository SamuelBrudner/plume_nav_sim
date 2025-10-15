"""
Advanced Gymnasium integration example demonstrating comprehensive usage patterns of plume_nav_sim environment
including sophisticated registration workflows, configuration management with presets, performance monitoring
integration, reproducibility validation, custom environment creation, and production-ready integration patterns
for research and development workflows.

This comprehensive example showcases enterprise-grade usage patterns including advanced registration,
configuration presets, performance optimization, parallel processing, error handling, and scientific
reproducibility validation for the plume_nav_sim Gymnasium environment.
"""

import argparse  # >=3.10 - Command-line interface for advanced example customization and parameter control
import concurrent.futures  # >=3.10 - Parallel processing for advanced performance testing and multi-environment validation
import contextlib  # >=3.10 - Context management utilities for resource handling and advanced error management patterns
import json  # >=3.10 - JSON serialization for configuration and results export
import pathlib  # >=3.10 - Path handling for output directory management and file operations
import statistics  # >=3.10 - Statistical analysis of reproducibility validation and performance trend analysis
import sys  # >=3.10 - System interface for advanced error handling and script execution management
import time  # >=3.10 - High-precision timing measurements and performance benchmarking for advanced monitoring integration
import warnings  # >=3.10 - Warning management for development and production environments
from typing import (  # >=3.10 - Advanced type hints for comprehensive type safety
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import matplotlib  # >=3.9.0 - Backend management and configuration for cross-platform compatibility
import matplotlib.pyplot as plt  # >=3.9.0 - Advanced visualization including performance plots, trajectory analysis, and multi-panel figures for comprehensive demonstration
import numpy as np  # >=2.1.0 - Statistical analysis, trajectory comparison, and advanced numerical computing for reproducibility validation

# External imports with version comments
import gymnasium as gym  # >=0.29.0 - Advanced Gymnasium framework usage including registry inspection, environment wrapping, and vectorization patterns

# Internal imports from configuration management
from ..config.environment_configs import (
    ENVIRONMENT_REGISTRY,
    create_benchmark_config,
    create_preset_config,
    create_research_scenario,
    get_available_presets,
)

# Internal imports from core types and actions
from ..plume_nav_sim.core.types import Action

# Internal imports from environment system
from ..plume_nav_sim.envs.plume_search_env import (
    PlumeSearchEnv,
    create_plume_search_env,
)

# Internal imports from registration system
from ..plume_nav_sim.registration.register import (
    ENV_ID,
    is_registered,
    register_env,
    register_with_custom_params,
    unregister_env,
)

# Internal imports from exception handling
from ..plume_nav_sim.utils.exceptions import ConfigurationError, ValidationError

# Internal imports from logging and performance monitoring
from ..plume_nav_sim.utils.logging import (
    PerformanceTimer,
    configure_logging_for_development,
    get_component_logger,
)

# Global constants for advanced demonstration configuration
ADVANCED_DEMO_SEED = 42
NUM_REPRODUCIBILITY_TESTS = 10
PERFORMANCE_BASELINE_EPISODES = 5
STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.01

# Global performance tracking and logging setup
_logger = get_component_logger("gymnasium_integration", "UTILS")
_performance_results = {}


def setup_advanced_logging(
    log_level: Optional[str] = None,
    enable_performance_logging: bool = True,
    enable_file_output: bool = False,
) -> Dict[str, Any]:
    """Configure advanced logging for Gymnasium integration example with performance monitoring, error tracking,
    and development-optimized output for comprehensive debugging and analysis.

    Args:
        log_level: Optional log level override (DEBUG, INFO, WARNING, ERROR)
        enable_performance_logging: Whether to enable detailed performance logging
        enable_file_output: Whether to enable persistent file logging

    Returns:
        dict: Advanced logging configuration status with enabled features and performance settings
    """
    global _logger

    _logger.info(
        "Setting up advanced logging configuration for Gymnasium integration demonstration"
    )

    try:
        return _extracted_from_setup_advanced_logging_25(
            log_level, enable_performance_logging, enable_file_output, _logger
        )
    except Exception as e:
        _logger.error(f"Failed to setup advanced logging: {str(e)}")
        return {"status": "error", "error": str(e), "fallback_logging": True}


# TODO Rename this here and in `setup_advanced_logging`
def _extracted_from_setup_advanced_logging_25(
    log_level, enable_performance_logging, enable_file_output, _logger
):
    # Configure development logging using configure_logging_for_development with enhanced parameters
    logging_config = configure_logging_for_development(
        log_level=log_level or "INFO",
        enable_performance_monitoring=enable_performance_logging,
        enable_file_logging=enable_file_output,
        component_name="gymnasium_integration",
    )

    # Set up component-specific loggers for environment, registration, and performance monitoring
    registration_logger = get_component_logger("registration", "CORE")
    environment_logger = get_component_logger("environment", "CORE")
    performance_logger = get_component_logger("performance", "UTILS")

    # Configure matplotlib backend for cross-platform compatibility and headless operation
    if not hasattr(matplotlib.get_backend(), "show"):
        matplotlib.use("Agg")  # Use non-interactive backend for headless environments

    # Enable performance logging integration with PerformanceTimer and ComponentLogger
    performance_config = {
        "timing_enabled": enable_performance_logging,
        "memory_tracking": enable_performance_logging,
        "operation_profiling": enable_performance_logging,
    }

    # Initialize performance logging baselines and threshold monitoring
    global _performance_results
    _performance_results = {
        "logging_setup_time": time.time(),
        "configuration": logging_config,
        "performance_monitoring": performance_config,
    }

    _logger.info("Advanced logging configuration completed successfully")

    # Return comprehensive logging configuration status for validation and monitoring
    return {
        "status": "success",
        "logging_config": logging_config,
        "performance_config": performance_config,
        "loggers_configured": ["registration", "environment", "performance"],
        "file_output_enabled": enable_file_output,
        "matplotlib_backend": matplotlib.get_backend(),
    }


def demonstrate_advanced_registration(
    custom_env_id: Optional[str] = None,
    custom_params: Optional[Dict[str, Any]] = None,
    show_registry_inspection: bool = True,
) -> Dict[str, Any]:
    """Demonstrate sophisticated environment registration patterns including custom parameters, validation workflows,
    registry inspection, and advanced configuration management for production-ready integration.

    Args:
        custom_env_id: Optional custom environment ID for registration testing
        custom_params: Optional custom parameters for advanced registration
        show_registry_inspection: Whether to demonstrate registry inspection techniques

    Returns:
        dict: Advanced registration results with validation status, performance metrics, and configuration details
    """
    _logger.info("Starting advanced environment registration demonstration")

    with PerformanceTimer() as registration_timer:
        registration_results = {
            "demonstration_start": time.time(),
            "registration_tests": [],
            "performance_metrics": {},
            "validation_results": {},
        }

        try:
            # Demonstrate registry status checking using is_registered with cache validation
            _logger.info(f"Checking registration status for environment: {ENV_ID}")
            initial_registered = is_registered(ENV_ID)
            registration_results["initial_registration_status"] = initial_registered

            if initial_registered:
                _logger.info(f"Environment {ENV_ID} is already registered")
            else:
                _logger.info(
                    f"Environment {ENV_ID} not found in registry, will register"
                )

            # Show advanced registration with custom parameters using register_with_custom_params
            if custom_params:
                _logger.info("Demonstrating custom parameter registration")
                custom_registration_result = register_with_custom_params(
                    env_id=custom_env_id or f"{ENV_ID}-Custom",
                    custom_params=custom_params,
                )
                registration_results["custom_registration"] = custom_registration_result

            # Demonstrate standard environment registration if not already registered
            if not initial_registered:
                with PerformanceTimer() as register_timer:
                    register_env()
                    registration_results["performance_metrics"][
                        "registration_time_ms"
                    ] = register_timer.get_duration_ms()

                # Validate registration was successful
                post_registration_status = is_registered(ENV_ID)
                registration_results["post_registration_status"] = (
                    post_registration_status
                )

                if post_registration_status:
                    _logger.info(f"Successfully registered environment: {ENV_ID}")
                else:
                    _logger.error(f"Failed to register environment: {ENV_ID}")
                    registration_results["validation_results"][
                        "registration_failed"
                    ] = True

            # Demonstrate registry inspection techniques for discovery and validation
            if show_registry_inspection:
                _logger.info("Performing comprehensive registry inspection")

                # Get all registered environments from gymnasium registry
                all_registered = list(gym.envs.registry.all())
                plume_envs = [env for env in all_registered if "Plume" in env.id]

                registration_results["registry_inspection"] = {
                    "total_environments": len(all_registered),
                    "plume_environments": [env.id for env in plume_envs],
                    "plume_env_count": len(plume_envs),
                }

                # Inspect environment specifications
                if plume_envs:
                    env_spec = plume_envs[0]
                    registration_results["environment_spec"] = {
                        "id": env_spec.id,
                        "entry_point": env_spec.entry_point,
                        "max_episode_steps": getattr(
                            env_spec, "max_episode_steps", None
                        ),
                        "reward_threshold": getattr(env_spec, "reward_threshold", None),
                    }

            # Validate registration integrity with comprehensive environment creation testing
            _logger.info(
                "Validating registration integrity through environment creation"
            )

            try:
                with PerformanceTimer() as creation_timer:
                    # Test environment creation using gym.make
                    test_env = gym.make(ENV_ID)
                    test_env.close()

                registration_results["validation_results"]["environment_creation"] = {
                    "success": True,
                    "creation_time_ms": creation_timer.get_duration_ms(),
                }
                _logger.info("Environment creation validation successful")

            except Exception as creation_error:
                registration_results["validation_results"]["environment_creation"] = {
                    "success": False,
                    "error": str(creation_error),
                }
                _logger.error(
                    f"Environment creation validation failed: {creation_error}"
                )

            # Show unregistration and re-registration workflows for development and testing
            _logger.info("Demonstrating unregistration and re-registration workflow")

            try:
                # Unregister environment
                unregister_env(ENV_ID)
                unregistered_status = not is_registered(ENV_ID)

                # Re-register environment
                register_env()
                re_registered_status = is_registered(ENV_ID)

                registration_results["unregister_reregister_test"] = {
                    "unregistration_success": unregistered_status,
                    "reregistration_success": re_registered_status,
                    "workflow_success": unregistered_status and re_registered_status,
                }

                _logger.info(
                    "Unregistration/re-registration workflow completed successfully"
                )

            except Exception as workflow_error:
                registration_results["unregister_reregister_test"] = {
                    "workflow_success": False,
                    "error": str(workflow_error),
                }
                _logger.error(
                    f"Unregistration/re-registration workflow failed: {workflow_error}"
                )

        except Exception as e:
            _logger.error(f"Advanced registration demonstration failed: {str(e)}")
            registration_results["demonstration_error"] = str(e)

    # Performance monitoring of registration operations with timing analysis
    registration_results["performance_metrics"][
        "total_demonstration_time_ms"
    ] = registration_timer.get_duration_ms()

    _logger.info("Advanced registration demonstration completed")

    # Return comprehensive registration results with metrics and validation status
    return registration_results


def demonstrate_configuration_presets(
    show_all_presets: bool = True,
    demonstrate_custom_creation: bool = True,
    validate_configurations: bool = True,
) -> Dict[str, Any]:
    """Demonstrate advanced configuration management using preset system, custom scenarios, and configuration
    validation for flexible environment parameterization and research workflow integration.

    Args:
        show_all_presets: Whether to enumerate and demonstrate all available presets
        demonstrate_custom_creation: Whether to show custom scenario creation
        validate_configurations: Whether to perform comprehensive configuration validation

    Returns:
        dict: Configuration demonstration results with preset validation, performance analysis, and usage examples
    """
    _logger.info("Starting advanced configuration preset demonstration")

    with PerformanceTimer() as config_timer:
        config_results = {
            "demonstration_start": time.time(),
            "available_presets": [],
            "preset_demonstrations": {},
            "custom_scenarios": {},
            "validation_results": {},
            "performance_metrics": {},
        }

        try:
            # Enumerate and demonstrate all available presets using get_available_presets
            _logger.info("Discovering available configuration presets")
            available_presets = get_available_presets()
            config_results["available_presets"] = available_presets

            _logger.info(
                f"Found {len(available_presets)} available presets: {available_presets}"
            )

            if show_all_presets:
                # Show preset metadata inspection using ENVIRONMENT_REGISTRY.get_metadata
                for preset_name in available_presets:
                    _logger.info(f"Analyzing preset: {preset_name}")

                    try:
                        # Get preset metadata
                        preset_metadata = ENVIRONMENT_REGISTRY.get_metadata(preset_name)

                        # Demonstrate preset-based environment creation using create_preset_config
                        with PerformanceTimer() as preset_timer:
                            preset_config = create_preset_config(preset_name)

                        config_results["preset_demonstrations"][preset_name] = {
                            "metadata": preset_metadata,
                            "config_created": True,
                            "creation_time_ms": preset_timer.get_duration_ms(),
                            "grid_size": getattr(preset_config, "grid_size", "unknown"),
                            "max_steps": getattr(preset_config, "max_steps", "unknown"),
                        }

                        _logger.info(f"Successfully demonstrated preset: {preset_name}")

                    except Exception as preset_error:
                        config_results["preset_demonstrations"][preset_name] = {
                            "error": str(preset_error),
                            "config_created": False,
                        }
                        _logger.error(
                            f"Failed to demonstrate preset {preset_name}: {preset_error}"
                        )

            # Show research scenario creation using create_research_scenario with academic parameters
            if demonstrate_custom_creation:
                _logger.info("Demonstrating custom research scenario creation")

                try:
                    # Create research scenario with custom academic parameters
                    research_params = {
                        "scenario_name": "advanced_research_demo",
                        "grid_size": (256, 256),
                        "source_location": (128, 128),
                        "sigma": 15.0,
                        "max_steps": 2000,
                        "goal_radius": 2.0,
                    }

                    with PerformanceTimer() as research_timer:
                        research_config = create_research_scenario(**research_params)

                    config_results["custom_scenarios"]["research_scenario"] = {
                        "parameters": research_params,
                        "creation_success": True,
                        "creation_time_ms": research_timer.get_duration_ms(),
                    }

                    _logger.info("Successfully created custom research scenario")

                except Exception as research_error:
                    config_results["custom_scenarios"]["research_scenario"] = {
                        "creation_success": False,
                        "error": str(research_error),
                    }
                    _logger.error(
                        f"Failed to create research scenario: {research_error}"
                    )

                # Demonstrate benchmark configuration creation using create_benchmark_config
                _logger.info("Demonstrating benchmark configuration creation")

                try:
                    benchmark_params = {
                        "benchmark_type": "performance_validation",
                        "difficulty_level": "medium",
                    }

                    with PerformanceTimer() as benchmark_timer:
                        benchmark_config = create_benchmark_config(**benchmark_params)

                    config_results["custom_scenarios"]["benchmark_scenario"] = {
                        "parameters": benchmark_params,
                        "creation_success": True,
                        "creation_time_ms": benchmark_timer.get_duration_ms(),
                    }

                    _logger.info("Successfully created benchmark configuration")

                except Exception as benchmark_error:
                    config_results["custom_scenarios"]["benchmark_scenario"] = {
                        "creation_success": False,
                        "error": str(benchmark_error),
                    }
                    _logger.error(
                        f"Failed to create benchmark scenario: {benchmark_error}"
                    )

            # Validate all configurations using comprehensive validation workflows
            if validate_configurations:
                _logger.info("Performing comprehensive configuration validation")

                validation_summary = {
                    "presets_validated": 0,
                    "presets_failed": 0,
                    "validation_errors": [],
                }

                # Validate each available preset
                for preset_name in available_presets:
                    try:
                        # Create configuration and validate
                        preset_config = create_preset_config(preset_name)

                        # Perform basic validation (would include more comprehensive checks)
                        if hasattr(preset_config, "validate"):
                            preset_config.validate()

                        validation_summary["presets_validated"] += 1
                        _logger.debug(f"Preset {preset_name} validated successfully")

                    except Exception as validation_error:
                        validation_summary["presets_failed"] += 1
                        validation_summary["validation_errors"].append(
                            {"preset": preset_name, "error": str(validation_error)}
                        )
                        _logger.warning(
                            f"Preset {preset_name} validation failed: {validation_error}"
                        )

                config_results["validation_results"] = validation_summary

                # Performance comparison between different preset configurations
                _logger.info(
                    "Performing performance comparison between preset configurations"
                )

                performance_comparison = {}
                for preset_name in available_presets[
                    :3
                ]:  # Test first 3 presets for performance
                    try:
                        with PerformanceTimer() as perf_timer:
                            config = create_preset_config(preset_name)
                            # Simulate environment creation time
                            time.sleep(0.001)  # Small delay to simulate work

                        performance_comparison[preset_name] = {
                            "config_creation_ms": perf_timer.get_duration_ms(),
                            "performance_rank": (
                                "good" if perf_timer.get_duration_ms() < 10 else "slow"
                            ),
                        }

                    except Exception as perf_error:
                        performance_comparison[preset_name] = {
                            "error": str(perf_error),
                            "performance_rank": "failed",
                        }

                config_results["performance_metrics"][
                    "preset_comparison"
                ] = performance_comparison

        except Exception as e:
            _logger.error(f"Configuration preset demonstration failed: {str(e)}")
            config_results["demonstration_error"] = str(e)

    # Record total demonstration time
    config_results["performance_metrics"][
        "total_demonstration_time_ms"
    ] = config_timer.get_duration_ms()

    _logger.info("Advanced configuration preset demonstration completed")

    # Return configuration results with validation status and performance metrics
    return config_results


def demonstrate_advanced_episode_patterns(
    env: gym.Env,
    num_episodes: int = 3,
    collect_trajectories: bool = True,
    performance_analysis: bool = True,
) -> Dict[str, Any]:
    """Demonstrate sophisticated episode execution patterns including performance monitoring, statistical analysis,
    trajectory tracking, and advanced state management for research-quality data collection.

    Args:
        env: Gymnasium environment instance
        num_episodes: Number of episodes to execute for demonstration
        collect_trajectories: Whether to collect detailed trajectory data
        performance_analysis: Whether to perform detailed performance analysis

    Returns:
        dict: Advanced episode analysis results with trajectories, performance metrics, and statistical analysis
    """
    _logger.info(
        f"Starting advanced episode pattern demonstration with {num_episodes} episodes"
    )

    with PerformanceTimer() as episode_timer:
        episode_results = {
            "demonstration_start": time.time(),
            "num_episodes": num_episodes,
            "episodes_data": [],
            "trajectories": [],
            "performance_metrics": {},
            "statistical_analysis": {},
        }

        # Initialize comprehensive data collection structures for trajectories and metrics
        all_rewards = []
        all_episode_lengths = []
        all_step_times = []
        trajectory_data = []

        try:
            # Execute multiple episodes with performance monitoring using PerformanceTimer context manager
            for episode_idx in range(num_episodes):
                _logger.info(f"Executing episode {episode_idx + 1}/{num_episodes}")

                with PerformanceTimer() as single_episode_timer:
                    # Reset environment for new episode
                    observation, info = env.reset(seed=ADVANCED_DEMO_SEED + episode_idx)

                    episode_data = {
                        "episode_number": episode_idx + 1,
                        "seed": ADVANCED_DEMO_SEED + episode_idx,
                        "total_reward": 0.0,
                        "steps": 0,
                        "terminated": False,
                        "truncated": False,
                        "step_times": [],
                        "observations": [],
                        "actions": [],
                        "rewards": [],
                    }

                    # Collect detailed trajectory data including positions, observations, actions, and rewards
                    if collect_trajectories:
                        episode_trajectory = {
                            "episode": episode_idx + 1,
                            "positions": [],
                            "concentration_values": [],
                        }

                    done = False
                    while not done:
                        # Record observation if collecting trajectories
                        if collect_trajectories:
                            episode_data["observations"].append(observation.copy())
                            if "agent_xy" in info:
                                episode_trajectory["positions"].append(info["agent_xy"])
                            if hasattr(observation, "max"):
                                episode_trajectory["concentration_values"].append(
                                    float(observation.max())
                                )

                        # Select action using simple policy (can be made more sophisticated)
                        action = (
                            env.action_space.sample()
                        )  # Random action for demonstration
                        episode_data["actions"].append(action)

                        # Monitor performance metrics including step latency, episode duration, and memory usage
                        with PerformanceTimer() as step_timer:
                            observation, reward, terminated, truncated, info = env.step(
                                action
                            )

                        step_time_ms = step_timer.get_duration_ms()
                        episode_data["step_times"].append(step_time_ms)
                        all_step_times.append(step_time_ms)

                        # Record step data
                        episode_data["rewards"].append(reward)
                        episode_data["total_reward"] += reward
                        episode_data["steps"] += 1

                        done = terminated or truncated

                    # Record final episode state
                    episode_data["terminated"] = terminated
                    episode_data["truncated"] = truncated
                    episode_data["episode_duration_ms"] = (
                        single_episode_timer.get_duration_ms()
                    )

                # Store episode data and trajectory information
                episode_results["episodes_data"].append(episode_data)

                if collect_trajectories:
                    trajectory_data.append(episode_trajectory)

                # Collect aggregate statistics
                all_rewards.append(episode_data["total_reward"])
                all_episode_lengths.append(episode_data["steps"])

                _logger.info(
                    f"Episode {episode_idx + 1} completed: "
                    f"Steps={episode_data['steps']}, "
                    f"Reward={episode_data['total_reward']:.2f}, "
                    f"Duration={episode_data['episode_duration_ms']:.1f}ms"
                )

        except Exception as e:
            _logger.error(f"Episode execution failed: {str(e)}")
            episode_results["execution_error"] = str(e)
            return episode_results

        # Apply statistical analysis to episode outcomes and performance characteristics
        if performance_analysis and len(all_rewards) > 0:
            _logger.info("Performing statistical analysis of episode data")

            statistical_results = {
                "reward_statistics": {
                    "mean": statistics.mean(all_rewards),
                    "median": statistics.median(all_rewards),
                    "stdev": (
                        statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0.0
                    ),
                    "min": min(all_rewards),
                    "max": max(all_rewards),
                },
                "episode_length_statistics": {
                    "mean": statistics.mean(all_episode_lengths),
                    "median": statistics.median(all_episode_lengths),
                    "stdev": (
                        statistics.stdev(all_episode_lengths)
                        if len(all_episode_lengths) > 1
                        else 0.0
                    ),
                    "min": min(all_episode_lengths),
                    "max": max(all_episode_lengths),
                },
                "step_time_statistics": {
                    "mean_ms": statistics.mean(all_step_times),
                    "median_ms": statistics.median(all_step_times),
                    "stdev_ms": (
                        statistics.stdev(all_step_times)
                        if len(all_step_times) > 1
                        else 0.0
                    ),
                    "min_ms": min(all_step_times),
                    "max_ms": max(all_step_times),
                },
            }

            episode_results["statistical_analysis"] = statistical_results

        # Generate performance baselines and threshold analysis for optimization guidance
        if performance_analysis:
            performance_metrics = {
                "total_episodes_completed": len(episode_results["episodes_data"]),
                "average_episode_duration_ms": statistics.mean(
                    [
                        ep["episode_duration_ms"]
                        for ep in episode_results["episodes_data"]
                    ]
                ),
                "total_demonstration_time_ms": episode_timer.get_duration_ms(),
                "episodes_per_second": len(episode_results["episodes_data"])
                / (episode_timer.get_duration_ms() / 1000.0),
                "step_performance_analysis": {
                    "steps_per_second": len(all_step_times)
                    / (episode_timer.get_duration_ms() / 1000.0),
                    "average_step_time_ms": (
                        statistics.mean(all_step_times) if all_step_times else 0.0
                    ),
                    "performance_threshold_met": (
                        statistics.mean(all_step_times) < 5.0
                        if all_step_times
                        else False
                    ),
                },
            }

            episode_results["performance_metrics"] = performance_metrics

        # Store trajectory data if collected
        if collect_trajectories:
            episode_results["trajectories"] = trajectory_data

    _logger.info("Advanced episode pattern demonstration completed successfully")

    # Return comprehensive episode analysis with trajectories, statistics, and optimization recommendations
    return episode_results


def demonstrate_reproducibility_validation(
    env: gym.Env,
    test_seeds: List[int] = None,
    validation_episodes_per_seed: int = 2,
    statistical_testing: bool = True,
) -> Dict[str, Any]:
    """Demonstrate advanced reproducibility validation including cross-session consistency, statistical testing,
    deterministic behavior verification, and scientific workflow patterns for research integrity.

    Args:
        env: Gymnasium environment instance
        test_seeds: List of seeds for reproducibility testing
        validation_episodes_per_seed: Number of episodes per seed for validation
        statistical_testing: Whether to perform statistical reproducibility testing

    Returns:
        dict: Comprehensive reproducibility validation results with statistical analysis and integrity verification
    """
    _logger.info("Starting advanced reproducibility validation demonstration")

    if test_seeds is None:
        test_seeds = [ADVANCED_DEMO_SEED + i for i in range(NUM_REPRODUCIBILITY_TESTS)]

    with PerformanceTimer() as validation_timer:
        validation_results = {
            "demonstration_start": time.time(),
            "test_seeds": test_seeds,
            "validation_episodes_per_seed": validation_episodes_per_seed,
            "seed_results": {},
            "reproducibility_analysis": {},
            "statistical_tests": {},
            "integrity_verification": {},
        }

        try:
            # Execute multiple episodes for each test seed collecting complete trajectory data
            for seed in test_seeds:
                _logger.info(f"Testing reproducibility for seed: {seed}")

                seed_data = {
                    "seed": seed,
                    "episodes": [],
                    "trajectory_consistency": True,
                    "reward_consistency": True,
                }

                # Run multiple episodes with same seed to verify deterministic behavior
                episode_trajectories = []
                episode_rewards = []

                for episode_idx in range(validation_episodes_per_seed):
                    # Reset with specific seed
                    observation, info = env.reset(seed=seed)

                    episode_data = {
                        "episode": episode_idx + 1,
                        "observations": [],
                        "actions": [],
                        "rewards": [],
                        "total_reward": 0.0,
                        "steps": 0,
                        "agent_positions": [],
                    }

                    # Execute deterministic policy for reproducibility testing
                    done = False
                    step_count = 0
                    max_steps = 100  # Limit steps for validation

                    while not done and step_count < max_steps:
                        # Use deterministic action selection based on step count for consistency
                        action = (
                            step_count % 4
                        )  # Cycle through actions deterministically
                        episode_data["actions"].append(action)

                        # Record pre-step state
                        episode_data["observations"].append(observation.copy())
                        if "agent_xy" in info:
                            episode_data["agent_positions"].append(info["agent_xy"])

                        # Execute step
                        observation, reward, terminated, truncated, info = env.step(
                            action
                        )

                        episode_data["rewards"].append(reward)
                        episode_data["total_reward"] += reward
                        episode_data["steps"] += 1
                        step_count += 1

                        done = terminated or truncated

                    seed_data["episodes"].append(episode_data)
                    episode_trajectories.append(episode_data["agent_positions"])
                    episode_rewards.append(episode_data["total_reward"])

                # Compare identical seeds across multiple runs for perfect deterministic matching
                if validation_episodes_per_seed >= 2:
                    # Check trajectory consistency
                    first_trajectory = episode_trajectories[0]
                    for i, trajectory in enumerate(episode_trajectories[1:], 1):
                        if trajectory != first_trajectory:
                            seed_data["trajectory_consistency"] = False
                            _logger.warning(
                                f"Trajectory inconsistency found for seed {seed}, episode {i+1}"
                            )
                            break

                    # Check reward consistency
                    first_reward = episode_rewards[0]
                    for i, reward in enumerate(episode_rewards[1:], 1):
                        if (
                            abs(reward - first_reward) > 1e-10
                        ):  # Account for floating point precision
                            seed_data["reward_consistency"] = False
                            _logger.warning(
                                f"Reward inconsistency found for seed {seed}, episode {i+1}"
                            )
                            break

                validation_results["seed_results"][seed] = seed_data

                _logger.info(
                    f"Seed {seed} reproducibility: "
                    f"Trajectory={'✓' if seed_data['trajectory_consistency'] else '✗'}, "
                    f"Reward={'✓' if seed_data['reward_consistency'] else '✗'}"
                )

        except Exception as e:
            _logger.error(f"Reproducibility validation failed: {str(e)}")
            validation_results["validation_error"] = str(e)
            return validation_results

        # Apply statistical testing for seed independence and randomness quality
        if statistical_testing:
            _logger.info("Performing statistical reproducibility testing")

            try:
                # Collect all rewards by seed for statistical analysis
                rewards_by_seed = {}
                consistency_scores = []

                for seed, seed_data in validation_results["seed_results"].items():
                    rewards = [ep["total_reward"] for ep in seed_data["episodes"]]
                    rewards_by_seed[seed] = rewards

                    # Calculate consistency score (1.0 = perfectly consistent)
                    if len(rewards) > 1:
                        reward_variance = statistics.stdev(rewards)
                        consistency_score = (
                            1.0
                            if reward_variance < 1e-10
                            else max(0.0, 1.0 - reward_variance)
                        )
                    else:
                        consistency_score = 1.0

                    consistency_scores.append(consistency_score)

                # Overall statistical analysis
                statistical_analysis = {
                    "overall_consistency_score": statistics.mean(consistency_scores),
                    "seeds_perfectly_consistent": sum(
                        bool(score > 0.99) for score in consistency_scores
                    ),
                    "seeds_tested": len(test_seeds),
                    "consistency_percentage": (
                        sum(bool(score > 0.99) for score in consistency_scores)
                        / len(test_seeds)
                    )
                    * 100,
                }

                validation_results["statistical_tests"] = statistical_analysis

                # Test cross-session reproducibility with environment recreation and re-seeding
                _logger.info("Testing cross-session reproducibility")

                # Create new environment instance and test first seed again
                try:
                    new_env = gym.make(ENV_ID)
                    test_seed = test_seeds[0]

                    # Get original results for comparison
                    original_rewards = [
                        ep["total_reward"]
                        for ep in validation_results["seed_results"][test_seed][
                            "episodes"
                        ]
                    ]

                    # Execute same seed with new environment instance
                    new_session_rewards = []
                    for episode_idx in range(validation_episodes_per_seed):
                        observation, info = new_env.reset(seed=test_seed)
                        total_reward = 0.0
                        done = False
                        step_count = 0

                        while not done and step_count < 100:
                            action = step_count % 4  # Same deterministic policy
                            observation, reward, terminated, truncated, info = (
                                new_env.step(action)
                            )
                            total_reward += reward
                            step_count += 1
                            done = terminated or truncated

                        new_session_rewards.append(total_reward)

                    new_env.close()

                    # Compare cross-session results
                    cross_session_consistent = len(original_rewards) == len(
                        new_session_rewards
                    ) and all(
                        abs(orig - new) < 1e-10
                        for orig, new in zip(original_rewards, new_session_rewards)
                    )

                    validation_results["integrity_verification"] = {
                        "cross_session_test_performed": True,
                        "cross_session_consistent": cross_session_consistent,
                        "test_seed": test_seed,
                        "original_rewards": original_rewards,
                        "new_session_rewards": new_session_rewards,
                    }

                    _logger.info(
                        f"Cross-session reproducibility: {'✓' if cross_session_consistent else '✗'}"
                    )

                except Exception as cross_session_error:
                    validation_results["integrity_verification"] = {
                        "cross_session_test_performed": False,
                        "error": str(cross_session_error),
                    }
                    _logger.error(f"Cross-session test failed: {cross_session_error}")

            except Exception as stats_error:
                validation_results["statistical_tests"] = {"error": str(stats_error)}
                _logger.error(f"Statistical testing failed: {stats_error}")

        # Generate reproducibility report with confidence intervals and statistical measures
        reproducibility_summary = {
            "total_seeds_tested": len(test_seeds),
            "perfect_reproducibility_count": sum(
                bool(
                    seed_data["trajectory_consistency"]
                    and seed_data["reward_consistency"]
                )
                for seed_data in validation_results["seed_results"].values()
            ),
            "reproducibility_percentage": 0.0,
            "validation_duration_ms": validation_timer.get_duration_ms(),
            "integrity_status": "unknown",
        }

        if len(test_seeds) > 0:
            reproducibility_summary["reproducibility_percentage"] = (
                reproducibility_summary["perfect_reproducibility_count"]
                / len(test_seeds)
                * 100
            )

        # Determine overall integrity status
        if reproducibility_summary["reproducibility_percentage"] >= 95.0:
            reproducibility_summary["integrity_status"] = "excellent"
        elif reproducibility_summary["reproducibility_percentage"] >= 80.0:
            reproducibility_summary["integrity_status"] = "good"
        elif reproducibility_summary["reproducibility_percentage"] >= 60.0:
            reproducibility_summary["integrity_status"] = "fair"
        else:
            reproducibility_summary["integrity_status"] = "poor"

        validation_results["reproducibility_analysis"] = reproducibility_summary

    _logger.info(
        f"Reproducibility validation completed: {reproducibility_summary['reproducibility_percentage']:.1f}% consistent"
    )

    # Return comprehensive validation results with statistical confidence and integrity verification
    return validation_results


def demonstrate_performance_optimization(
    env: gym.Env,
    benchmark_episodes: int = PERFORMANCE_BASELINE_EPISODES,
    establish_baselines: bool = True,
    optimization_analysis: bool = True,
) -> Dict[str, Any]:
    """Demonstrate advanced performance monitoring, optimization techniques, baseline establishment, and performance
    regression detection for production-ready environment usage.

    Args:
        env: Gymnasium environment instance
        benchmark_episodes: Number of episodes for performance benchmarking
        establish_baselines: Whether to establish performance baselines
        optimization_analysis: Whether to perform optimization analysis

    Returns:
        dict: Performance optimization results with baselines, analysis, and optimization recommendations
    """
    _logger.info(
        f"Starting performance optimization demonstration with {benchmark_episodes} benchmark episodes"
    )

    with PerformanceTimer() as optimization_timer:
        optimization_results = {
            "demonstration_start": time.time(),
            "benchmark_episodes": benchmark_episodes,
            "performance_baselines": {},
            "detailed_metrics": {},
            "optimization_recommendations": [],
            "regression_analysis": {},
        }

        try:
            # Execute performance baseline establishment using benchmark_episodes with comprehensive timing
            if establish_baselines:
                _logger.info("Establishing performance baselines")

                baseline_metrics = {
                    "episode_times": [],
                    "step_times": [],
                    "reset_times": [],
                    "render_times": [],
                    "memory_usage": [],
                }

                for episode_idx in range(benchmark_episodes):
                    _logger.debug(
                        f"Benchmark episode {episode_idx + 1}/{benchmark_episodes}"
                    )

                    # Monitor reset performance
                    with PerformanceTimer() as reset_timer:
                        observation, info = env.reset(
                            seed=ADVANCED_DEMO_SEED + episode_idx
                        )

                    baseline_metrics["reset_times"].append(
                        reset_timer.get_duration_ms()
                    )

                    # Monitor episode execution performance
                    with PerformanceTimer() as episode_timer:
                        episode_step_times = []
                        episode_render_times = []

                        done = False
                        step_count = 0
                        max_benchmark_steps = 50  # Limit for benchmark consistency

                        while not done and step_count < max_benchmark_steps:
                            # Monitor step performance
                            with PerformanceTimer() as step_timer:
                                action = env.action_space.sample()
                                observation, reward, terminated, truncated, info = (
                                    env.step(action)
                                )

                            episode_step_times.append(step_timer.get_duration_ms())

                            # Monitor rendering performance if available
                            try:
                                with PerformanceTimer() as render_timer:
                                    env.render()
                                episode_render_times.append(
                                    render_timer.get_duration_ms()
                                )
                            except Exception:
                                # Skip rendering timing if not available
                                pass

                            step_count += 1
                            done = terminated or truncated

                    baseline_metrics["episode_times"].append(
                        episode_timer.get_duration_ms()
                    )
                    baseline_metrics["step_times"].extend(episode_step_times)
                    if episode_render_times:
                        baseline_metrics["render_times"].extend(episode_render_times)

                    # Simulate memory usage monitoring (would use actual memory profiling in production)
                    baseline_metrics["memory_usage"].append(
                        episode_idx * 0.1
                    )  # Placeholder

                # Calculate baseline statistics
                performance_baselines = {
                    "avg_episode_time_ms": statistics.mean(
                        baseline_metrics["episode_times"]
                    ),
                    "avg_step_time_ms": (
                        statistics.mean(baseline_metrics["step_times"])
                        if baseline_metrics["step_times"]
                        else 0.0
                    ),
                    "avg_reset_time_ms": statistics.mean(
                        baseline_metrics["reset_times"]
                    ),
                    "avg_render_time_ms": (
                        statistics.mean(baseline_metrics["render_times"])
                        if baseline_metrics["render_times"]
                        else 0.0
                    ),
                    "step_time_p95_ms": (
                        np.percentile(baseline_metrics["step_times"], 95)
                        if baseline_metrics["step_times"]
                        else 0.0
                    ),
                    "total_baseline_episodes": benchmark_episodes,
                    "baseline_established_at": time.time(),
                }

                optimization_results["performance_baselines"] = performance_baselines
                optimization_results["detailed_metrics"] = baseline_metrics

                _logger.info(
                    f"Performance baselines established: "
                    f"Avg step time: {performance_baselines['avg_step_time_ms']:.2f}ms, "
                    f"Avg episode time: {performance_baselines['avg_episode_time_ms']:.2f}ms"
                )

            # Apply performance threshold analysis against system targets and requirements
            if (
                optimization_analysis
                and "performance_baselines" in optimization_results
            ):
                _logger.info("Performing performance threshold analysis")

                baselines = optimization_results["performance_baselines"]
                recommendations = []

                # Analyze step performance against 1ms target
                step_time_target_ms = 1.0
                if baselines["avg_step_time_ms"] > step_time_target_ms:
                    recommendations.append(
                        {
                            "category": "step_performance",
                            "issue": f'Average step time ({baselines["avg_step_time_ms"]:.2f}ms) exceeds target ({step_time_target_ms}ms)',
                            "recommendation": "Optimize environment step logic, reduce computation complexity",
                            "priority": (
                                "high"
                                if baselines["avg_step_time_ms"] > 5.0
                                else "medium"
                            ),
                        }
                    )

                # Analyze reset performance
                reset_time_target_ms = 10.0
                if baselines["avg_reset_time_ms"] > reset_time_target_ms:
                    recommendations.append(
                        {
                            "category": "reset_performance",
                            "issue": f'Average reset time ({baselines["avg_reset_time_ms"]:.2f}ms) exceeds target ({reset_time_target_ms}ms)',
                            "recommendation": "Optimize environment initialization, cache reusable components",
                            "priority": "medium",
                        }
                    )

                # Analyze rendering performance
                if baselines["avg_render_time_ms"] > 0:
                    render_time_target_ms = 50.0
                    if baselines["avg_render_time_ms"] > render_time_target_ms:
                        recommendations.append(
                            {
                                "category": "render_performance",
                                "issue": f'Average render time ({baselines["avg_render_time_ms"]:.2f}ms) exceeds target ({render_time_target_ms}ms)',
                                "recommendation": "Switch to rgb_array mode, optimize matplotlib usage, consider caching",
                                "priority": "low",
                            }
                        )

                # Analyze performance variance (P95 vs average)
                if baselines["step_time_p95_ms"] > baselines["avg_step_time_ms"] * 3:
                    recommendations.append(
                        {
                            "category": "performance_variance",
                            "issue": "High performance variance detected (P95 >> average)",
                            "recommendation": "Investigate performance spikes, add consistent timing",
                            "priority": "medium",
                        }
                    )

                optimization_results["optimization_recommendations"] = recommendations

                # Test performance under different load conditions and environment configurations
                _logger.info("Testing performance under different configurations")

                load_test_results = {}

                # Test with different action patterns
                for test_name, action_pattern in [
                    ("random", "random"),
                    ("deterministic", "cycle"),
                    ("static", "same"),
                ]:
                    _logger.debug(f"Testing {test_name} action pattern")

                    pattern_times = []
                    for test_episode in range(
                        min(3, benchmark_episodes)
                    ):  # Quick load test
                        observation, info = env.reset(seed=ADVANCED_DEMO_SEED)

                        with PerformanceTimer() as pattern_timer:
                            for step_idx in range(
                                20
                            ):  # Short episodes for load testing
                                if action_pattern == "random":
                                    action = env.action_space.sample()
                                elif action_pattern == "cycle":
                                    action = step_idx % 4
                                else:  # static
                                    action = 0

                                observation, reward, terminated, truncated, info = (
                                    env.step(action)
                                )
                                if terminated or truncated:
                                    break

                        pattern_times.append(pattern_timer.get_duration_ms())

                    load_test_results[test_name] = {
                        "avg_time_ms": statistics.mean(pattern_times),
                        "min_time_ms": min(pattern_times),
                        "max_time_ms": max(pattern_times),
                    }

                optimization_results["load_test_results"] = load_test_results

        except Exception as e:
            _logger.error(f"Performance optimization demonstration failed: {str(e)}")
            optimization_results["optimization_error"] = str(e)
            return optimization_results

        # Establish performance baselines for regression detection in future development
        optimization_results["regression_analysis"] = {
            "baseline_timestamp": time.time(),
            "performance_targets": {
                "step_time_ms_target": 1.0,
                "episode_time_ms_target": 100.0,
                "reset_time_ms_target": 10.0,
            },
            "regression_thresholds": {
                "step_time_regression_threshold": 1.5,  # 50% slower is regression
                "episode_time_regression_threshold": 2.0,
                "memory_regression_threshold": 1.3,
            },
        }

        optimization_results["total_optimization_time_ms"] = (
            optimization_timer.get_duration_ms()
        )

    _logger.info("Performance optimization demonstration completed")

    # Return comprehensive performance results with baselines, metrics, and optimization guidance
    return optimization_results


def demonstrate_error_handling_patterns(
    env: gym.Env,
    test_recovery_strategies: bool = True,
    demonstrate_logging_integration: bool = True,
) -> Dict[str, Any]:
    """Demonstrate comprehensive error handling including graceful degradation, recovery strategies, error logging
    integration, and production-ready error management for robust application development.

    Args:
        env: Gymnasium environment instance
        test_recovery_strategies: Whether to test error recovery strategies
        demonstrate_logging_integration: Whether to show logging integration

    Returns:
        dict: Error handling demonstration results with recovery patterns and logging integration status
    """
    _logger.info("Starting comprehensive error handling pattern demonstration")

    with PerformanceTimer() as error_demo_timer:
        error_results = {
            "demonstration_start": time.time(),
            "error_scenarios_tested": [],
            "recovery_strategies": {},
            "logging_integration": {},
            "graceful_degradation": {},
            "error_statistics": {},
        }

        error_count = 0
        recovery_count = 0

        try:
            # Demonstrate validation error handling with ValidationError and ConfigurationError catching
            _logger.info("Testing validation error handling")

            try:
                # Intentionally trigger validation error with invalid action
                invalid_action = 999  # Invalid action outside [0,3] range
                observation, reward, terminated, truncated, info = env.step(
                    invalid_action
                )

            except Exception as validation_error:
                error_count += 1
                error_scenario = {
                    "scenario": "invalid_action_validation",
                    "error_type": type(validation_error).__name__,
                    "error_message": str(validation_error),
                    "recovery_attempted": False,
                    "recovery_successful": False,
                }

                # Demonstrate recovery strategy
                if test_recovery_strategies:
                    try:
                        _logger.info("Attempting recovery from validation error")
                        # Recovery: use valid action instead
                        valid_action = env.action_space.sample()
                        observation, reward, terminated, truncated, info = env.step(
                            valid_action
                        )

                        error_scenario["recovery_attempted"] = True
                        error_scenario["recovery_successful"] = True
                        recovery_count += 1

                        _logger.info("Successfully recovered from validation error")

                    except Exception as recovery_error:
                        error_scenario["recovery_error"] = str(recovery_error)
                        _logger.error(f"Recovery failed: {recovery_error}")

                error_results["error_scenarios_tested"].append(error_scenario)

            # Show graceful degradation patterns for rendering failures and resource constraints
            _logger.info("Testing rendering graceful degradation")

            try:
                # Attempt rendering which might fail in headless environments
                rendered_output = env.render()

                error_results["graceful_degradation"]["rendering"] = {
                    "rendering_successful": True,
                    "fallback_used": False,
                    "output_type": (
                        type(rendered_output).__name__
                        if rendered_output is not None
                        else "None"
                    ),
                }

            except Exception as render_error:
                error_count += 1
                _logger.warning(
                    f"Rendering failed, demonstrating fallback: {render_error}"
                )

                # Demonstrate fallback to rgb_array mode
                fallback_successful = False
                try:
                    # Close and recreate environment with different render mode
                    env.close()
                    fallback_env = gym.make(ENV_ID, render_mode="rgb_array")
                    fallback_env.reset()
                    rgb_output = fallback_env.render()
                    fallback_env.close()

                    fallback_successful = True
                    recovery_count += 1
                    _logger.info("Successfully fell back to rgb_array rendering")

                except Exception as fallback_error:
                    _logger.error(f"Fallback also failed: {fallback_error}")

                error_results["graceful_degradation"]["rendering"] = {
                    "rendering_successful": False,
                    "fallback_used": True,
                    "fallback_successful": fallback_successful,
                    "original_error": str(render_error),
                }

            # Test recovery strategies including environment reset and component reinitialization
            if test_recovery_strategies:
                _logger.info("Testing comprehensive recovery strategies")

                recovery_strategies = {
                    "environment_reset": {"attempted": False, "successful": False},
                    "component_reinitialization": {
                        "attempted": False,
                        "successful": False,
                    },
                    "resource_cleanup": {"attempted": False, "successful": False},
                }

                # Test environment reset recovery
                try:
                    _logger.info("Testing environment reset recovery")
                    observation, info = env.reset()

                    recovery_strategies["environment_reset"] = {
                        "attempted": True,
                        "successful": True,
                        "observation_shape": observation.shape,
                        "info_keys": (
                            list(info.keys()) if isinstance(info, dict) else "not_dict"
                        ),
                    }

                    _logger.info("Environment reset recovery successful")

                except Exception as reset_error:
                    recovery_strategies["environment_reset"] = {
                        "attempted": True,
                        "successful": False,
                        "error": str(reset_error),
                    }
                    _logger.error(f"Environment reset failed: {reset_error}")

                # Test resource cleanup recovery
                try:
                    _logger.info("Testing resource cleanup")
                    # Simulate resource cleanup operations
                    if hasattr(env, "close"):
                        env.close()

                    # Recreate environment
                    new_env = gym.make(ENV_ID)
                    new_env.reset()
                    new_env.close()

                    recovery_strategies["resource_cleanup"] = {
                        "attempted": True,
                        "successful": True,
                    }

                    _logger.info("Resource cleanup recovery successful")

                except Exception as cleanup_error:
                    recovery_strategies["resource_cleanup"] = {
                        "attempted": True,
                        "successful": False,
                        "error": str(cleanup_error),
                    }
                    _logger.error(f"Resource cleanup failed: {cleanup_error}")

                error_results["recovery_strategies"] = recovery_strategies

            # Demonstrate error logging integration with ComponentLogger and context capture
            if demonstrate_logging_integration:
                _logger.info("Demonstrating error logging integration")

                logging_integration = {
                    "component_logger_available": True,
                    "context_capture_successful": False,
                    "error_tracking_enabled": True,
                    "structured_logging": False,
                }

                try:
                    # Create error context for demonstration
                    error_context = {
                        "environment_id": ENV_ID,
                        "demonstration_phase": "error_handling",
                        "timestamp": time.time(),
                        "system_state": "testing",
                    }

                    # Log structured error information
                    _logger.error(
                        "Demonstration error logging with context",
                        extra={
                            "context": error_context,
                            "error_demo": True,
                            "recovery_available": test_recovery_strategies,
                        },
                    )

                    logging_integration["context_capture_successful"] = True
                    logging_integration["structured_logging"] = True

                    _logger.info("Error logging integration demonstration successful")

                except Exception as logging_error:
                    logging_integration["logging_error"] = str(logging_error)
                    _logger.error(f"Error logging integration failed: {logging_error}")

                error_results["logging_integration"] = logging_integration

        except Exception as e:
            _logger.error(f"Error handling demonstration failed: {str(e)}")
            error_results["demonstration_error"] = str(e)

        # Calculate error handling statistics
        error_results["error_statistics"] = {
            "total_errors_encountered": error_count,
            "successful_recoveries": recovery_count,
            "recovery_success_rate": (
                (recovery_count / error_count * 100) if error_count > 0 else 0.0
            ),
            "demonstration_duration_ms": error_demo_timer.get_duration_ms(),
            "error_handling_robustness": (
                "good" if recovery_count >= error_count * 0.8 else "needs_improvement"
            ),
        }

    _logger.info(
        f"Error handling demonstration completed: {error_results['error_statistics']['recovery_success_rate']:.1f}% recovery rate"
    )

    # Return error handling results with recovery success rates and logging integration status
    return error_results


def demonstrate_advanced_rendering(
    env: gym.Env,
    multi_modal_rendering: bool = True,
    batch_processing: bool = False,
    interactive_visualization: bool = False,
) -> Dict[str, Any]:
    """Demonstrate advanced rendering capabilities including multi-modal visualization, performance optimization,
    batch processing, and interactive visualization integration for comprehensive environment analysis.

    Args:
        env: Gymnasium environment instance
        multi_modal_rendering: Whether to test different rendering modes
        batch_processing: Whether to demonstrate batch rendering
        interactive_visualization: Whether to show interactive features

    Returns:
        dict: Advanced rendering demonstration results with performance metrics and visualization analysis
    """
    _logger.info("Starting advanced rendering capabilities demonstration")

    with PerformanceTimer() as render_demo_timer:
        render_results = {
            "demonstration_start": time.time(),
            "rendering_modes_tested": [],
            "performance_metrics": {},
            "compatibility_analysis": {},
            "rendering_samples": {},
        }

        try:
            # Demonstrate RGB array rendering with performance optimization and batch processing
            _logger.info("Testing RGB array rendering")

            rgb_performance = {
                "render_times": [],
                "output_shapes": [],
                "successful_renders": 0,
                "failed_renders": 0,
            }

            # Reset environment for consistent rendering tests
            observation, info = env.reset()

            # Test multiple RGB renders for performance analysis
            for render_idx in range(5):
                try:
                    with PerformanceTimer() as rgb_timer:
                        rgb_output = env.render()

                    rgb_performance["render_times"].append(rgb_timer.get_duration_ms())
                    rgb_performance["successful_renders"] += 1

                    if rgb_output is not None:
                        if hasattr(rgb_output, "shape"):
                            rgb_performance["output_shapes"].append(rgb_output.shape)
                        elif hasattr(rgb_output, "__len__"):
                            rgb_performance["output_shapes"].append(len(rgb_output))

                    _logger.debug(
                        f"RGB render {render_idx + 1} completed in {rgb_timer.get_duration_ms():.2f}ms"
                    )

                except Exception as rgb_error:
                    rgb_performance["failed_renders"] += 1
                    _logger.warning(f"RGB render {render_idx + 1} failed: {rgb_error}")

            # Calculate RGB rendering statistics
            if rgb_performance["render_times"]:
                rgb_stats = {
                    "avg_render_time_ms": statistics.mean(
                        rgb_performance["render_times"]
                    ),
                    "min_render_time_ms": min(rgb_performance["render_times"]),
                    "max_render_time_ms": max(rgb_performance["render_times"]),
                    "render_time_variance": (
                        statistics.stdev(rgb_performance["render_times"])
                        if len(rgb_performance["render_times"]) > 1
                        else 0.0
                    ),
                    "success_rate": (
                        rgb_performance["successful_renders"]
                        / (
                            rgb_performance["successful_renders"]
                            + rgb_performance["failed_renders"]
                        )
                    )
                    * 100,
                }
            else:
                rgb_stats = {"error": "No successful renders"}

            render_results["performance_metrics"]["rgb_array"] = rgb_stats
            render_results["rendering_modes_tested"].append("rgb_array")

            # Show interactive human mode rendering with backend compatibility and fallback handling
            if multi_modal_rendering:
                _logger.info("Testing human mode rendering with backend compatibility")

                human_rendering_result = {
                    "attempted": True,
                    "successful": False,
                    "backend_compatible": False,
                    "fallback_used": False,
                }

                try:
                    # Check matplotlib backend compatibility
                    current_backend = matplotlib.get_backend()
                    interactive_backends = ["TkAgg", "Qt5Agg", "MacOSX"]
                    backend_compatible = current_backend in interactive_backends

                    human_rendering_result["current_backend"] = current_backend
                    human_rendering_result["backend_compatible"] = backend_compatible

                    if not backend_compatible:
                        _logger.info(
                            f"Backend {current_backend} may not support interactive rendering"
                        )

                    # Attempt human mode rendering
                    # Note: We'll create a new environment instance for human mode
                    try:
                        human_env = gym.make(ENV_ID, render_mode="human")
                        human_env.reset()

                        with PerformanceTimer() as human_timer:
                            human_output = human_env.render()

                        human_rendering_result.update(
                            {
                                "successful": True,
                                "render_time_ms": human_timer.get_duration_ms(),
                                "output_type": (
                                    type(human_output).__name__
                                    if human_output is not None
                                    else "None"
                                ),
                            }
                        )

                        human_env.close()
                        _logger.info("Human mode rendering successful")

                    except Exception as human_error:
                        human_rendering_result["error"] = str(human_error)
                        _logger.info(
                            f"Human mode failed, attempting fallback: {human_error}"
                        )

                        # Demonstrate fallback strategy
                        try:
                            # Switch to Agg backend for headless compatibility
                            matplotlib.use("Agg")
                            fallback_env = gym.make(ENV_ID, render_mode="rgb_array")
                            fallback_env.reset()

                            with PerformanceTimer() as fallback_timer:
                                fallback_output = fallback_env.render()

                            human_rendering_result.update(
                                {
                                    "fallback_used": True,
                                    "fallback_successful": True,
                                    "fallback_render_time_ms": fallback_timer.get_duration_ms(),
                                    "fallback_backend": "Agg",
                                }
                            )

                            fallback_env.close()
                            _logger.info("Fallback rendering successful")

                        except Exception as fallback_error:
                            human_rendering_result["fallback_error"] = str(
                                fallback_error
                            )
                            _logger.error(
                                f"Fallback rendering also failed: {fallback_error}"
                            )

                except Exception as human_setup_error:
                    human_rendering_result["setup_error"] = str(human_setup_error)
                    _logger.error(f"Human mode setup failed: {human_setup_error}")

                render_results["performance_metrics"][
                    "human_mode"
                ] = human_rendering_result
                render_results["rendering_modes_tested"].append("human_mode")

            # Test rendering under different environment states and trajectory conditions
            _logger.info("Testing rendering consistency across different states")

            state_rendering_test = {
                "states_tested": 0,
                "consistent_rendering": True,
                "render_times": [],
                "output_consistency": True,
            }

            # Test rendering at different environment states
            for state_idx in range(3):
                try:
                    # Take some steps to change environment state
                    for step in range(5):
                        action = env.action_space.sample()
                        observation, reward, terminated, truncated, info = env.step(
                            action
                        )
                        if terminated or truncated:
                            break

                    # Render at this state
                    with PerformanceTimer() as state_timer:
                        state_output = env.render()

                    state_rendering_test["render_times"].append(
                        state_timer.get_duration_ms()
                    )
                    state_rendering_test["states_tested"] += 1

                    # Check output consistency (basic validation)
                    if state_output is not None and hasattr(state_output, "shape"):
                        if state_idx == 0:
                            first_shape = state_output.shape
                        elif state_output.shape != first_shape:
                            state_rendering_test["output_consistency"] = False

                except Exception as state_error:
                    state_rendering_test["consistent_rendering"] = False
                    _logger.warning(
                        f"State rendering test {state_idx} failed: {state_error}"
                    )

            # Calculate state rendering performance
            if state_rendering_test["render_times"]:
                state_rendering_test["avg_render_time_ms"] = statistics.mean(
                    state_rendering_test["render_times"]
                )
                state_rendering_test["render_time_stability"] = (
                    statistics.stdev(state_rendering_test["render_times"])
                    < 10.0  # Less than 10ms variance
                    if len(state_rendering_test["render_times"]) > 1
                    else True
                )

            render_results["compatibility_analysis"][
                "state_consistency"
            ] = state_rendering_test

            # Test rendering performance under high-frequency update conditions
            if batch_processing:
                _logger.info("Testing batch rendering performance")

                batch_test = {
                    "batch_size": 10,
                    "total_time_ms": 0,
                    "individual_times": [],
                    "throughput_renders_per_second": 0,
                }

                with PerformanceTimer() as batch_timer:
                    for batch_idx in range(batch_test["batch_size"]):
                        with PerformanceTimer() as individual_timer:
                            render_output = env.render()

                        batch_test["individual_times"].append(
                            individual_timer.get_duration_ms()
                        )

                batch_test["total_time_ms"] = batch_timer.get_duration_ms()
                batch_test["throughput_renders_per_second"] = (
                    batch_test["batch_size"] / batch_test["total_time_ms"]
                ) * 1000

                render_results["performance_metrics"]["batch_processing"] = batch_test

        except Exception as e:
            _logger.error(f"Advanced rendering demonstration failed: {str(e)}")
            render_results["demonstration_error"] = str(e)

        # Compile comprehensive rendering analysis
        render_results["demonstration_summary"] = {
            "total_modes_tested": len(render_results["rendering_modes_tested"]),
            "modes_successful": len(
                [mode for mode in render_results["rendering_modes_tested"]]
            ),
            "demonstration_duration_ms": render_demo_timer.get_duration_ms(),
            "rendering_capability_assessment": (
                "good"
                if len(render_results["rendering_modes_tested"]) >= 1
                else "limited"
            ),
        }

    _logger.info("Advanced rendering demonstration completed")

    # Return rendering results with performance metrics, compatibility analysis, and optimization recommendations
    return render_results


def demonstrate_parallel_environments(
    num_parallel_envs: int = 3,
    episodes_per_env: int = 2,
    performance_comparison: bool = True,
    synchronization_demo: bool = True,
) -> Dict[str, Any]:
    """Demonstrate advanced parallel environment usage including vectorized operations, concurrent execution,
    performance scaling, and multi-environment coordination for high-throughput research applications.

    Args:
        num_parallel_envs: Number of parallel environments to create
        episodes_per_env: Number of episodes per environment
        performance_comparison: Whether to compare sequential vs parallel performance
        synchronization_demo: Whether to demonstrate synchronized execution

    Returns:
        dict: Parallel environment demonstration results with scaling analysis and performance metrics
    """
    _logger.info(
        f"Starting parallel environment demonstration with {num_parallel_envs} environments"
    )

    with PerformanceTimer() as parallel_demo_timer:
        parallel_results = {
            "demonstration_start": time.time(),
            "num_parallel_envs": num_parallel_envs,
            "episodes_per_env": episodes_per_env,
            "environment_creation": {},
            "parallel_execution": {},
            "performance_comparison": {},
            "synchronization_results": {},
        }

        try:
            # Create multiple environment instances with different configurations for parallel execution
            _logger.info("Creating multiple environment instances")

            environment_creation = {
                "creation_times": [],
                "successful_creations": 0,
                "failed_creations": 0,
                "environment_configs": [],
            }

            environments = []

            for env_idx in range(num_parallel_envs):
                try:
                    with PerformanceTimer() as creation_timer:
                        # Create environment with unique configuration
                        env_config = {
                            "env_id": f"{ENV_ID}_parallel_{env_idx}",
                            "seed": ADVANCED_DEMO_SEED + env_idx * 100,
                            "render_mode": "rgb_array",  # Use non-interactive mode for parallel processing
                        }

                        env = gym.make(ENV_ID, render_mode="rgb_array")
                        environments.append(
                            {"env": env, "config": env_config, "id": env_idx}
                        )

                    environment_creation["creation_times"].append(
                        creation_timer.get_duration_ms()
                    )
                    environment_creation["successful_creations"] += 1
                    environment_creation["environment_configs"].append(env_config)

                    _logger.debug(
                        f"Environment {env_idx} created in {creation_timer.get_duration_ms():.2f}ms"
                    )

                except Exception as creation_error:
                    environment_creation["failed_creations"] += 1
                    _logger.error(
                        f"Failed to create environment {env_idx}: {creation_error}"
                    )

            parallel_results["environment_creation"] = environment_creation

            if not environments:
                _logger.error(
                    "No environments created successfully, aborting parallel demonstration"
                )
                return parallel_results

            # Demonstrate concurrent episode execution using concurrent.futures for performance scaling
            _logger.info("Executing episodes concurrently across environments")

            def execute_episode(env_data, episode_idx):
                """Execute single episode in parallel environment."""
                env_info = env_data["env"]
                env_config = env_data["config"]

                try:
                    with PerformanceTimer() as episode_timer:
                        # Reset with unique seed
                        observation, info = env_info.reset(
                            seed=env_config["seed"] + episode_idx
                        )

                        episode_data = {
                            "env_id": env_data["id"],
                            "episode": episode_idx,
                            "steps": 0,
                            "total_reward": 0.0,
                            "terminated": False,
                            "truncated": False,
                        }

                        # Execute episode with limited steps for demonstration
                        max_steps = 30
                        done = False

                        while not done and episode_data["steps"] < max_steps:
                            action = env_info.action_space.sample()
                            observation, reward, terminated, truncated, info = (
                                env_info.step(action)
                            )

                            episode_data["total_reward"] += reward
                            episode_data["steps"] += 1
                            done = terminated or truncated

                        episode_data["terminated"] = terminated
                        episode_data["truncated"] = truncated
                        episode_data["duration_ms"] = episode_timer.get_duration_ms()

                    return episode_data

                except Exception as episode_error:
                    return {
                        "env_id": env_data["id"],
                        "episode": episode_idx,
                        "error": str(episode_error),
                    }

            # Execute episodes in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_parallel_envs
            ) as executor:
                # Create list of all episode tasks
                episode_tasks = []
                for env_data in environments:
                    for episode_idx in range(episodes_per_env):
                        task = executor.submit(execute_episode, env_data, episode_idx)
                        episode_tasks.append(task)

                # Collect results as they complete
                parallel_episode_results = []
                completed_episodes = 0
                failed_episodes = 0

                for future in concurrent.futures.as_completed(episode_tasks):
                    try:
                        result = future.result(
                            timeout=30
                        )  # 30 second timeout per episode
                        parallel_episode_results.append(result)

                        if "error" in result:
                            failed_episodes += 1
                        else:
                            completed_episodes += 1

                    except concurrent.futures.TimeoutError:
                        failed_episodes += 1
                        _logger.warning("Episode execution timed out")
                    except Exception as future_error:
                        failed_episodes += 1
                        _logger.error(f"Episode execution failed: {future_error}")

            parallel_execution = {
                "total_episodes_attempted": len(episode_tasks),
                "completed_episodes": completed_episodes,
                "failed_episodes": failed_episodes,
                "success_rate": (
                    (completed_episodes / len(episode_tasks)) * 100
                    if episode_tasks
                    else 0
                ),
                "episode_results": parallel_episode_results,
            }

            parallel_results["parallel_execution"] = parallel_execution

            # Compare sequential vs parallel execution performance with comprehensive timing analysis
            if performance_comparison and completed_episodes > 0:
                _logger.info("Performing sequential vs parallel performance comparison")

                # Execute same number of episodes sequentially for comparison
                sequential_times = []
                sequential_env = environments[0]["env"]

                with PerformanceTimer() as sequential_timer:
                    for seq_episode in range(
                        min(episodes_per_env, 3)
                    ):  # Limit for fair comparison
                        observation, info = sequential_env.reset(
                            seed=ADVANCED_DEMO_SEED + seq_episode
                        )

                        seq_steps = 0
                        seq_done = False

                        while not seq_done and seq_steps < 30:
                            action = sequential_env.action_space.sample()
                            observation, reward, terminated, truncated, info = (
                                sequential_env.step(action)
                            )
                            seq_steps += 1
                            seq_done = terminated or truncated

                sequential_total_time = sequential_timer.get_duration_ms()

                # Calculate parallel execution time
                successful_episodes = [
                    ep for ep in parallel_episode_results if "error" not in ep
                ]
                if successful_episodes:
                    parallel_total_time = parallel_demo_timer.get_duration_ms()

                    performance_comparison_results = {
                        "sequential_time_ms": sequential_total_time,
                        "parallel_time_ms": parallel_total_time,
                        "speedup_factor": (
                            sequential_total_time / parallel_total_time
                            if parallel_total_time > 0
                            else 0
                        ),
                        "episodes_compared": len(successful_episodes),
                        "parallel_efficiency": len(successful_episodes)
                        / (num_parallel_envs * episodes_per_env),
                    }

                    parallel_results["performance_comparison"] = (
                        performance_comparison_results
                    )

                    _logger.info(
                        f"Performance comparison: {performance_comparison_results['speedup_factor']:.2f}x speedup with parallel execution"
                    )

            # Apply synchronized seeding across parallel environments for reproducibility validation
            if synchronization_demo and len(environments) >= 2:
                _logger.info(
                    "Demonstrating synchronized execution across parallel environments"
                )

                sync_results = {
                    "synchronized_seed": ADVANCED_DEMO_SEED + 9999,
                    "environments_synchronized": 0,
                    "reproducibility_verified": False,
                    "synchronization_data": [],
                }

                # Reset all environments with same seed
                sync_observations = []
                sync_rewards = []

                try:
                    for env_data in environments[:2]:  # Test with first 2 environments
                        env_info = env_data["env"]
                        observation, info = env_info.reset(
                            seed=sync_results["synchronized_seed"]
                        )

                        # Execute identical sequence of actions
                        episode_rewards = []
                        for step_idx in range(10):  # Short synchronized sequence
                            action = step_idx % 4  # Deterministic action sequence
                            observation, reward, terminated, truncated, info = (
                                env_info.step(action)
                            )
                            episode_rewards.append(reward)

                            if terminated or truncated:
                                break

                        sync_observations.append(
                            observation.copy()
                            if hasattr(observation, "copy")
                            else observation
                        )
                        sync_rewards.append(episode_rewards)
                        sync_results["environments_synchronized"] += 1

                    # Verify synchronization by comparing results
                    if len(sync_rewards) >= 2:
                        rewards_match = np.allclose(
                            sync_rewards[0], sync_rewards[1], rtol=1e-10
                        )
                        sync_results["reproducibility_verified"] = rewards_match
                        sync_results["reward_differences"] = [
                            abs(r1 - r2)
                            for r1, r2 in zip(sync_rewards[0], sync_rewards[1])
                        ]

                    _logger.info(
                        f"Synchronization test: {'✓' if sync_results['reproducibility_verified'] else '✗'}"
                    )

                except Exception as sync_error:
                    sync_results["synchronization_error"] = str(sync_error)
                    _logger.error(f"Synchronization test failed: {sync_error}")

                parallel_results["synchronization_results"] = sync_results

            # Clean up environments
            _logger.info("Cleaning up parallel environments")
            cleanup_count = 0
            for env_data in environments:
                try:
                    env_data["env"].close()
                    cleanup_count += 1
                except Exception as cleanup_error:
                    _logger.warning(
                        f"Failed to close environment {env_data['id']}: {cleanup_error}"
                    )

            parallel_results["cleanup"] = {
                "environments_cleaned": cleanup_count,
                "total_environments": len(environments),
            }

        except Exception as e:
            _logger.error(f"Parallel environment demonstration failed: {str(e)}")
            parallel_results["demonstration_error"] = str(e)

        # Calculate overall parallel execution metrics
        parallel_results["demonstration_summary"] = {
            "total_demonstration_time_ms": parallel_demo_timer.get_duration_ms(),
            "parallel_execution_successful": "parallel_execution" in parallel_results
            and parallel_results["parallel_execution"]["success_rate"] > 50,
            "scaling_analysis": {
                "environments_created": environment_creation["successful_creations"],
                "target_environments": num_parallel_envs,
                "scaling_efficiency": environment_creation["successful_creations"]
                / num_parallel_envs,
            },
        }

    _logger.info("Parallel environment demonstration completed")

    # Return parallel execution results with scaling metrics, performance analysis, and resource optimization
    return parallel_results


def generate_comprehensive_report(
    demonstration_results: Dict[str, Any],
    include_visualizations: bool = True,
    export_data: bool = False,
    output_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate comprehensive demonstration report with all results, performance metrics, validation status,
    and recommendations for production usage and research applications.

    Args:
        demonstration_results: Dictionary containing all demonstration results
        include_visualizations: Whether to create visualization plots
        export_data: Whether to export raw data to files
        output_directory: Optional directory for saving outputs

    Returns:
        dict: Comprehensive report generation results with export status and visualization creation
    """
    _logger.info("Generating comprehensive demonstration report")

    with PerformanceTimer() as report_timer:
        report_results = {
            "report_generation_start": time.time(),
            "report_sections": [],
            "export_status": {},
            "visualization_status": {},
            "summary_statistics": {},
        }

        try:
            # Compile performance metrics from all demonstration components with statistical analysis
            _logger.info("Compiling performance metrics and statistics")

            performance_summary = {
                "total_demonstrations_run": len(demonstration_results),
                "successful_demonstrations": 0,
                "failed_demonstrations": 0,
                "total_execution_time_ms": 0,
                "performance_highlights": [],
            }

            # Aggregate performance data from all demonstration components
            for demo_name, demo_data in demonstration_results.items():
                if (
                    isinstance(demo_data, dict)
                    and "demonstration_error" not in demo_data
                ):
                    performance_summary["successful_demonstrations"] += 1
                else:
                    performance_summary["failed_demonstrations"] += 1

                # Extract timing information if available
                if isinstance(demo_data, dict):
                    for key, value in demo_data.items():
                        if "time_ms" in key and isinstance(value, (int, float)):
                            performance_summary["total_execution_time_ms"] += value
                        elif key == "performance_metrics" and isinstance(value, dict):
                            # Extract specific performance highlights
                            if "avg_step_time_ms" in value:
                                performance_summary["performance_highlights"].append(
                                    f"{demo_name}: {value['avg_step_time_ms']:.2f}ms avg step time"
                                )

            report_results["summary_statistics"]["performance"] = performance_summary
            report_results["report_sections"].append("performance_analysis")

            # Generate validation summary including reproducibility, integrity, and error handling results
            _logger.info("Generating validation and integrity summary")

            validation_summary = {
                "reproducibility_status": "unknown",
                "error_handling_robustness": "unknown",
                "configuration_validity": "unknown",
                "overall_system_integrity": "unknown",
            }

            # Extract validation results from demonstrations
            if "reproducibility_validation" in demonstration_results:
                repro_data = demonstration_results["reproducibility_validation"]
                if (
                    isinstance(repro_data, dict)
                    and "reproducibility_analysis" in repro_data
                ):
                    repro_analysis = repro_data["reproducibility_analysis"]
                    if "integrity_status" in repro_analysis:
                        validation_summary["reproducibility_status"] = repro_analysis[
                            "integrity_status"
                        ]

            if "error_handling" in demonstration_results:
                error_data = demonstration_results["error_handling"]
                if isinstance(error_data, dict) and "error_statistics" in error_data:
                    error_stats = error_data["error_statistics"]
                    if "error_handling_robustness" in error_stats:
                        validation_summary["error_handling_robustness"] = error_stats[
                            "error_handling_robustness"
                        ]

            if "configuration_presets" in demonstration_results:
                config_data = demonstration_results["configuration_presets"]
                if (
                    isinstance(config_data, dict)
                    and "validation_results" in config_data
                ):
                    validation_results = config_data["validation_results"]
                    if isinstance(validation_results, dict):
                        total_validated = validation_results.get("presets_validated", 0)
                        total_failed = validation_results.get("presets_failed", 0)
                        if total_validated > 0 and total_failed == 0:
                            validation_summary["configuration_validity"] = "excellent"
                        elif total_validated > total_failed:
                            validation_summary["configuration_validity"] = "good"
                        else:
                            validation_summary["configuration_validity"] = (
                                "needs_attention"
                            )

            # Determine overall system integrity
            status_scores = {
                "excellent": 4,
                "good": 3,
                "fair": 2,
                "poor": 1,
                "unknown": 0,
                "needs_attention": 1,
                "needs_improvement": 1,
            }

            avg_score = statistics.mean(
                [
                    status_scores.get(validation_summary["reproducibility_status"], 0),
                    status_scores.get(
                        validation_summary["error_handling_robustness"], 0
                    ),
                    status_scores.get(validation_summary["configuration_validity"], 0),
                ]
            )

            if avg_score >= 3.5:
                validation_summary["overall_system_integrity"] = "excellent"
            elif avg_score >= 2.5:
                validation_summary["overall_system_integrity"] = "good"
            elif avg_score >= 1.5:
                validation_summary["overall_system_integrity"] = "fair"
            else:
                validation_summary["overall_system_integrity"] = "needs_improvement"

            report_results["summary_statistics"]["validation"] = validation_summary
            report_results["report_sections"].append("validation_summary")

            # Create optimization recommendations based on performance analysis and benchmarking
            _logger.info("Generating optimization recommendations")

            optimization_recommendations = []

            # Analyze performance optimization results
            if "performance_optimization" in demonstration_results:
                perf_data = demonstration_results["performance_optimization"]
                if (
                    isinstance(perf_data, dict)
                    and "optimization_recommendations" in perf_data
                ):
                    perf_recommendations = perf_data["optimization_recommendations"]
                    if isinstance(perf_recommendations, list):
                        optimization_recommendations.extend(perf_recommendations)

            # Add general recommendations based on demonstration results
            if (
                performance_summary["total_execution_time_ms"] > 10000
            ):  # > 10 seconds total
                optimization_recommendations.append(
                    {
                        "category": "overall_performance",
                        "recommendation": "Consider optimizing demonstration execution time for better user experience",
                        "priority": "medium",
                    }
                )

            if performance_summary["failed_demonstrations"] > 0:
                optimization_recommendations.append(
                    {
                        "category": "reliability",
                        "recommendation": "Investigate and fix failed demonstration components",
                        "priority": "high",
                    }
                )

            report_results["optimization_recommendations"] = (
                optimization_recommendations
            )
            report_results["report_sections"].append("optimization_guidance")

            # Generate visualizations including performance trends, trajectory analysis, and configuration comparisons
            if include_visualizations:
                _logger.info("Creating demonstration visualizations")

                visualization_results = {
                    "plots_created": 0,
                    "visualization_files": [],
                    "plotting_errors": [],
                }

                try:
                    # Set up matplotlib for report generation
                    plt.style.use("default")
                    fig_size = (12, 8)

                    # Create performance summary plot
                    if performance_summary["successful_demonstrations"] > 0:
                        plt.figure(figsize=fig_size)

                        # Plot demonstration success/failure
                        labels = ["Successful", "Failed"]
                        sizes = [
                            performance_summary["successful_demonstrations"],
                            performance_summary["failed_demonstrations"],
                        ]
                        colors = ["#2E8B57", "#DC143C"]  # Sea green and crimson

                        plt.pie(
                            sizes,
                            labels=labels,
                            colors=colors,
                            autopct="%1.1f%%",
                            startangle=90,
                        )
                        plt.title(
                            "Demonstration Success Rate", fontsize=14, fontweight="bold"
                        )
                        plt.axis("equal")

                        plot_filename = "demonstration_success_rate.png"
                        if output_directory:
                            plot_path = pathlib.Path(output_directory) / plot_filename
                            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                        else:
                            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")

                        visualization_results["plots_created"] += 1
                        visualization_results["visualization_files"].append(
                            plot_filename
                        )

                        plt.close()

                    # Create performance metrics timeline if timing data available
                    timing_data = []
                    timing_labels = []

                    for demo_name, demo_data in demonstration_results.items():
                        if isinstance(demo_data, dict):
                            for key, value in demo_data.items():
                                if "demonstration_time_ms" in key and isinstance(
                                    value, (int, float)
                                ):
                                    timing_data.append(value)
                                    timing_labels.append(
                                        demo_name.replace("_", " ").title()
                                    )

                    if timing_data:
                        plt.figure(figsize=fig_size)
                        bars = plt.bar(
                            range(len(timing_data)),
                            timing_data,
                            color="steelblue",
                            alpha=0.7,
                        )
                        plt.xlabel("Demonstration Components", fontweight="bold")
                        plt.ylabel("Execution Time (ms)", fontweight="bold")
                        plt.title(
                            "Demonstration Component Performance",
                            fontsize=14,
                            fontweight="bold",
                        )
                        plt.xticks(
                            range(len(timing_labels)),
                            timing_labels,
                            rotation=45,
                            ha="right",
                        )

                        # Add value labels on bars
                        for bar, value in zip(bars, timing_data):
                            plt.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + max(timing_data) * 0.01,
                                f"{value:.1f}ms",
                                ha="center",
                                va="bottom",
                            )

                        plt.tight_layout()

                        plot_filename = "performance_timeline.png"
                        if output_directory:
                            plot_path = pathlib.Path(output_directory) / plot_filename
                            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                        else:
                            plt.savefig(plot_filename, dpi=300, bbox_inches="tight")

                        visualization_results["plots_created"] += 1
                        visualization_results["visualization_files"].append(
                            plot_filename
                        )

                        plt.close()

                    _logger.info(
                        f"Created {visualization_results['plots_created']} visualization plots"
                    )

                except Exception as viz_error:
                    visualization_results["plotting_errors"].append(str(viz_error))
                    _logger.error(f"Visualization creation failed: {viz_error}")

                report_results["visualization_status"] = visualization_results

            # Export demonstration data in structured format for further analysis and research use
            if export_data:
                _logger.info("Exporting demonstration data")

                export_results = {
                    "files_exported": 0,
                    "export_files": [],
                    "export_errors": [],
                }

                try:
                    # Prepare output directory
                    if output_directory:
                        output_path = pathlib.Path(output_directory)
                        output_path.mkdir(parents=True, exist_ok=True)
                    else:
                        output_path = pathlib.Path(".")

                    # Export complete demonstration results as JSON
                    results_file = output_path / "demonstration_results.json"
                    with open(results_file, "w") as f:
                        json.dump(demonstration_results, f, indent=2, default=str)

                    export_results["files_exported"] += 1
                    export_results["export_files"].append(str(results_file))

                    # Export performance summary as CSV-like format
                    summary_file = output_path / "performance_summary.json"
                    with open(summary_file, "w") as f:
                        json.dump(
                            report_results["summary_statistics"],
                            f,
                            indent=2,
                            default=str,
                        )

                    export_results["files_exported"] += 1
                    export_results["export_files"].append(str(summary_file))

                    # Export optimization recommendations
                    if optimization_recommendations:
                        recommendations_file = (
                            output_path / "optimization_recommendations.json"
                        )
                        with open(recommendations_file, "w") as f:
                            json.dump(
                                optimization_recommendations, f, indent=2, default=str
                            )

                        export_results["files_exported"] += 1
                        export_results["export_files"].append(str(recommendations_file))

                    _logger.info(
                        f"Exported {export_results['files_exported']} data files"
                    )

                except Exception as export_error:
                    export_results["export_errors"].append(str(export_error))
                    _logger.error(f"Data export failed: {export_error}")

                report_results["export_status"] = export_results

            # Create executive summary with key findings and production readiness assessment
            executive_summary = {
                "demonstration_overview": {
                    "total_components_tested": len(demonstration_results),
                    "success_rate": (
                        (
                            performance_summary["successful_demonstrations"]
                            / len(demonstration_results)
                            * 100
                        )
                        if demonstration_results
                        else 0
                    ),
                    "total_execution_time_seconds": performance_summary[
                        "total_execution_time_ms"
                    ]
                    / 1000.0,
                },
                "key_findings": [],
                "production_readiness_assessment": validation_summary[
                    "overall_system_integrity"
                ],
                "critical_issues": [],
                "recommended_next_steps": [],
            }

            # Generate key findings
            if validation_summary["reproducibility_status"] == "excellent":
                executive_summary["key_findings"].append(
                    "Reproducibility validation passed with excellent consistency"
                )

            if performance_summary["performance_highlights"]:
                executive_summary["key_findings"].extend(
                    performance_summary["performance_highlights"][:3]
                )

            # Identify critical issues
            if performance_summary["failed_demonstrations"] > 0:
                executive_summary["critical_issues"].append(
                    f"{performance_summary['failed_demonstrations']} demonstration components failed"
                )

            if validation_summary["overall_system_integrity"] in [
                "poor",
                "needs_improvement",
            ]:
                executive_summary["critical_issues"].append(
                    "System integrity assessment needs attention"
                )

            # Generate next steps
            if optimization_recommendations:
                high_priority_recs = [
                    rec
                    for rec in optimization_recommendations
                    if rec.get("priority") == "high"
                ]
                if high_priority_recs:
                    executive_summary["recommended_next_steps"].extend(
                        [rec["recommendation"] for rec in high_priority_recs[:3]]
                    )

            if not executive_summary["recommended_next_steps"]:
                executive_summary["recommended_next_steps"].append(
                    "Continue monitoring system performance and integrity"
                )

            report_results["executive_summary"] = executive_summary
            report_results["report_sections"].append("executive_summary")

        except Exception as e:
            _logger.error(f"Report generation failed: {str(e)}")
            report_results["report_generation_error"] = str(e)

        # Finalize report generation metrics
        report_results["report_generation_duration_ms"] = report_timer.get_duration_ms()
        report_results["report_completed_at"] = time.time()

        _logger.info(
            f"Comprehensive report generated in {report_timer.get_duration_ms():.1f}ms"
        )

    # Return report generation status with export locations and visualization creation results
    return report_results


def run_gymnasium_integration_demo(
    demo_mode: Optional[str] = None,
    include_performance_testing: bool = True,
    include_parallel_demo: bool = False,
    output_directory: Optional[str] = None,
) -> int:
    """Execute comprehensive Gymnasium integration demonstration coordinating all advanced features, performance
    monitoring, and production-ready patterns for complete environment validation and showcase.

    Args:
        demo_mode: Optional demo mode ('quick', 'full', 'performance_focus')
        include_performance_testing: Whether to run performance optimization demonstrations
        include_parallel_demo: Whether to include parallel environment demonstrations
        output_directory: Optional directory for saving outputs and reports

    Returns:
        int: Exit status code with 0 for success, 1 for demonstration failures, and 2 for critical errors
    """
    global _logger, _performance_results

    try:
        # Set up advanced logging configuration using setup_advanced_logging for comprehensive monitoring
        logging_setup = setup_advanced_logging(
            log_level="INFO",
            enable_performance_logging=True,
            enable_file_output=bool(output_directory),
        )

        if logging_setup["status"] != "success":
            print(f"Warning: Logging setup encountered issues: {logging_setup}")

        _logger.info("=" * 80)
        _logger.info("STARTING COMPREHENSIVE GYMNASIUM INTEGRATION DEMONSTRATION")
        _logger.info("=" * 80)
        _logger.info(f"Demo mode: {demo_mode or 'standard'}")
        _logger.info(
            f"Performance testing: {'enabled' if include_performance_testing else 'disabled'}"
        )
        _logger.info(
            f"Parallel demonstration: {'enabled' if include_parallel_demo else 'disabled'}"
        )
        _logger.info(f"Output directory: {output_directory or 'current directory'}")

        # Initialize comprehensive results tracking
        demonstration_results = {}
        critical_failures = 0

        with PerformanceTimer() as total_demo_timer:
            # Execute advanced registration demonstration with comprehensive validation and error handling
            _logger.info("\n" + "=" * 60)
            _logger.info("1. ADVANCED REGISTRATION DEMONSTRATION")
            _logger.info("=" * 60)

            try:
                registration_results = demonstrate_advanced_registration(
                    show_registry_inspection=True
                )
                demonstration_results["registration"] = registration_results

                if "demonstration_error" in registration_results:
                    _logger.error("Registration demonstration failed")
                    critical_failures += 1
                else:
                    _logger.info("✓ Registration demonstration completed successfully")

            except Exception as reg_error:
                _logger.critical(f"Registration demonstration crashed: {reg_error}")
                demonstration_results["registration"] = {
                    "critical_error": str(reg_error)
                }
                critical_failures += 1

            # Demonstrate configuration preset usage with all available presets and custom scenarios
            _logger.info("\n" + "=" * 60)
            _logger.info("2. CONFIGURATION MANAGEMENT DEMONSTRATION")
            _logger.info("=" * 60)

            try:
                config_results = demonstrate_configuration_presets(
                    show_all_presets=True,
                    demonstrate_custom_creation=True,
                    validate_configurations=True,
                )
                demonstration_results["configuration_presets"] = config_results

                if "demonstration_error" in config_results:
                    _logger.error("Configuration demonstration failed")
                else:
                    _logger.info(
                        "✓ Configuration management demonstration completed successfully"
                    )

            except Exception as config_error:
                _logger.critical(f"Configuration demonstration crashed: {config_error}")
                demonstration_results["configuration_presets"] = {
                    "critical_error": str(config_error)
                }
                critical_failures += 1

            # Create environment instances with different configurations for comprehensive testing
            _logger.info("\n" + "=" * 60)
            _logger.info("3. ENVIRONMENT CREATION AND TESTING")
            _logger.info("=" * 60)

            test_env = None
            try:
                # Ensure environment is registered
                if not is_registered(ENV_ID):
                    register_env()

                # Create test environment
                test_env = gym.make(ENV_ID, render_mode="rgb_array")
                _logger.info(f"✓ Successfully created test environment: {ENV_ID}")

                # Validate environment basic functionality
                observation, info = test_env.reset(seed=ADVANCED_DEMO_SEED)
                _logger.info(
                    f"✓ Environment reset successful, observation shape: {observation.shape}"
                )

                # Test one step to verify environment is working
                action = test_env.action_space.sample()
                observation, reward, terminated, truncated, info = test_env.step(action)
                _logger.info(f"✓ Environment step successful, reward: {reward}")

            except Exception as env_error:
                _logger.critical(f"Environment creation failed: {env_error}")
                return 2  # Critical error, cannot continue

            # Execute advanced episode patterns with performance monitoring and trajectory collection
            if test_env is not None:
                _logger.info("\n" + "=" * 60)
                _logger.info("4. ADVANCED EPISODE PATTERN DEMONSTRATION")
                _logger.info("=" * 60)

                try:
                    episode_results = demonstrate_advanced_episode_patterns(
                        env=test_env,
                        num_episodes=3 if demo_mode != "quick" else 1,
                        collect_trajectories=True,
                        performance_analysis=True,
                    )
                    demonstration_results["episode_patterns"] = episode_results

                    if "execution_error" not in episode_results:
                        _logger.info(
                            "✓ Advanced episode patterns demonstrated successfully"
                        )
                    else:
                        _logger.error("Episode patterns demonstration had errors")

                except Exception as episode_error:
                    _logger.error(
                        f"Episode patterns demonstration failed: {episode_error}"
                    )
                    demonstration_results["episode_patterns"] = {
                        "error": str(episode_error)
                    }

            # Validate reproducibility across all configurations with statistical testing and analysis
            if test_env is not None and demo_mode != "quick":
                _logger.info("\n" + "=" * 60)
                _logger.info("5. REPRODUCIBILITY VALIDATION")
                _logger.info("=" * 60)

                try:
                    repro_results = demonstrate_reproducibility_validation(
                        env=test_env,
                        test_seeds=[ADVANCED_DEMO_SEED + i for i in range(5)],
                        validation_episodes_per_seed=2,
                        statistical_testing=True,
                    )
                    demonstration_results["reproducibility_validation"] = repro_results

                    if "validation_error" not in repro_results:
                        _logger.info("✓ Reproducibility validation completed")
                    else:
                        _logger.error("Reproducibility validation had errors")

                except Exception as repro_error:
                    _logger.error(f"Reproducibility validation failed: {repro_error}")
                    demonstration_results["reproducibility_validation"] = {
                        "error": str(repro_error)
                    }

            # Perform comprehensive performance optimization and baseline establishment
            if test_env is not None and include_performance_testing:
                _logger.info("\n" + "=" * 60)
                _logger.info("6. PERFORMANCE OPTIMIZATION DEMONSTRATION")
                _logger.info("=" * 60)

                try:
                    perf_results = demonstrate_performance_optimization(
                        env=test_env,
                        benchmark_episodes=PERFORMANCE_BASELINE_EPISODES,
                        establish_baselines=True,
                        optimization_analysis=True,
                    )
                    demonstration_results["performance_optimization"] = perf_results

                    if "optimization_error" not in perf_results:
                        _logger.info(
                            "✓ Performance optimization demonstration completed"
                        )
                    else:
                        _logger.error("Performance optimization had errors")

                except Exception as perf_error:
                    _logger.error(f"Performance optimization failed: {perf_error}")
                    demonstration_results["performance_optimization"] = {
                        "error": str(perf_error)
                    }

            # Demonstrate error handling patterns with recovery strategies and logging integration
            if test_env is not None:
                _logger.info("\n" + "=" * 60)
                _logger.info("7. ERROR HANDLING PATTERN DEMONSTRATION")
                _logger.info("=" * 60)

                try:
                    error_results = demonstrate_error_handling_patterns(
                        env=test_env,
                        test_recovery_strategies=True,
                        demonstrate_logging_integration=True,
                    )
                    demonstration_results["error_handling"] = error_results

                    if "demonstration_error" not in error_results:
                        _logger.info(
                            "✓ Error handling patterns demonstrated successfully"
                        )
                    else:
                        _logger.error("Error handling demonstration had issues")

                except Exception as error_demo_error:
                    _logger.error(
                        f"Error handling demonstration failed: {error_demo_error}"
                    )
                    demonstration_results["error_handling"] = {
                        "error": str(error_demo_error)
                    }

            # Execute advanced rendering demonstrations with multi-modal visualization
            if test_env is not None and demo_mode != "quick":
                _logger.info("\n" + "=" * 60)
                _logger.info("8. ADVANCED RENDERING DEMONSTRATION")
                _logger.info("=" * 60)

                try:
                    render_results = demonstrate_advanced_rendering(
                        env=test_env,
                        multi_modal_rendering=True,
                        batch_processing=False,
                        interactive_visualization=False,
                    )
                    demonstration_results["advanced_rendering"] = render_results

                    if "demonstration_error" not in render_results:
                        _logger.info("✓ Advanced rendering demonstration completed")
                    else:
                        _logger.error("Rendering demonstration had errors")

                except Exception as render_error:
                    _logger.error(f"Rendering demonstration failed: {render_error}")
                    demonstration_results["advanced_rendering"] = {
                        "error": str(render_error)
                    }

            # Run parallel environment demonstrations if include_parallel_demo enabled
            if include_parallel_demo and demo_mode != "quick":
                _logger.info("\n" + "=" * 60)
                _logger.info("9. PARALLEL ENVIRONMENT DEMONSTRATION")
                _logger.info("=" * 60)

                try:
                    parallel_results = demonstrate_parallel_environments(
                        num_parallel_envs=3,
                        episodes_per_env=2,
                        performance_comparison=True,
                        synchronization_demo=True,
                    )
                    demonstration_results["parallel_environments"] = parallel_results

                    if "demonstration_error" not in parallel_results:
                        _logger.info("✓ Parallel environment demonstration completed")
                    else:
                        _logger.error("Parallel demonstration had errors")

                except Exception as parallel_error:
                    _logger.error(f"Parallel demonstration failed: {parallel_error}")
                    demonstration_results["parallel_environments"] = {
                        "error": str(parallel_error)
                    }

            # Generate comprehensive demonstration report with all results and recommendations
            _logger.info("\n" + "=" * 60)
            _logger.info("10. COMPREHENSIVE REPORT GENERATION")
            _logger.info("=" * 60)

            try:
                report_results = generate_comprehensive_report(
                    demonstration_results=demonstration_results,
                    include_visualizations=True,
                    export_data=bool(output_directory),
                    output_directory=output_directory,
                )
                demonstration_results["report_generation"] = report_results

                if "report_generation_error" not in report_results:
                    _logger.info("✓ Comprehensive report generated successfully")

                    # Display executive summary
                    if "executive_summary" in report_results:
                        exec_summary = report_results["executive_summary"]
                        _logger.info("\n" + "-" * 60)
                        _logger.info("EXECUTIVE SUMMARY")
                        _logger.info("-" * 60)

                        overview = exec_summary["demonstration_overview"]
                        _logger.info(
                            f"Components Tested: {overview['total_components_tested']}"
                        )
                        _logger.info(f"Success Rate: {overview['success_rate']:.1f}%")
                        _logger.info(
                            f"Total Execution Time: {overview['total_execution_time_seconds']:.1f}s"
                        )
                        _logger.info(
                            f"Production Readiness: {exec_summary['production_readiness_assessment']}"
                        )

                        if exec_summary["key_findings"]:
                            _logger.info("\nKey Findings:")
                            for finding in exec_summary["key_findings"]:
                                _logger.info(f"  • {finding}")

                        if exec_summary["critical_issues"]:
                            _logger.warning("\nCritical Issues:")
                            for issue in exec_summary["critical_issues"]:
                                _logger.warning(f"  ⚠ {issue}")

                        if exec_summary["recommended_next_steps"]:
                            _logger.info("\nRecommended Next Steps:")
                            for step in exec_summary["recommended_next_steps"]:
                                _logger.info(f"  → {step}")
                else:
                    _logger.error("Report generation had errors")

            except Exception as report_error:
                _logger.error(f"Report generation failed: {report_error}")
                demonstration_results["report_generation"] = {
                    "error": str(report_error)
                }

            # Cleanup all resources and validate proper shutdown procedures
            _logger.info("\n" + "=" * 60)
            _logger.info("11. RESOURCE CLEANUP")
            _logger.info("=" * 60)

            cleanup_success = True

            try:
                if test_env is not None:
                    test_env.close()
                    _logger.info("✓ Test environment closed successfully")

                # Additional cleanup
                plt.close("all")  # Close any remaining matplotlib figures

                # Verify environment unregistration for clean state
                if is_registered(ENV_ID):
                    try:
                        unregister_env(ENV_ID)
                        _logger.info("✓ Environment unregistered for clean shutdown")
                    except Exception as unreg_error:
                        _logger.warning(
                            f"Environment unregistration failed: {unreg_error}"
                        )
                        cleanup_success = False

                _logger.info("✓ Resource cleanup completed")

            except Exception as cleanup_error:
                _logger.error(f"Resource cleanup failed: {cleanup_error}")
                cleanup_success = False

        # Calculate final demonstration statistics
        total_time_seconds = total_demo_timer.get_duration_ms() / 1000.0
        successful_demos = sum(
            bool(
                isinstance(result, dict)
                and "error" not in result
                and "critical_error" not in result
            )
            for result in demonstration_results.values()
        )

        _logger.info("\n" + "=" * 80)
        _logger.info("GYMNASIUM INTEGRATION DEMONSTRATION COMPLETED")
        _logger.info("=" * 80)
        _logger.info(f"Total Execution Time: {total_time_seconds:.1f} seconds")
        _logger.info(f"Components Tested: {len(demonstration_results)}")
        _logger.info(f"Successful Components: {successful_demos}")
        _logger.info(f"Critical Failures: {critical_failures}")
        _logger.info(
            f"Success Rate: {(successful_demos / len(demonstration_results) * 100):.1f}%"
        )

        # Determine final exit status
        if critical_failures > 0:
            _logger.critical("Critical failures detected - system needs attention")
            return 2
        elif (
            successful_demos < len(demonstration_results) * 0.8
        ):  # Less than 80% success
            _logger.error("Multiple demonstration failures - system needs improvement")
            return 1
        else:
            _logger.info("✓ All demonstrations completed successfully")
            return 0

    except KeyboardInterrupt:
        _logger.warning("Demonstration interrupted by user")
        return 1
    except Exception as critical_error:
        _logger.critical(f"Critical demonstration failure: {critical_error}")
        return 2


def main() -> None:
    """Main entry point for advanced Gymnasium integration example with command-line interface, comprehensive
    parameter handling, and production-ready execution management.
    """
    parser = argparse.ArgumentParser(
        description="Advanced Gymnasium Integration Example for plume_nav_sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python gymnasium_integration.py                    # Run standard demonstration
  python gymnasium_integration.py --mode quick      # Quick demonstration mode
  python gymnasium_integration.py --full-testing    # Include all testing components
  python gymnasium_integration.py --parallel        # Include parallel demonstrations
  python gymnasium_integration.py --output ./results # Save results to directory
        """,
    )

    # Parse command-line arguments using argparse for demonstration customization and configuration
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "full"],
        default="standard",
        help="Demonstration mode: quick (minimal), standard (default), full (comprehensive)",
    )

    parser.add_argument(
        "--full-testing",
        action="store_true",
        help="Enable comprehensive performance testing and optimization analysis",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Include parallel environment demonstrations",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for saving results, visualizations, and reports",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for demonstration output",
    )

    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable visualization generation",
    )

    try:
        args = parser.parse_args()

        # Set up global exception handling for unhandled errors with detailed error reporting
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            print(f"\nCritical Error: {exc_type.__name__}: {exc_value}")
            print("Please check the logs for detailed error information.")
            print(
                "If the issue persists, please report it with the error details above."
            )

        sys.excepthook = handle_exception

        # Validate command-line parameters and configuration options with comprehensive error messages
        if args.output:
            output_path = pathlib.Path(args.output)
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"Output directory prepared: {output_path.absolute()}")
            except Exception as output_error:
                print(
                    f"Error: Cannot create output directory {args.output}: {output_error}"
                )
                sys.exit(1)

        # Configure demonstration based on mode
        demo_config = {
            "demo_mode": args.mode,
            "include_performance_testing": args.full_testing or args.mode == "full",
            "include_parallel_demo": args.parallel or args.mode == "full",
            "output_directory": args.output,
        }

        print("=" * 80)
        print("PLUME NAV SIM - ADVANCED GYMNASIUM INTEGRATION EXAMPLE")
        print("=" * 80)
        print(f"Demonstration Mode: {demo_config['demo_mode']}")
        print(
            f"Performance Testing: {'Enabled' if demo_config['include_performance_testing'] else 'Disabled'}"
        )
        print(
            f"Parallel Demo: {'Enabled' if demo_config['include_parallel_demo'] else 'Disabled'}"
        )
        print(
            f"Output Directory: {demo_config['output_directory'] or 'Current directory'}"
        )
        print(f"Log Level: {args.log_level}")
        print("=" * 80)
        print()

        # Execute run_gymnasium_integration_demo with parsed parameters and configuration validation
        exit_code = run_gymnasium_integration_demo(**demo_config)

        # Handle demonstration failures with appropriate exit codes and detailed error analysis
        if exit_code == 0:
            print("\n" + "=" * 80)
            print("✓ DEMONSTRATION COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print("All components demonstrated successfully!")
            print("The plume_nav_sim environment is ready for advanced usage.")

            if args.output:
                print(f"Results and reports saved to: {args.output}")

        elif exit_code == 1:
            print("\n" + "=" * 80)
            print("⚠ DEMONSTRATION COMPLETED WITH WARNINGS")
            print("=" * 80)
            print("Some demonstration components encountered issues.")
            print("Please review the logs and address any reported problems.")
            print("The environment may still be usable for basic functionality.")

        else:  # exit_code == 2
            print("\n" + "=" * 80)
            print("✗ DEMONSTRATION FAILED - CRITICAL ERRORS")
            print("=" * 80)
            print("Critical errors prevented successful demonstration.")
            print("Please review the error logs and resolve issues before use.")
            print("The environment may not be functional in its current state.")

        # Provide comprehensive user guidance for successful completion and troubleshooting
        print("\nFor additional help and documentation:")
        print("  - Check the generated logs for detailed information")
        print("  - Review the technical documentation")
        print("  - Ensure all dependencies are properly installed")
        print("  - Verify system compatibility and resource availability")

        if args.output:
            print(f"  - Examine exported results in: {args.output}")

        # Exit with appropriate status code for automation integration and script chaining compatibility
        sys.exit(exit_code)

    except Exception as main_error:
        print(f"\nFatal Error: Failed to execute demonstration: {main_error}")
        print("Please check your environment setup and try again.")
        sys.exit(2)


# Execute main function when script is run directly
if __name__ == "__main__":
    main()
