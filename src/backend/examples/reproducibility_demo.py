#!/usr/bin/env python3
"""
Comprehensive reproducibility demonstration script showcasing deterministic episode generation,
scientific reproducibility validation, seed management, and cross-session consistency verification
for plume_nav_sim environment.

This educational demonstration script provides complete coverage of reproducibility concepts,
seeding best practices, episode comparison, statistical validation, and reproducibility tracking
for reinforcement learning research workflows with comprehensive logging, performance analysis,
and research-grade documentation.

Usage:
    python reproducibility_demo.py [options]

Example:
    # Basic reproducibility demo
    python reproducibility_demo.py

    # Advanced demo with custom parameters
    python reproducibility_demo.py --seeds 42,123,456 --episodes-per-seed 3 --detailed-output

    # Performance analysis
    python reproducibility_demo.py --performance-tests 10 --validate-targets
"""

import argparse  # >=3.10 - Command-line interface for demo configuration
import hashlib  # >=3.10 - Cryptographic hash functions for episode checksums
import json  # >=3.10 - JSON serialization for reproducibility reports
import logging  # >=3.10 - Comprehensive logging for demonstration tracking
import sys  # >=3.10 - System interface for exit handling
import time  # >=3.10 - High-precision timing measurements
import uuid  # >=3.10 - Unique identifier generation for session tracking
from pathlib import Path  # >=3.10 - Path handling for report output
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt  # >=3.9.0 - Visualization for reproducibility analysis (optional)
import numpy as np  # >=2.1.0 - Random number generation and statistical analysis

# External imports with version comments for dependency management
import gymnasium  # >=0.29.0 - Reinforcement learning environment framework
from plume_nav_sim.core.constants import DEFAULT_MAX_STEPS
from plume_nav_sim.core.types import Action

# Internal imports for plume navigation environment
from plume_nav_sim.registration.register import ENV_ID, register_env
from plume_nav_sim.utils.exceptions import PlumeNavSimError, ValidationError
from plume_nav_sim.utils.seeding import (
    ReproducibilityTracker,
    SeedManager,
    generate_deterministic_seed,
    validate_seed,
)

# Global constants for reproducibility demonstration
DEFAULT_TEST_SEED = 12345
DEFAULT_NUM_REPRODUCIBILITY_TESTS = 5
DEFAULT_EPISODE_LENGTH = 100
DEFAULT_TOLERANCE = 1e-10
REPRODUCIBILITY_TEST_SEEDS = [42, 123, 456, 789, 999]
DEMO_EXPERIMENT_NAMES = [
    "baseline",
    "validation",
    "stress_test",
    "cross_session",
    "statistical",
]
REPORT_OUTPUT_DIR = "reproducibility_reports"

# Module logger for comprehensive demonstration tracking
_logger = logging.getLogger(__name__)

# Public API exports for reproducibility demonstration
__all__ = [
    "run_reproducibility_demo",
    "demonstrate_basic_reproducibility",
    "demonstrate_advanced_reproducibility",
    "validate_cross_session_reproducibility",
    "generate_comprehensive_reproducibility_report",
    "compare_episodes",
    "create_deterministic_policy",
]


def setup_reproducibility_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    include_timestamps: bool = True,
) -> None:
    """
    Configure comprehensive logging for reproducibility demonstration with detailed formatting,
    multiple output destinations, and scientific documentation standards for research workflow
    integration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR), defaults to INFO
        log_file: Optional log file path for persistent logging
        include_timestamps: Whether to include high-precision timestamps in log messages
    """
    # Set default log level to INFO for demonstration visibility
    effective_log_level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)

    # Configure detailed logging format with comprehensive context information
    if include_timestamps:
        log_format = (
            "[%(asctime)s.%(msecs)03d] %(levelname)-8s "
            "%(name)s.%(funcName)s:%(lineno)d - %(message)s"
        )
        date_format = "%Y-%m-%d %H:%M:%S"
    else:
        log_format = "%(levelname)-8s %(name)s.%(funcName)s - %(message)s"
        date_format = None

    # Create specialized formatter for reproducibility demonstration
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Configure root logger with appropriate level
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_log_level)

    # Clear existing handlers to prevent duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(effective_log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set up file handler if log_file provided
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Create file handler with comprehensive logging
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Always log all details to file
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            _logger.info(f"Logging configured: console={log_level}, file={log_file}")

        except Exception as e:
            _logger.warning(f"Failed to setup file logging: {e}")

    # Log logging setup completion with configuration details
    _logger.info("Reproducibility demonstration logging configured")
    _logger.debug(
        f"Log level: {effective_log_level}, Include timestamps: {include_timestamps}"
    )


def create_deterministic_policy(
    env: gymnasium.Env,
    policy_seed: Optional[int] = None,
    policy_type: Optional[str] = None,
) -> Callable:
    """
    Create deterministic policy function for reproducible action selection using fixed seed
    and predictable action patterns enabling identical episode generation for reproducibility
    validation.

    Args:
        env: Gymnasium environment instance for action space extraction
        policy_seed: Seed for deterministic action generation, defaults to 42
        policy_type: Policy type ('random', 'cyclic', 'spiral'), defaults to 'random'

    Returns:
        Deterministic policy function that generates identical action sequences
    """
    # Apply defaults for policy configuration
    effective_seed = policy_seed if policy_seed is not None else 42
    effective_policy_type = policy_type or "random"

    # Initialize numpy random generator with policy_seed for deterministic behavior
    policy_rng = np.random.RandomState(effective_seed)

    # Extract action space size from environment
    action_space_size = env.action_space.n

    _logger.debug(f"Creating {effective_policy_type} policy with seed {effective_seed}")

    # Initialize policy state for stateful policies
    policy_state = {"step_count": 0, "action_sequence": []}

    def policy_function(observation: np.ndarray) -> int:
        """
        Deterministic policy function generating predictable action sequences.

        Args:
            observation: Current environment observation

        Returns:
            Action integer for environment step
        """
        step = policy_state["step_count"]

        if effective_policy_type == "random":
            # Generate deterministic "random" action using seeded RNG
            action = policy_rng.randint(0, action_space_size)

        elif effective_policy_type == "cyclic":
            # Cycle through actions in deterministic sequence
            action = step % action_space_size

        elif effective_policy_type == "spiral":
            # Generate spiral movement pattern for spatial exploration
            direction_sequence = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
            # Increase steps in each direction: 1,1,2,2,3,3,4,4...
            steps_per_direction = (step // 2) + 1
            direction_index = (step // steps_per_direction) % len(direction_sequence)
            action = direction_sequence[direction_index].value

        else:
            raise ValidationError(
                f"Unknown policy type: {effective_policy_type}",
                parameter_name="policy_type",
                invalid_value=effective_policy_type,
                expected_format="'random', 'cyclic', or 'spiral'",
            )

        # Track action in sequence for reproducibility validation
        policy_state["action_sequence"].append(action)
        policy_state["step_count"] += 1

        return action

    # Add metadata to policy function for debugging
    policy_function.seed = effective_seed
    policy_function.policy_type = effective_policy_type
    policy_function.get_action_sequence = lambda: policy_state["action_sequence"].copy()

    _logger.debug(f"Created deterministic {effective_policy_type} policy")
    return policy_function


def execute_reproducibility_episode(
    env: gymnasium.Env,
    policy: Callable,
    episode_seed: int,
    max_steps: Optional[int] = None,
    track_all_states: bool = True,
    calculate_checksums: bool = True,
) -> Dict[str, Any]:
    """
    Execute single episode with comprehensive tracking for reproducibility validation including
    action sequences, observations, state transitions, checksums, and performance metrics for
    scientific comparison.

    Args:
        env: Gymnasium environment instance
        policy: Deterministic policy function for action selection
        episode_seed: Seed for episode initialization
        max_steps: Maximum episode steps, defaults to DEFAULT_EPISODE_LENGTH
        track_all_states: Whether to track all intermediate states
        calculate_checksums: Whether to calculate data integrity checksums

    Returns:
        Comprehensive episode data dictionary for reproducibility comparison
    """
    # Apply default max_steps using DEFAULT_EPISODE_LENGTH
    effective_max_steps = max_steps or DEFAULT_EPISODE_LENGTH

    _logger.debug(
        f"Executing episode with seed {episode_seed}, max_steps {effective_max_steps}"
    )

    # Initialize episode tracking dictionary
    episode_data = {
        "episode_seed": episode_seed,
        "max_steps": effective_max_steps,
        "start_time": time.time(),
        "action_sequence": [],
        "observation_sequence": [],
        "reward_sequence": [],
        "info_sequence": [],
        "state_transitions": [],
        "checksums": {},
        "metadata": {
            "policy_type": getattr(policy, "policy_type", "unknown"),
            "policy_seed": getattr(policy, "seed", None),
        },
    }

    try:
        # Reset environment with episode_seed
        initial_observation, initial_info = env.reset(seed=episode_seed)

        # Record initial state
        episode_data["initial_observation"] = initial_observation.copy()
        episode_data["initial_info"] = initial_info.copy()

        if track_all_states:
            episode_data["observation_sequence"].append(initial_observation.copy())
            episode_data["info_sequence"].append(initial_info.copy())

        _logger.debug(f"Episode initialized with seed {episode_seed}")

        # Execute episode step loop
        observation = initial_observation
        total_reward = 0.0
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and step_count < effective_max_steps:
            # Select action using deterministic policy
            action = policy(observation)
            episode_data["action_sequence"].append(action)

            # Execute environment step
            next_observation, reward, terminated, truncated, info = env.step(action)

            # Track all state information
            episode_data["reward_sequence"].append(reward)
            total_reward += reward

            if track_all_states:
                episode_data["observation_sequence"].append(next_observation.copy())
                episode_data["info_sequence"].append(info.copy())

                # Track complete state transitions for detailed analysis
                episode_data["state_transitions"].append(
                    {
                        "step": step_count,
                        "action": action,
                        "observation": observation.copy(),
                        "next_observation": next_observation.copy(),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info.copy(),
                    }
                )

            # Update state for next iteration
            observation = next_observation
            step_count += 1

        # Record episode completion metadata
        episode_data.update(
            {
                "total_steps": step_count,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
                "final_observation": observation.copy(),
                "end_time": time.time(),
            }
        )

        episode_data["duration"] = episode_data["end_time"] - episode_data["start_time"]

        # Calculate checksums for data integrity verification
        if calculate_checksums:
            episode_data["checksums"] = {
                "action_sequence": hashlib.md5(
                    json.dumps(episode_data["action_sequence"]).encode()
                ).hexdigest(),
                "reward_sequence": hashlib.md5(
                    json.dumps(episode_data["reward_sequence"]).encode()
                ).hexdigest(),
                "total_reward": hashlib.md5(str(total_reward).encode()).hexdigest(),
            }

            if track_all_states:
                # Create checksum for observation sequence (convert to list for JSON serialization)
                obs_list = [
                    obs.tolist() for obs in episode_data["observation_sequence"]
                ]
                episode_data["checksums"]["observation_sequence"] = hashlib.md5(
                    json.dumps(obs_list).encode()
                ).hexdigest()

        _logger.debug(
            f"Episode completed: {step_count} steps, reward {total_reward:.4f}"
        )

    except Exception as e:
        _logger.error(f"Episode execution failed: {e}")
        episode_data["execution_error"] = str(e)
        episode_data["error_time"] = time.time()
        raise

    return episode_data


def compare_episodes(
    episode1: Dict[str, Any],
    episode2: Dict[str, Any],
    tolerance: Optional[float] = None,
    detailed_analysis: bool = True,
    generate_diff_report: bool = False,
) -> Dict[str, Any]:
    """
    Comprehensive episode comparison function with detailed analysis of action sequences,
    observations, state transitions, statistical validation, and discrepancy identification
    for reproducibility verification.

    Args:
        episode1: First episode data dictionary for comparison
        episode2: Second episode data dictionary for comparison
        tolerance: Tolerance for floating point comparisons, defaults to DEFAULT_TOLERANCE
        detailed_analysis: Whether to perform detailed statistical analysis
        generate_diff_report: Whether to generate comprehensive difference report

    Returns:
        Detailed comparison results with match status and analysis
    """
    # Apply default tolerance for floating point comparison
    effective_tolerance = tolerance if tolerance is not None else DEFAULT_TOLERANCE

    _logger.debug("Starting comprehensive episode comparison")

    # Initialize comparison results dictionary
    comparison_results = {
        "comparison_timestamp": time.time(),
        "tolerance": effective_tolerance,
        "detailed_analysis": detailed_analysis,
        "episodes_identical": True,
        "metadata_match": {},
        "sequence_comparisons": {},
        "discrepancies": [],
        "statistics": {},
        "validation_status": "pending",
    }

    try:
        # Compare episode metadata
        metadata_comparison = {
            "seeds_match": episode1.get("episode_seed") == episode2.get("episode_seed"),
            "max_steps_match": episode1.get("max_steps") == episode2.get("max_steps"),
            "total_steps_match": episode1.get("total_steps")
            == episode2.get("total_steps"),
            "terminated_match": episode1.get("terminated")
            == episode2.get("terminated"),
            "truncated_match": episode1.get("truncated") == episode2.get("truncated"),
        }

        comparison_results["metadata_match"] = metadata_comparison

        # Check if episodes should be identical (same seed)
        seeds_identical = metadata_comparison["seeds_match"]

        # Compare action sequences for exact equality
        action_seq1 = episode1.get("action_sequence", [])
        action_seq2 = episode2.get("action_sequence", [])

        actions_match = action_seq1 == action_seq2
        comparison_results["sequence_comparisons"]["actions_match"] = actions_match
        comparison_results["sequence_comparisons"]["action_sequence_lengths"] = {
            "episode1": len(action_seq1),
            "episode2": len(action_seq2),
        }

        if not actions_match:
            comparison_results["episodes_identical"] = False
            if len(action_seq1) != len(action_seq2):
                comparison_results["discrepancies"].append(
                    {
                        "type": "sequence_length_mismatch",
                        "component": "actions",
                        "episode1_length": len(action_seq1),
                        "episode2_length": len(action_seq2),
                    }
                )
            else:
                # Find first differing action
                for i, (a1, a2) in enumerate(zip(action_seq1, action_seq2)):
                    if a1 != a2:
                        comparison_results["discrepancies"].append(
                            {
                                "type": "action_mismatch",
                                "step": i,
                                "episode1_action": a1,
                                "episode2_action": a2,
                            }
                        )
                        break

        # Compare reward sequences using tolerance-based comparison
        reward_seq1 = episode1.get("reward_sequence", [])
        reward_seq2 = episode2.get("reward_sequence", [])

        rewards_match = True
        if len(reward_seq1) != len(reward_seq2):
            rewards_match = False
        else:
            for i, (r1, r2) in enumerate(zip(reward_seq1, reward_seq2)):
                if abs(r1 - r2) > effective_tolerance:
                    rewards_match = False
                    comparison_results["discrepancies"].append(
                        {
                            "type": "reward_mismatch",
                            "step": i,
                            "episode1_reward": r1,
                            "episode2_reward": r2,
                            "difference": abs(r1 - r2),
                        }
                    )
                    break

        comparison_results["sequence_comparisons"]["rewards_match"] = rewards_match

        if not rewards_match:
            comparison_results["episodes_identical"] = False

        # Compare total rewards
        total_reward1 = episode1.get("total_reward", 0.0)
        total_reward2 = episode2.get("total_reward", 0.0)
        total_rewards_match = abs(total_reward1 - total_reward2) <= effective_tolerance

        comparison_results["sequence_comparisons"][
            "total_rewards_match"
        ] = total_rewards_match
        comparison_results["sequence_comparisons"]["total_rewards"] = {
            "episode1": total_reward1,
            "episode2": total_reward2,
            "difference": abs(total_reward1 - total_reward2),
        }

        if not total_rewards_match:
            comparison_results["episodes_identical"] = False

        # Compare observation sequences if available
        if "observation_sequence" in episode1 and "observation_sequence" in episode2:
            obs_seq1 = episode1["observation_sequence"]
            obs_seq2 = episode2["observation_sequence"]

            observations_match = True
            if len(obs_seq1) != len(obs_seq2):
                observations_match = False
            else:
                for i, (obs1, obs2) in enumerate(zip(obs_seq1, obs_seq2)):
                    if not np.allclose(obs1, obs2, atol=effective_tolerance):
                        observations_match = False
                        comparison_results["discrepancies"].append(
                            {
                                "type": "observation_mismatch",
                                "step": i,
                                "max_difference": np.max(np.abs(obs1 - obs2)),
                            }
                        )
                        break

            comparison_results["sequence_comparisons"][
                "observations_match"
            ] = observations_match

            if not observations_match:
                comparison_results["episodes_identical"] = False

        # Validate checksum equality if available
        checksums1 = episode1.get("checksums", {})
        checksums2 = episode2.get("checksums", {})

        if checksums1 and checksums2:
            checksums_match = checksums1 == checksums2
            comparison_results["sequence_comparisons"][
                "checksums_match"
            ] = checksums_match

            if not checksums_match:
                comparison_results["episodes_identical"] = False
                for key in set(checksums1.keys()) | set(checksums2.keys()):
                    if checksums1.get(key) != checksums2.get(key):
                        comparison_results["discrepancies"].append(
                            {
                                "type": "checksum_mismatch",
                                "component": key,
                                "episode1_checksum": checksums1.get(key, "missing"),
                                "episode2_checksum": checksums2.get(key, "missing"),
                            }
                        )

        # Perform detailed statistical analysis if requested
        if detailed_analysis:
            stats = {
                "action_sequence_entropy": {},
                "reward_statistics": {},
                "timing_analysis": {},
            }

            # Calculate action sequence entropy
            if action_seq1:
                unique_actions1, counts1 = np.unique(action_seq1, return_counts=True)
                probs1 = counts1 / len(action_seq1)
                entropy1 = -np.sum(probs1 * np.log2(probs1 + 1e-10))
                stats["action_sequence_entropy"]["episode1"] = entropy1

            if action_seq2:
                unique_actions2, counts2 = np.unique(action_seq2, return_counts=True)
                probs2 = counts2 / len(action_seq2)
                entropy2 = -np.sum(probs2 * np.log2(probs2 + 1e-10))
                stats["action_sequence_entropy"]["episode2"] = entropy2

            # Calculate reward statistics
            if reward_seq1:
                stats["reward_statistics"]["episode1"] = {
                    "mean": np.mean(reward_seq1),
                    "std": np.std(reward_seq1),
                    "min": np.min(reward_seq1),
                    "max": np.max(reward_seq1),
                }

            if reward_seq2:
                stats["reward_statistics"]["episode2"] = {
                    "mean": np.mean(reward_seq2),
                    "std": np.std(reward_seq2),
                    "min": np.min(reward_seq2),
                    "max": np.max(reward_seq2),
                }

            # Timing analysis
            duration1 = episode1.get("duration", 0.0)
            duration2 = episode2.get("duration", 0.0)
            stats["timing_analysis"] = {
                "episode1_duration": duration1,
                "episode2_duration": duration2,
                "duration_difference": abs(duration1 - duration2),
            }

            comparison_results["statistics"] = stats

        # Generate comprehensive diff report if requested
        if generate_diff_report and not comparison_results["episodes_identical"]:
            diff_report = []

            for discrepancy in comparison_results["discrepancies"]:
                if discrepancy["type"] == "action_mismatch":
                    diff_report.append(
                        f"Step {discrepancy['step']}: Action differs "
                        f"({discrepancy['episode1_action']} vs {discrepancy['episode2_action']})"
                    )
                elif discrepancy["type"] == "reward_mismatch":
                    diff_report.append(
                        f"Step {discrepancy['step']}: Reward differs "
                        f"({discrepancy['episode1_reward']:.6f} vs {discrepancy['episode2_reward']:.6f})"
                    )
                elif discrepancy["type"] == "observation_mismatch":
                    diff_report.append(
                        f"Step {discrepancy['step']}: Observation differs "
                        f"(max diff: {discrepancy['max_difference']:.6f})"
                    )
                elif discrepancy["type"] == "checksum_mismatch":
                    diff_report.append(
                        f"Checksum mismatch in {discrepancy['component']}"
                    )

            comparison_results["diff_report"] = diff_report

        # Set final validation status
        if comparison_results["episodes_identical"]:
            if seeds_identical:
                comparison_results["validation_status"] = "reproducible"
            else:
                comparison_results["validation_status"] = "identical_different_seeds"
        else:
            if seeds_identical:
                comparison_results["validation_status"] = "non_reproducible"
            else:
                comparison_results["validation_status"] = "different_as_expected"

        _logger.debug(
            f"Episode comparison completed: {comparison_results['validation_status']}"
        )

    except Exception as e:
        _logger.error(f"Episode comparison failed: {e}")
        comparison_results["comparison_error"] = str(e)
        comparison_results["validation_status"] = "comparison_failed"

    return comparison_results


def demonstrate_basic_reproducibility(
    env: gymnasium.Env,
    demo_seed: Optional[int] = None,
    episode_length: Optional[int] = None,
    show_detailed_output: bool = False,
) -> bool:
    """
    Demonstrate fundamental reproducibility concepts with identical seed comparison,
    basic validation, educational logging, and simple statistical analysis for
    learning workflow introduction.

    Args:
        env: Gymnasium environment instance
        demo_seed: Demonstration seed, defaults to DEFAULT_TEST_SEED
        episode_length: Episode length, defaults to DEFAULT_EPISODE_LENGTH
        show_detailed_output: Whether to show detailed step-by-step output

    Returns:
        True if basic reproducibility demonstrated successfully
    """
    # Apply default parameters
    effective_seed = demo_seed if demo_seed is not None else DEFAULT_TEST_SEED
    effective_length = (
        episode_length if episode_length is not None else DEFAULT_EPISODE_LENGTH
    )

    _logger.info("=== BASIC REPRODUCIBILITY DEMONSTRATION ===")
    _logger.info("This demo shows that identical seeds produce identical episodes")
    _logger.info(f"Using seed: {effective_seed}, episode length: {effective_length}")

    try:
        # Create deterministic policy for reproducible actions
        policy = create_deterministic_policy(
            env, policy_seed=12345, policy_type="random"
        )
        _logger.info(
            f"Created deterministic {policy.policy_type} policy with seed {policy.seed}"
        )

        # Execute first episode with demonstration seed
        _logger.info("Executing first episode...")
        episode1 = execute_reproducibility_episode(
            env=env,
            policy=policy,
            episode_seed=effective_seed,
            max_steps=effective_length,
            track_all_states=show_detailed_output,
            calculate_checksums=True,
        )

        # Reset policy state for second episode
        policy = create_deterministic_policy(
            env, policy_seed=12345, policy_type="random"
        )

        # Execute second episode with identical seed
        _logger.info("Executing second episode with identical seed...")
        episode2 = execute_reproducibility_episode(
            env=env,
            policy=policy,
            episode_seed=effective_seed,
            max_steps=effective_length,
            track_all_states=show_detailed_output,
            calculate_checksums=True,
        )

        # Compare episodes for reproducibility validation
        _logger.info("Comparing episodes for reproducibility...")
        comparison = compare_episodes(
            episode1=episode1,
            episode2=episode2,
            detailed_analysis=True,
            generate_diff_report=True,
        )

        # Display detailed output if requested
        if show_detailed_output:
            _logger.info(
                f"Episode 1 actions: {episode1['action_sequence'][:20]}{'...' if len(episode1['action_sequence']) > 20 else ''}"
            )
            _logger.info(
                f"Episode 2 actions: {episode2['action_sequence'][:20]}{'...' if len(episode2['action_sequence']) > 20 else ''}"
            )
            _logger.info(f"Episode 1 total reward: {episode1['total_reward']:.6f}")
            _logger.info(f"Episode 2 total reward: {episode2['total_reward']:.6f}")

            if "checksums" in episode1 and "checksums" in episode2:
                _logger.info("Checksums comparison:")
                for key in episode1["checksums"]:
                    match = (
                        "✓"
                        if episode1["checksums"][key]
                        == episode2.get("checksums", {}).get(key)
                        else "✗"
                    )
                    _logger.info(f"  {key}: {match}")

        # Analyze reproducibility results
        reproducible = comparison["episodes_identical"]
        validation_status = comparison["validation_status"]

        if reproducible:
            _logger.info(
                "✓ SUCCESS: Episodes are identical - reproducibility confirmed!"
            )
            _logger.info(
                f"  - Action sequences match: {comparison['sequence_comparisons']['actions_match']}"
            )
            _logger.info(
                f"  - Reward sequences match: {comparison['sequence_comparisons']['rewards_match']}"
            )
            _logger.info(
                f"  - Total rewards match: {comparison['sequence_comparisons']['total_rewards_match']}"
            )

            if "checksums_match" in comparison["sequence_comparisons"]:
                _logger.info(
                    f"  - Data checksums match: {comparison['sequence_comparisons']['checksums_match']}"
                )
        else:
            _logger.error("✗ FAILURE: Episodes differ - reproducibility not achieved!")
            _logger.error(f"  - Validation status: {validation_status}")
            _logger.error(
                f"  - Number of discrepancies: {len(comparison['discrepancies'])}"
            )

            if comparison.get("diff_report"):
                _logger.error("  - Key differences:")
                for diff in comparison["diff_report"][:5]:  # Show first 5 differences
                    _logger.error(f"    {diff}")

        # Provide educational explanation
        _logger.info("\n=== EDUCATIONAL INSIGHTS ===")
        _logger.info("Reproducibility in RL environments requires:")
        _logger.info("1. Seeding the environment with reset(seed=X)")
        _logger.info("2. Using deterministic policies or seeded random policies")
        _logger.info("3. Ensuring no external randomness sources")
        _logger.info("4. Consistent environment configuration")

        if reproducible:
            _logger.info(
                "✓ This environment correctly implements reproducibility standards"
            )
        else:
            _logger.warning("⚠ Check environment seeding implementation")

        return reproducible

    except Exception as e:
        _logger.error(f"Basic reproducibility demonstration failed: {e}")
        raise


def demonstrate_advanced_reproducibility(
    env: gymnasium.Env,
    test_seeds: Optional[List[int]] = None,
    num_episodes_per_seed: Optional[int] = None,
    use_seed_manager: bool = True,
    use_reproducibility_tracker: bool = True,
) -> Dict[str, Any]:
    """
    Demonstrate advanced reproducibility features including cross-seed validation,
    statistical analysis, session management, comprehensive tracking, and research-grade
    validation for scientific workflows.

    Args:
        env: Gymnasium environment instance
        test_seeds: List of seeds for testing, defaults to REPRODUCIBILITY_TEST_SEEDS
        num_episodes_per_seed: Episodes per seed, defaults to 2
        use_seed_manager: Whether to use SeedManager for centralized seed management
        use_reproducibility_tracker: Whether to use ReproducibilityTracker

    Returns:
        Comprehensive advanced reproducibility results
    """
    # Apply default parameters
    effective_test_seeds = test_seeds or REPRODUCIBILITY_TEST_SEEDS
    effective_episodes_per_seed = num_episodes_per_seed or 2

    _logger.info("=== ADVANCED REPRODUCIBILITY DEMONSTRATION ===")
    _logger.info(
        f"Testing {len(effective_test_seeds)} seeds with {effective_episodes_per_seed} episodes each"
    )
    _logger.info(
        f"Using SeedManager: {use_seed_manager}, ReproducibilityTracker: {use_reproducibility_tracker}"
    )

    # Initialize results dictionary
    results = {
        "test_configuration": {
            "test_seeds": effective_test_seeds,
            "episodes_per_seed": effective_episodes_per_seed,
            "use_seed_manager": use_seed_manager,
            "use_reproducibility_tracker": use_reproducibility_tracker,
        },
        "seed_results": {},
        "cross_seed_analysis": {},
        "reproducibility_statistics": {},
        "validation_summary": {},
    }

    try:
        # Initialize SeedManager if requested
        seed_manager = None
        if use_seed_manager:
            seed_manager = SeedManager()
            _logger.info("Initialized SeedManager for centralized seed management")

        # Initialize ReproducibilityTracker if requested
        reproducibility_tracker = None
        if use_reproducibility_tracker:
            reproducibility_tracker = ReproducibilityTracker()
            _logger.info(
                "Initialized ReproducibilityTracker for comprehensive analysis"
            )

        # Test each seed for reproducibility consistency
        successful_reproductions = 0
        total_comparisons = 0

        for seed in effective_test_seeds:
            _logger.info(f"Testing reproducibility for seed {seed}...")

            seed_results = {
                "seed": seed,
                "episodes": [],
                "comparisons": [],
                "reproducible": True,
                "statistics": {},
            }

            # Generate episode seed if using SeedManager
            if seed_manager:
                episode_seed = seed_manager.generate_episode_seed(base_seed=seed)
                _logger.debug(f"SeedManager generated episode seed: {episode_seed}")
            else:
                episode_seed = seed

            # Run multiple episodes with same seed
            episodes = []
            for episode_num in range(effective_episodes_per_seed):
                _logger.debug(
                    f"  Episode {episode_num + 1}/{effective_episodes_per_seed} for seed {seed}"
                )

                # Create fresh policy for each episode
                policy = create_deterministic_policy(
                    env, policy_seed=42, policy_type="random"
                )

                # Execute episode
                episode_data = execute_reproducibility_episode(
                    env=env,
                    policy=policy,
                    episode_seed=episode_seed,
                    max_steps=DEFAULT_EPISODE_LENGTH,
                    track_all_states=True,
                    calculate_checksums=True,
                )

                episodes.append(episode_data)
                seed_results["episodes"].append(episode_data)

                # Record in ReproducibilityTracker if enabled
                if reproducibility_tracker:
                    reproducibility_tracker.record_episode(
                        episode_seed=episode_seed,
                        action_sequence=episode_data["action_sequence"],
                        observation_sequence=episode_data.get(
                            "observation_sequence", []
                        ),
                        episode_metadata={
                            "total_reward": episode_data["total_reward"],
                            "total_steps": episode_data["total_steps"],
                            "terminated": episode_data["terminated"],
                        },
                    )

            # Compare all episode pairs for consistency
            for i in range(len(episodes)):
                for j in range(i + 1, len(episodes)):
                    comparison = compare_episodes(
                        episode1=episodes[i],
                        episode2=episodes[j],
                        detailed_analysis=True,
                    )

                    seed_results["comparisons"].append(comparison)
                    total_comparisons += 1

                    if comparison["episodes_identical"]:
                        successful_reproductions += 1
                    else:
                        seed_results["reproducible"] = False
                        _logger.warning(
                            f"Reproducibility failure for seed {seed}: episodes {i+1} and {j+1} differ"
                        )

            # Verify reproducibility using ReproducibilityTracker if enabled
            if reproducibility_tracker and len(episodes) >= 2:
                tracker_verification = (
                    reproducibility_tracker.verify_episode_reproducibility(
                        episode_seed=episode_seed,
                        reference_actions=episodes[0]["action_sequence"],
                        reference_observations=episodes[0].get(
                            "observation_sequence", []
                        ),
                    )
                )

                seed_results["tracker_verification"] = tracker_verification

                if not tracker_verification["reproducible"]:
                    _logger.warning(
                        f"ReproducibilityTracker verification failed for seed {seed}"
                    )

            # Calculate seed-specific statistics
            if episodes:
                total_rewards = [ep["total_reward"] for ep in episodes]
                total_steps = [ep["total_steps"] for ep in episodes]

                seed_results["statistics"] = {
                    "total_rewards": {
                        "mean": np.mean(total_rewards),
                        "std": np.std(total_rewards),
                        "range": (np.min(total_rewards), np.max(total_rewards)),
                    },
                    "total_steps": {
                        "mean": np.mean(total_steps),
                        "std": np.std(total_steps),
                        "range": (np.min(total_steps), np.max(total_steps)),
                    },
                }

                # Check if statistics show perfect reproducibility
                perfect_reproducibility = (
                    seed_results["statistics"]["total_rewards"]["std"] == 0.0
                    and seed_results["statistics"]["total_steps"]["std"] == 0.0
                )
                seed_results["perfect_reproducibility"] = perfect_reproducibility

            results["seed_results"][seed] = seed_results

            # Log seed results
            if seed_results["reproducible"]:
                _logger.info(f"✓ Seed {seed}: Reproducible across all episodes")
            else:
                _logger.warning(f"✗ Seed {seed}: Reproducibility issues detected")

        # Perform cross-seed statistical analysis
        _logger.info("Performing cross-seed statistical analysis...")

        all_seeds_reproducible = all(
            results["seed_results"][seed]["reproducible"]
            for seed in effective_test_seeds
        )

        # Analyze differences between seeds (should be different)
        cross_seed_differences = []
        for i, seed1 in enumerate(effective_test_seeds):
            for seed2 in effective_test_seeds[i + 1 :]:
                if (
                    results["seed_results"][seed1]["episodes"]
                    and results["seed_results"][seed2]["episodes"]
                ):

                    ep1 = results["seed_results"][seed1]["episodes"][0]
                    ep2 = results["seed_results"][seed2]["episodes"][0]

                    comparison = compare_episodes(ep1, ep2, detailed_analysis=False)
                    cross_seed_differences.append(
                        {
                            "seed1": seed1,
                            "seed2": seed2,
                            "episodes_different": not comparison["episodes_identical"],
                        }
                    )

        seeds_produce_different_results = all(
            diff["episodes_different"] for diff in cross_seed_differences
        )

        results["cross_seed_analysis"] = {
            "all_seeds_reproducible": all_seeds_reproducible,
            "seeds_produce_different_results": seeds_produce_different_results,
            "cross_seed_comparisons": cross_seed_differences,
        }

        # Generate comprehensive statistics
        success_rate = (
            successful_reproductions / total_comparisons
            if total_comparisons > 0
            else 0.0
        )

        results["reproducibility_statistics"] = {
            "total_comparisons": total_comparisons,
            "successful_reproductions": successful_reproductions,
            "success_rate": success_rate,
            "seeds_tested": len(effective_test_seeds),
            "episodes_per_seed": effective_episodes_per_seed,
        }

        # Generate validation summary
        validation_status = (
            "PASS" if (all_seeds_reproducible and success_rate >= 0.95) else "FAIL"
        )

        results["validation_summary"] = {
            "overall_status": validation_status,
            "reproducibility_achieved": all_seeds_reproducible,
            "cross_seed_diversity": seeds_produce_different_results,
            "success_rate": success_rate,
            "recommendations": [],
        }

        # Add recommendations based on results
        if not all_seeds_reproducible:
            results["validation_summary"]["recommendations"].append(
                "Review environment seeding implementation"
            )

        if not seeds_produce_different_results:
            results["validation_summary"]["recommendations"].append(
                "Verify seed manager generates diverse seeds"
            )

        if success_rate < 0.95:
            results["validation_summary"]["recommendations"].append(
                "Investigate sources of non-determinism"
            )

        # Generate ReproducibilityTracker report if enabled
        if reproducibility_tracker:
            tracker_report = reproducibility_tracker.generate_reproducibility_report()
            results["tracker_report"] = tracker_report
            _logger.info(
                f"ReproducibilityTracker processed {tracker_report.get('total_episodes', 0)} episodes"
            )

        # Log final results
        _logger.info(f"=== ADVANCED REPRODUCIBILITY RESULTS ===")
        _logger.info(f"Overall status: {validation_status}")
        _logger.info(f"Success rate: {success_rate:.2%}")
        _logger.info(f"All seeds reproducible: {all_seeds_reproducible}")
        _logger.info(
            f"Seeds produce different results: {seeds_produce_different_results}"
        )

        if validation_status == "PASS":
            _logger.info("✓ Advanced reproducibility validation PASSED")
        else:
            _logger.warning("⚠ Advanced reproducibility validation had issues")
            for rec in results["validation_summary"]["recommendations"]:
                _logger.warning(f"  Recommendation: {rec}")

        return results

    except Exception as e:
        _logger.error(f"Advanced reproducibility demonstration failed: {e}")
        results["error"] = str(e)
        results["validation_summary"] = {"overall_status": "ERROR"}
        raise


def demonstrate_seed_string_reproducibility(
    env: gymnasium.Env,
    experiment_names: Optional[List[str]] = None,
    episodes_per_experiment: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Demonstrate string-based deterministic seed generation for experiment naming and
    configuration-based reproducibility enabling research workflow integration with
    meaningful experiment identifiers.

    Args:
        env: Gymnasium environment instance
        experiment_names: List of experiment names, defaults to DEMO_EXPERIMENT_NAMES
        episodes_per_experiment: Episodes per experiment, defaults to 2

    Returns:
        String-based seed reproducibility results with experiment naming validation
    """
    # Apply defaults
    effective_names = experiment_names or DEMO_EXPERIMENT_NAMES
    effective_episodes = episodes_per_experiment or 2

    _logger.info("=== STRING-BASED SEED REPRODUCIBILITY DEMONSTRATION ===")
    _logger.info("This demo shows deterministic seed generation from experiment names")
    _logger.info(f"Testing experiments: {effective_names}")

    results = {
        "experiment_names": effective_names,
        "episodes_per_experiment": effective_episodes,
        "experiment_results": {},
        "seed_generation_analysis": {},
        "reproducibility_validation": {},
    }

    try:
        seed_generation_results = {}

        # Test seed generation for each experiment name
        for experiment_name in effective_names:
            _logger.info(f"Testing experiment: '{experiment_name}'")

            # Generate deterministic seed from experiment name
            generated_seed = generate_deterministic_seed(experiment_name)
            _logger.info(f"  Generated seed: {generated_seed}")

            # Validate seed generation is consistent
            seed2 = generate_deterministic_seed(experiment_name)
            seed3 = generate_deterministic_seed(experiment_name)

            seed_consistency = generated_seed == seed2 == seed3

            seed_generation_results[experiment_name] = {
                "generated_seed": generated_seed,
                "seed_consistent": seed_consistency,
                "multiple_generations": [generated_seed, seed2, seed3],
            }

            if not seed_consistency:
                _logger.error(f"Seed generation inconsistency for '{experiment_name}'")

            # Run multiple episodes with string-derived seed
            experiment_episodes = []
            for episode_num in range(effective_episodes):
                policy = create_deterministic_policy(
                    env, policy_seed=42, policy_type="cyclic"
                )

                episode_data = execute_reproducibility_episode(
                    env=env,
                    policy=policy,
                    episode_seed=generated_seed,
                    max_steps=50,  # Shorter episodes for string demo
                    track_all_states=False,
                    calculate_checksums=True,
                )

                experiment_episodes.append(episode_data)

            # Compare episodes for reproducibility
            episode_comparisons = []
            episodes_reproducible = True

            for i in range(len(experiment_episodes)):
                for j in range(i + 1, len(experiment_episodes)):
                    comparison = compare_episodes(
                        episode1=experiment_episodes[i],
                        episode2=experiment_episodes[j],
                        detailed_analysis=False,
                    )
                    episode_comparisons.append(comparison)

                    if not comparison["episodes_identical"]:
                        episodes_reproducible = False

            results["experiment_results"][experiment_name] = {
                "generated_seed": generated_seed,
                "seed_consistent": seed_consistency,
                "episodes": experiment_episodes,
                "comparisons": episode_comparisons,
                "episodes_reproducible": episodes_reproducible,
            }

            _logger.info(f"  Seed consistency: {'✓' if seed_consistency else '✗'}")
            _logger.info(
                f"  Episodes reproducible: {'✓' if episodes_reproducible else '✗'}"
            )

        # Analyze seed generation properties
        _logger.info("Analyzing seed generation properties...")

        generated_seeds = [
            results["experiment_results"][name]["generated_seed"]
            for name in effective_names
        ]

        # Check for seed uniqueness (different names should produce different seeds)
        unique_seeds = len(set(generated_seeds))
        all_seeds_unique = unique_seeds == len(generated_seeds)

        # Analyze seed distribution
        seed_stats = {
            "total_experiments": len(effective_names),
            "unique_seeds": unique_seeds,
            "all_seeds_unique": all_seeds_unique,
            "seed_range": (min(generated_seeds), max(generated_seeds)),
            "seed_mean": np.mean(generated_seeds),
            "seed_std": np.std(generated_seeds),
        }

        results["seed_generation_analysis"] = seed_stats

        # Test experiment name variations
        _logger.info("Testing experiment name variations...")

        variations_test = {}
        base_name = effective_names[0] if effective_names else "test"

        name_variations = [
            base_name,
            base_name.upper(),
            base_name.lower(),
            base_name + "_v1",
            base_name + "_v2",
        ]

        for variation in name_variations:
            var_seed = generate_deterministic_seed(variation)
            variations_test[variation] = var_seed

        # Check that variations produce different seeds
        variation_seeds = list(variations_test.values())
        variations_unique = len(set(variation_seeds)) == len(variation_seeds)

        results["seed_generation_analysis"]["name_variations"] = {
            "variations_tested": variations_test,
            "all_variations_unique": variations_unique,
        }

        # Overall validation
        all_experiments_reproducible = all(
            results["experiment_results"][name]["episodes_reproducible"]
            for name in effective_names
        )

        all_seed_generation_consistent = all(
            results["experiment_results"][name]["seed_consistent"]
            for name in effective_names
        )

        validation_status = (
            all_experiments_reproducible
            and all_seed_generation_consistent
            and all_seeds_unique
        )

        results["reproducibility_validation"] = {
            "all_experiments_reproducible": all_experiments_reproducible,
            "all_seed_generation_consistent": all_seed_generation_consistent,
            "all_seeds_unique": all_seeds_unique,
            "validation_passed": validation_status,
        }

        # Log results
        _logger.info("=== STRING-BASED SEED RESULTS ===")
        _logger.info(f"Seed generation consistent: {all_seed_generation_consistent}")
        _logger.info(f"All experiments reproducible: {all_experiments_reproducible}")
        _logger.info(f"All seeds unique: {all_seeds_unique}")
        _logger.info(f"Name variations unique: {variations_unique}")

        if validation_status:
            _logger.info("✓ String-based seed reproducibility PASSED")
        else:
            _logger.warning("⚠ String-based seed reproducibility had issues")

        # Demonstrate research workflow integration
        _logger.info("\n=== RESEARCH WORKFLOW INTEGRATION ===")
        _logger.info("String-based seeding enables:")
        _logger.info("1. Reproducible experiments from configuration files")
        _logger.info("2. Human-readable experiment identifiers")
        _logger.info("3. Version control friendly experiment tracking")
        _logger.info("4. Collision-resistant seed generation")

        return results

    except Exception as e:
        _logger.error(f"String-based seed demonstration failed: {e}")
        results["error"] = str(e)
        raise


def validate_cross_session_reproducibility(
    session_id: Optional[str] = None,
    base_seed: Optional[int] = None,
    num_validation_episodes: Optional[int] = None,
    save_session_data: bool = True,
    compare_with_previous: bool = False,
) -> Dict[str, Any]:
    """
    Validate reproducibility across different sessions, processes, and time periods with
    comprehensive state persistence, session management, and long-term consistency
    verification for scientific research.

    Args:
        session_id: Session identifier, generates UUID if not provided
        base_seed: Base seed for validation, defaults to DEFAULT_TEST_SEED
        num_validation_episodes: Number of episodes, defaults to 3
        save_session_data: Whether to save session data to disk
        compare_with_previous: Whether to compare with previous session data

    Returns:
        Cross-session reproducibility validation results
    """
    # Generate session parameters
    effective_session_id = session_id or str(uuid.uuid4())[:8]
    effective_base_seed = base_seed if base_seed is not None else DEFAULT_TEST_SEED
    effective_num_episodes = num_validation_episodes or 3

    _logger.info("=== CROSS-SESSION REPRODUCIBILITY VALIDATION ===")
    _logger.info(f"Session ID: {effective_session_id}")
    _logger.info(f"Base seed: {effective_base_seed}")
    _logger.info(f"Validation episodes: {effective_num_episodes}")

    results = {
        "session_id": effective_session_id,
        "base_seed": effective_base_seed,
        "num_episodes": effective_num_episodes,
        "session_timestamp": time.time(),
        "validation_episodes": [],
        "reproducibility_analysis": {},
        "session_persistence": {},
        "previous_session_comparison": {},
    }

    try:
        # Register and create environment for cross-session testing
        register_env()
        env = gymnasium.make(ENV_ID)

        _logger.info("Executing validation episodes for session consistency...")

        # Execute multiple episodes for session validation
        session_episodes = []

        for episode_num in range(effective_num_episodes):
            _logger.info(f"Session episode {episode_num + 1}/{effective_num_episodes}")

            # Create deterministic policy for session
            policy = create_deterministic_policy(
                env, policy_seed=999, policy_type="spiral"
            )

            # Generate episode seed with session context
            episode_seed = effective_base_seed + episode_num

            # Execute episode with comprehensive tracking
            episode_data = execute_reproducibility_episode(
                env=env,
                policy=policy,
                episode_seed=episode_seed,
                max_steps=75,
                track_all_states=True,
                calculate_checksums=True,
            )

            # Add session metadata
            episode_data["session_metadata"] = {
                "session_id": effective_session_id,
                "episode_number": episode_num,
                "session_timestamp": results["session_timestamp"],
            }

            session_episodes.append(episode_data)
            results["validation_episodes"].append(episode_data)

        env.close()

        # Analyze session consistency
        _logger.info("Analyzing session consistency...")

        session_consistency = True
        consistency_checks = []

        # Verify episode reproducibility within session
        for i, episode in enumerate(session_episodes):
            # Re-run episode with same parameters to test consistency
            env_verify = gymnasium.make(ENV_ID)
            policy_verify = create_deterministic_policy(
                env_verify, policy_seed=999, policy_type="spiral"
            )

            episode_verify = execute_reproducibility_episode(
                env=env_verify,
                policy=policy_verify,
                episode_seed=episode["episode_seed"],
                max_steps=75,
                track_all_states=True,
                calculate_checksums=True,
            )

            env_verify.close()

            # Compare original and verification episodes
            comparison = compare_episodes(
                episode1=episode, episode2=episode_verify, detailed_analysis=True
            )

            consistency_checks.append(
                {
                    "episode_number": i,
                    "consistent": comparison["episodes_identical"],
                    "comparison": comparison,
                }
            )

            if not comparison["episodes_identical"]:
                session_consistency = False
                _logger.warning(f"Session consistency failed for episode {i}")

        results["reproducibility_analysis"] = {
            "session_consistent": session_consistency,
            "consistency_checks": consistency_checks,
            "total_episodes_tested": len(session_episodes),
            "consistent_episodes": sum(
                1 for check in consistency_checks if check["consistent"]
            ),
        }

        # Save session data if requested
        if save_session_data:
            _logger.info("Saving session data for cross-session comparison...")

            session_data_dir = Path(REPORT_OUTPUT_DIR) / "session_data"
            session_data_dir.mkdir(parents=True, exist_ok=True)

            session_file = session_data_dir / f"session_{effective_session_id}.json"

            # Prepare session data for JSON serialization
            session_data_for_saving = {
                "session_id": effective_session_id,
                "base_seed": effective_base_seed,
                "session_timestamp": results["session_timestamp"],
                "episodes": [],
            }

            for episode in session_episodes:
                # Convert numpy arrays to lists for JSON serialization
                episode_for_json = episode.copy()

                # Convert observation sequences
                if "observation_sequence" in episode_for_json:
                    episode_for_json["observation_sequence"] = [
                        obs.tolist() for obs in episode_for_json["observation_sequence"]
                    ]

                # Convert single observations
                for obs_key in ["initial_observation", "final_observation"]:
                    if obs_key in episode_for_json:
                        episode_for_json[obs_key] = episode_for_json[obs_key].tolist()

                # Remove state transitions (too large for JSON)
                episode_for_json.pop("state_transitions", None)

                session_data_for_saving["episodes"].append(episode_for_json)

            # Save session data
            with open(session_file, "w") as f:
                json.dump(session_data_for_saving, f, indent=2)

            results["session_persistence"] = {
                "data_saved": True,
                "save_path": str(session_file),
                "data_size_bytes": session_file.stat().st_size,
            }

            _logger.info(f"Session data saved: {session_file}")

        # Compare with previous session if requested
        if compare_with_previous and save_session_data:
            _logger.info("Comparing with previous session data...")

            session_data_dir = Path(REPORT_OUTPUT_DIR) / "session_data"

            if session_data_dir.exists():
                # Find previous session files
                previous_sessions = list(session_data_dir.glob("session_*.json"))
                previous_sessions = [
                    f
                    for f in previous_sessions
                    if f.stem != f"session_{effective_session_id}"
                ]

                if previous_sessions:
                    # Load most recent previous session
                    latest_previous = max(
                        previous_sessions, key=lambda f: f.stat().st_mtime
                    )

                    try:
                        with open(latest_previous, "r") as f:
                            previous_data = json.load(f)

                        _logger.info(f"Loaded previous session: {latest_previous.stem}")

                        # Compare session configurations
                        config_match = (
                            previous_data.get("base_seed") == effective_base_seed
                            and len(previous_data.get("episodes", [])) >= 1
                            and len(session_episodes) >= 1
                        )

                        if config_match and previous_data.get("episodes"):
                            # Compare first episode from each session
                            prev_episode = previous_data["episodes"][0]
                            curr_episode = session_episodes[0]

                            # Simple comparison of key metrics
                            cross_session_match = (
                                prev_episode.get("total_reward")
                                == curr_episode.get("total_reward")
                                and prev_episode.get("total_steps")
                                == curr_episode.get("total_steps")
                                and prev_episode.get("action_sequence")
                                == curr_episode.get("action_sequence")
                            )

                            results["previous_session_comparison"] = {
                                "previous_session_found": True,
                                "previous_session_id": previous_data.get("session_id"),
                                "previous_timestamp": previous_data.get(
                                    "session_timestamp"
                                ),
                                "configuration_match": config_match,
                                "cross_session_reproducible": cross_session_match,
                            }

                            if cross_session_match:
                                _logger.info(
                                    "✓ Cross-session reproducibility confirmed"
                                )
                            else:
                                _logger.warning("⚠ Cross-session differences detected")
                        else:
                            results["previous_session_comparison"] = {
                                "previous_session_found": True,
                                "configuration_match": False,
                                "message": "Configuration mismatch with previous session",
                            }

                    except Exception as e:
                        _logger.warning(f"Failed to load previous session data: {e}")
                        results["previous_session_comparison"] = {
                            "previous_session_found": False,
                            "error": str(e),
                        }
                else:
                    results["previous_session_comparison"] = {
                        "previous_session_found": False,
                        "message": "No previous sessions found",
                    }
            else:
                results["previous_session_comparison"] = {
                    "previous_session_found": False,
                    "message": "Session data directory does not exist",
                }

        # Generate validation summary
        validation_passed = session_consistency

        if "cross_session_reproducible" in results.get(
            "previous_session_comparison", {}
        ):
            validation_passed = (
                validation_passed
                and results["previous_session_comparison"]["cross_session_reproducible"]
            )

        results["validation_summary"] = {
            "session_consistency": session_consistency,
            "validation_passed": validation_passed,
            "temporal_stability": (
                "verified" if session_consistency else "issues_detected"
            ),
            "recommendations": [],
        }

        # Add recommendations
        if not session_consistency:
            results["validation_summary"]["recommendations"].append(
                "Review environment state management for session consistency"
            )

        if save_session_data:
            results["validation_summary"]["recommendations"].append(
                "Session data saved for long-term reproducibility tracking"
            )

        # Log results
        _logger.info("=== CROSS-SESSION VALIDATION RESULTS ===")
        _logger.info(f"Session consistency: {'✓' if session_consistency else '✗'}")
        _logger.info(f"Episodes tested: {len(session_episodes)}")
        _logger.info(
            f"Consistent episodes: {results['reproducibility_analysis']['consistent_episodes']}"
        )

        if (
            results["previous_session_comparison"].get("cross_session_reproducible")
            is not None
        ):
            cross_session_status = results["previous_session_comparison"][
                "cross_session_reproducible"
            ]
            _logger.info(
                f"Cross-session reproducible: {'✓' if cross_session_status else '✗'}"
            )

        if validation_passed:
            _logger.info("✓ Cross-session reproducibility validation PASSED")
        else:
            _logger.warning("⚠ Cross-session reproducibility validation had issues")

        return results

    except Exception as e:
        _logger.error(f"Cross-session validation failed: {e}")
        results["error"] = str(e)
        results["validation_summary"] = {"validation_passed": False}
        raise


def demonstrate_reproducibility_performance(
    env: gymnasium.Env,
    num_performance_tests: Optional[int] = None,
    validate_performance_targets: bool = False,
    include_memory_analysis: bool = False,
) -> Dict[str, Any]:
    """
    Demonstrate and validate reproducibility performance characteristics including seeding
    overhead, memory usage, timing analysis, and optimization verification for research
    workflow efficiency.

    Args:
        env: Gymnasium environment instance
        num_performance_tests: Number of performance tests, defaults to 10
        validate_performance_targets: Whether to validate <1ms seeding targets
        include_memory_analysis: Whether to include memory usage analysis

    Returns:
        Reproducibility performance analysis with timing and optimization recommendations
    """
    # Apply defaults
    effective_num_tests = num_performance_tests or 10

    _logger.info("=== REPRODUCIBILITY PERFORMANCE DEMONSTRATION ===")
    _logger.info(f"Performance tests: {effective_num_tests}")
    _logger.info(f"Validate targets: {validate_performance_targets}")
    _logger.info(f"Memory analysis: {include_memory_analysis}")

    results = {
        "num_tests": effective_num_tests,
        "seeding_performance": {},
        "episode_performance": {},
        "reproducibility_overhead": {},
        "memory_analysis": {},
        "performance_validation": {},
        "optimization_recommendations": [],
    }

    try:
        # Measure seeding overhead
        _logger.info("Measuring seeding overhead...")

        seeding_times = []
        rng_initialization_times = []

        for test_num in range(effective_num_tests):
            # Measure seed validation time
            start_time = time.perf_counter()
            validate_seed(12345 + test_num)
            seed_validation_time = time.perf_counter() - start_time

            # Measure RNG initialization time
            start_time = time.perf_counter()
            np.random.RandomState(12345 + test_num)
            rng_init_time = time.perf_counter() - start_time

            # Measure environment reset with seed
            start_time = time.perf_counter()
            env.reset(seed=12345 + test_num)
            env_reset_time = time.perf_counter() - start_time

            total_seeding_time = seed_validation_time + rng_init_time + env_reset_time

            seeding_times.append(
                {
                    "test_num": test_num,
                    "seed_validation": seed_validation_time * 1000,  # Convert to ms
                    "rng_initialization": rng_init_time * 1000,
                    "env_reset": env_reset_time * 1000,
                    "total_seeding": total_seeding_time * 1000,
                }
            )

            rng_initialization_times.append(rng_init_time * 1000)

        # Calculate seeding performance statistics
        total_times = [t["total_seeding"] for t in seeding_times]

        results["seeding_performance"] = {
            "individual_tests": seeding_times,
            "statistics": {
                "mean_ms": np.mean(total_times),
                "std_ms": np.std(total_times),
                "min_ms": np.min(total_times),
                "max_ms": np.max(total_times),
                "median_ms": np.median(total_times),
            },
            "target_1ms": np.mean(total_times) < 1.0,  # <1ms target
        }

        _logger.info(
            f"Seeding overhead: {np.mean(total_times):.3f}ms (avg), {np.max(total_times):.3f}ms (max)"
        )

        # Measure episode execution performance with reproducibility tracking
        _logger.info("Measuring episode performance with reproducibility tracking...")

        episode_times = []
        episode_times_no_tracking = []

        for test_num in range(min(effective_num_tests, 5)):  # Fewer for episode tests
            # Create policy for performance testing
            policy = create_deterministic_policy(
                env, policy_seed=42, policy_type="random"
            )

            # Measure episode with full reproducibility tracking
            start_time = time.perf_counter()
            episode_with_tracking = execute_reproducibility_episode(
                env=env,
                policy=policy,
                episode_seed=12345 + test_num,
                max_steps=50,
                track_all_states=True,
                calculate_checksums=True,
            )
            episode_time_with_tracking = time.perf_counter() - start_time

            # Reset policy for comparison
            policy_minimal = create_deterministic_policy(
                env, policy_seed=42, policy_type="random"
            )

            # Measure episode with minimal tracking
            start_time = time.perf_counter()
            episode_minimal = execute_reproducibility_episode(
                env=env,
                policy=policy_minimal,
                episode_seed=12345 + test_num,
                max_steps=50,
                track_all_states=False,
                calculate_checksums=False,
            )
            episode_time_minimal = time.perf_counter() - start_time

            episode_times.append(episode_time_with_tracking * 1000)
            episode_times_no_tracking.append(episode_time_minimal * 1000)

        # Calculate reproducibility overhead
        tracking_overhead = np.mean(episode_times) - np.mean(episode_times_no_tracking)
        overhead_percentage = (
            tracking_overhead / np.mean(episode_times_no_tracking)
        ) * 100

        results["episode_performance"] = {
            "with_tracking": {
                "mean_ms": np.mean(episode_times),
                "std_ms": np.std(episode_times),
                "individual_times": episode_times,
            },
            "without_tracking": {
                "mean_ms": np.mean(episode_times_no_tracking),
                "std_ms": np.std(episode_times_no_tracking),
                "individual_times": episode_times_no_tracking,
            },
        }

        results["reproducibility_overhead"] = {
            "absolute_ms": tracking_overhead,
            "percentage": overhead_percentage,
            "acceptable": overhead_percentage < 50.0,  # Less than 50% overhead
        }

        _logger.info(
            f"Reproducibility tracking overhead: {tracking_overhead:.3f}ms ({overhead_percentage:.1f}%)"
        )

        # Memory analysis if requested
        if include_memory_analysis:
            _logger.info("Analyzing memory usage patterns...")

            import os

            import psutil

            process = psutil.Process(os.getpid())

            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Memory with SeedManager
            seed_manager = SeedManager()
            memory_with_seed_manager = process.memory_info().rss / 1024 / 1024

            # Memory with ReproducibilityTracker
            reproducibility_tracker = ReproducibilityTracker()

            # Add some episodes to tracker
            for i in range(10):
                reproducibility_tracker.record_episode(
                    episode_seed=12345 + i,
                    action_sequence=list(range(50)),
                    observation_sequence=[np.ones((64, 64)) for _ in range(50)],
                    episode_metadata={"test": True},
                )

            memory_with_tracker = process.memory_info().rss / 1024 / 1024

            results["memory_analysis"] = {
                "baseline_mb": baseline_memory,
                "with_seed_manager_mb": memory_with_seed_manager,
                "with_tracker_mb": memory_with_tracker,
                "seed_manager_overhead_mb": memory_with_seed_manager - baseline_memory,
                "tracker_overhead_mb": memory_with_tracker - memory_with_seed_manager,
                "total_overhead_mb": memory_with_tracker - baseline_memory,
            }

            _logger.info(
                f"Memory usage: baseline={baseline_memory:.1f}MB, with tracking={memory_with_tracker:.1f}MB"
            )

        # Performance target validation if requested
        if validate_performance_targets:
            _logger.info("Validating performance targets...")

            # Target: <1ms seeding overhead
            seeding_target_met = results["seeding_performance"]["target_1ms"]

            # Target: Reasonable tracking overhead
            tracking_overhead_acceptable = results["reproducibility_overhead"][
                "acceptable"
            ]

            # Memory target: <50MB additional memory for tracking
            if include_memory_analysis:
                memory_target_met = (
                    results["memory_analysis"]["total_overhead_mb"] < 50.0
                )
            else:
                memory_target_met = True  # Assume OK if not tested

            results["performance_validation"] = {
                "seeding_target_met": seeding_target_met,
                "tracking_overhead_acceptable": tracking_overhead_acceptable,
                "memory_target_met": memory_target_met,
                "all_targets_met": all(
                    [
                        seeding_target_met,
                        tracking_overhead_acceptable,
                        memory_target_met,
                    ]
                ),
            }

        # Generate optimization recommendations
        recommendations = []

        if results["seeding_performance"]["statistics"]["mean_ms"] > 1.0:
            recommendations.append(
                "Consider optimizing seed validation for <1ms target"
            )

        if results["reproducibility_overhead"]["percentage"] > 25.0:
            recommendations.append("Reduce reproducibility tracking overhead")

        if (
            include_memory_analysis
            and results["memory_analysis"]["total_overhead_mb"] > 25.0
        ):
            recommendations.append(
                "Optimize memory usage in reproducibility components"
            )

        if not recommendations:
            recommendations.append(
                "Performance characteristics are within acceptable ranges"
            )

        results["optimization_recommendations"] = recommendations

        # Log performance results
        _logger.info("=== REPRODUCIBILITY PERFORMANCE RESULTS ===")
        _logger.info(
            f"Average seeding time: {results['seeding_performance']['statistics']['mean_ms']:.3f}ms"
        )

        if validate_performance_targets:
            all_targets_met = results["performance_validation"]["all_targets_met"]
            _logger.info(f"Performance targets met: {'✓' if all_targets_met else '✗'}")

        _logger.info("Optimization recommendations:")
        for rec in recommendations:
            _logger.info(f"  - {rec}")

        return results

    except Exception as e:
        _logger.error(f"Performance demonstration failed: {e}")
        results["error"] = str(e)
        raise


def generate_comprehensive_reproducibility_report(
    test_results: Dict[str, Any],
    output_file: Optional[str] = None,
    report_format: Optional[str] = None,
    include_visualizations: bool = False,
    include_raw_data: bool = False,
) -> str:
    """
    Generate detailed reproducibility report with statistical analysis, validation results,
    performance metrics, research documentation, and actionable recommendations for
    scientific publication and workflow optimization.

    Args:
        test_results: Dictionary containing all test results from demonstrations
        output_file: Optional output file path for report
        report_format: Report format ('markdown', 'html', 'text'), defaults to 'markdown'
        include_visualizations: Whether to generate statistical plots
        include_raw_data: Whether to include raw episode data

    Returns:
        Generated comprehensive reproducibility report content
    """
    # Apply defaults
    effective_format = report_format or "markdown"

    _logger.info("Generating comprehensive reproducibility report...")
    _logger.info(f"Report format: {effective_format}")
    _logger.info(f"Include visualizations: {include_visualizations}")
    _logger.info(f"Include raw data: {include_raw_data}")

    # Initialize report content based on format
    if effective_format == "markdown":
        report_lines = [
            "# Comprehensive Reproducibility Analysis Report",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"**Environment:** PlumeNav-StaticGaussian-v0",
            "",
            "## Executive Summary",
            "",
        ]
    elif effective_format == "html":
        report_lines = [
            "<!DOCTYPE html>",
            "<html><head><title>Reproducibility Analysis Report</title></head><body>",
            "<h1>Comprehensive Reproducibility Analysis Report</h1>",
            f"<p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</p>",
            f"<p><strong>Environment:</strong> PlumeNav-StaticGaussian-v0</p>",
            "<h2>Executive Summary</h2>",
        ]
    else:  # text format
        report_lines = [
            "COMPREHENSIVE REPRODUCIBILITY ANALYSIS REPORT",
            "=" * 50,
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"Environment: PlumeNav-StaticGaussian-v0",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20,
            "",
        ]

    try:
        # Generate executive summary
        summary_items = []

        # Basic reproducibility results
        if "basic_reproducibility" in test_results:
            basic_result = test_results["basic_reproducibility"]
            if basic_result:
                summary_items.append("✓ Basic reproducibility validation PASSED")
            else:
                summary_items.append("✗ Basic reproducibility validation FAILED")

        # Advanced reproducibility results
        if "advanced_reproducibility" in test_results:
            advanced_result = test_results["advanced_reproducibility"]
            if isinstance(advanced_result, dict):
                status = advanced_result.get("validation_summary", {}).get(
                    "overall_status", "UNKNOWN"
                )
                summary_items.append(f"Advanced reproducibility: {status}")

                if "reproducibility_statistics" in advanced_result:
                    success_rate = advanced_result["reproducibility_statistics"].get(
                        "success_rate", 0
                    )
                    summary_items.append(f"Success rate: {success_rate:.1%}")

        # Cross-session results
        if "cross_session" in test_results:
            cross_session = test_results["cross_session"]
            if isinstance(cross_session, dict):
                validation_passed = cross_session.get("validation_summary", {}).get(
                    "validation_passed", False
                )
                status_symbol = "✓" if validation_passed else "✗"
                summary_items.append(f"{status_symbol} Cross-session validation")

        # Performance results
        if "performance" in test_results:
            performance = test_results["performance"]
            if isinstance(performance, dict):
                if "performance_validation" in performance:
                    targets_met = performance["performance_validation"].get(
                        "all_targets_met", False
                    )
                    status_symbol = "✓" if targets_met else "✗"
                    summary_items.append(f"{status_symbol} Performance targets")

        # Add summary items to report
        for item in summary_items:
            if effective_format == "markdown":
                report_lines.append(f"- {item}")
            elif effective_format == "html":
                report_lines.append(f"<li>{item}</li>")
            else:
                report_lines.append(f"  {item}")

        # Add section separator
        if effective_format == "markdown":
            report_lines.extend(["", "## Detailed Analysis", ""])
        elif effective_format == "html":
            report_lines.extend(["<h2>Detailed Analysis</h2>"])
        else:
            report_lines.extend(["", "DETAILED ANALYSIS", "-" * 20, ""])

        # Basic reproducibility section
        if "basic_reproducibility" in test_results:
            if effective_format == "markdown":
                report_lines.extend(["### Basic Reproducibility Test", ""])
            elif effective_format == "html":
                report_lines.extend(["<h3>Basic Reproducibility Test</h3>"])
            else:
                report_lines.extend(["Basic Reproducibility Test:", ""])

            basic_result = test_results["basic_reproducibility"]
            result_text = "PASSED" if basic_result else "FAILED"

            if effective_format == "markdown":
                report_lines.append(f"**Result:** {result_text}")
                report_lines.append("")
                report_lines.append(
                    "This test validates that identical seeds produce identical episodes."
                )
            elif effective_format == "html":
                report_lines.append(f"<p><strong>Result:</strong> {result_text}</p>")
                report_lines.append(
                    "<p>This test validates that identical seeds produce identical episodes.</p>"
                )
            else:
                report_lines.append(f"Result: {result_text}")
                report_lines.append(
                    "This test validates that identical seeds produce identical episodes."
                )

            report_lines.append("")

        # Advanced reproducibility section
        if "advanced_reproducibility" in test_results:
            if effective_format == "markdown":
                report_lines.extend(["### Advanced Reproducibility Analysis", ""])
            elif effective_format == "html":
                report_lines.extend(["<h3>Advanced Reproducibility Analysis</h3>"])
            else:
                report_lines.extend(["Advanced Reproducibility Analysis:", ""])

            advanced = test_results["advanced_reproducibility"]
            if isinstance(advanced, dict):
                # Test configuration
                config = advanced.get("test_configuration", {})
                seeds_tested = len(config.get("test_seeds", []))
                episodes_per_seed = config.get("episodes_per_seed", 0)

                if effective_format == "markdown":
                    report_lines.extend(
                        [
                            f"**Seeds tested:** {seeds_tested}",
                            f"**Episodes per seed:** {episodes_per_seed}",
                            "",
                        ]
                    )
                elif effective_format == "html":
                    report_lines.extend(
                        [
                            f"<p><strong>Seeds tested:</strong> {seeds_tested}</p>",
                            f"<p><strong>Episodes per seed:</strong> {episodes_per_seed}</p>",
                        ]
                    )
                else:
                    report_lines.extend(
                        [
                            f"Seeds tested: {seeds_tested}",
                            f"Episodes per seed: {episodes_per_seed}",
                            "",
                        ]
                    )

                # Statistics
                if "reproducibility_statistics" in advanced:
                    stats = advanced["reproducibility_statistics"]
                    success_rate = stats.get("success_rate", 0)
                    total_comparisons = stats.get("total_comparisons", 0)
                    successful = stats.get("successful_reproductions", 0)

                    if effective_format == "markdown":
                        report_lines.extend(
                            [
                                "**Statistics:**",
                                f"- Success rate: {success_rate:.1%}",
                                f"- Total comparisons: {total_comparisons}",
                                f"- Successful reproductions: {successful}",
                                "",
                            ]
                        )
                    elif effective_format == "html":
                        report_lines.extend(
                            [
                                "<p><strong>Statistics:</strong></p>",
                                "<ul>",
                                f"<li>Success rate: {success_rate:.1%}</li>",
                                f"<li>Total comparisons: {total_comparisons}</li>",
                                f"<li>Successful reproductions: {successful}</li>",
                                "</ul>",
                            ]
                        )
                    else:
                        report_lines.extend(
                            [
                                "Statistics:",
                                f"  Success rate: {success_rate:.1%}",
                                f"  Total comparisons: {total_comparisons}",
                                f"  Successful reproductions: {successful}",
                                "",
                            ]
                        )

                # Validation summary
                if "validation_summary" in advanced:
                    validation = advanced["validation_summary"]
                    overall_status = validation.get("overall_status", "UNKNOWN")
                    reproducibility_achieved = validation.get(
                        "reproducibility_achieved", False
                    )

                    if effective_format == "markdown":
                        report_lines.extend(
                            [
                                "**Validation Summary:**",
                                f"- Overall status: {overall_status}",
                                f"- Reproducibility achieved: {'Yes' if reproducibility_achieved else 'No'}",
                                "",
                            ]
                        )
                    elif effective_format == "html":
                        report_lines.extend(
                            [
                                "<p><strong>Validation Summary:</strong></p>",
                                "<ul>",
                                f"<li>Overall status: {overall_status}</li>",
                                f"<li>Reproducibility achieved: {'Yes' if reproducibility_achieved else 'No'}</li>",
                                "</ul>",
                            ]
                        )
                    else:
                        report_lines.extend(
                            [
                                "Validation Summary:",
                                f"  Overall status: {overall_status}",
                                f"  Reproducibility achieved: {'Yes' if reproducibility_achieved else 'No'}",
                                "",
                            ]
                        )

        # Performance analysis section
        if "performance" in test_results:
            if effective_format == "markdown":
                report_lines.extend(["### Performance Analysis", ""])
            elif effective_format == "html":
                report_lines.extend(["<h3>Performance Analysis</h3>"])
            else:
                report_lines.extend(["Performance Analysis:", ""])

            performance = test_results["performance"]
            if isinstance(performance, dict):
                # Seeding performance
                if "seeding_performance" in performance:
                    seeding = performance["seeding_performance"]
                    if "statistics" in seeding:
                        stats = seeding["statistics"]
                        mean_ms = stats.get("mean_ms", 0)
                        target_met = seeding.get("target_1ms", False)

                        if effective_format == "markdown":
                            report_lines.extend(
                                [
                                    "**Seeding Performance:**",
                                    f"- Average seeding time: {mean_ms:.3f}ms",
                                    f"- Target (<1ms) met: {'Yes' if target_met else 'No'}",
                                    "",
                                ]
                            )
                        elif effective_format == "html":
                            report_lines.extend(
                                [
                                    "<p><strong>Seeding Performance:</strong></p>",
                                    "<ul>",
                                    f"<li>Average seeding time: {mean_ms:.3f}ms</li>",
                                    f"<li>Target (&lt;1ms) met: {'Yes' if target_met else 'No'}</li>",
                                    "</ul>",
                                ]
                            )
                        else:
                            report_lines.extend(
                                [
                                    "Seeding Performance:",
                                    f"  Average seeding time: {mean_ms:.3f}ms",
                                    f"  Target (<1ms) met: {'Yes' if target_met else 'No'}",
                                    "",
                                ]
                            )

                # Reproducibility overhead
                if "reproducibility_overhead" in performance:
                    overhead = performance["reproducibility_overhead"]
                    absolute_ms = overhead.get("absolute_ms", 0)
                    percentage = overhead.get("percentage", 0)
                    acceptable = overhead.get("acceptable", False)

                    if effective_format == "markdown":
                        report_lines.extend(
                            [
                                "**Reproducibility Tracking Overhead:**",
                                f"- Absolute overhead: {absolute_ms:.3f}ms",
                                f"- Percentage overhead: {percentage:.1f}%",
                                f"- Acceptable: {'Yes' if acceptable else 'No'}",
                                "",
                            ]
                        )
                    elif effective_format == "html":
                        report_lines.extend(
                            [
                                "<p><strong>Reproducibility Tracking Overhead:</strong></p>",
                                "<ul>",
                                f"<li>Absolute overhead: {absolute_ms:.3f}ms</li>",
                                f"<li>Percentage overhead: {percentage:.1f}%</li>",
                                f"<li>Acceptable: {'Yes' if acceptable else 'No'}</li>",
                                "</ul>",
                            ]
                        )
                    else:
                        report_lines.extend(
                            [
                                "Reproducibility Tracking Overhead:",
                                f"  Absolute overhead: {absolute_ms:.3f}ms",
                                f"  Percentage overhead: {percentage:.1f}%",
                                f"  Acceptable: {'Yes' if acceptable else 'No'}",
                                "",
                            ]
                        )

        # Recommendations section
        all_recommendations = []

        # Collect recommendations from all test results
        for test_name, test_data in test_results.items():
            if isinstance(test_data, dict):
                if "recommendations" in test_data:
                    all_recommendations.extend(test_data["recommendations"])
                elif (
                    "validation_summary" in test_data
                    and "recommendations" in test_data["validation_summary"]
                ):
                    all_recommendations.extend(
                        test_data["validation_summary"]["recommendations"]
                    )
                elif "optimization_recommendations" in test_data:
                    all_recommendations.extend(
                        test_data["optimization_recommendations"]
                    )

        if all_recommendations:
            if effective_format == "markdown":
                report_lines.extend(["## Recommendations", ""])
            elif effective_format == "html":
                report_lines.extend(["<h2>Recommendations</h2>"])
            else:
                report_lines.extend(["RECOMMENDATIONS", "-" * 15, ""])

            # Remove duplicates while preserving order
            unique_recommendations = []
            for rec in all_recommendations:
                if rec not in unique_recommendations:
                    unique_recommendations.append(rec)

            for rec in unique_recommendations:
                if effective_format == "markdown":
                    report_lines.append(f"- {rec}")
                elif effective_format == "html":
                    report_lines.append(f"<li>{rec}</li>")
                else:
                    report_lines.append(f"  - {rec}")

            if effective_format == "html" and unique_recommendations:
                report_lines.insert(-len(unique_recommendations), "<ul>")
                report_lines.append("</ul>")

            report_lines.append("")

        # Add raw data section if requested
        if include_raw_data:
            if effective_format == "markdown":
                report_lines.extend(["## Raw Data", "", "```json"])
            elif effective_format == "html":
                report_lines.extend(["<h2>Raw Data</h2>", "<pre><code>"])
            else:
                report_lines.extend(["RAW DATA", "-" * 10, ""])

            # Add sanitized raw data (remove large arrays)
            sanitized_results = {}
            for key, value in test_results.items():
                if isinstance(value, dict):
                    sanitized_value = {}
                    for sub_key, sub_value in value.items():
                        # Skip large data structures
                        if sub_key in [
                            "observation_sequence",
                            "state_transitions",
                            "individual_tests",
                        ]:
                            sanitized_value[sub_key] = (
                                f"<{len(sub_value)} items>"
                                if isinstance(sub_value, list)
                                else "<data_truncated>"
                            )
                        else:
                            sanitized_value[sub_key] = sub_value
                    sanitized_results[key] = sanitized_value
                else:
                    sanitized_results[key] = value

            raw_data_json = json.dumps(sanitized_results, indent=2)
            report_lines.append(raw_data_json)

            if effective_format == "markdown":
                report_lines.append("```")
            elif effective_format == "html":
                report_lines.append("</code></pre>")

        # Close HTML if needed
        if effective_format == "html":
            report_lines.append("</body></html>")

        # Join report lines
        report_content = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report_content)

                _logger.info(f"Report saved to: {output_path}")
                _logger.info(f"Report size: {len(report_content)} characters")

            except Exception as e:
                _logger.error(f"Failed to save report to {output_file}: {e}")

        # Generate visualizations if requested
        if include_visualizations:
            _logger.info("Generating visualizations...")

            try:
                # Create visualization plots (requires matplotlib)
                fig_dir = Path(REPORT_OUTPUT_DIR) / "visualizations"
                fig_dir.mkdir(parents=True, exist_ok=True)

                # Performance timeline plot
                if "performance" in test_results:
                    performance = test_results["performance"]
                    if (
                        "seeding_performance" in performance
                        and "individual_tests" in performance["seeding_performance"]
                    ):
                        seeding_tests = performance["seeding_performance"][
                            "individual_tests"
                        ]

                        plt.figure(figsize=(10, 6))
                        test_nums = [t["test_num"] for t in seeding_tests]
                        total_times = [t["total_seeding"] for t in seeding_tests]

                        plt.plot(
                            test_nums, total_times, "o-", label="Total Seeding Time"
                        )
                        plt.axhline(
                            y=1.0, color="r", linestyle="--", label="1ms Target"
                        )
                        plt.xlabel("Test Number")
                        plt.ylabel("Seeding Time (ms)")
                        plt.title("Seeding Performance Over Time")
                        plt.legend()
                        plt.grid(True, alpha=0.3)

                        plt.tight_layout()
                        plt.savefig(fig_dir / "seeding_performance.png", dpi=150)
                        plt.close()

                        _logger.info(
                            f"Saved seeding performance plot: {fig_dir / 'seeding_performance.png'}"
                        )

                # Success rate visualization
                if "advanced_reproducibility" in test_results:
                    advanced = test_results["advanced_reproducibility"]
                    if "seed_results" in advanced:
                        seed_results = advanced["seed_results"]

                        seeds = list(seed_results.keys())
                        reproducible = [
                            seed_results[seed]["reproducible"] for seed in seeds
                        ]

                        plt.figure(figsize=(8, 6))
                        success_counts = [1 if r else 0 for r in reproducible]
                        plt.bar(
                            range(len(seeds)),
                            success_counts,
                            color=["green" if r else "red" for r in reproducible],
                        )
                        plt.xlabel("Test Seeds")
                        plt.ylabel("Reproducible (1=Yes, 0=No)")
                        plt.title("Reproducibility by Seed")
                        plt.xticks(range(len(seeds)), [str(s) for s in seeds])
                        plt.ylim(0, 1.2)

                        for i, (seed, repro) in enumerate(zip(seeds, reproducible)):
                            plt.text(
                                i,
                                0.1 if repro else 0.9,
                                "✓" if repro else "✗",
                                ha="center",
                                fontsize=12,
                                fontweight="bold",
                            )

                        plt.tight_layout()
                        plt.savefig(fig_dir / "reproducibility_by_seed.png", dpi=150)
                        plt.close()

                        _logger.info(
                            f"Saved reproducibility plot: {fig_dir / 'reproducibility_by_seed.png'}"
                        )

            except Exception as e:
                _logger.warning(f"Failed to generate visualizations: {e}")

        _logger.info("Reproducibility report generation completed")
        return report_content

    except Exception as e:
        _logger.error(f"Report generation failed: {e}")
        return f"Report generation failed: {e}"


def run_reproducibility_demo(demo_config: Optional[Dict[str, Any]] = None) -> int:
    """
    Execute comprehensive reproducibility demonstration coordinating all reproducibility
    components with environment setup, validation testing, performance analysis, and
    comprehensive reporting for educational and research purposes.

    Args:
        demo_config: Optional configuration dictionary for demonstration parameters

    Returns:
        Exit status code: 0 for success, 1 for failures
    """
    # Parse demonstration configuration
    config = demo_config or {}

    # Extract configuration parameters with defaults
    log_level = config.get("log_level", "INFO")
    log_file = config.get("log_file")
    test_seeds = config.get("test_seeds", REPRODUCIBILITY_TEST_SEEDS)
    episodes_per_seed = config.get("episodes_per_seed", 2)
    validate_performance = config.get("validate_performance", True)
    generate_report = config.get("generate_report", True)
    report_format = config.get("report_format", "markdown")
    include_visualizations = config.get("include_visualizations", False)

    try:
        # Set up comprehensive logging
        setup_reproducibility_logging(
            log_level=log_level, log_file=log_file, include_timestamps=True
        )

        _logger.info("=" * 60)
        _logger.info("PLUME NAVIGATION REPRODUCIBILITY DEMONSTRATION")
        _logger.info("=" * 60)
        _logger.info("This comprehensive demo validates all aspects of reproducibility")
        _logger.info(
            f"Configuration: seeds={len(test_seeds)}, episodes_per_seed={episodes_per_seed}"
        )

        # Initialize results dictionary
        all_results = {"demo_start_time": time.time(), "configuration": config}

        # Register environment with error handling
        _logger.info("Registering plume navigation environment...")
        try:
            env_id = register_env()
            _logger.info(f"Environment registered successfully: {env_id}")
        except Exception as e:
            _logger.error(f"Environment registration failed: {e}")
            return 1

        # Create environment instance
        _logger.info("Creating environment instance...")
        try:
            env = gymnasium.make(env_id)
            _logger.info("Environment created successfully")
        except Exception as e:
            _logger.error(f"Environment creation failed: {e}")
            return 1

        # Execute basic reproducibility demonstration
        _logger.info("\n" + "=" * 40)
        _logger.info("PHASE 1: BASIC REPRODUCIBILITY")
        _logger.info("=" * 40)

        try:
            basic_result = demonstrate_basic_reproducibility(
                env=env,
                demo_seed=DEFAULT_TEST_SEED,
                episode_length=DEFAULT_EPISODE_LENGTH,
                show_detailed_output=config.get("show_detailed_output", False),
            )
            all_results["basic_reproducibility"] = basic_result

            if basic_result:
                _logger.info("✓ Basic reproducibility demonstration PASSED")
            else:
                _logger.warning("⚠ Basic reproducibility demonstration had issues")

        except Exception as e:
            _logger.error(f"Basic reproducibility demonstration failed: {e}")
            all_results["basic_reproducibility"] = False

        # Execute advanced reproducibility demonstration
        _logger.info("\n" + "=" * 40)
        _logger.info("PHASE 2: ADVANCED REPRODUCIBILITY")
        _logger.info("=" * 40)

        try:
            advanced_result = demonstrate_advanced_reproducibility(
                env=env,
                test_seeds=test_seeds,
                num_episodes_per_seed=episodes_per_seed,
                use_seed_manager=True,
                use_reproducibility_tracker=True,
            )
            all_results["advanced_reproducibility"] = advanced_result

            status = advanced_result.get("validation_summary", {}).get(
                "overall_status", "UNKNOWN"
            )
            if status == "PASS":
                _logger.info("✓ Advanced reproducibility demonstration PASSED")
            else:
                _logger.warning(f"⚠ Advanced reproducibility demonstration: {status}")

        except Exception as e:
            _logger.error(f"Advanced reproducibility demonstration failed: {e}")
            all_results["advanced_reproducibility"] = {
                "validation_summary": {"overall_status": "ERROR"}
            }

        # Execute string-based seed reproducibility demonstration
        _logger.info("\n" + "=" * 40)
        _logger.info("PHASE 3: STRING-BASED SEED REPRODUCIBILITY")
        _logger.info("=" * 40)

        try:
            string_seed_result = demonstrate_seed_string_reproducibility(
                env=env,
                experiment_names=DEMO_EXPERIMENT_NAMES,
                episodes_per_experiment=2,
            )
            all_results["string_seed_reproducibility"] = string_seed_result

            validation_passed = string_seed_result.get(
                "reproducibility_validation", {}
            ).get("validation_passed", False)
            if validation_passed:
                _logger.info("✓ String-based seed reproducibility demonstration PASSED")
            else:
                _logger.warning("⚠ String-based seed reproducibility had issues")

        except Exception as e:
            _logger.error(f"String-based seed demonstration failed: {e}")
            all_results["string_seed_reproducibility"] = {
                "reproducibility_validation": {"validation_passed": False}
            }

        # Execute cross-session reproducibility validation
        _logger.info("\n" + "=" * 40)
        _logger.info("PHASE 4: CROSS-SESSION REPRODUCIBILITY")
        _logger.info("=" * 40)

        try:
            cross_session_result = validate_cross_session_reproducibility(
                base_seed=DEFAULT_TEST_SEED,
                num_validation_episodes=3,
                save_session_data=True,
                compare_with_previous=True,
            )
            all_results["cross_session"] = cross_session_result

            validation_passed = cross_session_result.get("validation_summary", {}).get(
                "validation_passed", False
            )
            if validation_passed:
                _logger.info("✓ Cross-session reproducibility validation PASSED")
            else:
                _logger.warning("⚠ Cross-session reproducibility had issues")

        except Exception as e:
            _logger.error(f"Cross-session validation failed: {e}")
            all_results["cross_session"] = {
                "validation_summary": {"validation_passed": False}
            }

        # Execute reproducibility performance analysis
        if validate_performance:
            _logger.info("\n" + "=" * 40)
            _logger.info("PHASE 5: PERFORMANCE ANALYSIS")
            _logger.info("=" * 40)

            try:
                performance_result = demonstrate_reproducibility_performance(
                    env=env,
                    num_performance_tests=config.get("performance_tests", 10),
                    validate_performance_targets=True,
                    include_memory_analysis=config.get(
                        "include_memory_analysis", False
                    ),
                )
                all_results["performance"] = performance_result

                if "performance_validation" in performance_result:
                    targets_met = performance_result["performance_validation"].get(
                        "all_targets_met", False
                    )
                    if targets_met:
                        _logger.info("✓ Performance analysis: All targets met")
                    else:
                        _logger.warning("⚠ Performance analysis: Some targets not met")

            except Exception as e:
                _logger.error(f"Performance analysis failed: {e}")
                all_results["performance"] = {
                    "performance_validation": {"all_targets_met": False}
                }

        # Close environment
        try:
            env.close()
            _logger.info("Environment closed successfully")
        except Exception as e:
            _logger.warning(f"Environment cleanup warning: {e}")

        # Generate comprehensive report
        if generate_report:
            _logger.info("\n" + "=" * 40)
            _logger.info("PHASE 6: REPORT GENERATION")
            _logger.info("=" * 40)

            try:
                # Prepare report output path
                report_dir = Path(REPORT_OUTPUT_DIR)
                report_dir.mkdir(parents=True, exist_ok=True)

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                report_filename = f"reproducibility_report_{timestamp}.{report_format}"
                report_path = report_dir / report_filename

                # Generate comprehensive report
                report_content = generate_comprehensive_reproducibility_report(
                    test_results=all_results,
                    output_file=str(report_path),
                    report_format=report_format,
                    include_visualizations=include_visualizations,
                    include_raw_data=config.get("include_raw_data", False),
                )

                _logger.info(f"Comprehensive report generated: {report_path}")

                # Log report summary to console
                report_summary_lines = report_content.split("\n")[:20]  # First 20 lines
                _logger.info("Report summary preview:")
                for line in report_summary_lines:
                    if line.strip():
                        _logger.info(f"  {line}")

            except Exception as e:
                _logger.error(f"Report generation failed: {e}")

        # Record demonstration completion
        all_results["demo_end_time"] = time.time()
        all_results["demo_duration"] = (
            all_results["demo_end_time"] - all_results["demo_start_time"]
        )

        # Log comprehensive summary
        _logger.info("\n" + "=" * 60)
        _logger.info("REPRODUCIBILITY DEMONSTRATION COMPLETED")
        _logger.info("=" * 60)

        # Analyze overall results
        results_summary = []

        basic_passed = all_results.get("basic_reproducibility", False)
        results_summary.append(
            f"Basic reproducibility: {'PASS' if basic_passed else 'FAIL'}"
        )

        advanced_status = (
            all_results.get("advanced_reproducibility", {})
            .get("validation_summary", {})
            .get("overall_status", "UNKNOWN")
        )
        results_summary.append(f"Advanced reproducibility: {advanced_status}")

        string_seed_passed = (
            all_results.get("string_seed_reproducibility", {})
            .get("reproducibility_validation", {})
            .get("validation_passed", False)
        )
        results_summary.append(
            f"String-based seeds: {'PASS' if string_seed_passed else 'FAIL'}"
        )

        cross_session_passed = (
            all_results.get("cross_session", {})
            .get("validation_summary", {})
            .get("validation_passed", False)
        )
        results_summary.append(
            f"Cross-session: {'PASS' if cross_session_passed else 'FAIL'}"
        )

        if validate_performance:
            performance_passed = (
                all_results.get("performance", {})
                .get("performance_validation", {})
                .get("all_targets_met", False)
            )
            results_summary.append(
                f"Performance: {'PASS' if performance_passed else 'FAIL'}"
            )

        # Log results
        _logger.info("FINAL RESULTS:")
        for result in results_summary:
            _logger.info(f"  {result}")

        # Calculate overall success
        critical_tests = [basic_passed, advanced_status == "PASS"]
        overall_success = all(critical_tests)

        if overall_success:
            _logger.info(
                "🎉 OVERALL STATUS: SUCCESS - Reproducibility fully validated!"
            )
            _logger.info(
                "The plume_nav_sim environment demonstrates excellent reproducibility characteristics."
            )
        else:
            _logger.warning("⚠ OVERALL STATUS: PARTIAL SUCCESS - Some issues detected.")
            _logger.warning(
                "Review detailed results and recommendations for improvements."
            )

        _logger.info(
            f"Total demonstration time: {all_results['demo_duration']:.2f} seconds"
        )

        if generate_report:
            _logger.info(f"Detailed analysis available in: {REPORT_OUTPUT_DIR}")

        _logger.info("Reproducibility demonstration complete.")

        # Return appropriate exit code
        return 0 if overall_success else 1

    except Exception as e:
        _logger.error(f"Reproducibility demonstration failed: {e}")
        _logger.error(
            "This indicates a serious issue with the environment or demonstration script."
        )
        return 1


def main() -> None:
    """
    Main entry point for reproducibility demonstration script with command-line interface,
    comprehensive argument parsing, and scientific workflow integration with error handling
    and research guidance.
    """
    try:
        # Set up comprehensive argument parser
        parser = argparse.ArgumentParser(
            description="Comprehensive reproducibility demonstration for plume_nav_sim environment",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python reproducibility_demo.py
  python reproducibility_demo.py --seeds 42,123,456 --episodes-per-seed 3
  python reproducibility_demo.py --performance-tests 20 --validate-targets
  python reproducibility_demo.py --detailed-output --include-visualizations
  python reproducibility_demo.py --log-level DEBUG --log-file demo.log

This demonstration validates all aspects of reproducibility in the plume navigation
environment including basic seeding, advanced validation, cross-session consistency,
and performance characteristics suitable for scientific research workflows.
            """,
        )

        # Reproducibility testing parameters
        parser.add_argument(
            "--seeds",
            type=str,
            default=",".join(map(str, REPRODUCIBILITY_TEST_SEEDS)),
            help=f'Comma-separated list of seeds for testing (default: {",".join(map(str, REPRODUCIBILITY_TEST_SEEDS))})',
        )

        parser.add_argument(
            "--episodes-per-seed",
            type=int,
            default=2,
            help="Number of episodes to run per seed for reproducibility validation (default: 2)",
        )

        parser.add_argument(
            "--episode-length",
            type=int,
            default=DEFAULT_EPISODE_LENGTH,
            help=f"Maximum steps per episode (default: {DEFAULT_EPISODE_LENGTH})",
        )

        # Performance testing parameters
        parser.add_argument(
            "--performance-tests",
            type=int,
            default=10,
            help="Number of performance tests to run (default: 10)",
        )

        parser.add_argument(
            "--validate-targets",
            action="store_true",
            help="Validate performance targets (<1ms seeding, reasonable overhead)",
        )

        parser.add_argument(
            "--include-memory-analysis",
            action="store_true",
            help="Include memory usage analysis in performance testing",
        )

        # Output and logging parameters
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Logging level (default: INFO)",
        )

        parser.add_argument(
            "--log-file", type=str, help="Optional log file path for persistent logging"
        )

        parser.add_argument(
            "--detailed-output",
            action="store_true",
            help="Show detailed step-by-step output during demonstrations",
        )

        # Report generation parameters
        parser.add_argument(
            "--no-report",
            action="store_true",
            help="Skip comprehensive report generation",
        )

        parser.add_argument(
            "--report-format",
            choices=["markdown", "html", "text"],
            default="markdown",
            help="Report output format (default: markdown)",
        )

        parser.add_argument(
            "--include-visualizations",
            action="store_true",
            help="Generate statistical plots and visualizations",
        )

        parser.add_argument(
            "--include-raw-data",
            action="store_true",
            help="Include raw episode data in report",
        )

        # Development and debugging parameters
        parser.add_argument(
            "--quick-test",
            action="store_true",
            help="Run abbreviated tests for quick validation",
        )

        parser.add_argument(
            "--test-basic-only",
            action="store_true",
            help="Run only basic reproducibility test",
        )

        # Parse command-line arguments
        args = parser.parse_args()

        # Parse seeds list
        try:
            test_seeds = [int(s.strip()) for s in args.seeds.split(",")]
        except ValueError as e:
            print(f"Error parsing seeds: {e}")
            sys.exit(1)

        # Create demonstration configuration
        demo_config = {
            "log_level": args.log_level,
            "log_file": args.log_file,
            "test_seeds": test_seeds,
            "episodes_per_seed": args.episodes_per_seed,
            "episode_length": args.episode_length,
            "validate_performance": not args.test_basic_only,
            "performance_tests": args.performance_tests,
            "validate_targets": args.validate_targets,
            "include_memory_analysis": args.include_memory_analysis,
            "show_detailed_output": args.detailed_output,
            "generate_report": not args.no_report,
            "report_format": args.report_format,
            "include_visualizations": args.include_visualizations,
            "include_raw_data": args.include_raw_data,
        }

        # Apply quick test modifications
        if args.quick_test:
            demo_config.update(
                {
                    "test_seeds": test_seeds[:2],  # Only first 2 seeds
                    "episodes_per_seed": 1,
                    "episode_length": 25,
                    "performance_tests": 3,
                    "include_memory_analysis": False,
                }
            )

        # Apply basic-only test modifications
        if args.test_basic_only:
            demo_config.update(
                {
                    "test_seeds": [DEFAULT_TEST_SEED],
                    "episodes_per_seed": 1,
                    "validate_performance": False,
                }
            )

        # Set up global exception handling
        def handle_unexpected_error(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                print("\n\nDemonstration interrupted by user.")
                sys.exit(1)
            else:
                print(f"\nUnexpected error occurred: {exc_type.__name__}: {exc_value}")
                print(
                    "This may indicate an issue with the environment or demonstration script."
                )
                print("Please check the log output for detailed error information.")
                sys.exit(1)

        sys.excepthook = handle_unexpected_error

        # Execute demonstration
        exit_code = run_reproducibility_demo(demo_config)

        # Provide user guidance based on results
        if exit_code == 0:
            print("\n🎉 Reproducibility demonstration completed successfully!")
            print(
                "The plume_nav_sim environment demonstrates excellent reproducibility."
            )

            if not args.no_report:
                print(f"\n📊 Detailed analysis available in: {REPORT_OUTPUT_DIR}/")

            print("\n✨ Research Workflow Integration:")
            print("  - Use identical seeds for reproducible experiments")
            print("  - Leverage string-based seed generation for experiment naming")
            print("  - Implement cross-session validation for long-term studies")
            print("  - Monitor performance characteristics for scalability")

        else:
            print("\n⚠️  Reproducibility demonstration completed with issues.")
            print("Some reproducibility tests did not pass as expected.")

            if not args.no_report:
                print(f"\n🔍 Review detailed results in: {REPORT_OUTPUT_DIR}/")

            print("\n🛠️  Troubleshooting Steps:")
            print("  1. Check environment seeding implementation")
            print("  2. Verify deterministic policy behavior")
            print("  3. Review error logs for specific failures")
            print(
                "  4. Consider running with --log-level DEBUG for detailed diagnostics"
            )

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\nFailed to start demonstration: {e}")
        print("Check your Python environment and dependencies.")
        sys.exit(1)


if __name__ == "__main__":
    main()
