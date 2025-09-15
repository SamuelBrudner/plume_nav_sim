"""
Basic usage example script demonstrating essential plume navigation environment functionality 
including environment registration, instantiation, episode lifecycle, random agent interaction, 
and basic rendering. Provides educational demonstration of Gymnasium-compatible API usage with 
comprehensive error handling, performance logging, and cleanup procedures for new users and 
researchers getting started with plume_nav_sim package.

This module serves as a comprehensive tutorial and validation script for the PlumeNav-StaticGaussian-v0 
environment, showcasing proper usage patterns for environment registration, lifecycle management, 
action space exploration, rendering capabilities, seeding for reproducibility, and robust error 
handling techniques essential for reinforcement learning research workflows.
"""

# External imports with version comments
import gymnasium  # >=0.29.0 - Reinforcement learning environment framework for gym.make() calls and standard RL API usage
import numpy as np  # >=2.1.0 - Array operations and random number generation for action selection and data analysis
import time  # >=3.10 - Performance timing measurements and demonstration of step latency monitoring
import logging  # >=3.10 - Example logging setup and operation tracking for educational debugging demonstration
import sys  # >=3.10 - System interface for exit handling and error status reporting in example scripts

# Internal imports for environment functionality and configuration
from plume_nav_sim.registration.register import (
    register_env,  # Environment registration function for Gymnasium compatibility enabling gym.make() instantiation
    ENV_ID  # Environment identifier constant 'PlumeNav-StaticGaussian-v0' for gym.make() calls
)
from plume_nav_sim.core.types import Action  # Action enumeration for demonstrating discrete action space usage and movement directions
from plume_nav_sim.utils.exceptions import (
    ValidationError,  # Exception handling for input validation failures and error demonstration
    PlumeNavSimError  # Base exception handling for comprehensive error management and user guidance
)

# Global demonstration constants for example configuration and reproducibility
EXAMPLE_SEED = 42  # Default seed value for reproducible example execution and scientific validation
NUM_DEMO_STEPS = 50  # Number of steps to demonstrate in basic episode for educational timing
PERFORMANCE_TARGET_MS = 1.0  # Performance target threshold in milliseconds for step execution monitoring
_logger = logging.getLogger(__name__)  # Logger instance for example script with hierarchical naming


def setup_example_logging(log_level: str = None) -> None:
    """
    Configure logging for basic usage example with appropriate level and format for educational 
    demonstration and debugging support including timestamp formatting and clear message structure.
    
    Args:
        log_level: Optional logging level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                  Defaults to INFO for educational visibility unless specified otherwise
    
    Returns:
        None: Sets up logging configuration for example execution with console handler
    """
    # Set default log level to INFO for educational visibility unless log_level specified
    effective_log_level = log_level or 'INFO'
    
    # Convert string log level to logging module constant with validation
    try:
        numeric_level = getattr(logging, effective_log_level.upper())
    except AttributeError:
        numeric_level = logging.INFO
        print(f"Warning: Invalid log level '{effective_log_level}', using INFO")
    
    # Configure basic logging with timestamp, level, and message format for clear output
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Override any existing logging configuration
    )
    
    # Create logger instance for basic usage example with appropriate handler setup
    global _logger
    _logger = logging.getLogger(__name__)
    
    # Log example initialization with configuration details for user orientation
    _logger.info(f"Basic usage example logging initialized with level: {effective_log_level}")
    _logger.info(f"Example configuration - Seed: {EXAMPLE_SEED}, Demo steps: {NUM_DEMO_STEPS}")


def demonstrate_environment_registration() -> bool:
    """
    Demonstrate environment registration process with error handling and validation for educational 
    understanding of Gymnasium integration including registry inspection and usage instructions.
    
    Returns:
        bool: True if registration successful, False if registration failed with detailed error information
    """
    # Log registration demonstration start with environment ID information
    _logger.info("=" * 60)
    _logger.info("DEMONSTRATION: Environment Registration")
    _logger.info(f"Registering environment with ID: {ENV_ID}")
    
    try:
        # Call register_env() function to register PlumeNav-StaticGaussian-v0 with Gymnasium
        _logger.info("Calling register_env() to register environment with Gymnasium registry...")
        register_env()
        
        # Verify successful registration using gymnasium registry inspection
        _logger.info("Verifying registration in Gymnasium registry...")
        if ENV_ID in gymnasium.envs.registration.registry:
            # Log registration success with environment ID and usage instructions
            _logger.info(f"✓ Environment {ENV_ID} successfully registered!")
            _logger.info(f"Environment can now be created using: gymnasium.make('{ENV_ID}')")
            _logger.info("Registration includes entry point and configuration parameters")
            return True
        else:
            _logger.error(f"✗ Environment {ENV_ID} not found in registry after registration")
            return False
            
    except Exception as e:
        # Handle registration errors gracefully with informative error messages
        _logger.error(f"✗ Environment registration failed: {e}")
        _logger.error("Troubleshooting suggestions:")
        _logger.error("  1. Verify plume_nav_sim package is properly installed")
        _logger.error("  2. Check for import path conflicts")
        _logger.error("  3. Ensure all dependencies are available")
        return False


def demonstrate_environment_creation(render_mode: str = None) -> gymnasium.Env:
    """
    Demonstrate environment instantiation using gym.make() with error handling and configuration 
    validation for educational API usage including action space and observation space inspection.
    
    Args:
        render_mode: Optional render mode ('human', 'rgb_array', or None)
                    Defaults to None for headless operation during basic demonstration
    
    Returns:
        gymnasium.Env: Environment instance if creation successful, None if failed with error details
    """
    # Log environment creation demonstration with ENV_ID and render_mode information
    _logger.info("=" * 60)
    _logger.info("DEMONSTRATION: Environment Creation")
    _logger.info(f"Creating environment using gym.make('{ENV_ID}', render_mode='{render_mode}')")
    
    try:
        # Create environment using gym.make(ENV_ID, render_mode=render_mode) with parameter validation
        _logger.info("Instantiating environment through Gymnasium interface...")
        env = gymnasium.make(ENV_ID, render_mode=render_mode)
        
        # Validate environment action space and observation space for API compliance demonstration
        _logger.info("Environment created successfully! Inspecting environment properties:")
        _logger.info(f"  Action space: {env.action_space}")
        _logger.info(f"  Observation space: {env.observation_space}")
        _logger.info(f"  Render mode: {render_mode}")
        
        # Additional environment metadata inspection for educational context
        if hasattr(env, 'spec') and env.spec:
            _logger.info(f"  Environment spec ID: {env.spec.id}")
            _logger.info(f"  Max episode steps: {getattr(env.spec, 'max_episode_steps', 'Unlimited')}")
        
        # Log successful environment creation with space information and configuration details
        _logger.info("✓ Environment instantiation completed successfully")
        _logger.info("Environment ready for reset() and step() operations")
        
        return env
        
    except Exception as e:
        # Handle creation errors with detailed error messages and troubleshooting guidance
        _logger.error(f"✗ Environment creation failed: {e}")
        _logger.error("Troubleshooting guidance:")
        _logger.error("  1. Ensure environment is properly registered first")
        _logger.error("  2. Check render_mode parameter validity")
        _logger.error("  3. Verify all dependencies are installed")
        _logger.error("  4. Try creating without render_mode parameter")
        return None


def demonstrate_basic_episode(env: gymnasium.Env, seed: int = None, max_steps: int = None) -> dict:
    """
    Demonstrate complete episode lifecycle including reset, step loop, and termination handling 
    with performance monitoring and educational logging for comprehensive API usage understanding.
    
    Args:
        env: Gymnasium environment instance for episode demonstration
        seed: Optional seed for reproducible episode generation (uses EXAMPLE_SEED if not provided)
        max_steps: Optional maximum steps override (uses NUM_DEMO_STEPS if not provided)
    
    Returns:
        dict: Episode statistics including steps, reward, completion status, and performance metrics
    """
    # Apply default seed and max_steps using EXAMPLE_SEED and NUM_DEMO_STEPS if not provided
    effective_seed = seed if seed is not None else EXAMPLE_SEED
    effective_max_steps = max_steps if max_steps is not None else NUM_DEMO_STEPS
    
    # Log episode demonstration start with seed and configuration information
    _logger.info("=" * 60)
    _logger.info("DEMONSTRATION: Basic Episode Lifecycle")
    _logger.info(f"Episode parameters - Seed: {effective_seed}, Max steps: {effective_max_steps}")
    
    # Initialize episode statistics tracking including step count, total reward, and timing
    episode_stats = {
        'steps_taken': 0,
        'total_reward': 0.0,
        'episode_completed': False,
        'termination_reason': None,
        'average_step_time_ms': 0.0,
        'total_episode_time_ms': 0.0,
        'performance_issues': []
    }
    
    episode_start_time = time.perf_counter()
    step_times = []
    
    try:
        # Reset environment using env.reset(seed=seed) and handle initial observation
        _logger.info("Resetting environment to start new episode...")
        observation, info = env.reset(seed=effective_seed)
        
        # Log initial state including agent position and observation value for educational context
        _logger.info(f"Episode started! Initial observation shape: {observation.shape}")
        if 'agent_pos' in info:
            _logger.info(f"Initial agent position: {info['agent_pos']}")
        if 'source_pos' in info:
            _logger.info(f"Source position: {info['source_pos']}")
        _logger.info(f"Initial concentration at agent: {observation[0] if len(observation) > 0 else 'N/A'}")
        
        # Execute step loop with random action selection and comprehensive result processing
        _logger.info(f"Beginning episode simulation for {effective_max_steps} steps...")
        
        for step_num in range(effective_max_steps):
            # For each step: sample random action, measure step timing, call env.step(), and log results
            step_start_time = time.perf_counter()
            
            # Sample random action from environment action space
            action = env.action_space.sample()
            action_name = Action(action).name if hasattr(Action, '_name2value_') and action in Action._name2value_.values() else f"Action({action})"
            
            # Execute environment step with comprehensive timing measurement
            observation, reward, terminated, truncated, info = env.step(action)
            
            step_end_time = time.perf_counter()
            step_duration_ms = (step_end_time - step_start_time) * 1000
            step_times.append(step_duration_ms)
            
            # Update episode statistics with step results
            episode_stats['steps_taken'] += 1
            episode_stats['total_reward'] += reward
            
            # Log step results with performance information every 10 steps
            if step_num % 10 == 0 or reward != 0.0 or terminated or truncated:
                _logger.info(f"Step {step_num + 1}: Action={action_name}, Reward={reward:.3f}, "
                           f"Time={step_duration_ms:.3f}ms")
                if 'agent_pos' in info:
                    _logger.info(f"  Agent position: {info['agent_pos']}")
                
            # Check for performance issues against target threshold
            if step_duration_ms > PERFORMANCE_TARGET_MS:
                episode_stats['performance_issues'].append(f"Step {step_num + 1}: {step_duration_ms:.3f}ms")
            
            # Handle termination and truncation conditions with appropriate logging and analysis
            if terminated:
                episode_stats['episode_completed'] = True
                episode_stats['termination_reason'] = 'goal_reached'
                _logger.info(f"🎯 Episode terminated successfully at step {step_num + 1}!")
                _logger.info(f"Goal reached with total reward: {episode_stats['total_reward']:.3f}")
                break
            elif truncated:
                episode_stats['episode_completed'] = True
                episode_stats['termination_reason'] = 'truncated'
                _logger.info(f"⏰ Episode truncated at step {step_num + 1}")
                _logger.info("Episode reached maximum step limit")
                break
                
        # Calculate episode statistics including completion reason, performance metrics, and success indicators
        episode_end_time = time.perf_counter()
        episode_stats['total_episode_time_ms'] = (episode_end_time - episode_start_time) * 1000
        
        if step_times:
            episode_stats['average_step_time_ms'] = sum(step_times) / len(step_times)
            episode_stats['min_step_time_ms'] = min(step_times)
            episode_stats['max_step_time_ms'] = max(step_times)
        
        # Log episode completion with comprehensive statistics and performance analysis
        _logger.info("=" * 40)
        _logger.info("EPISODE COMPLETION SUMMARY:")
        _logger.info(f"  Steps taken: {episode_stats['steps_taken']}")
        _logger.info(f"  Total reward: {episode_stats['total_reward']:.3f}")
        _logger.info(f"  Termination reason: {episode_stats['termination_reason']}")
        _logger.info(f"  Total episode time: {episode_stats['total_episode_time_ms']:.3f}ms")
        _logger.info(f"  Average step time: {episode_stats['average_step_time_ms']:.3f}ms")
        
        # Report performance issues if any detected
        if episode_stats['performance_issues']:
            _logger.warning(f"Performance issues detected in {len(episode_stats['performance_issues'])} steps:")
            for issue in episode_stats['performance_issues'][:5]:  # Show first 5 issues
                _logger.warning(f"  {issue}")
                
    except Exception as e:
        _logger.error(f"✗ Episode execution failed: {e}")
        episode_stats['termination_reason'] = 'error'
        episode_stats['error'] = str(e)
    
    # Return episode statistics dictionary for analysis and reporting
    return episode_stats


def demonstrate_action_usage(env: gymnasium.Env) -> None:
    """
    Demonstrate discrete action space usage with Action enumeration, validation, and movement 
    analysis for educational action space understanding including boundary enforcement.
    
    Args:
        env: Gymnasium environment instance for action space demonstration
    
    Returns:
        None: Demonstrates action space usage without return value, logs comprehensive action information
    """
    # Log action demonstration start with action space information
    _logger.info("=" * 60)
    _logger.info("DEMONSTRATION: Action Space Usage")
    _logger.info(f"Environment action space: {env.action_space}")
    
    # Display available actions using Action enumeration values (UP, RIGHT, DOWN, LEFT)
    _logger.info("Available actions from Action enumeration:")
    try:
        for action_enum in Action:
            _logger.info(f"  {action_enum.name} = {action_enum.value}")
    except Exception as e:
        _logger.warning(f"Could not enumerate Action values: {e}")
        _logger.info("Available actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT")
    
    # Demonstrate action space sampling using env.action_space.sample()
    _logger.info("\nAction space sampling demonstration:")
    for i in range(8):
        sampled_action = env.action_space.sample()
        _logger.info(f"  Sample {i + 1}: {sampled_action}")
    
    # Show action validation using env.action_space.contains() for educational API usage
    _logger.info("\nAction validation demonstration:")
    test_actions = [-1, 0, 1, 2, 3, 4, 10]
    for action in test_actions:
        is_valid = env.action_space.contains(action)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        _logger.info(f"  Action {action}: {status}")
    
    # Demonstrate manual action selection using Action enumeration values
    _logger.info("\nManual action execution demonstration:")
    try:
        # Reset environment to known state for demonstration
        observation, info = env.reset(seed=EXAMPLE_SEED)
        initial_pos = info.get('agent_pos', 'Unknown')
        _logger.info(f"Starting position: {initial_pos}")
        
        # Test each action type with position tracking
        test_actions = [Action.UP.value, Action.RIGHT.value, Action.DOWN.value, Action.LEFT.value]
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        
        for action_val, action_name in zip(test_actions, action_names):
            # Log each action type with corresponding integer value and movement direction
            _logger.info(f"Executing action: {action_name} (value={action_val})")
            
            observation, reward, terminated, truncated, info = env.step(action_val)
            new_pos = info.get('agent_pos', 'Unknown')
            
            # Show action processing results including position changes and boundary enforcement
            _logger.info(f"  Result: Position={new_pos}, Reward={reward:.3f}")
            
            if terminated or truncated:
                _logger.info(f"  Episode ended: terminated={terminated}, truncated={truncated}")
                break
                
    except Exception as e:
        _logger.error(f"Action demonstration error: {e}")
        _logger.info("Manual action testing incomplete due to error")
    
    _logger.info("Action space demonstration completed successfully")


def demonstrate_rendering(env: gymnasium.Env) -> dict:
    """
    Demonstrate dual-mode rendering capabilities with RGB array and human visualization modes, 
    including fallback handling and performance analysis for comprehensive rendering understanding.
    
    Args:
        env: Gymnasium environment instance for rendering demonstration
    
    Returns:
        dict: Rendering demonstration results including mode compatibility and performance metrics
    """
    # Log rendering demonstration start with environment render mode information
    _logger.info("=" * 60)
    _logger.info("DEMONSTRATION: Rendering Capabilities")
    _logger.info(f"Environment render mode: {getattr(env, 'render_mode', 'None')}")
    
    # Initialize rendering results dictionary for comprehensive reporting
    render_results = {
        'rgb_array_supported': False,
        'human_mode_supported': False,
        'rgb_array_time_ms': 0.0,
        'human_mode_time_ms': 0.0,
        'rgb_array_shape': None,
        'error_messages': []
    }
    
    try:
        # Ensure environment is in known state for rendering demonstration
        observation, info = env.reset(seed=EXAMPLE_SEED)
        _logger.info("Environment reset for rendering demonstration")
        
        # Test RGB array rendering using env.render() with performance timing measurement
        _logger.info("Testing RGB array rendering mode...")
        
        try:
            rgb_start_time = time.perf_counter()
            
            # Create temporary environment with rgb_array mode if needed
            if env.render_mode != 'rgb_array':
                _logger.info("Creating temporary environment with rgb_array render mode...")
                temp_env = gymnasium.make(ENV_ID, render_mode='rgb_array')
                temp_env.reset(seed=EXAMPLE_SEED)
                rgb_array = temp_env.render()
                temp_env.close()
            else:
                rgb_array = env.render()
            
            rgb_end_time = time.perf_counter()
            render_results['rgb_array_time_ms'] = (rgb_end_time - rgb_start_time) * 1000
            
            # Validate RGB array output format including shape, dtype, and value ranges
            if rgb_array is not None:
                render_results['rgb_array_supported'] = True
                render_results['rgb_array_shape'] = rgb_array.shape
                
                # Log RGB array properties and performance metrics for educational analysis
                _logger.info(f"✓ RGB array rendering successful!")
                _logger.info(f"  Array shape: {rgb_array.shape}")
                _logger.info(f"  Array dtype: {rgb_array.dtype}")
                _logger.info(f"  Value range: [{rgb_array.min()}, {rgb_array.max()}]")
                _logger.info(f"  Rendering time: {render_results['rgb_array_time_ms']:.3f}ms")
            else:
                render_results['error_messages'].append("RGB array render returned None")
                
        except Exception as e:
            render_results['error_messages'].append(f"RGB array rendering failed: {e}")
            _logger.error(f"✗ RGB array rendering error: {e}")
        
        # Test human mode rendering if supported with fallback handling for headless environments
        _logger.info("\nTesting human mode rendering...")
        
        try:
            human_start_time = time.perf_counter()
            
            # Create temporary environment with human mode if needed
            if env.render_mode != 'human':
                _logger.info("Creating temporary environment with human render mode...")
                temp_env_human = gymnasium.make(ENV_ID, render_mode='human')
                temp_env_human.reset(seed=EXAMPLE_SEED)
                result = temp_env_human.render()
                temp_env_human.close()
            else:
                result = env.render()
            
            human_end_time = time.perf_counter()
            render_results['human_mode_time_ms'] = (human_end_time - human_start_time) * 1000
            
            # Human mode typically returns None as it displays directly
            render_results['human_mode_supported'] = True
            _logger.info("✓ Human mode rendering executed successfully!")
            _logger.info(f"  Rendering time: {render_results['human_mode_time_ms']:.3f}ms")
            _logger.info("  Note: Human mode displays window (may not be visible in headless environment)")
            
        except Exception as e:
            render_results['error_messages'].append(f"Human mode rendering failed: {e}")
            _logger.warning(f"⚠ Human mode rendering issue: {e}")
            _logger.info("This is expected in headless environments or without display")
        
    except Exception as e:
        render_results['error_messages'].append(f"General rendering error: {e}")
        _logger.error(f"✗ Rendering demonstration error: {e}")
    
    # Compile rendering results including mode compatibility and performance analysis
    _logger.info("=" * 40)
    _logger.info("RENDERING DEMONSTRATION SUMMARY:")
    _logger.info(f"  RGB array support: {'✓' if render_results['rgb_array_supported'] else '✗'}")
    _logger.info(f"  Human mode support: {'✓' if render_results['human_mode_supported'] else '✗'}")
    
    if render_results['rgb_array_supported']:
        _logger.info(f"  RGB performance: {render_results['rgb_array_time_ms']:.3f}ms")
    if render_results['human_mode_supported']:
        _logger.info(f"  Human performance: {render_results['human_mode_time_ms']:.3f}ms")
    
    # Log rendering demonstration completion with capability summary and recommendations
    if render_results['error_messages']:
        _logger.info("  Issues encountered:")
        for error_msg in render_results['error_messages']:
            _logger.info(f"    - {error_msg}")
    
    # Return rendering results dictionary for performance analysis and troubleshooting
    return render_results


def demonstrate_seeding_reproducibility(env: gymnasium.Env, test_seed: int, num_test_steps: int) -> bool:
    """
    Demonstrate deterministic behavior through seeding with identical episode generation and 
    reproducibility validation for scientific research workflows and debugging consistency.
    
    Args:
        env: Gymnasium environment instance for reproducibility testing
        test_seed: Seed value to use for reproducibility demonstration
        num_test_steps: Number of steps to execute in each reproducibility test episode
    
    Returns:
        bool: True if reproducibility demonstrated successfully, False if reproducibility issues detected
    """
    # Log reproducibility demonstration start with test_seed and configuration information
    _logger.info("=" * 60)
    _logger.info("DEMONSTRATION: Seeding and Reproducibility")
    _logger.info(f"Testing reproducibility with seed: {test_seed}")
    _logger.info(f"Episode length for comparison: {num_test_steps} steps")
    
    try:
        # Execute first episode with test_seed recording all step outcomes and observations
        _logger.info("Executing first episode for reproducibility comparison...")
        
        first_episode_data = []
        observation1, info1 = env.reset(seed=test_seed)
        first_episode_data.append({
            'step': 0,
            'observation': observation1.copy(),
            'info': info1.copy(),
            'action': None,
            'reward': 0.0
        })
        
        for step_num in range(num_test_steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            first_episode_data.append({
                'step': step_num + 1,
                'observation': observation.copy(),
                'info': info.copy(),
                'action': action,
                'reward': reward
            })
            
            if terminated or truncated:
                _logger.info(f"First episode terminated/truncated at step {step_num + 1}")
                break
        
        # Reset environment and execute second episode with identical test_seed
        _logger.info("Executing second episode with identical seed for comparison...")
        
        second_episode_data = []
        observation2, info2 = env.reset(seed=test_seed)
        second_episode_data.append({
            'step': 0,
            'observation': observation2.copy(),
            'info': info2.copy(),
            'action': None,
            'reward': 0.0
        })
        
        for step_num in range(len(first_episode_data) - 1):  # Match first episode length
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            second_episode_data.append({
                'step': step_num + 1,
                'observation': observation.copy(),
                'info': info.copy(),
                'action': action,
                'reward': reward
            })
            
            if terminated or truncated:
                _logger.info(f"Second episode terminated/truncated at step {step_num + 1}")
                break
        
        # Compare observations, rewards, and state progressions between episodes for identity verification
        _logger.info("Comparing episode data for reproducibility validation...")
        
        reproducibility_passed = True
        comparison_results = {
            'matching_steps': 0,
            'total_steps_compared': min(len(first_episode_data), len(second_episode_data)),
            'observation_mismatches': 0,
            'reward_mismatches': 0,
            'info_mismatches': 0
        }
        
        for i in range(comparison_results['total_steps_compared']):
            step_data_1 = first_episode_data[i]
            step_data_2 = second_episode_data[i]
            
            # Compare observations with numerical tolerance
            obs_match = np.allclose(step_data_1['observation'], step_data_2['observation'], rtol=1e-10)
            reward_match = abs(step_data_1['reward'] - step_data_2['reward']) < 1e-10
            
            # Compare agent positions if available in info
            info_match = True
            if 'agent_pos' in step_data_1['info'] and 'agent_pos' in step_data_2['info']:
                info_match = step_data_1['info']['agent_pos'] == step_data_2['info']['agent_pos']
            
            if obs_match and reward_match and info_match:
                comparison_results['matching_steps'] += 1
            else:
                reproducibility_passed = False
                if not obs_match:
                    comparison_results['observation_mismatches'] += 1
                if not reward_match:
                    comparison_results['reward_mismatches'] += 1
                if not info_match:
                    comparison_results['info_mismatches'] += 1
                
                # Log first few mismatches for debugging
                if comparison_results['observation_mismatches'] <= 3:
                    _logger.error(f"Observation mismatch at step {i}:")
                    _logger.error(f"  Episode 1: {step_data_1['observation'][:5]}...")
                    _logger.error(f"  Episode 2: {step_data_2['observation'][:5]}...")
        
        # Validate deterministic behavior by checking exact match of episode trajectories
        success_rate = comparison_results['matching_steps'] / comparison_results['total_steps_compared'] * 100
        
        # Log reproducibility validation results with success status and trajectory analysis
        _logger.info("=" * 40)
        _logger.info("REPRODUCIBILITY VALIDATION RESULTS:")
        _logger.info(f"  Steps compared: {comparison_results['total_steps_compared']}")
        _logger.info(f"  Matching steps: {comparison_results['matching_steps']}")
        _logger.info(f"  Success rate: {success_rate:.1f}%")
        
        if reproducibility_passed:
            _logger.info("✓ Perfect reproducibility achieved!")
            _logger.info("Episodes with identical seeds produced identical results")
        else:
            # Handle reproducibility failures with detailed analysis and debugging information
            _logger.error("✗ Reproducibility issues detected!")
            _logger.error(f"  Observation mismatches: {comparison_results['observation_mismatches']}")
            _logger.error(f"  Reward mismatches: {comparison_results['reward_mismatches']}")
            _logger.error(f"  Info mismatches: {comparison_results['info_mismatches']}")
            
            _logger.error("Potential causes:")
            _logger.error("  1. Improper seeding implementation")
            _logger.error("  2. Non-deterministic operations in environment")
            _logger.error("  3. External randomness sources")
            _logger.error("  4. Floating-point precision issues")
        
        # Return reproducibility status for demonstration validation and educational feedback
        return reproducibility_passed
        
    except Exception as e:
        _logger.error(f"✗ Reproducibility demonstration failed: {e}")
        _logger.error("This may indicate serious issues with environment seeding")
        return False


def demonstrate_error_handling(env: gymnasium.Env) -> None:
    """
    Demonstrate comprehensive error handling including invalid actions, configuration errors, 
    and recovery procedures for robust usage patterns and educational error management.
    
    Args:
        env: Gymnasium environment instance for error handling demonstration
    
    Returns:
        None: Demonstrates error handling patterns without return value, logs error scenarios and recovery
    """
    # Log error handling demonstration start with error scenario preparation
    _logger.info("=" * 60)
    _logger.info("DEMONSTRATION: Error Handling Patterns")
    _logger.info("Testing various error conditions and recovery procedures...")
    
    # Demonstrate invalid action handling by attempting out-of-bounds action values
    _logger.info("\n1. Testing invalid action handling:")
    
    try:
        # Reset environment to known state for error testing
        env.reset(seed=EXAMPLE_SEED)
        
        invalid_actions = [-1, 4, 10, -5]
        for invalid_action in invalid_actions:
            _logger.info(f"Attempting invalid action: {invalid_action}")
            
            try:
                observation, reward, terminated, truncated, info = env.step(invalid_action)
                _logger.warning(f"  Unexpected: Invalid action {invalid_action} was accepted")
                
            except ValidationError as ve:
                # Show ValidationError catching and appropriate error message display
                _logger.info(f"  ✓ ValidationError caught correctly: {ve}")
                
            except Exception as e:
                _logger.info(f"  ✓ Action {invalid_action} rejected with: {type(e).__name__}: {e}")
                
    except Exception as e:
        _logger.error(f"Error during invalid action testing: {e}")
    
    # Demonstrate environment state error recovery with graceful handling procedures
    _logger.info("\n2. Testing environment state error recovery:")
    
    try:
        # Attempt operations on closed environment to test error handling
        _logger.info("Testing operations on closed environment...")
        
        # Create temporary environment for closing test
        temp_env = gymnasium.make(ENV_ID)
        temp_env.close()
        
        try:
            temp_env.reset()
            _logger.warning("  Unexpected: reset() succeeded on closed environment")
        except Exception as e:
            _logger.info(f"  ✓ Closed environment error handled: {type(e).__name__}")
        
        try:
            temp_env.step(0)
            _logger.warning("  Unexpected: step() succeeded on closed environment")
        except Exception as e:
            _logger.info(f"  ✓ Closed environment step error handled: {type(e).__name__}")
            
    except Exception as e:
        _logger.error(f"Error during state recovery testing: {e}")
    
    # Show proper exception hierarchy usage with PlumeNavSimError base class handling
    _logger.info("\n3. Testing exception hierarchy and custom errors:")
    
    try:
        # Demonstrate catching PlumeNavSimError base class
        _logger.info("Testing PlumeNavSimError base class exception handling...")
        
        try:
            # This might trigger various plume_nav_sim specific errors
            invalid_env = gymnasium.make(ENV_ID, render_mode='invalid_mode')
        except PlumeNavSimError as pnse:
            _logger.info(f"  ✓ PlumeNavSimError caught: {pnse}")
        except Exception as e:
            _logger.info(f"  Other error type: {type(e).__name__}: {e}")
            
    except Exception as e:
        _logger.error(f"Error during exception hierarchy testing: {e}")
    
    # Demonstrate rendering error handling with fallback mode switching
    _logger.info("\n4. Testing rendering error handling:")
    
    try:
        _logger.info("Testing rendering error scenarios...")
        
        # Test rendering on environment that might have display issues
        try:
            if hasattr(env, 'render'):
                # This might fail in headless environments
                render_result = env.render()
                _logger.info(f"  Rendering succeeded: {type(render_result)}")
            else:
                _logger.info("  Environment has no render method")
                
        except Exception as e:
            _logger.info(f"  ✓ Rendering error handled gracefully: {type(e).__name__}")
            _logger.info("  Fallback: Using rgb_array mode or disabling visualization")
            
    except Exception as e:
        _logger.error(f"Error during rendering error testing: {e}")
    
    # Log each error scenario with recovery procedure and user guidance information
    _logger.info("\n5. Error handling best practices summary:")
    _logger.info("  ✓ Always validate inputs before environment operations")
    _logger.info("  ✓ Use specific exception types (ValidationError, PlumeNavSimError)")
    _logger.info("  ✓ Implement graceful fallbacks for non-critical failures")
    _logger.info("  ✓ Log errors with sufficient detail for debugging")
    _logger.info("  ✓ Test error scenarios during development")
    
    # Show best practices for error logging and user-friendly error reporting
    _logger.info("\nError handling demonstration completed successfully!")
    _logger.info("Robust error handling enables stable research workflows")


def run_basic_usage_demo() -> int:
    """
    Execute complete basic usage demonstration coordinating all example components with comprehensive 
    logging, performance monitoring, and educational guidance for new users getting started with 
    plume_nav_sim package functionality.
    
    Returns:
        int: Exit status code: 0 for success, 1 for demonstration failures, 2 for critical errors
    """
    # Set up example logging using setup_example_logging() for clear output formatting
    setup_example_logging('INFO')
    
    # Log basic usage demonstration start with package version and configuration information
    _logger.info("🚀 PLUME NAV SIM - BASIC USAGE DEMONSTRATION")
    _logger.info("=" * 80)
    _logger.info("This demonstration showcases essential plume navigation environment functionality")
    _logger.info("including registration, instantiation, episode lifecycle, and API usage patterns.")
    _logger.info("")
    
    env = None
    exit_code = 0
    
    try:
        # Demonstrate environment registration using demonstrate_environment_registration()
        _logger.info("Phase 1: Environment Registration")
        registration_success = demonstrate_environment_registration()
        
        # Handle registration failures with user-friendly error messages and troubleshooting guidance
        if not registration_success:
            _logger.error("❌ Environment registration failed - cannot proceed with demonstration")
            _logger.error("Please check installation and try running again")
            return 1
        
        # Create environment instance using demonstrate_environment_creation() with default render mode
        _logger.info("\nPhase 2: Environment Creation")
        env = demonstrate_environment_creation(render_mode=None)
        
        # Handle environment creation errors with detailed diagnostics and resolution suggestions
        if env is None:
            _logger.error("❌ Environment creation failed - cannot proceed with demonstration")
            _logger.error("Please verify registration success and dependency installation")
            return 1
        
        # Demonstrate action space usage using demonstrate_action_usage() for API education
        _logger.info("\nPhase 3: Action Space Exploration")
        demonstrate_action_usage(env)
        
        # Execute basic episode using demonstrate_basic_episode() with performance monitoring
        _logger.info("\nPhase 4: Basic Episode Execution")
        episode_stats = demonstrate_basic_episode(env, seed=EXAMPLE_SEED, max_steps=NUM_DEMO_STEPS)
        
        # Analyze episode performance for educational feedback
        if episode_stats.get('termination_reason') == 'error':
            _logger.warning("⚠ Episode execution encountered errors but demonstration continues")
            exit_code = 1
        elif episode_stats.get('performance_issues'):
            _logger.info(f"ℹ Performance advisory: {len(episode_stats['performance_issues'])} slow steps detected")
        
        # Demonstrate rendering capabilities using demonstrate_rendering() with fallback handling
        _logger.info("\nPhase 5: Rendering Capabilities")
        render_results = demonstrate_rendering(env)
        
        if not render_results.get('rgb_array_supported') and not render_results.get('human_mode_supported'):
            _logger.warning("⚠ Neither rendering mode fully functional - this may be expected in headless environments")
        
        # Show seeding reproducibility using demonstrate_seeding_reproducibility() for scientific workflows
        _logger.info("\nPhase 6: Seeding and Reproducibility")
        reproducibility_success = demonstrate_seeding_reproducibility(env, test_seed=EXAMPLE_SEED, num_test_steps=10)
        
        if not reproducibility_success:
            _logger.warning("⚠ Reproducibility issues detected - this may affect research reliability")
            exit_code = 1
        
        # Demonstrate error handling patterns using demonstrate_error_handling() for robust usage
        _logger.info("\nPhase 7: Error Handling Patterns")
        demonstrate_error_handling(env)
        
    except KeyboardInterrupt:
        _logger.info("\n🛑 Demonstration interrupted by user")
        exit_code = 2
        
    except Exception as e:
        _logger.error(f"❌ Critical error during demonstration: {e}")
        _logger.error("This indicates a serious issue with the environment or dependencies")
        exit_code = 2
        
    finally:
        # Cleanup environment resources properly using env.close() with resource management demonstration
        if env is not None:
            try:
                _logger.info("\nPhase 8: Environment Cleanup")
                _logger.info("Closing environment and releasing resources...")
                env.close()
                _logger.info("✓ Environment closed successfully")
            except Exception as e:
                _logger.error(f"Error during environment cleanup: {e}")
                exit_code = max(exit_code, 1)
    
    # Log demonstration completion with comprehensive summary and next steps guidance
    _logger.info("=" * 80)
    _logger.info("🏁 BASIC USAGE DEMONSTRATION COMPLETED")
    
    if exit_code == 0:
        _logger.info("✅ All demonstration phases completed successfully!")
        _logger.info("\nNext Steps:")
        _logger.info("1. Study this example script to understand API patterns")
        _logger.info("2. Experiment with different seeds and episode lengths")
        _logger.info("3. Try both rendering modes to understand visualization")
        _logger.info("4. Implement your own agent using the demonstrated patterns")
        _logger.info("5. Explore advanced features in the documentation")
        
    elif exit_code == 1:
        _logger.warning("⚠ Demonstration completed with warnings")
        _logger.info("Some features may not be fully functional in this environment")
        _logger.info("Review warning messages above for specific issues")
        
    else:
        _logger.error("❌ Demonstration encountered critical errors")
        _logger.error("Please check your installation and environment setup")
    
    _logger.info(f"\nExiting with status code: {exit_code}")
    
    # Return success status code for script execution validation and automation integration
    return exit_code


def main() -> None:
    """
    Main entry point for basic usage example script with command-line interface support and 
    comprehensive demonstration execution including error handling and system exit management.
    
    Returns:
        None: Script entry point with system exit handling and appropriate status codes
    """
    try:
        # Handle command-line arguments for optional configuration and demonstration customization
        # For now, use default configuration - future enhancement could add argument parsing
        
        # Set up global exception handling for unhandled errors with user-friendly messages
        _logger.info("Starting plume_nav_sim basic usage demonstration...")
        
        # Execute run_basic_usage_demo() with comprehensive error catching and status reporting
        exit_status = run_basic_usage_demo()
        
        # Handle demonstration failures with appropriate exit codes and error reporting
        if exit_status == 0:
            print("\n🎉 Basic usage demonstration completed successfully!")
            print("The plume_nav_sim environment is ready for your research!")
        elif exit_status == 1:
            print("\n⚠ Demonstration completed with some issues.")
            print("Check the logs above for details on any warnings or non-critical errors.")
        else:
            print("\n❌ Demonstration failed with critical errors.")
            print("Please check your installation and environment setup.")
        
        # Provide user guidance for successful completion and next steps in learning workflow
        print(f"\nFor more information, see the documentation and additional examples.")
        
        # Exit with appropriate status code for automation and script integration compatibility
        sys.exit(exit_status)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demonstration interrupted by user. Goodbye!")
        sys.exit(2)
        
    except Exception as e:
        print(f"\n❌ Unhandled error in main(): {e}")
        print("This indicates a serious issue. Please report this error.")
        sys.exit(2)


# Script execution guard for direct execution and import safety
if __name__ == "__main__":
    main()