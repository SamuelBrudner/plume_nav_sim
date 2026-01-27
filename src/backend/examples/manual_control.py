"""
Interactive manual control demonstration script enabling real-time human control of the plume navigation agent through keyboard input.

This module provides comprehensive manual control functionality with immediate visual feedback, educational exploration capabilities,
intuitive WASD/arrow key controls, and detailed gameplay statistics for hands-on learning and algorithm development insight.
Designed for researchers, educators, and developers to manually explore environment dynamics, plume characteristics, and navigation challenges.

The script demonstrates real-time agent interaction, keyboard input handling, matplotlib-based visualization, performance tracking,
and educational guidance throughout the manual control session with comprehensive error handling and graceful resource cleanup.
"""

import logging  # standard library - Manual control session logging, user action tracking, and error reporting for debugging
import queue  # standard library - Thread-safe communication between keyboard input thread and main control loop
import sys  # standard library - System interaction for exit code management and keyboard input handling in manual control
import threading  # standard library - Background input handling and non-blocking keyboard input processing for smooth manual control
import time  # standard library - High-precision timing for manual control performance tracking and session duration analysis

# External imports with version comments
import gymnasium as gym  # >=0.29.0 - Reinforcement learning environment framework for manual control environment instantiation and step processing
import matplotlib.pyplot as plt  # >=3.9.0 - Interactive visualization framework for real-time manual control display with keyboard event handling
import numpy as np  # >=2.1.0 - Array operations for manual control statistics, position tracking, and performance analysis

from ..plume_nav_sim.core.constants import DEFAULT_SOURCE_LOCATION
from ..plume_nav_sim.core.types import Action

# Internal imports
from ..plume_nav_sim.registration.register import ENV_ID, register_env

# Global constants for manual control configuration
CONTROL_INSTRUCTIONS = (
    "MANUAL CONTROL: Use WASD or Arrow Keys to move. Q to quit, R to reset, H for help."
)
KEYBOARD_MAPPING = {
    "w": Action.UP,
    "s": Action.DOWN,
    "a": Action.LEFT,
    "d": Action.RIGHT,
    "up": Action.UP,
    "down": Action.DOWN,
    "left": Action.LEFT,
    "right": Action.RIGHT,
}
CONTROL_UPDATE_DELAY = 0.05  # 50ms delay for smooth control response
SESSION_TIMEOUT_MINUTES = 30  # 30 minute session timeout for automatic cleanup
STATISTICS_DISPLAY_FREQUENCY = 10  # Display statistics every 10 actions
DEFAULT_MANUAL_SEED = 42  # Default seed for reproducible manual sessions

# Module exports for manual control functionality
__all__ = [
    "main",
    "run_manual_control_session",
    "setup_manual_control",
    "create_keyboard_handler",
    "display_control_instructions",
    "update_session_statistics",
    "handle_manual_input",
]


def setup_manual_control(
    seed: int = None,
    enable_debug_logging: bool = False,
    check_backend_compatibility: bool = True,
) -> tuple:
    """Initialize manual control environment with registration validation, matplotlib configuration for keyboard input,
    backend compatibility checking, and interactive session preparation for real-time agent control.

    Args:
        seed: Random seed for reproducible sessions, defaults to DEFAULT_MANUAL_SEED
        enable_debug_logging: Enable detailed debug logging for development
        check_backend_compatibility: Check matplotlib backend compatibility for interactive input

    Returns:
        tuple: (environment, session_config) with configured environment and manual control session settings

    Raises:
        ConfigurationError: If environment setup or matplotlib configuration fails
        IntegrationError: If Gymnasium or matplotlib integration issues occur
    """
    # Configure logging system for manual control session with appropriate verbosity level
    log_level = logging.DEBUG if enable_debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("manual_control.setup")

    try:
        # Register environment using register_env() with validation and error handling
        logger.info("Registering plume navigation environment...")
        register_env(env_id=ENV_ID, force_reregister=True)
        logger.info(f"Successfully registered environment: {ENV_ID}")

        # Determine current matplotlib backend once so it is always defined
        current_backend = plt.get_backend()

        # Check matplotlib backend compatibility for interactive keyboard input handling
        if check_backend_compatibility:
            logger.debug("Checking matplotlib backend compatibility...")
            logger.info(f"Current matplotlib backend: {current_backend}")

            if current_backend == "Agg":  # Non-interactive backend
                # Switch to interactive backend if available or warn about limitations
                interactive_backends = ["TkAgg", "Qt5Agg", "GTK3Agg"]
                for backend in interactive_backends:
                    try:
                        plt.switch_backend(backend)
                        current_backend = backend
                        logger.info(f"Switched to interactive backend: {backend}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to switch to {backend}: {e}")
                        continue
                else:
                    logger.warning(
                        "No interactive matplotlib backend available - keyboard input may be limited"
                    )

        # Create environment instance using gym.make() with human render mode for real-time visualization
        logger.info(f"Creating environment instance: {ENV_ID}")
        env = gym.make(ENV_ID, render_mode="human")

        # Initialize environment with provided seed or DEFAULT_MANUAL_SEED for reproducible sessions
        if seed is None:
            seed = DEFAULT_MANUAL_SEED

        logger.info(f"Initializing environment with seed: {seed}")
        env.reset(seed=seed)

        # Verify keyboard input capabilities and warn about platform-specific limitations
        logger.debug("Verifying keyboard input capabilities...")
        # This would include platform-specific checks in a full implementation

        # Create session configuration dictionary with timeout, controls, and display settings
        session_config = {
            "timeout_seconds": SESSION_TIMEOUT_MINUTES * 60,
            "update_delay": CONTROL_UPDATE_DELAY,
            "statistics_frequency": STATISTICS_DISPLAY_FREQUENCY,
            "keyboard_mapping": KEYBOARD_MAPPING.copy(),
            "control_instructions": CONTROL_INSTRUCTIONS,
            "debug_logging": enable_debug_logging,
            "session_seed": seed,
            "backend_compatible": current_backend != "Agg",
        }

        # Log manual control setup completion with environment configuration and input capabilities
        logger.info("Manual control setup completed successfully")
        logger.debug(f"Session configuration: {session_config}")

        # Return configured environment and session settings ready for interactive control
        return env, session_config

    except Exception as e:
        logger.error(f"Manual control setup failed: {e}")
        raise


def create_keyboard_handler(
    input_queue: queue.Queue, key_mapping: dict, enable_special_commands: bool = True
) -> threading.Thread:
    """Create cross-platform keyboard input handler with non-blocking input processing, action mapping,
    and thread-safe communication for smooth real-time manual control without blocking environment updates.

    Args:
        input_queue: Thread-safe queue for keyboard input communication
        key_mapping: Dictionary mapping keyboard characters to Action enums
        enable_special_commands: Whether to enable special commands (Q=quit, R=reset, H=help)

    Returns:
        threading.Thread: Background keyboard input thread with proper cleanup and termination handling

    Raises:
        ValidationError: If input_queue or key_mapping parameters are invalid
    """
    logger = logging.getLogger("manual_control.keyboard")

    def keyboard_input_loop():
        """Background keyboard input processing loop with cross-platform compatibility."""
        logger.debug("Starting keyboard input handler thread...")

        try:
            # Configure platform-specific keyboard input handling (Windows/Unix compatibility)
            import select
            import sys
            import termios
            import tty

            if sys.platform.startswith("win"):
                # Windows keyboard input handling
                import msvcrt

                while True:
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode("utf-8").lower()
                        logger.debug(f"Windows key detected: {key}")

                        # Map keyboard characters to Action enums using KEYBOARD_MAPPING
                        if key in key_mapping:
                            input_queue.put(("action", key_mapping[key]))
                        # Handle special commands (Q=quit, R=reset, H=help) if enable_special_commands is True
                        elif enable_special_commands:
                            if key == "q":
                                input_queue.put(("command", "quit"))
                            elif key == "r":
                                input_queue.put(("command", "reset"))
                            elif key == "h":
                                input_queue.put(("command", "help"))

                    # Implement non-blocking input reading to prevent environment freezing
                    time.sleep(0.01)  # Small delay to prevent high CPU usage

            else:
                # Unix/Linux keyboard input handling
                old_settings = termios.tcgetattr(sys.stdin)

                try:
                    tty.setcbreak(sys.stdin.fileno())

                    while True:
                        # Use select for non-blocking input
                        if select.select([sys.stdin], [], [], 0.01) == (
                            [sys.stdin],
                            [],
                            [],
                        ):
                            key = sys.stdin.read(1).lower()
                            logger.debug(f"Unix key detected: {key}")

                            # Queue validated actions and commands using thread-safe queue operations
                            if key in key_mapping:
                                input_queue.put(("action", key_mapping[key]))
                            elif enable_special_commands:
                                if key == "q":
                                    input_queue.put(("command", "quit"))
                                elif key == "r":
                                    input_queue.put(("command", "reset"))
                                elif key == "h":
                                    input_queue.put(("command", "help"))

                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        except Exception as e:
            logger.error(f"Keyboard input handler error: {e}")
            # Add proper error handling for keyboard input failures and platform incompatibilities
            input_queue.put(("error", str(e)))

    # Create daemon thread for background input processing with clean termination
    keyboard_thread = threading.Thread(target=keyboard_input_loop, daemon=True)
    keyboard_thread.name = "KeyboardInputHandler"

    logger.info("Keyboard input handler thread created successfully")
    # Return configured keyboard input thread ready for manual control session
    return keyboard_thread


def display_control_instructions(
    env: gym.Env,
    show_environment_info: bool = True,
    show_advanced_controls: bool = False,
) -> None:
    """Display comprehensive control instructions, gameplay information, environment details, and real-time statistics
    for user guidance and educational value during manual control sessions.

    Args:
        env: Gymnasium environment instance for extracting environment information
        show_environment_info: Whether to display environment details (grid size, source location)
        show_advanced_controls: Whether to show advanced control features and tips

    Raises:
        ValidationError: If environment parameter is invalid
    """
    logger = logging.getLogger("manual_control.instructions")

    try:
        # Display welcome message and manual control session information
        print("\n" + "=" * 80)
        print("🎮 PLUME NAVIGATION - MANUAL CONTROL DEMONSTRATION")
        print("=" * 80)

        # Show basic movement controls (WASD/Arrow keys) with clear action mapping
        print("\n📋 BASIC CONTROLS:")
        print("  Movement:")
        print("    W / ↑  : Move UP")
        print("    S / ↓  : Move DOWN")
        print("    A / ←  : Move LEFT")
        print("    D / →  : Move RIGHT")

        # Display special command instructions (Q=quit, R=reset, H=help)
        print("\n  Special Commands:")
        print("    Q      : Quit session")
        print("    R      : Reset environment")
        print("    H      : Show help")

        # Show environment information if show_environment_info enabled (grid size, source location)
        if show_environment_info:
            print("\n🌐 ENVIRONMENT INFORMATION:")
            try:
                # Extract environment details from unwrapped environment
                unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env

                if hasattr(unwrapped, "grid_size"):
                    grid_size = unwrapped.grid_size
                    print(f"    Grid Size: {grid_size[0]} × {grid_size[1]}")

                if hasattr(unwrapped, "source_location"):
                    source_loc = unwrapped.source_location
                    print(f"    Source Location: ({source_loc[0]}, {source_loc[1]})")
                else:
                    print(f"    Source Location: {DEFAULT_SOURCE_LOCATION}")

                print(f"    Environment ID: {ENV_ID}")

            except Exception as e:
                logger.debug(f"Could not extract environment info: {e}")
                print("    Environment details not available")

        # Display advanced control features if show_advanced_controls enabled
        if show_advanced_controls:
            print("\n⚡ ADVANCED FEATURES:")
            print("    • Real-time concentration visualization")
            print("    • Session performance statistics")
            print("    • Educational feedback and tips")
            print("    • Automatic session timeout protection")

        # Show gameplay objectives and goal achievement criteria
        print("\n🎯 GAMEPLAY OBJECTIVES:")
        print("    • Navigate the agent through the plume concentration field")
        print("    • Find and reach the plume source (white cross marker)")
        print("    • Agent appears as red square on the visualization")
        print("    • Higher concentration values indicate proximity to source")

        # Display real-time statistics legend and interpretation guide
        print("\n📊 STATISTICS LEGEND:")
        print("    • Steps: Total movement actions taken")
        print("    • Reward: Points earned (1.0 for reaching goal)")
        print("    • Distance: Current distance to source location")
        print("    • Duration: Session elapsed time")

        # Show performance tips and optimal navigation strategies
        print("\n💡 NAVIGATION TIPS:")
        print("    • Watch concentration gradients for navigation hints")
        print("    • Higher values (brighter colors) indicate closer proximity")
        print("    • Use systematic search patterns for efficient exploration")
        print("    • Reset environment (R) to try different strategies")

        # Display troubleshooting information for common input issues
        print("\n🔧 TROUBLESHOOTING:")
        print("    • If keyboard input is unresponsive, check terminal focus")
        print("    • On some systems, run in terminal with proper permissions")
        print("    • Press 'H' anytime during session for help")

        print("\n" + "=" * 80)
        print("Press any movement key to begin manual control session...")
        print("=" * 80 + "\n")

        # Log instruction display completion for session tracking
        logger.info("Control instructions displayed successfully")

    except Exception as e:
        logger.error(f"Failed to display control instructions: {e}")
        print(f"Error displaying instructions: {e}")


def run_manual_control_session(
    env: gym.Env,
    session_timeout_seconds: int,
    display_statistics: bool = True,
    enable_performance_tracking: bool = True,
) -> dict:
    """Main manual control session orchestrator managing real-time keyboard input processing, environment step execution,
    visualization updates, session statistics tracking, and user interaction for complete hands-on plume navigation experience.

    Args:
        env: Configured Gymnasium environment instance
        session_timeout_seconds: Maximum session duration in seconds
        display_statistics: Whether to display periodic statistics updates
        enable_performance_tracking: Whether to track detailed performance metrics

    Returns:
        dict: Complete session results with statistics, performance metrics, user actions, and educational outcomes

    Raises:
        StateError: If environment is in invalid state for manual control
        ResourceError: If system resources are insufficient for session
    """
    logger = logging.getLogger("manual_control.session")

    # Initialize session tracking variables (start time, action count, statistics)
    session_start_time = time.time()
    session_stats = {
        "start_time": session_start_time,
        "total_actions": 0,
        "successful_actions": 0,
        "goal_reached": False,
        "total_reward": 0.0,
        "action_history": [],
        "position_history": [],
        "session_duration": 0.0,
        "performance_metrics": {} if enable_performance_tracking else None,
    }

    try:
        # Create input queue and keyboard handler thread for non-blocking input processing
        input_queue = queue.Queue()
        keyboard_thread = create_keyboard_handler(
            input_queue, KEYBOARD_MAPPING, enable_special_commands=True
        )

        # Start keyboard input thread with proper error handling and platform compatibility
        logger.info("Starting keyboard input handler...")
        keyboard_thread.start()

        # Reset environment and display initial state with agent position and concentration field
        logger.info("Resetting environment for manual control session...")
        observation, info = env.reset()
        env.render()  # Display initial visualization

        logger.info("Manual control session started - use WASD or arrow keys to move")

        # Initialize performance tracking if enabled
        if enable_performance_tracking:
            session_stats["performance_metrics"]["step_times"] = []
            session_stats["performance_metrics"]["render_times"] = []

        # Enter main control loop with real-time input processing and environment updates
        session_active = True
        last_statistics_display = 0

        while session_active:
            _ = time.time()

            # Check for session timeout and goal achievement for session termination
            if _check_manual_session_timeout(
                logger=logger,
                session_start_time=session_start_time,
                session_timeout_seconds=session_timeout_seconds,
            ):
                break

            try:
                # Process queued keyboard inputs and convert to environment actions
                input_type, input_value = input_queue.get(timeout=CONTROL_UPDATE_DELAY)

                if input_type == "action":
                    # Execute environment step with user action and capture response
                    reward, terminated, truncated, last_statistics_display = (
                        _process_manual_session_action(
                            env=env,
                            input_value=input_value,
                            session_stats=session_stats,
                            display_statistics=display_statistics,
                            enable_performance_tracking=enable_performance_tracking,
                            last_statistics_display=last_statistics_display,
                        )
                    )

                    # Check for goal achievement
                    if reward > 0 or terminated:
                        session_stats["goal_reached"] = True
                        logger.info(f"Goal reached! Reward: {reward}")
                    if terminated:
                        session_active = False

                        # Display periodic statistics updates if display_statistics enabled

                elif input_type == "command":
                    # Handle special commands (reset environment, display help, quit session)
                    session_active = _process_manual_session_command(
                        input_value=input_value,
                        env=env,
                        session_stats=session_stats,
                        logger=logger,
                        session_active=session_active,
                    )

                elif input_type == "error":
                    logger.error(f"Keyboard input error: {input_value}")
                    # Continue session but log error for debugging

            except queue.Empty:
                # No input received within timeout - continue main loop
                continue

            # Check if environment episode ended
            _log_manual_episode_end_if_needed(terminated, truncated, logger)

        # Calculate final session statistics
        session_end_time = time.time()
        session_stats["session_duration"] = session_end_time - session_start_time

        # Clean up keyboard input thread and environment resources upon session completion
        logger.info("Cleaning up manual control session...")
        # Keyboard thread is daemon thread and will terminate automatically

        # Generate comprehensive session report with user performance, learning metrics, and recommendations
        logger.info("Generating session report...")
        return generate_session_report(session_stats, enable_performance_tracking)
    except Exception as e:
        logger.error(f"Manual control session error: {e}")
        session_stats["error"] = str(e)
        session_stats["session_duration"] = time.time() - session_start_time
        return session_stats


def _process_manual_session_action(
    env: gym.Env,
    input_value,
    session_stats: dict,
    display_statistics: bool,
    enable_performance_tracking: bool,
    last_statistics_display: int,
) -> tuple:
    # Execute environment step with user action and capture response
    step_start_time = time.time()

    action_int = input_value.value if isinstance(input_value, Action) else input_value
    observation, reward, terminated, truncated, info = env.step(action_int)

    step_end_time = time.time()

    # Update session statistics including steps taken, reward accumulated, and user performance
    session_stats["total_actions"] += 1
    session_stats["successful_actions"] += 1
    session_stats["total_reward"] += reward
    session_stats["action_history"].append(action_int)

    # Track position history if available in info
    if "agent_xy" in info:
        session_stats["position_history"].append(info["agent_xy"])

    # Record performance metrics if enabled
    if enable_performance_tracking:
        step_duration = (step_end_time - step_start_time) * 1000  # Convert to ms
        session_stats["performance_metrics"]["step_times"].append(step_duration)

    # Update real-time visualization with new agent position and environment state
    render_start_time = time.time()
    env.render()
    render_end_time = time.time()

    if enable_performance_tracking:
        render_duration = (render_end_time - render_start_time) * 1000  # Convert to ms
        session_stats["performance_metrics"]["render_times"].append(render_duration)

    # Display periodic statistics updates if display_statistics enabled
    if (
        display_statistics
        and session_stats["total_actions"] - last_statistics_display
        >= STATISTICS_DISPLAY_FREQUENCY
    ):
        update_session_statistics(session_stats, info, display_to_console=True)
        last_statistics_display = session_stats["total_actions"]

    return reward, terminated, truncated, last_statistics_display


def _check_manual_session_timeout(
    *,
    logger: logging.Logger,
    session_start_time: float,
    session_timeout_seconds: int,
) -> bool:
    current_time = time.time()
    elapsed_time = current_time - session_start_time

    if elapsed_time > session_timeout_seconds:
        logger.info("Session timeout reached - ending manual control")
        return True
    return False


def _log_manual_episode_end_if_needed(
    terminated: bool, truncated: bool, logger: logging.Logger
) -> None:
    if terminated or truncated:
        logger.info("Episode ended - environment will be reset on next action")


def _process_manual_session_command(
    *,
    input_value: str,
    env: gym.Env,
    session_stats: dict,
    logger: logging.Logger,
    session_active: bool,
) -> bool:
    # Handle special commands (reset environment, display help, quit session)
    if input_value == "quit":
        logger.info("User requested session quit")
        return False
    if input_value == "reset":
        return _extracted_from__process_manual_session_command_14(
            logger, env, session_stats, session_active
        )
    if input_value == "help":
        display_control_instructions(
            env, show_environment_info=True, show_advanced_controls=True
        )
        return session_active
    return session_active


# TODO Rename this here and in `_process_manual_session_command`
def _extracted_from__process_manual_session_command_14(
    logger, env, session_stats, session_active
):
    logger.info("User requested environment reset")
    observation, info = env.reset()
    env.render()
    # Reset some session statistics but preserve overall session data
    session_stats["total_reward"] = 0.0
    session_stats["goal_reached"] = False
    return session_active


def handle_manual_input(
    input_key: str, env: gym.Env, session_state: dict, provide_feedback: bool = True
) -> dict:
    """Process individual keyboard input events with action validation, special command handling, input filtering,
    and user feedback for responsive manual control with comprehensive error handling and user guidance.

    Args:
        input_key: Keyboard input character or special key
        env: Gymnasium environment instance
        session_state: Current session state dictionary
        provide_feedback: Whether to provide user feedback for actions

    Returns:
        dict: Input processing result with action, command type, validity status, and user feedback

    Raises:
        ValidationError: If input_key or session_state parameters are invalid
    """
    logger = logging.getLogger("manual_control.input")

    # Initialize processing result dictionary
    processing_result = {
        "action_taken": None,
        "command_executed": None,
        "input_valid": False,
        "feedback_message": None,
        "session_updates": {},
    }

    try:
        # Normalize input key to lowercase and handle key variations (arrow keys, WASD)
        normalized_key = input_key.lower().strip()

        # Check if input represents movement action using KEYBOARD_MAPPING lookup
        if normalized_key in KEYBOARD_MAPPING:
            action = KEYBOARD_MAPPING[normalized_key]
            _process_manual_movement_input(
                action=action,
                env=env,
                session_state=session_state,
                provide_feedback=provide_feedback,
                logger=logger,
                processing_result=processing_result,
            )

        elif normalized_key == "q":
            processing_result |= {
                "command_executed": "quit",
                "input_valid": True,
                "feedback_message": "Quitting manual control session...",
            }

        elif normalized_key == "r":
            _process_manual_reset_input(
                env=env,
                session_state=session_state,
                logger=logger,
                processing_result=processing_result,
            )

        elif normalized_key == "h":
            processing_result |= {
                "command_executed": "help",
                "input_valid": True,
                "feedback_message": "Displaying help information...",
            }

        else:
            _process_manual_invalid_input(
                input_key=input_key,
                provide_feedback=provide_feedback,
                logger=logger,
                processing_result=processing_result,
            )

        # Log input processing with action taken, results achieved, and any errors encountered
        logger.debug(f"Input processing complete: {processing_result}")

        # Return processing result with action outcome, session updates, and user feedback information
        return processing_result

    except Exception as e:
        logger.error(f"Input processing error: {e}")
        return {
            "action_taken": None,
            "command_executed": None,
            "input_valid": False,
            "feedback_message": f"Processing error: {e}",
            "session_updates": {},
        }


def _process_manual_movement_input(
    *,
    action: Action,
    env: gym.Env,
    session_state: dict,
    provide_feedback: bool,
    logger: logging.Logger,
    processing_result: dict,
) -> None:
    # Validate action against environment action space for compatibility
    try:
        action_int = action.value if isinstance(action, Action) else action

        # Process movement action by executing environment step and capturing response
        observation, reward, terminated, truncated, info = env.step(action_int)

        # Update session_state with action results and performance tracking
        session_state["total_actions"] = session_state.get("total_actions", 0) + 1
        session_state["total_reward"] = session_state.get("total_reward", 0.0) + reward

        if "action_history" in session_state:
            session_state["action_history"].append(action_int)

        # Set processing result for successful action
        processing_result |= {
            "action_taken": action,
            "input_valid": True,
            "session_updates": {
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            },
        }

        # Provide user feedback if provide_feedback enabled (success confirmations)
        if provide_feedback:
            if reward > 0:
                processing_result["feedback_message"] = (
                    f"Excellent! Goal reached with reward: {reward}"
                )
            elif terminated:
                processing_result["feedback_message"] = "Episode completed!"
            else:
                distance = info.get("distance_to_source", "unknown")
                processing_result["feedback_message"] = (
                    f"Moved {action.name}. Distance to source: {distance}"
                )

        logger.debug(f"Successfully processed action: {action.name}")

    except Exception as e:
        logger.error(f"Failed to execute action {action}: {e}")
        processing_result["feedback_message"] = f"Action failed: {e}"


def _process_manual_reset_input(
    *,
    env: gym.Env,
    session_state: dict,
    logger: logging.Logger,
    processing_result: dict,
) -> None:
    try:
        observation, info = env.reset()
        session_state["total_reward"] = 0.0
        session_state["goal_reached"] = False

        processing_result |= {
            "command_executed": "reset",
            "input_valid": True,
            "feedback_message": "Environment reset successfully",
            "session_updates": {
                "reset": True,
                "observation": observation,
                "info": info,
            },
        }

    except Exception as e:
        logger.error(f"Reset command failed: {e}")
        processing_result["feedback_message"] = f"Reset failed: {e}"


def _process_manual_invalid_input(
    *,
    input_key: str,
    provide_feedback: bool,
    logger: logging.Logger,
    processing_result: dict,
) -> None:
    # Invalid input - provide user feedback with guidance
    processing_result |= {
        "input_valid": False,
        "feedback_message": f"Invalid input '{input_key}'. Use WASD/arrow keys for movement, Q to quit, R to reset, H for help.",
    }

    if provide_feedback:
        logger.debug(f"Invalid input received: {input_key}")


def update_session_statistics(
    session_stats: dict,
    env_info: dict,
    display_to_console: bool = True,
    include_advanced_metrics: bool = False,
) -> dict:
    """Update and display comprehensive session statistics including movement patterns, performance metrics,
    goal achievement progress, and educational insights for user learning and session analysis.

    Args:
        session_stats: Current session statistics dictionary
        env_info: Environment information from step response
        display_to_console: Whether to display statistics to console
        include_advanced_metrics: Whether to include advanced performance metrics

    Returns:
        dict: Updated statistics with calculated metrics, performance analysis, and educational insights

    Raises:
        ValidationError: If session_stats parameter is invalid
    """
    logger = logging.getLogger("manual_control.statistics")

    try:
        # Update basic statistics (total steps, session duration, actions taken)
        current_time = time.time()
        session_start_time = session_stats.get("start_time", current_time)
        session_duration = current_time - session_start_time

        session_stats["session_duration"] = session_duration
        total_actions = session_stats.get("total_actions", 0)

        (
            movement_efficiency,
            current_distance,
            total_reward,
            goal_reached,
            action_distribution,
            educational_insights,
        ) = _compute_manual_session_statistics(
            session_stats=session_stats,
            env_info=env_info,
            total_actions=total_actions,
            include_advanced_metrics=include_advanced_metrics,
            session_duration=session_duration,
        )

        # Display statistics to console if display_to_console enabled with formatted output
        if display_to_console:
            _print_manual_session_statistics(
                session_duration=session_duration,
                total_actions=total_actions,
                total_reward=total_reward,
                current_distance=current_distance,
                movement_efficiency=movement_efficiency,
                action_distribution=action_distribution,
                educational_insights=educational_insights,
                session_stats=session_stats,
            )

        # Return updated statistics dictionary with comprehensive session analysis
        logger.debug("Session statistics updated successfully")
        return session_stats

    except Exception as e:
        logger.error(f"Failed to update session statistics: {e}")
        return session_stats


def _compute_manual_session_statistics(
    *,
    session_stats: dict,
    env_info: dict,
    total_actions: int,
    include_advanced_metrics: bool,
    session_duration: float,
):
    # Calculate movement efficiency and navigation patterns from action history
    action_history = session_stats.get("action_history", [])
    position_history = session_stats.get("position_history", [])

    movement_efficiency = _calculate_manual_movement_efficiency(
        action_history, position_history, total_actions
    )

    session_stats["movement_efficiency"] = movement_efficiency

    # Track distance to goal and progress toward source location
    current_distance = env_info.get("distance_to_source", "unknown")
    session_stats["current_distance_to_source"] = current_distance

    # Compute reward accumulation and achievement metrics
    total_reward = session_stats.get("total_reward", 0.0)
    goal_reached = session_stats.get("goal_reached", False)

    # Analyze action distribution and movement preferences (directional bias)
    action_distribution = _build_manual_action_distribution(action_history)
    session_stats["action_distribution"] = action_distribution

    # Calculate advanced metrics if include_advanced_metrics enabled
    if include_advanced_metrics:
        _compute_manual_advanced_performance_metrics(session_stats)

    # Update performance indicators including steps per second and response time
    if session_duration > 0:
        session_stats["actions_per_second"] = total_actions / session_duration

    # Generate educational insights about navigation strategies and plume characteristics
    educational_insights = _generate_manual_educational_insights(
        total_actions=total_actions,
        movement_efficiency=movement_efficiency,
        action_distribution=action_distribution,
        goal_reached=goal_reached,
        current_distance=current_distance,
    )
    session_stats["educational_insights"] = educational_insights

    return (
        movement_efficiency,
        current_distance,
        total_reward,
        goal_reached,
        action_distribution,
        educational_insights,
    )


def _calculate_manual_movement_efficiency(
    action_history, position_history, total_actions: int
) -> float:
    if not (total_actions > 0 and position_history and len(position_history) >= 2):
        return 0.0

    start_pos = position_history[0]
    current_pos = position_history[-1]
    straight_distance = np.sqrt(
        (current_pos[0] - start_pos[0]) ** 2 + (current_pos[1] - start_pos[1]) ** 2
    )

    return straight_distance / total_actions


def _build_manual_action_distribution(action_history) -> dict[str, int]:
    action_distribution: dict[str, int] = {}
    if not action_history:
        return action_distribution

    for action in action_history:
        action_name = Action(action).name if isinstance(action, int) else str(action)
        action_distribution[action_name] = action_distribution.get(action_name, 0) + 1
    return action_distribution


def _compute_manual_advanced_performance_metrics(session_stats: dict) -> None:
    if not (
        "performance_metrics" in session_stats and session_stats["performance_metrics"]
    ):
        return

    perf_metrics = session_stats["performance_metrics"]

    # Calculate average step and render times
    step_times = perf_metrics.get("step_times", [])
    render_times = perf_metrics.get("render_times", [])

    if step_times:
        session_stats["avg_step_time_ms"] = np.mean(step_times)
    if render_times:
        session_stats["avg_render_time_ms"] = np.mean(render_times)


def _generate_manual_educational_insights(
    *,
    total_actions: int,
    movement_efficiency: float,
    action_distribution: dict[str, int],
    goal_reached: bool,
    current_distance,
) -> list[str]:
    educational_insights: list[str] = []

    if total_actions > 10:
        if movement_efficiency < 0.3:
            educational_insights.append(
                "Try more direct paths toward higher concentration areas"
            )
        elif movement_efficiency > 0.8:
            educational_insights.append("Excellent navigation efficiency!")

    if action_distribution:
        most_used_action = max(action_distribution, key=action_distribution.get)
        educational_insights.append(f"Most used movement: {most_used_action}")

    if goal_reached:
        educational_insights.append("Congratulations on reaching the source!")
    elif current_distance != "unknown" and isinstance(current_distance, (int, float)):
        if current_distance < 5:
            educational_insights.append(
                "Very close to the source - you're doing great!"
            )
        elif current_distance < 20:
            educational_insights.append("Getting closer to the source - keep going!")

    return educational_insights


def _print_manual_session_statistics(
    *,
    session_duration: float,
    total_actions: int,
    total_reward: float,
    current_distance,
    movement_efficiency: float,
    action_distribution: dict,
    educational_insights: list,
    session_stats: dict,
) -> None:
    # Display statistics to console if display_to_console enabled with formatted output
    print("\n" + "─" * 60)
    print("📊 SESSION STATISTICS")
    print("─" * 60)
    print(f"⏱️  Duration: {session_duration:.1f} seconds")
    print(f"🎯 Actions: {total_actions}")
    print(f"🏆 Reward: {total_reward:.2f}")
    print(f"📍 Distance to Source: {current_distance}")
    print(f"⚡ Actions/Second: {session_stats.get('actions_per_second', 0):.2f}")

    if movement_efficiency > 0:
        print(f"📈 Movement Efficiency: {movement_efficiency:.2f}")

    if action_distribution:
        print("🧭 Movement Distribution:")
        for action, count in action_distribution.items():
            percentage = (count / total_actions) * 100
            print(f"   {action}: {count} ({percentage:.1f}%)")

    if educational_insights:
        print("💡 Insights:")
        for insight in educational_insights:
            print(f"   • {insight}")

    print("─" * 60 + "\n")


def display_session_summary(
    session_results: dict,
    include_performance_analysis: bool = True,
    include_learning_recommendations: bool = True,
) -> None:
    """Generate and display comprehensive session summary with performance analysis, learning outcomes,
    navigation effectiveness, and educational recommendations for session review and improvement guidance.

    Args:
        session_results: Complete session results dictionary
        include_performance_analysis: Whether to include detailed performance analysis
        include_learning_recommendations: Whether to include learning improvement suggestions

    Raises:
        ValidationError: If session_results parameter is invalid
    """
    logger = logging.getLogger("manual_control.summary")

    try:
        _render_manual_control_session_summary(
            session_results,
            include_performance_analysis,
            include_learning_recommendations,
            logger,
        )
    except Exception as e:
        logger.error(f"Failed to display session summary: {e}")
        print(f"Error displaying summary: {e}")


# TODO Rename this here and in `display_session_summary`
def _render_manual_control_session_summary(
    session_results,
    include_performance_analysis,
    include_learning_recommendations,
    logger,
):
    # Display session overview with duration, total actions, and goal achievement status
    print("\n" + "═" * 80)
    print("🎮 MANUAL CONTROL SESSION SUMMARY")
    print("═" * 80)

    # Extract key metrics from session results
    duration = session_results.get("session_duration", 0)
    total_actions = session_results.get("total_actions", 0)
    goal_reached = session_results.get("goal_reached", False)
    total_reward = session_results.get("total_reward", 0.0)

    # Show navigation statistics including steps taken, distance covered, and movement efficiency
    movement_efficiency = session_results.get("movement_efficiency", 0)
    actions_per_second = session_results.get("actions_per_second", 0)
    current_distance = session_results.get("current_distance_to_source", "unknown")

    _print_manual_control_session_overview_and_navigation(
        duration,
        total_actions,
        goal_reached,
        total_reward,
        movement_efficiency,
        actions_per_second,
        current_distance,
    )

    # Show movement pattern analysis including directional preferences and exploration behavior
    action_distribution = session_results.get("action_distribution", {})
    _print_manual_control_movement_analysis(action_distribution)

    # Include performance analysis if include_performance_analysis enabled
    if include_performance_analysis and "performance_metrics" in session_results:
        perf_metrics = session_results["performance_metrics"]
        if perf_metrics:
            _extracted_from_display_session_summary_80(session_results)
        # Provide learning recommendations if include_learning_recommendations enabled
    if include_learning_recommendations:
        _extracted_from_display_session_summary_100(
            goal_reached,
            movement_efficiency,
            total_actions,
            action_distribution,
        )
    # Display plume navigation insights and educational takeaways
    educational_insights = session_results.get("educational_insights", [])
    if educational_insights:
        print("\n🎓 EDUCATIONAL INSIGHTS:")
        for insight in educational_insights:
            print(f"   • {insight}")

    # Show comparison with theoretical optimal paths and algorithmic approaches
    if include_performance_analysis:
        _print_manual_control_algorithmic_comparison(
            goal_reached=goal_reached,
            duration=duration,
            total_actions=total_actions,
        )

    # Provide suggestions for future sessions and skill development opportunities
    _print_manual_control_next_steps()

    logger.info("Session summary displayed successfully")


def _print_manual_control_session_overview_and_navigation(
    duration,
    total_actions,
    goal_reached,
    total_reward,
    movement_efficiency,
    actions_per_second,
    current_distance,
) -> None:
    print("\n📋 SESSION OVERVIEW:")
    print(f"   Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"   Total Actions: {total_actions}")
    print(f"   Goal Achieved: {'✅ YES' if goal_reached else '❌ No'}")
    print(f"   Total Reward: {total_reward:.2f}")

    print("\n🧭 NAVIGATION STATISTICS:")
    print(f"   Final Distance to Source: {current_distance}")
    print(f"   Movement Efficiency: {movement_efficiency:.2f}")
    print(f"   Average Actions per Second: {actions_per_second:.2f}")

    # Display reward accumulation and performance metrics
    print("\n🏆 PERFORMANCE METRICS:")
    if goal_reached:
        print("   🎯 Successfully reached the plume source!")
        time_to_goal = duration
        efficiency_rating = (
            "Excellent"
            if time_to_goal < 60
            else "Good" if time_to_goal < 180 else "Needs Improvement"
        )
        print(f"   ⏱️  Time to Goal: {time_to_goal:.1f} seconds ({efficiency_rating})")
    else:
        print("   🔍 Source not reached - try different navigation strategies")


def _print_manual_control_movement_analysis(action_distribution) -> None:
    if not action_distribution:
        return

    print("\n🗺️  MOVEMENT PATTERN ANALYSIS:")
    total_movements = sum(action_distribution.values())
    for direction, count in action_distribution.items():
        percentage = (count / total_movements) * 100 if total_movements > 0 else 0
        bar_length = int(percentage / 5)  # Scale for display
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"   {direction:>6}: {bar} {percentage:5.1f}% ({count} moves)")


def _print_manual_control_algorithmic_comparison(
    *, goal_reached: bool, duration: float, total_actions: int
) -> None:
    # Show comparison with theoretical optimal paths and algorithmic approaches
    print("\n🤖 ALGORITHMIC COMPARISON:")

    if goal_reached and duration > 0:
        human_efficiency = total_actions / max(duration, 1)  # actions per second
        print(f"   Human Performance: {human_efficiency:.2f} actions/second")
        print(
            "   💭 Algorithms like A* or gradient descent might find optimal paths faster"
        )
        print("   🧠 But human intuition can adapt to complex, dynamic environments!")
    else:
        print("   🔄 Try completing the task to unlock algorithmic comparisons")


def _print_manual_control_next_steps() -> None:
    # Provide suggestions for future sessions and skill development opportunities
    print("\n🚀 NEXT STEPS:")
    next_steps = [
        "Try different starting strategies in a new session",
        "Experiment with systematic vs. intuitive exploration",
        "Challenge yourself with faster completion times",
        "Share insights with others learning plume navigation",
    ]

    for i, step in enumerate(next_steps, 1):
        print(f"   {i}. {step}")

    print("\n" + "═" * 80)
    print("Thank you for using the Manual Control demonstration! 🎮")
    print("═" * 80 + "\n")


# TODO Rename this here and in `display_session_summary`
def _extracted_from_display_session_summary_100(
    goal_reached, movement_efficiency, total_actions, action_distribution
):
    print("\n💡 LEARNING RECOMMENDATIONS:")

    recommendations = []

    # Generate specific recommendations based on session performance
    if not goal_reached:
        recommendations.append(
            "Focus on following concentration gradients toward brighter areas"
        )
        recommendations.append(
            "Try systematic search patterns (spiral, grid-based exploration)"
        )
        recommendations.append(
            "Use the reset command (R) to practice different approaches"
        )

    if movement_efficiency < 0.3:
        recommendations.append("Work on more direct navigation paths")
        recommendations.append("Avoid excessive backtracking or random movements")

    if total_actions < 20:
        recommendations.append("Take more time to explore the environment thoroughly")
    elif total_actions > 200 and not goal_reached:
        recommendations.append("Consider more strategic navigation approaches")

    # Action distribution insights
    if action_distribution:
        movement_bias = max(action_distribution.values()) / sum(
            action_distribution.values()
        )
        if movement_bias > 0.6:
            recommendations.append("Try more balanced movement in all directions")

    # Display recommendations
    if recommendations:
        for i, recommendation in enumerate(
            recommendations[:5], 1
        ):  # Limit to 5 recommendations
            print(f"   {i}. {recommendation}")
    else:
        print(
            "   🌟 Excellent performance! Try increasing difficulty or exploring advanced features."
        )


# TODO Rename this here and in `display_session_summary`
def _extracted_from_display_session_summary_80(session_results):
    print("\n⚡ TECHNICAL PERFORMANCE:")

    avg_step_time = session_results.get("avg_step_time_ms", 0)
    avg_render_time = session_results.get("avg_render_time_ms", 0)

    if avg_step_time > 0:
        print(f"   Average Step Time: {avg_step_time:.2f} ms")
    if avg_render_time > 0:
        print(f"   Average Render Time: {avg_render_time:.2f} ms")

    # Performance rating
    if avg_step_time < 2.0:
        print("   ✅ Excellent system responsiveness")
    elif avg_step_time < 5.0:
        print("   ⚠️  Good system performance")
    else:
        print("   ❌ System may be experiencing performance issues")


def cleanup_manual_control(
    env: gym.Env,
    input_thread: threading.Thread = None,
    session_data: dict = None,
    save_session_data: bool = False,
) -> bool:
    """Clean up manual control resources including keyboard input thread termination, environment closure,
    matplotlib figure cleanup, and session data preservation for proper resource management and session archival.

    Args:
        env: Gymnasium environment instance to close
        input_thread: Keyboard input thread to terminate
        session_data: Session data to optionally save
        save_session_data: Whether to save session data for future analysis

    Returns:
        bool: True if cleanup successful, False if issues encountered during cleanup

    Raises:
        ResourceError: If critical cleanup operations fail
    """
    logger = logging.getLogger("manual_control.cleanup")
    cleanup_success = True

    try:
        logger.info("Starting manual control cleanup...")

        # Stop keyboard input thread gracefully with timeout handling
        if input_thread and input_thread.is_alive():
            logger.debug("Waiting for keyboard input thread to terminate...")
            # Since thread is daemon, it will terminate automatically
            # Give it a moment to finish current operations
            time.sleep(0.1)

            if input_thread.is_alive():
                logger.warning(
                    "Keyboard input thread still active - it will terminate with main process"
                )

        # Close environment and release computational resources
        if env:
            try:
                logger.debug("Closing Gymnasium environment...")
                env.close()
                logger.info("Environment closed successfully")
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
                cleanup_success = False

        # Clean up matplotlib figures and interactive windows
        try:
            logger.debug("Cleaning up matplotlib figures...")
            plt.close("all")  # Close all matplotlib figures

            # Clear any remaining interactive elements
            if plt.get_backend() != "Agg":
                plt.ioff()  # Turn off interactive mode

            logger.debug("Matplotlib cleanup completed")

        except Exception as e:
            logger.warning(f"Matplotlib cleanup warning: {e}")
            # Non-critical error, don't mark cleanup as failed

        # Save session data if save_session_data enabled for future analysis
        if save_session_data and session_data:
            try:
                logger.debug("Saving session data...")
                # In a full implementation, this would save to file or database
                # For now, just log the session summary
                session_duration = session_data.get("session_duration", 0)
                total_actions = session_data.get("total_actions", 0)
                goal_reached = session_data.get("goal_reached", False)

                logger.info(
                    f"Session summary - Duration: {session_duration:.1f}s, Actions: {total_actions}, Goal: {goal_reached}"
                )

            except Exception as e:
                logger.error(f"Error saving session data: {e}")
                cleanup_success = False

        # Clear any remaining input queue items to prevent memory leaks
        # Input queue cleanup is handled automatically by garbage collection

        # Log cleanup completion with resource status and any warnings
        if cleanup_success:
            logger.info("Manual control cleanup completed successfully")
        else:
            logger.warning("Manual control cleanup completed with some issues")

        # Return cleanup success status for error handling and user notification
        return cleanup_success

    except Exception as e:
        logger.error(f"Critical cleanup error: {e}")
        return False


def generate_session_report(
    session_stats: dict, include_performance: bool = True
) -> dict:
    """Generate comprehensive session report with performance analysis and educational insights.

    Args:
        session_stats: Session statistics dictionary
        include_performance: Whether to include detailed performance metrics

    Returns:
        dict: Comprehensive session report with analysis and recommendations
    """
    logger = logging.getLogger("manual_control.report")

    try:
        # Create comprehensive session report
        report = session_stats.copy()

        # Add analysis and insights
        report["session_analysis"] = {
            "completion_status": (
                "completed"
                if session_stats.get("goal_reached", False)
                else "incomplete"
            ),
            "efficiency_rating": _calculate_efficiency_rating(session_stats),
            "learning_progress": _assess_learning_progress(session_stats),
        }

        # Add performance analysis if requested
        if include_performance and "performance_metrics" in session_stats:
            report["performance_analysis"] = _analyze_performance_metrics(
                session_stats["performance_metrics"]
            )

        logger.debug("Session report generated successfully")
        return report

    except Exception as e:
        logger.error(f"Error generating session report: {e}")
        # Return basic session stats if report generation fails
        return session_stats


def _calculate_efficiency_rating(session_stats: dict) -> str:
    """Calculate efficiency rating based on session performance."""
    movement_efficiency = session_stats.get("movement_efficiency", 0)
    goal_reached = session_stats.get("goal_reached", False)
    duration = session_stats.get("session_duration", 0)

    if goal_reached:
        if duration < 60 and movement_efficiency > 0.5:
            return "excellent"
        elif duration < 180 and movement_efficiency > 0.3:
            return "good"
        else:
            return "adequate"
    else:
        if movement_efficiency > 0.4:
            return "promising"
        else:
            return "needs_improvement"


def _assess_learning_progress(session_stats: dict) -> str:
    """Assess learning progress based on session patterns."""
    total_actions = session_stats.get("total_actions", 0)
    goal_reached = session_stats.get("goal_reached", False)

    if goal_reached and total_actions < 100:
        return "advanced"
    elif goal_reached:
        return "proficient"
    elif total_actions > 50:
        return "learning"
    else:
        return "beginner"


def _analyze_performance_metrics(performance_metrics: dict) -> dict:
    """Analyze technical performance metrics."""
    analysis = {}

    step_times = performance_metrics.get("step_times", [])
    if step_times:
        analysis["step_performance"] = {
            "avg_time_ms": np.mean(step_times),
            "max_time_ms": np.max(step_times),
            "consistency": np.std(step_times)
            < 2.0,  # Low variance indicates consistent performance
        }

    render_times = performance_metrics.get("render_times", [])
    if render_times:
        analysis["render_performance"] = {
            "avg_time_ms": np.mean(render_times),
            "max_time_ms": np.max(render_times),
            "frame_rate_fps": (
                1000 / np.mean(render_times) if np.mean(render_times) > 0 else 0
            ),
        }

    return analysis


def main() -> int:
    """Main entry point for manual control demonstration orchestrating complete interactive session with setup,
    keyboard input handling, real-time visualization, performance tracking, and educational guidance for hands-on
    plume navigation learning.

    Returns:
        int: Exit code (0 for successful session, 1 for errors, 2 for user cancellation)

    Raises:
        SystemExit: On critical errors or user cancellation
    """
    # Configure logging system for manual control session with appropriate detail level
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("manual_control.main")

    env = None
    session_data = None

    try:
        # Display manual control demonstration header with instructions and objectives
        print("\n🎮 Welcome to Plume Navigation Manual Control!")
        print("This interactive demonstration allows you to manually control an agent")
        print("navigating through a plume concentration field to find the source.")

        # Setup manual control environment using setup_manual_control with configuration validation
        logger.info("Setting up manual control environment...")
        env, session_config = setup_manual_control(
            seed=DEFAULT_MANUAL_SEED,
            enable_debug_logging=False,
            check_backend_compatibility=True,
        )

        # Display comprehensive control instructions and gameplay guidance
        display_control_instructions(
            env, show_environment_info=True, show_advanced_controls=False
        )

        # Check system compatibility for keyboard input and interactive visualization
        backend_compatible = session_config.get("backend_compatible", True)
        if not backend_compatible:
            logger.warning(
                "Limited backend compatibility - some features may not work optimally"
            )

        # Initialize session tracking and performance monitoring systems
        timeout_seconds = session_config["timeout_seconds"]

        # Run interactive manual control session with real-time input processing and visualization
        logger.info("Starting manual control session...")
        session_data = run_manual_control_session(
            env=env,
            session_timeout_seconds=timeout_seconds,
            display_statistics=True,
            enable_performance_tracking=True,
        )

        # Handle session termination gracefully (user quit, goal achieved, timeout)
        if session_data.get("goal_reached"):
            logger.info(
                "Congratulations! Session completed successfully with goal reached!"
            )
        elif "error" in session_data:
            logger.error(f"Session ended with error: {session_data['error']}")
        else:
            logger.info("Session completed")

        # Display comprehensive session summary with performance analysis and learning insights
        display_session_summary(
            session_data,
            include_performance_analysis=True,
            include_learning_recommendations=True,
        )

        # Clean up all resources using cleanup_manual_control with proper termination
        cleanup_success = cleanup_manual_control(
            env=env,
            input_thread=None,  # Thread cleanup handled internally
            session_data=session_data,
            save_session_data=False,
        )

        if not cleanup_success:
            logger.warning("Cleanup completed with some issues")

        # Return appropriate exit code indicating session outcome and user experience success
        if session_data and session_data.get("goal_reached"):
            logger.info("Manual control demonstration completed successfully!")
            return 0  # Success
        elif session_data and "error" in session_data:
            logger.error("Manual control demonstration ended with errors")
            return 1  # Error
        else:
            logger.info("Manual control demonstration completed")
            return 0  # Success

    except KeyboardInterrupt:
        # Handle all errors gracefully with informative messages and recovery suggestions
        logger.info("Manual control session interrupted by user")
        print("\n🛑 Session interrupted by user. Cleaning up...")

        if env:
            cleanup_manual_control(env, session_data=session_data)

        return 2  # User cancellation

    except Exception as e:
        logger.error(f"Manual control demonstration failed: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check your environment setup and try again.")

        # Attempt cleanup even on error
        if env:
            try:
                cleanup_manual_control(env, session_data=session_data)
            except Exception as cleanup_error:
                logger.error(f"Cleanup error: {cleanup_error}")

        return 1  # Error


if __name__ == "__main__":
    """Entry point for direct script execution with proper exit code handling."""
    exit_code = main()
    sys.exit(exit_code)
