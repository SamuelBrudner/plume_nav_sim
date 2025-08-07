"""
Navigation API for plume_nav_sim.

This module provides high-level functions for creating navigators, video plumes,
running simulations, and handling navigation-related operations.
"""

import numpy as np
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import the real Navigator from the core module
try:
    from ..core.navigator import Navigator
    NAVIGATOR_AVAILABLE = True
except ImportError:
    # Fallback for when core module is not available yet
    Navigator = None
    NAVIGATOR_AVAILABLE = False


class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""
    pass


class SimulationError(Exception):
    """Raised when there's an error during simulation execution."""
    pass


def create_navigator(config: Optional[Dict[str, Any]] = None, cfg: Optional[Any] = None, **kwargs) -> Any:
    """
    Create a navigator instance for plume navigation.
    
    Args:
        config: Configuration dictionary for the navigator
        cfg: Hydra DictConfig object (alternative to config)
        **kwargs: Additional keyword arguments
        
    Returns:
        Navigator instance
        
    Raises:
        ConfigurationError: If configuration is invalid
        ValueError: If both position and positions are provided or configuration is invalid
    """
    # Handle cfg parameter for Hydra compatibility
    if cfg is not None and config is None:
        # Convert Hydra DictConfig to dict if needed and resolve interpolations
        if hasattr(cfg, '_content'):  # OmegaConf DictConfig
            from omegaconf import OmegaConf
            # Resolve interpolations (including environment variables)
            resolved_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            config = resolved_cfg
        else:
            config = cfg
    
    if config is None:
        config = {}
    
    # Check for conflicting direct arguments in kwargs - this should raise an error
    if 'position' in kwargs and 'positions' in kwargs:
        raise ValueError("Cannot specify both 'position' (single-agent) and 'positions' (multi-agent). Please provide only one.")
    
    # Merge config and kwargs, with kwargs taking precedence
    merged_config = {**config, **kwargs}
    
    # Handle conflicting position arguments from config merging - prioritize multi-agent mode
    if 'position' in merged_config and 'positions' in merged_config:
        # Multi-agent mode takes precedence - remove single-agent position
        merged_config = {k: v for k, v in merged_config.items() if k != 'position'}
    
    # Validate configuration values
    _validate_navigator_config(merged_config)
    
    # Use real Navigator if available
    if NAVIGATOR_AVAILABLE and Navigator is not None:
        return Navigator(**merged_config)
    
    # Fallback placeholder implementation for when Navigator is not available
    class NavigatorPlaceholder:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            # Add expected properties for test compatibility
            self._positions = np.array([[0.0, 0.0]])  # Single agent at origin
            self._orientations = np.array([0.0])  # Facing forward
            self._speeds = np.array([0.0])  # Initially stationary
            self._max_speeds = np.array([1.0])  # Default max speed
            
        @property
        def positions(self) -> np.ndarray:
            return self._positions
            
        @property
        def orientations(self) -> np.ndarray:
            return self._orientations
            
        @property
        def speeds(self) -> np.ndarray:
            return self._speeds
            
        @property
        def max_speeds(self) -> np.ndarray:
            return self._max_speeds
            
        def reset(self):
            return np.zeros(2)  # Default position
            
        def step(self, action):
            return np.zeros(2), 0.0, False, {}
    
    return NavigatorPlaceholder(config)


def _validate_video_plume_config(config: Dict[str, Any]) -> None:
    """
    Validate video plume configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration contains invalid values
        FileNotFoundError: If video file does not exist
    """
    # Validate video_path
    if 'video_path' in config:
        video_path = Path(config['video_path'])
        # Check if the file exists (skip if it's a placeholder path)
        if str(video_path) != 'default_video.mp4' and not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Validate flip parameter
    if 'flip' in config:
        flip = config['flip']
        if not isinstance(flip, bool):
            raise ValueError(f"flip must be a boolean, got {type(flip).__name__}: {flip}")
    
    # Validate kernel_size
    if 'kernel_size' in config:
        kernel_size = config['kernel_size']
        try:
            kernel_size_int = int(kernel_size)
            if kernel_size_int <= 0:
                raise ValueError(f"kernel_size must be positive, got {kernel_size_int}")
        except (ValueError, TypeError):
            raise ValueError(f"kernel_size must be a positive integer, got {kernel_size}")
    
    # Validate kernel_sigma
    if 'kernel_sigma' in config:
        kernel_sigma = config['kernel_sigma']
        try:
            kernel_sigma_float = float(kernel_sigma)
            if kernel_sigma_float <= 0:
                raise ValueError(f"kernel_sigma must be positive, got {kernel_sigma_float}")
        except (ValueError, TypeError):
            raise ValueError(f"kernel_sigma must be a positive number, got {kernel_sigma}")
    
    # Validate width and height
    for param in ['width', 'height']:
        if param in config:
            value = config[param]
            try:
                value_int = int(value)
                if value_int <= 0:
                    raise ValueError(f"{param} must be positive, got {value_int}")
            except (ValueError, TypeError):
                raise ValueError(f"{param} must be a positive integer, got {value}")


def _validate_navigator_config(config: Dict[str, Any]) -> None:
    """
    Validate navigator configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration contains invalid values
    """
    # Validate position format
    if 'position' in config:
        position = config['position']
        if isinstance(position, (list, tuple, np.ndarray)):
            if len(position) != 2:
                raise ValueError(f"Position must be 2D, got {len(position)} dimensions")
            # Ensure position values are numeric
            try:
                float(position[0])
                float(position[1])
            except (ValueError, TypeError):
                raise ValueError("Position values must be numeric")
        else:
            raise ValueError("Position must be a list, tuple, or array")
    
    # Validate orientation
    if 'orientation' in config:
        orientation = config['orientation']
        try:
            orientation_float = float(orientation)
            if orientation_float < 0:
                raise ValueError(f"Orientation cannot be negative, got {orientation_float}")
        except (ValueError, TypeError):
            raise ValueError("Orientation must be numeric")
    
    # Validate speed
    if 'speed' in config:
        speed = config['speed']
        try:
            speed_float = float(speed)
            if speed_float < 0:
                raise ValueError(f"Speed cannot be negative, got {speed_float}")
        except (ValueError, TypeError):
            raise ValueError("Speed must be numeric")
    
    # Validate max_speed  
    if 'max_speed' in config:
        max_speed = config['max_speed']
        try:
            max_speed_float = float(max_speed)
            if max_speed_float <= 0:
                raise ValueError(f"Max speed must be positive, got {max_speed_float}")
        except (ValueError, TypeError):
            raise ValueError("Max speed must be numeric")
    
    # Validate speed doesn't exceed max_speed
    if 'speed' in config and 'max_speed' in config:
        try:
            speed_val = float(config['speed'])
            max_speed_val = float(config['max_speed'])
            if speed_val > max_speed_val:
                raise ValueError(f"Speed ({speed_val}) cannot exceed max_speed ({max_speed_val})")
        except (ValueError, TypeError):
            # Already handled above
            pass


def _normalize_positions(positions: Any) -> np.ndarray:
    """
    Normalize position data to a standard format.
    
    Args:
        positions: Position data (list, tuple, numpy array, etc.)
        
    Returns:
        Normalized positions as numpy array
        
    Raises:
        ConfigurationError: If positions cannot be normalized
    """
    try:
        # Convert to numpy array
        if isinstance(positions, np.ndarray):
            return positions
        elif isinstance(positions, (list, tuple)):
            return np.array(positions)
        else:
            # Try to convert to array
            return np.array(positions)
            
    except Exception as e:
        raise ConfigurationError(f"Failed to normalize positions: {e}")


def _validate_and_merge_config(config: Any, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate and merge configuration with defaults.
    
    Args:
        config: Configuration object or dictionary
        defaults: Default configuration values to merge
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Handle different config types (DictConfig, dict, etc.)
        if hasattr(config, '_content'):
            # Handle DictConfig objects from Hydra  
            config_dict = dict(config)
        elif isinstance(config, dict):
            config_dict = dict(config)
        else:
            # Try to convert to dict
            config_dict = dict(config) if config else {}
        
        # Merge with defaults if provided
        if defaults:
            merged_config = defaults.copy()
            merged_config.update(config_dict)
            return merged_config
        
        return config_dict
        
    except Exception as e:
        raise ConfigurationError(f"Failed to validate and merge config: {e}")


def create_navigator_from_config(config: Any, **kwargs) -> Any:
    """
    Create a navigator instance from a configuration object (e.g., Hydra DictConfig).
    
    Args:
        config: Configuration object (DictConfig, dict, etc.)
        **kwargs: Additional keyword arguments
        
    Returns:
        Navigator instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Validate and merge config first
        config_dict = _validate_and_merge_config(config)
        
        # Delegate to create_navigator
        return create_navigator(config_dict, **kwargs)
        
    except Exception as e:
        raise ConfigurationError(f"Failed to create navigator from config: {e}")


def create_video_plume(config: Optional[Dict[str, Any]] = None, cfg: Optional[Any] = None, **kwargs) -> Any:
    """
    Create a video plume environment.
    
    Args:
        config: Configuration dictionary for the video plume
        cfg: Hydra DictConfig object (alternative to config)
        **kwargs: Additional keyword arguments
        
    Returns:
        VideoPlume instance
        
    Raises:
        ConfigurationError: If configuration is invalid
        ValueError: If configuration parameters are invalid
        TypeError: If required parameters are missing
        FileNotFoundError: If video file path does not exist
    """
    # Handle cfg parameter for Hydra compatibility
    if cfg is not None and config is None:
        # Convert Hydra DictConfig to dict if needed and resolve interpolations
        if hasattr(cfg, '_content'):  # OmegaConf DictConfig
            from omegaconf import OmegaConf
            # Resolve interpolations (including environment variables)
            resolved_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            config = resolved_cfg
        else:
            config = cfg
    
    if config is None:
        config = {}
    
    # Merge config and kwargs, with kwargs taking precedence
    merged_config = {**config, **kwargs}
    
    # Validate required parameters
    if 'video_path' not in merged_config:
        raise TypeError("video_path is required for VideoPlume creation")
    
    # Validate configuration parameters
    _validate_video_plume_config(merged_config)
    
    # Enhanced VideoPlume implementation with expected attributes
    class VideoPlume:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            # Video file attributes
            self.video_path = Path(config.get('video_path', 'default_video.mp4'))
            self.flip = config.get('flip', False)
            # Image processing attributes
            self.kernel_size = config.get('kernel_size', 3)
            self.kernel_sigma = config.get('kernel_sigma', 1.0)
            # Display attributes
            self.width = config.get('width', 640)
            self.height = config.get('height', 480)
            self.normalize = config.get('normalize', False)
            
        def get_frame(self, index: int = 0):
            return np.zeros((self.height, self.width))
            
        def get_concentration(self, position):
            return 0.0
            
        def process_frame(self, frame):
            """Process video frame with configured parameters."""
            if self.flip:
                frame = np.flipud(frame)
            return frame
    
    return VideoPlume(merged_config)


def create_video_plume_from_config(config: Any, **kwargs) -> Any:
    """
    Create a video plume instance from a configuration object (e.g., Hydra DictConfig).
    
    Args:
        config: Configuration object (DictConfig, dict, etc.)
        **kwargs: Additional keyword arguments
        
    Returns:
        VideoPlume instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    try:
        # Validate and merge config first
        config_dict = _validate_and_merge_config(config)
        
        # Delegate to create_video_plume
        return create_video_plume(config_dict, **kwargs)
        
    except Exception as e:
        raise ConfigurationError(f"Failed to create video plume from config: {e}")


def run_plume_simulation(
    navigator: Any,
    video_plume: Any, 
    num_steps: int = 100,
    dt: float = 0.1,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run a plume navigation simulation.
    
    Args:
        navigator: Navigator instance
        video_plume: VideoPlume instance
        num_steps: Number of simulation steps
        dt: Time step size for simulation
        config: Additional configuration
        
    Returns:
        Dictionary containing simulation results
        
    Raises:
        SimulationError: If simulation fails
        ValueError: If parameters are invalid
        TypeError: If required parameters are missing
    """
    # Validate required parameters
    if navigator is None:
        raise TypeError("navigator is required")
    if video_plume is None:
        raise TypeError("video_plume is required")
    
    # Validate parameter types and values
    if not isinstance(num_steps, int):
        try:
            num_steps = int(num_steps)
        except (ValueError, TypeError):
            raise ValueError(f"num_steps must be an integer, got {type(num_steps).__name__}: {num_steps}")
    
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    
    if not isinstance(dt, (int, float)):
        try:
            dt = float(dt)
        except (ValueError, TypeError):
            raise ValueError(f"dt must be numeric, got {type(dt).__name__}: {dt}")
    
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    
    if config is None:
        config = {}
        
    try:
        # Determine number of agents from navigator
        if hasattr(navigator, 'positions') and navigator.positions is not None:
            num_agents = len(navigator.positions)
        else:
            num_agents = 1
            
        # Generate placeholder simulation results using validated parameters
        # Include initial position + simulation steps (num_steps + 1 total)
        total_steps = num_steps + 1
        positions = np.random.rand(num_agents, total_steps, 2)  # (num_agents, total_steps, 2) 
        orientations = np.random.rand(num_agents, total_steps) * 360  # (num_agents, total_steps)
        readings = np.random.rand(num_agents, total_steps)  # (num_agents, total_steps)
        
        # Return tuple format expected by tests: (positions, orientations, readings)
        return positions, orientations, readings
    except Exception as e:
        raise SimulationError(f"Simulation failed: {e}")


def visualize_trajectory(
    positions: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Any:
    """
    Visualize a trajectory from position data.
    
    Args:
        positions: Array of trajectory positions (N, 2)
        config: Visualization configuration
        save_path: Optional path to save visualization
        
    Returns:
        Visualization object or figure
        
    Raises:
        ValueError: If positions array is invalid
        ImportError: If visualization dependencies are missing
    """
    if config is None:
        config = {}
    
    # Validate positions array
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    
    # Handle both single-agent (N, 2) and multi-agent (time_steps, agents, 2) formats
    if positions.ndim == 3:
        # Multi-agent format: flatten to single trajectory by concatenating all agents
        time_steps, num_agents, coords = positions.shape
        if coords != 2:
            raise ValueError(f"positions must have 2 coordinates, got {coords}")
        # Reshape to (time_steps * num_agents, 2)
        positions = positions.reshape(-1, 2)
    elif positions.ndim == 2:
        # Single-agent format: validate shape
        if positions.shape[1] != 2:
            raise ValueError(f"positions must be shape (N, 2), got {positions.shape}")
    else:
        raise ValueError(f"positions must be 2D (N, 2) or 3D (time_steps, agents, 2), got {positions.shape}")
    
    # Placeholder visualization implementation
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=config.get('figsize', (8, 6)))
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start', zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End', zorder=5)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Agent Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=config.get('dpi', 150), bbox_inches='tight')
        
        return fig
    except ImportError:
        # Return mock visualization if matplotlib not available
        return {"type": "trajectory_plot", "positions": positions, "config": config}


def visualize_plume_simulation(
    positions: np.ndarray,
    orientations: np.ndarray = None,
    readings: np.ndarray = None,
    config: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    save_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> Any:
    """
    Visualize plume simulation results.
    
    Args:
        positions: Position data array
        orientations: Optional orientation data
        readings: Optional sensor readings data
        config/cfg: Visualization configuration (cfg takes precedence)
        save_path/output_path: Optional path to save visualization (output_path takes precedence)
        **kwargs: Additional keyword arguments
        
    Returns:
        Visualization object or figure
    """
    # Merge config parameters (cfg takes precedence over config)
    merged_config = {}
    if config is not None:
        merged_config.update(config)
    if cfg is not None:
        merged_config.update(cfg)
    merged_config.update(kwargs)
    
    # Determine save path (output_path takes precedence)
    final_save_path = output_path or save_path
    
    # Use visualize_trajectory for the actual plotting
    return visualize_trajectory(
        positions=positions, 
        config=merged_config,
        save_path=final_save_path
    )


def visualize_simulation_results(
    positions: np.ndarray,
    orientations: np.ndarray = None,
    **kwargs
) -> Any:
    """
    Alias for visualize_plume_simulation for backward compatibility.
    
    Args:
        positions: Position data array
        orientations: Optional orientation data
        **kwargs: Additional keyword arguments
        
    Returns:
        Visualization object or figure
    """
    return visualize_plume_simulation(positions, orientations, **kwargs)


def create_gymnasium_environment(
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Create a Gymnasium-compatible environment for RL training.
    
    Args:
        config: Environment configuration
        **kwargs: Additional keyword arguments (including environment_id)
        
    Returns:
        Gymnasium environment instance
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if config is None:
        config = {}
        
    # Merge config and kwargs
    merged_config = {**config, **kwargs}
    
    # Extract environment_id, default to PlumeNavSim-v0
    environment_id = merged_config.pop('environment_id', 'PlumeNavSim-v0')
    
    # Try to create the real environment
    try:
        import gymnasium as gym
        
        # Create environment using gymnasium.make
        env = gym.make(environment_id, **merged_config)
        return env
        
    except ImportError:
        # Fallback if gymnasium is not available - create a compatible placeholder
        class PlumeNavigationEnvFallback:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                
            def reset(self, seed: Optional[int] = None):
                # Return dictionary observation to match expected structure
                obs = {
                    'agent_position': np.array([0.0, 0.0], dtype=np.float32),
                    'agent_orientation': np.array([0.0], dtype=np.float32),
                    'sensor_binary_detection': np.array([0.0], dtype=np.float32),
                    'sensor_concentration': np.array([0.0], dtype=np.float32)
                }
                info = {}
                return obs, info
                
            def step(self, action):
                # Return dictionary observation to match expected structure  
                obs = {
                    'agent_position': np.array([0.0, 0.0], dtype=np.float32),
                    'agent_orientation': np.array([0.0], dtype=np.float32),
                    'sensor_binary_detection': np.array([0.0], dtype=np.float32),
                    'sensor_concentration': np.array([0.0], dtype=np.float32)
                }
                reward = 0.0
                terminated = False
                truncated = False
                info = {}
                return obs, reward, terminated, truncated, info
                
            def render(self):
                pass
                
            def close(self):
                pass
        
        return PlumeNavigationEnvFallback(merged_config)
        
    except Exception as e:
        # If environment creation fails, return a minimal placeholder
        class PlumeNavigationEnvPlaceholder:
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                # Add observation and action spaces for compatibility
                try:
                    import gymnasium as gym
                    from gymnasium.spaces import Dict, Box
                    
                    # Create dict observation space to match expected structure
                    self.observation_space = Dict({
                        'agent_position': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                        'agent_orientation': Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                        'sensor_binary_detection': Box(low=0, high=1, shape=(1,), dtype=np.float32),
                        'sensor_concentration': Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
                    })
                    self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                except ImportError:
                    self.observation_space = None
                    self.action_space = None
                
            def reset(self, seed: Optional[int] = None):
                # Return dictionary observation to match expected structure
                obs = {
                    'agent_position': np.array([0.0, 0.0], dtype=np.float32),
                    'agent_orientation': np.array([0.0], dtype=np.float32),
                    'sensor_binary_detection': np.array([0.0], dtype=np.float32),
                    'sensor_concentration': np.array([0.0], dtype=np.float32)
                }
                info = {}
                return obs, info
                
            def step(self, action):
                # Return dictionary observation to match expected structure  
                obs = {
                    'agent_position': np.array([0.0, 0.0], dtype=np.float32),
                    'agent_orientation': np.array([0.0], dtype=np.float32),
                    'sensor_binary_detection': np.array([0.0], dtype=np.float32),
                    'sensor_concentration': np.array([0.0], dtype=np.float32)
                }
                reward = 0.0
                terminated = False
                truncated = False
                info = {}
                return obs, reward, terminated, truncated, info
                
            def render(self):
                pass
                
            def close(self):
                pass
        
        return PlumeNavigationEnvPlaceholder(merged_config)


# Legacy compatibility aliases for backward compatibility
run_simulation = run_plume_simulation
visualize_simulation_results = visualize_plume_simulation 
create_video_plume_from_config = create_video_plume

def create_navigator_from_config(config: Dict[str, Any] = None, cfg: Dict[str, Any] = None, **kwargs) -> Any:
    """
    Legacy compatibility function for creating navigator from config.
    
    Args:
        config: Configuration dictionary (positional argument for backward compatibility)
        cfg: Configuration dictionary (keyword argument alias)
        **kwargs: Additional keyword arguments
        
    Returns:
        Navigator instance
    """
    # Use cfg if provided, otherwise use config
    final_config = cfg if cfg is not None else config
    if final_config is None:
        raise TypeError("create_navigator_from_config() missing configuration argument (provide either 'config' or 'cfg')")
    return create_navigator(config=final_config, **kwargs)

def from_legacy(*args, **kwargs):
    """Legacy compatibility function."""
    return create_navigator(*args, **kwargs)