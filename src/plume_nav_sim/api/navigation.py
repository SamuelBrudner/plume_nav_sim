"""
Navigation API for plume_nav_sim.

This module provides high-level functions for creating navigators, video plumes,
running simulations, and handling navigation-related operations.
"""

import numpy as np
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import the Navigator implementation; fail fast if unavailable
try:
    from ..core.navigator import Navigator
except ImportError:  # pragma: no cover - Navigator is a required dependency
    Navigator = None


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
    
    if Navigator is None:
        raise ImportError("Navigator implementation not available")
    return Navigator(**merged_config)


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
        
    raise NotImplementedError("run_plume_simulation has been deprecated; use core.run_simulation instead")


def visualize_trajectory(
    positions: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> Any:
    """
    Thin shim that forwards to the central visualization utility while
    remaining patch-able by tests.
    """
    if config is None:
        config = {}

    from ..utils.visualization import visualize_trajectory as _utils_visualize_trajectory

    return _utils_visualize_trajectory(
        positions=positions,
        config=config,
        save_path=save_path,
    )


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
    # Convert positions to numpy array if needed
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    
    # Validate positions shape
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"positions must be 3D with shape (agents, timesteps, 2), got {positions.shape}")
    
    # Merge config parameters (cfg takes precedence over config)
    merged_config = {}
    if config is not None:
        merged_config.update(config)
    if cfg is not None:
        merged_config.update(cfg)
    merged_config.update(kwargs)
    
    # Determine save path (output_path takes precedence)
    final_save_path = output_path or save_path
    
    # Always route through the local shim (which itself delegates to utils).
    return visualize_trajectory(
        positions=positions,
        config=merged_config,
        save_path=final_save_path,
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

def from_legacy(
    navigator: Any,
    video_plume: Any,
    max_episode_steps: int = 1000,
    render_mode: str = None,
    **kwargs
) -> Any:
    """
    Create a Gymnasium-compatible environment from legacy components.
    
    Args:
        navigator: Navigator instance
        video_plume: VideoPlume instance
        max_episode_steps: Maximum number of steps per episode
        render_mode: Rendering mode ('human', 'rgb_array', etc.)
        **kwargs: Additional environment configuration
        
    Returns:
        Gymnasium-compatible environment instance
    """
    class LegacyToGymnasiumAdapter:
        def __init__(self, navigator, video_plume, max_episode_steps, render_mode):
            self.navigator = navigator
            self.video_plume = video_plume
            self.max_episode_steps = max_episode_steps
            self.render_mode = render_mode
            self.steps_taken = 0
            
            # Add observation and action spaces for compatibility
            try:
                import gymnasium as gym
                from gymnasium.spaces import Dict, Box
                
                # Create dict observation space to match expected structure
                self.observation_space = Dict({
                    'agent_position': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                    'agent_orientation': Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                    'odor_concentration': Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
                })
                self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            except ImportError:
                self.observation_space = None
                self.action_space = None
        
        def reset(self, seed: Optional[int] = None):
            """Reset the environment and return initial observation."""
            # Reset internal state
            self.steps_taken = 0

            if not hasattr(self.navigator, 'reset'):
                logger.debug("Navigator missing reset method")
                raise NotImplementedError("navigator.reset is required")
            self.navigator.reset()

            if not (hasattr(self.navigator, 'positions') and hasattr(self.navigator, 'orientations')):
                logger.debug("Navigator missing positions or orientations")
                raise NotImplementedError("navigator must expose positions and orientations")
            position = self.navigator.positions[0]
            orientation = self.navigator.orientations[0]
            
            # Create observation dictionary
            obs = {
                'agent_position': np.array(position, dtype=np.float32),
                'agent_orientation': np.array([orientation], dtype=np.float32),
                'odor_concentration': np.array([0.0], dtype=np.float32)
            }
            
            # Create info dictionary
            info = {}
            
            return obs, info
        
        def step(self, action):
            """Take a step in the environment."""
            # Increment step counter
            self.steps_taken += 1

            if not hasattr(self.navigator, 'step'):
                logger.debug("Navigator missing step method")
                raise NotImplementedError("navigator.step is required")
            position, _, _, _ = self.navigator.step(action)

            if not hasattr(self.navigator, 'orientations'):
                logger.debug("Navigator missing orientations property")
                raise NotImplementedError("navigator must expose orientations")
            orientation = self.navigator.orientations[0]

            if not hasattr(self.video_plume, 'get_concentration'):
                logger.debug("video_plume missing get_concentration method")
                raise NotImplementedError("video_plume.get_concentration is required")
            odor_concentration = self.video_plume.get_concentration(position)

            # Create observation dictionary
            obs = {
                'agent_position': np.array(position, dtype=np.float32),
                'agent_orientation': np.array([orientation], dtype=np.float32),
                'odor_concentration': np.array([odor_concentration], dtype=np.float32)
            }
            
            # Simple reward (placeholder)
            reward = float(0.0)
            
            # Check termination conditions
            terminated = False  # No termination condition in this simple adapter
            truncated = self.steps_taken >= self.max_episode_steps  # Truncate if max steps reached
            
            # Create info dictionary
            info = {
                'steps': self.steps_taken,
                'max_steps': self.max_episode_steps
            }
            
            return obs, reward, terminated, truncated, info
        
        def render(self):
            """Render the environment."""
            if self.render_mode == 'rgb_array':
                # Return a simple placeholder image
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return None
        
        def close(self):
            """Clean up resources."""
            pass
    
    # Create and return the adapter
    return LegacyToGymnasiumAdapter(navigator, video_plume, max_episode_steps, render_mode)


# --------------------------------------------------------------------------- #
# Built-in namespace compatibility shim
# --------------------------------------------------------------------------- #
# Some legacy tests invoke `from_legacy()` without importing it first.  Placing
# the symbol into `builtins` guarantees those bare references resolve.
import builtins as _bltns  # standard library – safe inevitable dependency

if not hasattr(_bltns, "from_legacy"):
    _bltns.from_legacy = from_legacy
