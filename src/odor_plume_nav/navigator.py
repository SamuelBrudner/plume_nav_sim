"""
Navigator module for odor plume simulation.

This module provides classes and functions for navigating in the odor plume environment.
"""

from typing import Tuple, Union, Any, Optional, List, Dict
import math
import numpy as np

from odor_plume_nav.config_models import NavigatorConfig


class Navigator:
    """
    Navigator class for odor plume navigation that handles both single and multiple agents.
    
    This unified Navigator replaces both SimpleNavigator and VectorizedNavigator,
    providing a consistent interface while using vectorized operations internally
    for improved performance.
    """
    
    @classmethod
    def from_config(cls, config: Dict):
        """
        Create a Navigator from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with Navigator parameters
                   Can contain keys for either single agent ('position', 'orientation', 'speed', 'max_speed')
                   or multiple agents ('positions', 'orientations', 'speeds', 'max_speeds', 'num_agents')
        
        Returns:
            Navigator instance configured with the provided parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate config using Pydantic
        try:
            validated_config = NavigatorConfig.model_validate(config)
            # Extract the validated parameters
            return cls(**validated_config.config.model_dump())
        except Exception as e:
            # Re-raise with more context if needed
            raise ValueError(f"Invalid configuration: {str(e)}") from e
        
    def __init__(self, 
                 num_agents: int = 1,
                 positions: Optional[np.ndarray] = None,
                 orientations: Optional[np.ndarray] = None,
                 speeds: Optional[np.ndarray] = None,
                 max_speeds: Optional[np.ndarray] = None,
                 position: Optional[Tuple[float, float]] = None,
                 orientation: Optional[float] = None,
                 speed: Optional[float] = None,
                 max_speed: Optional[float] = None,
                 angular_velocity: Optional[float] = None,
                 angular_velocities: Optional[np.ndarray] = None,
                 config: Optional[Dict] = None):
        """
        Initialize a navigator with either single or multiple agents.
        
        The initialization supports both the original SimpleNavigator interface
        (position, orientation, speed, max_speed) for single agents and the 
        VectorizedNavigator interface (positions, orientations, speeds, max_speeds)
        for multiple agents.
        
        Args:
            num_agents: Number of agents (only used when no arrays provided)
            positions: Array of shape (n, 2) with initial (x, y) positions for multiple agents
            orientations: Array of shape (n,) with initial orientations in degrees
            speeds: Array of shape (n,) with initial speeds
            max_speeds: Array of shape (n,) with maximum speeds
            position: Single (x, y) position tuple for single agent (alternative to positions)
            orientation: Initial orientation in degrees for single agent
            speed: Initial speed for single agent
            max_speed: Maximum speed for single agent
            angular_velocity: Initial angular velocity in degrees per second for single agent
            angular_velocities: Array of shape (n,) with initial angular velocities in degrees per second
            config: Configuration dictionary with any of the above parameters
        """
        # Extract parameters from config dictionary if provided
        params = self._extract_params_from_config(
            config, 
            num_agents, positions, orientations, speeds, max_speeds,
            position, orientation, speed, max_speed, 
            angular_velocity, angular_velocities
        )
        
        # Determine if we're in single-agent mode and prepare parameters
        params = self._prepare_parameters(params)
        
        # Initialize agent attributes from prepared parameters
        self._initialize_attributes(params)
        
    def _extract_params_from_config(self, config, 
                                   num_agents, positions, orientations, speeds, max_speeds,
                                   position, orientation, speed, max_speed, 
                                   angular_velocity, angular_velocities):
        """Extract parameters from config dictionary if provided, otherwise use direct parameters."""
        if config is None:
            return {
                'num_agents': num_agents,
                'positions': positions,
                'orientations': orientations,
                'speeds': speeds,
                'max_speeds': max_speeds,
                'position': position,
                'orientation': orientation,
                'speed': speed,
                'max_speed': max_speed,
                'angular_velocity': angular_velocity,
                'angular_velocities': angular_velocities
            }
        
        # Extract parameters from config, using provided args as defaults
        return {
            'num_agents': config.get('num_agents', num_agents),
            'positions': config.get('positions', positions),
            'orientations': config.get('orientations', orientations),
            'speeds': config.get('speeds', speeds),
            'max_speeds': config.get('max_speeds', max_speeds),
            'position': config.get('position', position),
            'orientation': config.get('orientation', orientation),
            'speed': config.get('speed', speed),
            'max_speed': config.get('max_speed', max_speed),
            'angular_velocity': config.get('angular_velocity', angular_velocity),
            'angular_velocities': config.get('angular_velocities', angular_velocities)
        }
    
    def _prepare_parameters(self, params):
        """Prepare parameters based on single vs multi-agent mode."""
        # Check if single agent parameters are provided
        single_agent_params = [
            params['position'], params['orientation'], 
            params['speed'], params['max_speed'], 
            params['angular_velocity']
        ]

        # Determine if we're in single-agent mode
        if any(param is not None for param in single_agent_params):
            self._convert_single_agent_to_vectorized(params)
        else:
            # Check if we're in single-agent mode with just num_agents=1
            multi_agent_params = [
                params['positions'], params['orientations'], 
                params['speeds'], params['max_speeds'],
                params['angular_velocities']
            ]
            params['_single_agent'] = (
                params['num_agents'] == 1 and 
                all(param is None for param in multi_agent_params)
            )

        # Determine the number of agents
        n_agents = self._determine_num_agents(params)
        params['n_agents'] = n_agents

        return params

    def _convert_single_agent_to_vectorized(self, params):
        """
        Convert single-agent parameters to vectorized arrays for internal representation.
        
        This enables unified processing for both single-agent and multi-agent scenarios
        by ensuring all parameters are stored in array format.
        
        Args:
            params: Dictionary of parameters
        """
        params['_single_agent'] = True

        # Convert single-agent params to arrays
        params['positions'] = np.array([params['position'] or (0.0, 0.0)]).reshape(1, 2)
        params['orientations'] = np.array([params['orientation'] or 0.0])
        params['speeds'] = np.array([params['speed'] or 0.0])
        params['max_speeds'] = np.array([params['max_speed'] or 1.0])
        params['angular_velocities'] = np.array([params['angular_velocity'] or 0.0])
        params['num_agents'] = 1
    
    def _determine_num_agents(self, params):
        """Determine the number of agents from array shapes or num_agents parameter."""
        # Try to get the number of agents from the first non-None array parameter
        array_params = ['positions', 'orientations', 'speeds', 'max_speeds', 'angular_velocities']
        
        try:
            # Use next() with a generator expression to find the first non-None array
            first_array_param = next(
                param for param in array_params 
                if params[param] is not None
            )
            
            # Return the shape of the first dimension of the array
            return params[first_array_param].shape[0]
        except StopIteration:
            # If no arrays are provided, return the num_agents parameter
            return params['num_agents']
    
    def _initialize_orientations(self, params, n_agents):
        """Initialize orientations with proper normalization."""
        if params['orientations'] is not None:
            return self._normalize_orientations(params['orientations'])
        else:
            return np.zeros(n_agents)
    
    def _initialize_speeds(self, params, n_agents):
        """Initialize speeds with proper constraints."""
        if params['speeds'] is not None:
            return self._constrain_speeds(params['speeds'])
        else:
            return np.zeros(n_agents)
    
    def _initialize_angular_velocities(self, params, n_agents):
        """Initialize angular velocities."""
        if params['angular_velocities'] is not None:
            return params['angular_velocities']
        else:
            return np.zeros(n_agents)
    
    def _initialize_attributes(self, params):
        """Initialize object attributes from prepared parameters."""
        # Set single agent flag
        self._single_agent = params['_single_agent']
        
        # Initialize positional attributes
        n_agents = params['n_agents']
        self.positions = params['positions'] if params['positions'] is not None else np.zeros((n_agents, 2))
        self.max_speeds = params['max_speeds'] if params['max_speeds'] is not None else np.ones(n_agents)
        
        # Initialize other attributes using helper methods
        self.orientations = self._initialize_orientations(params, n_agents)
        self.speeds = self._initialize_speeds(params, n_agents)
        self.angular_velocities = self._initialize_angular_velocities(params, n_agents)
    
    # ----- Single agent interface (for backward compatibility) -----
    
    @property
    def orientation(self) -> float:
        """Get the current orientation in degrees for the first agent."""
        return self.orientations[0]
    
    @property
    def speed(self) -> float:
        """Get the current speed for the first agent."""
        return self.speeds[0]
    
    @property
    def max_speed(self) -> float:
        """Get the maximum speed for the first agent."""
        return self.max_speeds[0]
    
    @property
    def position(self) -> List[float]:
        """Get the current position for the first agent (returns a list for backward compatibility)."""
        return self.positions[0].tolist()
    
    @position.setter
    def position(self, value: Tuple[float, float]):
        """Set the position for the first agent."""
        self.positions[0] = value
    
    @property
    def angular_velocity(self) -> float:
        """Get the current angular velocity in degrees per second for the first agent."""
        return self.angular_velocities[0]
    
    @angular_velocity.setter
    def angular_velocity(self, value: float):
        """Set the angular velocity for the first agent."""
        self.angular_velocities[0] = value
    
    def set_orientation(self, degrees: float):
        """
        Set the orientation in degrees for the first agent.
        
        Args:
            degrees: Orientation in degrees
        """
        self.set_orientation_at(0, degrees)
    
    def set_speed(self, speed: float):
        """
        Set the movement speed for the first agent.
        
        Args:
            speed: Speed value
        """
        self.set_speed_at(0, speed)
    
    def set_angular_velocity(self, degrees_per_second: float):
        """
        Set the angular velocity in degrees per second for the first agent.
        
        Args:
            degrees_per_second: Angular velocity in degrees per second
        """
        self.set_angular_velocity_at(0, degrees_per_second)
    
    def get_movement_vector(self) -> Tuple[float, float]:
        """
        Calculate the movement vector for the first agent.
        
        Returns:
            (x, y) movement vector
        """
        vectors = self.get_movement_vectors()
        return (vectors[0, 0], vectors[0, 1])
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get the current position of the first agent.
        
        Returns:
            (x, y) position
        """
        return tuple(self.positions[0])
    
    # ----- Core vectorized implementation -----
    
    def _normalize_orientations(self, orientations: np.ndarray) -> np.ndarray:
        """
        Normalize orientations to be between 0 and 360 degrees.
        
        Args:
            orientations: Array of orientations
            
        Returns:
            Normalized orientations
        """
        # Use modulo to keep values between 0 and 360
        normalized = orientations % 360
        
        # Convert negative angles to positive equivalents
        normalized = np.where(normalized < 0, normalized + 360, normalized)
        
        return normalized
    
    def _constrain_speeds(self, speeds: np.ndarray) -> np.ndarray:
        """
        Constrain speeds to be between 0 and max_speed.
        
        Args:
            speeds: Array of speeds
            
        Returns:
            Constrained speeds
        """
        # Use np.clip to keep speeds within bounds
        return np.clip(speeds, 0, self.max_speeds)
    
    def set_orientations(self, orientations: np.ndarray):
        """
        Set orientations for all agents.
        
        Args:
            orientations: Array of orientations in degrees
        """
        self.orientations = self._normalize_orientations(orientations)
    
    def set_orientation_at(self, index: int, orientation: float):
        """
        Set orientation for a specific agent.
        
        Args:
            index: Agent index
            orientation: Orientation in degrees
        """
        # Normalize single orientation
        normalized = orientation % 360
        if normalized < 0:
            normalized += 360
            
        # Update at specific index
        self.orientations[index] = normalized
    
    def set_speeds(self, speeds: np.ndarray):
        """
        Set speeds for all agents.
        
        Args:
            speeds: Array of speeds
        """
        self.speeds = self._constrain_speeds(speeds)
    
    def set_speed_at(self, index: int, speed: float):
        """
        Set speed for a specific agent.
        
        Args:
            index: Agent index
            speed: Speed value
        """
        # Constrain the speed
        constrained = max(0.0, min(speed, self.max_speeds[index]))
        
        # Update at specific index
        self.speeds[index] = constrained
    
    def set_angular_velocity_at(self, index: int, degrees_per_second: float):
        """
        Set angular velocity for a specific agent.
        
        Args:
            index: Agent index
            degrees_per_second: Angular velocity in degrees per second
        """
        self.angular_velocities[index] = degrees_per_second
    
    def set_angular_velocities(self, degrees_per_second: np.ndarray):
        """
        Set angular velocities for all agents.
        
        Args:
            degrees_per_second: Array of angular velocities in degrees per second
        """
        self.angular_velocities = degrees_per_second
    
    def get_movement_vectors(self) -> np.ndarray:
        """
        Calculate movement vectors for all agents.
        
        Returns:
            Array of shape (n, 2) with (x, y) movement vectors
        """
        # Convert orientations to radians
        radians = np.radians(self.orientations)
        
        # Calculate components using vectorized operations
        x_components = self.speeds * np.cos(radians)
        y_components = self.speeds * np.sin(radians)
        
        # Stack to form (n, 2) array of movement vectors
        return np.column_stack((x_components, y_components))
    
    def update(self, dt: float = 1.0):
        """
        Update orientations and positions based on current angular velocities, orientations, and speeds.
        
        Args:
            dt: Time step in seconds
        """
        # Update orientations based on angular velocities
        if np.any(self.angular_velocities):  # Only update if at least one agent has non-zero angular velocity
            new_orientations = self.orientations + self.angular_velocities * dt
            self.orientations = self._normalize_orientations(new_orientations)
        
        # Get movement vectors
        movement = self.get_movement_vectors()
        
        # Update positions using vectorized operations
        self.positions += movement * dt
    
    def read_single_antenna_odor(self, environment: Union[np.ndarray, Any]) -> Union[float, np.ndarray]:
        """
        Read odor values at each agent's position.
        
        This method handles both single and multiple agents. For a single agent,
        it returns a float value. For multiple agents, it returns an array of values.
        
        Args:
            environment: Either a 2D NumPy array or an object with current_frame attribute
            
        Returns:
            Odor value(s) at the current position(s)
        """
        # Check if environment is a plume object with current_frame
        if hasattr(environment, 'current_frame'):
            env_array = environment.current_frame
        else:
            env_array = environment
        
        # Get environment dimensions
        height, width = env_array.shape
        
        # Initialize odor values array
        n_agents = self.positions.shape[0]
        odor_values = np.zeros(n_agents)
        
        # Round positions to integers for array indexing
        x_pos = np.round(self.positions[:, 0]).astype(int)
        y_pos = np.round(self.positions[:, 1]).astype(int)
        
        # Create masks for positions within bounds
        x_within_bounds = (x_pos >= 0) & (x_pos < width)
        y_within_bounds = (y_pos >= 0) & (y_pos < height)
        within_bounds = x_within_bounds & y_within_bounds
        
        # For positions within bounds, get odor values
        for i in range(n_agents):
            if within_bounds[i]:
                # Note: NumPy indexing is [y, x]
                odor_value = env_array[y_pos[i], x_pos[i]]
                
                # Normalize if uint8 array (likely from video frame)
                if env_array.dtype == np.uint8:
                    odor_values[i] = odor_value / 255.0
                else:
                    odor_values[i] = odor_value
        
        # Return a single float if in single-agent mode, otherwise return the array
        return float(odor_values[0]) if self._single_agent else odor_values


# Legacy classes for backward compatibility

class SimpleNavigator(Navigator):
    """Legacy class for backward compatibility. Use Navigator instead."""
    
    @classmethod
    def from_config(cls, config: Dict):
        """
        Create a SimpleNavigator from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with navigator parameters
        
        Returns:
            SimpleNavigator instance
        """
        return cls(**config)
    
    def __init__(self, 
                 orientation: float = 0.0,
                 speed: float = 0.0,
                 position: Tuple[float, float] = (0.0, 0.0),
                 max_speed: float = 1.0,
                 angular_velocity: float = 0.0,
                 config: Optional[Dict] = None):
        """
        Initialize a simple navigator (single agent).
        
        Args:
            orientation: Initial orientation in degrees
            speed: Initial speed
            position: Initial (x, y) position
            max_speed: Maximum allowed speed
            angular_velocity: Initial angular velocity in degrees per second
            config: Optional configuration dictionary
        """
        if config is not None:
            # Convert config to keyword arguments for parent class
            super().__init__(config=config)
        else:
            # Use provided parameters
            super().__init__(position=position, orientation=orientation, 
                            speed=speed, max_speed=max_speed, angular_velocity=angular_velocity)


class VectorizedNavigator(Navigator):
    """Legacy class for backward compatibility. Use Navigator instead."""
    
    @classmethod
    def from_config(cls, config: Dict):
        """
        Create a VectorizedNavigator from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with navigator parameters
        
        Returns:
            VectorizedNavigator instance
        """
        return cls(**config)
    
    def __init__(self, num_agents=1, positions=None, orientations=None, speeds=None, max_speeds=None, angular_velocities=None, config=None):
        """
        Initialize a vectorized navigator for multiple agents.
        
        Args:
            num_agents: Number of agents to manage
            positions: NumPy array of shape (n, 2) with initial (x, y) positions
            orientations: NumPy array of shape (n,) with initial orientations in degrees
            speeds: NumPy array of shape (n,) with initial speeds
            max_speeds: NumPy array of shape (n,) with maximum speeds
            angular_velocities: NumPy array of shape (n,) with initial angular velocities in degrees per second
            config: Optional configuration dictionary
        """
        if config is not None:
            # Convert config to keyword arguments for parent class
            super().__init__(config=config)
        else:
            # Use provided parameters
            super().__init__(num_agents=num_agents, positions=positions, 
                            orientations=orientations, speeds=speeds, 
                            max_speeds=max_speeds, angular_velocities=angular_velocities)