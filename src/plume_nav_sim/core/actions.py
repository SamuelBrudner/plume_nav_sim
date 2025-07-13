"""
Action interface implementations for standardized action processing.

This module implements the ActionInterfaceProtocol interface for unified action handling
across different RL frameworks and navigation strategies. Provides seamless switching
between continuous and discrete control paradigms via configuration.

Key Components:
- ActionInterfaceProtocol: Protocol interface defining standardized action processing
- Continuous2DAction: Continuous 2D navigation control with velocity commands
- CardinalDiscreteAction: Discrete directional movement (N, S, E, W, NE, NW, SE, SW)

The action interface layer enables:
- Unified action translation across different RL frameworks (Stable-Baselines3, Ray RLlib, etc.)
- Seamless switching between action space types via Hydra configuration
- Efficient action validation and transformation with minimal performance overhead
- Support for both velocity-based and directional control paradigms

Performance Requirements:
- Action translation: <0.1ms per agent for minimal control overhead
- Validation: <0.05ms per agent for constraint checking
- Memory efficiency: <100 bytes per action for structured representations

Examples:
    Continuous 2D action interface:
    >>> action_interface = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)
    >>> action = np.array([1.5, 15.0])  # [linear_velocity, angular_velocity]
    >>> nav_command = action_interface.translate_action(action)
    
    Cardinal discrete action interface:
    >>> action_interface = CardinalDiscreteAction(speed=1.0)
    >>> action = 2  # East direction
    >>> nav_command = action_interface.translate_action(action)
    
    Factory function for configuration-driven creation:
    >>> config = {'type': 'Continuous2D', 'max_velocity': 2.0}
    >>> action_interface = create_action_interface(config)
"""

from __future__ import annotations
from typing import Union, Optional, Dict, Any, Tuple, List, TYPE_CHECKING
import numpy as np

# Try to import ActionInterfaceProtocol from protocols, define locally if not available
try:
    from .protocols import ActionInterfaceProtocol
except ImportError:
    # Define ActionInterfaceProtocol locally - this will be moved to protocols.py by another agent
    from typing import Protocol, runtime_checkable
    
    @runtime_checkable
    class ActionInterfaceProtocol(Protocol):
        """
        Protocol defining standardized action processing interface.
        
        This protocol ensures consistent action handling across different control
        modalities while maintaining type safety and framework compatibility.
        All action interface implementations must provide these core methods.
        """
        
        def translate_action(self, action: Union[int, np.ndarray]) -> Dict[str, Any]:
            """
            Translate RL framework action to navigation command.
            
            Args:
                action: Action from RL framework (array for continuous, int for discrete)
                
            Returns:
                Dict[str, Any]: Navigation command dictionary
            """
            ...
        
        def validate_action(self, action: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
            """
            Validate and constrain action to valid range.
            
            Args:
                action: Action to validate
                
            Returns:
                Union[int, np.ndarray]: Validated action
            """
            ...
        
        def get_action_space(self) -> Optional[spaces.Space]:
            """
            Get Gymnasium action space for this interface.
            
            Returns:
                Optional[spaces.Space]: Action space or None if Gymnasium unavailable
            """
            ...

# Gymnasium imports for action space construction
try:
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    # Fallback for environments without Gymnasium
    try:
        import gym.spaces as spaces
        GYMNASIUM_AVAILABLE = True
    except ImportError:
        spaces = None
        GYMNASIUM_AVAILABLE = False


class Continuous2DAction(ActionInterfaceProtocol):
    """
    Continuous 2D action interface for continuous navigation control.
    
    This implementation supports continuous 2D navigation with velocity commands,
    providing seamless integration with RL frameworks that use continuous action
    spaces (e.g., PPO, SAC, TD3).
    
    The action space consists of two continuous values:
    - Linear velocity: Controls forward/backward movement speed
    - Angular velocity: Controls rotation rate in degrees per second
    
    Features:
    - Configurable velocity bounds with dynamic adjustment
    - Efficient action validation with clipping to valid ranges
    - Gymnasium Box space generation for RL framework integration
    - Performance optimized for real-time control applications
    
    Performance Requirements:
    - translate_action(): <0.05ms per call for minimal control overhead
    - validate_action(): <0.02ms per call for efficient constraint checking
    - Memory usage: <200 bytes per instance for lightweight operation
    
    Examples:
        Basic continuous action processing:
        >>> action_interface = Continuous2DAction(max_velocity=2.0, max_angular_velocity=45.0)
        >>> action = np.array([1.5, 15.0])
        >>> nav_command = action_interface.translate_action(action)
        >>> # Returns: {'linear_velocity': 1.5, 'angular_velocity': 15.0}
        
        Action validation with bounds checking:
        >>> action = np.array([3.0, 60.0])  # Exceeds bounds
        >>> valid_action = action_interface.validate_action(action)
        >>> # Returns: [2.0, 45.0] (clipped to bounds)
        
        Dynamic bounds adjustment:
        >>> action_interface.set_bounds(max_velocity=3.0, max_angular_velocity=60.0)
        >>> new_space = action_interface.get_action_space()
    """
    
    def __init__(
        self,
        max_velocity: float = 2.0,
        max_angular_velocity: float = 45.0,
        min_velocity: float = -2.0,
        min_angular_velocity: float = -45.0
    ):
        """
        Initialize continuous 2D action interface.
        
        Args:
            max_velocity: Maximum linear velocity in units per time step
            max_angular_velocity: Maximum angular velocity in degrees per second
            min_velocity: Minimum linear velocity (negative for backward movement)
            min_angular_velocity: Minimum angular velocity (negative for counter-rotation)
            
        Raises:
            ValueError: If velocity bounds are invalid (min >= max)
        """
        if min_velocity >= max_velocity:
            raise ValueError(f"min_velocity ({min_velocity}) must be less than max_velocity ({max_velocity})")
        if min_angular_velocity >= max_angular_velocity:
            raise ValueError(f"min_angular_velocity ({min_angular_velocity}) must be less than max_angular_velocity ({max_angular_velocity})")
        
        self._max_velocity = float(max_velocity)
        self._max_angular_velocity = float(max_angular_velocity)
        self._min_velocity = float(min_velocity)
        self._min_angular_velocity = float(min_angular_velocity)
    
    def translate_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Translate RL action to navigation command.
        
        Args:
            action: Action array with shape (2,) containing [linear_velocity, angular_velocity]
                   or scalar values for single component actions
                   
        Returns:
            Dict[str, Any]: Navigation command dictionary with keys:
                - 'linear_velocity': Desired linear velocity
                - 'angular_velocity': Desired angular velocity
                - 'action_type': 'continuous_2d' for identification
                
        Raises:
            ValueError: If action shape is invalid
            TypeError: If action is not a numpy array
            
        Notes:
            Action values are automatically validated and clipped to bounds.
            Performance optimized with vectorized operations for efficiency.
            
        Examples:
            Standard 2D velocity command:
            >>> action = np.array([1.5, 20.0])
            >>> command = action_interface.translate_action(action)
            >>> # Returns: {'linear_velocity': 1.5, 'angular_velocity': 20.0, 'action_type': 'continuous_2d'}
            
            Single component action (linear only):
            >>> action = np.array([1.0])
            >>> command = action_interface.translate_action(action)
            >>> # Returns: {'linear_velocity': 1.0, 'angular_velocity': 0.0, 'action_type': 'continuous_2d'}
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # Validate and handle different action shapes
        if action.shape == ():
            # Scalar action - treat as linear velocity only
            linear_velocity = float(action)
            angular_velocity = 0.0
        elif action.shape == (1,):
            # Single element array - treat as linear velocity only
            linear_velocity = float(action[0])
            angular_velocity = 0.0
        elif action.shape == (2,):
            # Standard 2D action
            linear_velocity = float(action[0])
            angular_velocity = float(action[1])
        else:
            raise ValueError(f"Invalid action shape: {action.shape}. Expected (), (1,), or (2,)")
        
        # Validate action bounds
        validated_action = self.validate_action(np.array([linear_velocity, angular_velocity]))
        
        return {
            'linear_velocity': float(validated_action[0]),
            'angular_velocity': float(validated_action[1]),
            'action_type': 'continuous_2d'
        }
    
    def validate_action(self, action: np.ndarray) -> np.ndarray:
        """
        Validate and constrain action values within bounds.
        
        Args:
            action: Action array to validate
            
        Returns:
            np.ndarray: Validated action with bounds applied
            
        Notes:
            Uses numpy.clip for efficient bounds checking.
            Invalid values (NaN, inf) are replaced with zero.
            
        Examples:
            Clip action to bounds:
            >>> action = np.array([3.0, 60.0])  # Exceeds bounds
            >>> valid_action = action_interface.validate_action(action)
            >>> # Returns: [2.0, 45.0] (clipped to max bounds)
            
            Handle invalid values:
            >>> action = np.array([np.nan, np.inf])
            >>> valid_action = action_interface.validate_action(action)
            >>> # Returns: [0.0, 0.0] (replaced invalid values)
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # Handle different action shapes
        if action.shape == ():
            action = np.array([action, 0.0])
        elif action.shape == (1,):
            action = np.array([action[0], 0.0])
        elif action.shape != (2,):
            raise ValueError(f"Invalid action shape for validation: {action.shape}")
        
        # Replace invalid values with zero
        action = np.where(np.isfinite(action), action, 0.0)
        
        # Clip to bounds
        validated_action = np.clip(
            action,
            [self._min_velocity, self._min_angular_velocity],
            [self._max_velocity, self._max_angular_velocity]
        )
        
        return validated_action.astype(np.float32)
    
    def get_action_space(self) -> Optional[spaces.Space]:
        """
        Get Gymnasium action space for this interface.
        
        Returns:
            Optional[spaces.Space]: Box action space with velocity bounds,
                                   or None if Gymnasium is not available
                                   
        Notes:
            Action space shape is (2,) for [linear_velocity, angular_velocity].
            Bounds are set to the configured min/max velocity values.
            Uses float32 dtype for memory efficiency and RL framework compatibility.
            
        Examples:
            Get action space for RL training:
            >>> action_space = action_interface.get_action_space()
            >>> assert action_space.shape == (2,)
            >>> assert action_space.low[0] == -2.0  # min_velocity
            >>> assert action_space.high[0] == 2.0   # max_velocity
        """
        if not GYMNASIUM_AVAILABLE:
            return None
        
        return spaces.Box(
            low=np.array([self._min_velocity, self._min_angular_velocity], dtype=np.float32),
            high=np.array([self._max_velocity, self._max_angular_velocity], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
    
    def set_bounds(
        self,
        max_velocity: Optional[float] = None,
        max_angular_velocity: Optional[float] = None,
        min_velocity: Optional[float] = None,
        min_angular_velocity: Optional[float] = None
    ) -> None:
        """
        Update velocity bounds dynamically.
        
        Args:
            max_velocity: New maximum linear velocity (optional)
            max_angular_velocity: New maximum angular velocity (optional)
            min_velocity: New minimum linear velocity (optional)
            min_angular_velocity: New minimum angular velocity (optional)
            
        Raises:
            ValueError: If new bounds are invalid (min >= max)
            
        Notes:
            Only specified bounds are updated; others remain unchanged.
            Bounds are validated to ensure min < max for both velocity types.
            
        Examples:
            Update maximum velocities:
            >>> action_interface.set_bounds(max_velocity=3.0, max_angular_velocity=60.0)
            
            Update all bounds:
            >>> action_interface.set_bounds(
            ...     min_velocity=-1.0, max_velocity=3.0,
            ...     min_angular_velocity=-30.0, max_angular_velocity=60.0
            ... )
        """
        # Update bounds if provided
        if max_velocity is not None:
            self._max_velocity = float(max_velocity)
        if max_angular_velocity is not None:
            self._max_angular_velocity = float(max_angular_velocity)
        if min_velocity is not None:
            self._min_velocity = float(min_velocity)
        if min_angular_velocity is not None:
            self._min_angular_velocity = float(min_angular_velocity)
        
        # Validate bounds
        if self._min_velocity >= self._max_velocity:
            raise ValueError(f"min_velocity ({self._min_velocity}) must be less than max_velocity ({self._max_velocity})")
        if self._min_angular_velocity >= self._max_angular_velocity:
            raise ValueError(f"min_angular_velocity ({self._min_angular_velocity}) must be less than max_angular_velocity ({self._max_angular_velocity})")
    
    def get_max_velocity(self) -> float:
        """
        Get current maximum linear velocity bound.
        
        Returns:
            float: Maximum linear velocity
            
        Examples:
            >>> max_vel = action_interface.get_max_velocity()
            >>> print(f"Max velocity: {max_vel}")
        """
        return self._max_velocity
    
    def get_max_angular_velocity(self) -> float:
        """
        Get current maximum angular velocity bound.
        
        Returns:
            float: Maximum angular velocity in degrees per second
            
        Examples:
            >>> max_ang_vel = action_interface.get_max_angular_velocity()
            >>> print(f"Max angular velocity: {max_ang_vel} deg/s")
        """
        return self._max_angular_velocity


class CardinalDiscreteAction(ActionInterfaceProtocol):
    """
    Cardinal discrete action interface for discrete directional movement.
    
    This implementation supports discrete directional movement with cardinal
    and intercardinal directions, providing integration with RL frameworks
    that use discrete action spaces (e.g., DQN, A2C).
    
    Supported directions:
    - 4-direction mode: North, South, East, West
    - 8-direction mode: N, S, E, W, NE, NW, SE, SW (default)
    - Stay action: Remain in current position (always available)
    
    Features:
    - Configurable movement speed with dynamic adjustment
    - Efficient action mapping with O(1) lookup for translation
    - Gymnasium Discrete space generation for RL framework integration
    - Support for both 4-direction and 8-direction movement modes
    
    Performance Requirements:
    - translate_action(): <0.03ms per call for minimal control overhead
    - validate_action(): <0.01ms per call for efficient constraint checking
    - Memory usage: <150 bytes per instance for lightweight operation
    
    Examples:
        Basic discrete action processing:
        >>> action_interface = CardinalDiscreteAction(speed=1.0, use_8_directions=True)
        >>> action = 2  # East direction
        >>> nav_command = action_interface.translate_action(action)
        >>> # Returns: {'linear_velocity': 1.0, 'angular_velocity': 0.0, 'direction': 'EAST'}
        
        8-direction movement:
        >>> action = 4  # Northeast direction
        >>> nav_command = action_interface.translate_action(action)
        >>> # Returns: {'linear_velocity': 0.707, 'angular_velocity': 0.0, 'direction': 'NORTHEAST'}
        
        Speed adjustment:
        >>> action_interface.set_speed(2.0)
        >>> action = 1  # North direction with new speed
        >>> nav_command = action_interface.translate_action(action)
        >>> # Returns: {'linear_velocity': 2.0, 'angular_velocity': 0.0, 'direction': 'NORTH'}
    """
    
    def __init__(
        self,
        speed: float = 1.0,
        use_8_directions: bool = True,
        include_stay_action: bool = True
    ):
        """
        Initialize cardinal discrete action interface.
        
        Args:
            speed: Movement speed for all directions in units per time step
            use_8_directions: If True, use 8 directions (N,S,E,W,NE,NW,SE,SW), 
                             if False, use 4 directions (N,S,E,W)
            include_stay_action: If True, include stay-in-place action (action 0)
            
        Raises:
            ValueError: If speed is negative or zero
        """
        if speed <= 0:
            raise ValueError(f"Speed must be positive, got {speed}")
        
        self._speed = float(speed)
        self._use_8_directions = bool(use_8_directions)
        self._include_stay_action = bool(include_stay_action)
        
        # Build action mapping
        self._action_mapping = self._build_action_mapping()
        self._num_actions = len(self._action_mapping)
    
    def _build_action_mapping(self) -> Dict[int, Dict[str, Any]]:
        """
        Build mapping from discrete actions to movement vectors.
        
        Returns:
            Dict[int, Dict[str, Any]]: Mapping from action indices to movement data
        """
        mapping = {}
        action_idx = 0
        
        # Stay action (if enabled)
        if self._include_stay_action:
            mapping[action_idx] = {
                'direction': 'STAY',
                'velocity_x': 0.0,
                'velocity_y': 0.0,
                'linear_velocity': 0.0,
                'angular_velocity': 0.0
            }
            action_idx += 1
        
        # Cardinal directions (always included)
        cardinal_directions = [
            ('NORTH', 0.0, -1.0),      # North: negative y (up)
            ('SOUTH', 0.0, 1.0),       # South: positive y (down)
            ('EAST', 1.0, 0.0),        # East: positive x (right)
            ('WEST', -1.0, 0.0),       # West: negative x (left)
        ]
        
        for direction, vx, vy in cardinal_directions:
            mapping[action_idx] = {
                'direction': direction,
                'velocity_x': vx * self._speed,
                'velocity_y': vy * self._speed,
                'linear_velocity': self._speed,
                'angular_velocity': 0.0
            }
            action_idx += 1
        
        # Intercardinal directions (8-direction mode only)
        if self._use_8_directions:
            # Diagonal speed is reduced by sqrt(2) to maintain consistent step size
            diagonal_speed = self._speed / np.sqrt(2)
            intercardinal_directions = [
                ('NORTHEAST', 1.0, -1.0),
                ('NORTHWEST', -1.0, -1.0),
                ('SOUTHEAST', 1.0, 1.0),
                ('SOUTHWEST', -1.0, 1.0),
            ]
            
            for direction, vx, vy in intercardinal_directions:
                mapping[action_idx] = {
                    'direction': direction,
                    'velocity_x': vx * diagonal_speed,
                    'velocity_y': vy * diagonal_speed,
                    'linear_velocity': diagonal_speed,
                    'angular_velocity': 0.0
                }
                action_idx += 1
        
        return mapping
    
    def translate_action(self, action: Union[int, np.ndarray]) -> Dict[str, Any]:
        """
        Translate discrete action to navigation command.
        
        Args:
            action: Discrete action index or array containing single index
                   
        Returns:
            Dict[str, Any]: Navigation command dictionary with keys:
                - 'linear_velocity': Computed linear velocity magnitude
                - 'angular_velocity': Angular velocity (always 0.0 for discrete)
                - 'velocity_x': X-component of velocity vector
                - 'velocity_y': Y-component of velocity vector
                - 'direction': Direction name (e.g., 'NORTH', 'NORTHEAST', 'STAY')
                - 'action_type': 'cardinal_discrete' for identification
                
        Raises:
            ValueError: If action index is invalid
            TypeError: If action is not an integer or array
            
        Notes:
            Diagonal movements are automatically normalized to maintain
            consistent movement distance across all directions.
            
        Examples:
            Cardinal direction action:
            >>> action = 1  # North
            >>> command = action_interface.translate_action(action)
            >>> # Returns: {'linear_velocity': 1.0, 'angular_velocity': 0.0, 
            >>> #          'velocity_x': 0.0, 'velocity_y': -1.0, 'direction': 'NORTH'}
            
            Diagonal direction action (8-direction mode):
            >>> action = 4  # Northeast  
            >>> command = action_interface.translate_action(action)
            >>> # Returns: {'linear_velocity': 0.707, 'angular_velocity': 0.0,
            >>> #          'velocity_x': 0.707, 'velocity_y': -0.707, 'direction': 'NORTHEAST'}
        """
        # Handle numpy array input
        if isinstance(action, np.ndarray):
            if action.shape == ():
                action = int(action.item())
            elif action.shape == (1,):
                action = int(action[0])
            else:
                raise ValueError(f"Invalid action array shape: {action.shape}")
        elif not isinstance(action, (int, np.integer)):
            raise TypeError(f"Action must be integer or array, got {type(action)}")
        
        action = int(action)
        
        # Validate action
        if action not in self._action_mapping:
            raise ValueError(f"Invalid action {action}. Valid actions: 0-{self._num_actions-1}")
        
        # Get movement data
        movement_data = self._action_mapping[action].copy()
        movement_data['action_type'] = 'cardinal_discrete'
        
        return movement_data
    
    def validate_action(self, action: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """
        Validate and constrain discrete action to valid range.
        
        Args:
            action: Discrete action to validate
            
        Returns:
            Union[int, np.ndarray]: Validated action clipped to valid range
            
        Notes:
            Invalid actions are clipped to the nearest valid action index.
            Negative actions are clipped to 0, actions >= num_actions are clipped to max.
            
        Examples:
            Clip invalid action:
            >>> action = 10  # Invalid (too high)
            >>> valid_action = action_interface.validate_action(action)
            >>> # Returns: 8 (clipped to max valid action)
            
            Handle negative action:
            >>> action = -1  # Invalid (negative)
            >>> valid_action = action_interface.validate_action(action)
            >>> # Returns: 0 (clipped to minimum)
        """
        if isinstance(action, np.ndarray):
            original_shape = action.shape
            action_flat = action.flatten()
            validated_flat = np.clip(action_flat, 0, self._num_actions - 1)
            return validated_flat.reshape(original_shape).astype(np.int32)
        else:
            return int(np.clip(action, 0, self._num_actions - 1))
    
    def get_action_space(self) -> Optional[spaces.Space]:
        """
        Get Gymnasium action space for this interface.
        
        Returns:
            Optional[spaces.Space]: Discrete action space with appropriate number of actions,
                                   or None if Gymnasium is not available
                                   
        Notes:
            Action space size depends on configuration:
            - 4-direction + stay: 5 actions
            - 4-direction no stay: 4 actions  
            - 8-direction + stay: 9 actions
            - 8-direction no stay: 8 actions
            
        Examples:
            Get action space for RL training:
            >>> action_space = action_interface.get_action_space()
            >>> assert isinstance(action_space, gymnasium.spaces.Discrete)
            >>> print(f"Number of actions: {action_space.n}")
        """
        if not GYMNASIUM_AVAILABLE:
            return None
        
        return spaces.Discrete(self._num_actions)
    
    def get_action_mapping(self) -> Dict[int, str]:
        """
        Get mapping from action indices to direction names.
        
        Returns:
            Dict[int, str]: Mapping from action index to direction name
            
        Examples:
            Get human-readable action descriptions:
            >>> mapping = action_interface.get_action_mapping()
            >>> print(f"Action 0: {mapping[0]}")  # e.g., "STAY"
            >>> print(f"Action 1: {mapping[1]}")  # e.g., "NORTH"
        """
        return {idx: data['direction'] for idx, data in self._action_mapping.items()}
    
    def set_speed(self, new_speed: float) -> None:
        """
        Update movement speed and rebuild action mapping.
        
        Args:
            new_speed: New movement speed for all directions
            
        Raises:
            ValueError: If speed is negative or zero
            
        Notes:
            Updates all movement vectors with new speed while preserving
            direction relationships and diagonal normalization.
            
        Examples:
            Increase movement speed:
            >>> action_interface.set_speed(2.0)
            >>> # All movements now use speed 2.0
            
            Dynamic speed adjustment:
            >>> original_speed = action_interface.get_speed()
            >>> action_interface.set_speed(original_speed * 1.5)
        """
        if new_speed <= 0:
            raise ValueError(f"Speed must be positive, got {new_speed}")
        
        self._speed = float(new_speed)
        self._action_mapping = self._build_action_mapping()
    
    def get_available_actions(self) -> List[int]:
        """
        Get list of all valid action indices.
        
        Returns:
            List[int]: List of valid action indices
            
        Examples:
            Get all available actions:
            >>> actions = action_interface.get_available_actions()
            >>> print(f"Available actions: {actions}")  # e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8]
        """
        return list(self._action_mapping.keys())
    
    def get_speed(self) -> float:
        """
        Get current movement speed.
        
        Returns:
            float: Current movement speed for all directions
            
        Examples:
            Get current speed setting:
            >>> speed = action_interface.get_speed()
            >>> print(f"Current speed: {speed}")
        """
        return self._speed
    
    def get_num_actions(self) -> int:
        """
        Get total number of available actions.
        
        Returns:
            int: Number of discrete actions available
            
        Examples:
            >>> num_actions = action_interface.get_num_actions()
            >>> print(f"Action space size: {num_actions}")
        """
        return self._num_actions
    
    def get_direction_for_action(self, action: int) -> str:
        """
        Get direction name for a specific action index.
        
        Args:
            action: Action index to query
            
        Returns:
            str: Direction name for the action
            
        Raises:
            ValueError: If action index is invalid
            
        Examples:
            >>> direction = action_interface.get_direction_for_action(1)
            >>> print(f"Action 1 direction: {direction}")  # e.g., "NORTH"
        """
        if action not in self._action_mapping:
            raise ValueError(f"Invalid action {action}. Valid actions: 0-{self._num_actions-1}")
        return self._action_mapping[action]['direction']


def create_action_interface(config: Dict[str, Any]) -> ActionInterfaceProtocol:
    """
    Factory function for creating action interfaces from configuration.
    
    This factory enables configuration-driven action interface instantiation
    supporting both programmatic creation and Hydra configuration integration.
    
    Args:
        config: Configuration dictionary specifying action interface type and parameters.
               Required fields:
               - 'type': Action interface type ('Continuous2D' or 'CardinalDiscrete')
               - Additional fields depend on interface type
               
               For 'Continuous2D':
               - 'max_velocity': Maximum linear velocity (default: 2.0)
               - 'max_angular_velocity': Maximum angular velocity (default: 45.0)
               - 'min_velocity': Minimum linear velocity (default: -2.0)
               - 'min_angular_velocity': Minimum angular velocity (default: -45.0)
               
               For 'CardinalDiscrete':
               - 'speed': Movement speed (default: 1.0)
               - 'use_8_directions': Use 8 directions vs 4 (default: True)
               - 'include_stay_action': Include stay action (default: True)
               
    Returns:
        ActionInterfaceProtocol: Configured action interface instance
        
    Raises:
        ValueError: If configuration type is unknown or parameters are invalid
        KeyError: If required configuration fields are missing
        
    Examples:
        Continuous 2D action interface:
        >>> config = {
        ...     'type': 'Continuous2D',
        ...     'max_velocity': 2.5,
        ...     'max_angular_velocity': 60.0
        ... }
        >>> action_interface = create_action_interface(config)
        
        Cardinal discrete action interface:
        >>> config = {
        ...     'type': 'CardinalDiscrete',
        ...     'speed': 1.5,
        ...     'use_8_directions': False
        ... }
        >>> action_interface = create_action_interface(config)
        
        From Hydra configuration:
        >>> # In config.yaml:
        >>> # action:
        >>> #   type: Continuous2D
        >>> #   max_velocity: 2.0
        >>> #   max_angular_velocity: 45.0
        >>> action_interface = create_action_interface(cfg.action)
    """
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a dictionary, got {type(config)}")
    
    if 'type' not in config:
        raise KeyError("Configuration must specify 'type' field")
    
    action_type = config['type']
    
    if action_type == 'Continuous2D':
        # Extract parameters with defaults and validation
        try:
            params = {
                'max_velocity': float(config.get('max_velocity', 2.0)),
                'max_angular_velocity': float(config.get('max_angular_velocity', 45.0)),
                'min_velocity': float(config.get('min_velocity', -2.0)),
                'min_angular_velocity': float(config.get('min_angular_velocity', -45.0)),
            }
            return Continuous2DAction(**params)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid parameters for Continuous2D action interface: {e}")
    
    elif action_type == 'CardinalDiscrete':
        # Extract parameters with defaults and validation
        try:
            params = {
                'speed': float(config.get('speed', 1.0)),
                'use_8_directions': bool(config.get('use_8_directions', True)),
                'include_stay_action': bool(config.get('include_stay_action', True)),
            }
            return CardinalDiscreteAction(**params)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid parameters for CardinalDiscrete action interface: {e}")
    
    else:
        supported_types = ['Continuous2D', 'CardinalDiscrete']
        raise ValueError(
            f"Unknown action interface type: '{action_type}'. "
            f"Supported types: {supported_types}"
        )


# Utility functions for action interface management

def list_available_action_types() -> List[str]:
    """
    Get list of all available action interface types.
    
    Returns:
        List[str]: List of supported action interface type names
        
    Examples:
        >>> types = list_available_action_types()
        >>> print(f"Available action types: {types}")
        >>> # Output: ['Continuous2D', 'CardinalDiscrete']
    """
    return ['Continuous2D', 'CardinalDiscrete']


def validate_action_config(config: Dict[str, Any]) -> bool:
    """
    Validate action interface configuration without creating instance.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Examples:
        >>> config = {'type': 'Continuous2D', 'max_velocity': 2.0}
        >>> is_valid = validate_action_config(config)
        >>> assert is_valid
        
        >>> invalid_config = {'type': 'InvalidType'}
        >>> is_valid = validate_action_config(invalid_config)
        >>> assert not is_valid
    """
    try:
        create_action_interface(config)
        return True
    except (ValueError, KeyError, TypeError):
        return False


def get_action_space_info(action_interface: ActionInterfaceProtocol) -> Dict[str, Any]:
    """
    Get comprehensive information about an action interface's action space.
    
    Args:
        action_interface: Action interface to analyze
        
    Returns:
        Dict[str, Any]: Action space information including type, size, bounds, etc.
        
    Examples:
        >>> info = get_action_space_info(action_interface)
        >>> print(f"Action space type: {info['type']}")
        >>> print(f"Action space size: {info['size']}")
    """
    info = {}
    
    if isinstance(action_interface, Continuous2DAction):
        info['type'] = 'continuous'
        info['interface_class'] = 'Continuous2DAction'
        info['dimensions'] = 2
        info['max_velocity'] = action_interface.get_max_velocity()
        info['max_angular_velocity'] = action_interface.get_max_angular_velocity()
        
        action_space = action_interface.get_action_space()
        if action_space is not None:
            info['gymnasium_type'] = 'Box'
            info['shape'] = action_space.shape
            info['low'] = action_space.low.tolist()
            info['high'] = action_space.high.tolist()
        
    elif isinstance(action_interface, CardinalDiscreteAction):
        info['type'] = 'discrete'
        info['interface_class'] = 'CardinalDiscreteAction'
        info['num_actions'] = action_interface.get_num_actions()
        info['speed'] = action_interface.get_speed()
        info['action_mapping'] = action_interface.get_action_mapping()
        info['available_actions'] = action_interface.get_available_actions()
        
        action_space = action_interface.get_action_space()
        if action_space is not None:
            info['gymnasium_type'] = 'Discrete'
            info['size'] = action_space.n
    
    else:
        info['type'] = 'unknown'
        info['interface_class'] = type(action_interface).__name__
    
    return info


# Export all public components
__all__ = [
    'ActionInterfaceProtocol',
    'Continuous2DAction',
    'CardinalDiscreteAction',
    'create_action_interface',
    'list_available_action_types',
    'validate_action_config',
    'get_action_space_info',
]