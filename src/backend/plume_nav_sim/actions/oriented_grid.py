"""
OrientedGridActions: 3-action surge/turn control with orientation tracking.

Contract: src/backend/contracts/action_processor_interface.md

This action processor implements orientation-relative movement where the agent
has a heading and actions are relative to that heading. Forward moves in the
current heading direction, turns rotate the heading without moving.
"""

from typing import Any, Dict

import gymnasium as gym
import numpy as np

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState


class OrientedGridActions:
    """3-action surge/turn control with orientation tracking.

    Satisfies ActionProcessor protocol via duck typing.

    Action Space:
        Discrete(3): 0=FORWARD, 1=TURN_LEFT, 2=TURN_RIGHT

    Movement:
        - FORWARD: Move in current heading direction
        - TURN_LEFT: Rotate heading +90° (counterclockwise)
        - TURN_RIGHT: Rotate heading -90° (clockwise)
        - Boundary clamping (agent cannot leave grid)
        - Orientation updated by turn actions

    Orientation Convention:
        - 0° = East (+x direction)
        - 90° = North (+y direction)
        - 180° = West (-x direction)
        - 270° = South (-y direction)

    Properties:
        - Boundary Safety: Always stays within grid bounds
        - Deterministic: Same inputs → same output
        - Pure: No side effects, returns new AgentState
        - Orientation-Aware: Movement depends on heading

    Example:
        >>> actions = OrientedGridActions(step_size=1)
        >>> state = AgentState(position=Coordinates(10, 10), orientation=0.0)
        >>> grid = GridSize(32, 32)
        >>> new_state = actions.process_action(0, state, grid)  # FORWARD (East)
        >>> new_state.position
        Coordinates(x=11, y=10)
        >>> new_state = actions.process_action(1, state, grid)  # TURN_LEFT
        >>> new_state.orientation
        90.0
    """

    def __init__(self, step_size: int = 1):
        """Initialize OrientedGridActions.

        Args:
            step_size: Number of grid cells to move per forward action (default: 1)

        Raises:
            ValueError: If step_size is not positive
        """
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")

        self.step_size = step_size

        # Create action space
        self._action_space = gym.spaces.Discrete(3)

    @property
    def action_space(self) -> gym.Space:
        """Gymnasium action space definition.

        Returns:
            Discrete(3) space for forward/turn_left/turn_right

        Contract: action_processor_interface.md - Postcondition C2
        Immutable: Returns same instance every call
        """
        return self._action_space

    def process_action(
        self,
        action: int,
        current_state: AgentState,
        grid_size: GridSize,
    ) -> AgentState:
        """Process oriented action to compute new agent state.

        Args:
            action: Action from action_space (0=FORWARD, 1=TURN_LEFT, 2=TURN_RIGHT)
            current_state: Agent's current state (position, orientation, etc.)
            grid_size: Grid bounds for boundary enforcement

        Returns:
            New AgentState after action (position within grid bounds)

        Contract: action_processor_interface.md - process_action()

        Movement Calculation:
            FORWARD:
                - angle_rad = radians(orientation)
                - dx = step_size * cos(angle_rad)
                - dy = step_size * sin(angle_rad)
                - new_position = current_position + (dx, dy) [clamped to grid]

            TURN_LEFT:
                - new_orientation = (current_orientation + 90) % 360

            TURN_RIGHT:
                - new_orientation = (current_orientation - 90) % 360

        Postconditions:
            C1: result is valid AgentState
            C2: grid_size.contains(result.position) = True (stays in bounds)
            C3: Result is deterministic (same inputs → same output)
            C4: result is new instance (not mutated current_state)
        """
        if not self.validate_action(action):
            raise ValueError(f"Invalid action: {action}, must be in {{0, 1, 2}}")

        # Initialize with current state
        new_position = current_state.position
        new_orientation = current_state.orientation

        if action == 0:  # FORWARD
            # Move in current heading direction
            # Convert orientation to radians for trig functions
            angle_rad = np.deg2rad(current_state.orientation)

            # Calculate movement components
            # Note: In standard math convention:
            #   - 0° = East (+x), 90° = North (+y)
            #   - cos gives x-component, sin gives y-component
            dx = self.step_size * np.cos(angle_rad)
            dy = self.step_size * np.sin(angle_rad)

            # Compute new position
            new_x = int(round(current_state.position.x + dx))
            new_y = int(round(current_state.position.y + dy))

            # Enforce boundaries (clamp to grid)
            clamped_x = max(0, min(new_x, grid_size.width - 1))
            clamped_y = max(0, min(new_y, grid_size.height - 1))

            new_position = Coordinates(clamped_x, clamped_y)

        elif action == 1:  # TURN_LEFT (+90°)
            new_orientation = (current_state.orientation + 90.0) % 360.0

        elif action == 2:  # TURN_RIGHT (-90°)
            new_orientation = (current_state.orientation - 90.0) % 360.0

        # Create new AgentState with updated position and/or orientation
        new_state = AgentState(
            position=new_position,
            orientation=new_orientation,
            step_count=current_state.step_count,
            total_reward=current_state.total_reward,
            goal_reached=current_state.goal_reached,
        )

        return new_state

    def validate_action(self, action: int) -> bool:
        """Check if action is valid for this processor.

        Args:
            action: Action to validate

        Returns:
            True if action is valid (0, 1, or 2)

        Contract: action_processor_interface.md - validate_action()
        """
        # Accept both Python int and numpy integer types
        if isinstance(action, (int, np.integer)):
            return 0 <= action < 3
        return False

    def get_metadata(self) -> Dict[str, Any]:
        """Return action processor metadata.

        Returns:
            Dictionary containing:
            - 'type': str - Action processor type
            - 'parameters': dict - Configuration
            - 'movement_model': Description of movement semantics

        Contract: action_processor_interface.md - get_metadata()
        """
        return {
            "type": "oriented_grid",
            "modality": "forward_turn",
            "parameters": {
                "step_size": self.step_size,
                "n_actions": 3,
                "action_names": ["FORWARD", "TURN_LEFT", "TURN_RIGHT"],
                "turn_angle_degrees": 90.0,
            },
            "movement_model": "3-action surge/turn control (orientation-relative)",
            "orientation_dependent": True,
        }
