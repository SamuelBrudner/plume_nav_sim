from typing import Any, Dict

import gymnasium as gym
import numpy as np

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState


class OrientedGridActions:
    def __init__(self, step_size: int = 1):
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")

        self.step_size = step_size

        # Create action space
        self._action_space = gym.spaces.Discrete(3)

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def process_action(
        self,
        action: int,
        current_state: AgentState,
        grid_size: GridSize,
    ) -> AgentState:
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
        # Accept both Python int and numpy integer types
        if isinstance(action, (int, np.integer)):
            return 0 <= action < 3
        return False

    def get_metadata(self) -> Dict[str, Any]:
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
