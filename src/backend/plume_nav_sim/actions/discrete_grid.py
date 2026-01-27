from typing import Any, Dict

import gymnasium as gym

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState


class DiscreteGridActions:
    def __init__(self, step_size: int = 1):
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")

        self.step_size = step_size

        # Define movement vectors for each action
        # UP=0, RIGHT=1, DOWN=2, LEFT=3
        self._movements = {
            0: (0, self.step_size),  # UP: +y
            1: (self.step_size, 0),  # RIGHT: +x
            2: (0, -self.step_size),  # DOWN: -y
            3: (-self.step_size, 0),  # LEFT: -x
        }

        # Create action space
        self._action_space = gym.spaces.Discrete(4)

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
            raise ValueError(f"Invalid action: {action}, must be in {{0, 1, 2, 3}}")

        # Get movement vector for action
        dx, dy = self._movements[action]

        # Compute new position
        new_x = current_state.position.x + dx
        new_y = current_state.position.y + dy

        # Enforce boundaries (clamp to grid)
        clamped_x = max(0, min(new_x, grid_size.width - 1))
        clamped_y = max(0, min(new_y, grid_size.height - 1))

        return AgentState(
            position=Coordinates(clamped_x, clamped_y),
            orientation=current_state.orientation,
            step_count=current_state.step_count,
            total_reward=current_state.total_reward,
            goal_reached=current_state.goal_reached,
        )

    def validate_action(self, action: int) -> bool:
        import numpy as np

        # Accept both Python int and numpy integer types
        return 0 <= action < 4 if isinstance(action, (int, np.integer)) else False

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": "discrete_grid",
            "modality": "absolute_cardinal",
            "parameters": {
                "step_size": self.step_size,
                "n_actions": 4,
                "action_names": ["UP", "RIGHT", "DOWN", "LEFT"],
            },
            "movement_model": "4-directional cardinal movement (absolute directions)",
            "orientation_dependent": False,
        }
