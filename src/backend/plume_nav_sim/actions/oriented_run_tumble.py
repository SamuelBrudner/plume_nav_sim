"""Run/Tumble oriented action processor.

Action space: Discrete(2)
  - 0 = RUN (move forward in current heading)
  - 1 = TUMBLE (reset orientation uniformly at random in [0, 360), then move forward)

Determinism: Environment calls set_rng() on reset with its episode RNG so that
TUMBLE is reproducible under seeding.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState


class OrientedRunTumbleActions:
    """Oriented 2-action processor implementing RUN/TUMBLE semantics.

    - RUN keeps the current heading and advances one step.
    - TUMBLE samples a uniform new heading in [0, 360) and then advances.
    """

    def __init__(self, step_size: int = 1):
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size = step_size
        self._action_space = gym.spaces.Discrete(2)
        self._rng: Optional[np.random.Generator] = None

    def set_rng(self, rng: np.random.Generator) -> None:
        self._rng = rng

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def process_action(
        self,
        action: int,
        current_state: AgentState,
        grid_size: GridSize,
    ) -> AgentState:
        if not isinstance(action, (int, np.integer)) or action not in (0, 1):
            raise ValueError(f"Invalid action: {action}, must be 0 (RUN) or 1 (TUMBLE)")

        # Default: keep current orientation
        new_orientation = current_state.orientation

        if action == 1:  # TUMBLE: random orientation reset
            rng = self._rng or np.random.default_rng()
            new_orientation = float(rng.uniform(0.0, 360.0))

        # Move forward one step in the (possibly new) heading
        angle_rad = np.deg2rad(new_orientation)
        dx = self.step_size * np.cos(angle_rad)
        dy = self.step_size * np.sin(angle_rad)

        new_x = int(round(current_state.position.x + dx))
        new_y = int(round(current_state.position.y + dy))

        clamped_x = max(0, min(new_x, grid_size.width - 1))
        clamped_y = max(0, min(new_y, grid_size.height - 1))

        new_position = Coordinates(clamped_x, clamped_y)

        return AgentState(
            position=new_position,
            orientation=new_orientation,
            step_count=current_state.step_count,
            total_reward=current_state.total_reward,
            goal_reached=current_state.goal_reached,
        )

    def validate_action(self, action: int) -> bool:
        return isinstance(action, (int, np.integer)) and int(action) in (0, 1)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "type": "oriented_run_tumble",
            "modality": "run_tumble",
            "parameters": {
                "step_size": self.step_size,
                "n_actions": 2,
                "action_names": ["RUN", "TUMBLE"],
            },
        }
