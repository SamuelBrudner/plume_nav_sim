"""
DiscreteGridActions: Standard 4-directional grid movement.

Contract: src/backend/contracts/action_processor_interface.md

This action processor implements absolute cardinal direction movement on a grid,
matching the existing default behavior of the environment. Actions move the agent
one step in the specified direction, with boundary clamping.
"""

from typing import Any, Dict

import gymnasium as gym
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.core.state import AgentState


class DiscreteGridActions:
    """Standard 4-directional discrete movement.

    Satisfies ActionProcessor protocol via duck typing.

    Action Space:
        Discrete(4): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

    Movement:
        - Absolute directions (not orientation-dependent)
        - step_size=1 by default (can be configured)
        - Boundary clamping (agent cannot leave grid)
        - Orientation unchanged

    Properties:
        - Boundary Safety: Always stays within grid bounds
        - Deterministic: Same inputs → same output
        - Pure: No side effects, returns new AgentState

    Example:
        >>> actions = DiscreteGridActions(step_size=1)
        >>> state = AgentState(position=Coordinates(10, 10), orientation=45.0)
        >>> grid = GridSize(32, 32)
        >>> new_state = actions.process_action(1, state, grid)  # Move RIGHT
        >>> new_state.position
        Coordinates(x=11, y=10)
    """

    def __init__(self, step_size: int = 1):
        """Initialize DiscreteGridActions.

        Args:
            step_size: Number of grid cells to move per action (default: 1)

        Raises:
            ValueError: If step_size is not positive
        """
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
        """Gymnasium action space definition.

        Returns:
            Discrete(4) space for 4-directional movement

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
        """Process discrete action to compute new agent state.

        Args:
            action: Action from action_space (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
            current_state: Agent's current state (position, orientation, etc.)
            grid_size: Grid bounds for boundary enforcement

        Returns:
            New AgentState after action (position within grid bounds)

        Contract: action_processor_interface.md - process_action()

        Preconditions:
            P1: action ∈ {0, 1, 2, 3}
            P2: current_state is valid AgentState
            P3: grid_size.contains(current_state.position) = True

        Postconditions:
            C1: result is valid AgentState
            C2: grid_size.contains(result.position) = True (stays in bounds)
            C3: Result is deterministic (same inputs → same output)
            C4: result is new instance (not mutated current_state)
        """
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

        # Create new AgentState with updated position
        # Orientation unchanged for absolute movement
        new_state = AgentState(
            position=Coordinates(clamped_x, clamped_y),
            orientation=current_state.orientation,
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
            True if action is valid (0, 1, 2, or 3)

        Contract: action_processor_interface.md - validate_action()
        """
        import numpy as np

        # Accept both Python int and numpy integer types
        if isinstance(action, (int, np.integer)):
            return 0 <= action < 4
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
