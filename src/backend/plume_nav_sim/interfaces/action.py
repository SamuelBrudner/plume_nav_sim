"""
Action Processor Protocol Definition.

Contract: src/backend/contracts/action_processor_interface.md

This protocol defines the universal interface for action processing, enabling
diverse action spaces without environment modification.
"""

# Use forward references to avoid circular imports
from typing import TYPE_CHECKING, Any, Dict, Protocol, Union, runtime_checkable

import numpy as np

import gymnasium as gym

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    import numpy as np

    NDArray = np.ndarray  # type: ignore[assignment]

if TYPE_CHECKING:
    from ..core.geometry import GridSize
    from ..core.state import AgentState

# Type alias for action inputs
ActionType = Union[int, NDArray[np.floating]]


@runtime_checkable
class ActionProcessor(Protocol):
    """Protocol defining action processor interface.

    Contract: action_processor_interface.md

    Universal Properties:
        1. Boundary Safety: Result position always within grid
        2. Determinism: Same (action, state, grid) → same result
        3. Purity: No side effects, no mutations
        4. Updates: May update position and/or orientation

    Type Signature:
        ActionProcessor: (ActionType, AgentState, GridSize) → AgentState

    All implementations must satisfy:
        - Defines action_space (Gymnasium Space)
        - Processes action + current_state → new_state
        - Updates position and potentially orientation
        - Enforces boundary constraints
    """

    @property
    def action_space(self) -> gym.Space:
        """Gymnasium action space definition.

        Returns:
            Gymnasium Space defining valid actions (Discrete, Box, etc.)

        Postconditions:
            C1: Returns valid gym.Space instance
            C2: Space is immutable (same instance every call)
            C3: Space defines valid action representation
        """
        ...

    def process_action(
        self,
        action: ActionType,
        current_state: "AgentState",
        grid_size: "GridSize",
    ) -> "AgentState":
        """Process action to compute new agent state.

        Args:
            action: Action from action_space
            current_state: Agent's current state (position, orientation, etc.)
            grid_size: Grid bounds for boundary enforcement

        Returns:
            New AgentState after action (position within grid bounds)

        Preconditions:
            P1: action ∈ self.action_space
            P2: current_state is valid AgentState
            P3: grid_size.contains(current_state.position) = True

        Postconditions:
            C1: result is valid AgentState
            C2: grid_size.contains(result.position) = True (stays in bounds)
            C3: Result is deterministic (same inputs → same output)
            C4: result is new instance (not mutated current_state)

        Properties:
            1. Boundary Safety: Result position always within grid
            2. Determinism: Same (action, state, grid) → same result
            3. Purity: No side effects, no mutation of current_state
            4. Updates: May update position and/or orientation
        """
        ...

    def validate_action(self, action: ActionType) -> bool:
        """Check if action is valid for this processor.

        Args:
            action: Action to validate

        Returns:
            True if action is valid for this action_space
        """
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Return action processor metadata.

        Returns:
            Dictionary containing:
            - 'type': str - Action processor type
            - 'parameters': dict - Configuration
            - 'movement_model': Description of movement semantics
        """
        ...
