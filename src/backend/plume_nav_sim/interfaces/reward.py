"""
Reward Function Protocol Definition.

Contract: src/backend/contracts/reward_function_interface.md

This protocol defines the universal interface for all reward function implementations.
All reward functions must conform to this interface to be compatible with the
environment and config system.
"""

# Use forward references to avoid circular imports
from typing import TYPE_CHECKING, Any, Dict, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..core.state import AgentState
    from ..plume.concentration_field import ConcentrationField


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol defining reward function interface.

    Contract: reward_function_interface.md

    Universal Properties:
        1. Determinism: Same inputs always produce same reward
        2. Purity: No side effects, no mutations
        3. Finiteness: Result is always finite (not NaN, not inf)

    Type Signature:
        RewardFunction: (AgentState, Action, AgentState, ConcentrationField) → ℝ

    All implementations must satisfy:
        - Deterministic (same inputs → same output)
        - Pure (no side effects)
        - Returns finite float
        - Passes all property tests
    """

    def compute_reward(
        self,
        prev_state: "AgentState",
        action: int,
        next_state: "AgentState",
        plume_field: "ConcentrationField",
    ) -> float:
        """Compute reward for state transition.

        Args:
            prev_state: AgentState before action
            action: Action taken (integer from action space)
            next_state: AgentState after action
            plume_field: ConcentrationField for context

        Returns:
            Scalar reward value (implementation-specific range)

        Preconditions:
            P1: prev_state is valid AgentState
            P2: action is valid Action
            P3: next_state is valid AgentState
            P4: plume_field is valid ConcentrationField

        Postconditions:
            C1: result is finite float (not NaN, not inf)
            C2: result is deterministic (same inputs → same output)
            C3: isinstance(result, (float, int, np.floating, np.integer))

        Properties:
            1. Determinism: Same (s, a, s') always produces same reward
            2. Purity: No side effects or hidden state
            3. Finite: Result is always finite
        """
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """Return reward function metadata for logging/reproducibility.

        Returns:
            Dictionary containing:
            - 'type': str - Reward function type identifier
            - 'parameters': dict - Configuration parameters
            - Additional implementation-specific metadata

        Postconditions:
            C1: Returns dictionary with at least 'type' key
            C2: All values are JSON-serializable
        """
        ...
