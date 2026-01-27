# Use forward references to avoid circular imports
from typing import TYPE_CHECKING, Any, Dict, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..core.state import AgentState
    from ..plume.protocol import ConcentrationField
    from .action import ActionType


@runtime_checkable
class RewardFunction(Protocol):
    def compute_reward(
        self,
        prev_state: "AgentState",
        action: "ActionType",
        next_state: "AgentState",
        plume_field: "ConcentrationField",
    ) -> float:
        ...

    def get_metadata(self) -> Dict[str, Any]:
        ...
