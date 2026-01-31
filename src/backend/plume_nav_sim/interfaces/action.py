# Use forward references to avoid circular imports
from typing import TYPE_CHECKING, Any, Dict, Protocol, Union, runtime_checkable

import gymnasium as gym
import numpy as np

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]

if TYPE_CHECKING:
    from ..core.geometry import GridSize
    from ..core.state import AgentState

# Type alias for action inputs
ActionType = Union[int, NDArray[np.floating]]


@runtime_checkable
class ActionProcessor(Protocol):
    @property
    def action_space(self) -> gym.Space:
        ...

    def process_action(
        self,
        action: ActionType,
        current_state: "AgentState",
        grid_size: "GridSize",
    ) -> "AgentState":
        ...

    def validate_action(self, action: ActionType) -> bool:
        ...

    def get_metadata(self) -> Dict[str, Any]:
        ...
