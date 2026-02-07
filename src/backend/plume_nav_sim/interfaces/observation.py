from typing import Any, Dict, Protocol, Union, runtime_checkable

import gymnasium as gym
import numpy as np

from ..core.types import EnvState

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]

# Type alias for observation outputs
ObservationType = Union[NDArray[np.floating], Dict[str, Any], tuple[Any, ...]]


@runtime_checkable
class ObservationModel(Protocol):
    @property
    def observation_space(self) -> gym.Space:
        ...

    def get_observation(self, env_state: EnvState) -> ObservationType:
        ...

    def get_metadata(self) -> Dict[str, Any]:
        ...
