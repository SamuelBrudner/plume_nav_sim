from typing import Protocol, runtime_checkable

import numpy as np

try:  # pragma: no cover - numpy<1.20 compatibility
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover
    NDArray = np.ndarray  # type: ignore[assignment]

from ..core.geometry import Coordinates


@runtime_checkable
class ScalarField(Protocol):
    """Protocol for scalar fields (e.g., odor concentration)."""

    def sample(self, position: Coordinates) -> float: ...


@runtime_checkable
class VectorField(Protocol):
    """Protocol for vector fields (e.g., wind velocity)."""

    def sample(self, position: Coordinates) -> NDArray[np.floating]: ...
