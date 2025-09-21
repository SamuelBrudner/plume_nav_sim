"""
Custom type aliases for the plume navigation simulation.
"""

from typing import Any, Dict, List, Tuple, Union

import numpy as np

from .enums import Action
from .geometry import Coordinates

RGBArray = np.ndarray[Any, np.dtype[np.uint8]]
ActionType = Union[Action, int]
MovementVector = Tuple[int, int]
CoordinateType = Union[Coordinates, Tuple[int, int]]
ObservationType = np.ndarray

RewardType = float
InfoType = Dict[str, Any]


class PerformanceMetrics:
    """Minimal performance metrics collector used by lightweight components.

    Provides a record_timing(name, value_ms) API compatible with callers in the
    core modules without introducing heavy dependencies.
    """

    def __init__(self) -> None:
        self.timings: Dict[str, List[float]] = {}

    def record_timing(self, name: str, value_ms: float) -> None:
        self.timings.setdefault(name, []).append(float(value_ms))
