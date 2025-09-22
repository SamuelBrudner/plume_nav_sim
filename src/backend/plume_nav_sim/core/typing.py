"""
Custom type aliases for the plume navigation simulation.
"""

raise ImportError(
    "plume_nav_sim.core.typing has been removed. Import from plume_nav_sim.core.types instead."
)

from typing import Any, Dict, List, Tuple, Union

import numpy as np
from typing_extensions import TypedDict

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

    def get_performance_summary(self) -> Dict[str, object]:
        """Return a lightweight summary of recorded timings.

        The summary includes per-operation timings, total time across all operations,
        and an average step time if a "step" timing series is present.
        """
        # Calculate totals
        total_ms = 0.0
        for series in self.timings.values():
            total_ms += sum(float(v) for v in series)

        avg_step_ms = 0.0
        step_series = self.timings.get("episode_step") or self.timings.get("step")
        if step_series:
            count = len(step_series)
            if count:
                avg_step_ms = float(sum(step_series)) / count

        return {
            "timings": {k: list(v) for k, v in self.timings.items()},
            "total_step_time_ms": total_ms,
            "average_step_time_ms": avg_step_ms,
        }
