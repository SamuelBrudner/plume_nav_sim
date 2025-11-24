from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from ..actions.oriented_grid import OrientedGridActions
from ..interfaces import Policy
from ._concentration_extractor import extract_concentration


@dataclass
class TemporalDerivativeDeterministicPolicy(Policy):
    """Deterministic temporal-gradient policy for oriented control.

    - Maintains 1-back concentration measured only after FORWARD steps
    - Surges FORWARD on non-decreasing concentration (dC >= threshold)
    - Otherwise casts by turning; enforces a FORWARD probe right after any TURN
    - Casting alternates deterministically RIGHT/LEFT/RIGHT/...

    Observation handling:
    - Expects a scalar concentration value; will raise a descriptive error if
      given multi-sensor arrays unless ``sensor_index`` is provided.
    - For dict/tuple observations, use ``concentration_key``/``modality_index`` to
      locate the concentration modality.
    """

    threshold: float = 1e-6
    cast_right_first: bool = True
    alternate_cast: bool = True  # if False, always turn RIGHT on negative derivative
    # Optional adapters for multi-modal observations
    concentration_key: Optional[str] = None  # key to pull concentration from dict obs
    modality_index: int = 0  # index when observation is a tuple/list of modalities
    sensor_index: Optional[int] = None  # index when observation is a 1D vector >1

    def __post_init__(self) -> None:
        self._actions = OrientedGridActions()
        self._last_c: Optional[float] = None
        self._last_action: Optional[int] = None
        self._cast_right_next: bool = self.cast_right_first

    @property
    def action_space(self) -> gym.Space:
        return self._actions.action_space

    def reset(self, *, seed: int | None = None) -> None:
        self._last_c = None
        self._last_action = None
        self._cast_right_next = self.cast_right_first

    def select_action(self, observation: Any, *, explore: bool = False) -> int:
        c = extract_concentration(
            observation,
            policy_name=self.__class__.__name__,
            concentration_key=self.concentration_key,
            modality_index=self.modality_index,
            sensor_index=self.sensor_index,
        )

        # Initialize reference on first call
        if self._last_c is None:
            self._last_c = c
            self._last_action = 0
            return 0

        # Compute dc identically every step
        dc = c - self._last_c

        # Decision gating: probe after any turn
        if self._last_action in (1, 2):
            action = 0
        else:
            if dc >= self.threshold:
                action = 0
            else:
                if self.alternate_cast:
                    action = 2 if self._cast_right_next else 1
                    self._cast_right_next = not self._cast_right_next
                else:
                    action = 2

        # Update reference and last action
        self._last_c = c
        self._last_action = action
        return action
