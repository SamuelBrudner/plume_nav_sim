from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np

from ..actions.oriented_grid import OrientedGridActions
from ..interfaces import Policy
from ._concentration_extractor import extract_concentration

CastMode = Literal["random_turn", "alternating_turn", "zigzag"]


@dataclass
class SurgeAndCastPolicy(Policy):
    """Explicit surge-and-cast controller for oriented actions.

    Behavior:
    - Surge FORWARD while the temporal derivative stays above threshold.
    - Cast (turn) when the derivative drops below threshold using a configurable
      turn pattern.
    - After each cast, persist in FORWARD for ``persistence`` steps to probe the
      new heading.

    Cast modes:
    - random_turn: sample LEFT/RIGHT uniformly (deterministic LEFT when
      explore=False).
    - alternating_turn: alternate LEFT/RIGHT on each cast, reset on reset().
    - zigzag: alternate LEFT/RIGHT across casts but do not reset between
      positive gradients (only on reset()).

    Observation handling mirrors TemporalDerivativePolicy for compatibility.
    """

    threshold: float = 1e-6
    cast_mode: CastMode = "random_turn"
    persistence: int = 1
    eps_seed: Optional[int] = None
    concentration_key: Optional[str] = None
    modality_index: int = 0
    sensor_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.persistence, int):
            raise ValueError("persistence must be an integer")
        if self.persistence < 0:
            raise ValueError("persistence must be non-negative")
        if self.cast_mode not in ("random_turn", "alternating_turn", "zigzag"):
            raise ValueError(
                f"cast_mode must be one of "
                f"'random_turn', 'alternating_turn', or 'zigzag'; got {self.cast_mode}"
            )

        self._actions = OrientedGridActions()
        self._rng = np.random.default_rng(self.eps_seed)
        self._last_c: Optional[float] = None
        self._persistence_remaining = 0
        self._next_alternating_left = True
        self._zigzag_left = True

    # Policy protocol -----------------------------------------------------
    @property
    def action_space(self) -> gym.Space:
        return self._actions.action_space

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_c = None
        self._persistence_remaining = 0
        self._next_alternating_left = True
        self._zigzag_left = True

    def select_action(self, observation: Any, *, explore: bool = True) -> int:
        c = extract_concentration(
            observation,
            policy_name=self.__class__.__name__,
            concentration_key=self.concentration_key,
            modality_index=self.modality_index,
            sensor_index=self.sensor_index,
        )

        if self._last_c is None:
            self._last_c = c
            return 0

        if self._persistence_remaining > 0:
            action = 0
            self._persistence_remaining -= 1
            self._last_c = c
            return action

        dc = c - self._last_c

        if dc >= self.threshold:
            action = 0
        else:
            action = self._select_cast_action(explore=explore)
            if self.persistence:
                self._persistence_remaining = self.persistence

        self._last_c = c
        return action

    # ------------------------------------------------------------------
    def _select_cast_action(self, *, explore: bool) -> int:
        if self.cast_mode == "random_turn":
            if not explore:
                return 1
            return int(self._rng.integers(1, 3))

        if self.cast_mode == "alternating_turn":
            action = 1 if self._next_alternating_left else 2
            self._next_alternating_left = not self._next_alternating_left
            # reset zigzag memory after a positive run so alternating always
            # restarts from left when reset() is called.
            self._zigzag_left = True
            return action

        # zigzag: persistent alternation across casts until reset()
        action = 1 if self._zigzag_left else 2
        self._zigzag_left = not self._zigzag_left
        return action
