from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from ..actions.oriented_grid import OrientedGridActions
from ..interfaces import Policy
from ._concentration_extractor import extract_concentration


@dataclass
class TemporalDerivativePolicy(Policy):
    eps: float = 0.05  # exploration rate
    eps_after_turn: float = (
        0.0  # exploration right after turn (default off to avoid spin)
    )
    eps_greedy_forward_bias: float = (
        0.0  # optional bias to select forward during exploration
    )
    eps_seed: Optional[int] = None
    threshold: float = 1e-6
    # When True, if dC <= threshold, sample uniformly from all actions (0,1,2)
    # rather than only turning. This models bacterial-like random tumbles that
    # can include a forward step even without improvement.
    uniform_random_on_non_increase: bool = False
    # Optional adapters for multi-modal observations
    concentration_key: Optional[str] = None  # key to pull concentration from dict obs
    modality_index: int = 0  # index when observation is a tuple/list of modalities
    sensor_index: Optional[int] = None  # index when observation is a 1D vector >1

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.eps_seed)
        self._actions = OrientedGridActions()
        self._last_c: Optional[float] = None
        self._last_action: Optional[int] = None
        self._prev_c: Optional[float] = None
        self._prev_action: Optional[int] = None

    # Policy protocol -----------------------------------------------------
    @property
    def action_space(self) -> gym.Space:
        return self._actions.action_space

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._last_c = None
        self._last_action = None
        self._prev_c = None
        self._prev_action = None

    def select_action(self, observation: Any, *, explore: bool = True) -> int:
        c = extract_concentration(
            observation,
            policy_name=self.__class__.__name__,
            concentration_key=self.concentration_key,
            modality_index=self.modality_index,
            sensor_index=self.sensor_index,
        )

        # Initialize reference on first call
        if self._last_c is None:
            self._prev_c = None
            self._prev_action = None
            self._last_c = c
            self._last_action = 0
            return 0

        self._prev_c = self._last_c
        self._prev_action = self._last_action

        # Compute dc identically every step
        dc = c - self._last_c

        # After a TURN, force a FORWARD probe (optionally explore)
        if self._last_action in (1, 2):
            if explore and self._rng.random() < self.eps_after_turn:
                action = self._sample_explore(after_turn=True)
            else:
                action = 0
        else:
            if dc >= self.threshold:
                action = 0
            else:
                if self.uniform_random_on_non_increase:
                    action = int(self._rng.integers(0, 3))
                else:
                    action = 1 if self._rng.random() < 0.5 else 2
                if explore and self._rng.random() < self.eps:
                    action = self._sample_explore()

        # Update reference and last action
        self._last_c = c
        self._last_action = action
        return action

    # --------------------------------------------------------------------
    def _sample_explore(self, *, after_turn: bool = False) -> int:
        if after_turn:
            # Mostly prefer forward; allow a tiny chance to turn to keep stochasticity
            if self._rng.random() < max(0.8, 1.0 - self.eps):
                return 0
            return 1 if self._rng.random() < 0.5 else 2

        # General epsilon sampling with optional forward bias
        r = self._rng.random()
        if r < self.eps_greedy_forward_bias:
            return 0
        # Uniform among three actions
        return int(self._rng.integers(0, 3))
