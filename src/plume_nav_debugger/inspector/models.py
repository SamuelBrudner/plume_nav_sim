from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


def _softmax(x: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        x = x.ravel()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = (x - np.max(x)) / float(temperature)
    e = np.exp(z)
    s = e.sum()
    if s == 0 or not np.isfinite(s):
        # Fallback to uniform to avoid NaNs in UI
        return np.full_like(e, 1.0 / e.size)
    return e / s


@dataclass
class ObservationSummary:
    shape: tuple[int, ...]
    vmin: float
    vmean: float
    vmax: float


@dataclass
class ObservationPanelModel:
    summary: Optional[ObservationSummary] = None

    def update(self, observation: np.ndarray) -> None:
        if not isinstance(observation, np.ndarray):
            self.summary = None
            return
        try:
            arr = np.asarray(observation)
            self.summary = ObservationSummary(
                shape=tuple(int(s) for s in arr.shape),
                vmin=float(np.min(arr)) if arr.size else 0.0,
                vmean=float(np.mean(arr)) if arr.size else 0.0,
                vmax=float(np.max(arr)) if arr.size else 0.0,
            )
        except Exception:
            self.summary = None


@dataclass
class ActionPanelState:
    action_index: Optional[int] = None
    action_label: str = "-"
    distribution: Optional[List[float]] = None
    distribution_source: Optional[str] = None  # "probs" | "q_values" | "logits" | None


@dataclass
class ActionPanelModel:
    action_names: List[str] = field(default_factory=list)
    state: ActionPanelState = field(default_factory=ActionPanelState)

    def set_action_names(self, names: Sequence[str]) -> None:
        self.action_names = [str(n) for n in names]

    def update_event(self, action: int | np.integer | None) -> None:
        if isinstance(action, (int, np.integer)):
            idx = int(action)
            label = self._label_for(idx)
            self.state.action_index = idx
            self.state.action_label = label
        else:
            self.state.action_index = None
            self.state.action_label = "-"

    def probe_distribution(self, policy: object, observation: np.ndarray) -> None:
        # Provider-only UI: do not derive distributions from policy without provider
        self.state.distribution = None
        self.state.distribution_source = None

    # ------------------------------------------------------------------
    def _label_for(self, idx: int) -> str:
        if 0 <= idx < len(self.action_names):
            return f"{self.action_names[idx]} ({idx})"
        return str(idx)
