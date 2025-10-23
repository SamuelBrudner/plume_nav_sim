"""Minimal spaces module compatible with the subset of Gymnasium API used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


class Space:
    def contains(self, x: Any) -> bool:  # pragma: no cover - very basic shim
        return True

    def __contains__(self, x: Any) -> bool:  # Allow `x in space` checks
        return self.contains(x)


@dataclass
class Box(Space):
    low: Any
    high: Any
    shape: Tuple[int, ...]
    dtype: Any = np.float32

    def is_bounded(self) -> str:
        """Check if the space has finite bounds.

        Returns:
            "both" if bounded below and above, "below" if only below,
            "above" if only above, "neither" if unbounded
        """
        bounded_below = np.all(np.isfinite(self.low))
        bounded_above = np.all(np.isfinite(self.high))

        if bounded_below and bounded_above:
            return "both"
        elif bounded_below:
            return "below"
        elif bounded_above:
            return "above"
        else:
            return "neither"

    @property
    def bounded_below(self) -> np.ndarray:
        """Check which dimensions are bounded below."""
        return np.isfinite(self.low)

    @property
    def bounded_above(self) -> np.ndarray:
        """Check which dimensions are bounded above."""
        return np.isfinite(self.high)

    def sample(self) -> np.ndarray:
        """Sample a random observation from the space."""
        # Return midpoint between low and high to ensure valid sample
        low_arr = np.asarray(self.low)
        high_arr = np.asarray(self.high)

        # Broadcast to the correct shape if needed
        if low_arr.shape != self.shape:
            low_arr = np.broadcast_to(low_arr, self.shape)
        if high_arr.shape != self.shape:
            high_arr = np.broadcast_to(high_arr, self.shape)

        sample = (low_arr + high_arr) / 2.0
        return sample.astype(self.dtype)

    def contains(self, x: Any) -> bool:
        """Check if observation x is within box bounds."""
        try:
            x_array = np.asarray(x)
            # Check shape matches
            if x_array.shape != self.shape:
                return False
            # Check all values are within bounds
            return bool(np.all(x_array >= self.low) and np.all(x_array <= self.high))
        except Exception:
            return False


@dataclass
class Discrete(Space):
    n: int

    def __post_init__(self) -> None:
        # Local RNG to support deterministic sampling when seeding via env
        self._rng = np.random.default_rng()

    def seed(self, seed: Optional[int] = None) -> None:  # type: ignore[override]
        # Gymnasium seeds spaces via a seed() method, not constructor kwargs
        self._rng = np.random.default_rng(None if seed is None else int(seed))

    def sample(self) -> int:  # pragma: no cover - trivial
        return int(self._rng.integers(0, self.n))

    def contains(self, x: Any) -> bool:
        try:
            xi = int(x)
            return 0 <= xi < self.n
        except Exception:
            return False


class Dict(Space):  # noqa: N801 - match API name
    def __init__(self, spaces: Mapping[str, Space]):
        self.spaces: Dict[str, Space] = dict(spaces)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, Mapping):
            return False
        for k, sp in self.spaces.items():
            if k not in x or not sp.contains(x[k]):
                return False
        return True


__all__ = ["Space", "Box", "Discrete", "Dict"]
