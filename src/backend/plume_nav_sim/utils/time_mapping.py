"""
Deterministic mapping from simulation step index to video frame index.

This module defines a small, explicit contract for mapping a simulation step
``k`` to a source video frame index ``f(k)`` given a video timebase. It is
pure and deterministic by construction and provides clear policies for out-of-
range behavior.

Timebase
--------
- ``fps``: Frames per second as a float (e.g., 30.0)
- ``timebase``: A pair ``(numerator, denominator)`` representing seconds per
  frame as a rational number: seconds_per_frame = numerator / denominator,
  so ``fps = denominator / numerator``.

If both ``fps`` and ``timebase`` are provided, they must agree within a small
absolute tolerance or a ValueError is raised. If neither is provided, a
ValueError is raised.

Mapping
-------
Given step index ``k`` (0-based), an optional simulation rate
``steps_per_second`` and an optional fractional ``offset_frames``:

- If ``steps_per_second`` is provided: ``f_float = k * fps / steps_per_second + offset_frames``
- Otherwise: ``f_float = k + offset_frames`` (index policy)

The resulting float index is rounded by the selected rule and then adapted to
the available range using the selected policy:

- Rounding: ``'nearest'`` (half-up), ``'floor'``, ``'ceil'``
- Policy: ``'index'`` (error on out-of-range), ``'clamp'`` (saturate),
  ``'wrap'`` (modulo). Default policy is ``'wrap'``.

This helper is intentionally small and self-contained so it can be reused by
future loaders and validators without bringing in heavy dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final, Optional, Tuple


class FrameMappingPolicy(str, Enum):
    """Out-of-range handling policy for frame selection.

    - index: Use computed index as-is; raise on out-of-range when ``total_frames``
      is provided.
    - clamp: Clamp index to [0, total_frames - 1] when ``total_frames`` is provided.
    - wrap: Wrap index modulo ``total_frames`` when provided.
    """

    INDEX = "index"
    CLAMP = "clamp"
    WRAP = "wrap"

    @classmethod
    def from_str(cls, s: str) -> "FrameMappingPolicy":
        try:
            return cls(s.lower())
        except Exception as e:  # pragma: no cover - defensive path
            raise ValueError(f"Unknown FrameMappingPolicy: {s}") from e


DEFAULT_FRAME_MAPPING_POLICY: Final[FrameMappingPolicy] = FrameMappingPolicy.WRAP


def _round_half_up(x: float) -> int:
    """Round to nearest integer with ties (x.5) rounded away from zero.

    For non-negative values (the expected domain here), this is equivalent to
    floor(x + 0.5). Defined explicitly to avoid Python's bankers' rounding.
    """

    if x >= 0:
        return int(math.floor(x + 0.5))
    # For completeness, support negative values deterministically
    return int(math.ceil(x - 0.5))


def resolve_fps(
    *,
    fps: Optional[float] = None,
    timebase: Optional[Tuple[int, int]] = None,
    tol: float = 1e-6,
) -> float:
    """Resolve frames-per-second from ``fps`` or ``timebase``.

    Parameters
    ----------
    fps:
        Frames per second as a positive float.
    timebase:
        Pair (numerator, denominator) representing seconds per frame as a
        rational number; fps = denominator / numerator.
    tol:
        Absolute tolerance for consistency check when both provided.

    Returns
    -------
    float
        Frames per second.
    """

    fps_from_tb: Optional[float] = None
    if timebase is not None:
        num, den = timebase
        if num <= 0 or den <= 0:
            raise ValueError("timebase numerator and denominator must be positive")
        fps_from_tb = den / num

    if fps is None and fps_from_tb is None:
        raise ValueError("Either fps or timebase must be provided")

    if fps is None:
        assert fps_from_tb is not None  # for type checkers
        if fps_from_tb <= 0:
            raise ValueError("Derived fps from timebase must be positive")
        return fps_from_tb

    # fps is provided; ensure positive
    if fps <= 0:
        raise ValueError("fps must be positive")

    if fps_from_tb is None:
        return float(fps)

    # Both provided: require consistency
    if abs(float(fps) - fps_from_tb) > tol:
        raise ValueError(
            f"Inconsistent fps ({fps}) and timebase-derived fps ({fps_from_tb})"
        )
    return float(fps)


@dataclass(frozen=True)
class FrameSelectorConfig:
    """Configuration for deterministic step→frame mapping."""

    fps: float
    steps_per_second: Optional[float] = None
    offset_frames: float = 0.0
    rounding: str = "nearest"  # 'nearest' (half-up), 'floor', 'ceil'
    policy: FrameMappingPolicy = DEFAULT_FRAME_MAPPING_POLICY


def map_step_to_frame(
    step: int,
    *,
    total_frames: Optional[int] = None,
    fps: Optional[float] = None,
    timebase: Optional[Tuple[int, int]] = None,
    steps_per_second: Optional[float] = None,
    offset_frames: float = 0.0,
    rounding: str = "nearest",
    policy: FrameMappingPolicy = DEFAULT_FRAME_MAPPING_POLICY,
) -> int:
    """Map simulation step index to a source video frame index.

    This function is pure and deterministic. It does not perform any IO.

    Parameters
    ----------
    step:
        Simulation step index (0-based, non-negative).
    total_frames:
        When provided, enables policy handling ('index'|'clamp'|'wrap').
    fps:
        Frames per second as a positive float.
    timebase:
        (numerator, denominator) seconds-per-frame; fps = den / num.
    steps_per_second:
        If provided, use time mapping k * fps / steps_per_second; otherwise use
        index mapping (k).
    offset_frames:
        Optional fractional frame offset applied before rounding and policy.
    rounding:
        'nearest' (half-up), 'floor', or 'ceil'.
    policy:
        Out-of-range handling when total_frames is provided.

    Returns
    -------
    int
        Selected frame index (0-based).
    """

    if step < 0:
        raise ValueError("step must be non-negative")
    if total_frames is not None and total_frames <= 0:
        raise ValueError("total_frames must be positive when provided")

    fps_val = resolve_fps(fps=fps, timebase=timebase)

    if steps_per_second is not None:
        if steps_per_second <= 0:
            raise ValueError("steps_per_second must be positive when provided")
        f_float = (step * fps_val) / steps_per_second + offset_frames
    else:
        f_float = step + offset_frames

    if rounding == "nearest":
        idx = _round_half_up(f_float)
    elif rounding == "floor":
        idx = int(math.floor(f_float))
    elif rounding == "ceil":
        idx = int(math.ceil(f_float))
    else:
        raise ValueError("rounding must be one of: 'nearest', 'floor', 'ceil'")

    if idx < 0:
        # Maintain non-negativity post rounding
        idx = 0

    if total_frames is None:
        return idx

    if policy == FrameMappingPolicy.INDEX:
        if idx >= total_frames:
            raise IndexError(
                f"Frame index {idx} out of range for total_frames={total_frames}"
            )
        return idx
    elif policy == FrameMappingPolicy.CLAMP:
        return max(0, min(idx, total_frames - 1))
    elif policy == FrameMappingPolicy.WRAP:
        # Modulo wrap for any idx; total_frames > 0 validated above
        return idx % total_frames
    else:  # pragma: no cover - defensive path
        raise ValueError(f"Unknown policy: {policy}")


__all__ = [
    "FrameMappingPolicy",
    "DEFAULT_FRAME_MAPPING_POLICY",
    "FrameSelectorConfig",
    "resolve_fps",
    "map_step_to_frame",
]
