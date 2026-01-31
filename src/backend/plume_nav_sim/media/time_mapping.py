from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final, Optional, Tuple


class FrameMappingPolicy(str, Enum):
    INDEX = "index"
    CLAMP = "clamp"
    WRAP = "wrap"

    @classmethod
    def from_str(cls, s: str) -> "FrameMappingPolicy":
        try:
            return cls(s.lower())
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Unknown FrameMappingPolicy: {s}") from exc


DEFAULT_FRAME_MAPPING_POLICY: Final[FrameMappingPolicy] = FrameMappingPolicy.WRAP


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def resolve_fps(
    *,
    fps: Optional[float] = None,
    timebase: Optional[Tuple[int, int]] = None,
    tol: float = 1e-6,
) -> float:
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

    if fps <= 0:
        raise ValueError("fps must be positive")

    fps_value = float(fps)

    if fps_from_tb is None:
        return fps_value

    if abs(fps_value - fps_from_tb) > tol:
        raise ValueError(
            f"Inconsistent fps ({fps}) and timebase-derived fps ({fps_from_tb})"
        )
    return fps_value


@dataclass(frozen=True)
class FrameSelectorConfig:
    """Configuration for deterministic step→frame mapping."""

    fps: float
    steps_per_second: Optional[float] = None
    offset_frames: float = 0.0
    rounding: str = "nearest"  # 'nearest' (half-up), 'floor', 'ceil'
    policy: FrameMappingPolicy = DEFAULT_FRAME_MAPPING_POLICY


def _validate_step_and_total_frames(step: int, total_frames: Optional[int]) -> None:
    if step < 0:
        raise ValueError("step must be non-negative")
    if total_frames is not None and total_frames <= 0:
        raise ValueError("total_frames must be positive when provided")


def _compute_floating_index(
    step: int,
    fps_val: float,
    steps_per_second: Optional[float],
    offset_frames: float,
) -> float:
    if steps_per_second is not None:
        if steps_per_second <= 0:
            raise ValueError("steps_per_second must be positive when provided")
        return (step * fps_val) / steps_per_second + offset_frames
    return step + offset_frames


def _round_index(f_float: float, rounding: str) -> int:
    if rounding == "nearest":
        return _round_half_up(f_float)
    if rounding == "floor":
        return int(math.floor(f_float))
    if rounding == "ceil":
        return int(math.ceil(f_float))
    raise ValueError("rounding must be one of: 'nearest', 'floor', 'ceil'")


def _ensure_non_negative(idx: int) -> int:
    return max(idx, 0)


def _apply_frame_policy(
    idx: int,
    total_frames: Optional[int],
    policy: FrameMappingPolicy,
) -> int:
    if total_frames is None:
        return idx

    if policy == FrameMappingPolicy.INDEX:
        if idx < 0 or idx >= total_frames:
            raise IndexError(
                f"Frame index {idx} out of range for total_frames={total_frames}"
            )
        return idx
    if policy == FrameMappingPolicy.CLAMP:
        return max(0, min(idx, total_frames - 1))
    if policy == FrameMappingPolicy.WRAP:
        return idx % total_frames
    raise ValueError(f"Unknown policy: {policy}")


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
    _validate_step_and_total_frames(step, total_frames)

    fps_val = resolve_fps(fps=fps, timebase=timebase)
    f_float = _compute_floating_index(
        step=step,
        fps_val=fps_val,
        steps_per_second=steps_per_second,
        offset_frames=offset_frames,
    )
    idx = _round_index(f_float, rounding)

    if total_frames is None:
        idx = _ensure_non_negative(idx)

    return _apply_frame_policy(idx, total_frames, policy)


__all__ = [
    "FrameMappingPolicy",
    "DEFAULT_FRAME_MAPPING_POLICY",
    "FrameSelectorConfig",
    "resolve_fps",
    "map_step_to_frame",
]
