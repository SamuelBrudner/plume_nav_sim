from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional

# Boundary policies
STEP_POLICY_INDEX: Literal["index"] = "index"  # error if out-of-range
STEP_POLICY_CLAMP: Literal["clamp"] = "clamp"  # clamp to [0, T-1]
STEP_POLICY_WRAP: Literal["wrap"] = "wrap"  # modulo wrap around T
DEFAULT_STEP_POLICY: Literal["clamp"] = STEP_POLICY_CLAMP

# Rounding policies
ROUND_FLOOR: Literal["floor"] = "floor"
ROUND_NEAREST: Literal["nearest"] = "nearest"
ROUND_CEIL: Literal["ceil"] = "ceil"
DEFAULT_ROUNDING: Literal["floor"] = ROUND_FLOOR


@dataclass(frozen=True)
class VideoTimebase:
    fps: Optional[float] = None
    timebase_numer: Optional[int] = None
    timebase_denom: Optional[int] = None
    tol: float = 1e-6

    def __post_init__(self) -> None:
        fps = self.fps
        tn = self.timebase_numer
        td = self.timebase_denom

        if fps is None and (tn is None or td is None):
            raise ValueError(
                "VideoTimebase requires either fps or timebase_numer/timebase_denom"
            )
        if fps is not None and fps <= 0:
            raise ValueError("fps must be positive")
        if tn is not None:
            if tn <= 0:
                raise ValueError("timebase_numer must be positive")
        if td is not None:
            if td <= 0:
                raise ValueError("timebase_denom must be positive")
        if fps is not None and tn is not None and td is not None:
            rational_fps = tn / td
            if abs(rational_fps - fps) > self.tol:
                raise ValueError(
                    f"Inconsistent fps ({fps}) vs timebase ({tn}/{td}={rational_fps})"
                )

    @property
    def fps_value(self) -> float:
        """Return the FPS as a float, derived from the provided representation."""
        if self.fps is not None:
            return float(self.fps)
        assert self.timebase_numer is not None and self.timebase_denom is not None
        return self.timebase_numer / self.timebase_denom


def map_step_to_frame(
    *,
    step: int,
    total_frames: int,
    timebase: VideoTimebase,
    step_hz: Optional[float] = None,
    offset_frames: int = 0,
    boundary: Literal["index", "clamp", "wrap"] = DEFAULT_STEP_POLICY,
    rounding: Literal["floor", "nearest", "ceil"] = DEFAULT_ROUNDING,
) -> int:
    _validate_step_and_total_frames(step=step, total_frames=total_frames)

    fps = timebase.fps_value
    hz = _resolve_step_rate(step_hz=step_hz, fps=fps)

    # Compute fractional frame index from time mapping
    # t_step = step / hz seconds; frames = t_step * fps
    fractional = _compute_fractional_index(step=step, step_hz=hz, fps=fps)

    idx = _compute_index_with_offset(
        fractional=fractional,
        offset_frames=offset_frames,
        rounding=rounding,
    )

    return _map_index_to_frame(idx=idx, total_frames=total_frames, boundary=boundary)


def _validate_step_and_total_frames(step: int, total_frames: int) -> None:
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")


def _resolve_step_rate(step_hz: Optional[float], fps: float) -> float:
    hz = step_hz if step_hz is not None else fps
    if hz <= 0:
        raise ValueError("step_hz must be positive")
    return hz


def _compute_fractional_index(step: int, step_hz: float, fps: float) -> float:
    return (step / step_hz) * fps


def _compute_index_with_offset(
    *, fractional: float, offset_frames: int, rounding: str
) -> int:
    base_index = _apply_rounding(fractional, rounding)
    return base_index + int(offset_frames)


def _apply_rounding(fractional: float, rounding: str) -> int:
    if rounding == ROUND_FLOOR:
        return math.floor(fractional)
    elif rounding == ROUND_NEAREST:
        return int(round(fractional))
    elif rounding == ROUND_CEIL:
        return math.ceil(fractional)
    else:
        raise ValueError(f"Unknown rounding policy: {rounding}")


def _map_index_to_frame(idx: int, total_frames: int, boundary: str) -> int:
    if 0 <= idx < total_frames:
        return idx
    return _apply_boundary_policy(idx, total_frames, boundary)


def _apply_boundary_policy(idx: int, total_frames: int, boundary: str) -> int:
    if boundary == STEP_POLICY_INDEX:
        raise IndexError(
            f"Frame index out of range (index={idx}, total_frames={total_frames})"
        )
    elif boundary == STEP_POLICY_CLAMP:
        if idx < 0:
            return 0
        return total_frames - 1
    elif boundary == STEP_POLICY_WRAP:
        # Ensure a positive modulo result
        return idx % total_frames
    else:
        raise ValueError(f"Unknown boundary policy: {boundary}")


__all__ = [
    # Policies
    "STEP_POLICY_INDEX",
    "STEP_POLICY_CLAMP",
    "STEP_POLICY_WRAP",
    "DEFAULT_STEP_POLICY",
    # Rounding
    "ROUND_FLOOR",
    "ROUND_NEAREST",
    "ROUND_CEIL",
    "DEFAULT_ROUNDING",
    # Types & functions
    "VideoTimebase",
    "map_step_to_frame",
]


def video_timebase_from_attrs(attrs: Mapping[str, Any]) -> VideoTimebase:
    def _maybe_float(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            raise ValueError(f"Invalid fps value: {x!r}")

    def _maybe_int(x: Any, name: str) -> Optional[int]:
        if x is None:
            return None
        try:
            return int(x)
        except Exception:
            raise ValueError(f"Invalid {name} value: {x!r}")

    fps = _maybe_float(attrs.get("fps"))
    tn = _maybe_int(attrs.get("timebase_numer"), "timebase_numer")
    td = _maybe_int(attrs.get("timebase_denom"), "timebase_denom")

    return VideoTimebase(fps=fps, timebase_numer=tn, timebase_denom=td)


__all__.append("video_timebase_from_attrs")
