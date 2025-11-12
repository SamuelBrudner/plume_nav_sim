"""Deterministic mapping from simulation step index to video frame index.

This module defines a small, explicit contract for how simulation steps
map to video frames for movie-backed plume fields. It provides:

- A validated timebase that can be expressed as either a float FPS or a
  rational numerator/denominator (e.g., 30000/1001 for NTSC ~29.97 fps).
- Boundary policies for how to handle steps beyond the last frame
  ("index" error, "clamp", or "wrap").
- A helper to compute the frame index from a step with clear rounding.

Notes on timebase:
- If both `fps` and `(timebase_numer, timebase_denom)` are provided,
  they must be consistent within a small tolerance.
- The rational form is interpreted as `fps = timebase_numer / timebase_denom`.
  For example, NTSC 29.97 is 30000/1001.

Rounding policy applies to the fractional frame index before boundary handling.
"""

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
    """Validated video timebase.

    Provide either `fps` or the rational `(timebase_numer, timebase_denom)`.
    If both are provided they must agree within `tol`.
    """

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
    """Map a simulation step index to a video frame index deterministically.

    Args:
        step: Simulation step index (0-based).
        total_frames: Total number of frames `T` in the video (T > 0).
        timebase: Video timebase (fps) used for mapping.
        step_hz: Simulation step rate in steps per second. If None, defaults to
            `timebase.fps_value`, i.e., one video frame per simulation step.
        offset_frames: Constant frame offset applied after time mapping.
        boundary: Policy for handling indices outside [0, T-1]:
            - "index": raise IndexError
            - "clamp": clamp to endpoints 0..T-1
            - "wrap": modulo wrap around T
        rounding: Rounding policy applied to the fractional frame index before
            boundary handling: "floor", "nearest", or "ceil".

    Returns:
        An integer frame index in [0, T-1] (except "index" may raise).

    Raises:
        ValueError: If inputs are invalid.
        IndexError: If boundary policy is "index" and the computed index is OOB.
    """

    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if step < 0:
        raise ValueError("step must be non-negative")

    fps = timebase.fps_value
    hz = step_hz if step_hz is not None else fps
    if hz <= 0:
        raise ValueError("step_hz must be positive")

    # Compute fractional frame index from time mapping
    # t_step = step / hz seconds; frames = t_step * fps
    fractional = (step / hz) * fps

    if rounding == ROUND_FLOOR:
        base_index = math.floor(fractional)
    elif rounding == ROUND_NEAREST:
        base_index = int(round(fractional))
    elif rounding == ROUND_CEIL:
        base_index = math.ceil(fractional)
    else:
        raise ValueError(f"Unknown rounding policy: {rounding}")

    idx = base_index + int(offset_frames)

    # Apply boundary policy
    if 0 <= idx < total_frames:
        return idx

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
    """Construct a VideoTimebase from dataset attributes mapping.

    Expected keys (optional):
      - "fps": float-like (frames per second)
      - "timebase_numer": int-like
      - "timebase_denom": int-like

    At least one representation must be present, else ValueError is raised.
    """

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
