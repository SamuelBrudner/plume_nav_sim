# Media Time to Step Mapping Contract

Status: Alpha (living doc).

## Purpose

Define a deterministic mapping from simulation step index `k` to a source video frame index `f(k)`.

## Timebase

- `fps` (float): frames per second, must be > 0.
- `timebase` (num, den): seconds per frame as a rational.
  - `fps == den / num`.
- If both `fps` and `timebase` are provided, they must agree within tolerance or raise.

## Mapping

- If `steps_per_second` is provided:
  - `f_float = k * fps / steps_per_second + offset_frames`
- Else (index policy):
  - `f_float = k + offset_frames`

Rounding from `f_float` to integer frame index:

- `nearest` (half-up)
- `floor`
- `ceil`

## Policy (when total_frames is known)

- `index`: use the integer index as-is; raise IndexError if out of range
- `clamp`: clamp to `[0, total_frames - 1]`
- `wrap` (default): modulo wrap by `total_frames`

## Constants and Helpers

- `DEFAULT_FRAME_MAPPING_POLICY = "wrap"`
- `resolve_fps(fps=None, timebase=None, tol=1e-6) -> float`
- `map_step_to_frame(step, total_frames=None, fps=None, timebase=None, steps_per_second=None, offset_frames=0.0, rounding="nearest", policy="wrap") -> int`

## Notes

- This contract is pure and deterministic (no RNG).
- Intended for movie/video plume loaders and validators.
