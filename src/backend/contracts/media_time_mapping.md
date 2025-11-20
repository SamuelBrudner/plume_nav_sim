# Media Time→Step Mapping Contract

Component: Media/Loader Utilities  
Version: 1.0.0  
Status: Canonical – used by loaders and validators

---

Purpose
- Define a deterministic, explicit mapping from simulation step index `k` to a source video frame index `f(k)`.
- Specify timebase representation, rounding, and out‑of‑range behavior.

Timebase
- fps (float): frames per second; must be > 0
- timebase (num, den): seconds per frame as a rational; fps = den / num
- If both provided, they must agree within tolerance; otherwise error.

Mapping
- If steps_per_second provided: f_float = k * fps / steps_per_second + offset_frames
- Else (index policy): f_float = k + offset_frames
- Rounding: nearest (half‑up), floor, or ceil → integer index

Policy (when total_frames known)
- index: use integer index as‑is; raise IndexError if out‑of‑range
- clamp: clamp to [0, total_frames − 1]
- wrap (default): modulo wrap by total_frames

Constants & Helper
- DEFAULT_FRAME_MAPPING_POLICY = 'wrap'
- resolve_fps(fps=None, timebase=None, tol=1e‑6) → float fps
- map_step_to_frame(step, total_frames=None, fps=None, timebase=None, steps_per_second=None, offset_frames=0.0, rounding='nearest', policy='wrap') → int

Notes
- This contract is pure and deterministic (no RNG). With fixed inputs, the selected frame is identical across runs.
- Intended for use by MoviePlumeField loader (issue 88) and associated tests (issue 90).
