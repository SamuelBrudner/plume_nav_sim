from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _make_gaussian_frame(
    width: int = 128,
    height: int = 128,
    sigma: float = 40.0,
    center_x: float | None = None,
    center_y: float | None = None,
) -> np.ndarray:
    """Return a single circular Gaussian frame as 8-bit grayscale."""

    if center_x is None:
        center_x = (width - 1) / 2.0
    if center_y is None:
        center_y = (height - 1) / 2.0

    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]

    dx = x - float(center_x)
    dy = y - float(center_y)
    r2 = dx * dx + dy * dy

    frame = np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32)
    peak = float(frame.max())
    if peak > 0.0:
        frame /= peak

    return (frame * 255.0).astype(np.uint8)


def _build_frames(num_frames: int = 60) -> Sequence[np.ndarray]:
    base_uint8 = _make_gaussian_frame()
    base = base_uint8.astype(np.float32) / 255.0

    rng = np.random.default_rng()
    frames = []
    for _ in range(num_frames):
        noise = rng.normal(loc=0.0, scale=0.05, size=base.shape).astype(np.float32)
        frame_f = np.clip(base + noise, 0.0, 1.0)
        frames.append((frame_f * 255.0).astype(np.uint8))

    return frames


def _get_output_path() -> Path:
    here = Path(__file__).resolve()
    tests_dir = here.parents[1]
    out_dir = tests_dir / "data" / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "gaussian_plume_demo.avi"


def _write_avi(frames: Sequence[np.ndarray], output: Path, fps: int = 60) -> None:
    try:
        import imageio.v3 as iio  # type: ignore

        iio.imwrite(output, frames, fps=fps)
    except Exception:
        import imageio  # type: ignore

        imageio.mimwrite(output, frames, fps=fps)


def main() -> None:
    output = _get_output_path()
    frames = _build_frames(num_frames=60)
    _write_avi(frames, output, fps=60)
    print(f"Wrote {len(frames)} circular Gaussian frames to {output}")


if __name__ == "__main__":
    main()
