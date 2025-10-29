"""
Video saving utilities for frames and runner streams.

These helpers provide a thin, optional wrapper around imageio.v3 so callers can
persist RGB frames (H, W, 3) to GIF/MP4 without wiring a writer manually.

Optional dependency:
    pip install imageio  # and ffmpeg for MP4

The functions raise ImportError with a clear message if imageio is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _import_imageio_v3():
    try:
        import imageio.v3 as iio  # type: ignore
    except (
        Exception
    ) as e:  # pragma: no cover - exercised via tests with sys.modules shims
        raise ImportError(
            "imageio is required for video export. Install optional media extras or 'pip install imageio'."
        ) from e
    return iio


def _coerce_path(path: str | Path) -> str:
    return str(path) if isinstance(path, Path) else path


def save_video_frames(
    frames: Iterable[np.ndarray],
    path: str | Path,
    *,
    fps: int = 10,
    codec: Optional[str] = None,
    loop: Optional[int] = None,
    **kwargs,
) -> None:
    """Save a sequence of RGB frames to a video file.

    Parameters
    ----------
    frames:
        Iterable of (H, W, 3) uint8 RGB arrays.
    path:
        Output filepath; extension determines writer (e.g., .gif, .mp4).
    fps:
        Frames-per-second for playback.
    codec:
        Optional codec hint for formats like MP4 (e.g., 'libx264').
    loop:
        GIF loop parameter; 0 for infinite (default when writing GIFs).
    kwargs:
        Additional keyword args forwarded to imageio.v3.imwrite.
    """
    iio = _import_imageio_v3()
    out = _coerce_path(path)
    ext = Path(out).suffix.lower()

    # imageio can accept an iterator, but materialize to list to avoid surprises
    # with one-shot generators in tests and to support multiple passes when needed.
    frames_list = list(frames)
    if not frames_list:
        raise ValueError("No frames provided for video export")

    # Basic sanity: ensure arrays are numpy ndarrays
    for f in frames_list:
        if not isinstance(f, np.ndarray):
            raise TypeError("Frames must be numpy.ndarrays")

    save_kwargs = dict(kwargs)
    save_kwargs.setdefault("fps", int(fps))

    if ext == ".gif":
        # Default to infinite loop for GIFs unless caller overrides
        if loop is None:
            save_kwargs.setdefault("loop", 0)
        else:
            save_kwargs["loop"] = int(loop)
    elif ext in {".mp4", ".m4v", ".mov"}:
        if codec is not None:
            save_kwargs.setdefault("codec", codec)

    iio.imwrite(out, frames_list, **save_kwargs)


def save_video_events(
    events: Iterable[object],
    path: str | Path,
    *,
    fps: int = 10,
    codec: Optional[str] = None,
    loop: Optional[int] = None,
    **kwargs,
) -> None:
    """Save frames from a runner StepEvent stream to a video file.

    The iterable is expected to yield objects with an attribute 'frame' holding
    an (H, W, 3) uint8 RGB array or None. Events with frame=None are skipped.
    """

    def _iter_frames():
        for ev in events:
            f = getattr(ev, "frame", None)
            if isinstance(f, np.ndarray):
                yield f

    save_video_frames(_iter_frames(), path, fps=fps, codec=codec, loop=loop, **kwargs)
