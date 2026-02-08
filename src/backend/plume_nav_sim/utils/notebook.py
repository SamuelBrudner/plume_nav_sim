from __future__ import annotations

import itertools
import logging
import struct
import time
import zlib
from typing import Any, Callable, Optional

import numpy as np

from plume_nav_sim.runner import runner

logger = logging.getLogger(__name__)

_IPYTHON_IMPORT_MESSAGE = (
    "IPython is not available; notebook display helpers are disabled "
    "and will run in headless mode."
)
_WARNED_MISSING_IPYTHON = False


def _load_ipython_display() -> tuple[Optional[type], Optional[Callable[..., Any]]]:
    """Load IPython display primitives with a soft import guard."""

    global _WARNED_MISSING_IPYTHON
    try:
        from IPython.display import Image, display
    except ImportError:
        if not _WARNED_MISSING_IPYTHON:
            logger.info(_IPYTHON_IMPORT_MESSAGE)
            _WARNED_MISSING_IPYTHON = True
        return None, None
    return Image, display


def _as_uint8_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy ndarray")

    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(
            "frame must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)"
        )

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr)
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if min_val >= 0.0 and max_val <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return (
        struct.pack("!I", len(payload))
        + chunk_type
        + payload
        + struct.pack("!I", crc)
    )


def _frame_to_png_bytes(frame: np.ndarray) -> bytes:
    """Encode an RGB/RGBA frame into PNG bytes using only stdlib + numpy."""

    arr = _as_uint8_frame(frame)
    height, width, channels = arr.shape
    color_type = 6 if channels == 4 else 2

    ihdr = struct.pack("!IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    # Filter byte 0 per row (no PNG filter), then zlib-compress.
    raw_scanlines = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))
    idat = zlib.compress(raw_scanlines, level=6)

    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def display_frame(frame: np.ndarray) -> None:
    """Display a single frame as PNG in a notebook, if IPython is available."""

    image_cls, display_fn = _load_ipython_display()
    if image_cls is None or display_fn is None:
        return

    png_bytes = _frame_to_png_bytes(frame)
    display_fn(image_cls(data=png_bytes, format="png"))


def live_display(
    env: Any,
    policy: Any,
    *,
    seed: int | None = None,
    max_steps: int = 500,
    fps: float = 10,
) -> list[np.ndarray]:
    """Stream an episode and update a notebook display handle in place.

    Returns captured RGB frames. In headless mode (no IPython), frames are still
    captured and returned without attempting interactive output.
    """

    if max_steps <= 0:
        return []

    image_cls, display_fn = _load_ipython_display()
    handle: Any = None
    display_failed = False

    delay_seconds = 0.0
    if fps > 0:
        delay_seconds = 1.0 / float(fps)

    frames: list[np.ndarray] = []
    events = itertools.islice(
        runner.stream(env, policy, seed=seed, render=True),
        max_steps,
    )

    for ev in events:
        frame = ev.frame
        if not isinstance(frame, np.ndarray):
            continue

        frames.append(frame)

        if image_cls is None or display_fn is None or display_failed:
            continue

        png = _frame_to_png_bytes(frame)
        image = image_cls(data=png, format="png")

        if handle is None:
            try:
                candidate = display_fn(image, display_id=True)
                if candidate is not None and hasattr(candidate, "update"):
                    handle = candidate
                else:
                    display_failed = True
            except Exception:
                display_failed = True
        else:
            try:
                handle.update(image)
            except Exception:
                display_failed = True

        if delay_seconds > 0.0:
            time.sleep(delay_seconds)

    if frames:
        # Always show a final static frame so a persistent PNG remains in output.
        display_frame(frames[-1])

    return frames


__all__ = ["display_frame", "live_display"]
