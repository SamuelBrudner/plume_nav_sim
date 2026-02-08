from __future__ import annotations

# Usage:
#   from plume_nav_sim.utils.video import save_video
#   save_video(frames, 'output.gif', fps=10)

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

_SUPPORTED_EXTENSIONS = {".gif", ".mp4", ".avi", ".webm"}
_DEFAULT_CODECS = {
    ".mp4": "libx264",
    ".avi": "mpeg4",
    ".webm": "libvpx-vp9",
}


def _import_imageio_v3():
    """Import imageio.v3 if available; otherwise return None."""
    try:
        import imageio.v3 as iio
    except ImportError:
        return None
    return iio


def _import_pillow_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for GIF fallback when imageio is unavailable. "
            "Install it with 'pip install Pillow'."
        ) from exc
    return Image


def _save_gif_with_pillow(
    frames: list[np.ndarray],
    output: Path,
    *,
    fps: int,
) -> None:
    Image = _import_pillow_image()
    pil_frames = [Image.fromarray(frame) for frame in frames]
    frame_duration_ms = max(1, int(round(1000 / fps)))
    pil_frames[0].save(
        str(output),
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )


def _raise_imageio_required() -> None:
    raise ImportError(
        "imageio is required for video export. Install media extras with "
        "'pip install plume-nav-sim[media]'."
    )


def _validate_frame(frame: np.ndarray) -> None:
    if frame.dtype != np.uint8:
        raise TypeError(
            f"Frame dtype must be uint8, got {frame.dtype!s}."
        )
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Frame must have shape (H, W, 3), got {frame.shape!r}."
        )


def frames_from_events(events: Iterable[object]) -> Iterator[np.ndarray]:
    """Yield valid RGB frames from StepEvent-like objects with a `.frame` attribute."""
    for event in events:
        frame = getattr(event, "frame", None)
        if frame is None:
            continue
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Event frame must be a numpy.ndarray when present."
            )
        _validate_frame(frame)
        yield frame


def _iter_frames(source: Iterable[object]) -> Iterator[np.ndarray]:
    for item in source:
        if isinstance(item, np.ndarray):
            _validate_frame(item)
            yield item
            continue

        if not hasattr(item, "frame"):
            raise TypeError(
                "save_video expects an iterable of np.ndarray frames or StepEvent-like "
                "objects exposing a '.frame' attribute."
            )

        frame = getattr(item, "frame")
        if frame is None:
            continue
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                "Event frame must be a numpy.ndarray when present."
            )
        _validate_frame(frame)
        yield frame


def save_video(
    source: Iterable[object],
    output_path: str | Path,
    *,
    fps: int = 30,
    codec: str | None = None,
) -> None:
    """Save RGB frames to GIF/MP4/AVI/WEBM using imageio."""
    if int(fps) <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}.")

    output = Path(output_path)
    ext = output.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported output extension. Expected one of: "
            ".gif, .mp4, .avi, .webm"
        )

    frames = list(_iter_frames(source))
    if not frames:
        raise ValueError("No frames available to write.")

    iio = _import_imageio_v3()
    if iio is not None:
        write_kwargs = {"fps": int(fps)}

        if ext == ".gif":
            write_kwargs["loop"] = 0
        else:
            selected_codec = codec if codec is not None else _DEFAULT_CODECS.get(ext)
            if selected_codec is not None:
                write_kwargs["codec"] = selected_codec

        iio.imwrite(str(output), frames, **write_kwargs)
        return

    if ext == ".gif":
        try:
            _save_gif_with_pillow(frames, output, fps=int(fps))
            return
        except ImportError as exc:
            raise ImportError(
                "Video export requires imageio or Pillow for GIF output. "
                "Install media extras with 'pip install plume-nav-sim[media]'."
            ) from exc

    _raise_imageio_required()


__all__ = ["frames_from_events", "save_video"]
