"""Video-backed concentration field that advances with simulation steps.

This lightweight field reads frames from a Zarr/xarray dataset with variable
`concentration: (t,y,x)` and exposes the current 2D frame via `field_array`.
It supports simple step policies (wrap, clamp) to map environment steps to
frame indices. Metadata is validated via media.schema.VideoPlumeAttrs.

Dependencies: xarray, zarr (optional). When missing, clear errors are raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

from ..core.geometry import GridSize
from ..media.schema import (
    DIMS_TYX,
    VARIABLE_CONCENTRATION,
    VideoPlumeAttrs,
    validate_xarray_like,
)
from ..utils.exceptions import ValidationError
from ..utils.logging import get_component_logger

StepPolicy = str  # "wrap" | "clamp"


@dataclass
class MovieConfig:
    path: str
    fps: Optional[float] = None
    pixel_to_grid: Optional[Tuple[float, float]] = None
    origin: Optional[Tuple[float, float]] = None
    extent: Optional[Tuple[float, float]] = None
    step_policy: StepPolicy = "wrap"


class MoviePlumeField:
    """Video-backed field exposing current 2D frame via `field_array`.

    Attributes:
        grid_size: GridSize derived from dataset dims (x, y)
        field_array: np.ndarray (float32) of shape (y, x) for current frame
        attrs: VideoPlumeAttrs describing calibration metadata
        num_frames: Total number of frames (T)
        frame_index: Current frame index
        step_policy: Mapping policy from step count to frame index
    """

    def __init__(self, cfg: MovieConfig) -> None:
        self._logger = get_component_logger("concentration_field")

        # Resolve and validate dataset path
        path = Path(cfg.path)
        if not path.exists():
            raise ValidationError(f"movie.path not found: {cfg.path}")
        if not path.is_dir():
            # Zarr stores are directories; support file-backed stores via xarray if possible
            # but fail fast for obviously wrong inputs
            pass

        # Lazy import optional deps with helpful errors
        try:
            import xarray as xr  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Movie plume requires optional dependency 'xarray'. "
                "Install extras: pip install xarray zarr numcodecs"
            ) from exc

        # Open dataset and validate schema
        try:
            ds = xr.open_zarr(str(path), consolidated=True)  # type: ignore[attr-defined]
        except Exception:
            # Fallback to non-consolidated stores
            ds = xr.open_zarr(str(path))  # type: ignore[attr-defined]

        attrs = validate_xarray_like(ds)

        # Apply overrides from cfg when provided
        attrs_overrides = {
            "fps": cfg.fps if cfg.fps is not None else attrs.fps,
            "pixel_to_grid": (
                cfg.pixel_to_grid if cfg.pixel_to_grid else attrs.pixel_to_grid
            ),
            "origin": cfg.origin if cfg.origin else attrs.origin,
            "extent": cfg.extent if cfg.extent else attrs.extent,
            "schema_version": attrs.schema_version,
            # Optional in some schema versions; include when present
            "timebase": getattr(attrs, "timebase", None),
            "source_dtype": getattr(attrs, "source_dtype"),
        }
        self.attrs: VideoPlumeAttrs = VideoPlumeAttrs(**attrs_overrides)  # type: ignore[arg-type]

        # Resolve dims and sizes
        var = ds[VARIABLE_CONCENTRATION]
        dims = tuple(getattr(var, "dims"))
        if dims != DIMS_TYX:
            raise ValidationError(f"Movie variable dims must be {DIMS_TYX}, got {dims}")
        size_t, size_y, size_x = (int(var.sizes[d]) for d in DIMS_TYX)  # type: ignore[index]
        self.num_frames = size_t
        self.grid_size = GridSize(width=size_x, height=size_y)

        # Backing handle for lazy frame selection
        self._data_array = var
        self._dataset_path = str(path)
        self.step_policy: StepPolicy = cfg.step_policy or "wrap"
        if self.step_policy not in ("wrap", "clamp"):
            raise ValidationError(
                f"movie.step_policy must be 'wrap' or 'clamp', got {self.step_policy!r}"
            )

        # Initialize to first frame
        self.frame_index = 0
        self.field_array = self._load_frame(0)

    # ------------------------------------------------------------------
    # Step/reset integration hooks
    def on_reset(self) -> None:
        self.frame_index = 0
        self.field_array = self._load_frame(0)

    def advance_to_step(self, step_count: int) -> None:
        """Advance current frame based on step count and step_policy."""
        if self.num_frames <= 0:
            return
        if self.step_policy == "wrap":
            idx = int(step_count) % self.num_frames
        else:  # clamp
            idx = min(int(step_count), self.num_frames - 1)
        if idx != self.frame_index:
            self.field_array = self._load_frame(idx)
            self.frame_index = idx

    # ------------------------------------------------------------------
    # Internal helpers
    def _load_frame(self, idx: int) -> np.ndarray:
        try:
            arr = self._data_array.isel(
                t=int(idx)
            ).values.astype(  # type: ignore[attr-defined]
                np.float32, copy=False
            )
        except Exception as exc:
            raise ValidationError(
                f"Failed to load frame {idx} from {self._dataset_path}"
            ) from exc

        if arr.ndim != 2:
            raise ValidationError(
                f"Loaded frame has shape {arr.shape}, expected 2D (y,x)"
            )
        h, w = arr.shape
        if h != self.grid_size.height or w != self.grid_size.width:
            raise ValidationError(
                f"Frame size {w}x{h} mismatches dataset grid {self.grid_size.width}x{self.grid_size.height}"
            )
        return arr


def resolve_movie_dataset_path(
    source_path: Union[str, Path],
    *,
    fps: Optional[float] = None,
    pixel_to_grid: Optional[Tuple[float, float]] = None,
    origin: Optional[Tuple[float, float]] = None,
    extent: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
) -> Path:
    """Resolve a user-facing movie path into a dataset root for MoviePlumeField.

    Accepts either:

    - A directory path (assumed to be a ready-to-use dataset root).
    - A file path (e.g. .avi, .mp4, or an image-sequence directory), which is
      ingested on demand into a sibling dataset directory using the
      ``video_ingest`` implementation.
    """

    path = Path(source_path)

    # If this already looks like a dataset root, return it directly.
    if path.is_dir():
        return path

    # Ingest media file or frames directory into a dataset root next to the input.
    output = path.with_suffix(".zarr")
    if output.exists():
        return output

    # Provide conservative defaults if not specified explicitly.
    if fps is None:
        fps = 60.0
    if pixel_to_grid is None:
        pixel_to_grid = (1.0, 1.0)
    if origin is None:
        origin = (0.0, 0.0)

    from plume_nav_sim.cli.video_ingest import IngestConfig, ingest_video

    cfg = IngestConfig(
        input=path,
        output=output,
        fps=float(fps),
        pixel_to_grid=pixel_to_grid,
        origin=origin,
        extent=extent,
        normalize=normalize,
    )
    return ingest_video(cfg)


__all__ = ["MoviePlumeField", "MovieConfig", "resolve_movie_dataset_path"]
