"""Video-backed plume with lazy frame loading from Zarr/HDF5 datasets."""

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
from ..media.sidecar import MovieMetadataSidecar, load_movie_sidecar
from ..utils.exceptions import ValidationError

StepPolicy = str  # "wrap" | "clamp"


@dataclass
class VideoConfig:
    """Configuration for loading a video-backed plume dataset."""

    path: str
    fps: Optional[float] = None
    pixel_to_grid: Optional[Tuple[float, float]] = None
    origin: Optional[Tuple[float, float]] = None
    extent: Optional[Tuple[float, float]] = None
    step_policy: StepPolicy = "wrap"
    data_array: Any | None = None


def _open_movie_dataset(path: Path) -> Any:
    try:
        import xarray as xr  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Video plume requires optional dependency 'xarray'. "
            "Install extras: pip install xarray zarr numcodecs"
        ) from exc

    try:
        return xr.open_zarr(str(path), consolidated=True)  # type: ignore[attr-defined]
    except Exception:
        return xr.open_zarr(str(path))  # type: ignore[attr-defined]


def _resolve_movie_data(cfg: VideoConfig) -> tuple[Any, Any, str]:
    data_array: Any = getattr(cfg, "data_array", None)
    path = Path(cfg.path) if cfg.path else None

    if data_array is None:
        if path is None:
            raise ValidationError("video.path must be provided")
        if not path.exists():
            raise ValidationError(f"video.path not found: {cfg.path}")
        ds = _open_movie_dataset(path)
        var = ds[VARIABLE_CONCENTRATION]
        dataset_path = str(path)
        return var, ds, dataset_path

    try:
        attrs_source = data_array.to_dataset(name=VARIABLE_CONCENTRATION)
    except Exception:
        attrs_source = {VARIABLE_CONCENTRATION: data_array}
    dataset_path = str(path) if path is not None else "<in-memory>"
    return data_array, attrs_source, dataset_path


def _apply_attr_overrides(cfg: VideoConfig, attrs: VideoPlumeAttrs) -> VideoPlumeAttrs:
    overrides = {
        "fps": cfg.fps if cfg.fps is not None else attrs.fps,
        "pixel_to_grid": cfg.pixel_to_grid or attrs.pixel_to_grid,
        "origin": cfg.origin or attrs.origin,
        "extent": cfg.extent or attrs.extent,
        "schema_version": attrs.schema_version,
        "timebase": getattr(attrs, "timebase", None),
        "source_dtype": getattr(attrs, "source_dtype"),
    }
    return VideoPlumeAttrs(**overrides)  # type: ignore[arg-type]


def _resolve_movie_sizes(var: Any) -> tuple[int, int, int]:
    dims = tuple(getattr(var, "dims"))
    if dims != DIMS_TYX:
        raise ValidationError(f"Video dims must be {DIMS_TYX}, got {dims}")
    size_t, size_y, size_x = (int(var.sizes[d]) for d in DIMS_TYX)  # type: ignore[index]
    return size_t, size_y, size_x


class VideoPlume:
    """Video-backed plume with a current 2D frame in `field_array`."""

    def __init__(self, cfg: VideoConfig) -> None:
        var, attrs_source, dataset_path = _resolve_movie_data(cfg)
        attrs = validate_xarray_like(attrs_source)
        self.attrs = _apply_attr_overrides(cfg, attrs)
        size_t, size_y, size_x = _resolve_movie_sizes(var)
        self.num_frames = size_t
        self.grid_size = GridSize(width=size_x, height=size_y)

        self._data_array = var
        self._dataset_path = dataset_path
        self.step_policy: StepPolicy = cfg.step_policy or "wrap"
        if self.step_policy not in ("wrap", "clamp"):
            raise ValidationError(
                f"video.step_policy must be 'wrap' or 'clamp', got {self.step_policy!r}"
            )

        self.frame_index = 0
        self.field_array = self._load_frame(0)
        self._seed: Optional[int] = None

    def reset(self, seed: int | None = None) -> None:
        self._seed = None if seed is None else int(seed)
        self.on_reset()

    def on_reset(self) -> None:
        self.frame_index = 0
        self.field_array = self._load_frame(0)

    def advance_to_step(self, step_count: int) -> None:
        if self.num_frames <= 0:
            return
        if self.step_policy == "wrap":
            idx = int(step_count) % self.num_frames
        else:
            idx = min(int(step_count), self.num_frames - 1)
        if idx != self.frame_index:
            self.field_array = self._load_frame(idx)
            self.frame_index = idx

    def sample(self, x: float, y: float, t: float | None = None) -> float:
        if t is not None:
            self.advance_to_step(int(round(t)))
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < self.grid_size.width and 0 <= iy < self.grid_size.height:
            return float(self.field_array[iy, ix])
        return 0.0

    def _load_frame(self, idx: int) -> np.ndarray:
        try:
            arr = self._data_array.isel(t=int(idx)).values.astype(  # type: ignore[attr-defined]
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
    movie_h5_dataset: Optional[str] = None,
) -> Path:
    """Resolve a user-facing movie path into a dataset root for VideoPlume."""

    path = Path(source_path)

    if path.is_dir():
        return path

    output = path.with_suffix(".zarr")
    if output.exists():
        return output

    sidecar: MovieMetadataSidecar = load_movie_sidecar(path)
    sidecar_fps = float(sidecar.fps)
    sidecar_pixel_to_grid = _sidecar_pixel_to_grid(sidecar)
    sidecar_origin: Tuple[float, float] = (0.0, 0.0)

    _validate_movie_overrides(
        sidecar_fps,
        sidecar_pixel_to_grid,
        sidecar_origin,
        fps,
        pixel_to_grid,
        origin,
        extent,
    )

    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return _ingest_hdf5_from_sidecar(
            path,
            output,
            sidecar,
            sidecar_fps,
            sidecar_pixel_to_grid,
            sidecar_origin,
            normalize,
            movie_h5_dataset,
        )

    return _ingest_video_from_sidecar(
        path,
        output,
        sidecar_fps,
        sidecar_pixel_to_grid,
        sidecar_origin,
        normalize,
    )


def _sidecar_pixel_to_grid(sidecar: MovieMetadataSidecar) -> Tuple[float, float]:
    if sidecar.spatial_unit == "pixel":
        return (1.0, 1.0)
    if sidecar.pixels_per_unit is None:
        raise ValidationError(
            "pixels_per_unit must be provided in movie sidecar when spatial_unit is not 'pixel'"
        )
    py, px = sidecar.pixels_per_unit
    if py <= 0.0 or px <= 0.0:
        raise ValidationError("pixels_per_unit entries must be positive")
    return (1.0 / py, 1.0 / px)


def _validate_movie_overrides(
    sidecar_fps: float,
    sidecar_pixel_to_grid: Tuple[float, float],
    sidecar_origin: Tuple[float, float],
    fps: Optional[float],
    pixel_to_grid: Optional[Tuple[float, float]],
    origin: Optional[Tuple[float, float]],
    extent: Optional[Tuple[float, float]],
) -> None:
    if fps is not None and float(fps) != sidecar_fps:
        raise ValidationError(
            "movie_fps must match sidecar.fps when a movie metadata sidecar is present"
        )
    if pixel_to_grid is not None and pixel_to_grid != sidecar_pixel_to_grid:
        raise ValidationError(
            "movie_pixel_to_grid must match sidecar-derived pixel_to_grid when a movie metadata sidecar is present"
        )
    if origin is not None and origin != sidecar_origin:
        raise ValidationError(
            "movie_origin must match the fixed origin (0,0) implied by the movie metadata sidecar"
        )
    if extent is not None:
        raise ValidationError(
            "movie_extent overrides are not supported when using a movie metadata sidecar; extent is derived from data shape and sidecar calibration"
        )


def _ingest_hdf5_from_sidecar(
    path: Path,
    output: Path,
    sidecar: MovieMetadataSidecar,
    sidecar_fps: float,
    sidecar_pixel_to_grid: Tuple[float, float],
    sidecar_origin: Tuple[float, float],
    normalize: bool,
    movie_h5_dataset: Optional[str],
) -> Path:
    if sidecar.h5_dataset is None:
        raise ValidationError(
            "h5_dataset must be provided in the movie metadata sidecar for HDF5 sources"
        )
    if movie_h5_dataset is not None and movie_h5_dataset != sidecar.h5_dataset:
        raise ValidationError(
            "movie_h5_dataset must match h5_dataset in the movie metadata sidecar for HDF5 sources"
        )

    from plume_nav_sim.media.h5_movie import H5MovieIngestConfig, ingest_h5_movie

    cfg_h5 = H5MovieIngestConfig(
        input=path,
        dataset=sidecar.h5_dataset,
        output=output,
        t_start=0,
        t_stop=None,
        fps=sidecar_fps,
        pixel_to_grid=sidecar_pixel_to_grid,
        origin=sidecar_origin,
        extent=None,
        normalize=normalize,
        chunk_t=None,
    )
    return ingest_h5_movie(cfg_h5)


def _ingest_video_from_sidecar(
    path: Path,
    output: Path,
    sidecar_fps: float,
    sidecar_pixel_to_grid: Tuple[float, float],
    sidecar_origin: Tuple[float, float],
    normalize: bool,
) -> Path:
    from plume_nav_sim.cli.video_ingest import IngestConfig, ingest_video

    cfg = IngestConfig(
        input=path,
        output=output,
        fps=sidecar_fps,
        pixel_to_grid=sidecar_pixel_to_grid,
        origin=sidecar_origin,
        extent=None,
        normalize=normalize,
    )
    return ingest_video(cfg)


__all__ = ["VideoPlume", "VideoConfig", "resolve_movie_dataset_path"]
