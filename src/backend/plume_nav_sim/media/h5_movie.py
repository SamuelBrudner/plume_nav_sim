from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

CHUNKS_TYX: tuple[int, int, int] = (8, 64, 64)


def _try_import_h5py():
    try:
        import h5py

        return h5py
    except ImportError as e:  # pragma: no cover - exercised when optional deps missing
        raise ImportError(
            "h5py is required for HDF5 movie ingest. Install with 'pip install plume-nav-sim[all]' "
            "(or from source: pip install -e \".[all]\")."
        ) from e


def _git_sha() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


@dataclass
class H5MovieIngestConfig:
    input: Path
    dataset: str
    output: Path
    t_start: int = 0
    t_stop: Optional[int] = None
    fps: Optional[float] = None
    pixel_to_grid: Optional[Tuple[float, float]] = None
    origin: Optional[Tuple[float, float]] = None
    extent: Optional[Tuple[float, float]] = None
    source_location_px: Optional[Tuple[int, int]] = None
    normalize: bool = False
    chunk_t: Optional[int] = None


def _get_h5_movie_dataset(f, dataset_name: str):
    if not dataset_name:
        raise ValueError("dataset must be provided for HDF5 movie ingest")
    if dataset_name not in f:
        raise ValueError(f"HDF5 dataset not found: {dataset_name}")
    dset = f[dataset_name]
    if dset.ndim != 3:
        raise ValueError(
            f"HDF5 movie dataset must be 3D (t,y,x); got ndim={dset.ndim}, shape={dset.shape}"
        )
    t_raw, size_y, size_x = map(int, dset.shape)
    return dset, t_raw, size_y, size_x


def _compute_time_window(cfg: H5MovieIngestConfig, t_raw: int) -> tuple[int, int, int]:
    t_start = int(cfg.t_start) if cfg.t_start is not None else 0
    if t_start < 0:
        t_start = 0
    t_stop = int(cfg.t_stop) if cfg.t_stop is not None else t_raw
    if t_stop > t_raw:
        t_stop = t_raw
    if t_stop <= t_start:
        raise ValueError(
            f"Invalid time window for HDF5 movie ingest: t_start={t_start}, t_stop={t_stop}, total={t_raw}"
        )
    t = t_stop - t_start
    return t_start, t_stop, t


def _describe_source_dtype(src_dtype) -> str:
    return src_dtype.name if hasattr(src_dtype, "name") else str(src_dtype)


def _resolve_fps(cfg: H5MovieIngestConfig, f) -> float:
    fps = cfg.fps
    if fps is not None:
        return float(fps)
    try:
        frame_rate_ds = f["Attributes/imagingParameters/frameRate"]
        fps_val = float(frame_rate_ds[0, 0])
        if fps_val <= 0:
            raise ValueError
        return fps_val
    except Exception:
        raise ValueError(
            "fps must be provided for HDF5 ingest when imagingParameters/frameRate is unavailable or invalid"
        )


def _resolve_spatial_metadata(
    cfg: H5MovieIngestConfig, size_y: int, size_x: int
) -> tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    pixel_to_grid = cfg.pixel_to_grid or (1.0, 1.0)
    origin = cfg.origin or (0.0, 0.0)
    if cfg.extent is not None:
        extent = cfg.extent
    else:
        extent = (
            float(size_y) * float(pixel_to_grid[0]),
            float(size_x) * float(pixel_to_grid[1]),
        )
    return pixel_to_grid, origin, extent


def _create_zarr_and_write_attrs(
    _zarr,
    create_zarr_array,
    output_path: Path,
    t: int,
    size_y: int,
    size_x: int,
    chunks_tyx,
    dims_tyx,
    variable_name: str,
    attrs_model,
    input_path: Path,
    cfg: H5MovieIngestConfig,
    t_start: int,
    t_stop: int,
    fps: float,
    pixel_to_grid: Tuple[float, float],
    origin: Tuple[float, float],
    extent: Tuple[float, float],
):
    arr = create_zarr_array(
        output_path,
        name=variable_name,
        shape=(t, size_y, size_x),
        dtype="float32",
        chunks=chunks_tyx,
        overwrite=True,
        extra_attrs={"_ARRAY_DIMENSIONS": list(dims_tyx)},
    )
    grp = _zarr.open_group(output_path, mode="a")
    for k, v in attrs_model.model_dump().items():
        grp.attrs[k] = v
    grp.attrs["ingest_args"] = {
        "input": str(input_path),
        "dataset": cfg.dataset,
        "t_start": int(t_start),
        "t_stop": int(t_stop),
        "fps": float(fps),
        "pixel_to_grid": list(pixel_to_grid),
        "origin": list(origin),
        "extent": list(extent),
        "normalize": bool(cfg.normalize),
    }
    if cfg.source_location_px is not None:
        grp.attrs["source_location_px"] = [
            int(cfg.source_location_px[0]),
            int(cfg.source_location_px[1]),
        ]
        grp.attrs["ingest_args"]["source_location_px"] = [
            int(cfg.source_location_px[0]),
            int(cfg.source_location_px[1]),
        ]
    return arr


def _normalize_block(block, normalize: bool) -> np.ndarray:
    if not normalize:
        return block.astype(np.float32)
    if np.issubdtype(block.dtype, np.integer):
        maxv = np.iinfo(block.dtype).max
        return (block.astype(np.float32) / float(maxv)).astype(np.float32)
    return block.astype(np.float32)


def _write_frames_to_zarr(
    dset,
    arr,
    t_start: int,
    t: int,
    chunk_t_cfg: Optional[int],
    chunks_tyx,
    normalize: bool,
) -> None:
    chunk_t = int(chunk_t_cfg) if chunk_t_cfg is not None else int(chunks_tyx[0])
    if chunk_t <= 0:
        chunk_t = int(chunks_tyx[0])
    current = 0
    while current < t:
        end = min(current + chunk_t, t)
        block = dset[t_start + current : t_start + end, :, :]
        block_f = _normalize_block(block, normalize)
        arr[current:end, :, :] = block_f
        current = end


def _try_import_zarr():
    try:
        import zarr as _zarr

        return _zarr
    except Exception as e:  # pragma: no cover - exercised when optional deps missing
        raise ImportError(
            "zarr is required for HDF5 movie ingest. Install with 'pip install plume-nav-sim[all]' "
            "(or from source: pip install -e \".[all]\")."
        ) from e


def _create_zarr_array(
    store_path: Path,
    *,
    name: str,
    shape: tuple[int, int, int],
    dtype: str,
    chunks: tuple[int, int, int],
    overwrite: bool,
    extra_attrs: Optional[dict],
):
    compressor = None
    try:
        from numcodecs import Blosc

        compressor = Blosc(cname="zstd", clevel=3, shuffle=1)
    except Exception:
        compressor = None

    _zarr = _try_import_zarr()
    grp = _zarr.open_group(store_path, mode="a")
    if overwrite and name in grp:
        del grp[name]
    arr = grp.require_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
    )
    if extra_attrs:
        arr.attrs.update(extra_attrs)
    return arr


def ingest_h5_movie(cfg: H5MovieIngestConfig) -> Path:
    """Convert an HDF5 plume movie into a Zarr dataset compatible with MoviePlumeField."""

    h5py = _try_import_h5py()
    _zarr = _try_import_zarr()

    from ..video.schema import DIMS_TYX, VARIABLE_NAME, VideoPlumeAttrs, validate_attrs
    from .manifest import build_provenance_manifest, write_manifest

    input_path = Path(cfg.input)
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as f:
        dset, t_raw, size_y, size_x = _get_h5_movie_dataset(f, cfg.dataset)
        t_start, t_stop, t = _compute_time_window(cfg, t_raw)
        src_dtype_str = _describe_source_dtype(dset.dtype)
        fps = _resolve_fps(cfg, f)
        pixel_to_grid, origin, extent = _resolve_spatial_metadata(cfg, size_y, size_x)

        attrs = VideoPlumeAttrs(
            fps=float(fps),
            source_dtype=str(src_dtype_str),
            pixel_to_grid=pixel_to_grid,
            origin=origin,
            extent=extent,
        )
        valid = validate_attrs(attrs.model_dump())

        arr = _create_zarr_and_write_attrs(
            _zarr,
            _create_zarr_array,
            output_path,
            t,
            size_y,
            size_x,
            CHUNKS_TYX,
            DIMS_TYX,
            VARIABLE_NAME,
            valid,
            input_path,
            cfg,
            t_start,
            t_stop,
            fps,
            pixel_to_grid,
            origin,
            extent,
        )

        _write_frames_to_zarr(
            dset,
            arr,
            t_start,
            t,
            cfg.chunk_t,
            CHUNKS_TYX,
            cfg.normalize,
        )

    manifest = build_provenance_manifest(
        source_dtype=str(src_dtype_str),
        cli_args=None,
        git_sha=_git_sha(),
    )
    write_manifest(output_path, manifest)

    _zarr.consolidate_metadata(output_path)
    return output_path
