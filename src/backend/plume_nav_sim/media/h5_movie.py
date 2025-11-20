from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _try_import_h5py():
    try:
        import h5py  # type: ignore

        return h5py
    except Exception as e:  # pragma: no cover - exercised when media extras missing
        raise ImportError(
            "h5py is required for HDF5 movie ingest. Install with 'pip install h5py'."
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
    """Configuration for converting an HDF5 plume movie into a Zarr dataset.

    The fields mirror the CLI-style options used by ingest helpers and tests,
    and are kept deliberately small so that ingest_h5_movie can be exercised
    directly from Python as well as via DVC/CI pipelines.
    """

    input: Path
    dataset: str
    output: Path
    t_start: int = 0
    t_stop: Optional[int] = None
    fps: Optional[float] = None
    pixel_to_grid: Optional[Tuple[float, float]] = None
    origin: Optional[Tuple[float, float]] = None
    extent: Optional[Tuple[float, float]] = None
    normalize: bool = False
    chunk_t: Optional[int] = None


def ingest_h5_movie(cfg: H5MovieIngestConfig) -> Path:
    """Convert an HDF5 plume movie into a Zarr dataset compatible with MoviePlumeField."""

    h5py = _try_import_h5py()

    from ..storage import CHUNKS_TYX, create_zarr_array
    from ..video.schema import DIMS_TYX, VARIABLE_NAME, VideoPlumeAttrs, validate_attrs
    from .manifest import build_provenance_manifest, write_manifest

    try:
        import zarr as _zarr  # type: ignore
    except Exception as e:  # pragma: no cover - exercised when media extras missing
        raise ImportError(
            "zarr is required for HDF5 movie ingest. Install with 'pip install zarr numcodecs'."
        ) from e

    if not cfg.dataset:
        raise ValueError("dataset must be provided for HDF5 movie ingest")

    input_path = Path(cfg.input)
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as f:
        if cfg.dataset not in f:
            raise ValueError(f"HDF5 dataset not found: {cfg.dataset}")

        dset = f[cfg.dataset]
        if dset.ndim != 3:
            raise ValueError(
                f"HDF5 movie dataset must be 3D (t,y,x); got ndim={dset.ndim}, shape={dset.shape}"
            )

        t_raw, size_y, size_x = map(int, dset.shape)

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

        src_dtype = dset.dtype
        src_dtype_str = src_dtype.name if hasattr(src_dtype, "name") else str(src_dtype)

        fps = cfg.fps
        if fps is None:
            # Best-effort inference from Attributes/imagingParameters/frameRate
            try:
                frame_rate_ds = f["Attributes/imagingParameters/frameRate"]
                fps_val = float(frame_rate_ds[0, 0])
                if fps_val <= 0:
                    raise ValueError
                fps = fps_val
            except Exception:
                raise ValueError(
                    "fps must be provided for HDF5 ingest when imagingParameters/frameRate is unavailable or invalid"
                )

        pixel_to_grid = cfg.pixel_to_grid or (1.0, 1.0)
        origin = cfg.origin or (0.0, 0.0)
        if cfg.extent is not None:
            extent = cfg.extent
        else:
            extent = (
                float(size_y) * float(pixel_to_grid[0]),
                float(size_x) * float(pixel_to_grid[1]),
            )

        attrs = VideoPlumeAttrs(
            fps=float(fps),
            source_dtype=str(src_dtype_str),
            pixel_to_grid=pixel_to_grid,
            origin=origin,
            extent=extent,
        )
        valid = validate_attrs(attrs.model_dump())

        arr = create_zarr_array(
            output_path,
            name=VARIABLE_NAME,
            shape=(t, size_y, size_x),
            dtype="float32",
            chunks=CHUNKS_TYX,
            overwrite=True,
            extra_attrs={"_ARRAY_DIMENSIONS": list(DIMS_TYX)},
        )

        grp = _zarr.open_group(output_path, mode="a")
        for k, v in valid.model_dump().items():
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

        chunk_t = int(cfg.chunk_t) if cfg.chunk_t is not None else int(CHUNKS_TYX[0])
        if chunk_t <= 0:
            chunk_t = int(CHUNKS_TYX[0])

        current = 0
        while current < t:
            end = min(current + chunk_t, t)
            block = dset[t_start + current : t_start + end, :, :]

            if cfg.normalize:
                if np.issubdtype(block.dtype, np.integer):
                    maxv = np.iinfo(block.dtype).max
                    block_f = (block.astype(np.float32) / float(maxv)).astype(
                        np.float32
                    )
                else:
                    block_f = block.astype(np.float32)
            else:
                block_f = block.astype(np.float32)

            arr[current:end, :, :] = block_f
            current = end

    manifest = build_provenance_manifest(
        source_dtype=str(src_dtype_str),
        cli_args=None,
        git_sha=_git_sha(),
    )
    write_manifest(output_path, manifest)

    _zarr.consolidate_metadata(output_path)
    return output_path
