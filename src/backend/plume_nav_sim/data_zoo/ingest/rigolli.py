from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from plume_nav_sim.video.schema import DIMS_TYX, SCHEMA_VERSION

from ..downloader import DatasetDownloadError, _http_request
from ..registry import RigolliDNSIngest
from ..stats import compute_concentration_stats, store_stats_in_zarr
from . import _compute_chunk_t, _normalize_concentration, _require_ingest_deps

LOG = logging.getLogger("plume_nav_sim.data_zoo.download")


def _rigolli_axis_candidates(
    shape: tuple[int, ...], *, x_len: int, y_len: int
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for time_axis in (0, 1, 2):
        other_axes = [ax for ax in (0, 1, 2) if ax != time_axis]
        for y_axis, x_axis in (other_axes, other_axes[::-1]):
            if shape[y_axis] == y_len and shape[x_axis] == x_len:
                candidates.append((time_axis, y_axis, x_axis))
    return candidates


def _pick_rigolli_axes(
    candidates: list[tuple[int, int, int]], shape: tuple[int, ...]
) -> tuple[int, int, int]:
    if len(candidates) == 1:
        return candidates[0]

    if candidates:
        # Prefer MATLAB-like ordering when ambiguous: (x, y, t) -> (t, y, x)
        if (2, 1, 0) in candidates:
            return (2, 1, 0)
        # Otherwise prefer time-last when possible.
        for time_axis, y_axis, x_axis in candidates:
            if time_axis == 2:
                return (time_axis, y_axis, x_axis)
        return candidates[0]

    time_axis = max(range(3), key=lambda ax: shape[ax])
    other_axes = [ax for ax in (0, 1, 2) if ax != time_axis]
    return (time_axis, other_axes[0], other_axes[1])


def _infer_time_y_x_axes_for_rigolli(
    shape: tuple[int, ...], *, x_len: int, y_len: int
) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise DatasetDownloadError(
            f"Expected 3D concentration array, got shape {shape}"
        )

    candidates = _rigolli_axis_candidates(shape, x_len=x_len, y_len=y_len)
    return _pick_rigolli_axes(candidates, shape)


def _ensure_coords_file(spec: RigolliDNSIngest, source_path: Path) -> Path:
    coords_path = source_path.parent / "coordinates.mat"
    if coords_path.exists():
        return coords_path

    LOG.info("Downloading coordinates file for %s", source_path.name)
    coords_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            urllib.request.urlopen(_http_request(spec.coords_url)) as response,
            coords_path.open("wb") as fh,
        ):
            shutil.copyfileobj(response, fh)
    except Exception as exc:
        raise DatasetDownloadError(
            f"Failed to download coordinates from {spec.coords_url}: {exc}"
        ) from exc

    hasher = hashlib.md5(usedforsecurity=False)
    with coords_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    if hasher.hexdigest().lower() != spec.coords_checksum.lower():
        coords_path.unlink()
        raise DatasetDownloadError(
            f"Coordinates file checksum mismatch for {spec.coords_url}"
        )

    return coords_path


def _try_load_coords_with_scipy(
    spec: RigolliDNSIngest,
    coords_path: Path,
    np: Any,
    errors: list[Exception],
) -> tuple[Any, Any] | None:
    try:
        from scipy.io import loadmat
    except Exception as exc:  # pragma: no cover - scipy is optional
        errors.append(exc)
        return None

    try:
        coords_mat = loadmat(str(coords_path))
        x_coords = np.asarray(coords_mat[spec.x_key]).flatten().astype(np.float32)
        y_coords = np.asarray(coords_mat[spec.y_key]).flatten().astype(np.float32)
        return x_coords, y_coords
    except Exception as exc:
        errors.append(exc)
        return None


def _try_load_coords_with_h5py(
    spec: RigolliDNSIngest,
    coords_path: Path,
    h5py: Any,
    np: Any,
    errors: list[Exception],
) -> tuple[Any, Any] | None:
    try:
        with h5py.File(coords_path, "r") as coords_file:
            x_data = coords_file[spec.x_key][:]
            y_data = coords_file[spec.y_key][:]
            x_coords = np.asarray(x_data).flatten().astype(np.float32)
            y_coords = np.asarray(y_data).flatten().astype(np.float32)
            return x_coords, y_coords
    except Exception as exc:
        errors.append(exc)
        return None


def _load_rigolli_coords(
    spec: RigolliDNSIngest, coords_path: Path, *, h5py: Any, np: Any
) -> tuple[Any, Any]:
    errors: list[Exception] = []
    result = _try_load_coords_with_scipy(spec, coords_path, np, errors)
    if result is not None:
        return result
    result = _try_load_coords_with_h5py(spec, coords_path, h5py, np, errors)
    if result is not None:
        return result
    details = "; ".join(type(e).__name__ for e in errors)
    raise DatasetDownloadError(
        "Failed to load Rigolli DNS coordinates file. "
        "If this is a legacy MATLAB v5 file, install SciPy; "
        f"load attempts failed with: {details}"
    )


def _get_mat_concentration_dataset(
    mat_file: Any, spec: RigolliDNSIngest, source_path: Path
) -> Any:
    if spec.concentration_key not in mat_file:
        raise DatasetDownloadError(
            f"Could not find '{spec.concentration_key}' in {source_path.name}. "
            f"Available keys: {list(mat_file.keys())}"
        )
    return mat_file[spec.concentration_key]


def _init_rigolli_store(
    *,
    output_path: Path,
    n_frames: int,
    ny: int,
    nx: int,
    chunk_t: int,
    x_coords: Any,
    y_coords: Any,
    zarr: Any,
) -> tuple[Any, Any]:
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store, overwrite=True)
    conc = root.create_dataset(
        "concentration",
        shape=(n_frames, ny, nx),
        chunks=(chunk_t, ny, nx),
        dtype="float32",
    )
    conc.attrs["_ARRAY_DIMENSIONS"] = list(DIMS_TYX)
    x_arr = root.create_dataset("x", data=x_coords, dtype="float32")
    x_arr.attrs["_ARRAY_DIMENSIONS"] = ["x"]
    y_arr = root.create_dataset("y", data=y_coords, dtype="float32")
    y_arr.attrs["_ARRAY_DIMENSIONS"] = ["y"]
    return root, conc


def _stream_rigolli_chunks(
    *,
    conc_dataset: Any,
    conc: Any,
    n_frames: int,
    chunk_t: int,
    time_axis: int,
    y_axis: int,
    x_axis: int,
    normalize: bool,
    np: Any,
) -> tuple[float, float]:
    global_min, global_max = np.inf, -np.inf
    for t_start in range(0, n_frames, chunk_t):
        t_end = min(t_start + chunk_t, n_frames)
        slicer = [slice(None), slice(None), slice(None)]
        slicer[time_axis] = slice(t_start, t_end)
        chunk = conc_dataset[tuple(slicer)]
        chunk = np.transpose(chunk, (time_axis, y_axis, x_axis)).astype(np.float32)
        conc[t_start:t_end] = chunk

        if normalize:
            global_min = min(global_min, float(chunk.min()))
            global_max = max(global_max, float(chunk.max()))

        if (t_start // chunk_t) % 10 == 0:
            LOG.info("  Processed %d / %d frames...", t_end, n_frames)

    return global_min, global_max


def _write_rigolli_attrs(
    root: Any,
    *,
    spec: RigolliDNSIngest,
    n_frames: int,
    ny: int,
    nx: int,
    x_coords: Any,
    y_coords: Any,
) -> None:
    fps = float(spec.fps) if spec.fps is not None else 1.0
    if spec.fps is None:
        LOG.warning(
            "Rigolli DNS ingest missing fps; defaulting to fps=1.0 for xarray compatibility."
        )
        root.attrs["fps_is_placeholder"] = True

    x_vals = x_coords
    y_vals = y_coords
    try:
        x_min = float(min(x_vals))
        x_max = float(max(x_vals))
    except Exception:
        x_min = 0.0
        x_max = float(nx)
    try:
        y_min = float(min(y_vals))
        y_max = float(max(y_vals))
    except Exception:
        y_min = 0.0
        y_max = float(ny)

    pixel_to_grid_y = (y_max - y_min) / float(max(1, len(y_vals) - 1))
    pixel_to_grid_x = (x_max - x_min) / float(max(1, len(x_vals) - 1))

    root.attrs["schema_version"] = SCHEMA_VERSION
    root.attrs["fps"] = fps
    root.attrs["source_dtype"] = "float32"
    root.attrs["pixel_to_grid"] = [float(pixel_to_grid_y), float(pixel_to_grid_x)]
    root.attrs["origin"] = [float(y_min), float(x_min)]
    root.attrs["extent"] = [float(y_max - y_min), float(x_max - x_min)]
    root.attrs["dims"] = list(DIMS_TYX)

    root.attrs["n_frames"] = int(n_frames)
    root.attrs["shape"] = [int(n_frames), int(ny), int(nx)]
    root.attrs["source_format"] = "matlab_v73"
    root.attrs["normalized"] = bool(spec.normalize)

    source_location_px = getattr(spec, "source_location_px", None)
    if source_location_px is None:
        source_location_px = (0, int(ny) // 2)
        root.attrs["source_location_px_is_heuristic"] = True
        LOG.warning(
            "Rigolli DNS ingest missing source_location_px; defaulting to heuristic (x=%d, y=%d).",
            source_location_px[0],
            source_location_px[1],
        )
    root.attrs["source_location_px"] = [
        int(source_location_px[0]),
        int(source_location_px[1]),
    ]


def _ingest_mat_to_zarr(
    spec: RigolliDNSIngest,
    source_path: Path,
    output_path: Path,
    compute_stats: bool = True,
) -> Path:
    try:
        h5py, np, zarr = _require_ingest_deps("MATLAB ingest")

        coords_path = _ensure_coords_file(spec, source_path)
        x_coords, y_coords = _load_rigolli_coords(spec, coords_path, h5py=h5py, np=np)

        # Load concentration data from source MATLAB file (streaming in time)
        with h5py.File(source_path, "r") as mat_file:
            conc_dataset = _get_mat_concentration_dataset(mat_file, spec, source_path)
            src_shape = tuple(int(s) for s in conc_dataset.shape)
            time_axis, y_axis, x_axis = _infer_time_y_x_axes_for_rigolli(
                src_shape, x_len=len(x_coords), y_len=len(y_coords)
            )

            n_frames = src_shape[time_axis]
            ny = src_shape[y_axis]
            nx = src_shape[x_axis]

            # Sanity: match inferred spatial dims to coordinate arrays when possible.
            if ny != len(y_coords) or nx != len(x_coords):
                LOG.warning(
                    "Rigolli DNS coordinate length mismatch: inferred (ny=%d,nx=%d) "
                    "but coords are (len(y)=%d,len(x)=%d). Proceeding with inferred dims.",
                    ny,
                    nx,
                    len(y_coords),
                    len(x_coords),
                )

            # Choose chunk size to keep Zarr chunks well below codec limits.
            bytes_per_frame = ny * nx * 4  # float32
            chunk_t = _compute_chunk_t(spec.chunk_t, bytes_per_frame)

            root, conc = _init_rigolli_store(
                output_path=output_path,
                n_frames=n_frames,
                ny=ny,
                nx=nx,
                chunk_t=chunk_t,
                x_coords=x_coords,
                y_coords=y_coords,
                zarr=zarr,
            )

            global_min, global_max = _stream_rigolli_chunks(
                conc_dataset=conc_dataset,
                conc=conc,
                n_frames=n_frames,
                chunk_t=chunk_t,
                time_axis=time_axis,
                y_axis=y_axis,
                x_axis=x_axis,
                normalize=bool(spec.normalize),
                np=np,
            )

        # Normalize in a second pass if needed
        if spec.normalize and global_max > global_min:
            LOG.info("Normalizing data (min=%.4f, max=%.4f)...", global_min, global_max)
            _normalize_concentration(
                conc,
                n_frames=n_frames,
                chunk_t=chunk_t,
                global_min=float(global_min),
                global_max=float(global_max),
            )

        # Store metadata in .zattrs
        _write_rigolli_attrs(
            root,
            spec=spec,
            n_frames=n_frames,
            ny=ny,
            nx=nx,
            x_coords=x_coords,
            y_coords=y_coords,
        )

        LOG.info(
            "Ingested MATLAB file to Zarr: %s -> %s (shape=[%d, %d, %d])",
            source_path.name,
            output_path.name,
            n_frames,
            ny,
            nx,
        )

        # Compute and store concentration statistics
        if compute_stats:
            LOG.info("Computing concentration stats for %s...", output_path.name)
            stats = compute_concentration_stats(output_path)
            if spec.normalize:
                stats["normalized_during_ingest"] = True
            store_stats_in_zarr(output_path, stats)

        zarr.consolidate_metadata(output_path)

        return output_path

    except Exception as exc:
        # Clean up partial output on failure
        if output_path.exists():
            shutil.rmtree(output_path)
        raise DatasetDownloadError(
            f"Failed to ingest MATLAB file {source_path} into Zarr: {exc}"
        ) from exc

