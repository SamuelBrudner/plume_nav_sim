from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from plume_nav_sim.video.schema import DIMS_TYX, SCHEMA_VERSION

from ..downloader import (
    DatasetDownloadError,
    _http_request,
    _resolve_dryad_presigned_url,
)
from ..registry import EmonetSmokeIngest
from ..stats import compute_concentration_stats, store_stats_in_zarr
from . import _compute_chunk_t, _normalize_concentration, _require_ingest_deps

LOG = logging.getLogger("plume_nav_sim.data_zoo.download")


def _compute_emonet_background(
    *,
    frames_dataset: Any,
    n_bg: int,
    np: Any,
) -> Any:
    background = None
    for idx in range(n_bg):
        frame = frames_dataset[idx]
        frame = np.transpose(frame, (1, 0)).astype(np.float32)
        if background is None:
            background = np.zeros_like(frame, dtype=np.float32)
        background += frame
    if background is None:
        raise DatasetDownloadError(
            "Failed to compute Emonet background: no frames read"
        )
    return background / float(n_bg)


def _emonet_frame_signal(
    frame: Any,
    *,
    background: Any | None,
    subtract_background: bool,
    np: Any,
) -> float:
    frame_f = frame.astype(np.float32)
    if subtract_background and background is not None:
        frame_f = frame_f - background
        frame_f = np.clip(frame_f, 0.0, None)
    return float(frame_f.mean())


def _estimate_emonet_start_frame(
    *,
    frames_dataset: Any,
    n_frames: int,
    background: Any | None,
    spec: Any,
    np: Any,
) -> tuple[int, dict[str, float]]:
    n_bg = int(getattr(spec, "background_n_frames", 0))
    n_scan = min(int(getattr(spec, "trim_max_scan", 0)), n_frames)
    if n_scan <= 0:
        return 0, {}

    subtract_background = bool(getattr(spec, "background_subtract", False))
    baseline_end = min(max(n_bg, 1), n_scan)
    baseline: list[float] = []
    for idx in range(baseline_end):
        frame = frames_dataset[idx]
        frame = np.transpose(frame, (1, 0))
        baseline.append(
            _emonet_frame_signal(
                frame,
                background=background,
                subtract_background=subtract_background,
                np=np,
            )
        )
    baseline_arr = np.asarray(baseline, dtype=np.float32)
    baseline_mean = float(baseline_arr.mean())
    baseline_std = float(baseline_arr.std())

    trim_abs_threshold = getattr(spec, "trim_abs_threshold", None)
    if trim_abs_threshold is not None:
        try:
            threshold = float(trim_abs_threshold)
        except Exception as exc:
            raise DatasetDownloadError(
                f"Invalid trim_abs_threshold={trim_abs_threshold!r} (must be float)"
            ) from exc
        if not np.isfinite(threshold) or threshold < 0.0:
            raise DatasetDownloadError(
                f"Invalid trim_abs_threshold={threshold} (must be finite and >= 0)"
            )
    else:
        threshold = (
            baseline_mean + float(getattr(spec, "trim_sigma", 0.0)) * baseline_std
        )
    consecutive_needed = int(getattr(spec, "trim_consecutive", 1))
    consecutive = 0
    for idx in range(baseline_end, n_scan):
        frame = frames_dataset[idx]
        frame = np.transpose(frame, (1, 0))
        signal = _emonet_frame_signal(
            frame,
            background=background,
            subtract_background=subtract_background,
            np=np,
        )
        if signal > threshold:
            consecutive += 1
            if consecutive >= consecutive_needed:
                return idx - consecutive_needed + 1, {
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "threshold": float(threshold),
                }
        else:
            consecutive = 0

    return 0, {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "threshold": float(threshold),
    }


def _estimate_emonet_end_frame(
    *,
    frames_dataset: Any,
    n_frames: int,
    start_frame: int,
    background: Any | None,
    spec: Any,
    np: Any,
) -> tuple[int, dict[str, float]]:
    if n_frames <= 0:
        return 0, {}
    if start_frame < 0:
        start_frame = 0
    if start_frame >= n_frames:
        return n_frames, {}

    subtract_background = bool(getattr(spec, "background_subtract", False))
    consecutive_needed = int(getattr(spec, "end_consecutive", 1))
    if consecutive_needed < 1:
        raise DatasetDownloadError(
            f"Invalid end_consecutive={consecutive_needed} (must be >= 1)"
        )

    end_abs_threshold = getattr(spec, "end_abs_threshold", None)
    if end_abs_threshold is None:
        end_abs_threshold = getattr(spec, "trim_abs_threshold", None)

    stats: dict[str, float] = {}
    if end_abs_threshold is not None:
        try:
            threshold = float(end_abs_threshold)
        except Exception as exc:
            raise DatasetDownloadError(
                f"Invalid end_abs_threshold={end_abs_threshold!r} (must be float)"
            ) from exc
        if not np.isfinite(threshold) or threshold < 0.0:
            raise DatasetDownloadError(
                f"Invalid end_abs_threshold={threshold} (must be finite and >= 0)"
            )
    else:
        baseline_n = int(getattr(spec, "background_n_frames", 0))
        baseline_n = max(baseline_n, 1)
        baseline_n = min(baseline_n, n_frames - start_frame)
        if baseline_n <= 0:
            return n_frames, {}

        baseline: list[float] = []
        baseline_start = n_frames - baseline_n
        for idx in range(baseline_start, n_frames):
            frame = frames_dataset[idx]
            frame = np.transpose(frame, (1, 0))
            baseline.append(
                _emonet_frame_signal(
                    frame,
                    background=background,
                    subtract_background=subtract_background,
                    np=np,
                )
            )

        baseline_arr = np.asarray(baseline, dtype=np.float32)
        baseline_mean = float(baseline_arr.mean())
        baseline_std = float(baseline_arr.std())
        threshold = (
            baseline_mean + float(getattr(spec, "end_sigma", 0.0)) * baseline_std
        )
        stats["baseline_mean"] = baseline_mean
        stats["baseline_std"] = baseline_std

    stats["threshold"] = float(threshold)

    tail_len = 0
    for idx in range(n_frames - 1, start_frame - 1, -1):
        frame = frames_dataset[idx]
        frame = np.transpose(frame, (1, 0))
        signal = _emonet_frame_signal(
            frame,
            background=background,
            subtract_background=subtract_background,
            np=np,
        )
        if signal <= threshold:
            tail_len += 1
        else:
            break

    stats["tail_len"] = float(tail_len)
    if tail_len < consecutive_needed:
        return n_frames, stats
    return n_frames - tail_len, stats


def _download_emonet_metadata(
    spec: EmonetSmokeIngest, source_path: Path
) -> Path | None:
    if not spec.metadata_url:
        return None

    meta_path = source_path.parent / "metadata.mat"
    if meta_path.exists():
        return meta_path

    LOG.info("Downloading metadata file for %s", source_path.name)
    try:
        resolved_url = _resolve_dryad_presigned_url(spec.metadata_url)
        with (
            urllib.request.urlopen(_http_request(resolved_url)) as response,
            meta_path.open("wb") as fh,
        ):
            shutil.copyfileobj(response, fh)
    except Exception as exc:
        LOG.warning("Failed to download metadata: %s (continuing without)", exc)
        return None

    return meta_path


def _parse_emonet_metadata(meta_path: Path, *, h5py: Any, np: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        with h5py.File(meta_path, "r") as mf:
            for key in mf.keys():
                try:
                    metadata[key] = np.asarray(mf[key]).item()
                except (ValueError, TypeError):
                    pass
    except Exception as exc:
        LOG.warning("Failed to parse metadata file: %s", exc)
    return metadata


def _load_emonet_metadata(
    spec: EmonetSmokeIngest, source_path: Path, *, h5py: Any, np: Any
) -> dict[str, Any]:
    meta_path = _download_emonet_metadata(spec, source_path)
    if meta_path is None:
        return {}
    return _parse_emonet_metadata(meta_path, h5py=h5py, np=np)


def _resolve_frames_key(
    mat_file: Any, spec: EmonetSmokeIngest, source_path: Path
) -> str:
    frames_key = spec.frames_key

    # h5py allows nested paths (e.g., "ComplexPlume/frames"). Prefer the
    # explicitly configured key if it exists.
    try:
        obj = mat_file.get(frames_key)
    except Exception:
        obj = None
    if obj is not None and hasattr(obj, "ndim") and int(obj.ndim) == 3:
        return frames_key

    # Otherwise, search for candidate 3D datasets and pick the largest (by
    # element count). This is metadata-only and does not read the dataset.
    candidates: list[tuple[int, str]] = []

    def _visit(name: str, node: Any) -> None:
        if not hasattr(node, "shape"):
            return
        try:
            shape = tuple(int(x) for x in node.shape)
        except Exception:
            return
        if len(shape) != 3:
            return
        n = 1
        for s in shape:
            n *= int(s)
        candidates.append((n, name))

    try:
        mat_file.visititems(_visit)
    except Exception as exc:
        raise DatasetDownloadError(
            f"Failed while scanning {source_path} for a 3D frames dataset: {exc}"
        ) from exc

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    raise DatasetDownloadError(
        f"Could not find frames array in {source_path}. "
        f"Available keys: {list(mat_file.keys())}"
    )


def _init_emonet_store(
    *,
    output_path: Path,
    n_frames: int,
    n_y: int,
    n_x: int,
    chunk_t: int,
    x_coords: Any,
    y_coords: Any,
    zarr: Any,
) -> tuple[Any, Any]:
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store, overwrite=True)
    conc = root.create_dataset(
        "concentration",
        shape=(n_frames, n_y, n_x),
        chunks=(chunk_t, n_y, n_x),
        dtype="float32",
    )
    conc.attrs["_ARRAY_DIMENSIONS"] = list(DIMS_TYX)
    x_arr = root.create_dataset("x", data=x_coords, dtype="float32")
    x_arr.attrs["_ARRAY_DIMENSIONS"] = ["x"]
    y_arr = root.create_dataset("y", data=y_coords, dtype="float32")
    y_arr.attrs["_ARRAY_DIMENSIONS"] = ["y"]
    return root, conc


def _stream_emonet_chunks(
    *,
    frames_dataset: Any,
    conc: Any,
    n_frames: int,
    start_frame: int,
    end_frame: int,
    batch_size: int,
    normalize: bool,
    background: Any | None,
    subtract_background: bool,
    np: Any,
) -> tuple[float, float]:
    if end_frame <= start_frame:
        raise DatasetDownloadError(
            f"Invalid trim range: start_frame={start_frame} end_frame={end_frame}"
        )
    global_min, global_max = np.inf, -np.inf
    for t_start in range(0, n_frames, batch_size):
        t_end = min(t_start + batch_size, n_frames)
        src_start = start_frame + t_start
        src_end = start_frame + t_end
        if src_end > end_frame:
            raise DatasetDownloadError(
                "Emonet stream attempted to read past end_frame: "
                f"src_end={src_end} end_frame={end_frame}"
            )
        chunk = frames_dataset[src_start:src_end]
        chunk = np.transpose(chunk, (0, 2, 1)).astype(np.float32)
        if subtract_background and background is not None:
            chunk = chunk - background[None, :, :]
            chunk = np.clip(chunk, 0.0, None)
        conc[t_start:t_end] = chunk

        if normalize:
            global_min = min(global_min, float(chunk.min()))
            global_max = max(global_max, float(chunk.max()))

        if (t_start // batch_size) % 10 == 0:
            LOG.info("  Processed %d / %d frames...", t_end, n_frames)

    return global_min, global_max


def _write_emonet_attrs(
    root: Any,
    *,
    metadata: dict[str, Any],
    spec: EmonetSmokeIngest,
    n_frames: int,
    n_y: int,
    n_x: int,
    x_coords: Any,
    y_coords: Any,
    start_frame: int,
    end_frame: int,
    background_n_frames: int,
    trim_stats: dict[str, float],
    end_trim_stats: dict[str, float],
) -> None:
    fps = float(metadata.get("fps", spec.fps))

    x_vals = x_coords
    y_vals = y_coords
    try:
        x_min = float(min(x_vals))
        x_max = float(max(x_vals))
    except Exception:
        x_min = 0.0
        x_max = float(n_x)
    try:
        y_min = float(min(y_vals))
        y_max = float(max(y_vals))
    except Exception:
        y_min = 0.0
        y_max = float(n_y)

    pixel_to_grid_y = (y_max - y_min) / float(max(1, len(y_vals) - 1))
    pixel_to_grid_x = (x_max - x_min) / float(max(1, len(x_vals) - 1))

    root.attrs["schema_version"] = SCHEMA_VERSION
    root.attrs["fps"] = fps
    root.attrs["source_dtype"] = "float32"
    root.attrs["pixel_to_grid"] = [float(pixel_to_grid_y), float(pixel_to_grid_x)]
    root.attrs["origin"] = [float(y_min), float(x_min)]
    root.attrs["extent"] = [float(y_max - y_min), float(x_max - x_min)]
    root.attrs["dims"] = list(DIMS_TYX)

    root.attrs["n_frames"] = n_frames
    root.attrs["shape"] = [n_frames, n_y, n_x]
    root.attrs["arena_size_mm"] = list(spec.arena_size_mm)
    root.attrs["source_format"] = "emonet_flywalk"
    root.attrs["normalized"] = spec.normalize

    root.attrs["background_subtract"] = bool(
        getattr(spec, "background_subtract", False)
    )
    root.attrs["background_n_frames"] = int(background_n_frames)
    root.attrs["auto_trim_start"] = bool(getattr(spec, "auto_trim_start", False))
    root.attrs["skip_initial_frames"] = int(getattr(spec, "skip_initial_frames", 0))
    trim_abs_threshold = getattr(spec, "trim_abs_threshold", None)
    if trim_abs_threshold is not None:
        root.attrs["trim_abs_threshold"] = float(trim_abs_threshold)
    root.attrs["trim_sigma"] = float(getattr(spec, "trim_sigma", 0.0))
    root.attrs["trim_consecutive"] = int(getattr(spec, "trim_consecutive", 1))
    root.attrs["trim_max_scan"] = int(getattr(spec, "trim_max_scan", 0))
    root.attrs["start_frame"] = int(start_frame)
    for key, val in trim_stats.items():
        root.attrs[f"trim_{key}"] = float(val)

    root.attrs["auto_trim_end"] = bool(getattr(spec, "auto_trim_end", False))
    end_abs_threshold = getattr(spec, "end_abs_threshold", None)
    if end_abs_threshold is not None:
        root.attrs["end_abs_threshold"] = float(end_abs_threshold)
    root.attrs["end_sigma"] = float(getattr(spec, "end_sigma", 0.0))
    root.attrs["end_consecutive"] = int(getattr(spec, "end_consecutive", 1))
    root.attrs["end_frame"] = int(end_frame)
    for key, val in end_trim_stats.items():
        if key == "tail_len":
            root.attrs["end_tail_len"] = int(val)
        else:
            root.attrs[f"end_{key}"] = float(val)

    source_location_px = getattr(spec, "source_location_px", None)
    if source_location_px is None:
        source_location_px = (0, int(n_y) // 2)
        root.attrs["source_location_px_is_heuristic"] = True
        LOG.warning(
            "Emonet ingest missing source_location_px; defaulting to heuristic (x=%d, y=%d).",
            source_location_px[0],
            source_location_px[1],
        )
    root.attrs["source_location_px"] = [
        int(source_location_px[0]),
        int(source_location_px[1]),
    ]


def _ingest_emonet_to_zarr(
    spec: EmonetSmokeIngest,
    source_path: Path,
    output_path: Path,
    compute_stats: bool = True,
) -> Path:
    h5py, np, zarr = _require_ingest_deps("Emonet ingest")
    metadata = _load_emonet_metadata(spec, source_path, h5py=h5py, np=np)

    try:
        # Open HDF5 file and get shape without loading data
        with h5py.File(source_path, "r") as mat_file:
            frames_key = _resolve_frames_key(mat_file, spec, source_path)
            frames_dataset = mat_file[frames_key]
            src_shape = frames_dataset.shape
            LOG.info(
                "Found frames at key '%s' with shape %s, processing in chunks...",
                frames_key,
                src_shape,
            )

            # Emonet format per README: (Height x Width x n_frames) = (H, W, T)
            # But actual file has (T, X, Y) = (n_frames, Width, Height)
            # Arena: 300mm x 180mm, px_per_mm ~6.83
            # 300mm * 6.83 = 2048px (X/Width), 180mm * 6.83 ≈ 1200px (Y/Height)
            # We want (T, Y, X) -> swap axes 1 and 2 during chunk processing
            n_frames, n_x, n_y = src_shape

            background_n_frames = int(getattr(spec, "background_n_frames", 0))
            if background_n_frames < 0:
                raise DatasetDownloadError(
                    f"Invalid background_n_frames={background_n_frames} for {source_path}"
                )
            background_n_frames = min(background_n_frames, n_frames)

            background = None
            if bool(getattr(spec, "background_subtract", False)):
                if background_n_frames <= 0:
                    raise DatasetDownloadError(
                        "background_subtract=True requires background_n_frames > 0"
                    )
                LOG.info(
                    "Computing background image from first %d frames...",
                    background_n_frames,
                )
                background = _compute_emonet_background(
                    frames_dataset=frames_dataset,
                    n_bg=background_n_frames,
                    np=np,
                )

            trim_stats: dict[str, float] = {}
            start_frame = int(max(0, getattr(spec, "skip_initial_frames", 0)))
            if bool(getattr(spec, "auto_trim_start", False)):
                auto_start, trim_stats = _estimate_emonet_start_frame(
                    frames_dataset=frames_dataset,
                    n_frames=n_frames,
                    background=background,
                    spec=spec,
                    np=np,
                )
                start_frame = max(start_frame, int(auto_start))

            end_trim_stats: dict[str, float] = {}
            end_frame = n_frames
            if bool(getattr(spec, "auto_trim_end", False)):
                end_frame, end_trim_stats = _estimate_emonet_end_frame(
                    frames_dataset=frames_dataset,
                    n_frames=n_frames,
                    start_frame=start_frame,
                    background=background,
                    spec=spec,
                    np=np,
                )

            LOG.info(
                "Emonet preprocessing: background_subtract=%s background_n_frames=%d auto_trim_start=%s "
                "skip_initial_frames=%d -> start_frame=%d auto_trim_end=%s -> end_frame=%d",
                bool(getattr(spec, "background_subtract", False)),
                background_n_frames,
                bool(getattr(spec, "auto_trim_start", False)),
                int(getattr(spec, "skip_initial_frames", 0)),
                start_frame,
                bool(getattr(spec, "auto_trim_end", False)),
                int(end_frame),
            )
            if trim_stats:
                LOG.info(
                    "Emonet trim stats: baseline_mean=%.6f baseline_std=%.6f threshold=%.6f (sigma=%.2f, consecutive=%d, max_scan=%d)",
                    float(trim_stats.get("baseline_mean", 0.0)),
                    float(trim_stats.get("baseline_std", 0.0)),
                    float(trim_stats.get("threshold", 0.0)),
                    float(getattr(spec, "trim_sigma", 0.0)),
                    int(getattr(spec, "trim_consecutive", 1)),
                    int(getattr(spec, "trim_max_scan", 0)),
                )
            if end_trim_stats:
                LOG.info(
                    "Emonet end trim stats: tail_len=%d threshold=%.6f (sigma=%.2f, consecutive=%d)",
                    int(end_trim_stats.get("tail_len", 0.0)),
                    float(end_trim_stats.get("threshold", 0.0)),
                    float(getattr(spec, "end_sigma", 0.0)),
                    int(getattr(spec, "end_consecutive", 1)),
                )

            if start_frame >= n_frames:
                raise DatasetDownloadError(
                    f"Start frame {start_frame} is out of bounds for {source_path} with {n_frames} frames"
                )
            if end_frame > n_frames:
                raise DatasetDownloadError(
                    f"End frame {end_frame} is out of bounds for {source_path} with {n_frames} frames"
                )
            if end_frame <= start_frame:
                raise DatasetDownloadError(
                    f"End frame {end_frame} must be greater than start frame {start_frame}"
                )
            n_frames_out = end_frame - start_frame

            # Generate spatial coordinates from arena size
            arena_x, arena_y = spec.arena_size_mm
            x_coords = np.linspace(0, arena_x, n_x, dtype=np.float32)
            y_coords = np.linspace(0, arena_y, n_y, dtype=np.float32)

            # Create Zarr store with pre-allocated empty array
            # Use smaller time chunks to stay under 2GB codec limit
            # Each frame is n_y * n_x * 4 bytes = ~10 MB for Emonet
            bytes_per_frame = n_y * n_x * 4
            chunk_t = _compute_chunk_t(spec.chunk_t, bytes_per_frame)

            LOG.info(
                "Using chunks: (%d, %d, %d) for shape (%d, %d, %d), ~%.0f MB/chunk",
                chunk_t,
                n_y,
                n_x,
                n_frames,
                n_y,
                n_x,
                chunk_t * bytes_per_frame / 1e6,
            )

            root, conc = _init_emonet_store(
                output_path=output_path,
                n_frames=n_frames_out,
                n_y=n_y,
                n_x=n_x,
                chunk_t=chunk_t,
                x_coords=x_coords,
                y_coords=y_coords,
                zarr=zarr,
            )

            batch_size = chunk_t
            global_min, global_max = _stream_emonet_chunks(
                frames_dataset=frames_dataset,
                conc=conc,
                n_frames=n_frames_out,
                start_frame=start_frame,
                end_frame=end_frame,
                batch_size=batch_size,
                normalize=bool(spec.normalize),
                background=background,
                subtract_background=bool(getattr(spec, "background_subtract", False)),
                np=np,
            )

        # Normalize in a second pass if needed
        if spec.normalize and global_max > global_min:
            LOG.info("Normalizing data (min=%.2f, max=%.2f)...", global_min, global_max)
            _normalize_concentration(
                conc,
                n_frames=n_frames_out,
                chunk_t=batch_size,
                global_min=float(global_min),
                global_max=float(global_max),
            )

        # Store metadata
        _write_emonet_attrs(
            root,
            metadata=metadata,
            spec=spec,
            n_frames=n_frames_out,
            n_y=n_y,
            n_x=n_x,
            x_coords=x_coords,
            y_coords=y_coords,
            start_frame=start_frame,
            end_frame=end_frame,
            background_n_frames=background_n_frames,
            trim_stats=trim_stats,
            end_trim_stats=end_trim_stats,
        )

        LOG.info(
            "Ingested Emonet smoke video to Zarr: %s -> %s (shape=[%d, %d, %d])",
            source_path.name,
            output_path.name,
            n_frames_out,
            n_y,
            n_x,
        )

        # Compute and store concentration statistics
        # For large Emonet datasets, consider compute_stats=False and run separately
        if compute_stats:
            LOG.info("Computing concentration stats for %s...", output_path.name)
            stats = compute_concentration_stats(output_path)
            if spec.normalize:
                stats["normalized_during_ingest"] = True
                # Store original range for reference
                stats["original_min"] = (
                    float(global_min) if global_max > global_min else None
                )
                stats["original_max"] = (
                    float(global_max) if global_max > global_min else None
                )
            store_stats_in_zarr(output_path, stats)

        zarr.consolidate_metadata(output_path)

        return output_path

    except Exception as exc:
        if output_path.exists():
            shutil.rmtree(output_path)
        raise DatasetDownloadError(
            f"Failed to ingest Emonet video {source_path} into Zarr: {exc}"
        ) from exc

