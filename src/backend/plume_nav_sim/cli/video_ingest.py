from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def _try_import_imageio_v3():
    try:
        import imageio.v3 as iio  # type: ignore

        return iio
    except Exception as e:  # pragma: no cover - exercised in media-optional envs
        raise ImportError(
            "imageio is required for video ingest. Install with 'pip install imageio'."
        ) from e


def _try_import_zarr():
    try:
        import zarr  # type: ignore

        return zarr
    except Exception as e:  # pragma: no cover - exercised in media-optional envs
        raise ImportError(
            "zarr is required for video ingest. Install with 'pip install zarr numcodecs'."
        ) from e


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _as_2tuple(value: Sequence[float] | float) -> Tuple[float, float]:
    if isinstance(value, (int, float)):
        v = float(value)
        return (v, v)
    seq = list(value)  # type: ignore[arg-type]
    if len(seq) != 2:
        raise ValueError("Expected 2 values")
    return (float(seq[0]), float(seq[1]))


def _parse_yx_pair(
    arg: Optional[str], *, default: Tuple[float, float]
) -> Tuple[float, float]:
    if arg is None:
        return default
    parts = [p for p in arg.replace(",", " ").split() if p]
    if len(parts) == 1:
        v = float(parts[0])
        return (v, v)
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f"Expected 1 or 2 numbers, got: {arg}")


def _rgb_to_luma_u8(rgb: np.ndarray) -> np.ndarray:
    # Expect (H, W, 3); compute ITU-R BT.601 luma, return uint8
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    y = np.clip(y, 0, 255)
    return y.astype(np.uint8)


def _iter_frames_from_dir(iio, directory: Path) -> Iterator[np.ndarray]:
    # Accept common image extensions
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = sorted([p for p in directory.iterdir() if p.suffix.lower() in exts])
    for fp in files:
        arr = iio.imread(fp)
        yield arr


def _iter_frames(iio, input_path: Path) -> Iterator[np.ndarray]:
    if input_path.is_dir():
        yield from _iter_frames_from_dir(iio, input_path)
        return
    # Treat as single media file readable by imageio
    try:
        # Prefer iterator to avoid loading entire file
        yield from iio.imiter(str(input_path))
    except Exception:
        # Fallback: read into array, then split on first axis if multi-frame
        arr = iio.imread(str(input_path))
        if arr.ndim >= 4:  # e.g., (T, H, W, C)
            for f in arr:
                yield f
        else:
            yield arr


def _stack_to_concentration(
    frames: Iterable[np.ndarray], *, normalize: bool
) -> Tuple[np.ndarray, str]:
    """Convert input frames to a (T,Y,X) float32 concentration array.

    Returns (data, source_dtype_str).
    - Accepts grayscale (H,W) or RGB (H,W,3) inputs; RGB is converted to luma.
    - When normalize=True, output is in [0,1] float32; else float32 preserving range.
    """
    frames_list: List[np.ndarray] = []
    src_dtype: Optional[np.dtype] = None
    for f in frames:
        if not isinstance(f, np.ndarray):
            raise TypeError("Frame is not a numpy array")
        if f.ndim == 3 and f.shape[-1] == 3:
            f2 = _rgb_to_luma_u8(f)
        elif f.ndim == 2:
            f2 = f
        else:
            raise ValueError(f"Unsupported frame shape: {f.shape}")
        if src_dtype is None:
            src_dtype = f2.dtype
        frames_list.append(f2)

    if not frames_list:
        raise ValueError("No frames to ingest")

    stack = np.stack(frames_list, axis=0)  # (T, Y, X)

    # Normalize/cast
    if normalize:
        # Map integer inputs to [0,1]; pass-through float32 treated as already scaled
        if np.issubdtype(stack.dtype, np.integer):
            maxv = np.iinfo(stack.dtype).max
            data = (stack.astype(np.float32) / float(maxv)).astype(np.float32)
        else:
            data = stack.astype(np.float32)
        src_dtype_str = str(
            src_dtype.name if isinstance(src_dtype, np.dtype) else src_dtype
        )
    else:
        data = stack.astype(np.float32)
        src_dtype_str = str(
            src_dtype.name if isinstance(src_dtype, np.dtype) else src_dtype
        )

    return data, src_dtype_str


@dataclass
class IngestConfig:
    input: Path
    output: Path
    fps: float
    pixel_to_grid: Tuple[float, float]
    origin: Tuple[float, float]
    extent: Optional[Tuple[float, float]]
    normalize: bool


def _resolve_extent(
    h: int,
    w: int,
    pixel_to_grid: Tuple[float, float],
    override: Optional[Tuple[float, float]],
):
    if override is not None:
        return override
    y = h * pixel_to_grid[0]
    x = w * pixel_to_grid[1]
    return (float(y), float(x))


def ingest_video(cfg: IngestConfig) -> Path:
    """Run ingest: read frames, write Zarr store with attrs and manifest."""
    # Deferred imports to enable clean skips
    iio = _try_import_imageio_v3()
    _ = _try_import_zarr()

    # Contracts and helpers
    from plume_nav_sim.media import build_provenance_manifest, write_manifest
    from plume_nav_sim.storage import CHUNKS_TYX, create_zarr_array
    from plume_nav_sim.video.schema import (
        DIMS_TYX,
        SCHEMA_VERSION,
        VARIABLE_NAME,
        VideoPlumeAttrs,
        validate_attrs,
    )

    # Read frames and convert
    frames = list(_iter_frames(iio, cfg.input))
    data, source_dtype_str = _stack_to_concentration(frames, normalize=cfg.normalize)
    t, y, x = data.shape

    # Prepare attrs and validate
    extent = _resolve_extent(y, x, cfg.pixel_to_grid, cfg.extent)
    attrs = VideoPlumeAttrs(
        fps=float(cfg.fps),
        source_dtype=source_dtype_str,
        pixel_to_grid=cfg.pixel_to_grid,
        origin=cfg.origin,
        extent=extent,
    )
    # Round-trip through validator to ensure compliance
    valid = validate_attrs(attrs.model_dump())

    # Create dataset and write
    out = cfg.output
    out.parent.mkdir(parents=True, exist_ok=True)

    arr = create_zarr_array(
        out,
        name=VARIABLE_NAME,
        shape=(t, y, x),
        dtype="float32",
        chunks=CHUNKS_TYX,
        overwrite=True,
        extra_attrs={"_ARRAY_DIMENSIONS": list(DIMS_TYX)},
    )
    # Write in chunked slices along time to keep memory modest
    arr[:] = data  # small inputs; for very large, loop over t

    # Write group/global attrs
    import zarr as _zarr  # type: ignore

    grp = _zarr.open_group(out, mode="a")
    for k, v in valid.model_dump().items():
        grp.attrs[k] = v
    # Also record CLI args for transparency
    grp.attrs["ingest_args"] = {
        "fps": cfg.fps,
        "pixel_to_grid": list(cfg.pixel_to_grid),
        "origin": list(cfg.origin),
        "extent": list(extent),
        "normalize": cfg.normalize,
    }

    # Build and write provenance manifest
    manifest = build_provenance_manifest(
        source_dtype=source_dtype_str,
        cli_args=list(shlex.split(" ".join(os.sys.argv))) if os.sys.argv else None,
        git_sha=_git_sha(),
    )
    write_manifest(out, manifest)

    _zarr.consolidate_metadata(out)

    return out


def _setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("plume_nav_sim.video_ingest")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(levelname)s %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest video/images into a Zarr plume dataset"
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input media path (mp4/gif or directory of images)",
    )
    p.add_argument("--output", required=True, help="Output Zarr store path")
    p.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second if not present in metadata",
    )
    p.add_argument(
        "--pixel-to-grid",
        dest="pixel_to_grid",
        type=str,
        default=None,
        help="Scale from pixels to grid units for y,x (e.g., '1.0 1.0')",
    )
    p.add_argument(
        "--origin",
        type=str,
        default=None,
        help="Grid origin y,x (e.g., '0 0')",
    )
    p.add_argument(
        "--extent",
        type=str,
        default=None,
        help="Grid extent y,x; defaults to pixel dims * pixel_to_grid",
    )
    p.add_argument(
        "--normalize", action="store_true", help="Normalize inputs to [0,1] float32"
    )
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logger = _setup_logger(args.log_level)

    # Graceful skip if media deps are unavailable
    try:
        _ = _try_import_imageio_v3()
        _ = _try_import_zarr()
    except ImportError as e:
        logger.info("media deps unavailable; skipping ingest: %s", e)
        return 0

    input_path = Path(args.input)
    output_path = Path(args.output)

    pix2grid = _parse_yx_pair(args.pixel_to_grid, default=(1.0, 1.0))
    origin = _parse_yx_pair(args.origin, default=(0.0, 0.0))
    extent = _parse_yx_pair(args.extent, default=None) if args.extent else None

    cfg = IngestConfig(
        input=input_path,
        output=output_path,
        fps=float(args.fps),
        pixel_to_grid=pix2grid,
        origin=origin,
        extent=extent,
        normalize=bool(args.normalize),
    )

    try:
        dest = ingest_video(cfg)
    except Exception as e:
        logger.error("ingest failed: %s", e)
        return 2

    logger.info("wrote Zarr dataset to %s", dest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
