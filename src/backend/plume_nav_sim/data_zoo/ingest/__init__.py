from __future__ import annotations

from typing import Any

from ..downloader import DatasetDownloadError


def _require_ingest_deps(label: str) -> tuple[Any, Any, Any]:
    try:
        import h5py
        import numpy as np
        import zarr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DatasetDownloadError(
            f"{label} requires optional dependencies (h5py, zarr, numpy). "
            "Install media extras or provide a pre-ingested dataset."
        ) from exc
    return h5py, np, zarr


def _compute_chunk_t(spec_chunk: int | None, bytes_per_frame: int) -> int:
    max_chunk_bytes = 500 * 1024 * 1024  # 500MB target
    return min(spec_chunk or 100, max(1, max_chunk_bytes // max(1, bytes_per_frame)))


def _normalize_concentration(
    conc: Any,
    *,
    n_frames: int,
    chunk_t: int,
    global_min: float,
    global_max: float,
) -> None:
    for t_start in range(0, n_frames, chunk_t):
        t_end = min(t_start + chunk_t, n_frames)
        chunk = conc[t_start:t_end]
        conc[t_start:t_end] = (chunk - global_min) / (global_max - global_min)

