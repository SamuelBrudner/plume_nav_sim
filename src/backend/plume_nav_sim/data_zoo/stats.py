from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np

LOG = logging.getLogger(__name__)


QuantileStats = dict[str, float]
ConcentrationStats = dict[str, object]


def compute_concentration_stats(
    zarr_path: Path,
    sample_frames: Optional[int] = None,
    chunk_size: int = 100,
) -> ConcentrationStats:
    import zarr

    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.open_group(store, mode="r")
    conc = root["concentration"]

    n_frames, n_y, n_x = conc.shape
    LOG.info("Computing stats for %s (shape=%s)", zarr_path.name, conc.shape)

    # First pass: compute min, max, mean, nonzero count
    running_sum = 0.0
    running_sq_sum = 0.0
    running_min = np.inf
    running_max = -np.inf
    nonzero_count = 0
    total_count = 0

    for t_start in range(0, n_frames, chunk_size):
        t_end = min(t_start + chunk_size, n_frames)
        chunk = conc[t_start:t_end]

        running_sum += chunk.sum()
        running_sq_sum += (chunk**2).sum()
        running_min = min(running_min, chunk.min())
        running_max = max(running_max, chunk.max())
        nonzero_count += np.count_nonzero(chunk)
        total_count += chunk.size

        if t_start % (chunk_size * 10) == 0:
            LOG.info("  Pass 1: %d / %d frames...", t_end, n_frames)

    mean = running_sum / total_count
    # Var = E[X^2] - E[X]^2
    variance = (running_sq_sum / total_count) - (mean**2)
    std = np.sqrt(max(0, variance))  # Guard against numerical issues
    nonzero_fraction = nonzero_count / total_count

    LOG.info(
        "  min=%.4f, max=%.4f, mean=%.4f, std=%.4f", running_min, running_max, mean, std
    )

    # Second pass: compute quantiles
    # For large datasets, sample frames to estimate quantiles
    if sample_frames is not None and sample_frames < n_frames:
        frame_indices = np.random.default_rng(42).choice(
            n_frames, size=sample_frames, replace=False
        )
        frame_indices = np.sort(frame_indices)
        LOG.info("  Sampling %d frames for quantile estimation", sample_frames)
    else:
        frame_indices = np.arange(n_frames)

    # Collect samples for quantile computation
    # Use reservoir sampling approach for memory efficiency
    max_samples = 10_000_000  # ~40MB for float32
    samples_per_frame = max(1, max_samples // len(frame_indices))

    all_samples = []
    for i, t in enumerate(frame_indices):
        frame = conc[t]
        # Random sample from frame
        flat = frame.flatten()
        if len(flat) > samples_per_frame:
            idx = np.random.default_rng(42 + i).choice(
                len(flat), size=samples_per_frame, replace=False
            )
            all_samples.append(flat[idx])
        else:
            all_samples.append(flat)

        if i % 100 == 0 and i > 0:
            LOG.info("  Pass 2: %d / %d frames...", i, len(frame_indices))

    samples = np.concatenate(all_samples)
    LOG.info("  Computing quantiles from %d samples", len(samples))

    quantiles: QuantileStats = {
        "q01": float(np.percentile(samples, 1)),
        "q05": float(np.percentile(samples, 5)),
        "q25": float(np.percentile(samples, 25)),
        "q50": float(np.percentile(samples, 50)),
        "q75": float(np.percentile(samples, 75)),
        "q95": float(np.percentile(samples, 95)),
        "q99": float(np.percentile(samples, 99)),
        "q999": float(np.percentile(samples, 99.9)),
    }

    return {
        "min": float(running_min),
        "max": float(running_max),
        "mean": float(mean),
        "std": float(std),
        "quantiles": quantiles,
        "nonzero_fraction": float(nonzero_fraction),
        "original_min": None,
        "original_max": None,
        "normalized_during_ingest": False,
    }


def store_stats_in_zarr(zarr_path: Path, stats: ConcentrationStats) -> None:
    """Store concentration stats in zarr attrs."""
    import zarr

    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.open_group(store, mode="r+")

    # Store as nested dict in attrs
    root.attrs["concentration_stats"] = dict(stats)
    LOG.info("Stored concentration stats in %s", zarr_path.name)


def load_stats_from_zarr(zarr_path: Path) -> Optional[ConcentrationStats]:
    """Load concentration stats from zarr attrs."""
    import zarr

    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.open_group(store, mode="r")

    stats_dict = root.attrs.get("concentration_stats")
    if stats_dict is None:
        return None

    return dict(stats_dict)


NormalizationMethod = Literal["minmax", "robust", "zscore", None]


def normalize_array(
    data: np.ndarray,
    stats: ConcentrationStats,
    method: NormalizationMethod = "minmax",
) -> np.ndarray:
    if method is None:
        return data

    if method == "minmax":
        dmin, dmax = stats["min"], stats["max"]
        if dmax > dmin:
            return (data - dmin) / (dmax - dmin)
        return data - dmin  # Constant data

    if method == "robust":
        q05 = stats["quantiles"]["q05"]
        q95 = stats["quantiles"]["q95"]
        if q95 > q05:
            normalized = (data - q05) / (q95 - q05)
            return np.clip(normalized, 0, 1)
        return np.zeros_like(data)

    if method == "zscore":
        if stats["std"] > 0:
            return (data - stats["mean"]) / stats["std"]
        return data - stats["mean"]

    raise ValueError(f"Unknown normalization method: {method}")
