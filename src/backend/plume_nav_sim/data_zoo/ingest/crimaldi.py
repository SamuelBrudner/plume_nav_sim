from __future__ import annotations

import logging
from pathlib import Path

from ..downloader import DatasetDownloadError
from ..registry import CrimaldiFluorescenceIngest
from ..stats import compute_concentration_stats, store_stats_in_zarr

LOG = logging.getLogger("plume_nav_sim.data_zoo.download")


def _ingest_hdf5_to_zarr(
    spec: CrimaldiFluorescenceIngest,
    source_path: Path,
    output_path: Path,
    compute_stats: bool = True,
) -> Path:
    try:
        from plume_nav_sim.media.h5_movie import H5MovieIngestConfig, ingest_h5_movie
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DatasetDownloadError(
            "HDF5 ingest requires optional dependencies (h5py, zarr, numcodecs). "
            "Install media extras or provide a pre-ingested dataset."
        ) from exc

    cfg = H5MovieIngestConfig(
        input=source_path,
        dataset=spec.dataset,
        output=output_path,
        t_start=0,
        t_stop=None,
        fps=spec.fps,
        pixel_to_grid=spec.pixel_to_grid,
        origin=spec.origin,
        extent=spec.extent,
        source_location_px=spec.source_location_px,
        normalize=spec.normalize,
        chunk_t=spec.chunk_t,
    )

    try:
        result_path = ingest_h5_movie(cfg)
        if compute_stats:
            LOG.info("Computing concentration stats for %s...", output_path.name)
            stats = compute_concentration_stats(result_path)
            if spec.normalize:
                stats["normalized_during_ingest"] = True
            store_stats_in_zarr(result_path, stats)
        return result_path
    except Exception as exc:  # pragma: no cover - ingestion errors bubble up
        raise DatasetDownloadError(
            f"Failed to ingest HDF5 movie {source_path} into Zarr: {exc}"
        ) from exc

