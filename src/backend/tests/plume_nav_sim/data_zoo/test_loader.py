from pathlib import Path

import numpy as np
import pytest

from plume_nav_sim.data_zoo.loader import load_plume
from plume_nav_sim.data_zoo.registry import (
    DATASET_REGISTRY,
    DatasetArtifact,
    DatasetMetadata,
    DatasetRegistryEntry,
)


def _seed_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_id: str,
    with_stats: bool = True,
):
    pytest.importorskip("xarray")
    zarr = pytest.importorskip("zarr")

    cache_root = tmp_path / "cache"
    expected_root = cache_root / dataset_id / "1.0.0" / "demo.zarr"
    expected_root.mkdir(parents=True)

    data = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    store = zarr.DirectoryStore(str(expected_root))
    root = zarr.group(store, overwrite=True)
    conc = root.create_dataset("concentration", data=data, chunks=(1, 2, 2))
    conc.attrs["_ARRAY_DIMENSIONS"] = ["t", "y", "x"]
    root.create_dataset("x", data=np.array([0.0, 1.0], dtype=np.float32))
    root.create_dataset("y", data=np.array([0.0, 1.0], dtype=np.float32))

    stats = None
    if with_stats:
        flat = data.flatten()
        stats = {
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "quantiles": {
                "q01": float(np.percentile(flat, 1)),
                "q05": float(np.percentile(flat, 5)),
                "q25": float(np.percentile(flat, 25)),
                "q50": float(np.percentile(flat, 50)),
                "q75": float(np.percentile(flat, 75)),
                "q95": float(np.percentile(flat, 95)),
                "q99": float(np.percentile(flat, 99)),
            },
            "nonzero_fraction": float(np.count_nonzero(flat) / flat.size),
            "original_min": None,
            "original_max": None,
            "normalized_during_ingest": False,
        }
        root.attrs["concentration_stats"] = stats

    # Minimal attrs to mirror ingests
    root.attrs.update(
        {
            "fps": 15.0,
            "source_dtype": "float32",
            "pixel_to_grid": [1.0, 1.0],
            "origin": [0.0, 0.0],
            "extent": [1.0, 1.0],
            "dims": ["t", "y", "x"],
        }
    )

    entry = DatasetRegistryEntry(
        dataset_id=dataset_id,
        version="1.0.0",
        cache_subdir=dataset_id,
        expected_root="demo.zarr",
        artifact=DatasetArtifact(
            url="file:///demo.zarr",
            checksum="unused",
            archive_type="none",
            layout="zarr",
        ),
        metadata=DatasetMetadata(
            title="Demo dataset",
            description="Loader test dataset",
            citation="N/A",
            license="MIT",
        ),
    )
    monkeypatch.setitem(DATASET_REGISTRY, dataset_id, entry)
    return cache_root, entry, data, stats


def test_load_plume_normalizes_and_attaches_coords(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("dask.array")
    cache_root, entry, data, stats = _seed_dataset(
        tmp_path, monkeypatch, "demo_stats", with_stats=True
    )

    arr = load_plume(entry.dataset_id, cache_root=cache_root, normalize="minmax")

    assert arr.dims == ("t", "y", "x")
    assert arr.name == "concentration"
    assert getattr(arr.data, "chunks", None) is not None
    assert "x" in arr.coords and "y" in arr.coords
    assert arr.attrs["normalized"] == "minmax"
    assert "concentration_stats" in arr.attrs
    np.testing.assert_allclose(arr.compute().values, data / float(stats["max"]))  # type: ignore[index]


def test_load_plume_requires_stats_for_normalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache_root, entry, _, _ = _seed_dataset(
        tmp_path, monkeypatch, "demo_no_stats", with_stats=False
    )

    with pytest.raises(ValueError, match="concentration_stats"):
        load_plume(entry.dataset_id, cache_root=cache_root, normalize="zscore")
