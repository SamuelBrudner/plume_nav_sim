from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_h5_movie_ingest_and_loader_smoke(tmp_path: Path) -> None:
    """Smoke test for HDF5 movie ingest -> Zarr -> VideoPlume loading.

    This mirrors the existing video ingest smoke test but uses a synthetic
    HDF5 movie source and the dedicated ingest_h5_movie helper.
    """

    try:
        import h5py  # type: ignore
        import xarray as xr  # type: ignore

        from plume_nav_sim.media import (
            MANIFEST_FILENAME,
            get_default_manifest_path,
            load_manifest,
        )
        from plume_nav_sim.media.h5_movie import H5MovieIngestConfig, ingest_h5_movie
        from plume_nav_sim.plume.video import VideoConfig, VideoPlume
    except Exception as e:  # pragma: no cover - optional media deps
        pytest.skip(f"required modules unavailable: {e}")

    # 1) Create a tiny synthetic HDF5 movie dataset: (T, Y, X) float32
    h5_path = tmp_path / "movie.h5"
    dataset_path = "Plume Data/dataset_001"

    with h5py.File(h5_path, "w") as f:
        # Minimal imagingParameters/frameRate metadata so ingest_h5_movie can infer fps
        attrs_grp = f.create_group("Attributes")
        imaging = attrs_grp.create_group("imagingParameters")
        imaging.create_dataset("frameRate", data=[[10.0]])

        plume_grp = f.create_group("Plume Data")
        data = np.linspace(0.0, 1.0, 4 * 16 * 8, dtype=np.float32).reshape(4, 16, 8)
        plume_grp.create_dataset("dataset_001", data=data)

    # 2) Ingest HDF5 movie into Zarr via helper
    out_zarr = tmp_path / "movie.zarr"
    cfg = H5MovieIngestConfig(
        input=h5_path,
        dataset=dataset_path,
        output=out_zarr,
    )
    dest = ingest_h5_movie(cfg)
    assert dest == out_zarr
    assert out_zarr.is_dir(), "Zarr store not created from HDF5 movie"

    # 3) Inspect dataset with xarray and validate basic schema
    ds = xr.open_zarr(str(out_zarr), consolidated=False)
    assert (
        "concentration" in ds
    ), "Missing 'concentration' variable in HDF5-ingested dataset"
    da = ds["concentration"]
    assert tuple(da.dims) == ("t", "y", "x"), f"Unexpected dims: {da.dims}"
    assert str(da.dtype) == "float32", f"Unexpected dtype: {da.dtype}"
    assert da.shape == (4, 16, 8)

    # Required attrs present on dataset or variable
    required_keys = [
        "fps",
        "pixel_to_grid",
        "origin",
        "extent",
        "schema_version",
        "source_dtype",
    ]
    attrs = {**(ds.attrs or {}), **(da.attrs or {})}
    for k in required_keys:
        assert k in attrs, f"Missing required attr: {k}"

    # 3b) Provenance manifest present and parseable
    mpath = get_default_manifest_path(out_zarr)
    assert mpath.name == MANIFEST_FILENAME
    assert mpath.exists(), "Provenance manifest missing for HDF5-ingested dataset"
    m = load_manifest(out_zarr)
    assert hasattr(m, "source_dtype") and isinstance(getattr(m, "source_dtype"), str)

    # 4) Loader smoke: VideoPlume can consume the ingested dataset
    field = VideoPlume(VideoConfig(path=str(out_zarr), step_policy="wrap"))

    # Determinism check at a fixed step
    field.advance_to_step(3)
    a = field.field_array.copy()
    field.on_reset()
    field.advance_to_step(3)
    b = field.field_array.copy()
    assert np.array_equal(
        a, b
    ), "VideoPlume not deterministic at fixed step for HDF5 source"

    # Frame type/shape sanity
    assert a.dtype == np.float32
    assert a.shape == (16, 8)
