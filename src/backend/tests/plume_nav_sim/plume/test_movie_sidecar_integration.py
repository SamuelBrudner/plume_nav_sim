from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_sidecar_yaml(sidecar_path: Path, data: dict) -> None:
    yaml = pytest.importorskip("yaml")
    sidecar_path.write_text(yaml.safe_dump(data))


def _read_video_attrs(dataset_root: Path) -> tuple[dict, tuple[int, int, int]]:
    """Open a Zarr dataset and return merged attrs plus shape.

    This helper mirrors the pattern used in existing video smoke tests: attrs
    may live on the dataset or the primary data variable.
    """

    try:
        import xarray as xr  # type: ignore
    except Exception as e:  # pragma: no cover - optional media deps
        pytest.skip(f"required modules unavailable: {e}")

    ds = xr.open_zarr(str(dataset_root), consolidated=False)
    if "concentration" not in ds:
        raise AssertionError("Missing 'concentration' variable in movie dataset")
    da = ds["concentration"]
    attrs = {**(ds.attrs or {}), **(da.attrs or {})}
    shape = tuple(int(s) for s in da.shape)
    return attrs, shape  # type: ignore[return-value]


def test_resolve_movie_dataset_path_video_sidecar_integration(tmp_path: Path) -> None:
    """Video+sidecar -> resolve_movie_dataset_path -> Zarr attrs match sidecar.

    This exercises the non-HDF5 ingest path driven by MovieMetadataSidecar and
    ensures fps and spatial calibration (pixel_to_grid, origin, extent) are
    derived from the sidecar rather than ad-hoc overrides.
    """

    try:
        import imageio.v3 as iio  # type: ignore

        from plume_nav_sim.media.sidecar import get_default_sidecar_path
        from plume_nav_sim.plume.video import resolve_movie_dataset_path
    except Exception as e:  # pragma: no cover - optional media deps
        pytest.skip(f"required modules unavailable: {e}")

    # 1) Synthesize a tiny single-frame TIFF "video".
    movie_path = tmp_path / "sidecar_movie.tiff"
    frame = np.full((8, 12), 42, dtype=np.uint8)
    iio.imwrite(movie_path, frame)

    # 2) Write a matching sidecar declaring non-pixel spatial units so that
    # pixel_to_grid must be derived from pixels_per_unit.
    sidecar_path = get_default_sidecar_path(movie_path)
    _write_sidecar_yaml(
        sidecar_path,
        {
            "version": 1,
            "path": "sidecar_movie.tiff",
            "fps": 12.5,
            "spatial_unit": "mm",
            "pixels_per_unit": [2.0, 4.0],
        },
    )

    # 3) Resolve dataset path; this should ingest via video_ingest.ingest_video
    # using sidecar-derived fps and spatial calibration.
    dataset_root = resolve_movie_dataset_path(movie_path, normalize=True)
    assert dataset_root.is_dir(), "Expected Zarr dataset directory from video ingest"

    attrs, shape = _read_video_attrs(dataset_root)
    t, h, w = shape
    assert (t, h, w) == (1, 8, 12)

    # Sidecar declares pixels_per_unit=(py, px)=(2.0, 4.0); pixel_to_grid must
    # be (1/py, 1/px).
    expected_fps = 12.5
    expected_pixel_to_grid = (0.5, 0.25)
    expected_origin = (0.0, 0.0)
    expected_extent = (h * expected_pixel_to_grid[0], w * expected_pixel_to_grid[1])

    assert float(attrs["fps"]) == pytest.approx(expected_fps)
    assert tuple(attrs["pixel_to_grid"]) == pytest.approx(expected_pixel_to_grid)
    assert tuple(attrs["origin"]) == pytest.approx(expected_origin)
    assert tuple(attrs["extent"]) == pytest.approx(expected_extent)


def test_resolve_movie_dataset_path_hdf5_sidecar_integration(tmp_path: Path) -> None:
    """HDF5+sidecar -> resolve_movie_dataset_path -> Zarr attrs match sidecar.

    This exercises the HDF5 ingest path driven by MovieMetadataSidecar and
    verifies that fps, spatial calibration, and dataset path are taken from the
    sidecar (with movie_h5_dataset validated for equality when provided).
    """

    try:
        import h5py  # type: ignore

        # Hard skip when optional media deps missing
        pytest.importorskip("zarr")
        pytest.importorskip("numcodecs")

        from plume_nav_sim.media import (
            MANIFEST_FILENAME,
            get_default_manifest_path,
            load_manifest,
        )
        from plume_nav_sim.media.sidecar import get_default_sidecar_path
        from plume_nav_sim.plume.video import resolve_movie_dataset_path
    except Exception as e:  # pragma: no cover - optional media deps
        pytest.skip(f"required modules unavailable: {e}")

    h5_path = tmp_path / "movie.h5"
    dataset_name = "Plume Data/dataset_001"

    # 1) Create a minimal 3D HDF5 movie dataset (T, Y, X) with optional
    # imagingParameters/frameRate group present for backward compatibility.
    with h5py.File(h5_path, "w") as f:
        attrs_grp = f.create_group("Attributes")
        imaging = attrs_grp.create_group("imagingParameters")
        imaging.create_dataset("frameRate", data=[[30.0]])

        plume_grp = f.create_group("Plume Data")
        data = np.linspace(0.0, 1.0, 4 * 10 * 6, dtype=np.float32).reshape(4, 10, 6)
        plume_grp.create_dataset("dataset_001", data=data)

    # 2) Write a sidecar specifying fps, spatial calibration, and h5_dataset.
    sidecar_path = get_default_sidecar_path(h5_path)
    _write_sidecar_yaml(
        sidecar_path,
        {
            "version": 1,
            "path": "movie.h5",
            "fps": 7.5,
            "spatial_unit": "mm",
            "pixels_per_unit": [2.0, 3.0],
            "h5_dataset": "Plume Data/dataset_001",
        },
    )

    # 3) Resolve dataset path via the sidecar-driven helper. Providing
    # movie_h5_dataset is optional but must agree with the sidecar if set.
    dataset_root = resolve_movie_dataset_path(
        h5_path,
        normalize=True,
        movie_h5_dataset=dataset_name,
    )
    assert dataset_root.is_dir(), "Expected Zarr dataset directory from HDF5 ingest"

    attrs, shape = _read_video_attrs(dataset_root)
    t, h, w = shape
    assert (t, h, w) == (4, 10, 6)

    expected_fps = 7.5
    expected_pixel_to_grid = (0.5, 1.0 / 3.0)
    expected_origin = (0.0, 0.0)
    expected_extent = (h * expected_pixel_to_grid[0], w * expected_pixel_to_grid[1])

    assert float(attrs["fps"]) == pytest.approx(expected_fps)
    assert tuple(attrs["pixel_to_grid"]) == pytest.approx(expected_pixel_to_grid)
    assert tuple(attrs["origin"]) == pytest.approx(expected_origin)
    assert tuple(attrs["extent"]) == pytest.approx(expected_extent)

    # Provenance manifest should exist and record ingest arguments, including
    # the dataset path that was taken from the sidecar.
    mpath = get_default_manifest_path(dataset_root)
    assert mpath.name == MANIFEST_FILENAME
    assert mpath.exists(), "Provenance manifest missing for HDF5 sidecar ingest"
    manifest = load_manifest(dataset_root)
    ingest_args = getattr(manifest, "ingest_args", None)
    if isinstance(ingest_args, dict):
        assert ingest_args.get("dataset") == dataset_name


def test_resolve_movie_dataset_path_conflicting_overrides_raise_validation_error(
    tmp_path: Path,
) -> None:
    """Conflicting overrides (e.g., movie_fps) raise ValidationError.

    This test exercises the guard rails that ensure env/CLI-level overrides
    cannot silently diverge from the sidecar, keeping the sidecar as the single
    source of truth for movie-level metadata for raw media sources.
    """

    from plume_nav_sim._compat import ValidationError
    from plume_nav_sim.media.sidecar import get_default_sidecar_path
    from plume_nav_sim.plume.video import resolve_movie_dataset_path

    movie_path = tmp_path / "conflict.avi"
    movie_path.write_bytes(b"")

    sidecar_path = get_default_sidecar_path(movie_path)
    _write_sidecar_yaml(
        sidecar_path,
        {
            "version": 1,
            "path": "conflict.avi",
            "fps": 20.0,
            "spatial_unit": "pixel",
        },
    )

    with pytest.raises(ValidationError, match="movie_fps must match sidecar.fps"):
        resolve_movie_dataset_path(movie_path, fps=15.0)
