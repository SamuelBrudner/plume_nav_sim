from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_ingest_cli_and_loader_smoke(tmp_path: Path) -> None:
    # Require xarray and core plume_nav_sim modules; imageio only needed if we synthesize frames
    try:
        import xarray as xr  # type: ignore

        from plume_nav_sim.cli import video_ingest
        from plume_nav_sim.media import (
            MANIFEST_FILENAME,
            get_default_manifest_path,
            load_manifest,
        )
        from plume_nav_sim.plume.movie_field import MovieConfig, MoviePlumeField
    except Exception as e:
        pytest.skip(f"required modules unavailable: {e}")

    # Prefer a stored dummy dataset if present in the repo
    tests_dir = Path(__file__).resolve().parents[1]
    repo_zarr = tests_dir / "data" / "video" / "smoke.zarr"
    if repo_zarr.exists():
        out_zarr = repo_zarr
    else:
        # 1) Create tiny grayscale frames (4 x 32 x 32)
        try:
            import imageio.v3 as iio  # type: ignore
        except Exception as e:
            pytest.skip(f"imageio required to synthesize frames: {e}")
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for t in range(4):
            img = np.full((32, 32), t * 60, dtype=np.uint8)
            iio.imwrite(frames_dir / f"{t:03d}.png", img)

        # 2) Ingest to Zarr via CLI (transient)
        out_zarr = tmp_path / "smoke.zarr"
        argv = [
            "--input",
            str(frames_dir),
            "--output",
            str(out_zarr),
            "--fps",
            "10",
            "--pixel-to-grid",
            "1 1",
            "--origin",
            "0 0",
            "--normalize",
        ]
        rc = int(video_ingest.main(argv))
        assert rc == 0, "video_ingest CLI returned non-zero exit code"
        assert out_zarr.is_dir(), "Zarr store not created"

    # 3) Inspect dataset with xarray
    ds = xr.open_zarr(str(out_zarr), consolidated=False)
    assert "concentration" in ds, "Missing 'concentration' variable"
    da = ds["concentration"]
    assert tuple(da.dims) == ("t", "y", "x"), f"Unexpected dims: {da.dims}"
    assert str(da.dtype) == "float32", f"Unexpected dtype: {da.dtype}"
    assert da.shape == (4, 32, 32)

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
    assert mpath.exists(), "Provenance manifest missing"
    m = load_manifest(out_zarr)
    assert hasattr(m, "source_dtype") and isinstance(getattr(m, "source_dtype"), str)

    # 4) Loader smoke: MoviePlumeField
    field = MoviePlumeField(MovieConfig(path=str(out_zarr), step_policy="wrap"))
    # Determinism check at a fixed step
    field.advance_to_step(3)
    a = field.field_array.copy()
    field.on_reset()
    field.advance_to_step(3)
    b = field.field_array.copy()
    assert np.array_equal(a, b), "MoviePlumeField not deterministic at fixed step"

    # Frame type/shape sanity
    assert a.dtype == np.float32
    assert a.shape == (32, 32)
