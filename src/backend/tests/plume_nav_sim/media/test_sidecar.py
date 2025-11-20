from __future__ import annotations

from pathlib import Path

import pytest

from plume_nav_sim.media import (
    MovieMetadataSidecar,
    get_default_sidecar_path,
    load_movie_sidecar,
)


def _write_sidecar_yaml(sidecar_path: Path, data: dict) -> None:
    yaml = pytest.importorskip("yaml")
    sidecar_path.write_text(yaml.safe_dump(data))


def test_movie_metadata_sidecar_pixel_unit_forbids_pixels_per_unit():
    sidecar = MovieMetadataSidecar(
        version=1,
        path="movie.avi",
        fps=30.0,
        spatial_unit="pixel",
    )
    assert sidecar.spatial_unit == "pixel"
    assert sidecar.pixels_per_unit is None

    with pytest.raises(Exception):
        MovieMetadataSidecar(
            version=1,
            path="movie.avi",
            fps=30.0,
            spatial_unit="pixel",
            pixels_per_unit=(1.0, 1.0),
        )


def test_movie_metadata_sidecar_requires_pixels_per_unit_for_physical_units():
    with pytest.raises(Exception):
        MovieMetadataSidecar(
            version=1,
            path="movie_mm.avi",
            fps=15.0,
            spatial_unit="mm",
        )

    s = MovieMetadataSidecar(
        version=1,
        path="movie_mm.avi",
        fps=15.0,
        spatial_unit="mm",
        pixels_per_unit=(1.3513513514, 1.3513513514),
    )
    assert s.spatial_unit == "mm"
    assert s.pixels_per_unit == (1.3513513514, 1.3513513514)


def test_get_default_sidecar_path_suffix(tmp_path: Path):
    movie_path = tmp_path / "demo_10s.h5"
    sidecar_path = get_default_sidecar_path(movie_path)
    assert sidecar_path.name == "demo_10s.h5.plume-movie.yaml"


def test_load_movie_sidecar_for_non_hdf5(tmp_path: Path):
    movie_path = tmp_path / "gaussian_plume_demo.avi"
    movie_path.write_bytes(b"")

    sidecar_path = get_default_sidecar_path(movie_path)
    _write_sidecar_yaml(
        sidecar_path,
        {
            "version": 1,
            "path": "gaussian_plume_demo.avi",
            "fps": 30.0,
            "spatial_unit": "pixel",
        },
    )

    sidecar = load_movie_sidecar(movie_path)
    assert sidecar.fps == 30.0
    assert sidecar.spatial_unit == "pixel"
    assert sidecar.pixels_per_unit is None
    assert sidecar.h5_dataset is None


def test_load_movie_sidecar_requires_h5_dataset_for_hdf5(tmp_path: Path):
    movie_path = tmp_path / "demo_10s.h5"
    movie_path.write_bytes(b"")

    sidecar_path = get_default_sidecar_path(movie_path)
    _write_sidecar_yaml(
        sidecar_path,
        {
            "version": 1,
            "path": "demo_10s.h5",
            "fps": 15.0,
            "spatial_unit": "mm",
            "pixels_per_unit": [1.0, 1.0],
        },
    )

    with pytest.raises(ValueError):
        load_movie_sidecar(movie_path)


def test_load_movie_sidecar_accepts_hdf5_with_dataset(tmp_path: Path):
    movie_path = tmp_path / "demo_10s.h5"
    movie_path.write_bytes(b"")

    sidecar_path = get_default_sidecar_path(movie_path)
    _write_sidecar_yaml(
        sidecar_path,
        {
            "version": 1,
            "path": "demo_10s.h5",
            "fps": 15.0,
            "spatial_unit": "mm",
            "pixels_per_unit": [1.3513513514, 1.3513513514],
            "h5_dataset": "/Plume Data/plume",
        },
    )

    sidecar = load_movie_sidecar(movie_path)
    assert sidecar.fps == 15.0
    assert sidecar.spatial_unit == "mm"
    assert sidecar.pixels_per_unit == (1.3513513514, 1.3513513514)
    assert sidecar.h5_dataset == "/Plume Data/plume"
