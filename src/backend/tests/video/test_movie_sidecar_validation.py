from __future__ import annotations

from pathlib import Path

import pytest


def _write_sidecar_yaml(sidecar_path: Path, data: dict) -> None:
    yaml = pytest.importorskip("yaml")
    sidecar_path.write_text(yaml.safe_dump(data))


def test_sidecar_fps_conflict_raises_validation_error(tmp_path: Path) -> None:
    """Conflicting movie_fps vs sidecar.fps yields a clear ValidationError."""

    from plume_nav_sim.media.sidecar import get_default_sidecar_path
    from plume_nav_sim.plume.movie_field import resolve_movie_dataset_path
    from plume_nav_sim.utils.exceptions import ValidationError

    movie_path = tmp_path / "movie.avi"
    movie_path.write_bytes(b"dummy")

    sidecar_path = get_default_sidecar_path(movie_path)
    _write_sidecar_yaml(
        sidecar_path,
        {
            "version": 1,
            "path": None,
            "fps": 10.0,
            "spatial_unit": "pixel",
            "pixels_per_unit": None,
            "h5_dataset": None,
        },
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve_movie_dataset_path(str(movie_path), fps=12.34)

    msg = str(excinfo.value)
    assert "movie_fps must match sidecar.fps" in msg


def test_sidecar_pixel_to_grid_conflict_raises_validation_error(tmp_path: Path) -> None:
    """Conflicting movie_pixel_to_grid vs sidecar-derived value is rejected."""

    from plume_nav_sim.media.sidecar import get_default_sidecar_path
    from plume_nav_sim.plume.movie_field import resolve_movie_dataset_path
    from plume_nav_sim.utils.exceptions import ValidationError

    movie_path = tmp_path / "movie.mp4"
    movie_path.write_bytes(b"dummy")

    sidecar_path = get_default_sidecar_path(movie_path)
    _write_sidecar_yaml(
        sidecar_path,
        {
            "version": 1,
            "path": None,
            "fps": 5.0,
            "spatial_unit": "um",
            "pixels_per_unit": (10.0, 10.0),
            "h5_dataset": None,
        },
    )

    with pytest.raises(ValidationError) as excinfo:
        resolve_movie_dataset_path(str(movie_path), pixel_to_grid=(0.5, 0.5))

    msg = str(excinfo.value)
    assert "movie_pixel_to_grid must match sidecar-derived pixel_to_grid" in msg
