from __future__ import annotations

import numpy as np
import pytest

from plume_nav_sim.core.geometry import GridSize
from plume_nav_sim.data_zoo.movie_plume import MoviePlume
from plume_nav_sim.envs import factory as env_factory
from tests.plume_nav_sim.data_zoo.test_loader import _seed_dataset


def test_movie_plume_from_registry_loads_normalized(monkeypatch, tmp_path):
    pytest.importorskip("dask.array")
    pytest.importorskip("xarray")

    cache_root, entry, data, stats = _seed_dataset(
        tmp_path, monkeypatch, "demo_movie_plume", with_stats=True
    )

    plume = MoviePlume.from_registry(
        entry.dataset_id,
        normalize="minmax",
        cache_root=cache_root,
        auto_download=False,
    )
    arr = plume.load()

    assert plume.dataset_path is not None
    assert plume.dataset_path.exists()
    assert arr.attrs.get("normalized") == "minmax"
    np.testing.assert_allclose(
        arr.compute().values, data / float(stats["max"])  # type: ignore[index]
    )

    kwargs = plume.env_kwargs()
    assert kwargs["movie_dataset_id"] == entry.dataset_id
    assert kwargs["movie_cache_root"] == cache_root
    assert kwargs["movie_normalize"] == "minmax"
    assert kwargs["movie_data"] is arr


def test_create_component_environment_uses_loader_when_normalize(monkeypatch, tmp_path):
    pytest.importorskip("dask.array")
    pytest.importorskip("xarray")

    cache_root, entry, _, _ = _seed_dataset(
        tmp_path, monkeypatch, "demo_movie_env", with_stats=True
    )

    captured_cfg: dict[str, object] = {}

    class DummyField:
        def __init__(self, cfg) -> None:
            captured_cfg["cfg"] = cfg
            data_array = getattr(cfg, "data_array", None)
            width = int(data_array.sizes["x"])
            height = int(data_array.sizes["y"])
            self.grid_size = GridSize(width=width, height=height)
            self.field_array = np.zeros((height, width), dtype=np.float32)

    monkeypatch.setattr(env_factory, "VideoPlume", DummyField)

    env_factory.create_component_environment(
        plume="movie",
        movie_dataset_id=entry.dataset_id,
        movie_cache_root=cache_root,
        movie_auto_download=False,
        movie_normalize="robust",
        movie_chunks=None,
    )

    assert captured_cfg, "VideoPlume was not constructed"
    cfg = captured_cfg["cfg"]
    arr = getattr(cfg, "data_array", None)
    assert arr is not None
    assert arr.attrs.get("normalized") == "robust"
    values = arr.compute().values  # type: ignore[attr-defined]
    assert 0.0 <= float(values.min()) <= 1.0
    assert 0.0 <= float(values.max()) <= 1.0
