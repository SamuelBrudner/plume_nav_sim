from __future__ import annotations

import sys

import numpy as np
import pytest


def _ensure_src_on_path() -> None:
    import pathlib

    # Repository root is 4 levels up from this file: src/backend/tests/compose
    root = pathlib.Path(__file__).resolve().parents[4]
    src = root / "src"
    p = str(src)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_src_on_path()

from plume_nav_sim.compose.builders import prepare  # noqa: E402
from plume_nav_sim.compose.specs import PolicySpec, SimulationSpec  # noqa: E402
from plume_nav_sim.core.geometry import GridSize  # noqa: E402
from plume_nav_sim.envs import factory as env_factory  # noqa: E402


def test_movie_plume_env_factory_preserves_overrides_for_dataset_dirs(
    tmp_path, monkeypatch
):
    """When movie_path is a dataset directory, explicit overrides are forwarded."""

    dataset_dir = tmp_path / "smoke.zarr"
    dataset_dir.mkdir()

    calls: dict[str, object] = {}

    def fake_resolve_movie_dataset_path(
        source_path: str,
        *,
        fps,
        pixel_to_grid,
        origin,
        extent,
        movie_h5_dataset,
    ):
        calls["source_path"] = source_path
        calls["fps"] = fps
        calls["pixel_to_grid"] = pixel_to_grid
        calls["origin"] = origin
        calls["extent"] = extent
        calls["movie_h5_dataset"] = movie_h5_dataset
        return dataset_dir

    captured_cfgs = []

    class DummyField:
        def __init__(self, cfg) -> None:
            captured_cfgs.append(cfg)
            self.grid_size = GridSize(width=32, height=16)
            self.field_array = np.zeros((16, 32), dtype=np.float32)

    monkeypatch.setattr(
        env_factory,
        "resolve_movie_dataset_path",
        fake_resolve_movie_dataset_path,
    )
    monkeypatch.setattr(env_factory, "VideoPlume", DummyField)

    env_factory.create_component_environment(
        plume="movie",
        movie_path=str(dataset_dir),
        movie_fps=9.99,
        movie_pixel_to_grid=(0.5, 0.5),
        movie_origin=(1.0, 2.0),
        movie_extent=(8.0, 16.0),
        movie_step_policy="clamp",
        movie_h5_dataset="foo",
    )

    assert calls["source_path"] == str(dataset_dir)
    assert calls["fps"] == 9.99
    assert calls["pixel_to_grid"] == (0.5, 0.5)
    assert calls["origin"] == (1.0, 2.0)
    assert calls["extent"] == (8.0, 16.0)
    assert calls["movie_h5_dataset"] == "foo"

    # Dataset directories preserve explicit overrides when constructing VideoConfig
    assert captured_cfgs, "VideoPlume was not constructed"
    cfg = captured_cfgs[0]
    assert cfg.path == str(dataset_dir)
    assert cfg.fps == 9.99
    assert cfg.pixel_to_grid == (0.5, 0.5)
    assert cfg.origin == (1.0, 2.0)
    assert cfg.extent == (8.0, 16.0)
    assert cfg.step_policy == "clamp"


def test_movie_plume_env_factory_drops_overrides_for_raw_sources(tmp_path, monkeypatch):
    """For raw media sources, overrides are used only for sidecar validation."""

    raw_path = tmp_path / "movie.avi"
    raw_path.write_bytes(b"dummy")
    dataset_dir = tmp_path / "movie.avi.zarr"

    calls: dict[str, object] = {}

    def fake_resolve_movie_dataset_path(
        source_path: str,
        *,
        fps,
        pixel_to_grid,
        origin,
        extent,
        movie_h5_dataset,
    ):
        calls["source_path"] = source_path
        calls["fps"] = fps
        calls["pixel_to_grid"] = pixel_to_grid
        calls["origin"] = origin
        calls["extent"] = extent
        calls["movie_h5_dataset"] = movie_h5_dataset
        return dataset_dir

    captured_cfgs = []

    class DummyField:
        def __init__(self, cfg) -> None:
            captured_cfgs.append(cfg)
            self.grid_size = GridSize(width=32, height=16)
            self.field_array = np.zeros((16, 32), dtype=np.float32)

    monkeypatch.setattr(
        env_factory,
        "resolve_movie_dataset_path",
        fake_resolve_movie_dataset_path,
    )
    monkeypatch.setattr(env_factory, "VideoPlume", DummyField)

    env_factory.create_component_environment(
        plume="movie",
        movie_path=str(raw_path),
        movie_fps=9.99,
        movie_pixel_to_grid=(0.5, 0.5),
        movie_origin=(1.0, 2.0),
        movie_extent=(8.0, 16.0),
        movie_step_policy="wrap",
        movie_h5_dataset="bar",
    )

    assert calls["source_path"] == str(raw_path)
    assert calls["fps"] == 9.99
    assert calls["pixel_to_grid"] == (0.5, 0.5)
    assert calls["origin"] == (1.0, 2.0)
    assert calls["extent"] == (8.0, 16.0)
    assert calls["movie_h5_dataset"] == "bar"

    # For non-directory sources, VideoConfig relies on dataset/sidecar metadata
    assert captured_cfgs, "VideoPlume was not constructed"
    cfg = captured_cfgs[0]
    assert cfg.path == str(dataset_dir)
    assert cfg.fps is None
    assert cfg.pixel_to_grid is None
    assert cfg.origin is None
    assert cfg.extent is None
    assert cfg.step_policy == "wrap"


def test_simulation_spec_movie_plume_round_trip(tmp_path, monkeypatch):
    """SimulationSpec with plume='movie' wires through to the factory."""

    dataset_dir = tmp_path / "movie_dataset.zarr"
    dataset_dir.mkdir()

    calls: dict[str, object] = {}

    def fake_resolve_movie_dataset_path(
        source_path: str,
        *,
        fps,
        pixel_to_grid,
        origin,
        extent,
        movie_h5_dataset,
    ):
        calls["source_path"] = source_path
        calls["fps"] = fps
        calls["pixel_to_grid"] = pixel_to_grid
        calls["origin"] = origin
        calls["extent"] = extent
        calls["movie_h5_dataset"] = movie_h5_dataset
        return dataset_dir

    captured_cfgs = []

    class DummyField:
        def __init__(self, cfg) -> None:
            captured_cfgs.append(cfg)
            self.grid_size = GridSize(width=40, height=24)
            self.field_array = np.zeros((24, 40), dtype=np.float32)

    monkeypatch.setattr(
        env_factory,
        "resolve_movie_dataset_path",
        fake_resolve_movie_dataset_path,
    )
    monkeypatch.setattr(env_factory, "VideoPlume", DummyField)

    sim = SimulationSpec(
        grid_size=(16, 16),
        max_steps=100,
        action_type="oriented",
        observation_type="concentration",
        reward_type="step_penalty",
        render=False,
        plume="movie",
        movie_path=str(dataset_dir),
        movie_fps=15.0,
        movie_pixel_to_grid=(1.0, 1.0),
        movie_origin=(0.0, 0.0),
        movie_extent=(24.0, 40.0),
        movie_step_policy="wrap",
        policy=PolicySpec(builtin="deterministic_td"),
    )

    env, policy = prepare(sim)
    assert policy is not None

    assert calls["source_path"] == str(dataset_dir)
    assert calls["fps"] == 15.0
    assert calls["pixel_to_grid"] == (1.0, 1.0)
    assert calls["origin"] == (0.0, 0.0)
    assert calls["extent"] == (24.0, 40.0)
    assert calls["movie_h5_dataset"] is None

    assert captured_cfgs, "VideoPlume was not constructed via SimulationSpec.prepare"
    cfg = captured_cfgs[0]
    assert cfg.path == str(dataset_dir)
    assert cfg.fps == 15.0
    assert cfg.pixel_to_grid == (1.0, 1.0)
    assert cfg.origin == (0.0, 0.0)
    assert cfg.extent == (24.0, 40.0)
    assert cfg.step_policy == "wrap"

    # Env grid_size should reflect the VideoPlume grid, not the original grid_size
    gs = getattr(env, "grid_size", None)
    assert gs is not None
    width = getattr(gs, "width", None) or int(gs[0])
    height = getattr(gs, "height", None) or int(gs[1])
    assert (width, height) == (40, 24)


def test_resolve_movie_dataset_prefers_local_override(monkeypatch, tmp_path):
    override = tmp_path / "local_override.zarr"
    override.mkdir()

    download_calls = {"count": 0}

    def fake_ensure_dataset_available(*args, **kwargs):
        download_calls["count"] += 1
        return tmp_path / "registry" / "downloaded"

    monkeypatch.setattr(
        env_factory, "ensure_dataset_available", fake_ensure_dataset_available
    )

    resolved = env_factory._resolve_movie_dataset(
        movie_path=str(override),
        movie_dataset_id="registry_id",
        movie_auto_download=False,
        movie_cache_root=None,
        movie_fps=None,
        movie_pixel_to_grid=None,
        movie_origin=None,
        movie_extent=None,
        movie_h5_dataset=None,
    )

    assert resolved == override
    assert download_calls["count"] == 0


def test_resolve_movie_dataset_unknown_id_has_clear_error(monkeypatch):
    def fake_ensure_dataset_available(*args, **kwargs):
        raise KeyError("missing")

    monkeypatch.setattr(
        env_factory, "ensure_dataset_available", fake_ensure_dataset_available
    )
    monkeypatch.setattr(
        env_factory, "get_dataset_registry", lambda: {"known_dataset": object()}
    )

    with pytest.raises(ValueError) as excinfo:
        env_factory._resolve_movie_dataset(
            movie_path=None,
            movie_dataset_id="missing_id",
            movie_auto_download=False,
            movie_cache_root=None,
            movie_fps=None,
            movie_pixel_to_grid=None,
            movie_origin=None,
            movie_extent=None,
            movie_h5_dataset=None,
        )

    msg = str(excinfo.value)
    assert "missing_id" in msg
    assert "known_dataset" in msg
