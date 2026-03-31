from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from plume_nav_sim.cli import capture


def test_locate_hydra_config_path_uses_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "conf"
    assert capture._locate_hydra_config_path(str(explicit)) == explicit


def test_locate_hydra_config_path_uses_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved = tmp_path / "resolved-conf"
    monkeypatch.setattr(capture, "_resolve_config_dir", lambda _: str(resolved))
    assert capture._locate_hydra_config_path(None) == resolved


def test_build_hydra_overrides_returns_empty_for_none() -> None:
    assert capture._build_hydra_overrides(None) == []


def test_build_hydra_overrides_copies_input_list() -> None:
    overrides = ["seed=42", "env.plume=movie"]
    built = capture._build_hydra_overrides(overrides)
    assert built == overrides
    assert built is not overrides


def test_map_hydra_cfg_to_run_config_accepts_mapping() -> None:
    raw = {"output": "results", "env": {"grid_size": [64, 64]}}
    mapped = capture._map_hydra_cfg_to_run_config(raw, cfg_is_mapping=True)
    assert mapped == raw
    assert mapped is not raw


def test_map_hydra_cfg_to_run_config_uses_omegaconf_for_dictconfig(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOmegaConf:
        @staticmethod
        def to_container(cfg: object, resolve: bool = False) -> dict[str, object]:
            assert cfg == "dictconfig"
            assert resolve is True
            return {"experiment": "default"}

    monkeypatch.setattr(capture, "_require_omegaconf", lambda: FakeOmegaConf)
    mapped = capture._map_hydra_cfg_to_run_config("dictconfig")
    assert mapped == {"experiment": "default"}


def test_map_hydra_cfg_to_run_config_rejects_non_mapping() -> None:
    with pytest.raises(SystemExit, match="must resolve to a mapping"):
        capture._map_hydra_cfg_to_run_config(["not", "a", "mapping"], cfg_is_mapping=True)


class _FakeEnv:
    def __init__(self, *, width: int, height: int, goal_location: tuple[int, int] | None = None):
        self.grid_size = SimpleNamespace(width=width, height=height)
        if goal_location is not None:
            self.goal_location = goal_location
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _fail_make_env(**kwargs):
    raise AssertionError(f"capture should not call pns.make_env: {kwargs}")


def test_base_component_env_kwargs_wires_goal_radius_and_ignores_render() -> None:
    kwargs = capture._base_component_env_kwargs(
        {
            "goal_radius": 3.5,
            "render": True,
        }
    )

    assert kwargs["goal_radius"] == 3.5
    assert "render" not in kwargs
    assert "render_mode" not in kwargs


def test_env_from_cfg_static_uses_component_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_factory(**kwargs):
        seen["kwargs"] = kwargs
        return _FakeEnv(width=32, height=24)

    monkeypatch.setattr(capture, "create_component_environment", fake_factory)
    monkeypatch.setattr(capture.pns, "make_env", _fail_make_env)

    env, width, height = capture._env_from_cfg(
        {
            "env": {
                "grid_size": [32, 24],
                "goal_radius": 4.5,
                "action_type": "run_tumble",
                "observation_type": "antennae",
                "reward_type": "sparse",
                "max_steps": 77,
            }
        }
    )

    assert env is not None
    assert (width, height) == (32, 24)
    assert seen["kwargs"] == {
        "grid_size": (32, 24),
        "goal_radius": 4.5,
        "action_type": "run_tumble",
        "observation_type": "antennae",
        "reward_type": "sparse",
        "max_steps": 77,
    }


def test_env_from_cfg_movie_uses_component_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_factory(**kwargs):
        seen["kwargs"] = kwargs
        return _FakeEnv(width=11, height=17)

    monkeypatch.setattr(capture, "create_component_environment", fake_factory)
    monkeypatch.setattr(capture.pns, "make_env", _fail_make_env)

    env, width, height = capture._env_from_cfg(
        {
            "env": {
                "plume": "movie",
                "action_type": "oriented",
                "observation_type": "wind_vector",
                "reward_type": "step_penalty",
            },
            "movie": {
                "dataset_id": "colorado_jet_v1",
                "auto_download": True,
                "cache_root": "/tmp/cache",
                "path": "/tmp/demo.zarr",
                "fps": 12.5,
                "pixel_to_grid": [1.5, 0.5],
                "origin": [0.0, 1.0],
                "extent": [10.0, 20.0],
                "step_policy": "clamp",
                "h5_dataset": "concentration",
                "normalize": "zscore",
                "chunks": "auto",
            },
        }
    )

    assert env is not None
    assert (width, height) == (11, 17)
    assert seen["kwargs"] == {
        "action_type": "oriented",
        "observation_type": "wind_vector",
        "reward_type": "step_penalty",
        "plume": "movie",
        "movie_auto_download": True,
        "movie_step_policy": "clamp",
        "movie_path": "/tmp/demo.zarr",
        "movie_dataset_id": "colorado_jet_v1",
        "movie_cache_root": "/tmp/cache",
        "movie_fps": 12.5,
        "movie_pixel_to_grid": (1.5, 0.5),
        "movie_origin": (0.0, 1.0),
        "movie_extent": (10.0, 20.0),
        "movie_h5_dataset": "concentration",
        "movie_normalize": "zscore",
        "movie_chunks": "auto",
    }


def test_capture_env_config_uses_goal_location_when_source_missing() -> None:
    env = _FakeEnv(width=24, height=12, goal_location=(7, 5))
    env.max_steps = 88
    env.goal_radius = 2.5

    cfg = capture._capture_env_config(env, default_grid=(1, 1))

    assert cfg.grid_size.to_tuple() == (24, 12)
    assert cfg.source_location.to_tuple() == (7, 5)
    assert cfg.max_steps == 88
    assert cfg.goal_radius == 2.5


def test_main_without_config_uses_component_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_factory(**kwargs):
        env = _FakeEnv(width=8, height=10)
        seen["env"] = env
        seen["kwargs"] = kwargs
        return env

    def fake_run_capture(env, w, h, *, args, cfg_hash, capture_cfg):
        seen["run"] = {
            "env": env,
            "grid": (w, h),
            "experiment": args.experiment,
            "cfg_hash": cfg_hash,
            "capture_cfg": capture_cfg,
        }

    monkeypatch.setattr(capture, "create_component_environment", fake_factory)
    monkeypatch.setattr(capture, "_run_capture", fake_run_capture)
    monkeypatch.setattr(capture.pns, "make_env", _fail_make_env)

    exit_code = capture.main(
        [
            "--output",
            str(tmp_path / "results"),
            "--grid",
            "8x10",
        ]
    )

    assert exit_code == 0
    assert seen["kwargs"] == {
        "grid_size": (8, 10),
        "action_type": "oriented",
        "observation_type": "concentration",
        "reward_type": "step_penalty",
    }
    assert seen["run"] == {
        "env": seen["env"],
        "grid": (8, 10),
        "experiment": "default",
        "cfg_hash": None,
        "capture_cfg": None,
    }
    assert seen["env"].closed is True
