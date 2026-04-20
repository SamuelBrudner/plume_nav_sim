"""Tests for PlumeEnv plume backend wiring."""

from __future__ import annotations

import numpy as np
import pytest
import warnings

import plume_nav_sim as pns
from plume_nav_sim._compat import ValidationError
from plume_nav_sim.envs.plume_env import create_plume_env
from plume_nav_sim.media.schema import DIMS_TYX, SCHEMA_VERSION


class _FakeSlice:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _FakeDataset:
    def __init__(self, data_vars: dict, attrs: dict) -> None:
        self.data_vars = data_vars
        self.attrs = attrs

    def __getitem__(self, key: str):
        return self.data_vars[key]


class _FakeDataArray:
    def __init__(self, data: np.ndarray, attrs: dict) -> None:
        self._data = np.asarray(data, dtype=np.float32)
        self.dims = DIMS_TYX
        self.sizes = {
            "t": self._data.shape[0],
            "y": self._data.shape[1],
            "x": self._data.shape[2],
        }
        self.attrs = {"_ARRAY_DIMENSIONS": DIMS_TYX}
        self._dataset_attrs = attrs

    def isel(self, t: int):
        return _FakeSlice(self._data[int(t)])

    def to_dataset(self, name: str):
        return _FakeDataset({name: self}, attrs=self._dataset_attrs)


def _make_video_data_array(
    *, frames: int = 3, height: int = 5, width: int = 4
) -> _FakeDataArray:
    data = np.linspace(
        0.0, 1.0, num=frames * height * width, dtype=np.float32
    ).reshape(frames, height, width)
    attrs = {
        "schema_version": SCHEMA_VERSION,
        "fps": 10.0,
        "source_dtype": "float32",
        "pixel_to_grid": (1.0, 1.0),
        "origin": (0.0, 0.0),
        "extent": (float(height), float(width)),
        "dims": DIMS_TYX,
    }
    return _FakeDataArray(data, attrs)


def test_create_plume_env_gaussian_backend_runs():
    env = create_plume_env(
        plume_type="gaussian",
        grid_size=(10, 10),
        source_location=(5, 5),
        max_steps=5,
    )
    try:
        obs, info = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (1,)
        assert info.get("seed") == 123

        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info.get("step_count") == 1
    finally:
        env.close()


def test_create_plume_env_video_backend_advances_frames():
    data_array = _make_video_data_array()
    env = create_plume_env(
        plume_type="video",
        video_data=data_array,
        max_steps=3,
    )
    try:
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert env.grid_size == (4, 5)
        assert env.plume.frame_index == 0

        env.step(0)
        assert env.plume.frame_index == 1
    finally:
        env.close()


def test_create_plume_env_rejects_explicit_out_of_bounds_source_location():
    with pytest.raises(ValidationError, match="within grid bounds"):
        create_plume_env(
            plume_type="gaussian",
            grid_size=(8, 8),
            source_location=(99, 99),
            max_steps=5,
        )


def test_make_env_rejects_legacy_out_of_bounds_source_location():
    with pytest.raises(ValidationError, match="coordinates outside grid bounds"):
        pns.make_env(
            grid_size=(8, 8),
            source_location=(99, 99),
            action_type="discrete",
        )


def test_make_env_normalizes_static_plume_to_stable_plume_env():
    env = pns.make_env(grid_size=(8, 8), source_location=(4, 4), plume="static")

    try:
        assert isinstance(env, pns.PlumeEnv)
        _, info = env.reset(seed=5)
        assert info["seed"] == 5
    finally:
        env.close()


def test_make_env_normalizes_movie_plume_to_stable_video_env():
    with pytest.warns(
        DeprecationWarning,
        match="make_env\\(plume='movie'\\) is deprecated; use plume_type='video' instead",
    ):
        env = pns.make_env(
            plume="movie",
            video_data=_make_video_data_array(),
            max_steps=3,
        )

    try:
        assert isinstance(env, pns.PlumeEnv)
        obs, _ = env.reset(seed=0)
        assert isinstance(obs, np.ndarray)
        assert env.grid_size == (4, 5)
        assert env.plume.frame_index == 0
    finally:
        env.close()


def test_make_env_rejects_mixed_stable_and_component_kwargs():
    with pytest.raises(
        ValidationError,
        match="Cannot mix deprecated component kwargs with stable PlumeEnv kwargs",
    ):
        pns.make_env(
            plume="movie",
            video_data=_make_video_data_array(),
            action_type="discrete",
        )


def test_make_env_component_route_warns_and_returns_compatibility_env():
    with pytest.warns(
        DeprecationWarning,
        match="routing to component environment factory",
    ):
        env = pns.make_env(
            grid_size=(8, 8),
            source_location=(4, 4),
            action_type="discrete",
        )

    try:
        assert not isinstance(env, pns.PlumeEnv)
        _, info = env.reset(seed=7)
        assert info["seed"] == 7
    finally:
        env.close()


def test_make_env_legacy_info_surface_matches_default_info_keys():
    default_env = pns.make_env(grid_size=(8, 8), source_location=(4, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_env = pns.make_env(
            grid_size=(8, 8),
            source_location=(4, 4),
            action_type="discrete",
        )

    try:
        _, default_reset_info = default_env.reset(seed=123)
        _, legacy_reset_info = legacy_env.reset(seed=123)
        assert set(default_reset_info.keys()) == set(legacy_reset_info.keys())

        _, _, _, _, default_step_info = default_env.step(0)
        _, _, _, _, legacy_step_info = legacy_env.step(0)
        assert set(default_step_info.keys()) == set(legacy_step_info.keys())
    finally:
        default_env.close()
        legacy_env.close()
