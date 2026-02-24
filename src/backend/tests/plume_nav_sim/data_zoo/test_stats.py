"""Tests for data_zoo stats.py and the _normalize_arg helper from loader.py.

All tests are offline-safe -- no network access required.
Zarr-dependent tests are skipped when the ``zarr`` package is not installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from plume_nav_sim.data_zoo.stats import normalize_array

_HAS_ZARR = importlib.util.find_spec("zarr") is not None
requires_zarr = pytest.mark.skipif(not _HAS_ZARR, reason="zarr is not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stats(
    *,
    min_val: float = 0.0,
    max_val: float = 10.0,
    mean: float = 5.0,
    std: float = 2.5,
    q05: float = 0.5,
    q95: float = 9.5,
) -> dict:
    """Build a minimal concentration-stats dict with controllable fields."""
    return {
        "min": min_val,
        "max": max_val,
        "mean": mean,
        "std": std,
        "quantiles": {
            "q01": 0.1,
            "q05": q05,
            "q25": 2.5,
            "q50": 5.0,
            "q75": 7.5,
            "q95": q95,
            "q99": 9.9,
            "q999": 9.99,
        },
        "nonzero_fraction": 0.95,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_stats() -> dict:
    return _make_stats()


def _create_zarr_store(tmp_path: Path, name: str, data: np.ndarray) -> Path:
    """Helper that creates a zarr store with a ``concentration`` dataset."""
    import zarr

    zarr_path = tmp_path / name
    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.open_group(store, mode="w")
    root.create_dataset("concentration", data=data, chunks=(5, 4, 4))
    return zarr_path


@pytest.fixture()
def zarr_store(tmp_path: Path) -> Path:
    """Small zarr store with a known concentration array of shape (10, 4, 4)."""
    data = np.arange(160, dtype=np.float32).reshape(10, 4, 4)
    return _create_zarr_store(tmp_path, "test.zarr", data)


@pytest.fixture()
def constant_zarr_store(tmp_path: Path) -> Path:
    """Zarr store where all concentration values are a constant (7.0)."""
    data = np.full((10, 4, 4), 7.0, dtype=np.float32)
    return _create_zarr_store(tmp_path, "constant.zarr", data)


@pytest.fixture()
def zeros_zarr_store(tmp_path: Path) -> Path:
    """Zarr store filled entirely with zeros."""
    data = np.zeros((10, 4, 4), dtype=np.float32)
    return _create_zarr_store(tmp_path, "zeros.zarr", data)


@pytest.fixture()
def negative_zarr_store(tmp_path: Path) -> Path:
    """Zarr store with values ranging from -10 to +10."""
    data = np.linspace(-10.0, 10.0, 160, dtype=np.float32).reshape(10, 4, 4)
    return _create_zarr_store(tmp_path, "negative.zarr", data)


# ===================================================================
# normalize_array  --  minmax
# ===================================================================


class TestNormalizeArrayMinmax:
    def test_basic(self, default_stats):
        data = np.array([0.0, 5.0, 10.0])
        result = normalize_array(data, default_stats, method="minmax")
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_intermediate_values(self, default_stats):
        data = np.array([2.0, 4.0, 8.0])
        result = normalize_array(data, default_stats, method="minmax")
        np.testing.assert_allclose(result, [0.2, 0.4, 0.8])

    def test_constant_data(self):
        """When max == min, returns data - min (all zeros)."""
        stats = _make_stats(min_val=5.0, max_val=5.0)
        data = np.array([5.0, 5.0, 5.0])
        result = normalize_array(data, stats, method="minmax")
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_2d_array(self, default_stats):
        data = np.array([[0.0, 10.0], [5.0, 2.5]])
        result = normalize_array(data, default_stats, method="minmax")
        np.testing.assert_allclose(result, [[0.0, 1.0], [0.5, 0.25]])

    def test_values_outside_range(self, default_stats):
        """Values outside [min, max] are extrapolated, not clipped."""
        data = np.array([-5.0, 15.0])
        result = normalize_array(data, default_stats, method="minmax")
        np.testing.assert_allclose(result, [-0.5, 1.5])


# ===================================================================
# normalize_array  --  robust
# ===================================================================


class TestNormalizeArrayRobust:
    def test_basic(self, default_stats):
        data = np.array([0.5, 5.0, 9.5])
        result = normalize_array(data, default_stats, method="robust")
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_clips_to_zero_one(self, default_stats):
        """Values outside [q05, q95] are clipped to [0, 1]."""
        data = np.array([-100.0, 100.0])
        result = normalize_array(data, default_stats, method="robust")
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_constant_quantiles(self):
        """When q95 == q05, returns zeros."""
        stats = _make_stats(q05=5.0, q95=5.0)
        data = np.array([1.0, 5.0, 10.0])
        result = normalize_array(data, stats, method="robust")
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])
        assert result.shape == data.shape


# ===================================================================
# normalize_array  --  zscore
# ===================================================================


class TestNormalizeArrayZscore:
    def test_basic(self, default_stats):
        data = np.array([5.0, 7.5, 2.5])
        result = normalize_array(data, default_stats, method="zscore")
        np.testing.assert_allclose(result, [0.0, 1.0, -1.0])

    def test_zero_std(self):
        """When std == 0, returns data - mean."""
        stats = _make_stats(mean=3.0, std=0.0)
        data = np.array([3.0, 5.0, 1.0])
        result = normalize_array(data, stats, method="zscore")
        np.testing.assert_allclose(result, [0.0, 2.0, -2.0])


# ===================================================================
# normalize_array  --  None / unknown
# ===================================================================


class TestNormalizeArrayNone:
    def test_returns_data_unchanged(self, default_stats):
        data = np.array([1.0, 2.0, 3.0])
        result = normalize_array(data, default_stats, method=None)
        assert result is data


class TestNormalizeArrayUnknownMethod:
    def test_raises_value_error(self, default_stats):
        data = np.array([1.0])
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_array(data, default_stats, method="banana")


# ===================================================================
# compute_concentration_stats  (requires zarr)
# ===================================================================


@requires_zarr
class TestComputeConcentrationStats:
    def _compute(self, zarr_path, **kwargs):
        from plume_nav_sim.data_zoo.stats import compute_concentration_stats

        return compute_concentration_stats(zarr_path, **kwargs)

    def test_basic_stats(self, zarr_store):
        stats = self._compute(zarr_store)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(159.0)
        assert stats["mean"] == pytest.approx(79.5)
        assert stats["std"] > 0
        assert 0.0 < stats["nonzero_fraction"] < 1.0

    def test_quantiles_present(self, zarr_store):
        stats = self._compute(zarr_store)
        q = stats["quantiles"]
        for key in ("q01", "q05", "q25", "q50", "q75", "q95", "q99", "q999"):
            assert key in q
            assert isinstance(q[key], float)

    def test_quantiles_ordered(self, zarr_store):
        stats = self._compute(zarr_store)
        q = stats["quantiles"]
        assert q["q01"] <= q["q05"] <= q["q25"] <= q["q50"]
        assert q["q50"] <= q["q75"] <= q["q95"] <= q["q99"] <= q["q999"]

    def test_sample_frames(self, zarr_store):
        """Passing sample_frames < n_frames still produces valid stats."""
        stats = self._compute(zarr_store, sample_frames=3)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(159.0)
        assert "quantiles" in stats

    def test_chunk_size(self, zarr_store):
        """Different chunk_size should produce the same first-pass stats."""
        stats_small = self._compute(zarr_store, chunk_size=2)
        stats_large = self._compute(zarr_store, chunk_size=100)
        assert stats_small["min"] == pytest.approx(stats_large["min"])
        assert stats_small["max"] == pytest.approx(stats_large["max"])
        assert stats_small["mean"] == pytest.approx(stats_large["mean"])
        assert stats_small["std"] == pytest.approx(stats_large["std"])

    def test_extra_fields(self, zarr_store):
        stats = self._compute(zarr_store)
        assert stats["original_min"] is None
        assert stats["original_max"] is None
        assert stats["normalized_during_ingest"] is False

    def test_constant_data(self, constant_zarr_store):
        stats = self._compute(constant_zarr_store)
        assert stats["min"] == pytest.approx(7.0)
        assert stats["max"] == pytest.approx(7.0)
        assert stats["mean"] == pytest.approx(7.0)
        assert stats["std"] == pytest.approx(0.0, abs=1e-6)

    def test_all_zeros(self, zeros_zarr_store):
        stats = self._compute(zeros_zarr_store)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(0.0)
        assert stats["mean"] == pytest.approx(0.0)
        assert stats["std"] == pytest.approx(0.0, abs=1e-6)
        assert stats["nonzero_fraction"] == pytest.approx(0.0)

    def test_negative_values(self, negative_zarr_store):
        stats = self._compute(negative_zarr_store)
        assert stats["min"] < 0
        assert stats["max"] > 0


# ===================================================================
# store_stats_in_zarr / load_stats_from_zarr  round-trip (requires zarr)
# ===================================================================


@requires_zarr
class TestStoreLoadStatsRoundTrip:
    def _store(self, zarr_path, stats):
        from plume_nav_sim.data_zoo.stats import store_stats_in_zarr

        store_stats_in_zarr(zarr_path, stats)

    def _load(self, zarr_path):
        from plume_nav_sim.data_zoo.stats import load_stats_from_zarr

        return load_stats_from_zarr(zarr_path)

    def _compute(self, zarr_path):
        from plume_nav_sim.data_zoo.stats import compute_concentration_stats

        return compute_concentration_stats(zarr_path)

    def test_round_trip(self, zarr_store):
        original = self._compute(zarr_store)
        self._store(zarr_store, original)
        loaded = self._load(zarr_store)

        assert loaded is not None
        assert loaded["min"] == pytest.approx(original["min"])
        assert loaded["max"] == pytest.approx(original["max"])
        assert loaded["mean"] == pytest.approx(original["mean"])
        assert loaded["std"] == pytest.approx(original["std"])
        assert loaded["nonzero_fraction"] == pytest.approx(
            original["nonzero_fraction"]
        )

        for key in original["quantiles"]:
            assert loaded["quantiles"][key] == pytest.approx(
                original["quantiles"][key]
            )

    def test_load_returns_none_when_absent(self, zarr_store):
        """Before any stats are stored, load should return None."""
        result = self._load(zarr_store)
        assert result is None

    def test_overwrite(self, zarr_store):
        """Storing stats twice overwrites the first set."""
        stats_a = {
            "min": 0.0,
            "max": 1.0,
            "mean": 0.5,
            "std": 0.3,
            "quantiles": {},
            "nonzero_fraction": 1.0,
        }
        stats_b = {
            "min": -1.0,
            "max": 2.0,
            "mean": 0.0,
            "std": 1.0,
            "quantiles": {},
            "nonzero_fraction": 0.5,
        }

        self._store(zarr_store, stats_a)
        self._store(zarr_store, stats_b)

        loaded = self._load(zarr_store)
        assert loaded is not None
        assert loaded["min"] == pytest.approx(-1.0)
        assert loaded["max"] == pytest.approx(2.0)


# ===================================================================
# _normalize_arg  (from loader.py)
# ===================================================================


class TestNormalizeArg:
    def _normalize_arg(self, value):
        from plume_nav_sim.data_zoo.loader import _normalize_arg

        return _normalize_arg(value)

    def test_none_returns_none(self):
        assert self._normalize_arg(None) is None

    def test_minmax_lowercase(self):
        assert self._normalize_arg("minmax") == "minmax"

    def test_robust_lowercase(self):
        assert self._normalize_arg("robust") == "robust"

    def test_zscore_lowercase(self):
        assert self._normalize_arg("zscore") == "zscore"

    def test_case_insensitive_upper(self):
        assert self._normalize_arg("ROBUST") == "robust"

    def test_case_insensitive_mixed(self):
        assert self._normalize_arg("MinMax") == "minmax"

    def test_case_insensitive_zscore(self):
        assert self._normalize_arg("ZSCORE") == "zscore"

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="normalize must be one of"):
            self._normalize_arg("invalid")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            self._normalize_arg("")
