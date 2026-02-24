"""Tests for Data Zoo ingest helper functions.

Covers the pure/testable helpers from the three ingest modules:

- ingest/__init__.py : _compute_chunk_t, _normalize_concentration
- ingest/rigolli.py  : _rigolli_axis_candidates, _pick_rigolli_axes,
                        _infer_time_y_x_axes_for_rigolli
- ingest/emonet.py   : _emonet_frame_signal, _compute_emonet_background

These tests use real numpy arrays (no mocks) and do NOT exercise the full
_ingest_*_to_zarr pipelines, which require HDF5 files and heavy dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

from plume_nav_sim.data_zoo.downloader import DatasetDownloadError
from plume_nav_sim.data_zoo.ingest import _compute_chunk_t, _normalize_concentration
from plume_nav_sim.data_zoo.ingest.emonet import (
    _compute_emonet_background,
    _emonet_frame_signal,
)
from plume_nav_sim.data_zoo.ingest.rigolli import (
    _infer_time_y_x_axes_for_rigolli,
    _pick_rigolli_axes,
    _rigolli_axis_candidates,
)


# ============================================================================
# _compute_chunk_t
# ============================================================================


class TestComputeChunkT:
    """_compute_chunk_t(spec_chunk, bytes_per_frame) -> int

    The chunk size in the time dimension is the minimum of:
      - spec_chunk (or 100 if None)
      - floor(500 MB / bytes_per_frame), but at least 1
    """

    def test_none_spec_defaults_to_100(self):
        # With small frames, 500 MB holds far more than 100 frames.
        result = _compute_chunk_t(None, bytes_per_frame=1000)
        assert result == 100

    def test_spec_chunk_smaller_than_budget(self):
        # spec_chunk=50 fits within the 500 MB budget, so it is returned.
        result = _compute_chunk_t(50, bytes_per_frame=1000)
        assert result == 50

    def test_budget_caps_spec_chunk(self):
        # 500 MB = 524_288_000 bytes.  If each frame is 100 MB = 104_857_600 B,
        # the budget allows floor(524_288_000 / 104_857_600) = 5 frames.
        big_frame = 100 * 1024 * 1024  # 100 MB
        result = _compute_chunk_t(200, big_frame)
        assert result == 5

    def test_budget_floors_to_at_least_one(self):
        # A frame bigger than 500 MB should still produce chunk_t >= 1.
        huge_frame = 1024 * 1024 * 1024  # 1 GB
        result = _compute_chunk_t(100, huge_frame)
        assert result == 1

    def test_zero_bytes_per_frame_returns_spec(self):
        # bytes_per_frame=0 is degenerate; max(1, 0)=1, budget = 500 MB.
        result = _compute_chunk_t(42, 0)
        assert result == 42

    def test_spec_chunk_one(self):
        result = _compute_chunk_t(1, bytes_per_frame=1000)
        assert result == 1

    def test_returns_int(self):
        result = _compute_chunk_t(None, 4096)
        assert isinstance(result, int)


# ============================================================================
# _normalize_concentration
# ============================================================================


class _ZarrLikeArray:
    """Thin wrapper around a numpy array that supports slice-based __getitem__
    and __setitem__, mimicking the zarr dataset interface used by
    _normalize_concentration.
    """

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def __getitem__(self, key):
        return self._arr[key]

    def __setitem__(self, key, value):
        self._arr[key] = value

    @property
    def data(self) -> np.ndarray:
        return self._arr


class TestNormalizeConcentration:
    """_normalize_concentration(conc, *, n_frames, chunk_t, global_min, global_max)

    Applies in-place min-max normalization: (val - min) / (max - min).
    Processes the time dimension in chunks of chunk_t.
    """

    def test_basic_normalization(self):
        raw = np.array([[[0.0, 5.0], [10.0, 15.0]],
                        [[20.0, 25.0], [30.0, 35.0]]], dtype=np.float32)
        conc = _ZarrLikeArray(raw.copy())
        _normalize_concentration(conc, n_frames=2, chunk_t=2,
                                 global_min=0.0, global_max=35.0)
        expected = raw / 35.0
        np.testing.assert_allclose(conc.data, expected, atol=1e-6)

    def test_normalization_range(self):
        raw = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        conc = _ZarrLikeArray(raw.copy())
        _normalize_concentration(conc, n_frames=4, chunk_t=2,
                                 global_min=0.0, global_max=23.0)
        assert float(conc.data.min()) == pytest.approx(0.0, abs=1e-6)
        assert float(conc.data.max()) == pytest.approx(1.0, abs=1e-6)

    def test_nonzero_global_min(self):
        raw = np.array([[[10.0, 20.0]], [[30.0, 40.0]]], dtype=np.float32)
        conc = _ZarrLikeArray(raw.copy())
        _normalize_concentration(conc, n_frames=2, chunk_t=2,
                                 global_min=10.0, global_max=40.0)
        expected = (raw - 10.0) / 30.0
        np.testing.assert_allclose(conc.data, expected, atol=1e-6)

    def test_chunk_processing_matches_single_pass(self):
        """Chunked processing should produce the same result as one big pass."""
        raw = np.random.default_rng(42).random((10, 3, 4)).astype(np.float32)
        gmin, gmax = float(raw.min()), float(raw.max())

        # Single-chunk pass
        conc_single = _ZarrLikeArray(raw.copy())
        _normalize_concentration(conc_single, n_frames=10, chunk_t=10,
                                 global_min=gmin, global_max=gmax)

        # Multi-chunk pass (chunk_t=3 means chunks of 3, 3, 3, 1)
        conc_multi = _ZarrLikeArray(raw.copy())
        _normalize_concentration(conc_multi, n_frames=10, chunk_t=3,
                                 global_min=gmin, global_max=gmax)

        np.testing.assert_allclose(conc_single.data, conc_multi.data, atol=1e-6)

    def test_chunk_t_larger_than_n_frames(self):
        """If chunk_t > n_frames, one iteration covers everything."""
        raw = np.ones((2, 3, 3), dtype=np.float32) * 5.0
        conc = _ZarrLikeArray(raw.copy())
        _normalize_concentration(conc, n_frames=2, chunk_t=100,
                                 global_min=0.0, global_max=10.0)
        np.testing.assert_allclose(conc.data, 0.5, atol=1e-6)

    def test_in_place_modification(self):
        """The array should be mutated in place."""
        raw = np.array([[[2.0, 4.0]]], dtype=np.float32)
        conc = _ZarrLikeArray(raw)
        _normalize_concentration(conc, n_frames=1, chunk_t=1,
                                 global_min=0.0, global_max=4.0)
        # raw itself should have changed because conc wraps it
        np.testing.assert_allclose(raw, [[[0.5, 1.0]]], atol=1e-6)


# ============================================================================
# _rigolli_axis_candidates
# ============================================================================


class TestRigolliAxisCandidates:
    """_rigolli_axis_candidates(shape, *, x_len, y_len) -> list of (time, y, x) tuples

    Considers all permutations of three axes and returns those where the
    y and x dimensions match y_len and x_len respectively.
    """

    def test_unambiguous_tyx(self):
        # shape (100, 50, 30) with y_len=50, x_len=30 -> time must be axis 0
        candidates = _rigolli_axis_candidates((100, 50, 30), x_len=30, y_len=50)
        assert (0, 1, 2) in candidates

    def test_unambiguous_single_candidate(self):
        # All dimensions differ, only one valid permutation exists.
        candidates = _rigolli_axis_candidates((200, 80, 60), x_len=60, y_len=80)
        assert len(candidates) == 1
        assert candidates[0] == (0, 1, 2)

    def test_no_candidates_when_no_match(self):
        # shape dimensions do not include y_len or x_len
        candidates = _rigolli_axis_candidates((100, 50, 30), x_len=99, y_len=88)
        assert candidates == []

    def test_square_spatial_dims_gives_multiple(self):
        # When y_len == x_len and shape has that value in two positions,
        # multiple candidates are produced.
        candidates = _rigolli_axis_candidates((100, 64, 64), x_len=64, y_len=64)
        assert len(candidates) >= 2
        # The time axis should be 0 in all of them since 100 != 64
        for t, y, x in candidates:
            assert t == 0

    def test_ambiguous_matlab_ordering(self):
        # shape (x_len, y_len, time) = (30, 50, 200) -- MATLAB-like (x, y, t)
        # y_len=50, x_len=30 should match with time=2, y=1, x=0
        candidates = _rigolli_axis_candidates((30, 50, 200), x_len=30, y_len=50)
        assert (2, 1, 0) in candidates


# ============================================================================
# _pick_rigolli_axes
# ============================================================================


class TestPickRigolliAxes:
    """_pick_rigolli_axes(candidates, shape) -> (time, y, x)

    Selection rules:
    1. Single candidate -> return it directly.
    2. Multiple candidates -> prefer (2, 1, 0) if present (MATLAB ordering).
    3. Multiple candidates without (2, 1, 0) -> prefer time_axis==2.
    4. No candidates -> fallback: largest dim is time, remaining sorted.
    """

    def test_single_candidate(self):
        assert _pick_rigolli_axes([(0, 1, 2)], (100, 50, 30)) == (0, 1, 2)

    def test_prefers_matlab_ordering(self):
        candidates = [(0, 2, 1), (2, 1, 0), (2, 0, 1)]
        assert _pick_rigolli_axes(candidates, (30, 50, 200)) == (2, 1, 0)

    def test_prefers_time_last_without_matlab(self):
        # (2, 1, 0) not present, but time=2 entries exist.
        candidates = [(0, 1, 2), (2, 0, 1)]
        result = _pick_rigolli_axes(candidates, (30, 50, 200))
        assert result[0] == 2  # time axis is last

    def test_fallback_first_candidate(self):
        # No (2, *, *) candidates, returns the first.
        candidates = [(0, 1, 2), (1, 0, 2)]
        # Neither has time_axis==2 -- wait, (1, 0, 2) does not either.
        # Actually (0,1,2) has time_axis=0, (1,0,2) has time_axis=1.
        # So we should get the first one back.
        result = _pick_rigolli_axes(candidates, (50, 50, 50))
        assert result == (0, 1, 2)

    def test_empty_candidates_uses_largest_dim(self):
        # No spatial matches; the largest dimension becomes time.
        shape = (10, 30, 20)
        result = _pick_rigolli_axes([], shape)
        assert result[0] == 1  # axis 1 has size 30, the largest

    def test_empty_candidates_remaining_sorted(self):
        shape = (10, 30, 20)
        t, y, x = _pick_rigolli_axes([], shape)
        assert t == 1
        # Remaining axes are [0, 2] in order
        assert y == 0
        assert x == 2


# ============================================================================
# _infer_time_y_x_axes_for_rigolli
# ============================================================================


class TestInferTimeYXAxesForRigolli:
    """_infer_time_y_x_axes_for_rigolli(shape, *, x_len, y_len) -> (time, y, x)

    Full entry point: validates shape, generates candidates, picks best.
    """

    def test_unambiguous_match(self):
        result = _infer_time_y_x_axes_for_rigolli(
            (500, 80, 60), x_len=60, y_len=80
        )
        assert result == (0, 1, 2)

    def test_ambiguous_prefers_matlab(self):
        # MATLAB ordering: (x, y, t) -> axes = (2, 1, 0)
        result = _infer_time_y_x_axes_for_rigolli(
            (60, 80, 500), x_len=60, y_len=80
        )
        assert result == (2, 1, 0)

    def test_fallback_largest_dim_is_time(self):
        # No axis matches x_len or y_len, so fallback applies.
        result = _infer_time_y_x_axes_for_rigolli(
            (10, 300, 20), x_len=99, y_len=88
        )
        # Largest dim is axis 1 (300), so time=1
        assert result[0] == 1

    def test_non_3d_raises(self):
        with pytest.raises(DatasetDownloadError, match="Expected 3D"):
            _infer_time_y_x_axes_for_rigolli((100, 50), x_len=50, y_len=100)

    def test_4d_raises(self):
        with pytest.raises(DatasetDownloadError, match="Expected 3D"):
            _infer_time_y_x_axes_for_rigolli((10, 20, 30, 40), x_len=30, y_len=20)

    def test_all_dims_equal(self):
        # shape = (64, 64, 64), x_len=64, y_len=64: many candidates.
        # Should still return a valid (t, y, x) tuple.
        result = _infer_time_y_x_axes_for_rigolli(
            (64, 64, 64), x_len=64, y_len=64
        )
        t, y, x = result
        assert {t, y, x} == {0, 1, 2}

    def test_time_in_middle(self):
        # shape = (80, 500, 60), y_len=80, x_len=60 -> time at axis 1
        result = _infer_time_y_x_axes_for_rigolli(
            (80, 500, 60), x_len=60, y_len=80
        )
        assert result == (1, 0, 2)


# ============================================================================
# _emonet_frame_signal
# ============================================================================


class TestEmonetFrameSignal:
    """_emonet_frame_signal(frame, *, background, subtract_background, np)

    Returns the mean of the frame.  If subtract_background is True and
    background is not None, subtracts the background first and clips to >= 0.
    """

    def test_no_background_subtraction(self):
        frame = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = _emonet_frame_signal(
            frame, background=None, subtract_background=False, np=np
        )
        assert result == pytest.approx(2.5)

    def test_background_subtraction(self):
        frame = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        bg = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        result = _emonet_frame_signal(
            frame, background=bg, subtract_background=True, np=np
        )
        # (10-5 + 20-5 + 30-5 + 40-5) / 4 = (5+15+25+35)/4 = 20.0
        assert result == pytest.approx(20.0)

    def test_background_clipping(self):
        frame = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        bg = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        result = _emonet_frame_signal(
            frame, background=bg, subtract_background=True, np=np
        )
        # All differences are negative, clipped to 0 -> mean is 0
        assert result == pytest.approx(0.0)

    def test_background_partial_clip(self):
        frame = np.array([[10.0, 1.0]], dtype=np.float32)
        bg = np.array([[5.0, 5.0]], dtype=np.float32)
        result = _emonet_frame_signal(
            frame, background=bg, subtract_background=True, np=np
        )
        # (max(10-5,0) + max(1-5,0)) / 2 = (5 + 0) / 2 = 2.5
        assert result == pytest.approx(2.5)

    def test_subtract_false_ignores_background(self):
        frame = np.array([[10.0, 20.0]], dtype=np.float32)
        bg = np.array([[100.0, 100.0]], dtype=np.float32)
        result = _emonet_frame_signal(
            frame, background=bg, subtract_background=False, np=np
        )
        # Background is provided but not used
        assert result == pytest.approx(15.0)

    def test_background_none_with_subtract_true(self):
        frame = np.array([[3.0, 7.0]], dtype=np.float32)
        result = _emonet_frame_signal(
            frame, background=None, subtract_background=True, np=np
        )
        # background is None, so subtraction is skipped
        assert result == pytest.approx(5.0)

    def test_integer_frame_cast_to_float(self):
        frame = np.array([[2, 8]], dtype=np.int32)
        result = _emonet_frame_signal(
            frame, background=None, subtract_background=False, np=np
        )
        assert result == pytest.approx(5.0)

    def test_returns_python_float(self):
        frame = np.array([[1.0, 2.0]], dtype=np.float32)
        result = _emonet_frame_signal(
            frame, background=None, subtract_background=False, np=np
        )
        assert isinstance(result, float)


# ============================================================================
# _compute_emonet_background
# ============================================================================


class TestComputeEmonetBackground:
    """_compute_emonet_background(*, frames_dataset, n_bg, np)

    Averages the first n_bg frames (each transposed from (X, Y) to (Y, X))
    to produce a background image.  frames_dataset is an indexable sequence
    of 2D arrays with shape (X, Y).
    """

    def test_single_frame(self):
        # One frame of shape (X=3, Y=2): after transpose -> (2, 3)
        frames = [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)]
        bg = _compute_emonet_background(frames_dataset=frames, n_bg=1, np=np)
        expected = frames[0].T  # (2, 3)
        np.testing.assert_allclose(bg, expected)
        assert bg.shape == (2, 3)

    def test_average_of_two_frames(self):
        f1 = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)   # (2, 2)
        f2 = np.array([[10.0, 12.0], [14.0, 16.0]], dtype=np.float32)
        frames = [f1, f2]
        bg = _compute_emonet_background(frames_dataset=frames, n_bg=2, np=np)
        expected = (f1.T + f2.T) / 2.0
        np.testing.assert_allclose(bg, expected)

    def test_average_of_many_frames(self):
        rng = np.random.default_rng(7)
        n_bg = 5
        x_dim, y_dim = 4, 3
        frames = [rng.random((x_dim, y_dim)).astype(np.float32) for _ in range(n_bg)]
        bg = _compute_emonet_background(frames_dataset=frames, n_bg=n_bg, np=np)
        # Manual computation
        expected = np.zeros((y_dim, x_dim), dtype=np.float32)
        for f in frames:
            expected += f.T
        expected /= n_bg
        np.testing.assert_allclose(bg, expected, atol=1e-6)

    def test_output_is_float32(self):
        frames = [np.ones((3, 2), dtype=np.float64)]
        bg = _compute_emonet_background(frames_dataset=frames, n_bg=1, np=np)
        assert bg.dtype == np.float32

    def test_uses_only_first_n_bg_frames(self):
        f1 = np.ones((2, 2), dtype=np.float32) * 10.0
        f2 = np.ones((2, 2), dtype=np.float32) * 100.0  # should be ignored
        frames = [f1, f2]
        bg = _compute_emonet_background(frames_dataset=frames, n_bg=1, np=np)
        expected = f1.T
        np.testing.assert_allclose(bg, expected)

    def test_zero_frames_raises(self):
        # n_bg=0 means the loop never runs, background stays None, should raise.
        with pytest.raises(DatasetDownloadError, match="no frames"):
            _compute_emonet_background(frames_dataset=[], n_bg=0, np=np)

    def test_transpose_applied(self):
        # Verify that each frame is transposed: input (X=3, Y=2) -> output (Y=2, X=3)
        frame = np.arange(6, dtype=np.float32).reshape(3, 2)  # (X=3, Y=2)
        bg = _compute_emonet_background(frames_dataset=[frame], n_bg=1, np=np)
        assert bg.shape == (2, 3)
        np.testing.assert_allclose(bg, frame.T)
