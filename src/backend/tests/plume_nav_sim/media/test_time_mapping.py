from __future__ import annotations

import pytest

from plume_nav_sim.media import (
    DEFAULT_ROUNDING,
    DEFAULT_STEP_POLICY,
    ROUND_CEIL,
    ROUND_FLOOR,
    ROUND_NEAREST,
    STEP_POLICY_CLAMP,
    STEP_POLICY_INDEX,
    STEP_POLICY_WRAP,
    VideoTimebase,
    map_step_to_frame,
    video_timebase_from_attrs,
)


def test_timebase_validation_and_fps_value():
    # Missing both
    with pytest.raises(ValueError):
        VideoTimebase()

    # Negative fps
    with pytest.raises(ValueError):
        VideoTimebase(fps=-1)

    # Rational only
    tb = VideoTimebase(timebase_numer=30000, timebase_denom=1001)
    assert 29.0 < tb.fps_value < 30.0

    # Both, consistent
    tb2 = VideoTimebase(fps=30.0, timebase_numer=30000, timebase_denom=1000)
    assert tb2.fps_value == pytest.approx(30.0)

    # Both, inconsistent
    with pytest.raises(ValueError):
        VideoTimebase(fps=25.0, timebase_numer=30000, timebase_denom=1001)

    # From attrs mapping
    tb3 = video_timebase_from_attrs({"fps": 24.0})
    assert tb3.fps_value == pytest.approx(24.0)

    tb4 = video_timebase_from_attrs({"timebase_numer": 30000, "timebase_denom": 1001})
    assert tb4.fps_value == pytest.approx(30000 / 1001)

    with pytest.raises(ValueError):
        video_timebase_from_attrs({})


def test_default_mapping_equals_index_when_step_hz_matches_fps():
    tb = VideoTimebase(fps=30.0)

    # Default: step_hz=None -> uses fps
    # With clamp boundary
    T = 5
    assert map_step_to_frame(step=0, total_frames=T, timebase=tb) == 0
    assert map_step_to_frame(step=1, total_frames=T, timebase=tb) == 1
    assert map_step_to_frame(step=4, total_frames=T, timebase=tb) == 4
    # Clamp at end
    assert (
        map_step_to_frame(
            step=5, total_frames=T, timebase=tb, boundary=STEP_POLICY_CLAMP
        )
        == 4
    )

    # Wrap at end
    assert (
        map_step_to_frame(
            step=5, total_frames=T, timebase=tb, boundary=STEP_POLICY_WRAP
        )
        == 0
    )
    assert (
        map_step_to_frame(
            step=6, total_frames=T, timebase=tb, boundary=STEP_POLICY_WRAP
        )
        == 1
    )

    # Index policy raises when OOB
    with pytest.raises(IndexError):
        map_step_to_frame(
            step=5, total_frames=T, timebase=tb, boundary=STEP_POLICY_INDEX
        )


def test_mapping_with_step_rate_and_rounding():
    tb = VideoTimebase(fps=30.0)
    T = 20

    # step_hz slower than fps -> multiple frames per step
    # fractional = step * (fps/step_hz) = step * 2.0
    assert map_step_to_frame(step=1, total_frames=T, timebase=tb, step_hz=15.0) == 2
    assert map_step_to_frame(step=2, total_frames=T, timebase=tb, step_hz=15.0) == 4

    # step_hz faster than fps -> may skip repeating indices depending on rounding
    # fractional at step=3: 3 * (30/60) = 1.5
    assert (
        map_step_to_frame(
            step=3, total_frames=T, timebase=tb, step_hz=60.0, rounding=ROUND_FLOOR
        )
        == 1
    )
    assert (
        map_step_to_frame(
            step=3, total_frames=T, timebase=tb, step_hz=60.0, rounding=ROUND_NEAREST
        )
        == 2
    )
    assert (
        map_step_to_frame(
            step=3, total_frames=T, timebase=tb, step_hz=60.0, rounding=ROUND_CEIL
        )
        == 2
    )


def test_offset_and_boundary_policies_small_T():
    tb = VideoTimebase(timebase_numer=24, timebase_denom=1)  # 24 fps
    T = 3

    # Simple sequence without offset, wrap
    seq = [
        map_step_to_frame(
            step=k, total_frames=T, timebase=tb, boundary=STEP_POLICY_WRAP
        )
        for k in range(8)
    ]
    assert seq == [0, 1, 2, 0, 1, 2, 0, 1]

    # With +1 offset, wrap
    seq_off = [
        map_step_to_frame(
            step=k,
            total_frames=T,
            timebase=tb,
            boundary=STEP_POLICY_WRAP,
            offset_frames=1,
        )
        for k in range(6)
    ]
    assert seq_off == [1, 2, 0, 1, 2, 0]

    # Clamp at end with offset
    assert (
        map_step_to_frame(
            step=5,
            total_frames=T,
            timebase=tb,
            boundary=STEP_POLICY_CLAMP,
            offset_frames=1,
        )
        == 2
    )


def test_determinism_same_inputs_same_outputs():
    tb = VideoTimebase(fps=25.0)
    T = 7
    params = dict(
        step_hz=50.0, offset_frames=2, boundary=STEP_POLICY_WRAP, rounding=ROUND_NEAREST
    )

    seq1 = [
        map_step_to_frame(step=k, total_frames=T, timebase=tb, **params)
        for k in range(20)
    ]
    seq2 = [
        map_step_to_frame(step=k, total_frames=T, timebase=tb, **params)
        for k in range(20)
    ]
    assert seq1 == seq2
