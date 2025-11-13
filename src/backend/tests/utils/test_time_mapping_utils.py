from __future__ import annotations

import math

import pytest

from plume_nav_sim.utils.time_mapping import (
    DEFAULT_FRAME_MAPPING_POLICY,
    FrameMappingPolicy,
    map_step_to_frame,
    resolve_fps,
)


def test_resolve_fps_from_fps_only():
    assert math.isclose(resolve_fps(fps=30.0), 30.0)


def test_resolve_fps_from_timebase_only():
    # 1/25 seconds per frame → 25 fps
    assert math.isclose(resolve_fps(timebase=(1, 25)), 25.0)


def test_resolve_fps_inconsistent_inputs_raises():
    with pytest.raises(ValueError):
        resolve_fps(fps=30.0, timebase=(1, 24))


def test_resolve_fps_missing_raises():
    with pytest.raises(ValueError):
        resolve_fps()


@pytest.mark.parametrize(
    "rounding,expected",
    [
        ("nearest", [0, 1, 2, 3, 4]),
        ("floor", [0, 1, 2, 3, 4]),
        ("ceil", [0, 1, 2, 3, 4]),
    ],
)
def test_index_policy_no_timebase(rounding: str, expected: list[int]):
    # Index mapping: f(k) = k + offset; total_frames not provided
    out = [map_step_to_frame(k, fps=30.0, rounding=rounding) for k in range(5)]
    assert out == expected


def test_offset_frames_applied_before_rounding():
    # With 0.6 offset and nearest-half-up rounding, indices shift by +1
    out = [
        map_step_to_frame(k, fps=30.0, offset_frames=0.6, rounding="nearest")
        for k in range(5)
    ]
    assert out == [1, 2, 3, 4, 5]


def test_time_mapping_with_steps_per_second():
    # fps=10, steps_per_second=5 → f(k) ≈ 2k
    out = [
        map_step_to_frame(
            k,
            fps=10.0,
            steps_per_second=5.0,
            rounding="nearest",
        )
        for k in range(5)
    ]
    assert out == [0, 2, 4, 6, 8]


def test_time_mapping_with_offset_and_rounding():
    # Apply an offset of 1 frame
    out = [
        map_step_to_frame(
            k,
            fps=10.0,
            steps_per_second=5.0,
            offset_frames=1.0,
            rounding="nearest",
        )
        for k in range(5)
    ]
    assert out == [1, 3, 5, 7, 9]


def test_policy_clamp_and_wrap_with_total_frames():
    total = 3
    # For k=0..5 under index mapping with offset 0, indices 0..5
    wrap = [
        map_step_to_frame(
            k, fps=30.0, total_frames=total, policy=FrameMappingPolicy.WRAP
        )
        for k in range(6)
    ]
    clamp = [
        map_step_to_frame(
            k, fps=30.0, total_frames=total, policy=FrameMappingPolicy.CLAMP
        )
        for k in range(6)
    ]
    assert wrap == [0, 1, 2, 0, 1, 2]
    assert clamp == [0, 1, 2, 2, 2, 2]


def test_policy_index_raises_on_oob():
    with pytest.raises(IndexError):
        map_step_to_frame(5, fps=30.0, total_frames=3, policy=FrameMappingPolicy.INDEX)


def test_default_policy_is_wrap():
    assert DEFAULT_FRAME_MAPPING_POLICY == FrameMappingPolicy.WRAP


def test_determinism_across_calls():
    # Repeated calls with same inputs produce identical sequence
    seq1 = [map_step_to_frame(k, fps=24.0, total_frames=5) for k in range(20)]
    seq2 = [map_step_to_frame(k, fps=24.0, total_frames=5) for k in range(20)]
    assert seq1 == seq2
