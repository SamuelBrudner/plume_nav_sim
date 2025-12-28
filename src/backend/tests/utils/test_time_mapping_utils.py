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


@pytest.mark.parametrize("timebase", [(0, 25), (-1, 25), (1, 0)])
def test_resolve_fps_invalid_timebase_raises(timebase: tuple[int, int]):
    with pytest.raises(ValueError):
        resolve_fps(timebase=timebase)


def test_resolve_fps_negative_fps_raises():
    with pytest.raises(ValueError):
        resolve_fps(fps=-30.0)


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


def test_rounding_policies_on_half_values():
    # step=3, fps=30, steps_per_second=60 -> f_float = 1.5
    base_kwargs = {"step": 3, "fps": 30.0, "steps_per_second": 60.0}

    assert map_step_to_frame(**base_kwargs, rounding="floor") == 1
    assert map_step_to_frame(**base_kwargs, rounding="nearest") == 2
    assert map_step_to_frame(**base_kwargs, rounding="ceil") == 2


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


def test_negative_step_raises_value_error():
    with pytest.raises(ValueError):
        map_step_to_frame(-1, fps=30.0)


def test_non_positive_total_frames_raises_value_error():
    for total in (0, -5):
        with pytest.raises(ValueError):
            map_step_to_frame(0, fps=30.0, total_frames=total)


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


def test_negative_offset_with_known_total_frames_respects_policy():
    T = 5

    # Negative offset with wrap should wrap around the end of the range
    assert (
        map_step_to_frame(
            0,
            fps=30.0,
            total_frames=T,
            offset_frames=-1.0,
            policy=FrameMappingPolicy.WRAP,
        )
        == T - 1
    )

    # Clamp policy should clamp negative indices to zero
    assert (
        map_step_to_frame(
            0,
            fps=30.0,
            total_frames=T,
            offset_frames=-1.0,
            policy=FrameMappingPolicy.CLAMP,
        )
        == 0
    )

    # Index policy should raise on negative index
    with pytest.raises(IndexError):
        map_step_to_frame(
            0,
            fps=30.0,
            total_frames=T,
            offset_frames=-1.0,
            policy=FrameMappingPolicy.INDEX,
        )


def test_policy_index_raises_on_oob():
    with pytest.raises(IndexError):
        map_step_to_frame(5, fps=30.0, total_frames=3, policy=FrameMappingPolicy.INDEX)


def test_negative_offset_without_total_frames_clamps_to_zero():
    idx = map_step_to_frame(0, fps=30.0, offset_frames=-10.5)
    assert idx == 0


def test_default_policy_is_wrap():
    assert DEFAULT_FRAME_MAPPING_POLICY == FrameMappingPolicy.WRAP


def test_determinism_across_calls():
    # Repeated calls with same inputs produce identical sequence
    seq1 = [map_step_to_frame(k, fps=24.0, total_frames=5) for k in range(20)]
    seq2 = [map_step_to_frame(k, fps=24.0, total_frames=5) for k in range(20)]
    assert seq1 == seq2


def test_non_integer_fps_with_steps_per_second():
    fps = 2.5
    steps_per_second = 2.0
    total_frames = 8

    idxs = [
        map_step_to_frame(
            k,
            fps=fps,
            steps_per_second=steps_per_second,
            total_frames=total_frames,
            rounding="nearest",
            policy=FrameMappingPolicy.CLAMP,
        )
        for k in range(6)
    ]
    assert idxs == [0, 1, 3, 4, 5, 6]
