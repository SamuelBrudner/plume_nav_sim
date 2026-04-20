from __future__ import annotations

import re

import numpy as np

from plume_nav_sim.core.constants import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SPACE_SIZE,
    ACTION_UP,
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLUME_SIGMA,
    DEFAULT_SOURCE_LOCATION,
    ENVIRONMENT_ID,
    FIELD_DTYPE,
    MAX_GRID_SIZE,
    MAX_PLUME_SIGMA,
    MEMORY_LIMIT_TOTAL_MB,
    MIN_GRID_SIZE,
    MIN_PLUME_SIGMA,
    MOVEMENT_VECTORS,
    OBSERVATION_DTYPE,
    PACKAGE_NAME,
    PACKAGE_VERSION,
    PERFORMANCE_TARGET_RGB_RENDER_MS,
    PERFORMANCE_TARGET_STEP_LATENCY_MS,
    REWARD_DEFAULT,
    REWARD_GOAL_REACHED,
    SUPPORTED_RENDER_MODES,
    VALIDATION_ERROR_MESSAGES,
    get_action_space_constants,
    get_default_environment_constants,
    get_performance_constants,
    get_plume_model_constants,
    get_rendering_constants,
    get_testing_constants,
    get_validation_constants,
    validate_constant_consistency,
)
from plume_nav_sim.core.types import Action, RenderMode


def test_package_metadata_and_default_geometry_are_consistent() -> None:
    assert PACKAGE_NAME == "plume_nav_sim"
    assert re.fullmatch(r"\d+\.\d+\.\d+", PACKAGE_VERSION)
    assert re.fullmatch(r"PlumeNav(?:-.+)?-v\d+", ENVIRONMENT_ID)

    assert DEFAULT_GRID_SIZE == (128, 128)
    assert MIN_GRID_SIZE[0] <= DEFAULT_GRID_SIZE[0] <= MAX_GRID_SIZE[0]
    assert MIN_GRID_SIZE[1] <= DEFAULT_GRID_SIZE[1] <= MAX_GRID_SIZE[1]
    assert DEFAULT_SOURCE_LOCATION == (64, 64)
    assert 0 <= DEFAULT_SOURCE_LOCATION[0] < DEFAULT_GRID_SIZE[0]
    assert 0 <= DEFAULT_SOURCE_LOCATION[1] < DEFAULT_GRID_SIZE[1]
    assert DEFAULT_MAX_STEPS > 0
    assert DEFAULT_GOAL_RADIUS >= 0


def test_action_constants_match_enum_and_movement_vectors() -> None:
    assert [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT] == [0, 1, 2, 3]
    assert ACTION_SPACE_SIZE == 4
    assert MOVEMENT_VECTORS == {
        ACTION_UP: (0, 1),
        ACTION_RIGHT: (1, 0),
        ACTION_DOWN: (0, -1),
        ACTION_LEFT: (-1, 0),
    }
    assert ACTION_UP == Action.UP.value
    assert ACTION_RIGHT == Action.RIGHT.value
    assert ACTION_DOWN == Action.DOWN.value
    assert ACTION_LEFT == Action.LEFT.value


def test_dtype_and_reward_constants_support_core_runtime_expectations() -> None:
    assert FIELD_DTYPE == np.float32
    assert OBSERVATION_DTYPE == np.float32
    assert np.dtype(FIELD_DTYPE).itemsize == 4
    assert np.dtype(OBSERVATION_DTYPE).itemsize == 4

    assert MIN_PLUME_SIGMA <= DEFAULT_PLUME_SIGMA <= MAX_PLUME_SIGMA
    assert REWARD_GOAL_REACHED > REWARD_DEFAULT
    assert PERFORMANCE_TARGET_STEP_LATENCY_MS > 0
    assert PERFORMANCE_TARGET_RGB_RENDER_MS > 0
    assert MEMORY_LIMIT_TOTAL_MB > 0


def test_render_modes_and_validation_messages_match_public_types() -> None:
    assert SUPPORTED_RENDER_MODES == [
        RenderMode.RGB_ARRAY.value,
        RenderMode.HUMAN.value,
    ]
    assert VALIDATION_ERROR_MESSAGES.keys() >= {
        "invalid_action",
        "invalid_coordinates",
        "invalid_grid_size",
        "invalid_sigma",
    }


def test_constant_factory_helpers_return_expected_relationships() -> None:
    env = get_default_environment_constants()
    plume = get_plume_model_constants()
    actions = get_action_space_constants()
    rendering = get_rendering_constants()
    performance = get_performance_constants()
    validation = get_validation_constants()
    testing = get_testing_constants()

    assert env["DEFAULT_GRID_SIZE"] == DEFAULT_GRID_SIZE
    assert env["DEFAULT_SOURCE_LOCATION"] == DEFAULT_SOURCE_LOCATION
    assert plume["MIN_PLUME_SIGMA"] == MIN_PLUME_SIGMA
    assert plume["MAX_PLUME_SIGMA"] == MAX_PLUME_SIGMA
    assert actions["MOVEMENT_VECTORS"] == MOVEMENT_VECTORS
    assert rendering["supported_modes"] == SUPPORTED_RENDER_MODES
    assert performance["step_latency_target_ms"] == PERFORMANCE_TARGET_STEP_LATENCY_MS
    assert validation["sigma_range"] == (MIN_PLUME_SIGMA, MAX_PLUME_SIGMA)
    assert testing["test_grid_size"] == (32, 32)
    assert testing["test_source_location"] == (16, 16)

    env["DEFAULT_GRID_SIZE"] = (1, 1)
    actions["MOVEMENT_VECTORS"][ACTION_UP] = (9, 9)
    assert get_default_environment_constants()["DEFAULT_GRID_SIZE"] == DEFAULT_GRID_SIZE
    assert get_action_space_constants()["MOVEMENT_VECTORS"] == MOVEMENT_VECTORS


def test_validate_constant_consistency_reports_current_defaults() -> None:
    is_valid, report = validate_constant_consistency(strict_mode=False)
    strict_valid, strict_report = validate_constant_consistency(strict_mode=True)

    assert is_valid is True
    assert strict_valid is True
    assert report["errors"] == []
    assert strict_report["errors"] == []
    assert report["metadata"]["package"] == PACKAGE_NAME
    assert strict_report["metadata"]["strict_mode"] is True
    assert report["checks"][:3] == [
        "sigma_bounds",
        "grid_size_shape",
        "source_within_grid",
    ]
