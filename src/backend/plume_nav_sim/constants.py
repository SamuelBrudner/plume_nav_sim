"""Shared constants for plume_nav_sim (non-core types)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.types import (
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
    MOVEMENT_VECTORS,
)

PACKAGE_NAME = "plume_nav_sim"
PACKAGE_VERSION = "0.1.0"
ENVIRONMENT_ID = "PlumeNav-v0"

MIN_GRID_SIZE = (1, 1)
MAX_GRID_SIZE = (2048, 2048)

CONCENTRATION_RANGE = (0.0, 1.0)
GAUSSIAN_PRECISION = 1e-6
DISTANCE_PRECISION = 1e-12
MIN_PLUME_SIGMA = 0.1
MAX_PLUME_SIGMA = 100.0
STATIC_GAUSSIAN_MODEL_TYPE = "static_gaussian"
DEFAULT_PLUME_MODEL_TYPE = STATIC_GAUSSIAN_MODEL_TYPE
PLUME_MODEL_TYPES = [STATIC_GAUSSIAN_MODEL_TYPE]

REWARD_GOAL_REACHED = 1.0
REWARD_DEFAULT = 0.0

FIELD_DTYPE = np.float32
OBSERVATION_DTYPE = np.float32
RGB_DTYPE = np.uint8

MEMORY_LIMIT_TOTAL_MB = 50
MEMORY_LIMIT_PLUME_FIELD_MB = 40

PERFORMANCE_TARGET_STEP_LATENCY_MS = 1.0
PERFORMANCE_TARGET_RGB_RENDER_MS = 5.0
PERFORMANCE_TARGET_HUMAN_RENDER_MS = 33.0
PERFORMANCE_TARGET_PLUME_GENERATION_MS = 10.0
PERFORMANCE_TARGET_EPISODE_RESET_MS = 10.0
PERFORMANCE_TARGET_BOUNDARY_ENFORCEMENT_MS = 0.5
PERFORMANCE_BENCHMARK_ITERATIONS = 100

LOG_LEVEL_DEFAULT = "INFO"
COMPONENT_NAMES = [
    "environment",
    "plume_model",
    "rendering",
    "concentration_field",
    "registration",
    "plume",
    "render",
    "envs",
]

SEED_MIN_VALUE = 0
SEED_MAX_VALUE = 2**32 - 1
VALID_SEED_TYPES = [int, np.integer]

VALIDATION_ERROR_MESSAGES = {
    "invalid_action": "Invalid action provided for action space constraints.",
    "invalid_coordinates": "Invalid coordinates provided for grid bounds.",
    "invalid_grid_size": "Invalid grid size provided; must be positive dimensions.",
    "invalid_sigma": "Invalid plume sigma provided; must be within valid range.",
}

AGENT_MARKER_COLOR = (255, 0, 0)
SOURCE_MARKER_COLOR = (255, 255, 255)
AGENT_MARKER_SIZE = (3, 3)
SOURCE_MARKER_SIZE = (5, 5)
PIXEL_VALUE_MIN = 0
PIXEL_VALUE_MAX = 255
SUPPORTED_RENDER_MODES = ["rgb_array", "human"]
BACKEND_PRIORITY_LIST = [
    "module://ipympl.backend_nbagg",
    "TkAgg",
    "Qt5Agg",
    "Agg",
]
MATPLOTLIB_DEFAULT_FIGSIZE = (8, 8)


def get_default_environment_constants() -> dict[str, Any]:
    return {
        "DEFAULT_GRID_SIZE": DEFAULT_GRID_SIZE,
        "MIN_GRID_SIZE": MIN_GRID_SIZE,
        "MAX_GRID_SIZE": MAX_GRID_SIZE,
        "DEFAULT_SOURCE_LOCATION": DEFAULT_SOURCE_LOCATION,
        "DEFAULT_PLUME_SIGMA": DEFAULT_PLUME_SIGMA,
        "DEFAULT_GOAL_RADIUS": DEFAULT_GOAL_RADIUS,
        "DEFAULT_MAX_STEPS": DEFAULT_MAX_STEPS,
    }


def get_plume_model_constants() -> dict[str, Any]:
    return {
        "CONCENTRATION_RANGE": CONCENTRATION_RANGE,
        "GAUSSIAN_PRECISION": GAUSSIAN_PRECISION,
        "DISTANCE_PRECISION": DISTANCE_PRECISION,
        "MIN_PLUME_SIGMA": MIN_PLUME_SIGMA,
        "MAX_PLUME_SIGMA": MAX_PLUME_SIGMA,
        "STATIC_GAUSSIAN_MODEL_TYPE": STATIC_GAUSSIAN_MODEL_TYPE,
        "DEFAULT_PLUME_MODEL_TYPE": DEFAULT_PLUME_MODEL_TYPE,
        "PLUME_MODEL_TYPES": list(PLUME_MODEL_TYPES),
    }


def get_action_space_constants() -> dict[str, Any]:
    return {
        "ACTION_UP": ACTION_UP,
        "ACTION_RIGHT": ACTION_RIGHT,
        "ACTION_LEFT": ACTION_LEFT,
        "ACTION_SPACE_SIZE": ACTION_SPACE_SIZE,
        "MOVEMENT_VECTORS": dict(MOVEMENT_VECTORS),
    }


def get_rendering_constants() -> dict[str, Any]:
    return {
        "supported_modes": list(SUPPORTED_RENDER_MODES),
        "agent_marker_color": AGENT_MARKER_COLOR,
        "source_marker_color": SOURCE_MARKER_COLOR,
        "agent_marker_size": AGENT_MARKER_SIZE,
        "source_marker_size": SOURCE_MARKER_SIZE,
        "pixel_value_min": PIXEL_VALUE_MIN,
        "pixel_value_max": PIXEL_VALUE_MAX,
        "rgb_dtype": RGB_DTYPE,
    }


def get_performance_constants() -> dict[str, Any]:
    return {
        "tracking_enabled": True,
        "step_latency_target_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
        "rgb_render_target_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
        "human_render_target_ms": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
        "plume_generation_target_ms": PERFORMANCE_TARGET_PLUME_GENERATION_MS,
        "episode_reset_target_ms": PERFORMANCE_TARGET_EPISODE_RESET_MS,
        "boundary_enforcement_target_ms": PERFORMANCE_TARGET_BOUNDARY_ENFORCEMENT_MS,
        "memory_limits_mb": {
            "total": MEMORY_LIMIT_TOTAL_MB,
            "plume_field": MEMORY_LIMIT_PLUME_FIELD_MB,
        },
        "optimization_thresholds": {
            "step_latency_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
            "render_latency_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
        },
        "benchmark_iterations": PERFORMANCE_BENCHMARK_ITERATIONS,
    }


def get_validation_constants() -> dict[str, Any]:
    return {
        "distance_precision": DISTANCE_PRECISION,
        "gaussian_precision": GAUSSIAN_PRECISION,
        "sigma_range": (MIN_PLUME_SIGMA, MAX_PLUME_SIGMA),
        "grid_size_limits": (MIN_GRID_SIZE, MAX_GRID_SIZE),
        "coordinate_bounds_checking": True,
        "error_messages": VALIDATION_ERROR_MESSAGES,
        "seed_validation": {"min": SEED_MIN_VALUE, "max": SEED_MAX_VALUE},
        "memory_validation_mb": MEMORY_LIMIT_TOTAL_MB,
        "action_validation_range": list(MOVEMENT_VECTORS.keys()),
    }


def get_testing_constants() -> dict[str, Any]:
    return {
        "test_grid_size": (32, 32),
        "test_source_location": (16, 16),
        "test_max_steps": 100,
        "test_tolerance": 1e-6,
        "reproducibility_seeds": [0, 1, 2, 3, 4],
        "performance_test_iterations": PERFORMANCE_BENCHMARK_ITERATIONS,
        "expected_results": {},
    }


def validate_constant_consistency(
    strict_mode: bool = False,
) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {
        "checks": [],
        "errors": [],
        "warnings": [],
        "metadata": {
            "strict_mode": bool(strict_mode),
            "package": PACKAGE_NAME,
            "version": PACKAGE_VERSION,
        },
    }
    ok = True
    report["checks"].append("sigma_bounds")
    if not (MIN_PLUME_SIGMA <= DEFAULT_PLUME_SIGMA <= MAX_PLUME_SIGMA):
        ok = False
        report["errors"].append("DEFAULT_PLUME_SIGMA out of bounds")
    report["checks"].append("grid_size_shape")
    if not isinstance(DEFAULT_GRID_SIZE, tuple) or len(DEFAULT_GRID_SIZE) != 2:
        ok = False
        report["errors"].append("DEFAULT_GRID_SIZE must be length 2 tuple")
    report["checks"].append("source_within_grid")
    try:
        sx, sy = DEFAULT_SOURCE_LOCATION
        gx, gy = DEFAULT_GRID_SIZE
        if not (0 <= sx < gx and 0 <= sy < gy):
            ok = False
            report["errors"].append("DEFAULT_SOURCE_LOCATION outside DEFAULT_GRID_SIZE")
    except Exception:
        ok = False
        report["errors"].append("DEFAULT_SOURCE_LOCATION invalid format")
    if strict_mode and PERFORMANCE_TARGET_STEP_LATENCY_MS <= 0:
        ok = False
        report["errors"].append("Non-positive step latency target")
    return ok, report


__all__ = [
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_SPACE_SIZE",
    "ACTION_UP",
    "AGENT_MARKER_COLOR",
    "AGENT_MARKER_SIZE",
    "BACKEND_PRIORITY_LIST",
    "COMPONENT_NAMES",
    "CONCENTRATION_RANGE",
    "DEFAULT_GOAL_RADIUS",
    "DEFAULT_GRID_SIZE",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_PLUME_SIGMA",
    "DEFAULT_PLUME_MODEL_TYPE",
    "DEFAULT_SOURCE_LOCATION",
    "DISTANCE_PRECISION",
    "ENVIRONMENT_ID",
    "FIELD_DTYPE",
    "GAUSSIAN_PRECISION",
    "LOG_LEVEL_DEFAULT",
    "MATPLOTLIB_DEFAULT_FIGSIZE",
    "MAX_GRID_SIZE",
    "MAX_PLUME_SIGMA",
    "MEMORY_LIMIT_PLUME_FIELD_MB",
    "MEMORY_LIMIT_TOTAL_MB",
    "MIN_GRID_SIZE",
    "MIN_PLUME_SIGMA",
    "MOVEMENT_VECTORS",
    "OBSERVATION_DTYPE",
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "PERFORMANCE_BENCHMARK_ITERATIONS",
    "PERFORMANCE_TARGET_BOUNDARY_ENFORCEMENT_MS",
    "PERFORMANCE_TARGET_EPISODE_RESET_MS",
    "PERFORMANCE_TARGET_HUMAN_RENDER_MS",
    "PERFORMANCE_TARGET_PLUME_GENERATION_MS",
    "PERFORMANCE_TARGET_RGB_RENDER_MS",
    "PERFORMANCE_TARGET_STEP_LATENCY_MS",
    "PIXEL_VALUE_MAX",
    "PIXEL_VALUE_MIN",
    "PLUME_MODEL_TYPES",
    "REWARD_DEFAULT",
    "REWARD_GOAL_REACHED",
    "RGB_DTYPE",
    "SEED_MAX_VALUE",
    "SEED_MIN_VALUE",
    "SOURCE_MARKER_COLOR",
    "SOURCE_MARKER_SIZE",
    "STATIC_GAUSSIAN_MODEL_TYPE",
    "SUPPORTED_RENDER_MODES",
    "VALIDATION_ERROR_MESSAGES",
    "VALID_SEED_TYPES",
    "get_action_space_constants",
    "get_default_environment_constants",
    "get_performance_constants",
    "get_plume_model_constants",
    "get_rendering_constants",
    "get_testing_constants",
    "get_validation_constants",
    "validate_constant_consistency",
]
