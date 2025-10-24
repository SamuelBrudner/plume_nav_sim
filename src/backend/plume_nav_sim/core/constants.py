"""Core constants used throughout the `plume_nav_sim` package.

Primitive numerical values live directly in this module, while higher level
metadata (package identifiers, performance thresholds, testing seeds) are
loaded from `config/constants.yaml` for easier maintenance and reproducibility.

The factory helpers at the bottom of the file intentionally return minimal
dictionary payloads because several tests assert their exact shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "constants.yaml"
)

_DEFAULT_CONFIG: Dict[str, Any] = {
    "package": {
        "name": "plume_nav_sim",
        "version": "0.0.0",
        # Backward-compatible default ID expected by registration tests
        # Registration suites enforce the '-v0' suffix convention.
        "environment_id": "PlumeNav-Sim-v0",
    },
    "performance": {
        "tracking_enabled": True,
        "target_step_latency_ms": 1.0,
        "target_rgb_render_ms": 5.0,
        "target_human_render_ms": 33.0,
        "target_episode_reset_ms": 10.0,
        "target_plume_generation_ms": 10.0,
        "boundary_enforcement_ms": 0.5,
    },
    "testing": {
        "default_seeds": [42, 123, 456, 789, 999],
    },
}


def _load_constants_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return _DEFAULT_CONFIG

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return _DEFAULT_CONFIG

    merged = _DEFAULT_CONFIG.copy()
    for key, value in data.items():
        if isinstance(value, dict):
            merged.setdefault(key, {}).update(value)
        else:
            merged[key] = value
    return merged


_CONFIG = _load_constants_config()


PACKAGE_NAME = _CONFIG["package"].get("name", _DEFAULT_CONFIG["package"]["name"])
PACKAGE_VERSION = _CONFIG["package"].get(
    "version", _DEFAULT_CONFIG["package"]["version"]
)
ENVIRONMENT_ID = _CONFIG["package"].get(
    "environment_id", _DEFAULT_CONFIG["package"]["environment_id"]
)


# Environment defaults
DEFAULT_GRID_SIZE = (128, 128)
MIN_GRID_SIZE = (1, 1)
MAX_GRID_SIZE = (1024, 1024)
DEFAULT_SOURCE_LOCATION = (64, 64)
DEFAULT_PLUME_SIGMA = 12.0
DEFAULT_GOAL_RADIUS = float(np.finfo(np.float32).eps)
DEFAULT_MAX_STEPS = 1000


# Action space
ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_SPACE_SIZE = 4
MOVEMENT_VECTORS = {
    ACTION_UP: (0, 1),
    ACTION_RIGHT: (1, 0),
    ACTION_DOWN: (0, -1),
    ACTION_LEFT: (-1, 0),
}


# Rewards and precision
REWARD_GOAL_REACHED = 1.0
REWARD_DEFAULT = 0.0
CONCENTRATION_RANGE = (0.0, 1.0)
GAUSSIAN_PRECISION = 1e-6
DISTANCE_PRECISION = 1e-6
NUMERIC_PRECISION_TOLERANCE = 1e-6
MIN_PLUME_SIGMA = 0.1
MAX_PLUME_SIGMA = 100.0
STATIC_GAUSSIAN_MODEL_TYPE = "static_gaussian"

# Boundary validation and caching
BOUNDARY_VALIDATION_CACHE_SIZE = 1000

# Plume model registry identifiers
DEFAULT_PLUME_MODEL_TYPE = STATIC_GAUSSIAN_MODEL_TYPE
PLUME_MODEL_TYPES = [STATIC_GAUSSIAN_MODEL_TYPE]

# Seeding
SEED_MIN_VALUE = 0
SEED_MAX_VALUE = (
    2**32 - 1
)  # Upper bound used modulo in utilities; tests may bound to 2**31-1
VALID_SEED_TYPES = [int, np.integer]


# Data types
FIELD_DTYPE = np.float32
OBSERVATION_DTYPE = np.float32
RGB_DTYPE = np.uint8
COORDINATE_DTYPE = np.int32


# Rendering
AGENT_MARKER_COLOR = (255, 0, 0)
SOURCE_MARKER_COLOR = (255, 255, 255)
AGENT_MARKER_SIZE = (3, 3)
SOURCE_MARKER_SIZE = (5, 5)
PIXEL_VALUE_MIN = 0
PIXEL_VALUE_MAX = 255
SUPPORTED_RENDER_MODES = ["rgb_array", "human"]

# Rendering backend integration (used by matplotlib renderer and configs)
# Prefer ipympl in notebooks, then desktop GUI backends, then headless Agg
BACKEND_PRIORITY_LIST = [
    "module://ipympl.backend_nbagg",
    "TkAgg",
    "Qt5Agg",
    "Agg",
]
MATPLOTLIB_DEFAULT_FIGSIZE = (8, 8)


# Performance / resource limits
PERFORMANCE_TRACKING_ENABLED = _CONFIG["performance"].get(
    "tracking_enabled", _DEFAULT_CONFIG["performance"]["tracking_enabled"]
)
PERFORMANCE_TARGET_STEP_LATENCY_MS = _CONFIG["performance"].get(
    "target_step_latency_ms",
    _DEFAULT_CONFIG["performance"]["target_step_latency_ms"],
)
PERFORMANCE_TARGET_RGB_RENDER_MS = _CONFIG["performance"].get(
    "target_rgb_render_ms",
    _DEFAULT_CONFIG["performance"]["target_rgb_render_ms"],
)
PERFORMANCE_TARGET_HUMAN_RENDER_MS = _CONFIG["performance"].get(
    "target_human_render_ms",
    _DEFAULT_CONFIG["performance"]["target_human_render_ms"],
)
PERFORMANCE_TARGET_EPISODE_RESET_MS = _CONFIG["performance"].get(
    "target_episode_reset_ms",
    _DEFAULT_CONFIG["performance"]["target_episode_reset_ms"],
)
PERFORMANCE_TARGET_PLUME_GENERATION_MS = _CONFIG["performance"].get(
    "target_plume_generation_ms",
    _DEFAULT_CONFIG["performance"]["target_plume_generation_ms"],
)
BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS = _CONFIG["performance"].get(
    "boundary_enforcement_ms",
    _DEFAULT_CONFIG["performance"]["boundary_enforcement_ms"],
)
MEMORY_LIMIT_TOTAL_MB = 50
MEMORY_LIMIT_PLUME_FIELD_MB = 40

# Logging defaults and component identifiers (used by utils.logging and logging package)
LOG_LEVEL_DEFAULT = "INFO"
COMPONENT_NAMES = [
    "environment",
    "plume_model",
    "rendering",
    "action_processor",
    "reward_calculator",
    "state_manager",
    "boundary_enforcer",
    "episode_manager",
    "concentration_field",
    "utils",
]


# Seed defaults
DEFAULT_TEST_SEEDS = list(_CONFIG["testing"].get("default_seeds", [])) or list(
    _DEFAULT_CONFIG["testing"]["default_seeds"]
)

# Aggregate testing constants for config helpers
TESTING_CONSTANTS: Dict[str, Any] = {
    "default_seeds": list(DEFAULT_TEST_SEEDS),
}


VALIDATION_ERROR_MESSAGES = {
    "invalid_action": "Action must be integer in range [0, 3] for cardinal directions",
    "invalid_coordinates": "Coordinates must be non-negative integers within grid bounds",
    "invalid_grid_size": "Grid size must be tuple of positive integers within supported range",
    "invalid_sigma": "Sigma must be positive float within configured limits",
    "invalid_seed": "Seed must be 32-bit integer or None",
    "memory_limit_exceeded": f"Configuration exceeds memory limit of {MEMORY_LIMIT_TOTAL_MB} MB",
    "performance_target_missed": "Operation exceeded performance target threshold",
    "invalid_render_mode": "Render mode must be one of the supported modes",
}


def get_default_environment_constants() -> Dict[str, Any]:
    """Returns a dictionary of environment constants."""
    return {
        "DEFAULT_GRID_SIZE": DEFAULT_GRID_SIZE,
        "MIN_GRID_SIZE": MIN_GRID_SIZE,
        "MAX_GRID_SIZE": MAX_GRID_SIZE,
        "DEFAULT_SOURCE_LOCATION": DEFAULT_SOURCE_LOCATION,
        "DEFAULT_PLUME_SIGMA": DEFAULT_PLUME_SIGMA,
        "DEFAULT_GOAL_RADIUS": DEFAULT_GOAL_RADIUS,
        "DEFAULT_MAX_STEPS": DEFAULT_MAX_STEPS,
    }


def get_action_space_constants() -> Dict[str, Any]:
    """Returns a dictionary of action space constants."""
    return {
        "ACTION_UP": ACTION_UP,
        "ACTION_RIGHT": ACTION_RIGHT,
        "ACTION_LEFT": ACTION_LEFT,
        "ACTION_SPACE_SIZE": ACTION_SPACE_SIZE,
        "MOVEMENT_VECTORS": MOVEMENT_VECTORS,
    }


def get_plume_model_constants() -> Dict[str, Any]:
    """Returns a dictionary of plume model constants."""
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


def get_rendering_constants() -> Dict[str, Any]:
    """Return rendering-related constants with test-expected schema."""
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


def get_performance_constants() -> Dict[str, Any]:
    """Return performance constants with monitoring-ready schema."""
    return {
        "tracking_enabled": PERFORMANCE_TRACKING_ENABLED,
        "step_latency_target_ms": PERFORMANCE_TARGET_STEP_LATENCY_MS,
        "rgb_render_target_ms": PERFORMANCE_TARGET_RGB_RENDER_MS,
        "human_render_target_ms": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
        "plume_generation_target_ms": PERFORMANCE_TARGET_PLUME_GENERATION_MS,
        "episode_reset_target_ms": PERFORMANCE_TARGET_EPISODE_RESET_MS,
        "boundary_enforcement_target_ms": BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS,
        "memory_limits_mb": {
            "total": MEMORY_LIMIT_TOTAL_MB,
            "plume_field": MEMORY_LIMIT_PLUME_FIELD_MB,
        },
        "optimization_thresholds": {
            "numeric_tolerance": NUMERIC_PRECISION_TOLERANCE,
        },
        "benchmark_iterations": 100,
    }


def get_validation_constants() -> Dict[str, Any]:
    """Return validation constants with explicit parameter ranges and templates."""
    return {
        "distance_precision": DISTANCE_PRECISION,
        "gaussian_precision": GAUSSIAN_PRECISION,
        "sigma_range": (MIN_PLUME_SIGMA, MAX_PLUME_SIGMA),
        "grid_size_limits": {
            "min": MIN_GRID_SIZE,
            "max": MAX_GRID_SIZE,
        },
        "coordinate_bounds_checking": True,
        "error_messages": VALIDATION_ERROR_MESSAGES.copy(),
        "seed_validation": {"min": SEED_MIN_VALUE, "max": int(SEED_MAX_VALUE)},
        "memory_validation_mb": MEMORY_LIMIT_TOTAL_MB,
        "action_validation_range": (0, ACTION_SPACE_SIZE - 1),
    }


def get_testing_constants() -> Dict[str, Any]:
    """Return a dictionary of testing-friendly constants and parameters.

    The structure is intentionally richer to satisfy tests that expect
    specific keys for size, tolerance, and seeds. Values are conservative
    defaults suitable for unit/property tests.
    """
    data = {
        "test_grid_size": (16, 16),
        "test_source_location": (8, 8),
        "test_max_steps": 100,
        "test_tolerance": NUMERIC_PRECISION_TOLERANCE,
        "reproducibility_seeds": list(DEFAULT_TEST_SEEDS),
        "performance_test_iterations": 100,
        "expected_results": {},
    }
    return data


def validate_constant_consistency(
    strict_mode: bool = False,
) -> tuple[bool, Dict[str, Any]]:
    """Validate consistency of constants and return (is_valid, report).

    Does not raise on validation issues; instead reports findings in the
    returned report dictionary for test consumption.
    """
    report: Dict[str, Any] = {
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

    # Sigma bounds
    report["checks"].append("sigma_bounds")
    if not (MIN_PLUME_SIGMA <= DEFAULT_PLUME_SIGMA <= MAX_PLUME_SIGMA):
        ok = False
        report["errors"].append("DEFAULT_PLUME_SIGMA out of bounds")

    # Grid size sanity
    report["checks"].append("grid_size_shape")
    if not isinstance(DEFAULT_GRID_SIZE, tuple) or len(DEFAULT_GRID_SIZE) != 2:
        ok = False
        report["errors"].append("DEFAULT_GRID_SIZE must be length 2 tuple")

    # Source within grid
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

    # Strict extras
    if strict_mode:
        report["checks"].append("strict_extras")
        if PERFORMANCE_TARGET_STEP_LATENCY_MS <= 0:
            ok = False
            report["errors"].append("Non-positive step latency target")
        if MEMORY_LIMIT_TOTAL_MB <= 0:
            ok = False
            report["errors"].append("Non-positive memory limit")

    return ok, report
