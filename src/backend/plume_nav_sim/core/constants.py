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

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "constants.yaml"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "package": {
        "name": "plume_nav_sim",
        "version": "0.0.0",
        "environment_id": "PlumeNav-Sim-v1",
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
MIN_GRID_SIZE = (16, 16)
MAX_GRID_SIZE = (1024, 1024)
DEFAULT_SOURCE_LOCATION = (64, 64)
DEFAULT_PLUME_SIGMA = 12.0
DEFAULT_GOAL_RADIUS = 0.0
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
BACKEND_PRIORITY_LIST = ["TkAgg", "Qt5Agg", "Agg"]
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
    """Returns a dictionary of rendering constants."""
    return {
        "AGENT_MARKER_COLOR": AGENT_MARKER_COLOR,
        "SOURCE_MARKER_COLOR": SOURCE_MARKER_COLOR,
        "AGENT_MARKER_SIZE": AGENT_MARKER_SIZE,
        "SOURCE_MARKER_SIZE": SOURCE_MARKER_SIZE,
        "PIXEL_VALUE_MIN": PIXEL_VALUE_MIN,
        "PIXEL_VALUE_MAX": PIXEL_VALUE_MAX,
        "SUPPORTED_RENDER_MODES": list(SUPPORTED_RENDER_MODES),
    }


def get_performance_constants() -> Dict[str, Any]:
    """Returns a dictionary of performance constants."""
    return {
        "PERFORMANCE_TRACKING_ENABLED": PERFORMANCE_TRACKING_ENABLED,
        "PERFORMANCE_TARGET_STEP_LATENCY_MS": PERFORMANCE_TARGET_STEP_LATENCY_MS,
        "PERFORMANCE_TARGET_RGB_RENDER_MS": PERFORMANCE_TARGET_RGB_RENDER_MS,
        "PERFORMANCE_TARGET_HUMAN_RENDER_MS": PERFORMANCE_TARGET_HUMAN_RENDER_MS,
        "PERFORMANCE_TARGET_EPISODE_RESET_MS": PERFORMANCE_TARGET_EPISODE_RESET_MS,
        "PERFORMANCE_TARGET_PLUME_GENERATION_MS": PERFORMANCE_TARGET_PLUME_GENERATION_MS,
        "BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS": BOUNDARY_ENFORCEMENT_PERFORMANCE_TARGET_MS,
    }


def get_validation_constants() -> Dict[str, Any]:
    """Returns a dictionary of validation constants."""
    return {
        "VALIDATION_ERROR_MESSAGES": VALIDATION_ERROR_MESSAGES.copy(),
        "NUMERIC_PRECISION_TOLERANCE": NUMERIC_PRECISION_TOLERANCE,
    }


def get_testing_constants() -> Dict[str, Any]:
    """Returns a dictionary of testing constants."""
    return {
        "DEFAULT_SEEDS": list(DEFAULT_TEST_SEEDS),
    }


def validate_constant_consistency(strict_mode: bool = False) -> None:
    """Validate consistency of constants.

    Args:
        strict_mode: Enable strict validation checks (currently unused, for future expansion)
    """
    if DEFAULT_PLUME_SIGMA < MIN_PLUME_SIGMA or DEFAULT_PLUME_SIGMA > MAX_PLUME_SIGMA:
        raise ValueError("DEFAULT_PLUME_SIGMA out of bounds")
    if len(DEFAULT_GRID_SIZE) != 2:
        raise ValueError("DEFAULT_GRID_SIZE must be length 2")
