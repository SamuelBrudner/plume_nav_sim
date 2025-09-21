"""Lightweight plume search environment used in the trimmed kata build.

The original project ships a considerably more sophisticated Gymnasium
environment.  Shipping the full simulator is unnecessary for the unit tests in
this kata, yet a realistic surface is still valuable: the edge-case and
performance suites expect an environment that can reset, step through episodes,
render deterministic RGB arrays, and perform basic validation.  This module
implements a compact but fully functional replacement that intentionally focuses
on determinism, clarity, and aggressive input validation so the surrounding
tests can execute meaningful scenarios.

The implementation favours explicit logging and predictable behaviour over raw
feature parity.  Each public method guards against misuse, raising the
exceptions defined in :mod:`plume_nav_sim.utils.exceptions` so callers receive
actionable feedback instead of silent fallbacks.  The environment keeps enough
state to service the tests while remaining easy to reason about for students.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.constants import (
    ACTION_SPACE_SIZE,
    AGENT_MARKER_COLOR,
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLUME_SIGMA,
    DEFAULT_SOURCE_LOCATION,
    MAX_PLUME_SIGMA,
    MIN_PLUME_SIGMA,
    PIXEL_VALUE_MAX,
    PIXEL_VALUE_MIN,
    RGB_DTYPE,
    SEED_MAX_VALUE,
    SEED_MIN_VALUE,
)
from ..core.types import Action
from ..utils.exceptions import (
    ConfigurationError,
    RenderingError,
    StateError,
    ValidationError,
)

__all__ = ["PlumeSearchEnv", "create_plume_search_env", "validate_plume_search_config"]

logger = logging.getLogger(__name__)


def _validate_grid_size(grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """Normalise and validate the incoming grid dimensions."""

    if grid_size is None:
        return DEFAULT_GRID_SIZE

    try:
        width = int(grid_size[0])
        height = int(grid_size[1])
    except (TypeError, ValueError, IndexError) as exc:  # pragma: no cover - defensive
        raise ConfigurationError(
            "grid_size must be a length-2 iterable of integers"
        ) from exc

    if width <= 0 or height <= 0:
        raise ConfigurationError("grid_size values must be positive integers")

    return width, height


def _validate_source_location(
    source: Optional[Tuple[int, int]], grid: Tuple[int, int]
) -> Tuple[int, int]:
    """Validate that the source lies within the configured grid bounds."""

    if source is None:
        # Default to the centre of the configured grid so small test grids remain valid.
        width, height = grid
        source = (width // 2, height // 2)

    try:
        x = int(source[0])
        y = int(source[1])
    except (TypeError, ValueError, IndexError) as exc:  # pragma: no cover - defensive
        raise ConfigurationError(
            "source_location must be a length-2 iterable of integers"
        ) from exc

    width, height = grid
    if not (0 <= x < width and 0 <= y < height):
        raise ConfigurationError(
            f"source_location {source!r} must be within grid bounds 0..{width - 1}, 0..{height - 1}"
        )

    return x, y


def _validate_sigma(value: Optional[float]) -> float:
    """Validate plume spread (sigma) parameters."""

    if value is None:
        return DEFAULT_PLUME_SIGMA

    try:
        sigma = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ConfigurationError("plume_params.sigma must be numeric") from exc

    if math.isnan(sigma) or math.isinf(sigma) or sigma <= 0:
        raise ConfigurationError("plume_params.sigma must be a positive finite number")

    if not (MIN_PLUME_SIGMA <= sigma <= MAX_PLUME_SIGMA):
        raise ConfigurationError(
            f"plume_params.sigma must be within [{MIN_PLUME_SIGMA}, {MAX_PLUME_SIGMA}]"
        )

    return sigma


def _resolve_seed(seed: Optional[Any]) -> Optional[int]:
    """Coerce seeds to integers while preserving deterministic behaviour."""

    if seed is None:
        return None

    if isinstance(seed, (np.integer, int)):
        seed_value = int(seed)
    elif isinstance(seed, float) and seed.is_integer():
        seed_value = int(seed)
    else:
        raise ValidationError(
            "Seed must be an integer, float integral value, or None",
            parameter_name="seed",
        )

    if not (SEED_MIN_VALUE <= seed_value <= SEED_MAX_VALUE):
        raise ValidationError(
            "Seed is outside the supported range", parameter_name="seed"
        )

    return seed_value


class _DiscreteActionSpace:
    """Minimal discrete action space exposing the API the tests exercise."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.n = ACTION_SPACE_SIZE
        self._rng = rng

    def sample(self) -> int:
        value = int(self._rng.integers(0, self.n))
        logger.debug("Sampled action value %s from discrete space", value)
        return value

    def contains(self, item: Any) -> bool:
        try:
            value = int(item)
        except (TypeError, ValueError):
            return False
        return 0 <= value < self.n


@dataclass
class _ObservationSpace:
    """Very small observation-space stub used by the test-suite assertions."""

    shape: Tuple[int, int, int]
    dtype: Any = np.float32


class PlumeSearchEnv:
    """Deterministic grid environment with Gaussian concentration field."""

    metadata: Dict[str, Any] = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        *,
        grid_size: Optional[Tuple[int, int]] = None,
        source_location: Optional[Tuple[int, int]] = None,
        max_steps: Optional[int] = None,
        goal_radius: Optional[float] = None,
        plume_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.debug(
                "Ignoring unsupported configuration fields for PlumeSearchEnv: %s",
                sorted(kwargs.keys()),
            )
        self.grid_size = _validate_grid_size(grid_size)
        self.source_location = _validate_source_location(
            source_location, self.grid_size
        )

        if max_steps is None:
            max_steps = DEFAULT_MAX_STEPS
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ConfigurationError("max_steps must be a positive integer")
        self.max_steps = int(max_steps)

        if goal_radius is None:
            goal_radius = DEFAULT_GOAL_RADIUS
        if not isinstance(goal_radius, (int, float)) or goal_radius < 0:
            raise ConfigurationError("goal_radius must be a non-negative number")
        self.goal_radius = float(goal_radius)

        sigma_input = None
        if plume_params:
            sigma_input = plume_params.get("sigma")
        self._sigma = _validate_sigma(sigma_input)

        self._rng_seed: Optional[int] = None
        self._rng = np.random.default_rng()
        self.action_space = _DiscreteActionSpace(self._rng)
        self.observation_space = _ObservationSpace(
            shape=(self.grid_size[0], self.grid_size[1], 1)
        )

        self._step_count = 0
        self._episode_active = False
        self._closed = False
        self._agent_position = self.source_location
        self._lock = threading.RLock()

        logger.info(
            "Initialised PlumeSearchEnv grid=%s source=%s max_steps=%s goal_radius=%s sigma=%.3f",
            self.grid_size,
            self.source_location,
            self.max_steps,
            self.goal_radius,
            self._sigma,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    def _ensure_active(self) -> None:
        if self._closed:
            raise StateError(
                "Environment has been closed",
                current_state="closed",
                component_name="env",
            )

    def _update_rng(self, seed: Optional[int]) -> None:
        if seed is None:
            seed = int(time.time_ns() % (SEED_MAX_VALUE + 1))
        self._rng_seed = seed
        self._rng = np.random.default_rng(seed)
        self.action_space = _DiscreteActionSpace(self._rng)
        logger.debug("Reset RNG with seed %s", seed)

    def _random_agent_position(self) -> Tuple[int, int]:
        width, height = self.grid_size
        x = int(self._rng.integers(0, width))
        y = int(self._rng.integers(0, height))
        return x, y

    def _compute_distance(self, position: Tuple[int, int]) -> float:
        sx, sy = self.source_location
        px, py = position
        return math.hypot(px - sx, py - sy)

    def _compute_concentration(self, position: Tuple[int, int]) -> float:
        distance = self._compute_distance(position)
        exponent = -((distance**2) / (2 * (self._sigma**2)))
        value = math.exp(exponent)
        return max(0.0, min(1.0, value))

    def _observation(self) -> np.ndarray:
        concentration = self._compute_concentration(self._agent_position)
        normalised_steps = self._step_count / max(1, self.max_steps)
        distance = self._compute_distance(self._agent_position)
        obs = np.array([concentration, normalised_steps, distance], dtype=np.float32)
        return obs

    def _info(self) -> Dict[str, Any]:
        return {
            "agent_xy": self._agent_position,
            "plume_peak_xy": self.source_location,
            "distance_to_source": self._compute_distance(self._agent_position),
            "step_count": self._step_count,
            "seed": self._rng_seed,
        }

    # ------------------------------------------------------------------
    # Gymnasium-style API
    def reset(
        self, *, seed: Optional[Any] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        with self._lock:
            self._ensure_active()
            resolved_seed = _resolve_seed(seed)
            self._update_rng(resolved_seed)
            self._step_count = 0
            self._episode_active = True
            self._agent_position = self._random_agent_position()
            logger.debug(
                "Environment reset with seed=%s position=%s",
                resolved_seed,
                self._agent_position,
            )
            return self._observation(), self._info()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        with self._lock:
            self._ensure_active()
            if not self._episode_active:
                raise StateError(
                    "Environment must be reset before stepping",
                    current_state="inactive",
                    component_name="env",
                )

            try:
                action_enum = Action(int(action))
            except (ValueError, TypeError):
                raise ValidationError(
                    "Action must be an integer in range [0, 3]", parameter_name="action"
                )

            dx, dy = action_enum.to_vector()
            x, y = self._agent_position
            width, height = self.grid_size
            new_x = int(np.clip(x + dx, 0, width - 1))
            new_y = int(np.clip(y + dy, 0, height - 1))
            self._agent_position = (new_x, new_y)
            self._step_count += 1

            distance = self._compute_distance(self._agent_position)
            reward = float(1.0 - min(distance, width + height) / (width + height))

            terminated = distance <= self.goal_radius
            truncated = self._step_count >= self.max_steps
            if terminated or truncated:
                self._episode_active = False

            observation = self._observation()
            info = self._info()
            logger.debug(
                "Step action=%s position=%s reward=%.3f terminated=%s truncated=%s",
                action_enum.name,
                self._agent_position,
                reward,
                terminated,
                truncated,
            )
            return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        with self._lock:
            self._ensure_active()
            mode = mode or "human"
            if mode not in self.metadata["render_modes"]:
                raise RenderingError(
                    f"Unsupported render mode: {mode}", render_mode=mode
                )

            if mode == "rgb_array":
                return self._render_rgb_array()

            try:
                import matplotlib.pyplot as plt  # type: ignore[import-not-found]

                field = self._render_rgb_array()
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(field)
                ax.set_title("Plume Navigation Environment")
                ax.axis("off")
                plt.close(fig)
                return None
            except Exception as exc:  # pragma: no cover - exercised via tests
                logger.warning(
                    "Human rendering failed, falling back to rgb_array: %s", exc
                )
                try:
                    return self._render_rgb_array()
                except Exception as fallback_exc:  # pragma: no cover - defensive
                    raise RenderingError(
                        "Rendering failed even after attempting rgb_array fallback",
                        render_mode=mode,
                        underlying_error=fallback_exc,
                    ) from exc

    def close(self) -> None:
        with self._lock:
            self._episode_active = False
            self._closed = True
            logger.info("Closed PlumeSearchEnv")

    # ------------------------------------------------------------------
    # Additional helpers exercised by the tests
    def validate_environment_integrity(self) -> bool:
        with self._lock:
            healthy = (
                not self._closed and self.grid_size[0] > 0 and self.grid_size[1] > 0
            )
            healthy = healthy and isinstance(self._rng_seed, (int, type(None)))
            healthy = healthy and isinstance(self._agent_position, tuple)
            return healthy

    # ------------------------------------------------------------------
    # Internal rendering helpers
    def _render_rgb_array(self) -> np.ndarray:
        width, height = self.grid_size
        field = np.zeros((height, width, 3), dtype=RGB_DTYPE)

        # Create a smooth gradient so tests can check determinism without storing huge fixtures
        x_indices = np.linspace(0, 1, num=width, dtype=np.float32)
        y_indices = np.linspace(0, 1, num=height, dtype=np.float32)
        gradient = np.outer(y_indices, x_indices)
        base_layer = (gradient * PIXEL_VALUE_MAX).astype(RGB_DTYPE)
        field[:, :, 0] = base_layer
        field[:, :, 1] = base_layer
        field[:, :, 2] = base_layer

        ax, ay = self._agent_position
        sx, sy = self.source_location
        field[ay, ax] = np.array(AGENT_MARKER_COLOR, dtype=RGB_DTYPE)
        field[sy, sx] = np.array(AGENT_MARKER_COLOR, dtype=RGB_DTYPE)
        return field


def create_plume_search_env(**kwargs: Any) -> PlumeSearchEnv:
    """Factory mirroring the public API of the original project."""

    return PlumeSearchEnv(**kwargs)


def validate_plume_search_config(**kwargs: Any) -> Dict[str, Any]:
    """Expose a helper used by the tests for sanity-checking parameters."""

    grid = _validate_grid_size(kwargs.get("grid_size"))
    source = _validate_source_location(kwargs.get("source_location"), grid)
    sigma = _validate_sigma(
        kwargs.get("plume_params", {}).get("sigma")
        if kwargs.get("plume_params")
        else None
    )
    result = {"grid_size": grid, "source_location": source, "sigma": sigma}
    logger.debug("Validated plume configuration: %s", result)
    return result
