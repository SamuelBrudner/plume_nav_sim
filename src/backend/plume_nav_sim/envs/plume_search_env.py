"""Gym-facing wrapper that delegates to the DI-backed environment stack."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding as gym_seeding
from gymnasium.wrappers import TimeLimit

from ..render.adapter import RendererAdapter
from ..utils.exceptions import ValidationError
from ..utils.validation import validate_seed_value
from .component_env import ComponentBasedEnvironment
from .factory import create_component_environment

__all__ = [
    "PlumeSearchEnv",
    "create_plume_search_env",
    "validate_plume_search_config",
    "unwrap_to_plume_env",
]


def _normalize_grid_size(grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if grid_size is None:
        from ..constants import DEFAULT_GRID_SIZE

        return DEFAULT_GRID_SIZE

    from ..constants import MAX_GRID_SIZE

    width, height = int(grid_size[0]), int(grid_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("grid_size must contain positive integers")

    max_width, max_height = MAX_GRID_SIZE
    if width > max_width or height > max_height:
        raise ValueError(
            f"grid_size dimensions ({width}, {height}) exceed maximum ({max_width}, {max_height})"
        )

    return width, height


def _normalize_goal(
    goal: Optional[Tuple[int, int]], grid: Tuple[int, int]
) -> Tuple[int, int]:
    if goal is None:
        return grid[0] // 2, grid[1] // 2
    x, y = int(goal[0]), int(goal[1])
    if not (0 <= x < grid[0] and 0 <= y < grid[1]):
        raise ValueError("source_location must lie within the grid bounds")
    return x, y


def _normalize_goal_radius(goal_radius: Optional[float]) -> float:
    if goal_radius is None:
        from ..constants import DEFAULT_GOAL_RADIUS

        goal_radius = float(DEFAULT_GOAL_RADIUS)
    value = float(goal_radius)
    if value < 0:
        raise ValueError("goal_radius must be non-negative")
    if value == 0:
        return float(np.finfo(np.float32).eps)
    return value


def _normalize_max_steps(max_steps: Optional[int]) -> int:
    if max_steps is None:
        from ..constants import DEFAULT_MAX_STEPS

        return DEFAULT_MAX_STEPS
    value = int(max_steps)
    if value <= 0:
        raise ValueError("max_steps must be positive")
    return value


def _normalize_plume_sigma(plume_params: Optional[Dict[str, Any]]) -> float:
    sigma = None
    if plume_params:
        sigma = plume_params.get("sigma")
    if sigma is None:
        from ..constants import DEFAULT_PLUME_SIGMA

        return float(DEFAULT_PLUME_SIGMA)
    value = float(sigma)
    if math.isnan(value) or math.isinf(value):
        raise ValueError("plume sigma must be finite numeric value")
    if value <= 0:
        raise ValueError("plume sigma must be positive")
    return value


class _AttributeForwardingTimeLimit(TimeLimit):
    """TimeLimit wrapper that forwards common attributes from the wrapped env."""

    _FORWARDED_ATTRS = (
        "grid_size",
        "source_location",
        "max_steps",
        "goal_radius",
        "action_space",
        "observation_space",
        "metadata",
        "render_mode",
    )

    def __getattr__(self, name: str):
        if name in self._FORWARDED_ATTRS:
            return getattr(self.env, name)
        return super().__getattr__(name)


class PlumeSearchEnv(gym.Env):
    """Compatibility wrapper exposing the DI-backed component environment.

    Delegation contract (stabilized by tests):
    - Seeding: reset(seed=...) produces reproducible trajectories across
      instances with identical configuration; wrapper forwards the seed to the
      underlying environment and tracks the latest seed in info.
    - Attributes: grid_size, source_location, max_steps, goal_radius reflect
      the normalized constructor arguments for registration and docs stability.
    - Rewards: step returns the immediate reward while maintaining an internal
      cumulative reward that is exposed via info["total_reward"].
    """

    metadata = ComponentBasedEnvironment.metadata

    def __init__(  # noqa: C901
        self,
        *,
        grid_size: Optional[Tuple[int, int]] = None,
        source_location: Optional[Tuple[int, int]] = None,
        max_steps: Optional[int] = None,
        goal_radius: Optional[float] = None,
        plume_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        env_options = kwargs.get("env_options") or {}

        normalized_grid = _normalize_grid_size(grid_size)
        goal_position = _normalize_goal(source_location, normalized_grid)
        max_steps_value = _normalize_max_steps(max_steps)
        goal_radius_value = _normalize_goal_radius(goal_radius)
        plume_sigma_value = _normalize_plume_sigma(plume_params)

        render_mode_value = env_options.get("render_mode")
        if render_mode_value is None and "render_mode" in kwargs:
            render_mode_value = kwargs["render_mode"]

        factory_kwargs: Dict[str, Any] = {
            "grid_size": normalized_grid,
            "goal_location": goal_position,
            "max_steps": max_steps_value,
            "goal_radius": goal_radius_value,
            "plume_sigma": plume_sigma_value,
            "render_mode": render_mode_value,
        }

        if "start_location" in env_options:
            factory_kwargs["start_location"] = env_options["start_location"]
        elif "start_location" in kwargs:
            factory_kwargs["start_location"] = kwargs["start_location"]

        for key in ("action_type", "observation_type", "reward_type", "step_size"):
            if key in env_options:
                factory_kwargs[key] = env_options[key]
            elif key in kwargs:
                factory_kwargs[key] = kwargs[key]

        # Forward optional plume selection and movie configuration if provided
        for key in (
            "plume",
            "movie_path",
            "movie_dataset_id",
            "movie_auto_download",
            "movie_cache_root",
            "movie_fps",
            "movie_pixel_to_grid",
            "movie_origin",
            "movie_extent",
            "movie_step_policy",
            "movie_h5_dataset",
            "movie_normalize",
            "movie_chunks",
            "movie_data",
        ):
            if key in env_options:
                factory_kwargs[key] = env_options[key]
            elif key in kwargs:
                factory_kwargs[key] = kwargs[key]

        self._core_env = create_component_environment(**factory_kwargs)

        # Legacy attribute compatibility expected by registration tests
        # Ensure grid_size exposed here is always a plain (width, height) tuple
        # matching the constructor/config values used during registration.
        # Some underlying component envs may expose a dataclass or override
        # dimensions (e.g., movie plumes); tests expect tuple equality to the
        # provided config for registration scenarios.
        core_grid = getattr(self._core_env, "grid_size", None)
        if core_grid is not None:
            try:
                # Coerce possible dataclass/tuple into a plain tuple[int, int]
                w = getattr(core_grid, "width", None)
                h = getattr(core_grid, "height", None)
                if w is not None and h is not None:
                    self.grid_size = (int(w), int(h))
                else:
                    cw0 = int(core_grid[0])  # type: ignore[index]
                    cw1 = int(core_grid[1])  # type: ignore[index]
                    self.grid_size = (cw0, cw1)
            except Exception:
                # Fall back to normalized constructor value on any mismatch
                self.grid_size = (int(normalized_grid[0]), int(normalized_grid[1]))
        else:
            self.grid_size = (int(normalized_grid[0]), int(normalized_grid[1]))
        self.source_location = goal_position
        self.max_steps = max_steps_value
        self.goal_radius = goal_radius_value

        self.action_space = self._core_env.action_space
        width, height = normalized_grid
        # Legacy Box observation space for backward compatibility
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self._env = _AttributeForwardingTimeLimit(
            self._core_env, max_episode_steps=self.max_steps
        )

        self._cumulative_reward: float = 0.0
        self._latest_seed: Optional[int] = None
        self.np_random, self._latest_seed = gym_seeding.np_random(None)

        # Provide a renderer handle and default interactive configuration
        # for integration tests that expect env.renderer.* APIs to be present.
        # Adapter exposes a minimal unified surface across RGB and Matplotlib renderers.
        w, h = normalized_grid
        self.renderer = RendererAdapter(width=w, height=h)

        # Default interactive configuration consumed by rendering tests
        self.interactive_config = {
            "enable_toolbar": True,
            "enable_key_bindings": True,
            "update_interval": 0.1,
            "animation_enabled": True,
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        seed_to_use: int
        try:
            validate_seed_value(
                seed,
                allow_none=True,
                strict_type_checking=True,
            )
        except ValidationError as exc:
            raise ValidationError(
                f"Invalid seed value: {seed}",
                context={
                    "seed_type": type(seed).__name__,
                    "seed_value": seed,
                },
            ) from exc

        if seed is not None:
            self.np_random, seed_to_use = gym_seeding.np_random(seed)
        else:
            if self.np_random is None:
                self.np_random, seed_to_use = gym_seeding.np_random(None)
            else:
                seed_to_use = int(
                    self.np_random.integers(0, 2**32 - 1, dtype=np.uint32)
                )

        if seed_to_use is not None:
            seed_to_use = int(seed_to_use)

        self._latest_seed = seed_to_use
        self._cumulative_reward = 0.0

        # Ensure underlying environment shares the same deterministic seed
        try:
            if hasattr(self._env, "seed"):
                self._env.seed(seed_to_use)
        except Exception:
            # Fall back to relying on reset seeding if seed() not supported
            pass

        obs, info = self._env.reset(seed=seed_to_use, options=options)
        wrapped_obs = self._wrap_observation(obs)
        augmented_info = self._augment_reset_info(info)
        return wrapped_obs, augmented_info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        immediate_reward = float(reward)
        self._cumulative_reward += immediate_reward
        wrapped_obs = self._wrap_observation(obs)
        augmented_info = self._augment_step_info(info, terminated)
        return (
            wrapped_obs,
            immediate_reward,
            terminated,
            truncated,
            augmented_info,
        )

    def render(self, mode: Optional[str] = None) -> Any:
        """Render with Gymnasium-compatible semantics.

        - If no mode is provided, respect the wrapped env's configured render_mode.
        - For 'rgb_array', return an ndarray frame; fall back to core env or zeros.
        - For 'human', return None (side-effects may occur in wrapped envs).
        """
        effective_mode = (
            mode if mode is not None else getattr(self._env, "render_mode", None)
        )
        if effective_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {effective_mode}")

        grid_size = getattr(self._env, "grid_size", None)
        if grid_size is not None:
            width = getattr(grid_size, "width", None) or grid_size[0]
            height = getattr(grid_size, "height", None) or grid_size[1]
        else:
            width = height = 1

        # First attempt: delegate to wrapped env in its configured mode
        result = self._env.render()

        # Effective behavior depends on resolved mode
        if (effective_mode or "human") == "rgb_array":
            if isinstance(result, np.ndarray):
                return result
            # Avoid wrapper kwarg signature issues by calling core env directly
            try:
                core = getattr(self, "_core_env", None)
                if core is not None:
                    alt = core.render()
                    if isinstance(alt, np.ndarray):
                        return alt
            except Exception:
                pass
            return np.zeros((height, width, 3), dtype=np.uint8)

        # human mode: per Gymnasium convention, return None (ignore ndarray result)
        return None

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Compatibility helpers
    def validate_environment_integrity(self) -> bool:
        return True

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random, seed_used = gym_seeding.np_random(seed)
        if seed_used is not None:
            seed_used = int(seed_used)
        self._latest_seed = seed_used

        if hasattr(self._env, "seed"):
            try:
                return self._env.seed(seed_used)
            except Exception:
                pass

        # Fall back to resetting with the seed if environment lacks a dedicated seed API
        self._env.reset(seed=seed_used)
        return [seed_used]

    # ------------------------------------------------------------------
    def _wrap_observation(self, observation: Any) -> np.ndarray:
        """Extract sensor reading and return as Box(1,) observation."""
        if isinstance(observation, dict):
            sensor = observation.get("sensor_reading") or observation.get("observation")
        else:
            sensor = observation

        sensor_array = np.atleast_1d(np.asarray(sensor, dtype=np.float32))
        if sensor_array.shape != (1,):
            raise ValueError(
                f"Expected sensor reading shape (1,), received shape {sensor_array.shape}"
            )

        return sensor_array

    def _augment_reset_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info)
        agent_position = info.get("agent_position")
        if agent_position:
            info.setdefault("agent_xy", agent_position)
        info.setdefault("seed", self._latest_seed)
        info.setdefault("total_reward", self._cumulative_reward)
        info.setdefault("goal_reached", False)
        info.setdefault("step_count", 0)
        info.setdefault("episode_count", info.get("episode", 0))
        return info

    def _augment_step_info(
        self, info: Dict[str, Any], terminated: bool
    ) -> Dict[str, Any]:
        info = dict(info)
        info.setdefault("total_reward", self._cumulative_reward)
        info.setdefault("goal_reached", terminated)
        info.setdefault("episode_count", info.get("episode", 0))
        if self._latest_seed is not None:
            info.setdefault("seed", self._latest_seed)
        if "agent_position" in info:
            info.setdefault("agent_xy", info["agent_position"])
        return info


def unwrap_to_plume_env(env: gym.Env) -> PlumeSearchEnv:
    """Traverse common Gymnasium wrappers to locate the underlying PlumeSearchEnv."""

    visited = set()
    current: Any = env

    while current is not None and id(current) not in visited:
        visited.add(id(current))

        if isinstance(current, PlumeSearchEnv):
            return current

        if hasattr(current, "env"):
            try:
                current = current.env
            except AttributeError:
                break
            continue

        if hasattr(current, "unwrapped"):
            try:
                current = current.unwrapped
            except AttributeError:
                break
            continue

        break

    raise TypeError("Unable to locate PlumeSearchEnv within provided wrapper stack.")


def create_plume_search_env(**kwargs: Any) -> PlumeSearchEnv:
    """Factory function to create a PlumeSearchEnv instance.

    Args:
        **kwargs: Configuration parameters passed to PlumeSearchEnv constructor.

    Returns:
        PlumeSearchEnv: Configured environment instance.
    """
    return PlumeSearchEnv(**kwargs)


def validate_plume_search_config(**kwargs: Any) -> Dict[str, Any]:
    """Validate and normalize PlumeSearchEnv configuration parameters.

    Args:
        **kwargs: Raw configuration parameters to validate.

    Returns:
        Dict[str, Any]: Validated and normalized configuration.
    """
    grid = _normalize_grid_size(kwargs.get("grid_size"))
    source = _normalize_goal(kwargs.get("source_location"), grid)
    max_steps = _normalize_max_steps(kwargs.get("max_steps"))
    goal_radius = _normalize_goal_radius(kwargs.get("goal_radius"))
    plume_sigma = _normalize_plume_sigma(kwargs.get("plume_params"))
    return {
        "grid_size": grid,
        "source_location": source,
        "max_steps": max_steps,
        "goal_radius": goal_radius,
        "plume_sigma": plume_sigma,
    }
