"""Gym-facing wrapper that delegates to the DI-backed environment stack."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import gymnasium as gym

from .component_env import ComponentBasedEnvironment
from .factory import create_component_environment

__all__ = ["PlumeSearchEnv", "create_plume_search_env", "validate_plume_search_config"]


def _normalize_grid_size(grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if grid_size is None:
        from ..core.constants import DEFAULT_GRID_SIZE

        return DEFAULT_GRID_SIZE
    width, height = int(grid_size[0]), int(grid_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("grid_size must contain positive integers")
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
        from ..core.constants import DEFAULT_GOAL_RADIUS

        goal_radius = float(DEFAULT_GOAL_RADIUS)
    value = float(goal_radius)
    if value < 0:
        raise ValueError("goal_radius must be non-negative")
    if value == 0:
        return float(np.finfo(np.float32).eps)
    return value


def _normalize_max_steps(max_steps: Optional[int]) -> int:
    if max_steps is None:
        from ..core.constants import DEFAULT_MAX_STEPS

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
        from ..core.constants import DEFAULT_PLUME_SIGMA

        return float(DEFAULT_PLUME_SIGMA)
    value = float(sigma)
    if value <= 0:
        raise ValueError("plume sigma must be positive")
    return value


class PlumeSearchEnv(gym.Env):
    """Compatibility wrapper exposing the DI-backed component environment."""

    metadata = ComponentBasedEnvironment.metadata

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

        self._env = create_component_environment(**factory_kwargs)

        self.action_space = self._env.action_space
        width, height = normalized_grid
        self.observation_space = gym.spaces.Dict(
            {
                "agent_position": gym.spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([width - 1, height - 1], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "sensor_reading": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "source_location": gym.spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([width - 1, height - 1], dtype=np.float32),
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        self._cumulative_reward: float = 0.0
        self._latest_seed: Optional[int] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        self._latest_seed = seed
        self._cumulative_reward = 0.0
        obs, info = self._env.reset(seed=seed, options=options)
        wrapped_obs = self._wrap_observation(obs)
        augmented_info = self._augment_reset_info(info)
        return wrapped_obs, augmented_info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._cumulative_reward += float(reward)
        wrapped_obs = self._wrap_observation(obs)
        augmented_info = self._augment_step_info(info, terminated)
        return wrapped_obs, float(reward), terminated, truncated, augmented_info

    def render(self, mode: str = "human") -> Any:
        if mode not in {"human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {mode}")

        result = self._env.render()
        if mode == "rgb_array":
            if result is not None:
                return result
            height, width = self._env.grid_size.height, self._env.grid_size.width
            return np.zeros((height, width, 3), dtype=np.uint8)

        # human mode: defer to underlying environment even if it returns rgb
        return result

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Compatibility helpers
    def validate_environment_integrity(self) -> bool:
        return True

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self._latest_seed = seed
        self._env.reset(seed=seed)
        return [seed]

    # ------------------------------------------------------------------
    def _wrap_observation(self, observation: Any) -> Dict[str, np.ndarray]:
        if isinstance(observation, dict):
            return observation

        agent_state = getattr(self._env, "_agent_state", None)
        if agent_state is None:
            raise RuntimeError("Underlying environment did not initialize agent state")

        agent_pos = np.array(
            [agent_state.position.x, agent_state.position.y], dtype=np.float32
        )
        sensor = np.atleast_1d(np.asarray(observation, dtype=np.float32))
        if sensor.shape != (1,):
            raise ValueError(
                "Wrapped observation expected shape (1,) from ConcentrationSensor"
            )

        goal = getattr(self._env, "goal_location", None)
        if goal is None:
            raise RuntimeError("Underlying environment missing goal_location")
        goal_pos = np.array([goal.x, goal.y], dtype=np.float32)

        return {
            "agent_position": agent_pos,
            "sensor_reading": sensor,
            "source_location": goal_pos,
        }

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


def create_plume_search_env(**kwargs: Any) -> PlumeSearchEnv:
    return PlumeSearchEnv(**kwargs)


def validate_plume_search_config(**kwargs: Any) -> Dict[str, Any]:
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
