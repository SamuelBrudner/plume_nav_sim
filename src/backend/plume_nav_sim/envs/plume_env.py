"""PlumeEnv: flattened environment with injected components."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding as gym_seeding

from .._compat import StateError, ValidationError, validate_seed_value
from ..actions import DiscreteGridActions
from ..constants import (
    AGENT_MARKER_COLOR,
    AGENT_MARKER_SIZE,
    DEFAULT_GOAL_RADIUS,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_PLUME_SIGMA,
    MAX_GRID_SIZE,
    SOURCE_MARKER_COLOR,
    SOURCE_MARKER_SIZE,
)
from ..core.types import AgentState, Coordinates, GridSize
from ..interfaces import ActionProcessor, ObservationModel, RewardFunction, VectorField
from ..observations import ConcentrationSensor
from ..plume.gaussian import GaussianPlume
from ..plume.protocol import ConcentrationField
from ..rewards import SparseGoalReward
from .state import EnvironmentState

__all__ = ["PlumeEnv", "create_plume_env"]

logger = logging.getLogger(__name__)


def _coerce_plume_grid(plume: ConcentrationField) -> Optional[GridSize]:
    candidate = getattr(plume, "grid_size", None)
    if candidate is None:
        return None
    if isinstance(candidate, GridSize):
        return candidate
    if isinstance(candidate, (tuple, list)) and len(candidate) == 2:
        try:
            return GridSize(width=int(candidate[0]), height=int(candidate[1]))
        except Exception:
            return None
    width = getattr(candidate, "width", None)
    height = getattr(candidate, "height", None)
    if width is not None and height is not None:
        try:
            return GridSize(width=int(width), height=int(height))
        except Exception:
            return None
    return None


class PlumeEnv(gym.Env):
    """Gymnasium environment with direct component injection."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        *,
        grid_size: Optional[tuple[int, int] | GridSize] = None,
        source_location: Optional[tuple[int, int] | Coordinates] = None,
        goal_radius: Optional[float] = None,
        max_steps: Optional[int] = None,
        plume: Optional[ConcentrationField] = None,
        sensor_model: Optional[ObservationModel] = None,
        action_model: Optional[ActionProcessor] = None,
        reward_fn: Optional[RewardFunction] = None,
        plume_params: Optional[Mapping[str, Any]] = None,
        start_location: Optional[tuple[int, int] | Coordinates] = None,
        render_mode: Optional[str] = None,
        wind_field: Optional[VectorField] = None,
    ) -> None:
        super().__init__()

        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValidationError(
                "render_mode must be supported",
                parameter_name="render_mode",
                parameter_value=render_mode,
                expected_format=str(self.metadata["render_modes"]),
            )

        plume_grid = _coerce_plume_grid(plume) if plume is not None else None
        if grid_size is None and plume_grid is not None:
            grid_value = plume_grid
        else:
            grid_value = DEFAULT_GRID_SIZE if grid_size is None else grid_size
        if isinstance(grid_value, GridSize):
            grid = grid_value
        else:
            if not isinstance(grid_value, (tuple, list)) or len(grid_value) != 2:
                raise ValidationError(
                    "grid_size must be a length-2 tuple",
                    parameter_name="grid_size",
                    parameter_value=str(grid_value),
                )
            width, height = int(grid_value[0]), int(grid_value[1])
            if width <= 0 or height <= 0:
                raise ValidationError(
                    "grid_size must contain positive integers",
                    parameter_name="grid_size",
                    parameter_value=(width, height),
                )
            max_width, max_height = MAX_GRID_SIZE
            if width > max_width or height > max_height:
                raise ValidationError(
                    "grid_size exceeds maximum",
                    parameter_name="grid_size",
                    parameter_value=(width, height),
                    expected_format=f"<= {MAX_GRID_SIZE}",
                )
            grid = GridSize(width=width, height=height)

        def _to_coordinates(
            value: tuple[int, int] | Coordinates, name: str
        ) -> Coordinates:
            if isinstance(value, Coordinates):
                coords = value
            else:
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise ValidationError(
                        f"{name} must be a length-2 tuple",
                        parameter_name=name,
                        parameter_value=str(value),
                    )
                coords = Coordinates(int(value[0]), int(value[1]))
            if not grid.contains(coords):
                raise ValidationError(
                    f"{name} must be within grid bounds",
                    parameter_name=name,
                    parameter_value=coords.to_tuple(),
                )
            return coords

        goal = (
            grid.center()
            if source_location is None
            else _to_coordinates(source_location, "source_location")
        )

        radius = DEFAULT_GOAL_RADIUS if goal_radius is None else float(goal_radius)
        if radius < 0:
            raise ValidationError(
                "goal_radius must be non-negative",
                parameter_name="goal_radius",
                parameter_value=radius,
            )
        if radius == 0:
            radius = float(np.finfo(np.float32).eps)

        steps = DEFAULT_MAX_STEPS if max_steps is None else int(max_steps)
        if steps <= 0:
            raise ValidationError(
                "max_steps must be positive",
                parameter_name="max_steps",
                parameter_value=steps,
            )

        if plume_grid is not None and (
            grid.width != plume_grid.width or grid.height != plume_grid.height
        ):
            raise ValidationError(
                "grid_size must match plume grid_size",
                parameter_name="grid_size",
                parameter_value=grid.to_tuple(),
                expected_format=str(plume_grid.to_tuple()),
            )

        self._grid_size = grid
        self.grid_size = grid.to_tuple()
        self._goal_location = goal
        self.goal_location = goal
        self.source_location = goal.to_tuple()
        self.goal_radius = radius
        self.max_steps = steps
        self.render_mode = render_mode

        self._start_location = (
            None
            if start_location is None
            else _to_coordinates(start_location, "start_location")
        )
        self.start_location = self._start_location

        if plume is None:
            sigma = DEFAULT_PLUME_SIGMA
            if plume_params is not None:
                if not isinstance(plume_params, Mapping):
                    raise ValidationError(
                        "plume_params must be mapping-like",
                        parameter_name="plume_params",
                        parameter_value=str(plume_params),
                    )
                sigma = plume_params.get("sigma", DEFAULT_PLUME_SIGMA)
            sigma_value = float(sigma)
            if not np.isfinite(sigma_value) or sigma_value <= 0:
                raise ValidationError(
                    "plume_params.sigma must be positive",
                    parameter_name="plume_params.sigma",
                    parameter_value=sigma_value,
                )
            plume = GaussianPlume(
                grid_size=grid,
                source_location=goal,
                sigma=sigma_value,
            )
        self.plume = plume
        self.concentration_field = plume
        self._concentration_field = plume
        self.plume_params = dict(plume_params) if plume_params is not None else None

        if action_model is None:
            action_model = DiscreteGridActions()
        if sensor_model is None:
            sensor_model = ConcentrationSensor()
        if reward_fn is None:
            reward_fn = SparseGoalReward(goal_position=goal, goal_radius=radius)

        self.action_model = action_model
        self.sensor_model = sensor_model
        self.reward_fn = reward_fn
        self.action_processor = action_model
        self.observation_model = sensor_model
        self.reward_function = reward_fn
        self._action_processor = action_model
        self._observation_model = sensor_model
        self._reward_function = reward_fn
        self._wind_field = wind_field
        self.wind_field = wind_field

        self.action_space = action_model.action_space
        self.observation_space = sensor_model.observation_space

        self._agent_state: Optional[AgentState] = None
        self._step_count = 0
        self._episode_count = 0
        self._latest_seed: Optional[int] = None
        self._seed: Optional[int] = None
        self.np_random: Optional[np.random.Generator] = None
        self._rng: Optional[np.random.Generator] = None

        self._state = EnvironmentState.CREATED

        self.interactive_config = {
            "enable_toolbar": True,
            "enable_key_bindings": True,
            "update_interval": 0.1,
            "animation_enabled": True,
        }

        logger.info(
            "PlumeEnv initialized: grid=%s goal=%s max_steps=%s",
            self.grid_size,
            self.source_location,
            self.max_steps,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[Any, dict]:
        if self._state == EnvironmentState.CLOSED:
            raise StateError("Cannot reset closed environment")

        try:
            validated_seed = validate_seed_value(
                seed, allow_none=True, strict_type_checking=True
            )
        except ValidationError as exc:
            raise ValidationError(
                f"Invalid seed value: {seed}",
                parameter_name="seed",
                parameter_value=seed,
            ) from exc

        seed_to_use: Optional[int]
        if validated_seed is not None:
            self.np_random, seed_to_use = gym_seeding.np_random(validated_seed)
            self._latest_seed = None if seed_to_use is None else int(seed_to_use)
        else:
            if self.np_random is None:
                self.np_random, _ = gym_seeding.np_random(None)
            # Unseeded episodes remain stochastic, but do not advertise a replay seed.
            self._latest_seed = None
        self._seed = self._latest_seed
        self._rng = self.np_random
        self._step_count = 0
        self._episode_count += 1

        start_position = self._choose_start_position()
        self._agent_state = AgentState(position=start_position, orientation=0.0)
        self.start_location = start_position

        self._state = EnvironmentState.READY

        if hasattr(self.action_model, "set_rng"):
            try:
                self.action_model.set_rng(self._rng)
            except Exception:
                pass

        self._reset_dynamic_fields()

        observation = self._get_observation()
        pos = self._agent_state.position
        info = {
            "agent_position": (pos.x, pos.y),
            "agent_xy": (pos.x, pos.y),
            "goal_location": self.source_location,
            "source_location": self.source_location,
            "step_count": 0,
            "episode_count": self._episode_count,
            "total_reward": self._agent_state.total_reward,
            "goal_reached": False,
        }
        if self._latest_seed is not None:
            info["seed"] = self._latest_seed
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        self._ensure_ready_for_step()

        if hasattr(self.action_model, "validate_action"):
            if not self.action_model.validate_action(action):
                raise ValidationError(
                    "Invalid action",
                    parameter_name="action",
                    parameter_value=action,
                )
        elif hasattr(self.action_space, "contains") and not self.action_space.contains(
            action
        ):
            raise ValidationError(
                "Invalid action",
                parameter_name="action",
                parameter_value=action,
            )

        new_state = self.action_model.process_action(
            action, self._agent_state, self._grid_size
        )
        self._step_count += 1
        new_state.step_count = self._step_count

        self._advance_dynamic_fields(self._step_count)

        reward = float(
            self.reward_fn.compute_reward(
                prev_state=self._agent_state,
                action=action,
                next_state=new_state,
                plume_field=self.plume,
            )
        )
        new_state.total_reward = self._agent_state.total_reward + reward

        dx = new_state.position.x - self._goal_location.x
        dy = new_state.position.y - self._goal_location.y
        distance_to_goal = float(np.sqrt(dx * dx + dy * dy))
        terminated = distance_to_goal <= self.goal_radius
        if terminated:
            new_state.goal_reached = True
            self._state = EnvironmentState.TERMINATED
        truncated = self._step_count >= self.max_steps
        if truncated and not terminated:
            self._state = EnvironmentState.TRUNCATED

        self._agent_state = new_state

        observation = self._get_observation()
        pos = self._agent_state.position
        info = {
            "agent_position": (pos.x, pos.y),
            "agent_xy": (pos.x, pos.y),
            "distance_to_goal": distance_to_goal,
            "goal_location": self.source_location,
            "source_location": self.source_location,
            "step_count": self._step_count,
            "total_reward": self._agent_state.total_reward,
            "goal_reached": terminated,
            "episode_count": self._episode_count,
        }
        if self._latest_seed is not None:
            info["seed"] = self._latest_seed
        return observation, reward, terminated, truncated, info

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        effective_mode = mode if mode is not None else self.render_mode
        if effective_mode not in {None, "human", "rgb_array"}:
            raise ValueError(f"Unsupported render mode: {effective_mode}")

        if effective_mode == "rgb_array":
            return self._render_rgb_array()

        # Human mode (or None) returns None; side effects are optional.
        return None

    def close(self) -> None:
        if self._state == EnvironmentState.CLOSED:
            return
        self._state = EnvironmentState.CLOSED
        self._agent_state = None

    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        self.np_random, seed_used = gym_seeding.np_random(seed)
        if seed_used is not None:
            seed_used = int(seed_used)
        self._latest_seed = seed_used
        self._seed = seed_used
        self._rng = self.np_random
        return [seed_used]

    def _ensure_ready_for_step(self) -> None:
        if self._state == EnvironmentState.CREATED:
            raise StateError("Must call reset() before step()")
        if self._state == EnvironmentState.CLOSED:
            raise StateError("Cannot step closed environment")
        if self._state != EnvironmentState.READY:
            raise StateError("Environment must be in READY state to step; call reset()")

    def _choose_start_position(self) -> Coordinates:
        if self._start_location is not None:
            return self._start_location
        rng = self._rng or self.np_random or np.random.default_rng()
        x = int(rng.integers(0, self._grid_size.width))
        y = int(rng.integers(0, self._grid_size.height))
        return Coordinates(x=x, y=y)

    def _reset_dynamic_fields(self) -> None:
        reset = getattr(self.plume, "reset", None)
        if callable(reset):
            reset(self._latest_seed)
        else:
            on_reset = getattr(self.plume, "on_reset", None)
            if callable(on_reset):
                on_reset()
        if self._wind_field is not None:
            wind_reset = getattr(self._wind_field, "reset", None)
            if callable(wind_reset):
                wind_reset(self._latest_seed)

    def _advance_dynamic_fields(self, step_count: int) -> None:
        advance = getattr(self.plume, "advance_to_step", None)
        if callable(advance):
            advance(step_count)
        if self._wind_field is not None:
            wind_advance = getattr(self._wind_field, "advance_to_step", None)
            if callable(wind_advance):
                wind_advance(step_count)

    def _get_plume_field(self) -> np.ndarray:
        field = getattr(self.plume, "field_array", None)
        if isinstance(field, np.ndarray):
            return field
        return self._sample_plume_field()

    def _sample_plume_field(self) -> np.ndarray:
        field = np.zeros(
            (self._grid_size.height, self._grid_size.width), dtype=np.float32
        )
        for y in range(self._grid_size.height):
            for x in range(self._grid_size.width):
                field[y, x] = float(self.plume.sample(x, y, self._step_count))
        return field

    def _build_env_state_dict(
        self, agent_state: Optional[AgentState] = None
    ) -> dict[str, Any]:
        return {
            "agent_state": agent_state or self._agent_state,
            "plume_field": self._get_plume_field(),
            "concentration_field": self.plume,
            "wind_field": self._wind_field,
            "goal_location": self._goal_location,
            "grid_size": self._grid_size,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "rng": self._rng,
        }

    def _get_observation(self) -> Any:
        return self.sensor_model.get_observation(self._build_env_state_dict())

    def _render_rgb_array(self) -> np.ndarray:
        height = self._grid_size.height
        width = self._grid_size.width
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        field = self._get_plume_field()
        if isinstance(field, np.ndarray) and field.shape == (height, width):
            flat = field.reshape(-1)
            finite = np.isfinite(flat)
            if finite.any():
                min_val = float(flat[finite].min())
                max_val = float(flat[finite].max())
                if max_val > min_val:
                    normalized = (field - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(field, dtype=np.float32)
            else:
                normalized = np.zeros_like(field, dtype=np.float32)
            grayscale = (np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
            canvas[:, :, :] = grayscale[:, :, None]

        sx, sy = self._goal_location.x, self._goal_location.y
        self._draw_marker(canvas, sx, sy, SOURCE_MARKER_SIZE, SOURCE_MARKER_COLOR)

        if self._agent_state is not None:
            ax, ay = int(self._agent_state.position.x), int(
                self._agent_state.position.y
            )
        elif self._start_location is not None:
            ax, ay = int(self._start_location.x), int(self._start_location.y)
        else:
            center = self._grid_size.center()
            ax, ay = int(center.x), int(center.y)
        self._draw_marker(canvas, ax, ay, AGENT_MARKER_SIZE, AGENT_MARKER_COLOR)

        return canvas

    def _draw_marker(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        size: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        height, width = canvas.shape[0], canvas.shape[1]
        if not (0 <= x < width and 0 <= y < height):
            return
        marker_w, marker_h = size
        half_w = max(int(marker_w) // 2, 0)
        half_h = max(int(marker_h) // 2, 0)
        y0 = max(0, y - half_h)
        y1 = min(height, y + half_h + 1)
        x0 = max(0, x - half_w)
        x1 = min(width, x + half_w + 1)
        canvas[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)


def create_plume_env(**kwargs: Any) -> PlumeEnv:
    """Create a PlumeEnv via direct kwargs or the canonical selector surface."""

    if "plume_type" in kwargs:
        raise ValidationError(
            "plume_type is no longer supported. Use plume='static' or plume='movie'.",
            parameter_name="plume_type",
            parameter_value=kwargs.get("plume_type"),
        )

    removed_video_keys = sorted(key for key in kwargs if key.startswith("video_"))
    if removed_video_keys:
        alias = removed_video_keys[0]
        raise ValidationError(
            f"{alias} is no longer supported. Use the matching movie_* selector kwarg.",
            parameter_name=alias,
            parameter_value=kwargs.get(alias),
        )

    selector_keys = {
        "goal_location",
        "action_type",
        "observation_type",
        "reward_type",
        "plume_sigma",
        "step_size",
        "enable_wind",
        "wind_direction_deg",
        "wind_speed",
        "wind_vector",
        "wind_noise_std",
    }
    uses_selector_route = any(key in kwargs for key in selector_keys) or any(
        key.startswith("movie_") for key in kwargs
    ) or isinstance(kwargs.get("plume"), str)

    if not uses_selector_route:
        return PlumeEnv(**kwargs)

    selector_kwargs = dict(kwargs)
    source_location = selector_kwargs.pop("source_location", None)
    if source_location is not None and "goal_location" not in selector_kwargs:
        selector_kwargs["goal_location"] = source_location

    plume_params = selector_kwargs.pop("plume_params", None)
    if plume_params is not None and "plume_sigma" not in selector_kwargs:
        sigma = plume_params.get("sigma") if isinstance(plume_params, dict) else None
        if sigma is not None:
            selector_kwargs["plume_sigma"] = float(sigma)

    from .factory import _create_plume_env_from_selectors

    return _create_plume_env_from_selectors(**selector_kwargs)
