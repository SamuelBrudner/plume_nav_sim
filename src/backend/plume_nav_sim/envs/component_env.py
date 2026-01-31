from __future__ import annotations

import contextlib
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import gymnasium as gym
import numpy as np

from .._compat import StateError, ValidationError
from ..constants import (
    AGENT_MARKER_COLOR,
    AGENT_MARKER_SIZE,
    SOURCE_MARKER_COLOR,
    SOURCE_MARKER_SIZE,
)
from ..core.types import AgentState, Coordinates, GridSize
from .state import EnvironmentState

if TYPE_CHECKING:
    from ..interfaces import (
        ActionProcessor,
        ObservationModel,
        RewardFunction,
        VectorField,
    )
    from ..plume.protocol import ConcentrationField

__all__ = ["ComponentBasedEnvironment", "EnvironmentState"]

logger = logging.getLogger(__name__)


class ComponentBasedEnvironment(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        *,
        action_processor: ActionProcessor,
        observation_model: ObservationModel,
        reward_function: RewardFunction,
        concentration_field: ConcentrationField,
        wind_field: Optional[VectorField] = None,
        grid_size: GridSize,
        max_steps: int = 1000,
        goal_location: Coordinates,
        goal_radius: float = 5.0,
        start_location: Optional[Coordinates] = None,
        render_mode: Optional[str] = None,
        _warn_deprecated: bool = True,
    ):
        super().__init__()
        if _warn_deprecated:
            warnings.warn(
                "ComponentBasedEnvironment is deprecated and will be removed in a future "
                "release. Use PlumeEnv or create_plume_env instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Contract: environment_state_machine.md - P1, P2, P3
        if max_steps <= 0:
            raise ValidationError(f"max_steps must be positive, got {max_steps}")
        if goal_radius < 0:
            raise ValidationError(
                f"goal_radius must be non-negative, got {goal_radius}"
            )
        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValidationError(
                f"render_mode must be in {self.metadata['render_modes']}, got {render_mode}"
            )

        # Store injected components
        self._action_processor = action_processor
        self._observation_model = observation_model
        self._reward_function = reward_function
        self._concentration_field = concentration_field
        self._wind_field = wind_field

        # Gymnasium spaces from components
        # Contract: environment_state_machine.md - C5, C6
        self.action_space = action_processor.action_space
        self.observation_space = observation_model.observation_space

        # Environment configuration
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.goal_location = goal_location
        self.goal_radius = goal_radius
        self._user_start_location = start_location
        if start_location is not None:
            self.start_location = start_location
        else:
            self.start_location = Coordinates(
                grid_size.width // 2, grid_size.height // 2
            )
        self.render_mode = render_mode

        # Episode state (initialized in reset())
        self._agent_state: Optional[AgentState] = None
        self._step_count: int = 0
        self._episode_count: int = 0
        self._seed: Optional[int] = None

        # State machine
        # Contract: environment_state_machine.md - C1
        self._state = EnvironmentState.CREATED

        logger.info(
            f"ComponentBasedEnvironment initialized: grid={grid_size}, "
            f"goal={goal_location}, max_steps={max_steps}"
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[Any, dict]:
        # Contract: environment_state_machine.md - P1
        if self._state == EnvironmentState.CLOSED:
            raise StateError("Cannot reset closed environment")

        # Seed RNG and init counters
        self._handle_seed(seed)
        self._reset_episode_counters()

        # Initialize agent state at chosen start position
        start_position = self._choose_start_position()
        self._init_agent_state(start_position)

        # Transition to READY and propagate RNG to components
        self._activate_ready_state()
        self._reset_concentration_dynamic_state()

        # Initial observation and info
        observation, info = self._initial_observation_and_info()

        logger.debug(
            f"Episode {self._episode_count} reset: agent at {self._agent_state.position}"
        )

        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        self._ensure_ready_for_step()

        # Validate action
        # Contract: environment_state_machine.md - P2
        if not self._action_processor.validate_action(action):
            raise ValidationError(f"Invalid action: {action}")

        # Process action to compute new agent state
        # Delegation to ActionProcessor
        new_agent_state = self._action_processor.process_action(
            action, self._agent_state, self.grid_size
        )

        # Update step count
        new_agent_state.step_count = self._step_count + 1
        self._step_count += 1

        # Advance dynamic plume state for this step if supported
        self._advance_dynamic_fields()

        # Compute reward
        # Delegation to RewardFunction
        # Protocol: compute_reward(prev_state, action, next_state, plume_field)
        reward = self._reward_function.compute_reward(
            prev_state=self._agent_state,
            action=action,
            next_state=new_agent_state,
            plume_field=self._concentration_field,
        )

        # Update total reward
        new_agent_state.total_reward = self._agent_state.total_reward + reward

        # Check termination conditions
        terminated = self._check_goal_reached(new_agent_state.position)
        truncated = self._step_count >= self.max_steps

        # Update agent state
        self._agent_state = new_agent_state

        self._update_state_after_step(terminated=terminated, truncated=truncated)

        # Generate observation
        # Delegation to ObservationModel
        env_state_dict = self._build_env_state_dict()
        observation = self._observation_model.get_observation(env_state_dict)

        # Build info dict
        info = self._build_step_info()

        return observation, float(reward), terminated, truncated, info

    def _ensure_ready_for_step(self) -> None:
        # Contract: environment_state_machine.md - P1
        if self._state == EnvironmentState.CREATED:
            raise StateError("Must call reset() before step()")
        if self._state == EnvironmentState.CLOSED:
            raise StateError("Cannot step closed environment")
        if self._state != EnvironmentState.READY:
            raise StateError("Environment must be in READY state to step; call reset()")

    def _advance_dynamic_field(self, field: Any) -> None:
        try:
            adv = getattr(field, "advance_to_step", None)
            if callable(adv):
                adv(self._step_count)
        except Exception:
            # Defensive: optional dynamic hook
            pass

    def _advance_dynamic_fields(self) -> None:
        self._advance_dynamic_field(self._concentration_field)
        self._advance_dynamic_field(self._wind_field)

    def _update_state_after_step(self, *, terminated: bool, truncated: bool) -> None:
        if terminated:
            self._state = EnvironmentState.TERMINATED
            self._agent_state.goal_reached = True
            logger.info(
                f"Episode {self._episode_count} terminated: "
                f"goal reached at step {self._step_count}"
            )
        elif truncated:
            self._state = EnvironmentState.TRUNCATED
            logger.info(
                f"Episode {self._episode_count} truncated: "
                f"max_steps={self.max_steps} reached"
            )

    def _build_step_info(self) -> dict[str, Any]:
        return {
            "step_count": self._step_count,
            "agent_position": (
                self._agent_state.position.x,
                self._agent_state.position.y,
            ),
            "distance_to_goal": self._distance_to_goal(self._agent_state.position),
        }

    def _sample_random_start_location(self) -> Coordinates:
        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = np.random.default_rng()
            self._rng = rng

        start_x = int(rng.integers(low=0, high=self.grid_size.width))
        start_y = int(rng.integers(low=0, high=self.grid_size.height))
        return Coordinates(x=start_x, y=start_y)

    def close(self) -> None:
        if self._state == EnvironmentState.CLOSED:
            return  # Idempotent

        self._state = EnvironmentState.CLOSED
        self._agent_state = None
        logger.info(
            f"Environment closed after {self._episode_count} episodes, "
            f"{self._step_count} total steps"
        )

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None

        height, width = self.grid_size.height, self.grid_size.width
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Compose layers
        self._render_field_grayscale(canvas, height, width)
        self._draw_source_marker(canvas, height, width)

        ax, ay = self._agent_position_xy(height, width)
        self._draw_agent_marker(canvas, height, width, ax, ay)

        return canvas

    # Private helper methods

    # ---- Episode helpers ----

    def _handle_seed(self, seed: Optional[int]) -> None:
        """Initialize RNG from seed or ensure RNG exists."""
        if seed is None:
            if not hasattr(self, "_rng"):
                self._rng = np.random.default_rng()
            return
        if not isinstance(seed, int) or seed < 0:
            raise ValidationError(f"Seed must be non-negative int, got {seed}")
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        logger.debug(f"Environment seeded: {seed}")

    def _reset_episode_counters(self) -> None:
        """Reset per-episode counters."""
        self._step_count = 0
        self._episode_count += 1

    def _choose_start_position(self) -> Coordinates:
        """Choose start position from user override or random sampling."""
        if self._user_start_location is not None:
            return self._user_start_location
        return self._sample_random_start_location()

    def _init_agent_state(self, start_position: Coordinates) -> None:
        """Initialize agent state for the episode."""
        self.start_location = start_position
        self._agent_state = AgentState(
            position=self.start_location,
            orientation=0.0,
            step_count=0,
            total_reward=0.0,
        )

    def _activate_ready_state(self) -> None:
        """Transition to READY and propagate RNG to action processor if supported."""
        self._state = EnvironmentState.READY
        if hasattr(self._action_processor, "set_rng"):
            try:
                self._action_processor.set_rng(self._rng)
            except Exception:
                # Optional integration; swallow to keep behaviour identical
                pass

    def _reset_concentration_dynamic_state(self) -> None:
        """Allow concentration and wind fields to reset any dynamic state."""
        with contextlib.suppress(Exception):
            reset = getattr(self._concentration_field, "reset", None)
            if callable(reset):
                reset(self._seed)
            else:
                on_reset = getattr(self._concentration_field, "on_reset", None)
                if callable(on_reset):
                    on_reset()
        with contextlib.suppress(Exception):
            wind_reset = getattr(self, "_wind_field", None)
            on_reset = getattr(wind_reset, "on_reset", None)
            if callable(on_reset):
                on_reset()

    def _initial_observation_and_info(self) -> tuple[Any, dict]:
        """Generate initial observation and info dict."""
        env_state = self._build_env_state_dict()
        observation = self._observation_model.get_observation(env_state)
        info = {
            "seed": self._seed,
            "agent_position": (
                self._agent_state.position.x,
                self._agent_state.position.y,
            ),
            "goal_location": (self.goal_location.x, self.goal_location.y),
        }
        return observation, info

    # ---- Rendering helpers ----

    def _render_field_grayscale(
        self, canvas: np.ndarray, height: int, width: int
    ) -> None:
        """Render the concentration field as grayscale onto the canvas."""
        field = getattr(self, "_concentration_field", None)
        field_array = getattr(field, "field_array", None)
        if isinstance(field_array, np.ndarray) and field_array.size == height * width:
            if not hasattr(self, "_render_norm_range"):
                self._render_norm_range = None

            flat = field_array.reshape(-1)
            finite_mask = np.isfinite(flat)
            if finite_mask.any():
                normalized_2d = self._extracted_from__render_field_grayscale_14(
                    flat, finite_mask, field, field_array
                )
            else:
                normalized_2d = np.zeros_like(field_array, dtype=np.float32)

            grayscale = (np.clip(normalized_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
            canvas[:, :, :] = grayscale[:, :, None]

    # TODO Rename this here and in `_render_field_grayscale`
    def _extracted_from__render_field_grayscale_14(
        self, flat, finite_mask, field, field_array
    ):
        finite_vals = flat[finite_mask]
        norm_range = getattr(self, "_render_norm_range", None)
        if norm_range is None:
            dataset_path = getattr(field, "_dataset_path", None)
            if isinstance(dataset_path, str) and Path(dataset_path).is_dir():
                try:
                    import zarr

                    root = zarr.open_group(dataset_path, mode="r")
                    stats = root.attrs.get("concentration_stats")
                    if isinstance(stats, dict):
                        quantiles = stats.get("quantiles")
                        stats_max = stats.get("max")
                        vmin = 0.0
                        vmax = (
                            float(
                                quantiles.get("q999", quantiles.get("q99", stats_max))
                            )
                            if isinstance(quantiles, dict)
                            else float(stats_max)
                        )
                        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                            norm_range = (vmin, vmax)
                except Exception:
                    norm_range = None

            if norm_range is not None:
                self._render_norm_range = norm_range

        if norm_range is not None:
            field_min, field_max = float(norm_range[0]), float(norm_range[1])
        else:
            field_min = float(finite_vals.min())
            field_max = float(finite_vals.max())
        normalized = np.zeros_like(flat, dtype=np.float32)
        if field_max > field_min:
            span = field_max - field_min
            normalized[finite_mask] = (finite_vals - field_min) / span
        result = normalized.reshape(field_array.shape)
        return result

    def _draw_source_marker(self, canvas: np.ndarray, height: int, width: int) -> None:
        """Draw source marker and optional dashed goal radius."""
        source = getattr(self, "goal_location", None)
        if source is None:
            return
        sx, sy = int(source.x), int(source.y)
        if not (0 <= sy < height and 0 <= sx < width):
            return

        sw, sh = SOURCE_MARKER_SIZE
        half_sw = max(int(sw) // 2, 0)
        half_sh = max(int(sh) // 2, 0)
        y0 = max(0, sy - half_sh)
        y1 = min(height, sy + half_sh + 1)
        x0 = max(0, sx - half_sw)
        x1 = min(width, sx + half_sw + 1)
        color_arr = np.array(SOURCE_MARKER_COLOR, dtype=np.uint8)
        canvas[y0:y1, x0:x1] = color_arr

        goal_radius = getattr(self, "goal_radius", None)
        if not (isinstance(goal_radius, (int, float)) and goal_radius > 0):
            return
        radius = float(goal_radius)
        max_dim = max(width, height)
        if radius >= max_dim * 2:
            return

        num_steps = max(int(2.0 * np.pi * radius), 16)
        angles = np.linspace(0.0, 2.0 * np.pi, num_steps, endpoint=False)
        dash_period = 8
        dash_on = 4
        for idx, theta in enumerate(angles):
            if (idx % dash_period) >= dash_on:
                continue
            cx = int(round(sx + radius * np.cos(theta)))
            cy = int(round(sy + radius * np.sin(theta)))
            if 0 <= cy < height and 0 <= cx < width:
                canvas[cy, cx] = color_arr

    def _agent_position_xy(self, height: int, width: int) -> tuple[int, int]:
        """Compute current agent pixel position with fallback to start/center."""
        agent_state = getattr(self, "_agent_state", None)
        if agent_state is not None and agent_state.position is not None:
            return int(agent_state.position.x), int(agent_state.position.y)
        start = getattr(self, "start_location", None)
        if start is not None:
            return int(start.x), int(start.y)
        return width // 2, height // 2

    def _draw_agent_marker(
        self, canvas: np.ndarray, height: int, width: int, ax: int, ay: int
    ) -> None:
        """Draw the agent marker if within bounds."""
        if not (0 <= ay < height and 0 <= ax < width):
            return
        aw, ah = AGENT_MARKER_SIZE
        half_aw = max(int(aw) // 2, 0)
        half_ah = max(int(ah) // 2, 0)
        y0 = max(0, ay - half_ah)
        y1 = min(height, ay + half_ah + 1)
        x0 = max(0, ax - half_aw)
        x1 = min(width, ax + half_aw + 1)
        canvas[y0:y1, x0:x1] = np.array(AGENT_MARKER_COLOR, dtype=np.uint8)

    # Private helper methods

    def _build_env_state_dict(
        self, agent_state: Optional[AgentState] = None
    ) -> dict[str, Any]:
        state = agent_state or self._agent_state

        return {
            "agent_state": state,
            "plume_field": self._concentration_field.field_array,  # Numpy array for ObservationModel
            "concentration_field": self._concentration_field,  # Full object (if needed)
            "wind_field": self._wind_field,
            "goal_location": self.goal_location,
            "grid_size": self.grid_size,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "rng": getattr(self, "_rng", None),
        }

    def _check_goal_reached(self, position: Coordinates) -> bool:
        """Check if agent reached goal."""
        return self._distance_to_goal(position) <= self.goal_radius

    def _distance_to_goal(self, position: Coordinates) -> float:
        """Compute Euclidean distance to goal."""
        dx = position.x - self.goal_location.x
        dy = position.y - self.goal_location.y
        return float(np.sqrt(dx * dx + dy * dy))
