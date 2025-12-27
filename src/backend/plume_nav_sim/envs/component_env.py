"""
Component-based environment using dependency injection.

This module implements a Gymnasium environment that delegates to injected
components (ActionProcessor, ObservationModel, RewardFunction) rather than
using abstract methods. This architecture enables component swapping and
testing without modifying the environment class.

Contract: environment_state_machine.md

Key Features:
- Component dependency injection (no inheritance required)
- Protocol-based interfaces (duck typing)
- Clean separation of concerns
- Easy testing and component composition

Architecture:
    Environment → ActionProcessor (for actions)
               → ObservationModel (for observations)
               → RewardFunction (for rewards)
               → ConcentrationField (plume model)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import gymnasium as gym
import numpy as np

from ..core.constants import (
    AGENT_MARKER_COLOR,
    AGENT_MARKER_SIZE,
    SOURCE_MARKER_COLOR,
    SOURCE_MARKER_SIZE,
)
from ..core.geometry import Coordinates, GridSize
from ..core.state import AgentState
from ..utils.exceptions import StateError, ValidationError

if TYPE_CHECKING:
    from ..interfaces import (
        ActionProcessor,
        ObservationModel,
        RewardFunction,
        VectorField,
    )
    from ..plume.concentration_field import ConcentrationField

__all__ = ["ComponentBasedEnvironment", "EnvironmentState"]

logger = logging.getLogger(__name__)


class EnvironmentState(Enum):
    """Formal states for Environment lifecycle.

    Contract: environment_state_machine.md - State Definition
    """

    CREATED = "created"  # After __init__(), must reset() before step()
    READY = "ready"  # After reset(), can step()
    TERMINATED = "terminated"  # Episode ended (goal reached)
    TRUNCATED = "truncated"  # Episode timeout
    CLOSED = "closed"  # Resources released, terminal state


class ComponentBasedEnvironment(gym.Env):
    """
    Gymnasium environment using component dependency injection.

    This environment delegates to injected components rather than using
    abstract methods, enabling flexible composition and testing.

    Contract: environment_state_machine.md

    Components (injected):
        action_processor: Processes actions and computes new agent states
        observation_model: Generates observations from environment state
        reward_function: Computes rewards based on state transitions
        concentration_field: Plume model for odor concentration
        wind_field: Wind model providing velocity vectors (optional)

    Attributes:
        action_space: From action_processor.action_space
        observation_space: From observation_model.observation_space
        grid_size: Spatial bounds for the environment
        max_steps: Episode step limit
        goal_location: Target position for agent
        goal_radius: Success threshold distance

    State Machine:
        CREATED --reset()--> READY --step()--> {READY, TERMINATED, TRUNCATED}
        {TERMINATED, TRUNCATED} --reset()--> READY
        * --close()--> CLOSED
    """

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
    ):
        """
        Initialize environment with injected components.

        Contract: environment_state_machine.md - __init__() preconditions

        Args:
            action_processor: Component for action processing
            observation_model: Component for observation generation
            reward_function: Component for reward calculation
            concentration_field: Plume model for odor
            wind_field: Wind model providing velocity vectors (optional)
            grid_size: Spatial bounds (width, height)
            max_steps: Episode step limit
            goal_location: Target position
            goal_radius: Success threshold
            start_location: Initial agent position (default: grid center)
            render_mode: Visualization mode ('rgb_array' or 'human')

        Raises:
            ValidationError: If parameters invalid
            TypeError: If components don't satisfy protocols

        Postconditions:
            - self._state = CREATED
            - action_space and observation_space assigned from components
            - No episode active
        """
        super().__init__()

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
        """
        Begin new episode, transition to READY state.

        Contract: environment_state_machine.md - reset() contract

        Args:
            seed: RNG seed for reproducibility
            options: Additional reset options (unused)

        Returns:
            observation: From observation_model.get_observation()
            info: Episode metadata

        Raises:
            StateError: If state == CLOSED
            ValidationError: If seed invalid

        Postconditions:
            - self._state = READY
            - self._step_count = 0
            - self._episode_count incremented
            - Agent positioned at start_location
        """
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
        """
        Execute action, advance one timestep.

        Contract: environment_state_machine.md - step() contract

        Args:
            action: Action from action_space

        Returns:
            observation: Next observation
            reward: Reward for this transition
            terminated: Whether episode ended (goal reached)
            truncated: Whether episode timed out
            info: Step metadata

        Raises:
            StateError: If state not READY
            ValidationError: If action invalid

        Postconditions:
            - self._step_count incremented
            - self._state may transition to TERMINATED or TRUNCATED
        """
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
        """
        Release resources, transition to CLOSED state.

        Contract: environment_state_machine.md - close() contract

        Postconditions:
            - self._state = CLOSED
            - Subsequent operations raise StateError

        Idempotent: Multiple calls safe
        """
        if self._state == EnvironmentState.CLOSED:
            return  # Idempotent

        self._state = EnvironmentState.CLOSED
        self._agent_state = None
        logger.info(
            f"Environment closed after {self._episode_count} episodes, "
            f"{self._step_count} total steps"
        )

    def render(self) -> Optional[np.ndarray]:
        """
        Generate visualization composed of the concentration field with agent/source markers.

        Contract: environment_state_machine.md - render() contract

        Returns:
            RGB array if render_mode='rgb_array', else None
        """
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
        try:
            on_reset = getattr(self._concentration_field, "on_reset", None)
            if callable(on_reset):
                on_reset()
        except Exception:
            # Defensive: dynamic hooks are optional
            pass

        try:
            wind_reset = getattr(self, "_wind_field", None)
            on_reset = getattr(wind_reset, "on_reset", None)
            if callable(on_reset):
                on_reset()
        except Exception:
            # Defensive: dynamic hooks are optional
            pass

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
            flat = field_array.reshape(-1)
            finite_mask = np.isfinite(flat)
            if finite_mask.any():
                finite_vals = flat[finite_mask]
                field_min = float(finite_vals.min())
                field_max = float(finite_vals.max())
                normalized = np.zeros_like(flat, dtype=np.float32)
                if field_max > field_min:
                    span = field_max - field_min
                    normalized[finite_mask] = (finite_vals - field_min) / span
                normalized_2d = normalized.reshape(field_array.shape)
            else:
                normalized_2d = np.zeros_like(field_array, dtype=np.float32)

            grayscale = (normalized_2d * 255.0).astype(np.uint8)
            canvas[:, :, :] = grayscale[:, :, None]

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
        """
        Build environment state dictionary for component protocols.

        Contract: observation_model_interface.md, reward_function_interface.md

        Args:
            agent_state: Override current agent state (for lookahead)

        Returns:
            Dictionary with required keys for component protocols
        """
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
