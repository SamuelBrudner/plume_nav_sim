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

import numpy as np

import gymnasium as gym

from ..core.geometry import Coordinates, GridSize
from ..core.state import AgentState
from ..utils.exceptions import StateError, ValidationError

if TYPE_CHECKING:
    from ..interfaces import ActionProcessor, ObservationModel, RewardFunction
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

        # Gymnasium spaces from components
        # Contract: environment_state_machine.md - C5, C6
        self.action_space = action_processor.action_space
        self.observation_space = observation_model.observation_space

        # Environment configuration
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.goal_location = goal_location
        self.goal_radius = goal_radius
        self.start_location = start_location or Coordinates(
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

        # Seed RNG if provided
        if seed is not None:
            if not isinstance(seed, int) or seed < 0:
                raise ValidationError(f"Seed must be non-negative int, got {seed}")
            self._seed = seed
            np.random.seed(seed)
            logger.debug(f"Environment seeded: {seed}")

        # Reset episode counters
        # Contract: environment_state_machine.md - C2, C3
        self._step_count = 0
        self._episode_count += 1

        # Initialize agent state
        # Contract: environment_state_machine.md - C4
        self._agent_state = AgentState(
            position=self.start_location,
            orientation=0.0,
            step_count=0,
            total_reward=0.0,
        )

        # Transition to READY
        # Contract: environment_state_machine.md - C1
        self._state = EnvironmentState.READY

        # Generate initial observation
        env_state = self._build_env_state_dict()
        observation = self._observation_model.get_observation(env_state)

        # Build info dict
        # Contract: environment_state_machine.md - C6
        info = {
            "seed": self._seed,
            "episode": self._episode_count,
            "agent_position": (
                self._agent_state.position.x,
                self._agent_state.position.y,
            ),
            "goal_location": (self.goal_location.x, self.goal_location.y),
        }

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
        # Contract: environment_state_machine.md - P1
        if self._state == EnvironmentState.CREATED:
            raise StateError("Must call reset() before step()")
        if self._state == EnvironmentState.CLOSED:
            raise StateError("Cannot step closed environment")
        if self._state != EnvironmentState.READY:
            raise StateError("Environment must be in READY state to step; call reset()")

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

        # Update state machine
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

        # Generate observation
        # Delegation to ObservationModel
        env_state_dict = self._build_env_state_dict()
        observation = self._observation_model.get_observation(env_state_dict)

        # Build info dict
        info = {
            "step_count": self._step_count,
            "episode": self._episode_count,
            "agent_position": (
                self._agent_state.position.x,
                self._agent_state.position.y,
            ),
            "distance_to_goal": self._distance_to_goal(self._agent_state.position),
        }

        return observation, float(reward), terminated, truncated, info

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
        Generate visualization (not implemented in minimal version).

        Contract: environment_state_machine.md - render() contract

        Returns:
            RGB array if render_mode='rgb_array', else None
        """
        if self.render_mode == "rgb_array":
            # Placeholder: return blank image
            return np.zeros((100, 100, 3), dtype=np.uint8)
        return None

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
            "goal_location": self.goal_location,
            "grid_size": self.grid_size,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
        }

    def _check_goal_reached(self, position: Coordinates) -> bool:
        """Check if agent reached goal."""
        return self._distance_to_goal(position) <= self.goal_radius

    def _distance_to_goal(self, position: Coordinates) -> float:
        """Compute Euclidean distance to goal."""
        dx = position.x - self.goal_location.x
        dy = position.y - self.goal_location.y
        return float(np.sqrt(dx * dx + dy * dy))
