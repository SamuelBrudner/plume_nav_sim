# Extending plume_nav_sim (Researcher Entry Point)

This is the fastest path to plug in your own navigation logic.

`ComponentBasedEnvironment` composes 3 swappable components:
- `ActionProcessor`: maps policy action -> next `AgentState`
- `ObservationModel`: maps `EnvState` -> observation
- `RewardFunction`: maps transition -> scalar reward

## 1) Component Architecture

On each `env.step(action)` call:
1. `ActionProcessor.process_action(...)` computes the next state.
2. `RewardFunction.compute_reward(...)` scores the transition.
3. `ObservationModel.get_observation(env_state)` builds the observation.

Core protocols are defined in:
- `src/backend/plume_nav_sim/interfaces/action.py`
- `src/backend/plume_nav_sim/interfaces/observation.py`
- `src/backend/plume_nav_sim/interfaces/reward.py`

Minimal required signatures:

```python
# ActionProcessor
@property
def action_space(self) -> gym.Space: ...

def process_action(self, action: ActionType, current_state: AgentState, grid_size: GridSize) -> AgentState: ...

def validate_action(self, action: ActionType) -> bool: ...

def get_metadata(self) -> dict[str, Any]: ...

# ObservationModel
@property
def observation_space(self) -> gym.Space: ...

def get_observation(self, env_state: EnvState) -> ObservationType: ...

def get_metadata(self) -> dict[str, Any]: ...

# RewardFunction
def compute_reward(self, prev_state: AgentState, action: ActionType, next_state: AgentState, plume_field: ConcentrationField) -> float: ...

def get_metadata(self) -> dict[str, Any]: ...
```

## 2) Implementing Each Interface (Minimal)

```python
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from plume_nav_sim.core.types import AgentState, Coordinates, EnvState, GridSize
from plume_nav_sim.interfaces import ActionType


class MyActionProcessor:
    def __init__(self) -> None:
        self._action_space = gym.spaces.Discrete(2)  # 0=stay, 1=move right

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def process_action(
        self,
        action: ActionType,
        current_state: AgentState,
        grid_size: GridSize,
    ) -> AgentState:
        x = current_state.position.x
        if int(action) == 1:
            x = min(x + 1, grid_size.width - 1)
        return AgentState(
            position=Coordinates(x=x, y=current_state.position.y),
            orientation=current_state.orientation,
            step_count=current_state.step_count,
            total_reward=current_state.total_reward,
            goal_reached=current_state.goal_reached,
        )

    def validate_action(self, action: ActionType) -> bool:
        return isinstance(action, (int, np.integer)) and int(action) in (0, 1)

    def get_metadata(self) -> dict[str, Any]:
        return {"type": "my_action_processor"}


class MyObservationModel:
    def __init__(self) -> None:
        self._observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    def get_observation(self, env_state: EnvState) -> np.ndarray:
        agent_state = env_state["agent_state"]
        if agent_state is None:
            return np.zeros(2, dtype=np.float32)
        grid = env_state["grid_size"]
        return np.array(
            [agent_state.position.x / max(1, grid.width - 1), agent_state.position.y / max(1, grid.height - 1)],
            dtype=np.float32,
        )

    def get_metadata(self) -> dict[str, Any]:
        return {"type": "my_observation_model"}


class MyRewardFunction:
    def __init__(self, goal: Coordinates, goal_radius: float = 1.0) -> None:
        self.goal = goal
        self.goal_radius = float(goal_radius)

    def compute_reward(
        self,
        prev_state: AgentState,
        action: ActionType,
        next_state: AgentState,
        plume_field,
    ) -> float:
        dx = next_state.position.x - self.goal.x
        dy = next_state.position.y - self.goal.y
        return 1.0 if (dx * dx + dy * dy) ** 0.5 <= self.goal_radius else 0.0

    def get_metadata(self) -> dict[str, Any]:
        return {"type": "my_reward_function", "goal_radius": self.goal_radius}
```

## 3) Register and Use with `ComponentBasedEnvironment`

"Register" here means instantiate your components and inject them into the env.

```python
from plume_nav_sim.core.types import Coordinates, GridSize
from plume_nav_sim.envs.component_env import ComponentBasedEnvironment
from plume_nav_sim.interfaces import ActionProcessor, ObservationModel, RewardFunction
from plume_nav_sim.plume.gaussian import GaussianPlume

grid = GridSize(width=64, height=64)
goal = Coordinates(x=32, y=32)

action_processor = MyActionProcessor()
observation_model = MyObservationModel()
reward_function = MyRewardFunction(goal=goal, goal_radius=2.0)

# Optional runtime protocol checks (all interfaces are runtime_checkable)
assert isinstance(action_processor, ActionProcessor)
assert isinstance(observation_model, ObservationModel)
assert isinstance(reward_function, RewardFunction)

field = GaussianPlume(grid_size=grid, source_location=goal, sigma=12.0)

env = ComponentBasedEnvironment(
    action_processor=action_processor,
    observation_model=observation_model,
    reward_function=reward_function,
    concentration_field=field,
    grid_size=grid,
    goal_location=goal,
    goal_radius=2.0,
    max_steps=500,
)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## 4) `EnvState` Reference for Observation Providers

Observation models receive `EnvState` from `src/backend/plume_nav_sim/core/types.py`.
Available keys:
- `agent_state`
- `plume_field` (2D `numpy.ndarray`)
- `concentration_field` (full field object)
- `wind_field`
- `goal_location`
- `grid_size`
- `step_count`
- `max_steps`
- `rng`

Use this TypedDict as the source of truth when adding custom observation providers.

## Existing Implementations to Copy From

- Actions: `src/backend/plume_nav_sim/actions/`
- Observations: `src/backend/plume_nav_sim/observations/`
- Rewards: `src/backend/plume_nav_sim/rewards/`
- End-to-end example: `src/backend/examples/custom_components.py`
