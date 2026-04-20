# Component Injection (Public API)

This guide shows how to assemble the environment from pluggable components using
protocol-based dependency injection. It covers direct assembly, config-driven
creation, Gymnasium registration, and how to write your own components.

- Actions: `plume_nav_sim.interfaces.ActionProcessor`
- Observations: `plume_nav_sim.interfaces.ObservationModel`
- Rewards: `plume_nav_sim.interfaces.RewardFunction`
- Env: `plume_nav_sim.envs.PlumeEnv`

See also: contracts for each interface under `contracts/*.md`.

---

## Overview

- Components define spaces and behavior (no inheritance required; duck typing is used).
- The environment delegates to injected components for step, observation, and reward.
- You can wire components directly, via selectors, or register a Gymnasium ID that
  uses `PlumeEnv` under the hood.

---

## Environment State Schema (`env_state`)

Observation models compute their output from a dictionary assembled by the env:

- `agent_state`: `AgentState` (position, orientation, counters)
- `plume_field`: 2D `numpy.ndarray` concentration field (or full object if needed)
- `grid_size`: `GridSize` (width, height)
- `time_step`: `int` (optional)
- `goal_location`: `Coordinates` (optional)

Implementations may ignore unused keys. Custom envs/wrappers can override
`_build_env_state_dict()` (or `_get_env_state()` in legacy docs) to add keys.

---

## Quick Start: Selector Assembly

```python
import plume_nav_sim as pns

env = pns.make_env(
    grid_size=(128, 128),
    goal_location=(64, 64),
    action_type='oriented',        # 'discrete', 'oriented', or 'run_tumble'
    observation_type='antennae',   # 'concentration' or 'antennae'
    reward_type='step_penalty',    # 'sparse' or 'step_penalty'
    goal_radius=2.0,
)
obs, info = env.reset(seed=42)
```

- `action_space` is taken from the action processor
- `observation_space` is taken from the observation model

---

## Manual Wiring (Advanced)

```python
import numpy as np
from plume_nav_sim.actions import DiscreteGridActions
from plume_nav_sim.core.geometry import Coordinates, GridSize
from plume_nav_sim.envs import PlumeEnv
from plume_nav_sim.observations import ConcentrationSensor
from plume_nav_sim.plume.gaussian import GaussianPlume
from plume_nav_sim.rewards import SparseGoalReward

# Components
grid = GridSize(64, 64)
goal = Coordinates(32, 32)

actions = DiscreteGridActions(step_size=1)
obs_model = ConcentrationSensor()
reward_fn = SparseGoalReward(goal_position=goal, goal_radius=1.0)

# Simple Gaussian field
field = GaussianPlume(grid_size=grid, source_location=goal, sigma=20.0)

# Env
env = PlumeEnv(
    grid_size=grid,
    source_location=goal,
    plume=field,
    action_model=actions,
    sensor_model=obs_model,
    reward_fn=reward_fn,
)
```

---

## Config-Driven Creation

Use `SimulationSpec` for reproducible selector-driven experiments. The older
component-config models remain compatibility adapters only.

```python
from plume_nav_sim.config import SimulationSpec, build_env

spec = SimulationSpec(
    grid_size=(128, 128),
    source_location=(64, 64),
    max_steps=1000,
    action_type='oriented',
    observation_type='antennae',
    reward_type='step_penalty',
    goal_radius=2.0,
    plume_sigma=20.0,
)

env = build_env(spec)
```

See also (external plug‑and‑play, config‑based DI):

- Demo README section “Config‑based DI via SimulationSpec”: `plug-and-play-demo/README.md`
- Minimal, runnable spec file: `plug-and-play-demo/configs/simulation_spec.json`
- Stable notebook showing `SimulationSpec` env construction: `notebooks/stable/di_simulation_spec_component_env.ipynb`

---

## Writing Custom Components

You can satisfy protocols by shape (duck typing) without subclassing.

### RewardFunction

Use `ActionType` from `plume_nav_sim.interfaces` (discrete int or Box vector)
to stay aligned with your action processor.

```python
from typing import Any, Dict
from plume_nav_sim.core.state import AgentState
from plume_nav_sim.interfaces import ActionType

class MyReward:
    def compute_reward(self, prev_state: AgentState, action: ActionType, next_state: AgentState, plume_field: Any) -> float:
        return 1.0 if next_state.position == prev_state.position else 0.0

    def get_metadata(self) -> Dict[str, Any]:
        return {'type': 'my_reward', 'parameters': {}}
```

### ObservationModel

```python
import numpy as np
import gymnasium as gym
from typing import Any, Dict

class MyObservation:
    def __init__(self):
        self._space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

    @property
    def observation_space(self) -> gym.Space:
        return self._space

    def get_observation(self, env_state: Dict[str, Any]):
        # Two features: normalized x,y in [0,1]
        pos = env_state['agent_state'].position
        grid = env_state['grid_size']
        return np.array([pos.x / grid.width, pos.y / grid.height], dtype=np.float32)

    def get_metadata(self) -> Dict[str, Any]:
        return {'type': 'my_observation', 'parameters': {}}
```

### ActionProcessor

```python
import gymnasium as gym
from plume_nav_sim.core.geometry import Coordinates

class MyActions:
    def __init__(self):
        self._space = gym.spaces.Discrete(2)  # 0=stay, 1=right

    @property
    def action_space(self) -> gym.Space:
        return self._space

    def process_action(self, action, current_state, grid_size):
        if action == 1:
            new_x = min(current_state.position.x + 1, grid_size.width - 1)
            return type(current_state)(position=Coordinates(new_x, current_state.position.y), orientation=current_state.orientation)
        return current_state

    def validate_action(self, action) -> bool:
        return action in (0, 1)

    def get_metadata(self) -> dict:
        return {'type': 'my_actions', 'parameters': {'n_actions': 2}}
```

---

## Testing Your Components

All implementations should pass the universal property tests:

- Rewards: `tests/contracts/test_reward_function_interface.py`
- Observations: `tests/contracts/test_observation_model_interface.py`
- Actions: `tests/contracts/test_action_processor_interface.py`

Create a concrete test that inherits from the interface suite and provides a fixture
returning your implementation.

---

## Backward Compatibility

For new work, target `PlumeEnv` plus either selector kwargs or manual component
injection.
