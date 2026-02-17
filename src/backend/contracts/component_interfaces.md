# Component Interfaces Contract

Status: Alpha (living doc).

`PlumeEnv` is intentionally built around a small set of swappable components. This contract defines the boundaries between those components so new movement models, sensors, and reward functions can be added without rewriting the environment.

## Components

`PlumeEnv` composes four primary components:

- `plume: ConcentrationField`
  - Source of concentration samples / fields.
- `action_model: ActionProcessor`
  - Defines `action_space` and state transition semantics.
- `sensor_model: ObservationModel`
  - Defines `observation_space` and maps environment state to observations.
- `reward_fn: RewardFunction`
  - Computes reward from a state transition.

See the interface protocols in `plume_nav_sim/interfaces/`.

## PlumeEnv Public Surface (for wrappers)

The environment exposes (at minimum) these attributes for wrappers and downstream tools:

- `action_space`, `observation_space` (derived from injected components)
- `action_model`, `sensor_model`, `reward_fn`, `plume`
- `grid_size` (tuple[int, int]) and `source_location` (tuple[int, int])
- `goal_radius: float`, `max_steps: int`, `render_mode: str | None`

Wrappers should treat the injected components and the environment's internal state as read-only.

## Step Data Flow (conceptual)

1. Validate action.
2. Compute next state:
   - `next_state = action_model.process_action(action, agent_state, grid_size)`
3. Advance plume (if supported).
4. Compute reward:
   - `reward = reward_fn.compute_reward(prev_state, action, next_state, plume)`
5. Update environment state (agent state, counters, termination flags).
6. Assemble `env_state` and compute observation:
   - `obs = sensor_model.get_observation(env_state)`
7. Return `(obs, reward, terminated, truncated, info)`.

## EnvState Dictionary

Observation models receive a dictionary assembled by the environment. The concrete set can evolve, but the current environment supplies keys consistent with `plume_nav_sim.core.types.EnvState`.

Typical keys:

- `agent_state: AgentState | None`
- `plume_field: np.ndarray` (2D array indexed `[y, x]`)
- `concentration_field: ConcentrationField` (the plume object)
- `grid_size: GridSize`
- `goal_location: Coordinates` (same as `source_location` in the default env)
- `step_count: int`
- `max_steps: int`
- `rng: np.random.Generator | None`

Optional keys may exist (for example `wind_field`). Observation models must:

- Treat `env_state` as read-only.
- Only rely on keys they declare in `get_metadata()["required_state_keys"]`.

## Swappability Requirements

- `action_space` must come from `action_model.action_space`.
- `observation_space` must come from `sensor_model.observation_space`.
- The environment should not special-case a particular action model / sensor model / reward function beyond optional hooks.

## Optional Hooks

Components may expose optional methods that the environment will call if present:

- `plume.reset(seed)` or `plume.on_reset()`
- `plume.advance_to_step(step_count)`
- `action_model.set_rng(rng)`

These hooks must be safe to call repeatedly and must preserve determinism under fixed seeds.

## Minimal Integration Tests (recommended)

- Construct `PlumeEnv` with each component swapped out and verify `reset()` and `step()` run.
- Verify `action_space` / `observation_space` reflect the injected components.
- Verify `sensor_model.get_observation(...)` only reads `env_state` (no mutation).

## Related Contracts

- `action_processor_interface.md`
- `observation_model_interface.md`
- `reward_function_interface.md`
- `environment_state_machine.md`
- `gymnasium_api.md`
- `core_types.md`
