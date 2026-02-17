# Environment State Machine Contract

Status: Alpha (living doc).

This contract describes the lifecycle rules for `plume_nav_sim.envs.plume_env.PlumeEnv`.

## States

The environment uses `plume_nav_sim.envs.state.EnvironmentState`:

- `CREATED`: after `__init__()`. No episode has started.
- `READY`: after `reset()`. An episode is active and `step()` is allowed.
- `TERMINATED`: episode ended by terminal condition (goal reached).
- `TRUNCATED`: episode ended by timeout (`max_steps`).
- `CLOSED`: resources released; terminal state.

## Valid Transitions

- `__init__()` -> `CREATED`
- `reset()`:
  - `CREATED` -> `READY`
  - `READY` -> `READY` (start a new episode)
  - `TERMINATED` -> `READY`
  - `TRUNCATED` -> `READY`
  - `CLOSED` -> error
- `step()`:
  - `READY` -> `READY` (non-terminal)
  - `READY` -> `TERMINATED` (goal reached)
  - `READY` -> `TRUNCATED` (step limit reached)
  - otherwise -> error
- `close()`:
  - any non-closed state -> `CLOSED`
  - `CLOSED` -> `CLOSED` (idempotent)

## Method Contracts (summary)

### `reset(seed=..., options=...)`

Preconditions:

- State is not `CLOSED`.

Effects:

- Initializes RNG from the provided seed (or generates a seed if none).
- Initializes episode counters.
- Initializes agent state (including start location).
- Resets the plume if it supports `reset(seed)`.
- Returns `(observation, info)`.

Postconditions:

- State becomes `READY`.
- Returned observation is contained in `observation_space`.

### `step(action)`

Preconditions:

- State is `READY`.
- `action` must be valid for the action model.

Effects:

- Computes `next_state` via the action model.
- Advances the plume if it supports time stepping.
- Computes reward via `reward_fn`.
- Updates internal counters.
- Computes observation via `sensor_model`.
- Updates state to `TERMINATED`/`TRUNCATED` when appropriate.
- Returns `(observation, reward, terminated, truncated, info)`.

Postconditions:

- Returned observation is contained in `observation_space`.
- `reward` is a finite `float`.

### `close()`

- Idempotent.
- After close, `reset()` and `step()` raise `StateError`.

### `render(mode=None)`

- Supports `"rgb_array"` and `"human"` (see `gymnasium_api.md`).
- Rendering is best-effort and should not affect determinism of `step()`.

## Error Conditions

The environment raises `StateError` for invalid lifecycle usage:

- Calling `step()` before the first `reset()`.
- Calling `step()` after `TERMINATED` or `TRUNCATED` without an intervening `reset()`.
- Calling `reset()` or `step()` after `close()`.

## Determinism and Seeding

Under deterministic components, the environment should be deterministic given:

- the `reset(seed=...)` value
- the action sequence

Any stochasticity in components must be fully controlled by the environment-provided seed/RNG.
