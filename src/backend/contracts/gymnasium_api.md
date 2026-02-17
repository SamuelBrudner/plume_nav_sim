# Gymnasium API Contract

Status: Alpha (living doc).

This document describes the public Gymnasium-facing behavior of `plume_nav_sim.envs.plume_env.PlumeEnv`.

## Scope

- Target API: Gymnasium v0.29+.
- This contract describes the *external* API surface consumed by RL libraries.

## Required Attributes

`PlumeEnv` must expose:

- `action_space: gymnasium.Space`
- `observation_space: gymnasium.Space`
- `metadata: dict` including `render_modes`

Spaces are derived from injected components:

- `action_space == action_model.action_space`
- `observation_space == sensor_model.observation_space`

Spaces should be treated as immutable after environment construction.

## reset()

Signature:

```python
reset(*, seed: int | None = None, options: dict | None = None) -> (observation, info)
```

Preconditions:

- Environment is not closed.

Postconditions:

- Environment state becomes `READY`.
- Returned observation satisfies `observation_space.contains(observation)`.
- `info` is a `dict[str, Any]`.

Seeding:

- Passing a seed should make the episode start state reproducible (given deterministic components).

## step()

Signature:

```python
step(action) -> (observation, reward, terminated, truncated, info)
```

Preconditions:

- `reset()` has been called and the environment is in `READY`.
- `action` is valid for the action model.

Postconditions:

- Returned observation satisfies `observation_space.contains(observation)`.
- `reward` is a finite `float`.
- `terminated` and `truncated` are booleans.
- `info` is a `dict[str, Any]`.

Action validation:

- If `action_model.validate_action(...)` exists, the environment uses it.
- Otherwise, the environment falls back to `action_space.contains(action)`.
- Invalid actions raise `ValidationError`.

## Info Dictionary

The `info` dictionary is intentionally flexible. The environment currently includes a set of common fields for debugging and analysis.

Recommended stable keys (present in both `reset()` and `step()`):

- `step_count: int`
- `episode_count: int`
- `total_reward: float`
- `goal_reached: bool`
- `agent_xy: tuple[int, int]` (agent position)

Common keys returned by `reset()`:

- `seed: int | None`
- `goal_location: tuple[int, int]` (same as `source_location` in the default env)
- `source_location: tuple[int, int]`

Common keys returned by `step()`:

- `distance_to_goal: float`

Notes:

- Additional keys may be added without breaking the API.
- `total_reward` may be negative when reward functions include step penalties.

## render()

Supported render modes are advertised in `metadata["render_modes"]`.

- `render(mode="rgb_array")` returns `np.ndarray` with shape `(H, W, 3)` and dtype `uint8`.
- `render(mode="human")` returns `None` (side effects are optional).

Rendering should not affect `step()` determinism.

## close()

- `close()` is idempotent.
- After closing, calling `reset()` or `step()` raises `StateError`.

## Determinism (high level)

With deterministic components, the environment should be deterministic given:

- the seed passed to `reset(seed=...)`
- the action sequence

If a component is stochastic, its randomness must be controlled by the environment-provided seed/RNG.
