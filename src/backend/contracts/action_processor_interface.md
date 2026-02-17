# Action Processor Interface Contract

Status: Alpha (living doc).

Action processors define how an external action representation (Gymnasium action space) updates the agent's state while keeping the agent inside the grid.

## Purpose

- Decouple movement semantics from `PlumeEnv`.
- Enable swapping action spaces (discrete, oriented, continuous) without rewriting the environment.

## Interface (current code)

Defined by `plume_nav_sim.interfaces.action.ActionProcessor`.

Required members:

- `action_space: gym.Space`
  - Must be stable per instance (same object each access).
- `process_action(action, current_state, grid_size) -> AgentState`
- `validate_action(action) -> bool`
- `get_metadata() -> Dict[str, Any]`

`ActionType` is typically:

- `int` / NumPy integer for `gym.spaces.Discrete`.
- `np.ndarray` for `gym.spaces.Box`.

## Universal Invariants

These are the invariants we rely on across all action models.

### 1. Boundary Safety

If the current state is in-bounds, the returned state must also be in-bounds:

- Pre: `grid_size.contains(current_state.position) is True`
- Post: `grid_size.contains(result.position) is True`

Recommended behavior is to clamp to bounds (the current `DiscreteGridActions` does this), but any deterministic in-bounds policy is acceptable as long as it is documented.

### 2. No Mutation

`process_action(...)` must not mutate its inputs.

- Do not mutate `current_state` (return a new `AgentState`).
- Do not mutate `grid_size`.

Unless explicitly documented, the action model should preserve non-movement fields from `current_state` (orientation, reward counters, etc.). The environment updates `step_count` / `total_reward` at the environment layer.

### 3. Determinism

Given identical inputs, `process_action(...)` must return an identical result.

If an action model needs randomness, it must be controlled by an explicit RNG/seed (for example via a `set_rng(...)` hook that `PlumeEnv.reset(seed=...)` wires up).

### 4. Space Consistency

- `validate_action(action)` should agree with `action_space.contains(action)` for the supported action dtypes.
- `action_space` should be treated as immutable after construction.

## Error Handling

- `validate_action(...)` returns `bool` and should never raise for typical action inputs.
- `process_action(...)` may raise `ValueError`/`ValidationError` on invalid actions.
  - `PlumeEnv.step(...)` typically validates the action before calling `process_action(...)`.

## Minimal Example (clamped cardinal actions)

```python
# See: plume_nav_sim.actions.discrete_grid.DiscreteGridActions
# Semantics: absolute cardinal moves on the grid; position is clamped to bounds.

new_state = action_model.process_action(action, current_state, grid_size)
assert grid_size.contains(new_state.position)
```

## Recommended Tests

- Boundary safety: random in-bounds `current_state.position` + valid actions never yield out-of-bounds positions.
- Determinism: calling `process_action(...)` twice with the same inputs returns equal `AgentState`.
- No mutation: `current_state` is unchanged after calling `process_action(...)`.
- Space consistency: `validate_action(a)` matches `action_space.contains(a)` for representative samples.

## Metadata

`get_metadata()` should be JSON-serializable and include enough detail to reproduce behavior in logs:

- `type` (e.g. `"discrete_grid"`, `"oriented_grid"`)
- `modality` (e.g. `"absolute_cardinal"`, `"orientation_relative"`)
- `parameters` (step size, number of actions, etc.)
- `orientation_dependent: bool`
