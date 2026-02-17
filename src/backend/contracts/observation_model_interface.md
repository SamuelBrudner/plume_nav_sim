# Observation Model Interface Contract

Status: Alpha (living doc).

Observation models define how environment state is converted into an observation that matches a Gymnasium `observation_space`. This makes sensors swappable without changing environment dynamics.

## Interface (current code)

Defined by `plume_nav_sim.interfaces.observation.ObservationModel`.

Required members:

- `observation_space: gym.Space`
  - Must be stable per instance (same object each access).
- `get_observation(env_state: EnvState) -> ObservationType`
- `get_metadata() -> Dict[str, Any]`

`ObservationType` may be:

- `np.ndarray` (most common)
- `dict[str, Any]` (for composite spaces)
- `tuple[Any, ...]` (for tuple spaces)

The returned value must always be contained by `observation_space`.

## Universal Invariants

### 1. Space Containment

For any `env_state` provided by the environment:

- `observation_space.contains(get_observation(env_state)) is True`

### 2. Determinism

With identical inputs, the returned observation must be identical.

If an observation model requires randomness (e.g., sensor noise), it must be fully controlled by an explicit RNG/seed and documented in metadata.

### 3. Purity / No Mutation

- `get_observation(...)` must not mutate `env_state`.
- It must not mutate arrays stored inside `env_state` (e.g., `plume_field`).

### 4. Shape and Dtype Stability

- For array observations, shape and dtype must remain consistent with `observation_space`.
- Prefer finite values (no NaN/Inf) unless the `observation_space` explicitly allows them.

## Required env_state Keys

Observation models must only rely on keys they explicitly require.

Recommendation:

- Include `required_state_keys` in `get_metadata()`.

Built-in example:

- `ConcentrationSensor` requires `agent_state` and `plume_field`.

## Minimal Example

```python
# See: plume_nav_sim.observations.concentration.ConcentrationSensor
obs = sensor_model.get_observation(env_state)
assert sensor_model.observation_space.contains(obs)
```

## Recommended Tests

- Space containment for representative `env_state` values.
- Determinism: identical `env_state` yields identical observations.
- Purity: `env_state` and its arrays are not mutated.
- If the model is stateful, ensure behavior is documented and reset semantics are explicit.

## Metadata

`get_metadata()` should be JSON-serializable and include:

- `type` and `modality` (e.g., `olfactory`, `wind`)
- `parameters`
- `required_state_keys`
- observation shape/dtype information when helpful
