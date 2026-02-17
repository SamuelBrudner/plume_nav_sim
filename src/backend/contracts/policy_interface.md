# Policy Interface Contract

Status: Alpha (living doc).

Policies (controllers) map observations to actions. The environment itself does not require a policy, but this interface makes it easier to run benchmarks and reference agents consistently.

## Interface (current code)

Defined by `plume_nav_sim.interfaces.policy.Policy`.

Required members:

- `action_space: gym.Space`
  - Must be stable per instance (same object each access).
- `reset(*, seed: int | None = None) -> None`
- `select_action(observation, *, explore: bool = True) -> ActionType`

`ActionType` and `ObservationType` come from `plume_nav_sim.core.types`.

## Universal Invariants

- Action-space containment:
  - `policy.action_space.contains(policy.select_action(obs)) is True`.
- Seed determinism:
  - resetting with the same seed and providing the same observation sequence must yield the same action sequence.
- No mutation:
  - `select_action(...)` must not mutate the input observation.
- Stable action space:
  - `policy.action_space` is treated as immutable after construction.

## Notes on `explore`

- `explore=True` enables exploratory stochasticity.
- `explore=False` requests greedy/low-noise behavior.
- Regardless of `explore`, behavior must remain seed-deterministic when inputs and seeds match.

## Recommended Tests

- `action_space.contains(action)` for representative observations.
- Reset determinism (same seed, same observations).
- Observation immutability (arrays/dicts unchanged after `select_action`).
