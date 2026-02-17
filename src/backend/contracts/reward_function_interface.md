# Reward Function Interface Contract

Status: Alpha (living doc).

Reward functions compute a scalar reward for a state transition. The environment injects a reward function so reward design can change independently of environment dynamics.

## Interface (current code)

Defined by `plume_nav_sim.interfaces.reward.RewardFunction`.

Required members:

- `compute_reward(prev_state, action, next_state, plume_field) -> float`
- `get_metadata() -> Dict[str, Any]`

Notes:

- `action` matches the action model's action type.
- `plume_field` is a `ConcentrationField` protocol object; built-in rewards also accept a raw NumPy array for backward compatibility.

## Universal Invariants

These are the properties relied on by `PlumeEnv`.

- Determinism:
  - identical inputs must produce identical reward.
  - no dependence on wall-clock time, call order, or hidden global RNG.
- Purity:
  - must not mutate `prev_state`, `next_state`, `plume_field`, or `action`.
  - no I/O side effects.
- Finiteness:
  - must return a finite Python `float` (no NaN/Inf).

If a reward function is stateful, the state must be fully controlled and documented. Prefer stateless reward functions.

## Metadata

`get_metadata()` must be JSON-serializable and should include:

- `type`: stable identifier
- parameters needed to reproduce behavior
- (optional) reward range information

## Recommended Tests

- Determinism: `compute_reward(...)` called twice returns the same value.
- Purity: inputs unchanged after `compute_reward(...)`.
- Finiteness: returned value is finite for representative transitions.
