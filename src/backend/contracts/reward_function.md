# Reward Function Contract (built-in rewards)

Status: Alpha (living doc).

This document describes the behavior of the built-in reward functions shipped in `plume_nav_sim.rewards`. For the universal interface contract, see `reward_function_interface.md`.

## Scope

Built-in reward functions are currently goal-distance based and ignore plume concentration.

- `SparseGoalReward`
- `StepPenaltyReward`

Both use Euclidean distance from the agent to `goal_position`.

## SparseGoalReward

Implementation: `plume_nav_sim.rewards.sparse_goal.SparseGoalReward`.

Parameters:

- `goal_position: Coordinates`
- `goal_radius: float` (must be `> 0`)

Reward definition:

- Let `d = distance(next_state.position, goal_position)`.
- Reward is:
  - `1.0` if `d <= goal_radius`
  - `0.0` otherwise

Key properties:

- Boundary is inclusive (`<=`).
- Deterministic and pure.
- Reward range is `[0.0, 1.0]`.

## StepPenaltyReward

Implementation: `plume_nav_sim.rewards.step_penalty.StepPenaltyReward`.

Parameters:

- `goal_position: Coordinates`
- `goal_radius: float` (must be `> 0`)
- `goal_reward: float` (default `1.0`, must be finite)
- `step_penalty: float` (default `0.01`, must be `>= 0`)

Reward definition:

- Let `d = distance(next_state.position, goal_position)`.
- Reward is:
  - `goal_reward` if `d <= goal_radius`
  - `-step_penalty` otherwise

Key properties:

- Boundary is inclusive (`<=`).
- Deterministic and pure.
- Reward range depends on parameters.

## Common Invariants (for built-in rewards)

- `compute_reward(...)` returns a finite `float`.
- Inputs are not mutated.
- `get_metadata()` is JSON-serializable and captures parameters.

## Recommended Tests

- Boundary behavior at `d == goal_radius`.
- Determinism: repeated calls with identical inputs.
- Finiteness (no NaN/Inf) for typical states.
