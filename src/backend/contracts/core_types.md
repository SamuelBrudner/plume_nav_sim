# Core Data Types Contract

Status: Alpha (living doc).

This document summarizes the core types used across `plume_nav_sim`. The authoritative definitions live in `plume_nav_sim.core.types`.

## Coordinate and Array Conventions

- `Coordinates` represent positions as `(x, y)`.
- NumPy arrays are indexed `[y, x]`.
- Use `Coordinates.to_array_index(grid_size)` when mapping a coordinate to an array index.

This convention is relied on by the built-in concentration sensor and plume field generation.

## Coordinates

Defined as a frozen dataclass:

```python
@dataclass(frozen=True)
class Coordinates:
    x: int
    y: int
```

Semantics:

- `x` and `y` are integers.
- Instances are immutable and hashable.
- Negative values are allowed for generic coordinates, but environment positions must be within bounds.

Key methods:

- `to_tuple() -> tuple[int, int]`
- `is_within_bounds(grid_bounds: GridSize) -> bool`
- `distance_to(other: Coordinates) -> float` (Euclidean)
- `move(movement: Action | tuple[int, int], bounds: GridSize | None = None) -> Coordinates`
  - If `bounds` is provided, the moved coordinate is clamped into bounds.
- `to_array_index(grid_bounds: GridSize) -> tuple[int, int]` returning `(y, x)`

## GridSize

Defined as a frozen dataclass:

```python
@dataclass(frozen=True)
class GridSize:
    width: int
    height: int
```

Semantics:

- `width` and `height` are intended to be positive integers.
- Bounds checks use `0 <= x < width` and `0 <= y < height`.

Key methods:

- `to_tuple() -> tuple[int, int]`
- `total_cells() -> int`
- `center() -> Coordinates`
- `contains(coord: Coordinates) -> bool`
- `estimate_memory_mb(field_dtype: np.dtype | None = None) -> float`

## AgentState

Defined as a mutable dataclass:

```python
@dataclass
class AgentState:
    position: Coordinates
    orientation: float = 0.0
    step_count: int = 0
    total_reward: float = 0.0
    goal_reached: bool = False
```

Semantics:

- `position` is the agent's current location on the grid.
- `orientation` is reserved for orientation-aware action models. The common convention is degrees, but this is not currently enforced.
- `step_count` and `total_reward` are episode counters.
- `goal_reached` indicates the episode reached its terminal goal condition.

Key methods:

- `update_position(new_position: Coordinates) -> None`
- `add_reward(reward: float) -> None`
- `increment_step() -> None`
- `mark_goal_reached() -> None`

Environment invariants (when `AgentState` is owned by `PlumeEnv`):

- `grid_size.contains(agent_state.position)` is always true.
- `step_count` is non-decreasing within an episode and resets on `reset()`.

## Helper Constructors (common utilities)

`plume_nav_sim.core.types` also provides helpers that coerce user inputs:

- `create_coordinates(...) -> Coordinates`
- `create_grid_size(...) -> GridSize`

These are used to normalize inputs from configs / Gymnasium wrappers.
