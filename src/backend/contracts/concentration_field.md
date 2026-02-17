# Concentration Field Contract

Status: Alpha (living doc).

A concentration field represents the scalar chemical/odor concentration over the environment grid. It is used by the environment, reward functions, and observation models.

## Purpose

- Provide a deterministic way to sample concentration at a location (and optionally time).
- Optionally provide a dense 2D array representation (`field_array`) for fast sensor queries.

## Interface (current code)

Defined by `plume_nav_sim.plume.protocol.ConcentrationField`:

- `sample(x: float, y: float, t: float | None = None) -> float`
- `reset(seed: int | None = None) -> None`

Many implementations also expose (not required by the protocol):

- `grid_size: GridSize` (or equivalent)
- `field_array: np.ndarray` with shape `(height, width)`

## Universal Invariants (used by plume_nav_sim)

These are the expectations relied on by built-in environments and sensors.

- Determinism: with identical internal state and identical `(x, y, t)` inputs, `sample(...)` returns the same value.
- Purity: `sample(...)` should not mutate external state.
- Finiteness: `sample(...)` returns a finite Python `float`.
- Normalization (by convention in this project): concentrations are expected to be in `[0.0, 1.0]`.
  - If a model is not naturally normalized, it must document its range and clamp/scale for sensors that assume `[0, 1]`.

Out-of-bounds behavior must be defined. The current `GaussianPlume` returns `0.0` when sampling outside the grid.

## Array and Coordinate Conventions

If a plume exposes a dense array (`field_array`):

- `field_array.shape == (grid_size.height, grid_size.width)`
- Indexing is NumPy-style `[y, x]`.
- `Coordinates` are `(x, y)`.

## GaussianPlume (current default implementation)

The default plume used by `PlumeEnv` is `plume_nav_sim.plume.gaussian.GaussianPlume`.

Parameters:

- `grid_size`: `GridSize` or `(width, height)` with positive dimensions.
- `source_location`: `Coordinates` within the grid (default: grid center).
- `sigma`: float, must be `> 0`.

Generated field (precomputed `field_array`):

```
C(x, y) = exp(-((x-x_s)^2 + (y-y_s)^2) / (2 * sigma^2))

where (x_s, y_s) is source_location.
```

Implementation notes:

- `field_array` is `float32` and clipped to `[0.0, 1.0]`.
- `sample(x, y, t)` rounds `(x, y)` to the nearest integer indices.
- Out-of-bounds indices return `0.0`.

## Model-Specific Properties (Gaussian)

These properties are useful for tests and intuition but are not universal physical laws.

- Maximum at source: the maximum concentration occurs at (or includes) the source cell.
- Radial symmetry (discrete): points at the same Euclidean distance from the source tend to have equal concentrations (within discretization effects).
- Monotonic decay (discrete): concentration generally decreases as distance increases, with ties due to discretization/rounding.

## Recommended Tests

Universal:

- Values are finite.
- Values are in `[0, 1]`.
- Determinism: repeated calls return identical values.
- Reset determinism: `reset(seed); sample(...)` matches across runs with the same seed.

Gaussian-specific:

- Source cell is a maximum.
- Symmetry checks with tolerances (float32).

## Performance Notes

- Static plumes should precompute `field_array` so `sample(...)` is O(1).
- Avoid allocating large arrays in `sample(...)` or per-step in observation models.
