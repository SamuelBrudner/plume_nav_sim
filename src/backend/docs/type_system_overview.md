# Core Type System Overview

This page describes the canonical core types implemented in `plume_nav_sim.core.types` and
how to use them consistently across the codebase.

## What Exists (Single Source of Truth)
- Module: `plume_nav_sim.core.types` (implemented)
  - Data classes: `Coordinates`, `GridSize`, `AgentState`
  - Config: `EnvironmentConfig` (in `plume_nav_sim.envs.config_types`)
  - Enums (re-exported): `Action`, `RenderMode`
  - Plume params are plain dicts (e.g., `{"source_location": Coordinates, "sigma": float}`)
  - Type aliases: `ActionType`, `CoordinateType`, `GridDimensions`, `MovementVector`
  - Factories: `create_coordinates`, `create_grid_size`
  - Utilities: `validate_action`, `get_movement_vector`, `calculate_euclidean_distance`

Core types are defined in `src/backend/plume_nav_sim/core/types.py`.

## Usage Examples
Create validated coordinates and grid sizes:

```python
from plume_nav_sim.core.types import create_coordinates, create_grid_size

pos = create_coordinates(5, 10)           # Coordinates(x=5, y=10)
grid = create_grid_size((128, 128))       # GridSize(width=128, height=128)
```

Build environment configuration (with normalized plume parameters):

```python
from plume_nav_sim.envs.config_types import EnvironmentConfig, create_environment_config

cfg = create_environment_config(
    {
        "grid_size": (64, 64),
        "source_location": (32, 32),
        "max_steps": 500,
        "goal_radius": 2.0,
        "plume_params": {"sigma": 12.0},
    }
)

assert isinstance(cfg.grid_size, type(create_grid_size((1,1))))
```

Work with agent and episode state:

```python
from plume_nav_sim.core.types import AgentState, Coordinates

agent = AgentState(position=Coordinates(10, 10), orientation=90.0)
```

Validate and interpret actions:

```python
from plume_nav_sim.core.types import validate_action, get_movement_vector

action = validate_action(1)        # -> Action enum
dx, dy = get_movement_vector(action)
```

## Error Model
- All validation routes through the shared `ValidationError` in
  `plume_nav_sim.utils.exceptions`.

## Tests and Contracts
- Contract tests exercise invariants for these types in
  `src/backend/tests/contracts/test_semantic_invariants.py:1` and related suites.
- Formal contracts are documented in `src/backend/CONTRACTS.md:751` (see Core Types).

## Migration Notes
- Core types are consolidated under `core.types` with a minimal surface area.
- Environment configuration is now in `plume_nav_sim.envs.config_types`.
