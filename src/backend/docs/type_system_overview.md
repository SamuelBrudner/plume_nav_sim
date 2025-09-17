# Core Type System Harmonization

## Goal
Establish a single, authoritative set of core types for `plume_nav_sim` so that every
component—environments, plume models, utilities, and configuration helpers—shares the same
vocabulary. The objective is high impact: type mismatches currently break imports, make the
public API unusable, and leave tests unable to guard against regressions.

## Architecture Analysis
- `plume_nav_sim.core.__init__` advertises a rich type system (`Action`, `Coordinates`,
  `EnvironmentConfig`, `PlumeParameters`, factory helpers, validation utilities), but the
  referenced `core.types` module does not exist. Any consumer that relies on the documented
  API fails at import time.
- Multiple modules fill the void with ad-hoc aliases. For example,
  `config.environment_configs` renames `PlumeModel` to `PlumeParameters`, creating redundant
  semantics for the same entity. This redundancy cascades through the architecture because
  plume components depend on consistent parameter definitions.
- Tests such as `tests/plume_nav_sim/core/test_types.py` are empty, so there is no contractual
  protection ensuring that the public API remains synchronized with the actual
  implementations. As a result, data-model inconsistencies slip through silently.

## Conceptual Plan
- Introduce a concrete `plume_nav_sim.core.types` module that imports canonical dataclasses
  (`Coordinates`, `GridSize`, `AgentState`, `EpisodeState`, `PlumeModel`) and exposes them
  under stable names.
- Provide type aliases for frequently used unions (e.g., `ActionType`, `CoordinateType`) and
  tuples (e.g., `MovementVector`, `GridDimensions`) so the rest of the codebase can share a
  unified vocabulary.
- Implement factory helpers (`create_coordinates`, `create_grid_size`, `create_agent_state`)
  that centralize validation and conversion logic, ensuring every component constructs data
  in the same way.
- Define a validated `EnvironmentConfig` dataclass within `core.types` and update
  configuration utilities to rely on it instead of duplicating configuration models.
- Re-export `PlumeModel` as `PlumeParameters` within the new module, eliminating ad-hoc
  aliases and guaranteeing that plume-specific components refer to a single canonical type.
- Augment the test suite so that it codifies the new contracts, preventing future divergence
  between documentation and implementation.
