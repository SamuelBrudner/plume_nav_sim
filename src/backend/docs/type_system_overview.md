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

## Configuration Module Harmonization

### Goal
Eliminate the remaining redundant configuration type definitions so the configuration
package surfaces the same canonical entities as the core type system. This is high impact:
consumers import `plume_nav_sim.config.EnvironmentConfig` as their primary entrypoint, so
any divergence from `core.types.EnvironmentConfig` reintroduces the ambiguity this effort
is trying to remove.

### Architecture Analysis
- `config.__init__` still defines fallback classes named `EnvironmentConfig`,
  `PlumeConfig`, `RenderConfig`, and `PerformanceConfig`. Those placeholders disagree with
  the canonical types in `core.types` and `config.default_config` when `default_config`
  is available.
- The module-level fallbacks silently swallow import errors and mutate `__all__`, so
  callers cannot rely on a stable contract about which entities they are importing.
- Tests do not currently assert that the configuration package re-exports the canonical
  types. As a result, regressions can slip in without a failing test.

### Conceptual Requirements
- Remove the bespoke fallback class definitions and instead import the canonical
  `EnvironmentConfig` from `plume_nav_sim.core.types` (the source of truth for the data
  model) and the concrete configuration helpers from `config.default_config`.
- Ensure the package fails loudly if the default configuration module is unavailable; the
  user instructions prioritise explicit failure over silent fallbacks.
- Add structured logging around the import boundary so configuration initialisation remains
  observable.

### Contractual Tests
- Add a specification-level test that asserts `plume_nav_sim.config.EnvironmentConfig` and
  `plume_nav_sim.core.types.EnvironmentConfig` refer to the same class object. This
  prevents the reintroduction of divergent aliases.
- Assert that the configuration package's public surface advertises the canonical type in
  its `__all__` export list.

### Functional Tests
- Add a unit test that exercises `plume_nav_sim.config.get_default_environment_config()`
  and confirms it returns an actual `EnvironmentConfig` instance using the canonical data
  model.
