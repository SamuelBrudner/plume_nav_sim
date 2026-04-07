# Active Contracts

This document summarizes the active public contracts in the backend package. It is a working reference, not an “immutable spec”. Code and tests are the source of truth when behavior changes.

## Public Entry Points

Primary package entry points:

- `plume_nav_sim.make_env`
- `plume_nav_sim.initialize_package`
- `plume_nav_sim.get_package_info`
- `plume_nav_sim.get_conf_dir`

Environment factories:

- `plume_nav_sim.envs.plume_env.create_plume_env`
- `plume_nav_sim.envs.factory.create_component_environment`

Registration helpers:

- `plume_nav_sim.registration.register_env`
- `plume_nav_sim.registration.unregister_env`
- `plume_nav_sim.registration.ensure_registered`
- `plume_nav_sim.registration.get_registration_status`

CLI entry points:

- `plume-nav-capture`
- `plume-nav-data-zoo`
- `plume-nav-video-ingest`

## Environment Contract

`PlumeEnv` is the stable Gymnasium-style environment surface. It follows the standard `reset`, `step`, `render`, and `close` lifecycle.

Current expectations:

- `reset()` returns `(observation, info)`.
- `step()` returns `(observation, reward, terminated, truncated, info)`.
- `step()` before `reset()` raises `StateError`.
- Invalid public inputs raise `ValidationError` rather than failing silently.
- Explicit out-of-bounds user coordinates are rejected rather than clamped or recentered.
- Reusing the same seed should produce deterministic behavior for equivalent configuration and policy inputs.

`create_component_environment()` remains supported for the component-based and compatibility surface. It accepts the currently documented `action_type`, `observation_type`, `reward_type`, and movie-plume configuration options. Deprecated compatibility routing still exists through `make_env(...)` for legacy kwargs, but the preferred stable surface is explicit factory usage. The default and compatibility-routed envs share the same common public info keys (`step_count`, `episode_count`, `total_reward`, `goal_reached`, `agent_xy`, `goal_location`, `source_location`).

## Error Contract

The actively used exception types are defined in `plume_nav_sim._compat` and surfaced throughout the package:

- `ValidationError`
- `ComponentError`
- `ConfigurationError`
- `StateError`

Documentation and tests should describe current constructor parameters and usage, but should not promise that every parameter list is frozen forever. The important contract is behavior:

- invalid external input raises a descriptive validation/configuration error
- invalid lifecycle transitions raise `StateError`
- internal component failures are wrapped or surfaced as component/runtime failures with context

## Data and Replay Contract

Replay and capture artifacts are defined by the schemas in `plume_nav_sim.data_capture.schemas` and loaded through `plume_nav_sim.data_capture.loader`.

Current expectations:

- replay directories contain `run.json`, `steps*.jsonl.gz`, and `episodes*.jsonl.gz`
- schema version mismatches fail loudly
- malformed positions or run-id mismatches raise `ReplayLoadError`

Video-backed plume datasets follow the more specific contract docs in `src/backend/docs/contracts/video_plume_dataset.md`.

## Finer-Grained Interface Docs

The documents in `src/backend/contracts/` remain the detailed reference for the active narrow interfaces:

- action processor
- observation model
- reward function
- policy
- environment state machine
- media time mapping

When these docs drift, update them or the code together. Do not add new “canonical” claims unless enforcement exists in tests or runtime validation.
