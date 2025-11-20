---
title: "plume-nav-sim: Gymnasium-compatible plume navigation environments for reproducible reinforcement-learning research"

authors:  # Placeholder metadata; update authors/ORCIDs/affiliations before submission
  - name: "Samuel Brudner"
    orcid: "0000-0000-0000-0000"
    affiliation: 1

affiliations:
  - name: "To be completed"
    index: 1

date: 2025-11-20

# JOSS metadata
bibliography: paper.bib

# Example tags – adjust as needed
tags:
  - Python
  - reinforcement learning
  - animal behavior
  - simulation
  - odor plume
  - gymnasium
---

# Summary

`plume-nav-sim` is a Gymnasium-compatible reinforcement learning environment for navigation in chemical plumes. It provides a configurable grid world with static and video-derived concentration fields, deterministic seeding utilities, and a pluggable architecture for observations, rewards, and policies.

The package targets researchers and educators building agents that must localize a plume source under uncertainty. It emphasizes:

- Deterministic seeding and reproducible workflows.
- A clear separation between core environment logic, data capture, and visualization.
- Extensibility via typed configuration and dependency injection.

## Statement of need

    Chemical plume navigation is a canonical problem in robotics and behavioral neuroscience. Many researchers use in-house simulation code to investigate plume navigation. This approach makes it more difficult to compare work across labs. It also increases the uptime between identifying a research question and testing it in simulation. {hightlight lack of guis for debugging and visualization} In this approach, software development is directly and exclusively tied to specific research questions, limiting the bandwidth to focus on general best practices in data management and software design. This last point can especially bite when the simulations are meant to support reinforcement learning methods.

`plume-nav-sim` fills this gap by providing a small, well-scoped environment that can serve as:

- A research testbed for new RL algorithms in plume navigation settings.
- A reusable backend for higher-level applications (e.g., the included Qt debugger and plug-and-play demo).

## Software description

### Functionality

At its core, `plume-nav-sim` exposes a Gymnasium environment with:

- A 2D grid and plume source defined by a static Gaussian plume model or a video-backed plume field.
- Discrete and oriented action spaces (e.g., run/tumble policies).
- Observation models ranging from scalar concentration at the agent to history-based wrappers.
- Reward models capturing source discovery, penalties, and step costs.

The public API centers on:

- `plume_nav_sim.make_env(...)` – factory for default environments.
- Configuration-driven composition via `SimulationSpec` and `prepare(...)`.
- A movie plume field that reads Zarr/xarray datasets validated against a documented schema.

### Architecture

The implementation emphasizes clear separation of concerns:

- **Core environment and components** live under `plume_nav_sim.envs`, `plume_nav_sim.plume`, and `plume_nav_sim.policies`.
- **Configuration and composition** are handled by `plume_nav_sim.config`, which defines typed specs and helpers for building environments and policies.
- **Data capture and media** live in `plume_nav_sim.data_capture`, `plume_nav_sim.media`, and `plume_nav_sim.video`, providing schemas and helpers for validated artifacts.
- **Visualization and debugging** are provided by rendering utilities and an optional Qt debugger (`plume_nav_debugger`).

This layout allows users to either:

- Treat `plume-nav-sim` as a black-box environment via `make_env`.
- Or selectively reuse components (e.g., plume models, seeding utilities, capture pipeline) in larger systems.

## Dependencies

The package is implemented in Python (3.10+) and builds on widely used scientific libraries:

- Gymnasium for the RL interface.
- NumPy for array operations.
- Matplotlib for visualization.
- Optional extras for data workflows (pandas, pandera, pyarrow) and media handling (imageio, xarray, zarr, numcodecs).

## Quality

`plume-nav-sim` includes an extensive automated test suite:

- Unit and integration tests for core environment behavior, seeding, and contracts.
- Property-based tests for key invariants (e.g., observation and action space compatibility).
- Performance and regression tests guarding rendering and data capture behavior.
- Continuous integration workflows that run tests and code quality checks on every push.

Documentation is provided via:

- A user-facing README describing installation, quickstart examples, and advanced usage.
- Backend documentation describing configuration, data capture schemas, and video plume datasets.
- Example scripts and notebooks demonstrating typical workflows (e.g., capture pipelines and plug-and-play demos).

## Acknowledgements

`plume-nav-sim` builds on the scientific Python ecosystem, in particular Gymnasium, NumPy, and Matplotlib. We thank the maintainers of these projects and the broader open-source community for providing the foundations used here.

## References

The key scientific and software dependencies are cited in `paper.bib` and referenced throughout the text.
