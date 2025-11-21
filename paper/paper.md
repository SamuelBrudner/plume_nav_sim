---
title: "plume-nav-sim: Gymnasium-compatible plume navigation environments for reproducible reinforcement-learning research"

authors:
  - name: Samuel N. Brudner
    orcid: 0000-0002-6043-9328
    affiliation: 1

affiliations:
  - name: Molecular, Cellular, and Developmental Biology, Yale University, USA
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

`plume-nav-sim` is a Gymnasium-compatible environment library for simulating odor plume navigation tasks. It provides simple plume fields (analytical Gaussian and a short plume movie) and chemosensory observation and reward models tailored to odor source-finding problems. The library is built around explicit abstractions for plume fields, sensor models, and experiment configuration, together with a schema-validated data capture pipeline for recording trajectories and metadata.

The package targets researchers building agents that must localize a plume source using smell. It emphasizes:

- **Plume Semantics:** A clean separation between spatial domains, plume fields, and sensor models.
- **Workflow Reproducibility:** Deterministic seeding, typed configuration, and validated data capture.
- **Data-Driven Environments:** A schema-based architecture for loading external plume datasets (Zarr/xarray).

## Statement of need

Chemical plume navigation is a canonical problem in computational ethology, behavioral neuroscience, and robotics [@vergassola2007infotaxis; @carde2008navigational; @reddy2016learning]. Existing Gymnasium-compatible navigation tasks—including classic gridworlds, Minigrid-style environments, and 3D point-goal benchmarks—are designed as general-purpose control problems: they describe abstract states, obstacles, and goals, but they do not model odor concentration fields or chemosensory observations. In these tasks, each grid cell or pixel is simply a symbolic tile or rendered surface, and auxiliary information (such as episode configuration) is typically stored in unstructured `info` dictionaries.

Gymnasium’s gridworld families, such as Minigrid and its derivatives, provide rich goal-directed tasks but treat observations as symbolic grids or rendered images with no special semantics for scalar fields such as odor concentration. Furthermore, they do not define schemas for plugging in external datasets or for capturing experiment artifacts.

By contrast, plume navigation experiments in ethology and robotics are organized around odor fields and local sensors: agents must move through a concentration landscape and make decisions based on noisy scalar measurements. Implementations of such experiments often re-create this logic from scratch, with bespoke code for concentration fields, observation models, seeding, and logging. As a result, plume datasets and trajectories are hard to reuse across projects, and reproducing prior simulation results requires reconstructing one-off codebases rather than reusing a shared environment.

`plume-nav-sim` targets this gap by providing a small but focused Gymnasium-compatible environment in which the central objects are odor concentration fields (analytic Gaussian or short plume movies), chemosensory observation models, and experiment specifications. Environments and policies are constructed from typed configuration objects, and each run produces schema-validated artifacts that record trajectories, configuration, seeds, and plume dataset identifiers. This design makes it straightforward to reproduce, share, and analyze plume navigation experiments without re-implementing the scaffold around each new research question.

## Software description

### Design goals: beyond a toy gridworld

The current release of `plume-nav-sim` uses deliberately simple plume models: a static Gaussian plane and a short example plume movie. From the agent’s perspective, these are scalar-valued fields discretized on a grid and rendered into observations. The goal of the library is not to provide high-fidelity fluid dynamics, but to standardize how such fields, chemosensory observations, experiment configurations, and logged artifacts are represented in a Gymnasium-compatible environment.

In contrast to general-purpose gridworld libraries such as Minigrid, which encode symbolic tile types and tasks (walls, goals, doors, keys) in small discrete arrays or RGB images, `plume-nav-sim` is organized around odor concentration fields and chemosensory sensing. Even when instantiated with simple plumes, the library provides:

1. **A first-class PlumeField abstraction** (analytic or movie-based) that decouples the odor landscape from the spatial grid.
2. **Observation models** built around local concentration (with optional short histories), rather than agent coordinates or symbolic maps.
3. **A schema-validated data capture pipeline** that records trajectories, environment configuration, seeds, and plume dataset identifiers.

This makes the environment a small but reusable substrate for plume-navigation experiments, rather than a generic gridworld benchmark.

### Functionality

`plume-nav-sim` exposes a Gymnasium environment designed to simulate olfactory search tasks. A key design goal is to standardize the interface between plume datasets and RL environments.

- **Plume Dynamics as Data Interface:** A pluggable backend supporting both analytical models (e.g., Gaussian with turbulence) and empirical fields. Although the example plume movie shipped with `plume-nav-sim` is intentionally short and simple, the underlying movie interface is designed around an explicit xarray/Zarr schema for plume datasets. This makes it possible to plug in richer experimental or simulated plumes without changing the environment code, and to record which dataset and version were used in each experiment artifact.
- **Sensor Models:** Configurable agent sensors that expose local concentration at the agent and optional short histories, without revealing the agent’s absolute position or the full plume field.
- **Navigation:** Discrete action spaces tailored for bio-inspired locomotion (e.g., run-and-tumble, surge-and-cast).
- **Reward Structures:** Flexible reward functions for source discovery (reward on entering a neighborhood around the source) with optional penalties per step or for leaving the domain.

The public API centers on:

- `plume_nav_sim.make_env(...)` – factory for default environments.
- Configuration-driven composition via `SimulationSpec` and `prepare(...)`.
- A movie plume field that reads Zarr/xarray datasets validated against a documented schema.

### Architecture

The implementation emphasizes clear separation of concerns:

- **Core environment and components** live under `plume_nav_sim.envs`, `plume_nav_sim.plume`, and `plume_nav_sim.policies`.
- **Configuration and composition** are handled by `plume_nav_sim.config`, which defines typed specs and helpers for building environments and policies.
- **Data capture and media** live in `plume_nav_sim.data_capture`, `plume_nav_sim.media`, and `plume_nav_sim.video`, providing schemas and helpers for validated artifacts. Standard Gymnasium environments typically rely on unstructured `info` dictionaries for auxiliary data. `plume-nav-sim` improves upon this by enforcing strict, validated schemas for data capture, ensuring that simulation artifacts are self-describing and analysis-ready.
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
- Property-based tests for key invariants (e.g., compatibility between pluggable policies and environments).
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
