---
title: "plume-nav-sim: Gymnasium-compatible plume navigation environments for reproducible reinforcement-learning research"

authors:
  - name: Samuel N. Brudner
    orcid: 0000-0002-6043-9328
    affiliation: 1

affiliations:
  - name: Molecular, Cellular, and Developmental Biology, Yale University, USA
    index: 1

date: 2026-02-24

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

`plume-nav-sim` is a Gymnasium-compatible environment library designed to standardize the simulation of odor plume navigation. It prioritizes reproducibility and ease of use through a decoupled architecture that supports config-driven injection of custom plume sources, sensor configurations, action models, and policies, while providing a schema-driven data system out of the box. To facilitate quick prototyping, the library features a curated 'Data Zoo' for effortless access to plume datasets, a library of common bio-inspired sensor and action models, and a pluggable, interactive inspection suite that allows researchers to visualize the behavior and performance of their simulations easily. By building on rigorous testing and explicit abstractions, `plume-nav-sim` ensures that every simulation run is reproducible, self-describing, and easy to debug.

The package targets researchers building agents that must localize a plume source using smell. It emphasizes:

- **Plume Semantics:** A clean separation between spatial domains, plume fields, and sensor models.
- **Workflow Reproducibility:** Deterministic seeding, typed configuration, and validated data capture.
- **Interactive Inspection:** A dedicated Qt-based debugger that allows researchers to step through simulations frame-by-frame, including step-back for recent transitions. This tool is designed to be pluggable: users can visualize their own agent's internal state (e.g., belief maps, memory) alongside the ground-truth plume and sensor readings via a standardized, side-effect-free interface.
- **Data-Driven Environments:** A schema-based architecture for loading external plume datasets (Zarr/xarray).

## Statement of need

Chemical plume navigation is a canonical problem in computational ethology, behavioral neuroscience, and robotics [@vergassola2007infotaxis; @carde2008navigational; @reddy2016learning]. Existing Gymnasium-compatible navigation tasks—including classic gridworlds, Minigrid-style environments, and 3D point-goal benchmarks—are designed as general-purpose control problems. They cannot be used out of the box to simulate plume navigation.

Gymnasium’s gridworld families, such as Minigrid and its derivatives, provide rich goal-directed tasks but treat observations as symbolic grids or rendered images with no special semantics for scalar fields such as odor concentration. Furthermore, they do not define schemas for plugging in external datasets or for capturing experiment artifacts.

By contrast, plume navigation experiments in ethology and robotics are organized around odor fields and local sensors: agents must move through a concentration landscape and make decisions based on noisy scalar measurements. Implementations of such experiments often re-create this logic from scratch, with bespoke code for concentration fields, observation models, seeding, and logging. As a result, plume datasets and trajectories are hard to reuse across projects, and reproducing prior simulation results requires reconstructing one-off codebases rather than reusing a shared environment.

`plume-nav-sim` targets this gap by providing a small but focused Gymnasium-compatible environment in which the central objects are odor concentration fields (analytic Gaussian or short plume movies), chemosensory observation models, and experiment specifications. Environments and policies are constructed from typed configuration objects, and each run produces schema-validated artifacts that record trajectories, configuration, seeds, and plume dataset identifiers. This design makes it straightforward to reproduce, share, and analyze plume navigation experiments without re-implementing the scaffold around each new research question.

## Software description

### Functionality

`plume-nav-sim` exposes a Gymnasium environment designed to simulate olfactory search tasks. A key design goal is to standardize the interface between plume datasets and RL environments.

- **Plume Dynamics as Data Interface:** A pluggable backend supporting both analytical models (e.g., Gaussian) and empirical fields. Although the example plume movie shipped with `plume-nav-sim` is intentionally short and simple, the underlying movie interface is designed around an explicit xarray/Zarr schema for plume datasets. This makes it possible to plug in richer experimental or simulated plumes without changing the environment code, and to record which dataset and version were used in each experiment artifact. The library includes a curated "Data Zoo" providing one-line access to published plume datasets, including PLIF fluorescence measurements [@connor2018plif], DNS turbulent plume simulations [@rigolli2022alternation], and wind-tunnel smoke videos [@demir2020walking]. All dataset metadata follows the DataCite 4.5 schema, enabling direct export to Zenodo and other DOI registries with structured creator information, ORCIDs, and provenance links. For simulation-generated datasets, dedicated metadata classes capture software version, configuration hash, random seeds, and runtime parameters to ensure computational reproducibility.
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
- **Visualization and debugging** are provided by rendering utilities and an optional Qt-based interactive debugger (`plume_nav_debugger`). This application enables researchers to step through simulations forward or backward, switch policies dynamically, and inspect agent decisions and sensory inputs via a decoupled "Opinionated Debugger Contract" (ODC) interface.

This layout allows users to either:

- Treat `plume-nav-sim` as a black-box environment via `make_env`.
- Or selectively reuse components (e.g., plume models, seeding utilities, capture pipeline) in larger systems.

## Dependencies

The package is implemented in Python (3.10+) and builds on widely used scientific libraries:

- Gymnasium for the RL interface.
- NumPy for array operations.
- Matplotlib for visualization.
- Optional extras for data workflows (pandas, pandera, pyarrow), media handling (imageio, xarray, zarr, numcodecs), and GUI debugging (PySide6).

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
