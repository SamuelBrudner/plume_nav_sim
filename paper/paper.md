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

    Chemical plume navigation is a canonical problem in computational/theoretical ethology, behavioral neuroscience, and robotics [@vergassola2007infotaxis; @carde2008navigational; @reddy2016learning]. Simulation has become a primary methodology for dissecting the algorithmic basis of odor tracking, allowing researchers to isolate variables that are conflated in biological experiments and to train reinforcement learning agents in controlled environments [cite cite cite]. Conducting this research usually requires developing a custom simulation environment, built specifically as a tool for a single research question. However, many of these bespoke tools recreate shared conceptual components from one to the next. This redundant work across projects adds unnecessary overhead to the research process. The approach also makes it more difficult to compare work across projects. The requirement to build these shared components increases the latency between identifying a research question and testing it in simulation. In this approach, software development is directly and exclusively tied to specific research questions, limiting the bandwidth to focus on general tooling conveniences or best practices in data management and software design. These codebases often lack integrated graphical user interfaces for real-time debugging and visualization, modern data management practices, or modular architecture designs. While the Gymnasium API provides a standard interface for RL agents, it does not provide the domain-specific spatial logic or sensor models required for plume navigation research. Researchers are often left to implement these complex dynamics themselves, leading to fragmentation.

`plume-nav-sim` fills this gap by providing a well-scoped environment that can serve as:

- A research testbed for new RL algorithms in plume navigation settings.
- A reusable backend for higher-level applications (e.g., the included Qt debugger and plug-and-play demo).

## Software description

### Functionality

`plume-nav-sim` exposes a Gymnasium environment with:

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
