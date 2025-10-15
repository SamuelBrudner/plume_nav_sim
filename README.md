# plume-nav-sim

Gymnasium-compatible plume navigation environments engineered for reproducible robotics and reinforcement-learning research. Begin with a single `make_env()` call, customize through typed options, and inject bespoke components when experiments demand it.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 1. What You Get

- **Turnkey environment** – `plume_nav_sim.make_env()` returns a Gymnasium-compatible environment with metrics, logging, and sensible defaults.
- **Deterministic runs** – Centralized seeding utilities keep experiments reproducible across machines and CI.
- **Pluggable architecture** – Swap observation, reward, action, or plume components via dependency injection.
- **Research data workflow** – Built-in metadata export and curated examples accelerate analysis pipelines.

## 2. Quick Start

```python
import plume_nav_sim as pns

env = pns.make_env()
obs, info = env.reset(seed=42)
print(info["agent_xy"])  # starting position
```

- **Gymnasium integration**: `gym.make("PlumeNav-v0", action_type="oriented")`
- **Component knobs**: pass string options such as `observation_type="antennae"`
- **See also**: `src/backend/examples/quickstart.py`

## 3. Progressive Customization

| Stage | Goal | Example |
|-------|------|---------|
| **Customize** | Tune built-ins with typed kwargs | `examples/custom_configuration.py` |
| **Extend** | Inject custom components (reward, observation, plume) | `examples/custom_components.py` |
| **Reproduce** | Capture seeds and metadata for benchmarking | `examples/reproducibility.py` |

All examples live in `src/backend/examples/`. Run one from the backend root:

```bash
python -m examples.custom_components
```

## 4. Installation

```bash
git clone https://github.com/plume-nav-sim/plume_nav_sim.git
cd plume_nav_sim/src/backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

Validate the install:

```bash
python -c "import plume_nav_sim as pns; pns.make_env()"
```

## 5. Architecture Overview

- `plume_nav_sim.make_env()` → default environment (auto-registered as `PlumeNav-v0`)
- `ComponentBasedEnvironment` → DI-powered core that consumes protocol-compliant components (`plume_nav_sim.interfaces`)
- `plume_nav_sim.envs.factory.create_component_environment()` → factory with validation and curated defaults
- `docs/extending/` → deep dives on protocols, wiring, and testing

## 6. Reproducibility Workflow

```python
import plume_nav_sim as pns
from plume_nav_sim.utils.seeding import SeedManager

manager = SeedManager()
base_seed = 2025
episode_seed = manager.derive_seed(base_seed, "episode-0")

env = pns.make_env()
obs, info = env.reset(seed=episode_seed)
```

- `SeedManager` tracks provenance across episodes.
- `examples/reproducibility.py` writes JSON reports for audits.
- `docs/extending/README.md` covers DI best practices for packaging reproducible components.

## 7. Extending PlumeNav

- **Protocols**: `docs/extending/protocol_interfaces.md`
- **Component wiring**: `docs/extending/component_injection.md`
- **Custom guides**: `docs/extending/custom_rewards.md`, `custom_observations.md`, `custom_actions.md`
- **Testing**: contract suites under `src/backend/tests/contracts/`

## 8. Contributing

- Run `pre-commit run --all-files`
- Execute targeted tests: `conda run -n plume-nav-sim pytest src/backend/tests/plume_nav_sim/registration/ -q`
- Update docs/examples for user-facing changes

See `src/backend/CONTRIBUTING.md` for the full checklist.

## 9. License

MIT License. See [LICENSE](LICENSE).

## 10. Citation

```bibtex
@software{plume_nav_sim_2024,
  title        = {plume-nav-sim: Gymnasium-compatible reinforcement learning environment for plume navigation},
  author       = {Sam Brudner},
  year         = {2024},
  version      = {0.0.1},
  url          = {https://github.com/plume-nav-sim/plume_nav_sim}
}
```

---

Questions or ideas? Open an issue or start a discussion at https://github.com/plume-nav-sim/plume_nav_sim.
