# plume-nav-sim

🧪 Proof-of-Life Gymnasium Environment for Plume Navigation Research

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Gymnasium](https://img.shields.io/badge/gymnasium-0.29%2B-green.svg)](https://gymnasium.farama.org/) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

`plume-nav-sim` is a Gymnasium-compatible plume navigation environment built for researcher workflows: start quickly with a default setup, then plug in your own action, observation, and reward logic through a component architecture designed for controlled experimentation and reproducible comparisons. For direct component extension patterns, start with [`EXTENDING.md`](./EXTENDING.md).

## Quick Start

Install:

```bash
# Source install (recommended until PyPI publishing is set up)
pip install -e .

# Once published:
# pip install plume-nav-sim
```

Run a minimal episode (5 lines):

```python
import plume_nav_sim as pns
env = pns.make_env()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Extend It

The fastest way to add your own navigation behavior is to build custom components and inject them into `ComponentBasedEnvironment`. Use `EXTENDING.md` as the researcher entry point: it shows minimal interfaces, implementation templates, and a full custom-components example.

Read: [`EXTENDING.md`](./EXTENDING.md)

## Architecture Overview

`plume-nav-sim` supports two main usage modes:

- `PlumeEnv`: standard, ready-to-run Gymnasium environment
- `ComponentBasedEnvironment`: dependency-injected environment for custom research logic

`ComponentBasedEnvironment` centers on three swappable components:

- `ActionProcessor`: maps policy actions to next agent state
- `ObservationModel`: maps environment state to observations
- `RewardFunction`: maps transitions to scalar reward

These interfaces make it straightforward to test alternate assumptions without rewriting the full environment stack.

## Installation Details

Requirements:

- Python 3.10+
- Gymnasium 0.29+

Extras:

```bash
pip install -e ".[notebooks]"  # Jupyter widgets
pip install -e ".[media]"      # video/movie plume support
pip install -e ".[data]"       # data capture and analysis

# Once published:
# pip install plume-nav-sim[media]
```

Editable install for local development:

```bash
git clone https://github.com/SamuelBrudner/plume_nav_sim.git
cd plume_nav_sim/src/backend
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

Verify install:

```bash
python -c "import plume_nav_sim; print(plume_nav_sim.PACKAGE_VERSION)"
```

## Contributing / License

- Contributing guide: [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- License: [MIT (`../../LICENSE`)](../../LICENSE)
