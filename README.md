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

Spec-driven observation wrappers

- You can declare observation adapters (wrappers) directly in a `SimulationSpec` so that the full runtime behavior is defined in one place. See the backend guide for a concrete example using the core 1‑back concentration history wrapper:
  - `src/backend/README.md` (section: "Compose: Applying observation wrappers via SimulationSpec")

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

### Debugger (Qt MVP)

A minimal Qt debugger is available for stepping the environment and viewing RGB frames.

- Install Qt toolkit into your conda env:

```bash
make install-qt ENV_NAME=plume-nav-sim
```

- Run the debugger from source (uses `PYTHONPATH=src`):

```bash
make debugger ENV_NAME=plume-nav-sim
```

Controls:
- Start/Pause, Step, Reset with seed; adjust interval (ms)
- Keyboard: Space (toggle run), N (step), R (reset)

Policies:
- Built-ins: Stochastic TD, Deterministic TD, Random Sampler
- Custom: enter `module:ClassOrCallable` (or `module.sub.Class`) and click Load
  - Contract: either implement `select_action(obs, explore=False)` (preferred) or be a simple callable `policy(obs) -> action`
  - Optional: `reset(seed=...)` will be called if provided


### Jupyter notebooks (interactive plots)

If you plan to run the example notebooks or want interactive Matplotlib widgets inside Jupyter, install the notebooks extra which includes ipympl:

```bash
pip install -e .[notebooks]
```

In your notebook, enable the widget backend before plotting:

```python
%matplotlib widget
```

If the widget backend is not recognized, install the runtime deps into the SAME kernel and restart it:

- pip:
  - `%pip install -U ipympl ipywidgets matplotlib ipykernel`
- conda:
  - `conda install -c conda-forge ipympl ipywidgets matplotlib ipykernel`

Classic Notebook only (not JupyterLab): enable widgets extension once:

```bash
jupyter nbextension enable --py widgetsnbextension
```

Troubleshooting “'widget' is not a recognised backend name”:
- Ensure `ipympl` and `ipywidgets` are installed in the kernel, then restart it.
- Clear any forced backend: `import os; os.environ.pop('MPLBACKEND', None)` in the first cell.
- Fallback safely when ipympl is missing:
  -
  ```python
  import matplotlib as mpl
  try:
      %matplotlib widget
  except Exception as e:
      print("ipympl unavailable, falling back to inline:", e)
      %matplotlib inline
  print("Backend:", mpl.get_backend())
  ```

### Test and performance requirements

Running the full test matrix (contracts, property-based suites, and performance checks) requires optional packages that are not included in the base install.

```bash
# Add property-based testing and perf monitors
pip install -e .[test,benchmark]
```

This installs `hypothesis` (for property/contract suites) and `psutil` (for performance benchmarks). Without them, `pytest` will report import-time failures.

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

Operational logging vs. data capture

- Use loguru for operational, human-readable logs (console/file). Keep it separate from analysis data capture.
- Enable loguru sinks and stdlib bridge:

```python
from plume_nav_sim.logging.loguru_bootstrap import setup_logging

setup_logging(level="INFO", console=True, file_path="run.log", rotation="10 MB", retention="7 days")
```

- Use the data capture pipeline for analysis-ready data (JSONL.gz schemas, optional Parquet export). See backend README for details.

Data capture dependencies

- Install the optional data extras to enable fast JSONL, Pandera validation, and Parquet export:

```bash
pip install -e .[data]
```

This pulls in:
- orjson (faster JSON serialization for JSONL)
- pandas + pandera (batch validation of captured data)
- pyarrow (optional Parquet export)

Notes:
- JSONL.gz capture works without extras; extras are needed for validation/parquet.
- Operational logging (loguru) is a separate optional extra: `pip install -e .[ops]`.

Validation and Parquet examples

- Validate captured artifacts with Pandera (end-of-run):

```python
from pathlib import Path
from plume_nav_sim.data_capture.validate import validate_run_artifacts

run_dir = Path("results/demo/<run_id>")
report = validate_run_artifacts(run_dir)
print(report)  # {"steps": {"ok": True, "rows": N}, "episodes": {"ok": True, "rows": M}}
```

- Load JSONL.gz into pandas and export to Parquet:

```python
import pandas as pd

# Read JSONL.gz (newline-delimited JSON)
steps_df = pd.read_json(run_dir / "steps.jsonl.gz", lines=True, compression="gzip")
episodes_df = pd.read_json(run_dir / "episodes.jsonl.gz", lines=True, compression="gzip")

# Export to Parquet (requires pyarrow)
steps_df.to_parquet(run_dir / "steps.parquet", index=False)
episodes_df.to_parquet(run_dir / "episodes.parquet", index=False)

# Read Parquet back
df = pd.read_parquet(run_dir / "steps.parquet")
```

- Or let the recorder/CLI write Parquet at end-of-run:

```bash
plume-nav-capture --output results --experiment demo --episodes 2 --grid 8x8 --parquet
```

Schema reference

- See detailed field definitions and evolution policy in `src/backend/docs/data_capture_schemas.md`.

Questions or ideas? Open an issue or start a discussion at <https://github.com/plume-nav-sim/plume_nav_sim>.
