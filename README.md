# plume-nav-sim

Gymnasium-compatible plume navigation environments engineered for reproducible simulations. Begin with a single `make_env()` call, customize through typed options, and inject bespoke components when experiments demand it.


[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Install

Install from PyPI once the release is published:

```bash
pip install plume-nav-sim
```

For unreleased changes, install from source:

```bash
pip install -e src/backend
```

Or install directly from GitHub (no clone required):

```bash
pip install "plume-nav-sim @ git+https://github.com/SamuelBrudner/plume_nav_sim.git#subdirectory=src/backend"
```

Use this pinned dependency in another project's `pyproject.toml`:

```toml
[project]
dependencies = [
  "plume-nav-sim @ git+https://github.com/SamuelBrudner/plume_nav_sim.git@v0.1.0#subdirectory=src/backend",
]
```

Or in `requirements.txt`:

```text
plume-nav-sim @ git+https://github.com/SamuelBrudner/plume_nav_sim.git@v0.1.0#subdirectory=src/backend
```

## 1. What You Get

- **Turnkey environment** – `plume_nav_sim.make_env()` returns a Gymnasium-compatible environment with sensible defaults. Optional operational logging is available via `plume_nav_sim.logging.setup_logging`; analysis data capture is a separate workflow (see "Operational logging vs. data capture" below).
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

- Visible artifact out-of-the-box:
  - Run the bundled example to generate a short GIF (falls back to PNG if media extras are not installed):

    ```bash
    # From repo root
    cd src/backend
    python -m examples.quickstart
    # -> writes quickstart.gif (or quickstart.png without media extras)
    ```

- **Gymnasium integration**:

  ```python
  from plume_nav_sim.registration import ensure_registered, ENV_ID
  ensure_registered()  # make ENV_ID available to gym.make()

  import gymnasium as gym
  env = gym.make(ENV_ID)
  ```

- **Selector knobs**: use `make_env(...)` directly for `action_type`, `observation_type`, `reward_type`, wind, or movie-plume options:

  ```python
  import plume_nav_sim as pns

  env = pns.make_env(
      action_type="run_tumble",
      observation_type="antennae",
      reward_type="step_penalty",
  )
  ```
- **Wind sensing (optional)**: request `observation_type="wind_vector"` and either set `enable_wind=True` or pass `wind_direction_deg`/`wind_speed`/`wind_vector`; omit to keep odor-only behavior unchanged.
- **See also**: `src/backend/examples/quickstart.py` (writes `quickstart.gif` by default)

### Data zoo (registry-backed movie plumes)

- Curated datasets:
  - `colorado_jet_v1` v1.0.0 → Zenodo record 4971113 (Dryad DOI 10.5061/dryad.g27mq71) PLIF acetone plume (`a0004_nearbed_10cm_s.zarr`), license `CC-BY-4.0`, cite Connor, McHugh, & Crimaldi 2018 (Experiments in Fluids).
  - `rigolli_dns_nose_v1` v1.0.0 → Zenodo 15469831 DNS turbulent plume (nose level), license `CC-BY-4.0`, cite Rigolli et al. 2022 (eLife, DOI 10.7554/eLife.76989).
  - `rigolli_dns_ground_v1` v1.0.0 → Zenodo 15469831 DNS turbulent plume (ground level), license `CC-BY-4.0`, cite Rigolli et al. 2022 (eLife, DOI 10.7554/eLife.76989).
  - `emonet_smoke_v1` v1.0.0 → Dryad smoke plume video (walking Drosophila), license `CC0-1.0`, cite Demir et al. 2020 (eLife, DOI 10.7554/eLife.57524).
- Cache root defaults to `~/.cache/plume_nav_sim/data_zoo/<cache_subdir>/<version>/<expected_root>`; override with `movie_cache_root` or CLI `--movie-cache-root`.
- Config usage (registry resolves path and verifies checksum):

  ```python
  from plume_nav_sim.config.composition import SimulationSpec, prepare

  sim = SimulationSpec(
      plume="movie",
      movie_dataset_id="colorado_jet_v1",
      movie_auto_download=True,
  )
  env, _ = prepare(sim)
  ```

- CLI usage (plug-and-play demo):

  ```bash
  python plug-and-play-demo/main.py \
    --plume movie \
    --movie-dataset-id colorado_jet_v1 \
    --movie-auto-download
  ```

- Notes on `emonet_smoke_v1` (Dryad) access + preprocessing:
  - Some Dryad API endpoints can return `401 Unauthorized` for automated downloads.
  - If auto-download fails, manually download the large frames `.mat` artifact and symlink it into the Data Zoo cache.
  - The Emonet ingest performs background subtraction + auto-trimming of initial baseline frames (configurable in `EmonetSmokeIngest`).
  - For local analysis, use `scripts/compute_emonet_mean_intensity.py` to write a background-subtracted mean-intensity CSV, then `scripts/plot_emonet_mean_intensity.py` to inspect threshold crossings and choose smoke-onset trimming values.

- Attribution and new-entry workflow (checksums, ingest specs) are documented in `src/backend/docs/plume_types.md`.

#### DataCite-compliant metadata

All Data Zoo entries include [DataCite 4.5](https://schema.datacite.org/)-compliant metadata for interoperability with Zenodo, Figshare, and other DOI registries:

```python
from plume_nav_sim.data_zoo import DATASET_REGISTRY

entry = DATASET_REGISTRY["emonet_smoke_v1"]

# Structured creators with ORCIDs
for creator in entry.metadata.creators:
    print(f"{creator.name} ({creator.orcid})")

# Export for Zenodo upload
datacite_json = entry.metadata.to_datacite()
```

For simulation-generated datasets, use `SimulationMetadata`:

```python
from plume_nav_sim.data_zoo import SimulationMetadata

meta = SimulationMetadata.from_config(
    title="My Plume Simulation (1000 episodes)",
    creator_name="Your Name",
    config=hydra_cfg,  # auto-hashed for reproducibility
    seed=42,
    software_version="0.1.0",
)
```

This captures software provenance, config hash, random seed, and parameters — ready for publication to Zenodo with proper citation metadata.

## 3. Progressive Customization

| Stage | Goal | Example |
| --- | --- | --- |
| **Customize** | Tune built-ins with typed kwargs | `examples/custom_configuration.py` |
| **Extend** | Inject custom components (reward, observation, plume) | `examples/custom_components.py` |
| **Reproduce** | Capture seeds and metadata for benchmarking | `examples/reproducibility.py` |

All examples live in `src/backend/examples/`. Run one from the backend root:

```bash
python -m examples.custom_components
```

### Spec-driven observation wrappers

- You can declare observation adapters (wrappers) directly in a `SimulationSpec` so that the full runtime behavior is defined in one place. See the backend guide for a concrete example using the core 1‑back concentration history wrapper:
  - `src/backend/README.md` (section: "Compose: Applying observation wrappers via SimulationSpec")

## 4. Installation

```bash
git clone https://github.com/SamuelBrudner/plume_nav_sim.git
cd plume_nav_sim/src/backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Validate the install:

```bash
python -c "import plume_nav_sim as pns; pns.make_env()"
```

### Local Lint (mirrors CI)

Use the repo’s canonical lint target:

```bash
make lint
```

To mirror the active CI checks directly:

```bash
pip install -e "src/backend[lint]"
python -m ruff check --config src/backend/pyproject.toml src/backend
python -m mypy --strict --config-file src/backend/mypy.ini --follow-imports=skip -p plume_nav_sim.registration
```

### Debugger (Qt MVP)

A minimal Qt debugger is available for stepping the environment and viewing RGB frames. It includes dockable tools for inspection and configuration (Inspector, Live Config, Replay Config) plus optional frame overlays (purely visual).

- Install the GUI extra:

```bash
pip install -e "src/backend[gui]"
```

- Launch the debugger from the repo root:

```bash
make demo-debugger
```

Controls:

- Start/Pause, Step, Step Back, Reset with seed; toggle Explore (policy explore=True) and adjust interval (ms)
- Keyboard: Space (toggle run), N (step), B (step back), R (reset)
- View menu: toggle Inspector, Live Config, Replay Config, and Frame overlays

Policies:

- Built-ins: Stochastic TD, Deterministic TD, Random Sampler
- Custom: enter `module:ClassOrCallable` (or `module.sub.Class`) and click Load
  - Contract: either implement `select_action(obs, explore=False)` (preferred) or be a simple callable `policy(obs) -> action`
  - Optional: `reset(seed=...)` will be called if provided

Inspector (information-only):

- Dockable window (View → Inspector) with tabs:
  - Action: shows the expected action for the current frame and, when an ODC provider supplies it, an action distribution preview.
    - Provider‑only: labels and distributions are displayed only when a DebuggerProvider (ODC) is detected.
    - No side effects: the inspector never calls `select_action` and never influences the simulation.
  - Observation: shows observation shape and min/mean/max summary.
- The Inspector is intentionally read-only; controls that change simulation behavior (start/pause/step, reset, policy selection) remain in the main toolbar.

Live configuration:

- Dockable window (View → Live Config) provides presets and an editable `DebuggerConfig` (seed, plume, action_type, max_steps, movie_dataset_id/movie_path). Click Apply to reinitialize the live environment.

Replay configuration (read-only):

- Dockable window (View → Replay Config) shows the resolved replay environment settings (including the resolved/inferred action_type) for the currently loaded run.

#### Replay captured runs

- Point the debugger at a run directory produced by `plume-nav-capture` (expects `run.json`, `steps*.jsonl.gz`, and `episodes*.jsonl.gz`; Parquet may be exported alongside for analysis but is not used by replay).
- Loader hard-validates schema_version `0.1` and consistent `run_id` across `run.json`, steps, and episodes; multipart shards (`*.partNNNN.jsonl.gz`) are accepted and merged in order.
- Replay reconstructs the environment from recorded `env_config`, inferring `max_steps` from truncation markers when missing; RGB frames require `enable_rendering=True` in the capture.
- Replay resolves `action_type` from run metadata when present; otherwise it falls back to inferring it from recorded steps and surfaces divergences as explicit errors (see Replay Config).
- Replay regression coverage lives in `tests/debugger/test_replay_navigation.py`, `tests/debugger/test_replay_validation_diff.py`, and `tests/debugger/test_movie_registry_ui.py`; these cover replay navigation, validation-diff surfacing, and replay metadata resolution in the debugger.
- Version mismatches are treated as hard failures to avoid mixing incompatible capture formats.

Provider Plugins (ODC):

- The debugger auto-detects application-specific actions/observations/pipeline via the Opinionated Debugger Contract (ODC).
- Preferred integration is an entry-point plugin:
  - `pyproject.toml`:

    ```toml
    [project.entry-points."plume_nav_sim.debugger_plugins"]
    my_app = "my_app.debugger:provider_factory"
    ```

  - `my_app/debugger.py`:

    ```python
    from plume_nav_debugger.odc.provider import DebuggerProvider
    from plume_nav_debugger.odc.models import ActionInfo, PipelineInfo

    class MyProvider(DebuggerProvider):
        def get_action_info(self, env):
            return ActionInfo(names=["RUN", "TUMBLE"])  # len == action_space.n
        def policy_distribution(self, policy, observation):
            return {"probs": [0.8, 0.2]}  # side-effect free
        def get_pipeline(self, env):
            return PipelineInfo(names=[type(env).__name__, "MyWrapper", "CoreEnv"])

    def provider_factory(env, policy):
        return MyProvider()
    ```

- Strict provider‑only mode is always enabled. If no provider is detected, a banner appears and the inspector shows limited information. Heuristic fallbacks are removed and there is no CLI or preference to disable strict mode.

ODC Provider Contract (Developer Guide):

- Purpose: let the debugger introspect your app (action names, observation kind/label, policy distribution preview, pipeline) with zero debugger changes and no side effects.
- Implement `plume_nav_debugger.odc.provider.DebuggerProvider` with optional, side‑effect‑free methods:
  - `get_action_info(env) -> ActionInfo` — returns `names: list[str]` aligned to action indices `0..n-1`.
  - `describe_observation(observation, *, context=None) -> ObservationInfo | None` — returns `kind ∈ {"vector","image","scalar","unknown"}` and optional `label` for UI.
  - `policy_distribution(policy, observation) -> dict | None` — exactly one of `{ "probs" | "q_values" | "logits" }: list[float]`; must not call `select_action`.
  - `get_pipeline(env) -> PipelineInfo | None` — ordered wrapper/component names from outermost to core.
- Data models: `ActionInfo`, `ObservationInfo`, `PipelineInfo` live in `plume_nav_debugger.odc.models`.
- Discovery order: entry‑point plugin (recommended) → reflection on `env`/`policy` (`get_debugger_provider()`, `__debugger_provider__`, `debugger_provider`).
- Strict mode: labels, distributions, and pipeline appear only when a provider is detected; no heuristics.
- Examples and tests:
  - Minimal provider: `src/examples/odc_provider_example.py`
  - Discovery and mux behavior: `tests/debugger/test_odc_discovery.py`, `tests/debugger/test_odc_provider_mux.py`, `tests/debugger/test_odc_example_provider.py`
- Full specification: see `src/plume_nav_debugger/odc/SPEC.md`.

Preferences & Config:

- Edit → Preferences provides toggles for: Show pipeline, Show observation preview, Show sparkline, Show frame overlays, Default interval (ms), Theme (light/dark)
- Settings persist via QSettings and are also written to JSON at `~/.config/plume_nav_sim/debugger.json` (legacy `~/.config/plume-nav-sim/debugger.json` is migrated).
- Strict provider-only mode is not configurable.

### Jupyter notebooks (interactive plots)

If you plan to run the example notebooks or want interactive Matplotlib widgets inside Jupyter, install the notebooks extra which includes ipympl:

```bash
pip install -e ".[notebooks]"
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

- Fallback safely when ipympl is missing

  ```python
  import matplotlib as mpl
  try:
      %matplotlib widget
  except Exception as e:
      print("ipympl unavailable, falling back to inline:", e)
      %matplotlib inline
  print("Backend:", mpl.get_backend())
  ```

### Plug-and-play demo and capture notebooks

- Plug-and-play demo (external-style usage)
  - Quick run from repo root: `python plug-and-play-demo/main.py`
  - Notebook: `plug-and-play-demo/plug_and_play_demo.ipynb`
  - Details and options: `plug-and-play-demo/README.md`
  - Movie plumes use a per-movie YAML sidecar `<movie>.<ext>.plume-movie.yaml` as
    the canonical source of movie metadata. In this regime, `fps` is always
    interpreted as frames per second (time unit = seconds) and spatial
    calibration (`pixel_to_grid` and the physical spatial unit of the plume
    field) is derived from the sidecar's `spatial_unit` and `pixels_per_unit`.

- Capture workflow notebook (stable)
  - Notebook: `notebooks/stable/capture_end_to_end.ipynb`
  - Render to HTML into backend docs: `make nb-render`
    - Output: `src/backend/docs/notebooks/capture_end_to_end.html`
  - Related docs: `src/backend/docs/data_capture_schemas.md`, `src/backend/docs/data_catalog_capture.md`

- Stable environment-construction notebook
  - Notebook: `notebooks/stable/di_simulation_spec_component_env.ipynb`
  - Demonstrates selector-based env construction and spec-first composition via `SimulationSpec` + `prepare()`

### Test and performance requirements

Running the full test matrix (contracts, property-based suites, and performance checks) requires optional packages that are not included in the base install.

```bash
# Add property-based testing and perf monitors
pip install -e ".[test,benchmark]"
```

This installs `hypothesis` (for property/contract suites) and `psutil` (for performance benchmarks). Without them, `pytest` will report import-time failures.

### Zarr storage policy (chunks + compression)

- Default chunks for time-indexed video/plume tensors follow `CHUNKS_TYX = (8, 64, 64)`.
- Compression uses Blosc with Zstandard at `clevel=5` when available; otherwise falls back to Blosc LZ4 with a warning while preserving the same interface.
- Helper API and constants live in `plume_nav_sim/storage/zarr_policies.py`:
  - `create_blosc_compressor()` → returns a configured numcodecs compressor
  - `create_zarr_array(path, name, shape, dtype, ...)` → creates a Zarr dataset and records policy attrs

These policies are referenced by dataset ingest and loader components to ensure consistent on-disk formats across tools and CI.

## 5. Architecture Overview

- `plume_nav_sim.make_env(...)` → canonical selector-based environment construction for static and movie plumes
- `plume_nav_sim.EnvironmentConfig` and `create_environment_config(...)` → canonical env-init configuration
- `plume_nav_sim.config.SimulationSpec`, `create_simulation_spec(...)`, and `prepare(...)` → spec-first composition
- `PlumeEnv(...)` → direct component injection when you need custom plume, sensor, action, or reward objects
- For `gym.make()`, call `ensure_registered()` and use `ENV_ID` (`PlumeNav-v0`)
- `docs/extending/` → deep dives on protocols, wiring, and testing

### Alpha Migration Notes

This backend cleanup intentionally removed compatibility APIs that duplicated the active `PlumeEnv` construction path.

- `plume_nav_sim.envs.factory.create_component_environment(...)` → use `plume_nav_sim.make_env(...)` for selector kwargs or `PlumeEnv(...)` for direct injection
- `plume_nav_sim.config.ComponentEnvironmentConfig` and `plume_nav_sim.config.EnvironmentConfig` → use `plume_nav_sim.EnvironmentConfig`
- `component_environment_config_to_spec(...)` → use `create_simulation_spec(...)`
- `create_component_environment_from_config(...)` and `create_environment_from_config(...)` → use `build_env(create_simulation_spec(...))` or `prepare(...)`
- `plume_type` and `video_*` kwargs → use `plume="static"|"movie"` and `movie_*` kwargs

## 6. Reproducibility Workflow

```python
import plume_nav_sim as pns
from plume_nav_sim._compat import SeedManager

manager = SeedManager()
base_seed = 2025
manager.seed(base_seed)
episode_seed = int(manager.rng.integers(0, 2**32 - 1))

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

- Pre-commit hooks
  - From repo root (uses backend config):
    - `pre-commit install -c src/backend/.pre-commit-config.yaml`
    - `pre-commit run -c src/backend/.pre-commit-config.yaml --all-files`
  - Or from backend directory:
    - `(cd src/backend && pre-commit install && pre-commit run --all-files)`
- Execute targeted tests: `conda run -n plume-nav-sim pytest src/backend/tests/plume_nav_sim/registration/ -q`
- Update docs/examples for user-facing changes

See `src/backend/CONTRIBUTING.md` for the full checklist.

### Issue Tracking (bd beads)

Internal task tracking uses `bd` (beads), not GitHub Issues:

- Check ready work: `bd ready --json`
- Claim/update: `bd update <id> --status in_progress --json`
- Create: `bd create "Title" -t bug|feature|task -p 0-4 --json`
- Link discovered work: `bd create "Found bug" -p 1 --deps discovered-from:<parent-id> --json`
- Close: `bd close <id> --reason "Completed" --json`

Notes:

- Beads auto-sync to `.beads/issues.jsonl` in the repo.
- Community users may open GitHub Issues/Discussions; maintainers triage into beads as needed.

## 9. License

MIT License. See [LICENSE](LICENSE).

## 10. Citation

```bibtex
@software{plume_nav_sim_2025,
  title        = {plume-nav-sim: Gymnasium-compatible reinforcement learning environment for plume navigation},
  author       = {plume_nav_sim Development Team},
  year         = {2025},
  version      = {0.0.1},
  url          = {https://github.com/SamuelBrudner/plume_nav_sim}
}
```

---

### Operational logging vs. data capture

- Use loguru for operational, human-readable logs (console/file). Keep it separate from analysis data capture.
- Enable loguru sinks and stdlib bridge:

```python
from plume_nav_sim.logging import setup_logging

# Minimal ops logging setup
setup_logging(level="INFO", console=True)
```

- Use the data capture pipeline for analysis-ready data (JSONL.gz schemas, optional Parquet export). See backend README for details.

### Data capture dependencies

- Install the optional data extras to enable fast JSONL, Pandera validation, and Parquet export:

```bash
pip install -e ".[data]"
```

This pulls in:

- orjson (faster JSON serialization for JSONL)
- pandas + pandera (batch validation of captured data)
- pyarrow (optional Parquet export)

Notes:

- JSONL.gz capture works without extras; extras are needed for validation/parquet.
- Operational logging (loguru) is a separate optional extra: `pip install -e ".[ops]"`.

### Validation and Parquet examples

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

### Schema reference

- See detailed field definitions and evolution policy in `src/backend/docs/data_capture_schemas.md`.

### Data catalog

- See consolidated consumer docs and loading examples in `src/backend/docs/data_catalog_capture.md`.

Questions or ideas? Open an issue or start a discussion at <https://github.com/SamuelBrudner/plume_nav_sim>.
