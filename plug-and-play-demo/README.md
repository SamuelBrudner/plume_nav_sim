# Plug-and-play demo (external-style usage)

This folder mimics a small, separate project that imports `plume_nav_sim` as a library and runs a minimal demo using a run–tumble policy. You can treat it as an example of "how my own project might call the library".

If you just want to see something run and get a picture out, try this from the repo root:

1. Install the backend in editable mode:

   ```bash
   pip install -e src/backend
   ```

2. Run the demo and save a short movie:

   ```bash
   python plug-and-play-demo/main.py --save-gif demo.gif
   ```

3. Open `demo.gif` in your image viewer to watch the agent follow the plume.

You will see some logs printed in the terminal (including an `Episode summary:` line). On some setups you may also see a one-line Gymnasium registry warning; for this demo that is expected and safe to ignore.

The rest of this README shows ways to customize the policy, capture datasets, and use the debugger. You can come back to those parts later.

Quick start (how to run)

- Prerequisite: install `plume_nav_sim` in your environment.
  - One‑liner (from repo root): `pip install -e src/backend`
  - Once published: `pip install plume_nav_sim`

- From repo root:
  - `python plug-and-play-demo/main.py`
  - With custom policy: `python plug-and-play-demo/main.py --policy-spec my_pkg.mod:MyPolicy`
  - Save frames: `python plug-and-play-demo/main.py --save-gif out.gif`
  - Run the bundled movie plume: `python plug-and-play-demo/main.py --plume movie`
    - Override dataset path: `--movie-path plug-and-play-demo/assets/gaussian_plume_demo.zarr`
    - Optional playback controls: `--movie-fps 60`, `--movie-step-policy wrap|clamp`
  - Config-based DI via `SimulationSpec` (JSON/TOML/YAML):
    - Example: `python plug-and-play-demo/main.py --config plug-and-play-demo/configs/simulation_spec.json`
    - CLI flags act as overrides when specified (e.g., `--grid 64x64`, `--policy-spec ...`).

- From this folder:
  - `python main.py`

What it does

- Builds an environment configured for run–tumble actions
- Applies the core `ConcentrationNBackWrapper(n=2)` via `SimulationSpec` so the policy sees `[c_prev, c_now]`
- Instantiates the stateless run–tumble policy implemented in this demo via dotted-path
- Runs a single episode with `runner.run_episode`, collecting RGB frames via the per-step callback
- Prints a concise episode summary and the number of frames captured
- Includes a minimal ODC provider so the debugger can display action labels (RUN/TUMBLE) and a distribution preview when this policy is loaded

Config-based composition

- You can define the entire run via a `SimulationSpec` config file and pass it to `--config`.
- Supported formats: `.json` (built-in), `.toml` (Python 3.11+), `.yaml` (requires PyYAML).
- Example config: `plug-and-play-demo/configs/simulation_spec.json`.
- If wrappers are omitted in the config, the demo applies the default `ConcentrationNBackWrapper(n=2)`.

Policy interface (see inline comments in `plug_and_play_demo/stateless_policy.py`)

- Matches the `plume_nav_sim.interfaces.policy.Policy` protocol
- Required members:
  - `action_space` property (Gymnasium space)
  - `reset(seed=...)` (recommended for determinism)
  - `select_action(obs, explore=...) -> action`

Files

- `plug-and-play-demo/main.py` – standalone script importing only `plume_nav_sim`
- `plug-and-play-demo/plug_and_play_demo/stateless_policy.py` – stateless run–tumble policy (uses [c_prev, c_now])
- `plug-and-play-demo/plug_and_play_demo.ipynb` – notebook demo (spec-first + frames)

Bundled media assets

- We ship two artifacts for the Gaussian plume demo movie so you can either preview the raw AVI or load the ready-to-use Zarr dataset via `MoviePlumeField`:
  - Raw movie: `plug-and-play-demo/assets/gaussian_plume_demo.avi`
  - Zarr dataset: `plug-and-play-demo/assets/gaussian_plume_demo.zarr`
    - Default `--plume movie` runs point to this dataset automatically.
    - Schema matches the `MoviePlumeField` contract (`concentration[t,y,x]`, dtype `float32`).
- Regeneration workflow:
  1. Generate or refresh the AVI under `src/backend/tests/data/video/gaussian_plume_demo.avi` (bead 214 script).
  2. Copy it into this demo: `cp src/backend/tests/data/video/gaussian_plume_demo.avi plug-and-play-demo/assets/`.
  3. Convert to Zarr (from repo root):

     ```bash
     conda run -n plume-nav-sim python -m plume_nav_sim.cli.video_ingest \
       --input plug-and-play-demo/assets/gaussian_plume_demo.avi \
       --output plug-and-play-demo/assets/gaussian_plume_demo.zarr \
       --fps 60 --pixel-to-grid "1 1" --origin "0 0" --normalize
     ```

  4. Commit both assets so the demo stays plug-and-play.

Plume types (static vs movie)

- The demo supports two plume sources, selected via `--plume`:
  - `static` (default): procedurally generates a static Gaussian plume centered on the goal.
    - Use as the simplest, zero‑dependency option for quick runs.
    - Example: `python plug-and-play-demo/main.py --plume static`
    - Notes: time‑invariant field; defaults come from the environment (e.g., grid, goal, `plume_sigma`).
  - `movie`: loads a prebuilt movie‑backed plume dataset (Zarr) via `MoviePlumeField`.
    - Best when you want repeatable, time‑varying plume dynamics.
    - Example: `python plug-and-play-demo/main.py --plume movie`
    - Defaults to the bundled dataset: `plug-and-play-demo/assets/gaussian_plume_demo.zarr`.
    - Options:
      - `--movie-path <path>` – target a specific dataset
      - `--movie-fps <float>` – override frames‑per‑second metadata for playback
      - `--movie-step-policy wrap|clamp` – control how time advances at the end of the clip
    - Grid size is inferred from the dataset; if the dataset is missing, see "Bundled media assets" above for regeneration steps.

Notes

- This demo intentionally stays minimal to highlight the plug‑and‑play flow.
- Determinism, subset checks, and other deeper features are covered by tests and docs in `src/backend/`.

 Capture Workflow (optional)

 Why capture?

- Produce analysis‑ready, versioned datasets for notebooks, ML, and reports
- Reproducible runs with explicit seeds and environment config
- Validated schemas to catch drift; optional Parquet export for fast loading

Prerequisites

- Install data extras to enable validation and Parquet:

```
pip install -e src/backend[data]
```

Quick start (CLI)

- Capture one or more episodes to a capture root (`results/` by default):

```
plume-nav-capture --output results --experiment demo --episodes 2 --grid 8x8
```

- Export Parquet at end of run (requires `pyarrow`):

```
plume-nav-capture --output results --experiment demo --episodes 2 --grid 8x8 --parquet
```

- Validate captured artifacts (JSONL.gz) after the run:

```python
from pathlib import Path
from plume_nav_sim.data_capture.validate import validate_run_artifacts

run_dir = Path("results/demo/<run_id>")
report = validate_run_artifacts(run_dir)
assert report["steps"]["ok"] and report["episodes"]["ok"], report
```

Artifacts

- Each run writes under `results/<experiment>/<run_id>/` (capture root → experiment → run):
  - `run.json` – run metadata and provenance
  - `steps.jsonl.gz` – per‑step events (newline‑delimited JSON, gzip)
  - `episodes.jsonl.gz` – per‑episode summaries (newline‑delimited JSON, gzip)
  - Optional: `steps.parquet`, `episodes.parquet` – columnar export
  - Optional: `manifest.json` – provenance/validation manifest when present

Cross‑links

- Schemas and versioning details: `src/backend/docs/data_capture_schemas.md`
- Data catalog (loading examples, Parquet notes, DVC pointers): `src/backend/docs/data_catalog_capture.md`
- Ops runbook (Hydra config, publishing/DVC, manifests): `src/backend/docs/ops_runbook_data_capture.md`

Notes

- This plug‑and‑play demo focuses on running a minimal episode and saving an optional GIF. For producing reusable datasets, prefer the `plume-nav-capture` CLI above.

 Capture from this script (alternative)

- The demo script can capture runs directly without the separate CLI. From repo root:

```
python plug-and-play-demo/main.py \
  --grid 8x8 \
  --capture-root results \
  --experiment demo \
  --episodes 2 \
  --parquet \
  --validate
```

- Flags map to the same capture pipeline used by `plume-nav-capture`:
  - `--capture-root` → capture root (default `results`)
  - `--experiment` → subfolder under capture root
  - `--episodes` → number of episodes to record
  - `--parquet` → export Parquet at end of run (if available)
  - `--validate` → validate JSONL.gz artifacts and print a summary

- Output layout and files are identical to the CLI variant (see Artifacts above).

 Debugger + ODC Walkthrough (optional)

- Goal: launch the Qt debugger, load this demo policy, and see action labels, a distribution preview, and a simple pipeline via the ODC provider.

If you are brand new to `plume_nav_sim`, you can skip this section on your first read. It is meant for when you are ready to inspect policies interactively.

Setup

- Install backend and Qt in your env (from repo root):

```
make dev-core ENV_NAME=plume-nav-sim
make install-qt ENV_NAME=plume-nav-sim
```

- Make the demo package importable (either approach works):
  - Easiest: install the demo package (also registers the ODC plugin):

    ```bash
    pip install -e plug-and-play-demo
    ```

  - Or add this folder to PYTHONPATH when launching the debugger:

    ```bash
    PYTHONPATH=src:plug-and-play-demo make debugger ENV_NAME=plume-nav-sim
    ```

Run the debugger

- If you installed the demo package: `make debugger ENV_NAME=plume-nav-sim`
  - If you didn’t install the demo package, use the `PYTHONPATH=src:plug-and-play-demo` variant above.
- In the Policy selector, enter `plug_and_play_demo:DeltaBasedRunTumblePolicy` and click Load.
- Start/Pause, Step, or Reset with a seed to drive the episode.
- Open View → Inspector:
  - Action tab shows RUN/TUMBLE labels and a live probability preview derived from dC (side‑effect free; the inspector never calls `select_action`).
  - When available, the Pipeline view lists wrappers from outermost to core (e.g., `ConcentrationNBackWrapper(n=2)` → core env).

Notes

- The ODC provider is exposed in two ways for convenience:
  - Entry‑point plugin (active when you `pip install -e plug-and-play-demo`).
  - Reflection on the demo policy (`get_debugger_provider`) so it works even without installation when the module is importable.
