Plug-and-play demo (external-style usage)

This folder mimics a separate project that imports `plume_nav_sim` as a library and runs a minimal demo using the run–tumble temporal-derivative policy with the runner.

How to run

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

- From this folder:
  - `python main.py`

What it does

- Builds an environment configured for run–tumble actions
- Applies the core `ConcentrationNBackWrapper(n=2)` via SimulationSpec so the policy sees `[c_prev, c_now]`
- Instantiates the stateless run–tumble policy implemented in this demo via dotted-path
- Runs a single episode with `runner.run_episode`, collecting RGB frames via the per-step callback
- Prints a concise episode summary and the number of frames captured

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

Notes

- This demo intentionally stays minimal to highlight the plug‑and‑play flow.
- Determinism, subset checks, and other deeper features are covered by tests and docs in `src/backend/`.

Capture Workflow

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
