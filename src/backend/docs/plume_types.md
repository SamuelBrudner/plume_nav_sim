# Plume Types and Movie Plume Overview

This page summarizes the available plume sources in plume-nav-sim and how to use the video-backed Movie Plume field.

Supported plume sources

- Static Gaussian (default)
  - Deterministic, parameterized by `sigma` and `source_location`.
  - Backed by a generated 2D float32 field; fast and reproducible.
  - Selected via `plume="static"` (default) in factories/specs.

- Movie Plume (video-backed)
  - Reads a precomputed concentration movie stored in Zarr with variable `concentration (t, y, x)`.
  - Advances one frame per env step with configurable step policy (`wrap` or `clamp`).
  - Validates dataset metadata via `VideoPlumeAttrs` (fps, pixel_to_grid, origin, extent).
  - Requires optional dependencies: `xarray`, `zarr`, `numcodecs`.
  - Curated registry dataset: `colorado_jet_v1` (Zenodo record 4971113 / Dryad 10.5061/dryad.g27mq71 near-bed acetone plume, 150 frames @ 15 FPS, 406x216 px, normalized concentrations).

Select a plume source

- Component factory (`create_component_environment`):

  ```python
  from plume_nav_sim.envs.factory import create_component_environment

  # Static Gaussian (default)
  env = create_component_environment(plume="static", plume_sigma=20.0)

  # Movie plume (registry-backed or direct path)
  env = create_component_environment(
      plume="movie",
      # Option A: curated registry dataset id
      movie_dataset_id="colorado_jet_v1",
      movie_auto_download=False,  # set True to fetch if cache is missing
      # Option B: direct path to a Zarr dataset or raw movie + sidecar
      movie_path="plug-and-play-demo/assets/gaussian_plume_demo.zarr",
      movie_step_policy="wrap",  # or "clamp"
  )
  ```

- Spec-first composition (`SimulationSpec` → `prepare()`):

  ```python
  from plume_nav_sim.compose import SimulationSpec, PolicySpec, prepare

  # Static Gaussian
  sim = SimulationSpec(plume="static", plume_sigma=20.0)
  env, policy = prepare(sim)

  # Movie plume
  sim = SimulationSpec(
      plume="movie",
      movie_dataset_id="colorado_jet_v1",
      movie_auto_download=False,
      movie_path="plug-and-play-demo/assets/gaussian_plume_demo.zarr",  # optional override
      movie_step_policy="wrap",
  )
  env, policy = prepare(sim)
  ```

Movie Plume field (details)

- Loader: `plume_nav_sim.plume.MoviePlumeField` with config `MovieConfig`.
- Dataset schema: variable `concentration` with dims `(t, y, x)`, dtype `float32`.
- Metadata (dataset root attrs): `fps`, `pixel_to_grid (y,x)`, `origin (y,x)`, `extent (y,x)`, `schema_version`, `source_dtype`.
- Step policy:
  - `wrap`: step `n` maps to `n % T` (loops at end).
  - `clamp`: step `n` maps to `min(n, T-1)` (holds last frame).
- Grid size is derived from the dataset; the env’s `grid_size` is overridden to match.

Create a dataset from a video

- Use the bundled CLI to ingest an AVI/MP4 or a frames directory into a Zarr dataset:

  ```bash
  plume-nav-video-ingest \
    --input path/to/video.avi \
    --output path/to/video.zarr \
    --fps 60 \
    --pixel-to-grid "1 1" \
    --origin "0 0" \
    --normalize
  ```

- The resulting dataset must contain `concentration (t,y,x)` as float32 and a `manifest.json` with CLI args for provenance.
- Contract reference: `src/backend/docs/contracts/video_plume_dataset.md`.
- For workflows that start from a raw movie file at runtime (for example, `movie_path="my_movie.avi"` or `movie_path="my_movie.h5"` passed into `create_component_environment`), movie metadata such as `fps` and spatial calibration MUST be provided via the per‑movie YAML sidecar (`*.plume-movie.yaml`); see `src/backend/SEMANTIC_MODEL.md` for the "Movie metadata sidecar (canonical movie metadata)" section.
- Registry-backed datasets default to `~/.cache/plume_nav_sim/data_zoo`; set `movie_cache_root` to an HPC scratch/work directory if home is constrained. Pre-stage the expected `cache_subdir/version/expected_root` from the registry and run with `movie_auto_download=False` when offline.

Quick demo with the bundled movie plume

- A ready-to-use dataset is included: `plug-and-play-demo/assets/gaussian_plume_demo.zarr`.
- Try the external-style demo using the movie plume:

  ```bash
  python plug-and-play-demo/main.py --plume movie --save-gif movie_demo.gif
  # Options: --movie-fps 60, --movie-step-policy wrap|clamp, --movie-path <zarr>
  ```

Dependencies

- Movie plume requires media extras: `pip install xarray zarr numcodecs` (or install backend extras if available).

Curated data zoo (registry-backed datasets)

- Where the cache lives: registry datasets unpack to `~/.cache/plume_nav_sim/data_zoo/<cache_subdir>/<version>/<expected_root>` by default. Override with `movie_cache_root=/scratch/pns` (CLI `--movie-cache-root`).
- Ready-to-use datasets:
  - `colorado_jet_v1` v1.0.0 → Zenodo record 4971113 / Dryad DOI 10.5061/dryad.g27mq71 PLIF acetone plume (`a0004_nearbed_10cm_s.zarr`), license `CC-BY-4.0`, cite Connor, McHugh, & Crimaldi 2018 (Experiments in Fluids).
  - `rigolli_dns_nose_v1` v1.0.0 → Zenodo 15469831 DNS turbulent plume (nose level), license `CC-BY-4.0`, cite Rigolli et al. 2022 (eLife, DOI 10.7554/eLife.76989).
  - `rigolli_dns_ground_v1` v1.0.0 → Zenodo 15469831 DNS turbulent plume (ground level), license `CC-BY-4.0`, cite Rigolli et al. 2022 (eLife, DOI 10.7554/eLife.76989).
  - `emonet_smoke_v1` v1.0.0 → Dryad smoke plume video (walking Drosophila), license `CC0-1.0`, cite Demir et al. 2020 (eLife, DOI 10.7554/eLife.57524).
- Configuration usage:

  ```python
  # Spec-first (prepare handles registry lookup + download/cache)
  sim = SimulationSpec(
      plume="movie",
      movie_dataset_id="colorado_jet_v1",
      movie_auto_download=True,            # fetch if cache missing
      movie_cache_root="~/scratch/data",   # optional override
  )
  env, _ = prepare(sim)
  ```

- CLI usage (plug-and-play demo):

  ```bash
  python plug-and-play-demo/main.py \
    --plume movie \
    --movie-dataset-id colorado_jet_v1 \
    --movie-auto-download \
    --movie-cache-root ~/scratch/data
  ```

- Pre-fetch from the shell without starting an env:

  ```bash
  python - <<'PY'
  from pathlib import Path
  from plume_nav_sim.data_zoo.download import ensure_dataset_available

  path = ensure_dataset_available(
      "colorado_jet_v1",
      cache_root=Path("~/scratch/data").expanduser(),
      auto_download=True,
  )
  print(f"Dataset ready at: {path}")
  PY
  ```

- Adding a new registry entry (checksum workflow):
  - Compute the archive checksum: `python - <<'PY'\nfrom hashlib import sha256\nfrom pathlib import Path\np = Path('path/to/archive.zip')\nprint(sha256(p.read_bytes()).hexdigest())\nPY`
  - Add a `DatasetRegistryEntry` in `plume_nav_sim/data_zoo/registry.py` with `dataset_id`, `version`, `cache_subdir`, `expected_root`, `artifact` (url, checksum, archive_type, archive_member when needed), `metadata` (title, description, license, citation, doi/contact as available), and optional `ingest` for HDF5→Zarr.
  - Validate layout expectations and checksum handling with `pytest src/backend/tests/plume_nav_sim/data_zoo`.
  - For offline testing, stage the unpacked payload at `<cache_root>/<cache_subdir>/<version>/<expected_root>` and rerun `ensure_dataset_available(..., auto_download=False)` to confirm it is accepted without network access.

Further reading

- Schema + validator: `src/backend/plume_nav_sim/video/schema.py`
- xarray-like dataset validation: `src/backend/plume_nav_sim/media/schema.py`
- Ingest CLI: `src/backend/plume_nav_sim/cli/video_ingest.py`
- Movie metadata sidecar (canonical movie metadata → `VideoPlumeAttrs`): `src/backend/SEMANTIC_MODEL.md`, `src/backend/docs/contracts/video_plume_dataset.md`
