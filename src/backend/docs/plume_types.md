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

Select a plume source

- Component factory (`create_component_environment`):

  ```python
  from plume_nav_sim.envs.factory import create_component_environment

  # Static Gaussian (default)
  env = create_component_environment(plume="static", plume_sigma=20.0)

  # Movie plume (Zarr dataset)
  env = create_component_environment(
      plume="movie",
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
      movie_path="plug-and-play-demo/assets/gaussian_plume_demo.zarr",
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

Quick demo with the bundled movie plume

- A ready-to-use dataset is included: `plug-and-play-demo/assets/gaussian_plume_demo.zarr`.
- Try the external-style demo using the movie plume:

  ```bash
  python plug-and-play-demo/main.py --plume movie --save-gif movie_demo.gif
  # Options: --movie-fps 60, --movie-step-policy wrap|clamp, --movie-path <zarr>
  ```

Dependencies

- Movie plume requires media extras: `pip install xarray zarr numcodecs` (or install backend extras if available).

Further reading

- Schema + validator: `src/backend/plume_nav_sim/video/schema.py`
- xarray-like dataset validation: `src/backend/plume_nav_sim/media/schema.py`
- Ingest CLI: `src/backend/plume_nav_sim/cli/video_ingest.py`
