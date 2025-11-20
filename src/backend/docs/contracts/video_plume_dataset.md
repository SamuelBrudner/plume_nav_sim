# Video Plume Dataset Contract (Source of Truth)

This document is the authoritative contract for video-derived plume datasets written to Zarr and consumed via xarray-like loaders. It consolidates the current behavior and requirements enforced by:

- `src/backend/plume_nav_sim/video/schema.py` (canonical constants and attrs model)
- `src/backend/plume_nav_sim/media/schema.py` (xarray-like dataset validation)
- `src/backend/plume_nav_sim/cli/video_ingest.py` (writer behavior)

The goal is to make the variable name, dimension order, attrs, dtype, and manifest requirements unambiguous, and to state loader/writer expectations. This serves as the source of truth ahead of unification work in bead `plume_nav_sim-206`.

## Variable and Dimensions

- Variable name: `concentration`
  - Defined in `src/backend/plume_nav_sim/video/schema.py:VARIABLE_NAME` and mirrored in `src/backend/plume_nav_sim/media/schema.py:VARIABLE_CONCENTRATION`.
- Dimension order: `("t", "y", "x")`
  - Defined in `src/backend/plume_nav_sim/video/schema.py:DIMS_TYX` and mirrored in `src/backend/plume_nav_sim/media/schema.py:DIMS_TYX`.
  - This is the dims source of truth for array layout. A redundant dataset/global `dims` attr may be present but is informational only and MUST match `("t","y","x")` if present.

## Required Dataset/Global Attributes

All required attrs live on the dataset/group (Zarr root) `attrs` mapping, not on the array:

- `schema_version` (str): must equal `1.0.0`.
- `fps` (float): frames per second of the original video.
- `source_dtype` (str): dtype of the original frames before normalization.
- `pixel_to_grid` (tuple[float, float]): scale to convert pixels to grid units `(y, x)`.
- `origin` (tuple[float, float]): origin of the grid `(y, x)`.
- `extent` (tuple[float, float]): extent/size of the grid `(y, x)`.

See `src/backend/plume_nav_sim/video/schema.py:VideoPlumeAttrs` for the canonical Pydantic model and validation.

## Array Requirements

- Data type: `float32`
- Layout: dims declared via `_ARRAY_DIMENSIONS` on the array must exactly be `("t","y","x")`.

## Manifest

- `manifest.json` must live at the dataset root.
- It validates against `src/backend/plume_nav_sim/media/manifest.py:ProvenanceManifest`.
- Writers should include CLI transparency for reproducibility: e.g., `{"cli_args": ["video_ingest", "--input", ...]}` in the manifest.

## Movie Metadata Sidecar (Canonical Movie Metadata)

When a movie‑backed plume dataset is created from a raw movie file (rather than from an already‑materialized Zarr dataset), `plume_nav_sim` expects a **per‑movie YAML metadata sidecar** and treats it as the canonical source of truth for movie metadata.

- Model: `plume_nav_sim.media.sidecar.MovieMetadataSidecar`.
- Loader: `plume_nav_sim.media.sidecar.load_movie_sidecar`.
- Ingest integration: `plume_nav_sim.plume.movie_field.resolve_movie_dataset_path`.

### Sidecar location

For a movie at `path/to/movie.ext`, the sidecar is expected at:

- `path/to/movie.ext.plume-movie.yaml`

This convention is implemented by `get_default_sidecar_path` and is used by `resolve_movie_dataset_path` whenever the environment is pointed at a non‑Zarr movie source.

### v1 sidecar schema and invariants

The v1 sidecar is a flat YAML mapping with the following fields (see `MovieMetadataSidecar` for the executable spec):

- `version: int` – sidecar schema version, currently `1`.
- `path: Optional[str]` – optional original movie path; informational only.
- `fps: PositiveFloat` – frames per second (FPS); **always** interpreted as frames per second (no `time_unit` field).
- `spatial_unit: str` – spatial unit of the movie coordinate system:
  - `"pixel"` → pixel space; `pixels_per_unit` MUST be omitted.
  - any other unit (e.g., `"mm"`, `"cm"`) → physical space; `pixels_per_unit` MUST be provided.
- `pixels_per_unit: Optional[Tuple[float, float]]` – number of pixels per one spatial unit `(y, x)`:
  - required when `spatial_unit` is not `"pixel"`.
  - must be omitted when `spatial_unit == "pixel"`.
  - entries must be strictly positive.
- `h5_dataset: Optional[str]` – dataset path inside an HDF5 movie:
  - required for `.h5` / `.hdf5` sources.
  - must not be set for non‑HDF5 sources.

### Mapping sidecar → `VideoPlumeAttrs`

`resolve_movie_dataset_path` uses the sidecar to produce `VideoPlumeAttrs` that are written to the Zarr dataset and then treated as canonical by loaders:

- `attrs.fps` is set directly from `sidecar.fps`.
- `attrs.pixel_to_grid` encodes grid units per pixel `(y, x)`:
  - if `spatial_unit == "pixel"`: `pixel_to_grid = (1.0, 1.0)`.
  - otherwise: `pixel_to_grid = (1.0 / pixels_per_unit_y, 1.0 / pixels_per_unit_x)`.
- `attrs.origin` is fixed to `(0.0, 0.0)`.
- `attrs.extent` is derived from the movie shape and `pixel_to_grid` (see `_resolve_extent` in the writer):
  - `extent_y = height * pixel_to_grid_y`
  - `extent_x = width * pixel_to_grid_x`

After ingest, the Zarr dataset’s `VideoPlumeAttrs` are the authoritative movie metadata. The sidecar is authoritative at ingest time; runtime behavior (`MoviePlumeField`) depends only on the attrs on disk.

### Container metadata vs sidecar

Container‑specific metadata (for example, `Attributes/imagingParameters/frameRate` in certain HDF5 formats) may be consulted by ingest helpers as a validation or best‑effort default when no sidecar is provided directly. It is **not** a second source of ground truth:

- When a sidecar is present, values derived from container metadata MUST agree with the sidecar or be rejected.
- In the canonical movie plume path (`resolve_movie_dataset_path`), `fps` and spatial calibration always come from the sidecar, and container metadata is used only for checks.

For a semantic overview of how the sidecar relates to the environment and movie plume behavior, see `src/backend/SEMANTIC_MODEL.md` (section “Movie metadata sidecar (canonical movie metadata)”).

### Example sidecars

Uncalibrated pixel‑space AVI:

```yaml
# my_movie.avi.plume-movie.yaml
version: 1
path: my_movie.avi
fps: 30.0
spatial_unit: pixel
```

Calibrated HDF5 plume movie in millimeters:

```yaml
# my_plume_movie.h5.plume-movie.yaml
version: 1
path: my_plume_movie.h5
fps: 120.0
spatial_unit: mm
pixels_per_unit: [50.0, 50.0]  # 50 pixels per mm in (y, x)
h5_dataset: /exchange/data
```

Both examples satisfy the v1 invariants:

- `fps` is positive and interpreted as frames per second.
- In the pixel‑space case, `pixels_per_unit` is omitted.
- In the calibrated HDF5 case, `pixels_per_unit` is provided and strictly positive, and `h5_dataset` is required because the container is `.h5`.

## Writer and Validator References

- Writer (reference): `src/backend/plume_nav_sim/cli/video_ingest.py`
- Schema + attrs validator: `src/backend/plume_nav_sim/video/schema.py`
- xarray-like dataset validator: `src/backend/plume_nav_sim/media/schema.py:validate_xarray_like`

## Compatibility Notes

- The `video` schema module is dependency-light to support usage in tools without the full media/IO extras.
- The `media` package provides dataset-level utilities (manifests, mapping between simulation steps and video frames, and validation helpers) and intentionally reuses the canonical constants from `video.schema`.
