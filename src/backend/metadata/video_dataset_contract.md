# Video Plume Dataset Contract (Source of Truth)

Version: 1.0.0

This document is the authoritative contract for video-derived plume datasets written to Zarr and consumed via xarray-like loaders. It consolidates the current behavior and requirements enforced by:
- `src/backend/plume_nav_sim/video/schema.py` (canonical constants and attrs model)
- `src/backend/plume_nav_sim/media/schema.py` (xarray-like dataset validation)
- `src/backend/plume_nav_sim/cli/video_ingest.py` (writer behavior)

The goal is to make the variable name, dimension order, attrs, dtype, and manifest requirements unambiguous, and to state loader/writer expectations. This serves as the source of truth ahead of unification work in bead `plume_nav_sim-206`.

## Canonical Names and Dims

- Variable name: `concentration`
  - Defined in `src/backend/plume_nav_sim/video/schema.py:VARIABLE_NAME` and mirrored in `src/backend/plume_nav_sim/media/schema.py:VARIABLE_CONCENTRATION`.
- Dimension order: `("t", "y", "x")`
  - Defined in `src/backend/plume_nav_sim/video/schema.py:DIMS_TYX` and mirrored in `src/backend/plume_nav_sim/media/schema.py:DIMS_TYX`.
- Dims surface: Array attribute `_ARRAY_DIMENSIONS` on the `concentration` array MUST equal `["t", "y", "x"]`.
  - This is the dims source of truth for array layout. A redundant dataset/global `dims` attr may be present but is informational only and MUST match `("t","y","x")` if present.

## Required Dataset/Global Attributes

All required attrs live on the dataset/group (Zarr root) `attrs` mapping, not on the array:
- `schema_version` (str): must equal `1.0.0`.
- `fps` (float): frames per second, strictly `> 0`.
- `source_dtype` (str): original media frame dtype before conversion; one of `{"uint8","uint16","float32"}`.
- `pixel_to_grid` (tuple[float, float]): `(scale_y, scale_x)`; positive. A scalar is allowed at write-time, interpreted as `(v, v)`.
- `origin` (tuple[float, float]): `(y0, x0)` in grid units.
- `extent` (tuple[float, float]): `(height, width)` in grid units; strictly positive.

Optional attrs (may be present):
- `dims` (tuple[str, str, str]): If present, MUST equal `("t","y","x")`. Writers may include this (see `video/schema.py`).
- `timebase` (object): Optional rational timebase accepted by some loaders (`media/schema.py:Timebase`). Not required by this contract.
- `ingest_args` (object): Writer transparency record (`cli/video_ingest.py`); not required.

## Array Dtype and Normalization

- The `concentration` array stored on disk MUST have dtype `float32`.
- Normalization: When ingesting integer sources, the writer may normalize to `[0,1]` by dividing by the integer max when `--normalize` is used. Otherwise, values are converted to `float32` without rescaling. Consumers MUST NOT assume values are in `[0,1]` unless they enforce their own normalization step.
- The dataset MUST record the original source media dtype in `attrs.source_dtype` (e.g., `"uint8"`).

## Provenance Manifest

- A provenance manifest MUST be present at dataset root: `manifest.json`.
- Schema: `src/backend/plume_nav_sim/media/manifest.py:ProvenanceManifest`.
- Required fields include at least: `created_at` (ISO datetime), `source_dtype` (str). Common fields: `git_sha`, `package_version`, `cli_args`, `config_hash`, and a minimal `env` section.

## Loader and Writer Expectations

- Writer (reference): `src/backend/plume_nav_sim/cli/video_ingest.py`
  - Writes `concentration` as `float32` with chunks `CHUNKS_TYX` and array attr `_ARRAY_DIMENSIONS=["t","y","x"]`.
  - Writes required dataset/global attrs as above; includes optional `ingest_args` for transparency.
  - Writes `manifest.json` and runs Zarr metadata consolidation.
- Loader (reference validators): `src/backend/plume_nav_sim/media/schema.py:validate_xarray_like`
  - MUST find a variable named `concentration` with dims exactly `("t","y","x")`.
  - MUST validate dataset/global attrs against `VideoPlumeAttrs` (media) or `VideoPlumeAttrs` (video) with the required keys listed above.
  - SHOULD rely on dataset/global attrs as the metadata source of truth and `_ARRAY_DIMENSIONS` for dimension order.

## Minimal Acceptable Dataset (Example)

Zarr store layout (illustrative):
- `<root>/.zgroup`
- `<root>/.zattrs` contains:
  - `schema_version: "1.0.0"`
  - `fps: 12.0`
  - `source_dtype: "uint8"`
  - `pixel_to_grid: [1.0, 1.0]`
  - `origin: [0.0, 0.0]`
  - `extent: [H, W]` (grid units)
  - Optionally `dims: ["t","y","x"]`
- `<root>/concentration/.zarray` (dtype `"float32"`)
- `<root>/concentration/.zattrs` contains:
  - `_ARRAY_DIMENSIONS: ["t","y","x"]`
- `<root>/manifest.json` present with at least `created_at` and `source_dtype` fields.

This dataset MUST pass `validate_xarray_like(dataset)` from `media/schema.py` when opened via xarray-like tooling.

## Acceptance Criteria (Testable)

A dataset is contract-compliant if and only if all of the following hold:
- Variable named `concentration` exists and its dims are exactly `("t","y","x")` as declared by `_ARRAY_DIMENSIONS`.
- Array dtype is `float32`.
- Dataset/global attrs include: `schema_version=="1.0.0"`, `fps>0`, `source_dtype in {"uint8","uint16","float32"}`, `pixel_to_grid` two positive floats, `origin` two floats, `extent` two positive floats.
- `manifest.json` exists at the dataset root and validates against `ProvenanceManifest`.
- `validate_xarray_like` succeeds for the dataset.

## Cross-Reference and Forward Plan

- Current definitions:
  - Variable/dims and canonical attrs model: `src/backend/plume_nav_sim/video/schema.py`
  - Xarray-like validation helpers: `src/backend/plume_nav_sim/media/schema.py`
  - Writer behavior: `src/backend/plume_nav_sim/cli/video_ingest.py`
- Upcoming unification: bead `plume_nav_sim-206` will unify duplicate constants (`VARIABLE_NAME` vs `VARIABLE_CONCENTRATION`) and consolidate the single source of truth for validation into the `video` schema module. This document remains the authoritative contract regardless of internal module layout.
