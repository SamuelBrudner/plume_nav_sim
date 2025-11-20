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

## Writer and Validator References

- Writer (reference): `src/backend/plume_nav_sim/cli/video_ingest.py`
- Schema + attrs validator: `src/backend/plume_nav_sim/video/schema.py`
- xarray-like dataset validator: `src/backend/plume_nav_sim/media/schema.py:validate_xarray_like`

## Compatibility Notes

- The `video` schema module is dependency-light to support usage in tools without the full media/IO extras.
- The `media` package provides dataset-level utilities (manifests, mapping between simulation steps and video frames, and validation helpers) and intentionally reuses the canonical constants from `video.schema`.
