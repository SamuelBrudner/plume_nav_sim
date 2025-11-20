# Data Directories Overview

This guide clarifies the roles of the data‑related packages and where to make changes when working on capture, datasets/metadata, or video ingestion.

## Mapping

- `src/backend/plume_nav_sim/data_capture/`
  - Purpose: Runtime data capture pipeline (JSONL.gz artifacts, validation, optional Parquet export).
  - Contents: recorder, wrapper, schemas, validation utilities, CLI (`plume-nav-capture`).
  - Docs: `src/backend/docs/data_capture_schemas.md`, `src/backend/docs/data_capture_versioning.md`, `src/backend/docs/data_catalog_capture.md`, `src/backend/docs/ops_runbook_data_capture.md`.

- `src/backend/plume_nav_sim/media/`
  - Purpose: Dataset metadata and utilities for video‑derived datasets (provenance manifest, dataset validation, time mapping).
  - Contents: `manifest.py` (ProvenanceManifest), `schema.py` (xarray‑like dataset validation reusing canonical constants), `mapping.py` (simulation step ↔ video frame mapping).
  - Note: Name retained as `media` for backward‑compatibility; it represents dataset metadata/manifests. Future rename to `datasets/` may add an alias if needed.
  - Docs: `src/backend/docs/video_plume_manifest.md` (provenance manifest), `src/backend/docs/contracts/video_plume_dataset.md` (dataset and movie sidecar contract).

- `src/backend/plume_nav_sim/video/`
  - Purpose: Canonical schema for video plume datasets (constants and Pydantic model) and attribute validation with minimal dependencies.
  - Contents: `schema.py` (VARIABLE_NAME, DIMS_TYX, SCHEMA_VERSION, VideoPlumeAttrs, validate_attrs).
  - Used by: `media.schema` validators and CLI writer.
  - Contract: `src/backend/docs/contracts/video_plume_dataset.md` (source of truth).

## Where to change code

- Capture pipeline changes → `plume_nav_sim/data_capture/` (writer behavior, field schemas, validation)
- Dataset manifest or dataset‑level validation → `plume_nav_sim/media/`
- Video dataset schema (attrs/dims/variable) → `plume_nav_sim/video/schema.py`
- Video ingestion/writing CLI → `plume_nav_sim/cli/video_ingest.py`

## Contracts

- Video plume dataset + movie metadata sidecar contract: `src/backend/docs/contracts/video_plume_dataset.md`
- Capture schemas and versioning: `src/backend/docs/data_capture_schemas.md`, `src/backend/docs/data_capture_versioning.md`

For a higher-level overview of the repository layout and public API (including where `plume_nav_sim` exports its main entrypoints and where to add new components), see:

- `src/backend/README.md` – section "Public API and Repository Layout".
- `src/backend/CONTRIBUTING.md` – section "Repository Layout and Public API".
