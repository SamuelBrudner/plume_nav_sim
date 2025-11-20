# Data Catalog: Plume Navigation Capture Dataset

This page documents the analysis‑ready dataset emitted by the plume_nav_sim data capture pipeline. It consolidates what gets written, how to load and validate it, and how versioning and governance work.

Audience: data consumers (analysis, ML, reporting) and operators publishing datasets.

## Overview

Each capture run writes a versioned bundle under `results/<experiment>/<run_id>/`:

- `run.json` – run metadata and provenance (schema_version, env config, seeds, system info)
- `steps.jsonl.gz` – per‑step events (newline‑delimited JSON, gzip)
- `episodes.jsonl.gz` – per‑episode summaries (newline‑delimited JSON, gzip)
- Optional: `steps.parquet`, `episodes.parquet` – columnar export (requires `pyarrow`)

All records include `schema_version` and are validated by Pandera. Current version: `1.0.0`.

Reference: field‑level schemas are documented in `src/backend/docs/data_capture_schemas.md`.

## Provenance and Manifest

The capture bundle is designed to include a machine‑readable provenance manifest (see plume_nav_sim-152). The manifest captures:

- Code: package version and git SHA
- Configuration: serialized environment config and a stable config hash
- Seeding: base seed and per‑episode seeds
- Validation: Pandera validation report and schema versions
- System: host/platform/user, Python version

When present, the manifest is stored alongside artifacts (e.g., `manifest.json`).

## Storage and DVC Workflow

Datasets are intended to be versioned via DVC (see plume_nav_sim-152). A typical consumer flow:

```bash
# Ensure DVC is initialized and a remote is configured
dvc remote list

# Pull latest cataloged datasets (examples)
dvc pull path/to/datasets/index.dvc                  # index/meta file
dvc pull results/<experiment>/<run_id>/steps.parquet  # or per-run artifacts
```

Producers publish new versions via `dvc add` + `dvc push` or a staged pipeline. See the operations runbook (plume_nav_sim-153) for the authoritative procedure.

## Loading Examples

Install the data extras to enable fast JSONL and validation:

```bash
pip install -e .[data]
```

### Pandas

```python
from pathlib import Path
import pandas as pd
from plume_nav_sim.data_capture.validate import validate_run_artifacts

run_dir = Path("results/demo/<run_id>")

# JSONL.gz
steps = pd.read_json(run_dir / "steps.jsonl.gz", lines=True, compression="gzip")
episodes = pd.read_json(run_dir / "episodes.jsonl.gz", lines=True, compression="gzip")

# Optional Parquet (requires pyarrow)
if (run_dir / "steps.parquet").exists():
    steps = pd.read_parquet(run_dir / "steps.parquet")
if (run_dir / "episodes.parquet").exists():
    episodes = pd.read_parquet(run_dir / "episodes.parquet")

# Validate against Pandera schemas
report = validate_run_artifacts(run_dir)
assert report["steps"]["ok"] and report["episodes"]["ok"], report
```

### Polars

```python
from pathlib import Path
import polars as pl

run_dir = Path("results/demo/<run_id>")

# JSONL.gz (ndjson)
steps = pl.read_ndjson(run_dir / "steps.jsonl.gz", infer_schema_length=1000, compression="gzip")
episodes = pl.read_ndjson(run_dir / "episodes.jsonl.gz", infer_schema_length=1000, compression="gzip")

# Parquet (preferred when available)
if (run_dir / "steps.parquet").exists():
    steps = pl.read_parquet(run_dir / "steps.parquet")
if (run_dir / "episodes.parquet").exists():
    episodes = pl.read_parquet(run_dir / "episodes.parquet")

# Example: select a few analysis columns
steps_small = steps.select(["ts", "episode_id", "step", "reward", "agent_position", "distance_to_goal"])  # JSONL shape
```

Note: JSONL rows nest positions (e.g., `agent_position: {x, y}`); parquet exports include flattened columns (`agent_x`, `agent_y`, etc.). See schema docs for details.

If your Polars version lacks gzip support for NDJSON, decompress first:

```python
import gzip, io, polars as pl
with gzip.open(run_dir / "steps.jsonl.gz", "rt", encoding="utf-8") as f:
    steps = pl.read_ndjson(io.StringIO(f.read()))
```

## Validation

End‑to‑end validation is provided by `validate_run_artifacts(run_dir)`, which:

- Reads `steps*.jsonl.gz` and `episodes*.jsonl.gz` (supports multipart `*.part*.jsonl.gz`)
- Flattens nested fields for validation
- Applies Pandera schemas with coercion and constraints
- Returns a report dict: `{"steps": {"ok": bool, ...}, "episodes": {"ok": bool, ...}}`

Run it after capture; CI uses the same check (see plume_nav_sim-154).

## Schema Versioning and Compatibility

- The `schema_version` field is present in all records. Current value is `1.0.0`.
- Backward‑compatible additions (new nullable fields) retain the minor version; breaking changes bump the major.
- Consumers should:
  - Check `schema_version` on load and gate logic accordingly
  - Prefer parquet exports for stable column naming and types
  - Use the Pandera validation to catch regressions early

## Governance

- Retention: keep raw JSONL.gz alongside parquet; parquet is derived and can be regenerated.
- Privacy: the standard dataset contains no PII. If custom envs add sensitive fields, strip them before publishing.
- Reproducibility: publish manifest with git SHA, config hash, seeds, and validation report (see Provenance and Manifest).

## Cross‑References

- Schemas: `src/backend/docs/data_capture_schemas.md`
- Operations runbook (Hydra/config, publishing): plume_nav_sim-153
- Exploration notebook (end‑to‑end example): notebooks/stable/capture_end_to_end.ipynb
  - Render to HTML: `make nb-render` → outputs `src/backend/docs/notebooks/capture_end_to_end.html`
- CLI capture: `plume-nav-capture --help`

## Indexing

This catalog should be linked from top‑level READMEs and any docs index so data consumers have a single entry point.
