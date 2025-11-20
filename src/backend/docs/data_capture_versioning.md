# Data Capture Versioning (DVC)

This repository promotes per-run capture artifacts to a versioned dataset using DVC.

Artifacts produced by the capture pipeline live under `results/<experiment>/<run_id>/`:

- `run.json` — run metadata and provenance
- `steps.jsonl.gz` — JSON Lines of per-step records (gzipped)
- `episodes.jsonl.gz` — JSON Lines of per-episode summaries (gzipped)
- `manifest.json` — provenance manifest (auto-generated at finalize)
- `*.parquet` — optional parquet exports (auto-generated if `pyarrow` is installed)

## Provenance Manifest

`RunRecorder.finalize()` now writes `manifest.json` with:

- `git_sha`, `package_version`, `schema_version`
- `config_hash` and `env_config`
- validation report from `plume_nav_sim.data_capture.validate.validate_run_artifacts`
- file inventory and a basic seed summary

Manifests are generated best-effort and never block capture completion.

## DVC Stage

A reusable stage ingests a completed run into a versioned dataset hierarchy under `data/captures/<experiment>/<run_id>`.

- Stage definition: `dvc.yaml`
- Parameters: `params.yaml` → `ingest_capture.run_dir`, `ingest_capture.out_dir`
- Helper script: `src/backend/scripts/ingest_capture.py`

### Usage

1) Set params for the run you want to ingest:

```
sed -i '' "s#run_dir: .*#run_dir: results/demo/<run_id>#" params.yaml
```

2) Reproduce the stage (ingest + optional parquet if `pyarrow` is installed):

```
dvc repro -s capture_ingest
```

3) Push versioned dataset to remote storage (configure `dvc remote` first):

```
dvc push
```

The stage tracks `data/captures` as an output directory and will create or update subfolders for each ingested run: `data/captures/<experiment>/<run_id>/`.

### Retrieving Historical Runs

- List versions via your VCS history (commits/tags) and DVC cache.
- Checkout a specific commit/tag and run:

```
dvc pull
```

- Consume artifacts directly via the manifest and files in `data/captures/<experiment>/<run_id>/`.

### Notes

- Parquet files are generated automatically by the ingestion script if `pyarrow` is available; otherwise the JSONL.gz remains the canonical output.
- The manifest is copied from the run directory; if missing, the ingestion script will generate a minimal one during ingest.
- The stage is designed to be simple and reproducible—inputs are the run directory and the script; outputs are written under `data/captures`.

## Validation

Validation is performed both during manifest generation and can be run ad-hoc:

```python
from pathlib import Path
from plume_nav_sim.data_capture.validate import validate_run_artifacts

report = validate_run_artifacts(Path('results/demo/<run_id>'))
print(report)
```

A summary is embedded in `manifest.json` for quick inspection.
