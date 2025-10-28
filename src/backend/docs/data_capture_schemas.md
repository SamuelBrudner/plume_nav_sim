# Data Capture Schemas (v1.0.0)

This document defines the analysis‑ready data formats written by the plume_nav_sim data capture pipeline.

Status: stable. Schema version: 1.0.0

The capture pipeline writes three primary artifacts in `results/<experiment>/<run_id>/`:

- `run.json` — run metadata and provenance
- `steps.jsonl.gz` — one JSON object per step (gzipped JSON Lines)
- `episodes.jsonl.gz` — one JSON object per episode summary (gzipped JSON Lines)

Optionally, Parquet files can be exported either via the CLI (`--parquet`) or programmatically.

## 1) run.json

Top‑level fields:

- `schema_version` (string, required): must equal `"1.0.0"`
- `run_id` (string, required)
- `experiment` (string, optional)
- `package_version` (string, optional)
- `git_sha` (string, optional)
- `start_time` (string, ISO‑8601 UTC)
- `env_config` (object, required): serialized EnvironmentConfig
  - `grid_size`: [width, height]
  - `source_location`: [x, y]
  - `max_steps`: int
  - `goal_radius`: float
  - `enable_rendering`: bool
  - `plume_params`: { `source_location`: [x, y], `sigma`: float }
- `base_seed` (int, optional)
- `episode_seeds` (array[int], optional)
- `system` (object, required; empty values allowed):
  - `hostname`: string|null
  - `platform`: string|null (e.g., `macOS-14.5-arm64-arm-64bit`)
  - `python_version`: string|null
  - `pid`: int|null
  - `user`: string|null

Example:

```
{
  "schema_version": "1.0.0",
  "run_id": "run-20250101-120000",
  "experiment": "demo",
  "package_version": "0.0.1",
  "git_sha": "abc1234",
  "start_time": "2025-01-01T12:00:00Z",
  "env_config": {
    "grid_size": [64, 64],
    "source_location": [32, 32],
    "max_steps": 500,
    "goal_radius": 1.0,
    "enable_rendering": true,
    "plume_params": {"source_location": [32, 32], "sigma": 12.0}
  },
  "base_seed": 123,
  "episode_seeds": [123, 124],
  "system": {
    "hostname": "m1-pro",
    "platform": "macOS-14.5-arm64-arm-64bit",
    "python_version": "3.10.14",
    "pid": 4242,
    "user": "sam"
  }
}
```

## 2) steps.jsonl.gz

Each line is a JSON object describing one environment step.

Fields:

- `schema_version` (string, required): `"1.0.0"`
- `ts` (float, seconds since epoch)
- `run_id` (string)
- `episode_id` (string)
- `step` (int, >= 1)
- `action` (int)
- `reward` (float)
- `terminated` (bool)
- `truncated` (bool)
- `agent_position` (object): `{ "x": int, "y": int }`
- `distance_to_goal` (float, >= 0)
- `observation_summary` (array[float], optional; short vector or scalar)
- `seed` (int, optional)

Example line:

```
{"schema_version":"1.0.0","ts":1735728000.0,"run_id":"run-20250101-120000","episode_id":"ep-000001","step":1,"action":0,"reward":0.0,"terminated":false,"truncated":false,"agent_position":{"x":16,"y":16},"distance_to_goal":22.6,"observation_summary":[0.12],"seed":123}
```

## 3) episodes.jsonl.gz

Each line summarizes a completed episode.

Fields:

- `schema_version` (string, required): `"1.0.0"`
- `run_id` (string)
- `episode_id` (string)
- `terminated` (bool)
- `truncated` (bool)
- `total_steps` (int, >= 0)
- `total_reward` (float)
- `final_position` (object): `{ "x": int, "y": int }`
- `final_distance_to_goal` (float, >= 0, nullable)
- `duration_ms` (float, >= 0, nullable)
- `avg_step_time_ms` (float, >= 0, nullable)

Example line:

```
{"schema_version":"1.0.0","run_id":"run-20250101-120000","episode_id":"ep-000001","terminated":true,"truncated":false,"total_steps":42,"total_reward":1.0,"final_position":{"x":48,"y":48},"final_distance_to_goal":0.0,"duration_ms":830.2,"avg_step_time_ms":19.8}
```

## Validation

- Runtime: every record is validated with Pydantic models before writing.
- Batch: `validate_run_artifacts(run_dir)` loads JSONL.gz into DataFrames and validates with Pandera schemas.

## Parquet Export

- End‑of‑run export to single Parquet files is available via `RunRecorder.finalize(export_parquet=True)` or CLI `--parquet`.
- For very large runs, a partitioned Parquet layout (per‑episode files) can be added in future iterations.

## Performance & Storage

- JSONL is gzipped; writing uses buffered IO and `orjson` (if available) for faster serialization than stdlib json.
- Size‑based rotation is supported (opt‑in) for long‑running captures.

## Evolution Policy

- Minor, backward‑compatible changes add optional fields only.
- Breaking changes (field removal/rename or type changes) bump `schema_version` and MUST be documented.
- Consumers should:
  - Check `schema_version` and error clearly on unsupported versions
  - Treat unknown fields as ignorable
  - Prefer additive evolution

## Quick Workflows

- Pandera validation:

```
from pathlib import Path
from plume_nav_sim.data_capture.validate import validate_run_artifacts
report = validate_run_artifacts(Path("results/demo/<run_id>"))
```

- JSONL → Parquet:

```
import pandas as pd
steps = pd.read_json("steps.jsonl.gz", lines=True, compression="gzip")
steps.to_parquet("steps.parquet", index=False)
```

- CLI capture + Parquet:

```
plume-nav-capture --output results --experiment demo --episodes 2 --grid 8x8 --parquet
```
