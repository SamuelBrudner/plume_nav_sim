# Operations Runbook: Data Capture Pipeline

This runbook describes how to run plume-nav-sim data capture jobs using Hydra configs and how to manage outputs.

## Prerequisites

- Create and activate the project environment (with data extras):
  - `conda env create -f src/backend/environment.yml` (or `pip install -e src/backend[ops,data]`)
- Optional: enable Parquet export by installing the `data` extra (`pyarrow`, `pandas`).

## Configs

Hydra configs live under `conf/data_capture/`:
- `conf/data_capture/config.yaml` – base config
- `conf/data_capture/experiment/default.yaml` – demo defaults

Key parameters:
- Top-level: `output`, `experiment`, `episodes`, `seed`, `rotate_size`, `parquet`
- `env.*`: `grid_size`, `max_steps`, `action_type`, `observation_type`, `reward_type`

## Commands

- Legacy CLI flags (backward compatible):
  - `plume-nav-capture --output results --experiment demo --episodes 2 --seed 100 --grid 8x8`

- Hydra-driven run (recommended for ops):
  - `plume-nav-capture --config-name data_capture/config episodes=5 env.max_steps=250 env.grid_size=[64,64] experiment=batch-20231106`
  - Override any field using Hydra syntax (space separated key=value pairs).

Outputs land at `results/<experiment>/<run_id>/`:
- `run.json` – run metadata (includes `config_hash` when Hydra is used)
- `steps.jsonl.gz`, `episodes.jsonl.gz` – compressed JSONL event logs
- `*.parquet` – optional Parquet export when `--parquet` or `parquet: true`

## Reproducibility

- The CLI computes a stable SHA-256 `config_hash` from the resolved Hydra config.
- Record seeds via CLI (`--seed`) or config (`seed:`); episodes use `seed + i`.
- For batch runs, pin configs and overrides in job specs to reproduce.

## Validation

- Validate artifacts programmatically:
  - `from plume_nav_sim.data_capture.validate import validate_run_artifacts`
  - Run on a run directory to check schema conformance.

## Notes

- Config directory discovery defaults to the repository `conf/` tree. To use an alternate config root, pass `--config-path <path>`.
- For DVC-based publishing and manifests (config hash, git SHA, validation report), see bead plume_nav_sim-152.
