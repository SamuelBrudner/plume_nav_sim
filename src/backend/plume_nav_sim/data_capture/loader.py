from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from .schemas import SCHEMA_VERSION, EpisodeRecord, RunMeta, StepRecord


class ReplayLoadError(RuntimeError):
    """Raised when replay artifacts cannot be loaded or validated."""


@dataclass(frozen=True)
class ReplayArtifacts:
    """Container for replay-ready artifacts."""

    run_dir: Path
    run_meta: RunMeta
    steps: list[StepRecord]
    episodes: list[EpisodeRecord]
    source_format: Literal["jsonl", "parquet"]


def load_replay_artifacts(
    run_dir: str | Path, *, prefer_parquet: bool = True
) -> ReplayArtifacts:
    """Load replay artifacts (run.json + steps/episodes) with schema checks.

    Supports:
    - JSONL (.jsonl.gz with optional .partNNNN.jsonl.gz segments)
    - Parquet (flattened columns produced by RunRecorder.finalize)

    Preference is given to Parquet when present and readable; otherwise the
    loader falls back to JSONL.
    """
    run_path = Path(run_dir)
    if not run_path.exists() or not run_path.is_dir():
        raise ReplayLoadError(f"Run directory not found: {run_path}")

    run_meta = _load_run_meta(run_path)

    steps: list[StepRecord] | None = None
    episodes: list[EpisodeRecord] | None = None
    source_format: Literal["jsonl", "parquet"] = "jsonl"

    if prefer_parquet:
        parquet_loaded = _maybe_load_parquet(run_path)
        if parquet_loaded is not None:
            steps, episodes = parquet_loaded
            source_format = "parquet"

    if steps is None or episodes is None:
        steps = _load_steps_jsonl(run_path)
        episodes = _load_episodes_jsonl(run_path)
        source_format = "jsonl"

    _ensure_run_id_consistency(run_meta.run_id, steps, episodes, run_path)

    return ReplayArtifacts(
        run_dir=run_path,
        run_meta=run_meta,
        steps=steps,
        episodes=episodes,
        source_format=source_format,
    )


def _load_run_meta(run_dir: Path) -> RunMeta:
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        raise ReplayLoadError(f"Missing run.json in {run_dir}")
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta_obj = json.load(fh)
    except Exception as e:  # pragma: no cover - defensive
        raise ReplayLoadError(f"Unable to read run.json at {meta_path}: {e}") from e

    _ensure_schema_version(meta_obj, f"{meta_path.name}")
    try:
        return RunMeta.model_validate(meta_obj)
    except Exception as e:
        raise ReplayLoadError(f"run.json is invalid at {meta_path}: {e}") from e


def _maybe_load_parquet(
    run_dir: Path,
) -> tuple[list[StepRecord], list[EpisodeRecord]] | None:
    steps_path = run_dir / "steps.parquet"
    episodes_path = run_dir / "episodes.parquet"
    if not (steps_path.exists() and episodes_path.exists()):
        return None

    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    try:
        steps_df = pd.read_parquet(steps_path)
        episodes_df = pd.read_parquet(episodes_path)
    except ImportError:
        # Missing parquet engine; fall back to JSONL
        return None
    except Exception as e:
        raise ReplayLoadError(
            f"Failed to read Parquet artifacts from {run_dir}: {e}"
        ) from e

    steps_records = _steps_from_parquet(steps_df.to_dict(orient="records"), steps_path)
    episodes_records = _episodes_from_parquet(
        episodes_df.to_dict(orient="records"), episodes_path
    )
    return steps_records, episodes_records


def _steps_from_parquet(
    rows: Sequence[dict], source: Path
) -> list[StepRecord]:  # pragma: no cover - exercised via integration tests
    records: list[StepRecord] = []
    for idx, row in enumerate(rows, start=1):
        obj = dict(row)
        ctx = f"{source.name} row {idx}"
        _ensure_schema_version(obj, ctx)

        if "agent_position" not in obj:
            ax = obj.pop("agent_x", None)
            ay = obj.pop("agent_y", None)
            if ax is None or ay is None:
                raise ReplayLoadError(
                    f"Missing agent_x/agent_y columns in {ctx} for Parquet data"
                )
            obj["agent_position"] = {"x": int(ax), "y": int(ay)}

        try:
            records.append(StepRecord.model_validate(obj))
        except Exception as e:
            raise ReplayLoadError(f"Invalid step record in {ctx}: {e}") from e

    if not records:
        raise ReplayLoadError(f"No step records found in {source}")
    return records


def _episodes_from_parquet(
    rows: Sequence[dict], source: Path
) -> list[EpisodeRecord]:  # pragma: no cover - exercised via integration tests
    records: list[EpisodeRecord] = []
    for idx, row in enumerate(rows, start=1):
        obj = dict(row)
        ctx = f"{source.name} row {idx}"
        _ensure_schema_version(obj, ctx)

        if "final_position" not in obj:
            fx = obj.pop("final_x", None)
            fy = obj.pop("final_y", None)
            if fx is None or fy is None:
                raise ReplayLoadError(
                    f"Missing final_x/final_y columns in {ctx} for Parquet data"
                )
            obj["final_position"] = {"x": int(fx), "y": int(fy)}

        try:
            records.append(EpisodeRecord.model_validate(obj))
        except Exception as e:
            raise ReplayLoadError(f"Invalid episode record in {ctx}: {e}") from e

    if not records:
        raise ReplayLoadError(f"No episode records found in {source}")
    return records


def _load_steps_jsonl(run_dir: Path) -> list[StepRecord]:
    paths = _jsonl_paths(run_dir, "steps")
    records: list[StepRecord] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ReplayLoadError(
                        f"Invalid JSON in {path.name} line {line_no}: {e}"
                    ) from e
                ctx = f"{path.name} line {line_no}"
                _ensure_schema_version(obj, ctx)
                try:
                    records.append(StepRecord.model_validate(obj))
                except Exception as e:
                    raise ReplayLoadError(f"Invalid step record in {ctx}: {e}") from e
    if not records:
        raise ReplayLoadError(f"No step records found in {run_dir}")
    return records


def _load_episodes_jsonl(run_dir: Path) -> list[EpisodeRecord]:
    paths = _jsonl_paths(run_dir, "episodes")
    records: list[EpisodeRecord] = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ReplayLoadError(
                        f"Invalid JSON in {path.name} line {line_no}: {e}"
                    ) from e
                ctx = f"{path.name} line {line_no}"
                _ensure_schema_version(obj, ctx)
                try:
                    records.append(EpisodeRecord.model_validate(obj))
                except Exception as e:
                    raise ReplayLoadError(
                        f"Invalid episode record in {ctx}: {e}"
                    ) from e
    if not records:
        raise ReplayLoadError(f"No episode records found in {run_dir}")
    return records


def _jsonl_paths(run_dir: Path, stem: str) -> list[Path]:
    base = run_dir / f"{stem}.jsonl.gz"
    parts = sorted(run_dir.glob(f"{stem}.part*.jsonl.gz"))
    paths: list[Path] = []
    if base.exists():
        paths.append(base)
    paths.extend(parts)
    if not paths:
        raise ReplayLoadError(
            f"Missing {stem}.jsonl.gz artifacts in {run_dir}; expected "
            f"{stem}.jsonl.gz (plus optional {stem}.partNNNN.jsonl.gz)"
        )
    return paths


def _ensure_schema_version(obj: dict, context: str) -> None:
    version = obj.get("schema_version")
    if version is None:
        raise ReplayLoadError(f"Missing schema_version in {context}")
    if version != SCHEMA_VERSION:
        raise ReplayLoadError(
            f"Unsupported schema_version '{version}' in {context}; "
            f"expected {SCHEMA_VERSION}"
        )


def _ensure_run_id_consistency(
    run_id: str,
    steps: Sequence[StepRecord],
    episodes: Sequence[EpisodeRecord],
    run_dir: Path,
) -> None:
    if not steps:
        raise ReplayLoadError(f"No step records found in {run_dir}")
    if not episodes:
        raise ReplayLoadError(f"No episode records found in {run_dir}")

    for rec in steps:
        if rec.run_id != run_id:
            raise ReplayLoadError(
                f"Run ID mismatch: run.json has '{run_id}' but step "
                f"record has '{rec.run_id}'"
            )
    for rec in episodes:
        if rec.run_id != run_id:
            raise ReplayLoadError(
                f"Run ID mismatch: run.json has '{run_id}' but episode "
                f"record has '{rec.run_id}'"
            )


__all__ = [
    "ReplayArtifacts",
    "ReplayLoadError",
    "load_replay_artifacts",
]
