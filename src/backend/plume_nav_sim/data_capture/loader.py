from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .schemas import EpisodeRecord, Position, RunMeta, SCHEMA_VERSION, StepRecord


class ReplayLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayArtifacts:
    run_dir: Path
    run_meta: RunMeta
    steps: list[StepRecord]
    episodes: list[EpisodeRecord]


def _ensure_schema_version(obj: dict[str, Any], *, context: str) -> None:
    version = obj.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ReplayLoadError(
            f"Unsupported schema_version '{version}' in {context}; expected {SCHEMA_VERSION}"
        )


def _parse_position(value: object, *, context: str) -> Position:
    if isinstance(value, Position):
        return value
    if isinstance(value, dict):
        try:
            return Position(x=int(value["x"]), y=int(value["y"]))
        except Exception as e:
            raise ReplayLoadError(f"Invalid position in {context}: {e}") from e
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return Position(x=int(value[0]), y=int(value[1]))
        except Exception as e:
            raise ReplayLoadError(f"Invalid position in {context}: {e}") from e
    raise ReplayLoadError(f"Invalid position in {context}: {value!r}")


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


def _load_run_meta(run_dir: Path) -> RunMeta:
    meta_path = run_dir / "run.json"
    if not meta_path.exists():
        raise ReplayLoadError(f"Missing run.json in {run_dir}")

    try:
        meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ReplayLoadError(f"Unable to read run.json at {meta_path}: {e}") from e

    if not isinstance(meta_obj, dict):
        raise ReplayLoadError(f"run.json must be a JSON object at {meta_path}")

    _ensure_schema_version(meta_obj, context="run.json")
    try:
        return RunMeta(**meta_obj)
    except Exception as e:
        raise ReplayLoadError(f"run.json is invalid at {meta_path}: {e}") from e


def _load_steps_jsonl(run_dir: Path) -> list[StepRecord]:
    records: list[StepRecord] = []
    for path in _jsonl_paths(run_dir, "steps"):
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
                if not isinstance(obj, dict):
                    raise ReplayLoadError(
                        f"Invalid step record in {path.name} line {line_no}: expected object"
                    )
                ctx = f"{path.name} line {line_no}"
                _ensure_schema_version(obj, context=ctx)

                obj2 = dict(obj)
                obj2["agent_position"] = _parse_position(
                    obj2.get("agent_position"), context=ctx
                )
                try:
                    records.append(StepRecord(**obj2))
                except Exception as e:
                    raise ReplayLoadError(f"Invalid step record in {ctx}: {e}") from e

    if not records:
        raise ReplayLoadError(f"No step records found in {run_dir}")
    return records


def _load_episodes_jsonl(run_dir: Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for path in _jsonl_paths(run_dir, "episodes"):
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
                if not isinstance(obj, dict):
                    raise ReplayLoadError(
                        f"Invalid episode record in {path.name} line {line_no}: expected object"
                    )
                ctx = f"{path.name} line {line_no}"
                _ensure_schema_version(obj, context=ctx)

                obj2 = dict(obj)
                obj2["final_position"] = _parse_position(
                    obj2.get("final_position"), context=ctx
                )
                try:
                    records.append(EpisodeRecord(**obj2))
                except Exception as e:
                    raise ReplayLoadError(f"Invalid episode record in {ctx}: {e}") from e

    if not records:
        raise ReplayLoadError(f"No episode records found in {run_dir}")
    return records


def load_replay_artifacts(run_dir: Path | str) -> ReplayArtifacts:
    run_path = Path(run_dir)
    if not run_path.exists():
        raise ReplayLoadError(f"Run directory not found: {run_path}")

    run_meta = _load_run_meta(run_path)
    steps = _load_steps_jsonl(run_path)
    episodes = _load_episodes_jsonl(run_path)

    if run_meta.run_id != steps[0].run_id:
        raise ReplayLoadError(
            f"Run ID mismatch: run.json has '{run_meta.run_id}' but steps have '{steps[0].run_id}'"
        )

    for ep in episodes:
        if ep.run_id != run_meta.run_id:
            raise ReplayLoadError(
                f"Run ID mismatch: run.json has '{run_meta.run_id}' but episode has '{ep.run_id}'"
            )

    return ReplayArtifacts(run_dir=run_path, run_meta=run_meta, steps=steps, episodes=episodes)


def iter_episode_steps(steps: Sequence[StepRecord]) -> list[list[StepRecord]]:
    if not steps:
        return []
    out: list[list[StepRecord]] = []
    current: list[StepRecord] = []
    current_id = steps[0].episode_id
    for rec in steps:
        if current and rec.episode_id != current_id:
            out.append(current)
            current = []
            current_id = rec.episode_id
        current.append(rec)
    if current:
        out.append(current)
    return out
