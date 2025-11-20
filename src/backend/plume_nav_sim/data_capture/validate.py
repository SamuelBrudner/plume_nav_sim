from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List


def _read_jsonl_gz_files(stem: Path) -> List[dict]:
    files: List[Path] = []
    base = (
        stem.with_suffix(stem.suffix + ".jsonl.gz")
        if stem.suffix
        else Path(str(stem) + ".jsonl.gz")
    )
    if base.exists():
        files.append(base)
    files.extend(sorted(stem.parent.glob(f"{stem.name}.part*.jsonl.gz")))
    rows: List[dict] = []
    for p in files:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def _flatten_steps(rows: List[dict]):
    if not rows:
        return None
    # Flatten nested agent_position -> agent_x, agent_y
    flat: List[dict] = []
    for r in rows:
        rp = dict(r)
        pos = rp.pop("agent_position", None)
        if isinstance(pos, dict):
            rp["agent_x"] = pos.get("x")
            rp["agent_y"] = pos.get("y")
        # Provide conservative defaults for missing fields (tolerant validation)
        rp.setdefault("distance_to_goal", 0.0)
        rp.setdefault("terminated", False)
        rp.setdefault("truncated", False)
        if "schema_version" not in rp:
            try:
                from .schemas import (  # local import to avoid heavy deps at import time
                    SCHEMA_VERSION,
                )

                rp["schema_version"] = SCHEMA_VERSION
            except Exception:
                rp["schema_version"] = "1.0.0"
        flat.append(rp)
    # Construct DataFrame lazily to avoid import cost at module import time
    import pandas as pd  # type: ignore

    return pd.DataFrame(flat)


def _flatten_episodes(rows: List[dict]):
    if not rows:
        return None
    flat: List[dict] = []
    for r in rows:
        rp = dict(r)
        pos = rp.pop("final_position", None)
        if isinstance(pos, dict):
            rp["final_x"] = pos.get("x")
            rp["final_y"] = pos.get("y")
        # Defaults for robustness
        if "schema_version" not in rp:
            try:
                from .schemas import SCHEMA_VERSION

                rp["schema_version"] = SCHEMA_VERSION
            except Exception:
                rp["schema_version"] = "1.0.0"
        flat.append(rp)
    import pandas as pd  # type: ignore

    return pd.DataFrame(flat)


def _schemas():
    # Prefer pandas-specific API to avoid deprecation warnings
    try:  # pandera >= recommended API
        import pandera.pandas as pa  # type: ignore
        from pandera.pandas import Column, DataFrameSchema  # type: ignore
    except Exception:  # fallback for older pandera
        import pandera as pa  # type: ignore
        from pandera import Column, DataFrameSchema  # type: ignore
    from pandera.dtypes import Bool, Float, Int, String  # type: ignore

    steps_schema = DataFrameSchema(
        {
            "schema_version": Column(String),
            "ts": Column(Float),
            "run_id": Column(String),
            "episode_id": Column(String),
            "step": Column(Int, checks=pa.Check.ge(1)),
            "action": Column(Int),
            "reward": Column(Float),
            "terminated": Column(Bool),
            "truncated": Column(Bool),
            "agent_x": Column(Int),
            "agent_y": Column(Int),
            "distance_to_goal": Column(Float, checks=pa.Check.ge(0)),
        },
        coerce=True,
    )

    episodes_schema = DataFrameSchema(
        {
            "schema_version": Column(String),
            "run_id": Column(String),
            "episode_id": Column(String),
            "terminated": Column(Bool),
            "truncated": Column(Bool),
            "total_steps": Column(Int, checks=pa.Check.ge(0)),
            "total_reward": Column(Float),
            "final_x": Column(Int),
            "final_y": Column(Int),
            "final_distance_to_goal": Column(
                Float, nullable=True, checks=pa.Check.ge(0)
            ),
        },
        coerce=True,
    )

    return steps_schema, episodes_schema


def validate_run_artifacts(run_dir: Path) -> Dict[str, object]:
    """Validate steps.jsonl.gz and episodes.jsonl.gz with Pandera.

    Returns a report dict with per-file status and optional error messages.
    """
    report: Dict[str, object] = {"steps": None, "episodes": None}
    steps_rows = _read_jsonl_gz_files(run_dir / "steps")
    ep_rows = _read_jsonl_gz_files(run_dir / "episodes")
    try:
        if steps_rows:
            sdf = _flatten_steps(steps_rows)
            steps_schema, _ = _schemas()
            steps_schema.validate(sdf, lazy=True)
            report["steps"] = {"ok": True, "rows": len(sdf) if sdf is not None else 0}
        else:
            report["steps"] = {"ok": True, "rows": 0}
    except Exception as e:
        report["steps"] = {"ok": False, "error": str(e)}
    try:
        if ep_rows:
            edf = _flatten_episodes(ep_rows)
            _, episodes_schema = _schemas()
            episodes_schema.validate(edf, lazy=True)
            report["episodes"] = {
                "ok": True,
                "rows": len(edf) if edf is not None else 0,
            }
        else:
            report["episodes"] = {"ok": True, "rows": 0}
    except Exception as e:
        report["episodes"] = {"ok": False, "error": str(e)}
    return report
