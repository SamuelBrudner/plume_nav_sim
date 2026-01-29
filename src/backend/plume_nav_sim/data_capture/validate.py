from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def _has_jsonl_gz(root: Path, stem: str) -> bool:
    if (root / f"{stem}.jsonl.gz").exists():
        return True
    return any(root.glob(f"{stem}.part*.jsonl.gz"))


def validate_run_artifacts(run_dir: Path | str) -> Dict[str, object]:
    """Lightweight validation of capture artifacts.

    Checks for run.json and steps/episodes JSONL.gz files.
    """
    run_path = Path(run_dir)
    report: Dict[str, object] = {
        "run_dir": str(run_path),
        "ok": True,
        "errors": [],
        "warnings": [],
    }

    if not run_path.exists():
        report["ok"] = False
        report["errors"].append("run_dir_not_found")
        return report

    meta_path = run_path / "run.json"
    if not meta_path.exists():
        report["ok"] = False
        report["errors"].append("missing_run_json")
    else:
        try:
            json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            report["ok"] = False
            report["errors"].append("invalid_run_json")

    if not _has_jsonl_gz(run_path, "steps"):
        report["ok"] = False
        report["errors"].append("missing_steps_jsonl_gz")

    if not _has_jsonl_gz(run_path, "episodes"):
        report["ok"] = False
        report["errors"].append("missing_episodes_jsonl_gz")

    return report
