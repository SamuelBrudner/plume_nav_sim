"""Strip outputs and execution counts from Jupyter notebooks.

Usage:
    python strip_notebook_outputs.py notebook.ipynb [more.ipynb ...]

This script modifies notebooks in place, removing cell outputs and execution
counts to keep commits clean and diffs minimal. Metadata is preserved except
for common transient keys under cell metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

TRANSIENT_CELL_META = {
    "execution",  # VSCode/Jupyter transient execution info
    "collapsed",
    "scrolled",
    "ExecuteTime",
    "papermill",
    "vscode",
    "widgets",
}


def clean_notebook(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(
            f"[nb-clean] Skipping {path}: failed to read/parse JSON ({e})",
            file=sys.stderr,
        )
        return False

    cells = data.get("cells", [])
    changed = False

    for cell in cells:
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                cell["outputs"] = []
                changed = True
            if cell.get("execution_count") is not None:
                cell["execution_count"] = None
                changed = True

        # Clean transient metadata keys but keep other metadata (e.g., tags)
        meta: Dict[str, Any] = cell.get("metadata", {})
        removed = False
        for k in list(meta.keys()):
            if k in TRANSIENT_CELL_META:
                meta.pop(k, None)
                removed = True
        if removed:
            cell["metadata"] = meta
            changed = True

    if changed:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    return changed


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "Usage: python strip_notebook_outputs.py notebook.ipynb [more.ipynb ...]",
            file=sys.stderr,
        )
        return 2

    any_changed = False
    for arg in argv[1:]:
        p = Path(arg)
        if not p.exists() or p.suffix.lower() != ".ipynb":
            print(
                f"[nb-clean] Skipping {p}: not found or not a .ipynb", file=sys.stderr
            )
            continue
        changed = clean_notebook(p)
        if changed:
            print(f"[nb-clean] Cleaned: {p}")
            any_changed = True
        else:
            print(f"[nb-clean] Already clean: {p}")

    return 0 if any_changed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
