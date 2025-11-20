from __future__ import annotations

import json
from pathlib import Path


def test_capture_end_to_end_notebook_structure():
    nb_path = Path("notebooks/stable/capture_end_to_end.ipynb")
    assert nb_path.exists(), "capture_end_to_end.ipynb not found"
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4
    assert isinstance(data.get("cells"), list) and len(data["cells"]) > 0
    # First cell should be a markdown title
    first = data["cells"][0]
    assert first.get("cell_type") == "markdown"
    assert any("Capture Pipeline" in (line or "") for line in first.get("source", []))
