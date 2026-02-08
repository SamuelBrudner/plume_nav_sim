from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_MAIN = PROJECT_ROOT / "plug-and-play-demo" / "main.py"


def test_demo_movie_dataset_id_requires_auto_download_when_uncached(
    tmp_path: Path,
) -> None:
    # Use a private cache root so the test is deterministic and never relies on
    # the user's global cache.
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(DEMO_MAIN),
        "--plume",
        "movie",
        "--movie-dataset-id",
        "colorado_jet_v1",
        "--movie-cache-root",
        str(cache_root),
        "--no-render",
        "--max-steps",
        "1",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    # If the cache is empty and auto-download is not enabled, the demo should
    # fail with a clear message.
    assert result.returncode != 0, (
        "Expected demo to fail without --movie-auto-download when cache is missing.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    combined = (result.stdout + "\n" + result.stderr).lower()
    assert "auto_download=true" in combined or "--movie-auto-download" in combined
