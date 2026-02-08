from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_MAIN = PROJECT_ROOT / "plug-and-play-demo" / "main.py"


def _count_gif_frames(gif_path: Path) -> int:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"Pillow unavailable for GIF frame validation: {exc}")

    frames = 0
    with Image.open(gif_path) as image:
        try:
            while True:
                frames += 1
                image.seek(image.tell() + 1)
        except EOFError:
            pass
    return frames


def test_capture_mode_save_gif_writes_non_empty_gif(tmp_path: Path) -> None:
    gif_path = tmp_path / "demo.gif"
    capture_root = tmp_path / "capture"

    cmd = [
        sys.executable,
        str(DEMO_MAIN),
        "--max-steps",
        "120",
        "--save-gif",
        str(gif_path),
        "--capture-root",
        str(capture_root),
        "--experiment",
        "demo",
        "--episodes",
        "1",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"demo run failed with code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert gif_path.exists(), f"expected GIF not written: {gif_path}"
    assert gif_path.stat().st_size > 0, "GIF file is empty"
    assert _count_gif_frames(gif_path) > 0, "GIF should contain at least one frame"
