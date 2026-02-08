from __future__ import annotations

import builtins
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from plume_nav_sim.utils.video import save_video


class _StubIIO:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, object]]] = []

    def imwrite(self, path, frames, **kwargs):  # type: ignore[no-untyped-def]
        frame_count = len(frames) if isinstance(frames, list) else len(list(frames))
        self.calls.append((path, frame_count, kwargs))


def _install_imageio_stub(monkeypatch: pytest.MonkeyPatch) -> _StubIIO:
    stub = _StubIIO()
    mod_imageio = types.ModuleType("imageio")
    mod_v3 = types.ModuleType("imageio.v3")
    mod_v3.imwrite = stub.imwrite  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", mod_imageio)
    monkeypatch.setitem(sys.modules, "imageio.v3", mod_v3)
    return stub


def _tiny_frames(n: int = 3) -> list[np.ndarray]:
    return [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]


def test_save_video_gif_from_frames(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    stub = _install_imageio_stub(monkeypatch)
    out = tmp_path / "tiny.gif"

    save_video(_tiny_frames(3), out)

    assert stub.calls, "imageio.imwrite was not called"
    path, frame_count, kwargs = stub.calls[-1]
    assert path == str(out)
    assert frame_count == 3
    assert kwargs.get("fps") == 30
    assert kwargs.get("loop") == 0


def test_save_video_from_step_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    stub = _install_imageio_stub(monkeypatch)

    @dataclass
    class MockStepEvent:
        frame: np.ndarray | None

    frames = _tiny_frames(2)
    events = [MockStepEvent(frames[0]), MockStepEvent(None), MockStepEvent(frames[1])]
    out = tmp_path / "tiny.mp4"

    save_video(events, out, fps=12)

    assert stub.calls, "imageio.imwrite was not called"
    path, frame_count, kwargs = stub.calls[-1]
    assert path == str(out)
    assert frame_count == 2
    assert kwargs.get("fps") == 12
    assert kwargs.get("codec") == "libx264"


def test_save_video_missing_imageio(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    original_import = builtins.__import__

    def _raise_when_imageio(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("imageio"):
            raise ImportError("imageio missing for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "imageio", raising=False)
    monkeypatch.delitem(sys.modules, "imageio.v3", raising=False)
    monkeypatch.setattr(builtins, "__import__", _raise_when_imageio)

    with pytest.raises(ImportError, match=r"pip install plume-nav-sim\[media\]"):
        save_video(_tiny_frames(1), tmp_path / "missing.gif")
