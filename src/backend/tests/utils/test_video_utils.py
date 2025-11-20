from __future__ import annotations

import types

import numpy as np
import pytest

from plume_nav_sim.utils.video import save_video_events, save_video_frames


class _StubIIO:
    def __init__(self):
        self.calls = []

    def imwrite(self, path, frames, **kwargs):
        # Record a shallow copy of frames info (length) to avoid holding arrays
        length = len(frames) if isinstance(frames, list) else len(list(frames))
        self.calls.append((path, length, kwargs))


def _install_imageio_stub(monkeypatch: pytest.MonkeyPatch) -> _StubIIO:
    stub = _StubIIO()
    # Create a fake imageio.v3 module
    mod_imageio = types.ModuleType("imageio")
    mod_v3 = types.ModuleType("imageio.v3")
    mod_v3.imwrite = stub.imwrite  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "imageio", mod_imageio)
    monkeypatch.setitem(__import__("sys").modules, "imageio.v3", mod_v3)
    return stub


def _make_frames(n=3, size=(8, 8)):
    h, w = size
    return [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_save_video_frames_calls_imageio(monkeypatch: pytest.MonkeyPatch, tmp_path):
    stub = _install_imageio_stub(monkeypatch)
    frames = _make_frames(4)
    out = tmp_path / "demo.gif"
    save_video_frames(frames, out, fps=12)
    assert stub.calls, "imwrite was not called"
    path, nframes, kwargs = stub.calls[-1]
    assert str(out) == path
    assert nframes == 4
    assert kwargs.get("fps") == 12
    # GIF defaults to infinite loop unless overridden
    assert kwargs.get("loop", 0) == 0


def test_save_video_events_filters_none_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    stub = _install_imageio_stub(monkeypatch)

    class Ev:
        def __init__(self, frame):
            self.frame = frame

    frames = _make_frames(3)
    events = [Ev(frames[0]), Ev(None), Ev(frames[1]), Ev(frames[2])]
    out = tmp_path / "demo.mp4"
    save_video_events(events, out, fps=8, codec="libx264")
    assert stub.calls, "imwrite was not called"
    path, nframes, kwargs = stub.calls[-1]
    assert str(out) == path
    assert nframes == 3  # one None skipped
    assert kwargs.get("fps") == 8
    assert kwargs.get("codec") == "libx264"


def test_save_video_import_error(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import builtins

    original_import = builtins.__import__

    def _raise_when_imageio(name, *args, **kwargs):
        if name.startswith("imageio"):
            raise ImportError("imageio not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_when_imageio)

    with pytest.raises(ImportError):
        save_video_frames(_make_frames(1), tmp_path / "x.gif")
