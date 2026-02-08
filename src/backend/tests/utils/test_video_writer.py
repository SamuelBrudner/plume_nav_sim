from __future__ import annotations

import builtins
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from plume_nav_sim.utils.video import frames_from_events, save_video


class _StubIIO:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, object]]] = []

    def imwrite(self, path, frames, **kwargs):  # type: ignore[no-untyped-def]
        frame_count = len(frames) if isinstance(frames, list) else len(list(frames))
        self.calls.append((path, frame_count, kwargs))


class _StubPILImage:
    def __init__(self) -> None:
        self.fromarray_calls = 0
        self.save_calls: list[tuple[str, dict[str, object]]] = []

    class _ImageFrame:
        def __init__(self, owner: "_StubPILImage") -> None:
            self._owner = owner

        def save(self, path, **kwargs):  # type: ignore[no-untyped-def]
            self._owner.save_calls.append((path, kwargs))
            with open(path, "wb") as fh:
                fh.write(b"GIF89a")

    def fromarray(self, _frame):  # type: ignore[no-untyped-def]
        self.fromarray_calls += 1
        return self._ImageFrame(self)


def _install_imageio_stub(monkeypatch: pytest.MonkeyPatch) -> _StubIIO:
    stub = _StubIIO()
    mod_imageio = types.ModuleType("imageio")
    mod_v3 = types.ModuleType("imageio.v3")
    mod_v3.imwrite = stub.imwrite  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", mod_imageio)
    monkeypatch.setitem(sys.modules, "imageio.v3", mod_v3)
    return stub


def _install_pillow_stub(monkeypatch: pytest.MonkeyPatch) -> _StubPILImage:
    stub = _StubPILImage()
    mod_pil = types.ModuleType("PIL")
    mod_image = types.ModuleType("PIL.Image")
    mod_image.fromarray = stub.fromarray  # type: ignore[attr-defined]
    mod_pil.Image = mod_image  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", mod_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", mod_image)
    return stub


def _block_imageio_import(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def _raise_when_imageio(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("imageio"):
            raise ImportError("imageio missing for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "imageio", raising=False)
    monkeypatch.delitem(sys.modules, "imageio.v3", raising=False)
    monkeypatch.setattr(builtins, "__import__", _raise_when_imageio)


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

    def _raise_when_media_missing(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("imageio") or name.startswith("PIL"):
            raise ImportError("media dependency missing for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "imageio", raising=False)
    monkeypatch.delitem(sys.modules, "imageio.v3", raising=False)
    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)
    monkeypatch.setattr(builtins, "__import__", _raise_when_media_missing)

    with pytest.raises(ImportError, match=r"pip install plume-nav-sim\[media\]"):
        save_video(_tiny_frames(1), tmp_path / "missing.gif")


def test_save_video_pil_fallback_gif(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _block_imageio_import(monkeypatch)
    stub = _install_pillow_stub(monkeypatch)
    out = tmp_path / "fallback.gif"

    save_video(_tiny_frames(2), out, fps=10)

    assert stub.fromarray_calls == 2
    assert stub.save_calls, "Pillow save was not called"
    path, kwargs = stub.save_calls[-1]
    assert path == str(out)
    assert out.exists()
    assert kwargs.get("save_all") is True
    assert kwargs.get("loop") == 0
    assert kwargs.get("duration") == 100
    append_images = kwargs.get("append_images")
    assert isinstance(append_images, list)
    assert len(append_images) == 1


def test_save_video_mp4_no_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    original_import = builtins.__import__

    def _raise_for_media(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("imageio"):
            raise ImportError("imageio missing for test")
        if name.startswith("PIL"):
            raise AssertionError("PIL fallback should not be used for MP4 output")
        return original_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "imageio", raising=False)
    monkeypatch.delitem(sys.modules, "imageio.v3", raising=False)
    monkeypatch.delitem(sys.modules, "PIL", raising=False)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)
    monkeypatch.setattr(builtins, "__import__", _raise_for_media)

    with pytest.raises(ImportError, match=r"pip install plume-nav-sim\[media\]"):
        save_video(_tiny_frames(1), tmp_path / "missing.mp4")


def test_frames_from_events_extracts_frames() -> None:
    @dataclass
    class MockStepEvent:
        frame: np.ndarray | None

    frames = _tiny_frames(2)
    events = [MockStepEvent(frames[0]), MockStepEvent(None), MockStepEvent(frames[1])]

    extracted = list(frames_from_events(events))

    assert len(extracted) == 2
    assert np.array_equal(extracted[0], frames[0])
    assert np.array_equal(extracted[1], frames[1])
