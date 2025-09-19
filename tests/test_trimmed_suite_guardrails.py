"""Tests ensuring trimmed test suite surfaces missing dependencies loudly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
BACKEND_SRC_DIR = SRC_DIR / "backend"
for path in (SRC_DIR, BACKEND_SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    """Load a module from the src tree without triggering package side effects."""

    full_path = SRC_DIR / relative_path
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Unable to load module spec for {module_name} from {full_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_require_psutil_fails_loud_when_unavailable(monkeypatch):
    """_require_psutil should raise a loud failure when psutil is missing."""

    conftest = _load_module("trimmed_conftest", "backend/tests/conftest.py")
    original_psutil = getattr(conftest, "_psutil", None)
    monkeypatch.setattr(conftest, "_psutil", None, raising=False)
    monkeypatch.setattr(
        conftest.pytest,
        "skip",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("skip invoked")),
    )

    with pytest.raises(RuntimeError) as excinfo:
        conftest._require_psutil()

    message = str(excinfo.value)
    assert "psutil" in message
    assert "performance" in message.lower()

    if original_psutil is not None:
        monkeypatch.setattr(conftest, "_psutil", original_psutil, raising=False)


def test_require_full_test_suite_raises_instead_of_skip(monkeypatch):
    """The trimmed suite hook should raise an explicit error instead of skipping."""

    module = _load_module(
        "trimmed_plume_nav_sim_tests",
        "backend/tests/plume_nav_sim/__init__.py",
    )
    if hasattr(module, "pytest"):
        monkeypatch.setattr(
            module.pytest,
            "skip",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("skip invoked")),
        )

    with pytest.raises(RuntimeError) as excinfo:
        module.require_full_test_suite()

    message = str(excinfo.value).lower()
    assert "full" in message
    assert "test" in message
    assert "suite" in message


def test_render_namespace_access_is_informative():
    """Attempting to access trimmed render helpers should raise a descriptive error."""

    module = _load_module(
        "trimmed_plume_nav_sim_render",
        "backend/tests/plume_nav_sim/render/__init__.py",
    )

    with pytest.raises(AttributeError) as excinfo:
        getattr(module, "nonexistent_helper")

    message = str(excinfo.value).lower()
    assert "render" in message
    assert "not included" in message
