"""Regression tests ensuring backend test suite fails fast when dependencies are missing."""

import importlib
import sys
import types
from pathlib import Path

import pytest
from _pytest.outcomes import Skipped


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
BACKEND_SRC = SRC_ROOT / "backend"

for path in (PROJECT_ROOT, SRC_ROOT, BACKEND_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.mark.usefixtures("monkeypatch")
def test_performance_module_missing_psutil_is_error(monkeypatch):
    """Importing the performance tests without psutil should raise instead of skipping."""
    module_name = "src.backend.tests.test_performance"
    sys.modules.pop(module_name, None)

    # Make sure psutil import fails loudly.
    monkeypatch.setitem(sys.modules, "psutil", None)

    with pytest.raises(RuntimeError) as excinfo:
        try:
            importlib.import_module(module_name)
        except Skipped as skipped_exc:  # pragma: no cover - triggers red phase
            raise AssertionError("performance tests should fail, not skip, when psutil is missing") from skipped_exc

    assert "psutil" in str(excinfo.value)


@pytest.mark.usefixtures("monkeypatch")
def test_tests_package_missing_conftest_raises(monkeypatch):
    """If the shared conftest module is unavailable, importing the package should raise."""
    package_name = "src.backend.tests"
    sys.modules.pop(package_name, None)
    sys.modules.pop(f"{package_name}.conftest", None)

    stub_psutil = types.ModuleType("psutil")
    stub_psutil.Process = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psutil", stub_psutil)

    broken_conftest = types.ModuleType(f"{package_name}.conftest")
    monkeypatch.setitem(sys.modules, f"{package_name}.conftest", broken_conftest)

    with pytest.raises(ImportError):
        try:
            importlib.import_module(package_name)
        except Skipped as skipped_exc:  # pragma: no cover - triggers red phase
            raise AssertionError("tests package should fail loudly when conftest is missing") from skipped_exc
