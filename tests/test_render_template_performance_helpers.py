from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import MethodType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND_SRC = ROOT / "src" / "backend"
for path in (ROOT, BACKEND_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _ensure_stub_gymnasium() -> None:
    """Install a minimal gymnasium.utils.seeding stub for imports.

    The trimmed test environment does not provide the full Gymnasium
    dependency, but the seeding utilities only require np_random-like
    behavior. Installing a stub keeps imports working without changing
    package code.
    """

    import types

    # Base gymnasium package stub
    if "gymnasium" not in sys.modules:
        gym_module = types.ModuleType("gymnasium")
        # Mark as package-like for submodule imports (spaces, utils)
        gym_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules["gymnasium"] = gym_module
    else:
        gym_module = sys.modules["gymnasium"]

    # Stub spaces submodule with minimal Box/Discrete types
    if "gymnasium.spaces" not in sys.modules:
        spaces_module = types.ModuleType("gymnasium.spaces")

        class _BaseSpace:  # Minimal stand-in for gymnasium.spaces.Space
            def __init__(self, *args, **kwargs):  # noqa: D401, ANN002, ANN003
                """Lightweight placeholder accepting any arguments."""

        class Discrete(_BaseSpace):  # type: ignore[too-many-ancestors]
            pass

        class Box(_BaseSpace):  # type: ignore[too-many-ancestors]
            pass

        spaces_module.Discrete = Discrete  # type: ignore[attr-defined]
        spaces_module.Box = Box  # type: ignore[attr-defined]
        sys.modules["gymnasium.spaces"] = spaces_module
        gym_module.spaces = spaces_module  # type: ignore[attr-defined]

    # Stub utils.seeding for RNG creation helpers
    if "gymnasium.utils" not in sys.modules:
        utils_module = types.ModuleType("gymnasium.utils")
        sys.modules["gymnasium.utils"] = utils_module
        gym_module.utils = utils_module  # type: ignore[attr-defined]
    else:
        utils_module = sys.modules["gymnasium.utils"]

    seeding_module = types.ModuleType("gymnasium.utils.seeding")

    def np_random(seed=None):  # type: ignore[override]
        rng = np.random.default_rng(seed)
        used_seed = 0 if seed is None else int(seed)
        return rng, used_seed

    seeding_module.np_random = np_random  # type: ignore[attr-defined]
    sys.modules["gymnasium.utils.seeding"] = seeding_module
    utils_module.seeding = seeding_module  # type: ignore[attr-defined]


_ensure_stub_gymnasium()


def _make_rgb_template():
    from plume_nav_sim.core.geometry import GridSize

    templates_module = _load_templates_module()
    TemplateConfig = templates_module.TemplateConfig
    RGBTemplate = templates_module.RGBTemplate

    config = TemplateConfig(grid_size=GridSize(16, 16))
    return RGBTemplate(config)


def _load_templates_module():
    """Load render.templates in isolation to avoid heavy dependencies.

    This mirrors the pattern used in the trimmed guardrail tests to load
    backend modules directly from the src tree without importing the
    higher-level render package (which pulls in gymnasium-heavy pieces).
    """

    module_name = "plume_nav_sim.render.templates_testshim"
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Ensure top-level package is importable so absolute imports used by
    # templates.py (plume_nav_sim.core, etc.) resolve normally.
    import plume_nav_sim  # noqa: F401  # imported for side effects

    # Install a lightweight stub package for plume_nav_sim.render so that
    # relative imports ("from .colormaps import ...") inside templates.py
    # resolve without importing the heavyweight render __init__.
    package_name = "plume_nav_sim.render"
    if package_name not in sys.modules:
        render_pkg = types.ModuleType(package_name)
        render_pkg.__path__ = [
            str(BACKEND_SRC / "plume_nav_sim" / "render")
        ]  # type: ignore[attr-defined]
        sys.modules[package_name] = render_pkg

    templates_path = BACKEND_SRC / "plume_nav_sim" / "render" / "templates.py"
    spec = importlib.util.spec_from_file_location(module_name, templates_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to load templates module from {templates_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Install a lightweight default color scheme that does not depend on
    # modern matplotlib.colormaps APIs. The production colormaps module
    # uses newer matplotlib features that are unavailable in the trimmed
    # test environment, but the performance helpers under test only need
    # a stub with the expected methods.

    class _StubColorScheme:
        def optimize_for_render_mode(self, mode):  # noqa: D401, ANN001
            """No-op optimization stub for tests."""

        def get_concentration_colormap(self, use_cache: bool = True):  # noqa: D401
            """Return a dummy object in place of a real colormap."""

            class _DummyColormap:
                pass

            return _DummyColormap()

        def to_dict(self) -> dict:
            return {}

    module.create_default_scheme = lambda: _StubColorScheme()  # type: ignore[attr-defined]

    return module


def test_rgb_collect_test_results_tracks_violations_and_meets_flag():
    """_rgb_collect_test_results should aggregate per-grid results into analysis.

    This exercises the branch where at least one successful scenario fails
    the performance target and ensures the analysis structure is updated
    consistently.
    """

    template = _make_rgb_template()

    analysis = {"meets_target": True, "target_violations": []}
    rgb_target_ms = 5.0

    def fake_run_single_rgb_scenario(self, scenario_name, grid, target_ms):
        scenario_id = f"{scenario_name}_{grid[0]}x{grid[1]}"
        if scenario_name == "fast":
            return {
                "scenario": scenario_id,
                "success": True,
                "avg_render_time_ms": target_ms - 1.0,
                "max_render_time_ms": target_ms,
                "meets_target": True,
            }
        return {
            "scenario": scenario_id,
            "success": True,
            "avg_render_time_ms": target_ms + 2.0,
            "max_render_time_ms": target_ms + 3.0,
            "meets_target": False,
        }

    template._run_single_rgb_scenario = MethodType(
        fake_run_single_rgb_scenario, template
    )

    test_scenarios = {
        "fast": {"grid_sizes": [(16, 16)]},
        "slow": {"grid_sizes": [(32, 32)]},
    }

    results = template._rgb_collect_test_results(
        test_scenarios, rgb_target_ms, analysis
    )

    assert {r["scenario"] for r in results} == {
        "fast_16x16",
        "slow_32x32",
    }
    assert analysis["meets_target"] is False
    assert len(analysis["target_violations"]) == 1
    violation = analysis["target_violations"][0]
    assert violation["scenario"] == "slow_32x32"
    assert violation["target_time_ms"] == rgb_target_ms
    assert pytest.approx(violation["excess_time_ms"] + rgb_target_ms) == pytest.approx(
        violation["actual_time_ms"]
    )


def test_validate_template_performance_aggregates_benchmarks(monkeypatch):
    """validate_template_performance should summarize benchmark helpers consistently.

    This covers the happy-path and failure-path behavior of
    _run_benchmarks_for_template by stubbing out the low-level benchmark
    iteration helper.
    """

    from plume_nav_sim.core.constants import PERFORMANCE_TARGET_RGB_RENDER_MS

    templates_module = _load_templates_module()
    validate_template_performance = templates_module.validate_template_performance

    template = _make_rgb_template()
    target_ms = PERFORMANCE_TARGET_RGB_RENDER_MS

    def fake_execute_benchmark_iteration(
        template_obj,
        test_field,
        test_agent,
        test_source,
        grid,
    ):
        # Fast for smaller grids, deliberately slow for larger grids so
        # that at least one scenario fails the target.
        base_seconds = target_ms / 1000.0
        if grid[0] <= 16:
            elapsed = base_seconds * 0.5
        else:
            elapsed = base_seconds * 2.0
        return elapsed, 0.0

    monkeypatch.setattr(
        templates_module,
        "_execute_benchmark_iteration",
        fake_execute_benchmark_iteration,
        raising=True,
    )

    test_scenarios = {
        "fast": {"grid_sizes": [(16, 16)], "iterations": 3},
        "slow": {"grid_sizes": [(32, 32)], "iterations": 3},
    }

    passed, report = validate_template_performance(
        template, test_scenarios, strict_validation=True
    )

    assert passed is False

    # Per-scenario summaries include success and target compliance rates.
    assert set(report["test_results"].keys()) == {"fast", "slow"}
    fast_summary = report["test_results"]["fast"]["summary"]
    slow_summary = report["test_results"]["slow"]["summary"]

    assert fast_summary["overall_success_rate"] == pytest.approx(1.0)
    assert fast_summary["target_compliance_rate"] == pytest.approx(1.0)
    assert slow_summary["overall_success_rate"] == pytest.approx(1.0)
    assert slow_summary["target_compliance_rate"] == pytest.approx(0.0)

    # Top-level performance analysis reflects aggregated benchmark data.
    performance = report["performance_analysis"]
    assert performance["target_time_ms"] == pytest.approx(target_ms)
    assert performance["overall_success_rate"] <= 1.0
    assert performance["target_compliance_rate"] < 1.0
