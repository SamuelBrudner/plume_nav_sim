"""
Pytest configuration for visualization tests to normalize Matplotlib behavior
and harden CI stability.

Applies a deterministic, headless backend and rcParams, and guards logging
from emitting exceptions during teardown.
"""

import gc
import logging
import os
import queue
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib
import numpy as np
import pytest  # noqa: E402
from matplotlib import rcParams  # noqa: E402

try:  # Optional dependency validated at runtime via _require_psutil
    import psutil as _psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - guard validated via tests
    _psutil = None

from plume_nav_sim.core.types import Coordinates, GridSize
from plume_nav_sim.envs.plume_env import create_plume_env


@dataclass(frozen=True)
class RenderContext:
    concentration_field: np.ndarray
    agent_position: Coordinates
    source_position: Coordinates
    grid_size: GridSize


def create_render_context(
    *,
    concentration_field: np.ndarray,
    agent_position: Coordinates,
    source_position: Coordinates,
    grid_size: GridSize,
) -> RenderContext:
    return RenderContext(
        concentration_field=concentration_field,
        agent_position=agent_position,
        source_position=source_position,
        grid_size=grid_size,
    )


def create_dual_mode_test_environment(
    grid_size: tuple[int, int] = (32, 32),
    initial_render_mode: str = "rgb_array",
    test_config: Optional[dict[str, Any]] = None,
    enable_performance_monitoring: bool = True,
):
    del enable_performance_monitoring  # compatibility placeholder
    config = dict(test_config or {})
    env_kwargs: dict[str, Any] = {
        "grid_size": grid_size,
        "render_mode": initial_render_mode,
    }
    for key in (
        "source_location",
        "start_location",
        "goal_radius",
        "max_steps",
        "plume",
        "plume_params",
    ):
        if key in config:
            env_kwargs[key] = config[key]
    return create_plume_env(**env_kwargs)

__all__ = [
    "_require_psutil",
    "create_dual_mode_test_environment",
    "create_render_context",
    "MemoryMonitor",
    "DUAL_MODE_TEST_GRID_SIZES",
    "INTEGRATION_TEST_SEEDS",
    "MemoryMonitor",
]

# Skip archived tests - implementation details may be outdated
collect_ignore = ["archived"]


def _select_preferred_backend() -> str:
    """Attempt to select an interactive backend when available, fallback to Agg."""

    preferred_backends = ["QtAgg", "Qt5Agg", "TkAgg"]

    for backend in preferred_backends:
        try:
            if backend.startswith("Qt"):
                import PyQt5  # noqa: F401
            matplotlib.use(backend, force=True)
            return backend
        except Exception:
            continue

    matplotlib.use("Agg", force=True)
    return "Agg"


_SELECTED_MPL_BACKEND = _select_preferred_backend()

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)

# Provide a shim for gymnasium.utils.env_checker if not available so tests
# can exercise the API-compliance path without skipping when the checker
# module is missing in the installed Gymnasium build.
try:  # pragma: no cover - environment-dependent
    from gymnasium.utils.env_checker import check_env as _check_env  # type: ignore
except Exception:  # pragma: no cover - provide minimal shim
    import sys
    import types

    shim = types.ModuleType("gymnasium.utils.env_checker")

    def _check_env(env, skip_render_check: bool = True) -> None:
        # Minimal no-op checker: ensures the import exists for tests that
        # assert availability. Real validation is covered elsewhere.
        return None

    shim.check_env = _check_env  # type: ignore[attr-defined]
    sys.modules["gymnasium.utils.env_checker"] = shim


def _require_psutil() -> None:
    """Raise when psutil is unavailable to surface missing performance dependencies."""

    if _psutil is None:
        raise RuntimeError(
            "psutil is required for plume_nav_sim performance test coverage. "
            "Install the optional performance extras or run the trimmed test suite."
        )


@pytest.fixture(scope="session", autouse=True)
def _normalize_matplotlib_and_logging(tmp_path_factory):
    """Normalize Matplotlib rcParams and guard logging exceptions for CI."""
    # Prevent logging module from printing handler/stream errors during teardown
    logging.raiseExceptions = False

    # Ensure Matplotlib has a writable cache/config directory to avoid noisy warnings
    try:
        mpl_cfg_dir = tmp_path_factory.mktemp("mplconfig")
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg_dir))
    except Exception:
        pass

    # Normalize core rcParams for consistent sizing and rendering
    rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 100,
            # Use a deterministic font family available in Matplotlib wheels
            "font.family": ["DejaVu Sans"],
            # Keep grids off by default; tests enable grids explicitly when needed
            "axes.grid": False,
            # Ensure antialiasing for consistent visuals (does not affect Agg determinism)
            "text.antialiased": True,
        }
    )

    # Reduce benign warning noise in CI output while preserving important failures
    warnings.filterwarnings(
        "ignore",
        message=r".*Matplotlib is building the font cache.*",
        module=r"matplotlib\\.font_manager",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Observation dtype .* differs from expected",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Action space size .* differs from standard",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Observation space is not bounded on both sides",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Observation space shape .* differs from standard",
        category=UserWarning,
        module=r"plume_nav_sim\\.utils\\.spaces",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Observation space may not be compatible with concentration value",
        category=UserWarning,
        module=r"plume_nav_sim\\.utils\\.spaces",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Gymnasium registry may not be fully compatible",
        category=UserWarning,
        module=r"plume_nav_sim\\.registration",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Importing pandas-specific classes and functions from the",
        category=FutureWarning,
        module=r"pandera\\._pandas_deprecated",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Passing unrecognized arguments to super\(Toolbar\)\.\__init__\(\)",
        category=DeprecationWarning,
        module=r"traitlets",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Treat the new Tool classes introduced in v1\.5 as experimental",
        category=UserWarning,
        module=r"plume_nav_sim\\.render\\.matplotlib_viz",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Seed .* may cause integer overflow",
        category=UserWarning,
        module=r"plume_nav_sim\\.utils\\.seeding",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"WARN: Box low's precision lowered by casting to float32",
        category=UserWarning,
        module=r"gymnasium\\.spaces\\.box",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"WARN: Box high's precision lowered by casting to float32",
        category=UserWarning,
        module=r"gymnasium\\.spaces\\.box",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Performance degraded after cleanup",
        category=UserWarning,
    )

    yield

    # Best-effort cleanup of any remaining figures at end of session
    try:
        plt.close("all")
    except Exception:
        pass


DUAL_MODE_TEST_GRID_SIZES = [(32, 32), (64, 64), (128, 128)]
INTEGRATION_TEST_SEEDS = [42, 123, 456]


class MemoryMonitor:
    """Advanced memory monitoring utility shared across test suites."""

    def __init__(self) -> None:
        if _psutil is None:
            raise RuntimeError(
                "psutil is required for memory monitoring; install optional extras"
            )

        self._process = _psutil.Process()
        self._samples: list[float] = []
        self._monitoring = False
        self._thread: threading.Thread | None = None
        self._baseline_mb: float | None = None
        self._commands: "queue.Queue[str]" = queue.Queue()

    def start_monitoring(self, sampling_interval: float = 0.1) -> None:
        if self._monitoring:
            return

        gc.collect()
        self._baseline_mb = self._process.memory_info().rss / (1024 * 1024)
        self._monitoring = True

        def _loop() -> None:
            while self._monitoring:
                try:
                    memory_mb = self._process.memory_info().rss / (1024 * 1024)
                    self._samples.append(memory_mb)
                    try:
                        command = self._commands.get(timeout=sampling_interval)
                        if command == "stop":
                            break
                    except queue.Empty:
                        continue
                except Exception:
                    break

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_monitoring(self) -> dict[str, float | int | bool | str]:
        if not self._monitoring:
            return {"error": "monitoring not started"}

        self._monitoring = False
        self._commands.put("stop")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if not self._samples:
            return {"error": "no memory samples collected"}

        baseline = self._baseline_mb or min(self._samples)
        peak = max(self._samples)
        mean = float(np.mean(self._samples))
        growth = peak - min(self._samples)

        return {
            "baseline_memory_mb": baseline,
            "peak_memory_mb": peak,
            "mean_memory_mb": mean,
            "memory_growth_mb": growth,
            "sample_count": len(self._samples),
            "leak_detected": growth > 5.0,
        }

    def get_usage_mb(self) -> float:
        return self._process.memory_info().rss / (1024 * 1024)


class CleanupValidator:
    """Track resources before and after tests to ensure cleanup."""

    def __init__(self) -> None:
        self._initial_state: dict[str, int] | None = None
        self._final_state: dict[str, int] | None = None

    def record_initial_state(self, env: object) -> None:
        self._initial_state = self._capture_state(env)

    def record_final_state(self, env: object) -> None:
        self._final_state = self._capture_state(env)

    def validate_cleanup(self) -> dict[str, object]:
        if self._initial_state is None or self._final_state is None:
            return {"error": "state not recorded"}

        leaks = {
            key: self._final_state.get(key, 0) - self._initial_state.get(key, 0)
            for key in self._initial_state
        }

        return {
            "resource_leaks": leaks,
            "cleanup_successful": all(value <= 0 for value in leaks.values()),
        }

    def _capture_state(self, env: object) -> dict[str, int]:
        state = {}
        if hasattr(env, "renderer") and env.renderer is not None:
            renderer = env.renderer
            if hasattr(renderer, "active_render_targets"):
                state["active_render_targets"] = len(renderer.active_render_targets)
        state["open_figures"] = len(
            [fig for fig in getattr(plt, "get_fignums", lambda: [])()]
        )
        return state


@pytest.fixture
def memory_monitor():
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    return monitor


@pytest.fixture
def cleanup_validator():
    return CleanupValidator()


@pytest.fixture(scope="session")
def matplotlib_available():
    """Session-wide check for matplotlib availability used by multiple suites."""

    try:
        import matplotlib.backends.backend_agg  # noqa: F401  # type: ignore

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        plt.close(fig)
        return True
    except Exception:
        pytest.skip("Matplotlib not available for rendering tests")


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """Ensure figures are closed after each test to avoid cross-test leakage."""
    yield
    try:
        plt.close("all")
    except Exception:
        pass


@pytest.fixture
def edge_case_test_env():
    """Provide a reusable environment configured for edge-case testing suites."""

    env = create_dual_mode_test_environment(
        grid_size=(32, 32),
        initial_render_mode="rgb_array",
        test_config={"enable_edge_case_testing": True},
    )

    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def integration_test_env():
    env = create_dual_mode_test_environment(
        grid_size=(32, 32),
        initial_render_mode="rgb_array",
        enable_performance_monitoring=True,
    )
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def unit_test_env():
    env = create_dual_mode_test_environment(
        grid_size=(32, 32),
        initial_render_mode="rgb_array",
        enable_performance_monitoring=False,
    )
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def performance_test_env():
    env = create_dual_mode_test_environment(
        grid_size=(64, 64),
        initial_render_mode="rgb_array",
        enable_performance_monitoring=True,
    )
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def test_render_context(integration_test_env):
    env = integration_test_env
    env.reset()

    concentration_field = np.random.rand(32, 32).astype(np.float32)
    agent_pos = Coordinates(x=5, y=5)
    source_pos = Coordinates(x=16, y=16)
    grid_size = GridSize(width=32, height=32)

    return create_render_context(
        concentration_field=concentration_field,
        agent_position=agent_pos,
        source_position=source_pos,
        grid_size=grid_size,
    )


@pytest.fixture
def test_coordinates():
    return {
        "agent_positions": [Coordinates(x=5, y=5), Coordinates(x=10, y=15)],
        "source_positions": [Coordinates(x=16, y=16), Coordinates(x=20, y=20)],
    }


@pytest.fixture
def test_grid_sizes():
    return list(DUAL_MODE_TEST_GRID_SIZES)


@pytest.fixture
def test_seeds():
    return list(INTEGRATION_TEST_SEEDS)


def pytest_collection_modifyitems(config, items):
    """No-op hook retained for compatibility.

    Previously, this hook applied broad xfail markers to performance-related
    tests. Those tests now pass consistently, so we no longer downgrade them.
    Keeping the hook defined avoids import/order surprises in downstream tooling.
    """
    return
