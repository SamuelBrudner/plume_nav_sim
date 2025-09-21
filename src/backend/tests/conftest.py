"""Pytest fixtures that provide realistic stand-ins for the trimmed test suite."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import tracemalloc
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

import pytest

try:
    import psutil as _psutil
except (
    ImportError
):  # pragma: no cover - dependency may be unavailable in lightweight environments
    _psutil = None


def _require_psutil() -> "psutil":
    """Return the real :mod:`psutil` module or fail loudly when unavailable."""

    if _psutil is None:
        message = (
            "psutil is required for performance monitoring fixtures; install psutil to run "
            "the trimmed performance suite."
        )
        logging.getLogger(__name__).error(message)
        raise RuntimeError(message)
    return _psutil


from plume_nav_sim.envs.plume_search_env import PlumeSearchEnv, create_plume_search_env

logger = logging.getLogger(__name__)

try:
    import matplotlib  # noqa: F401  # Imported for capability detection only

    matplotlib_available = True
except Exception:  # pragma: no cover - matplotlib is optional in CI
    matplotlib_available = False

single_threaded_only = False


@dataclass
class PerformanceTracker(MutableMapping[str, Any]):
    """Lightweight performance tracker used by the edge-case suite.

    The historical fixture behaved like a dictionary in the performance tests, so this
    implementation exposes a :class:`MutableMapping` interface while still providing
    the richer helper methods needed by the edge-case scenarios. The mapping view is
    backed by thread-safe storage so that tests can append timing samples or record
    benchmark summaries without race conditions.
    """

    _session_name: Optional[str] = None
    _start_time: Optional[float] = None
    _step_durations: list[float] = field(default_factory=list)
    _total_steps: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _process: "psutil.Process" = field(
        default_factory=lambda: _require_psutil().Process()
    )
    _store: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        with self._lock:
            baseline_memory = self._process.memory_info().rss / (1024 * 1024)
            self._store.update(
                {
                    "timing_measurements": self._step_durations,
                    "memory_samples": [],
                    "baseline_memory": baseline_memory,
                    "process_monitor": self._process,
                }
            )

    # MutableMapping interface -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._store[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._store[key]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._store.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    # Performance tracking helpers --------------------------------------------
    def start_monitoring(self, session_name: str) -> None:
        with self._lock:
            logger.info("Starting performance session: %s", session_name)
            self._session_name = session_name
            self._start_time = time.perf_counter()
            self._step_durations.clear()
            self._total_steps = 0

    def record_step(self, duration_seconds: float) -> None:
        with self._lock:
            if self._session_name is None:
                return
            self._step_durations.append(float(duration_seconds))
            self._total_steps += 1

    def get_metrics(self) -> Dict[str, float | int | str | None]:
        with self._lock:
            average = (
                sum(self._step_durations) / len(self._step_durations)
                if self._step_durations
                else 0.0
            )
            maximum = max(self._step_durations) if self._step_durations else 0.0
            total_runtime = 0.0
            if self._start_time is not None and self._session_name is not None:
                total_runtime = time.perf_counter() - self._start_time
            return {
                "session": self._session_name,
                "total_steps": self._total_steps,
                "average_step_time": average,
                "max_step_time": maximum,
                "runtime": total_runtime,
            }

    def stop_monitoring(self) -> Dict[str, float | int | str | None]:
        metrics = self.get_metrics()
        with self._lock:
            logger.info("Stopping performance session: %s", self._session_name)
            self._session_name = None
            self._start_time = None
            self._step_durations.clear()
            self._total_steps = 0
        return metrics

    def reset(self) -> None:
        with self._lock:
            logger.debug("Resetting performance tracker state")
            self._session_name = None
            self._start_time = None
            self._step_durations.clear()
            self._total_steps = 0
            self._store["timing_measurements"] = self._step_durations
            self._store.setdefault("memory_samples", []).clear()
            self._store["baseline_memory"] = self._process.memory_info().rss / (
                1024 * 1024
            )


class MemoryMonitor:
    """Track Python heap usage via :mod:`tracemalloc`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._baseline: Optional[tuple[int, int]] = None

    def start_monitoring(self) -> None:
        with self._lock:
            self._baseline = tracemalloc.get_traced_memory()
            logger.debug("Memory baseline captured: %s", self._baseline)

    def stop_monitoring(self) -> Dict[str, float]:
        with self._lock:
            current, peak = tracemalloc.get_traced_memory()
            baseline_current = self._baseline[0] if self._baseline else 0
            delta = max(0, current - baseline_current)
            logger.info("Memory usage delta %.2f KB", delta / 1024)
            return {
                "memory_mb": current / (1024 * 1024),
                "peak_memory_mb": peak / (1024 * 1024),
                "delta_mb": delta / (1024 * 1024),
                "timestamp": time.time(),
            }

    def get_current_usage(self) -> Dict[str, float]:
        with self._lock:
            current, peak = tracemalloc.get_traced_memory()
            return {
                "memory_mb": current / (1024 * 1024),
                "peak_memory_mb": peak / (1024 * 1024),
                "timestamp": time.time(),
            }


class CleanupValidator:
    """Helper that reasons about resource cleanup."""

    def validate(self, before_mb: float, after_mb: float) -> Dict[str, Any]:
        reclaimed = max(0.0, before_mb - after_mb)
        logger.debug("Cleanup reclaimed %.4f MB", reclaimed)
        return {"reclaimed_mb": reclaimed, "cleanup_successful": reclaimed >= 0.0}

    def suggest_actions(self) -> str:
        return "Release references and invoke gc.collect() to reclaim memory."


class TestEnvironmentManager:
    """Context manager that provisions and tears down test environments."""

    def __init__(self, **config: Any) -> None:
        self._config = config
        self._environment: Optional[PlumeSearchEnv] = None

    def __enter__(self) -> PlumeSearchEnv:
        self._environment = create_plume_search_env(**self._config)
        logger.info("Provisioned test environment with config: %s", self._config)
        return self._environment

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._environment is not None:
            self._environment.close()
            logger.info("Closed managed environment")
        self._environment = None


@pytest.fixture(scope="session")
def performance_tracker() -> PerformanceTracker:
    """Session-wide performance tracker shared across tests."""

    return PerformanceTracker()


@pytest.fixture(scope="session")
def memory_monitor() -> MemoryMonitor:
    """Provide a shared memory monitor so long-running tests can compare deltas."""

    monitor = MemoryMonitor()
    monitor.start_monitoring()
    return monitor


@pytest.fixture(scope="session")
def cleanup_validator() -> CleanupValidator:
    """Return the cleanup validator utility expected by the tests."""

    return CleanupValidator()


@pytest.fixture
def test_environment_manager() -> TestEnvironmentManager:
    """Factory fixture returning a new :class:`TestEnvironmentManager`."""

    return TestEnvironmentManager()


@pytest.fixture
def edge_case_test_env(
    performance_tracker: PerformanceTracker,
) -> Generator[PlumeSearchEnv, None, None]:
    """Provision a deterministic environment instrumented for the edge-case suite."""

    environment = create_plume_search_env(grid_size=(64, 64), source_location=(32, 32))

    original_step = environment.step

    def tracked_step(action: Any) -> Any:
        start = time.perf_counter()
        result = original_step(action)
        duration = time.perf_counter() - start
        performance_tracker.record_step(duration)
        return result

    environment.step = tracked_step  # type: ignore[assignment]

    try:
        yield environment
    finally:
        environment.close()


@pytest.fixture
def edge_case_fixture() -> Generator[TestEnvironmentManager, None, None]:
    """Compatibility fixture used by a handful of parametrised tests."""

    manager = TestEnvironmentManager()
    try:
        yield manager
    finally:
        with contextlib.suppress(Exception):
            manager.__exit__(None, None, None)


@pytest.fixture
def unit_test_env() -> Generator[PlumeSearchEnv, None, None]:
    """Lightweight environment instance for unit test scenarios."""

    env = create_plume_search_env(grid_size=(16, 16))
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def integration_test_env() -> Generator[PlumeSearchEnv, None, None]:
    """Environment instance used by integration-style checks."""

    env = create_plume_search_env(grid_size=(24, 24))
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def performance_test_env(
    performance_tracker: PerformanceTracker,
) -> Generator[PlumeSearchEnv, None, None]:
    """Environment instrumented for performance tests."""

    env = create_plume_search_env(grid_size=(32, 32))
    original_step = env.step

    def tracked_step(action: Any) -> Any:
        start = time.perf_counter()
        result = original_step(action)
        duration = time.perf_counter() - start
        performance_tracker.record_step(duration)
        return result

    env.step = tracked_step  # type: ignore[assignment]

    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def reproducibility_test_env() -> Generator[PlumeSearchEnv, None, None]:
    """Environment dedicated to reproducibility checks."""

    env = create_plume_search_env(grid_size=(20, 20))
    try:
        yield env
    finally:
        env.close()
