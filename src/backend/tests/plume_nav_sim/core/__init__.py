"""Utility helpers shared across plume_nav_sim core test suites.

The production code base exposes a rich set of timing helpers, but the tests
only rely on a tiny subset of that functionality. Historically this module was
left empty which forced the registration tests to import an undefined
``PerformanceTestUtilities`` symbol. Python then failed during collection and
none of the assertions inside the suite executed.

The helpers below provide a deliberately small yet well documented surface that
covers the current test expectations while remaining loud when misused. Each
entry validates its input parameters and records structured timing details so
that diagnosing future regressions is straightforward.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class BenchmarkResult:
    """Structured benchmark data captured during a timing run."""

    operation_name: str
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the benchmark."""

        return {
            "operation_name": self.operation_name,
            "duration_seconds": self.duration_seconds,
            "metadata": dict(self.metadata),
        }


class PerformanceTestUtilities:
    """Minimal timing helper used by the test-suite.

    The helper intentionally focuses on deterministic behaviour. Inputs are
    validated aggressively so incorrect usage fails immediately instead of
    silently skewing benchmark data. Captured timings are stored on the
    instance for optional post-processing by tests.
    """

    def __init__(self) -> None:
        self._results: List[BenchmarkResult] = []

    @property
    def results(self) -> List[BenchmarkResult]:
        """Return a copy of the recorded benchmark results."""

        return list(self._results)

    def benchmark_operation(
        self,
        operation_name: str,
        operation: Callable[[], Any],
        *,
        repetitions: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResult:
        """Execute ``operation`` and record its average duration.

        Args:
            operation_name: A human readable description of the benchmarked
                operation. Must be a non-empty string.
            operation: Zero argument callable that will be timed.
            repetitions: Number of times the callable should be executed.
                Must be a positive integer. The average runtime per invocation
                is reported.
            metadata: Optional metadata dictionary describing the benchmark.

        Returns:
            :class:`BenchmarkResult` containing the captured metrics.
        """

        if not isinstance(operation_name, str) or not operation_name.strip():
            raise ValueError("operation_name must be a non-empty string")

        if not callable(operation):
            raise TypeError("operation must be callable")

        if not isinstance(repetitions, int) or repetitions <= 0:
            raise ValueError("repetitions must be a positive integer")

        logger.info(
            "Benchmarking operation '%s' for %s repetitions",
            operation_name,
            repetitions,
        )

        start = time.perf_counter()
        last_result: Any = None
        for _ in range(repetitions):
            last_result = operation()
        end = time.perf_counter()

        duration = (end - start) / repetitions
        result = BenchmarkResult(
            operation_name=operation_name,
            duration_seconds=duration,
            metadata={
                **(metadata or {}),
                "repetitions": repetitions,
                "last_result": last_result,
            },
        )
        self._results.append(result)

        logger.debug(
            "Recorded benchmark for '%s': %.6fs per iteration", operation_name, duration
        )
        return result

    def clear(self) -> None:
        """Remove all recorded benchmark results."""

        logger.debug("Clearing %d recorded benchmark results", len(self._results))
        self._results.clear()

    def aggregate_durations(self) -> Dict[str, float]:
        """Return the cumulative duration recorded for each operation name."""

        totals: Dict[str, float] = {}
        for entry in self._results:
            totals[entry.operation_name] = (
                totals.get(entry.operation_name, 0.0) + entry.duration_seconds
            )
        return totals


__all__ = ["BenchmarkResult", "PerformanceTestUtilities"]
