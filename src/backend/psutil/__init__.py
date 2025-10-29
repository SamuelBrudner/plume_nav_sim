from __future__ import annotations

"""
Lightweight psutil shim for test environments without psutil installed.

Implements a minimal subset used by the test suite:
- Process().memory_info().rss
- cpu_percent(interval=...)
- cpu_count()
- getloadavg()

This is not a full psutil replacement.
"""

import os
from typing import Tuple


def _get_rss_bytes() -> int:
    try:
        import resource  # type: ignore

        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss = int(ru.ru_maxrss)
        # Heuristic: ru_maxrss is kilobytes on many Unix systems, bytes on macOS
        if rss < 10_000_000:  # < ~10 MB: likely reported in KB
            rss *= 1024
        return int(rss)
    except Exception:
        return 0


class _MemInfo:
    def __init__(self, rss: int) -> None:
        self.rss = int(rss)


class Process:
    def __init__(self, pid: int | None = None) -> None:
        self._pid = int(pid) if pid is not None else os.getpid()

    def memory_info(self) -> _MemInfo:
        return _MemInfo(_get_rss_bytes())


def cpu_percent(interval: float | int = 0.0) -> float:
    return 0.0


def cpu_count() -> int:
    return int(os.cpu_count() or 1)


def getloadavg() -> Tuple[float, float, float]:
    try:
        return os.getloadavg()  # type: ignore[attr-defined]
    except Exception:
        return (0.0, 0.0, 0.0)
