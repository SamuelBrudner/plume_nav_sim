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


class _VirtualMemory:
    __slots__ = ("total", "available", "percent", "used", "free")

    def __init__(self, total: int, available: int) -> None:
        total = max(int(total), 0)
        available = (
            min(max(int(available), 0), total) if total else max(int(available), 0)
        )
        used = max(total - available, 0)
        self.total = total
        self.available = available
        self.used = used
        self.free = available
        self.percent = float((used / total) * 100) if total else 0.0


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


def _physical_memory_bytes() -> Tuple[int, int]:
    """Best-effort physical/available memory estimation without psutil."""

    page_size = None
    phys_pages = None
    avail_pages = None

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]
    except Exception:
        page_size = 4096

    try:
        phys_pages = os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
    except Exception:
        phys_pages = None

    try:
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")  # type: ignore[attr-defined]
    except Exception:
        avail_pages = None

    if phys_pages is not None:
        total = int(phys_pages) * int(page_size)
    else:
        total = 0

    if avail_pages is not None:
        available = int(avail_pages) * int(page_size)
    else:
        available = max(total - _get_rss_bytes(), 0)

    # Guard against bogus values
    if total <= 0:
        total = 0
        available = max(available, 0)
    else:
        available = min(max(available, 0), total)

    return total, available


def virtual_memory() -> _VirtualMemory:
    """Return minimal virtual memory information compatible with psutil.svmem."""

    total, available = _physical_memory_bytes()
    return _VirtualMemory(total, available)
