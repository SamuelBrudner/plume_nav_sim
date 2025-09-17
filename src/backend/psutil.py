"""Minimal psutil stub for test environments without the real dependency."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class _MemoryInfo:
    rss: int = 0
    vms: int = 0


@dataclass
class _VirtualMemory:
    total: int = 8 * 1024 * 1024 * 1024  # 8 GB
    available: int = 4 * 1024 * 1024 * 1024  # 4 GB
    percent: float = 50.0
    used: int = total - available
    free: int = available


def virtual_memory() -> _VirtualMemory:
    """Return placeholder virtual memory statistics."""
    return _VirtualMemory()


def cpu_count(logical: bool | None = True) -> int:
    """Return a deterministic CPU count for tests."""
    return os.cpu_count() or 1


def cpu_percent(interval: float | None = None) -> float:
    """Return a safe default CPU utilization value."""
    return 0.0


def getloadavg() -> Tuple[float, float, float]:
    """Provide a harmless default system load average."""
    return (0.0, 0.0, 0.0)


class Process:
    """Lightweight stand-in for psutil.Process used in tests."""

    def __init__(self, pid: int | None = None):
        self.pid = pid if pid is not None else os.getpid()

    def memory_info(self) -> _MemoryInfo:
        return _MemoryInfo()

    def cpu_percent(self, interval: float | None = None) -> float:
        return 0.0

    def num_threads(self) -> int:
        return 1


__all__ = [
    "Process",
    "virtual_memory",
    "cpu_count",
    "cpu_percent",
    "getloadavg",
]
