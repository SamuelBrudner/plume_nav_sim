"""Minimal formatter bridge providing the performance formatter interface."""

from __future__ import annotations

import logging


class PerformanceFormatter(logging.Formatter):
    """Simple formatter that annotates records with millisecond durations."""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "duration_ms"):
            record.duration_ms = 0.0  # type: ignore[attr-defined]
        return super().format(record)
