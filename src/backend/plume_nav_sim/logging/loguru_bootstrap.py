from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from loguru import logger as _logger  # type: ignore
except Exception:  # pragma: no cover
    _logger = None  # type: ignore


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging records to loguru."""

    def emit(
        self, record: logging.LogRecord
    ) -> None:  # pragma: no cover - thin wrapper
        if _logger is None:
            return
        try:
            level = _logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    *,
    level: str = "INFO",
    console: bool = True,
    file_path: Optional[str | Path] = None,
    rotation: Optional[str | int] = None,
    retention: Optional[str | int] = None,
    serialize: bool = False,
) -> None:
    """Configure loguru logging for ops logs only.

    - Adds console sink (stderr) and optional file sink
    - Bridges stdlib logging to loguru
    - Does not interact with data capture pipeline
    """
    if _logger is None:
        raise ImportError(
            "loguru is not installed. Install with 'pip install -e .[ops]'"
        )

    _logger.remove()
    lvl = level.upper()
    if console:
        _logger.add(
            sys.stderr, level=lvl, backtrace=False, diagnose=False, serialize=serialize
        )
    if file_path:
        _logger.add(
            str(file_path),
            level=lvl,
            rotation=rotation,
            retention=retention,
            backtrace=False,
            diagnose=False,
            serialize=serialize,
        )
    _bridge_stdlib(level=lvl)


def _bridge_stdlib(level: str = "INFO") -> None:
    """Route stdlib logging into loguru."""
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(getattr(logging, level, logging.INFO))
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


def get_logger():  # pragma: no cover - trivial accessor
    """Return the configured loguru logger instance.

    Raises
    ------
    ImportError
        If loguru is not installed/available.
    """
    if _logger is None:
        raise ImportError("loguru is not installed.")
    return _logger
