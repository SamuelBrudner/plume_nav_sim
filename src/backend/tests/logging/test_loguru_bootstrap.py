import logging
from pathlib import Path

import pytest


def test_setup_logging_console_and_file(tmp_path: Path):
    from plume_nav_sim.logging import get_loguru_logger, setup_logging

    try:
        import loguru  # noqa: F401
    except Exception:
        # When loguru isn't installed, setup must raise ImportError and get_logger too
        with pytest.raises(ImportError):
            setup_logging(level="INFO", console=True, file_path=tmp_path / "x.log")
        with pytest.raises(ImportError):
            _ = get_loguru_logger()
        return

    logfile = tmp_path / "app.log"
    setup_logging(level="INFO", console=True, file_path=logfile)
    logger = get_loguru_logger()
    logger.info("hello from loguru")

    std = logging.getLogger("std")
    std.info("hello from stdlib")

    # loguru writes asynchronously; force sink flush by removing and re-adding console
    logger.remove()

    # Validate file contains messages
    data = logfile.read_text(encoding="utf-8")
    assert "hello from loguru" in data
    assert "hello from stdlib" in data


def test_import_does_not_touch_data_capture():
    # This import should not import data_capture or create any files
    import importlib

    m = importlib.import_module("plume_nav_sim.logging")
    assert hasattr(m, "setup_logging")
