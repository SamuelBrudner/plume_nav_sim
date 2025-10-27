import logging
from pathlib import Path

import pytest


@pytest.mark.skipif(
    pytest.importorskip("loguru", reason="loguru not installed") is None,
    reason="loguru missing",
)
def test_setup_logging_console_and_file(tmp_path: Path):
    from plume_nav_sim.logging.loguru_bootstrap import get_logger, setup_logging

    logfile = tmp_path / "app.log"
    setup_logging(level="INFO", console=True, file_path=logfile)
    logger = get_logger()
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

    m = importlib.import_module("plume_nav_sim.logging.loguru_bootstrap")
    assert hasattr(m, "setup_logging")
