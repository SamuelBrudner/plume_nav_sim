import logging

import pytest


def test_get_component_logger_respects_level_override():
    from plume_nav_sim.utils.logging import ComponentType, get_component_logger

    # Create a logger with an explicit DEBUG level override
    logger = get_component_logger(
        "qa_level_override",
        ComponentType.UTILS,
        logger_level="DEBUG",
        enable_performance_tracking=False,
    )

    # ComponentLogger proxies level to its underlying base logger
    assert getattr(logger, "level", None) == logging.DEBUG


@pytest.mark.parametrize(
    "duration_ms,target_ms,expected_levels",
    [
        # Below target: logs at DEBUG
        (5.0, 10.0, [logging.DEBUG]),
        # Slightly above target (<=1.2x): throttled, no log
        (11.0, 10.0, []),
        # Moderately above target: WARNING
        (15.0, 10.0, [logging.WARNING]),
        # Far above target: ERROR
        (25.0, 10.0, [logging.ERROR]),
    ],
)
def test_log_performance_threshold_levels(
    duration_ms, target_ms, expected_levels, caplog
):
    from plume_nav_sim.utils.logging import log_performance

    # Use an isolated logger with a memory handler for capture
    test_logger = logging.getLogger("plume_nav_sim.qa.test_perf")
    test_logger.handlers.clear()
    test_logger.setLevel(logging.DEBUG)

    stream = logging.StreamHandler()  # simple handler suitable for caplog
    stream.setLevel(logging.DEBUG)
    test_logger.addHandler(stream)

    caplog.clear()
    caplog.set_level(logging.DEBUG, logger=test_logger.name)

    # Exercise log_performance
    log_performance(
        test_logger,
        operation_name="qa_operation",
        duration_ms=duration_ms,
        additional_metrics={"unit": "ms"},
        compare_to_baseline=False,
        target_ms=target_ms,
    )

    # Collect observed levels for this logger only
    observed = [rec.levelno for rec in caplog.records if rec.name == test_logger.name]
    assert observed == expected_levels
