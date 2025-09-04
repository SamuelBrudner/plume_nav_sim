"""Debug utilities for Plume Navigation Simulation."""

from __future__ import annotations

import importlib.util

from plume_nav_sim.utils.logging_setup import get_logger

logger = get_logger(__name__)


def _detect_backend_availability() -> dict[str, bool]:
    """Detect available GUI backends and log their status."""
    availability = {"pyside6": False, "streamlit": False}

    try:
        if importlib.util.find_spec("PySide6") is not None:
            availability["pyside6"] = True
            logger.debug("PySide6 backend available for Qt-based debug GUI")
        else:  # pragma: no cover - no backend found
            logger.debug("PySide6 backend not available")
    except (ImportError, AttributeError, ModuleNotFoundError):  # pragma: no cover - defensive
        logger.debug("PySide6 backend not available")

    try:
        if importlib.util.find_spec("streamlit") is not None:
            availability["streamlit"] = True
            logger.debug("Streamlit backend available for web-based debug GUI")
        else:  # pragma: no cover - no backend found
            logger.debug("Streamlit backend not available")
    except (ImportError, AttributeError, ModuleNotFoundError):  # pragma: no cover - defensive
        logger.debug("Streamlit backend not available")

    return availability


_backend_availability = _detect_backend_availability()

try:  # pragma: no cover - import error path exercised via tests
    from plume_nav_sim.debug.gui import (
        DebugGUI,
        DebugSession,
        DebugConfig,
        plot_initial_state,
        launch_viewer,
    )
except ImportError as exc:  # pragma: no cover - fail fast
    raise ImportError("Debug GUI dependencies are required") from exc

try:  # pragma: no cover - import error path exercised via tests
    from plume_nav_sim.debug.cli import debug_group
except ImportError as exc:  # pragma: no cover - fail fast
    raise ImportError("Debug CLI dependencies are required") from exc

__availability__ = {
    "gui": True,
    "cli": True,
    "pyside6": _backend_availability["pyside6"],
    "streamlit": _backend_availability["streamlit"],
}

logger.info(
    "Debug module initialized",
    extra={
        "module_name": "plume_nav_sim.debug",
        "availability": __availability__,
        "backend_support": _backend_availability,
    },
)

__all__ = [
    "DebugGUI",
    "DebugSession",
    "DebugConfig",
    "plot_initial_state",
    "launch_viewer",
    "debug_group",
    "__availability__",
]
