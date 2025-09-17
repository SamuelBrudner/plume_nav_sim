"""Minimal logging configuration bridge for plume_nav_sim utilities."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ComponentType(Enum):
    """Enumeration mirroring the legacy component categories."""

    ENVIRONMENT = "environment"
    PLUME_MODEL = "plume_model"
    RENDERING = "rendering"
    UTILS = "utils"


def get_logger(name: str | None = None, **kwargs: Any) -> logging.Logger:
    """Return a namespaced logger using the standard library backend."""
    logger_name = kwargs.get("logger_name") or name or "plume_nav_sim"
    return logging.getLogger(str(logger_name))


def configure_development_logging(
    *, level: int | str = logging.DEBUG, format: str = DEFAULT_FORMAT, **_: Any
) -> Dict[str, Any]:
    """Apply a simple development logging configuration."""
    logging.basicConfig(level=level, format=format)
    return {"level": level, "format": format}
