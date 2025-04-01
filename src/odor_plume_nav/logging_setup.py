"""
Logging configuration module for odor plume navigation.

This module provides a consistent logging setup across the application,
using loguru for advanced logging capabilities.
"""

import sys
import os
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional, List, Union


# Default log format
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Module format with module name included
MODULE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<blue>module={extra[module]}</blue> - "
    "<level>{message}</level>"
)

# Log levels with corresponding colors for better visibility
LOG_LEVELS = {
    "TRACE": {"color": "<cyan>"},
    "DEBUG": {"color": "<blue>"},
    "INFO": {"color": "<green>"},
    "SUCCESS": {"color": "<green>"},
    "WARNING": {"color": "<yellow>"},
    "ERROR": {"color": "<red>"},
    "CRITICAL": {"color": "<red>"},
}


def setup_logger(
    sink: Union[str, Path, None] = None,
    level: str = "INFO",
    format: str = DEFAULT_FORMAT,
    rotation: Optional[str] = "10 MB",
    retention: Optional[str] = "1 week",
    enqueue: bool = True,
    backtrace: bool = True,
    diagnose: bool = True,
) -> None:
    """
    Configure the logger with the specified settings.
    
    Args:
        sink: Output path for log file, or None for console only
        level: Minimum log level to display
        format: Log message format
        rotation: When to rotate log files (e.g., "10 MB" or "1 day")
        retention: How long to keep log files
        enqueue: Whether to enqueue log messages (better for multiprocessing)
        backtrace: Whether to include a backtrace for exceptions
        diagnose: Whether to diagnose exceptions with better tracebacks
    """
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        format=format,
        level=level,
        backtrace=backtrace,
        diagnose=diagnose,
    )
    
    # Add file logger if sink is provided
    if sink:
        # Make sure directory exists
        if isinstance(sink, str):
            directory = os.path.dirname(sink)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        logger.add(
            str(sink),  # Ensure sink is a string
            format=format,
            level=level,
            rotation=rotation,
            retention=retention,
            enqueue=enqueue,
            backtrace=backtrace,
            diagnose=diagnose,
        )


def get_module_logger(name: str) -> logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Loguru logger instance
    """
    # Create a logger that has the module name as extra context
    return logger.bind(module=name)


# Default setup for console logging
setup_logger()
