"""Lean logging helpers for plume_nav_sim."""

from __future__ import annotations

import logging
import os
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from loguru import logger as _loguru_logger
except Exception:  # pragma: no cover - optional dependency
    _loguru_logger = None  # type: ignore

LOGGER_NAME_PREFIX = "plume_nav_sim"

DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DEVELOPMENT_LOG_FORMAT = (
    "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

REDACTION_PLACEHOLDER = "[REDACTED]"
MAX_MESSAGE_LENGTH = 1000

COLOR_CODES: Dict[str, str] = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "DEBUG": "\033[94m",
    "INFO": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[91m",
}

CONSOLE_COLOR_CODES = COLOR_CODES

SENSITIVE_REGEX_PATTERNS = [
    re.compile(
        r"\b(?:password|token|api_key|secret|auth|credential)\s*[=:]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Za-z0-9+/]{20,}={0,2}\b"),
    re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
]


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        return logging._nameToLevel.get(self.value, logging.INFO)

    @classmethod
    def from_string(cls, level_string: str) -> "LogLevel":
        return cls(level_string.upper())


class ComponentType(Enum):
    ENVIRONMENT = "ENVIRONMENT"
    PLUME_MODEL = "PLUME_MODEL"
    RENDERING = "RENDERING"
    ACTION_PROCESSOR = "ACTION_PROCESSOR"
    REWARD_CALCULATOR = "REWARD_CALCULATOR"
    STATE_MANAGER = "STATE_MANAGER"
    BOUNDARY_ENFORCER = "BOUNDARY_ENFORCER"
    EPISODE_MANAGER = "EPISODE_MANAGER"
    UTILS = "UTILS"


def detect_color_support() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = sys.stdout
    isatty = getattr(stream, "isatty", None)
    return bool(isatty() if callable(isatty) else False)


def sanitize_message(
    message: Any,
    patterns: Optional[Sequence[re.Pattern]] = None,
    max_length: int = MAX_MESSAGE_LENGTH,
) -> str:
    text = "" if message is None else str(message)
    active_patterns = patterns or SENSITIVE_REGEX_PATTERNS
    for pattern in active_patterns:
        text = pattern.sub(REDACTION_PLACEHOLDER, text)
    if max_length and len(text) > max_length:
        return f"{text[: max_length - 3]}..."
    return text


class SecurityFilter(logging.Filter):
    def __init__(
        self,
        patterns: Optional[Sequence[re.Pattern]] = None,
        placeholder: str = REDACTION_PLACEHOLDER,
    ) -> None:
        super().__init__()
        self._patterns = list(patterns or SENSITIVE_REGEX_PATTERNS)
        self._placeholder = placeholder

    def filter(self, record: logging.LogRecord) -> bool:
        message = sanitize_message(record.getMessage(), self._patterns)
        if self._placeholder:
            message = message.replace(REDACTION_PLACEHOLDER, self._placeholder)
        record.msg = message
        record.args = ()
        return True


SensitiveInfoFilter = SecurityFilter


class LogFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str = DEFAULT_LOG_FORMAT,
        datefmt: str = DEFAULT_DATE_FORMAT,
        *,
        enable_redaction: bool = True,
        redaction_patterns: Optional[Sequence[re.Pattern]] = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._enable_redaction = enable_redaction
        self._redaction_patterns = list(redaction_patterns or SENSITIVE_REGEX_PATTERNS)

    def format(self, record: logging.LogRecord) -> str:
        output = super().format(record)
        if self._enable_redaction:
            return sanitize_message(output, self._redaction_patterns)
        return output


class ConsoleFormatter(LogFormatter):
    def __init__(
        self,
        fmt: str = DEFAULT_LOG_FORMAT,
        datefmt: str = DEFAULT_DATE_FORMAT,
        *,
        enable_redaction: bool = True,
        redaction_patterns: Optional[Sequence[re.Pattern]] = None,
        use_color: Optional[bool] = None,
        color_codes: Mapping[str, str] = COLOR_CODES,
    ) -> None:
        self._use_color = detect_color_support() if use_color is None else use_color
        self._color_codes = color_codes
        super().__init__(
            fmt=fmt,
            datefmt=datefmt,
            enable_redaction=enable_redaction,
            redaction_patterns=redaction_patterns,
        )

    def format(self, record: logging.LogRecord) -> str:
        output = super().format(record)
        if not self._use_color:
            return output
        color = self._color_codes.get(record.levelname.upper(), "")
        reset = self._color_codes.get("RESET", "")
        return f"{color}{output}{reset}" if color else output


class PerformanceFormatter(LogFormatter):
    def __init__(
        self,
        *,
        enable_memory_tracking: bool = False,
        timing_thresholds: Optional[Mapping[str, float]] = None,
        fmt: str = DEFAULT_LOG_FORMAT,
        datefmt: str = DEFAULT_DATE_FORMAT,
        enable_redaction: bool = True,
        redaction_patterns: Optional[Sequence[re.Pattern]] = None,
    ) -> None:
        super().__init__(
            fmt=fmt,
            datefmt=datefmt,
            enable_redaction=enable_redaction,
            redaction_patterns=redaction_patterns,
        )
        self.enable_memory_tracking = enable_memory_tracking
        self.timing_thresholds = dict(timing_thresholds or {})

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "duration_ms"):
            record.duration_ms = 0.0
        return super().format(record)

    def format_timing(
        self,
        operation_name: str,
        duration_ms: float,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        suffix = ""
        if additional_data:
            suffix = " " + " ".join(
                f"{key}={value}" for key, value in additional_data.items()
            )
        return f"{operation_name}: {duration_ms:.3f}ms{suffix}"

    def add_baseline(self, operation_name: str, threshold_ms: float) -> None:
        self.timing_thresholds[operation_name] = threshold_ms


def _resolve_log_level(level: Any) -> int:
    if isinstance(level, LogLevel):
        return level.to_logging_level()
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        return logging._nameToLevel.get(level.upper(), logging.INFO)
    return logging.INFO


def _normalize_logger_name(name: str, component_type: ComponentType) -> str:
    if not name:
        return f"{LOGGER_NAME_PREFIX}.{component_type.value.lower()}"
    if name.startswith(LOGGER_NAME_PREFIX):
        return name
    if "." in name:
        return f"{LOGGER_NAME_PREFIX}.{name}"
    return f"{LOGGER_NAME_PREFIX}.{component_type.value.lower()}.{name}"


def configure_logging(
    *,
    log_level: Any = "INFO",
    enable_console_logging: bool = True,
    enable_file_logging: bool = False,
    log_file_path: Optional[str | Path] = None,
    enable_color_output: bool = True,
    log_format: Optional[str] = None,
    datefmt: str = DEFAULT_DATE_FORMAT,
    redaction_patterns: Optional[Sequence[re.Pattern]] = None,
    force: bool = True,
) -> Dict[str, Any]:
    level = _resolve_log_level(log_level)
    handlers = []

    if enable_console_logging:
        formatter_cls = (
            ConsoleFormatter
            if enable_color_output and detect_color_support()
            else LogFormatter
        )
        console_formatter = formatter_cls(
            fmt=log_format or DEFAULT_LOG_FORMAT,
            datefmt=datefmt,
            redaction_patterns=redaction_patterns,
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(SecurityFilter(patterns=redaction_patterns))
        handlers.append(console_handler)

    log_path = None
    if enable_file_logging:
        path = (
            Path(log_file_path) if log_file_path else Path(f"{LOGGER_NAME_PREFIX}.log")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            LogFormatter(
                fmt=log_format or DEFAULT_LOG_FORMAT,
                datefmt=datefmt,
                redaction_patterns=redaction_patterns,
            )
        )
        file_handler.addFilter(SecurityFilter(patterns=redaction_patterns))
        handlers.append(file_handler)
        log_path = str(path)

    if not handlers:
        handlers.append(logging.NullHandler())

    logging.basicConfig(level=level, handlers=handlers, force=force)

    return {
        "status": "configured",
        "log_level": logging.getLevelName(level),
        "console_output": enable_console_logging,
        "color_output": enable_color_output,
        "file_output": bool(log_path),
        "file_path": log_path,
    }


def configure_development_logging(
    enable_verbose_output: bool = True,
    enable_color_console: bool = True,
    log_to_file: bool = True,
    development_log_level: str = "DEBUG",
) -> Dict[str, Any]:
    result = configure_logging(
        log_level=development_log_level,
        enable_console_logging=True,
        enable_file_logging=log_to_file,
        enable_color_output=enable_color_console,
        log_format=DEVELOPMENT_LOG_FORMAT,
    )
    result.update(
        {
            "status": "success",
            "log_level": development_log_level,
            "verbose_output": enable_verbose_output,
            "color_console": enable_color_console,
            "file_logging": log_to_file,
            "performance_logging": False,
        }
    )
    return result


def get_logger(
    name: str,
    component_type: ComponentType = ComponentType.UTILS,
    enable_performance_tracking: bool = False,
    logger_config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    logger_name = _normalize_logger_name(name, component_type)
    logger = logging.getLogger(logger_name)
    if logger_config and "level" in logger_config:
        logger.setLevel(_resolve_log_level(logger_config["level"]))
    if enable_performance_tracking:
        setattr(logger, "performance_tracking", True)
    return logger


def get_component_logger(
    name: str,
    component_type: ComponentType = ComponentType.UTILS,
    enable_performance_tracking: bool = False,
    logger_config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    return get_logger(
        name,
        component_type=component_type,
        enable_performance_tracking=enable_performance_tracking,
        logger_config=logger_config,
    )


class InterceptHandler(logging.Handler):
    def emit(
        self, record: logging.LogRecord
    ) -> None:  # pragma: no cover - thin wrapper
        if _loguru_logger is None:
            return
        try:
            level = _loguru_logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _bridge_stdlib(level: str = "INFO") -> None:
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(getattr(logging, level, logging.INFO))
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


def setup_logging(
    *,
    level: str = "INFO",
    console: bool = True,
    file_path: Optional[str | Path] = None,
    rotation: Optional[str | int] = None,
    retention: Optional[str | int] = None,
    serialize: bool = False,
) -> None:
    if _loguru_logger is None:
        raise ImportError(
            "loguru is not installed. Install with 'pip install -e .[ops]'"
        )

    _loguru_logger.remove()
    lvl = level.upper()
    if console:
        _loguru_logger.add(
            sys.stderr, level=lvl, backtrace=False, diagnose=False, serialize=serialize
        )
    if file_path:
        _loguru_logger.add(
            str(file_path),
            level=lvl,
            rotation=rotation,
            retention=retention,
            backtrace=False,
            diagnose=False,
            serialize=serialize,
        )
    _bridge_stdlib(level=lvl)


def get_loguru_logger():
    if _loguru_logger is None:
        raise ImportError("loguru is not installed.")
    return _loguru_logger
